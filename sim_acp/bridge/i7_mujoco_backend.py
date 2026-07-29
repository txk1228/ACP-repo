"""i7 真机 MJCF 后端：右臂 IK + 重力补偿 PD + mesh 接触力。

高保真：末端碰撞为官方夹爪/法兰 mesh；工具点为不可见 site（无 tip 球）。

防晃要点：
- IK 只更新 q_des，禁止每步硬改仿真 qpos（否则速度不一致 → 狂甩）
- 直接对工具点 site 做 IK（脚本），避免 tip↔法兰反推振荡
- 接触力 / 笛卡尔目标低通；q_des 速率限制
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from sim_acp.bridge.backend_base import RobotState
from sim_acp.bridge.i7_scene import compile_i7_acp_model, ensure_i7_acp_scene

RIGHT_JOINTS = [
    "Shoulder_Pitch_Right",
    "Shoulder_Roll_Right",
    "Elbow_Yaw_Right",
    "Elbow_Pitch_Right",
    "Wrist_Yaw_Right",
    "Wrist_Roll_Right",
    "Wrist_Pitch_Right",
]

HOME_Q = np.array(
    [
        0.0,
        0.0,
        -0.6,
        0.0,
        -1.2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.6,
        0.0,
        -1.2,
        0.0,
        0.0,
        0.0,
    ],
    dtype=float,
)

_N_ROBOT_Q = 15

_EE_COLLISION_GEOMS = (
    "acp_tip_ball",
    "Gripper_Right_visual",
    "falan_Right_visual",
    "Wrist_Pitch_Right_visual",
)


def resolve_i7_scene() -> Path:
    try:
        return ensure_i7_acp_scene()
    except FileNotFoundError:
        root = Path(
            os.environ.get(
                "ACP_I7_MODEL_ROOT",
                "/home/zj/robot-control-v1.5/model_new",
            )
        )
        scene = root / "mjcf" / "scene.xml"
        if not scene.is_file():
            raise FileNotFoundError(
                f"找不到 i7 场景: {scene}\n请设置 ACP_I7_MODEL_ROOT"
            )
        return scene


def _clamp_q(model, q: np.ndarray) -> np.ndarray:
    out = q.copy()
    for j in range(model.njnt):
        if not model.jnt_limited[j]:
            continue
        adr = int(model.jnt_qposadr[j])
        lo, hi = model.jnt_range[j]
        out[adr] = float(np.clip(out[adr], lo, hi))
    return out


def contact_force_on_geoms(model, data, geom_ids: list[int]) -> np.ndarray:
    """环境作用在一组 EE geom 上的合力（世界系）。"""
    import mujoco

    force = np.zeros(3, dtype=float)
    id_set = {int(g) for g in geom_ids if int(g) >= 0}
    if not id_set:
        return force
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        ee1 = g1 in id_set
        ee2 = g2 in id_set
        if ee1 == ee2:
            continue
        cf = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, i, cf)
        n = c.frame[0:3]
        t1 = c.frame[3:6]
        t2 = c.frame[6:9]
        f_world = n * cf[0] + t1 * cf[1] + t2 * cf[2]
        if ee1:
            force += f_world
        else:
            force -= f_world
    return -force


class I7MujocoBackend:
    """右臂笛卡尔跟踪；mesh 接触力；工具点 = acp_ee_site。"""

    def __init__(
        self,
        model_path: str | Path | None = None,
        render: bool = False,
        cam_width: int = 224,
        cam_height: int = 224,
        kp: float = 180.0,
        kd: float = 50.0,
        tip_clamp: bool = False,
        max_dq: float = 0.03,
        force_lpf: float = 0.06,
        target_lpf: float = 0.28,
    ):
        import mujoco

        self._mj = mujoco
        if model_path is None:
            self.model, path = compile_i7_acp_model()
        else:
            path = Path(model_path)
            self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.kp = float(kp)
        self.kd = float(kd)
        self._max_dq = float(max_dq)
        self._force_lpf = float(np.clip(force_lpf, 0.01, 1.0))
        self._target_lpf = float(np.clip(target_lpf, 0.01, 1.0))
        self._render = bool(render)
        self._viewer = None
        self._viewer_cm = None
        self._force = np.zeros(3, dtype=float)
        self._contact_f = np.zeros(3, dtype=float)
        self._contact_f_raw = np.zeros(3, dtype=float)
        self._contact_force_clip = 35.0
        self._tip_clamp = bool(tip_clamp)
        self._max_tip_penetration = 0.008 if self._tip_clamp else 1.0
        self._table_gid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "acp_table_top"
        )
        self._obj_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "acp_obj"
        )

        self._ee_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "falan_Right"
        )
        if self._ee_body < 0:
            raise RuntimeError("body falan_Right not found")

        self._gripper_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "Gripper_Right"
        )
        self._tool_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "acp_ee_site"
        )
        self._tip_geom = -1

        self._ee_geom_ids: list[int] = []
        for name in _EE_COLLISION_GEOMS:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._ee_geom_ids.append(int(gid))

        self._qadr: list[int] = []
        self._dadr: list[int] = []
        for name in RIGHT_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"joint {name} not found")
            self._qadr.append(int(self.model.jnt_qposadr[jid]))
            self._dadr.append(int(self.model.jnt_dofadr[jid]))

        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "acp_wrist_cam"
        )
        if cam_id < 0:
            cam_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "acp_cam"
            )
        if cam_id < 0:
            cam_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_manual"
            )
        self._cam_id = cam_id if cam_id >= 0 else -1
        cam_name = (
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self._cam_id)
            if self._cam_id >= 0
            else "none"
        )
        print(
            f"[i7] scene={path.name} rgb_cam={cam_name} cam_id={self._cam_id} "
            f"ee_mesh={len(self._ee_geom_ids)} site={self._tool_site} tip+mesh"
        )
        self._cam_w = int(cam_width)
        self._cam_h = int(cam_height)

        # 官方 MJCF 腕部力矩极小（~±26Nm），PD 跟不住就会靠硬传送甩臂。
        # 联调仿真放宽力矩；真机后端另接。
        self.model.actuator_ctrlrange[:, 0] = -800.0
        self.model.actuator_ctrlrange[:, 1] = 800.0

        self.data.qpos[:_N_ROBOT_Q] = HOME_Q
        self.data.qvel[:] = 0.0
        self._q_des = self.data.qpos.copy()
        self._q_des[:_N_ROBOT_Q] = HOME_Q
        self._q_des = _clamp_q(self.model, self._q_des)
        self.data.qpos[: self.model.nq] = self._q_des
        self._lock_base()
        mujoco.mj_forward(self.model, self.data)
        for _ in range(120):
            self._apply_pd()
            mujoco.mj_step(self.model, self.data)
            self._lock_base()
        self._ee_target = self.tip_pos()
        self._contact_f = contact_force_on_geoms(
            self.model, self.data, self._ee_geom_ids
        )

        self._renderer = None
        self._display_renderer = None
        self._display_wh: tuple[int, int] | None = None
        self._overview_renderer = None
        self._overview_wh: tuple[int, int] | None = None
        self._step_count = 0
        # viewer.sync 节流：每 N 仿真步同步一次（1=每步；越大越快）
        self._viewer_sync_every = 5
        if self._render:
            self._open_viewer()

    def set_viewer_sync_every(self, n: int) -> None:
        self._viewer_sync_every = max(1, int(n))

    def _make_offscreen_renderer(self, height: int, width: int):
        """离屏 RGB：有 passive viewer 时用 EGL，避免与 GLFW 抢上下文。"""
        if self._render and self._viewer is not None:
            prev_gl = os.environ.get("MUJOCO_GL")
            os.environ["MUJOCO_GL"] = "egl"
            try:
                return self._mj.Renderer(
                    self.model, height=int(height), width=int(width)
                )
            finally:
                if prev_gl is None:
                    os.environ.pop("MUJOCO_GL", None)
                else:
                    os.environ["MUJOCO_GL"] = prev_gl
        return self._mj.Renderer(self.model, height=int(height), width=int(width))

    def _ensure_renderer(self) -> None:
        """离屏 RGB 才建 Renderer，避免与 interactive viewer 抢 GLFW 导致退出段错误。"""
        if self._renderer is not None:
            return
        self._renderer = self._make_offscreen_renderer(self._cam_h, self._cam_w)

    def _lock_base(self) -> None:
        self.data.qpos[0] = 0.0
        self.data.qvel[0] = 0.0

    def _open_viewer(self) -> None:
        try:
            import mujoco.viewer

            self._viewer_cm = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )
            self._viewer = self._viewer_cm.__enter__()
        except Exception as exc:
            print(f"[warn] viewer 打开失败: {exc}")
            self._viewer = None
            self._viewer_cm = None
            self._render = False

    def viewer_running(self) -> bool:
        if self._viewer is None:
            return False
        try:
            return bool(self._viewer.is_running())
        except Exception:
            return False

    def table_top_z(self) -> float:
        if self._table_gid < 0:
            return -1e9
        return float(
            self.data.geom_xpos[self._table_gid, 2]
            + self.model.geom_size[self._table_gid, 2]
        )

    def tip_radius(self) -> float:
        for name in ("Gripper_Right_visual",):
            gid = self._mj.mj_name2id(
                self.model, self._mj.mjtObj.mjOBJ_GEOM, name
            )
            if gid >= 0:
                return float(
                    max(self.model.geom_size[gid, 0], self.model.geom_size[gid, 1])
                )
        return 0.03

    def tip_pos(self) -> np.ndarray:
        self._mj.mj_forward(self.model, self.data)
        if self._tool_site >= 0:
            return self.data.site_xpos[self._tool_site].copy()
        if self._gripper_body >= 0:
            return self.data.xpos[self._gripper_body].copy()
        return self.data.xpos[self._ee_body].copy()

    def cube_pos(self) -> np.ndarray:
        if self._obj_body < 0:
            return np.zeros(3)
        self._mj.mj_forward(self.model, self.data)
        return self.data.xpos[self._obj_body].copy()

    def cube_tilt_rad(self) -> float:
        if self._obj_body < 0:
            return 0.0
        self._mj.mj_forward(self.model, self.data)
        R = self.data.xmat[self._obj_body].reshape(3, 3)
        up_z = float(np.clip(R[2, 2], -1.0, 1.0))
        return float(np.arccos(up_z))

    def reset_episode(
        self,
        cube_xy: tuple[float, float] | None = None,
    ) -> None:
        """重置右臂 home + 方块位姿（可选 xy 扰动），供多 episode 采集。"""
        from sim_acp.bridge.i7_scene import cube_spawn_xyz

        mujoco = self._mj
        self.data.qpos[:_N_ROBOT_Q] = HOME_Q
        self.data.qvel[:] = 0.0
        self._q_des = self.data.qpos.copy()
        self._q_des[:_N_ROBOT_Q] = HOME_Q
        self._q_des = _clamp_q(self.model, self._q_des)
        self.data.qpos[: self.model.nq] = self._q_des

        if self._obj_body >= 0:
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "acp_obj_free"
            )
            if jid >= 0:
                adr = int(self.model.jnt_qposadr[jid])
                cx, cy, cz = cube_spawn_xyz()
                if cube_xy is not None:
                    cx, cy = float(cube_xy[0]), float(cube_xy[1])
                self.data.qpos[adr : adr + 7] = [
                    cx,
                    cy,
                    cz,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                dadr = int(self.model.jnt_dofadr[jid])
                self.data.qvel[dadr : dadr + 6] = 0.0

        self._force[:] = 0.0
        self._contact_f[:] = 0.0
        self._contact_f_raw[:] = 0.0
        self._lock_base()
        mujoco.mj_forward(self.model, self.data)
        for _ in range(80):
            self._apply_pd()
            mujoco.mj_step(self.model, self.data)
            self._lock_base()
        self._ee_target = self.tip_pos()
        self._contact_f = contact_force_on_geoms(
            self.model, self.data, self._ee_geom_ids
        )

    def set_tip_clamp(self, enabled: bool, max_pen: float = 0.008) -> None:
        self._tip_clamp = bool(enabled)
        self._max_tip_penetration = float(max_pen) if enabled else 1.0

    def falan_for_tip_target(self, tip_xyz: np.ndarray) -> np.ndarray:
        """兼容旧接口：由 tip 反推法兰（策略路径仍可能用）。"""
        self._mj.mj_forward(self.model, self.data)
        falan = self.data.xpos[self._ee_body]
        tip = self.tip_pos()
        rel = tip - falan
        return np.asarray(tip_xyz, dtype=float).reshape(3) - rel

    def suggest_press_target(self, lateral: np.ndarray | None = None) -> np.ndarray:
        self._mj.mj_forward(self.model, self.data)
        p = self.data.xpos[self._ee_body].copy()
        tip = self.tip_pos()
        rel_z = float(tip[2] - p[2])
        want_tip_z = self.table_top_z() + self.tip_radius() - 0.003
        p[2] = want_tip_z - rel_z
        if lateral is not None:
            lat = np.asarray(lateral, dtype=float).reshape(3)
            p[0] += float(lat[0])
            p[1] += float(lat[1])
        return p

    def _lift_target_if_tip_penetrates(self, target: np.ndarray) -> np.ndarray:
        """仅 tip_clamp 时抬高法兰目标；探测不永久改写仿真状态。"""
        if not self._tip_clamp or self._table_gid < 0:
            return np.asarray(target, dtype=float).reshape(3).copy()
        mujoco = self._mj
        t = np.asarray(target, dtype=float).reshape(3).copy()
        r = self.tip_radius()
        max_pen = float(self._max_tip_penetration)
        q_save = self.data.qpos.copy()
        v_save = self.data.qvel.copy()
        try:
            for _ in range(6):
                self._ik_to_point(t, use_site=False, rate_limit=False)
                self.data.qpos[: self.model.nq] = self._q_des
                self._lock_base()
                mujoco.mj_forward(self.model, self.data)
                tip_z = float(self.tip_pos()[2])
                top = self.table_top_z()
                bottom = tip_z - r
                if bottom >= top - max_pen:
                    return t
                t[2] += (top - max_pen) - bottom
        finally:
            self.data.qpos[: self.model.nq] = q_save
            self.data.qvel[:] = v_save
            self._lock_base()
            mujoco.mj_forward(self.model, self.data)
        return t

    def _ik_to_point(
        self,
        target_pos: np.ndarray,
        *,
        use_site: bool,
        max_iters: int = 40,
        rate_limit: bool = True,
    ) -> None:
        """位置 IK → 只写 _q_des；迭代临时改 qpos，结束必须还原。"""
        mujoco = self._mj
        target = np.asarray(target_pos, dtype=float).reshape(3)
        q_save = self.data.qpos.copy()
        v_save = self.data.qvel.copy()
        q = q_save.copy()
        for adr in self._qadr:
            q[adr] = float(self._q_des[adr])
        q[0] = 0.0

        def _point() -> np.ndarray:
            if use_site and self._tool_site >= 0:
                return self.data.site_xpos[self._tool_site]
            return self.data.xpos[self._ee_body]

        def _jac() -> np.ndarray:
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            if use_site and self._tool_site >= 0:
                mujoco.mj_jacSite(
                    self.model, self.data, jacp, jacr, self._tool_site
                )
            else:
                mujoco.mj_jacBody(
                    self.model, self.data, jacp, jacr, self._ee_body
                )
            return jacp[:, self._dadr]

        try:
            for _ in range(max_iters):
                self.data.qpos[: self.model.nq] = q
                self._lock_base()
                mujoco.mj_forward(self.model, self.data)
                err = target - _point()
                if float(np.linalg.norm(err)) < 1.2e-3:
                    break
                J = _jac()
                lam = 4e-2
                dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
                dq = np.clip(dq, -0.05, 0.05)
                for i, adr in enumerate(self._qadr):
                    q[adr] += float(dq[i])
                q = _clamp_q(self.model, q)
                q[0] = 0.0
        finally:
            self.data.qpos[: self.model.nq] = q_save
            self.data.qvel[:] = v_save
            self._lock_base()
            mujoco.mj_forward(self.model, self.data)

        if not rate_limit:
            q[0] = 0.0
            self._q_des = _clamp_q(self.model, q)
            return

        if use_site:
            cart_err = float(np.linalg.norm(target - self.tip_pos()))
        else:
            cart_err = float(
                np.linalg.norm(target - self.data.xpos[self._ee_body])
            )
        # 远则加快追赶，近则限速防抖
        max_dq = self._max_dq * (2.5 if cart_err > 0.06 else 1.0)
        q_cmd = self._q_des.copy()
        for adr in self._qadr:
            d = float(q[adr] - q_cmd[adr])
            q_cmd[adr] += float(np.clip(d, -max_dq, max_dq))
        q_cmd[0] = 0.0
        self._q_des = _clamp_q(self.model, q_cmd)

    def _apply_pd(self) -> None:
        mujoco = self._mj
        self._lock_base()
        mujoco.mj_forward(self.model, self.data)
        q = self.data.qpos
        qd = self.data.qvel
        kd = self.kd
        if float(np.linalg.norm(self._contact_f)) > 5.0:
            kd = self.kd * 2.0
        tau = np.zeros(self.model.nv, dtype=float)
        tau[:_N_ROBOT_Q] = (
            self.data.qfrc_bias[:_N_ROBOT_Q]
            + self.kp * (self._q_des[:_N_ROBOT_Q] - q[:_N_ROBOT_Q])
            - kd * qd[:_N_ROBOT_Q]
        )
        tau[0] = 0.0
        for i in range(self.model.nu):
            lo, hi = self.model.actuator_ctrlrange[i]
            self.data.ctrl[i] = float(np.clip(tau[i], lo, hi))
        self.data.xfrc_applied[:] = 0.0
        self.data.xfrc_applied[self._ee_body, :3] = self._force

    def read_state(self) -> RobotState:
        mujoco = self._mj
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.xpos[self._ee_body].copy()
        quat_wxyz = self.data.xquat[self._ee_body].copy()
        pose = np.array(
            [
                pos[0],
                pos[1],
                pos[2],
                quat_wxyz[1],
                quat_wxyz[2],
                quat_wxyz[3],
                quat_wxyz[0],
            ],
            dtype=float,
        )
        wrench = np.zeros(6, dtype=float)
        clip = float(self._contact_force_clip)
        wrench[:3] = np.clip(self._contact_f, -clip, clip) + self._force
        q = np.array([self.data.qpos[a] for a in self._qadr], dtype=float)
        return RobotState(
            pose_xyzquat=pose,
            wrench_W=wrench,
            q=q,
            timestamp_ns=time.time_ns(),
        )

    def write_ee_pose(self, pose_xyzquat: np.ndarray, timestamp_ns: int) -> None:
        """跟踪法兰位置（Flip Spec / 策略 x_virt）。"""
        del timestamp_ns
        p = np.asarray(pose_xyzquat, dtype=float).reshape(7)
        raw = self._lift_target_if_tip_penetrates(p[:3])
        a = self._target_lpf
        self._ee_target = (1.0 - a) * self._ee_target + a * raw
        self._ik_to_point(self._ee_target, use_site=False)

    def write_tip_pos(self, tip_xyz: np.ndarray, timestamp_ns: int = 0) -> None:
        """直接跟踪工具点 site（脚本翻方块，避免 tip↔法兰反推振荡）。"""
        del timestamp_ns
        raw = np.asarray(tip_xyz, dtype=float).reshape(3).copy()
        a = self._target_lpf
        self._ee_target = (1.0 - a) * self._ee_target + a * raw
        self._ik_to_point(self._ee_target, use_site=True)

    def inject_force_W(self, force_xyz: np.ndarray) -> None:
        self._force = np.asarray(force_xyz, dtype=float).reshape(3).copy()

    def step(self, realtime: bool | None = None) -> bool:
        """推进一仿真步。

        realtime: True 时 sleep(timestep)；默认 False（演示/策略不卡实时）。
        返回 False 仅表示 viewer 已关闭；sync 偶发失败不中断仿真。
        """
        if realtime is None:
            realtime = False
        if self._render and self._viewer is not None and not self.viewer_running():
            return False
        # 轻度软伺服：每步最多跟几毫弧度；接触时更小步长+更强阻尼，抑抖
        in_contact = float(np.linalg.norm(self._contact_f)) > 3.0
        dq_slew = 0.0012 if in_contact else 0.0025
        vel_damp = 0.94 if in_contact else 0.98
        for adr, did in zip(self._qadr, self._dadr):
            d = float(self._q_des[adr] - self.data.qpos[adr])
            self.data.qpos[adr] += float(np.clip(d, -dq_slew, dq_slew))
            self.data.qvel[did] *= vel_damp
        self._apply_pd()
        self._mj.mj_step(self.model, self.data)
        self._lock_base()
        raw_f = contact_force_on_geoms(
            self.model, self.data, self._ee_geom_ids
        )
        # 接触力尖峰截断后再 LPF，避免 mesh 撞击脉冲灌进柔顺环
        raw_f = np.clip(raw_f, -self._contact_force_clip, self._contact_force_clip)
        self._contact_f_raw = raw_f
        a = self._force_lpf
        self._contact_f = (1.0 - a) * self._contact_f + a * raw_f
        self._step_count += 1
        if (
            self._viewer is not None
            and self.viewer_running()
            and (self._step_count % self._viewer_sync_every == 0)
        ):
            try:
                self._viewer.sync()
            except Exception:
                # 偶发 GLFW sync 失败：若窗口仍在则继续，不中断本轮
                if not self.viewer_running():
                    return False
        if realtime:
            time.sleep(float(self.model.opt.timestep))
        return True

    def render_rgb(self, width: int | None = None, height: int | None = None) -> np.ndarray:
        w = int(width) if width is not None else self._cam_w
        h = int(height) if height is not None else self._cam_h
        if w == self._cam_w and h == self._cam_h:
            self._ensure_renderer()
            renderer = self._renderer
        else:
            if self._display_renderer is None or self._display_wh != (w, h):
                if self._display_renderer is not None:
                    try:
                        close_fn = getattr(self._display_renderer, "close", None)
                        if callable(close_fn):
                            close_fn()
                    except Exception:
                        pass
                self._display_renderer = self._make_offscreen_renderer(h, w)
                self._display_wh = (w, h)
            renderer = self._display_renderer
        assert renderer is not None
        self._mj.mj_forward(self.model, self.data)
        if self._cam_id >= 0:
            renderer.update_scene(self.data, camera=self._cam_id)
        else:
            renderer.update_scene(self.data)
        return renderer.render().copy()

    def render_overview_rgb(
        self,
        width: int = 720,
        height: int = 480,
        distance: float = 1.55,
        azimuth: float = 135.0,
        elevation: float = -22.0,
    ) -> np.ndarray:
        """外置自由机位 RGB（录屏用，不改 MJCF）。"""
        w, h = int(width), int(height)
        if self._overview_renderer is None or self._overview_wh != (w, h):
            if self._overview_renderer is not None:
                try:
                    close_fn = getattr(self._overview_renderer, "close", None)
                    if callable(close_fn):
                        close_fn()
                except Exception:
                    pass
            self._overview_renderer = self._make_offscreen_renderer(h, w)
            self._overview_wh = (w, h)

        self._mj.mj_forward(self.model, self.data)
        tip = self.tip_pos()
        cube = self.cube_pos()
        look = 0.55 * tip + 0.45 * cube
        look[2] = max(float(look[2]), float(self.table_top_z()) + 0.05)

        cam = self._mj.MjvCamera()
        self._mj.mjv_defaultFreeCamera(self.model, cam)
        cam.lookat[:] = look
        cam.distance = float(distance)
        cam.azimuth = float(azimuth)
        cam.elevation = float(elevation)
        self._overview_renderer.update_scene(self.data, camera=cam)
        return self._overview_renderer.render().copy()

    def close(self) -> None:
        # 顺序：先关离屏 Renderer，再关 passive viewer。
        # 两者共用 GLFW 时乱序销毁会 SIGSEGV（MuJoCo 已知问题）。
        for attr in ("_overview_renderer", "_display_renderer", "_renderer"):
            renderer = getattr(self, attr, None)
            setattr(self, attr, None)
            if renderer is not None:
                try:
                    close_fn = getattr(renderer, "close", None)
                    if callable(close_fn):
                        close_fn()
                except Exception:
                    pass

        viewer = self._viewer
        cm = self._viewer_cm
        self._viewer = None
        self._viewer_cm = None
        self._render = False
        try:
            if viewer is not None:
                try:
                    if getattr(viewer, "is_running", lambda: False)():
                        viewer.close()
                except Exception:
                    pass
            if cm is not None:
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    pass
        except Exception:
            pass
        # 给 GLFW 后台线程一点时间退出，减少解释器 atexit 二次析构
        time.sleep(0.05)