"""方案 B：纯 Python MuJoCo 后端（仅 ACP 仓）。"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from sim_acp.bridge.backend_base import RobotState

_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "ee_plane.xml"


class MujocoSimBackend:
    def __init__(
        self,
        model_path: str | Path | None = None,
        render: bool = False,
        camera_name: str = "wrist_cam",
        cam_width: int = 224,
        cam_height: int = 224,
    ):
        import mujoco

        path = Path(model_path) if model_path else _ASSETS
        if not path.is_file():
            raise FileNotFoundError(path)
        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self._ee_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_z"
        )
        self._site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if self._cam_id < 0:
            raise RuntimeError(f"camera '{camera_name}' not found in MJCF")
        self._cam_w = int(cam_width)
        self._cam_h = int(cam_height)
        self._target = np.array([0.4, 0.0, 0.35], dtype=float)
        self._force = np.zeros(3, dtype=float)
        self._render = bool(render)
        self._viewer = None
        self._viewer_cm = None
        self._renderer = None

        home = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home)
            # ctrl 只对应 3 个 slide
            self.data.ctrl[:3] = self._target
        else:
            self.data.qpos[:3] = self._target
            self.data.ctrl[:3] = self._target
        mujoco.mj_forward(self.model, self.data)

        # offscreen renderer（策略 RGB；与 interactive viewer 独立）
        self._renderer = mujoco.Renderer(
            self.model, height=self._cam_h, width=self._cam_w
        )

        if self._render:
            self._open_viewer()

    def _open_viewer(self) -> None:
        try:
            import mujoco.viewer

            self._viewer_cm = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )
            self._viewer = self._viewer_cm.__enter__()
        except Exception as exc:
            print(f"[warn] viewer 打开失败，改为无头: {exc}")
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

    def render_rgb(self) -> np.ndarray:
        """从固定相机渲一帧 RGB，uint8 HWC。"""
        assert self._renderer is not None
        self._mj.mj_forward(self.model, self.data)
        self._renderer.update_scene(self.data, camera=self._cam_id)
        return self._renderer.render().copy()

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
        wrench[:3] = self._force
        q = self.data.qpos[:3].copy()  # 仅臂滑动关节
        return RobotState(
            pose_xyzquat=pose,
            wrench_W=wrench,
            q=q,
            timestamp_ns=time.time_ns(),
        )

    def write_ee_pose(self, pose_xyzquat: np.ndarray, timestamp_ns: int) -> None:
        del timestamp_ns
        p = np.asarray(pose_xyzquat, dtype=float).reshape(7)
        self._target = p[:3].copy()
        n = min(3, self.model.nu)
        self.data.ctrl[:n] = self._target[:n]

    def inject_force_W(self, force_xyz: np.ndarray) -> None:
        self._force = np.asarray(force_xyz, dtype=float).reshape(3).copy()
        self.data.xfrc_applied[self._ee_body, :3] = self._force
        self.data.xfrc_applied[self._ee_body, 3:] = 0.0

    def step(self, realtime: bool | None = None) -> bool:
        if realtime is None:
            realtime = self._render

        if self._render and self._viewer is not None and not self.viewer_running():
            return False

        n = min(3, self.model.nu)
        self.data.ctrl[:n] = self._target[:n]
        self.data.xfrc_applied[self._ee_body, :3] = self._force
        self._mj.mj_step(self.model, self.data)

        if self._viewer is not None and self.viewer_running():
            try:
                self._viewer.sync()
            except Exception:
                return False

        if realtime:
            time.sleep(float(self.model.opt.timestep))
        return True

    def close(self) -> None:
        # 先关 viewer，避免与 Renderer/GLFW 退出顺序冲突导致段错误
        viewer = self._viewer
        cm = self._viewer_cm
        self._viewer = None
        self._viewer_cm = None
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass
        elif viewer is not None:
            try:
                if getattr(viewer, "is_running", lambda: False)():
                    viewer.close()
            except Exception:
                pass
        self._renderer = None
