"""查看 robot-control 已有 i7 URDF/MJCF 模型（不复制 mesh 进 ACP 仓）。

为何会「掉下去」：
  MJCF 执行器是 <motor>（力矩），默认 ctrl=0 → 等于没通电，重力下双臂塌掉。
  另外 Shoulder_Base 是竖直滑台，力矩上限 ±200 扛不住整机重力，必须锁住。

本脚本：锁住 base 滑台 + 重力补偿(qfrc_bias) + PD 托住 home 姿态。

用法：
  python -m sim_acp.scripts.view_i7_urdf
  python -m sim_acp.scripts.view_i7_urdf --wave
  python -m sim_acp.scripts.view_i7_urdf --no-gravity
  python -m sim_acp.scripts.view_i7_urdf --headless
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

HOME_Q = np.array(
    [
        0.0,  # Shoulder_Base (locked)
        0.0,
        -0.6,
        0.0,
        -1.2,
        0.0,
        0.0,
        0.0,  # right 7
        0.0,
        0.6,
        0.0,
        -1.2,
        0.0,
        0.0,
        0.0,  # left 7
    ],
    dtype=float,
)


def resolve_scene():
    from sim_acp.bridge.i7_scene import compile_i7_acp_model

    return compile_i7_acp_model()


def clamp_q(model, q: np.ndarray) -> np.ndarray:
    out = q.copy()
    for j in range(model.njnt):
        if not model.jnt_limited[j]:
            continue
        adr = int(model.jnt_qposadr[j])
        lo, hi = model.jnt_range[j]
        out[adr] = float(np.clip(out[adr], lo, hi))
    return out


def lock_base(data) -> None:
    """Shoulder_Base 竖直滑台力矩不够托整机，查看时钉死在 0。"""
    data.qpos[0] = 0.0
    data.qvel[0] = 0.0


def apply_hold(model, data, q_des: np.ndarray, kp: float, kd: float) -> None:
    """重力补偿 + PD → motor ctrl（力矩）。只控机器人关节，不管 freejoint。"""
    import mujoco

    lock_base(data)
    mujoco.mj_forward(model, data)
    n = min(model.nu, 15)
    tau = np.zeros(model.nv, dtype=float)
    tau[:n] = (
        data.qfrc_bias[:n]
        + kp * (q_des[:n] - data.qpos[:n])
        - kd * data.qvel[:n]
    )
    tau[0] = 0.0  # base 不出力，靠 qpos 钉死
    for i in range(model.nu):
        lo, hi = model.actuator_ctrlrange[i]
        data.ctrl[i] = float(np.clip(tau[i], lo, hi))


def main() -> int:
    parser = argparse.ArgumentParser(description="View i7 URDF/MJCF in MuJoCo")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--wave", action="store_true", help="右臂在 home 附近缓慢摆动")
    parser.add_argument("--no-gravity", action="store_true", help="关闭重力")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--kp", type=float, default=250.0)
    parser.add_argument("--kd", type=float, default=25.0)
    args = parser.parse_args()

    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("请先: pip install mujoco")
        return 1

    model, scene = resolve_scene()
    print(f"loading {scene}")
    data = mujoco.MjData(model)

    if args.no_gravity:
        model.opt.gravity[:] = 0.0
        print("  gravity OFF")
    else:
        print("  gravity ON：锁 base + 重力补偿 + PD（修复「双臂掉下去」）")

    # 保留桌上 freejoint 初值
    data.qpos[: len(HOME_Q)] = HOME_Q
    q_home = clamp_q(model, data.qpos.copy())
    data.qpos[: model.nq] = q_home
    data.qvel[:] = 0.0
    lock_base(data)
    mujoco.mj_forward(model, data)
    print(f"  nq={model.nq} nu={model.nu}")
    print(f"  home q[:8] = {q_home[:8].round(3)}")

    right_pitch = int(
        model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Shoulder_Pitch_Right")
        ]
    )
    right_elbow = int(
        model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Elbow_Pitch_Right")
        ]
    )
    falan = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "falan_Right")

    def step_once(t: float) -> None:
        q_des = q_home.copy()
        if args.wave:
            q_des[right_pitch] = q_home[right_pitch] + 0.25 * math.sin(0.7 * t)
            q_des[right_elbow] = q_home[right_elbow] + 0.20 * math.sin(0.7 * t + 0.5)
            q_des = clamp_q(model, q_des)
        if args.no_gravity and not args.wave:
            data.qpos[: len(HOME_Q)] = q_des[: len(HOME_Q)]
            data.qvel[:15] = 0.0
            data.ctrl[:] = 0.0
            lock_base(data)
            mujoco.mj_forward(model, data)
        else:
            apply_hold(model, data, q_des, kp=args.kp, kd=args.kd)
            mujoco.mj_step(model, data)
            lock_base(data)

    if args.headless:
        z0 = float(data.xpos[falan, 2])
        n = int(2.0 / model.opt.timestep)
        for i in range(n):
            step_once(i * model.opt.timestep)
        z1 = float(data.xpos[falan, 2])
        print(f"  falan_Right z: {z0:.3f} -> {z1:.3f} m")
        if abs(z1 - z0) > 0.15:
            print("FAIL: 末端高度漂移过大")
            return 2
        print("headless OK")
        return 0

    print("打开窗口：关闭窗口或 Ctrl+C 退出")
    t0 = time.time()
    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=True, show_right_ui=False
    ) as viewer:
        while viewer.is_running():
            t = time.time() - t0
            step_once(t)
            viewer.sync()
            time.sleep(float(model.opt.timestep))
            if t > args.seconds:
                break
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
