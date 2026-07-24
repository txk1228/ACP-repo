"""右臂 IK 跟踪慢圆（i7 MJCF）。"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("请先: pip install mujoco")
        return 1

    import numpy as np

    from sim_acp.bridge.i7_mujoco_backend import I7MujocoBackend

    backend = I7MujocoBackend(render=False)
    st0 = backend.read_state()
    center = st0.pose_xyzquat[:3].copy()
    print(f"EE start = {center.round(3)}")
    # 水平小圆，保持高度
    radius = 0.06
    errs = []
    steps = 2500
    t0 = time.time()
    for i in range(steps):
        t = i * backend.model.opt.timestep
        target = center + np.array(
            [radius * math.cos(0.8 * t), radius * math.sin(0.8 * t), 0.0]
        )
        pose = np.array(
            [target[0], target[1], target[2], 0, 0, 0, 1], dtype=float
        )
        backend.write_ee_pose(pose, time.time_ns())
        backend.step(realtime=False)
        if i > 400:
            pos = backend.read_state().pose_xyzquat[:3]
            errs.append(float(np.linalg.norm(pos - target)))
    backend.close()
    mean_err = float(np.mean(errs)) if errs else float("nan")
    print("i7 IK track circle")
    print(f"  mean |e| = {mean_err:.4f} m")
    print(f"  wall = {time.time() - t0:.2f}s")
    ok = mean_err < 0.04
    print("  PASS" if ok else "  FAIL (expect < 0.04 m)")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
