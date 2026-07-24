"""M1 冒烟：无策略，末端跟踪水平慢圆。"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("请先: pip install mujoco")
        return 1

    import numpy as np

    from sim_acp.bridge.mujoco_backend import MujocoSimBackend

    backend = MujocoSimBackend(render=False)
    center = np.array([0.4, 0.0, 0.35])
    radius = 0.08
    errs = []
    t0 = time.time()
    steps = 2000
    for i in range(steps):
        t = i * backend.model.opt.timestep
        target = center + np.array(
            [radius * math.cos(t), radius * math.sin(t), 0.0]
        )
        pose = np.array(
            [target[0], target[1], target[2], 0, 0, 0, 1], dtype=float
        )
        backend.write_ee_pose(pose, time.time_ns())
        backend.step()
        if i > 200:
            pos = backend.read_state().pose_xyzquat[:3]
            errs.append(np.linalg.norm(pos - target))
    backend.close()
    mean_err = float(np.mean(errs)) if errs else float("nan")
    print("M1 track circle done")
    print(f"  mean |e| after settle = {mean_err:.4f} m")
    print(f"  wall time = {time.time() - t0:.2f}s for {steps} steps")
    ok = mean_err < 0.03
    print("  PASS" if ok else "  FAIL (expect < 0.03 m)")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
