"""M0：加载精简 MJCF 并 step 若干拍。"""
from __future__ import annotations

import sys
from pathlib import Path

# 允许直接 python sim_acp/scripts/smoke_load_model.py
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    try:
        import mujoco
    except ImportError:
        print("请先: pip install mujoco")
        return 1

    from sim_acp.bridge.mujoco_backend import MujocoSimBackend

    backend = MujocoSimBackend(render=False)
    for _ in range(100):
        backend.step()
    st = backend.read_state()
    print("M0 OK")
    print("  model nq=", backend.model.nq, "nu=", backend.model.nu)
    print("  pose=", st.pose_xyzquat.round(4))
    print("  q=", st.q.round(4))
    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
