"""保存 i7 ACP 相机一帧预览。"""
from __future__ import annotations

import sys
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

    from sim_acp.bridge.i7_mujoco_backend import I7MujocoBackend

    backend = I7MujocoBackend(render=False)
    for _ in range(80):
        backend.step(realtime=False)
    rgb = backend.render_rgb()
    out = _REPO / "sim_acp" / "output"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "i7_cam_preview.png"
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path)
    except Exception:
        import numpy as np

        path = out / "i7_cam_preview.npy"
        np.save(path, rgb)
    print("i7 cam OK")
    print(f"  shape={rgb.shape} mean={rgb.mean():.1f}")
    print(f"  saved {path}")
    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
