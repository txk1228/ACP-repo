"""M5：相机渲 RGB 冒烟（保存一张预览图）。"""
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

    from sim_acp.bridge.mujoco_backend import MujocoSimBackend

    backend = MujocoSimBackend(render=False)
    for _ in range(50):
        backend.step(realtime=False)
    rgb = backend.render_rgb()
    out = _REPO / "sim_acp" / "output"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "cam_preview.png"
    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, rgb)
    except Exception:
        # 无 imageio 时写原始 ppm 头 + 数据不便；用 PIL 或跳过
        try:
            from PIL import Image

            Image.fromarray(rgb).save(path)
        except Exception:
            path = out / "cam_preview.npy"
            import numpy as np

            np.save(path, rgb)
    print("M5 cam RGB OK")
    print(f"  shape={rgb.shape} dtype={rgb.dtype} mean={rgb.mean():.1f}")
    print(f"  saved {path}")
    backend.close()
    assert rgb.shape == (224, 224, 3)
    assert rgb.dtype == __import__("numpy").uint8
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
