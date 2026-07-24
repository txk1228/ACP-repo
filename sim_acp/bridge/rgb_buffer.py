"""短时 RGB 帧缓冲（策略 horizon 用）。"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np


def resize_rgb(rgb_hwc: np.ndarray, h: int, w: int) -> np.ndarray:
    """最近邻缩放到 (h,w,3) uint8。"""
    img = np.asarray(rgb_hwc, dtype=np.uint8)
    assert img.ndim == 3 and img.shape[2] == 3
    if img.shape[0] == h and img.shape[1] == w:
        return img.copy()
    try:
        import cv2

        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    except ImportError:
        # 无 cv2：简单分块采样
        ys = (np.linspace(0, img.shape[0] - 1, h)).astype(int)
        xs = (np.linspace(0, img.shape[1] - 1, w)).astype(int)
        return img[ys][:, xs].copy()


class RgbRingBuffer:
    def __init__(self, capacity: int = 8, h: int = 224, w: int = 224):
        self.capacity = int(capacity)
        self.h = int(h)
        self.w = int(w)
        self._frames: deque[np.ndarray] = deque(maxlen=self.capacity)

    def push(self, rgb_hwc: np.ndarray) -> None:
        img = resize_rgb(rgb_hwc, self.h, self.w)
        self._frames.append(img)

    def latest(self) -> np.ndarray:
        if not self._frames:
            return np.full((self.h, self.w, 3), 128, dtype=np.uint8)
        return self._frames[-1].copy()

    def stack_last(self, n: int) -> np.ndarray:
        """返回 (n, H, W, 3)；不足时用最早/灰色填充。"""
        n = int(n)
        if not self._frames:
            return np.full((n, self.h, self.w, 3), 128, dtype=np.uint8)
        frames = list(self._frames)
        while len(frames) < n:
            frames.insert(0, frames[0])
        frames = frames[-n:]
        return np.stack(frames, axis=0)

    def save_latest(self, path: str | Path) -> Path | None:
        if not self._frames:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image

            Image.fromarray(self.latest()).save(path)
        except ImportError:
            # 极简 PPM
            img = self.latest()
            with open(path.with_suffix(".ppm"), "wb") as f:
                header = f"P6\n{img.shape[1]} {img.shape[0]}\n255\n".encode()
                f.write(header)
                f.write(img.tobytes())
            return path.with_suffix(".ppm")
        return path
