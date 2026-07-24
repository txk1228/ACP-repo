"""末端位姿环形缓冲（策略 pose horizon）。"""
from __future__ import annotations

from collections import deque

import numpy as np


class PoseRingBuffer:
    """存 sim pose7: [x,y,z,qx,qy,qz,qw]。"""

    def __init__(self, capacity: int = 16):
        self.capacity = int(capacity)
        self._poses: deque[np.ndarray] = deque(maxlen=self.capacity)

    def push(self, pose_xyzquat: np.ndarray) -> None:
        p = np.asarray(pose_xyzquat, dtype=np.float64).reshape(7)
        self._poses.append(p.copy())

    def latest(self) -> np.ndarray:
        if not self._poses:
            return np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64)
        return self._poses[-1].copy()

    def stack_last(self, n: int) -> np.ndarray:
        """返回 (n, 7)；不足时用最早帧填充。"""
        n = int(n)
        if not self._poses:
            return np.tile(
                np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64), (n, 1)
            )
        frames = list(self._poses)
        while len(frames) < n:
            frames.insert(0, frames[0])
        return np.stack(frames[-n:], axis=0)
