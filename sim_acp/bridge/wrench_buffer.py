"""高频 wrench 环形缓冲。"""
from __future__ import annotations

import numpy as np


class WrenchRingBuffer:
    def __init__(self, capacity: int = 7000, dim: int = 6):
        self.capacity = int(capacity)
        self.dim = int(dim)
        self._buf = np.zeros((self.capacity, self.dim), dtype=np.float64)
        self._n = 0
        self._i = 0

    def push(self, wrench: np.ndarray) -> None:
        w = np.asarray(wrench, dtype=np.float64).reshape(self.dim)
        self._buf[self._i] = w
        self._i = (self._i + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)

    def latest(self) -> np.ndarray:
        if self._n == 0:
            return np.zeros(self.dim)
        return self._buf[(self._i - 1) % self.capacity].copy()

    def window(self, length: int | None = None) -> np.ndarray:
        if self._n == 0:
            return np.zeros((0, self.dim))
        length = self._n if length is None else min(int(length), self._n)
        start = (self._i - length) % self.capacity
        if start + length <= self.capacity:
            return self._buf[start : start + length].copy()
        first = self.capacity - start
        return np.vstack(
            [self._buf[start:], self._buf[: length - first]]
        )
