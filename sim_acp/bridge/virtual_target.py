"""方案 A：力对齐变刚度 → 虚拟目标（与 ACP 论文 / 官方 demo 一致）。

软轴 = **强滤波**接触力方向（k_low）；正交方向 = k_high。
k_low 按力幅调度（论文 Eq.7）；x_virt = x_ref + 平滑后的偏移。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def schedule_k_low(
    force_norm: float,
    *,
    k_max: float,
    k_min: float,
    f_low: float,
    f_high: float,
) -> float:
    """论文 Eq.7：|f| 小 → 硬；|f| 大 → 软。"""
    k_max = float(k_max)
    k_min = float(min(k_min, k_max))
    f_low = float(f_low)
    f_high = float(max(f_high, f_low + 1e-6))
    fn = float(force_norm)
    if fn < f_low:
        return k_max
    if fn > f_high:
        return k_min
    return k_max - (k_max - k_min) * (fn - f_low) / (f_high - f_low)


def stiffness_matrix_force_aligned(
    force_xyz: np.ndarray,
    k_low: float,
    k_high: float = 2000.0,
) -> np.ndarray:
    """世界系 3×3：第一主轴沿力方向为 k_low，其余为 k_high。"""
    f = np.asarray(force_xyz, dtype=float).reshape(3)
    n = float(np.linalg.norm(f))
    if n < 1e-9:
        return np.eye(3) * float(k_high)
    x_axis = f / n
    y_axis = np.cross(x_axis, np.array([0.0, 0.0, 1.0]))
    if float(np.linalg.norm(y_axis)) < 1e-6:
        y_axis = np.cross(x_axis, np.array([0.0, 1.0, 0.0]))
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    S = np.column_stack([x_axis, y_axis, z_axis])
    M = np.diag([float(k_low), float(k_high), float(k_high)])
    return S @ M @ S.T


def virtual_target_pos(
    x_ref_pos: np.ndarray,
    force_xyz: np.ndarray,
    k_low: float,
    k_high: float = 2000.0,
    f_eps: float = 1e-3,
) -> np.ndarray:
    """
    x_virt = x_ref + K(f)^{-1} f

    K 在力方向为 k_low、正交为 k_high → 偏移平行于 f，幅度 |f|/k_low。
    """
    x_ref_pos = np.asarray(x_ref_pos, dtype=float).reshape(3)
    f = np.asarray(force_xyz, dtype=float).reshape(3)
    if float(np.linalg.norm(f)) < f_eps:
        return x_ref_pos.copy()
    k_low = max(float(k_low), 1.0)
    k_high = max(float(k_high), k_low)
    K = stiffness_matrix_force_aligned(f, k_low=k_low, k_high=k_high)
    try:
        offset = np.linalg.solve(K, f)
    except np.linalg.LinAlgError:
        offset = f / k_low
    return x_ref_pos + offset


def offset_along_force_ratio(
    x_ref: np.ndarray, x_virt: np.ndarray, force_xyz: np.ndarray
) -> float:
    """|投影到力方向的偏移| / |总偏移|；力对齐时应接近 1。"""
    d = np.asarray(x_virt, dtype=float).reshape(3) - np.asarray(x_ref, dtype=float).reshape(3)
    f = np.asarray(force_xyz, dtype=float).reshape(3)
    dn = float(np.linalg.norm(d))
    fn = float(np.linalg.norm(f))
    if dn < 1e-9 or fn < 1e-9:
        return 1.0
    return abs(float(np.dot(d, f / fn))) / dn


@dataclass
class SchemeAStep:
    x_virt: np.ndarray
    k_low: float
    f_filt: np.ndarray
    soft_axis: np.ndarray  # unit; zeros if no force


class SchemeAController:
    """方案 A：强滤波力 → 调度 k_low → 平滑偏移 → x_virt。

    抑抖要点：
      1) 力 EMA 很慢 + 无力时衰减
      2) 软轴方向连续（防号翻转）
      3) 偏移 EMA + 每步速率限制（不断幅、不抬高 k_eff）
    """

    def __init__(
        self,
        *,
        k_max: float = 2800.0,
        k_min: float = 400.0,
        k_high: float = 2800.0,
        f_low: float = 2.0,
        f_high: float = 25.0,
        force_ema_alpha: float = 0.025,
        offset_ema_alpha: float = 0.06,
        k_ema_alpha: float = 0.05,
        max_offset_step_m: float = 0.0004,  # 0.4 mm/step
        f_eps: float = 1.5,
        schedule: bool = True,
    ):
        self.k_max = float(k_max)
        self.k_min = float(k_min)
        self.k_high = float(k_high)
        self.f_low = float(f_low)
        self.f_high = float(f_high)
        self.alpha_f = float(np.clip(force_ema_alpha, 1e-4, 1.0))
        self.alpha_off = float(np.clip(offset_ema_alpha, 1e-4, 1.0))
        self.alpha_k = float(np.clip(k_ema_alpha, 1e-4, 1.0))
        self.max_offset_step = float(max(max_offset_step_m, 0.0))
        self.f_eps = float(f_eps)
        self.schedule = bool(schedule)
        self._f_filt = np.zeros(3, dtype=float)
        self._offset = np.zeros(3, dtype=float)
        self._soft = np.zeros(3, dtype=float)
        self._k_filt = float(k_max)
        self._initialized = False

    def reset(self) -> None:
        self._f_filt[:] = 0.0
        self._offset[:] = 0.0
        self._soft[:] = 0.0
        self._k_filt = float(self.k_max)
        self._initialized = False

    def filter_force(self, force_xyz: np.ndarray) -> np.ndarray:
        f = np.asarray(force_xyz, dtype=float).reshape(3)
        # 小力死区：抑制接触边缘颤振
        if float(np.linalg.norm(f)) < self.f_eps * 0.5:
            f = np.zeros(3)
        if not self._initialized:
            self._f_filt = f.copy()
            self._initialized = True
        else:
            a = self.alpha_f
            self._f_filt = a * f + (1.0 - a) * self._f_filt
            # 无力时更快泄掉残余，避免软轴空转
            if float(np.linalg.norm(f)) < 1e-9:
                self._f_filt *= 0.92
        return self._f_filt.copy()

    def _blend_soft_axis(self, f_filt: np.ndarray) -> np.ndarray:
        fn = float(np.linalg.norm(f_filt))
        if fn < self.f_eps:
            self._soft *= 0.9
            if float(np.linalg.norm(self._soft)) < 1e-3:
                self._soft[:] = 0.0
            return self._soft.copy()
        u = f_filt / fn
        if float(np.linalg.norm(self._soft)) < 1e-6:
            self._soft = u.copy()
        else:
            # 防止 180° 翻转造成跳变
            if float(np.dot(self._soft, u)) < 0.0:
                u = -u
            a = min(self.alpha_f * 2.0, 0.15)
            v = (1.0 - a) * self._soft + a * u
            n = float(np.linalg.norm(v))
            self._soft = v / n if n > 1e-9 else u
        return self._soft.copy()

    def step(self, x_ref_pos: np.ndarray, force_xyz: np.ndarray) -> SchemeAStep:
        x_ref_pos = np.asarray(x_ref_pos, dtype=float).reshape(3)
        f_filt = self.filter_force(force_xyz)
        fn = float(np.linalg.norm(f_filt))
        soft = self._blend_soft_axis(f_filt)

        if self.schedule:
            k_raw = schedule_k_low(
                fn,
                k_max=self.k_max,
                k_min=self.k_min,
                f_low=self.f_low,
                f_high=self.f_high,
            )
        else:
            k_raw = self.k_min
        self._k_filt = (
            self.alpha_k * k_raw + (1.0 - self.alpha_k) * self._k_filt
        )
        k_low = float(self._k_filt)

        # 期望偏移（沿滤波力）；再用 EMA + 速率限制平滑
        if fn < self.f_eps:
            offset_cmd = np.zeros(3)
        else:
            offset_cmd = f_filt / max(k_low, 1.0)

        a = self.alpha_off
        offset_tgt = a * offset_cmd + (1.0 - a) * self._offset
        d = offset_tgt - self._offset
        dn = float(np.linalg.norm(d))
        if self.max_offset_step > 0.0 and dn > self.max_offset_step:
            d = d * (self.max_offset_step / dn)
        self._offset = self._offset + d

        x_virt = x_ref_pos + self._offset
        return SchemeAStep(
            x_virt=x_virt, k_low=k_low, f_filt=f_filt, soft_axis=soft
        )


def soft_axis_from_policy(
    x_ref_pos: np.ndarray,
    x_virt_pos: np.ndarray,
    force_xyz: np.ndarray | None = None,
    eps: float = 1e-3,
) -> np.ndarray:
    """官方重建：软轴优先 = normalize(x_virt − x_ref)；过小则退回力方向。"""
    d = (
        np.asarray(x_virt_pos, dtype=float).reshape(3)
        - np.asarray(x_ref_pos, dtype=float).reshape(3)
    )
    dn = float(np.linalg.norm(d))
    if dn >= eps:
        return d / dn
    if force_xyz is not None:
        f = np.asarray(force_xyz, dtype=float).reshape(3)
        fn = float(np.linalg.norm(f))
        if fn >= eps:
            return f / fn
    return np.zeros(3)


class PolicyVirtTracker:
    """平滑跟踪策略 x_virt（EMA + 每步限速），抑域差跳变。"""

    def __init__(
        self,
        *,
        ema_alpha: float = 0.08,
        max_step_m: float = 0.002,
    ):
        self.alpha = float(np.clip(ema_alpha, 1e-4, 1.0))
        self.max_step = float(max(max_step_m, 0.0))
        self._x: np.ndarray | None = None

    def reset(self, x0: np.ndarray | None = None) -> None:
        self._x = None if x0 is None else np.asarray(x0, dtype=float).reshape(3).copy()

    def step(self, x_virt_cmd: np.ndarray) -> np.ndarray:
        cmd = np.asarray(x_virt_cmd, dtype=float).reshape(3)
        if self._x is None:
            self._x = cmd.copy()
            return self._x.copy()
        tgt = self.alpha * cmd + (1.0 - self.alpha) * self._x
        d = tgt - self._x
        dn = float(np.linalg.norm(d))
        if self.max_step > 0.0 and dn > self.max_step:
            d = d * (self.max_step / dn)
        self._x = self._x + d
        return self._x.copy()
