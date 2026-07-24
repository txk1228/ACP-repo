"""RobotBackend 契约：仿真 / 真机共用。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass
class RobotState:
    """与真机 tcp_pos / tcp_wrench 约定对齐。"""

    pose_xyzquat: np.ndarray  # (7,) [x,y,z,qx,qy,qz,qw]
    wrench_W: np.ndarray  # (6,) [fx,fy,fz,mx,my,mz]
    q: np.ndarray  # (n,)
    timestamp_ns: int


class RobotBackend(Protocol):
    def read_state(self) -> RobotState: ...

    def write_ee_pose(
        self, pose_xyzquat: np.ndarray, timestamp_ns: int
    ) -> None: ...

    def step(self) -> None: ...

    def inject_force_W(self, force_xyz: np.ndarray) -> None:
        """仿真联调用；真机后端可 no-op。"""
        ...
