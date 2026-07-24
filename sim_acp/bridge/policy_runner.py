"""加载 Flip Spec：RGB + wrench + pose 三模态观测 → x_ref/x_virt/k_low。"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_ACP = _REPO / "adaptive_compliance_policy"
if str(_ACP) not in sys.path:
    sys.path.insert(0, str(_ACP))


@dataclass
class PolicyAction:
    """解码后的物理量（世界系）。"""

    x_ref_pos: np.ndarray  # (3,) 第 0 步
    x_virt_pos: np.ndarray  # (3,) 第 0 步
    k_low: float
    soft_axis: np.ndarray  # (3,) unit from virt−ref
    raw_action_t0: np.ndarray  # (19,)
    inference_s: float
    # 完整 sparse action 轨迹（世界系位置），用于开环执行 horizon
    x_ref_traj: np.ndarray | None = None  # (H, 3)
    x_virt_traj: np.ndarray | None = None  # (H, 3)
    k_traj: np.ndarray | None = None  # (H,)


def _ensure_acp_path() -> None:
    if str(_ACP) not in sys.path:
        sys.path.insert(0, str(_ACP))


def default_flip_spec_ckpt() -> Path:
    root = Path(
        os.environ.get("PYRITE_CHECKPOINT_FOLDERS", Path.home() / "training_outputs")
    )
    return (
        root
        / "2026.07.17_14.42.42_flip_up_new_resnet_230"
        / "checkpoints"
        / "latest.ckpt"
    )


def pose_xyzquat_sim_to_pose9(pose7_sim: np.ndarray) -> np.ndarray:
    """sim [x,y,z,qx,qy,qz,qw] → ACP pose9 [xyz + rot6d]。"""
    _ensure_acp_path()
    import PyriteUtility.spatial_math.spatial_utilities as su

    p = np.asarray(pose7_sim, dtype=np.float64).reshape(7)
    pose7_acp = np.array(
        [p[0], p[1], p[2], p[6], p[3], p[4], p[5]], dtype=np.float64
    )
    return su.SE3_to_pose9(su.pose7_to_SE3(pose7_acp[None, :]))[0]


class FlipSpecPolicyRunner:
    """Flip Spec：真三模态（RGB / wrench / pose 历史）→ predict_action。"""

    def __init__(
        self,
        ckpt_path: str | Path | None = None,
        device: str | None = None,
    ):
        _ensure_acp_path()
        import torch
        from PyriteUtility.pytorch_utils.model_io import load_policy

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        ckpt = Path(ckpt_path) if ckpt_path else default_flip_spec_ckpt()
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"找不到 checkpoint: {ckpt}\n"
                "请确认 PYRITE_CHECKPOINT_FOLDERS 与 Flip Spec 路径。"
            )
        print(f"[policy] loading {ckpt} on {self.device} ...")
        self.policy, self.shape_meta = load_policy(str(ckpt), self.device)
        self.id_list = list(self.shape_meta.get("id_list", [0]))
        sample = self.shape_meta["sample"]["obs"]["sparse"]
        self.rgb_h = int(sample["rgb_0"]["horizon"])
        self.pose_h = int(sample["robot0_eef_pos"]["horizon"])
        self.wrench_h = int(sample["robot0_eef_wrench"]["horizon"])
        rgb_shape = self.shape_meta["obs"]["rgb_0"]["shape"]  # C,H,W
        self.rgb_chw = tuple(int(x) for x in rgb_shape)
        print(
            f"[policy] ready: rgb_h={self.rgb_h} pose_h={self.pose_h} "
            f"wrench_h={self.wrench_h} rgb_chw={self.rgb_chw} "
            f"action_dim={self.shape_meta['action']['shape'][0]}"
        )

    @property
    def rgb_hw(self) -> tuple[int, int]:
        _, h, w = self.rgb_chw
        return int(h), int(w)

    def build_obs_sparse(
        self,
        pose_history_sim: np.ndarray,
        wrench_history: np.ndarray,
        rgb_uint8: Optional[np.ndarray] = None,
        *,
        fake_rgb: bool = False,
    ) -> dict:
        """构造 obs_sparse。

        pose_history_sim: (T,7) 或 (7,) sim xyzquat；T 应对齐 pose_h。
        rgb_uint8: (H,W,3) 或 (rgb_h,H,W,3)；fake_rgb 时忽略并填灰图。
        """
        from sim_acp.bridge.rgb_buffer import resize_rgb

        ph = np.asarray(pose_history_sim, dtype=np.float64)
        if ph.ndim == 1:
            ph = ph.reshape(1, 7)
        if ph.shape[0] < self.pose_h:
            pad = np.repeat(ph[:1], self.pose_h - ph.shape[0], axis=0)
            ph = np.concatenate([pad, ph], axis=0)
        ph = ph[-self.pose_h :]

        pos = np.zeros((self.pose_h, 3), dtype=np.float32)
        rot = np.zeros((self.pose_h, 6), dtype=np.float32)
        for i in range(self.pose_h):
            p9 = pose_xyzquat_sim_to_pose9(ph[i])
            pos[i] = p9[:3]
            rot[i] = p9[3:]

        wh = np.asarray(wrench_history, dtype=np.float32)
        if wh.ndim == 1:
            wh = wh.reshape(1, -1)
        if wh.shape[0] < self.wrench_h:
            pad = np.zeros((self.wrench_h - wh.shape[0], 6), dtype=np.float32)
            if wh.shape[0] > 0:
                pad[:] = wh[0]
            wh = np.concatenate([pad, wh], axis=0)
        wh = wh[-self.wrench_h :]
        if wh.shape[1] != 6:
            raise ValueError(f"wrench dim {wh.shape[1]} != 6")

        _, h, w = self.rgb_chw
        if fake_rgb or rgb_uint8 is None:
            frame = np.full((h, w, 3), 128, dtype=np.uint8)
            rgb = np.stack([frame] * self.rgb_h, axis=0)
        else:
            rgb = np.asarray(rgb_uint8)
            if rgb.ndim == 3:
                rgb = resize_rgb(rgb, h, w)
                rgb = np.stack([rgb] * self.rgb_h, axis=0)
            else:
                assert rgb.ndim == 4 and rgb.shape[0] == self.rgb_h
                rgb = np.stack(
                    [resize_rgb(rgb[i], h, w) for i in range(self.rgb_h)], axis=0
                )

        return {
            "rgb_0": rgb,
            "robot0_eef_pos": pos,
            "robot0_eef_rot_axis_angle": rot,
            "robot0_eef_wrench": wh,
        }

    def predict(
        self,
        pose_history_sim: np.ndarray,
        wrench_history: np.ndarray,
        rgb_uint8: Optional[np.ndarray] = None,
        *,
        fake_rgb: bool = False,
        force_xyz: Optional[np.ndarray] = None,
    ) -> PolicyAction:
        _ensure_acp_path()
        import time

        import torch
        from einops import rearrange
        from PyriteConfig.tasks.common import common_type_conversions as task
        from PyriteUtility.common import dict_apply

        from sim_acp.bridge.virtual_target import soft_axis_from_policy

        obs_sparse = self.build_obs_sparse(
            pose_history_sim,
            wrench_history,
            rgb_uint8,
            fake_rgb=fake_rgb,
        )
        obs_sample_np, SE3_WBase = task.sparse_obs_to_obs_sample(
            obs_sparse=obs_sparse,
            shape_meta=self.shape_meta,
            reshape_mode="reshape",
            id_list=self.id_list,
            ignore_rgb=False,
        )
        obs_batch = dict_apply(obs_sample_np, lambda x: rearrange(x, "... -> 1 ..."))
        obs_torch = dict_apply(
            obs_batch, lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
        )

        t0 = time.perf_counter()
        with torch.no_grad():
            result = self.policy.predict_action({"sparse": obs_torch})
            raw = result["sparse"][0].detach().cpu().numpy()
        dt = time.perf_counter() - t0

        mats_ref, mats_vt, stiff = task.action19_postprocess(
            raw, SE3_WBase, self.id_list
        )
        # mats_*[id][t] = 4x4; stiff[id][t]
        H = len(mats_vt[0])
        x_ref_traj = np.zeros((H, 3), dtype=np.float64)
        x_virt_traj = np.zeros((H, 3), dtype=np.float64)
        k_traj = np.zeros(H, dtype=np.float64)
        for t in range(H):
            x_ref_traj[t] = mats_ref[0][t][:3, 3]
            x_virt_traj[t] = mats_vt[0][t][:3, 3]
            k_traj[t] = float(np.asarray(stiff[0][t]).reshape(-1)[0])
        x_ref = x_ref_traj[0].copy()
        x_virt = x_virt_traj[0].copy()
        k0 = float(k_traj[0])
        soft = soft_axis_from_policy(x_ref, x_virt, force_xyz=force_xyz)

        return PolicyAction(
            x_ref_pos=x_ref,
            x_virt_pos=x_virt,
            k_low=k0,
            soft_axis=soft,
            raw_action_t0=raw[0].copy(),
            inference_s=dt,
            x_ref_traj=x_ref_traj,
            x_virt_traj=x_virt_traj,
            k_traj=k_traj,
        )
