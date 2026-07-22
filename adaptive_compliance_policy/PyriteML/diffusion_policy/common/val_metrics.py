"""Physical validation metrics for ACP actions (pose / virtual target / stiffness).

Action layout (pose9pose9s1), per arm of 19 dims:
  [0:3]   reference pose xyz (m)
  [3:9]   reference pose rot6d
  [9:12]  virtual target xyz (m)
  [12:18] virtual target rot6d
  [18]    stiffness (N/m)

Dual-arm vase: 38 = arm0 (0:19) + arm1 (19:38).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_policy.common.pytorch_util import dict_apply


def _arm_slices(action_dim: int) -> list[tuple[int, int]]:
    """Return (start, end) for each 19-dim arm block."""
    if action_dim % 19 != 0:
        raise ValueError(f"Unexpected action_dim={action_dim}, expected multiple of 19")
    n_arms = action_dim // 19
    return [(i * 19, (i + 1) * 19) for i in range(n_arms)]


def physical_rmse_from_actions(
    pred: torch.Tensor, gt: torch.Tensor
) -> Dict[str, float]:
    """Compute physical RMSE from denormalized action tensors.

    Args:
        pred, gt: (B, T, A) float tensors in physical units.
    """
    assert pred.shape == gt.shape
    action_dim = pred.shape[-1]
    slices = _arm_slices(action_dim)

    ref_pos_err = []
    vt_pos_err = []
    stiff_err = []
    for s, e in slices:
        arm_p = pred[..., s:e]
        arm_g = gt[..., s:e]
        ref_pos_err.append((arm_p[..., 0:3] - arm_g[..., 0:3]).reshape(-1, 3))
        vt_pos_err.append((arm_p[..., 9:12] - arm_g[..., 9:12]).reshape(-1, 3))
        stiff_err.append((arm_p[..., 18] - arm_g[..., 18]).reshape(-1))

    ref = torch.cat(ref_pos_err, dim=0)
    vt = torch.cat(vt_pos_err, dim=0)
    stiff = torch.cat(stiff_err, dim=0)

    ref_rmse_m = torch.sqrt(torch.mean(ref**2)).item()
    vt_rmse_m = torch.sqrt(torch.mean(vt**2)).item()
    stiff_rmse = torch.sqrt(torch.mean(stiff**2)).item()

    return {
        "reference_pose_pos_rmse_m": ref_rmse_m,
        "reference_pose_pos_rmse_mm": ref_rmse_m * 1000.0,
        "virtual_target_pos_rmse_m": vt_rmse_m,
        "virtual_target_pos_rmse_mm": vt_rmse_m * 1000.0,
        "stiffness_rmse_Npm": stiff_rmse,
        # screenshot-style aliases
        "reference pose pos RMSE (mm)": ref_rmse_m * 1000.0,
        "virtual target pos RMSE (mm)": vt_rmse_m * 1000.0,
        "stiffness RMSE (N/m)": stiff_rmse,
    }


@torch.no_grad()
def evaluate_val_metrics(
    policy,
    val_dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run diffusion prediction on val set and return physical RMSE metrics."""
    policy.eval()
    preds = []
    gts = []
    n_batches = 0
    iterator = val_dataloader
    if show_progress:
        iterator = tqdm(val_dataloader, desc="val metrics", leave=False)

    for batch in iterator:
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        gt_action = batch["action"]["sparse"]
        pred_action = policy.predict_action(batch["obs"])["sparse"]
        # align time dim if needed
        t = min(pred_action.shape[1], gt_action.shape[1])
        preds.append(pred_action[:, :t].detach().float().cpu())
        gts.append(gt_action[:, :t].detach().float().cpu())
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    if not preds:
        raise RuntimeError("Validation dataloader produced no batches")

    pred = torch.cat(preds, dim=0)
    gt = torch.cat(gts, dim=0)
    metrics = physical_rmse_from_actions(pred, gt)
    metrics["num_batches"] = n_batches
    metrics["num_samples"] = int(pred.shape[0])
    metrics["action_dim"] = int(pred.shape[-1])
    metrics["horizon"] = int(pred.shape[1])
    return metrics


def save_eval_metrics(
    metrics: Dict[str, Any],
    output_dir: str | Path,
    epoch: Optional[int] = None,
    train_loss: Optional[float] = None,
    tag: str = "latest",
) -> Path:
    """Write eval_{tag}_val_metrics.json (and epoch-tagged copy)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(metrics)
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if train_loss is not None:
        payload["final_train_loss"] = float(train_loss)
        payload["train_loss"] = float(train_loss)

    # keep JSON serializable
    clean = {}
    for k, v in payload.items():
        if isinstance(v, (float, np.floating)):
            clean[k] = float(v)
        elif isinstance(v, (int, np.integer)):
            clean[k] = int(v)
        elif isinstance(v, str):
            clean[k] = v
        elif v is None or isinstance(v, (bool, list, dict)):
            clean[k] = v
        else:
            clean[k] = float(v) if hasattr(v, "item") else str(v)

    latest_path = output_dir / f"eval_{tag}_val_metrics.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if epoch is not None:
        epoch_path = output_dir / f"eval_epoch_{epoch:04d}_val_metrics.json"
        with open(epoch_path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return latest_path


def metrics_for_logger(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Flatten key physical metrics as plain floats for JsonLogger / wandb."""
    return {
        "val_reference_pose_pos_rmse_mm": float(metrics["reference_pose_pos_rmse_mm"]),
        "val_virtual_target_pos_rmse_mm": float(metrics["virtual_target_pos_rmse_mm"]),
        "val_stiffness_rmse_Npm": float(metrics["stiffness_rmse_Npm"]),
    }
