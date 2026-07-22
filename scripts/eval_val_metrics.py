#!/usr/bin/env python3
"""Offline val metrics for an ACP checkpoint → eval_latest_val_metrics.json

Example:
  conda activate pyrite
  export PYTHONNOUSERSITE=1
  export PYTHONPATH=$REPO/adaptive_compliance_policy/PyriteML:$REPO/adaptive_compliance_policy
  source $REPO/scripts/setup_env.sh

  python $REPO/scripts/eval_val_metrics.py \\
    --ckpt ~/training_outputs/2026.07.17_14.42.42_flip_up_new_resnet_230/checkpoints/latest.ckpt \\
    --output-dir ~/training_outputs/2026.07.17_14.42.42_flip_up_new_resnet_230
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("PYTHONNOUSERSITE", "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "adaptive_compliance_policy" / "PyriteML"))
sys.path.insert(0, str(REPO / "adaptive_compliance_policy"))

import dill
import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.val_metrics import evaluate_val_metrics, save_eval_metrics
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
    TrainDiffusionUnetImageWorkspace,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write eval_latest_val_metrics.json (default: ckpt parent/parent)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit val batches (default: full val set)",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    ckpt = args.ckpt.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else ckpt.parent.parent
    )

    payload = torch.load(ckpt.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    OmegaConf.resolve(cfg)

    workspace = TrainDiffusionUnetImageWorkspace(cfg, output_dir=str(output_dir))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer_path = output_dir / "sparse_normalizer.pkl"
    if normalizer_path.is_file():
        sparse_normalizer = pickle.load(open(normalizer_path, "rb"))
    else:
        sparse_normalizer = dataset.get_normalizer()

    workspace.model.set_normalizer(sparse_normalizer)
    if cfg.training.use_ema and workspace.ema_model is not None:
        workspace.ema_model.set_normalizer(sparse_normalizer)

    val_dataset = dataset.get_validation_dataset()
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        persistent_workers=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device)
    policy.eval()

    print(f"Evaluating {ckpt}")
    print(f"val samples={len(val_dataset)} batches={len(val_dataloader)}")
    metrics = evaluate_val_metrics(
        policy=policy,
        val_dataloader=val_dataloader,
        device=device,
        max_batches=args.max_batches,
        show_progress=True,
    )

    # try attach last train_loss from logs.json.txt
    train_loss = None
    logs = output_dir / "logs.json.txt"
    if logs.is_file():
        import json

        last = None
        for line in logs.open():
            o = json.loads(line)
            if "train_loss" in o:
                last = o["train_loss"]
        train_loss = last

    epoch = int(getattr(workspace, "epoch", -1))
    path = save_eval_metrics(
        metrics,
        output_dir=output_dir,
        epoch=epoch if epoch >= 0 else None,
        train_loss=train_loss,
        tag="latest",
    )
    print("Wrote", path)
    print(
        f"reference pose pos RMSE ≈ {metrics['reference_pose_pos_rmse_mm']:.3f} mm\n"
        f"virtual target pos RMSE ≈ {metrics['virtual_target_pos_rmse_mm']:.3f} mm\n"
        f"stiffness RMSE ≈ {metrics['stiffness_rmse_Npm']:.2f} N/m"
    )
    if train_loss is not None:
        print(f"final train loss ≈ {train_loss:.6f}")


if __name__ == "__main__":
    main()
