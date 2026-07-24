"""从真机 Flip Spec ckpt 启动仿真微调。

用法：
  conda activate pyrite && source scripts/setup_env.sh
  python -m sim_acp.scripts.finetune_flip_spec --epochs 30
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_ACP_ML = _REPO / "adaptive_compliance_policy" / "PyriteML"


def default_pretrained() -> Path:
    root = Path(
        os.environ.get("PYRITE_CHECKPOINT_FOLDERS", Path.home() / "training_outputs")
    )
    return (
        root
        / "2026.07.17_14.42.42_flip_up_new_resnet_230"
        / "checkpoints"
        / "latest.ckpt"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Finetune Flip Spec on sim demos")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="真机 Flip Spec ckpt（默认 training_outputs/.../latest.ckpt）",
    )
    parser.add_argument("--epochs", type=int, default=30, help="额外微调 epoch 数")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="覆盖 task.dataset_path",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pretrained = Path(args.pretrained) if args.pretrained else default_pretrained()
    if not pretrained.is_file():
        print(f"找不到预训练权重: {pretrained}")
        return 1

    # 读取 ckpt epoch
    import dill
    import torch

    payload = torch.load(pretrained.open("rb"), pickle_module=dill, map_location="cpu")
    start_epoch = int(dill.loads(payload["pickles"]["epoch"]))
    target_epochs = start_epoch + int(args.epochs)
    print(f"[ft] pretrained epoch={start_epoch} → train until {target_epochs}")

    out_root = Path(
        os.environ.get("PYRITE_CHECKPOINT_FOLDERS", Path.home() / "training_outputs")
    )
    from datetime import datetime

    stamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    run_dir = out_root / f"{stamp}_flip_up_sim_flip_sim_ft"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dest = ckpt_dir / "latest.ckpt"
    shutil.copy2(pretrained, dest)
    print(f"[ft] seeded {dest}")

    # Hydra 会再建一层时间戳目录；用 HYDRA_FULL_ERROR + 指定 output
    # 通过环境变量把 resume 目录固定到我们 seeded 的 run_dir
    overrides = [
        f"training.num_epochs={target_epochs}",
        f"training.resume=True",
        f"optimizer.lr={args.lr}",
        f"dataloader.batch_size={args.batch_size}",
        f"val_dataloader.batch_size={args.batch_size}",
        f"hydra.run.dir={run_dir}",
        f"multi_run.run_dir={run_dir}",
    ]
    if args.dataset:
        overrides.append(f"task.dataset_path={args.dataset}")

    cmd = [
        sys.executable,
        str(
            _ACP_ML
            / "diffusion_policy"
            / "workspace"
            / "train_diffusion_unet_image_workspace.py"
        ),
        "--config-name=train_spec_sim_finetune_workspace",
        *overrides,
    ]
    print("[ft] cmd:", " ".join(cmd))
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    # workspace chdirs to PyriteML
    return subprocess.call(cmd, cwd=str(_ACP_ML), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
