# Conv（时序卷积力编码）对比实验配置

> 对应论文消融：**ACP w/o FFT**  
> **不修改** 官方原有 `train_conv_workspace.yaml` / `*_conv.yaml`；本目录用独立副本启动。

Spec 基线归档：[`TRAINING_RESULTS_SPEC.zh.md`](TRAINING_RESULTS_SPEC.zh.md)

---

## Spec vs Conv：必须对齐 / 故意不同的项

| 项 | Spec 基线（已完成） | Conv compare（本实验） |
|----|---------------------|-------------------------|
| 力编码器 | `TimmObsEncoderWithForceSpec` + vit-force | `TimmObsEncoderWithForce` + **causalconv** |
| wrench horizon | 7000 | **32** |
| wrench downsample | 1 | **4** |
| normalize_wrench | false | **true**（官方 conv task） |
| 视觉 / 扩散 / lr / epochs | ViT-CLIP，300 epoch，lr 3e-4 | **保持一致** |
| vase batch | 32 | **写入 compare vase workspace = 32** |
| flip batch | 128 | **128** |
| Hydra name（输出目录后缀） | `resnet_230` | `conv_compare_230`（便于和 Spec 目录区分） |

---

## 新建文件（勿改原文件）

均在 `adaptive_compliance_policy/PyriteML/diffusion_policy/config/`：

| 文件 | 用途 |
|------|------|
| `train_conv_compare_flip_workspace.yaml` | 单臂 Conv 对比入口 |
| `train_conv_compare_vase_workspace.yaml` | 双臂 Conv 对比入口（batch=32） |
| `task/flip_up_conv_compare.yaml` | 自 `flip_up_conv.yaml` 复制 |
| `task/vase_wiping_conv_compare.yaml` | 自 `vase_wiping_conv.yaml` 复制 |

原文件仍保留：`train_conv_workspace.yaml`、`flip_up_conv.yaml`、`vase_wiping_conv.yaml`。

---

## 启动命令

```bash
conda activate pyrite
export PYTHONNOUSERSITE=1
export PYRITE_CHECKPOINT_FOLDERS=~/training_outputs
export PYRITE_DATASET_FOLDERS=~/data/real_processed
cd ~/ACP_fx/adaptive_compliance_policy/PyriteML

# 单臂 Flip Conv 对比
HYDRA_FULL_ERROR=1 accelerate launch train.py \
  --config-name=train_conv_compare_flip_workspace \
  logging.mode=disabled

# 双臂 Vase Conv 对比（batch 已在 yaml 内设为 32）
HYDRA_FULL_ERROR=1 accelerate launch train.py \
  --config-name=train_conv_compare_vase_workspace \
  logging.mode=disabled \
  training.eval_metrics_max_batches=20
```

预期输出目录形如：

```text
~/training_outputs/<date>_<time>_flip_up_new_conv_compare_230
~/training_outputs/<date>_<time>_vase_wiping_conv_compare_230
```

跑完后把 loss / `eval_latest_val_metrics.json` 填回与 Spec 表对照即可。

---

## 当前跑次（2026-07-23）

| 任务 | 状态 | run 目录 | 终盘 / 当前 |
|------|------|----------|-------------|
| Flip Conv | ✅ 300 | `~/training_outputs/2026.07.22_10.58.33_flip_up_new_conv_compare_230` | train_loss ≈ 0.0106；ref/virt RMSE 8.94 / 9.96 mm；刚度 492 N/m |
| Vase Conv | 🔄 约 96/300 | `~/training_outputs/2026.07.22_14.38.26_vase_wiping_conv_compare_230` | 训完后填终盘；中断用 `scripts/train_resume.sh` |

Flip Conv 与 Spec（见 `TRAINING_RESULTS_SPEC`）接近，位姿略好、刚度略差。交接总览：[`HANDOVER.zh.md`](HANDOVER.zh.md)。

---

## 对比时看什么

1. **train_loss** 曲线（平台高度、是否更抖）  
2. Vase：**位姿 RMSE (mm)、刚度 RMSE (N/m)**（同 `eval_*_val_metrics.json`）  
3. 真机（若有）：成功率 / 接触力是否更炸  

论文预期：Spec 通常 ≥ Conv；若 Conv 接近，说明你们任务对高频力谱不敏感。
