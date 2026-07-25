# Conv（时序卷积力编码）对比实验配置

> 对应论文消融：**ACP w/o FFT**  
> **不修改** 官方原有 `train_conv_workspace.yaml` / `*_conv.yaml`；本目录用独立副本启动。  
> **归档日期**：2026-07-25（Flip + Vase 均已跑满 300 epoch）

Spec 基线归档：[`TRAINING_RESULTS_SPEC.zh.md`](TRAINING_RESULTS_SPEC.zh.md)  
仓库快照：`docs/training_snapshots/conv_runs_index.json` 及 `*_conv_eval_*.json`

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

规范输出目录：

```text
~/training_outputs/2026.07.22_10.58.33_flip_up_new_conv_compare_230
~/training_outputs/2026.07.22_14.38.26_vase_wiping_conv_compare_230
```

---

## 终盘结果（2026-07-25）

进度面板终盘截图（Spec + Conv 四条均为 DONE）：[`docs/figures/train_progress_panel_final.png`](figures/train_progress_panel_final.png) · 读法见 [`TRAINING_RESULTS_SPEC.zh.md`](TRAINING_RESULTS_SPEC.zh.md)。

| 任务 | 状态 | run 目录 | 终盘 |
|------|------|----------|------|
| Flip Conv | ✅ 300 | `~/training_outputs/2026.07.22_10.58.33_flip_up_new_conv_compare_230` | train_loss @299 ≈ **0.0106**；`latest.ckpt` ≈ epoch **290** |
| Vase Conv | ✅ 300 | `~/training_outputs/2026.07.22_14.38.26_vase_wiping_conv_compare_230` | train_loss @299 ≈ **0.0102**；`latest.ckpt` ≈ epoch **290** |

与 Spec 一样：`checkpoint_every=10`，范围内最后一存是 290；日志已覆盖 epoch **0～299**。`eval_latest_val_metrics.json` 可能标成 epoch **300**（结束后多一次 eval），部署仍用 `latest.ckpt`。

### Spec vs Conv 验证集 RMSE（对齐 @ epoch 290）

| 任务 | 编码 | train_loss @299 | 位姿 RMSE ref / virt (mm) | 刚度 RMSE (N/m) |
|------|------|----------------:|--------------------------:|----------------:|
| Flip | Spec | ≈ 0.0106 | **9.17 / 10.37** | **482** |
| Flip | Conv | ≈ 0.0106 | **9.04 / 10.03** | **477** |
| Vase | Spec | ≈ 0.0102 | **22.46 / 24.29** | **834** |
| Vase | Conv | ≈ 0.0102 | **22.53 / 24.20** | **809** |

> Flip Conv 的 `eval_latest`（标 epoch 300）为 8.94 / 9.96 mm、刚度 492 N/m；Vase Conv `eval_latest` 为 21.98 / 24.03 mm、刚度 865 N/m。上表用 **@290** 与 Spec 归档口径对齐。

仓库快照：

- `docs/training_snapshots/flip_conv_eval_epoch_0290_val_metrics.json`
- `docs/training_snapshots/flip_conv_eval_latest_val_metrics.json`
- `docs/training_snapshots/vase_conv_eval_epoch_0290_val_metrics.json`
- `docs/training_snapshots/vase_conv_eval_latest_val_metrics.json`
- `docs/training_snapshots/conv_runs_index.json`

---

## 对比时看什么

1. **train_loss** 曲线（平台高度、是否更抖）  
2. Vase：**位姿 RMSE (mm)、刚度 RMSE (N/m)**（同 `eval_*_val_metrics.json`）  
3. 真机（若有）：成功率 / 接触力是否更炸  

**离线结论**：Flip / Vase 上 Spec≈Conv（位姿与刚度同量级），说明这两份数据对高频力谱不敏感；论文预期「Spec ≥ Conv」在本复现 offline RMSE 上未拉开差距。**最终以真机为准**。

交接总览：[`HANDOVER.zh.md`](HANDOVER.zh.md)。
