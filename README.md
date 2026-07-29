# ACP-repo

[Adaptive Compliance Policy (ACP)](https://arxiv.org/abs/2410.09309) 复现仓库：算法 Demo、Spec/Conv 训练、**MuJoCo 仿真翻方块（`sim_acp/`）**，以及至简 i7 Pro 真机适配方案。

- 论文主页：[adaptive-compliance.github.io](https://adaptive-compliance.github.io/)
- 官方代码：[yifan-hou/adaptive_compliance_policy](https://github.com/yifan-hou/adaptive_compliance_policy)

### 效果预览

![ACP sim flip v2-ft trimodal](docs/media/sim_flip_v2ft.gif)

**仿真翻方块 v2-ft**：在官方 **UR5e 真机数据**上训练 Flip Spec 后，迁移至本仓 **i7** MuJoCo URDF；零样本受域差限制，经仿真专家数据微调后策略可翻。  
左：外置机位 + 腕部 RGB；右：刚度面板。接触段 `k_soft` < `k_hard`、软轴 `û` 随接触转向，`tilt` → ~90°。

完整视频：[`sim_flip_v2ft.mp4`](docs/media/sim_flip_v2ft.mp4) · 运行：`bash scripts/run_sim_flip.sh`  
读图与终盘曲线：[效果展示页](docs/media/README.md)

递进验收：`v1` 脚本专家 → `v2` Spec 零样本（验链路）→ `v2-ft` 仿真微调（可翻）。可选 UI（同一 `ACP_SIM_FT_CKPT`）：`bash scripts/run_sim_flip.sh v2-ft-live`。

---

## 流程总览

按下列顺序推进；每步对应本页小节。

| 步骤 | 内容 | 跳转 |
|------|------|------|
| **0** | 项目状态与路径 | [→ 项目状态](#0-项目状态) |
| **1** | 环境搭建 | [→ 环境搭建](#1-环境搭建) |
| **2** | 阶段一 Demo（无需 GPU） | [→ Demo](#2-阶段一-demo) |
| **3** | 阶段二训练（冒烟 → 完整 → 续训 → 对照） | [→ 训练](#3-阶段二训练) |
| **4** | 仿真翻方块 `sim_acp/` | [→ 仿真](#4-仿真翻方块-sim_acp) |
| **5** | 真机适配（未开始） | [→ 真机](#5-真机适配) |

参考：[目录结构](#目录结构) · [文档索引](#文档索引) · [同步](#同步) · [License](#license)

### 进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| 阶段一 Demo | ✅ | 虚拟目标 + 变刚度可视化 |
| 阶段二 Spec | ✅ | Flip-up + Vase，300 epoch；[`TRAINING_RESULTS_SPEC`](docs/TRAINING_RESULTS_SPEC.zh.md) |
| 阶段二 Conv | ✅ | Flip + Vase 对照；[`TRAINING_CONV_COMPARE`](docs/TRAINING_CONV_COMPARE.zh.md) |
| 仿真 `sim_acp/` | ✅ | v2-ft 策略翻方块 PASS（腕部 RGB + 微调） |
| 阶段三 真机 | 📋 | 方案见 [`I7_ACP_ADAPTATION`](docs/I7_ACP_ADAPTATION.zh.md) |

推荐路径：Spec/Conv 归档 → **`sim_acp/` 仿真翻方块** → 有机后按适配文档上真机。

---

## 0. 项目状态

通读 [`docs/HANDOVER.zh.md`](docs/HANDOVER.zh.md)：进度、本机路径、续训入口、真机待办。

---

## 1. 环境搭建

```bash
git clone https://github.com/txk1228/ACP-repo.git
cd ACP-repo
```

训练环境（GPU / CUDA，首次）：

```bash
bash scripts/setup_pyrite_env_cuda.sh
conda activate pyrite
source scripts/setup_env.sh
```

Demo 环境（可无 GPU）：

```bash
bash scripts/setup_demo_env.sh
conda activate acp-demo
```

`setup_env.sh` 设置：`PYRITE_DATASET_FOLDERS=~/data/real_processed`、`PYRITE_CHECKPOINT_FOLDERS=~/training_outputs`。

---

## 2. 阶段一 Demo

虚拟目标 + 变刚度可视化，无需 GPU。

```bash
conda activate acp-demo
bash scripts/run_demo.sh
```

入口：`demo/virtual_target_stiffness_demo.py`；图示见 [效果展示页](docs/media/README.md)。输出目录：`demo/output/`（gitignore）。

---

## 3. 阶段二训练

流程：冒烟 → 完整训练 → 进度面板 → 断点续训 → Spec/Conv 结果对照。

### 冒烟 → 完整训练

```bash
conda activate pyrite
source scripts/setup_env.sh

bash scripts/train_acp_smoke.sh      # 短跑冒烟
bash scripts/train_acp.sh spec       # Spec（论文主方法 / FFT）
# bash scripts/train_vase.sh         # 双臂 vase Spec
```

- 默认每 **10** epoch 写入 `~/training_outputs/<run_dir>/checkpoints/latest.ckpt`
- Conv 消融（ACP w/o FFT）：[`docs/TRAINING_CONV_COMPARE.zh.md`](docs/TRAINING_CONV_COMPARE.zh.md)

### 进度面板

```bash
bash scripts/run_progress_panel.sh   # 后台
# http://127.0.0.1:8765
```

| 操作 | 命令 |
|------|------|
| 启动 | `bash scripts/run_progress_panel.sh` |
| 前台调试 | `python3 scripts/train_progress_server.py` |
| 停止 | `fuser -k 8765/tcp` |

读取 `~/training_outputs/*/logs.json.txt` 与 `logs/train_*.log`。

### 断点续训

中断后对**同一 run 目录**设置 `training.resume=true`，从 `latest.ckpt` 恢复；勿另开新 run。

```bash
ls -lt ~/training_outputs | head
ls ~/training_outputs/<run_dir>/checkpoints/latest.ckpt

bash scripts/train_resume.sh <run_dir> <config-name> [hydra 覆盖项...]
```

示例：

```bash
bash scripts/train_vase_resume.sh

bash scripts/train_resume.sh \
  ~/training_outputs/2026.07.22_14.38.26_vase_wiping_conv_compare_230 \
  train_conv_compare_vase_workspace \
  training.eval_metrics_max_batches=20

bash scripts/train_resume.sh \
  ~/training_outputs/2026.07.17_14.42.42_flip_up_new_resnet_230 \
  train_spec_workspace \
  task=flip_up_spec

bash scripts/train_resume.sh \
  ~/training_outputs/2026.07.22_10.58.33_flip_up_new_conv_compare_230 \
  train_conv_compare_flip_workspace
```

| 要点 | 说明 |
|------|------|
| 指向原 run 目录 | 否则 Hydra 会新建目录，等价重开 |
| 覆盖项与初训一致 | 如 vase 的 `batch_size=32` |
| 尚无 `latest.ckpt` | 未满存盘间隔，无法续训 |
| 已满 300 epoch | 续训立即结束 |

### 结果与对照

- Spec：[`docs/TRAINING_RESULTS_SPEC.zh.md`](docs/TRAINING_RESULTS_SPEC.zh.md)
- Conv：[`docs/TRAINING_CONV_COMPARE.zh.md`](docs/TRAINING_CONV_COMPARE.zh.md)
- JSON 快照：`docs/training_snapshots/`
- 离线验证集 RMSE：`scripts/eval_val_metrics.py`

---

## 4. 仿真翻方块（`sim_acp/`）

仓内 Python MuJoCo 联调（方案 B），不修改 `robot-control-v1.5`。  
手册：[`sim_acp/README.md`](sim_acp/README.md) · 设计：[`docs/MUJOCO_ACP_SIM_MVP.zh.md`](docs/MUJOCO_ACP_SIM_MVP.zh.md)

### 递进版本

| 版本 | 含义 | 状态 |
|------|------|------|
| **v1** | 脚本专家 + tip，无 RGB | ✅ ~93° |
| **v2** | Spec 三模态零样本（验链路） | ✅ 链路通；域差下不稳翻 |
| **v2-ft** | 仿真微调后策略可翻 | ✅ 可翻，真 RGB |

依赖：`conda activate pyrite`、i7 MJCF（`ACP_I7_MODEL_ROOT`）、微调权重。

### 启动

```bash
conda activate pyrite
source scripts/setup_env.sh
# export ACP_I7_MODEL_ROOT=/path/to/robot-control-v1.5/model_new
# export DISPLAY=:1

bash scripts/run_sim_flip.sh              # 默认 v2-ft + MuJoCo 窗口 + 循环
bash scripts/run_sim_flip.sh v1           # 脚本专家
bash scripts/run_sim_flip.sh v2           # Spec 零样本
bash scripts/run_sim_flip.sh v2-ft-once   # 单轮
bash scripts/run_sim_flip.sh v2-ft-live   # 可选：同权重 Live 分屏
bash scripts/run_sim_flip.sh help
```

验收：v2-ft 同场景 max tilt ≥55°（腕部 `acp_wrist_cam`）；不承诺真机零样本。  
权重路径：`ACP_SIM_FT_CKPT`。重录媒体：`bash scripts/make_github_media.sh`。

---

## 5. 真机适配

阶段三未开始；仿真侧 `RobotBackend` 契约已冻结，有机后实现 `RealBackend`。

1. 方案：[`docs/I7_ACP_ADAPTATION.zh.md`](docs/I7_ACP_ADAPTATION.zh.md) — `RealBackend` + `force_admittance`
2. 约束：右臂 + 锁底盘 + flip；方案 A；与 `sim_acp` 共用接口契约

---

## 目录结构

```text
ACP-repo/
├── README.md
├── config/
├── demo/                            # 阶段一 Demo
├── sim_acp/                         # MuJoCo 单臂翻方块（方案 B）
│   ├── README.md
│   ├── run_flip_cube_demo.py
│   ├── bridge/
│   ├── data/
│   ├── scripts/
│   └── outputs/
├── docs/
│   ├── HANDOVER.zh.md
│   ├── REPRODUCTION.zh.md
│   ├── I7_ACP_ADAPTATION.zh.md
│   ├── MUJOCO_ACP_SIM_MVP.zh.md
│   ├── ACP_DATA_FLOW.zh.md
│   ├── ACP_ALGORITHM_GUIDE.zh.md
│   ├── TRAINING_RESULTS_SPEC.zh.md
│   ├── TRAINING_CONV_COMPARE.zh.md
│   ├── media/
│   ├── figures/
│   └── training_snapshots/
├── scripts/
│   ├── setup_env.sh / setup_pyrite_env_cuda.sh
│   ├── run_demo.sh / run_sim_flip.sh / make_github_media.sh
│   ├── train_acp.sh / train_resume.sh / ...
│   └── train_progress_server.py
├── logs/
└── adaptive_compliance_policy/      # 官方 ACP + 本仓复现改动
```

脚本在 `scripts/`，文档在 `docs/`，仿真联调在 `sim_acp/`。  
上游对照：`bash scripts/clone_upstream.sh`。

---

## 文档索引

| 文档 | 用途 |
|------|------|
| [`docs/HANDOVER.zh.md`](docs/HANDOVER.zh.md) | 项目状态、路径、续训、待办 |
| [`docs/REPRODUCTION.zh.md`](docs/REPRODUCTION.zh.md) | Demo → 训练步骤 |
| [`docs/ACP_DATA_FLOW.zh.md`](docs/ACP_DATA_FLOW.zh.md) | 数据流与代码阅读顺序 |
| [`docs/ACP_ALGORITHM_GUIDE.zh.md`](docs/ACP_ALGORITHM_GUIDE.zh.md) | 算法说明 |
| [`docs/TRAINING_RESULTS_SPEC.zh.md`](docs/TRAINING_RESULTS_SPEC.zh.md) | Spec 权重与指标 |
| [`docs/TRAINING_CONV_COMPARE.zh.md`](docs/TRAINING_CONV_COMPARE.zh.md) | Conv 对照实验 |
| [`docs/media/README.md`](docs/media/README.md) | 效果展示与读图 |
| [`sim_acp/README.md`](sim_acp/README.md) | 仿真操作手册 |
| [`docs/MUJOCO_ACP_SIM_MVP.zh.md`](docs/MUJOCO_ACP_SIM_MVP.zh.md) | 仿真架构与验收 |
| [`docs/I7_ACP_ADAPTATION.zh.md`](docs/I7_ACP_ADAPTATION.zh.md) | 真机适配方案 |

---

## 同步

```bash
bash scripts/sync_to_github.sh "commit message"
```

---

## License

本仓库自定义脚本与文档用于 ACP 复现与工程落地。官方代码遵循其仓库 [MIT License](https://github.com/yifan-hou/adaptive_compliance_policy/blob/main/LICENSE)。
