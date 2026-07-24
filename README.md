# ACP-repo

[Adaptive Compliance Policy (ACP)](https://arxiv.org/abs/2410.09309) 论文复现工作区 — Demo、训练、**MuJoCo 仿真翻方块（`sim_acp/`）** 与 i7 Pro 适配笔记。

- 论文主页：[adaptive-compliance.github.io](https://adaptive-compliance.github.io/)
- 官方代码：[yifan-hou/adaptive_compliance_policy](https://github.com/yifan-hou/adaptive_compliance_policy)

---

## 上手路线（建议按顺序）

> 第一次接手，请**从上往下**依次点开，每一步都能直接跳转到本页对应小节：

| 顺序 | 做什么 | 跳转 |
|------|--------|------|
| **Step 0** | 先读交接总览，了解现状与主线 | [→ 交接总览](#step-0-先读交接总览) |
| **Step 1** | 搭训练 / Demo 环境 | [→ 环境搭建](#step-1-环境搭建) |
| **Step 2** | 跑通 Demo（阶段一，无需 GPU） | [→ 跑通-demo](#step-2-跑通-demo) |
| **Step 3** | 训练：冒烟 → 完整（阶段二） | [→ 训练](#step-3-训练) |
| **Step 4** | 用进度面板监控训练 | [→ 进度面板](#step-4-进度面板) |
| **Step 5** | 断点续训（断电 / 中断后） | [→ 断点续训](#step-5-断点续训) |
| **Step 6** | 看结果与 Spec/Conv 对比 | [→ 结果与对比](#step-6-结果与对比) |
| **Step 7** | **MuJoCo 仿真翻方块**（`sim_acp/`，当前优先） | [→ 仿真翻方块](#step-7-仿真翻方块-sim_acp) |
| **Step 8** | 真机适配（阶段三，未开始） | [→ 真机适配](#step-8-真机适配) |

其余参考：[目录结构](#目录结构) ·  [文档导图](#文档导图) ·  [同步更新](#同步更新) ·  [License](#license)

### 当前进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| 阶段一 Demo | ✅ | 虚拟目标 + 变刚度可视化 |
| 阶段二 Spec 训练 | ✅ | Flip-up + Vase wiping，300 epoch，见 [`docs/TRAINING_RESULTS_SPEC.zh.md`](docs/TRAINING_RESULTS_SPEC.zh.md) |
| 阶段二 Conv 对比 | 🔄 | Flip Conv ✅；Vase Conv 训练中，见 [`docs/TRAINING_CONV_COMPARE.zh.md`](docs/TRAINING_CONV_COMPARE.zh.md) |
| **仿真 `sim_acp/`** | ✅ | **策略在仿真翻方块 PASS**（腕部 RGB + 微调）；[`sim_acp/README.md`](sim_acp/README.md) |
| 阶段三 真机 | 📋 | 真机未开始；方案见 [`docs/I7_ACP_ADAPTATION.zh.md`](docs/I7_ACP_ADAPTATION.zh.md) |

---

## Step 0 先读交接总览

接手第一件事：通读 [`docs/HANDOVER.zh.md`](docs/HANDOVER.zh.md)（交接总览：进度、路径、续训、真机待办）。

**建议主线**：Vase Conv 到 300 → Spec/Conv 对照 → **`sim_acp/` 仿真翻方块**（已通）→ 有机后按 [`docs/I7_ACP_ADAPTATION.zh.md`](docs/I7_ACP_ADAPTATION.zh.md) 做真机。

---

## Step 1 环境搭建

先克隆仓库（已包含官方 ACP 代码副本 + 本地复现改动）：

```bash
git clone https://github.com/txk1228/ACP-repo.git
cd ACP-repo
```

训练环境（GPU / 4090 / CUDA，仅首次）：

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

`setup_env.sh` 关键环境变量：`PYRITE_DATASET_FOLDERS=~/data/real_processed`、`PYRITE_CHECKPOINT_FOLDERS=~/training_outputs`。

---

## Step 2 跑通 Demo

阶段一：虚拟目标 + 变刚度可视化，无需 GPU，用来先建立直觉。

```bash
conda activate acp-demo
bash scripts/run_demo.sh
```

代码入口：`demo/virtual_target_stiffness_demo.py`，输出在 `demo/output/`。

---

## Step 3 训练

阶段二。先冒烟短跑验证环境，再跑完整训练：

```bash
conda activate pyrite
source scripts/setup_env.sh

bash scripts/train_acp_smoke.sh      # 1. 短跑冒烟，验证环境/数据
bash scripts/train_acp.sh spec       # 2. 完整 Spec 训练（论文主方法 / FFT）
# bash scripts/train_vase.sh         #    双臂 vase Spec 训练
```

- 训练默认每 **10** 个 epoch 写入 `~/training_outputs/<run_dir>/checkpoints/latest.ckpt`。
- Conv 对比（消融：ACP w/o FFT）的 yaml 与启动方式见 [`docs/TRAINING_CONV_COMPARE.zh.md`](docs/TRAINING_CONV_COMPARE.zh.md)。

---

## Step 4 进度面板

代码：`scripts/train_progress_server.py`（纯标准库 HTTP 服务，读 `~/training_outputs/*/logs.json.txt` 与 `logs/train_*.log`）。

```bash
bash scripts/run_progress_panel.sh   # 一键启动（后台）
# 浏览器打开 http://127.0.0.1:8765
```

| 操作 | 命令 |
|------|------|
| 启动 | `bash scripts/run_progress_panel.sh` |
| 打开页面 | 浏览器访问 http://127.0.0.1:8765 |
| 前台调试 | `python3 scripts/train_progress_server.py` |
| 停止 | `fuser -k 8765/tcp` |

面板展示 Spec / Conv 的 Flip、Vase 进度、loss 与物理 RMSE；训练日志需落在 `logs/`（如 `logs/train_vase_wiping_conv.log`）。

---

## Step 5 断点续训

> 断电 / 中断后接着跑。

中断后**不要**再开一条全新命令从头训；应对**同一个 run 目录**设 `training.resume=true`，从 `latest.ckpt` 恢复 epoch / 优化器状态。

### 流程

```bash
# 1. 找到被中断的那次 run（按时间或面板里的路径）
ls -lt ~/training_outputs | head

# 2. 确认有权重
ls ~/training_outputs/<run_dir>/checkpoints/latest.ckpt

# 3. 续训（通用入口）
bash scripts/train_resume.sh <run_dir> <config-name> [与当初相同的 hydra 覆盖项...]
```

### 常用示例

```bash
# vase Spec（正式基线；也可不传路径，脚本有默认目录）
bash scripts/train_vase_resume.sh
# 或显式指定：
bash scripts/train_vase_resume.sh ~/training_outputs/2026.07.18_10.23.05_vase_wiping_resnet_230

# vase Conv 对比（把 run_dir 换成你中断的那次）
bash scripts/train_resume.sh \
  ~/training_outputs/2026.07.22_14.38.26_vase_wiping_conv_compare_230 \
  train_conv_compare_vase_workspace \
  training.eval_metrics_max_batches=20

# flip Spec
bash scripts/train_resume.sh \
  ~/training_outputs/2026.07.17_14.42.42_flip_up_new_resnet_230 \
  train_spec_workspace \
  task=flip_up_spec

# flip Conv 对比
bash scripts/train_resume.sh \
  ~/training_outputs/2026.07.22_10.58.33_flip_up_new_conv_compare_230 \
  train_conv_compare_flip_workspace
```

### 注意

| 要点 | 说明 |
|------|------|
| 必须指向**原 run 目录** | `hydra.run.dir=该目录`，否则会新建输出目录等于重开 |
| 覆盖项尽量与初训一致 | 如 vase 的 `batch_size=32`、`num_workers=2` |
| 尚无 `latest.ckpt` | 说明还没存过盘（&lt;10 epoch），无法续，只能重开 |
| 已跑满 300 epoch | 续训会直接结束，无需再跑 |
| 进度面板 | 续训日志继续写原 `logs.json.txt`，面板可接着看 |

---

## Step 6 结果与对比

- Spec 终盘结果（权重路径、loss、RMSE）：[`docs/TRAINING_RESULTS_SPEC.zh.md`](docs/TRAINING_RESULTS_SPEC.zh.md)
- Conv 对比实验（yaml、启动命令、终盘对照）：[`docs/TRAINING_CONV_COMPARE.zh.md`](docs/TRAINING_CONV_COMPARE.zh.md)
- 离线补跑验证集 RMSE：`scripts/eval_val_metrics.py`

Vase Conv 训完后：确认 epoch 299 + `latest.ckpt` → 用 `eval_*_val_metrics.json` 做 Spec/Conv 终盘对照 → 把数字补进 `TRAINING_CONV_COMPARE.zh.md`。

---

## Step 7 仿真翻方块（`sim_acp/`）

> **仓内 Python MuJoCo 联调**（方案 B）：不改 `robot-control-v1.5`。  
> 目录：[`sim_acp/`](sim_acp/) · 手册：[`sim_acp/README.md`](sim_acp/README.md) · 设计：[`docs/MUJOCO_ACP_SIM_MVP.zh.md`](docs/MUJOCO_ACP_SIM_MVP.zh.md)

### 做了什么

| 版本 | 含义 | 状态 |
|------|------|------|
| **v1** | 脚本专家 + tip 球，无 RGB，翻方块 | ✅ ~93° |
| **v2** | Flip Spec 三模态（腕部 RGB + wrench + pose） | ✅ 链路通；零样本不翻（域差） |
| **v2-ft** | 仿真专家数据微调后，**策略可翻** | ✅ ~70°，真 RGB |

依赖：`conda activate pyrite`、本机 i7 MJCF（`ACP_I7_MODEL_ROOT`）、Flip Spec 权重。

### 一键启动

```bash
conda activate pyrite
source scripts/setup_env.sh
# 可选：export ACP_I7_MODEL_ROOT=/path/to/robot-control-v1.5/model_new
# 可选：export DISPLAY=:1

bash scripts/run_sim_flip.sh              # ★ 默认：v2-ft + 窗口 + 循环
bash scripts/run_sim_flip.sh v1           # 脚本专家
bash scripts/run_sim_flip.sh v2           # 真机 ckpt 零样本（验链路）
bash scripts/run_sim_flip.sh v2-ft-once   # 微调策略单轮
bash scripts/run_sim_flip.sh help
```

`sim_acp/` 内含：`run_flip_cube_demo.py`（主入口）、`bridge/`（i7 后端 / tip / 腕部相机）、`scripts/record_*` + `finetune_*`（数据→微调）、`data/label_virtual_target.py`。

验收：**v2-ft** 同场景 max tilt ≥55°（腕部 `acp_wrist_cam`）；不承诺真机零样本。

---

## Step 8 真机适配

> 阶段三：真机未开始；仿真链路与契约已在 Step 7 冻结，有机后换 `RealBackend`。

1. **真机方案**：[`docs/I7_ACP_ADAPTATION.zh.md`](docs/I7_ACP_ADAPTATION.zh.md) — `RealBackend` + `force_admittance`  
2. 要点：右臂 + 锁底盘 + flip；方案 A；与 `sim_acp` 共用 `RobotBackend` 契约  

---

## 目录结构

```text
ACP-repo/
├── README.md
├── .gitignore
│
├── config/                          # 本仓库配置
│   ├── accelerate_config.yaml
│   └── conda_environment_demo.yaml
│
├── demo/                            # 核心算法独立 Demo（无需 GPU）
│   ├── virtual_target_stiffness_demo.py
│   └── output/
│
├── sim_acp/                         # ★ MuJoCo 单臂翻方块联调（方案 B）
│   ├── README.md                    # 仿真操作手册
│   ├── run_flip_cube_demo.py        # v1 / v2 / v2-ft / --loop
│   ├── run_acp_effect_demo.py       # 方案 A 注入力 demo
│   ├── bridge/                      # RobotBackend、i7 IK、tip、腕部相机
│   ├── data/                        # virtual target 标注
│   ├── scripts/                     # 录制 / 微调 / smoke
│   └── outputs/                     # 曲线、腕部 RGB 落盘
│
├── docs/                            # 全部文档与归档
│   ├── HANDOVER.zh.md
│   ├── REPRODUCTION.zh.md
│   ├── I7_ACP_ADAPTATION.zh.md      # 真机适配
│   ├── MUJOCO_ACP_SIM_MVP.zh.md     # 仿真设计 / 里程碑
│   ├── ACP_DATA_FLOW.zh.md
│   ├── ACP_ALGORITHM_GUIDE.zh.md
│   ├── TRAINING_RESULTS_SPEC.zh.md
│   ├── TRAINING_CONV_COMPARE.zh.md
│   ├── figures/
│   └── training_snapshots/
│
├── scripts/                         # 环境 / 训练 / 仿真启动
│   ├── setup_env.sh
│   ├── setup_pyrite_env_cuda.sh
│   ├── run_demo.sh
│   ├── run_sim_flip.sh              # ★ 仿真一键启动（v1/v2/v2-ft）
│   ├── train_acp.sh / train_resume.sh / ...
│   ├── train_progress_server.py
│   └── sync_to_github.sh
│
├── logs/
│
└── adaptive_compliance_policy/      # 官方 ACP + 本复现改动
    ├── PyriteML/
    ├── PyriteUtility/
    ├── PyriteConfig/
    └── PyriteEnvSuites/
```

根目录保留 `README` 与上述目录：脚本走 `scripts/`，文档走 `docs/`，**仿真联调走 `sim_acp/`**。  
`adaptive_compliance_policy/` 来自 [官方仓库](https://github.com/yifan-hou/adaptive_compliance_policy)，并含本复现改动（Conv 对比 yaml、`val_metrics`、仿真微调 task yaml）。可选对照上游：`bash scripts/clone_upstream.sh`。

---

## 文档导图

点击文档名可直接打开：

| 文档 | 用途 |
|------|------|
| [`docs/HANDOVER.zh.md`](docs/HANDOVER.zh.md) | **交接总览**：进度、路径、续训、真机待办（先读） |
| [`docs/REPRODUCTION.zh.md`](docs/REPRODUCTION.zh.md) | 从 Demo 到完整训练的步骤 |
| [`docs/ACP_DATA_FLOW.zh.md`](docs/ACP_DATA_FLOW.zh.md) | 按数据流读核心代码 |
| [`docs/ACP_ALGORITHM_GUIDE.zh.md`](docs/ACP_ALGORITHM_GUIDE.zh.md) | 算法完整讲解 |
| [`docs/TRAINING_RESULTS_SPEC.zh.md`](docs/TRAINING_RESULTS_SPEC.zh.md) | Spec 权重路径、loss、RMSE |
| [`docs/TRAINING_CONV_COMPARE.zh.md`](docs/TRAINING_CONV_COMPARE.zh.md) | Conv 对比 yaml 与启动方式 |
| [`sim_acp/README.md`](sim_acp/README.md) | **仿真操作手册 + 一键启动（日常入口）** |
| [`docs/MUJOCO_ACP_SIM_MVP.zh.md`](docs/MUJOCO_ACP_SIM_MVP.zh.md) | 仿真架构、里程碑、验收口径 |
| [`docs/I7_ACP_ADAPTATION.zh.md`](docs/I7_ACP_ADAPTATION.zh.md) | 真机适配方案 |

---

## 同步更新

```bash
bash scripts/sync_to_github.sh "commit message"
```

---

## License

本仓库中的自定义脚本与文档用于 ACP 复现与工程交接。ACP 官方代码遵循其原仓库 [MIT License](https://github.com/yifan-hou/adaptive_compliance_policy/blob/main/LICENSE)。
