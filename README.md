# ACP-repo

[Adaptive Compliance Policy (ACP)](https://arxiv.org/abs/2410.09309) 论文复现工作区 — Demo、训练脚本与 i7 Pro 适配笔记。

- 论文主页：[adaptive-compliance.github.io](https://adaptive-compliance.github.io/)
- 官方代码：[yifan-hou/adaptive_compliance_policy](https://github.com/yifan-hou/adaptive_compliance_policy)

## 当前进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| 阶段一 Demo | ✅ | 虚拟目标 + 变刚度可视化 |
| 阶段二 Spec 训练 | ✅ | Flip-up + Vase wiping，300 epoch，见 `docs/TRAINING_RESULTS_SPEC.zh.md` |
| 阶段二 Conv 对比 | ✅ | 配置已就绪，见 `docs/TRAINING_CONV_COMPARE.zh.md` |
| 阶段三 真机 | 📋 | i7 Pro 适配方案见 `docs/I7_ACP_ADAPTATION.zh.md` |

## 快速开始

```bash
# 1. 克隆本仓库（已包含官方 ACP 代码副本 + 本地复现改动）
git clone https://github.com/txk1228/ACP-repo.git
cd ACP-repo

# 2. Demo 环境并运行
bash scripts/setup_demo_env.sh
conda activate acp-demo
bash scripts/run_demo.sh
```

训练环境（GPU）与冒烟测试：

```bash
bash scripts/setup_pyrite_env_cuda.sh   # 4090 / CUDA
source scripts/setup_env.sh
bash scripts/train_acp_smoke.sh         # 短跑验证
# bash scripts/train_acp.sh spec        # 完整 Spec 训练
```

## 训练进度可视化面板

代码：`scripts/train_progress_server.py`（纯标准库 HTTP 服务，读 `~/training_outputs/*/logs.json.txt` 与 `logs/train_*.log`）。

```bash
# 一键启动（后台）
bash scripts/run_progress_panel.sh

# 浏览器打开
# http://127.0.0.1:8765
```

| 操作 | 命令 |
|------|------|
| 启动 | `bash scripts/run_progress_panel.sh` |
| 打开页面 | 浏览器访问 http://127.0.0.1:8765 |
| 前台调试 | `python3 scripts/train_progress_server.py` |
| 停止 | `fuser -k 8765/tcp` |

面板展示 Spec / Conv 的 Flip、Vase 进度、loss 与物理 RMSE；训练日志需落在 `logs/`（如 `logs/train_vase_wiping_conv.log`）。

## 断点续训（断电 / 中断后接着跑）

训练默认每 **10** 个 epoch 写入 `~/training_outputs/<run_dir>/checkpoints/latest.ckpt`。  
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

## 目录结构

```text
ACP-repo/
├── README.md
├── .gitignore
│
├── config/                          # 本仓库配置
│   ├── accelerate_config.yaml       # Accelerate 启动配置
│   └── conda_environment_demo.yaml  # Demo conda 环境
│
├── demo/                            # 核心算法独立 Demo（无需 GPU）
│   ├── virtual_target_stiffness_demo.py
│   └── output/                      # 可视化输出
│
├── docs/                            # 全部文档与归档
│   ├── REPRODUCTION.zh.md           # 完整复现指南
│   ├── I7_ACP_ADAPTATION.zh.md      # 至简 i7 Pro 适配
│   ├── ACP_DATA_FLOW.zh.md          # 数据流 / 学习顺序
│   ├── ACP_FEISHU_STUDY.zh.md       # 论文研读笔记
│   ├── ACP_ALGORITHM_GUIDE.zh.md    # 算法讲解
│   ├── WORKSTATION_4090_SETUP.zh.md # 4090 工作站配置
│   ├── TRAINING_RESULTS_SPEC.zh.md  # Spec 训练结果归档
│   ├── TRAINING_CONV_COMPARE.zh.md  # Conv 对比实验说明
│   ├── figures/                     # 文档配图
│   └── training_snapshots/          # 指标 JSON 快照
│
├── scripts/                         # 环境 / 训练 / 工具脚本
│   ├── clone_upstream.sh            # 克隆官方 ACP 代码
│   ├── setup_demo_env.sh            # 轻量 Demo 环境
│   ├── setup_env.sh                 # 训练用环境变量
│   ├── setup_pyrite_env.sh          # CPU 训练环境（旧）
│   ├── setup_pyrite_env_cuda.sh     # CUDA / 4090 训练环境
│   ├── run_demo.sh                  # 一键跑 Demo
│   ├── train_acp.sh                 # 通用训练入口 (spec|conv)
│   ├── train_acp_smoke.sh           # GPU 冒烟短跑
│   ├── train_vase.sh                # 双臂 vase Spec 训练
│   ├── train_vase_resume.sh         # vase Spec 断点续训（快捷）
│   ├── train_resume.sh              # 通用断点续训入口
│   ├── eval_val_metrics.py          # 离线 val RMSE
│   ├── train_progress_server.py     # 训练进度面板服务（:8765）
│   ├── run_progress_panel.sh        # 一键启动进度面板
│   ├── refresh_train_progress_canvas.py
│   └── sync_to_github.sh            # 本地改动推送
│
├── logs/                            # 运行日志（gitignore，本地保留）
│
└── adaptive_compliance_policy/      # 官方 ACP 代码（已纳入本仓库，~2MB）
    ├── PyriteML/                    # 训练入口 train.py + diffusion_policy
    ├── PyriteUtility/               # 柔顺控制 / 空间数学等
    ├── PyriteConfig/                # 任务配置
    └── PyriteEnvSuites/             # 环境与 runner
```

根目录只保留 `README` 与上述目录：脚本走 `scripts/`，文档走 `docs/`，日志进 `logs/`。  
`adaptive_compliance_policy/` 来自 [官方仓库](https://github.com/yifan-hou/adaptive_compliance_policy)，并含本复现的本地改动（如 Conv 对比 yaml、`val_metrics`）。可选对照上游：`bash scripts/clone_upstream.sh`。

## 文档导图

| 文档 | 用途 |
|------|------|
| `docs/REPRODUCTION.zh.md` | 从 Demo 到完整训练的步骤 |
| `docs/ACP_DATA_FLOW.zh.md` | 按数据流读核心代码 |
| `docs/ACP_FEISHU_STUDY.zh.md` | 网络结构 / 数据对齐 / 信息注入 |
| `docs/ACP_ALGORITHM_GUIDE.zh.md` | 算法完整讲解 |
| `docs/WORKSTATION_4090_SETUP.zh.md` | 4090 开箱到可训 |
| `docs/TRAINING_RESULTS_SPEC.zh.md` | Spec 权重路径、loss、RMSE |
| `docs/TRAINING_CONV_COMPARE.zh.md` | Conv 对比 yaml 与启动方式 |
| `docs/I7_ACP_ADAPTATION.zh.md` | 真机适配方案 |

## 同步更新

```bash
bash scripts/sync_to_github.sh "commit message"
```

## License

本仓库中的自定义脚本与文档用于 ACP 复现与工程交接。ACP 官方代码遵循其原仓库 [MIT License](https://github.com/yifan-hou/adaptive_compliance_policy/blob/main/LICENSE)。
