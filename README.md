# ACP-repo

[Adaptive Compliance Policy (ACP)](https://arxiv.org/abs/2410.09309) 论文复现工作区 — Demo、训练脚本与 i7 Pro 适配笔记。

- 论文主页：[adaptive-compliance.github.io](https://adaptive-compliance.github.io/)
- 官方代码：[yifan-hou/adaptive_compliance_policy](https://github.com/yifan-hou/adaptive_compliance_policy)

## 当前进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| 阶段一 Demo | ✅ | 虚拟目标 + 变刚度可视化 |
| 阶段二 Spec 训练 | ✅ | Flip-up + Vase wiping，300 epoch，见 `docs/TRAINING_RESULTS_SPEC.zh.md` |
| 阶段二 Conv 对比 | 📋 | 配置已就绪，见 `docs/TRAINING_CONV_COMPARE.zh.md` |
| 阶段三 真机 | 📋 | i7 Pro 适配方案见 `docs/I7_ACP_ADAPTATION.zh.md` |

## 快速开始

```bash
# 1. 克隆本仓库
git clone https://github.com/txk1228/ACP-repo.git
cd ACP-repo

# 2. 克隆 ACP 官方代码（子目录，已在 .gitignore）
bash scripts/clone_upstream.sh

# 3. Demo 环境并运行
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
│   ├── ACP_ALGORITHM_GUIDE.zh.md    # 算法讲解（答辩版）
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
│   ├── train_vase_resume.sh         # vase 断点续训
│   ├── eval_val_metrics.py          # 离线 val RMSE
│   ├── train_progress_server.py     # 训练进度面板 :8765
│   ├── refresh_train_progress_canvas.py
│   └── sync_to_github.sh            # 本地改动推送
│
├── logs/                            # 运行日志（gitignore，本地保留）
│
└── adaptive_compliance_policy/      # 官方上游（gitignore，需 clone）
    ├── PyriteML/                    # 训练入口 train.py + diffusion_policy
    ├── PyriteUtility/               # 柔顺控制 / 空间数学等
    ├── PyriteConfig/                # 任务配置
    └── PyriteEnvSuites/             # 环境与 runner
```

根目录只保留 `README` 与上述目录：脚本走 `scripts/`，文档走 `docs/`，日志进 `logs/`。

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

本仓库中的自定义脚本与文档为个人学习复现用途。ACP 官方代码遵循其原仓库 [MIT License](https://github.com/yifan-hou/adaptive_compliance_policy/blob/main/LICENSE)。
