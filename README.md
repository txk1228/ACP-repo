# ACP-repo

[Adaptive Compliance Policy (ACP)](https://arxiv.org/abs/2410.09309) 论文复现工作区 — 个人复现笔记、Demo 与 i7 Pro 适配方案。

- 论文主页：[adaptive-compliance.github.io](https://adaptive-compliance.github.io/)
- 官方代码：[yifan-hou/adaptive_compliance_policy](https://github.com/yifan-hou/adaptive_compliance_policy)

## 学习导图

- `docs/ACP_ALGORITHM_GUIDE.zh.md`：ACP 算法完整讲解（考核答辩版）

## 当前进度

| 阶段 | 状态 | 说明 |
|------|------|------|
| 阶段一 Demo | ✅ | 虚拟目标 + 变刚度可视化，无需 GPU |
| 阶段二 训练 | ⏸ | 待 RTX 4090，暂不下载数据集 |
| 阶段三 真机 | 📋 | i7 Pro 适配方案已文档化 |

## 快速开始

```bash
# 1. 克隆本仓库
git clone https://github.com/txk1228/ACP-repo.git
cd ACP-repo

# 2. 克隆 ACP 官方代码（子目录）
bash scripts/clone_upstream.sh

# 3. 创建 Demo 环境并运行
bash setup_demo_env.sh
conda activate acp-demo
bash run_demo.sh
```

## 目录结构

```
ACP-repo/
├── demo/                    # 核心算法独立 Demo
├── scripts/                 # 辅助脚本
├── REPRODUCTION.zh.md       # 完整复现指南
├── I7_ACP_ADAPTATION.zh.md  # 至简 i7 Pro 适配方案
├── setup_demo_env.sh        # 轻量 Demo 环境
├── run_demo.sh              # 一键运行 Demo
└── setup_pyrite_env.sh      # 训练环境（GPU 阶段使用）
```

## 同步更新

本地修改后推送到 GitHub：

```bash
bash scripts/sync_to_github.sh "commit message"
```

## License

本仓库中的自定义脚本与文档为个人学习复现用途。ACP 官方代码遵循其原仓库 [MIT License](https://github.com/yifan-hou/adaptive_compliance_policy/blob/main/LICENSE)。
