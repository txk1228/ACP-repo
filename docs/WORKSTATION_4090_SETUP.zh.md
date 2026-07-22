# RTX 4090 工作站配置清单（从开箱到能训练）

> 适用：全新工作站 + 外接显示器 + 准备复现 ACP  
> 目标：`nvidia-smi` 能看到 4090 → PyTorch 能用 CUDA → 能启动 ACP 训练

---

## 0. 先确认：你在哪台机器上操作？

| 检查项 | 旧 ThinkPad | 新 4090 工作站 |
|--------|-------------|----------------|
| 主机名 | `xiaoke-ThinkPad-L14-Gen-4` | 通常是品牌机名 / 你自定义名 |
| GPU | 无 / 核显 | RTX 4090 |
| 内存 | ~16GB | 通常 ≥32GB |

在终端执行：

```bash
hostname
nvidia-smi
```

- 若提示 `找不到命令 nvidia-smi`，且主机名还是 ThinkPad → **你还在旧笔记本上**，请改在新工作站上操作。
- Cursor 也要在新工作站上打开工程，或 Remote-SSH 连到新工作站。

---

## 1. 硬件与显示器（开箱当天）

1. **电源**：确认工作站电源线插好，机箱电源开关打开。
2. **显卡供电**：4090 通常需要独立供电线（16-pin / 转接），确认已接牢。
3. **显示器**：
   - 优先插 **显卡背面接口**（DisplayPort / HDMI），不要插主板后面的核显口。
   - 若黑屏：换 DP 线、换接口、或进 BIOS 看是否默认核显输出。
4. **网络**：插网线或连 Wi‑Fi，后面要下驱动、PyTorch、数据集。

进系统后先确认显示器分辨率正常、能上网。

---

## 2. 系统与 NVIDIA 驱动（最关键）

推荐：**Ubuntu 22.04 LTS**（与 ACP 官方环境一致）。

### 2.1 检查是否已装驱动

```bash
nvidia-smi
```

若能看到类似：

```text
NVIDIA-SMI ...    Driver Version: 550.xx    CUDA Version: 12.x
...
GeForce RTX 4090
```

→ 驱动 OK，跳到第 3 节。

### 2.2 未装驱动时（Ubuntu）

```bash
# 更新
sudo apt update && sudo apt upgrade -y

# 推荐用 Ubuntu 自带驱动管理（稳）
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall

# 或指定较新驱动（示例，以 devices 输出为准）
# sudo apt install -y nvidia-driver-550

sudo reboot
```

重启后再跑：

```bash
nvidia-smi
```

**验收标准：** 能看到 `RTX 4090`，无报错。

---

## 3. 开发基础软件

```bash
sudo apt update
sudo apt install -y git curl wget build-essential
```

安装 Miniconda（若新机没有）：

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 安装结束后重新开终端
conda config --set auto_activate_base false
```

---

## 4. 拉取 ACP 代码

```bash
# 建议工作目录
mkdir -p ~/work && cd ~/work
git clone git@github.com:txk1228/ACP-repo.git
# 或 HTTPS：
# git clone https://github.com/txk1228/ACP-repo.git

cd ACP-repo
bash scripts/clone_upstream.sh
```

若官方代码已在仓库里，可跳过 `clone_upstream.sh`。

---

## 5. 创建 CUDA 版训练环境（4090 专用）

旧笔记本上的 `setup_pyrite_env.sh` 装的是 **CPU 版 PyTorch**，新机要用 **CUDA 版**。

### 5.1 一键创建（推荐）

在新工作站上执行（脚本见同目录 `setup_pyrite_env_cuda.sh`）：

```bash
cd ~/work/ACP-repo   # 或你的实际路径
bash scripts/setup_pyrite_env_cuda.sh
```

### 5.2 手动创建（等价步骤）

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n pyrite python=3.10 pip -y
conda activate pyrite
export PYTHONNOUSERSITE=1

# PyTorch CUDA 12.1（4090 常用稳定组合）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ACP 依赖
pip install \
  huggingface_hub wandb timm diffusers accelerate \
  threadpoolctl plotly dill einops hydra-core \
  ipykernel ipython matplotlib omegaconf opencv-python \
  pandas pyyaml scipy tqdm zarr spatialmath-python \
  numcodecs scikit-video scikit-fda \
  v4l2py toppra atomics vit-pytorch imagecodecs cvxpy
```

### 5.3 验收 GPU

```bash
conda activate pyrite
export PYTHONNOUSERSITE=1
python - <<'EOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "N/A")
EOF
```

**验收标准：**

```text
cuda available: True
device: NVIDIA GeForce RTX 4090
```

若 `False`：先修驱动/`nvidia-smi`，再重装对应 CUDA 的 PyTorch，不要继续训练。

---

## 6. 环境变量与目录

写入 `~/.bashrc`（路径按你新机习惯改）：

```bash
export PYRITE_RAW_DATASET_FOLDERS=$HOME/data/real
export PYRITE_DATASET_FOLDERS=$HOME/data/real_processed
export PYRITE_CHECKPOINT_FOLDERS=$HOME/training_outputs
export PYRITE_HARDWARE_CONFIG_FOLDERS=$HOME/config
export PYRITE_CONTROL_LOG_FOLDERS=$HOME/data/control_log
export PYTHONNOUSERSITE=1
```

然后：

```bash
mkdir -p ~/data/real ~/data/real_processed ~/training_outputs ~/config ~/data/control_log
source ~/.bashrc
```

---

## 7. 下载数据集（磁盘要够）

`flip_up_230.zip` 约 **9GB+**，解压后更大。建议磁盘剩余 **≥50GB**。

```bash
mkdir -p ~/data/real_processed && cd ~/data/real_processed
wget https://real.stanford.edu/adaptive-compliance/data/flip_up_230.zip
unzip flip_up_230.zip
ls
```

先只下翻转任务即可；擦拭任务后续再下。

---

## 8. 首次训练配置与冒烟

```bash
conda activate pyrite
cd ~/work/ACP-repo/adaptive_compliance_policy/PyriteML

# 首次交互配置 accelerate（单卡 4090：选 This machine / No distributed / GPU）
accelerate config

# 或用仓库里的非交互配置（若已有 accelerate_config.yaml）
# accelerate launch --config_file ../../config/accelerate_config.yaml ...
```

冒烟训练（短跑，确认能进 GPU）：

```bash
cd ~/work/ACP-repo
bash scripts/train_acp_smoke.sh
# 或正式训练：
# bash scripts/train_acp.sh spec
```

---

## 9. Cursor / VS Code 在新机上怎么配

1. 在新工作站安装 Cursor，打开 `ACP-repo` 文件夹。  
2. 选择解释器：`~/miniconda3/envs/pyrite/bin/python`。  
3. 终端里确认：

```bash
hostname
nvidia-smi
which python   # 应指向 pyrite
```

若你仍用旧笔记本远程开发：用 **Remote-SSH** 连到新工作站，不要在旧机本地训练。

---

## 10. 推荐当天完成顺序（打勾）

- [ ] 新机开机，显示器接在 **显卡口**，能进桌面
- [ ] `nvidia-smi` 看到 RTX 4090
- [ ] 安装 git / conda
- [ ] 克隆 `ACP-repo` + 上游代码
- [ ] 创建 `pyrite` 环境，`torch.cuda.is_available() == True`
- [ ] 配置数据目录环境变量
- [ ] 下载 `flip_up_230`（可过夜）
- [ ] `accelerate config` + smoke 训练跑通

---

## 常见问题

| 现象 | 处理 |
|------|------|
| 黑屏 / 无信号 | 改插显卡 DP/HDMI；确认显卡供电；必要时接主板口进系统再装驱动 |
| `nvidia-smi` 找不到 | 安装驱动并 reboot |
| `cuda available: False` | 驱动与 PyTorch CUDA 版本不匹配；用 `cu121` wheel 重装 |
| conda 很慢 | 换清华/中科大镜像，或用 mamba |
| 旧机误训练 | 看 `hostname` / `nvidia-smi`，确认在新工作站 |

---

## 和旧笔记本的关系

| 机器 | 用途 |
|------|------|
| ThinkPad | 看文档、改代码、跑无 GPU Demo |
| 4090 工作站 | 下数据集、训练、以后推理服务 |

**下一步：** 在新工作站打开终端，把下面两行输出发我，我可以按你的实际状态继续往下配（驱动 / 环境 / 数据）：

```bash
hostname
nvidia-smi
```
