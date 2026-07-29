# ACP 论文复现指南

> **阶段划分**：先完成 Demo 环境与核心算法可视化；完整训练需 GPU 与官方数据集。  
> 当前仓库已含 Spec/Conv 训练结果与仿真翻方块联调（见主 README）。

论文：**Adaptive Compliance Policy: Learning Approximate Compliance for Diffusion Guided Control** (ICRA 2025)

- 项目主页：https://adaptive-compliance.github.io/
- 论文：https://arxiv.org/abs/2410.09309
- 官方代码：`adaptive_compliance_policy/`（已克隆）

---

## 阶段一：Demo（无需 GPU）

### 1. 创建 Demo 环境（~2 分钟，无需 GPU）

```bash
bash scripts/setup_demo_env.sh
conda activate acp-demo
```

依赖：`numpy scipy matplotlib spatialmath-python cvxpy plotly`（无 PyTorch）

### 2. 运行核心算法 Demo

```bash
bash scripts/run_demo.sh
# 或无 GUI 环境：
bash scripts/run_demo.sh --no-show
```

输出：
- `demo/output/acp_virtual_target_stiffness_demo.png` — 四联图
- `demo/output/demo_summary.txt` — 数值摘要

Demo 演示 ACP 核心公式：
- `u = f / ||f||`（柔顺主轴 = 力方向）
- `K = S · diag(k_low, k_high, k_high) · S⁻¹`（空间可变刚度）
- `x_virt = x_ref + K⁻¹ · f`（虚拟目标）

源码直接调用官方 `VirtualTargetEstimator`（`compliance_helpers.py`）。

### 3. i7 Pro 适配说明

见 [`I7_ACP_ADAPTATION.zh.md`](I7_ACP_ADAPTATION.zh.md)

---

## 阶段二：完整训练（待 RTX 4090）

等 GPU 工作站就绪后，再执行以下步骤。


## 复现路径选择

| 路径 | 需要什么 | 能复现什么 |
|------|----------|------------|
| **A. 算法理解** | 仅阅读代码 | 核心方法：虚拟目标、刚度矩阵、扩散策略 |
| **B. 训练复现** | GPU + conda + 官方数据集 | 训练 ACP 模型，对比 loss / checkpoint |
| **C. 真机部署** | UR5e + 六维力传感器 + 相机 + C++ 控制器 | 完整论文实验（翻转 / 擦拭） |

---

## 当前环境状态

- 代码：已克隆至 `adaptive_compliance_policy/`
- GPU：当前机器未检测到 NVIDIA GPU（训练需 CUDA）
- conda/mamba：未安装（官方推荐 mamba）

---

## 路径 A：核心算法（无需硬件）

### 1. 示教标签：虚拟目标 + 刚度

```bash
# 文件位置
adaptive_compliance_policy/PyriteEnvSuites/scripts/postprocess_add_virtual_target_label.py
adaptive_compliance_policy/PyriteUtility/planning_control/compliance_helpers.py
```

核心公式：
- 力方向单位向量：`u = f / ||f||`
- 虚拟目标：`x_virt = x_ref + K^{-1} f`
- 刚度矩阵：`K = S @ diag(k_low, k_high, ...) @ S^{-1}`

### 2. 策略网络

```bash
adaptive_compliance_policy/PyriteML/diffusion_policy/policy/diffusion_unet_timm_mod1_policy.py
adaptive_compliance_policy/PyriteML/diffusion_policy/model/vision/timm_obs_encoder_with_force_spec.py  # FFT 力编码
```

输入：2 帧 RGB + 力时序（FFT 频谱或 TCN）+ 3 帧位姿  
输出：参考位姿(9D) + 虚拟目标(9D) + 刚度幅值 k_low(1D) = 19 维

### 3. 推理时刚度重建

```bash
adaptive_compliance_policy/PyriteEnvSuites/env_runners/virtual_target_real_env_runner.py
```

---

## 路径 B：训练复现

### Step 1：安装 mamba（Ubuntu 22.04）

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
source ~/miniforge3/bin/activate
```

### Step 2：创建环境

```bash
cd adaptive_compliance_policy
mamba env create -f conda_environment.yaml
mamba activate pyrite
pip install v4l2py toppra atomics vit-pytorch imagecodecs
```

### Step 3：环境变量（写入 ~/.bashrc）

```bash
export PYRITE_RAW_DATASET_FOLDERS=$HOME/data/real
export PYRITE_DATASET_FOLDERS=$HOME/data/real_processed
export PYRITE_CHECKPOINT_FOLDERS=$HOME/training_outputs
export PYRITE_HARDWARE_CONFIG_FOLDERS=$HOME/config  # 真机才需要
export PYRITE_CONTROL_LOG_FOLDERS=$HOME/data/control_log
```

### Step 4：下载官方数据集

```bash
mkdir -p ~/data/real_processed && cd ~/data/real_processed

# 物品翻转（230 条示教）
wget https://real.stanford.edu/adaptive-compliance/data/flip_up_230.zip
unzip flip_up_230.zip

# 花瓶擦拭（200 条示教）
wget https://real.stanford.edu/adaptive-compliance/data/vase_wiping_200.zip
unzip vase_wiping_200.zip

# 预训练 checkpoint（可选，用于评估）
wget https://real.stanford.edu/adaptive-compliance/checkpoints.zip
unzip checkpoints.zip
```

### Step 5：训练

```bash
cd adaptive_compliance_policy/PyriteML
accelerate config   # 首次配置

# ACP（FFT 力编码）— 翻转任务
HYDRA_FULL_ERROR=1 accelerate launch train.py --config-name=train_spec_workspace

# ACP w/o FFT（时序卷积）— 消融对比
HYDRA_FULL_ERROR=1 accelerate launch train.py --config-name=train_conv_workspace
```

切换任务：编辑 `train_spec_workspace.yaml` 中 `defaults` 的 task：
- `flip_up_spec.yaml` — 物品翻转
- `vase_wiping_spec.yaml` — 花瓶擦拭

训练配置要点：batch_size=128，300 epochs，需 CUDA GPU。

---

## 路径 C：真机部署

### 硬件要求（与论文一致）

- 双 UR5e 协作机械臂（擦拭任务）或单臂（翻转任务）
- ATI Mini45 六维力传感器
- GoPro / RGB 相机
- 力觉示教手柄（数据采集）

### C++ 控制器依赖

```bash
git clone https://github.com/yifan-hou/cpplibrary.git
git clone https://github.com/yifan-hou/force_control.git
git clone https://github.com/yifan-hou/hardware_interfaces.git
# 按各 repo README 编译安装
```

导纳控制器频率 ≥ 500 Hz。

### 数据采集 → 后处理 → 训练 → 评估

1. **采集**：`hardware_interfaces/applications/manipulation_data_collection`
2. **转 zarr**：`PyriteUtility/data_pipeline/real_data_processing.py`
3. **加标签**：`postprocess_add_virtual_target_label.py`
4. **评估**：`virtual_target_real_env_runner.py`

---

## 论文关键超参数（标签生成）

来自 `postprocess_add_virtual_target_label.py`：

| 参数 | 值 | 含义 |
|------|-----|------|
| k_max | 5000 | 无接触时最大刚度 (N/m) |
| k_min | 200 | 强接触时最小刚度 |
| f_low | 0.5 | 刚度开始下降的力阈值 (N) |
| f_high | 5 | 刚度降至最低的力阈值 (N) |

---

## 预期实验结果（论文 Table）

**物品翻转**（总平均成功率）：
- ACP: 96% | ACP w/o FFT: 95% | 固定柔顺: 23% | 刚性: 14%

**花瓶擦拭**：
- ACP: 93.75% | ACP w/o FFT: 81.25% | 固定柔顺: 43.75%

---

## 下一步（按阶段）

### 阶段一 ✅ 已完成
- [x] 轻量 Demo 环境 `acp-demo`（`scripts/setup_demo_env.sh`）
- [x] 核心算法 Demo：虚拟目标 + 刚度可视化（`scripts/run_demo.sh`）

### 阶段二：完整训练（需 GPU）
1. **配置训练环境**（在 GPU 机器上运行，无需 mamba，miniconda 即可）：
   ```bash
   bash scripts/setup_pyrite_env.sh   # 创建 pyrite 环境 + PyTorch
   source scripts/setup_env.sh
   ```
2. **下载数据集并训练**（约 10 GB）：
   ```bash
   mkdir -p ~/data/real_processed && cd ~/data/real_processed
   wget https://real.stanford.edu/adaptive-compliance/data/flip_up_230.zip
   unzip flip_up_230.zip
   bash scripts/train_acp.sh spec
   ```

### 阶段三（可与阶段二并行）
- [ ] i7 Pro 真机：启用 `force_admittance`，验证柔顺响应（见 `I7_ACP_ADAPTATION.zh.md`）
- [ ] 编写 `acp_i7_bridge.py` 桥接层，部署训练好的 checkpoint
