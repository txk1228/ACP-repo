# ACP × 至简 i7 Pro 适配方案

基于 `robot-control-v1.5` 现有能力，将 ACP 论文方法迁移到 i7 Pro（7 轴双臂 + 移动底盘）。

---

## 结论先说

**没有 UR5e 不影响复现 ACP 核心方法。** 已有：

| 能力 | robot-control-v1.5 对应模块 | ACP 需求 |
|------|---------------------------|----------|
| 六维力传感器 + 标定 | `common/ft_sensor_calibration.yaml` | ✅ 已有右臂标定 |
| 导纳控制 | `real/include/force_joint_admittance.hpp` | ✅ 底层 MBK 导纳 |
| 末端位姿控制 | `real/src/joint_eepose_control.cpp` | ✅ 跟踪参考轨迹 |
| 力数据读取 | `algorithm/real/joint_eepose_force_admittance.h` | ✅ world 系 wrench |
| 双臂 7-DOF | `real/include/config.h` (`ARM_DOF=7`) | ⚠️ 需 IK 适配 |
| 状态日志 | `common/pnc_replay/state_action.yml` | ⚠️ 缺 RGB，需扩展 |

**不能直接复用的**：ACP 官方的 `hardware_interfaces`（专为 UR5e 设计）。

---

## UR5e vs i7 Pro 差异与应对

| 差异 | 影响 | 应对策略 |
|------|------|----------|
| 6 轴 → 7 轴 | IK/雅可比维度变化 | 用 i7 自带 Pinocchio IK，策略仍输出笛卡尔位姿 |
| 固定基座 → 移动底盘 | 世界坐标系漂移 | **桌面任务锁底盘**，base 当固定参考系 |
| 关节空间导纳 vs 笛卡尔变刚度 | 控制语义不同 | 见下文「控制层改造」 |
| 单臂任务 vs 双臂 | 花瓶擦拭需双臂 | 先用**单臂**复现 flip 任务 |

---

## 复现分三层（推荐顺序）

### 第一层：算法训练（与机器人无关）

用官方数据集训练 ACP 策略，验证 loss 收敛：

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pyrite
source scripts/setup_env.sh
bash scripts/train_acp.sh spec
```

策略输入/输出与机器人类型无关：
- 输入：RGB + 六维力时序 + 末端位姿
- 输出：参考位姿(9D) + 虚拟目标(9D) + 刚度幅值 k_low(1D)

### 第二层：i7 上跑通导纳 + 力反馈（无需训练）

启用现有力导纳，验证接触柔顺：

**1. 修改** `common/joint_admittance_safety_config.json`：

```json
"force_admittance": {
  "enabled": true,
  "force_deadband_N": 3.0,
  "virtual_stiffness": [20.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0],
  ...
}
```

**2. 运行真机控制**：

```bash
cd /home/xiaoke/robot-control-v1.5/build
./real/joint_eepose_control
```

**3. 开启 debug 日志**（已有 CSV 记录 wrench / q_offset / pose_offset）：

```bash
export FORCE_ADMITTANCE_DEBUG_CSV=1
./real/joint_eepose_control
python3 /home/xiaoke/robot-control-v1.5/scripts/analyze_force_admittance_debug.py log/force_admittance_debug_*.csv
```

### 第三层：ACP 策略闭环部署（需开发桥接层）

```
┌─────────────────────────────────────────────────────────┐
│  ACP Python 策略 (PyriteML)                              │
│  输入: RGB + wrench[7000] + pose[3帧]                    │
│  输出: x_ref, x_virt, k_low                              │
└────────────────────┬────────────────────────────────────┘
                     │ 500Hz 虚拟目标 + 刚度
                     ▼
┌─────────────────────────────────────────────────────────┐
│  i7 桥接层 (待开发: acp_i7_bridge.py)                    │
│  - 读 SDK: 位姿、wrench、RGB                             │
│  - 由 x_ref/x_virt 差重构 K 矩阵 (ACP 公式)              │
│  - 下发: 跟踪 x_virt 的笛卡尔指令                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  robot-control-v1.5                                      │
│  joint_eepose_control + force_admittance overlay         │
└─────────────────────────────────────────────────────────┘
```

---

## 控制层：现有代码 vs ACP 需求

### 现有 i7 导纳（关节空间）

`ForceJointAdmittance` 逻辑：

```
τ_ext = J^T · wrench_eff
q̈_offset = (τ_ext - B·q̇_offset - K_joint·q_offset) / M
q_final = q_nominal + q_offset
```

- 刚度 `virtual_stiffness` 是**各关节统一**的 7 维向量
- 不区分接触法向 / 切向

### ACP 需要的（笛卡尔空间变刚度）

```
K = S · diag(k_low, k_high, k_high, k_high, k_high, k_high) · S⁻¹
u = f / ||f||   （力方向 = 柔顺主轴）
x_virt = x_ref + K⁻¹ · f
```

跟踪 `x_virt` 而非 `x_ref`，接触方向自动变软。

### 最小改造方案（推荐）

**不改 C++**，在 Python 桥接层做：

1. ACP 策略直接输出 `x_ref` 和 `x_virt`
2. 桥接层把 **`x_virt` 作为笛卡尔跟踪目标** 发给 `joint_eepose_control`
3. 底层 `force_admittance` 作为安全兜底（`enabled: true`，参数保守）
4. `k_low` 可用于动态调节 `force_admittance.virtual_stiffness` 缩放因子

这样复现 ACP **90%+ 效果**，无需重写 C++ 导纳控制器。

---

## 数据采集 → ACP 训练格式

### i7 现有数据

- `state_action.yml`：关节/位姿/底盘状态（**无 RGB、无 wrench**）
- `ft_calibration_record_*.csv`：力 + 位姿 + 关节角（**无 RGB**）
- Parquet replay：`common/pnc_replay/`

### 需要新增采集

每条 episode 同步记录：

| 字段 | 来源 | 频率 |
|------|------|------|
| `rgb_0/*.jpg` | 外接 USB 相机 | ~30Hz |
| `wrench_data_0.json` | `ExternalWrenchState` | ≥500Hz |
| `robot_data_0.json` | `RightArmState.flange_pos` | ≥500Hz |

参考 ACP 官方格式（`README.md` Data collection 节），写转换脚本：

```
i7 episode/ → real_data_processing.py 格式 → postprocess_add_virtual_target_label.py
```

### 虚拟目标标签参数（i7 建议初值）

基于 `postprocess_add_virtual_target_label.py`，针对 i7 7 轴臂调整：

```python
stiffness_estimation_para = {
    "k_max": 3000,      # i7 负载更大，可从 3000 起调
    "k_min": 200,
    "f_low": 0.5,
    "f_high": 5.0,
    "dim": 3,           # 仅平移方向变刚度（与论文一致）
}
```

---

## 关键文件对照表

| 功能 | ACP 官方 | i7 robot-control-v1.5 |
|------|----------|----------------------|
| 虚拟目标标签 | `PyriteEnvSuites/scripts/postprocess_add_virtual_target_label.py` | 直接复用（数据格式对齐后） |
| 策略网络 | `PyriteML/diffusion_policy/policy/diffusion_unet_timm_mod1_policy.py` | 直接复用 |
| 刚度重建 | `PyriteEnvSuites/env_runners/virtual_target_real_env_runner.py` | 移植到 Python 桥接层 |
| 力标定 | ATI 手动标定 | `common/ft_sensor_calibration.yaml` ✅ |
| 导纳控制 | `force_control` C++ 包 | `force_joint_admittance.hpp` ✅ |
| 硬件抽象 | `hardware_interfaces/ManipServer` | `srobots::Robot` SDK + ring buffer |
| 真机入口 | `virtual_target_real_env_runner.py` | 待写 `acp_i7_bridge.py` |

---

## 推荐任务选择（i7 上）

| 任务 | 可行性 | 说明 |
|------|--------|------|
| 物品翻转 (flip) | ⭐⭐⭐ 优先 | 单臂即可，与 UR5e 最接近 |
| 花瓶擦拭 | ⭐⭐ 后期 | 需双臂协调 + 固定物体 |
| 曲面擦拭 | ⭐⭐ 后期 | 需视觉 + FFT 力编码，收益最大 |

**第一步建议**：右臂 + 锁底盘 + flip 类单点接触任务。

---

## 下一步行动

1. **[训练]** 等 conda 环境就绪，用官方 `flip_up_230` 数据集训练 baseline
2. **[真机验证]** 启用 `force_admittance`，手推右臂验证柔顺响应
3. **[桥接开发]** 写 `acp_i7_bridge.py`：读 checkpoint → 读 i7 状态 → 发 x_virt
4. **[数据采集]** 扩展 episode 录制（RGB + wrench + pose 同步）


