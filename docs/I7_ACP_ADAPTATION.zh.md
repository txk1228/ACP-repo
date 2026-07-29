# ACP × 至简 i7 Pro 适配方案

基于 `robot-control-v1.5`（本机 `/home/zj/robot-control-v1.5`，`main` @ `6e39948b`），把 ACP 迁移到 i7 Pro（7 轴双臂 + 移动底盘）。

> i7 侧结论已对照 v1.5 源码核实（引用形如 `文件:行`）。ACP 侧代码在本仓库 `adaptive_compliance_policy/`。

---

## 文档导读

| 顺序 | 内容 | 跳转 |
|------|------|------|
| **1** | 结论与能力对照 | [→ §1](#1-结论与能力对照) |
| **2** | 推荐主线 | [→ §2](#2-推荐主线) |
| **2b** | MuJoCo 联调仿真 MVP（当前优先） | [`MUJOCO_ACP_SIM_MVP.zh.md`](MUJOCO_ACP_SIM_MVP.zh.md) |
| **3** | 控制栈分层 | [→ §3](#3-控制栈分层往哪接) |
| **4** | 硬件差异与应对 | [→ §4](#4-ur5e-vs-i7-差异与应对) |
| **5** | 三层落地步骤 | [→ §5](#5-三层落地推荐顺序) |
| **6** | 变刚度方案 | [→ §6](#6-变刚度三种落地方案) |
| **7** | 桥接接口 | [→ §7](#7-桥接接口蓝图-acp_i7_bridgepy) |
| **8** | 数据采集格式 | [→ §8](#8-数据采集--acp-训练格式) |
| **9** | 文件对照 / 风险 / 里程碑 | [→ §9](#9-关键文件对照) · [§10](#10-风险与缺口) · [§11](#11-里程碑清单) |

附录：[v1.5 关键更新速查](#附录-v15-关键更新速查)

---

## 1. 结论与能力对照

**没有 UR5e 不影响复现 ACP 核心。** v1.5 已具备：补偿后的世界系六维力、笛卡尔位姿跟踪、关节空间力导纳兜底，以及任务空间 3D 导纳骨架（`AdmittanceController3D`）。

| ACP 需求 | v1.5 能力 | 就绪度 |
|----------|-----------|--------|
| 世界系六维力（已补偿） | 服务端补偿 → RT 包 `tcp_wrench[6]`（`rt_wire.h:54`；读 `joint_eepose_force_admittance.h:446`） | ✅ 直接用 |
| 力标定 | 双臂已标定 `enabled: true`（`ft_sensor_calibration.yaml`；右臂力 RMSE≈0.24N） | ✅ |
| 安全柔顺兜底 | 关节空间 `force_admittance`（`force_joint_admittance.hpp` + `joint_admittance_safety_config.json`） | ✅ 默认关，需开 |
| 笛卡尔位姿跟踪 | NRT 笛卡尔 + SE3 限速 + IK，1kHz（`joint_eepose_control.cpp`） | ✅ |
| 策略下发 | NRT 环 / SDK `Robot.moveNRT(Nrt*ArmCartesianWorld)` | ⚠️ 缺 Python 笛卡尔示例，需先核实 |
| 状态回读 | `RobotStateRaw` 环或 SDK `getLatestState()`（`robot.h:703`） | ✅ |
| 笛卡尔变刚度 | `AdmittanceController3D`（`admittance_controller_3d.{h,cpp}`） | ⚠️ K 构造期固定，要运行时可调需加 setter |
| 训练数据日志 | `state_action.yml` 遥操作格式 | ⚠️ **缺 RGB、缺 wrench** |

**不能直接复用**：ACP 官方 `hardware_interfaces`（UR5e/ATI/ROS）。只需自写胶水桥接，不重写底层控制器。

---

## 2. 推荐主线

> **当前（真机受限时）**：按 [`MUJOCO_ACP_SIM_MVP.zh.md`](MUJOCO_ACP_SIM_MVP.zh.md) 推进 **单臂翻方块**（脚本 + 方案 A，再接 Flip Spec），冻结与真机同构的 `RobotBackend`。  
> 注入力 demo 为方案 A 单元测试；主线为翻方块。  
> **有真机窗口后**：右臂 + 锁底盘 + flip + **方案 A** → 替换为 `RealBackend`。

| 阶段 | 内容 | 状态 |
|------|------|------|
| 算法侧 | Spec/Conv 训练 | ✅ 见 `TRAINING_RESULTS_SPEC.zh.md` / `TRAINING_CONV_COMPARE.zh.md` |
| **仿真：翻方块** | `sim_acp/run_flip_cube_demo`（脚本+方案 A / `--policy`） | ✅ 脚本可翻；策略在环已接 |
| 真机准备 | 力标定 + `force_admittance` 手推 | ⏳ 有机后 |
| 真机闭环 | RealBackend + 同一 bridge | ⏳ 仿真契约冻结后 |
| 保真升级 | 方案 B：`AdmittanceController3D` 运行时刚度 | 按需 |

---

## 3. 控制栈分层（往哪接）

```
ACP Python 策略 (~10–15Hz)
  入: RGB + wrench 历史(FFT) + pose
  出: x_ref(9D) + x_virt(9D) + k_low(1D)
        │
        ▼  桥接插值 → 200–500Hz 流式下发（本文重点）
acp_i7_bridge.py（待写）
  READ : RobotStateRaw → RtPacket.{tcp_pos, tcp_wrench}
  WRITE: NRT 笛卡尔 → Left/RightArmRpcCmdNrt
         （或 SDK Robot.moveNRT）
        │
        ▼  NRT 环  pose=[x,y,z,qx,qy,qz,qw]
joint_eepose_control (1kHz)
  parseLoopCommandFrame → BoundedSe3RefGenerator(≤1m/s)
  → IK → (可选) force_admittance 叠加 q_offset
  → Left/RightArmRtCmd
        │
        ▼
simple-robot-server（电机 + 六维力补偿 + 世界系 wrench）
```

**三条硬约束**（已核实）：

1. 外部下发**只认 NRT 环**；`joint_eepose_control` 不读 RT 环、不读 planner chunk 环。
2. 控制器内部直接读写 SHM；外部 Python **优先 SDK** `moveNRT()` / `getLatestState()`。
3. SpaceMouse（`test_remote3dmouse.cpp`）走关节 RT，**不是**本适配路径。

---

## 4. UR5e vs i7 差异与应对

| 差异 | 应对 |
|------|------|
| 6 轴 → 7 轴 | 策略只出笛卡尔位姿；冗余自由度交给 i7 解析/迭代 IK |
| 固定基座 → 移动底盘 | **桌面任务锁底盘**，base 当固定系 |
| 关节导纳 vs 笛卡尔变刚度 | 见 [§6](#6-变刚度三种落地方案)；首选方案 A |
| 单臂 vs 双臂 | 先右臂 flip；双臂 vase 靠后 |
| ATI/ROS 硬件抽象 | 用 i7 SDK/环替代，只写胶水层 |

---

## 5. 三层落地（推荐顺序）

### 第一层 · 算法训练（与机器人无关）✅ 已完成

策略 I/O 与机型无关：

- 输入：RGB + 六维力时序 + 末端位姿  
- 输出：`x_ref`(9D) + `x_virt`(9D) + `k_low`(1D)（9D = 平移 3 + rot6d）

结果归档：`docs/TRAINING_RESULTS_SPEC.zh.md`。

### 第二层 · 真机导纳 + 力反馈（无需策略）

目的：确认力读数、柔顺方向、安全兜底。

**Step A — 力标定**（换负载后重标）：

```bash
cd /home/zj/robot-control-v1.5
sudo /opt/srobot/apps/rc/tools/ft_calibration/ft_calibrate.sh --dry-run
sudo /opt/srobot/apps/rc/tools/ft_calibration/ft_calibrate.sh --side right
```

**Step B — 开启关节空间力导纳**（`common/joint_admittance_safety_config.json` → `force_admittance`）：

```jsonc
"force_admittance": {
  "enabled": true,
  "force_deadband_N": 50.0,   // 初调保守，验证后可下调
  "max_force_N": 150.0,
  "wrench_lpf_tau_s": 0.05,
  "virtual_mass":      [10,10,10,10,10,10,10],
  "virtual_damping":   [28,28,28,28,28,28,28],
  "virtual_stiffness": [20,40,40,40,40,40,40],
  "wrench_weight": [0.1, 0.5, 0.1, 0.0, 0.0, 0.0]  // 只用平移力
}
```

> 注意：这是 **7 维关节刚度**，不是笛卡尔刚度；只作安全 overlay。

**Step C — 手推验证 + debug CSV**：

```bash
cd /home/zj/robot-control-v1.5/build
export FORCE_ADMITTANCE_DEBUG_CSV=1
./real/joint_eepose_control
python3 ../scripts/analyze_force_admittance_debug.py force_admittance_debug_*.csv
```

若“越推越顶”，翻转 `force_to_motion_sign`。

### 第三层 · 策略闭环（需桥接）

控制方案见 [§6](#6-变刚度三种落地方案)，接口细节见 [§7](#7-桥接接口蓝图-acp_i7_bridgepy)。

---

## 6. 变刚度三种落地方案

ACP 论文公式：

```
u = f / ||f||                                      # 力方向 = 最柔顺轴
K = S · diag(k_low, k_high, …) · S⁻¹
x_virt = x_ref + K⁻¹ · f
```

| 方案 | 做法 | 改 C++？ | 保真度 | 适用 |
|------|------|----------|--------|------|
| **A（首选）** | 桥接用实测 `f` 算 `x_virt`，NRT 跟踪；底层 `force_admittance` 兜底 | 否 | 中（准静态够用） | flip 首通 |
| **B（高保真）** | 复用 `AdmittanceController3D`，加 `setStiffness`，周期注入 `k_low` | 少量 | 高 | A 跑通后升级 |
| **C（备选）** | `NRT_JOINT_CARTESIAN_MOTION_FORCE` 下发力设定 | 否 | 力控≠变刚度 | 恒力擦拭类 |

**落地建议**：A 打通闭环 → 按需上 B；C 不作为 flip 主路径。

---

## 7. 桥接接口蓝图（`acp_i7_bridge.py`）

### 7.1 读：pose + wrench + q

同机优先 SHM 环（延迟最低）：

| 量 | 来源 | 字段 |
|----|------|------|
| 末端位姿 | `RobotStateRaw` → `RtPacket.{left,right}` | `tcp_pos[7]` = `[x,y,z,qx,qy,qz,qw]` |
| 世界系六维力（已补偿） | 同上 | `tcp_wrench[6]` = `[fx,fy,fz,mx,my,mz]` |
| 关节状态 | `LeftArmState` / `RightArmState` | `libcom::RobotState` |
| 备选 wrench | `ExternalWrenchState`（`/ext_wrench_state`） | left/right `wrench[6]` |

远程/省事：`Robot.getLatestState()` → `eepose` / `ee_wrench` / `q`。  
读环参考：`real/src/ft_calibration_recorder.cpp:240`。

> **勿再自补偿**：FT 偏置/重力已由服务端 `six_dof_force` 处理（见附录）。

### 7.2 写：NRT 笛卡尔目标

| 项 | 约定 |
|----|------|
| 位姿 | `pose[7]=[x,y,z,qx,qy,qz,qw]`，世界系 |
| 模式 | `Mode::NRT_JOINT_CARTESIAN_MOTION` |
| 时间戳 | `timestamp_ns` **每帧必须变化**（NRT 按时间戳去重） |
| SDK | `Robot.moveNRT(NrtLeftArmCartesianWorld{pose})`（`motion_types.h:543`） |
| 直接写环 | `LeftArmRpcCmdNrt` / `RightArmRpcCmdNrt`（字段填法见 `joint_eepose_control_smoke.cpp:81`） |

消费端：`joint_eepose_control.cpp:1369` `parseLoopCommandFrame` → `switchToCartesianNrtMode`。

### 7.3 频率与缓冲

| 环节 | 频率 / 要求 |
|------|-------------|
| 控制环 | 1kHz；NRT SE3 在目标间插值 |
| 策略推理 | ~10–15Hz，短 horizon |
| 桥接下发 | 将 horizon **插值成 200–500Hz** NRT 流，避免跳变触发限速滞后 |
| wrench 历史 | 服务端 ≈1kHz；桥接维护环形缓冲（FFT 窗约 7000 点） |

### 7.4 安全（三层都要）

1. 底层：`force_admittance.enabled=true`  
2. 控制：NRT SE3 限速（≈1 m/s）+ 碰撞检测独立线程  
3. 桥接：NaN 校验、下发超时看门狗、`‖x_virt − x_ref‖` 上限钳制  

（`AdmittanceController3D` 自身还有 ±0.08m / ±0.05 m/s 限幅，方案 B 时生效。）

---

## 8. 数据采集 → ACP 训练格式

### 现状

`common/pnc_replay/state_action.yml`：双臂位姿 / 关节 / 电流 / 底盘等遥操作字段齐全，但 **无 RGB、无 wrench**。

### 需新增同步采集

| 字段 | 来源 | 频率 |
|------|------|------|
| `rgb_0/*.jpg` | 外接 USB 相机 | ~30Hz |
| `wrench_data_0.json` | `tcp_wrench` 或 `ExternalWrenchState` | ≈1kHz |
| `robot_data_0.json` | `tcp_pos` 或 `flange_pos` | ≥500Hz |

转换流水线：

```
i7 episode/ → real_data_processing.py → postprocess_add_virtual_target_label.py
```

### 虚拟目标标签初值（i7）

```python
stiffness_estimation_para = {
    "k_max": 3000,  # i7 负载更大，可从 3000 起调
    "k_min": 200,
    "f_low": 0.5,
    "f_high": 5.0,
    "dim": 3,       # 仅平移变刚度（与论文一致）
}
```

---

## 9. 关键文件对照

| 功能 | ACP 官方 | i7 v1.5 |
|------|----------|---------|
| 策略网络 | `diffusion_unet_timm_mod1_policy.py` | 直接复用 |
| 虚拟目标标签 | `postprocess_add_virtual_target_label.py` | 数据对齐后复用 |
| 刚度 / 虚拟目标运行时 | `virtual_target_real_env_runner.py` | 方案 A→桥接；方案 B→`AdmittanceController3D` |
| 世界系六维力 | ATI + 手动补偿 | `tcp_wrench`（服务端已补偿） |
| 力标定 | 手动 | `ft_sensor_calibration.yaml` |
| 安全导纳 | `force_control` | `force_joint_admittance.hpp` |
| 硬件抽象 | `hardware_interfaces/ManipServer` | SDK `Robot` + SHM ring |
| 真机入口 | `virtual_target_real_env_runner.py` | 待写 `acp_i7_bridge.py` |

---

## 10. 风险与缺口

| 优先级 | 缺口 | 建议 |
|--------|------|------|
| P0 | Python 无笛卡尔 NRT / 读环示例（仅有关节 `move_arm_position`） | **先核实**部署 SDK 是否有 `moveNRT`/`getLatestState`；否则补 ctypes 或 C++ 小桥 |
| P1 | NRT 无逐帧刚度字段 | 方案 A 规避；真变刚度走方案 B |
| P1 | 日志缺 RGB / wrench | 采集端扩展同步录制 |
| P2 | 策略 10Hz vs 控制 1kHz | 桥接做 horizon→高频插值 |
| P2 | 底盘漂移 | 桌面任务锁底盘 |

---

## 11. 里程碑清单

### 11a. 仿真优先 → 详见 [`MUJOCO_ACP_SIM_MVP.zh.md`](MUJOCO_ACP_SIM_MVP.zh.md)

- [ ] **SM0** 跑通现有 `sim/admittance`（或等价）
- [ ] **SM1** SimBackend：读 pose/wrench/q + 写笛卡尔跟踪
- [ ] **SM2** 力注入 + 方案 A（无网络）演示虚拟目标偏移
- [ ] **SM3–SM4** 接入 Flip Spec + 一键闭环脚本（MVP Done）

### 11b. 真机（仿真契约冻结后）

- [ ] **M0** 核实部署机 SDK Python：`moveNRT(笛卡尔)` + `getLatestState`
- [ ] **M1** 开启 `force_admittance`，手推右臂，力方向正确
- [ ] **M2–M3** RealBackend 替换 SimBackend，小幅度联调
- [ ] **M4** episode 同步录制 → 格式转换
- [ ] **M5**（可选）方案 B 运行时刚度

**任务优先级**：MuJoCo 联调 ≫ 真机 flip（右臂）≫ 双臂 vase。

---

## 附录 · v1.5 关键更新速查

| 更新 | 对适配的意义 |
|------|--------------|
| 移除 robot-control 内 FT 补偿（`9bf12d97`） | 桥接**直接读**已补偿世界系 wrench，勿再补偿 |
| 0.2.7 末端六维力导纳 | `force_admittance` 可作底层兜底 |
| 0.2.7 QP / 碰撞独立线程 | 跟踪更稳，利于闭环安全 |
| 0.2.8 Thor 一键标定 | `ft_calibrate.sh --side both` |
| 0.2.4.1「外部写入参考构型」 | ⚠️ 仅内部 IK 关节种子，**不是**外部位姿 API |
| `AdmittanceController3D` | 任务空间变刚度骨架（方案 B） |

控制栈与 ACP-repo 分离；部署机路径以实际为准（本机示例：`/home/zj/robot-control-v1.5`）。
