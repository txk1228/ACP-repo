# ACP × MuJoCo 联调仿真方案（MVP）

> **已定案**：**方案 B** — 仅在 ACP 仓内用 Python `mujoco` 跑仿真与桥接，**不改** `robot-control-v1.5`。  
> **仿真主目标**：**单臂翻方块** — 从脚本专家到 **Flip Spec 策略在仿真内可翻**（tilt ≥55°）。  
> **非目标**：论文级真机成功率、Isaac、改 robot-control C++。

关联：[`I7_ACP_ADAPTATION.zh.md`](I7_ACP_ADAPTATION.zh.md)（真机；`RobotBackend` 契约一致，日后只换 RealBackend）。  
操作手册 / 一键启动：[`sim_acp/README.md`](../sim_acp/README.md) · `bash scripts/run_sim_flip.sh`

---

## 0. 导读

| 顺序 | 内容 | 跳转 |
|------|------|------|
| **1** | 为什么做、成功标准 | [→ §1](#1-为什么做--什么叫做成) |
| **2** | MVP 范围 | [→ §2](#2-mvp-范围已定案-b) |
| **3** | 架构与接口 | [→ §3](#3-架构与接口) |
| **4** | 我们完成了什么（v1→v2-ft） | [→ §4](#4-我们完成了什么v1v2-ft) |
| **5** | 分阶段里程碑 | [→ §5](#5-分阶段里程碑) |
| **6** | 目录与依赖 | [→ §6](#6-目录与依赖) |
| **7** | 风险与验收 | [→ §7](#7-风险与验收清单) |

---

## 1. 为什么做 / 什么叫「做成」

### 1.1 价值

| 有价值的产出 | 说明 |
|--------------|------|
| **单臂翻方块可演示** | v1 脚本专家；v2-ft **策略** 在同场景翻起 |
| **ACP 柔顺可验收** | 方案 A 力调度；`run_acp_effect_demo` |
| **策略在环联调** | Flip Spec 三模态 → 开环 action horizon 执行 |
| **桥接契约冻结** | `RobotBackend` 与真机同构 → 真机只换 IO |
| **仿真→策略数据闭环** | 专家 zarr → VT 标注 → 短训微调 |

| 暂不承诺 | 原因 |
|----------|------|
| 真机零样本翻成功率 | GoPro/UR5e/点指 vs i7+腕部相机 |
| 仿真数字孪生 1:1 | 联调验证公式与接口，非动力学复刻 |

### 1.2 成功标准（当前 Done）

| ID | 标准 | 状态 |
|----|------|------|
| S1–S3 | 状态读写 / 跟踪 / Flip Spec 前向 | ✅ |
| S4 | 方案 A 力对齐单元验收 | ✅ |
| S5 | **脚本翻方块** tilt ≥55° | ✅ v1 ~93° |
| S6 | 三模态闭环有位移（零样本） | ✅ v2 |
| **S7** | **微调策略在仿真翻方块** tilt ≥55°，真 RGB | ✅ v2-ft 5/5 ~70° |

---

## 2. MVP 范围（已定案 B）

### 2.1 做

| 项 | 选择 |
|----|------|
| 落地 | Python `mujoco` + 全在 ACP 仓 |
| 模型 | 本机 symlink **i7 完整 MJCF**（`ACP_I7_MODEL_ROOT`）；精简 `ee_plane.xml` 保留作 smoke |
| 臂 | **仅右臂**；底盘锁死 |
| 控制 | v1 刚性 tip IK；v2-ft 策略 `x_virt` 开环 horizon |
| 策略 | 复用真机 Flip Spec ckpt；**仿真数据短训** |
| 力 | mesh 接触力 → wrench；与真机 ATI 频谱不同，微调必须用仿真力 |
| RGB | **`acp_wrist_cam` 腕部相机** 224×224（非固定外置机位） |

### 2.2 不做

- 改 `robot-control-v1.5` C++  
- Isaac、双臂 vase 仿真成功率、从零重训整网  
- 真机 SDK（预留 `RealBackend`）

### 2.3 与真机关系

```
真机 Flip Spec ckpt ──微调──► 仿真 v2-ft ckpt
         │                           │
         │ 域差（RGB/力/几何）         │ 同场景 PASS
         ▼                           ▼
   零样本 v2 链路通              策略可翻（本阶段目标）
         │
         └──────► 日后 RealBackend + 真机数据
```

---

## 3. 架构与接口

```
┌──────────────────────────────────────────────────────────┐
│  sim_acp/                                                 │
│                                                           │
│  Flip Spec (pyrite)  ← ckpt；RGB + wrench + pose 历史      │
│       │                                                   │
│       ▼  action horizon (x_ref, x_virt, k) ×16          │
│  run_flip_cube_demo --policy                              │
│    开环 exec_h=12, action_ds=50 → write_tip_pos           │
│       │                                                   │
│       ▼                                                   │
│  I7MujocoBackend (RobotBackend)                           │
│    i7 IK · mesh 接触力 · acp_wrist_cam render_rgb         │
│    acp_tip_ball 主接触 · mj_step · viewer                 │
└──────────────────────────────────────────────────────────┘
```

### 3.1 `RobotBackend` 契约

```python
class RobotBackend(Protocol):
    def read_state(self) -> RobotState:   # pose7, wrench6, q, ts
    def write_ee_pose(...) / write_tip_pos(...)
    def step(self) -> bool
    def inject_force_W(self, force_xyz)   # 仿真专用
    def render_rgb(self) -> (H,W,3)      # acp_wrist_cam
    def reset_episode(self, cube_xy=...)  # 循环演示
```

### 3.2 方案 A（柔顺公式）

```
û = f / |f|
K = R · diag(k_low, k_high, k_high) · Rᵀ
x_virt = x_ref + f / k_low   （软轴沿力；正交方向保精度）
```

v1 翻物阶段用刚性 tip 跟踪以保证翻起；力轨迹仍入库供 VT 标注。

### 3.3 三模态与 Flip Spec

| 模态 | 仿真来源 | shape_meta 对齐 |
|------|----------|-----------------|
| RGB | `acp_wrist_cam` 224×224 | horizon=2, ds=10 |
| pose | tip 位置 + 法兰姿态 → pose9 | horizon=3, ds=5 |
| wrench | 接触力工具系，7× 上采样 | horizon=7000, ds=1 |
| action | x_ref + x_virt + k（19 维） | horizon=16, ds=50 |

---

## 4. 我们完成了什么（v1→v2-ft）

### 4.1 场景与接触（`i7_scene.py`）

- 桌 + 自由方块 + i7 右臂（本机 MJCF）  
- **`acp_tip_ball`**：夹爪 tip 球，主碰撞体，避免大 mesh 先蹭桌  
- **`acp_wrist_cam`**：腕部 RGB，替代早期固定外置 `acp_cam`  
- 曾尝试 fixture 墙；方块易滑，改回 **自由方块 + tip** 后 v1 稳定 PASS  

### 4.2 v1 脚本专家

- 航点：接近 → 下棱接触 → 扫过翻起  
- **刚性 tip 跟踪**（非 Scheme A 偏移），max tilt ~93°  
- 入口：`python -m sim_acp.run_flip_cube_demo`

### 4.3 v2 三模态闭环

- `FlipSpecPolicyRunner`：RGB + wrench + pose → `x_ref / x_virt / k_low`  
- 真机 ckpt **零样本**：推理通、有位移，**不翻**（域差预期）  
- 入口：`--policy`

### 4.4 仿真数据 + 微调（v2-ft 主杠杆）

```mermaid
flowchart LR
  v1[v1 脚本专家] --> rec[record_flip_episodes]
  rec --> zarr[flip_up_sim_v1 zarr]
  zarr --> lab[label_virtual_target]
  lab --> ft[finetune_flip_spec]
  ft --> eval["--require-flip tilt≥55°"]
```

| 步骤 | 脚本 | 产出 |
|------|------|------|
| 采集 | `record_flip_episodes.py` | ≥50 成功 ep；RGB/wrench/pose |
| 标注 | `label_virtual_target.py` | `ts_pose_virtual_target_0`, `stiffness_0` |
| 微调 | `finetune_flip_spec.py` | 真机 ckpt +30 epoch |
| 评测 | `--require-flip` | 5/5 PASS，~70° |

数据集：`$PYRITE_DATASET_FOLDERS/flip_up_sim_v1`  
微调 ckpt：`~/training_outputs/2026.07.24_16.16.52_flip_up_sim_flip_sim_ft/checkpoints/latest.ckpt`

### 4.5 演示增强

- **`--loop`**：循环 reset + 重跑，Ctrl+C 退出  
- **`--render`**：MuJoCo viewer（`DISPLAY=:1`）  
- RGB 落盘：`sim_acp/outputs/wrist_rgb_ft/`

---

## 5. 分阶段里程碑

| ID | 内容 | 验收 |
|----|------|------|
| M0 | mujoco 加载 / step | ✅ |
| M1 | Backend 跟踪 | ✅ |
| M2 | 方案 A + 注入力 demo | ✅ |
| M3 | Flip Spec 前向 | ✅ |
| M4 | 策略 → x_virt 一键脚本 | ✅ |
| M5 | i7 场景 + 接触力 + 真 RGB | ✅ |
| **M6** | **专家数据 + 微调 → 策略翻方块** | ✅ **v2-ft** |
| M7 | 真机 RealBackend | 📋 后续 |

---

## 6. 目录与依赖

```text
ACP-repo/
├── docs/MUJOCO_ACP_SIM_MVP.zh.md    # 本文
├── sim_acp/
│   ├── README.md                    # 操作手册
│   ├── run_flip_cube_demo.py        # v1 / v2 / v2-ft / --loop
│   ├── bridge/
│   │   ├── i7_mujoco_backend.py
│   │   ├── i7_scene.py              # tip + wrist_cam
│   │   ├── policy_runner.py
│   │   └── virtual_target.py
│   ├── data/label_virtual_target.py
│   └── scripts/
│       ├── record_flip_episodes.py
│       └── finetune_flip_spec.py
└── adaptive_compliance_policy/      # Flip Spec 训练配置（只读加载 + 微调 yaml）
```

### 环境

```bash
conda activate pyrite
source scripts/setup_env.sh
export ACP_I7_MODEL_ROOT=/path/to/robot-control-v1.5/model_new
pip install mujoco   # 已在 pyrite 环境
```

| 变量 | 含义 |
|------|------|
| `ACP_I7_MODEL_ROOT` | i7 MJCF + meshes |
| `PYRITE_DATASET_FOLDERS` | 数据集根目录 |
| `PYRITE_CHECKPOINT_FOLDERS` | 训练输出 / ckpt |

真机预训练：`~/training_outputs/2026.07.17_14.42.42_flip_up_new_resnet_230/checkpoints/latest.ckpt`

---

## 7. 风险与验收清单

| 风险 | 缓解 |
|------|------|
| 仿真 wrench ≠ ATI 7kHz | 微调必须用仿真力分布 |
| 腕部相机 ≠ GoPro | 接受；本阶段以仿真 PASS 为准 |
| i7 夹爪 ≠ 论文点指 | tip 球 + 专家轨迹可学 |
| 零样本不翻 | 文档明确；走 v2-ft |
| viewer 退出 SIGSEGV | `--render` 结束 `os._exit`；Renderer 有序 close |

### 验收（已全部勾选）

- [x] 仅 ACP 仓可运行（无 robot-control 编译）  
- [x] 方案 A / ACP 效果 demo  
- [x] **v1 脚本翻方块** tilt ≥55°  
- [x] **v2 三模态闭环**（零样本链路）  
- [x] **v2-ft 策略翻方块**（微调 + 腕部 RGB + `--require-flip`）  
- [x] 专家数据 ≥50 ep + VT 标注 + 短训 pipeline  

### 口径

> **v1**：脚本刚性专家 + tip，无 RGB。  
> **v2**：腕部 RGB + wrench + pose → Flip Spec → 开环 horizon；零样本不宣称翻成功。  
> **v2-ft**：仿真演示微调后 **tilt ≥55°**（当前 ~70°，5/5 复现）。  
> **真机**：另开里程碑，见 `I7_ACP_ADAPTATION.zh.md`。

---

## 8. 下一步

1. ~~M0–M6 仿真翻物~~ → 已完成  
2. 真机 `RealBackend` + 腕部/外置相机对齐  
3. 可选：更多 episode、更长微调、fixture 几何再探索  
