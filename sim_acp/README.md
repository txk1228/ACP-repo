# sim_acp — ACP MuJoCo 联调（方案 B）

**仿真主目标：单臂翻方块** — 脚本专家 → 腕部 RGB 三模态 → 仿真微调后 **策略可翻**（tilt ≥55°）。  
**不依赖、不修改** `robot-control-v1.5`（i7 MJCF 仅本机 symlink）。

| 文档 | 内容 |
|------|------|
| 本文 | 操作手册 + 启动命令 |
| [`docs/MUJOCO_ACP_SIM_MVP.zh.md`](../docs/MUJOCO_ACP_SIM_MVP.zh.md) | 架构、里程碑、验收口径 |
| [`docs/I7_ACP_ADAPTATION.zh.md`](../docs/I7_ACP_ADAPTATION.zh.md) | 真机适配（后续） |

---

## 1. 我们做了什么

| 阶段 | 内容 | 状态 |
|------|------|------|
| 链路 | `RobotBackend` + 方案 A + i7 IK | ✅ |
| 场景 | 桌 + 方块 + **tip 球** + **腕部相机** | ✅ |
| **v1** | 刚性脚本专家翻方块（无 RGB） | ✅ ~93° |
| **v2** | Flip Spec 三模态闭环（零样本） | ✅ 链路通，不翻 |
| 数据 | 50 ep 成功演示 + VT 标注 | ✅ |
| **v2-ft** | 仿真微调 → **策略可翻** | ✅ 5/5 ~70° |

**结论**：真机权重零样本翻不起来是 **域差**（GoPro/UR5e vs i7+腕部相机），不是链路坏了。主杠杆 = **仿真专家数据 + 微调**。

### 效果预览（v2-ft，分屏变刚度）

![sim flip v2-ft](../docs/media/sim_flip_v2ft.gif)

左：MuJoCo + 腕部 RGB；右：\|f\| / \|Δ\|、**k_soft vs k_hard**、soft-axis û + tilt。  
[MP4](../docs/media/sim_flip_v2ft.mp4) · [刚度曲线](../docs/media/sim_flip_v2ft_stiffness.png) · [**读图与指标**](../docs/media/README.md#终盘刚度曲线怎么读sim_flip_v2ft_stiffnesspng) · 重录：`bash scripts/make_github_media.sh`

**指标摘要**：tilt≥55°（本录屏~90°）= 翻成功；接触段 \|f\| 抬升；\(k_\text{soft}\) 相对 \(k_\text{hard}\) 更软且 soft û 随接触变化 = ACP 方向性柔顺。

---

## 2. 一键启动（推荐）

```bash
cd /path/to/ACP_fx
conda activate pyrite

# ★ 默认：v2-ft + MuJoCo 窗口 + 循环播放（Ctrl+C / 关窗退出）
bash scripts/run_sim_flip.sh

# 其它模式
bash scripts/run_sim_flip.sh v1            # 脚本专家
bash scripts/run_sim_flip.sh v1-loop       # 脚本循环
bash scripts/run_sim_flip.sh v2            # 真机 ckpt 零样本（验链路）
bash scripts/run_sim_flip.sh v2-ft         # 微调策略循环（同默认）
bash scripts/run_sim_flip.sh v2-ft-once    # 微调策略单轮
bash scripts/run_sim_flip.sh v2-ft-headless  # 无头评测 PASS/FAIL
bash scripts/run_sim_flip.sh record        # 录制专家数据
bash scripts/run_sim_flip.sh label         # VT 标注
bash scripts/run_sim_flip.sh finetune      # 短训
bash scripts/run_sim_flip.sh pipeline      # record→label→finetune
bash scripts/run_sim_flip.sh help
```

脚本会自动：`conda activate pyrite`、`source scripts/setup_env.sh`、检查 i7 MJCF / ckpt。

### 环境变量（可覆盖）

| 变量 | 默认 | 含义 |
|------|------|------|
| `ACP_I7_MODEL_ROOT` | `/home/zj/robot-control-v1.5/model_new` | i7 MJCF + meshes |
| `DISPLAY` | `:1` | MuJoCo 窗口显示 |
| `ACP_SIM_FT_CKPT` | 见下「当前权重」 | v2-ft 微调权重 |
| `ACP_SIM_REAL_CKPT` | Flip Spec 真机 latest | v2 零样本权重 |
| `ACP_SIM_DATASET` | `$PYRITE_DATASET_FOLDERS/flip_up_sim_v1` | 仿真 zarr |
| `ACP_SIM_RECORD_N` | `50` | 录制条数 |

示例：
```bash
export ACP_SIM_FT_CKPT=~/training_outputs/<新目录>/checkpoints/latest.ckpt
bash scripts/run_sim_flip.sh v2-ft
```

---

## 3. 版本分流

| 版本 | 启动 | 观测 / 控制 | 验收 |
|------|------|-------------|------|
| **v1** | `run_sim_flip.sh v1` | 无 RGB；脚本航点 + 刚性 tip | tilt ≥55° |
| **v2** | `run_sim_flip.sh v2` | 腕部 RGB + wrench + pose → 真机 Spec | 推理通 + 有位移；**不宣称翻成功** |
| **v2-ft** | `run_sim_flip.sh v2-ft` | 同上 + 微调 ckpt | **tilt ≥55°**（真 RGB） |

**v2-ft 执行**（对齐真机 runner）：每次推理开环执行 **12** 个路点，间隔 **50** 仿真步。

---

## 4. 场景与传感器

生成：`$ACP_I7_MODEL_ROOT/mjcf/_acp_i7_scene.xml`（`i7_scene.py` 自动写，可删）

| 组件 | 说明 |
|------|------|
| `acp_tip_ball` | tip 球，**主接触** |
| `acp_ee_site` | 工具点（脚本 / 策略跟踪） |
| `acp_wrist_cam` | **腕部 RGB** 224×224，随臂 |
| `acp_obj` | 自由方块 |
| wrench | mesh 接触力 → 工具系（微调用仿真力分布） |

样张：`sim_acp/outputs/wrist_rgb/`、`sim_acp/outputs/wrist_rgb_ft/`

---

## 5. 等价 Python 命令

若不用 shell 包装，等价于：

```bash
conda activate pyrite
source scripts/setup_env.sh
export DISPLAY=:1 MUJOCO_GL=glfw
export ACP_I7_MODEL_ROOT=/path/to/robot-control-v1.5/model_new

FT=~/training_outputs/2026.07.24_16.16.52_flip_up_sim_flip_sim_ft/checkpoints/latest.ckpt

# v2-ft 循环
python -m sim_acp.run_flip_cube_demo \
  --policy --require-flip --ckpt "$FT" \
  --render --loop \
  --steps 3600 --exec-horizon 12 --action-ds 50 \
  --viewer-sync-every 5 --no-plot
```

---

## 6. 数据 → 微调流水线

```bash
bash scripts/run_sim_flip.sh pipeline
# 或分步：
bash scripts/run_sim_flip.sh record
bash scripts/run_sim_flip.sh label
bash scripts/run_sim_flip.sh finetune
```

| 产物 | 路径 |
|------|------|
| 仿真数据集 | `~/data/real_processed/flip_up_sim_v1/` |
| 真机预训练（起点） | `~/training_outputs/2026.07.17_14.42.42_flip_up_new_resnet_230/checkpoints/latest.ckpt` |
| **当前 v2-ft 权重** | `~/training_outputs/2026.07.24_16.16.52_flip_up_sim_flip_sim_ft/checkpoints/latest.ckpt` |

Hydra：`flip_up_sim_finetune.yaml`、`train_spec_sim_finetune_workspace.yaml`

换腕部相机后须 **重录 + 重标 + 再微调**，旧外置相机权重不可直接用。

---

## 7. 目录结构

```text
sim_acp/
├── README.md
├── run_flip_cube_demo.py        # ★ 主入口
├── run_acp_effect_demo.py       # 方案 A 注入力 demo
├── bridge/
│   ├── i7_mujoco_backend.py     # IK + 接触力 + RGB + reset
│   ├── i7_scene.py              # tip + wrist_cam
│   ├── policy_runner.py
│   └── virtual_target.py
├── data/label_virtual_target.py
├── scripts/
│   ├── record_flip_episodes.py
│   ├── finetune_flip_spec.py
│   └── smoke_*.py
└── outputs/
scripts/run_sim_flip.sh          # ★ 一键启动
```

---

## 8. CLI 速查

| 参数 | 说明 |
|------|------|
| `--render` | MuJoCo 窗口 |
| `--loop` | 循环；FAIL 不退出；Ctrl+C / 关窗退出 |
| `--policy` | Flip Spec 三模态 |
| `--require-flip` | 验收 tilt≥55° + 真 RGB |
| `--ckpt` | 权重路径 |
| `--exec-horizon 12` / `--action-ds 50` | 开环执行 |
| `--viewer-sync-every 5` | viewer 同步节流 |
| `--fake-rgb` | 假灰图（勿用于 v2-ft 验收） |
| `--no-plot` | 不写刚度曲线 |

---

## 9. 口径与已知限制

| 现象 | 说明 |
|------|------|
| v2 零样本不翻 | 预期域差 |
| v2-ft 可翻 | 仿真分布微调结果 |
| 真机迁移 | 需 RealBackend + 真机数据，见适配文档 |

**本阶段成功标准**：仿真内 **策略 + 腕部 RGB** 稳定翻方块（≥55°）。
