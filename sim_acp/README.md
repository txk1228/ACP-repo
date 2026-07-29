# sim_acp — ACP MuJoCo 联调（方案 B）

**目标**：单臂翻方块。递进路径为脚本专家 → 腕部 RGB 三模态 → 仿真微调后策略可翻（tilt ≥55°）。  
`run_acp_effect_demo` 仅为方案 A 单元测试，非主线。  
不修改 `robot-control-v1.5`（i7 MJCF 本机 symlink）。

| 文档 | 内容 |
|------|------|
| 本文 | 操作手册与启动命令 |
| [`docs/MUJOCO_ACP_SIM_MVP.zh.md`](../docs/MUJOCO_ACP_SIM_MVP.zh.md) | 架构、里程碑、验收标准 |
| [`docs/I7_ACP_ADAPTATION.zh.md`](../docs/I7_ACP_ADAPTATION.zh.md) | 真机适配（后续） |

---

## 1. 阶段与结果

| 阶段 | 内容 | 状态 |
|------|------|------|
| 链路 | `RobotBackend` + 方案 A + i7 IK | ✅ |
| 场景 | 桌 + 方块 + tip 球 + 腕部相机 | ✅ |
| **v1** | 刚性脚本专家翻方块（无 RGB） | ✅ ~93° |
| **v2** | Flip Spec 三模态闭环（零样本） | ✅ 链路通，不稳翻 |
| 数据 | 50 ep 成功演示 + VT 标注 | ✅ |
| **v2-ft** | 仿真微调 → 策略可翻 | ✅ 5/5 ~70° |

零样本不稳翻来自**域差**（UR5e+GoPro 鱼眼+ATI ≠ i7 仿真腕部渲染+接触力），非链路故障。闭环手段：仿真专家数据 + 微调。

### 效果预览（v2-ft）

![sim flip v2-ft](../docs/media/sim_flip_v2ft.gif)

左：MuJoCo + 腕部 RGB；右：`|f|` / `|Δ|`、`k_soft` vs `k_hard`、软轴 `û` + `tilt`。  
权重来源：官方真机数据上训练 Spec，再经 i7 仿真微调（非官方发布包直接部署）。  
[MP4](../docs/media/sim_flip_v2ft.mp4) · `bash scripts/run_sim_flip.sh`  
读图：[效果展示页](../docs/media/README.md) · 重录：`bash scripts/make_github_media.sh`

可选 UI（同一 `ACP_SIM_FT_CKPT`）：`bash scripts/run_sim_flip.sh v2-ft-live`

**判定**：接触段 `|f|` 抬升、`k_soft` < `k_hard` 且 `û` 随接触转向；max `tilt` ≥ 55°（录屏约 90°）。

---

## 2. 启动

```bash
cd /path/to/ACP-repo
conda activate pyrite

bash scripts/run_sim_flip.sh                 # 默认：v2-ft + MuJoCo 窗口 + 循环
bash scripts/run_sim_flip.sh v1              # 脚本专家
bash scripts/run_sim_flip.sh v1-loop
bash scripts/run_sim_flip.sh v2              # Spec 零样本（验链路）
bash scripts/run_sim_flip.sh v2-ft           # 同默认
bash scripts/run_sim_flip.sh v2-ft-once
bash scripts/run_sim_flip.sh v2-ft-live      # 可选 Live 分屏
bash scripts/run_sim_flip.sh v2-ft-live-once
bash scripts/run_sim_flip.sh v2-ft-headless  # 无头评测
bash scripts/run_sim_flip.sh record|label|finetune|pipeline
bash scripts/run_sim_flip.sh help
```

脚本自动：`conda activate pyrite`、`source scripts/setup_env.sh`、检查 MJCF / ckpt。

### 环境变量

| 变量 | 默认 | 含义 |
|------|------|------|
| `ACP_I7_MODEL_ROOT` | `/home/zj/robot-control-v1.5/model_new` | i7 MJCF + meshes |
| `DISPLAY` | 自动检测 | MuJoCo / Live 显示 |
| `ACP_SIM_FT_CKPT` | 见下表 | v2-ft 权重 |
| `ACP_SIM_REAL_CKPT` | Flip Spec latest | v2 零样本权重 |
| `ACP_SIM_DATASET` | `$PYRITE_DATASET_FOLDERS/flip_up_sim_v1` | 仿真 zarr |
| `ACP_SIM_RECORD_N` | `50` | 录制条数 |

```bash
export ACP_SIM_FT_CKPT=~/training_outputs/<run_dir>/checkpoints/latest.ckpt
bash scripts/run_sim_flip.sh v2-ft
```

---

## 3. 版本说明

| 版本 | 启动 | 观测 / 控制 | 验收 |
|------|------|-------------|------|
| **v1** | `v1` | 脚本航点 + 刚性 tip | tilt ≥55° |
| **v2** | `v2` | RGB + wrench + pose → Spec 零样本 | 推理通 + 有位移；不宣称翻成功 |
| **v2-ft** | `v2-ft`（默认） | 仿真微调 ckpt；MuJoCo 主窗口 | **tilt ≥55°** |
| **v2-ft-live** | `v2-ft-live` | 同一 ckpt；Live 分屏 | 同左（展示） |

执行参数（对齐真机 runner）：开环 **12** 路点，间隔 **50** 仿真步。

- 主录屏：`bash scripts/make_github_media.sh` → `docs/media/sim_flip_v2ft.*`
- Live UI：`v2-ft-live`；RGB 落盘 `sim_acp/outputs/live_rgb_ft/`

---

## 4. 场景与传感器

生成：`$ACP_I7_MODEL_ROOT/mjcf/_acp_i7_scene.xml`（`i7_scene.py` 自动写入，可删）

| 组件 | 说明 |
|------|------|
| `acp_tip_ball` | tip 球，主接触 |
| `acp_ee_site` | 工具点 |
| `acp_wrist_cam` | 腕部 RGB 224×224 |
| `acp_obj` | 自由方块 |
| wrench | mesh 接触力 → 工具系 |

样张：`sim_acp/outputs/wrist_rgb_ft/`（v2-ft）、`live_rgb_ft/`（live）

---

## 5. 等价 Python 命令

```bash
conda activate pyrite
source scripts/setup_env.sh
export DISPLAY=:1 MUJOCO_GL=glfw
export ACP_I7_MODEL_ROOT=/path/to/robot-control-v1.5/model_new

FT=~/training_outputs/2026.07.24_16.16.52_flip_up_sim_flip_sim_ft/checkpoints/latest.ckpt

python -m sim_acp.run_flip_cube_demo \
  --policy --require-flip --ckpt "$FT" \
  --render --loop \
  --steps 3600 --exec-horizon 12 --action-ds 50 \
  --viewer-sync-every 5 --no-plot
```

---

## 6. 数据 → 微调

```bash
bash scripts/run_sim_flip.sh pipeline
# 或：record → label → finetune
```

| 产物 | 路径 |
|------|------|
| 仿真数据集 | `~/data/real_processed/flip_up_sim_v1/` |
| Spec 预训练（微调起点） | `~/training_outputs/2026.07.17_14.42.42_flip_up_new_resnet_230/checkpoints/latest.ckpt` |
| **v2-ft 权重** | `~/training_outputs/2026.07.24_16.16.52_flip_up_sim_flip_sim_ft/checkpoints/latest.ckpt` |

Hydra：`flip_up_sim_finetune.yaml`、`train_spec_sim_finetune_workspace.yaml`  
更换腕部相机后需重录、重标、再微调。

---

## 7. 目录结构

```text
sim_acp/
├── README.md
├── run_flip_cube_demo.py
├── run_acp_effect_demo.py
├── bridge/
│   ├── i7_mujoco_backend.py
│   ├── i7_scene.py
│   ├── policy_runner.py
│   └── virtual_target.py
├── data/label_virtual_target.py
├── scripts/
│   ├── record_flip_episodes.py
│   ├── finetune_flip_spec.py
│   ├── record_github_media.py
│   └── smoke_*.py
└── outputs/
```

仓库根目录：`scripts/run_sim_flip.sh`

---

## 8. CLI 速查

| 参数 | 说明 |
|------|------|
| `--render` | MuJoCo 窗口 |
| `--loop` | 循环；FAIL 不退出 |
| `--policy` | Flip Spec 三模态 |
| `--require-flip` | 验收 tilt≥55° + 真 RGB |
| `--ckpt` | 权重路径 |
| `--exec-horizon` / `--action-ds` | 开环执行 |
| `--fake-rgb` | 假灰图（勿用于验收） |
| `--no-plot` | 不写刚度曲线 |
| `--show-live-panel` | Live 分屏 |

---

## 9. 限制与标准

| 现象 | 说明 |
|------|------|
| v2 零样本不翻 | 预期域差 |
| v2-ft 可翻 | 仿真分布微调结果 |
| 真机迁移 | 需 `RealBackend` + 真机数据 |

**成功标准**：仿真内策略 + 腕部 RGB 稳定翻方块（tilt ≥55°）。
