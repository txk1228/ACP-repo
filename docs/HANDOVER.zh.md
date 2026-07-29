# ACP 复现项目状态

> 更新：2026-07-29（UTC+8）  
> 仓库：https://github.com/txk1228/ACP-repo（Public）  
> 本机工作区：`/home/zj/ACP-repo`（RTX 4090；历史路径可能仍见 `ACP_fx`）

算法侧 Demo、Spec/Conv 对照与仿真翻方块（`sim_acp/`）已完成；真机复现未启动。

```bash
git clone https://github.com/txk1228/ACP-repo.git
cd ACP-repo
```

推送权限由 Owner（`txk1228`）在 GitHub Settings → Collaborators 添加。

---

## 1. 项目状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| 阶段一 Demo | ✅ | 虚拟目标 + 变刚度可视化，无需 GPU |
| 阶段二 Spec（FFT） | ✅ | Flip + Vase，各 300 epoch；权重与 RMSE 已归档 |
| 阶段二 Conv（消融） | ✅ | Flip + Vase，各 300 epoch；终盘对照已归档 |
| 仿真 `sim_acp/` | ✅ | v2-ft 策略可翻方块（腕部 RGB） |
| 阶段三 真机（i7 Pro） | 📋 | 仅有适配方案；无桥接代码 / 真机实验 |

推荐路径：Spec/Conv 归档 → **MuJoCo 单臂翻方块**（已通）→ 有机后按 `I7_ACP_ADAPTATION` 上真机（优先单臂 flip）。

进度面板：`bash scripts/run_progress_panel.sh` → http://127.0.0.1:8765

---

## 2. 文档索引

| 文档 | 用途 |
|------|------|
| 本文 | 状态、路径、命令入口 |
| `README.md` | 流程总览、续训、进度面板 |
| `docs/REPRODUCTION.zh.md` | Demo → 训练步骤 |
| `docs/TRAINING_RESULTS_SPEC.zh.md` | Spec 权重与指标 |
| `docs/TRAINING_CONV_COMPARE.zh.md` | Conv 对照 |
| `docs/I7_ACP_ADAPTATION.zh.md` | 真机适配 |
| `docs/MUJOCO_ACP_SIM_MVP.zh.md` | 仿真 MVP（方案 B） |
| `docs/ACP_ALGORITHM_GUIDE.zh.md` / `ACP_DATA_FLOW.zh.md` | 算法与数据流 |

仓库已含 `adaptive_compliance_policy/` 及本仓改动，一般无需再 clone 上游。

---

## 3. 本机资产（不在 Git）

| 类型 | 路径 | 备注 |
|------|------|------|
| Flip 数据集 | `~/data/real_processed/flip_up_new_v5` | ~10G |
| Vase 数据集 | `~/data/real_processed/vase_wiping_v6.3` | ~42G |
| 训练输出 | `~/training_outputs/` | ckpt / logs / eval |
| 训练日志 | `logs/train_*.log`（仓库内，gitignore） | |

换机：同步 `data/real_processed` 与所需 `training_outputs/<run_dir>`。

### 规范 run（基线）

| 实验 | 状态 | 目录 |
|------|------|------|
| Flip Spec | ✅ | `~/training_outputs/2026.07.17_14.42.42_flip_up_new_resnet_230` |
| Vase Spec | ✅ | `~/training_outputs/2026.07.18_10.23.05_vase_wiping_resnet_230` |
| Flip Conv | ✅ | `~/training_outputs/2026.07.22_10.58.33_flip_up_new_conv_compare_230` |
| Vase Conv | ✅ | `~/training_outputs/2026.07.22_14.38.26_vase_wiping_conv_compare_230` |

部署权重：各目录 `checkpoints/latest.ckpt`（约 epoch 290，训满 0–299 属正常）。  
同前缀短跑 / 中断目录不作为基线。  
指标快照：`docs/training_snapshots/`。

仿真微调权重（翻方块）：`~/training_outputs/2026.07.24_16.16.52_flip_up_sim_flip_sim_ft/checkpoints/latest.ckpt`

---

## 4. 指标摘要（验证集）

对齐 @ epoch 290；详见 TRAINING_* 文档。

| 任务 | 编码 | train_loss @299 | 位姿 RMSE ref / virt (mm) | 刚度 RMSE (N/m) |
|------|------|----------------:|--------------------------:|----------------:|
| Flip | Spec | ≈ 0.0106 | 9.17 / 10.37 | 482 |
| Flip | Conv | ≈ 0.0106 | 9.04 / 10.03 | 477 |
| Vase | Spec | ≈ 0.0102 | 22.46 / 24.29 | 834 |
| Vase | Conv | ≈ 0.0102 | 22.53 / 24.20 | 809 |

离线结论：两任务 Spec≈Conv，对高频力谱不敏感；最终以真机为准。

---

## 5. 环境与命令

```bash
bash scripts/setup_pyrite_env_cuda.sh
conda activate pyrite
source scripts/setup_env.sh

bash scripts/setup_demo_env.sh && conda activate acp-demo
bash scripts/run_demo.sh

bash scripts/run_progress_panel.sh
bash scripts/run_sim_flip.sh
```

- `PYRITE_DATASET_FOLDERS=~/data/real_processed`
- `PYRITE_CHECKPOINT_FOLDERS=~/training_outputs`

续训见 `README.md` §3。四条规范 run 已满 300 epoch，通常无需再续。

---

## 6. 真机与后续

**未完成**：导纳真机验证、策略桥接、i7 数据采集、闭环部署。

**仿真**：[`MUJOCO_ACP_SIM_MVP.zh.md`](MUJOCO_ACP_SIM_MVP.zh.md) / [`sim_acp/README.md`](../sim_acp/README.md) — 单臂翻方块 v2-ft PASS；不报任务成功率。

真机方案：[`I7_ACP_ADAPTATION.zh.md`](I7_ACP_ADAPTATION.zh.md)。要点：

- 不依赖 UR5e；服务端输出补偿世界系 wrench；NRT 笛卡尔跟踪 + `force_admittance`
- 不可直接使用官方 `hardware_interfaces`；仿真与真机共用 `RobotBackend` 契约
- 优先：右臂 + 锁底盘 + flip；方案 A（预计算 `x_virt`，零改 C++）
- 阻塞项：核实 SDK Python 的笛卡尔 `moveNRT` / `getLatestState`

推荐顺序：

1. MuJoCo 翻方块（已完成）
2. 开启 `force_admittance` → `RealBackend` → Flip Spec 小幅度联调
3. 扩展 episode 后本机微调；按需方案 B

控制栈：`/home/zj/robot-control-v1.5`（与 ACP-repo 分离）。

### 本机未入库：`sim_vase/`（双臂擦拭 spike）

- 路径：本机 `~/ACP_fx/sim_vase/`（未推 GitHub）
- 定位：实验线，**非**当前主线（主线仍为翻方块 → 真机 flip）
- 已完成：双 tip 脚本擦固定花瓶；双腕 RGB；双臂方案 A 单元验收
- 未完成：Vase Spec 闭环、示教落盘与微调、论文级擦拭指标

相对单臂翻方块的额外难点：双臂契约、持续双边接触、成功判据更碎、进策略闭环路径更长；须独立场景与 backend，避免污染 flip 链路。

---

## 7. 验证清单

- [ ] clone `https://github.com/txk1228/ACP-repo`
- [ ] 本机可见 `~/data/real_processed`、`~/training_outputs`
- [ ] `conda activate pyrite` + `source scripts/setup_env.sh`
- [ ] 四条规范 run 的 `latest.ckpt` 存在
- [ ] 阅读 `TRAINING_RESULTS_SPEC`、`TRAINING_CONV_COMPARE`、`I7_ACP_ADAPTATION`
- [ ] `bash scripts/run_sim_flip.sh` 可启动
- [ ] 确认后续是否推进真机，以及 flip / vase 优先级

---

## 8. 边界

| 项 | 说明 |
|----|------|
| 仓库 Owner | `txk1228` |
| 本机用户 | `zj` @ 4090 |
| 已交付 | 算法复现 + Spec/Conv 基线 + 仿真翻方块 |
| 不在本仓 | 原始大数据集、大体积 checkpoint、conda 环境；另有未入库 `sim_vase/` |
