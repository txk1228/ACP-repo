# ACP 复现交接说明

> 快照时间：2026-07-25（UTC+8）  
> 背景：负责人约 **8 月 1 日**返校，**真机复现未启动**；算法侧 Demo + Spec + Conv 对比均已完成；仿真翻方块（`sim_acp/`）已通。  
> 仓库（Private）：https://github.com/txk1228/ACP-repo  
> 本机工作区：`/home/zj/ACP_fx`（RTX 4090）

接手前请让仓库 Owner（`txk1228`）在 GitHub **Settings → Collaborators** 添加账号并接受邀请。

---

## 1. 一分钟现状

| 阶段 | 状态 | 说明 |
|------|------|------|
| 阶段一 Demo | ✅ 完成 | 虚拟目标 + 变刚度可视化，无需 GPU |
| 阶段二 Spec（论文主方法 / FFT） | ✅ 完成 | Flip + Vase，各 300 epoch；权重与 RMSE 已归档 |
| 阶段二 Conv（消融：ACP w/o FFT） | ✅ 完成 | Flip + Vase，各 300 epoch；终盘对照已归档 |
| 仿真 `sim_acp/` | ✅ 完成 | v2-ft 策略可翻方块（腕部 RGB） |
| 阶段三 真机（至简 i7 Pro） | 📋 未开始 | 仅有适配方案，无桥接代码 / 真机实验 |

**建议主线**：Spec/Conv 已对照 → **MuJoCo 单臂翻方块**（`sim_acp/`，已通）→ 有机后再按 `I7_ACP_ADAPTATION` 上真机（优先单臂 flip）。

进度面板（本机）：http://127.0.0.1:8765 — `bash scripts/run_progress_panel.sh`

---

## 2. 代码与文档

```bash
git clone https://github.com/txk1228/ACP-repo.git
cd ACP-repo
```

仓库已含官方 ACP 副本 `adaptive_compliance_policy/` 及本复现改动，一般不必再单独 clone 上游。

| 文档 | 用途 |
|------|------|
| **本文** `docs/HANDOVER.zh.md` | 交接总览（先读） |
| `README.md` | 快速开始、续训、进度面板 |
| `docs/REPRODUCTION.zh.md` | Demo → 训练步骤 |
| `docs/TRAINING_RESULTS_SPEC.zh.md` | Spec 权重路径、loss、RMSE |
| `docs/TRAINING_CONV_COMPARE.zh.md` | Conv 对比 yaml、**终盘 Spec/Conv 对照** |
| `docs/I7_ACP_ADAPTATION.zh.md` | 真机适配与桥接思路 |
| `docs/MUJOCO_ACP_SIM_MVP.zh.md` | **MuJoCo 单臂翻方块 MVP（方案 B / `sim_acp/`）** |
| `docs/ACP_ALGORITHM_GUIDE.zh.md` / `ACP_DATA_FLOW.zh.md` | 算法与读代码顺序 |

---

## 3. 本机资产（不在 Git）

| 类型 | 路径 | 备注 |
|------|------|------|
| Flip 数据集 | `~/data/real_processed/flip_up_new_v5` | ~10G |
| Vase 数据集 | `~/data/real_processed/vase_wiping_v6.3` | ~42G |
| 训练输出 | `~/training_outputs/` | ckpt / `logs.json.txt` / eval JSON |
| 训练日志 | `~/ACP_fx/logs/train_*.log` | gitignore |

换机接手：继续用本站，或拷贝 `data/real_processed` + 需要的 `training_outputs/<run_dir>`。

### 规范 run（部署 / 对比只认这些）

| 实验 | 状态 | 目录 |
|------|------|------|
| Flip Spec | ✅ | `~/training_outputs/2026.07.17_14.42.42_flip_up_new_resnet_230` |
| Vase Spec | ✅ | `~/training_outputs/2026.07.18_10.23.05_vase_wiping_resnet_230` |
| Flip Conv | ✅ | `~/training_outputs/2026.07.22_10.58.33_flip_up_new_conv_compare_230` |
| Vase Conv | ✅ | `~/training_outputs/2026.07.22_14.38.26_vase_wiping_conv_compare_230` |

部署权重：各目录 `checkpoints/latest.ckpt`。每 10 epoch 存盘，latest 对应约 **epoch 290**（已训满 0～299）属正常。

同前缀还有若干短跑 / 中断目录，**不要**当正式基线。

仓库内指标快照（防目录被挪）：`docs/training_snapshots/`（`spec_runs_index.json`、`conv_runs_index.json` 及各 `*_eval_*.json`）。

---

## 4. 指标摘要（验证集）

### Spec vs Conv（对齐 @ epoch 290；详见 TRAINING_* 文档）

| 任务 | 编码 | train_loss @299 | 位姿 RMSE ref / virt (mm) | 刚度 RMSE (N/m) |
|------|------|----------------:|--------------------------:|----------------:|
| Flip | Spec | ≈ 0.0106 | 9.17 / 10.37 | 482 |
| Flip | Conv | ≈ 0.0106 | 9.04 / 10.03 | 477 |
| Vase | Spec | ≈ 0.0102 | 22.46 / 24.29 | 834 |
| Vase | Conv | ≈ 0.0102 | 22.53 / 24.20 | 809 |

**离线结论**：两任务上 Spec≈Conv，对高频力谱不敏感；**最终以真机为准**。

---

## 5. 环境与常用命令

```bash
# 训练环境（4090，仅首次）
bash scripts/setup_pyrite_env_cuda.sh
conda activate pyrite
source scripts/setup_env.sh

# Demo（可无 GPU）
bash scripts/setup_demo_env.sh && conda activate acp-demo
bash scripts/run_demo.sh

# 进度面板
bash scripts/run_progress_panel.sh   # → http://127.0.0.1:8765

# 仿真翻方块（默认 v2-ft）
bash scripts/run_sim_flip.sh
```

环境变量（`setup_env.sh`）：

- `PYRITE_DATASET_FOLDERS=~/data/real_processed`
- `PYRITE_CHECKPOINT_FOLDERS=~/training_outputs`

断点续训示例见 `README.md`「断点续训」（四条正式 run 已满 300，一般无需再续）。

---

## 6. 真机（阶段三）——后续工作

**未做**：导纳真机验证、策略桥接、i7 数据采集、闭环部署。

**仿真已通**：[`docs/MUJOCO_ACP_SIM_MVP.zh.md`](MUJOCO_ACP_SIM_MVP.zh.md) / [`sim_acp/README.md`](../sim_acp/README.md)——**单臂翻方块**（v2-ft PASS）；**不报任务成功率**。

真机方案：[`docs/I7_ACP_ADAPTATION.zh.md`](I7_ACP_ADAPTATION.zh.md)（已按 v1.5 源码核实）。要点：

- 不依赖 UR5e；服务端已输出补偿后世界系 wrench；有 NRT 笛卡尔跟踪 + `force_admittance` 兜底  
- 不能直接用官方 `hardware_interfaces`；桥接与仿真共用契约，真机侧实现 `RealBackend`  
- **优先右臂 + 锁底盘 + flip**；首选方案 A（桥接预计算 `x_virt`，零改 C++）  
- 阻塞项（真机时）：核实 SDK Python 是否有 `moveNRT(笛卡尔)` / `getLatestState`  

推荐顺序：

1. ~~MuJoCo 翻方块~~（已通）  
2. 有机后：开启 `force_admittance` 手推 → RealBackend 换上 → Flip Spec 小幅度联调  
3. 扩展 episode（RGB + wrench + pose）后再本机微调；按需上方案 B  

控制栈：`/home/zj/robot-control-v1.5`（与 ACP-repo 分离；部署机路径以实际为准）。

---

## 7. 接手检查清单

- [ ] GitHub Collaborator 已接受，能 clone 私有仓  
- [ ] 能登录 4090 站，看到 `~/ACP_fx`、`~/data/real_processed`、`~/training_outputs`  
- [ ] `conda activate pyrite` + `source scripts/setup_env.sh` 正常  
- [ ] 四条规范 run 的 `checkpoints/latest.ckpt` 存在  
- [ ] 读完 `TRAINING_RESULTS_SPEC` + `TRAINING_CONV_COMPARE` + `I7_ACP_ADAPTATION`  
- [ ] `bash scripts/run_sim_flip.sh` 能开仿真  
- [ ] 与项目方确认：是否继续真机、优先 flip 还是 vase、时间表  

---

## 8. 联系与边界

| 项 | 说明 |
|----|------|
| 仓库 Owner | GitHub `txk1228` |
| 本机用户 | `zj` @ 4090 工作站 |
| 交接点 | 算法可复现 + Spec/Conv 基线权重 + 仿真翻方块；真机未做 |
| 不在本仓 | 原始大数据集、G 级 checkpoint、conda 环境本体 |

有疑问：本文 → `README.md` → 对应 `docs/*.zh.md`。
