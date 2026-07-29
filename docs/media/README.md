# 效果展示

可视化结果：仿真翻方块 **v2-ft**，以及阶段一算法 Demo。  
结论：接触方向自适应变柔顺、正交方向保持刚性，完成方块翻转。

返回：[项目主页](../../README.md) · [仿真手册](../../sim_acp/README.md)

---

## 背景

官方 ACP 基于 **UR5e 真机**（腕部 GoPro + ATI）数据训练；本仓迁移至 **i7** MuJoCo URDF（`acp_wrist_cam` + mesh 接触力）。  
真机域 Spec 零样本受域差限制；可翻权重 = **真机数据 Spec 训练 → 仿真专家数据微调**（`ACP_SIM_FT_CKPT`）。

---

## 1. 仿真翻方块 v2-ft

![ACP sim flip v2-ft trimodal](sim_flip_v2ft.gif)

腕部 RGB + wrench + pose → 微调策略闭环。左：外置机位与腕部相机；右：`|f|`、`|Δ|`、`k_soft` / `k_hard`、软轴 `û`、倾角 `tilt`。

完整视频：[`sim_flip_v2ft.mp4`](sim_flip_v2ft.mp4) · 运行：`bash scripts/run_sim_flip.sh`  
可选同权重 UI：`bash scripts/run_sim_flip.sh v2-ft-live`

---

## 2. 终盘刚度曲线

![v2-ft 终盘刚度曲线](sim_flip_v2ft_stiffness.png)

| 符号 | 含义 |
|------|------|
| `tip` | 末端工具点 |
| `tilt` | 方块倾角（0° 直立 → ≈90° 翻倒） |
| `û` | 软轴方向（接触方向） |

四联图（时间轴自上而下）：

1. **接触力** — 前段 `|f|`≈0 为接近；中段尖峰为接触。
2. **合规偏移** — 接触段 `|x_virt − x_ref|` 抬升。
3. **刚度** — `k_soft` 低于 `k_hard` 参照线；仅接触轴变软。
4. **软轴 + 倾角** — `û` 随接触转向，`tilt` → ~90°。

| 判定项 | 标准 | 本次录制 |
|--------|------|----------|
| 接触 | 接触段 `\|f\|` > 0 | ✅ |
| 方向性变软 | `k_soft` < `k_hard` | ✅ |
| 软轴自适应 | `û` 非零且随接触变化 | ✅ |
| 翻转成功 | max `tilt` ≥ 55° | ✅ ~90° |

<details>
<summary>tip 轨迹（辅助）</summary>

![tip 轨迹](sim_flip_v2ft_stiffness_traj.png)

`tip` / `x_ref` / `x_virt` 跟踪一致；变刚度证据以上述第 3、4 联为准。

</details>

---

## 3. 阶段一 Demo

![ACP virtual target + stiffness demo](virtual_target_stiffness_demo.png)

合成接触下：力增大时沿接触方向刚度下降、虚拟目标退让，正交方向保持高刚度。  
入口：`demo/virtual_target_stiffness_demo.py`。

---

## 文件与重录

| 文件 | 内容 |
|------|------|
| `sim_flip_v2ft.gif` / `.mp4` | 主录屏 |
| `sim_flip_v2ft_stiffness.png` | 终盘刚度四联图 |
| `sim_flip_v2ft_stiffness_traj.png` | tip / `x_ref` / `x_virt` 轨迹 |
| `virtual_target_stiffness_demo.png` | 阶段一静态图 |
| `sim_flip_v2ft_live.*` | 可选 Live UI 录屏 |

```bash
bash scripts/make_github_media.sh           # 主线 v2-ft
bash scripts/make_github_media.sh --also-live
```
