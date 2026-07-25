# Demo media（GitHub 展示）

| 文件 | 内容 |
|------|------|
| `sim_flip_v2ft.mp4` / `.gif` | **分屏**：左 MuJoCo（外置机位 + 腕部 RGB）；右实时 ACP 面板（\|f\| / \|Δ\|、**k_soft vs k_hard**、soft-axis û + tilt） |
| `sim_flip_v2ft_stiffness.png` | 同一次录制的终盘刚度曲线（另有 `_traj.png`） |
| `virtual_target_stiffness_demo.png` | 阶段一核心算法 Demo 静态图 |

右侧面板要表达的本质：**接触方向（soft û）上 \(k\) 变软，正交方向 \(k_\text{hard}\) 保持高刚度**。

重新生成：

```bash
bash scripts/make_github_media.sh
```
