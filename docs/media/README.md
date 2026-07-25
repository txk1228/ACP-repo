# Demo media（GitHub 展示）

| 文件 | 内容 |
|------|------|
| `sim_flip_v2ft.mp4` / `.gif` | MuJoCo **v2-ft 三模态**（腕部 RGB + wrench + pose）翻方块；外置机位 + 右下角腕部 RGB |
| `virtual_target_stiffness_demo.mp4` / `.gif` / `.png` | 阶段一核心算法 Demo：虚拟目标 + 变刚度 |

重新生成：

```bash
bash scripts/make_github_media.sh
# 仅翻方块 / 仅 Demo：
bash scripts/make_github_media.sh --skip-demo
bash scripts/make_github_media.sh --skip-flip
```
