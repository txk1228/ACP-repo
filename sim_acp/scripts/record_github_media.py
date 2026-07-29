#!/usr/bin/env python3
"""录制 GitHub 展示用媒体：v2-ft 三模态翻方块 + 虚拟目标 Demo 动画。

输出目录默认：docs/media/

  conda activate pyrite
  source scripts/setup_env.sh
  python -m sim_acp.scripts.record_github_media
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))


def _encode_mp4(frame_dir: Path, pattern: str, out_mp4: Path, fps: int) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_mp4.with_suffix(".tmp.mp4")
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / pattern),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "26",
            "-movflags",
            "+faststart",
            str(tmp),
        ]
        subprocess.run(cmd, check=True)
        tmp.replace(out_mp4)
        return

    # Fallback：无系统 ffmpeg 时用 OpenCV（体积略大，但 README 可嵌入）
    frames = sorted(frame_dir.glob(pattern.replace("%05d", "*")))
    if not frames:
        raise FileNotFoundError(f"no frames matching {pattern} in {frame_dir}")
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"failed to read {frames[0]}")
    h, w = first.shape[:2]
    if h % 2:
        h -= 1
    if w % 2:
        w -= 1
    writer = cv2.VideoWriter(
        str(tmp),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV VideoWriter failed to open (install ffmpeg preferred)")
    try:
        for fp in frames:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(img)
    finally:
        writer.release()
    tmp.replace(out_mp4)


def maybe_make_gif(mp4: Path, gif: Path, *, scale: int = 480, fps: int = 10) -> Path | None:
    if not mp4.is_file():
        return None
    gif.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(mp4),
            "-vf",
            f"fps={fps},scale={scale}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
            str(gif),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            return None
        return gif if gif.is_file() else None

    # Fallback：抽帧写 GIF（无 ffmpeg）
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return None
    frames_rgb: list[np.ndarray] = []
    src_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps)
    step = max(1, int(round(src_fps / max(1, fps))))
    i = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % step == 0:
                h, w = bgr.shape[:2]
                if w > scale:
                    nh = max(1, int(round(h * (scale / float(w)))))
                    bgr = cv2.resize(bgr, (scale, nh), interpolation=cv2.INTER_AREA)
                frames_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            i += 1
    finally:
        cap.release()
    if not frames_rgb:
        return None
    try:
        import imageio.v2 as imageio

        imageio.mimsave(str(gif), frames_rgb, fps=fps, loop=0)
    except Exception:
        return None
    return gif if gif.is_file() else None


def classify_contact_phase(
    *,
    force_norm: float,
    k_soft: float,
    k_hard: float,
    delta_mm: float,
    contacted_started: bool,
    f_on_n: float = 1.5,
    f_hold_n: float = 0.6,
    f_exit_n: float = 0.35,
    k_soft_ratio: float = 0.92,
    delta_on_mm: float = 0.8,
) -> tuple[str, bool, str]:
    """接触阶段判定。

    - PRE：尚未发生真实接触（必须 |f| 超过 f_on 才算第一次接触）
    - CONTACT：接触中，力仍明显，或力略降但柔顺/偏移仍在操作
    - POST：已离开接触（|f| 回落到 f_exit 以下），即使 k_soft/|Δ| 尚未完全归零
    """
    force_touch = force_norm >= f_on_n
    force_hold = force_norm >= f_hold_n
    soft_active = k_soft < k_hard * k_soft_ratio
    delta_active = delta_mm >= delta_on_mm

    if not contacted_started:
        if force_touch:
            return "CONTACT", True, "manipulation: |f|"
        return "PRE", False, "approach, no contact yet"

    # 力已释放：结束接触，不再因 k_soft/|Δ| 残留而保持红色
    if force_norm < f_exit_n:
        return "POST", contacted_started, "after contact, released"

    if force_hold:
        reasons = ["|f|"]
        if soft_active:
            reasons.append("k_soft↓")
        if delta_active:
            reasons.append("|Δ|")
        return "CONTACT", contacted_started, "manipulation: " + "+".join(reasons)

    # f_exit <= |f| < f_hold：短暂力回落，仅当柔顺仍在操作时才保持红色
    if soft_active or delta_active:
        reasons: list[str] = []
        if soft_active:
            reasons.append("k_soft↓")
        if delta_active:
            reasons.append("|Δ|")
        return "CONTACT", contacted_started, "manipulation: " + "+".join(reasons)

    return "POST", contacted_started, "after contact, released"


def _center_crop_zoom(bgr: np.ndarray, zoom: float = 1.0) -> np.ndarray:
    """中心裁切放大，zoom>1 时裁掉四周留白、让主体更满。"""
    z = float(zoom)
    if z <= 1.0 + 1e-6:
        return bgr
    h, w = bgr.shape[:2]
    nw = max(1, int(round(w / z)))
    nh = max(1, int(round(h / z)))
    x0 = max(0, (w - nw) // 2)
    y0 = max(0, (h - nh) // 2)
    crop = bgr[y0 : y0 + nh, x0 : x0 + nw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)


def _overlay_pip(overview_bgr: np.ndarray, wrist_rgb: np.ndarray) -> np.ndarray:
    """左下角叠腕部 RGB（右侧留给刚度面板）。"""
    h, w = overview_bgr.shape[:2]
    pip_w = max(160, w // 3)
    pip_h = max(160, h // 3)
    wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
    sh, sw = wrist_bgr.shape[:2]
    # INTER_AREA 更适合下采样；需要放大时用线性/立方避免过度糊化
    if pip_w <= sw and pip_h <= sh:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC
    pip = cv2.resize(wrist_bgr, (pip_w, pip_h), interpolation=interp)
    x0 = 12
    y0 = h - pip_h - 12
    overview_bgr[y0 : y0 + pip_h, x0 : x0 + pip_w] = pip
    cv2.rectangle(
        overview_bgr, (x0 - 2, y0 - 2), (x0 + pip_w + 1, y0 + pip_h + 1), (240, 240, 240), 2
    )
    cv2.putText(
        overview_bgr,
        "wrist RGB",
        (x0, y0 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return overview_bgr


def _draw_hud(
    frame_bgr: np.ndarray,
    tilt_deg: float,
    step: int,
    k_low: float | None,
    *,
    contact_state: str | None = None,
    contact_detail: str | None = None,
    title: str = "ACP v2-ft trimodal",
) -> None:
    # 颜色高亮：接触前/接触中/接触后
    # BGR：
    #   PRE:   灰绿 — 尚未接触，主要靠视觉接近
    #   CONTACT: 红 — |f|、k_soft 下降或 |Δ| 表明力控/柔顺在主导
    #   POST:  青 — 曾接触过，当前力与柔顺偏移都较小
    bar_color = (20, 20, 20)
    marker = ""
    detail = contact_detail or ""
    if contact_state:
        s = contact_state.upper()
        if s.startswith("PRE"):
            bar_color = (25, 75, 25)
            marker = "PRE-CONTACT"
            if not detail:
                detail = "approach, no contact yet"
        elif s.startswith("CONTACT"):
            bar_color = (40, 40, 200)
            marker = "IN CONTACT"
            if not detail:
                detail = "force or compliance active"
        elif s.startswith("POST"):
            bar_color = (25, 180, 150)
            marker = "POST-CONTACT"
            if not detail:
                detail = "after contact, released"
        else:
            marker = s

    cv2.rectangle(frame_bgr, (0, 0), (frame_bgr.shape[1], 36), bar_color, -1)
    label = f"{title}  tilt={tilt_deg:5.1f}deg  step={step}"
    if marker:
        label += f"  {marker}"
    if detail:
        label += f" ({detail})"
    if k_low is not None:
        label += f"  k_soft={k_low:.0f}N/m"
    cv2.putText(
        frame_bgr,
        label,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )


class CompliancePanel:
    """右侧实时面板：力 / k_soft vs k_hard / soft-axis + tilt。"""

    def __init__(
        self,
        width: int = 420,
        height: int = 360,
        k_hard: float = 5000.0,
        render_scale: float = 2.0,
    ):
        import matplotlib

        matplotlib.use("Agg")
        self.width = int(width)
        self.height = int(height)
        self.k_hard = float(k_hard)
        self._render_scale = max(1.0, float(render_scale))
        self._fig_wh: tuple[int, int] | None = None
        self.steps: list[int] = []
        self.fn: list[float] = []
        self.k_soft: list[float] = []
        self.delta_mm: list[float] = []
        self.rgb_diff: list[float] = []
        self.soft: list[np.ndarray] = []
        self.tilt: list[float] = []
        self._fig = None
        self._axes = None
        self._ax0b = None
        self._ax2c = None
        # 窄面板（≈400px 宽）字号/边距
        self._fs_title = 7.0
        self._fs_label = 6.0
        self._fs_tick = 5.5
        self._fs_legend = 5.0

    def push(
        self,
        step: int,
        *,
        force_xyz: np.ndarray,
        k_soft: float,
        soft_axis: np.ndarray,
        delta_m: float,
        tilt_deg: float,
        rgb_diff: float = 0.0,
    ) -> None:
        self.steps.append(int(step))
        self.fn.append(float(np.linalg.norm(force_xyz)))
        self.k_soft.append(float(k_soft))
        self.delta_mm.append(float(delta_m) * 1000.0)
        self.rgb_diff.append(float(rgb_diff))
        self.soft.append(np.asarray(soft_axis, dtype=float).reshape(3).copy())
        self.tilt.append(float(tilt_deg))

    def _style_ax(self, ax, *, label_fs: float | None = None) -> None:
        fs = self._fs_tick if label_fs is None else label_fs
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(
            colors="#b8b8b8",
            labelsize=fs,
            length=2.5,
            width=0.6,
            pad=1.5,
        )
        for spine in ax.spines.values():
            spine.set_color("#4a4a4a")
            spine.set_linewidth(0.6)
        ax.grid(True, alpha=0.22, color="#5a5a5a", linewidth=0.5)

    def _set_title(self, ax, text: str, *, pad: float | None = None) -> None:
        ax.set_title(
            text,
            fontsize=self._fs_title,
            color="#ececec",
            pad=pad if pad is not None else 5,
            loc="left",
            fontweight="normal",
        )

    def _compact_legend(
        self,
        ax,
        *,
        loc: str = "upper left",
        ncol: int = 1,
        y: float = 0.96,
    ) -> None:
        anchor = {
            "upper left": (0.0, y),
            "upper right": (1.0, y),
            "lower left": (0.0, 0.04),
            "lower right": (1.0, 0.04),
        }.get(loc, (0.0, y))
        ax.legend(
            loc=loc,
            bbox_to_anchor=anchor,
            ncol=ncol,
            fontsize=self._fs_legend,
            frameon=True,
            facecolor="#1c1c1c",
            edgecolor="#3d3d3d",
            labelcolor="#d8d8d8",
            framealpha=0.92,
            handlelength=1.4,
            handletextpad=0.35,
            borderpad=0.25,
            labelspacing=0.25,
            columnspacing=0.6,
        )

    def _style_xaxis(self, ax, t: np.ndarray, *, show_label: bool) -> None:
        import matplotlib.ticker as mticker

        t0 = float(t[0])
        t1 = float(t[-1]) if len(t) > 1 else t0 + 1.0
        ax.set_xlim(t0, t1)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True, min_n_ticks=2))
        if show_label:
            ax.set_xlabel("step", fontsize=self._fs_label, color="#bdbdbd", labelpad=2)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)

    def _apply_layout(self) -> None:
        # top 留足首图标题空间，避免裁切
        self._fig.subplots_adjust(
            left=0.17,
            right=0.86,
            top=0.90,
            bottom=0.08,
            hspace=0.64,
        )

    def render_bgr(self) -> np.ndarray:
        import matplotlib.pyplot as plt

        ss = self._render_scale
        rw = max(self.width, int(round(self.width * ss)))
        rh = max(self.height, int(round(self.height * ss)))
        if self._fig is None or self._fig_wh != (rw, rh):
            if self._fig is not None:
                plt.close(self._fig)
            self._fig = None
            self._axes = None
            self._ax0b = None
            self._ax2c = None
            dpi = 192
            self._fig, self._axes = plt.subplots(
                3,
                1,
                figsize=(rw / dpi, rh / dpi),
                dpi=dpi,
            )
            self._fig_wh = (rw, rh)
            self._fig.patch.set_facecolor("#111111")
            self._apply_layout()

            self._ax0b = self._axes[0].twinx()
            self._ax2c = self._axes[2].twinx()
            for ax in list(self._axes) + [self._ax0b, self._ax2c]:
                self._style_ax(ax)

        ax0, ax1, ax2 = self._axes
        ax0b, ax2c = self._ax0b, self._ax2c
        for ax in (ax0, ax1, ax2, ax0b, ax2c):
            ax.cla()
            self._style_ax(ax)

        if not self.steps:
            blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            blank[:] = (17, 17, 17)
            return blank

        t = np.asarray(self.steps, dtype=float)
        fn = np.asarray(self.fn, dtype=float)
        ks = np.asarray(self.k_soft, dtype=float)
        dmm = np.asarray(self.delta_mm, dtype=float)
        rgbd = np.asarray(self.rgb_diff, dtype=float)
        soft = np.stack(self.soft, axis=0)
        tilt = np.asarray(self.tilt, dtype=float)

        # 1) force + compliance + RGB diff
        self._set_title(ax0, "Force · Compliance · Vision", pad=7)
        ax0.plot(t, fn, color="#4fc3f7", lw=1.2, label="|f|")
        ax0.set_ylabel("N", fontsize=self._fs_label, color="#4fc3f7", labelpad=2)
        ax0.tick_params(axis="y", labelcolor="#4fc3f7", colors="#4fc3f7")
        ax0b.plot(t, dmm, color="#ffb74d", lw=1.0, label="|Δ|")
        ax0b.plot(t, rgbd, color="#ffd54f", lw=0.9, ls="--", label="RGB")
        ax0b.set_ylabel("mm", fontsize=self._fs_label, color="#ffb74d", labelpad=2)
        ax0b.tick_params(axis="y", labelcolor="#ffb74d", colors="#ffb74d")
        self._style_xaxis(ax0, t, show_label=False)
        self._compact_legend(ax0, loc="upper left", ncol=2, y=0.90)
        self._compact_legend(ax0b, loc="upper right", ncol=1, y=0.90)

        # 2) stiffness
        self._set_title(ax1, "Adaptive Stiffness")
        ax1.plot(t, ks, color="#ef5350", lw=1.3, label="k_soft")
        ax1.axhline(
            self.k_hard,
            color="#66bb6a",
            lw=1.1,
            ls="--",
            label="k_hard",
        )
        ax1.set_ylabel("N/m", fontsize=self._fs_label, color="#cccccc", labelpad=2)
        ax1.set_ylim(0.0, max(self.k_hard * 1.12, float(ks.max()) * 1.08, 1.0))
        self._style_xaxis(ax1, t, show_label=False)
        self._compact_legend(ax1, loc="upper right", ncol=2)

        # 3) soft axis + tilt
        self._set_title(ax2, "Soft Axis & Tilt")
        ax2.plot(t, soft[:, 0], color="#42a5f5", lw=1.0, label="x")
        ax2.plot(t, soft[:, 1], color="#ab47bc", lw=1.0, label="y")
        ax2.plot(t, soft[:, 2], color="#26a69a", lw=1.0, label="z")
        ax2.set_ylabel("û", fontsize=self._fs_label, color="#cccccc", labelpad=2)
        ax2.set_ylim(-1.1, 1.1)
        ax2c.plot(t, tilt, color="#e0e0e0", lw=1.2, label="tilt")
        ax2c.set_ylabel("deg", fontsize=self._fs_label, color="#e0e0e0", labelpad=2)
        ax2c.tick_params(axis="y", labelcolor="#e0e0e0", colors="#e0e0e0")
        self._style_xaxis(ax2, t, show_label=True)
        self._compact_legend(ax2, loc="upper left", ncol=3, y=0.92)
        self._compact_legend(ax2c, loc="upper right", ncol=1, y=0.92)

        self._apply_layout()
        self._fig.canvas.draw()
        rgba = np.asarray(self._fig.canvas.buffer_rgba())
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        if bgr.shape[1] != self.width or bgr.shape[0] != self.height:
            bgr = cv2.resize(
                bgr, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
        return bgr

    def close(self) -> None:
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._axes = None
            self._ax0b = None
            self._ax2c = None
            self._fig_wh = None


def _compose_split(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    h = max(left_bgr.shape[0], right_bgr.shape[0])
    if left_bgr.shape[0] != h:
        interp = cv2.INTER_AREA if h < left_bgr.shape[0] else cv2.INTER_CUBIC
        left_bgr = cv2.resize(left_bgr, (left_bgr.shape[1], h), interpolation=interp)
    if right_bgr.shape[0] != h:
        interp = cv2.INTER_AREA if h < right_bgr.shape[0] else cv2.INTER_CUBIC
        right_bgr = cv2.resize(right_bgr, (right_bgr.shape[1], h), interpolation=interp)
    # 中间分隔线
    sep = np.full((h, 4, 3), 60, dtype=np.uint8)
    out = np.concatenate([left_bgr, sep, right_bgr], axis=1)
    # yuv420p 要求偶数边
    hh, ww = out.shape[:2]
    if hh % 2:
        out = out[:-1, :, :]
    if ww % 2:
        out = out[:, :-1, :]
    return out


def record_sim_flip(
    out_mp4: Path,
    ckpt: Path,
    *,
    width: int = 640,
    height: int = 360,
    panel_width: int = 420,
    fps: int = 15,
    frame_every: int = 6,
    steps: int = 5200,
    exec_horizon: int = 12,
    action_ds: int = 50,
    flip_done_deg: float = 85.0,
    hold_after_flip: int = 500,
    k_hard: float = 5000.0,
    stiffness_png: Path | None = None,
    overview_zoom: float = 1.0,
    render_scale: float = 1.0,
    overview_distance: float = 1.55,
    hud_title: str = "ACP v2-ft trimodal",
    panel_render_scale: float = 2.0,
) -> dict:
    from sim_acp.bridge.i7_mujoco_backend import I7MujocoBackend
    from sim_acp.bridge.plot_compliance import save_compliance_plots
    from sim_acp.bridge.policy_runner import FlipSpecPolicyRunner
    from sim_acp.bridge.pose_buffer import PoseRingBuffer
    from sim_acp.bridge.rgb_buffer import RgbRingBuffer
    from sim_acp.bridge.virtual_target import soft_axis_from_policy
    from sim_acp.bridge.wrench_buffer import WrenchRingBuffer

    backend = I7MujocoBackend(render=False, tip_clamp=False)
    runner = FlipSpecPolicyRunner(ckpt_path=str(ckpt))
    rh, rw = runner.rgb_hw
    buf = WrenchRingBuffer(capacity=max(8000, runner.wrench_h + 10))
    pose_buf = PoseRingBuffer(capacity=max(16, runner.pose_h + 4))
    rgb_buf = RgbRingBuffer(capacity=max(8, runner.rgb_h + 2), h=rh, w=rw)
    from sim_acp.bridge.i7_scene import OFFSCREEN_MAX_H, OFFSCREEN_MAX_W

    panel = CompliancePanel(
        width=panel_width,
        height=height,
        k_hard=k_hard,
        render_scale=float(panel_render_scale),
    )
    hist: list[dict] = []

    for _ in range(80):
        backend.inject_force_W(np.zeros(3))
        backend.step(realtime=False)

    st0 = backend.read_state()
    tip0 = backend.tip_pos()
    pose0 = st0.pose_xyzquat.copy()
    pose0[:3] = tip0
    w0 = st0.wrench_W.copy()
    for _ in range(runner.wrench_h):
        buf.push(w0)
    for _ in range(runner.pose_h):
        pose_buf.push(pose0)
    rgb0 = backend.render_rgb()
    for _ in range(max(2, runner.rgb_h)):
        rgb_buf.push(rgb0)
    for _ in range(max(2, runner.rgb_h)):
        backend.step(realtime=False)
        st0 = backend.read_state()
        p = st0.pose_xyzquat.copy()
        p[:3] = backend.tip_pos()
        pose_buf.push(p)
        buf.push(st0.wrench_W)
        rgb_buf.push(backend.render_rgb())

    p0 = backend.tip_pos().copy()
    clip_lo = np.array([0.55, -0.70, 0.75])
    clip_hi = np.array([0.90, -0.40, 1.00])
    max_tilt = backend.cube_tilt_rad()
    i_step = 0
    k_show = float(k_hard)
    soft_show = np.zeros(3)
    x_ref_show = p0.copy()
    x_virt_show = p0.copy()
    frame_i = 0
    flip_done_rad = math.radians(float(flip_done_deg))
    hold_left = -1
    contacted = False
    contact_detail = "approach, no contact yet"
    rs = max(1.0, float(render_scale))
    oz = max(1.0, float(overview_zoom))

    tmp_root = Path(tempfile.mkdtemp(prefix="acp_media_flip_"))
    try:
        def _capture() -> None:
            nonlocal frame_i, max_tilt, contacted, contact_detail
            render_w = min(int(round(width * rs)), int(OFFSCREEN_MAX_W))
            render_h = min(int(round(height * rs)), int(OFFSCREEN_MAX_H))
            overview = backend.render_overview_rgb(
                width=render_w,
                height=render_h,
                distance=float(overview_distance),
            )
            # 腕部 PIP 略超采样，避免放大发糊
            pip_side = max(160, width // 3)
            wrist_side = min(
                int(round(pip_side * rs * 1.15)),
                int(OFFSCREEN_MAX_W),
                int(OFFSCREEN_MAX_H),
            )
            wrist = backend.render_rgb(width=wrist_side, height=wrist_side)
            tilt = math.degrees(backend.cube_tilt_rad())
            max_tilt = max(max_tilt, backend.cube_tilt_rad())
            st = backend.read_state()
            f = st.wrench_W[:3].copy()
            fn = float(np.linalg.norm(f))
            delta = float(np.linalg.norm(x_virt_show - x_ref_show))
            contact_state, contacted, contact_detail = classify_contact_phase(
                force_norm=fn,
                k_soft=float(k_show),
                k_hard=float(k_hard),
                delta_mm=delta * 1000.0,
                contacted_started=contacted,
            )
            panel.push(
                i_step,
                force_xyz=f,
                k_soft=float(k_show),
                soft_axis=soft_show,
                delta_m=delta,
                tilt_deg=tilt,
            )
            left = cv2.cvtColor(overview, cv2.COLOR_RGB2BGR)
            left = _center_crop_zoom(left, zoom=oz)
            if left.shape[1] != width or left.shape[0] != height:
                left = cv2.resize(left, (width, height), interpolation=cv2.INTER_AREA)
            left = _overlay_pip(left, wrist)
            _draw_hud(
                left,
                tilt,
                i_step,
                k_show,
                contact_state=contact_state,
                contact_detail=contact_detail,
                title=hud_title,
            )
            right = panel.render_bgr()
            frame = _compose_split(left, right)
            cv2.imwrite(str(tmp_root / f"frame_{frame_i:05d}.png"), frame)
            frame_i += 1

        _capture()
        while i_step < steps:
            st = backend.read_state()
            rgb_buf.push(backend.render_rgb())
            pose_hist = pose_buf.stack_last(runner.pose_h)
            act = runner.predict(
                pose_hist,
                buf.window(runner.wrench_h),
                rgb_uint8=rgb_buf.stack_last(runner.rgb_h),
                fake_rgb=False,
                force_xyz=st.wrench_W[:3],
            )
            virt_traj = (
                act.x_virt_traj
                if act.x_virt_traj is not None
                else act.x_virt_pos.reshape(1, 3)
            )
            ref_traj = (
                act.x_ref_traj
                if act.x_ref_traj is not None
                else act.x_ref_pos.reshape(1, 3)
            )
            k_traj = act.k_traj if act.k_traj is not None else np.array([act.k_low])
            H = min(exec_horizon, len(virt_traj))
            tip_cur = backend.tip_pos()
            print(
                f"  [record i={i_step}] k0={act.k_low:.0f} "
                f"tilt={math.degrees(max_tilt):.1f}deg "
                f"hold={hold_left} frames={frame_i}"
            )
            for h in range(H):
                wp = np.clip(virt_traj[h], clip_lo, clip_hi)
                x_ref = np.asarray(ref_traj[min(h, len(ref_traj) - 1)], dtype=float).reshape(3)
                k_show = float(k_traj[min(h, len(k_traj) - 1)])
                soft_show = soft_axis_from_policy(
                    x_ref, wp, force_xyz=backend.read_state().wrench_W[:3]
                )
                x_ref_show = x_ref.copy()
                x_virt_show = wp.copy()
                for s in range(action_ds):
                    if i_step >= steps:
                        break
                    a = (s + 1) / max(1, action_ds)
                    tip_ref = (1.0 - a) * tip_cur + a * wp
                    backend.inject_force_W(np.zeros(3))
                    st = backend.read_state()
                    pose_tcp = st.pose_xyzquat.copy()
                    pose_tcp[:3] = backend.tip_pos()
                    buf.push(st.wrench_W)
                    pose_buf.push(pose_tcp)
                    if i_step % 5 == 0:
                        rgb_buf.push(backend.render_rgb())
                    backend.write_tip_pos(tip_ref, st.timestamp_ns)
                    backend.step(realtime=False)
                    tip_now = backend.tip_pos()
                    tilt = backend.cube_tilt_rad()
                    max_tilt = max(max_tilt, tilt)
                    hist.append(
                        {
                            "f": st.wrench_W[:3].copy(),
                            "x_ref": x_ref.copy(),
                            "x_virt": tip_ref.copy(),
                            "x_tip": tip_now.copy(),
                            "tilt_deg": math.degrees(tilt),
                            "k_low": k_show,
                            "soft_axis": soft_show.copy(),
                        }
                    )
                    if i_step % frame_every == 0:
                        x_virt_show = tip_ref.copy()
                        _capture()
                    i_step += 1
                    if hold_left < 0 and tilt >= flip_done_rad:
                        hold_left = int(hold_after_flip)
                        print(
                            f"  [flip-done] tilt={math.degrees(tilt):.1f}deg "
                            f"→ hold {hold_left} steps"
                        )
                    if hold_left >= 0:
                        hold_left -= 1
                        if hold_left <= 0:
                            i_step = steps
                            break
                tip_cur = wp.copy()
                if i_step >= steps:
                    break
        _capture()
        _encode_mp4(tmp_root, "frame_%05d.png", out_mp4, fps=fps)

        if stiffness_png is not None and hist:
            save_compliance_plots(
                hist,
                stiffness_png,
                k_low=float(np.median([h["k_low"] for h in hist])),
                k_high=float(k_hard),
                title="v2-ft trimodal flip — soft-axis k vs orthogonal k_hard",
                delta_clip_m=0.0,
            )
    finally:
        panel.close()
        try:
            backend.close()
        except Exception:
            pass
        shutil.rmtree(tmp_root, ignore_errors=True)

    return {
        "out": str(out_mp4),
        "frames": frame_i,
        "max_tilt_deg": math.degrees(max_tilt),
        "bytes": out_mp4.stat().st_size if out_mp4.is_file() else 0,
        "stiffness_png": str(stiffness_png) if stiffness_png else None,
    }


def record_stiffness_demo(out_mp4: Path, out_png: Path, *, fps: int = 12) -> dict:
    """把阶段一 Demo 做成时序动画（力 / 刚度 / 轨迹）。"""
    # 用系统/conda python 跑 demo 逻辑；不强制 acp-demo 解释器检查。
    demo_path = _REPO / "demo" / "virtual_target_stiffness_demo.py"
    # 动态加载，绕过 _check_runtime 对 acp-demo 的硬性要求
    import importlib.util

    # 确保能 import PyriteUtility
    acp_src = _REPO / "adaptive_compliance_policy"
    if str(acp_src) not in sys.path:
        sys.path.insert(0, str(acp_src))

    # 直接复用 demo 文件中的函数：临时禁用其 runtime check
    src = demo_path.read_text(encoding="utf-8")
    src = src.replace("_check_runtime()\n", "# _check_runtime() skipped for media\n")
    tmp_mod = Path(tempfile.mkdtemp(prefix="acp_demo_mod_")) / "demo_mod.py"
    tmp_mod.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("acp_demo_mod", tmp_mod)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    t, pose7, wrench, contact_start = mod.synthesize_contact_episode()
    k_series, virtual_pose7, offsets, _ = mod.run_virtual_target_pipeline(pose7, wrench)
    force_norm = np.linalg.norm(wrench[:, :3], axis=1)

    # 静态终盘图（进 docs/media）
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = mod.plot_results(
        t, pose7, virtual_pose7, wrench, k_series, offsets, contact_start, None
    )
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)

    tmp_root = Path(tempfile.mkdtemp(prefix="acp_media_demo_"))
    n = len(t)
    stride = max(1, n // 90)  # ~90 帧
    frame_i = 0
    try:
        for i in range(0, n, stride):
            fig = plt.figure(figsize=(10.5, 5.2), dpi=100)
            fig.suptitle(
                "ACP Demo: Virtual Target + Adaptive Stiffness",
                fontsize=12,
                fontweight="bold",
            )
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.plot(t[: i + 1], force_norm[: i + 1], "C0", lw=2, label="|f|")
            ax1.plot(t[: i + 1], k_series[: i + 1] / 100.0, "C3", lw=2, label="k/100")
            ax1.axvline(t[contact_start], color="gray", ls="--", alpha=0.6)
            ax1.set_xlim(t[0], t[-1])
            ax1.set_ylim(0, max(force_norm.max(), k_series.max() / 100.0) * 1.1)
            ax1.set_xlabel("time (s)")
            ax1.set_title("Force & stiffness")
            ax1.legend(loc="upper left", fontsize=8)
            ax1.grid(alpha=0.3)

            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
            ax2.plot(
                pose7[: i + 1, 0],
                pose7[: i + 1, 1],
                pose7[: i + 1, 2],
                "r-",
                lw=2,
                label="x_ref",
            )
            ax2.plot(
                virtual_pose7[: i + 1, 0],
                virtual_pose7[: i + 1, 1],
                virtual_pose7[: i + 1, 2],
                "b--",
                lw=2,
                label="x_virt",
            )
            ax2.set_title("Reference vs virtual target")
            ax2.legend(fontsize=8)
            fig.tight_layout()
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            # even dims for yuv420p
            h, w = bgr.shape[:2]
            if h % 2 or w % 2:
                bgr = cv2.resize(bgr, (w - w % 2, h - h % 2))
            cv2.imwrite(str(tmp_root / f"frame_{frame_i:05d}.png"), bgr)
            plt.close(fig)
            frame_i += 1
            _ = cm  # keep import used for parity with demo

        _encode_mp4(tmp_root, "frame_%05d.png", out_mp4, fps=fps)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        shutil.rmtree(tmp_mod.parent, ignore_errors=True)

    return {
        "out": str(out_mp4),
        "png": str(out_png),
        "frames": frame_i,
        "bytes": out_mp4.stat().st_size if out_mp4.is_file() else 0,
    }





def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_REPO / "docs" / "media"),
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.environ.get(
            "ACP_SIM_FT_CKPT",
            str(
                Path.home()
                / "training_outputs/2026.07.24_16.16.52_flip_up_sim_flip_sim_ft/checkpoints/latest.ckpt"
            ),
        ),
    )
    parser.add_argument("--skip-flip", action="store_true", help="跳过 GitHub 原版 v2-ft 录屏")
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="跳过 v2-ft-live UI 录屏（make_github_media.sh 默认带此开关）",
    )
    parser.add_argument(
        "--also-live",
        action="store_true",
        help="额外录 v2-ft-live UI 媒体（与主线同权重，仅窗口不同）",
    )
    parser.add_argument(
        "--only-live",
        action="store_true",
        help="只录 v2-ft-live（等同 --skip-flip --also-live）",
    )
    parser.add_argument(
        "--demo-png",
        action="store_true",
        help="仅刷新 virtual_target_stiffness_demo.png（不生成 Demo gif/mp4）",
    )
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--panel-width", type=int, default=420)
    parser.add_argument("--k-hard", type=float, default=5000.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[media] out_dir={out_dir}")

    results = {}
    if args.demo_png:
        print("[media] refresh stiffness demo PNG only…")
        results["demo"] = record_stiffness_demo(
            out_dir / "_unused_demo.mp4",
            out_dir / "virtual_target_stiffness_demo.png",
        )
        unused = out_dir / "_unused_demo.mp4"
        if unused.is_file():
            unused.unlink()
        print(" ", {"png": results["demo"]["png"]})

    skip_flip = bool(args.skip_flip or args.only_live)
    skip_live = bool(args.skip_live) and not bool(args.also_live or args.only_live)
    if args.also_live or args.only_live:
        skip_live = False
    ckpt = Path(args.ckpt)

    if not skip_flip:
        if not ckpt.is_file():
            print(f"[error] missing ckpt: {ckpt}")
            return 1
        print(f"[media] recording v2-ft (GitHub 原版分屏)… ckpt={ckpt}")
        results["flip"] = record_sim_flip(
            out_dir / "sim_flip_v2ft.mp4",
            ckpt,
            width=int(args.width),
            height=int(args.height),
            panel_width=int(args.panel_width),
            k_hard=float(args.k_hard),
            stiffness_png=out_dir / "sim_flip_v2ft_stiffness.png",
            hud_title="ACP v2-ft trimodal",
        )
        print(" ", results["flip"])
        if results["flip"]["max_tilt_deg"] < 85.0:
            print("[warn] max tilt < 85° — cube may not look fully flipped")
        if not args.no_gif:
            g = maybe_make_gif(
                out_dir / "sim_flip_v2ft.mp4",
                out_dir / "sim_flip_v2ft.gif",
                scale=720,
                fps=8,
            )
            print(f"  gif={g}")

    if not skip_live:
        if not ckpt.is_file():
            print(f"[error] missing ckpt: {ckpt}")
            return 1
        print(f"[media] recording v2-ft-live (增强分屏：6:4 + 接触阶段 + 超采样)… ckpt={ckpt}")
        results["live"] = record_sim_flip(
            out_dir / "sim_flip_v2ft_live.mp4",
            ckpt,
            width=600,
            height=480,
            panel_width=400,
            k_hard=float(args.k_hard),
            stiffness_png=None,
            overview_zoom=1.22,
            render_scale=2.5,
            overview_distance=1.12,
            panel_render_scale=2.0,
            hud_title="ACP v2-ft-live",
        )
        print(" ", results["live"])
        if results["live"]["max_tilt_deg"] < 85.0:
            print("[warn] live max tilt < 85° — cube may not look fully flipped")
        if not args.no_gif:
            g = maybe_make_gif(
                out_dir / "sim_flip_v2ft_live.mp4",
                out_dir / "sim_flip_v2ft_live.gif",
                scale=720,
                fps=8,
            )
            print(f"  gif={g}")

    print("[media] done")
    for p in sorted(out_dir.glob("*")):
        if p.is_file():
            print(f"  {p.name:40s} {p.stat().st_size/1e6:6.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
