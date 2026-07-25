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
    cmd = [
        "ffmpeg",
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


def _overlay_pip(overview_bgr: np.ndarray, wrist_rgb: np.ndarray) -> np.ndarray:
    """右下角叠腕部 RGB + 标题条。"""
    h, w = overview_bgr.shape[:2]
    pip_w = max(160, w // 4)
    pip_h = max(160, h // 4)
    wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
    pip = cv2.resize(wrist_bgr, (pip_w, pip_h), interpolation=cv2.INTER_AREA)
    x0 = w - pip_w - 16
    y0 = h - pip_h - 16
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


def _draw_hud(frame_bgr: np.ndarray, tilt_deg: float, step: int, k_low: float | None) -> None:
    cv2.rectangle(frame_bgr, (0, 0), (frame_bgr.shape[1], 36), (20, 20, 20), -1)
    label = f"ACP v2-ft trimodal  tilt={tilt_deg:5.1f}deg  step={step}"
    if k_low is not None:
        label += f"  k={k_low:.0f}N/m"
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


def record_sim_flip(
    out_mp4: Path,
    ckpt: Path,
    *,
    width: int = 640,
    height: int = 360,
    fps: int = 15,
    frame_every: int = 8,
    steps: int = 3600,
    exec_horizon: int = 12,
    action_ds: int = 50,
) -> dict:
    from sim_acp.bridge.i7_mujoco_backend import I7MujocoBackend
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
    k_show: float | None = None
    frame_i = 0

    tmp_root = Path(tempfile.mkdtemp(prefix="acp_media_flip_"))
    try:
        def _capture() -> None:
            nonlocal frame_i, max_tilt
            overview = backend.render_overview_rgb(width=width, height=height)
            wrist = backend.render_rgb()
            tilt = math.degrees(backend.cube_tilt_rad())
            max_tilt = max(max_tilt, backend.cube_tilt_rad())
            bgr = cv2.cvtColor(overview, cv2.COLOR_RGB2BGR)
            bgr = _overlay_pip(bgr, wrist)
            _draw_hud(bgr, tilt, i_step, k_show)
            cv2.imwrite(str(tmp_root / f"frame_{frame_i:05d}.png"), bgr)
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
                f"tilt={math.degrees(max_tilt):.1f}deg frames={frame_i}"
            )
            for h in range(H):
                wp = np.clip(virt_traj[h], clip_lo, clip_hi)
                x_ref = ref_traj[min(h, len(ref_traj) - 1)]
                k_show = float(k_traj[min(h, len(k_traj) - 1)])
                soft_axis_from_policy(
                    x_ref, wp, force_xyz=backend.read_state().wrench_W[:3]
                )
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
                    tilt = backend.cube_tilt_rad()
                    max_tilt = max(max_tilt, tilt)
                    if i_step % frame_every == 0:
                        _capture()
                    i_step += 1
                    if tilt >= math.radians(70.0):
                        i_step = steps
                        break
                tip_cur = wp.copy()
                if i_step >= steps:
                    break
        _capture()
        _encode_mp4(tmp_root, "frame_%05d.png", out_mp4, fps=fps)
    finally:
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


def maybe_make_gif(mp4: Path, gif: Path, *, scale: int = 480, fps: int = 10) -> Path | None:
    if not mp4.is_file():
        return None
    gif.parent.mkdir(parents=True, exist_ok=True)
    # palette GIF，体积可控，适合 README 内嵌预览
    cmd = [
        "ffmpeg",
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
    parser.add_argument("--skip-flip", action="store_true")
    parser.add_argument("--skip-demo", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[media] out_dir={out_dir}")

    results = {}
    if not args.skip_demo:
        print("[media] recording stiffness demo…")
        results["demo"] = record_stiffness_demo(
            out_dir / "virtual_target_stiffness_demo.mp4",
            out_dir / "virtual_target_stiffness_demo.png",
        )
        print(" ", results["demo"])
        if not args.no_gif:
            g = maybe_make_gif(
                out_dir / "virtual_target_stiffness_demo.mp4",
                out_dir / "virtual_target_stiffness_demo.gif",
                scale=560,
                fps=8,
            )
            print(f"  gif={g}")

    if not args.skip_flip:
        ckpt = Path(args.ckpt)
        if not ckpt.is_file():
            print(f"[error] missing ckpt: {ckpt}")
            return 1
        print(f"[media] recording v2-ft flip… ckpt={ckpt}")
        # EGL 无头更稳；若失败可改 MUJOCO_GL=glfw + DISPLAY
        results["flip"] = record_sim_flip(
            out_dir / "sim_flip_v2ft.mp4",
            ckpt,
            width=int(args.width),
            height=int(args.height),
        )
        print(" ", results["flip"])
        if results["flip"]["max_tilt_deg"] < 55.0:
            print("[warn] tilt < 55° — video still saved, check ckpt/env")
        if not args.no_gif:
            g = maybe_make_gif(
                out_dir / "sim_flip_v2ft.mp4",
                out_dir / "sim_flip_v2ft.gif",
                scale=480,
                fps=10,
            )
            print(f"  gif={g}")

    print("[media] done")
    for p in sorted(out_dir.glob("*")):
        if p.is_file():
            print(f"  {p.name:40s} {p.stat().st_size/1e6:6.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
