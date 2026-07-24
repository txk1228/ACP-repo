"""变刚度仿真日志 → 图。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


def save_compliance_plots(
    hist: Sequence[dict[str, Any]],
    out_path: str | Path,
    *,
    k_low: float,
    k_high: float,
    title: str = "Scheme A flip",
    delta_clip_m: Optional[float] = None,
) -> Path:
    """画翻方块 / 方案 A 过程中的力、偏移、软轴、有效刚度。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not hist:
        raise ValueError("empty hist")

    t = np.arange(len(hist))
    f = np.stack([h["f"] for h in hist])
    has_filt = "f_filt" in hist[0]
    f_use = np.stack([h["f_filt"] for h in hist]) if has_filt else f
    xref = np.stack([h["x_ref"] for h in hist])
    xvirt = np.stack([h["x_virt"] for h in hist])
    xtip = np.stack([h["x_tip"] for h in hist])
    fn_raw = np.linalg.norm(f, axis=1)
    fn = np.linalg.norm(f_use, axis=1)
    delta = xvirt - xref
    dn = np.linalg.norm(delta, axis=1)
    # k_eff 用驱动 x_virt 的力（滤波力）对 |Δ|
    k_eff = np.full_like(fn, np.nan)
    mask = (fn > 0.5) & (dn > 1e-5)
    k_eff[mask] = fn[mask] / dn[mask]

    if "soft_axis" in hist[0]:
        soft_dir = np.stack([h["soft_axis"] for h in hist])
    else:
        soft_dir = np.zeros_like(f_use)
        m = fn > 1e-3
        soft_dir[m] = f_use[m] / fn[m, None]

    tilt_deg = np.array([h.get("tilt_deg", 0.0) for h in hist], dtype=float)
    if "k_low" in hist[0]:
        k_cmd = np.array([float(h["k_low"]) for h in hist], dtype=float)
    else:
        k_cmd = np.full(len(hist), float(k_low), dtype=float)
    dn_theory = np.where(fn > 1e-3, fn / np.maximum(k_cmd, 1.0), 0.0)
    clip = float(delta_clip_m) if delta_clip_m and delta_clip_m > 0 else None
    clipped = (
        dn_theory > (clip + 1e-9) if clip is not None else np.zeros(len(hist), dtype=bool)
    )

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    clip_txt = f"|Δ|_clip={clip*1000:.0f}mm" if clip else "no |Δ| hard-clip"
    fig.suptitle(
        f"{title}\n"
        f"k_cmd median≈{float(np.nanmedian(k_cmd)):.0f}  k_high={k_high:.0f}  "
        f"{clip_txt}  (soft axis = filtered force)",
        fontsize=11,
    )

    ax = axes[0]
    ax.plot(t, fn_raw, label="|f| raw", color="0.55", lw=1.0)
    ax.plot(t, fn, label="|f| filt" if has_filt else "|f|", color="k", lw=1.4)
    ax.plot(t, f_use[:, 0], label="fx_filt" if has_filt else "fx", lw=0.9, alpha=0.85)
    ax.plot(t, f_use[:, 1], label="fy_filt" if has_filt else "fy", lw=0.9, alpha=0.85)
    ax.plot(t, f_use[:, 2], label="fz_filt" if has_filt else "fz", lw=0.9, alpha=0.85)
    ax.set_ylabel("force (N)")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("contact wrench (world)" + (" — soft-axis uses EMA" if has_filt else ""))

    ax = axes[1]
    ax.plot(t, dn * 1000, label="|Δ| actual", lw=1.3)
    ax.plot(t, dn_theory * 1000, label="|f_filt|/k_low", lw=1.0, ls="--", alpha=0.85)
    if clip is not None:
        ax.axhline(clip * 1000, color="r", ls=":", label="clip")
    ax.set_ylabel("|x_virt − x_ref| (mm)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("compliance offset  (|Δ| should track |f|/k_low when unclipped)")

    ax = axes[2]
    ax.plot(t, k_cmd, label="k_low (cmd, scheduled)", lw=1.2)
    ax.plot(t, np.full_like(t, k_high, dtype=float), label="k_high", lw=1.0, ls="--")
    ax.plot(t, k_eff, label="k_eff = |f_filt|/|Δ|", lw=1.3, color="C3")
    if np.any(clipped):
        idx = np.where(clipped)[0]
        step = max(1, len(idx) // 40)
        ax.scatter(t[idx][::step], k_eff[idx][::step], s=12, c="red", zorder=5, label="clip")
    ax.set_ylabel("stiffness (N/m)")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title("commanded vs effective stiffness (should match without clip)")

    ax = axes[3]
    ax.plot(t, soft_dir[:, 0], label="soft ûx", lw=1.0)
    ax.plot(t, soft_dir[:, 1], label="soft ûy", lw=1.0)
    ax.plot(t, soft_dir[:, 2], label="soft ûz", lw=1.0)
    ax2 = ax.twinx()
    ax2.plot(t, tilt_deg, color="k", lw=1.4, label="cube tilt")
    ax.set_ylabel("soft-axis unit (world)")
    ax2.set_ylabel("tilt (deg)")
    ax.set_xlabel("step")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("soft axis (= filtered force dir) + cube tilt")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    out2 = out_path.with_name(out_path.stem + "_traj" + out_path.suffix)
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
    fig2.suptitle(f"{title} — tip trajectories", fontsize=11)
    ax = axes2[0]
    ax.plot(xref[:, 0], xref[:, 2], label="x_ref", lw=1.2)
    ax.plot(xvirt[:, 0], xvirt[:, 2], label="x_virt", lw=1.2)
    ax.plot(xtip[:, 0], xtip[:, 2], label="tip actual", lw=1.0, alpha=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("XZ (sweep plane)")

    ax = axes2[1]
    for name, arr, ls in (
        ("x_ref", xref, "-"),
        ("x_virt", xvirt, "--"),
        ("tip", xtip, ":"),
    ):
        ax.plot(t, arr[:, 0], ls=ls, label=f"{name}.x", alpha=0.9)
        ax.plot(t, arr[:, 2], ls=ls, label=f"{name}.z", alpha=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("pos (m)")
    ax.legend(ncol=3, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("x / z vs time")

    fig2.tight_layout()
    fig2.savefig(out2, dpi=140)
    plt.close(fig2)
    return out_path
