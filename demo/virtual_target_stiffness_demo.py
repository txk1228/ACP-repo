#!/usr/bin/env python3
"""
ACP 核心算法独立 Demo（无需 GPU / 数据集）

演示内容：
  1. 合成接触场景（接近 → 碰撞 → 持续推压）
  2. VirtualTargetEstimator 计算虚拟目标位姿与刚度幅值 k
  3. 由力方向重构 6×6 笛卡尔刚度矩阵 K
  4. 生成可视化图表

用法：
  conda activate acp-demo
  export PYTHONNOUSERSITE=1
  python demo/virtual_target_stiffness_demo.py
  python demo/virtual_target_stiffness_demo.py --save output/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 避免 ~/.local 旧包污染（常见：系统 python + 用户 site 的 matplotlib/numpy 冲突）
os.environ.setdefault("PYTHONNOUSERSITE", "1")

# 引用 ACP 官方核心模块
ROOT = Path(__file__).resolve().parents[1]
ACP_SRC = ROOT / "adaptive_compliance_policy"
sys.path.insert(0, str(ACP_SRC))


def _check_runtime() -> None:
    """在依赖导入前给出可操作的环境提示。"""
    exe = Path(sys.executable).resolve()
    expected = Path.home() / "miniconda3/envs/acp-demo/bin/python"
    if "acp-demo" not in str(exe):
        print(
            "错误：当前 Python 不是 acp-demo 环境。\n"
            f"  当前: {exe}\n"
            f"  期望: {expected}\n\n"
            "请任选其一：\n"
            "  1) Cursor 右下角选择解释器 → acp-demo\n"
            "  2) 终端执行: conda activate acp-demo && bash scripts/run_demo.sh\n"
            "  3) 直接用: ~/miniconda3/envs/acp-demo/bin/python "
            "demo/virtual_target_stiffness_demo.py --save demo/output\n",
            file=sys.stderr,
        )
        sys.exit(1)


_check_runtime()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from spatialmath import SE3
from spatialmath.base import q2r, r2q

from PyriteUtility.planning_control import compliance_helpers as ch  # noqa: E402
from PyriteUtility.spatial_math import spatial_utilities as su  # noqa: E402

# 与官方 postprocess_add_virtual_target_label.py 一致的默认参数
STIFFNESS_PARA = {
    "k_max": 5000.0,
    "k_min": 200.0,
    "f_low": 0.5,
    "f_high": 5.0,
    "dim": 3,
    "characteristic_length": 0.02,
    "vel_tol": 999.0,
}


def synthesize_contact_episode(n_steps: int = 400, dt: float = 0.01):
    """合成单臂沿 X 轴运动、Z 向接触推压的示教轨迹。"""
    t = np.arange(n_steps) * dt

    # 参考轨迹：X 方向匀速靠近"墙面" z=0.15
    x = np.linspace(0.30, 0.42, n_steps)
    y = np.full(n_steps, 0.0)
    z = np.full(n_steps, 0.15)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n_steps, 1))  # w,x,y,z 无旋转
    pose7 = np.hstack([x[:, None], y[:, None], z[:, None], quat])

    # 六维力：接触前为零；接触后 Z 向力线性上升至 12N（工具坐标系）
    contact_start = int(n_steps * 0.55)
    wrench = np.zeros((n_steps, 6))
    for i in range(contact_start, n_steps):
        phase = (i - contact_start) / max(1, n_steps - contact_start - 1)
        wrench[i, 2] = -12.0 * phase  # 推压为负 Z
        # 轻微摩擦
        wrench[i, 0] = 1.5 * phase

    return t, pose7, wrench, contact_start


def build_stiffness_matrix_6d(
    compliance_direction_tool: np.ndarray,
    k_low: float,
    k_high: float = 5000.0,
    k_rot: float = 100.0,
) -> np.ndarray:
    """由柔顺方向与 k_low 重构 6×6 刚度矩阵（与 virtual_target_real_env_runner 一致）。"""
    d = compliance_direction_tool.reshape(3).astype(float)
    if np.linalg.norm(d) < 1e-6:
        d = np.array([1.0, 0.0, 0.0])
    d /= np.linalg.norm(d)

    X = d
    Y = np.cross(X, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(Y) < 1e-6:
        Y = np.cross(X, np.array([0.0, 1.0, 0.0]))
    Y /= np.linalg.norm(Y)
    Z = np.cross(X, Y)

    M = np.diag([k_low, k_high, k_high])
    S = np.column_stack([X, Y, Z])
    K_trans = S @ M @ np.linalg.inv(S)

    K = np.eye(6) * k_rot
    K[:3, :3] = K_trans
    return K


def run_virtual_target_pipeline(pose7: np.ndarray, wrench: np.ndarray):
    """逐帧运行 VirtualTargetEstimator，返回标签序列。"""
    pe = ch.VirtualTargetEstimator(**STIFFNESS_PARA)
    n = len(pose7)

    k_series = np.zeros(n)
    virtual_pose7 = np.zeros_like(pose7)
    offsets = np.zeros((n, 3))
    adjusted = np.zeros(n, dtype=bool)

    half_window = 10
    for i in range(n):
        SE3_WT = SE3.Rt(q2r(pose7[i, 3:7]), pose7[i, :3], check=False)
        wrench_T = wrench[i]

        i0 = max(0, i - half_window)
        i1 = min(n - 1, i + half_window)
        SE3_start = su.pose7_to_SE3(pose7[i0])
        SE3_end = su.pose7_to_SE3(pose7[i1])
        twist = su.SE3_to_spt(su.SE3_inv(SE3_start) @ SE3_end)

        k, pos_TC, flag_adj = pe.update(wrench_T, twist)
        SE3_WC = SE3_WT * SE3.Rt(np.eye(3), pos_TC)

        virtual_pose7[i] = np.concatenate([SE3_WC.t, r2q(SE3_WC.R)])
        k_series[i] = k
        offsets[i] = pos_TC
        adjusted[i] = flag_adj

    return k_series, virtual_pose7, offsets, adjusted


def plot_results(
    t,
    pose7,
    virtual_pose7,
    wrench,
    k_series,
    offsets,
    contact_start,
    save_dir: Path | None,
):
    """生成四联图并保存。"""
    force_norm = np.linalg.norm(wrench[:, :3], axis=1)

    # 取最大接触力时刻重构刚度矩阵
    idx_peak = int(np.argmax(force_norm))
    d_tool = offsets[idx_peak]
    if np.linalg.norm(d_tool) < 1e-6:
        d_tool = -wrench[idx_peak, :3] / max(force_norm[idx_peak], 1e-6)
    K6 = build_stiffness_matrix_6d(d_tool, k_series[idx_peak])

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "ACP Core Demo: Virtual Target + Adaptive Stiffness\n"
        "(synthetic contact, no GPU / no dataset)",
        fontsize=13,
        fontweight="bold",
    )

    # 1) 力与刚度时序
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(t, force_norm, "C0", lw=2, label="|f| (N)")
    ax1.axvline(t[contact_start], color="gray", ls="--", alpha=0.7, label="contact")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("force (N)")
    ax1.set_title("Contact force")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax1b = ax1.twinx()
    ax1b.plot(t, k_series, "C3", lw=2, alpha=0.85, label="k (N/m)")
    ax1b.set_ylabel("stiffness k (N/m)")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper left")

    # 2) 参考轨迹 vs 虚拟目标（3D）
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2.plot(
        pose7[:, 0], pose7[:, 1], pose7[:, 2],
        "r-", lw=2, label="x_ref (reference)",
    )
    ax2.plot(
        virtual_pose7[:, 0], virtual_pose7[:, 1], virtual_pose7[:, 2],
        "b--", lw=2, label="x_virt (virtual target)",
    )
    ax2.scatter(
        pose7[contact_start, 0], pose7[contact_start, 1], pose7[contact_start, 2],
        c="orange", s=60, label="contact onset",
    )
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.set_title("Reference vs virtual target trajectory")
    ax2.legend(fontsize=8)

    # 3) 虚拟目标偏移量（= K^{-1} f 的几何体现）
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t, offsets[:, 0] * 1000, label="offset X (mm)")
    ax3.plot(t, offsets[:, 1] * 1000, label="offset Y (mm)")
    ax3.plot(t, offsets[:, 2] * 1000, label="offset Z (mm)")
    ax3.axvline(t[contact_start], color="gray", ls="--", alpha=0.7)
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("tool-frame offset (mm)")
    ax3.set_title("Virtual target offset (tool frame)")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4) 刚度矩阵热力图 + 特征值
    ax4 = fig.add_subplot(2, 2, 4)
    vmax = max(np.abs(K6).max(), 1.0)
    im = ax4.imshow(K6, cmap=cm.RdYlBu_r, vmin=-vmax, vmax=vmax)
    ax4.set_xticks(range(6))
    ax4.set_yticks(range(6))
    ax4.set_xticklabels(["fx", "fy", "fz", "tx", "ty", "tz"])
    ax4.set_yticklabels(["dx", "dy", "dz", "drx", "dry", "drz"])
    ax4.set_title(
        f"Stiffness matrix K at peak force\n"
        f"k_low={k_series[idx_peak]:.0f} N/m, t={t[idx_peak]:.2f}s"
    )
    fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    eigvals = np.linalg.eigvalsh(K6[:3, :3])
    inset = f"K_trans eigenvalues: [{eigvals[0]:.0f}, {eigvals[1]:.0f}, {eigvals[2]:.0f}] N/m"
    ax4.text(0.02, -0.18, inset, transform=ax4.transAxes, fontsize=9)

    plt.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "acp_virtual_target_stiffness_demo.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[saved] {out}")

        summary = save_dir / "demo_summary.txt"
        summary.write_text(
            f"ACP Virtual Target + Stiffness Demo\n"
            f"===================================\n"
            f"Steps: {len(t)}\n"
            f"Contact start: t={t[contact_start]:.3f}s\n"
            f"Peak force: {force_norm.max():.2f} N at t={t[idx_peak]:.3f}s\n"
            f"Stiffness range: [{k_series.min():.0f}, {k_series.max():.0f}] N/m\n"
            f"At peak: k_low={k_series[idx_peak]:.0f} N/m\n"
            f"K_trans eigenvalues: {eigvals}\n"
            f"\nCore formula:\n"
            f"  u = f / ||f||\n"
            f"  K = S @ diag(k_low, k_high, k_high) @ S^-1\n"
            f"  x_virt = x_ref + K^-1 @ f\n",
            encoding="utf-8",
        )
        print(f"[saved] {summary}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="ACP virtual target + stiffness demo")
    parser.add_argument(
        "--save", type=str, default=str(ROOT / "demo" / "output"),
        help="output directory for figures",
    )
    parser.add_argument("--no-show", action="store_true", help="do not open GUI window")
    args = parser.parse_args()

    print("=" * 60)
    print("ACP 核心算法 Demo — 虚拟目标 + 变刚度")
    print("=" * 60)

    t, pose7, wrench, contact_start = synthesize_contact_episode()
    print(f"[1/3] 合成轨迹: {len(t)} 帧, 接触起始 t={t[contact_start]:.2f}s")

    k_series, virtual_pose7, offsets, _ = run_virtual_target_pipeline(pose7, wrench)
    print(
        f"[2/3] VirtualTargetEstimator 完成: "
        f"k ∈ [{k_series.min():.0f}, {k_series.max():.0f}] N/m"
    )

    save_dir = Path(args.save) if args.save else None
    fig = plot_results(
        t, pose7, virtual_pose7, wrench, k_series, offsets,
        contact_start, save_dir,
    )
    print(f"[3/3] 可视化完成")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)

    print("\n关键结论:")
    print("  - 无接触时 k ≈ k_max，虚拟目标 ≈ 参考轨迹")
    print("  - 接触力增大时 k 降至 k_min，虚拟目标沿力方向退让")
    print("  - 柔顺主轴方向刚度最低，其余方向保持高刚度")


if __name__ == "__main__":
    main()
