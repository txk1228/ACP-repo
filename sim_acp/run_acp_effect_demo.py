"""ACP 效果演示（方案 A）：力方向软、正交方向硬。

用「已知注入力」演示（不依赖穿桌接触），这样效果不被 IK/穿模钳制淹没：

  x_ref 在 XY 扫描（正交方向应跟准）
  中段注入 +Z 力 → x_virt 只沿 Z 抬起 |f|/k_low
  EE 跟踪 x_virt

用法：
  python -m sim_acp.run_acp_effect_demo
  python -m sim_acp.run_acp_effect_demo --render
  python -m sim_acp.run_acp_effect_demo --ball
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description="ACP effect demo (Scheme A)")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--ball", action="store_true", help="蓝球后端")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--k-low", type=float, default=500.0)
    parser.add_argument("--k-high", type=float, default=2500.0)
    parser.add_argument("--fz", type=float, default=20.0, help="中段注入的法向力 (N)")
    args = parser.parse_args()

    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("请先: pip install mujoco")
        return 1

    import numpy as np

    from sim_acp.bridge.virtual_target import (
        offset_along_force_ratio,
        virtual_target_pos,
    )

    if args.ball:
        from sim_acp.bridge.mujoco_backend import MujocoSimBackend

        backend = MujocoSimBackend(render=args.render)
        model_name = "ee sphere"
    else:
        from sim_acp.bridge.i7_mujoco_backend import I7MujocoBackend

        backend = I7MujocoBackend(render=args.render)
        model_name = "i7 right arm"

    k_low = float(args.k_low)
    k_high = float(args.k_high)
    fz = float(args.fz)
    p0 = backend.read_state().pose_xyzquat[:3].copy()

    print("=" * 60)
    print("ACP 效果演示（方案 A = 力对齐变刚度，非神经网络）")
    print(f"  model: {model_name}")
    print(f"  k_low={k_low} N/m（力方向软）  k_high={k_high}（正交硬）")
    print(f"  中段注入 f=[0,0,{fz}] → 期望 Δz = fz/k_low = {fz/k_low:.4f} m")
    print(f"  全程 XY 扫描：x_virt.xy 应等于 x_ref.xy（正交保精度）")
    print("=" * 60)

    n = int(args.steps)
    hist = []
    for i in range(n):
        phase = i / max(1, n - 1)
        # 参考：XY 扫椭圆，高度保持
        x_ref = np.array(
            [
                p0[0] + 0.04 * math.sin(2.0 * math.pi * phase * 2.0),
                p0[1] + 0.02 * math.cos(2.0 * math.pi * phase * 2.0),
                p0[2],
            ],
            dtype=float,
        )
        # 中段 25%–75% 注入力
        if 0.25 <= phase <= 0.75:
            f_cmd = np.array([0.0, 0.0, fz])
        else:
            f_cmd = np.zeros(3)
        backend.inject_force_W(f_cmd)

        st = backend.read_state()
        f = st.wrench_W[:3].copy()
        x_virt = virtual_target_pos(x_ref, f, k_low=k_low, k_high=k_high)

        backend.write_ee_pose(
            np.array([x_virt[0], x_virt[1], x_virt[2], 0, 0, 0, 1], dtype=float),
            st.timestamp_ns,
        )
        # 注入力演示：暂时关闭 tip 穿模抬升（否则会盖住柔顺偏移）
        if hasattr(backend, "_max_tip_penetration"):
            backend._max_tip_penetration = 1.0  # 实质禁用

        if not backend.step():
            print("viewer 已关闭")
            break

        ee = backend.read_state().pose_xyzquat[:3]
        hist.append(
            {
                "x_ref": x_ref.copy(),
                "x_virt": x_virt.copy(),
                "ee": ee.copy(),
                "f": f.copy(),
                "phase": phase,
                "align": offset_along_force_ratio(x_ref, x_virt, f),
            }
        )
        if i % 500 == 0:
            print(
                f"i={i} phase={phase:.2f} f_z={f[2]:.1f} "
                f"Δz={(x_virt[2]-x_ref[2]):.4f} "
                f"|Δxy|={np.linalg.norm(x_virt[:2]-x_ref[:2]):.5f} "
                f"ee.z={ee[2]:.3f}"
            )

    backend.close()
    if len(hist) < 20:
        print("FAIL: too few steps")
        return 2

    mid = [h for h in hist if 0.35 <= h["phase"] <= 0.65]
    free = [h for h in hist if h["phase"] < 0.2 or h["phase"] > 0.8]
    dz_mid = np.array([h["x_virt"][2] - h["x_ref"][2] for h in mid])
    xy_mid = np.array(
        [np.linalg.norm(h["x_virt"][:2] - h["x_ref"][:2]) for h in mid]
    )
    xy_free = np.array(
        [np.linalg.norm(h["x_virt"][:2] - h["x_ref"][:2]) for h in free]
    )
    expect_dz = fz / k_low
    dz_mean = float(np.mean(dz_mid)) if len(mid) else 0.0
    xy_mid_m = float(np.mean(xy_mid)) if len(mid) else 0.0
    align_m = float(np.mean([h["align"] for h in mid])) if mid else 0.0

    print("-" * 60)
    print("ACP 效果验收（方案 A）")
    print(f"  期望 Δz = {expect_dz*1000:.1f} mm")
    print(f"  实测 mean Δz (有力段) = {dz_mean*1000:.1f} mm")
    print(f"  实测 mean |Δxy| (有力段) = {xy_mid_m*1000:.2f} mm  (应 ≈ 0)")
    print(f"  force-align ratio = {align_m:.3f}")
    if free:
        print(f"  mean |Δxy| (无力段) = {float(np.mean(xy_free))*1000:.2f} mm")

    ok_dz = abs(dz_mean - expect_dz) < 0.003
    ok_xy = xy_mid_m < 1e-4
    ok_align = align_m > 0.99
    ok = ok_dz and ok_xy and ok_align
    print("  PASS" if ok else "  FAIL")
    if not ok_dz:
        print("  reason: Δz != |f|/k_low along force")
    if not ok_xy:
        print("  reason: lateral virtual offset (stiff axes broken)")
    if not ok_align:
        print("  reason: offset not along force")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
