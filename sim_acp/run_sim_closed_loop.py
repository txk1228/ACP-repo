"""闭环入口：蓝球 / i7；方案 A（力对齐 K）或 Flip Spec。

要看清 ACP 柔顺效果，优先：
  python -m sim_acp.run_acp_effect_demo [--render]
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
    parser = argparse.ArgumentParser(description="ACP MuJoCo closed-loop")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--k-low", type=float, default=500.0)
    parser.add_argument("--k-high", type=float, default=2500.0)
    parser.add_argument("--force", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--i7", action="store_true", help="i7 URDF + 右臂 IK")
    parser.add_argument("--policy", action="store_true", help="Flip Spec 策略网")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--policy-every", type=int, default=200)
    parser.add_argument("--fake-rgb", action="store_true")
    parser.add_argument(
        "--contact",
        action="store_true",
        help="ACP 效果：x_ref 压向桌下，跟踪力对齐 x_virt（推荐改用 run_acp_effect_demo）",
    )
    args = parser.parse_args()

    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("请先: pip install mujoco")
        return 1

    import numpy as np

    from sim_acp.bridge.rgb_buffer import RgbRingBuffer
    from sim_acp.bridge.virtual_target import virtual_target_pos
    from sim_acp.bridge.wrench_buffer import WrenchRingBuffer

    if args.contact and not args.i7:
        print("--contact 需要同时加 --i7（或直接: python -m sim_acp.run_acp_effect_demo）")
        return 2

    if args.i7:
        from sim_acp.bridge.i7_mujoco_backend import I7MujocoBackend

        backend = I7MujocoBackend(render=args.render)
        p0 = backend.read_state().pose_xyzquat[:3]
        clip_lo = p0 + np.array([-0.15, -0.15, -0.12])
        clip_hi = p0 + np.array([0.15, 0.15, 0.12])
        model_name = "i7 right arm (IK)"
        if args.contact:
            model_name = "i7 ACP effect (force-aligned K)"
    else:
        from sim_acp.bridge.mujoco_backend import MujocoSimBackend

        backend = MujocoSimBackend(render=args.render)
        clip_lo = np.array([0.1, -0.4, 0.15])
        clip_hi = np.array([0.8, 0.4, 0.7])
        model_name = "ee sphere"

    buf = WrenchRingBuffer(capacity=8000)
    rgb_buf = RgbRingBuffer(capacity=8, h=224, w=224)
    st0 = backend.read_state()
    x_ref = st0.pose_xyzquat[:3].copy()
    x_virt = x_ref.copy()
    k_low = float(args.k_low)
    k_high = float(args.k_high)
    force = np.array(args.force, dtype=float)
    p_touch = x_ref.copy()
    z_surface = float(x_ref[2])

    if args.contact:
        z_surface = float(backend.suggest_press_target()[2])
        print(
            f"  --contact ACP: x_ref → below table; track x_virt "
            f"(k_low={k_low} along force, k_high={k_high})"
        )
        print(f"  table_top_z={backend.table_top_z():.3f} z_surface≈{z_surface:.3f}")
        for j in range(500):
            a = min(1.0, j / 400.0)
            x = st0.pose_xyzquat[:3] * (1 - a) + np.array(
                [st0.pose_xyzquat[0], st0.pose_xyzquat[1], z_surface]
            ) * a
            backend.inject_force_W(np.zeros(3))
            backend.write_ee_pose(np.array([*x, 0, 0, 0, 1], dtype=float), 0)
            backend.step(realtime=False)
        p_touch = backend.read_state().pose_xyzquat[:3].copy()

    runner = None
    if args.policy:
        from sim_acp.bridge.policy_runner import FlipSpecPolicyRunner

        runner = FlipSpecPolicyRunner(ckpt_path=args.ckpt)
        for _ in range(runner.wrench_h):
            buf.push(np.zeros(6))
        for _ in range(max(2, runner.rgb_h)):
            backend.step(realtime=False)
            rgb_buf.push(backend.render_rgb())

    use_cam = args.policy and not args.fake_rgb
    print(f"closed-loop model={model_name}")
    if args.policy:
        print(
            "  --policy: Flip Spec；RGB="
            + ("MuJoCo 相机" if use_cam else "假灰图")
        )
    if args.render:
        print("  --render: ≈ %.0f ms/步" % (backend.model.opt.timestep * 1000))

    try:
        for i in range(args.steps):
            if args.contact:
                backend.inject_force_W(np.zeros(3))
                phase = i / max(1, args.steps - 1)
                z_ref = z_surface - 0.04 * min(1.0, phase / 0.25)
                x_ref = np.array(
                    [
                        p_touch[0] + 0.03 * math.sin(2.0 * math.pi * phase * 2.0),
                        p_touch[1],
                        z_ref,
                    ],
                    dtype=float,
                )
            elif not args.policy:
                if 1000 <= i < 2000:
                    f = (
                        force
                        if np.linalg.norm(force) > 0
                        else np.array([20.0, 0.0, 0.0])
                    )
                    backend.inject_force_W(f)
                else:
                    backend.inject_force_W(np.zeros(3))
            else:
                backend.inject_force_W(np.zeros(3))

            st = backend.read_state()
            buf.push(st.wrench_W)
            if use_cam and (i % 5 == 0):
                rgb_buf.push(backend.render_rgb())

            if runner is not None and (i % max(1, args.policy_every) == 0):
                rgb = None if args.fake_rgb else rgb_buf.stack_last(runner.rgb_h)
                act = runner.predict(
                    st.pose_xyzquat,
                    buf.window(runner.wrench_h),
                    rgb_uint8=rgb,
                    fake_rgb=bool(args.fake_rgb),
                    force_xyz=st.wrench_W[:3],
                )
                x_ref = act.x_ref_pos
                x_virt = act.x_virt_pos
                k_low = act.k_low
                print(
                    f"[policy i={i}] infer={act.inference_s:.2f}s "
                    f"x_ref={x_ref.round(3)} x_virt={x_virt.round(3)} k={k_low:.1f}"
                )
            elif runner is None:
                f = st.wrench_W[:3].copy()
                if args.contact:
                    f = np.array([0.0, 0.0, max(0.0, float(f[2]))])
                x_virt = virtual_target_pos(
                    x_ref, f, k_low=k_low, k_high=k_high
                )

            x_virt = np.clip(x_virt, clip_lo, clip_hi)
            backend.write_ee_pose(
                np.array(
                    [x_virt[0], x_virt[1], x_virt[2], 0, 0, 0, 1], dtype=float
                ),
                st.timestamp_ns,
            )
            if not backend.step():
                print("viewer 已关闭，提前结束")
                break
            if runner is None and i % 500 == 0:
                ee = backend.read_state().pose_xyzquat[:3]
                print(
                    f"i={i} x_ref={x_ref.round(3)} x_virt={x_virt.round(3)} "
                    f"ee={ee.round(3)} f={st.wrench_W[:3].round(1)}"
                )
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("done")
        try:
            backend.close()
        except Exception as exc:
            print(f"[warn] close: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
