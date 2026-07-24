"""单臂翻方块演示：脚本轨迹 + 方案 A（力调度变刚度），可选 Flip Spec。

阶段 1（默认）：
  tip 从方块 -x 侧接近 → 压低棱边 → 沿 +x 扫过，方块绕棱翻起。
  全程方案 A：力 EMA → 按力调度 k_low → 软轴沿滤波力方向；无 |Δ| 硬钳制。

阶段 2（--policy）：
  同一场景接 Flip Spec；不宣称成功率。

用法：
  python -m sim_acp.run_flip_cube_demo
  python -m sim_acp.run_flip_cube_demo --render
  python -m sim_acp.run_flip_cube_demo --policy --policy-every 150 --steps 2500
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _lerp(a, b, t: float):
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    t = float(np.clip(t, 0.0, 1.0))
    return a * (1.0 - t) + b * t


def _make_scheme_a(args):
    from sim_acp.bridge.virtual_target import SchemeAController

    return SchemeAController(
        k_max=float(args.k_max),
        k_min=float(args.k_min),
        k_high=float(args.k_high),
        f_low=float(args.f_low),
        f_high=float(args.f_high),
        force_ema_alpha=float(args.force_ema),
        offset_ema_alpha=float(args.offset_ema),
        k_ema_alpha=float(args.k_ema),
        max_offset_step_m=float(args.max_offset_step),
        f_eps=float(args.f_eps),
        schedule=not bool(args.fixed_k),
    )


def _goto_tip(
    backend,
    tip_xyz,
    steps: int,
    scheme_a,
    hist: list | None,
    *,
    use_compliance: bool = True,
):
    """缓动到 tip 目标；可选方案 A。专家翻物默认刚性跟 ref（保证翻起）。"""
    import numpy as np

    tip0 = backend.tip_pos()
    for i in range(steps):
        a = (i + 1) / max(1, steps)
        tip_ref = _lerp(tip0, tip_xyz, a)
        backend.inject_force_W(np.zeros(3))
        st = backend.read_state()
        f = st.wrench_W[:3].copy()
        if use_compliance:
            out = scheme_a.step(tip_ref, f)
            tip_cmd = out.x_virt
            k_low = out.k_low
            f_filt = out.f_filt.copy()
            soft = out.soft_axis.copy()
        else:
            tip_cmd = tip_ref
            k_low = float(scheme_a.k_max)
            f_filt = f.copy()
            soft = np.zeros(3)
            if float(np.linalg.norm(f)) > 1e-6:
                soft = f / np.linalg.norm(f)
        backend.write_tip_pos(tip_cmd, st.timestamp_ns)
        if not backend.step(realtime=False):
            return False
        if hist is not None:
            hist.append(
                {
                    "f": f,
                    "f_filt": f_filt,
                    "x_ref": tip_ref.copy(),
                    "x_virt": tip_cmd.copy(),
                    "x_tip": backend.tip_pos().copy(),
                    "tilt_deg": math.degrees(backend.cube_tilt_rad()),
                    "k_low": k_low,
                    "soft_axis": soft,
                }
            )
    return True


def run_scripted_flip(backend, args) -> int:
    import numpy as np

    from sim_acp.bridge.i7_scene import CUBE_HALF, TIP_RADIUS

    backend.set_tip_clamp(False)
    scheme_a = _make_scheme_a(args)
    hist: list[dict] = []
    # 专家：刚性笛卡尔跟踪，避免柔顺把 tip 抬飞；力仍写入 hist 供标注
    stiff = not bool(getattr(args, "script_compliance", False))

    for _ in range(200):
        backend.inject_force_W(np.zeros(3))
        backend.step(realtime=False)

    cube0 = backend.cube_pos()
    table_z = backend.table_top_z()
    tip_clear = float(TIP_RADIUS) + 0.004
    print(
        f"[flip] cube0={cube0.round(3)} table_z={table_z:.3f} "
        f"tilt0={math.degrees(backend.cube_tilt_rad()):.1f}deg "
        f"tip_clear={tip_clear:.3f} stiff_track={stiff}"
    )

    # 已验证可翻轨迹（tip 间隙）：接近 → 下棱 → 扫过翻起
    approach = np.array(
        [
            cube0[0] - CUBE_HALF - tip_clear - 0.05,
            cube0[1],
            table_z + CUBE_HALF + 0.035,
        ]
    )
    contact = np.array(
        [
            cube0[0] - CUBE_HALF - tip_clear + 0.002,
            cube0[1],
            table_z + tip_clear + 0.004,
        ]
    )
    sweep = np.array(
        [
            cube0[0] + CUBE_HALF + 0.05,
            cube0[1],
            table_z + CUBE_HALF + 0.045,
        ]
    )
    finish = np.array(
        [
            cube0[0] + CUBE_HALF + 0.10,
            cube0[1],
            table_z + 0.10,
        ]
    )

    print("[flip] phase: approach")
    if not _goto_tip(
        backend, approach, 700, scheme_a, hist, use_compliance=not stiff
    ):
        return 2
    print("[flip] phase: contact")
    if not _goto_tip(
        backend, contact, 600, scheme_a, hist, use_compliance=not stiff
    ):
        return 2

    print("[flip] phase: settle")
    for _ in range(100):
        backend.inject_force_W(np.zeros(3))
        st = backend.read_state()
        backend.write_tip_pos(contact, st.timestamp_ns)
        if not backend.step(realtime=False):
            return 2

    print("[flip] phase: sweep")
    max_tilt = backend.cube_tilt_rad()
    n_sweep = int(args.sweep_steps)
    tip_start = backend.tip_pos()
    for i in range(n_sweep):
        a = (i + 1) / max(1, n_sweep)
        if a < 0.65:
            tip_ref = _lerp(tip_start, sweep, a / 0.65)
        else:
            tip_ref = _lerp(sweep, finish, (a - 0.65) / 0.35)
        backend.inject_force_W(np.zeros(3))
        st = backend.read_state()
        f = st.wrench_W[:3].copy()
        if stiff:
            tip_cmd = tip_ref
            k_low = float(scheme_a.k_max)
            f_filt = f
            soft = f / (np.linalg.norm(f) + 1e-9)
        else:
            out = scheme_a.step(tip_ref, f)
            tip_cmd, k_low, f_filt, soft = (
                out.x_virt,
                out.k_low,
                out.f_filt,
                out.soft_axis,
            )
        backend.write_tip_pos(tip_cmd, st.timestamp_ns)
        if not backend.step(realtime=False):
            return 2
        tilt = backend.cube_tilt_rad()
        max_tilt = max(max_tilt, tilt)
        hist.append(
            {
                "f": f.copy(),
                "f_filt": np.asarray(f_filt).copy(),
                "x_ref": tip_ref.copy(),
                "x_virt": np.asarray(tip_cmd).copy(),
                "x_tip": backend.tip_pos().copy(),
                "tilt_deg": math.degrees(tilt),
                "k_low": k_low,
                "soft_axis": np.asarray(soft).copy(),
            }
        )
        if i % 400 == 0:
            print(
                f"  i={i} tip={backend.tip_pos().round(3)} "
                f"cube={backend.cube_pos().round(3)} "
                f"tilt={math.degrees(tilt):.1f}deg |f|={np.linalg.norm(f):.1f}"
            )
        if tilt > math.radians(70.0) and a > 0.3:
            print(f"  early stop at i={i}, tilt={math.degrees(tilt):.1f}deg")
            break

    for _ in range(300):
        backend.inject_force_W(np.zeros(3))
        backend.step(realtime=False)
        max_tilt = max(max_tilt, backend.cube_tilt_rad())

    final_tilt = backend.cube_tilt_rad()
    print("-" * 50)
    print("翻方块验收（脚本 + tip）")
    print(f"  max tilt = {math.degrees(max_tilt):.1f} deg")
    print(f"  final tilt = {math.degrees(final_tilt):.1f} deg")
    print(f"  cube pos = {backend.cube_pos().round(3)}")
    ok = max_tilt > math.radians(55.0)
    print("  PASS" if ok else "  FAIL")
    if not ok:
        print("  reason: cube did not tip enough")

    if args.plot and hist:
        from sim_acp.bridge.plot_compliance import save_compliance_plots

        out = Path(args.plot_out)
        p = save_compliance_plots(
            hist,
            out,
            k_low=float(args.k_min),
            k_high=float(args.k_high),
            title="scripted flip (fixture+tip, stiff expert)",
            delta_clip_m=0.0,
        )
        print(f"  stiffness plot: {p}")
        print(f"  traj plot:      {p.with_name(p.stem + '_traj' + p.suffix)}")
    return 0 if ok else 2


def run_policy_flip(backend, args, runner=None) -> int:
    """v2 / v2-ft：RGB + wrench + pose → Flip Spec → 开环执行 action horizon。

    与真机 runner 一致：每次推理执行 sparse_execution_horizon 个路点
    （默认 12），路点间隔 down_sample_steps（默认 50）仿真步。
    runner: 可选预加载（--loop 时复用，避免每轮重载 ckpt）。
    """
    import numpy as np

    from sim_acp.bridge.policy_runner import FlipSpecPolicyRunner
    from sim_acp.bridge.pose_buffer import PoseRingBuffer
    from sim_acp.bridge.rgb_buffer import RgbRingBuffer
    from sim_acp.bridge.virtual_target import soft_axis_from_policy
    from sim_acp.bridge.wrench_buffer import WrenchRingBuffer

    backend.set_tip_clamp(False)
    if runner is None:
        runner = FlipSpecPolicyRunner(ckpt_path=args.ckpt)
    rh, rw = runner.rgb_hw
    buf = WrenchRingBuffer(capacity=max(8000, runner.wrench_h + 10))
    pose_buf = PoseRingBuffer(capacity=max(16, runner.pose_h + 4))
    rgb_buf = RgbRingBuffer(capacity=max(8, runner.rgb_h + 2), h=rh, w=rw)

    use_cam = not bool(args.fake_rgb)
    rgb_out = Path(args.rgb_dump_dir)
    if use_cam:
        rgb_out.mkdir(parents=True, exist_ok=True)

    exec_h = int(getattr(args, "exec_horizon", 12))
    ds_steps = int(getattr(args, "action_ds", 50))
    require_flip = bool(getattr(args, "require_flip", False))
    sync_every = int(getattr(args, "viewer_sync_every", 5))
    if hasattr(backend, "set_viewer_sync_every"):
        backend.set_viewer_sync_every(sync_every)

    # reset 后 settle：真实 wrench/pose/RGB 预填（不用零 wrench）
    for _ in range(80):
        backend.inject_force_W(np.zeros(3))
        if not backend.step(realtime=False):
            print("  abort: viewer closed during settle")
            return 3

    st0 = backend.read_state()
    tip0 = backend.tip_pos()
    pose0 = st0.pose_xyzquat.copy()
    pose0[:3] = tip0
    w0 = st0.wrench_W.copy()
    for _ in range(runner.wrench_h):
        buf.push(w0)
    for _ in range(runner.pose_h):
        pose_buf.push(pose0)
    rgb0 = None
    if use_cam:
        rgb0 = backend.render_rgb()
        for _ in range(max(2, runner.rgb_h)):
            rgb_buf.push(rgb0)
    for _ in range(max(2, runner.rgb_h)):
        if not backend.step(realtime=False):
            print("  abort: viewer closed during obs warm-up")
            return 3
        st0 = backend.read_state()
        p = st0.pose_xyzquat.copy()
        p[:3] = backend.tip_pos()
        pose_buf.push(p)
        buf.push(st0.wrench_W)
        if use_cam:
            rgb_buf.push(backend.render_rgb())

    p0 = backend.tip_pos().copy()
    clip_lo = np.array([0.55, -0.70, 0.75])
    clip_hi = np.array([0.90, -0.40, 1.00])
    max_tilt = backend.cube_tilt_rad()
    max_disp = 0.0
    max_cmd_disp = 0.0
    hist: list[dict] = []
    n_infer = 0
    pose_delta_sum = 0.0
    i_step = 0
    total_steps = int(args.steps)
    aborted_viewer = False

    print(
        f"[flip-policy {'v2-ft' if require_flip else 'v2'}] "
        f"RGB={'cam' if use_cam else 'fake'} "
        f"exec_h={exec_h} ds={ds_steps} steps={total_steps}"
    )
    if require_flip:
        print("  验收：真 RGB + max tilt ≥ 55°")
    else:
        print("  验收：三模态推理通 + 可测位移；不宣称翻成功率")

    while i_step < total_steps:
        st = backend.read_state()
        if use_cam:
            rgb_buf.push(backend.render_rgb())
        pose_hist = pose_buf.stack_last(runner.pose_h)
        if pose_hist.shape[0] >= 2:
            pose_delta_sum = float(
                np.linalg.norm(pose_hist[-1, :3] - pose_hist[0, :3])
            )
        rgb = None if args.fake_rgb else rgb_buf.stack_last(runner.rgb_h)
        act = runner.predict(
            pose_hist,
            buf.window(runner.wrench_h),
            rgb_uint8=rgb,
            fake_rgb=bool(args.fake_rgb),
            force_xyz=st.wrench_W[:3],
        )
        n_infer += 1
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
        k_traj = (
            act.k_traj if act.k_traj is not None else np.array([act.k_low])
        )
        H = min(exec_h, len(virt_traj))
        clipped = np.clip(virt_traj[:H], clip_lo, clip_hi)
        max_cmd_disp = max(
            max_cmd_disp, float(np.linalg.norm(clipped - p0, axis=1).max())
        )
        print(
            f"  [policy i={i_step}] k0={act.k_low:.0f} "
            f"virt0={virt_traj[0].round(3)} virtH={virt_traj[H-1].round(3)} "
            f"|Δpose_hist|={pose_delta_sum*1000:.1f}mm "
            f"infer={act.inference_s:.2f}s"
        )
        if use_cam and n_infer % max(1, int(args.rgb_save_every)) == 0:
            p = rgb_buf.save_latest(rgb_out / f"rgb_{i_step:05d}.png")
            if p is not None:
                print(f"    saved RGB → {p}")

        tip_cur = backend.tip_pos()
        for h in range(H):
            wp = np.clip(virt_traj[h], clip_lo, clip_hi)
            x_ref = ref_traj[min(h, len(ref_traj) - 1)]
            k_low = float(k_traj[min(h, len(k_traj) - 1)])
            soft_axis = soft_axis_from_policy(
                x_ref, wp, force_xyz=backend.read_state().wrench_W[:3]
            )
            for s in range(ds_steps):
                if i_step >= total_steps:
                    break
                a = (s + 1) / max(1, ds_steps)
                tip_ref = (1.0 - a) * tip_cur + a * wp
                backend.inject_force_W(np.zeros(3))
                st = backend.read_state()
                pose_tcp = st.pose_xyzquat.copy()
                pose_tcp[:3] = backend.tip_pos()
                buf.push(st.wrench_W)
                pose_buf.push(pose_tcp)
                if use_cam and (i_step % max(1, int(args.rgb_every)) == 0):
                    rgb_buf.push(backend.render_rgb())
                backend.write_tip_pos(tip_ref, st.timestamp_ns)
                if not backend.step(realtime=False):
                    aborted_viewer = True
                    i_step = total_steps
                    break
                tip_now = backend.tip_pos()
                tilt = backend.cube_tilt_rad()
                max_disp = max(max_disp, float(np.linalg.norm(tip_now - p0)))
                max_tilt = max(max_tilt, tilt)
                hist.append(
                    {
                        "f": st.wrench_W[:3].copy(),
                        "x_ref": np.asarray(x_ref).copy(),
                        "x_virt": tip_ref.copy(),
                        "x_tip": tip_now.copy(),
                        "tilt_deg": math.degrees(tilt),
                        "k_low": k_low,
                        "soft_axis": soft_axis.copy(),
                    }
                )
                i_step += 1
                if require_flip and tilt >= math.radians(70.0):
                    print(
                        f"  early stop tilt={math.degrees(tilt):.1f}deg at i={i_step}"
                    )
                    i_step = total_steps
                    break
            tip_cur = wp.copy()
            if i_step >= total_steps:
                break

    if aborted_viewer:
        print("  abort: viewer closed")
        return 3

    rgb_n = 0
    if use_cam and rgb_out.is_dir():
        rgb_n = len(list(rgb_out.glob("rgb_*.png"))) + len(
            list(rgb_out.glob("rgb_*.ppm"))
        )

    print("-" * 50)
    print(
        "翻方块联调（Flip Spec 三模态 "
        + ("v2-ft" if require_flip else "v2")
        + "）"
    )
    print(f"  RGB = {'cam' if use_cam else 'fake'}")
    print(f"  inferences = {n_infer}")
    print(f"  last |Δpose_hist| = {pose_delta_sum*1000:.1f} mm（>0 表示真时序）")
    print(f"  max |x_virt_cmd - start| = {max_cmd_disp*1000:.1f} mm")
    print(f"  max |ee - start| = {max_disp*1000:.1f} mm")
    print(f"  max cube tilt = {math.degrees(max_tilt):.1f} deg")
    if use_cam:
        print(f"  RGB dumps = {rgb_n} files under {rgb_out}")
    ok_infer = n_infer >= 1
    ok_motion = max_disp > 0.005 or max_cmd_disp > 0.01
    ok_rgb = (not use_cam) or rgb_n > 0
    ok_flip = max_tilt >= math.radians(55.0)
    if require_flip:
        ok = ok_infer and ok_rgb and ok_flip and use_cam
        print("  PASS (tilt≥55°, true RGB)" if ok else "  FAIL (v2-ft flip)")
        if not ok_flip:
            print("  reason: cube tilt < 55°")
        if not use_cam:
            print("  reason: v2-ft requires true RGB (not --fake-rgb)")
    else:
        ok = ok_infer and ok_motion and ok_rgb
        print(
            "  PASS (trimodal closed-loop)"
            if ok
            else "  FAIL (trimodal link incomplete)"
        )
        print("  note: 不宣称翻成功率（域差大）；微调后用 --require-flip")
        if max_tilt > math.radians(40.0):
            print("  note: cube tipped — unexpected bonus")
    if not ok_infer:
        print("  reason: no policy inference")
    if not require_flip and not ok_motion:
        print("  reason: no measurable motion / command")
    if not ok_rgb:
        print("  reason: expected RGB dumps missing")

    if args.plot and hist:
        from sim_acp.bridge.plot_compliance import save_compliance_plots

        k_series = np.array([h.get("k_low", args.k_min) for h in hist], dtype=float)
        out = Path(args.plot_out)
        if out.name == "flip_stiffness.png":
            out = out.with_name("flip_policy_stiffness.png")
        p = save_compliance_plots(
            hist,
            out,
            k_low=float(np.median(k_series)),
            k_high=float(args.k_high),
            title="Flip Spec trimodal (RGB+wrench+pose → exec horizon)",
            delta_clip_m=0.0,
        )
        print(f"  stiffness plot: {p}")
        print(f"  traj plot:      {p.with_name(p.stem + '_traj' + p.suffix)}")
        print(f"  k_low range: [{k_series.min():.0f}, {k_series.max():.0f}] N/m")
        if use_cam:
            print(f"  RGB dumps: {rgb_out}")
    return 0 if ok else 2



def main() -> int:
    parser = argparse.ArgumentParser(description="ACP single-arm flip cube")
    parser.add_argument("--render", action="store_true")
    # 方案 A：力调度（论文 Eq.7）；默认按仿真接触力量级调过
    parser.add_argument("--k-max", type=float, default=2800.0, help="无接触/小力时刚度")
    parser.add_argument("--k-min", type=float, default=400.0, help="大力时软轴刚度")
    parser.add_argument("--k-high", type=float, default=2800.0, help="正交方向刚度")
    parser.add_argument("--f-low", type=float, default=1.5, help="|f|<f_low → k_max")
    parser.add_argument("--f-high", type=float, default=15.0, help="|f|>f_high → k_min")
    parser.add_argument(
        "--force-ema",
        type=float,
        default=0.02,
        help="力 EMA 系数（越小越稳，默认 0.02）",
    )
    parser.add_argument(
        "--offset-ema",
        type=float,
        default=0.05,
        help="柔顺偏移 EMA（越小 tip 越不抖）",
    )
    parser.add_argument(
        "--k-ema",
        type=float,
        default=0.04,
        help="k_low 调度 EMA",
    )
    parser.add_argument(
        "--max-offset-step",
        type=float,
        default=0.00035,
        help="每仿真步最大柔顺位移 (m)，抑抖",
    )
    parser.add_argument(
        "--f-eps",
        type=float,
        default=1.5,
        help="力死区 (N)，小于此不开启柔顺偏移",
    )
    parser.add_argument(
        "--fixed-k",
        action="store_true",
        help="关闭力调度，k_low 固定为 k_min",
    )
    # 兼容旧参数名
    parser.add_argument("--k-low", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sweep-steps", type=int, default=2200)
    parser.add_argument(
        "--policy",
        action="store_true",
        help="v2 三模态：RGB+wrench+pose → Flip Spec",
    )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument(
        "--require-flip",
        action="store_true",
        help="v2-ft：验收 max tilt≥55°（真 RGB）；零样本 v2 不要开",
    )
    parser.add_argument(
        "--exec-horizon",
        type=int,
        default=12,
        help="每次推理执行的 action 路点数（真机 flip=12）",
    )
    parser.add_argument(
        "--action-ds",
        type=int,
        default=50,
        help="action 路点间隔仿真步（与 sparse_action_down_sample_steps 对齐）",
    )
    parser.add_argument("--policy-every", type=int, default=150)
    parser.add_argument("--steps", type=int, default=2500, help="policy 模式步数")
    parser.add_argument(
        "--fake-rgb",
        action="store_true",
        help="假灰图（调试）；默认用 acp_wrist_cam 腕部真 RGB",
    )
    parser.add_argument(
        "--rgb-every",
        type=int,
        default=5,
        help="每隔多少仿真步采一帧 RGB",
    )
    parser.add_argument(
        "--rgb-save-every",
        type=int,
        default=3,
        help="每 N 次推理存一张 RGB 预览",
    )
    parser.add_argument(
        "--rgb-dump-dir",
        type=str,
        default=str(_REPO / "sim_acp" / "outputs" / "policy_rgb"),
    )
    parser.add_argument(
        "--virt-ema",
        type=float,
        default=0.15,
        help="策略 x_virt 跟踪 EMA",
    )
    parser.add_argument(
        "--virt-max-step",
        type=float,
        default=0.004,
        help="策略 x_virt 每步最大位移 (m)",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="结束后画变刚度曲线（默认开）",
    )
    parser.add_argument(
        "--plot-out",
        type=str,
        default=str(_REPO / "sim_acp" / "outputs" / "flip_stiffness.png"),
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="循环播放（重置场景后重跑）；Ctrl+C / 关窗退出",
    )
    parser.add_argument(
        "--viewer-sync-every",
        type=int,
        default=5,
        help="MuJoCo viewer 每 N 仿真步 sync 一次（越大越快）",
    )
    args = parser.parse_args()
    if args.k_low is not None:
        # 旧 CLI：--k-low 当作固定柔顺值
        args.k_min = float(args.k_low)
        args.fixed_k = True

    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("请先: pip install mujoco")
        return 1

    from sim_acp.bridge.i7_mujoco_backend import I7MujocoBackend

    print("=" * 60)
    print("ACP 单臂翻方块")
    print(
        "  mode:",
        "Flip Spec trimodal v2" if args.policy else "scripted + Scheme A (baseline v1)",
    )
    if args.loop:
        print("  loop: on（Ctrl+C 或关闭 viewer 退出；FAIL 不中断）")
    print("=" * 60)

    if args.loop and args.plot:
        args.plot = False

    backend = I7MujocoBackend(render=args.render, tip_clamp=False)
    if hasattr(backend, "set_viewer_sync_every"):
        backend.set_viewer_sync_every(int(args.viewer_sync_every))
    policy_runner = None
    if args.policy:
        from sim_acp.bridge.policy_runner import FlipSpecPolicyRunner

        policy_runner = FlipSpecPolicyRunner(ckpt_path=args.ckpt)

    rc = 2
    ep = 0
    n_pass = 0
    n_fail = 0
    try:
        while True:
            if ep > 0:
                if args.render and not backend.viewer_running():
                    print("[loop] viewer 已关闭，退出")
                    rc = 0
                    break
                backend.reset_episode()
            print(f"\n{'=' * 40}\n[loop] episode {ep}\n{'=' * 40}")
            if args.policy:
                rc_ep = run_policy_flip(backend, args, runner=policy_runner)
            else:
                rc_ep = run_scripted_flip(backend, args)
            if rc_ep == 3:
                # viewer closed mid-episode
                print("[loop] viewer 关闭，退出")
                rc = 0
                break
            if rc_ep == 0:
                n_pass += 1
            else:
                n_fail += 1
                if args.loop:
                    print(
                        f"[loop] episode {ep} FAIL — 继续下一轮 "
                        f"(pass={n_pass} fail={n_fail})"
                    )
            rc = rc_ep
            ep += 1
            if not args.loop:
                break
        if args.loop and ep > 0:
            print(
                f"[loop] 结束：episodes={ep} PASS={n_pass} FAIL={n_fail}"
            )
            rc = 0
    except KeyboardInterrupt:
        print(
            f"\n[loop] Ctrl+C，停止 "
            f"(episodes={ep} PASS={n_pass} FAIL={n_fail})"
        )
        rc = 0
    finally:
        print("done")
        try:
            backend.close()
        except Exception as exc:
            print(f"[warn] close: {exc}")
        # --render 时 MuJoCo/GLFW 析构偶发 SIGSEGV；任务已结束则直接退出，
        # 避免解释器 atexit 再次踩坏的 GL 上下文。
        if args.render:
            os._exit(int(rc) if isinstance(rc, int) else 0)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
