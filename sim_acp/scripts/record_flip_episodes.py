"""录制仿真翻方块专家演示 → Flip Spec 兼容 zarr（成功 episode 才入库）。

用法：
  conda activate pyrite && source scripts/setup_env.sh
  python -m sim_acp.scripts.record_flip_episodes --n 50 --out $PYRITE_DATASET_FOLDERS/flip_up_sim_v1
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _lerp(a, b, t: float):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    t = float(np.clip(t, 0.0, 1.0))
    return a * (1.0 - t) + b * t


def _pose7_sim_to_acp(pose7_sim: np.ndarray) -> np.ndarray:
    p = np.asarray(pose7_sim, dtype=np.float64).reshape(7)
    return np.array(
        [p[0], p[1], p[2], p[6], p[3], p[4], p[5]], dtype=np.float64
    )


def _wrench_world_to_tool(wrench_W: np.ndarray, pose7_acp: np.ndarray) -> np.ndarray:
    from spatialmath.base import q2r

    R = q2r(pose7_acp[3:7])
    w = np.asarray(wrench_W, dtype=np.float64).reshape(6)
    out = np.zeros(6, dtype=np.float64)
    out[:3] = R.T @ w[:3]
    out[3:] = R.T @ w[3:]
    return out


def _upsample_wrench(
    wrench: np.ndarray, t_ms: np.ndarray, factor: int
) -> tuple[np.ndarray, np.ndarray]:
    """将仿真步 wrench 上采样到接近 ATI 7kHz（默认 factor=7 @ 1kHz）。"""
    if factor <= 1:
        return wrench, t_ms
    n = len(wrench)
    out_w = np.repeat(wrench, factor, axis=0)
    # 在相邻样本间线性插时间
    out_t = np.zeros(n * factor, dtype=np.float64)
    for i in range(n):
        t0 = float(t_ms[i])
        if i + 1 < n:
            t1 = float(t_ms[i + 1])
        else:
            t1 = t0 + (float(t_ms[i]) - float(t_ms[i - 1]) if i > 0 else 1.0 / factor)
        for k in range(factor):
            out_t[i * factor + k] = t0 + (t1 - t0) * (k / factor)
    return out_w, out_t


def run_one_expert_episode(
    backend,
    *,
    sweep_steps: int = 2200,
    rgb_every: int = 16,
    collect_rgb: bool = True,
) -> dict | None:
    """跑一条刚性专家翻物；成功(tilt≥55°)返回轨迹 dict，否则 None。"""
    from sim_acp.bridge.i7_scene import CUBE_HALF, TIP_RADIUS

    backend.set_tip_clamp(False)
    for _ in range(120):
        backend.inject_force_W(np.zeros(3))
        backend.step(realtime=False)

    cube0 = backend.cube_pos()
    table_z = backend.table_top_z()
    tip_clear = float(TIP_RADIUS) + 0.004
    dt_ms = float(backend.model.opt.timestep) * 1000.0

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

    rgb_list: list[np.ndarray] = []
    rgb_t: list[float] = []
    pose_fb: list[np.ndarray] = []
    pose_cmd: list[np.ndarray] = []
    robot_t: list[float] = []
    wrench_list: list[np.ndarray] = []
    wrench_t: list[float] = []

    t_ms = 0.0
    step_i = 0
    max_tilt = 0.0

    def _record(tip_ref: np.ndarray) -> None:
        nonlocal t_ms, step_i, max_tilt
        st = backend.read_state()
        tip = backend.tip_pos()
        # TCP = tip 位置 + 法兰姿态（接触点与专家控制量一致）
        pose_sim = st.pose_xyzquat.copy()
        pose_sim[:3] = tip
        pose_acp = _pose7_sim_to_acp(pose_sim)
        cmd_sim = pose_sim.copy()
        cmd_sim[:3] = tip_ref
        cmd_acp = _pose7_sim_to_acp(cmd_sim)
        w_tool = _wrench_world_to_tool(st.wrench_W, pose_acp)

        pose_fb.append(pose_acp)
        pose_cmd.append(cmd_acp)
        robot_t.append(t_ms)
        wrench_list.append(w_tool)
        wrench_t.append(t_ms)

        if collect_rgb and (step_i % max(1, rgb_every) == 0):
            rgb_list.append(backend.render_rgb().copy())
            rgb_t.append(t_ms)

        max_tilt = max(max_tilt, backend.cube_tilt_rad())
        t_ms += dt_ms
        step_i += 1

    def _goto(tip_xyz: np.ndarray, steps: int) -> bool:
        tip0 = backend.tip_pos()
        for i in range(steps):
            a = (i + 1) / max(1, steps)
            tip_ref = _lerp(tip0, tip_xyz, a)
            backend.inject_force_W(np.zeros(3))
            backend.write_tip_pos(tip_ref, 0)
            if not backend.step(realtime=False):
                return False
            _record(tip_ref)
        return True

    if not _goto(approach, 700):
        return None
    if not _goto(contact, 600):
        return None

    for _ in range(100):
        backend.inject_force_W(np.zeros(3))
        backend.write_tip_pos(contact, 0)
        if not backend.step(realtime=False):
            return None
        _record(contact)

    tip_start = backend.tip_pos()
    for i in range(int(sweep_steps)):
        a = (i + 1) / max(1, sweep_steps)
        if a < 0.65:
            tip_ref = _lerp(tip_start, sweep, a / 0.65)
        else:
            tip_ref = _lerp(sweep, finish, (a - 0.65) / 0.35)
        backend.inject_force_W(np.zeros(3))
        backend.write_tip_pos(tip_ref, 0)
        if not backend.step(realtime=False):
            return None
        _record(tip_ref)
        if backend.cube_tilt_rad() > math.radians(70.0) and a > 0.3:
            break

    for _ in range(200):
        backend.inject_force_W(np.zeros(3))
        tip_hold = backend.tip_pos()
        backend.write_tip_pos(tip_hold, 0)
        if not backend.step(realtime=False):
            break
        _record(tip_hold)
        max_tilt = max(max_tilt, backend.cube_tilt_rad())

    if max_tilt < math.radians(55.0):
        return None
    if len(pose_fb) < 100 or len(rgb_list) < 8:
        return None

    return {
        "rgb": np.stack(rgb_list, axis=0).astype(np.uint8),
        "rgb_time_stamps": np.asarray(rgb_t, dtype=np.float64),
        "ts_pose_fb": np.stack(pose_fb, axis=0),
        "ts_pose_command": np.stack(pose_cmd, axis=0),
        "robot_time_stamps": np.asarray(robot_t, dtype=np.float64),
        "wrench": np.stack(wrench_list, axis=0),
        "wrench_time_stamps": np.asarray(wrench_t, dtype=np.float64),
        "max_tilt_deg": math.degrees(max_tilt),
    }


def save_episode_zarr(
    root,
    episode_id: int,
    ep: dict,
    *,
    wrench_upsample: int = 7,
) -> None:
    import zarr

    data = root.require_group("data")
    # 用 100000+id，保证字典序与数值序一致，且 sampler 的 f"episode_{id}" 可还原
    name = f"episode_{episode_id}"
    if name in data:
        del data[name]
    g = data.create_group(name)

    rgb = ep["rgb"]
    n, h, w, c = rgb.shape
    g.create_dataset(
        "rgb_0",
        data=rgb,
        chunks=(1, h, w, c),
        dtype=np.uint8,
    )
    g["rgb_time_stamps_0"] = zarr.array(ep["rgb_time_stamps"])
    g["ts_pose_fb_0"] = zarr.array(ep["ts_pose_fb"])
    g["ts_pose_command_0"] = zarr.array(ep["ts_pose_command"])
    g["robot_time_stamps_0"] = zarr.array(ep["robot_time_stamps"])

    wh, wt = _upsample_wrench(ep["wrench"], ep["wrench_time_stamps"], wrench_upsample)
    g["wrench_0"] = zarr.array(wh)
    g["wrench_filtered_0"] = zarr.array(wh.copy())
    g["wrench_time_stamps_0"] = zarr.array(wt)


def write_meta(root, n_eps: int, lengths: dict) -> None:
    import zarr

    if "meta" in root:
        del root["meta"]
    meta = root.create_group("meta")
    meta["episode_rgb0_len"] = zarr.array(np.asarray(lengths["rgb"], dtype=np.int64))
    meta["episode_robot0_len"] = zarr.array(
        np.asarray(lengths["robot"], dtype=np.int64)
    )
    meta["episode_wrench0_len"] = zarr.array(
        np.asarray(lengths["wrench"], dtype=np.int64)
    )
    assert len(lengths["rgb"]) == n_eps


def main() -> int:
    parser = argparse.ArgumentParser(description="Record sim flip expert demos")
    parser.add_argument("--n", type=int, default=50, help="成功 episode 目标数")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出 zarr 目录（默认 $PYRITE_DATASET_FOLDERS/flip_up_sim_v1）",
    )
    parser.add_argument("--sweep-steps", type=int, default=2200)
    parser.add_argument("--rgb-every", type=int, default=16)
    parser.add_argument("--xy-noise", type=float, default=0.012, help="方块 xy 扰动 (m)")
    parser.add_argument("--wrench-upsample", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=0, help="0=自动 3*n")
    parser.add_argument("--no-rgb", action="store_true")
    args = parser.parse_args()

    out = Path(
        args.out
        or os.path.join(
            os.environ.get("PYRITE_DATASET_FOLDERS", str(Path.home() / "data")),
            "flip_up_sim_v1",
        )
    )
    out.mkdir(parents=True, exist_ok=True)

    import zarr

    from sim_acp.bridge.i7_mujoco_backend import I7MujocoBackend
    from sim_acp.bridge.i7_scene import CUBE_XY

    store = zarr.DirectoryStore(str(out))
    root = zarr.open(store=store, mode="a")
    rng = np.random.default_rng(int(args.seed))
    max_attempts = int(args.max_attempts) or int(args.n) * 3

    backend = I7MujocoBackend(render=False, tip_clamp=False)
    saved = 0
    lengths = {"rgb": [], "robot": [], "wrench": []}
    # 若目录已有 episode，从 max id+1 继续
    existing: list[int] = []
    if "data" in root:
        existing = [
            int(k.split("_")[-1])
            for k in root["data"].keys()
            if k.startswith("episode_")
        ]
        if existing:
            # meta 顺序必须与 data.keys() 迭代顺序一致（通常为字典序）
            for key in root["data"].keys():
                if not key.startswith("episode_"):
                    continue
                ep = root["data"][key]
                lengths["rgb"].append(int(ep["rgb_0"].shape[0]))
                lengths["robot"].append(int(ep["ts_pose_fb_0"].shape[0]))
                lengths["wrench"].append(int(ep["wrench_0"].shape[0]))
            saved = len(existing)
            print(f"[record] resume: already have {saved} episodes in {out}")

    # 100000+ 保证 episode_100000.. 字典序正确，且 sampler 可还原
    EP_BASE = 100000
    next_id = (max(existing) + 1) if existing else EP_BASE
    if next_id < EP_BASE:
        next_id = EP_BASE + saved
    attempts = 0
    t0 = time.time()
    try:
        while saved < int(args.n) and attempts < max_attempts:
            attempts += 1
            dx = float(rng.uniform(-args.xy_noise, args.xy_noise))
            dy = float(rng.uniform(-args.xy_noise, args.xy_noise))
            cube_xy = (CUBE_XY[0] + dx, CUBE_XY[1] + dy)
            backend.reset_episode(cube_xy=cube_xy)
            print(
                f"[record] attempt {attempts} saved={saved}/{args.n} "
                f"cube_xy=({cube_xy[0]:.3f},{cube_xy[1]:.3f})"
            )
            ep = run_one_expert_episode(
                backend,
                sweep_steps=int(args.sweep_steps),
                rgb_every=int(args.rgb_every),
                collect_rgb=not bool(args.no_rgb),
            )
            if ep is None:
                print("  → FAIL (tilt or short)")
                continue
            wh, _ = _upsample_wrench(
                ep["wrench"], ep["wrench_time_stamps"], int(args.wrench_upsample)
            )
            save_episode_zarr(
                root,
                next_id,
                ep,
                wrench_upsample=int(args.wrench_upsample),
            )
            lengths["rgb"].append(int(ep["rgb"].shape[0]))
            lengths["robot"].append(int(ep["ts_pose_fb"].shape[0]))
            lengths["wrench"].append(int(wh.shape[0]))
            print(
                f"  → PASS episode_{next_id} tilt={ep['max_tilt_deg']:.1f}deg "
                f"T_robot={ep['ts_pose_fb'].shape[0]} "
                f"T_rgb={ep['rgb'].shape[0]} T_wrench={wh.shape[0]}"
            )
            next_id += 1
            saved += 1
            write_meta(root, saved, lengths)
    finally:
        backend.close()

    write_meta(root, saved, lengths)
    elapsed = time.time() - t0
    print("=" * 50)
    print(f"saved {saved}/{args.n} episodes → {out}")
    print(f"attempts={attempts} elapsed={elapsed:.1f}s")
    return 0 if saved >= int(args.n) else 2


if __name__ == "__main__":
    raise SystemExit(main())
