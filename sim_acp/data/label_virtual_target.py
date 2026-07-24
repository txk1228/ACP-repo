"""仿真 episode 的 virtual target / stiffness 标注（对齐 Flip Spec 后处理）。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_ACP = _REPO / "adaptive_compliance_policy"
if str(_ACP) not in sys.path:
    sys.path.insert(0, str(_ACP))

# 与 postprocess_add_virtual_target_label.py 对齐
DEFAULT_STIFFNESS_PARA = {
    "k_max": 5000,
    "k_min": 200,
    "f_low": 0.5,
    "f_high": 5,
    "dim": 3,
    "characteristic_length": 0.02,
    "vel_tol": 999.002,
}

WRENCH_MA_WINDOW = 7000


def _pose7_sim_to_acp(pose7_sim: np.ndarray) -> np.ndarray:
    """sim [x,y,z,qx,qy,qz,qw] → ACP [x,y,z,qw,qx,qy,qz]。"""
    p = np.asarray(pose7_sim, dtype=np.float64).reshape(-1, 7)
    out = np.zeros_like(p)
    out[:, :3] = p[:, :3]
    out[:, 3] = p[:, 6]
    out[:, 4:7] = p[:, 3:6]
    return out


def wrench_world_to_tool(wrench_W: np.ndarray, pose7_acp: np.ndarray) -> np.ndarray:
    """世界系 wrench(6) → 工具系（用姿态旋转力/力矩）。"""
    from spatialmath import SE3
    from spatialmath.base import q2r

    w = np.asarray(wrench_W, dtype=np.float64).reshape(-1, 6)
    p = np.asarray(pose7_acp, dtype=np.float64).reshape(-1, 7)
    out = np.zeros_like(w)
    for i in range(len(w)):
        R = q2r(p[i, 3:7])
        f_T = R.T @ w[i, :3]
        t_T = R.T @ w[i, 3:]
        out[i, :3] = f_T
        out[i, 3:] = t_T
    return out


def label_episode_arrays(
    ts_pose_fb_acp: np.ndarray,
    wrench_tool: np.ndarray,
    robot_time_stamps_ms: np.ndarray,
    wrench_time_stamps_ms: np.ndarray,
    *,
    stiffness_para: dict | None = None,
    wrench_ma_window: int = WRENCH_MA_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """对单条 episode 计算 ts_pose_virtual_target / stiffness。

    Args:
        ts_pose_fb_acp: (T,7) ACP 四元数顺序
        wrench_tool: (Tw,6) 工具系 wrench（与真机 ATI 一致）
        robot_time_stamps_ms / wrench_time_stamps_ms: 毫秒时间戳
    Returns:
        ts_pose_virtual_target (T,7), stiffness (T,)
    """
    from spatialmath import SE3
    from spatialmath.base import q2r, r2q

    from PyriteUtility.planning_control import compliance_helpers as ch
    from PyriteUtility.spatial_math import spatial_utilities as su

    para = dict(DEFAULT_STIFFNESS_PARA)
    if stiffness_para:
        para.update(stiffness_para)

    pose = np.asarray(ts_pose_fb_acp, dtype=np.float64)
    wrench = np.asarray(wrench_tool, dtype=np.float64).copy()
    rts = np.asarray(robot_time_stamps_ms, dtype=np.float64)
    wts = np.asarray(wrench_time_stamps_ms, dtype=np.float64)
    T = len(rts)
    assert pose.shape == (T, 7)

    n_off = min(200, max(1, len(wrench) // 10))
    wrench = wrench - np.mean(wrench[:n_off], axis=0)

    N = int(wrench_ma_window)
    wrench_ma = np.zeros_like(wrench)
    for d in range(6):
        wrench_ma[:, d] = np.convolve(wrench[:, d], np.ones(N) / N, mode="same")

    pe = ch.VirtualTargetEstimator(
        para["k_max"],
        para["k_min"],
        para["f_low"],
        para["f_high"],
        para["dim"],
        para["characteristic_length"],
        para["vel_tol"],
    )

    vt = np.zeros((T, 7), dtype=np.float64)
    stiff = np.zeros(T, dtype=np.float64)
    for t in range(T):
        pose7_WT = pose[t]
        SE3_WT = SE3.Rt(q2r(pose7_WT[3:7]), pose7_WT[0:3], check=False)
        t_wrench = int(np.argmin(np.abs(wts - rts[t])))
        wrench_T = wrench_ma[t_wrench]

        half = 10
        id_start = max(0, t - half)
        id_end = min(T - 1, t + half)
        SE3_start = su.pose7_to_SE3(pose[id_start])
        SE3_end = su.pose7_to_SE3(pose[id_end])
        twist_diff = su.SE3_to_spt(su.SE3_inv(SE3_start) @ SE3_end)

        if para["dim"] == 6:
            k, mat_TC, _ = pe.update(wrench_T, twist_diff)
            SE3_TC = SE3(mat_TC)
        else:
            k, pos_TC, _ = pe.update(wrench_T, twist_diff)
            SE3_TC = SE3.Rt(np.eye(3), pos_TC)
        SE3_WC = SE3_WT * SE3_TC
        vt[t] = np.concatenate([SE3_WC.t, r2q(SE3_WC.R)])
        stiff[t] = k
    return vt, stiff


def label_zarr_dataset(
    dataset_path: str | Path,
    *,
    stiffness_para: dict | None = None,
) -> int:
    """就地写入 ts_pose_virtual_target_0 / stiffness_0，返回处理的 episode 数。"""
    import zarr

    root = zarr.open(str(dataset_path), mode="r+")
    data = root["data"]
    n = 0
    for ep_name in sorted(data.keys(), key=lambda s: int(s.split("_")[-1])):
        ep = data[ep_name]
        if "ts_pose_virtual_target_0" in ep and "stiffness_0" in ep:
            print(f"[label] skip {ep_name} (already labeled)")
            n += 1
            continue
        pose = np.asarray(ep["ts_pose_fb_0"][:])
        wrench = np.asarray(ep["wrench_0"][:])
        rts = np.asarray(ep["robot_time_stamps_0"][:])
        wts = np.asarray(ep["wrench_time_stamps_0"][:])
        # 标注用下采样 wrench 对齐机器人时钟（与真机 argmin 时间对齐一致）
        vt, stiff = label_episode_arrays(
            pose, wrench, rts, wts, stiffness_para=stiffness_para
        )
        if "ts_pose_virtual_target_0" in ep:
            del ep["ts_pose_virtual_target_0"]
        if "stiffness_0" in ep:
            del ep["stiffness_0"]
        ep.create_dataset("ts_pose_virtual_target_0", data=vt, overwrite=True)
        ep.create_dataset("stiffness_0", data=stiff, overwrite=True)
        print(
            f"[label] {ep_name}: T={len(stiff)} "
            f"k∈[{stiff.min():.0f},{stiff.max():.0f}]"
        )
        n += 1
    return n


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Label sim flip episodes with x_virt/k")
    p.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(
            os.environ.get("PYRITE_DATASET_FOLDERS", str(Path.home() / "data")),
            "flip_up_sim_v1",
        ),
    )
    args = p.parse_args()
    path = Path(args.dataset)
    if not path.is_dir():
        print(f"dataset not found: {path}")
        return 1
    n = label_zarr_dataset(path)
    print(f"Done: labeled {n} episodes in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
