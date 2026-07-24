"""接触任务冒烟：浅压桌面，检查 mesh tip 接触力与竖直方案 A。"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("请先: pip install mujoco")
        return 1

    import numpy as np

    from sim_acp.bridge.i7_mujoco_backend import I7MujocoBackend

    backend = I7MujocoBackend(render=False, tip_clamp=False)
    p0 = backend.read_state().pose_xyzquat[:3].copy()
    x_ref = backend.suggest_press_target()
    k_low = 800.0
    max_fz = 0.0
    max_pen = 0.0
    best_f = np.zeros(3)
    top = backend.table_top_z()
    r = backend.tip_radius()

    print(
        f"press x_ref={x_ref.round(3)} (from {p0.round(3)}) "
        f"table_top={top:.3f} tip_r~{r:.3f} (mesh, no tip ball)"
    )

    for i in range(900):
        a = min(1.0, i / 550.0)
        x_cmd = p0 * (1.0 - a) + x_ref * a
        pose = np.array([x_cmd[0], x_cmd[1], x_cmd[2], 0, 0, 0, 1], dtype=float)
        backend.write_ee_pose(pose, 0)
        backend.step(realtime=False)
        st = backend.read_state()
        f = st.wrench_W[:3]
        if float(f[2]) > max_fz:
            max_fz = float(f[2])
            best_f = f.copy()
        tip_z = float(backend.tip_pos()[2])
        # 粗估：工具点低于桌面即穿入
        pen = max(0.0, top - tip_z)
        max_pen = max(max_pen, pen)
        if i % 200 == 0:
            print(
                f"  ramp i={i} falan={st.pose_xyzquat[:3].round(3)} "
                f"tip_z={tip_z:.3f} pen={pen*1000:.1f}mm f={f.round(1)}"
            )

    for _ in range(200):
        backend.write_ee_pose(
            np.array([x_ref[0], x_ref[1], x_ref[2], 0, 0, 0, 1], dtype=float), 0
        )
        backend.step(realtime=False)
        st = backend.read_state()
        f = st.wrench_W[:3]
        if float(f[2]) > max_fz:
            max_fz = float(f[2])
            best_f = f.copy()
        tip_z = float(backend.tip_pos()[2])
        pen = max(0.0, top - tip_z)
        max_pen = max(max_pen, pen)

    x_virt0 = x_ref + np.array([0.0, 0.0, float(best_f[2]) / k_low])
    tip_z = float(backend.tip_pos()[2])
    print(f"hold: pos={st.pose_xyzquat[:3].round(3)} f={st.wrench_W[:3].round(1)}")
    print(f"best f={best_f.round(1)} max_fz={max_fz:.1f}")
    print(f"scheme A z-only: x_virt={x_virt0.round(3)} (x_ref={x_ref.round(3)})")
    print(f"max tip penetration = {max_pen*1000:.1f} mm  tip_z={tip_z:.3f}")

    backend.close()
    ok_force = max_fz > 5.0
    ok_comply = float(x_virt0[2]) > float(x_ref[2]) + 0.005
    ok_pen = max_pen < 0.025
    print("i7 contact press")
    print("  PASS" if (ok_force and ok_comply and ok_pen) else "  FAIL")
    if not ok_force:
        print("  reason: tip contact fz too small (need +z support)")
    if not ok_comply:
        print("  reason: scheme A did not lift x_virt above x_ref")
    if not ok_pen:
        print(f"  reason: tip penetrated too deep ({max_pen*1000:.1f} mm)")
    return 0 if (ok_force and ok_comply and ok_pen) else 2


if __name__ == "__main__":
    raise SystemExit(main())
