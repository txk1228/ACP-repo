"""M3：加载 Flip Spec，假 RGB + 常数 pose/wrench，跑通一次前向。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description="M3 Flip Spec forward smoke")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    import numpy as np

    from sim_acp.bridge.policy_runner import FlipSpecPolicyRunner

    runner = FlipSpecPolicyRunner(ckpt_path=args.ckpt, device=args.device)
    pose = np.array([0.4, 0.0, 0.35, 0.0, 0.0, 0.0, 1.0], dtype=float)
    wrench = np.zeros((runner.wrench_h, 6), dtype=np.float32)
    # 末段给一点力，看输出是否有限
    wrench[-100:, 0] = 5.0

    act = runner.predict(pose, wrench, fake_rgb=True)
    print("M3 OK")
    print(f"  inference_s = {act.inference_s:.3f}")
    print(f"  x_ref_pos   = {act.x_ref_pos.round(4)}")
    print(f"  x_virt_pos  = {act.x_virt_pos.round(4)}")
    print(f"  k_low       = {act.k_low:.2f}")
    print(f"  soft_axis   = {act.soft_axis.round(3)}")
    print(f"  raw_t0[:6]  = {act.raw_action_t0[:6].round(4)}")
    assert np.isfinite(act.x_ref_pos).all()
    assert np.isfinite(act.x_virt_pos).all()
    assert np.isfinite(act.k_low)
    print("  PASS (finite outputs; fake RGB → 动作不必合理)")

    # 若 MuJoCo 可用，再测一帧相机 RGB 前向
    try:
        import mujoco  # noqa: F401

        from sim_acp.bridge.mujoco_backend import MujocoSimBackend

        backend = MujocoSimBackend(render=False)
        for _ in range(30):
            backend.step(realtime=False)
        rgb = backend.render_rgb()
        rgb_stack = np.stack([rgb, rgb], axis=0)
        act2 = runner.predict(pose, wrench, rgb_uint8=rgb_stack, fake_rgb=False)
        print("M5 cam→policy OK")
        print(f"  rgb shape={rgb.shape} mean={rgb.mean():.1f}")
        print(f"  infer={act2.inference_s:.3f}s k_low={act2.k_low:.2f}")
        backend.close()
    except Exception as exc:
        print(f"  (skip cam forward: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
