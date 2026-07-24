"""冒烟：力对齐 K — 偏移只沿力方向；正交分量可忽略。"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    import numpy as np

    from sim_acp.bridge.virtual_target import (
        offset_along_force_ratio,
        virtual_target_pos,
    )

    x_ref = np.array([0.4, -0.5, 0.9])
    cases = [
        ("+Z", np.array([0.0, 0.0, 20.0])),
        ("+X", np.array([15.0, 0.0, 0.0])),
        ("diag", np.array([10.0, 5.0, 8.0])),
    ]
    k_low, k_high = 400.0, 2000.0
    ok = True
    for name, f in cases:
        xv = virtual_target_pos(x_ref, f, k_low=k_low, k_high=k_high)
        d = xv - x_ref
        align = offset_along_force_ratio(x_ref, xv, f)
        # 应约等于 f/k_low
        expected = f / k_low
        err = float(np.linalg.norm(d - expected))
        print(
            f"  {name}: Δ={d.round(4)} expect≈{expected.round(4)} "
            f"align={align:.3f} err={err:.2e}"
        )
        if align < 0.99 or err > 1e-6:
            ok = False
    print("force-aligned K", "PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
