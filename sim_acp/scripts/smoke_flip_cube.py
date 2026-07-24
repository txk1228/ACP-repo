"""冒烟：脚本翻方块（无窗口）。"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    from sim_acp.run_flip_cube_demo import main as flip_main
    import sys as _sys

    _sys.argv = ["run_flip_cube_demo", "--sweep-steps", "2200"]
    return flip_main()


if __name__ == "__main__":
    raise SystemExit(main())
