#!/usr/bin/env python3
"""Print the RobotNav GPU rendering preflight report."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robotnav.gpu_environment import (  # noqa: E402
    check_gpu_environment,
    format_gpu_environment_report,
)


def main() -> None:
    report = check_gpu_environment()
    print(format_gpu_environment_report(report))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
