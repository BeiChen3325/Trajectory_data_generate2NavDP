"""Validate an already packaged scene using only the target directory contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robotnav.dataset.package_dataset import validate_target_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a packaged RobotNav target scene.")
    parser.add_argument("scene_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_target_scene(args.scene_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
