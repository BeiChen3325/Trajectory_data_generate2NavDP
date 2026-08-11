"""Convenience orchestrator for the three independent file-based stages."""

from __future__ import annotations

import argparse
import subprocess
import sys

from robotnav.config import load_render_config
from robotnav.gpu_environment import require_cuda_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all target dataset build stages in order.")
    parser.add_argument("--config", default="dataset_build.toml")
    parser.add_argument("--render-config", default="render.toml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_config = load_render_config(args.render_config)
    if not render_config.render.require_cuda:
        raise ValueError("Formal dataset builds require [render].require_cuda=true")
    require_cuda_environment()
    commands = [
        [
            sys.executable,
            "-m",
            "robotnav.dataset.trajectory_to_camera",
            "--config",
            args.config,
        ],
        [
            sys.executable,
            "-m",
            "robotnav.dataset.render_trajectory",
            "--config",
            args.config,
            "--render-config",
            args.render_config,
        ],
        [
            sys.executable,
            "-m",
            "robotnav.dataset.package_dataset",
            "--config",
            args.config,
        ],
    ]
    for command in commands:
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
