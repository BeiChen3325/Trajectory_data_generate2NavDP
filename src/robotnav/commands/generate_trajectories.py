"""CLI for independent deterministic batch trajectory generation."""

from __future__ import annotations

import argparse
from dataclasses import replace

from robotnav.navigation.scene.artifact import load_scene_artifact
from robotnav.navigation.trajectory.batch import plan_trajectory_batch
from robotnav.navigation.trajectory.config import (
    TrajectoryRequest,
    load_trajectory_generation_config,
)


def _xz(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected X,Z, for example -2.0,3.5")
    return (float(parts[0]), float(parts[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one or more trajectories from a persisted navigation scene."
    )
    parser.add_argument("--config", default="trajectories.toml")
    parser.add_argument("--start-xz", type=_xz)
    parser.add_argument("--goal-xz", type=_xz)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_trajectory_generation_config(args.config)
    if args.start_xz is not None or args.goal_xz is not None:
        request = TrajectoryRequest("cli_route", args.start_xz, args.goal_xz)
        config = replace(
            config,
            batch=replace(config.batch, count=1, requests=(request,)),
        )
    scene = load_scene_artifact(config.paths.scene_dir)
    manifest = plan_trajectory_batch(config, scene)
    print(
        f"Generated {manifest['trajectory_count']} trajectories: "
        f"{config.paths.output_dir / config.batch.manifest_filename}"
    )


if __name__ == "__main__":
    main()
