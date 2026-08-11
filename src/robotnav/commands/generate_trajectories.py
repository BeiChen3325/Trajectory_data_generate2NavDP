"""CLI for independent deterministic batch trajectory generation."""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

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
    model = scene.model
    width = int(model.planning_blocked.shape[1])
    height = int(model.planning_blocked.shape[0])
    span_x, span_z = model.max_xz - model.origin_xz
    print(
        "Scene bounds: "
        f"x=[{model.origin_xz[0]:.3f}, {model.max_xz[0]:.3f}], "
        f"z=[{model.origin_xz[1]:.3f}, {model.max_xz[1]:.3f}]"
    )
    print(f"Map size: width={width}, height={height}, resolution={model.resolution_m:.3f} m")
    if float(np.hypot(span_x, span_z)) < config.trajectory_sampling.max_length_m:
        print(
            "WARNING: scene map diagonal is shorter than the configured maximum "
            f"trajectory length ({config.trajectory_sampling.max_length_m:.3f} m)."
        )
    manifest = plan_trajectory_batch(config, scene)
    print(
        f"Generated {manifest['trajectory_count']} trajectories: "
        f"{config.paths.output_dir / config.batch.manifest_filename}"
    )
    statistics = manifest["length_statistics"]
    print(
        "Trajectory length statistics: "
        f"min={statistics['min_m']:.3f} m, max={statistics['max_m']:.3f} m, "
        f"mean={statistics['mean_m']:.3f} m"
    )


if __name__ == "__main__":
    main()
