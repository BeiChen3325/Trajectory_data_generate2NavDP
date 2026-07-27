"""Convenience orchestrator that invokes the three public navigation stages."""

from __future__ import annotations

import argparse

from robotnav.navigation.scene.builder import build_scene
from robotnav.navigation.scene.config import load_scene_build_config
from robotnav.navigation.semantic_pointcloud.config import load_pointcloud_export_config
from robotnav.navigation.semantic_pointcloud.exporter import export_semantic_pointcloud
from robotnav.navigation.trajectory.batch import plan_trajectory_batch
from robotnav.navigation.trajectory.config import load_trajectory_generation_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build all navigation data products.")
    parser.add_argument("--scene-config", default="navigation_scene.toml")
    parser.add_argument("--trajectory-config", default="trajectories.toml")
    parser.add_argument("--pointcloud-config", default="pointcloud_export.toml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_config = load_scene_build_config(args.scene_config)
    trajectory_config = load_trajectory_generation_config(args.trajectory_config)
    pointcloud_config = load_pointcloud_export_config(args.pointcloud_config)
    scene_dir = scene_config.paths.output_dir.resolve()
    if trajectory_config.paths.scene_dir.resolve() != scene_dir:
        raise ValueError("Trajectory scene_dir must match navigation-scene output_dir")
    if pointcloud_config.paths.scene_dir.resolve() != scene_dir:
        raise ValueError("Pointcloud scene_dir must match navigation-scene output_dir")
    scene = build_scene(scene_config)
    trajectory_manifest = plan_trajectory_batch(trajectory_config, scene)
    pointcloud_report = export_semantic_pointcloud(pointcloud_config, scene)
    print(
        f"Prepared scene, {trajectory_manifest['trajectory_count']} trajectories, "
        f"and {pointcloud_report['pointcloud']}"
    )


if __name__ == "__main__":
    main()
