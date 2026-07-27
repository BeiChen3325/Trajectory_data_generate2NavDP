"""CLI for independent semantic PLY export."""

from __future__ import annotations

import argparse

from robotnav.navigation.scene.artifact import load_scene_artifact
from robotnav.navigation.semantic_pointcloud.config import load_pointcloud_export_config
from robotnav.navigation.semantic_pointcloud.exporter import export_semantic_pointcloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a semantic source-point PLY from LAS and a navigation scene."
    )
    parser.add_argument("--config", default="pointcloud_export.toml")
    return parser.parse_args()


def main() -> None:
    config = load_pointcloud_export_config(parse_args().config)
    scene = load_scene_artifact(config.paths.scene_dir)
    report = export_semantic_pointcloud(config, scene)
    print(f"Exported semantic point cloud: {report['pointcloud']}")


if __name__ == "__main__":
    main()
