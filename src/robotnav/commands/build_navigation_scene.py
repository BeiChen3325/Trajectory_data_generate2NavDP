"""CLI for the independent navigation-scene build stage."""

from __future__ import annotations

import argparse

from robotnav.navigation.scene.builder import build_scene
from robotnav.navigation.scene.config import load_scene_build_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reusable navigation scene from LAS.")
    parser.add_argument("--config", default="navigation_scene.toml")
    return parser.parse_args()


def main() -> None:
    config = load_scene_build_config(parse_args().config)
    artifact = build_scene(config)
    print(f"Built navigation scene: {artifact.scene_dir}")
    print(f"  model sha256={artifact.model_sha256}")


if __name__ == "__main__":
    main()
