from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robotnav.config import ConfigurationError
from robotnav.navigation.scene.artifact import SceneArtifact
from robotnav.navigation.scene.contracts import SceneObstacleModel
from robotnav.navigation.trajectory.batch import plan_trajectory_batch
from robotnav.navigation.trajectory.config import (
    BatchConfig,
    PlannerConfig,
    TrajectoryGenerationConfig,
    TrajectoryPaths,
    TrajectoryRequest,
    load_trajectory_generation_config,
)


def make_artifact(tmp_path: Path) -> SceneArtifact:
    shape = (50, 50)
    empty = np.zeros(shape, dtype=bool)
    free_distance = np.full(shape, 10.0, dtype=np.float32)
    model = SceneObstacleModel(
        obstacle_counts=np.zeros(shape, dtype=np.uint32),
        ground_counts=np.ones(shape, dtype=np.uint32),
        raw_ground=np.ones(shape, dtype=bool),
        traversable_ground=np.ones(shape, dtype=bool),
        raw_obstacles=empty.copy(),
        cleaned_obstacles=empty.copy(),
        inflated_obstacles=empty.copy(),
        planning_blocked=empty.copy(),
        raw_distance_m=free_distance.copy(),
        planning_distance_m=free_distance.copy(),
        origin_xz=np.array([0.0, 0.0]),
        max_xz=np.array([50.0, 50.0]),
        resolution_m=1.0,
        floor_y=1.0,
        axis_transform="none",
    )
    return SceneArtifact(
        scene_dir=tmp_path / "scene",
        model=model,
        manifest={
            "config": {"robot": {"ground_margin_m": 0.1}},
            "source_las": {"path": "unused", "size_bytes": 0, "sha256": "unused"},
        },
        model_sha256="scene-hash",
    )


def make_config(tmp_path: Path, *, count: int = 8) -> TrajectoryGenerationConfig:
    return TrajectoryGenerationConfig(
        paths=TrajectoryPaths(tmp_path / "scene", tmp_path / "trajectories"),
        planner=PlannerConfig(
            obstacle_cost_weight=0.0,
            obstacle_cost_power=1.5,
            shortcut_passes=10,
            smooth_samples_per_meter=2.0,
            smooth_iterations=1,
        ),
        batch=BatchConfig(
            count=count,
            seed=7,
            min_start_goal_distance_m=5.0,
            min_endpoint_separation_m=1.0,
            max_sampling_attempts=200,
            manifest_filename="trajectory_manifest.json",
            requests=(),
        ),
    )


def test_default_batch_config_loads() -> None:
    config = load_trajectory_generation_config()
    assert config.batch.count == 8
    assert config.batch.requests == ()
    assert not hasattr(config.paths, "las_path")


def test_batch_generates_exact_deterministic_count(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)
    first = plan_trajectory_batch(make_config(tmp_path), artifact)
    payloads = [
        json.loads((tmp_path / "trajectories" / entry["path"]).read_text())
        for entry in first["trajectories"]
    ]
    second = plan_trajectory_batch(make_config(tmp_path), artifact)

    assert first == second
    assert first["trajectory_count"] == 8
    pairs = {(tuple(item["grid_start_xy"]), tuple(item["grid_goal_xy"])) for item in payloads}
    assert len(pairs) == 8
    assert all(not item["smooth_path_collides"] for item in payloads)


def test_explicit_duplicate_endpoint_pair_fails(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)
    config = make_config(tmp_path, count=2)
    requests = (
        TrajectoryRequest("first", (2.0, 2.0), (20.0, 20.0)),
        TrajectoryRequest("second", (2.0, 2.0), (20.0, 20.0)),
    )
    config = TrajectoryGenerationConfig(
        config.paths,
        config.planner,
        BatchConfig(**{**config.batch.__dict__, "requests": requests}),
    )
    with pytest.raises(ValueError, match="Explicit trajectory"):
        plan_trajectory_batch(config, artifact)


def test_requests_cannot_exceed_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "too_many.toml"
    config_path.write_text(
        """
[paths]
scene_dir = "scene"
output_dir = "out"
[planner]
obstacle_cost_weight = 0.0
obstacle_cost_power = 1.0
shortcut_passes = 0
smooth_samples_per_meter = 1.0
smooth_iterations = 0
[trajectory_batch]
count = 1
seed = 1
min_start_goal_distance_m = 1.0
min_endpoint_separation_m = 0.0
max_sampling_attempts = 2
manifest_filename = "manifest.json"
requests = [
  { id = "a", start_xz = [], goal_xz = [] },
  { id = "b", start_xz = [], goal_xz = [] },
]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "robotnav.navigation.trajectory.config.CONFIG_DIR",
        tmp_path,
    )
    with pytest.raises(ConfigurationError, match="cannot exceed"):
        load_trajectory_generation_config("too_many.toml")
