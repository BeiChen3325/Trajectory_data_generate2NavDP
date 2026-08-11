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
    TrajectorySamplingConfig,
    ValidRegionConfig,
    load_trajectory_generation_config,
)
from robotnav.navigation.trajectory.valid_region import (
    ValidRegion,
    build_valid_sampling_cells,
    load_valid_region,
    save_valid_region,
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


def test_long_mode_rejects_paths_outside_final_length_bounds(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)
    base = make_config(tmp_path, count=2)
    config = TrajectoryGenerationConfig(
        base.paths,
        base.planner,
        base.batch,
        base.valid_region,
        TrajectorySamplingConfig("long", 5.0, 50.0),
    )
    manifest = plan_trajectory_batch(config, artifact)

    assert manifest["trajectory_sampling"]["trajectory_mode"] == "long"
    assert manifest["sampling_statistics"]["episode_success_rate"] == 1.0
    assert manifest["sampling_statistics"]["candidate_acceptance_rate"] <= 1.0
    lengths = [entry["path_length_m"] for entry in manifest["trajectories"]]
    assert all(5.0 <= length <= 50.0 for length in lengths)
    assert manifest["length_statistics"]["min_m"] == min(lengths)


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


def test_valid_region_yaml_mask_erosion_and_sampling(tmp_path: Path) -> None:
    artifact = make_artifact(tmp_path)
    region_path = tmp_path / "valid_region.yaml"
    region = ValidRegion(
        resolution_m=1.0,
        origin_xz=np.array([0.0, 0.0]),
        polygon_xz=np.array([[5.0, 5.0], [45.0, 5.0], [45.0, 45.0], [5.0, 45.0]]),
    )
    save_valid_region(region_path, region)
    raw_mask, valid_cells, erosion_cells = build_valid_sampling_cells(
        artifact,
        load_valid_region(region_path),
        robot_radius_m=1.0,
        safety_margin_m=1.0,
    )
    assert erosion_cells == 2
    assert raw_mask[5, 5]
    assert not valid_cells[5, 5]
    assert valid_cells[10, 10]

    config = make_config(tmp_path, count=2)
    config = TrajectoryGenerationConfig(
        config.paths,
        config.planner,
        config.batch,
        ValidRegionConfig(region_path, 1.0, 1.0),
    )
    manifest = plan_trajectory_batch(config, artifact)
    mask = np.load(tmp_path / "trajectories" / "valid_region_mask.npy")
    assert manifest["valid_region"]["safety_distance_m"] == 2.0
    for entry in manifest["trajectories"]:
        route = json.loads((tmp_path / "trajectories" / entry["path"]).read_text())
        cells = np.asarray(route["astar_path_xz"])
        grid = np.floor(cells).astype(int)
        assert np.all(mask[grid[:, 1], grid[:, 0]])
