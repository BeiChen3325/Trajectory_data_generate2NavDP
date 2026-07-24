from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData

from robotnav.navigation.config import PointCloudConfig, load_map_config
from robotnav.navigation.pointcloud_export import (
    VoxelAccumulator,
    build_pointcloud_points,
    write_colored_pointcloud,
)
from robotnav.navigation.scene_obstacles import (
    SceneObstacleModel,
    load_scene_obstacle_model,
    save_scene_obstacle_model,
)


def make_model() -> SceneObstacleModel:
    shape = (3, 3)
    cleaned = np.zeros(shape, dtype=bool)
    cleaned[1, 1] = True
    inflated = np.ones(shape, dtype=bool)
    return SceneObstacleModel(
        obstacle_counts=np.ones(shape, dtype=np.uint32),
        ground_counts=np.ones(shape, dtype=np.uint32),
        raw_ground=np.ones(shape, dtype=bool),
        traversable_ground=np.ones(shape, dtype=bool),
        raw_obstacles=cleaned.copy(),
        cleaned_obstacles=cleaned,
        inflated_obstacles=inflated,
        planning_blocked=inflated.copy(),
        raw_distance_m=np.ones(shape, dtype=np.float32),
        planning_distance_m=np.zeros(shape, dtype=np.float32),
        origin_xz=np.array([0.0, 0.0]),
        max_xz=np.array([3.0, 3.0]),
        resolution_m=1.0,
        floor_y=1.0,
        axis_transform="none",
    )


def make_config(*, include_context: bool) -> PointCloudConfig:
    return PointCloudConfig(
        enabled=True,
        filename="pointcloud.ply",
        report_filename="pointcloud_report.json",
        obstacle_color_rgb=(0, 0, 128),
        context_color_rgb=(192, 192, 192),
        obstacle_voxel_size_m=0.1,
        context_voxel_size_m=0.2,
        include_context=include_context,
    )


def test_default_pointcloud_config_loads() -> None:
    config = load_map_config().pointcloud
    assert config.enabled
    assert config.filename == "pointcloud.ply"
    assert config.obstacle_color_rgb == (0, 0, 128)
    assert config.include_context
    assert config.obstacle_voxel_size_m == 0.02
    assert config.context_voxel_size_m == 0.05


def test_scene_obstacle_model_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "occupancy_map.npz"
    save_scene_obstacle_model(path, make_model())
    loaded = load_scene_obstacle_model(path)

    np.testing.assert_array_equal(loaded.cleaned_obstacles, make_model().cleaned_obstacles)
    np.testing.assert_array_equal(loaded.inflated_obstacles, make_model().inflated_obstacles)
    assert loaded.floor_y == 1.0
    assert loaded.spec["width"] == 3


def test_voxel_accumulator_keeps_real_source_point() -> None:
    points = np.array(
        [
            [0.01, 0.02, 0.03],
            [0.049, 0.049, 0.049],
            [0.12, 0.11, 0.13],
        ],
        dtype=np.float64,
    )
    accumulator = VoxelAccumulator(0.1)
    accumulator.add(points[:1])
    accumulator.add(points[1:])
    sampled = accumulator.array()

    assert sampled.shape == (2, 3)
    np.testing.assert_allclose(sampled[0], points[1])
    np.testing.assert_allclose(sampled[1], points[2])
    assert any(np.allclose(sampled[0], point) for point in points)


def test_pointcloud_preserves_full_height_physical_obstacles(tmp_path: Path) -> None:
    points = np.array(
        [
            [1.2, 0.5, 1.2],  # cleaned physical obstacle
            [0.2, 0.5, 0.2],  # inflated/planning blocked only
            [2.2, 0.5, 2.2],  # inflated/planning blocked only
            [1.2, 0.0, 1.2],  # above robot height, still the same rigid obstacle
        ],
        dtype=np.float64,
    )
    config = make_config(include_context=True)
    obstacles, context, counts = build_pointcloud_points(
        [points],
        make_model(),
        config,
        ground_margin_m=0.1,
    )

    assert obstacles.shape == (2, 3)
    assert context.shape == (2, 3)
    assert counts["obstacle_candidate_points"] == 2
    assert counts["obstacle_representative_points"] == 2
    assert any(np.allclose(obstacle, points[3]) for obstacle in obstacles)

    path = tmp_path / "pointcloud.ply"
    write_colored_pointcloud(path, obstacles, context, config)
    vertices = PlyData.read(path)["vertex"]
    colors = np.stack([vertices[channel] for channel in ("red", "green", "blue")], axis=-1)
    np.testing.assert_array_equal(colors[0], np.array([0, 0, 128], dtype=np.uint8))
    assert np.count_nonzero(np.all(colors == np.array([0, 0, 128]), axis=1)) == 2
