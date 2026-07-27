from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

from robotnav.dataset.config import (
    DatasetBuildConfig,
    DatasetBuildPaths,
    DatasetOutputConfig,
    EpisodeRenderingConfig,
    TrajectoryToCameraConfig,
    load_dataset_build_config,
)
from robotnav.dataset.contracts import (
    CONTRACT_VERSION,
    DEPTH_UNITS_PER_METER,
    INVALID_DEPTH_VALUE,
    file_sha256,
    save_camera_trajectory,
)
from robotnav.dataset.package_dataset import run_package_dataset, validate_target_scene
from robotnav.dataset.trajectory_to_camera import build_camera_trajectory, path_tangents


def make_config(tmp_path: Path) -> DatasetBuildConfig:
    return DatasetBuildConfig(
        paths=DatasetBuildPaths(
            trajectory_dir=tmp_path / "trajectory",
            trajectory_filename="trajectory.json",
            semantic_pointcloud_dir=tmp_path / "input",
            semantic_pointcloud_filename="pointcloud.ply",
            work_dir=tmp_path / "work",
            dataset_root=tmp_path / "target",
        ),
        trajectory_to_camera=TrajectoryToCameraConfig(
            height_above_floor_m=0.5,
            base_extrinsic=tuple(np.eye(4).reshape(-1).tolist()),
        ),
        rendering=EpisodeRenderingConfig(camera_batch_size=2),
        dataset=DatasetOutputConfig(group_dir="group", scene_dir="scene", overwrite=False),
    )


def write_semantic_ply(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.array(
        [(0.0, 0.0, 0.0, 0, 0, 128), (1.0, 0.0, 1.0, 255, 255, 255)],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)


def test_default_dataset_build_config_loads() -> None:
    config = load_dataset_build_config()
    assert config.paths.trajectory_filename == "auto_000.json"
    assert config.paths.semantic_pointcloud_filename == "pointcloud.ply"
    assert config.paths.semantic_pointcloud_dir != config.paths.trajectory_dir
    assert config.rendering.camera_batch_size > 0


def test_path_tangents_skip_duplicate_points() -> None:
    tangents = path_tangents(np.array([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]]))
    np.testing.assert_allclose(tangents, np.array([[1.0, 0.0]] * 3))


def test_camera_trajectory_preserves_path_and_inverse(tmp_path: Path) -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    trajectory = build_camera_trajectory(
        points,
        floor_y=2.0,
        height_above_floor_m=0.5,
        source_trajectory=tmp_path / "trajectory.json",
        source_sha256="test",
        coordinate_convention="Y-up",
    )
    np.testing.assert_allclose(trajectory.camera_to_world[:, 0, 3], points[:, 0])
    np.testing.assert_allclose(trajectory.camera_to_world[:, 1, 3], 1.5)
    np.testing.assert_allclose(trajectory.camera_to_world[:, 2, 3], points[:, 1])
    products = trajectory.camera_to_world @ trajectory.world_to_camera
    np.testing.assert_allclose(products, np.repeat(np.eye(4)[None, :, :], 3, axis=0), atol=1e-6)


def test_package_dataset_from_fixed_files(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    trajectory = build_camera_trajectory(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        floor_y=1.0,
        height_above_floor_m=0.5,
        source_trajectory=config.paths.trajectory_path,
        source_sha256="test",
        coordinate_convention="Y-up",
    )
    save_camera_trajectory(
        trajectory,
        config.paths.camera_trajectory_path,
        config.paths.camera_manifest_path,
    )
    render_dir = config.paths.rendered_episode_dir
    (render_dir / "rgb").mkdir(parents=True)
    (render_dir / "depth").mkdir(parents=True)
    rgb_paths = []
    depth_paths = []
    for index in range(trajectory.frame_count):
        name = f"{index:03d}.png"
        rgb_relative = f"rgb/{name}"
        depth_relative = f"depth/{name}"
        assert cv2.imwrite(str(render_dir / rgb_relative), np.zeros((3, 4, 3), np.uint8))
        assert cv2.imwrite(str(render_dir / depth_relative), np.ones((3, 4), np.uint16))
        rgb_paths.append(rgb_relative)
        depth_paths.append(depth_relative)
    render_manifest = {
        "contract_version": CONTRACT_VERSION,
        "frame_count": trajectory.frame_count,
        "camera_intrinsic": [2.0, 0.0, 2.0, 0.0, 2.0, 1.5, 0.0, 0.0, 1.0],
        "width": 4,
        "height": 3,
        "rgb_paths": rgb_paths,
        "depth_paths": depth_paths,
        "depth_units_per_meter": DEPTH_UNITS_PER_METER,
        "invalid_depth_value": INVALID_DEPTH_VALUE,
        "camera_trajectory_npz_sha256": file_sha256(config.paths.camera_trajectory_path),
        "camera_trajectory_manifest_sha256": file_sha256(config.paths.camera_manifest_path),
    }
    (render_dir / "render_manifest.json").write_text(json.dumps(render_manifest), encoding="utf-8")
    write_semantic_ply(config.paths.semantic_pointcloud_path)

    scene_dir = run_package_dataset(config)
    report = validate_target_scene(scene_dir)

    assert report["frame_count"] == 2
    assert (scene_dir / "data/chunk-000/000.parquet").is_file()
    assert (scene_dir / "meta/pointcloud.ply").is_file()
