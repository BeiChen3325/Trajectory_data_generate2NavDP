from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
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
)
from robotnav.dataset.package_dataset import run_package_dataset, validate_target_scene
from robotnav.dataset.trajectory_manifest import load_trajectory_batch
from robotnav.dataset.trajectory_to_camera import (
    build_camera_trajectory,
    path_tangents,
    run_trajectory_to_camera,
)

SCENE_SHA256 = "scene-hash"


def make_config(tmp_path: Path) -> DatasetBuildConfig:
    return DatasetBuildConfig(
        paths=DatasetBuildPaths(
            trajectory_manifest=tmp_path / "trajectories" / "trajectory_manifest.json",
            semantic_pointcloud_dir=tmp_path / "input",
            semantic_pointcloud_filename="pointcloud.ply",
            semantic_pointcloud_report_filename="pointcloud_report.json",
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


def write_trajectory_batch(config: DatasetBuildConfig, lengths: tuple[int, ...]) -> None:
    manifest_path = config.paths.trajectory_manifest
    routes_dir = manifest_path.parent / "routes"
    routes_dir.mkdir(parents=True)
    entries = []
    for index, length in enumerate(lengths):
        trajectory_id = f"route_{index:03d}"
        route_path = routes_dir / f"{trajectory_id}.json"
        route = {
            "contract_version": 1,
            "trajectory_id": trajectory_id,
            "source_scene_model_sha256": SCENE_SHA256,
            "floor_y": 1.0,
            "coordinate_convention": "Y-up test",
            "smooth_path_xz": [[float(frame), float(index)] for frame in range(length)],
        }
        route_path.write_text(json.dumps(route), encoding="utf-8")
        entries.append(
            {
                "trajectory_id": trajectory_id,
                "path": route_path.relative_to(manifest_path.parent).as_posix(),
                "trajectory_sha256": file_sha256(route_path),
            }
        )
    manifest = {
        "contract_version": 1,
        "requested_count": len(entries),
        "trajectory_count": len(entries),
        "source_scene_model_sha256": SCENE_SHA256,
        "trajectories": entries,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


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


def write_pointcloud_inputs(config: DatasetBuildConfig) -> None:
    write_semantic_ply(config.paths.semantic_pointcloud_path)
    report = {
        "contract_version": 2,
        "source_scene_model_sha256": SCENE_SHA256,
        "pointcloud_sha256": file_sha256(config.paths.semantic_pointcloud_path),
    }
    config.paths.semantic_pointcloud_report_path.write_text(json.dumps(report), encoding="utf-8")


def write_rendered_episodes(config: DatasetBuildConfig) -> None:
    batch = load_trajectory_batch(
        config.paths.trajectory_manifest,
        config.paths.episodes_dir,
    )
    for episode in batch.episodes:
        render_dir = episode.paths.rendered_episode_dir
        (render_dir / "rgb").mkdir(parents=True)
        (render_dir / "depth").mkdir(parents=True)
        rgb_paths = []
        depth_paths = []
        for index in range(episode.points_xz.shape[0]):
            name = f"{index:03d}.png"
            rgb_relative = f"rgb/{name}"
            depth_relative = f"depth/{name}"
            assert cv2.imwrite(str(render_dir / rgb_relative), np.zeros((3, 4, 3), np.uint8))
            assert cv2.imwrite(str(render_dir / depth_relative), np.ones((3, 4), np.uint16))
            rgb_paths.append(rgb_relative)
            depth_paths.append(depth_relative)
        render_manifest = {
            "contract_version": CONTRACT_VERSION,
            "trajectory_id": episode.trajectory_id,
            "episode_index": episode.episode_index,
            "frame_count": int(episode.points_xz.shape[0]),
            "camera_intrinsic": [2.0, 0.0, 2.0, 0.0, 2.0, 1.5, 0.0, 0.0, 1.0],
            "width": 4,
            "height": 3,
            "rgb_paths": rgb_paths,
            "depth_paths": depth_paths,
            "depth_units_per_meter": DEPTH_UNITS_PER_METER,
            "invalid_depth_value": INVALID_DEPTH_VALUE,
            "source_trajectory_sha256": episode.trajectory_sha256,
            "source_batch_manifest_sha256": batch.manifest_sha256,
            "source_scene_model_sha256": batch.source_scene_model_sha256,
            "camera_trajectory_npz_sha256": file_sha256(episode.paths.camera_trajectory_path),
            "camera_trajectory_manifest_sha256": file_sha256(episode.paths.camera_manifest_path),
        }
        episode.paths.render_manifest_path.write_text(json.dumps(render_manifest), encoding="utf-8")


def test_default_dataset_build_config_loads() -> None:
    config = load_dataset_build_config()
    assert config.paths.trajectory_manifest.name == "trajectory_manifest.json"
    assert config.paths.semantic_pointcloud_filename == "pointcloud.ply"
    assert config.paths.semantic_pointcloud_report_filename == "pointcloud_report.json"
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


def test_manifest_hash_mismatch_fails_before_conversion(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    write_trajectory_batch(config, (2, 3))
    route = config.paths.trajectory_manifest.parent / "routes/route_001.json"
    route.write_text(route.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_trajectory_to_camera(config)


def test_camera_conversion_isolates_multiple_episodes(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    write_trajectory_batch(config, (2, 3))
    batch, trajectories = run_trajectory_to_camera(config)
    assert [item.frame_count for item in trajectories] == [2, 3]
    assert len({episode.paths.root for episode in batch.episodes}) == 2
    assert all(episode.paths.camera_trajectory_path.is_file() for episode in batch.episodes)
    assert config.paths.batch_manifest_path.is_file()


def test_package_multiple_episodes_from_fixed_files(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    write_trajectory_batch(config, (2, 3))
    run_trajectory_to_camera(config)
    write_rendered_episodes(config)
    write_pointcloud_inputs(config)

    scene_dir = run_package_dataset(config)
    report = validate_target_scene(scene_dir)

    assert report["episode_count"] == 2
    assert report["total_frame_count"] == 5
    assert [item["image_index"] for item in report["episodes"]] == [
        {"min": 0, "max": 1},
        {"min": 2, "max": 4},
    ]
    assert (scene_dir / "data/chunk-000/000.parquet").is_file()
    assert (scene_dir / "data/chunk-000/001.parquet").is_file()
    assert len(list((scene_dir / "videos/chunk-000/observation.images.rgb").glob("*.png"))) == 5
    assert (scene_dir / "meta/pointcloud.ply").is_file()


def test_pointcloud_scene_mismatch_prevents_publish(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    write_trajectory_batch(config, (2,))
    run_trajectory_to_camera(config)
    write_rendered_episodes(config)
    write_pointcloud_inputs(config)
    report = json.loads(config.paths.semantic_pointcloud_report_path.read_text())
    report["source_scene_model_sha256"] = "another-scene"
    config.paths.semantic_pointcloud_report_path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="different scenes"):
        run_package_dataset(config)
    assert not config.scene_dir.exists()


def test_missing_second_episode_image_prevents_publish(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    write_trajectory_batch(config, (2, 3))
    run_trajectory_to_camera(config)
    write_rendered_episodes(config)
    write_pointcloud_inputs(config)
    batch = load_trajectory_batch(
        config.paths.trajectory_manifest,
        config.paths.episodes_dir,
    )
    missing = batch.episodes[1].paths.rendered_episode_dir / "rgb/001.png"
    missing.unlink()

    with pytest.raises(FileNotFoundError):
        run_package_dataset(config)
    assert not config.scene_dir.exists()
