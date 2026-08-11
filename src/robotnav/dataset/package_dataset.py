"""Stage 3: package all camera/RGB-D episodes into the target dataset contract."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from plyfile import PlyData

from robotnav.dataset.config import DatasetBuildConfig, load_dataset_build_config
from robotnav.dataset.contracts import (
    CONTRACT_VERSION,
    DEPTH_UNITS_PER_METER,
    INVALID_DEPTH_VALUE,
    CameraTrajectory,
    file_sha256,
    load_camera_trajectory,
)
from robotnav.dataset.trajectory_manifest import (
    EpisodeSpec,
    TrajectoryBatch,
    load_trajectory_batch,
)
from robotnav.navigation.semantic_pointcloud.exporter import (
    POINTCLOUD_REPORT_CONTRACT_VERSION,
)

CHUNK_NAME = "chunk-000"
OBSTACLE_COLOR = np.array([0.0, 0.0, 0.5], dtype=np.float64)
OBSTACLE_COLOR_DISTANCE = 0.05


@dataclass(frozen=True)
class EpisodeArtifacts:
    spec: EpisodeSpec
    camera: CameraTrajectory
    render_manifest: dict[str, Any]
    rgb_paths: tuple[Path, ...]
    depth_paths: tuple[Path, ...]
    intrinsic: np.ndarray


def _load_render_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "contract_version",
        "trajectory_id",
        "episode_index",
        "frame_count",
        "camera_intrinsic",
        "width",
        "height",
        "rgb_depth_alignment",
        "rgb_paths",
        "depth_paths",
        "depth_units_per_meter",
        "invalid_depth_value",
        "source_trajectory_sha256",
        "source_batch_manifest_sha256",
        "source_scene_model_sha256",
        "camera_trajectory_npz_sha256",
        "camera_trajectory_manifest_sha256",
    }
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"Render manifest is missing fields: {sorted(missing)}")
    if manifest["contract_version"] != CONTRACT_VERSION:
        raise ValueError(f"Unsupported render manifest contract: {manifest['contract_version']}")
    return manifest


def _artifact_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Artifact path must stay below render directory: {relative}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Artifact path escapes render directory: {relative}") from error
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _normalized_ply_colors(ply_path: Path) -> np.ndarray:
    if not ply_path.is_file():
        raise FileNotFoundError(ply_path)
    ply = PlyData.read(ply_path)
    if "vertex" not in ply:
        raise ValueError("pointcloud.ply must contain a vertex element")
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or ())
    required = {"x", "y", "z", "red", "green", "blue"}
    missing = required - names
    if missing:
        raise ValueError(f"pointcloud.ply is missing vertex properties: {sorted(missing)}")
    points = np.stack([vertex[axis] for axis in ("x", "y", "z")], axis=-1)
    colors = np.stack([vertex[channel] for channel in ("red", "green", "blue")], axis=-1)
    if points.shape[0] == 0 or not np.isfinite(points).all():
        raise ValueError("pointcloud.ply must contain finite points")
    if np.issubdtype(colors.dtype, np.integer):
        colors = colors.astype(np.float64) / float(np.iinfo(colors.dtype).max)
    else:
        colors = colors.astype(np.float64)
    if not np.isfinite(colors).all() or np.any(colors < 0.0) or np.any(colors > 1.0):
        raise ValueError("pointcloud.ply colors must map to the [0,1] range")
    return colors


def validate_semantic_pointcloud(ply_path: Path) -> None:
    colors = _normalized_ply_colors(ply_path)
    distances = np.linalg.norm(colors - OBSTACLE_COLOR[None, :], axis=1)
    if not np.any(distances < OBSTACLE_COLOR_DISTANCE):
        raise ValueError("pointcloud.ply contains no target obstacle-color points")


def _validate_source_images(
    render_dir: Path, manifest: dict[str, Any]
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    count = int(manifest["frame_count"])
    rgb_values = manifest["rgb_paths"]
    depth_values = manifest["depth_paths"]
    if not isinstance(rgb_values, list) or not isinstance(depth_values, list):
        raise ValueError("rgb_paths and depth_paths must be lists")
    if len(rgb_values) != count or len(depth_values) != count:
        raise ValueError("Render manifest frame_count does not match RGB/Depth path counts")
    width = int(manifest["width"])
    height = int(manifest["height"])
    rgb_paths = tuple(_artifact_path(render_dir, str(value)) for value in rgb_values)
    depth_paths = tuple(_artifact_path(render_dir, str(value)) for value in depth_values)
    for rgb_path, depth_path in zip(rgb_paths, depth_paths, strict=True):
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or rgb.dtype != np.uint8 or rgb.shape != (height, width, 3):
            raise ValueError(f"Invalid RGB image: {rgb_path}")
        if depth is None or depth.dtype != np.uint16 or depth.shape != (height, width):
            raise ValueError(f"Invalid depth image: {depth_path}")
    return rgb_paths, depth_paths


def _write_episode_parquet(
    path: Path,
    t_world_ground: np.ndarray,
    t_world_base_link: np.ndarray,
    t_world_camera: np.ndarray,
    intrinsic: np.ndarray,
    t_base_from_camera: np.ndarray,
    t_camera_from_base: np.ndarray,
    timestamp_indices: np.ndarray,
) -> None:
    count = t_world_camera.shape[0]
    if t_world_ground.shape != (count, 4, 4):
        raise ValueError("T_world_ground must align with T_world_camera")
    if t_world_base_link.shape != (count, 4, 4):
        raise ValueError("T_world_base_link must align with T_world_camera")
    if t_base_from_camera.shape != (4, 4) or t_camera_from_base.shape != (4, 4):
        raise ValueError("Camera extrinsics must be static 4x4 matrices")
    if timestamp_indices.shape != (count,):
        raise ValueError("timestamp_indices must align with T_world_camera")
    intrinsic_values = intrinsic.reshape(-1).tolist()
    t_base_from_camera_values = t_base_from_camera.reshape(-1).tolist()
    t_camera_from_base_values = t_camera_from_base.reshape(-1).tolist()
    frame = pd.DataFrame(
        {
            "timestamp_index": timestamp_indices.astype(np.int64),
            "observation.camera_intrinsic": [intrinsic_values.copy() for _ in range(count)],
            "observation.T_base_from_camera": [
                t_base_from_camera_values.copy() for _ in range(count)
            ],
            "observation.T_camera_from_base": [
                t_camera_from_base_values.copy() for _ in range(count)
            ],
            "observation.robot_ground_pose": [
                matrix.reshape(-1).tolist() for matrix in t_world_ground
            ],
            "observation.robot_base_pose": [
                matrix.reshape(-1).tolist() for matrix in t_world_base_link
            ],
            "observation.T_world_camera": [
                matrix.reshape(-1).tolist() for matrix in t_world_camera
            ],
            "action": [matrix.reshape(-1).tolist() for matrix in t_world_camera],
        }
    )
    frame.to_parquet(path, engine="pyarrow", index=False)


def _load_episode_artifacts(
    batch: TrajectoryBatch,
    episode: EpisodeSpec,
) -> EpisodeArtifacts:
    camera = load_camera_trajectory(
        episode.paths.camera_trajectory_path,
        episode.paths.camera_manifest_path,
    )
    expected_camera_metadata = {
        "trajectory_id": episode.trajectory_id,
        "episode_index": episode.episode_index,
        "source_trajectory_sha256": episode.trajectory_sha256,
        "source_batch_manifest_sha256": batch.manifest_sha256,
        "source_scene_model_sha256": batch.source_scene_model_sha256,
    }
    for field, value in expected_camera_metadata.items():
        if camera.metadata.get(field) != value:
            raise ValueError(f"Camera manifest {episode.trajectory_id!r} does not match {field}")
    render_manifest = _load_render_manifest(episode.paths.render_manifest_path)
    expected = {
        "trajectory_id": episode.trajectory_id,
        "episode_index": episode.episode_index,
        "frame_count": camera.frame_count,
        "source_trajectory_sha256": episode.trajectory_sha256,
        "source_batch_manifest_sha256": batch.manifest_sha256,
        "source_scene_model_sha256": batch.source_scene_model_sha256,
        "camera_trajectory_npz_sha256": file_sha256(episode.paths.camera_trajectory_path),
        "camera_trajectory_manifest_sha256": file_sha256(episode.paths.camera_manifest_path),
    }
    for field, value in expected.items():
        if render_manifest.get(field) != value:
            raise ValueError(f"Render manifest {episode.trajectory_id!r} does not match {field}")
    if render_manifest["depth_units_per_meter"] != DEPTH_UNITS_PER_METER:
        raise ValueError("Rendered depth unit does not match target_data.md")
    if render_manifest["invalid_depth_value"] != INVALID_DEPTH_VALUE:
        raise ValueError("Rendered invalid depth value does not match target_data.md")
    alignment = render_manifest["rgb_depth_alignment"]
    expected_alignment = {
        "pixel_coordinate_frame": "color",
        "rgb_intrinsic": "K_color",
        "depth_intrinsic": "K_color",
        "view_transform": "T_camera_world",
        "method": "same 3DGS projection",
    }
    if alignment != expected_alignment:
        raise ValueError(
            "Rendered RGB and depth do not declare the required color-camera alignment"
        )
    intrinsic = np.asarray(render_manifest["camera_intrinsic"], dtype=np.float64)
    if intrinsic.size != 9 or not np.isfinite(intrinsic).all():
        raise ValueError("Render manifest camera_intrinsic must contain 9 finite values")
    rgb_paths, depth_paths = _validate_source_images(
        episode.paths.rendered_episode_dir,
        render_manifest,
    )
    return EpisodeArtifacts(
        spec=episode,
        camera=camera,
        render_manifest=render_manifest,
        rgb_paths=rgb_paths,
        depth_paths=depth_paths,
        intrinsic=intrinsic.reshape(3, 3),
    )


def _validate_pointcloud_binding(config: DatasetBuildConfig, batch: TrajectoryBatch) -> dict:
    report_path = config.paths.semantic_pointcloud_report_path
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("contract_version") != POINTCLOUD_REPORT_CONTRACT_VERSION:
        raise ValueError(
            f"Unsupported pointcloud report contract: {report.get('contract_version')}"
        )
    if report.get("source_scene_model_sha256") != batch.source_scene_model_sha256:
        raise ValueError("Semantic pointcloud and trajectory batch belong to different scenes")
    expected_pointcloud_sha256 = report.get("pointcloud_sha256")
    if not isinstance(expected_pointcloud_sha256, str):
        raise ValueError("pointcloud_report.json has no pointcloud_sha256")
    if expected_pointcloud_sha256 != file_sha256(config.paths.semantic_pointcloud_path):
        raise ValueError("Semantic pointcloud SHA-256 does not match pointcloud_report.json")
    validate_semantic_pointcloud(config.paths.semantic_pointcloud_path)
    return report


def validate_target_scene(scene_dir: Path) -> dict[str, Any]:
    """Validate a complete multi-episode scene independently of its build inputs."""
    data_dir = scene_dir / "data" / CHUNK_NAME
    rgb_dir = scene_dir / "videos" / CHUNK_NAME / "observation.images.rgb"
    depth_dir = scene_dir / "videos" / CHUNK_NAME / "observation.images.depth"
    stats_path = scene_dir / "meta" / "episodes_stats.jsonl"
    pointcloud_path = scene_dir / "meta" / "pointcloud.ply"
    parquet_paths = sorted(data_dir.glob("*.parquet"))
    rgb_paths = sorted(rgb_dir.glob("*.png"))
    depth_paths = sorted(depth_dir.glob("*.png"))
    if not parquet_paths:
        raise ValueError("Target scene must contain at least one parquet file")
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    stats_lines = [line for line in stats_path.read_text(encoding="utf-8").splitlines() if line]
    if len(stats_lines) != len(parquet_paths):
        raise ValueError("episodes_stats.jsonl line count must match parquet count")
    if not rgb_paths or len(rgb_paths) != len(depth_paths):
        raise ValueError("RGB and Depth file counts must be equal and non-zero")

    required_columns = {
        "timestamp_index",
        "observation.camera_intrinsic",
        "observation.T_base_from_camera",
        "observation.T_camera_from_base",
        "observation.robot_ground_pose",
        "observation.robot_base_pose",
        "observation.T_world_camera",
        "action",
    }
    expected_min = 0
    episodes = []
    for index, (parquet_path, stats_line) in enumerate(
        zip(parquet_paths, stats_lines, strict=True)
    ):
        stats = json.loads(stats_line)
        image_index = stats.get("image_index")
        if not isinstance(image_index, dict):
            raise ValueError(f"Episode {index} has no image_index")
        minimum = image_index.get("min")
        maximum = image_index.get("max")
        if (
            not isinstance(minimum, int)
            or not isinstance(maximum, int)
            or minimum != expected_min
            or maximum < minimum
            or maximum >= len(rgb_paths)
        ):
            raise ValueError(f"Episode {index} has an invalid or non-contiguous image range")
        frame_count = maximum - minimum + 1
        frame = pd.read_parquet(parquet_path)
        if not required_columns.issubset(frame.columns):
            raise ValueError(f"Parquet is missing target columns: {parquet_path}")
        if len(frame) != frame_count:
            raise ValueError(f"Parquet row count does not match image range: {parquet_path}")
        intrinsic = np.asarray(frame["observation.camera_intrinsic"].tolist()[0]).reshape(3, 3)
        t_base_from_camera = np.asarray(
            frame["observation.T_base_from_camera"].tolist()[0]
        ).reshape(4, 4)
        t_camera_from_base = np.asarray(
            frame["observation.T_camera_from_base"].tolist()[0]
        ).reshape(4, 4)
        t_world_ground = np.stack(frame["observation.robot_ground_pose"].to_numpy()).reshape(
            -1, 4, 4
        )
        t_world_base_link = np.stack(frame["observation.robot_base_pose"].to_numpy()).reshape(
            -1, 4, 4
        )
        t_world_camera = np.stack(frame["observation.T_world_camera"].to_numpy()).reshape(-1, 4, 4)
        actions = np.stack(frame["action"].to_numpy()).reshape(-1, 4, 4)
        timestamps = frame["timestamp_index"].to_numpy(dtype=np.int64)
        if (
            not np.isfinite(intrinsic).all()
            or not np.isfinite(t_base_from_camera).all()
            or not np.isfinite(t_camera_from_base).all()
        ):
            raise ValueError(f"Parquet camera matrices must be finite: {parquet_path}")
        if (
            not np.isfinite(t_world_ground).all()
            or not np.isfinite(t_world_base_link).all()
            or not np.isfinite(t_world_camera).all()
            or not np.isfinite(actions).all()
            or actions.shape[0] != frame_count
        ):
            raise ValueError(f"Parquet poses do not match image range: {parquet_path}")
        if not np.array_equal(timestamps, np.arange(minimum, maximum + 1, dtype=np.int64)):
            raise ValueError(f"Parquet timestamps do not match image range: {parquet_path}")
        if not np.allclose(
            t_camera_from_base,
            np.linalg.inv(t_base_from_camera),
            atol=1e-5,
        ):
            raise ValueError(
                f"T_camera_from_base is not inverse(T_base_from_camera): {parquet_path}"
            )
        if not np.allclose(
            t_world_camera,
            t_world_base_link @ t_base_from_camera[None, :, :],
            atol=1e-5,
        ):
            raise ValueError(
                f"T_world_camera does not match robot_base_pose and extrinsic: {parquet_path}"
            )
        if not np.allclose(actions, t_world_camera, atol=1e-5):
            raise ValueError(f"Parquet action must equal T_world_camera: {parquet_path}")
        episodes.append(
            {
                "episode_index": index,
                "parquet": str(parquet_path),
                "image_index": {"min": minimum, "max": maximum},
                "frame_count": frame_count,
            }
        )
        expected_min = maximum + 1
    if expected_min != len(rgb_paths):
        raise ValueError("Episode image ranges do not cover every RGB/Depth image")

    for rgb_path, depth_path in zip(rgb_paths, depth_paths, strict=True):
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Invalid target RGB image: {rgb_path}")
        if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
            raise ValueError(f"Invalid target depth image: {depth_path}")
        if rgb.shape[:2] != depth.shape:
            raise ValueError(f"Target RGB/Depth dimensions differ at {rgb_path.name}")
    validate_semantic_pointcloud(pointcloud_path)
    return {
        "episode_count": len(episodes),
        "total_frame_count": len(rgb_paths),
        "rgb_count": len(rgb_paths),
        "depth_count": len(depth_paths),
        "pointcloud": str(pointcloud_path),
        "episodes": episodes,
    }


def run_package_dataset(config: DatasetBuildConfig) -> Path:
    batch = load_trajectory_batch(
        config.paths.trajectory_manifest,
        config.paths.episodes_dir,
    )
    _validate_pointcloud_binding(config, batch)
    artifacts = tuple(_load_episode_artifacts(batch, episode) for episode in batch.episodes)
    scene_dir = config.scene_dir
    scene_parent = scene_dir.parent
    scene_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{scene_dir.name}-staging-", dir=scene_parent))
    try:
        data_dir = staging / "data" / CHUNK_NAME
        rgb_dir = staging / "videos" / CHUNK_NAME / "observation.images.rgb"
        depth_dir = staging / "videos" / CHUNK_NAME / "observation.images.depth"
        meta_dir = staging / "meta"
        for directory in (data_dir, rgb_dir, depth_dir, meta_dir):
            directory.mkdir(parents=True, exist_ok=True)

        total_frames = sum(item.camera.frame_count for item in artifacts)
        frame_digits = max(6, len(str(total_frames - 1)))
        stats_lines = []
        manifest_episodes = []
        global_index = 0
        for item in artifacts:
            start = global_index
            for rgb_source, depth_source in zip(item.rgb_paths, item.depth_paths, strict=True):
                name = f"{global_index:0{frame_digits}d}.png"
                shutil.copy2(rgb_source, rgb_dir / name)
                shutil.copy2(depth_source, depth_dir / name)
                global_index += 1
            end = global_index - 1
            parquet_path = data_dir / f"{item.spec.episode_name}.parquet"
            _write_episode_parquet(
                parquet_path,
                item.camera.t_world_ground,
                item.camera.t_world_base_link,
                item.camera.t_world_camera,
                item.intrinsic,
                item.camera.t_base_from_camera,
                item.camera.t_camera_from_base,
                np.arange(start, end + 1, dtype=np.int64),
            )
            image_index = {"min": start, "max": end}
            stats_lines.append(json.dumps({"image_index": image_index}, separators=(",", ":")))
            manifest_episodes.append(
                {
                    "episode_index": item.spec.episode_index,
                    "episode_name": item.spec.episode_name,
                    "trajectory_id": item.spec.trajectory_id,
                    "trajectory_sha256": item.spec.trajectory_sha256,
                    "frame_count": item.camera.frame_count,
                    "image_index": image_index,
                    "parquet": parquet_path.relative_to(staging).as_posix(),
                    "render_manifest_sha256": file_sha256(item.spec.paths.render_manifest_path),
                }
            )
        (meta_dir / "episodes_stats.jsonl").write_text(
            "\n".join(stats_lines) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(config.paths.semantic_pointcloud_path, meta_dir / "pointcloud.ply")
        run_manifest = {
            "contract_version": 2,
            "trajectory_batch": {
                "path": str(batch.manifest_path),
                "sha256": batch.manifest_sha256,
                "source_scene_model_sha256": batch.source_scene_model_sha256,
            },
            "semantic_pointcloud": {
                "path": str(config.paths.semantic_pointcloud_path),
                "sha256": file_sha256(config.paths.semantic_pointcloud_path),
                "report_path": str(config.paths.semantic_pointcloud_report_path),
                "report_sha256": file_sha256(config.paths.semantic_pointcloud_report_path),
            },
            "episode_count": len(artifacts),
            "total_frame_count": total_frames,
            "chunk_name": CHUNK_NAME,
            "episodes": manifest_episodes,
            "dataset": {
                "group_dir": config.dataset.group_dir,
                "scene_dir": config.dataset.scene_dir,
            },
        }
        (meta_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2), encoding="utf-8"
        )
        validate_target_scene(staging)
        if scene_dir.exists():
            if not config.dataset.overwrite:
                raise FileExistsError(
                    f"Target scene already exists and overwrite=false: {scene_dir}"
                )
            shutil.rmtree(scene_dir)
        staging.rename(scene_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return scene_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package all manifest episodes as one target dataset scene."
    )
    parser.add_argument("--config", default="dataset_build.toml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_build_config(args.config)
    scene_dir = run_package_dataset(config)
    report = validate_target_scene(scene_dir)
    print(f"Saved target dataset scene: {scene_dir}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
