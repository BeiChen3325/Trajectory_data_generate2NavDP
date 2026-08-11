"""Convert one packaged RobotNav target scene into the NavDP directory contract.

This command intentionally consumes the immutable output of ``package-dataset``.
It neither imports nor invokes any RobotNav generation stage.  NavDP camera
extrinsics are computed from RobotNav's declared ground, base, and camera pose
chain; no camera trajectory is inferred or regenerated.
"""

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
import pyarrow as pa
import pyarrow.parquet as pq
from plyfile import PlyData, PlyElement

from robotnav.dataset.contracts import DEPTH_UNITS_PER_METER
from robotnav.dataset.package_dataset import CHUNK_NAME
from robotnav.navigation.scene.artifact import load_scene_artifact
from robotnav.utils.coordinate_transform import (
    NAVDP_FROM_ROBOTNAV,
    transform_pointcloud,
    transform_pose,
)

NAVDP_CHUNK_NAME = "chunk-000"
CAMERA_INTRINSIC_COLUMN = "observation.camera_intrinsic"
CAMERA_EXTRINSIC_COLUMN = "observation.camera_extrinsic"
SOURCE_CAMERA_EXTRINSIC_COLUMN = "observation.T_base_from_camera"
CAMERA_FROM_BASE_COLUMN = "observation.T_camera_from_base"
WORLD_CAMERA_COLUMN = "observation.T_world_camera"
ACTION_COLUMN = "action"
OUTPUT_COLUMNS = (
    "index",
    "original_index",
    CAMERA_INTRINSIC_COLUMN,
    CAMERA_EXTRINSIC_COLUMN,
    SOURCE_CAMERA_EXTRINSIC_COLUMN,
    CAMERA_FROM_BASE_COLUMN,
    WORLD_CAMERA_COLUMN,
    ACTION_COLUMN,
)
RGB_WIDTH = 848
RGB_HEIGHT = 480
RGB_JPEG_QUALITY = 95
DEPTH_MIN_M = 0.1
DEPTH_MAX_M = 5.0
DEPTH_SAMPLE_SIZE = 16
OCCUPANCY_RESOLUTION_M = 0.05
BOUNDARY_SAMPLING_M = 0.025
OBSTACLE_RGB = np.array([0, 0, 128], dtype=np.uint8)
GROUND_FROM_CAMERA_TO_NAVDP = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CameraTransformStats:
    max_inverse_error: float
    ground_base_z_max_abs: float
    camera_heights: np.ndarray
    forward_trajectory_dots: np.ndarray


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def _source_paths(scene_dir: Path) -> tuple[Path, Path, Path, Path, list[Path]]:
    """Return validated source locations without interpreting pose values."""
    source_data = scene_dir / "data" / CHUNK_NAME
    source_rgb = scene_dir / "videos" / CHUNK_NAME / "observation.images.rgb"
    source_depth = scene_dir / "videos" / CHUNK_NAME / "observation.images.depth"
    source_meta = scene_dir / "meta"
    source_parquets = sorted(source_data.glob("*.parquet"))
    if not source_parquets:
        raise ValueError(f"No source parquet episodes in {source_data}")
    return source_rgb, source_depth, source_meta, source_data, source_parquets


def _validate_source_navdp_input(
    source_rgb: Path, source_depth: Path, source_meta: Path, source_parquets: list[Path]
) -> None:
    """Validate source episode/image bindings without reading legacy semantic PLY."""
    rgb_paths = sorted(source_rgb.glob("*.png"))
    depth_paths = sorted(source_depth.glob("*.png"))
    stats_path = source_meta / "episodes_stats.jsonl"
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    stats = [
        json.loads(line) for line in stats_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if not rgb_paths or len(rgb_paths) != len(depth_paths):
        raise ValueError("Source RGB and depth frame counts must be equal and non-zero")
    if len(stats) != len(source_parquets):
        raise ValueError("Source episode stats and parquet counts must be equal")
    expected_first = 0
    for index, (parquet, episode) in enumerate(zip(source_parquets, stats, strict=True)):
        image_index = episode.get("image_index", {})
        minimum, maximum = image_index.get("min"), image_index.get("max")
        if minimum != expected_first or not isinstance(maximum, int) or maximum < minimum:
            raise ValueError(f"Invalid source image range for episode {index}")
        if maximum >= len(rgb_paths):
            raise ValueError(f"Source image range exceeds available frames for episode {index}")
        if len(pd.read_parquet(parquet, engine="pyarrow")) != maximum - minimum + 1:
            raise ValueError(f"Source parquet/image range length mismatch: {parquet}")
        expected_first = maximum + 1
    if expected_first != len(rgb_paths):
        raise ValueError("Source episode ranges do not cover every RGB/depth frame")


def _validate_manifest(manifest: dict[str, Any], source_parquets: list[Path]) -> None:
    """Check only manifest bindings that are needed to avoid silent reordering."""
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(source_parquets):
        raise ValueError("run_manifest.json episode count does not match source parquet count")
    if manifest.get("chunk_name") != CHUNK_NAME:
        raise ValueError("run_manifest.json chunk_name does not match the RobotNav target contract")
    for index, (episode, parquet) in enumerate(zip(episodes, source_parquets, strict=True)):
        if not isinstance(episode, dict):
            raise ValueError(f"run_manifest episode {index} is not an object")
        if episode.get("episode_index") != index:
            raise ValueError("run_manifest episode indexes must be contiguous and sorted")
        if episode.get("parquet") != (Path("data") / CHUNK_NAME / parquet.name).as_posix():
            raise ValueError(f"run_manifest episode {index} does not bind to {parquet.name}")


def _trajectory_records(run_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load only trajectory metrics actually recorded by generate-trajectories."""
    batch = run_manifest["trajectory_batch"]
    trajectory_manifest_path = Path(str(batch["path"]))
    trajectory_manifest = _read_json(trajectory_manifest_path)
    entries = trajectory_manifest.get("trajectories")
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("trajectory_manifest.json has no valid trajectories list")
    records: dict[str, dict[str, Any]] = {}
    for entry in entries:
        trajectory_id = entry.get("trajectory_id")
        if not isinstance(trajectory_id, str) or trajectory_id in records:
            raise ValueError("trajectory_manifest.json has invalid or duplicate trajectory_id")
        required = {
            "path_length_m",
            "point_count",
            "start_xz",
            "goal_xz",
            "smooth_path_collides",
            "seed",
        }
        if not required.issubset(entry):
            raise ValueError(f"trajectory manifest has incomplete metrics for {trajectory_id}")
        route_path = trajectory_manifest_path.parent / str(entry["path"])
        route = _read_json(route_path)
        if route.get("trajectory_id") != trajectory_id or "smooth_path_xz" not in route:
            raise ValueError(f"Route metrics do not bind to trajectory {trajectory_id}")
        records[trajectory_id] = {
            "trajectory_type": "planned_smooth_xz",
            "path_length_m": float(entry["path_length_m"]),
            "trajectory_point_count": int(entry["point_count"]),
            "start_xz": entry["start_xz"],
            "goal_xz": entry["goal_xz"],
            "smooth_path_collides": bool(entry["smooth_path_collides"]),
            "seed": int(entry["seed"]),
        }
    return records


def _missing_metadata(field: str, producer: str) -> dict[str, str | None]:
    return {
        "status": "missing",
        "value": None,
        "reason": f"{field} is not recorded by the current generation artifacts",
        "producer_to_update": producer,
    }


def _metadata_validation_report(
    info: dict[str, Any], episodes: list[dict[str, Any]], episode_stats: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate metadata consistency without treating missing generator fields as values."""
    required_info = {
        "features",
        "fps",
        "trajectory_type",
        "start_frame",
        "velocity_configuration",
        "controller_version",
        "depth_encoding",
    }
    missing_info = sorted(required_info - info.keys())
    if missing_info:
        raise ValueError(f"info.json is missing required NavDP metadata: {missing_info}")
    if len(episodes) != len(episode_stats) or info["total_episodes"] != len(episodes):
        raise ValueError("Episode metadata counts are inconsistent")
    total_frames = 0
    for index, (episode, stats) in enumerate(zip(episodes, episode_stats, strict=True)):
        if episode.get("episode_index") != index or stats.get("episode_index") != index:
            raise ValueError("Episode indexes must be contiguous")
        frames = episode.get("frame_count")
        if not isinstance(frames, int) or frames <= 0 or stats.get("frames") != frames:
            raise ValueError(f"Episode {index} frame_count is inconsistent")
        if episode.get("start_frame") != stats.get("image_index", {}).get("min"):
            raise ValueError(f"Episode {index} start frame is inconsistent")
        if not isinstance(stats.get("trajectory_metrics"), dict):
            raise ValueError(f"Episode {index} lacks trajectory_metrics")
        total_frames += frames
    if info["total_frames"] != total_frames:
        raise ValueError("info.json total_frames is inconsistent")
    missing_fields = {
        key: value
        for key, value in info.items()
        if isinstance(value, dict) and value.get("status") == "missing"
    }
    return {
        "status": "valid",
        "episode_count": len(episodes),
        "frame_count": total_frames,
        "loader_readable": {
            "info_json": True,
            "episodes_jsonl": True,
            "episodes_stats_jsonl": True,
            "tasks_jsonl": True,
        },
        "missing_generation_metadata": missing_fields,
    }


def _matrix_column(
    frame: pd.DataFrame, column: str, shape: tuple[int, int], source: Path
) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"Source parquet is missing {column}: {source}")
    values = frame[column].tolist()
    try:
        matrices = np.stack(
            [
                np.asarray(
                    value.tolist() if isinstance(value, np.ndarray) else value, dtype=np.float64
                ).reshape(shape)
                for value in values
            ],
            axis=0,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{column} is not {shape} per frame: {source}") from error
    if not np.isfinite(matrices).all():
        raise ValueError(f"{column} contains non-finite values: {source}")
    return matrices


def convert_camera_extrinsic(
    t_world_ground: np.ndarray,
    t_world_base: np.ndarray,
    t_base_from_camera: np.ndarray,
) -> tuple[np.ndarray, CameraTransformStats]:
    """Legacy ground-frame camera-pose conversion helper.

    The source matrices use ``T_A_from_B`` notation.  Therefore the actual
    ground-to-camera pose is computed from source data, never inferred from an
    independently generated camera trajectory:

    ``T_ground_from_camera = inverse(T_world_ground) @ T_world_base @ T_base_from_camera``

    It remains available for standalone coordinate-transform tests.  It is
    not used for NavDP ``observation.camera_extrinsic``: that training field
    is the fixed ``T_base_from_camera`` calibration and is copied directly.
    """
    ground = np.asarray(t_world_ground, dtype=np.float64)
    base = np.asarray(t_world_base, dtype=np.float64)
    base_from_camera = np.asarray(t_base_from_camera, dtype=np.float64)
    if ground.ndim != 3 or ground.shape[1:] != (4, 4):
        raise ValueError("robot_ground_pose must have shape (T,4,4)")
    if base.shape != ground.shape:
        raise ValueError("robot_base_pose must align with robot_ground_pose")
    if base_from_camera.shape == (4, 4):
        base_from_camera = np.broadcast_to(base_from_camera, ground.shape)
    if base_from_camera.shape != ground.shape:
        raise ValueError("T_base_from_camera must be (4,4) or (T,4,4)")
    if not (
        np.isfinite(ground).all()
        and np.isfinite(base).all()
        and np.isfinite(base_from_camera).all()
    ):
        raise ValueError("Camera transform inputs must be finite")

    t_ground_from_world = np.linalg.inv(ground)
    t_ground_from_camera = t_ground_from_world @ base @ base_from_camera
    converted = GROUND_FROM_CAMERA_TO_NAVDP @ t_ground_from_camera
    identity = np.eye(4, dtype=np.float64)
    inverse_error = np.max(
        np.abs(t_ground_from_camera @ np.linalg.inv(t_ground_from_camera) - identity)
    )
    # robot_ground_pose is the base's ground projection; its origin must remain
    # at z=0 in its own ground frame rather than at the elevated base_link origin.
    ground_base = t_ground_from_world @ ground
    ground_base_z_max_abs = float(np.max(np.abs(ground_base[:, 2, 3])))
    stats = CameraTransformStats(
        max_inverse_error=float(inverse_error),
        ground_base_z_max_abs=ground_base_z_max_abs,
        camera_heights=converted[:, 2, 3].copy(),
        forward_trajectory_dots=np.empty(0, dtype=np.float64),
    )
    return converted, stats


def _camera_forward_trajectory_dots(action: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
    """Compare converted camera optical +Z with finite-difference robot XY travel.

    Camera optical-center displacement cannot represent robot trajectory when
    the camera has an extrinsic offset: during a turn, rotation of that offset
    adds motion unrelated to the robot's ground/base trajectory.
    """
    if len(action) != len(robot_pose):
        raise ValueError("action and robot_pose must have the same frame count")
    if len(action) < 2:
        return np.empty(0, dtype=np.float64)
    forward_xy = action[:-1, :2, 2]
    motion_xy = robot_pose[1:, :2, 3] - robot_pose[:-1, :2, 3]
    forward_norm = np.linalg.norm(forward_xy, axis=1)
    motion_norm = np.linalg.norm(motion_xy, axis=1)
    valid = (forward_norm > 1e-8) & (motion_norm > 1e-8)
    if not np.any(valid):
        raise ValueError(
            "No non-zero trajectory displacement is available for camera-forward validation"
        )
    return np.sum(forward_xy[valid] * motion_xy[valid], axis=1) / (
        forward_norm[valid] * motion_norm[valid]
    )


def _write_navdp_parquet(
    destination: Path,
    index: np.ndarray,
    original_index: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    t_base_from_camera: np.ndarray,
    t_camera_from_base: np.ndarray,
    t_world_camera: np.ndarray,
    action: np.ndarray,
) -> None:
    """Write local NavDP indexes while preserving RobotNav source indexes."""
    matrix3x3 = pa.list_(pa.list_(pa.float32(), 3), 3)
    matrix4x4 = pa.list_(pa.list_(pa.float32(), 4), 4)
    table = pa.table(
        {
            "index": pa.array(index, type=pa.int64()),
            "original_index": pa.array(original_index, type=pa.int64()),
            CAMERA_INTRINSIC_COLUMN: pa.array(intrinsic.tolist(), type=matrix3x3),
            CAMERA_EXTRINSIC_COLUMN: pa.array(extrinsic.tolist(), type=matrix4x4),
            SOURCE_CAMERA_EXTRINSIC_COLUMN: pa.array(
                t_base_from_camera.tolist(), type=matrix4x4
            ),
            CAMERA_FROM_BASE_COLUMN: pa.array(t_camera_from_base.tolist(), type=matrix4x4),
            WORLD_CAMERA_COLUMN: pa.array(t_world_camera.tolist(), type=matrix4x4),
            ACTION_COLUMN: pa.array(action.tolist(), type=matrix4x4),
        }
    )
    pq.write_table(table, destination)


def _convert_parquet(
    source: Path, destination: Path
) -> tuple[int, CameraTransformStats, dict[str, int]]:
    """Convert one globally indexed RobotNav episode to local NavDP indexes."""
    frame = pd.read_parquet(source, engine="pyarrow")
    if frame.empty:
        raise ValueError(f"Source parquet has no frames: {source}")
    if "timestamp_index" not in frame.columns:
        raise ValueError(f"Source parquet is missing timestamp_index: {source}")
    original_index = frame["timestamp_index"].to_numpy(dtype=np.int64)
    expected_original_index = np.arange(
        original_index[0], original_index[0] + len(frame), dtype=np.int64
    )
    if not np.array_equal(original_index, expected_original_index):
        raise ValueError(
            f"timestamp_index must be a contiguous ascending sequence within its episode: {source}"
        )
    index = np.arange(len(frame), dtype=np.int64)
    intrinsic = _matrix_column(frame, CAMERA_INTRINSIC_COLUMN, (3, 3), source).astype(
        np.float32, copy=False
    )
    if "observation.robot_ground_pose" in frame.columns:
        robot_trajectory_pose = _matrix_column(
            frame, "observation.robot_ground_pose", (4, 4), source
        )
    elif "observation.robot_base_pose" in frame.columns:
        robot_trajectory_pose = _matrix_column(frame, "observation.robot_base_pose", (4, 4), source)
    else:
        raise ValueError(
            "Source parquet needs observation.robot_ground_pose or observation.robot_base_pose "
            f"for camera-forward validation: {source}"
        )
    t_world_ground = _matrix_column(frame, "observation.robot_ground_pose", (4, 4), source)
    t_world_base = _matrix_column(frame, "observation.robot_base_pose", (4, 4), source)
    t_base_from_camera = _matrix_column(frame, SOURCE_CAMERA_EXTRINSIC_COLUMN, (4, 4), source)
    t_camera_from_base = _matrix_column(frame, CAMERA_FROM_BASE_COLUMN, (4, 4), source)
    t_world_camera = _matrix_column(frame, WORLD_CAMERA_COLUMN, (4, 4), source)
    # InternNav's trajectory reader defines camera_extrinsic as the fixed
    # calibration T_base<-camera.  It is deliberately *not* a ground/world
    # pose and therefore receives neither the NavDP world transform nor any
    # trajectory-dependent composition.
    extrinsic = t_base_from_camera.astype(np.float32, copy=False)
    identity = np.eye(4, dtype=np.float64)
    inverse_error = float(np.max(np.abs(t_base_from_camera @ t_camera_from_base - identity)))
    ground_base = np.linalg.inv(t_world_ground) @ t_world_ground
    transform_stats = CameraTransformStats(
        max_inverse_error=inverse_error,
        ground_base_z_max_abs=float(np.max(np.abs(ground_base[:, 2, 3]))),
        camera_heights=extrinsic[:, 2, 3].copy(),
        forward_trajectory_dots=np.empty(0, dtype=np.float64),
    )
    action = transform_pose(_matrix_column(frame, ACTION_COLUMN, (4, 4), source)).astype(
        np.float32, copy=False
    )
    transform_stats = CameraTransformStats(
        max_inverse_error=transform_stats.max_inverse_error,
        ground_base_z_max_abs=transform_stats.ground_base_z_max_abs,
        camera_heights=transform_stats.camera_heights,
        forward_trajectory_dots=_camera_forward_trajectory_dots(
            action, transform_pose(robot_trajectory_pose)
        ),
    )
    _write_navdp_parquet(
        destination,
        index,
        original_index,
        intrinsic,
        extrinsic,
        t_base_from_camera.astype(np.float32, copy=False),
        t_camera_from_base.astype(np.float32, copy=False),
        t_world_camera.astype(np.float32, copy=False),
        action,
    )
    return (
        len(frame),
        transform_stats,
        {
            "original_index_min": int(original_index[0]),
            "original_index_max": int(original_index[-1]),
            "converted_index_min": int(index[0]),
            "converted_index_max": int(index[-1]),
            "frame_count": len(frame),
        },
    )


def _navigation_scene_dir(run_manifest: dict[str, Any]) -> Path:
    """Follow the source manifest chain to the hash-checked collision artifact."""
    batch_record = run_manifest.get("trajectory_batch")
    if not isinstance(batch_record, dict) or not isinstance(batch_record.get("path"), str):
        raise ValueError("run_manifest.json has no trajectory_batch.path")
    trajectory_manifest_path = Path(batch_record["path"])
    trajectory_manifest = _read_json(trajectory_manifest_path)
    source_model_value = trajectory_manifest.get("source_scene_model")
    if not isinstance(source_model_value, str):
        raise ValueError("trajectory_manifest.json has no source_scene_model")
    source_model_path = Path(source_model_value)
    if not source_model_path.is_absolute():
        source_model_path = trajectory_manifest_path.parent / source_model_path
    artifact = load_scene_artifact(source_model_path.resolve().parent)
    if artifact.model_sha256 != batch_record.get("source_scene_model_sha256"):
        raise ValueError("navigation_scene SHA-256 does not match run_manifest.json")
    if artifact.model_sha256 != trajectory_manifest.get("source_scene_model_sha256"):
        raise ValueError("navigation_scene SHA-256 does not match trajectory_manifest.json")
    return artifact.scene_dir


def _occupancy_points_from_collision_geometry(scene_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Resample cleaned collision cells into NavDP XY occupancy and boundaries."""
    artifact = load_scene_artifact(scene_dir)
    model = artifact.model
    collision = model.cleaned_obstacles
    if not np.any(collision):
        raise ValueError("navigation_scene collision geometry has no cleaned obstacles")
    extent = model.max_xz - model.origin_xz
    counts = np.ceil(extent / OCCUPANCY_RESOLUTION_M).astype(np.int64)
    x_centers = model.origin_xz[0] + (np.arange(counts[0]) + 0.5) * OCCUPANCY_RESOLUTION_M
    z_centers = model.origin_xz[1] + (np.arange(counts[1]) + 0.5) * OCCUPANCY_RESOLUTION_M
    grid_x, grid_z = np.meshgrid(x_centers, z_centers, indexing="xy")
    source_i = np.floor((grid_x - model.origin_xz[0]) / model.resolution_m).astype(np.int64)
    source_j = np.floor((grid_z - model.origin_xz[1]) / model.resolution_m).astype(np.int64)
    valid = (
        (source_i >= 0)
        & (source_i < collision.shape[1])
        & (source_j >= 0)
        & (source_j < collision.shape[0])
    )
    occupancy = np.zeros_like(grid_x, dtype=bool)
    occupancy[valid] = collision[source_j[valid], source_i[valid]]
    if not np.any(occupancy):
        raise ValueError("Collision geometry disappears at requested 0.05m occupancy resolution")
    centers_xz = np.column_stack((grid_x[occupancy], grid_z[occupancy]))

    boundary_xz: list[tuple[float, float]] = []
    offsets = np.arange(0.0, OCCUPANCY_RESOLUTION_M + 1e-12, BOUNDARY_SAMPLING_M)
    height, width = occupancy.shape
    for row, column in zip(*np.nonzero(occupancy), strict=True):
        left = model.origin_xz[0] + column * OCCUPANCY_RESOLUTION_M
        bottom = model.origin_xz[1] + row * OCCUPANCY_RESOLUTION_M
        if row == 0 or not occupancy[row - 1, column]:
            boundary_xz.extend((left + offset, bottom) for offset in offsets)
        if row == height - 1 or not occupancy[row + 1, column]:
            boundary_xz.extend(
                (left + offset, bottom + OCCUPANCY_RESOLUTION_M) for offset in offsets
            )
        if column == 0 or not occupancy[row, column - 1]:
            boundary_xz.extend((left, bottom + offset) for offset in offsets)
        if column == width - 1 or not occupancy[row, column + 1]:
            boundary_xz.extend(
                (left + OCCUPANCY_RESOLUTION_M, bottom + offset) for offset in offsets
            )
    boundary = np.unique(np.round(np.asarray(boundary_xz, dtype=np.float64), 10), axis=0)
    source_plane_points = np.vstack((centers_xz, boundary))
    # Projection removes source vertical position.  Homogeneous A maps source
    # X-Z ground coordinates into NavDP X-Y; the resulting NavDP Z is exactly 0.
    raw_projected = np.column_stack(
        (source_plane_points[:, 0], np.zeros(len(source_plane_points)), source_plane_points[:, 1])
    )
    navdp_points = transform_pointcloud(raw_projected)
    if not np.allclose(navdp_points[:, 2], 0.0, atol=1e-12):
        raise ValueError("Occupancy point projection did not produce z=0")
    report = {
        "point_count": len(navdp_points),
        "xyz_range": {
            "min": navdp_points.min(axis=0).tolist(),
            "max": navdp_points.max(axis=0).tolist(),
        },
        "resolution_m": OCCUPANCY_RESOLUTION_M,
        "boundary_sampling_m": BOUNDARY_SAMPLING_M,
        "obstacle_only_ratio": 1.0,
        "occupancy_center_point_count": len(centers_xz),
        "boundary_point_count": len(boundary),
        "source_collision_geometry": "navigation_scene.cleaned_obstacles",
        "source_height_band_m": {
            "min": artifact.ground_margin_m,
            "max": float(artifact.manifest["config"]["robot"]["height_m"]),
        },
    }
    return navdp_points, report


def _write_navdp_occupancy_pointcloud(scene_dir: Path, destination: Path) -> dict[str, Any]:
    """Write only blue z=0 collision occupancy points, never semantic PLY points."""
    points, report = _occupancy_points_from_collision_geometry(scene_dir)
    points = points.astype(np.float32)
    report["xyz_range"] = {
        "min": points.min(axis=0).astype(np.float64).tolist(),
        "max": points.max(axis=0).astype(np.float64).tolist(),
    }
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"] = OBSTACLE_RGB[0]
    vertices["green"] = OBSTACLE_RGB[1]
    vertices["blue"] = OBSTACLE_RGB[2]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(destination)
    return report


def _convert_rgb_images(source_paths: list[Path], destination_dir: Path) -> dict[str, Any]:
    """Decode source PNGs and encode independent JPEG files at a fixed quality."""
    if not source_paths:
        raise ValueError("RGB source sequence is empty")
    for index, source in enumerate(source_paths):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None or image.shape != (RGB_HEIGHT, RGB_WIDTH, 3):
            raise ValueError(f"RGB source must be {RGB_WIDTH}x{RGB_HEIGHT} BGR/RGB: {source}")
        destination = destination_dir / f"{index:06d}.jpg"
        if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, RGB_JPEG_QUALITY]):
            raise ValueError(f"Failed to encode JPEG: {destination}")
        # Ensure this is a newly encoded JPEG, rather than a renamed PNG.
        if destination.read_bytes()[:3] != b"\xff\xd8\xff":
            raise ValueError(f"JPEG encoder produced an invalid JPEG header: {destination}")
        decoded = cv2.imread(str(destination), cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape != (RGB_HEIGHT, RGB_WIDTH, 3):
            raise ValueError(f"Encoded JPEG has invalid dimensions: {destination}")
    return {
        "frame_count": len(source_paths),
        "resolution": {"width": RGB_WIDTH, "height": RGB_HEIGHT},
        "format": "JPEG",
        "compression_quality": RGB_JPEG_QUALITY,
    }


def _depth_report(depth_paths: list[Path], rgb_paths: list[Path]) -> dict[str, Any]:
    """Check sampled uint16 optical-Z depth and RGB/depth pixel alignment."""
    if not depth_paths or len(depth_paths) != len(rgb_paths):
        raise ValueError("Depth and RGB frame counts must be equal and non-zero")
    sample_count = min(DEPTH_SAMPLE_SIZE, len(depth_paths))
    sample_indexes = np.sort(
        np.random.default_rng(0).choice(len(depth_paths), sample_count, replace=False)
    )
    sample_index_set = set(sample_indexes.tolist())
    sampled_values: list[np.ndarray] = []
    valid_pixel_count = 0
    pixel_count = 0
    for index, (depth_path, rgb_path) in enumerate(zip(depth_paths, rgb_paths, strict=True)):
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if depth is None or depth.dtype != np.uint16 or depth.shape != (RGB_HEIGHT, RGB_WIDTH):
            raise ValueError(f"Depth must be uint16 {RGB_WIDTH}x{RGB_HEIGHT}: {depth_path}")
        if rgb is None or rgb.shape != (RGB_HEIGHT, RGB_WIDTH, 3):
            raise ValueError(f"RGB/depth pixel alignment resolution mismatch at frame {index}")
        if np.any(depth > DEPTH_MAX_M * DEPTH_UNITS_PER_METER):
            raise ValueError(f"Depth exceeds 5m encoding range: {depth_path}")
        if np.any((depth > 0) & (depth < DEPTH_MIN_M * DEPTH_UNITS_PER_METER)):
            raise ValueError(f"Depth below 0.1m must be encoded as zero: {depth_path}")
        if index not in sample_index_set:
            continue
        depth_m = depth.astype(np.float32) / float(DEPTH_UNITS_PER_METER)
        valid = (depth_m >= DEPTH_MIN_M) & (depth_m <= DEPTH_MAX_M)
        valid_pixel_count += int(valid.sum())
        pixel_count += int(depth.size)
        if np.any(valid):
            sampled_values.append(depth_m[valid])
    if not sampled_values:
        raise ValueError("Sampled depth frames contain no values in the valid 0.1m..5m range")
    values = np.concatenate(sampled_values)
    return {
        "frame_count": len(depth_paths),
        "sampled_frame_count": sample_count,
        "min_m": float(values.min()),
        "max_m": float(values.max()),
        "valid_ratio": float(valid_pixel_count / pixel_count),
        "resolution": {"width": RGB_WIDTH, "height": RGB_HEIGHT},
        "dtype": "uint16",
        "units": "meter * 10000",
        "depth_semantics": "optical_axis_z_depth",
        "rgb_depth_alignment": "same global frame index and identical 848x480 pixel grid",
    }


def _convert_depth_images(
    source_paths: list[Path], destination_dir: Path, rgb_paths: list[Path]
) -> dict[str, Any]:
    """Threshold metric optical-Z depth and genuinely re-encode uint16 PNG files."""
    if not source_paths:
        raise ValueError("Depth source sequence is empty")
    for index, source in enumerate(source_paths):
        raw = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16 or raw.shape != (RGB_HEIGHT, RGB_WIDTH):
            raise ValueError(f"Depth source must be uint16 {RGB_WIDTH}x{RGB_HEIGHT}: {source}")
        # The source contract declares Z-depth in meter * DEPTH_UNITS_PER_METER.
        # No ray-length or Euclidean-distance conversion is applied here.
        depth_m = raw.astype(np.float32) / float(DEPTH_UNITS_PER_METER)
        valid = np.isfinite(depth_m) & (depth_m >= DEPTH_MIN_M) & (depth_m <= DEPTH_MAX_M)
        encoded = np.zeros(raw.shape, dtype=np.uint16)
        encoded[valid] = (depth_m[valid] * DEPTH_UNITS_PER_METER).astype(np.uint16)
        destination = destination_dir / f"{index:06d}.png"
        if not cv2.imwrite(str(destination), encoded):
            raise ValueError(f"Failed to encode depth PNG: {destination}")
    return _depth_report(sorted(destination_dir.glob("*.png")), rgb_paths)


def _features() -> dict[str, dict[str, Any]]:
    """Describe NavDP fields, including retained RobotNav pose provenance."""
    return {
        CAMERA_INTRINSIC_COLUMN: {"dtype": "float32", "shape": [3, 3]},
        CAMERA_EXTRINSIC_COLUMN: {"dtype": "float32", "shape": [4, 4]},
        SOURCE_CAMERA_EXTRINSIC_COLUMN: {"dtype": "float32", "shape": [4, 4]},
        CAMERA_FROM_BASE_COLUMN: {"dtype": "float32", "shape": [4, 4]},
        WORLD_CAMERA_COLUMN: {"dtype": "float32", "shape": [4, 4]},
        ACTION_COLUMN: {"dtype": "float32", "shape": [4, 4]},
        "index": {"dtype": "int64", "shape": [1]},
        "original_index": {"dtype": "int64", "shape": [1]},
    }


def validate_navdp_dataset(output_dir: Path) -> dict[str, int]:
    """Validate the subset of the NavDP contract consumed by its dataset class."""
    data_dir = output_dir / "data" / NAVDP_CHUNK_NAME
    rgb_dir = output_dir / "videos" / NAVDP_CHUNK_NAME / "observation.images.rgb"
    depth_dir = output_dir / "videos" / NAVDP_CHUNK_NAME / "observation.images.depth"
    meta_dir = output_dir / "meta"
    required_meta = (
        "camera_transform_report.json",
        "conversion_report.json",
        "info.json",
        "episodes.jsonl",
        "episodes_stats.jsonl",
        "tasks.jsonl",
        "rgb_conversion_report.json",
        "depth_conversion_report.json",
        "pointcloud_report.json",
        "metadata_validation_report.json",
        "pointcloud.ply",
    )
    missing = [name for name in required_meta if not (meta_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing NavDP metadata: {missing}")
    parquets = sorted(data_dir.glob("episode_*.parquet"))
    rgb_paths = sorted(rgb_dir.glob("*.jpg"))
    depth_paths = sorted(depth_dir.glob("*.png"))
    stats = _read_jsonl(meta_dir / "episodes_stats.jsonl")
    if not parquets or len(parquets) != len(stats):
        raise ValueError("NavDP parquet and episodes_stats counts must be equal and non-zero")
    if not rgb_paths or len(rgb_paths) != len(depth_paths):
        raise ValueError("NavDP RGB and depth image counts must be equal and non-zero")
    if list(rgb_dir.glob("*.png")) or len(rgb_paths) != len(list(rgb_dir.iterdir())):
        raise ValueError("NavDP RGB directory must contain only converted JPEG frames")
    rgb_report = _read_json(meta_dir / "rgb_conversion_report.json")
    expected_rgb_report = {
        "frame_count": len(rgb_paths),
        "resolution": {"width": RGB_WIDTH, "height": RGB_HEIGHT},
        "format": "JPEG",
        "compression_quality": RGB_JPEG_QUALITY,
    }
    if rgb_report != expected_rgb_report:
        raise ValueError("rgb_conversion_report.json does not match RGB JPEG output")
    for rgb_path in rgb_paths:
        if rgb_path.read_bytes()[:3] != b"\xff\xd8\xff":
            raise ValueError(f"RGB output is not JPEG encoded: {rgb_path}")
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None or rgb.shape != (RGB_HEIGHT, RGB_WIDTH, 3):
            raise ValueError(f"RGB JPEG has invalid resolution: {rgb_path}")
    depth_report = _depth_report(depth_paths, rgb_paths)
    if _read_json(meta_dir / "depth_conversion_report.json") != depth_report:
        raise ValueError("depth_conversion_report.json does not match converted depth output")
    pointcloud = PlyData.read(meta_dir / "pointcloud.ply")
    if "vertex" not in pointcloud:
        raise ValueError("NavDP occupancy pointcloud has no vertex element")
    vertices = pointcloud["vertex"].data
    names = set(vertices.dtype.names or ())
    if not {"x", "y", "z", "red", "green", "blue"}.issubset(names):
        raise ValueError("NavDP occupancy pointcloud lacks XYZ/RGB fields")
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float64)
    colors = np.column_stack((vertices["red"], vertices["green"], vertices["blue"]))
    if len(points) == 0 or not np.isfinite(points).all() or not np.allclose(points[:, 2], 0.0):
        raise ValueError("NavDP occupancy pointcloud must contain finite z=0 points only")
    if not np.array_equal(colors, np.broadcast_to(OBSTACLE_RGB, colors.shape)):
        raise ValueError("NavDP occupancy pointcloud must contain only [0,0,128] obstacle colors")
    pointcloud_report = _read_json(meta_dir / "pointcloud_report.json")
    expected_pointcloud_report = {
        **pointcloud_report,
        "point_count": len(points),
        "xyz_range": {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()},
        "resolution_m": OCCUPANCY_RESOLUTION_M,
        "boundary_sampling_m": BOUNDARY_SAMPLING_M,
        "obstacle_only_ratio": 1.0,
    }
    if pointcloud_report != expected_pointcloud_report:
        raise ValueError("pointcloud_report.json does not match NavDP occupancy pointcloud")
    expected_first = 0
    frame_total = 0
    intrinsic_shape: list[int] | None = None
    extrinsic_shape: list[int] | None = None
    action_shape: list[int] | None = None
    camera_extrinsics: list[np.ndarray] = []
    episode_index_mappings: list[dict[str, int]] = []
    for episode_index, (parquet, stat) in enumerate(zip(parquets, stats, strict=True)):
        image_index = stat.get("image_index", {})
        minimum, maximum = image_index.get("min"), image_index.get("max")
        if minimum != expected_first or not isinstance(maximum, int) or maximum < minimum:
            raise ValueError(f"Invalid image range for NavDP episode {episode_index}")
        arrow_schema = pq.read_schema(parquet)
        if arrow_schema.names != list(OUTPUT_COLUMNS):
            raise ValueError(f"NavDP parquet must contain only {OUTPUT_COLUMNS}: {parquet}")
        if arrow_schema.field("index").type != pa.int64():
            raise ValueError(f"index must be int64: {parquet}")
        if arrow_schema.field("original_index").type != pa.int64():
            raise ValueError(f"original_index must be int64: {parquet}")
        expected_matrix_types = {
            CAMERA_INTRINSIC_COLUMN: pa.list_(pa.list_(pa.float32(), 3), 3),
            CAMERA_EXTRINSIC_COLUMN: pa.list_(pa.list_(pa.float32(), 4), 4),
            SOURCE_CAMERA_EXTRINSIC_COLUMN: pa.list_(pa.list_(pa.float32(), 4), 4),
            CAMERA_FROM_BASE_COLUMN: pa.list_(pa.list_(pa.float32(), 4), 4),
            WORLD_CAMERA_COLUMN: pa.list_(pa.list_(pa.float32(), 4), 4),
            ACTION_COLUMN: pa.list_(pa.list_(pa.float32(), 4), 4),
        }
        for column, expected_type in expected_matrix_types.items():
            if arrow_schema.field(column).type != expected_type:
                raise ValueError(
                    f"{column} must be nested float32 with its declared matrix shape: {parquet}"
                )
        frame = pd.read_parquet(parquet, engine="pyarrow")
        for column, shape in (
            (CAMERA_INTRINSIC_COLUMN, (3, 3)),
            (CAMERA_EXTRINSIC_COLUMN, (4, 4)),
            (SOURCE_CAMERA_EXTRINSIC_COLUMN, (4, 4)),
            (CAMERA_FROM_BASE_COLUMN, (4, 4)),
            (WORLD_CAMERA_COLUMN, (4, 4)),
            (ACTION_COLUMN, (4, 4)),
        ):
            _matrix_column(frame, column, shape, parquet)
        if len(frame) != maximum - minimum + 1:
            raise ValueError(f"Parquet/image range length mismatch: {parquet}")
        if not np.array_equal(frame["index"].to_numpy(dtype=np.int64), np.arange(len(frame))):
            raise ValueError(f"index must be the local contiguous sequence 0..T-1: {parquet}")
        original_index = frame["original_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(
            original_index,
            np.arange(original_index[0], original_index[0] + len(frame), dtype=np.int64),
        ):
            raise ValueError(f"original_index must be contiguous within its episode: {parquet}")
        intrinsic_shape = list(
            _matrix_column(frame, CAMERA_INTRINSIC_COLUMN, (3, 3), parquet).shape[1:]
        )
        extrinsic_shape = list(
            _matrix_column(frame, CAMERA_EXTRINSIC_COLUMN, (4, 4), parquet).shape[1:]
        )
        extrinsic = _matrix_column(frame, CAMERA_EXTRINSIC_COLUMN, (4, 4), parquet)
        source_extrinsic = _matrix_column(frame, SOURCE_CAMERA_EXTRINSIC_COLUMN, (4, 4), parquet)
        camera_from_base = _matrix_column(frame, CAMERA_FROM_BASE_COLUMN, (4, 4), parquet)
        world_camera = _matrix_column(frame, WORLD_CAMERA_COLUMN, (4, 4), parquet)
        action = _matrix_column(frame, ACTION_COLUMN, (4, 4), parquet)
        if not np.allclose(extrinsic, source_extrinsic, atol=1e-6):
            raise ValueError(
                f"{CAMERA_EXTRINSIC_COLUMN} must directly equal {SOURCE_CAMERA_EXTRINSIC_COLUMN}: {parquet}"
            )
        if not np.allclose(extrinsic @ camera_from_base, np.eye(4), atol=1e-5):
            raise ValueError(f"camera extrinsic and T_camera_from_base are not inverses: {parquet}")
        if not np.allclose(extrinsic[:, 3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
            raise ValueError(f"camera extrinsic has an invalid homogeneous bottom row: {parquet}")
        rotation = extrinsic[:, :3, :3]
        if not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-5):
            raise ValueError(f"camera extrinsic rotation is not orthonormal: {parquet}")
        if not np.allclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError(f"camera extrinsic rotation determinant is not +1: {parquet}")
        if not np.isfinite(world_camera).all():
            raise ValueError(f"T_world_camera must be finite: {parquet}")
        action_shape = list(action.shape[1:])
        camera_extrinsics.append(extrinsic)
        episode_index_mappings.append(
            {
                "episode_index": episode_index,
                "original_index_range": {
                    "min": int(original_index[0]),
                    "max": int(original_index[-1]),
                },
                "converted_timestamp_range": {"min": 0, "max": int(len(frame) - 1)},
                "frame_count": len(frame),
            }
        )
        expected_first = maximum + 1
        frame_total += len(frame)
    if expected_first != len(rgb_paths):
        raise ValueError("Episode image ranges do not cover all NavDP images")
    report = {
        "frames": frame_total,
        "index_valid": True,
        "original_index_valid": True,
        "episode_count": len(episode_index_mappings),
        "episodes": episode_index_mappings,
        "intrinsic_shape": intrinsic_shape,
        "extrinsic_shape": extrinsic_shape,
        "action_shape": action_shape,
    }
    report_path = meta_dir / "conversion_report.json"
    existing_report = _read_json(report_path)
    if existing_report != report:
        raise ValueError(
            f"conversion_report.json does not match converted parquet content: {report_path}"
        )
    extrinsic_values = np.concatenate(camera_extrinsics, axis=0)
    identity = np.eye(4, dtype=np.float64)
    max_inverse_error = float(
        np.max(np.abs(extrinsic_values @ np.linalg.inv(extrinsic_values) - identity))
    )
    heights = extrinsic_values[:, 2, 3]
    camera_report = _read_json(meta_dir / "camera_transform_report.json")
    expected_camera_fields = {
        "max_inverse_error",
        "camera_height_mean",
        "camera_height_std",
        "pose_count",
        "camera_forward_trajectory_min_dot",
        "camera_forward_trajectory_mean_dot",
    }
    if not expected_camera_fields.issubset(camera_report):
        raise ValueError("camera_transform_report.json is missing required fields")
    if camera_report["pose_count"] != frame_total:
        raise ValueError("camera_transform_report pose_count does not match parquet frames")
    if not np.isclose(camera_report["max_inverse_error"], max_inverse_error, atol=1e-6):
        raise ValueError("camera_transform_report max_inverse_error does not match parquet")
    if not np.isclose(camera_report["camera_height_mean"], heights.mean(), atol=1e-6):
        raise ValueError("camera_transform_report camera_height_mean does not match parquet")
    if not np.isclose(camera_report["camera_height_std"], heights.std(), atol=1e-6):
        raise ValueError("camera_transform_report camera_height_std does not match parquet")
    if camera_report.get("ground_base_z_max_abs", float("inf")) > 1e-6:
        raise ValueError("Converted source ground-base z validation failed")
    if camera_report["max_inverse_error"] > 1e-5:
        raise ValueError("Converted camera extrinsics have excessive inverse error")
    if camera_report["camera_height_std"] > 1e-5:
        raise ValueError("Converted camera height is not constant")
    if (
        camera_report["camera_forward_trajectory_min_dot"] <= 0.0
        or camera_report["camera_forward_trajectory_mean_dot"] < 0.99
    ):
        raise ValueError("Converted camera forward does not align with trajectory direction")
    metadata_report = _metadata_validation_report(
        _read_json(meta_dir / "info.json"),
        _read_jsonl(meta_dir / "episodes.jsonl"),
        stats,
    )
    if _read_json(meta_dir / "metadata_validation_report.json") != metadata_report:
        raise ValueError("metadata_validation_report.json does not match metadata content")
    return report


def convert_to_navdp_dataset(input_dir: Path, output_dir: Path, *, overwrite: bool = False) -> Path:
    """Create one NavDP scene directory from one validated RobotNav target scene."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("--input and --output must be different directories")
    source_rgb, source_depth, source_meta, _, source_parquets = _source_paths(input_dir)
    # Deliberately excludes legacy pointcloud.ply: NavDP occupancy is rebuilt
    # from navigation_scene collision geometry below.
    _validate_source_navdp_input(source_rgb, source_depth, source_meta, source_parquets)
    manifest = _read_json(source_meta / "run_manifest.json")
    _validate_manifest(manifest, source_parquets)
    trajectory_records = _trajectory_records(manifest)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite): {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-staging-", dir=output_dir.parent))
    try:
        data_dir = staging / "data" / NAVDP_CHUNK_NAME
        rgb_dir = staging / "videos" / NAVDP_CHUNK_NAME / "observation.images.rgb"
        depth_dir = staging / "videos" / NAVDP_CHUNK_NAME / "observation.images.depth"
        meta_dir = staging / "meta"
        for directory in (data_dir, rgb_dir, depth_dir, meta_dir):
            directory.mkdir(parents=True, exist_ok=True)

        rgb_sources = sorted(source_rgb.glob("*.png"))
        depth_sources = sorted(source_depth.glob("*.png"))
        rgb_report = _convert_rgb_images(rgb_sources, rgb_dir)
        depth_report = _convert_depth_images(
            depth_sources, depth_dir, sorted(rgb_dir.glob("*.jpg"))
        )
        navigation_scene_dir = _navigation_scene_dir(manifest)
        pointcloud_report = _write_navdp_occupancy_pointcloud(
            navigation_scene_dir, meta_dir / "pointcloud.ply"
        )

        episode_stats: list[dict[str, Any]] = []
        episodes: list[dict[str, Any]] = []
        camera_stats: list[CameraTransformStats] = []
        index_mappings: list[dict[str, Any]] = []
        first_image = 0
        source_manifest_episodes = manifest["episodes"]
        for episode_index, source in enumerate(source_parquets):
            source_episode = source_manifest_episodes[episode_index]
            trajectory_id = source_episode["trajectory_id"]
            if trajectory_id not in trajectory_records:
                raise ValueError(f"No recorded trajectory metrics for {trajectory_id}")
            trajectory_metrics = trajectory_records[trajectory_id]
            destination = data_dir / f"episode_{episode_index:06d}.parquet"
            frame_count, transform_stats, index_mapping = _convert_parquet(source, destination)
            camera_stats.append(transform_stats)
            index_mappings.append(
                {
                    "episode_index": episode_index,
                    "original_index_range": {
                        "min": index_mapping["original_index_min"],
                        "max": index_mapping["original_index_max"],
                    },
                    "converted_timestamp_range": {
                        "min": index_mapping["converted_index_min"],
                        "max": index_mapping["converted_index_max"],
                    },
                    "frame_count": index_mapping["frame_count"],
                }
            )
            last_image = first_image + frame_count - 1
            episode_stats.append(
                {
                    "episode_index": episode_index,
                    "frames": frame_count,
                    "image_index": {"min": first_image, "max": last_image},
                    "trajectory_metrics": trajectory_metrics,
                }
            )
            # RobotNav target data has no language task field. Preserve that fact.
            episodes.append(
                {
                    "episode_id": f"episode_{episode_index:06d}",
                    "episode_index": episode_index,
                    "trajectory_id": trajectory_id,
                    "frame_count": frame_count,
                    "start_frame": first_image,
                    "tasks": [],
                }
            )
            first_image = last_image + 1

        _jsonl(meta_dir / "episodes_stats.jsonl", episode_stats)
        _jsonl(meta_dir / "episodes.jsonl", episodes)
        # A valid, empty JSONL collection is readable by the NavDP loader and
        # does not fabricate a text-navigation task absent from source data.
        _jsonl(meta_dir / "tasks.jsonl", [])
        conversion_report = {
            "frames": first_image,
            "index_valid": True,
            "original_index_valid": True,
            "episode_count": len(index_mappings),
            "episodes": index_mappings,
            "intrinsic_shape": [3, 3],
            "extrinsic_shape": [4, 4],
            "action_shape": [4, 4],
        }
        (meta_dir / "conversion_report.json").write_text(
            json.dumps(conversion_report, indent=2) + "\n", encoding="utf-8"
        )
        (meta_dir / "rgb_conversion_report.json").write_text(
            json.dumps(rgb_report, indent=2) + "\n", encoding="utf-8"
        )
        (meta_dir / "depth_conversion_report.json").write_text(
            json.dumps(depth_report, indent=2) + "\n", encoding="utf-8"
        )
        (meta_dir / "pointcloud_report.json").write_text(
            json.dumps(pointcloud_report, indent=2) + "\n", encoding="utf-8"
        )
        camera_heights = np.concatenate([item.camera_heights for item in camera_stats])
        forward_dots = np.concatenate([item.forward_trajectory_dots for item in camera_stats])
        camera_report = {
            "max_inverse_error": max(item.max_inverse_error for item in camera_stats),
            "camera_height_mean": float(camera_heights.mean()),
            "camera_height_std": float(camera_heights.std()),
            "pose_count": first_image,
            "ground_base_z_max_abs": max(item.ground_base_z_max_abs for item in camera_stats),
            "camera_forward_trajectory_min_dot": float(forward_dots.min()),
            "camera_forward_trajectory_mean_dot": float(forward_dots.mean()),
        }
        if camera_report["ground_base_z_max_abs"] > 1e-6:
            raise ValueError("robot_ground_pose does not place its base ground origin at z=0")
        if camera_report["max_inverse_error"] > 1e-5:
            raise ValueError("T_base_from_camera and T_camera_from_base are not numerically inverse")
        if camera_report["camera_height_std"] > 1e-5:
            raise ValueError("camera height varies across source poses")
        if (
            camera_report["camera_forward_trajectory_min_dot"] <= 0.0
            or camera_report["camera_forward_trajectory_mean_dot"] < 0.99
        ):
            raise ValueError("camera forward is not aligned with the converted trajectory")
        (meta_dir / "camera_transform_report.json").write_text(
            json.dumps(camera_report, indent=2) + "\n", encoding="utf-8"
        )
        info = {
            "codebase_version": "v2.1",
            "robot_type": "robotnav-go2",
            "total_episodes": len(source_parquets),
            "total_frames": first_image,
            "total_tasks": 0,
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": _missing_metadata("fps", "trajectory-to-camera or render-trajectory"),
            "splits": {"train": f"0:{len(source_parquets)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": _features(),
            "trajectory_type": {
                "status": "recorded",
                "value": "planned_smooth_xz",
                "source": "generate-trajectories trajectory manifest and route JSON",
            },
            "start_frame": {"status": "recorded", "value": 0},
            "velocity_configuration": _missing_metadata(
                "velocity configuration and measured speed statistics", "trajectory-to-camera"
            ),
            "controller_version": _missing_metadata("controller version", "trajectory-to-camera"),
            "depth_encoding": {
                "status": "recorded",
                "dtype": "uint16",
                "units": "meter * 10000",
                "projection": "optical_axis_z_depth",
                "valid_range_m": [DEPTH_MIN_M, DEPTH_MAX_M],
                "invalid_value": 0,
                "source": "NavDP converter depth conversion",
            },
            "robotnav_conversion": {
                "source_scene": str(input_dir),
                "source_run_manifest": "meta/run_manifest.json",
                "camera_extrinsic_mapping": {
                    "destination_column": CAMERA_EXTRINSIC_COLUMN,
                    "source_columns": [SOURCE_CAMERA_EXTRINSIC_COLUMN],
                    "operation": "identity: T_base_from_camera",
                    "transform_convention": "T_base<-camera fixed calibration, column vectors, row-major float32 [4,4]",
                },
                "task_mapping": "No source language task exists; tasks.jsonl and every episode tasks list are empty.",
                "index_mapping": "Contiguous RobotNav timestamp_index values are preserved as original_index; NavDP index is regenerated locally as 0..T-1 per episode.",
                "coordinate_mapping": {
                    "matrix": NAVDP_FROM_ROBOTNAV.tolist(),
                    "action": "A @ T_world_camera",
                    "pointcloud": "A @ [x, y, z, 1]^T",
                    "camera_extrinsic": "not transformed; directly retained T_base_from_camera",
                },
                "pointcloud_mapping": {
                    "source": "navigation_scene.cleaned_obstacles; not semantic_pointcloud",
                    "projection": "source X-Z collision occupancy -> homogeneous A -> NavDP X-Y; z=0",
                    "resolution_m": OCCUPANCY_RESOLUTION_M,
                    "boundary_sampling_m": BOUNDARY_SAMPLING_M,
                },
            },
        }
        (meta_dir / "info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        metadata_report = _metadata_validation_report(info, episodes, episode_stats)
        (meta_dir / "metadata_validation_report.json").write_text(
            json.dumps(metadata_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        validate_navdp_dataset(staging)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a packaged RobotNav target scene to NavDP format."
    )
    parser.add_argument("--input", type=Path, required=True, help="RobotNav target scene directory")
    parser.add_argument("--output", type=Path, required=True, help="New NavDP scene directory")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output directory atomically"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = convert_to_navdp_dataset(args.input, args.output, overwrite=args.overwrite)
    report = validate_navdp_dataset(output)
    print(f"Saved NavDP dataset: {output}")
    print(f"Frames: {report['frames']}, index_valid: {report['index_valid']}")


if __name__ == "__main__":
    main()
