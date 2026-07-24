"""Stage 1: convert a planned X-Z trajectory into versioned camera poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robotnav.dataset.config import DatasetBuildConfig, load_dataset_build_config
from robotnav.dataset.contracts import (
    CONTRACT_VERSION,
    CameraTrajectory,
    file_sha256,
    save_camera_trajectory,
)


def _normalize(vector: np.ndarray, *, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        raise ValueError(f"Cannot normalize zero-length {name}")
    return vector / norm


def look_at_world_to_camera(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Return the gsplat-compatible world-to-camera matrix used by existing rendering code."""
    forward = _normalize(target - eye, name="look direction")
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        alternate_up = np.array([0.0, 1.0, 0.0])
        if abs(float(forward[1])) >= 0.9:
            alternate_up = np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, alternate_up)
    right = _normalize(right, name="camera right vector")
    down = _normalize(np.cross(forward, right), name="camera down vector")

    view = np.eye(4, dtype=np.float64)
    view[:3, :3] = np.stack([right, down, forward], axis=0)
    view[:3, 3] = -(view[:3, :3] @ eye)
    return view


def path_tangents(points_xz: np.ndarray) -> np.ndarray:
    """Compute a stable tangent for every point, skipping duplicate neighbors."""
    if points_xz.ndim != 2 or points_xz.shape[1] != 2:
        raise ValueError(f"smooth_path_xz must have shape (T,2), got {points_xz.shape}")
    count = points_xz.shape[0]
    if count < 2:
        raise ValueError("smooth_path_xz must contain at least two distinct path points")

    tangents = np.zeros_like(points_xz, dtype=np.float64)
    for index in range(count):
        previous = None
        for candidate in range(index - 1, -1, -1):
            if np.linalg.norm(points_xz[index] - points_xz[candidate]) >= 1e-10:
                previous = points_xz[candidate]
                break
        following = None
        for candidate in range(index + 1, count):
            if np.linalg.norm(points_xz[candidate] - points_xz[index]) >= 1e-10:
                following = points_xz[candidate]
                break
        if previous is not None and following is not None:
            direction = following - previous
        elif following is not None:
            direction = following - points_xz[index]
        elif previous is not None:
            direction = points_xz[index] - previous
        else:
            raise ValueError("smooth_path_xz contains no distinct path points")
        tangents[index] = _normalize(direction, name=f"path tangent at frame {index}")
    return tangents


def build_camera_trajectory(
    points_xz: np.ndarray,
    floor_y: float,
    height_above_floor_m: float,
    *,
    source_trajectory: Path,
    source_sha256: str,
    coordinate_convention: str,
) -> CameraTrajectory:
    """Convert path points to mutually inverse camera matrices."""
    points_xz = np.asarray(points_xz, dtype=np.float64)
    if not np.isfinite(points_xz).all() or not np.isfinite(floor_y):
        raise ValueError("Trajectory points and floor_y must be finite")
    tangents = path_tangents(points_xz)
    count = points_xz.shape[0]
    world_to_camera = np.empty((count, 4, 4), dtype=np.float64)
    camera_to_world = np.empty_like(world_to_camera)
    up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    camera_y = float(floor_y) - height_above_floor_m

    for index, (point, tangent) in enumerate(zip(points_xz, tangents, strict=True)):
        eye = np.array([point[0], camera_y, point[1]], dtype=np.float64)
        target = eye + np.array([tangent[0], 0.0, tangent[1]], dtype=np.float64)
        view = look_at_world_to_camera(eye, target, up)
        world_to_camera[index] = view
        camera_to_world[index] = np.linalg.inv(view)

    metadata = {
        "contract_version": CONTRACT_VERSION,
        "frame_count": count,
        "coordinate_convention": coordinate_convention,
        "pose_convention": {
            "camera_to_world": "target_data action",
            "world_to_camera": "gsplat viewmat",
        },
        "source_trajectory": str(source_trajectory),
        "source_trajectory_sha256": source_sha256,
        "floor_y": float(floor_y),
        "height_above_floor_m": float(height_above_floor_m),
    }
    return CameraTrajectory(
        camera_to_world=camera_to_world,
        world_to_camera=world_to_camera,
        frame_index=np.arange(count, dtype=np.int64),
        metadata=metadata,
    )


def run_trajectory_to_camera(config: DatasetBuildConfig) -> CameraTrajectory:
    trajectory_path = config.paths.trajectory_path
    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)
    raw = json.loads(trajectory_path.read_text(encoding="utf-8"))
    for field in ("floor_y", "smooth_path_xz", "coordinate_convention"):
        if field not in raw:
            raise ValueError(f"Trajectory JSON is missing required field: {field}")
    trajectory = build_camera_trajectory(
        np.asarray(raw["smooth_path_xz"], dtype=np.float64),
        float(raw["floor_y"]),
        config.trajectory_to_camera.height_above_floor_m,
        source_trajectory=trajectory_path,
        source_sha256=file_sha256(trajectory_path),
        coordinate_convention=str(raw["coordinate_convention"]),
    )
    save_camera_trajectory(
        trajectory,
        config.paths.camera_trajectory_path,
        config.paths.camera_manifest_path,
    )
    return trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert trajectory.json into camera pose files.")
    parser.add_argument("--config", default="dataset_build.toml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_build_config(args.config)
    trajectory = run_trajectory_to_camera(config)
    print(f"Saved {trajectory.frame_count} camera poses")
    print(f"  {config.paths.camera_trajectory_path}")
    print(f"  {config.paths.camera_manifest_path}")


if __name__ == "__main__":
    main()
