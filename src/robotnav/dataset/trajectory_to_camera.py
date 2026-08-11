"""Stage 1: convert a planned X-Z trajectory into versioned camera poses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from robotnav.dataset.batch_manifest import write_batch_manifest
from robotnav.dataset.config import DatasetBuildConfig, load_dataset_build_config
from robotnav.dataset.contracts import (
    CONTRACT_VERSION,
    CameraTrajectory,
    save_camera_trajectory,
)
from robotnav.dataset.trajectory_manifest import (
    EpisodeSpec,
    TrajectoryBatch,
    load_trajectory_batch,
)


def _normalize(vector: np.ndarray, *, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        raise ValueError(f"Cannot normalize zero-length {name}")
    return vector / norm


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


def rotation_from_rpy_degrees(rpy_deg: tuple[float, float, float]) -> np.ndarray:
    """Return the standard ROS fixed-axis Rz(yaw) @ Ry(pitch) @ Rx(roll) rotation."""
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def go2_camera_link_to_optical_rotation() -> np.ndarray:
    """Return R_camera_link_camera_optical for ROS link axes to pinhole optical axes.

    Go2 link axes are X forward, Y left, Z up. The calibrated pinhole and gsplat
    camera axes are X right, Y down, Z forward. Its columns express optical axes
    in camera-link coordinates, so it maps p_camera_optical to p_camera_link.
    """
    return np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)


def go2_t_base_from_camera(
    translation_m: tuple[float, float, float],
    rpy_deg: tuple[float, float, float],
) -> np.ndarray:
    """Build T_base_from_camera for a ROS optical camera from the Go2 camera-link pose.

    The resource pose is expressed in the Go2/ROS base frame (X forward, Y left,
    Z up). gsplat uses optical axes (X right, Y down, Z forward), so the fixed
    link-to-optical rotation is applied after the calibrated camera-link RPY.
    """
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_from_rpy_degrees(rpy_deg) @ go2_camera_link_to_optical_rotation()
    transform[:3, 3] = np.asarray(translation_m, dtype=np.float64)
    return transform


def go2_t_world_from_ground(
    point_xz: np.ndarray,
    tangent_xz: np.ndarray,
    floor_y: float,
) -> np.ndarray:
    """Build the yaw-aligned ground frame at the base-link ground projection."""
    forward_x, forward_z = tangent_xz
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [
            [forward_x, -forward_z, 0.0],
            [0.0, 0.0, -1.0],
            [forward_z, forward_x, 0.0],
        ],
        dtype=np.float64,
    )
    transform[:3, 3] = [
        float(point_xz[0]),
        float(floor_y),
        float(point_xz[1]),
    ]
    return transform


def go2_t_world_from_base_link(
    point_xz: np.ndarray,
    tangent_xz: np.ndarray,
    floor_y: float,
    base_height_above_floor_m: float,
) -> np.ndarray:
    """Build ``T_world_base_link`` from Go2 base-link height above the floor."""
    t_world_ground = go2_t_world_from_ground(point_xz, tangent_xz, floor_y)
    t_world_base_link = t_world_ground.copy()
    t_world_base_link[:3, 3] += (
        t_world_ground[:3, :3] @ np.array([0.0, 0.0, base_height_above_floor_m])
    )
    return t_world_base_link


def renderer_t_camera_from_world(t_world_camera: np.ndarray) -> np.ndarray:
    """Convert T_world_camera into the renderer view matrix T_camera_world.

    Both gsplat and the LAS debug renderer use p_camera = T_camera_world @
    p_world_h and project with the camera's positive Z axis as forward depth.
    """
    t_world_camera = np.asarray(t_world_camera, dtype=np.float64)
    if t_world_camera.shape != (4, 4) or not np.isfinite(t_world_camera).all():
        raise ValueError("T_world_camera must be a finite 4x4 matrix")
    return np.linalg.inv(t_world_camera)


def build_camera_trajectory(
    points_xz: np.ndarray,
    floor_y: float,
    base_height_above_floor_m: float,
    t_base_from_camera: np.ndarray,
    *,
    source_trajectory: Path,
    source_sha256: str,
    coordinate_convention: str,
    extra_metadata: dict[str, object] | None = None,
) -> CameraTrajectory:
    """Convert ground paths into distinct ground, base-link, and camera poses."""
    points_xz = np.asarray(points_xz, dtype=np.float64)
    if not np.isfinite(points_xz).all() or not np.isfinite(floor_y):
        raise ValueError("Trajectory points and floor_y must be finite")
    tangents = path_tangents(points_xz)
    count = points_xz.shape[0]
    t_base_from_camera = np.asarray(t_base_from_camera, dtype=np.float64)
    if t_base_from_camera.shape != (4, 4) or not np.isfinite(t_base_from_camera).all():
        raise ValueError("T_base_from_camera must be a finite 4x4 matrix")
    t_camera_from_base = np.linalg.inv(t_base_from_camera)
    t_world_ground = np.empty((count, 4, 4), dtype=np.float64)
    t_ground_world = np.empty_like(t_world_ground)
    t_world_base_link = np.empty_like(t_world_ground)
    t_base_link_world = np.empty_like(t_world_ground)
    t_world_camera = np.empty_like(t_world_ground)
    t_camera_world = np.empty_like(t_world_ground)

    for index, (point, tangent) in enumerate(zip(points_xz, tangents, strict=True)):
        t_world_ground_at_frame = go2_t_world_from_ground(
            point,
            tangent,
            floor_y,
        )
        t_world_base_link_at_frame = go2_t_world_from_base_link(
            point,
            tangent,
            floor_y,
            base_height_above_floor_m,
        )
        t_world_camera_at_frame = t_world_base_link_at_frame @ t_base_from_camera
        t_world_ground[index] = t_world_ground_at_frame
        t_ground_world[index] = np.linalg.inv(t_world_ground_at_frame)
        t_world_base_link[index] = t_world_base_link_at_frame
        t_base_link_world[index] = np.linalg.inv(t_world_base_link_at_frame)
        t_world_camera[index] = t_world_camera_at_frame
        t_camera_world[index] = renderer_t_camera_from_world(t_world_camera_at_frame)

    metadata = {
        "contract_version": CONTRACT_VERSION,
        "frame_count": count,
        "coordinate_convention": coordinate_convention,
        "pose_convention": {
            "transform_notation": "T_A_B maps homogeneous p_B to p_A",
            "T_world_ground": "ground-frame coordinates to world coordinates",
            "T_ground_world": "world coordinates to ground-frame coordinates",
            "T_world_base_link": "Go2 base_link coordinates to world coordinates",
            "T_base_link_world": "world coordinates to Go2 base_link coordinates",
            "T_base_from_camera": "camera optical coordinates to base coordinates",
            "T_camera_from_base": "base coordinates to camera optical coordinates",
            "T_world_camera": "camera optical coordinates to world coordinates; target_data action",
            "T_camera_world": "world coordinates to camera optical coordinates; gsplat and LAS renderer view matrix",
        },
        "source_trajectory": str(source_trajectory),
        "source_trajectory_sha256": source_sha256,
        "floor_y": float(floor_y),
        "base_height_above_floor_m": float(base_height_above_floor_m),
        "T_base_from_camera": t_base_from_camera.tolist(),
        "T_camera_from_base": t_camera_from_base.tolist(),
        "robot_ground_pose_convention": (
            "T_world_ground; origin is the base_link vertical projection onto floor_y; "
            "axes follow base_link yaw"
        ),
        "robot_base_pose_convention": (
            "T_world_base_link; origin is Go2 base_link; axes are X forward, Y left, Z up"
        ),
        "camera_pose_convention": "T_world_camera; camera axes are X right, Y down, Z forward",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return CameraTrajectory(
        t_world_ground=t_world_ground,
        t_ground_world=t_ground_world,
        t_world_base_link=t_world_base_link,
        t_base_link_world=t_base_link_world,
        t_base_from_camera=t_base_from_camera,
        t_camera_from_base=t_camera_from_base,
        t_world_camera=t_world_camera,
        t_camera_world=t_camera_world,
        frame_index=np.arange(count, dtype=np.int64),
        metadata=metadata,
    )


def build_episode_camera_trajectory(
    episode: EpisodeSpec,
    batch: TrajectoryBatch,
    config: DatasetBuildConfig,
) -> CameraTrajectory:
    """Build and persist one episode using the shared pose algorithm."""
    camera_config = config.trajectory_to_camera
    t_base_from_camera = go2_t_base_from_camera(
        camera_config.base_from_camera_link_translation_m,
        camera_config.base_from_camera_link_rpy_deg,
    )
    trajectory = build_camera_trajectory(
        episode.points_xz,
        episode.floor_y,
        camera_config.base_height_above_floor_m,
        t_base_from_camera,
        source_trajectory=episode.trajectory_path,
        source_sha256=episode.trajectory_sha256,
        coordinate_convention=episode.coordinate_convention,
        extra_metadata={
            "trajectory_id": episode.trajectory_id,
            "episode_index": episode.episode_index,
            "episode_name": episode.episode_name,
            "source_batch_manifest": str(batch.manifest_path),
            "source_batch_manifest_sha256": batch.manifest_sha256,
            "source_scene_model_sha256": batch.source_scene_model_sha256,
            "camera_frame": camera_config.camera_frame,
            "camera_pose_resource": camera_config.camera_pose_resource,
            "base_from_camera_link_translation_m": list(
                camera_config.base_from_camera_link_translation_m
            ),
            "base_from_camera_link_rpy_deg": list(camera_config.base_from_camera_link_rpy_deg),
        },
    )
    save_camera_trajectory(
        trajectory,
        episode.paths.camera_trajectory_path,
        episode.paths.camera_manifest_path,
    )
    return trajectory


def run_trajectory_to_camera(
    config: DatasetBuildConfig,
) -> tuple[TrajectoryBatch, tuple[CameraTrajectory, ...]]:
    batch = load_trajectory_batch(
        config.paths.trajectory_manifest,
        config.paths.episodes_dir,
    )
    trajectories = tuple(
        build_episode_camera_trajectory(episode, batch, config) for episode in batch.episodes
    )
    write_batch_manifest(batch, config.paths.batch_manifest_path)
    return batch, trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a trajectory manifest into per-episode camera pose files."
    )
    parser.add_argument("--config", default="dataset_build.toml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_build_config(args.config)
    batch, trajectories = run_trajectory_to_camera(config)
    print(f"Saved {sum(item.frame_count for item in trajectories)} camera poses")
    print(f"  episodes: {len(batch.episodes)}")
    print(f"  work directory: {config.paths.episodes_dir}")


if __name__ == "__main__":
    main()
