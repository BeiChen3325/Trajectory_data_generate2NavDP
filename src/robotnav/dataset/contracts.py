"""Versioned intermediate-file contracts for dataset build stages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CONTRACT_VERSION = 4
DEPTH_UNITS_PER_METER = 10_000
INVALID_DEPTH_VALUE = 0


@dataclass(frozen=True)
class CameraTrajectory:
    t_world_ground: np.ndarray
    t_ground_world: np.ndarray
    t_world_base_link: np.ndarray
    t_base_link_world: np.ndarray
    t_base_from_camera: np.ndarray
    t_camera_from_base: np.ndarray
    t_world_camera: np.ndarray
    t_camera_world: np.ndarray
    frame_index: np.ndarray
    metadata: dict[str, Any]

    @property
    def frame_count(self) -> int:
        return int(self.frame_index.shape[0])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_camera_trajectory(trajectory: CameraTrajectory) -> None:
    count = trajectory.frame_count
    if count <= 0:
        raise ValueError("Camera trajectory must contain at least one frame")
    expected_shape = (count, 4, 4)
    if trajectory.t_world_ground.shape != expected_shape:
        raise ValueError(
            f"T_world_ground must have shape {expected_shape}, got {trajectory.t_world_ground.shape}"
        )
    if trajectory.t_ground_world.shape != expected_shape:
        raise ValueError(
            f"T_ground_world must have shape {expected_shape}, got {trajectory.t_ground_world.shape}"
        )
    if trajectory.t_world_base_link.shape != expected_shape:
        raise ValueError(
            "T_world_base_link must have shape "
            f"{expected_shape}, got {trajectory.t_world_base_link.shape}"
        )
    if trajectory.t_base_link_world.shape != expected_shape:
        raise ValueError(
            "T_base_link_world must have shape "
            f"{expected_shape}, got {trajectory.t_base_link_world.shape}"
        )
    if trajectory.t_base_from_camera.shape != (4, 4):
        raise ValueError(
            "T_base_from_camera must be one static 4x4 transform, "
            f"got {trajectory.t_base_from_camera.shape}"
        )
    if trajectory.t_camera_from_base.shape != (4, 4):
        raise ValueError(
            "T_camera_from_base must be one static 4x4 transform, "
            f"got {trajectory.t_camera_from_base.shape}"
        )
    if trajectory.t_world_camera.shape != expected_shape:
        raise ValueError(
            f"T_world_camera must have shape {expected_shape}, got {trajectory.t_world_camera.shape}"
        )
    if trajectory.t_camera_world.shape != expected_shape:
        raise ValueError(
            f"T_camera_world must have shape {expected_shape}, got {trajectory.t_camera_world.shape}"
        )
    expected_index = np.arange(count, dtype=np.int64)
    if not np.array_equal(trajectory.frame_index, expected_index):
        raise ValueError("frame_index must be the contiguous sequence 0..T-1")
    if (
        not np.isfinite(trajectory.t_world_ground).all()
        or not np.isfinite(trajectory.t_ground_world).all()
        or not np.isfinite(trajectory.t_world_base_link).all()
        or not np.isfinite(trajectory.t_base_link_world).all()
        or not np.isfinite(trajectory.t_base_from_camera).all()
        or not np.isfinite(trajectory.t_camera_from_base).all()
        or not np.isfinite(trajectory.t_world_camera).all()
        or not np.isfinite(trajectory.t_camera_world).all()
    ):
        raise ValueError("Camera matrices must contain only finite values")
    identity = np.eye(4, dtype=np.float64)
    ground_products = trajectory.t_world_ground @ trajectory.t_ground_world
    if not np.allclose(ground_products, identity[None, :, :], atol=1e-5):
        raise ValueError("T_world_ground and T_ground_world are not mutual inverses")
    base_link_products = trajectory.t_world_base_link @ trajectory.t_base_link_world
    if not np.allclose(base_link_products, identity[None, :, :], atol=1e-5):
        raise ValueError("T_world_base_link and T_base_link_world are not mutual inverses")
    base_height = trajectory.metadata.get("base_height_above_floor_m")
    if not isinstance(base_height, (int, float)) or isinstance(base_height, bool) or base_height <= 0:
        raise ValueError("Camera trajectory metadata must contain a positive base_height_above_floor_m")
    t_ground_base_link = np.eye(4, dtype=np.float64)
    t_ground_base_link[2, 3] = float(base_height)
    if not np.allclose(
        trajectory.t_world_base_link,
        trajectory.t_world_ground @ t_ground_base_link[None, :, :],
        atol=1e-5,
    ):
        raise ValueError("T_world_base_link must be lifted from T_world_ground by base-link height")
    if not np.allclose(
        trajectory.t_camera_from_base,
        np.linalg.inv(trajectory.t_base_from_camera),
        atol=1e-5,
    ):
        raise ValueError("T_camera_from_base must equal inverse(T_base_from_camera)")
    camera_products = trajectory.t_world_camera @ trajectory.t_camera_world
    if not np.allclose(camera_products, identity[None, :, :], atol=1e-5):
        raise ValueError("T_world_camera and T_camera_world are not mutual inverses")
    expected_camera = trajectory.t_world_base_link @ trajectory.t_base_from_camera[None, :, :]
    if not np.allclose(trajectory.t_world_camera, expected_camera, atol=1e-5):
        raise ValueError("T_world_camera must equal T_world_base_link @ T_base_from_camera")
    for name, matrices in (
        ("T_world_ground", trajectory.t_world_ground),
        ("T_ground_world", trajectory.t_ground_world),
        ("T_world_base_link", trajectory.t_world_base_link),
        ("T_base_link_world", trajectory.t_base_link_world),
        ("T_world_camera", trajectory.t_world_camera),
        ("T_camera_world", trajectory.t_camera_world),
    ):
        if not np.allclose(matrices[:, 3, :], identity[3][None, :], atol=1e-6):
            raise ValueError(f"{name} matrices must have homogeneous last rows")
    if trajectory.metadata.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(f"Unsupported camera trajectory contract: {trajectory.metadata}")
    if trajectory.metadata.get("frame_count") != count:
        raise ValueError("Camera trajectory manifest frame_count does not match NPZ data")


def save_camera_trajectory(
    trajectory: CameraTrajectory, npz_path: Path, manifest_path: Path
) -> None:
    validate_camera_trajectory(trajectory)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        T_world_ground=trajectory.t_world_ground.astype(np.float32),
        T_ground_world=trajectory.t_ground_world.astype(np.float32),
        T_world_base_link=trajectory.t_world_base_link.astype(np.float32),
        T_base_link_world=trajectory.t_base_link_world.astype(np.float32),
        T_base_from_camera=trajectory.t_base_from_camera.astype(np.float32),
        T_camera_from_base=trajectory.t_camera_from_base.astype(np.float32),
        T_world_camera=trajectory.t_world_camera.astype(np.float32),
        T_camera_world=trajectory.t_camera_world.astype(np.float32),
        frame_index=trajectory.frame_index.astype(np.int64),
    )
    manifest_path.write_text(
        json.dumps(trajectory.metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_camera_trajectory(npz_path: Path, manifest_path: Path) -> CameraTrajectory:
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with np.load(npz_path, allow_pickle=False) as arrays:
        required = {
            "T_world_ground",
            "T_ground_world",
            "T_world_base_link",
            "T_base_link_world",
            "T_base_from_camera",
            "T_camera_from_base",
            "T_world_camera",
            "T_camera_world",
            "frame_index",
        }
        missing = required - set(arrays.files)
        if missing:
            raise ValueError(f"Camera trajectory NPZ is missing arrays: {sorted(missing)}")
        trajectory = CameraTrajectory(
            t_world_ground=np.asarray(arrays["T_world_ground"], dtype=np.float64),
            t_ground_world=np.asarray(arrays["T_ground_world"], dtype=np.float64),
            t_world_base_link=np.asarray(arrays["T_world_base_link"], dtype=np.float64),
            t_base_link_world=np.asarray(arrays["T_base_link_world"], dtype=np.float64),
            t_base_from_camera=np.asarray(arrays["T_base_from_camera"], dtype=np.float64),
            t_camera_from_base=np.asarray(arrays["T_camera_from_base"], dtype=np.float64),
            t_world_camera=np.asarray(arrays["T_world_camera"], dtype=np.float64),
            t_camera_world=np.asarray(arrays["T_camera_world"], dtype=np.float64),
            frame_index=np.asarray(arrays["frame_index"], dtype=np.int64),
            metadata=json.loads(manifest_path.read_text(encoding="utf-8")),
        )
    validate_camera_trajectory(trajectory)
    return trajectory
