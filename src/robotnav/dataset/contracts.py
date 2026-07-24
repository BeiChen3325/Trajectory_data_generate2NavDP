"""Versioned intermediate-file contracts for dataset build stages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CONTRACT_VERSION = 1
DEPTH_UNITS_PER_METER = 10_000
INVALID_DEPTH_VALUE = 0


@dataclass(frozen=True)
class CameraTrajectory:
    camera_to_world: np.ndarray
    world_to_camera: np.ndarray
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
    if trajectory.camera_to_world.shape != expected_shape:
        raise ValueError(
            f"camera_to_world must have shape {expected_shape}, got {trajectory.camera_to_world.shape}"
        )
    if trajectory.world_to_camera.shape != expected_shape:
        raise ValueError(
            f"world_to_camera must have shape {expected_shape}, got {trajectory.world_to_camera.shape}"
        )
    expected_index = np.arange(count, dtype=np.int64)
    if not np.array_equal(trajectory.frame_index, expected_index):
        raise ValueError("frame_index must be the contiguous sequence 0..T-1")
    if (
        not np.isfinite(trajectory.camera_to_world).all()
        or not np.isfinite(trajectory.world_to_camera).all()
    ):
        raise ValueError("Camera matrices must contain only finite values")
    identity = np.eye(4, dtype=np.float64)
    products = trajectory.camera_to_world @ trajectory.world_to_camera
    if not np.allclose(products, identity[None, :, :], atol=1e-5):
        raise ValueError("camera_to_world and world_to_camera are not mutual inverses")
    for name, matrices in (
        ("camera_to_world", trajectory.camera_to_world),
        ("world_to_camera", trajectory.world_to_camera),
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
        camera_to_world=trajectory.camera_to_world.astype(np.float32),
        world_to_camera=trajectory.world_to_camera.astype(np.float32),
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
        required = {"camera_to_world", "world_to_camera", "frame_index"}
        missing = required - set(arrays.files)
        if missing:
            raise ValueError(f"Camera trajectory NPZ is missing arrays: {sorted(missing)}")
        trajectory = CameraTrajectory(
            camera_to_world=np.asarray(arrays["camera_to_world"], dtype=np.float64),
            world_to_camera=np.asarray(arrays["world_to_camera"], dtype=np.float64),
            frame_index=np.asarray(arrays["frame_index"], dtype=np.int64),
            metadata=json.loads(manifest_path.read_text(encoding="utf-8")),
        )
    validate_camera_trajectory(trajectory)
    return trajectory
