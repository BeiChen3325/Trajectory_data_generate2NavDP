"""Versioned, lossless scene-obstacle model shared by planning and dataset export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCENE_OBSTACLE_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class SceneObstacleModel:
    obstacle_counts: np.ndarray
    ground_counts: np.ndarray
    raw_ground: np.ndarray
    traversable_ground: np.ndarray
    raw_obstacles: np.ndarray
    cleaned_obstacles: np.ndarray
    inflated_obstacles: np.ndarray
    planning_blocked: np.ndarray
    raw_distance_m: np.ndarray
    planning_distance_m: np.ndarray
    origin_xz: np.ndarray
    max_xz: np.ndarray
    resolution_m: float
    floor_y: float
    axis_transform: str

    @property
    def spec(self) -> dict[str, object]:
        height, width = self.cleaned_obstacles.shape
        return {
            "origin_xz": self.origin_xz,
            "max_xz": self.max_xz,
            "resolution": self.resolution_m,
            "width": width,
            "height": height,
        }


def validate_scene_obstacle_model(model: SceneObstacleModel) -> None:
    shape = model.cleaned_obstacles.shape
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("Scene obstacle maps must be non-empty 2D arrays")
    arrays = (
        model.obstacle_counts,
        model.ground_counts,
        model.raw_ground,
        model.traversable_ground,
        model.raw_obstacles,
        model.inflated_obstacles,
        model.planning_blocked,
        model.raw_distance_m,
        model.planning_distance_m,
    )
    if any(array.shape != shape for array in arrays):
        raise ValueError("All scene obstacle maps must have the same shape")
    if model.origin_xz.shape != (2,) or model.max_xz.shape != (2,):
        raise ValueError("Scene obstacle bounds must be X-Z pairs")
    if model.resolution_m <= 0 or not np.isfinite(model.floor_y):
        raise ValueError("Scene obstacle resolution and floor must be valid")
    if model.axis_transform not in {"none", "zup-to-yup"}:
        raise ValueError(f"Unsupported axis transform: {model.axis_transform}")
    if not all(
        np.isfinite(array).all() for array in (model.raw_distance_m, model.planning_distance_m)
    ):
        raise ValueError("Scene obstacle distance maps must be finite")


def save_scene_obstacle_model(path: Path, model: SceneObstacleModel) -> None:
    validate_scene_obstacle_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        contract_version=np.array(SCENE_OBSTACLE_CONTRACT_VERSION, dtype=np.int64),
        obstacle_counts=model.obstacle_counts.astype(np.uint32),
        ground_counts=model.ground_counts.astype(np.uint32),
        raw_ground=model.raw_ground.astype(np.bool_),
        traversable_ground=model.traversable_ground.astype(np.bool_),
        raw_obstacles=model.raw_obstacles.astype(np.bool_),
        cleaned_obstacles=model.cleaned_obstacles.astype(np.bool_),
        inflated_obstacles=model.inflated_obstacles.astype(np.bool_),
        planning_blocked=model.planning_blocked.astype(np.bool_),
        raw_distance_m=model.raw_distance_m.astype(np.float32),
        planning_distance_m=model.planning_distance_m.astype(np.float32),
        origin_xz=np.asarray(model.origin_xz, dtype=np.float64),
        max_xz=np.asarray(model.max_xz, dtype=np.float64),
        resolution_m=np.array(model.resolution_m, dtype=np.float64),
        floor_y=np.array(model.floor_y, dtype=np.float64),
        axis_transform=np.array(model.axis_transform),
    )


def load_scene_obstacle_model(path: Path) -> SceneObstacleModel:
    with np.load(path, allow_pickle=False) as data:
        if int(data["contract_version"]) != SCENE_OBSTACLE_CONTRACT_VERSION:
            raise ValueError(f"Unsupported scene obstacle contract: {data['contract_version']}")
        model = SceneObstacleModel(
            obstacle_counts=data["obstacle_counts"].copy(),
            ground_counts=data["ground_counts"].copy(),
            raw_ground=data["raw_ground"].copy(),
            traversable_ground=data["traversable_ground"].copy(),
            raw_obstacles=data["raw_obstacles"].copy(),
            cleaned_obstacles=data["cleaned_obstacles"].copy(),
            inflated_obstacles=data["inflated_obstacles"].copy(),
            planning_blocked=data["planning_blocked"].copy(),
            raw_distance_m=data["raw_distance_m"].copy(),
            planning_distance_m=data["planning_distance_m"].copy(),
            origin_xz=data["origin_xz"].copy(),
            max_xz=data["max_xz"].copy(),
            resolution_m=float(data["resolution_m"]),
            floor_y=float(data["floor_y"]),
            axis_transform=str(data["axis_transform"]),
        )
    validate_scene_obstacle_model(model)
    return model
