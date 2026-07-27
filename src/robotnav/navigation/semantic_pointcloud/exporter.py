"""Export the shared physical-obstacle model as a target-compatible colored PLY."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement

from robotnav.navigation.scene.artifact import (
    SceneArtifact,
    file_sha256,
    validate_source_las,
)
from robotnav.navigation.scene.contracts import SceneObstacleModel
from robotnav.navigation.scene.las_io import iter_las_xyz
from robotnav.navigation.scene.occupancy_map import world_to_grid
from robotnav.navigation.semantic_pointcloud.config import (
    PointCloudConfig,
    PointCloudExportConfig,
)

POINTCLOUD_REPORT_CONTRACT_VERSION = 2


class VoxelAccumulator:
    """Keep one deterministic source point for each occupied 3D voxel."""

    def __init__(self, voxel_size_m: float) -> None:
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        self.voxel_size_m = float(voxel_size_m)
        self._entries: dict[
            tuple[int, int, int],
            tuple[tuple[float, float, float, float], tuple[float, float, float]],
        ] = {}

    def add(self, points: np.ndarray) -> None:
        if points.size == 0:
            return
        keys = np.floor(points / self.voxel_size_m).astype(np.int64)
        centers = (keys.astype(np.float64) + 0.5) * self.voxel_size_m
        distance_squared = np.sum((points - centers) ** 2, axis=1)
        order = np.lexsort(
            (
                points[:, 2],
                points[:, 1],
                points[:, 0],
                distance_squared,
                keys[:, 2],
                keys[:, 1],
                keys[:, 0],
            )
        )
        sorted_keys = keys[order]
        first_in_voxel = np.ones(order.shape[0], dtype=bool)
        first_in_voxel[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
        for point_index in order[first_in_voxel]:
            key_array = keys[point_index]
            key = (int(key_array[0]), int(key_array[1]), int(key_array[2]))
            point = (
                float(points[point_index, 0]),
                float(points[point_index, 1]),
                float(points[point_index, 2]),
            )
            score = (float(distance_squared[point_index]), *point)
            current = self._entries.get(key)
            if current is None or score < current[0]:
                self._entries[key] = (score, point)

    def array(self) -> np.ndarray:
        if not self._entries:
            return np.empty((0, 3), dtype=np.float32)
        points = [self._entries[key][1] for key in sorted(self._entries)]
        return np.asarray(points, dtype=np.float32)


def classify_scene_points(
    xyz_yup: np.ndarray,
    model: SceneObstacleModel,
    *,
    ground_margin_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split in-bounds source points into full-height obstacles and context."""
    finite = np.isfinite(xyz_yup).all(axis=1)
    in_bounds = (
        finite
        & (xyz_yup[:, 0] >= model.origin_xz[0])
        & (xyz_yup[:, 0] <= model.max_xz[0])
        & (xyz_yup[:, 2] >= model.origin_xz[1])
        & (xyz_yup[:, 2] <= model.max_xz[1])
    )
    points = xyz_yup[in_bounds]
    if points.shape[0] == 0:
        return points, points

    ij = world_to_grid(points[:, [0, 2]], model.spec)
    height, width = model.cleaned_obstacles.shape
    in_grid = (ij[:, 0] >= 0) & (ij[:, 0] < width) & (ij[:, 1] >= 0) & (ij[:, 1] < height)
    heights = model.floor_y - points[:, 1]
    above_ground = heights >= ground_margin_m
    obstacle = np.zeros(points.shape[0], dtype=bool)
    eligible = in_grid & above_ground
    obstacle[eligible] = model.cleaned_obstacles[ij[eligible, 1], ij[eligible, 0]]
    return points[obstacle], points[~obstacle]


def build_pointcloud_points(
    chunks: Iterable[np.ndarray],
    model: SceneObstacleModel,
    config: PointCloudConfig,
    *,
    ground_margin_m: float,
    max_stream_points: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    obstacle_voxels = VoxelAccumulator(config.obstacle_voxel_size_m)
    context_voxels = VoxelAccumulator(config.context_voxel_size_m)
    seen_points = 0
    obstacle_candidates = 0
    context_candidates = 0

    for chunk in chunks:
        if max_stream_points and seen_points >= max_stream_points:
            break
        if max_stream_points and seen_points + chunk.shape[0] > max_stream_points:
            chunk = chunk[: max_stream_points - seen_points]
        seen_points += chunk.shape[0]
        obstacles, context = classify_scene_points(
            chunk,
            model,
            ground_margin_m=ground_margin_m,
        )
        obstacle_candidates += obstacles.shape[0]
        context_candidates += context.shape[0]
        obstacle_voxels.add(obstacles)
        if config.include_context:
            context_voxels.add(context)

    obstacle_points = obstacle_voxels.array()
    context_points = context_voxels.array()
    if obstacle_points.shape[0] == 0:
        raise ValueError("Shared obstacle model produced no obstacle points for pointcloud.ply")
    return (
        obstacle_points,
        context_points,
        {
            "seen_points": seen_points,
            "obstacle_candidate_points": obstacle_candidates,
            "context_candidate_points": context_candidates,
            "obstacle_representative_points": int(obstacle_points.shape[0]),
            "context_representative_points": int(context_points.shape[0]),
        },
    )


def write_colored_pointcloud(
    path: Path,
    obstacle_points: np.ndarray,
    context_points: np.ndarray,
    config: PointCloudConfig,
) -> None:
    points = np.concatenate([obstacle_points, context_points], axis=0)
    vertices = np.empty(
        points.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    for axis, index in zip(("x", "y", "z"), range(3), strict=True):
        vertices[axis] = points[:, index]
    obstacle_count = obstacle_points.shape[0]
    vertices["red"][:obstacle_count] = config.obstacle_color_rgb[0]
    vertices["green"][:obstacle_count] = config.obstacle_color_rgb[1]
    vertices["blue"][:obstacle_count] = config.obstacle_color_rgb[2]
    vertices["red"][obstacle_count:] = config.context_color_rgb[0]
    vertices["green"][obstacle_count:] = config.context_color_rgb[1]
    vertices["blue"][obstacle_count:] = config.context_color_rgb[2]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def export_las_pointcloud(
    las_path: Path,
    output_dir: Path,
    model: SceneObstacleModel,
    config: PointCloudConfig,
    *,
    ground_margin_m: float,
    chunk_size: int,
    max_stream_points: int = 0,
) -> dict[str, Any]:
    chunks = iter_las_xyz(
        las_path,
        chunk_size=chunk_size,
        axis_transform=model.axis_transform,
    )
    obstacle_points, context_points, counts = build_pointcloud_points(
        chunks,
        model,
        config,
        ground_margin_m=ground_margin_m,
        max_stream_points=max_stream_points,
    )
    pointcloud_path = output_dir / config.filename
    write_colored_pointcloud(pointcloud_path, obstacle_points, context_points, config)
    all_points = np.concatenate([obstacle_points, context_points], axis=0)
    report: dict[str, Any] = {
        "contract_version": POINTCLOUD_REPORT_CONTRACT_VERSION,
        "source_las": str(las_path),
        "pointcloud": str(pointcloud_path),
        "axis_transform": model.axis_transform,
        "floor_y": model.floor_y,
        "sampling_method": "nearest_source_point_to_voxel_center",
        "representatives_are_source_points": True,
        "bounds_xyz": {
            "min": all_points.min(axis=0).astype(float).tolist(),
            "max": all_points.max(axis=0).astype(float).tolist(),
        },
        "counts": counts,
        "pointcloud_config": asdict(config),
    }
    (output_dir / config.report_filename).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def export_semantic_pointcloud(
    config: PointCloudExportConfig,
    scene: SceneArtifact,
) -> dict[str, Any]:
    """Export PLY independently while enforcing the persisted scene/LAS hash chain."""
    source_las = validate_source_las(scene, config.paths.las_path)
    output_dir = config.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report = export_las_pointcloud(
        config.paths.las_path,
        output_dir,
        scene.model,
        config.pointcloud,
        ground_margin_m=scene.ground_margin_m,
        chunk_size=config.runtime.chunk_size,
        max_stream_points=config.runtime.max_stream_points,
    )
    report["source_las"] = source_las
    report["source_scene_model"] = str(scene.model_path.resolve())
    report["source_scene_model_sha256"] = scene.model_sha256
    report["pointcloud_sha256"] = file_sha256(output_dir / config.pointcloud.filename)
    (output_dir / config.pointcloud.report_filename).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
