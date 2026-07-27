"""Build one reusable navigation scene from LAS input."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from robotnav.navigation.scene.artifact import (
    SCENE_ARTIFACT_CONTRACT_VERSION,
    SCENE_MANIFEST_FILENAME,
    SCENE_MODEL_FILENAME,
    SceneArtifact,
    file_sha256,
    load_scene_artifact,
    source_file_record,
)
from robotnav.navigation.scene.config import SceneBuildConfig
from robotnav.navigation.scene.contracts import SceneObstacleModel, save_scene_obstacle_model
from robotnav.navigation.scene.ground_estimation import estimate_floor_y
from robotnav.navigation.scene.las_io import iter_las_xyz, parse_las_header, sample_las_xyz
from robotnav.navigation.scene.occupancy_map import (
    accumulate_ground_counts,
    accumulate_obstacle_counts,
    bounds_from_roi,
    clean_and_inflate_obstacles,
    clean_ground_mask,
    compute_bounds_from_header_yup,
    distance_transform_meters,
    filter_points_in_xz_bounds,
    make_grid_spec,
    save_map_debug,
)


@dataclass(frozen=True)
class _OccupancyConfig:
    open_kernel_cells: int
    close_kernel_cells: int
    min_obstacle_component_cells: int
    ground_close_kernel_cells: int
    robot_radius_m: float
    safety_margin_m: float
    resolution_m: float


def build_scene(config: SceneBuildConfig) -> SceneArtifact:
    """Build and persist a scene model without planning routes or exporting PLY."""
    scene = config.scene
    robot = config.robot
    las_path = config.paths.las_path
    output_dir = config.paths.output_dir
    debug_dir = output_dir / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    header = parse_las_header(las_path)
    if scene.roi_center_xz is not None and scene.roi_size_xz is not None:
        min_xz, max_xz = bounds_from_roi(scene.roi_center_xz, scene.roi_size_xz, padding=0.5)
    else:
        min_xz, max_xz = compute_bounds_from_header_yup(header, scene.axis_transform, padding=0.5)

    if scene.floor_y_override is None:
        sample = sample_las_xyz(
            las_path,
            max_points=scene.floor_sample_limit,
            chunk_size=scene.chunk_size,
            axis_transform=scene.axis_transform,
        )
        sample = filter_points_in_xz_bounds(sample, min_xz, max_xz)
        if sample.shape[0] == 0:
            raise ValueError("No sampled LAS points fall inside the configured scene bounds")
        floor_y, floor_report = estimate_floor_y(
            sample,
            bins=scene.floor_hist_bins,
            xz_resolution=scene.floor_xy_resolution_m,
            output_dir=output_dir,
            search_y_min=scene.floor_search_y_min,
            search_y_max=scene.floor_search_y_max,
        )
    else:
        floor_y = float(scene.floor_y_override)
        floor_report = {"floor_y": floor_y, "source": "manual_override"}
        (output_dir / "floor_estimation_report.json").write_text(
            json.dumps(floor_report, indent=2), encoding="utf-8"
        )

    spec = make_grid_spec(min_xz, max_xz, scene.resolution_m)
    obstacle_counts = np.zeros((spec["height"], spec["width"]), dtype=np.uint32)
    ground_counts = np.zeros_like(obstacle_counts)
    seen_points = 0
    obstacle_points = 0
    ground_points = 0
    for xyz in iter_las_xyz(
        las_path,
        chunk_size=scene.chunk_size,
        axis_transform=scene.axis_transform,
    ):
        if scene.max_stream_points and seen_points >= scene.max_stream_points:
            break
        if scene.max_stream_points and seen_points + xyz.shape[0] > scene.max_stream_points:
            xyz = xyz[: scene.max_stream_points - seen_points]
        seen_points += xyz.shape[0]
        xyz = filter_points_in_xz_bounds(xyz, min_xz, max_xz)
        obstacle_points += accumulate_obstacle_counts(
            obstacle_counts,
            xyz,
            floor_y,
            robot.height_m,
            robot.ground_margin_m,
            spec,
        )
        ground_points += accumulate_ground_counts(
            ground_counts,
            xyz,
            floor_y,
            scene.ground_band_m,
            spec,
        )

    occupancy_config = _OccupancyConfig(
        open_kernel_cells=scene.open_kernel_cells,
        close_kernel_cells=scene.close_kernel_cells,
        min_obstacle_component_cells=scene.min_obstacle_component_cells,
        ground_close_kernel_cells=scene.ground_close_kernel_cells,
        robot_radius_m=robot.radius_m,
        safety_margin_m=robot.safety_margin_m,
        resolution_m=scene.resolution_m,
    )
    raw_ground = ground_counts >= scene.min_ground_points_per_cell
    traversable_ground = clean_ground_mask(raw_ground, occupancy_config)
    raw_obstacles = obstacle_counts >= scene.min_points_per_cell
    cleaned, inflated, inflate_cells = clean_and_inflate_obstacles(raw_obstacles, occupancy_config)
    planning_blocked = inflated | (~traversable_ground)
    raw_distance_m = distance_transform_meters(cleaned, scene.resolution_m)
    planning_distance_m = distance_transform_meters(planning_blocked, scene.resolution_m)
    model = SceneObstacleModel(
        obstacle_counts=obstacle_counts,
        ground_counts=ground_counts,
        raw_ground=raw_ground,
        traversable_ground=traversable_ground,
        raw_obstacles=raw_obstacles,
        cleaned_obstacles=cleaned,
        inflated_obstacles=inflated,
        planning_blocked=planning_blocked,
        raw_distance_m=raw_distance_m,
        planning_distance_m=planning_distance_m,
        origin_xz=np.asarray(spec["origin_xz"], dtype=np.float64),
        max_xz=np.asarray(spec["max_xz"], dtype=np.float64),
        resolution_m=scene.resolution_m,
        floor_y=float(floor_y),
        axis_transform=scene.axis_transform,
    )
    model_path = output_dir / SCENE_MODEL_FILENAME
    save_scene_obstacle_model(model_path, model)
    save_map_debug(
        debug_dir,
        obstacle_counts,
        ground_counts,
        raw_ground,
        traversable_ground,
        raw_obstacles,
        cleaned,
        inflated,
        planning_blocked,
        raw_distance_m,
        spec,
        {
            "floor_report": floor_report,
            "seen_points": int(seen_points),
            "obstacle_height_points": int(obstacle_points),
            "ground_band_points": int(ground_points),
            "inflate_cells": int(inflate_cells),
            "config": config.manifest_config(),
        },
    )
    manifest = {
        "contract_version": SCENE_ARTIFACT_CONTRACT_VERSION,
        "producer": "build-scene",
        "source_las": source_file_record(las_path),
        "scene_model": SCENE_MODEL_FILENAME,
        "scene_model_sha256": file_sha256(model_path),
        "floor_y": float(floor_y),
        "axis_transform": scene.axis_transform,
        "bounds_xz": {"min": min_xz.tolist(), "max": max_xz.tolist()},
        "grid": {
            "resolution_m": scene.resolution_m,
            "width": spec["width"],
            "height": spec["height"],
        },
        "config": config.manifest_config(),
    }
    (output_dir / SCENE_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return load_scene_artifact(output_dir)
