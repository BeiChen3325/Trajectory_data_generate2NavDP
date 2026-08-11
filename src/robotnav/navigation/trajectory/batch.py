"""Deterministic batch orchestration over the reusable single-route planner."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from robotnav.navigation.scene.artifact import SceneArtifact, file_sha256
from robotnav.navigation.scene.occupancy_map import grid_to_world, world_to_grid
from robotnav.navigation.trajectory.config import (
    TrajectoryGenerationConfig,
    TrajectoryRequest,
)
from robotnav.navigation.trajectory.contracts import PlannedTrajectory, ResolvedTrajectoryTask
from robotnav.navigation.trajectory.endpoint_sampler import resolve_endpoints
from robotnav.navigation.trajectory.planner import plan_trajectory
from robotnav.navigation.trajectory.valid_region import (
    build_valid_sampling_cells,
    load_valid_region,
)
from robotnav.navigation.trajectory.visualization import draw_path_debug

TRAJECTORY_MANIFEST_CONTRACT_VERSION = 2


def _endpoint_separated(
    candidate: np.ndarray,
    existing: list[np.ndarray],
    min_cells: float,
) -> bool:
    return all(float(np.linalg.norm(candidate - other)) >= min_cells for other in existing)


def _validate_candidate(
    start: np.ndarray,
    goal: np.ndarray,
    *,
    used_pairs: set[tuple[int, int, int, int]],
    used_starts: list[np.ndarray],
    used_goals: list[np.ndarray],
    resolution_m: float,
    min_start_goal_distance_m: float,
    min_endpoint_separation_m: float,
) -> str | None:
    pair = (int(start[0]), int(start[1]), int(goal[0]), int(goal[1]))
    if pair in used_pairs:
        return "duplicate snapped start/goal pair"
    if float(np.linalg.norm(start - goal)) * resolution_m < min_start_goal_distance_m:
        return "start/goal distance is below the configured minimum"
    min_cells = min_endpoint_separation_m / resolution_m
    if not _endpoint_separated(start, used_starts, min_cells):
        return "start endpoint is too close to an existing start"
    if not _endpoint_separated(goal, used_goals, min_cells):
        return "goal endpoint is too close to an existing goal"
    return None


def _trajectory_json(scene: SceneArtifact, planned: PlannedTrajectory) -> dict[str, object]:
    task = planned.task
    start_xz = grid_to_world(task.start_cell, scene.model.spec)
    goal_xz = grid_to_world(task.goal_cell, scene.model.spec)
    return {
        "contract_version": 1,
        "trajectory_id": task.trajectory_id,
        "seed": task.seed,
        "coordinate_convention": (
            "World coordinates: ground plane is X-Z; physical down is +Y and physical up is -Y."
        ),
        "floor_y": scene.model.floor_y,
        "robot_ground_pose": {
            "frame": "ground",
            "origin_y": scene.model.floor_y,
            "path_xz": planned.smooth_xz.tolist(),
            "orientation": "base_link yaw is derived from the path tangent downstream",
        },
        "source_scene_model_sha256": scene.model_sha256,
        "start_xz": start_xz.tolist(),
        "goal_xz": goal_xz.tolist(),
        "astar_path_xz": planned.astar_xz.tolist(),
        "shortcut_path_xz": planned.shortcut_xz.tolist(),
        "smooth_path_xz": planned.smooth_xz.tolist(),
        "smooth_path_collides": planned.smooth_path_collides,
        "grid_start_xy": task.start_cell.tolist(),
        "grid_goal_xy": task.goal_cell.tolist(),
    }


def _write_route(
    output_dir: Path,
    scene: SceneArtifact,
    planned: PlannedTrajectory,
) -> dict[str, object]:
    routes_dir = output_dir / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)
    route_path = routes_dir / f"{planned.task.trajectory_id}.json"
    debug_path = routes_dir / f"{planned.task.trajectory_id}_debug.png"
    payload = _trajectory_json(scene, planned)
    route_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    model = scene.model
    draw_path_debug(
        debug_path,
        model.cleaned_obstacles,
        model.inflated_obstacles,
        model.planning_blocked,
        model.planning_distance_m,
        planned.astar_cells,
        planned.shortcut_cells,
        smooth_xy=planned.smooth_xz,
        world_to_grid_fn=lambda points: world_to_grid(points, model.spec),
    )
    return {
        "trajectory_id": planned.task.trajectory_id,
        "path": route_path.relative_to(output_dir).as_posix(),
        "trajectory_sha256": file_sha256(route_path),
        "debug_image": debug_path.relative_to(output_dir).as_posix(),
        "start_xz": payload["start_xz"],
        "goal_xz": payload["goal_xz"],
        "path_length_m": planned.path_length_m,
        "point_count": int(planned.smooth_xz.shape[0]),
        "smooth_path_collides": planned.smooth_path_collides,
        "seed": planned.task.seed,
    }


def _length_rejection_reason(
    path_length_m: float,
    config: TrajectoryGenerationConfig,
) -> str | None:
    sampling = config.trajectory_sampling
    if sampling.trajectory_mode != "long":
        return None
    if path_length_m < sampling.min_length_m:
        return (
            f"path length {path_length_m:.3f} m is below "
            f"long-mode minimum {sampling.min_length_m:.3f} m"
        )
    if path_length_m > sampling.max_length_m:
        return (
            f"path length {path_length_m:.3f} m exceeds "
            f"long-mode maximum {sampling.max_length_m:.3f} m"
        )
    return None


def _length_statistics(entries: list[dict[str, object]]) -> dict[str, float]:
    lengths = np.asarray([float(entry["path_length_m"]) for entry in entries], dtype=np.float64)
    return {
        "min_m": float(lengths.min()),
        "max_m": float(lengths.max()),
        "mean_m": float(lengths.mean()),
        "std_m": float(lengths.std()),
    }


def plan_trajectory_batch(config: TrajectoryGenerationConfig, scene: SceneArtifact) -> dict:
    """Generate exactly the configured number of routes or fail explicitly."""
    output_dir = config.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = config.batch
    valid_sampling_cells = None
    valid_region_metadata: dict[str, object] | None = None
    if config.valid_region.yaml_path is not None:
        region_path = config.valid_region.yaml_path
        if not region_path.is_file():
            raise FileNotFoundError(f"valid region YAML does not exist: {region_path}")
        region = load_valid_region(region_path)
        raw_mask, valid_sampling_cells, erosion_cells = build_valid_sampling_cells(
            scene,
            region,
            robot_radius_m=config.valid_region.robot_radius_m,
            safety_margin_m=config.valid_region.safety_margin_m,
        )
        np.save(output_dir / "valid_region_mask.npy", valid_sampling_cells)
        valid_region_metadata = {
            "yaml_path": os.path.relpath(region_path, output_dir),
            "robot_radius_m": config.valid_region.robot_radius_m,
            "safety_margin_m": config.valid_region.safety_margin_m,
            "safety_distance_m": (
                config.valid_region.robot_radius_m + config.valid_region.safety_margin_m
            ),
            "erosion_cells": erosion_cells,
            "raw_region_cells": int(np.count_nonzero(raw_mask)),
            "valid_sampling_cells": int(np.count_nonzero(valid_sampling_cells)),
            "mask": "valid_region_mask.npy",
        }
    requests = list(batch.requests)
    used_ids = {request.trajectory_id for request in requests}
    auto_index = 0
    while len(requests) < batch.count:
        while f"auto_{auto_index:03d}" in used_ids:
            auto_index += 1
        trajectory_id = f"auto_{auto_index:03d}"
        used_ids.add(trajectory_id)
        requests.append(TrajectoryRequest(trajectory_id, None, None))
        auto_index += 1

    used_pairs: set[tuple[int, int, int, int]] = set()
    used_starts: list[np.ndarray] = []
    used_goals: list[np.ndarray] = []
    entries = []
    rejection_counts: dict[str, int] = {}
    attempt_counts: list[int] = []
    for route_index, request in enumerate(requests):
        is_fully_explicit = request.start_xz is not None and request.goal_xz is not None
        attempt_limit = 1 if is_fully_explicit else batch.max_sampling_attempts
        planned = None
        last_reason = "no candidate attempted"
        for attempt in range(attempt_limit):
            seed = batch.seed + route_index * 1009 + attempt
            try:
                start, goal = resolve_endpoints(
                    scene,
                    request,
                    min_distance_m=batch.min_start_goal_distance_m,
                    seed=seed,
                    valid_sampling_cells=valid_sampling_cells,
                )
                reason = _validate_candidate(
                    start,
                    goal,
                    used_pairs=used_pairs,
                    used_starts=used_starts,
                    used_goals=used_goals,
                    resolution_m=scene.model.resolution_m,
                    min_start_goal_distance_m=batch.min_start_goal_distance_m,
                    min_endpoint_separation_m=batch.min_endpoint_separation_m,
                )
                if reason is not None:
                    raise ValueError(reason)
                task = ResolvedTrajectoryTask(request.trajectory_id, start, goal, seed)
                planned = plan_trajectory(scene, task, config.planner, valid_sampling_cells)
                reason = _length_rejection_reason(planned.path_length_m, config)
                if reason is not None:
                    raise ValueError(reason)
                break
            except ValueError as error:
                last_reason = str(error)
                rejection_counts[last_reason] = rejection_counts.get(last_reason, 0) + 1
                if is_fully_explicit:
                    raise ValueError(
                        f"Explicit trajectory {request.trajectory_id!r} is invalid: {error}"
                    ) from error
        if planned is None:
            summary = ", ".join(
                f"{reason}: {count}"
                for reason, count in sorted(
                    rejection_counts.items(), key=lambda item: item[1], reverse=True
                )[:3]
            )
            raise ValueError(
                f"Generated {len(entries)}/{batch.count} trajectories; "
                f"request {request.trajectory_id!r} failed after {attempt_limit} attempts. "
                f"Last reason: {last_reason}. Rejections: {summary}"
            )
        pair = (
            int(planned.task.start_cell[0]),
            int(planned.task.start_cell[1]),
            int(planned.task.goal_cell[0]),
            int(planned.task.goal_cell[1]),
        )
        used_pairs.add(pair)
        used_starts.append(planned.task.start_cell.copy())
        used_goals.append(planned.task.goal_cell.copy())
        entries.append(_write_route(output_dir, scene, planned))
        attempt_counts.append(attempt + 1)

    manifest = {
        "contract_version": TRAJECTORY_MANIFEST_CONTRACT_VERSION,
        "requested_count": batch.count,
        "trajectory_count": len(entries),
        "batch_seed": batch.seed,
        "source_scene_model": os.path.relpath(scene.model_path, output_dir),
        "source_scene_model_sha256": scene.model_sha256,
        "planner_config": asdict(config.planner),
        "trajectory_sampling": asdict(config.trajectory_sampling),
        "sampling_statistics": {
            "attempts_total": int(sum(attempt_counts)),
            "attempts_mean": float(np.mean(attempt_counts)),
            "episode_success_rate": float(len(entries) / batch.count),
            "candidate_acceptance_rate": float(len(entries) / sum(attempt_counts)),
            "rejections": rejection_counts,
        },
        "length_statistics": _length_statistics(entries),
        "trajectories": entries,
    }
    if valid_region_metadata is not None:
        manifest["valid_region"] = valid_region_metadata
    manifest_path = output_dir / batch.manifest_filename
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
