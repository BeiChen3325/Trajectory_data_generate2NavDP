"""Resolve fixed and automatic route endpoints against one free-space map."""

from __future__ import annotations

import cv2
import numpy as np

from robotnav.navigation.scene.artifact import SceneArtifact
from robotnav.navigation.scene.occupancy_map import world_to_grid
from robotnav.navigation.trajectory.astar import choose_auto_start_goal, nearest_free_cell
from robotnav.navigation.trajectory.config import TrajectoryRequest


def _component_labels(free: np.ndarray) -> np.ndarray:
    _, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=8)
    return labels


def _auto_in_component(
    scene: SceneArtifact,
    fixed_cell: np.ndarray,
    *,
    min_distance_m: float,
    seed: int,
    free: np.ndarray | None = None,
) -> np.ndarray:
    model = scene.model
    labels = _component_labels(~model.planning_blocked if free is None else free)
    label = int(labels[fixed_cell[1], fixed_cell[0]])
    candidates = np.argwhere(labels == label)
    if candidates.shape[0] < 2:
        raise ValueError("The fixed endpoint has no reachable automatic counterpart")
    fixed_yx = fixed_cell[[1, 0]].astype(np.float64)
    min_cells = min_distance_m / model.resolution_m
    distances = np.linalg.norm(candidates.astype(np.float64) - fixed_yx, axis=1)
    candidates = candidates[distances >= min_cells]
    if candidates.shape[0] == 0:
        raise ValueError("No endpoint in the fixed endpoint's component meets minimum distance")
    clearances = model.planning_distance_m[candidates[:, 0], candidates[:, 1]]
    top = candidates[np.argsort(clearances)[-min(5000, candidates.shape[0]) :]]
    rng = np.random.default_rng(seed)
    chosen = top[int(rng.integers(0, top.shape[0]))]
    return np.array([chosen[1], chosen[0]], dtype=np.int64)


def resolve_endpoints(
    scene: SceneArtifact,
    request: TrajectoryRequest,
    *,
    min_distance_m: float,
    seed: int,
    valid_sampling_cells: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model = scene.model
    free = ~model.planning_blocked if valid_sampling_cells is None else valid_sampling_cells
    if free.shape != model.planning_blocked.shape:
        raise ValueError("valid sampling cells shape does not match the scene map")
    start = (
        nearest_free_cell(world_to_grid(request.start_xz, model.spec), free)
        if request.start_xz is not None
        else None
    )
    goal = (
        nearest_free_cell(world_to_grid(request.goal_xz, model.spec), free)
        if request.goal_xz is not None
        else None
    )
    if start is None and goal is None:
        return choose_auto_start_goal(
            free,
            model.planning_distance_m,
            model.spec,
            min_distance_m=min_distance_m,
            seed=seed,
        )
    if start is None:
        assert goal is not None
        start = _auto_in_component(
            scene,
            goal,
            min_distance_m=min_distance_m,
            seed=seed,
            free=free,
        )
    if goal is None:
        goal = _auto_in_component(
            scene,
            start,
            min_distance_m=min_distance_m,
            seed=seed,
            free=free,
        )
    return start, goal
