"""Pure single-route planning over an already-built scene artifact."""

from __future__ import annotations

import numpy as np

from robotnav.navigation.scene.artifact import SceneArtifact
from robotnav.navigation.scene.occupancy_map import grid_to_world, world_to_grid
from robotnav.navigation.trajectory.astar import astar
from robotnav.navigation.trajectory.config import PlannerConfig
from robotnav.navigation.trajectory.contracts import PlannedTrajectory, ResolvedTrajectoryTask
from robotnav.navigation.trajectory.smoothing import (
    chaikin_smooth,
    densify_polyline,
    path_collides_world,
    shortcut_path,
)


def plan_trajectory(
    scene: SceneArtifact,
    task: ResolvedTrajectoryTask,
    config: PlannerConfig,
    valid_sampling_cells: np.ndarray | None = None,
) -> PlannedTrajectory:
    """Plan one route without reading LAS or writing any output."""
    model = scene.model
    path = astar(
        task.start_cell,
        task.goal_cell,
        model.planning_blocked,
        distance_m=model.planning_distance_m,
        resolution=model.resolution_m,
        obstacle_cost_weight=config.obstacle_cost_weight,
        obstacle_cost_power=config.obstacle_cost_power,
    )
    shortcut = shortcut_path(
        path,
        model.planning_blocked,
        passes=config.shortcut_passes,
        seed=task.seed,
    )
    astar_world = grid_to_world(path, model.spec)
    shortcut_world = grid_to_world(shortcut, model.spec)
    dense = densify_polyline(
        shortcut_world,
        samples_per_meter=config.smooth_samples_per_meter,
    )
    smooth = chaikin_smooth(dense, iterations=config.smooth_iterations)
    collides = path_collides_world(
        smooth,
        model.planning_blocked,
        lambda points: world_to_grid(points, model.spec),
    )
    if collides:
        smooth = dense
        collides = path_collides_world(
            smooth,
            model.planning_blocked,
            lambda points: world_to_grid(points, model.spec),
        )
    if collides:
        raise ValueError(f"Collision-free smoothing failed for trajectory {task.trajectory_id}")
    if valid_sampling_cells is not None:
        if valid_sampling_cells.shape != model.planning_blocked.shape:
            raise ValueError("valid sampling cells shape does not match the scene map")
        for name, cells in (("A*", path), ("shortcut", shortcut)):
            if not np.all(valid_sampling_cells[cells[:, 1], cells[:, 0]]):
                raise ValueError(f"{name} path leaves the valid sampling region")
        smooth_cells = world_to_grid(smooth, model.spec)
        height, width = valid_sampling_cells.shape
        in_bounds = (
            (smooth_cells[:, 0] >= 0)
            & (smooth_cells[:, 0] < width)
            & (smooth_cells[:, 1] >= 0)
            & (smooth_cells[:, 1] < height)
        )
        if not np.all(in_bounds) or not np.all(
            valid_sampling_cells[smooth_cells[in_bounds, 1], smooth_cells[in_bounds, 0]]
        ):
            raise ValueError("smoothed path leaves the valid sampling region")
    return PlannedTrajectory(
        task=task,
        astar_cells=path,
        shortcut_cells=shortcut,
        astar_xz=astar_world,
        shortcut_xz=shortcut_world,
        smooth_xz=np.asarray(smooth, dtype=np.float64),
        smooth_path_collides=False,
    )
