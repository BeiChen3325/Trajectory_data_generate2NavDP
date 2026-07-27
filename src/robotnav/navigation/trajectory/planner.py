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
    return PlannedTrajectory(
        task=task,
        astar_cells=path,
        shortcut_cells=shortcut,
        astar_xz=astar_world,
        shortcut_xz=shortcut_world,
        smooth_xz=np.asarray(smooth, dtype=np.float64),
        smooth_path_collides=False,
    )
