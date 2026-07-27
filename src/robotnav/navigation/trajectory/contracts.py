"""In-memory contracts for resolved tasks and planned trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResolvedTrajectoryTask:
    trajectory_id: str
    start_cell: np.ndarray
    goal_cell: np.ndarray
    seed: int


@dataclass(frozen=True)
class PlannedTrajectory:
    task: ResolvedTrajectoryTask
    astar_cells: np.ndarray
    shortcut_cells: np.ndarray
    astar_xz: np.ndarray
    shortcut_xz: np.ndarray
    smooth_xz: np.ndarray
    smooth_path_collides: bool

    @property
    def path_length_m(self) -> float:
        if self.smooth_xz.shape[0] < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.smooth_xz, axis=0), axis=1).sum())
