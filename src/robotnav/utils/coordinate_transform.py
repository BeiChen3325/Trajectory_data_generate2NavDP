"""Homogeneous RobotNav-world to NavDP-world coordinate conversion.

The conversion is defined by ``NAVDP_FROM_ROBOTNAV`` and is intentionally
implemented as matrix multiplication, not by swapping NumPy array axes.
"""

from __future__ import annotations

import numpy as np

NAVDP_FROM_ROBOTNAV = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def transform_pose(pose_world: np.ndarray) -> np.ndarray:
    """Convert ``T_world_camera`` pose(s) with ``A @ T_world_camera``.

    ``pose_world`` may be one ``(4, 4)`` pose or a batch with trailing shape
    ``(4, 4)``.  The returned array is float64 so the caller chooses its final
    storage dtype explicitly.
    """
    pose = np.asarray(pose_world, dtype=np.float64)
    if pose.ndim < 2 or pose.shape[-2:] != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("pose_world must be a finite (4,4) matrix or batch of matrices")
    if not np.allclose(pose[..., 3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise ValueError("pose_world must use homogeneous [0,0,0,1] last rows")
    return NAVDP_FROM_ROBOTNAV @ pose


def transform_pointcloud(points_raw: np.ndarray) -> np.ndarray:
    """Convert ``(N,3)`` point positions using homogeneous ``A @ p_raw``."""
    points = np.asarray(points_raw, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_raw must be a finite (N,3) array")
    homogeneous = np.concatenate((points, np.ones((len(points), 1), dtype=np.float64)), axis=1)
    converted = (NAVDP_FROM_ROBOTNAV @ homogeneous.T).T
    if not np.allclose(converted[:, 3], 1.0, atol=1e-12):
        raise ValueError("coordinate conversion produced invalid homogeneous point coordinates")
    return converted[:, :3]
