"""Explicit shared world-to-camera projection utilities.

The navigation scene, Gaussian means, and Go2 trajectory all use the project
world frame. Keeping the projection calculation here makes that assumption
testable without relying on renderer-specific matrix conventions.
"""

from __future__ import annotations

import numpy as np


def project_world_points(
    points_world: np.ndarray,
    t_camera_world: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points using ``p_camera = T_camera_world @ p_world``.

    Returns camera coordinates, pixel coordinates ``(u, v)``, and a mask for
    points with positive camera-optical depth. This is the same pinhole
    convention supplied to gsplat: camera ``+X`` right, ``+Y`` down, ``+Z``
    forward.
    """
    points = np.asarray(points_world, dtype=np.float64)
    transform = np.asarray(t_camera_world, dtype=np.float64)
    k_matrix = np.asarray(intrinsic, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_world must be a finite (N,3) array")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("T_camera_world must be a finite (4,4) matrix")
    if k_matrix.shape != (3, 3) or not np.isfinite(k_matrix).all():
        raise ValueError("intrinsic must be a finite (3,3) matrix")

    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    camera = (transform @ homogeneous.T).T[:, :3]
    positive_depth = camera[:, 2] > 0.0
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    depth = camera[positive_depth, 2]
    pixels[positive_depth, 0] = (
        k_matrix[0, 0] * camera[positive_depth, 0] / depth + k_matrix[0, 2]
    )
    pixels[positive_depth, 1] = (
        k_matrix[1, 1] * camera[positive_depth, 1] / depth + k_matrix[1, 2]
    )
    return camera, pixels, positive_depth
