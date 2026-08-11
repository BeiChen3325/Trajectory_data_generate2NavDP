from __future__ import annotations

import numpy as np

from robotnav.tools.convert_to_navdp_dataset import (
    GROUND_FROM_CAMERA_TO_NAVDP,
    _camera_forward_trajectory_dots,
    convert_camera_extrinsic,
)
from robotnav.utils.coordinate_transform import (
    NAVDP_FROM_ROBOTNAV,
    transform_pointcloud,
    transform_pose,
)


def test_identity_pose_becomes_the_declared_coordinate_transform() -> None:
    np.testing.assert_allclose(transform_pose(np.eye(4)), NAVDP_FROM_ROBOTNAV)


def test_single_axis_points_use_homogeneous_matrix_multiplication() -> None:
    raw = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    np.testing.assert_allclose(transform_pointcloud(raw), expected)


def test_camera_pose_left_multiplies_world_coordinates_only() -> None:
    t_world_camera = np.eye(4)
    t_world_camera[:3, 3] = [2.0, 3.0, 4.0]
    converted = transform_pose(t_world_camera)
    np.testing.assert_allclose(converted, NAVDP_FROM_ROBOTNAV @ t_world_camera)
    np.testing.assert_allclose(converted[:3, 3], [2.0, -4.0, 3.0])


def test_camera_extrinsic_uses_declared_ground_base_camera_chain() -> None:
    t_world_ground = np.eye(4)[None, ...]
    t_world_base = np.eye(4)[None, ...]
    t_world_base[0, :3, 3] = [0.0, 0.0, 0.53]
    t_base_from_camera = np.eye(4)
    t_base_from_camera[:3, 3] = [0.30, 0.0, 0.12]

    extrinsic, stats = convert_camera_extrinsic(t_world_ground, t_world_base, t_base_from_camera)

    expected_ground_from_camera = t_world_base[0] @ t_base_from_camera
    np.testing.assert_allclose(
        extrinsic[0], GROUND_FROM_CAMERA_TO_NAVDP @ expected_ground_from_camera
    )
    np.testing.assert_allclose(extrinsic[0, :3, 3], [0.0, 0.30, 0.65])
    assert stats.ground_base_z_max_abs == 0.0


def test_camera_forward_validation_uses_robot_motion_not_offset_camera_motion() -> None:
    """A turning camera offset must not replace the robot trajectory direction."""
    action = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
    robot_ground_pose = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
    action[:, :3, :3] = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    # Optical +Z is +X. The robot moves +X, but an offset camera center sweeps
    # +Y as the robot turns, which the validation must deliberately ignore.
    action[0, :3, 3] = [0.30, 0.0, 0.0]
    action[1, :3, 3] = [0.30, 1.0, 0.0]
    robot_ground_pose[1, :3, 3] = [1.0, 0.0, 0.0]

    dots = _camera_forward_trajectory_dots(action, robot_ground_pose)

    np.testing.assert_allclose(dots, [1.0])
