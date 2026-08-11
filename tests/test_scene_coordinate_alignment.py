from __future__ import annotations

import numpy as np

from robotnav.config import load_render_config
from robotnav.coordinates import project_world_points
from robotnav.dataset.trajectory_to_camera import (
    go2_t_base_from_camera,
    go2_t_world_from_base_link,
    renderer_t_camera_from_world,
)
from robotnav.navigation.scene.las_io import transform_points_xyz


def test_hall_las_gaussian_and_go2_camera_share_one_world_projection() -> None:
    """A known LAS obstacle point reaches the exact RGB camera-center pixel."""
    # This raw LAS point becomes a 2 m forward obstacle after [X,Y,Z] -> [X,-Z,Y].
    p_las_raw = np.array([[2.30, 0.0, 0.65]], dtype=np.float64)
    p_world_robot = transform_points_xyz(p_las_raw, "zup-to-yup")
    p_world_scene = p_world_robot.copy()  # 3DGS means are loaded at unit scale, unchanged.

    t_world_base_link = go2_t_world_from_base_link(
        point_xz=np.array([0.0, 0.0]),
        tangent_xz=np.array([1.0, 0.0]),
        floor_y=0.0,
        base_height_above_floor_m=0.53,
    )
    t_base_from_camera = go2_t_base_from_camera((0.30, 0.0, 0.12), (0.0, 0.0, 0.0))
    t_camera_world = renderer_t_camera_from_world(t_world_base_link @ t_base_from_camera)
    render = load_render_config("render_hall.toml")
    k_color = np.array(
        [
            [render.camera.fx, 0.0, render.camera.cx],
            [0.0, render.camera.fy, render.camera.cy],
            [0.0, 0.0, 1.0],
        ]
    )

    las_camera, las_pixel, las_valid = project_world_points(p_world_robot, t_camera_world, k_color)
    gaussian_camera, gaussian_pixel, gaussian_valid = project_world_points(
        p_world_scene, t_camera_world, k_color
    )

    assert las_valid.tolist() == [True]
    assert gaussian_valid.tolist() == [True]
    np.testing.assert_allclose(las_camera, gaussian_camera)
    np.testing.assert_allclose(las_pixel, gaussian_pixel)
    np.testing.assert_allclose(las_camera[0], [0.0, 0.0, 2.0], atol=1e-8)
    np.testing.assert_allclose(las_pixel[0], [render.camera.cx, render.camera.cy], atol=1e-8)


def test_render_config_uses_explicit_hall_assets() -> None:
    render = load_render_config("render_hall.toml")
    assert render.paths.las_filename == "hall.las"
    assert render.paths.ply_filename == "hall3DGS_yup.ply"
    assert render.paths.las_path.is_file()
    assert render.paths.ply_path is not None and render.paths.ply_path.is_file()


def test_scene_render_configs_are_isolated() -> None:
    hall = load_render_config("render_hall.toml")
    local = load_render_config("render_forth_local.toml")
    global_scene = load_render_config("render_forth_global.toml")

    assert (hall.paths.las_filename, hall.paths.ply_filename) == ("hall.las", "hall3DGS_yup.ply")
    assert (local.paths.las_filename, local.paths.ply_filename) == (
        "forth_local.las",
        "forth_local3DGS_yup_filtered.ply",
    )
    assert (global_scene.paths.las_filename, global_scene.paths.ply_filename) == (
        "forth_global.las",
        "forth_global3DGS_yup.ply",
    )
    assert len({hall.paths.output_dir, local.paths.output_dir, global_scene.paths.output_dir}) == 3
