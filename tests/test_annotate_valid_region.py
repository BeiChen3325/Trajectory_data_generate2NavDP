from __future__ import annotations

import numpy as np

from robotnav.commands.annotate_valid_region import (
    VIEW_PADDING_PX,
    fit_camera,
    grid_to_screen,
    pan_camera,
    screen_to_grid,
    zoom_at,
)


def test_fit_camera_contains_large_map_and_centers_it() -> None:
    camera = fit_camera((2000, 2000), (1000, 800))
    assert camera.scale == (800 - 2 * VIEW_PADDING_PX) / 2000
    assert camera.offset_x == (1000 - 2000 * camera.scale) * 0.5
    assert camera.offset_y == VIEW_PADDING_PX


def test_viewport_transform_round_trip_and_mouse_anchored_zoom() -> None:
    camera = fit_camera((2000, 3000), (1000, 800))
    grid = np.array([1234.5, 678.25])
    screen = grid_to_screen(grid, camera)
    assert np.allclose(screen_to_grid(tuple(screen), camera), grid)

    anchor = (400, 300)
    before = screen_to_grid(anchor, camera)
    zoom_at(camera, anchor, 1.25)
    assert np.allclose(screen_to_grid(anchor, camera), before)


def test_directional_pan_updates_camera_offsets() -> None:
    camera = fit_camera((100, 100), (400, 400))
    original = (camera.offset_x, camera.offset_y)
    pan_camera(camera, "up", 30.0)
    pan_camera(camera, "left", 20.0)
    assert (camera.offset_x, camera.offset_y) == (original[0] + 20.0, original[1] + 30.0)
