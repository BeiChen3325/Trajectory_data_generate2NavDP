from __future__ import annotations

import numpy as np

from robotnav.navigation.scene.builder import resolve_scene_bounds
from robotnav.navigation.scene.config import SceneConfig
from robotnav.navigation.scene.occupancy_map import make_grid_spec


def _scene(*, enable_roi_crop: bool) -> SceneConfig:
    return SceneConfig(
        axis_transform="none",
        floor_y_override=None,
        enable_roi_crop=enable_roi_crop,
        roi_center_xz=(0.0, 0.0),
        roi_size_xz=(12.0, 12.0),
        floor_search_y_min=0.0,
        floor_search_y_max=3.0,
        resolution_m=0.08,
        ground_band_m=0.08,
        min_points_per_cell=2,
        min_ground_points_per_cell=2,
        open_kernel_cells=1,
        close_kernel_cells=2,
        min_obstacle_component_cells=8,
        ground_close_kernel_cells=2,
        chunk_size=100,
        max_stream_points=0,
        floor_sample_limit=100,
        floor_hist_bins=10,
        floor_xy_resolution_m=0.25,
    )


def test_global_scene_bounds_ignore_configured_debug_roi() -> None:
    header = {
        "min_xyz": (-30.0, -2.0, -40.0),
        "max_xyz": (30.0, 4.0, 40.0),
    }
    global_min, global_max, source = resolve_scene_bounds(header, _scene(enable_roi_crop=False))
    local_min, local_max, local_source = resolve_scene_bounds(header, _scene(enable_roi_crop=True))

    assert source == "las_header"
    assert local_source == "roi_crop"
    np.testing.assert_allclose(global_min, [-30.5, -40.5])
    np.testing.assert_allclose(global_max, [30.5, 40.5])
    global_grid = make_grid_spec(global_min, global_max, 0.08)
    local_grid = make_grid_spec(local_min, local_max, 0.08)
    assert global_grid["width"] > local_grid["width"]
    assert global_grid["height"] > local_grid["height"]
