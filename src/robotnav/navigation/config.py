"""Typed navigation configuration loaded from ``configs/trajectory.toml``."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

from robotnav.config import ConfigurationError, load_command_toml, load_path_config


@dataclass(frozen=True)
class PointCloudConfig:
    enabled: bool
    filename: str
    report_filename: str
    obstacle_color_rgb: tuple[int, int, int]
    context_color_rgb: tuple[int, int, int]
    obstacle_voxel_size_m: float
    context_voxel_size_m: float
    include_context: bool


@dataclass(frozen=True)
class MapConfig:
    las_path: Path
    output_dir: Path
    axis_transform: str
    floor_y_override: float | None
    roi_center_xz: tuple[float, float] | None
    roi_size_xz: tuple[float, float] | None
    floor_search_y_min: float
    floor_search_y_max: float
    resolution_m: float
    robot_radius_m: float
    robot_height_m: float
    ground_margin_m: float
    safety_margin_m: float
    ground_band_m: float
    min_points_per_cell: int
    min_ground_points_per_cell: int
    open_kernel_cells: int
    close_kernel_cells: int
    min_obstacle_component_cells: int
    ground_close_kernel_cells: int
    chunk_size: int
    max_stream_points: int
    floor_sample_limit: int
    floor_hist_bins: int
    floor_xy_resolution_m: float
    start_xz: tuple[float, float] | None
    goal_xz: tuple[float, float] | None
    min_start_goal_distance_m: float
    obstacle_cost_weight: float
    obstacle_cost_power: float
    random_seed: int
    shortcut_passes: int
    smooth_samples_per_meter: float
    pointcloud: PointCloudConfig


def _optional_pair(value: Any, name: str) -> tuple[float, float] | None:
    if value == []:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigurationError(f"[navigation].{name} must be [] or an array of two numbers")
    if not all(isinstance(item, (int, float)) for item in value):
        raise ConfigurationError(f"[navigation].{name} must contain only numbers")
    return (float(value[0]), float(value[1]))


def _rgb_triplet(value: Any, name: str) -> tuple[int, int, int]:
    valid = (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        and all(0 <= item <= 255 for item in value)
    )
    if not valid:
        raise ConfigurationError(f"[pointcloud].{name} must contain three integers in [0,255]")
    return cast(tuple[int, int, int], tuple(value))


def load_map_config(filename: str = "trajectory.toml") -> MapConfig:
    """Load the complete trajectory configuration without flattening TOML sections."""
    raw = load_command_toml(filename, sections={"paths", "navigation", "pointcloud"})
    values = raw["navigation"]
    if not isinstance(values, dict):
        raise ConfigurationError("[navigation] must be a TOML table")
    expected = {field.name for field in fields(MapConfig)} - {
        "las_path",
        "output_dir",
        "pointcloud",
    }
    missing = expected - values.keys()
    unknown = values.keys() - expected
    if missing or unknown:
        details = []
        if missing:
            details.append("missing keys: " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown keys: " + ", ".join(sorted(unknown)))
        raise ConfigurationError(f"Invalid [navigation] ({'; '.join(details)})")
    if values["floor_y_override"] == "auto":
        values = {**values, "floor_y_override": None}
    elif not isinstance(values["floor_y_override"], (int, float)):
        raise ConfigurationError("[navigation].floor_y_override must be a number or auto")
    if values["axis_transform"] not in {"zup-to-yup", "none"}:
        raise ConfigurationError("[navigation].axis_transform must be zup-to-yup or none")
    if cast(float, values["resolution_m"]) <= 0 or cast(int, values["chunk_size"]) <= 0:
        raise ConfigurationError("[navigation].resolution_m and chunk_size must be positive")

    pointcloud_values = raw["pointcloud"]
    if not isinstance(pointcloud_values, dict):
        raise ConfigurationError("[pointcloud] must be a TOML table")
    pointcloud_expected = {field.name for field in fields(PointCloudConfig)}
    missing = pointcloud_expected - pointcloud_values.keys()
    unknown = pointcloud_values.keys() - pointcloud_expected
    if missing or unknown:
        details = []
        if missing:
            details.append("missing keys: " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown keys: " + ", ".join(sorted(unknown)))
        raise ConfigurationError(f"Invalid [pointcloud] ({'; '.join(details)})")
    for key in ("enabled", "include_context"):
        if not isinstance(pointcloud_values[key], bool):
            raise ConfigurationError(f"[pointcloud].{key} must be a boolean")
    for key in ("filename", "report_filename"):
        value = pointcloud_values[key]
        if not isinstance(value, str) or not value or Path(value).name != value:
            raise ConfigurationError(f"[pointcloud].{key} must be a plain filename")
    for key in ("obstacle_voxel_size_m", "context_voxel_size_m"):
        value = pointcloud_values[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ConfigurationError(f"[pointcloud].{key} must be positive")
    pointcloud = PointCloudConfig(
        enabled=pointcloud_values["enabled"],
        filename=pointcloud_values["filename"],
        report_filename=pointcloud_values["report_filename"],
        obstacle_color_rgb=_rgb_triplet(
            pointcloud_values["obstacle_color_rgb"], "obstacle_color_rgb"
        ),
        context_color_rgb=_rgb_triplet(pointcloud_values["context_color_rgb"], "context_color_rgb"),
        obstacle_voxel_size_m=float(pointcloud_values["obstacle_voxel_size_m"]),
        context_voxel_size_m=float(pointcloud_values["context_voxel_size_m"]),
        include_context=pointcloud_values["include_context"],
    )
    paths = load_path_config(filename)
    return MapConfig(
        las_path=paths.las_path,
        output_dir=paths.output_dir,
        pointcloud=pointcloud,
        **cast(
            Any,
            {
                **values,
                "roi_center_xz": _optional_pair(values["roi_center_xz"], "roi_center_xz"),
                "roi_size_xz": _optional_pair(values["roi_size_xz"], "roi_size_xz"),
                "start_xz": _optional_pair(values["start_xz"], "start_xz"),
                "goal_xz": _optional_pair(values["goal_xz"], "goal_xz"),
            },
        ),
    )
