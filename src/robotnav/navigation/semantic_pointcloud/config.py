"""Configuration owned by the independent semantic PLY export stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from robotnav.config import CONFIG_DIR, PROJECT_ROOT, ConfigurationError, load_toml


@dataclass(frozen=True)
class PointCloudPaths:
    data_dir: Path
    las_filename: str
    scene_dir: Path
    output_dir: Path

    @property
    def las_path(self) -> Path:
        return self.data_dir / self.las_filename


@dataclass(frozen=True)
class PointCloudConfig:
    filename: str
    report_filename: str
    obstacle_color_rgb: tuple[int, int, int]
    context_color_rgb: tuple[int, int, int]
    obstacle_voxel_size_m: float
    context_voxel_size_m: float
    include_context: bool


@dataclass(frozen=True)
class PointCloudRuntimeConfig:
    chunk_size: int
    max_stream_points: int


@dataclass(frozen=True)
class PointCloudExportConfig:
    paths: PointCloudPaths
    pointcloud: PointCloudConfig
    runtime: PointCloudRuntimeConfig


def _path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{field} must be a non-empty string")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _table(raw: dict[str, Any], name: str, expected: set[str]) -> dict[str, Any]:
    values = raw.get(name)
    if not isinstance(values, dict):
        raise ConfigurationError(f"Missing or invalid [{name}] section")
    missing = expected - values.keys()
    unknown = values.keys() - expected
    if missing or unknown:
        raise ConfigurationError(
            f"Invalid [{name}] (missing={sorted(missing)}, unknown={sorted(unknown)})"
        )
    return values


def _rgb(value: Any, field: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(
            isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 255
            for item in value
        )
    ):
        raise ConfigurationError(f"{field} must contain three integers in [0,255]")
    return cast(tuple[int, int, int], tuple(value))


def load_pointcloud_export_config(
    filename: str = "pointcloud_export.toml",
) -> PointCloudExportConfig:
    config_path = CONFIG_DIR / filename
    if config_path.parent != CONFIG_DIR or config_path.suffix != ".toml":
        raise ConfigurationError(f"Expected a TOML file directly in {CONFIG_DIR}: {filename}")
    raw = load_toml(config_path)
    expected_sections = {"paths", "pointcloud", "runtime"}
    if raw.keys() != expected_sections:
        raise ConfigurationError(
            f"Invalid {filename}; expected sections: {', '.join(sorted(expected_sections))}"
        )
    path_values = _table(
        raw,
        "paths",
        {"data_dir", "las_filename", "scene_dir", "output_dir"},
    )
    point_values = _table(raw, "pointcloud", set(PointCloudConfig.__dataclass_fields__))
    runtime_values = _table(raw, "runtime", set(PointCloudRuntimeConfig.__dataclass_fields__))
    for key in ("las_filename",):
        if not isinstance(path_values[key], str) or Path(path_values[key]).name != path_values[key]:
            raise ConfigurationError(f"[paths].{key} must be one filename")
    for key in ("filename", "report_filename"):
        if (
            not isinstance(point_values[key], str)
            or not point_values[key]
            or Path(point_values[key]).name != point_values[key]
        ):
            raise ConfigurationError(f"[pointcloud].{key} must be one filename")
    for key in ("obstacle_voxel_size_m", "context_voxel_size_m"):
        if point_values[key] <= 0:
            raise ConfigurationError(f"[pointcloud].{key} must be positive")
    if not isinstance(point_values["include_context"], bool):
        raise ConfigurationError("[pointcloud].include_context must be boolean")
    if runtime_values["chunk_size"] <= 0 or runtime_values["max_stream_points"] < 0:
        raise ConfigurationError("[runtime] chunk_size must be positive and limit non-negative")
    return PointCloudExportConfig(
        paths=PointCloudPaths(
            data_dir=_path(path_values["data_dir"], "[paths].data_dir"),
            las_filename=path_values["las_filename"],
            scene_dir=_path(path_values["scene_dir"], "[paths].scene_dir"),
            output_dir=_path(path_values["output_dir"], "[paths].output_dir"),
        ),
        pointcloud=PointCloudConfig(
            **cast(
                Any,
                {
                    **point_values,
                    "obstacle_color_rgb": _rgb(
                        point_values["obstacle_color_rgb"], "[pointcloud].obstacle_color_rgb"
                    ),
                    "context_color_rgb": _rgb(
                        point_values["context_color_rgb"], "[pointcloud].context_color_rgb"
                    ),
                },
            )
        ),
        runtime=PointCloudRuntimeConfig(**runtime_values),
    )
