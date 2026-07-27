"""Configuration owned by the LAS-to-navigation-scene stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from robotnav.config import CONFIG_DIR, PROJECT_ROOT, ConfigurationError, load_toml


@dataclass(frozen=True)
class ScenePaths:
    data_dir: Path
    las_filename: str
    output_dir: Path

    @property
    def las_path(self) -> Path:
        return self.data_dir / self.las_filename


@dataclass(frozen=True)
class SceneConfig:
    axis_transform: str
    floor_y_override: float | None
    roi_center_xz: tuple[float, float] | None
    roi_size_xz: tuple[float, float] | None
    floor_search_y_min: float
    floor_search_y_max: float
    resolution_m: float
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


@dataclass(frozen=True)
class RobotConfig:
    radius_m: float
    height_m: float
    ground_margin_m: float
    safety_margin_m: float


@dataclass(frozen=True)
class SceneBuildConfig:
    paths: ScenePaths
    scene: SceneConfig
    robot: RobotConfig

    @property
    def output_dir(self) -> Path:
        return self.paths.output_dir

    def manifest_config(self) -> dict[str, Any]:
        raw = {"scene": asdict(self.scene), "robot": asdict(self.robot)}
        return raw


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
        details = []
        if missing:
            details.append("missing keys: " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown keys: " + ", ".join(sorted(unknown)))
        raise ConfigurationError(f"Invalid [{name}] ({'; '.join(details)})")
    return values


def _optional_pair(value: Any, field: str) -> tuple[float, float] | None:
    if value == []:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    ):
        raise ConfigurationError(f"{field} must be [] or two numbers")
    return (float(value[0]), float(value[1]))


def load_scene_build_config(filename: str = "navigation_scene.toml") -> SceneBuildConfig:
    config_path = CONFIG_DIR / filename
    if config_path.parent != CONFIG_DIR or config_path.suffix != ".toml":
        raise ConfigurationError(f"Expected a TOML file directly in {CONFIG_DIR}: {filename}")
    raw = load_toml(config_path)
    expected_sections = {"paths", "scene", "robot"}
    if raw.keys() != expected_sections:
        raise ConfigurationError(
            f"Invalid {filename}; expected sections: {', '.join(sorted(expected_sections))}"
        )
    path_values = _table(raw, "paths", {"data_dir", "las_filename", "output_dir"})
    if (
        not isinstance(path_values["las_filename"], str)
        or Path(path_values["las_filename"]).name != path_values["las_filename"]
    ):
        raise ConfigurationError("[paths].las_filename must be one filename")
    scene_fields = set(SceneConfig.__dataclass_fields__)
    scene_values = _table(raw, "scene", scene_fields)
    robot_values = _table(raw, "robot", set(RobotConfig.__dataclass_fields__))
    floor_override = scene_values["floor_y_override"]
    if floor_override == "auto":
        floor_override = None
    elif not isinstance(floor_override, (int, float)) or isinstance(floor_override, bool):
        raise ConfigurationError("[scene].floor_y_override must be a number or auto")
    if scene_values["axis_transform"] not in {"none", "zup-to-yup"}:
        raise ConfigurationError("[scene].axis_transform must be none or zup-to-yup")
    positive_scene = (
        "resolution_m",
        "ground_band_m",
        "min_points_per_cell",
        "min_ground_points_per_cell",
        "chunk_size",
        "floor_sample_limit",
        "floor_hist_bins",
        "floor_xy_resolution_m",
    )
    if any(scene_values[key] <= 0 for key in positive_scene):
        raise ConfigurationError(f"[scene] values must be positive: {', '.join(positive_scene)}")
    if any(robot_values[key] <= 0 for key in ("radius_m", "height_m", "ground_margin_m")):
        raise ConfigurationError("[robot] radius_m, height_m, and ground_margin_m must be positive")
    if robot_values["safety_margin_m"] < 0:
        raise ConfigurationError("[robot].safety_margin_m must be non-negative")
    return SceneBuildConfig(
        paths=ScenePaths(
            data_dir=_path(path_values["data_dir"], "[paths].data_dir"),
            las_filename=path_values["las_filename"],
            output_dir=_path(path_values["output_dir"], "[paths].output_dir"),
        ),
        scene=SceneConfig(
            **cast(
                Any,
                {
                    **scene_values,
                    "floor_y_override": None if floor_override is None else float(floor_override),
                    "roi_center_xz": _optional_pair(
                        scene_values["roi_center_xz"], "[scene].roi_center_xz"
                    ),
                    "roi_size_xz": _optional_pair(
                        scene_values["roi_size_xz"], "[scene].roi_size_xz"
                    ),
                },
            )
        ),
        robot=RobotConfig(**robot_values),
    )
