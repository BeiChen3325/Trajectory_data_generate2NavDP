"""Configuration owned by trajectory planning and batch generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from robotnav.config import CONFIG_DIR, PROJECT_ROOT, ConfigurationError, load_toml

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class TrajectoryPaths:
    scene_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class PlannerConfig:
    obstacle_cost_weight: float
    obstacle_cost_power: float
    shortcut_passes: int
    smooth_samples_per_meter: float
    smooth_iterations: int


@dataclass(frozen=True)
class TrajectoryRequest:
    trajectory_id: str
    start_xz: tuple[float, float] | None
    goal_xz: tuple[float, float] | None


@dataclass(frozen=True)
class BatchConfig:
    count: int
    seed: int
    min_start_goal_distance_m: float
    min_endpoint_separation_m: float
    max_sampling_attempts: int
    manifest_filename: str
    requests: tuple[TrajectoryRequest, ...]


@dataclass(frozen=True)
class TrajectorySamplingConfig:
    """Optional length acceptance policy for automatically sampled trajectories."""

    trajectory_mode: str = "short"
    min_length_m: float = 0.0
    max_length_m: float = float("inf")


@dataclass(frozen=True)
class ValidRegionConfig:
    yaml_path: Path | None
    robot_radius_m: float
    safety_margin_m: float


@dataclass(frozen=True)
class TrajectoryGenerationConfig:
    paths: TrajectoryPaths
    planner: PlannerConfig
    batch: BatchConfig
    valid_region: ValidRegionConfig = ValidRegionConfig(None, 0.30, 0.10)
    trajectory_sampling: TrajectorySamplingConfig = TrajectorySamplingConfig()


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


def _requests(values: Any) -> tuple[TrajectoryRequest, ...]:
    if not isinstance(values, list):
        raise ConfigurationError("[trajectory_batch].requests must be an array of tables")
    result = []
    ids: set[str] = set()
    for index, item in enumerate(values):
        if not isinstance(item, dict) or item.keys() != {"id", "start_xz", "goal_xz"}:
            raise ConfigurationError(
                f"[trajectory_batch].requests[{index}] must contain id, start_xz, and goal_xz"
            )
        item = cast(dict[str, Any], item)
        trajectory_id = item["id"]
        if not isinstance(trajectory_id, str) or not SAFE_ID.fullmatch(trajectory_id):
            raise ConfigurationError(
                f"[trajectory_batch].requests[{index}].id is not filename-safe"
            )
        if trajectory_id in ids:
            raise ConfigurationError(f"Duplicate trajectory request id: {trajectory_id}")
        ids.add(trajectory_id)
        result.append(
            TrajectoryRequest(
                trajectory_id=trajectory_id,
                start_xz=_optional_pair(item["start_xz"], f"request {trajectory_id}.start_xz"),
                goal_xz=_optional_pair(item["goal_xz"], f"request {trajectory_id}.goal_xz"),
            )
        )
    return tuple(result)


def load_trajectory_generation_config(
    filename: str = "trajectories.toml",
) -> TrajectoryGenerationConfig:
    config_path = CONFIG_DIR / filename
    if config_path.parent != CONFIG_DIR or config_path.suffix != ".toml":
        raise ConfigurationError(f"Expected a TOML file directly in {CONFIG_DIR}: {filename}")
    raw = load_toml(config_path)
    expected_sections = {
        "paths",
        "planner",
        "trajectory_batch",
        "trajectory_sampling",
        "valid_region",
    }
    if raw.keys() - expected_sections:
        raise ConfigurationError(
            f"Invalid {filename}; expected only sections: {', '.join(sorted(expected_sections))}"
        )
    path_values = _table(raw, "paths", {"scene_dir", "output_dir"})
    planner_values = _table(raw, "planner", set(PlannerConfig.__dataclass_fields__))
    batch_values = _table(raw, "trajectory_batch", set(BatchConfig.__dataclass_fields__))
    sampling_values = raw.get("trajectory_sampling", {})
    if not isinstance(sampling_values, dict) or sampling_values.keys() - {
        "trajectory_mode",
        "min_length_m",
        "max_length_m",
    }:
        raise ConfigurationError("Invalid [trajectory_sampling]")
    trajectory_mode = sampling_values.get("trajectory_mode", "short")
    min_length_m = sampling_values.get("min_length_m", 0.0)
    max_length_m = sampling_values.get("max_length_m", float("inf"))
    if trajectory_mode not in {"short", "long"}:
        raise ConfigurationError("[trajectory_sampling].trajectory_mode must be short or long")
    for key, value in (("min_length_m", min_length_m), ("max_length_m", max_length_m)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ConfigurationError(f"[trajectory_sampling].{key} must be non-negative")
    if min_length_m > max_length_m:
        raise ConfigurationError("[trajectory_sampling].min_length_m must not exceed max_length_m")
    if trajectory_mode == "long" and min_length_m <= 0:
        raise ConfigurationError("[trajectory_sampling].min_length_m must be positive in long mode")
    region_values = raw.get("valid_region", {})
    if not isinstance(region_values, dict) or region_values.keys() - {
        "yaml_path",
        "robot_radius_m",
        "safety_margin_m",
    }:
        raise ConfigurationError("Invalid [valid_region]")
    yaml_value = region_values.get("yaml_path", "")
    if not isinstance(yaml_value, str):
        raise ConfigurationError("[valid_region].yaml_path must be a string")
    yaml_path = _path(yaml_value, "[valid_region].yaml_path") if yaml_value else None
    robot_radius_m = region_values.get("robot_radius_m", 0.30)
    safety_margin_m = region_values.get("safety_margin_m", 0.10)
    for key, value in (
        ("robot_radius_m", robot_radius_m),
        ("safety_margin_m", safety_margin_m),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ConfigurationError(f"[valid_region].{key} must be non-negative")
    requests = _requests(batch_values["requests"])
    count = batch_values["count"]
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ConfigurationError("[trajectory_batch].count must be a positive integer")
    if len(requests) > count:
        raise ConfigurationError("Trajectory request count cannot exceed trajectory_batch.count")
    if (
        not isinstance(batch_values["max_sampling_attempts"], int)
        or batch_values["max_sampling_attempts"] <= 0
    ):
        raise ConfigurationError(
            "[trajectory_batch].max_sampling_attempts must be a positive integer"
        )
    for key in ("min_start_goal_distance_m", "min_endpoint_separation_m"):
        if batch_values[key] < 0:
            raise ConfigurationError(f"[trajectory_batch].{key} must be non-negative")
    manifest_filename = batch_values["manifest_filename"]
    if (
        not isinstance(manifest_filename, str)
        or not manifest_filename
        or Path(manifest_filename).name != manifest_filename
    ):
        raise ConfigurationError("[trajectory_batch].manifest_filename must be one filename")
    if planner_values["smooth_samples_per_meter"] <= 0:
        raise ConfigurationError("[planner].smooth_samples_per_meter must be positive")
    if planner_values["shortcut_passes"] < 0 or planner_values["smooth_iterations"] < 0:
        raise ConfigurationError("[planner] pass and iteration counts must be non-negative")
    return TrajectoryGenerationConfig(
        paths=TrajectoryPaths(
            scene_dir=_path(path_values["scene_dir"], "[paths].scene_dir"),
            output_dir=_path(path_values["output_dir"], "[paths].output_dir"),
        ),
        planner=PlannerConfig(**planner_values),
        batch=BatchConfig(
            **cast(
                Any,
                {
                    **batch_values,
                    "requests": requests,
                },
            )
        ),
        valid_region=ValidRegionConfig(
            yaml_path=yaml_path,
            robot_radius_m=float(robot_radius_m),
            safety_margin_m=float(safety_margin_m),
        ),
        trajectory_sampling=TrajectorySamplingConfig(
            trajectory_mode=trajectory_mode,
            min_length_m=float(min_length_m),
            max_length_m=float(max_length_m),
        ),
    )
