"""Configuration for the file-based target dataset build stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robotnav.config import CONFIG_DIR, PROJECT_ROOT, ConfigurationError, load_toml


@dataclass(frozen=True)
class DatasetBuildPaths:
    trajectory_manifest: Path
    semantic_pointcloud_dir: Path
    semantic_pointcloud_filename: str
    semantic_pointcloud_report_filename: str
    work_dir: Path
    dataset_root: Path

    @property
    def semantic_pointcloud_path(self) -> Path:
        return self.semantic_pointcloud_dir / self.semantic_pointcloud_filename

    @property
    def semantic_pointcloud_report_path(self) -> Path:
        return self.semantic_pointcloud_dir / self.semantic_pointcloud_report_filename

    @property
    def episodes_dir(self) -> Path:
        return self.work_dir / "episodes"

    @property
    def batch_manifest_path(self) -> Path:
        return self.work_dir / "batch_manifest.json"


@dataclass(frozen=True)
class TrajectoryToCameraConfig:
    camera_pose_resource: str
    base_height_above_floor_m: float
    camera_frame: str
    base_from_camera_link_translation_m: tuple[float, float, float]
    base_from_camera_link_rpy_deg: tuple[float, float, float]


@dataclass(frozen=True)
class EpisodeRenderingConfig:
    camera_batch_size: int


@dataclass(frozen=True)
class DatasetOutputConfig:
    group_dir: str
    scene_dir: str
    overwrite: bool


@dataclass(frozen=True)
class DatasetBuildConfig:
    paths: DatasetBuildPaths
    trajectory_to_camera: TrajectoryToCameraConfig
    rendering: EpisodeRenderingConfig
    dataset: DatasetOutputConfig

    @property
    def scene_dir(self) -> Path:
        return self.paths.dataset_root / self.dataset.group_dir / self.dataset.scene_dir


def _project_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{field} must be a non-empty string")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _table(raw: dict[str, Any], name: str, keys: set[str]) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Missing or invalid [{name}] section")
    missing = keys - value.keys()
    unknown = value.keys() - keys
    if missing or unknown:
        details = []
        if missing:
            details.append("missing keys: " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown keys: " + ", ".join(sorted(unknown)))
        raise ConfigurationError(f"Invalid [{name}] ({'; '.join(details)})")
    return value


def _safe_component(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ConfigurationError(f"{field} must be a non-empty directory name")
    if Path(value).name != value:
        raise ConfigurationError(f"{field} must be one directory name, not a path")
    return value


def _triple(value: Any, field: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    ):
        raise ConfigurationError(f"{field} must contain exactly three numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def _load_camera_pose_resource(resource_ref: Any) -> TrajectoryToCameraConfig:
    """Resolve one Go2 camera-pose record from camera_resource.

    The TOML stores only this reference.  The camera height and external pose
    must not be duplicated in dataset_build.toml, where they could drift from
    the selected camera resource.
    """
    field = "[trajectory_to_camera].camera_pose_resource"
    if not isinstance(resource_ref, str) or not resource_ref:
        raise ConfigurationError(f"{field} must be a non-empty resource reference")
    resource_name, separator, fragment = resource_ref.partition("#")
    if not separator or not resource_name or not fragment:
        raise ConfigurationError(f"{field} must have the form path.yaml#mapping.path")
    resource_path = _project_path(resource_name, field)
    resource_root = PROJECT_ROOT / "src" / "camera_resource"
    if not resource_path.is_relative_to(resource_root):
        raise ConfigurationError(f"{field} must refer to a file below {resource_root}")
    if resource_path.suffix not in {".yaml", ".yml"}:
        raise ConfigurationError(f"{field} must refer to a YAML camera resource")
    try:
        with resource_path.open(encoding="utf-8") as handle:
            resource_data = yaml.safe_load(handle)
    except OSError as error:
        raise ConfigurationError(f"Cannot read Go2 camera resource {resource_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML in Go2 camera resource {resource_path}: {error}") from error

    record: Any = resource_data
    for component in fragment.split("."):
        if not component or not isinstance(record, dict) or component not in record:
            raise ConfigurationError(f"{field} cannot resolve #{fragment} in {resource_path}")
        record = record[component]
    if not isinstance(record, dict):
        raise ConfigurationError(f"{field} #{fragment} must resolve to a mapping")
    expected = {"camera_frame", "base_height_above_floor_m", "position_m", "rpy_deg"}
    if set(record) != expected:
        raise ConfigurationError(
            f"{field} #{fragment} must contain exactly: {', '.join(sorted(expected))}"
        )
    camera_frame = record["camera_frame"]
    if not isinstance(camera_frame, str) or not camera_frame:
        raise ConfigurationError(f"{field} #{fragment}.camera_frame must be a non-empty string")
    base_height = record["base_height_above_floor_m"]
    if not isinstance(base_height, (int, float)) or isinstance(base_height, bool) or base_height <= 0:
        raise ConfigurationError(f"{field} #{fragment}.base_height_above_floor_m must be positive")
    return TrajectoryToCameraConfig(
        camera_pose_resource=resource_ref,
        base_height_above_floor_m=float(base_height),
        camera_frame=camera_frame,
        base_from_camera_link_translation_m=_triple(
            record["position_m"], f"{field} #{fragment}.position_m"
        ),
        base_from_camera_link_rpy_deg=_triple(record["rpy_deg"], f"{field} #{fragment}.rpy_deg"),
    )


def load_dataset_build_config(filename: str = "dataset_build.toml") -> DatasetBuildConfig:
    """Load the configuration shared by all three file-based dataset stages."""
    config_path = CONFIG_DIR / filename
    if config_path.parent != CONFIG_DIR or config_path.suffix != ".toml":
        raise ConfigurationError(f"Expected a TOML file directly in {CONFIG_DIR}: {filename}")
    raw = load_toml(config_path)
    expected_sections = {"paths", "trajectory_to_camera", "rendering", "dataset"}
    missing_sections = expected_sections - raw.keys()
    unknown_sections = raw.keys() - expected_sections
    if missing_sections or unknown_sections:
        details = []
        if missing_sections:
            details.append("missing sections: " + ", ".join(sorted(missing_sections)))
        if unknown_sections:
            details.append("unknown sections: " + ", ".join(sorted(unknown_sections)))
        raise ConfigurationError(f"Invalid {filename} ({'; '.join(details)})")

    path_values = _table(
        raw,
        "paths",
        {
            "trajectory_manifest",
            "semantic_pointcloud_dir",
            "semantic_pointcloud_filename",
            "semantic_pointcloud_report_filename",
            "work_dir",
            "dataset_root",
        },
    )
    for filename_key in (
        "semantic_pointcloud_filename",
        "semantic_pointcloud_report_filename",
    ):
        value = path_values[filename_key]
        if not isinstance(value, str) or not value or Path(value).name != value:
            raise ConfigurationError(f"[paths].{filename_key} must be one filename")
    paths = DatasetBuildPaths(
        trajectory_manifest=_project_path(
            path_values["trajectory_manifest"], "[paths].trajectory_manifest"
        ),
        semantic_pointcloud_dir=_project_path(
            path_values["semantic_pointcloud_dir"], "[paths].semantic_pointcloud_dir"
        ),
        semantic_pointcloud_filename=path_values["semantic_pointcloud_filename"],
        semantic_pointcloud_report_filename=path_values["semantic_pointcloud_report_filename"],
        work_dir=_project_path(path_values["work_dir"], "[paths].work_dir"),
        dataset_root=_project_path(path_values["dataset_root"], "[paths].dataset_root"),
    )

    camera_values = _table(
        raw,
        "trajectory_to_camera",
        {"camera_pose_resource"},
    )
    trajectory_to_camera = _load_camera_pose_resource(camera_values["camera_pose_resource"])

    rendering_values = _table(raw, "rendering", {"camera_batch_size"})
    batch_size = rendering_values["camera_batch_size"]
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ConfigurationError("[rendering].camera_batch_size must be a positive integer")

    dataset_values = _table(raw, "dataset", {"group_dir", "scene_dir", "overwrite"})
    if not isinstance(dataset_values["overwrite"], bool):
        raise ConfigurationError("[dataset].overwrite must be true or false")

    return DatasetBuildConfig(
        paths=paths,
        trajectory_to_camera=trajectory_to_camera,
        rendering=EpisodeRenderingConfig(camera_batch_size=batch_size),
        dataset=DatasetOutputConfig(
            group_dir=_safe_component(dataset_values["group_dir"], "[dataset].group_dir"),
            scene_dir=_safe_component(dataset_values["scene_dir"], "[dataset].scene_dir"),
            overwrite=dataset_values["overwrite"],
        ),
    )
