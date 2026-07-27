"""Configuration for the file-based target dataset build stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    height_above_floor_m: float
    base_extrinsic: tuple[float, ...]


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

    camera_values = _table(raw, "trajectory_to_camera", {"height_above_floor_m", "base_extrinsic"})
    height = camera_values["height_above_floor_m"]
    base_extrinsic = camera_values["base_extrinsic"]
    if not isinstance(height, (int, float)) or height <= 0:
        raise ConfigurationError("[trajectory_to_camera].height_above_floor_m must be positive")
    if (
        not isinstance(base_extrinsic, list)
        or len(base_extrinsic) != 16
        or not all(isinstance(value, (int, float)) for value in base_extrinsic)
    ):
        raise ConfigurationError("[trajectory_to_camera].base_extrinsic must contain 16 numbers")

    rendering_values = _table(raw, "rendering", {"camera_batch_size"})
    batch_size = rendering_values["camera_batch_size"]
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ConfigurationError("[rendering].camera_batch_size must be a positive integer")

    dataset_values = _table(raw, "dataset", {"group_dir", "scene_dir", "overwrite"})
    if not isinstance(dataset_values["overwrite"], bool):
        raise ConfigurationError("[dataset].overwrite must be true or false")

    return DatasetBuildConfig(
        paths=paths,
        trajectory_to_camera=TrajectoryToCameraConfig(
            height_above_floor_m=float(height),
            base_extrinsic=tuple(float(value) for value in base_extrinsic),
        ),
        rendering=EpisodeRenderingConfig(camera_batch_size=batch_size),
        dataset=DatasetOutputConfig(
            group_dir=_safe_component(dataset_values["group_dir"], "[dataset].group_dir"),
            scene_dir=_safe_component(dataset_values["scene_dir"], "[dataset].scene_dir"),
            overwrite=dataset_values["overwrite"],
        ),
    )
