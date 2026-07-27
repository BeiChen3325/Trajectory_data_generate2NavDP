"""Validated TOML configuration shared by RobotNav commands."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class ConfigurationError(ValueError):
    """Raised when a configuration file does not match its schema."""


def load_toml(path: Path) -> dict[str, Any]:
    """Read one TOML document."""
    if not path.is_file():
        raise FileNotFoundError(path)
    import tomli as tomllib

    with path.open("rb") as file:
        return tomllib.load(file)


def load_command_toml(filename: str, *, sections: set[str]) -> dict[str, Any]:
    """Load one command TOML and reject misspelled sections early."""
    path = CONFIG_DIR / filename
    if path.parent != CONFIG_DIR or path.suffix != ".toml":
        raise ConfigurationError(f"Expected a TOML file directly in {CONFIG_DIR}: {filename}")
    raw = load_toml(path)
    missing = sections - raw.keys()
    unknown = raw.keys() - sections
    if missing or unknown:
        details = []
        if missing:
            details.append("missing sections: " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown sections: " + ", ".join(sorted(unknown)))
        raise ConfigurationError(f"Invalid {filename} ({'; '.join(details)})")
    return raw


def load_dataclass_section(raw: dict[str, Any], section: str, schema: type[Any]) -> dict[str, Any]:
    """Validate TOML keys against a dataclass while retaining its defaults."""
    values = raw[section]
    if not isinstance(values, dict):
        raise ConfigurationError(f"[{section}] must be a TOML table")
    declared = {field.name for field in fields(schema)}
    required = {
        field.name
        for field in fields(schema)
        if field.default is MISSING and field.default_factory is MISSING
    }
    missing = required - values.keys()
    unknown = values.keys() - declared
    if missing or unknown:
        details = []
        if missing:
            details.append("missing keys: " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown keys: " + ", ".join(sorted(unknown)))
        raise ConfigurationError(f"Invalid [{section}] ({'; '.join(details)})")
    return values


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class PathConfig:
    data_dir: Path
    output_dir: Path
    las_filename: str
    ply_filename: str | None = None

    @property
    def las_path(self) -> Path:
        return self.data_dir / self.las_filename

    @property
    def ply_path(self) -> Path | None:
        return self.data_dir / self.ply_filename if self.ply_filename else None


def load_path_config(filename: str) -> PathConfig:
    """Load and validate the [paths] table shared by command configurations."""
    raw = load_toml(CONFIG_DIR / filename)
    paths = raw.get("paths")
    if not isinstance(paths, dict):
        raise ConfigurationError(f"Missing [paths] section in {filename}")
    required = {"data_dir", "output_dir", "las_filename"}
    missing = required - paths.keys()
    allowed = required | {"ply_filename"}
    unknown = paths.keys() - allowed
    if missing or unknown:
        details = []
        if missing:
            details.append("missing keys: " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown keys: " + ", ".join(sorted(unknown)))
        raise ConfigurationError(f"Invalid [paths] in {filename} ({'; '.join(details)})")
    if not all(isinstance(paths[key], str) for key in paths):
        raise ConfigurationError(f"All [paths] values in {filename} must be strings")
    return PathConfig(
        data_dir=_resolve_path(paths["data_dir"]),
        output_dir=_resolve_path(paths["output_dir"]),
        las_filename=paths["las_filename"],
        ply_filename=paths.get("ply_filename"),
    )


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    fov: float
    up_axis: str


@dataclass(frozen=True)
class RenderRuntimeConfig:
    background: str
    chunk_size: int
    seed: int


@dataclass(frozen=True)
class RenderConfig:
    paths: PathConfig
    camera: CameraConfig
    runtime: RenderRuntimeConfig


def load_render_config(filename: str = "render.toml") -> RenderConfig:
    """Load all render defaults from one validated TOML document."""
    raw = load_command_toml(filename, sections={"paths", "camera", "runtime"})
    paths = load_path_config(filename)
    camera = CameraConfig(**load_dataclass_section(raw, "camera", CameraConfig))
    runtime = RenderRuntimeConfig(**load_dataclass_section(raw, "runtime", RenderRuntimeConfig))
    if camera.width <= 0 or camera.height <= 0 or camera.fov <= 0:
        raise ConfigurationError("[camera] width, height, and fov must be positive")
    if camera.up_axis not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
        raise ConfigurationError("[camera].up_axis must be a signed axis")
    if runtime.background not in {"black", "white"} or runtime.chunk_size <= 0:
        raise ConfigurationError("Invalid [runtime] background or chunk_size")
    return RenderConfig(paths=paths, camera=camera, runtime=runtime)


def ensure_output_dirs() -> None:
    """Create output directories declared by validated command configurations."""
    load_render_config().paths.output_dir.mkdir(parents=True, exist_ok=True)
    from robotnav.navigation.scene.config import load_scene_build_config
    from robotnav.navigation.semantic_pointcloud.config import load_pointcloud_export_config
    from robotnav.navigation.trajectory.config import load_trajectory_generation_config

    load_scene_build_config().paths.output_dir.mkdir(parents=True, exist_ok=True)
    load_trajectory_generation_config().paths.output_dir.mkdir(parents=True, exist_ok=True)
    load_pointcloud_export_config().paths.output_dir.mkdir(parents=True, exist_ok=True)
