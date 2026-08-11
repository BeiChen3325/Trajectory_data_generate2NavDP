"""Validated TOML configuration shared by RobotNav commands."""

from __future__ import annotations

import json
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
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    distortion_coeffs: tuple[float, ...]
    calibration_path: Path
    stream: str
    up_axis: str


@dataclass(frozen=True)
class RenderRuntimeConfig:
    background: str
    chunk_size: int
    seed: int


@dataclass(frozen=True)
class RenderPolicyConfig:
    require_cuda: bool = True


@dataclass(frozen=True)
class RenderConfig:
    paths: PathConfig
    camera: CameraConfig
    runtime: RenderRuntimeConfig
    render: RenderPolicyConfig


def load_render_config(filename: str = "render.toml") -> RenderConfig:
    """Load all render defaults from one validated TOML document."""
    raw = load_toml(CONFIG_DIR / filename)
    allowed_sections = {"paths", "camera", "runtime", "render"}
    missing_sections = {"paths", "camera", "runtime"} - raw.keys()
    unknown_sections = raw.keys() - allowed_sections
    if missing_sections or unknown_sections:
        details = []
        if missing_sections:
            details.append("missing sections: " + ", ".join(sorted(missing_sections)))
        if unknown_sections:
            details.append("unknown sections: " + ", ".join(sorted(unknown_sections)))
        raise ConfigurationError(f"Invalid {filename} ({'; '.join(details)})")
    paths = load_path_config(filename)
    camera_values = raw["camera"]
    if not isinstance(camera_values, dict) or set(camera_values) != {
        "calibration_file",
        "stream",
        "up_axis",
    }:
        raise ConfigurationError("[camera] must contain calibration_file, stream, and up_axis")
    calibration_file = camera_values["calibration_file"]
    stream_name = camera_values["stream"]
    if not isinstance(calibration_file, str) or not calibration_file:
        raise ConfigurationError("[camera].calibration_file must be a non-empty path")
    if not isinstance(stream_name, str) or not stream_name:
        raise ConfigurationError("[camera].stream must be a non-empty stream name")
    calibration_path = _resolve_path(calibration_file)
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(
            f"Cannot read camera calibration {calibration_path}: {error}"
        ) from error
    stream = calibration.get(stream_name)
    intrinsics = stream.get("intrinsics") if isinstance(stream, dict) else None
    if not isinstance(intrinsics, dict):
        raise ConfigurationError(
            f"Camera calibration {calibration_path} has no {stream_name}.intrinsics table"
        )
    required_intrinsics = {
        "width",
        "height",
        "fx",
        "fy",
        "cx_ppx",
        "cy_ppy",
        "distortion_model",
        "coeffs",
    }
    if set(intrinsics) != required_intrinsics:
        raise ConfigurationError(
            f"Camera calibration {calibration_path} has invalid {stream_name}.intrinsics keys"
        )
    width, height = intrinsics["width"], intrinsics["height"]
    fx, fy, cx, cy = (
        intrinsics["fx"],
        intrinsics["fy"],
        intrinsics["cx_ppx"],
        intrinsics["cy_ppy"],
    )
    coefficients = intrinsics["coeffs"]
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or any(not isinstance(value, (int, float)) for value in (fx, fy, cx, cy))
        or not isinstance(intrinsics["distortion_model"], str)
        or not isinstance(coefficients, list)
        or not all(isinstance(value, (int, float)) for value in coefficients)
    ):
        raise ConfigurationError(
            f"Camera calibration {calibration_path} has invalid intrinsic values"
        )
    camera = CameraConfig(
        width=width,
        height=height,
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        distortion_model=intrinsics["distortion_model"],
        distortion_coeffs=tuple(float(value) for value in coefficients),
        calibration_path=calibration_path,
        stream=stream_name,
        up_axis=camera_values["up_axis"],
    )
    runtime = RenderRuntimeConfig(**load_dataclass_section(raw, "runtime", RenderRuntimeConfig))
    render = (
        RenderPolicyConfig(**load_dataclass_section(raw, "render", RenderPolicyConfig))
        if "render" in raw
        else RenderPolicyConfig()
    )
    if camera.width <= 0 or camera.height <= 0 or camera.fx <= 0 or camera.fy <= 0:
        raise ConfigurationError("[camera] width, height, fx, and fy must be positive")
    if not (0 <= camera.cx <= camera.width and 0 <= camera.cy <= camera.height):
        raise ConfigurationError("[camera] principal point must be inside the image")
    if any(abs(value) > 1e-12 for value in camera.distortion_coeffs):
        raise ConfigurationError(
            "Renderer supports rectified pinhole intrinsics only; calibrate or rectify non-zero distortion"
        )
    if camera.up_axis not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
        raise ConfigurationError("[camera].up_axis must be a signed axis")
    if runtime.background not in {"black", "white"} or runtime.chunk_size <= 0:
        raise ConfigurationError("Invalid [runtime] background or chunk_size")
    if not isinstance(render.require_cuda, bool):
        raise ConfigurationError("[render].require_cuda must be true or false")
    return RenderConfig(paths=paths, camera=camera, runtime=runtime, render=render)


def ensure_output_dirs() -> None:
    """Create output directories declared by validated command configurations."""
    load_render_config().paths.output_dir.mkdir(parents=True, exist_ok=True)
    from robotnav.navigation.scene.config import load_scene_build_config
    from robotnav.navigation.semantic_pointcloud.config import load_pointcloud_export_config
    from robotnav.navigation.trajectory.config import load_trajectory_generation_config

    load_scene_build_config().paths.output_dir.mkdir(parents=True, exist_ok=True)
    load_trajectory_generation_config().paths.output_dir.mkdir(parents=True, exist_ok=True)
    load_pointcloud_export_config().paths.output_dir.mkdir(parents=True, exist_ok=True)
