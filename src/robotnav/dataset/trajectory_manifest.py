"""Validated multi-episode input contract for dataset construction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from robotnav.dataset.contracts import file_sha256

TRAJECTORY_MANIFEST_CONTRACT_VERSION = 1
SAFE_TRAJECTORY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class EpisodePaths:
    root: Path

    @property
    def camera_trajectory_path(self) -> Path:
        return self.root / "camera_trajectory.npz"

    @property
    def camera_manifest_path(self) -> Path:
        return self.root / "camera_trajectory.json"

    @property
    def rendered_episode_dir(self) -> Path:
        return self.root / "rendered_episode"

    @property
    def render_manifest_path(self) -> Path:
        return self.rendered_episode_dir / "render_manifest.json"


@dataclass(frozen=True)
class EpisodeSpec:
    episode_index: int
    episode_name: str
    trajectory_id: str
    trajectory_path: Path
    trajectory_sha256: str
    source_scene_model_sha256: str
    floor_y: float
    points_xz: np.ndarray
    coordinate_convention: str
    paths: EpisodePaths


@dataclass(frozen=True)
class TrajectoryBatch:
    manifest_path: Path
    manifest_sha256: str
    source_scene_model_sha256: str
    episodes: tuple[EpisodeSpec, ...]


def _safe_relative_file(root: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must stay below the trajectory manifest directory")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{field} escapes the trajectory manifest directory") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_route(
    entry: dict[str, Any],
    *,
    manifest_dir: Path,
    scene_sha256: str,
    episode_index: int,
    episode_digits: int,
    episodes_dir: Path,
) -> EpisodeSpec:
    required = {"trajectory_id", "path", "trajectory_sha256"}
    missing = required - entry.keys()
    if missing:
        raise ValueError(f"Trajectory manifest entry is missing fields: {sorted(missing)}")
    trajectory_id = entry["trajectory_id"]
    if not isinstance(trajectory_id, str) or not SAFE_TRAJECTORY_ID.fullmatch(trajectory_id):
        raise ValueError(f"Trajectory id is not directory-safe: {trajectory_id!r}")
    expected_sha256 = entry["trajectory_sha256"]
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(f"Invalid trajectory SHA-256 for {trajectory_id!r}")
    trajectory_path = _safe_relative_file(
        manifest_dir,
        entry["path"],
        field=f"trajectory {trajectory_id!r} path",
    )
    actual_sha256 = file_sha256(trajectory_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Trajectory SHA-256 mismatch for {trajectory_id!r}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    route = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if not isinstance(route, dict):
        raise ValueError(f"Trajectory file must contain a JSON object: {trajectory_path}")
    for field in (
        "trajectory_id",
        "source_scene_model_sha256",
        "floor_y",
        "smooth_path_xz",
        "coordinate_convention",
    ):
        if field not in route:
            raise ValueError(f"Trajectory {trajectory_id!r} is missing required field: {field}")
    if route["trajectory_id"] != trajectory_id:
        raise ValueError(f"Trajectory id mismatch in {trajectory_path}")
    if route["source_scene_model_sha256"] != scene_sha256:
        raise ValueError(f"Trajectory {trajectory_id!r} belongs to a different scene model")
    points_xz = np.asarray(route["smooth_path_xz"], dtype=np.float64)
    if points_xz.ndim != 2 or points_xz.shape[1:] != (2,) or points_xz.shape[0] < 2:
        raise ValueError(f"Trajectory {trajectory_id!r} smooth_path_xz must have shape (T,2), T>=2")
    floor_y = float(route["floor_y"])
    if not np.isfinite(points_xz).all() or not np.isfinite(floor_y):
        raise ValueError(f"Trajectory {trajectory_id!r} contains non-finite coordinates")
    coordinate_convention = route["coordinate_convention"]
    if not isinstance(coordinate_convention, str) or not coordinate_convention:
        raise ValueError(f"Trajectory {trajectory_id!r} has no coordinate convention")
    return EpisodeSpec(
        episode_index=episode_index,
        episode_name=f"{episode_index:0{episode_digits}d}",
        trajectory_id=trajectory_id,
        trajectory_path=trajectory_path,
        trajectory_sha256=actual_sha256,
        source_scene_model_sha256=scene_sha256,
        floor_y=floor_y,
        points_xz=points_xz,
        coordinate_convention=coordinate_convention,
        paths=EpisodePaths(episodes_dir / trajectory_id),
    )


def load_trajectory_batch(manifest_path: Path, episodes_dir: Path) -> TrajectoryBatch:
    """Load all routes in manifest order and validate their complete hash chain."""
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Trajectory manifest must contain a JSON object")
    if raw.get("contract_version") != TRAJECTORY_MANIFEST_CONTRACT_VERSION:
        raise ValueError(f"Unsupported trajectory manifest contract: {raw.get('contract_version')}")
    entries = raw.get("trajectories")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Trajectory manifest must contain at least one trajectory")
    if not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("Every trajectory manifest entry must be an object")
    entries = cast(list[dict[str, Any]], entries)
    count = len(entries)
    if raw.get("requested_count") != count or raw.get("trajectory_count") != count:
        raise ValueError("Trajectory manifest count fields do not match its entries")
    scene_sha256 = raw.get("source_scene_model_sha256")
    if not isinstance(scene_sha256, str) or not scene_sha256:
        raise ValueError("Trajectory manifest has no source_scene_model_sha256")
    digits = max(3, len(str(count - 1)))
    episodes = tuple(
        _load_route(
            entry,
            manifest_dir=manifest_path.parent,
            scene_sha256=scene_sha256,
            episode_index=index,
            episode_digits=digits,
            episodes_dir=episodes_dir,
        )
        for index, entry in enumerate(entries)
    )
    ids = [episode.trajectory_id for episode in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Trajectory manifest contains duplicate trajectory ids")
    return TrajectoryBatch(
        manifest_path=manifest_path,
        manifest_sha256=file_sha256(manifest_path),
        source_scene_model_sha256=scene_sha256,
        episodes=episodes,
    )
