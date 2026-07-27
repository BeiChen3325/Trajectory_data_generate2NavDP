"""Build a deterministic status manifest for per-episode dataset artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robotnav.dataset.contracts import CONTRACT_VERSION, file_sha256, load_camera_trajectory
from robotnav.dataset.trajectory_manifest import EpisodeSpec, TrajectoryBatch


def _camera_is_current(batch: TrajectoryBatch, episode: EpisodeSpec) -> bool:
    try:
        camera = load_camera_trajectory(
            episode.paths.camera_trajectory_path,
            episode.paths.camera_manifest_path,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return False
    expected = {
        "trajectory_id": episode.trajectory_id,
        "episode_index": episode.episode_index,
        "source_trajectory_sha256": episode.trajectory_sha256,
        "source_batch_manifest_sha256": batch.manifest_sha256,
        "source_scene_model_sha256": batch.source_scene_model_sha256,
    }
    return all(camera.metadata.get(field) == value for field, value in expected.items())


def _render_is_current(batch: TrajectoryBatch, episode: EpisodeSpec) -> bool:
    if not episode.paths.render_manifest_path.is_file():
        return False
    try:
        manifest = json.loads(episode.paths.render_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "contract_version": CONTRACT_VERSION,
        "trajectory_id": episode.trajectory_id,
        "episode_index": episode.episode_index,
        "source_trajectory_sha256": episode.trajectory_sha256,
        "source_batch_manifest_sha256": batch.manifest_sha256,
        "source_scene_model_sha256": batch.source_scene_model_sha256,
        "camera_trajectory_npz_sha256": file_sha256(episode.paths.camera_trajectory_path),
        "camera_trajectory_manifest_sha256": file_sha256(episode.paths.camera_manifest_path),
    }
    return all(manifest.get(field) == value for field, value in expected.items())


def write_batch_manifest(batch: TrajectoryBatch, path: Path) -> None:
    episodes: list[dict[str, Any]] = []
    for episode in batch.episodes:
        item: dict[str, Any] = {
            "episode_index": episode.episode_index,
            "episode_name": episode.episode_name,
            "trajectory_id": episode.trajectory_id,
            "trajectory": str(episode.trajectory_path),
            "trajectory_sha256": episode.trajectory_sha256,
            "frame_count": int(episode.points_xz.shape[0]),
            "status": "pending",
        }
        camera_current = _camera_is_current(batch, episode)
        if camera_current:
            item.update(
                {
                    "status": "camera",
                    "camera_trajectory": str(episode.paths.camera_trajectory_path),
                    "camera_trajectory_sha256": file_sha256(episode.paths.camera_trajectory_path),
                    "camera_manifest": str(episode.paths.camera_manifest_path),
                    "camera_manifest_sha256": file_sha256(episode.paths.camera_manifest_path),
                }
            )
        if camera_current and _render_is_current(batch, episode):
            item.update(
                {
                    "status": "rendered",
                    "render_manifest": str(episode.paths.render_manifest_path),
                    "render_manifest_sha256": file_sha256(episode.paths.render_manifest_path),
                }
            )
        episodes.append(item)
    payload = {
        "contract_version": CONTRACT_VERSION,
        "source_trajectory_manifest": str(batch.manifest_path),
        "source_trajectory_manifest_sha256": batch.manifest_sha256,
        "source_scene_model_sha256": batch.source_scene_model_sha256,
        "episode_count": len(batch.episodes),
        "episodes": episodes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
