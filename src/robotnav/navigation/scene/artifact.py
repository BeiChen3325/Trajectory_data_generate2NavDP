"""Integrity-checked scene artifact composed of an NPZ model and JSON manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robotnav.navigation.scene.contracts import SceneObstacleModel, load_scene_obstacle_model

SCENE_ARTIFACT_CONTRACT_VERSION = 1
SCENE_MODEL_FILENAME = "scene_model.npz"
SCENE_MANIFEST_FILENAME = "scene_manifest.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": file_sha256(resolved),
    }


@dataclass(frozen=True)
class SceneArtifact:
    scene_dir: Path
    model: SceneObstacleModel
    manifest: dict[str, Any]
    model_sha256: str

    @property
    def model_path(self) -> Path:
        return self.scene_dir / SCENE_MODEL_FILENAME

    @property
    def source_las(self) -> dict[str, Any]:
        return self.manifest["source_las"]

    @property
    def ground_margin_m(self) -> float:
        return float(self.manifest["config"]["robot"]["ground_margin_m"])


def load_scene_artifact(scene_dir: Path) -> SceneArtifact:
    scene_dir = Path(scene_dir)
    manifest_path = scene_dir / SCENE_MANIFEST_FILENAME
    model_path = scene_dir / SCENE_MODEL_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract_version") != SCENE_ARTIFACT_CONTRACT_VERSION:
        raise ValueError(f"Unsupported scene artifact contract: {manifest.get('contract_version')}")
    expected_hash = manifest.get("scene_model_sha256")
    actual_hash = file_sha256(model_path)
    if expected_hash != actual_hash:
        raise ValueError(
            f"Scene model hash mismatch: manifest={expected_hash!r}, actual={actual_hash!r}"
        )
    source = manifest.get("source_las")
    config = manifest.get("config")
    if not isinstance(source, dict) or not isinstance(config, dict):
        raise ValueError("Scene manifest is missing source_las or config")
    model = load_scene_obstacle_model(model_path)
    return SceneArtifact(
        scene_dir=scene_dir,
        model=model,
        manifest=manifest,
        model_sha256=actual_hash,
    )


def validate_source_las(artifact: SceneArtifact, las_path: Path) -> dict[str, Any]:
    actual = source_file_record(las_path)
    expected = artifact.source_las
    if actual["size_bytes"] != expected.get("size_bytes") or actual["sha256"] != expected.get(
        "sha256"
    ):
        raise ValueError(
            "LAS source does not match the scene artifact "
            f"(expected {expected.get('path')}, got {actual['path']})"
        )
    return actual
