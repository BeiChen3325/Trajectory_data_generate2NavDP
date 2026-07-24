"""Stage 3: package fixed camera/RGB-D artifacts into the target dataset contract."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from plyfile import PlyData

from robotnav.dataset.config import DatasetBuildConfig, load_dataset_build_config
from robotnav.dataset.contracts import (
    CONTRACT_VERSION,
    DEPTH_UNITS_PER_METER,
    INVALID_DEPTH_VALUE,
    file_sha256,
    load_camera_trajectory,
)

CHUNK_NAME = "chunk-000"
EPISODE_NAME = "000"
OBSTACLE_COLOR = np.array([0.0, 0.0, 0.5], dtype=np.float64)
OBSTACLE_COLOR_DISTANCE = 0.05


def _load_render_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "contract_version",
        "frame_count",
        "camera_intrinsic",
        "width",
        "height",
        "rgb_paths",
        "depth_paths",
        "depth_units_per_meter",
        "invalid_depth_value",
        "camera_trajectory_npz_sha256",
        "camera_trajectory_manifest_sha256",
    }
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"Render manifest is missing fields: {sorted(missing)}")
    if manifest["contract_version"] != CONTRACT_VERSION:
        raise ValueError(f"Unsupported render manifest contract: {manifest['contract_version']}")
    return manifest


def _artifact_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Artifact path must stay below render directory: {relative}")
    resolved = root / path
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _normalized_ply_colors(ply_path: Path) -> np.ndarray:
    if not ply_path.is_file():
        raise FileNotFoundError(ply_path)
    ply = PlyData.read(ply_path)
    if "vertex" not in ply:
        raise ValueError("pointcloud.ply must contain a vertex element")
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or ())
    required = {"x", "y", "z", "red", "green", "blue"}
    missing = required - names
    if missing:
        raise ValueError(f"pointcloud.ply is missing vertex properties: {sorted(missing)}")
    points = np.stack([vertex[axis] for axis in ("x", "y", "z")], axis=-1)
    colors = np.stack([vertex[channel] for channel in ("red", "green", "blue")], axis=-1)
    if points.shape[0] == 0 or not np.isfinite(points).all():
        raise ValueError("pointcloud.ply must contain finite points")
    if np.issubdtype(colors.dtype, np.integer):
        colors = colors.astype(np.float64) / float(np.iinfo(colors.dtype).max)
    else:
        colors = colors.astype(np.float64)
    if not np.isfinite(colors).all() or np.any(colors < 0.0) or np.any(colors > 1.0):
        raise ValueError("pointcloud.ply colors must map to the [0,1] range")
    return colors


def validate_semantic_pointcloud(ply_path: Path) -> None:
    colors = _normalized_ply_colors(ply_path)
    distances = np.linalg.norm(colors - OBSTACLE_COLOR[None, :], axis=1)
    if not np.any(distances < OBSTACLE_COLOR_DISTANCE):
        raise ValueError("pointcloud.ply contains no target obstacle-color points")


def _validate_source_images(
    render_dir: Path, manifest: dict[str, Any]
) -> tuple[list[Path], list[Path]]:
    count = int(manifest["frame_count"])
    rgb_values = manifest["rgb_paths"]
    depth_values = manifest["depth_paths"]
    if not isinstance(rgb_values, list) or not isinstance(depth_values, list):
        raise ValueError("rgb_paths and depth_paths must be lists")
    if len(rgb_values) != count or len(depth_values) != count:
        raise ValueError("Render manifest frame_count does not match RGB/Depth path counts")
    width = int(manifest["width"])
    height = int(manifest["height"])
    rgb_paths = [_artifact_path(render_dir, str(value)) for value in rgb_values]
    depth_paths = [_artifact_path(render_dir, str(value)) for value in depth_values]
    for rgb_path, depth_path in zip(rgb_paths, depth_paths, strict=True):
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or rgb.dtype != np.uint8 or rgb.shape != (height, width, 3):
            raise ValueError(f"Invalid RGB image: {rgb_path}")
        if depth is None or depth.dtype != np.uint16 or depth.shape != (height, width):
            raise ValueError(f"Invalid depth image: {depth_path}")
    return rgb_paths, depth_paths


def _write_episode_parquet(
    path: Path,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    base_extrinsic: np.ndarray,
) -> None:
    count = camera_to_world.shape[0]
    intrinsic_values = intrinsic.reshape(-1).tolist()
    extrinsic_values = base_extrinsic.reshape(-1).tolist()
    frame = pd.DataFrame(
        {
            "observation.camera_intrinsic": [intrinsic_values.copy() for _ in range(count)],
            "observation.camera_extrinsic": [extrinsic_values.copy() for _ in range(count)],
            "action": [matrix.reshape(-1).tolist() for matrix in camera_to_world],
        }
    )
    frame.to_parquet(path, engine="pyarrow", index=False)


def validate_target_scene(scene_dir: Path) -> dict[str, Any]:
    """Validate the final directory independently of the packaging write path."""
    data_dir = scene_dir / "data" / CHUNK_NAME
    rgb_dir = scene_dir / "videos" / CHUNK_NAME / "observation.images.rgb"
    depth_dir = scene_dir / "videos" / CHUNK_NAME / "observation.images.depth"
    stats_path = scene_dir / "meta" / "episodes_stats.jsonl"
    pointcloud_path = scene_dir / "meta" / "pointcloud.ply"
    parquet_paths = sorted(data_dir.glob("*.parquet"))
    rgb_paths = sorted(rgb_dir.glob("*.png"))
    depth_paths = sorted(depth_dir.glob("*.png"))
    if len(parquet_paths) != 1:
        raise ValueError("First-version target scene must contain exactly one parquet file")
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    stats_lines = [line for line in stats_path.read_text(encoding="utf-8").splitlines() if line]
    if len(stats_lines) != 1:
        raise ValueError("episodes_stats.jsonl must contain exactly one line")
    stats = json.loads(stats_lines[0])
    image_index = stats.get("image_index")
    if image_index != {"min": 0, "max": len(rgb_paths) - 1}:
        raise ValueError("episodes_stats.jsonl image range does not match RGB files")
    if not rgb_paths or len(rgb_paths) != len(depth_paths):
        raise ValueError("RGB and Depth file counts must be equal and non-zero")
    frame = pd.read_parquet(parquet_paths[0])
    required_columns = {
        "observation.camera_intrinsic",
        "observation.camera_extrinsic",
        "action",
    }
    if not required_columns.issubset(frame.columns):
        raise ValueError("Parquet is missing target columns")
    if len(frame) != len(rgb_paths):
        raise ValueError("Parquet row count does not match image count")
    intrinsic = np.asarray(frame["observation.camera_intrinsic"].tolist()[0]).reshape(3, 3)
    extrinsic = np.asarray(frame["observation.camera_extrinsic"].tolist()[0]).reshape(4, 4)
    actions = np.stack(frame["action"].to_numpy()).reshape(-1, 4, 4)
    if not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all():
        raise ValueError("Parquet camera matrices must be finite")
    if not np.isfinite(actions).all() or actions.shape[0] < len(rgb_paths):
        raise ValueError("Parquet actions are invalid or shorter than image sequence")
    validate_semantic_pointcloud(pointcloud_path)
    return {
        "frame_count": len(rgb_paths),
        "parquet": str(parquet_paths[0]),
        "rgb_count": len(rgb_paths),
        "depth_count": len(depth_paths),
        "pointcloud": str(pointcloud_path),
    }


def run_package_dataset(config: DatasetBuildConfig) -> Path:
    camera = load_camera_trajectory(
        config.paths.camera_trajectory_path, config.paths.camera_manifest_path
    )
    render_dir = config.paths.rendered_episode_dir
    render_manifest_path = render_dir / "render_manifest.json"
    render_manifest = _load_render_manifest(render_manifest_path)
    count = camera.frame_count
    if int(render_manifest["frame_count"]) != count:
        raise ValueError("Camera and render manifests have different frame counts")
    if render_manifest["depth_units_per_meter"] != DEPTH_UNITS_PER_METER:
        raise ValueError("Rendered depth unit does not match target_data.md")
    if render_manifest["invalid_depth_value"] != INVALID_DEPTH_VALUE:
        raise ValueError("Rendered invalid depth value does not match target_data.md")
    if render_manifest["camera_trajectory_npz_sha256"] != file_sha256(
        config.paths.camera_trajectory_path
    ):
        raise ValueError("Rendered images do not match the current camera trajectory NPZ")
    if render_manifest["camera_trajectory_manifest_sha256"] != file_sha256(
        config.paths.camera_manifest_path
    ):
        raise ValueError("Rendered images do not match the current camera trajectory manifest")
    intrinsic = np.asarray(render_manifest["camera_intrinsic"], dtype=np.float64)
    if intrinsic.size != 9 or not np.isfinite(intrinsic).all():
        raise ValueError("Render manifest camera_intrinsic must contain 9 finite values")
    intrinsic = intrinsic.reshape(3, 3)
    base_extrinsic = np.asarray(
        config.trajectory_to_camera.base_extrinsic, dtype=np.float64
    ).reshape(4, 4)
    if not np.isfinite(base_extrinsic).all():
        raise ValueError("Configured base_extrinsic must be finite")
    rgb_sources, depth_sources = _validate_source_images(render_dir, render_manifest)
    validate_semantic_pointcloud(config.paths.semantic_pointcloud_path)

    scene_dir = config.scene_dir
    scene_parent = scene_dir.parent
    scene_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{scene_dir.name}-staging-", dir=scene_parent))
    try:
        data_dir = staging / "data" / CHUNK_NAME
        rgb_dir = staging / "videos" / CHUNK_NAME / "observation.images.rgb"
        depth_dir = staging / "videos" / CHUNK_NAME / "observation.images.depth"
        meta_dir = staging / "meta"
        for directory in (data_dir, rgb_dir, depth_dir, meta_dir):
            directory.mkdir(parents=True, exist_ok=True)
        digits = max(3, len(str(count - 1)))
        for frame_index, (rgb_source, depth_source) in enumerate(
            zip(rgb_sources, depth_sources, strict=True)
        ):
            name = f"{frame_index:0{digits}d}.png"
            shutil.copy2(rgb_source, rgb_dir / name)
            shutil.copy2(depth_source, depth_dir / name)
        parquet_path = data_dir / f"{EPISODE_NAME}.parquet"
        _write_episode_parquet(parquet_path, camera.camera_to_world, intrinsic, base_extrinsic)
        (meta_dir / "episodes_stats.jsonl").write_text(
            json.dumps({"image_index": {"min": 0, "max": count - 1}}, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(config.paths.semantic_pointcloud_path, meta_dir / "pointcloud.ply")
        run_manifest = {
            "contract_version": CONTRACT_VERSION,
            "frame_count": count,
            "chunk_name": CHUNK_NAME,
            "episode_name": EPISODE_NAME,
            "inputs": {
                "camera_trajectory": str(config.paths.camera_trajectory_path),
                "camera_trajectory_sha256": file_sha256(config.paths.camera_trajectory_path),
                "render_manifest": str(render_manifest_path),
                "render_manifest_sha256": file_sha256(render_manifest_path),
                "semantic_pointcloud": str(config.paths.semantic_pointcloud_path),
                "semantic_pointcloud_sha256": file_sha256(config.paths.semantic_pointcloud_path),
            },
            "dataset": {
                "group_dir": config.dataset.group_dir,
                "scene_dir": config.dataset.scene_dir,
            },
        }
        (meta_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2), encoding="utf-8"
        )
        validate_target_scene(staging)
        if scene_dir.exists():
            if not config.dataset.overwrite:
                raise FileExistsError(
                    f"Target scene already exists and overwrite=false: {scene_dir}"
                )
            shutil.rmtree(scene_dir)
        staging.rename(scene_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return scene_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package camera and RGB-D files as target data.")
    parser.add_argument("--config", default="dataset_build.toml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_build_config(args.config)
    scene_dir = run_package_dataset(config)
    report = validate_target_scene(scene_dir)
    print(f"Saved target dataset scene: {scene_dir}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
