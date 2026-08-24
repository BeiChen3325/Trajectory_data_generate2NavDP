#!/usr/bin/env python3
"""Summarize the completed long100 RobotNav/NavDP acceptance artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {
    "observation.camera_intrinsic",
    "observation.camera_extrinsic",
    "observation.T_base_from_camera",
    "observation.T_camera_from_base",
    "observation.T_world_camera",
    "action",
}


def _tree(path: Path) -> dict[str, int]:
    files = 0
    size = 0
    for root, _dirs, names in os.walk(path):
        files += len(names)
        for name in names:
            size += (Path(root) / name).stat().st_size
    return {"file_count": files, "size_bytes": size}


def _matrix(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = []
    for value in frame[column]:
        array = np.asarray(value)
        if array.dtype == object:
            array = np.vstack(array)
        values.append(array.astype(np.float64))
    return np.stack(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--loader-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root
    outputs = root / "outputs"
    robotnav = root / "RobotNav/robotnav" / args.scene_name
    navdp = root / "NavDP/robotnav" / args.scene_name
    trajectory = json.loads((outputs / "trajectories/trajectory_manifest.json").read_text())
    entries = trajectory["trajectories"]
    lengths = np.asarray([item["path_length_m"] for item in entries], dtype=np.float64)
    frames = np.asarray([item["point_count"] for item in entries], dtype=np.int64)

    quality_paths = sorted(outputs.glob("dataset_build/episodes/*/rendered_episode/depth_quality_report.json"))
    qualities = [json.loads(path.read_text()) for path in quality_paths]
    render_manifests = sorted(outputs.glob("dataset_build/episodes/*/rendered_episode/render_manifest.json"))
    raw_rgb = sorted(outputs.glob("dataset_build/episodes/*/rendered_episode/rgb/*.png"))
    raw_depth = sorted(outputs.glob("dataset_build/episodes/*/rendered_episode/depth/*.png"))

    robotnav_parquets = sorted((robotnav / "data/chunk-000").glob("*.parquet"))
    navdp_parquets = sorted((navdp / "data/chunk-000").glob("*.parquet"))
    robotnav_rgb = sorted((robotnav / "videos/chunk-000/observation.images.rgb").glob("*.png"))
    robotnav_depth = sorted((robotnav / "videos/chunk-000/observation.images.depth").glob("*.png"))
    navdp_rgb = sorted((navdp / "videos/chunk-000/observation.images.rgb").glob("*.jpg"))
    navdp_depth = sorted((navdp / "videos/chunk-000/observation.images.depth").glob("*.png"))

    missing_columns: dict[str, list[str]] = {}
    navdp_rows = 0
    max_equal_error = 0.0
    max_inverse_error = 0.0
    max_action_contract_error = 0.0
    all_finite = True
    info = json.loads((navdp / "meta/info.json").read_text())
    coordinate_mapping = np.asarray(
        info["robotnav_conversion"]["coordinate_mapping"]["matrix"], dtype=np.float64
    )
    for parquet in navdp_parquets:
        frame = pd.read_parquet(parquet, engine="pyarrow")
        missing = sorted(REQUIRED - set(frame.columns))
        if missing:
            missing_columns[parquet.name] = missing
        extrinsic = _matrix(frame, "observation.camera_extrinsic")
        base_from_camera = _matrix(frame, "observation.T_base_from_camera")
        camera_from_base = _matrix(frame, "observation.T_camera_from_base")
        world_camera = _matrix(frame, "observation.T_world_camera")
        action = _matrix(frame, "action")
        intrinsic = _matrix(frame, "observation.camera_intrinsic")
        max_equal_error = max(max_equal_error, float(np.abs(extrinsic - base_from_camera).max()))
        max_inverse_error = max(
            max_inverse_error,
            float(np.abs(extrinsic @ camera_from_base - np.eye(4)).max()),
        )
        max_action_contract_error = max(
            max_action_contract_error,
            float(np.abs(action - coordinate_mapping @ world_camera).max()),
        )
        all_finite = all_finite and all(
            np.isfinite(value).all()
            for value in (extrinsic, base_from_camera, camera_from_base, world_camera, action, intrinsic)
        )
        navdp_rows += len(frame)

    robotnav_rows = sum(len(pd.read_parquet(path, columns=["action"])) for path in robotnav_parquets)
    conversion = json.loads((navdp / "meta/conversion_report.json").read_text())
    metadata = json.loads((navdp / "meta/metadata_validation_report.json").read_text())
    camera = json.loads((navdp / "meta/camera_transform_report.json").read_text())
    loader = json.loads(args.loader_report.read_text())
    trees = {name: _tree(path) for name, path in (("outputs", outputs), ("RobotNav", root / "RobotNav"), ("NavDP", root / "NavDP"))}

    report = {
        "status": "PASS",
        "root": str(root),
        "trajectory": {
            "count": len(entries),
            "mode": trajectory["trajectory_sampling"]["trajectory_mode"],
            "seed": trajectory["batch_seed"],
            "length_m": {
                "min": float(lengths.min()),
                "max": float(lengths.max()),
                "mean": float(lengths.mean()),
                "std": float(lengths.std()),
            },
            "frames": {
                "min": int(frames.min()),
                "max": int(frames.max()),
                "mean": float(frames.mean()),
                "std": float(frames.std()),
                "total": int(frames.sum()),
            },
            "collisions": int(sum(bool(item["smooth_path_collides"]) for item in entries)),
        },
        "counts": {
            "render_manifests": len(render_manifests),
            "raw_rgb_png": len(raw_rgb),
            "raw_depth_png": len(raw_depth),
            "robotnav_parquet_files": len(robotnav_parquets),
            "robotnav_parquet_rows": robotnav_rows,
            "robotnav_rgb_png": len(robotnav_rgb),
            "robotnav_depth_png": len(robotnav_depth),
            "navdp_parquet_files": len(navdp_parquets),
            "navdp_parquet_rows": navdp_rows,
            "navdp_rgb_jpeg": len(navdp_rgb),
            "navdp_depth_png": len(navdp_depth),
        },
        "depth": {
            "quality_report_count": len(qualities),
            "saturated_65535_pixels": int(sum(item["saturated_65535_pixels"] for item in qualities)),
            "total_pixels": int(sum(item["total_pixels"] for item in qualities)),
            "valid_pixels": int(sum(item["valid_pixels"] for item in qualities)),
        },
        "contracts": {
            "missing_required_columns": missing_columns,
            "all_matrices_finite": bool(all_finite),
            "camera_extrinsic_equals_T_base_from_camera_max_abs_error": max_equal_error,
            "camera_extrinsic_times_T_camera_from_base_max_abs_error": max_inverse_error,
            "action_equals_T_NavDP_world_from_camera_max_abs_error": max_action_contract_error,
            "converter_index_valid": conversion["index_valid"],
            "converter_original_index_valid": conversion["original_index_valid"],
            "converter_episode_count": conversion["episode_count"],
            "converter_frames": conversion["frames"],
            "camera_transform_report": camera,
            "metadata_status": metadata["status"],
        },
        "loader": loader,
        "trees": trees,
        "total_size_bytes": int(sum(item["size_bytes"] for item in trees.values())),
    }
    expected = args.expected_frames
    count_values = report["counts"]
    if (
        report["trajectory"]["count"] != args.expected_episodes
        or report["trajectory"]["collisions"] != 0
        or report["trajectory"]["length_m"]["min"] < 5.0
        or report["trajectory"]["length_m"]["max"] > 50.0
        or any(value != expected for key, value in count_values.items() if key.endswith(("_rows", "_png", "_jpeg")))
        or count_values["robotnav_parquet_files"] != args.expected_episodes
        or count_values["navdp_parquet_files"] != args.expected_episodes
        or report["depth"]["saturated_65535_pixels"] != 0
        or missing_columns
        or not all_finite
        or max_equal_error > 1e-6
        or max_inverse_error > 1e-5
        or max_action_contract_error > 1e-5
        or loader["status"] != "PASS"
    ):
        raise AssertionError(json.dumps(report))
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
