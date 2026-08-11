"""Report LAS, navigation-scene, and 3DGS world-bound alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from robotnav.navigation.scene.artifact import load_scene_artifact
from robotnav.navigation.scene.las_io import parse_las_header
from robotnav.navigation.scene.occupancy_map import compute_bounds_from_header_yup


def _overlap(
    first_min: np.ndarray, first_max: np.ndarray, second_min: np.ndarray, second_max: np.ndarray
) -> dict[str, object]:
    overlap_min = np.maximum(first_min, second_min)
    overlap_max = np.minimum(first_max, second_max)
    extent = np.maximum(0.0, overlap_max - overlap_min)
    return {
        "min_xz": overlap_min.tolist(),
        "max_xz": overlap_max.tolist(),
        "extent_m": extent.tolist(),
        "positive_area": bool(np.all(extent > 0.0)),
    }


def _ply_bounds_xz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertex = PlyData.read(path)["vertex"].data
    names = set(vertex.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"PLY has no x/y/z vertex properties: {path}")
    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)
    if xyz.size == 0 or not np.isfinite(xyz).all():
        raise ValueError(f"PLY has no finite vertices: {path}")
    return xyz[:, [0, 2]].min(axis=0), xyz[:, [0, 2]].max(axis=0)


def build_alignment_report(scene_dir: Path, las_path: Path, ply_path: Path) -> dict[str, object]:
    scene = load_scene_artifact(scene_dir)
    model = scene.model
    header = parse_las_header(las_path)
    las_min, las_max = compute_bounds_from_header_yup(header, model.axis_transform, padding=0.5)
    ply_min, ply_max = _ply_bounds_xz(ply_path)
    scene_min, scene_max = model.origin_xz, model.max_xz
    las_extent = las_max - las_min
    ply_extent = ply_max - ply_min
    return {
        "scene_dir": str(scene.scene_dir.resolve()),
        "las_path": str(las_path.resolve()),
        "ply_path": str(ply_path.resolve()),
        "coordinate_transform": {
            "las_to_navigation_world": model.axis_transform,
            "ply_assumption": "PLY x/y/z are already navigation-world coordinates",
            "navigation_ground_plane": "X-Z",
        },
        "bounds_xz": {
            "las_header_with_padding": {"min": las_min.tolist(), "max": las_max.tolist()},
            "navigation_scene": {"min": scene_min.tolist(), "max": scene_max.tolist()},
            "ply": {"min": ply_min.tolist(), "max": ply_max.tolist()},
        },
        "bounds_overlap": {
            "las_scene": _overlap(las_min, las_max, scene_min, scene_max),
            "scene_ply": _overlap(scene_min, scene_max, ply_min, ply_max),
            "las_ply": _overlap(las_min, las_max, ply_min, ply_max),
        },
        "scale_consistency": {
            "las_extent_m": las_extent.tolist(),
            "ply_extent_m": ply_extent.tolist(),
            "ply_to_las_extent_ratio_xz": (ply_extent / las_extent).tolist(),
            "finite": bool(np.isfinite(ply_extent / las_extent).all()),
        },
        "scene_map": {
            "shape_yx": list(model.planning_blocked.shape),
            "resolution_m": model.resolution_m,
            "world_bounds_source": scene.manifest.get("world_bounds", {}).get("source"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LAS, scene-map, and 3DGS bounds.")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--las", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_alignment_report(args.scene_dir, args.las, args.ply)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved alignment report: {args.output}")
    print(f"Scene/PLY positive overlap: {report['bounds_overlap']['scene_ply']['positive_area']}")


if __name__ == "__main__":
    main()
