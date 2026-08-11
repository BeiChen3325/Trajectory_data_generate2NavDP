"""Filter spatially implausible 3DGS Gaussians without altering their coordinates.

The input PLY is never modified.  All vertex fields are preserved verbatim for
retained rows, including rotation and spherical-harmonic color properties.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement

from robotnav.navigation.scene.las_io import parse_las_header


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    return np.where(
        values >= 0.0,
        1.0 / (1.0 + np.exp(-values)),
        np.exp(values) / (1.0 + np.exp(values)),
    )


def _quantiles(values: np.ndarray, quantiles: np.ndarray) -> dict[str, Any]:
    return {"quantiles": quantiles, "values": np.quantile(values, quantiles, axis=0)}


def _navigation_bounds(las_path: Path) -> tuple[np.ndarray, np.ndarray]:
    header = parse_las_header(las_path)
    source_min = np.asarray(header["min_xyz"], dtype=np.float64)
    source_max = np.asarray(header["max_xyz"], dtype=np.float64)
    return (
        np.array([source_min[0], -source_max[2], source_min[1]], dtype=np.float64),
        np.array([source_max[0], -source_min[2], source_max[1]], dtype=np.float64),
    )


def _bounds(points: np.ndarray) -> dict[str, np.ndarray]:
    return {"min": points.min(axis=0), "max": points.max(axis=0)}


def filter_gaussians(
    *,
    ply_path: Path,
    las_path: Path,
    output_path: Path,
    report_path: Path,
    before_after_path: Path,
    spatial_margin_ratio: float,
    scale_threshold_m: float,
    low_opacity_quantile: float,
    huge_scale_quantile: float,
) -> dict[str, Any]:
    """Apply data-derived masks and write a filtered PLY plus JSON reports."""
    ply = PlyData.read(ply_path)
    if "vertex" not in ply:
        raise ValueError(f"{ply_path} has no vertex element")
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"}
    missing = required - names
    if missing:
        raise ValueError(f"{ply_path} is missing required vertex fields: {sorted(missing)}")

    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)
    log_scales = np.column_stack(
        (vertex["scale_0"], vertex["scale_1"], vertex["scale_2"])
    ).astype(np.float64)
    metric_scales = np.exp(log_scales)
    max_metric_scale = metric_scales.max(axis=1)
    opacity_raw = np.asarray(vertex["opacity"], dtype=np.float64)
    opacity = _stable_sigmoid(opacity_raw)
    nav_min, nav_max = _navigation_bounds(las_path)
    margin_m = (nav_max - nav_min) * spatial_margin_ratio
    extended_min, extended_max = nav_min - margin_m, nav_max + margin_m

    spatial_outlier = np.any((xyz < extended_min) | (xyz > extended_max), axis=1)
    outside_las_bounds = np.any((xyz < nav_min) | (xyz > nav_max), axis=1)
    large_scale = max_metric_scale > scale_threshold_m
    low_opacity_threshold = float(np.quantile(opacity, low_opacity_quantile))
    low_opacity = opacity <= low_opacity_threshold
    huge_scale_threshold_m = float(
        max(scale_threshold_m, np.quantile(max_metric_scale, huge_scale_quantile))
    )
    huge_scale = max_metric_scale >= huge_scale_threshold_m

    # A large Gaussian is not deleted merely for its size.  It must also be
    # spatially implausible or nearly transparent.  The opacity mask catches
    # extreme, low-contribution training residue that remains inside the margin.
    scale_outlier = large_scale & (spatial_outlier | low_opacity)
    opacity_outlier = low_opacity & huge_scale
    remove = spatial_outlier | scale_outlier | opacity_outlier
    keep = ~remove
    if not np.any(keep):
        raise ValueError("Filtering removed every Gaussian; refusing to write an empty PLY")

    kept_vertex = vertex[keep]
    elements = [
        PlyElement.describe(kept_vertex, "vertex") if element.name == "vertex" else element
        for element in ply.elements
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        elements,
        text=ply.text,
        byte_order=ply.byte_order,
        comments=ply.comments,
        obj_info=ply.obj_info,
    ).write(output_path)

    quantile_levels = np.asarray([0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0])
    report = {
        "inputs": {"ply": str(ply_path), "las": str(las_path)},
        "output": str(output_path),
        "policy": {
            "coordinate_operation": "delete-only; no scale, translation, or rotation",
            "spatial_margin_ratio": spatial_margin_ratio,
            "spatial_margin_m_xyz": margin_m,
            "scale_threshold_m": scale_threshold_m,
            "low_opacity_quantile": low_opacity_quantile,
            "low_opacity_threshold": low_opacity_threshold,
            "huge_scale_quantile": huge_scale_quantile,
            "huge_scale_threshold_m": huge_scale_threshold_m,
        },
        "original_gaussian_count": int(len(vertex)),
        "filtered_gaussian_count": int(keep.sum()),
        "removed_gaussian_count": int(remove.sum()),
        "retention_ratio": float(keep.mean()),
        "removal_ratio": float(remove.mean()),
        "navigation_las_bounds": {"min": nav_min, "max": nav_max},
        "extended_spatial_bounds": {"min": extended_min, "max": extended_max},
        "mask_statistics": {
            "spatial_outlier": int(spatial_outlier.sum()),
            "outside_raw_las_bounds": int(outside_las_bounds.sum()),
            "large_scale": int(large_scale.sum()),
            "low_opacity": int(low_opacity.sum()),
            "large_scale_and_spatial_outlier": int((large_scale & spatial_outlier).sum()),
            "large_scale_and_low_opacity": int((large_scale & low_opacity).sum()),
            "large_scale_low_opacity_spatial_outlier": int(
                (large_scale & low_opacity & spatial_outlier).sum()
            ),
            "scale_outlier": int(scale_outlier.sum()),
            "opacity_outlier": int(opacity_outlier.sum()),
            "combined_removed": int(remove.sum()),
        },
        "xyz_bounds": {"before": _bounds(xyz), "after": _bounds(xyz[keep])},
        "scale_metric_statistics": {
            "before": _quantiles(metric_scales, quantile_levels),
            "after": _quantiles(metric_scales[keep], quantile_levels),
        },
        "opacity_statistics": {
            "before": _quantiles(opacity, quantile_levels),
            "after": _quantiles(opacity[keep], quantile_levels),
        },
        "preserved_vertex_fields": list(vertex.dtype.names or ()),
        "rotation_field_count": len([name for name in names if name.startswith("rot_")]),
        "sh_color_field_count": len(
            [name for name in names if name.startswith("f_dc_") or name.startswith("f_rest_")]
        ),
    }
    before_after = {
        "before": {"xyz_bounds": _bounds(xyz), "gaussian_count": int(len(vertex))},
        "after": {"xyz_bounds": _bounds(xyz[keep]), "gaussian_count": int(keep.sum())},
    }
    report_path.write_text(json.dumps(report, indent=2, default=_json_value) + "\n", encoding="utf-8")
    before_after_path.write_text(
        json.dumps(before_after, indent=2, default=_json_value) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, default=Path("data/input/forth_local3DGS_yup.ply"))
    parser.add_argument("--las", type=Path, default=Path("data/input/forth_local.las"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/input/forth_local3DGS_yup_filtered.ply")
    )
    parser.add_argument("--report", type=Path, default=Path("gaussian_filter_report.json"))
    parser.add_argument("--before-after", type=Path, default=Path("before_after_bounds.json"))
    parser.add_argument("--spatial-margin-ratio", type=float, default=0.10)
    parser.add_argument("--scale-threshold-m", type=float, default=10.0)
    parser.add_argument("--low-opacity-quantile", type=float, default=0.05)
    parser.add_argument("--huge-scale-quantile", type=float, default=0.999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.spatial_margin_ratio <= 1.0:
        raise ValueError("--spatial-margin-ratio must be in [0, 1]")
    if args.scale_threshold_m <= 0.0:
        raise ValueError("--scale-threshold-m must be positive")
    if not 0.0 < args.low_opacity_quantile < 1.0:
        raise ValueError("--low-opacity-quantile must be in (0, 1)")
    if not 0.0 < args.huge_scale_quantile < 1.0:
        raise ValueError("--huge-scale-quantile must be in (0, 1)")
    report = filter_gaussians(
        ply_path=args.ply,
        las_path=args.las,
        output_path=args.output,
        report_path=args.report,
        before_after_path=args.before_after,
        spatial_margin_ratio=args.spatial_margin_ratio,
        scale_threshold_m=args.scale_threshold_m,
        low_opacity_quantile=args.low_opacity_quantile,
        huge_scale_quantile=args.huge_scale_quantile,
    )
    print(
        "Filtered "
        f"{report['original_gaussian_count']} -> {report['filtered_gaussian_count']} Gaussians "
        f"({report['removal_ratio']:.4%} removed): {args.output}"
    )


if __name__ == "__main__":
    main()
