"""Create a filtered 3DGS PLY without modifying the source file.

The filter is intentionally conservative: it removes only Gaussians outside the
LAS navigation-world AABB (with a margin), oversized splats, and nearly
transparent splats.  It preserves every vertex property used by gsplat.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from robotnav.navigation.scene.las_io import parse_las_header


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--las", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roi-margin-m", type=float, default=1.0)
    parser.add_argument("--max-scale-m", type=float, default=5.0)
    parser.add_argument("--min-opacity", type=float, default=0.01)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.roi_margin_m < 0 or args.max_scale_m <= 0 or not 0 <= args.min_opacity < 1:
        raise ValueError("Invalid filter threshold")

    ply = PlyData.read(str(args.input), mmap=True)
    if "vertex" not in ply or len(ply.elements) != 1:
        raise ValueError("Expected a PLY with exactly one vertex element")
    vertex = ply["vertex"].data
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"}
    if not required.issubset(vertex.dtype.names or ()):
        raise ValueError("PLY lacks required Gaussian properties")

    header = parse_las_header(args.las)
    las_min = np.asarray(header["min_xyz"], dtype=np.float64)
    las_max = np.asarray(header["max_xyz"], dtype=np.float64)
    world_min = np.array([las_min[0], -las_max[2], las_min[1]]) - args.roi_margin_m
    world_max = np.array([las_max[0], -las_min[2], las_max[1]]) + args.roi_margin_m

    xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float64)
    log_scales = np.column_stack(
        [vertex[name] for name in ("scale_0", "scale_1", "scale_2")]
    ).astype(np.float64)
    max_scale = np.exp(log_scales).max(axis=1)
    opacity = 1.0 / (1.0 + np.exp(-vertex["opacity"].astype(np.float64)))
    keep = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(log_scales).all(axis=1)
        & np.isfinite(opacity)
        & np.all((xyz >= world_min) & (xyz <= world_max), axis=1)
        & (max_scale <= args.max_scale_m)
        & (opacity >= args.min_opacity)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered = np.asarray(vertex[keep]).copy()
    PlyData([PlyElement.describe(filtered, "vertex")], text=False, byte_order=ply.byte_order).write(
        str(args.output)
    )
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "input_count": int(len(vertex)),
                "kept_count": int(keep.sum()),
                "removed_count": int((~keep).sum()),
                "removed_ratio": float((~keep).mean()),
                "world_aabb_min": world_min.tolist(),
                "world_aabb_max": world_max.tolist(),
                "max_scale_m": args.max_scale_m,
                "min_opacity": args.min_opacity,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
