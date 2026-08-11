"""Ground-plane estimation for navigation scene construction."""

import json
from pathlib import Path

import cv2
import numpy as np


def estimate_floor_y(
    points_yup,
    bins=180,
    xz_resolution=0.25,
    output_dir=None,
    search_y_min=0.0,
    search_y_max=3.0,
    coverage_ratio=0.25,
):
    if points_yup.size == 0:
        raise ValueError("Cannot estimate floor from an empty point sample.")

    all_y = points_yup[:, 1]
    search_mask = (all_y >= search_y_min) & (all_y <= search_y_max)
    if not np.any(search_mask):
        raise ValueError(
            "No points inside the floor search Y range "
            f"[{search_y_min}, {search_y_max}]. Adjust --floor-search-y-min/max."
        )

    points = points_yup[search_mask]
    y = points[:, 1]
    x = points[:, 0]
    z = points[:, 2]
    hist_counts, edges = np.histogram(y, bins=bins)

    x_cell = np.floor((x - x.min()) / xz_resolution).astype(np.int64)
    z_cell = np.floor((z - z.min()) / xz_resolution).astype(np.int64)
    bin_ids = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, bins - 1)
    flat_xz = x_cell * (z_cell.max() + 1) + z_cell

    coverage = np.zeros(bins, dtype=np.int64)
    for bid in range(bins):
        mask = bin_ids == bid
        if np.any(mask):
            coverage[bid] = np.unique(flat_xz[mask]).size

    max_coverage = int(coverage.max())
    if max_coverage <= 0:
        floor_bin = int(np.argmax(hist_counts))
    else:
        candidates = np.where(coverage >= max_coverage * coverage_ratio)[0]
        if candidates.size == 0:
            floor_bin = int(np.argmax(coverage))
        else:
            # In this scene convention, physical downward is +Y. The floor is
            # the high-Y broad horizontal plane; ceilings are lower Y planes.
            floor_bin = int(candidates[np.argmax(edges[candidates])])

    floor_y = float((edges[floor_bin] + edges[floor_bin + 1]) * 0.5)
    report = {
        "floor_y": floor_y,
        "floor_bin": floor_bin,
        "bin_low": float(edges[floor_bin]),
        "bin_high": float(edges[floor_bin + 1]),
        "max_coverage": max_coverage,
        "floor_bin_coverage": int(coverage[floor_bin]),
        "floor_bin_count": int(hist_counts[floor_bin]),
        "sample_count": int(points_yup.shape[0]),
        "search_sample_count": int(points.shape[0]),
        "search_y_min": float(search_y_min),
        "search_y_max": float(search_y_max),
        "coverage_ratio": float(coverage_ratio),
        "all_y_min": float(all_y.min()),
        "all_y_max": float(all_y.max()),
        "search_y_observed_min": float(y.min()),
        "search_y_observed_max": float(y.max()),
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "floor_estimation_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        save_floor_histogram_debug(edges, hist_counts, coverage, floor_bin, output_dir)

    return floor_y, report


def save_floor_histogram_debug(edges, counts, coverage, floor_bin, output_dir):
    width, height = 1200, 500
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    pad = 48
    plot_w = width - pad * 2
    plot_h = height - pad * 2
    max_count = max(float(counts.max()), 1.0)
    max_cov = max(float(coverage.max()), 1.0)

    for i in range(len(counts)):
        x0 = pad + int(i / len(counts) * plot_w)
        x1 = pad + int((i + 1) / len(counts) * plot_w)
        bar_h = int(counts[i] / max_count * plot_h)
        cov_h = int(coverage[i] / max_cov * plot_h)
        cv2.rectangle(
            canvas, (x0, height - pad - bar_h), (max(x1, x0 + 1), height - pad), (190, 210, 235), -1
        )
        cv2.line(
            canvas,
            (x0, height - pad - cov_h),
            (max(x1, x0 + 1), height - pad - cov_h),
            (50, 120, 40),
            2,
        )

    fx = pad + int((floor_bin + 0.5) / len(counts) * plot_w)
    cv2.line(canvas, (fx, pad), (fx, height - pad), (0, 0, 255), 2)
    label = f"floor_y ~= {(edges[floor_bin] + edges[floor_bin + 1]) * 0.5:.3f}"
    cv2.putText(canvas, label, (pad, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite(str(Path(output_dir) / "floor_y_histogram.png"), canvas)
