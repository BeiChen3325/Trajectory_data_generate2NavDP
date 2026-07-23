import json
from pathlib import Path

import cv2
import numpy as np


def compute_bounds_from_header_yup(header, axis_transform, padding=0.5):
    corners = np.array(
        [
            [header["min_xyz"][0], header["min_xyz"][1], header["min_xyz"][2]],
            [header["min_xyz"][0], header["min_xyz"][1], header["max_xyz"][2]],
            [header["min_xyz"][0], header["max_xyz"][1], header["min_xyz"][2]],
            [header["min_xyz"][0], header["max_xyz"][1], header["max_xyz"][2]],
            [header["max_xyz"][0], header["min_xyz"][1], header["min_xyz"][2]],
            [header["max_xyz"][0], header["min_xyz"][1], header["max_xyz"][2]],
            [header["max_xyz"][0], header["max_xyz"][1], header["min_xyz"][2]],
            [header["max_xyz"][0], header["max_xyz"][1], header["max_xyz"][2]],
        ],
        dtype=np.float64,
    )
    if axis_transform == "zup-to-yup":
        transformed = np.empty_like(corners)
        transformed[:, 0] = corners[:, 0]
        transformed[:, 1] = -corners[:, 2]
        transformed[:, 2] = corners[:, 1]
        corners = transformed
    elif axis_transform != "none":
        raise ValueError(f"Unknown axis transform: {axis_transform}")

    min_xz = np.array([corners[:, 0].min() - padding, corners[:, 2].min() - padding])
    max_xz = np.array([corners[:, 0].max() + padding, corners[:, 2].max() + padding])
    return min_xz, max_xz


def make_grid_spec(min_xz, max_xz, resolution):
    size = np.ceil((max_xz - min_xz) / resolution).astype(np.int64) + 1
    return {
        "origin_xz": np.asarray(min_xz, dtype=np.float64),
        "max_xz": np.asarray(max_xz, dtype=np.float64),
        "resolution": float(resolution),
        "width": int(size[0]),
        "height": int(size[1]),
    }


def bounds_from_roi(center_xz, size_xz, padding=0.0):
    center = np.asarray(center_xz, dtype=np.float64)
    size = np.asarray(size_xz, dtype=np.float64)
    half = size * 0.5 + float(padding)
    return center - half, center + half


def filter_points_in_xz_bounds(xyz_yup, min_xz, max_xz):
    mask = (
        (xyz_yup[:, 0] >= min_xz[0])
        & (xyz_yup[:, 0] <= max_xz[0])
        & (xyz_yup[:, 2] >= min_xz[1])
        & (xyz_yup[:, 2] <= max_xz[1])
    )
    return xyz_yup[mask]


def world_to_grid(xz, spec):
    xz = np.asarray(xz, dtype=np.float64)
    ij = np.floor((xz - spec["origin_xz"]) / spec["resolution"]).astype(np.int64)
    return ij


def grid_to_world(ij, spec):
    ij = np.asarray(ij, dtype=np.float64)
    return spec["origin_xz"] + (ij + 0.5) * spec["resolution"]


def accumulate_obstacle_counts(counts, xyz_yup, floor_y, robot_height, ground_margin, spec):
    # +Y is physical downward in this scene convention. Points above the floor
    # have smaller y, so height above floor is floor_y - y.
    heights = floor_y - xyz_yup[:, 1]
    mask = (heights >= ground_margin) & (heights <= robot_height)
    if not np.any(mask):
        return 0

    xz = xyz_yup[mask][:, [0, 2]]
    ij = world_to_grid(xz, spec)
    valid = (
        (ij[:, 0] >= 0) & (ij[:, 0] < spec["width"]) & (ij[:, 1] >= 0) & (ij[:, 1] < spec["height"])
    )
    if not np.any(valid):
        return 0
    ij = ij[valid]
    np.add.at(counts, (ij[:, 1], ij[:, 0]), 1)
    return int(ij.shape[0])


def accumulate_ground_counts(counts, xyz_yup, floor_y, ground_band, spec):
    mask = np.abs(xyz_yup[:, 1] - floor_y) <= ground_band
    if not np.any(mask):
        return 0

    xz = xyz_yup[mask][:, [0, 2]]
    ij = world_to_grid(xz, spec)
    valid = (
        (ij[:, 0] >= 0) & (ij[:, 0] < spec["width"]) & (ij[:, 1] >= 0) & (ij[:, 1] < spec["height"])
    )
    if not np.any(valid):
        return 0
    ij = ij[valid]
    np.add.at(counts, (ij[:, 1], ij[:, 0]), 1)
    return int(ij.shape[0])


def remove_small_components(binary, min_cells):
    if min_cells <= 1:
        return binary
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    cleaned = np.zeros_like(binary, dtype=bool)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_cells:
            cleaned[labels == label] = True
    return cleaned


def clean_and_inflate_obstacles(raw_obstacles, config):
    obstacles = raw_obstacles.astype(np.uint8)
    if config.open_kernel_cells > 0:
        k = 2 * config.open_kernel_cells + 1
        kernel = np.ones((k, k), np.uint8)
        obstacles = cv2.morphologyEx(obstacles, cv2.MORPH_OPEN, kernel)
    if config.close_kernel_cells > 0:
        k = 2 * config.close_kernel_cells + 1
        kernel = np.ones((k, k), np.uint8)
        obstacles = cv2.morphologyEx(obstacles, cv2.MORPH_CLOSE, kernel)

    cleaned = remove_small_components(obstacles > 0, config.min_obstacle_component_cells)
    inflate_cells = int(
        np.ceil((config.robot_radius_m + config.safety_margin_m) / config.resolution_m)
    )
    if inflate_cells > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * inflate_cells + 1, 2 * inflate_cells + 1)
        )
        inflated = cv2.dilate(cleaned.astype(np.uint8), kernel) > 0
    else:
        inflated = cleaned
    return cleaned, inflated, inflate_cells


def clean_ground_mask(raw_ground, config):
    ground = raw_ground.astype(np.uint8)
    if config.ground_close_kernel_cells > 0:
        k = 2 * config.ground_close_kernel_cells + 1
        kernel = np.ones((k, k), np.uint8)
        ground = cv2.morphologyEx(ground, cv2.MORPH_CLOSE, kernel)
    return ground > 0


def distance_transform_meters(obstacles, resolution):
    free_u8 = (~obstacles).astype(np.uint8)
    dist_cells = cv2.distanceTransform(free_u8, cv2.DIST_L2, 5)
    return dist_cells.astype(np.float32) * float(resolution)


def save_map_debug(
    output_dir,
    obstacle_counts,
    ground_counts,
    raw_ground,
    traversable_ground,
    raw_obstacles,
    cleaned,
    inflated,
    planning_blocked,
    distance_m,
    spec,
    metadata,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count_img = np.zeros_like(obstacle_counts, dtype=np.uint8)
    positive = obstacle_counts[obstacle_counts > 0]
    if positive.size:
        hi = np.percentile(positive, 99.0)
        hi = max(float(hi), 1.0)
        count_img = np.clip(np.log1p(obstacle_counts) / np.log1p(hi) * 255.0, 0, 255).astype(
            np.uint8
        )
    write_debug_image(output_dir / "01_obstacle_point_counts.png", count_img)
    write_debug_image(output_dir / "02_raw_obstacles.png", raw_obstacles.astype(np.uint8) * 255)
    write_debug_image(output_dir / "03_cleaned_obstacles.png", cleaned.astype(np.uint8) * 255)
    write_debug_image(
        output_dir / "04_planning_obstacles_inflated.png", inflated.astype(np.uint8) * 255
    )
    write_debug_image(output_dir / "05_raw_ground.png", raw_ground.astype(np.uint8) * 255)
    write_debug_image(
        output_dir / "06_traversable_ground.png", traversable_ground.astype(np.uint8) * 255
    )
    write_debug_image(
        output_dir / "07_planning_blocked_unknown_or_obstacle.png",
        planning_blocked.astype(np.uint8) * 255,
    )

    dist_vis = np.clip(distance_m / max(float(np.percentile(distance_m, 98.0)), 0.1), 0.0, 1.0)
    dist_vis = (dist_vis * 255.0).astype(np.uint8)
    write_debug_image(output_dir / "08_distance_transform.png", dist_vis)

    serializable_spec = {
        "origin_xz": spec["origin_xz"].tolist(),
        "max_xz": spec["max_xz"].tolist(),
        "resolution": spec["resolution"],
        "width": spec["width"],
        "height": spec["height"],
    }
    report = dict(metadata)
    report["grid_spec"] = serializable_spec
    (output_dir / "map_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_debug_image(path, image):
    scale = max(1, min(8, 900 // max(image.shape[0], image.shape[1], 1)))
    if scale > 1:
        image = cv2.resize(
            image, (image.shape[1] * scale, image.shape[0] * scale), interpolation=cv2.INTER_NEAREST
        )
    cv2.imwrite(str(path), image)
