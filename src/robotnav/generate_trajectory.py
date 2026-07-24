import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from robotnav.navigation.astar_planner import astar, choose_auto_start_goal, nearest_free_cell
from robotnav.navigation.config import load_map_config
from robotnav.navigation.ground_estimation import estimate_floor_y
from robotnav.navigation.las_io import iter_las_xyz, parse_las_header, sample_las_xyz
from robotnav.navigation.occupancy_map import (
    accumulate_ground_counts,
    accumulate_obstacle_counts,
    bounds_from_roi,
    clean_and_inflate_obstacles,
    clean_ground_mask,
    compute_bounds_from_header_yup,
    distance_transform_meters,
    filter_points_in_xz_bounds,
    grid_to_world,
    make_grid_spec,
    save_map_debug,
    world_to_grid,
)
from robotnav.navigation.path_smoothing import (
    chaikin_smooth,
    densify_polyline,
    path_collides_world,
    shortcut_path,
)
from robotnav.navigation.pointcloud_export import export_las_pointcloud
from robotnav.navigation.scene_obstacles import (
    SceneObstacleModel,
    save_scene_obstacle_model,
)
from robotnav.navigation.visualization import draw_path_debug

BASE_CONFIG = load_map_config()


def parse_xz(value):
    if value is None:
        return None
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected X,Z format, for example -2.0,3.5")
    return (float(parts[0]), float(parts[1]))


def parse_optional_xz(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    if str(value).lower() in ("none", "full", "global"):
        return None
    return parse_xz(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a 2.5D map from LAS and generate one A* trajectory."
    )
    parser.add_argument("--las", default=str(BASE_CONFIG.las_path))
    parser.add_argument("--output-dir", default=str(BASE_CONFIG.output_dir))
    parser.add_argument(
        "--axis-transform", choices=["zup-to-yup", "none"], default=BASE_CONFIG.axis_transform
    )
    parser.add_argument(
        "--floor-y",
        type=float,
        default=BASE_CONFIG.floor_y_override,
        help="Override automatic floor_y estimation.",
    )
    parser.add_argument("--floor-search-y-min", type=float, default=BASE_CONFIG.floor_search_y_min)
    parser.add_argument("--floor-search-y-max", type=float, default=BASE_CONFIG.floor_search_y_max)
    parser.add_argument(
        "--roi-center-xz",
        type=parse_optional_xz,
        default=BASE_CONFIG.roi_center_xz,
        help='ROI center in X,Z. Use "none" for full-scene mapping.',
    )
    parser.add_argument(
        "--roi-size-xz",
        type=parse_optional_xz,
        default=BASE_CONFIG.roi_size_xz,
        help='ROI size in X,Z meters. Use "none" for full-scene mapping.',
    )
    parser.add_argument("--resolution", type=float, default=BASE_CONFIG.resolution_m)
    parser.add_argument("--robot-radius", type=float, default=BASE_CONFIG.robot_radius_m)
    parser.add_argument("--robot-height", type=float, default=BASE_CONFIG.robot_height_m)
    parser.add_argument("--ground-margin", type=float, default=BASE_CONFIG.ground_margin_m)
    parser.add_argument("--safety-margin", type=float, default=BASE_CONFIG.safety_margin_m)
    parser.add_argument("--min-points-per-cell", type=int, default=BASE_CONFIG.min_points_per_cell)
    parser.add_argument("--start-xz", type=parse_xz, default=BASE_CONFIG.start_xz)
    parser.add_argument("--goal-xz", type=parse_xz, default=BASE_CONFIG.goal_xz)
    parser.add_argument("--seed", type=int, default=BASE_CONFIG.random_seed)
    parser.add_argument("--chunk-size", type=int, default=BASE_CONFIG.chunk_size)
    parser.add_argument(
        "--max-stream-points",
        type=int,
        default=BASE_CONFIG.max_stream_points,
        help="Debug limit for obstacle-map accumulation. 0 means use the full LAS.",
    )
    parser.add_argument("--floor-sample-limit", type=int, default=BASE_CONFIG.floor_sample_limit)
    parser.add_argument(
        "--min-start-goal-distance", type=float, default=BASE_CONFIG.min_start_goal_distance_m
    )
    return parser.parse_args()


def config_from_args(args):
    return replace(
        BASE_CONFIG,
        las_path=Path(args.las),
        output_dir=Path(args.output_dir),
        axis_transform=args.axis_transform,
        floor_y_override=args.floor_y,
        roi_center_xz=args.roi_center_xz,
        roi_size_xz=args.roi_size_xz,
        floor_search_y_min=args.floor_search_y_min,
        floor_search_y_max=args.floor_search_y_max,
        resolution_m=args.resolution,
        robot_radius_m=args.robot_radius,
        robot_height_m=args.robot_height,
        ground_margin_m=args.ground_margin,
        safety_margin_m=args.safety_margin,
        min_points_per_cell=args.min_points_per_cell,
        start_xz=args.start_xz,
        goal_xz=args.goal_xz,
        random_seed=args.seed,
        chunk_size=args.chunk_size,
        max_stream_points=args.max_stream_points,
        floor_sample_limit=args.floor_sample_limit,
        min_start_goal_distance_m=args.min_start_goal_distance,
    )


def main():
    args = parse_args()
    cfg = config_from_args(args)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("Stage 1/8: reading LAS header, resolving ROI, and sampling points for floor estimation")
    header = parse_las_header(cfg.las_path)
    if cfg.roi_center_xz is not None and cfg.roi_size_xz is not None:
        min_xz, max_xz = bounds_from_roi(cfg.roi_center_xz, cfg.roi_size_xz, padding=0.5)
        print(f"  using ROI XZ min={min_xz.tolist()}, max={max_xz.tolist()}")
    else:
        min_xz, max_xz = compute_bounds_from_header_yup(header, cfg.axis_transform, padding=0.5)
        print("  using full LAS bounds")

    if cfg.floor_y_override is None:
        sample = sample_las_xyz(
            cfg.las_path,
            max_points=cfg.floor_sample_limit,
            chunk_size=cfg.chunk_size,
            axis_transform=cfg.axis_transform,
        )
        sample = filter_points_in_xz_bounds(sample, min_xz, max_xz)
        if sample.shape[0] == 0:
            raise ValueError(
                "No sampled points inside ROI. Increase --roi-size-xz or use --roi-center-xz none."
            )
        floor_y, floor_report = estimate_floor_y(
            sample,
            bins=cfg.floor_hist_bins,
            xz_resolution=cfg.floor_xy_resolution_m,
            output_dir=cfg.output_dir,
            search_y_min=cfg.floor_search_y_min,
            search_y_max=cfg.floor_search_y_max,
        )
    else:
        floor_y = float(cfg.floor_y_override)
        floor_report = {"floor_y": floor_y, "source": "manual_override"}
        (cfg.output_dir / "floor_estimation_report.json").write_text(
            json.dumps(floor_report, indent=2), encoding="utf-8"
        )
    print(f"  estimated floor_y={floor_y:.4f}")

    print("Stage 2/8: building X-Z grid and accumulating obstacle-height points")
    spec = make_grid_spec(min_xz, max_xz, cfg.resolution_m)
    obstacle_counts = np.zeros((spec["height"], spec["width"]), dtype=np.uint32)
    ground_counts = np.zeros((spec["height"], spec["width"]), dtype=np.uint32)
    used_points = 0
    ground_points = 0
    seen_points = 0
    for xyz in iter_las_xyz(
        cfg.las_path, chunk_size=cfg.chunk_size, axis_transform=cfg.axis_transform
    ):
        if cfg.max_stream_points and seen_points >= cfg.max_stream_points:
            break
        if cfg.max_stream_points and seen_points + xyz.shape[0] > cfg.max_stream_points:
            xyz = xyz[: cfg.max_stream_points - seen_points]
        seen_points += xyz.shape[0]
        xyz = filter_points_in_xz_bounds(xyz, min_xz, max_xz)
        if xyz.shape[0] == 0:
            print(
                f"  streamed {seen_points}/{header['point_count']} points, no ROI points in chunk"
            )
            continue
        used_points += accumulate_obstacle_counts(
            counts=obstacle_counts,
            xyz_yup=xyz,
            floor_y=floor_y,
            robot_height=cfg.robot_height_m,
            ground_margin=cfg.ground_margin_m,
            spec=spec,
        )
        ground_points += accumulate_ground_counts(
            counts=ground_counts,
            xyz_yup=xyz,
            floor_y=floor_y,
            ground_band=cfg.ground_band_m,
            spec=spec,
        )
        print(
            f"  streamed {seen_points}/{header['point_count']} points, "
            f"obstacle-height points={used_points}, ground-band points={ground_points}"
        )

    print(
        "Stage 3/8: cleaning, inflating obstacles, computing traversable ground, and distance transform"
    )
    raw_ground = ground_counts >= cfg.min_ground_points_per_cell
    traversable_ground = clean_ground_mask(raw_ground, cfg)
    raw_obstacles = obstacle_counts >= cfg.min_points_per_cell
    cleaned, inflated, inflate_cells = clean_and_inflate_obstacles(raw_obstacles, cfg)
    planning_blocked = inflated | (~traversable_ground)
    raw_distance_m = distance_transform_meters(cleaned, cfg.resolution_m)
    planning_distance_m = distance_transform_meters(planning_blocked, cfg.resolution_m)
    scene_obstacles = SceneObstacleModel(
        obstacle_counts=obstacle_counts,
        ground_counts=ground_counts,
        raw_ground=raw_ground,
        traversable_ground=traversable_ground,
        raw_obstacles=raw_obstacles,
        cleaned_obstacles=cleaned,
        inflated_obstacles=inflated,
        planning_blocked=planning_blocked,
        raw_distance_m=raw_distance_m,
        planning_distance_m=planning_distance_m,
        origin_xz=np.asarray(spec["origin_xz"], dtype=np.float64),
        max_xz=np.asarray(spec["max_xz"], dtype=np.float64),
        resolution_m=cfg.resolution_m,
        floor_y=float(floor_y),
        axis_transform=cfg.axis_transform,
    )
    save_scene_obstacle_model(cfg.output_dir / "occupancy_map.npz", scene_obstacles)
    save_map_debug(
        cfg.output_dir,
        obstacle_counts,
        ground_counts,
        raw_ground,
        traversable_ground,
        raw_obstacles,
        cleaned,
        inflated,
        planning_blocked,
        raw_distance_m,
        spec,
        {
            "floor_report": floor_report,
            "seen_points": int(seen_points),
            "obstacle_height_points": int(used_points),
            "ground_band_points": int(ground_points),
            "inflate_cells": int(inflate_cells),
            "config": serialize_config(cfg),
        },
    )

    print("Stage 4/8: selecting start/goal and running A*")
    free = ~planning_blocked
    if cfg.start_xz is not None:
        start = nearest_free_cell(world_to_grid(cfg.start_xz, spec), free)
    else:
        start = None
    if cfg.goal_xz is not None:
        goal = nearest_free_cell(world_to_grid(cfg.goal_xz, spec), free)
    else:
        goal = None
    if start is None or goal is None:
        auto_start, auto_goal = choose_auto_start_goal(
            free,
            planning_distance_m,
            spec,
            min_distance_m=cfg.min_start_goal_distance_m,
            seed=cfg.random_seed,
        )
        start = auto_start if start is None else start
        goal = auto_goal if goal is None else goal

    astar_path = astar(
        start,
        goal,
        planning_blocked,
        distance_m=planning_distance_m,
        resolution=cfg.resolution_m,
        obstacle_cost_weight=cfg.obstacle_cost_weight,
        obstacle_cost_power=cfg.obstacle_cost_power,
    )
    print(f"  A* cells: {len(astar_path)}")

    print("Stage 5/8: shortcutting and smoothing path")
    shortcut = shortcut_path(
        astar_path, planning_blocked, passes=cfg.shortcut_passes, seed=cfg.random_seed
    )
    shortcut_world = grid_to_world(shortcut, spec)
    dense = densify_polyline(shortcut_world, samples_per_meter=cfg.smooth_samples_per_meter)
    smooth = chaikin_smooth(dense, iterations=2)
    smooth_collides = path_collides_world(
        smooth, planning_blocked, lambda pts: world_to_grid(pts, spec)
    )
    if smooth_collides:
        print("  smoothed path collides after interpolation; falling back to dense shortcut path")
        smooth = dense
        smooth_collides = path_collides_world(
            smooth, planning_blocked, lambda pts: world_to_grid(pts, spec)
        )

    print("Stage 6/8: saving trajectory and debug images")
    astar_world = grid_to_world(astar_path, spec)
    start_world = grid_to_world(start, spec)
    goal_world = grid_to_world(goal, spec)
    trajectory = {
        "coordinate_convention": "Y-up internal coordinates; ground plane is X-Z; physical down is +Y and physical up is -Y.",
        "floor_y": float(floor_y),
        "robot_base_y": float(floor_y),
        "start_xz": start_world.tolist(),
        "goal_xz": goal_world.tolist(),
        "astar_path_xz": astar_world.tolist(),
        "shortcut_path_xz": shortcut_world.tolist(),
        "smooth_path_xz": smooth.tolist(),
        "smooth_path_collides": bool(smooth_collides),
        "grid_start_xy": start.tolist(),
        "grid_goal_xy": goal.tolist(),
        "config": serialize_config(cfg),
    }
    (cfg.output_dir / "trajectory.json").write_text(
        json.dumps(trajectory, indent=2), encoding="utf-8"
    )
    draw_path_debug(
        cfg.output_dir / "06_path_debug.png",
        cleaned,
        inflated,
        planning_blocked,
        planning_distance_m,
        astar_path,
        shortcut,
        smooth_xy=smooth,
        world_to_grid_fn=lambda pts: world_to_grid(pts, spec),
    )

    print("Stage 7/8: exporting target-compatible physical-obstacle point cloud")
    if cfg.pointcloud.enabled:
        pointcloud_report = export_las_pointcloud(
            cfg.las_path,
            cfg.output_dir,
            scene_obstacles,
            cfg.pointcloud,
            ground_margin_m=cfg.ground_margin_m,
            chunk_size=cfg.chunk_size,
            max_stream_points=cfg.max_stream_points,
        )
        obstacle_points = pointcloud_report["counts"]["obstacle_representative_points"]
        context_points = pointcloud_report["counts"]["context_representative_points"]
        print(
            f"  pointcloud representative points: obstacles={obstacle_points}, "
            f"context={context_points}"
        )
    else:
        print("  pointcloud export disabled by configuration")

    print("Stage 8/8: done")
    print(f"  output_dir={cfg.output_dir}")
    print(f"  trajectory points={len(smooth)}, collision={smooth_collides}")


def serialize_config(cfg):
    raw = asdict(cfg)
    raw["las_path"] = str(raw["las_path"])
    raw["output_dir"] = str(raw["output_dir"])
    return raw


if __name__ == "__main__":
    main()
