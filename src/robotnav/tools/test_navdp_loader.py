"""Run an integration read using InternNav's actual NavDP dataset loader."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

INTERNNAV_LOADER_ENV = "INTERNNAV_LOADER_PATH"
SAMPLE_FRAME_COUNT = 5


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_loader_class(loader_path: Path) -> type:
    if not loader_path.is_file():
        raise FileNotFoundError(loader_path)
    spec = importlib.util.spec_from_file_location("navdp_lerobot_dataset_under_test", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import NavDP loader: {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NavDP_Base_Datset


def _matrix(value: Any, shape: tuple[int, int]) -> np.ndarray:
    data = value.tolist() if isinstance(value, np.ndarray) else value
    return np.asarray(data, dtype=np.float64).reshape(shape)


def _depth_geometry_diagnostic(
    depth_paths: list[Path],
    actions: np.ndarray,
    intrinsic: np.ndarray,
    pointcloud_path: Path,
    frame_indexes: np.ndarray,
) -> dict[str, Any]:
    """Project optical-Z depth through actions and compare diagnostically to occupancy."""
    import cv2

    import open3d as o3d

    pointcloud = o3d.io.read_point_cloud(str(pointcloud_path))
    occupancy_xy = np.asarray(pointcloud.points, dtype=np.float64)[:, :2]
    if len(occupancy_xy) == 0:
        raise ValueError("NavDP occupancy pointcloud is empty")
    nearest_distances = []
    projected_points = []
    rng = np.random.default_rng(17)
    for frame_index in frame_indexes:
        depth = cv2.imread(str(depth_paths[int(frame_index)]), cv2.IMREAD_UNCHANGED)
        valid_pixels = np.argwhere((depth >= 1000) & (depth <= 50000))
        if not len(valid_pixels):
            continue
        row, column = valid_pixels[rng.integers(len(valid_pixels))]
        z_depth = float(depth[row, column]) / 10000.0
        point_camera = np.array(
            [
                (float(column) - intrinsic[0, 2]) * z_depth / intrinsic[0, 0],
                (float(row) - intrinsic[1, 2]) * z_depth / intrinsic[1, 1],
                z_depth,
                1.0,
            ]
        )
        point_world = actions[int(frame_index)] @ point_camera
        if not np.isfinite(point_world).all() or point_world[3] == 0.0:
            raise ValueError("Depth optical-Z projection produced a non-finite world point")
        point_xy = point_world[:2] / point_world[3]
        projected_points.append(point_xy)
        nearest_distances.append(
            float(np.linalg.norm(occupancy_xy - point_xy[None, :], axis=1).min())
        )
    if not projected_points:
        raise ValueError("No valid sampled depth pixels are available for optical-Z projection")
    return {
        "projection": "p_camera=[(u-cx)Z/fx,(v-cy)Z/fy,Z,1]; p_navdp=action@p_camera",
        "sample_count": len(projected_points),
        "finite_projected_points": True,
        "nearest_occupancy_distance_m": {
            "min": float(np.min(nearest_distances)),
            "mean": float(np.mean(nearest_distances)),
        },
        "warning": (
            "Occupancy contains only collision-height obstacle projections, not every rendered surface; "
            "nearest distance is a geometry sanity diagnostic, not a pixelwise depth error."
        ),
    }


def _resolve_loader_path(loader_path: Path | None) -> Path:
    if loader_path is not None:
        return loader_path
    configured = os.environ.get(INTERNNAV_LOADER_ENV)
    if configured:
        return Path(configured)
    raise ValueError(
        "InternNav loader path is required; pass --loader-path or set "
        f"{INTERNNAV_LOADER_ENV}"
    )


def run_loader_test(
    dataset_root: Path, scene: str, loader_path: Path | None = None
) -> dict[str, Any]:
    """Load the requested scene with InternNav and run one sample read."""
    scene_dir = dataset_root / scene
    report_path = scene_dir / "meta" / "loader_test_report.json"
    report: dict[str, Any] = {
        "load_success": False,
        "sample_frames": [],
        "errors": [],
        "warnings": [],
    }
    try:
        parquet_path = scene_dir / "data" / "chunk-000" / "episode_000000.parquet"
        rgb_dir = scene_dir / "videos" / "chunk-000" / "observation.images.rgb"
        depth_dir = scene_dir / "videos" / "chunk-000" / "observation.images.depth"
        if not parquet_path.is_file():
            raise FileNotFoundError(f"schema/path: missing parquet {parquet_path}")
        all_rgb_paths = sorted(rgb_dir.glob("*.jpg"))
        all_depth_paths = sorted(depth_dir.glob("*.png"))
        frame = pd.read_parquet(parquet_path, engine="pyarrow")
        episode_stats = [
            json.loads(line)
            for line in (scene_dir / "meta" / "episodes_stats.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line
        ]
        if not episode_stats:
            raise ValueError("schema/path: episodes_stats.jsonl is empty")
        image_index = episode_stats[0].get("image_index", {})
        minimum, maximum = image_index.get("min"), image_index.get("max")
        if (
            not isinstance(minimum, int)
            or not isinstance(maximum, int)
            or minimum < 0
            or maximum < minimum
            or maximum >= len(all_rgb_paths)
            or maximum >= len(all_depth_paths)
        ):
            raise ValueError("schema/path: episode 0 has an invalid image range")
        # Images are global scene sequences.  A parquet contains exactly one
        # episode, so use only its declared global image interval.
        rgb_paths = all_rgb_paths[minimum : maximum + 1]
        depth_paths = all_depth_paths[minimum : maximum + 1]
        if not np.array_equal(frame["index"].to_numpy(dtype=np.int64), np.arange(len(frame))):
            raise ValueError("schema/index: parquet index is not the contiguous sequence 0..T-1")
        if len(frame) != 408:
            report["warnings"].append(f"frame count is {len(frame)}, not smoke-test-specific 408")
        if len(rgb_paths) != len(frame) or len(depth_paths) != len(frame):
            raise ValueError("path/frame-count: RGB, depth, and parquet frame counts differ")

        loader_class = _load_loader_class(_resolve_loader_path(loader_path))
        preload_path = Path(tempfile.gettempdir()) / "navdp_loader_test_preload.json"
        dataset = loader_class(
            str(dataset_root),
            str(preload_path),
            memory_size=8,
            predict_size=24,
            image_size=224,
            scene_data_scale=1.0,
            trajectory_data_scale=1.0,
            preload=False,
        )
        selected_parquet = parquet_path.resolve()
        dataset_index = next(
            (
                index
                for index, candidate in enumerate(dataset.trajectory_data_dir)
                if Path(candidate).resolve() == selected_parquet
            ),
            None,
        )
        if dataset_index is None:
            raise ValueError("loader: selected episode parquet is absent from InternNav dataset index")
        intrinsic, extrinsic, actions, trajectory_length = dataset.process_data_parquet(dataset_index)
        if (
            intrinsic.shape != (3, 3)
            or extrinsic.shape != (4, 4)
            or actions.shape != (len(frame), 4, 4)
            or trajectory_length != len(frame)
        ):
            raise ValueError(
                "schema/shape: NavDP loader returned invalid intrinsic/extrinsic/action shapes"
            )
        sample_indexes = np.sort(
            np.random.default_rng(17).choice(len(frame), SAMPLE_FRAME_COUNT, replace=False)
        )
        for index in sample_indexes:
            rgb = dataset.load_image(str(rgb_paths[int(index)]))
            depth = dataset.load_depth(str(depth_paths[int(index)]))
            if rgb.shape != (480, 848, 3) or rgb.dtype != np.uint8:
                raise ValueError(f"dtype/shape: invalid RGB at frame {index}")
            if depth.shape != (480, 848) or depth.dtype != np.uint16:
                raise ValueError(f"dtype/shape: invalid depth at frame {index}")
            report["sample_frames"].append(int(index))
        # This invokes InternNav's real __getitem__ path, including images,
        # depth, parquet, occupancy PLY, action processing, and collation inputs.
        sample = dataset[dataset_index]
        if len(sample) != 10:
            raise ValueError("loader sample: NavDP __getitem__ did not return ten expected values")

        # Exercise the same process boundary and default tensor collation used
        # by training, while retaining the selected scene rather than global
        # dataset index zero (which may belong to another scene).
        import torch

        selected_scene_indices = [
            index
            for index, candidate in enumerate(dataset.trajectory_data_dir)
            if Path(candidate).resolve().parent == selected_parquet.parent
        ]
        if len(selected_scene_indices) < 2:
            raise ValueError("loader: selected scene has fewer than two indexed episodes")
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, selected_scene_indices[:2]),
            batch_size=2,
            num_workers=0,
        )
        batch = next(iter(loader))
        if len(batch) != 10 or batch[3].shape[0] != 2 or batch[4].shape[0] != 2:
            raise ValueError("loader DataLoader: batch_size=2 collation failed")

        parquet_extrinsics = np.stack(
            [_matrix(value, (4, 4)) for value in frame["observation.camera_extrinsic"]]
        )
        heights = parquet_extrinsics[:, 2, 3]
        if heights.std() > 1e-5:
            raise ValueError("coordinate: camera height is not constant")
        report.update(
            {
                "load_success": True,
                "frame_index": {"min": int(frame["index"].min()), "max": int(frame["index"].max())},
                "image_index": {"min": minimum, "max": maximum},
                "loader_dataset_index": int(dataset_index),
                "camera": {
                    "intrinsic_shape": list(intrinsic.shape),
                    "extrinsic_shape": list(extrinsic.shape),
                    "action_shape": list(actions.shape),
                    "calibration_z_mean": float(heights.mean()),
                    "calibration_z_std": float(heights.std()),
                },
                "dataloader": {
                    "batch_size": 2,
                    "num_workers": 0,
                    "selected_dataset_indices": selected_scene_indices[:2],
                    "getitem_calls": 3,
                    "rgb": {"shape": list(batch[3].shape), "dtype": str(batch[3].dtype)},
                    "depth": {"shape": list(batch[4].shape), "dtype": str(batch[4].dtype)},
                    "intrinsic": {"shape": list(intrinsic.shape), "dtype": str(intrinsic.dtype)},
                    "camera_extrinsic": {
                        "shape": list(extrinsic.shape),
                        "dtype": str(extrinsic.dtype),
                    },
                    "action": {"shape": list(actions.shape), "dtype": str(actions.dtype)},
                },
                "depth_geometry": _depth_geometry_diagnostic(
                    depth_paths,
                    actions,
                    intrinsic,
                    scene_dir / "meta" / "pointcloud.ply",
                    sample_indexes,
                ),
            }
        )
    except Exception as error:
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    _write_report(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a NavDP dataset with InternNav's real loader."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("traj_data"))
    parser.add_argument("--scene", default="robotnav/trajectory_000000")
    parser.add_argument(
        "--loader-path",
        type=Path,
        help=f"Path to InternNav navdp_lerobot_dataset.py (or set {INTERNNAV_LOADER_ENV})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_loader_test(args.dataset_root, args.scene, args.loader_path)
    print(json.dumps(report, ensure_ascii=False))
    if not report["load_success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
