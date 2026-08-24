"""Read-only coordinate and label audit for the NavDP long300 training set.

This script deliberately re-derives the quantities used by InternNav's
``NavDP_Base_Datset`` from parquet poses.  It does not import the Dataset, load
images, mutate the index, or touch the training repository.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def _matrices(values: list[Any]) -> np.ndarray:
    return np.stack(
        [np.asarray(value.tolist() if hasattr(value, "tolist") else value, dtype=np.float64).reshape(4, 4)
         for value in values],
        axis=0,
    )


def _summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "q01": float(np.quantile(values, 0.01)),
        "q10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.90)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "negative_fraction": float(np.mean(values < -1e-6)),
        "near_zero_fraction": float(np.mean(np.abs(values) <= 1e-6)),
        "positive_fraction": float(np.mean(values > 1e-6)),
    }


def _side_summary(lateral: np.ndarray, threshold_m: float) -> dict[str, Any]:
    lateral = np.asarray(lateral, dtype=np.float64)
    left = lateral > threshold_m
    right = lateral < -threshold_m
    center = ~(left | right)
    return {
        "threshold_m": threshold_m,
        "left_count": int(left.sum()),
        "center_count": int(center.sum()),
        "right_count": int(right.sum()),
        "left_fraction": float(left.mean()),
        "center_fraction": float(center.mean()),
        "right_fraction": float(right.mean()),
        "signed_mean_m": float(lateral.mean()),
        "left_magnitude_sum_m": float(lateral[left].sum()),
        "right_magnitude_sum_m": float(-lateral[right].sum()),
    }


def _scene_name(path: Path) -> str:
    return path.parents[2].name


def _unique_parquets(index_path: Path) -> list[Path]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    values = index.get("trajectory_data_dir")
    if not isinstance(values, list):
        raise ValueError(f"trajectory_data_dir is absent from {index_path}")
    unique = list(dict.fromkeys(str(value) for value in values))
    paths = [Path(value).resolve() for value in unique]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing parquet files (first 5): {missing[:5]}")
    return paths


def _local_displacement(
    world_from_camera: np.ndarray,
    base_from_camera: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
) -> np.ndarray:
    """Return displacement in the current Go2 base frame: [forward,left,up]."""
    world_from_base_rotation = (
        world_from_camera[starts, :3, :3] @ np.linalg.inv(base_from_camera[starts, :3, :3])
    )
    world_delta = world_from_camera[stops, :3, 3] - world_from_camera[starts, :3, 3]
    return np.einsum("nji,nj->ni", world_from_base_rotation, world_delta)


def _append(store: dict[str, list[np.ndarray]], key: str, values: np.ndarray) -> None:
    store[key].append(np.asarray(values))


def _finalize(store: dict[str, list[np.ndarray]], threshold_m: float) -> dict[str, Any]:
    values = {key: np.concatenate(parts) for key, parts in store.items()}
    canonical_forward = values["canonical_forward"]
    canonical_left = values["canonical_left"]
    dataset_dim0 = values["dataset_dim0"]
    dataset_dim1 = values["dataset_dim1"]
    endpoint_forward = values["endpoint_forward"]
    endpoint_left = values["endpoint_left"]
    endpoint_dataset_dim0 = values["endpoint_dataset_dim0"]
    endpoint_dataset_dim1 = values["endpoint_dataset_dim1"]
    heading = np.arctan2(canonical_left, canonical_forward)
    distance = np.hypot(canonical_forward, canonical_left)
    return {
        "stride4_displacements_m": {
            "canonical_forward": _summary(canonical_forward),
            "canonical_left": _summary(canonical_left),
            "dataset_dim0": _summary(dataset_dim0),
            "dataset_dim1": _summary(dataset_dim1),
            "canonical_left_center_right": _side_summary(canonical_left, threshold_m),
            "deployment_interpretation_left_center_right": _side_summary(dataset_dim1, threshold_m),
            "heading_rad": _summary(heading),
            "distance_m": _summary(distance),
            "exact_mapping_max_abs_error": {
                "dataset_dim0_minus_canonical_left": float(np.max(np.abs(dataset_dim0 - canonical_left))),
                "dataset_dim1_plus_canonical_forward": float(np.max(np.abs(dataset_dim1 + canonical_forward))),
            },
            "forward_motion_mapped_to_deployment_right_fraction": float(
                np.mean((canonical_forward > threshold_m) & (dataset_dim1 < -threshold_m))
            ),
        },
        "sampled_point_goals_m": {
            "canonical_forward": _summary(endpoint_forward),
            "canonical_left": _summary(endpoint_left),
            "dataset_dim0": _summary(endpoint_dataset_dim0),
            "dataset_dim1": _summary(endpoint_dataset_dim1),
            "canonical_left_center_right": _side_summary(endpoint_left, threshold_m),
            "deployment_interpretation_left_center_right": _side_summary(
                endpoint_dataset_dim1, threshold_m
            ),
            "canonical_forward_mapped_to_dataset_dim1_negative_fraction": float(
                np.mean((endpoint_forward > threshold_m) & (endpoint_dataset_dim1 < -threshold_m))
            ),
        },
    }


def audit(
    index_path: Path | None,
    parquet_root: Path | None,
    *,
    stride: int,
    samples_per_episode: int,
    seed: int,
) -> dict[str, Any]:
    if parquet_root is not None:
        parquets = sorted(parquet_root.rglob("*.parquet"))
        if not parquets:
            raise ValueError(f"No parquet files below {parquet_root}")
    elif index_path is not None:
        parquets = _unique_parquets(index_path)
    else:
        raise ValueError("Either index_path or parquet_root is required")
    all_store: dict[str, list[np.ndarray]] = defaultdict(list)
    scene_stores: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    scene_rows: dict[str, int] = defaultdict(int)
    rng = np.random.default_rng(seed)

    for parquet in parquets:
        frame = pq.read_table(
            parquet,
            columns=["action", "observation.camera_extrinsic"],
        ).to_pandas()
        action = _matrices(frame["action"].tolist())
        base_from_camera = _matrices(frame["observation.camera_extrinsic"].tolist())
        if len(action) <= stride + 2:
            raise ValueError(f"Episode is too short for stride {stride}: {parquet}")
        starts = np.arange(0, len(action) - stride, dtype=np.int64)
        stops = starts + stride
        canonical = _local_displacement(action, base_from_camera, starts, stops)
        # This is the exact final adapter in InternNav.relative_pose.
        dataset_xy = np.column_stack((canonical[:, 1], -canonical[:, 0]))

        pixel_starts = rng.integers(0, len(action) // 2, size=samples_per_episode)
        targets = np.array(
            [rng.integers(start + 1, len(action) - 1) for start in pixel_starts], dtype=np.int64
        )
        memory_starts = np.array(
            [rng.integers(start, target) for start, target in zip(pixel_starts, targets, strict=True)],
            dtype=np.int64,
        )
        endpoints = _local_displacement(action, base_from_camera, memory_starts, targets)
        endpoint_dataset_xy = np.column_stack((endpoints[:, 1], -endpoints[:, 0]))

        scene = _scene_name(parquet)
        scene_rows[scene] += len(action)
        for store in (all_store, scene_stores[scene]):
            _append(store, "canonical_forward", canonical[:, 0])
            _append(store, "canonical_left", canonical[:, 1])
            _append(store, "dataset_dim0", dataset_xy[:, 0])
            _append(store, "dataset_dim1", dataset_xy[:, 1])
            _append(store, "endpoint_forward", endpoints[:, 0])
            _append(store, "endpoint_left", endpoints[:, 1])
            _append(store, "endpoint_dataset_dim0", endpoint_dataset_xy[:, 0])
            _append(store, "endpoint_dataset_dim1", endpoint_dataset_xy[:, 1])

    threshold_m = 0.05
    return {
        "status": "PASS",
        "index_path": None if index_path is None else str(index_path.resolve()),
        "parquet_root": None if parquet_root is None else str(parquet_root.resolve()),
        "unique_episode_count": len(parquets),
        "scene_count": len(scene_stores),
        "parquet_rows": int(sum(scene_rows.values())),
        "stride_frames": stride,
        "point_goal_samples_per_episode": samples_per_episode,
        "seed": seed,
        "coordinate_contract": {
            "canonical": "[forward,left,up] from R_world_base.T @ (future_camera_origin-current_camera_origin)",
            "internnav_relative_pose_adapter": "[canonical_left,-canonical_forward,canonical_up]",
            "deployment": "trajectory[:,0]=forward, trajectory[:,1]=left; negative left is right",
        },
        "overall": _finalize(all_store, threshold_m),
        "scenes": {
            scene: {"episodes": sum(_scene_name(path) == scene for path in parquets),
                    "rows": scene_rows[scene],
                    **_finalize(store, threshold_m)}
            for scene, store in sorted(scene_stores.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("/home/ely/Desktop/InternNav/data/datasets/navdp_long300_union_index.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/navdp_right_bias_audit/data_statistics.json"),
    )
    parser.add_argument("--parquet-root", type=Path)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--samples-per-episode", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    result = audit(
        None if args.parquet_root is not None else args.index,
        args.parquet_root,
        stride=args.stride,
        samples_per_episode=args.samples_per_episode,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output.resolve()),
        "episodes": result["unique_episode_count"],
        "overall": result["overall"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
