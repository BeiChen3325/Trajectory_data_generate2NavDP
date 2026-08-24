#!/usr/bin/env python3
"""Sample distributed NavDP episodes through InternNav's real DataLoader path."""

from __future__ import annotations

import argparse
import json
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


class _JsonLinesReader:
    def __init__(self, path: str, mode: str) -> None:
        self._stream = open(path, mode, encoding="utf-8")

    def __enter__(self):
        return map(json.loads, self._stream)

    def __exit__(self, *_args) -> None:
        self._stream.close()


try:
    import jsonlines  # noqa: F401
except ModuleNotFoundError:
    sys.modules["jsonlines"] = types.SimpleNamespace(
        open=lambda path, mode: _JsonLinesReader(path, mode)
    )

from internnav.dataset.navdp_lerobot_dataset import (  # noqa: E402
    NavDP_Base_Datset,
    navdp_collate_fn,
)


def _finite(value: object) -> bool:
    return not torch.is_tensor(value) or bool(torch.isfinite(value).all())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--sample-episodes", type=int, default=20)
    args = parser.parse_args()

    np.random.seed(17)
    torch.manual_seed(17)
    preload_path = Path("/tmp/navdp_long100_loader_index.json")
    dataset = NavDP_Base_Datset(
        str(args.dataset_root),
        str(preload_path),
        memory_size=8,
        predict_size=24,
        batch_size=4,
        image_size=224,
        scene_data_scale=1.0,
        trajectory_data_scale=1.0,
        pixel_channel=4,
        preload=False,
        random_digit=False,
        prior_sample=False,
    )
    unique_episode_count = len(dataset.trajectory_data_dir) // 50
    if unique_episode_count != args.expected_episodes:
        raise AssertionError(
            f"Expected {args.expected_episodes} unique episodes, got {unique_episode_count}"
        )
    sample_count = min(args.sample_episodes, unique_episode_count)
    selected = np.linspace(0, unique_episode_count - 1, sample_count, dtype=np.int64).tolist()

    pixel_flags: list[float] = []
    batch_shapes: list[list[list[int]]] = []
    caught: list[dict[str, str]] = []
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        loader = DataLoader(
            Subset(dataset, selected),
            batch_size=4,
            num_workers=0,
        )
        for batch in loader:
            if len(batch) != 10 or not all(_finite(value) for value in batch):
                raise AssertionError("Default DataLoader returned invalid or non-finite tensors")
            pixel_flags.extend(float(value) for value in batch[9].tolist())
            batch_shapes.append([list(value.shape) for value in batch if torch.is_tensor(value)])

        # Exercise the training collate function as a separate real DataLoader batch.
        training_batch = next(
            iter(
                DataLoader(
                    Subset(dataset, selected[:4]),
                    batch_size=4,
                    num_workers=0,
                    collate_fn=navdp_collate_fn,
                )
            )
        )
        if not training_batch or not all(_finite(value) for value in training_batch.values()):
            raise AssertionError("Training collate returned empty or non-finite tensors")

        workers8_batches = 0
        workers8_loader = DataLoader(
            Subset(dataset, selected),
            batch_size=min(4, sample_count),
            num_workers=8,
            collate_fn=navdp_collate_fn,
        )
        for workers8_batch in workers8_loader:
            if not workers8_batch or not all(
                _finite(value) for value in workers8_batch.values()
            ):
                raise AssertionError("workers=8 training collate returned invalid tensors")
            workers8_batches += 1
        caught = [
            {"category": item.category.__name__, "message": str(item.message)} for item in records
        ]

    runtime_warnings = [item for item in caught if item["category"] == "RuntimeWarning"]
    report = {
        "status": "PASS",
        "dataset_root": str(args.dataset_root),
        "dataset_length_with_replication": len(dataset),
        "unique_episode_count": unique_episode_count,
        "selected_episode_indices": selected,
        "sampled_episode_count": len(pixel_flags),
        "dataloader_batch_count": len(batch_shapes),
        "dataloader_workers0": "PASS",
        "dataloader_workers8": "PASS",
        "dataloader_workers8_batch_count": workers8_batches,
        "pixel_flag_valid_count": int(sum(flag == 1.0 for flag in pixel_flags)),
        "pixel_flag_total_count": len(pixel_flags),
        "pixel_flag_valid_ratio": float(np.mean(pixel_flags)),
        "runtime_warnings": runtime_warnings,
        "all_warnings": caught,
        "training_collate_keys": sorted(training_batch),
        "training_collate_shapes": {
            key: list(value.shape) for key, value in training_batch.items()
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
