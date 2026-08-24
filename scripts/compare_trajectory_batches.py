"""Compare trajectory geometry within and across generated batches."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _sha(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_batch(label: str, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for entry in manifest["trajectories"]:
        route = json.loads((manifest_path.parent / entry["path"]).read_text(encoding="utf-8"))
        points = route["smooth_path_xz"]
        quantized = [[round(float(x), 8), round(float(z), 8)] for x, z in points]
        records.append(
            {
                "trajectory_id": entry["trajectory_id"],
                "start_xz": entry["start_xz"],
                "goal_xz": entry["goal_xz"],
                "path_length_m": entry["path_length_m"],
                "point_count": entry["point_count"],
                "raw_geometry_sha256": _sha(points),
                "quantized_8dp_geometry_sha256": _sha(quantized),
            }
        )
    lengths = [float(record["path_length_m"]) for record in records]
    return {
        "label": label,
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "batch_seed": manifest["batch_seed"],
        "trajectory_count": len(records),
        "length_m": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="append", nargs=2, metavar=("LABEL", "MANIFEST"), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    batches = {
        label: _load_batch(label, Path(manifest)) for label, manifest in args.batch
    }
    target = batches[args.target]
    target_raw = Counter(record["raw_geometry_sha256"] for record in target["records"])
    target_quantized = Counter(
        record["quantized_8dp_geometry_sha256"] for record in target["records"]
    )

    comparisons = []
    for label, batch in batches.items():
        if label == args.target:
            continue
        raw_index: dict[str, list[str]] = defaultdict(list)
        quantized_index: dict[str, list[str]] = defaultdict(list)
        for record in batch["records"]:
            raw_index[record["raw_geometry_sha256"]].append(record["trajectory_id"])
            quantized_index[record["quantized_8dp_geometry_sha256"]].append(record["trajectory_id"])
        raw_matches = []
        quantized_matches = []
        for record in target["records"]:
            for old_id in raw_index[record["raw_geometry_sha256"]]:
                raw_matches.append([record["trajectory_id"], old_id])
            for old_id in quantized_index[record["quantized_8dp_geometry_sha256"]]:
                quantized_matches.append([record["trajectory_id"], old_id])
        comparisons.append(
            {
                "target": args.target,
                "reference": label,
                "raw_exact_match_count": len(raw_matches),
                "raw_exact_matches": raw_matches,
                "quantized_8dp_exact_match_count": len(quantized_matches),
                "quantized_8dp_exact_matches": quantized_matches,
            }
        )

    summary = {
        "status": "FAIL_EXACT_DUPLICATES" if any(
            item["quantized_8dp_exact_match_count"] for item in comparisons
        ) else "PASS",
        "target": args.target,
        "batch_summary": {
            label: {key: value for key, value in batch.items() if key != "records"}
            for label, batch in batches.items()
        },
        "target_internal_duplicates": {
            "raw": sorted(key for key, count in target_raw.items() if count > 1),
            "quantized_8dp": sorted(key for key, count in target_quantized.items() if count > 1),
        },
        "cross_batch_comparisons": comparisons,
    }
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
