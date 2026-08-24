"""Assemble a deterministic, geometry-unique trajectory batch without changing coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


def _json_sha(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("contract_version") != 2:
        raise ValueError(f"Unsupported manifest contract: {path}")
    if value.get("trajectory_count") != len(value.get("trajectories", [])):
        raise ValueError(f"Manifest count mismatch: {path}")
    return value


def _record(manifest_path: Path, batch_seed: int, entry: dict[str, Any]) -> dict[str, Any]:
    route_path = manifest_path.parent / entry["path"]
    if _file_sha(route_path) != entry["trajectory_sha256"]:
        raise ValueError(f"Route SHA mismatch: {route_path}")
    route = json.loads(route_path.read_text(encoding="utf-8"))
    if route["trajectory_id"] != entry["trajectory_id"]:
        raise ValueError(f"Route ID mismatch: {route_path}")
    points = route["smooth_path_xz"]
    if len(points) < 2 or any(
        len(point) != 2 or not all(math.isfinite(float(value)) for value in point)
        for point in points
    ):
        raise ValueError(f"Invalid smooth_path_xz: {route_path}")
    quantized = [[round(float(x), 8), round(float(z), 8)] for x, z in points]
    if not 5.0 <= float(entry["path_length_m"]) <= 50.0:
        raise ValueError(f"Out-of-range trajectory: {route_path}")
    if bool(entry["smooth_path_collides"]):
        raise ValueError(f"Colliding trajectory: {route_path}")
    if int(entry["point_count"]) != len(points):
        raise ValueError(f"Point count mismatch: {route_path}")
    return {
        "manifest_path": manifest_path.resolve(),
        "manifest_sha256": _file_sha(manifest_path),
        "batch_seed": batch_seed,
        "entry": entry,
        "route": route,
        "route_path": route_path,
        "debug_path": manifest_path.parent / entry["debug_image"],
        "raw_hash": _json_sha(points),
        "quantized_hash": _json_sha(quantized),
    }


def _records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _load_manifest(path)
    return manifest, [
        _record(path, int(manifest["batch_seed"]), entry)
        for entry in manifest["trajectories"]
    ]


def _assert_hash_decision(raw_seen: set[str], quantized_seen: set[str], record: dict[str, Any]) -> bool:
    raw_match = record["raw_hash"] in raw_seen
    quantized_match = record["quantized_hash"] in quantized_seen
    if raw_match != quantized_match:
        raise ValueError(
            "Raw and quantized geometry hashes disagree for "
            f"{record['manifest_path']}:{record['entry']['trajectory_id']}"
        )
    return raw_match


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", action="append", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=Path, default=[])
    parser.add_argument("--scene-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=100)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite assembled output: {args.output_dir}")

    history_raw: set[str] = set()
    history_quantized: set[str] = set()
    history_summary = []
    scene_hashes: set[str] = set()
    for path in args.history:
        manifest, records = _records(path)
        scene_hashes.add(manifest["source_scene_model_sha256"])
        history_summary.append(
            {"manifest": str(path.resolve()), "sha256": _file_sha(path), "count": len(records)}
        )
        history_raw.update(record["raw_hash"] for record in records)
        history_quantized.update(record["quantized_hash"] for record in records)

    base_manifest, base_records = _records(args.base)
    scene_hashes.add(base_manifest["source_scene_model_sha256"])
    candidate_batches = []
    for path in args.candidate:
        manifest, records = _records(path)
        scene_hashes.add(manifest["source_scene_model_sha256"])
        candidate_batches.append((path, manifest, records))
    if len(scene_hashes) != 1 or _file_sha(args.scene_model) not in scene_hashes:
        raise ValueError("All trajectory batches and the final scene model must have the same SHA-256")

    selected: list[dict[str, Any]] = []
    selected_raw: set[str] = set()
    selected_quantized: set[str] = set()
    rejected = Counter()

    def consider(record: dict[str, Any], source_class: str) -> None:
        if len(selected) >= args.target_count:
            return
        if _assert_hash_decision(history_raw, history_quantized, record):
            rejected[f"{source_class}_history_duplicate"] += 1
            return
        if _assert_hash_decision(selected_raw, selected_quantized, record):
            rejected[f"{source_class}_selected_duplicate"] += 1
            return
        selected.append({**record, "source_class": source_class})
        selected_raw.add(record["raw_hash"])
        selected_quantized.add(record["quantized_hash"])

    for record in base_records:
        consider(record, "seed19_base")
    for _, _, records in candidate_batches:
        for record in records:
            consider(record, "supplement")
    if len(selected) != args.target_count:
        raise RuntimeError(f"Only selected {len(selected)} unique trajectories; need {args.target_count}")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}-staging-", dir=args.output_dir.parent))
    try:
        routes_dir = staging / "routes"
        routes_dir.mkdir()
        final_entries = []
        provenance_entries = []
        for index, record in enumerate(selected):
            new_id = f"auto_{index:03d}"
            route = json.loads(json.dumps(record["route"]))
            original_points = record["route"]["smooth_path_xz"]
            route["trajectory_id"] = new_id
            if route["smooth_path_xz"] != original_points or route["robot_ground_pose"]["path_xz"] != original_points:
                raise AssertionError("Assembly changed trajectory coordinates")
            route_path = routes_dir / f"{new_id}.json"
            route_path.write_text(json.dumps(route, indent=2), encoding="utf-8")
            debug_path = routes_dir / f"{new_id}_debug.png"
            shutil.copy2(record["debug_path"], debug_path)
            source = record["entry"]
            final_entries.append(
                {
                    **source,
                    "trajectory_id": new_id,
                    "path": f"routes/{new_id}.json",
                    "trajectory_sha256": _file_sha(route_path),
                    "debug_image": f"routes/{new_id}_debug.png",
                }
            )
            provenance_entries.append(
                {
                    "trajectory_id": new_id,
                    "source_class": record["source_class"],
                    "source_seed": int(record["batch_seed"]),
                    "source_task_seed": int(source["seed"]),
                    "source_trajectory_id": source["trajectory_id"],
                    "source_manifest": str(record["manifest_path"]),
                    "source_manifest_sha256": record["manifest_sha256"],
                    "source_route_sha256": source["trajectory_sha256"],
                    "raw_geometry_sha256": record["raw_hash"],
                    "quantized_8dp_geometry_sha256": record["quantized_hash"],
                }
            )

        lengths = [float(entry["path_length_m"]) for entry in final_entries]
        manifest = {
            "contract_version": 2,
            "requested_count": args.target_count,
            "trajectory_count": len(final_entries),
            "batch_seed": -1,
            "source_scene_model": "../navigation_scene/scene_model.npz",
            "source_scene_model_sha256": next(iter(scene_hashes)),
            "planner_config": base_manifest["planner_config"],
            "trajectory_sampling": base_manifest["trajectory_sampling"],
            "sampling_statistics": {
                "assembly": True,
                "considered": len(base_records) + sum(len(records) for _, _, records in candidate_batches),
                "selected": len(final_entries),
                "rejections": dict(rejected),
            },
            "length_statistics": {
                "min_m": min(lengths),
                "max_m": max(lengths),
                "mean_m": sum(lengths) / len(lengths),
                "std_m": math.sqrt(sum((value - sum(lengths) / len(lengths)) ** 2 for value in lengths) / len(lengths)),
            },
            "assembly": {
                "strategy": "stable base-first selection with strict raw and quantized geometry exclusion",
                "mixed_seed_batch": True,
                "provenance": "batch3_trajectory_provenance.json",
                "duplicate_report": "batch3_trajectory_duplicate_report.json",
            },
            "trajectories": final_entries,
        }
        (staging / "trajectory_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        provenance = {
            "status": "PASS",
            "history": history_summary,
            "base_manifest": str(args.base.resolve()),
            "candidate_manifests": [str(path.resolve()) for path, _, _ in candidate_batches],
            "selection_order": "base manifest order, then candidate argument order and manifest order",
            "target_count": args.target_count,
            "source_seed_counts": dict(sorted(Counter(item["source_seed"] for item in provenance_entries).items())),
            "entries": provenance_entries,
        }
        (staging / "batch3_trajectory_provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )
        report = {
            "status": "PASS",
            "trajectory_count": len(final_entries),
            "ids_contiguous": [entry["trajectory_id"] for entry in final_entries]
            == [f"auto_{index:03d}" for index in range(args.target_count)],
            "internal_raw_duplicates": len(final_entries) - len(selected_raw),
            "internal_quantized_8dp_duplicates": len(final_entries) - len(selected_quantized),
            "history_raw_duplicates": sum(record["raw_hash"] in history_raw for record in selected),
            "history_quantized_8dp_duplicates": sum(
                record["quantized_hash"] in history_quantized for record in selected
            ),
            "source_seed_counts": provenance["source_seed_counts"],
            "rejections": dict(rejected),
            "length_statistics": manifest["length_statistics"],
            "waypoint_statistics": {
                "min": min(int(entry["point_count"]) for entry in final_entries),
                "max": max(int(entry["point_count"]) for entry in final_entries),
                "mean": sum(int(entry["point_count"]) for entry in final_entries) / len(final_entries),
                "total": sum(int(entry["point_count"]) for entry in final_entries),
            },
        }
        (staging / "batch3_trajectory_duplicate_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        staging.rename(args.output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
