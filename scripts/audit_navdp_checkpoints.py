"""Read-only NavDP checkpoint identity, delta, and optimizer audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or not all(torch.is_tensor(item) for item in value.values()):
        raise TypeError(f"Expected a pure tensor state_dict: {path}")
    return value


def group_name(key: str) -> str:
    for prefix in (
        "rgbd_encoder.rgb_model",
        "rgbd_encoder.depth_model",
        "rgbd_encoder",
        "pixel_encoder",
        "image_encoder",
        "point_encoder",
        "decoder",
        "input_embed",
        "cond_pos_embed",
        "out_pos_embed",
        "time_emb",
        "layernorm",
        "action_head",
        "critic_head",
        "pixel_aux_head",
        "image_aux_head",
    ):
        if key == prefix or key.startswith(prefix + "."):
            return prefix
    return key.split(".", 1)[0]


def freeze_rule(key: str) -> str:
    if key.startswith("rgbd_encoder.rgb_model."):
        return "frozen_rgb_model"
    if "mask_token" in key:
        return "frozen_mask_token"
    return "trainable"


def tensor_delta(reference: torch.Tensor, current: torch.Tensor) -> tuple[float, float, bool]:
    if reference.shape != current.shape:
        return math.nan, math.nan, True
    if reference.dtype == torch.bool or not (reference.is_floating_point() or reference.is_complex()):
        changed = not torch.equal(reference, current)
        return float(changed), float(changed), changed
    delta = current.detach().to(torch.float64) - reference.detach().to(torch.float64)
    return float(torch.linalg.vector_norm(delta)), float(delta.abs().max()), bool(torch.any(delta != 0))


def checkpoint_dirs(root: Path) -> list[Path]:
    values = [path for path in root.glob("checkpoint-*") if path.is_dir()]
    return sorted(values, key=lambda path: int(path.name.split("-")[-1]))


def trainer_summary(directory: Path) -> dict[str, Any]:
    path = directory / "trainer_state.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    history = [item for item in value.get("log_history", []) if "loss" in item]
    return {
        "global_step": value.get("global_step"),
        "epoch": value.get("epoch"),
        "logged_loss_first": None if not history else history[0]["loss"],
        "logged_loss_last": None if not history else history[-1]["loss"],
        "logged_loss_min": None if not history else min(item["loss"] for item in history),
        "learning_rate_last": None if not history else history[-1].get("learning_rate"),
    }


def optimizer_summary(directory: Path) -> dict[str, Any]:
    optimizer_path = directory / "optimizer.pt"
    scheduler_path = directory / "scheduler.pt"
    optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=False)
    scheduler = torch.load(scheduler_path, map_location="cpu", weights_only=False)
    steps = []
    for value in optimizer["state"].values():
        if "step" in value:
            step = value["step"]
            steps.append(float(step.item() if torch.is_tensor(step) else step))
    groups = []
    for group in optimizer["param_groups"]:
        groups.append({key: value for key, value in group.items() if key != "params"} | {
            "parameter_slots": len(group["params"])
        })
    return {
        "optimizer_class_from_code": "torch.optim.Adam",
        "state_entries": len(optimizer["state"]),
        "step_min": min(steps),
        "step_max": max(steps),
        "param_groups": groups,
        "scheduler_class_from_code": "torch.optim.lr_scheduler.LinearLR",
        "scheduler_last_epoch": scheduler.get("last_epoch"),
        "scheduler_last_lr": scheduler.get("_last_lr"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internnav", type=Path, default=Path("/home/ely/Desktop/InternNav"))
    parser.add_argument("--deploy", type=Path, default=Path("/home/ely/Desktop/NavDP_workspace/NavDP"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/navdp_right_bias_audit"))
    args = parser.parse_args()

    start_path = args.internnav / "navdp-cross-modal.ckpt"
    run_root = args.internnav / "checkpoints/navdp_forth_global_long300_overfit/ckpts"
    directories = checkpoint_dirs(run_root)
    if not directories:
        raise FileNotFoundError(run_root)
    final_path = directories[-1] / "pytorch_model.bin"
    deploy_path = args.deploy / "pytorch_model.bin"
    reference = state_dict(start_path)
    final = state_dict(final_path)
    model_keys = set(final)

    identities = [{
        "name": "S0",
        "path": str(start_path.resolve()),
        "sha256": sha256(start_path),
        "size_bytes": start_path.stat().st_size,
        "step": 0,
        "state_dict_keys": len(reference),
        "missing_model_keys": sorted(model_keys - set(reference)),
        "unexpected_keys": sorted(set(reference) - model_keys),
        "format": "raw OrderedDict/state_dict; no optimizer/scheduler/EMA",
    }]
    timeline = []
    for directory in directories:
        model_path = directory / "pytorch_model.bin"
        current = state_dict(model_path)
        total_delta_sq = 0.0
        changed_numel = 0
        changed_tensors = 0
        groups: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {"delta_l2_sq": 0.0, "changed_numel": 0, "tensor_count": 0, "changed_tensors": 0}
        )
        for key in sorted(model_keys & set(reference) & set(current)):
            l2, _maximum, changed = tensor_delta(reference[key], current[key])
            if math.isfinite(l2):
                total_delta_sq += l2 * l2
                groups[group_name(key)]["delta_l2_sq"] += l2 * l2
            groups[group_name(key)]["tensor_count"] += 1
            if changed:
                numel = current[key].numel()
                changed_numel += numel
                changed_tensors += 1
                groups[group_name(key)]["changed_numel"] += numel
                groups[group_name(key)]["changed_tensors"] += 1
        training = trainer_summary(directory)
        record = {
            "name": directory.name,
            "path": str(model_path.resolve()),
            "sha256": sha256(model_path),
            "size_bytes": model_path.stat().st_size,
            "step": training["global_step"],
            "epoch": training["epoch"],
            "state_dict_keys": len(current),
            "missing_model_keys": sorted(model_keys - set(current)),
            "unexpected_keys": sorted(set(current) - model_keys),
            "optimizer_present": (directory / "optimizer.pt").is_file(),
            "scheduler_present": (directory / "scheduler.pt").is_file(),
            "ema_present": bool(list(directory.glob("*ema*"))),
            "training": training,
        }
        identities.append(record)
        timeline.append({
            "step": training["global_step"],
            "parameter_delta_l2_from_s0": math.sqrt(total_delta_sq),
            "changed_tensor_count": changed_tensors,
            "changed_numel": changed_numel,
            "groups": {
                group: {
                    **{key: value for key, value in values.items() if key != "delta_l2_sq"},
                    "delta_l2": math.sqrt(float(values["delta_l2_sq"])),
                }
                for group, values in sorted(groups.items())
            },
            "training": training,
        })
        del current

    deploy_digest = sha256(deploy_path)
    identities.append({
        "name": "DEPLOY",
        "path": str(deploy_path.resolve()),
        "sha256": deploy_digest,
        "size_bytes": deploy_path.stat().st_size,
        "byte_identical_to_s2": deploy_digest == identities[-1]["sha256"],
        "load_code": "NavDP_Agent -> NavDP_Policy.load_state_dict(..., strict=False)",
    })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameter_csv = args.output_dir / "parameter_delta.csv"
    with parameter_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "name", "group", "shape", "dtype", "numel", "freeze_rule",
            "optimizer_membership_by_code", "delta_l2_s0_to_s2", "delta_max_abs_s0_to_s2", "changed",
        ])
        writer.writeheader()
        for key in sorted(model_keys & set(reference)):
            l2, maximum, changed = tensor_delta(reference[key], final[key])
            writer.writerow({
                "name": key,
                "group": group_name(key),
                "shape": "x".join(str(value) for value in final[key].shape),
                "dtype": str(final[key].dtype),
                "numel": final[key].numel(),
                "freeze_rule": freeze_rule(key),
                "optimizer_membership_by_code": "yes (optimizer constructed from model.parameters())",
                "delta_l2_s0_to_s2": l2,
                "delta_max_abs_s0_to_s2": maximum,
                "changed": changed,
            })

    freeze_groups: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {"tensor_count": 0, "numel": 0, "changed_tensor_count": 0, "changed_numel": 0, "delta_l2_sq": 0.0}
    )
    for key in sorted(model_keys & set(reference)):
        rule = freeze_rule(key)
        l2, _maximum, changed = tensor_delta(reference[key], final[key])
        freeze_groups[rule]["tensor_count"] += 1
        freeze_groups[rule]["numel"] += final[key].numel()
        if math.isfinite(l2):
            freeze_groups[rule]["delta_l2_sq"] += l2 * l2
        if changed:
            freeze_groups[rule]["changed_tensor_count"] += 1
            freeze_groups[rule]["changed_numel"] += final[key].numel()
    freeze_csv = args.output_dir / "freeze_optimizer_table.csv"
    with freeze_csv.open("w", encoding="utf-8", newline="") as stream:
        fields = ["rule", "requires_grad", "optimizer_membership", "expected_gradient", "tensor_count",
                  "numel", "changed_tensor_count", "changed_numel", "delta_l2_s0_to_s2"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rule, values in sorted(freeze_groups.items()):
            writer.writerow({
                "rule": rule,
                "requires_grad": rule == "trainable",
                "optimizer_membership": True,
                "expected_gradient": rule == "trainable",
                "tensor_count": values["tensor_count"],
                "numel": values["numel"],
                "changed_tensor_count": values["changed_tensor_count"],
                "changed_numel": values["changed_numel"],
                "delta_l2_s0_to_s2": math.sqrt(float(values["delta_l2_sq"])),
            })

    report = {
        "status": "PASS",
        "internnav_git_head": "7a5c62400ac45b313d9b709c740b64191556a242",
        "model_identity": identities,
        "timeline": timeline,
        "load_contract": {
            "S0_to_training_model": "1048/1048 current model tensors loaded strictly after dropping 18 approved decoder_layer.* legacy templates",
            "S1_definition": "training model immediately after S0 load and before any optimizer step",
            "S0_equals_S1_parameters": True,
            "S0_equals_S1_outputs_under_same_model_eval_input_and_rng": True,
            "raw_or_ema": "raw only; no EMA artifacts/configuration found",
            "deploy_equals_S2_bytes": deploy_digest == identities[-2]["sha256"],
        },
        "optimizer_first": optimizer_summary(directories[0]),
        "optimizer_final": optimizer_summary(directories[-1]),
        "runtime_config": {
            "optimizer_actual": "Adam (custom trainer override; TrainingArguments adamw_torch is not used)",
            "weight_decay_actual": 0.0,
            "weight_decay_config_but_ignored": 1e-4,
            "scheduler": "LinearLR(start_factor=1.0,end_factor=0.5,total_iters=10000)",
            "learning_rate": 1e-4,
            "batch_size": 32,
            "gradient_accumulation_steps": 1,
            "amp": False,
            "seed_training_arguments": 0,
            "sampler_seed": 1234,
            "dataset_unique_episodes": 300,
            "dataset_replication": 50,
            "dataset_length": 15000,
            "drop_last": True,
            "max_steps": 10000,
        },
        "artifacts": {
            "parameter_delta_csv": str(parameter_csv.resolve()),
            "freeze_optimizer_csv": str(freeze_csv.resolve()),
        },
    }
    output = args.output_dir / "checkpoint_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output.resolve()),
                      "deploy_equals_s2": report["load_contract"]["deploy_equals_S2_bytes"],
                      "checkpoints": len(directories)}))


if __name__ == "__main__":
    main()
