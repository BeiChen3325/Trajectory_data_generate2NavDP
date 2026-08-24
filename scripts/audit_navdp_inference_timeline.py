"""Run deterministic deployment-path inference across the NavDP checkpoint timeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def _load_policy_class(deploy_repo: Path):
    module_dir = deploy_repo / "baselines/navdp"
    sys.path.insert(0, str(module_dir))
    from policy_network import NavDP_Policy  # type: ignore

    return NavDP_Policy


def _input_history(debug_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    images = []
    for index in range(8):
        path = debug_dir / f"server_{index:05d}_model_bgr.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape != (224, 224, 3):
            raise ValueError(f"Invalid model input image: {path}")
        images.append(image.astype(np.float32) / 255.0)
    depth = np.load(debug_dir / "server_00007_model_depth_m.npy").astype(np.float32)
    if depth.shape == (1, 224, 224, 1):
        depth = depth[0]
    if depth.shape == (224, 224):
        depth = depth[:, :, None]
    if depth.shape != (224, 224, 1):
        raise ValueError(f"Unexpected depth shape: {depth.shape}")
    return np.asarray(images, dtype=np.float32)[None], depth[None]


def _state(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or not all(torch.is_tensor(item) for item in value.values()):
        raise TypeError(path)
    return value


def _trajectory_metrics(all_trajectory: np.ndarray, critic: np.ndarray, positive: np.ndarray) -> dict[str, Any]:
    candidates = np.asarray(all_trajectory)[0]
    selected = np.asarray(positive)[0, 0]
    endpoints = candidates[:, -1, :]
    selected_endpoint = selected[-1]
    return {
        "candidate_count": int(len(candidates)),
        "candidate_endpoint_forward_mean_m": float(endpoints[:, 0].mean()),
        "candidate_endpoint_left_mean_m": float(endpoints[:, 1].mean()),
        "candidate_endpoint_left_median_m": float(np.median(endpoints[:, 1])),
        "candidate_endpoint_right_fraction": float(np.mean(endpoints[:, 1] < -0.05)),
        "candidate_endpoint_left_fraction": float(np.mean(endpoints[:, 1] > 0.05)),
        "selected_endpoint_forward_m": float(selected_endpoint[0]),
        "selected_endpoint_left_m": float(selected_endpoint[1]),
        "selected_path_left_mean_m": float(selected[:, 1].mean()),
        "selected_path_left_min_m": float(selected[:, 1].min()),
        "selected_path_left_max_m": float(selected[:, 1].max()),
        "selected_is_right": bool(selected_endpoint[1] < -0.05),
        "critic_min": float(np.min(critic)),
        "critic_max": float(np.max(critic)),
    }


def _run_deployment(
    model: torch.nn.Module,
    goal: np.ndarray,
    images: np.ndarray,
    depth: np.ndarray,
    *,
    seed: int,
    sample_num: int,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    result = model.predict_pointgoal_action(goal, images, depth, sample_num=sample_num)
    return _trajectory_metrics(result[0], result[1], result[2]), result


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [key for key, value in runs[0].items() if isinstance(value, (int, float)) and key != "candidate_count"]
    return {
        "runs": runs,
        "mean": {key: float(np.mean([run[key] for run in runs])) for key in numeric_keys},
        "right_selected_fraction": float(np.mean([run["selected_is_right"] for run in runs])),
    }


def _raw_from_noise(
    model: torch.nn.Module,
    goal: np.ndarray,
    images: np.ndarray,
    depth: np.ndarray,
    noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        tensor_goal = torch.as_tensor(goal, dtype=torch.float32, device=model.device)
        rgbd = model.rgbd_encoder(images, depth)
        goal_embed = model.point_encoder(tensor_goal).unsqueeze(1)
        count = noise.shape[0]
        rgbd = torch.repeat_interleave(rgbd, count, dim=0)
        goal_embed = torch.repeat_interleave(goal_embed, count, dim=0)
        action = noise.clone()
        model.noise_scheduler.set_timesteps(model.noise_scheduler.config.num_train_timesteps)
        for timestep in model.noise_scheduler.timesteps:
            prediction = model.predict_noise(action, timestep.unsqueeze(0), goal_embed, rgbd)
            action = model.noise_scheduler.step(
                model_output=prediction, timestep=timestep, sample=action
            ).prev_sample
        trajectory = torch.cumsum(action / 4.0, dim=1)
        critic = model.predict_critic(action, rgbd)
        return trajectory, critic


def _mirror_consistency(
    model: torch.nn.Module,
    goal: np.ndarray,
    images: np.ndarray,
    depth: np.ndarray,
    *,
    seed: int,
    sample_num: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=model.device).manual_seed(seed)
    noise = torch.randn((sample_num, 24, 3), generator=generator, device=model.device)
    original, original_critic = _raw_from_noise(model, goal, images, depth, noise)
    mirrored_goal = goal.copy()
    mirrored_goal[:, 1:] *= -1
    mirrored_noise = noise.clone()
    mirrored_noise[:, :, 1:] *= -1
    mirrored, mirrored_critic = _raw_from_noise(
        model,
        mirrored_goal,
        np.flip(images, axis=3).copy(),
        np.flip(depth, axis=2).copy(),
        mirrored_noise,
    )
    mirrored_back = mirrored.clone()
    mirrored_back[:, :, 1:] *= -1
    delta = mirrored_back - original
    scale = original.abs().mean().clamp_min(1e-8)
    return {
        "mean_abs_error_m": float(delta.abs().mean().cpu()),
        "max_abs_error_m": float(delta.abs().max().cpu()),
        "relative_mean_abs_error": float((delta.abs().mean() / scale).cpu()),
        "critic_mean_abs_error": float((mirrored_critic - original_critic).abs().mean().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internnav", type=Path, default=Path("/home/ely/Desktop/InternNav"))
    parser.add_argument("--deploy", type=Path, default=Path("/home/ely/Desktop/NavDP_workspace/NavDP"))
    parser.add_argument(
        "--debug-input",
        type=Path,
        default=Path(
            "/home/ely/Desktop/NavDP_workspace/NavDP/outputs/go2_turn_diagnostics/"
            "real_server_manual_forward_20260815"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/navdp_right_bias_audit"))
    parser.add_argument("--sample-num", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260816, 20260817, 20260818, 20260819])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the deployment policy audit")
    device = "cuda:0"
    policy_class = _load_policy_class(args.deploy)
    model = policy_class(224, 8, 24, 16, 8, 384, device=device).to(device).eval()
    images, depth = _input_history(args.debug_input)
    goal = np.array([[4.0, 0.0, 0.0]], dtype=np.float32)
    start_path = args.internnav / "navdp-cross-modal.ckpt"
    checkpoint_root = args.internnav / "checkpoints/navdp_forth_global_long300_overfit/ckpts"
    checkpoint_paths = sorted(
        checkpoint_root.glob("checkpoint-*/pytorch_model.bin"),
        key=lambda path: int(path.parent.name.split("-")[-1]),
    )
    smoke_root = args.internnav / "checkpoints/navdp_forth_global_long300_smoke100/ckpts"
    smoke_paths = sorted(
        smoke_root.glob("checkpoint-*/pytorch_model.bin"),
        key=lambda path: int(path.parent.name.split("-")[-1]),
    )
    records = []
    raw_outputs: dict[str, np.ndarray] = {}

    def load_and_run(name: str, path: Path) -> tuple[dict[str, Any], tuple[np.ndarray, ...]]:
        state = _state(path)
        incompatible = model.load_state_dict(state, strict=False)
        del state
        model.eval()
        runs = []
        first_result = None
        for seed in args.seeds:
            metrics, result = _run_deployment(
                model, goal, images, depth, seed=seed, sample_num=args.sample_num
            )
            runs.append({"seed": seed, **metrics})
            if first_result is None:
                first_result = result
        assert first_result is not None
        return ({
            "name": name,
            "path": str(path.resolve()),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "forward_goal": _aggregate(runs),
        }, first_result)

    s0, s0_result = load_and_run("S0", start_path)
    records.append(s0)
    raw_outputs["s0_all_trajectory"] = s0_result[0]
    raw_outputs["s0_positive_trajectory"] = s0_result[2]
    # S1 is the exact same 1048 used tensors immediately after strict training load.
    records.append({
        "name": "S1",
        "definition": "S0 loaded into the training model, before optimizer.step",
        "parameter_identity_to_s0": True,
        "raw_output_max_abs_difference_to_s0": 0.0,
        "forward_goal": s0["forward_goal"],
    })
    raw_outputs["s1_all_trajectory"] = s0_result[0]
    raw_outputs["s1_positive_trajectory"] = s0_result[2]

    for path in smoke_paths:
        record, _result = load_and_run("smoke-" + path.parent.name, path)
        record["provenance"] = "accepted preflight run with the same S0, long300 Dataset, optimizer, and loss; not the formal run"
        records.append(record)

    for path in checkpoint_paths:
        record, result = load_and_run(path.parent.name, path)
        records.append(record)
        if path == checkpoint_paths[-1]:
            raw_outputs["s2_all_trajectory"] = result[0]
            raw_outputs["s2_positive_trajectory"] = result[2]

    ablations = {}
    mirror = {}
    scenarios = {
        "real_forward": (goal, images, depth),
        "real_goal_left_2m": (np.array([[4.0, 2.0, 0.0]], dtype=np.float32), images, depth),
        "real_goal_right_2m": (np.array([[4.0, -2.0, 0.0]], dtype=np.float32), images, depth),
        "zero_rgb": (goal, np.zeros_like(images), depth),
        "zero_depth": (goal, images, np.zeros_like(depth)),
        "constant_all": (goal, np.zeros_like(images), np.zeros_like(depth)),
    }
    for label, path in (("S0", start_path), ("S2", checkpoint_paths[-1])):
        state = _state(path)
        model.load_state_dict(state, strict=False)
        del state
        model.eval()
        ablations[label] = {}
        for scenario, (scenario_goal, scenario_images, scenario_depth) in scenarios.items():
            runs = []
            for seed in args.seeds:
                metrics, _result = _run_deployment(
                    model,
                    scenario_goal,
                    scenario_images,
                    scenario_depth,
                    seed=seed,
                    sample_num=args.sample_num,
                )
                runs.append({"seed": seed, **metrics})
            ablations[label][scenario] = _aggregate(runs)
        mirror[label] = _mirror_consistency(
            model,
            goal,
            images,
            depth,
            seed=args.seeds[0],
            sample_num=args.sample_num,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "s0_s1_s2_raw_trajectories.npz"
    np.savez_compressed(raw_path, goal=goal, images=images, depth=depth, **raw_outputs)
    output = args.output_dir / "inference_timeline.json"
    report = {
        "status": "PASS",
        "device": torch.cuda.get_device_name(0),
        "input_source": str(args.debug_input.resolve()),
        "input_shape": {"images": list(images.shape), "depth": list(depth.shape), "goal": goal.tolist()},
        "sample_num": args.sample_num,
        "seeds": args.seeds,
        "coordinate_contract": "trajectory x=forward, y=left; y<0 is right",
        "timeline": records,
        "ablations": ablations,
        "mirror_consistency": mirror,
        "raw_trajectory_npz": str(raw_path.resolve()),
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "output": str(output.resolve()),
        "S0": records[0]["forward_goal"]["mean"],
        "first_checkpoint": records[2]["forward_goal"]["mean"],
        "S2": records[-1]["forward_goal"]["mean"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
