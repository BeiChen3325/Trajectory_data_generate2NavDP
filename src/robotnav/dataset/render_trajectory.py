"""Stage 2: render a versioned camera trajectory into an RGB-D episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from robotnav.config import RenderConfig, load_render_config
from robotnav.dataset.config import DatasetBuildConfig, load_dataset_build_config
from robotnav.dataset.contracts import (
    CONTRACT_VERSION,
    DEPTH_UNITS_PER_METER,
    INVALID_DEPTH_VALUE,
    file_sha256,
    load_camera_trajectory,
)
from robotnav.rendering.depth_render import make_uint16_depth
from robotnav.rendering.render_one_view import (
    _rasterize_compat,
    load_ply_to_torch,
    make_intrinsics,
)

DEPTH_MODE = "ED"
ALPHA_THRESHOLD = 1.0e-4
NEAR_PLANE = 0.001
FAR_PLANE = 1.0e10
PLY_UNIT_SCALE = 1.0
SH_DEGREE = "auto"


def _camera_intrinsic(render_config: RenderConfig, device: torch.device) -> torch.Tensor:
    args = SimpleNamespace(
        width=render_config.camera.width,
        height=render_config.camera.height,
        fov=render_config.camera.fov,
        fx=None,
        fy=None,
        cx=None,
        cy=None,
    )
    return make_intrinsics(args, device)


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def run_render_trajectory(build_config: DatasetBuildConfig, render_config: RenderConfig) -> Path:
    """Render all contract poses while communicating only through files."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Trajectory 3DGS rendering requires CUDA.")
    ply_path = render_config.paths.ply_path
    if ply_path is None:
        raise ValueError("render.toml must define [paths].ply_filename")
    if not ply_path.is_file():
        raise FileNotFoundError(ply_path)

    camera_trajectory = load_camera_trajectory(
        build_config.paths.camera_trajectory_path,
        build_config.paths.camera_manifest_path,
    )
    output_dir = build_config.paths.rendered_episode_dir
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    means, scales, quats, opacities, colors, center, radius, sh_degree = load_ply_to_torch(
        ply_path,
        device=device,
        unit_scale=PLY_UNIT_SCALE,
        max_gaussians=0,
        sh_degree_arg=SH_DEGREE,
    )
    intrinsic = _camera_intrinsic(render_config, device)
    background_value = 1.0 if render_config.runtime.background == "white" else 0.0
    background = torch.full((3,), background_value, dtype=torch.float32, device=device)

    rgb_paths: list[str] = []
    depth_paths: list[str] = []
    valid_pixels: list[int] = []
    batch_size = build_config.rendering.camera_batch_size
    width = render_config.camera.width
    height = render_config.camera.height
    frame_digits = max(3, len(str(camera_trajectory.frame_count - 1)))

    for start in range(0, camera_trajectory.frame_count, batch_size):
        stop = min(start + batch_size, camera_trajectory.frame_count)
        viewmats = torch.as_tensor(
            camera_trajectory.world_to_camera[start:stop], dtype=torch.float32, device=device
        )
        intrinsics = intrinsic.unsqueeze(0).expand(stop - start, -1, -1)
        with torch.no_grad():
            rgb_renders, _rgb_alphas, _rgb_meta = _rasterize_compat(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmats,
                Ks=intrinsics,
                width=width,
                height=height,
                backgrounds=background,
                render_mode="RGB",
                sh_degree=sh_degree,
                near_plane=NEAR_PLANE,
                far_plane=FAR_PLANE,
                packed=True,
            )
            depth_renders, depth_alphas, _depth_meta = _rasterize_compat(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmats,
                Ks=intrinsics,
                width=width,
                height=height,
                render_mode=DEPTH_MODE,
                sh_degree=sh_degree,
                near_plane=NEAR_PLANE,
                far_plane=FAR_PLANE,
                packed=True,
            )
            torch.cuda.synchronize()

        rgb_batch = rgb_renders.clamp(0.0, 1.0).detach().cpu().numpy()
        depth_batch = depth_renders[..., 0].detach().cpu().numpy().astype(np.float32)
        alpha_batch = depth_alphas[..., 0].detach().cpu().numpy().astype(np.float32)

        for local_index, frame in enumerate(range(start, stop)):
            name = f"{int(camera_trajectory.frame_index[frame]):0{frame_digits}d}.png"
            rgb_relative = Path("rgb") / name
            depth_relative = Path("depth") / name
            rgb_u8 = np.rint(rgb_batch[local_index] * 255.0).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
            valid = alpha_batch[local_index] > ALPHA_THRESHOLD
            metric_depth = np.where(
                valid, depth_batch[local_index], float(INVALID_DEPTH_VALUE)
            ).astype(np.float32)
            depth_u16, _ = make_uint16_depth(
                metric_depth,
                valid,
                mode="metric",
                metric_scale=float(DEPTH_UNITS_PER_METER),
            )
            _write_png(output_dir / rgb_relative, rgb_bgr)
            _write_png(output_dir / depth_relative, depth_u16)
            rgb_paths.append(rgb_relative.as_posix())
            depth_paths.append(depth_relative.as_posix())
            valid_pixels.append(int(np.count_nonzero(valid)))
        print(f"Rendered frames {start}..{stop - 1}")

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "frame_count": camera_trajectory.frame_count,
        "camera_intrinsic": intrinsic.detach().cpu().numpy().reshape(-1).tolist(),
        "width": width,
        "height": height,
        "rgb_paths": rgb_paths,
        "depth_paths": depth_paths,
        "depth_units_per_meter": DEPTH_UNITS_PER_METER,
        "invalid_depth_value": INVALID_DEPTH_VALUE,
        "depth_mode": DEPTH_MODE,
        "alpha_threshold": ALPHA_THRESHOLD,
        "near_plane": NEAR_PLANE,
        "far_plane": FAR_PLANE,
        "ply_unit_scale": PLY_UNIT_SCALE,
        "ply": str(ply_path),
        "ply_sha256": file_sha256(ply_path),
        "camera_trajectory_npz_sha256": file_sha256(build_config.paths.camera_trajectory_path),
        "camera_trajectory_manifest_sha256": file_sha256(build_config.paths.camera_manifest_path),
        "scene_center": center.detach().cpu().numpy().tolist(),
        "scene_radius": float(radius.detach().cpu().item()),
        "valid_pixels": valid_pixels,
        "render_config": {
            "width": width,
            "height": height,
            "fov": render_config.camera.fov,
            "up_axis": render_config.camera.up_axis,
            "background": render_config.runtime.background,
            "camera_batch_size": batch_size,
        },
    }
    manifest_path = output_dir / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render camera trajectory files into RGB-D images."
    )
    parser.add_argument("--config", default="dataset_build.toml")
    parser.add_argument("--render-config", default="render.toml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_config = load_dataset_build_config(args.config)
    render_config = load_render_config(args.render_config)
    manifest_path = run_render_trajectory(build_config, render_config)
    print(f"Saved rendered episode manifest: {manifest_path}")


if __name__ == "__main__":
    main()
