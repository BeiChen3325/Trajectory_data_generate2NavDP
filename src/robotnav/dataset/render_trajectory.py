"""Stage 2: render all versioned camera trajectories into RGB-D episodes."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch

from robotnav.config import RenderConfig, load_render_config
from robotnav.dataset.batch_manifest import write_batch_manifest
from robotnav.dataset.config import DatasetBuildConfig, load_dataset_build_config
from robotnav.dataset.contracts import (
    CONTRACT_VERSION,
    DEPTH_UNITS_PER_METER,
    INVALID_DEPTH_VALUE,
    file_sha256,
    load_camera_trajectory,
)
from robotnav.dataset.trajectory_manifest import (
    EpisodeSpec,
    TrajectoryBatch,
    load_trajectory_batch,
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


@dataclass(frozen=True)
class LoadedRenderScene:
    means: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor
    center: torch.Tensor
    radius: torch.Tensor
    sh_degree: Any
    intrinsic: torch.Tensor
    rgb_background: torch.Tensor
    ply_path: Path
    ply_sha256: str
    device: torch.device


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


def load_render_scene(render_config: RenderConfig) -> LoadedRenderScene:
    """Load shared 3DGS scene data once for every episode in the batch."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Trajectory 3DGS rendering requires CUDA.")
    ply_path = render_config.paths.ply_path
    if ply_path is None:
        raise ValueError("render.toml must define [paths].ply_filename")
    if not ply_path.is_file():
        raise FileNotFoundError(ply_path)
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
    rgb_background = torch.full((3,), background_value, dtype=torch.float32, device=device)
    return LoadedRenderScene(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        colors=colors,
        center=center,
        radius=radius,
        sh_degree=sh_degree,
        intrinsic=intrinsic,
        rgb_background=rgb_background,
        ply_path=ply_path,
        ply_sha256=file_sha256(ply_path),
        device=device,
    )


def _validate_camera_binding(
    batch: TrajectoryBatch,
    episode: EpisodeSpec,
    metadata: dict[str, Any],
) -> None:
    expected_metadata = {
        "trajectory_id": episode.trajectory_id,
        "episode_index": episode.episode_index,
        "source_trajectory_sha256": episode.trajectory_sha256,
        "source_batch_manifest_sha256": batch.manifest_sha256,
        "source_scene_model_sha256": batch.source_scene_model_sha256,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"Camera trajectory {episode.trajectory_id!r} has stale {field}: "
                f"{metadata.get(field)!r}"
            )


def render_episode(
    build_config: DatasetBuildConfig,
    render_config: RenderConfig,
    batch: TrajectoryBatch,
    episode: EpisodeSpec,
    scene: LoadedRenderScene,
) -> Path:
    """Render one validated episode into its isolated output directory."""
    camera_trajectory = load_camera_trajectory(
        episode.paths.camera_trajectory_path,
        episode.paths.camera_manifest_path,
    )
    _validate_camera_binding(batch, episode, camera_trajectory.metadata)

    episode.paths.root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".rendered_episode-staging-", dir=episode.paths.root))
    output_dir = episode.paths.rendered_episode_dir
    (staging / "rgb").mkdir()
    (staging / "depth").mkdir()

    rgb_paths: list[str] = []
    depth_paths: list[str] = []
    valid_pixels: list[int] = []
    batch_size = build_config.rendering.camera_batch_size
    width = render_config.camera.width
    height = render_config.camera.height
    frame_digits = max(3, len(str(camera_trajectory.frame_count - 1)))

    try:
        for start in range(0, camera_trajectory.frame_count, batch_size):
            stop = min(start + batch_size, camera_trajectory.frame_count)
            viewmats = torch.as_tensor(
                camera_trajectory.world_to_camera[start:stop],
                dtype=torch.float32,
                device=scene.device,
            )
            intrinsics = scene.intrinsic.unsqueeze(0).expand(stop - start, -1, -1)
            with torch.no_grad():
                rgb_renders, _rgb_alphas, _rgb_meta = _rasterize_compat(
                    means=scene.means,
                    quats=scene.quats,
                    scales=scene.scales,
                    opacities=scene.opacities,
                    colors=scene.colors,
                    viewmats=viewmats,
                    Ks=intrinsics,
                    width=width,
                    height=height,
                    backgrounds=scene.rgb_background,
                    render_mode="RGB",
                    sh_degree=scene.sh_degree,
                    near_plane=NEAR_PLANE,
                    far_plane=FAR_PLANE,
                    packed=True,
                )
                depth_renders, depth_alphas, _depth_meta = _rasterize_compat(
                    means=scene.means,
                    quats=scene.quats,
                    scales=scene.scales,
                    opacities=scene.opacities,
                    colors=scene.colors,
                    viewmats=viewmats,
                    Ks=intrinsics,
                    width=width,
                    height=height,
                    backgrounds=None,
                    render_mode=DEPTH_MODE,
                    sh_degree=scene.sh_degree,
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
                _write_png(staging / rgb_relative, rgb_bgr)
                _write_png(staging / depth_relative, depth_u16)
                rgb_paths.append(rgb_relative.as_posix())
                depth_paths.append(depth_relative.as_posix())
                valid_pixels.append(int(np.count_nonzero(valid)))
            print(
                f"Rendered {episode.trajectory_id} frames {start}..{stop - 1}",
                flush=True,
            )

        manifest = {
            "contract_version": CONTRACT_VERSION,
            "trajectory_id": episode.trajectory_id,
            "episode_index": episode.episode_index,
            "episode_name": episode.episode_name,
            "frame_count": camera_trajectory.frame_count,
            "camera_intrinsic": scene.intrinsic.detach().cpu().numpy().reshape(-1).tolist(),
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
            "ply": str(scene.ply_path),
            "ply_sha256": scene.ply_sha256,
            "source_trajectory_sha256": episode.trajectory_sha256,
            "source_batch_manifest_sha256": batch.manifest_sha256,
            "source_scene_model_sha256": batch.source_scene_model_sha256,
            "camera_trajectory_npz_sha256": file_sha256(episode.paths.camera_trajectory_path),
            "camera_trajectory_manifest_sha256": file_sha256(episode.paths.camera_manifest_path),
            "scene_center": scene.center.detach().cpu().numpy().tolist(),
            "scene_radius": float(scene.radius.detach().cpu().item()),
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
        (staging / "render_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.rename(output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return episode.paths.render_manifest_path


def run_render_trajectory(
    build_config: DatasetBuildConfig, render_config: RenderConfig
) -> tuple[Path, ...]:
    """Render every manifest episode while loading the shared scene only once."""
    batch = load_trajectory_batch(
        build_config.paths.trajectory_manifest,
        build_config.paths.episodes_dir,
    )
    scene = load_render_scene(render_config)
    manifest_paths = tuple(
        render_episode(build_config, render_config, batch, episode, scene)
        for episode in batch.episodes
    )
    write_batch_manifest(batch, build_config.paths.batch_manifest_path)
    return manifest_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render all manifest camera trajectories into per-episode RGB-D images."
    )
    parser.add_argument("--config", default="dataset_build.toml")
    parser.add_argument("--render-config", default="render.toml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_config = load_dataset_build_config(args.config)
    render_config = load_render_config(args.render_config)
    manifest_paths = run_render_trajectory(build_config, render_config)
    print(f"Saved {len(manifest_paths)} rendered episode manifests")


if __name__ == "__main__":
    main()
