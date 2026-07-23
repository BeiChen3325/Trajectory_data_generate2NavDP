"""Render metric depth maps for RobotNav training data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from robotnav.config import load_render_config
from robotnav.rendering.compare_render import render_las_points
from robotnav.rendering.render_one_view import (
    _rasterize_compat,
    axis_to_vector,
    build_single_view,
    load_ply_to_torch,
    make_intrinsics,
)


def parse_args() -> argparse.Namespace:
    config = load_render_config()
    parser = argparse.ArgumentParser(
        description="Render a single metric depth map from 3DGS PLY or LAS point cloud."
    )
    parser.add_argument(
        "--backend",
        choices=["3dgs", "las"],
        default="las",
        help="Depth source. 3dgs uses gsplat expected depth; las uses point-cloud z-buffer.",
    )
    parser.add_argument("--ply", default=str(config.paths.ply_path), help="Input 3DGS PLY file.")
    parser.add_argument("--las", default=str(config.paths.las_path), help="Input LAS file.")
    parser.add_argument(
        "--output-dir",
        default=str(config.paths.output_dir / "depth"),
        help="Directory for depth outputs and manifest.",
    )
    parser.add_argument("--name", default="depth", help="Output filename stem.")
    parser.add_argument("--width", type=int, default=config.camera.width)
    parser.add_argument("--height", type=int, default=config.camera.height)
    parser.add_argument("--fov", type=float, default=config.camera.fov)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--eye", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--look-at", type=float, nargs=3, default=None)
    parser.add_argument("--look-dir", type=float, nargs=3, default=[-1.0, 0.0, 0.0])
    parser.add_argument("--viewmat", type=float, nargs=16, default=None)
    parser.add_argument("--look-distance", type=float, default=1.0)
    parser.add_argument(
        "--up-axis", choices=["+x", "-x", "+y", "-y", "+z", "-z"], default=config.camera.up_axis
    )
    parser.add_argument("--near-plane", type=float, default=0.001)
    parser.add_argument("--far-plane", type=float, default=1.0e10)
    parser.add_argument(
        "--background", choices=["black", "white"], default=config.runtime.background
    )
    parser.add_argument(
        "--depth-mode",
        choices=["D", "ED"],
        default="ED",
        help="3DGS depth mode. ED is expected depth; D is accumulated weighted depth.",
    )
    parser.add_argument("--alpha-threshold", type=float, default=1.0e-4)
    parser.add_argument("--invalid-depth", type=float, default=0.0)
    parser.add_argument("--ply-unit-scale", type=float, default=1.0)
    parser.add_argument("--max-gaussians", type=int, default=0)
    parser.add_argument("--sh-degree", default="auto")
    parser.add_argument("--chunk-size", type=int, default=config.runtime.chunk_size)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--las-splat-radius", type=int, default=1)
    parser.add_argument("--las-color-gain", type=float, default=1.0)
    parser.add_argument(
        "--las-axis-transform",
        choices=["zup-to-yup", "none"],
        default="zup-to-yup",
        help="Transform LAS coordinates before projection.",
    )
    parser.add_argument(
        "--depth-png-scale",
        type=float,
        default=1000.0,
        help="Scale depth to uint16 PNG units when --uint16-mode metric. 1000 writes millimeters.",
    )
    parser.add_argument(
        "--uint16-mode",
        choices=["adaptive", "metric"],
        default="adaptive",
        help="adaptive writes a viewable uint16 PNG; metric writes depth * --depth-png-scale.",
    )
    return parser.parse_args()


def camera_from_args(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    camera_args = SimpleNamespace(
        width=args.width,
        height=args.height,
        fov=args.fov,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        eye=args.eye,
        look_at=args.look_at,
        look_dir=args.look_dir,
        viewmat=args.viewmat,
        look_distance=args.look_distance,
    )
    k_matrix = make_intrinsics(camera_args, device)
    up = axis_to_vector(args.up_axis, device)
    _, eye, target, viewmat = build_single_view(camera_args, device, up)
    return k_matrix, eye, target, viewmat


def render_3dgs_depth(
    args: argparse.Namespace, device: torch.device
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    k_matrix, eye, target, viewmat = camera_from_args(args, device)
    means, scales, quats, opacities, colors, center, radius, sh_degree = load_ply_to_torch(
        Path(args.ply),
        device=device,
        unit_scale=args.ply_unit_scale,
        max_gaussians=args.max_gaussians,
        sh_degree_arg=args.sh_degree,
    )

    with torch.no_grad():
        renders, alphas, meta = _rasterize_compat(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat.unsqueeze(0),
            Ks=k_matrix.unsqueeze(0),
            width=args.width,
            height=args.height,
            render_mode=args.depth_mode,
            sh_degree=sh_degree,
            near_plane=args.near_plane,
            far_plane=args.far_plane,
            packed=True,
        )
        torch.cuda.synchronize()

    depth = renders[0, ..., 0].detach().cpu().numpy().astype(np.float32)
    alpha = np.squeeze(alphas[0].detach().cpu().numpy().astype(np.float32), axis=-1)
    valid = alpha > args.alpha_threshold
    depth = np.where(valid, depth, float(args.invalid_depth)).astype(np.float32)
    info = {
        "backend": "3dgs",
        "depth_mode": args.depth_mode,
        "ply": str(Path(args.ply)),
        "eye": eye.detach().cpu().numpy().tolist(),
        "target": target.detach().cpu().numpy().tolist(),
        "K": k_matrix.detach().cpu().numpy().tolist(),
        "viewmat_row_major": viewmat.detach().cpu().numpy().reshape(-1).tolist(),
        "scene_center": center.detach().cpu().numpy().tolist(),
        "scene_radius": float(radius.detach().cpu().item()),
        "valid_pixels": int(np.count_nonzero(valid)),
        "meta_keys": sorted(meta.keys()),
    }
    return depth, valid, info


def render_las_depth(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_matrix, eye, target, viewmat = camera_from_args(args, device)
    _rgb, zbuffer, header, covered, in_front, in_image = render_las_points(
        Path(args.las),
        viewmat.detach().cpu().numpy(),
        k_matrix.detach().cpu().numpy(),
        args,
    )
    valid = np.isfinite(zbuffer)
    depth = np.where(valid, zbuffer, float(args.invalid_depth)).astype(np.float32)
    info = {
        "backend": "las",
        "las": str(Path(args.las)),
        "eye": eye.detach().cpu().numpy().tolist(),
        "target": target.detach().cpu().numpy().tolist(),
        "K": k_matrix.detach().cpu().numpy().tolist(),
        "viewmat_row_major": viewmat.detach().cpu().numpy().reshape(-1).tolist(),
        "las_header": header,
        "las_points_in_front": int(in_front),
        "las_points_projected_in_image_before_zbuffer": int(in_image),
        "valid_pixels": int(covered),
    }
    return depth, valid, info


def depth_display_range(depth: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    if not np.any(valid):
        return 0.0, 1.0
    values = depth[valid]
    lo, hi = np.percentile(values, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def normalized_depth(depth: np.ndarray, valid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    normalized = np.zeros(depth.shape, dtype=np.float32)
    if np.any(valid):
        normalized[valid] = np.clip((depth[valid] - lo) / (hi - lo), 0.0, 1.0)
    return normalized


def make_depth_preview(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    lo, hi = depth_display_range(depth, valid)
    gray = (normalized_depth(depth, valid, lo, hi) * 220.0).astype(np.uint8)
    preview = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return preview


def make_uint16_depth(
    depth: np.ndarray,
    valid: np.ndarray,
    mode: str,
    metric_scale: float,
) -> tuple[np.ndarray, dict[str, object]]:
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    info: dict[str, object] = {"uint16_mode": mode}
    if not np.any(valid):
        return encoded, info
    if mode == "metric":
        if metric_scale > 0:
            values = np.clip(depth[valid] * metric_scale, 0, np.iinfo(np.uint16).max)
            encoded[valid] = values.astype(np.uint16)
        info["depth_png_scale"] = float(metric_scale)
        return encoded, info

    lo, hi = depth_display_range(depth, valid)
    normalized = normalized_depth(depth, valid, lo, hi)
    encoded[valid] = np.clip(normalized[valid] * 50000.0, 1, 50000).astype(np.uint16)
    info["adaptive_depth_min"] = lo
    info["adaptive_depth_max"] = hi
    return encoded, info


def json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def save_outputs(
    output_dir: Path,
    name: str,
    depth: np.ndarray,
    valid: np.ndarray,
    info: dict[str, object],
    depth_png_scale: float,
    uint16_mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_npy = output_dir / f"{name}.npy"
    valid_png = output_dir / f"{name}_valid.png"
    preview_png = output_dir / f"{name}_preview.png"
    uint16_png = output_dir / f"{name}_uint16.png"
    manifest = output_dir / f"{name}_manifest.json"

    np.save(depth_npy, depth.astype(np.float32))
    cv2.imwrite(str(valid_png), valid.astype(np.uint8) * 255)
    cv2.imwrite(str(preview_png), make_depth_preview(depth, valid))

    encoded_depth, uint16_info = make_uint16_depth(depth, valid, uint16_mode, depth_png_scale)
    cv2.imwrite(str(uint16_png), encoded_depth)

    finite_values = depth[valid]
    stats = {
        "depth_min": float(finite_values.min()) if finite_values.size else None,
        "depth_max": float(finite_values.max()) if finite_values.size else None,
        "depth_mean": float(finite_values.mean()) if finite_values.size else None,
        **uint16_info,
        "outputs": {
            "depth_npy": str(depth_npy),
            "valid_png": str(valid_png),
            "preview_png": str(preview_png),
            "uint16_png": str(uint16_png),
        },
    }
    manifest.write_text(json.dumps(json_ready({**info, **stats}), indent=2), encoding="utf-8")
    print(f"Saved depth npy: {depth_npy}")
    print(f"Saved depth preview: {preview_png}")
    print(f"Saved uint16 depth: {uint16_png}")
    print(f"Saved manifest: {manifest}")


def main() -> None:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    args = parse_args()
    if args.backend == "3dgs":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. 3DGS depth rendering needs CUDA.")
        depth, valid, info = render_3dgs_depth(args, torch.device("cuda"))
    else:
        depth, valid, info = render_las_depth(args)
    save_outputs(
        Path(args.output_dir),
        args.name,
        depth,
        valid,
        info,
        args.depth_png_scale,
        args.uint16_mode,
    )


if __name__ == "__main__":
    main()
