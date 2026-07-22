import argparse
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from gsplat import rasterization

try:
    from render_one_view import (
        axis_to_vector,
        build_single_view,
        load_ply_to_torch,
        make_intrinsics,
    )
except ModuleNotFoundError as exc:
    missing = exc.name or str(exc)
    raise ModuleNotFoundError(
        f"Missing dependency '{missing}'. Run this script in the same environment "
        "that can run render_one_view.py, for example your gsplat_pure environment."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT.parent / "MindCloudXAI_output"

DEFAULT_LAS = DATA_DIR / "test1-pointcloud-0704.las"
DEFAULT_PLY = DATA_DIR / "test1_yup.ply"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "render_output2D" / "compare_las_ply_origin"


def parse_las_header(path):
    with open(path, "rb") as stream:
        data = stream.read(375)

    if data[:4] != b"LASF":
        raise ValueError(f"Not a LAS file: {path}")

    version_major = data[24]
    version_minor = data[25]
    header_size = struct.unpack_from("<H", data, 94)[0]
    offset_to_points = struct.unpack_from("<I", data, 96)[0]
    raw_point_format = data[104]
    point_format = raw_point_format & 0b00111111
    compressed = bool(raw_point_format & 0b10000000)
    point_record_length = struct.unpack_from("<H", data, 105)[0]
    legacy_point_count = struct.unpack_from("<I", data, 107)[0]
    scale = np.array(struct.unpack_from("<ddd", data, 131), dtype=np.float64)
    offset = np.array(struct.unpack_from("<ddd", data, 155), dtype=np.float64)
    max_x, min_x, max_y, min_y, max_z, min_z = struct.unpack_from("<dddddd", data, 179)

    point_count = legacy_point_count
    if version_major == 1 and version_minor >= 4:
        extended_point_count = struct.unpack_from("<Q", data, 247)[0]
        if extended_point_count:
            point_count = extended_point_count

    return {
        "version": f"{version_major}.{version_minor}",
        "header_size": header_size,
        "offset_to_point_data": offset_to_points,
        "raw_point_format": raw_point_format,
        "point_format": point_format,
        "compressed": compressed,
        "point_record_length": point_record_length,
        "point_count": int(point_count),
        "scale": scale,
        "offset": offset,
        "min_xyz": (min_x, min_y, min_z),
        "max_xyz": (max_x, max_y, max_z),
    }


def las_dtype_for_format(point_format, record_length):
    if point_format not in (2, 3, 5):
        raise ValueError(
            "This script currently supports LAS point formats 2, 3, and 5 "
            f"for RGB point rendering; got format {point_format}."
        )

    if point_format == 2:
        rgb_offset = 20
    elif point_format == 3:
        rgb_offset = 28
    else:
        rgb_offset = 28

    return np.dtype(
        {
            "names": ["X", "Y", "Z", "red", "green", "blue"],
            "formats": ["<i4", "<i4", "<i4", "<u2", "<u2", "<u2"],
            "offsets": [0, 4, 8, rgb_offset, rgb_offset + 2, rgb_offset + 4],
            "itemsize": record_length,
        }
    )


def focal_from_matrix(K):
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def normalize_las_rgb(red, green, blue, color_gain):
    rgb16 = np.stack([red, green, blue], axis=1).astype(np.float32)
    rgb = rgb16 / 65535.0
    rgb *= float(color_gain)
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def transform_las_points(xyz, transform_name):
    if transform_name == "none":
        return xyz
    if transform_name == "zup-to-yup":
        transformed = np.empty_like(xyz)
        transformed[:, 0] = xyz[:, 0]
        transformed[:, 1] = -xyz[:, 2]
        transformed[:, 2] = xyz[:, 1]
        return transformed
    raise ValueError(f"Unknown LAS axis transform: {transform_name}")


def update_zbuffer(image, zbuffer, u, v, depth, rgb, width, height, splat_radius):
    offsets = [(0, 0)]
    if splat_radius > 0:
        offsets = [
            (dx, dy)
            for dy in range(-splat_radius, splat_radius + 1)
            for dx in range(-splat_radius, splat_radius + 1)
            if dx * dx + dy * dy <= splat_radius * splat_radius
        ]

    for dx, dy in offsets:
        uu = u + dx
        vv = v + dy
        valid = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
        if not np.any(valid):
            continue

        pix = vv[valid] * width + uu[valid]
        local_depth = depth[valid]
        local_rgb = rgb[valid]

        order = np.lexsort((local_depth, pix))
        pix = pix[order]
        local_depth = local_depth[order]
        local_rgb = local_rgb[order]

        unique_pix, first = np.unique(pix, return_index=True)
        candidate_depth = local_depth[first]
        candidate_rgb = local_rgb[first]

        current = zbuffer.reshape(-1)[unique_pix]
        better = candidate_depth < current
        if np.any(better):
            chosen_pix = unique_pix[better]
            zbuffer.reshape(-1)[chosen_pix] = candidate_depth[better]
            image.reshape(-1, 3)[chosen_pix] = candidate_rgb[better]


def render_las_points(las_path, viewmat_np, K_np, args):
    header = parse_las_header(las_path)
    if header["compressed"]:
        raise ValueError("Compressed LAZ/LAS is not supported by the raw reader.")

    dtype = las_dtype_for_format(header["point_format"], header["point_record_length"])
    width = args.width
    height = args.height
    fx, fy, cx, cy = focal_from_matrix(K_np)
    world_to_cam = viewmat_np[:3, :].astype(np.float64)

    bg = 255 if args.background == "white" else 0
    image = np.full((height, width, 3), bg, dtype=np.uint8)
    zbuffer = np.full((height, width), np.inf, dtype=np.float32)

    total = header["point_count"]
    projected_total = 0
    visible_total = 0

    print(f"Reading LAS in chunks: {las_path}")
    print(f"LAS points: {total}, format={header['point_format']}, record_len={header['point_record_length']}")

    with open(las_path, "rb") as stream:
        stream.seek(header["offset_to_point_data"])
        remaining = total
        chunk_index = 0
        while remaining > 0:
            count = min(args.chunk_size, remaining)
            raw = stream.read(count * header["point_record_length"])
            if not raw:
                break

            records = np.frombuffer(raw, dtype=dtype, count=count)
            xyz = np.empty((records.shape[0], 3), dtype=np.float64)
            xyz[:, 0] = records["X"] * header["scale"][0] + header["offset"][0]
            xyz[:, 1] = records["Y"] * header["scale"][1] + header["offset"][1]
            xyz[:, 2] = records["Z"] * header["scale"][2] + header["offset"][2]
            xyz = transform_las_points(xyz, args.las_axis_transform)

            cam = xyz @ world_to_cam[:, :3].T + world_to_cam[:, 3]
            depth = cam[:, 2]
            in_front = (depth > args.near_plane) & (depth < args.far_plane)
            if np.any(in_front):
                cam = cam[in_front]
                depth = depth[in_front].astype(np.float32)
                projected_total += int(depth.size)

                u = np.rint(fx * cam[:, 0] / depth + cx).astype(np.int32)
                v = np.rint(fy * cam[:, 1] / depth + cy).astype(np.int32)
                in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
                if np.any(in_image):
                    source_indices = np.nonzero(in_front)[0][in_image]
                    rgb = normalize_las_rgb(
                        records["red"][source_indices],
                        records["green"][source_indices],
                        records["blue"][source_indices],
                        args.las_color_gain,
                    )
                    update_zbuffer(
                        image=image,
                        zbuffer=zbuffer,
                        u=u[in_image],
                        v=v[in_image],
                        depth=depth[in_image],
                        rgb=rgb,
                        width=width,
                        height=height,
                        splat_radius=args.las_splat_radius,
                    )
                    visible_total += int(np.count_nonzero(in_image))

            remaining -= count
            chunk_index += 1
            if chunk_index % args.progress_every == 0 or remaining == 0:
                done = total - remaining
                print(
                    f"  LAS progress: {done}/{total} points, "
                    f"in_front={projected_total}, in_image={visible_total}"
                )

    covered = np.isfinite(zbuffer)
    return image, zbuffer, header, int(np.count_nonzero(covered)), projected_total, visible_total


def render_ply_gaussians(ply_path, viewmat, K, args, device):
    means, scales, quats, opacities, colors, center, radius, sh_degree = load_ply_to_torch(
        ply_path,
        device=device,
        unit_scale=args.ply_unit_scale,
        max_gaussians=args.max_gaussians,
        sh_degree_arg=args.sh_degree,
    )

    bg_value = 1.0 if args.background == "white" else 0.0
    backgrounds = torch.tensor([[bg_value, bg_value, bg_value]], dtype=torch.float32, device=device)

    with torch.no_grad():
        renders, alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=args.width,
            height=args.height,
            backgrounds=backgrounds,
            render_mode="RGB",
            sh_degree=sh_degree,
            near_plane=args.near_plane,
            far_plane=args.far_plane,
            packed=True,
        )
        torch.cuda.synchronize()

    image = (renders[0].clamp(0.0, 1.0).detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    alpha = alphas[0].detach().cpu().numpy()
    visible = 0
    gaussian_ids = meta.get("gaussian_ids")
    if gaussian_ids is not None:
        visible = int(torch.unique(gaussian_ids).numel())

    return image, alpha, center.detach().cpu().numpy(), radius.detach().cpu().item(), visible


def save_depth_preview(zbuffer, path):
    finite = np.isfinite(zbuffer)
    preview = np.zeros(zbuffer.shape, dtype=np.uint8)
    if np.any(finite):
        values = zbuffer[finite]
        lo, hi = np.percentile(values, [1.0, 99.0])
        if hi <= lo:
            hi = lo + 1.0
        normalized = (zbuffer - lo) / (hi - lo)
        preview[finite] = np.clip((1.0 - normalized[finite]) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), preview)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render one matched camera view from a LAS point cloud and a 3DGS PLY "
            "to visually check whether their coordinate systems agree."
        )
    )
    parser.add_argument("--las", default=str(DEFAULT_LAS))
    parser.add_argument("--ply", default=str(DEFAULT_PLY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov", type=float, default=50.0)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--eye", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--look-at", type=float, nargs=3, default=None)
    parser.add_argument("--look-dir", type=float, nargs=3, default=[-1.0, 0.0, 0.0])
    parser.add_argument("--viewmat", type=float, nargs=16, default=None)
    parser.add_argument("--look-distance", type=float, default=1.0)
    parser.add_argument("--up-axis", choices=["+x", "-x", "+y", "-y", "+z", "-z"], default="-y")
    parser.add_argument("--near-plane", type=float, default=0.001)
    parser.add_argument("--far-plane", type=float, default=1.0e10)
    parser.add_argument("--background", choices=["black", "white"], default="black")
    parser.add_argument(
        "--ply-unit-scale",
        type=float,
        default=1.0,
        help="Scale applied only to the 3DGS PLY. Use 1.0 for meter-level PLY/LAS comparison.",
    )
    parser.add_argument("--max-gaussians", type=int, default=0)
    parser.add_argument("--sh-degree", default="auto")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--las-splat-radius", type=int, default=1)
    parser.add_argument("--las-color-gain", type=float, default=1.0)
    parser.add_argument(
        "--las-axis-transform",
        choices=["zup-to-yup", "none"],
        default="zup-to-yup",
        help="Transform LAS coordinates before projection. Default maps Z-up LAS to the Y-up PLY convention.",
    )
    return parser.parse_args()


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. 3DGS rendering needs CUDA here.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
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
    K = make_intrinsics(camera_args, device)
    up = axis_to_vector(args.up_axis, device)
    _, eye, target, viewmat = build_single_view(camera_args, device, up)

    print(f"Camera eye: {eye.detach().cpu().numpy()}")
    print(f"Camera target: {target.detach().cpu().numpy()}")
    print(f"K: {K.detach().cpu().numpy().tolist()}")
    print(f"Viewmat row-major: {viewmat.detach().cpu().numpy().reshape(-1).tolist()}")

    ply_rgb, ply_alpha, ply_center, ply_radius, ply_visible = render_ply_gaussians(
        Path(args.ply), viewmat, K, args, device
    )
    las_rgb, las_depth, las_header, las_covered, las_in_front, las_in_image = render_las_points(
        Path(args.las),
        viewmat.detach().cpu().numpy(),
        K.detach().cpu().numpy(),
        args,
    )

    ply_path = output_dir / "ply_3dgs_rgb.png"
    las_path = output_dir / "las_pointcloud_rgb.png"
    side_path = output_dir / "side_by_side_ply_left_las_right.png"
    alpha_path = output_dir / "ply_alpha.png"
    depth_path = output_dir / "las_depth_preview.png"
    manifest_path = output_dir / "compare_manifest.txt"

    cv2.imwrite(str(ply_path), cv2.cvtColor(ply_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(las_path), cv2.cvtColor(las_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(side_path), cv2.cvtColor(np.concatenate([ply_rgb, las_rgb], axis=1), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(alpha_path), np.clip(ply_alpha * 255.0, 0, 255).astype(np.uint8))
    save_depth_preview(las_depth, depth_path)

    manifest = [
        f"las: {Path(args.las)}",
        f"ply: {Path(args.ply)}",
        f"output_dir: {output_dir}",
        f"width_height: {args.width} {args.height}",
        f"eye: {eye.detach().cpu().numpy().tolist()}",
        f"target: {target.detach().cpu().numpy().tolist()}",
        f"look_dir: {args.look_dir}",
        f"up_axis: {args.up_axis}",
        f"K: {K.detach().cpu().numpy().tolist()}",
        f"viewmat_row_major: {viewmat.detach().cpu().numpy().reshape(-1).tolist()}",
        f"near_far: {args.near_plane} {args.far_plane}",
        f"background: {args.background}",
        f"ply_unit_scale: {args.ply_unit_scale}",
        f"las_axis_transform: {args.las_axis_transform}",
        f"ply_center_after_scale: {ply_center.tolist()}",
        f"ply_radius_after_scale: {ply_radius}",
        f"ply_visible_gaussians: {ply_visible}",
        f"las_header: {las_header}",
        f"las_points_in_front: {las_in_front}",
        f"las_points_projected_in_image_before_zbuffer: {las_in_image}",
        f"las_covered_pixels_after_zbuffer: {las_covered}",
        f"las_splat_radius: {args.las_splat_radius}",
        "",
        "outputs:",
        f"  {ply_path.name}",
        f"  {las_path.name}",
        f"  {side_path.name}",
        f"  {alpha_path.name}",
        f"  {depth_path.name}",
    ]
    manifest_path.write_text("\n".join(manifest), encoding="utf-8")

    print(f"Saved PLY render: {ply_path}")
    print(f"Saved LAS render: {las_path}")
    print(f"Saved side-by-side comparison: {side_path}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
