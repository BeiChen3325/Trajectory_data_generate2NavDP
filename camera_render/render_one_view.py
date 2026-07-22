import argparse
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from plyfile import PlyData


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT.parent / "MindCloudXAI_output"

DEFAULT_PLY = DATA_DIR / "test1_yup.ply"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "render_output2D"

SH_C0 = 0.28209479177387814


def _sorted_property_names(vertex, prefix):
    names = [p.name for p in vertex.properties if p.name.startswith(prefix)]
    return sorted(names, key=lambda name: int(name.split("_")[-1]))


def _require_properties(vertex, names):
    available = {p.name for p in vertex.properties}
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"PLY is missing required 3DGS properties: {missing}")


def load_ply_to_torch(
    ply_path, device="cuda", unit_scale=1.0, max_gaussians=0, sh_degree_arg="auto"
):
    print(f"Reading 3DGS PLY: {ply_path}")
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]

    _require_properties(
        vertex,
        [
            "x",
            "y",
            "z",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        ],
    )

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)
    scale_names = _sorted_property_names(vertex, "scale_")
    rot_names = _sorted_property_names(vertex, "rot_")
    rest_names = _sorted_property_names(vertex, "f_rest_")

    scales_np = np.stack([vertex[name] for name in scale_names], axis=-1).astype(np.float32)
    quats_np = np.stack([vertex[name] for name in rot_names], axis=-1).astype(np.float32)
    opacity_np = np.asarray(vertex["opacity"], dtype=np.float32)
    f_dc = np.stack(
        [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1
    ).astype(np.float32)
    f_rest = None
    if rest_names:
        f_rest = np.stack([vertex[name] for name in rest_names], axis=-1).astype(np.float32)

    total = xyz.shape[0]
    if max_gaussians and total > max_gaussians:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(total, size=max_gaussians, replace=False))
        xyz = xyz[keep]
        scales_np = scales_np[keep]
        quats_np = quats_np[keep]
        opacity_np = opacity_np[keep]
        f_dc = f_dc[keep]
        if f_rest is not None:
            f_rest = f_rest[keep]
        print(f"Subsampled {total} -> {len(keep)} gaussians for a quick render.")

    means = torch.from_numpy(xyz).to(device) * unit_scale
    scales = torch.exp(torch.from_numpy(scales_np).to(device)) * unit_scale
    quats = F.normalize(torch.from_numpy(quats_np).to(device), dim=-1)
    opacities = torch.sigmoid(torch.from_numpy(opacity_np).to(device))

    sh_degree = None
    if sh_degree_arg.lower() == "dc":
        colors = torch.clamp(torch.from_numpy(f_dc).to(device) * SH_C0 + 0.5, 0.0, 1.0)
    else:
        max_sh_degree = 0
        if f_rest is not None and f_rest.shape[1] % 3 == 0:
            coeff_count = 1 + f_rest.shape[1] // 3
            root = int(math.sqrt(coeff_count))
            if root * root == coeff_count:
                max_sh_degree = root - 1

        requested = max_sh_degree if sh_degree_arg.lower() == "auto" else int(sh_degree_arg)
        sh_degree = min(requested, max_sh_degree)
        if sh_degree < 0:
            sh_degree = None
            colors = torch.clamp(torch.from_numpy(f_dc).to(device) * SH_C0 + 0.5, 0.0, 1.0)
        elif sh_degree == 0:
            colors = torch.from_numpy(f_dc[:, None, :]).to(device)
        else:
            needed_rest_coeffs = (sh_degree + 1) ** 2 - 1
            rest = f_rest.reshape(f_rest.shape[0], -1, 3)[:, :needed_rest_coeffs, :]
            sh_coeffs = np.concatenate([f_dc[:, None, :], rest], axis=1)
            colors = torch.from_numpy(sh_coeffs).to(device)

    bbox_min = means.min(dim=0).values
    bbox_max = means.max(dim=0).values
    center = (bbox_min + bbox_max) * 0.5
    extent = bbox_max - bbox_min
    radius = torch.linalg.norm(extent) * 0.5

    print(f"Loaded gaussians: {means.shape[0]}")
    print(f"Scene center: {center.detach().cpu().numpy()}")
    print(f"Scene extent: {extent.detach().cpu().numpy()} (unit_scale={unit_scale})")
    print(f"Scale range: {scales.min().item():.6g} .. {scales.max().item():.6g}")
    print(f"Opacity range: {opacities.min().item():.6g} .. {opacities.max().item():.6g}")
    print(f"Color mode: {'DC RGB' if sh_degree is None else f'SH degree {sh_degree}'}")

    return means, scales, quats, opacities, colors, center, radius, sh_degree


def focal_from_fov(width, height, fov_degrees, device):
    fov = math.radians(fov_degrees)
    focal = 0.5 * max(width, height) / math.tan(0.5 * fov)
    return torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )


def make_intrinsics(args, device):
    if args.fx is None and args.fy is None and args.cx is None and args.cy is None:
        return focal_from_fov(args.width, args.height, args.fov, device)

    fallback = focal_from_fov(args.width, args.height, args.fov, device)
    fx = fallback[0, 0].item() if args.fx is None else args.fx
    fy = fallback[1, 1].item() if args.fy is None else args.fy
    cx = fallback[0, 2].item() if args.cx is None else args.cx
    cy = fallback[1, 2].item() if args.cy is None else args.cy
    return torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )


def look_at_world_to_camera(eye, target, up):
    forward = F.normalize(target - eye, dim=0)
    right = torch.cross(forward, up, dim=0)
    if torch.linalg.norm(right) < 1e-6:
        if abs(float(forward[1].detach().cpu())) < 0.9:
            alt_up = torch.tensor([0.0, 1.0, 0.0], dtype=eye.dtype, device=eye.device)
        else:
            alt_up = torch.tensor([1.0, 0.0, 0.0], dtype=eye.dtype, device=eye.device)
        right = torch.cross(forward, alt_up, dim=0)
    right = F.normalize(right, dim=0)
    down = F.normalize(torch.cross(forward, right, dim=0), dim=0)

    viewmat = torch.eye(4, dtype=torch.float32, device=eye.device)
    viewmat[:3, :3] = torch.stack([right, down, forward], dim=0)
    viewmat[:3, 3] = -viewmat[:3, :3] @ eye
    return viewmat


def build_single_view(args, device, up):
    eye = torch.tensor(args.eye, dtype=torch.float32, device=device)

    if args.viewmat:
        if len(args.viewmat) != 16:
            raise ValueError("--viewmat expects exactly 16 float values in row-major order.")
        viewmat = torch.tensor(args.viewmat, dtype=torch.float32, device=device).reshape(4, 4)
        inv = torch.linalg.inv(viewmat)
        eye = inv[:3, 3]
        target = eye + inv[:3, 2]
        return "single", eye, target, viewmat

    if args.look_at is not None:
        target = torch.tensor(args.look_at, dtype=torch.float32, device=device)
    else:
        direction = torch.tensor(args.look_dir, dtype=torch.float32, device=device)
        if torch.linalg.norm(direction) < 1e-8:
            raise ValueError("--look-dir must be non-zero.")
        target = eye + F.normalize(direction, dim=0) * args.look_distance

    return "single", eye, target, look_at_world_to_camera(eye, target, up)


def axis_to_vector(axis, device):
    table = {
        "+x": [1.0, 0.0, 0.0],
        "-x": [-1.0, 0.0, 0.0],
        "+y": [0.0, 1.0, 0.0],
        "-y": [0.0, -1.0, 0.0],
        "+z": [0.0, 0.0, 1.0],
        "-z": [0.0, 0.0, -1.0],
    }
    return torch.tensor(table[axis], dtype=torch.float32, device=device)


def format_coord_token(value, digits=3):
    text = f"{float(value):.{digits}f}"
    return text.replace("-", "m").replace(".", "p")


def format_vec_token(vec, prefix):
    x, y, z = vec
    return (
        f"{prefix}_x{format_coord_token(x)}"
        f"_y{format_coord_token(y)}"
        f"_z{format_coord_token(z)}"
    )


def find_dense_focus(means, opacities, grid_size=80, neighborhood=1, min_radius=0.15):
    xyz = means.detach().cpu().numpy().astype(np.float32)
    weights = opacities.detach().cpu().numpy().astype(np.float32)
    weights = np.clip(weights, 0.05, 1.0)

    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    extent = np.maximum(bbox_max - bbox_min, 1e-6)

    cell = np.floor((xyz - bbox_min) / extent * grid_size).astype(np.int32)
    cell = np.clip(cell, 0, grid_size - 1)
    flat = cell[:, 0] + grid_size * (cell[:, 1] + grid_size * cell[:, 2])
    hist = np.bincount(flat, weights=weights, minlength=grid_size**3)
    peak_flat = int(hist.argmax())
    peak = np.array(
        [
            peak_flat % grid_size,
            (peak_flat // grid_size) % grid_size,
            peak_flat // (grid_size * grid_size),
        ],
        dtype=np.int32,
    )

    selected = np.all(np.abs(cell - peak[None, :]) <= neighborhood, axis=1)
    if selected.sum() < 256:
        for radius in range(neighborhood + 1, min(4, grid_size // 2) + 1):
            selected = np.all(np.abs(cell - peak[None, :]) <= radius, axis=1)
            if selected.sum() >= 256:
                break

    selected_xyz = xyz[selected]
    selected_weights = weights[selected]
    if selected_xyz.size == 0:
        selected_xyz = xyz
        selected_weights = weights

    target = np.average(selected_xyz, axis=0, weights=selected_weights)
    dists = np.linalg.norm(selected_xyz - target[None, :], axis=1)
    local_radius = float(np.percentile(dists, 95))
    local_radius = max(local_radius, min_radius)

    return {
        "target": torch.tensor(target, dtype=torch.float32),
        "local_radius": local_radius,
        "peak_cell": tuple(int(x) for x in peak.tolist()),
        "selected_count": int(selected_xyz.shape[0]),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
    }


def make_view_specs():
    return [
        ("posx", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        ("negx", np.array([-1.0, 0.0, 0.0], dtype=np.float32)),
        ("posy", np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ("negy", np.array([0.0, -1.0, 0.0], dtype=np.float32)),
        ("posz", np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ("negz", np.array([0.0, 0.0, -1.0], dtype=np.float32)),
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render multiple 2D views from a 3D Gaussian Splatting PLY file."
    )
    parser.add_argument("--ply", default=str(DEFAULT_PLY), help="Input 3DGS PLY file.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for rendered PNGs and manifest text.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output PNG path for single mode.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov", type=float, default=50.0, help="Horizontal-ish FOV.")
    parser.add_argument("--fx", type=float, default=None, help="Camera focal length fx.")
    parser.add_argument("--fy", type=float, default=None, help="Camera focal length fy.")
    parser.add_argument("--cx", type=float, default=None, help="Camera principal point cx.")
    parser.add_argument("--cy", type=float, default=None, help="Camera principal point cy.")
    parser.add_argument(
        "--unit-scale",
        type=float,
        default=1.0,
        help="Scale applied to PLY xyz and Gaussian scales. Default 1.0 keeps coordinates aligned with LAS/trajectory_work.",
    )
    parser.add_argument(
        "--distance-mult",
        type=float,
        default=1.25,
        help="Camera distance as a multiplier of the local dense-region radius.",
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=0,
        help="Optional random subset size for quick testing. 0 means use all gaussians.",
    )
    parser.add_argument(
        "--sh-degree",
        default="auto",
        help='Use "auto" for all available SH coefficients, "dc" for DC-only RGB, or an integer degree.',
    )
    parser.add_argument(
        "--background",
        choices=["black", "white"],
        default="black",
        help="Background color for uncovered pixels.",
    )
    parser.add_argument(
        "--views",
        type=int,
        default=6,
        help="How many axis views to render.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "panorama", "orbit"],
        default="single",
        help="single renders one RGB image; panorama keeps the camera at --eye and looks outward; orbit places cameras around the dense focus and looks inward.",
    )
    parser.add_argument(
        "--eye",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="Camera position for panorama mode, in scaled world coordinates.",
    )
    parser.add_argument(
        "--look-at",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Target point for single mode. Overrides --look-dir when provided.",
    )
    parser.add_argument(
        "--look-dir",
        type=float,
        nargs=3,
        default=[-1.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="Look direction for single mode when --look-at is not provided.",
    )
    parser.add_argument(
        "--viewmat",
        type=float,
        nargs=16,
        default=None,
        metavar="M",
        help="World-to-camera 4x4 matrix in row-major order for single mode. Overrides --eye/--look-at/--look-dir.",
    )
    parser.add_argument(
        "--look-distance",
        type=float,
        default=1.0,
        help="Target distance from the camera for single/panorama mode.",
    )
    parser.add_argument(
        "--focus-grid",
        type=int,
        default=80,
        help="Voxel grid size used to locate the densest interior region.",
    )
    parser.add_argument(
        "--focus-neighborhood",
        type=int,
        default=1,
        help="Neighborhood radius around the densest voxel used to compute the target point.",
    )
    parser.add_argument(
        "--min-local-radius",
        type=float,
        default=0.15,
        help="Lower bound for the dense-region radius before applying --distance-mult.",
    )
    parser.add_argument(
        "--up-axis",
        choices=["+x", "-x", "+y", "-y", "+z", "-z"],
        default="-y",
        help="World-space up axis for the virtual camera.",
    )
    parser.add_argument("--near-plane", type=float, default=0.001)
    parser.add_argument("--far-plane", type=float, default=1.0e10)
    return parser.parse_args()


def render_batch():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. gsplat rendering needs CUDA here.")

    device = torch.device("cuda")
    ply_path = Path(args.ply)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    means, scales, quats, opacities, colors, center, radius, sh_degree = load_ply_to_torch(
        ply_path,
        device=device,
        unit_scale=args.unit_scale,
        max_gaussians=args.max_gaussians,
        sh_degree_arg=args.sh_degree,
    )

    K = make_intrinsics(args, device)
    up = axis_to_vector(args.up_axis, device)
    bg_value = 1.0 if args.background == "white" else 0.0
    background = torch.tensor(
        [[bg_value, bg_value, bg_value]], dtype=torch.float32, device=device
    )

    viewmats = []
    backgrounds = []
    names = []
    focus = None
    local_radius = None
    distance = None

    if args.mode == "single":
        name, eye, target, viewmat = build_single_view(args, device, up)
        viewmats.append(viewmat)
        backgrounds.append(background[0])
        names.append((name, eye.detach().cpu().numpy(), target.detach().cpu().numpy()))
    else:
        view_specs = make_view_specs()
        if args.views > 0:
            view_specs = view_specs[: args.views]
        if not view_specs:
            raise ValueError("No views selected for rendering.")

    if args.mode == "orbit":
        focus = find_dense_focus(
            means=means,
            opacities=opacities,
            grid_size=args.focus_grid,
            neighborhood=args.focus_neighborhood,
            min_radius=args.min_local_radius,
        )
        base_target = focus["target"].to(device)
        local_radius = focus["local_radius"]
        distance = max(local_radius * args.distance_mult, 0.15)
    elif args.mode == "panorama":
        base_eye = torch.tensor(args.eye, dtype=torch.float32, device=device)

    if args.mode in ("panorama", "orbit"):
        for name, direction in view_specs:
            direction_tensor = torch.tensor(direction, dtype=torch.float32, device=device)
            if args.mode == "orbit":
                target = base_target
                eye = target + direction_tensor * distance
            else:
                eye = base_eye
                target = eye + direction_tensor * args.look_distance
            viewmats.append(look_at_world_to_camera(eye, target, up))
            backgrounds.append(background[0])
            names.append(
                (
                    name,
                    eye.detach().cpu().numpy(),
                    target.detach().cpu().numpy(),
                )
            )

    viewmats = torch.stack(viewmats, dim=0)
    Ks = K.unsqueeze(0).repeat(viewmats.shape[0], 1, 1)
    backgrounds = torch.stack(backgrounds, dim=0)

    print(f"Loaded gaussians: {means.shape[0]}")
    print(f"Scene center: {center.detach().cpu().numpy()}")
    if args.mode == "orbit":
        print(
            f"Focus target: {base_target.detach().cpu().numpy()}, peak_cell={focus['peak_cell']}, selected={focus['selected_count']}"
        )
        print(f"Local radius: {local_radius:.6g}, render distance: {distance:.6g}")
    elif args.mode == "panorama":
        print(f"Panorama eye: {base_eye.detach().cpu().numpy()}")
        print(f"Look distance: {args.look_distance:.6g}")
    else:
        print(f"Single eye: {names[0][1]}")
        print(f"Single target: {names[0][2]}")
    print(f"K: {K.detach().cpu().numpy().tolist()}")
    print(f"Rendering {len(names)} views with gsplat.rasterization...")

    with torch.no_grad():
        renders, alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
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

    gaussian_ids = meta.get("gaussian_ids")
    camera_ids = meta.get("camera_ids")
    visible_by_view = {}
    if gaussian_ids is not None and camera_ids is not None:
        for cid in range(viewmats.shape[0]):
            mask = camera_ids == cid
            if mask.any():
                visible_by_view[cid] = int(torch.unique(gaussian_ids[mask]).numel())
            else:
                visible_by_view[cid] = 0

    manifest_lines = []
    manifest_lines.append(f"ply: {ply_path}")
    manifest_lines.append(f"mode: {args.mode}")
    manifest_lines.append(f"scene_center: {format_vec_token(center.detach().cpu().numpy(), 'scene')}")
    if args.mode == "orbit":
        manifest_lines.append(
            f"focus_target: {format_vec_token(base_target.detach().cpu().numpy(), 'target')}"
        )
        manifest_lines.append(f"peak_cell: {focus['peak_cell']}")
        manifest_lines.append(f"local_radius: {local_radius:.6g}")
        manifest_lines.append(f"distance: {distance:.6g}")
    elif args.mode == "panorama":
        manifest_lines.append(
            f"panorama_eye: {format_vec_token(base_eye.detach().cpu().numpy(), 'eye')}"
        )
        manifest_lines.append(f"look_distance: {args.look_distance:.6g}")
    else:
        manifest_lines.append(f"single_eye: {format_vec_token(names[0][1], 'eye')}")
        manifest_lines.append(f"single_target: {format_vec_token(names[0][2], 'target')}")
    manifest_lines.append(f"K: {K.detach().cpu().numpy().tolist()}")
    manifest_lines.append(f"near_plane: {args.near_plane:.6g}")
    manifest_lines.append(f"views: {len(names)}")
    manifest_lines.append("")

    for vid, (name, eye_np, target_np) in enumerate(names):
        render_img = renders[vid].clamp(0.0, 1.0).detach().cpu().numpy()
        alpha = alphas[vid].detach().cpu()
        image_u8 = (render_img * 255.0).round().astype(np.uint8)
        image_bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)

        if args.mode == "single" and args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_name = out_path.name
        else:
            out_name = (
                f"{vid:02d}_{args.mode}_{name}_"
                f"{format_vec_token(target_np, 'tgt')}_"
                f"{format_vec_token(eye_np, 'eye')}.png"
            )
            out_path = output_dir / out_name
        if not cv2.imwrite(str(out_path), image_bgr):
            raise RuntimeError(f"Failed to write output image: {out_path}")

        visible = visible_by_view.get(vid, 0)
        print(
            f"[{vid:02d}] {name}: eye={eye_np}, target={target_np}, visible={visible}, alpha={alpha.min().item():.4g}..{alpha.max().item():.4g}"
        )
        manifest_lines.append(f"{out_name}")
        manifest_lines.append(f"  eye: {eye_np.tolist()}")
        manifest_lines.append(f"  target: {target_np.tolist()}")
        manifest_lines.append(f"  visible_gaussians: {visible}")

    manifest_path = output_dir / "render_manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    print(f"Saved render set to: {output_dir}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    render_batch()
