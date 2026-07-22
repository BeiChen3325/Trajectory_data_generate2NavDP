#!/usr/bin/env python3
"""Convert a 3D Gaussian Splatting PLY file to USD ASCII.

The output is a UsdGeom.Points prim. Standard USD viewers can display it as a
colored point cloud, while Gaussian-specific values are preserved as custom
primvars for downstream tools that know how to reconstruct splats.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


SH_C0 = 0.28209479177387814


PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a 3DGS binary_little_endian PLY to USD ASCII."
    )
    parser.add_argument("input", type=Path, help="Input 3DGS .ply file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .usda path. Defaults to input path with .usda suffix.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Evenly sample at most this many points. 0 keeps all points.",
    )
    parser.add_argument(
        "--include-sh-rest",
        action="store_true",
        help="Also write f_rest_* SH coefficients. This can make the USD very large.",
    )
    parser.add_argument(
        "--include-normals",
        action="store_true",
        help="Write nx/ny/nz as normals when present.",
    )
    parser.add_argument(
        "--up-axis",
        choices=("Y", "Z"),
        default="Y",
        help="USD stage up axis metadata.",
    )
    parser.add_argument(
        "--width-scale",
        type=float,
        default=2.0,
        help="Multiplier applied to max Gaussian scale to create USD point widths.",
    )
    parser.add_argument(
        "--min-width",
        type=float,
        default=0.0,
        help="Clamp USD point widths to at least this world-space value.",
    )
    return parser.parse_args()


def read_ply_header(path: Path) -> tuple[int, int, list[tuple[str, str]]]:
    header_lines: list[str] = []
    with path.open("rb") as stream:
        while True:
            raw = stream.readline()
            if raw == b"":
                raise ValueError("Unexpected EOF before end_header")
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError("PLY header is not ASCII") from exc
            header_lines.append(line)
            if line == "end_header":
                break
        data_offset = stream.tell()

    if not header_lines or header_lines[0] != "ply":
        raise ValueError("Input is not a PLY file")

    fmt = None
    vertex_count = None
    vertex_props: list[tuple[str, str]] = []
    current_element = None

    for line in header_lines[1:]:
        if not line or line.startswith("comment"):
            continue
        parts = line.split()
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if parts[1] == "list":
                raise ValueError("List properties on vertex are not supported")
            vertex_props.append((parts[2], parts[1]))

    if fmt != "binary_little_endian":
        raise ValueError(f"Only binary_little_endian PLY is supported, got {fmt!r}")
    if vertex_count is None:
        raise ValueError("PLY has no vertex element")
    if not vertex_props:
        raise ValueError("PLY vertex element has no properties")

    return data_offset, vertex_count, vertex_props


def load_vertices(path: Path) -> np.ndarray:
    data_offset, vertex_count, vertex_props = read_ply_header(path)
    dtype_fields = []
    for name, ply_type in vertex_props:
        try:
            dtype_fields.append((name, PLY_TYPES[ply_type]))
        except KeyError as exc:
            raise ValueError(f"Unsupported PLY property type: {ply_type}") from exc

    dtype = np.dtype(dtype_fields)
    with path.open("rb") as stream:
        stream.seek(data_offset)
        vertices = np.fromfile(stream, dtype=dtype, count=vertex_count)

    if len(vertices) != vertex_count:
        raise ValueError(f"Expected {vertex_count} vertices, read {len(vertices)}")
    return vertices


def require_fields(vertices: np.ndarray, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in vertices.dtype.names]
    if missing:
        raise ValueError(f"Missing required PLY fields: {', '.join(missing)}")


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def stack_fields(vertices: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([vertices[name] for name in names]).astype(np.float32)


def sample_vertices(vertices: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(vertices) <= max_points:
        return vertices
    indices = np.linspace(0, len(vertices) - 1, max_points, dtype=np.int64)
    return vertices[indices]


def usd_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def write_vec_array(
    stream,
    usd_type: str,
    attr_name: str,
    values: np.ndarray,
    fmt: str,
    interpolation: str | None = None,
) -> None:
    stream.write(f"        {usd_type} {attr_name} = [\n")
    if len(values) > 1:
        np.savetxt(stream, values[:-1], fmt=f"            {fmt},")
    if len(values) > 0:
        np.savetxt(stream, values[-1:], fmt=f"            {fmt}")
    stream.write("        ]")
    if interpolation:
        stream.write(f' (\n            interpolation = "{interpolation}"\n        )')
    stream.write("\n")


def write_scalar_array(
    stream,
    usd_type: str,
    attr_name: str,
    values: np.ndarray,
    interpolation: str | None = None,
) -> None:
    stream.write(f"        {usd_type} {attr_name} = [\n")
    values_2d = values.reshape(-1, 1)
    if len(values_2d) > 1:
        np.savetxt(stream, values_2d[:-1], fmt="            %.9g,")
    if len(values_2d) > 0:
        np.savetxt(stream, values_2d[-1:], fmt="            %.9g")
    stream.write("        ]")
    if interpolation:
        stream.write(f' (\n            interpolation = "{interpolation}"\n        )')
    stream.write("\n")


def write_usda(
    output_path: Path,
    source_path: Path,
    vertices: np.ndarray,
    args: argparse.Namespace,
) -> None:
    require_fields(vertices, ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"))

    positions = stack_fields(vertices, ("x", "y", "z"))
    f_dc = stack_fields(vertices, ("f_dc_0", "f_dc_1", "f_dc_2"))
    colors = np.clip(f_dc * SH_C0 + 0.5, 0.0, 1.0)

    opacity = (
        sigmoid(vertices["opacity"].astype(np.float32))
        if "opacity" in vertices.dtype.names
        else np.ones(len(vertices), dtype=np.float32)
    )

    if {"scale_0", "scale_1", "scale_2"}.issubset(vertices.dtype.names):
        raw_scales = stack_fields(vertices, ("scale_0", "scale_1", "scale_2"))
        scales = np.exp(np.clip(raw_scales, -30.0, 30.0)).astype(np.float32)
        widths = np.max(scales, axis=1) * float(args.width_scale)
        if args.min_width > 0:
            widths = np.maximum(widths, args.min_width)
    else:
        scales = None
        widths = np.ones(len(vertices), dtype=np.float32)

    if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(vertices.dtype.names):
        rotations = stack_fields(vertices, ("rot_0", "rot_1", "rot_2", "rot_3"))
        norms = np.linalg.norm(rotations, axis=1, keepdims=True)
        rotations = rotations / np.maximum(norms, 1.0e-12)
    else:
        rotations = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("#usda 1.0\n")
        stream.write("(\n")
        stream.write('    defaultPrim = "GaussianSplatPointCloud"\n')
        stream.write("    metersPerUnit = 1\n")
        stream.write(f'    upAxis = "{args.up_axis}"\n')
        stream.write(")\n\n")
        stream.write('def Xform "GaussianSplatPointCloud" (\n')
        stream.write("    customData = {\n")
        stream.write(f'        string sourcePly = "{usd_string(str(source_path))}"\n')
        stream.write(f"        int gaussianCount = {len(vertices)}\n")
        stream.write(
            '        string representation = "UsdGeom.Points with 3DGS custom primvars"\n'
        )
        stream.write("    }\n")
        stream.write(")\n")
        stream.write("{\n")
        stream.write('    def Points "points"\n')
        stream.write("    {\n")
        write_vec_array(
            stream, "point3f[]", "points", positions, "(%.9g, %.9g, %.9g)"
        )
        write_scalar_array(stream, "float[]", "widths", widths)
        write_vec_array(
            stream,
            "color3f[]",
            "primvars:displayColor",
            colors,
            "(%.9g, %.9g, %.9g)",
            "vertex",
        )
        write_scalar_array(
            stream, "float[]", "primvars:displayOpacity", opacity, "vertex"
        )
        write_vec_array(
            stream,
            "float3[]",
            "primvars:gs_f_dc",
            f_dc,
            "(%.9g, %.9g, %.9g)",
            "vertex",
        )

        if scales is not None:
            write_vec_array(
                stream,
                "float3[]",
                "primvars:gs_scale",
                scales,
                "(%.9g, %.9g, %.9g)",
                "vertex",
            )
        if rotations is not None:
            write_vec_array(
                stream,
                "float4[]",
                "primvars:gs_rotation",
                rotations,
                "(%.9g, %.9g, %.9g, %.9g)",
                "vertex",
            )

        if args.include_normals:
            normal_fields = ("nx", "ny", "nz")
            if set(normal_fields).issubset(vertices.dtype.names):
                normals = stack_fields(vertices, normal_fields)
                write_vec_array(
                    stream,
                    "normal3f[]",
                    "normals",
                    normals,
                    "(%.9g, %.9g, %.9g)",
                    "vertex",
                )

        if args.include_sh_rest:
            rest_names = sorted(
                (
                    name
                    for name in vertices.dtype.names
                    if name.startswith("f_rest_")
                ),
                key=lambda item: int(item.rsplit("_", 1)[1]),
            )
            for name in rest_names:
                write_scalar_array(
                    stream,
                    "float[]",
                    f"primvars:gs_{name}",
                    vertices[name].astype(np.float32),
                    "vertex",
                )

        stream.write("    }\n")
        stream.write("}\n")


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output or input_path.with_suffix(".usda")

    vertices = load_vertices(input_path)
    vertices = sample_vertices(vertices, args.max_points)

    write_usda(output_path, input_path, vertices, args)
    size_mb = output_path.stat().st_size / (1024.0 * 1024.0)
    print(f"Wrote {output_path} ({len(vertices)} points, {size_mb:.1f} MiB)")


if __name__ == "__main__":
    main()
