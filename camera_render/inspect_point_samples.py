import argparse
import math
import random
import struct
from pathlib import Path

import numpy as np
from plyfile import PlyData


DEFAULT_LAS = r"C:\task\xlk_work\MindCloudXAI_output\test1-pointcloud-0704.las"
DEFAULT_PLY = r"C:\task\xlk_work\MindCloudXAI_output\test1_yup.ply"
DEFAULT_TXT = r"C:\task\xlk_work\tools\point_sample_report.txt"
SH_C0 = 0.28209479177387814


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-float(x)))


def sorted_property_names(vertex, prefix):
    names = [p.name for p in vertex.properties if p.name.startswith(prefix)]
    return sorted(names, key=lambda name: int(name.split("_")[-1]))


def read_c_string(raw):
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()


def parse_las_header(path):
    with open(path, "rb") as stream:
        data = stream.read(375)

    if data[:4] != b"LASF":
        raise ValueError(f"Not a LAS file: {path}")

    version_major = data[24]
    version_minor = data[25]
    header_size = struct.unpack_from("<H", data, 94)[0]
    offset_to_points = struct.unpack_from("<I", data, 96)[0]
    vlr_count = struct.unpack_from("<I", data, 100)[0]
    raw_point_format = data[104]
    point_format = raw_point_format & 0b00111111
    compressed = bool(raw_point_format & 0b10000000)
    point_record_length = struct.unpack_from("<H", data, 105)[0]
    legacy_point_count = struct.unpack_from("<I", data, 107)[0]
    scale = struct.unpack_from("<ddd", data, 131)
    offset = struct.unpack_from("<ddd", data, 155)
    max_x, min_x, max_y, min_y, max_z, min_z = struct.unpack_from("<dddddd", data, 179)

    point_count = legacy_point_count
    if version_major == 1 and version_minor >= 4 and len(data) >= 255:
        extended_point_count = struct.unpack_from("<Q", data, 247)[0]
        if extended_point_count:
            point_count = extended_point_count

    return {
        "file_signature": "LASF",
        "version": f"{version_major}.{version_minor}",
        "system_identifier": read_c_string(data[26:58]),
        "generating_software": read_c_string(data[58:90]),
        "creation_day_of_year": struct.unpack_from("<H", data, 90)[0],
        "creation_year": struct.unpack_from("<H", data, 92)[0],
        "header_size": header_size,
        "offset_to_point_data": offset_to_points,
        "vlr_count": vlr_count,
        "raw_point_format": raw_point_format,
        "point_format": point_format,
        "compressed": compressed,
        "point_record_length": point_record_length,
        "point_count": point_count,
        "scale": scale,
        "offset": offset,
        "min_xyz": (min_x, min_y, min_z),
        "max_xyz": (max_x, max_y, max_z),
    }


def parse_las_point(record, point_format, scale, offset):
    xi, yi, zi = struct.unpack_from("<iii", record, 0)
    intensity = struct.unpack_from("<H", record, 12)[0]
    flags = record[14]

    result = {
        "raw_XYZ_int": (xi, yi, zi),
        "XYZ_scaled": (
            xi * scale[0] + offset[0],
            yi * scale[1] + offset[1],
            zi * scale[2] + offset[2],
        ),
        "intensity": intensity,
    }

    if point_format <= 5:
        result.update(
            {
                "return_number": flags & 0b00000111,
                "number_of_returns": (flags >> 3) & 0b00000111,
                "scan_direction_flag": (flags >> 6) & 0b00000001,
                "edge_of_flight_line": (flags >> 7) & 0b00000001,
                "classification": record[15],
                "scan_angle_rank": struct.unpack_from("<b", record, 16)[0],
                "user_data": record[17],
                "point_source_id": struct.unpack_from("<H", record, 18)[0],
            }
        )
        cursor = 20
    else:
        classification_flags = record[15]
        result.update(
            {
                "return_number": flags & 0b00001111,
                "number_of_returns": (flags >> 4) & 0b00001111,
                "classification_flags": classification_flags,
                "scanner_channel": (classification_flags >> 4) & 0b00000011,
                "scan_direction_flag": (classification_flags >> 6) & 0b00000001,
                "edge_of_flight_line": (classification_flags >> 7) & 0b00000001,
                "classification": record[16],
                "user_data": record[17],
                "scan_angle": struct.unpack_from("<h", record, 18)[0] * 0.006,
                "point_source_id": struct.unpack_from("<H", record, 20)[0],
            }
        )
        cursor = 22

    if point_format in (1, 3, 4, 5, 6, 7, 8, 9, 10) and len(record) >= cursor + 8:
        result["gps_time"] = struct.unpack_from("<d", record, cursor)[0]
        cursor += 8

    if point_format in (2, 3, 5, 7, 8, 10) and len(record) >= cursor + 6:
        result["rgb"] = struct.unpack_from("<HHH", record, cursor)
        cursor += 6

    if point_format in (8, 10) and len(record) >= cursor + 2:
        result["nir"] = struct.unpack_from("<H", record, cursor)[0]

    return result


def sample_las_points(path, header, count, seed):
    if header["compressed"]:
        return [
            {
                "error": (
                    "This looks like compressed LAS/LAZ data. "
                    "Install laspy with laz support to decode point records."
                )
            }
        ]

    total = int(header["point_count"])
    if total <= 0:
        return []

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(total), min(count, total)))
    points = []
    with open(path, "rb") as stream:
        for index in indices:
            stream.seek(header["offset_to_point_data"] + index * header["point_record_length"])
            record = stream.read(header["point_record_length"])
            parsed = parse_las_point(
                record,
                header["point_format"],
                header["scale"],
                header["offset"],
            )
            parsed["index"] = index
            points.append(parsed)
    return points


def format_value(value):
    if isinstance(value, float):
        return f"{value:.9g}"
    if isinstance(value, tuple):
        return "(" + ", ".join(format_value(item) for item in value) + ")"
    if isinstance(value, np.ndarray):
        return format_value(tuple(float(item) for item in value.tolist()))
    return str(value)


def append_mapping(lines, title, mapping, indent="  "):
    lines.append(title)
    for key, value in mapping.items():
        lines.append(f"{indent}{key}: {format_value(value)}")


def sample_ply_gaussians(path, count, seed):
    plydata = PlyData.read(path)
    vertex = plydata["vertex"]
    total = len(vertex)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(total), min(count, total)))

    scale_names = sorted_property_names(vertex, "scale_")
    rot_names = sorted_property_names(vertex, "rot_")
    rest_names = sorted_property_names(vertex, "f_rest_")
    property_names = [p.name for p in vertex.properties]

    samples = []
    for index in indices:
        raw_scale = np.array([vertex[name][index] for name in scale_names], dtype=np.float64)
        raw_rot = np.array([vertex[name][index] for name in rot_names], dtype=np.float64)
        rot_norm = np.linalg.norm(raw_rot)
        quat_normed = raw_rot / rot_norm if rot_norm > 0 else raw_rot
        f_dc = np.array(
            [vertex["f_dc_0"][index], vertex["f_dc_1"][index], vertex["f_dc_2"][index]],
            dtype=np.float64,
        )

        rest_preview = {}
        for name in rest_names[:6]:
            rest_preview[name] = float(vertex[name][index])

        sample = {
            "index": index,
            "xyz": (
                float(vertex["x"][index]),
                float(vertex["y"][index]),
                float(vertex["z"][index]),
            ),
            "opacity_raw": float(vertex["opacity"][index]),
            "opacity_sigmoid": sigmoid(vertex["opacity"][index]),
            "scale_raw_log": tuple(float(x) for x in raw_scale),
            "scale_exp": tuple(float(x) for x in np.exp(raw_scale)),
            "quat_raw": tuple(float(x) for x in raw_rot),
            "quat_normalized": tuple(float(x) for x in quat_normed),
            "f_dc_raw": tuple(float(x) for x in f_dc),
            "dc_rgb_approx": tuple(float(x) for x in np.clip(f_dc * SH_C0 + 0.5, 0.0, 1.0)),
        }
        if rest_preview:
            sample["f_rest_preview_first_6"] = rest_preview
        samples.append(sample)

    return {
        "vertex_count": total,
        "property_count": len(property_names),
        "properties": property_names,
        "samples": samples,
    }


def build_report(args):
    lines = []
    las_path = Path(args.las)
    ply_path = Path(args.ply)

    lines.append("LAS header and point samples")
    lines.append("=" * 36)
    header = parse_las_header(las_path)
    append_mapping(lines, "Header:", header)

    lines.append("")
    lines.append(f"Random LAS point samples, count={args.count}, seed={args.seed}:")
    for point in sample_las_points(las_path, header, args.count, args.seed):
        lines.append(f"- point index {point.get('index', 'n/a')}")
        for key, value in point.items():
            if key != "index":
                lines.append(f"  {key}: {format_value(value)}")

    lines.append("")
    lines.append("3DGS PLY gaussian samples")
    lines.append("=" * 36)
    ply_info = sample_ply_gaussians(ply_path, args.count, args.seed)
    lines.append(f"vertex_count: {ply_info['vertex_count']}")
    lines.append(f"property_count: {ply_info['property_count']}")
    lines.append("properties: " + ", ".join(ply_info["properties"]))
    lines.append("")
    lines.append(f"Random PLY gaussian samples, count={args.count}, seed={args.seed}:")
    for sample in ply_info["samples"]:
        lines.append(f"- gaussian index {sample['index']}")
        for key, value in sample.items():
            if key != "index":
                if isinstance(value, dict):
                    lines.append(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        lines.append(f"    {sub_key}: {format_value(sub_value)}")
                else:
                    lines.append(f"  {key}: {format_value(value)}")

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect LAS header/point samples and 3DGS PLY gaussian samples."
    )
    parser.add_argument("--las", default=DEFAULT_LAS, help="Input LAS file.")
    parser.add_argument("--ply", default=DEFAULT_PLY, help="Input 3DGS PLY file.")
    parser.add_argument("--count", type=int, default=3, help="Number of samples per file.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--save-threshold",
        type=int,
        default=12000,
        help="Save to text file if report length exceeds this many characters.",
    )
    parser.add_argument("--txt", default=DEFAULT_TXT, help="Output text path if needed.")
    return parser.parse_args()


def main():
    args = parse_args()
    report = build_report(args)
    if len(report) > args.save_threshold:
        txt_path = Path(args.txt)
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(report, encoding="utf-8")
        print(f"Report is long ({len(report)} chars). Saved to: {txt_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
