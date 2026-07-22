import struct
from pathlib import Path

import numpy as np


def parse_las_header(path):
    path = Path(path)
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


def las_xyz_dtype(record_length):
    return np.dtype(
        {
            "names": ["X", "Y", "Z"],
            "formats": ["<i4", "<i4", "<i4"],
            "offsets": [0, 4, 8],
            "itemsize": record_length,
        }
    )


def transform_points_xyz(xyz, transform_name):
    if transform_name == "none":
        return xyz
    if transform_name == "zup-to-yup":
        transformed = np.empty_like(xyz)
        transformed[:, 0] = xyz[:, 0]
        transformed[:, 1] = -xyz[:, 2]
        transformed[:, 2] = xyz[:, 1]
        return transformed
    raise ValueError(f"Unknown axis transform: {transform_name}")


def iter_las_xyz(path, chunk_size=1_000_000, axis_transform="zup-to-yup"):
    path = Path(path)
    header = parse_las_header(path)
    if header["compressed"]:
        raise ValueError("Compressed LAS/LAZ is not supported by this raw reader.")

    dtype = las_xyz_dtype(header["point_record_length"])
    total = header["point_count"]

    with open(path, "rb") as stream:
        stream.seek(header["offset_to_point_data"])
        remaining = total
        while remaining > 0:
            count = min(chunk_size, remaining)
            raw = stream.read(count * header["point_record_length"])
            if not raw:
                break
            records = np.frombuffer(raw, dtype=dtype, count=count)
            xyz = np.empty((records.shape[0], 3), dtype=np.float64)
            xyz[:, 0] = records["X"] * header["scale"][0] + header["offset"][0]
            xyz[:, 1] = records["Y"] * header["scale"][1] + header["offset"][1]
            xyz[:, 2] = records["Z"] * header["scale"][2] + header["offset"][2]
            yield transform_points_xyz(xyz, axis_transform)
            remaining -= count


def sample_las_xyz(path, max_points, chunk_size=1_000_000, axis_transform="zup-to-yup"):
    chunks = []
    collected = 0
    for xyz in iter_las_xyz(path, chunk_size=chunk_size, axis_transform=axis_transform):
        if collected + xyz.shape[0] <= max_points:
            chunks.append(xyz.copy())
            collected += xyz.shape[0]
        else:
            need = max_points - collected
            if need > 0:
                chunks.append(xyz[:need].copy())
            break
    if not chunks:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(chunks, axis=0)

