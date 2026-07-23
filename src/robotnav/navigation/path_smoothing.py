from itertools import pairwise

import numpy as np


def line_is_free(a, b, obstacle_mask):
    x0, y0 = map(int, a)
    x1, y1 = map(int, b)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        if obstacle_mask[y, x]:
            return False
        if x == x1 and y == y1:
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def shortcut_path(path_xy, obstacle_mask, passes=120, seed=7):
    if len(path_xy) <= 2:
        return path_xy
    rng = np.random.default_rng(seed)
    path = [tuple(p) for p in path_xy.tolist()]
    for _ in range(passes):
        if len(path) <= 2:
            break
        i = int(rng.integers(0, len(path) - 2))
        j = int(rng.integers(i + 2, len(path)))
        if line_is_free(path[i], path[j], obstacle_mask):
            path = path[: i + 1] + path[j:]
    return np.array(path, dtype=np.int64)


def densify_polyline(points_xz, samples_per_meter=8.0):
    if len(points_xz) <= 1:
        return points_xz
    result = [points_xz[0]]
    for a, b in pairwise(points_xz):
        length = float(np.linalg.norm(b - a))
        steps = max(int(np.ceil(length * samples_per_meter)), 1)
        for k in range(1, steps + 1):
            result.append(a + (b - a) * (k / steps))
    return np.asarray(result, dtype=np.float64)


def chaikin_smooth(points_xz, iterations=2):
    points = np.asarray(points_xz, dtype=np.float64)
    if len(points) <= 2:
        return points
    for _ in range(iterations):
        new_points = [points[0]]
        for a, b in pairwise(points):
            new_points.append(0.75 * a + 0.25 * b)
            new_points.append(0.25 * a + 0.75 * b)
        new_points.append(points[-1])
        points = np.asarray(new_points, dtype=np.float64)
    return points


def path_collides_world(points_xz, obstacle_mask, world_to_grid_fn):
    ij = world_to_grid_fn(points_xz)
    h, w = obstacle_mask.shape
    valid = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    if not np.all(valid):
        return True
    return bool(np.any(obstacle_mask[ij[:, 1], ij[:, 0]]))
