"""Reusable A* and endpoint-selection primitives."""

import heapq
import math

import cv2
import numpy as np

NEIGHBORS_8 = [
    (-1, -1, math.sqrt(2.0)),
    (0, -1, 1.0),
    (1, -1, math.sqrt(2.0)),
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (-1, 1, math.sqrt(2.0)),
    (0, 1, 1.0),
    (1, 1, math.sqrt(2.0)),
]


def nearest_free_cell(cell, free_mask, max_radius=80):
    x, y = int(cell[0]), int(cell[1])
    h, w = free_mask.shape
    if 0 <= x < w and 0 <= y < h and free_mask[y, x]:
        return np.array([x, y], dtype=np.int64)
    best = None
    best_d2 = None
    for r in range(1, max_radius + 1):
        y0, y1 = max(0, y - r), min(h - 1, y + r)
        x0, x1 = max(0, x - r), min(w - 1, x + r)
        candidates = []
        candidates.extend([(xx, y0) for xx in range(x0, x1 + 1)])
        candidates.extend([(xx, y1) for xx in range(x0, x1 + 1)])
        candidates.extend([(x0, yy) for yy in range(y0 + 1, y1)])
        candidates.extend([(x1, yy) for yy in range(y0 + 1, y1)])
        for xx, yy in candidates:
            if free_mask[yy, xx]:
                d2 = (xx - x) ** 2 + (yy - y) ** 2
                if best is None or best_d2 is None or d2 < best_d2:
                    best = np.array([xx, yy], dtype=np.int64)
                    best_d2 = d2
        if best is not None:
            return best
    raise ValueError(f"No free cell found near {cell}.")


def choose_auto_start_goal(free_mask, distance_m, spec, min_distance_m, seed=7):
    component = largest_free_component(free_mask)
    candidates = np.argwhere(component & (distance_m > max(0.15, spec["resolution"] * 2.0)))
    if candidates.shape[0] < 2:
        candidates = np.argwhere(component)
    if candidates.shape[0] < 2:
        raise ValueError("Not enough free cells to choose start and goal.")

    rng = np.random.default_rng(seed)
    scored = candidates[
        np.argsort(distance_m[candidates[:, 0], candidates[:, 1]])[-min(5000, len(candidates)) :]
    ]
    first = scored[rng.integers(0, scored.shape[0])]
    first_xy = np.array([first[1], first[0]], dtype=np.float64)
    min_cells = min_distance_m / spec["resolution"]
    diffs = scored[:, [1, 0]].astype(np.float64) - first_xy
    far = scored[np.linalg.norm(diffs, axis=1) >= min_cells]
    if far.size == 0:
        diffs = candidates[:, [1, 0]].astype(np.float64) - first_xy
        far = candidates[[int(np.argmax(np.linalg.norm(diffs, axis=1)))]]
    # Keep the original clearance bias while allowing deterministic variation
    # across route seeds; always taking argmax collapses many batches to a few
    # corner goals on maps with uniform clearance.
    second_pool = far[np.argsort(distance_m[far[:, 0], far[:, 1]])[-min(5000, far.shape[0]) :]]
    second = second_pool[rng.integers(0, second_pool.shape[0])]
    return np.array([first[1], first[0]], dtype=np.int64), np.array(
        [second[1], second[0]], dtype=np.int64
    )


def largest_free_component(free_mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        free_mask.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return np.zeros_like(free_mask, dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas) + 1)
    return labels == label


def astar(
    start_xy,
    goal_xy,
    obstacle_mask,
    distance_m=None,
    resolution=0.08,
    obstacle_cost_weight=0.8,
    obstacle_cost_power=1.5,
):
    h, w = obstacle_mask.shape
    free = ~obstacle_mask
    start = tuple(map(int, start_xy))
    goal = tuple(map(int, goal_xy))
    if not free[start[1], start[0]]:
        raise ValueError(f"Start cell is not free: {start}")
    if not free[goal[1], goal[0]]:
        raise ValueError(f"Goal cell is not free: {goal}")

    g_score = np.full((h, w), np.inf, dtype=np.float32)
    parent_x = np.full((h, w), -1, dtype=np.int32)
    parent_y = np.full((h, w), -1, dtype=np.int32)
    closed = np.zeros((h, w), dtype=bool)

    def heuristic(x, y):
        return math.hypot(goal[0] - x, goal[1] - y)

    def clearance_cost(x, y):
        if distance_m is None or obstacle_cost_weight <= 0:
            return 0.0
        d = max(float(distance_m[y, x]), resolution)
        return obstacle_cost_weight / ((d + 0.05) ** obstacle_cost_power)

    g_score[start[1], start[0]] = 0.0
    heap = [(heuristic(*start), 0.0, start[0], start[1])]

    while heap:
        _, g, x, y = heapq.heappop(heap)
        if closed[y, x]:
            continue
        closed[y, x] = True
        if (x, y) == goal:
            return reconstruct_path(parent_x, parent_y, start, goal)

        for dx, dy, step_cost in NEIGHBORS_8:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if closed[ny, nx] or not free[ny, nx]:
                continue
            tentative = g + step_cost + clearance_cost(nx, ny)
            if tentative < g_score[ny, nx]:
                g_score[ny, nx] = tentative
                parent_x[ny, nx] = x
                parent_y[ny, nx] = y
                heapq.heappush(heap, (tentative + heuristic(nx, ny), tentative, nx, ny))

    raise ValueError("A* failed to find a path between start and goal.")


def reconstruct_path(parent_x, parent_y, start, goal):
    path = []
    current = goal
    while current != start:
        path.append(current)
        px = parent_x[current[1], current[0]]
        py = parent_y[current[1], current[0]]
        if px < 0 or py < 0:
            raise ValueError("Broken A* parent chain.")
        current = (int(px), int(py))
    path.append(start)
    path.reverse()
    return np.array(path, dtype=np.int64)
