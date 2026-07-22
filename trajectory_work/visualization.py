from pathlib import Path

import cv2
import numpy as np


def colorize_planning_map(cleaned, inflated, blocked, distance_m):
    h, w = inflated.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    dist_norm = np.clip(distance_m / max(float(np.percentile(distance_m, 98.0)), 0.1), 0, 1)
    canvas[:, :, 1] = (dist_norm * 120).astype(np.uint8)
    canvas[blocked] = (25, 25, 25)
    canvas[cleaned] = (80, 80, 220)
    canvas[inflated] = (40, 40, 120)
    return canvas


def draw_path_debug(output_path, cleaned, inflated, blocked, distance_m, astar_path_xy, shortcut_xy, smooth_xy=None, world_to_grid_fn=None):
    canvas = colorize_planning_map(cleaned, inflated, blocked, distance_m)
    if astar_path_xy is not None and len(astar_path_xy) > 1:
        draw_grid_polyline(canvas, astar_path_xy, (255, 180, 60), 1)
    if shortcut_xy is not None and len(shortcut_xy) > 1:
        draw_grid_polyline(canvas, shortcut_xy, (255, 255, 255), 2)
    if smooth_xy is not None and len(smooth_xy) > 1 and world_to_grid_fn is not None:
        smooth_grid = world_to_grid_fn(smooth_xy)
        draw_grid_polyline(canvas, smooth_grid, (40, 230, 255), 2)

    # Image coordinates already use grid y downward. Flip vertically for a more
    # map-like preview where larger Z appears upward.
    canvas = cv2.flip(canvas, 0)
    scale = max(1, min(8, 900 // max(canvas.shape[0], canvas.shape[1], 1)))
    if scale > 1:
        canvas = cv2.resize(canvas, (canvas.shape[1] * scale, canvas.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(Path(output_path)), canvas)


def draw_grid_polyline(canvas, points_xy, color, thickness):
    points = np.asarray(points_xy, dtype=np.int32)
    h = canvas.shape[0]
    pts = points.copy()
    pts[:, 1] = h - 1 - pts[:, 1]
    cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=thickness)
    cv2.circle(canvas, tuple(pts[0]), 4, (0, 255, 0), -1)
    cv2.circle(canvas, tuple(pts[-1]), 4, (0, 0, 255), -1)
