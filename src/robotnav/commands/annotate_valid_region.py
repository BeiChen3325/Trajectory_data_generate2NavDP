"""Interactive OpenCV viewer for annotating a world-coordinate navigation ROI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from robotnav.navigation.scene.artifact import load_scene_artifact
from robotnav.navigation.scene.occupancy_map import grid_to_world, world_to_grid
from robotnav.navigation.trajectory.valid_region import ValidRegion, save_valid_region

INITIAL_WINDOW_SIZE = (1100, 800)
VIEW_PADDING_PX = 24
ZOOM_FACTOR = 1.25
MIN_SCALE = 0.02
MAX_SCALE = 64.0
BUTTON_SIZE_PX = 32
BUTTON_GAP_PX = 4


@dataclass
class Camera:
    """Affine map-grid to screen transform for the current viewer viewport."""

    scale: float
    offset_x: float
    offset_y: float


def fit_camera(map_shape: tuple[int, int], viewport_size: tuple[int, int]) -> Camera:
    """Fit the complete map centrally into the viewport, preserving aspect ratio."""
    map_height, map_width = map_shape
    viewport_width, viewport_height = viewport_size
    usable_width = max(1, viewport_width - 2 * VIEW_PADDING_PX)
    usable_height = max(1, viewport_height - 2 * VIEW_PADDING_PX)
    scale = min(usable_width / map_width, usable_height / map_height)
    return Camera(
        scale=scale,
        offset_x=(viewport_width - map_width * scale) * 0.5,
        offset_y=(viewport_height - map_height * scale) * 0.5,
    )


def grid_to_screen(grid_xy: np.ndarray, camera: Camera) -> np.ndarray:
    grid_xy = np.asarray(grid_xy, dtype=np.float64)
    return grid_xy * camera.scale + np.array([camera.offset_x, camera.offset_y])


def screen_to_grid(screen_xy: tuple[float, float], camera: Camera) -> np.ndarray:
    return (
        np.asarray(screen_xy, dtype=np.float64) - (camera.offset_x, camera.offset_y)
    ) / camera.scale


def zoom_at(camera: Camera, screen_xy: tuple[int, int], factor: float) -> None:
    """Zoom while keeping the map location below the mouse fixed on screen."""
    grid_xy = screen_to_grid(screen_xy, camera)
    camera.scale = float(np.clip(camera.scale * factor, MIN_SCALE, MAX_SCALE))
    camera.offset_x = float(screen_xy[0] - grid_xy[0] * camera.scale)
    camera.offset_y = float(screen_xy[1] - grid_xy[1] * camera.scale)


def pan_camera(camera: Camera, direction: str, pan_step: float) -> None:
    """Apply one screen-space directional pan to the viewport camera."""
    if direction == "up":
        camera.offset_y += pan_step
    elif direction == "down":
        camera.offset_y -= pan_step
    elif direction == "left":
        camera.offset_x += pan_step
    elif direction == "right":
        camera.offset_x -= pan_step
    else:
        raise ValueError(f"Unknown pan direction: {direction}")


def _base_image(blocked: np.ndarray) -> np.ndarray:
    image = np.full((*blocked.shape, 3), 245, dtype=np.uint8)
    image[blocked] = (45, 45, 45)
    return image


def _button_rects(viewport_size: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    width, _ = viewport_size
    size = BUTTON_SIZE_PX
    gap = BUTTON_GAP_PX
    center_x = width - VIEW_PADDING_PX - size * 2 - gap
    top_y = VIEW_PADDING_PX
    return {
        "up": (center_x + size + gap, top_y, size, size),
        "left": (center_x, top_y + size + gap, size, size),
        "down": (center_x + size + gap, top_y + size + gap, size, size),
        "right": (center_x + 2 * (size + gap), top_y + size + gap, size, size),
    }


def _button_at(screen_xy: tuple[int, int], viewport_size: tuple[int, int]) -> str | None:
    x, y = screen_xy
    for direction, (left, top, width, height) in _button_rects(viewport_size).items():
        if left <= x < left + width and top <= y < top + height:
            return direction
    return None


def _draw_controls(canvas: np.ndarray) -> None:
    arrows = {"up": "^", "down": "v", "left": "<", "right": ">"}
    for direction, (x, y, width, height) in _button_rects(
        (canvas.shape[1], canvas.shape[0])
    ).items():
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (30, 30, 30), -1)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (210, 210, 210), 1)
        cv2.putText(
            canvas,
            arrows[direction],
            (x + 10, y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _viewport_size(window: str, fallback: tuple[int, int]) -> tuple[int, int]:
    try:
        _, _, width, height = cv2.getWindowImageRect(window)
    except cv2.error:
        return fallback
    return (width, height) if width > 0 and height > 0 else fallback


def _render(
    base_image: np.ndarray,
    points_world: list[np.ndarray],
    model_spec: dict[str, object],
    camera: Camera,
    viewport_size: tuple[int, int],
) -> np.ndarray:
    width, height = viewport_size
    canvas = cv2.warpAffine(
        base_image,
        np.array([[camera.scale, 0.0, camera.offset_x], [0.0, camera.scale, camera.offset_y]]),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(25, 25, 25),
    )
    if points_world:
        grid_xy = world_to_grid(np.asarray(points_world), model_spec)
        screen_xy = np.rint(grid_to_screen(grid_xy, camera)).astype(np.int32)
        cv2.polylines(canvas, [screen_xy.reshape(-1, 1, 2)], len(points_world) >= 3, (0, 180, 0), 2)
        for point in screen_xy:
            cv2.circle(canvas, tuple(point), 4, (0, 0, 255), -1)
    _draw_controls(canvas)
    cv2.putText(
        canvas,
        "Left: point | Middle drag: pan | Space: pan mode | Wheel: zoom | u: undo | c: save | q: cancel",
        (VIEW_PADDING_PX, max(20, height - VIEW_PADDING_PX)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a valid navigation polygon on a scene map.")
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Output valid_region.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = load_scene_artifact(args.scene_dir)
    model = scene.model
    map_height, map_width = model.planning_blocked.shape
    base_image = _base_image(model.planning_blocked)
    points_world: list[np.ndarray] = []
    window = "Valid navigation region"
    viewport_size = INITIAL_WINDOW_SIZE
    camera = fit_camera((map_height, map_width), viewport_size)
    pan_anchor: tuple[int, int] | None = None
    pan_mode = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    debug_map_path = args.output.parent / "debug_map.png"
    if not cv2.imwrite(str(debug_map_path), base_image):
        raise RuntimeError(f"Failed to save debug map: {debug_map_path}")

    print(f"map file path: {scene.model_path.resolve()}")
    print(f"map size: width={map_width}, height={map_height}")
    print(
        f"planning_blocked: shape={model.planning_blocked.shape}, dtype={model.planning_blocked.dtype}"
    )
    print(f"initial scale: {camera.scale:.6f}")
    print(f"origin: {model.origin_xz.tolist()}")
    print(f"resolution: {model.resolution_m}")
    print(
        "world bounds: "
        f"x_min={model.origin_xz[0]}, x_max={model.max_xz[0]}, "
        f"z_min={model.origin_xz[1]}, z_max={model.max_xz[1]}"
    )
    print(f"debug map: {debug_map_path.resolve()}")

    def pan(direction: str) -> None:
        pan_step = max(40.0, min(viewport_size) * 0.15)
        pan_camera(camera, direction, pan_step)
        print(f"camera.offset_x={camera.offset_x:.2f}, camera.offset_y={camera.offset_y:.2f}")

    def on_mouse(event: int, x: int, y: int, flags: int, _param: object) -> None:
        nonlocal pan_anchor
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = cv2.getMouseWheelDelta(flags)
            if delta:
                zoom_at(camera, (x, y), ZOOM_FACTOR if delta > 0 else 1.0 / ZOOM_FACTOR)
            return
        if event == cv2.EVENT_MBUTTONDOWN:
            pan_anchor = (x, y)
            return
        if (
            event == cv2.EVENT_MOUSEMOVE
            and pan_anchor is not None
            and (flags & cv2.EVENT_FLAG_MBUTTON or flags & cv2.EVENT_FLAG_LBUTTON)
        ):
            camera.offset_x += x - pan_anchor[0]
            camera.offset_y += y - pan_anchor[1]
            pan_anchor = (x, y)
            print(f"camera.offset_x={camera.offset_x:.2f}, camera.offset_y={camera.offset_y:.2f}")
            return
        if event == cv2.EVENT_MBUTTONUP or event == cv2.EVENT_LBUTTONUP:
            pan_anchor = None
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        direction = _button_at((x, y), viewport_size)
        if direction is not None:
            pan(direction)
            return
        if pan_mode:
            pan_anchor = (x, y)
            return
        grid_xy = np.rint(screen_to_grid((x, y), camera)).astype(np.int64)
        if 0 <= grid_xy[0] < map_width and 0 <= grid_xy[1] < map_height:
            points_world.append(grid_to_world(grid_xy, model.spec))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, *INITIAL_WINDOW_SIZE)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            observed_size = _viewport_size(window, viewport_size)
            if observed_size != viewport_size:
                viewport_size = observed_size
            cv2.imshow(
                window,
                _render(base_image, points_world, model.spec, camera, viewport_size),
            )
            key = cv2.waitKey(16) & 0xFF
            if key == ord(" "):
                # OpenCV HighGUI exposes key presses but no portable key-up event;
                # Space therefore toggles a left-drag pan mode. Middle-button drag
                # is always available as the backend-independent direct gesture.
                pan_mode = not pan_mode
                print(f"pan mode: {'enabled' if pan_mode else 'disabled'}")
            elif key == ord("u") and points_world:
                points_world.pop()
            elif key == ord("c"):
                if len(points_world) < 3:
                    print("At least three points are required before saving.")
                    continue
                save_valid_region(
                    args.output,
                    ValidRegion(
                        model.resolution_m,
                        model.origin_xz.copy(),
                        np.asarray(points_world),
                    ),
                )
                print(f"Saved {args.output}")
                return
            elif key in (ord("q"), 27):
                return
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
