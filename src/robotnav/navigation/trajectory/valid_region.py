"""World-coordinate valid-navigation ROI loading, rasterization, and annotation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from robotnav.navigation.scene.artifact import SceneArtifact


@dataclass(frozen=True)
class ValidRegion:
    resolution_m: float
    origin_xz: np.ndarray
    polygon_xz: np.ndarray


def _number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
        raise ValueError(f"valid region {name} must be a finite number")
    return float(value)


def _pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"valid region {name} must be an [x, z] pair")
    return (_number(value[0], f"{name}[0]"), _number(value[1], f"{name}[1]"))


def load_valid_region(path: Path) -> ValidRegion:
    """Load the intentionally small, portable world-coordinate ROI YAML format."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"resolution", "origin", "polygon"}:
        raise ValueError("valid region YAML must contain exactly resolution, origin, and polygon")
    resolution_m = _number(raw["resolution"], "resolution")
    if resolution_m <= 0:
        raise ValueError("valid region resolution must be positive")
    origin_xz = np.asarray(_pair(raw["origin"], "origin"), dtype=np.float64)
    polygon = raw["polygon"]
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError("valid region polygon must contain at least three [x, z] points")
    polygon_xz = np.asarray(
        [_pair(point, f"polygon[{index}]") for index, point in enumerate(polygon)]
    )
    if abs(cv2.contourArea(polygon_xz.astype(np.float32))) <= 1e-9:
        raise ValueError("valid region polygon has zero area")
    return ValidRegion(resolution_m, origin_xz, polygon_xz)


def rasterize_valid_region(region: ValidRegion, scene: SceneArtifact) -> np.ndarray:
    """Return an ROI mask whose pixels correspond to navigation-map cell centers."""
    model = scene.model
    if not np.isclose(region.resolution_m, model.resolution_m, rtol=0.0, atol=1e-9):
        raise ValueError(
            "valid region resolution does not match the scene map "
            f"({region.resolution_m} != {model.resolution_m})"
        )
    if not np.allclose(region.origin_xz, model.origin_xz, rtol=0.0, atol=1e-9):
        raise ValueError("valid region origin does not match the scene map origin")
    polygon_xy = np.rint((region.polygon_xz - model.origin_xz) / model.resolution_m - 0.5).astype(
        np.int32
    )
    mask = np.zeros(model.planning_blocked.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_xy.reshape(-1, 1, 2)], 1, lineType=cv2.LINE_8)
    return mask.astype(bool)


def erode_valid_region(
    mask: np.ndarray, *, resolution_m: float, safety_distance_m: float
) -> tuple[np.ndarray, int]:
    """Inset the ROI by a circular robot safety boundary expressed in metres."""
    if safety_distance_m < 0:
        raise ValueError("valid region safety distance must be non-negative")
    erosion_cells = int(np.ceil(safety_distance_m / resolution_m))
    if erosion_cells == 0:
        return mask.astype(bool, copy=True), 0
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * erosion_cells + 1, 2 * erosion_cells + 1)
    )
    return cv2.erode(mask.astype(np.uint8), kernel) > 0, erosion_cells


def build_valid_sampling_cells(
    scene: SceneArtifact,
    region: ValidRegion,
    *,
    robot_radius_m: float,
    safety_margin_m: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build raw ROI, eroded ROI, and free cells eligible for endpoint sampling."""
    raw_mask = rasterize_valid_region(region, scene)
    eroded_mask, erosion_cells = erode_valid_region(
        raw_mask,
        resolution_m=scene.model.resolution_m,
        safety_distance_m=robot_radius_m + safety_margin_m,
    )
    valid_sampling_cells = (~scene.model.planning_blocked) & eroded_mask
    if not np.any(valid_sampling_cells):
        raise ValueError("valid region contains no free cells after safety erosion")
    return raw_mask, valid_sampling_cells, erosion_cells


def save_valid_region(path: Path, region: ValidRegion) -> None:
    payload = {
        "resolution": float(region.resolution_m),
        "origin": region.origin_xz.tolist(),
        "polygon": region.polygon_xz.tolist(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
