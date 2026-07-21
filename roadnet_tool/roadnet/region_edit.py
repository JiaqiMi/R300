"""Stable primitives for ROI/Ignore and mask brush editing.

All coordinates handled by this module are original image pixel coordinates.
The GUI is responsible for converting scene/display coordinates before calling
these functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
from uuid import uuid4

import cv2
import numpy as np


@dataclass
class PolygonRegion:
    id: str
    region_type: str  # "roi" | "ignore"
    points: list[tuple[float, float]] = field(default_factory=list)
    enabled: bool = True
    name: str = ""

    @classmethod
    def create(
        cls,
        region_type: str,
        points: Iterable[Sequence[float]],
        *,
        name: str = "",
    ) -> "PolygonRegion":
        if region_type not in {"roi", "ignore"}:
            raise ValueError(f"unsupported region_type: {region_type!r}")
        clean_points = [(float(p[0]), float(p[1])) for p in points]
        if len(clean_points) < 3:
            raise ValueError("a polygon region requires at least three points")
        return cls(
            id=f"{region_type}_{uuid4().hex[:12]}",
            region_type=region_type,
            points=clean_points,
            enabled=True,
            name=name,
        )


def ensure_mask_uint8(mask: np.ndarray, *, copy: bool = False) -> np.ndarray:
    """Return a writable, single-channel uint8 mask without hiding bad input."""
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"mask must be numpy.ndarray, got {type(mask).__name__}")
    if mask.dtype == object:
        raise TypeError("mask dtype=object is invalid")
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, 3]
        elif mask.shape[2] >= 3:
            mask = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            mask = mask[:, :, 0]
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError(f"mask must be a non-empty 2D array, shape={mask.shape}")
    if mask.dtype != np.uint8:
        if np.issubdtype(mask.dtype, np.floating) and float(np.nanmax(mask)) <= 1.0:
            mask = mask * 255.0
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    elif copy:
        mask = mask.copy()
    if not mask.flags.writeable:
        mask = mask.copy()
    return mask


def ensure_mask_image_size(
    mask: np.ndarray, image_size: tuple[int, int]
) -> np.ndarray:
    """Normalize a mask and align it to the original image pixel grid."""
    normalized = ensure_mask_uint8(mask)
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid original image size: {image_size}")
    if normalized.shape != (height, width):
        print(
            f"[RegionEdit] resize mask from {normalized.shape} to "
            f"original image size {(height, width)} using nearest-neighbor"
        )
        normalized = cv2.resize(
            normalized, (width, height), interpolation=cv2.INTER_NEAREST
        )
    return normalized


def build_region_mask(
    shape: tuple[int, int],
    regions: Iterable[PolygonRegion],
    region_type: str,
) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid mask shape: {shape}")
    result = np.zeros((h, w), dtype=np.uint8)
    for region in regions:
        if not region.enabled or region.region_type != region_type:
            continue
        if len(region.points) < 3:
            continue
        points = np.rint(np.asarray(region.points, dtype=np.float64)).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, w - 1)
        points[:, 1] = np.clip(points[:, 1], 0, h - 1)
        cv2.fillPoly(result, [points.reshape((-1, 1, 2))], 255)
    return result


def apply_roi_regions(
    mask: np.ndarray, regions: Iterable[PolygonRegion]
) -> tuple[np.ndarray, int]:
    edited = ensure_mask_uint8(mask, copy=True)
    roi_mask = build_region_mask(edited.shape, regions, "roi")
    before = int(np.count_nonzero(edited))
    edited[roi_mask == 0] = 0
    return edited, before - int(np.count_nonzero(edited))


def apply_ignore_regions(
    mask: np.ndarray, regions: Iterable[PolygonRegion]
) -> tuple[np.ndarray, int]:
    edited = ensure_mask_uint8(mask, copy=True)
    ignore_mask = build_region_mask(edited.shape, regions, "ignore")
    before = int(np.count_nonzero(edited))
    edited[ignore_mask > 0] = 0
    return edited, before - int(np.count_nonzero(edited))


def paint_mask_segment(
    mask: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    radius: int,
    erase: bool = False,
) -> np.ndarray:
    edited = ensure_mask_uint8(mask)
    radius = max(1, min(100, int(radius)))
    value = 0 if erase else 255
    p1 = (int(round(start[0])), int(round(start[1])))
    p2 = (int(round(end[0])), int(round(end[1])))
    cv2.line(edited, p1, p2, value, max(1, radius * 2), cv2.LINE_8)
    cv2.circle(edited, p2, radius, value, -1, cv2.LINE_8)
    return edited


def save_mask_png_verified(mask: np.ndarray, output_path) -> Path:
    """Save a mask as PNG and immediately verify it can be read back."""
    normalized = ensure_mask_uint8(mask)
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", normalized)
    if not ok:
        raise OSError("OpenCV could not encode the current mask as PNG")
    encoded.tofile(str(path))
    if not path.is_file() or path.stat().st_size <= 0:
        raise OSError(f"saved mask does not exist or is empty: {path}")
    decoded = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if decoded is None or decoded.shape != normalized.shape:
        raise OSError(
            f"saved mask verification failed: expected={normalized.shape}, "
            f"actual={None if decoded is None else decoded.shape}"
        )
    return path
