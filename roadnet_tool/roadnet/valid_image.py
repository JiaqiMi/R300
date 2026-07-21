"""Shared black-border / invalid-image masking for all road-network stages.

Only large near-black components connected to an image border are invalid.
Dark pixels inside the image remain valid so shadows and dark roads cannot be
punched out of the road mask or skeleton.
"""

from __future__ import annotations

import cv2
import json
import numpy as np
from pathlib import Path


DEFAULT_MIN_BLACK_COMPONENT_AREA = 4096


def analyze_valid_image_mask(
    image_rgb: np.ndarray,
    black_threshold: int = 10,
    min_black_component_area: int = DEFAULT_MIN_BLACK_COMPONENT_AREA,
    road_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Build a valid mask from large border-connected black components.

    A flood-fill is used instead of a full int32 connected-component label
    image.  This keeps the peak memory reasonable for very large imagery.
    """
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"RGB 图像格式无效: shape={image.shape}")
    threshold = max(0, min(255, int(black_threshold)))
    min_area = max(1, int(min_black_component_area))
    height, width = image.shape[:2]
    if threshold == 0:
        candidate = np.zeros((height, width), dtype=np.uint8)
    else:
        candidate = cv2.inRange(
            image[..., :3],
            np.array((0, 0, 0), dtype=np.uint8),
            np.array((threshold - 1,) * 3, dtype=np.uint8),
        )

    # `working` retains internal black components as 255. Border components
    # are visited once, marked temporarily as 128, then cleared to 0.
    working = candidate.copy()
    invalid = np.zeros((height, width), dtype=np.uint8)
    border_seeds = []
    if width and height:
        border_seeds.extend((x, 0) for x in range(width))
        if height > 1:
            border_seeds.extend((x, height - 1) for x in range(width))
        border_seeds.extend((0, y) for y in range(1, max(1, height - 1)))
        if width > 1:
            border_seeds.extend((width - 1, y) for y in range(1, max(1, height - 1)))

    kept_component_count = 0
    rejected_small_border_components = 0
    for x, y in border_seeds:
        if working[y, x] != 255:
            continue
        area, _, _, rect = cv2.floodFill(
            working, None, (int(x), int(y)), 128, flags=8
        )
        rx, ry, rw, rh = rect
        component = working[ry:ry + rh, rx:rx + rw] == 128
        if int(area) >= min_area:
            target = invalid[ry:ry + rh, rx:rx + rw]
            target[component] = 255
            kept_component_count += 1
        else:
            rejected_small_border_components += 1
        working_view = working[ry:ry + rh, rx:rx + rw]
        working_view[component] = 0

    valid = cv2.bitwise_not(invalid)
    removed_road = 0
    if road_mask is not None:
        road = np.asarray(road_mask)
        if road.ndim == 3:
            road = road[..., 0]
        if road.shape != invalid.shape:
            road = cv2.resize(
                road.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            )
        removed_road = int(np.count_nonzero((road > 0) & (invalid > 0)))
    black_pixels = int(np.count_nonzero(candidate))
    invalid_pixels = int(np.count_nonzero(invalid))
    report = {
        "black_threshold": threshold,
        "min_black_component_area": min_area,
        "black_candidate_pixels": black_pixels,
        "border_connected_invalid_pixels": invalid_pixels,
        "internal_black_pixels_kept": black_pixels - invalid_pixels,
        "removed_road_pixels_estimate": removed_road,
        "valid_area_ratio": round(
            float(np.count_nonzero(valid)) / float(valid.size) if valid.size else 0.0,
            8,
        ),
        "invalid_component_count": kept_component_count,
        "small_border_components_kept": rejected_small_border_components,
    }
    return valid, report


def compute_valid_image_mask(
    image_rgb: np.ndarray,
    black_threshold: int = 10,
    min_black_component_area: int = DEFAULT_MIN_BLACK_COMPONENT_AREA,
) -> np.ndarray:
    """Compatibility wrapper returning only the uint8 valid mask."""
    valid, _ = analyze_valid_image_mask(
        image_rgb, black_threshold, min_black_component_area
    )
    return valid


def save_valid_mask_outputs(
    output_dir: str | Path,
    valid_image_mask: np.ndarray,
    report: dict,
) -> tuple[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mask_path = out / "valid_image_mask.png"
    report_path = out / "valid_mask_report.json"
    if not cv2.imwrite(str(mask_path), np.asarray(valid_image_mask, dtype=np.uint8)):
        raise IOError(f"无法保存 {mask_path}")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(mask_path), str(report_path)


def apply_valid_image_mask(mask: np.ndarray, valid_image_mask: np.ndarray) -> np.ndarray:
    """Return a uint8 mask cleared outside the valid image area."""
    result = np.asarray(mask, dtype=np.uint8).copy()
    valid = np.asarray(valid_image_mask)
    if valid.ndim == 3:
        valid = valid[..., 0]
    if valid.shape != result.shape[:2]:
        valid = cv2.resize(valid.astype(np.uint8),
                           (result.shape[1], result.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    result[valid == 0] = 0
    return result


def valid_area_ratio(valid_image_mask: np.ndarray) -> float:
    valid = np.asarray(valid_image_mask)
    return float(np.count_nonzero(valid)) / float(valid.size) if valid.size else 0.0
