"""Canonical raw/optimized skeleton state, reports, and debug artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def binary_skeleton(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8) * 255


def skeleton_endpoint_count(skeleton: np.ndarray) -> int:
    binary = (binary_skeleton(skeleton) > 0).astype(np.uint8)
    neighbors = cv2.filter2D(
        binary, cv2.CV_16S, np.ones((3, 3), dtype=np.int16),
        borderType=cv2.BORDER_CONSTANT,
    ) - binary
    return int(np.count_nonzero((binary > 0) & (neighbors == 1)))


def skeleton_component_count(skeleton: np.ndarray) -> int:
    binary = (binary_skeleton(skeleton) > 0).astype(np.uint8)
    if not np.any(binary):
        return 0
    count, _ = cv2.connectedComponents(binary, connectivity=8)
    return max(0, int(count) - 1)


def _overlay(
    processed_mask: np.ndarray,
    skeleton: np.ndarray,
    image_rgb: np.ndarray | None,
    color: tuple[int, int, int],
) -> np.ndarray:
    mask = np.asarray(processed_mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    height, width = mask.shape[:2]
    if image_rgb is not None and np.asarray(image_rgb).shape[:2] == (height, width):
        base = np.asarray(image_rgb)[..., :3].astype(np.uint8).copy()
    else:
        gray = np.where(mask > 0, 80, 20).astype(np.uint8)
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    line = binary_skeleton(skeleton) > 0
    # Dilate only for readable debug overlays; saved skeleton remains 1 px.
    line = cv2.dilate(line.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    base[line] = np.asarray(color, dtype=np.uint8)
    return base


def build_skeleton_optimize_report(
    processed_mask: np.ndarray,
    raw_skeleton: np.ndarray,
    optimized_skeleton: np.ndarray,
    *,
    min_branch_length: int,
    min_center_dist: float,
    endpoint_connect_distance: float,
    skeleton_state_input: str,
    skeleton_state_output: str = "optimized",
) -> dict:
    raw = binary_skeleton(raw_skeleton)
    optimized = binary_skeleton(optimized_skeleton)
    raw_pixels = int(np.count_nonzero(raw))
    optimized_pixels = int(np.count_nonzero(optimized))
    removed = max(0, raw_pixels - optimized_pixels)
    return {
        "processed_mask_pixels": int(np.count_nonzero(processed_mask)),
        "raw_skeleton_pixels": raw_pixels,
        "optimized_skeleton_pixels": optimized_pixels,
        "removed_pixels": removed,
        "removed_ratio": round(removed / raw_pixels if raw_pixels else 0.0, 8),
        "endpoint_count_before": skeleton_endpoint_count(raw),
        "endpoint_count_after": skeleton_endpoint_count(optimized),
        "connected_components_before": skeleton_component_count(raw),
        "connected_components_after": skeleton_component_count(optimized),
        "min_branch_length": int(min_branch_length),
        "min_center_dist": float(min_center_dist),
        "endpoint_connect_distance": float(endpoint_connect_distance),
        "skeleton_state_input": str(skeleton_state_input),
        "skeleton_state_output": str(skeleton_state_output),
    }


def save_skeleton_artifacts(
    output_dir: str | Path,
    processed_mask: np.ndarray,
    raw_skeleton: np.ndarray,
    optimized_skeleton: np.ndarray,
    report: dict,
    image_rgb: np.ndarray | None = None,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = binary_skeleton(raw_skeleton)
    optimized = binary_skeleton(optimized_skeleton)
    paths = {
        "raw_skeleton": out / "raw_skeleton.png",
        "optimized_skeleton": out / "optimized_skeleton.png",
        "overlay_before": out / "skeleton_overlay_before.png",
        "overlay_after": out / "skeleton_overlay_after.png",
        "report": out / "skeleton_optimize_report.json",
    }
    cv2.imwrite(str(paths["raw_skeleton"]), raw)
    cv2.imwrite(str(paths["optimized_skeleton"]), optimized)
    before = _overlay(processed_mask, raw, image_rgb, (255, 215, 0))
    after = _overlay(processed_mask, optimized, image_rgb, (0, 255, 255))
    cv2.imwrite(str(paths["overlay_before"]), cv2.cvtColor(before, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(paths["overlay_after"]), cv2.cvtColor(after, cv2.COLOR_RGB2BGR))
    paths["report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {key: str(path) for key, path in paths.items()}
