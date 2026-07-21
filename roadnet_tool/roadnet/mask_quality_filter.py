"""Conservative road-mask component screening and safe Ignore suggestions.

Candidates are derived exclusively from foreground connected components in the
road mask.  Background, ROI polygons, and invalid-image areas are never turned
into Ignore candidates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from roadnet.optimized_skeleton import normalize_road_mask, skeletonize_thin
from roadnet.valid_image import apply_valid_image_mask


@dataclass
class MaskQualityFilterConfig:
    min_component_area: int = 80
    warning_component_area: int = 250
    max_estimated_width: float = 80.0
    warning_estimated_width: float = 40.0
    large_area_threshold: int = 10000
    min_large_component_skeleton_ratio: float = 0.012
    min_road_aspect_ratio: float = 1.7
    max_candidate_area_ratio: float = 0.01
    max_total_ignore_area_ratio: float = 0.08
    min_overlap_with_road_mask: float = 0.30
    high_confidence_threshold: float = 0.90
    max_high_confidence_candidate_count: int = 200
    protect_final_graph_buffer_px: int = 15
    protect_planned_path_buffer_px: int = 20
    protect_task_points_buffer_px: int = 30


@dataclass
class MaskQualityFilterResult:
    cleaned_mask: np.ndarray
    candidate_ignore_regions: list[dict]
    report: dict
    mask_before_ignore: Optional[np.ndarray] = None


def _native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_native(v) for v in value]
    return value


def _xy(item, x_names=("x", "x_pixel", "pixel_x", "snapped_x"),
        y_names=("y", "y_pixel", "pixel_y", "snapped_y")):
    for x_name in x_names:
        x = item.get(x_name) if isinstance(item, dict) else getattr(item, x_name, None)
        if x is not None:
            break
    else:
        return None
    for y_name in y_names:
        y = item.get(y_name) if isinstance(item, dict) else getattr(item, y_name, None)
        if y is not None:
            return float(x), float(y)
    return None


def _graph_lists(final_graph):
    if final_graph is None:
        return [], []
    if isinstance(final_graph, dict):
        return list(final_graph.get("nodes", [])), list(final_graph.get("edges", []))
    nodes = getattr(final_graph, "nodes", getattr(final_graph, "_nodes", []))
    edges = getattr(final_graph, "edges", getattr(final_graph, "_edges", []))
    return list(nodes or []), list(edges or [])


def _build_protection_masks(shape, cfg, final_graph=None, planned_path=None,
                            task_points=None, snapped_task_points=None):
    height, width = shape
    graph_mask = np.zeros(shape, dtype=np.uint8)
    path_mask = np.zeros(shape, dtype=np.uint8)
    task_mask = np.zeros(shape, dtype=np.uint8)

    nodes, edges = _graph_lists(final_graph)
    nodes_by_id = {node.get("id"): node for node in nodes}
    graph_width = max(1, int(cfg.protect_final_graph_buffer_px) * 2 + 1)
    for edge in edges:
        if not edge.get("enabled", True):
            continue
        points = edge.get("points_pixel", edge.get("polyline", [])) or []
        if len(points) < 2:
            a, b = nodes_by_id.get(edge.get("start")), nodes_by_id.get(edge.get("end"))
            if a is not None and b is not None:
                points = [_xy(a), _xy(b)]
        points = [point for point in points if point is not None]
        if len(points) >= 2:
            cv2.polylines(graph_mask, [np.rint(points).astype(np.int32).reshape(-1, 1, 2)],
                          False, 255, graph_width, cv2.LINE_8)
    for node in nodes:
        point = _xy(node)
        if point is not None:
            cv2.circle(graph_mask, tuple(np.rint(point).astype(int)),
                       int(cfg.protect_final_graph_buffer_px), 255, -1)

    path = [point for point in (planned_path or []) if point is not None and len(point) >= 2]
    if len(path) >= 2:
        cv2.polylines(
            path_mask, [np.rint(path).astype(np.int32).reshape(-1, 1, 2)], False, 255,
            max(1, int(cfg.protect_planned_path_buffer_px) * 2 + 1), cv2.LINE_8,
        )
    for collection in (task_points or [], snapped_task_points or []):
        for item in collection:
            point = _xy(item)
            if point is not None:
                cv2.circle(task_mask, tuple(np.rint(point).astype(int)),
                           int(cfg.protect_task_points_buffer_px), 255, -1)
    return graph_mask, path_mask, task_mask


def _encode_runs(component_roi: np.ndarray, offset_x: int, offset_y: int) -> list[list[int]]:
    runs = []
    binary = component_roi.astype(bool)
    for row_index, row in enumerate(binary):
        padded = np.pad(row.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        runs.extend([
            [int(offset_y + row_index), int(offset_x + start), int(offset_x + end)]
            for start, end in zip(starts, ends)
        ])
    return runs


def candidate_mask_from_runs(candidate: dict, shape) -> np.ndarray:
    """Rebuild the exact source-component pixels; never fill its bounding box."""
    result = np.zeros(tuple(shape[:2]), dtype=np.uint8)
    height, width = result.shape
    add_candidate_runs_to_mask(result, candidate)
    return result


def add_candidate_runs_to_mask(target: np.ndarray, candidate: dict) -> None:
    """Paint exact candidate pixels into an existing uint8 mask in place."""
    height, width = target.shape[:2]
    for run in candidate.get("pixel_runs", []):
        if not isinstance(run, (list, tuple)) or len(run) != 3:
            continue
        y, x0, x1 = map(int, run)
        if 0 <= y < height:
            x0, x1 = max(0, x0), min(width - 1, x1)
            if x0 <= x1:
                target[y, x0:x1 + 1] = 255


def _component_polygons(component_roi, offset_x, offset_y):
    contours, _ = cv2.findContours(
        component_roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        epsilon = min(2.0, max(0.5, cv2.arcLength(contour, True) * 0.002))
        simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(simplified) >= 3:
            polygons.append([
                [int(point[0] + offset_x), int(point[1] + offset_y)] for point in simplified
            ])
    return polygons


def filter_mask_quality(
    road_mask: np.ndarray,
    processed_mask: Optional[np.ndarray] = None,
    valid_image_mask: Optional[np.ndarray] = None,
    roi_polygons=None,
    final_graph=None,
    planned_path=None,
    task_points: Optional[Iterable] = None,
    snapped_task_points: Optional[Iterable] = None,
    config: Optional[MaskQualityFilterConfig] = None,
    output_dir: Optional[str | Path] = None,
) -> MaskQualityFilterResult:
    """Analyze only ``road_mask > 0`` components and return safe candidates."""
    cfg = config or MaskQualityFilterConfig()
    source = processed_mask if processed_mask is not None else road_mask
    original_mask = np.ascontiguousarray(normalize_road_mask(source))
    mask = original_mask.copy()
    invalid_road_pixels = 0
    if valid_image_mask is not None:
        valid = np.asarray(valid_image_mask)
        if valid.ndim == 3:
            valid = valid[..., 0]
        if valid.shape != mask.shape:
            valid = cv2.resize(valid.astype(np.uint8), (mask.shape[1], mask.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        invalid_road_pixels = int(np.count_nonzero((mask > 0) & (valid == 0)))
        mask = apply_valid_image_mask(mask, valid)
    mask = np.ascontiguousarray((mask > 0).astype(np.uint8) * 255)
    image_area = max(1, mask.size)

    graph_protect, path_protect, task_protect = _build_protection_masks(
        mask.shape, cfg, final_graph, planned_path, task_points, snapped_task_points
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    skeleton = skeletonize_thin(mask)
    cleaned = mask.copy()
    components, candidates = [], []
    kept = rejected = warned = 0
    skipped_large = skipped_low_overlap = 0
    skipped_near_graph = skipped_near_path = skipped_near_task = 0

    for label in range(1, count):
        x, y, width, height, area = [int(v) for v in stats[label]]
        roi_labels = labels[y:y + height, x:x + width]
        component_roi = roi_labels == label
        skeleton_length = int(np.count_nonzero(
            (skeleton[y:y + height, x:x + width] > 0) & component_roi
        ))
        aspect_ratio = float(max(width, height)) / float(max(1, min(width, height)))
        estimated_width = float(area) / float(max(1, skeleton_length))
        extent = float(area) / float(max(1, width * height))
        skeleton_ratio = float(skeleton_length) / float(max(1, area))
        area_ratio = float(area) / float(image_area)

        decision, reason, confidence, score = "keep", "道路形态正常", 0.55, 0.55
        if area < cfg.min_component_area:
            decision, reason, score = "reject", "孤立噪声", 0.05
            confidence = min(0.99, 0.90 + 0.09 * (1.0 - area / max(1, cfg.min_component_area)))
        elif estimated_width > cfg.max_estimated_width:
            decision, reason, score = "reject", "过宽区域", 0.12
            confidence = min(0.97, 0.90 + 0.07 * (estimated_width / cfg.max_estimated_width - 1.0))
        elif area >= cfg.large_area_threshold and skeleton_ratio < cfg.min_large_component_skeleton_ratio:
            decision, reason, confidence, score = "reject", "面积很大但骨架长度过短", 0.92, 0.15
        elif (estimated_width > cfg.warning_estimated_width
              and aspect_ratio < cfg.min_road_aspect_ratio and extent > 0.45):
            decision, reason, confidence, score = "warning", "疑似草地/屋顶或块状误检", 0.72, 0.32
        elif area < cfg.warning_component_area and aspect_ratio < 1.35:
            decision, reason, confidence, score = "warning", "小型非长条连通域", 0.66, 0.40
        else:
            elongation_score = min(1.0, aspect_ratio / 5.0)
            width_score = max(0.0, 1.0 - estimated_width / max(1.0, cfg.max_estimated_width))
            score = min(1.0, 0.45 + 0.35 * elongation_score + 0.20 * width_score)

        # The footprint is the exact road-mask component, therefore its road
        # overlap is normally 1.0. Keep the explicit guard for imported data.
        mask_roi = mask[y:y + height, x:x + width]
        road_overlap = float(np.count_nonzero(component_roi & (mask_roi > 0))) / float(max(1, area))
        near_graph = bool(np.any(graph_protect[y:y + height, x:x + width][component_roi] > 0))
        near_path = bool(np.any(path_protect[y:y + height, x:x + width][component_roi] > 0))
        near_task = bool(np.any(task_protect[y:y + height, x:x + width][component_roi] > 0))
        auto_eligible = decision == "reject"
        safety_reasons = []
        if area_ratio > cfg.max_candidate_area_ratio and decision != "keep":
            decision = "warning"
            confidence = min(confidence, cfg.high_confidence_threshold - 0.01)
            auto_eligible = False
            safety_reasons.append("单个候选面积超过整图 1%，必须人工确认")
            skipped_large += 1
        if road_overlap < cfg.min_overlap_with_road_mask and decision != "keep":
            decision = "drop"
            auto_eligible = False
            skipped_low_overlap += 1
        if (near_graph or near_path or near_task) and decision not in ("keep", "drop"):
            decision = "warning"
            confidence = min(confidence, cfg.high_confidence_threshold - 0.01)
            auto_eligible = False
            safety_reasons.append("候选区域靠近已有路网/规划路径/任务点，需人工确认")
            skipped_near_graph += int(near_graph)
            skipped_near_path += int(near_path)
            skipped_near_task += int(near_task)
        auto_eligible = bool(
            auto_eligible
            and confidence >= cfg.high_confidence_threshold
            and road_overlap >= cfg.min_overlap_with_road_mask
            and area_ratio <= cfg.max_candidate_area_ratio
        )
        full_reason = reason + ("；" + "；".join(safety_reasons) if safety_reasons else "")

        component = {
            "component_id": int(label), "area": area, "bbox": [x, y, width, height],
            "bbox_width": width, "bbox_height": height,
            "aspect_ratio": round(aspect_ratio, 4), "skeleton_length": skeleton_length,
            "estimated_width": round(estimated_width, 4), "extent": round(extent, 4),
            "component_score": round(score, 4), "area_ratio": round(area_ratio, 8),
            "overlap_with_road_mask": round(road_overlap, 4),
            "decision": decision, "keep": decision not in ("reject",),
            "reason": full_reason, "confidence": round(confidence, 4),
            "auto_apply_eligible": auto_eligible,
        }
        components.append(component)
        if decision == "reject":
            cleaned_roi = cleaned[y:y + height, x:x + width]
            cleaned_roi[component_roi] = 0
            rejected += 1
        elif decision == "warning":
            warned += 1
        else:
            kept += 1

        if decision in ("reject", "warning"):
            polygons = _component_polygons(component_roi, x, y)
            candidates.append({
                "id": f"ignore_{len(candidates) + 1:03d}",
                "component_id": int(label),
                "geometry_type": "road_mask_component_runs",
                "bbox": [x, y, width, height],
                "polygons": polygons,
                "polygon": polygons[0] if polygons else [],
                "pixel_runs": _encode_runs(component_roi, x, y),
                "area": area,
                "area_ratio": round(area_ratio, 8),
                "overlap_with_road_mask": round(road_overlap, 4),
                "reason": full_reason,
                "confidence": round(confidence, 4),
                "auto_apply_eligible": auto_eligible,
                "near_final_graph": near_graph,
                "near_planned_path": near_path,
                "near_task_points": near_task,
                "recommended_action": "apply_ignore" if auto_eligible else "review",
                "status": "pending",
            })

    high = [item for item in candidates if item["auto_apply_eligible"]]
    affected_pixels = sum(int(item["area"]) for item in high)
    total_ratio = affected_pixels / float(image_area)
    block_reasons = []
    if total_ratio > cfg.max_total_ignore_area_ratio:
        block_reasons.append("total_ignore_area_ratio_exceeded")
    if len(high) > cfg.max_high_confidence_candidate_count:
        block_reasons.append("high_confidence_candidate_count_exceeded")
    # ``cleaned_mask`` is safe to consume downstream: if batch safety fails it
    # remains identical to the analyzed mask. Otherwise it removes only exact,
    # auto-eligible component pixels (never warnings or background).
    cleaned = mask.copy()
    if not block_reasons:
        safe_ignore = np.zeros(mask.shape, dtype=np.uint8)
        for candidate in high:
            add_candidate_runs_to_mask(safe_ignore, candidate)
        cleaned[safe_ignore > 0] = 0
    report = {
        "input_mask_pixels": int(np.count_nonzero(mask)),
        "cleaned_mask_pixels": int(np.count_nonzero(cleaned)),
        "removed_pixels_hypothetical": int(np.count_nonzero(mask) - np.count_nonzero(cleaned)),
        "component_count": max(0, count - 1),
        "kept_component_count": kept,
        "rejected_component_count": rejected,
        "warning_component_count": warned,
        "candidate_count": len(candidates),
        "candidate_ignore_count": len(candidates),
        "high_confidence_count": len(high),
        "applied_count": 0,
        "skipped_large_area_count": skipped_large,
        "skipped_low_overlap_count": skipped_low_overlap,
        "skipped_near_graph_count": skipped_near_graph,
        "skipped_near_planned_path_count": skipped_near_path,
        "skipped_near_task_points_count": skipped_near_task,
        "total_ignore_area_ratio": round(total_ratio, 8),
        "affected_road_mask_pixels": affected_pixels,
        "auto_apply_blocked": bool(block_reasons),
        "auto_apply_block_reasons": block_reasons,
        "invalid_area_road_pixels_skipped": invalid_road_pixels,
        "candidate_source": "road_mask_foreground_components_only",
        "roi_used_for_candidate_generation": False,
        "valid_image_mask_used_only_to_skip_invalid_area": valid_image_mask is not None,
        "config": asdict(cfg),
        "components": components,
    }
    result = MaskQualityFilterResult(cleaned, candidates, report, mask.copy())
    if output_dir is not None:
        save_mask_filter_outputs(output_dir, result)
    return result


def render_ignore_candidates_overlay(mask, candidates, show_numbers=True, max_labels=20):
    base = normalize_road_mask(mask)
    overlay = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    tint = overlay.copy()
    combined = np.zeros(base.shape, dtype=np.uint8)
    for candidate in candidates:
        add_candidate_runs_to_mask(combined, candidate)
    tint[combined > 0] = (40, 40, 230)
    overlay = cv2.addWeighted(tint, 0.45, overlay, 0.55, 0)
    for index, candidate in enumerate(candidates):
        for polygon in candidate.get("polygons", []):
            points = np.asarray(polygon, dtype=np.int32)
            if len(points) >= 3:
                cv2.polylines(overlay, [points.reshape(-1, 1, 2)], True, (0, 0, 255), 2)
        if show_numbers and index < max_labels:
            x, y, _, _ = candidate.get("bbox", [0, 0, 0, 0])
            cv2.putText(overlay, str(candidate.get("id", index + 1)), (int(x), int(y) + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def save_mask_filter_outputs(output_dir: str | Path, result: MaskQualityFilterResult) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    before = result.mask_before_ignore if result.mask_before_ignore is not None else result.cleaned_mask
    paths = {
        "cleaned_mask": out / "cleaned_mask.png",
        "mask_before_ignore": out / "mask_before_ignore.png",
        "mask_after_ignore": out / "mask_after_ignore.png",
        "overlay": out / "ignore_candidates_overlay.png",
        "candidate_ignore_regions": out / "candidate_ignore_regions.json",
        "report": out / "mask_filter_report.json",
    }
    images = {
        paths["cleaned_mask"]: result.cleaned_mask,
        paths["mask_before_ignore"]: before,
        # Analysis is preview-only; before and after remain identical until a
        # user explicitly confirms application.
        paths["mask_after_ignore"]: before,
        paths["overlay"]: render_ignore_candidates_overlay(before, result.candidate_ignore_regions),
    }
    for path, image in images.items():
        if not cv2.imwrite(str(path), image):
            raise IOError(f"无法保存 {path}")
    paths["candidate_ignore_regions"].write_text(
        json.dumps(_native(result.candidate_ignore_regions), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["report"].write_text(
        json.dumps(_native(result.report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {key: str(value) for key, value in paths.items()}


def update_mask_filter_apply_outputs(output_dir, mask_before, mask_after, candidates,
                                     report, applied_count):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(report)
    report["applied_count"] = int(applied_count)
    report["affected_road_mask_pixels"] = int(
        np.count_nonzero((normalize_road_mask(mask_before) > 0)
                         & (normalize_road_mask(mask_after) == 0))
    )
    report["total_ignore_area_ratio"] = round(
        report["affected_road_mask_pixels"] / float(max(1, np.asarray(mask_after).size)), 8
    )
    cv2.imwrite(str(out / "mask_before_ignore.png"), normalize_road_mask(mask_before))
    cv2.imwrite(str(out / "mask_after_ignore.png"), normalize_road_mask(mask_after))
    cv2.imwrite(str(out / "ignore_candidates_overlay.png"),
                render_ignore_candidates_overlay(mask_before, candidates))
    (out / "candidate_ignore_regions.json").write_text(
        json.dumps(_native(candidates), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "mask_filter_report.json").write_text(
        json.dumps(_native(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
