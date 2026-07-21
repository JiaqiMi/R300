"""大图 Mask 种子线清理（large image mode only）。

目标：用户在主路上画几笔短种子线后，自动从当前 working mask 中
保留主路相关连通域、删除远离种子线的孤立误检，减少手工擦除工作量。

不做全图激进桥接；没有 seed corridor 时只做保守筛选。
小图流程不要调用本模块。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from roadnet.optimized_skeleton import skeletonize_thin


DEFAULT_MASK_CLEAN_CONFIG: Dict[str, Any] = {
    "use_preview_level": True,
    "preview_max_side": 2000,

    # seed corridor
    "seed_corridor_width_preview": 80,
    "seed_line_thickness": 3,
    "near_seed_distance_preview": 40,
    "task_buffer_preview": 50,

    # component keep / remove
    "min_component_area_preview": 60,
    "min_skeleton_length_preview": 40,
    "keep_near_seed": True,
    "keep_touching_roi": True,
    "keep_touching_task": True,
    "grow_connected_to_kept": True,
    "max_grow_iterations": 3,

    # light morphology (preview px)
    "close_kernel": 5,
    "open_kernel": 3,
    "fill_small_holes": True,
    "max_hole_area_preview": 200,

    # without seed: do not aggressively clean
    "require_seed": True,
}


def _binarize(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        return np.zeros((1, 1), dtype=np.uint8)
    arr = mask
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return (arr > 0).astype(np.uint8) * 255


def _polygons_to_mask(shape, polygons) -> np.ndarray:
    out = np.zeros(shape[:2], dtype=np.uint8)
    for poly in polygons or []:
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(out, [pts.reshape(-1, 1, 2)], 255)
    return out


def _points_to_mask(shape, points, radius: int) -> np.ndarray:
    out = np.zeros(shape[:2], dtype=np.uint8)
    r = max(1, int(radius))
    for pt in points or []:
        x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            cv2.circle(out, (x, y), r, 255, -1)
    return out


def _strokes_to_mask(shape, strokes, thickness: int) -> np.ndarray:
    out = np.zeros(shape[:2], dtype=np.uint8)
    t = max(1, int(thickness))
    for stroke in strokes or []:
        pts = np.asarray(stroke, dtype=np.int32).reshape(-1, 2)
        if len(pts) == 1:
            x, y = int(pts[0][0]), int(pts[0][1])
            cv2.circle(out, (x, y), t, 255, -1)
        elif len(pts) >= 2:
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, 255, t)
    return out


def _ellipse(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _downscale(mask: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = mask.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return mask, 1.0
    scale = max_side / float(side)
    pw = max(1, int(round(w * scale)))
    ph = max(1, int(round(h * scale)))
    return cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST), scale


def _scale_geom(items, scale: float, kind: str = "poly"):
    if not items or scale == 1.0:
        return items
    if kind == "stroke":
        return [[(float(x) * scale, float(y) * scale) for x, y in stroke]
                for stroke in items]
    if kind == "point":
        return [(float(x) * scale, float(y) * scale) for x, y in items]
    return [[(float(x) * scale, float(y) * scale) for x, y in poly]
            for poly in items]


def _fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    inv = (mask == 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    out = mask.copy()
    h, w = mask.shape[:2]
    for k in range(1, num):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area > max_area:
            continue
        x, y, bw, bh = (
            int(stats[k, cv2.CC_STAT_LEFT]),
            int(stats[k, cv2.CC_STAT_TOP]),
            int(stats[k, cv2.CC_STAT_WIDTH]),
            int(stats[k, cv2.CC_STAT_HEIGHT]),
        )
        if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h:
            continue
        out[labels == k] = 255
    return out


def clean_working_road_mask(
    mask: np.ndarray,
    roi_polygons: Optional[Sequence] = None,
    ignore_polygons: Optional[Sequence] = None,
    main_road_seed_strokes: Optional[Sequence] = None,
    task_points: Optional[Sequence] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """根据主路种子线清理 working mask，返回 cleaned_mask 与报告。"""
    cfg = dict(DEFAULT_MASK_CLEAN_CONFIG)
    cfg.update(config or {})
    t0 = time.time()

    full = _binarize(mask)
    oh, ow = full.shape[:2]
    total_px = float(oh * ow)
    before_nz = int(np.count_nonzero(full))

    report: Dict[str, Any] = {
        "component_count_before": 0,
        "component_count_after": 0,
        "removed_component_count": 0,
        "kept_component_count": 0,
        "seed_stroke_count": len(main_road_seed_strokes or []),
        "roi_count": len(roi_polygons or []),
        "task_point_count": len(task_points or []),
        "mask_nonzero_ratio_before": round(before_nz / total_px, 6) if total_px else 0.0,
        "mask_nonzero_ratio_after": 0.0,
        "close_kernel": int(cfg["close_kernel"]),
        "open_kernel": int(cfg["open_kernel"]),
        "elapsed_seconds": 0.0,
        "warnings": [],
        "used_preview_level": False,
        "preview_scale": 1.0,
        "input_shape": [oh, ow],
    }

    seeds = main_road_seed_strokes or []
    if cfg.get("require_seed", True) and not seeds:
        report["warnings"].append(
            "未提供主路种子线，拒绝激进清理。请先绘制主路种子线。"
        )
        report["elapsed_seconds"] = round(time.time() - t0, 3)
        report["refused"] = True
        return np.zeros_like(full), report

    scale = 1.0
    work = full
    if cfg.get("use_preview_level", True):
        work, scale = _downscale(full, int(cfg["preview_max_side"]))
        report["used_preview_level"] = scale < 1.0
        report["preview_scale"] = float(scale)

    seeds_s = _scale_geom(seeds, scale, "stroke")
    rois_s = _scale_geom(roi_polygons, scale, "poly")
    ign_s = _scale_geom(ignore_polygons, scale, "poly")
    tasks_s = _scale_geom(task_points, scale, "point")

    if ign_s:
        ign = _polygons_to_mask(work.shape, ign_s)
        work = cv2.bitwise_and(work, cv2.bitwise_not(ign))

    seed_mask = _strokes_to_mask(work.shape, seeds_s, int(cfg["seed_line_thickness"]))
    half = max(1, int(cfg["seed_corridor_width_preview"]) // 2)
    corridor = cv2.dilate(seed_mask, _ellipse(half)) if np.any(seed_mask) else np.zeros_like(work)

    near_r = max(1, int(cfg["near_seed_distance_preview"]))
    near_seed = cv2.dilate(seed_mask, _ellipse(near_r)) if np.any(seed_mask) else corridor

    roi_mask = _polygons_to_mask(work.shape, rois_s) if rois_s else np.zeros_like(work)
    task_mask = (
        _points_to_mask(work.shape, tasks_s, int(cfg["task_buffer_preview"]))
        if tasks_s else np.zeros_like(work)
    )

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (work > 0).astype(np.uint8), connectivity=8
    )
    report["component_count_before"] = int(num - 1)
    if num <= 1:
        report["warnings"].append("清理后无道路连通域（输入可能为空）。")
        report["elapsed_seconds"] = round(time.time() - t0, 3)
        return np.zeros((oh, ow), dtype=np.uint8), report

    skel = skeletonize_thin(work)
    skel_counts = np.bincount(labels[skel > 0], minlength=num)
    skel_counts[0] = 0

    if np.any(seed_mask):
        dist = cv2.distanceTransform((seed_mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
    else:
        dist = np.full(work.shape, 1e6, dtype=np.float32)

    seed_hits = set(np.unique(labels[seed_mask > 0]).tolist()) if np.any(seed_mask) else set()
    near_hits = set(np.unique(labels[near_seed > 0]).tolist()) if np.any(near_seed) else set()
    roi_hits = set(np.unique(labels[roi_mask > 0]).tolist()) if np.any(roi_mask) else set()
    task_hits = set(np.unique(labels[task_mask > 0]).tolist()) if np.any(task_mask) else set()
    for s in (seed_hits, near_hits, roi_hits, task_hits):
        s.discard(0)

    min_area = int(cfg["min_component_area_preview"])
    min_skel = int(cfg["min_skeleton_length_preview"])
    keep_ids: set = set()

    for k in range(1, num):
        area = int(stats[k, cv2.CC_STAT_AREA])
        skel_len = int(skel_counts[k])
        touches_seed = k in seed_hits
        near = k in near_hits
        touches_roi = k in roi_hits and bool(cfg.get("keep_touching_roi", True))
        touches_task = k in task_hits and bool(cfg.get("keep_touching_task", True))

        cy, cx = float(centroids[k][1]), float(centroids[k][0])
        iy, ix = int(round(cy)), int(round(cx))
        iy = max(0, min(work.shape[0] - 1, iy))
        ix = max(0, min(work.shape[1] - 1, ix))
        d_seed = float(dist[iy, ix])

        keep = False
        if touches_seed:
            keep = True
        elif cfg.get("keep_near_seed", True) and near and skel_len >= min_skel:
            keep = True
        elif touches_task and skel_len >= min_skel:
            keep = True
        elif touches_roi and area >= min_area:
            if near or d_seed <= near_r * 1.5:
                keep = True

        if area < min_area and skel_len < min_skel and not touches_seed:
            keep = False

        if keep:
            keep_ids.add(k)

    if cfg.get("grow_connected_to_kept", True) and keep_ids:
        for _ in range(int(cfg.get("max_grow_iterations", 3))):
            kept_mask = np.isin(labels, list(keep_ids))
            dilated = cv2.dilate(kept_mask.astype(np.uint8), _ellipse(2))
            touch = set(np.unique(labels[dilated > 0]).tolist())
            touch.discard(0)
            added = False
            for k in touch:
                if k in keep_ids:
                    continue
                area = int(stats[k, cv2.CC_STAT_AREA])
                skel_len = int(skel_counts[k])
                if area < min_area and skel_len < min_skel:
                    continue
                comp = labels == k
                if np.any(comp & (corridor > 0)) or np.any(comp & (near_seed > 0)):
                    keep_ids.add(k)
                    added = True
            if not added:
                break

    cleaned = np.zeros_like(work)
    for k in keep_ids:
        cleaned[labels == k] = 255

    report["kept_component_count"] = len(keep_ids)
    report["removed_component_count"] = max(0, (num - 1) - len(keep_ids))

    morph_region = corridor
    if not np.any(morph_region):
        morph_region = near_seed
    if np.any(morph_region):
        morph_region = cv2.dilate(morph_region, _ellipse(max(2, near_r // 2)))

    ck = int(cfg["close_kernel"])
    ok = int(cfg["open_kernel"])
    if ck > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)
        if np.any(morph_region):
            extra = cv2.bitwise_and(closed, cv2.bitwise_not(cleaned))
            extra = cv2.bitwise_and(extra, morph_region)
            cleaned = np.maximum(cleaned, extra)
        else:
            cleaned = closed
    if ok > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
        opened = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_open)
        if np.any(seed_mask):
            cleaned = np.maximum(opened, cv2.bitwise_and(cleaned, seed_mask))
        else:
            cleaned = opened

    if cfg.get("fill_small_holes", True):
        cleaned = _fill_small_holes(cleaned, int(cfg["max_hole_area_preview"]))

    n2, _, _, _ = cv2.connectedComponentsWithStats(
        (cleaned > 0).astype(np.uint8), connectivity=8
    )
    report["component_count_after"] = int(n2 - 1)

    if scale < 1.0:
        cleaned_full = cv2.resize(cleaned, (ow, oh), interpolation=cv2.INTER_NEAREST)
    else:
        cleaned_full = cleaned

    after_nz = int(np.count_nonzero(cleaned_full))
    report["mask_nonzero_ratio_after"] = round(after_nz / total_px, 6) if total_px else 0.0
    report["elapsed_seconds"] = round(time.time() - t0, 3)
    report["refused"] = False
    report["stages"] = {
        "cleaned_preview": cleaned,
        "corridor_preview": corridor,
        "seed_mask_preview": seed_mask,
    }

    if after_nz == 0:
        report["warnings"].append("清理结果为空，请检查种子线是否画在道路上。")

    return cleaned_full, report


def save_cleaned_mask_artifacts(
    cleaned_mask: np.ndarray,
    report: Dict[str, Any],
    output_dir: str,
    preview_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, str]:
    """保存 cleaned_working_mask / preview / report。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}

    full_path = out / "cleaned_working_mask.png"
    cv2.imwrite(str(full_path), (cleaned_mask > 0).astype(np.uint8) * 255)
    saved["cleaned_working_mask.png"] = str(full_path)

    stages = report.get("stages") or {}
    preview = stages.get("cleaned_preview")
    if preview is None:
        if preview_size and preview_size[0] > 0 and preview_size[1] > 0:
            pw, ph = preview_size
            preview = cv2.resize(
                cleaned_mask, (pw, ph), interpolation=cv2.INTER_NEAREST,
            )
        else:
            preview = cleaned_mask
    prev_path = out / "cleaned_working_mask_preview.png"
    cv2.imwrite(str(prev_path), (preview > 0).astype(np.uint8) * 255)
    saved["cleaned_working_mask_preview.png"] = str(prev_path)

    def _jsonable(obj):
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items() if k != "stages"}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return None
        return obj

    report_path = out / "large_mask_clean_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(report), f, ensure_ascii=False, indent=2)
    saved["large_mask_clean_report.json"] = str(report_path)
    return saved
