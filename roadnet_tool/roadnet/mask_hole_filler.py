"""Road Ribbon Guided Hole & Gap Fill（仅大图 Mask 精修）。

在 road_ribbon_mask 约束下补内部小孔洞与道路带内缺口。
不修改影像，只修改 Road Mask 数据；不重跑分割 / SAM / graph。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_RIBBON_HOLE_GAP_CONFIG: Dict[str, Any] = {
    # hole fill
    "max_hole_area_px": 500,
    "max_hole_diameter_px": 25,
    "min_surround_ratio_for_hole": 0.70,
    # gap fill
    "max_gap_area_px": 800,
    "max_gap_diameter_px": 35,
    "min_surround_ratio_for_gap": 0.45,
    "max_gap_distance_to_mask_px": 8,
    # ribbon constraint
    "ribbon_buffer_px": 10,
    "require_inside_ribbon": True,
    # light morphology after fill
    "close_kernel": 3,
    "open_kernel": 0,
    # speed: work inside ribbon bbox; optionally downscale huge crops
    "preview_max_side": 2500,
    "surround_ring_px": 2,
}


def _imwrite_unicode(path: str | Path, arr: np.ndarray) -> bool:
    """OpenCV imwrite 在 Windows 非 ASCII 路径上常失败，改用 imencode+tofile。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower() or ".png"
    if not ext.startswith("."):
        ext = "." + ext
    ok, buf = cv2.imencode(ext, arr)
    if not ok:
        return False
    buf.tofile(str(p))
    return True


def _binarize(mask: Optional[np.ndarray], shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if mask is None:
        if shape is None:
            raise ValueError("mask is None and shape not provided")
        return np.zeros(shape, dtype=np.uint8)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    out = (arr > 0).astype(np.uint8) * 255
    if shape is not None and out.shape[:2] != shape:
        out = cv2.resize(out, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        out = (out > 0).astype(np.uint8) * 255
    return out


def _dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    r = int(max(0, radius_px))
    if r <= 0:
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask, k)


def _component_diameter(stats_row) -> float:
    w = float(stats_row[cv2.CC_STAT_WIDTH])
    h = float(stats_row[cv2.CC_STAT_HEIGHT])
    return float(max(w, h))


def _surround_ratio(
    component_mask: np.ndarray,
    road_mask: np.ndarray,
    ring_px: int = 2,
) -> float:
    """道路像素占 component 外环的比例。"""
    r = max(1, int(ring_px))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    dilated = cv2.dilate(component_mask, k)
    ring = (dilated > 0) & (component_mask == 0)
    ring_n = int(np.count_nonzero(ring))
    if ring_n <= 0:
        return 0.0
    road_on_ring = int(np.count_nonzero((road_mask > 0) & ring))
    return float(road_on_ring) / float(ring_n)


def _min_distance_to_mask(component_mask: np.ndarray, road_mask: np.ndarray) -> float:
    """component 到已有 road mask 的最小距离（像素）。接触则为 0。"""
    if np.any((component_mask > 0) & (road_mask > 0)):
        return 0.0
    if not np.any(road_mask > 0):
        return float("inf")
    # distance to nearest road pixel
    inv = np.where(road_mask > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    vals = dist[component_mask > 0]
    if vals.size == 0:
        return float("inf")
    return float(vals.min())


def _touches_image_border(labels: np.ndarray, label_id: int) -> bool:
    h, w = labels.shape[:2]
    return bool(
        np.any(labels[0, :] == label_id)
        or np.any(labels[h - 1, :] == label_id)
        or np.any(labels[:, 0] == label_id)
        or np.any(labels[:, w - 1] == label_id)
    )


def _crop_roi(
    *masks: np.ndarray,
    margin: int = 2,
) -> Tuple[Tuple[int, int, int, int], List[np.ndarray]]:
    """按第一个非空 mask 的 bbox 裁剪；返回 (y0,y1,x0,x1), crops。"""
    ref = None
    for m in masks:
        if m is not None and np.any(m > 0):
            ref = m
            break
    if ref is None:
        h, w = masks[0].shape[:2]
        return (0, h, 0, w), [m.copy() for m in masks]
    ys, xs = np.where(ref > 0)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(ref.shape[0], int(ys.max()) + 1 + margin)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(ref.shape[1], int(xs.max()) + 1 + margin)
    return (y0, y1, x0, x1), [m[y0:y1, x0:x1].copy() for m in masks]


def _maybe_downscale(
    masks: Sequence[np.ndarray],
    max_side: int,
) -> Tuple[List[np.ndarray], float]:
    h, w = masks[0].shape[:2]
    side = max(h, w)
    if max_side <= 0 or side <= max_side:
        return [m.copy() for m in masks], 1.0
    scale = float(max_side) / float(side)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    out = [
        cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
        for m in masks
    ]
    return out, scale


def _scale_px(value: float, scale: float) -> float:
    if scale >= 0.999:
        return float(value)
    return float(value) * float(scale)


def fill_holes_and_gaps_guided_by_ribbon(
    mask_uint8: np.ndarray,
    road_ribbon_mask: np.ndarray,
    ignore_mask: Optional[np.ndarray] = None,
    valid_area_mask: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """在 road_ribbon 约束下补内部孔洞与带内缺口。

    Returns:
        repaired_mask (uint8 0/255), report dict
    """
    t0 = time.perf_counter()
    cfg = {**DEFAULT_RIBBON_HOLE_GAP_CONFIG, **(config or {})}

    mask = _binarize(mask_uint8)
    h, w = mask.shape[:2]
    ribbon = _binarize(road_ribbon_mask, (h, w))
    ignore = _binarize(ignore_mask, (h, w)) if ignore_mask is not None else np.zeros((h, w), dtype=np.uint8)
    if valid_area_mask is None:
        valid = np.full((h, w), 255, dtype=np.uint8)
    else:
        valid = _binarize(valid_area_mask, (h, w))

    ribbon_buffer = int(cfg.get("ribbon_buffer_px", 10))
    require_ribbon = bool(cfg.get("require_inside_ribbon", True))
    ribbon_allowed = _dilate(ribbon, ribbon_buffer) if ribbon_buffer > 0 else ribbon.copy()

    report: Dict[str, Any] = {
        "candidate_hole_count": 0,
        "filled_hole_count": 0,
        "rejected_hole_count": 0,
        "candidate_gap_count": 0,
        "filled_gap_count": 0,
        "rejected_gap_count": 0,
        "max_hole_area_px": int(cfg["max_hole_area_px"]),
        "max_gap_area_px": int(cfg["max_gap_area_px"]),
        "max_hole_diameter_px": int(cfg["max_hole_diameter_px"]),
        "max_gap_diameter_px": int(cfg["max_gap_diameter_px"]),
        "ribbon_buffer_px": ribbon_buffer,
        "min_surround_ratio_for_hole": float(cfg["min_surround_ratio_for_hole"]),
        "min_surround_ratio_for_gap": float(cfg["min_surround_ratio_for_gap"]),
        "max_gap_distance_to_mask_px": float(cfg["max_gap_distance_to_mask_px"]),
        "require_inside_ribbon": require_ribbon,
        "accepted_holes": [],
        "accepted_gaps": [],
        "rejected_holes": [],
        "rejected_gaps": [],
        "elapsed_seconds": 0.0,
        "work_scale": 1.0,
        "warnings": [],
    }

    if not np.any(ribbon > 0):
        report["warnings"].append("road_ribbon_mask 为空，未执行补洞/补缺口。")
        report["elapsed_seconds"] = round(time.perf_counter() - t0, 4)
        return mask.copy(), report

    # ── work inside ribbon ROI for speed ──
    margin = max(ribbon_buffer + 4, 8)
    (y0, y1, x0, x1), crops = _crop_roi(
        ribbon_allowed, mask, ignore, valid, ribbon, margin=margin,
    )
    c_allowed, c_mask, c_ignore, c_valid, c_ribbon = crops

    max_side = int(cfg.get("preview_max_side", 2500))
    scaled, scale = _maybe_downscale(
        [c_allowed, c_mask, c_ignore, c_valid, c_ribbon], max_side,
    )
    s_allowed, s_mask, s_ignore, s_valid, s_ribbon = scaled
    report["work_scale"] = float(scale)

    # scale thresholds to work resolution
    max_hole_area = max(1, int(round(_scale_px(cfg["max_hole_area_px"], scale ** 2))))
    max_gap_area = max(1, int(round(_scale_px(cfg["max_gap_area_px"], scale ** 2))))
    max_hole_diam = max(1.0, _scale_px(cfg["max_hole_diameter_px"], scale))
    max_gap_diam = max(1.0, _scale_px(cfg["max_gap_diameter_px"], scale))
    max_gap_dist = max(0.0, _scale_px(cfg["max_gap_distance_to_mask_px"], scale))
    ring_px = max(1, int(round(_scale_px(cfg.get("surround_ring_px", 2), scale))))
    min_hole_sr = float(cfg["min_surround_ratio_for_hole"])
    min_gap_sr = float(cfg["min_surround_ratio_for_gap"])

    # accepted fill masks at work scale (crop)
    accept_fill = np.zeros_like(s_mask)
    hole_cand_vis = np.zeros_like(s_mask)
    gap_cand_vis = np.zeros_like(s_mask)
    accepted_holes_vis = np.zeros_like(s_mask)
    accepted_gaps_vis = np.zeros_like(s_mask)
    rejected_vis = np.zeros_like(s_mask)

    # ── Step 1: internal holes ──
    ch, cw = s_mask.shape[:2]
    padded = np.zeros((ch + 2, cw + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = s_mask
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    hole_region = np.zeros((ch + 2, cw + 2), dtype=np.uint8)
    hole_region[(padded == 0) & (flood == 0)] = 255
    hole_bin = hole_region[1:-1, 1:-1]
    # only consider holes near ribbon
    if require_ribbon:
        hole_bin = cv2.bitwise_and(hole_bin, s_allowed)

    n_h, labels_h, stats_h, centroids_h = cv2.connectedComponentsWithStats(hole_bin, connectivity=8)
    report["candidate_hole_count"] = max(0, n_h - 1)

    for lid in range(1, n_h):
        area = int(stats_h[lid, cv2.CC_STAT_AREA])
        diam = _component_diameter(stats_h[lid])
        cx, cy = float(centroids_h[lid][0]), float(centroids_h[lid][1])
        comp = (labels_h == lid).astype(np.uint8) * 255
        hole_cand_vis[comp > 0] = 255

        reason = None
        # border touch on full work crop (internal holes shouldn't)
        if _touches_image_border(labels_h, lid):
            # in padded sense holes shouldn't touch, but if crop edge mimics border skip cautiously
            # only reject if also outside ribbon buffer already filtered; keep soft
            pass
        if require_ribbon and not np.any((comp > 0) & (s_allowed > 0)):
            reason = "outside_ribbon"
        elif np.any((comp > 0) & (s_ignore > 0)):
            reason = "inside_ignore"
        elif np.any((comp > 0) & (s_valid == 0)):
            reason = "outside_valid_area"
        elif area > max_hole_area:
            reason = "area_too_large"
        elif diam > max_hole_diam:
            reason = "diameter_too_large"
        else:
            sr = _surround_ratio(comp, s_mask, ring_px=ring_px)
            if sr < min_hole_sr:
                reason = "surround_ratio_low"
            else:
                accept_fill[comp > 0] = 255
                accepted_holes_vis[comp > 0] = 255
                report["filled_hole_count"] += 1
                report["accepted_holes"].append({
                    "id": int(lid),
                    "area": area,
                    "diameter": round(diam, 2),
                    "center": [round(cx, 1), round(cy, 1)],
                    "surround_ratio": round(sr, 4),
                })
                continue

        rejected_vis[comp > 0] = 255
        report["rejected_hole_count"] += 1
        report["rejected_holes"].append({
            "id": int(lid),
            "area": area,
            "diameter": round(diam, 2),
            "center": [round(cx, 1), round(cy, 1)],
            "reason": reason or "rejected",
        })

    # ── Step 2: ribbon gaps ──
    after_holes = s_mask.copy()
    after_holes[accept_fill > 0] = 255
    # 用形态学 close 提出「可桥接」的带内缺口，避免 ribbon 边带连成超大连通域
    kdiam = int(round(max_gap_diam)) | 1
    kdiam = max(3, min(kdiam, 51))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kdiam, kdiam))
    closed_prop = cv2.morphologyEx(after_holes, cv2.MORPH_CLOSE, close_k)
    candidate_gap = (closed_prop > 0) & (after_holes == 0)
    if require_ribbon:
        candidate_gap = candidate_gap & (s_allowed > 0)
    else:
        candidate_gap = candidate_gap & ((s_ribbon > 0) | (s_allowed > 0))
    gap_bin = candidate_gap.astype(np.uint8) * 255

    n_g, labels_g, stats_g, centroids_g = cv2.connectedComponentsWithStats(gap_bin, connectivity=8)
    report["candidate_gap_count"] = max(0, n_g - 1)

    for lid in range(1, n_g):
        area = int(stats_g[lid, cv2.CC_STAT_AREA])
        diam = _component_diameter(stats_g[lid])
        cx, cy = float(centroids_g[lid][0]), float(centroids_g[lid][1])
        comp = (labels_g == lid).astype(np.uint8) * 255
        gap_cand_vis[comp > 0] = 255

        reason = None
        if require_ribbon and not np.any((comp > 0) & (s_allowed > 0)):
            reason = "outside_ribbon"
        elif np.any((comp > 0) & (s_ignore > 0)):
            reason = "inside_ignore"
        elif np.any((comp > 0) & (s_valid == 0)):
            reason = "outside_valid_area"
        elif area > max_gap_area:
            reason = "area_too_large"
        elif diam > max_gap_diam:
            reason = "diameter_too_large"
        else:
            sr = _surround_ratio(comp, after_holes, ring_px=ring_px)
            dist = _min_distance_to_mask(comp, after_holes)
            if sr < min_gap_sr:
                reason = "surround_ratio_low"
            elif dist > max_gap_dist:
                reason = "too_far_from_existing_mask"
            else:
                accept_fill[comp > 0] = 255
                accepted_gaps_vis[comp > 0] = 255
                report["filled_gap_count"] += 1
                report["accepted_gaps"].append({
                    "id": int(lid),
                    "area": area,
                    "diameter": round(diam, 2),
                    "center": [round(cx, 1), round(cy, 1)],
                    "surround_ratio": round(sr, 4),
                    "distance_to_mask": round(dist, 3),
                })
                continue

        rejected_vis[comp > 0] = 255
        report["rejected_gap_count"] += 1
        report["rejected_gaps"].append({
            "id": int(lid),
            "area": area,
            "diameter": round(diam, 2),
            "center": [round(cx, 1), round(cy, 1)],
            "reason": reason or "rejected",
        })

    # map accept_fill back to crop full-res then full image
    if scale < 0.999:
        fill_crop = cv2.resize(
            accept_fill, (c_mask.shape[1], c_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        hole_cand_c = cv2.resize(hole_cand_vis, (c_mask.shape[1], c_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        gap_cand_c = cv2.resize(gap_cand_vis, (c_mask.shape[1], c_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        acc_h_c = cv2.resize(accepted_holes_vis, (c_mask.shape[1], c_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        acc_g_c = cv2.resize(accepted_gaps_vis, (c_mask.shape[1], c_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        rej_c = cv2.resize(rejected_vis, (c_mask.shape[1], c_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        fill_crop = accept_fill
        hole_cand_c = hole_cand_vis
        gap_cand_c = gap_cand_vis
        acc_h_c = accepted_holes_vis
        acc_g_c = accepted_gaps_vis
        rej_c = rejected_vis

    # never fill outside ribbon_allowed / ignore / invalid
    fill_crop = cv2.bitwise_and(fill_crop, c_allowed)
    fill_crop[c_ignore > 0] = 0
    fill_crop[c_valid == 0] = 0

    repaired = mask.copy()
    repaired[y0:y1, x0:x1][fill_crop > 0] = 255

    # light morphology, clipped to ribbon_allowed
    close_k = int(cfg.get("close_kernel", 3) or 0)
    open_k = int(cfg.get("open_kernel", 0) or 0)
    if close_k >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        closed = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, k)
        # only allow close growth inside ribbon_allowed
        growth = (closed > 0) & (repaired == 0) & (ribbon_allowed > 0) & (ignore == 0) & (valid > 0)
        repaired[growth] = 255
    if open_k >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        opened = cv2.morphologyEx(repaired, cv2.MORPH_OPEN, k)
        # open only removes; keep outside ribbon unchanged
        inside = ribbon_allowed > 0
        repaired[inside] = opened[inside]

    # final hard constraints
    repaired[ignore > 0] = 0
    repaired[valid == 0] = 0
    # do not add road outside ribbon_allowed relative to original
    added = (repaired > 0) & (mask == 0)
    repaired[added & (ribbon_allowed == 0)] = 0

    # full-size debug masks for artifacts
    hole_cand_full = np.zeros((h, w), dtype=np.uint8)
    gap_cand_full = np.zeros((h, w), dtype=np.uint8)
    acc_h_full = np.zeros((h, w), dtype=np.uint8)
    acc_g_full = np.zeros((h, w), dtype=np.uint8)
    rej_full = np.zeros((h, w), dtype=np.uint8)
    hole_cand_full[y0:y1, x0:x1] = hole_cand_c
    gap_cand_full[y0:y1, x0:x1] = gap_cand_c
    acc_h_full[y0:y1, x0:x1] = acc_h_c
    acc_g_full[y0:y1, x0:x1] = acc_g_c
    rej_full[y0:y1, x0:x1] = rej_c

    report["debug_masks"] = {
        "hole_candidates": hole_cand_full,
        "gap_candidates": gap_cand_full,
        "accepted_holes": acc_h_full,
        "accepted_gaps": acc_g_full,
        "rejected_candidates": rej_full,
        "ribbon_allowed": ribbon_allowed,
        "road_ribbon": ribbon,
    }
    report["elapsed_seconds"] = round(time.perf_counter() - t0, 4)
    return repaired, report


def _overlay_rgb(
    base_mask: np.ndarray,
    green: Optional[np.ndarray] = None,
    red: Optional[np.ndarray] = None,
    blue: Optional[np.ndarray] = None,
) -> np.ndarray:
    rgb = cv2.cvtColor((base_mask > 0).astype(np.uint8) * 80, cv2.COLOR_GRAY2BGR)
    if green is not None:
        rgb[green > 0] = (40, 220, 40)
    if red is not None:
        rgb[red > 0] = (40, 40, 230)
    if blue is not None:
        rgb[blue > 0] = (230, 160, 40)
    return rgb


def save_ribbon_hole_gap_artifacts(
    mask_before: np.ndarray,
    repaired: np.ndarray,
    report: Dict[str, Any],
    out_dir: str | Path,
    *,
    preview_size: Optional[Tuple[int, int]] = None,
    input_mask_path: str = "",
    road_ribbon_mask_path: str = "",
) -> Dict[str, str]:
    """写出补洞补缺口调试产物与报告。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    debug = report.get("debug_masks") or {}

    def _prev(arr: np.ndarray) -> np.ndarray:
        if not preview_size:
            return arr
        pw, ph = int(preview_size[0]), int(preview_size[1])
        if pw <= 0 or ph <= 0:
            return arr
        return cv2.resize(arr, (pw, ph), interpolation=cv2.INTER_NEAREST)

    paths: Dict[str, str] = {}

    def _write(name: str, arr: np.ndarray, *, also_preview: bool = False):
        p = out / name
        _imwrite_unicode(p, arr)
        paths[name] = str(p)
        if also_preview and preview_size:
            pp = out / name.replace(".png", "_preview.png")
            _imwrite_unicode(pp, _prev(arr))
            paths[pp.name] = str(pp)

    before = _binarize(mask_before)
    ribbon = debug.get("road_ribbon")
    if ribbon is None:
        ribbon = np.zeros_like(before)

    _write("mask_before_ribbon_fill.png", before)
    _write("road_ribbon_mask.png", _binarize(ribbon, before.shape[:2]))
    _write("hole_candidates.png", _binarize(debug.get("hole_candidates"), before.shape[:2]))
    _write("gap_candidates.png", _binarize(debug.get("gap_candidates"), before.shape[:2]))

    acc_h = _binarize(debug.get("accepted_holes"), before.shape[:2])
    acc_g = _binarize(debug.get("accepted_gaps"), before.shape[:2])
    rej = _binarize(debug.get("rejected_candidates"), before.shape[:2])

    _write("accepted_holes_overlay.png", _overlay_rgb(before, green=acc_h))
    _write("accepted_gaps_overlay.png", _overlay_rgb(before, blue=acc_g))
    _write("rejected_candidates_overlay.png", _overlay_rgb(before, red=rej))

    repaired_u8 = _binarize(repaired, before.shape[:2])
    _write("ribbon_hole_gap_filled_mask.png", repaired_u8)
    prev = _prev(repaired_u8)
    prev_path = out / "ribbon_hole_gap_filled_mask_preview.png"
    _imwrite_unicode(prev_path, prev)
    paths["ribbon_hole_gap_filled_mask_preview.png"] = str(prev_path)

    # strip heavy arrays from JSON report
    json_report = {
        "input_mask_path": input_mask_path,
        "road_ribbon_mask_path": road_ribbon_mask_path or paths.get("road_ribbon_mask.png", ""),
        "candidate_hole_count": report.get("candidate_hole_count", 0),
        "filled_hole_count": report.get("filled_hole_count", 0),
        "rejected_hole_count": report.get("rejected_hole_count", 0),
        "candidate_gap_count": report.get("candidate_gap_count", 0),
        "filled_gap_count": report.get("filled_gap_count", 0),
        "rejected_gap_count": report.get("rejected_gap_count", 0),
        "max_hole_area_px": report.get("max_hole_area_px"),
        "max_gap_area_px": report.get("max_gap_area_px"),
        "max_hole_diameter_px": report.get("max_hole_diameter_px"),
        "max_gap_diameter_px": report.get("max_gap_diameter_px"),
        "ribbon_buffer_px": report.get("ribbon_buffer_px"),
        "min_surround_ratio_for_hole": report.get("min_surround_ratio_for_hole"),
        "min_surround_ratio_for_gap": report.get("min_surround_ratio_for_gap"),
        "max_gap_distance_to_mask_px": report.get("max_gap_distance_to_mask_px"),
        "elapsed_seconds": report.get("elapsed_seconds"),
        "work_scale": report.get("work_scale"),
        "accepted_holes": report.get("accepted_holes", []),
        "accepted_gaps": report.get("accepted_gaps", []),
        "rejected_holes": report.get("rejected_holes", []),
        "rejected_gaps": report.get("rejected_gaps", []),
        "warnings": report.get("warnings", []),
        "artifacts": paths,
    }
    report_path = out / "ribbon_hole_gap_fill_report.json"
    report_path.write_text(
        json.dumps(json_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["ribbon_hole_gap_fill_report.json"] = str(report_path)
    return paths
