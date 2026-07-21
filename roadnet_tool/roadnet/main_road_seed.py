"""Main-road seed strokes for large-image mask repair.

These seeds are centerlines used to generate a road-ribbon mask that is merged
into working_road_mask.  They are NOT graph edges.

Coordinate system: original_image_pixel.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


PRESET_WIDTHS_M = {
    "normal": 8.0,
    "main_road": 12.0,
    "junction": 16.0,
}


def next_seed_id(existing: Sequence[dict], prefix: str = "seed_") -> str:
    max_n = 0
    for item in existing or []:
        sid = str(item.get("id") or "")
        if sid.startswith(prefix):
            try:
                max_n = max(max_n, int(sid[len(prefix):]))
            except ValueError:
                pass
    return f"{prefix}{max_n + 1:03d}"


def points_from_stroke(stroke) -> List[Tuple[float, float]]:
    """Normalize a stroke (dict or point-list) to [(x,y), ...]."""
    if stroke is None:
        return []
    if isinstance(stroke, dict):
        pts = []
        for p in stroke.get("points") or []:
            if isinstance(p, dict):
                pts.append((float(p["x"]), float(p["y"])))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append((float(p[0]), float(p[1])))
        return pts
    pts = []
    for p in stroke:
        if isinstance(p, dict):
            pts.append((float(p["x"]), float(p["y"])))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            pts.append((float(p[0]), float(p[1])))
    return pts


def stroke_length_px(stroke) -> float:
    pts = points_from_stroke(stroke)
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for i in range(len(pts) - 1):
        total += math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
    return total


def compute_road_radius_px(
    road_width_m: float,
    gsd_m_per_px: Optional[float] = None,
    road_radius_px: Optional[float] = None,
) -> float:
    if road_radius_px is not None and float(road_radius_px) > 0:
        return float(road_radius_px)
    gsd = float(gsd_m_per_px or 0.0)
    if gsd > 1e-9:
        return max(1.0, float(road_width_m) / 2.0 / gsd)
    return max(1.0, float(road_width_m) / 2.0)


def make_seed_stroke(
    points: Sequence[Sequence[float]],
    *,
    stroke_id: str,
    road_width_m: float = 8.0,
    road_radius_px: Optional[float] = None,
    gsd_m_per_px: Optional[float] = None,
    mode: str = "normal",
    source: str = "polyline_click",
) -> dict:
    pts = [{"x": float(p[0]), "y": float(p[1])} for p in points if p is not None and len(p) >= 2]
    radius = compute_road_radius_px(road_width_m, gsd_m_per_px, road_radius_px)
    stype = "line" if len(pts) == 2 else "polyline"
    return {
        "id": stroke_id,
        "type": stype,
        "points": pts,
        "road_width_m": float(road_width_m),
        "road_radius_px": float(radius),
        "mode": mode if mode in PRESET_WIDTHS_M or mode == "custom" else "normal",
        "coordinate_system": "original_image_pixel",
        "source": source,
    }


def normalize_stroke_list(raw_strokes: Sequence) -> List[dict]:
    """Accept legacy [[x,y],...] lists or rich dicts → list of seed dicts."""
    out: List[dict] = []
    for idx, stroke in enumerate(raw_strokes or [], 1):
        if isinstance(stroke, dict) and "points" in stroke:
            pts = points_from_stroke(stroke)
            if len(pts) < 1:
                continue
            sid = stroke.get("id") or f"seed_{idx:03d}"
            width = float(stroke.get("road_width_m") or PRESET_WIDTHS_M.get(
                str(stroke.get("mode") or "normal"), 8.0
            ))
            radius = stroke.get("road_radius_px")
            out.append(make_seed_stroke(
                pts,
                stroke_id=sid,
                road_width_m=width,
                road_radius_px=float(radius) if radius is not None else None,
                mode=str(stroke.get("mode") or "normal"),
                source=str(stroke.get("source") or "legacy"),
            ))
        else:
            pts = points_from_stroke(stroke)
            if len(pts) < 1:
                continue
            out.append(make_seed_stroke(
                pts,
                stroke_id=f"seed_{idx:03d}",
                road_width_m=8.0,
                source="legacy",
            ))
    return out


def strokes_to_point_lists(strokes: Sequence) -> List[List[Tuple[float, float]]]:
    return [points_from_stroke(s) for s in (strokes or []) if points_from_stroke(s)]


def build_road_ribbon_mask(
    shape: Tuple[int, int],
    seed_strokes: Sequence,
    *,
    default_radius_px: float = 8.0,
) -> np.ndarray:
    """Dilate each seed centerline into a road ribbon mask (uint8 0/255)."""
    h, w = int(shape[0]), int(shape[1])
    ribbon = np.zeros((h, w), dtype=np.uint8)
    for stroke in normalize_stroke_list(seed_strokes):
        pts = points_from_stroke(stroke)
        if not pts:
            continue
        radius = int(round(float(stroke.get("road_radius_px") or default_radius_px)))
        radius = max(1, radius)
        thickness = max(1, 2 * radius)
        arr = np.asarray(pts, dtype=np.int32).reshape(-1, 2)
        if len(arr) == 1:
            x, y = int(arr[0][0]), int(arr[0][1])
            cv2.circle(ribbon, (x, y), radius, 255, -1)
        else:
            cv2.polylines(ribbon, [arr.reshape(-1, 1, 2)], False, 255, thickness)
            for x, y in arr:
                cv2.circle(ribbon, (int(x), int(y)), radius, 255, -1)
    return ribbon


def _polygons_to_mask(shape, polygons) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for poly in polygons or []:
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
    return mask


def rebuild_mask_from_seed_ribbons(
    working_mask: np.ndarray,
    seed_strokes: Sequence,
    *,
    ignore_polygons: Optional[Sequence] = None,
    far_component_distance_px: float = 40.0,
    default_radius_px: float = 8.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """working OR ribbon → drop components far from ribbon → apply ignore."""
    work = np.asarray(working_mask)
    if work.ndim == 3:
        work = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    work = (work > 0).astype(np.uint8) * 255
    h, w = work.shape[:2]

    strokes = normalize_stroke_list(seed_strokes)
    ribbon = build_road_ribbon_mask((h, w), strokes, default_radius_px=default_radius_px)
    repaired = cv2.bitwise_or(work, ribbon)

    report: Dict[str, Any] = {
        "seed_stroke_count": len(strokes),
        "ribbon_nonzero": int(np.count_nonzero(ribbon)),
        "working_nonzero_before": int(np.count_nonzero(work)),
        "repaired_nonzero_before_filter": int(np.count_nonzero(repaired)),
        "removed_component_count": 0,
        "kept_component_count": 0,
        "warnings": [],
    }
    if not strokes:
        report["warnings"].append("无种子线，跳过重建。")
        return work.copy(), report

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (repaired > 0).astype(np.uint8), connectivity=8
    )
    if np.any(ribbon):
        dist = cv2.distanceTransform((ribbon == 0).astype(np.uint8), cv2.DIST_L2, 3)
    else:
        dist = np.full((h, w), 1e6, dtype=np.float32)

    keep = np.zeros((h, w), dtype=np.uint8)
    removed = 0
    kept = 0
    thr = float(far_component_distance_px)
    for cid in range(1, num):
        comp = labels == cid
        if np.any(ribbon[comp]):
            keep[comp] = 255
            kept += 1
            continue
        min_d = float(dist[comp].min()) if np.any(comp) else 1e9
        if min_d <= thr:
            keep[comp] = 255
            kept += 1
        else:
            removed += 1
    report["removed_component_count"] = removed
    report["kept_component_count"] = kept

    if ignore_polygons:
        ign = _polygons_to_mask((h, w), ignore_polygons)
        keep = cv2.bitwise_and(keep, cv2.bitwise_not(ign))
        report["ignore_applied"] = True
    else:
        report["ignore_applied"] = False

    report["repaired_nonzero_after"] = int(np.count_nonzero(keep))
    return keep, report


def serialize_seed_strokes(strokes: Sequence) -> dict:
    norms = normalize_stroke_list(strokes)
    return {
        "coordinate_system": "original_image_pixel",
        "stroke_count": len(norms),
        "strokes": norms,
        "legacy_point_lists": [
            [[p["x"], p["y"]] for p in s["points"]] for s in norms
        ],
    }


def deserialize_seed_strokes(payload: dict) -> List[dict]:
    if not isinstance(payload, dict):
        return []
    strokes = payload.get("strokes") or []
    if strokes and isinstance(strokes[0], list):
        return normalize_stroke_list(strokes)
    return normalize_stroke_list(strokes)


def apply_angle_constraint(
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> Tuple[float, float]:
    """Snap end to nearest horizontal / vertical / 45° from start (Shift)."""
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return end
    ang = math.degrees(math.atan2(dy, dx))
    snapped = round(ang / 45.0) * 45.0
    length = math.hypot(dx, dy)
    rad = math.radians(snapped)
    return (x0 + length * math.cos(rad), y0 + length * math.sin(rad))


def snap_point_to_candidates(
    x: float,
    y: float,
    candidates: Sequence[Tuple[float, float]],
    snap_px: float,
) -> Tuple[float, float, bool]:
    best = None
    best_d = float(snap_px)
    for cx, cy in candidates or []:
        d = math.hypot(float(cx) - x, float(cy) - y)
        if d <= best_d:
            best_d = d
            best = (float(cx), float(cy))
    if best is None:
        return x, y, False
    return best[0], best[1], True
