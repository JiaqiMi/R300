"""Vehicle waypoint post-export validation (does not affect A*/Dijkstra).

Pipeline (export stage only)::

    dense_path → adaptive resample → vehicle_waypoints
      → duplicate cleanup → spacing / curve / junction / task checks
      → line-of-sight repair → bad_segments → validation report
      → subject1_waypoints.yaml  (only if export_valid)
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .adaptive_waypoint_resampler import (
    AdaptiveWaypointConfig,
    _cumulative_distance,
    _heading_change,
    _interpolate,
    _normalise_path,
    _path_coordinates,
    _point_value,
    _project_to_path,
    _segment_chord_error_m,
    _segment_mask_support_ratio,
    dense_index_for_s,
    densify_dense_path,
)


@dataclass
class WaypointValidationConfig:
    consecutive_duplicate_m: float = 0.3
    near_duplicate_warn_m: float = 0.5
    aba_return_m: float = 0.5
    aba_detour_m: float = 1.0
    straight_min_m: float = 7.0
    straight_max_m: float = 12.0
    dense_zone_min_m: float = 1.0
    dense_zone_max_m: float = 3.0
    max_allowed_spacing_m: float = 12.0
    target_straight_spacing_m: float = 10.0
    hard_fail_spacing_m: float = 20.0
    allow_long_straight: bool = False
    curve_angle_threshold_deg: float = 15.0
    curve_buffer_m: float = 5.0
    curve_spacing_m: float = 2.0
    junction_buffer_m: float = 8.0
    junction_spacing_m: float = 2.0
    task_buffer_m: float = 5.0
    task_spacing_m: float = 2.0
    min_mask_support_ratio: float = 0.75
    max_chord_error_m: float = 1.0
    los_sample_step_m: float = 0.5
    max_insert_iterations: int = 8
    max_distance_to_dense_m: float = 5.0
    max_distance_to_graph_px: float = 25.0
    dense_densify_step_m: float = 0.5
    s_m_epsilon_m: float = 0.05


@dataclass
class WaypointValidationResult:
    waypoints: list[dict]
    report: dict
    bad_segments: list[dict]
    duplicate_indices: list[int] = field(default_factory=list)
    export_valid: bool = False
    yaml_text: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wp_xy_metric(wp: dict) -> Optional[Tuple[float, float]]:
    if wp.get("x_enu") is not None and wp.get("y_enu") is not None:
        return float(wp["x_enu"]), float(wp["y_enu"])
    return None


def _wp_xy_pixel(wp: dict) -> Tuple[float, float]:
    return float(wp["x_pixel"]), float(wp["y_pixel"])


def _distance_m(a: dict, b: dict) -> float:
    ma, mb = _wp_xy_metric(a), _wp_xy_metric(b)
    if ma is not None and mb is not None:
        return math.hypot(mb[0] - ma[0], mb[1] - ma[1])
    ax, ay = _wp_xy_pixel(a)
    bx, by = _wp_xy_pixel(b)
    return math.hypot(bx - ax, by - ay)


def _renumber_waypoints(waypoints: Sequence[dict]) -> list[dict]:
    out = []
    for idx, wp in enumerate(waypoints):
        item = dict(wp)
        item["seq"] = idx + 1
        item["name"] = f"wp_{idx + 1:03d}"
        if idx > 0:
            item["spacing_to_prev_m"] = round(_distance_m(out[-1], item), 3)
        else:
            item["spacing_to_prev_m"] = 0.0
        out.append(item)
    return out


def _spacing_mode(wp: dict) -> str:
    mode = str(wp.get("spacing_mode") or "")
    if mode:
        return mode
    tag = str(wp.get("tag") or "straight")
    if tag in {"start", "goal", "task"}:
        return "task_2m"
    if tag == "intersection":
        return "junction_2m"
    if tag in {"corner", "curve", "sharp_turn"}:
        return "curve_2m"
    if tag == "inserted_for_validation":
        return "inserted_for_validation"
    return "straight_10m"


def _mode_soft_range(mode: str, cfg: WaypointValidationConfig) -> Tuple[float, float]:
    if mode == "straight_10m":
        return cfg.straight_min_m, cfg.straight_max_m
    if mode == "inserted_for_validation":
        # 插点用于 LOS / max-spacing 修复，不强制当作 2m 加密区
        return cfg.dense_zone_min_m, cfg.max_allowed_spacing_m
    if mode in {"curve_2m", "junction_2m", "task_2m"}:
        return cfg.dense_zone_min_m, cfg.dense_zone_max_m
    return cfg.dense_zone_min_m, cfg.max_allowed_spacing_m


def _is_keep_waypoint(wp: dict) -> bool:
    """Anchors that must not be deleted by dedupe / ABA auto-fix / RDP."""
    if bool(wp.get("keep")):
        return True
    if bool(wp.get("forced")):
        return True
    tag = str(wp.get("tag") or "")
    if tag in {
        "start", "goal", "task", "intersection",
        "corner", "sharp_turn", "curve", "inserted_for_validation",
    }:
        return True
    if wp.get("is_task_point") or wp.get("is_intersection") or wp.get("is_corner"):
        return True
    return False


def _ensure_keep(wp: dict) -> dict:
    wp = dict(wp)
    wp["keep"] = _is_keep_waypoint(wp)
    return wp


def _wp_s_m(wp: dict, dense_pixel=None, cumulative=None) -> float:
    if wp.get("s_m") is not None:
        try:
            return float(wp["s_m"])
        except (TypeError, ValueError):
            pass
    if wp.get("path_distance_m") is not None:
        try:
            return float(wp["path_distance_m"])
        except (TypeError, ValueError):
            pass
    if dense_pixel is not None and cumulative is not None:
        return _waypoint_path_s(wp, dense_pixel, cumulative)
    return 0.0


def _wp_dense_index(wp: dict, cumulative=None) -> int:
    if wp.get("dense_index") is not None:
        try:
            return int(wp["dense_index"])
        except (TypeError, ValueError):
            pass
    if cumulative is not None:
        return int(dense_index_for_s(cumulative, _wp_s_m(wp)))
    return 0


def _attach_path_meta(
    wp: dict,
    dense_pixel,
    cumulative,
    *,
    segment_index: Optional[int] = None,
) -> dict:
    wp = _ensure_keep(wp)
    s = _wp_s_m(wp, dense_pixel, cumulative)
    wp["s_m"] = round(float(s), 3)
    wp["path_distance_m"] = wp["s_m"]
    wp["dense_index"] = int(dense_index_for_s(cumulative, s))
    if segment_index is not None:
        wp["segment_index"] = int(segment_index)
    elif wp.get("segment_index") is None:
        wp["segment_index"] = 0
    mode = _spacing_mode(wp)
    wp["spacing_mode"] = mode
    wp["source_mode"] = wp.get("source_mode") or mode
    return wp


# ---------------------------------------------------------------------------
# Duplicate / ABA cleanup
# ---------------------------------------------------------------------------

def remove_and_report_duplicate_waypoints(
    waypoints: Sequence[dict],
    *,
    consecutive_duplicate_m: float = 0.3,
    near_duplicate_warn_m: float = 0.5,
    aba_return_m: float = 0.5,
    aba_detour_m: float = 1.0,
    auto_fix_aba: bool = True,
) -> Tuple[list[dict], dict]:
    """Remove consecutive near-duplicates only; warn on non-consecutive near dups.

    Rules
    -----
    * consecutive distance < consecutive_duplicate_m → drop later point
      **only if** it is not a keep/anchor point
    * non-consecutive distance < near_duplicate_warn_m → warning only
      (合法于环形路径；不删除；不单独导致 export_valid=false)
    * A→B→A (d(A,C)<aba_return_m and d(A,B)>aba_detour_m) →
      auto-delete B if not keep; else mark as residual ABA
    """
    source = [_ensure_keep(dict(wp)) for wp in (waypoints or [])]
    warnings: list[str] = []
    removed = 0
    consecutive_hits = 0
    kept: list[dict] = []
    removed_indices_original: list[int] = []

    for idx, wp in enumerate(source):
        if not kept:
            kept.append(wp)
            continue
        dist = _distance_m(kept[-1], wp)
        if dist < consecutive_duplicate_m:
            consecutive_hits += 1
            if not _is_keep_waypoint(wp):
                # drop later non-keep
                removed += 1
                removed_indices_original.append(idx)
                continue
            if not _is_keep_waypoint(kept[-1]):
                # later is keep → replace earlier non-keep
                kept[-1] = wp
                removed += 1
                removed_indices_original.append(idx - 1)
                continue
            # both keep: keep both (tiny consecutive anchors)
            kept.append(wp)
            continue
        kept.append(wp)

    # ABA detect + auto-fix
    aba_fixed = 0
    guard = 0
    while guard < 64:
        guard += 1
        deleted = False
        for i in range(1, len(kept) - 1):
            a, b, c = kept[i - 1], kept[i], kept[i + 1]
            if _distance_m(a, c) < aba_return_m and _distance_m(a, b) > aba_detour_m:
                # Ring closing: if C is clearly further along the path than B, not ABA
                sa = float(a.get("s_m") if a.get("s_m") is not None else a.get("path_distance_m") or 0.0)
                sb = float(b.get("s_m") if b.get("s_m") is not None else b.get("path_distance_m") or 0.0)
                sc = float(c.get("s_m") if c.get("s_m") is not None else c.get("path_distance_m") or 0.0)
                if sc > sb + 1.0 and sb > sa + 1.0:
                    continue
                if auto_fix_aba and not _is_keep_waypoint(b):
                    warnings.append(
                        f"已自动删除 A-B-A 回跳中间点: "
                        f"{a.get('name', i)} → {b.get('name', i + 1)} → {c.get('name', i + 2)}"
                    )
                    del kept[i]
                    aba_fixed += 1
                    removed += 1
                    deleted = True
                    break
        if not deleted:
            break

    # Residual ABA (keep-protected)
    aba_indices: list[int] = []
    for i in range(1, len(kept) - 1):
        a, b, c = kept[i - 1], kept[i], kept[i + 1]
        if _distance_m(a, c) < aba_return_m and _distance_m(a, b) > aba_detour_m:
            sa = float(a.get("s_m") if a.get("s_m") is not None else a.get("path_distance_m") or 0.0)
            sb = float(b.get("s_m") if b.get("s_m") is not None else b.get("path_distance_m") or 0.0)
            sc = float(c.get("s_m") if c.get("s_m") is not None else c.get("path_distance_m") or 0.0)
            if sc > sb + 1.0 and sb > sa + 1.0:
                continue
            aba_indices.append(i)
            warnings.append(
                f"检测到 A-B-A 回跳(不可自动删除): "
                f"{a.get('name', i)} → {b.get('name', i + 1)} → {c.get('name', i + 2)}"
            )
    aba_count = len(aba_indices)

    # Non-consecutive near duplicates (warn only — never delete)
    near_warn_count = 0
    near_pairs: list[dict] = []
    for i in range(len(kept)):
        for j in range(i + 2, len(kept)):
            if _distance_m(kept[i], kept[j]) < near_duplicate_warn_m:
                near_warn_count += 1
                near_pairs.append({
                    "from_wp": kept[i].get("name", f"wp_{i + 1:03d}"),
                    "to_wp": kept[j].get("name", f"wp_{j + 1:03d}"),
                    "from_index": i,
                    "to_index": j,
                    "distance_m": round(_distance_m(kept[i], kept[j]), 3),
                    "from_dense_index": kept[i].get("dense_index"),
                    "to_dense_index": kept[j].get("dense_index"),
                    "from_s_m": kept[i].get("s_m"),
                    "to_s_m": kept[j].get("s_m"),
                })
                if near_warn_count <= 10:
                    warnings.append(
                        f"non_consecutive_near_duplicate: "
                        f"{kept[i].get('name', i + 1)} 与 "
                        f"{kept[j].get('name', j + 1)} 距离 < {near_duplicate_warn_m}m"
                    )
                break

    kept = _renumber_waypoints(kept)
    report = {
        "duplicate_consecutive_count": consecutive_hits,
        "duplicate_removed_count": removed,
        "aba_backtrack_count": aba_count,
        "aba_fixed_count": aba_fixed,
        "aba_indices": aba_indices,
        "near_duplicate_warning_count": near_warn_count,
        "non_consecutive_near_duplicate_count": near_warn_count,
        "non_consecutive_near_duplicates": near_pairs,
        "removed_original_indices": removed_indices_original,
        "warnings": warnings,
    }
    return kept, report


# ---------------------------------------------------------------------------
# Zone helpers (curve / junction / task)
# ---------------------------------------------------------------------------

def _curve_zone_s_ranges(
    dense_metric: Sequence[Sequence[float]],
    cumulative: Sequence[float],
    cfg: WaypointValidationConfig,
) -> list[Tuple[float, float]]:
    ranges: list[Tuple[float, float]] = []
    if len(dense_metric) < 3:
        return ranges
    for i in range(1, len(dense_metric) - 1):
        angle = _heading_change(dense_metric[i - 1], dense_metric[i], dense_metric[i + 1])
        if angle >= cfg.curve_angle_threshold_deg:
            s = float(cumulative[i])
            ranges.append((max(0.0, s - cfg.curve_buffer_m), s + cfg.curve_buffer_m))
    return _merge_ranges(ranges)


def _junction_points_pixel(final_graph, path_node_sequence=None) -> list[Tuple[float, float]]:
    if final_graph is None:
        return []
    if isinstance(final_graph, dict):
        nodes = list(final_graph.get("nodes", []) or [])
        edges = list(final_graph.get("edges", []) or [])
    else:
        nodes = list(getattr(final_graph, "nodes", []) or [])
        edges = list(getattr(final_graph, "edges", []) or [])
    if not nodes or not edges:
        return []
    degree: dict = {}
    for edge in edges:
        if _point_value(edge, "enabled", default=True) is False:
            continue
        s = _point_value(edge, "start", "source")
        t = _point_value(edge, "end", "target")
        if s is None or t is None:
            continue
        degree[s] = degree.get(s, 0) + 1
        degree[t] = degree.get(t, 0) + 1
    allowed = set(path_node_sequence) if path_node_sequence else None
    pts = []
    for node in nodes:
        nid = _point_value(node, "id")
        if degree.get(nid, 0) < 3:
            continue
        if allowed is not None and nid not in allowed:
            continue
        x = _point_value(node, "x", "x_pixel")
        y = _point_value(node, "y", "y_pixel")
        if x is None or y is None:
            continue
        pts.append((float(x), float(y)))
    return pts


def _task_points_pixel(task_points) -> list[Tuple[float, float]]:
    pts = []
    for item in task_points or []:
        status = str(_point_value(item, "status", default="ok"))
        if status == "failed":
            x = _point_value(item, "original_x", "pixel_x", "x")
            y = _point_value(item, "original_y", "pixel_y", "y")
        else:
            x = _point_value(item, "snapped_x", "pixel_x", "x")
            y = _point_value(item, "snapped_y", "pixel_y", "y")
        if x is None or y is None:
            continue
        pts.append((float(x), float(y)))
    return pts


def _merge_ranges(ranges: Sequence[Tuple[float, float]]) -> list[Tuple[float, float]]:
    if not ranges:
        return []
    ordered = sorted((float(a), float(b)) for a, b in ranges if b > a)
    merged = [ordered[0]]
    for a, b in ordered[1:]:
        if a <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _s_in_ranges(s: float, ranges: Sequence[Tuple[float, float]]) -> bool:
    for a, b in ranges:
        if a - 1e-9 <= s <= b + 1e-9:
            return True
    return False


def _waypoint_path_s(wp: dict, dense_pixel, cumulative) -> float:
    s, _, _ = _project_to_path(
        [float(wp["x_pixel"]), float(wp["y_pixel"])], dense_pixel, cumulative
    )
    return float(s)


def _make_inserted_waypoint(
    mid_px: Sequence[float],
    mid_s: float,
    *,
    dense_index: int = 0,
    segment_index: int = 0,
    metric_converter=None,
    geo_converter=None,
    default_altitude: float = 0.0,
    near_junction: bool = False,
    near_task: bool = False,
) -> dict:
    x, y = float(mid_px[0]), float(mid_px[1])
    x_enu = y_enu = None
    if callable(metric_converter):
        mx, my = metric_converter(x, y)
        x_enu, y_enu = float(mx), float(my)
    lon = lat = None
    if callable(geo_converter):
        lon, lat = geo_converter(x, y)
        lon, lat = float(lon), float(lat)
    return {
        "seq": 0,
        "name": "wp_tmp",
        "x_pixel": x,
        "y_pixel": y,
        "x_enu": x_enu,
        "y_enu": y_enu,
        "longitude": lon,
        "latitude": lat,
        "latitude_deg": lat,
        "longitude_deg": lon,
        "altitude": default_altitude,
        "altitude_m": default_altitude,
        "tag": "inserted_for_validation",
        "spacing_mode": "inserted_for_validation",
        "source_mode": "inserted_for_validation",
        "dense_index": int(dense_index),
        "s_m": round(float(mid_s), 3),
        "segment_index": int(segment_index),
        "keep": True,
        "forced": True,
        "near_junction": near_junction,
        "near_task_point": near_task,
        "local_angle_deg": 0.0,
        "mask_support_ratio": 1.0,
        "path_distance_m": round(float(mid_s), 3),
        "is_task_point": False,
        "is_corner": False,
        "is_intersection": False,
        "inside_image": True,
        "pass_through": True,
    }


def _pick_insert_dense_index(
    start_idx: int,
    end_idx: int,
    sa: float,
    sb: float,
    dense_metric,
    cumulative,
) -> Tuple[Optional[int], Optional[str]]:
    """Pick a dense_path index strictly between start_idx and end_idx.

    Returns (index, error_reason). error_reason set when order invalid.
    """
    if start_idx >= end_idx:
        return None, "dense_index_order_invalid"
    if end_idx - start_idx < 2:
        return None, "dense_index_span_too_small"

    # Prefer cumulative midpoint within the open interval
    mid_s = 0.5 * (float(sa) + float(sb))
    mid_idx = int(dense_index_for_s(cumulative, mid_s))
    mid_idx = max(start_idx + 1, min(end_idx - 1, mid_idx))

    # Among candidates, pick the point that minimizes max chord error of A-mid and mid-B
    best_idx = mid_idx
    best_score = float("inf")
    # Sample a few candidates around midpoint for speed
    lo = start_idx + 1
    hi = end_idx - 1
    step = max(1, (hi - lo) // 16)
    for k in range(lo, hi + 1, step):
        err1 = _segment_chord_error_m(sa, float(cumulative[k]), dense_metric, cumulative)
        err2 = _segment_chord_error_m(float(cumulative[k]), sb, dense_metric, cumulative)
        score = max(err1, err2)
        if score < best_score:
            best_score = score
            best_idx = k
    # Also evaluate true midpoint index
    err1 = _segment_chord_error_m(sa, float(cumulative[mid_idx]), dense_metric, cumulative)
    err2 = _segment_chord_error_m(float(cumulative[mid_idx]), sb, dense_metric, cumulative)
    if max(err1, err2) <= best_score:
        best_idx = mid_idx
    return int(best_idx), None


def _insert_midpoints_for_segments(
    waypoints: list[dict],
    segment_indices: Sequence[int],
    dense_pixel,
    dense_metric,
    cumulative,
    *,
    metric_converter=None,
    geo_converter=None,
    default_altitude: float = 0.0,
) -> Tuple[list[dict], int, list[dict]]:
    """Insert dense_path midpoints only within each segment's dense_index range.

    Returns (waypoints, inserted_count, order_errors).
    """
    if not segment_indices:
        return waypoints, 0, []
    wps = list(waypoints)
    inserted = 0
    order_errors: list[dict] = []
    for i in sorted(set(int(x) for x in segment_indices), reverse=True):
        if i < 0 or i >= len(wps) - 1:
            continue
        a, b = wps[i], wps[i + 1]
        sa = _wp_s_m(a, dense_pixel, cumulative)
        sb = _wp_s_m(b, dense_pixel, cumulative)
        di = _wp_dense_index(a, cumulative)
        dj = _wp_dense_index(b, cumulative)
        pick, err = _pick_insert_dense_index(di, dj, sa, sb, dense_metric, cumulative)
        if err is not None or pick is None:
            order_errors.append({
                "segment_index": i,
                "from_wp": a.get("name"),
                "to_wp": b.get("name"),
                "from_dense_index": di,
                "to_dense_index": dj,
                "from_s_m": sa,
                "to_s_m": sb,
                "distance_m": round(_distance_m(a, b), 3),
                "mask_support_ratio": float(a.get("mask_support_ratio") or 1.0),
                "reason": err or "dense_index_order_invalid",
                "is_curve_zone": _spacing_mode(a) == "curve_2m" or _spacing_mode(b) == "curve_2m",
                "is_junction_zone": bool(a.get("near_junction") or b.get("near_junction")),
                "near_task_point": bool(a.get("near_task_point") or b.get("near_task_point")),
                "fix_attempted": False,
                "fix_result": "skipped_order_invalid",
            })
            continue
        mid_s = float(cumulative[pick])
        if abs(mid_s - sa) < 0.05 or abs(sb - mid_s) < 0.05:
            continue
        mid_px = list(map(float, dense_pixel[pick]))
        near_j = bool(a.get("near_junction") or b.get("near_junction"))
        near_t = bool(a.get("near_task_point") or b.get("near_task_point"))
        seg_i = int(a.get("segment_index") or 0)
        mid = _make_inserted_waypoint(
            mid_px, mid_s,
            dense_index=pick,
            segment_index=seg_i,
            metric_converter=metric_converter,
            geo_converter=geo_converter,
            default_altitude=default_altitude,
            near_junction=near_j,
            near_task=near_t,
        )
        wps.insert(i + 1, mid)
        inserted += 1
    return _renumber_waypoints(wps), inserted, order_errors


def enforce_max_spacing_along_dense(
    waypoints: list[dict],
    dense_pixel,
    dense_metric,
    cumulative,
    *,
    max_allowed_spacing_m: float = 12.0,
    max_insert_iterations: int = 8,
    metric_converter=None,
    geo_converter=None,
    default_altitude: float = 0.0,
) -> Tuple[list[dict], int, list[dict]]:
    """Final兜底：相邻航点距离 > max_allowed 时沿 dense_index 区间插点。"""
    wps = list(waypoints)
    inserted_total = 0
    unresolved: list[dict] = []
    for _ in range(max(1, int(max_insert_iterations))):
        viol = []
        for i in range(len(wps) - 1):
            if _distance_m(wps[i], wps[i + 1]) > max_allowed_spacing_m + 1e-9:
                viol.append(i)
        if not viol:
            unresolved = []
            break
        wps, n_ins, order_errs = _insert_midpoints_for_segments(
            wps, viol, dense_pixel, dense_metric, cumulative,
            metric_converter=metric_converter,
            geo_converter=geo_converter,
            default_altitude=default_altitude,
        )
        inserted_total += n_ins
        unresolved = order_errs
        if n_ins == 0:
            break
    # Final unresolved spacing
    leftover = []
    for i in range(len(wps) - 1):
        dist = _distance_m(wps[i], wps[i + 1])
        if dist > max_allowed_spacing_m + 1e-9:
            a, b = wps[i], wps[i + 1]
            leftover.append({
                "segment_index": i,
                "from_wp": a.get("name"),
                "to_wp": b.get("name"),
                "from_dense_index": _wp_dense_index(a, cumulative),
                "to_dense_index": _wp_dense_index(b, cumulative),
                "from_s_m": _wp_s_m(a, dense_pixel, cumulative),
                "to_s_m": _wp_s_m(b, dense_pixel, cumulative),
                "distance_m": round(dist, 3),
                "mask_support_ratio": float(a.get("mask_support_ratio") or 1.0),
                "reason": "max_spacing_unresolved",
                "is_curve_zone": False,
                "is_junction_zone": bool(a.get("near_junction") or b.get("near_junction")),
                "near_task_point": bool(a.get("near_task_point") or b.get("near_task_point")),
                "fix_attempted": True,
                "fix_result": "failed",
            })
    return wps, inserted_total, unresolved + leftover


def check_s_m_monotonic(
    waypoints: Sequence[dict],
    *,
    epsilon_m: float = 0.05,
) -> Tuple[bool, list[dict]]:
    """Require wp[i+1].s_m > wp[i].s_m (epsilon allowed for float noise)."""
    bad = []
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        sa = float(a.get("s_m") if a.get("s_m") is not None else a.get("path_distance_m") or 0.0)
        sb = float(b.get("s_m") if b.get("s_m") is not None else b.get("path_distance_m") or 0.0)
        if sb + 1e-12 < sa - float(epsilon_m):
            bad.append({
                "segment_index": i,
                "from_wp": a.get("name"),
                "to_wp": b.get("name"),
                "from_dense_index": a.get("dense_index"),
                "to_dense_index": b.get("dense_index"),
                "from_s_m": sa,
                "to_s_m": sb,
                "distance_m": round(_distance_m(a, b), 3),
                "mask_support_ratio": float(a.get("mask_support_ratio") or 1.0),
                "reason": "waypoint_order_invalid_by_s",
                "is_curve_zone": False,
                "is_junction_zone": bool(a.get("near_junction") or b.get("near_junction")),
                "near_task_point": bool(a.get("near_task_point") or b.get("near_task_point")),
                "fix_attempted": False,
                "fix_result": "n/a",
            })
    return len(bad) == 0, bad


# ---------------------------------------------------------------------------
# YAML format check
# ---------------------------------------------------------------------------

def validate_subject1_yaml_text(yaml_text: str) -> Tuple[bool, list[str]]:
    errors: list[str] = []
    if not yaml_text or not str(yaml_text).strip():
        return False, ["YAML 为空"]
    text = str(yaml_text)
    if "coordinate_system:" in text:
        errors.append("正式 YAML 不应包含 coordinate_system")
    if not text.lstrip().startswith("subject1_waypoints:"):
        errors.append("顶层 key 必须是 subject1_waypoints")
    if "\n  waypoints:" not in text and not text.startswith("subject1_waypoints:\n  waypoints:"):
        errors.append("第二级 key 必须是 waypoints")
    # Parse names / field order lightly
    names = []
    blocks = text.split("\n    - name:")
    for block in blocks[1:]:
        lines = block.strip().splitlines()
        if not lines:
            errors.append("航点 name 缺失")
            continue
        name = lines[0].strip()
        names.append(name)
        body = "\n".join(lines[1:])
        for key in ("latitude_deg:", "longitude_deg:", "altitude_m:"):
            if key not in body:
                errors.append(f"{name} 缺少字段 {key[:-1]}")
        # order: latitude before longitude
        li = body.find("latitude_deg:")
        lo = body.find("longitude_deg:")
        if li >= 0 and lo >= 0 and lo < li:
            errors.append(f"{name} 经纬度字段顺序错误（longitude 在 latitude 前）")
    for idx, name in enumerate(names, 1):
        expected = f"wp_{idx:03d}"
        if name != expected:
            errors.append(f"name 编号不连续: 期望 {expected}，实际 {name}")
            break
    return len(errors) == 0, errors


def build_subject1_yaml_text(
    waypoints: Sequence[dict],
    *,
    default_altitude_m: float = 21.741,
) -> str:
    lines = ["subject1_waypoints:", "  waypoints:"]
    for idx, item in enumerate(waypoints, 1):
        name = f"wp_{idx:03d}"
        lat = item.get("latitude_deg", item.get("latitude"))
        lon = item.get("longitude_deg", item.get("longitude"))
        raw_alt = item.get("altitude_m", item.get("altitude"))
        try:
            alt = float(raw_alt) if raw_alt is not None else float(default_altitude_m)
        except (TypeError, ValueError):
            alt = float(default_altitude_m)
        if raw_alt is None or not math.isfinite(alt) or abs(alt) < 1e-9:
            alt = float(default_altitude_m)
        if lat is None or lon is None:
            raise ValueError(f"{name} 缺少经纬度")
        lines.append(f"    - name: {name}")
        lines.append(f"      latitude_deg: {float(lat):.8f}")
        lines.append(f"      longitude_deg: {float(lon):.8f}")
        lines.append(f"      altitude_m: {float(alt):.3f}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate_vehicle_waypoints(
    waypoints: Sequence[dict],
    *,
    dense_path_pixel: Optional[Sequence[Sequence[float]]] = None,
    geo_calibration=None,
    final_graph=None,
    task_points=None,
    road_mask=None,
    path_node_sequence=None,
    config: Optional[WaypointValidationConfig] = None,
    adaptive_config: Optional[AdaptiveWaypointConfig] = None,
    default_altitude_m: float = 21.741,
    allow_long_straight: bool = False,
    output_dir: Optional[str] = None,
    preview_image=None,
    image_width: int = 0,
    image_height: int = 0,
) -> WaypointValidationResult:
    """Validate / repair vehicle waypoints after adaptive resampling."""
    cfg = config or WaypointValidationConfig()
    if adaptive_config is not None:
        cfg.curve_angle_threshold_deg = float(adaptive_config.corner_angle_threshold_deg)
        cfg.curve_buffer_m = float(adaptive_config.corner_buffer_m)
        cfg.curve_spacing_m = float(adaptive_config.curve_spacing_m)
        cfg.junction_buffer_m = float(adaptive_config.intersection_buffer_m)
        cfg.junction_spacing_m = float(adaptive_config.intersection_spacing_m)
        cfg.task_buffer_m = float(adaptive_config.task_point_buffer_m)
        cfg.task_spacing_m = float(adaptive_config.task_point_spacing_m)
        cfg.min_mask_support_ratio = float(adaptive_config.min_mask_support_ratio)
        cfg.max_chord_error_m = float(adaptive_config.max_chord_error_m)
        cfg.los_sample_step_m = float(adaptive_config.los_sample_step_m)
        cfg.max_insert_iterations = int(adaptive_config.max_insert_iterations)
        cfg.dense_densify_step_m = float(getattr(adaptive_config, "dense_densify_step_m", 0.5))
        cfg.max_allowed_spacing_m = float(adaptive_config.max_waypoint_spacing_m)
    cfg.allow_long_straight = bool(allow_long_straight)

    warnings: list[str] = []
    wps = [dict(wp) for wp in (waypoints or [])]
    if len(wps) < 2:
        report = {
            "yaml_format_valid": False,
            "coordinate_valid": False,
            "waypoint_count": len(wps),
            "export_valid": False,
            "geometry_valid": False,
            "warnings": ["车辆航点少于 2 个"],
            "bad_segment_count": 1,
        }
        return WaypointValidationResult(wps, report, [], export_valid=False)

    # Dense path setup
    dense_raw = _normalise_path(dense_path_pixel) if dense_path_pixel is not None else [
        [wp["x_pixel"], wp["y_pixel"]] for wp in wps
    ]
    metric, metric_mode, unit, coord_warnings, metric_converter, _ = _path_coordinates(
        dense_raw, geo_calibration
    )
    warnings.extend(coord_warnings)
    dense_pixel, dense_metric = densify_dense_path(
        dense_raw, metric, cfg.dense_densify_step_m
    )
    cumulative = _cumulative_distance(dense_metric)
    total_length = float(cumulative[-1])

    geo_converter = None
    if geo_calibration is not None:
        geo_converter = getattr(geo_calibration, "pixel_to_wgs84", None)
        if not callable(geo_converter):
            geo_converter = getattr(geo_calibration, "pixel_to_lonlat", None)

    mpp = getattr(geo_calibration, "pixel_resolution_estimated_m", None) if geo_calibration else None
    try:
        mpp = float(mpp) if mpp else None
    except (TypeError, ValueError):
        mpp = None
    if mpp is None or mpp <= 0:
        path_px = _cumulative_distance(dense_pixel)[-1]
        mpp = (total_length / max(1e-6, path_px)) if metric_mode else 1.0

    # 1) Duplicate cleanup (consecutive only) + ABA auto-fix
    wps, dup_report = remove_and_report_duplicate_waypoints(
        wps,
        consecutive_duplicate_m=cfg.consecutive_duplicate_m,
        near_duplicate_warn_m=cfg.near_duplicate_warn_m,
        aba_return_m=cfg.aba_return_m,
        aba_detour_m=cfg.aba_detour_m,
        auto_fix_aba=True,
    )
    warnings.extend(dup_report.get("warnings") or [])

    # Annotate zones + attach dense_index / s_m
    curve_ranges = _curve_zone_s_ranges(dense_metric, cumulative, cfg)
    junction_px = _junction_points_pixel(final_graph, path_node_sequence)
    task_px = _task_points_pixel(task_points)

    def _annotate(wp: dict) -> dict:
        wp = _attach_path_meta(wp, dense_pixel, cumulative)
        s = float(wp["s_m"])
        near_curve = _s_in_ranges(s, curve_ranges)
        near_j = bool(wp.get("near_junction"))
        for jx, jy in junction_px:
            js, _, dpx = _project_to_path([jx, jy], dense_pixel, cumulative)
            if abs(s - js) <= cfg.junction_buffer_m or dpx * mpp <= cfg.junction_buffer_m:
                if math.hypot(wp["x_pixel"] - jx, wp["y_pixel"] - jy) * mpp <= cfg.junction_buffer_m * 1.5 \
                        or abs(s - js) <= cfg.junction_buffer_m:
                    near_j = True
                    break
        near_t = bool(wp.get("near_task_point") or wp.get("is_task_point"))
        for tx, ty in task_px:
            ts, _, _ = _project_to_path([tx, ty], dense_pixel, cumulative)
            if abs(s - ts) <= cfg.task_buffer_m:
                near_t = True
                break
            if math.hypot(wp["x_pixel"] - tx, wp["y_pixel"] - ty) * mpp <= cfg.task_buffer_m:
                near_t = True
                break
        wp["near_junction"] = near_j
        wp["near_task_point"] = near_t
        mode = _spacing_mode(wp)
        if near_t:
            mode = "task_2m"
        elif near_j:
            mode = "junction_2m"
        elif near_curve and mode == "straight_10m":
            mode = "curve_2m"
        wp["spacing_mode"] = mode
        wp["source_mode"] = wp.get("source_mode") or mode
        wp["keep"] = _is_keep_waypoint(wp)
        return wp

    wps = [_annotate(wp) for wp in wps]
    wps = _renumber_waypoints(wps)

    # Ensure task snap points are present
    for tx, ty in task_px:
        ts, _, dist_px = _project_to_path([tx, ty], dense_pixel, cumulative)
        best_i = min(
            range(len(wps)),
            key=lambda i: abs(_wp_s_m(wps[i], dense_pixel, cumulative) - ts),
        )
        if abs(_wp_s_m(wps[best_i], dense_pixel, cumulative) - ts) > 1.0:
            mid_idx = int(dense_index_for_s(cumulative, ts))
            mid_px = list(map(float, dense_pixel[mid_idx]))
            mid = _make_inserted_waypoint(
                mid_px, ts,
                dense_index=mid_idx,
                segment_index=int(wps[best_i].get("segment_index") or 0),
                metric_converter=metric_converter,
                geo_converter=geo_converter,
                default_altitude=default_altitude_m,
                near_task=True,
            )
            mid["spacing_mode"] = "task_2m"
            mid["source_mode"] = "task_2m"
            mid["near_task_point"] = True
            mid["is_task_point"] = True
            mid["keep"] = True
            insert_at = best_i
            if _wp_s_m(wps[best_i], dense_pixel, cumulative) < ts:
                insert_at = best_i + 1
            wps.insert(insert_at, mid)
            wps = _renumber_waypoints(wps)
        else:
            wps[best_i]["near_task_point"] = True
            wps[best_i]["keep"] = True
            if dist_px * mpp < 2.0:
                wps[best_i]["is_task_point"] = True
                wps[best_i]["spacing_mode"] = "task_2m"

    inserted_total = 0
    order_errors: list[dict] = []

    def _zone_spacing_violations(wps_local: list[dict]) -> Tuple[list[int], dict]:
        curve_v = junction_v = task_v = 0
        indices = []
        for i in range(len(wps_local) - 1):
            a, b = wps_local[i], wps_local[i + 1]
            dist = _distance_m(a, b)
            sa = _wp_s_m(a, dense_pixel, cumulative)
            sb = _wp_s_m(b, dense_pixel, cumulative)
            mid_s = 0.5 * (sa + sb)
            in_curve = _s_in_ranges(mid_s, curve_ranges) or a.get("spacing_mode") == "curve_2m" \
                or b.get("spacing_mode") == "curve_2m"
            in_j = bool(a.get("near_junction") or b.get("near_junction"))
            in_t = bool(a.get("near_task_point") or b.get("near_task_point"))
            if in_curve and dist > cfg.dense_zone_max_m:
                curve_v += 1
                indices.append(i)
            if in_j and dist > cfg.dense_zone_max_m:
                junction_v += 1
                indices.append(i)
            if in_t and dist > cfg.dense_zone_max_m:
                task_v += 1
                indices.append(i)
        return indices, {
            "curve_spacing_violation_count": curve_v,
            "junction_spacing_violation_count": junction_v,
            "task_spacing_violation_count": task_v,
        }

    # 2–7) Iterative densify + LOS repair (insert ONLY within dense_index range)
    zone_stats = {
        "curve_spacing_violation_count": 0,
        "junction_spacing_violation_count": 0,
        "task_spacing_violation_count": 0,
    }
    los_failed = 0
    for _iteration in range(max(0, int(cfg.max_insert_iterations))):
        wps = [_annotate(wp) for wp in wps]
        wps = _renumber_waypoints(wps)
        viol_idx, zone_stats = _zone_spacing_violations(wps)
        los_idx = []
        for i in range(len(wps) - 1):
            a, b = wps[i], wps[i + 1]
            ratio = _segment_mask_support_ratio(
                _wp_xy_pixel(a), _wp_xy_pixel(b), road_mask,
                step_m=cfg.los_sample_step_m, metres_per_pixel=mpp,
            )
            wps[i]["mask_support_ratio"] = round(ratio, 4)
            sa = _wp_s_m(a, dense_pixel, cumulative)
            sb = _wp_s_m(b, dense_pixel, cumulative)
            chord = _segment_chord_error_m(sa, sb, dense_metric, cumulative)
            if ratio < cfg.min_mask_support_ratio or chord > cfg.max_chord_error_m:
                los_idx.append(i)
        if not viol_idx and not los_idx:
            break
        merge_idx = sorted(set(viol_idx + los_idx))
        wps, n_ins, errs = _insert_midpoints_for_segments(
            wps, merge_idx, dense_pixel, dense_metric, cumulative,
            metric_converter=metric_converter,
            geo_converter=geo_converter,
            default_altitude=default_altitude_m,
        )
        inserted_total += n_ins
        order_errors.extend(errs)
        if n_ins == 0:
            break

    # 7b) Max spacing 最终兜底
    wps, n_space, space_errs = enforce_max_spacing_along_dense(
        wps, dense_pixel, dense_metric, cumulative,
        max_allowed_spacing_m=cfg.max_allowed_spacing_m,
        max_insert_iterations=cfg.max_insert_iterations,
        metric_converter=metric_converter,
        geo_converter=geo_converter,
        default_altitude=default_altitude_m,
    )
    inserted_total += n_space
    order_errors.extend(space_errs)

    # Final duplicate cleanup + ABA re-fix after inserts
    wps, dup_report2 = remove_and_report_duplicate_waypoints(
        wps,
        consecutive_duplicate_m=cfg.consecutive_duplicate_m,
        near_duplicate_warn_m=cfg.near_duplicate_warn_m,
        aba_return_m=cfg.aba_return_m,
        aba_detour_m=cfg.aba_detour_m,
        auto_fix_aba=True,
    )
    for key in (
        "duplicate_consecutive_count", "duplicate_removed_count",
        "aba_fixed_count",
    ):
        dup_report[key] = int(dup_report.get(key, 0)) + int(dup_report2.get(key, 0))
    # ABA / near-dup counts from FINAL pass only
    dup_report["aba_backtrack_count"] = int(dup_report2.get("aba_backtrack_count", 0))
    dup_report["aba_indices"] = list(dup_report2.get("aba_indices") or [])
    dup_report["non_consecutive_near_duplicate_count"] = int(
        dup_report2.get("non_consecutive_near_duplicate_count", 0)
    )
    dup_report["non_consecutive_near_duplicates"] = list(
        dup_report2.get("non_consecutive_near_duplicates") or []
    )
    warnings.extend(dup_report2.get("warnings") or [])
    wps = [_annotate(wp) for wp in wps]
    wps = _renumber_waypoints(wps)

    # s_m monotonicity
    s_m_monotonic_valid, s_m_bad = check_s_m_monotonic(
        wps, epsilon_m=cfg.s_m_epsilon_m,
    )
    if not s_m_monotonic_valid:
        warnings.append("waypoint_order_invalid_by_s: s_m 非单调递增")

    # Final zone + LOS counts (no more inserts)
    _, zone_stats = _zone_spacing_violations(wps)
    bad_segments: list[dict] = []
    spacing_violation_count = 0
    max_spacing = 0.0
    spacings = []
    los_failed = 0

    def _seg_row(i, a, b, dist, reasons, *, fix_attempted=False, fix_result="n/a",
                 is_curve=False, extra=None):
        row = {
            "segment_index": i,
            "from_wp": a.get("name", f"wp_{i + 1:03d}"),
            "to_wp": b.get("name", f"wp_{i + 2:03d}"),
            "from_dense_index": _wp_dense_index(a, cumulative),
            "to_dense_index": _wp_dense_index(b, cumulative),
            "from_s_m": round(_wp_s_m(a, dense_pixel, cumulative), 3),
            "to_s_m": round(_wp_s_m(b, dense_pixel, cumulative), 3),
            "distance_m": round(dist, 3),
            "spacing_mode": _spacing_mode(a),
            "local_angle_deg": round(float(a.get("local_angle_deg") or 0), 3),
            "mask_support_ratio": round(float(a.get("mask_support_ratio") or 1.0), 4),
            "reason": ";".join(reasons) if isinstance(reasons, list) else str(reasons),
            "is_curve_zone": bool(is_curve),
            "is_junction_zone": bool(a.get("near_junction") or b.get("near_junction")),
            "near_task_point": bool(a.get("near_task_point") or b.get("near_task_point")),
            "fix_attempted": bool(fix_attempted),
            "fix_result": fix_result,
        }
        if extra:
            row.update(extra)
        return row

    for i in range(len(wps) - 1):
        a, b = wps[i], wps[i + 1]
        dist = _distance_m(a, b)
        spacings.append(dist)
        max_spacing = max(max_spacing, dist)
        mode = _spacing_mode(b) if _spacing_mode(b) != "straight_10m" else _spacing_mode(a)
        if a.get("near_task_point") or b.get("near_task_point"):
            mode = "task_2m"
        elif a.get("near_junction") or b.get("near_junction"):
            mode = "junction_2m"
        elif _spacing_mode(a) == "curve_2m" or _spacing_mode(b) == "curve_2m":
            mode = "curve_2m"
        soft_lo, soft_hi = _mode_soft_range(mode, cfg)
        ratio = _segment_mask_support_ratio(
            _wp_xy_pixel(a), _wp_xy_pixel(b), road_mask,
            step_m=cfg.los_sample_step_m, metres_per_pixel=mpp,
        )
        wps[i]["mask_support_ratio"] = round(ratio, 4)
        sa = _wp_s_m(a, dense_pixel, cumulative)
        sb = _wp_s_m(b, dense_pixel, cumulative)
        chord = _segment_chord_error_m(sa, sb, dense_metric, cumulative)
        mid_s = 0.5 * (sa + sb)
        local_angle = 0.0
        for k in range(1, len(dense_metric) - 1):
            if abs(float(cumulative[k]) - mid_s) <= 1.0:
                local_angle = max(
                    local_angle,
                    _heading_change(dense_metric[k - 1], dense_metric[k], dense_metric[k + 1]),
                )
        _, _, dist_to_dense_px = _project_to_path(
            [a["x_pixel"], a["y_pixel"]], dense_pixel, cumulative
        )
        dist_to_dense_m = dist_to_dense_px * mpp

        reasons = []
        reason_kind = None
        if dist > cfg.hard_fail_spacing_m:
            reasons.append(f"distance_m>{cfg.hard_fail_spacing_m}")
            reason_kind = "max_spacing"
        elif dist > cfg.max_allowed_spacing_m:
            if not (cfg.allow_long_straight and mode == "straight_10m" and ratio >= cfg.min_mask_support_ratio):
                reasons.append(f"distance_m>{cfg.max_allowed_spacing_m}")
                spacing_violation_count += 1
                reason_kind = "max_spacing"
            else:
                warnings.append(
                    f"长直线段允许: {a.get('name')}→{b.get('name')} = {dist:.2f}m (LOS通过)"
                )
        if soft_lo <= dist <= soft_hi:
            pass
        elif mode == "straight_10m" and dist > soft_hi and dist <= cfg.max_allowed_spacing_m:
            spacing_violation_count += 1
            warnings.append(
                f"直线点距偏大: {a.get('name')}→{b.get('name')} = {dist:.2f}m (期望 {soft_lo}-{soft_hi}m)"
            )
        elif mode != "straight_10m" and dist > soft_hi:
            spacing_violation_count += 1
            reasons.append(f"{mode}_spacing>{soft_hi}")

        if ratio < cfg.min_mask_support_ratio:
            reasons.append(f"mask_support_ratio<{cfg.min_mask_support_ratio}")
            los_failed += 1
            reason_kind = reason_kind or "los"
        if chord > cfg.max_chord_error_m:
            reasons.append(f"chord_error_m>{cfg.max_chord_error_m}")
            los_failed += 1
            reason_kind = reason_kind or "los"
        if local_angle >= cfg.curve_angle_threshold_deg and dist > cfg.dense_zone_max_m:
            reasons.append("curve_not_densified")
        if dist_to_dense_m > cfg.max_distance_to_dense_m:
            reasons.append("far_from_dense_path")
        d_graph = a.get("distance_to_graph_px")
        if d_graph is not None and float(d_graph) > cfg.max_distance_to_graph_px:
            reasons.append("far_from_graph_edge")
        # dense_index order check
        di = _wp_dense_index(a, cumulative)
        dj = _wp_dense_index(b, cumulative)
        if di > dj:
            reasons.append("dense_index_order_invalid")

        if reasons:
            bad_segments.append(_seg_row(
                i, a, b, dist, reasons,
                fix_attempted=True,
                fix_result="failed",
                is_curve=_s_in_ranges(mid_s, curve_ranges),
                extra={"reason_kind": reason_kind or "other", "spacing_mode": mode},
            ))

    # Order / s_m / spacing unresolved errors
    for err in order_errors:
        bad_segments.append(err)
    for err in s_m_bad:
        bad_segments.append(err)

    # Residual ABA as bad segments (keep points that couldn't be auto-fixed)
    for idx in dup_report.get("aba_indices") or []:
        if 0 <= idx < len(wps):
            a = wps[idx - 1] if idx > 0 else wps[idx]
            b = wps[idx]
            c = wps[idx + 1] if idx + 1 < len(wps) else wps[idx]
            bad_segments.append(_seg_row(
                max(0, idx - 1), a, b, _distance_m(a, b),
                ["aba_backtrack"],
                fix_attempted=True,
                fix_result="blocked_keep_point",
                extra={"reason_kind": "aba", "to_wp": c.get("name")},
            ))

    # Coordinate validity
    coordinate_valid = True
    for wp in wps:
        lat = wp.get("latitude_deg", wp.get("latitude"))
        lon = wp.get("longitude_deg", wp.get("longitude"))
        if lat is None or lon is None:
            coordinate_valid = False
            warnings.append(f"{wp.get('name')} 缺少经纬度")
            break
        try:
            lat_f, lon_f = float(lat), float(lon)
        except (TypeError, ValueError):
            coordinate_valid = False
            break
        if abs(lon_f) <= 90 and abs(lat_f) > 90:
            coordinate_valid = False
            warnings.append("检测到经纬度可能写反")
            break
        if not (3.0 <= lat_f <= 54.0 and 73.0 <= lon_f <= 135.0):
            warnings.append(f"{wp.get('name')} 经纬度超出常见中国范围")

    avg_spacing = (sum(spacings) / len(spacings)) if spacings else 0.0
    mode_counts = {
        "straight_10m": 0,
        "curve_2m": 0,
        "junction_2m": 0,
        "task_2m": 0,
        "inserted_for_validation": 0,
    }
    for wp in wps:
        mode_counts[_spacing_mode(wp)] = mode_counts.get(_spacing_mode(wp), 0) + 1

    geometry_valid = (
        los_failed == 0
        and zone_stats["curve_spacing_violation_count"] == 0
        and zone_stats["junction_spacing_violation_count"] == 0
        and zone_stats["task_spacing_violation_count"] == 0
        and not any(
            "mask_support" in (s.get("reason") or "")
            or "chord_error" in (s.get("reason") or "")
            or "dense_index_order_invalid" in (s.get("reason") or "")
            or "waypoint_order_invalid_by_s" in (s.get("reason") or "")
            for s in bad_segments
        )
        and s_m_monotonic_valid
    )
    if zone_stats["curve_spacing_violation_count"] or zone_stats["junction_spacing_violation_count"] \
            or zone_stats["task_spacing_violation_count"]:
        geometry_valid = False

    hard_spacing_fail = max_spacing > cfg.hard_fail_spacing_m
    soft_spacing_fail = (
        max_spacing > cfg.max_allowed_spacing_m
        and not cfg.allow_long_straight
    )
    if max_spacing > cfg.max_allowed_spacing_m and cfg.allow_long_straight:
        soft_spacing_fail = any(
            "distance_m>" in (s.get("reason") or "") and s.get("spacing_mode") != "straight_10m"
            for s in bad_segments
        )

    yaml_text = None
    yaml_format_valid = False
    try:
        yaml_text = build_subject1_yaml_text(wps, default_altitude_m=default_altitude_m)
        yaml_format_valid, yaml_errors = validate_subject1_yaml_text(yaml_text)
        warnings.extend(yaml_errors)
    except Exception as exc:
        yaml_format_valid = False
        warnings.append(f"YAML 生成失败: {exc}")
        yaml_text = None

    duplicate_consecutive_remaining = 0
    for i in range(len(wps) - 1):
        if _distance_m(wps[i], wps[i + 1]) < cfg.consecutive_duplicate_m:
            # remaining consecutive only counts if neither is forced keep pair
            if not (_is_keep_waypoint(wps[i]) and _is_keep_waypoint(wps[i + 1])):
                duplicate_consecutive_remaining += 1

    aba_count = int(dup_report.get("aba_backtrack_count", 0))
    bad_segment_count = len(bad_segments)
    near_dup_count = int(dup_report.get("non_consecutive_near_duplicate_count", 0))

    # 正式导出条件（non_consecutive_near_duplicate 不单独导致失败）
    export_valid = (
        yaml_format_valid
        and coordinate_valid
        and s_m_monotonic_valid
        and aba_count == 0
        and bad_segment_count == 0
        and geometry_valid
        and los_failed == 0
        and not hard_spacing_fail
        and max_spacing <= cfg.max_allowed_spacing_m + 1e-6
    )
    if (
        not export_valid
        and cfg.allow_long_straight
        and yaml_format_valid
        and coordinate_valid
        and s_m_monotonic_valid
        and aba_count == 0
        and geometry_valid
        and los_failed == 0
        and not hard_spacing_fail
        and bad_segment_count == 0
        and max_spacing > cfg.max_allowed_spacing_m
    ):
        export_valid = True

    if max_spacing > cfg.max_allowed_spacing_m:
        warnings.append("存在过稀疏航点，请检查重采样。")
    if hard_spacing_fail:
        warnings.append(
            f"存在相邻点距 > {cfg.hard_fail_spacing_m}m，validation failed，不生成正式 YAML。"
        )
    if near_dup_count > 0:
        warnings.append(
            f"non_consecutive_near_duplicate_count={near_dup_count}（环形路径常见，仅 warning）"
        )

    report = {
        "yaml_format_valid": yaml_format_valid,
        "coordinate_valid": coordinate_valid,
        "s_m_monotonic_valid": s_m_monotonic_valid,
        "waypoint_count": len(wps),
        "total_length_m": round(total_length, 3) if metric_mode else None,
        "average_spacing_m": round(avg_spacing, 3),
        "max_spacing_m": round(max_spacing, 3),
        "duplicate_consecutive_count": duplicate_consecutive_remaining,
        "duplicate_removed_count": int(dup_report.get("duplicate_removed_count", 0)),
        "aba_backtrack_count": aba_count,
        "aba_fixed_count": int(dup_report.get("aba_fixed_count", 0)),
        "aba_indices": list(dup_report.get("aba_indices") or []),
        "non_consecutive_near_duplicate_count": near_dup_count,
        "non_consecutive_near_duplicates": list(
            dup_report.get("non_consecutive_near_duplicates") or []
        ),
        "straight_waypoint_count": mode_counts.get("straight_10m", 0),
        "curve_waypoint_count": mode_counts.get("curve_2m", 0),
        "junction_waypoint_count": mode_counts.get("junction_2m", 0),
        "task_waypoint_count": mode_counts.get("task_2m", 0),
        "inserted_for_validation_count": int(
            mode_counts.get("inserted_for_validation", 0) + inserted_total
        ),
        "spacing_violation_count": spacing_violation_count,
        "curve_spacing_violation_count": zone_stats["curve_spacing_violation_count"],
        "junction_spacing_violation_count": zone_stats["junction_spacing_violation_count"],
        "task_spacing_violation_count": zone_stats["task_spacing_violation_count"],
        "line_of_sight_failed_count": los_failed,
        "bad_segment_count": bad_segment_count,
        "geometry_valid": geometry_valid,
        "export_valid": export_valid,
        "hard_fail_spacing_m": cfg.hard_fail_spacing_m,
        "max_allowed_spacing_m": cfg.max_allowed_spacing_m,
        "allow_long_straight": cfg.allow_long_straight,
        "parameters": asdict(cfg),
        "warnings": warnings,
        "distance_unit": unit,
        "failure_reasons": _collect_failure_reasons(
            yaml_format_valid, coordinate_valid, duplicate_consecutive_remaining,
            aba_count, bad_segment_count, geometry_valid, hard_spacing_fail,
            soft_spacing_fail, max_spacing, cfg,
            s_m_monotonic_valid=s_m_monotonic_valid,
            los_failed=los_failed,
        ),
    }

    if output_dir:
        write_validation_artifacts(
            output_dir, wps, report, bad_segments,
            preview_image=preview_image,
            image_width=image_width,
            image_height=image_height,
            default_altitude_m=default_altitude_m,
            export_valid=export_valid,
            yaml_text=yaml_text if export_valid else None,
            near_duplicates=dup_report.get("non_consecutive_near_duplicates") or [],
            aba_indices=dup_report.get("aba_indices") or [],
        )

    return WaypointValidationResult(
        waypoints=wps,
        report=report,
        bad_segments=bad_segments,
        duplicate_indices=list(dup_report.get("removed_original_indices") or []),
        export_valid=export_valid,
        yaml_text=yaml_text if export_valid else None,
    )


def _collect_failure_reasons(
    yaml_ok, coord_ok, dup_rem, aba, bad_n, geom_ok, hard_sp, soft_sp, max_sp, cfg,
    *,
    s_m_monotonic_valid: bool = True,
    los_failed: int = 0,
) -> list[str]:
    reasons = []
    if not yaml_ok:
        reasons.append("yaml_format_valid=false")
    if not coord_ok:
        reasons.append("coordinate_valid=false")
    if not s_m_monotonic_valid:
        reasons.append("s_m_monotonic_valid=false")
    if dup_rem:
        reasons.append(f"duplicate_consecutive_count={dup_rem}")
    if aba:
        reasons.append(f"aba_backtrack_count={aba}")
    if bad_n:
        reasons.append(f"bad_segment_count={bad_n}")
    if not geom_ok:
        reasons.append("geometry_valid=false")
    if los_failed:
        reasons.append(f"line_of_sight_failed_count={los_failed}")
    if hard_sp:
        reasons.append(f"max_spacing_m={max_sp:.2f}>{cfg.hard_fail_spacing_m}")
    if soft_sp:
        reasons.append(f"max_spacing_m={max_sp:.2f}>{cfg.max_allowed_spacing_m}")
    return reasons


# ---------------------------------------------------------------------------
# Artifact writers / overlay
# ---------------------------------------------------------------------------

def bad_segments_csv_text(bad_segments: Sequence[dict]) -> str:
    cols = (
        "segment_index", "from_wp", "to_wp",
        "from_dense_index", "to_dense_index", "from_s_m", "to_s_m",
        "distance_m", "mask_support_ratio", "reason",
        "is_curve_zone", "is_junction_zone", "near_task_point",
        "fix_attempted", "fix_result",
    )
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(cols)
    for row in bad_segments:
        writer.writerow([row.get(c, "") for c in cols])
    return buf.getvalue()


def vehicle_waypoints_invalid_csv_text(
    waypoints: Sequence[dict],
    *,
    default_altitude_m: float = 21.741,
) -> str:
    cols = (
        "seq", "name", "latitude_deg", "longitude_deg", "altitude_m",
        "x_pixel", "y_pixel", "dense_index", "s_m", "segment_index",
        "spacing_mode", "source_mode", "local_angle_deg",
        "near_junction", "near_task_point", "mask_support_ratio",
        "spacing_to_prev_m", "keep",
    )
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(cols)
    for wp in waypoints:
        lat = wp.get("latitude_deg", wp.get("latitude"))
        lon = wp.get("longitude_deg", wp.get("longitude"))
        alt = wp.get("altitude_m", wp.get("altitude"))
        if alt is None or abs(float(alt or 0)) < 1e-9:
            alt = default_altitude_m
        writer.writerow([
            wp.get("seq", ""),
            wp.get("name", ""),
            f"{float(lat):.8f}" if lat is not None else "",
            f"{float(lon):.8f}" if lon is not None else "",
            f"{float(alt):.3f}",
            f"{float(wp['x_pixel']):.3f}",
            f"{float(wp['y_pixel']):.3f}",
            wp.get("dense_index", ""),
            wp.get("s_m", wp.get("path_distance_m", "")),
            wp.get("segment_index", ""),
            _spacing_mode(wp),
            wp.get("source_mode", _spacing_mode(wp)),
            f"{float(wp.get('local_angle_deg', 0)):.3f}",
            "true" if wp.get("near_junction") else "false",
            "true" if wp.get("near_task_point") else "false",
            f"{float(wp.get('mask_support_ratio', 1.0)):.4f}",
            f"{float(wp.get('spacing_to_prev_m', 0.0)):.3f}",
            "true" if wp.get("keep") else "false",
        ])
    return buf.getvalue()


def render_validation_overlay(
    waypoints: Sequence[dict],
    bad_segments: Sequence[dict],
    preview_image=None,
    *,
    image_width: int = 0,
    image_height: int = 0,
    near_duplicates: Optional[Sequence[dict]] = None,
    aba_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Render vehicle waypoints + invalid segments for debugging.

    Colors
    ------
    * red circle  – ABA backtrack point
    * red line    – bad segment (generic)
    * orange line – max spacing failure
    * purple line – LOS failure
    * pale yellow – non-consecutive near duplicate (warning only)
    """
    offset_x = offset_y = 0.0
    sx = sy = 1.0

    if preview_image is not None and isinstance(preview_image, np.ndarray) and preview_image.size:
        if preview_image.ndim == 2:
            base = cv2.cvtColor(preview_image, cv2.COLOR_GRAY2BGR)
        elif preview_image.shape[2] == 4:
            base = cv2.cvtColor(preview_image, cv2.COLOR_RGBA2BGR)
        else:
            base = np.ascontiguousarray(preview_image[:, :, :3].copy())
            if base.dtype != np.uint8:
                base = np.clip(base, 0, 255).astype(np.uint8)
            base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        h, w = base.shape[:2]
        if image_width > 0 and image_height > 0 and (image_width != w or image_height != h):
            sx = w / float(image_width)
            sy = h / float(image_height)
    else:
        xs = [float(wp["x_pixel"]) for wp in waypoints] or [0.0]
        ys = [float(wp["y_pixel"]) for wp in waypoints] or [0.0]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad = 40.0
        w = max(400, int(max_x - min_x + 2 * pad))
        h = max(400, int(max_y - min_y + 2 * pad))
        base = np.zeros((h, w, 3), dtype=np.uint8)
        offset_x, offset_y = min_x - pad, min_y - pad

    def _to(x, y):
        return int((float(x) - offset_x) * sx), int((float(y) - offset_y) * sy)

    colors = {
        "straight_10m": (102, 209, 255),
        "curve_2m": (40, 140, 255),
        "junction_2m": (255, 130, 60),
        "task_2m": (220, 70, 160),
        "inserted_for_validation": (80, 220, 80),
    }

    name_to_xy = {wp.get("name"): _to(wp["x_pixel"], wp["y_pixel"]) for wp in waypoints}

    # Non-consecutive near duplicates — pale yellow markers (not severe)
    for pair in near_duplicates or []:
        for key in ("from_wp", "to_wp"):
            xy = name_to_xy.get(pair.get(key))
            if xy:
                cv2.circle(base, xy, 4, (180, 230, 255), 1, cv2.LINE_AA)

    for seg in bad_segments:
        a = name_to_xy.get(seg.get("from_wp"))
        b = name_to_xy.get(seg.get("to_wp"))
        reason = str(seg.get("reason") or "")
        kind = str(seg.get("reason_kind") or "")
        if "aba" in reason or kind == "aba":
            color = (0, 0, 255)  # red
        elif "mask_support" in reason or "chord_error" in reason or kind == "los":
            color = (200, 0, 200)  # purple
        elif "distance_m>" in reason or "max_spacing" in reason or kind == "max_spacing":
            color = (0, 140, 255)  # orange
        else:
            color = (0, 0, 255)
        if a and b:
            cv2.line(base, a, b, color, 3, cv2.LINE_AA)
            mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            cv2.putText(
                base, reason[:40], mid, cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                color, 1, cv2.LINE_AA,
            )

    bad_names = set()
    for seg in bad_segments:
        if seg.get("from_wp"):
            bad_names.add(seg["from_wp"])
        if seg.get("to_wp"):
            bad_names.add(seg["to_wp"])

    aba_set = set(int(i) for i in (aba_indices or []))

    for idx, wp in enumerate(waypoints):
        cx, cy = _to(wp["x_pixel"], wp["y_pixel"])
        mode = _spacing_mode(wp)
        color = colors.get(mode, (200, 200, 200))
        r = 5 if mode == "straight_10m" else 6
        cv2.circle(base, (cx, cy), r, color, -1, cv2.LINE_AA)
        cv2.circle(base, (cx, cy), r + 1, (20, 20, 20), 1, cv2.LINE_AA)
        seq = int(wp.get("seq", idx + 1))
        show_label = (seq == 1 or seq == len(waypoints) or seq % 10 == 0)
        if idx in aba_set:
            cv2.circle(base, (cx, cy), r + 6, (0, 0, 255), 2, cv2.LINE_AA)
            show_label = True
        if wp.get("name") in bad_names:
            show_label = True
            cv2.circle(base, (cx, cy), r + 4, (0, 0, 255), 2, cv2.LINE_AA)
        if show_label:
            cv2.putText(
                base, str(seq), (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
            )
    return base


def write_validation_artifacts(
    output_dir: str,
    waypoints: Sequence[dict],
    report: dict,
    bad_segments: Sequence[dict],
    *,
    preview_image=None,
    image_width: int = 0,
    image_height: int = 0,
    default_altitude_m: float = 21.741,
    export_valid: bool = False,
    yaml_text: Optional[str] = None,
    near_duplicates: Optional[Sequence[dict]] = None,
    aba_indices: Optional[Sequence[int]] = None,
) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    written = []
    report_path = os.path.join(output_dir, "waypoint_validation_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    written.append(report_path)

    bad_path = os.path.join(output_dir, "bad_segments.csv")
    with open(bad_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(bad_segments_csv_text(bad_segments))
    written.append(bad_path)

    overlay = render_validation_overlay(
        waypoints, bad_segments, preview_image,
        image_width=image_width, image_height=image_height,
        near_duplicates=near_duplicates,
        aba_indices=aba_indices,
    )
    overlay_path = os.path.join(output_dir, "waypoint_validation_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    written.append(overlay_path)

    if export_valid and yaml_text:
        yaml_path = os.path.join(output_dir, "subject1_waypoints.yaml")
        with open(yaml_path, "w", encoding="utf-8") as fh:
            fh.write(yaml_text)
        written.append(yaml_path)
        ok_csv = os.path.join(output_dir, "vehicle_waypoints_adaptive.csv")
        with open(ok_csv, "w", encoding="utf-8", newline="") as fh:
            fh.write(vehicle_waypoints_invalid_csv_text(
                waypoints, default_altitude_m=default_altitude_m
            ))
        written.append(ok_csv)
    else:
        inv = os.path.join(output_dir, "vehicle_waypoints_adaptive_INVALID.csv")
        with open(inv, "w", encoding="utf-8", newline="") as fh:
            fh.write(vehicle_waypoints_invalid_csv_text(
                waypoints, default_altitude_m=default_altitude_m
            ))
        written.append(inv)

    return written
