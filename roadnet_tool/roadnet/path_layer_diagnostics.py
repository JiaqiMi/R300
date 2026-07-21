"""分层路径诊断：planned_segments → dense_path → vehicle_waypoints。

deprecated: 正式主流程请使用 roadnet.vehicle_waypoint_pipeline。
本模块仅保留兼容/调试，UI 已隐藏分层诊断入口。

不修改 mask / 骨架 / graph 生成；仅诊断与约束导出链路。
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from roadnet.global_planner import (
    EdgeGeometryMissingError,
    _edge_polyline_points,
    _node_x,
    _node_y,
    expand_edge_path_to_polyline,
)
from roadnet.task_snapping import insert_virtual_nodes


DEFAULT_ENDPOINT_TOLERANCE_PX = 8.0
DEFAULT_MAX_STEP_M = 25.0
DEFAULT_ABA_RETURN_M = 0.5
DEFAULT_ABA_DETOUR_M = 1.0
DEFAULT_SEGMENT_JUMP_M = 30.0


@dataclass
class PathLayerDiagConfig:
    endpoint_tolerance_px: float = DEFAULT_ENDPOINT_TOLERANCE_PX
    max_step_m: float = DEFAULT_MAX_STEP_M
    aba_return_m: float = DEFAULT_ABA_RETURN_M
    aba_detour_m: float = DEFAULT_ABA_DETOUR_M
    segment_jump_m: float = DEFAULT_SEGMENT_JUMP_M
    s_m_epsilon_m: float = 0.05
    metres_per_pixel: float = 0.5


@dataclass
class PathLayerDiagResult:
    planned_segments_valid: bool = False
    dense_path_valid: bool = False
    vehicle_waypoints_valid: Optional[bool] = None
    first_failure: Optional[dict] = None
    planned_segments_rows: list = field(default_factory=list)
    dense_path_rows: list = field(default_factory=list)
    dense_path_points: list = field(default_factory=list)
    dense_bad_segments: list = field(default_factory=list)
    virtual_split_rows: list = field(default_factory=list)
    dense_report: dict = field(default_factory=dict)
    planned_report: dict = field(default_factory=dict)
    artifact_paths: dict = field(default_factory=dict)
    aba_source: Optional[str] = None
    warnings: list = field(default_factory=list)


def _dist(a, b) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _ids_equal(a, b) -> bool:
    """Compare node/edge ids tolerating int/str mismatches from planning serialization."""
    if a is None or b is None:
        return a is b
    if a == b:
        return True
    return str(a) == str(b)


def _index_by_id(items: Sequence[dict], key: str = "id") -> dict:
    """Build lookup that accepts both raw and str(id) keys."""
    out: dict = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        iid = item.get(key)
        if iid is None:
            continue
        out[iid] = item
        out[str(iid)] = item
        try:
            out[int(iid)] = item
        except (TypeError, ValueError):
            pass
    return out


def _heading_deg(a, b, c) -> float:
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 <= 1e-9 or n2 <= 1e-9:
        return 0.0
    cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosine))


def _node_pos(nodes_by_id: dict, nid) -> Optional[Tuple[float, float]]:
    n = nodes_by_id.get(nid)
    if n is None:
        return None
    return float(_node_x(n)), float(_node_y(n))


def check_edge_polyline_endpoints(
    edge: dict,
    nodes_by_id: dict,
    *,
    tolerance_px: float = DEFAULT_ENDPOINT_TOLERANCE_PX,
) -> Tuple[bool, str, str]:
    """Return (ok, orientation, reason).

    orientation: 'forward' means polyline[0]≈source, polyline[-1]≈target
                 'reverse' means polyline[0]≈target, polyline[-1]≈source
    """
    pts = _edge_polyline_points(edge)
    if len(pts) < 2:
        return False, "unknown", "edge_has_polyline=false"
    sid, tid = edge.get("start"), edge.get("end")
    sp, tp = _node_pos(nodes_by_id, sid), _node_pos(nodes_by_id, tid)
    if sp is None or tp is None:
        return False, "unknown", "edge_endpoint_node_missing"
    d0s = _dist(pts[0], sp)
    d1t = _dist(pts[-1], tp)
    d0t = _dist(pts[0], tp)
    d1s = _dist(pts[-1], sp)
    tol = float(tolerance_px)
    if d0s <= tol and d1t <= tol:
        return True, "forward", "ok"
    if d0t <= tol and d1s <= tol:
        return True, "reverse", "ok"
    return False, "unknown", "edge_polyline_endpoint_mismatch"


def orient_polyline_for_travel(
    edge: dict,
    from_node,
    to_node,
    nodes_by_id: dict,
    *,
    tolerance_px: float = DEFAULT_ENDPOINT_TOLERANCE_PX,
) -> Tuple[List[List[float]], str, Optional[str]]:
    """Orient edge polyline for from_node→to_node travel.

    Returns (oriented_pts, direction_label, error_reason).
    Never substitutes node-to-node straight line.
    """
    pts = _edge_polyline_points(edge)
    if len(pts) < 2:
        return [], "unknown", "edge_has_polyline=false"

    ok, stored_orient, reason = check_edge_polyline_endpoints(
        edge, nodes_by_id, tolerance_px=tolerance_px,
    )
    if not ok:
        return [], "unknown", reason

    sid, tid = edge.get("start"), edge.get("end")
    # stored polyline may be reverse of graph start→end
    if stored_orient == "reverse":
        base = list(reversed(pts))
    else:
        base = list(pts)

    # Now base[0]≈sid, base[-1]≈tid
    if _ids_equal(from_node, sid) and _ids_equal(to_node, tid):
        return [[float(p[0]), float(p[1])] for p in base], "forward", None
    if _ids_equal(from_node, tid) and _ids_equal(to_node, sid):
        rev = list(reversed(base))
        return [[float(p[0]), float(p[1])] for p in rev], "reverse", None
    return [], "unknown", f"edge_travel_nodes_mismatch({from_node}->{to_node}, edge {sid}->{tid})"


def expand_segment_edges_debug(
    node_path: Sequence,
    edge_path: Sequence,
    nodes: Sequence[dict],
    edges: Sequence[dict],
    *,
    segment_index: int,
    task_from_seq: int,
    task_to_seq: int,
    metres_per_pixel: float = 0.5,
    tolerance_px: float = DEFAULT_ENDPOINT_TOLERANCE_PX,
    s0: float = 0.0,
) -> Tuple[List[dict], List[dict], List[List[float]], Optional[str]]:
    """Expand one planned segment into dense debug rows.

    Returns (dense_rows, edge_debug_rows, dense_points, fatal_reason).
    """
    nodes_by_id = _index_by_id(nodes)
    edges_by_id = _index_by_id(edges)
    dense_rows: List[dict] = []
    edge_rows: List[dict] = []
    dense_pts: List[List[float]] = []
    s_m = float(s0)
    global_index_base = 0

    if len(node_path) < 2 or len(edge_path) < 1:
        return [], [], [], "empty_segment_path"

    for ei, eid in enumerate(edge_path):
        e = edges_by_id.get(eid)
        from_n = node_path[ei]
        to_n = node_path[ei + 1] if ei + 1 < len(node_path) else None
        row = {
            "segment_index": segment_index,
            "task_from_seq": task_from_seq,
            "task_to_seq": task_to_seq,
            "edge_order_index": ei,
            "edge_id": eid,
            "source_node": e.get("start") if e else None,
            "target_node": e.get("end") if e else None,
            "planned_from_node": from_n,
            "planned_to_node": to_n,
            "edge_has_polyline": False,
            "edge_polyline_point_count": 0,
            "edge_length_m": 0.0,
            "edge_valid": False,
            "reason": "ok",
            "polyline_direction": "unknown",
        }
        if e is None:
            row["reason"] = "edge_not_found"
            edge_rows.append(row)
            return dense_rows, edge_rows, dense_pts, f"edge_not_found:{eid}"
        if e.get("enabled") is False:
            row["reason"] = "edge_valid=false"
            edge_rows.append(row)
            return dense_rows, edge_rows, dense_pts, f"edge_disabled:{eid}"

        oriented, direction, err = orient_polyline_for_travel(
            e, from_n, to_n, nodes_by_id, tolerance_px=tolerance_px,
        )
        pts_raw = _edge_polyline_points(e)
        row["edge_has_polyline"] = len(pts_raw) >= 2
        row["edge_polyline_point_count"] = len(pts_raw)
        row["polyline_direction"] = direction
        if err is not None or len(oriented) < 2:
            row["reason"] = err or "edge_has_polyline=false"
            edge_rows.append(row)
            return dense_rows, edge_rows, dense_pts, row["reason"]

        length_px = sum(_dist(oriented[i], oriented[i + 1]) for i in range(len(oriented) - 1))
        row["edge_length_m"] = round(length_px * float(metres_per_pixel), 3)
        row["edge_valid"] = True
        edge_rows.append(row)

        start_k = 0
        if dense_pts and _dist(dense_pts[-1], oriented[0]) < 0.5:
            start_k = 1
        for k in range(start_k, len(oriented)):
            pt = oriented[k]
            step = 0.0
            if dense_pts:
                step_px = _dist(dense_pts[-1], pt)
                step = step_px * float(metres_per_pixel)
                s_m += step
            turn = 0.0
            if len(dense_pts) >= 2:
                turn = _heading_deg(dense_pts[-2], dense_pts[-1], pt)
            dense_pts.append([float(pt[0]), float(pt[1])])
            dense_rows.append({
                "global_index": len(dense_rows),
                "segment_index": segment_index,
                "task_from_seq": task_from_seq,
                "task_to_seq": task_to_seq,
                "edge_id": eid,
                "edge_point_index": k,
                "node_from": from_n,
                "node_to": to_n,
                "polyline_direction": direction,
                "x_pixel": round(float(pt[0]), 3),
                "y_pixel": round(float(pt[1]), 3),
                "longitude": None,
                "latitude": None,
                "s_m": round(s_m, 3),
                "step_distance_m": round(step, 3),
                "turn_angle_deg": round(turn, 3),
                "source": "edge_polyline",
            })

    return dense_rows, edge_rows, dense_pts, None


def validate_virtual_node_splits(
    original_nodes: Sequence[dict],
    original_edges: Sequence[dict],
    snapped_points: Sequence,
    *,
    tolerance_px: float = DEFAULT_ENDPOINT_TOLERANCE_PX,
) -> Tuple[List[dict], bool]:
    """Validate edge→virtual splits; return (rows, all_valid)."""
    rows: List[dict] = []
    all_ok = True
    # Only edge-projection snaps
    by_edge: Dict[Any, list] = {}
    for sp in snapped_points or []:
        status = getattr(sp, "status", None) or (sp.get("status") if isinstance(sp, dict) else "ok")
        if status == "failed":
            continue
        eid = getattr(sp, "edge_id", None) if not isinstance(sp, dict) else sp.get("edge_id")
        nid = getattr(sp, "node_id", None) if not isinstance(sp, dict) else sp.get("node_id")
        vnid = getattr(sp, "virtual_node_id", None) if not isinstance(sp, dict) else sp.get("virtual_node_id")
        if eid is None or nid is not None or vnid is None:
            continue
        by_edge.setdefault(eid, []).append(sp)

    edges_by_id = _index_by_id(original_edges)
    nodes_by_id = _index_by_id(original_nodes)

    for eid, sps in by_edge.items():
        e = edges_by_id.get(eid)
        new_nodes, new_edges = insert_virtual_nodes(original_nodes, original_edges, sps)
        split_edges = [ee for ee in new_edges if ee.get("source") == "task_split"]
        for sp in sps:
            seq = getattr(sp, "seq", None) if not isinstance(sp, dict) else sp.get("seq")
            vnid = getattr(sp, "virtual_node_id", None) if not isinstance(sp, dict) else sp.get("virtual_node_id")
            left = [ee for ee in split_edges if ee.get("end") == vnid]
            right = [ee for ee in split_edges if ee.get("start") == vnid]
            left_e = left[0] if left else None
            right_e = right[0] if right else None
            left_ok = right_ok = False
            left_n = right_n = 0
            reasons = []
            if left_e is None:
                reasons.append("left_edge_missing")
            else:
                left_n = len(_edge_polyline_points(left_e))
                ok, _, r = check_edge_polyline_endpoints(
                    left_e, {**nodes_by_id, **{n["id"]: n for n in new_nodes}},
                    tolerance_px=tolerance_px,
                )
                left_ok = ok and left_n >= 2
                if not left_ok:
                    reasons.append(f"left:{r}")
            if right_e is None:
                reasons.append("right_edge_missing")
            else:
                right_n = len(_edge_polyline_points(right_e))
                ok, _, r = check_edge_polyline_endpoints(
                    right_e, {**nodes_by_id, **{n["id"]: n for n in new_nodes}},
                    tolerance_px=tolerance_px,
                )
                right_ok = ok and right_n >= 2
                if not right_ok:
                    reasons.append(f"right:{r}")
            # Detect straight-line only (2 points == endpoints) as weak but allowed if endpoints match
            if left_ok and right_ok and left_n == 2 and right_n == 2 and e is not None:
                orig_n = len(_edge_polyline_points(e))
                if orig_n > 4:
                    reasons.append("split_polyline_collapsed_to_line")
                    # still valid geometrically if endpoints match; warn only
            valid = left_ok and right_ok
            if not valid:
                all_ok = False
            rows.append({
                "task_seq": seq,
                "original_edge_id": eid,
                "virtual_node_id": vnid,
                "split_index": 0,
                "left_edge_id": left_e.get("id") if left_e else None,
                "right_edge_id": right_e.get("id") if right_e else None,
                "left_polyline_count": left_n,
                "right_polyline_count": right_n,
                "left_valid": left_ok,
                "right_valid": right_ok,
                "reason": ";".join(reasons) if reasons else "ok",
            })
    return rows, all_ok


def validate_dense_path(
    dense_rows: Sequence[dict],
    *,
    config: Optional[PathLayerDiagConfig] = None,
) -> Tuple[bool, dict, List[dict]]:
    """Validate dense_path debug rows. Returns (ok, report, bad_segments)."""
    cfg = config or PathLayerDiagConfig()
    bad: List[dict] = []
    warnings: List[str] = []
    if len(dense_rows) < 2:
        report = {
            "dense_path_valid": False,
            "point_count": len(dense_rows),
            "reason": "dense_path_too_short",
            "s_m_monotonic_valid": False,
            "aba_count": 0,
            "max_step_m": 0.0,
        }
        return False, report, bad

    s_m_ok = True
    max_step = 0.0
    for i in range(1, len(dense_rows)):
        a, b = dense_rows[i - 1], dense_rows[i]
        sa, sb = float(a.get("s_m") or 0.0), float(b.get("s_m") or 0.0)
        step = float(b.get("step_distance_m") or max(0.0, sb - sa))
        max_step = max(max_step, step)
        if sb + 1e-12 < sa - cfg.s_m_epsilon_m:
            s_m_ok = False
            bad.append({
                "segment_index": b.get("segment_index"),
                "from_index": a.get("global_index"),
                "to_index": b.get("global_index"),
                "from_s_m": sa,
                "to_s_m": sb,
                "distance_m": round(step, 3),
                "edge_id": b.get("edge_id"),
                "reason": "from_s_m>to_s_m",
            })
        if step > cfg.max_step_m:
            bad.append({
                "segment_index": b.get("segment_index"),
                "from_index": a.get("global_index"),
                "to_index": b.get("global_index"),
                "from_s_m": sa,
                "to_s_m": sb,
                "distance_m": round(step, 3),
                "edge_id": b.get("edge_id"),
                "reason": "step_distance_too_large",
            })
        # segment splice jump
        if (
            a.get("segment_index") != b.get("segment_index")
            and step > cfg.segment_jump_m
        ):
            bad.append({
                "segment_index": b.get("segment_index"),
                "from_index": a.get("global_index"),
                "to_index": b.get("global_index"),
                "from_s_m": sa,
                "to_s_m": sb,
                "distance_m": round(step, 3),
                "edge_id": b.get("edge_id"),
                "reason": "segment_splice_jump",
            })

    # ABA on consecutive triples (local geometric backtrack).
    # Note: s_m still increases when walking A→B→A along a polyline, so do NOT
    # suppress ABA solely because arc-length is monotonic.
    aba_count = 0
    for i in range(1, len(dense_rows) - 1):
        a = (float(dense_rows[i - 1]["x_pixel"]), float(dense_rows[i - 1]["y_pixel"]))
        b = (float(dense_rows[i]["x_pixel"]), float(dense_rows[i]["y_pixel"]))
        c = (float(dense_rows[i + 1]["x_pixel"]), float(dense_rows[i + 1]["y_pixel"]))
        sa = float(dense_rows[i - 1].get("s_m") or 0)
        sc = float(dense_rows[i + 1].get("s_m") or 0)
        d_ac = _dist(a, c) * cfg.metres_per_pixel
        d_ab = _dist(a, b) * cfg.metres_per_pixel
        if d_ac < cfg.aba_return_m and d_ab > cfg.aba_detour_m:
            aba_count += 1
            bad.append({
                "segment_index": dense_rows[i].get("segment_index"),
                "from_index": dense_rows[i - 1].get("global_index"),
                "to_index": dense_rows[i + 1].get("global_index"),
                "from_s_m": sa,
                "to_s_m": sc,
                "distance_m": round(d_ac, 3),
                "edge_id": dense_rows[i].get("edge_id"),
                "reason": "aba_backtrack",
            })

    ok = s_m_ok and aba_count == 0 and not bad
    # soft: only step warnings when under threshold already handled in bad
    report = {
        "dense_path_valid": ok,
        "point_count": len(dense_rows),
        "s_m_monotonic_valid": s_m_ok,
        "aba_count": aba_count,
        "bad_segment_count": len(bad),
        "max_step_m": round(max_step, 3),
        "max_allowed_step_m": cfg.max_step_m,
        "total_length_m": round(float(dense_rows[-1].get("s_m") or 0.0), 3),
        "warnings": warnings,
    }
    return ok, report, bad


def run_layered_path_diagnostics(
    planning_result,
    nodes: Sequence[dict],
    edges: Sequence[dict],
    snapped_task_points=None,
    *,
    dense_path_pixel: Optional[Sequence] = None,
    geo_calibration=None,
    config: Optional[PathLayerDiagConfig] = None,
    output_dir: Optional[str] = None,
    preview_image=None,
    image_width: int = 0,
    image_height: int = 0,
) -> PathLayerDiagResult:
    """Run planned_segments + dense_path diagnostics (before vehicle waypoints).

    If ``planning_result`` has no segments, planned_segments is treated as vacuously
    valid and dense_path is built from ``dense_path_pixel`` (legacy / unit-test path).
    """
    cfg = config or PathLayerDiagConfig()
    mpp = float(cfg.metres_per_pixel)
    if geo_calibration is not None:
        try:
            est = getattr(geo_calibration, "pixel_resolution_estimated_m", None)
            if est:
                mpp = float(est)
                cfg.metres_per_pixel = mpp
        except (TypeError, ValueError):
            pass

    result = PathLayerDiagResult()
    geo_converter = None
    if geo_calibration is not None:
        geo_converter = getattr(geo_calibration, "pixel_to_wgs84", None)
        if not callable(geo_converter):
            geo_converter = getattr(geo_calibration, "pixel_to_lonlat", None)

    # Virtual split check (global)
    vrows, v_ok = validate_virtual_node_splits(
        nodes, edges, snapped_task_points or [],
        tolerance_px=cfg.endpoint_tolerance_px,
    )
    result.virtual_split_rows = vrows
    if not v_ok:
        result.warnings.append("virtual_node_split 存在无效段")

    segments = list(getattr(planning_result, "segments", []) or []) if planning_result else []
    # Build task lookup
    sp_by_seq = {}
    for sp in snapped_task_points or []:
        seq = getattr(sp, "seq", None) if not isinstance(sp, dict) else sp.get("seq")
        if seq is not None:
            sp_by_seq[int(seq)] = sp

    all_edge_rows: List[dict] = []
    all_dense_rows: List[dict] = []
    all_dense_pts: List[List[float]] = []
    planned_ok = True
    s_cursor = 0.0
    first_failure = None

    for seg_i, seg in enumerate(segments, 1):
        from_seq = int(getattr(seg, "from_seq", 0) or 0)
        to_seq = int(getattr(seg, "to_seq", 0) or 0)
        status = getattr(seg, "status", "")
        node_path = list(getattr(seg, "node_path", []) or [])
        edge_path = list(getattr(seg, "edge_path", []) or [])

        if status != "ok":
            planned_ok = False
            row = {
                "segment_index": seg_i,
                "task_from_seq": from_seq,
                "task_to_seq": to_seq,
                "edge_order_index": -1,
                "edge_id": None,
                "source_node": None,
                "target_node": None,
                "planned_from_node": node_path[0] if node_path else None,
                "planned_to_node": node_path[-1] if node_path else None,
                "edge_has_polyline": False,
                "edge_polyline_point_count": 0,
                "edge_length_m": 0.0,
                "edge_valid": False,
                "reason": getattr(seg, "error", None) or f"segment_status={status}",
            }
            all_edge_rows.append(row)
            if first_failure is None:
                first_failure = {
                    "layer": "planned_segments",
                    "segment_index": seg_i,
                    "edge_id": None,
                    "reason": row["reason"],
                }
            continue

        # Task order continuity
        if seg_i > 1:
            prev = segments[seg_i - 2]
            if int(getattr(prev, "to_seq", -1) or -1) != from_seq:
                planned_ok = False
                if first_failure is None:
                    first_failure = {
                        "layer": "planned_segments",
                        "segment_index": seg_i,
                        "edge_id": None,
                        "reason": "task_seq_order_invalid",
                    }

        # Rebuild segment-local graph with only endpoint virtuals
        src_sp = sp_by_seq.get(from_seq)
        dst_sp = sp_by_seq.get(to_seq)
        seg_sps = [x for x in (src_sp, dst_sp) if x is not None]
        if seg_sps:
            seg_nodes, seg_edges = insert_virtual_nodes(nodes, edges, seg_sps)
        else:
            seg_nodes, seg_edges = list(nodes), list(edges)

        dense_rows, edge_rows, dense_pts, fatal = expand_segment_edges_debug(
            node_path, edge_path, seg_nodes, seg_edges,
            segment_index=seg_i,
            task_from_seq=from_seq,
            task_to_seq=to_seq,
            metres_per_pixel=mpp,
            tolerance_px=cfg.endpoint_tolerance_px,
            s0=s_cursor,
        )
        all_edge_rows.extend(edge_rows)
        if fatal:
            planned_ok = False
            if first_failure is None:
                first_failure = {
                    "layer": "planned_segments",
                    "segment_index": seg_i,
                    "edge_id": edge_rows[-1].get("edge_id") if edge_rows else None,
                    "reason": fatal,
                }
            continue

        # stitch dense rows (reindex global)
        if all_dense_pts and dense_pts and _dist(all_dense_pts[-1], dense_pts[0]) < 0.5:
            dense_rows = dense_rows[1:]
            dense_pts = dense_pts[1:]
        for r in dense_rows:
            r["global_index"] = len(all_dense_rows)
            if callable(geo_converter):
                try:
                    lon, lat = geo_converter(r["x_pixel"], r["y_pixel"])
                    r["longitude"] = round(float(lon), 8)
                    r["latitude"] = round(float(lat), 8)
                except Exception:
                    pass
            all_dense_rows.append(r)
        all_dense_pts.extend(dense_pts)
        if all_dense_rows:
            s_cursor = float(all_dense_rows[-1]["s_m"])

    # Vacuous planned_segments: no segments to validate (legacy dense-only export)
    if not segments:
        result.planned_segments_valid = bool(v_ok)
        result.planned_segments_rows = []
        # Build dense rows from provided pixel path
        fallback = list(dense_path_pixel or [])
        s_m = 0.0
        for i, pt in enumerate(fallback):
            x, y = float(pt[0]), float(pt[1])
            step = 0.0
            if i > 0:
                step = _dist(fallback[i - 1], pt) * mpp
                s_m += step
            lon = lat = None
            if callable(geo_converter):
                try:
                    lon, lat = geo_converter(x, y)
                    lon, lat = round(float(lon), 8), round(float(lat), 8)
                except Exception:
                    pass
            all_dense_rows.append({
                "global_index": i,
                "segment_index": 0,
                "task_from_seq": None,
                "task_to_seq": None,
                "edge_id": None,
                "edge_point_index": i,
                "node_from": None,
                "node_to": None,
                "polyline_direction": "forward",
                "x_pixel": round(x, 3),
                "y_pixel": round(y, 3),
                "longitude": lon,
                "latitude": lat,
                "s_m": round(s_m, 3),
                "step_distance_m": round(step, 3),
                "turn_angle_deg": 0.0,
                "source": "dense_path_pixel_fallback",
            })
            all_dense_pts.append([x, y])
    else:
        edge_checks = [
            r for r in all_edge_rows if r.get("edge_order_index", -1) >= 0
        ]
        result.planned_segments_valid = bool(
            planned_ok
            and v_ok
            and edge_checks
            and all(r.get("edge_valid") for r in edge_checks)
        )
        result.planned_segments_rows = all_edge_rows

    result.dense_path_rows = all_dense_rows
    result.dense_path_points = all_dense_pts
    result.planned_report = {
        "planned_segments_valid": result.planned_segments_valid,
        "edge_row_count": len(all_edge_rows),
        "virtual_split_valid": v_ok,
        "virtual_split_count": len(vrows),
        "segment_count": len(segments),
    }

    dense_ok, dense_report, dense_bad = validate_dense_path(all_dense_rows, config=cfg)
    # If planned failed, dense cannot be valid for export
    if not result.planned_segments_valid:
        dense_ok = False
        dense_report["dense_path_valid"] = False
        dense_report["blocked_by_planned_segments"] = True
    result.dense_path_valid = dense_ok
    result.dense_report = dense_report
    result.dense_bad_segments = dense_bad

    if first_failure is None and not dense_ok and dense_bad:
        b0 = dense_bad[0]
        first_failure = {
            "layer": "dense_path",
            "segment_index": b0.get("segment_index"),
            "edge_id": b0.get("edge_id"),
            "from_wp": b0.get("from_index"),
            "to_wp": b0.get("to_index"),
            "reason": b0.get("reason"),
        }
    elif first_failure is None and not result.planned_segments_valid:
        first_failure = {
            "layer": "planned_segments",
            "segment_index": None,
            "edge_id": None,
            "reason": "planned_segments_invalid",
        }
    result.first_failure = first_failure

    if output_dir:
        result.artifact_paths = write_path_layer_artifacts(
            output_dir, result,
            preview_image=preview_image,
            image_width=image_width,
            image_height=image_height,
        )
        if first_failure is not None:
            first_failure["debug_overlay_path"] = result.artifact_paths.get(
                "dense_path_validation_overlay.png"
            )
            result.first_failure = first_failure

    return result


def _csv_text(rows: Sequence[dict], cols: Sequence[str]) -> str:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for row in rows:
        w.writerow([row.get(c, "") for c in cols])
    return buf.getvalue()


def render_dense_path_overlay(
    dense_rows: Sequence[dict],
    bad_segments: Sequence[dict],
    preview_image=None,
    *,
    image_width: int = 0,
    image_height: int = 0,
) -> np.ndarray:
    if preview_image is not None and isinstance(preview_image, np.ndarray) and preview_image.size:
        if preview_image.ndim == 2:
            base = cv2.cvtColor(preview_image, cv2.COLOR_GRAY2BGR)
        else:
            base = np.ascontiguousarray(preview_image[:, :, :3].copy())
            if base.dtype != np.uint8:
                base = np.clip(base, 0, 255).astype(np.uint8)
            base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        h, w = base.shape[:2]
        sx = (w / float(image_width)) if image_width > 0 else 1.0
        sy = (h / float(image_height)) if image_height > 0 else 1.0
        ox = oy = 0.0
    else:
        xs = [float(r["x_pixel"]) for r in dense_rows] or [0.0]
        ys = [float(r["y_pixel"]) for r in dense_rows] or [0.0]
        pad = 40.0
        w = max(400, int(max(xs) - min(xs) + 2 * pad))
        h = max(400, int(max(ys) - min(ys) + 2 * pad))
        base = np.zeros((h, w, 3), dtype=np.uint8)
        ox, oy = min(xs) - pad, min(ys) - pad
        sx = sy = 1.0

    def _to(x, y):
        return int((float(x) - ox) * sx), int((float(y) - oy) * sy)

    for i in range(len(dense_rows) - 1):
        a = _to(dense_rows[i]["x_pixel"], dense_rows[i]["y_pixel"])
        b = _to(dense_rows[i + 1]["x_pixel"], dense_rows[i + 1]["y_pixel"])
        cv2.line(base, a, b, (180, 80, 200), 2, cv2.LINE_AA)

    by_gi = {int(r["global_index"]): r for r in dense_rows if r.get("global_index") is not None}
    for seg in bad_segments:
        fa = by_gi.get(int(seg.get("from_index") or -1))
        fb = by_gi.get(int(seg.get("to_index") or -1))
        if not fa or not fb:
            continue
        a = _to(fa["x_pixel"], fa["y_pixel"])
        b = _to(fb["x_pixel"], fb["y_pixel"])
        reason = str(seg.get("reason") or "")
        color = (0, 0, 255) if "aba" in reason else (0, 140, 255)
        cv2.line(base, a, b, color, 3, cv2.LINE_AA)
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        cv2.putText(base, reason[:36], mid, cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return base


def write_path_layer_artifacts(
    output_dir: str,
    result: PathLayerDiagResult,
    *,
    preview_image=None,
    image_width: int = 0,
    image_height: int = 0,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    seg_cols = (
        "segment_index", "task_from_seq", "task_to_seq", "edge_order_index",
        "edge_id", "source_node", "target_node", "planned_from_node", "planned_to_node",
        "edge_has_polyline", "edge_polyline_point_count", "edge_length_m",
        "edge_valid", "reason",
    )
    p1 = os.path.join(output_dir, "planned_segments_debug.csv")
    with open(p1, "w", encoding="utf-8", newline="") as fh:
        fh.write(_csv_text(result.planned_segments_rows, seg_cols))
    paths["planned_segments_debug.csv"] = p1

    dense_cols = (
        "global_index", "segment_index", "task_from_seq", "task_to_seq",
        "edge_id", "edge_point_index", "node_from", "node_to", "polyline_direction",
        "x_pixel", "y_pixel", "longitude", "latitude",
        "s_m", "step_distance_m", "turn_angle_deg", "source",
    )
    p2 = os.path.join(output_dir, "dense_path_debug.csv")
    with open(p2, "w", encoding="utf-8", newline="") as fh:
        fh.write(_csv_text(result.dense_path_rows, dense_cols))
    paths["dense_path_debug.csv"] = p2

    vcols = (
        "task_seq", "original_edge_id", "virtual_node_id", "split_index",
        "left_edge_id", "right_edge_id", "left_polyline_count", "right_polyline_count",
        "left_valid", "right_valid", "reason",
    )
    p3 = os.path.join(output_dir, "virtual_node_split_debug.csv")
    with open(p3, "w", encoding="utf-8", newline="") as fh:
        fh.write(_csv_text(result.virtual_split_rows, vcols))
    paths["virtual_node_split_debug.csv"] = p3

    bad_cols = (
        "segment_index", "from_index", "to_index", "from_s_m", "to_s_m",
        "distance_m", "edge_id", "reason",
    )
    p4 = os.path.join(output_dir, "dense_path_bad_segments.csv")
    with open(p4, "w", encoding="utf-8", newline="") as fh:
        fh.write(_csv_text(result.dense_bad_segments, bad_cols))
    paths["dense_path_bad_segments.csv"] = p4

    report = {
        **result.planned_report,
        **result.dense_report,
        "planned_segments_valid": result.planned_segments_valid,
        "dense_path_valid": result.dense_path_valid,
        "first_failure": result.first_failure,
        "warnings": result.warnings,
    }
    p5 = os.path.join(output_dir, "dense_path_validation_report.json")
    with open(p5, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    paths["dense_path_validation_report.json"] = p5

    overlay = render_dense_path_overlay(
        result.dense_path_rows, result.dense_bad_segments, preview_image,
        image_width=image_width, image_height=image_height,
    )
    p6 = os.path.join(output_dir, "dense_path_validation_overlay.png")
    cv2.imwrite(p6, overlay)
    paths["dense_path_validation_overlay.png"] = p6
    return paths


def classify_aba_source(
    dense_report: dict,
    vehicle_report: dict,
) -> Optional[str]:
    dense_aba = int(dense_report.get("aba_count") or 0)
    veh_aba = int(vehicle_report.get("aba_backtrack_count") or 0)
    if dense_aba > 0:
        return "dense_path"
    if veh_aba > 0:
        return "vehicle_waypoints"
    return None
