"""Unified vehicle-waypoint pipeline (task path → subject1 YAML).

Official mainline only:

  final_graph + task_points + geo_calibration
    → snap → plan route edge path → expand edge.polyline → dense_path
    → zone label → sample vehicle_waypoints → validate → repair
    → subject1_waypoints.yaml

Does NOT re-run mask / skeleton / graph build.
Does NOT sample waypoints from the full graph.
Does NOT invent YAML from global_path_geo bypassing dense_path.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from roadnet.global_planner import (
    EdgeGeometryMissingError,
    PlannerConfig,
    plan_global_path,
)
from roadnet.path_layer_diagnostics import _index_by_id, orient_polyline_for_travel
from roadnet.task_points import TaskPoint, get_plan_sequence
from roadnet.task_snapping import SnapConfig, SnappedTaskPoint, snap_all_task_points


DEFAULT_ALTITUDE_M = 21.741

DENSE_PATH_FIELDS = [
    "dense_index", "segment_index", "task_from_seq", "task_to_seq",
    "edge_id", "edge_point_index", "x_pixel", "y_pixel",
    "latitude_deg", "longitude_deg", "altitude_m",
    "s_m", "step_distance_m", "source",
]
DENSE_LABELED_EXTRA = [
    "spacing_mode", "local_heading_deg", "local_turn_angle_deg",
    "nearest_junction_node_id", "distance_to_junction_m",
]
VEHICLE_WP_FIELDS = [
    "seq", "name", "latitude_deg", "longitude_deg", "altitude_m",
    "x_pixel", "y_pixel", "dense_index", "s_m", "spacing_mode",
    "distance_from_prev_m", "cumulative_distance_m",
    "segment_index", "edge_id", "source_mode", "keep",
]


@dataclass
class PipelineConfig:
    straight_spacing_m: float = 15.0
    curve_spacing_m: float = 2.0
    junction_spacing_m: float = 2.0
    task_spacing_m: float = 2.0
    curve_angle_threshold_deg: float = 15.0
    curve_buffer_m: float = 6.0
    junction_buffer_m: float = 8.0
    task_buffer_m: float = 3.0
    duplicate_distance_m: float = 0.3
    aba_distance_m: float = 0.5
    aba_detour_m: float = 1.0
    max_straight_spacing_m: float = 16.5
    max_curve_spacing_m: float = 3.0
    max_junction_spacing_m: float = 3.0
    max_task_spacing_m: float = 3.0
    default_altitude_m: float = DEFAULT_ALTITUDE_M
    max_insert_iterations: int = 8


@dataclass
class PipelineStatus:
    graph_valid: bool = False
    task_points_loaded: bool = False
    dense_path_generated: bool = False
    vehicle_waypoints_generated: bool = False
    waypoints_checked: bool = False
    waypoints_repaired: bool = False
    yaml_exported: bool = False
    export_ready: bool = False
    usable_for_vehicle: bool = False
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PipelineResult:
    output_dir: str
    status: PipelineStatus = field(default_factory=PipelineStatus)
    snapped_points: list = field(default_factory=list)
    route_segments: list = field(default_factory=list)
    dense_path: list = field(default_factory=list)
    dense_path_labeled: list = field(default_factory=list)
    vehicle_waypoints: list = field(default_factory=list)
    vehicle_waypoints_repaired: list = field(default_factory=list)
    validation_report: dict = field(default_factory=dict)
    repair_report: dict = field(default_factory=dict)
    error: Optional[str] = None
    suggestion: Optional[str] = None


def _mpp(geo_calibration) -> float:
    if geo_calibration is None:
        return 0.5
    for attr in ("pixel_resolution_estimated_m", "metres_per_pixel", "meters_per_pixel"):
        v = getattr(geo_calibration, attr, None)
        if v is not None:
            try:
                f = float(v)
                if f > 1e-9:
                    return f
            except (TypeError, ValueError):
                pass
    return 0.5


def _pixel_to_latlon(geo_calibration, x: float, y: float) -> Tuple[float, float]:
    if geo_calibration is None:
        return float("nan"), float("nan")
    if hasattr(geo_calibration, "pixel_to_wgs84"):
        lon, lat = geo_calibration.pixel_to_wgs84(x, y)
        return float(lat), float(lon)
    if hasattr(geo_calibration, "pixel_to_lonlat"):
        lon, lat = geo_calibration.pixel_to_lonlat(x, y)
        return float(lat), float(lon)
    return float("nan"), float("nan")


def _ensure_task_pixels(task_points: Sequence, geo_calibration) -> List[TaskPoint]:
    out: List[TaskPoint] = []
    for tp in task_points:
        item = tp
        if not isinstance(tp, TaskPoint):
            item = TaskPoint(
                seq=int(tp.get("seq", 0)),
                longitude=tp.get("longitude"),
                latitude=tp.get("latitude"),
                altitude=float(tp.get("altitude", 0) or 0),
                point_type=int(tp.get("point_type", 2)),
                pixel_x=tp.get("pixel_x"),
                pixel_y=tp.get("pixel_y"),
                source=str(tp.get("source", "")),
            )
        if (item.pixel_x is None or item.pixel_y is None) and geo_calibration is not None:
            lon, lat = item.longitude, item.latitude
            if lon is not None and lat is not None:
                if hasattr(geo_calibration, "wgs84_to_pixel"):
                    px, py = geo_calibration.wgs84_to_pixel(float(lon), float(lat))
                    item.pixel_x, item.pixel_y = float(px), float(py)
                elif hasattr(geo_calibration, "lonlat_to_pixel"):
                    px, py = geo_calibration.lonlat_to_pixel(float(lon), float(lat))
                    item.pixel_x, item.pixel_y = float(px), float(py)
        out.append(item)
    return out


def _write_csv(path: str, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for k in fieldnames:
                v = row.get(k)
                if isinstance(v, bool):
                    out[k] = "true" if v else "false"
                elif v is None:
                    out[k] = ""
                else:
                    out[k] = v
            writer.writerow(out)


def _read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _dist_m(a: dict, b: dict) -> float:
    if a.get("s_m") is not None and b.get("s_m") is not None:
        return abs(float(b["s_m"]) - float(a["s_m"]))
    dx = float(a.get("x_pixel", 0)) - float(b.get("x_pixel", 0))
    dy = float(a.get("y_pixel", 0)) - float(b.get("y_pixel", 0))
    return math.hypot(dx, dy)


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _to_float(v, default=float("nan")) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _renumber_waypoints(waypoints: Sequence[dict]) -> List[dict]:
    out = []
    cum = 0.0
    prev = None
    for i, wp in enumerate(waypoints):
        item = dict(wp)
        item["seq"] = i + 1
        item["name"] = f"wp_{i + 1:03d}"
        if prev is None:
            d = 0.0
        else:
            d = _dist_m(prev, item)
            if not math.isfinite(d):
                d = 0.0
        cum += d
        item["distance_from_prev_m"] = round(d, 6)
        item["cumulative_distance_m"] = round(cum, 6)
        out.append(item)
        prev = item
    return out


def _is_anchor_keep(wp: dict) -> bool:
    """Protected anchors: task_anchor / endpoint / keep=True (junction centers etc.)."""
    source = str(wp.get("source_mode") or "").strip().lower()
    if source in ("task_anchor", "endpoint"):
        return True
    return _truthy(wp.get("keep"))


def cleanup_anchor_aware_duplicates(
    waypoints: Sequence[dict],
    config: Optional[PipelineConfig] = None,
) -> Tuple[List[dict], List[str]]:
    """Remove near-duplicate adjacent waypoints without touching protected anchors.

    Rules (distance < duplicate_distance_m, default 0.3m):
      - keep=True + keep=False → drop the non-keep point
      - both keep=True → keep both, emit warning
      - both keep=False → drop the latter
    Does not re-sample; renumbers seq/name and recomputes distances.
    """
    cfg = config or PipelineConfig()
    thr = float(cfg.duplicate_distance_m)
    warnings: List[str] = []
    rows = [dict(w) for w in waypoints]
    if len(rows) < 2:
        return _renumber_waypoints(rows), warnings

    cleaned: List[dict] = [rows[0]]
    for cur in rows[1:]:
        prev = cleaned[-1]
        d = _dist_m(prev, cur)
        if not math.isfinite(d):
            d = 0.0
        if d >= thr - 1e-12:
            cleaned.append(cur)
            continue

        k_prev = _is_anchor_keep(prev)
        k_cur = _is_anchor_keep(cur)
        if k_prev and not k_cur:
            # Drop ordinary non-keep near an anchor
            continue
        if k_cur and not k_prev:
            # Drop previous ordinary point; keep the anchor
            cleaned[-1] = cur
            continue
        if k_prev and k_cur:
            warnings.append(
                "keep-keep near duplicate: "
                f"{prev.get('source_mode') or 'keep'}@{float(prev.get('s_m', 0)):.3f}m "
                f"and {cur.get('source_mode') or 'keep'}@{float(cur.get('s_m', 0)):.3f}m "
                f"(d={d:.4f}m)"
            )
            cleaned.append(cur)
            continue
        # both ordinary: drop the latter
        continue

    return _renumber_waypoints(cleaned), warnings


def default_pipeline_output_dir(base_dir: str, now: Optional[datetime] = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, "outputs", "vehicle_waypoints", f"run_{stamp}")


def snap_task_points_to_graph(
    task_points: Sequence,
    final_graph,
    geo_calibration=None,
    *,
    snap_config: Optional[SnapConfig] = None,
    output_dir: Optional[str] = None,
) -> List[dict]:
    """Snap task points to nearest graph edges; write snapped_task_points.json."""
    nodes = list(getattr(final_graph, "nodes", None) or final_graph.get("nodes", []))
    edges = list(getattr(final_graph, "edges", None) or final_graph.get("edges", []))
    tps = _ensure_task_pixels(get_plan_sequence(list(task_points)), geo_calibration)
    cfg = snap_config or SnapConfig()
    snapped = snap_all_task_points(tps, nodes, edges, cfg)
    mpp = _mpp(geo_calibration)

    rows = []
    for sp, tp in zip(snapped, tps):
        lat, lon = _pixel_to_latlon(geo_calibration, sp.original_x, sp.original_y)
        if tp.latitude is not None:
            lat = float(tp.latitude)
        if tp.longitude is not None:
            lon = float(tp.longitude)
        rows.append({
            "seq": int(sp.seq),
            "point_type": int(sp.point_type),
            "original_longitude": lon,
            "original_latitude": lat,
            "original_x_pixel": float(sp.original_x),
            "original_y_pixel": float(sp.original_y),
            "snapped_x_pixel": float(sp.snapped_x),
            "snapped_y_pixel": float(sp.snapped_y),
            "snapped_edge_id": sp.edge_id,
            "virtual_node_id": sp.virtual_node_id,
            "snap_distance_px": float(sp.snap_distance),
            "snap_distance_m": float(sp.snap_distance) * mpp,
            "status": sp.status,
            "warning": sp.warning,
            "_snapped_obj": sp,
        })

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        serializable = [{k: v for k, v in r.items() if k != "_snapped_obj"} for r in rows]
        path = os.path.join(output_dir, "snapped_task_points.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(serializable, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    return rows


def plan_route_by_task_sequence(
    snapped_rows: Sequence[dict],
    final_graph,
    *,
    algorithm: str = "astar",
    output_dir: Optional[str] = None,
    metres_per_pixel: float = 0.5,
) -> Tuple[List[dict], bool]:
    """Plan edge paths between consecutive snapped task points."""
    nodes = list(getattr(final_graph, "nodes", None) or final_graph.get("nodes", []))
    edges = list(getattr(final_graph, "edges", None) or final_graph.get("edges", []))

    snapped_objs = []
    for row in snapped_rows:
        obj = row.get("_snapped_obj")
        if isinstance(obj, SnappedTaskPoint):
            snapped_objs.append(obj)
        else:
            snapped_objs.append(SnappedTaskPoint(
                seq=int(row["seq"]),
                point_type=int(row["point_type"]),
                original_x=float(row["original_x_pixel"]),
                original_y=float(row["original_y_pixel"]),
                snapped_x=float(row["snapped_x_pixel"]),
                snapped_y=float(row["snapped_y_pixel"]),
                snap_distance=float(row.get("snap_distance_px", 0)),
                edge_id=row.get("snapped_edge_id"),
                virtual_node_id=row.get("virtual_node_id"),
                status=str(row.get("status", "ok")),
            ))

    plan = plan_global_path(
        snapped_objs, nodes, edges,
        PlannerConfig(algorithm=algorithm),
    )
    segments = []
    for i, seg in enumerate(plan.segments or []):
        length_px = float(getattr(seg, "length_px", 0) or 0)
        ok = (
            str(getattr(seg, "status", "")) == "ok"
            and len(getattr(seg, "edge_path", []) or []) > 0
            and not getattr(seg, "error", "")
        )
        segments.append({
            "segment_index": i,
            "task_from_seq": int(getattr(seg, "from_seq", 0)),
            "task_to_seq": int(getattr(seg, "to_seq", 0)),
            "start_node_id": (getattr(seg, "node_path", [None]) or [None])[0],
            "goal_node_id": (getattr(seg, "node_path", [None]) or [None])[-1],
            "edge_ids": list(getattr(seg, "edge_path", []) or []),
            "node_ids": list(getattr(seg, "node_path", []) or []),
            "length_m": length_px * metres_per_pixel,
            "length_px": length_px,
            "success": ok,
            "error": getattr(seg, "error", "") or "",
        })

    all_ok = bool(plan.success) and bool(segments) and all(s["success"] for s in segments)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "route_segments.json"), "w", encoding="utf-8") as fh:
            json.dump({"success": all_ok, "segments": segments}, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    return segments, all_ok


def expand_route_edges_to_dense_path(
    route_segments: Sequence[dict],
    final_graph,
    geo_calibration=None,
    *,
    snapped_rows: Optional[Sequence[dict]] = None,
    default_altitude_m: float = DEFAULT_ALTITUDE_M,
) -> List[dict]:
    """Expand each segment's edge.polyline into ordered dense_path rows.

    Rebuilds a per-segment graph copy with virtual nodes (same as planning),
    because planned edge_ids after split do not exist on the original graph.
    """
    from roadnet.task_snapping import insert_virtual_nodes

    nodes = list(getattr(final_graph, "nodes", None) or final_graph.get("nodes", []))
    edges = list(getattr(final_graph, "edges", None) or final_graph.get("edges", []))
    mpp = _mpp(geo_calibration)

    snapped_by_seq: Dict[int, SnappedTaskPoint] = {}
    for row in snapped_rows or []:
        obj = row.get("_snapped_obj")
        if isinstance(obj, SnappedTaskPoint):
            snapped_by_seq[int(obj.seq)] = obj
        else:
            snapped_by_seq[int(row["seq"])] = SnappedTaskPoint(
                seq=int(row["seq"]),
                point_type=int(row["point_type"]),
                original_x=float(row["original_x_pixel"]),
                original_y=float(row["original_y_pixel"]),
                snapped_x=float(row["snapped_x_pixel"]),
                snapped_y=float(row["snapped_y_pixel"]),
                snap_distance=float(row.get("snap_distance_px", 0)),
                edge_id=row.get("snapped_edge_id"),
                virtual_node_id=row.get("virtual_node_id"),
                status=str(row.get("status", "ok")),
            )

    dense: List[dict] = []
    s_m = 0.0
    dense_index = 0

    for seg in route_segments:
        if not seg.get("success", True):
            raise RuntimeError(
                f"segment {seg.get('segment_index')} planning failed: {seg.get('error')}"
            )
        node_ids = list(seg.get("node_ids") or [])
        edge_ids = list(seg.get("edge_ids") or [])
        if len(node_ids) < 2 or not edge_ids:
            raise RuntimeError(f"segment {seg.get('segment_index')} missing edge path")

        src = snapped_by_seq.get(int(seg.get("task_from_seq", -1)))
        dst = snapped_by_seq.get(int(seg.get("task_to_seq", -1)))
        if src is not None and dst is not None:
            seg_nodes, seg_edges = insert_virtual_nodes(nodes, edges, [src, dst])
        else:
            seg_nodes, seg_edges = list(nodes), list(edges)
        nodes_by_id = _index_by_id(seg_nodes)
        edges_by_id = _index_by_id(seg_edges)

        for ei, eid in enumerate(edge_ids):
            e = edges_by_id.get(eid)
            if e is None and eid is not None:
                e = edges_by_id.get(str(eid))
                if e is None:
                    try:
                        e = edges_by_id.get(int(eid))
                    except (TypeError, ValueError):
                        e = None
            if e is None:
                raise EdgeGeometryMissingError(eid, "edge not found")
            from_n = node_ids[ei]
            to_n = node_ids[ei + 1]
            oriented, _dir, err = orient_polyline_for_travel(
                e, from_n, to_n, nodes_by_id,
            )
            if err or len(oriented) < 2:
                raise EdgeGeometryMissingError(eid, err or "polyline missing")

            start_k = 0
            if dense:
                if (
                    abs(dense[-1]["x_pixel"] - oriented[0][0]) < 0.5
                    and abs(dense[-1]["y_pixel"] - oriented[0][1]) < 0.5
                ):
                    start_k = 1

            for pi in range(start_k, len(oriented)):
                x, y = float(oriented[pi][0]), float(oriented[pi][1])
                step = 0.0
                if dense:
                    dx = x - dense[-1]["x_pixel"]
                    dy = y - dense[-1]["y_pixel"]
                    step = math.hypot(dx, dy) * mpp
                    s_m += step
                lat, lon = _pixel_to_latlon(geo_calibration, x, y)
                dense.append({
                    "dense_index": dense_index,
                    "segment_index": int(seg.get("segment_index", 0)),
                    "task_from_seq": int(seg.get("task_from_seq", 0)),
                    "task_to_seq": int(seg.get("task_to_seq", 0)),
                    "edge_id": eid,
                    "edge_point_index": pi,
                    "x_pixel": x,
                    "y_pixel": y,
                    "latitude_deg": lat,
                    "longitude_deg": lon,
                    "altitude_m": float(default_altitude_m),
                    "s_m": round(s_m, 6),
                    "step_distance_m": round(step, 6),
                    "source": "segment_join" if (pi == start_k and start_k == 1) else "edge_polyline",
                })
                dense_index += 1

    return dense


def densify_dense_path_rows(
    dense_path: Sequence[dict],
    step_m: float = 1.0,
) -> List[dict]:
    """Densify polyline rows to ~step_m before zone classify / sampling.

    Does not change graph; only export-stage densification of dense_path.
    """
    rows = list(dense_path)
    if len(rows) < 2:
        return rows
    step = max(0.2, float(step_m))
    out: List[dict] = []
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        if not out:
            out.append(dict(a))
        sa, sb = float(a["s_m"]), float(b["s_m"])
        seg = abs(sb - sa)
        if seg <= step * 1.01:
            out.append(dict(b))
            continue
        n_insert = max(1, int(math.ceil(seg / step)))
        for k in range(1, n_insert + 1):
            t = k / n_insert
            item = dict(a)
            item["x_pixel"] = float(a["x_pixel"]) + t * (float(b["x_pixel"]) - float(a["x_pixel"]))
            item["y_pixel"] = float(a["y_pixel"]) + t * (float(b["y_pixel"]) - float(a["y_pixel"]))
            item["s_m"] = sa + t * (sb - sa)
            la = _to_float(a.get("latitude_deg"))
            lb = _to_float(b.get("latitude_deg"))
            loa = _to_float(a.get("longitude_deg"))
            lob = _to_float(b.get("longitude_deg"))
            if math.isfinite(la) and math.isfinite(lb):
                item["latitude_deg"] = la + t * (lb - la)
                item["longitude_deg"] = loa + t * (lob - loa)
            item["step_distance_m"] = abs(item["s_m"] - float(out[-1]["s_m"]))
            item["source"] = "densify" if k < n_insert else b.get("source", "edge_polyline")
            item["edge_id"] = a.get("edge_id") if t < 1.0 else b.get("edge_id")
            item["segment_index"] = a.get("segment_index") if t < 1.0 else b.get("segment_index")
            item["task_from_seq"] = a.get("task_from_seq")
            item["task_to_seq"] = a.get("task_to_seq")
            out.append(item)
    # Renormalize s_m from 0 and re-index
    base = float(out[0]["s_m"]) if out else 0.0
    for i, r in enumerate(out):
        r["dense_index"] = i
        r["s_m"] = round(float(r["s_m"]) - base, 6)
        if i == 0:
            r["step_distance_m"] = 0.0
        else:
            r["step_distance_m"] = round(abs(float(r["s_m"]) - float(out[i - 1]["s_m"])), 6)
    return out


def export_dense_path_csv(dense_path: Sequence[dict], output_path: str) -> str:
    _write_csv(output_path, dense_path, DENSE_PATH_FIELDS)
    return output_path


def classify_dense_path_zones(
    dense_path: Sequence[dict],
    final_graph,
    snapped_rows: Optional[Sequence[dict]] = None,
    config: Optional[PipelineConfig] = None,
    *,
    metres_per_pixel: float = 0.5,
) -> List[dict]:
    """Label dense_path points with spacing_mode: straight/curve/junction/task.

    Classification is intentionally tight:
    - junction: only graph nodes with degree>=3 that lie near the route
      (spatial attach), then ±junction_buffer_m along path
    - curve: only local turn-angle peaks ≥ threshold, then ±curve_buffer_m
    - task: ±task_buffer_m around snapped task anchors
    - else straight

    Priority: task > junction > curve > straight
    """
    cfg = config or PipelineConfig()
    nodes = list(getattr(final_graph, "nodes", None) or final_graph.get("nodes", []))
    edges = list(getattr(final_graph, "edges", None) or final_graph.get("edges", []))
    mpp = float(metres_per_pixel) if metres_per_pixel and metres_per_pixel > 0 else 0.5
    # Attach threshold: junction must be near the route, not merely nearest in the whole map
    max_attach_px = max(12.0, 3.0 / max(mpp, 1e-6))

    degree: Dict[Any, int] = {}
    for e in edges:
        if e.get("enabled") is False:
            continue
        for key in ("start", "end"):
            nid = e.get(key)
            degree[nid] = degree.get(nid, 0) + 1

    junction_nodes = []
    for n in nodes:
        nid = n.get("id")
        if degree.get(nid, 0) >= 3:
            junction_nodes.append((nid, float(n.get("x", 0)), float(n.get("y", 0))))

    rows = [dict(r) for r in dense_path]
    n = len(rows)
    if n == 0:
        return rows

    # Heading / turn using along-path neighbours ~2m away (less noise than adjacent vertices)
    headings = [0.0] * n
    turns = [0.0] * n
    look_m = 2.0

    def _index_at_s_offset(i: int, delta_m: float) -> int:
        target = float(rows[i]["s_m"]) + delta_m
        if delta_m >= 0:
            j = i
            while j + 1 < n and float(rows[j]["s_m"]) < target:
                j += 1
            return j
        j = i
        while j - 1 >= 0 and float(rows[j]["s_m"]) > target:
            j -= 1
        return j

    for i in range(n):
        i0 = _index_at_s_offset(i, -look_m)
        i1 = _index_at_s_offset(i, look_m)
        if i0 == i1:
            if i + 1 < n:
                i1 = i + 1
            elif i - 1 >= 0:
                i0 = i - 1
        dx = float(rows[i1]["x_pixel"]) - float(rows[i0]["x_pixel"])
        dy = float(rows[i1]["y_pixel"]) - float(rows[i0]["y_pixel"])
        headings[i] = math.degrees(math.atan2(dy, dx)) % 360.0

    for i in range(1, n - 1):
        d = abs(headings[i + 1] - headings[i - 1])
        if d > 180:
            d = 360 - d
        turns[i] = d

    modes = ["straight"] * n
    nearest_j = [None] * n
    dist_j = [float("inf")] * n

    # Junction: only if the node is spatially near the dense path
    for jid, jx, jy in junction_nodes:
        best_i, best_d = None, float("inf")
        for i, r in enumerate(rows):
            d = math.hypot(float(r["x_pixel"]) - jx, float(r["y_pixel"]) - jy)
            if d < best_d:
                best_d, best_i = d, i
        if best_i is None or best_d > max_attach_px:
            continue  # junction not on this route
        s0 = float(rows[best_i]["s_m"])
        for i, r in enumerate(rows):
            along = abs(float(r["s_m"]) - s0)
            if along <= cfg.junction_buffer_m:
                if along < dist_j[i]:
                    dist_j[i] = along
                    nearest_j[i] = jid
                if modes[i] != "task":
                    modes[i] = "junction"

    # Curve: only local maxima of turn angle above threshold
    for i in range(1, n - 1):
        turn = turns[i]
        if turn < cfg.curve_angle_threshold_deg:
            continue
        if turn < turns[i - 1] or turn < turns[i + 1]:
            continue  # not a local peak
        s0 = float(rows[i]["s_m"])
        for j, r in enumerate(rows):
            if abs(float(r["s_m"]) - s0) <= cfg.curve_buffer_m:
                if modes[j] == "straight":
                    modes[j] = "curve"

    # Task buffers (highest priority)
    if snapped_rows:
        for sp in snapped_rows:
            sx = float(sp["snapped_x_pixel"])
            sy = float(sp["snapped_y_pixel"])
            best_i, best_d = None, float("inf")
            for i, r in enumerate(rows):
                d = math.hypot(float(r["x_pixel"]) - sx, float(r["y_pixel"]) - sy)
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is None:
                continue
            s0 = float(rows[best_i]["s_m"])
            rows[best_i]["source"] = "task_anchor"
            for j, r in enumerate(rows):
                if abs(float(r["s_m"]) - s0) <= cfg.task_buffer_m:
                    modes[j] = "task"

    for i, r in enumerate(rows):
        r["spacing_mode"] = modes[i]
        r["local_heading_deg"] = round(headings[i], 3)
        r["local_turn_angle_deg"] = round(turns[i], 3)
        r["nearest_junction_node_id"] = nearest_j[i]
        r["distance_to_junction_m"] = (
            None if not math.isfinite(dist_j[i]) else round(dist_j[i], 3)
        )
    return rows


def export_dense_path_labeled_csv(rows: Sequence[dict], output_path: str) -> str:
    fields = DENSE_PATH_FIELDS + DENSE_LABELED_EXTRA
    _write_csv(output_path, rows, fields)
    return output_path


def _spacing_for_mode(mode: str, cfg: PipelineConfig) -> float:
    if mode == "curve":
        return cfg.curve_spacing_m
    if mode == "junction":
        return cfg.junction_spacing_m
    if mode == "task":
        return cfg.task_spacing_m
    return cfg.straight_spacing_m


def _priority_mode(modes: set) -> str:
    cleaned = {str(m).strip().lower() for m in modes if m}
    if "task" in cleaned:
        return "task"
    if "junction" in cleaned:
        return "junction"
    if "curve" in cleaned:
        return "curve"
    return "straight"


def _find_dense_bracket(
    dense_path: Sequence[dict], target_s_m: float,
) -> Tuple[dict, dict, float]:
    """Return (p0, p1, ratio) for target_s_m on dense_path."""
    rows = list(dense_path)
    if not rows:
        raise ValueError("dense_path is empty")
    if len(rows) == 1:
        return rows[0], rows[0], 0.0
    s = float(target_s_m)
    s0 = float(rows[0]["s_m"])
    s1 = float(rows[-1]["s_m"])
    if s <= s0:
        return rows[0], rows[0], 0.0
    if s >= s1:
        return rows[-1], rows[-1], 0.0
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        sa, sb = float(a["s_m"]), float(b["s_m"])
        if sa <= s <= sb:
            if sb <= sa + 1e-12:
                return a, b, 0.0
            return a, b, (s - sa) / (sb - sa)
    return rows[-2], rows[-1], 1.0


def spacing_mode_at_s(
    dense_path_labeled: Sequence[dict],
    target_s_m: float,
) -> str:
    """spacing_mode for target_s by priority on the dense bracket interval."""
    p0, p1, _ = _find_dense_bracket(dense_path_labeled, target_s_m)
    return _priority_mode({
        p0.get("spacing_mode") or "straight",
        p1.get("spacing_mode") or "straight",
    })


def spacing_mode_for_s_interval(
    dense_path_labeled: Sequence[dict],
    s_a: float,
    s_b: float,
) -> str:
    """Highest-priority spacing_mode overlapping [s_a, s_b] on dense_path."""
    lo, hi = (float(s_a), float(s_b)) if s_a <= s_b else (float(s_b), float(s_a))
    modes = set()
    rows = list(dense_path_labeled)
    for i, r in enumerate(rows):
        s = float(r["s_m"])
        if lo - 1e-9 <= s <= hi + 1e-9:
            modes.add(r.get("spacing_mode") or "straight")
        if i + 1 < len(rows):
            s_next = float(rows[i + 1]["s_m"])
            # segment overlaps interval
            if not (s_next < lo or s > hi):
                modes.add(r.get("spacing_mode") or "straight")
                modes.add(rows[i + 1].get("spacing_mode") or "straight")
    if not modes:
        modes.add(spacing_mode_at_s(dense_path_labeled, 0.5 * (lo + hi)))
    return _priority_mode(modes)


def interpolate_dense_path_at_s(
    dense_path: Sequence[dict],
    target_s_m: float,
) -> dict:
    """Linearly interpolate a point on dense_path at target cumulative s_m.

    Works even when adjacent dense vertices are far apart.
    """
    rows = list(dense_path)
    if not rows:
        raise ValueError("dense_path is empty")
    p0, p1, ratio = _find_dense_bracket(rows, target_s_m)
    s = float(target_s_m)
    s = max(float(rows[0]["s_m"]), min(s, float(rows[-1]["s_m"])))

    def _lerp(a, b):
        fa, fb = _to_float(a), _to_float(b)
        if not math.isfinite(fa) and not math.isfinite(fb):
            return float("nan")
        if not math.isfinite(fa):
            return fb
        if not math.isfinite(fb):
            return fa
        return fa + ratio * (fb - fa)

    mode = _priority_mode({
        p0.get("spacing_mode") or "straight",
        p1.get("spacing_mode") or "straight",
    })
    di0 = _to_float(p0.get("dense_index"), 0.0)
    di1 = _to_float(p1.get("dense_index"), di0)
    return {
        "x_pixel": _lerp(p0.get("x_pixel"), p1.get("x_pixel")),
        "y_pixel": _lerp(p0.get("y_pixel"), p1.get("y_pixel")),
        "latitude_deg": _lerp(p0.get("latitude_deg"), p1.get("latitude_deg")),
        "longitude_deg": _lerp(p0.get("longitude_deg"), p1.get("longitude_deg")),
        "altitude_m": _lerp(
            p0.get("altitude_m", DEFAULT_ALTITUDE_M),
            p1.get("altitude_m", DEFAULT_ALTITUDE_M),
        ),
        "s_m": round(s, 6),
        "dense_index": round(di0 + ratio * (di1 - di0), 6),
        "segment_index": p0.get("segment_index"),
        "edge_id": p0.get("edge_id"),
        "spacing_mode": mode,
        "task_from_seq": p0.get("task_from_seq"),
        "task_to_seq": p0.get("task_to_seq"),
    }


def _source_mode_for_sample(mode: str, *, keep: bool, is_endpoint: bool, is_task: bool) -> str:
    if is_endpoint:
        return "endpoint"
    if is_task:
        return "task_anchor"
    if mode == "curve":
        return "curve_sample"
    if mode == "junction":
        return "junction_sample"
    if mode == "straight":
        return "straight_sample"
    return "interpolated_sample"


def sample_vehicle_waypoints_from_dense_path(
    dense_path_labeled: Sequence[dict],
    snapped_rows: Optional[Sequence[dict]] = None,
    config: Optional[PipelineConfig] = None,
) -> List[dict]:
    """Sample vehicle waypoints along dense_path s_m with interpolation.

    straight → 15m; curve / junction / task → 2m.
    Does NOT only pick existing dense vertices.
    """
    cfg = config or PipelineConfig()
    rows = sorted(list(dense_path_labeled), key=lambda r: float(r["s_m"]))
    if len(rows) < 2:
        return []

    s_start = float(rows[0]["s_m"])
    s_end = float(rows[-1]["s_m"])
    if s_end <= s_start:
        return []

    # Mandatory s positions (task anchors + ends)
    must_s: List[Tuple[float, str]] = [
        (s_start, "endpoint"),
        (s_end, "endpoint"),
    ]
    task_s: List[float] = []
    if snapped_rows:
        for sp in snapped_rows:
            sx = float(sp["snapped_x_pixel"])
            sy = float(sp["snapped_y_pixel"])
            best_i, best_d = 0, float("inf")
            for i, r in enumerate(rows):
                d = math.hypot(float(r["x_pixel"]) - sx, float(r["y_pixel"]) - sy)
                if d < best_d:
                    best_d, best_i = d, i
            s_t = float(rows[best_i]["s_m"])
            task_s.append(s_t)
            must_s.append((s_t, "task_anchor"))

    # Junction centers / curve peaks as keep anchors (optional densify anchors)
    junction_best: Dict[Any, Tuple[float, float]] = {}
    for r in rows:
        if str(r.get("spacing_mode")) != "junction":
            continue
        jid = r.get("nearest_junction_node_id")
        dist = float(r.get("distance_to_junction_m") or 1e9)
        if jid is None:
            continue
        prev = junction_best.get(jid)
        if prev is None or dist < prev[1]:
            junction_best[jid] = (float(r["s_m"]), dist)
    for s_j, _d in junction_best.values():
        must_s.append((s_j, "junction_keep"))

    for i, r in enumerate(rows):
        if str(r.get("spacing_mode")) != "curve":
            continue
        turn = float(r.get("local_turn_angle_deg") or 0)
        if turn < cfg.curve_angle_threshold_deg:
            continue
        prev_t = float(rows[i - 1].get("local_turn_angle_deg") or 0) if i > 0 else 0
        next_t = float(rows[i + 1].get("local_turn_angle_deg") or 0) if i + 1 < len(rows) else 0
        if turn >= prev_t and turn >= next_t:
            must_s.append((float(r["s_m"]), "curve_keep"))

    # Unique must s (tolerance 0.05m)
    must_s_sorted = sorted(must_s, key=lambda x: x[0])
    must_unique: List[Tuple[float, str]] = []
    for s_m, tag in must_s_sorted:
        if not must_unique or abs(s_m - must_unique[-1][0]) > 0.05:
            must_unique.append((s_m, tag))
        else:
            # Prefer task_anchor / endpoint tags
            if tag in ("task_anchor", "endpoint") and must_unique[-1][1] not in (
                "task_anchor", "endpoint",
            ):
                must_unique[-1] = (must_unique[-1][0], tag)

    task_set = {round(s, 3) for s in task_s}

    # Walk along s_m with adaptive step + interpolation
    sample_s: List[Tuple[float, str]] = []  # (s, reason_tag)
    sample_s.append((s_start, "endpoint"))
    cur = s_start
    guard = 0
    max_guard = int(max(1000, (s_end - s_start) / 0.5 + 100))
    while cur < s_end - 1e-6 and guard < max_guard:
        guard += 1
        mode_here = spacing_mode_at_s(rows, cur)
        step = _spacing_for_mode(mode_here, cfg)
        # If the upcoming interval requires denser spacing, use denser step
        probe = min(cur + step, s_end)
        interval_mode = spacing_mode_for_s_interval(rows, cur, probe)
        denser = _spacing_for_mode(interval_mode, cfg)
        if denser + 1e-9 < step:
            step = denser
            probe = min(cur + step, s_end)

        # Next mandatory anchor in (cur, probe]
        next_must = None
        next_tag = "interpolated_sample"
        for s_m, tag in must_unique:
            if cur + 1e-6 < s_m <= probe + 1e-9:
                next_must = s_m
                next_tag = tag
                break

        if next_must is not None:
            cur = float(next_must)
            sample_s.append((cur, next_tag))
        else:
            cur = float(probe)
            if cur >= s_end - 1e-6:
                cur = s_end
                sample_s.append((cur, "endpoint"))
            else:
                sample_s.append((cur, "interpolated_sample"))

    if abs(sample_s[-1][0] - s_end) > 1e-6:
        sample_s.append((s_end, "endpoint"))

    # Deduplicate nearly equal s
    cleaned: List[Tuple[float, str]] = []
    for s_m, tag in sample_s:
        if not cleaned or abs(s_m - cleaned[-1][0]) > 0.05:
            cleaned.append((s_m, tag))
        elif tag in ("endpoint", "task_anchor"):
            cleaned[-1] = (cleaned[-1][0], tag)

    waypoints = []
    for s_m, tag in cleaned:
        pt = interpolate_dense_path_at_s(rows, s_m)
        mode = str(pt.get("spacing_mode") or spacing_mode_at_s(rows, s_m)).strip().lower()
        if mode not in ("straight", "curve", "junction", "task"):
            mode = "straight"
        is_endpoint = tag == "endpoint" or abs(s_m - s_start) < 1e-6 or abs(s_m - s_end) < 1e-6
        is_task = tag == "task_anchor" or round(s_m, 3) in task_set
        keep = is_endpoint or is_task or tag in ("junction_keep", "curve_keep")
        source = _source_mode_for_sample(
            mode, keep=keep, is_endpoint=is_endpoint, is_task=is_task,
        )
        waypoints.append({
            "seq": 0,
            "name": "",
            "latitude_deg": pt.get("latitude_deg"),
            "longitude_deg": pt.get("longitude_deg"),
            "altitude_m": pt.get("altitude_m", cfg.default_altitude_m),
            "x_pixel": pt["x_pixel"],
            "y_pixel": pt["y_pixel"],
            "dense_index": pt.get("dense_index"),
            "s_m": float(pt["s_m"]),
            "spacing_mode": mode,
            "distance_from_prev_m": 0.0,
            "cumulative_distance_m": 0.0,
            "segment_index": pt.get("segment_index"),
            "edge_id": pt.get("edge_id"),
            "source_mode": source,
            "keep": bool(keep),
        })
    return _renumber_waypoints(waypoints)


def thin_straight_waypoints(
    waypoints: Sequence[dict],
    config: Optional[PipelineConfig] = None,
) -> List[dict]:
    """Thin over-dense consecutive straight points to ~straight_spacing_m.

    Keeps: path ends, keep=True, curve/junction/task, task_anchor.
    Never leaves a straight gap larger than max_straight_spacing_m.
    """
    cfg = config or PipelineConfig()
    wps = [dict(w) for w in waypoints]
    if len(wps) < 3:
        return _renumber_waypoints(wps)

    target = float(cfg.straight_spacing_m)
    max_gap = float(cfg.max_straight_spacing_m)

    def _force_keep(w: dict) -> bool:
        mode = str(w.get("spacing_mode") or "straight").strip().lower()
        source = str(w.get("source_mode") or "")
        return (
            _truthy(w.get("keep"))
            or mode in ("curve", "junction", "task")
            or source == "task_anchor"
        )

    force = [_force_keep(w) for w in wps]
    force[0] = True
    force[-1] = True

    selected = set()
    # Process runs between consecutive force-keep anchors
    anchors = [i for i, f in enumerate(force) if f]
    if not anchors:
        anchors = [0, len(wps) - 1]
    if anchors[0] != 0:
        anchors = [0] + anchors
    if anchors[-1] != len(wps) - 1:
        anchors = anchors + [len(wps) - 1]

    for ai in range(len(anchors) - 1):
        left, right = anchors[ai], anchors[ai + 1]
        selected.add(left)
        selected.add(right)
        s_left = float(wps[left]["s_m"])
        s_right = float(wps[right]["s_m"])
        gap = abs(s_right - s_left)
        interior = list(range(left + 1, right))
        if not interior:
            continue
        all_straight = all(
            str(wps[i].get("spacing_mode") or "straight").strip().lower() == "straight"
            and not force[i]
            for i in interior
        )
        if not all_straight:
            for i in interior:
                selected.add(i)
            continue

        # Walk interior: keep ~every target, but never allow gap to right > max_gap
        last_s = s_left
        for i in interior:
            s = float(wps[i]["s_m"])
            # Must keep if skipping this point would make last→right exceed max_gap
            # and this point is still needed as a stepping stone.
            remain_if_skip = abs(s_right - last_s)
            next_after = None
            for j in interior:
                if j > i:
                    next_after = j
                    break
            # Keep when spacing from last reaches target
            if s - last_s >= target - 1e-6:
                selected.add(i)
                last_s = s
                continue
            # Keep if without this point we cannot reach right within max_gap
            # even using later interior points as bridges
            if remain_if_skip > max_gap + 1e-6:
                # Check whether later points alone can bridge last_s → s_right
                can_bridge = False
                probe_s = last_s
                for j in interior:
                    if j <= i:
                        continue
                    sj = float(wps[j]["s_m"])
                    if sj - probe_s <= max_gap + 1e-6:
                        probe_s = sj
                    if s_right - probe_s <= max_gap + 1e-6:
                        can_bridge = True
                        break
                if s_right - probe_s <= max_gap + 1e-6:
                    can_bridge = True
                if not can_bridge and (s - last_s) <= max_gap + 1e-6:
                    selected.add(i)
                    last_s = s

        # Final safety: if selected gap still > max_gap, add densest available points
        run_sel = sorted(
            [left, right] + [i for i in interior if i in selected]
        )
        for si in range(len(run_sel) - 1):
            a_i, b_i = run_sel[si], run_sel[si + 1]
            sa, sb = float(wps[a_i]["s_m"]), float(wps[b_i]["s_m"])
            if abs(sb - sa) <= max_gap + 1e-6:
                continue
            # Insert existing interior points greedily
            mid_candidates = [i for i in interior if a_i < i < b_i]
            cur_s = sa
            for i in mid_candidates:
                s = float(wps[i]["s_m"])
                if s - cur_s >= min(target, max_gap) - 1e-6 or (
                    sb - cur_s > max_gap + 1e-6 and sb - s <= max_gap + 1e-6
                ):
                    selected.add(i)
                    cur_s = s
            # Re-check; if still too large, keep all mids in this subgap
            if abs(sb - cur_s) > max_gap + 1e-6:
                for i in mid_candidates:
                    selected.add(i)

    ordered = sorted(selected)
    return _renumber_waypoints([wps[i] for i in ordered])


def build_vehicle_waypoint_summary(
    waypoints: Sequence[dict],
    dense_path_labeled: Optional[Sequence[dict]] = None,
) -> dict:
    """Build vehicle_waypoint_summary.json stats (post-sample spacing validation)."""
    wps = list(waypoints)
    counts = {"straight": 0, "curve": 0, "junction": 0, "task": 0}
    for w in wps:
        mode = str(w.get("spacing_mode") or "straight").strip().lower()
        if mode not in counts:
            mode = "straight"
        counts[mode] += 1

    all_sp: List[float] = []
    for i in range(len(wps) - 1):
        d = abs(float(wps[i + 1]["s_m"]) - float(wps[i]["s_m"]))
        all_sp.append(d)

    def _mode_spacings(mode: str) -> List[float]:
        out = []
        for i in range(len(wps) - 1):
            a, b = wps[i], wps[i + 1]
            ma = str(a.get("spacing_mode") or "straight").strip().lower()
            mb = str(b.get("spacing_mode") or "straight").strip().lower()
            if mode == "straight":
                if ma == "straight" and mb == "straight":
                    out.append(abs(float(b["s_m"]) - float(a["s_m"])))
            else:
                if ma == mode or mb == mode:
                    out.append(abs(float(b["s_m"]) - float(a["s_m"])))
        return out

    straight_sp = _mode_spacings("straight")
    curve_sp = _mode_spacings("curve")
    junction_sp = _mode_spacings("junction")
    task_sp = _mode_spacings("task")

    dense_counts = {"straight": 0, "curve": 0, "junction": 0, "task": 0}
    if dense_path_labeled:
        for r in dense_path_labeled:
            mode = str(r.get("spacing_mode") or "straight").strip().lower()
            if mode not in dense_counts:
                mode = "straight"
            dense_counts[mode] += 1

    def _avg(vals: List[float]):
        return None if not vals else round(sum(vals) / len(vals), 3)

    def _max(vals: List[float]):
        return None if not vals else round(max(vals), 3)

    straight_avg = _avg(straight_sp)
    summary = {
        "waypoint_count": len(wps),
        "total_waypoint_count": len(wps),
        "average_spacing_m": _avg(all_sp),
        "max_spacing_m": _max(all_sp),
        "straight_waypoint_count": counts["straight"],
        "curve_waypoint_count": counts["curve"],
        "junction_waypoint_count": counts["junction"],
        "task_waypoint_count": counts["task"],
        "straight_average_spacing_m": straight_avg,
        "straight_max_spacing_m": _max(straight_sp),
        "curve_average_spacing_m": _avg(curve_sp),
        "curve_max_spacing_m": _max(curve_sp),
        "junction_average_spacing_m": _avg(junction_sp),
        "junction_max_spacing_m": _max(junction_sp),
        "task_average_spacing_m": _avg(task_sp),
        "task_max_spacing_m": _max(task_sp),
        "straight_waypoints_too_dense": bool(
            straight_avg is not None and straight_avg < 10.0
        ),
        "dense_path_mode_counts": dense_counts,
        "dense_path_point_count": sum(dense_counts.values()),
        "pass_straight_max_le_16_5": bool(
            not straight_sp or max(straight_sp) <= 16.5 + 1e-6
        ),
        "pass_curve_max_le_3": bool(not curve_sp or max(curve_sp) <= 3.0 + 1e-6),
        "pass_junction_max_le_3": bool(
            not junction_sp or max(junction_sp) <= 3.0 + 1e-6
        ),
        "pass_task_max_le_3": bool(not task_sp or max(task_sp) <= 3.0 + 1e-6),
    }
    return summary


def export_vehicle_waypoint_summary(summary: dict, output_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return output_path


def export_vehicle_waypoints_csv(waypoints: Sequence[dict], output_path: str) -> str:
    _write_csv(output_path, waypoints, VEHICLE_WP_FIELDS)
    return output_path


def _pair_spacing_info(
    a: dict,
    b: dict,
    dense_labeled: Sequence[dict],
    cfg: PipelineConfig,
) -> Tuple[str, float, int, int]:
    """Resolve pair_spacing_mode + allowed_m from dense_path interval.

    Mode is chosen by **path-length majority** inside
    [from_dense_index, to_dense_index] (not “any junction/task exists”).
    Priority task > junction > curve > straight is only a tie-breaker.

    Returns (pair_spacing_mode, allowed_m, from_dense_index, to_dense_index).
    """
    di = _to_float(a.get("dense_index"), -1.0)
    dj = _to_float(b.get("dense_index"), -1.0)
    from_f, to_f = (di, dj) if di <= dj else (dj, di)
    from_i = int(math.floor(from_f))
    to_i = int(math.ceil(to_f))

    interval_rows: List[dict] = []
    if dense_labeled:
        for row in dense_labeled:
            idx = _to_float(row.get("dense_index"), -10**9)
            if from_f - 1e-9 <= idx <= to_f + 1e-9:
                interval_rows.append(row)
        if not interval_rows and 0 <= from_i < len(dense_labeled) and 0 <= to_i < len(dense_labeled):
            interval_rows = list(dense_labeled[from_i:to_i + 1])

    length_by_mode = {"straight": 0.0, "curve": 0.0, "junction": 0.0, "task": 0.0}
    if len(interval_rows) >= 2:
        ordered = sorted(
            interval_rows,
            key=lambda r: (_to_float(r.get("s_m"), 0.0), _to_float(r.get("dense_index"), 0.0)),
        )
        for i in range(len(ordered) - 1):
            ds = abs(float(ordered[i + 1]["s_m"]) - float(ordered[i]["s_m"]))
            mode = str(ordered[i].get("spacing_mode") or "straight").strip().lower()
            if mode not in length_by_mode:
                mode = "straight"
            length_by_mode[mode] += ds
    elif interval_rows:
        mode = str(interval_rows[0].get("spacing_mode") or "straight").strip().lower()
        if mode not in length_by_mode:
            mode = "straight"
        length_by_mode[mode] = 1.0
    else:
        for wp in (a, b):
            mode = str(wp.get("spacing_mode") or "straight").strip().lower()
            if mode not in length_by_mode:
                mode = "straight"
            length_by_mode[mode] += 1.0

    # Majority by length; tie-break by priority
    priority = {"task": 3, "junction": 2, "curve": 1, "straight": 0}
    pair_mode = max(
        length_by_mode.keys(),
        key=lambda m: (length_by_mode[m], priority[m]),
    )
    # If majority is straight but a tiny higher-priority sliver exists, keep straight
    # unless non-straight covers >= 35% of the interval length.
    total = sum(length_by_mode.values()) or 1.0
    non_straight = total - length_by_mode["straight"]
    if length_by_mode["straight"] >= length_by_mode[pair_mode] and (non_straight / total) < 0.35:
        pair_mode = "straight"
    elif pair_mode == "straight":
        pass
    else:
        # Keep chosen non-straight majority
        pass

    if pair_mode == "task":
        allowed = float(cfg.max_task_spacing_m)
    elif pair_mode == "junction":
        allowed = float(cfg.max_junction_spacing_m)
    elif pair_mode == "curve":
        allowed = float(cfg.max_curve_spacing_m)
    else:
        allowed = float(cfg.max_straight_spacing_m)
    return pair_mode, allowed, from_i, to_i


def _target_spacing_for_mode(mode: str, cfg: PipelineConfig) -> float:
    """Insert target spacing (not the max-allowed threshold)."""
    if mode in ("curve", "junction", "task"):
        return float(cfg.curve_spacing_m)  # 2.0
    return float(cfg.straight_spacing_m)  # 15.0


def _max_allowed_for_pair(
    a: dict, b: dict, dense_labeled: Sequence[dict], cfg: PipelineConfig,
) -> float:
    """Backward-compatible wrapper → allowed_m only."""
    _mode, allowed, _fi, _ti = _pair_spacing_info(a, b, dense_labeled, cfg)
    return allowed


def validate_vehicle_waypoints_csv(
    vehicle_waypoints,
    *,
    dense_path_labeled: Optional[Sequence[dict]] = None,
    snapped_rows: Optional[Sequence[dict]] = None,
    config: Optional[PipelineConfig] = None,
    output_dir: Optional[str] = None,
) -> Tuple[dict, List[dict]]:
    """Validate vehicle waypoints only (no graph / layered diagnostics)."""
    cfg = config or PipelineConfig()
    if isinstance(vehicle_waypoints, str):
        wps = _read_csv(vehicle_waypoints)
    else:
        wps = [dict(w) for w in vehicle_waypoints]

    errors: List[str] = []
    warnings: List[str] = []
    bad: List[dict] = []

    if wps:
        missing = [
            f for f in ("seq", "name", "latitude_deg", "longitude_deg", "dense_index", "s_m")
            if f not in wps[0]
        ]
        if missing:
            errors.append(f"CSV 字段缺失: {missing}")

    for w in wps:
        w["seq"] = int(_to_float(w.get("seq"), 0))
        # Keep float dense_index (repair may insert fractional indices)
        w["dense_index"] = _to_float(w.get("dense_index"), -1.0)
        w["s_m"] = _to_float(w.get("s_m"), float("nan"))
        w["latitude_deg"] = _to_float(w.get("latitude_deg"))
        w["longitude_deg"] = _to_float(w.get("longitude_deg"))
        w["keep"] = _truthy(w.get("keep"))
        if "x_pixel" in w:
            w["x_pixel"] = _to_float(w.get("x_pixel"), 0.0)
        if "y_pixel" in w:
            w["y_pixel"] = _to_float(w.get("y_pixel"), 0.0)

    csv_valid = not bool(errors)
    if len(wps) < 2:
        errors.append("航点数量 < 2")
        csv_valid = False

    for i, w in enumerate(wps):
        if w["seq"] != i + 1:
            errors.append(f"seq 不连续: index={i} seq={w['seq']}")
            break
        expect_name = f"wp_{i + 1:03d}"
        if str(w.get("name")) != expect_name:
            warnings.append(f"name 不连续: 期望 {expect_name} 实际 {w.get('name')}")

    coord_valid = True
    for w in wps:
        lat, lon = w["latitude_deg"], w["longitude_deg"]
        if not math.isfinite(lat) or not math.isfinite(lon) or abs(lat) > 90 or abs(lon) > 180:
            coord_valid = False
            errors.append(f"坐标非法: {w.get('name')} lat={lat} lon={lon}")
            break

    dense_ok = all(
        wps[i]["dense_index"] <= wps[i + 1]["dense_index"] for i in range(len(wps) - 1)
    ) if len(wps) >= 2 else False
    if not dense_ok and len(wps) >= 2:
        errors.append("dense_index 非单调递增")

    s_ok = all(
        wps[i]["s_m"] <= wps[i + 1]["s_m"] + 1e-6 for i in range(len(wps) - 1)
    ) if len(wps) >= 2 else False
    if not s_ok and len(wps) >= 2:
        errors.append("s_m 非单调递增")

    dup = aba = spacing_v = 0
    straight_v = curve_v = junction_v = task_v = 0
    spacings = []
    dense_list = list(dense_path_labeled or [])

    for i in range(len(wps) - 1):
        a, b = wps[i], wps[i + 1]
        d = abs(float(b["s_m"]) - float(a["s_m"]))
        spacings.append(d)
        if d < cfg.duplicate_distance_m:
            if _is_anchor_keep(a) and _is_anchor_keep(b):
                warnings.append(
                    "keep-keep near duplicate retained: "
                    f"{a.get('name')}({a.get('source_mode')}) → "
                    f"{b.get('name')}({b.get('source_mode')}) d={d:.4f}m"
                )
            else:
                dup += 1
                bad.append({
                    "from_seq": a["seq"],
                    "to_seq": b["seq"],
                    "from_dense_index": a["dense_index"],
                    "to_dense_index": b["dense_index"],
                    "pair_spacing_mode": "",
                    "allowed_m": "",
                    "distance_m": round(d, 6),
                    "reason": "consecutive_duplicate",
                })
        pair_mode, allowed, from_di, to_di = _pair_spacing_info(
            a, b, dense_list, cfg,
        )
        if d > allowed + 1e-6:
            spacing_v += 1
            if pair_mode == "task":
                task_v += 1
            elif pair_mode == "junction":
                junction_v += 1
            elif pair_mode == "curve":
                curve_v += 1
            else:
                straight_v += 1
            bad.append({
                "from_seq": a["seq"],
                "to_seq": b["seq"],
                "from_dense_index": from_di,
                "to_dense_index": to_di,
                "pair_spacing_mode": pair_mode,
                "allowed_m": allowed,
                "distance_m": round(d, 6),
                "reason": "spacing_violation",
            })

    for i in range(1, len(wps) - 1):
        a, b, c = wps[i - 1], wps[i], wps[i + 1]
        dac = abs(float(c["s_m"]) - float(a["s_m"]))
        dab = abs(float(b["s_m"]) - float(a["s_m"]))
        if dac < cfg.aba_distance_m and dab > cfg.aba_detour_m:
            aba += 1
            bad.append({
                "from_seq": a["seq"],
                "to_seq": c["seq"],
                "from_dense_index": a["dense_index"],
                "to_dense_index": c["dense_index"],
                "pair_spacing_mode": "",
                "allowed_m": "",
                "distance_m": round(dac, 6),
                "reason": "aba_backtrack",
                "mid_seq": b["seq"],
            })

    task_missing = 0
    if snapped_rows and wps:
        for sp in snapped_rows:
            sx = float(sp["snapped_x_pixel"])
            sy = float(sp["snapped_y_pixel"])
            ok = any(
                math.hypot(float(w.get("x_pixel", 0)) - sx, float(w.get("y_pixel", 0)) - sy) < 1.5
                for w in wps
            )
            if not ok:
                task_missing += 1
                errors.append(f"任务点锚点缺失: seq={sp.get('seq')}")

    max_sp = max(spacings) if spacings else 0.0
    avg_sp = (sum(spacings) / len(spacings)) if spacings else 0.0
    total_len = float(wps[-1]["s_m"] - wps[0]["s_m"]) if len(wps) >= 2 else 0.0

    export_ready = (
        csv_valid and coord_valid and dense_ok and s_ok
        and dup == 0 and aba == 0 and spacing_v == 0 and task_missing == 0
        and len(wps) >= 2 and not any("非单调" in e or "非法" in e or "缺失" in e for e in errors)
    )
    # Keep export_ready false if hard errors remain (except name warnings)
    hard_errors = [e for e in errors if "name 不连续" not in e]
    if hard_errors:
        export_ready = False

    report = {
        "waypoint_count": len(wps),
        "total_length_m": round(total_len, 3),
        "average_spacing_m": round(avg_sp, 3),
        "max_spacing_m": round(max_sp, 3),
        "duplicate_consecutive_count": dup,
        "aba_backtrack_count": aba,
        "spacing_violation_count": spacing_v,
        "straight_spacing_violation_count": straight_v,
        "curve_spacing_violation_count": curve_v,
        "junction_spacing_violation_count": junction_v,
        "task_spacing_violation_count": task_v,
        "task_anchor_missing_count": task_missing,
        "dense_index_order_valid": dense_ok,
        "s_m_order_valid": s_ok,
        "coordinate_valid": coord_valid,
        "csv_valid": csv_valid,
        "export_ready": export_ready,
        "warnings": warnings,
        "errors": errors,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "waypoint_validation_report.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        bad_fields = [
            "from_seq", "to_seq", "from_dense_index", "to_dense_index",
            "pair_spacing_mode", "allowed_m", "distance_m", "reason", "mid_seq",
        ]
        _write_csv(os.path.join(output_dir, "bad_waypoint_segments.csv"), bad, bad_fields)

    return report, bad


def _interp_dense(dense: Sequence[dict], target_s: float) -> dict:
    """Linear interpolate along dense_path by s_m (never chord midpoints of WPs)."""
    if not dense:
        raise ValueError("cannot_interpolate_dense_path: empty dense_path")
    out = interpolate_dense_path_at_s(dense, target_s)
    out["source"] = "inserted_for_repair"
    return out


def _slice_dense_by_index(
    dense: Sequence[dict], from_i: int, to_i: int,
) -> Tuple[List[dict], Optional[str]]:
    """Return dense rows with dense_index in [from_i, to_i], ordered by s_m."""
    if from_i > to_i:
        return [], "dense_index_order_invalid"
    rows = []
    for row in dense:
        idx = int(_to_float(row.get("dense_index"), -10**9))
        if from_i <= idx <= to_i:
            rows.append(dict(row))
    if not rows and 0 <= from_i < len(dense) and 0 <= to_i < len(dense):
        rows = [dict(r) for r in dense[from_i:to_i + 1]]
    if not rows:
        return [], "dense_interval_empty"
    rows.sort(key=lambda r: (_to_float(r.get("s_m"), 0.0), _to_float(r.get("dense_index"), 0)))
    return rows, None


def repair_vehicle_waypoints_using_dense_path(
    vehicle_waypoints: Sequence[dict],
    dense_path_labeled: Sequence[dict],
    validation_report: Optional[dict] = None,
    config: Optional[PipelineConfig] = None,
    *,
    output_dir: Optional[str] = None,
) -> Tuple[List[dict], dict]:
    """Repair only true spacing_violation / dup / ABA using dense_path inserts."""
    cfg = config or PipelineConfig()
    dense = list(dense_path_labeled)
    wps = [dict(w) for w in vehicle_waypoints]
    for w in wps:
        w["keep"] = _truthy(w.get("keep"))
        w["dense_index"] = _to_float(w.get("dense_index"), -1)
        w["s_m"] = _to_float(w.get("s_m"), 0.0)

    actions: List[str] = []
    unresolved: List[str] = []
    failure_reasons: List[str] = []

    # 1) consecutive duplicates
    cleaned = []
    for w in wps:
        if not cleaned:
            cleaned.append(w)
            continue
        prev = cleaned[-1]
        d = abs(float(w["s_m"]) - float(prev["s_m"]))
        if d < cfg.duplicate_distance_m:
            if w["keep"] and not prev["keep"]:
                cleaned[-1] = w
                actions.append(f"drop non-keep duplicate before {w.get('name')}")
            elif not w["keep"]:
                actions.append(f"drop consecutive duplicate {w.get('name')}")
            else:
                prefer_w = (
                    str(w.get("source_mode")) == "task_anchor"
                    and str(prev.get("source_mode")) != "task_anchor"
                )
                if prefer_w:
                    cleaned[-1] = w
                actions.append(f"merge keep-duplicate near {w.get('name')}")
        else:
            cleaned.append(w)
    wps = cleaned

    # 2) ABA
    i = 1
    while i < len(wps) - 1:
        a, b, c = wps[i - 1], wps[i], wps[i + 1]
        dac = abs(float(c["s_m"]) - float(a["s_m"]))
        dab = abs(float(b["s_m"]) - float(a["s_m"]))
        if dac < cfg.aba_distance_m and dab > cfg.aba_detour_m:
            if not b["keep"]:
                actions.append(f"remove ABA mid {b.get('name')}")
                del wps[i]
                continue
            unresolved.append(f"keep_point_causes_unresolved_aba at {b.get('name')}")
            failure_reasons.append("keep_point_causes_unresolved_aba")
        i += 1

    # 3) spacing inserts — only when distance_m > allowed_m
    for _ in range(cfg.max_insert_iterations):
        inserted_any = False
        if len(wps) < 2:
            break
        new_wps = [wps[0]]
        for i in range(len(wps) - 1):
            a, b = wps[i], wps[i + 1]
            d = abs(float(b["s_m"]) - float(a["s_m"]))
            pair_mode, allowed, from_di, to_di = _pair_spacing_info(a, b, dense, cfg)
            if from_di > to_di:
                failure_reasons.append("dense_index_order_invalid")
                unresolved.append(f"dense_index_order_invalid {from_di}->{to_di}")
                new_wps.append(b)
                continue
            # Do NOT insert for straight gaps <= 16.5 (or any gap <= allowed)
            if d <= allowed + 1e-6:
                new_wps.append(b)
                continue

            if from_di < 0 or to_di < 0:
                failure_reasons.append("dense_index_missing")
                unresolved.append(f"dense_index_missing pair {a.get('name')}->{b.get('name')}")
                new_wps.append(b)
                continue

            slice_dense, slice_err = _slice_dense_by_index(dense, from_di, to_di)
            if slice_err or len(slice_dense) < 1:
                failure_reasons.append(slice_err or "dense_interval_empty")
                unresolved.append(
                    f"{slice_err or 'dense_interval_empty'} {from_di}->{to_di}"
                )
                new_wps.append(b)
                continue

            target = _target_spacing_for_mode(pair_mode, cfg)
            n_need = int(math.ceil(d / target)) - 1
            n_need = max(1, min(n_need, 20))
            sa = float(slice_dense[0]["s_m"])
            sb = float(slice_dense[-1]["s_m"])
            try:
                for k in range(1, n_need + 1):
                    target_s = float(a["s_m"]) + d * (k / (n_need + 1))
                    target_s = min(max(target_s, sa), sb)
                    mid = _interp_dense(slice_dense, target_s)
                    new_wps.append({
                        "seq": 0,
                        "name": "",
                        "latitude_deg": mid.get("latitude_deg"),
                        "longitude_deg": mid.get("longitude_deg"),
                        "altitude_m": mid.get("altitude_m", cfg.default_altitude_m),
                        "x_pixel": mid["x_pixel"],
                        "y_pixel": mid["y_pixel"],
                        "dense_index": mid["dense_index"],
                        "s_m": float(mid["s_m"]),
                        "spacing_mode": mid.get("spacing_mode", pair_mode),
                        "distance_from_prev_m": 0.0,
                        "cumulative_distance_m": 0.0,
                        "segment_index": mid.get("segment_index"),
                        "edge_id": mid.get("edge_id"),
                        "source_mode": "inserted_for_repair",
                        "keep": False,
                    })
                    inserted_any = True
                    actions.append(
                        f"insert for {pair_mode} "
                        f"allowed={allowed} target={target} s={target_s:.2f}"
                    )
            except ValueError as exc:
                failure_reasons.append("cannot_interpolate_dense_path")
                unresolved.append(str(exc))
            new_wps.append(b)
        wps = _renumber_waypoints(new_wps)
        if not inserted_any:
            break

    # Final consecutive-duplicate sweep
    final = []
    for w in wps:
        if not final:
            final.append(w)
            continue
        prev = final[-1]
        d = abs(float(w["s_m"]) - float(prev["s_m"]))
        if d < cfg.duplicate_distance_m and not w["keep"]:
            actions.append(f"final drop duplicate {w.get('name')}")
            continue
        if d < cfg.duplicate_distance_m and w["keep"] and not prev["keep"]:
            final[-1] = w
            continue
        if d < cfg.duplicate_distance_m and w["keep"] and prev["keep"]:
            actions.append(f"final merge keep-duplicate near {w.get('name')}")
            continue
        final.append(w)
    wps = _renumber_waypoints(final)

    # Note: do NOT thin after s_m-interpolated sampling — thinning previously
    # created straight gaps > max_straight_spacing_m (e.g. 17m). Sampling already
    # targets straight=15m / curve|junction|task=2m.

    # Re-validate to detect unresolved spacing
    post_report, post_bad = validate_vehicle_waypoints_csv(
        wps, dense_path_labeled=dense, config=cfg, output_dir=None,
    )
    if int(post_report.get("spacing_violation_count", 0) or 0) > 0:
        failure_reasons.append("spacing_violation_unresolved")
        unresolved.append(
            f"spacing_violation_unresolved count="
            f"{post_report.get('spacing_violation_count')}"
        )

    # Unique failure reasons
    seen = set()
    uniq_failures = []
    for fr in failure_reasons:
        if fr not in seen:
            seen.add(fr)
            uniq_failures.append(fr)

    report = {
        "actions": actions,
        "unresolved": unresolved,
        "failure_reasons": uniq_failures,
        "repair_failed": uniq_failures,
        "repaired_count": len(wps),
        "input_validation": validation_report or {},
        "post_validation": {
            "export_ready": post_report.get("export_ready"),
            "spacing_violation_count": post_report.get("spacing_violation_count"),
            "duplicate_consecutive_count": post_report.get("duplicate_consecutive_count"),
            "aba_backtrack_count": post_report.get("aba_backtrack_count"),
            "dense_index_order_valid": post_report.get("dense_index_order_valid"),
            "s_m_order_valid": post_report.get("s_m_order_valid"),
        },
        "repair_success": bool(post_report.get("export_ready")),
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        export_vehicle_waypoints_csv(
            wps, os.path.join(output_dir, "vehicle_waypoints_repaired.csv"),
        )
        summary = build_vehicle_waypoint_summary(wps, dense)
        export_vehicle_waypoint_summary(
            summary, os.path.join(output_dir, "vehicle_waypoint_summary.json"),
        )
        with open(os.path.join(output_dir, "waypoint_repair_report.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        if not report["repair_success"]:
            with open(
                os.path.join(output_dir, "waypoint_repair_failed_report.json"),
                "w", encoding="utf-8",
            ) as fh:
                json.dump({
                    "failure_reasons": uniq_failures,
                    "unresolved": unresolved,
                    "post_validation": report["post_validation"],
                }, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            bad_fields = [
                "from_seq", "to_seq", "from_dense_index", "to_dense_index",
                "pair_spacing_mode", "allowed_m", "distance_m", "reason", "mid_seq",
            ]
            _write_csv(
                os.path.join(output_dir, "bad_waypoint_segments_after_repair.csv"),
                post_bad, bad_fields,
            )
    return wps, report


def export_subject1_yaml_from_vehicle_csv(
    vehicle_csv_or_rows,
    output_path: str,
    *,
    default_altitude_m: float = DEFAULT_ALTITUDE_M,
    also_debug_copy: bool = True,
) -> str:
    """Format-only conversion: repaired CSV → subject1_waypoints.yaml."""
    if isinstance(vehicle_csv_or_rows, str):
        rows = _read_csv(vehicle_csv_or_rows)
    else:
        rows = [dict(r) for r in vehicle_csv_or_rows]
    rows = sorted(rows, key=lambda r: int(_to_float(r.get("seq"), 0)))

    lines = ["subject1_waypoints:", "  waypoints:"]
    for i, r in enumerate(rows):
        name = str(r.get("name") or f"wp_{i + 1:03d}")
        lat = _to_float(r.get("latitude_deg"))
        lon = _to_float(r.get("longitude_deg"))
        alt = _to_float(r.get("altitude_m"), default_altitude_m)
        if not math.isfinite(lat) or not math.isfinite(lon):
            raise ValueError(f"非法经纬度: {name}")
        lines.append(f"    - name: {name}")
        lines.append(f"      latitude_deg: {lat:.8f}")
        lines.append(f"      longitude_deg: {lon:.8f}")
        lines.append(f"      altitude_m: {alt:.3f}")
    text = "\n".join(lines) + "\n"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(text)

    if also_debug_copy:
        debug_path = os.path.join(os.path.dirname(output_path), "waypoints_debug.yaml")
        with open(debug_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return output_path


def evaluate_usable_for_vehicle(
    status: PipelineStatus,
    validation_report: dict,
    yaml_path: Optional[str],
) -> bool:
    if not (
        status.graph_valid
        and status.task_points_loaded
        and status.dense_path_generated
        and status.waypoints_repaired
        and status.yaml_exported
        and yaml_path
        and os.path.isfile(yaml_path)
    ):
        return False
    return bool(
        validation_report.get("export_ready")
        and int(validation_report.get("duplicate_consecutive_count", 1)) == 0
        and int(validation_report.get("aba_backtrack_count", 1)) == 0
        and int(validation_report.get("spacing_violation_count", 1)) == 0
        and bool(validation_report.get("dense_index_order_valid"))
        and bool(validation_report.get("s_m_order_valid"))
    )


def run_vehicle_waypoint_pipeline(
    final_graph,
    task_points: Sequence,
    geo_calibration,
    output_dir: str,
    *,
    config: Optional[PipelineConfig] = None,
    algorithm: str = "astar",
) -> PipelineResult:
    """Run the full official pipeline into output_dir."""
    cfg = config or PipelineConfig()
    os.makedirs(output_dir, exist_ok=True)
    result = PipelineResult(output_dir=output_dir)
    status = result.status

    nodes = list(getattr(final_graph, "nodes", None) or final_graph.get("nodes", []))
    edges = list(getattr(final_graph, "edges", None) or final_graph.get("edges", []))
    status.graph_valid = bool(nodes and edges)
    status.task_points_loaded = len(list(task_points or [])) >= 2

    if not status.graph_valid:
        result.error = "final_graph 不存在或为空"
        result.suggestion = "请先生成并检查 final_graph"
        return result
    if not status.task_points_loaded:
        result.error = "未生成 dense_path：请先输入任务点并规划路径"
        result.suggestion = "请导入或手动设置任务点"
        return result
    if geo_calibration is None or not bool(getattr(geo_calibration, "is_valid", True)):
        result.error = "geo_calibration 无效"
        result.suggestion = "请先完成坐标校准"
        return result

    snapped = snap_task_points_to_graph(
        task_points, final_graph, geo_calibration, output_dir=output_dir,
    )
    result.snapped_points = snapped
    if any(r.get("status") == "failed" for r in snapped):
        result.error = "任务点吸附失败"
        result.suggestion = "检查任务点是否落在路网附近"
        return result

    segments, ok = plan_route_by_task_sequence(
        snapped, final_graph, algorithm=algorithm, output_dir=output_dir,
        metres_per_pixel=_mpp(geo_calibration),
    )
    result.route_segments = segments
    if not ok:
        result.error = "dense_path 为空：任务点之间路径规划失败"
        result.suggestion = "请检查任务点连通性或 final_graph 边几何"
        return result

    try:
        dense = expand_route_edges_to_dense_path(
            segments, final_graph, geo_calibration,
            snapped_rows=snapped,
            default_altitude_m=cfg.default_altitude_m,
        )
    except Exception as exc:
        result.error = f"dense_path 展开失败: {exc}"
        result.suggestion = "edge geometry missing：需要修 graph edge polyline"
        return result
    if len(dense) < 2:
        result.error = "dense_path 为空：任务点之间路径规划失败"
        result.suggestion = "请先输入任务点并规划路径"
        return result
    # Densify before classify/sample so zone buffers don't claim whole polyline spans
    dense = densify_dense_path_rows(dense, step_m=1.0)
    export_dense_path_csv(dense, os.path.join(output_dir, "dense_path.csv"))
    result.dense_path = dense
    status.dense_path_generated = True

    labeled = classify_dense_path_zones(
        dense, final_graph, snapped, cfg,
        metres_per_pixel=_mpp(geo_calibration),
    )
    export_dense_path_labeled_csv(
        labeled, os.path.join(output_dir, "dense_path_labeled.csv"),
    )
    result.dense_path_labeled = labeled

    wps = sample_vehicle_waypoints_from_dense_path(labeled, snapped, cfg)
    wps, dup_warnings = cleanup_anchor_aware_duplicates(wps, cfg)
    export_vehicle_waypoints_csv(wps, os.path.join(output_dir, "vehicle_waypoints.csv"))
    result.vehicle_waypoints = wps
    status.vehicle_waypoints_generated = True
    summary = build_vehicle_waypoint_summary(wps, labeled)
    if dup_warnings:
        summary["anchor_duplicate_warnings"] = list(dup_warnings)
    export_vehicle_waypoint_summary(
        summary, os.path.join(output_dir, "vehicle_waypoint_summary.json"),
    )

    report, _bad = validate_vehicle_waypoints_csv(
        wps, dense_path_labeled=labeled, snapped_rows=snapped,
        config=cfg, output_dir=output_dir,
    )
    if dup_warnings:
        report.setdefault("warnings", [])
        report["warnings"] = list(report.get("warnings") or []) + list(dup_warnings)
        if output_dir:
            with open(
                os.path.join(output_dir, "waypoint_validation_report.json"),
                "w", encoding="utf-8",
            ) as fh:
                json.dump(report, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
    result.validation_report = report
    status.waypoints_checked = True

    repaired, repair_report = repair_vehicle_waypoints_using_dense_path(
        wps, labeled, report, cfg, output_dir=output_dir,
    )
    result.vehicle_waypoints_repaired = repaired
    result.repair_report = repair_report
    status.waypoints_repaired = True

    report2, _ = validate_vehicle_waypoints_csv(
        repaired, dense_path_labeled=labeled, snapped_rows=snapped,
        config=cfg, output_dir=output_dir,
    )
    result.validation_report = report2

    if not report2.get("export_ready"):
        result.error = "YAML 未生成：航点 CSV 检查未通过"
        if report2.get("spacing_violation_count"):
            result.error = "航点点距不符合：请点击自动修复航点 CSV"
            result.suggestion = "请查看 bad_waypoint_segments.csv"
        elif repair_report.get("repair_failed"):
            result.error = "自动修复失败：请查看 bad_waypoint_segments.csv"
            result.suggestion = "请查看 waypoint_repair_report.json"
        else:
            result.suggestion = "请查看 waypoint_validation_report.json / bad_waypoint_segments.csv"
        status.message = result.error
        return result

    yaml_path = export_subject1_yaml_from_vehicle_csv(
        repaired,
        os.path.join(output_dir, "subject1_waypoints.yaml"),
        default_altitude_m=cfg.default_altitude_m,
    )
    status.yaml_exported = True
    status.export_ready = True
    status.usable_for_vehicle = evaluate_usable_for_vehicle(status, report2, yaml_path)
    status.message = "可用于小车" if status.usable_for_vehicle else "已导出但未完全满足可用条件"
    return result
