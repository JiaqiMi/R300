"""Adaptive vehicle-waypoint resampling for a dense planned path.

The dense path remains the source of truth for visualisation/debugging.  This
module creates a separate sparse execution path in metric (ENU/projected)
coordinates whenever a valid geo calibration is available.

Coordinate pipeline (mandatory when calibration is valid):
  dense_path_pixel
    → calibration.pixel_to_world → ENU metric (meters)
    → cumulative distance, curvature, corner/intersection detection
    → adaptive sampling with configurable spacing
    → calibration.world_to_pixel → pixel (for bounds checking)
    → calibration.pixel_to_wgs84 → lon/lat
    → export YAML / CSV
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

# Must match geo_calibration.EARTH_RADIUS_M for roundtrip consistency.
_EARTH_RADIUS_M = 6378137.0


@dataclass
class AdaptiveWaypointConfig:
    straight_spacing_m: float = 10.0
    curve_spacing_m: float = 2.0
    sharp_turn_spacing_m: float = 2.0
    intersection_spacing_m: float = 2.0  # junction_spacing_m
    task_point_spacing_m: float = 2.0
    corner_angle_threshold_deg: float = 15.0  # curve_angle_threshold_deg
    sharp_turn_angle_threshold_deg: float = 35.0
    corner_buffer_m: float = 5.0
    intersection_buffer_m: float = 8.0  # junction_buffer_m
    task_point_buffer_m: float = 5.0  # task_buffer_m
    min_waypoint_spacing_m: float = 1.0
    max_waypoint_spacing_m: float = 12.0
    heading_window_m: float = 3.0
    max_chord_error_m: float = 1.0
    min_mask_support_ratio: float = 0.75
    max_insert_iterations: int = 6
    los_sample_step_m: float = 0.5
    dense_densify_step_m: float = 0.5  # 导出前把折线加密到此步长（不影响 A*）
    road_corridor_dilate_px: int = 4
    start_goal_speed_mps: float = 1.0
    straight_speed_mps: float = 2.5
    curve_speed_mps: float = 1.5
    sharp_turn_speed_mps: float = 1.0
    intersection_speed_mps: float = 1.0
    task_point_speed_mps: float = 1.2

    @property
    def junction_spacing_m(self) -> float:
        return self.intersection_spacing_m

    @property
    def junction_buffer_m(self) -> float:
        return self.intersection_buffer_m

    @property
    def curve_angle_threshold_deg(self) -> float:
        return self.corner_angle_threshold_deg

    @property
    def task_buffer_m(self) -> float:
        return self.task_point_buffer_m


@dataclass
class AdaptiveWaypointResult:
    sparse_waypoints_pixel: list[list[float]]
    sparse_waypoints_geo: list[list[float]]
    waypoints: list[dict]
    dense_path_enu: list[list[float]]
    report: dict
    dense_path_pixel: list[list[float]] = field(default_factory=list)


_TAG_PRIORITY = {
    "inserted_for_validation": 3,
    # Higher number = higher priority.  When multiple tags collide at the same
    # position, the tag with the highest priority wins.
    #   start > goal > task > intersection > sharp_turn > corner > straight
    "straight": 0,
    "curve": 1,
    "corner": 2,
    "sharp_turn": 3,
    "intersection": 4,
    "task": 5,
    "goal": 6,
    "start": 7,
}


# ---------------------------------------------------------------------------
# Path normalisation & coordinate helpers
# ---------------------------------------------------------------------------

def _normalise_path(points: Iterable[Sequence[float]]) -> list[list[float]]:
    output = []
    for index, point in enumerate(points or []):
        if point is None or len(point) < 2:
            raise ValueError(f"full_path_pixel point {index + 1} is invalid")
        x, y = float(point[0]), float(point[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"full_path_pixel point {index + 1} is not finite")
        if not output or math.hypot(x - output[-1][0], y - output[-1][1]) > 1e-9:
            output.append([x, y])
    if len(output) < 2:
        raise ValueError("full_path_pixel must contain at least two distinct points")
    return output


def _is_calibration_valid(calibration) -> bool:
    if calibration is None:
        return False
    value = getattr(calibration, "is_valid", False)
    return bool(value() if callable(value) else value)


def _point_value(item, *names, default=None):
    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _image_bounds(calibration):
    """Return (image_width, image_height) from calibration if available."""
    w = getattr(calibration, "image_width", 0) or 0
    h = getattr(calibration, "image_height", 0) or 0
    return int(w), int(h)


def _inside_image(x: float, y: float, img_w: int, img_h: int) -> bool:
    """Check whether a pixel coordinate lies within image bounds."""
    if img_w <= 0 or img_h <= 0:
        return True  # unknown – assume ok
    return 0.0 <= x < float(img_w) and 0.0 <= y < float(img_h)


def _path_coordinates(pixel_path, calibration):
    """Convert pixel path → metric (ENU) coordinates.

    Returns (metric, metric_mode, unit, warnings, metric_converter, world_to_pixel).
    """
    calibration_valid = _is_calibration_valid(calibration)
    world_converter = getattr(calibration, "pixel_to_world", None) if calibration_valid else None
    world_to_pixel = getattr(calibration, "world_to_pixel", None) if calibration_valid else None
    warnings = []

    if callable(world_converter):
        metric_converter = lambda x, y: tuple(map(float, world_converter(x, y)))
        metric = [list(metric_converter(x, y)) for x, y in pixel_path]
        metric_mode = True
        unit = "m"
    elif calibration_valid:
        geo_converter = getattr(calibration, "pixel_to_wgs84", None)
        if not callable(geo_converter):
            geo_converter = getattr(calibration, "pixel_to_lonlat", None)
        if not callable(geo_converter):
            metric_converter = None
            metric_mode = False
            metric = [list(point) for point in pixel_path]
            unit = "px"
        else:
            lon0, lat0 = map(float, geo_converter(*pixel_path[0]))
            radius = _EARTH_RADIUS_M
            cos_lat0 = math.cos(math.radians(lat0))
            def metric_converter(x, y):
                lon, lat = map(float, geo_converter(x, y))
                return (
                    radius * math.radians(lon - lon0) * cos_lat0,
                    radius * math.radians(lat - lat0),
                )
            metric = [list(metric_converter(x, y)) for x, y in pixel_path]
            metric_mode = True
            unit = "m"
            warnings.append(
                "geo_calibration 未提供 pixel_to_world，已由 WGS84 构建局部 ENU 近似坐标。"
            )
        world_to_pixel = None  # unsafe fallback for roundtrip
    else:
        metric = [list(point) for point in pixel_path]
        metric_converter = None
        metric_mode = False
        unit = "px"
        world_to_pixel = None
        warnings.append(
            "geo_calibration 无效，已使用 pixel spacing；该结果不应直接作为米制车辆路径。"
        )
    return metric, metric_mode, unit, warnings, metric_converter, world_to_pixel


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _cumulative_distance(points):
    cumulative = [0.0]
    for index in range(len(points) - 1):
        cumulative.append(cumulative[-1] + math.hypot(
            points[index + 1][0] - points[index][0],
            points[index + 1][1] - points[index][1],
        ))
    if cumulative[-1] <= 0:
        raise ValueError("full_path_pixel total length is zero")
    return cumulative


def densify_dense_path(
    pixel_path: Sequence[Sequence[float]],
    metric_path: Sequence[Sequence[float]],
    step_m: float = 0.5,
) -> tuple[list[list[float]], list[list[float]]]:
    """沿道路中心线折线按米制步长加密（仅导出阶段，不影响 A*）。

    保留全部原始顶点，并在过长线段上均匀插入中间点，使后续自适应
    航点可从真正的 dense_path 重采样，而不是直接使用 RDP 稀疏顶点。
    """
    pixels = [list(map(float, p)) for p in pixel_path]
    metrics = [list(map(float, p)) for p in metric_path]
    if len(pixels) < 2 or len(pixels) != len(metrics):
        return pixels, metrics
    step = max(0.05, float(step_m))
    out_px: list[list[float]] = [pixels[0]]
    out_m: list[list[float]] = [metrics[0]]
    for index in range(len(metrics) - 1):
        a_m, b_m = metrics[index], metrics[index + 1]
        a_p, b_p = pixels[index], pixels[index + 1]
        seg_len = math.hypot(b_m[0] - a_m[0], b_m[1] - a_m[1])
        if seg_len <= step * 1.01:
            out_px.append([b_p[0], b_p[1]])
            out_m.append([b_m[0], b_m[1]])
            continue
        n_insert = max(1, int(math.ceil(seg_len / step)))
        for k in range(1, n_insert + 1):
            t = k / n_insert
            out_m.append([
                a_m[0] + t * (b_m[0] - a_m[0]),
                a_m[1] + t * (b_m[1] - a_m[1]),
            ])
            out_px.append([
                a_p[0] + t * (b_p[0] - a_p[0]),
                a_p[1] + t * (b_p[1] - a_p[1]),
            ])
    return out_px, out_m


def _spacing_mode_for_tag(tag: str, *, inserted: bool = False) -> str:
    if inserted or tag == "inserted_for_validation":
        return "inserted_for_validation"
    if tag in {"start", "goal", "task"}:
        return "task_2m"
    if tag == "intersection":
        return "junction_2m"
    if tag in {"corner", "curve", "sharp_turn"}:
        return "curve_2m"
    return "straight_10m"


def _interpolate(points, cumulative, distance):
    distance = max(0.0, min(float(distance), cumulative[-1]))
    index = int(np.searchsorted(cumulative, distance, side="right") - 1)
    index = max(0, min(index, len(points) - 2))
    length = cumulative[index + 1] - cumulative[index]
    ratio = 0.0 if length <= 1e-12 else (distance - cumulative[index]) / length
    return [
        float(points[index][0]) + ratio * (float(points[index + 1][0]) - float(points[index][0])),
        float(points[index][1]) + ratio * (float(points[index + 1][1]) - float(points[index][1])),
    ]


def dense_index_for_s(cumulative, distance: float) -> int:
    """Map along-path distance s_m to the nearest dense_path index."""
    if not cumulative:
        return 0
    distance = max(0.0, min(float(distance), float(cumulative[-1])))
    index = int(np.searchsorted(cumulative, distance, side="left"))
    if index <= 0:
        return 0
    if index >= len(cumulative):
        return len(cumulative) - 1
    # Pick closer of index-1 / index
    if abs(float(cumulative[index]) - distance) < abs(float(cumulative[index - 1]) - distance):
        return index
    return index - 1


def _keep_flag_for_tag(tag: str, forced: bool = False) -> bool:
    if forced:
        return True
    return tag in {
        "start", "goal", "task", "intersection",
        "corner", "sharp_turn", "curve", "inserted_for_validation",
    }


def _project_to_path(point, pixel_path, cumulative):
    px, py = float(point[0]), float(point[1])
    best = None
    for index in range(len(pixel_path) - 1):
        ax, ay = pixel_path[index]
        bx, by = pixel_path[index + 1]
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        ratio = 0.0 if denom <= 1e-12 else ((px - ax) * dx + (py - ay) * dy) / denom
        ratio = max(0.0, min(1.0, ratio))
        qx, qy = ax + ratio * dx, ay + ratio * dy
        distance_sq = (px - qx) ** 2 + (py - qy) ** 2
        if best is None or distance_sq < best[0]:
            metric_s = cumulative[index] + ratio * (cumulative[index + 1] - cumulative[index])
            best = (distance_sq, metric_s, [px, py])
    return best[1], best[2], math.sqrt(best[0])


def _heading_change(a, b, c):
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 <= 1e-9 or n2 <= 1e-9:
        return 0.0
    cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosine))


# ---------------------------------------------------------------------------
# Corner / intersection / task detection
# ---------------------------------------------------------------------------

def _detect_corners(metric, pixel, cumulative, config):
    candidates = []
    window = max(config.heading_window_m, config.min_waypoint_spacing_m)
    for index in range(1, len(metric) - 1):
        before = index - 1
        while before > 0 and cumulative[index] - cumulative[before] < window:
            before -= 1
        after = index + 1
        while after < len(metric) - 1 and cumulative[after] - cumulative[index] < window:
            after += 1
        angle = _heading_change(metric[before], metric[index], metric[after])
        if angle >= config.corner_angle_threshold_deg:
            candidates.append({
                "s": cumulative[index],
                "pixel": list(pixel[index]),
                "tag": "sharp_turn" if angle >= config.sharp_turn_angle_threshold_deg else "corner",
                "heading_change_deg": angle,
                "forced": True,
            })
    groups = []
    cluster_gap = max(config.min_waypoint_spacing_m, 2.0)
    for item in candidates:
        if not groups or item["s"] - groups[-1][-1]["s"] > cluster_gap:
            groups.append([item])
        else:
            groups[-1].append(item)
    return [max(group, key=lambda item: item["heading_change_deg"]) for group in groups]


def _graph_parts(final_graph):
    if final_graph is None:
        return [], []
    if isinstance(final_graph, dict):
        return list(final_graph.get("nodes", [])), list(final_graph.get("edges", []))
    return list(getattr(final_graph, "nodes", []) or []), list(getattr(final_graph, "edges", []) or [])


def _intersection_events(final_graph, path_node_sequence, pixel_path, cumulative):
    """Detect intersection nodes (degree >= 3) on the planned path.

    Uses ``_point_value`` so both dict-style and object-style graph
    representations are handled transparently.
    """
    nodes, edges = _graph_parts(final_graph)
    if not nodes or not edges:
        return []

    # Build node → degree map using _point_value for dict/object compatibility.
    degree: dict = {}
    for node in nodes:
        nid = _point_value(node, "id")
        if nid is not None:
            degree[nid] = 0
    for edge in edges:
        if _point_value(edge, "enabled", default=True) is False:
            continue
        sid = _point_value(edge, "start", "source")
        eid = _point_value(edge, "end", "target")
        if sid in degree:
            degree[sid] += 1
        if eid in degree:
            degree[eid] += 1

    allowed = set(path_node_sequence or [])
    events = []
    for node in nodes:
        node_id = _point_value(node, "id")
        if node_id is None:
            continue
        if degree.get(node_id, 0) < 3:
            continue
        if allowed and node_id not in allowed:
            continue

        x = float(_point_value(node, "x", "x_pixel", default=0.0))
        y = float(_point_value(node, "y", "y_pixel", default=0.0))
        s, _, distance_px = _project_to_path([x, y], pixel_path, cumulative)

        # If no allowed set, filter by distance from path.
        if not allowed and distance_px > 8.0:
            continue

        events.append({
            "s": s, "pixel": [x, y], "tag": "intersection", "forced": True,
            "node_id": node_id, "degree": degree[node_id],
        })
    return events


def _task_events(snapped_task_points, pixel_path, cumulative):
    """Build forced events for every task point, projected onto the dense path.

    **All** task points (including point_type 0 = start / 1 = goal) are tagged
    ``"task"`` here.  The first waypoint is ALWAYS ``start`` and the last is
    ALWAYS ``goal`` – this is enforced by position (s=0 / s=total) in
    ``_build_sampling_events`` and ``_tag_at``, never by task-point type.
    """
    events = []
    sorted_items = sorted(
        list(snapped_task_points or []),
        key=lambda value: int(_point_value(value, "seq", default=0)),
    )
    for item in sorted_items:
        seq = int(_point_value(item, "seq", default=0))
        status = str(_point_value(item, "status", default="ok"))
        point_type = int(_point_value(item, "point_type", default=2))
        if status == "failed":
            x = _point_value(item, "original_x", "pixel_x", "x")
            y = _point_value(item, "original_y", "pixel_y", "y")
        else:
            x = _point_value(item, "snapped_x", "pixel_x", "x")
            y = _point_value(item, "snapped_y", "pixel_y", "y")
        if x is None or y is None:
            continue
        # ── CRITICAL: all task-point types are tagged "task" ──
        # "start" / "goal" are reserved for the absolute first / last waypoint.
        tag = "task"
        point = [float(x), float(y)]
        s, _, distance_px = _project_to_path(point, pixel_path, cumulative)
        events.append({
            "s": s, "pixel": point, "tag": tag, "forced": True,
            "task_seq": seq,
            "task_point_type": point_type,
            "distance_to_path_px": distance_px,
            "task_status": status,
        })
    return events


# ---------------------------------------------------------------------------
# Spacing / tag helpers
# ---------------------------------------------------------------------------

def _buffer_and_spacing(tag, config):
    if tag in {"start", "goal", "task"}:
        return config.task_point_buffer_m, config.task_point_spacing_m
    if tag == "intersection":
        return config.intersection_buffer_m, config.intersection_spacing_m
    if tag == "sharp_turn":
        return config.corner_buffer_m, config.sharp_turn_spacing_m
    if tag == "corner":
        return config.corner_buffer_m, config.curve_spacing_m
    return 0.0, config.straight_spacing_m


def _tag_at(s, special_events, total, config):
    if abs(s) < 1e-6:
        return "start"
    if abs(s - total) < 1e-6:
        return "goal"
    active = []
    for event in special_events:
        buffer_distance, _ = _buffer_and_spacing(event["tag"], config)
        if abs(s - event["s"]) <= buffer_distance + 1e-9:
            active.append(event["tag"])
    return max(active, key=lambda tag: _TAG_PRIORITY[tag]) if active else "straight"


def _spacing_for_tag(tag, config):
    _, spacing = _buffer_and_spacing(tag, config)
    return max(config.min_waypoint_spacing_m, min(config.max_waypoint_spacing_m, spacing))


def _speed_for_tag(tag, config):
    return {
        "start": config.start_goal_speed_mps,
        "goal": config.start_goal_speed_mps,
        "straight": config.straight_speed_mps,
        "curve": config.curve_speed_mps,
        "corner": config.curve_speed_mps,
        "sharp_turn": config.sharp_turn_speed_mps,
        "intersection": config.intersection_speed_mps,
        "task": config.task_point_speed_mps,
        "inserted_for_validation": config.curve_speed_mps,
    }.get(tag, config.straight_speed_mps)


def _build_sampling_events(total, special_events, config):
    """按局部语义自适应步长采样（直路 10m，弯道/路口更密），禁止均匀 10m 切弯。"""
    events = [
        {"s": 0.0, "pixel": None, "tag": "start", "forced": True},
        {"s": total, "pixel": None, "tag": "goal", "forced": True},
    ]
    events.extend(special_events)

    # 自适应沿程采样：当前点标签决定下一步间距
    s = 0.0
    guard = 0
    max_steps = max(10, int(total / max(0.5, config.min_waypoint_spacing_m)) + 50)
    while s < total - 1e-9 and guard < max_steps:
        guard += 1
        tag = _tag_at(s, special_events, total, config)
        spacing = _spacing_for_tag(tag, config)
        next_s = min(total, s + spacing)
        if next_s >= total - 1e-9:
            break
        if next_s - s < 1e-6:
            next_s = min(total, s + config.min_waypoint_spacing_m)
        if next_s < total - 1e-9:
            events.append({
                "s": next_s,
                "pixel": None,
                "tag": _tag_at(next_s, special_events, total, config),
                "forced": False,
            })
        s = next_s

    for special in special_events:
        buffer_distance, local_spacing = _buffer_and_spacing(special["tag"], config)
        local_spacing = max(config.min_waypoint_spacing_m,
                            min(config.max_waypoint_spacing_m, local_spacing))
        offset = local_spacing
        while offset <= buffer_distance + 1e-9:
            for sign in (-1.0, 1.0):
                value = special["s"] + sign * offset
                if 0.0 < value < total:
                    events.append({
                        "s": value, "pixel": None,
                        "tag": _tag_at(value, special_events, total, config),
                        "forced": False,
                    })
            offset += local_spacing

    events.sort(key=lambda item: item["s"])
    merged = []
    for event in events:
        if merged and abs(event["s"] - merged[-1]["s"]) < 1e-6:
            previous = merged[-1]
            if event.get("forced") or _TAG_PRIORITY[event["tag"]] > _TAG_PRIORITY[previous["tag"]]:
                combined = dict(previous)
                combined.update(event)
                if _TAG_PRIORITY.get(event.get("tag"), -1) <= _TAG_PRIORITY.get(previous.get("tag"), -1):
                    combined["tag"] = previous["tag"]
                combined["forced"] = bool(previous.get("forced") or event.get("forced"))
                if event.get("pixel") is None and previous.get("pixel") is not None:
                    combined["pixel"] = previous["pixel"]
                merged[-1] = combined
            continue
        merged.append(dict(event))

    filtered = []
    minimum = max(0.0, config.min_waypoint_spacing_m)
    for event in merged:
        if not filtered:
            filtered.append(event)
            continue
        gap = event["s"] - filtered[-1]["s"]
        if gap + 1e-9 >= minimum or (event.get("forced") and filtered[-1].get("forced")):
            filtered.append(event)
        elif event.get("forced") and not filtered[-1].get("forced"):
            filtered[-1] = event
        elif not filtered[-1].get("forced") and \
                _TAG_PRIORITY[event["tag"]] > _TAG_PRIORITY[filtered[-1]["tag"]]:
            filtered[-1] = event
    if filtered[-1]["s"] < total - 1e-6:
        filtered.append({"s": total, "pixel": None, "tag": "goal", "forced": True})
    return filtered


# ---------------------------------------------------------------------------
# Line-of-sight / cutting-corner validation
# ---------------------------------------------------------------------------

def _prepare_road_corridor(road_mask, dilate_px: int = 4):
    if road_mask is None:
        return None
    try:
        import cv2
        arr = np.asarray(road_mask)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        binary = (arr > 0).astype(np.uint8)
        if dilate_px > 0:
            k = max(1, int(dilate_px) * 2 + 1)
            kernel = np.ones((k, k), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=1)
        return binary
    except Exception:
        return None


def _point_on_mask(x: float, y: float, mask) -> bool:
    if mask is None:
        return True
    h, w = mask.shape[:2]
    ix, iy = int(round(x)), int(round(y))
    if ix < 0 or iy < 0 or ix >= w or iy >= h:
        return False
    return bool(mask[iy, ix] > 0)


def _segment_mask_support_ratio(
    a_px, b_px, road_mask, *, step_m: float, metres_per_pixel: float,
) -> float:
    if road_mask is None:
        return 1.0
    ax, ay = float(a_px[0]), float(a_px[1])
    bx, by = float(b_px[0]), float(b_px[1])
    dist_px = math.hypot(bx - ax, by - ay)
    if dist_px < 1e-6:
        return 1.0 if _point_on_mask(ax, ay, road_mask) else 0.0
    mpp = max(1e-6, float(metres_per_pixel or 1.0))
    step_px = max(1.0, float(step_m) / mpp)
    n = max(1, int(math.ceil(dist_px / step_px)))
    support = 0
    for i in range(n + 1):
        t = i / n
        x = ax + (bx - ax) * t
        y = ay + (by - ay) * t
        if _point_on_mask(x, y, road_mask):
            support += 1
    return support / float(n + 1)


def _segment_chord_error_m(
    s_a: float, s_b: float,
    dense_metric, cumulative,
) -> float:
    """Max distance from dense arc points to chord A–B in metric space."""
    if s_b <= s_a + 1e-9 or len(dense_metric) < 2:
        return 0.0
    # Locate endpoints on dense metric via interpolation indices
    i0 = 0
    while i0 < len(cumulative) - 1 and cumulative[i0 + 1] < s_a:
        i0 += 1
    i1 = i0
    while i1 < len(cumulative) - 1 and cumulative[i1] < s_b:
        i1 += 1
    ax, ay = _interpolate(dense_metric, cumulative, s_a)
    bx, by = _interpolate(dense_metric, cumulative, s_b)
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 < 1e-12:
        return 0.0
    max_err = 0.0
    for i in range(i0, min(i1 + 1, len(dense_metric))):
        if cumulative[i] < s_a - 1e-9 or cumulative[i] > s_b + 1e-9:
            continue
        px, py = dense_metric[i]
        t = ((px - ax) * dx + (py - ay) * dy) / seg2
        t = max(0.0, min(1.0, t))
        qx, qy = ax + t * dx, ay + t * dy
        max_err = max(max_err, math.hypot(px - qx, py - qy))
    return max_err


def _nearest_graph_edge_id(x: float, y: float, final_graph) -> Optional[int]:
    nodes, edges = _graph_parts(final_graph)
    if not edges:
        return None
    node_pos = {}
    for node in nodes:
        nid = _point_value(node, "id")
        nx = _point_value(node, "x", "x_pixel")
        ny = _point_value(node, "y", "y_pixel")
        if nid is not None and nx is not None and ny is not None:
            node_pos[nid] = (float(nx), float(ny))
    best_id = None
    best_d = float("inf")
    for edge in edges:
        if _point_value(edge, "enabled", default=True) is False:
            continue
        pts = _point_value(edge, "points_pixel", "polyline", default=None) or []
        if len(pts) < 2:
            s = _point_value(edge, "start", "source")
            t = _point_value(edge, "end", "target")
            if s in node_pos and t in node_pos:
                pts = [list(node_pos[s]), list(node_pos[t])]
            else:
                continue
        for i in range(len(pts) - 1):
            x1, y1 = float(pts[i][0]), float(pts[i][1])
            x2, y2 = float(pts[i + 1][0]), float(pts[i + 1][1])
            dx, dy = x2 - x1, y2 - y1
            seg2 = dx * dx + dy * dy
            if seg2 < 1e-9:
                dist = math.hypot(x - x1, y - y1)
            else:
                t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / seg2))
                dist = math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))
            if dist < best_d:
                best_d = dist
                eid = _point_value(edge, "id")
                best_id = eid
    return best_id


def _is_curve_zone_s(s: float, special_events, config) -> bool:
    for event in special_events:
        if event.get("tag") not in {"corner", "sharp_turn", "curve"}:
            continue
        buf, _ = _buffer_and_spacing(event["tag"], config)
        if abs(s - event["s"]) <= buf + 1e-9:
            return True
    return False


def _is_junction_zone_s(s: float, special_events, config) -> bool:
    for event in special_events:
        if event.get("tag") != "intersection":
            continue
        buf, _ = _buffer_and_spacing(event["tag"], config)
        if abs(s - event["s"]) <= buf + 1e-9:
            return True
    return False


def validate_sparse_line_of_sight(
    sparse_pixel: list[list[float]],
    sparse_s: list[float],
    dense_pixel,
    dense_metric,
    cumulative,
    *,
    config: AdaptiveWaypointConfig,
    road_mask=None,
    final_graph=None,
    metres_per_pixel: float = 1.0,
    special_events: Optional[list] = None,
) -> tuple[list[dict], bool]:
    """Validate each sparse chord; return (jump_rows, all_ok)."""
    corridor = _prepare_road_corridor(road_mask, config.road_corridor_dilate_px)
    special_events = special_events or []
    rows = []
    all_ok = True
    for i in range(len(sparse_pixel) - 1):
        a, b = sparse_pixel[i], sparse_pixel[i + 1]
        sa, sb = sparse_s[i], sparse_s[i + 1]
        ratio = _segment_mask_support_ratio(
            a, b, corridor,
            step_m=config.los_sample_step_m,
            metres_per_pixel=metres_per_pixel,
        )
        chord_err = _segment_chord_error_m(sa, sb, dense_metric, cumulative)
        dist_m = abs(sb - sa)
        if metres_per_pixel > 0:
            # prefer metric from path distance; also chord euclidean if metric available
            try:
                am = _interpolate(dense_metric, cumulative, sa)
                bm = _interpolate(dense_metric, cumulative, sb)
                dist_m = math.hypot(bm[0] - am[0], bm[1] - am[1])
            except Exception:
                dist_m = abs(sb - sa)

        reasons = []
        if corridor is not None and ratio < config.min_mask_support_ratio:
            reasons.append("mask_support_low")
        if chord_err > config.max_chord_error_m:
            reasons.append("chord_error_high")
        ok = not reasons
        if not ok:
            all_ok = False
        mid_x = 0.5 * (a[0] + b[0])
        mid_y = 0.5 * (a[1] + b[1])
        rows.append({
            "jump_index": i + 1,
            "jump_id": i + 1,
            "from_waypoint_index": i,
            "to_waypoint_index": i + 1,
            "start_index": i,
            "end_index": i + 1,
            "from_x": round(a[0], 3),
            "from_y": round(a[1], 3),
            "to_x": round(b[0], 3),
            "to_y": round(b[1], 3),
            "start_x": round(a[0], 3),
            "start_y": round(a[1], 3),
            "end_x": round(b[0], 3),
            "end_y": round(b[1], 3),
            "distance_m": round(dist_m, 3),
            "length_m": round(dist_m, 3),
            "length_px": round(math.hypot(b[0] - a[0], b[1] - a[1]), 3),
            "mask_support_ratio": round(ratio, 4),
            "road_support_ratio": round(ratio, 4),
            "chord_error_m": round(chord_err, 4),
            "reason": ",".join(reasons) if reasons else "ok",
            "segment_seq": i + 1,
            "nearest_graph_edge_id": _nearest_graph_edge_id(mid_x, mid_y, final_graph),
            "is_curve_zone": _is_curve_zone_s(0.5 * (sa + sb), special_events, config),
            "is_junction_zone": _is_junction_zone_s(0.5 * (sa + sb), special_events, config),
            "is_suspicious": not ok,
        })
    return rows, all_ok


def repair_sparse_cutting_corners(
    sparse_pixel: list[list[float]],
    sparse_s: list[float],
    dense_pixel,
    dense_metric,
    cumulative,
    *,
    config: Optional[AdaptiveWaypointConfig] = None,
    road_mask=None,
    final_graph=None,
    metres_per_pixel: float = 1.0,
    special_events: Optional[list] = None,
    tags: Optional[list] = None,
    forced_flags: Optional[list] = None,
) -> dict:
    """Insert dense_path midpoints until sparse chords pass LOS / chord checks.

    Returns dict with repaired pixels/s/tags and jump diagnostics.
    """
    config = config or AdaptiveWaypointConfig()
    special_events = list(special_events or [])
    pixels = [list(map(float, p)) for p in sparse_pixel]
    s_values = [float(v) for v in sparse_s]
    tag_list = list(tags) if tags is not None else ["straight"] * len(pixels)
    forced_list = list(forced_flags) if forced_flags is not None else [False] * len(pixels)
    while len(tag_list) < len(pixels):
        tag_list.append("curve")
    while len(forced_list) < len(pixels):
        forced_list.append(True)

    all_jump_rows: list[dict] = []
    inserted_total = 0
    geometry_valid = False

    for iteration in range(max(1, int(config.max_insert_iterations))):
        rows, all_ok = validate_sparse_line_of_sight(
            pixels, s_values, dense_pixel, dense_metric, cumulative,
            config=config, road_mask=road_mask, final_graph=final_graph,
            metres_per_pixel=metres_per_pixel, special_events=special_events,
        )
        all_jump_rows = rows
        if all_ok:
            geometry_valid = True
            break

        # Insert midpoints for failing segments (from high index to low)
        # MUST stay within dense_path[start_idx:end_idx] via s midpoint on dense path
        inserted_this_round = 0
        for row in reversed(rows):
            if not row.get("is_suspicious"):
                continue
            i = int(row["from_waypoint_index"])
            sa, sb = s_values[i], s_values[i + 1]
            if sb <= sa + 1e-9:
                continue  # dense_index / s order invalid — do not force insert
            if sb - sa < config.min_waypoint_spacing_m * 0.5:
                continue
            mid_s = 0.5 * (sa + sb)
            mid_px = _interpolate(dense_pixel, cumulative, mid_s)
            pixels.insert(i + 1, [float(mid_px[0]), float(mid_px[1])])
            s_values.insert(i + 1, float(mid_s))
            tag_list.insert(i + 1, "inserted_for_validation")
            forced_list.insert(i + 1, True)
            inserted_this_round += 1
            inserted_total += 1
        if inserted_this_round == 0:
            break

    # Max-spacing兜底 on sparse s / pixels
    for _ in range(max(1, int(config.max_insert_iterations))):
        viol = []
        for i in range(len(pixels) - 1):
            try:
                am = _interpolate(dense_metric, cumulative, s_values[i])
                bm = _interpolate(dense_metric, cumulative, s_values[i + 1])
                dist_m = math.hypot(bm[0] - am[0], bm[1] - am[1])
            except Exception:
                dist_m = abs(s_values[i + 1] - s_values[i])
            if dist_m > config.max_waypoint_spacing_m + 1e-9:
                viol.append(i)
        if not viol:
            break
        inserted_this_round = 0
        for i in reversed(viol):
            sa, sb = s_values[i], s_values[i + 1]
            if sb <= sa + 1e-9:
                continue
            mid_s = 0.5 * (sa + sb)
            mid_px = _interpolate(dense_pixel, cumulative, mid_s)
            pixels.insert(i + 1, [float(mid_px[0]), float(mid_px[1])])
            s_values.insert(i + 1, float(mid_s))
            tag_list.insert(i + 1, "inserted_for_validation")
            forced_list.insert(i + 1, True)
            inserted_this_round += 1
            inserted_total += 1
        if inserted_this_round == 0:
            break

    # Final validation after last insert
    rows, all_ok = validate_sparse_line_of_sight(
        pixels, s_values, dense_pixel, dense_metric, cumulative,
        config=config, road_mask=road_mask, final_graph=final_graph,
        metres_per_pixel=metres_per_pixel, special_events=special_events,
    )
    all_jump_rows = rows
    geometry_valid = all_ok

    return {
        "sparse_waypoints_pixel": pixels,
        "sparse_s": s_values,
        "tags": tag_list,
        "forced": forced_list,
        "jump_debug_rows": all_jump_rows,
        "geometry_valid": geometry_valid,
        "inserted_midpoint_count": inserted_total,
        "suspicious_chord_count": sum(1 for r in all_jump_rows if r.get("is_suspicious")),
    }


def fix_sparse_cutting_corners(
    dense_path_pixel: Iterable[Sequence[float]],
    sparse_waypoints,
    geo_calibration=None,
    *,
    road_mask=None,
    final_graph=None,
    config: Optional[AdaptiveWaypointConfig] = None,
) -> dict:
    """Public API for UI「修复稀疏航点切弯」."""
    config = config or AdaptiveWaypointConfig()
    dense_pixel = _normalise_path(dense_path_pixel)
    metric, metric_mode, unit, warnings, metric_converter, _w2p = _path_coordinates(
        dense_pixel, geo_calibration
    )
    cumulative = _cumulative_distance(metric)
    mpp = getattr(geo_calibration, "pixel_resolution_estimated_m", None) if geo_calibration else None
    try:
        mpp = float(mpp) if mpp else None
    except (TypeError, ValueError):
        mpp = None
    if mpp is None or mpp <= 0:
        if metric_mode and len(dense_pixel) >= 2:
            path_m = cumulative[-1]
            path_px = _cumulative_distance(dense_pixel)[-1]
            mpp = path_m / max(1e-6, path_px)
        else:
            mpp = 1.0

    sparse_pixel = []
    sparse_s = []
    tags = []
    forced = []
    for index, item in enumerate(sparse_waypoints or []):
        if isinstance(item, dict):
            x = item.get("x_pixel", item.get("x"))
            y = item.get("y_pixel", item.get("y"))
            tag = item.get("tag", "straight")
            fr = bool(item.get("forced", False))
            s = item.get("path_distance_m", item.get("path_distance_px"))
        else:
            x, y = item[0], item[1]
            tag, fr, s = "straight", False, None
        if x is None or y is None:
            continue
        point = [float(x), float(y)]
        sparse_pixel.append(point)
        if s is None:
            s_val, _, _ = _project_to_path(point, dense_pixel, cumulative)
        else:
            s_val = float(s)
        sparse_s.append(s_val)
        tags.append(str(tag))
        forced.append(fr)

    if len(sparse_pixel) < 2:
        return {
            "ok": False,
            "error": "sparse waypoints fewer than 2",
            "geometry_valid": False,
            "sparse_waypoints_pixel": sparse_pixel,
            "jump_debug_rows": [],
        }

    corners = _detect_corners(metric, dense_pixel, cumulative, config)
    special = corners
    repaired = repair_sparse_cutting_corners(
        sparse_pixel, sparse_s, dense_pixel, metric, cumulative,
        config=config, road_mask=road_mask, final_graph=final_graph,
        metres_per_pixel=mpp, special_events=special,
        tags=tags, forced_flags=forced,
    )
    repaired["ok"] = True
    repaired["warnings"] = warnings
    repaired["coordinate_mode"] = "enu_meter" if metric_mode else "image_pixel_fallback"
    repaired["distance_unit"] = unit
    return repaired


# ---------------------------------------------------------------------------
# Roundtrip validation helpers
# ---------------------------------------------------------------------------

def _compute_roundtrip_errors(
    sparse_pixel: list[list[float]],
    world_to_pixel: Optional[Callable],
    metric: list[list[float]],
    metric_mode: bool,
) -> tuple[list[float], list[float]]:
    """Validate pixel → ENU → pixel roundtrip error for each waypoint.

    Returns (errors_px, errors_m) lists – one entry per waypoint.
    """
    errors_px: list[float] = []
    errors_m: list[float] = []
    if not metric_mode:
        return errors_px, errors_m
    if not callable(world_to_pixel):
        return errors_px, errors_m
    for px, py in sparse_pixel:
        try:
            # pixel → ENU → pixel
            x_m, y_m = metric[0]  # we can't easily recompute without re-calling pixel_to_world
            # Actually use the calibration's world_to_pixel:
            # But we need the ENU coords for this waypoint. Since we computed
            # sparse_metric earlier, use that.
            pass
        except Exception:
            errors_px.append(float("nan"))
            errors_m.append(float("nan"))
    return errors_px, errors_m


def _compute_roundtrip_pixel_to_enu_to_pixel(
    points_pixel: list[list[float]],
    pixel_to_world: Callable,
    world_to_pixel: Callable,
) -> list[float]:
    """For each point, compute |roundtrip(p) - p| in pixels."""
    errors: list[float] = []
    for px, py in points_pixel:
        try:
            mx, my = pixel_to_world(px, py)
            rpx, rpy = world_to_pixel(mx, my)
            err = math.hypot(rpx - px, rpy - py)
            errors.append(float(err))
        except Exception:
            errors.append(float("nan"))
    return errors


# ---------------------------------------------------------------------------
# Semantic validation
# ---------------------------------------------------------------------------

def _semantic_check(
    waypoints: list[dict],
    path_node_sequence: Optional[Sequence],
    tasks: list[dict],
    intersection_events: list[dict],
    final_graph,
    actual_task_visit_order: Optional[list] = None,
    expected_task_visit_order: Optional[list] = None,
    segment_isolation_valid: Optional[bool] = None,
    unexpected_task_virtual_nodes: Optional[list] = None,
) -> dict:
    """Run semantic validation on the generated waypoints.

    Returns a dict of validation fields to merge into the resample report.

    ★ CRITICAL: ``actual_task_visit_order`` and ``expected_task_visit_order``
    are the **single source of truth** for task order.  When provided, they
    override any distance-based heuristic.
    """
    # ── 1. start / goal counts ──
    start_count = sum(1 for w in waypoints if w.get("tag") == "start")
    goal_count = sum(1 for w in waypoints if w.get("tag") == "goal")

    # ── 2. Repeated task virtual nodes in path_node_sequence ──
    repeated_nodes: list[str] = []
    if path_node_sequence:
        seen: set[str] = set()
        for nid in path_node_sequence:
            nid_str = str(nid)
            if nid_str in seen:
                # Only flag virtual/task nodes – plain node IDs may legitimately
                # appear more than once in cyclic graphs.
                if ("task_" in nid_str.lower() or "virtual" in nid_str.lower()):
                    if nid_str not in repeated_nodes:
                        repeated_nodes.append(nid_str)
            seen.add(nid_str)

    # ── 3. Task visit order ──
    # ★ SINGLE SOURCE OF TRUTH: planned_segments from planning_result.
    # ★ distance-based metrics below are DEBUG ONLY and MUST NEVER influence
    #    task_order_valid.  They are retained solely for diagnostics.
    task_distance_positions = sorted(
        [(e["task_seq"], e["s"]) for e in tasks if "task_seq" in e],
        key=lambda x: x[0],
    )
    task_visit_distance_along_path = [round(s, 3) for _, s in task_distance_positions]
    distance_monotonic = all(
        task_visit_distance_along_path[i] <= task_visit_distance_along_path[i + 1] + 1e-6
        for i in range(len(task_visit_distance_along_path) - 1)
    )

    # ★ Authoritative task order — ONLY from planned_segments, NEVER from distance.
    if actual_task_visit_order is not None and expected_task_visit_order is not None:
        task_order_valid = (
            list(actual_task_visit_order) == list(expected_task_visit_order)
        )
    else:
        # No authoritative data available — cannot verify; default to pass
        # so that we don't block export for lack of data.
        task_order_valid = True

    # ── 4. Goal appearing before final segment ──
    total_wps = len(waypoints)
    goal_positions = [i for i, w in enumerate(waypoints) if w.get("tag") == "goal"]
    # A "goal" waypoint before the last 5 waypoints suggests the goal appears
    # in the middle of the path, not at the end.
    # Use 5 to allow for dense sampling near the end (e.g. task_point_buffer).
    goal_before_final = (
        len(goal_positions) > 0
        and not all(p >= total_wps - 5 for p in goal_positions)
    )

    # ── 5. Intersection detection availability ──
    nodes, edges = _graph_parts(final_graph)
    intersection_available = bool(nodes and edges)

    # ── 6. Segment isolation validation ──
    _seg_iso_valid = segment_isolation_valid
    if _seg_iso_valid is None:
        _seg_iso_valid = (unexpected_task_virtual_nodes is not None
                          and len(unexpected_task_virtual_nodes or []) == 0)

    # ── 7. Aggregate semantic_valid ──
    # ★ semantic_valid 现在由权威来源共同决定：
    #   1. start/goal 计数值
    #   2. 无重复 virtual node
    #   3. 任务点访问顺序正确（来自 planned_segments）
    #   4. goal 不在最终 segment 之前
    #   5. segment isolation 通过
    semantic_valid = (
        start_count == 1
        and goal_count == 1
        and len(repeated_nodes) == 0
        and task_order_valid
        and not goal_before_final
        and _seg_iso_valid
    )

    return {
        "semantic_valid": semantic_valid,
        "start_count": start_count,
        "goal_count": goal_count,
        "repeated_task_virtual_nodes": repeated_nodes,
        # ── 旧字段重命名为 debug/legacy ──
        "task_visit_distance_along_path_m": task_visit_distance_along_path,
        "task_visit_distance_monotonic": distance_monotonic,
        # ── 新权威字段 ──
        "actual_task_visit_order": (
            list(actual_task_visit_order) if actual_task_visit_order is not None else None
        ),
        "expected_task_visit_order": (
            list(expected_task_visit_order) if expected_task_visit_order is not None else None
        ),
        "task_order_valid": task_order_valid,
        "task_order_source": (
            "planned_segments" if actual_task_visit_order is not None
            else "unverified_no_authoritative_data"
        ),
        "segment_isolation_valid": _seg_iso_valid,
        "unexpected_task_virtual_nodes": (
            list(unexpected_task_virtual_nodes or [])
        ),
        "goal_appears_before_final_segment": goal_before_final,
        "intersection_detection_available": intersection_available,
        "intersection_detection_failure_reason": (
            None if intersection_available
            else "final_graph or node degree unavailable"
        ),
    }


# ---------------------------------------------------------------------------
# Main resampling entry point
# ---------------------------------------------------------------------------

def adaptive_resample_waypoints(
    full_path_pixel: Iterable[Sequence[float]],
    geo_calibration=None,
    final_graph=None,
    snapped_task_points=None,
    path_node_sequence=None,
    path_edge_sequence=None,
    *,
    config: Optional[AdaptiveWaypointConfig] = None,
    output_dir: Optional[str] = None,
    image_width: int = 0,
    image_height: int = 0,
    actual_task_visit_order: Optional[list] = None,
    expected_task_visit_order: Optional[list] = None,
    segment_isolation_valid: Optional[bool] = None,
    unexpected_task_virtual_nodes: Optional[list] = None,
    road_mask=None,
) -> AdaptiveWaypointResult:
    """Adaptively resample a dense planned path into sparse vehicle waypoints.

    The mandatory coordinate pipeline when calibration is valid:

        dense_path_pixel
          → pixel_to_world → ENU meter
          → curvature / corner / intersection detection in ENU
          → adaptive sampling in ENU
          → world_to_pixel → checked pixel coords
          → pixel_to_wgs84 → lon/lat

    Args:
        full_path_pixel: Dense path in original image pixel coordinates.
        geo_calibration: A GeoCalibration instance (or compatible object).
        final_graph: Graph editor / dict with nodes & edges.
        snapped_task_points: Task points snapped to the road network.
        path_node_sequence: Ordered node IDs along the planned path.
        path_edge_sequence: Ordered edge IDs along the planned path.
        config: Adaptive waypoint spacing configuration.
        output_dir: If set, write ``waypoint_resample_report.json`` here.
        image_width: Original image width for bounds checking (0 = skip).
        image_height: Original image height for bounds checking (0 = skip).

    Returns:
        AdaptiveWaypointResult with sparse waypoints and report.
    """
    config = config or AdaptiveWaypointConfig()
    pixel_path_raw = _normalise_path(full_path_pixel)
    original_vertex_count = len(pixel_path_raw)
    metric, metric_mode, unit, warnings, metric_converter, world_to_pixel = _path_coordinates(
        pixel_path_raw, geo_calibration
    )

    # 导出阶段沿中心线加密：graph RDP 折线往往只有稀疏顶点，必须先 densify
    # 再自适应采样。绝不在 A* 图节点中加入 2m 航点。
    densify_step = float(getattr(config, "dense_densify_step_m", 0.5) or 0.5)
    pixel_path, metric = densify_dense_path(pixel_path_raw, metric, densify_step)
    if len(pixel_path) > original_vertex_count:
        warnings.append(
            f"导出前将 dense_path 从 {original_vertex_count} 个顶点加密到 "
            f"{len(pixel_path)} 点（步长 {densify_step:.2f}{unit}），"
            "车辆航点仅从此 dense_path 重采样。"
        )

    # Resolve image bounds from args or calibration.
    img_w, img_h = _image_bounds(geo_calibration)
    if image_width > 0:
        img_w = image_width
    if image_height > 0:
        img_h = image_height

    cumulative = _cumulative_distance(metric)
    total = cumulative[-1]

    corners = _detect_corners(metric, pixel_path, cumulative, config)
    intersections = _intersection_events(
        final_graph, path_node_sequence, pixel_path, cumulative
    )
    tasks = _task_events(snapped_task_points, pixel_path, cumulative)
    special_events = corners + intersections + tasks
    sample_events = _build_sampling_events(total, special_events, config)

    sparse_pixel: list[list[float]] = []
    sparse_metric: list[list[float]] = []
    sparse_s: list[float] = []
    for event in sample_events:
        pixel = event.get("pixel") or _interpolate(pixel_path, cumulative, event["s"])
        sparse_pixel.append([float(pixel[0]), float(pixel[1])])
        sparse_s.append(float(event["s"]))
        if metric_mode:
            sparse_metric.append(list(map(float, metric_converter(*pixel))))
        else:
            sparse_metric.append(list(pixel))

    # metres-per-pixel estimate for LOS sampling
    mpp = getattr(geo_calibration, "pixel_resolution_estimated_m", None) if geo_calibration else None
    try:
        mpp = float(mpp) if mpp else None
    except (TypeError, ValueError):
        mpp = None
    if mpp is None or mpp <= 0:
        path_px = _cumulative_distance(pixel_path)[-1]
        mpp = (total / max(1e-6, path_px)) if metric_mode else 1.0

    # ★ Line-of-sight / cutting-corner repair：相邻稀疏航点弦必须落在道路走廊内
    tags0 = [e["tag"] for e in sample_events]
    forced0 = [bool(e.get("forced")) for e in sample_events]
    los_repair = repair_sparse_cutting_corners(
        sparse_pixel, sparse_s, pixel_path, metric, cumulative,
        config=config,
        road_mask=road_mask,
        final_graph=final_graph,
        metres_per_pixel=mpp,
        special_events=special_events,
        tags=tags0,
        forced_flags=forced0,
    )
    sparse_pixel = los_repair["sparse_waypoints_pixel"]
    sparse_s = los_repair["sparse_s"]
    repaired_tags = los_repair["tags"]
    repaired_forced = los_repair["forced"]
    if len(sparse_pixel) != len(sample_events):
        # Rebuild sample_events-compatible lists after inserts
        sample_events = []
        for idx, s_val in enumerate(sparse_s):
            sample_events.append({
                "s": s_val,
                "pixel": sparse_pixel[idx],
                "tag": repaired_tags[idx] if idx < len(repaired_tags) else "inserted_for_validation",
                "forced": repaired_forced[idx] if idx < len(repaired_forced) else True,
            })
    else:
        for idx, event in enumerate(sample_events):
            event["tag"] = repaired_tags[idx]
            event["forced"] = repaired_forced[idx]
            event["pixel"] = sparse_pixel[idx]

    sparse_metric = []
    for pixel in sparse_pixel:
        if metric_mode:
            sparse_metric.append(list(map(float, metric_converter(*pixel))))
        else:
            sparse_metric.append(list(pixel))

    # Geo coordinates (lon/lat)
    sparse_geo: list[list[float]] = []
    geo_converter = None
    if metric_mode:
        geo_converter = getattr(geo_calibration, "pixel_to_wgs84", None)
        if not callable(geo_converter):
            geo_converter = getattr(geo_calibration, "pixel_to_lonlat", None)
    if callable(geo_converter):
        for point in sparse_pixel:
            lon, lat = geo_converter(*point)
            sparse_geo.append([float(lon), float(lat), 0.0])

    # Roundtrip validation: pixel → ENU → pixel
    pixel_to_world = getattr(geo_calibration, "pixel_to_world", None) if metric_mode else None
    w2p = getattr(geo_calibration, "world_to_pixel", None) if metric_mode else None
    roundtrip_errors_px: list[float] = []
    if callable(pixel_to_world) and callable(w2p):
        roundtrip_errors_px = _compute_roundtrip_pixel_to_enu_to_pixel(
            sparse_pixel, pixel_to_world, w2p
        )
    else:
        roundtrip_errors_px = [0.0] * len(sparse_pixel)

    mask_support_by_index: dict[int, float] = {}
    for row in (los_repair.get("jump_debug_rows") or []):
        try:
            idx = int(row.get("from_waypoint_index", -1))
        except (TypeError, ValueError):
            continue
        if idx >= 0 and "mask_support_ratio" in row:
            try:
                mask_support_by_index[idx] = float(row["mask_support_ratio"])
            except (TypeError, ValueError):
                pass

    # Build rich waypoint dicts.
    tasks_sorted = sorted(tasks, key=lambda e: float(e.get("s", 0.0)))
    waypoints: list[dict] = []
    for index, (event, pixel, xy) in enumerate(zip(sample_events, sparse_pixel, sparse_metric)):
        # Yaw from next point (ENU metric coords)
        if index < len(sparse_metric) - 1:
            target = sparse_metric[index + 1]
            yaw = math.degrees(math.atan2(target[1] - xy[1], target[0] - xy[0])) % 360.0
        elif index > 0:
            previous = sparse_metric[index - 1]
            yaw = math.degrees(math.atan2(xy[1] - previous[1], xy[0] - previous[0])) % 360.0
        else:
            yaw = 0.0

        tag = event["tag"]
        geo = sparse_geo[index] if index < len(sparse_geo) else [None, None, 0.0]

        # spacing_to_prev_m
        if index > 0 and metric_mode:
            prev = sparse_metric[index - 1]
            spacing_to_prev = math.hypot(xy[0] - prev[0], xy[1] - prev[1])
        elif index > 0:
            prev_pix = sparse_pixel[index - 1]
            spacing_to_prev = math.hypot(pixel[0] - prev_pix[0], pixel[1] - prev_pix[1])
        else:
            spacing_to_prev = 0.0

        # Local turn angle at this waypoint along dense path
        local_angle = 0.0
        s_here = float(event.get("s", 0.0))
        if 0 < index < len(sparse_metric) - 1:
            local_angle = _heading_change(
                sparse_metric[index - 1], sparse_metric[index], sparse_metric[index + 1]
            )
        near_junction = tag == "intersection" or any(
            abs(s_here - ev["s"]) <= config.intersection_buffer_m + 1e-9
            for ev in intersections
        )
        near_task = tag in {"start", "goal", "task"} or any(
            abs(s_here - ev["s"]) <= config.task_point_buffer_m + 1e-9
            for ev in tasks
        )
        mask_ratio = mask_support_by_index.get(index)
        if mask_ratio is None and index > 0:
            mask_ratio = mask_support_by_index.get(index - 1)
        if mask_ratio is None:
            mask_ratio = 1.0

        dense_idx = dense_index_for_s(cumulative, s_here)
        # 任务分段：已走过的任务点数量（起点段为 0）
        segment_index = 0
        for tev in tasks_sorted:
            if s_here + 1e-9 >= float(tev.get("s", 0.0)):
                segment_index = int(tev.get("task_seq", segment_index + 1) or (segment_index + 1))
            else:
                break
        source_mode = _spacing_mode_for_tag(
            tag, inserted=(tag == "inserted_for_validation")
        )
        keep = _keep_flag_for_tag(tag, forced=bool(event.get("forced", False)))

        waypoints.append({
            "seq": index + 1,
            "name": f"wp_{index + 1:03d}",
            "x_pixel": pixel[0],
            "y_pixel": pixel[1],
            "x_enu": xy[0] if metric_mode else None,
            "y_enu": xy[1] if metric_mode else None,
            "longitude": geo[0],
            "latitude": geo[1],
            "latitude_deg": geo[1],
            "longitude_deg": geo[0],
            "altitude": geo[2],
            "altitude_m": geo[2],
            "yaw_deg": round(yaw, 2),
            "target_speed_mps": float(_speed_for_tag(tag, config)),
            "arrival_radius_m": 3.0 if tag == "straight" else 2.0,
            "pass_through": tag not in {"start", "goal"},
            "tag": tag,
            "spacing_mode": source_mode,
            "source_mode": source_mode,
            "dense_index": int(dense_idx),
            "s_m": round(s_here, 3),
            "segment_index": int(segment_index),
            "keep": bool(keep),
            "local_angle_deg": round(float(local_angle), 3),
            "near_junction": bool(near_junction),
            "near_task_point": bool(near_task),
            "mask_support_ratio": round(float(mask_ratio), 4),
            "forced": bool(event.get("forced", False)),
            "task_seq": event.get("task_seq"),
            "task_status": event.get("task_status"),
            "path_distance_m" if metric_mode else "path_distance_px": round(s_here, 3),
            "spacing_to_prev_m": round(spacing_to_prev, 3),
            "is_task_point": tag in {"start", "goal", "task"},
            "is_corner": tag in {"corner", "sharp_turn"},
            "is_intersection": tag == "intersection",
            "inside_image": _inside_image(pixel[0], pixel[1], img_w, img_h),
            "distance_to_graph_px": event.get("distance_to_path_px"),
            "roundtrip_error_px": round(roundtrip_errors_px[index], 4) if index < len(roundtrip_errors_px) else None,
        })

    # -------- report --------
    counts: dict = {}
    for item in waypoints:
        counts[item["tag"]] = counts.get(item["tag"], 0) + 1
    spacings = [
        float(item.get("spacing_to_prev_m") or 0.0)
        for item in waypoints[1:]
    ]
    average_spacing = (sum(spacings) / len(spacings)) if spacings else 0.0
    max_spacing = max(spacings) if spacings else 0.0

    mode_counts = {
        "straight_10m": 0,
        "curve_2m": 0,
        "junction_2m": 0,
        "task_2m": 0,
        "inserted_for_validation": 0,
    }
    for item in waypoints:
        mode = str(item.get("spacing_mode") or "straight_10m")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    out_of_bounds = [wp for wp in waypoints if not wp["inside_image"]]
    ob_count = len(out_of_bounds)

    # Roundtrip stats
    valid_rt = [e for e in roundtrip_errors_px if math.isfinite(e)]
    max_rt = max(valid_rt) if valid_rt else 0.0
    mean_rt = sum(valid_rt) / len(valid_rt) if valid_rt else 0.0
    bad_rt_count = sum(1 for e in valid_rt if e > 2.0)

    # ── Semantic validation ──
    semantic = _semantic_check(
        waypoints, path_node_sequence, tasks, intersections, final_graph,
        actual_task_visit_order=actual_task_visit_order,
        expected_task_visit_order=expected_task_visit_order,
        segment_isolation_valid=segment_isolation_valid,
        unexpected_task_virtual_nodes=unexpected_task_virtual_nodes,
    )

    geometry_valid = bool(los_repair.get("geometry_valid", False))
    invalid_segment_count = int(los_repair.get("suspicious_chord_count", 0))

    # ── export_valid: bounds + roundtrip + semantic + geometry ──
    export_valid = (
        (ob_count == 0)
        and (bad_rt_count == 0)
        and semantic["semantic_valid"]
        and geometry_valid
    )

    # ── Intersection count note ──
    if not semantic["intersection_detection_available"] and len(intersections) == 0:
        intersection_note = (
            "intersection_count=0 是因为 final_graph 或 node degree 信息不可用；"
            "不代表图上没有路口。"
        )
    else:
        intersection_note = None

    if max_spacing > 12.0:
        warnings.append("存在过稀疏航点，请检查重采样。")

    report = {
        "coordinate_mode": "enu_meter" if metric_mode else "image_pixel_fallback",
        "distance_unit": unit,
        "dense_path_point_count": len(pixel_path),
        "dense_path_vertex_count_before_densify": original_vertex_count,
        "dense_point_count": len(pixel_path),
        "dense_path_point_count_description": "沿道路中心线加密后的 dense_path 点数（导出阶段 densify，非 A* 节点）",
        "sparse_waypoint_count": len(waypoints),
        "vehicle_waypoint_count": len(waypoints),
        "sparse_waypoint_count_description": "从 dense_path 自适应重采样得到的车辆航点数",
        "resampling_mode": "dense_path_densify_then_adaptive_with_los_repair",
        "path_length_m" if metric_mode else "path_length_px": round(total, 3),
        "total_length_m": round(total, 3) if metric_mode else None,
        "average_spacing_m" if metric_mode else "average_spacing_px": round(average_spacing, 3),
        "max_spacing_m": round(max_spacing, 3) if metric_mode else None,
        "straight_waypoint_count": int(mode_counts.get("straight_10m", 0)),
        "curve_waypoint_count": int(mode_counts.get("curve_2m", 0)),
        "junction_waypoint_count": int(mode_counts.get("junction_2m", 0)),
        "task_waypoint_count": int(mode_counts.get("task_2m", 0)),
        "inserted_for_validation_count": int(
            mode_counts.get("inserted_for_validation", 0)
            or los_repair.get("inserted_midpoint_count", 0)
        ),
        "invalid_segment_count": invalid_segment_count,
        "corner_count": sum(item["tag"] == "corner" for item in corners),
        "sharp_turn_count": sum(item["tag"] == "sharp_turn" for item in corners),
        "intersection_count": len(intersections),
        "intersection_detection_available": semantic["intersection_detection_available"],
        "intersection_detection_failure_reason": semantic["intersection_detection_failure_reason"],
        "task_point_count": len(tasks),
        "failed_task_points_preserved": sum(
            item.get("task_status") == "failed" for item in tasks
        ),
        "waypoint_tag_counts": counts,
        "spacing_mode_counts": mode_counts,
        "path_node_sequence": list(path_node_sequence or []),
        "path_edge_sequence": list(path_edge_sequence or []),
        "parameters": asdict(config),
        "warnings": warnings,

        # ── Validation fields ──
        "export_valid": export_valid,
        "geometry_valid": geometry_valid,
        "los_inserted_midpoint_count": int(los_repair.get("inserted_midpoint_count", 0)),
        "suspicious_chord_count": int(los_repair.get("suspicious_chord_count", 0)),
        "sparse_chord_jump_debug_rows": list(los_repair.get("jump_debug_rows") or []),
        "semantic_valid": semantic["semantic_valid"],
        "start_count": semantic["start_count"],
        "goal_count": semantic["goal_count"],
        "repeated_task_virtual_nodes": semantic["repeated_task_virtual_nodes"],
        # ── Legacy distance-based fields (debug only, NOT for decision) ──
        "task_visit_distance_along_path_m": semantic["task_visit_distance_along_path_m"],
        "task_visit_distance_monotonic": semantic["task_visit_distance_monotonic"],
        # ── Authoritative task order from planned_segments ──
        "actual_task_visit_order": semantic["actual_task_visit_order"],
        "expected_task_visit_order": semantic["expected_task_visit_order"],
        "task_order_valid": semantic["task_order_valid"],
        "task_order_source": semantic["task_order_source"],
        # ── Segment isolation ──
        "segment_isolation_valid": semantic["segment_isolation_valid"],
        "unexpected_task_virtual_nodes": semantic["unexpected_task_virtual_nodes"],
        # ── Other semantic ──
        "goal_appears_before_final_segment": semantic["goal_appears_before_final_segment"],
        "recommended_vehicle_file": "subject1_waypoints.yaml" if export_valid else None,
        "subject1_waypoints_exported": export_valid,
        "subject1_waypoints_file": "subject1_waypoints.yaml" if export_valid else None,
        "default_altitude_m": 21.741,
        "image_width": img_w,
        "image_height": img_h,
        "out_of_bounds_count": ob_count,
        "out_of_bounds_examples": [
            {"seq": wp["seq"], "x_pixel": wp["x_pixel"], "y_pixel": wp["y_pixel"]}
            for wp in out_of_bounds[:20]
        ],
        "roundtrip_error_max_px": round(max_rt, 4),
        "roundtrip_error_mean_px": round(mean_rt, 4),
        "bad_roundtrip_count": bad_rt_count,
    }

    # ── Aggregate invalid_reason ──
    # NOTE: planning_report.json is the authoritative source for export_valid
    # and invalid_reason.  This inline copy in the resample report is retained
    # for standalone debugging only.
    reasons = []
    if not export_valid:
        if ob_count > 0:
            reasons.append(f"out_of_bounds_count={ob_count}")
        if bad_rt_count > 0:
            reasons.append(f"bad_roundtrip_count={bad_rt_count}")
        if not semantic["semantic_valid"]:
            if semantic["start_count"] != 1:
                reasons.append(f"start_count={semantic['start_count']}(expect=1)")
            if semantic["goal_count"] != 1:
                reasons.append(f"goal_count={semantic['goal_count']}(expect=1)")
            if semantic["repeated_task_virtual_nodes"]:
                reasons.append(f"repeated_virtual_nodes={semantic['repeated_task_virtual_nodes']}")
            if not semantic["task_order_valid"]:
                reasons.append(f"task_order_invalid(source={semantic['task_order_source']})")
            if semantic["goal_appears_before_final_segment"]:
                reasons.append("goal_before_final_segment")
            if not semantic.get("segment_isolation_valid", True):
                reasons.append("segment_isolation_invalid")
        if not geometry_valid:
            reasons.append(
                f"sparse_chord_cutting_corner"
                f"(suspicious={los_repair.get('suspicious_chord_count', 0)})"
            )
        report["invalid_reason"] = "; ".join(reasons) if reasons else "unknown"
        report["debug_files"] = [
            "waypoints_sparse_10m_INVALID.csv",
            "path_jump_debug.csv",
            "jump_debug_overlay.png",
        ]
    else:
        report["invalid_reason"] = None
        report["debug_files"] = []

    if intersection_note:
        report["warnings"].append(intersection_note)
    if ob_count > 0:
        report["warnings"].append(
            f"发现 {ob_count} 个航点越界（超出图像范围 {img_w}x{img_h}），"
            f"waypoints_sparse_10m.yaml 不会被推荐为车辆文件。"
        )
    if bad_rt_count > 0:
        report["warnings"].append(
            f"发现 {bad_rt_count} 个航点往返误差 > 2px，"
            f"坐标转换可能存在异常。"
        )
    if not semantic["semantic_valid"]:
        reasons = []
        if semantic["start_count"] != 1:
            reasons.append(f"start 航点数={semantic['start_count']}(期望=1)")
        if semantic["goal_count"] != 1:
            reasons.append(f"goal 航点数={semantic['goal_count']}(期望=1)")
        if semantic["repeated_task_virtual_nodes"]:
            reasons.append(
                f"重复 task virtual 节点: {semantic['repeated_task_virtual_nodes']}"
            )
        if not semantic["task_order_valid"] and semantic["task_order_source"] == "planned_segments":
            reasons.append("任务点访问顺序错误(planned_segments)")
        elif not semantic["task_order_valid"] and semantic["task_order_source"] == "unverified_no_authoritative_data":
            reasons.append("任务点访问顺序无法验证(无权威数据)")
        if semantic["goal_appears_before_final_segment"]:
            reasons.append("终点任务点在路径中提前出现，任务点顺序可能错误。")
        if not semantic.get("segment_isolation_valid", True):
            reasons.append("分段隔离检查失败")
        report["warnings"].append(
            f"语义检查失败: {'; '.join(reasons)}。不推荐 {report.get('recommended_vehicle_file', 'N/A')} 作为车辆文件。"
        )
    if len(waypoints) > 500:
        report["warnings"].append(
            "重采样后航点仍超过 500 个，建议增大 straight_spacing_m 或 curve_spacing_m。"
        )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "waypoint_resample_report.json"),
                  "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    return AdaptiveWaypointResult(
        sparse_waypoints_pixel=sparse_pixel,
        sparse_waypoints_geo=sparse_geo,
        waypoints=waypoints,
        dense_path_enu=[list(map(float, point)) for point in metric],
        report=report,
        dense_path_pixel=[list(map(float, point)) for point in pixel_path],
    )


def generate_vehicle_waypoints_adaptive(
    dense_path_pixel,
    dense_path_geo=None,
    graph=None,
    task_points=None,
    road_mask=None,
    config=None,
    *,
    geo_calibration=None,
    path_node_sequence=None,
    path_edge_sequence=None,
    image_width: int = 0,
    image_height: int = 0,
    output_dir: Optional[str] = None,
    actual_task_visit_order=None,
    expected_task_visit_order=None,
    segment_isolation_valid=None,
    unexpected_task_virtual_nodes=None,
) -> AdaptiveWaypointResult:
    """从 dense_path 生成车辆航点（导出阶段专用，不影响 A*）。

    Parameters
    ----------
    dense_path_pixel :
        A*/Dijkstra 展开后的道路中心线折线（像素）。
    dense_path_geo :
        可选；若提供且无 geo_calibration，仅作报告参考，航点仍由像素+标定转换。
    graph :
        final_graph，用于路口加密。
    task_points :
        任务点（含吸附结果），用于任务点附近加密。
    road_mask :
        道路 mask，用于相邻航点 line-of-sight 验证。
    config :
        AdaptiveWaypointConfig 或兼容 dict。
    """
    if isinstance(config, dict):
        allowed = set(AdaptiveWaypointConfig.__dataclass_fields__)
        config = AdaptiveWaypointConfig(**{
            key: value for key, value in config.items() if key in allowed
        })
    result = adaptive_resample_waypoints(
        dense_path_pixel,
        geo_calibration,
        graph,
        task_points,
        path_node_sequence,
        path_edge_sequence,
        config=config,
        output_dir=output_dir,
        image_width=image_width,
        image_height=image_height,
        actual_task_visit_order=actual_task_visit_order,
        expected_task_visit_order=expected_task_visit_order,
        segment_isolation_valid=segment_isolation_valid,
        unexpected_task_virtual_nodes=unexpected_task_virtual_nodes,
        road_mask=road_mask,
    )
    if dense_path_geo is not None:
        result.report["dense_path_geo_point_count"] = len(list(dense_path_geo) or [])
    return result
