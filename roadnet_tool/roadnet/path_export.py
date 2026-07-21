"""Planned-path export helpers.

This module deliberately has no Qt dependency.  Planning data is converted to
the vehicle-facing export files here; the GUI is only responsible for choosing
the destination and presenting errors.

Export file inventory
---------------------
Vehicle files (ONLY generated when all checks pass):
  waypoints_sparse_10m.yaml    ← yaml (legacy name)
  waypoints_sparse_10m.csv     ← human-inspection table
  subject1_waypoints.yaml      ← recommended_vehicle_file

Debug / report files (always generated):
  global_path_dense_pixel.json   ← dense path, image pixel coords
  global_path_dense_geo.json     ← dense path, lon/lat coords
  waypoint_preview.png           ← visual preview
  waypoint_resample_report.json  ← resampling validation report
  planning_report.json           ← planning & bounds-check summary

Invalid output (generated when bounds/sanity checks fail):
  waypoints_sparse_10m_INVALID.csv  ← debug file, NOT for vehicle use
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from datetime import datetime
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .adaptive_waypoint_resampler import (
    AdaptiveWaypointConfig,
    AdaptiveWaypointResult,
    _point_value,
    adaptive_resample_waypoints,
)
from .waypoint_validator import (
    WaypointValidationConfig,
    bad_segments_csv_text,
    validate_vehicle_waypoints,
    vehicle_waypoints_invalid_csv_text,
)

# ---------------------------------------------------------------------------
# Filename constants
# ---------------------------------------------------------------------------

# Files that are always generated.
_EXPORT_DEBUG_FILES = (
    "global_path_dense_pixel.json",
    "global_path_dense_geo.json",
    "global_path_dense_geo.csv",
    "vehicle_waypoints_adaptive.csv",
    "waypoint_preview.png",
    "debug_preview.png",
    "waypoint_resample_report.json",
    "waypoint_validation_report.json",
    "bad_segments.csv",
    "waypoint_validation_overlay.png",
    "planning_report.json",
    "planned_segments_debug.csv",
    "dense_path_debug.csv",
    "dense_path_validation_report.json",
    "dense_path_bad_segments.csv",
    "dense_path_validation_overlay.png",
    "virtual_node_split_debug.csv",
)

# Vehicle-facing files – only when export_valid == True.
_EXPORT_VEHICLE_FILES = (
    "waypoints_sparse_10m.yaml",
    "waypoints_sparse_10m.csv",
    "subject1_waypoints.yaml",
)

# ── subject1_waypoints.yaml configuration ──
DEFAULT_ALTITUDE_M = 21.741
WAYPOINT_NAME_DIGITS = 3  # wp_001, wp_002, ...
MIN_EXPORT_SPACING_M = 0.5  # discard consecutive waypoints closer than this

# Invalid debug output when bounds check fails.
_EXPORT_INVALID_FILE = "waypoints_sparse_10m_INVALID.csv"
_EXPORT_VEHICLE_INVALID_CSV = "vehicle_waypoints_adaptive_INVALID.csv"

# Full legacy compatibility list.
EXPORT_FILENAMES = _EXPORT_DEBUG_FILES + _EXPORT_VEHICLE_FILES + (
    # Compatibility aliases retained for existing vehicle integrations.
    "global_path_pixel.json",
    "global_path_geo.json",
    "global_path.csv",
    "global_path_geo.csv",
    "waypoints.yaml",
)

RECOMMENDED_VEHICLE_FILE = "subject1_waypoints.yaml"

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------

MAX_SEGMENT_LENGTH_M = 30.0   # suspicious jump threshold (meters)
MAX_SEGMENT_LENGTH_PX = 200   # suspicious jump threshold (pixels)
MAX_WAYPOINT_NUMBER_LABELS = 50  # max seq labels in preview


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PathExportError(RuntimeError):
    """Raised when a planned path cannot be exported safely."""


class InvalidGeoCalibrationError(PathExportError):
    """Raised when the geo calibration cannot convert path points."""


class PathOutOfBoundsError(PathExportError):
    """Raised when path or waypoints contain out-of-bounds coordinates."""


class SuspiciousJumpError(PathExportError):
    """Raised when path contains abnormally long segments."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def default_path_export_dir(base_dir: str, now: Optional[datetime] = None) -> str:
    """Return ``outputs/path_planning/run_YYYYMMDD_HHMMSS`` below *base_dir*."""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return os.path.join(os.path.abspath(base_dir), "outputs", "path_planning", f"run_{stamp}")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalise_pixel_path(points: Iterable[Sequence[float]]) -> List[List[float]]:
    result: List[List[float]] = []
    for index, point in enumerate(points or []):
        if point is None or len(point) < 2:
            raise PathExportError(f"planned_path 第 {index + 1} 个点格式无效")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise PathExportError(f"planned_path 第 {index + 1} 个点不是有效数值") from exc
        if not (math.isfinite(x) and math.isfinite(y)):
            raise PathExportError(f"planned_path 第 {index + 1} 个点包含非有限坐标")
        if result and x == result[-1][0] and y == result[-1][1]:
            continue
        result.append([x, y])
    if len(result) < 2:
        raise PathExportError("planned_path 为空或有效点数少于 2")
    return result


# ---------------------------------------------------------------------------
# Geo calibration helpers
# ---------------------------------------------------------------------------

def _calibration_converter(calibration) -> Callable[[float, float], Tuple[float, float]]:
    if calibration is None:
        raise InvalidGeoCalibrationError("geo_calibration 无效：尚未完成坐标标定")

    valid = getattr(calibration, "is_valid", False)
    if callable(valid):
        valid = valid()
    if not bool(valid):
        raise InvalidGeoCalibrationError("geo_calibration 无效：请先完成坐标标定")

    converter = getattr(calibration, "pixel_to_wgs84", None)
    if converter is None:
        converter = getattr(calibration, "pixel_to_lonlat", None)
    if not callable(converter):
        raise InvalidGeoCalibrationError(
            "geo_calibration 无效：缺少 pixel_to_wgs84/pixel_to_lonlat 转换接口"
        )
    return converter


def convert_pixel_path_to_geo(
    points: Iterable[Sequence[float]], calibration
) -> List[List[float]]:
    """Convert pixel points to ``[longitude, latitude, altitude]``."""
    pixel_points = _normalise_pixel_path(points)
    converter = _calibration_converter(calibration)
    result: List[List[float]] = []
    for index, (x, y) in enumerate(pixel_points):
        try:
            lon, lat = converter(x, y)
            lon, lat = float(lon), float(lat)
        except Exception as exc:
            raise InvalidGeoCalibrationError(
                f"pixel_to_wgs84 转换失败（路径点 {index + 1}, pixel=({x:.3f}, {y:.3f})）：{exc}"
            ) from exc
        if not (math.isfinite(lon) and math.isfinite(lat)):
            raise InvalidGeoCalibrationError(f"路径点 {index + 1} 转换得到非有限经纬度")
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            raise InvalidGeoCalibrationError(
                f"路径点 {index + 1} 转换后的经纬度越界：({lon}, {lat})"
            )
        result.append([lon, lat, 0.0])
    return result


# ---------------------------------------------------------------------------
# Bounds checking
# ---------------------------------------------------------------------------

def _check_dense_path_bounds(
    pixel_points: List[List[float]], image_width: int, image_height: int
) -> dict:
    """Check whether any dense path point falls outside the image.

    Returns a dict with: point_count, out_of_bounds_count, out_of_bounds_examples.
    """
    if image_width <= 0 or image_height <= 0:
        return {"dense_path_point_count": len(pixel_points), "dense_path_out_of_bounds_count": 0,
                "dense_path_out_of_bounds_examples": []}
    oob = []
    for idx, (x, y) in enumerate(pixel_points):
        if not (0.0 <= x < float(image_width) and 0.0 <= y < float(image_height)):
            oob.append({"index": idx, "seq": idx + 1, "x_pixel": round(x, 3), "y_pixel": round(y, 3)})
    return {
        "dense_path_point_count": len(pixel_points),
        "dense_path_out_of_bounds_count": len(oob),
        "dense_path_out_of_bounds_examples": oob[:20],
    }


def _check_waypoint_bounds(waypoints: List[dict], image_width: int, image_height: int) -> dict:
    """Check waypoints for out-of-bounds and collect stats."""
    if image_width <= 0 or image_height <= 0:
        return {"out_of_bounds_count": 0, "out_of_bounds_list": [], "all_inside": True}
    oob = []
    for wp in waypoints:
        if not wp.get("inside_image", True):
            oob.append({
                "seq": wp["seq"],
                "x_pixel": round(wp["x_pixel"], 3),
                "y_pixel": round(wp["y_pixel"], 3),
                "tag": wp.get("tag", ""),
            })
    return {
        "out_of_bounds_count": len(oob),
        "out_of_bounds_list": oob,
        "all_inside": len(oob) == 0,
    }


# ---------------------------------------------------------------------------
# Suspicious jump detection (two-level)
# ---------------------------------------------------------------------------

MAX_SEGMENT_LENGTH_M = 30.0     # Level-1 geometric threshold (meters)
MAX_SEGMENT_LENGTH_PX = 200     # Level-1 geometric threshold (pixels)
ROAD_SUPPORT_RATIO_MIN = 0.7    # Level-2 road validation threshold
ROAD_SAMPLE_SPACING_M = 2.0     # Sampling step along long segment for road check
ROAD_BUFFER_RADIUS_PX = 10.0    # Buffer radius around graph/skeleton in pixels


def _validate_long_segment_road_support(
    start_px: Tuple[float, float],
    end_px: Tuple[float, float],
    geo_converter: Optional[Callable],
    final_graph=None,
    road_mask: Optional[np.ndarray] = None,
    sample_spacing_m: float = ROAD_SAMPLE_SPACING_M,
    buffer_radius_px: float = ROAD_BUFFER_RADIUS_PX,
) -> Tuple[float, str]:
    """Level-2 road legality check on a long segment candidate.

    Samples the segment every *sample_spacing_m* meters and checks whether
    sample points fall within the road region (final_graph buffer or
    road_mask).

    Args:
        start_px, end_px: segment endpoints in original image pixel.
        geo_converter: pixel→(lon, lat) callable for metric spacing.
        final_graph: graph with nodes/edges (for buffer-based check).
        road_mask: binary road mask image (alternative check).
        sample_spacing_m: metric step between samples.
        buffer_radius_px: pixel radius for graph/mask buffer.

    Returns:
        (road_support_ratio, reason)
        road_support_ratio in [0, 1]; reason is a human-readable string.
    """
    import numpy as np

    px_dist = math.hypot(end_px[0] - start_px[0], end_px[1] - start_px[1])
    if px_dist < 1.0:
        return 1.0, "segment_too_short_for_validation"

    # Estimate metres-per-pixel from the segment endpoints if possible
    mpp = None
    if geo_converter is not None:
        try:
            lon_s, lat_s = geo_converter(start_px[0], start_px[1])
            lon_e, lat_e = geo_converter(end_px[0], end_px[1])
            m_dist = _haversine_m([lon_s, lat_s, 0], [lon_e, lat_e, 0])
            if m_dist > 0.01:
                mpp = m_dist / px_dist
        except Exception:
            pass
    if mpp is None or mpp <= 0:
        mpp = 0.5  # fallback

    # Compute step in pixel space
    step_px = sample_spacing_m / mpp
    if step_px <= 0:
        step_px = 1.0

    num_samples = max(1, int(px_dist / step_px))
    if num_samples > 100:
        num_samples = 100  # cap for performance

    # Generate sample points along the line
    dx = end_px[0] - start_px[0]
    dy = end_px[1] - start_px[1]
    samples = []
    for i in range(num_samples + 1):
        t = i / max(1, num_samples)
        samples.append((start_px[0] + t * dx, start_px[1] + t * dy))

    # Build road buffer from final_graph (simplified: rasterize graph edges)
    if final_graph is not None:
        try:
            nodes = getattr(final_graph, "nodes", None) or []
            edges = getattr(final_graph, "edges", None) or []
            if nodes and edges:
                # Build a quick raster buffer using graph edge polyline
                support_count = 0
                for spx, spy in samples:
                    on_road = _is_near_graph_edge(
                        spx, spy, nodes, edges, buffer_radius_px
                    )
                    if on_road:
                        support_count += 1
                ratio = support_count / max(1, len(samples))
                reason = "graph_buffer_check"
                return ratio, reason
        except Exception:
            pass

    # Fallback: road_mask check
    if road_mask is not None and road_mask.size > 0:
        try:
            mask_arr = np.asarray(road_mask)
            if mask_arr.ndim >= 2:
                h, w = mask_arr.shape[:2]
                support_count = 0
                for spx, spy in samples:
                    ix, iy = int(round(spx)), int(round(spy))
                    if 0 <= ix < w and 0 <= iy < h:
                        if mask_arr.ndim == 2:
                            val = mask_arr[iy, ix]
                        else:
                            val = mask_arr[iy, ix, 0]
                        if val > 0:
                            support_count += 1
                ratio = support_count / max(1, len(samples))
                reason = "road_mask_check"
                return ratio, reason
        except Exception:
            pass

    # Cannot validate – treat as suspicious
    return 0.0, "no_road_data_available_for_validation"


def _is_near_graph_edge(
    px: float, py: float,
    nodes: list, edges: list,
    buffer_radius_px: float = 10.0,
) -> bool:
    """Check if point (px,py) is within *buffer_radius_px* of any graph edge."""
    import math as _m
    # Build node dict
    node_pos = {}
    for n in nodes:
        nid = n.get("id")
        x = n.get("x", n.get("x_pixel", None))
        y = n.get("y", n.get("y_pixel", None))
        if nid is not None and x is not None and y is not None:
            node_pos[nid] = (float(x), float(y))

    for e in edges:
        pts = e.get("points_pixel", [])
        if len(pts) < 2:
            s, t = e.get("start"), e.get("end")
            if s in node_pos and t in node_pos:
                pts = [list(node_pos[s]), list(node_pos[t])]
            else:
                continue
        for i in range(len(pts) - 1):
            x1, y1 = pts[i][0], pts[i][1]
            x2, y2 = pts[i+1][0], pts[i+1][1]
            # Point-to-segment distance
            dx = x2 - x1
            dy = y2 - y1
            seg_len2 = dx * dx + dy * dy
            if seg_len2 < 1e-9:
                dist = _m.hypot(px - x1, py - y1)
            else:
                t = ((px - x1) * dx + (py - y1) * dy) / seg_len2
                t = max(0.0, min(1.0, t))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist = _m.hypot(px - proj_x, py - proj_y)
            if dist <= buffer_radius_px:
                return True
    return False


def _detect_suspicious_jumps(
    pixel_points: List[List[float]],
    geo_converter: Optional[Callable],
    max_px: float = MAX_SEGMENT_LENGTH_PX,
    max_m: float = MAX_SEGMENT_LENGTH_M,
    final_graph=None,
    road_mask: Optional[np.ndarray] = None,
) -> dict:
    """Two-level detection of abnormally long segments in the dense path.

    Level 1 – Geometry: adjacent dense-point distance exceeds *max_px* or
                         *max_m* → candidate.

    Level 2 – Road legality: sample the candidate segment and check whether
                             samples lie within road region (graph buffer or
                             mask).  High road support → legitimate long
                             straight road; low support → truly suspicious.

    Returns dict with:
        suspicious_jump_count, long_segment_candidate_count, jumps list,
        has_jumps, jump_debug_rows.
    """
    candidates = []
    jump_debug_rows: list[dict] = []

    for i in range(len(pixel_points) - 1):
        ax, ay = pixel_points[i]
        bx, by = pixel_points[i + 1]
        dist_px = math.hypot(bx - ax, by - ay)
        is_candidate = dist_px > max_px
        dist_m = None

        if geo_converter is not None and callable(geo_converter):
            try:
                lon_a, lat_a = geo_converter(ax, ay)
                lon_b, lat_b = geo_converter(bx, by)
                dist_m = _haversine_m([lon_a, lat_a, 0], [lon_b, lat_b, 0])
                if dist_m is not None and dist_m > max_m:
                    is_candidate = True
            except Exception:
                pass

        if is_candidate:
            candidates.append({
                "jump_id": len(candidates) + 1,
                "start_index": i,
                "end_index": i + 1,
                "start_x": round(ax, 3),
                "start_y": round(ay, 3),
                "end_x": round(bx, 3),
                "end_y": round(by, 3),
                "length_m": round(dist_m, 3) if dist_m is not None else None,
                "length_px": round(dist_px, 3),
            })

    # Level 2: validate each candidate against road region
    suspicious = []
    for cand in candidates:
        ratio, reason = _validate_long_segment_road_support(
            (cand["start_x"], cand["start_y"]),
            (cand["end_x"], cand["end_y"]),
            geo_converter,
            final_graph=final_graph,
            road_mask=road_mask,
        )
        road_support_ratio = round(ratio, 4)
        is_suspicious = road_support_ratio < ROAD_SUPPORT_RATIO_MIN
        cand["road_support_ratio"] = road_support_ratio
        cand["is_suspicious"] = is_suspicious
        cand["reason"] = reason
        if is_suspicious:
            suspicious.append(cand)
        jump_debug_rows.append(cand)

    return {
        "long_segment_candidate_count": len(candidates),
        "suspicious_jump_count": len(suspicious),
        "suspicious_jump_examples": [
            {
                "jump_id": j["jump_id"],
                "start_index": j["start_index"],
                "end_index": j["end_index"],
                "length_m": j["length_m"],
                "road_support_ratio": j["road_support_ratio"],
                "start_pixel": [j["start_x"], j["start_y"]],
                "end_pixel": [j["end_x"], j["end_y"]],
            }
            for j in suspicious[:20]
        ],
        "jumps": suspicious,
        "all_candidates": candidates,
        "has_jumps": len(suspicious) > 0,
        "jump_debug_rows": jump_debug_rows,
    }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _haversine_m(a: Sequence[float], b: Sequence[float]) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * 6378137.0 * math.asin(min(1.0, math.sqrt(h)))


def _polyline_length(points: Sequence[Sequence[float]], distance) -> float:
    return sum(distance(points[i], points[i + 1]) for i in range(len(points) - 1))


def _resample_by_segment_lengths(
    pixel_points: Sequence[Sequence[float]],
    segment_lengths: Sequence[float],
    spacing: float,
) -> List[List[float]]:
    if not math.isfinite(spacing) or spacing <= 0:
        raise PathExportError("重采样间距必须大于 0")

    total = sum(max(0.0, length) for length in segment_lengths)
    if total <= 0:
        raise PathExportError("planned_path 总长度为 0")

    targets: List[float] = []
    value = spacing
    while value < total:
        targets.append(value)
        value += spacing

    output = [[float(pixel_points[0][0]), float(pixel_points[0][1])]]
    target_index = 0
    traversed = 0.0
    for index, seg_len in enumerate(segment_lengths):
        if seg_len <= 0:
            continue
        start = pixel_points[index]
        end = pixel_points[index + 1]
        segment_end = traversed + seg_len
        while target_index < len(targets) and targets[target_index] <= segment_end:
            ratio = (targets[target_index] - traversed) / seg_len
            output.append([
                float(start[0]) + ratio * (float(end[0]) - float(start[0])),
                float(start[1]) + ratio * (float(end[1]) - float(start[1])),
            ])
            target_index += 1
        traversed = segment_end

    last = [float(pixel_points[-1][0]), float(pixel_points[-1][1])]
    if output[-1] != last:
        output.append(last)
    return output


def resample_pixel_path(
    points: Iterable[Sequence[float]], spacing_px: float
) -> List[List[float]]:
    """Resample a pixel polyline at approximately equal pixel spacing."""
    pixel_points = _normalise_pixel_path(points)
    lengths = [
        math.hypot(pixel_points[i + 1][0] - pixel_points[i][0],
                   pixel_points[i + 1][1] - pixel_points[i][1])
        for i in range(len(pixel_points) - 1)
    ]
    return _resample_by_segment_lengths(pixel_points, lengths, float(spacing_px))


def resample_path_m(
    points: Iterable[Sequence[float]], calibration, spacing_m: float
) -> List[List[float]]:
    """Resample a pixel polyline by WGS84 ground distance."""
    pixel_points = _normalise_pixel_path(points)
    geo_points = convert_pixel_path_to_geo(pixel_points, calibration)
    lengths = [_haversine_m(geo_points[i], geo_points[i + 1]) for i in range(len(geo_points) - 1)]
    return _resample_by_segment_lengths(pixel_points, lengths, float(spacing_m))


def _ordered_task_points_report(snapped_task_points) -> list[dict]:
    """Build the ``ordered_task_points`` list for ``planning_report.json``."""
    if not snapped_task_points:
        return []
    items = sorted(
        list(snapped_task_points),
        key=lambda t: int(_point_value(t, "seq", default=0)),
    )
    result = []
    for tp in items:
        seq = int(_point_value(tp, "seq", default=0))
        point_type = int(_point_value(tp, "point_type", default=2))
        status = str(_point_value(tp, "status", default="ok"))
        entry = {
            "seq": seq,
            "point_type": point_type,
            "point_type_label": {0: "start", 1: "goal"}.get(point_type, "task"),
            "status": status,
        }
        # Coordinates
        for field, key in (("pixel_x", "snapped_x"), ("pixel_y", "snapped_y"),
                           ("pixel_x", "pixel_x"), ("pixel_y", "pixel_y"),
                           ("pixel_x", "x"), ("pixel_y", "y")):
            if entry.get(field) is None:
                val = _point_value(tp, key)
                if val is not None:
                    entry[field] = round(float(val), 3)
        for field, key in (("lon", "longitude"), ("lat", "latitude")):
            val = _point_value(tp, key)
            if val is not None:
                entry[field] = round(float(val), 8)
        nid = _point_value(tp, "snapped_node_id", "node_id")
        if nid is not None:
            entry["snapped_node_id"] = nid
        result.append(entry)
    return result


def _planned_segments_report(planning_result, metres_per_pixel) -> list[dict]:
    """Build the ``planned_segments`` list for ``planning_report.json``."""
    segments = getattr(planning_result, "segments", None) or []
    result = []
    for seg in segments:
        length_px = float(getattr(seg, "length_px", 0.0) or 0.0)
        status = getattr(seg, "status", "")
        item = {
            "segment_index": len(result) + 1,
            "from_seq": getattr(seg, "from_seq", None),
            "to_seq": getattr(seg, "to_seq", None),
            "from_virtual_node": getattr(seg, "from_virtual_node", ""),
            "to_virtual_node": getattr(seg, "to_virtual_node", ""),
            "success": status == "ok",
            "path_node_count": len(getattr(seg, "node_path", []) or []),
            "path_length_m": round(length_px * metres_per_pixel, 3) if metres_per_pixel else None,
            "task_virtual_nodes_inside_segment": [
                getattr(seg, "from_virtual_node", ""),
                getattr(seg, "to_virtual_node", ""),
            ],
            "unexpected_task_virtual_nodes": list(
                getattr(seg, "unexpected_task_virtual_nodes", []) or []
            ),
        }
        error = getattr(seg, "error", "")
        if error:
            item["error"] = str(error)
        result.append(item)
    return result


def _segment_report(planning_result, metres_per_pixel: Optional[float]) -> List[dict]:
    reports: List[dict] = []
    for segment in getattr(planning_result, "segments", []) or []:
        status = getattr(segment, "status", "")
        length_px = float(getattr(segment, "length_px", 0.0) or 0.0)
        item = {
            "from_seq": getattr(segment, "from_seq", None),
            "to_seq": getattr(segment, "to_seq", None),
            "success": status == "ok",
            "length_px": round(length_px, 3),
        }
        if metres_per_pixel is not None:
            item["length_m"] = round(length_px * metres_per_pixel, 3)
        error = getattr(segment, "error", "")
        if error:
            item["error"] = str(error)
        reports.append(item)
    return reports


# ---------------------------------------------------------------------------
# YAML text generation
# ---------------------------------------------------------------------------

def _yaml_text(geo_points: Sequence[Sequence[float]]) -> str:
    """Legacy helper — emits clean subject1_waypoints format only."""
    lines = ["subject1_waypoints:", "  waypoints:"]
    for seq, (lon, lat, altitude) in enumerate(geo_points, 1):
        lines.extend((
            f"    - name: wp_{seq:03d}",
            f"      latitude_deg: {float(lat):.8f}",
            f"      longitude_deg: {float(lon):.8f}",
            f"      altitude_m: {float(altitude):.3f}",
        ))
    return "\n".join(lines) + "\n"


def _adaptive_yaml_text(waypoints: Sequence[dict]) -> str:
    """Official vehicle YAML text — subject1_waypoints format only.

    No coordinate_system / epsg / seq / yaw at the top level.
    """
    from roadnet.waypoint_validator import build_subject1_yaml_text

    normalized = []
    for item in waypoints:
        wp = dict(item)
        if wp.get("latitude") is None and wp.get("latitude_deg") is not None:
            wp["latitude"] = wp["latitude_deg"]
        if wp.get("longitude") is None and wp.get("longitude_deg") is not None:
            wp["longitude"] = wp["longitude_deg"]
        if wp.get("latitude") is None or wp.get("longitude") is None:
            raise InvalidGeoCalibrationError(
                "稀疏航点缺少经纬度，无法生成无人车 waypoints YAML"
            )
        # build_subject1_yaml_text reads latitude_deg / longitude_deg preferentially
        wp.setdefault("latitude_deg", wp["latitude"])
        wp.setdefault("longitude_deg", wp["longitude"])
        normalized.append(wp)
    return build_subject1_yaml_text(normalized, default_altitude_m=DEFAULT_ALTITUDE_M)


def _adaptive_yaml_debug_text(waypoints: Sequence[dict]) -> str:
    """Debug-only YAML (seq/yaw/tags). Never use as vehicle upload file."""
    lines = [
        "# DEBUG ONLY — not for vehicle upload",
        "# Official file: subject1_waypoints.yaml",
        "debug_waypoints:",
    ]
    for item in waypoints:
        lon = item.get("longitude", item.get("longitude_deg"))
        lat = item.get("latitude", item.get("latitude_deg"))
        if lon is None or lat is None:
            raise InvalidGeoCalibrationError(
                "稀疏航点缺少经纬度，无法生成调试 waypoints YAML"
            )
        alt = item.get("altitude_m", item.get("altitude", DEFAULT_ALTITUDE_M))
        seq = int(item.get("seq", 0) or 0)
        lines.extend((
            f"  - seq: {seq}",
            f"    name: {item.get('name') or f'wp_{seq:03d}'}",
            f"    latitude_deg: {float(lat):.8f}",
            f"    longitude_deg: {float(lon):.8f}",
            f"    altitude_m: {float(alt or 0.0):.3f}",
            f"    yaw_deg: {float(item.get('yaw_deg', 0.0)):.2f}",
            f"    target_speed_mps: {float(item.get('target_speed_mps', 1.0)):.2f}",
            f"    arrival_radius_m: {float(item.get('arrival_radius_m', 2.0)):.2f}",
            f"    pass_through: {'true' if item.get('pass_through', True) else 'false'}",
            f"    tag: {item.get('tag', 'straight')}",
        ))
    return "\n".join(lines) + "\n"


def _assert_subject1_yaml_text(yaml_text: str, *, label: str = "yaml") -> str:
    """Reject legacy vehicle YAML payloads before writing."""
    text = (yaml_text or "").lstrip("\ufeff").lstrip()
    if not text.startswith("subject1_waypoints:"):
        raise PathExportError(
            f"{label} 顶层必须是 subject1_waypoints，禁止 coordinate_system/seq 旧格式"
        )
    if "\n  waypoints:" not in text and not text.startswith(
        "subject1_waypoints:\n  waypoints:"
    ):
        raise PathExportError(f"{label} 缺少第二级 waypoints")
    head = text.split("waypoints:", 1)[0]
    for banned in ("coordinate_system:", "epsg:", "note:"):
        if banned in head:
            raise PathExportError(
                f"{label} 含旧顶层字段 {banned.rstrip(':')}，禁止作为正式小车 YAML"
            )
    return yaml_text if yaml_text.endswith("\n") else (yaml_text + "\n")


def _remove_legacy_vehicle_yaml(output_dir: str) -> None:
    """Delete leftover legacy vehicle YAML so stale files cannot mislead users."""
    for name in (
        "waypoints.yaml",
        "subject1_waypoints.yaml",
        "waypoints_sparse_10m.yaml",
    ):
        path = os.path.join(output_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                head = fh.read(120)
        except OSError:
            continue
        head_l = head.lstrip("\ufeff").lstrip()
        is_legacy = (
            head_l.startswith("coordinate_system:")
            or head_l.startswith("waypoints:")
            or head_l.startswith("epsg:")
            or ("\n- seq:" in head and "subject1_waypoints:" not in head_l[:40])
        )
        if is_legacy or not head_l.startswith("subject1_waypoints:"):
            try:
                os.remove(path)
            except OSError:
                pass


# ────────────────────────────────────────────────────────────────────
#  subject1_waypoints.yaml — clean vehicle-facing YAML
#  Only name / latitude_deg / longitude_deg / altitude_m.
#  NO yaw_deg, speed, tag, pass_through, or other extra fields.
# ────────────────────────────────────────────────────────────────────

def _haversine_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Haversine distance in metres between two WGS84 points."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 6371000.0 * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _deduplicate_subject1_waypoints(
    waypoints: Sequence[dict],
    *,
    min_spacing_m: float = MIN_EXPORT_SPACING_M,
    closed_loop: bool = False,
) -> Tuple[List[dict], int, bool]:
    """Remove consecutive near-duplicate waypoints.

    Returns (deduplicated_list, removed_count, closed_loop).

    Rules:
    - If two consecutive waypoints are closer than *min_spacing_m*, discard
      the second one **unless** it is a task point (is_task_point==True).
    - If the last point duplicates the first, keep it when *closed_loop* is True,
      otherwise remove it.
    """
    if len(waypoints) < 2:
        return list(waypoints), 0, closed_loop

    deduped: List[dict] = [dict(waypoints[0])]
    removed = 0

    for i in range(1, len(waypoints)):
        prev = deduped[-1]
        curr = waypoints[i]

        lon1 = float(prev.get("longitude", 0))
        lat1 = float(prev.get("latitude", 0))
        lon2 = float(curr.get("longitude", 0))
        lat2 = float(curr.get("latitude", 0))

        is_task = bool(curr.get("is_task_point", False))
        dist_m = _haversine_distance_m(lon1, lat1, lon2, lat2)

        if dist_m < min_spacing_m and not is_task:
            removed += 1
            continue
        deduped.append(dict(curr))

    # ── Check if last point duplicates first ──
    if len(deduped) >= 2:
        first = deduped[0]
        last = deduped[-1]
        d_first_last = _haversine_distance_m(
            float(first.get("longitude", 0)), float(first.get("latitude", 0)),
            float(last.get("longitude", 0)), float(last.get("latitude", 0)),
        )
        if d_first_last < min_spacing_m:
            closed_loop = True
            # Don't remove — keep for closed loop scenarios

    return deduped, removed, closed_loop


def _subject1_yaml_text(
    waypoints: Sequence[dict],
    *,
    default_altitude_m: float = DEFAULT_ALTITUDE_M,
    name_digits: int = WAYPOINT_NAME_DIGITS,
    min_export_spacing_m: float = MIN_EXPORT_SPACING_M,
) -> Tuple[str, dict]:
    """Generate clean **vehicle-facing** subject1_waypoints.yaml.

    Returns ``(yaml_text, stats_dict)`` where *stats_dict* contains::

        subject1_waypoint_count
        removed_duplicate_count
        coordinate_order_checked
        lat_lon_swapped_detected
        default_altitude_m
        closed_loop
    """
    # ── Deduplicate ──
    deduped, removed_count, closed_loop = _deduplicate_subject1_waypoints(
        waypoints, min_spacing_m=min_export_spacing_m,
    )

    # ── Coordinate order sanity check ──
    # Longitude should be in [73, 135] (China), latitude in [3, 54]
    # If we find longitude values that look like latitudes and vice-versa,
    # flag it but still write what the user expects.
    lat_lon_swapped = False
    for item in deduped:
        lon = float(item.get("longitude", 0))
        lat = float(item.get("latitude", 0))
        if abs(lon) <= 90 and abs(lat) > 90:
            # lon looks like a latitude, lat looks like a longitude
            lat_lon_swapped = True
            break

    # ── Build YAML lines ──
    fmt = f"wp_{{:0{name_digits}d}}"
    lines = ["subject1_waypoints:", "  waypoints:"]

    for idx, item in enumerate(deduped):
        name = fmt.format(idx + 1)
        lat = float(item.get("latitude", 0))
        lon = float(item.get("longitude", 0))
        raw_alt = item.get("altitude_m", item.get("altitude"))
        try:
            alt = float(raw_alt) if raw_alt is not None else float(default_altitude_m)
        except (TypeError, ValueError):
            alt = float(default_altitude_m)
        # 无有效高度时使用任务点/配置默认高度
        if raw_alt is None or not math.isfinite(alt) or abs(alt) < 1e-9:
            alt = float(default_altitude_m)

        lines.append(f"    - name: {name}")
        lines.append(f"      latitude_deg: {lat:.8f}")
        lines.append(f"      longitude_deg: {lon:.8f}")
        lines.append(f"      altitude_m: {alt:.3f}")

    stats = {
        "subject1_waypoint_count": len(deduped),
        "removed_duplicate_count": removed_count,
        "coordinate_order_checked": True,
        "lat_lon_swapped_detected": lat_lon_swapped,
        "default_altitude_m": default_altitude_m,
        "closed_loop": closed_loop,
    }

    return "\n".join(lines) + "\n", stats


# ---------------------------------------------------------------------------
# CSV text generation (expanded columns for human inspection)
# ---------------------------------------------------------------------------

_CSV_COLUMNS = (
    "seq", "longitude", "latitude", "altitude",
    "x_pixel", "y_pixel", "x_enu", "y_enu",
    "yaw_deg", "spacing_to_prev_m", "tag",
    "target_speed_mps", "arrival_radius_m", "pass_through",
    "is_task_point", "is_corner", "is_intersection",
    "inside_image", "distance_to_graph_px",
)


def _adaptive_csv_text(waypoints: Sequence[dict]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_CSV_COLUMNS)
    for item in waypoints:
        writer.writerow((
            item["seq"],
            f"{float(item['longitude']):.8f}" if item.get("longitude") is not None else "",
            f"{float(item['latitude']):.8f}" if item.get("latitude") is not None else "",
            f"{float(item.get('altitude', 0.0)):.3f}",
            f"{float(item['x_pixel']):.3f}",
            f"{float(item['y_pixel']):.3f}",
            f"{float(item['x_enu']):.3f}" if item.get("x_enu") is not None else "",
            f"{float(item['y_enu']):.3f}" if item.get("y_enu") is not None else "",
            f"{float(item.get('yaw_deg', 0.0)):.2f}",
            f"{float(item.get('spacing_to_prev_m', 0.0)):.3f}",
            item.get("tag", "straight"),
            f"{float(item.get('target_speed_mps', 1.0)):.2f}",
            f"{float(item.get('arrival_radius_m', 2.0)):.2f}",
            "true" if item.get("pass_through", True) else "false",
            "true" if item.get("is_task_point", False) else "false",
            "true" if item.get("is_corner", False) else "false",
            "true" if item.get("is_intersection", False) else "false",
            "true" if item.get("inside_image", True) else "false",
            f"{float(item.get('distance_to_graph_px', 0)):.3f}" if item.get("distance_to_graph_px") is not None else "",
        ))
    return buffer.getvalue()


_VEHICLE_ADAPTIVE_CSV_COLUMNS = (
    "seq", "name", "latitude_deg", "longitude_deg", "altitude_m",
    "x_pixel", "y_pixel", "spacing_mode", "segment_distance_m",
    "local_angle_deg", "near_junction", "near_task_point", "mask_support_ratio",
)


def _vehicle_waypoints_adaptive_csv_text(
    waypoints: Sequence[dict],
    *,
    default_altitude_m: float = DEFAULT_ALTITUDE_M,
) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_VEHICLE_ADAPTIVE_CSV_COLUMNS)
    for item in waypoints:
        lat = item.get("latitude_deg", item.get("latitude"))
        lon = item.get("longitude_deg", item.get("longitude"))
        alt = item.get("altitude_m", item.get("altitude"))
        if alt is None:
            alt = default_altitude_m
        name = item.get("name") or f"wp_{int(item.get('seq', 0)):03d}"
        seg_dist = item.get(
            "segment_distance_m",
            item.get("spacing_to_prev_m", item.get("distance_to_prev_m", 0.0)),
        )
        writer.writerow((
            item.get("seq", ""),
            name,
            f"{float(lat):.8f}" if lat is not None else "",
            f"{float(lon):.8f}" if lon is not None else "",
            f"{float(alt):.3f}",
            f"{float(item.get('x_pixel', 0.0)):.3f}",
            f"{float(item.get('y_pixel', 0.0)):.3f}",
            item.get("spacing_mode", "straight_10m"),
            f"{float(seg_dist or 0.0):.3f}",
            f"{float(item.get('local_angle_deg', 0.0)):.3f}",
            "true" if item.get("near_junction") else "false",
            "true" if item.get("near_task_point") else "false",
            f"{float(item.get('mask_support_ratio', 1.0)):.4f}",
        ))
    return buffer.getvalue()


def vehicle_export_gate_ok(
    report: Optional[dict],
    *,
    max_allowed_spacing_m: float = 12.0,
) -> Tuple[bool, List[str]]:
    """Return (ok, reasons) for formal subject1_waypoints.yaml generation."""
    report = dict(report or {})
    reasons: List[str] = []
    if not bool(report.get("export_valid", False)):
        reasons.append("export_valid=false")
    if not bool(report.get("geometry_valid", False)):
        reasons.append("geometry_valid=false")
    if int(report.get("bad_segment_count", 0) or 0) > 0:
        reasons.append(f"bad_segment_count={report.get('bad_segment_count')}")
    if int(report.get("aba_backtrack_count", 0) or 0) > 0:
        reasons.append(f"aba_backtrack_count={report.get('aba_backtrack_count')}")
    if int(report.get("duplicate_consecutive_count", 0) or 0) > 0:
        reasons.append(
            f"duplicate_consecutive_count={report.get('duplicate_consecutive_count')}"
        )
    if int(report.get("line_of_sight_failed_count", 0) or 0) > 0:
        reasons.append(
            f"line_of_sight_failed_count={report.get('line_of_sight_failed_count')}"
        )
    max_sp = float(report.get("max_spacing_m") or 0.0)
    if max_sp > float(max_allowed_spacing_m) + 1e-6:
        reasons.append(f"max_spacing_m={max_sp}>{max_allowed_spacing_m}")
    return len(reasons) == 0, reasons


def export_subject1_waypoints_yaml(
    vehicle_waypoints: Sequence[dict],
    output_path: str,
    default_altitude_m: float = DEFAULT_ALTITUDE_M,
) -> dict:
    """Write formal subject1_waypoints.yaml (clean vehicle format).

    Returns stats dict including yaml_text / subject1_waypoint_count.
    """
    from roadnet.waypoint_validator import build_subject1_yaml_text

    # Prefer validator builder (latitude_deg before longitude_deg, wp_001…)
    # Normalize lon/lat aliases for _subject1 path as fallback.
    try:
        yaml_text = build_subject1_yaml_text(
            vehicle_waypoints, default_altitude_m=float(default_altitude_m),
        )
        stats = {
            "subject1_waypoint_count": len(vehicle_waypoints),
            "removed_duplicate_count": 0,
            "coordinate_order_checked": True,
            "lat_lon_swapped_detected": False,
            "default_altitude_m": float(default_altitude_m),
            "closed_loop": False,
            "yaml_text": yaml_text,
        }
    except Exception:
        # Fallback: coerce latitude/longitude keys then use path_export builder
        normalized = []
        for wp in vehicle_waypoints:
            item = dict(wp)
            if item.get("latitude") is None and item.get("latitude_deg") is not None:
                item["latitude"] = item["latitude_deg"]
            if item.get("longitude") is None and item.get("longitude_deg") is not None:
                item["longitude"] = item["longitude_deg"]
            normalized.append(item)
        yaml_text, stats = _subject1_yaml_text(
            normalized, default_altitude_m=float(default_altitude_m),
        )
        stats["yaml_text"] = yaml_text

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(yaml_text)
    return stats


def write_official_vehicle_waypoint_bundle(
    output_dir: str,
    vehicle_waypoints: Sequence[dict],
    validation_report: Optional[dict] = None,
    *,
    default_altitude_m: float = DEFAULT_ALTITUDE_M,
    preview_image=None,
    image_width: int = 0,
    image_height: int = 0,
    bad_segments: Optional[Sequence[dict]] = None,
    max_allowed_spacing_m: float = 12.0,
) -> dict:
    """Write subject1 / adaptive CSV / validation artifacts for any export entry.

    Formal YAML is only written when ``vehicle_export_gate_ok`` passes.
    ``waypoints.yaml`` is written as an identical copy of subject1 when valid.
    """
    from roadnet.waypoint_validator import (
        bad_segments_csv_text,
        render_validation_overlay,
        vehicle_waypoints_invalid_csv_text,
    )

    os.makedirs(output_dir, exist_ok=True)
    # Always scrub stale legacy waypoints.yaml left by older exporters
    _remove_legacy_vehicle_yaml(output_dir)
    report = dict(validation_report or {})
    wps = list(vehicle_waypoints or [])
    bad = list(bad_segments or [])
    gate_ok, gate_reasons = vehicle_export_gate_ok(
        report, max_allowed_spacing_m=max_allowed_spacing_m,
    )
    written: List[str] = []
    yaml_text = None
    subject1_stats: dict = {}

    # Always write validation report + adaptive CSV (or INVALID)
    report_path = os.path.join(output_dir, "waypoint_validation_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    written.append("waypoint_validation_report.json")

    bad_path = os.path.join(output_dir, "bad_segments.csv")
    with open(bad_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(bad_segments_csv_text(bad))
    written.append("bad_segments.csv")

    try:
        overlay = render_validation_overlay(
            wps, bad, preview_image,
            image_width=image_width, image_height=image_height,
            near_duplicates=report.get("non_consecutive_near_duplicates"),
            aba_indices=report.get("aba_indices"),
        )
        overlay_path = os.path.join(output_dir, "waypoint_validation_overlay.png")
        cv2.imwrite(overlay_path, overlay)
        written.append("waypoint_validation_overlay.png")
    except Exception:
        pass

    adaptive_csv = _vehicle_waypoints_adaptive_csv_text(
        wps, default_altitude_m=default_altitude_m,
    )
    if gate_ok and wps:
        s1_path = os.path.join(output_dir, "subject1_waypoints.yaml")
        subject1_stats = export_subject1_waypoints_yaml(
            wps, s1_path, default_altitude_m=default_altitude_m,
        )
        yaml_text = _assert_subject1_yaml_text(
            subject1_stats.get("yaml_text") or "",
            label="subject1_waypoints.yaml",
        )
        # Re-write after assert (export_subject1 already wrote; ensure identical)
        with open(s1_path, "w", encoding="utf-8") as fh:
            fh.write(yaml_text)
        written.append("subject1_waypoints.yaml")
        # Compatibility copy — identical subject1 content (NOT legacy format)
        wp_path = os.path.join(output_dir, "waypoints.yaml")
        with open(wp_path, "w", encoding="utf-8") as fh:
            fh.write(yaml_text)
        written.append("waypoints.yaml")
        csv_path = os.path.join(output_dir, "vehicle_waypoints_adaptive.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            fh.write(adaptive_csv)
        written.append("vehicle_waypoints_adaptive.csv")
    else:
        # Ensure no stale formal YAML remains when gate fails
        for name in ("subject1_waypoints.yaml", "waypoints.yaml"):
            path = os.path.join(output_dir, name)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        inv = os.path.join(output_dir, "vehicle_waypoints_adaptive_INVALID.csv")
        with open(inv, "w", encoding="utf-8", newline="") as fh:
            fh.write(vehicle_waypoints_invalid_csv_text(
                wps, default_altitude_m=default_altitude_m,
            ) if wps else adaptive_csv)
        written.append("vehicle_waypoints_adaptive_INVALID.csv")
        # Still write adaptive csv for inspection when we have waypoints
        if wps:
            csv_path = os.path.join(output_dir, "vehicle_waypoints_adaptive.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                fh.write(adaptive_csv)
            written.append("vehicle_waypoints_adaptive.csv")

    return {
        "yaml_export_valid": gate_ok and bool(wps),
        "gate_ok": gate_ok,
        "gate_reasons": gate_reasons,
        "official_vehicle_yaml": "subject1_waypoints.yaml" if gate_ok and wps else None,
        "vehicle_waypoint_count": len(wps),
        "average_spacing_m": report.get("average_spacing_m"),
        "max_spacing_m": report.get("max_spacing_m"),
        "geometry_valid": bool(report.get("geometry_valid", False)),
        "export_valid": bool(report.get("export_valid", False)),
        "bad_segment_count": int(report.get("bad_segment_count", 0) or 0),
        "los_failed_count": int(report.get("line_of_sight_failed_count", 0) or 0),
        "duplicate_count": int(report.get("duplicate_consecutive_count", 0) or 0),
        "aba_backtrack_count": int(report.get("aba_backtrack_count", 0) or 0),
        "written_files": written,
        "yaml_text": yaml_text,
        "subject1_stats": subject1_stats,
        "block_reason": "; ".join(gate_reasons) if not gate_ok else None,
    }


def _dense_path_geo_csv_text(dense_geo: Sequence[Sequence[float]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("seq", "longitude", "latitude", "altitude"))
    for seq, point in enumerate(dense_geo, 1):
        lon = float(point[0]) if len(point) > 0 else 0.0
        lat = float(point[1]) if len(point) > 1 else 0.0
        alt = float(point[2]) if len(point) > 2 else 0.0
        writer.writerow((seq, f"{lon:.8f}", f"{lat:.8f}", f"{alt:.3f}"))
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Waypoint preview rendering
# ---------------------------------------------------------------------------

def _render_waypoint_preview(
    dense_pixels: Sequence[Sequence[float]],
    waypoints: Sequence[dict],
    image_rgb=None,
    image_width: int = 0,
    image_height: int = 0,
) -> np.ndarray:
    """Render a preview image showing dense path + sparse waypoints.

    Coordinate system: original image pixel → preview pixel.
    The preview image is scaled so that the original image maps 1:1 onto
    the canvas (if image_rgb is provided, it is used as background after
    downscaling to fit a max-side of 2400 px).

    Drawing rules:
      - Background: the preview image (or a solid canvas).
      - Dense path: thin purple line, NO point numbers.
      - Sparse waypoints: filled circles, colour-coded by tag.
      - Max 50 waypoint seq-number labels.
      - Start / goal / task points get prominent labels.
      - Out-of-bounds waypoints: red X marker.
      - Direction arrows: every ~N waypoints (adaptive).
    """
    max_side = 2400

    if isinstance(image_rgb, np.ndarray) and image_rgb.ndim in (2, 3) and image_rgb.size:
        base = image_rgb
        if base.ndim == 2:
            base = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            base = cv2.cvtColor(base[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        bg_height, bg_width = base.shape[:2]

        # Determine if the background is preview or original.
        # If the dense path points refer to original image pixel coords,
        # and background is a preview (downsampled), we need to scale.
        scale = min(1.0, max_side / float(max(bg_height, bg_width)))
        if scale < 1.0:
            base = cv2.resize(base, (int(round(bg_width * scale)),
                                     int(round(bg_height * scale))),
                              interpolation=cv2.INTER_AREA)
            bg_height, bg_width = base.shape[:2]

        # The pixel coordinates are in ORIGINAL image space.
        # If the background is the original image, scale is simple.
        # If unknown, assume pixel coords match background dimensions.
        if image_width > 0 and image_height > 0:
            # Original image known; compute mapping.
            img_scale = bg_width / float(image_width) if image_width > 0 else scale
        else:
            img_scale = bg_width / float(max(bg_width, 1))  # assume 1:1

        offset_x = 0.0
        offset_y = 0.0

    else:
        # No background – auto-compute canvas from dense path bounds.
        coords = np.asarray(dense_pixels, dtype=np.float64)
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        span_x, span_y = max(1.0, max_x - min_x), max(1.0, max_y - min_y)
        scale = min(4.0, 1800.0 / span_x, 1200.0 / span_y)
        offset_x, offset_y = min_x - 25.0 / scale, min_y - 25.0 / scale
        bg_width = max(120, int(math.ceil(span_x * scale + 50)))
        bg_height = max(120, int(math.ceil(span_y * scale + 50)))
        base = np.full((bg_height, bg_width, 3), 245, dtype=np.uint8)
        img_scale = scale

    # Unified coordinate transform: original pixel → preview pixel.
    def _to_canvas(px: float, py: float) -> Tuple[int, int]:
        return (int(round((px - offset_x) * img_scale)),
                int(round((py - offset_y) * img_scale)))

    # ---- 1. Dense path: thin purple line, no numbers ----
    dense_line = np.asarray([_to_canvas(p[0], p[1]) for p in dense_pixels], dtype=np.int32)
    if len(dense_line) >= 2:
        cv2.polylines(base, [dense_line.reshape((-1, 1, 2))], False, (190, 70, 210),
                      max(1, int(round(2 * max(0.5, img_scale)))), cv2.LINE_AA)

    # ---- 2. Sparse waypoints: colour-coded dots ----
    tag_colors = {
        "start":        (40, 190, 40),    # green
        "goal":         (40, 40, 230),    # blue
        "task":         (230, 120, 30),   # orange
        "intersection": (0, 190, 240),    # cyan (darker)
        "sharp_turn":   (0, 90, 255),     # red-orange
        "corner":       (255, 120, 0),    # orange-red
        "curve":        (255, 120, 0),    # orange
        "straight":     (40, 220, 220),   # cyan (lighter)
    }
    out_of_bounds_color = (60, 60, 255)  # red

    sparse_xy = [_to_canvas(wp["x_pixel"], wp["y_pixel"]) for wp in waypoints]

    # ---- 3. Direction arrows (sparse, adaptive) ----
    arrow_interval = max(1, len(waypoints) // 25)
    for i in range(len(sparse_xy) - 1):
        if i % arrow_interval == 0:
            cv2.arrowedLine(base, sparse_xy[i], sparse_xy[i + 1],
                            (100, 30, 100), 1, cv2.LINE_AA, tipLength=0.3)

    # ---- 4. Draw waypoint dots and limited labels ----
    total_wps = len(waypoints)
    label_stride = max(1, total_wps // min(total_wps, MAX_WAYPOINT_NUMBER_LABELS))

    for idx, ((cx, cy), wp) in enumerate(zip(sparse_xy, waypoints)):
        inside = wp.get("inside_image", True)
        tag = wp.get("tag", "straight")
        is_task = tag in {"start", "goal", "task"}
        is_important = is_task or tag in {"intersection", "sharp_turn"}

        if not inside:
            # Out-of-bounds waypoint: red X marker
            cv2.line(base, (cx - 5, cy - 5), (cx + 5, cy + 5),
                     out_of_bounds_color, 2, cv2.LINE_AA)
            cv2.line(base, (cx - 5, cy + 5), (cx + 5, cy - 5),
                     out_of_bounds_color, 2, cv2.LINE_AA)
            continue

        color = tag_colors.get(tag, tag_colors["straight"])

        # Smaller dots for regular waypoints, larger for important ones
        if is_important:
            cv2.circle(base, (cx, cy), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(base, (cx, cy), 4, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(base, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(base, (cx, cy), 3, color, -1, cv2.LINE_AA)

        # Label: always for task points, limited for regular
        show_label = is_important or (label_stride > 0 and idx % label_stride == 0)
        if show_label:
            font_scale = 0.45 if is_important else 0.32
            thickness = 2 if is_important else 1
            cv2.putText(base, str(wp["seq"]), (cx + 7, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20),
                        thickness, cv2.LINE_AA)

    # ---- 5. Highlight start / goal with prominent labels ----
    for wp in waypoints:
        cx, cy = _to_canvas(wp["x_pixel"], wp["y_pixel"])
        tag = wp.get("tag", "")
        if tag == "start":
            cv2.rectangle(base, (cx + 10, cy - 28), (cx + 75, cy - 5),
                          (200, 245, 200), -1, cv2.LINE_AA)
            cv2.putText(base, "START", (cx + 13, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 0), 2, cv2.LINE_AA)
        elif tag == "goal":
            cv2.rectangle(base, (cx + 10, cy - 28), (cx + 72, cy - 5),
                          (220, 220, 250), -1, cv2.LINE_AA)
            cv2.putText(base, "GOAL", (cx + 13, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 30, 200), 2, cv2.LINE_AA)

    # ---- 6. Legend (top-left overlay) ----
    legend_items = (
        ("dense path", (190, 70, 210)),
        ("start", (40, 190, 40)),
        ("goal", (40, 40, 230)),
        ("task", (230, 120, 30)),
        ("intersection", (0, 190, 240)),
        ("corner/s.turn", (255, 120, 0)),
        ("straight", (40, 220, 220)),
    )
    lx, ly = 10, 10
    cv2.rectangle(base, (lx, ly), (lx + 155, ly + 20 + 18 * len(legend_items)),
                  (240, 240, 240), -1, cv2.LINE_AA)
    cv2.rectangle(base, (lx, ly), (lx + 155, ly + 20 + 18 * len(legend_items)),
                  (180, 180, 180), 1, cv2.LINE_AA)
    ly += 16
    for label, color in legend_items:
        if label == "dense path":
            cv2.line(base, (lx + 4, ly), (lx + 24, ly), color, 2, cv2.LINE_AA)
        else:
            cv2.circle(base, (lx + 9, ly - 1), 4, color, -1, cv2.LINE_AA)
        cv2.putText(base, label, (lx + 28, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 30), 1, cv2.LINE_AA)
        ly += 18

    return base


def _render_debug_preview(
    dense_pixels: Sequence[Sequence[float]],
    waypoints: Sequence[dict],
    image_rgb=None,
    image_width: int = 0,
    image_height: int = 0,
) -> np.ndarray:
    """Render a debug preview showing ALL waypoint numbers (for debugging)."""
    max_side = 2400

    if isinstance(image_rgb, np.ndarray) and image_rgb.ndim in (2, 3) and image_rgb.size:
        base = image_rgb
        if base.ndim == 2:
            base = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            base = cv2.cvtColor(base[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        bg_height, bg_width = base.shape[:2]
        scale = min(1.0, max_side / float(max(bg_height, bg_width)))
        if scale < 1.0:
            base = cv2.resize(base, (int(round(bg_width * scale)),
                                     int(round(bg_height * scale))),
                              interpolation=cv2.INTER_AREA)
            bg_height, bg_width = base.shape[:2]
        if image_width > 0 and image_height > 0:
            img_scale = bg_width / float(image_width)
        else:
            img_scale = bg_width / float(max(bg_width, 1))
        offset_x = 0.0
        offset_y = 0.0
    else:
        coords = np.asarray(dense_pixels, dtype=np.float64)
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        span_x, span_y = max(1.0, max_x - min_x), max(1.0, max_y - min_y)
        scale = min(4.0, 1800.0 / span_x, 1200.0 / span_y)
        offset_x, offset_y = min_x - 25.0 / scale, min_y - 25.0 / scale
        bg_width = max(120, int(math.ceil(span_x * scale + 50)))
        bg_height = max(120, int(math.ceil(span_y * scale + 50)))
        base = np.full((bg_height, bg_width, 3), 245, dtype=np.uint8)
        img_scale = scale

    def _to_canvas(px: float, py: float) -> Tuple[int, int]:
        return (int(round((px - offset_x) * img_scale)),
                int(round((py - offset_y) * img_scale)))

    # Dense path
    dense_line = np.asarray([_to_canvas(p[0], p[1]) for p in dense_pixels], dtype=np.int32)
    if len(dense_line) >= 2:
        cv2.polylines(base, [dense_line.reshape((-1, 1, 2))], False, (190, 70, 210),
                      max(1, int(round(2 * max(0.5, img_scale)))), cv2.LINE_AA)

    # Sparse waypoints – ALL labels shown
    for idx, wp in enumerate(waypoints):
        cx, cy = _to_canvas(wp["x_pixel"], wp["y_pixel"])
        inside = wp.get("inside_image", True)
        tag = wp.get("tag", "straight")
        color = {
            "start": (40, 190, 40), "goal": (40, 40, 230),
            "task": (230, 120, 30), "intersection": (0, 190, 240),
            "sharp_turn": (0, 90, 255), "corner": (255, 120, 0),
            "curve": (255, 120, 0), "straight": (40, 220, 220),
        }.get(tag, (40, 220, 220))

        if not inside:
            cv2.line(base, (cx - 5, cy - 5), (cx + 5, cy + 5), (60, 60, 255), 2)
            cv2.line(base, (cx - 5, cy + 5), (cx + 5, cy - 5), (60, 60, 255), 2)

        cv2.circle(base, (cx, cy), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(base, (cx, cy), 4, color, -1, cv2.LINE_AA)

        # ALL labels shown in debug mode
        cv2.putText(base, str(wp["seq"]), (cx + 5, cy - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (20, 20, 20), 1, cv2.LINE_AA)

    # START / GOAL labels
    for wp in waypoints:
        cx, cy = _to_canvas(wp["x_pixel"], wp["y_pixel"])
        tag = wp.get("tag", "")
        if tag == "start":
            cv2.putText(base, "START", (cx + 10, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 0), 2, cv2.LINE_AA)
        elif tag == "goal":
            cv2.putText(base, "GOAL", (cx + 10, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 30, 200), 2, cv2.LINE_AA)

    # Legend
    legend_y = 20
    for label, color in (("start=green", (40, 190, 40)), ("goal=blue", (40, 40, 230)),
                         ("task=orange", (230, 120, 30)), ("intersect=cyan", (0, 190, 240)),
                         ("corner=red", (255, 120, 0)), ("straight=ltcyan", (40, 220, 220))):
        cv2.circle(base, (15, legend_y), 5, (255, 255, 255), -1)
        cv2.circle(base, (15, legend_y), 4, color, -1)
        cv2.putText(base, label, (25, legend_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
        legend_y += 20

    return base


# ---------------------------------------------------------------------------
# Jump debug visualization
# ---------------------------------------------------------------------------

def _render_jump_debug_overlay(
    dense_pixels: Sequence[Sequence[float]],
    jump_debug_rows: list[dict],
    image_rgb=None,
    image_width: int = 0,
    image_height: int = 0,
) -> np.ndarray:
    """Render a jump debug overlay image.

    - Legitimate long straight roads: yellow line.
    - Truly suspicious jumps: red line with red circles at endpoints.
    - Each jump labelled with jump_id.
    """
    max_side = 2400

    if isinstance(image_rgb, np.ndarray) and image_rgb.ndim in (2, 3) and image_rgb.size:
        base = image_rgb
        if base.ndim == 2:
            base = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            base = cv2.cvtColor(base[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        bg_height, bg_width = base.shape[:2]
        scale = min(1.0, max_side / float(max(bg_height, bg_width)))
        if scale < 1.0:
            base = cv2.resize(base, (int(round(bg_width * scale)),
                                     int(round(bg_height * scale))),
                              interpolation=cv2.INTER_AREA)
            bg_height, bg_width = base.shape[:2]
        if image_width > 0 and image_height > 0:
            img_scale = bg_width / float(image_width)
        else:
            img_scale = bg_width / float(max(bg_width, 1))
        offset_x = 0.0
        offset_y = 0.0
    else:
        coords = np.asarray(dense_pixels, dtype=np.float64)
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        span_x, span_y = max(1.0, max_x - min_x), max(1.0, max_y - min_y)
        scale = min(4.0, 1800.0 / span_x, 1200.0 / span_y)
        offset_x, offset_y = min_x - 25.0 / scale, min_y - 25.0 / scale
        bg_width = max(120, int(math.ceil(span_x * scale + 50)))
        bg_height = max(120, int(math.ceil(span_y * scale + 50)))
        base = np.full((bg_height, bg_width, 3), 245, dtype=np.uint8)
        img_scale = scale

    def _to_canvas(px: float, py: float) -> Tuple[int, int]:
        return (int(round((px - offset_x) * img_scale)),
                int(round((py - offset_y) * img_scale)))

    # Light dense path underlay
    dense_line = np.asarray([_to_canvas(p[0], p[1]) for p in dense_pixels], dtype=np.int32)
    if len(dense_line) >= 2:
        cv2.polylines(base, [dense_line.reshape((-1, 1, 2))], False, (180, 180, 180),
                      1, cv2.LINE_AA)

    yellow = (0, 220, 220)   # BGR: legitimate long straight
    red = (60, 60, 255)      # BGR: truly suspicious

    for row in jump_debug_rows:
        sx, sy = _to_canvas(row["start_x"], row["start_y"])
        ex, ey = _to_canvas(row["end_x"], row["end_y"])
        is_suspicious = row.get("is_suspicious", True)
        jump_id = row.get("jump_id", 0)

        color = red if is_suspicious else yellow
        thickness = 3 if is_suspicious else 2
        cv2.line(base, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)

        if is_suspicious:
            cv2.circle(base, (sx, sy), 6, red, -1, cv2.LINE_AA)
            cv2.circle(base, (ex, ey), 6, red, -1, cv2.LINE_AA)

        # Label
        mid_x = (sx + ex) // 2
        mid_y = (sy + ey) // 2
        jump_index = row.get("jump_index", jump_id)
        dist_m = row.get("distance_m", row.get("length_m"))
        ratio = row.get("mask_support_ratio", row.get("road_support_ratio", 0))
        reason = str(row.get("reason", ""))[:18]
        dist_s = "" if dist_m is None else f" {float(dist_m):.1f}m"
        label = f"J{jump_index}{dist_s} r={float(ratio or 0):.2f} {reason}"
        cv2.putText(base, label, (mid_x + 4, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

    # Legend
    lx, ly = 10, 10
    cv2.rectangle(base, (lx, ly), (lx + 200, ly + 55), (240, 240, 240), -1)
    cv2.rectangle(base, (lx, ly), (lx + 200, ly + 55), (180, 180, 180), 1)
    cv2.line(base, (lx + 5, ly + 18), (lx + 25, ly + 18), yellow, 2, cv2.LINE_AA)
    cv2.putText(base, "合法长直路", (lx + 30, ly + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.line(base, (lx + 5, ly + 42), (lx + 25, ly + 42), red, 3, cv2.LINE_AA)
    cv2.putText(base, "异常跳边", (lx + 30, ly + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 30), 1, cv2.LINE_AA)

    return base


def _generate_jump_debug_files(
    output_dir: str,
    dense_pixels: Sequence[Sequence[float]],
    jump_debug_rows: list[dict],
    preview_image=None,
    image_width: int = 0,
    image_height: int = 0,
):
    """Generate path_jump_debug.csv and jump_debug_overlay.png."""
    if not jump_debug_rows:
        return

    # ---- CSV ----
    csv_path = os.path.join(output_dir, "path_jump_debug.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([
            "jump_index", "jump_id",
            "from_waypoint_index", "to_waypoint_index",
            "start_index", "end_index",
            "from_x", "from_y", "to_x", "to_y",
            "start_x", "start_y", "end_x", "end_y",
            "distance_m", "length_m", "length_px",
            "mask_support_ratio", "road_support_ratio", "chord_error_m",
            "is_suspicious", "reason",
            "segment_seq", "nearest_graph_edge_id",
            "is_curve_zone", "is_junction_zone",
        ])
        for row in jump_debug_rows:
            writer.writerow([
                row.get("jump_index", row.get("jump_id", "")),
                row.get("jump_id", ""),
                row.get("from_waypoint_index", row.get("start_index", "")),
                row.get("to_waypoint_index", row.get("end_index", "")),
                row.get("start_index", ""),
                row.get("end_index", ""),
                row.get("from_x", row.get("start_x", "")),
                row.get("from_y", row.get("start_y", "")),
                row.get("to_x", row.get("end_x", "")),
                row.get("to_y", row.get("end_y", "")),
                row.get("start_x", ""),
                row.get("start_y", ""),
                row.get("end_x", ""),
                row.get("end_y", ""),
                row.get("distance_m", row.get("length_m", "")),
                row.get("length_m", ""),
                row.get("length_px", ""),
                row.get("mask_support_ratio", row.get("road_support_ratio", "")),
                row.get("road_support_ratio", ""),
                row.get("chord_error_m", ""),
                "true" if row.get("is_suspicious") else "false",
                row.get("reason", ""),
                row.get("segment_seq", ""),
                row.get("nearest_graph_edge_id", ""),
                "true" if row.get("is_curve_zone") else "false",
                "true" if row.get("is_junction_zone") else "false",
            ])

    # ---- Overlay PNG ----
    overlay = _render_jump_debug_overlay(
        dense_pixels, jump_debug_rows,
        image_rgb=preview_image,
        image_width=image_width,
        image_height=image_height,
    )
    _write_png_atomically(os.path.join(output_dir, "jump_debug_overlay.png"), overlay)


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _write_png_atomically(path: str, image: np.ndarray):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise PathExportError(f"无法编码航点预览图：{path}")
    temp_path = path + ".tmp"
    try:
        encoded.tofile(temp_path)
        os.replace(temp_path, path)
    except Exception as exc:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        raise PathExportError(f"写入航点预览图失败：{path}\n原因：{exc}") from exc


def _write_text_atomically(path: str, content: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp_path = path + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        os.replace(temp_path, path)
    except Exception as exc:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        raise PathExportError(f"写入文件失败：{path}\n原因：{exc}") from exc


def _write_bundle_atomically(output_dir: str, contents: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    temp_paths = {}
    try:
        for filename, content in contents.items():
            final_path = os.path.join(output_dir, filename)
            temp_path = final_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8", newline="") as stream:
                stream.write(content)
            temp_paths[filename] = temp_path
        for filename, temp_path in temp_paths.items():
            os.replace(temp_path, os.path.join(output_dir, filename))
    except Exception as exc:
        for temp_path in temp_paths.values():
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
        raise PathExportError(f"写入路径导出目录失败：{output_dir}\n原因：{exc}") from exc


# ---------------------------------------------------------------------------
# Main export entry point
# ---------------------------------------------------------------------------

def export_planned_path(
    planned_path_pixel: Iterable[Sequence[float]],
    output_dir: str,
    geo_calibration,
    *,
    planning_result=None,
    task_point_count: int = 0,
    planned_path_edges: Optional[Sequence] = None,
    resample_spacing_m: Optional[float] = None,
    adaptive_config: Optional[AdaptiveWaypointConfig] = None,
    final_graph=None,
    snapped_task_points=None,
    path_node_sequence: Optional[Sequence] = None,
    preview_image=None,
    image_width: int = 0,
    image_height: int = 0,
    default_altitude_m: float = DEFAULT_ALTITUDE_M,
    road_mask=None,
) -> dict:
    """Export dense debug paths and an adaptive sparse vehicle waypoint bundle.

    Validation flow
    ---------------
    1. Normalise dense path.
    2. Check dense path for out-of-bounds points (→ planning_report.json).
    3. If dense path is out of bounds, **stop** — do not resample, do not
       generate vehicle files.  Raise PathOutOfBoundsError.
    4. Detect suspicious jumps in dense path (record in report, optional stop).
    5. Pre-validate geo calibration.
    6. Adaptive resample with bounds & roundtrip checks.
    7. Check sparse waypoints for out-of-bounds.
    8. Generate files:
         - DEBUG files: always.
         - VEHICLE files (waypoints_sparse_10m.yaml/.csv): only if all checks pass.
         - waypoints_sparse_10m_INVALID.csv: if sparse waypoints are out of bounds.
    """
    original_pixels = _normalise_pixel_path(planned_path_pixel)
    if not output_dir or not str(output_dir).strip():
        raise PathExportError("输出文件夹路径为空")
    output_dir = os.path.abspath(output_dir)

    # Resolve image dimensions
    img_w = image_width or int(getattr(geo_calibration, "image_width", 0) or 0)
    img_h = image_height or int(getattr(geo_calibration, "image_height", 0) or 0)

    # ---- STAGE 1: Validate geo calibration ----
    geo_converter = None
    try:
        geo_converter = _calibration_converter(geo_calibration)
    except InvalidGeoCalibrationError:
        geo_converter = None

    # ---- STAGE 2: Dense path bounds check ----
    dense_bounds = _check_dense_path_bounds(original_pixels, img_w, img_h)
    dense_oob = dense_bounds["dense_path_out_of_bounds_count"] > 0

    # ---- STAGE 3: Suspicious jump detection (two-level) ----
    # Prefer explicit road_mask; fall back to graph editor layer if present.
    if road_mask is None:
        try:
            if final_graph is not None and hasattr(final_graph, "get_layer_data"):
                road_mask = final_graph.get_layer_data("mask")
        except Exception:
            road_mask = None
    jump_report = _detect_suspicious_jumps(
        original_pixels, geo_converter,
        final_graph=final_graph,
        road_mask=road_mask,
    )

    # ---- STAGE 4: Stop early if dense path is out of bounds ----
    # Generate a minimal report first so the user has debug info.
    if dense_oob:
        _generate_bounds_failure_report(output_dir, original_pixels, dense_bounds,
                                         jump_report, img_w, img_h)
        # Also attempt to generate a preview showing the problem.
        try:
            preview = _render_waypoint_preview(original_pixels, [], preview_image,
                                               image_width=img_w, image_height=img_h)
            _write_png_atomically(os.path.join(output_dir, "waypoint_preview.png"), preview)
        except Exception:
            pass
        raise PathOutOfBoundsError(
            f"原始规划路径已经越界，请检查 final_graph 或路径规划结果。\n\n"
            f"总点数: {dense_bounds['dense_path_point_count']}, "
            f"越界点数: {dense_bounds['dense_path_out_of_bounds_count']}\n\n"
            f"图像尺寸: {img_w}×{img_h}\n"
            f"详见 planning_report.json"
        )

    # ---- STAGE 5: Geo conversion of dense path ----
    if geo_converter is None:
        raise InvalidGeoCalibrationError(
            "geo_calibration 无效，无法生成经纬度路径和无人车 waypoints。\n"
            "请先完成有效的坐标标定。"
        )
    dense_geo = convert_pixel_path_to_geo(original_pixels, geo_calibration)

    # ---- STAGE 5b: Layered path diagnostics (planned_segments → dense_path) ----
    from .path_layer_diagnostics import (
        PathLayerDiagConfig,
        classify_aba_source,
        run_layered_path_diagnostics,
    )

    def _extract_graph_nodes_edges(graph):
        if graph is None:
            return [], []
        if isinstance(graph, dict):
            return list(graph.get("nodes") or []), list(graph.get("edges") or [])
        nodes = list(getattr(graph, "nodes", None) or [])
        edges = list(getattr(graph, "edges", None) or [])
        # normalize object-style edges/nodes to dicts if needed
        def _as_dict(obj):
            if isinstance(obj, dict):
                return obj
            if hasattr(obj, "__dict__"):
                return dict(obj.__dict__)
            return obj
        return [_as_dict(n) for n in nodes], [_as_dict(e) for e in edges]

    graph_nodes, graph_edges = _extract_graph_nodes_edges(final_graph)
    mpp_est = getattr(geo_calibration, "pixel_resolution_estimated_m", None)
    try:
        mpp_est = float(mpp_est) if mpp_est else 0.5
    except (TypeError, ValueError):
        mpp_est = 0.5
    layer_diag = run_layered_path_diagnostics(
        planning_result,
        graph_nodes,
        graph_edges,
        snapped_task_points=snapped_task_points,
        dense_path_pixel=original_pixels,
        geo_calibration=geo_calibration,
        config=PathLayerDiagConfig(metres_per_pixel=mpp_est),
        output_dir=output_dir,
        preview_image=preview_image,
        image_width=img_w,
        image_height=img_h,
    )
    planned_segments_valid = bool(layer_diag.planned_segments_valid)
    dense_path_raw_valid = bool(layer_diag.dense_path_valid)
    dense_path_layer_valid = dense_path_raw_valid  # alias for legacy fields
    dense_bad_segments = list(layer_diag.dense_bad_segments or [])
    dense_step_too_large = any(
        str(b.get("reason") or "") == "step_distance_too_large" for b in dense_bad_segments
    )
    # Prefer edge-expanded dense path whenever expansion produced points
    # (even if raw spacing is coarse — resampling will densify for vehicles).
    if planned_segments_valid and layer_diag.dense_path_points:
        original_pixels = [
            [float(p[0]), float(p[1])] for p in layer_diag.dense_path_points
        ]
        try:
            dense_geo = convert_pixel_path_to_geo(original_pixels, geo_calibration)
        except Exception:
            pass

    # ---- STAGE 6: Compute authoritative task order from planning_result ----
    # ★ STRICTLY from segment endpoints — do NOT scan path_node_sequence.
    # ★ This is the SINGLE SOURCE OF TRUTH for task visit order.
    actual_task_visit_order: list = []
    segment_validation_errors: list[str] = []
    all_unexpected_vns: list[str] = []
    if planning_result is not None:
        for seg in getattr(planning_result, "segments", []) or []:
            from_seq = getattr(seg, "from_seq", None)
            to_seq = getattr(seg, "to_seq", None)
            is_ok = getattr(seg, "status", "") == "ok"
            unexpected_vns = list(getattr(seg, "unexpected_task_virtual_nodes", []) or [])

            if is_ok:
                if from_seq is not None and from_seq not in actual_task_visit_order:
                    actual_task_visit_order.append(int(from_seq))
                if to_seq is not None and to_seq not in actual_task_visit_order:
                    actual_task_visit_order.append(int(to_seq))
                if unexpected_vns:
                    segment_validation_errors.append(
                        f"段 {from_seq}→{to_seq} 意外经过 task virtual 节点: {unexpected_vns}"
                    )
                    for vn in unexpected_vns:
                        if vn not in all_unexpected_vns:
                            all_unexpected_vns.append(str(vn))
            else:
                err = getattr(seg, "error", "") or "unknown"
                segment_validation_errors.append(
                    f"段 {from_seq}→{to_seq} 规划失败: {err[:80]}"
                )
    # ★ 生成 expected_task_visit_order = [1, 2, ..., N]
    n_tasks = task_point_count or len(getattr(planning_result, "task_sequence", []) or [])
    expected_full = list(range(1, n_tasks + 1)) if n_tasks > 0 else []
    # ★ Authoritative task order check
    task_order_matches = (actual_task_visit_order == expected_full)
    # ★ Segment isolation check: no unexpected VNs in any segment
    segment_isolation_valid = (len(segment_validation_errors) == 0 and len(all_unexpected_vns) == 0)

    # ---- STAGE 7: Adaptive resampling ----
    # dense_path_raw_valid=false (e.g. step_distance_too_large) is a mid-layer
    # warning only; vehicle resampling may resolve it. Only planned_segments
    # hard-fail blocks the vehicle pipeline.
    config = adaptive_config or AdaptiveWaypointConfig()
    if resample_spacing_m is not None:
        spacing = float(resample_spacing_m)
        config.straight_spacing_m = spacing
        config.curve_spacing_m = min(config.curve_spacing_m, spacing)
        config.sharp_turn_spacing_m = min(config.sharp_turn_spacing_m, spacing)
        config.intersection_spacing_m = min(config.intersection_spacing_m, spacing)
        config.task_point_spacing_m = min(config.task_point_spacing_m, spacing)

    skip_vehicle_pipeline = not planned_segments_valid

    adaptive: AdaptiveWaypointResult = adaptive_resample_waypoints(
        original_pixels,
        geo_calibration,
        final_graph,
        snapped_task_points,
        path_node_sequence,
        planned_path_edges,
        config=config,
        image_width=img_w,
        image_height=img_h,
        actual_task_visit_order=actual_task_visit_order,
        expected_task_visit_order=expected_full,
        segment_isolation_valid=segment_isolation_valid,
        unexpected_task_virtual_nodes=all_unexpected_vns,
        road_mask=road_mask,
    )
    pixels = adaptive.sparse_waypoints_pixel
    geo_points = adaptive.sparse_waypoints_geo
    waypoints = adaptive.waypoints
    resample_report = adaptive.report
    densified_pixels = list(adaptive.dense_path_pixel) if adaptive.dense_path_pixel else list(original_pixels)
    try:
        dense_geo = convert_pixel_path_to_geo(densified_pixels, geo_calibration)
    except Exception:
        dense_geo = convert_pixel_path_to_geo(original_pixels, geo_calibration)

    # ---- STAGE 7b: Post-resample vehicle waypoint validation ----
    validation_cfg = WaypointValidationConfig(
        max_allowed_spacing_m=float(config.max_waypoint_spacing_m),
        hard_fail_spacing_m=20.0,
        allow_long_straight=False,
    )
    validation = validate_vehicle_waypoints(
        waypoints,
        dense_path_pixel=densified_pixels,
        geo_calibration=geo_calibration,
        final_graph=final_graph,
        task_points=snapped_task_points,
        road_mask=road_mask,
        path_node_sequence=path_node_sequence,
        config=validation_cfg,
        adaptive_config=config,
        default_altitude_m=default_altitude_m,
        image_width=img_w,
        image_height=img_h,
    )
    waypoints = validation.waypoints
    pixels = [[float(wp["x_pixel"]), float(wp["y_pixel"])] for wp in waypoints]
    geo_points = []
    for wp in waypoints:
        lon = wp.get("longitude_deg", wp.get("longitude"))
        lat = wp.get("latitude_deg", wp.get("latitude"))
        alt = wp.get("altitude_m", wp.get("altitude", 0.0))
        if lon is not None and lat is not None:
            geo_points.append([float(lon), float(lat), float(alt or 0.0)])
    validation_report = dict(validation.report or {})
    bad_segments = list(validation.bad_segments or [])
    # Fold validation flags into resample report for downstream consumers
    resample_report = dict(resample_report)
    resample_report["geometry_valid"] = bool(validation_report.get("geometry_valid"))
    resample_report["validation"] = validation_report
    resample_report["vehicle_waypoint_count"] = len(waypoints)
    resample_report["max_spacing_m"] = validation_report.get("max_spacing_m")
    resample_report["average_spacing_m"] = validation_report.get("average_spacing_m")

    # ---- STAGE 8: Sparse waypoint bounds check ----
    wp_bounds = _check_waypoint_bounds(waypoints, img_w, img_h)

    # ★ CROSS-VALIDATION: verify resample_report matches planning_report
    # If these diverge, something is broken in the data pipeline.
    _resample_tov = resample_report.get("task_order_valid", None)
    _pipeline_warning: Optional[str] = None
    if (_resample_tov is not None and task_order_matches is not None
            and _resample_tov != task_order_matches):
        # Data pipeline mismatch — resample report disagrees with planning.
        # This should never happen; if it does, the resample path received
        # wrong arguments or _semantic_check fallback logic triggered.
        _pipeline_warning = (
            f"planning_task_order_matches={task_order_matches} "
            f"but resample_task_order_valid={_resample_tov} "
            f"(source={resample_report.get('task_order_source', '?')})"
        )

    # ---- Path length calculations ----
    path_length_m = float(resample_report.get("path_length_m", 0.0))
    path_length_px = _polyline_length(original_pixels,
                                      lambda a, b: math.hypot(b[0] - a[0], b[1] - a[1]))
    metres_per_pixel = getattr(geo_calibration, "pixel_resolution_estimated_m", None)
    try:
        metres_per_pixel = float(metres_per_pixel) if metres_per_pixel is not None else None
    except (TypeError, ValueError):
        metres_per_pixel = None
    if metres_per_pixel is None and path_length_px > 0:
        metres_per_pixel = path_length_m / path_length_px

    # ★ export_valid: 综合所有权威检查（统一判定）+ 航点验收器
    semantic_valid = resample_report.get("semantic_valid", False)
    sparse_geometry_valid = bool(resample_report.get("geometry_valid", True))
    dense_geometry_valid = (jump_report["suspicious_jump_count"] == 0)
    geometry_valid = dense_geometry_valid and sparse_geometry_valid and bool(
        validation_report.get("geometry_valid", False)
    )
    waypoint_ob_count = wp_bounds.get("out_of_bounds_count", 0)
    bad_roundtrip = resample_report.get("bad_roundtrip_count", 0)
    coordinate_valid = (
        wp_bounds["all_inside"]
        and bad_roundtrip == 0
        and bool(validation_report.get("coordinate_valid", False))
    )

    # Merge sparse chord LOS failures into jump debug rows
    sparse_jump_rows = list(resample_report.get("sparse_chord_jump_debug_rows") or [])
    for row in sparse_jump_rows:
        if not row.get("is_suspicious"):
            continue
        jump_report.setdefault("jump_debug_rows", []).append(row)
        jump_report["suspicious_jump_count"] = int(jump_report.get("suspicious_jump_count", 0)) + 1
        jump_report["has_jumps"] = True

    # ★ YAML gate: vehicle-layer acceptance (dense_path_raw is mid-layer warning only)
    # Block only when vehicle waypoints fail geometry / spacing / ABA / LOS / duplicates.
    max_spacing_m = float(validation_report.get("max_spacing_m") or 0.0)
    max_spacing_ok = max_spacing_m <= float(validation_cfg.max_allowed_spacing_m) + 1e-6
    vehicle_waypoints_valid = (
        bool(validation.export_valid)
        and bool(validation_report.get("geometry_valid", False))
        and int(validation_report.get("bad_segment_count", 0)) == 0
        and int(validation_report.get("aba_backtrack_count", 0)) == 0
        and int(validation_report.get("duplicate_consecutive_count", 0)) == 0
        and int(validation_report.get("line_of_sight_failed_count", 0)) == 0
        and max_spacing_ok
        and bool(validation_report.get("yaml_format_valid", False))
        and bool(validation_report.get("s_m_monotonic_valid", True))
    )
    # Raw dense step warnings resolved by vehicle resampling / insert
    dense_path_warning_resolved_by_resampling = bool(
        (not dense_path_raw_valid)
        and dense_step_too_large
        and vehicle_waypoints_valid
        and max_spacing_ok
    )
    # Raw dense long-step / jump diagnostics must not alone block YAML once
    # vehicle waypoints have been resampled and validated.
    export_valid = (
        vehicle_waypoints_valid
        and planned_segments_valid
        and coordinate_valid
        and semantic_valid
        and waypoint_ob_count == 0
        and bad_roundtrip == 0
        and sparse_geometry_valid
    )
    if not dense_path_warning_resolved_by_resampling:
        # Keep dense jump gate only when the long-step warning was NOT resolved
        export_valid = export_valid and dense_geometry_valid and geometry_valid
    else:
        # Prefer vehicle geometry as the authoritative export geometry
        geometry_valid = bool(validation_report.get("geometry_valid", False)) and sparse_geometry_valid

    aba_source = classify_aba_source(layer_diag.dense_report, validation_report)
    validation_report["aba_source"] = aba_source
    validation_report["planned_segments_valid"] = planned_segments_valid
    validation_report["dense_path_raw_valid"] = dense_path_raw_valid
    validation_report["dense_path_valid"] = dense_path_raw_valid  # legacy alias
    validation_report["vehicle_waypoints_valid"] = vehicle_waypoints_valid
    validation_report["dense_path_warning_resolved_by_resampling"] = (
        dense_path_warning_resolved_by_resampling
    )
    if skip_vehicle_pipeline:
        export_valid = False
        vehicle_waypoints_valid = False
        validation_report["vehicle_waypoints_valid"] = False
        validation_report["blocked_by_planned_segments"] = True
        dense_path_warning_resolved_by_resampling = False
        validation_report["dense_path_warning_resolved_by_resampling"] = False


    # ---- Compute snapped_task_points for report ----
    snapped_task_pts: list[dict] = []
    if snapped_task_points:
        for tp in snapped_task_points:
            entry = {
                "seq": int(getattr(tp, "seq", 0)),
                "point_type": int(getattr(tp, "point_type", 2)),
                "point_type_label": {0: "start", 1: "goal"}.get(
                    getattr(tp, "point_type", 2), "task"
                ),
                "status": str(getattr(tp, "status", "ok")),
                "original_pixel_x": round(float(getattr(tp, "original_x", 0)), 3),
                "original_pixel_y": round(float(getattr(tp, "original_y", 0)), 3),
                "snapped_pixel_x": round(float(getattr(tp, "snapped_x", 0)), 3),
                "snapped_pixel_y": round(float(getattr(tp, "snapped_y", 0)), 3),
                "virtual_node_id": getattr(tp, "virtual_node_id", None),
                "edge_id": getattr(tp, "edge_id", None),
                "node_id": getattr(tp, "node_id", None),
                "snap_distance": round(float(getattr(tp, "snap_distance", 0)), 3),
                "snap_method": str(getattr(tp, "snap_method", "none")),
            }
            snapped_task_pts.append(entry)

    # ---- AGGREGATED valid reasons ----
    valid_reasons: list[str] = []
    dense_warnings: list[str] = []
    if not planned_segments_valid:
        valid_reasons.append("planned_segments_valid=false")
        ff = layer_diag.first_failure or {}
        if ff.get("reason"):
            valid_reasons.append(f"first_failure={ff.get('reason')}")
    if not dense_path_raw_valid:
        if dense_path_warning_resolved_by_resampling:
            dense_warnings.append(
                "dense_path_raw_valid=false (step_distance_too_large resolved by resampling)"
            )
        else:
            dense_warnings.append("dense_path_raw_valid=false")
            if layer_diag.dense_report.get("aba_count"):
                dense_warnings.append(
                    f"dense_aba_count={layer_diag.dense_report.get('aba_count')}"
                )
    if not coordinate_valid:
        if not wp_bounds["all_inside"]:
            valid_reasons.append(f"waypoint_out_of_bounds({wp_bounds['out_of_bounds_count']})")
        if resample_report.get("bad_roundtrip_count", 0) > 0:
            valid_reasons.append(f"roundtrip_bad({resample_report['bad_roundtrip_count']})")
    if not semantic_valid:
        valid_reasons.append("semantic_invalid")
        if resample_report.get("repeated_task_virtual_nodes"):
            valid_reasons.append(
                f"repeated_virtual_nodes={resample_report['repeated_task_virtual_nodes']}"
            )
        if not resample_report.get("task_order_valid", False):
            valid_reasons.append("task_order_invalid")
        if resample_report.get("goal_appears_before_final_segment", False):
            valid_reasons.append("goal_before_final_segment")
        if not resample_report.get("segment_isolation_valid", True):
            valid_reasons.append("segment_isolation_invalid")
    if not geometry_valid and not dense_path_warning_resolved_by_resampling:
        valid_reasons.append(f"suspicious_jumps({jump_report['suspicious_jump_count']})")
    elif not dense_geometry_valid and dense_path_warning_resolved_by_resampling:
        dense_warnings.append(
            f"raw_dense_suspicious_jumps({jump_report['suspicious_jump_count']}) resolved by resampling"
        )
    if not vehicle_waypoints_valid:
        for reason in validation_report.get("failure_reasons") or []:
            valid_reasons.append(f"validation:{reason}")
        if not max_spacing_ok:
            valid_reasons.append(f"max_spacing_m={max_spacing_m}>12")
    if not task_order_matches:
        valid_reasons.append(
            f"actual_task_visit_order={actual_task_visit_order}≠expected={expected_full}"
        )
    if segment_validation_errors:
        valid_reasons.append(f"segment_validation_errors={len(segment_validation_errors)}")
    invalid_reason = "; ".join(valid_reasons) if valid_reasons else None

    # ---- STAGE 9: Build planning report ----
    ordered_task_pts = _ordered_task_points_report(snapped_task_points)
    planned_segs = _planned_segments_report(planning_result, metres_per_pixel)

    planning_report = {
        "success": bool(getattr(planning_result, "success", True)),
        "task_point_count": int(task_point_count),
        "ordered_task_points": ordered_task_pts,
        "snapped_task_points": snapped_task_pts,
        "planned_segments": planned_segs,
        # ── Authoritative task order ──
        "expected_task_visit_order": expected_full,
        "actual_task_visit_order": actual_task_visit_order,
        "task_order_matches": task_order_matches,
        # ── Segment isolation ──
        "segment_isolation_valid": segment_isolation_valid,
        "segment_validation_errors": segment_validation_errors,
        "unexpected_task_virtual_nodes": all_unexpected_vns,
        # ── Geometry ──
        "geometry_valid": geometry_valid,
        # ── Layered path diagnostics ──
        "planned_segments_valid": planned_segments_valid,
        "dense_path_raw_valid": dense_path_raw_valid,
        "dense_path_valid": dense_path_raw_valid,  # legacy alias (= raw)
        "vehicle_waypoints_valid": vehicle_waypoints_valid,
        "dense_path_warning_resolved_by_resampling": dense_path_warning_resolved_by_resampling,
        "dense_path_warnings": dense_warnings,
        "aba_source": aba_source,
        "path_layer_first_failure": layer_diag.first_failure,
        "dense_path_validation": layer_diag.dense_report,
        # ── Comprehensive checks ──
        "coordinate_valid": coordinate_valid,
        "roundtrip_valid": (bad_roundtrip == 0),
        "semantic_valid": semantic_valid,
        "export_valid": export_valid,
        "waypoint_validation": validation_report,
        "bad_segment_count": int(validation_report.get("bad_segment_count", 0)),
        "max_spacing_m": validation_report.get("max_spacing_m"),
        "dense_path_point_count": len(original_pixels),
        "dense_path_out_of_bounds_count": dense_bounds["dense_path_out_of_bounds_count"],
        "dense_path_out_of_bounds_examples": dense_bounds["dense_path_out_of_bounds_examples"][:20],
        "long_segment_candidate_count": jump_report.get("long_segment_candidate_count", 0),
        "suspicious_jump_count": jump_report["suspicious_jump_count"],
        "suspicious_jump_examples": jump_report.get("suspicious_jump_examples", [])[:20],
        "suspicious_jumps": jump_report["jumps"][:20],
        "sparse_waypoint_count": len(waypoints),
        "path_length_m": round(path_length_m, 3),
        "path_length_px": round(path_length_px, 3),
        "resample_mode": "adaptive_enu",
        "straight_spacing_m": float(config.straight_spacing_m),
        "average_waypoint_spacing_m": resample_report.get("average_spacing_m"),
        "planned_path_edges": list(planned_path_edges or []),
        "segments": _segment_report(planning_result, metres_per_pixel),
        "repeated_task_virtual_nodes": resample_report.get("repeated_task_virtual_nodes", []),
        "start_count": resample_report.get("start_count", -1),
        "goal_count": resample_report.get("goal_count", -1),
        "task_order_valid": resample_report.get("task_order_valid", None),
        "goal_appears_before_final_segment": resample_report.get(
            "goal_appears_before_final_segment", None
        ),
        "invalid_reason": invalid_reason,
        "recommended_vehicle_file": RECOMMENDED_VEHICLE_FILE if export_valid else None,
        "waypoints_out_of_bounds_count": wp_bounds["out_of_bounds_count"],
        "debug_files": list(_EXPORT_DEBUG_FILES) + (
            ["path_jump_debug.csv", "jump_debug_overlay.png"]
            if jump_report.get("jump_debug_rows")
            else []
        ),
        "_data_pipeline_warning": _pipeline_warning,
    }

    # ---- STAGE 10: Build dense path debug docs ----
    dense_pixel_doc = {
        "coordinate_system": "image_pixel",
        "point_count": len(densified_pixels),
        "source_vertex_count": len(original_pixels),
        "note": "densified along road centerline at export time; not A* search nodes",
        "path": [
            {"seq": seq, "x": round(point[0], 3), "y": round(point[1], 3)}
            for seq, point in enumerate(densified_pixels, 1)
        ],
    }
    dense_geo_doc = {
        "coordinate_system": "EPSG:4326",
        "point_count": len(dense_geo),
        "path": [
            {"seq": seq, "longitude": round(point[0], 8),
             "latitude": round(point[1], 8), "altitude": round(point[2], 3)}
            for seq, point in enumerate(dense_geo, 1)
        ],
    }

    # ---- STAGE 11: Build sparse path docs (for legacy compatibility aliases) ----
    pixel_doc = {
        "coordinate_system": "image_pixel",
        "point_count": len(pixels),
        "path": [
            {"seq": seq, "x": round(point[0], 3), "y": round(point[1], 3)}
            for seq, point in enumerate(pixels, 1)
        ],
    }
    geo_doc = {
        "coordinate_system": "EPSG:4326",
        "point_count": len(geo_points),
        "path": [
            {"seq": seq, "longitude": round(point[0], 8),
             "latitude": round(point[1], 8), "altitude": round(point[2], 3)}
            for seq, point in enumerate(geo_points, 1)
        ],
    }

    # ---- STAGE 12: Assemble and write output bundle ----
    # Always-written content
    contents = {
        # Debug dense files
        "global_path_dense_pixel.json": json.dumps(dense_pixel_doc, ensure_ascii=False, indent=2) + "\n",
        "global_path_dense_geo.json": json.dumps(dense_geo_doc, ensure_ascii=False, indent=2) + "\n",
        "global_path_dense_geo.csv": _dense_path_geo_csv_text(dense_geo),
        "vehicle_waypoints_adaptive.csv": _vehicle_waypoints_adaptive_csv_text(
            waypoints, default_altitude_m=default_altitude_m,
        ),
        "waypoint_validation_report.json": json.dumps(validation_report, ensure_ascii=False, indent=2) + "\n",
        "bad_segments.csv": bad_segments_csv_text(bad_segments),
        # Reports
        "waypoint_resample_report.json": json.dumps(resample_report, ensure_ascii=False, indent=2) + "\n",
        "planning_report.json": json.dumps(planning_report, ensure_ascii=False, indent=2) + "\n",
        "dense_path_validation_report.json": json.dumps({
            **(layer_diag.dense_report or {}),
            **(layer_diag.planned_report or {}),
            "dense_path_raw_valid": dense_path_raw_valid,
            "dense_path_valid": dense_path_raw_valid,
            "vehicle_waypoints_valid": vehicle_waypoints_valid,
            "export_valid": export_valid,
            "dense_path_warning_resolved_by_resampling": dense_path_warning_resolved_by_resampling,
            "dense_path_warnings": dense_warnings,
            "first_failure": layer_diag.first_failure,
        }, ensure_ascii=False, indent=2) + "\n",
        # Legacy compatibility aliases
        "global_path_pixel.json": json.dumps(pixel_doc, ensure_ascii=False, indent=2) + "\n",
        "global_path_geo.json": json.dumps(geo_doc, ensure_ascii=False, indent=2) + "\n",
    }

    # Vehicle files – only if valid
    subject1_stats: dict = {}
    if export_valid:
        # Official subject1 text
        if validation.yaml_text:
            _s1_text = validation.yaml_text
            subject1_stats = {
                "subject1_waypoint_count": len(waypoints),
                "removed_duplicate_count": int(validation_report.get("duplicate_removed_count", 0)),
                "coordinate_order_checked": True,
                "lat_lon_swapped_detected": False,
                "default_altitude_m": default_altitude_m,
                "closed_loop": False,
            }
        else:
            _s1_text, subject1_stats = _subject1_yaml_text(
                waypoints,
                default_altitude_m=default_altitude_m,
            )
        _s1_text = _assert_subject1_yaml_text(_s1_text, label="subject1_waypoints.yaml")
        contents["subject1_waypoints.yaml"] = _s1_text
        # Compatibility alias — MUST be identical subject1 format
        contents["waypoints.yaml"] = _s1_text
        # Sparse check file also uses subject1 format (no legacy seq header)
        contents["waypoints_sparse_10m.yaml"] = _s1_text
        contents["waypoints_sparse_10m.csv"] = _adaptive_csv_text(waypoints)
        # Optional debug YAML with seq/yaw — clearly named, not for vehicle
        try:
            contents["waypoints_sparse_10m_debug.yaml"] = _adaptive_yaml_debug_text(waypoints)
        except Exception:
            pass
        contents["global_path.csv"] = _adaptive_csv_text(waypoints)
        contents["global_path_geo.csv"] = _adaptive_csv_text(waypoints)
    else:
        # ★ Write INVALID CSV for debugging — DO NOT generate vehicle files
        invalid_csv = _adaptive_csv_text(waypoints)
        contents[_EXPORT_INVALID_FILE] = invalid_csv
        contents[_EXPORT_VEHICLE_INVALID_CSV] = vehicle_waypoints_invalid_csv_text(
            waypoints, default_altitude_m=default_altitude_m,
        )
        # Keep adaptive csv as the validated (but failed) set for inspection
        contents["vehicle_waypoints_adaptive.csv"] = vehicle_waypoints_invalid_csv_text(
            waypoints, default_altitude_m=default_altitude_m,
        )

    # ── Fold subject1_waypoints stats into planning_report ──
    planning_report["subject1_waypoints_exported"] = export_valid
    planning_report["subject1_waypoints_file"] = (
        "subject1_waypoints.yaml" if export_valid else None
    )
    planning_report["subject1_waypoint_count"] = subject1_stats.get(
        "subject1_waypoint_count", 0 if not export_valid else len(waypoints)
    )
    planning_report["default_altitude_m"] = default_altitude_m
    planning_report["removed_duplicate_count"] = subject1_stats.get(
        "removed_duplicate_count", 0
    )
    planning_report["coordinate_order_checked"] = subject1_stats.get(
        "coordinate_order_checked", False
    )
    planning_report["lat_lon_swapped_detected"] = subject1_stats.get(
        "lat_lon_swapped_detected", False
    )
    planning_report["closed_loop"] = subject1_stats.get("closed_loop", False)

    _remove_legacy_vehicle_yaml(output_dir)
    _write_bundle_atomically(output_dir, contents)

    # ---- STAGE 12: Generate waypoint preview ----
    preview = _render_waypoint_preview(densified_pixels, waypoints, preview_image,
                                       image_width=img_w, image_height=img_h)
    _write_png_atomically(os.path.join(output_dir, "waypoint_preview.png"), preview)

    # ---- STAGE 12b: Generate debug preview (all labels) ----
    debug_preview = _render_debug_preview(densified_pixels, waypoints, preview_image,
                                          image_width=img_w, image_height=img_h)
    _write_png_atomically(os.path.join(output_dir, "debug_preview.png"), debug_preview)

    # ---- STAGE 12c: Validation overlay ----
    from .waypoint_validator import render_validation_overlay
    validation_overlay = render_validation_overlay(
        waypoints, bad_segments, preview_image,
        image_width=img_w, image_height=img_h,
        near_duplicates=validation_report.get("non_consecutive_near_duplicates")
        if isinstance(validation_report.get("non_consecutive_near_duplicates"), list)
        else None,
        aba_indices=validation_report.get("aba_indices")
        if isinstance(validation_report.get("aba_indices"), list)
        else None,
    )
    _write_png_atomically(
        os.path.join(output_dir, "waypoint_validation_overlay.png"),
        validation_overlay,
    )

    # ---- STAGE 13c: Generate jump debug files (CSV + overlay PNG) ----
    jump_debug_rows = jump_report.get("jump_debug_rows", [])
    if jump_debug_rows:
        _generate_jump_debug_files(
            output_dir, original_pixels, jump_debug_rows,
            preview_image=preview_image,
            image_width=img_w, image_height=img_h,
        )

    # ---- Collect exported file list ----
    written_files = []
    for name in _EXPORT_DEBUG_FILES:
        written_files.append(name)
    if export_valid:
        written_files.extend(_EXPORT_VEHICLE_FILES)
        # Legacy aliases (only when valid)
        written_files.extend(("global_path.csv", "global_path_geo.csv", "waypoints.yaml"))
    else:
        written_files.append(_EXPORT_INVALID_FILE)
        written_files.append(_EXPORT_VEHICLE_INVALID_CSV)
    # Always-written legacy aliases
    written_files.extend(("global_path_pixel.json", "global_path_geo.json"))
    # Jump debug files if generated
    if jump_debug_rows:
        written_files.extend(("path_jump_debug.csv", "jump_debug_overlay.png"))

    return {
        "output_dir": output_dir,
        "exported_files": [os.path.join(output_dir, name) for name in written_files],
        "export_valid": export_valid,
        "recommended_vehicle_file": RECOMMENDED_VEHICLE_FILE if export_valid else None,
        "planned_segments_valid": planned_segments_valid,
        "dense_path_valid": dense_path_raw_valid,
        "dense_path_raw_valid": dense_path_raw_valid,
        "vehicle_waypoints_valid": vehicle_waypoints_valid,
        "dense_path_warning_resolved_by_resampling": dense_path_warning_resolved_by_resampling,
        "aba_source": aba_source,
        "path_layer_first_failure": layer_diag.first_failure,
        "path_layer_artifact_paths": dict(layer_diag.artifact_paths or {}),
        "waypoints": waypoints,
        "sparse_waypoints_pixel": pixels,
        "sparse_waypoints_geo": geo_points,
        "waypoint_resample_report": resample_report,
        "waypoint_validation_report": validation_report,
        "bad_segments": bad_segments,
        "planning_report": planning_report,
        "waypoint_bounds": wp_bounds,
        "suspicious_jump_report": jump_report,
        "planned_path_pixel": pixels,
        "dense_path_pixel": densified_pixels,
        "dense_path_geo": dense_geo,
        "dense_path_bounds": dense_bounds,
    }


def _generate_bounds_failure_report(
    output_dir: str,
    original_pixels: List[List[float]],
    dense_bounds: dict,
    jump_report: dict,
    img_w: int,
    img_h: int,
):
    """Generate debug files when dense path is out of bounds."""
    os.makedirs(output_dir, exist_ok=True)

    # Save dense path as debug file
    dense_pixel_doc = {
        "coordinate_system": "image_pixel",
        "point_count": len(original_pixels),
        "image_width": img_w,
        "image_height": img_h,
        "path": [
            {"seq": seq, "x": round(point[0], 3), "y": round(point[1], 3)}
            for seq, point in enumerate(original_pixels, 1)
        ],
    }

    failure_report = {
        "export_valid": False,
        "error": "dense_path_out_of_bounds",
        "error_description": "原始规划路径已经越界，无法继续导出。",
        "image_width": img_w,
        "image_height": img_h,
        **dense_bounds,
        **jump_report,
    }

    contents = {
        "global_path_dense_pixel.json": json.dumps(dense_pixel_doc, ensure_ascii=False, indent=2) + "\n",
        "planning_report.json": json.dumps(failure_report, ensure_ascii=False, indent=2) + "\n",
    }
    _write_bundle_atomically(output_dir, contents)


# ---------------------------------------------------------------------------
# Compatibility alias
# ---------------------------------------------------------------------------

def save_global_path(*args, **kwargs) -> dict:
    """Vehicle-path export alias retained for explicit GUI/service bindings."""
    return export_planned_path(*args, **kwargs)
