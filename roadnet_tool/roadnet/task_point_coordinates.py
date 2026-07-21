"""Task-point coordinate conversion and diagnostics.

All coordinates stored on TaskPoint.pixel_x/pixel_y are pixels in the active
RoadNet image, never scene or widget coordinates.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Iterable, Optional, Sequence


def _haversine_m(lon1, lat1, lon2, lat2) -> float:
    radius = 6378137.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    value = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * radius * math.asin(min(1.0, math.sqrt(value)))


def calibration_size(calibration) -> tuple[int, int]:
    return (
        int(getattr(calibration, "image_width", 0) or 0),
        int(getattr(calibration, "image_height", 0) or 0),
    )


def convert_task_points_to_image(points: Iterable, calibration, image_size: Sequence[int]):
    """Mutate task points with active-image pixel coordinates and return diagnostics."""
    if calibration is None or not bool(getattr(calibration, "is_valid", False)):
        raise RuntimeError("geo_calibration 无效，请先完成坐标校准。")
    converter = getattr(calibration, "wgs84_to_pixel", None)
    reverse = getattr(calibration, "pixel_to_wgs84", None)
    if not callable(converter):
        raise RuntimeError("geo_calibration 缺少 wgs84_to_pixel(lon, lat) 接口。")
    if not callable(reverse):
        raise RuntimeError("geo_calibration 缺少 pixel_to_wgs84(x, y) 接口。")

    width, height = int(image_size[0]), int(image_size[1])
    cal_width, cal_height = calibration_size(calibration)
    calibration_size_missing = cal_width <= 0 or cal_height <= 0
    size_mismatch = bool(
        cal_width > 0 and cal_height > 0 and
        (cal_width != width or cal_height != height)
    )
    scale_x = width / cal_width if size_mismatch else 1.0
    scale_y = height / cal_height if size_mismatch else 1.0

    print(f"[TaskImport] image_size = {width} x {height}")
    print(f"[TaskImport] calibration_image_size = {cal_width} x {cal_height}")
    print(f"[TaskImport] geo_calibration.valid = {bool(calibration.is_valid)}")
    if size_mismatch:
        print("[TaskImport] WARNING 校准文件尺寸与当前影像尺寸不一致，"
              f"按比例修正 pixel: scale_x={scale_x:.8f}, scale_y={scale_y:.8f}")
    elif calibration_size_missing:
        print("[TaskImport] WARNING calibration.json 缺少 image_width/image_height，无法检查尺寸一致性")

    diagnostics = []
    for point in points:
        lon = float(point.longitude)
        lat = float(point.latitude)
        alt = float(getattr(point, "altitude", 0.0) or 0.0)
        print(f"[TaskImport] seq={point.seq} lon={lon:.10f} lat={lat:.10f} alt={alt:.3f}")
        try:
            cal_x, cal_y = converter(lon, lat)
            pixel_x = float(cal_x) * scale_x
            pixel_y = float(cal_y) * scale_y
            inside = 0.0 <= pixel_x < width and 0.0 <= pixel_y < height
            point.pixel_x = pixel_x
            point.pixel_y = pixel_y
            point.inside_image = inside
            point.status = "ok" if inside else "warning"

            # Reverse active-image pixels back into calibration pixels first.
            reverse_x = pixel_x / scale_x
            reverse_y = pixel_y / scale_y
            back_lon, back_lat = reverse(reverse_x, reverse_y)
            roundtrip_error_m = _haversine_m(lon, lat, float(back_lon), float(back_lat))
            print(f"[TaskImport] seq={point.seq} -> pixel_x={pixel_x:.3f} pixel_y={pixel_y:.3f}")
            print(f"[TaskImport] seq={point.seq} inside_image={inside}")
            print(f"[TaskImport] seq={point.seq} point_type={point.point_type}")
            print(f"[TaskImport] seq={point.seq} back lon/lat = "
                  f"{float(back_lon):.10f}, {float(back_lat):.10f}")
            print(f"[TaskImport] seq={point.seq} roundtrip error meters = {roundtrip_error_m:.6f}")
            diagnostics.append({
                "seq": int(point.seq), "lon": lon, "lat": lat, "altitude": alt,
                "point_type": int(point.point_type), "pixel_x": pixel_x, "pixel_y": pixel_y,
                "inside_image": inside, "back_lon": float(back_lon), "back_lat": float(back_lat),
                "roundtrip_error_m": roundtrip_error_m, "status": point.status,
                "size_scaled": size_mismatch,
            })
        except Exception as exc:
            point.pixel_x = None
            point.pixel_y = None
            point.inside_image = False
            point.status = "failed"
            print(f"[TaskImport] seq={point.seq} conversion failed: {exc}")
            diagnostics.append({
                "seq": int(point.seq), "lon": lon, "lat": lat, "altitude": alt,
                "point_type": int(point.point_type), "pixel_x": None, "pixel_y": None,
                "inside_image": False, "back_lon": None, "back_lat": None,
                "roundtrip_error_m": None, "status": "failed", "error": str(exc),
                "size_scaled": size_mismatch,
            })
    return diagnostics, {
        "image_width": width, "image_height": height,
        "calibration_image_width": cal_width, "calibration_image_height": cal_height,
        "calibration_size_missing": calibration_size_missing,
        "size_mismatch": size_mismatch, "scale_x": scale_x, "scale_y": scale_y,
    }


def _point_segment_distance(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def nearest_graph_edge(pixel_x: float, pixel_y: float, edges: Optional[Iterable]):
    best_id, best_distance = None, math.inf
    for edge in edges or []:
        if edge.get("enabled", True) is False:
            continue
        points = edge.get("points_pixel") or edge.get("polyline") or []
        for left, right in zip(points, points[1:]):
            distance = _point_segment_distance(
                pixel_x, pixel_y, float(left[0]), float(left[1]), float(right[0]), float(right[1])
            )
            if distance < best_distance:
                best_id, best_distance = edge.get("id"), distance
    return best_id, (None if not math.isfinite(best_distance) else best_distance)


def build_task_point_debug_rows(points: Iterable, image_size, graph_edges=None):
    width, height = int(image_size[0]), int(image_size[1])
    rows = []
    for point in sorted(points or [], key=lambda item: int(item.seq)):
        px, py = point.pixel_x, point.pixel_y
        inside = bool(px is not None and py is not None and 0 <= px < width and 0 <= py < height)
        edge_id, distance = (None, None)
        if px is not None and py is not None:
            edge_id, distance = nearest_graph_edge(float(px), float(py), graph_edges)
        status = "failed" if px is None or py is None else ("ok" if inside else "warning")
        rows.append({
            "seq": int(point.seq), "lon": point.longitude, "lat": point.latitude,
            "altitude": float(point.altitude or 0.0), "point_type": int(point.point_type),
            "pixel_x": px, "pixel_y": py, "inside_image": inside,
            "nearest_edge_id": edge_id, "distance_px": distance, "status": status,
        })
    return rows


def save_task_points_debug_csv(rows, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "seq", "lon", "lat", "altitude", "point_type", "pixel_x", "pixel_y",
        "inside_image", "nearest_edge_id", "distance_px", "status",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path
