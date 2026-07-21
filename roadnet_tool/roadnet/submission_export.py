"""Competition-ready road-network overlay and submission bundle export."""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from roadnet.path_visualization import ordered_task_markers, sample_direction_arrows


SUBMISSION_FILES = (
    "competition_roadnet_overlay.png",
    "competition_roadnet_overlay.jpg",
    "competition_roadnet_overlay_debug.png",
    "final_graph.json",
    "global_path_geo.csv",
    "vehicle_waypoints_adaptive.csv",
    "waypoint_validation_report.json",
    "waypoint_validation_overlay.png",
    "subject1_waypoints.yaml",
    "waypoints.yaml",
    "submission_report.json",
)

# Always written (overlay + graph + path debug). Vehicle YAML files are gated.
SUBMISSION_CORE_FILES = (
    "competition_roadnet_overlay.png",
    "competition_roadnet_overlay.jpg",
    "competition_roadnet_overlay_debug.png",
    "final_graph.json",
    "global_path_geo.csv",
    "submission_report.json",
)


class SubmissionExportError(RuntimeError):
    pass


def _type_name(value) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _array_diagnostics(value, source: str) -> str:
    return (
        f"source={source}, type={_type_name(value)}, "
        f"shape={getattr(value, 'shape', None)}, "
        f"dtype={getattr(value, 'dtype', None)}, "
        f"size={getattr(value, 'size', None)}"
    )


def _qt_image_classes():
    """Load Qt types lazily so headless export/tests do not require PySide6."""
    try:
        from PySide6.QtGui import QImage, QPixmap
        return QImage, QPixmap
    except (ImportError, ModuleNotFoundError):
        return None, None


def _is_qpixmap(value) -> bool:
    _, qpixmap_type = _qt_image_classes()
    if qpixmap_type is not None and isinstance(value, qpixmap_type):
        return True
    # The method check also permits a faithful test double in headless CI.
    return type(value).__name__ == "QPixmap" and callable(getattr(value, "toImage", None))


def _is_qimage(value) -> bool:
    qimage_type, _ = _qt_image_classes()
    if qimage_type is not None and isinstance(value, qimage_type):
        return True
    required = ("bits", "bytesPerLine", "width", "height")
    return type(value).__name__ == "QImage" and all(callable(getattr(value, name, None)) for name in required)


def _qimage_to_bgr(image, source: str) -> np.ndarray:
    """Convert QImage storage to an owned uint8 BGR OpenCV array."""
    qimage_type, _ = _qt_image_classes()
    rgba_format = None
    if qimage_type is not None:
        rgba_format = getattr(qimage_type, "Format_RGBA8888", None)
        if rgba_format is None and hasattr(qimage_type, "Format"):
            rgba_format = getattr(qimage_type.Format, "Format_RGBA8888", None)
    # Real QImage instances are normalized to a known four-channel layout.
    # Headless test doubles already expose RGBA8888 bytes.
    if rgba_format is not None and callable(getattr(image, "convertToFormat", None)):
        image = image.convertToFormat(rgba_format)

    width = int(image.width())
    height = int(image.height())
    bytes_per_line = int(image.bytesPerLine())
    if width <= 0 or height <= 0 or bytes_per_line < width * 4:
        raise SubmissionExportError(
            f"Qt 图像数据无效：source={source}, type={_type_name(image)}, "
            f"width={width}, height={height}, bytesPerLine={bytes_per_line}"
        )
    byte_count = height * bytes_per_line
    bits = image.bits()
    try:
        raw = np.frombuffer(bits, dtype=np.uint8, count=byte_count)
    except (TypeError, ValueError, BufferError) as exc:
        # Some older bindings require explicitly exposing the buffer length.
        if hasattr(bits, "setsize"):
            bits.setsize(byte_count)
            raw = np.frombuffer(bits, dtype=np.uint8, count=byte_count)
        else:
            raise SubmissionExportError(
                f"无法读取 QImage.bits()：source={source}, type={_type_name(image)}, "
                f"bytesPerLine={bytes_per_line}, expectedBytes={byte_count}, error={exc}"
            ) from exc
    if raw.size < byte_count:
        raise SubmissionExportError(
            f"QImage 缓冲区长度不足：source={source}, type={_type_name(image)}, "
            f"actualBytes={raw.size}, expectedBytes={byte_count}"
        )
    rgba = raw.reshape(height, bytes_per_line)[:, :width * 4].reshape(height, width, 4)
    # cvtColor returns owned data, so it remains valid after the QImage dies.
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def _numpy_to_cv_image(value: np.ndarray, source: str, *, color_order: str) -> np.ndarray:
    diagnostics = _array_diagnostics(value, source)
    if value.dtype == object:
        raise SubmissionExportError(
            "比赛路网图输入不能是 dtype=object；这通常表示 QImage/QPixmap 被错误地 "
            f"np.asarray()，或 list 中包含不同尺寸图像。{diagnostics}"
        )
    if value.size == 0:
        raise SubmissionExportError(f"比赛路网图输入为空。{diagnostics}")
    if value.ndim not in (2, 3):
        raise SubmissionExportError(f"比赛路网图输入维度无效。{diagnostics}")
    if value.ndim == 3 and value.shape[2] not in (1, 3, 4):
        raise SubmissionExportError(f"比赛路网图输入通道数无效。{diagnostics}")
    if not (np.issubdtype(value.dtype, np.number) or value.dtype == np.bool_):
        raise SubmissionExportError(f"比赛路网图输入必须是数值数组。{diagnostics}")

    if value.dtype == np.uint8:
        array = np.ascontiguousarray(value)
    elif value.dtype == np.bool_:
        array = np.ascontiguousarray(value.astype(np.uint8) * 255)
    else:
        # Numeric inputs are converted deliberately; object inputs were
        # rejected above and are never hidden by astype(uint8).
        finite = np.nan_to_num(value, nan=0.0, posinf=255.0, neginf=0.0)
        if np.issubdtype(value.dtype, np.floating) and finite.size and float(np.max(finite)) <= 1.0:
            finite = finite * 255.0
        array = np.ascontiguousarray(np.clip(finite, 0, 255).astype(np.uint8))

    if array.ndim == 2:
        return array
    if array.shape[2] == 1:
        return array[..., 0]
    if color_order == "rgb":
        code = cv2.COLOR_RGBA2BGR if array.shape[2] == 4 else cv2.COLOR_RGB2BGR
        return cv2.cvtColor(array, code)
    if color_order == "bgr":
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR) if array.shape[2] == 4 else array
    raise SubmissionExportError(f"内部颜色顺序无效：source={source}, color_order={color_order}")


def image_input_to_bgr(value, source: str = "image_rgb", *, numpy_color_order: str = "rgb") -> np.ndarray:
    """Strictly convert QPixmap, QImage, or ndarray into an OpenCV image."""
    if value is None:
        raise SubmissionExportError(f"比赛路网图输入为 None：source={source}")
    if _is_qpixmap(value):
        value = value.toImage()
        if value is None or not _is_qimage(value):
            raise SubmissionExportError(
                f"QPixmap.toImage() 未返回有效 QImage：source={source}, type={_type_name(value)}"
            )
    if _is_qimage(value):
        return _qimage_to_bgr(value, source)
    if isinstance(value, np.ndarray):
        return _numpy_to_cv_image(value, source, color_order=numpy_color_order)
    raise SubmissionExportError(
        "比赛路网图只支持 QPixmap、QImage 或 numpy.ndarray；"
        f"{_array_diagnostics(value, source)}"
    )


_LAYER_IMAGE_KEYS = (
    # Skeleton state dictionaries should prefer the final displayed result.
    "optimized_skeleton",
    "current",
    "current_skeleton",
    "raw_skeleton",
    "skeleton",
    "image",
    "array",
    "mask",
    "data",
    "pixmap",
    "qimage",
)


def extract_image_array_from_layer(layer_data, source_name: str = "", warnings=None, _visited=None):
    """Best-effort image extraction for optional layer-manager payloads."""
    source_name = source_name or "layer"
    if layer_data is None:
        return None
    if isinstance(layer_data, np.ndarray):
        return layer_data
    try:
        if _is_qpixmap(layer_data):
            qimage = layer_data.toImage()
            if qimage is None or not _is_qimage(qimage):
                raise SubmissionExportError(
                    f"QPixmap.toImage() did not return a valid QImage; source={source_name}"
                )
            return _qimage_to_bgr(qimage, source_name)
        if _is_qimage(layer_data):
            return _qimage_to_bgr(layer_data, source_name)
    except Exception as exc:
        if warnings is not None:
            warnings.append(f"{source_name} layer image conversion failed: {exc}")
        return None

    if isinstance(layer_data, dict):
        visited = set() if _visited is None else _visited
        object_id = id(layer_data)
        if object_id in visited:
            if warnings is not None:
                warnings.append(f"{source_name} layer dictionary contains a recursive reference")
            return None
        visited.add(object_id)
        try:
            for key in _LAYER_IMAGE_KEYS:
                if key not in layer_data or layer_data[key] is None:
                    continue
                extracted = extract_image_array_from_layer(
                    layer_data[key], f"{source_name}.{key}", None, visited,
                )
                if extracted is not None:
                    return extracted
        finally:
            visited.discard(object_id)
        if warnings is not None:
            warnings.append(
                f"{source_name} layer is dict and no image array could be extracted; "
                f"available_keys={sorted(str(key) for key in layer_data.keys())}"
            )
        return None

    if warnings is not None:
        warnings.append(
            f"{source_name} layer has unsupported type and was skipped; "
            f"{_array_diagnostics(layer_data, source_name)}"
        )
    return None


def _mask_input_to_gray(value, source: str) -> np.ndarray:
    image = image_input_to_bgr(value, source, numpy_color_order="rgb")
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _checked_resize(image: np.ndarray, size, interpolation, source: str) -> np.ndarray:
    """Validate exact resize input diagnostics before entering OpenCV."""
    if not isinstance(image, np.ndarray):
        raise SubmissionExportError(
            f"cv2.resize 输入不是 numpy.ndarray：{_array_diagnostics(image, source)}"
        )
    diagnostics = _array_diagnostics(image, source)
    if image.dtype == object:
        raise SubmissionExportError(f"禁止将 dtype=object 传给 cv2.resize。{diagnostics}")
    if image.size == 0 or image.ndim not in (2, 3):
        raise SubmissionExportError(f"cv2.resize 输入无效。{diagnostics}")
    return cv2.resize(image, size, interpolation=interpolation)


def default_submission_dir(base_dir: str, now: Optional[datetime] = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return os.path.join(os.path.abspath(base_dir), "outputs", "submission", f"run_{stamp}")


def _native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    return value


def _point(item, x_key="x", y_key="y"):
    return int(round(float(item.get(x_key, item.get("x_pixel", 0))))), int(round(float(item.get(y_key, item.get("y_pixel", 0)))))


def _edge_points(edge, nodes_by_id):
    points = edge.get("points_pixel")
    if points is None or len(points) == 0:
        points = edge.get("polyline")
    if points is None:
        points = []
    if len(points) >= 2:
        return np.asarray([[float(p[0]), float(p[1])] for p in points], dtype=np.int32)
    start = nodes_by_id.get(edge.get("start"))
    end = nodes_by_id.get(edge.get("end"))
    if start is None or end is None:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray([_point(start), _point(end)], dtype=np.int32)


def _draw_graph(image, nodes, edges, *, debug=False, scale=1.0):
    nodes_by_id = {node.get("id"): node for node in nodes}
    width = max(2, int(round((3 if not debug else 2) * scale)))
    color = (40, 220, 255)  # BGR: 醒目的黄色/青色
    for edge in edges:
        if not edge.get("enabled", True):
            continue
        points = _edge_points(edge, nodes_by_id)
        if len(points) >= 2:
            cv2.polylines(image, [points.reshape((-1, 1, 2))], False, color, width, cv2.LINE_AA)
    if debug:
        for node in nodes:
            cv2.circle(image, _point(node), max(2, int(3 * scale)), (0, 255, 255), -1, cv2.LINE_AA)


def _draw_text(image, text, origin, scale=1.0, font_scale=0.65):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = font_scale * scale
    thickness = max(1, int(round(scale)))
    cv2.putText(image, str(text), origin, font, fs, (20, 20, 20), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, str(text), origin, font, fs, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_path(image, path, *, spacing=100.0, arrow_size=12.0, scale=1.0):
    if len(path) < 2:
        return
    points = np.asarray([[float(p[0]), float(p[1])] for p in path], dtype=np.int32)
    cv2.polylines(
        image, [points.reshape((-1, 1, 2))], False,
        (220, 70, 230), max(3, int(round(6 * scale))), cv2.LINE_AA,
    )
    for arrow in sample_direction_arrows(path, spacing, arrow_size):
        triangle = np.asarray(arrow["triangle"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillConvexPoly(image, triangle, (250, 220, 255), cv2.LINE_AA)
        cv2.polylines(image, [triangle], True, (80, 20, 100), max(1, int(scale)), cv2.LINE_AA)


def _normalize_track_waypoints(waypoints) -> list:
    """Normalize sparse / vehicle waypoints to [{seq,x,y,lon,lat}, ...]."""
    result = []
    for index, item in enumerate(waypoints or [], 1):
        if item is None:
            continue
        if isinstance(item, dict):
            x = item.get("x_pixel", item.get("x", item.get("pixel_x")))
            y = item.get("y_pixel", item.get("y", item.get("pixel_y")))
            lon = item.get("longitude", item.get("longitude_deg", item.get("lon")))
            lat = item.get("latitude", item.get("latitude_deg", item.get("lat")))
            seq = item.get("seq", item.get("index", index))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = item[0], item[1]
            lon = item[2] if len(item) >= 3 else None
            lat = item[3] if len(item) >= 4 else None
            seq = index
        else:
            x = getattr(item, "x_pixel", getattr(item, "pixel_x", getattr(item, "x", None)))
            y = getattr(item, "y_pixel", getattr(item, "pixel_y", getattr(item, "y", None)))
            lon = getattr(item, "longitude", getattr(item, "longitude_deg", None))
            lat = getattr(item, "latitude", getattr(item, "latitude_deg", None))
            seq = getattr(item, "seq", index)
        if x is None or y is None:
            continue
        result.append({
            "seq": int(seq),
            "x": float(x),
            "y": float(y),
            "lon": None if lon is None or lon == "" else float(lon),
            "lat": None if lat is None or lat == "" else float(lat),
        })
    return result


def _sample_track_waypoints_from_path(path, *, spacing_px: float = 40.0) -> list:
    """Sample visible track points from dense planned path when sparse is absent."""
    if len(path) < 2:
        return []
    spacing_px = max(8.0, float(spacing_px))
    samples = [{"seq": 1, "x": float(path[0][0]), "y": float(path[0][1]),
                "lon": None, "lat": None}]
    traveled = 0.0
    next_at = spacing_px
    for i in range(1, len(path)):
        x0, y0 = float(path[i - 1][0]), float(path[i - 1][1])
        x1, y1 = float(path[i][0]), float(path[i][1])
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= 1e-9:
            continue
        while traveled + seg >= next_at:
            t = (next_at - traveled) / seg
            samples.append({
                "seq": len(samples) + 1,
                "x": x0 + (x1 - x0) * t,
                "y": y0 + (y1 - y0) * t,
                "lon": None,
                "lat": None,
            })
            next_at += spacing_px
        traveled += seg
    last = path[-1]
    if (abs(samples[-1]["x"] - float(last[0])) > 1.0
            or abs(samples[-1]["y"] - float(last[1])) > 1.0):
        samples.append({
            "seq": len(samples) + 1,
            "x": float(last[0]),
            "y": float(last[1]),
            "lon": None,
            "lat": None,
        })
    return samples


def _draw_track_waypoints(
    image,
    waypoints,
    *,
    converter=None,
    debug=False,
    scale=1.0,
    label_every: int = 5,
):
    """Draw discrete track/vehicle waypoints on competition overlay."""
    points = list(waypoints or [])
    if not points:
        return
    color = (70, 210, 255)  # BGR amber/cyan for track points
    radius = max(4, int(round(5.5 * scale)))
    for index, wp in enumerate(points):
        center = (int(round(wp["x"])), int(round(wp["y"])))
        cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, center, radius, (20, 20, 20), max(1, int(scale)), cv2.LINE_AA)

        show_label = (
            index == 0
            or index == len(points) - 1
            or (index + 1) % max(1, int(label_every)) == 0
            or debug
        )
        if not show_label:
            continue
        lon, lat = wp.get("lon"), wp.get("lat")
        if (lon is None or lat is None) and callable(converter):
            try:
                lon, lat = converter(wp["x"], wp["y"])
            except Exception:
                lon = lat = None
        if lon is not None and lat is not None:
            text = f"wp{wp['seq']} WGS84 {lat:.5f},{lon:.5f}"
        else:
            text = f"wp{wp['seq']}"
        _draw_text(
            image, text,
            (center[0] + radius + 3, center[1] - radius),
            scale, 0.40 if not debug else 0.45,
        )


def _draw_markers(image, task_points, snapped_points, *, debug=False, scale=1.0):
    markers = ordered_task_markers(task_points, snapped_points)
    colors = {"start": (65, 210, 80), "goal": (55, 65, 245), "waypoint": (240, 145, 40)}
    for marker in markers:
        center = (int(round(marker["x"])), int(round(marker["y"])))
        radius = max(7, int((11 if marker["role"] in ("start", "goal") else 9) * scale))
        if marker["status"] == "failed":
            d = radius
            cv2.line(image, (center[0] - d, center[1] - d), (center[0] + d, center[1] + d), (0, 0, 255), max(2, int(3 * scale)), cv2.LINE_AA)
            cv2.line(image, (center[0] - d, center[1] + d), (center[0] + d, center[1] - d), (0, 0, 255), max(2, int(3 * scale)), cv2.LINE_AA)
        else:
            marker_color = (0, 165, 255) if marker["status"] == "warning" else colors[marker["role"]]
            cv2.circle(image, center, radius, marker_color, -1, cv2.LINE_AA)
            cv2.circle(image, center, radius, (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)
        status_suffix = ""
        if debug and marker["status"] in ("warning", "failed"):
            status_suffix = f" [{marker['status'].upper()}]"
        _draw_text(image, marker["label"] + status_suffix,
                   (center[0] + radius + 4, center[1] - radius), scale)

    if debug:
        snapped_by_seq = {int(getattr(point, "seq", 0)): point for point in (snapped_points or [])}
        for task in task_points or []:
            x, y = getattr(task, "pixel_x", None), getattr(task, "pixel_y", None)
            if x is None or y is None:
                continue
            original = (int(round(x)), int(round(y)))
            cv2.drawMarker(image, original, (0, 0, 255), cv2.MARKER_CROSS, max(10, int(16 * scale)), max(1, int(2 * scale)))
            snapped = snapped_by_seq.get(int(getattr(task, "seq", 0)))
            if snapped is not None:
                target = (int(round(snapped.snapped_x)), int(round(snapped.snapped_y)))
                _draw_dashed_line(image, original, target, (220, 180, 80), max(1, int(scale)))


def _draw_dashed_line(image, start, end, color, thickness, dash=10):
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    if length <= 0:
        return
    dx, dy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
    value = 0.0
    while value < length:
        a = (int(start[0] + dx * value), int(start[1] + dy * value))
        b_value = min(length, value + dash)
        b = (int(start[0] + dx * b_value), int(start[1] + dy * b_value))
        cv2.line(image, a, b, color, thickness, cv2.LINE_AA)
        value += dash * 2


def _append_unique(values, value):
    if value not in values:
        values.append(value)


def _optional_layer_to_gray(layer_data, source_name, skipped_layers, warnings):
    if layer_data is None:
        return None
    extraction_warnings = []
    array = extract_image_array_from_layer(layer_data, source_name, extraction_warnings)
    if array is None:
        _append_unique(skipped_layers, source_name)
        for warning in extraction_warnings:
            _append_unique(warnings, warning)
        return None
    try:
        # QImage/QPixmap extraction already yields BGR. Masks and skeletons
        # are normally one-channel, so BGR validation is the safe common path.
        converted = _numpy_to_cv_image(array, source_name, color_order="bgr")
        if converted.ndim == 2:
            return converted
        return cv2.cvtColor(converted, cv2.COLOR_BGR2GRAY)
    except Exception as exc:
        _append_unique(skipped_layers, source_name)
        _append_unique(warnings, f"{source_name} layer was skipped: {exc}")
        return None


def _overlay_debug_layers(image, road_mask, skeleton, skipped_layers=None, warnings=None):
    skipped_layers = skipped_layers if skipped_layers is not None else []
    warnings = warnings if warnings is not None else []
    h, w = image.shape[:2]
    if road_mask is not None:
        try:
            mask = _optional_layer_to_gray(road_mask, "road_mask", skipped_layers, warnings)
            if mask is not None:
                if mask.shape[:2] != (h, w):
                    mask = _checked_resize(mask, (w, h), cv2.INTER_NEAREST, "road_mask")
                selected = mask > 0
                layer = image.copy()
                layer[selected] = (255, 120, 30)
                cv2.addWeighted(layer, 0.28, image, 0.72, 0, dst=image)
        except Exception as exc:
            _append_unique(skipped_layers, "road_mask")
            _append_unique(warnings, f"road_mask debug overlay was skipped: {exc}")
    if skeleton is not None:
        try:
            skel = _optional_layer_to_gray(skeleton, "skeleton", skipped_layers, warnings)
            if skel is not None:
                if skel.shape[:2] != (h, w):
                    skel = _checked_resize(skel, (w, h), cv2.INTER_NEAREST, "skeleton")
                selected = cv2.dilate(
                    (skel > 0).astype(np.uint8), np.ones((3, 3), np.uint8)
                ) > 0
                image[selected] = (40, 255, 80)
        except Exception as exc:
            _append_unique(skipped_layers, "skeleton")
            _append_unique(warnings, f"skeleton debug overlay was skipped: {exc}")
    return skipped_layers, warnings


def _calibration_info(calibration):
    if calibration is None:
        return False, None, None
    valid = getattr(calibration, "is_valid", False)
    if callable(valid):
        valid = valid()
    converter = getattr(calibration, "pixel_to_wgs84", None) or getattr(calibration, "pixel_to_lonlat", None)
    resolution = getattr(calibration, "pixel_resolution_estimated_m", None)
    try:
        resolution = float(resolution) if resolution else None
    except (TypeError, ValueError):
        resolution = None
    return bool(valid and callable(converter)), converter, resolution


def _draw_map_furniture(
    image, *, project_name, timestamp, calibrated, resolution, scale=1.0,
    track_waypoint_count: int = 0,
):
    h, w = image.shape[:2]
    title = project_name or "RoadNet Studio Competition Road Network"
    _draw_text(image, title, (int(24 * scale), int(38 * scale)), scale, 0.85)
    _draw_text(image, timestamp, (int(24 * scale), int(65 * scale)), scale, 0.52)

    # 坐标系徽章：明确标注比赛使用 WGS84
    crs_label = "CRS: WGS84 (EPSG:4326)  lon/lat" if calibrated else "CRS: image_pixel (no WGS84 yet)"
    _draw_text(image, crs_label, (int(24 * scale), int(90 * scale)), scale, 0.58)

    # 图例
    x0, y0 = int(22 * scale), int(110 * scale)
    box_w, box_h = int(300 * scale), int(185 * scale)
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, dst=image)
    entries = [
        ("Final graph", (40, 220, 255)),
        ("Planned path", (220, 70, 230)),
        ("Track waypoints", (70, 210, 255)),
        ("START", (65, 210, 80)),
        ("GOAL", (55, 65, 245)),
        ("Task waypoint", (240, 145, 40)),
    ]
    for index, (label, color) in enumerate(entries):
        y = y0 + int((24 + index * 25) * scale)
        cv2.line(image, (x0 + int(14 * scale), y), (x0 + int(45 * scale), y), color, max(3, int(5 * scale)), cv2.LINE_AA)
        _draw_text(image, label, (x0 + int(58 * scale), y + int(6 * scale)), scale, 0.48)

    if track_waypoint_count > 0:
        _draw_text(
            image,
            f"Track pts: {track_waypoint_count}  (WGS84 when calibrated)",
            (x0 + int(14 * scale), y0 + box_h - int(14 * scale)),
            scale, 0.42,
        )

    if calibrated and resolution and resolution > 0:
        bar_m = 100 if (100 / resolution) <= w * 0.35 else 50
        bar_px = max(1, int(round(bar_m / resolution)))
        sx, sy = int(35 * scale), h - int(42 * scale)
        cv2.line(image, (sx, sy), (sx + bar_px, sy), (255, 255, 255), max(3, int(5 * scale)), cv2.LINE_AA)
        cv2.line(image, (sx, sy - int(8 * scale)), (sx, sy + int(8 * scale)), (255, 255, 255), max(2, int(3 * scale)))
        cv2.line(image, (sx + bar_px, sy - int(8 * scale)), (sx + bar_px, sy + int(8 * scale)), (255, 255, 255), max(2, int(3 * scale)))
        _draw_text(image, f"{bar_m} m", (sx, sy - int(14 * scale)), scale, 0.55)

    if calibrated:
        # 有标定时显示北箭头；方向为影像上方，避免无标定时作错误暗示。
        nx, ny = w - int(65 * scale), int(95 * scale)
        cv2.arrowedLine(image, (nx, ny + int(55 * scale)), (nx, ny), (255, 255, 255), max(2, int(4 * scale)), cv2.LINE_AA, tipLength=0.28)
        _draw_text(image, "N", (nx - int(10 * scale), ny - int(12 * scale)), scale, 0.75)
        _draw_text(image, "WGS84", (w - int(120 * scale), int(40 * scale)), scale, 0.70)


def _write_image(path, image, params=()):
    extension = os.path.splitext(path)[1].lower()
    ok, encoded = cv2.imencode(extension, image, list(params))
    if not ok:
        raise SubmissionExportError(f"图像编码失败：{path}")
    encoded.tofile(path)


def _haversine(a, b):
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371008.8 * math.asin(min(1.0, math.sqrt(value)))


def export_competition_submission(
    output_dir: str,
    image_rgb: np.ndarray,
    nodes: Sequence[dict],
    edges: Sequence[dict],
    *,
    planned_path_pixel: Sequence[Sequence[float]] = (),
    sparse_waypoints=None,
    vehicle_waypoints=None,
    waypoint_validation_report=None,
    waypoint_bad_segments=None,
    task_points: Iterable = (),
    snapped_points: Iterable = (),
    road_mask=None,
    skeleton=None,
    geo_calibration=None,
    image_path: str = "",
    project_name: str = "",
    arrow_spacing_px: float = 80.0,
    arrow_size_px: float = 12.0,
    default_altitude_m: float = 21.741,
) -> dict:
    if not nodes and not edges:
        raise SubmissionExportError("final_graph 为空，无法导出比赛路网图")
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)
    base = image_input_to_bgr(image_rgb, "image_rgb", numpy_color_order="rgb")
    if base.ndim != 3 or base.shape[2] != 3:
        raise SubmissionExportError(
            f"原始影像必须是三通道彩色图：{_array_diagnostics(base, 'image_rgb')}"
        )
    base = cv2.convertScaleAbs(base, alpha=0.85, beta=8)
    h, w = base.shape[:2]
    draw_scale = max(1.0, min(3.0, max(w, h) / 2500.0))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    calibrated, converter, resolution = _calibration_info(geo_calibration)
    path = [[float(p[0]), float(p[1])] for p in (planned_path_pixel or []) if p is not None and len(p) >= 2]
    task_points = list(task_points or [])
    snapped_points = list(snapped_points or [])

    # Prefer validated vehicle waypoints for overlay; fall back to sparse / sampled
    official_wps = list(vehicle_waypoints or [])
    if not official_wps and sparse_waypoints:
        # Accept full waypoint dicts that already carry lon/lat
        for wp in sparse_waypoints:
            if isinstance(wp, dict) and (
                wp.get("latitude_deg") is not None or wp.get("latitude") is not None
            ):
                official_wps.append(dict(wp))
    track_waypoints = _normalize_track_waypoints(official_wps or sparse_waypoints)
    if not track_waypoints and len(path) >= 2:
        spacing_px = 40.0
        if resolution and resolution > 0:
            spacing_px = max(12.0, 10.0 / float(resolution))
        track_waypoints = _sample_track_waypoints_from_path(path, spacing_px=spacing_px)
    if calibrated and callable(converter):
        for wp in track_waypoints:
            if wp.get("lon") is not None and wp.get("lat") is not None:
                continue
            try:
                lon, lat = converter(wp["x"], wp["y"])
                wp["lon"], wp["lat"] = float(lon), float(lat)
            except Exception:
                pass

    clean = base.copy()
    _draw_graph(clean, nodes, edges, scale=draw_scale)
    _draw_path(clean, path, spacing=arrow_spacing_px, arrow_size=arrow_size_px, scale=draw_scale)
    _draw_track_waypoints(
        clean, track_waypoints, converter=converter, debug=False, scale=draw_scale,
        label_every=max(3, min(8, len(track_waypoints) // 8 or 1)),
    )
    _draw_markers(clean, task_points, snapped_points, scale=draw_scale)
    _draw_map_furniture(
        clean, project_name=project_name, timestamp=timestamp,
        calibrated=calibrated, resolution=resolution, scale=draw_scale,
        track_waypoint_count=len(track_waypoints),
    )
    _write_image(os.path.join(output_dir, SUBMISSION_CORE_FILES[0]), clean)
    _write_image(os.path.join(output_dir, SUBMISSION_CORE_FILES[1]), clean, (cv2.IMWRITE_JPEG_QUALITY, 95))
    del clean

    skipped_layers = []
    export_warnings = []
    debug = base.copy()
    _overlay_debug_layers(
        debug, road_mask, skeleton,
        skipped_layers=skipped_layers, warnings=export_warnings,
    )
    _draw_graph(debug, nodes, edges, debug=True, scale=draw_scale)
    _draw_path(debug, path, spacing=arrow_spacing_px, arrow_size=arrow_size_px, scale=draw_scale)
    _draw_track_waypoints(
        debug, track_waypoints, converter=converter, debug=True, scale=draw_scale,
        label_every=1,
    )
    _draw_markers(debug, task_points, snapped_points, debug=True, scale=draw_scale)
    _draw_map_furniture(
        debug, project_name=(project_name or "RoadNet") + " · DEBUG",
        timestamp=timestamp, calibrated=calibrated, resolution=resolution,
        scale=draw_scale, track_waypoint_count=len(track_waypoints),
    )
    _write_image(os.path.join(output_dir, SUBMISSION_CORE_FILES[2]), debug)
    del debug

    graph = {
        "coordinate_system": "image_pixel",
        "geodetic_crs": "WGS84" if calibrated else None,
        "geodetic_epsg": "EPSG:4326" if calibrated else None,
        "metadata": {"image_width": w, "image_height": h,
                     "node_count": len(nodes), "edge_count": len(edges)},
        "nodes": [
            {
                "id": _native(node.get("id")),
                "x_pixel": float(node.get("x", node.get("x_pixel", 0))),
                "y_pixel": float(node.get("y", node.get("y_pixel", 0))),
                "type": str(node.get("type", "")),
                "source": str(node.get("source", "auto")),
            }
            for node in nodes
        ],
        "edges": [
            {
                "id": _native(edge.get("id")),
                "start": _native(edge.get("start")),
                "end": _native(edge.get("end")),
                "length_pixel": float(edge.get("length_pixel", 0.0)),
                "points_pixel": _native(edge.get("points_pixel", edge.get("polyline", []))),
                "source": str(edge.get("source", "auto")),
                "enabled": bool(edge.get("enabled", True)),
            }
            for edge in edges
        ],
    }
    with open(os.path.join(output_dir, "final_graph.json"), "w", encoding="utf-8") as stream:
        json.dump(graph, stream, ensure_ascii=False, indent=2)

    geo_path = []
    if calibrated and len(path) >= 2:
        for x, y in path:
            lon, lat = converter(x, y)
            geo_path.append([float(lon), float(lat), 0.0])
    with open(os.path.join(output_dir, "global_path_geo.csv"), "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "seq", "longitude_wgs84", "latitude_wgs84", "altitude_m",
            "x_pixel", "y_pixel", "coordinate_system",
        ])
        for seq, pixel in enumerate(path, 1):
            if seq <= len(geo_path):
                geo = geo_path[seq - 1]
                writer.writerow([
                    seq, geo[0], geo[1], geo[2], pixel[0], pixel[1], "WGS84/EPSG:4326",
                ])
            else:
                writer.writerow([
                    seq, "", "", 0.0, pixel[0], pixel[1], "image_pixel",
                ])

    # 导出可见航迹点清单（与叠加图上的点一致；仅调试，非正式小车输入）
    with open(os.path.join(output_dir, "track_waypoints_overlay.csv"), "w",
              encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "seq", "longitude_wgs84", "latitude_wgs84", "altitude_m",
            "x_pixel", "y_pixel", "coordinate_system",
        ])
        for wp in track_waypoints:
            if wp.get("lon") is not None and wp.get("lat") is not None:
                writer.writerow([
                    wp["seq"], wp["lon"], wp["lat"], 0.0,
                    wp["x"], wp["y"], "WGS84/EPSG:4326",
                ])
            else:
                writer.writerow([
                    wp["seq"], "", "", 0.0, wp["x"], wp["y"], "image_pixel",
                ])

    # ── Official vehicle waypoints via unified exporter (NOT global_path_geo) ──
    from roadnet.path_export import write_official_vehicle_waypoint_bundle

    val_report = dict(waypoint_validation_report or {})
    # If caller only passed sparse overlay points without a validation report,
    # do NOT invent an old-format waypoints.yaml from geo_path.
    if official_wps and not val_report:
        export_warnings.append(
            "缺少 waypoint_validation_report：未生成正式 subject1_waypoints.yaml"
        )
    vehicle_bundle = write_official_vehicle_waypoint_bundle(
        output_dir,
        official_wps,
        val_report,
        default_altitude_m=float(default_altitude_m),
        preview_image=image_rgb,
        image_width=w,
        image_height=h,
        bad_segments=list(waypoint_bad_segments or []),
    )
    if not vehicle_bundle.get("yaml_export_valid"):
        reason = vehicle_bundle.get("block_reason") or "vehicle_waypoints 未通过验收"
        if not official_wps:
            reason = "无已验收的 vehicle_waypoints（禁止用 global_path_geo.csv 生成旧 waypoints.yaml）"
        # Only warn when a planned path exists (vehicle YAML is expected) or
        # waypoints were supplied but failed the gate.
        if len(path) >= 2 or bool(vehicle_waypoints) or bool(waypoint_validation_report):
            export_warnings.append(f"未生成 subject1_waypoints.yaml，原因：{reason}")

    path_length_m = 0.0
    if len(geo_path) >= 2:
        path_length_m = sum(_haversine(geo_path[i], geo_path[i + 1]) for i in range(len(geo_path) - 1))
    elif resolution and len(path) >= 2:
        path_length_m = resolution * sum(
            math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            for i in range(len(path) - 1)
        )

    exported_files = list(SUBMISSION_CORE_FILES) + ["track_waypoints_overlay.csv"]
    exported_files.extend(vehicle_bundle.get("written_files") or [])
    # de-dup preserve order
    seen = set()
    exported_files = [f for f in exported_files if not (f in seen or seen.add(f))]

    report = {
        "image": os.path.abspath(image_path) if image_path else "",
        "project_name": project_name,
        "export_time": timestamp,
        "has_final_graph": bool(nodes or edges),
        "has_planned_path": len(path) >= 2,
        "geo_calibrated": calibrated,
        "coordinate_system": "WGS84/EPSG:4326" if calibrated else "image_pixel",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "task_point_count": len(task_points),
        "path_point_count": len(path),
        "track_waypoint_count": len(track_waypoints),
        "path_length_m": round(path_length_m, 3),
        "exported_files": exported_files,
        "skipped_layers": skipped_layers,
        "warnings": export_warnings,
        # Official vehicle YAML fields
        "official_vehicle_yaml": vehicle_bundle.get("official_vehicle_yaml"),
        "yaml_export_valid": bool(vehicle_bundle.get("yaml_export_valid")),
        "vehicle_waypoint_count": int(vehicle_bundle.get("vehicle_waypoint_count") or 0),
        "average_spacing_m": vehicle_bundle.get("average_spacing_m"),
        "max_spacing_m": vehicle_bundle.get("max_spacing_m"),
        "geometry_valid": bool(vehicle_bundle.get("geometry_valid")),
        "export_valid": bool(vehicle_bundle.get("export_valid")),
        "bad_segment_count": int(vehicle_bundle.get("bad_segment_count") or 0),
        "los_failed_count": int(vehicle_bundle.get("los_failed_count") or 0),
        "duplicate_count": int(vehicle_bundle.get("duplicate_count") or 0),
        "aba_backtrack_count": int(vehicle_bundle.get("aba_backtrack_count") or 0),
        "subject1_block_reason": vehicle_bundle.get("block_reason"),
    }
    with open(os.path.join(output_dir, "submission_report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return {
        "output_dir": output_dir,
        "exported_files": exported_files,
        "report": report,
        "track_waypoints": track_waypoints,
        "vehicle_bundle": vehicle_bundle,
        "yaml_export_valid": bool(vehicle_bundle.get("yaml_export_valid")),
        "official_vehicle_yaml": vehicle_bundle.get("official_vehicle_yaml"),
    }
