"""
坐标校准模块

实现三点 GCP 仿射配准 + pyproj / 局部 ENU 坐标转换。
支持:
- pyproj UTM 模式 (优先)
- 局部 ENU 近似模式 (fallback)
- 像素坐标系 <-> 平面真实坐标 <-> 经纬度
"""

from __future__ import annotations

import json
import math
import os
import csv
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# 常量
# ============================================================================
EARTH_RADIUS_M = 6378137.0          # WGS84 椭球长半轴 (m)
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi


# ============================================================================
# 坐标变换工具
# ============================================================================

def _try_pyproj_setup(lon0: float, lat0: float) -> Optional[Dict[str, Any]]:
    """尝试使用 pyproj 建立 WGS84 <-> UTM 转换。

    返回 dict 包含:
        mode: "pyproj_utm"
        lon0, lat0: 均值原点
        crs_wgs84: "EPSG:4326"
        crs_projected: e.g. "EPSG:32649"
        transformer_to_xy: pyproj.Transformer
        transformer_to_lonlat: pyproj.Transformer
    如果 pyproj 不可用，返回 None。
    """
    try:
        import pyproj
    except ImportError:
        return None

    # 根据 lon0 计算 UTM zone
    zone = int((lon0 + 180) / 6) + 1
    if lat0 >= 0:
        epsg_projected = 32600 + zone  # 北半球
    else:
        epsg_projected = 32700 + zone  # 南半球

    crs_wgs84 = "EPSG:4326"
    crs_projected = f"EPSG:{epsg_projected}"

    try:
        transformer_to_xy = pyproj.Transformer.from_crs(
            crs_wgs84, crs_projected, always_xy=True
        )
        transformer_to_lonlat = pyproj.Transformer.from_crs(
            crs_projected, crs_wgs84, always_xy=True
        )
        # 测试转换
        x_test, y_test = transformer_to_xy.transform(lon0, lat0)
        return {
            "mode": "pyproj_utm",
            "lon0": lon0,
            "lat0": lat0,
            "crs_wgs84": crs_wgs84,
            "crs_projected": crs_projected,
            "transformer_to_xy": transformer_to_xy,
            "transformer_to_lonlat": transformer_to_lonlat,
        }
    except Exception:
        return None


def _lonlat_to_local_enu(lon: float, lat: float, lon0: float, lat0: float) -> Tuple[float, float]:
    """使用局部 ENU 近似: WGS84 经纬度 -> 局部平面 x, y (米)。"""
    lat0_rad = lat0 * DEG_TO_RAD
    cos_lat0 = math.cos(lat0_rad)
    x = EARTH_RADIUS_M * cos_lat0 * (lon - lon0) * DEG_TO_RAD
    y = EARTH_RADIUS_M * (lat - lat0) * DEG_TO_RAD
    return x, y


def _local_enu_to_lonlat(x: float, y: float, lon0: float, lat0: float) -> Tuple[float, float]:
    """局部 ENU 近似: 平面 x, y (米) -> WGS84 经纬度。"""
    lat0_rad = lat0 * DEG_TO_RAD
    cos_lat0 = math.cos(lat0_rad)
    lon = x / (EARTH_RADIUS_M * cos_lat0) * RAD_TO_DEG + lon0
    lat = y / EARTH_RADIUS_M * RAD_TO_DEG + lat0
    return lon, lat


def _compute_affine_3x3(pixel_pts: np.ndarray, world_pts: np.ndarray) -> Optional[np.ndarray]:
    """通过 3+ 个控制点计算 pixel -> world 的 3x3 仿射矩阵。

    Args:
        pixel_pts: (N, 2) 像素坐标 [u, v]
        world_pts: (N, 2) 世界坐标 [x_meter, y_meter]

    Returns:
        3x3 仿射矩阵 [[a,b,c],[d,e,f],[0,0,1]], 或 None
    """
    n = len(pixel_pts)
    if n < 3:
        return None

    # 构造 A * params = b
    # params = [a, b, c, d, e, f]
    A = np.zeros((2 * n, 6), dtype=np.float64)
    b = np.zeros((2 * n,), dtype=np.float64)

    for i in range(n):
        u, v = pixel_pts[i]
        x, y = world_pts[i]
        A[2 * i, 0] = u
        A[2 * i, 1] = v
        A[2 * i, 2] = 1.0
        A[2 * i + 1, 3] = u
        A[2 * i + 1, 4] = v
        A[2 * i + 1, 5] = 1.0
        b[2 * i] = x
        b[2 * i + 1] = y

    if n == 3:
        try:
            params = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
    else:
        params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    a, b_val, c, d, e, f = params

    matrix = np.array([
        [a, b_val, c],
        [d, e, f],
        [0, 0, 1],
    ], dtype=np.float64)

    return matrix


def _invert_affine_3x3(M: np.ndarray) -> Optional[np.ndarray]:
    """计算 3x3 仿射矩阵的逆。"""
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return None


def _triangle_area2(p1, p2, p3):
    """计算三点三角形面积的两倍 (2×Area)。"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


def _check_collinear(pts: np.ndarray, eps: float = 1.0) -> bool:
    """检查三个点是否近似共线。

    Args:
        pts: (N, 2) 坐标数组
        eps: 2×面积阈值。像素坐标建议 1.0 (px²)，世界坐标建议 1.0 (m²)
    """
    if len(pts) < 3:
        return False
    u1, v1 = pts[0]
    u2, v2 = pts[1]
    u3, v3 = pts[2]
    area2 = _triangle_area2((u1, v1), (u2, v2), (u3, v3))
    print(f"[DEBUG][GCP] collinear check: area2={area2:.4f}, eps={eps}, collinear={area2 < eps}")
    return area2 < eps


# ============================================================================
# GeoCalibration 主类
# ============================================================================

class GeoCalibration:
    """地理坐标校准"""

    def __init__(self, mode: str = "auto"):
        self.enabled: bool = False
        self.mode: str = mode            # "auto" / "manual"
        self.method: str = ""            # 标定方式: "corner_manual" / "corner_file" / "control_points_file" / "control_points_manual"
        self.transform_mode: Optional[str] = None  # "pyproj_utm" / "local_enu_fallback"
        self.control_points: List[Dict[str, Any]] = []
        self.lon0: Optional[float] = None
        self.lat0: Optional[float] = None
        self.crs_wgs84: str = "EPSG:4326"
        self.crs_projected: Optional[str] = None
        self.transformer_to_xy: Any = None
        self.transformer_to_lonlat: Any = None
        self.pixel_to_world_matrix: Optional[np.ndarray] = None
        self.world_to_pixel_matrix: Optional[np.ndarray] = None
        self.pixel_resolution_estimated_m: Optional[float] = None
        self.rms_error: Optional[float] = None  # RMS 误差 (米)
        self.image_width: int = 0
        self.image_height: int = 0
        # 局部 ENU 缓存
        self._enu_origin: Optional[Tuple[float, float]] = None
        # 三点图片顶点校准扩展元数据（可选）
        self.source_file: str = ""
        self.coordinate_system: str = ""
        self.corner_points: List[Dict[str, Any]] = []
        self.inferred_corners: List[Dict[str, Any]] = []
        self.calibration_mode: str = ""  # e.g. image_corner_3point_affine

    # ===================================================================
    # 控制点设置
    # ===================================================================
    def set_control_points(self, control_points: List[Dict[str, Any]]):
        """设置控制点列表。

        每个控制点格式:
        {
            "name": "top_left" / "manual_1",
            "pixel": [u, v],
            "lon": 105.123,
            "lat": 38.456,
        }

        注意：此方法只设置控制点值，不重新计算投影。
        每次调用 compute_affine 前会通过 setup_projection 自动计算 x_meter/y_meter。
        """
        self.control_points = control_points
        if control_points:
            lons = [cp["lon"] for cp in control_points]
            lats = [cp["lat"] for cp in control_points]
            self.lon0 = sum(lons) / len(lons)
            self.lat0 = sum(lats) / len(lats)
        else:
            self.lon0 = None
            self.lat0 = None

        # ★ 清除所有控制点的旧 x_meter/y_meter，确保不会残留上次计算结果
        for cp in (control_points or []):
            cp.pop("x_meter", None)
            cp.pop("y_meter", None)

    def reset_state(self):
        """重置仿射变换状态，保留控制点数据和标定方式。

        调用时机：每次重新计算前，确保干净的计算环境。
        """
        self.enabled = False
        self.transform_mode = None
        self.crs_projected = None
        self.transformer_to_xy = None
        self.transformer_to_lonlat = None
        self.pixel_to_world_matrix = None
        self.world_to_pixel_matrix = None
        self.pixel_resolution_estimated_m = None
        self.rms_error = None
        self._enu_origin = None
        # ★ 清除控制点中上次计算的 x_meter/y_meter
        for cp in self.control_points:
            cp.pop("x_meter", None)
            cp.pop("y_meter", None)
        # ★ 保留 method 和 image_width/image_height，不清空

    # ===================================================================
    # 投影设置
    # ===================================================================
    def setup_projection(self, force_local_enu: bool = False) -> bool:
        """根据控制点设置投影（优先 pyproj UTM，fallback local ENU）。

        Args:
            force_local_enu: True 时强制使用局部 ENU（三点图片顶点校准推荐）。
        """
        if self.lon0 is None or self.lat0 is None:
            print("[GeoCalibration] 控制点未设置，无法初始化投影。")
            return False

        result = None if force_local_enu else _try_pyproj_setup(self.lon0, self.lat0)
        if result is not None:
            self.transform_mode = result["mode"]
            self.crs_wgs84 = result["crs_wgs84"]
            self.crs_projected = result["crs_projected"]
            self.transformer_to_xy = result["transformer_to_xy"]
            self.transformer_to_lonlat = result["transformer_to_lonlat"]
            self.coordinate_system = self.crs_projected or "pyproj_utm"
            print(f"[GeoCalibration] 使用 pyproj UTM 模式: {self.crs_projected}")
        else:
            # Fallback / 强制: 局部 ENU
            self.transform_mode = "local_enu_fallback"
            self.transformer_to_xy = None
            self.transformer_to_lonlat = None
            self._enu_origin = (self.lon0, self.lat0)
            self.coordinate_system = "WGS84_to_local_ENU"
            print(
                f"[GeoCalibration] 使用局部 ENU 近似 "
                f"(原点: lon={self.lon0:.6f}, lat={self.lat0:.6f})"
                + (" [force]" if force_local_enu else "")
            )

        # 计算每个控制点的 x_meter, y_meter
        for cp in self.control_points:
            x_m, y_m = self.lonlat_to_xy(cp["lon"], cp["lat"])
            cp["x_meter"] = x_m
            cp["y_meter"] = y_m

        return True

    # ===================================================================
    # lon/lat <-> x/y 转换
    # ===================================================================
    def lonlat_to_xy(self, lon: float, lat: float) -> Tuple[float, float]:
        """经纬度 -> 平面坐标 (米)。"""
        if self.transform_mode == "pyproj_utm" and self.transformer_to_xy is not None:
            x, y = self.transformer_to_xy.transform(lon, lat)
            return float(x), float(y)
        # local_enu_fallback
        if self._enu_origin is not None:
            return _lonlat_to_local_enu(lon, lat, self._enu_origin[0], self._enu_origin[1])
        raise RuntimeError("投影未初始化，请先调用 setup_projection()。")

    def xy_to_lonlat(self, x: float, y: float) -> Tuple[float, float]:
        """平面坐标 (米) -> 经纬度。"""
        if self.transform_mode == "pyproj_utm" and self.transformer_to_lonlat is not None:
            lon, lat = self.transformer_to_lonlat.transform(x, y)
            return float(lon), float(lat)
        # local_enu_fallback
        if self._enu_origin is not None:
            return _local_enu_to_lonlat(x, y, self._enu_origin[0], self._enu_origin[1])
        raise RuntimeError("投影未初始化，请先调用 setup_projection()。")

    # ===================================================================
    # 仿射变换
    # ===================================================================
    def compute_affine(self) -> bool:
        """计算 pixel -> world 仿射变换矩阵。"""
        if len(self.control_points) < 3:
            print("[GeoCalibration] 错误：需要至少 3 个控制点。")
            return False

        # 确保投影已初始化
        if self.transform_mode is None:
            if not self.setup_projection():
                return False

        # ──── 调试：打印参与计算的控制点 ────
        print("[DEBUG][GCP] selected control points:")
        for cp in self.control_points:
            px = cp.get("pixel", [None, None])
            print(f"  name={cp.get('name', '?')}, pixel=({px[0]},{px[1]}), "
                  f"lon={cp.get('lon', float('nan')):.6f}, lat={cp.get('lat', float('nan')):.6f}")

        print("[DEBUG][GCP] world xy (after projection):")
        for cp in self.control_points:
            xm = cp.get("x_meter", float('nan'))
            ym = cp.get("y_meter", float('nan'))
            print(f"  {cp.get('name', '?')}: x={xm:.3f}, y={ym:.3f}")

        pixel_pts = np.array([cp["pixel"] for cp in self.control_points], dtype=np.float64)
        world_pts = np.array([[cp["x_meter"], cp["y_meter"]] for cp in self.control_points], dtype=np.float64)

        # 检查控制点像素是否为 None
        if np.any(np.isnan(pixel_pts)) or np.any(np.isinf(pixel_pts)):
            print("[GeoCalibration] 错误：控制点像素坐标包含无效值 (NaN/Inf)。")
            raise ValueError("控制点像素坐标未设置，请检查顶点选择逻辑。")

        # 检查共线性 — 像素阈值 1.0 px², 世界阈值 1.0 m²
        if _check_collinear(pixel_pts, eps=1.0):
            print("[GeoCalibration] 错误：像素控制点共线，无法计算仿射变换。")
            raise ValueError("三个像素控制点近似共线，无法计算仿射变换。")
        if _check_collinear(world_pts, eps=1.0):
            print("[GeoCalibration] 错误：世界控制点共线，无法计算仿射变换。")
            raise ValueError("三个世界控制点近似共线，无法计算仿射变换。")

        M = _compute_affine_3x3(pixel_pts, world_pts)
        if M is None:
            print("[GeoCalibration] 错误：仿射矩阵计算失败。")
            return False

        M_inv = _invert_affine_3x3(M)
        if M_inv is None:
            print("[GeoCalibration] 错误：仿射逆矩阵计算失败。")
            return False

        self.pixel_to_world_matrix = M
        self.world_to_pixel_matrix = M_inv

        # 估计分辨率
        self.estimate_pixel_resolution()

        self.enabled = True
        print(f"[GeoCalibration] 仿射矩阵计算成功，估计分辨率: {self.pixel_resolution_estimated_m:.3f} m/px")
        return True

    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """像素坐标 -> 平面世界坐标 (米)。"""
        if self.pixel_to_world_matrix is None:
            raise RuntimeError("仿射矩阵未计算，请先调用 compute_affine()。")
        vec = np.array([u, v, 1.0], dtype=np.float64)
        result = self.pixel_to_world_matrix @ vec
        return float(result[0]), float(result[1])

    def world_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        """平面世界坐标 (米) -> 像素坐标。"""
        if self.world_to_pixel_matrix is None:
            raise RuntimeError("仿射矩阵未计算，请先调用 compute_affine()。")
        vec = np.array([x, y, 1.0], dtype=np.float64)
        result = self.world_to_pixel_matrix @ vec
        return float(result[0]), float(result[1])

    def pixel_to_lonlat(self, u: float, v: float) -> Tuple[float, float]:
        """像素坐标 -> 经纬度。"""
        x, y = self.pixel_to_world(u, v)
        return self.xy_to_lonlat(x, y)

    def lonlat_to_pixel(self, lon: float, lat: float) -> Tuple[float, float]:
        """经纬度 -> 像素坐标。"""
        if not self.enabled:
            raise RuntimeError("尚未完成坐标校准，无法导入经纬度任务点。")
        x, y = self.lonlat_to_xy(lon, lat)
        return self.world_to_pixel(x, y)

    def wgs84_to_pixel(self, lon: float, lat: float) -> Tuple[float, float]:
        """统一任务点入口：WGS84 (longitude, latitude) -> image pixel。"""
        return self.lonlat_to_pixel(lon, lat)

    def pixel_to_wgs84(self, u: float, v: float) -> Tuple[float, float]:
        """统一反查入口：image pixel -> WGS84 (longitude, latitude)。"""
        return self.pixel_to_lonlat(u, v)

    def estimate_pixel_resolution(self) -> Optional[float]:
        """估计影像分辨率 (m/px)。"""
        if self.pixel_to_world_matrix is None:
            return None
        # 在图像中心计算 1px 位移对应的世界距离
        a = self.pixel_to_world_matrix[0, 0]
        b = self.pixel_to_world_matrix[0, 1]
        d = self.pixel_to_world_matrix[1, 0]
        e = self.pixel_to_world_matrix[1, 1]
        # 取 x 和 y 方向平均
        res_x = math.sqrt(a * a + d * d)
        res_y = math.sqrt(b * b + e * e)
        self.pixel_resolution_estimated_m = round((res_x + res_y) / 2.0, 4)
        return self.pixel_resolution_estimated_m

    # ===================================================================
    # 控制点验证
    # ===================================================================
    def validate_control_points(self) -> Tuple[bool, str]:
        """验证控制点数据是否合法。

        Returns:
            (is_valid, error_message)
        """
        if len(self.control_points) < 3:
            return False, "控制点数量不足：需要至少 3 个点。"

        for i, cp in enumerate(self.control_points):
            if "pixel" not in cp or len(cp["pixel"]) != 2:
                return False, f"控制点 {i+1} 缺少有效的 pixel 坐标。"
            if "lon" not in cp or "lat" not in cp:
                return False, f"控制点 {i+1} 缺少经纬度。"
            lon, lat = cp["lon"], cp["lat"]
            if not (-180 <= lon <= 180):
                return False, f"控制点 {i+1} 经度 {lon} 超出范围 [-180, 180]。"
            if not (-90 <= lat <= 90):
                return False, f"控制点 {i+1} 纬度 {lat} 超出范围 [-90, 90]。"

        # 检查像素点共线
        pixel_pts = np.array([cp["pixel"] for cp in self.control_points[:3]], dtype=np.float64)
        if _check_collinear(pixel_pts, eps=1.0):
            return False, "三个像素控制点近似共线，无法计算仿射变换。"

        # 检查世界点共线（仅在 setup_projection 之后、x_meter/y_meter 已计算时检查）
        has_world_coords = all(
            ("x_meter" in cp and "y_meter" in cp) for cp in self.control_points[:3]
        )
        if self.transform_mode is not None and has_world_coords:
            world_pts = np.array([[cp["x_meter"], cp["y_meter"]] for cp in self.control_points[:3]],
                                  dtype=np.float64)
            if _check_collinear(world_pts, eps=1.0):
                return False, "三个世界控制点近似共线，无法计算仿射变换。"

        return True, ""

    def check_resolution(self) -> Tuple[bool, str]:
        """检查估计分辨率是否合理。

        Returns:
            (is_reasonable, warning_message)
        """
        if self.pixel_resolution_estimated_m is None:
            return True, ""
        res = self.pixel_resolution_estimated_m
        if res < 0.1:
            return False, f"估计分辨率 {res:.4f} m/px 过小（< 0.1 m/px），请检查顶点坐标或点位顺序。"
        if res > 5.0:
            return False, f"估计分辨率 {res:.4f} m/px 过大（> 5.0 m/px），请检查顶点坐标或点位顺序。"
        return True, ""

    # ===================================================================
    # 应用到路网
    # ===================================================================
    def apply_to_graph(self, graph_editor) -> Dict[str, Any]:
        """将坐标校准结果应用到 graph_editor (GraphEditorQt 实例) 中的路网。

        Returns:
            包含校准后图形数据的字典: {"nodes": [...], "edges": [...]}
        """
        if not self.enabled or self.pixel_to_world_matrix is None:
            raise RuntimeError("校准未完成，无法应用到路网。")

        # 校准节点
        calibrated_nodes = []
        for n in graph_editor.nodes:
            nx, ny = n["x"], n["y"]
            x_m, y_m = self.pixel_to_world(nx, ny)
            lon, lat = self.pixel_to_lonlat(nx, ny)
            calibrated_nodes.append({
                **n,
                "x_meter": round(x_m, 3),
                "y_meter": round(y_m, 3),
                "lon": round(lon, 8),
                "lat": round(lat, 8),
            })

        # 校准边
        calibrated_edges = []
        for e in graph_editor.edges:
            pts_pixel = e.get("points_pixel", [])
            pts_meter = []
            pts_lonlat = []
            for pu, pv in pts_pixel:
                x_m, y_m = self.pixel_to_world(pu, pv)
                lon, lat = self.pixel_to_lonlat(pu, pv)
                pts_meter.append([round(x_m, 3), round(y_m, 3)])
                pts_lonlat.append([round(lon, 8), round(lat, 8)])

            # 计算米制长度
            length_m = self._compute_path_length_meter(pts_meter)

            calibrated_edges.append({
                **e,
                "points_meter": pts_meter,
                "points_lonlat": pts_lonlat,
                "length_meter": round(length_m, 3),
            })

        return {"nodes": calibrated_nodes, "edges": calibrated_edges}

    @staticmethod
    def _compute_path_length_meter(points_meter: List[List[float]]) -> float:
        """计算路径总长度 (米)。"""
        total = 0.0
        for i in range(len(points_meter) - 1):
            dx = points_meter[i + 1][0] - points_meter[i][0]
            dy = points_meter[i + 1][1] - points_meter[i][1]
            total += math.sqrt(dx * dx + dy * dy)
        return total

    # ===================================================================
    # 序列化
    # ===================================================================
    def to_dict(self) -> Dict[str, Any]:
        """序列化为统一格式的 calibration.json 字典。

        统一格式:
        {
          "is_valid": true/false,
          "method": "corner_manual"|"corner_file"|"control_points_file"|"control_points_manual",
          "image_width": ...,
          "image_height": ...,
          "origin_lon": ...,
          "origin_lat": ...,
          "pixel_to_enu_matrix": ...,
          "enu_to_pixel_matrix": ...,
          "control_points": [...],
          "rms_error_px": ...
        }
        """
        data: Dict[str, Any] = {
            "is_valid": self.is_valid,
            "method": self.method or "",
            "mode": self.calibration_mode or self.method or "",
            "source_file": self.source_file or "",
            "coordinate_system": self.coordinate_system or "",
            "image_width": self.image_width,
            "image_height": self.image_height,
            "enabled": self.enabled,
            "transform_mode": self.transform_mode,
            "control_points": self.control_points,
            "corner_points": self.corner_points or [],
            "inferred_corners": self.inferred_corners or [],
            "lon0": self.lon0,
            "lat0": self.lat0,
            "crs_wgs84": self.crs_wgs84,
            "crs_projected": self.crs_projected,
            "pixel_resolution_estimated_m": self.pixel_resolution_estimated_m,
            "rms_error_px": self.rms_error,
        }

        # 兼容旧字段名
        if self.pixel_to_world_matrix is not None:
            m = self.pixel_to_world_matrix.tolist()
            data["pixel_to_world_matrix"] = m
            data["pixel_to_enu_matrix"] = m
        else:
            data["pixel_to_world_matrix"] = None
            data["pixel_to_enu_matrix"] = None

        if self.world_to_pixel_matrix is not None:
            m = self.world_to_pixel_matrix.tolist()
            data["world_to_pixel_matrix"] = m
            data["enu_to_pixel_matrix"] = m
        else:
            data["world_to_pixel_matrix"] = None
            data["enu_to_pixel_matrix"] = None

        # ★ 生成 corners_wgs84 字典，格式 {name: [lon, lat]}，方便导出/导入
        if self.control_points:
            corners = {}
            for cp in self.control_points:
                name = cp.get("name", "")
                lon = cp.get("lon", 0)
                lat = cp.get("lat", 0)
                if name:
                    corners[name] = [round(float(lon), 8), round(float(lat), 8)]
            if corners:
                data["corners_wgs84"] = corners

        # 兼容字段
        data["origin_lon"] = self.lon0
        data["origin_lat"] = self.lat0

        return data

    def from_dict(self, data: Dict[str, Any]):
        """从字典恢复校准配置。"""
        self.enabled = data.get("enabled", False)
        self.method = data.get("method", "") or data.get("mode", "")
        self.calibration_mode = data.get("mode", "") or data.get("calibration_mode", "")
        self.source_file = data.get("source_file", "") or ""
        self.coordinate_system = data.get("coordinate_system", "") or ""
        self.corner_points = data.get("corner_points", []) or []
        self.inferred_corners = data.get("inferred_corners", []) or []
        self.image_width = data.get("image_width", 0)
        self.image_height = data.get("image_height", 0)
        self.transform_mode = data.get("transform_mode")
        self.control_points = data.get("control_points", [])
        self.lon0 = data.get("lon0") or data.get("origin_lon")
        self.lat0 = data.get("lat0") or data.get("origin_lat")
        self.crs_wgs84 = data.get("crs_wgs84", "EPSG:4326")
        self.crs_projected = data.get("crs_projected")
        self.pixel_resolution_estimated_m = data.get("pixel_resolution_estimated_m")
        self.rms_error = data.get("rms_error_px") or data.get("rms_error")
        # is_valid 由 enabled + matrix 属性决定；若 JSON 标记 valid 但 enabled 缺失则恢复
        if data.get("is_valid") and not self.enabled and (
            data.get("pixel_to_world_matrix") or data.get("pixel_to_enu_matrix")
        ):
            self.enabled = True

        p2w = data.get("pixel_to_world_matrix") or data.get("pixel_to_enu_matrix")
        if p2w is not None:
            self.pixel_to_world_matrix = np.array(p2w, dtype=np.float64)
        else:
            self.pixel_to_world_matrix = None

        w2p = data.get("world_to_pixel_matrix") or data.get("enu_to_pixel_matrix")
        if w2p is not None:
            self.world_to_pixel_matrix = np.array(w2p, dtype=np.float64)
        else:
            self.world_to_pixel_matrix = None

        # 重建投影
        if self.control_points and self.lon0 is not None:
            self.setup_projection()

    def save(self, path: str) -> bool:
        """保存校准配置到 calibration.json。"""
        try:
            data = self.to_dict()
            dirname = os.path.dirname(path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[GeoCalibration] 校准配置已保存: {path}")
            return True
        except Exception as e:
            print(f"[GeoCalibration] 保存失败: {e}")
            return False

    def load(self, path: str) -> bool:
        """从 calibration.json 加载校准配置。"""
        try:
            if not os.path.exists(path):
                return False
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.from_dict(data)
            print(f"[GeoCalibration] 校准配置已加载: {path}")
            return True
        except Exception as e:
            print(f"[GeoCalibration] 加载失败: {e}")
            return False

    def save_calibrated_graph(self, graph_editor, output_dir: str,
                               image_rgb=None, image_size=None) -> str:
        """应用校准并保存 final_graph_geo.{json,csv}。

        重要：此方法从 graph_editor 读取 pixel 坐标，通过 transform 计算经纬度，
        但 **绝不修改 graph_editor 中的原始 pixel 坐标**。
        final_graph.json 仍由 graph_editor 保存为纯 pixel 坐标。

        Args:
            graph_editor: GraphEditorQt 实例（只读，不修改）
            output_dir: 输出目录
            image_rgb: 可选的 RGB 图像（用于 overlay）
            image_size: 可选的图像尺寸 (w, h)

        Returns:
            保存的 json 文件路径
        """
        os.makedirs(output_dir, exist_ok=True)

        # ★ apply_to_graph 是只读操作：它创建新字典，不修改 graph_editor
        calibrated_data = self.apply_to_graph(graph_editor)
        calibrated_nodes = calibrated_data["nodes"]
        calibrated_edges = calibrated_data["edges"]

        w, h = image_size or (graph_editor._image_w, graph_editor._image_h)

        # ---- final_nodes_geo.csv ----
        nodes_path = os.path.join(output_dir, "final_nodes_geo.csv")
        with open(nodes_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "node_id", "x_pixel", "y_pixel", "x_meter", "y_meter",
                "lon", "lat", "type", "source"
            ])
            for n in calibrated_nodes:
                writer.writerow([
                    n["id"], n["x"], n["y"],
                    n.get("x_meter", ""), n.get("y_meter", ""),
                    n.get("lon", ""), n.get("lat", ""),
                    n.get("type", ""), n.get("source", "")
                ])

        # ---- final_edges_geo.csv ----
        edges_path = os.path.join(output_dir, "final_edges_geo.csv")
        with open(edges_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "edge_id", "start_node", "end_node",
                "length_pixel", "length_meter",
                "source", "enabled", "point_count"
            ])
            for e in calibrated_edges:
                writer.writerow([
                    e["id"], e["start"], e["end"],
                    e.get("length_pixel", 0), e.get("length_meter", ""),
                    e.get("source", ""), e.get("enabled", True),
                    len(e.get("points_pixel", []))
                ])

        # ---- final_graph_geo.json（★ 地理版本，独立于 final_graph.json）----
        # ★ coordinate_system: "wgs84_+_enu" — 同时包含 pixel、meter、lonlat 坐标
        #    pixel 坐标来自 graph_editor（image_pixel），lonlat 由 transform 计算
        graph_path = os.path.join(output_dir, "final_graph_geo.json")
        geo_info = self.to_dict()
        graph = {
            "coordinate_system": "wgs84_+_enu",
            "metadata": {
                "image_width": w,
                "image_height": h,
                "pixel_resolution_m": self.pixel_resolution_estimated_m or 0.5,
                "coordinate_calibrated": True,
                "geo_calibration": geo_info,
                "pixel_resolution_estimated_m": self.pixel_resolution_estimated_m,
                "crs_projected": self.crs_projected,
                "transform_mode": self.transform_mode,
                "node_count": len(calibrated_nodes),
                "edge_count": len(calibrated_edges),
            },
            "nodes": [
                {
                    "id": n["id"], "x_pixel": n["x"], "y_pixel": n["y"],
                    "x_meter": n.get("x_meter"), "y_meter": n.get("y_meter"),
                    "lon": n.get("lon"), "lat": n.get("lat"),
                    "type": n.get("type", ""), "source": n.get("source", "auto")
                }
                for n in calibrated_nodes
            ],
            "edges": [
                {
                    "id": e["id"], "start": e["start"], "end": e["end"],
                    "length_pixel": e.get("length_pixel", 0),
                    "length_meter": e.get("length_meter"),
                    "points_pixel": e.get("points_pixel", []),
                    "points_meter": e.get("points_meter", []),
                    "points_lonlat": e.get("points_lonlat", []),
                    "source": e.get("source", "auto"), "enabled": e.get("enabled", True)
                }
                for e in calibrated_edges
            ],
        }
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)

        # ---- calibration.json ----
        self.save(os.path.join(output_dir, "calibration.json"))

        # ---- overlay ----
        if image_rgb is not None:
            import cv2
            overlay_path = os.path.join(output_dir, "final_graph_calibrated_overlay.png")
            img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            graph_editor._draw_overlay(img)
            cv2.imwrite(overlay_path, img)

        print(f"[GeoCalibration] 已保存校准路网: {len(calibrated_nodes)} 节点, {len(calibrated_edges)} 边 → {output_dir}")
        return graph_path

    # ===================================================================
    # 校准状态判断（统一接口）
    # ===================================================================

    @property
    def is_valid(self) -> bool:
        """统一的地理标定有效性判断。

        标定有效 = enabled=True 且仿射矩阵存在且可逆。
        任务点模块、路径导出、项目保存等都必须通过此属性判断标定状态，
        不允许使用独立的 geo_calibrated / has_geo_reference 等标志。
        """
        return self.enabled and self.pixel_to_world_matrix is not None

    def is_calibrated(self) -> bool:
        """判断当前是否已完成有效标定（兼容旧调用）。"""
        return self.is_valid

    def get_mode_label(self) -> str:
        """返回校准模式标签（用于 UI 显示）。"""
        if not self.is_valid:
            return "未标定"
        # 优先使用 method 字段
        method_labels = {
            "corner_manual": "四角点手动输入",
            "corner_file": "导入四角坐标文件",
            "control_points_file": "导入控制点文件",
            "control_points_manual": "控制点图上配准",
            "image_corner_3point_affine": "三点图片顶点校准",
        }
        key = self.calibration_mode or self.method
        if key in method_labels:
            base = method_labels[key]
        elif self.method in method_labels:
            base = method_labels[self.method]
        else:
            base = "已标定"
        ncps = len(self.control_points)
        if ncps >= 4:
            return f"{base}（四点）"
        elif ncps >= 3:
            return f"{base}（三点）"
        return base

    def get_calibrated_warning(self) -> str:
        """若为三点标定，返回建议提示；否则返回空字符串。"""
        if self.is_valid and len(self.control_points) >= 3 and len(self.control_points) < 4:
            return "当前为三点仿射标定，建议补齐四点提高稳定性。"
        return ""

    def set_calibration_metadata(self, method: str, image_width: int = 0, image_height: int = 0):
        """设置标定元数据（方式 + 图像尺寸）。

        标定方式: "corner_manual" / "corner_file" / "control_points_file" / "control_points_manual"
        """
        self.method = method
        if image_width > 0:
            self.image_width = image_width
        if image_height > 0:
            self.image_height = image_height

    # ===================================================================
    # 任务点转换接口
    # ===================================================================
    def task_lonlat_to_pixel(self, lon: float, lat: float) -> Tuple[float, float]:
        """将任务点经纬度转换为像素坐标。"""
        return self.wgs84_to_pixel(lon, lat)

    def task_pixel_to_lonlat(self, u: float, v: float) -> Tuple[float, float]:
        """将任务点像素坐标转换为经纬度。"""
        return self.pixel_to_wgs84(u, v)
