"""
统一区域数据结构：ROI / Ignore 多边形。

所有多边形统一保存为 image_pixel 坐标。
支持旧矩形 → 四点多边形兼容转换。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PolygonRegion:
    """统一区域数据结构。"""
    id: str                           # "roi_001" / "ignore_001"
    region_type: str                  # "roi" 或 "ignore"
    points: List[List[float]] = field(default_factory=list)  # [[x,y], ...] image_pixel 坐标
    enabled: bool = True
    name: str = ""


def polygon_list_to_regions(
    polygon_list: List[List[List[float]]],
    region_type: str = "roi",
) -> List[PolygonRegion]:
    """将 [[[x,y],...], ...] 点列表转换为 PolygonRegion 列表。"""
    regions = []
    for i, pts in enumerate(polygon_list):
        rid = f"{region_type}_{i+1:03d}"
        regions.append(PolygonRegion(
            id=rid,
            region_type=region_type,
            points=[[float(p[0]), float(p[1])] for p in pts],
            enabled=True,
            name=f"{region_type.upper()} 区域 {i+1}",
        ))
    return regions


def regions_to_dict_list(regions: List[PolygonRegion]) -> List[Dict]:
    """把 PolygonRegion 列表转为可 JSON 序列化的字典列表。"""
    out = []
    for r in regions:
        out.append({
            "id": r.id,
            "region_type": r.region_type,
            "points": [[float(p[0]), float(p[1])] for p in r.points],
            "enabled": r.enabled,
            "name": r.name,
        })
    return out


def regions_from_dict_list(data: List[Dict]) -> List[PolygonRegion]:
    """从字典列表恢复 PolygonRegion 列表。"""
    regions = []
    for d in data:
        pts = []
        for p in d.get("points", []):
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append([float(p[0]), float(p[1])])
        regions.append(PolygonRegion(
            id=d.get("id", ""),
            region_type=d.get("region_type", "roi"),
            points=pts,
            enabled=d.get("enabled", True),
            name=d.get("name", ""),
        ))
    return regions


def rect_to_polygon_points(x: float, y: float, w: float, h: float) -> List[List[float]]:
    """将矩形 [x, y, w, h] 转换为四点多边形 [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]。"""
    return [
        [float(x), float(y)],
        [float(x + w), float(y)],
        [float(x + w), float(y + h)],
        [float(x), float(y + h)],
    ]


def save_regions(
    output_dir: str,
    roi_polygons: List[List[List[float]]] = None,
    ignore_polygons: List[List[List[float]]] = None,
    ignore_rects: List[List[float]] = None,
    filename: str = "regions.json",
) -> str:
    """保存 ROI 和 Ignore 区域到 regions.json。

    Args:
        output_dir: 输出目录
        roi_polygons: [[[x,y],...], ...] ROI 多边形列表
        ignore_polygons: [[[x,y],...], ...] Ignore 多边形列表
        ignore_rects: [[x,y,w,h], ...] 旧版矩形（自动转多边形）
        filename: 文件名

    Returns:
        保存的文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    regions_data = []

    # ROI → regions
    if roi_polygons:
        regions_data.extend(regions_to_dict_list(
            polygon_list_to_regions(roi_polygons, "roi")
        ))

    # Ignore polygons → regions
    if ignore_polygons:
        regions_data.extend(regions_to_dict_list(
            polygon_list_to_regions(ignore_polygons, "ignore")
        ))

    # ★ 兼容：旧矩形 → 四点多边形
    if ignore_rects:
        for i, r in enumerate(ignore_rects):
            if len(r) >= 4:
                pts = rect_to_polygon_points(r[0], r[1], r[2], r[3])
                rid = f"ignore_rect_{i+1:03d}"
                regions_data.append({
                    "id": rid,
                    "region_type": "ignore",
                    "points": pts,
                    "enabled": True,
                    "name": f"Ignore 矩形 {i+1}",
                })

    output = {
        "coordinate_system": "image_pixel",
        "regions": regions_data,
    }

    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[Regions] 已保存: {filepath} ({len(regions_data)} 个区域)")
    return filepath


def load_regions(filepath: str) -> List[PolygonRegion]:
    """从 regions.json 加载区域列表。

    Returns:
        PolygonRegion 列表
    """
    if not os.path.exists(filepath):
        print(f"[Regions] 文件不存在: {filepath}")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    regions_data = data.get("regions", [])
    return regions_from_dict_list(regions_data)
