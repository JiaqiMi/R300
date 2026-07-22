"""
项目管理：项目保存和加载（完整 project.json 格式）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProjectData:
    """项目数据结构 — 完整 project.json 格式"""

    # 基本信息
    project_name: str = ""
    version: str = "2.0"

    # 影像
    image_path: str = ""
    image_width: int = 0      # 预览图宽度
    image_height: int = 0     # 预览图高度
    original_width: int = 0   # 原图宽度（全局像素）
    original_height: int = 0  # 原图高度（全局像素）

    # 大图模式
    large_image_mode: bool = False
    preview_scale: float = 1.0
    large_image_project_path: str = ""
    tile_index_path: str = ""
    global_mask_path: str = ""
    global_graph_path: str = ""

    # 分辨率
    pixel_resolution_m: float = 0.5

    # 当前阶段
    current_stage: str = "import"

    # 采样（统一使用全局像素坐标）
    samples: Dict[str, List] = field(default_factory=lambda: {
        "positive_points": [],  # [{"x_global": x, "y_global": y}, ...] 或 [[x,y], ...]
        "negative_points": [],
    })

    # ROI / Ignore 多边形（统一使用全局像素坐标）
    roi_polygons: List = field(default_factory=list)      # [[[x_global,y_global],...], ...]
    ignore_rects: List = field(default_factory=list)       # [[x_global,y_global,w,h], ...]
    ignore_polygons: List = field(default_factory=list)    # [[[x_global,y_global],...], ...]

    # 文件路径
    mask_files: Dict[str, str] = field(default_factory=dict)       # {"raw": "", "clean": ""}
    skeleton_files: Dict[str, str] = field(default_factory=dict)   # {"raw": "", "optimized": ""}
    draft_graph_file: str = ""
    final_graph_file: str = ""

    # ★ Graph 编辑数据（节点/边）
    graph_nodes: List[Dict] = field(default_factory=list)
    graph_edges: List[Dict] = field(default_factory=list)
    graph_next_node_id: int = 0
    graph_next_edge_id: int = 0

    # ★ 任务点数据（序列化到项目文件）
    task_points_serialized: List[Dict] = field(default_factory=list)

    # 坐标校准
    geo_calibration: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "pixel_resolution_m": 0.5,
        "transform_mode": None,              # "pyproj_utm" / "local_enu_fallback"
        "control_points": [],                # [{"name":"", "pixel":[x,y], "lon":0, "lat":0, "x_meter":0, "y_meter":0}, ...]
        "lon0": None,
        "lat0": None,
        "crs_wgs84": "EPSG:4326",
        "crs_projected": None,
        "pixel_to_world_matrix": None,       # 3x3 list
        "world_to_pixel_matrix": None,       # 3x3 list
        "pixel_resolution_estimated_m": None,
    })

    # 任务点（使用全局像素坐标）
    task_points: List[Dict] = field(default_factory=list)  # [{"x_global": x, "y_global": y, ...}, ...]
    planned_path_file: str = ""

    # GUI 状态
    layer_visibility: Dict[str, bool] = field(default_factory=dict)
    current_tool: str = "pan"
    zoom_level: float = 1.0

    # 兼容旧字段
    config: Dict[str, Any] = field(default_factory=dict)
    output_dir: str = ""


class ProjectManager:
    """项目保存/加载"""

    DEFAULT_EXT = ".roadnet.json"

    def __init__(self):
        self._data = ProjectData()
        self._project_path: str = ""
        self._dirty: bool = False

    @property
    def data(self) -> ProjectData:
        return self._data

    @property
    def project_path(self) -> str:
        return self._project_path

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def mark_dirty(self):
        self._dirty = True

    def mark_clean(self):
        self._dirty = False

    # ===================================================================
    # 保存
    # ===================================================================

    def save(self, path: str = None) -> bool:
        """保存项目到 JSON 文件"""
        if path:
            self._project_path = path
        if not self._project_path:
            return False

        try:
            base = os.path.dirname(self._project_path)
            rel = lambda p: os.path.relpath(p, base) if p and not os.path.isabs(p) else p

            project_dict = {
                "project_name": self._data.project_name,
                "version": self._data.version,
                "image_path": self._data.image_path,
                "image_width": self._data.image_width,
                "image_height": self._data.image_height,
                "original_width": self._data.original_width,
                "original_height": self._data.original_height,
                "large_image_mode": self._data.large_image_mode,
                "preview_scale": self._data.preview_scale,
                "large_image_project_path": self._data.large_image_project_path,
                "tile_index_path": self._data.tile_index_path,
                "global_mask_path": self._data.global_mask_path,
                "global_graph_path": self._data.global_graph_path,
                "pixel_resolution_m": self._data.pixel_resolution_m,
                "current_stage": self._data.current_stage,
                "samples": self._data.samples,
                "roi_polygons": self._data.roi_polygons,
                "ignore_rects": self._data.ignore_rects,
                "ignore_polygons": self._data.ignore_polygons,
                "mask_files": self._data.mask_files,
                "skeleton_files": self._data.skeleton_files,
                "draft_graph_file": self._data.draft_graph_file,
                "final_graph_file": self._data.final_graph_file,
                "geo_calibration": self._data.geo_calibration,
                # Unified task-point payload (serialized TaskPoint dicts).
                # Prefer task_points_serialized; fall back to legacy task_points.
                "task_points": (
                    self._data.task_points_serialized
                    if self._data.task_points_serialized is not None
                    else self._data.task_points
                ),
                "task_points_serialized": self._data.task_points_serialized,
                "planned_path_file": self._data.planned_path_file,
                "layer_visibility": self._data.layer_visibility,
                "current_tool": self._data.current_tool,
                "zoom_level": self._data.zoom_level,
                "config": self._data.config,
                "output_dir": self._data.output_dir,
            }

            with open(self._project_path, "w", encoding="utf-8") as f:
                json.dump(project_dict, f, indent=2, ensure_ascii=False)

            self.mark_clean()
            return True

        except Exception as e:
            print(f"[ERROR] 保存项目失败: {e}")
            return False

    # ===================================================================
    # 加载
    # ===================================================================

    def load(self, path: str) -> Optional[ProjectData]:
        if not os.path.exists(path):
            print(f"[ERROR] 项目文件不存在: {path}")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._data = ProjectData(
                project_name=data.get("project_name", ""),
                version=data.get("version", "2.0"),
                image_path=data.get("image_path", ""),
                image_width=data.get("image_width", 0),
                image_height=data.get("image_height", 0),
                original_width=data.get("original_width", data.get("image_width", 0)),
                original_height=data.get("original_height", data.get("image_height", 0)),
                large_image_mode=data.get("large_image_mode", False),
                preview_scale=data.get("preview_scale", 1.0),
                large_image_project_path=data.get("large_image_project_path", ""),
                tile_index_path=data.get("tile_index_path", ""),
                global_mask_path=data.get("global_mask_path", ""),
                global_graph_path=data.get("global_graph_path", ""),
                pixel_resolution_m=data.get("pixel_resolution_m", 0.5),
                current_stage=data.get("current_stage", "import"),
                samples=data.get("samples", {"positive_points": [], "negative_points": []}),
                roi_polygons=data.get("roi_polygons", []),
                ignore_rects=data.get("ignore_rects", []),
                ignore_polygons=data.get("ignore_polygons", []),
                mask_files=data.get("mask_files", {}),
                skeleton_files=data.get("skeleton_files", {}),
                draft_graph_file=data.get("draft_graph_file", ""),
                final_graph_file=data.get("final_graph_file", ""),
                geo_calibration=data.get("geo_calibration", {
                    "enabled": False, "pixel_resolution_m": 0.5,
                    "transform_mode": None,
                    "control_points": [],
                    "lon0": None, "lat0": None,
                    "crs_wgs84": "EPSG:4326",
                    "crs_projected": None,
                    "pixel_to_world_matrix": None,
                    "world_to_pixel_matrix": None,
                    "pixel_resolution_estimated_m": None,
                }),
                task_points=data.get("task_points", []),
                task_points_serialized=(
                    data.get("task_points_serialized")
                    or data.get("task_points")
                    or []
                ),
                planned_path_file=data.get("planned_path_file", ""),
                layer_visibility=data.get("layer_visibility", {}),
                current_tool=data.get("current_tool", "pan"),
                zoom_level=data.get("zoom_level", 1.0),
                config=data.get("config", {}),
                output_dir=data.get("output_dir", ""),
            )
            self._project_path = path
            self.mark_clean()
            return self._data

        except Exception as e:
            print(f"[ERROR] 加载项目失败: {e}")
            return None

    def new_project(self):
        self._data = ProjectData()
        self._project_path = ""
        self.mark_clean()
