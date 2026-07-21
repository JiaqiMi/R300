"""
图层管理器：管理所有图层数据和显隐状态，并追踪 QGraphicsItem。

统一图层命名：
  - layer_image                : 原始影像
  - layer_sample_points        : 正负样本点（绿点/红叉）
  - layer_roi                  : ROI 区域（蓝色）
  - layer_ignore               : 屏蔽区域（红色）
  - layer_road_mask            : 道路 mask（绿色半透明）
  - layer_preview_segmentation : 快速预览分割 overlay（绿色半透明）
  - layer_skeleton             : 骨架线（黄色细线）
  - layer_skeleton_nodes       : 骨架端点/交叉点（调试）
  - layer_draft_graph          : 草稿路网图（橙色）
  - layer_final_graph          : 最终路网图（蓝色）
  - layer_task_points          : 任务点
  - layer_planned_path         : 规划路径（紫色）
  - layer_debug                : 调试图层

★ 关键：所有 QGraphicsItem 必须通过 add_item() 注册，
  这样 show_layer/hide_layer 才能控制它们的显隐。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtWidgets import QGraphicsItem


# 图层颜色定义
LAYER_COLORS: Dict[str, QColor] = {
    "layer_road_mask":     QColor(80, 250, 123),   # 绿色
    "mask":                QColor(80, 250, 123),   # 兼容别名
    "layer_cleaned_road_mask": QColor(0, 220, 180),  # 青绿 = cleaned mask
    "layer_final_edited_mask": QColor(255, 170, 80),  # 橙黄 = final edited mask
    "layer_preview_segmentation": QColor(80, 250, 123),  # 绿色（同 road_mask）
    "preview_seg_mask":    QColor(80, 250, 123),   # 兼容别名
    "layer_roi":           QColor(68, 153, 255),   # 蓝色
    "roi":                 QColor(68, 153, 255),
    "layer_ignore":        QColor(255, 85, 85),    # 红色
    "ignore":              QColor(255, 85, 85),
    "layer_skeleton":      QColor(241, 250, 140),  # 黄色 = cleaned skeleton
    "skeleton":            QColor(241, 250, 140),
    "layer_raw_skeleton":  QColor(200, 200, 200),  # 灰色 = raw skeleton（噪声多）
    "layer_center_filtered_skeleton": QColor(180, 220, 255),  # 浅蓝 = center filtered
    "layer_skeleton_nodes": QColor(255, 105, 180), # 粉色
    "layer_draft_graph":   QColor(255, 184, 108),  # 橙色
    "draft_graph":         QColor(255, 184, 108),
    "layer_final_graph":   QColor(137, 180, 250),  # 浅蓝
    "final_graph":         QColor(137, 180, 250),
    "layer_reference_graph": QColor(255, 220, 100),  # 金色（参考层）
    "reference_graph":     QColor(255, 220, 100),
    "samroad_raw_graph":   QColor(255, 220, 100),
    "layer_planned_path":  QColor(203, 166, 247),  # 紫色
    "planned_path":        QColor(203, 166, 247),
    "layer_sparse_waypoints": QColor(255, 209, 102),
    "layer_waypoint_validation": QColor(255, 85, 85),
    "layer_sample_points": QColor(200, 200, 200),  # 灰色
    "sample_points":       QColor(200, 200, 200),
    "layer_task_points":   QColor(250, 179, 135),  # 桃色
    "layer_main_road_seed": QColor(255, 0, 255),   # 品红（主路种子线，醒目）
    "layer_road_ribbon_preview": QColor(0, 220, 220),
    "edit":                QColor(245, 194, 66),   # 金色
    "graph":               QColor(137, 180, 250),  # 浅蓝
    "path":                QColor(203, 166, 247),  # 紫色
}

# 图层默认透明度 (0-255)
LAYER_DEFAULT_ALPHA: Dict[str, int] = {
    "layer_road_mask":     115,  # ≈0.45，调试显示下仍可见且不盖死底图
    "mask":                115,
    "layer_cleaned_road_mask": 140,  # cleaned 稍亮一点，便于对比
    "layer_final_edited_mask": 150,
    "layer_preview_segmentation": 90,   # 低于 working Road Mask，避免冒充正式 mask
    "preview_seg_mask":    90,
    "layer_roi":           180,
    "roi":                 180,
    "layer_ignore":        120,
    "ignore":              120,
    "layer_skeleton":      220,          # cleaned skeleton
    "skeleton":            220,
    "layer_raw_skeleton":  160,          # raw skeleton（默认半透明、不抢戏）
    "layer_center_filtered_skeleton": 180,
    "layer_skeleton_nodes": 200,
    "layer_draft_graph":   220,
    "draft_graph":         220,
    "layer_final_graph":   255,
    "final_graph":         255,
    "layer_reference_graph": 160,  # 半透明参考层
    "reference_graph":     160,
    "samroad_raw_graph":   160,
    "layer_planned_path":  240,
    "planned_path":        240,
    "layer_sparse_waypoints": 255,
    "layer_waypoint_validation": 255,
    "layer_sample_points": 255,
    "sample_points":       255,
    "layer_task_points":   255,
    "layer_main_road_seed": 255,
    "layer_road_ribbon_preview": 120,
    "edit":                200,
    "graph":               255,
    "path":                240,
}

# 所有需要显隐管理的矢量图层（不在 LayerManager raster overlay 中管理）
VECTOR_LAYERS = {"layer_sample_points", "sample_points", "layer_roi", "roi",
                 "layer_ignore", "ignore", "layer_task_points",
                 "layer_main_road_seed", "layer_road_ribbon_preview",
                 "layer_skeleton_nodes", "layer_sparse_waypoints",
                 "layer_waypoint_validation",
                 "edit", "graph", "path"}

# ============================================================================
# 阶段预设：每个阶段默认显示/隐藏的图层
# ============================================================================
STAGE_PRESETS: Dict[str, dict] = {
    "import": {
        "show":  [],
        "hide":  ["layer_sample_points", "layer_roi", "layer_ignore",
                  "layer_road_mask", "layer_skeleton", "layer_skeleton_nodes",
                  "layer_draft_graph", "layer_final_graph", "layer_reference_graph",
                  "layer_task_points", "layer_planned_path", "layer_sparse_waypoints"],
        "alias_show": [], "alias_hide": [],
    },
    "segment": {
        "show":  ["layer_sample_points", "layer_road_mask", "layer_preview_segmentation", "layer_roi"],
        "hide":  ["layer_ignore", "layer_skeleton",
                  "layer_skeleton_nodes", "layer_draft_graph", "layer_final_graph",
                  "layer_reference_graph",
                  "layer_task_points", "layer_planned_path", "layer_sparse_waypoints"],
        "alias_show": [], "alias_hide": [],
    },
    "edit": {
        "show":  ["layer_road_mask", "layer_roi", "layer_ignore", "layer_main_road_seed"],
        "hide":  ["layer_sample_points", "layer_skeleton",
                  "layer_cleaned_road_mask", "layer_final_edited_mask",
                  "layer_skeleton_nodes", "layer_draft_graph", "layer_final_graph",
                  "layer_reference_graph",
                  "layer_task_points", "layer_planned_path", "layer_sparse_waypoints"],
        "alias_show": [], "alias_hide": [],
    },
    "skeleton": {
        # 默认显示 Cleaned Skeleton；Raw Skeleton 噪声多，不默认勾选
        "show":  ["layer_skeleton", "layer_road_mask"],
        "hide":  ["layer_sample_points", "layer_roi", "layer_ignore",
                  "layer_cleaned_road_mask", "layer_final_edited_mask",
                  "layer_raw_skeleton", "layer_center_filtered_skeleton",
                  "layer_skeleton_nodes",
                  "layer_draft_graph", "layer_final_graph", "layer_reference_graph",
                  "layer_task_points", "layer_planned_path", "layer_sparse_waypoints"],
        "alias_show": [], "alias_hide": [],
    },
    "graph": {
        "show":  ["layer_skeleton", "layer_draft_graph", "layer_final_graph", "layer_reference_graph",
                  "layer_task_points"],  # ★ task_points 在 graph 阶段也显示
        "hide":  ["layer_sample_points", "layer_roi", "layer_ignore",
                  "layer_road_mask", "layer_center_filtered_skeleton",
                  "layer_raw_skeleton", "layer_skeleton_nodes",
                  "layer_planned_path", "layer_sparse_waypoints"],
        "alias_show": [], "alias_hide": [],
    },
    "calibrate": {
        "show":  ["layer_final_graph", "layer_skeleton",
                  "layer_task_points"],  # ★ task_points 在标定阶段也显示
        "hide":  ["layer_sample_points", "layer_roi", "layer_ignore",
                  "layer_road_mask", "layer_skeleton_nodes",
                  "layer_draft_graph", "layer_reference_graph",
                  "layer_planned_path", "layer_sparse_waypoints"],
        "alias_show": [], "alias_hide": [],
    },
    "export": {
        "show":  ["layer_final_graph", "layer_task_points", "layer_planned_path",
                  "layer_sparse_waypoints"],
        "hide":  ["layer_sample_points", "layer_roi", "layer_ignore",
                  "layer_road_mask", "layer_skeleton", "layer_skeleton_nodes",
                  "layer_draft_graph", "layer_reference_graph"],
        "alias_show": [], "alias_hide": [],
    },
}

# 简洁模式：额外隐藏哪些图层（叠加在阶段预设之上）
CLEAN_MODE_HIDE = [
    "layer_sample_points", "layer_roi", "layer_ignore",
    "layer_road_mask", "layer_skeleton_nodes",
]

# 调试模式：额外显示哪些图层
DEBUG_MODE_SHOW = [
    "layer_sample_points", "layer_roi", "layer_ignore",
    "layer_road_mask", "layer_preview_segmentation",
    "layer_skeleton", "layer_skeleton_nodes",
    "layer_draft_graph", "layer_final_graph",
]

# 清爽显示预设（展示用）
CLEAN_DISPLAY_SHOW = ["layer_skeleton", "layer_draft_graph", "layer_final_graph"]
CLEAN_DISPLAY_HIDE = ["layer_sample_points", "layer_roi", "layer_ignore",
                       "layer_road_mask", "layer_skeleton_nodes",
                       "layer_task_points", "layer_planned_path"]

# 调试显示预设
DEBUG_DISPLAY_SHOW = ["layer_sample_points", "layer_roi", "layer_ignore",
                       "layer_road_mask", "layer_skeleton", "layer_skeleton_nodes",
                       "layer_draft_graph", "layer_final_graph"]
DEBUG_DISPLAY_HIDE = ["layer_task_points", "layer_planned_path"]


# ============================================================================
# 图层名映射：统一新旧命名
# ============================================================================
_LAYER_ALIAS_MAP: Dict[str, str] = {
    # 旧名 → 新名
    "mask":         "layer_road_mask",
    "road_mask":    "layer_road_mask",
    "cleaned_road_mask": "layer_cleaned_road_mask",
    "final_edited_mask": "layer_final_edited_mask",
    "preview_seg_mask": "layer_preview_segmentation",
    "roi":          "layer_roi",
    "ignore":       "layer_ignore",
    "skeleton":     "layer_skeleton",
    "raw_skeleton": "layer_raw_skeleton",
    "center_filtered_skeleton": "layer_center_filtered_skeleton",
    "cleaned_skeleton": "layer_skeleton",
    "draft_graph":  "layer_draft_graph",
    "final_graph":  "layer_final_graph",
    "planned_path": "layer_planned_path",
    "sparse_waypoints": "layer_sparse_waypoints",
    "sample_points":"layer_sample_points",
    "graph":        "layer_final_graph",
    "path":         "layer_planned_path",
    "edit":         "layer_road_mask",  # edit 合并到 mask 层
}

def _resolve_name(name: str) -> str:
    """将旧名映射为新名"""
    return _LAYER_ALIAS_MAP.get(name, name)


@dataclass
class LayerInfo:
    """单层信息"""
    name: str
    visible: bool = True
    data: Optional[np.ndarray] = None       # 二值或彩色数据 (H, W) 或 (H, W, 3/4)
    pixmap: Optional[QPixmap] = None        # 缓存的 QPixmap
    preview_data: Optional[np.ndarray] = None
    opacity: int = 128                       # 0-255
    color: QColor = field(default_factory=lambda: QColor(255, 255, 255))


class LayerManager(QObject):
    """图层管理器"""

    layer_changed = Signal(str)           # 图层数据变化
    visibility_changed = Signal(str, bool) # 图层显隐变化
    mode_changed = Signal(str)            # 模式变化: "clean" / "debug"
    large_image_mode_changed = Signal(bool)  # 大图模式变化

    # 大图检测阈值
    LARGE_IMAGE_THRESHOLD_DIM = 4096
    LARGE_IMAGE_THRESHOLD_PIXELS = 16_000_000
    PREVIEW_MAX_SIZE = 3000

    def __init__(self, parent: QObject = None):
        super().__init__(parent)
        self._image_path: str = ""
        self._image_rgb: Optional[np.ndarray] = None  # 原始全分辨率图像
        self._image_rgb_full: Optional[np.ndarray] = None  # 原始全分辨率图像（别名）
        self._preview_image: Optional[np.ndarray] = None  # 预览图
        self._image_pixmap: Optional[QPixmap] = None
        self._image_size: Tuple[int, int] = (0, 0)

        # ★ 大图模式
        self._large_image_mode: bool = False
        self._preview_scale: float = 1.0
        self._original_width: int = 0
        self._original_height: int = 0
        self._preview_width: int = 0
        self._preview_height: int = 0

        # 当前模式
        self._mode: str = "clean"   # "clean" | "debug"
        self._current_stage: str = "import"

        # ★ QGraphicsItem 追踪：每个图层名下挂载的所有 scene item
        self._items: Dict[str, List[QGraphicsItem]] = {}

        # zValue 映射（大图调试显示建议顺序）
        # background=0, road_mask=20, tile/roi=30, skeleton=50, graph=60,
        # debug points=90, task/path=100
        self._zvalues: Dict[str, int] = {
            "layer_image": 0, "image": 0,
            "layer_preview_segmentation": 10, "preview_seg_mask": 10,
            "layer_road_mask": 20, "mask": 20, "road_mask": 20,
            "layer_cleaned_road_mask": 25,
            "layer_final_edited_mask": 26,
            "layer_sample_points": 25, "sample_points": 25,
            "layer_roi": 30, "roi": 30,
            "layer_ignore": 35, "ignore": 35,
            "layer_main_road_seed": 40,
            "layer_road_ribbon_preview": 38,
            "layer_raw_skeleton": 45,
            "layer_center_filtered_skeleton": 47,
            "layer_skeleton": 50, "skeleton": 50,
            "layer_skeleton_nodes": 90,
            "layer_draft_graph": 60, "draft_graph": 60,
            "layer_final_graph": 60, "final_graph": 60,
            "layer_task_points": 100,
            "layer_planned_path": 100, "planned_path": 100,
            "layer_sparse_waypoints": 110,
            "layer_waypoint_validation": 115,
            "edit": 120, "graph": 60, "path": 100,
        }

        # 初始化所有图层（以新名称）
        self._layers: Dict[str, LayerInfo] = {}
        _all_layer_names = [
            "layer_preview_segmentation",
            "layer_road_mask", "layer_cleaned_road_mask", "layer_final_edited_mask",
            "layer_roi", "layer_ignore",
            "layer_raw_skeleton", "layer_center_filtered_skeleton",
            "layer_skeleton", "layer_skeleton_nodes",
            "layer_draft_graph", "layer_final_graph",
            "layer_planned_path", "layer_sparse_waypoints",
            "layer_waypoint_validation",
            "layer_sample_points", "layer_task_points",
            "layer_main_road_seed", "layer_road_ribbon_preview",
            "edit", "graph", "path",
        ]
        for name in _all_layer_names:
            color = LAYER_COLORS.get(name, QColor(200, 200, 200))
            alpha = LAYER_DEFAULT_ALPHA.get(name, 128)
            self._layers[name] = LayerInfo(
                name=name,
                visible=False,
                color=color,
                opacity=alpha,
            )
            self._items[name] = []

    # ===================================================================
    # 属性
    # ===================================================================

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def current_stage(self) -> str:
        return self._current_stage

    @property
    def image_path(self) -> str:
        return self._image_path

    @property
    def image_rgb(self) -> Optional[np.ndarray]:
        return self._image_rgb

    @property
    def image_pixmap(self) -> Optional[QPixmap]:
        return self._image_pixmap

    @property
    def image_size(self) -> Tuple[int, int]:
        """返回当前显示图像的尺寸（预览图或原图）"""
        return self._image_size

    @property
    def original_size(self) -> Tuple[int, int]:
        """返回原始图像的尺寸（始终是全局像素尺寸）"""
        return (self._original_width, self._original_height)

    @property
    def preview_scale(self) -> float:
        """返回预览图缩放比例"""
        return self._preview_scale

    @property
    def is_large_image_mode(self) -> bool:
        """是否为大图模式"""
        return self._large_image_mode

    @property
    def display_image_rgb(self) -> Optional[np.ndarray]:
        """返回当前用于显示的图像（预览图或原图）"""
        if self._large_image_mode and self._preview_image is not None:
            return self._preview_image
        return self._image_rgb

    @property
    def full_image_rgb(self) -> Optional[np.ndarray]:
        """返回原始全分辨率图像"""
        return self._image_rgb_full

    # ===================================================================
    # 影像
    # ===================================================================

    def load_image(self, path: str) -> np.ndarray:
        """加载影像并生成 QPixmap（大图模式下使用预览图）"""
        from roadnet.large_image_project import (ImageRegionReader, is_large_image,
                                                  estimate_image_memory_mb)

        self._image_path = os.path.abspath(path)
        reader = ImageRegionReader(path)
        w, h = reader.size
        self._original_width = w
        self._original_height = h

        # 检测大图模式（使用统一标准：dim > 4096 或 pixels > 16M）
        self._large_image_mode = is_large_image(w, h)

        if self._large_image_mode:
            # 内存预算检查
            mem = estimate_image_memory_mb(w, h)
            print(f"[LayerManager] 大图模式: {w}x{h}, "
                  f"RGB内存估算={mem['raw_rgb_mb']:.1f}MB, "
                  f"RGBA叠加估算={mem['rgba_overlay_mb']:.1f}MB")

            # The full RGB image is deliberately not decoded or retained.
            self._preview_image = reader.read_preview(self.PREVIEW_MAX_SIZE)
            preview_h, preview_w = self._preview_image.shape[:2]
            self._preview_scale = min(preview_w / w, preview_h / h)
            self._preview_width, self._preview_height = preview_w, preview_h
            self._image_rgb_full = None
            self._image_size = (preview_w, preview_h)
            self._image_rgb = self._preview_image
            display_rgb = self._preview_image
        else:
            # 普通图像：原图即预览图
            original_rgb = reader.read_region(0, 0, w, h)
            self._image_rgb_full = original_rgb
            self._preview_scale = 1.0
            self._preview_width, self._preview_height = w, h
            self._preview_image = None
            self._image_size = (w, h)
            self._image_rgb = original_rgb
            display_rgb = self._image_rgb

        qimg = self._ndarray_to_qimage(display_rgb)
        self._image_pixmap = QPixmap.fromImage(qimg)

        # 清除所有图层缓存
        for layer in self._layers.values():
            layer.data = None
            layer.preview_data = None
            layer.pixmap = None
            layer.visible = False

        self.layer_changed.emit("image")
        self.large_image_mode_changed.emit(self._large_image_mode)

        if self._large_image_mode:
            print(f"[LayerManager] 大图模式已启用: {w}x{h} -> 预览 {preview_w}x{preview_h}, scale={self._preview_scale:.4f}")

        return display_rgb

    def load_large_image_preview(self, image_path: str, preview_path: str,
                                 original_size: Tuple[int, int],
                                 preview_scale: float) -> np.ndarray:
        """Load an existing large-image project preview without decoding source RGB."""
        from roadnet.large_image_project import ImageRegionReader
        preview_reader = ImageRegionReader(preview_path)
        preview = preview_reader.read_region(
            0, 0, preview_reader.width, preview_reader.height,
        )
        self._image_path = os.path.abspath(image_path)
        self._original_width, self._original_height = map(int, original_size)
        self._preview_scale = float(preview_scale)
        self._preview_image = preview
        self._image_rgb = preview
        self._image_rgb_full = None
        self._preview_height, self._preview_width = preview.shape[:2]
        self._image_size = (self._preview_width, self._preview_height)
        self._large_image_mode = True
        self._image_pixmap = QPixmap.fromImage(self._ndarray_to_qimage(preview))
        for layer in self._layers.values():
            layer.data = None
            layer.preview_data = None
            layer.pixmap = None
            layer.visible = False
        self.layer_changed.emit("image")
        self.large_image_mode_changed.emit(True)
        return preview

    def has_image(self) -> bool:
        return bool(self._image_path and self._image_pixmap is not None)

    # ===================================================================
    # 坐标转换
    # ===================================================================

    def preview_to_global(self, x: int, y: int) -> Tuple[int, int]:
        """预览图坐标 → 原图全局像素坐标"""
        if not self._large_image_mode:
            return (x, y)
        return (int(x / self._preview_scale), int(y / self._preview_scale))

    def preview_to_global_f(self, x: float, y: float) -> Tuple[float, float]:
        """预览图坐标 → 原图全局像素坐标（浮点）"""
        if not self._large_image_mode:
            return (x, y)
        return (x / self._preview_scale, y / self._preview_scale)

    def global_to_preview(self, x: int, y: int) -> Tuple[int, int]:
        """原图全局像素坐标 → 预览图坐标"""
        if not self._large_image_mode:
            return (x, y)
        return (int(x * self._preview_scale), int(y * self._preview_scale))

    def global_to_preview_f(self, x: float, y: float) -> Tuple[float, float]:
        """原图全局像素坐标 → 预览图坐标（浮点）"""
        if not self._large_image_mode:
            return (x, y)
        return (x * self._preview_scale, y * self._preview_scale)

    def polygon_preview_to_global(self, points: List[List[float]]) -> List[List[float]]:
        """多边形预览图坐标 → 全局坐标"""
        if not self._large_image_mode:
            return points
        scale = 1.0 / self._preview_scale
        return [[px * scale, py * scale] for px, py in points]

    def polygon_global_to_preview(self, points: List[List[float]]) -> List[List[float]]:
        """多边形全局坐标 → 预览图坐标"""
        if not self._large_image_mode:
            return points
        return [[px * self._preview_scale, py * self._preview_scale] for px, py in points]

    def point_preview_to_global(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """点预览图坐标 → 全局坐标（浮点）"""
        return self.preview_to_global_f(point[0], point[1])

    def point_global_to_preview(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """点全局坐标 → 预览图坐标（浮点）"""
        return self.global_to_preview_f(point[0], point[1])

    # ────────────────────────────────────────────────────────────────
    # 规范坐标变换 API（image = original image pixel / global pixel）
    #   image_to_scene  : original image pixel → QGraphicsScene 坐标
    #   scene_to_image  : QGraphicsScene 坐标 → original image pixel
    #
    # 所有图层绘制和鼠标交互必须只通过这两个统一入口，
    # 不允许各图层自己手动乘 scale 或自己算 offset。
    # ────────────────────────────────────────────────────────────────

    def image_to_scene(self, x: float, y: float) -> Tuple[float, float]:
        """original image pixel → QGraphicsScene 坐标（浮点）"""
        return self.global_to_preview_f(x, y)

    def scene_to_image(self, x: int, y: int) -> Tuple[int, int]:
        """QGraphicsScene 坐标 → original image pixel（整数）"""
        return self.preview_to_global(x, y)

    def scene_to_image_f(self, x: float, y: float) -> Tuple[float, float]:
        """QGraphicsScene 坐标 → original image pixel（浮点）"""
        return self.preview_to_global_f(x, y)

    def point_image_to_scene(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        """点 original image pixel → QGraphicsScene"""
        return self.image_to_scene(*pt)

    def point_scene_to_image(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        """点 QGraphicsScene → original image pixel（浮点）"""
        return self.scene_to_image_f(*pt)

    # ===================================================================
    # 大图信息
    # ===================================================================

    def get_large_image_info(self) -> dict:
        """获取大图信息"""
        return {
            "enabled": self._large_image_mode,
            "original_width": self._original_width,
            "original_height": self._original_height,
            "preview_scale": self._preview_scale,
            "preview_width": self._preview_width,
            "preview_height": self._preview_height,
        } if self._large_image_mode else {
            "enabled": False,
            "original_width": self._original_width,
            "original_height": self._original_height,
        }

    @property
    def preview_width(self) -> int:
        return int(self._original_width * self._preview_scale)

    @property
    def preview_height(self) -> int:
        return int(self._original_height * self._preview_scale)

    # ===================================================================
    # 图层操作（统一 API）
    # ===================================================================

    def layers(self) -> Dict[str, LayerInfo]:
        return self._layers

    # ===================================================================
    # ★ QGraphicsItem 管理（关键：让 LayerManager 控制所有 scene 元素）
    # ===================================================================

    def add_item(self, layer_name: str, item: QGraphicsItem):
        """将 QGraphicsItem 注册到指定图层。item 的显隐将由该图层统一控制。"""
        name = _resolve_name(layer_name)
        if name not in self._items:
            self._items[name] = []
        self._items[name].append(item)
        # 同步当前图层可见性
        visible = self._layers.get(name, LayerInfo(name=name)).visible
        item.setVisible(visible)

    def remove_item(self, layer_name: str, item: QGraphicsItem):
        """取消注册指定 item"""
        name = _resolve_name(layer_name)
        if name in self._items and item in self._items[name]:
            self._items[name].remove(item)

    def clear_layer_items(self, layer_name: str):
        """清空某图层所有注册的 scene item（不移除 scene，只解绑）"""
        name = _resolve_name(layer_name)
        if name in self._items:
            self._items[name].clear()

    def clear_all_items(self):
        """清空所有图层注册的 item"""
        for name in self._items:
            self._items[name].clear()

    def show_layer(self, name: str):
        """显示指定图层（数据层 + 注册的 scene item）"""
        name = _resolve_name(name)
        if name not in self._layers:
            self._create_layer_if_needed(name)
        if name in self._layers and not self._layers[name].visible:
            self._layers[name].visible = True
            self.visibility_changed.emit(name, True)
        # ★ 同步所有已注册的 scene item 显隐
        self._apply_item_visibility(name, True)

    def hide_layer(self, name: str):
        """隐藏指定图层（数据层 + 注册的 scene item）"""
        name = _resolve_name(name)
        if name in self._layers and self._layers[name].visible:
            self._layers[name].visible = False
            self.visibility_changed.emit(name, False)
        # ★ 同步所有已注册的 scene item 显隐
        self._apply_item_visibility(name, False)

    def _apply_item_visibility(self, name: str, visible: bool):
        """设置该图层下所有注册的 QGraphicsItem 显隐"""
        for item in self._items.get(name, []):
            try:
                item.setVisible(visible)
            except RuntimeError:
                pass  # item 已被删除

    def toggle_layer(self, name: str, checked: bool):
        """切换图层显隐"""
        if checked:
            self.show_layer(name)
        else:
            self.hide_layer(name)

    def show_only(self, names: list) -> list:
        """仅显示指定图层，隐藏其余。会同时解析别名。"""
        resolved = [_resolve_name(n) for n in names]
        shown = []
        for name in self._layers:
            should_show = name in resolved
            if should_show:
                if not self._layers[name].visible:
                    self._layers[name].visible = True
                    self.visibility_changed.emit(name, True)
                shown.append(name)
            else:
                if self._layers[name].visible:
                    self._layers[name].visible = False
                    self.visibility_changed.emit(name, False)
        return shown

    def hide_all(self):
        """隐藏所有图层"""
        for name in self._layers:
            if self._layers[name].visible:
                self._layers[name].visible = False
                self.visibility_changed.emit(name, False)

    def get_visible_layers(self) -> list:
        """获取当前可见图层名列表"""
        return [n for n, l in self._layers.items() if l.visible]

    # ===================================================================
    # 阶段预设
    # ===================================================================

    def apply_stage_preset(self, stage: str):
        """应用指定阶段的默认图层显隐规则"""
        self._current_stage = stage
        preset = STAGE_PRESETS.get(stage)
        if preset is None:
            return

        # 先应用阶段 preset 的 show/hide
        for name in preset.get("show", []):
            self.show_layer(name)
        for name in preset.get("hide", []):
            self.hide_layer(name)

        # 然后叠加当前模式（简洁/调试）
        if self._mode == "clean":
            for name in CLEAN_MODE_HIDE:
                self.hide_layer(name)
        elif self._mode == "debug":
            for name in DEBUG_MODE_SHOW:
                self.show_layer(name)

        # 区域修正稳定模式的核心图层不得被 clean/debug 预设覆盖。
        if stage == "edit":
            for name in ("layer_road_mask", "layer_roi", "layer_ignore"):
                self.show_layer(name)
        elif stage == "segment":
            self.show_layer("layer_roi")

    # ===================================================================
    # 简洁模式 / 调试模式
    # ===================================================================

    def apply_clean_mode(self):
        """切换到简洁模式：隐藏调试图层"""
        self._mode = "clean"
        for name in CLEAN_MODE_HIDE:
            self.hide_layer(name)
        if self._current_stage == "edit":
            for name in ("layer_road_mask", "layer_roi", "layer_ignore"):
                self.show_layer(name)
        elif self._current_stage == "segment":
            self.show_layer("layer_roi")
        self.mode_changed.emit("clean")

    def apply_debug_mode(self):
        """切换到调试模式：显示全部图层"""
        self._mode = "debug"
        for name in DEBUG_MODE_SHOW:
            self.show_layer(name)
        self.mode_changed.emit("debug")

    def toggle_mode(self):
        """切换简洁/调试模式"""
        if self._mode == "clean":
            self.apply_debug_mode()
        else:
            self.apply_clean_mode()

    # ===================================================================
    # 快捷预设
    # ===================================================================

    def apply_clean_display(self):
        """一键清爽显示：只突出当前阶段必要的路网图层。
        强制隐藏所有调试图层：sample_points / roi / ignore / mask / skeleton_nodes / debug
        """
        # 强制隐藏所有调试图层（不管阶段预设）
        _force_hide = [
            "layer_sample_points", "layer_roi", "layer_ignore",
            "layer_skeleton_nodes", "layer_task_points",
        ]
        for name in _force_hide:
            self.hide_layer(name)

        # 根据阶段显示必要图层
        stage = self._current_stage
        if stage in ("import",):
            pass  # 只显示 image
        elif stage in ("segment",):
            self.show_layer("layer_road_mask")
            self.show_layer("layer_roi")
            self.hide_layer("layer_sample_points")  # 清爽下也隐藏样本点
        elif stage in ("edit",):
            # 区域修正稳定模式必须始终显示 Mask / ROI / Ignore。
            self.show_layer("layer_road_mask")
            self.show_layer("layer_roi")
            self.show_layer("layer_ignore")
        elif stage in ("skeleton",):
            self.show_layer("layer_skeleton")
        elif stage in ("graph",):
            self.show_layer("layer_skeleton")
            self.show_layer("layer_draft_graph")
            self.show_layer("layer_final_graph")
        elif stage in ("export",):
            self.show_layer("layer_final_graph")
            self.show_layer("layer_planned_path")
            self.show_layer("layer_sparse_waypoints")

    def apply_debug_display(self):
        """一键调试显示：显示中间图层，但绝不覆盖/替换 Road Mask 数据。

        大图模式下优先显示 working Road Mask；preview_segmentation 仅在
        没有正式/工作 mask 时才作为回退显示，避免冒充正式 Road Mask。
        """
        # 先确保 Road Mask 可渲染且可见（含自动补 preview）
        has_working = self.ensure_working_mask_preview("layer_road_mask")
        road = self._layers.get("layer_road_mask")
        if road is not None:
            road.visible = True
            if road.opacity < 80:
                road.opacity = LAYER_DEFAULT_ALPHA.get("layer_road_mask", 115)

        _show = [
            "layer_sample_points", "layer_roi", "layer_ignore",
            "layer_road_mask",
            "layer_skeleton",  # cleaned skeleton
            "layer_draft_graph", "layer_final_graph",
            "layer_main_road_seed",
        ]
        for name in _show:
            self.show_layer(name)

        # preview_segmentation：有 working/formal mask 时隐藏，避免同色覆盖
        if has_working:
            self.hide_layer("layer_preview_segmentation")
        else:
            self.show_layer("layer_preview_segmentation")

        # raw skeleton / nodes：调试可见但默认不强制打开（噪声多）
        # 若图层已有数据则保持用户勾选状态；无数据则隐藏
        for name in ("layer_raw_skeleton", "layer_skeleton_nodes"):
            layer = self._layers.get(name)
            if layer is None or layer.data is None:
                self.hide_layer(name)

    # ===================================================================
    # 数据操作（保持原有兼容）
    # ===================================================================

    def _make_preview_from_full(self, data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Downscale full-resolution data to the preview canvas size.

        Used so a large-image layer always has a renderable preview overlay,
        instead of refusing to draw it (which used to make the Road Mask vanish).
        """
        if not isinstance(data, np.ndarray) or data.ndim < 2:
            return None
        pw, ph = self._preview_width, self._preview_height
        if pw <= 0 or ph <= 0:
            return None
        dh, dw = data.shape[:2]
        if (dw, dh) == (pw, ph):
            return data
        try:
            return cv2.resize(data, (pw, ph), interpolation=cv2.INTER_NEAREST)
        except Exception:
            return None

    def set_layer_data(self, name: str, data: np.ndarray,
                       preview_data: Optional[np.ndarray] = None):
        """设置图层数据"""
        name = _resolve_name(name)
        if name not in self._layers:
            self._create_layer_if_needed(name)
        if name not in self._layers:
            return
        layer = self._layers[name]
        layer.data = data
        # ★ 大图模式：调用方未提供 preview 时自动生成，避免全分辨率 overlay 被拒绝渲染
        #   （否则任何一次 set_layer_data(mask) 不带 preview 都会让 Road Mask 消失）。
        if (preview_data is None and self._large_image_mode
                and isinstance(data, np.ndarray) and data.ndim >= 2):
            dh, dw = data.shape[:2]
            if (self._preview_width > 0
                    and (dw, dh) != (self._preview_width, self._preview_height)):
                preview_data = self._make_preview_from_full(data)
        layer.preview_data = preview_data
        layer.pixmap = None
        layer.visible = True
        self.layer_changed.emit(name)

    def update_layer_preview_region(self, name: str, rect) -> None:
        """Refresh only a dirty original-pixel rectangle in a large-image preview.

        This is used by local mask brush/eraser edits.  It deliberately avoids
        resizing the complete global mask in the GUI thread.
        """
        name = _resolve_name(name)
        if not self._large_image_mode or name not in self._layers:
            return
        layer = self._layers[name]
        source = layer.data
        if not isinstance(source, np.ndarray) or source.ndim < 2:
            return
        ph, pw = self._preview_height, self._preview_width
        if ph <= 0 or pw <= 0:
            return
        # ★ 缺失或尺寸不匹配时，必须先从全图 data 生成完整 preview，
        #   绝不能用全零数组起步——否则画笔只更新脏区，其余道路会在
        #   refresh_scene / 调试显示时“消失”。
        if layer.preview_data is None or layer.preview_data.shape[:2] != (ph, pw):
            full_preview = self._make_preview_from_full(source)
            if full_preview is not None:
                layer.preview_data = full_preview
            else:
                shape = (ph, pw) + (() if source.ndim == 2 else (source.shape[2],))
                layer.preview_data = np.zeros(shape, dtype=source.dtype)
        x0, y0, x1, y1 = (int(v) for v in rect)
        h, w = source.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return
        px0 = max(0, min(pw - 1, int(np.floor(x0 * pw / w))))
        py0 = max(0, min(ph - 1, int(np.floor(y0 * ph / h))))
        px1 = max(px0 + 1, min(pw, int(np.ceil(x1 * pw / w))))
        py1 = max(py0 + 1, min(ph, int(np.ceil(y1 * ph / h))))
        patch = source[y0:y1, x0:x1]
        layer.preview_data[py0:py1, px0:px1] = cv2.resize(
            patch, (px1 - px0, py1 - py0), interpolation=cv2.INTER_NEAREST,
        )
        layer.pixmap = None
        self.layer_changed.emit(name)

    def ensure_working_mask_preview(self, name: str = "layer_road_mask") -> bool:
        """Guarantee the road-mask layer can be rendered.

        In large-image mode a full-resolution ``data`` array with a missing
        or suspiciously empty ``preview_data`` is auto-downscaled here so the
        Road Mask stays visible across display-mode switches.
        Returns True if a renderable mask exists.
        """
        name = _resolve_name(name)
        if name not in self._layers:
            return False
        layer = self._layers[name]
        data, _ = self._resolve_layer_data(layer.data, name)
        if data is None:
            return False
        if not self._large_image_mode:
            return True
        ph, pw = self._preview_height, self._preview_width
        if pw <= 0 or ph <= 0:
            return True
        need = (
            layer.preview_data is None
            or layer.preview_data.shape[:2] != (ph, pw)
        )
        # 画笔脏区更新曾用全零 preview 起步：data 有路但 preview 几乎为空 → 强制重生
        if (not need and isinstance(layer.preview_data, np.ndarray)
                and isinstance(data, np.ndarray)):
            prev_nz = int(np.count_nonzero(layer.preview_data))
            data_nz = int(np.count_nonzero(data))
            if data_nz > 1000 and prev_nz < max(50, data_nz // 200):
                need = True
        if need:
            gen = self._make_preview_from_full(data)
            if gen is None:
                return False
            layer.preview_data = gen
            layer.pixmap = None
        # 保证 Road Mask 可见且透明度合理（约 0.35～0.5）
        if layer.opacity < 80:
            layer.opacity = LAYER_DEFAULT_ALPHA.get(name, 115)
        layer.visible = True
        return True

    def load_layer_from_file(self, name: str, path: str) -> bool:
        if not os.path.exists(path):
            return False
        name = _resolve_name(name)
        if name not in self._layers:
            self._create_layer_if_needed(name)
        if name not in self._layers:
            return False
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if data is None:
            return False
        if len(data.shape) == 3:
            if data.shape[2] == 4:
                data = data[:, :, 3]
            else:
                data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        self.set_layer_data(name, data)
        return True

    def set_layer_visible(self, name: str, visible: bool):
        """设置图层显隐（兼容旧接口）"""
        if visible:
            self.show_layer(name)
        else:
            self.hide_layer(name)

    def clear_layer(self, name: str):
        """清空单个图层数据"""
        name = _resolve_name(name)
        if name in self._layers:
            self._layers[name].data = None
            self._layers[name].preview_data = None
            self._layers[name].pixmap = None
            self._layers[name].visible = False
            self.layer_changed.emit(name)
        # ★ 清空该图层注册的 scene item
        if name in self._items:
            self._items[name].clear()

    def clear_image(self):
        self._image_rgb = None
        self._image_rgb_full = None
        self._preview_image = None
        self._image_pixmap = None
        self._image_path = ""
        self._image_size = (0, 0)
        self._large_image_mode = False
        self._preview_scale = 1.0
        self._original_width = 0
        self._original_height = 0
        for layer in self._layers.values():
            layer.data = None
            layer.preview_data = None
            layer.pixmap = None
            layer.visible = False
        for name in self._items:
            self._items[name].clear()
        self.layer_changed.emit("image")
        self.large_image_mode_changed.emit(False)

    def set_layer_opacity(self, name: str, opacity: int):
        name = _resolve_name(name)
        if name not in self._layers:
            return
        self._layers[name].opacity = max(0, min(255, opacity))
        self._layers[name].pixmap = None
        self.layer_changed.emit(name)

    def set_layer_zvalue(self, name: str, z: int):
        self._zvalues[name] = z

    def get_layer_zvalue(self, name: str) -> int:
        return self._zvalues.get(name, 0)

    # ===================================================================
    # 兼容旧接口
    # ===================================================================

    def is_layer_visible(self, name: str) -> bool:
        name = _resolve_name(name)
        if name not in self._layers:
            return False
        return self._layers[name].visible

    def get_layer_data(self, name: str) -> Optional[np.ndarray]:
        name = _resolve_name(name)
        if name not in self._layers:
            return None
        return self._layers[name].data

    def get_layer_preview(self, name: str) -> Optional[np.ndarray]:
        name = _resolve_name(name)
        if name not in self._layers:
            return None
        return self._layers[name].preview_data

    def get_layer_pixmap(self, name: str) -> Optional[QPixmap]:
        name = _resolve_name(name)
        if name not in self._layers:
            return None
        layer = self._layers[name]
        if not layer.visible or layer.data is None:
            return None
        if layer.pixmap is None:
            layer.pixmap = self._build_layer_overlay(name)
        return layer.pixmap

    def invalidate_layer_cache(self, name: str = None):
        if name:
            name = _resolve_name(name)
            if name in self._layers:
                self._layers[name].pixmap = None
        else:
            for layer in self._layers.values():
                layer.pixmap = None

    def get_stage_layers(self, stage: str) -> list:
        """获取某阶段应显示的图层名列表（兼容旧接口）"""
        preset = STAGE_PRESETS.get(stage, {})
        return list(preset.get("show", []))

    # ===================================================================
    # 内部辅助
    # ===================================================================

    def _create_layer_if_needed(self, name: str):
        """按需创建图层"""
        if name in self._layers:
            return
        color = LAYER_COLORS.get(name, QColor(200, 200, 200))
        alpha = LAYER_DEFAULT_ALPHA.get(name, 128)
        self._layers[name] = LayerInfo(
            name=name, visible=False, color=color, opacity=alpha
        )

    def _resolve_layer_data(self, data, name: str):
        """统一解析图层 data，支持 ndarray 和多种 dict 格式。

        返回 (resolved_ndarray_or_None, used_preview_data_bool)。
        此方法绝不访问 .shape，保障 dict 不会泄漏到下游。
        """
        if data is None:
            return None, False

        # 已经是 ndarray → 直接返回
        if isinstance(data, np.ndarray):
            return data, False

        # dict 类型 → 按优先级解析
        if isinstance(data, dict):
            # 1) skeleton dict: {"raw_skeleton": ..., "optimized_skeleton": ...}
            for key in ("optimized_skeleton", "skeleton_optimized",
                         "skeleton", "raw_skeleton"):
                val = data.get(key)
                if isinstance(val, np.ndarray) and val.size > 0:
                    return val, False

            # 2) mask-path dict: {"global_road_mask_path": "...", ..., "preview_only": ...}
            for key in ("processed_global_mask_path", "processed_mask_path",
                         "global_road_mask_path", "global_mask_path",
                         "road_mask_path", "mask_path"):
                path = data.get(key, "")
                if isinstance(path, str) and os.path.isfile(path):
                    try:
                        arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        if arr is not None and arr.size > 0:
                            return arr, False
                    except Exception:
                        continue

            # 3) inline array 字段
            for key in ("array", "mask", "data"):
                val = data.get(key)
                if isinstance(val, np.ndarray) and val.size > 0:
                    return val, False

            # 无法解析
            return None, False

        # 其他类型（如 str path）→ 尝试作为文件路径读取
        if isinstance(data, str) and os.path.isfile(data):
            try:
                arr = cv2.imread(data, cv2.IMREAD_GRAYSCALE)
                if arr is not None and arr.size > 0:
                    return arr, False
            except Exception:
                pass

        # 最终回退：hasattr shape?
        return None, False

    def _build_layer_overlay(self, name: str) -> Optional[QPixmap]:
        """根据图层数据构建彩色半透明 QPixmap。

        支持两种数据格式：
        - 2D 二值数据（mask）：根据 layer.color / layer.opacity 构建纯色 ARGB overlay
        - 3 通道 RGB 数据（如 preview_seg_overlay.png）：直接转为 ARGB + alpha
        """
        layer = self._layers[name]
        data = layer.data
        if data is None or self._image_size == (0, 0):
            return None

        w, h = self._image_size

        # ★ 统一解析数据（dict → ndarray，文件路径 → ndarray）
        data, _used_preview = self._resolve_layer_data(data, name)
        if data is None:
            return None

        # ★ 大图安全防护：如果 data 是原图全分辨率尺寸（> 预览尺寸），
        #   且没有 preview_data，则自动生成一个预览级 overlay（而不是拒绝渲染，
        #   拒绝会导致 Road Mask 在切换显示时凭空消失）。
        if self._large_image_mode and layer.preview_data is None:
            orig_w, orig_h = self._original_width, self._original_height
            d_h, d_w = data.shape[:2]
            if (d_w, d_h) == (orig_w, orig_h) and orig_w > self._preview_width:
                gen = self._make_preview_from_full(data)
                if gen is not None:
                    layer.preview_data = gen
                    data = gen
                else:
                    return None

        if self._large_image_mode and layer.preview_data is not None:
            data = layer.preview_data

        # ★ 如果是 3 通道 RGB 数据（如 preview_seg_overlay），直接构建 ARGB
        if len(data.shape) == 3 and data.shape[2] in (3, 4):
            return self._build_rgb_overlay(data, w, h, layer.opacity, name)

        # --- 以下为 2D 二值数据 ---
        if data.shape[:2] != (h, w):
            data = cv2.resize(data, (w, h), interpolation=cv2.INTER_NEAREST)

        color = layer.color
        alpha = layer.opacity

        binary = (data > 0) if len(data.shape) == 2 else (data[:, :, 0] > 0)

        # 骨架图层膨胀1像素
        if name in ("layer_skeleton", "skeleton"):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.dilate(binary.astype(np.uint8), kernel, iterations=1).astype(bool)

        argb = np.zeros((h, w, 4), dtype=np.uint8)
        argb[binary, 0] = color.blue()
        argb[binary, 1] = color.green()
        argb[binary, 2] = color.red()
        argb[binary, 3] = alpha

        qimg = QImage(argb.data, w, h, w * 4, QImage.Format.Format_ARGB32)
        qimg = qimg.copy()
        return QPixmap.fromImage(qimg)

    def _build_rgb_overlay(self, data: np.ndarray, w: int, h: int,
                           alpha: int, name: str) -> Optional[QPixmap]:
        """从 3/4 通道 RGB(A) 数据构建 QPixmap overlay。

        用于 layer_preview_segmentation 等预渲染的叠加图层。
        """
        d_h, d_w = data.shape[:2]
        channels = data.shape[2]

        # 缩放以匹配显示尺寸（大图模式下为预览尺寸）
        if (d_w, d_h) != (w, h):
            data = cv2.resize(data, (w, h), interpolation=cv2.INTER_AREA)

        # 构建 ARGB
        argb = np.zeros((h, w, 4), dtype=np.uint8)

        if channels >= 3:
            # RGB → ARGB32 (BGRA byte order in memory on little-endian)
            # data[:,:,0]=R, data[:,:,1]=G, data[:,:,2]=B
            # argb byte 0 = B, byte 1 = G, byte 2 = R, byte 3 = A
            argb[:, :, 0] = data[:, :, 2]  # B
            argb[:, :, 1] = data[:, :, 1]  # G
            argb[:, :, 2] = data[:, :, 0]  # R
        if channels >= 4:
            # 已有 alpha channel，使用它
            argb[:, :, 3] = data[:, :, 3]
            # 但用 layer opacity 做整体 alpha 覆盖
            argb[:, :, 3] = (argb[:, :, 3].astype(np.uint16) * alpha // 255).astype(np.uint8)
        else:
            # 无 alpha channel：非零像素使用 layer opacity
            has_content = np.any(data > 0, axis=2)
            argb[has_content, 3] = alpha

        qimg = QImage(argb.data, w, h, w * 4, QImage.Format.Format_ARGB32)
        qimg = qimg.copy()
        return QPixmap.fromImage(qimg)

    @staticmethod
    def _ndarray_to_qimage(rgb: np.ndarray) -> QImage:
        h, w = rgb.shape[:2]
        if not rgb.flags['C_CONTIGUOUS']:
            rgb = np.ascontiguousarray(rgb)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        return qimg.copy()
