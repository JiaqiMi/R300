"""
主画布视图：基于 QGraphicsView/QGraphicsScene，支持缩放、平移和图层叠加。
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import (
    QWheelEvent, QMouseEvent, QKeyEvent,
    QPainter, QPen, QColor, QCursor, QFont, QBrush, QPolygonF,
    QImage, QPixmap,
)
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsLineItem,
    QGraphicsPolygonItem, QGraphicsItemGroup,
    QRubberBand,
)

try:
    from shiboken6 import isValid as _shiboken_is_valid
except ImportError:
    def _shiboken_is_valid(obj):
        try:
            return obj is not None
        except RuntimeError:
            return False


def _safe_remove_scene_item(scene, item):
    """安全移除 QGraphicsScene 中的 item。"""
    if item is None:
        return
    try:
        if _shiboken_is_valid(item) and item.scene() is scene:
            scene.removeItem(item)
    except (RuntimeError, AttributeError):
        pass

from .layer_manager import LayerManager
from roadnet.region_edit import (
    PolygonRegion, ensure_mask_image_size, paint_mask_segment,
)


class CanvasView(QGraphicsView):
    """带缩放平移和图层叠加的主画布"""

    # 信号
    mouse_moved = Signal(int, int)            # 图像像素坐标 (x, y)
    zoom_changed = Signal(float)              # 缩放比例
    tool_interaction = Signal(str, object)    # 工具交互事件
    sample_points_changed = Signal(int, int)  # 正样本数, 负样本数
    task_point_clicked = Signal(str, int, int)  # 任务点点击: (type, x_global, y_global)
    calibration_map_clicked = Signal(int, int)  # 控制点图上点击: (x_global, y_global)

    # 缩放限制
    MIN_ZOOM = 0.02
    MAX_ZOOM = 30.0
    ZOOM_FACTOR = 1.15

    # ROI 颜色常量
    ROI_COLOR = QColor(68, 153, 255)
    ROI_FILL_COLOR = QColor(68, 153, 255, 80)
    ROI_POINT_RADIUS = 5

    # Ignore 颜色常量
    IGNORE_COLOR = QColor(255, 85, 85)
    IGNORE_FILL_COLOR = QColor(255, 85, 85, 100)

    def __init__(self, layer_manager: LayerManager, parent=None):
        super().__init__(parent)

        self._layer_manager = layer_manager
        self._zoom_level = 1.0

        # Scene
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # 图层 items
        self._image_item: Optional[QGraphicsPixmapItem] = None
        self._overlay_items: dict[str, QGraphicsPixmapItem] = {}

        # 采样点数据
        self._positive_points: list[tuple] = []
        self._negative_points: list[tuple] = []
        self._sample_items: list = []  # QGraphicsItem 实例

        # ---- ROI 状态（多多边形支持） ----
        self._roi_points: list[QPointF] = []                 # 当前正在绘制的 ROI 顶点
        self._roi_dots: list[QGraphicsEllipseItem] = []      # 当前顶点圆点
        self._roi_temp_line: Optional[QGraphicsLineItem] = None  # 到鼠标的临时线
        self._roi_polygon_preview: Optional[QGraphicsPolygonItem] = None  # 当前多边形预览
        self._roi_polygons: list[QPolygonF] = []             # 所有已完成 ROI 的几何数据
        self._roi_regions: list[PolygonRegion] = []           # 原图像素坐标（权威数据）
        self._roi_items: list[QGraphicsPolygonItem] = []     # 所有已完成 ROI 的显示项
        self._roi_tile_overlay_items: list = []              # ROI tile 覆盖可视化边框

        # ---- Ignore 状态（★ 统一为多边形模式，与 ROI 交互一致） ----
        self._ignore_points: list[QPointF] = []                  # 当前正在绘制的忽略区顶点
        self._ignore_dots: list[QGraphicsEllipseItem] = []       # 当前顶点圆点
        self._ignore_temp_line: Optional[QGraphicsLineItem] = None  # 到鼠标的临时线
        self._ignore_polygon_preview: Optional[QGraphicsPolygonItem] = None  # 当前多边形预览
        self._ignore_polygons: list[QPolygonF] = []              # 所有已完成忽略区的几何数据
        self._ignore_regions: list[PolygonRegion] = []            # 原图像素坐标（权威数据）
        self._ignore_items: list[QGraphicsPolygonItem] = []      # 所有已完成忽略区的显示项

        # ★ 兼容旧矩形数据（加载项目时使用，转为 QPolygonF 存入 _ignore_polygons）
        self._ignore_rects_deprecated: list = []

        # ---- Mask 精修状态 ----（V3 实装） ----
        self._mask_brush_radius: int = 3
        self._mask_undo_stack: list = []
        self._mask_undo_max: int = 20
        # 运行时变量
        self.is_mask_drawing: bool = False
        self.mask_draw_mode: Optional[str] = None   # "add" / "erase"
        self.last_mask_point: Optional[tuple] = None  # (x, y) 图像坐标
        self._mask_stroke_changed: bool = False
        self._mask_stroke_before_patch = None
        self._mask_dirty_rect = None
        self._history_mask_patch_callback = None

        # 区域修正自检使用的最近一次坐标。
        self._last_region_scene_pos: Optional[tuple[float, float]] = None
        self._last_region_image_pos: Optional[tuple[float, float]] = None

        # ---- 画笔圆形预览 ----
        self._brush_preview_item: Optional[QGraphicsEllipseItem] = None

        # ---- 主路种子线（大图主路修复约束）----
        # 内部统一存 rich dict；get_main_road_seed_strokes() 仍返回点列表兼容旧逻辑。
        self._main_road_seed_strokes: list = []  # list[dict]
        self._main_road_seed_items: list = []       # 已完成种子线显示项
        self._seed_current_points: list[tuple[float, float]] = []           # 原图像素
        self._seed_current_items: list = []         # 当前进行中的线段项
        self._seed_drawing: bool = False
        self._corridor_overlay_item = None          # 主路 corridor 半透明预览项
        self._ribbon_preview_item = None
        # 绘制模式: freehand | two_point | polyline
        self._seed_draw_mode: str = "freehand"
        self._seed_continuous_two_point: bool = True
        self._seed_road_width_m: float = 8.0
        self._seed_road_radius_px: Optional[float] = None
        self._seed_gsd_m_per_px: Optional[float] = None
        self._seed_width_mode: str = "normal"  # normal / main_road / junction / custom
        self._seed_endpoint_snap_px: float = 15.0
        self._seed_graph_node_snap_px: float = 20.0
        self._seed_snap_candidates: list[tuple[float, float]] = []
        self._seed_temp_line_item = None
        self._seed_temp_dot_items: list = []
        self._seed_two_point_start: Optional[tuple[float, float]] = None  # image px
        self._seed_polyline_active: bool = False

        # ---- Skeleton 优化结果（V4.1） ----
        self.skeleton_result: Optional[dict] = None    # optimize_skeleton 返回的完整 dict

        # ---- 空格临时平移模式 ----
        self._space_pan_active: bool = False
        self._saved_tool_before_pan: str = "pan"

        # ---- Graph 数据（路网编辑阶段） ----
        self._draft_nodes: list = []
        self._draft_edges: list = []
        self._final_nodes: list = []
        self._final_edges: list = []

        # 配置
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # 优化渲染
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, False)

        # 背景
        self.setBackgroundBrush(QColor(30, 30, 46))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

        # 启用鼠标追踪
        self.setMouseTracking(True)

        # 当前工具
        self._current_tool: str = "pan"
        self.debug_click_marker_enabled = False

        # 焦点策略（接收键盘事件）
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # ★ 坐标变换调试
        self.debug_coord_enabled: bool = False
        self._debug_coord_text_items: list = []

        # ★ 全局撤销回调（由 MainWindow 注入）
        self._history = None
        self._history_push_callback = None

        # 连接图层信号
        layer_manager.layer_changed.connect(self._on_layer_changed)
        layer_manager.visibility_changed.connect(self._on_visibility_changed)

    # ===================================================================
    # 属性
    # ===================================================================

    @property
    def zoom_level(self) -> float:
        return self._zoom_level

    @property
    def current_tool(self) -> str:
        return self._current_tool

    @property
    def brush_radius(self) -> int:
        """画笔半径（px）"""
        return self._mask_brush_radius

    @brush_radius.setter
    def brush_radius(self, value: int):
        """设置画笔半径，限制在 1-100"""
        self._mask_brush_radius = max(1, min(100, value))

    @property
    def current_mask(self) -> Optional[np.ndarray]:
        """从 LayerManager 获取当前 mask（同步层）"""
        m = self._layer_manager.get_layer_data("mask")
        if m is not None:
            return m
        return self._layer_manager.get_layer_data("road_mask")

    @property
    def positive_points(self) -> list:
        return self._positive_points

    @property
    def negative_points(self) -> list:
        return self._negative_points

    @current_tool.setter
    def current_tool(self, tool: str):
        print(f"[DEBUG][Canvas] set_tool: {tool}")

        # ★ 切换工具时取消进行中的 ROI / Ignore / ManualEdge 绘制
        old_tool = self._current_tool
        if old_tool == "roi" and tool != "roi":
            self._clear_current_roi_drawing()
        if old_tool == "ignore" and tool != "ignore":
            self._clear_current_ignore_drawing()
        mask_tools = {"mask_refine", "mask_brush", "mask_eraser"}
        if old_tool in mask_tools and tool not in mask_tools:
            self.is_mask_drawing = False
            self.mask_draw_mode = None
            self.last_mask_point = None
            self._remove_brush_preview()
        if old_tool == "graph_draw_edge" and tool != "graph_draw_edge":
            # 取消进行中的手动画边
            self.tool_interaction.emit("cancel_manual_edge", None)
        if old_tool == "main_road_seed" and tool != "main_road_seed":
            if getattr(self, "_seed_draw_mode", "freehand") == "freehand":
                self._finalize_seed_stroke()
            else:
                self._cancel_seed_in_progress(push_history=False)

        self._current_tool = tool
        # 根据工具设置光标
        cursor_map = {
            "pan":               Qt.CursorShape.OpenHandCursor,
            "brush":             Qt.CursorShape.CrossCursor,
            "eraser":            Qt.CursorShape.CrossCursor,
            "calibrate_map_click": Qt.CursorShape.CrossCursor,  # 控制点图上配准
            "polyline":          Qt.CursorShape.CrossCursor,
            "roi":               Qt.CursorShape.CrossCursor,
            "ignore":            Qt.CursorShape.CrossCursor,
            "positive_sample":   Qt.CursorShape.PointingHandCursor,
            "negative_sample":   Qt.CursorShape.PointingHandCursor,
            "graph_add_node":    Qt.CursorShape.CrossCursor,
            "graph_delete_node": Qt.CursorShape.ForbiddenCursor,
            "graph_add_edge":    Qt.CursorShape.CrossCursor,
            "graph_delete_edge": Qt.CursorShape.ForbiddenCursor,
            "graph_move_node":   Qt.CursorShape.SizeAllCursor,
            "graph_merge_nodes": Qt.CursorShape.CrossCursor,
            "graph_draw_edge":   Qt.CursorShape.CrossCursor,
            "mask_refine":       Qt.CursorShape.CrossCursor,
            "mask_brush":        Qt.CursorShape.CrossCursor,
            "mask_eraser":       Qt.CursorShape.CrossCursor,
            "set_start":         Qt.CursorShape.CrossCursor,
            "set_end":           Qt.CursorShape.CrossCursor,
            "add_task":          Qt.CursorShape.CrossCursor,
            "main_road_seed":    Qt.CursorShape.CrossCursor,
        }
        self.setCursor(cursor_map.get(tool, Qt.CursorShape.ArrowCursor))

        if tool == "pan" or self._space_pan_active:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def set_tool(self, tool_name: str):
        """外部接口：设置当前工具"""
        print(f"[DEBUG][Canvas] set_tool: {tool_name}")
        self.current_tool = tool_name

    # ===================================================================
    # 画笔预览圆
    # ===================================================================
    BRUSH_PREVIEW_Z = 9998

    def _update_brush_preview(self, scene_x: int, scene_y: int):
        """在场景中显示画笔半径预览圆"""
        # 移除旧预览
        if self._brush_preview_item is not None:
            try:
                self._scene.removeItem(self._brush_preview_item)
            except RuntimeError:
                pass
            self._brush_preview_item = None

        if self._current_tool not in {"mask_refine", "mask_brush", "mask_eraser"}:
            return

        r = self._mask_brush_radius
        # 用 scene 坐标绘制（zoom 会影响视觉大小, 这正是我们需要的）
        item = QGraphicsEllipseItem(
            scene_x - r, scene_y - r, r * 2, r * 2
        )
        if self.mask_draw_mode == "erase" or self._current_tool == "mask_eraser":
            pen = QPen(QColor(255, 85, 85), 1)
        else:
            pen = QPen(QColor(80, 250, 123), 1)
        pen.setStyle(Qt.PenStyle.DashLine)
        item.setPen(pen)
        item.setBrush(QColor(0, 0, 0, 0))  # 透明填充
        item.setZValue(self.BRUSH_PREVIEW_Z)
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._brush_preview_item = item
        self._scene.addItem(item)

    def _remove_brush_preview(self):
        """移除画笔预览圆"""
        if self._brush_preview_item is not None:
            try:
                self._scene.removeItem(self._brush_preview_item)
            except RuntimeError:
                pass
            self._brush_preview_item = None

    def clear_samples(self):
        """清空所有采样点"""
        print("[DEBUG][Canvas] clear_samples")
        self._positive_points.clear()
        self._negative_points.clear()
        for item in self._sample_items:
            _safe_remove_scene_item(self._scene, item)
        self._sample_items.clear()
        self._layer_manager.clear_layer_items("layer_sample_points")  # ★ 解绑
        self.viewport().update()
        self.sample_points_changed.emit(0, 0)

    def clear_roi_and_ignore(self):
        """清空所有 ROI 和 Ignore（用于新建项目）"""
        # 避免重复推入撤销（_clear_all_* 内部已有检查）
        self._clear_all_roi()
        self._clear_all_ignore()

    def get_roi_polygons(self) -> list:
        """返回所有已完成 ROI（原始图像像素坐标，兼容旧接口）。"""
        return self._roi_polygons

    def _sync_region_models(self):
        """Keep legacy QPolygonF storage and PolygonRegion models in sync."""
        if len(self._roi_regions) != len(self._roi_polygons):
            self._roi_regions = [
                PolygonRegion.create("roi", [(p.x(), p.y()) for p in polygon])
                for polygon in self._roi_polygons if len(polygon) >= 3
            ]
        if len(self._ignore_regions) != len(self._ignore_polygons):
            self._ignore_regions = [
                PolygonRegion.create("ignore", [(p.x(), p.y()) for p in polygon])
                for polygon in self._ignore_polygons if len(polygon) >= 3
            ]

    def get_roi_regions(self) -> list[PolygonRegion]:
        self._sync_region_models()
        return list(self._roi_regions)

    def get_enabled_roi_count(self) -> int:
        """返回 enabled 状态的 ROI 多边形数量。"""
        self._sync_region_models()
        return sum(1 for region in self._roi_regions if region.enabled)

    def get_enabled_roi_polygon_points(self) -> list[list[tuple[float, float]]]:
        """返回 enabled ROI 多边形顶点（原图像素坐标）。"""
        self._sync_region_models()
        return [
            list(region.points)
            for region in self._roi_regions
            if region.enabled and len(region.points) >= 3
        ]

    def get_ignore_regions(self) -> list[PolygonRegion]:
        self._sync_region_models()
        return list(self._ignore_regions)

    def _global_point_to_scene(self, point: QPointF) -> QPointF:
        sx, sy = self.image_to_scene(point.x(), point.y())
        return QPointF(float(sx), float(sy))

    def _global_polygon_to_scene(self, polygon: QPolygonF) -> QPolygonF:
        return QPolygonF([self._global_point_to_scene(point) for point in polygon])

    def _log_region_coordinate(self, scene_x: float, scene_y: float):
        gx, gy = self.scene_to_global_xy(QPointF(scene_x, scene_y))
        self._last_region_scene_pos = (float(scene_x), float(scene_y))
        self._last_region_image_pos = (float(gx), float(gy))
        iw, ih = self._layer_manager.original_size
        if iw <= 0 or ih <= 0:
            iw, ih = self._layer_manager.image_size
        print(f"[RegionEdit] mouse scene=({scene_x:.2f},{scene_y:.2f})")
        print(f"[RegionEdit] image pixel=({gx},{gy})")
        print(f"[RegionEdit] image_size=({iw},{ih})")
        print(f"[RegionEdit] current_tool={self._current_tool}")
        return gx, gy

    # ===================================================================
    # zValue 统一常量
    # ===================================================================
    ZVAL_IMAGE          = 0     # 影像
    ZVAL_MASK           = 10    # 道路 Mask（绿色）
    ZVAL_SAMPLE         = 100   # 采样点
    ZVAL_ROI            = 200   # ROI 区域
    ZVAL_IGNORE         = 300   # Ignore 区域
    ZVAL_SKELETON       = 400   # Skeleton 骨架线
    ZVAL_AUTO_EDGE      = 500   # 自动边
    ZVAL_MANUAL_EDGE    = 510   # 人工边
    ZVAL_NODE           = 520   # 节点
    ZVAL_SELECTED       = 560   # 选中高亮
    ZVAL_PATH           = 600   # 规划路径
    ZVAL_DEBUG          = 9999  # 调试标记

    # ★★★ 调试阶段 zValue：绕过 LayerManager，确保可见 ★★★
    DEBUG_ROI_DOT_Z        = 9998
    DEBUG_ROI_PREVIEW_Z    = 9997
    DEBUG_ROI_FINAL_Z      = 9996
    DEBUG_ROI_TILE_Z       = 9995
    DEBUG_IGNORE_PREVIEW_Z = 9997
    DEBUG_IGNORE_FINAL_Z   = 9996

    def _redraw_samples(self):
        """将已存储的采样点重绘到场景中（从全局坐标转换为预览坐标显示）"""
        pos_color = QColor(80, 250, 123)   # 绿色
        neg_color = QColor(255, 85, 85)    # 红色
        lm = self._layer_manager
        for x_global, y_global in self._positive_points:
            # 全局坐标转换为预览坐标
            x, y = lm.global_to_preview(x_global, y_global)
            dot = QGraphicsEllipseItem(x - 4, y - 4, 8, 8)
            dot.setBrush(pos_color)
            dot.setPen(QColor(0, 0, 0, 0))
            dot.setZValue(self.ZVAL_SAMPLE)
            self._sample_items.append(dot)
            self._scene.addItem(dot)
            lm.add_item("layer_sample_points", dot)  # ★ 注册
        for x_global, y_global in self._negative_points:
            # 全局坐标转换为预览坐标
            x, y = lm.global_to_preview(x_global, y_global)
            pen = QPen(neg_color, 2)
            line1 = QGraphicsLineItem(x - 4, y - 4, x + 4, y + 4)
            line1.setPen(pen)
            line1.setZValue(self.ZVAL_SAMPLE)
            line2 = QGraphicsLineItem(x - 4, y + 4, x + 4, y - 4)
            line2.setPen(pen)
            line2.setZValue(self.ZVAL_SAMPLE)
            self._scene.addItem(line1)
            self._sample_items.append(line1)
            lm.add_item("layer_sample_points", line1)  # ★ 注册
            self._scene.addItem(line2)
            self._sample_items.append(line2)
            lm.add_item("layer_sample_points", line2)  # ★ 注册

    def _add_positive_sample(self, x: int, y: int):
        """添加道路正样本（绿点）- 使用全局像素坐标"""
        # 转换为全局像素坐标
        x_global, y_global = self._layer_manager.preview_to_global(x, y)
        print(f"[DEBUG][Canvas] add positive sample: preview=({x},{y}) -> global=({x_global},{y_global})")
        if self._history_push_callback:
            self._history_push_callback("positive_sample")
        # 保存全局坐标
        self._positive_points.append((x_global, y_global))
        # 在预览图上显示需要转换回预览坐标
        dot = QGraphicsEllipseItem(x - 4, y - 4, 8, 8)
        dot.setBrush(QColor(80, 250, 123))
        dot.setPen(QColor(0, 0, 0, 0))
        dot.setZValue(self.ZVAL_SAMPLE)
        self._sample_items.append(dot)
        self._scene.addItem(dot)
        self._layer_manager.add_item("layer_sample_points", dot)  # ★ 注册
        self.viewport().update()
        self.sample_points_changed.emit(len(self._positive_points), len(self._negative_points))

    def _add_negative_sample(self, x: int, y: int):
        """添加非道路负样本（红叉）- 使用全局像素坐标"""
        # 转换为全局像素坐标
        x_global, y_global = self._layer_manager.preview_to_global(x, y)
        print(f"[DEBUG][Canvas] add negative sample: preview=({x},{y}) -> global=({x_global},{y_global})")
        if self._history_push_callback:
            self._history_push_callback("negative_sample")
        # 保存全局坐标
        self._negative_points.append((x_global, y_global))
        pen = QPen(QColor(255, 85, 85), 2)
        line1 = QGraphicsLineItem(x - 4, y - 4, x + 4, y + 4)
        line1.setPen(pen)
        line1.setZValue(self.ZVAL_SAMPLE)
        line2 = QGraphicsLineItem(x - 4, y + 4, x + 4, y - 4)
        line2.setPen(pen)
        line2.setZValue(self.ZVAL_SAMPLE)
        self._sample_items.append(line1)
        self._sample_items.append(line2)
        self._scene.addItem(line1)
        self._scene.addItem(line2)
        self._layer_manager.add_item("layer_sample_points", line1)  # ★ 注册
        self._layer_manager.add_item("layer_sample_points", line2)  # ★ 注册
        self.viewport().update()
        self.sample_points_changed.emit(len(self._positive_points), len(self._negative_points))

    # ===================================================================
    # ROI 绘制（支持多个独立多边形）
    # ===================================================================

    def _add_roi_point(self, x: int, y: int):
        """添加当前 ROI 顶点；x/y 是 scene 坐标，数据保存为原图像素。"""
        gx, gy = self._log_region_coordinate(x, y)
        pt = QPointF(gx, gy)
        self._roi_points.append(pt)

        dot = QGraphicsEllipseItem(x - self.ROI_POINT_RADIUS, y - self.ROI_POINT_RADIUS,
                                    self.ROI_POINT_RADIUS * 2, self.ROI_POINT_RADIUS * 2)
        dot.setBrush(QBrush(self.ROI_COLOR))
        dot.setPen(QPen(self.ROI_COLOR))
        dot.setZValue(self.DEBUG_ROI_DOT_Z)
        dot.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._roi_dots.append(dot)
        self._scene.addItem(dot)
        self._layer_manager.add_item("layer_roi", dot)  # ★ 注册

        self._update_roi_polygon_preview()
        self._scene.update()
        self.viewport().update()

    def _update_roi_polygon_preview(self):
        """根据当前顶点重建蓝色多边形预览 (z=9997, 虚线+半透明填充)"""
        if self._roi_polygon_preview:
            self._scene.removeItem(self._roi_polygon_preview)
            self._roi_polygon_preview = None

        n = len(self._roi_points)
        if n < 2:
            return
        if n == 2:
            p1 = self._global_point_to_scene(self._roi_points[0])
            p2 = self._global_point_to_scene(self._roi_points[1])
            line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
            pen = QPen(self.ROI_COLOR, 2, Qt.PenStyle.DashLine)
            line.setPen(pen)
            line.setZValue(self.DEBUG_ROI_PREVIEW_Z)
            line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._roi_polygon_preview = line
        else:
            polygon = self._global_polygon_to_scene(QPolygonF(self._roi_points))
            poly_item = QGraphicsPolygonItem(polygon)
            poly_item.setPen(QPen(self.ROI_COLOR, 2, Qt.PenStyle.DashLine))
            poly_item.setBrush(QBrush(self.ROI_FILL_COLOR))
            poly_item.setZValue(self.DEBUG_ROI_PREVIEW_Z)
            poly_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._roi_polygon_preview = poly_item
        self._scene.addItem(self._roi_polygon_preview)
        self._layer_manager.add_item("layer_roi", self._roi_polygon_preview)  # ★ 注册

    def _update_roi_temp_line(self, x: int, y: int):
        """末点到鼠标的临时虚线"""
        self._clear_roi_temp_line()
        if not self._roi_points:
            return
        last = self._global_point_to_scene(self._roi_points[-1])
        line = QGraphicsLineItem(last.x(), last.y(), x, y)
        pen = QPen(self.ROI_COLOR, 1.5, Qt.PenStyle.DashLine)
        pen.setDashPattern([4, 4])
        line.setPen(pen)
        line.setZValue(self.DEBUG_ROI_PREVIEW_Z)
        line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._roi_temp_line = line
        self._scene.addItem(line)
        self._layer_manager.add_item("layer_roi", line)  # ★ 注册

    def _clear_roi_temp_line(self):
        if self._roi_temp_line:
            self._scene.removeItem(self._roi_temp_line)
            self._roi_temp_line = None

    def _finalize_roi(self):
        """闭合当前 ROI，保存到多边形列表，允许继续画下一个"""
        if len(self._roi_points) < 3:
            return

        # ★ 推入全局撤销
        if self._history_push_callback:
            self._history_push_callback("roi_add")

        polygon = QPolygonF(self._roi_points)  # original image pixel coordinates
        self._roi_polygons.append(polygon)
        self._roi_regions.append(PolygonRegion.create(
            "roi", [(point.x(), point.y()) for point in polygon]
        ))

        poly_item = QGraphicsPolygonItem(self._global_polygon_to_scene(polygon))
        poly_item.setBrush(QBrush(self.ROI_FILL_COLOR))
        poly_item.setPen(QPen(self.ROI_COLOR, 2))
        poly_item.setZValue(self.DEBUG_ROI_FINAL_Z)
        poly_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._roi_items.append(poly_item)
        self._scene.addItem(poly_item)
        self._layer_manager.add_item("layer_roi", poly_item)  # ★ 注册

        self._clear_current_roi_drawing()
        self._scene.update()
        self.viewport().update()
        self.tool_interaction.emit("regions_changed", {
            "roi": len(self._roi_regions), "ignore": len(self._ignore_regions)
        })

    def add_roi_from_image_points(
        self,
        points: list[tuple[float, float]],
        *,
        push_history: bool = True,
    ) -> bool:
        """以原图像素坐标添加一个 ROI 多边形（用于“当前视野作为 ROI”等）。"""
        if len(points) < 3:
            return False
        if push_history and self._history_push_callback:
            self._history_push_callback("roi_add")

        polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in points])
        self._roi_polygons.append(polygon)
        self._roi_regions.append(PolygonRegion.create("roi", points))

        poly_item = QGraphicsPolygonItem(self._global_polygon_to_scene(polygon))
        poly_item.setBrush(QBrush(self.ROI_FILL_COLOR))
        poly_item.setPen(QPen(self.ROI_COLOR, 2))
        poly_item.setZValue(self.DEBUG_ROI_FINAL_Z)
        poly_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._roi_items.append(poly_item)
        self._scene.addItem(poly_item)
        self._layer_manager.add_item("layer_roi", poly_item)

        self._scene.update()
        self.viewport().update()
        self.tool_interaction.emit("regions_changed", {
            "roi": len(self._roi_regions), "ignore": len(self._ignore_regions)
        })
        return True

    def get_visible_image_rect(self) -> tuple[float, float, float, float]:
        """将当前视口可见区域转换为原图像素矩形 (x0, y0, x1, y1)。"""
        if not self._layer_manager.has_image():
            return (0.0, 0.0, 0.0, 0.0)
        scene_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        corners = [
            self.scene_to_image_f(scene_rect.left(), scene_rect.top()),
            self.scene_to_image_f(scene_rect.right(), scene_rect.top()),
            self.scene_to_image_f(scene_rect.right(), scene_rect.bottom()),
            self.scene_to_image_f(scene_rect.left(), scene_rect.bottom()),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        iw, ih = self._layer_manager.original_size
        if iw <= 0 or ih <= 0:
            iw, ih = self._layer_manager.image_size
        x0 = max(0.0, min(xs))
        y0 = max(0.0, min(ys))
        x1 = min(float(iw), max(xs))
        y1 = min(float(ih), max(ys))
        if x1 <= x0 or y1 <= y0:
            return (0.0, 0.0, float(iw), float(ih))
        return (x0, y0, x1, y1)

    def show_roi_tile_overlay(self, tile_entries: list[dict]) -> None:
        """显示 ROI 覆盖 tile 分类边框。category: need_infer/cached/skipped_black/outside."""
        self.clear_roi_tile_overlay()
        style_map = {
            "need_infer": (QColor(255, 165, 0), 2.0, Qt.PenStyle.SolidLine),
            "cached": (QColor(80, 250, 123), 2.0, Qt.PenStyle.SolidLine),
            "skipped_black": (QColor(120, 120, 120), 1.5, Qt.PenStyle.DashLine),
            "roi_covered": (QColor(68, 153, 255), 1.0, Qt.PenStyle.DotLine),
        }
        for entry in tile_entries:
            tile = entry.get("tile")
            category = entry.get("category", "roi_covered")
            if not tile or len(tile) != 4:
                continue
            x0, y0, x1, y1 = tile
            sx0, sy0 = self.image_to_scene(float(x0), float(y0))
            sx1, sy1 = self.image_to_scene(float(x1), float(y1))
            w, h = sx1 - sx0, sy1 - sy0
            color, width, style = style_map.get(
                category, style_map["roi_covered"]
            )
            rect_item = QGraphicsRectItem(sx0, sy0, w, h)
            rect_item.setPen(QPen(color, width, style))
            rect_item.setBrush(QBrush(QColor(0, 0, 0, 0)))
            rect_item.setZValue(self.DEBUG_ROI_TILE_Z)
            rect_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._scene.addItem(rect_item)
            self._roi_tile_overlay_items.append(rect_item)
            self._layer_manager.add_item("layer_roi", rect_item)
        self._scene.update()
        self.viewport().update()

    def clear_roi_tile_overlay(self) -> None:
        """清除 ROI tile 覆盖可视化边框。"""
        for item in self._roi_tile_overlay_items:
            _safe_remove_scene_item(self._scene, item)
        self._roi_tile_overlay_items.clear()
        self._scene.update()
        self.viewport().update()

    # ===================================================================
    # 主路种子线（大图主路修复约束）
    # ===================================================================
    SEED_COLOR = QColor(255, 220, 40)   # 黄/金，中心线
    SEED_Z = 9996
    RIBBON_Z = 9995

    def set_seed_draw_mode(self, mode: str):
        """freehand | two_point | polyline"""
        self._cancel_seed_in_progress(push_history=False)
        mode = str(mode or "freehand")
        if mode not in {"freehand", "two_point", "polyline"}:
            mode = "freehand"
        self._seed_draw_mode = mode

    def set_seed_width_settings(
        self,
        *,
        width_mode: str = "normal",
        road_width_m: float = 8.0,
        road_radius_px: Optional[float] = None,
        gsd_m_per_px: Optional[float] = None,
        continuous_two_point: bool = True,
    ):
        self._seed_width_mode = str(width_mode or "normal")
        self._seed_road_width_m = float(road_width_m)
        self._seed_road_radius_px = (
            float(road_radius_px) if road_radius_px is not None and float(road_radius_px) > 0 else None
        )
        self._seed_gsd_m_per_px = (
            float(gsd_m_per_px) if gsd_m_per_px is not None and float(gsd_m_per_px) > 0 else None
        )
        self._seed_continuous_two_point = bool(continuous_two_point)

    def set_seed_snap_candidates(self, points: list):
        """Image-pixel candidates for endpoint snap (seed ends / graph / tasks)."""
        out = []
        for p in points or []:
            if p is None or len(p) < 2:
                continue
            out.append((float(p[0]), float(p[1])))
        self._seed_snap_candidates = out

    def _seed_pen(self) -> QPen:
        pen = QPen(self.SEED_COLOR, 3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        pen.setCosmetic(True)
        return pen

    def _clear_seed_temp_preview(self):
        if self._seed_temp_line_item is not None:
            _safe_remove_scene_item(self._scene, self._seed_temp_line_item)
            self._seed_temp_line_item = None
        for item in self._seed_temp_dot_items:
            _safe_remove_scene_item(self._scene, item)
        self._seed_temp_dot_items.clear()

    def _cancel_seed_in_progress(self, *, push_history: bool = False):
        """Esc：取消当前未完成的两点/折线/自由绘。"""
        self._clear_seed_temp_preview()
        for item in self._seed_current_items:
            _safe_remove_scene_item(self._scene, item)
        self._seed_current_items.clear()
        self._seed_current_points.clear()
        self._seed_drawing = False
        self._seed_two_point_start = None
        self._seed_polyline_active = False
        self._scene.update()
        self.viewport().update()

    def _snap_image_point(self, gx: float, gy: float, *, shift_constrain_from=None):
        from roadnet.main_road_seed import apply_angle_constraint, snap_point_to_candidates
        x, y = float(gx), float(gy)
        # endpoint / graph / task snap
        x, y, _ = snap_point_to_candidates(
            x, y, self._seed_snap_candidates, self._seed_endpoint_snap_px
        )
        # also snap to existing seed endpoints
        ends = []
        for stroke in self._main_road_seed_strokes:
            pts = stroke.get("points") if isinstance(stroke, dict) else None
            if not pts and not isinstance(stroke, dict):
                from roadnet.main_road_seed import points_from_stroke
                pl = points_from_stroke(stroke)
                if pl:
                    ends.append(pl[0])
                    ends.append(pl[-1])
            elif pts:
                ends.append((float(pts[0]["x"]), float(pts[0]["y"])))
                ends.append((float(pts[-1]["x"]), float(pts[-1]["y"])))
        x, y, _ = snap_point_to_candidates(x, y, ends, self._seed_endpoint_snap_px)
        if shift_constrain_from is not None:
            x, y = apply_angle_constraint(shift_constrain_from, (x, y))
        return x, y

    def _seed_press(self, sx: int, sy: int, *, modifiers=None, double_click: bool = False):
        """按模式分发：freehand 拖拽 / two_point 点击 / polyline 点击。"""
        gx, gy = self.scene_to_image_f(sx, sy)
        mode = getattr(self, "_seed_draw_mode", "freehand")
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier) if modifiers else False

        if mode == "two_point":
            self._seed_two_point_click(gx, gy, sx, sy, shift=shift)
            return
        if mode == "polyline":
            self._seed_polyline_click(gx, gy, sx, sy, shift=shift, finish=double_click)
            return

        # freehand: press-drag
        self._seed_drawing = True
        self._seed_current_points = [(gx, gy)]
        self._last_seed_scene = (sx, sy)
        dot = QGraphicsEllipseItem(sx - 3, sy - 3, 6, 6)
        dot.setBrush(QBrush(self.SEED_COLOR))
        dot.setPen(QPen(self.SEED_COLOR))
        dot.setZValue(self.SEED_Z)
        dot.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._scene.addItem(dot)
        self._layer_manager.add_item("layer_main_road_seed", dot)
        self._seed_current_items.append(dot)
        self.tool_interaction.emit("seed_status", {"message": "正在绘制主路种子线…"})

    def _seed_two_point_click(self, gx, gy, sx, sy, *, shift: bool = False):
        if self._seed_two_point_start is None:
            gx, gy = self._snap_image_point(gx, gy)
            self._seed_two_point_start = (gx, gy)
            self._seed_current_points = [(gx, gy)]
            self._clear_seed_temp_preview()
            sp = self._global_point_to_scene(QPointF(gx, gy))
            dot = QGraphicsEllipseItem(sp.x() - 4, sp.y() - 4, 8, 8)
            dot.setBrush(QBrush(QColor(255, 80, 80)))
            dot.setPen(QPen(QColor(255, 255, 255), 1))
            dot.setZValue(self.SEED_Z + 1)
            dot.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._scene.addItem(dot)
            self._layer_manager.add_item("layer_main_road_seed", dot)
            self._seed_temp_dot_items.append(dot)
            self.tool_interaction.emit("seed_status", {"message": "请选择主路线终点"})
            return

        start = self._seed_two_point_start
        gx, gy = self._snap_image_point(
            gx, gy, shift_constrain_from=start if shift else None
        )
        self._commit_seed_stroke(
            [start, (gx, gy)],
            source="two_point_click",
            stroke_type="line",
        )
        self._seed_two_point_start = None
        self._clear_seed_temp_preview()
        self._seed_current_points = []
        if not self._seed_continuous_two_point:
            self.tool_interaction.emit("seed_status", {
                "message": "已生成主路种子线（两点模式结束）",
                "exit_tool": True,
            })
        else:
            self.tool_interaction.emit("seed_status", {
                "message": "请选择主路线起点",
            })

    def _seed_polyline_click(self, gx, gy, sx, sy, *, shift: bool = False, finish: bool = False):
        constrain_from = self._seed_current_points[-1] if (shift and self._seed_current_points) else None
        gx, gy = self._snap_image_point(gx, gy, shift_constrain_from=constrain_from)
        if not self._seed_polyline_active:
            self._seed_polyline_active = True
            self._seed_current_points = [(gx, gy)]
            self._clear_seed_temp_preview()
            sp = self._global_point_to_scene(QPointF(gx, gy))
            dot = QGraphicsEllipseItem(sp.x() - 4, sp.y() - 4, 8, 8)
            dot.setBrush(QBrush(QColor(80, 200, 255)))
            dot.setPen(QPen(QColor(255, 255, 255), 1))
            dot.setZValue(self.SEED_Z + 1)
            dot.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._scene.addItem(dot)
            self._layer_manager.add_item("layer_main_road_seed", dot)
            self._seed_temp_dot_items.append(dot)
            self.tool_interaction.emit("seed_status", {
                "message": "多点主路线：继续点击加点，双击/右键结束"
            })
            return

        if finish:
            self._finish_polyline_seed()
            return

        prev = self._seed_current_points[-1]
        self._seed_current_points.append((gx, gy))
        sp0 = self._global_point_to_scene(QPointF(prev[0], prev[1]))
        sp1 = self._global_point_to_scene(QPointF(gx, gy))
        line = QGraphicsLineItem(sp0.x(), sp0.y(), sp1.x(), sp1.y())
        line.setPen(self._seed_pen())
        line.setZValue(self.SEED_Z)
        line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._scene.addItem(line)
        self._layer_manager.add_item("layer_main_road_seed", line)
        self._seed_current_items.append(line)
        dot = QGraphicsEllipseItem(sp1.x() - 3, sp1.y() - 3, 6, 6)
        dot.setBrush(QBrush(self.SEED_COLOR))
        dot.setPen(QPen(self.SEED_COLOR))
        dot.setZValue(self.SEED_Z + 1)
        dot.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._scene.addItem(dot)
        self._layer_manager.add_item("layer_main_road_seed", dot)
        self._seed_temp_dot_items.append(dot)

    def _finish_polyline_seed(self):
        if self._seed_polyline_active and len(self._seed_current_points) >= 2:
            # keep points before cancel clears them
            pts = list(self._seed_current_points)
            self._clear_seed_temp_preview()
            for item in self._seed_current_items:
                _safe_remove_scene_item(self._scene, item)
            self._seed_current_items.clear()
            self._seed_current_points.clear()
            self._seed_polyline_active = False
            self._commit_seed_stroke(pts, source="polyline_click", stroke_type="polyline")
            self.tool_interaction.emit("seed_status", {"message": "已结束多点主路线"})
            return
        self._cancel_seed_in_progress(push_history=False)
        self.tool_interaction.emit("seed_status", {"message": "已取消多点主路线"})

    def _seed_update_rubber_band(self, sx: int, sy: int, *, shift: bool = False):
        """鼠标移动时更新临时线（两点 / 折线）。"""
        mode = getattr(self, "_seed_draw_mode", "freehand")
        gx, gy = self.scene_to_image_f(sx, sy)
        if mode == "two_point" and self._seed_two_point_start is not None:
            start = self._seed_two_point_start
            gx, gy = self._snap_image_point(
                gx, gy, shift_constrain_from=start if shift else None
            )
            sp0 = self._global_point_to_scene(QPointF(start[0], start[1]))
            sp1 = self._global_point_to_scene(QPointF(gx, gy))
            if self._seed_temp_line_item is None:
                self._seed_temp_line_item = QGraphicsLineItem()
                pen = QPen(QColor(255, 220, 40), 2, Qt.PenStyle.DashLine)
                pen.setCosmetic(True)
                self._seed_temp_line_item.setPen(pen)
                self._seed_temp_line_item.setZValue(self.SEED_Z + 2)
                self._seed_temp_line_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                self._scene.addItem(self._seed_temp_line_item)
                self._layer_manager.add_item("layer_main_road_seed", self._seed_temp_line_item)
            self._seed_temp_line_item.setLine(sp0.x(), sp0.y(), sp1.x(), sp1.y())
            return
        if mode == "polyline" and self._seed_polyline_active and self._seed_current_points:
            prev = self._seed_current_points[-1]
            gx, gy = self._snap_image_point(
                gx, gy, shift_constrain_from=prev if shift else None
            )
            sp0 = self._global_point_to_scene(QPointF(prev[0], prev[1]))
            sp1 = self._global_point_to_scene(QPointF(gx, gy))
            if self._seed_temp_line_item is None:
                self._seed_temp_line_item = QGraphicsLineItem()
                pen = QPen(QColor(80, 200, 255), 2, Qt.PenStyle.DashLine)
                pen.setCosmetic(True)
                self._seed_temp_line_item.setPen(pen)
                self._seed_temp_line_item.setZValue(self.SEED_Z + 2)
                self._seed_temp_line_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                self._scene.addItem(self._seed_temp_line_item)
                self._layer_manager.add_item("layer_main_road_seed", self._seed_temp_line_item)
            self._seed_temp_line_item.setLine(sp0.x(), sp0.y(), sp1.x(), sp1.y())

    def _seed_move(self, sx: int, sy: int):
        """自由绘拖动加点。"""
        if not self._seed_drawing:
            return
        gx, gy = self.scene_to_image_f(sx, sy)
        if self._seed_current_points:
            lx, ly = self._last_seed_scene
            if abs(sx - lx) < 2 and abs(sy - ly) < 2:
                return
        self._seed_current_points.append((gx, gy))
        lx, ly = self._last_seed_scene
        line = QGraphicsLineItem(lx, ly, sx, sy)
        line.setPen(self._seed_pen())
        line.setZValue(self.SEED_Z)
        line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._scene.addItem(line)
        self._layer_manager.add_item("layer_main_road_seed", line)
        self._seed_current_items.append(line)
        self._last_seed_scene = (sx, sy)

    def _finalize_seed_stroke(self):
        """结束自由绘一笔。"""
        if not self._seed_drawing:
            return
        self._seed_drawing = False
        points = list(self._seed_current_points)
        self._seed_current_points = []
        # 清除临时 items（将由 commit 重建）
        for item in self._seed_current_items:
            _safe_remove_scene_item(self._scene, item)
        self._seed_current_items = []
        if len(points) >= 2:
            self._commit_seed_stroke(points, source="freehand_drag", stroke_type="polyline")
        self._scene.update()
        self.viewport().update()

    def _commit_seed_stroke(self, points, *, source: str, stroke_type: str = "polyline"):
        from roadnet.main_road_seed import make_seed_stroke, next_seed_id
        pts = [(float(p[0]), float(p[1])) for p in points]
        if len(pts) < 2:
            return None
        if self._history_push_callback:
            self._history_push_callback("main_road_seed_add")
        sid = next_seed_id(self._main_road_seed_strokes)
        stroke = make_seed_stroke(
            pts,
            stroke_id=sid,
            road_width_m=self._seed_road_width_m,
            road_radius_px=self._seed_road_radius_px,
            gsd_m_per_px=self._seed_gsd_m_per_px,
            mode=self._seed_width_mode,
            source=source,
        )
        stroke["type"] = "line" if len(pts) == 2 else stroke_type
        self._main_road_seed_strokes.append(stroke)
        self._render_seed_stroke_graphics(stroke)
        self.refresh_road_ribbon_preview()
        self.tool_interaction.emit("seed_changed", {
            "count": len(self._main_road_seed_strokes),
            "last_id": sid,
            "message": f"已生成主路种子线 {sid}",
        })
        return stroke

    def _render_seed_stroke_graphics(self, stroke: dict):
        from roadnet.main_road_seed import points_from_stroke
        pts = points_from_stroke(stroke)
        if len(pts) < 1:
            return
        scene_pts = [self._global_point_to_scene(QPointF(x, y)) for x, y in pts]
        for i in range(1, len(scene_pts)):
            line = QGraphicsLineItem(
                scene_pts[i - 1].x(), scene_pts[i - 1].y(),
                scene_pts[i].x(), scene_pts[i].y(),
            )
            line.setPen(self._seed_pen())
            line.setZValue(self.SEED_Z)
            line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            tip = (
                f"{stroke.get('id')}  "
                f"w={stroke.get('road_width_m')}m  "
                f"r={stroke.get('road_radius_px'):.1f}px"
            )
            line.setToolTip(tip)
            self._scene.addItem(line)
            self._layer_manager.add_item("layer_main_road_seed", line)
            self._main_road_seed_items.append(line)
        for sp in (scene_pts[0], scene_pts[-1]):
            dot = QGraphicsEllipseItem(sp.x() - 4, sp.y() - 4, 8, 8)
            dot.setBrush(QBrush(QColor(255, 80, 200)))
            dot.setPen(QPen(QColor(255, 255, 255), 1))
            dot.setZValue(self.SEED_Z + 1)
            dot.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            dot.setToolTip(str(stroke.get("id") or ""))
            self._scene.addItem(dot)
            self._layer_manager.add_item("layer_main_road_seed", dot)
            self._main_road_seed_items.append(dot)

    def get_main_road_seed_strokes(self) -> list:
        """兼容旧接口：返回 [[(x,y),...], ...]。"""
        from roadnet.main_road_seed import points_from_stroke
        return [points_from_stroke(s) for s in self._main_road_seed_strokes]

    def get_main_road_seed_stroke_dicts(self) -> list:
        return [dict(s) if isinstance(s, dict) else s for s in self._main_road_seed_strokes]

    def get_main_road_seed_count(self) -> int:
        return len(self._main_road_seed_strokes)

    def set_main_road_seed_stroke_dicts(self, strokes: list, *, emit: bool = True):
        """替换全部种子线并重绘（用于历史恢复 / 加载）。"""
        self.clear_main_road_seeds(push_history=False)
        from roadnet.main_road_seed import normalize_stroke_list
        for stroke in normalize_stroke_list(strokes):
            self._main_road_seed_strokes.append(stroke)
            self._render_seed_stroke_graphics(stroke)
        self.refresh_road_ribbon_preview()
        if emit:
            self.tool_interaction.emit("seed_changed", {
                "count": len(self._main_road_seed_strokes)
            })

    def add_main_road_seed_stroke(self, points_image: list, **kwargs) -> bool:
        """以原图像素坐标程序化添加一笔种子线。"""
        pts = [(float(x), float(y)) for x, y in points_image]
        if len(pts) < 1:
            return False
        if len(pts) == 1:
            pts = [pts[0], pts[0]]
        source = kwargs.get("source", "programmatic")
        self._commit_seed_stroke(pts, source=source, stroke_type="polyline")
        return True

    def undo_last_seed_stroke(self) -> bool:
        if not self._main_road_seed_strokes:
            return False
        if self._history_push_callback:
            self._history_push_callback("main_road_seed_undo_last")
        self._main_road_seed_strokes.pop()
        # 全量重绘
        strokes = list(self._main_road_seed_strokes)
        self.clear_main_road_seeds(push_history=False)
        self.set_main_road_seed_stroke_dicts(strokes, emit=True)
        return True

    def clear_main_road_seeds(self, push_history: bool = True):
        """清空所有主路种子线。"""
        if push_history and (self._main_road_seed_strokes or self._seed_current_items) and self._history_push_callback:
            self._history_push_callback("main_road_seed_clear")
        for item in self._main_road_seed_items:
            _safe_remove_scene_item(self._scene, item)
        for item in self._seed_current_items:
            _safe_remove_scene_item(self._scene, item)
        self._main_road_seed_items.clear()
        self._seed_current_items.clear()
        self._main_road_seed_strokes.clear()
        self._seed_current_points.clear()
        self._seed_drawing = False
        self._seed_two_point_start = None
        self._seed_polyline_active = False
        self._clear_seed_temp_preview()
        self.clear_road_ribbon_preview()
        self._scene.update()
        self.viewport().update()
        self.tool_interaction.emit("seed_changed", {"count": 0})

    def refresh_road_ribbon_preview(self):
        """根据当前种子线刷新 road ribbon 预览图层。"""
        from roadnet.main_road_seed import build_road_ribbon_mask
        self.clear_road_ribbon_preview()
        if not self._main_road_seed_strokes:
            return
        # 用原图尺寸估算：从图层或 seed 包围盒
        ow = int(getattr(self._layer_manager, "full_image_width", 0) or 0)
        oh = int(getattr(self._layer_manager, "full_image_height", 0) or 0)
        if ow <= 0 or oh <= 0:
            arr = getattr(self._layer_manager, "full_image_rgb", None)
            if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                oh, ow = arr.shape[:2]
        if ow <= 0 or oh <= 0:
            # fallback: bbox of seeds + padding
            xs, ys = [], []
            from roadnet.main_road_seed import points_from_stroke
            for s in self._main_road_seed_strokes:
                for x, y in points_from_stroke(s):
                    xs.append(x); ys.append(y)
            if not xs:
                return
            pad = 64
            minx, maxx = int(min(xs)) - pad, int(max(xs)) + pad
            miny, maxy = int(min(ys)) - pad, int(max(ys)) + pad
            # draw local ribbon then place
            local = build_road_ribbon_mask(
                (maxy - miny + 1, maxx - minx + 1),
                [
                    {
                        **s,
                        "points": [
                            {"x": p["x"] - minx, "y": p["y"] - miny}
                            for p in s["points"]
                        ],
                    }
                    if isinstance(s, dict) else s
                    for s in self._main_road_seed_strokes
                ],
            )
            self._show_ribbon_rgba(local, origin_xy=(minx, miny), size_wh=(maxx - minx + 1, maxy - miny + 1))
            return
        # Downscale for preview performance on huge images
        max_side = 2048
        scale = 1.0
        if max(ow, oh) > max_side:
            scale = max_side / float(max(ow, oh))
        sw, sh = max(1, int(ow * scale)), max(1, int(oh * scale))
        scaled_strokes = []
        for s in self._main_road_seed_strokes:
            if not isinstance(s, dict):
                continue
            scaled = dict(s)
            scaled["points"] = [
                {"x": float(p["x"]) * scale, "y": float(p["y"]) * scale}
                for p in s.get("points") or []
            ]
            if s.get("road_radius_px"):
                scaled["road_radius_px"] = float(s["road_radius_px"]) * scale
            scaled_strokes.append(scaled)
        ribbon = build_road_ribbon_mask((sh, sw), scaled_strokes)
        self._show_ribbon_rgba(ribbon, origin_xy=(0.0, 0.0), size_wh=(ow, oh), mask_wh=(sw, sh))

    def _show_ribbon_rgba(self, ribbon_mask, *, origin_xy, size_wh, mask_wh=None):
        mask = np.asarray(ribbon_mask)
        if mask.ndim != 2 or not mask.any():
            return
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        sel = mask > 0
        rgba[sel] = (0, 220, 220, 90)  # 半透明青色
        img = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        item = QGraphicsPixmapItem(QPixmap.fromImage(img))
        ox, oy = float(origin_xy[0]), float(origin_xy[1])
        tw, th = float(size_wh[0]), float(size_wh[1])
        mw, mh = (float(mask_wh[0]), float(mask_wh[1])) if mask_wh else (float(w), float(h))
        sx0, sy0 = self.image_to_scene(ox, oy)
        sx1, sy1 = self.image_to_scene(ox + tw, oy + th)
        scale_x = (sx1 - sx0) / mw if mw else 1.0
        scale_y = (sy1 - sy0) / mh if mh else 1.0
        item.setPos(sx0, sy0)
        item.setTransformOriginPoint(0, 0)
        # Prefer uniform if nearly square scale
        item.setScale(scale_x if abs(scale_x - scale_y) < 1e-3 else scale_x)
        if abs(scale_x - scale_y) >= 1e-3:
            from PySide6.QtGui import QTransform
            item.setTransform(QTransform.fromScale(scale_x, scale_y))
            item.setScale(1.0)
        item.setZValue(self.RIBBON_Z)
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        item.setToolTip("Road Ribbon Preview")
        self._scene.addItem(item)
        # Prefer dedicated layer if present, else seed layer
        layer_name = "layer_road_ribbon_preview"
        if layer_name not in getattr(self._layer_manager, "_layers", {}):
            layer_name = "layer_main_road_seed"
        self._layer_manager.add_item(layer_name, item)
        self._ribbon_preview_item = item
        self._scene.update()
        self.viewport().update()

    def clear_road_ribbon_preview(self):
        if self._ribbon_preview_item is not None:
            _safe_remove_scene_item(self._scene, self._ribbon_preview_item)
            self._ribbon_preview_item = None

    def show_corridor_overlay(self, corridor_mask, orig_w: int, orig_h: int):
        """以半透明品红显示主路 corridor（corridor_mask 为任意分辨率的二值掩膜）。"""
        self.clear_corridor_overlay()
        mask = np.asarray(corridor_mask)
        if mask.ndim != 2 or not mask.any():
            return
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        sel = mask > 0
        rgba[sel] = (255, 0, 255, 70)   # RGBA 半透明品红
        img = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        item = QGraphicsPixmapItem(QPixmap.fromImage(img))
        sx0, sy0 = self.image_to_scene(0.0, 0.0)
        sx1, sy1 = self.image_to_scene(float(orig_w), float(orig_h))
        scale_x = (sx1 - sx0) / float(w) if w else 1.0
        scale_y = (sy1 - sy0) / float(h) if h else 1.0
        item.setPos(sx0, sy0)
        item.setScale(scale_x if abs(scale_x - scale_y) < 1e-3 else scale_x)
        item.setZValue(self.SEED_Z - 1)
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._scene.addItem(item)
        self._layer_manager.add_item("layer_main_road_seed", item)
        self._corridor_overlay_item = item
        self._scene.update()
        self.viewport().update()

    def clear_corridor_overlay(self):
        """清除主路 corridor 预览。"""
        if self._corridor_overlay_item is not None:
            _safe_remove_scene_item(self._scene, self._corridor_overlay_item)
            self._corridor_overlay_item = None
            self._scene.update()
            self.viewport().update()

    def _clear_current_roi_drawing(self):
        """清除当前 ROI 绘制临时项（不删已完成的）"""
        self._roi_points.clear()
        for d in self._roi_dots:
            self._scene.removeItem(d)
        self._roi_dots.clear()
        self._clear_roi_temp_line()
        if self._roi_polygon_preview:
            self._scene.removeItem(self._roi_polygon_preview)
            self._roi_polygon_preview = None

    def _undo_roi_point(self):
        """撤销当前 ROI 最后一个顶点"""
        if not self._roi_points:
            return
        self._roi_points.pop()
        if self._roi_dots:
            self._scene.removeItem(self._roi_dots.pop())
        self._clear_roi_temp_line()
        self._update_roi_polygon_preview()
        self._scene.update()
        self.viewport().update()

    def _delete_last_roi(self):
        """删除最后一个已完成 ROI"""
        if not self._roi_items:
            return
        # ★ 推入全局撤销
        if self._history_push_callback:
            self._history_push_callback("roi_delete")
        self._scene.removeItem(self._roi_items.pop())
        self._roi_polygons.pop()
        if self._roi_regions:
            self._roi_regions.pop()
        self._scene.update()
        self.viewport().update()

    def _clear_all_roi(self):
        """清空所有 ROI（进行中 + 已完成）"""
        # ★ 推入全局撤销（仅在确实有数据时）
        if (self._roi_items or self._roi_polygons or self._roi_points) and self._history_push_callback:
            self._history_push_callback("roi_clear_all")
        self.clear_roi_tile_overlay()
        self._clear_current_roi_drawing()
        for item in self._roi_items:
            _safe_remove_scene_item(self._scene, item)
        self._roi_items.clear()
        self._roi_polygons.clear()
        self._roi_regions.clear()
        self._layer_manager.clear_layer_items("layer_roi")  # ★ 解绑
        self._scene.update()
        self.viewport().update()
        self.tool_interaction.emit("regions_changed", {
            "roi": 0, "ignore": len(self._ignore_regions)
        })

    # ===================================================================
    # Ignore 绘制（★ 统一为多边形模式，交互与 ROI 一致）
    # ===================================================================

    def _add_ignore_point(self, x: int, y: int):
        """添加当前 Ignore 顶点；x/y 是 scene 坐标，数据保存为原图像素。"""
        gx, gy = self._log_region_coordinate(x, y)
        pt = QPointF(gx, gy)
        self._ignore_points.append(pt)

        dot = QGraphicsEllipseItem(x - 5, y - 5, 10, 10)
        dot.setBrush(QBrush(self.IGNORE_COLOR))
        dot.setPen(QPen(self.IGNORE_COLOR))
        dot.setZValue(self.DEBUG_IGNORE_PREVIEW_Z)
        dot.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._ignore_dots.append(dot)
        self._scene.addItem(dot)
        self._layer_manager.add_item("layer_ignore", dot)

        self._update_ignore_polygon_preview()
        self._scene.update()
        self.viewport().update()

    def _update_ignore_polygon_preview(self):
        """根据当前顶点重建红色多边形预览 (虚线+半透明填充)"""
        if self._ignore_polygon_preview:
            self._scene.removeItem(self._ignore_polygon_preview)
            self._ignore_polygon_preview = None

        n = len(self._ignore_points)
        if n < 2:
            return
        if n == 2:
            p1 = self._global_point_to_scene(self._ignore_points[0])
            p2 = self._global_point_to_scene(self._ignore_points[1])
            line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
            pen = QPen(self.IGNORE_COLOR, 2, Qt.PenStyle.DashLine)
            line.setPen(pen)
            line.setZValue(self.DEBUG_IGNORE_PREVIEW_Z)
            line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._ignore_polygon_preview = line
        else:
            polygon = self._global_polygon_to_scene(QPolygonF(self._ignore_points))
            poly_item = QGraphicsPolygonItem(polygon)
            poly_item.setPen(QPen(self.IGNORE_COLOR, 2, Qt.PenStyle.DashLine))
            poly_item.setBrush(QBrush(self.IGNORE_FILL_COLOR))
            poly_item.setZValue(self.DEBUG_IGNORE_PREVIEW_Z)
            poly_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._ignore_polygon_preview = poly_item
        self._scene.addItem(self._ignore_polygon_preview)
        self._layer_manager.add_item("layer_ignore", self._ignore_polygon_preview)

    def _update_ignore_temp_line(self, x: int, y: int):
        """末点到鼠标的临时虚线"""
        self._clear_ignore_temp_line()
        if not self._ignore_points:
            return
        last = self._global_point_to_scene(self._ignore_points[-1])
        line = QGraphicsLineItem(last.x(), last.y(), x, y)
        pen = QPen(self.IGNORE_COLOR, 1.5, Qt.PenStyle.DashLine)
        pen.setDashPattern([4, 4])
        line.setPen(pen)
        line.setZValue(self.DEBUG_IGNORE_PREVIEW_Z)
        line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._ignore_temp_line = line
        self._scene.addItem(line)
        self._layer_manager.add_item("layer_ignore", line)

    def _clear_ignore_temp_line(self):
        if self._ignore_temp_line:
            self._scene.removeItem(self._ignore_temp_line)
            self._ignore_temp_line = None

    def _finalize_ignore(self):
        """闭合当前 Ignore 多边形，保存并允许继续画下一个"""
        if len(self._ignore_points) < 3:
            return

        if self._history_push_callback:
            self._history_push_callback("ignore_add")

        polygon = QPolygonF(self._ignore_points)  # original image pixel coordinates
        self._ignore_polygons.append(polygon)
        self._ignore_regions.append(PolygonRegion.create(
            "ignore", [(point.x(), point.y()) for point in polygon]
        ))

        poly_item = QGraphicsPolygonItem(self._global_polygon_to_scene(polygon))
        poly_item.setBrush(QBrush(self.IGNORE_FILL_COLOR))
        poly_item.setPen(QPen(self.IGNORE_COLOR, 2))
        poly_item.setZValue(self.DEBUG_IGNORE_FINAL_Z)
        poly_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._ignore_items.append(poly_item)
        self._scene.addItem(poly_item)
        self._layer_manager.add_item("layer_ignore", poly_item)

        self._clear_current_ignore_drawing()
        self._scene.update()
        self.viewport().update()
        self.tool_interaction.emit("regions_changed", {
            "roi": len(self._roi_regions), "ignore": len(self._ignore_regions)
        })

    def _clear_current_ignore_drawing(self):
        """清除当前 Ignore 绘制临时项（不删已完成的）"""
        self._ignore_points.clear()
        for d in self._ignore_dots:
            self._scene.removeItem(d)
        self._ignore_dots.clear()
        self._clear_ignore_temp_line()
        if self._ignore_polygon_preview:
            self._scene.removeItem(self._ignore_polygon_preview)
            self._ignore_polygon_preview = None

    def _undo_ignore_point(self):
        """撤销当前 Ignore 最后一个顶点"""
        if not self._ignore_points:
            return
        self._ignore_points.pop()
        if self._ignore_dots:
            self._scene.removeItem(self._ignore_dots.pop())
        self._clear_ignore_temp_line()
        self._update_ignore_polygon_preview()
        self._scene.update()
        self.viewport().update()

    def _delete_last_ignore(self):
        """删除最后一个已完成 Ignore 区域"""
        if not self._ignore_items:
            return
        if self._history_push_callback:
            self._history_push_callback("ignore_delete")
        self._scene.removeItem(self._ignore_items.pop())
        self._ignore_polygons.pop()
        if self._ignore_regions:
            self._ignore_regions.pop()
        self._scene.update()
        self.viewport().update()

    def _clear_all_ignore(self):
        """清空所有 Ignore（进行中 + 已完成）"""
        if (self._ignore_items or self._ignore_polygons or self._ignore_points) and self._history_push_callback:
            self._history_push_callback("ignore_clear_all")
        self._clear_current_ignore_drawing()
        for item in self._ignore_items:
            _safe_remove_scene_item(self._scene, item)
        self._ignore_items.clear()
        self._ignore_polygons.clear()
        self._ignore_regions.clear()
        self._layer_manager.clear_layer_items("layer_ignore")
        self._scene.update()
        self.viewport().update()
        self.tool_interaction.emit("regions_changed", {
            "roi": len(self._roi_regions), "ignore": 0
        })

    def get_ignore_polygons(self) -> list:
        """返回所有已完成 Ignore 多边形数据（QPolygonF 列表）。

        兼容旧矩形：将 _ignore_rects_deprecated 中的矩形也转为 QPolygonF。
        """
        self._sync_region_models()
        result = list(self._ignore_polygons)
        # 兼容旧矩形数据（加载项目时填充）
        for r in self._ignore_rects_deprecated:
            try:
                if hasattr(r, '__iter__') and len(r) == 4:
                    x, y, w, h = float(r[0]), float(r[1]), float(r[2]), float(r[3])
                    poly = QPolygonF([
                        QPointF(x, y), QPointF(x + w, y),
                        QPointF(x + w, y + h), QPointF(x, y + h),
                    ])
                    result.append(poly)
            except (TypeError, ValueError):
                pass
        return result

    # ===================================================================
    # 图层渲染
    # ===================================================================

    def refresh_scene(self):
        """完全重建场景（保留已完成的 ROI/Ignore 引用，重建后重新添加）"""
        # ★ 保存已完成项的数据（scene.clear() 后 item 失效）
        roi_finalized_polygons = list(self._roi_polygons)
        ignore_finalized_polygons = list(self._ignore_polygons)

        self._scene.clear()
        self._overlay_items.clear()
        self._image_item = None
        self._sample_items.clear()
        self._roi_dots.clear()
        self._roi_temp_line = None
        self._roi_polygon_preview = None
        self._roi_items.clear()
        self._ignore_dots.clear()
        self._ignore_temp_line = None
        self._ignore_polygon_preview = None
        self._ignore_items.clear()

        lm = self._layer_manager

        # 1. 基础影像层
        if lm.has_image() and lm.image_pixmap:
            self._image_item = QGraphicsPixmapItem(lm.image_pixmap)
            self._image_item.setZValue(self.ZVAL_IMAGE)
            self._image_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._scene.addItem(self._image_item)

            w, h = lm.image_size
            margin = 20
            self._scene.setSceneRect(QRectF(-margin, -margin, w + 2 * margin, h + 2 * margin))

            # 2. 各叠加图层（使用 LayerManager 的 zValue 映射）
            for name, layer in lm.layers().items():
                pixmap = lm.get_layer_pixmap(name)
                if pixmap:
                    item = QGraphicsPixmapItem(pixmap)
                    # 使用 LayerManager 定义的 z 值（回退到增序）
                    z_value = lm.get_layer_zvalue(name)
                    if z_value == 0 and name != "layer_image":
                        z_value = self.ZVAL_MASK
                    item.setZValue(z_value)
                    item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                    self._scene.addItem(item)
                    self._overlay_items[name] = item

            # 3. 重建采样点
            self._redraw_samples()

            # ★ 4. 重新添加已完成的 ROI（z=9996）
            for polygon in roi_finalized_polygons:
                poly_item = QGraphicsPolygonItem(self._global_polygon_to_scene(polygon))
                poly_item.setBrush(QBrush(self.ROI_FILL_COLOR))
                poly_item.setPen(QPen(self.ROI_COLOR, 2))
                poly_item.setZValue(self.DEBUG_ROI_FINAL_Z)
                poly_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                self._roi_items.append(poly_item)
                self._scene.addItem(poly_item)
                lm.add_item("layer_roi", poly_item)  # ★ 注册

            # ★ 5. 重新添加已完成的 Ignore（z=9996，统一多边形）
            for polygon in ignore_finalized_polygons:
                poly_item = QGraphicsPolygonItem(self._global_polygon_to_scene(polygon))
                poly_item.setBrush(QBrush(self.IGNORE_FILL_COLOR))
                poly_item.setPen(QPen(self.IGNORE_COLOR, 2))
                poly_item.setZValue(self.DEBUG_IGNORE_FINAL_Z)
                poly_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                self._ignore_items.append(poly_item)
                self._scene.addItem(poly_item)
                lm.add_item("layer_ignore", poly_item)  # ★ 注册

        else:
            hint_font = QFont("Segoe UI", 14)
            hint = self._scene.addText(
                "RoadNet Studio\n\n"
                "Ctrl+O  打开影像\n"
                "Ctrl+0  适应窗口\n"
                "Ctrl+=  放大  |  Ctrl+-  缩小\n"
                "鼠标左键拖拽  平移\n"
                "Ctrl+滚轮  缩放",
                hint_font,
            )
            hint.setDefaultTextColor(QColor(140, 150, 180))
            hint.setPos(40, 40)
            hint.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._scene.setSceneRect(QRectF(0, 0, 800, 600))

        self.viewport().update()

    def update_overlay(self, name: str):
        """更新单层叠加"""
        lm = self._layer_manager

        if name in self._overlay_items:
            old = self._overlay_items.pop(name)
            _safe_remove_scene_item(self._scene, old)

        pixmap = lm.get_layer_pixmap(name)
        if pixmap:
            item = QGraphicsPixmapItem(pixmap)
            z_map = {
                "mask":         self.ZVAL_MASK,
                "layer_road_mask": self.ZVAL_MASK,
                # 快速预览必须位于正式 Road Mask 之下，不能覆盖工作 mask
                "layer_preview_segmentation": self.ZVAL_MASK - 10,
                "preview_seg_mask": self.ZVAL_MASK - 10,
                "layer_raw_skeleton": self.ZVAL_MASK + 15,
                "roi":          self.ZVAL_MASK + 5,
                "layer_roi":    self.ZVAL_MASK + 5,
                "ignore":       self.ZVAL_MASK + 10,
                "layer_ignore": self.ZVAL_MASK + 10,
                "edit":         self.ZVAL_MASK + 15,
                "skeleton":     self.ZVAL_MASK + 20,
                "layer_skeleton": self.ZVAL_MASK + 20,
                "layer_skeleton_nodes": self.ZVAL_MASK + 22,
                "draft_graph":  self.ZVAL_MASK + 25,
                "layer_draft_graph": self.ZVAL_MASK + 25,
                "final_graph":  self.ZVAL_MASK + 30,
                "layer_final_graph": self.ZVAL_MASK + 30,
                "planned_path": self.ZVAL_MASK + 35,
                "layer_planned_path": self.ZVAL_MASK + 35,
                "graph":        self.ZVAL_MASK + 30,
                "path":         self.ZVAL_MASK + 35,
                "layer_sample_points": self.ZVAL_MASK + 2,
                "layer_task_points": self.ZVAL_MASK + 28,
            }
            item.setZValue(z_map.get(name, self.ZVAL_MASK))
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._scene.addItem(item)
            self._overlay_items[name] = item

        self.viewport().update()

    def _on_layer_changed(self, name: str):
        """图层数据变化"""
        if name == "image":
            self.refresh_scene()
            self.fit_to_window()
        else:
            self.update_overlay(name)

    def _on_visibility_changed(self, name: str, visible: bool):
        """图层显隐变化"""
        if visible:
            self.update_overlay(name)
        else:
            if name in self._overlay_items:
                item = self._overlay_items.pop(name)
                _safe_remove_scene_item(self._scene, item)
        self.viewport().update()

    # ===================================================================
    # 缩放与平移
    # ===================================================================

    def fit_to_window(self):
        """自适应窗口"""
        if self._image_item:
            self.fitInView(self._image_item, Qt.AspectRatioMode.KeepAspectRatio)
            t = self.transform()
            self._zoom_level = min(abs(t.m11()), abs(t.m22()))
            self.zoom_changed.emit(self._zoom_level)
            self.viewport().update()

    def zoom_in(self):
        """放大（以视图中心）"""
        old_anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._apply_zoom(self.ZOOM_FACTOR)
        self.setTransformationAnchor(old_anchor)

    def zoom_out(self):
        """缩小（以视图中心）"""
        old_anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._apply_zoom(1.0 / self.ZOOM_FACTOR)
        self.setTransformationAnchor(old_anchor)

    def zoom_to_fit(self):
        self.fit_to_window()

    def zoom_to_100(self):
        """缩放到 100%（1:1 像素）"""
        self.resetTransform()
        self._zoom_level = 1.0
        self.zoom_changed.emit(self._zoom_level)
        self.viewport().update()

    def zoom_to_full(self):
        """回到全图"""
        self.fit_to_window()

    def _apply_zoom(self, factor: float):
        new_zoom = self._zoom_level * factor
        if self.MIN_ZOOM <= new_zoom <= self.MAX_ZOOM:
            self.scale(factor, factor)
            self._zoom_level = new_zoom
            self.zoom_changed.emit(self._zoom_level)

    def _is_inside_image(self, pt: QPointF) -> bool:
        """检查 scene 坐标是否在图像范围内"""
        if self._image_item is None:
            return False
        return self._image_item.sceneBoundingRect().contains(pt)

    def resizeEvent(self, event):
        """窗口尺寸变化时自适应（仅当用户未手动缩放时）"""
        super().resizeEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮缩放 — 以鼠标位置为中心；Ctrl+滚轮调整画笔半径"""
        # ★ Ctrl+滚轮：调整画笔半径（mask_refine 或任何工具下都可）
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            step = 1 if delta > 0 else -1
            self._mask_brush_radius = max(1, min(100, self._mask_brush_radius + step))
            # 发射信号通知状态栏
            self.tool_interaction.emit("brush_radius_changed", self._mask_brush_radius)
            event.accept()
            return

        # ★ 始终使用滚轮缩放（不再需要 Ctrl）
        if self.hasFocus() or self.underMouse():
            delta = event.angleDelta().y()
            # 确保以鼠标位置为缩放中心
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            if delta > 0:
                self._apply_zoom(self.ZOOM_FACTOR)
            else:
                self._apply_zoom(1.0 / self.ZOOM_FACTOR)
            event.accept()
        else:
            super().wheelEvent(event)

    # ===================================================================
    # 坐标转换辅助
    # ===================================================================

    def _scene_pos(self, event: QMouseEvent) -> tuple:
        """辅助：获取当前鼠标的场景坐标 (int, int)"""
        pt = self.mapToScene(event.position().toPoint())
        return (int(pt.x()), int(pt.y()))

    def scene_to_display_xy(self, scene_pt: QPointF) -> tuple:
        """将 scene 坐标转换为当前显示/预览图的像素坐标。
        若 image_item 存在，以 image_item 左上角为 (0,0)；
        否则直接用 scene 坐标。
        """
        if self._image_item:
            bx = int(scene_pt.x() - self._image_item.x())
            by = int(scene_pt.y() - self._image_item.y())
        else:
            bx, by = int(scene_pt.x()), int(scene_pt.y())
        return (bx, by)

    def scene_to_image_xy(self, scene_pt: QPointF) -> tuple:
        """将 scene 坐标转换为原始影像像素坐标。"""
        preview_x, preview_y = self.scene_to_display_xy(scene_pt)
        return self._layer_manager.preview_to_global(preview_x, preview_y)

    def scene_to_global_xy(self, scene_pt: QPointF) -> tuple:
        """将 scene 坐标转换为原图全局像素坐标。
        大图模式下：preview_coord / preview_scale
        普通图像：直接返回 scene 坐标
        """
        preview_x, preview_y = self.scene_to_display_xy(scene_pt)
        return self._layer_manager.preview_to_global(preview_x, preview_y)

    def global_to_scene(self, x_global: int, y_global: int) -> tuple:
        """将原图全局像素坐标转换为 scene 坐标"""
        return self.image_to_scene(float(x_global), float(y_global))

    def global_to_scene_f(self, x_global: float, y_global: float) -> tuple:
        """将原图全局像素坐标转换为 scene 坐标（浮点）"""
        return self.image_to_scene(x_global, y_global)

    # ────────────────────────────────────────────────────────────────
    # 规范坐标变换 API（image = original image pixel / global pixel）
    # 所有图层绘制和鼠标交互必须只通过这两个统一入口。
    # ────────────────────────────────────────────────────────────────

    def image_to_scene(self, x: float, y: float) -> Tuple[float, float]:
        """original image pixel → QGraphicsScene 坐标（浮点）。

        这是所有图层绘制的 **唯一标准入口**。
        内部等价于 ``global_to_preview_f``，作用是把原始像素坐标
        映射到 QGraphicsScene 坐标系（也就是预览图像素坐标）。
        """
        return self._layer_manager.image_to_scene(x, y)

    def scene_to_image(self, x: int, y: int) -> Tuple[int, int]:
        """QGraphicsScene 坐标 → original image pixel（整数）。

        这是所有鼠标交互的 **唯一标准入口**。
        """
        return self._layer_manager.scene_to_image(x, y)

    def scene_to_image_f(self, x: float, y: float) -> Tuple[float, float]:
        """QGraphicsScene 坐标 → original image pixel（浮点）"""
        return self._layer_manager.scene_to_image_f(x, y)

    def _in_image_bounds(self, x: int, y: int) -> bool:
        """检查坐标是否在图像范围内"""
        if not self._layer_manager.has_image():
            return False
        w, h = self._layer_manager.image_size
        return 0 <= x < w and 0 <= y < h

    # ===================================================================
    # ★ 调试十字标记（诊断可视化根因）
    # ===================================================================

    DEBUG_MARKER_Z = 9999
    DEBUG_CROSS_SIZE = 12

    # ★ 坐标变换调试开关
    debug_coord_enabled: bool = False

    def add_debug_click_marker(self, scene_pos: QPointF):
        """在 scene_pos 放置一个黄色十字和圆圈，zValue=9999，忽略所有鼠标事件。"""
        x, y = int(scene_pos.x()), int(scene_pos.y())
        s = self.DEBUG_CROSS_SIZE
        scene = self.scene()

        # ── 诊断打印 ──
        img_x, img_y = self.scene_to_image(x, y)
        zoom = self._zoom_level
        print(f"[DEBUG][COORD] scene=({x},{y}) image=({img_x},{img_y}) zoom={zoom:.3f}")
        print(f"[DEBUG][COORD] 验证：image→scene=({self.image_to_scene(img_x, img_y)}) 应≈({x},{y})")
        print(f"[DEBUG][MARKER] scene id={id(scene)} item_count(before)={len(scene.items())}")
        if self._image_item:
            img_rect = self._image_item.sceneBoundingRect()
            print(f"[DEBUG][MARKER] image_item pos=({self._image_item.x():.1f},{self._image_item.y():.1f})")
            print(f"[DEBUG][MARKER] image_item boundingRect={img_rect}")
            in_img = img_rect.contains(scene_pos)
            print(f"[DEBUG][MARKER] point inside image bounding rect: {in_img}")

        # ── 黄色十字 ──
        pen_yellow = QPen(QColor(255, 255, 0), 3)
        pen_yellow.setCosmetic(True)  # ★ 关键：不受缩放影响

        h_line = QGraphicsLineItem(x - s, y, x + s, y)
        h_line.setPen(pen_yellow)
        h_line.setZValue(self.DEBUG_MARKER_Z)
        h_line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        scene.addItem(h_line)

        v_line = QGraphicsLineItem(x, y - s, x, y + s)
        v_line.setPen(pen_yellow)
        v_line.setZValue(self.DEBUG_MARKER_Z)
        v_line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        scene.addItem(v_line)

        # ── 黄色圆圈 ──
        circle = QGraphicsEllipseItem(x - s, y - s, s * 2, s * 2)
        circle.setPen(pen_yellow)
        circle.setZValue(self.DEBUG_MARKER_Z)
        circle.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        scene.addItem(circle)

        # ── 坐标信息文本 ──
        from PySide6.QtWidgets import QGraphicsTextItem
        coord_text = scene.addText(
            f"scene=({x},{y})\nimage=({img_x},{img_y})\nzoom={zoom:.2f}"
        )
        coord_text.setDefaultTextColor(QColor(255, 255, 0))
        coord_text.setPos(x + 15, y - 30)
        coord_text.setZValue(self.DEBUG_MARKER_Z + 1)
        coord_text.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        # ── 确保可见：强制所有 item 为可见 ──
        for item in scene.items():
            if not item.isVisible():
                print(f"[DEBUG][MARKER] WARNING: item is hidden! type={type(item).__name__} z={item.zValue()}")
                item.setVisible(True)

        # ── 强制刷新 ──
        scene.update()
        self.viewport().update()
        print(f"[DEBUG][MARKER] item_count(after)={len(scene.items())}")

    def debug_verify_task_points(self, task_points: list):
        """坐标变换调试：验证所有任务点在当前缩放下的位置一致性。

        对每个任务点打印:
          original_pixel=(px, py)  →  mapped_scene=(sx, sy)
        然后检查 mapped_scene → scene_to_image → 应还原为 original_pixel。
        """
        print("=" * 60)
        print(f"[DEBUG][VERIFY] zoom={self._zoom_level:.3f} 任务点坐标验证")
        print("-" * 60)
        for tp in task_points:
            px = getattr(tp, "pixel_x", None) or getattr(tp, "x", None)
            py = getattr(tp, "pixel_y", None) or getattr(tp, "y", None)
            if px is None or py is None:
                continue
            sx, sy = self.image_to_scene(float(px), float(py))
            rx, ry = self.scene_to_image_f(sx, sy)
            seq = getattr(tp, "seq", "?")
            err = ((rx - float(px)) ** 2 + (ry - float(py)) ** 2) ** 0.5
            status = "OK" if err < 1.0 else f"ERR offset={err:.1f}px"
            print(f"  seq={seq} original_pixel=({px:.1f}, {py:.1f})  "
                  f"mapped_scene=({sx:.1f}, {sy:.1f})  "
                  f"roundtrip=({rx:.1f}, {ry:.1f})  {status}")
        print("=" * 60)

    def debug_toggle_coord_info(self):
        """切换坐标变换调试信息显示"""
        self.debug_coord_enabled = not self.debug_coord_enabled
        state = "ON" if self.debug_coord_enabled else "OFF"
        print(f"[DEBUG][COORD] 坐标变换调试信息: {state}")
        return self.debug_coord_enabled

    # ===================================================================
    # 鼠标事件
    # ===================================================================

    GRAPH_TOOLS = frozenset({
        "graph_add_node", "graph_delete_node", "graph_add_edge",
        "graph_delete_edge", "graph_move_node", "graph_merge_nodes",
        "graph_draw_edge", "graph_local_rebuild", "graph_locate_jump",
    })

    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下"""
        # ★ 空格临时平移模式：左键平移
        if self._space_pan_active:
            fake_event = QMouseEvent(
                event.type(), event.position(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                event.modifiers()
            )
            super().mousePressEvent(fake_event)
            return

        # ★ 首先计算 scene 坐标（QPointF 保留精度）
        scene_pt = self.mapToScene(event.position().toPoint())
        sx, sy = int(scene_pt.x()), int(scene_pt.y())

        btn_name = {Qt.MouseButton.LeftButton: "Left", Qt.MouseButton.MiddleButton: "Middle",
                     Qt.MouseButton.RightButton: "Right"}.get(event.button(), str(event.button()))
        print(f"[DEBUG][Canvas] mousePress tool={self._current_tool} button={btn_name} scene=({sx},{sy})")

        # ★ 调试标记：使用新标志 debug_coord_enabled 或旧标志
        if getattr(self, "debug_coord_enabled", False) or getattr(self, "debug_click_marker_enabled", False):
            self.add_debug_click_marker(scene_pt)

        # ---- 中键拖拽平移 ----
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            fake_event = QMouseEvent(
                event.type(), event.position(),
                Qt.MouseButton.LeftButton,
                event.buttons() | Qt.MouseButton.LeftButton,
                event.modifiers()
            )
            super().mousePressEvent(fake_event)
            return

        # ---- 平移工具：用 QGraphicsView 内置 ScrollHandDrag（左键） ----
        if self._current_tool == "pan":
            super().mousePressEvent(event)
            return

        # ---- 其他工具：NoDrag 模式下自行处理 ----
        if not self._in_image_bounds(sx, sy):
            print(f"[DEBUG][Canvas] click outside image bounds, skip tool action")
            return

        # ---- 按工具分发（使用 scene 坐标绘制 GUI，必要时转 image 坐标给算法） ----
        tool = self._current_tool
        if tool == "positive_sample":
            self._add_positive_sample(sx, sy)
        elif tool == "negative_sample":
            self._add_negative_sample(sx, sy)
        elif tool == "roi":
            if event.button() == Qt.MouseButton.LeftButton:
                print(f"[DEBUG][ROI] mouse press scene=({sx},{sy})")
                self._add_roi_point(sx, sy)
            event.accept()
            return
        elif tool == "ignore":
            if event.button() == Qt.MouseButton.LeftButton:
                print(f"[DEBUG][IGNORE] mouse press scene=({sx},{sy})")
                self._add_ignore_point(sx, sy)
            event.accept()
            return
        elif tool in {"mask_refine", "mask_brush", "mask_eraser"}:
            print("[DEBUG][Canvas] enter mask_refine mousePress branch")
            self.handle_mask_refine_press(event, scene_pt)
            event.accept()
            return
        elif tool == "main_road_seed":
            if event.button() == Qt.MouseButton.RightButton:
                mode = getattr(self, "_seed_draw_mode", "freehand")
                if mode == "polyline" and self._seed_polyline_active:
                    self._finish_polyline_seed()
                elif mode == "two_point":
                    self._cancel_seed_in_progress()
                    self.tool_interaction.emit("seed_status", {
                        "message": "已退出两点主路线",
                        "exit_tool": True,
                    })
                event.accept()
                return
            if event.button() == Qt.MouseButton.LeftButton:
                self._seed_press(sx, sy, modifiers=event.modifiers())
            event.accept()
            return
        elif tool == "polyline":
            pass
        elif tool in ("set_start", "set_end", "add_task"):
            # ★ 任务点工具：转换为全局像素坐标后发送信号
            x_global, y_global = self._layer_manager.preview_to_global(sx, sy)
            tp_type = {"set_start": "start", "set_end": "goal", "add_task": "task"}[tool]
            self.task_point_clicked.emit(tp_type, x_global, y_global)
        elif tool == "calibrate_map_click":
            # ★ 控制点图上配准：转换为全局像素坐标后发送信号
            x_global, y_global = self._layer_manager.preview_to_global(sx, sy)
            self.calibration_map_clicked.emit(x_global, y_global)
        elif tool in self.GRAPH_TOOLS:
            # ★ 路网编辑工具 — 通过信号发送到 main_window
            self.tool_interaction.emit("press", {"tool": tool, "x": sx, "y": sy, "scene_pt": scene_pt})
            event.accept()
            return

    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动"""
        # ★ 空格临时平移模式
        if self._space_pan_active:
            super().mouseMoveEvent(event)
            return

        sx, sy = self._scene_pos(event)
        scene_pt_move = QPointF(sx, sy)

        # ★ 画笔预览圆更新
        if self._current_tool in {"mask_refine", "mask_brush", "mask_eraser"} and self._in_image_bounds(sx, sy):
            self._update_brush_preview(sx, sy)
        else:
            self._remove_brush_preview()

        # 发射坐标信号（仅图像范围内）
        if self._in_image_bounds(sx, sy):
            self.mouse_moved.emit(sx, sy)
            # ★ 调试模式：同步打印 image↔scene 变换信息
            if self.debug_coord_enabled:
                ix, iy = self.scene_to_image(sx, sy)
                print(f"[DEBUG][COORD] mouseMove scene=({sx},{sy}) image=({ix},{iy}) "
                      f"zoom={self._zoom_level:.3f}")

        # 做调试级别日志
        if event.buttons() != Qt.MouseButton.NoButton:
            print(f"[DEBUG][Canvas] mouseMove tool={self._current_tool} buttons={int(event.buttons().value)} scene=({sx},{sy})")

        # ---- 按工具分发绘制/拖拽逻辑 ----
        tool = self._current_tool
        if tool == "pan":
            super().mouseMoveEvent(event)
            return

        if not self._in_image_bounds(sx, sy):
            super().mouseMoveEvent(event)
            return

        if tool == "roi":
            self._update_roi_temp_line(sx, sy)
        elif tool == "ignore":
            self._update_ignore_temp_line(sx, sy)
        elif tool == "main_road_seed":
            mode = getattr(self, "_seed_draw_mode", "freehand")
            if mode in {"two_point", "polyline"}:
                self._seed_update_rubber_band(
                    sx, sy, shift=bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                )
                event.accept()
                return
            if event.buttons() & Qt.MouseButton.LeftButton:
                if self._seed_drawing:
                    self._seed_move(sx, sy)
                    event.accept()
                    return
        elif tool in {"mask_refine", "mask_brush", "mask_eraser"} and (event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)):
            if getattr(self, "is_mask_drawing", False):
                print("[DEBUG][Canvas] enter mask_refine mouseMove branch")
                self.handle_mask_refine_move(event, scene_pt_move)
                event.accept()
                return
        elif event.buttons() & Qt.MouseButton.LeftButton:
            if tool == "polyline":
                pass
            elif tool in self.GRAPH_TOOLS:
                # ★ 路网编辑工具拖拽 — 通过信号发送
                self.tool_interaction.emit("move", {"tool": tool, "x": sx, "y": sy})

        # ★ 路网编辑工具悬停（限流：不每次移动都发送）
        if tool in self.GRAPH_TOOLS:
            self.tool_interaction.emit("hover", {"tool": tool, "x": sx, "y": sy})

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放"""
        # ★ 空格临时平移模式
        if self._space_pan_active:
            super().mouseReleaseEvent(event)
            return

        sx, sy = self._scene_pos(event)
        btn_name = {Qt.MouseButton.LeftButton: "Left", Qt.MouseButton.MiddleButton: "Middle",
                     Qt.MouseButton.RightButton: "Right"}.get(event.button(), str(event.button()))
        print(f"[DEBUG][Canvas] mouseRelease tool={self._current_tool} button={btn_name} scene=({sx},{sy})")

        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(
                QGraphicsView.DragMode.ScrollHandDrag
                if (self._current_tool == "pan" or self._space_pan_active)
                else QGraphicsView.DragMode.NoDrag
            )

        # ---- Mask 精修：松开按键结束绘制 ----
        if self._current_tool in {"mask_refine", "mask_brush", "mask_eraser"} and getattr(self, "is_mask_drawing", False):
            print("[DEBUG][Canvas] enter mask_refine mouseRelease branch")
            self.handle_mask_refine_release(event)
            event.accept()
            return

        # ---- 主路种子线：松开左键结束自由绘一笔 ----
        if self._current_tool == "main_road_seed" and event.button() == Qt.MouseButton.LeftButton:
            if getattr(self, "_seed_draw_mode", "freehand") == "freehand":
                self._finalize_seed_stroke()
            event.accept()
            return

        # ---- Ignore / ROI 右键完成绘制 ----
        if event.button() == Qt.MouseButton.RightButton:
            if self._current_tool == "roi":
                self._finalize_roi()
                return
            elif self._current_tool == "ignore":
                self._finalize_ignore()
                return

        # ★ 路网编辑工具释放 — 通过信号发送
        tool = self._current_tool
        if tool in self.GRAPH_TOOLS:
            self.tool_interaction.emit("release", {"tool": tool, "x": sx, "y": sy})

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """双击 → 闭合 ROI / Ignore 多边形"""
        sx, sy = self._scene_pos(event)
        if not self._in_image_bounds(sx, sy):
            super().mouseDoubleClickEvent(event)
            return

        tool = self._current_tool
        if tool == "roi":
            self._finalize_roi()
        elif tool == "ignore":
            self._finalize_ignore()
        elif tool == "main_road_seed" and getattr(self, "_seed_draw_mode", "") == "polyline":
            self._finish_polyline_seed()
            event.accept()
            return
        elif tool == "graph_draw_edge":
            # 折线补路：双击结束
            self.tool_interaction.emit("confirm_manual_edge", None)
        else:
            super().mouseDoubleClickEvent(event)

    # ===================================================================
    # 键盘事件
    # ===================================================================

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        mods = event.modifiers()

        # ★ 空格键：临时进入平移模式
        if key == Qt.Key.Key_Space and not event.isAutoRepeat():
            if not self._space_pan_active:
                self._space_pan_active = True
                self._saved_tool_before_pan = self._current_tool
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        # 缩放快捷键（Ctrl+）
        if key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            self.zoom_in()
            return
        elif key == Qt.Key.Key_Minus:
            self.zoom_out()
            return
        elif key == Qt.Key.Key_0 and mods & Qt.KeyboardModifier.ControlModifier:
            self.fit_to_window()
            return
        elif key == Qt.Key.Key_1 and mods & Qt.KeyboardModifier.ControlModifier:
            self.zoom_to_100()
            return

        # ---- [ / ] 调整画笔半径 ----
        if key == Qt.Key.Key_BracketLeft:
            self._mask_brush_radius = max(1, self._mask_brush_radius - 1)
            self.tool_interaction.emit("brush_radius_changed", self._mask_brush_radius)
            return
        if key == Qt.Key.Key_BracketRight:
            self._mask_brush_radius = min(100, self._mask_brush_radius + 1)
            self.tool_interaction.emit("brush_radius_changed", self._mask_brush_radius)
            return

        # ---- Ctrl+Z / Ctrl+Y / Ctrl+Shift+Z 撤销重做 ----
        if key == Qt.Key.Key_Z and mods & Qt.KeyboardModifier.ControlModifier:
            if mods & Qt.KeyboardModifier.ShiftModifier:
                # Ctrl+Shift+Z → redo
                self.tool_interaction.emit("redo", None)
            else:
                # Ctrl+Z → undo
                self.tool_interaction.emit("undo", None)
            return
        if key == Qt.Key.Key_Y and mods & Qt.KeyboardModifier.ControlModifier:
            self.tool_interaction.emit("redo", None)
            return

        # ---- Ctrl+D / Ctrl+Shift+D：坐标变换调试开关 ----
        if key == Qt.Key.Key_D and mods & Qt.KeyboardModifier.ControlModifier:
            if mods & Qt.KeyboardModifier.ShiftModifier:
                # Ctrl+Shift+D → 切换调试十字标记
                self.debug_click_marker_enabled = not self.debug_click_marker_enabled
            else:
                # Ctrl+D → 切换坐标变换调试信息
                self.debug_toggle_coord_info()
            return

        # ---- Delete 键 ----
        if key == Qt.Key.Key_Delete:
            self.tool_interaction.emit("delete", None)
            return

        # ---- ROI / Ignore 操作快捷键 ----
        tool = self._current_tool

        if key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            # Enter: 闭合当前多边形
            if tool == "roi":
                self._finalize_roi()
                return
            elif tool == "ignore":
                self._finalize_ignore()
                return
            elif tool == "graph_draw_edge":
                self.tool_interaction.emit("confirm_manual_edge", None)
                return
            elif tool in self.GRAPH_TOOLS:
                # Enter on any graph tool: confirm pending operations
                self.tool_interaction.emit("key_enter", {"tool": tool})
                return

        if key == Qt.Key.Key_D:
            if tool == "roi":
                self._delete_last_roi()
                return
            elif tool == "ignore":
                self._delete_last_ignore()
                return

        if key == Qt.Key.Key_U:
            # u: 撤销
            if tool == "roi":
                if self._roi_points:
                    self._undo_roi_point()
                else:
                    self._delete_last_roi()
                return
            elif tool == "ignore":
                if self._ignore_points:
                    self._undo_ignore_point()
                else:
                    self._delete_last_ignore()
                return
            elif tool in {"mask_refine", "mask_brush", "mask_eraser"}:
                return
            elif tool == "graph_draw_edge":
                # Undo last manual edge point
                self.tool_interaction.emit("undo_manual_point", None)
                return

        if key == Qt.Key.Key_C:
            # c: 清空
            if tool == "roi":
                self._clear_all_roi()
                return
            elif tool == "ignore":
                self._clear_all_ignore()
                return
            elif tool == "graph_draw_edge":
                self.tool_interaction.emit("clear_manual_points", None)
                return

        if key == Qt.Key.Key_Backspace:
            # Backspace: 删除上一个顶点
            if tool == "roi" and self._roi_points:
                self._undo_roi_point()
                return
            elif tool == "ignore" and self._ignore_points:
                self._undo_ignore_point()
                return
            elif tool == "main_road_seed" and getattr(self, "_seed_draw_mode", "") == "polyline":
                if self._seed_current_points:
                    self._seed_current_points.pop()
                    if self._seed_temp_dot_items:
                        item = self._seed_temp_dot_items.pop()
                        _safe_remove_scene_item(self._scene, item)
                    if self._seed_current_items:
                        item = self._seed_current_items.pop()
                        _safe_remove_scene_item(self._scene, item)
                    if not self._seed_current_points:
                        self._seed_polyline_active = False
                        self._clear_seed_temp_preview()
                    self._scene.update()
                    self.viewport().update()
                return
            elif tool == "graph_draw_edge":
                self.tool_interaction.emit("undo_manual_point", None)
                return

        if key == Qt.Key.Key_S and tool in {"mask_refine", "mask_brush", "mask_eraser"}:
            return

        if key == Qt.Key.Key_Escape:
            if tool == "roi":
                # Esc 只取消当前未完成多边形，不删除已完成区域。
                self._clear_current_roi_drawing()
                self.tool_interaction.emit("roi_drawing_cancelled", None)
                return
            elif tool == "ignore":
                self._clear_current_ignore_drawing()
                return
            elif tool == "main_road_seed":
                mode = getattr(self, "_seed_draw_mode", "freehand")
                if mode == "two_point" and self._seed_two_point_start is not None:
                    self._cancel_seed_in_progress()
                    self.tool_interaction.emit("seed_status", {"message": "请选择主路线起点"})
                elif mode == "polyline" and self._seed_polyline_active:
                    self._cancel_seed_in_progress()
                    self.tool_interaction.emit("seed_status", {"message": "已取消多点主路线"})
                else:
                    self._cancel_seed_in_progress()
                    self.tool_interaction.emit("seed_status", {
                        "message": "已退出主路种子线工具",
                        "exit_tool": True,
                    })
                return
            elif tool == "graph_draw_edge":
                self.tool_interaction.emit("cancel_manual_edge", None)
                return
            elif tool in self.GRAPH_TOOLS:
                # Esc on graph tools: cancel current selection/operation
                self.tool_interaction.emit("key_escape", {"tool": tool})
                return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        """键盘释放 — 空格键恢复原工具"""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if self._space_pan_active:
                self._space_pan_active = False
                # 恢复原工具的光标和拖拽模式
                if self._saved_tool_before_pan == "pan":
                    self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
                else:
                    self.setDragMode(QGraphicsView.DragMode.NoDrag)
                    self.current_tool = self._saved_tool_before_pan  # 恢复光标
            return
        super().keyReleaseEvent(event)

    # ===================================================================
    # Mask 精修事件处理（V3 实装）
    # ===================================================================

    def _ensure_mask_exists(self):
        """如果 current_mask 为 None，自动创建空白 mask"""
        if self.current_mask is not None:
            return
        w, h = self._layer_manager.original_size
        if w <= 0 or h <= 0:
            w, h = self._layer_manager.image_size
        blank = np.zeros((h, w), dtype=np.uint8)
        self._layer_manager.set_layer_data("mask", blank)
        self._layer_manager.show_layer("mask")
        print(f"[DEBUG][MaskRefine] auto-created blank mask ({w}x{h})")

    def _ensure_editable_mask(self) -> np.ndarray:
        self._ensure_mask_exists()
        image_size = self._layer_manager.original_size
        if image_size[0] <= 0 or image_size[1] <= 0:
            image_size = self._layer_manager.image_size
        mask = ensure_mask_image_size(self.current_mask, image_size)
        if mask is not self.current_mask:
            self._layer_manager.set_layer_data("mask", mask)
        return mask

    def handle_mask_refine_press(self, event: QMouseEvent, scene_pos: QPointF):
        """Mask 精修：鼠标按下 → 开始绘制"""
        print("[DEBUG][MaskRefine] handle_mask_refine_press called")

        ix, iy = self.scene_to_image_xy(scene_pos)
        self._log_region_coordinate(scene_pos.x(), scene_pos.y())

        if not self._in_image_bounds(int(scene_pos.x()), int(scene_pos.y())):
            print("[DEBUG][MaskRefine] click outside image")
            return

        if self._current_tool == "mask_brush" and event.button() == Qt.MouseButton.LeftButton:
            self.mask_draw_mode = "add"
        elif self._current_tool == "mask_eraser" and event.button() == Qt.MouseButton.LeftButton:
            self.mask_draw_mode = "erase"
        elif self._current_tool == "mask_refine" and event.button() == Qt.MouseButton.LeftButton:
            self.mask_draw_mode = "add"
        elif self._current_tool == "mask_refine" and event.button() == Qt.MouseButton.RightButton:
            self.mask_draw_mode = "erase"
        else:
            print("[DEBUG][MaskRefine] unsupported mouse button")
            return

        # 自动创建并标准化为单通道 uint8；坏输入会明确报错。
        try:
            self._ensure_editable_mask()
        except (TypeError, ValueError) as exc:
            print(f"[RegionEdit] invalid current mask: {exc}")
            self.tool_interaction.emit("mask_edit_error", str(exc))
            return

        # ★ 推入全局撤销（mask 编辑开始）
        is_large = self._layer_manager.is_large_image_mode
        if self._history_push_callback and not is_large:
            action = (
                "mask_eraser_stroke" if self.mask_draw_mode == "erase"
                else "mask_brush_stroke"
            )
            self._history_push_callback(action)

        self.is_mask_drawing = True
        self._mask_stroke_changed = False
        self._mask_dirty_rect = None
        self._mask_stroke_before_patch = None
        self.last_mask_point = (ix, iy)

        print(f"[DEBUG][MaskRefine] start mode={self.mask_draw_mode}, image=({ix},{iy})")

        self.apply_mask_brush((ix, iy), (ix, iy))

    def handle_mask_refine_move(self, event: QMouseEvent, scene_pos: QPointF):
        """Mask 精修：鼠标拖拽 → 持续绘制"""
        ix, iy = self.scene_to_image_xy(scene_pos)

        if getattr(self, "last_mask_point", None) is None:
            self.last_mask_point = (ix, iy)

        last = self.last_mask_point
        print(
            f"[DEBUG][MaskRefine] move mode={self.mask_draw_mode}, "
            f"from=({last[0]},{last[1]}), to=({ix},{iy})"
        )

        self.apply_mask_brush(last, (ix, iy))
        self.last_mask_point = (ix, iy)

    def handle_mask_refine_release(self, event: QMouseEvent):
        """Mask 精修：鼠标释放 → 结束绘制"""
        print("[DEBUG][MaskRefine] release")

        mode = self.mask_draw_mode
        changed = self._mask_stroke_changed
        self.is_mask_drawing = False
        self.mask_draw_mode = None
        self.last_mask_point = None

        if (changed and self._mask_dirty_rect is not None
                and self._mask_stroke_before_patch is not None
                and self._history_mask_patch_callback is not None):
            x0, y0, x1, y1 = self._mask_dirty_rect
            action = "mask_eraser_stroke" if mode == "erase" else "mask_brush_stroke"
            self._history_mask_patch_callback(
                action, self._mask_dirty_rect,
                self._mask_stroke_before_patch,
                self.current_mask[y0:y1, x0:x1],
            )
        self.update_mask_overlay()
        if changed and self._mask_dirty_rect is not None:
            self._layer_manager.update_layer_preview_region(
                "layer_road_mask", self._mask_dirty_rect,
            )
        self._mask_dirty_rect = None
        self._mask_stroke_before_patch = None
        if changed:
            self.tool_interaction.emit("mask_stroke_finished", {"mode": mode})

    def apply_mask_brush(self, p1: tuple, p2: tuple):
        """在 current_mask 上绘制线段（p1→p2）"""
        mask = self.current_mask
        if mask is None:
            print("[WARNING][MaskRefine] current_mask is None，无法编辑")
            return

        r = int(self.brush_radius) + 2
        x0 = max(0, min(int(p1[0]), int(p2[0])) - r)
        y0 = max(0, min(int(p1[1]), int(p2[1])) - r)
        x1 = min(mask.shape[1], max(int(p1[0]), int(p2[0])) + r + 1)
        y1 = min(mask.shape[0], max(int(p1[1]), int(p2[1])) + r + 1)
        before_patch = mask[y0:y1, x0:x1].copy()
        new_dirty = (x0, y0, x1, y1)
        if self._layer_manager.is_large_image_mode:
            if self._mask_dirty_rect is None:
                self._mask_stroke_before_patch = before_patch.copy()
            else:
                ox0, oy0, ox1, oy1 = self._mask_dirty_rect
                nx0, ny0 = min(ox0, x0), min(oy0, y0)
                nx1, ny1 = max(ox1, x1), max(oy1, y1)
                if (nx0, ny0, nx1, ny1) != self._mask_dirty_rect:
                    expanded = mask[ny0:ny1, nx0:nx1].copy()
                    expanded[oy0-ny0:oy1-ny0, ox0-nx0:ox1-nx0] = self._mask_stroke_before_patch
                    self._mask_stroke_before_patch = expanded
                new_dirty = (nx0, ny0, nx1, ny1)
        paint_mask_segment(
            mask, p1, p2, radius=self.brush_radius,
            erase=self.mask_draw_mode == "erase",
        )
        self._mask_stroke_changed = (
            self._mask_stroke_changed
            or not np.array_equal(before_patch, mask[y0:y1, x0:x1])
        )
        dirty = new_dirty
        if self._mask_dirty_rect is None:
            self._mask_dirty_rect = dirty
        else:
            old = self._mask_dirty_rect
            self._mask_dirty_rect = (
                min(old[0], dirty[0]), min(old[1], dirty[1]),
                max(old[2], dirty[2]), max(old[3], dirty[3]),
            )

        print(
            f"[DEBUG][MaskRefine] draw {self.mask_draw_mode}, "
            f"p1=({p1[0]},{p1[1]}), p2=({p2[0]},{p2[1]}), radius={self.brush_radius}"
        )

    def update_mask_overlay(self):
        """刷新 LayerManager 中的 mask 缓存，触发画面重绘"""
        self._layer_manager.invalidate_layer_cache("layer_road_mask")
        self._layer_manager.layer_changed.emit("layer_road_mask")
        self.viewport().update()
