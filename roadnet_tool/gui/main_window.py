"""
主窗口：菜单栏、导航栏、三栏布局、状态栏。
RoadNet Studio — 无人车比赛半自动路网生成与编辑工具。
"""

from __future__ import annotations

import os
import re
import sys
import copy
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

from PySide6.QtCore import Qt, QSize, QTimer, QThread, QUrl, QPointF
from PySide6.QtGui import (
    QAction, QKeySequence, QIcon, QPen, QColor, QBrush, QFont,
    QPainter, QDesktopServices, QPolygonF,
)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QToolBar, QPushButton, QFileDialog,
    QMessageBox, QMenuBar, QMenu, QLabel,
    QApplication, QButtonGroup, QGraphicsEllipseItem,
    QGraphicsLineItem, QGraphicsPathItem, QGraphicsPolygonItem,
    QGraphicsItem, QGraphicsItemGroup, QGraphicsSimpleTextItem,
    QDoubleSpinBox, QProgressDialog,
    QInputDialog,
)

try:
    from shiboken6 import isValid as _shiboken_is_valid
except ImportError:
    def _shiboken_is_valid(obj):
        try:
            return obj is not None
        except RuntimeError:
            return False


def safe_remove_scene_item(scene, item):
    """安全移除 QGraphicsScene 中的 item，避免 C++ 对象已被删除的崩溃。"""
    if item is None:
        return
    try:
        if _shiboken_is_valid(item) and item.scene() is scene:
            scene.removeItem(item)
    except (RuntimeError, AttributeError):
        pass

from .canvas_view import CanvasView
from .tool_panel import ToolPanel
from .parameter_panel import ParameterPanel
from .layer_manager import LayerManager
from .status_bar import RoadNetStatusBar
from .project_manager import ProjectManager
from .history_manager import GlobalHistoryManager
from roadnet.geo_calibration import GeoCalibration
from roadnet.samroad_adapter import (
    load_samroad_output, validate_samroad_output,
    detect_samroad_outputs, load_graph_for_draft,
    load_skeleton_from_file, load_mask_clean_only,
)
from roadnet.samroad_runner import SAMRoadRunResult


# 导航步骤定义
NAV_STEPS = [
    ("import",    "① 导入影像"),
    ("segment",   "② 道路分割"),
    ("edit",      "③ 区域修正"),
    ("skeleton",  "④ 骨架优化"),
    ("graph",     "⑤ 路网编辑"),
    ("calibrate", "⑥ 坐标校准"),
    ("export",    "⑦ 路径规划/导出"),
]


class MainWindow(QMainWindow):
    """RoadNet Studio 主窗口"""

    def __init__(self):
        super().__init__()

        # 核心组件
        self._layer_manager = LayerManager(self)
        self._project_manager = ProjectManager()

        # UI 组件
        self._canvas: Optional[CanvasView] = None
        self._tool_panel: Optional[ToolPanel] = None
        self._param_panel: Optional[ParameterPanel] = None
        self._status_bar: Optional[RoadNetStatusBar] = None
        self._nav_buttons: dict[str, QPushButton] = {}

        # 当前阶段
        self._current_stage: str = "import"
        self._stage_completed: set = set()

        # ★ 路网编辑器（Qt 原生）— 管理 final_graph
        self._graph_editor = None  # GraphEditorQt 实例

        # ★ SAM-Road 参考图数据（独立于 final_graph）
        self._reference_graph_nodes: list = []
        self._reference_graph_edges: list = []

        # ★ 坐标校准
        self._geo_calibration = GeoCalibration(mode="auto")

        # ★ 控制点图上点击配准状态
        self._map_click_calibration_mode: bool = False
        self._map_click_calibration_queue: list = []
        self._map_click_calibration_index: int = 0

        # ★ 全局撤销/重做管理器
        self._history = GlobalHistoryManager(max_steps=50)

        # ★ Graph 场景渲染项
        self._graph_node_items: dict[int, QGraphicsEllipseItem] = {}
        self._graph_edge_items: dict[int, QGraphicsPathItem] = {}
        self._graph_endpoint_items: list = []  # ★ degree=1 端点可视化
        self._graph_manual_preview_items: list = []

        # Automatic diagnosis and semi-automatic repair candidates. These are
        # previews only; graph/mask data changes happen after explicit confirm.
        self._mask_ignore_candidates: list[dict] = []
        self._mask_candidate_items: list = []
        self._graph_repair_candidates: list[dict] = []
        self._graph_repair_items: list = []
        self._last_graph_diagnostics: dict = {}
        self._last_auto_repair_output_dir: Optional[str] = None
        self._mask_candidate_signature = None
        self._graph_repair_signature = None
        self._last_mask_filter_output_dir: Optional[str] = None
        self._last_mask_filter_report: dict = {}
        self._mask_before_auto_ignore: Optional[np.ndarray] = None
        self._region_edit_stable_mode: bool = True

        # ★ 参考图渲染项（SAM-Road raw graph）
        self._ref_node_items: list = []
        self._ref_edge_items: list = []

        # ★ SAM-Road 单图推理包输出（额外图层）
        self._samroad_single_itsc_mask: Optional[np.ndarray] = None  # itsc_mask.png
        self._samroad_single_viz: Optional[np.ndarray] = None        # viz.png
        self._samroad_single_overlay_items: list = []  # itsc/viz 的 scene items
        self._valid_image_mask: Optional[np.ndarray] = None
        self._valid_mask_report: dict = {}

        # Canonical skeleton state. The displayed layer is current_skeleton;
        # optimization always starts from raw_skeleton.
        self.raw_skeleton: Optional[np.ndarray] = None
        self.optimized_skeleton: Optional[np.ndarray] = None
        self.current_skeleton: Optional[np.ndarray] = None
        self.skeleton_state: str = "none"
        self._skeleton_backup_raw = None
        self._skeleton_backup_opt = None
        self._skeleton_backup_state = "none"

        # ★ 任务点相关
        self._task_points: list = []          # List[TaskPoint] — 原始任务点
        self._task_point_diagnostics: list = []
        self._snapped_points: list = []       # List[SnappedTaskPoint] — 吸附结果
        self.snapped_task_points: list = []   # 公开稳定字段，供路径/比赛图导出
        self._global_plan_result = None       # GlobalPlanResult — 全局规划结果
        # ★ 可导出的规划数据必须独立于 QGraphicsScene 持久存在。
        self.planned_path_pixel: list = []    # [[x, y], ...]
        self.planned_path_geo: list = []      # [[longitude, latitude, altitude], ...]
        self.planned_path_edges: list = []    # 按行驶顺序经过的 graph edge id
        self.planning_result = None            # GlobalPlanResult（公开稳定字段）
        self.sparse_waypoints_pixel: list = []
        self.sparse_waypoints_geo: list = []
        self.sparse_waypoints: list = []
        self._vehicle_waypoint_report: dict = {}
        self._waypoint_validation_report: dict = {}
        self._path_layer_diag_result = None
        self._path_layer_diag_dir: Optional[str] = None
        self._waypoint_bad_segments: list = []
        self._dense_path_for_waypoints: list = []
        self._vwp_result = None
        self._vwp_output_dir: Optional[str] = None
        # Sync UI waypoint layer with exported CSV (never stale cache)
        self._vwp_waypoint_layer_name: str = "vehicle_waypoints"
        self._vwp_waypoint_csv_path: Optional[str] = None
        self._vwp_dense_path_csv_path: Optional[str] = None

        # ★ 任务点和规划路径场景渲染项
        self._task_point_original_items: list = []   # 原始任务点（红色十字）
        self._task_point_snapped_items: list = []    # 吸附点（蓝色圆点）
        self._task_point_snap_lines: list = []       # 原始→吸附连线（蓝色虚线）
        self._planned_path_items: list = []          # 规划路径线条
        self._sparse_waypoint_items: list = []
        self._waypoint_validation_items: list = []

        # 模式
        self._clean_mode: bool = True  # 默认简洁模式

        # 后台任务。Worker 只计算，所有 UI 提交都在主线程槽函数中完成。
        self._segmentation_thread: Optional[QThread] = None
        self._segmentation_worker = None
        self._segmentation_image_ref = None
        self._preview_seg_thread: Optional[QThread] = None
        self._preview_seg_worker = None
        self._competition_fast_thread: Optional[QThread] = None
        self._competition_fast_worker = None
        self._lowres_formal_thread: Optional[QThread] = None
        self._lowres_formal_worker = None
        self._formal_extraction_thread: Optional[QThread] = None
        self._formal_extraction_worker = None
        # 当前正式 Road Mask 的注册元数据（OpenCV 正式提取时写入）。
        self._formal_mask_meta: dict = {}
        self._current_adapter_type: str = "auto"
        self._roi_draw_return_stage: Optional[str] = None
        self._roi_draw_baseline_count: int = 0
        self._roi_tile_overlay_visible: bool = False
        self._competition_fast_mode: bool = False
        self._pipeline_thread: Optional[QThread] = None
        self._pipeline_worker = None
        self._pipeline_source_mask = None
        self._pipeline_progress_dialog = None
        self._continue_pipeline_after_samroad = False
        self._samroad_pipeline_mask_imported = False
        self._large_image_project = None
        # ★ 大图模式 working mask 状态（唯一 current working mask）
        self._working_mask_source: str = "formal"   # formal / cleaned_working_mask / manual_after_cleaned / final_edited_mask / ...
        self._working_mask_dirty: bool = False
        self._working_mask_formal_ready: bool = False
        self._working_mask_preview_only: bool = False
        self._mask_edit_base: str = ""  # cleaned_working_mask / global_road_mask / ...
        self._working_road_mask_path: Optional[str] = None
        self._working_road_mask_preview_path: Optional[str] = None
        self._cleaned_working_mask_path: Optional[str] = None
        self._cleaned_working_mask_preview_path: Optional[str] = None
        self._final_edited_mask_path: Optional[str] = None
        self._final_edited_mask_preview_path: Optional[str] = None
        self._cleaned_mask_backup: Optional[np.ndarray] = None  # working before clean
        self._cleaned_mask_pending: Optional[np.ndarray] = None
        self._cleaned_mask_report: dict = {}
        self._ribbon_fill_backup: Optional[np.ndarray] = None
        self._ribbon_fill_pending: Optional[np.ndarray] = None
        self._ribbon_fill_report: dict = {}
        self._ribbon_fill_artifact_dir: Optional[str] = None
        self._ribbon_fill_paths: dict = {}
        self._large_project_thread: Optional[QThread] = None
        self._large_project_worker = None
        self._large_project_progress: Optional[QProgressDialog] = None
        self._large_post_thread: Optional[QThread] = None
        self._large_post_worker = None
        self._large_post_progress: Optional[QProgressDialog] = None
        self._main_road_thread: Optional[QThread] = None
        self._main_road_worker = None
        self._main_road_progress: Optional[QProgressDialog] = None
        self._main_road_backup_mask = None       # 修复前 mask 快照（供回滚）
        self._main_road_backup_meta = None
        self._main_road_seed_return_stage = None
        self._main_road_view_rect = None         # 当前视野作为修复范围
        self._main_road_use_tasks = False
        self._main_road_preview_only = False

        # 窗口设置
        self.setWindowTitle("RoadNet Studio - 无人车路网生成与编辑系统")
        self.setMinimumSize(1200, 750)
        self.resize(1400, 850)

        # 加载样式
        self._load_stylesheet()

        # 构建 UI
        self._setup_menu_bar()
        self._setup_nav_bar()
        self._setup_central_widget()
        self._setup_status_bar()

        # ★ 初始化路网编辑器
        from roadnet.graph_editor_qt import GraphEditorQt
        self._graph_editor = GraphEditorQt()

        # 信号连接
        self._connect_signals()

        # ★ 注入全局撤销管理器依赖
        self._history.inject(self._canvas, self._layer_manager, self._graph_editor, self)

        # ★ 设置 CanvasView 的历史推送回调（用于采样点等操作）
        self._canvas._history_push_callback = self._history.push_state
        self._canvas._history_mask_patch_callback = self._history.push_mask_patch

        # ★ 初始同步 brush radius（canvas 默认 3，与面板一致）
        default_radius = self._param_panel._get_config("edit.brush_radius", 3)
        self._canvas.brush_radius = int(default_radius)

    # ===================================================================
    # 样式
    # ===================================================================

    def _load_stylesheet(self):
        qss_path = os.path.join(os.path.dirname(__file__), "styles.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

    # ===================================================================
    # 菜单栏
    # ===================================================================

    def _setup_menu_bar(self):
        menubar = self.menuBar()

        # ---- 文件 ----
        file_menu = menubar.addMenu("📁 文件(&F)")

        act_new = QAction("新建项目", self)
        act_new.setShortcut(QKeySequence("Ctrl+N"))
        act_new.triggered.connect(self._on_new_project)
        file_menu.addAction(act_new)

        file_menu.addSeparator()

        act_open = QAction("打开影像...", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._on_open_image)
        file_menu.addAction(act_open)

        file_menu.addSeparator()

        act_open_project = QAction("打开项目...", self)
        act_open_project.setShortcut(QKeySequence("Ctrl+Shift+O"))
        act_open_project.triggered.connect(self._on_open_project)
        file_menu.addAction(act_open_project)

        act_save_project = QAction("保存项目", self)
        act_save_project.setShortcut(QKeySequence("Ctrl+Shift+S"))
        act_save_project.triggered.connect(self._on_save_project)
        file_menu.addAction(act_save_project)

        act_save_project_as = QAction("另存项目...", self)
        act_save_project_as.triggered.connect(self._on_save_project_as)
        file_menu.addAction(act_save_project_as)

        file_menu.addSeparator()

        act_exit = QAction("退出", self)
        act_exit.setShortcut(QKeySequence("Alt+F4"))
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # ---- 编辑 ----
        edit_menu = menubar.addMenu("✏ 编辑(&E)")

        self._act_undo = QAction("撤销(&U)", self)
        # 快捷键由下方 ApplicationShortcut 统一接管，避免重复 shortcut 歧义。
        self._act_undo.triggered.connect(self._on_global_undo)
        edit_menu.addAction(self._act_undo)

        self._act_redo = QAction("重做(&R)", self)
        self._act_redo.triggered.connect(self._on_global_redo)
        edit_menu.addAction(self._act_redo)

        # ★ ApplicationShortcut 确保在任何 Widget 焦点下都能响应
        from PySide6.QtGui import QShortcut
        self._shortcut_undo = QShortcut(QKeySequence.Undo, self)
        self._shortcut_undo.setContext(Qt.ApplicationShortcut)
        self._shortcut_undo.activated.connect(self._on_global_undo)

        self._shortcut_redo = QShortcut(QKeySequence.Redo, self)
        self._shortcut_redo.setContext(Qt.ApplicationShortcut)
        self._shortcut_redo.activated.connect(self._on_global_redo)

        self._shortcut_region_escape = QShortcut(QKeySequence("Esc"), self)
        self._shortcut_region_escape.setContext(Qt.ApplicationShortcut)
        self._shortcut_region_escape.activated.connect(self._on_region_escape_shortcut)

        self._shortcut_region_backspace = QShortcut(QKeySequence("Backspace"), self)
        self._shortcut_region_backspace.setContext(Qt.ApplicationShortcut)
        self._shortcut_region_backspace.activated.connect(self._on_region_backspace_shortcut)

        edit_menu.addSeparator()

        # 一键运行完整流程
        self._act_run_pipeline = QAction("一键生成初始路网(&P)", self)
        self._act_run_pipeline.setShortcut(QKeySequence("Ctrl+R"))
        self._act_run_pipeline.setToolTip(
            "自动执行完整流程：后处理 → 骨架生成 → 骨架优化 → 生成路网图\n"
            "要求：已加载 road_mask"
        )
        self._act_run_pipeline.triggered.connect(self._on_run_pipeline)
        edit_menu.addAction(self._act_run_pipeline)

        # ---- 视图 ----
        view_menu = menubar.addMenu("👁 视图(&V)")

        self._act_mask_visible = QAction("显示 Mask 图层", self)
        self._act_mask_visible.setCheckable(True)
        self._act_mask_visible.setChecked(False)
        self._act_mask_visible.triggered.connect(
            lambda v: self._toggle_layer("mask", v)
        )
        view_menu.addAction(self._act_mask_visible)

        self._act_skeleton_visible = QAction("显示 Skeleton 图层", self)
        self._act_skeleton_visible.setCheckable(True)
        self._act_skeleton_visible.setChecked(True)
        self._act_skeleton_visible.triggered.connect(
            lambda v: self._toggle_layer("skeleton", v)
        )
        view_menu.addAction(self._act_skeleton_visible)

        self._act_draft_visible = QAction("显示 Draft Graph 图层", self)
        self._act_draft_visible.setCheckable(True)
        self._act_draft_visible.setChecked(True)
        self._act_draft_visible.triggered.connect(
            lambda v: self._toggle_layer("draft_graph", v)
        )
        view_menu.addAction(self._act_draft_visible)

        self._act_final_visible = QAction("显示 Final Graph 图层", self)
        self._act_final_visible.setCheckable(True)
        self._act_final_visible.setChecked(True)
        self._act_final_visible.triggered.connect(
            lambda v: self._toggle_layer("final_graph", v)
        )
        view_menu.addAction(self._act_final_visible)

        self._act_reference_visible = QAction("显示 SAM-Road 参考图", self)
        self._act_reference_visible.setCheckable(True)
        self._act_reference_visible.setChecked(False)
        self._act_reference_visible.triggered.connect(
            lambda v: self._toggle_layer("reference_graph", v)
        )
        view_menu.addAction(self._act_reference_visible)

        view_menu.addSeparator()

        act_zoom_fit = QAction("适应窗口", self)
        act_zoom_fit.setShortcut(QKeySequence("Ctrl+0"))
        act_zoom_fit.triggered.connect(self._on_zoom_fit)
        view_menu.addAction(act_zoom_fit)

        act_zoom_100 = QAction("100%", self)
        act_zoom_100.setShortcut(QKeySequence("Ctrl+1"))
        act_zoom_100.triggered.connect(self._on_zoom_100)
        view_menu.addAction(act_zoom_100)

        act_zoom_in = QAction("放大", self)
        act_zoom_in.setShortcut(QKeySequence("Ctrl+="))
        act_zoom_in.triggered.connect(self._on_zoom_in)
        view_menu.addAction(act_zoom_in)

        act_zoom_out = QAction("缩小", self)
        act_zoom_out.setShortcut(QKeySequence("Ctrl+-"))
        act_zoom_out.triggered.connect(self._on_zoom_out)
        view_menu.addAction(act_zoom_out)

        # ---- 工具 ----
        tools_menu = menubar.addMenu("🔧 工具(&T)")
        act_region_stable = QAction("进入区域修正稳定模式", self)
        act_region_stable.triggered.connect(self._on_enter_region_edit_stable_mode)
        tools_menu.addAction(act_region_stable)

        act_region_self_check = QAction("区域修正自检", self)
        act_region_self_check.triggered.connect(self._on_region_edit_self_check)
        tools_menu.addAction(act_region_self_check)
        tools_menu.addSeparator()

        large_menu = tools_menu.addMenu("大图模式")
        act_open_large = QAction("打开大图项目…", self)
        act_open_large.triggered.connect(self._on_open_large_image_project)
        large_menu.addAction(act_open_large)
        act_large_preview = QAction("生成/刷新大图预览", self)
        act_large_preview.triggered.connect(
            lambda: self._start_large_project_task("preview", reload_preview=True)
        )
        large_menu.addAction(act_large_preview)
        act_large_index = QAction("重新生成 Tile Index", self)
        act_large_index.triggered.connect(self._on_generate_large_tile_index)
        large_menu.addAction(act_large_index)
        act_large_sam = QAction("运行 Tile SAM-RoadPlus…", self)
        act_large_sam.triggered.connect(
            lambda: self._on_run_samroad_single_extract(force_tile=True)
        )
        large_menu.addAction(act_large_sam)
        act_large_post = QAction("大图 Mask 后处理", self)
        act_large_post.triggered.connect(self._on_large_mask_postprocess)
        large_menu.addAction(act_large_post)
        act_large_edit = QAction("进入局部区域修正", self)
        act_large_edit.triggered.connect(lambda: self.set_stage("edit"))
        large_menu.addAction(act_large_edit)
        act_large_graph = QAction("生成全局路网", self)
        act_large_graph.triggered.connect(self._on_run_pipeline)
        large_menu.addAction(act_large_graph)
        act_large_tasks = QAction("导入任务点", self)
        act_large_tasks.triggered.connect(self._on_import_task_points)
        large_menu.addAction(act_large_tasks)
        act_large_plan = QAction("规划路径", self)
        act_large_plan.triggered.connect(self._on_run_global_plan)
        large_menu.addAction(act_large_plan)
        act_large_export = QAction("导出比赛数据", self)
        act_large_export.triggered.connect(self._on_export_planned_path)
        large_menu.addAction(act_large_export)
        large_menu.addSeparator()
        act_large_self_check = QAction("大图项目自检", self)
        act_large_self_check.triggered.connect(self._on_large_image_self_check)
        large_menu.addAction(act_large_self_check)

        tools_menu.addSeparator()

        act_postprocess = QAction("运行后处理", self)
        act_postprocess.triggered.connect(self._on_run_postprocess)
        tools_menu.addAction(act_postprocess)
        act_skeleton = QAction("运行 Skeleton", self)
        act_skeleton.triggered.connect(self._on_run_skeleton)
        tools_menu.addAction(act_skeleton)

        tools_menu.addSeparator()

        act_mask_post = QAction("SAM-Road mask 后处理...", self)
        act_mask_post.setToolTip("调整阈值、形态学参数、面积过滤等，迭代优化 road mask")
        act_mask_post.triggered.connect(self._on_mask_postprocess)
        tools_menu.addAction(act_mask_post)

        act_skel_gen = QAction("从当前 mask 生成 skeleton", self)
        act_skel_gen.setToolTip("对当前 mask 骨架化 + 短枝剪除 + 可选端点连接")
        act_skel_gen.triggered.connect(self._on_skeleton_from_mask)
        tools_menu.addAction(act_skel_gen)

        act_skel_to_graph = QAction("从 skeleton 生成 graph", self)
        act_skel_to_graph.setToolTip("从 skeleton 提取节点和边，生成最终可编辑路网图")
        act_skel_to_graph.triggered.connect(self._on_graph_from_skeleton)
        tools_menu.addAction(act_skel_to_graph)

        act_line_opt = QAction("优化 graph 线形...", self)
        act_line_opt.setToolTip(
            "对 final_graph 每条 edge 的 polyline 进行线形优化：\n"
            "RDP 简化 + 近似直线拉直 + 弯路平滑 + mask 校验\n"
            "不修改交叉口节点和端点，只优化 polyline 中间点"
        )
        act_line_opt.triggered.connect(self._on_graph_line_optimize)
        tools_menu.addAction(act_line_opt)

        act_skel_optimize = QAction("生成/优化道路骨架...", self)
        act_skel_optimize.setToolTip(
            "从当前 road mask 执行完整骨架化 → 优化流水线：\n"
            "mask 标准化 → 骨架生成 → 边界过滤 → 距离变换过滤 → "
            "毛刺删除 → junction 聚类 → 端点连接 → 保存验证输出"
        )
        act_skel_optimize.triggered.connect(self._on_skeleton_optimize_full)
        tools_menu.addAction(act_skel_optimize)

        tools_menu.addSeparator()

        # ── 主入口：SAM-Road 单图初提取（当前推荐工作流）──
        act_samroad_single = QAction("运行 SAM-Road 单图初提取...", self)
        act_samroad_single.setToolTip(
            "调用 D:/sam_road_single_image_share/infer_single.py 进行单图推理，\n"
            "生成 road_mask / itsc_mask / viz / graph.p，运行完成后自动导入结果"
        )
        act_samroad_single.triggered.connect(self._on_run_samroad_single_extract)
        tools_menu.addAction(act_samroad_single)

        act_import_samroad_single = QAction("导入 SAM-Road 单图结果...", self)
        act_import_samroad_single.setToolTip(
            "从 SAM-Road 单图推理输出目录导入 road_mask / itsc_mask / viz / graph.p\n"
            "（不运行推理，仅导入已有结果）"
        )
        act_import_samroad_single.triggered.connect(self._on_import_samroad_single)
        tools_menu.addAction(act_import_samroad_single)

        self._act_samroad_single_viz_visible = QAction("显示/隐藏 SAM-Road 单图 viz/itsc 参考层", self)
        self._act_samroad_single_viz_visible.setCheckable(True)
        self._act_samroad_single_viz_visible.setChecked(True)
        self._act_samroad_single_viz_visible.triggered.connect(
            lambda v: self._toggle_samroad_single_overlays(v)
        )
        tools_menu.addAction(self._act_samroad_single_viz_visible)

        # ── 高级/旧版功能子菜单 ──
        legacy_menu = tools_menu.addMenu("高级/旧版功能")

        act_samroad = QAction("旧版 SAM-Road 初提取...", self)
        act_samroad.setToolTip("[旧版·已废弃] 调用外部 SAM-Road 模型生成 road mask 和 draft graph。\n推荐使用「运行 SAM-Road 单图初提取」。")
        act_samroad.triggered.connect(self._on_run_samroad_extract)
        legacy_menu.addAction(act_samroad)

        act_import_samroad = QAction("旧版 导入 SAM-Road 结果...", self)
        act_import_samroad.setToolTip("[旧版·已废弃] 从旧版 SAM-Road 输出目录导入 mask、skeleton、draft_graph.json。\n推荐使用「导入 SAM-Road 单图结果」。")
        act_import_samroad.triggered.connect(self._on_import_samroad)
        legacy_menu.addAction(act_import_samroad)

        # ---- 任务点 ----
        tp_menu = menubar.addMenu("📍 任务点(&W)")
        act_import_tp = QAction("导入任务点文件...", self)
        act_import_tp.setToolTip("支持 txt/csv 格式，多种分隔符和编码")
        act_import_tp.triggered.connect(self._on_import_task_points)
        tp_menu.addAction(act_import_tp)

        act_manage_tp = QAction("管理任务点...", self)
        act_manage_tp.setToolTip("修改 seq / point_type，上移、下移、设置角色、删除和重新编号")
        act_manage_tp.triggered.connect(self._on_manage_task_points)
        tp_menu.addAction(act_manage_tp)

        act_validate_tp = QAction("验证任务点坐标...", self)
        act_validate_tp.setToolTip("验证 lon/lat → pixel → lon/lat，导出 task_points_debug.csv")
        act_validate_tp.triggered.connect(self._on_validate_task_point_coordinates)
        tp_menu.addAction(act_validate_tp)

        self._act_tp_original_visible = QAction("显示/隐藏原始任务点", self)
        self._act_tp_original_visible.setCheckable(True)
        self._act_tp_original_visible.setChecked(True)
        self._act_tp_original_visible.triggered.connect(
            lambda _v: self._render_task_points_to_scene()
        )
        tp_menu.addAction(self._act_tp_original_visible)

        tp_menu.addSeparator()

        act_snap_tp = QAction("自动吸附任务点", self)
        act_snap_tp.setToolTip("将任务点自动吸附到最近 graph edge（优先 edge 投影，多候选保底）")
        act_snap_tp.triggered.connect(self._on_snap_task_points)
        tp_menu.addAction(act_snap_tp)

        self._act_snapped_visible = QAction("显示/隐藏吸附结果", self)
        self._act_snapped_visible.setCheckable(True)
        self._act_snapped_visible.setChecked(True)
        self._act_snapped_visible.triggered.connect(
            lambda _v: self._render_task_points_to_scene()
        )
        tp_menu.addAction(self._act_snapped_visible)

        tp_menu.addSeparator()

        act_save_snap = QAction("保存吸附结果...", self)
        act_save_snap.setToolTip("保存 task_points_snapped.json 到 outputs 目录")
        act_save_snap.triggered.connect(self._on_save_snapped_points)
        tp_menu.addAction(act_save_snap)

        act_clear_tp = QAction("清空任务点", self)
        act_clear_tp.triggered.connect(self._on_clear_task_points)
        tp_menu.addAction(act_clear_tp)

        # ---- 规划 ----
        plan_menu = menubar.addMenu("🗺 规划(&P)")
        act_plan_astar = QAction("运行全局规划 A*", self)
        act_plan_astar.setToolTip("使用 A* 算法按任务点顺序分段规划（失败自动回退 Dijkstra）")
        act_plan_astar.triggered.connect(lambda: self._on_run_global_plan("astar"))
        plan_menu.addAction(act_plan_astar)

        act_plan_dijkstra = QAction("运行全局规划 Dijkstra", self)
        act_plan_dijkstra.setToolTip("使用 Dijkstra 算法按任务点顺序分段规划")
        act_plan_dijkstra.triggered.connect(lambda: self._on_run_global_plan("dijkstra"))
        plan_menu.addAction(act_plan_dijkstra)

        plan_menu.addSeparator()

        self._act_path_visible = QAction("显示/隐藏全局路径", self)
        self._act_path_visible.setCheckable(True)
        self._act_path_visible.setChecked(True)
        self._act_path_visible.triggered.connect(
            lambda v: self._toggle_layer("planned_path", v)
        )
        plan_menu.addAction(self._act_path_visible)

        plan_menu.addSeparator()

        act_export_planned_path = QAction("导出路径...", self)
        act_export_planned_path.setToolTip("导出无人车可执行的 JSON / CSV / YAML 路径数据")
        act_export_planned_path.triggered.connect(self._on_export_planned_path)
        plan_menu.addAction(act_export_planned_path)

        # ---- 导出 ----
        export_menu = menubar.addMenu("💾 导出(&E)")
        act_export_mask = QAction("导出 Mask...", self)
        act_export_mask.triggered.connect(self._on_export_mask)
        export_menu.addAction(act_export_mask)

        act_export_overlay = QAction("导出叠加图...", self)
        act_export_overlay.triggered.connect(self._on_export_overlay)
        export_menu.addAction(act_export_overlay)

        act_export_path = QAction("导出路径...", self)
        act_export_path.setToolTip("导出规划路径数据（不是 Mask PNG）")
        act_export_path.triggered.connect(self._on_export_planned_path)
        export_menu.addAction(act_export_path)

        act_export_competition = QAction("🏆 导出比赛路网图...", self)
        act_export_competition.setToolTip(
            "导出清爽版/调试版影像路网叠加图（含航迹点）及提交数据；地理坐标为 WGS84"
        )
        act_export_competition.triggered.connect(self._on_export_competition_roadnet)
        export_menu.addAction(act_export_competition)

        export_menu.addSeparator()

        # 路网导出子菜单
        act_save_graph = QAction("导出展示图（保存路网）", self)
        act_save_graph.setToolTip("保存路网：所见即所得，不显示节点编号")
        act_save_graph.triggered.connect(self._on_save_graph)
        export_menu.addAction(act_save_graph)

        act_save_graph_debug = QAction("导出调试图", self)
        act_save_graph_debug.setToolTip("导出调试图：显示节点编号、红绿节点，用于开发检查")
        act_save_graph_debug.triggered.connect(self._on_save_graph_debug)
        export_menu.addAction(act_save_graph_debug)

        # ---- 帮助 ----
        help_menu = menubar.addMenu("❓ 帮助(&H)")
        act_about = QAction("关于 RoadNet Studio", self)
        act_about.triggered.connect(self._on_about)
        help_menu.addAction(act_about)

    # ===================================================================
    # 导航栏（含缩放快捷按钮）
    # ===================================================================

    def _setup_nav_bar(self):
        nav_bar = QToolBar("导航")
        nav_bar.setObjectName("nav_bar")
        nav_bar.setMovable(False)
        nav_bar.setIconSize(QSize(16, 16))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, nav_bar)

        title = QLabel("  RoadNet Studio  ")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #89b4fa; padding: 0 8px;")
        nav_bar.addWidget(title)
        nav_bar.addSeparator()

        for step_id, step_label in NAV_STEPS:
            btn = QPushButton(step_label)
            btn.setObjectName("nav-step")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, sid=step_id: self._on_nav_step(sid))
            nav_bar.addWidget(btn)
            self._nav_buttons[step_id] = btn

        if "import" in self._nav_buttons:
            self._nav_buttons["import"].setChecked(True)

        nav_bar.addSeparator()

        # ★ 缩放快捷按钮
        for zid, zlabel, zslot in [
            ("fit", "适应窗口", self._on_zoom_fit),
            ("100", "100%", self._on_zoom_100),
            ("in", "放大", self._on_zoom_in),
            ("out", "缩小", self._on_zoom_out),
            ("full", "全图", self._on_zoom_fit),
        ]:
            zbtn = QPushButton(zlabel)
            zbtn.setObjectName("nav-step")
            zbtn.clicked.connect(zslot)
            zbtn.setToolTip(f"{zlabel} (可用空白键+拖拽平移)")
            nav_bar.addWidget(zbtn)

        nav_bar.addSeparator()

        self._btn_validate_task_points = QPushButton("验证任务点坐标")
        self._btn_validate_task_points.setObjectName("nav-step")
        self._btn_validate_task_points.setToolTip(
            "检查 lon/lat → image pixel → lon/lat，并导出 task_points_debug.csv"
        )
        self._btn_validate_task_points.clicked.connect(
            self._on_validate_task_point_coordinates
        )
        nav_bar.addWidget(self._btn_validate_task_points)

        nav_bar.addSeparator()

        # ★ 一键运行按钮
        self._btn_pipeline = QPushButton("▶ 一键生成路网")
        self._btn_pipeline.setObjectName("nav-step")
        self._btn_pipeline.setToolTip(
            "一键执行完整流程：后处理 → 骨架生成 → 骨架优化 → 生成路网图\n"
            "快捷键: Ctrl+R"
        )
        self._btn_pipeline.clicked.connect(self._on_run_pipeline)
        nav_bar.addWidget(self._btn_pipeline)

    # ===================================================================
    # 中央区域
    # ===================================================================

    def _setup_central_widget(self):
        central = QWidget()
        self.setCentralWidget(central)

        self._tool_panel = ToolPanel(v1_only=False)
        self._canvas = CanvasView(self._layer_manager)
        self._param_panel = ParameterPanel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._tool_panel)
        splitter.addWidget(self._canvas)
        splitter.addWidget(self._param_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([150, 900, 280])

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    # ===================================================================
    # 状态栏
    # ===================================================================

    def _setup_status_bar(self):
        self._status_bar = RoadNetStatusBar(self)
        self.setStatusBar(self._status_bar)

    # ===================================================================
    # 信号连接
    # ===================================================================

    def _connect_signals(self):
        # 工具栏 → 主窗口 set_tool
        self._tool_panel.tool_selected.connect(self.set_tool)

        # 画布 → 状态栏（含坐标校准坐标）	
        self._canvas.mouse_moved.connect(self._on_mouse_moved)
        self._canvas.zoom_changed.connect(self._status_bar.update_zoom)

        # 画布 → 参数面板
        self._canvas.sample_points_changed.connect(self._param_panel.update_counts)

        # ★ 画布交互信号 → 路网编辑
        self._canvas.tool_interaction.connect(self._on_tool_interaction)

        # ★ 任务点点击信号
        self._canvas.task_point_clicked.connect(self._on_task_point_clicked)

        # ★ 控制点图上配准信号
        self._canvas.calibration_map_clicked.connect(self._handle_map_click_calibration)

        # 参数面板
        self._param_panel.apply_requested.connect(self._on_apply_params)
        self._param_panel.param_changed.connect(self._on_param_changed)

        # ★ 所有图层 checkbox 回调（统一命名）
        _all_layers = [
            "layer_sample_points", "layer_roi", "layer_ignore",
            "layer_road_mask", "layer_preview_segmentation",
            "layer_skeleton", "layer_draft_graph",
            "layer_final_graph", "layer_planned_path", "layer_sparse_waypoints",
            "layer_waypoint_validation",
            "layer_skeleton_nodes",
        ]
        for lname in _all_layers:
            self._tool_panel.set_layer_toggle_callback(lname,
                lambda n, v, ln=lname: self._toggle_layer(ln, v))

        # ★ 模式切换
        self._tool_panel.mode_toggled.connect(self._on_mode_toggled)

        # ★ 清爽显示 / 调试显示
        self._tool_panel.clean_display_requested.connect(self._on_clean_display)
        self._tool_panel.debug_display_requested.connect(self._on_debug_display)

    def _toggle_layer(self, name: str, visible: bool):
        """图层显隐切换"""
        self._layer_manager.toggle_layer(name, visible)
        # 同步 checkbox
        self._tool_panel.set_layer_checkbox_state(name, visible)
        # 如果图在 Qt 场景中已渲染，刷新
        if name in ("layer_draft_graph", "layer_final_graph",
                     "draft_graph", "final_graph"):
            self._render_graph_to_scene()
        elif name in ("layer_reference_graph", "reference_graph", "samroad_raw_graph"):
            if visible:
                self._render_reference_graph_to_scene()
            else:
                self._clear_reference_graph_items()
        elif name in ("layer_task_points", "task_points"):
            if visible:
                self._render_task_points_to_scene()
            else:
                self._clear_task_point_items()
        elif name in ("layer_planned_path", "planned_path"):
            if visible:
                self._render_planned_path_to_scene()
            else:
                self._clear_planned_path_items()
        elif name in ("layer_sparse_waypoints", "sparse_waypoints"):
            if visible:
                self._render_sparse_waypoints_to_scene()
            else:
                self._clear_sparse_waypoint_items()
        elif name in ("layer_waypoint_validation", "waypoint_validation"):
            if visible:
                self._render_waypoint_validation_to_scene()
            else:
                self._clear_waypoint_validation_items()
        self._canvas.viewport().update()

    # ===================================================================
    # 模式切换
    # ===================================================================

    def _on_mode_toggled(self, mode: str):
        """简洁模式 / 调试模式切换"""
        if mode == "clean":
            self._layer_manager.apply_clean_mode()
            self._clean_mode = True
            self._status_bar.show_message("已切换到简洁模式")
        else:
            self._layer_manager.apply_debug_mode()
            self._clean_mode = False
            self._status_bar.show_message("已切换到调试模式")
        self._sync_layer_checkboxes()
        self._canvas.refresh_scene()
        # ★ 重新渲染 graph items
        if self._current_stage == "graph":
            self._render_graph_to_scene()
        if self._mask_ignore_candidates:
            self._render_mask_candidates()
        self._canvas.viewport().update()

    def _visible_layer_names(self) -> list:
        names = []
        for name, layer in self._layer_manager.layers().items():
            if layer.visible and (layer.data is not None or name.startswith("layer_")):
                if layer.visible:
                    names.append(name)
        return names

    def _log_display_switch(self, phase: str, preset: str):
        """记录切换显示模式前后 Road Mask / working mask 状态，便于定位隐藏问题。

        显示模式切换只允许改变可见性，不能覆盖/清空数据。
        """
        lm = self._layer_manager
        try:
            layer = lm.layers().get("layer_road_mask")
            visible = layer.visible if layer is not None else None
            opacity = layer.opacity if layer is not None else None
            data = layer.data if layer is not None else None
            preview = layer.preview_data if layer is not None else None
            print(
                f"[DisplaySwitch][{phase}] preset={preset}\n"
                f"  mask_source={getattr(self, '_working_mask_source', None)}\n"
                f"  formal_ready={getattr(self, '_working_mask_formal_ready', None)}\n"
                f"  preview_only={getattr(self, '_working_mask_preview_only', None)}\n"
                f"  working_mask_path={getattr(self, '_working_road_mask_path', None)}\n"
                f"  working_mask_preview_path={getattr(self, '_working_road_mask_preview_path', None)}\n"
                f"  layer_road_mask.visible={visible}\n"
                f"  layer_road_mask.opacity={opacity}\n"
                f"  layer_road_mask.data type={type(data).__name__}\n"
                f"  preview_exists={preview is not None}\n"
                f"  selected_preview_path={getattr(self, '_working_road_mask_preview_path', None)}\n"
                f"  visible_layers={self._visible_layer_names()}"
            )
        except Exception as exc:  # 日志失败不能影响显示
            print(f"[DisplaySwitch][{phase}] 记录状态失败: {exc}")

    def _ensure_large_working_mask_for_display(self) -> bool:
        """大图调试/清爽显示前：优先用 working_road_mask，必要时重生 preview。"""
        lm = self._layer_manager
        if not lm.is_large_image_mode:
            return lm.ensure_working_mask_preview("layer_road_mask")

        # 从项目/内存路径补齐 working mask 路径
        project = self._large_image_project
        working_path = getattr(self, "_working_road_mask_path", None) or ""
        preview_path = getattr(self, "_working_road_mask_preview_path", None) or ""
        if project is not None:
            working_path = working_path or getattr(project, "working_road_mask_path", "") or ""
            preview_path = preview_path or getattr(project, "working_road_mask_preview_path", "") or ""
            if getattr(project, "mask_source", ""):
                self._working_mask_source = project.mask_source

        layer = lm.layers().get("layer_road_mask")
        has_data = layer is not None and layer.data is not None

        # 磁盘有 working mask、图层却空 → 加载 working（不是旧 global）
        if (not has_data) and working_path and os.path.isfile(working_path):
            try:
                arr = cv2.imread(working_path, cv2.IMREAD_GRAYSCALE)
                if arr is not None and arr.size > 0:
                    prev = None
                    if preview_path and os.path.isfile(preview_path):
                        prev = cv2.imread(preview_path, cv2.IMREAD_GRAYSCALE)
                    lm.set_layer_data("mask", arr, preview_data=prev)
                    self._working_road_mask_path = working_path
                    has_data = True
            except Exception as exc:
                print(f"[DisplaySwitch] 加载 working_road_mask 失败: {exc}")

        # preview 文件缺失但 working 存在 → 自动重生 preview 文件与图层 preview
        if working_path and os.path.isfile(working_path):
            if not preview_path or not os.path.isfile(preview_path):
                try:
                    arr = cv2.imread(working_path, cv2.IMREAD_GRAYSCALE)
                    if arr is not None:
                        pw, ph = lm.preview_width, lm.preview_height
                        if pw > 0 and ph > 0:
                            prev = cv2.resize(arr, (pw, ph), interpolation=cv2.INTER_NEAREST)
                        else:
                            prev = arr
                        out_dir = Path(working_path).parent
                        preview_path = str(out_dir / "working_road_mask_preview.png")
                        cv2.imwrite(preview_path, prev)
                        self._working_road_mask_preview_path = preview_path
                        if project is not None:
                            project.working_road_mask_preview_path = preview_path
                            try:
                                project.save()
                            except Exception:
                                pass
                        if layer is not None and isinstance(layer.data, np.ndarray):
                            layer.preview_data = prev
                            layer.pixmap = None
                except Exception as exc:
                    print(f"[DisplaySwitch] 重生 working_road_mask_preview 失败: {exc}")

        ok = lm.ensure_working_mask_preview("layer_road_mask")
        # 强制 Road Mask 可见 + 合理透明度
        road = lm.layers().get("layer_road_mask")
        if road is not None and ok:
            road.visible = True
            if road.opacity < 80:
                road.opacity = 115
        return ok

    def _on_clean_display(self):
        """一键清爽显示（只改可见性，不动数据）"""
        self._log_display_switch("before", "clean_display")
        self._ensure_large_working_mask_for_display()
        self._layer_manager.apply_clean_display()
        self._sync_layer_checkboxes()
        self._canvas.refresh_scene()
        if self._current_stage == "graph":
            self._render_graph_to_scene()
        self._canvas.viewport().update()
        self._log_display_switch("after", "clean_display")
        self._status_bar.show_message("清爽显示：只突出路网")

    def _on_debug_display(self):
        """一键调试显示（只改可见性，不动数据；优先显示 working Road Mask）"""
        self._log_display_switch("before", "debug_display")
        has_mask = self._ensure_large_working_mask_for_display()
        self._layer_manager.apply_debug_display()
        # 再次强制 Road Mask 可见（preset 不得把它关掉）
        road = self._layer_manager.layers().get("layer_road_mask")
        if road is not None and has_mask:
            road.visible = True
            if road.opacity < 80:
                road.opacity = 115
        self._sync_layer_checkboxes()
        self._canvas.refresh_scene()
        if self._current_stage == "graph":
            self._render_graph_to_scene()
        self._canvas.viewport().update()
        self._log_display_switch("after", "debug_display")
        if has_mask:
            src = getattr(self, "_working_mask_source", None) or "formal"
            hint = getattr(self, "_working_road_mask_preview_path", None)
            suffix = f"（{os.path.basename(hint)}）" if hint else ""
            self._status_bar.show_message(
                f"调试显示已开启，当前 Road Mask: {src}{suffix}"
            )
        else:
            self._status_bar.show_message(
                "当前没有 working Road Mask，请先生成或保存正式 mask。"
            )

    def _sync_layer_checkboxes(self):
        """同步左侧复选框与 LayerManager 状态"""
        self._tool_panel.sync_all_checkboxes(self._layer_manager)

    # ===================================================================
    # 工具交互处理（路网编辑核心）
    # ===================================================================

    def _on_tool_interaction(self, action: str, data):
        """处理画布发出的工具交互事件"""
        # undo/redo/delete 不需要 graph_editor，直接处理
        if action == "undo":
            self._on_global_undo()
            return
        elif action == "redo":
            self._on_global_redo()
            return
        elif action == "delete":
            self._handle_delete()
            return
        elif action == "brush_radius_changed":
            # 画笔半径变化（来自 [ / ] 键或 Ctrl+滚轮）
            new_radius = data
            self._param_panel._set_config("edit.brush_radius", new_radius)
            widget = self._param_panel._widgets.get("edit.brush_radius")
            if widget is not None:
                widget.blockSignals(True)
                widget.setValue(int(new_radius))
                widget.blockSignals(False)
            self._status_bar.show_message(f"画笔半径 = {int(new_radius)} px")
            return
        elif action == "regions_changed":
            payload = data or {}
            roi_count = int(payload.get("roi", len(self._canvas.get_roi_regions())))
            self._param_panel.update_counts(
                roi=roi_count,
                ignore=int(payload.get("ignore", len(self._canvas.get_ignore_regions()))),
            )
            self._refresh_roi_status_panel()
            if self._roi_draw_return_stage is not None and roi_count > self._roi_draw_baseline_count:
                self._finish_roi_drawing_session()
            return
        elif action == "roi_drawing_cancelled":
            if self._roi_draw_return_stage is not None:
                self._roi_draw_return_stage = None
                self.set_stage("segment")
                self._status_bar.show_message("已取消 ROI 绘制。")
            return
        elif action == "seed_changed":
            count = int((data or {}).get("count", self._canvas.get_main_road_seed_count()))
            msg = (data or {}).get("message") or f"主路种子线：{count} 笔"
            self._status_bar.show_message(msg)
            if hasattr(self._param_panel, "update_main_road_seed_count"):
                self._param_panel.update_main_road_seed_count(count)
            self._save_main_road_seed_strokes()
            self._apply_seed_width_settings_to_canvas()
            return
        elif action == "seed_status":
            msg = (data or {}).get("message") or ""
            if msg:
                self._status_bar.show_message(msg)
            if (data or {}).get("exit_tool"):
                self.set_tool("pan")
            return
        elif action == "mask_stroke_finished":
            mask = self._layer_manager.get_layer_data("mask")
            if isinstance(mask, np.ndarray) and mask.size:
                self._status_bar.update_road_ratio(
                    float(np.count_nonzero(mask)) / float(mask.size)
                )
            # ★ 手动画笔/橡皮修正必须写入 working mask 状态
            base = getattr(self, "_mask_edit_base", "") or ""
            src_now = getattr(self, "_working_mask_source", "") or ""
            if (base == "cleaned_working_mask"
                    or src_now in ("cleaned_working_mask", "manual_after_cleaned")):
                self._working_mask_source = "manual_after_cleaned"
                self._mask_edit_base = "cleaned_working_mask"
            elif src_now == "final_edited_mask":
                if self._mask_edit_base == "cleaned_working_mask":
                    self._working_mask_source = "manual_after_cleaned"
                else:
                    if not self._mask_edit_base:
                        self._mask_edit_base = "global_road_mask"
                    self._working_mask_source = "manual_edited"
            else:
                # 直接在 global/working 上修
                if not self._mask_edit_base:
                    self._mask_edit_base = "global_road_mask"
                self._working_mask_source = "manual_edited"
            self._working_mask_dirty = True
            self._working_mask_formal_ready = True
            self._working_mask_preview_only = False
            if self._large_image_project is not None:
                self._large_image_project.mask_source = self._working_mask_source
                self._large_image_project.mask_edit_base = self._mask_edit_base
                self._large_image_project.mask_dirty = True
            mode = (data or {}).get("mode", "add")
            tip = "橡皮" if mode == "erase" else "画笔"
            self._update_large_mask_status_bar(f"{tip}已写入")
            return
        elif action == "mask_edit_error":
            QMessageBox.critical(self, "Mask 精修失败", str(data))
            return

        if self._graph_editor is None:
            return
        ge = self._graph_editor

        if action == "press":
            self._handle_press(data)
        elif action == "move":
            self._handle_move(data)
        elif action == "release":
            self._handle_release(data)
        elif action == "hover":
            self._handle_hover(data)
        elif action == "confirm_manual_edge":
            # 折线补路确认 — 先推状态再执行
            self._history.push_state("graph_draw_edge")
            ge.confirm_manual_edge()
            self._render_graph_to_scene()
            self._update_graph_stats()
            warns = getattr(ge, "_last_validation_warnings", None) or []
            if not warns and hasattr(ge, "validate_graph_local"):
                mask = None
                try:
                    mask, _ = self.get_current_mask_array(
                        for_skeleton=False, require_full_resolution=False
                    )
                except Exception:
                    mask = None
                warns = ge.validate_graph_local(road_mask=mask)
            if warns:
                self._status_bar.show_message(f"折线补路已确认｜注意: {warns[0]}")
            else:
                self._status_bar.show_message("折线补路已确认（已保存完整 polyline）")
        elif action == "cancel_manual_edge":
            ge.cancel_manual_edge()
            self._render_graph_to_scene()
        elif action == "undo_manual_point":
            # Undo last point in manual edge drawing
            if ge._manual_edge_points:
                ge._manual_edge_points.pop()
                self._render_graph_to_scene()
        elif action == "clear_manual_points":
            ge._manual_edge_points.clear()
            self._render_graph_to_scene()
        elif action == "key_escape":
            # Esc pressed while in graph tool: cancel current operation
            self._handle_escape(data)
        elif action == "key_enter":
            # Enter pressed while in graph tool: confirm pending operation
            self._handle_key_enter(data)

    def _handle_press(self, data):
        ge = self._graph_editor
        tool = data.get("tool", "")
        # ★ 大图模式：canvas 发送预览坐标，需要转换为全局坐标再传给 GraphEditorQt
        raw_x, raw_y = data.get("x", 0), data.get("y", 0)
        x, y = self._layer_manager.preview_to_global(raw_x, raw_y)

        if tool == "graph_add_node":
            print(f"[DEBUG][Graph] click for add_node at image=({x},{y})")
            # ★ 推入全局撤销状态
            self._history.push_state("graph_add_node")
            nid = ge.add_node(x, y)
            ge.select_node(nid)
            self._render_graph_to_scene()
            self._update_graph_stats()
            self._status_bar.show_message(f"添加节点成功 (id={nid})，已入撤销栈")

        elif tool == "graph_delete_node":
            nid = ge.find_node_at(x, y)
            if nid is not None:
                ge.select_node(nid)
                self._render_graph_to_scene()
                from PySide6.QtWidgets import QMessageBox
                reply = QMessageBox.question(
                    self, "删除节点",
                    "删除该节点会同时删除相关边，是否继续？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._history.push_state("graph_delete_node")
                    ge.delete_node(nid)
                    ge.clear_selection()
                    self._render_graph_to_scene()
                    self._update_graph_stats()
                    self._status_bar.show_message(f"删除节点成功 (id={nid})，已入撤销栈")
            else:
                ge.clear_selection()
                self._render_graph_to_scene()

        elif tool == "graph_delete_edge":
            eid = ge.find_edge_at(x, y)
            if eid is not None:
                ge.select_edge(eid)
                self._render_graph_to_scene()
                from PySide6.QtWidgets import QMessageBox
                reply = QMessageBox.question(
                    self, "删除边",
                    "确认删除选中的边？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._history.push_state("graph_delete_edge")
                    ge.delete_edge(eid)
                    ge.clear_selection()
                    self._render_graph_to_scene()
                    self._update_graph_stats()
                    self._status_bar.show_message(f"删除边成功 (id={eid})，已入撤销栈")
            else:
                ge.clear_selection()
                self._render_graph_to_scene()

        elif tool == "graph_add_edge":
            nid = ge.find_node_at(x, y)
            if nid is not None:
                if ge._edge_start_node is None:
                    # 选择起点
                    ge._edge_start_node = nid
                    ge.select_node(nid)
                    print(f"[DEBUG][Graph] selected start node id={nid}")
                    self._status_bar.show_message(f"已选起点: 节点 {nid}，请点击终点节点")
                    self._render_graph_to_scene()
                else:
                    if nid == ge._edge_start_node:
                        # 点击同一个节点 → 取消
                        print(f"[DEBUG][Graph] same node clicked, cancel edge start")
                        ge._edge_start_node = None
                        ge.clear_selection()
                        self._status_bar.show_message("已取消添加边")
                        self._render_graph_to_scene()
                    else:
                        # 选择终点，添加边
                        start = ge._edge_start_node
                        ge._edge_start_node = None
                        ge.clear_selection()
                        print(f"[DEBUG][Graph] selected end node id={nid}")
                        self._history.push_state("graph_add_edge")
                        result = ge.add_edge(start, nid)
                        if result is not None:
                            self._render_graph_to_scene()
                            self._update_graph_stats()
                            self._status_bar.show_message(f"添加边成功 ({start}→{nid})，已入撤销栈")
            else:
                # 点击位置没有节点
                if ge._edge_start_node is not None:
                    self._status_bar.show_message("请点击已有节点作为终点，或先添加节点。")
                else:
                    self._status_bar.show_message("请点击已有节点，或先添加节点。")

        elif tool == "graph_move_node":
            nid = ge.find_node_at(x, y)
            if nid is not None:
                # ★ 保存拖拽前状态（在 release 时推入）
                ge.set_dragging(nid)
                ge.select_node(nid)
            else:
                ge.clear_selection()
            self._render_graph_to_scene()

        elif tool == "graph_merge_nodes":
            nid = ge.find_node_at(x, y)
            if nid is not None:
                if len(ge.selected_nodes) < 2:
                    if nid not in ge.selected_nodes:
                        ge.select_node(nid)
                        print(f"[DEBUG][Graph] merge select node id={nid}")
                        self._status_bar.show_message(f"已选节点 {nid}，请再选第二个节点 (共{len(ge.selected_nodes)})")
                    else:
                        ge._selected_nodes.remove(nid)
                        print(f"[DEBUG][Graph] merge deselect node id={nid}")
                self._render_graph_to_scene()
                # 两个已选 → 执行合并
                if len(ge.selected_nodes) >= 2:
                    ids = list(ge.selected_nodes)[:2]
                    ge.clear_selection()
                    self._history.push_state("graph_merge_nodes")
                    ge.merge_nodes(ids[0], ids[1])
                    self._render_graph_to_scene()
                    self._update_graph_stats()
                    self._status_bar.show_message(f"已合并节点 {ids[0]} 和 {ids[1]}")
            else:
                ge.clear_selection()
                self._render_graph_to_scene()

        elif tool == "graph_draw_edge":
            ge.add_manual_point(x, y)
            print(f"[DEBUG][Graph] manual edge add point ({x},{y}), total={len(ge._manual_edge_points)}")
            self._render_graph_to_scene()

    def _handle_move(self, data):
        ge = self._graph_editor
        raw_x, raw_y = data.get("x", 0), data.get("y", 0)
        # ★ 大图模式：canvas 发送预览坐标，需要转换为全局坐标
        x, y = self._layer_manager.preview_to_global(raw_x, raw_y)
        if ge._dragging_node is not None:
            ge.move_node(ge._dragging_node, x, y)
            self._render_graph_to_scene()

    def _handle_release(self, data):
        ge = self._graph_editor
        if ge._dragging_node is not None:
            nid = ge._dragging_node
            self._history.push_state("graph_move_node")
            ge.set_dragging(None)
            self._render_graph_to_scene()
            self._update_graph_stats()
            self._status_bar.show_message(f"移动节点成功 (id={nid})，已入撤销栈")

    def _handle_hover(self, data):
        ge = self._graph_editor
        raw_x, raw_y = data.get("x", 0), data.get("y", 0)
        # ★ 大图模式：canvas 发送预览坐标，需要转换为全局坐标
        x, y = self._layer_manager.preview_to_global(raw_x, raw_y)
        tool = data.get("tool", "")
        # 只在移动节点或添加边时需要悬停高亮
        if tool in ("graph_move_node", "graph_add_edge", "graph_merge_nodes",
                     "graph_delete_node", "graph_delete_edge"):
            nid = ge.find_node_at(x, y)
            eid = None if nid is not None else ge.find_edge_at(x, y)
            if nid != ge._hovered_node or eid != ge._hovered_edge:
                ge.set_hover(nid, eid)
                if nid is not None:
                    degree = sum(
                        1 for edge in ge.edges
                        if edge.get("start") == nid or edge.get("end") == nid
                    )
                    self._status_bar.show_message(
                        f"node_id={nid}, degree={degree}"
                    )
                self._render_graph_to_scene()

    def _handle_escape(self, data):
        """Esc: 取消当前操作（清除选中、取消边起点）"""
        ge = self._graph_editor
        if ge._edge_start_node is not None:
            ge._edge_start_node = None
            self._status_bar.show_message("已取消添加边")
        if ge._manual_edge_points:
            ge.cancel_manual_edge()
            self._status_bar.show_message("已取消手动画边")
        if ge._dragging_node is not None:
            ge.set_dragging(None)
        ge.clear_selection()
        self._render_graph_to_scene()

    def _handle_key_enter(self, data):
        """Enter: 确认当前操作"""
        ge = self._graph_editor
        tool = data.get("tool", "")
        if tool == "graph_draw_edge" and len(ge._manual_edge_points) >= 2:
            ge.confirm_manual_edge()
            self._render_graph_to_scene()
            self._update_graph_stats()
            self._status_bar.show_message("手动画边已确认")

    def _handle_delete(self):
        """Delete 键：删除当前选中的节点或边"""
        ge = self._graph_editor
        if ge is None:
            return
        if ge.selected_edges:
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, "删除边",
                f"确认删除 {len(ge.selected_edges)} 条选中的边？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._history.push_state("graph_delete_edge")
                for eid in list(ge.selected_edges):
                    ge.delete_edge(eid)
        elif ge.selected_nodes:
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, "删除节点",
                "删除选中节点会同时删除相关边，是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._history.push_state("graph_delete_node")
                for nid in list(ge.selected_nodes):
                    ge.delete_node(nid)
        ge.clear_selection()
        self._render_graph_to_scene()
        self._update_graph_stats()

    def _update_graph_stats(self):
        if self._graph_editor is None:
            return
        stats = self._graph_editor.get_stats()
        self._param_panel.update_graph_stats(stats)
        self._status_bar.update_nodes(stats["node_count"])
        self._status_bar.update_edges(stats["edge_count"])

    # ===================================================================
    # Graph 场景渲染
    # ===================================================================

    def refresh_graph_layer(self):
        """统一刷新函数：清空并重绘所有 graph 节点和边"""
        self._render_graph_to_scene()
        self._update_graph_stats()
        self._canvas.viewport().update()

    def _render_graph_to_scene(self):
        """将 GraphEditorQt 的节点和边渲染到 QGraphicsScene 上。
        
        关键：GraphEditorQt 中的所有坐标均为原图全局像素坐标。
        在大图模式下，需要转换为预览图坐标才能在场景中正确显示。
        """
        if self._graph_editor is None:
            return

        scene = self._canvas.scene()
        if scene is None:
            return
        ge = self._graph_editor
        lm = self._layer_manager  # ★ 用于坐标转换

        degree = {n["id"]: 0 for n in ge.nodes}
        adjacency = {n["id"]: set() for n in ge.nodes}
        for edge in ge.edges:
            a, b = edge.get("start"), edge.get("end")
            if a in degree and b in degree:
                degree[a] += 1
                degree[b] += 1
                adjacency[a].add(b)
                adjacency[b].add(a)
        component_of = {}
        component_index = 0
        for node in ge.nodes:
            nid = node["id"]
            if nid in component_of:
                continue
            stack = [nid]
            while stack:
                current = stack.pop()
                if current in component_of:
                    continue
                component_of[current] = component_index
                stack.extend(adjacency.get(current, ()))
            component_index += 1
        component_colors = [
            QColor(137, 180, 250), QColor(166, 227, 161),
            QColor(250, 179, 135), QColor(203, 166, 247),
            QColor(249, 226, 175), QColor(148, 226, 213),
        ]

        # 清除旧的 graph items
        for item in self._graph_node_items.values():
            safe_remove_scene_item(scene, item)
        for item in self._graph_edge_items.values():
            safe_remove_scene_item(scene, item)
        for item in self._graph_manual_preview_items:
            safe_remove_scene_item(scene, item)
        self._graph_node_items.clear()
        self._graph_edge_items.clear()
        self._graph_manual_preview_items.clear()
        for item in self._graph_endpoint_items:
            safe_remove_scene_item(scene, item)
        self._graph_endpoint_items.clear()

        # ---- 渲染边 ----
        for e in ge.edges:
            pts = e.get("points_pixel", [])
            if len(pts) < 2:
                continue
            eid = e["id"]
            source = e.get("source", "auto")
            enabled = e.get("enabled", True)
            if not enabled:
                continue

            # 选中/悬停颜色
            if eid in ge.selected_edges:
                color = QColor(255, 213, 79)   # 金色选中高亮
                width = 3
                z = self._canvas.ZVAL_SELECTED
            elif eid == ge.hovered_edge:
                color = QColor(255, 255, 200)  # 浅黄悬停
                width = 3
                z = self._canvas.ZVAL_SELECTED - 1
            elif source == "auto":
                comp = component_of.get(e.get("start"), 0)
                color = component_colors[comp % len(component_colors)]
                width = 2
                z = self._canvas.ZVAL_AUTO_EDGE
            else:
                color = QColor(255, 184, 108)  # 橙色 — 人工 edge
                width = 2.5
                z = self._canvas.ZVAL_MANUAL_EDGE

            pen = QPen(color, width)
            pen.setCosmetic(True)

            from PySide6.QtGui import QPainterPath
            path = QPainterPath()
            # ★ 全局坐标 → 预览坐标转换
            px, py = lm.global_to_preview_f(pts[0][0], pts[0][1])
            path.moveTo(px, py)
            for pt in pts[1:]:
                px, py = lm.global_to_preview_f(pt[0], pt[1])
                path.lineTo(px, py)
            item = QGraphicsPathItem(path)
            item.setPen(pen)
            item.setZValue(z)
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            item.setToolTip(
                f"edge_id={eid}, component={component_of.get(e.get('start'), 0) + 1}"
            )
            scene.addItem(item)
            self._graph_edge_items[eid] = item

        # ---- 渲染节点 ----
        for n in ge.nodes:
            nid = n["id"]
            source = n.get("source", "auto")

            radius = 6

            if nid in ge.selected_nodes:
                color = QColor(68, 153, 255)    # 蓝色选中
                outline = QColor(255, 255, 255)
                ol_width = 2
                z = self._canvas.ZVAL_SELECTED
            elif nid == ge.hovered_node:
                color = QColor(255, 255, 180)   # 浅黄悬停
                outline = QColor(255, 255, 0)
                ol_width = 2
                z = self._canvas.ZVAL_SELECTED - 1
            elif degree.get(nid, 0) == 1:
                color = QColor(255, 70, 70)
                outline = QColor(255, 235, 235)
                ol_width = 2
                z = self._canvas.ZVAL_NODE + 1
            elif source == "auto":
                color = QColor(80, 250, 123)    # 绿色自动节点
                outline = QColor(40, 180, 80)
                ol_width = 1
                z = self._canvas.ZVAL_NODE
            else:
                color = QColor(137, 180, 250)   # 蓝色手动节点
                outline = QColor(80, 130, 200)
                ol_width = 1.5
                z = self._canvas.ZVAL_NODE

            # ★ 全局坐标 → 预览坐标转换
            nx, ny = lm.global_to_preview_f(n["x"], n["y"])
            item = QGraphicsEllipseItem(nx - radius, ny - radius,
                                         radius * 2, radius * 2)
            item.setBrush(QBrush(color))
            item.setPen(QPen(outline, ol_width))
            item.setZValue(z)
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            item.setToolTip(
                f"node_id={nid}, degree={degree.get(nid, 0)}, "
                f"component={component_of.get(nid, 0) + 1}"
            )
            scene.addItem(item)
            self._graph_node_items[nid] = item

        # ---- 渲染手动画边预览（亮黄色虚线 + 品红端点） ----
        if ge.manual_edge_points:
            pts = ge.manual_edge_points
            for x, y in pts:
                # ★ 全局坐标 → 预览坐标转换
                dx, dy = lm.global_to_preview_f(x, y)
                dot = QGraphicsEllipseItem(dx - 4, dy - 4, 8, 8)
                dot.setBrush(QBrush(QColor(255, 213, 79)))   # 金色预览点
                dot.setPen(QPen(Qt.PenStyle.NoPen))
                dot.setZValue(self._canvas.ZVAL_SELECTED)
                dot.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                scene.addItem(dot)
                self._graph_manual_preview_items.append(dot)

            if len(pts) >= 2:
                from PySide6.QtGui import QPainterPath
                path = QPainterPath()
                px, py = lm.global_to_preview_f(pts[0][0], pts[0][1])
                path.moveTo(px, py)
                for pt in pts[1:]:
                    px, py = lm.global_to_preview_f(pt[0], pt[1])
                    path.lineTo(px, py)
                item = QGraphicsPathItem(path)
                pen = QPen(QColor(255, 234, 0), 2, Qt.PenStyle.DashLine)  # 亮黄虚线
                pen.setCosmetic(True)
                pen.setDashPattern([6, 4])
                item.setPen(pen)
                item.setZValue(self._canvas.ZVAL_SELECTED)
                item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                scene.addItem(item)
                self._graph_manual_preview_items.append(item)

        self._canvas.viewport().update()

    def _render_reference_graph_to_scene(self):
        """将 SAM-Road 原始 graph（参考图层）渲染到 QGraphicsScene。

        与 final_graph 使用不同的视觉样式：
        - 金色半透明边线
        - 空心圆节点
        - 默认不响应鼠标交互
        """
        from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsPathItem
        from PySide6.QtGui import QPen, QBrush, QPainterPath

        scene = self._canvas.scene()
        if scene is None:
            return
        lm = self._layer_manager

        # 清除旧的参考图 items
        for item in self._ref_node_items:
            safe_remove_scene_item(scene, item)
        for item in self._ref_edge_items:
            safe_remove_scene_item(scene, item)
        self._ref_node_items.clear()
        self._ref_edge_items.clear()

        if not self._reference_graph_nodes and not self._reference_graph_edges:
            return

        # 渲染边 — 金色虚线
        for e in self._reference_graph_edges:
            path_data = e.get("path", [])
            if len(path_data) < 2:
                continue
            if "points_pixel" in e:
                pts = e["points_pixel"]
            else:
                # path 是 [[y,x],...] 格式，需转为 [[x,y],...]
                pts = [[p[1], p[0]] for p in path_data]

            pen = QPen(QColor(255, 200, 60), 1.5)
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.DashLine)

            qpath = QPainterPath()
            px, py = lm.global_to_preview_f(pts[0][0], pts[0][1])
            qpath.moveTo(px, py)
            for pt in pts[1:]:
                px, py = lm.global_to_preview_f(pt[0], pt[1])
                qpath.lineTo(px, py)
            item = QGraphicsPathItem(qpath)
            item.setPen(pen)
            item.setZValue(self._canvas.ZVAL_AUTO_EDGE - 2)  # 低于 final_graph
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            scene.addItem(item)
            self._ref_edge_items.append(item)

        # 渲染节点 — 金色空心圆
        for n in self._reference_graph_nodes:
            nx, ny = lm.global_to_preview_f(n["x"], n["y"])
            r = 4
            item = QGraphicsEllipseItem(nx - r, ny - r, r * 2, r * 2)
            item.setBrush(QBrush(QColor(255, 200, 60, 100)))
            item.setPen(QPen(QColor(200, 150, 30), 1.5))
            item.setZValue(self._canvas.ZVAL_NODE - 2)
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            scene.addItem(item)
            self._ref_node_items.append(item)

        self._canvas.viewport().update()

    def _clear_reference_graph_items(self):
        """清除参考图层的 scene items。"""
        scene = self._canvas.scene() if self._canvas else None
        if scene is None:
            return
        for item in self._ref_node_items:
            safe_remove_scene_item(scene, item)
        for item in self._ref_edge_items:
            safe_remove_scene_item(scene, item)
        self._ref_node_items.clear()
        self._ref_edge_items.clear()

    # ===================================================================
    # 文件操作
    # ===================================================================

    def _on_open_image(self):
        if ((self._segmentation_thread is not None and self._segmentation_thread.isRunning())
                or (self._pipeline_thread is not None and self._pipeline_thread.isRunning())):
            QMessageBox.information(self, "后台任务运行中", "请先取消并等待后台任务结束。")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "打开影像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*)"
        )
        if not path:
            return

        try:
            # ============================================================
            # Step 1: 安全读取尺寸（不解码像素）
            # ============================================================
            from roadnet.large_image_project import (
                get_image_size_safe, is_large_image,
                estimate_image_memory_mb, generate_preview_safe,
                write_open_large_image_log, check_memory_budget,
                LargeImageProject, current_process_memory_mb,
                DEFAULT_TILE_SIZE,
                RISK_MESSAGES, RISK_POLICIES,
                RISK_BLOCKED, RISK_HIGH_RISK,
            )
            import time
            import traceback as tb_module

            memory_before = current_process_memory_mb()
            file_size_mb = os.path.getsize(path) / (1024.0 * 1024.0)
            steps = []
            open_log_dir = ""
            error_stage = ""
            risk_level = ""
            raw_rgb_mb = 0.0

            w, h = get_image_size_safe(path)
            steps.append("get_image_size_safe")

            # 统一风险评估（无论大图小图都做，确保 high_risk 有 key）
            mem = check_memory_budget(w, h)
            risk_level = mem["risk_level"]
            raw_rgb_mb = mem["raw_rgb_mb"]
            is_large = is_large_image(w, h)

            if is_large:
                # ========================================================
                # 大图模式：极简打开流程
                # ========================================================
                error_stage = "large_image_entry"
                steps.append("large_image_entry")
                print(f"[OpenImage] 大图模式: {w}x{h}, "
                      f"文件={file_size_mb:.1f}MB, "
                      f"RGB内存估算={mem['raw_rgb_mb']:.1f}MB, "
                      f"risk_level={risk_level}")

                # 只有 blocked 才拒绝打开
                if risk_level == RISK_BLOCKED:
                    error_stage = "memory_risk_blocked"
                    steps.append("memory_risk_blocked")
                    msg = RISK_MESSAGES.get(risk_level, f"未知风险等级: {risk_level}")
                    raise RuntimeError(
                        f"{msg}\n"
                        f"原始尺寸: {w} x {h} px\n"
                        f"估算 RGB 内存: {mem['raw_rgb_mb']:.1f} MB (上限: {mem['max_critical_mb']} MB)"
                    )

                # high_risk / warning 都继续走 preview 流程，不阻止
                if risk_level == RISK_HIGH_RISK:
                    print(f"[OpenImage] high_risk: 强制 preview-only 模式")
                    steps.append("risk_high_risk_ack")
                elif mem.get("high_risk", False):
                    # backward compat — ensure we enter preview-only
                    print(f"[OpenImage] legacy high_risk flag: 强制 preview-only 模式")
                    steps.append("legacy_high_risk_ack")

                # 内存预算提示
                policy_msg = RISK_MESSAGES.get(risk_level, RISK_MESSAGES[RISK_HIGH_RISK])
                if risk_level == RISK_HIGH_RISK or mem.get("high_risk", False):
                    QMessageBox.information(
                        self, "大图预览模式",
                        f"{policy_msg}\n\n"
                        f"当前图像约需要 {mem['raw_rgb_mb']:.1f} MB 原始 RGB 内存\n"
                        f"原始尺寸: {w} x {h} px"
                    )

                # Step 2: 生成 large_image_project 目录和 preview
                from datetime import datetime
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_root = os.path.join(os.getcwd(), "outputs", "large_image_projects")
                project_dir = os.path.join(output_root, f"project_{stamp}")

                # 避免冲突
                suffix = 1
                base_dir = project_dir
                while os.path.exists(project_dir):
                    suffix += 1
                    project_dir = f"{base_dir}_{suffix}"
                os.makedirs(os.path.join(project_dir, "logs"), exist_ok=True)

                steps.append("create_project_dir")

                # Step 3: 生成 preview.png（安全方法）
                error_stage = "generate_preview"
                preview_path = os.path.join(project_dir, "preview.png")
                try:
                    preview = generate_preview_safe(path, preview_path, max_side=3000)
                    preview_h, preview_w = preview.shape[:2]
                    preview_scale = min(preview_w / w, preview_h / h)
                    steps.append("generate_preview_safe")
                except Exception as prev_err:
                    raise RuntimeError(f"生成 preview 失败: {prev_err}") from prev_err

                # Step 4: 保存 large_image_project.json
                project = LargeImageProject(
                    image_path=str(os.path.abspath(path)),
                    image_width=w,
                    image_height=h,
                    preview_path=str(preview_path),
                    preview_scale=float(preview_scale),
                    tile_size=DEFAULT_TILE_SIZE,
                    tile_overlap=256,
                    tile_index_path=str(os.path.join(project_dir, "tile_index.json")),
                    project_dir=str(project_dir),
                )
                project.save()
                steps.append("save_large_image_project_json")

                # Step 5: UI 只加载 preview
                self._layer_manager.load_large_image_preview(
                    path, preview_path,
                    (w, h), preview_scale,
                )
                self._canvas.refresh_scene()
                self._canvas.fit_to_window()
                steps.append("load_preview_to_ui")

                # Step 6: 设置项目状态
                self._valid_image_mask = None
                self._valid_mask_report = {}
                self._clear_skeleton_state(clear_layer=False)
                self._project_manager.data.image_path = path
                self._project_manager.data.image_width = preview_w
                self._project_manager.data.image_height = preview_h
                self._project_manager.data.original_width = w
                self._project_manager.data.original_height = h
                self._project_manager.data.large_image_mode = True
                self._project_manager.data.preview_scale = preview_scale
                self._project_manager.data.large_image_project_path = str(project.project_path)
                self._project_manager.data.tile_index_path = project.tile_index_path
                self._project_manager.data.output_dir = project_dir
                self._project_manager.mark_dirty()
                steps.append("set_project_state")

                # Step 7: 初始化轻量 GraphEditor + 撤销（但只用预览，不加载任何全分辨率数据）
                from roadnet.graph_editor_qt import GraphEditorQt
                self._graph_editor = GraphEditorQt(
                    image_size=(w, h),  # 始终使用全局像素尺寸
                    pixel_resolution_m=self._project_manager.data.pixel_resolution_m
                )
                self._history.clear()
                self._history.inject(self._canvas, self._layer_manager, self._graph_editor, self)
                self._history.push_state("large_image_loaded")
                steps.append("init_graph_editor")

                # Step 8: 只显示 preview，禁止加载任何 full-size layer
                # 不调用 apply_stage_preset("import") — 那会尝试加载旧图层
                self._sync_layer_checkboxes()
                self._canvas.viewport().update()
                self._set_nav_active("import")

                # 注意：大图模式故意不加载 mask、skeleton、graph、calibration
                # 这些 layer 需要用户手动触发
                steps.append("ui_ready")

                # Step 9: 保存大图 project 到引用
                self._large_image_project = project
                steps.append("large_project_referenced")

                # Step 10: 写打开日志
                error_stage = "write_log"
                memory_after = current_process_memory_mb()
                open_log_dir = project_dir
                log_path = write_open_large_image_log(
                    project_dir=project_dir,
                    image_path=path,
                    file_size_mb=file_size_mb,
                    width=w, height=h,
                    preview_path=preview_path,
                    preview_width=preview_w,
                    preview_height=preview_h,
                    preview_scale=preview_scale,
                    memory_before_mb=memory_before,
                    memory_after_mb=memory_after,
                    steps_completed=steps,
                    risk_level=risk_level,
                    raw_rgb_mb=raw_rgb_mb,
                    stage=error_stage,
                )
                steps.append(f"log_written:{log_path}")

                # Step 11: 状态栏显示大图信息
                self._status_bar.show_message(
                    f"大图模式: {os.path.basename(path)} | "
                    f"原始: {w}x{h} | 预览: {preview_w}x{preview_h} | "
                    f"缩放: {preview_scale:.4f}"
                )

                # 不自动启动 tile index 生成
                # 用户可以通过菜单手动触发
                print(f"[OpenImage] 大图打开完成, steps={steps}")
                print(f"[OpenImage] 日志: {log_path}")

            else:
                # ========================================================
                # 普通图片：原有流程不变
                # ========================================================
                self._layer_manager.load_image(path)
                self._valid_image_mask = None
                self._valid_mask_report = {}
                self._clear_skeleton_state(clear_layer=False)
                self._project_manager.data.image_path = path

                preview_w, preview_h = self._layer_manager.image_size
                original_w, original_h = self._layer_manager.original_size
                preview_scale = self._layer_manager.preview_scale

                self._project_manager.data.image_width = preview_w
                self._project_manager.data.image_height = preview_h
                self._project_manager.data.original_width = original_w
                self._project_manager.data.original_height = original_h
                self._project_manager.data.large_image_mode = False
                self._project_manager.data.preview_scale = preview_scale
                self._project_manager.mark_dirty()

                self._status_bar.show_message(f"已加载影像: {os.path.basename(path)} ({preview_w}x{preview_h})")

                from roadnet.graph_editor_qt import GraphEditorQt
                self._graph_editor = GraphEditorQt(
                    image_size=(original_w, original_h),
                    pixel_resolution_m=self._project_manager.data.pixel_resolution_m
                )
                self._history.clear()
                self._history.inject(self._canvas, self._layer_manager, self._graph_editor, self)
                self._history.push_state("initial")

                self._status_bar.update_resolution(self._project_manager.data.pixel_resolution_m)

                self._layer_manager.apply_stage_preset("import")
                self._sync_layer_checkboxes()
                self._canvas.viewport().update()
                self._set_nav_active("import")

                self._try_auto_load_calibration(path)

        except Exception as e:
            tb = tb_module.format_exc()
            print(f"[OpenImage] 打开失败:\n{tb}")

            # 安全获取异常上下文变量
            _is_large = locals().get("is_large", False)
            _open_log_dir = locals().get("open_log_dir", "")
            _w = locals().get("w", 0)
            _h = locals().get("h", 0)
            _file_size_mb = locals().get("file_size_mb", 0.0)
            _memory_before = locals().get("memory_before", None)
            _steps = locals().get("steps", [])
            _error_stage = locals().get("error_stage", "unknown")
            _risk_level = locals().get("risk_level", "unknown")
            _raw_rgb_mb_val = locals().get("raw_rgb_mb", 0.0)

            # 写入错误日志（含完整 traceback）
            log_written = ""
            if _is_large and _open_log_dir:
                try:
                    from roadnet.large_image_project import write_open_large_image_log
                    log_written = write_open_large_image_log(
                        project_dir=_open_log_dir,
                        image_path=path,
                        file_size_mb=_file_size_mb,
                        width=_w,
                        height=_h,
                        preview_path="",
                        preview_width=0, preview_height=0,
                        preview_scale=0,
                        memory_before_mb=_memory_before,
                        steps_completed=_steps,
                        error_traceback=tb,
                        risk_level=_risk_level,
                        raw_rgb_mb=_raw_rgb_mb_val,
                        stage=_error_stage,
                    )
                except Exception:
                    pass

            # 清理半生成的 QPixmap，回到空项目状态
            self.clear_project_state()
            self._canvas.refresh_scene()
            self._canvas.fit_to_window()

            # 改进的错误弹窗
            if _is_large:
                error_type = type(e).__name__
                error_msg = (
                    f"大图打开失败。\n\n"
                    f"阶段：{_error_stage}\n"
                    f"错误类型：{error_type}\n"
                    f"错误内容：{e}\n"
                    f"风险等级：{_risk_level}\n"
                    f"估算内存：{_raw_rgb_mb_val:.1f} MB\n"
                    f"图像尺寸：{_w} x {_h}\n\n"
                    f"详情见日志：\n{log_written or _open_log_dir + '/logs/open_large_image.log'}"
                )
            else:
                error_msg = f"无法加载影像:\n{e}"
            QMessageBox.critical(self, "错误", error_msg)

    def _try_auto_load_calibration(self, image_path: str):
        """尝试从影像所在目录或 outputs 目录自动加载 calibration.json。

        加载后：
        1. 恢复 geo_calibration 状态（含 is_valid）
        2. 更新坐标校准面板
        3. 更新状态栏
        4. 任务点模块可以直接使用
        """
        from pathlib import Path

        # 搜索优先级：
        # 1. 影像同目录下的 calibration.json
        # 2. outputs/calibration.json（项目目录）
        candidates = []
        if image_path:
            img_dir = os.path.dirname(os.path.abspath(image_path))
            candidates.append(os.path.join(img_dir, "calibration.json"))
        # outputs 目录
        candidates.append(os.path.join(os.getcwd(), "outputs", "calibration.json"))

        for cal_path in candidates:
            if not os.path.exists(cal_path):
                continue
            try:
                loaded = self._geo_calibration.load(cal_path)
                if loaded and self._geo_calibration.is_valid:
                    print(f"[Calibration] 自动加载 calibration.json: {cal_path}")
                    print(f"  enabled={self._geo_calibration.enabled}, "
                          f"valid={self._geo_calibration.is_valid}, "
                          f"cp_count={len(self._geo_calibration.control_points)}, "
                          f"mode={self._geo_calibration.transform_mode}")

                    # 更新 UI
                    self._param_panel.update_calibration_ui(self._geo_calibration)
                    self._update_calibration_corner_markers()
                    self._status_bar.update_resolution(
                        self._geo_calibration.pixel_resolution_estimated_m
                        or self._project_manager.data.pixel_resolution_m,
                        calibrated=True
                    )
                    cal_warning = self._geo_calibration.get_calibrated_warning()
                    if cal_warning:
                        print(f"[Calibration] {cal_warning}")
                    self._status_bar.show_message(
                        f"已自动加载地理标定: {self._geo_calibration.get_mode_label()} "
                        f"({len(self._geo_calibration.control_points)} 控制点)"
                    )
                    return
            except Exception as e:
                print(f"[Calibration] 加载 {cal_path} 失败: {e}")
                continue

        print("[Calibration] 未找到有效的 calibration.json，标定状态保持未标定。")

    def _on_new_project(self):
        if ((self._segmentation_thread is not None and self._segmentation_thread.isRunning())
                or (self._pipeline_thread is not None and self._pipeline_thread.isRunning())):
            QMessageBox.information(self, "后台任务运行中", "请先取消并等待后台任务结束。")
            return
        self.clear_project_state()
        self._canvas.refresh_scene()
        self._canvas.fit_to_window()
        self._status_bar.show_message("已创建新项目")

    def clear_project_state(self):
        self._history.clear()
        self._large_image_project = None  # ★ 清除大图项目引用
        self._formal_mask_meta = {}  # 清除正式 mask 注册元数据
        for name in ["mask", "roi", "ignore", "edit", "skeleton",
                      "draft_graph", "final_graph", "planned_path",
                      "valid_image_mask", "preview_seg_mask",
                      "layer_preview_segmentation",
                      "sample_points", "graph", "path",
                      "layer_road_mask", "layer_roi", "layer_ignore",
                      "layer_skeleton", "layer_skeleton_nodes",
                      "layer_draft_graph", "layer_final_graph",
                      "layer_reference_graph", "reference_graph",
                      "layer_planned_path", "layer_sparse_waypoints",
                      "layer_sample_points",
                      "layer_task_points"]:
            self._layer_manager.clear_layer(name)
        self._layer_manager.clear_image()
        self._canvas.clear_samples()
        self._canvas.clear_roi_and_ignore()

        # ★ 清除参考图数据
        self._reference_graph_nodes = []
        self._reference_graph_edges = []
        self._clear_reference_graph_items()

        # ★ 清除 SAM-Road 单图推理包额外图层
        self._samroad_single_itsc_mask = None
        self._samroad_single_viz = None
        self._valid_image_mask = None
        self._valid_mask_report = {}
        self._clear_mask_candidate_items()
        self._clear_graph_repair_items()
        self._mask_ignore_candidates = []
        self._graph_repair_candidates = []
        self._mask_candidate_signature = None
        self._graph_repair_signature = None
        self._last_graph_diagnostics = {}
        self._last_auto_repair_output_dir = None
        self._last_mask_filter_output_dir = None
        self._last_mask_filter_report = {}
        self._mask_before_auto_ignore = None
        self._param_panel.set_mask_candidate_apply_enabled(
            True, "请先执行自动筛选；只会应用 confidence >= 0.90 的安全候选。"
        )
        self.raw_skeleton = None
        self.optimized_skeleton = None
        self.current_skeleton = None
        self.skeleton_state = "none"
        self._last_samroad_source_dir = None  # ★ 用于恢复原始 road_mask
        self._clear_samroad_single_overlay_items()

        # ★ 清除任务点和规划数据
        self._task_points = []
        self._snapped_points = []
        self.snapped_task_points = []
        self._reset_planned_path_data()
        self._clear_task_point_items()
        self._clear_planned_path_items()
        self._canvas.refresh_scene()
        self._project_manager.new_project()

        # ★ 重置路网编辑器
        from roadnet.graph_editor_qt import GraphEditorQt
        self._graph_editor = GraphEditorQt()
        self._history.inject(self._canvas, self._layer_manager, self._graph_editor, self)
        self._render_graph_to_scene()

        self._current_stage = "import"
        self._stage_completed.clear()
        for sid, btn in self._nav_buttons.items():
            btn.setProperty("stageStatus", "normal")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self._status_bar.update_tool("pan")
        self._status_bar.update_road_ratio(None)
        self._status_bar.update_nodes(None)
        self._status_bar.update_edges(None)
        self._status_bar.show_message("项目已清空")


    # ===================================================================
    # 项目操作
    # ===================================================================

    def _on_open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开项目", "",
            "RoadNet 项目 (*.roadnet.json);;JSON 文件 (*.json);;所有文件 (*)"
        )
        if not path:
            return

        data = self._project_manager.load(path)
        if data:
            self._clear_mask_candidate_items()
            self._clear_graph_repair_items()
            self._mask_ignore_candidates = []
            self._graph_repair_candidates = []
            self._mask_candidate_signature = None
            self._graph_repair_signature = None
            self._last_mask_filter_output_dir = None
            self._last_mask_filter_report = {}
            self._mask_before_auto_ignore = None
            large_project_path = getattr(data, "large_image_project_path", "")
            if (data.large_image_mode and large_project_path
                    and os.path.isfile(large_project_path)):
                from roadnet.large_image_project import LargeImageProject, load_tile_index
                large_project = LargeImageProject.load(large_project_path)
                tile_index = (
                    load_tile_index(large_project.tile_index_path)
                    if large_project.tile_index_path and os.path.isfile(large_project.tile_index_path)
                    else {}
                )
                self._activate_large_image_project(
                    large_project, tile_index, reload_preview=True,
                )
            elif data.image_path and os.path.exists(data.image_path):
                self._layer_manager.load_image(data.image_path)
                self._valid_image_mask = None
                self._valid_mask_report = {}
                self._clear_skeleton_state(clear_layer=False)
            for name, visible in data.layer_visibility.items():
                self._layer_manager.set_layer_visible(name, visible)
            if data.config:
                self._param_panel.update_config(data.config)
            # ★ 恢复坐标校准
            if data.geo_calibration:
                self._geo_calibration.from_dict(data.geo_calibration)
            # ★ 恢复 ROI / Ignore 多边形（在 refresh_scene 之前推入 canvas）
            self._restore_regions_from_project(data)
            # ★ 恢复 final_graph 数据
            if data.graph_nodes or data.graph_edges:
                ge = self._graph_editor
                ge._nodes = copy.deepcopy(list(data.graph_nodes))
                ge._edges = copy.deepcopy(list(data.graph_edges))
                ge._next_node_id = data.graph_next_node_id
                ge._next_edge_id = data.graph_next_edge_id
                ge.clear_selection()
                self._render_graph_to_scene()
                self._update_graph_stats()
                print(f"[Project] 已恢复 graph: {len(ge.nodes)} 节点, {len(ge.edges)} 边")
            # ★ 恢复任务点
            if getattr(data, 'task_points_serialized', None):
                self._deserialize_task_points(data.task_points_serialized)
                self._render_task_points_to_scene()
            self._status_bar.update_resolution(data.pixel_resolution_m,
                calibrated=data.geo_calibration.get("enabled", False))
            self._status_bar.show_message(f"已加载项目: {os.path.basename(path)}")

    def _on_save_project(self):
        if not self._project_manager.project_path:
            self._on_save_project_as()
            return
        self._sync_project_data()
        if self._project_manager.save():
            self._save_regions_json()
            self._status_bar.show_message("项目已保存")

    def _on_save_project_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存项目", "project.roadnet.json",
            "RoadNet 项目 (*.roadnet.json);;JSON 文件 (*.json)"
        )
        if not path:
            return
        self._sync_project_data()
        if self._project_manager.save(path):
            self._save_regions_json()
            self._status_bar.show_message(f"项目已保存: {os.path.basename(path)}")

    def _sync_project_data(self):
        data = self._project_manager.data
        data.image_path = self._layer_manager.image_path
        data.image_width, data.image_height = self._layer_manager.image_size
        data.current_stage = self._current_stage
        data.config = self._param_panel.get_config()
        data.current_tool = self._tool_panel.current_tool
        data.zoom_level = self._canvas.zoom_level
        if self._large_image_project is not None:
            data.large_image_project_path = str(self._large_image_project.project_path)
            data.tile_index_path = self._large_image_project.tile_index_path
            data.global_mask_path = self._large_image_project.global_mask_path
            data.global_graph_path = self._large_image_project.global_graph_path
        for name, layer in self._layer_manager.layers().items():
            data.layer_visibility[name] = layer.visible
        # ★ 同步坐标校准
        if self._geo_calibration is not None:
            data.geo_calibration = self._geo_calibration.to_dict()
        # ★ 同步 ROI / Ignore 多边形数据
        data.roi_polygons = [
            [(p.x(), p.y()) for p in poly]
            for poly in self._canvas._roi_polygons
        ]
        data.ignore_polygons = [
            [(p.x(), p.y()) for p in poly]
            for poly in self._canvas._ignore_polygons
        ]
        # 兼容旧矩形
        data.ignore_rects = list(self._canvas._ignore_rects_deprecated)
        # ★ 同步 final_graph 数据（节点和边）
        if self._graph_editor is not None:
            data.graph_nodes = copy.deepcopy(list(self._graph_editor.nodes))
            data.graph_edges = copy.deepcopy(list(self._graph_editor.edges))
            data.graph_next_node_id = getattr(self._graph_editor, '_next_node_id', 0)
            data.graph_next_edge_id = getattr(self._graph_editor, '_next_edge_id', 0)
            print(f"[Project] 同步 graph 数据: {len(data.graph_nodes)} 节点, {len(data.graph_edges)} 边")
        else:
            data.graph_nodes = []
            data.graph_edges = []
            data.graph_next_node_id = 0
            data.graph_next_edge_id = 0
        # ★ 同步任务点数据
        if hasattr(self, '_task_points') and self._task_points:
            data.task_points_serialized = self._serialize_task_points()
        else:
            data.task_points_serialized = []

    def _restore_regions_from_project(self, data):
        """从项目数据恢复 ROI / Ignore 多边形到 canvas。"""
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QPolygonF
        from roadnet.region_edit import PolygonRegion

        canvas = self._canvas
        if canvas is None:
            return

        # ROI 多边形
        canvas._roi_polygons.clear()
        canvas._roi_items.clear()
        for pt_list in data.roi_polygons:
            poly = QPolygonF([QPointF(p[0], p[1]) for p in pt_list])
            canvas._roi_polygons.append(poly)
        canvas._roi_regions = [
            PolygonRegion.create("roi", [(p.x(), p.y()) for p in poly])
            for poly in canvas._roi_polygons if len(poly) >= 3
        ]

        # Ignore 多边形
        canvas._ignore_polygons.clear()
        canvas._ignore_items.clear()
        for pt_list in data.ignore_polygons:
            poly = QPolygonF([QPointF(p[0], p[1]) for p in pt_list])
            canvas._ignore_polygons.append(poly)

        # ★ 兼容旧矩形：转换为 ignore 多边形
        canvas._ignore_rects_deprecated.clear()
        for r in data.ignore_rects:
            if len(r) >= 4:
                canvas._ignore_rects_deprecated.append((r[0], r[1], r[2], r[3]))
                # 同时添加到 _ignore_polygons（在 refresh_scene 中显示）
                poly = QPolygonF([
                    QPointF(r[0], r[1]),
                    QPointF(r[0] + r[2], r[1]),
                    QPointF(r[0] + r[2], r[1] + r[3]),
                    QPointF(r[0], r[1] + r[3]),
                ])
                canvas._ignore_polygons.append(poly)
        canvas._ignore_regions = [
            PolygonRegion.create("ignore", [(p.x(), p.y()) for p in poly])
            for poly in canvas._ignore_polygons if len(poly) >= 3
        ]

    def _save_regions_json(self):
        """保存 regions.json（独立于 project.roadnet.json）"""
        try:
            from roadnet.regions import save_regions
            roi_data = [[(p.x(), p.y()) for p in poly]
                        for poly in self._canvas._roi_polygons]
            ignore_data = [[(p.x(), p.y()) for p in poly]
                           for poly in self._canvas._ignore_polygons]
            ignore_rects = list(self._canvas._ignore_rects_deprecated)

            outputs_dir = os.path.join(os.getcwd(), "outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            save_regions(outputs_dir,
                         roi_polygons=roi_data,
                         ignore_polygons=ignore_data,
                         ignore_rects=ignore_rects)
        except Exception as e:
            print(f"[PROJECT] 保存 regions.json 失败: {e}")

    # ===================================================================
    # 视图操作
    # ===================================================================

    def _on_refresh(self):
        self._canvas.refresh_scene()

    def _on_zoom_fit(self):
        self._canvas.fit_to_window()

    def _on_zoom_100(self):
        self._canvas.zoom_to_100()

    def _on_zoom_in(self):
        self._canvas.zoom_in()

    def _on_zoom_out(self):
        self._canvas.zoom_out()

    # ===================================================================
    # 工具/导航切换
    # ===================================================================

    def _on_enter_region_edit_stable_mode(self):
        if not self._layer_manager.has_image():
            QMessageBox.information(self, "区域修正稳定模式", "请先打开原始影像。")
            return
        self._region_edit_stable_mode = True
        self._clear_mask_candidate_items()
        self.set_stage("edit")
        for name in ("layer_road_mask", "layer_roi", "layer_ignore"):
            self._layer_manager.show_layer(name)
        self._sync_layer_checkboxes()
        self._canvas.refresh_scene()
        self._status_bar.show_message(
            "已进入区域修正稳定模式：ROI / Ignore / Mask 画笔 / Mask 橡皮"
        )

    def _on_region_edit_self_check(self):
        mask = self._layer_manager.get_layer_data("mask")
        mask_exists = isinstance(mask, np.ndarray)
        mask_shape = tuple(mask.shape) if mask_exists else None
        mask_dtype = str(mask.dtype) if mask_exists else None
        mask_nonzero = int(np.count_nonzero(mask)) if mask_exists else 0
        visibility = {
            name: bool(self._layer_manager.layers().get(name).visible)
            for name in ("layer_road_mask", "layer_roi", "layer_ignore")
            if self._layer_manager.layers().get(name) is not None
        }
        transform = self._canvas.transform()
        scene_scale = (float(transform.m11()), float(transform.m22()))
        report = {
            "mask_exists": mask_exists,
            "mask_shape": mask_shape,
            "mask_dtype": mask_dtype,
            "mask_nonzero": mask_nonzero,
            "roi_count": len(self._canvas.get_roi_regions()),
            "ignore_count": len(self._canvas.get_ignore_regions()),
            "current_tool": self._canvas.current_tool,
            "layer_visibility": visibility,
            "image_size": self._layer_manager.original_size,
            "display_size": self._layer_manager.image_size,
            "scene_scale": scene_scale,
            "history_undo_count": self._history.undo_count,
            "history_redo_count": self._history.redo_count,
            "last_mouse_image_pixel": self._canvas._last_region_image_pos,
        }
        print("[RegionEdit][SelfCheck]")
        for key, value in report.items():
            print(f"[RegionEdit][SelfCheck] {key} = {value}")
        details = "\n".join(f"{key}: {value}" for key, value in report.items())
        QMessageBox.information(self, "区域修正自检", details)
        return report

    # ===================================================================
    # Large-image project workflow
    # ===================================================================

    def _on_open_large_image_project(self):
        if self._large_project_thread is not None and self._large_project_thread.isRunning():
            QMessageBox.information(self, "大图任务运行中", "请等待当前大图任务完成或取消。")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "打开大图项目或原始影像", "",
            "大图项目 (large_image_project.json);;影像 (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;所有文件 (*)",
        )
        if not path:
            return
        try:
            if os.path.basename(path).lower() == "large_image_project.json":
                from roadnet.large_image_project import LargeImageProject, load_tile_index
                project = LargeImageProject.load(path)
                index = load_tile_index(project.tile_index_path) if os.path.isfile(project.tile_index_path) else {}
                self._activate_large_image_project(project, index, reload_preview=True)
            else:
                self._start_large_project_task("create", image_path=path, reload_preview=True)
        except Exception as exc:
            QMessageBox.critical(self, "打开大图项目失败", str(exc))

    def _start_large_project_task(self, action: str, *, image_path: str = "",
                                  reload_preview: bool = False):
        from roadnet.large_image_worker import LargeImageProjectWorker
        if action in {"index", "preview"} and self._large_image_project is None:
            QMessageBox.information(self, "生成 Tile Index", "请先打开或创建大图项目。")
            return
        output_root = os.path.join(os.getcwd(), "outputs", "large_image_projects")
        thread = QThread(self)
        worker = LargeImageProjectWorker(
            action=action,
            image_path=image_path,
            output_root=output_root,
            project=self._large_image_project,
            tile_size=(self._large_image_project.tile_size if self._large_image_project else 2048),
            overlap=(self._large_image_project.tile_overlap if self._large_image_project else 256),
            preview_max_side=3000,
            black_threshold=10,
            black_ratio_threshold=0.8,
        )
        worker.moveToThread(thread)
        progress = QProgressDialog("准备大图任务…", "取消", 0, 100, self)
        progress.setWindowTitle("大图项目")
        progress.setWindowModality(Qt.WindowModality.NonModal)
        progress.setAutoClose(False)
        progress.setValue(0)
        progress.canceled.connect(worker.cancel)
        thread.started.connect(worker.run)
        worker.progress.connect(
            lambda percent, current, total, message: (
                progress.setValue(int(percent)),
                progress.setLabelText(
                    f"{message}" + (f"\nTile {current}/{total}" if total else "")
                ),
            )
        )

        def finish(project, index):
            progress.close()
            self._activate_large_image_project(
                project, index,
                reload_preview=reload_preview or not self._layer_manager.has_image(),
            )

        def fail(message, error_path):
            progress.close()
            QMessageBox.critical(
                self, "大图项目任务失败",
                f"{message}\n\n错误日志：\n{error_path}",
            )

        def cancelled(message):
            progress.close()
            self._status_bar.show_message(f"{message}；已有项目数据未改变")

        worker.finished.connect(finish)
        worker.failed.connect(fail)
        worker.cancelled.connect(cancelled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_large_project_thread_finished)
        self._large_project_thread = thread
        self._large_project_worker = worker
        self._large_project_progress = progress
        progress.show()
        thread.start()

    def _on_large_project_thread_finished(self):
        self._large_project_thread = None
        self._large_project_worker = None
        self._large_project_progress = None

    def _activate_large_image_project(self, project, tile_index: dict,
                                      *, reload_preview: bool):
        """激活大图项目：只加载 preview，禁止自动加载 mask/skeleton/graph/calibration"""
        self._large_image_project = project
        already_loaded = (
            self._layer_manager.has_image()
            and os.path.normcase(os.path.abspath(self._layer_manager.image_path))
            == os.path.normcase(os.path.abspath(project.image_path))
        )
        if reload_preview:
            self._layer_manager.load_large_image_preview(
                project.image_path, project.preview_path,
                (project.image_width, project.image_height), project.preview_scale,
            )
            self._canvas.refresh_scene()
            self._canvas.fit_to_window()
        self._project_manager.data.image_path = project.image_path
        self._project_manager.data.image_width = self._layer_manager.image_size[0]
        self._project_manager.data.image_height = self._layer_manager.image_size[1]
        self._project_manager.data.original_width = project.image_width
        self._project_manager.data.original_height = project.image_height
        self._project_manager.data.large_image_mode = True
        self._project_manager.data.preview_scale = project.preview_scale
        self._project_manager.data.large_image_project_path = str(project.project_path)
        self._project_manager.data.tile_index_path = project.tile_index_path
        self._project_manager.data.global_mask_path = project.global_mask_path
        self._project_manager.data.global_graph_path = project.global_graph_path
        self._project_manager.data.output_dir = project.project_dir
        self._project_manager.mark_dirty()

        # 同步 cleaned / working / final 路径与主路种子线（不自动加载全图 mask 到 UI）
        self._cleaned_working_mask_path = getattr(
            project, "cleaned_working_mask_path", ""
        ) or None
        self._cleaned_working_mask_preview_path = getattr(
            project, "cleaned_working_mask_preview_path", ""
        ) or None
        self._working_road_mask_path = getattr(
            project, "working_road_mask_path", ""
        ) or None
        self._working_road_mask_preview_path = getattr(
            project, "working_road_mask_preview_path", ""
        ) or None
        self._final_edited_mask_path = getattr(
            project, "final_edited_mask_path", ""
        ) or None
        self._final_edited_mask_preview_path = getattr(
            project, "final_edited_mask_preview_path", ""
        ) or None
        if getattr(project, "mask_source", ""):
            self._working_mask_source = project.mask_source
        self._mask_edit_base = getattr(project, "mask_edit_base", "") or ""
        self._working_mask_dirty = bool(getattr(project, "mask_dirty", False))
        try:
            n_seeds = self._load_main_road_seed_strokes()
            if n_seeds:
                self._status_bar.show_message(f"已加载主路种子线 {n_seeds} 笔")
        except Exception:
            pass
        self._update_large_mask_status_bar()

        # ★ 大图项目：自动加载 calibration.json，保证任务点导入可复用
        cal_candidates = []
        gpath = getattr(project, "geo_calibration_path", "") or ""
        if gpath:
            cal_candidates.append(gpath)
        cal_candidates.append(os.path.join(project.project_dir, "calibration.json"))
        for cal_path in cal_candidates:
            if not cal_path or not os.path.isfile(cal_path):
                continue
            try:
                if self._geo_calibration.load(cal_path) and self._geo_calibration.is_valid:
                    self._param_panel.update_calibration_ui(self._geo_calibration)
                    self._update_calibration_corner_markers()
                    self._status_bar.update_resolution(
                        self._geo_calibration.pixel_resolution_estimated_m
                        or self._project_manager.data.pixel_resolution_m,
                        calibrated=True,
                    )
                    try:
                        self._project_manager.data.geo_calibration = (
                            self._geo_calibration.to_dict()
                        )
                    except Exception:
                        pass
                    print(f"[Calibration] 大图项目已加载校准: {cal_path}")
                    break
            except Exception as exc:
                print(f"[Calibration] 大图项目加载校准失败: {exc}")

        # ★ 大图模式：不自动加载 global_road_mask.png 到 UI
        # 用户需要通过菜单手动加载 mask preview
        # 不调用 _try_auto_load_calibration — 那也是自动动作
        # 不调用 apply_stage_preset — 会尝试加载旧图层数据

        if reload_preview or not already_loaded:
            from roadnet.graph_editor_qt import GraphEditorQt
            self._graph_editor = GraphEditorQt(
                image_size=(project.image_width, project.image_height),
                pixel_resolution_m=self._project_manager.data.pixel_resolution_m,
            )
            self._history.clear()
            self._history.inject(self._canvas, self._layer_manager, self._graph_editor, self)
            self._history.push_state("large_image_project_loaded")

        if reload_preview or not already_loaded:
            self._sync_layer_checkboxes()
            self._set_nav_active("import")

        self._status_bar.show_message(
            f"大图项目就绪：{project.image_width}x{project.image_height}，"
            f"tiles={tile_index.get('tile_count', 0)}，坐标=image_pixel"
        )
        if reload_preview or not already_loaded:
            QMessageBox.information(
                self, "大图项目就绪",
                f"项目目录：\n{project.project_dir}\n\n"
                f"原图：{project.image_width} x {project.image_height}\n"
                f"预览缩放：{project.preview_scale:.6f}\n"
                f"Tile：{tile_index.get('valid_tile_count', 0)} 有效 / "
                f"{tile_index.get('tile_count', 0)} 总数\n\n"
                "所有 ROI、Ignore、Graph 和任务点均使用 original image pixel。\n"
                "此模式仅显示预览图，不会自动加载 mask/skeleton/graph。",
            )

    def _on_generate_large_tile_index(self):
        self._start_large_project_task("index")

    def _on_large_mask_postprocess(self):
        if not self._layer_manager.is_large_image_mode:
            QMessageBox.information(self, "大图 Mask 后处理", "当前不是大图模式。")
            return
        if self._large_post_thread is not None and self._large_post_thread.isRunning():
            QMessageBox.information(self, "大图 Mask 后处理", "任务已在运行，请勿重复启动。")
            return
        mask = self._layer_manager.get_layer_data("mask")
        if not isinstance(mask, np.ndarray):
            QMessageBox.warning(self, "大图 Mask 后处理", "当前没有 global Road Mask。")
            return
        rois = [region.points for region in self._canvas.get_roi_regions() if region.enabled]
        ignores = [region.points for region in self._canvas.get_ignore_regions() if region.enabled]
        if not rois:
            answer = QMessageBox.question(
                self, "全图后处理确认",
                "当前没有 ROI，将对全部大图按 tile 后台处理。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        from datetime import datetime
        from roadnet.large_image_worker import LargeMaskPostprocessWorker
        root = Path(self._large_image_project.project_dir) if self._large_image_project else Path(os.getcwd()) / "outputs"
        output_dir = root / "masks" / f"postprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config = copy.deepcopy(self._param_panel.get_config().get("postprocess", {}))
        config.setdefault("fill_holes", False)
        config.setdefault("fill_small_holes", False)
        config.setdefault("max_hole_area", 500)
        thread = QThread(self)
        worker = LargeMaskPostprocessWorker(
            mask, str(output_dir), config=config,
            tile_size=(self._large_image_project.tile_size if self._large_image_project else 2048),
            overlap=(self._large_image_project.tile_overlap if self._large_image_project else 256),
            roi_polygons=rois, ignore_polygons=ignores,
            valid_image_mask=self._valid_image_mask,
        )
        worker.moveToThread(thread)
        progress = QProgressDialog("准备大图 Mask 后处理…", "取消", 0, 100, self)
        progress.setWindowTitle("大图 Mask 后处理")
        progress.setWindowModality(Qt.WindowModality.NonModal)
        progress.canceled.connect(worker.cancel)
        thread.started.connect(worker.run)
        worker.progress.connect(
            lambda percent, current, total, message: (
                progress.setValue(int(percent)),
                progress.setLabelText(f"{message}\nTile {current}/{total}"),
            )
        )
        worker.finished.connect(self._on_large_postprocess_finished)
        worker.failed.connect(self._on_large_postprocess_failed)
        worker.cancelled.connect(lambda message: self._status_bar.show_message(f"{message}；原 mask 未覆盖"))
        for signal in (worker.finished, worker.failed, worker.cancelled):
            signal.connect(thread.quit)
            signal.connect(progress.close)
            signal.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_large_post_thread_finished)
        self._large_post_thread = thread
        self._large_post_worker = worker
        self._large_post_progress = progress
        progress.show()
        thread.start()

    def _on_large_postprocess_finished(self, result):
        mask = cv2.imread(result.mask_path, cv2.IMREAD_GRAYSCALE)
        preview = cv2.imread(result.preview_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            self._on_large_postprocess_failed("无法重新读取处理后的 global mask", result.mask_path)
            return
        result_dir = Path(result.output_dir)
        self._history.push_mask_files(
            "large_mask_postprocess",
            str(result_dir / "global_mask_before.png"), result.mask_path,
            str(result_dir / "global_mask_preview_before.png"), result.preview_path,
        )
        self._layer_manager.set_layer_data("mask", mask, preview_data=preview)
        self._clear_skeleton_state(clear_layer=True)
        if self._large_image_project is not None:
            self._large_image_project.global_mask_path = result.mask_path
            self._large_image_project.save()
            self._project_manager.data.global_mask_path = result.mask_path
            self._project_manager.mark_dirty()
        self._canvas.refresh_scene()
        self._status_bar.show_message(
            f"大图 Mask 后处理完成：{result.report['tile_count']} tiles，"
            f"用时 {result.report['elapsed_seconds']:.1f}s"
        )

    def _on_large_postprocess_failed(self, message, error_path):
        if self._large_image_project is not None:
            self._large_image_project.last_error_log = error_path
            self._large_image_project.save()
        QMessageBox.critical(
            self, "大图 Mask 后处理失败",
            f"{message}\n\n原 global mask 未覆盖。\n错误日志：\n{error_path}",
        )

    def _on_large_post_thread_finished(self):
        self._large_post_thread = None
        self._large_post_worker = None
        self._large_post_progress = None

    def _on_large_image_self_check(self):
        if self._large_image_project is None:
            QMessageBox.information(self, "大图打开自检", "当前没有 large_image_project.json。")
            return None

        from roadnet.large_image_project import (
            large_image_self_check, current_process_memory_mb,
            estimate_image_memory_mb, determine_risk_level,
            RISK_MESSAGES,
        )
        mask = self._layer_manager.get_layer_data("mask")
        report = large_image_self_check(
            self._large_image_project,
            mask_shape=(mask.shape if isinstance(mask, np.ndarray) else None),
            calibration=self._geo_calibration,
            final_graph=self._graph_editor,
            task_points=self._task_points,
        )

        # 附加诊断
        report["large_image_mode"] = self._layer_manager.is_large_image_mode
        report["canvas_has_fullsize_QPixmap"] = False  # large_image_mode 下始终 false
        if self._layer_manager.is_large_image_mode:
            display_size = self._layer_manager.image_size
            orig_size = self._layer_manager.original_size
            report["display_pixmap_size"] = list(display_size)
            report["is_preview_only"] = (
                display_size != orig_size
                and self._layer_manager._image_rgb_full is None
            )
            # 检查是否有 full-size overlay
            full_size_layers = []
            for name, layer in self._layer_manager.layers().items():
                if layer.data is not None and isinstance(layer.data, np.ndarray):
                    d_h, d_w = layer.data.shape[:2]
                    orig_w, orig_h = orig_size
                    if (d_w, d_h) == (orig_w, orig_h):
                        full_size_layers.append(f"{name}({d_w}x{d_h})")
            report["full_size_overlay_layers"] = full_size_layers or ["none"]

        # numpy 缓存检查
        has_full_numpy = self._layer_manager._image_rgb_full is not None
        report["has_numpy_full_rgb_in_memory"] = has_full_numpy
        report["current_memory_mb"] = current_process_memory_mb()
        mem = estimate_image_memory_mb(
            self._large_image_project.image_width,
            self._large_image_project.image_height,
        )
        report["estimated_raw_rgb_mb"] = mem["raw_rgb_mb"]

        # 风险等级诊断
        risk_level = determine_risk_level(
            self._large_image_project.image_width,
            self._large_image_project.image_height,
            mem["raw_rgb_mb"],
        )
        report["risk_level"] = risk_level
        report["risk_message"] = RISK_MESSAGES.get(risk_level, f"未知风险等级: {risk_level}")

        # 最近日志
        log_candidate = Path(self._large_image_project.project_dir) / "logs" / "open_large_image.log"
        report["last_open_log_path"] = str(log_candidate) if log_candidate.is_file() else "none"

        print("[LargeImage][SelfCheck]")
        for key, value in report.items():
            print(f"[LargeImage][SelfCheck] {key} = {value}")

        lines = []
        for key, value in report.items():
            if key == "full_size_overlay_layers":
                lines.append(f"{key}: {value}")
            else:
                lines.append(f"{key}: {value}")
        QMessageBox.information(
            self, "大图打开自检",
            "\n".join(lines),
        )
        return report

    def _on_region_escape_shortcut(self):
        """Cancel only the unfinished region even when a side panel has focus."""
        tool = self._canvas.current_tool
        if tool == "roi" and self._canvas._roi_points:
            self._canvas._clear_current_roi_drawing()
        elif tool == "ignore" and self._canvas._ignore_points:
            self._canvas._clear_current_ignore_drawing()
        elif tool == "graph_draw_edge":
            self._canvas.tool_interaction.emit("cancel_manual_edge", None)
        elif tool in self._canvas.GRAPH_TOOLS:
            self._canvas.tool_interaction.emit("key_escape", {"tool": tool})

    def _on_region_backspace_shortcut(self):
        """Remove the last unfinished polygon vertex with application focus."""
        tool = self._canvas.current_tool
        if tool == "roi" and self._canvas._roi_points:
            self._canvas._undo_roi_point()
        elif tool == "ignore" and self._canvas._ignore_points:
            self._canvas._undo_ignore_point()
        elif tool == "graph_draw_edge":
            self._canvas.tool_interaction.emit("undo_manual_point", None)

    def set_tool(self, tool_id: str):
        print(f"[DEBUG][MainWindow] set_tool: {tool_id}")
        # "规划路径/导出路径"是立即执行动作，不是画布交互模式。
        # 工具栏按钮必须与右侧面板、菜单走同一个处理函数。
        if tool_id in ("plan", "export"):
            if tool_id == "plan":
                self._on_run_global_plan("astar")
            else:
                self._on_export_planned_path()
            self._tool_panel.set_current_tool("pan")
            self._canvas.set_tool("pan")
            self._status_bar.update_tool("pan")
            return
        if tool_id == "graph_locate_jump":
            self._on_graph_locate_jump()
            self._tool_panel.set_current_tool("pan")
            self._canvas.set_tool("pan")
            return
        if tool_id == "graph_local_rebuild":
            self._on_graph_local_rebuild()
            return
        if tool_id == "graph_draw_edge":
            self._configure_graph_repair_snaps()
            self._status_bar.show_message(
                "折线补路：沿道路中心线点击中间点，双击或 Enter 结束；自动吸附节点/拆边"
            )
        if self._tool_panel.current_tool != tool_id:
            self._tool_panel.set_current_tool(tool_id)
        self._canvas.set_tool(tool_id)
        self._status_bar.update_tool(tool_id)
        # ★ 进入 mask 精修时显示画笔半径
        if tool_id in {"mask_refine", "mask_brush", "mask_eraser"}:
            r = self._canvas.brush_radius
            mode_name = "橡皮" if tool_id == "mask_eraser" else "画笔"
            self._status_bar.show_message(
                f"Mask {mode_name}半径 = {r} px  ([ / ] 调整, Ctrl+滚轮调整)"
            )

    def _on_nav_step(self, step_id: str):
        self.set_stage(step_id)

    def _set_nav_active(self, step_id: str):
        self.set_stage(step_id)

    # ===================================================================
    # 路网数据完整性校验
    # ===================================================================
    def _hash_graph_geometry(self) -> str:
        """计算 final_graph 的几何哈希值。

        哈希内容：节点数、边数、所有节点的 x/y、所有边的 points_pixel 坐标。
        用于检测页面切换时 graph 是否被意外修改。
        """
        import hashlib
        if self._graph_editor is None:
            return "empty"
        ge = self._graph_editor
        h = hashlib.sha256()
        h.update(f"nodes={len(ge.nodes)}".encode())
        h.update(f"edges={len(ge.edges)}".encode())
        for n in sorted(ge.nodes, key=lambda n: n.get("id", 0)):
            h.update(f"n{n.get('id',0)}x{n.get('x',0)}y{n.get('y',0)}".encode())
        for e in sorted(ge.edges, key=lambda e: e.get("id", 0)):
            pts = e.get("points_pixel", [])
            pts_str = ";".join(f"{p[0]},{p[1]}" for p in pts)
            h.update(f"e{e.get('id',0)}p{pts_str}".encode())
        return h.hexdigest()[:16]

    def _check_graph_integrity(self, context: str = ""):
        """打印当前 graph 几何哈希，用于调试页面切换前后的数据一致性。"""
        h = self._hash_graph_geometry()
        print(f"[GRAPH_INTEGRITY] hash={h}  context={context}")
        return h

    def set_stage(self, stage: str):
        if not self._layer_manager.has_image() and stage != "import":
            QMessageBox.warning(self, "提示", "请先打开影像。")
            return

        # ★ 切换前记录 graph_hash
        hash_before = self._hash_graph_geometry()
        prev_stage = self._current_stage

        self._current_stage = stage
        if stage == "edit":
            # 导航进入区域修正即采用稳定模式，不触发任何自动处理链。
            self._region_edit_stable_mode = True
            self._clear_mask_candidate_items()

        # 导航按钮高亮
        for sid, btn in self._nav_buttons.items():
            if sid == stage:
                btn.setProperty("stageStatus", "active")
            elif sid in self._stage_completed:
                btn.setProperty("stageStatus", "done")
            else:
                btn.setProperty("stageStatus", "normal")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        # ★ 应用该阶段的图层预设（自动叠加当前模式）
        self._layer_manager.apply_stage_preset(stage)
        self._sync_layer_checkboxes()

        # 启用对应阶段工具
        self._enable_stage_tools(stage)

        # 参数面板切换
        self._param_panel.set_stage(stage)

        if stage == "segment":
            self._refresh_roi_status_panel()

        # 状态栏
        self._status_bar.update_stage(stage)
        self._status_bar.update_resolution(self._project_manager.data.pixel_resolution_m)

        # ★ graph 阶段：渲染当前路网图到场景
        if stage == "graph":
            self._render_graph_to_scene()
            self._update_graph_stats()
        elif stage == "calibrate":
            # ★ 校准阶段：显示 image + final_graph，隐藏中间图层
            # ★ 注意：只修改 UI 显示，不修改 graph_editor 数据
            self._render_graph_to_scene()
            self._update_graph_stats()
            self._update_calibration_status()
            self._update_calibration_corner_markers()
        elif stage == "export":
            self._render_graph_to_scene()
            self._render_task_points_to_scene()
            self._render_planned_path_to_scene()
            self._render_sparse_waypoints_to_scene()

        # 工具切回平移
        self.set_tool("pan")

        self._canvas.viewport().update()

        # ★ 切换后验证 graph_hash
        hash_after = self._hash_graph_geometry()
        if hash_before != hash_after:
            print(
                f"[BUG] final_graph changed during stage switch: "
                f"{prev_stage} → {stage}, "
                f"hash_before={hash_before}, hash_after={hash_after}"
            )


    def _enable_stage_tools(self, stage: str):
        stage_tools = {
            "import":    {"pan", "open"},
            "segment":   {"pan", "positive_sample", "negative_sample", "postprocess", "roi"},
            "edit":      {"pan", "roi", "ignore", "mask_brush", "mask_eraser"},
            "skeleton":  {"pan", "skeleton", "optimize"},
            "graph":     {"pan", "graph", "graph_add_node", "graph_add_edge",
                          "graph_delete_node", "graph_delete_edge",
                          "graph_move_node", "graph_merge_nodes", "graph_draw_edge",
                          "graph_local_rebuild", "graph_locate_jump",
                          "graph_save", "plan"},
            "calibrate": {"pan"},
            "export":    {"pan", "set_start", "set_end", "add_task", "plan", "export"},
        }
        allowed = stage_tools.get(stage, {"pan"})
        self._tool_panel.set_visible_tools(allowed)
        self._tool_panel.set_enabled_tools(allowed)

    # ===================================================================
    # 工具执行
    # ===================================================================

    def _ensure_valid_image_mask(self):
        if self._layer_manager.is_large_image_mode:
            expected_shape = (
                int(self._layer_manager.original_size[1]),
                int(self._layer_manager.original_size[0]),
            )
            project = self._large_image_project
            saved = getattr(project, "valid_image_mask_path", "") if project else ""
            if (self._valid_image_mask is None
                    or self._valid_image_mask.shape[:2] != expected_shape):
                if saved and os.path.isfile(saved):
                    self._valid_image_mask = cv2.imread(saved, cv2.IMREAD_GRAYSCALE)
                    if (self._valid_image_mask is not None
                            and self._valid_image_mask.shape[:2] == expected_shape):
                        return self._valid_image_mask
                # Do not synchronously decode the original RGB image here.
                # The tile workers create the border-connected valid mask.
                return None
            return self._valid_image_mask

        image = self._layer_manager.full_image_rgb
        if image is None:
            return None
        expected_shape = image.shape[:2]
        if (self._valid_image_mask is None
                or self._valid_image_mask.shape[:2] != expected_shape):
            from roadnet.valid_image import analyze_valid_image_mask
            seg_cfg = self._param_panel.get_config().get("segment", {})
            black_threshold = int(seg_cfg.get("black_threshold", 10))
            min_black_area = int(seg_cfg.get("min_black_component_area", 4096))
            self._valid_image_mask, self._valid_mask_report = analyze_valid_image_mask(
                image, black_threshold, min_black_area
            )
            self._layer_manager.set_layer_data("valid_image_mask", self._valid_image_mask)
            self._layer_manager.hide_layer("valid_image_mask")
        return self._valid_image_mask

    def _apply_valid_area(self, mask):
        valid = self._ensure_valid_image_mask()
        if valid is None:
            return np.asarray(mask, dtype=np.uint8).copy()
        from roadnet.valid_image import apply_valid_image_mask
        return apply_valid_image_mask(mask, valid)

    # ────────────────────────────────────────────────────────────────────
    # 统一 mask 获取函数（同时支持 ndarray 和 dict）
    # ────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────
    # 大图骨架输入 mask（唯一入口）
    # ────────────────────────────────────────────────────────────────────

    _PREVIEW_MASK_NAME_HINTS = (
        "preview_mask.png",
        "preview_seg_overlay.png",
        "global_road_mask_preview.png",
        "working_road_mask_preview.png",
        "cleaned_working_mask_preview.png",
        "final_edited_mask_preview.png",
        "preview_seg",
        "_preview.png",
    )

    def _is_preview_mask_path(self, path: str) -> bool:
        if not path:
            return False
        name = os.path.basename(path).lower()
        return any(h in name for h in self._PREVIEW_MASK_NAME_HINTS)

    def _mask_file_fingerprint(self, path: str) -> dict:
        """文件修改时间 + 内容 checksum（采样），用于 skeleton cache 失效。"""
        import hashlib
        info = {
            "path": path,
            "mtime": None,
            "size": None,
            "checksum": None,
        }
        if not path or not os.path.isfile(path):
            return info
        try:
            st = os.stat(path)
            info["mtime"] = float(st.st_mtime)
            info["size"] = int(st.st_size)
            h = hashlib.md5()
            h.update(f"{st.st_size}:{st.st_mtime}".encode("utf-8"))
            # 采样文件头尾，避免全图 MD5 过慢
            with open(path, "rb") as f:
                head = f.read(65536)
                h.update(head)
                if st.st_size > 131072:
                    f.seek(max(0, st.st_size - 65536))
                    h.update(f.read(65536))
            info["checksum"] = h.hexdigest()
        except Exception as exc:
            print(f"[SkeletonInput] fingerprint 失败: {exc}")
        return info

    def get_skeleton_input_mask_large(self):
        """大图正式骨架的唯一 mask 输入选择器（只读 full-size 文件，不读图层/preview）。

        优先级（cleaned 只是中间结果，最终手修优先）：
          1. final_edited_mask_path  （含 mask_source=ribbon_hole_gap_filled / final_edited_mask）
          2. working_road_mask_path
          3. cleaned_working_mask_path
          4. refined_main_road_mask_path
          5. global_road_mask_path / global_mask_path

        Returns:
            (mask_array, meta_dict) 或 (None, meta_dict)
        """
        from datetime import datetime

        meta = {
            "selected_mask_path": None,
            "mask_source": None,
            "mask_edit_base": getattr(self, "_mask_edit_base", "") or "",
            "mask_shape": None,
            "mask_dtype": None,
            "nonzero_pixels": 0,
            "nonzero_ratio": 0.0,
            "large_image_mode": True,
            "is_preview_mask": False,
            "is_full_resolution": False,
            "file_modified_time": None,
            "checksum": None,
            "error": None,
            "formal_ready": True,
            "preview_only": False,
            "source": "",
            "selected_path": None,
        }
        if not self._layer_manager.is_large_image_mode:
            meta["error"] = "get_skeleton_input_mask_large 仅用于大图模式"
            meta["large_image_mode"] = False
            return None, meta

        project = self._large_image_project
        if project is not None and getattr(project, "mask_edit_base", ""):
            meta["mask_edit_base"] = project.mask_edit_base
        expected_w = expected_h = 0
        if project is not None:
            expected_w = int(project.image_width or 0)
            expected_h = int(project.image_height or 0)

        def _path_from(*sources):
            for src in sources:
                if isinstance(src, str) and src.strip() and os.path.isfile(src):
                    return src
            return ""

        candidates = [
            (
                "final_edited_mask",
                _path_from(
                    getattr(self, "_final_edited_mask_path", None),
                    getattr(project, "final_edited_mask_path", "") if project else "",
                ),
            ),
            (
                "working_road_mask",
                _path_from(
                    getattr(self, "_working_road_mask_path", None),
                    getattr(project, "working_road_mask_path", "") if project else "",
                ),
            ),
            (
                "cleaned_working_mask",
                _path_from(
                    getattr(self, "_cleaned_working_mask_path", None),
                    getattr(project, "cleaned_working_mask_path", "") if project else "",
                ),
            ),
            (
                "refined_main_road_mask",
                _path_from(
                    getattr(project, "refined_main_road_mask_path", "") if project else "",
                ),
            ),
            (
                "global_road_mask",
                _path_from(
                    getattr(project, "global_road_mask_path", "") if project else "",
                    getattr(project, "global_mask_path", "") if project else "",
                ),
            ),
        ]

        selected_label = None
        selected_path = None
        for label, path in candidates:
            if not path:
                continue
            if self._is_preview_mask_path(path):
                continue
            selected_label, selected_path = label, path
            break

        if not selected_path:
            for label, path in candidates:
                if path and self._is_preview_mask_path(path):
                    meta["selected_mask_path"] = path
                    meta["selected_path"] = path
                    meta["mask_source"] = label
                    meta["is_preview_mask"] = True
                    meta["preview_only"] = True
                    meta["error"] = (
                        "当前选择的是 preview mask，不能用于正式骨架生成。"
                        "请使用 full-size final_edited_mask.png / working_road_mask.png "
                        "或 cleaned_working_mask.png。"
                    )
                    return None, meta
            meta["error"] = (
                "未找到可用于骨架的 full-size mask 文件。\n"
                "请先清理并手动修正后「保存当前 Mask」生成 final_edited_mask.png，"
                "或至少保存 working_road_mask.png。"
            )
            return None, meta

        try:
            arr = cv2.imread(selected_path, cv2.IMREAD_GRAYSCALE)
        except Exception as exc:
            meta["error"] = f"读取 mask 失败: {selected_path}\n{exc}"
            return None, meta
        if arr is None or arr.size == 0:
            meta["error"] = f"无法读取 mask 文件或文件为空: {selected_path}"
            return None, meta
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        arr = (arr > 0).astype(np.uint8) * 255

        fp = self._mask_file_fingerprint(selected_path)
        h, w = arr.shape[:2]
        nz = int(np.count_nonzero(arr))
        total = float(h * w) if h and w else 1.0
        is_full = bool(expected_w and expected_h and (w, h) == (expected_w, expected_h))
        is_preview = self._is_preview_mask_path(selected_path)
        if not is_preview and expected_w and expected_h:
            pw = getattr(self._layer_manager, "_preview_width", 0) or 0
            ph = getattr(self._layer_manager, "_preview_height", 0) or 0
            if (w, h) == (pw, ph) and (w, h) != (expected_w, expected_h):
                is_preview = True

        meta.update({
            "selected_mask_path": selected_path,
            "selected_path": selected_path,
            "mask_source": selected_label,
            "source": selected_label,
            "mask_shape": (h, w),
            "mask_dtype": str(arr.dtype),
            "nonzero_pixels": nz,
            "nonzero_ratio": round(nz / total, 6),
            "is_preview_mask": is_preview,
            "is_full_resolution": is_full,
            "file_modified_time": (
                datetime.fromtimestamp(fp["mtime"]).isoformat(timespec="seconds")
                if fp.get("mtime") else None
            ),
            "checksum": fp.get("checksum"),
            "preview_only": bool(is_preview),
            "formal_ready": not is_preview,
            "data_type": "ndarray",
        })

        if is_preview:
            meta["error"] = (
                "当前选择的是 preview mask，不能用于正式骨架生成。"
                "请使用 full-size final_edited_mask.png 或 working_road_mask.png。"
            )
            return None, meta

        if expected_w and expected_h and not is_full:
            meta["error"] = (
                f"当前 mask 尺寸为 {w}x{h}，与原图 {expected_w}x{expected_h} 不一致，"
                "不能用于正式骨架生成。"
            )
            return None, meta

        if nz == 0:
            meta["error"] = f"选中的 mask 全黑（无道路像素）: {selected_path}"
            return None, meta

        return arr, meta

    def _format_skeleton_input_diagnostics(self, meta: dict) -> str:
        shape = meta.get("mask_shape")
        shape_s = f"{shape[1]} x {shape[0]}" if shape and len(shape) == 2 else str(shape)
        return (
            "当前骨架输入 mask：\n"
            f"{meta.get('selected_mask_path') or meta.get('error') or '(无)'}\n"
            f"mask_source = {meta.get('mask_source')}\n"
            f"mask_edit_base = {meta.get('mask_edit_base')}\n"
            f"shape = {shape_s}\n"
            f"dtype = {meta.get('mask_dtype')}\n"
            f"nonzero_pixels = {meta.get('nonzero_pixels')}\n"
            f"nonzero_ratio = {meta.get('nonzero_ratio')}\n"
            f"large_image_mode = {meta.get('large_image_mode')}\n"
            f"is_preview_mask = {meta.get('is_preview_mask')}\n"
            f"is_full_resolution = {meta.get('is_full_resolution')}\n"
            f"file_modified_time = {meta.get('file_modified_time')}\n"
            f"input_mask_hash = {meta.get('checksum')}"
        )

    def _update_large_mask_status_bar(self, extra: str = ""):
        """底部状态栏显示当前大图 mask 生命周期状态。"""
        if not self._layer_manager.is_large_image_mode:
            return
        src = getattr(self, "_working_mask_source", "") or "unknown"
        base = getattr(self, "_mask_edit_base", "") or "-"
        dirty = bool(getattr(self, "_working_mask_dirty", False))
        saved = "未保存" if dirty else "已保存"
        msg = f"当前 Mask 来源：{src}｜基于：{base}｜{saved}"
        if extra:
            msg = f"{msg}｜{extra}"
        self._status_bar.show_message(msg)

    def _invalidate_large_skeleton_cache_if_needed(
        self, output_dir: str, input_meta: dict
    ) -> dict:
        """若输入 mask 路径/checksum 变化，删除旧 skeleton 产物。"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / "skeleton_report.json"
        cache_info = {
            "cache_used": False,
            "cache_invalidated": False,
            "previous_input_mask_path": None,
            "previous_checksum": None,
        }
        prev = {}
        if report_path.is_file():
            try:
                with report_path.open("r", encoding="utf-8") as f:
                    prev = json.load(f)
            except Exception:
                prev = {}
        cache_info["previous_input_mask_path"] = prev.get("input_mask_path")
        cache_info["previous_checksum"] = prev.get("input_mask_hash") or prev.get("checksum")

        same = (
            prev.get("input_mask_path") == input_meta.get("selected_mask_path")
            and prev.get("input_mask_hash")
            and prev.get("input_mask_hash") == input_meta.get("checksum")
            and (
                not prev.get("input_mask_mtime")
                or prev.get("input_mask_mtime") == input_meta.get("file_modified_time")
            )
        )
        if same:
            cache_info["cache_used"] = True
            return cache_info

        # 输入变化 → 清除旧缓存
        for name in (
            "raw_skeleton.png",
            "optimized_skeleton.png",
            "optimized_skeleton_preview.png",
            "skeleton_preview.png",
            "skeleton_graph.json",
            "skeleton_stats.json",
            "raw_skeleton_preview.png",
            "pruned_skeleton_preview.png",
        ):
            p = out / name
            if p.is_file():
                try:
                    p.unlink()
                    cache_info["cache_invalidated"] = True
                except Exception as exc:
                    print(f"[SkeletonCache] 删除 {p} 失败: {exc}")
        return cache_info

    def _write_large_skeleton_generation_report(
        self,
        output_dir: str,
        input_meta: dict,
        cache_info: dict,
        raw_skel: np.ndarray,
        opt_skel: np.ndarray,
        elapsed: float,
        extra: Optional[dict] = None,
    ) -> str:
        from datetime import datetime
        from roadnet.optimized_skeleton import _find_endpoints, _find_junctions

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        raw_bin = raw_skel > 0
        opt_bin = opt_skel > 0
        report = {
            "selected_mask_path": input_meta.get("selected_mask_path"),
            "mask_source": input_meta.get("mask_source"),
            "mask_edit_base": input_meta.get("mask_edit_base"),
            "mask_shape": list(input_meta.get("mask_shape") or []),
            "mask_nonzero_ratio": input_meta.get("nonzero_ratio"),
            "used_preview_mask": bool(input_meta.get("is_preview_mask")),
            "cache_used": bool(cache_info.get("cache_used")),
            "cache_invalidated": bool(cache_info.get("cache_invalidated")),
            "raw_skeleton_pixel_count": int(np.count_nonzero(raw_bin)),
            "optimized_skeleton_pixel_count": int(np.count_nonzero(opt_bin)),
            "endpoint_count": len(_find_endpoints(opt_bin)),
            "junction_count": len(_find_junctions(opt_bin)),
            "elapsed_seconds": round(float(elapsed), 3),
            "input_mask_path": input_meta.get("selected_mask_path"),
            "input_mask_mtime": input_meta.get("file_modified_time"),
            "input_mask_hash": input_meta.get("checksum"),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        if extra:
            report.update(extra)
        path = out / "large_skeleton_generation_report.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        # 同步轻量 skeleton_report.json 供 cache 比对
        light = {
            "input_mask_path": report["input_mask_path"],
            "input_mask_mtime": report["input_mask_mtime"],
            "input_mask_hash": report["input_mask_hash"],
            "generated_at": report["generated_at"],
            "mask_source": report["mask_source"],
        }
        with (out / "skeleton_report.json").open("w", encoding="utf-8") as f:
            json.dump(light, f, ensure_ascii=False, indent=2)
        return str(path)

    def get_current_mask_array(
        self,
        prefer_processed: bool = True,
        require_full_resolution: bool = True,
        for_skeleton: bool = True,
    ):
        """统一获取当前正式 Road Mask 的 numpy 数组。

        支持两种 mask 数据格式：
        - np.ndarray: 直接返回
        - dict: 按优先级顺序解析路径 / array 字段

        Args:
            prefer_processed: 优先使用 processed_* 路径
            require_full_resolution: 大图模式下是否要求 full-size mask
            for_skeleton: 是否用于骨架生成（会阻止 preview_only mask）

        Returns:
            (mask_array, metadata_dict)
            mask_array: np.ndarray (H, W) uint8, 或 None
            metadata_dict: {"source": str, "preview_only": bool, "large_image_mode": bool, ...}
        """
        import traceback
        from datetime import datetime

        meta = {
            "source": "",
            "source_resolved": "",
            "mask_shape": None,
            "mask_dtype": None,
            "preview_only": False,
            "large_image_mode": bool(self._layer_manager.is_large_image_mode),
            "large_image_project": self._large_image_project is not None,
            "error": None,
            "selected_path": None,
            "data_type": None,
            "data_keys": None,
            "nonzero_pixels": 0,
        }
        # 附带当前正式 mask 的注册元数据（mask_type / formal_ready / 坐标系）。
        if isinstance(getattr(self, "_formal_mask_meta", None), dict):
            for _k in ("mask_type", "formal_ready", "coordinate_system"):
                if _k in self._formal_mask_meta:
                    meta[_k] = self._formal_mask_meta[_k]

        # ★ 大图骨架：走统一文件优先级（final → working → cleaned → refined → global）
        if self._layer_manager.is_large_image_mode and for_skeleton:
            arr, sk_meta = self.get_skeleton_input_mask_large()
            if arr is not None:
                meta.update({
                    "source": sk_meta.get("mask_source") or "",
                    "mask_source": sk_meta.get("mask_source") or "",
                    "mask_edit_base": sk_meta.get("mask_edit_base") or "",
                    "selected_path": sk_meta.get("selected_mask_path"),
                    "source_resolved": sk_meta.get("mask_source") or "",
                    "data_type": "ndarray",
                    "mask_shape": sk_meta.get("mask_shape"),
                    "mask_dtype": sk_meta.get("mask_dtype"),
                    "nonzero_pixels": sk_meta.get("nonzero_pixels", 0),
                    "preview_only": bool(sk_meta.get("is_preview_mask")),
                    "formal_ready": bool(sk_meta.get("formal_ready", True)),
                })
                return arr, meta
            if sk_meta.get("error"):
                meta["error"] = sk_meta["error"]
                meta["selected_path"] = sk_meta.get("selected_mask_path")
                # 继续回退到图层 / 其它路径（兼容非骨架调用方）

        raw = self._layer_manager.get_layer_data("mask")
        if raw is None:
            # ★ 大图：图层空时按 final → working → cleaned → global 回退
            if self._layer_manager.is_large_image_mode:
                project = self._large_image_project
                fallbacks = []
                final_path = getattr(self, "_final_edited_mask_path", None) or ""
                if project is not None:
                    final_path = (
                        final_path
                        or getattr(project, "final_edited_mask_path", "")
                        or ""
                    )
                if final_path:
                    fallbacks.append(("final_edited_mask", final_path))
                working_path = getattr(self, "_working_road_mask_path", None) or ""
                if project is not None:
                    working_path = (
                        working_path
                        or getattr(project, "working_road_mask_path", "")
                        or ""
                    )
                if working_path:
                    fallbacks.append(("working_road_mask", working_path))
                cleaned_path = getattr(self, "_cleaned_working_mask_path", None) or ""
                if project is not None:
                    cleaned_path = (
                        cleaned_path
                        or getattr(project, "cleaned_working_mask_path", "")
                        or ""
                    )
                if cleaned_path:
                    fallbacks.append(("cleaned_working_mask", cleaned_path))
                if project is not None:
                    for key, label in (
                        ("refined_main_road_mask_path", "refined_main_road_mask"),
                        ("global_road_mask_path", "global_road_mask"),
                        ("global_mask_path", "global_road_mask"),
                    ):
                        p = getattr(project, key, "") or ""
                        if p:
                            fallbacks.append((label, p))
                for label, path in fallbacks:
                    if not path or not os.path.isfile(path):
                        continue
                    try:
                        working_arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        if working_arr is not None and working_arr.size > 0:
                            meta["source"] = label
                            meta["selected_path"] = path
                            meta["source_resolved"] = label
                            meta["data_type"] = "ndarray"
                            mask_array = working_arr
                            if mask_array.ndim == 3:
                                mask_array = mask_array[:, :, 0]
                            if mask_array.dtype != np.uint8:
                                mask_array = (mask_array > 0).astype(np.uint8) * 255
                            meta["mask_shape"] = tuple(mask_array.shape)
                            meta["mask_dtype"] = str(mask_array.dtype)
                            meta["nonzero_pixels"] = int(np.count_nonzero(mask_array))
                            meta["mask_source"] = getattr(
                                self, "_working_mask_source", "manual_edited"
                            )
                            return mask_array, meta
                    except Exception as e:
                        print(f"[get_current_mask_array] 读取 {path} 失败: {e}")
            meta["error"] = "mask 图层数据为 None"
            return None, meta

        mask_array = None
        meta["data_type"] = type(raw).__name__

        # ── 情况 1: 已经是 numpy array（含未保存的手动修正）──
        if isinstance(raw, np.ndarray):
            mask_array = raw
            meta["source"] = "layer_data (ndarray)"
            if getattr(self, "_working_mask_source", None):
                meta["mask_source"] = self._working_mask_source

        # ── 情况 2: 是 dict ─────────────────────────────────────
        elif isinstance(raw, dict):
            meta["data_keys"] = list(raw.keys())

            # 大图：优先 final → working → cleaned → refined → global
            key_order = []
            if prefer_processed:
                key_order = [
                    "final_edited_mask_path",
                    "working_road_mask_path", "edited_global_road_mask_path",
                    "cleaned_working_mask_path",
                    "refined_main_road_mask_path",
                    "processed_global_mask_path", "processed_mask_path",
                    "global_road_mask_path", "global_mask_path",
                    "road_mask_path", "mask_path",
                ]
            else:
                key_order = [
                    "final_edited_mask_path",
                    "working_road_mask_path", "edited_global_road_mask_path",
                    "cleaned_working_mask_path",
                    "refined_main_road_mask_path",
                    "global_road_mask_path", "global_mask_path",
                    "processed_global_mask_path", "processed_mask_path",
                    "road_mask_path", "mask_path",
                ]

            for key in key_order:
                val = raw.get(key, "")
                if isinstance(val, str) and os.path.isfile(val):
                    meta["selected_path"] = val
                    meta["source_resolved"] = key
                    try:
                        mask_array = cv2.imread(val, cv2.IMREAD_GRAYSCALE)
                        if mask_array is not None and mask_array.size > 0:
                            meta["source"] = f"dict['{key}'] → file"
                            break
                    except Exception as e:
                        print(f"[get_current_mask_array] 读取 {val} 失败: {e}")

            # 如果文件路径都没找到，尝试 array/mask/data 字段
            if mask_array is None:
                for key in ("array", "mask", "data"):
                    val = raw.get(key)
                    if isinstance(val, np.ndarray) and val.size > 0:
                        mask_array = val
                        meta["source"] = f"dict['{key}'] (ndarray)"
                        break

            # 检查 preview_only 标志
            if raw.get("preview_only", False):
                meta["preview_only"] = True
            # 检查路径名
            if meta.get("selected_path"):
                pn = os.path.basename(meta["selected_path"]).lower()
                if any(kw in pn for kw in (
                    "preview_mask.png", "preview_seg_overlay.png",
                    "global_road_mask_preview.png", "preview_seg",
                )):
                    meta["preview_only"] = True

        # ── 情况 3: 其他类型（list, str 等）───────────────────
        elif isinstance(raw, str) and os.path.isfile(raw):
            meta["selected_path"] = raw
            try:
                mask_array = cv2.imread(raw, cv2.IMREAD_GRAYSCALE)
                meta["source"] = "layer_data (str path)"
            except Exception:
                pass

        # ── 后处理: 统一 dtype / ndim ─────────────────────────
        if mask_array is not None:
            if mask_array.ndim != 2:
                if mask_array.ndim == 3:
                    mask_array = mask_array[:, :, 0]
                else:
                    meta["error"] = f"mask_array.ndim={mask_array.ndim}，无法转换为 2D"
                    return None, meta
            if mask_array.dtype != np.uint8:
                mask_array = (mask_array > 0).astype(np.uint8) * 255
            meta["mask_shape"] = tuple(mask_array.shape)
            meta["mask_dtype"] = str(mask_array.dtype)
            meta["nonzero_pixels"] = int(np.count_nonzero(mask_array))
        else:
            # 遍历了所有可能，仍未找到有效 mask
            meta["error"] = (
                "未找到有效 Road Mask 数组。"
                f"data_type={meta['data_type']}"
                + (f", keys={meta['data_keys']}" if meta["data_keys"] else "")
            )

        return mask_array, meta

    def _log_skeleton_error(self, mask_meta: dict, extra_info: str = ""):
        """将骨架生成失败诊断写入 skeleton_error.log。"""
        import json
        import datetime
        try:
            log_dir = None
            if self._large_image_project is not None:
                log_dir = os.path.join(self._large_image_project.project_dir, "skeleton")
            if log_dir is None and self._layer_manager.has_image():
                log_dir = os.path.join(os.path.dirname(
                    getattr(self._layer_manager, 'image_path', '.')), "skeleton")
            if log_dir is None:
                log_dir = os.path.join(os.getcwd(), "outputs", "skeleton")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "skeleton_error.log")
            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "mask_meta": {k: str(v) if not isinstance(v, (int, bool, type(None)))
                              else v for k, v in mask_meta.items()},
                "extra": extra_info,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _has_preview_segmentation_only(self) -> bool:
        """判断当前是否只有快速预览分割结果（layer_preview_segmentation 有数据）。"""
        try:
            data = self._layer_manager.get_layer_data("layer_preview_segmentation")
        except Exception:
            return False
        return data is not None

    def _skeleton_input_check(self, mask_array, mask_meta: dict) -> str | None:
        """检查 mask 是否可用于骨架生成。返回 None 表示通过，否则返回错误信息。"""
        if mask_array is None:
            err = mask_meta.get("error", "mask_array 为 None")
            self._log_skeleton_error(mask_meta, f"mask_array is None: {err}")
            # 大图模式下若只存在快速预览结果，给出明确指引。
            if mask_meta.get("large_image_mode") and self._has_preview_segmentation_only():
                return (
                    "当前只有快速预览结果，不能生成正式骨架。\n"
                    "请先运行大图 ROI 正式提取，生成 global_road_mask.png。"
                )
            return (
                f"骨架生成失败：当前 Road Mask 图层不是正式二值 mask，而是 metadata dict。\n"
                f"请检查是否已生成 global_road_mask.png 或 processed_global_mask.png。\n\n"
                f"诊断信息：\n"
                f"  data_type = {mask_meta.get('data_type', '?')}\n"
                f"  data_keys = {mask_meta.get('data_keys', '?')}\n"
                f"  large_image_mode = {mask_meta.get('large_image_mode')}"
            )

        if not isinstance(mask_array, np.ndarray):
            err = f"mask_array type={type(mask_array).__name__}，非 np.ndarray"
            self._log_skeleton_error(mask_meta, err)
            return (
                f"骨架生成失败：当前 Road Mask 类型不是 numpy 数组。\n"
                f"  type = {type(mask_array).__name__}\n"
                f"请确保已生成正式 Road Mask。"
            )

        if mask_array.ndim != 2:
            err = f"mask_array.ndim={mask_array.ndim}，应为 2"
            self._log_skeleton_error(mask_meta, err)
            return (
                f"骨架生成失败：Road Mask 不是单通道二维数组。\n"
                f"  ndim = {mask_array.ndim}\n"
                f"  shape = {mask_array.shape}"
            )

        if mask_array.dtype != np.uint8:
            err = f"mask_array.dtype={mask_array.dtype}，应为 uint8"
            self._log_skeleton_error(mask_meta, err)
            return (
                f"骨架生成失败：Road Mask 数据类型不正确。\n"
                f"  dtype = {mask_array.dtype}\n"
                f"  expected = uint8"
            )

        if mask_meta.get("nonzero_pixels", 1) == 0:
            err = "mask 全黑（无道路像素）"
            self._log_skeleton_error(mask_meta, err)
            return "骨架生成失败：Road Mask 全为黑色，没有道路像素。请检查分割结果。"

        # 大图模式：检查是否 preview_only
        if mask_meta.get("large_image_mode") and mask_meta.get("preview_only"):
            err = "mask 是 preview_only，不能用于骨架生成"
            self._log_skeleton_error(mask_meta, err)
            return (
                "当前只有快速预览分割结果，不能生成正式骨架。\n"
                "请先运行正式 tile 分割，生成 global_road_mask.png。\n\n"
                "高级选项：可将预览结果升采样为初始 mask（实验性功能）。"
            )

        # 大图模式：检查尺寸
        if mask_meta.get("large_image_mode") and self._large_image_project is not None:
            expected_w = self._large_image_project.image_width
            expected_h = self._large_image_project.image_height
            mask_h, mask_w = mask_array.shape[:2]
            if (mask_w, mask_h) != (expected_w, expected_h):
                preview_w = self._layer_manager._preview_width if hasattr(
                    self._layer_manager, '_preview_width') else 0
                preview_h = self._layer_manager._preview_height if hasattr(
                    self._layer_manager, '_preview_height') else 0
                if (mask_w, mask_h) == (preview_w, preview_h):
                    err = (f"mask 是 preview 尺寸 ({mask_w}x{mask_h})，"
                           f"不能直接生成正式骨架（期望 {expected_w}x{expected_h})")
                    self._log_skeleton_error(mask_meta, err)
                    return (
                        f"当前 mask 是 preview 尺寸 ({mask_w}x{mask_h})，"
                        f"不能直接生成正式骨架。\n"
                        f"正式骨架需要 full-size mask ({expected_w}x{expected_h})。"
                    )

        return None  # 通过检查

    def _clear_skeleton_state(self, clear_layer: bool = False):
        self.raw_skeleton = None
        self.optimized_skeleton = None
        self.current_skeleton = None
        self.skeleton_state = "none"
        if self._canvas is not None:
            self._canvas.skeleton_result = None
        if clear_layer and self._layer_manager is not None:
            self._layer_manager.clear_layer("skeleton")

    def _commit_raw_skeleton(self, raw_skeleton):
        from roadnet.skeleton_artifacts import binary_skeleton
        raw = binary_skeleton(raw_skeleton)
        self.raw_skeleton = raw.copy()
        self.optimized_skeleton = None
        self.current_skeleton = raw.copy()
        self.skeleton_state = "raw"

        is_large = self._layer_manager.is_large_image_mode
        preview_data = None

        # 确定输出目录
        if is_large and self._large_image_project is not None:
            output_dir = os.path.join(self._large_image_project.project_dir, "skeleton")
        else:
            output_dir = os.path.join(os.getcwd(), "outputs", "skeleton")
        os.makedirs(output_dir, exist_ok=True)

        # 保存 full-size raw_skeleton
        raw_skeleton_path = os.path.join(output_dir, "raw_skeleton.png")
        cv2.imwrite(raw_skeleton_path, self.raw_skeleton)

        # 大图模式：额外生成 preview 版本用于 UI 显示
        if is_large and self._layer_manager._preview_width > 0:
            ph, pw = self._layer_manager._preview_height, self._layer_manager._preview_width
            preview = cv2.resize(
                self.raw_skeleton, (pw, ph), interpolation=cv2.INTER_NEAREST,
            )
            preview_path = os.path.join(output_dir, "skeleton_preview.png")
            cv2.imwrite(preview_path, preview)
            preview_data = preview

        layer_data = {
            "raw_skeleton": self.raw_skeleton,
            "current_skeleton": self.current_skeleton,
            "skeleton_state": "raw",
            "raw_skeleton_path": raw_skeleton_path,
            "output_dir": output_dir,
        }
        self._layer_manager.set_layer_data("skeleton", layer_data, preview_data=preview_data)
        self._canvas.skeleton_result = layer_data

    def _commit_optimized_skeleton(self, raw_skeleton, optimized_skeleton, result=None,
                                   preview_data=None):
        from roadnet.skeleton_artifacts import binary_skeleton
        raw = binary_skeleton(raw_skeleton)
        optimized = binary_skeleton(optimized_skeleton)
        self.raw_skeleton = raw.copy()
        self.optimized_skeleton = optimized.copy()
        self.current_skeleton = optimized.copy()
        self.skeleton_state = "optimized"

        is_large = self._layer_manager.is_large_image_mode

        # 确定输出目录
        if is_large and self._large_image_project is not None:
            output_dir = os.path.join(self._large_image_project.project_dir, "skeleton")
        else:
            output_dir = os.path.join(os.getcwd(), "outputs", "skeleton")
        os.makedirs(output_dir, exist_ok=True)

        # 保存 full-size skeleton 文件
        cv2.imwrite(os.path.join(output_dir, "raw_skeleton.png"), self.raw_skeleton)
        cv2.imwrite(os.path.join(output_dir, "optimized_skeleton.png"), self.optimized_skeleton)

        # 大图模式：生成 preview 版本
        skeleton_preview = None
        if is_large and self._layer_manager._preview_width > 0 and preview_data is None:
            ph, pw = self._layer_manager._preview_height, self._layer_manager._preview_width
            skeleton_preview = cv2.resize(
                self.current_skeleton, (pw, ph), interpolation=cv2.INTER_NEAREST,
            )
            preview_path = os.path.join(output_dir, "skeleton_preview.png")
            cv2.imwrite(preview_path, skeleton_preview)
            preview_data = skeleton_preview

        layer_data = dict(result) if isinstance(result, dict) else {}
        layer_data.update({
            "raw_skeleton": self.raw_skeleton,
            "optimized_skeleton": self.optimized_skeleton,
            "current_skeleton": self.current_skeleton,
            "skeleton_state": "optimized",
            "output_dir": output_dir,
        })
        self._layer_manager.set_layer_data(
            "skeleton", layer_data, preview_data=preview_data,
        )
        self._canvas.skeleton_result = layer_data

    def _generate_raw_skeleton(self, processed_mask=None):
        """从 mask 生成原始骨架。会使用 get_current_mask_array 来自动处理 dict/ndarray。"""
        mask = processed_mask
        if mask is None:
            mask, mask_meta = self.get_current_mask_array(prefer_processed=True)
            if mask is None:
                raise ValueError(
                    "processed_mask 不存在，无法生成 raw_skeleton。\n"
                    f"mask_meta: {mask_meta}"
                )
            err = self._skeleton_input_check(mask, mask_meta)
            if err is not None:
                raise ValueError(err)
        mask = self._apply_valid_area(mask)
        from roadnet.optimized_skeleton import skeletonize_medial_axis, skeletonize_thin
        skel_cfg = self._param_panel.get_config().get("skeleton", {})
        if skel_cfg.get("method", "skeletonize") == "medial_axis":
            return skeletonize_medial_axis(mask)
        return skeletonize_thin(mask)

    def _sync_skeleton_state_from_layer(self):
        data = self._layer_manager.get_layer_data("skeleton")
        if data is None:
            self._clear_skeleton_state(clear_layer=False)
            return
        if isinstance(data, dict):
            state = data.get("skeleton_state", "raw")
            raw = data.get("raw_skeleton")
            optimized = data.get("optimized_skeleton")
            current = data.get("current_skeleton")
            self.raw_skeleton = None if raw is None else self._get_skeleton_array(raw)
            self.optimized_skeleton = (
                None if optimized is None else self._get_skeleton_array(optimized)
            )
            fallback = optimized if state == "optimized" else raw
            self.current_skeleton = self._get_skeleton_array(
                current if current is not None else fallback
            ) if (current is not None or fallback is not None) else None
            self.skeleton_state = state if state in ("raw", "optimized") else "raw"
        else:
            self.raw_skeleton = self._get_skeleton_array(data)
            self.optimized_skeleton = None
            self.current_skeleton = self.raw_skeleton.copy()
            self.skeleton_state = "raw"

    def _save_manual_skeleton_artifacts(self, mask, raw, optimized, report):
        from roadnet.skeleton_artifacts import save_skeleton_artifacts
        return save_skeleton_artifacts(
            os.path.join(os.getcwd(), "outputs", "skeleton"),
            mask,
            raw,
            optimized,
            report,
            image_rgb=self._layer_manager.full_image_rgb,
        )

    def _on_run_postprocess(self):
        """后处理（可选高级功能，带备份、异常检测和自动回滚）"""
        mask = self._layer_manager.get_layer_data("mask")
        if mask is None:
            QMessageBox.warning(self, "提示", "请先加载 Mask。")
            return

        self._history.push_state("postprocess")

        # ★ 备份当前 mask
        mask_backup = mask.copy()
        working_mask = self._apply_valid_area(mask)
        old_ratio = (working_mask > 0).sum() / working_mask.size

        try:
            from roadnet.postprocess import clean_pipeline, analyze_mask_anomalies
            config = self._param_panel.get_config()
            # ★ 传入后处理子配置
            post_cfg = config.get("postprocess", {})
            clean_mask, _ = clean_pipeline(
                working_mask, post_cfg, save_intermediate=False, output_dir=""
            )
            clean_mask = self._apply_valid_area(clean_mask)
        except Exception as e:
            import traceback
            traceback.print_exc()
            # ★ 回滚
            self._layer_manager.set_layer_data("mask", mask_backup)
            msg = f"后处理出错，已恢复原 Mask\n\n错误信息:\n{e}"
            QMessageBox.warning(self, "后处理失败", msg)
            return

        # ★ 验证后处理结果
        if clean_mask is None or clean_mask.size == 0:
            self._layer_manager.set_layer_data("mask", mask_backup)
            QMessageBox.warning(self, "后处理结果异常", "后处理输出了空 mask，已恢复原 Mask")
            return

        new_road = (clean_mask > 0).sum()
        if new_road == 0:
            self._layer_manager.set_layer_data("mask", mask_backup)
            QMessageBox.warning(self, "后处理结果异常", "后处理后道路像素数为 0，已恢复原 Mask")
            return

        total = clean_mask.size
        if new_road == total:
            self._layer_manager.set_layer_data("mask", mask_backup)
            QMessageBox.warning(self, "后处理结果异常", "后处理后 mask 全白（100% 道路），已恢复原 Mask")
            return

        new_ratio = new_road / total

        # ★ 面积异常检测
        anomaly_result = analyze_mask_anomalies(
            clean_mask,
            original_mask=working_mask,
            max_road_ratio=post_cfg.get("max_road_ratio_warn", 0.25),
            max_largest_ratio=post_cfg.get("max_largest_ratio_warn", 0.10),
            max_fill_added_ratio=post_cfg.get("max_fill_added_ratio_warn", 0.05),
        )

        anomaly_msg = ""
        if anomaly_result["is_anomalous"]:
            anomaly_msg = (
                f"\n\n⚠ 异常检测：\n"
                f"  道路占比: {anomaly_result['road_mask_area_ratio']*100:.1f}%\n"
                f"  最大连通域占比: {anomaly_result['largest_component_area_ratio']*100:.1f}%\n"
                f"  填洞新增: {anomaly_result['fill_added_area_ratio']*100:.1f}%\n\n"
                f"检测到以下警告:\n" +
                "\n".join(f"  • {w}" for w in anomaly_result["warnings"]) +
                "\n\n当前 mask 可能发生大面积误填，不建议继续生成 skeleton。\n"
                "建议降低 close、关闭 fill_small_holes、提高 min_area 或使用 Ignore 区域。"
            )

        # ★ 检查道路像素变化比例
        if new_ratio < old_ratio * 0.4 or new_ratio > old_ratio * 2.0:
            msg = (
                f"后处理前后道路像素占比变化较大:\n"
                f"  处理前: {old_ratio*100:.1f}%\n"
                f"  处理后: {new_ratio*100:.1f}%"
                + anomaly_msg
                + f"\n\n是否应用后处理结果？"
            )
            reply = QMessageBox.question(
                self, "后处理结果变化过大", msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                self._layer_manager.set_layer_data("mask", mask_backup)
                self._status_bar.show_message("后处理已取消，恢复原 Mask")
                return
        elif anomaly_result["is_anomalous"]:
            # 变化不剧烈但面积异常
            msg = (
                f"后处理结果可能大面积误填:\n"
                f"  处理前: {old_ratio*100:.1f}%\n"
                f"  处理后: {new_ratio*100:.1f}%"
                + anomaly_msg
                + f"\n\n是否仍要应用此结果？"
            )
            reply = QMessageBox.question(
                self, "后处理面积异常", msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                self._layer_manager.set_layer_data("mask", mask_backup)
                self._status_bar.show_message("后处理已取消（面积异常），恢复原 Mask")
                return

        # 应用结果
        self._layer_manager.set_layer_data("mask", clean_mask)
        self._clear_skeleton_state(clear_layer=True)
        self._status_bar.update_road_ratio(new_ratio)
        self._status_bar.show_message(f"后处理完成: 道路占比 {new_ratio*100:.1f}%")
        # ★ 保存备份用于可能的恢复
        self._mask_before_postprocess = mask_backup

    def _on_restore_pre_postprocess(self):
        """恢复后处理前的 mask"""
        if not hasattr(self, '_mask_before_postprocess') or self._mask_before_postprocess is None:
            QMessageBox.information(self, "提示", "没有可恢复的后处理前 Mask")
            return
        self._layer_manager.set_layer_data("mask", self._mask_before_postprocess)
        self._clear_skeleton_state(clear_layer=True)
        total = self._mask_before_postprocess.size
        road_px = (self._mask_before_postprocess > 0).sum()
        self._status_bar.update_road_ratio(road_px / total)
        self._status_bar.show_message("已恢复后处理前的 Mask")

    def _on_restore_original_roadmask(self):
        """从 SAM-Road 输出目录重新加载原始 road_mask.png，恢复原始 mask。

        用于紧急恢复被后处理/编辑污染的 mask。
        """
        source_dir = getattr(self, '_last_samroad_source_dir', None)
        if not source_dir or not os.path.isdir(source_dir):
            # 尝试旧版 samroad_adapter 路径
            reply = QMessageBox.question(
                self, "恢复原始 road_mask",
                "没有找到最近的 SAM-Road 输出目录。\n\n"
                "是否手动选择 SAM-Road 单图输出目录？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                source_dir = QFileDialog.getExistingDirectory(
                    self, "选择 SAM-Road 单图输出目录", os.getcwd(),
                    QFileDialog.Option.ShowDirsOnly,
                )
                if not source_dir or not os.path.isdir(source_dir):
                    return
            else:
                return

        # 尝试从文件直接加载
        road_mask_path = os.path.join(source_dir, "road_mask.png")
        if not os.path.exists(road_mask_path):
            QMessageBox.warning(self, "文件缺失",
                f"在目录中未找到 road_mask.png:\n{source_dir}")
            return

        try:
            mask_img = cv2.imread(road_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise ValueError("无法读取 road_mask.png")
        except Exception as e:
            QMessageBox.warning(self, "读取失败", f"读取 road_mask.png 失败:\n{e}")
            return

        # 二值化（如果非二值）
        unique_vals = np.unique(mask_img)
        is_binary = len(unique_vals) <= 2 or (
            len(unique_vals) <= 3 and 0 in unique_vals and 255 in unique_vals)
        if not is_binary:
            mask_bin = (mask_img > 30).astype(np.uint8) * 255
        else:
            mask_bin = mask_img
        mask_bin = self._apply_valid_area(mask_bin)

        self._history.push_state("restore_original_roadmask")
        self._layer_manager.set_layer_data("mask", mask_bin)
        self._clear_skeleton_state(clear_layer=True)
        total = mask_bin.size
        road_px = int((mask_bin > 0).sum())
        self._status_bar.update_road_ratio(road_px / total if total else 0)
        self._canvas.refresh_scene()
        self._status_bar.show_message(
            f"已从 {os.path.basename(source_dir)}/road_mask.png 恢复原始 road_mask "
            f"({mask_bin.shape[1]}x{mask_bin.shape[0]}, 道路占比 {road_px/total*100:.1f}%)"
        )
        # ★ 清除后处理备份（已过期）
        self._mask_before_postprocess = None
        QMessageBox.information(self, "恢复完成",
            f"已从 SAM-Road 输出目录恢复原始 road_mask。\n\n"
            f"来源: {source_dir}\n"
            f"尺寸: {mask_bin.shape[1]}x{mask_bin.shape[0]}\n"
            f"道路占比: {road_px/total*100:.1f}%")

    # ===================================================================
    # 新工作流：Mask 后处理 → Skeleton 生成 → Skeleton to Graph
    # ===================================================================

    def _on_mask_postprocess(self):
        """打开 SAM-Road mask 后处理参数对话框。

        允许用户反复调整阈值、形态学参数、面积过滤等，
        预览效果满意后应用写入 mask 图层。
        """
        if self._layer_manager.is_large_image_mode:
            self._on_large_mask_postprocess()
            return

        mask = self._layer_manager.get_layer_data("mask")
        if mask is None:
            QMessageBox.warning(self, "提示", "请先加载 Mask（可从 SAM-Road 导入或运行初提取）。")
            return

        roi_data = self._layer_manager.get_layer_data("roi")
        ignore_data = self._layer_manager.get_layer_data("ignore")
        outputs_dir = os.path.join(os.getcwd(), "outputs", "graph_build")

        from gui.mask_postprocess_dialog import MaskPostprocessDialog

        dialog = MaskPostprocessDialog(
            mask_data=self._apply_valid_area(mask),
            roi_data=roi_data,
            ignore_data=ignore_data,
            output_dir=outputs_dir,
            parent=self,
        )

        # 预览信号：临时显示到图层
        def _on_preview(processed, info):
            processed = self._apply_valid_area(processed)
            self._layer_manager.set_layer_data("mask", processed)
            self._canvas.refresh_scene()
            self._canvas.viewport().update()
            total = processed.size
            road_px = int((processed > 0).sum())
            self._status_bar.update_road_ratio(road_px / total if total else 0)
            self._status_bar.show_message(f"预览: {info}")

        # 应用信号：正式写入
        def _on_apply(processed):
            processed = self._apply_valid_area(processed)
            self._history.push_state("mask_postprocess")
            self._layer_manager.set_layer_data("mask", processed)
            self._clear_skeleton_state(clear_layer=True)
            self._canvas.refresh_scene()
            self._canvas.viewport().update()
            total = processed.size
            road_px = int((processed > 0).sum())
            self._status_bar.update_road_ratio(road_px / total if total else 0)
            self._status_bar.show_message(
                f"Mask 后处理已应用: {road_px} 道路像素 ({road_px / total * 100:.2f}%)"
            )

        dialog.preview_requested.connect(_on_preview)
        dialog.apply_requested.connect(_on_apply)
        dialog.exec()

    def _on_skeleton_from_mask(self):
        """从当前 mask 生成 skeleton 对话框。

        支持：
        - 选择骨架化方法
        - 调整短枝剪除长度
        - 可选端点连接（默认关闭）
        """
        mask, mask_meta = self.get_current_mask_array(prefer_processed=True)
        if mask is None:
            err_msg = mask_meta.get("error", "请先加载/处理后 Mask。")
            QMessageBox.warning(self, "提示",
                f"无法生成骨架：{err_msg}\n"
                "请确保已生成正式 global_road_mask.png。")
            return
        err = self._skeleton_input_check(mask, mask_meta)
        if err is not None:
            QMessageBox.warning(self, "骨架生成", err)
            return

        from gui.skeleton_gen_dialog import SkeletonGenDialog

        mask = self._apply_valid_area(mask)
        dialog = SkeletonGenDialog(mask_data=mask, parent=self)

        def _on_generated(skeleton, info):
            self._history.push_state("skeleton_from_mask")
            self._commit_raw_skeleton(skeleton)
            self._layer_manager.show_layer("skeleton")
            self._act_skeleton_visible.setChecked(True)
            self._sync_layer_checkboxes()
            self._canvas.refresh_scene()
            self._canvas.viewport().update()
            self._status_bar.show_message(info)
            self.mark_stage_done("segment")

        dialog.skeleton_generated.connect(_on_generated)
        dialog.exec()

    def _on_skeleton_optimize_full(self):
        """生成/优化道路骨架：完整流水线对话框。

        从当前 mask 出发，执行：
        mask 标准化 → 骨架化 → 边界过滤 → 距离变换过滤 →
        毛刺删除 → junction 聚类 → 可选端点连接 → 保存验证输出

        完成后写入：
        - processed_mask → mask 图层（标准化后的道路 mask）
        - skeleton → skeleton 图层（优化后的骨架）
        - skeleton_outputs/ 目录下保存所有验证文件
        """
        mask, mask_meta = self.get_current_mask_array(prefer_processed=True)
        if mask is None:
            err_msg = mask_meta.get("error", "请先加载/处理后 Mask。")
            QMessageBox.warning(self, "提示",
                f"无法生成/优化骨架：{err_msg}\n"
                "请确保已生成正式 global_road_mask.png。")
            return
        err = self._skeleton_input_check(mask, mask_meta)
        if err is not None:
            QMessageBox.warning(self, "骨架优化", err)
            return
        if self.skeleton_state == "optimized":
            reply = QMessageBox.question(
                self,
                "骨架已经优化",
                "当前骨架已经优化过。是否从 raw_skeleton 重新优化？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        mask = self._apply_valid_area(mask)
        image_rgb = self._layer_manager.image_rgb
        outputs_dir = os.path.join(os.getcwd(), "outputs")

        from gui.skeleton_optimize_dialog import SkeletonOptimizeDialog

        dialog = SkeletonOptimizeDialog(
            mask_data=mask,
            image_rgb=image_rgb,
            output_base_dir=outputs_dir,
            parent=self,
        )

        def _on_optimized(skeleton, result, info):
            self._history.push_state("skeleton_optimize_full")
            skeleton = self._apply_valid_area(skeleton)
            raw_skeleton = self._apply_valid_area(result.raw_skeleton)

            # 1. 写入 processed_mask → mask 图层
            if result.normalized_mask is not None:
                normalized_mask = self._apply_valid_area(result.normalized_mask)
                self._layer_manager.set_layer_data("mask", normalized_mask)
                total = normalized_mask.size
                road_px = int((normalized_mask > 0).sum())
                self._status_bar.update_road_ratio(road_px / total if total else 0)

            # 2. 写入 skeleton → skeleton 图层
            self._commit_optimized_skeleton(raw_skeleton, skeleton)
            self._layer_manager.show_layer("skeleton")
            self._act_skeleton_visible.setChecked(True)
            self._sync_layer_checkboxes()
            self._canvas.refresh_scene()
            self._canvas.viewport().update()

            # 3. 状态更新
            self._status_bar.show_message(info)
            self.mark_stage_done("segment")
            self.mark_stage_done("skeleton")

            # 4. 通知用户保存的文件
            saved = result.saved_files
            if saved:
                file_names = ", ".join(
                    os.path.basename(p) for p in saved.values()
                )
                self._status_bar.show_message(
                    f"{info} | 已保存: {file_names}"
                )
            if result.stats.get("removed_ratio", 0.0) > 0.60:
                QMessageBox.warning(
                    self,
                    "骨架可能过度剪枝",
                    "骨架优化删除比例过高，可能发生过度剪枝，请降低 "
                    "min_branch_length 或从 raw_skeleton 重新生成。",
                )

        dialog.skeleton_optimized.connect(_on_optimized)
        dialog.exec()

    def _on_graph_from_skeleton(self):
        """从当前 skeleton 生成 graph 对话框。

        使用 graph_build 统一流水线：
        1. 读取并验证 skeleton
        2. skeleton_to_graph 生成 raw graph
        3. 保存 final_graph_raw.json
        4. 加载到 graph_editor
        5. 渲染 Final Graph 图层
        6. 可选执行 graph_line_optimizer

        如果任何阶段失败，保留 raw graph。
        """
        skeleton = self._layer_manager.get_layer_data("skeleton")
        if skeleton is None:
            QMessageBox.warning(self, "提示", "请先生成 Skeleton。")
            return

        skeleton = self._get_skeleton_array(skeleton)
        outputs_dir = os.path.join(os.getcwd(), "outputs")

        from gui.skeleton_to_graph_dialog import SkeletonToGraphDialog

        dialog = SkeletonToGraphDialog(
            skeleton_data=skeleton,
            output_dir=outputs_dir,
            parent=self,
        )

        def _on_generated(nodes, edges, info):
            self._history.push_state("graph_from_skeleton")

            build_config, raw_debug, run_optimizer = dialog.get_build_options()
            road_mask = self._layer_manager.get_layer_data("mask")

            # ★ 使用 graph_build 统一流水线（与一键流程一致）
            from roadnet.graph_build import (
                build_graph_from_skeleton, build_graph_debug_mode,
            )

            if raw_debug:
                result = build_graph_debug_mode(
                    skeleton=skeleton,
                    output_dir=outputs_dir,
                    config=build_config,
                    road_mask=road_mask,
                )
            else:
                result = build_graph_from_skeleton(
                    skeleton=skeleton,
                    graph_editor=self._graph_editor,
                    output_dir=outputs_dir,
                    config=build_config,
                    processed_mask=road_mask,
                    run_optimization=run_optimizer,
                )

            if not result.success:
                # 流水线失败
                err_msgs = "\n".join(result.errors)
                QMessageBox.critical(
                    self, "Graph 生成失败",
                    f"在阶段 [{result.stage}] 失败:\n{err_msgs}\n\n"
                    "请检查 skeleton 是否为空，或查看控制台日志了解详情。"
                )
                self._status_bar.show_message(f"Graph 生成失败: {result.stage}")
                return

            # 流水线成功 → 渲染 graph
            if not raw_debug:
                self._render_graph_to_scene()
                self._update_graph_stats()
            self._status_bar.update_nodes(len(result.raw_nodes))
            self._status_bar.update_edges(len(result.raw_edges))

            # 构建成功消息
            errors_text = ""
            if result.errors:
                errors_text = f"\n⚠ 部分操作有警告:\n" + "\n".join(result.errors)
            self._status_bar.show_message(
                f"Graph 生成完成: {len(result.raw_nodes)} 节点, {len(result.raw_edges)} 边"
                f"{errors_text}"
            )
            self.mark_stage_done("skeleton")

            # ★ 显示详细日志（弹窗）
            log_lines = result.log.messages[-6:]  # 最近 6 条
            log_summary = "\n".join(f"  {m}" for m in log_lines)
            msg = (f"Graph 生成成功！\n\n"
                   f"节点: {len(result.raw_nodes)}, 边: {len(result.raw_edges)}\n\n"
                   f"日志:\n{log_summary}")
            if result.raw_graph_path:
                msg += f"\n\n已保存: {os.path.basename(result.raw_graph_path)}"
            if result.connectivity and result.connectivity.get("connected_components", 0) > 1:
                count = result.connectivity["connected_components"]
                msg += (f"\n\n当前 final_graph 不连通，共 {count} 个连通分量。"
                        "建议增大 endpoint_connect_distance 或手动补边。")
            QMessageBox.information(self, "Graph 生成完成", msg)

        dialog.graph_generated.connect(_on_generated)
        dialog.exec()

    def _on_graph_line_optimize(self):
        """对 final_graph 的 edge polyline 进行线形优化。

        只在已有 final_graph 时可用。
        流程: RDP 简化 → 直线拉直/弯路平滑 → mask 校验 → 更新 graph editor。
        """
        if self._graph_editor is None:
            QMessageBox.warning(self, "提示", "请先从 skeleton 生成 graph。")
            return

        ge = self._graph_editor
        if not ge.edges:
            QMessageBox.warning(self, "提示", "当前 final_graph 中没有边。")
            return

        # 获取 processed_mask（如果可用）
        processed_mask = self._layer_manager.get_layer_data("mask")

        # 获取原图用于预览
        image_rgb = None
        try:
            image_rgb = self._layer_manager.get_image_rgb()
        except Exception:
            pass

        # 输出目录
        outputs_dir = os.path.join(os.getcwd(), "outputs", "graph_line_optimize_outputs")

        from gui.graph_line_optimize_dialog import GraphLineOptimizeDialog

        dialog = GraphLineOptimizeDialog(
            edges=ge.edges,
            processed_mask=processed_mask,
            image_rgb=image_rgb,
            output_dir=outputs_dir,
            parent=self,
        )

        def _on_optimized(optimized_edges, report):
            # 推入撤销状态（优化前备份当前 edges 深拷贝）
            self._history.push_state("graph_line_optimize")

            # ★ 备份原始 edges（如果优化后面临回退）
            edges_before = [dict(e) for e in ge._edges]

            # 替换 graph editor 中的边数据
            ge._edges = optimized_edges
            ge._undo_stack.push(ge._nodes, ge._edges)

            # 重新渲染
            self._render_graph_to_scene()
            self._update_graph_stats()

            s = report["summary"]
            self._status_bar.show_message(
                f"Graph 线形优化完成: {s['straightened_edges']}条拉直, "
                f"{s['smoothed_edges']}条平滑, "
                f"{s['mask_rollback_edges']}条回退, "
                f"点数减少{s['points_reduction_pct']}%"
            )

            # ★ 保存 final_graph_optimized.json 到 outputs/ 主目录
            #   （与 final_graph_raw.json 同级，便于对比查找）
            try:
                from roadnet.skeleton_to_graph import save_graph_from_skeleton
                main_outputs = os.path.join(os.getcwd(), "outputs")
                os.makedirs(main_outputs, exist_ok=True)
                save_graph_from_skeleton(
                    [dict(n) for n in ge.nodes],
                    optimized_edges,
                    main_outputs,
                    filename="final_graph_optimized.json",
                )
                print("[OPT] final_graph_optimized.json 已保存")
            except Exception as save_err:
                print(f"[OPT] 保存 final_graph_optimized.json 失败: {save_err}")
                # ★ 保存失败不影响已优化的 graph 在编辑器中显示

        dialog.optimization_finished.connect(_on_optimized)
        dialog.exec()

    # ===================================================================
    # 任务点相关处理
    # ===================================================================

    @property
    def geo_calibration(self):
        """任务点模块使用的唯一地理标定对象。"""
        return self._geo_calibration

    @property
    def raw_task_points(self):
        """未经 graph 吸附的原始任务点稳定视图。"""
        return self._task_points

    @property
    def task_points(self):
        """统一任务点列表（文件导入 + 手动点击）。"""
        return self._task_points

    @task_points.setter
    def task_points(self, value):
        self._task_points = list(value or [])

    def _sync_task_points_table(self):
        if hasattr(self._param_panel, "update_task_points_table"):
            self._param_panel.update_task_points_table(
                self._task_points, self.snapped_task_points or self._snapped_points,
            )

    def _normalize_task_points(self, reason=""):
        """稳定归一化任务点并在顺序变化时废弃旧吸附/路径结果。

        文件导入点（source=file_import）只按 seq 排序，不按 type 重排。
        """
        from roadnet.task_points import normalize_task_point_sequence
        before = [(int(tp.seq), int(tp.point_type), int(getattr(tp, "created_order", 0)))
                  for tp in self._task_points]
        normalize_task_point_sequence(self._task_points)
        after = [(int(tp.seq), int(tp.point_type), int(getattr(tp, "created_order", 0)))
                 for tp in self._task_points]
        changed = before != after
        if changed:
            print(f"[TaskPoints] normalized ({reason}): {before} -> {after}")
            self._snapped_points = []
            self.snapped_task_points = []
            self._reset_planned_path_data()
            self._clear_planned_path_items()
        return changed

    def _on_task_point_clicked(self, tp_type: str, x_global: int, y_global: int):
        """画布点击添加任务点（通过工具按钮 set_start/set_end/add_task）"""
        from roadnet.task_points import TaskPoint

        self._history.push_state("add_task_point")
        next_seq = len(self._task_points) + 1
        created_order = max(
            [int(getattr(tp, "created_order", index)) for index, tp in enumerate(self._task_points)]
            or [-1]
        ) + 1

        pt_map = {"start": 0, "goal": 1, "task": 2}
        point_type = pt_map.get(tp_type, 2)

        # 添加新的起点/终点时，旧同类点自动降为必经点；添加阶段不报顺序错误。
        if point_type in (0, 1):
            for existing in self._task_points:
                if int(existing.point_type) == point_type:
                    existing.point_type = 2

        lon = lat = None
        status = "pixel_only"
        geo = self.geo_calibration
        if geo is not None and geo.is_valid:
            try:
                # pixel → WGS84 反算，写入 TaskPoint lon/lat
                lon, lat = geo.pixel_to_wgs84(float(x_global), float(y_global))
                status = "ok"
            except Exception as exc:
                print(f"[TaskPoints] manual pixel→WGS84 failed: {exc}")

        tp = TaskPoint(
            seq=next_seq,
            longitude=None if lon is None else float(lon),
            latitude=None if lat is None else float(lat),
            altitude=0.0,
            point_type=point_type,
            pixel_x=float(x_global),
            pixel_y=float(y_global),
            status=status,
            inside_image=True,
            created_order=created_order,
            source="manual_click",
        )
        self._task_points.append(tp)
        self._normalize_task_points("manual_add")
        self._snapped_points = []
        self.snapped_task_points = []
        self._reset_planned_path_data()
        self._clear_planned_path_items()

        # 重新渲染（增量）
        self._clear_task_point_items()
        self._render_task_points_to_scene()
        self._layer_manager.show_layer("task_points")
        self._sync_task_points_table()

        type_name = tp.type_name
        self._status_bar.show_message(
            f"已添加 {type_name} 点 (seq={tp.seq}, x={x_global}, y={y_global}"
            + (f", lon={lon:.6f}, lat={lat:.6f}" if lon is not None else "")
            + ")"
        )

        # 添加后自动切换回 pan 工具
        self.set_tool("pan")

    def _on_import_task_points(self):
        """导入比赛任务点 txt：序号;经度;纬度;高程;属性。

        ★ 必须已完成 geo_calibration；按 seq 排序规划，不按空间距离重排。
        """
        geo = self.geo_calibration
        if geo is None or not geo.is_valid:
            QMessageBox.warning(
                self, "请先完成坐标校准",
                "请先完成坐标校准，再导入或规划经纬度任务点。",
            )
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self, "导入任务点文件",
            os.getcwd(),
            "任务点文件 (*.txt *.csv);;所有文件 (*.*)",
        )
        if not filepath:
            return

        try:
            from roadnet.task_points import (
                parse_task_points_txt, apply_lon_lat_swap,
                save_task_points_loaded, save_task_points_import_report,
            )
            from roadnet.task_point_coordinates import convert_task_points_to_image

            parsed = parse_task_points_txt(filepath)
            if not parsed.get("ok"):
                QMessageBox.warning(
                    self, "导入失败",
                    parsed.get("error")
                    or "任务点文件格式错误，应为：序号;经度;纬度;高程;属性。",
                )
                return

            task_points = list(parsed["points"])
            if parsed.get("swap_suspect"):
                reply = QMessageBox.question(
                    self, "经纬度可能填反",
                    "检测到经纬度可能填反，是否交换经纬度？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    apply_lon_lat_swap(task_points)

            for wmsg in parsed.get("warnings") or []:
                print(f"[TaskImport] {wmsg}")

            self._history.push_state("import_task_points")

            w, h = self._layer_manager.original_size
            diagnostics, size_info = convert_task_points_to_image(
                task_points, geo, (w, h),
            )
            for tp in task_points:
                tp.source = "file_import"

            self._task_point_diagnostics = diagnostics
            # 文件导入：只按 seq 排序，不调用 type 重排
            task_points.sort(key=lambda tp: int(tp.seq))
            self._task_points = task_points
            out_of_bounds = sum(1 for point in task_points if point.inside_image is False)
            converted = sum(1 for point in task_points if point.pixel_x is not None)

            self._snapped_points = []
            self.snapped_task_points = []
            self._reset_planned_path_data()
            self._clear_planned_path_items()

            outputs_dir = os.path.join(os.getcwd(), "outputs")
            if self._large_image_project is not None:
                outputs_dir = os.path.join(
                    self._large_image_project.project_dir, "outputs"
                )
            os.makedirs(outputs_dir, exist_ok=True)
            save_task_points_loaded(task_points, outputs_dir)
            report = {
                "file_path": filepath,
                "point_count": len(task_points),
                "start_count": parsed.get("start_count", 0),
                "goal_count": parsed.get("goal_count", 0),
                "waypoint_count": parsed.get("waypoint_count", 0),
                "invalid_rows": parsed.get("invalid_rows", 0),
                "geo_calibration_valid": True,
                "converted_pixel_count": converted,
                "out_of_image_count": out_of_bounds,
                "warnings": list(parsed.get("warnings") or []),
                "visit_order_by_seq": [int(tp.seq) for tp in task_points],
            }
            if size_info.get("size_mismatch"):
                report["warnings"].append("校准影像尺寸与当前影像不一致")
            report_path = save_task_points_import_report(report, outputs_dir)

            self._layer_manager.show_layer("task_points")
            self._layer_manager.show_layer("planned_path")
            self._render_task_points_to_scene()
            self._act_tp_original_visible.setChecked(True)
            self._sync_layer_checkboxes()
            self._sync_task_points_table()

            start_count = report["start_count"]
            goal_count = report["goal_count"]
            via_count = report["waypoint_count"]
            msg = (
                f"任务点导入成功: {len(task_points)} 个 "
                f"(起点={start_count}, 终点={goal_count}, 必经={via_count})，"
                f"已按 seq 排序，已转像素 {converted}"
            )
            self._status_bar.show_message(msg)

            warn_text = ""
            if parsed.get("warnings"):
                warn_text = "\n\n" + "\n".join(parsed["warnings"])
            if size_info.get("size_mismatch"):
                QMessageBox.warning(
                    self, "影像尺寸不一致",
                    "校准文件尺寸与当前影像尺寸不一致，任务点转换可能错位。\n\n"
                    f"校准尺寸：{size_info['calibration_image_width']} × "
                    f"{size_info['calibration_image_height']}\n"
                    f"当前尺寸：{size_info['image_width']} × {size_info['image_height']}"
                )
            if out_of_bounds > 0:
                QMessageBox.warning(
                    self, "任务点超出影像范围",
                    f"{out_of_bounds} 个任务点转换后位于影像范围外。\n"
                    "图上已用红色 OUTSIDE 标记。"
                )
            QMessageBox.information(
                self, "任务点导入完成",
                f"{msg}\n"
                f"访问顺序：{' → '.join(str(s) for s in report['visit_order_by_seq'])}\n"
                f"报告：{report_path}"
                f"{warn_text}"
            )
            return
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", str(exc))
            import traceback
            traceback.print_exc()

    def _on_manage_task_points(self):
        """打开可编辑任务点列表；修改仅在用户保存后提交。"""
        if not self._task_points:
            QMessageBox.information(self, "任务点管理", "当前没有任务点。")
            return
        from gui.task_point_dialog import TaskPointManagerDialog
        dialog = TaskPointManagerDialog(self._task_points, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        self._history.push_state("edit_task_points")
        self._task_points = dialog.task_points()
        self._normalize_task_points("manager_save")
        self._snapped_points = []
        self.snapped_task_points = []
        self._reset_planned_path_data()
        self._clear_planned_path_items()
        self._render_task_points_to_scene()
        self._sync_task_points_table()
        self._status_bar.show_message(
            f"任务点修改已保存，共 {len(self._task_points)} 个；旧吸附结果已清除"
        )

    def _on_validate_task_point_coordinates(self):
        """执行 WGS84 往返检查、路网距离检查并导出 debug CSV。"""
        if not self._task_points:
            QMessageBox.information(self, "验证任务点坐标", "当前没有任务点。")
            return
        geo = self.geo_calibration
        if geo is None or not geo.is_valid:
            QMessageBox.warning(self, "验证任务点坐标", "geo_calibration 无效，请先完成坐标校准。")
            return
        try:
            from roadnet.task_point_coordinates import (
                build_task_point_debug_rows, convert_task_points_to_image,
                save_task_points_debug_csv,
            )
            w, h = self._layer_manager.original_size
            diagnostics, size_info = convert_task_points_to_image(
                self._task_points, geo, (w, h)
            )
            self._task_point_diagnostics = diagnostics
            edges = self._graph_editor.edges if self._graph_editor is not None else []
            rows = build_task_point_debug_rows(self._task_points, (w, h), edges)
            output_dir = os.path.join(os.getcwd(), "outputs", "task_points")
            csv_path = save_task_points_debug_csv(
                rows, os.path.join(output_dir, "task_points_debug.csv")
            )
            self._render_task_points_to_scene()
            ok_count = sum(row["status"] == "ok" for row in rows)
            warning_count = sum(row["status"] == "warning" for row in rows)
            failed_count = sum(row["status"] == "failed" for row in rows)
            max_roundtrip = max(
                [float(row.get("roundtrip_error_m") or 0.0) for row in diagnostics] or [0.0]
            )
            mismatch_text = "是（已按比例修正）" if size_info.get("size_mismatch") else "否"
            QMessageBox.information(
                self, "任务点坐标验证完成",
                f"任务点：{len(rows)}\n"
                f"ok={ok_count}, warning={warning_count}, failed={failed_count}\n"
                f"最大往返误差：{max_roundtrip:.6f} m\n"
                f"校准/影像尺寸不一致：{mismatch_text}\n\n"
                f"已导出：\n{csv_path}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "任务点坐标验证失败", str(exc))
            import traceback
            traceback.print_exc()

    def _on_snap_task_points(self):
        """自动吸附所有任务点到路网图"""
        if not self._task_points:
            QMessageBox.warning(self, "提示", "请先导入任务点。")
            return

        ge = self._graph_editor
        if not ge.nodes or not ge.edges:
            QMessageBox.warning(self, "提示", "请先生成或导入路网图 (final_graph)。")
            return

        # 检查坐标
        if all(tp.pixel_x is None for tp in self._task_points):
            QMessageBox.warning(
                self, "提示",
                "所有任务点都没有像素坐标。\n请先完成地理标定，或手动设置任务点像素坐标。"
            )
            return

        try:
            from roadnet.task_snapping import SnapConfig, snap_all_task_points

            config = SnapConfig(
                max_snap_distance_px=250.0,
                warning_distance_px=50.0,
                search_radius_px=150.0,
                top_k=5,
                prefer_edge=True,
                allow_virtual_node=True,
                expand_radius_on_fail=True,
                expanded_radius_px=250.0,
            )

            self._snapped_points = snap_all_task_points(
                self._task_points, ge.nodes, ge.edges, config
            )
            self.snapped_task_points = list(self._snapped_points)
            snap_by_seq = {int(sp.seq): sp for sp in self._snapped_points}
            for tp in self._task_points:
                sp = snap_by_seq.get(int(tp.seq))
                if sp is None:
                    continue
                tp.snap_status = sp.status
                tp.snap_distance = float(sp.snap_distance)

            # 统计结果
            ok_count = sum(1 for sp in self._snapped_points if sp.status == "ok")
            warn_count = sum(1 for sp in self._snapped_points if sp.status == "warning")
            fail_count = sum(1 for sp in self._snapped_points if sp.status == "failed")

            # 重新渲染（含吸附结果）
            self._clear_task_point_items()
            self._render_task_points_to_scene()
            self._sync_task_points_table()

            self._status_bar.show_message(
                f"吸附完成: {ok_count} ok, {warn_count} warning, {fail_count} failed"
            )
            if fail_count > 0:
                QMessageBox.warning(
                    self, "吸附失败点",
                    f"{fail_count} 个任务点吸附距离超过上限，已标记 failed。\n"
                    "正式导出前请修正任务点或路网。"
                )

        except Exception as e:
            QMessageBox.critical(self, "错误", f"吸附失败: {e}")
            import traceback
            traceback.print_exc()

    def _on_run_global_plan(self, algorithm: str = "astar"):
        """运行全局路径规划。

        ★ 前置检查（5项）：
        1. 是否已导入任务点文件
        2. 是否已完成坐标校准
        3. 是否存在 final_graph
        4. final_graph 节点数 > 0
        5. final_graph 边数 > 0
        """
        # ── 前置检查 1: 是否已导入任务点 ──
        if not self._task_points:
            QMessageBox.warning(self, "没有任务点",
                "请先导入任务点文件。\n\n"
                "菜单: 规划 → 导入任务点文件")
            return

        self._normalize_task_points("before_snap")

        # 添加阶段允许任意顺序；只在规划阶段统一归一化并验证。
        self._normalize_task_points("before_plan")
        from roadnet.task_points import validate_task_points_for_planning
        sequence_errors = validate_task_points_for_planning(self._task_points)
        if sequence_errors:
            QMessageBox.warning(
                self,
                "任务点顺序无效",
                "规划路径必须按任务点 seq 顺序经过，且包含一个起点和一个终点：\n\n"
                + "\n".join(f"• {error}" for error in sequence_errors),
            )
            return

        # ── 前置检查 2: 是否已完成坐标校准 ──
        needs_geo = any(
            getattr(tp, "source", "") == "file_import"
            or (tp.longitude is not None and tp.latitude is not None and tp.pixel_x is None)
            for tp in self._task_points
        )
        if needs_geo and (self._geo_calibration is None or not self._geo_calibration.is_valid):
            QMessageBox.warning(
                self, "请先完成坐标校准",
                "请先完成坐标校准，再导入或规划经纬度任务点。",
            )
            return
        if self._geo_calibration is None or not self._geo_calibration.is_valid:
            reply = QMessageBox.warning(self, "未完成坐标校准",
                "当前尚未完成坐标校准。\n\n"
                "如果任务点使用经纬度坐标，需要先完成校准才能转换为像素坐标。\n"
                "若任务点已是像素坐标，可忽略此警告继续。\n\n"
                "是否继续规划？",
                QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.No:
                # 允许继续，但检查像素坐标
                pass
            else:
                return

        # ── 前置检查 3: 是否存在 final_graph ──
        if self._graph_editor is None:
            QMessageBox.warning(self, "没有 final_graph",
                "当前还没有 final_graph，请先从 skeleton 生成 graph。\n\n"
                "菜单: 工具 → 从 Skeleton 生成 Graph")
            return

        ge = self._graph_editor

        # ── 前置检查 4: final_graph 节点数 > 0 ──
        if not ge.nodes or len(ge.nodes) == 0:
            QMessageBox.warning(self, "final_graph 节点为空",
                "当前 final_graph 中没有节点。\n\n"
                "请先从 skeleton 生成 graph。\n"
                "菜单: 工具 → 从 Skeleton 生成 Graph")
            return

        # ── 前置检查 5: final_graph 边数 > 0 ──
        if not ge.edges or len(ge.edges) == 0:
            QMessageBox.warning(self, "final_graph 边为空",
                "当前 final_graph 中没有边。\n\n"
                "请先从 skeleton 生成 graph。\n"
                "菜单: 工具 → 从 Skeleton 生成 Graph")
            return

        # ── 自动吸附（如果尚未吸附）──
        if not self._snapped_points:
            self._on_snap_task_points()
            if not self._snapped_points:
                QMessageBox.warning(self, "提示", "请先导入并吸附任务点。")
                return

        # ── 检查有无吸附失败点 ──
        failed = [sp for sp in self._snapped_points if sp.status == "failed"]
        if failed:
            failed_seqs = [sp.seq for sp in failed]
            reply = QMessageBox.question(
                self, "任务点距离路网太远",
                f"{len(failed)} 个任务点吸附失败 (seq={failed_seqs})，\n"
                f"距离过远，无法吸附到路网边。\n\n"
                f"是否跳过这些点继续规划？\n"
                f"（不可达的段将被跳过）",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        try:
            from roadnet.global_planner import (
                PlannerConfig, plan_global_path,
            )

            # ★ 获取像素分辨率（优先用校准结果，否则用项目默认值）
            pixel_res_m = 0.5
            if self._geo_calibration and self._geo_calibration.is_valid:
                pixel_res_m = self._geo_calibration.pixel_resolution_estimated_m or 0.5

            config = PlannerConfig(
                algorithm=algorithm,
                pixel_resolution_m=pixel_res_m,
                resample_spacing_px=20.0,
            )

            result = plan_global_path(
                self._snapped_points, ge.nodes, ge.edges, config
            )

            self._set_planned_path_result(result)

            # 渲染路径
            self._clear_planned_path_items()
            self._render_planned_path_to_scene()

            if result.success:
                seg_count = len(result.segments)
                self._status_bar.show_message(
                    f"规划完成: {seg_count} 段, 总长 {result.total_length_px:.1f}px"
                )
            else:
                unreach = [s for s in result.segments if s.status == "unreachable"]
                unreach_seqs = [(s.from_seq, s.to_seq) for s in unreach]
                repair_candidate = None
                failed_segment = None
                if unreach_seqs:
                    failed_segment = unreach_seqs[0]
                    repair_candidate = self._suggest_planning_bridge(*failed_segment)
                if repair_candidate is not None:
                    repair_candidate["from_seq"], repair_candidate["to_seq"] = failed_segment
                    repair_candidate["id"] = (
                        f"planning_bridge_{failed_segment[0]}_{failed_segment[1]}"
                    )
                    self._graph_repair_candidates = [repair_candidate]
                    self._graph_repair_signature = self._current_graph_signature()
                    self._render_graph_repair_candidates()
                    reply = QMessageBox.question(
                        self, "规划失败 - 发现补边建议",
                        f"seq={failed_segment[0]} 到 seq={failed_segment[1]} 不连通。\n"
                        f"最近可连接端点距离 {repair_candidate['distance_px']:.1f}px，建议补边。\n"
                        f"置信度：{repair_candidate['confidence']:.2f}\n\n"
                        "是否应用该修复并重新规划？",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        self._apply_graph_repairs({repair_candidate["id"]}, rerun_plan=True)
                        return
                self._status_bar.show_message(
                    f"规划部分成功, {len(unreach)} 段不可达"
                )
                QMessageBox.warning(
                    self, "两个任务点之间路网不连通",
                    f"{len(unreach)} 段路径不可达:\n"
                    + "\n".join(f"  seq={a}→{b}" for a, b in unreach_seqs)
                    + "\n\n请检查路网连通性或手动修正吸附点。"
                )

        except Exception as e:
            QMessageBox.critical(self, "规划失败", f"路径规划出错:\n{e}")
            import traceback
            traceback.print_exc()

    def _suggest_planning_bridge(self, from_seq, to_seq):
        from roadnet.graph_auto_repair import suggest_component_bridge
        snapped_by_seq = {int(point.seq): point for point in self._snapped_points}
        left = snapped_by_seq.get(int(from_seq))
        right = snapped_by_seq.get(int(to_seq))
        if left is None or right is None or self._graph_editor is None:
            return None

        def nearest_node_id(point):
            node_id = getattr(point, "node_id", None)
            if node_id is not None:
                return node_id
            px = float(getattr(point, "snapped_x", 0.0))
            py = float(getattr(point, "snapped_y", 0.0))
            best = None
            for node in self._graph_editor.nodes:
                distance = (float(node["x"]) - px) ** 2 + (float(node["y"]) - py) ** 2
                if best is None or distance < best[0]:
                    best = (distance, node["id"])
            return best[1] if best else None

        node_a, node_b = nearest_node_id(left), nearest_node_id(right)
        if node_a is None or node_b is None:
            return None
        return suggest_component_bridge(
            self._graph_editor.nodes,
            self._graph_editor.edges,
            node_a, node_b,
            max_distance=50.0,
            road_mask=self._layer_manager.get_layer_data("mask"),
        )

    def _reset_planned_path_data(self):
        """清空规划数据对象；场景图元由调用方按需清理。"""
        self._global_plan_result = None
        self.planning_result = None
        self.planned_path_pixel = []
        self.planned_path_geo = []
        self.planned_path_edges = []
        self.sparse_waypoints_pixel = []
        self.sparse_waypoints_geo = []
        self.sparse_waypoints = []
        self._clear_sparse_waypoint_items()

    def _set_planned_path_result(self, result):
        """把规划结果提交到稳定数据字段，再由这些字段驱动紫色线渲染。"""
        self._global_plan_result = result  # 旧代码兼容
        self.planning_result = result
        self.snapped_task_points = list(self._snapped_points)
        self.planned_path_pixel = [
            [float(point[0]), float(point[1])]
            for point in (getattr(result, "global_path_points", None) or [])
            if point is not None and len(point) >= 2
        ]
        self.planned_path_edges = [
            edge_id
            for segment in (getattr(result, "segments", None) or [])
            for edge_id in (getattr(segment, "edge_path", None) or [])
        ]
        self.planned_path_geo = []
        if len(self.planned_path_pixel) > 1 and self._geo_calibration is not None:
            try:
                from roadnet.path_export import convert_pixel_path_to_geo
                self.planned_path_geo = convert_pixel_path_to_geo(
                    self.planned_path_pixel, self._geo_calibration
                )
            except Exception as exc:
                # 像素路径仍然有效；导出时会给用户显示完整的标定错误。
                print(f"[Planning] 暂未生成 planned_path_geo: {exc}")

    def _on_save_snapped_points(self):
        """保存吸附结果"""
        if not self._snapped_points:
            QMessageBox.warning(self, "提示", "请先执行任务点吸附。")
            return
        try:
            from roadnet.task_points import save_snapped_results
            outputs_dir = os.path.join(os.getcwd(), "outputs")
            path = save_snapped_results(self._snapped_points, outputs_dir)
            self._status_bar.show_message(f"吸附结果已保存: {path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def _on_save_global_path(self):
        """兼容旧菜单/调用方：统一进入正式路径导出。"""
        self._on_export_planned_path()

    def _on_clear_task_points(self):
        """清空所有任务点"""
        self._history.push_state("clear_task_points")
        self._task_points = []
        self._task_point_diagnostics = []
        self._snapped_points = []
        self.snapped_task_points = []
        self._reset_planned_path_data()
        self._clear_task_point_items()
        self._clear_planned_path_items()
        self._canvas.viewport().update()
        self._sync_task_points_table()
        self._status_bar.show_message("任务点已清空")

    # ===================================================================
    # 任务点与规划路径渲染
    # ===================================================================

    def _render_task_points_to_scene(self):
        """将任务点和吸附结果渲染到 QGraphicsScene

        ★ 使用高 z-value (1000+) 确保任务点始终在最上层。
        ★ 如果 task_points 图层被隐藏但有任务点数据，自动强制显示。
        """
        from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsTextItem
        from PySide6.QtGui import QPen, QBrush, QPainterPath, QColor, QFont

        scene = self._canvas.scene()
        if scene is None:
            return
        lm = self._layer_manager

        # 清除旧 items
        self._clear_task_point_items()

        if not self._task_points:
            return

        # ★ 强制显示 task_points 图层（如果有点但被隐藏）
        if not lm.is_layer_visible("task_points"):
            lm.show_layer("task_points")
            # 尝试全名
            if not lm.is_layer_visible("task_points"):
                lm.show_layer("layer_task_points")

        has_coords = 0
        no_coords = 0
        w, h = lm.original_size

        show_raw = not hasattr(self, "_act_tp_original_visible") or self._act_tp_original_visible.isChecked()
        if show_raw:
            via_index = 0
            for tp in sorted(self._task_points, key=lambda point: int(point.seq)):
                px, py = tp.pixel_x, tp.pixel_y
                if px is None or py is None:
                    no_coords += 1
                    print(f"[TaskPoints] seq={tp.seq} lon={tp.longitude} lat={tp.latitude} "
                          "-> pixel=None (无像素坐标)")
                    continue

                inside = (0 <= px < w and 0 <= py < h)
                tp.inside_image = inside
                if not inside:
                    tp.status = "warning"
                    print(f"[TaskPoints] seq={tp.seq} lon={tp.longitude} lat={tp.latitude} "
                          f"-> pixel=({px:.1f},{py:.1f}), outside_image ({w}x{h})")
                else:
                    has_coords += 1

                # 超界点钉在最近的影像边缘并画红叉，避免静默消失。
                draw_x = min(max(float(px), 0.0), max(0.0, float(w) - 1.0))
                draw_y = min(max(float(py), 0.0), max(0.0, float(h) - 1.0))
                tx, ty = lm.global_to_preview_f(draw_x, draw_y)

                if not inside:
                    color = QColor(255, 45, 45)
                    label = f"OUTSIDE · seq={tp.seq} · type={tp.point_type}"
                elif tp.point_type == 0:
                    color = QColor(40, 200, 90)
                    label = f"S/START · seq={tp.seq} · type=0"
                elif tp.point_type == 1:
                    color = QColor(245, 70, 70)
                    label = f"G/GOAL · seq={tp.seq} · type=1"
                else:
                    via_index += 1
                    color = QColor(50, 145, 255)
                    label = f"P{via_index} · seq={tp.seq} · type=2"

                tp_z = 1000
                radius = 8 if tp.point_type in (0, 1) else 6
                pen = QPen(color, 3)
                pen.setCosmetic(True)
                if inside:
                    dot = QGraphicsEllipseItem(tx - radius, ty - radius, radius * 2, radius * 2)
                    dot.setPen(pen)
                    dot.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 150)))
                    dot.setZValue(tp_z)
                    scene.addItem(dot)
                    self._task_point_original_items.append(dot)
                else:
                    cross_size = 10
                    for line in (
                        scene.addLine(tx - cross_size, ty - cross_size, tx + cross_size, ty + cross_size, pen),
                        scene.addLine(tx - cross_size, ty + cross_size, tx + cross_size, ty - cross_size, pen),
                    ):
                        line.setZValue(tp_z)
                        self._task_point_original_items.append(line)

                text_item = QGraphicsTextItem(label)
                text_item.setDefaultTextColor(color)
                text_item.setFont(QFont("Arial", 10, QFont.Bold))
                text_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                text_item.setPos(tx + 12, ty - 12)
                text_item.setZValue(tp_z + 1)
                scene.addItem(text_item)
                self._task_point_original_items.append(text_item)

        print(f"[TaskPoints] rendered task point items = {len(self._task_point_original_items)}, "
              f"has_coords={has_coords}, no_coords={no_coords}")

        # 渲染吸附结果
        show_snapped = not hasattr(self, "_act_snapped_visible") or self._act_snapped_visible.isChecked()
        if self._snapped_points and show_snapped:
            for sp in self._snapped_points:
                ox, oy = sp.original_x, sp.original_y
                sx, sy = sp.snapped_x, sp.snapped_y

                ox_t, oy_t = lm.global_to_preview_f(ox, oy)
                sx_t, sy_t = lm.global_to_preview_f(sx, sy)

                # 原始 → 吸附 虚线
                dash_pen = QPen(QColor(100, 180, 255, 120), 1)
                dash_pen.setCosmetic(True)
                dash_pen.setStyle(Qt.PenStyle.DashLine)
                dash_item = scene.addLine(ox_t, oy_t, sx_t, sy_t, dash_pen)
                dash_item.setZValue(self._canvas.ZVAL_PATH - 1)
                self._task_point_snap_lines.append(dash_item)

                # 吸附点圆点
                r = 5
                if sp.status == "failed":
                    snap_color = QColor(255, 60, 60)       # 失败 = 红
                    snap_brush = QBrush(QColor(255, 60, 60, 80))
                elif sp.status == "warning":
                    snap_color = QColor(255, 180, 40)      # warning = 橙
                    snap_brush = QBrush(QColor(255, 180, 40, 120))
                else:
                    snap_color = QColor(40, 150, 255)      # ok = 蓝
                    snap_brush = QBrush(QColor(40, 150, 255, 150))

                dot = QGraphicsEllipseItem(sx_t - r, sy_t - r, r * 2, r * 2)
                dot.setPen(QPen(snap_color, 2))
                dot.setBrush(snap_brush)
                dot.setZValue(self._canvas.ZVAL_PATH)
                scene.addItem(dot)
                self._task_point_snapped_items.append(dot)

        self._canvas.viewport().update()

    def _render_planned_path_to_scene(self):
        """渲染规划路径（dense_path 折线），不是最终小车航点。"""
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QPen, QPainterPath, QColor, QPolygonF, QFont
        from PySide6.QtWidgets import QGraphicsTextItem
        from roadnet.path_visualization import ordered_task_markers, sample_direction_arrows

        scene = self._canvas.scene()
        if scene is None:
            return
        lm = self._layer_manager

        self._clear_planned_path_items()

        if len(self.planned_path_pixel) < 2:
            return

        visible = self._layer_manager.is_layer_visible("planned_path")
        if not visible:
            return

        points = self.planned_path_pixel
        viz_config = self._param_panel.get_config().get("visualization", {})
        path_width = float(viz_config.get("planned_path_width", 5.0))
        arrow_spacing = float(viz_config.get("arrow_spacing_px", 80.0))
        arrow_size = float(viz_config.get("arrow_size_px", 12.0))

        # 绘制主路径
        path = QPainterPath()
        px, py = lm.global_to_preview_f(points[0][0], points[0][1])
        path.moveTo(px, py)
        for p in points[1:]:
            px, py = lm.global_to_preview_f(p[0], p[1])
            path.lineTo(px, py)

        pen = QPen(QColor(203, 166, 247), path_width)  # 紫色粗线
        pen.setCosmetic(True)
        from PySide6.QtGui import QBrush
        from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsEllipseItem, QGraphicsPolygonItem
        item = QGraphicsPathItem(path)
        item.setPen(pen)
        item.setZValue(getattr(self._canvas, "ZVAL_PATH", 40))
        scene.addItem(item)
        self._planned_path_items.append(item)

        # Label: this polyline is dense_path, not vehicle waypoints
        try:
            label = QGraphicsTextItem("dense_path")
            label.setDefaultTextColor(QColor(203, 166, 247))
            font = QFont("Microsoft YaHei", 10)
            font.setBold(True)
            label.setFont(font)
            lx, ly = lm.global_to_preview_f(points[0][0], points[0][1])
            label.setPos(lx + 8, ly - 22)
            label.setZValue(45)
            scene.addItem(label)
            self._planned_path_items.append(label)
        except Exception:
            pass

        # 沿 planned_path_pixel 顺序绘制稀疏方向箭头。
        for arrow in sample_direction_arrows(points, arrow_spacing, arrow_size):
            gx, gy = arrow["center"]
            sx, sy = lm.global_to_preview_f(gx, gy)
            dx_global, dy_global = arrow["direction"]
            sx2, sy2 = lm.global_to_preview_f(gx + dx_global, gy + dy_global)
            dx, dy = sx2 - sx, sy2 - sy
            norm = max(1e-9, (dx * dx + dy * dy) ** 0.5)
            dx, dy = dx / norm, dy / norm
            nx, ny = -dy, dx
            half = arrow_size * 0.5
            preview_triangle = [
                QPointF(sx + dx * half, sy + dy * half),
                QPointF(sx - dx * half + nx * half * 0.65,
                        sy - dy * half + ny * half * 0.65),
                QPointF(sx - dx * half - nx * half * 0.65,
                        sy - dy * half - ny * half * 0.65),
            ]
            arrow_item = QGraphicsPolygonItem(QPolygonF(preview_triangle))
            arrow_item.setBrush(QBrush(QColor(235, 210, 255)))
            arrow_pen = QPen(QColor(75, 25, 110), 1.2)
            arrow_pen.setCosmetic(True)
            arrow_item.setPen(arrow_pen)
            arrow_item.setZValue(self._canvas.ZVAL_PATH + 2)
            scene.addItem(arrow_item)
            self._planned_path_items.append(arrow_item)

        # 任务点角色和顺序使用 seq 排序，优先显示吸附后的实际行驶位置。
        markers = ordered_task_markers(self._task_points, self.snapped_task_points)
        marker_colors = {
            "start": QColor(40, 200, 90),
            "goal": QColor(245, 70, 70),
            "waypoint": QColor(50, 145, 255),
        }
        for marker in markers:
            sx, sy = lm.global_to_preview_f(marker["x"], marker["y"])
            color = marker_colors[marker["role"]]
            radius = 9 if marker["role"] in ("start", "goal") else 7
            dot = QGraphicsEllipseItem(
                sx - radius, sy - radius, radius * 2, radius * 2
            )
            dot.setBrush(QBrush(color))
            dot_pen = QPen(QColor(255, 255, 255), 2)
            dot_pen.setCosmetic(True)
            dot.setPen(dot_pen)
            dot.setZValue(self._canvas.ZVAL_PATH + 3)
            scene.addItem(dot)
            self._planned_path_items.append(dot)

            label = QGraphicsTextItem()
            label.setHtml(
                "<span style='background-color:rgba(255,255,255,220);"
                "color:#111;padding:2px;font-weight:bold;'>"
                + marker["label"] + "</span>"
            )
            label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            label.setPos(sx + radius + 3, sy - 13)
            label.setZValue(self._canvas.ZVAL_PATH + 4)
            scene.addItem(label)
            self._planned_path_items.append(label)

        self._canvas.viewport().update()

    def _render_sparse_waypoints_to_scene(self):
        """Render final vehicle waypoints only (CSV / repaired), never full dense_path."""
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QColor, QBrush, QFont, QPen, QPolygonF
        from PySide6.QtWidgets import (
            QGraphicsEllipseItem, QGraphicsItem, QGraphicsPolygonItem,
            QGraphicsTextItem,
        )

        self._clear_sparse_waypoint_items()
        if not self.sparse_waypoints:
            return
        show_cfg = True
        try:
            show_cfg = bool(
                self._param_panel.get_config()
                .get("visualization", {})
                .get("show_vehicle_waypoints", True)
            )
        except Exception:
            show_cfg = True
        if not show_cfg:
            return
        lm = self._layer_manager
        if not lm.is_layer_visible("layer_sparse_waypoints"):
            return
        scene = self._canvas.scene()
        if scene is None:
            return

        # spacing_mode / tag → color (直线黄 / 弯道橙 / 路口红蓝 / 任务紫)
        colors = {
            "start": QColor(160, 70, 220),
            "goal": QColor(160, 70, 220),
            "task": QColor(160, 70, 220),
            "task_2m": QColor(160, 70, 220),
            "intersection": QColor(60, 130, 255),
            "junction_2m": QColor(60, 130, 255),
            "sharp_turn": QColor(255, 140, 40),
            "corner": QColor(255, 140, 40),
            "curve": QColor(255, 140, 40),
            "curve_2m": QColor(255, 140, 40),
            "straight": QColor(255, 209, 102),
            "straight_10m": QColor(255, 209, 102),
            "inserted_for_validation": QColor(80, 200, 90),
        }
        z_value = self._canvas.ZVAL_PATH + 10

        def register(item):
            scene.addItem(item)
            lm.add_item("layer_sparse_waypoints", item)
            self._sparse_waypoint_items.append(item)

        # Layer name must match exported CSV kind
        layer_name = getattr(self, "_vwp_waypoint_layer_name", None) or "vehicle_waypoints"
        try:
            first = self.sparse_waypoints[0]
            lx, ly = lm.global_to_preview_f(first["x_pixel"], first["y_pixel"])
            title = QGraphicsTextItem(str(layer_name))
            title.setDefaultTextColor(QColor(255, 209, 102))
            title.setFont(QFont("Microsoft YaHei", 10, QFont.Weight.Bold))
            title.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            title.setPos(lx + 10, ly - 28)
            title.setZValue(z_value + 5)
            register(title)
        except Exception:
            pass

        for first, second in zip(self.sparse_waypoints, self.sparse_waypoints[1:]):
            x1, y1 = lm.global_to_preview_f(first["x_pixel"], first["y_pixel"])
            x2, y2 = lm.global_to_preview_f(second["x_pixel"], second["y_pixel"])
            dx, dy = x2 - x1, y2 - y1
            length = max(1e-9, (dx * dx + dy * dy) ** 0.5)
            ux, uy = dx / length, dy / length
            nx, ny = -uy, ux
            cx, cy = x1 + dx * 0.55, y1 + dy * 0.55
            size = 7.0
            polygon = QPolygonF([
                QPointF(cx + ux * size, cy + uy * size),
                QPointF(cx - ux * size + nx * size * 0.55,
                        cy - uy * size + ny * size * 0.55),
                QPointF(cx - ux * size - nx * size * 0.55,
                        cy - uy * size - ny * size * 0.55),
            ])
            arrow = QGraphicsPolygonItem(polygon)
            arrow.setBrush(QBrush(QColor(255, 235, 150)))
            pen = QPen(QColor(85, 55, 10), 1.0)
            pen.setCosmetic(True)
            arrow.setPen(pen)
            arrow.setZValue(z_value)
            register(arrow)

        label_every = 5 if len(self.sparse_waypoints) <= 80 else 10
        for waypoint in self.sparse_waypoints:
            x, y = lm.global_to_preview_f(waypoint["x_pixel"], waypoint["y_pixel"])
            mode = str(waypoint.get("spacing_mode") or "")
            tag = str(waypoint.get("tag", "straight"))
            color = colors.get(mode) or colors.get(tag, colors["straight"])
            near_task = bool(waypoint.get("near_task_point") or tag in {"start", "goal", "task"})
            radius = 7.5 if near_task or tag == "intersection" else 5.0
            if near_task:
                color = colors["task_2m"]
            dot = QGraphicsEllipseItem(-radius, -radius, radius * 2, radius * 2)
            dot.setPos(x, y)
            dot.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            outline = QPen(QColor(25, 25, 25), 1.5)
            outline.setCosmetic(True)
            dot.setPen(outline)
            dot.setBrush(QBrush(color))
            dot.setZValue(z_value + 1)
            register(dot)

            seq = int(waypoint.get("seq", 0))
            if seq == 1 or seq == len(self.sparse_waypoints) or seq % label_every == 0:
                label = QGraphicsTextItem(str(seq))
                label.setDefaultTextColor(QColor(255, 255, 255))
                label.setFont(QFont("Arial", 8, QFont.Weight.Bold))
                label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                label.setPos(x + radius + 2, y - radius - 7)
                label.setZValue(z_value + 2)
                register(label)
        self._canvas.viewport().update()

    def _clear_sparse_waypoint_items(self):
        scene = self._canvas.scene() if getattr(self, "_canvas", None) else None
        if scene is not None:
            for item in getattr(self, "_sparse_waypoint_items", []):
                self._layer_manager.remove_item("layer_sparse_waypoints", item)
                safe_remove_scene_item(scene, item)
        if hasattr(self, "_sparse_waypoint_items"):
            self._sparse_waypoint_items.clear()

    def _clear_waypoint_validation_items(self):
        scene = self._canvas.scene() if getattr(self, "_canvas", None) else None
        if scene is not None:
            for item in getattr(self, "_waypoint_validation_items", []):
                self._layer_manager.remove_item("layer_waypoint_validation", item)
                safe_remove_scene_item(scene, item)
        if hasattr(self, "_waypoint_validation_items"):
            self._waypoint_validation_items.clear()

    def _render_waypoint_validation_to_scene(self):
        """显示异常段（红色粗线）与异常点编号。"""
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QColor, QFont, QPen
        from PySide6.QtWidgets import (
            QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem, QGraphicsTextItem,
        )

        self._clear_waypoint_validation_items()
        bad = list(getattr(self, "_waypoint_bad_segments", None) or [])
        if not bad or not self.sparse_waypoints:
            return
        lm = self._layer_manager
        if not lm.is_layer_visible("layer_waypoint_validation"):
            return
        scene = self._canvas.scene()
        if scene is None:
            return

        by_name = {wp.get("name"): wp for wp in self.sparse_waypoints}
        z_value = self._canvas.ZVAL_PATH + 20

        def register(item):
            scene.addItem(item)
            lm.add_item("layer_waypoint_validation", item)
            self._waypoint_validation_items.append(item)

        for seg in bad:
            a = by_name.get(seg.get("from_wp"))
            b = by_name.get(seg.get("to_wp"))
            if not a or not b:
                continue
            x1, y1 = lm.global_to_preview_f(a["x_pixel"], a["y_pixel"])
            x2, y2 = lm.global_to_preview_f(b["x_pixel"], b["y_pixel"])
            line = QGraphicsLineItem(x1, y1, x2, y2)
            pen = QPen(QColor(255, 40, 40), 4.0)
            pen.setCosmetic(True)
            line.setPen(pen)
            line.setZValue(z_value)
            register(line)

            for wp in (a, b):
                x, y = lm.global_to_preview_f(wp["x_pixel"], wp["y_pixel"])
                ring = QGraphicsEllipseItem(-9, -9, 18, 18)
                ring.setPos(x, y)
                ring.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                rpen = QPen(QColor(255, 30, 30), 2.0)
                rpen.setCosmetic(True)
                ring.setPen(rpen)
                ring.setBrush(Qt.BrushStyle.NoBrush)
                ring.setZValue(z_value + 1)
                register(ring)
                label = QGraphicsTextItem(
                    f"{wp.get('seq')}:{(seg.get('reason') or '')[:24]}"
                )
                label.setDefaultTextColor(QColor(255, 120, 120))
                label.setFont(QFont("Arial", 8, QFont.Weight.Bold))
                label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                label.setPos(x + 10, y + 4)
                label.setZValue(z_value + 2)
                register(label)
        self._canvas.viewport().update()

    def _clear_task_point_items(self):
        """清除任务点相关 scene items"""
        scene = self._canvas.scene() if self._canvas else None
        if scene is None:
            return
        for item in self._task_point_original_items:
            safe_remove_scene_item(scene, item)
        for item in self._task_point_snapped_items:
            safe_remove_scene_item(scene, item)
        for item in self._task_point_snap_lines:
            safe_remove_scene_item(scene, item)
        self._task_point_original_items.clear()
        self._task_point_snapped_items.clear()
        self._task_point_snap_lines.clear()

    def _serialize_task_points(self) -> list:
        """将 TaskPoint 列表序列化为可保存的字典列表"""
        result = []
        for tp in self._task_points:
            result.append({
                "seq": tp.seq,
                "longitude": tp.longitude,
                "latitude": tp.latitude,
                "altitude": tp.altitude,
                "point_type": tp.point_type,
                "reserve": tp.reserve,
                "pixel_x": tp.pixel_x,
                "pixel_y": tp.pixel_y,
                "map_x": tp.map_x,
                "map_y": tp.map_y,
                "status": getattr(tp, "status", "pending"),
                "inside_image": getattr(tp, "inside_image", None),
                "created_order": getattr(tp, "created_order", 0),
                "source": getattr(tp, "source", ""),
                "snap_status": getattr(tp, "snap_status", ""),
                "snap_distance": getattr(tp, "snap_distance", None),
            })
        return result

    def _deserialize_task_points(self, data: list):
        """从序列化数据恢复 TaskPoint 列表"""
        from roadnet.task_points import TaskPoint
        self._task_points = []
        for index, d in enumerate(data):
            tp = TaskPoint(
                seq=d.get("seq", 0),
                longitude=d.get("longitude", 0.0),
                latitude=d.get("latitude", 0.0),
                altitude=d.get("altitude", 0.0),
                point_type=d.get("point_type", 2),
                reserve=d.get("reserve", ""),
                pixel_x=d.get("pixel_x"),
                pixel_y=d.get("pixel_y"),
                map_x=d.get("map_x"),
                map_y=d.get("map_y"),
                status=d.get("status", "pending"),
                inside_image=d.get("inside_image"),
                created_order=d.get("created_order", index),
                source=d.get("source", ""),
                snap_status=d.get("snap_status", ""),
                snap_distance=d.get("snap_distance"),
            )
            self._task_points.append(tp)
        self._normalize_task_points("project_load")
        self._sync_task_points_table()
        print(f"[TaskPoints] 从项目恢复 {len(self._task_points)} 个任务点")

    def _clear_planned_path_items(self):
        """清除规划路径 scene items"""
        scene = self._canvas.scene() if self._canvas else None
        if scene is None:
            return
        for item in self._planned_path_items:
            safe_remove_scene_item(scene, item)
        self._planned_path_items.clear()

    # ===================================================================

    def _on_run_skeleton(self):
        # ★ 大图：走主路约束清理流水线，禁止直接对脏 mask skeletonize
        if self._layer_manager.is_large_image_mode:
            self._on_run_large_skeleton()
            return
        mask, mask_meta = self.get_current_mask_array(prefer_processed=True)
        if mask is None:
            err_msg = mask_meta.get("error", "请先加载或生成正式 Road Mask。")
            QMessageBox.warning(self, "提示",
                f"无法生成骨架：{err_msg}\n\n"
                "请确保已通过正式 tile 分割生成 global_road_mask.png。")
            return
        err = self._skeleton_input_check(mask, mask_meta)
        if err is not None:
            QMessageBox.warning(self, "骨架生成", err)
            return
        mask = self._apply_valid_area(mask)
        try:
            self._history.push_state("generate_skeleton")
            from roadnet.optimized_skeleton import skeletonize_medial_axis, skeletonize_thin
            config = self._param_panel.get_config()
            skel_cfg = config.get("skeleton", {})
            method = skel_cfg.get("method", "medial_axis")
            if method == "medial_axis":
                raw_skeleton = skeletonize_medial_axis(mask)
            else:
                raw_skeleton = skeletonize_thin(mask)
            raw_pixels = int((raw_skeleton > 0).sum())
            self._commit_raw_skeleton(raw_skeleton)
            self._layer_manager.show_layer("skeleton")
            self._act_skeleton_visible.setChecked(True)
            self._sync_layer_checkboxes()
            self._canvas.refresh_scene()
            self._canvas.viewport().update()
            msg = f"Skeleton 生成完成: {raw_pixels} 像素"
            self._status_bar.show_message(msg)
        except Exception as e:
            self._log_skeleton_error(mask_meta, str(e))
            QMessageBox.critical(self, "骨架生成",
                f"骨架生成失败：当前 Road Mask 图层不是正式二值 mask，而是 metadata dict。\n"
                f"请检查是否已生成 global_road_mask.png 或 processed_global_mask.png。\n\n"
                f"详细错误:\n{e}")

    def _collect_large_skeleton_constraints(self):
        """收集大图骨架主路约束（种子线 / ROI / 任务点 / Ignore）。"""
        rois, ignores, tasks = self._collect_roi_ignore_task_original()
        seeds = []
        if hasattr(self._canvas, "get_main_road_seed_strokes"):
            seeds = self._canvas.get_main_road_seed_strokes() or []
        return rois, ignores, tasks, seeds

    def _sync_layer_mask_to_disk_for_skeleton(self) -> Optional[str]:
        """骨架前：若图层有 full-size mask，把最新内容同步到 working/final 磁盘文件。

        解决「画布已改 / 已保存到别的路径，但骨架仍读旧 final」的问题。
        """
        if not self._layer_manager.is_large_image_mode:
            return None
        if self._large_image_project is None:
            return None
        layer = self._layer_manager.get_layer_data("mask")
        if not isinstance(layer, np.ndarray) or layer.size == 0:
            return None
        ow, oh = self._layer_manager.original_size
        if ow <= 0 or oh <= 0:
            return None
        h, w = layer.shape[:2]
        if (w, h) != (ow, oh):
            # 图层不是原图像素尺寸，绝不能覆盖正式 mask
            print(
                f"[SkeletonSync] 跳过同步：layer={w}x{h} != original={ow}x{oh}"
            )
            return None
        dirty = bool(getattr(self, "_working_mask_dirty", False))
        # dirty 或尚无 final 时，强制把当前图层写成 final_edited_mask
        need_final = dirty or not (
            getattr(self, "_final_edited_mask_path", None)
            and os.path.isfile(self._final_edited_mask_path)
        )
        if not need_final:
            # 即便已保存，也核对磁盘与图层是否一致（防止漏写）
            try:
                disk = cv2.imread(self._final_edited_mask_path, cv2.IMREAD_GRAYSCALE)
                if disk is not None and disk.shape == layer.shape:
                    if int(np.count_nonzero(disk)) == int(np.count_nonzero(layer)):
                        return self._final_edited_mask_path
            except Exception:
                pass
            need_final = True
        try:
            path = self._persist_working_mask(
                (layer > 0).astype(np.uint8) * 255,
                save_as_final=True,
            )
            self._working_mask_dirty = False
            print(f"[SkeletonSync] 已同步图层 mask → {path}")
            return path
        except Exception as exc:
            print(f"[SkeletonSync] 同步失败: {exc}")
            return None

    def _on_run_large_skeleton(self):
        """大图专用：严格从 full-size 文件选 mask，再生成骨架（不读图层/preview）。"""
        import time
        t0 = time.time()

        # ★ 先把当前图层正式 mask 同步到磁盘，避免骨架读到旧 final
        self._sync_layer_mask_to_disk_for_skeleton()

        mask, mask_meta = self.get_skeleton_input_mask_large()
        diag = self._format_skeleton_input_diagnostics(mask_meta)
        print(f"[LargeSkeleton]\n{diag}")

        if mask is None:
            QMessageBox.warning(
                self, "大图骨架",
                f"无法生成骨架：\n{mask_meta.get('error') or '未找到有效 mask'}\n\n{diag}",
            )
            return

        # 弹窗确认输入，避免误用旧 global
        confirm = QMessageBox(self)
        confirm.setWindowTitle("骨架输入 Mask 确认")
        confirm.setIcon(QMessageBox.Icon.Information)
        confirm.setText(
            f"将使用以下 mask 生成骨架：\n\n"
            f"{os.path.basename(mask_meta.get('selected_mask_path') or '')}\n"
            f"mask_source = {mask_meta.get('mask_source')}\n"
            f"mask_edit_base = {mask_meta.get('mask_edit_base')}\n"
            f"shape = {mask_meta.get('mask_shape')}\n"
            f"nonzero_ratio = {mask_meta.get('nonzero_ratio')}\n"
            f"file_modified_time = {mask_meta.get('file_modified_time')}\n\n"
            "正式保存的 mask 将作为输入；"
            "无种子/ROI 时不再砍成最长 3 段。"
        )
        confirm.setDetailedText(diag)
        confirm.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        if confirm.exec() != QMessageBox.StandardButton.Ok:
            return

        src = mask_meta.get("mask_source") or ""
        dirty = bool(getattr(self, "_working_mask_dirty", False))
        if dirty:
            warn = QMessageBox.warning(
                self, "Mask 未保存",
                "当前 Mask 有未保存的手动修改。\n"
                "已尝试同步图层到 final_edited_mask；若仍不匹配请先点「保存当前 Mask」。\n\n"
                "是否仍继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if warn != QMessageBox.StandardButton.Yes:
                return

        if src == "final_edited_mask":
            self._status_bar.show_message("正在使用 final_edited_mask 生成骨架")
        elif src == "cleaned_working_mask":
            self._status_bar.show_message("正在使用 cleaned_working_mask 生成骨架（中间结果）")
        elif src in ("working_road_mask", "edited_global_road_mask"):
            self._status_bar.show_message("正在使用 working_road_mask 生成骨架")
        elif src == "refined_main_road_mask":
            self._status_bar.show_message("正在使用 refined_main_road_mask 生成骨架")
        else:
            self._status_bar.show_message(
                f"正在使用 {src or 'global_road_mask'} 生成骨架"
                "（建议先清理并保存 final_edited_mask）"
            )
        QApplication.processEvents()

        if self._large_image_project is not None:
            output_dir = os.path.join(self._large_image_project.project_dir, "skeleton")
        else:
            output_dir = os.path.join(os.getcwd(), "outputs", "skeleton")
        os.makedirs(output_dir, exist_ok=True)

        cache_info = self._invalidate_large_skeleton_cache_if_needed(output_dir, mask_meta)

        # 备份以便回滚
        self._skeleton_backup_raw = (
            None if self.raw_skeleton is None else self.raw_skeleton.copy()
        )
        self._skeleton_backup_opt = (
            None if self.optimized_skeleton is None else self.optimized_skeleton.copy()
        )
        self._skeleton_backup_state = self.skeleton_state

        mask = self._apply_valid_area(mask)
        try:
            self._history.push_state("large_skeleton_generate")
            from roadnet.large_skeleton_optimizer import generate_large_clean_skeleton

            rois, ignores, tasks, seeds = self._collect_large_skeleton_constraints()
            self._status_bar.show_message(
                "大图骨架：mask_preclean → 中心线过滤 → graph 剪枝 → 保守桥接…"
            )
            QApplication.processEvents()

            # ★ 用户已保存/整理的正式 mask：信任输入，勿再砍成 top-3
            skel_cfg = {}
            trusted_sources = {
                "final_edited_mask",
                "working_road_mask",
                "cleaned_working_mask",
                "lowres_formal_mask",
                "manual_edited",
                "manual_after_cleaned",
            }
            if src in trusted_sources or (mask_meta.get("mask_edit_base") or "") in trusted_sources:
                skel_cfg = {
                    "trust_input_mask": True,
                    "keep_top_k_without_constraints": 999,
                    "keep_top_k_skel_without_constraints": 999,
                }

            cleaned_skel, graph, pack = generate_large_clean_skeleton(
                mask,
                roi_polygons=rois or None,
                ignore_polygons=ignores or None,
                main_road_seed_strokes=seeds or None,
                task_points=tasks or None,
                config=skel_cfg or None,
                output_dir=output_dir,
                input_meta=mask_meta,
            )
            raw_skel = pack.get("raw_skeleton")
            center_skel = pack.get("center_filtered_skeleton")
            if raw_skel is None:
                raw_skel = cleaned_skel
            if center_skel is None:
                center_skel = cleaned_skel
            cleaned_skel = self._apply_valid_area(cleaned_skel)
            raw_skel = self._apply_valid_area(raw_skel)
            center_skel = self._apply_valid_area(center_skel)
            pipeline = "generate_large_clean_skeleton"
            skel_report = pack.get("report") or {}

            ph = self._layer_manager.preview_height
            pw = self._layer_manager.preview_width
            cleaned_preview = None
            raw_preview = None
            center_preview = None
            stages = pack.get("stages") or {}
            if stages.get("pruned_skeleton_preview") is not None:
                cleaned_preview = stages["pruned_skeleton_preview"]
            if stages.get("raw_skeleton_preview") is not None:
                raw_preview = stages["raw_skeleton_preview"]
            if stages.get("center_filtered_skeleton_preview") is not None:
                center_preview = stages["center_filtered_skeleton_preview"]
            if pw > 0 and ph > 0:
                if cleaned_preview is None:
                    cleaned_preview = cv2.resize(
                        cleaned_skel, (pw, ph), interpolation=cv2.INTER_NEAREST,
                    )
                if raw_preview is None:
                    raw_preview = cv2.resize(
                        raw_skel, (pw, ph), interpolation=cv2.INTER_NEAREST,
                    )
                if center_preview is None:
                    center_preview = cv2.resize(
                        center_skel, (pw, ph), interpolation=cv2.INTER_NEAREST,
                    )

            self._commit_optimized_skeleton(
                raw_skel, cleaned_skel,
                result={
                    "mask_source": src,
                    "pipeline": pipeline,
                    "report": skel_report,
                },
                preview_data=cleaned_preview,
            )
            self._layer_manager.set_layer_data(
                "layer_raw_skeleton", raw_skel, preview_data=raw_preview,
            )
            self._layer_manager.set_layer_data(
                "layer_center_filtered_skeleton", center_skel,
                preview_data=center_preview,
            )
            self._layer_manager.show_layer("skeleton")
            self._layer_manager.hide_layer("layer_raw_skeleton")
            self._layer_manager.hide_layer("layer_center_filtered_skeleton")
            show_raw = False
            if hasattr(self._param_panel, "large_skeleton_show_raw"):
                show_raw = bool(self._param_panel.large_skeleton_show_raw())
            if show_raw:
                self._layer_manager.show_layer("layer_raw_skeleton")
            self._act_skeleton_visible.setChecked(True)

            nodes = (graph or {}).get("nodes") or []
            edges = (graph or {}).get("edges") or []
            if self._graph_editor is not None and (nodes or edges):
                try:
                    self._graph_editor.load_draft(nodes, edges)
                    self._layer_manager.show_layer("layer_draft_graph")
                    self._layer_manager.show_layer("layer_final_graph")
                except Exception as ge_exc:
                    print(f"[LargeSkeleton] 加载 graph 失败: {ge_exc}")

            self._sync_layer_checkboxes()
            self._canvas.refresh_scene()
            self._canvas.viewport().update()
            self.mark_stage_done("skeleton")

            elapsed = time.time() - t0
            report_path = self._write_large_skeleton_generation_report(
                output_dir, mask_meta, cache_info, raw_skel, cleaned_skel, elapsed,
                extra={
                    "pipeline": pipeline,
                    **{k: v for k, v in skel_report.items()
                       if k not in ("centerline_filter", "skeleton_component_filter")},
                },
            )
            opt_report = (pack.get("saved_files") or {}).get("large_skeleton_report.json")
            if hasattr(self._param_panel, "update_skeleton_stats"):
                self._param_panel.update_skeleton_stats({
                    "raw_pixels": int(np.count_nonzero(raw_skel)),
                    "optimized_pixels": int(np.count_nonzero(cleaned_skel)),
                    "center_filtered_pixels": int(np.count_nonzero(center_skel)),
                    "graph_nodes": len(nodes),
                    "graph_edges": len(edges),
                })

            msg = (
                f"大图骨架完成（{pipeline}）\n"
                f"输入：{os.path.basename(mask_meta.get('selected_mask_path') or '')}\n"
                f"mask_source = {src}\n"
                f"raw 像素 = {int(np.count_nonzero(raw_skel))}\n"
                f"center-filtered 像素 = {int(np.count_nonzero(center_skel))}\n"
                f"cleaned 像素 = {int(np.count_nonzero(cleaned_skel))}\n"
                f"graph = {len(nodes)} nodes / {len(edges)} edges\n"
                f"报告：{opt_report or report_path}"
            )
            self._status_bar.show_message(
                f"骨架完成：使用 {src}（{os.path.basename(mask_meta.get('selected_mask_path') or '')}）"
            )
            QMessageBox.information(self, "大图骨架完成", msg)
        except Exception as e:
            self._log_skeleton_error(mask_meta, str(e))
            QMessageBox.critical(self, "大图骨架失败", f"骨架生成失败：\n{e}\n\n{diag}")

    def _on_view_skeleton_input_mask(self):
        """查看当前将用于骨架生成的 full-size mask（诊断）。"""
        if not self._layer_manager.is_large_image_mode:
            QMessageBox.information(self, "小图模式", "该功能仅用于大图骨架输入诊断。")
            return
        mask, meta = self.get_skeleton_input_mask_large()
        diag = self._format_skeleton_input_diagnostics(meta)
        print(f"[SkeletonInputView]\n{diag}")
        if mask is None:
            QMessageBox.warning(self, "骨架输入 Mask", diag)
            return

        # 生成 preview 弹窗
        pw = self._layer_manager.preview_width or 800
        ph = self._layer_manager.preview_height or 600
        preview = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST)
        if self._large_image_project is not None:
            out = Path(self._large_image_project.project_dir) / "skeleton"
        else:
            out = Path(os.getcwd()) / "outputs" / "skeleton"
        out.mkdir(parents=True, exist_ok=True)
        preview_path = out / "skeleton_input_mask_view_preview.png"
        cv2.imwrite(str(preview_path), preview)
        self._show_image_dialog(
            str(preview_path),
            "骨架输入 Mask",
            (
                f"path={os.path.basename(meta.get('selected_mask_path') or '')} | "
                f"source={meta.get('mask_source')} | "
                f"nonzero={meta.get('nonzero_ratio')} | "
                f"mtime={meta.get('file_modified_time')} | "
                f"hash={meta.get('checksum')}"
            ),
        )
        self._status_bar.show_message(
            f"骨架输入：{meta.get('mask_source')} → "
            f"{os.path.basename(meta.get('selected_mask_path') or '')}"
        )

    def _on_large_skeleton_show_raw(self):
        if not self._layer_manager.is_large_image_mode:
            return
        self._layer_manager.show_layer("layer_raw_skeleton")
        self._layer_manager.hide_layer("skeleton")
        self._layer_manager.hide_layer("layer_center_filtered_skeleton")
        self._sync_layer_checkboxes()
        self._canvas.refresh_scene()
        self._status_bar.show_message("已显示 Raw Skeleton（噪声较多）")

    def _on_large_skeleton_show_cleaned(self):
        if not self._layer_manager.is_large_image_mode:
            return
        self._layer_manager.show_layer("skeleton")
        self._layer_manager.hide_layer("layer_raw_skeleton")
        self._layer_manager.hide_layer("layer_center_filtered_skeleton")
        self._sync_layer_checkboxes()
        self._canvas.refresh_scene()
        self._status_bar.show_message("已显示 Cleaned Skeleton")

    def _on_view_skeleton_bridges(self):
        if self._large_image_project is None:
            QMessageBox.information(self, "桥接候选", "当前没有大图项目。")
            return
        path = os.path.join(
            self._large_image_project.project_dir, "skeleton",
            "bridge_candidates_overlay.png",
        )
        if not os.path.isfile(path):
            QMessageBox.information(
                self, "桥接候选",
                "尚未生成 bridge_candidates_overlay.png，请先执行大图骨架生成。",
            )
            return
        self._show_image_dialog(path, "桥接候选", "绿=接受 红=拒绝 黄=待确认")

    def _on_accept_skeleton_result(self):
        self._skeleton_backup_raw = None
        self._skeleton_backup_opt = None
        self._status_bar.show_message("已接受当前骨架结果")

    def _on_rollback_skeleton_result(self):
        raw = getattr(self, "_skeleton_backup_raw", None)
        opt = getattr(self, "_skeleton_backup_opt", None)
        if raw is None and opt is None:
            QMessageBox.information(self, "回滚骨架", "没有可回滚的骨架备份。")
            return
        if opt is not None:
            self._commit_optimized_skeleton(raw if raw is not None else opt, opt)
        elif raw is not None:
            self._commit_raw_skeleton(raw)
        self._layer_manager.show_layer("skeleton")
        self._sync_layer_checkboxes()
        self._canvas.refresh_scene()
        self._status_bar.show_message("已回滚骨架结果")

    def _on_run_optimize(self):
        mask, mask_meta = self.get_current_mask_array(prefer_processed=True)
        if mask is None:
            err_msg = mask_meta.get("error", "请先加载或生成正式 Road Mask。")
            QMessageBox.warning(self, "提示",
                f"无法优化骨架：{err_msg}")
            return
        err = self._skeleton_input_check(mask, mask_meta)
        if err is not None:
            QMessageBox.warning(self, "骨架优化", err)
            return
        if self.skeleton_state == "none":
            QMessageBox.information(self, "请先生成骨架", "当前没有 raw_skeleton，请先点击「生成骨架」。")
            return
        if self.skeleton_state == "optimized":
            reply = QMessageBox.question(
                self,
                "骨架已经优化",
                "当前骨架已经优化过。是否从 raw_skeleton 重新优化？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        mask = self._apply_valid_area(mask)
        try:
            self._history.push_state("optimize_skeleton")
            raw_skeleton = self.raw_skeleton
            if raw_skeleton is None:
                raw_skeleton = self._generate_raw_skeleton(mask)
                self._commit_raw_skeleton(raw_skeleton)
            raw_skeleton = self._get_skeleton_array(raw_skeleton)
            from roadnet.optimized_skeleton import optimize_skeleton
            from roadnet.skeleton_artifacts import build_skeleton_optimize_report
            config = self._param_panel.get_config()
            skel_cfg = config.get("skeleton", {})
            result = optimize_skeleton(
                mask, raw_skeleton,
                min_center_dist=skel_cfg.get("min_center_dist", 2.0),
                border_margin=skel_cfg.get("border_margin", 10),
                min_branch_length=skel_cfg.get("min_branch_length", 20),
                max_connect_dist=skel_cfg.get("max_connect_dist", 25),
                max_connect_angle=skel_cfg.get("max_connect_angle", 45),
                min_line_mask_overlap=skel_cfg.get("min_line_mask_overlap", 0.65),
                junction_cluster_radius=skel_cfg.get("junction_cluster_radius", 10),
            )
            result["optimized_skeleton"] = self._apply_valid_area(
                result["optimized_skeleton"]
            )
            report = build_skeleton_optimize_report(
                mask,
                raw_skeleton,
                result["optimized_skeleton"],
                min_branch_length=skel_cfg.get("min_branch_length", 20),
                min_center_dist=result.get("stats", {}).get(
                    "effective_min_center_dist", skel_cfg.get("min_center_dist", 2.0)
                ),
                endpoint_connect_distance=skel_cfg.get("max_connect_dist", 25),
                skeleton_state_input="raw",
            )
            result["skeleton_optimize_report"] = report
            self._save_manual_skeleton_artifacts(
                mask, raw_skeleton, result["optimized_skeleton"], report
            )
            self._commit_optimized_skeleton(
                raw_skeleton, result["optimized_skeleton"], result
            )
            self._canvas.refresh_scene()
            self._canvas.viewport().update()
            stats = result["stats"]
            self._param_panel.update_skeleton_stats(stats)

            self._status_bar.show_message(
                f"骨架优化完成: {stats['optimized_pixels']}px, "
                f"{stats['optimized_endpoints']}端点, {stats['junction_cluster_count']}路口")
            if report["removed_ratio"] > 0.60:
                QMessageBox.warning(
                    self,
                    "骨架可能过度剪枝",
                    "骨架优化删除比例过高，可能发生过度剪枝，请降低 "
                    "min_branch_length 或从 raw_skeleton 重新生成。",
                )
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"骨架优化失败:\n{e}")

    def _get_skeleton_array(self, skeleton_data):
        if skeleton_data is None:
            raise ValueError("skeleton_data is None")
        if isinstance(skeleton_data, dict):
            for key in ["current_skeleton", "optimized_skeleton", "skeleton_optimized", "skeleton", "raw_skeleton"]:
                if key in skeleton_data:
                    skeleton = skeleton_data[key]
                    break
            else:
                raise ValueError(f"skeleton_data dict has no valid key: {list(skeleton_data.keys())}")
        else:
            skeleton = skeleton_data
        if skeleton.ndim == 3:
            skeleton = skeleton[:, :, 0]
        return (skeleton > 0).astype(np.uint8) * 255

    def _on_run_graph(self):
        """生成草稿路网（使用 graph_build 统一流水线）。"""
        skeleton = self._layer_manager.get_layer_data("skeleton")
        if skeleton is None:
            QMessageBox.warning(self, "提示", "请先生成 Skeleton。")
            return
        try:
            self._history.push_state("generate_draft_graph")
            skeleton_arr = self._get_skeleton_array(skeleton)
            skeleton_arr = self._apply_valid_area(skeleton_arr)
            outputs_dir = os.path.join(os.getcwd(), "outputs", "graph_build")
            config = self._param_panel.get_config()
            graph_cfg = config.get("graph", {})

            # ★ 使用 graph_build 统一流水线
            from roadnet.graph_build import build_graph_from_skeleton

            result = build_graph_from_skeleton(
                skeleton=skeleton_arr,
                graph_editor=self._graph_editor,
                output_dir=outputs_dir,
                config=config,
                processed_mask=self._apply_valid_area(
                    self._layer_manager.get_layer_data("mask")
                ),
                run_optimization=graph_cfg.get("enable_graph_line_optimizer", False),
            )

            if not result.success:
                err_msgs = "\n".join(result.errors)
                QMessageBox.critical(
                    self, "Graph 生成失败",
                    f"在阶段 [{result.stage}] 失败:\n{err_msgs}"
                )
                self._status_bar.show_message(f"Graph 生成失败: {result.stage}")
                return

            # 渲染
            self._clear_graph_repair_items()
            self._graph_repair_candidates = []
            self._graph_repair_signature = None
            self._render_graph_to_scene()
            self._update_graph_stats()

            self._status_bar.update_nodes(len(result.raw_nodes))
            self._status_bar.update_edges(len(result.raw_edges))
            self._status_bar.show_message(
                f"草稿路网生成完成: {len(result.raw_nodes)} 节点, {len(result.raw_edges)} 边"
            )
            self.mark_stage_done("skeleton")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"Graph 生成异常:\n{e}")

    def _on_run_pipeline(self):
        """Run the existing mask-to-graph pipeline without blocking Qt."""
        if self._pipeline_thread is not None and self._pipeline_thread.isRunning():
            self._cancel_pipeline()
            return
        if self._segmentation_thread is not None and self._segmentation_thread.isRunning():
            QMessageBox.information(self, "分割正在运行", "请先完成或取消分割任务。")
            return

        # ★ 使用 get_current_mask_array 统一解析，兼容 dict/ndarray
        mask, mask_meta = self.get_current_mask_array(prefer_processed=True,
                                                       for_skeleton=False)
        if mask is None:
            image_path = self._layer_manager.image_path
            if not image_path or not os.path.exists(image_path):
                QMessageBox.warning(self, "提示", "请先打开影像。")
                return
            QMessageBox.information(
                self,
                "一键生成路网",
                "当前尚无 Road Mask，将先运行 SAM-Road。大图会自动使用后台 tile 推理；"
                "Mask 导入成功后将继续执行后处理、骨架和 Graph 构建。",
            )
            self._on_run_samroad_single_extract(
                force_tile=True, continue_pipeline=True
            )
            return

        from roadnet.pipeline_worker import PipelineWorker

        thread = QThread(self)
        worker = PipelineWorker(
            mask=mask,
            config=self._param_panel.get_config(),
            output_root=(
                self._large_image_project.project_dir
                if self._large_image_project is not None
                else os.path.join(os.getcwd(), "outputs")
            ),
            valid_image_mask=self._ensure_valid_image_mask(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_pipeline_progress)
        worker.finished.connect(self._on_pipeline_finished)
        worker.failed.connect(self._on_pipeline_failed)
        worker.cancelled.connect(self._on_pipeline_cancelled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(self._on_pipeline_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._pipeline_thread = thread
        self._pipeline_worker = worker
        self._pipeline_source_mask = mask
        self._btn_pipeline.setText("取消一键流程")
        dialog = QProgressDialog(
            "正在启动后台流程…", "取消一键流程", 0, 100, self
        )
        dialog.setWindowTitle("一键生成路网")
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumDuration(0)
        dialog.canceled.connect(self._cancel_pipeline)
        self._pipeline_progress_dialog = dialog
        dialog.show()
        self._status_bar.show_message("一键生成路网已转入后台执行")
        thread.start()

    def _cancel_pipeline(self):
        if self._pipeline_worker is None:
            return
        self._pipeline_worker.cancel()
        if self._pipeline_thread is not None:
            self._pipeline_thread.requestInterruption()
        self._status_bar.show_message("正在取消一键流程…")
        if self._pipeline_progress_dialog is not None:
            self._pipeline_progress_dialog.setLabelText("正在取消；当前图层保持不变…")

    def _on_pipeline_progress(self, percent, message):
        if self._pipeline_progress_dialog is not None:
            self._pipeline_progress_dialog.setValue(int(percent))
            self._pipeline_progress_dialog.setLabelText(message)
        self._status_bar.show_message(message)

    def _on_pipeline_finished(self, result):
        # Atomic GUI commit: no partial mask/skeleton/graph was shown while the
        # worker ran, and graph_editor is only touched here on the GUI thread.
        if self._layer_manager.get_layer_data("mask") is not self._pipeline_source_mask:
            self._status_bar.show_message("mask 已改变，已丢弃旧 mask 的一键流程结果")
            return
        if not self._layer_manager.is_large_image_mode:
            self._history.push_state("pipeline_complete")
        self._layer_manager.set_layer_data(
            "mask", result.processed_mask,
            preview_data=getattr(result, "mask_preview", None),
        )
        self._commit_optimized_skeleton(
            result.raw_skeleton, result.optimized_skeleton,
            {"skeleton_optimize_report": result.skeleton_report},
            preview_data=getattr(result, "skeleton_preview", None),
        )
        self._graph_editor.load_draft(result.nodes, result.edges)
        self._clear_graph_repair_items()
        self._graph_repair_candidates = []
        self._graph_repair_signature = None
        self._canvas.refresh_scene()
        self._render_graph_to_scene()
        self._update_graph_stats()
        if self._large_image_project is not None and self._layer_manager.is_large_image_mode:
            raw_graph_path = os.path.join(
                self._large_image_project.project_dir,
                "graph_build", "final_graph_raw.json",
            )
            if os.path.isfile(raw_graph_path):
                self._large_image_project.global_graph_path = raw_graph_path
                self._project_manager.data.global_graph_path = raw_graph_path
            self._large_image_project.save()
            self._project_manager.mark_dirty()
        self.mark_stage_done("edit")
        self.mark_stage_done("skeleton")
        self.set_stage("graph")
        components = result.connectivity.get("connected_components", 0)
        self._status_bar.show_message(
            f"一键流程完成: {len(result.nodes)} 节点, {len(result.edges)} 边, "
            f"{components} 个连通分量"
        )
        if result.skeleton_report.get("removed_ratio", 0.0) > 0.60:
            QMessageBox.warning(
                self,
                "骨架可能过度剪枝",
                "骨架优化删除比例过高，可能发生过度剪枝，请降低 "
                "min_branch_length 或从 raw_skeleton 重新生成。",
            )

    def _on_pipeline_failed(self, stage, message, details):
        error_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(error_dir, exist_ok=True)
        error_path = os.path.join(error_dir, "pipeline_error.log")
        with open(error_path, "w", encoding="utf-8") as handle:
            handle.write(f"stage = {stage}\nerror = {message}\n\n{details}")
        self._status_bar.show_message(f"一键流程失败于 {stage}；原图层未改变")
        QMessageBox.critical(
            self, "一键流程失败",
            f"失败阶段: {stage}\n{message}\n\n原 mask/skeleton/graph 未改变。\n"
            f"日志: {error_path}"
        )

    def _on_pipeline_cancelled(self, message):
        self._status_bar.show_message("一键流程已取消；原图层未改变")

    def _on_pipeline_thread_finished(self):
        self._btn_pipeline.setText("一键生成路网")
        if self._pipeline_progress_dialog is not None:
            self._pipeline_progress_dialog.close()
            self._pipeline_progress_dialog.deleteLater()
        self._pipeline_progress_dialog = None
        self._pipeline_worker = None
        self._pipeline_thread = None
        self._pipeline_source_mask = None

    def _on_run_pipeline_legacy(self):
        """一键运行完整流程：后处理 → 骨架生成 → 骨架优化 → 生成路网图。
        
        包含后处理异常检测和自动暂停逻辑：
        - 后处理后检测 mask 面积异常，若异常则暂停询问用户
        - 一键流程中 close_kernel 强制不超过 5
        - 默认关闭 fill_holes/fill_small_holes
        
        要求：必须已有 road_mask 数据。
        """
        mask, mask_meta = self.get_current_mask_array(prefer_processed=True)
        if mask is None:
            QMessageBox.warning(self, "提示",
                f"无法启动一键流程：{mask_meta.get('error', '请先导入影像并运行道路分割。')}\n\n"
                "请确保已生成正式 global_road_mask.png。")
            return
        err = self._skeleton_input_check(mask, mask_meta)
        if err is not None:
            QMessageBox.warning(self, "一键流程", err)
            return

        # ★ 保存原始 mask 备份（用于异常时回滚）
        original_mask = mask.copy()
        self._mask_before_postprocess = original_mask

        self._status_bar.show_message("一键流程启动中...")
        QApplication.processEvents()

        steps_done = []
        errors = []

        # =================================================================
        # Step 1: 后处理（带异常检测和自动暂停）
        # =================================================================
        postprocess_ok = False
        try:
            self._status_bar.show_message("步骤 1/4: 后处理 road_mask...")
            QApplication.processEvents()
            self._history.push_state("pipeline_step1_postprocess")
            from roadnet.postprocess import clean_pipeline, analyze_mask_anomalies

            # ★ 提取后处理子配置，仅用于 clean_pipeline
            config = self._param_panel.get_config()
            post_cfg = config.get("postprocess", {})
            # ★ 一键流程中 close 不允许超过 5
            if post_cfg.get("close_kernel_size", 3) > 5:
                post_cfg["close_kernel_size"] = 5
            # ★ 一键流程默认不填孔洞
            post_cfg["fill_holes"] = False
            post_cfg["fill_small_holes"] = False

            clean_mask, _ = clean_pipeline(
                mask, post_cfg, save_intermediate=False, output_dir=""
            )

            # 验证后处理结果
            if clean_mask is None or clean_mask.size == 0 or (clean_mask > 0).sum() == 0:
                raise ValueError("后处理输出了空 mask")
            if (clean_mask > 0).sum() == clean_mask.size:
                raise ValueError("后处理后 mask 全白（100% 道路）")

            # ★ 面积异常检测
            anomaly_result = analyze_mask_anomalies(
                clean_mask,
                original_mask=original_mask,
                max_road_ratio=post_cfg.get("max_road_ratio_warn", 0.25),
                max_largest_ratio=post_cfg.get("max_largest_ratio_warn", 0.10),
                max_fill_added_ratio=post_cfg.get("max_fill_added_ratio_warn", 0.05),
            )

            # 先更新预览让用户看到结果
            self._layer_manager.set_layer_data("mask", clean_mask)
            self._canvas.refresh_scene()
            QApplication.processEvents()

            if anomaly_result["is_anomalous"]:
                # ★ 异常 → 自动暂停，让用户选择
                anomaly_details = (
                    f"道路占比: {anomaly_result['road_mask_area_ratio']*100:.1f}%\n"
                    f"最大连通域占比: {anomaly_result['largest_component_area_ratio']*100:.1f}%\n"
                    f"填洞新增: {anomaly_result['fill_added_area_ratio']*100:.1f}%\n"
                )
                warnings_text = "\n".join(
                    f"  • {w}" for w in anomaly_result["warnings"]
                )
                msg = (
                    f"后处理结果可能大面积误填，不建议继续生成 skeleton！\n\n"
                    f"检测数据：\n{anomaly_details}\n"
                    f"警告：\n{warnings_text}\n\n"
                    f"建议：降低 close、关闭 fill_small_holes、提高 min_area 或使用 Ignore 区域。\n\n"
                    f"请选择操作："
                )
                # ★ 三选一对话框
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("一键流程 - 面积异常暂停")
                msg_box.setText(msg)
                msg_box.setIcon(QMessageBox.Icon.Warning)
                btn_revert = msg_box.addButton("回退到原始 road_mask", QMessageBox.ButtonRole.RejectRole)
                btn_adjust = msg_box.addButton("重新调整参数", QMessageBox.ButtonRole.DestructiveRole)
                btn_force  = msg_box.addButton("强制继续（不推荐）", QMessageBox.ButtonRole.AcceptRole)
                msg_box.setDefaultButton(btn_adjust)
                msg_box.exec_()

                clicked = msg_box.clickedButton()
                if clicked is btn_revert:
                    # 回退到原始 mask
                    self._layer_manager.set_layer_data("mask", original_mask)
                    self._canvas.refresh_scene()
                    self._status_bar.show_message("已回退到原始 road_mask")
                    mask = original_mask
                elif clicked is btn_adjust:
                    # 恢复原始 mask，退出让用户调整参数后重试
                    self._layer_manager.set_layer_data("mask", original_mask)
                    self._canvas.refresh_scene()
                    self._status_bar.show_message("已取消一键流程，请调整后处理参数后重新运行")
                    self._mask_before_postprocess = original_mask
                    return
                elif clicked is btn_force:
                    # 强制继续
                    self._status_bar.show_message("⚠ 忽略面积异常，强制继续...")
                    mask = clean_mask
                else:
                    # 关闭窗口 → 等同于回退
                    self._layer_manager.set_layer_data("mask", original_mask)
                    self._canvas.refresh_scene()
                    self._status_bar.show_message("已取消一键流程")
                    return

            if not anomaly_result["is_anomalous"]:
                # 正常通过
                mask = clean_mask
                steps_done.append("后处理")
                postprocess_ok = True
            elif clicked is btn_force:
                # 强制继续也算完成
                steps_done.append("后处理（强制继续）")
                postprocess_ok = True
            # 如果是 btn_revert，postprocess_ok 仍为 False，跳过后续步骤

        except Exception as e:
            errors.append(f"后处理: {e}")
            import traceback
            traceback.print_exc()
            # 恢复原始 mask
            self._layer_manager.set_layer_data("mask", original_mask)
            mask = original_mask

        # ★ 只有当后处理成功（或强制继续 + 回退）才继续后续步骤
        if not postprocess_ok and not steps_done:
            # 后处理完全未成功执行（异常或用户回退），不继续 skeleton/graph
            QApplication.restoreOverrideCursor()
            if not errors:
                self._status_bar.show_message("一键流程: 后处理未成功，已跳过后续步骤")
            else:
                self._status_bar.show_message(f"一键流程: 后处理失败，已跳过后续步骤")
            return

        # =================================================================
        # Step 2: 生成 Skeleton
        # =================================================================
        try:
            self._status_bar.show_message("步骤 2/4: 生成骨架...")
            QApplication.processEvents()
            self._history.push_state("pipeline_step2_skeleton")
            from roadnet.optimized_skeleton import skeletonize_medial_axis
            config = self._param_panel.get_config()
            skel_cfg = config.get("skeleton", {})
            method = skel_cfg.get("method", "medial_axis")
            if method == "medial_axis":
                raw_skeleton = skeletonize_medial_axis(mask)
            else:
                from roadnet.optimized_skeleton import skeletonize_thin
                raw_skeleton = skeletonize_thin(mask)
            self._commit_raw_skeleton(raw_skeleton)
            self._layer_manager.show_layer("skeleton")
            self._canvas.refresh_scene()
            steps_done.append("骨架生成")
        except Exception as e:
            errors.append(f"骨架生成: {e}")
            import traceback
            traceback.print_exc()

        # =================================================================
        # Step 3: 优化 Skeleton
        # =================================================================
        skeleton = self._layer_manager.get_layer_data("skeleton")
        if skeleton is not None:
            try:
                self._status_bar.show_message("步骤 3/4: 优化骨架...")
                QApplication.processEvents()
                self._history.push_state("pipeline_step3_optimize")
                skeleton = self._get_skeleton_array(skeleton)
                from roadnet.optimized_skeleton import optimize_skeleton
                config = self._param_panel.get_config()
                skel_cfg = config.get("skeleton", {})
                result = optimize_skeleton(
                    mask, skeleton,
                    min_center_dist=skel_cfg.get("min_center_dist", 3),
                    min_branch_length=skel_cfg.get("min_branch_length", 10),
                    prune_short_branches=skel_cfg.get("prune_short_branches", True),
                    max_prune_iterations=skel_cfg.get("max_prune_iterations", 3),
                    junction_cluster_radius=skel_cfg.get("junction_cluster_radius", 10),
                    connect_endpoints=skel_cfg.get("connect_endpoints", False),
                    connect_endpoint_max_dist=skel_cfg.get("connect_endpoint_max_dist", 15),
                    connect_endpoint_max_gap_length=skel_cfg.get("connect_endpoint_max_gap_length", 2),
                    endpoint_search_radius=skel_cfg.get("endpoint_search_radius", 8),
                    min_road_width=skel_cfg.get("min_road_width", 2.0),
                    use_hole_detection=skel_cfg.get("use_hole_detection", False),
                    hole_filter_threshold=skel_cfg.get("hole_filter_threshold", 50),
                    use_distance_filter=skel_cfg.get("use_distance_filter", True),
                    min_pixel_count=skel_cfg.get("min_pixel_count", 50),
                    mask_dist_filter_alpha=skel_cfg.get("mask_dist_filter_alpha", 0.67),
                )
                self._canvas.skeleton_result = result
                optimized = result["optimized_skeleton"]
                self._commit_optimized_skeleton(
                    skeleton, optimized, result=result)
                self._layer_manager.show_layer("skeleton")
                self._canvas.refresh_scene()
                steps_done.append("骨架优化")
            except Exception as e:
                errors.append(f"骨架优化: {e}")
                import traceback
                traceback.print_exc()

        # =================================================================
        # Step 4: 生成路网图（graph_build 统一流水线）
        # =================================================================
        skeleton = self._layer_manager.get_layer_data("skeleton")
        if skeleton is not None:
            try:
                self._status_bar.show_message("步骤 4/4: 生成路网图...")
                QApplication.processEvents()
                self._history.push_state("pipeline_step4_graph")
                skeleton_arr = self._get_skeleton_array(skeleton)
                outputs_dir = os.path.join(os.getcwd(), "outputs", "graph_build")

                # ★ 使用 graph_build 统一流水线
                from roadnet.graph_build import build_graph_from_skeleton
                config = self._param_panel.get_config()
                graph_cfg = config.get("graph", {})

                result = build_graph_from_skeleton(
                    skeleton=skeleton_arr,
                    graph_editor=self._graph_editor,
                    output_dir=outputs_dir,
                    config=config,
                    processed_mask=self._layer_manager.get_layer_data("mask"),
                    run_optimization=graph_cfg.get("enable_graph_line_optimizer", False),
                )

                if not result.success:
                    err_msgs = "\n".join(result.errors)
                    errors.append(f"graph_build failed at [{result.stage}]: {err_msgs}")
                    print(f"[GraphBuild] pipeline step 4 failed: {result.stage}")
                else:
                    # 渲染 graph
                    self._render_graph_to_scene()
                    self._update_graph_stats()
                    steps_done.append("路网生成")

                    self._status_bar.update_nodes(len(result.raw_nodes))
                    self._status_bar.update_edges(len(result.raw_edges))

                    # ★ 日志摘要
                    log_lines = result.log.messages[-4:]
                    for line in log_lines:
                        print(f"[Pipeline] {line}")
            except Exception as e:
                errors.append(f"graph_build exception: {e}")
                import traceback
                traceback.print_exc()
                # ★ 保留已生成的 skeleton，不清空

        # =================================================================
        # 总结
        # =================================================================
        QApplication.restoreOverrideCursor()
        if steps_done:
            self.mark_stage_done("edit")
            self.mark_stage_done("skeleton")
            self.set_stage("graph")
            summary = "一键流程完成！\n\n" + "\n".join(f"  ✓ {s}" for s in steps_done)
            if errors:
                summary += "\n\n⚠ 部分步骤失败:\n" + "\n".join(f"  ✗ {e}" for e in errors)
            self._status_bar.show_message(
                f"一键流程完成: {', '.join(steps_done)} "
                f"({len(self._graph_editor.nodes)} 节点, {len(self._graph_editor.edges)} 边)"
            )
            QMessageBox.information(self, "一键流程完成", summary)
        elif errors:
            QMessageBox.critical(self, "一键流程失败",
                "所有步骤均失败:\n" + "\n".join(errors))
            self._status_bar.show_message("一键流程失败")
        else:
            self._status_bar.show_message("一键流程: 无可用数据")

    def _configure_graph_repair_snaps(self):
        """从参数面板读取大图局部修路网吸附参数。"""
        if self._graph_editor is None:
            return
        cfg = self._param_panel.get_config().get("graph", {})
        if hasattr(self._graph_editor, "configure_large_repair_snaps"):
            self._graph_editor.configure_large_repair_snaps(
                node_snap=int(cfg.get("node_snap_distance_px", 25) or 25),
                junction_merge=int(cfg.get("junction_merge_distance_px", 30) or 30),
                endpoint_snap=int(cfg.get("endpoint_snap_distance_px", 25) or 25),
                junction_cluster=int(cfg.get("junction_cluster_radius_px", 30) or 30),
            )

    def _on_graph_polyline_repair(self):
        """进入折线补路工具。"""
        if not self._layer_manager.is_large_image_mode:
            # 小图也可用同一工具，但不强制大图吸附
            self.set_tool("graph_draw_edge")
            return
        self.set_stage("graph")
        self.set_tool("graph_draw_edge")

    def _on_graph_delete_edge_tool(self):
        self.set_stage("graph")
        self.set_tool("graph_delete_edge")
        self._status_bar.show_message("删除错误边：点击高亮边，Delete 删除（支持 Ctrl+Z）")

    def _on_graph_merge_junctions(self):
        if self._graph_editor is None:
            QMessageBox.warning(self, "合并路口", "路网编辑器未初始化。")
            return
        self._configure_graph_repair_snaps()
        self._history.push_state("graph_merge_junctions")
        n = self._graph_editor.merge_nearby_junctions()
        self._render_graph_to_scene()
        self._update_graph_stats()
        warns = self._graph_editor.validate_graph_local()
        msg = f"已合并 {n} 组过近路口节点"
        if warns:
            msg += f"｜{warns[0]}"
        self._status_bar.show_message(msg)
        QMessageBox.information(self, "合并路口节点", msg)

    def _on_graph_local_rebuild(self):
        """局部 ROI graph 重建（仅大图）。"""
        if not self._layer_manager.is_large_image_mode:
            QMessageBox.information(
                self, "小图模式",
                "局部重建路网仅用于大图模式。",
            )
            return
        if self._graph_editor is None:
            QMessageBox.warning(self, "局部重建", "路网编辑器未初始化。")
            return

        rois = []
        if hasattr(self._canvas, "get_roi_polygons"):
            for poly in self._canvas.get_roi_polygons() or []:
                pts = [[float(p.x()), float(p.y())] for p in poly]
                if len(pts) >= 3:
                    rois.append(pts)
        if not rois:
            QMessageBox.information(
                self, "局部重建路网",
                "请先绘制一个局部 ROI（小范围），再点击本按钮。\n"
                "将仅在 ROI 内基于当前 working/final mask 重建 graph。",
            )
            self.set_tool("roi")
            return

        mask, meta = self.get_current_mask_array(
            for_skeleton=True, require_full_resolution=True
        )
        if mask is None:
            # 退回非骨架要求
            mask, meta = self.get_current_mask_array(
                for_skeleton=False, require_full_resolution=True
            )
        if mask is None:
            QMessageBox.warning(
                self, "局部重建",
                meta.get("error") or "未找到 formal working/final mask。",
            )
            return

        # 使用最后一个 ROI
        roi = rois[-1]
        confirm = QMessageBox.question(
            self, "局部重建路网",
            "将删除该 ROI 内现有 graph，并基于当前正式 mask 重建后拼回。\n"
            "确认继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        from roadnet.local_graph_repair import rebuild_graph_in_roi
        from roadnet.graph_utils import polyline_to_list

        self._history.push_state("graph_local_rebuild")
        self._status_bar.show_message("局部重建路网中…")
        QApplication.processEvents()
        try:
            new_nodes, new_edges, report = rebuild_graph_in_roi(
                mask,
                list(self._graph_editor.nodes),
                list(self._graph_editor.edges),
                roi,
            )
        except Exception as exc:
            QMessageBox.critical(self, "局部重建失败", str(exc))
            return

        if not report.get("ok"):
            QMessageBox.warning(
                self, "局部重建未完成",
                report.get("error") or "未知错误",
            )
            return

        # 直接写入编辑器（已是 points_pixel 坐标系），避免 load_draft 二次转换
        ge = self._graph_editor
        ge._nodes = [dict(n) for n in new_nodes]
        ge._edges = []
        for e in new_edges:
            ee = dict(e)
            pts = polyline_to_list(ee.get("points_pixel") or ee.get("polyline") or [])
            ee["points_pixel"] = pts
            ee["polyline"] = pts
            ee.setdefault("enabled", True)
            ee.setdefault("source", "local_repair")
            ge._edges.append(ee)
        ge._next_node_id = max((n["id"] for n in ge._nodes), default=-1) + 1
        ge._next_edge_id = max((e["id"] for e in ge._edges), default=-1) + 1
        ge.clear_selection()
        self._render_graph_to_scene()
        self._update_graph_stats()
        msg = (
            f"局部重建完成：删除节点 {report.get('removed_nodes')}，"
            f"新增节点 {report.get('added_nodes')} / 边 {report.get('added_edges')}"
        )
        self._status_bar.show_message(msg)
        QMessageBox.information(self, "局部重建路网", msg + "\n请检查拼接处后保存 Final Graph。")

    def _on_graph_locate_jump(self):
        """定位异常跳边：缩放并高亮，提供删除/替换。"""
        from roadnet.local_graph_repair import load_jump_debug_rows

        project_dir = ""
        if self._large_image_project is not None:
            project_dir = self._large_image_project.project_dir
        rows = load_jump_debug_rows(
            project_dir=project_dir,
            outputs_dir=os.path.join(os.getcwd(), "outputs"),
        )
        if not rows:
            QMessageBox.information(
                self, "定位异常跳边",
                "未找到 path_jump_debug.csv。\n请先导出路径生成 jump_debug 文件。",
            )
            return

        # 选第一条可疑跳边
        row = rows[0]
        try:
            x = float(row.get("from_x") or row.get("start_x") or 0)
            y = float(row.get("from_y") or row.get("start_y") or 0)
            x2 = float(row.get("to_x") or row.get("end_x") or x)
            y2 = float(row.get("to_y") or row.get("end_y") or y)
        except Exception:
            QMessageBox.warning(self, "定位异常跳边", "跳边坐标无效。")
            return

        cx, cy = (x + x2) / 2.0, (y + y2) / 2.0
        edge_id = row.get("nearest_graph_edge_id")
        ge = self._graph_editor
        if ge is not None:
            ge.clear_selection()
            eid = None
            try:
                if edge_id not in (None, "", "None"):
                    eid = int(float(edge_id))
            except Exception:
                eid = None
            if eid is None:
                hit = ge.find_edge_near(cx, cy, max_dist=80)
                eid = hit[0] if hit else None
            if eid is not None:
                ge.select_edge(eid)
            self._render_graph_to_scene()

        # 缩放到跳边中心
        try:
            if hasattr(self._canvas, "centerOn"):
                # 大图：原图像素 → scene
                sx, sy = cx, cy
                if self._layer_manager.is_large_image_mode and hasattr(
                    self._layer_manager, "global_to_preview"
                ):
                    sx, sy = self._layer_manager.global_to_preview(cx, cy)
                from PySide6.QtCore import QPointF
                self._canvas.centerOn(QPointF(float(sx), float(sy)))
                if hasattr(self._canvas, "zoom_to_100"):
                    pass
        except Exception as exc:
            print(f"[JumpLocate] zoom failed: {exc}")

        detail = (
            f"jump_id={row.get('jump_id') or row.get('jump_index')}\n"
            f"from=({x:.1f},{y:.1f}) → to=({x2:.1f},{y2:.1f})\n"
            f"edge_id={edge_id}\n"
            f"reason={row.get('reason')}\n"
            f"csv={row.get('_source_csv')}"
        )
        box = QMessageBox(self)
        box.setWindowTitle("定位异常跳边")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText("已定位到异常跳边附近。\n请选择修复操作：")
        box.setDetailedText(detail)
        btn_del = box.addButton("删除该边", QMessageBox.ButtonRole.AcceptRole)
        btn_poly = box.addButton("折线替换该边", QMessageBox.ButtonRole.ActionRole)
        btn_inv = box.addButton("标记 invalid", QMessageBox.ButtonRole.ActionRole)
        box.addButton("关闭", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if ge is None:
            return
        sel = list(ge.selected_edges)
        eid = sel[0] if sel else None
        if clicked == btn_del and eid is not None:
            self._history.push_state("graph_delete_edge")
            ge.delete_edge(eid)
            self._render_graph_to_scene()
            self._update_graph_stats()
            self._status_bar.show_message(f"已删除异常边 id={eid}")
        elif clicked == btn_poly:
            if eid is not None:
                self._history.push_state("graph_delete_edge")
                ge.delete_edge(eid)
                self._render_graph_to_scene()
            self.set_tool("graph_draw_edge")
            self._status_bar.show_message("请用折线补路重画该段道路中心线")
        elif clicked == btn_inv and eid is not None:
            self._history.push_state("graph_mark_invalid")
            ge.mark_edge_invalid(eid, reason="jump_debug")
            self._render_graph_to_scene()
            self._update_graph_stats()
            self._status_bar.show_message(f"已标记 edge {eid} 为 invalid")

    def _on_save_graph(self):
        """保存 final_graph（展示型导出）

        同时导出：
        - final_graph.json / final_nodes.csv / final_edges.csv（数据文件）
        - final_graph_overlay_preview.png（预览图 WYSIWYG，所见即所得）
        - final_graph_overlay_original.png（原图 + graph，大图模式时可能很大）

        ★ 保存后立即重新加载验证节点/边数量一致性。
        """
        if self._graph_editor is None:
            QMessageBox.warning(self, "提示", "路网编辑器未初始化。")
            return
        if not self._graph_editor.nodes and not self._graph_editor.edges:
            QMessageBox.warning(self, "提示",
                "当前没有可保存的路网，请先生成草稿路网或手动添加节点/边。")
            return

        ge = self._graph_editor
        n_before = len(ge.nodes)
        e_before = len(ge.edges)

        outputs_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        # ★ 使用原图尺寸保存数据文件
        original_w, original_h = self._layer_manager.original_size
        full_image = self._layer_manager.full_image_rgb

        # 1. 保存图数据文件（使用原图尺寸）
        try:
            graph_path = ge.save(
                output_dir=outputs_dir,
                image_rgb=full_image,
                image_size=(original_w, original_h),
                pixel_resolution_m=self._project_manager.data.pixel_resolution_m,
            )
            # 大图：标注 coordinate_system / 修边 source
            if self._layer_manager.is_large_image_mode and os.path.isfile(graph_path):
                with open(graph_path, encoding="utf-8") as stream:
                    payload = json.load(stream)
                payload["coordinate_system"] = "original_image_pixel"
                for e in payload.get("edges") or []:
                    if e.get("source") in ("manual", "manual_repair", "local_repair"):
                        e.setdefault("polyline", e.get("points_pixel"))
                with open(graph_path, "w", encoding="utf-8") as stream:
                    json.dump(payload, stream, ensure_ascii=False, indent=2)
                # 同步到大图项目目录
                if self._large_image_project is not None:
                    dest = os.path.join(
                        self._large_image_project.project_dir, "graph", "final_graph.json"
                    )
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(graph_path, dest)
        except Exception as e:
            QMessageBox.critical(self, "保存失败",
                f"final_graph.json 保存失败:\n{e}")
            import traceback
            traceback.print_exc()
            return

        # 2. ★ 保存后立即重新加载验证
        verify_ok = True
        verify_msg = ""
        if os.path.exists(graph_path):
            from roadnet.graph_editor_qt import GraphEditorQt
            verify_ge = GraphEditorQt()
            if verify_ge.load_from_file(graph_path):
                n_after = len(verify_ge.nodes)
                e_after = len(verify_ge.edges)
                if n_after == n_before and e_after == e_before:
                    verify_msg = f"（验证通过: {n_after} 节点, {e_after} 边 一致）"
                else:
                    verify_ok = False
                    verify_msg = f"（警告: 保存前 {n_before}节点/{e_before}边, 文件内 {n_after}节点/{e_after}边）"
                    print(f"[ERROR][Graph] 保存验证失败: "
                          f"save={n_before}/{e_before}, file={n_after}/{e_after}")
            else:
                verify_ok = False
                verify_msg = "（验证失败: 无法读取保存的文件）"

        # 3. 展示型预览叠加图（所见即所得）
        overlay_preview = os.path.join(outputs_dir, "final_graph_overlay_preview.png")
        success_preview = self.export_display_image(overlay_preview)

        # 4. 原图叠加图
        success_original = False
        if self._layer_manager.is_large_image_mode:
            if full_image is not None:
                try:
                    ge.save_overlay_preview(
                        full_image, outputs_dir, preview_scale=1.0,
                    )
                    src_path = os.path.join(outputs_dir, "final_graph_overlay_preview.png")
                    dst_path = os.path.join(outputs_dir, "final_graph_overlay_original.png")
                    if os.path.exists(src_path) and src_path != dst_path:
                        if os.path.exists(dst_path):
                            os.remove(dst_path)
                        os.rename(src_path, dst_path)
                    success_original = True
                except Exception as e:
                    print(f"[WARN] 原图叠加图导出失败（可能内存不足）: {e}")

        # ★ 状态栏消息
        msg = f"graph 已保存: nodes={n_before}, edges={e_before} {verify_msg}"
        if success_preview:
            msg += " | 展示图已导出"
        if success_original:
            msg += " | 原图叠加已导出"

        if verify_ok:
            self._status_bar.show_message(msg)
        else:
            self._status_bar.show_message(msg)
            QMessageBox.warning(self, "保存验证异常", verify_msg)

        self.mark_stage_done("graph")

    def _on_save_graph_debug(self):
        """保存 final_graph（调试型导出）

        同时导出：
        - final_graph_debug_preview.png（预览图 + 缩放坐标 graph）
        - final_graph_debug_original.png（原图 + 全局坐标 graph，大图模式可能很大）
        """
        if self._graph_editor is None:
            QMessageBox.warning(self, "提示", "路网编辑器未初始化。")
            return
        if not self._graph_editor.nodes and not self._graph_editor.edges:
            QMessageBox.warning(self, "提示",
                "当前没有可保存的路网，请先生成草稿路网或手动添加节点/边。")
            return

        outputs_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        # 使用原图尺寸保存数据
        original_w, original_h = self._layer_manager.original_size
        full_image = self._layer_manager.full_image_rgb
        self._graph_editor.save(
            output_dir=outputs_dir,
            image_rgb=full_image,
            image_size=(original_w, original_h),
            pixel_resolution_m=self._project_manager.data.pixel_resolution_m,
        )
        print("[DEBUG][Graph] 已保存图数据文件（原图尺寸）")

        # ★ 调试图：预览图叠加（graph 坐标 global→preview）
        preview_image = self._layer_manager.display_image_rgb
        preview_scale = self._layer_manager.preview_scale
        self._graph_editor.save_overlay_preview(
            preview_image, outputs_dir,
            preview_scale=preview_scale,
        )
        # rename to debug_preview
        src_path = os.path.join(outputs_dir, "final_graph_overlay_preview.png")
        dst_path = os.path.join(outputs_dir, "final_graph_debug_preview.png")
        if os.path.exists(src_path):
            if os.path.exists(dst_path):
                os.remove(dst_path)
            os.rename(src_path, dst_path)
        print(f"[Export] 调试图(预览)已保存: {dst_path}")

        # ★ 调试图：原图叠加（仅非大图或内存充足时）
        if not self._layer_manager.is_large_image_mode and full_image is not None:
            debug_original = os.path.join(outputs_dir, "final_graph_debug_original.png")
            img = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR)
            self._graph_editor._draw_overlay(img, scale=1.0)
            cv2.imwrite(debug_original, img)
            print(f"[Export] 调试图(原图)已保存: {debug_original}")

        self._status_bar.show_message(
            "调试图已保存: final_graph_debug_preview.png"
            + (" + original" if not self._layer_manager.is_large_image_mode else "")
        )
        self.mark_stage_done("graph")

    # ===================================================================
    # SAM-Road 单图推理包处理
    # ===================================================================

    def _on_run_samroad_single_extract(self, force_tile=False, continue_pipeline=False):
        """打开 SAM-Road 单图推理包运行参数对话框。

        流程：
        1. 弹出参数配置对话框
        2. 用户确认参数后通过 QProcess 异步启动 infer_single.py
        3. GUI 不卡死，实时显示日志
        4. 运行完成后自动导入结果
        """
        image_path = self._layer_manager.image_path
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "提示", "请先打开影像。")
            return

        from .samroad_single_run_dialog import SAMRoadSingleRunDialog

        self._continue_pipeline_after_samroad = bool(continue_pipeline)
        self._samroad_pipeline_mask_imported = False
        dialog = SAMRoadSingleRunDialog(image_path=image_path, parent=self)
        if force_tile:
            model_index = dialog._combo_model_type.findData("samroadplus_portable")
            if model_index >= 0:
                dialog._combo_model_type.setCurrentIndex(model_index)
            dialog.set_inference_mode("tile", force_auto_import=True)
            if self._large_image_project is not None:
                from datetime import datetime
                run_dir = Path(self._large_image_project.project_dir) / "samroad_large" / (
                    "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                )
                dialog._edit_output_dir.setText(str(run_dir))
        dialog.finished.connect(self._on_samroad_single_run_finished)
        dialog.exec()
        # Continue only after the SAM dialog has closed, so its result summary
        # cannot overlap the pipeline progress dialog.
        if self._continue_pipeline_after_samroad:
            self._continue_pipeline_after_samroad = False
            if self._samroad_pipeline_mask_imported:
                QTimer.singleShot(0, self._on_run_pipeline)

    def _on_samroad_single_run_finished(self, result):
        """SAM-Road 单图推理完成后回调：自动导入结果到当前项目。"""
        is_plus = getattr(result, "model_type", "") == "samroadplus_portable"
        run_name = "SAM-RoadPlus" if is_plus else (
            "DRY-RUN" if result.is_dry_run else "SAM-Road 单图"
        )
        if not result.success:
            self._continue_pipeline_after_samroad = False
            return

        output_dir = result.output_dir
        if not output_dir or not output_dir.is_dir():
            self._continue_pipeline_after_samroad = False
            return
        if not (Path(output_dir) / "road_mask.png").is_file():
            self._continue_pipeline_after_samroad = False
            self._samroad_pipeline_mask_imported = False
            files = sorted(
                str(path.relative_to(output_dir)).replace("\\", "/")
                for path in Path(output_dir).rglob("*") if path.is_file()
            )
            QMessageBox.critical(
                self, "SAM-Road 输出缺少 road_mask",
                "推理进程已经结束，但输出目录没有 road_mask.png。\n"
                "为保护当前项目，已停止自动导入和后续流水线。\n\n"
                f"输出目录：\n{output_dir}\n\n实际文件：\n"
                + ("\n".join(f"- {name}" for name in files[:80]) or "（无）")
            )
            return

        # 检查是否启用自动导入
        dialog = self.sender()
        if dialog and hasattr(dialog, '_chk_auto_import'):
            if not dialog._chk_auto_import.isChecked():
                self._status_bar.show_message(
                    f"{run_name} 完成，"
                    f"已跳过自动导入（用户未勾选）"
                )
                self._continue_pipeline_after_samroad = False
                return

        self._status_bar.show_message(
            f"正在自动导入 {run_name} 结果..."
        )
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            imported_parts = self._import_samroad_single_internal(str(output_dir))
        except Exception as e:
            QMessageBox.warning(self, "自动导入失败", str(e))
            imported_parts = []
        finally:
            QApplication.restoreOverrideCursor()

        graph_imported = any("[参考层]" in part for part in imported_parts)
        self._samroad_pipeline_mask_imported = any("Road Mask" in part for part in imported_parts)
        if not self._samroad_pipeline_mask_imported:
            self._continue_pipeline_after_samroad = False
        msg = f"{run_name} 推理完成"
        if imported_parts:
            msg += f" — 已导入: {', '.join(imported_parts)}"
        if graph_imported:
            msg += " ｜ Graph.p 已作为参考 graph 导入，不作为 final_graph"
        if is_plus and any("Road Mask" in part for part in imported_parts):
            msg = "SAM-RoadPlus 推理完成，已导入 road_mask。" + (
                " Graph 仅作参考，未覆盖 final_graph。" if graph_imported else ""
            )
        self._status_bar.show_message(msg)

    def _import_samroad_single_internal(self, source_dir: str) -> list[str]:
        """内部导入方法：从单图推理包输出目录导入结果。

        导入内容：
        - road_mask.png   → mask 图层（用于后处理）
        - itsc_mask.png   → 额外参考叠加层
        - viz.png         → 可视化参考叠加层
        - graph.p         → 参考图层 graph（不改写 final_graph）

        Returns:
            已导入的内容描述列表
        """
        from roadnet.samroad_single_adapter import (
            load_single_output,
            detect_single_outputs,
        )

        imported_parts: list[str] = []

        from roadnet.samroad_output_diagnostics import diagnose_and_standardize_samroad_outputs
        diagnostics = diagnose_and_standardize_samroad_outputs(source_dir)
        if not diagnostics.get("road_mask_exists"):
            raise FileNotFoundError(
                "SAM-Road 输出目录中没有 road_mask.png 或可映射的候选 mask；"
                "已停止导入。请查看 metadata.json 和 stderr 日志。"
            )

        # ── 探测文件 ──
        detected = detect_single_outputs(source_dir)
        if not detected.get("is_samroad_single", False):
            self._status_bar.show_message("所选目录不是 SAM-Road 单图推理输出目录")
            return imported_parts

        # ── 加载 ──
        output = load_single_output(source_dir)
        if not output.is_valid:
            for err in output.errors:
                self._status_bar.show_message(f"导入错误: {err}")
            return imported_parts

        # ★ 记住 SAM-Road 输出目录，用于后续恢复原始 road_mask
        self._last_samroad_source_dir = source_dir

        # ── 1. road_mask → mask 图层 ──
        if output.has_road_mask:
            if not self._layer_manager.is_large_image_mode:
                self._history.push_state("samroad_single_import")
            mask_img = output.road_mask

            # SAM-Road 单图输出的 road_mask 可能是得分图（0-255 灰度）
            # 而不是严格二值图。需要检查。
            unique_vals = np.unique(mask_img)
            is_binary = len(unique_vals) <= 2 or (
                len(unique_vals) <= 3 and 0 in unique_vals and 255 in unique_vals)

            if not is_binary:
                # 是得分图，做简单二值化
                mask_bin = (mask_img > 30).astype(np.uint8) * 255
                imported_parts.append(
                    f"Road Mask (得分图→二值化, {mask_img.shape[1]}x{mask_img.shape[0]})")
            else:
                mask_bin = mask_img
                imported_parts.append(
                    f"Road Mask (二值, {mask_img.shape[1]}x{mask_img.shape[0]})")

            valid_mask = self._ensure_valid_image_mask()
            if valid_mask is not None and mask_bin.shape[:2] != valid_mask.shape[:2]:
                mask_bin = cv2.resize(
                    mask_bin, (valid_mask.shape[1], valid_mask.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            valid_report = dict(self._valid_mask_report)
            valid_report["removed_road_pixels_estimate"] = int(
                np.count_nonzero((mask_bin > 0) & (valid_mask == 0))
            ) if valid_mask is not None else 0
            mask_bin = self._apply_valid_area(mask_bin)
            try:
                from roadnet.valid_image import save_valid_mask_outputs
                save_valid_mask_outputs(
                    source_dir, self._valid_image_mask, valid_report
                )
            except Exception:
                pass

            preview = None
            if self._layer_manager.is_large_image_mode:
                for preview_name in (
                    "global_road_mask_preview.png", "road_mask_preview.png",
                    "global_mask_preview.png",
                ):
                    candidate = Path(source_dir) / preview_name
                    if candidate.is_file():
                        preview = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
                        if preview is not None:
                            break
            self._layer_manager.set_layer_data("mask", mask_bin, preview_data=preview)
            if self._large_image_project is not None and self._layer_manager.is_large_image_mode:
                standardized = str(Path(source_dir) / "road_mask.png")
                self._large_image_project.global_mask_path = standardized
                valid_path = Path(source_dir) / "valid_image_mask.png"
                if valid_path.is_file():
                    project_valid = Path(self._large_image_project.project_dir) / "valid_image_mask.png"
                    shutil.copy2(valid_path, project_valid)
                    report_path = valid_path.with_name("valid_mask_report.json")
                    if report_path.is_file():
                        shutil.copy2(
                            report_path,
                            Path(self._large_image_project.project_dir) / "valid_mask_report.json",
                        )
                    self._large_image_project.valid_image_mask_path = str(project_valid)
                self._large_image_project.save()
                self._project_manager.data.global_mask_path = standardized
                self._project_manager.mark_dirty()
            self._clear_skeleton_state(clear_layer=True)
            total = mask_bin.size
            road_px = int((mask_bin > 0).sum())
            self._status_bar.update_road_ratio(road_px / total if total else 0)
            self.mark_stage_done("segment")

        # ── 2. itsc_mask → 参考叠加层 ──
        if output.has_itsc_mask:
            self._samroad_single_itsc_mask = output.itsc_mask.copy()
            imported_parts.append(
                f"ITSC Mask ({output.itsc_mask.shape[1]}x{output.itsc_mask.shape[0]})")

        # ── 3. viz → 参考叠加层 ──
        if output.has_viz:
            self._samroad_single_viz = output.viz.copy()
            imported_parts.append(f"Viz ({output.viz.shape[1]}x{output.viz.shape[0]})")

        # ── 4. graph.p → 参考图层 ──
        if output.has_graph:
            self._reference_graph_nodes = output.graph_nodes
            self._reference_graph_edges = output.graph_edges
            self._render_reference_graph_to_scene()
            self._layer_manager.show_layer("reference_graph")
            self._act_reference_visible.setChecked(True)
            self._sync_layer_checkboxes()
            graph_name = "graph.json" if "graph.json" in output.found_files else "graph.p"
            imported_parts.append(
                f"{graph_name} ({output.node_count} 节点, {output.edge_count} 边) [参考层]")
            self.mark_stage_done("skeleton")
        else:
            if output.graph_error:
                imported_parts.append(f"Graph.p 转换失败: {output.graph_error}")

        # ── 5. 渲染叠加层 ──
        self._render_samroad_single_overlays_to_scene()
        self._act_samroad_single_viz_visible.setChecked(True)

        return imported_parts

    # ===================================================================
    # SAM-Road 单图叠加层渲染
    # ===================================================================

    def _render_samroad_single_overlays_to_scene(self):
        """将 itsc_mask 和 viz 渲染为 QGraphicsScene 上的半透明叠加层。"""
        from PySide6.QtWidgets import QGraphicsPixmapItem
        from PySide6.QtGui import QImage, QPixmap, QPainter

        scene = self._canvas.scene()
        if scene is None:
            return
        lm = self._layer_manager

        self._clear_samroad_single_overlay_items()

        display_w, display_h = lm.image_size

        # ── itsc_mask 叠加层（橙色半透明） ──
        if self._samroad_single_itsc_mask is not None:
            itsc = self._samroad_single_itsc_mask
            # 创建 RGBA 叠加图
            h, w = itsc.shape[:2]
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            mask_bin = itsc > 30
            rgba[mask_bin, 0] = 0     # R
            rgba[mask_bin, 1] = 140   # G
            rgba[mask_bin, 2] = 255   # B — 橙色
            rgba[mask_bin, 3] = 100   # Alpha

            if (h, w) != (display_h, display_w):
                rgba = cv2.resize(rgba, (display_w, display_h),
                                  interpolation=cv2.INTER_NEAREST)

            qimg = QImage(rgba.data, display_w, display_h, display_w * 4,
                          QImage.Format.Format_RGBA8888)
            qimg = qimg.copy()  # 关键：防止 buffer 被回收
            pixmap = QPixmap.fromImage(qimg)
            item = QGraphicsPixmapItem(pixmap)
            item.setZValue(self._canvas.ZVAL_AUTO_EDGE - 1)  # 低于 graph
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            scene.addItem(item)
            self._samroad_single_overlay_items.append(item)

        # ── viz 叠加层（彩色半透明） ──
        if self._samroad_single_viz is not None:
            viz = self._samroad_single_viz
            h, w = viz.shape[:2]

            if (h, w) != (display_h, display_w):
                viz = cv2.resize(viz, (display_w, display_h),
                                 interpolation=cv2.INTER_AREA)

            # 转换为 RGBA，设置透明度
            if viz.shape[2] == 3:
                alpha = np.full((display_h, display_w, 1), 100, dtype=np.uint8)
                rgba = np.concatenate([viz, alpha], axis=-1)
            else:
                rgba = viz.copy()
                if rgba.shape[2] == 4:
                    rgba[:, :, 3] = np.minimum(rgba[:, :, 3], 100)

            qimg = QImage(rgba.data, display_w, display_h, display_w * 4,
                          QImage.Format.Format_RGBA8888)
            qimg = qimg.copy()
            pixmap = QPixmap.fromImage(qimg)
            item = QGraphicsPixmapItem(pixmap)
            item.setZValue(self._canvas.ZVAL_NODE - 3)  # 最低参考层
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            scene.addItem(item)
            self._samroad_single_overlay_items.append(item)

        self._canvas.viewport().update()

    def _clear_samroad_single_overlay_items(self):
        """清除单图叠加层的 scene items。"""
        scene = self._canvas.scene() if self._canvas else None
        if scene is None:
            return
        for item in self._samroad_single_overlay_items:
            safe_remove_scene_item(scene, item)
        self._samroad_single_overlay_items.clear()

    def _toggle_samroad_single_overlays(self, visible: bool):
        """切换 itsc_mask / viz 叠加层的显隐。"""
        if visible:
            self._render_samroad_single_overlays_to_scene()
        else:
            self._clear_samroad_single_overlay_items()

    def _on_import_samroad_single(self):
        """手动导入 SAM-Road 单图推理输出目录中的结果。

        打开文件夹选择对话框，将 road_mask / itsc_mask / viz / graph.p
        导入到当前项目，作为 mask 图层和参考图层。
        graph.p 作为参考 graph 导入（金色半透明），不会覆盖 final_graph。
        """
        if not self._layer_manager.has_image():
            QMessageBox.warning(self, "提示", "请先打开影像，以便验证尺寸兼容性。")
            return

        source_dir = QFileDialog.getExistingDirectory(
            self, "选择 SAM-Road 单图输出目录", os.getcwd(),
            QFileDialog.Option.ShowDirsOnly,
        )
        if not source_dir or not os.path.isdir(source_dir):
            return

        self._status_bar.show_message("正在导入 SAM-Road 单图结果...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            imported_parts = self._import_samroad_single_internal(source_dir)
        except Exception as e:
            QMessageBox.warning(self, "导入失败", str(e))
            imported_parts = []
        finally:
            QApplication.restoreOverrideCursor()

        if imported_parts:
            # 构建摘要
            summary_lines = [f"导入来源: {os.path.basename(source_dir)}", "", "✅ 已导入:"]
            for part in imported_parts:
                summary_lines.append(f"   • {part}")
            summary_lines.append("")
            summary_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")
            summary_lines.append("Graph.p 已作为参考 graph 导入（金色），不作为 final_graph。")
            summary_lines.append("后续工作流: road_mask → 后处理 → 骨架优化 → graph 提取 → final_graph")
            summary_text = "\n".join(summary_lines)

            # 状态栏明确提示
            self._status_bar.show_message(
                f"SAM-Road 单图 导入完成: {', '.join(imported_parts)} ｜ "
                f"Graph.p 为参考层，非 final_graph"
            )

            QMessageBox.information(self, "SAM-Road 单图导入完成", summary_text)
        else:
            self._status_bar.show_message(
                "导入失败：所选目录不是有效的 SAM-Road 单图推理输出目录"
            )

    # ===================================================================

    def _on_run_samroad_segment(self):
        """[已废弃] 旧的硬编码 SAM-Road 调用方式，现在重定向到对话框模式。"""
        self._on_run_samroad_extract()

    def _on_run_samroad_extract(self):
        """打开 SAM-Road 运行参数对话框，支持 QProcess 异步非阻塞运行。

        流程：
        1. 弹出参数配置对话框
        2. 用户确认参数后通过 QProcess 异步启动 bridge 脚本
        3. GUI 不卡死，实时显示日志
        4. 运行完成后自动导入结果（如果用户勾选了 auto_import）
        """
        image_path = self._layer_manager.image_path
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "提示", "请先打开影像。")
            return

        from .samroad_run_dialog import SAMRoadRunDialog

        dialog = SAMRoadRunDialog(image_path=image_path, parent=self)
        dialog.finished.connect(self._on_samroad_run_finished)
        dialog.exec()

    def _on_samroad_run_finished(self, result: SAMRoadRunResult):
        """SAM-Road 运行完成后的回调：自动导入结果到当前项目。

        如果用户在对话框中勾选了 auto_import_after_run，
        则自动调用 samroad_adapter 导入 mask → skeleton → graph。
        """
        if not result.success:
            return

        output_dir = result.output_dir
        if not output_dir or not output_dir.is_dir():
            return
        from roadnet.samroad_output_diagnostics import diagnose_and_standardize_samroad_outputs
        sender = self.sender()
        sender_config = getattr(sender, "_config", None)
        diagnostics = diagnose_and_standardize_samroad_outputs(
            output_dir,
            project_dir=getattr(sender_config, "project_dir", None),
        )
        if not diagnostics.get("road_mask_exists"):
            self._status_bar.show_message("SAM-Road 未生成 road_mask，已停止自动导入")
            QMessageBox.critical(
                self, "SAM-Road 输出缺少 road_mask",
                "SAM-Road 推理结束，但没有生成 road_mask 或其他候选 mask 文件。\n"
                "已停止自动导入及后续处理。请查看输出目录中的 metadata.json 和 stderr 日志。"
            )
            return

        # 检查是否启用自动导入
        dialog = sender
        if dialog and hasattr(dialog, '_chk_auto_import'):
            if not dialog._chk_auto_import.isChecked():
                self._status_bar.show_message(
                    f"{'DRY-RUN' if result.is_dry_run else 'SAM-Road'} 完成，"
                    f"已跳过自动导入（用户未勾选）"
                )
                return

        # 自动导入
        self._status_bar.show_message(
            f"正在自动导入 {'DRY-RUN' if result.is_dry_run else 'SAM-Road'} 结果..."
        )
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            imported_parts = self._import_samroad_internal(str(output_dir))
        except Exception as e:
            QMessageBox.warning(self, "自动导入失败", str(e))
            imported_parts = []
        finally:
            QApplication.restoreOverrideCursor()

        # 状态栏更新
        self._status_bar.show_message(
            f"{'DRY-RUN' if result.is_dry_run else 'SAM-Road'} 完成"
            + (f" — 已导入: {', '.join(imported_parts)}" if imported_parts else "")
        )

    def _import_samroad_internal(self, source_dir: str) -> list[str]:
        """内部导入方法：从 SAM-Road 输出目录导入结果到当前项目。

        此方法被 _on_import_samroad（手动导入）和 _on_samroad_run_finished（自动导入）共用。

        Returns:
            已导入的内容描述列表，例如 ["Mask", "Skeleton", "Graph (29 节点, 27 边)"]
        """
        imported_parts: list[str] = []

        # ── 探测文件 ──
        detected = detect_samroad_outputs(source_dir)
        if not detected["is_samroad"]:
            return imported_parts

        # ── 加载并验证 ──
        output = load_samroad_output(source_dir)
        current_size = self._layer_manager.original_size
        output = validate_samroad_output(output, expected_size=current_size)

        if not output.is_valid:
            return imported_parts

        # ── 应用 Mask ──
        mask_loaded = False
        if output.has_mask:
            self._history.push_state("samroad_import")
            mask_to_use = output.mask_raw
            if output.has_mask_clean:
                mask_to_use = output.mask_clean
                imported_parts.append(f"Mask (clean, {mask_to_use.shape[1]}x{mask_to_use.shape[0]})")
            else:
                imported_parts.append(f"Mask (raw, {mask_to_use.shape[1]}x{mask_to_use.shape[0]})")

            self._layer_manager.set_layer_data("mask", mask_to_use)
            self._clear_skeleton_state(clear_layer=True)
            total = mask_to_use.size
            road_px = int((mask_to_use > 0).sum())
            self._status_bar.update_road_ratio(road_px / total if total else 0)
            mask_loaded = True

        # ── 应用 Skeleton ──
        skeleton_loaded = False
        if output.has_skeleton:
            if not mask_loaded:
                self._history.push_state("samroad_import")
            skeleton_img = output.skeleton
            original_w, original_h = self._layer_manager.original_size
            sk_h, sk_w = skeleton_img.shape[:2]
            if (sk_w != original_w or sk_h != original_h):
                if sk_w > 0 and sk_h > 0:
                    skeleton_img = cv2.resize(
                        skeleton_img, (original_w, original_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    imported_parts.append(f"Skeleton ({sk_w}x{sk_h}→{original_w}x{original_h})")
                else:
                    skeleton_img = None
            if skeleton_img is not None:
                skeleton_bin = (skeleton_img > 0).astype(np.uint8) * 255
                self._commit_raw_skeleton(skeleton_bin)
                skeleton_loaded = True
        else:
            imported_parts.append("Skeleton (未找到)")

        # ── 应用 Graph（作为参考图层，不改写 final_graph）──
        if output.has_graph:
            if not mask_loaded and not skeleton_loaded:
                self._history.push_state("samroad_import")
            nodes_raw, edges_raw = load_graph_for_draft(source_dir)
            if nodes_raw or edges_raw:
                # ★ 保存为参考图层数据，不写入 graph_editor (final_graph)
                self._reference_graph_nodes = nodes_raw
                self._reference_graph_edges = edges_raw
                self._render_reference_graph_to_scene()
                self._layer_manager.show_layer("reference_graph")
                self._act_reference_visible.setChecked(True)
                self._sync_layer_checkboxes()
                imported_parts.append(f"SAM-Road Graph ({len(nodes_raw)} 节点, {len(edges_raw)} 边) [参考层]")
        else:
            imported_parts.append("Graph (未找到)")

        # ── 更新阶段 ──
        if mask_loaded or graph_loaded:
            self.mark_stage_done("segment")
        if skeleton_loaded or graph_loaded:
            self.mark_stage_done("skeleton")

        # ── 确保图层可见 ──
        if skeleton_loaded:
            self._layer_manager.show_layer("skeleton")
            self._act_skeleton_visible.setChecked(True)
            self._sync_layer_checkboxes()

        return imported_parts

    def _on_import_samroad(self):
        """从已有 SAM-Road 输出目录导入 mask、skeleton 和 draft graph 到当前项目。

        支持两种选择方式：
        1. 选择 SAM-Road 输出目录（自动检测其中的文件）
        2. 直接选择 draft_graph.json 文件（逆向查找同目录下的其他文件）

        导入内容（按文件存在情况自动加载）：
        - road_mask_raw.png          → mask 图层
        - road_mask.png               → 清理后 mask（如果有）
        - road_skeleton.png/skeleton.png → skeleton 图层
        - draft_graph.json            → graph（节点+边）
        """
        if not self._layer_manager.has_image():
            QMessageBox.warning(self, "提示", "请先打开影像，以便验证 SAM-Road 输出的尺寸兼容性。")
            return

        # ── 选择目录或 graph 文件 ──
        from PySide6.QtWidgets import QFileDialog

        reply = QMessageBox.question(
            self,
            "导入 SAM-Road 结果",
            "请选择导入方式：\n\n"
            "• 选择 SAM-Road 输出文件夹 → 自动检测并导入所有结果\n"
            "• 选择 draft_graph.json 文件 → 导入指定 graph 及同目录其他文件\n\n"
            "点击「是」选择文件夹，点击「否」选择文件。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )

        if reply == QMessageBox.StandardButton.Cancel:
            return

        source_dir = ""
        if reply == QMessageBox.StandardButton.Yes:
            source_dir = QFileDialog.getExistingDirectory(
                self, "选择 SAM-Road 输出目录", os.getcwd(),
                QFileDialog.Option.ShowDirsOnly,
            )
        else:
            graph_file, _ = QFileDialog.getOpenFileName(
                self, "选择 draft_graph.json", os.getcwd(),
                "JSON 文件 (*.json);;所有文件 (*)",
            )
            if graph_file:
                source_dir = os.path.dirname(graph_file)

        if not source_dir or not os.path.isdir(source_dir):
            return

        # ── 探测文件 ──
        detected = detect_samroad_outputs(source_dir)
        if not detected["is_samroad"]:
            QMessageBox.warning(
                self, "非 SAM-Road 目录",
                f"所选目录不包含 SAM-Road 输出文件：\n{source_dir}\n\n"
                "需要至少包含 road_mask_raw.png 或 draft_graph.json 之一。"
            )
            return

        # ── 加载并验证 ──
        self._status_bar.show_message("正在导入 SAM-Road 结果...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            output = load_samroad_output(source_dir)
            # 验证与当前图像的兼容性
            current_size = self._layer_manager.original_size
            output = validate_samroad_output(output, expected_size=current_size)
        finally:
            QApplication.restoreOverrideCursor()

        # 显示警告
        if output.warnings:
            warnings_text = "导入过程中有以下警告：\n• " + "\n• ".join(output.warnings)
            QMessageBox.warning(self, "SAM-Road 导入警告", warnings_text)

        if not output.is_valid:
            errors_text = "导入失败，以下错误无法自动修复：\n• " + "\n• ".join(output.errors)
            QMessageBox.critical(self, "SAM-Road 导入错误", errors_text)
            return

        # ── 统计导入内容 ──
        imported_parts = []

        # ── 应用 Mask ──
        mask_loaded = False
        if output.has_mask:
            self._history.push_state("samroad_import")
            # 优先使用清理后的 mask
            mask_to_use = output.mask_raw
            if output.has_mask_clean:
                mask_to_use = output.mask_clean
                imported_parts.append(f"清理 Mask ({output.mask_clean.shape[1]}x{output.mask_clean.shape[0]})")
            else:
                imported_parts.append(f"原始 Mask ({output.mask_raw.shape[1]}x{output.mask_raw.shape[0]})")

            self._layer_manager.set_layer_data("mask", mask_to_use)
            self._clear_skeleton_state(clear_layer=True)
            total = mask_to_use.size
            road_px = int((mask_to_use > 0).sum())
            self._status_bar.update_road_ratio(road_px / total if total else 0)
            mask_loaded = True

        # ── 应用 Skeleton ──
        skeleton_loaded = False
        if output.has_skeleton:
            if not mask_loaded:
                self._history.push_state("samroad_import")
            skeleton_img = output.skeleton

            # 尺寸检查：skeleton 应与原图尺寸匹配
            original_w, original_h = self._layer_manager.original_size
            sk_h, sk_w = skeleton_img.shape[:2]
            if (sk_w != original_w or sk_h != original_h):
                # 尝试 resize 到原图尺寸
                if sk_w > 0 and sk_h > 0:
                    skeleton_img = cv2.resize(
                        skeleton_img, (original_w, original_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    imported_parts.append(
                        f"Skeleton ({sk_w}x{sk_h}→{original_w}x{original_h}，已自动缩放)"
                    )
                else:
                    imported_parts.append(f"Skeleton ({sk_w}x{sk_h}，尺寸不匹配，已跳过)")
                    skeleton_img = None

            if skeleton_img is not None:
                # 二值化确保 0/255
                skeleton_bin = (skeleton_img > 0).astype(np.uint8) * 255
                self._commit_raw_skeleton(skeleton_bin)
                skeleton_loaded = True

        # ── 应用 Graph（作为参考图层）──
        graph_loaded = False
        if output.has_graph:
            if not mask_loaded and not skeleton_loaded:
                self._history.push_state("samroad_import")
            nodes_raw, edges_raw = load_graph_for_draft(source_dir)
            if nodes_raw or edges_raw:
                # ★ 保存为参考图层，不改写 final_graph
                self._reference_graph_nodes = nodes_raw
                self._reference_graph_edges = edges_raw
                self._render_reference_graph_to_scene()
                self._layer_manager.show_layer("reference_graph")
                self._act_reference_visible.setChecked(True)
                self._sync_layer_checkboxes()
                graph_loaded = True
                imported_parts.append(f"SAM-Road Graph ({len(nodes_raw)} 节点, {len(edges_raw)} 边) [参考层]")

        # ── 更新阶段 ──
        if mask_loaded:
            self.mark_stage_done("segment")
        if skeleton_loaded:
            self.mark_stage_done("skeleton")

        # ── 确保 skeleton 图层可见 ──
        if skeleton_loaded:
            self._layer_manager.show_layer("skeleton")
            # 同步视图菜单
            self._act_skeleton_visible.setChecked(True)
            self._sync_layer_checkboxes()

        # ★ 如果有 mask 但没有 skeleton，提示用户可自动生成
        generate_skeleton_hint = ""
        if mask_loaded and not skeleton_loaded:
            generate_skeleton_hint = (
                "\n\n💡 提示：已导入 road mask，但没有 skeleton。"
                "可以使用「工具 → 从当前 mask 生成 skeleton」自动生成。"
            )

        # ── 构建详细的导入摘要 ──
        summary_lines = [
            f"导入来源: {os.path.basename(source_dir)}",
            "",
        ]

        # 发现的文件列表
        found = detected.get("found_files", [])
        if found:
            summary_lines.append(f"📁 发现文件 ({len(found)} 个):")
            for f in sorted(found):
                summary_lines.append(f"   • {f}")
            summary_lines.append("")

        # 已导入的内容
        summary_lines.append("✅ 已导入:")
        for part in imported_parts:
            summary_lines.append(f"   • {part}")

        if not imported_parts:
            summary_lines.append("   (无)")

        # 未找到的可选文件
        missing = []
        if not detected.get("has_mask_raw"):
            missing.append("road_mask_raw.png")
        if not detected.get("has_mask_clean"):
            missing.append("road_mask.png (清理后 mask)")
        if not detected.get("has_mask_score"):
            missing.append("road_mask_samroad_score.png")
        if not detected.get("has_keypoint"):
            missing.append("keypoint_mask_samroad_score.png")
        if not detected.get("has_skeleton"):
            missing.append("road_skeleton.png / skeleton.png")
        if not detected.get("has_graph"):
            missing.append("draft_graph.json")
        if not detected.get("has_overlay"):
            missing.append("draft_graph_overlay.png")

        if missing:
            summary_lines.append("")
            summary_lines.append("⚠️ 未找到（可选的）:")
            for m in missing:
                summary_lines.append(f"   • {m}")

        summary_text = "\n".join(summary_lines) + generate_skeleton_hint

        QMessageBox.information(self, "SAM-Road 导入完成", summary_text)

        # ── 状态栏摘要 ──
        parts_short = []
        if mask_loaded:
            parts_short.append("Mask")
        if skeleton_loaded:
            parts_short.append("Skeleton")
        if graph_loaded:
            parts_short.append(f"Graph({output.node_count}N/{output.edge_count}E)")
        self._status_bar.show_message(
            f"SAM-Road 导入: {', '.join(parts_short)} — {os.path.basename(source_dir)}"
        )

    def _on_run_segment(self, preview: bool = False):
        if self._segmentation_thread is not None and self._segmentation_thread.isRunning():
            self._status_bar.show_message("分割任务已在运行，请勿重复启动")
            return
        if self._pipeline_thread is not None and self._pipeline_thread.isRunning():
            QMessageBox.information(self, "一键流程正在运行", "请先完成或取消一键流程。")
            return

        is_large = self._layer_manager.is_large_image_mode
        img = self._layer_manager.full_image_rgb
        if not is_large and img is None:
            img = self._layer_manager.display_image_rgb
        if is_large:
            width, height = self._layer_manager.original_size
            if not self._layer_manager.image_path or width <= 0 or height <= 0:
                QMessageBox.warning(self, "图像为空", "大图项目缺少原始影像路径或尺寸。")
                return
        elif img is None or not isinstance(img, np.ndarray) or img.size == 0:
            QMessageBox.warning(self, "图像为空", "请先打开有效影像。")
            return
        else:
            height, width = img.shape[:2]
        pos_points = list(self._canvas.positive_points)
        neg_points = list(self._canvas.negative_points)
        if not pos_points:
            QMessageBox.warning(self, "正样本为空", "请先添加道路正样本（绿点）。")
            return

        config = copy.deepcopy(self._param_panel.get_config())
        seg_cfg = config.get("segment", {})
        if seg_cfg.get("use_negative_samples", True) and not neg_points:
            QMessageBox.warning(self, "负样本为空", "当前启用了负样本约束，请先添加非道路负样本（红叉）。")
            return
        seg_cfg["require_negative_samples"] = bool(
            seg_cfg.get("use_negative_samples", True)
        )
        intensity = seg_cfg.get("intensity", "标准")
        if intensity == "保守":
            seg_cfg.update(h_margin=4, s_margin=15, v_margin=20)
        elif intensity == "宽松":
            seg_cfg.update(h_margin=10, s_margin=40, v_margin=45)

        def valid_points(points):
            return [(int(x), int(y)) for x, y in points
                    if 0 <= int(x) < width and 0 <= int(y) < height]

        pos_points = valid_points(pos_points)
        neg_points = valid_points(neg_points)
        if not pos_points:
            QMessageBox.warning(self, "样本点无效", "正样本点均位于图像范围之外。")
            return
        if is_large:
            from roadnet.large_image_project import ImageRegionReader
            reader = ImageRegionReader(self._layer_manager.image_path)
            pos_rgb = reader.read_pixels(pos_points)
            neg_rgb = reader.read_pixels(neg_points)
        else:
            pos_rgb = np.asarray([img[y, x] for x, y in pos_points], dtype=np.uint8)
            neg_rgb = np.asarray([img[y, x] for x, y in neg_points], dtype=np.uint8)

        def polygons_to_global(qpolygons):
            # Canvas region geometry is already stored in original image pixels.
            result = []
            for polygon in qpolygons:
                global_points = [[float(p.x()), float(p.y())] for p in polygon]
                if len(global_points) >= 3:
                    result.append(global_points)
            return result

        roi_polygons = polygons_to_global(self._canvas.get_roi_polygons())
        ignore_polygons = polygons_to_global(self._canvas.get_ignore_polygons())
        if not seg_cfg.get("use_roi_only", True):
            roi_polygons = []
        if is_large:
            QMessageBox.information(
                self, "大图分块分割",
                "当前为大图，建议使用 tile 分块分割。软件将自动使用后台分块模式，处理期间可取消。"
            )

        if is_large:
            from roadnet.large_image_worker import LargeImageSegmentationWorker
        else:
            from roadnet.segmentation_worker import SegmentationWorker

        tile_size = int(seg_cfg.get("tile_size", 1024))
        overlap = int(seg_cfg.get("overlap", 64))
        if overlap >= tile_size:
            QMessageBox.warning(self, "分割参数错误", "Tile 重叠必须小于 Tile 大小。")
            return
        from datetime import datetime
        if is_large and self._large_image_project is not None:
            output_dir = os.path.join(
                self._large_image_project.project_dir, "masks",
                f"segmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
        else:
            output_dir = os.path.join(
                os.getcwd(), "outputs", "segmentation",
                f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
        thread = QThread(self)
        common_worker_args = dict(
            positive_samples_rgb=pos_rgb, negative_samples_rgb=neg_rgb,
            config=seg_cfg, output_dir=output_dir,
            roi_polygons=roi_polygons, ignore_polygons=ignore_polygons,
            tile_size=tile_size, overlap=overlap,
            skip_black_area=bool(seg_cfg.get("skip_black_area", True)),
            black_threshold=int(seg_cfg.get("black_threshold", 10)),
            valid_pixel_ratio_threshold=float(seg_cfg.get("valid_pixel_ratio_threshold", 0.1)),
        )
        if is_large:
            worker = LargeImageSegmentationWorker(
                image_path=self._layer_manager.image_path,
                **common_worker_args,
            )
        else:
            worker = SegmentationWorker(
                image_rgb=img,
                preview_scale=(float(seg_cfg.get("preview_scale", 0.25)) if preview else 1.0),
                min_black_component_area=int(seg_cfg.get("min_black_component_area", 4096)),
                **common_worker_args,
            )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_segmentation_progress)
        worker.finished.connect(self._on_segmentation_finished)
        worker.failed.connect(self._on_segmentation_failed)
        worker.cancelled.connect(self._on_segmentation_cancelled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(self._on_segmentation_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._segmentation_thread = thread
        self._segmentation_worker = worker
        self._segmentation_image_ref = (
            os.path.abspath(self._layer_manager.image_path) if is_large else img
        )
        for nav_button in self._nav_buttons.values():
            nav_button.setEnabled(False)
        self._btn_pipeline.setEnabled(False)
        self._param_panel.set_segmentation_running(True)
        self._param_panel.update_segmentation_progress(0, 0, 0, "准备 tile 网格…")
        self._status_bar.show_message(
            f"{'快速预览' if preview else '后台精细分割'}已启动: "
            f"{width}x{height}, tile={tile_size}, overlap={overlap}"
        )
        thread.start()

    def _on_run_preview_segmentation(self):
        """快速预览分割：只处理 preview.png，不访问原始大图。

        与 _on_run_segment(preview=True) 完全不同：
        - 不取 tile 分块
        - 不加载全尺寸原图
        - 不做 connectedComponents / findContours / fill_holes
        - 不生成 global_road_mask.png
        - 不覆盖正式 mask 图层
        - 5 秒内返回
        """
        if self._preview_seg_thread is not None and self._preview_seg_thread.isRunning():
            self._status_bar.show_message("快速预览分割已在运行，请等待完成或取消。")
            return
        if self._segmentation_thread is not None and self._segmentation_thread.isRunning():
            QMessageBox.information(self, "正式分割运行中", "请等待正式分割完成后再执行快速预览。")
            return

        is_large = self._layer_manager.is_large_image_mode
        if not is_large:
            # 普通小图直接走原 preview 流程（25% 缩放 tile 分割）
            self._on_run_segment(preview=True)
            return

        # 大图模式：使用 preview.png
        if self._large_image_project is None:
            QMessageBox.warning(self, "无大图项目", "大图模式下缺少 large_image_project，无法获取预览图。")
            return

        preview_path = self._large_image_project.preview_path
        if not preview_path or not os.path.isfile(preview_path):
            QMessageBox.warning(self, "预览图丢失", f"找不到 preview.png: {preview_path}")
            return

        # 样本点校验
        width, height = self._layer_manager.original_size
        pos_points = list(self._canvas.positive_points)
        neg_points = list(self._canvas.negative_points)
        if not pos_points:
            QMessageBox.warning(self, "正样本为空", "请先添加道路正样本（绿点）。")
            return

        def valid_points(points):
            return [(int(x), int(y)) for x, y in points
                    if 0 <= int(x) < width and 0 <= int(y) < height]

        pos_points = valid_points(pos_points)
        neg_points = valid_points(neg_points)
        if not pos_points:
            QMessageBox.warning(self, "样本点无效", "正样本点均位于图像范围之外。")
            return

        # 从原始大图按需读取样本点颜色（只读像素级点，安全）
        from roadnet.large_image_project import ImageRegionReader
        reader = ImageRegionReader(self._layer_manager.image_path)
        pos_rgb = reader.read_pixels(pos_points)
        neg_rgb = reader.read_pixels(neg_points)

        # 配置
        config = copy.deepcopy(self._param_panel.get_config())
        seg_cfg = config.get("segment", {})
        if seg_cfg.get("use_negative_samples", True) and len(neg_rgb) == 0:
            QMessageBox.warning(self, "负样本为空", "当前启用了负样本约束，请先添加非道路负样本（红叉）。")
            return

        intensity = seg_cfg.get("intensity", "标准")
        if intensity == "保守":
            seg_cfg.update(h_margin=4, s_margin=15, v_margin=20)
        elif intensity == "宽松":
            seg_cfg.update(h_margin=10, s_margin=40, v_margin=45)

        # 比赛模式：从 checkbox 实时读取
        competition = bool(seg_cfg.get("competition_fast_mode", False))
        max_side = 1200 if competition else 1500

        # 输出目录（统一到大图项目下的 preview_segmentation）
        from datetime import datetime
        output_dir = os.path.join(
            self._large_image_project.project_dir, "preview_segmentation",
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        # 创建 Worker
        from roadnet.preview_segmentation import (
            PreviewSegmentationWorker, DEFAULT_PREVIEW_MAX_SIDE,
        )
        thread = QThread(self)
        worker = PreviewSegmentationWorker(
            preview_path=preview_path,
            pos_samples_rgb=pos_rgb,
            neg_samples_rgb=neg_rgb,
            output_dir=output_dir,
            config=seg_cfg,
            preview_max_side=max_side,
            competition_fast_mode=competition,
            save_debug_files=False,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_preview_seg_progress)
        worker.finished.connect(self._on_preview_seg_finished)
        worker.failed.connect(self._on_preview_seg_failed)
        worker.cancelled_signal.connect(self._on_preview_seg_cancelled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled_signal.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled_signal.connect(worker.deleteLater)
        thread.finished.connect(self._on_preview_seg_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._preview_seg_thread = thread
        self._preview_seg_worker = worker
        self._param_panel.set_preview_seg_running(True)
        self._status_bar.show_message(
            f"快速预览分割启动: preview {max_side}px, "
            f"{'比赛模式' if competition else '标准模式'}"
        )
        thread.start()

    def _on_emergency_hand_draw_graph(self):
        """应急手绘中心线（可能扣分）— 仅切换到手动画边工具。"""
        QMessageBox.warning(
            self, "应急手绘模式",
            "应急手绘中心线可能扣一半分。\n\n"
            "请优先使用「大图比赛快速路网生成」。\n"
            "仅在自动路网失败且时间不够时，用手动画边做少量补修。",
        )
        self.set_tool("graph_draw_edge")
        self._status_bar.show_message("已进入应急手绘模式（可能扣分）")

    def _on_competition_fast_roadnet(self):
        """大图比赛快速路网：回调只做轻量检查并启动 QThread，立即 return。

        ★ 禁止在此函数中：imread 大图 / resize / 分割 / skeleton / wait / worker.run
        ★ 不要求已有 Working Road Mask
        """
        # 若正在运行 → 当作取消
        if getattr(self, "_competition_fast_thread", None) is not None:
            if self._competition_fast_thread.isRunning():
                self._on_cancel_competition_fast_roadnet()
                return

        if not self._layer_manager.is_large_image_mode:
            QMessageBox.information(
                self, "小图模式",
                "Competition Fast Roadnet 仅用于大图模式。\n"
                "小图请继续使用原有提取流程。",
            )
            return
        image_path = self._layer_manager.image_path or ""
        if not image_path or not os.path.isfile(image_path):
            QMessageBox.warning(self, "无影像", "请先打开大图。")
            return
        if self._large_image_project is None:
            QMessageBox.warning(self, "无大图项目", "缺少 large_image_project。")
            return

        # 仅收集样本坐标（轻量）；颜色读取放到 worker 线程
        width, height = self._layer_manager.original_size
        pos_points = [
            (int(x), int(y)) for x, y in self._canvas.positive_points
            if 0 <= int(x) < width and 0 <= int(y) < height
        ]
        neg_points = [
            (int(x), int(y)) for x, y in self._canvas.negative_points
            if 0 <= int(x) < width and 0 <= int(y) < height
        ]
        if not pos_points:
            QMessageBox.warning(self, "正样本为空", "请先添加道路正样本（绿点）。")
            return

        config = self._param_panel.get_config()
        seg_cfg = config.get("segment", {})
        if seg_cfg.get("use_negative_samples", True) and not neg_points:
            QMessageBox.warning(self, "负样本为空", "请先添加非道路负样本（红叉）。")
            return

        self._competition_fast_mode = True
        max_side = int(seg_cfg.get("competition_preview_max_side", 1500) or 1500)
        max_side = max(1000, min(4096, max_side))

        from datetime import datetime
        from roadnet.competition_fast_roadnet import CompetitionFastConfig
        from roadnet.competition_fast_roadnet_worker import CompetitionFastRoadnetWorker
        from PySide6.QtCore import Qt as _Qt

        clicked_time = datetime.now().isoformat(timespec="seconds")
        output_dir = os.path.join(
            self._large_image_project.project_dir,
            "competition_fast_roadnet",
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, "无法创建输出目录", str(exc))
            return

        intensity = seg_cfg.get("intensity", "标准")
        h_margin = int(seg_cfg.get("h_margin", 6))
        s_margin = int(seg_cfg.get("s_margin", 25))
        v_margin = int(seg_cfg.get("v_margin", 30))
        if intensity == "保守":
            h_margin, s_margin, v_margin = 4, 15, 20
        elif intensity == "宽松":
            h_margin, s_margin, v_margin = 10, 40, 45

        fast_cfg = CompetitionFastConfig(
            competition_preview_max_side=max_side,
            process_full_resolution=False,
            use_preview_as_formal_graph_source=True,
            debug_mode=False,
            h_margin=h_margin,
            s_margin=s_margin,
            v_margin=v_margin,
            lab_margin=int(seg_cfg.get("lab_margin", 12)),
            use_negative_samples=bool(seg_cfg.get("use_negative_samples", True)),
            blur_kernel=int(seg_cfg.get("preview_blur_kernel", seg_cfg.get("blur_kernel", 3))),
            mode=str(seg_cfg.get("mode", "combined")),
            combine_method=str(seg_cfg.get("combine_method", "and")),
            sample_radius=int(seg_cfg.get("sample_radius", 3)),
            black_threshold=int(seg_cfg.get("black_threshold", 10)),
        )

        import threading as _threading
        thread = QThread(self)
        worker = CompetitionFastRoadnetWorker(
            image_path=image_path,
            pos_points=pos_points,
            neg_points=neg_points,
            output_dir=output_dir,
            config=fast_cfg,
            main_thread_id=int(_threading.get_ident()),
            button_clicked_time=clicked_time,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress_changed.connect(
            self._on_competition_fast_progress, _Qt.ConnectionType.QueuedConnection
        )
        worker.heartbeat.connect(
            self._on_competition_fast_heartbeat, _Qt.ConnectionType.QueuedConnection
        )
        worker.stage_changed.connect(
            self._on_competition_fast_stage, _Qt.ConnectionType.QueuedConnection
        )
        worker.finished.connect(
            self._on_competition_fast_finished, _Qt.ConnectionType.QueuedConnection
        )
        worker.error_occurred.connect(
            self._on_competition_fast_failed, _Qt.ConnectionType.QueuedConnection
        )
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_competition_fast_thread_finished)
        thread.finished.connect(thread.deleteLater)

        self._competition_fast_thread = thread
        self._competition_fast_worker = worker
        self._competition_fast_output_dir = output_dir
        self._param_panel.set_competition_fast_running(True)
        self._param_panel.update_competition_fast_status(
            max_side=max_side, ready=False,
        )
        self._status_bar.show_message(
            f"大图比赛快速路网生成中：precheck，已用 0 秒（max_side={max_side}）"
        )
        thread.start()
        return  # ★ 立即返回，不阻塞 UI

    def _on_cancel_competition_fast_roadnet(self):
        worker = getattr(self, "_competition_fast_worker", None)
        if worker is not None:
            worker.request_cancel()
            self._status_bar.show_message("正在取消快速路网生成…")
            self._param_panel.update_competition_fast_progress(0, "取消中…")

    def _on_competition_fast_stage(self, stage: str):
        self._status_bar.show_message(f"大图比赛快速路网生成中：{stage}")

    def _on_competition_fast_heartbeat(self, payload: dict):
        stage = payload.get("stage", "?")
        elapsed = payload.get("elapsed_seconds", 0)
        self._status_bar.show_message(
            f"大图比赛快速路网生成中：{stage}，已用 {elapsed:.0f} 秒"
        )
        pct = int(payload.get("progress", 0) or 0)
        self._param_panel.update_competition_fast_progress(
            pct, f"{stage} · {elapsed:.0f}s"
        )

    def _on_competition_fast_progress(self, percent: int, message: str):
        self._param_panel.update_competition_fast_progress(percent, message)
        self._status_bar.show_message(f"大图比赛快速路网生成中：{message}")

    def _on_competition_fast_failed(self, message: str):
        # finished 也会到；这里只更新状态栏，弹窗由 finished 统一处理避免双弹窗
        self._status_bar.show_message(f"比赛快速路网: {message}")

    def _on_competition_fast_thread_finished(self):
        self._competition_fast_thread = None
        self._competition_fast_worker = None
        self._param_panel.set_competition_fast_running(False)

    def _on_competition_fast_finished(self, result):
        """主线程：仅根据 signal 结果刷新 graph UI（不做重计算）。"""
        from roadnet.competition_fast_roadnet import CompetitionFastResult

        self._param_panel.set_competition_fast_running(False)
        if not isinstance(result, CompetitionFastResult):
            return

        report = result.report or {}
        self._param_panel.update_competition_fast_status(
            work_w=int(report.get("work_width", 0) or 0),
            work_h=int(report.get("work_height", 0) or 0),
            max_side=int(report.get("competition_preview_max_side", 1500) or 1500),
            elapsed_s=report.get("elapsed_total_seconds"),
            ready=bool(result.ok),
        )

        if report.get("cancelled") or (result.error and "取消" in str(result.error)):
            QMessageBox.information(
                self, "已取消",
                "快速路网生成已取消。\n当前 final_graph 未被覆盖。",
            )
            self._status_bar.show_message("比赛快速路网已取消")
            return

        if not result.ok:
            QMessageBox.warning(
                self, "比赛快速路网未完成",
                (result.error or result.warning or "未知错误")
                + f"\n\n报告目录：\n{result.output_dir}\n"
                "可降低 preview_max_side，或切换应急手绘。",
            )
            return

        # 载入 original-pixel graph（轻量）
        try:
            self._graph_editor.load_draft(result.nodes_original, result.edges_original)
            self._render_graph_to_scene()
            self._layer_manager.show_layer("layer_final_graph")
            if hasattr(self, "_tool_panel"):
                self._tool_panel.set_layer_checkbox_state("layer_final_graph", True)
        except Exception as exc:
            QMessageBox.critical(self, "载入 graph 失败", str(exc))
            return

        # 轻量复制 final_graph.json（文件已由 worker 写好）
        try:
            src = result.final_graph_path
            if src and os.path.isfile(src):
                dest_dirs = []
                if self._large_image_project is not None:
                    dest_dirs.append(
                        os.path.join(self._large_image_project.project_dir, "graph")
                    )
                    dest_dirs.append(
                        os.path.join(self._large_image_project.project_dir, "outputs")
                    )
                dest_dirs.append(os.path.join(os.getcwd(), "outputs"))
                for d in dest_dirs:
                    os.makedirs(d, exist_ok=True)
                    shutil.copy2(src, os.path.join(d, "final_graph.json"))
        except Exception as exc:
            print(f"[CompetitionFast] copy warning: {exc}")

        msg = (
            f"比赛快速路网完成。\n\n"
            f"工作图：{report.get('work_width')}×{report.get('work_height')}\n"
            f"原图：{report.get('original_width')}×{report.get('original_height')}\n"
            f"节点/边：{report.get('graph_node_count')} / {report.get('graph_edge_count')}\n"
            f"耗时：{report.get('elapsed_total_seconds')} s\n"
            f"坐标：original image pixel\n"
            f"可用于任务点吸附/规划：是\n\n"
            f"输出：\n{result.output_dir}"
        )
        QMessageBox.information(self, "大图比赛快速路网生成", msg)
        self._status_bar.show_message(
            f"比赛快速路网完成: nodes={report.get('graph_node_count')}, "
            f"{report.get('elapsed_total_seconds')}s"
        )

    def _on_lowres_formal_mask(self):
        """低像素快速生成正式 Mask：回调只启动 Worker，立即 return。

        ★ 只生成 working_road_mask，不跑 skeleton / graph / 路径规划。
        ★ 不要求已有 Working Road Mask。
        """
        if getattr(self, "_lowres_formal_thread", None) is not None:
            if self._lowres_formal_thread.isRunning():
                self._on_cancel_lowres_formal_mask()
                return

        if not self._layer_manager.is_large_image_mode:
            QMessageBox.information(
                self, "小图模式",
                "低像素正式 Mask 仅用于大图模式。\n小图请继续使用原有提取流程。",
            )
            return
        image_path = self._layer_manager.image_path or ""
        if not image_path or not os.path.isfile(image_path):
            QMessageBox.warning(self, "无影像", "请先打开大图。")
            return
        if self._large_image_project is None:
            QMessageBox.warning(self, "无大图项目", "缺少 large_image_project。")
            return

        width, height = self._layer_manager.original_size
        pos_points = [
            (int(x), int(y)) for x, y in self._canvas.positive_points
            if 0 <= int(x) < width and 0 <= int(y) < height
        ]
        neg_points = [
            (int(x), int(y)) for x, y in self._canvas.negative_points
            if 0 <= int(x) < width and 0 <= int(y) < height
        ]
        if not pos_points:
            QMessageBox.warning(self, "正样本为空", "请先添加道路正样本（绿点）。")
            return

        config = self._param_panel.get_config()
        seg_cfg = config.get("segment", {})
        if seg_cfg.get("use_negative_samples", True) and not neg_points:
            QMessageBox.warning(self, "负样本为空", "请先添加非道路负样本（红叉）。")
            return

        max_side = int(seg_cfg.get("lowres_formal_max_side", 2500) or 2500)
        max_side = max(1000, min(4096, max_side))

        from datetime import datetime
        import threading as _threading
        from roadnet.lowres_formal_mask import LowresFormalMaskConfig
        from roadnet.lowres_formal_mask_worker import LowresFormalMaskWorker
        from PySide6.QtCore import Qt as _Qt

        clicked_time = datetime.now().isoformat(timespec="seconds")
        output_dir = os.path.join(
            self._large_image_project.project_dir,
            "lowres_formal_mask",
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, "无法创建输出目录", str(exc))
            return

        intensity = seg_cfg.get("intensity", "标准")
        h_margin = int(seg_cfg.get("h_margin", 6))
        s_margin = int(seg_cfg.get("s_margin", 25))
        v_margin = int(seg_cfg.get("v_margin", 30))
        if intensity == "保守":
            h_margin, s_margin, v_margin = 4, 15, 20
        elif intensity == "宽松":
            h_margin, s_margin, v_margin = 10, 40, 45

        ocv = config.get("opencv_extraction", {}) or {}
        fast_cfg = LowresFormalMaskConfig(
            max_side=max_side,
            open_kernel=3,
            close_kernel=5,
            remove_small_components=True,
            min_component_area=80,
            fill_holes=False,
            valid_area_only=True,
            black_threshold=int(seg_cfg.get("black_threshold", 10)),
            h_margin=h_margin,
            s_margin=s_margin,
            v_margin=v_margin,
            lab_margin=int(seg_cfg.get("lab_margin", 12)),
            use_negative_samples=bool(seg_cfg.get("use_negative_samples", True)),
            blur_kernel=int(seg_cfg.get("preview_blur_kernel", seg_cfg.get("blur_kernel", 3))),
            mode=str(seg_cfg.get("mode", "combined")),
            combine_method=str(seg_cfg.get("combine_method", "and")),
            sample_radius=int(seg_cfg.get("sample_radius", 3)),
            use_roi=bool(ocv.get("use_roi", True)),
            use_ignore=bool(ocv.get("use_ignore", True)),
        )

        def polygons_to_global(qpolygons):
            result = []
            for polygon in qpolygons:
                global_points = [[float(p.x()), float(p.y())] for p in polygon]
                if len(global_points) >= 3:
                    result.append(global_points)
            return result

        roi_polygons = polygons_to_global(self._canvas.get_roi_polygons())
        ignore_polygons = polygons_to_global(self._canvas.get_ignore_polygons())

        thread = QThread(self)
        worker = LowresFormalMaskWorker(
            image_path=image_path,
            pos_points=pos_points,
            neg_points=neg_points,
            output_dir=output_dir,
            config=fast_cfg,
            roi_polygons=roi_polygons,
            ignore_polygons=ignore_polygons,
            main_thread_id=int(_threading.get_ident()),
            button_clicked_time=clicked_time,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress_changed.connect(
            self._on_lowres_formal_progress, _Qt.ConnectionType.QueuedConnection
        )
        worker.finished.connect(
            self._on_lowres_formal_finished, _Qt.ConnectionType.QueuedConnection
        )
        worker.error_occurred.connect(
            self._on_lowres_formal_failed, _Qt.ConnectionType.QueuedConnection
        )
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_lowres_formal_thread_finished)
        thread.finished.connect(thread.deleteLater)

        self._lowres_formal_thread = thread
        self._lowres_formal_worker = worker
        self._param_panel.set_lowres_formal_running(True)
        self._param_panel.update_lowres_formal_status(
            mask_source="lowres_formal_mask(running)",
            formal_ready=False,
            original_w=width,
            original_h=height,
        )
        self._status_bar.show_message(
            f"低像素正式 Mask 生成中… max_side={max_side}"
        )
        thread.start()
        return  # ★ 立即返回，不阻塞 UI

    def _on_cancel_lowres_formal_mask(self):
        worker = getattr(self, "_lowres_formal_worker", None)
        if worker is not None:
            worker.request_cancel()
            self._status_bar.show_message("正在取消低像素 Mask 生成…")
            self._param_panel.update_lowres_formal_progress(0, "取消中…")

    def _on_lowres_formal_progress(self, percent: int, message: str):
        self._param_panel.update_lowres_formal_progress(percent, message)
        self._status_bar.show_message(f"低像素正式 Mask：{message}")

    def _on_lowres_formal_failed(self, message: str):
        self._status_bar.show_message(f"低像素正式 Mask: {message}")

    def _on_lowres_formal_thread_finished(self):
        self._lowres_formal_thread = None
        self._lowres_formal_worker = None
        self._param_panel.set_lowres_formal_running(False)

    def _on_lowres_formal_finished(self, result):
        """主线程：注册 working_road_mask，不生成 graph。"""
        from roadnet.lowres_formal_mask import LowresFormalMaskResult

        self._param_panel.set_lowres_formal_running(False)
        if not isinstance(result, LowresFormalMaskResult):
            return

        report = result.report or {}
        self._param_panel.update_lowres_formal_status(
            mask_source="lowres_formal_mask",
            formal_ready=bool(result.ok),
            lowres_w=int(report.get("lowres_width", 0) or 0),
            lowres_h=int(report.get("lowres_height", 0) or 0),
            original_w=int(report.get("original_width", 0) or 0),
            original_h=int(report.get("original_height", 0) or 0),
            scale_x=float(report.get("scale_x", 0) or 0),
            scale_y=float(report.get("scale_y", 0) or 0),
        )

        if report.get("cancelled") or (result.error and "取消" in str(result.error)):
            QMessageBox.information(
                self, "已取消",
                "低像素正式 Mask 生成已取消。\n当前 working_road_mask 未被覆盖。",
            )
            self._status_bar.show_message("低像素正式 Mask 已取消")
            return

        if not result.ok:
            warn = result.error or result.warning or "未知错误"
            if result.timed_out or "超时" in warn or (
                float(report.get("elapsed_total_seconds", 0) or 0) > 120
                and int(report.get("max_side", 0) or 0) >= 3000
            ):
                warn += "\n\n建议降低「低像素 Mask 最长边」(max_side)。"
            QMessageBox.warning(
                self, "低像素正式 Mask 未完成",
                warn + f"\n\n报告目录：\n{result.output_dir}",
            )
            return

        # 加载 full-size mask 并注册为 working_road_mask
        try:
            mask = result.working_mask
            if mask is None:
                import cv2
                path = result.working_mask_path
                if not path or not os.path.isfile(path):
                    raise FileNotFoundError("working_road_mask.png 不存在")
                mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError("无法读取 working_road_mask")
            mask = (np.asarray(mask) > 0).astype(np.uint8) * 255

            self._working_mask_source = "lowres_formal_mask"
            self._mask_edit_base = "lowres_formal_mask"
            # 新正式 mask 生成后，清除过期 final，避免骨架仍优先读旧 final
            self._persist_working_mask(mask, clear_cleaned=True, clear_final=True)

            project = self._large_image_project
            if project is not None:
                project.mask_source = "lowres_formal_mask"
                project.mask_edit_base = "lowres_formal_mask"
                project.lowres_work_image_path = result.lowres_work_image_path or ""
                project.lowres_road_mask_path = result.lowres_road_mask_path or ""
                project.lowres_width = int(report.get("lowres_width", 0) or 0)
                project.lowres_height = int(report.get("lowres_height", 0) or 0)
                project.scale_x = float(report.get("scale_x", 0) or 0)
                project.scale_y = float(report.get("scale_y", 0) or 0)
                project.formal_ready = True
                project.preview_only = False
                project.mask_dirty = False
                # 同步 working 路径（_persist_working_mask 已写）
                project.save()

            self._formal_mask_meta = {
                **(self._formal_mask_meta or {}),
                "mask_type": "lowres_formal_mask",
                "mask_source": "lowres_formal_mask",
                "formal_ready": True,
                "preview_only": False,
                "lowres_width": report.get("lowres_width"),
                "lowres_height": report.get("lowres_height"),
                "scale_x": report.get("scale_x"),
                "scale_y": report.get("scale_y"),
                "interpolation": "INTER_NEAREST",
            }

            self._layer_manager.show_layer("layer_road_mask")
            if hasattr(self, "_tool_panel"):
                self._tool_panel.set_layer_checkbox_state("layer_road_mask", True)
            self._sync_layer_checkboxes()
            self._canvas.refresh_scene()
            self._canvas.viewport().update()
        except Exception as exc:
            QMessageBox.critical(self, "注册 working mask 失败", str(exc))
            return

        elapsed = report.get("elapsed_total_seconds")
        extra = ""
        if elapsed is not None and float(elapsed) > 120 and max(
            int(report.get("lowres_width", 0) or 0),
            int(report.get("lowres_height", 0) or 0),
        ) >= 2800:
            extra = "\n\n提示：本次较慢，下次可降低 max_side。"

        self._status_bar.show_message(
            "已通过低像素图生成正式 working Road Mask，可进入区域修正或骨架生成。"
        )
        QMessageBox.information(
            self, "低像素正式 Mask 完成",
            "已通过低像素图生成正式 working Road Mask，可进入区域修正或骨架生成。\n\n"
            f"来源：lowres_formal_mask\n"
            f"工作分辨率：{report.get('lowres_width')}×{report.get('lowres_height')}\n"
            f"原图分辨率：{report.get('original_width')}×{report.get('original_height')}\n"
            f"scale_x / scale_y：{report.get('scale_x'):.4f} / {report.get('scale_y'):.4f}\n"
            f"耗时：{elapsed} s\n"
            f"输出：\n{result.output_dir}"
            + extra,
        )

    def _on_preview_seg_progress(self, percent: int, message: str):
        self._param_panel.update_preview_seg_progress(percent, message)
        self._status_bar.show_message(f"快速预览: {message}")

    def _on_preview_seg_finished(self, result):
        """快速预览分割完成：不覆盖正式 mask，只加载到 layer_preview_segmentation。

        1. 检查输出文件（preview_mask.png / preview_seg_overlay.png）
        2. 加载 overlay 到独立图层
        3. 设置图层可见
        4. 刷新画布
        5. 状态栏显示
        6. 弹窗（含打开目录/查看文件按钮）
        """
        from roadnet.preview_segmentation import PreviewSegmentationResult
        import traceback

        if not isinstance(result, PreviewSegmentationResult):
            return

        output_dir = result.output_dir
        mask = result.preview_mask
        overlay_rgb = result.overlay_rgb

        # ── 阶段 1: 文件检查 ─────────────────────────────────────
        mask_path = os.path.join(output_dir, "preview_mask.png")
        overlay_path = os.path.join(output_dir, "preview_seg_overlay.png")
        report_path = os.path.join(output_dir, "preview_segmentation_report.json")
        error_log_path = os.path.join(output_dir, "preview_segmentation_error.log")

        mask_file_exists = os.path.isfile(mask_path)
        overlay_file_exists = os.path.isfile(overlay_path)

        lm = self._layer_manager
        preview_w, preview_h = lm.image_size if lm.has_image() else (0, 0)

        print(f"[PreviewSeg] ── 完成回调 ──")
        print(f"[PreviewSeg] output_dir = {output_dir}")
        print(f"[PreviewSeg] preview_mask exists = {mask_file_exists}")
        print(f"[PreviewSeg] preview_overlay exists = {overlay_file_exists}")
        print(f"[PreviewSeg] preview image size = {preview_w} x {preview_h}")
        if mask is not None:
            print(f"[PreviewSeg] preview_mask in-memory size = {mask.shape[1]} x {mask.shape[0]}")
        if overlay_rgb is not None:
            print(f"[PreviewSeg] overlay in-memory size = {overlay_rgb.shape[1]} x {overlay_rgb.shape[0]}")

        # ── 阶段 2: 结果检查 ─────────────────────────────────────
        if mask is None or mask.size == 0:
            self._status_bar.show_message("快速预览分割: 结果为空")
            QMessageBox.warning(
                self, "快速预览分割",
                "分割结果为空（mask 为 None 或 size=0）。\n\n"
                "可能原因：\n"
                "1. 样本点颜色与道路颜色差异过大\n"
                "2. 分割参数过于保守\n"
                f"3. 查看错误日志: {error_log_path}"
            )
            return

        # ── 阶段 3: 确定显示数据 ──────────────────────────────────
        # 优先使用 overlay（预渲染的 RGB 叠加图）；回退到 mask
        display_data = None
        display_source = ""

        if overlay_rgb is not None:
            display_data = overlay_rgb
            display_source = "overlay_rgb (memory)"
        elif overlay_file_exists:
            try:
                overlay_loaded = cv2.imread(overlay_path, cv2.IMREAD_COLOR)
                if overlay_loaded is not None:
                    display_data = cv2.cvtColor(overlay_loaded, cv2.COLOR_BGR2RGB)
                    display_source = "overlay file (disk)"
            except Exception as e:
                print(f"[PreviewSeg] ⚠ 无法从磁盘加载 overlay: {e}")

        if display_data is None and mask is not None:
            # 回退：用 mask 作为 2D 数据，layer_manager 会构建纯色 overlay
            display_data = mask
            display_source = "mask (2D binary)"

        if display_data is None:
            self._status_bar.show_message("快速预览分割: 无有效显示数据")
            QMessageBox.warning(
                self, "快速预览分割",
                "分割完成但无有效数据可显示。\n\n"
                f"overlay 文件存在: {overlay_file_exists}\n"
                f"mask 文件存在: {mask_file_exists}\n"
                f"查看错误日志: {error_log_path}"
            )
            return

        # ── 阶段 4: 尺寸检查 ─────────────────────────────────────
        d_h, d_w = display_data.shape[:2]
        preview_match = (d_w == preview_w and d_h == preview_h)
        print(f"[PreviewSeg] display_data size = {d_w} x {d_h} "
              f"({'✓ 匹配预览' if preview_match else '✗ 不匹配! 预览=' + str(preview_w) + 'x' + str(preview_h)}")

        if not preview_match and preview_w > 0:
            print(f"[PreviewSeg] ⚠ 显示数据尺寸 ({d_w}x{d_h}) 与预览尺寸 ({preview_w}x{preview_h}) 不匹配，"
                  f"将使用预览尺寸缩放。")

        # ── 阶段 5: 设置图层数据 ──────────────────────────────────
        layer_name = "layer_preview_segmentation"
        try:
            lm.set_layer_data(layer_name, display_data)
            print(f"[PreviewSeg] layer_data set from {display_source}")
        except Exception as e:
            print(f"[PreviewSeg] ✗ set_layer_data 失败: {e}")
            self._status_bar.show_message(f"快速预览分割: 设置图层数据失败 - {e}")
            QMessageBox.warning(
                self, "快速预览分割",
                f"图层数据设置失败:\n{e}\n\n"
                f"traceback:\n{traceback.format_exc()}"
            )
            return

        # ── 阶段 6: 设置透明度和可见性 ──────────────────────────
        alpha_float = float(self._param_panel.get_config().get("segment", {}).get(
            "preview_seg_alpha", 0.45
        ))
        alpha_int = max(1, min(255, int(alpha_float * 255)))
        lm.set_layer_opacity(layer_name, alpha_int)
        print(f"[PreviewSeg] layer opacity = {alpha_int} (alpha={alpha_float:.2f})")

        lm.show_layer(layer_name)
        layer_visible = lm.is_layer_visible(layer_name)
        print(f"[PreviewSeg] layer_visible = {layer_visible}")
        print(f"[PreviewSeg] import_success = True")

        # ── 阶段 7: 刷新画布 ─────────────────────────────────────
        self._canvas.update_overlay(layer_name)
        self._canvas.refresh_scene()
        self._canvas.viewport().update()
        print(f"[PreviewSeg] Canvas refreshed")

        # ── 阶段 8: 状态栏 ───────────────────────────────────────
        elapsed = result.report.get("elapsed_seconds", 0.0)
        cache_note = " (缓存)" if result.cache_used else ""
        nonzero_ratio = round((mask > 0).sum() / max(1, mask.size) * 100, 1)
        self._status_bar.show_message(
            f"快速预览分割完成，结果已显示。{cache_note} 耗时 {elapsed:.1f}s, "
            f"道路占比 {nonzero_ratio}%"
        )

        # ── 阶段 9: 日志 ─────────────────────────────────────────
        report = result.report
        print(f"[PreviewSeg] 完成: elapsed={elapsed:.3f}s, "
              f"cache={result.cache_used}, "
              f"steps={report.get('operation_steps', [])}")

        # ── 阶段 10: 完成后弹窗（含操作按钮）────────────────────
        msg = QMessageBox(self)
        msg.setWindowTitle("快速预览分割完成")
        msg.setIcon(QMessageBox.Information)
        msg.setText(
            f"快速预览分割完成。\n\n"
            f"耗时: {elapsed:.1f}s{cache_note}\n"
            f"道路占比: {nonzero_ratio}%\n"
            f"输出目录: {output_dir}\n"
            f"preview_mask exists: {mask_file_exists}\n"
            f"preview_overlay exists: {overlay_file_exists}\n"
            f"图层可见: {layer_visible}"
        )

        # 添加操作按钮
        btn_open_dir = msg.addButton("打开输出目录", QMessageBox.ActionRole)
        btn_show_mask = msg.addButton("查看 preview_mask", QMessageBox.ActionRole)
        btn_show_overlay = msg.addButton("查看 preview_overlay", QMessageBox.ActionRole)
        btn_view_report = msg.addButton("查看 report", QMessageBox.ActionRole)
        btn_view_error = msg.addButton("查看 error log", QMessageBox.ActionRole)
        btn_close = msg.addButton(QMessageBox.Ok)

        msg.exec()

        clicked = msg.clickedButton()
        if clicked == btn_open_dir:
            os.startfile(output_dir)
        elif clicked == btn_show_mask and mask_file_exists:
            os.startfile(mask_path)
        elif clicked == btn_show_overlay and overlay_file_exists:
            os.startfile(overlay_path)
        elif clicked == btn_view_report and os.path.isfile(report_path):
            os.startfile(report_path)
        elif clicked == btn_view_error and os.path.isfile(error_log_path):
            os.startfile(error_log_path)

    def _on_preview_seg_failed(self, stage: str, message: str, error_log_path: str):
        self._status_bar.show_message(f"快速预览失败 ({stage}): {message}")
        QMessageBox.critical(
            self, "快速预览分割失败",
            f"阶段: {stage}\n"
            f"错误: {message}\n\n"
            f"不影响当前项目状态。\n"
            f"日志: {error_log_path}"
        )

    def _on_preview_seg_cancelled(self, message: str):
        self._status_bar.show_message(f"快速预览已取消: {message}")

    def _on_preview_seg_thread_finished(self):
        self._param_panel.set_preview_seg_running(False)
        self._preview_seg_worker = None
        self._preview_seg_thread = None

    def _cancel_segmentation(self):
        # 优先取消正式提取
        fw = self._formal_extraction_worker
        ft = self._formal_extraction_thread
        if fw is not None and ft is not None and ft.isRunning():
            fw.cancel()
            ft.requestInterruption()
            self._status_bar.show_message("正在取消正式道路提取…")
            return

        # 优先取消快速预览
        pw = self._preview_seg_worker
        pt = self._preview_seg_thread
        if pw is not None and pt is not None and pt.isRunning():
            pw.cancel()
            pt.requestInterruption()
            self._status_bar.show_message("正在取消快速预览分割…")
            return

        worker = self._segmentation_worker
        thread = self._segmentation_thread
        if worker is None or thread is None or not thread.isRunning():
            return
        worker.cancel()
        thread.requestInterruption()
        self._param_panel.set_segmentation_running(True, cancelling=True)
        self._status_bar.show_message("正在取消分割；原 mask 将保持不变…")

    def _on_segmentation_progress(self, percent, current, total, message):
        self._param_panel.update_segmentation_progress(
            percent, current, total, message
        )
        self._status_bar.show_message(
            f"分割进度 {percent}% · tile {current}/{total}"
        )

    def _on_segmentation_finished(self, result):
        # This slot runs on the GUI thread. Commit only a fully completed mask.
        current_ref = (
            os.path.abspath(self._layer_manager.image_path)
            if self._layer_manager.is_large_image_mode
            else self._layer_manager.full_image_rgb
        )
        changed = (
            current_ref != self._segmentation_image_ref
            if isinstance(current_ref, str)
            else current_ref is not self._segmentation_image_ref
        )
        if changed:
            self._status_bar.show_message("影像已改变，已丢弃旧影像的分割结果")
            return
        if not self._layer_manager.is_large_image_mode:
            self._history.push_state("segment")
        # 传统分割结果非 OpenCV 正式提取，清除专用元数据。
        self._formal_mask_meta = {}
        mask = result.processed_mask
        self._valid_image_mask = result.valid_image_mask
        self._valid_mask_report = dict(result.valid_mask_report)
        self._layer_manager.set_layer_data(
            "valid_image_mask", result.valid_image_mask
        )
        self._layer_manager.hide_layer("valid_image_mask")
        preview_mask = getattr(result, "preview_mask", None)
        self._layer_manager.set_layer_data("mask", mask, preview_data=preview_mask)
        if self._large_image_project is not None and self._layer_manager.is_large_image_mode:
            global_path = os.path.join(result.output_dir, "global_road_mask.png")
            self._large_image_project.global_mask_path = global_path
            self._large_image_project.valid_image_mask_path = os.path.join(
                result.output_dir, "valid_image_mask.png"
            )
            project_valid = os.path.join(
                self._large_image_project.project_dir, "valid_image_mask.png"
            )
            shutil.copy2(self._large_image_project.valid_image_mask_path, project_valid)
            self._large_image_project.valid_image_mask_path = project_valid
            valid_report_path = os.path.join(result.output_dir, "valid_mask_report.json")
            if os.path.isfile(valid_report_path):
                shutil.copy2(
                    valid_report_path,
                    os.path.join(self._large_image_project.project_dir, "valid_mask_report.json"),
                )
            self._large_image_project.save()
            self._project_manager.data.global_mask_path = global_path
            self._project_manager.mark_dirty()
        self._clear_skeleton_state(clear_layer=True)
        self._canvas.refresh_scene()
        self._canvas.viewport().update()
        total = mask.size
        road_px = int((mask > 0).sum())
        ratio = road_px / total if total else 0.0
        self._status_bar.update_road_ratio(ratio)
        elapsed = result.report.get("elapsed_seconds", 0.0)
        self._status_bar.show_message(
            f"分割完成: {result.report['tile_count']} tiles, 用时 {elapsed:.1f}s, "
            f"道路占比 {ratio*100:.1f}%"
        )

    def _on_segmentation_failed(self, stage, message, error_log_path):
        self._status_bar.show_message(f"分割失败于 {stage}；原 mask 未改变")
        QMessageBox.critical(
            self, "分割失败",
            f"失败阶段: {stage}\n错误: {message}\n\n"
            f"原 mask、样本点、ROI 和 Ignore 均未改变。\n"
            f"错误日志: {error_log_path}"
        )

    def _on_segmentation_cancelled(self, message):
        self._status_bar.show_message("分割已取消；原 mask 未改变")

    def _on_segmentation_thread_finished(self):
        self._param_panel.set_segmentation_running(False)
        for nav_button in self._nav_buttons.values():
            nav_button.setEnabled(True)
        self._btn_pipeline.setEnabled(True)
        self._segmentation_worker = None
        self._segmentation_thread = None
        self._segmentation_image_ref = None

    # ===================================================================
    # 正式道路提取（ROI / 全图）
    # ===================================================================

    def _on_extract_roi(self):
        """ROI 正式道路提取：只处理 ROI 覆盖的 tile。"""
        if self._canvas.get_enabled_roi_count() == 0:
            self._status_bar.show_message("未设置 ROI，已取消 ROI 正式提取。")
            dialog = QMessageBox(self)
            dialog.setIcon(QMessageBox.Icon.Warning)
            dialog.setWindowTitle("需要 ROI 区域")
            dialog.setText(
                "当前没有 ROI 区域。\n"
                "ROI 正式提取只处理 ROI 覆盖的 tile。\n"
                "请先绘制 ROI，或选择全图正式提取。"
            )
            draw_button = dialog.addButton(
                "开始绘制 ROI", QMessageBox.ButtonRole.AcceptRole
            )
            view_button = dialog.addButton(
                "使用当前视野作为 ROI", QMessageBox.ButtonRole.ActionRole
            )
            dialog.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            dialog.exec()
            if dialog.clickedButton() is draw_button:
                self._begin_roi_drawing()
            elif dialog.clickedButton() is view_button:
                if self._use_current_view_as_roi():
                    self._start_formal_extraction(mode="roi")
            return
        self._start_formal_extraction(mode="roi")

    def _on_extract_full(self):
        """全图正式道路提取：处理全部 valid tile。"""
        self._start_formal_extraction(mode="full")

    # ===================================================================
    # 大图 OpenCV 正式提取（比赛默认流程，不使用 SAMRoadPlus）
    # ===================================================================

    def _require_large_image_for_opencv(self) -> bool:
        """OpenCV 正式提取仅用于大图模式；小图流程保持不变。"""
        if not self._layer_manager.is_large_image_mode:
            QMessageBox.information(
                self, "小图模式",
                "大图 OpenCV 正式提取仅用于大图模式。\n"
                "小图请继续使用原有分割 / SAM-Road 流程（不受影响）。"
            )
            return False
        if not self._layer_manager.image_path:
            QMessageBox.warning(self, "图像为空", "大图项目缺少原始影像路径。")
            return False
        return True

    def _collect_roi_ignore_task_original(self):
        """收集 original image pixel 坐标下的 ROI / Ignore / 任务点。"""
        def polys(qpolygons):
            out = []
            for polygon in qpolygons:
                pts = [[float(p.x()), float(p.y())] for p in polygon]
                if len(pts) >= 3:
                    out.append(pts)
            return out
        rois = polys(self._canvas.get_roi_polygons())
        ignores = polys(self._canvas.get_ignore_polygons())
        task_points = []
        for tp in getattr(self, "_task_points", []) or []:
            px = getattr(tp, "pixel_x", None)
            py = getattr(tp, "pixel_y", None)
            if px is not None and py is not None:
                task_points.append([float(px), float(py)])
        return rois, ignores, task_points

    def _apply_seed_width_settings_to_canvas(self):
        settings = {}
        if hasattr(self._param_panel, "get_seed_width_settings"):
            settings = self._param_panel.get_seed_width_settings()
        gsd = None
        if self._geo_calibration is not None:
            gsd = getattr(self._geo_calibration, "pixel_resolution_estimated_m", None)
        if not gsd:
            gsd = getattr(self._project_manager.data, "pixel_resolution_m", None)
        if hasattr(self._canvas, "set_seed_width_settings"):
            self._canvas.set_seed_width_settings(
                width_mode=settings.get("width_mode", "normal"),
                road_width_m=float(settings.get("road_width_m", 8.0)),
                road_radius_px=settings.get("road_radius_px"),
                gsd_m_per_px=float(gsd) if gsd else None,
                continuous_two_point=bool(settings.get("continuous_two_point", True)),
            )
        # snap candidates: existing seed ends + graph nodes + task points
        cands = []
        for stroke in (self._canvas.get_main_road_seed_strokes() or []):
            if stroke:
                cands.append(stroke[0])
                cands.append(stroke[-1])
        ge = getattr(self, "_graph_editor", None)
        if ge is not None:
            for node in getattr(ge, "nodes", []) or []:
                x = node.get("x") if isinstance(node, dict) else getattr(node, "x", None)
                y = node.get("y") if isinstance(node, dict) else getattr(node, "y", None)
                if x is not None and y is not None:
                    cands.append((float(x), float(y)))
        for tp in getattr(self, "_task_points", []) or []:
            px = getattr(tp, "pixel_x", None)
            py = getattr(tp, "pixel_y", None)
            if px is not None and py is not None:
                cands.append((float(px), float(py)))
        if hasattr(self._canvas, "set_seed_snap_candidates"):
            self._canvas.set_seed_snap_candidates(cands)

    def _begin_seed_drawing(self, mode: str = "freehand"):
        """进入主路种子线绘制（大图模式）。"""
        if not self._layer_manager.is_large_image_mode:
            QMessageBox.information(self, "小图模式", "主路种子线仅用于大图主路修复。")
            return
        if not self._layer_manager.has_image():
            QMessageBox.warning(self, "提示", "请先打开影像。")
            return
        self.set_stage("edit")
        self._apply_seed_width_settings_to_canvas()
        if hasattr(self._canvas, "set_seed_draw_mode"):
            self._canvas.set_seed_draw_mode(mode)
        self.set_tool("main_road_seed")
        self._layer_manager.show_layer("layer_main_road_seed")
        if "layer_road_ribbon_preview" in getattr(self._layer_manager, "_layers", {}):
            self._layer_manager.show_layer("layer_road_ribbon_preview")
        self._sync_layer_checkboxes()
        tips = {
            "two_point": "请选择主路线起点（Shift=角度约束，Esc取消起点）",
            "polyline": "多点主路线：左键加点，双击/右键结束，Backspace删点",
            "freehand": "自由绘：按住左键拖动绘制主路种子线",
        }
        self._status_bar.show_message(tips.get(mode, tips["freehand"]))

    def _begin_seed_drawing_two_point(self):
        self._begin_seed_drawing("two_point")

    def _begin_seed_drawing_polyline(self):
        self._begin_seed_drawing("polyline")

    def _undo_last_seed_stroke(self):
        if not self._layer_manager.is_large_image_mode:
            return
        if hasattr(self._canvas, "undo_last_seed_stroke"):
            ok = self._canvas.undo_last_seed_stroke()
            self._save_main_road_seed_strokes()
            if ok:
                self._status_bar.show_message("已撤销上一条主路种子线")
            else:
                self._status_bar.show_message("没有可撤销的种子线")

    def _seed_use_view(self):
        """将当前视野矩形设为主路修复范围（corridor 来源之一）。"""
        if not self._layer_manager.is_large_image_mode:
            QMessageBox.information(self, "小图模式", "该功能仅用于大图主路修复。")
            return
        x0, y0, x1, y1 = self._canvas.get_visible_image_rect()
        if x1 <= x0 or y1 <= y0:
            QMessageBox.warning(self, "无法设置范围", "当前画布没有有效可见区域。")
            return
        self._main_road_view_rect = [float(x0), float(y0), float(x1), float(y1)]
        self._status_bar.show_message(
            "已将当前视野设为主路修复范围，执行主路修复 / 预览桥接时生效。"
        )

    def _seed_use_tasks(self):
        """提示任务点将用于生成 corridor。"""
        if not self._layer_manager.is_large_image_mode:
            QMessageBox.information(self, "小图模式", "该功能仅用于大图主路修复。")
            return
        _, _, task_points = self._collect_roi_ignore_task_original()
        if not task_points:
            QMessageBox.warning(
                self, "无任务点",
                "当前没有任务点。请先导入任务点，再用其生成 corridor。"
            )
            return
        self._main_road_use_tasks = True
        QMessageBox.information(
            self, "任务点 corridor",
            f"将使用 {len(task_points)} 个任务点周围 buffer 作为 corridor 之一，"
            "执行主路修复时自动生效。"
        )

    def _clear_main_road_seeds(self):
        """清空主路种子线与修复范围。"""
        self._canvas.clear_main_road_seeds()
        self._main_road_view_rect = None
        self._canvas.clear_corridor_overlay()
        self._status_bar.show_message("已清空主路种子线与修复范围。")

    @property
    def main_road_seed_strokes(self):
        """原图像素坐标下的主路种子线（只读副本）。"""
        if hasattr(self._canvas, "get_main_road_seed_strokes"):
            return self._canvas.get_main_road_seed_strokes()
        return []

    def _save_main_road_seed_strokes(self) -> Optional[str]:
        """将主路种子线保存到大图项目目录，便于下次加载。"""
        if self._large_image_project is None:
            return None
        from roadnet.main_road_seed import serialize_seed_strokes
        if hasattr(self._canvas, "get_main_road_seed_stroke_dicts"):
            strokes = self._canvas.get_main_road_seed_stroke_dicts()
        else:
            strokes = self.main_road_seed_strokes
        out_dir = Path(self._large_image_project.project_dir) / "masks"
        out_dir.mkdir(parents=True, exist_ok=True)
        # 正式文件名：road_seed_strokes.json；同时写兼容名
        path = out_dir / "road_seed_strokes.json"
        legacy = out_dir / "main_road_seed_strokes.json"
        payload = serialize_seed_strokes(strokes)
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        with path.open("w", encoding="utf-8") as f:
            f.write(text)
        with legacy.open("w", encoding="utf-8") as f:
            f.write(text)
        self._large_image_project.main_road_seed_strokes_path = str(path)
        try:
            self._large_image_project.save()
        except Exception:
            pass
        return str(path)

    def _load_main_road_seed_strokes(self) -> int:
        """从大图项目加载主路种子线；返回笔数。"""
        if self._large_image_project is None:
            return 0
        from roadnet.main_road_seed import deserialize_seed_strokes
        path = getattr(self._large_image_project, "main_road_seed_strokes_path", "") or ""
        candidates = []
        if path:
            candidates.append(path)
        base = Path(self._large_image_project.project_dir) / "masks"
        candidates.extend([
            str(base / "road_seed_strokes.json"),
            str(base / "main_road_seed_strokes.json"),
        ])
        payload = None
        for p in candidates:
            if p and os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    break
                except Exception:
                    continue
        if not payload:
            return 0
        try:
            strokes = deserialize_seed_strokes(payload)
            if hasattr(self._canvas, "set_main_road_seed_stroke_dicts"):
                self._canvas.set_main_road_seed_stroke_dicts(strokes, emit=False)
            else:
                self._canvas.clear_main_road_seeds()
                for stroke in strokes:
                    pts = [(p["x"], p["y"]) for p in stroke.get("points") or []]
                    if pts and hasattr(self._canvas, "add_main_road_seed_stroke"):
                        self._canvas.add_main_road_seed_stroke(pts)
            count = self._canvas.get_main_road_seed_count()
            if hasattr(self._param_panel, "update_main_road_seed_count"):
                self._param_panel.update_main_road_seed_count(count)
            return count
        except Exception as exc:
            print(f"[SeedStrokes] 加载失败: {exc}")
            return 0

    def _on_seed_rebuild_mask(self):
        """根据种子线生成 road ribbon 并重建 working/final mask（仅大图）。"""
        if not self._layer_manager.is_large_image_mode:
            QMessageBox.information(self, "小图模式", "根据种子线重建 Mask 仅用于大图流程。")
            return
        if hasattr(self._canvas, "get_main_road_seed_stroke_dicts"):
            seeds = self._canvas.get_main_road_seed_stroke_dicts()
        else:
            seeds = self.main_road_seed_strokes
        if not seeds:
            reply = QMessageBox.warning(
                self, "缺少主路种子线",
                "请先绘制两点/多点主路种子线，再重建 Mask。\n\n是否现在开始两点绘制？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._begin_seed_drawing_two_point()
            return

        mask = None
        working_path = getattr(self, "_working_road_mask_path", None) or ""
        if self._large_image_project is not None:
            working_path = working_path or getattr(
                self._large_image_project, "working_road_mask_path", ""
            ) or ""
        if working_path and os.path.isfile(working_path) and not self._is_preview_mask_path(working_path):
            try:
                mask = cv2.imread(working_path, cv2.IMREAD_GRAYSCALE)
            except Exception:
                mask = None
        if mask is None:
            working = self._layer_manager.get_layer_data("mask")
            if isinstance(working, np.ndarray) and working.size > 0:
                mask = working
        if mask is None:
            # 允许纯种子生成：空 mask
            oh = int(getattr(self._layer_manager, "full_image_height", 0) or 0)
            ow = int(getattr(self._layer_manager, "full_image_width", 0) or 0)
            arr = getattr(self._layer_manager, "full_image_rgb", None)
            if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                oh, ow = arr.shape[:2]
            if oh <= 0 or ow <= 0:
                QMessageBox.warning(self, "重建 Mask", "无法确定原图尺寸。")
                return
            mask = np.zeros((oh, ow), dtype=np.uint8)

        _, ignores, _ = self._collect_roi_ignore_task_original()
        self._apply_seed_width_settings_to_canvas()
        self._save_main_road_seed_strokes()
        self._history.push_state("seed_rebuild_mask")

        try:
            from roadnet.main_road_seed import rebuild_mask_from_seed_ribbons
            repaired, report = rebuild_mask_from_seed_ribbons(
                mask, seeds, ignore_polygons=ignores or None,
            )
        except Exception as exc:
            QMessageBox.critical(self, "重建 Mask 失败", str(exc))
            return

        # Persist as final_edited_mask + update working
        try:
            out_dir = None
            if self._large_image_project is not None:
                out_dir = Path(self._large_image_project.project_dir) / "masks"
                out_dir.mkdir(parents=True, exist_ok=True)
            else:
                out_dir = Path(os.getcwd()) / "outputs" / "masks"
                out_dir.mkdir(parents=True, exist_ok=True)

            final_path = out_dir / "final_edited_mask.png"
            preview_path = out_dir / "final_edited_mask_preview.png"
            working_out = out_dir / "working_road_mask.png"
            cv2.imwrite(str(final_path), repaired)
            cv2.imwrite(str(working_out), repaired)

            # preview downscale
            h, w = repaired.shape[:2]
            max_side = 2000
            scale = min(1.0, max_side / float(max(h, w)))
            if scale < 1.0:
                pw, ph = max(1, int(w * scale)), max(1, int(h * scale))
                preview = cv2.resize(repaired, (pw, ph), interpolation=cv2.INTER_NEAREST)
            else:
                preview = repaired
            cv2.imwrite(str(preview_path), preview)

            # Register into layer / working state
            if hasattr(self, "_persist_working_mask"):
                try:
                    self._persist_working_mask(repaired, save_as_final=True)
                except Exception as persist_exc:
                    print(f"[SeedRebuild] persist warning: {persist_exc}")
                    cv2.imwrite(str(working_out), repaired)
            self._layer_manager.set_layer_data("mask", repaired)
            self._layer_manager.set_layer_data("layer_final_edited_mask", repaired)
            self._working_mask_source = "final_edited_mask"
            self._working_road_mask_path = str(working_out)
            if self._large_image_project is not None:
                self._large_image_project.working_road_mask_path = str(working_out)
                try:
                    self._large_image_project.final_edited_mask_path = str(final_path)
                except Exception:
                    pass
                try:
                    self._large_image_project.save()
                except Exception:
                    pass

            if hasattr(self._canvas, "refresh_road_ribbon_preview"):
                self._canvas.refresh_road_ribbon_preview()

            QMessageBox.information(
                self, "根据种子线重建 Mask",
                f"已合并 road ribbon 并写出 final_edited_mask。\n\n"
                f"种子线：{report.get('seed_stroke_count')}\n"
                f"保留连通域：{report.get('kept_component_count')}\n"
                f"删除误检连通域：{report.get('removed_component_count')}\n"
                f"输出：\n{final_path}\n{preview_path}",
            )
            self._status_bar.show_message("已根据种子线重建 Mask")
        except Exception as exc:
            QMessageBox.critical(self, "保存重建 Mask 失败", str(exc))

    def _on_seed_clean_mask(self):
        """根据主路种子线清理当前 working mask（仅大图）。"""
        if not self._layer_manager.is_large_image_mode:
            QMessageBox.information(self, "小图模式", "种子线清理 Mask 仅用于大图流程。")
            return
        seeds = self.main_road_seed_strokes
        if not seeds:
            reply = QMessageBox.warning(
                self, "缺少主路种子线",
                "请先绘制主路种子线，再清理当前 Mask。\n\n是否现在开始绘制？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._begin_seed_drawing()
            return

        # ★ 清理输入：优先读 working_road_mask.png（full-size），不要用 preview/图层冒充
        mask = None
        working_path = getattr(self, "_working_road_mask_path", None) or ""
        if self._large_image_project is not None:
            working_path = working_path or getattr(
                self._large_image_project, "working_road_mask_path", ""
            ) or ""
        if working_path and os.path.isfile(working_path) and not self._is_preview_mask_path(working_path):
            try:
                mask = cv2.imread(working_path, cv2.IMREAD_GRAYSCALE)
            except Exception:
                mask = None
        if mask is None:
            working = self._layer_manager.get_layer_data("mask")
            if isinstance(working, np.ndarray) and working.size > 0:
                mask = working
        if mask is None:
            QMessageBox.warning(
                self, "清理 Mask",
                "当前没有可清理的 working_road_mask。\n"
                "请先「保存当前 Mask」生成 working_road_mask.png。",
            )
            return

        rois, ignores, tasks = self._collect_roi_ignore_task_original()
        self._save_main_road_seed_strokes()

        try:
            from roadnet.large_mask_cleaner import (
                clean_working_road_mask, save_cleaned_mask_artifacts,
            )
            self._history.push_state("seed_clean_mask")
            self._status_bar.show_message("正在根据主路种子线清理 Mask…")
            QApplication.processEvents()

            cleaned, report = clean_working_road_mask(
                mask,
                roi_polygons=rois or None,
                ignore_polygons=ignores or None,
                main_road_seed_strokes=seeds,
                task_points=tasks or None,
            )
            if report.get("refused"):
                QMessageBox.warning(
                    self, "清理被拒绝",
                    "\n".join(report.get("warnings") or ["未提供主路种子线"]),
                )
                return
            if np.count_nonzero(cleaned) == 0:
                QMessageBox.warning(
                    self, "清理结果为空",
                    "清理后 Mask 为空，请确认种子线画在道路上。\n"
                    + "\n".join(report.get("warnings") or []),
                )
                return

            if self._large_image_project is not None:
                out_dir = os.path.join(self._large_image_project.project_dir, "masks")
            else:
                out_dir = os.path.join(os.getcwd(), "outputs", "masks")
            pw = self._layer_manager.preview_width
            ph = self._layer_manager.preview_height
            saved = save_cleaned_mask_artifacts(
                cleaned, report, out_dir, preview_size=(pw, ph),
            )

            # ★ 清理后：cleaned 成为当前可编辑 working（中间结果，可继续手修）
            if isinstance(mask, np.ndarray):
                self._cleaned_mask_backup = mask.copy()
            self._cleaned_mask_pending = cleaned
            self._cleaned_mask_report = report
            self._cleaned_working_mask_path = saved.get("cleaned_working_mask.png")
            self._cleaned_working_mask_preview_path = saved.get(
                "cleaned_working_mask_preview.png"
            )

            preview = stages.get("cleaned_preview") if (stages := (report.get("stages") or {})) else None
            if preview is None and pw > 0 and ph > 0:
                preview = cv2.resize(cleaned, (pw, ph), interpolation=cv2.INTER_NEAREST)

            # 写入 working 图层，供画笔继续编辑；同时保留 cleaned 图层供对比
            self._layer_manager.set_layer_data("mask", cleaned, preview_data=preview)
            self._layer_manager.set_layer_data(
                "layer_cleaned_road_mask", cleaned, preview_data=preview,
            )
            self._layer_manager.show_layer("layer_road_mask")
            self._layer_manager.hide_layer("layer_cleaned_road_mask")
            self._layer_manager.hide_layer("layer_final_edited_mask")
            # 同步 working 文件，保证后续保存/骨架可落到磁盘
            if self._large_image_project is not None:
                masks_dir = Path(self._large_image_project.project_dir) / "masks"
                masks_dir.mkdir(parents=True, exist_ok=True)
                working_path = masks_dir / "working_road_mask.png"
                working_prev = masks_dir / "working_road_mask_preview.png"
                cv2.imwrite(str(working_path), cleaned)
                if preview is not None:
                    cv2.imwrite(str(working_prev), preview)
                self._working_road_mask_path = str(working_path)
                self._working_road_mask_preview_path = str(working_prev)
                # 清理产生新中间结果后，旧 final_edited 失效
                self._final_edited_mask_path = None
                self._final_edited_mask_preview_path = None
                # 删除旧 final 文件，避免骨架仍优先选到过期 final
                for old_name in ("final_edited_mask.png", "final_edited_mask_preview.png"):
                    old_p = masks_dir / old_name
                    if old_p.is_file():
                        try:
                            old_p.unlink()
                        except Exception:
                            pass
                try:
                    self._layer_manager.set_layer_data("layer_final_edited_mask", None)
                except Exception:
                    pass

            self._mask_edit_base = "cleaned_working_mask"
            self._working_mask_source = "cleaned_working_mask"
            self._working_mask_dirty = False
            self._working_mask_formal_ready = True
            self._working_mask_preview_only = False

            self._sync_layer_checkboxes()
            self._canvas.refresh_scene()

            if self._large_image_project is not None:
                p = self._large_image_project
                p.cleaned_working_mask_path = self._cleaned_working_mask_path or ""
                p.cleaned_working_mask_preview_path = (
                    self._cleaned_working_mask_preview_path or ""
                )
                p.working_road_mask_path = self._working_road_mask_path or ""
                p.working_road_mask_preview_path = (
                    self._working_road_mask_preview_path or ""
                )
                p.final_edited_mask_path = ""
                p.final_edited_mask_preview_path = ""
                p.mask_source = "cleaned_working_mask"
                p.mask_edit_base = "cleaned_working_mask"
                p.formal_ready = True
                p.preview_only = False
                p.mask_dirty = False
                try:
                    p.save()
                except Exception:
                    pass

            self._formal_mask_meta = {
                **(self._formal_mask_meta or {}),
                "mask_source": "cleaned_working_mask",
                "mask_edit_base": "cleaned_working_mask",
                "working_mask_path": self._working_road_mask_path or "",
                "cleaned_mask_path": self._cleaned_working_mask_path or "",
                "cleaned_mask_preview_path": self._cleaned_working_mask_preview_path or "",
                "final_edited_mask_path": "",
                "formal_ready": True,
                "preview_only": False,
            }
            if self._large_image_project is not None:
                skel_dir = os.path.join(self._large_image_project.project_dir, "skeleton")
                self._invalidate_large_skeleton_cache_if_needed(
                    skel_dir,
                    {
                        "selected_mask_path": self._cleaned_working_mask_path,
                        "checksum": self._mask_file_fingerprint(
                            self._cleaned_working_mask_path or ""
                        ).get("checksum"),
                    },
                )

            msg = (
                f"种子线清理完成：连通域 {report.get('component_count_before')} → "
                f"{report.get('component_count_after')}，"
                f"删除 {report.get('removed_component_count')} 个误检块。\n"
                f"cleaned_working_mask 已设为当前可编辑 Mask（中间结果）。\n"
                f"请继续用画笔/橡皮修正后「保存当前 Mask」→ final_edited_mask，再生成骨架。\n"
                f"报告：{saved.get('large_mask_clean_report.json', '')}"
            )
            self._update_large_mask_status_bar("可继续手动修正")
            QMessageBox.information(self, "种子线清理 Mask", msg)
        except Exception as exc:
            QMessageBox.critical(self, "清理失败", f"根据种子线清理 Mask 失败：\n{exc}")

    def _on_seed_clean_compare(self):
        """查看清理前后对比（working vs cleaned）。"""
        if not self._layer_manager.is_large_image_mode:
            return
        cleaned_path = self._cleaned_working_mask_preview_path
        if not cleaned_path or not os.path.isfile(cleaned_path):
            # try project
            if self._large_image_project is not None:
                cleaned_path = getattr(
                    self._large_image_project, "cleaned_working_mask_preview_path", ""
                )
        if not cleaned_path or not os.path.isfile(cleaned_path):
            QMessageBox.information(
                self, "清理对比",
                "尚未生成 cleaned_working_mask_preview，请先执行「根据种子线清理当前 Mask」。",
            )
            return

        # 生成并排对比图（working preview | cleaned preview）
        working_prev = self._layer_manager.get_layer_preview("layer_road_mask")
        cleaned_prev = cv2.imread(cleaned_path, cv2.IMREAD_GRAYSCALE)
        if working_prev is None:
            working_prev = np.zeros_like(cleaned_prev) if cleaned_prev is not None else None
        if cleaned_prev is None:
            QMessageBox.warning(self, "清理对比", "无法读取 cleaned preview。")
            return
        if working_prev.shape[:2] != cleaned_prev.shape[:2]:
            working_prev = cv2.resize(
                working_prev, (cleaned_prev.shape[1], cleaned_prev.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        left = cv2.cvtColor((working_prev > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        right = cv2.cvtColor((cleaned_prev > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        # tint: working green, cleaned cyan
        left[:, :, 1] = np.maximum(left[:, :, 1], left[:, :, 0])
        right[:, :, 1] = np.maximum(right[:, :, 1], right[:, :, 0])
        right[:, :, 0] = 0
        gap = np.full((left.shape[0], 8, 3), 40, dtype=np.uint8)
        combo = np.concatenate([left, gap, right], axis=1)
        out_dir = Path(cleaned_path).parent
        cmp_path = out_dir / "mask_clean_compare_preview.png"
        cv2.imwrite(str(cmp_path), combo)
        self._show_image_dialog(
            str(cmp_path), "清理前后对比",
            "左 = working Road Mask（绿）　右 = cleaned Road Mask（青）",
        )

    def _on_seed_clean_accept(self):
        """接受 cleaned mask，写入 working 图层并持久化。"""
        if not self._layer_manager.is_large_image_mode:
            return
        cleaned = self._cleaned_mask_pending
        if cleaned is None:
            # 尝试从图层 / 磁盘加载
            cleaned = self._layer_manager.get_layer_data("layer_cleaned_road_mask")
            if not isinstance(cleaned, np.ndarray):
                path = self._cleaned_working_mask_path or ""
                if self._large_image_project is not None:
                    path = path or getattr(
                        self._large_image_project, "cleaned_working_mask_path", ""
                    )
                if path and os.path.isfile(path):
                    cleaned = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if not isinstance(cleaned, np.ndarray) or cleaned.size == 0:
            QMessageBox.information(self, "接受 cleaned mask", "没有可接受的 cleaned mask。")
            return

        if self._cleaned_mask_backup is None:
            cur = self._layer_manager.get_layer_data("mask")
            if isinstance(cur, np.ndarray):
                self._cleaned_mask_backup = cur.copy()

        pw = self._layer_manager.preview_width
        ph = self._layer_manager.preview_height
        preview = None
        if pw > 0 and ph > 0:
            preview = cv2.resize(cleaned, (pw, ph), interpolation=cv2.INTER_NEAREST)
        self._layer_manager.set_layer_data("mask", cleaned, preview_data=preview)
        self._layer_manager.show_layer("mask")
        self._working_mask_source = "cleaned_working_mask"
        self._mask_edit_base = "cleaned_working_mask"
        self._working_mask_dirty = False
        self._working_mask_formal_ready = True
        self._working_mask_preview_only = False

        # 持久化为 working（不写 final；cleaned 仍是中间结果）
        if self._large_image_project is not None:
            try:
                cleaned_path = self._cleaned_working_mask_path or ""
                cleaned_prev = self._cleaned_working_mask_preview_path or ""
                self._persist_working_mask(cleaned, clear_cleaned=False, save_as_final=False)
                self._working_mask_source = "cleaned_working_mask"
                self._mask_edit_base = "cleaned_working_mask"
                project = self._large_image_project
                project.mask_source = "cleaned_working_mask"
                project.mask_edit_base = "cleaned_working_mask"
                project.cleaned_working_mask_path = cleaned_path
                project.cleaned_working_mask_preview_path = cleaned_prev
                project.formal_ready = True
                project.preview_only = False
                project.save()
                self._cleaned_working_mask_path = cleaned_path
                self._cleaned_working_mask_preview_path = cleaned_prev
                self._formal_mask_meta = {
                    **(self._formal_mask_meta or {}),
                    "mask_source": "cleaned_working_mask",
                    "mask_edit_base": "cleaned_working_mask",
                    "working_mask_path": project.working_road_mask_path,
                    "cleaned_mask_path": cleaned_path,
                    "formal_ready": True,
                    "preview_only": False,
                }
            except Exception as exc:
                print(f"[SeedClean] 接受后持久化失败: {exc}")

        self._sync_layer_checkboxes()
        self._canvas.refresh_scene()
        self._update_large_mask_status_bar("已接受 cleaned（中间结果）")
        QMessageBox.information(
            self, "已接受",
            "cleaned mask 已写入当前可编辑 Working Road Mask（中间结果）。\n"
            "请继续手动修正后「保存当前 Mask」→ final_edited_mask，再生成骨架。\n"
            "骨架优先级：final_edited > working > cleaned > refined > global。",
        )

    def _on_seed_clean_rollback(self):
        """回滚到清理前的 working mask。"""
        if not self._layer_manager.is_large_image_mode:
            return
        backup = self._cleaned_mask_backup
        if not isinstance(backup, np.ndarray):
            QMessageBox.information(self, "回滚", "没有可回滚的 working mask 备份。")
            return
        pw = self._layer_manager.preview_width
        ph = self._layer_manager.preview_height
        preview = None
        if pw > 0 and ph > 0:
            preview = cv2.resize(backup, (pw, ph), interpolation=cv2.INTER_NEAREST)
        self._layer_manager.set_layer_data("mask", backup, preview_data=preview)
        self._layer_manager.hide_layer("layer_cleaned_road_mask")
        self._working_mask_source = "manual_edited"
        self._cleaned_mask_pending = None
        self._sync_layer_checkboxes()
        self._canvas.refresh_scene()
        self._status_bar.show_message("已回滚到清理前的 working mask")

    def _get_ribbon_fill_config(self) -> dict:
        cfg = {}
        panel = getattr(self, "_param_panel", None)
        if panel is None:
            return cfg
        get = getattr(panel, "_get_config", None)
        if not callable(get):
            return cfg
        keys = (
            ("ribbon_fill.max_hole_area_px", "max_hole_area_px", 500),
            ("ribbon_fill.max_gap_area_px", "max_gap_area_px", 800),
            ("ribbon_fill.max_hole_diameter_px", "max_hole_diameter_px", 25),
            ("ribbon_fill.max_gap_diameter_px", "max_gap_diameter_px", 35),
            ("ribbon_fill.ribbon_buffer_px", "ribbon_buffer_px", 10),
            ("ribbon_fill.min_surround_ratio_for_hole", "min_surround_ratio_for_hole", 0.70),
            ("ribbon_fill.min_surround_ratio_for_gap", "min_surround_ratio_for_gap", 0.45),
            ("ribbon_fill.max_gap_distance_to_mask_px", "max_gap_distance_to_mask_px", 8),
            ("ribbon_fill.require_inside_ribbon", "require_inside_ribbon", True),
        )
        for ui_key, cfg_key, default in keys:
            cfg[cfg_key] = get(ui_key, default)
        return cfg

    def _load_fullsize_working_mask_for_edit(self):
        """优先读 full-size working / final mask，避免 preview 冒充。"""
        mask = None
        used_path = ""
        candidates = []
        if getattr(self, "_final_edited_mask_path", None):
            candidates.append(self._final_edited_mask_path)
        if getattr(self, "_working_road_mask_path", None):
            candidates.append(self._working_road_mask_path)
        if self._large_image_project is not None:
            candidates.append(getattr(self._large_image_project, "final_edited_mask_path", "") or "")
            candidates.append(getattr(self._large_image_project, "working_road_mask_path", "") or "")
        for path in candidates:
            if path and os.path.isfile(path) and not self._is_preview_mask_path(path):
                try:
                    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                except Exception:
                    arr = None
                if isinstance(arr, np.ndarray) and arr.size > 0:
                    return arr, path
        working = self._layer_manager.get_layer_data("mask")
        if isinstance(working, np.ndarray) and working.size > 0:
            return working, used_path
        return None, used_path

    def _on_ribbon_hole_gap_fill(self):
        """道路带约束补洞 / 补缺口（仅大图）。"""
        if not self._layer_manager.is_large_image_mode:
            QMessageBox.information(self, "小图模式", "道路带内补洞补缺口仅用于大图流程。")
            return

        if hasattr(self._canvas, "get_main_road_seed_stroke_dicts"):
            seeds = self._canvas.get_main_road_seed_stroke_dicts()
        else:
            seeds = self.main_road_seed_strokes
        if not seeds:
            QMessageBox.warning(
                self, "缺少道路带",
                "请先绘制主路种子线并生成道路带，再执行道路带内补洞补缺口。",
            )
            return

        mask, mask_path = self._load_fullsize_working_mask_for_edit()
        if mask is None:
            QMessageBox.warning(
                self, "补洞补缺口",
                "当前没有可修补的 Road Mask。\n请先保存 working_road_mask 或生成正式 Mask。",
            )
            return

        # 尺寸校验
        if self._large_image_project is not None:
            ow = int(self._large_image_project.image_width or 0)
            oh = int(self._large_image_project.image_height or 0)
            if ow > 0 and oh > 0 and tuple(mask.shape[:2]) != (oh, ow):
                QMessageBox.warning(
                    self, "补洞补缺口",
                    f"当前 mask 尺寸 {mask.shape[1]}×{mask.shape[0]} 与原图 {ow}×{oh} 不一致。",
                )
                return

        self._apply_seed_width_settings_to_canvas()
        self._save_main_road_seed_strokes()
        _, ignores, _ = self._collect_roi_ignore_task_original()

        from roadnet.main_road_seed import build_road_ribbon_mask
        from roadnet.mask_hole_filler import (
            fill_holes_and_gaps_guided_by_ribbon,
            save_ribbon_hole_gap_artifacts,
        )

        h, w = mask.shape[:2]
        ribbon = build_road_ribbon_mask((h, w), seeds)
        if not np.any(ribbon > 0):
            QMessageBox.warning(
                self, "缺少道路带",
                "请先绘制主路种子线并生成道路带，再执行道路带内补洞补缺口。",
            )
            return

        ignore_mask = np.zeros((h, w), dtype=np.uint8)
        for poly in ignores or []:
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
            if len(pts) >= 3:
                cv2.fillPoly(ignore_mask, [pts.reshape(-1, 1, 2)], 255)

        valid = self._ensure_valid_image_mask()
        cfg = self._get_ribbon_fill_config()

        self._history.push_state("ribbon_hole_gap_fill")
        self._status_bar.show_message("正在执行道路带内补洞补缺口…")
        QApplication.processEvents()

        try:
            repaired, report = fill_holes_and_gaps_guided_by_ribbon(
                mask, ribbon,
                ignore_mask=ignore_mask,
                valid_area_mask=valid,
                config=cfg,
            )
        except Exception as exc:
            QMessageBox.critical(self, "补洞补缺口失败", str(exc))
            return

        if self._large_image_project is not None:
            out_dir = Path(self._large_image_project.project_dir) / "masks" / "ribbon_hole_gap_fill"
        else:
            out_dir = Path(os.getcwd()) / "outputs" / "masks" / "ribbon_hole_gap_fill"
        pw = self._layer_manager.preview_width
        ph = self._layer_manager.preview_height
        try:
            paths = save_ribbon_hole_gap_artifacts(
                mask, repaired, report, out_dir,
                preview_size=(pw, ph) if pw and ph else None,
                input_mask_path=mask_path or "",
                road_ribbon_mask_path=str(out_dir / "road_ribbon_mask.png"),
            )
        except Exception as exc:
            QMessageBox.critical(self, "写出结果失败", str(exc))
            return

        self._ribbon_fill_backup = mask.copy()
        self._ribbon_fill_pending = repaired
        self._ribbon_fill_report = report
        self._ribbon_fill_artifact_dir = str(out_dir)
        self._ribbon_fill_paths = paths

        preview = None
        if pw > 0 and ph > 0:
            preview = cv2.resize(repaired, (pw, ph), interpolation=cv2.INTER_NEAREST)
        self._layer_manager.set_layer_data("mask", repaired, preview_data=preview)
        self._layer_manager.show_layer("layer_road_mask")
        self._sync_layer_checkboxes()
        self._canvas.refresh_scene()

        msg = (
            f"补洞补缺口完成（{report.get('elapsed_seconds')}s）\n"
            f"孔洞：候选 {report.get('candidate_hole_count')} / "
            f"填充 {report.get('filled_hole_count')} / "
            f"拒绝 {report.get('rejected_hole_count')}\n"
            f"缺口：候选 {report.get('candidate_gap_count')} / "
            f"填充 {report.get('filled_gap_count')} / "
            f"拒绝 {report.get('rejected_gap_count')}\n\n"
            f"请「查看结果」核对后「接受结果」写入 final_edited_mask。\n"
            f"报告：{paths.get('ribbon_hole_gap_fill_report.json', '')}"
        )
        self._update_large_mask_status_bar("道路带补洞补缺口待接受")
        QMessageBox.information(self, "道路带内补洞补缺口", msg)

    def _on_ribbon_hole_gap_view(self):
        """查看补洞补缺口 overlay / preview。"""
        if not self._layer_manager.is_large_image_mode:
            return
        paths = getattr(self, "_ribbon_fill_paths", None) or {}
        preview = paths.get("ribbon_hole_gap_filled_mask_preview.png")
        overlay_h = paths.get("accepted_holes_overlay.png")
        overlay_g = paths.get("accepted_gaps_overlay.png")
        show_path = preview or overlay_h or overlay_g
        if not show_path or not os.path.isfile(show_path):
            # try artifact dir
            art = getattr(self, "_ribbon_fill_artifact_dir", None)
            if art:
                for name in (
                    "ribbon_hole_gap_filled_mask_preview.png",
                    "accepted_gaps_overlay.png",
                    "accepted_holes_overlay.png",
                ):
                    p = os.path.join(art, name)
                    if os.path.isfile(p):
                        show_path = p
                        break
        if not show_path or not os.path.isfile(show_path):
            QMessageBox.information(
                self, "查看结果",
                "尚未生成补洞补缺口结果，请先执行「道路带内补洞补缺口」。",
            )
            return
        report = getattr(self, "_ribbon_fill_report", {}) or {}
        info = (
            f"孔洞填充 {report.get('filled_hole_count', 0)}，"
            f"缺口填充 {report.get('filled_gap_count', 0)}，"
            f"耗时 {report.get('elapsed_seconds', '?')}s"
        )
        self._show_image_dialog(show_path, "补洞补缺口结果", info)

    def _on_ribbon_hole_gap_accept(self):
        """接受补洞补缺口结果 → final_edited_mask，mask_source=ribbon_hole_gap_filled。"""
        if not self._layer_manager.is_large_image_mode:
            return
        repaired = self._ribbon_fill_pending
        if not isinstance(repaired, np.ndarray) or repaired.size == 0:
            # try layer / artifact
            path = (self._ribbon_fill_paths or {}).get("ribbon_hole_gap_filled_mask.png")
            if path and os.path.isfile(path):
                repaired = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if not isinstance(repaired, np.ndarray) or repaired.size == 0:
            QMessageBox.information(self, "接受结果", "没有可接受的补洞补缺口结果。")
            return

        if self._ribbon_fill_backup is None:
            cur = self._layer_manager.get_layer_data("mask")
            if isinstance(cur, np.ndarray):
                self._ribbon_fill_backup = cur.copy()

        pw = self._layer_manager.preview_width
        ph = self._layer_manager.preview_height
        preview = None
        if pw > 0 and ph > 0:
            preview = cv2.resize(repaired, (pw, ph), interpolation=cv2.INTER_NEAREST)

        try:
            if self._large_image_project is not None:
                # 先写入 working + final
                self._persist_working_mask(repaired, save_as_final=True)
                # 覆盖 mask_source 为 ribbon_hole_gap_filled
                self._working_mask_source = "ribbon_hole_gap_filled"
                project = self._large_image_project
                project.mask_source = "ribbon_hole_gap_filled"
                project.formal_ready = True
                project.preview_only = False
                project.mask_dirty = False
                project.save()
                self._formal_mask_meta = {
                    **(self._formal_mask_meta or {}),
                    "mask_source": "ribbon_hole_gap_filled",
                    "final_edited_mask_path": project.final_edited_mask_path,
                    "formal_ready": True,
                    "preview_only": False,
                    "mask_dirty": False,
                }
                skel_dir = os.path.join(project.project_dir, "skeleton")
                self._invalidate_large_skeleton_cache_if_needed(
                    skel_dir,
                    {
                        "selected_mask_path": project.final_edited_mask_path,
                        "checksum": self._mask_file_fingerprint(
                            project.final_edited_mask_path or ""
                        ).get("checksum"),
                    },
                )
            else:
                self._layer_manager.set_layer_data("mask", repaired, preview_data=preview)
                self._working_mask_source = "ribbon_hole_gap_filled"
        except Exception as exc:
            QMessageBox.critical(self, "接受失败", str(exc))
            return

        self._sync_layer_checkboxes()
        self._canvas.refresh_scene()
        self._update_large_mask_status_bar("ribbon_hole_gap_filled")
        QMessageBox.information(
            self, "已接受",
            "补洞补缺口结果已保存为 final_edited_mask。\n"
            "mask_source = ribbon_hole_gap_filled\n"
            "后续骨架生成将优先使用该 final_edited_mask。",
        )

    def _on_ribbon_hole_gap_rollback(self):
        """回滚到补洞补缺口前的 mask。"""
        if not self._layer_manager.is_large_image_mode:
            return
        backup = self._ribbon_fill_backup
        if not isinstance(backup, np.ndarray):
            QMessageBox.information(self, "回滚", "没有可回滚的 mask 备份。")
            return
        pw = self._layer_manager.preview_width
        ph = self._layer_manager.preview_height
        preview = None
        if pw > 0 and ph > 0:
            preview = cv2.resize(backup, (pw, ph), interpolation=cv2.INTER_NEAREST)
        self._layer_manager.set_layer_data("mask", backup, preview_data=preview)
        self._ribbon_fill_pending = None
        self._working_mask_source = "manual_edited"
        self._sync_layer_checkboxes()
        self._canvas.refresh_scene()
        self._status_bar.show_message("已回滚到补洞补缺口前的 mask")

    def _show_main_road_corridor(self):
        """在画布上以半透明方式显示当前 corridor（种子 / ROI / 任务点）。"""
        if not self._layer_manager.is_large_image_mode:
            QMessageBox.information(self, "小图模式", "该功能仅用于大图主路修复。")
            return
        if getattr(self._canvas, "_corridor_overlay_item", None) is not None:
            self._canvas.clear_corridor_overlay()
            self._status_bar.show_message("已隐藏主路 corridor。")
            return
        ow, oh = self._layer_manager.original_size
        if ow <= 0 or oh <= 0:
            return
        scale = float(self._layer_manager.preview_scale) or 1.0
        pw, ph = max(1, int(ow * scale)), max(1, int(oh * scale))
        seeds = [[(x * scale, y * scale) for (x, y) in stroke]
                 for stroke in self._canvas.get_main_road_seed_strokes()]
        rois, _, task_points = self._collect_roi_ignore_task_original()
        rois_s = [[[px * scale, py * scale] for (px, py) in poly] for poly in rois]
        tasks_s = [[tx * scale, ty * scale] for (tx, ty) in task_points]
        view_s = ([v * scale for v in self._main_road_view_rect]
                  if getattr(self, "_main_road_view_rect", None) else None)
        from roadnet.main_road_postprocess import build_main_road_corridor, DEFAULT_MAIN_ROAD_CONFIG
        cfg = dict(DEFAULT_MAIN_ROAD_CONFIG)
        corridor, info = build_main_road_corridor(
            (ph, pw), seeds or None, rois_s or None, tasks_s or None, view_s, cfg
        )
        if not corridor.any():
            QMessageBox.warning(
                self, "无 corridor",
                "尚未提供主路种子线 / ROI / 任务点 / 修复范围，无法生成 corridor。"
            )
            return
        self._canvas.show_corridor_overlay(corridor, ow, oh)
        self._status_bar.show_message(
            f"主路 corridor 覆盖 {info.get('corridor_nonzero_ratio', 0)*100:.1f}% "
            "（再次点击可隐藏）。"
        )

    def _on_preview_bridges(self):
        """预览桥接候选（不改动 mask）。"""
        self._start_main_road_refine(preview_only=True)

    def _on_refine_main_road(self):
        """执行主路修复（种子/ROI/任务点约束）。"""
        self._start_main_road_refine(preview_only=False)

    def _acquire_main_road_source(self):
        """获取主路修复输入 mask。返回 (mask, source_is_preview) 或 (None, None)。"""
        mask = self._layer_manager.get_layer_data("mask")
        if isinstance(mask, np.ndarray):
            return mask, False
        if not self._has_preview_segmentation_only():
            QMessageBox.warning(
                self, "主路修复",
                "当前没有可用的 Road Mask。请先运行大图 OpenCV 正式提取。"
            )
            return None, None
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("仅有快速预览结果")
        dialog.setText(
            "当前只有快速预览分割结果，尚无正式 mask。\n"
            "基于快速预览的结果只适合快速测试，正式比赛建议\n"
            "基于 formal mask 或人工修正 mask。"
        )
        preview_btn = dialog.addButton("仅在预览上测试修复", QMessageBox.ButtonRole.AcceptRole)
        upsample_btn = dialog.addButton("升采样为初始 mask 后修复", QMessageBox.ButtonRole.ActionRole)
        dialog.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        dialog.exec()
        clicked = dialog.clickedButton()
        preview_data = self._layer_manager.get_layer_data("layer_preview_segmentation")
        if not isinstance(preview_data, np.ndarray):
            QMessageBox.warning(self, "主路修复", "无法读取快速预览数据。")
            return None, None
        if clicked is preview_btn:
            return np.asarray(preview_data), True
        if clicked is upsample_btn:
            ow, oh = self._layer_manager.original_size
            if ow <= 0 or oh <= 0:
                QMessageBox.warning(self, "主路修复", "大图尺寸无效。")
                return None, None
            return cv2.resize(np.asarray(preview_data, dtype=np.uint8), (ow, oh),
                              interpolation=cv2.INTER_NEAREST), False
        return None, None

    def _start_main_road_refine(self, preview_only: bool):
        """启动主路修复 worker（preview_only=True 仅预览桥接候选，不改动 mask）。"""
        if not self._layer_manager.is_large_image_mode:
            QMessageBox.information(
                self, "小图模式", "主路修复仅用于大图模式，小图流程不受影响。"
            )
            return
        if self._main_road_thread is not None and self._main_road_thread.isRunning():
            QMessageBox.information(self, "主路修复", "任务已在运行，请勿重复启动。")
            return

        # ── 约束检查：无 seed/ROI/task/view 时拒绝（不做全图自由修复）──
        seed_count = self._canvas.get_main_road_seed_count()
        rois, ignores, task_points = self._collect_roi_ignore_task_original()
        view_rect = getattr(self, "_main_road_view_rect", None)
        if not (seed_count or rois or task_points or view_rect):
            dialog = QMessageBox(self)
            dialog.setIcon(QMessageBox.Icon.Warning)
            dialog.setWindowTitle("需要主路约束")
            dialog.setText(
                "请先画主路种子线或设置 ROI，否则全图修复会误连大量噪声。"
            )
            seed_btn = dialog.addButton("绘制主路种子线", QMessageBox.ButtonRole.AcceptRole)
            view_btn = dialog.addButton("使用当前视野作为修复范围", QMessageBox.ButtonRole.ActionRole)
            dialog.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            dialog.exec()
            if dialog.clickedButton() is seed_btn:
                self._begin_seed_drawing()
            elif dialog.clickedButton() is view_btn:
                self._seed_use_view()
            return

        mask, source_is_preview = self._acquire_main_road_source()
        if mask is None:
            return

        seed_strokes = self._canvas.get_main_road_seed_strokes()

        from datetime import datetime
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if self._large_image_project is not None:
            base_dir = (Path(self._large_image_project.project_dir)
                        / "main_road_refine")
        else:
            base_dir = Path("outputs") / "large_image_projects" / "main_road_refine"
        output_dir = base_dir / run_name

        from roadnet.large_image_worker import MainRoadRefineWorker
        from roadnet.main_road_postprocess import DEFAULT_MAIN_ROAD_CONFIG
        config = dict(DEFAULT_MAIN_ROAD_CONFIG)

        # 修复前快照（供回滚）。preview 预览不覆盖 mask，不需要快照。
        if not preview_only:
            cur = self._layer_manager.get_layer_data("mask")
            self._main_road_backup_mask = (cur.copy() if isinstance(cur, np.ndarray) else None)
            self._main_road_backup_meta = dict(self._formal_mask_meta or {})

        self._main_road_preview_only = preview_only

        thread = QThread(self)
        worker = MainRoadRefineWorker(
            mask, str(output_dir), config=config,
            image_path=self._layer_manager.image_path or "",
            roi_polygons=rois, ignore_polygons=ignores, task_points=task_points,
            seed_strokes=seed_strokes, view_rect=view_rect,
            source_is_preview=source_is_preview,
        )
        worker.moveToThread(thread)
        title = "预览桥接候选" if preview_only else "主路修复"
        progress = QProgressDialog(f"准备{title}…", "取消", 0, 100, self)
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModality.NonModal)
        progress.canceled.connect(worker.cancel)
        thread.started.connect(worker.run)
        worker.progress.connect(
            lambda percent, current, total, message: (
                progress.setValue(int(percent)),
                progress.setLabelText(message),
            )
        )
        worker.finished.connect(self._on_refine_main_road_finished)
        worker.failed.connect(self._on_refine_main_road_failed)
        worker.cancelled.connect(
            lambda message: self._status_bar.show_message(f"{message}；原 mask 未改变")
        )
        for signal in (worker.finished, worker.failed, worker.cancelled):
            signal.connect(thread.quit)
            signal.connect(progress.close)
            signal.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_main_road_thread_finished)
        self._main_road_thread = thread
        self._main_road_worker = worker
        self._main_road_progress = progress
        progress.show()
        self._status_bar.show_message(f"{title}已启动（corridor 内分析）。")
        thread.start()

    def _on_refine_main_road_finished(self, result):
        report = result.report
        preview_only = getattr(self, "_main_road_preview_only", False)

        if preview_only:
            # 仅预览桥接候选：不改动 mask，弹出 overlay 供人工确认。
            overlay = os.path.join(result.output_dir, "bridge_candidates_overlay.png")
            self._status_bar.show_message(
                f"桥接候选：接受 {report.get('accepted_bridge_count', 0)}，"
                f"待确认 {report.get('pending_bridge_count', 0)}，"
                f"拒绝 {report.get('rejected_bridge_count', 0)}"
            )
            self._show_image_dialog(
                overlay, "桥接候选预览（绿=接受 / 红=拒绝 / 黄=待确认）",
                info=(
                    f"候选 {report.get('bridge_candidate_count', 0)}，"
                    f"接受 {report.get('accepted_bridge_count', 0)}，"
                    f"待确认 {report.get('pending_bridge_count', 0)}，"
                    f"拒绝 {report.get('rejected_bridge_count', 0)}\n"
                    "预览不改动当前 mask，如需应用请点“执行主路修复”。"
                )
            )
            return

        mask = cv2.imread(result.mask_path, cv2.IMREAD_GRAYSCALE)
        preview = cv2.imread(result.preview_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            self._on_refine_main_road_failed("无法读取 main_road_mask.png", result.mask_path)
            return
        self._layer_manager.set_layer_data("mask", mask, preview_data=preview)
        self._layer_manager.show_layer("mask")

        # ── 注册为正式 Road Mask（formal_opencv_mainroad_refined）──
        self._formal_mask_meta = {
            "mask_type": "formal_opencv_mainroad_refined",
            "formal_ready": True,
            "preview_only": False,
            "coordinate_system": "original_image_pixel",
            "global_mask_path": result.mask_path,
            "source": result.output_dir,
        }
        if self._large_image_project is not None:
            try:
                self._large_image_project.global_mask_path = result.mask_path
                self._large_image_project.save()
                if hasattr(self, "_project_manager") and self._project_manager is not None:
                    self._project_manager.data.global_mask_path = result.mask_path
                    self._project_manager.mark_dirty()
            except Exception:
                pass

        self._clear_skeleton_state(clear_layer=True)
        self._canvas.refresh_scene()
        self._canvas.viewport().update()

        self._status_bar.show_message(
            f"主路修复完成：删除 {report.get('removed_component_count', 0)} 误检块，"
            f"删孤立 {report.get('removed_unseeded_components', 0)}，"
            f"桥接 {report.get('accepted_bridge_count', 0)}，"
            f"用时 {report.get('total_elapsed_seconds', report.get('elapsed_seconds', 0.0)):.1f}s"
        )
        msg = QMessageBox(self)
        msg.setWindowTitle("主路修复完成")
        msg.setText(
            "已生成正式 Road Mask（mask_type = formal_opencv_mainroad_refined），"
            "可进入骨架 / graph / 路径规划。\n"
            "如效果不理想，可点“回滚修复结果”还原。"
        )
        msg.setInformativeText(
            f"连通域: {report.get('component_count_before', 0)} → "
            f"{report.get('component_count_after', 0)}\n"
            f"删除误检块: {report.get('removed_component_count', 0)}"
            f"（其中孤立 {report.get('removed_unseeded_components', 0)}）\n"
            f"skeleton edge: {report.get('edge_count_before', 0)} → "
            f"{report.get('edge_count_after', 0)}"
            f"（删短支路 {report.get('removed_short_branch_count', 0)}）\n"
            f"桥接: 候选 {report.get('bridge_candidate_count', 0)}，"
            f"接受 {report.get('accepted_bridge_count', 0)}，"
            f"待确认 {report.get('pending_bridge_count', 0)}\n"
            f"道路占比: {report.get('mask_nonzero_ratio_before', 0)*100:.2f}% → "
            f"{report.get('mask_nonzero_ratio_after', 0)*100:.2f}%\n\n"
            f"main_road_mask.png:\n{result.mask_path}\n\n"
            f"报告目录:\n{result.output_dir}"
        )
        msg.exec()

    def _on_refine_accept(self):
        """接受主路修复结果（清除回滚快照）。"""
        if self._main_road_backup_mask is None:
            self._status_bar.show_message("没有可接受的主路修复结果。")
            return
        self._main_road_backup_mask = None
        self._main_road_backup_meta = None
        self._status_bar.show_message("已接受主路修复结果。")

    def _on_refine_rollback(self):
        """回滚到主路修复前的 mask。"""
        if self._main_road_backup_mask is None:
            self._status_bar.show_message("没有可回滚的主路修复结果。")
            return
        backup = self._main_road_backup_mask
        self._layer_manager.set_layer_data("mask", backup)
        self._formal_mask_meta = dict(self._main_road_backup_meta or {})
        self._main_road_backup_mask = None
        self._main_road_backup_meta = None
        self._clear_skeleton_state(clear_layer=True)
        self._canvas.refresh_scene()
        self._canvas.viewport().update()
        self._status_bar.show_message("已回滚到主路修复前的 mask。")

    def _show_image_dialog(self, image_path: str, title: str, info: str = ""):
        """在一个可滚动对话框中显示图片（用于桥接候选 overlay 等）。"""
        if not image_path or not os.path.exists(image_path):
            QMessageBox.information(self, title, f"未找到图片：\n{image_path}")
            return
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QScrollArea
        from PySide6.QtGui import QPixmap
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(900, 700)
        layout = QVBoxLayout(dlg)
        if info:
            lbl_info = QLabel(info)
            lbl_info.setWordWrap(True)
            layout.addWidget(lbl_info)
        scroll = QScrollArea()
        img_label = QLabel()
        pix = QPixmap(image_path)
        img_label.setPixmap(pix)
        scroll.setWidget(img_label)
        scroll.setWidgetResizable(False)
        layout.addWidget(scroll)
        dlg.exec()

    def _on_refine_main_road_failed(self, message, error_path):
        detail = f"\n错误日志:\n{error_path}" if error_path else ""
        QMessageBox.critical(
            self, "主路修复失败",
            f"{message}\n\n原 mask 未改变。{detail}"
        )

    def _on_main_road_thread_finished(self):
        self._main_road_thread = None
        self._main_road_worker = None
        self._main_road_progress = None

    def _on_extract_roi_opencv(self):
        """大图 ROI 正式提取（OpenCV，比赛推荐）。只处理 ROI 覆盖 tile。"""
        if not self._require_large_image_for_opencv():
            return
        if self._canvas.get_enabled_roi_count() == 0:
            self._status_bar.show_message("未设置 ROI，已取消 ROI 正式提取。")
            dialog = QMessageBox(self)
            dialog.setIcon(QMessageBox.Icon.Warning)
            dialog.setWindowTitle("需要 ROI 区域")
            dialog.setText(
                "当前没有 ROI 区域，请先绘制 ROI，或使用当前视野作为 ROI。"
            )
            draw_button = dialog.addButton(
                "开始绘制 ROI", QMessageBox.ButtonRole.AcceptRole
            )
            view_button = dialog.addButton(
                "使用当前视野作为 ROI", QMessageBox.ButtonRole.ActionRole
            )
            dialog.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            dialog.exec()
            if dialog.clickedButton() is draw_button:
                self._begin_roi_drawing()
            elif dialog.clickedButton() is view_button:
                if self._use_current_view_as_roi():
                    self._start_opencv_formal_extraction("roi")
            return
        self._start_opencv_formal_extraction("roi")

    def _on_extract_full_opencv(self):
        """大图全图正式提取（OpenCV，赛前使用）。"""
        if not self._require_large_image_for_opencv():
            return
        self._start_opencv_formal_extraction("full")

    def _start_opencv_formal_extraction(self, mode: str):
        """启动大图 OpenCV 正式提取 worker（ROI 或全图）。"""
        if self._segmentation_thread is not None and self._segmentation_thread.isRunning():
            self._status_bar.show_message("分割任务已在运行，请勿重复启动")
            return
        if (self._formal_extraction_thread is not None
                and self._formal_extraction_thread.isRunning()):
            self._status_bar.show_message("正式提取任务已在运行，请勿重复启动")
            return

        width, height = self._layer_manager.original_size
        if width <= 0 or height <= 0:
            QMessageBox.warning(self, "图像为空", "大图项目尺寸无效。")
            return

        pos_points = list(self._canvas.positive_points)
        neg_points = list(self._canvas.negative_points)
        if not pos_points:
            QMessageBox.warning(self, "正样本为空", "请先添加道路正样本（绿点）。")
            return

        config = copy.deepcopy(self._param_panel.get_config())
        seg_base = dict(config.get("segment", {}))
        ocv = dict(config.get("opencv_extraction", {}))
        use_negative = bool(seg_base.get("use_negative_samples", True))
        if use_negative and not neg_points:
            QMessageBox.warning(
                self, "负样本为空",
                "当前启用了负样本约束，请先添加非道路负样本（红叉）。"
            )
            return

        def valid_points(points):
            return [(int(x), int(y)) for x, y in points
                    if 0 <= int(x) < width and 0 <= int(y) < height]

        pos_points = valid_points(pos_points)
        neg_points = valid_points(neg_points)
        if not pos_points:
            QMessageBox.warning(self, "样本点无效", "正样本点均位于图像范围之外。")
            return

        from roadnet.large_image_project import ImageRegionReader
        reader = ImageRegionReader(self._layer_manager.image_path)
        pos_rgb = reader.read_pixels(pos_points)
        neg_rgb = (reader.read_pixels(neg_points)
                   if neg_points else np.zeros((0, 3), dtype=np.uint8))

        from roadnet.opencv_road_segmenter import DEFAULT_LARGE_OPENCV_CONFIG
        seg_cfg = dict(seg_base)
        seg_cfg.update({
            "color_space": ocv.get("color_space", DEFAULT_LARGE_OPENCV_CONFIG["color_space"]),
            "blur_kernel": int(ocv.get("blur_kernel", DEFAULT_LARGE_OPENCV_CONFIG["blur_kernel"])),
            "open_kernel": int(ocv.get("open_kernel", DEFAULT_LARGE_OPENCV_CONFIG["open_kernel"])),
            "close_kernel": int(ocv.get("close_kernel", DEFAULT_LARGE_OPENCV_CONFIG["close_kernel"])),
            "min_area": int(ocv.get("min_area", DEFAULT_LARGE_OPENCV_CONFIG["min_area"])),
            "fill_holes": bool(ocv.get("fill_holes", DEFAULT_LARGE_OPENCV_CONFIG["fill_holes"])),
        })
        seg_cfg["use_negative_samples"] = use_negative
        seg_cfg["require_negative_samples"] = use_negative

        use_roi = bool(ocv.get("use_roi", True))
        use_ignore = bool(ocv.get("use_ignore", True))

        def polygons_to_global(qpolygons):
            result = []
            for polygon in qpolygons:
                pts = [[float(p.x()), float(p.y())] for p in polygon]
                if len(pts) >= 3:
                    result.append(pts)
            return result

        roi_polygons = (polygons_to_global(self._canvas.get_roi_polygons())
                        if (mode == "roi" and use_roi) else [])
        ignore_polygons = (polygons_to_global(self._canvas.get_ignore_polygons())
                           if use_ignore else [])

        if mode == "roi" and not roi_polygons:
            QMessageBox.warning(
                self, "需要 ROI 区域",
                "当前没有 ROI 区域，请先绘制 ROI，或使用当前视野作为 ROI。"
            )
            return

        tile_size = int(ocv.get("tile_size", seg_base.get("tile_size", 1024)))
        overlap = int(ocv.get("overlap", seg_base.get("overlap", 64)))
        if overlap >= tile_size:
            QMessageBox.warning(self, "分割参数错误", "Tile 重叠必须小于 Tile 大小。")
            return

        from datetime import datetime
        if self._large_image_project is not None:
            base_dir = os.path.join(
                self._large_image_project.project_dir, "formal_extraction"
            )
        else:
            base_dir = os.path.join("outputs", "formal_extraction")
        output_dir = os.path.join(
            base_dir, f"opencv_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        from roadnet.large_image_worker import LargeImageSegmentationWorker
        from PySide6.QtCore import QThread

        thread = QThread(self)
        worker = LargeImageSegmentationWorker(
            image_path=self._layer_manager.image_path,
            positive_samples_rgb=pos_rgb, negative_samples_rgb=neg_rgb,
            config=seg_cfg, output_dir=output_dir,
            roi_polygons=roi_polygons, ignore_polygons=ignore_polygons,
            tile_size=tile_size, overlap=overlap,
            skip_black_area=bool(seg_base.get("skip_black_area", True)),
            black_threshold=int(seg_base.get("black_threshold", 10)),
            valid_pixel_ratio_threshold=float(
                seg_base.get("valid_pixel_ratio_threshold", 0.1)
            ),
            extraction_label=f"opencv_{mode}",
            roi_required=(mode == "roi"),
            mask_type="formal_opencv",
        )
        worker.moveToThread(thread)
        self._opencv_formal_mode = mode
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_opencv_formal_progress)
        worker.finished.connect(self._on_opencv_formal_finished)
        worker.failed.connect(self._on_opencv_formal_failed)
        worker.cancelled.connect(self._on_opencv_formal_cancelled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(self._on_opencv_formal_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._segmentation_thread = thread
        self._segmentation_worker = worker
        self._segmentation_image_ref = os.path.abspath(self._layer_manager.image_path)
        self._param_panel.set_formal_extraction_running(True, mode)
        self._status_bar.show_message(
            f"大图 OpenCV 正式提取（{'ROI' if mode == 'roi' else '全图'}）已启动，"
            f"tile={tile_size}, overlap={overlap}"
        )
        thread.start()

    def _on_opencv_formal_progress(self, percent, current, total, message):
        detail = {
            "mode": getattr(self, "_opencv_formal_mode", ""),
            "infer_mode": "opencv",
            "stage": "infer_tiles",
            "success_tiles": current,
            "failed_tiles": 0,
            "cache_hit_tiles": 0,
        }
        self._param_panel.update_formal_extraction_progress(
            percent, current, total, detail
        )
        self._status_bar.show_message(f"{message} ({percent}%)")

    def _on_opencv_formal_finished(self, result):
        current_ref = (
            os.path.abspath(self._layer_manager.image_path)
            if self._layer_manager.is_large_image_mode else None
        )
        if (isinstance(self._segmentation_image_ref, str)
                and current_ref != self._segmentation_image_ref):
            self._status_bar.show_message("影像已改变，已丢弃旧影像的正式提取结果")
            return

        mask = result.processed_mask
        self._valid_image_mask = result.valid_image_mask
        self._valid_mask_report = dict(result.valid_mask_report)
        self._layer_manager.set_layer_data("valid_image_mask", result.valid_image_mask)
        self._layer_manager.hide_layer("valid_image_mask")
        preview_mask = getattr(result, "preview_mask", None)
        self._layer_manager.set_layer_data("mask", mask, preview_data=preview_mask)
        self._layer_manager.show_layer("mask")
        # ★ 正式提取原始结果作为新的 working mask 基线
        self._working_mask_source = "formal"
        self._working_mask_dirty = False
        self._working_mask_formal_ready = True
        self._working_mask_preview_only = False

        # ── 注册为正式 Road Mask（formal_opencv）──
        global_path = os.path.join(result.output_dir, "global_road_mask.png")
        self._formal_mask_meta = {
            "mask_type": "formal_opencv",
            "formal_ready": True,
            "preview_only": False,
            "coordinate_system": "original_image_pixel",
            "global_mask_path": global_path,
            "source": result.output_dir,
        }
        if self._large_image_project is not None:
            self._large_image_project.global_mask_path = global_path
            try:
                self._large_image_project.valid_image_mask_path = os.path.join(
                    result.output_dir, "valid_image_mask.png"
                )
                self._large_image_project.save()
            except Exception:
                pass
            if hasattr(self, "_project_manager") and self._project_manager is not None:
                self._project_manager.data.global_mask_path = global_path
                self._project_manager.mark_dirty()

        self._clear_skeleton_state(clear_layer=True)
        self._canvas.refresh_scene()
        self._canvas.viewport().update()
        total = mask.size
        road_px = int((mask > 0).sum())
        ratio = road_px / total if total else 0.0
        self._status_bar.update_road_ratio(ratio)

        report = result.report
        failed_roi = getattr(result, "failed_roi_tile_count", 0)
        elapsed = report.get("elapsed_seconds", 0.0)

        if failed_roi > 0:
            QMessageBox.warning(
                self, "部分 ROI tile 提取失败",
                f"有 {failed_roi} 个 ROI tile 提取失败，结果可能不完整。\n\n"
                f"详见:\n{report.get('tile_status_report_path', '')}\n"
                f"{report.get('tile_status_overlay_path', '')}"
            )

        self._status_bar.show_message(
            f"OpenCV 正式提取完成: {report.get('tile_count', 0)} tiles, "
            f"失败 {report.get('failed_tile_count', 0)}, 用时 {elapsed:.1f}s, "
            f"道路占比 {ratio*100:.1f}%"
        )

        msg = QMessageBox(self)
        msg.setWindowTitle("大图 OpenCV 正式提取完成")
        msg.setText(
            "已生成正式 Road Mask（mask_type = formal_opencv）。\n\n"
            f"处理 tile: {report.get('tile_count', 0)}\n"
            f"成功 tile: {report.get('success_tile_count', 0)}\n"
            f"失败 tile: {report.get('failed_tile_count', 0)}\n"
            f"跳过黑边 tile: {report.get('skipped_black_tile_count', 0)}\n"
            f"道路像素占比: {ratio*100:.1f}%\n"
            f"用时: {elapsed:.1f}s"
        )
        msg.setInformativeText(
            "该正式 mask 可进入 mask 后处理 / 骨架 / graph / 路径规划。\n"
            "快速预览结果不会冒充正式 mask。\n\n"
            f"global_road_mask.png:\n{global_path}"
        )
        msg.exec()

    def _on_opencv_formal_failed(self, stage, message, error_log_path):
        self._status_bar.show_message(f"OpenCV 正式提取失败于 {stage}；原 mask 未改变")
        QMessageBox.critical(
            self, "OpenCV 正式提取失败",
            f"当前提取后端: OpenCV（formal_opencv）\n"
            f"失败阶段: {stage}\n错误: {message}\n\n"
            f"原 mask、样本点、ROI 和 Ignore 均未改变。\n"
            f"错误日志: {error_log_path}"
        )

    def _on_opencv_formal_cancelled(self, message):
        self._status_bar.show_message("OpenCV 正式提取已取消；原 mask 未改变")

    def _on_opencv_formal_thread_finished(self):
        self._param_panel.set_formal_extraction_running(False)
        self._segmentation_worker = None
        self._segmentation_thread = None
        self._segmentation_image_ref = None

    def _begin_roi_drawing(self):
        """从任意页面进入 ROI 绘制，完成后自动回到道路分割页。"""
        if not self._layer_manager.has_image():
            QMessageBox.warning(self, "提示", "请先打开影像。")
            return
        self._roi_draw_return_stage = "segment"
        self._roi_draw_baseline_count = self._canvas.get_enabled_roi_count()
        self.set_stage("edit")
        self.set_tool("roi")
        self._layer_manager.show_layer("layer_roi")
        self._sync_layer_checkboxes()
        self._status_bar.show_message(
            "左键添加 ROI 顶点，右键或双击闭合，Esc 取消。"
        )

    def _finish_roi_drawing_session(self):
        """结束引导式 ROI 绘制并回到道路分割页。"""
        return_stage = self._roi_draw_return_stage or "segment"
        self._roi_draw_return_stage = None
        self.set_stage(return_stage)
        self._layer_manager.show_layer("layer_roi")
        self._sync_layer_checkboxes()
        self._param_panel.highlight_roi_extract_btn(True)
        self._refresh_roi_status_panel(recalculate=True)
        self._status_bar.show_message("ROI 已创建，可点击“ROI 正式提取”。")

    def _use_current_view_as_roi(self) -> bool:
        """将当前画布可见矩形保存为 original image pixel ROI。"""
        x0, y0, x1, y1 = self._canvas.get_visible_image_rect()
        if x1 <= x0 or y1 <= y0:
            QMessageBox.warning(self, "无法创建 ROI", "当前画布没有有效可见区域。")
            return False
        points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        if not self._canvas.add_roi_from_image_points(points):
            return False
        self._layer_manager.show_layer("layer_roi")
        self._sync_layer_checkboxes()
        self._param_panel.highlight_roi_extract_btn(True)
        self._refresh_roi_status_panel(recalculate=True)
        self._status_bar.show_message("已使用当前视野创建 ROI。")
        return True

    def _clear_roi_regions(self):
        if self._canvas.get_roi_regions():
            reply = QMessageBox.question(
                self, "清空 ROI", "确定清空全部 ROI 区域吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        self._canvas._clear_all_roi()
        self._canvas.clear_roi_tile_overlay()
        self._roi_tile_overlay_visible = False
        self._refresh_roi_status_panel()
        self._status_bar.show_message("ROI 已清空。")

    def _refresh_roi_status_panel(self, recalculate: bool = False):
        count = self._canvas.get_enabled_roi_count()
        if not recalculate or count == 0:
            self._param_panel.update_roi_status(count)
            return
        cfg = self._build_roi_estimate_config()
        estimate = self._estimate_roi_tiles(cfg) if cfg is not None else None
        if estimate:
            self._param_panel.update_roi_status(
                count, estimate["roi_tiles"], estimate["need_infer_tiles"]
            )
            self._status_bar.show_message(
                f"ROI 覆盖 {estimate['roi_tiles']} 个 tile，"
                f"待推理 {estimate['need_infer_tiles']} 个 tile。"
            )
        else:
            self._param_panel.update_roi_status(count)

    def _build_roi_estimate_config(self):
        """构建只用于 tile 预估/可视化的配置。"""
        image_path = self._layer_manager.image_path
        if not image_path or not os.path.isfile(image_path):
            return None
        from roadnet.formal_extraction_config import FormalExtractionConfig
        cfg = FormalExtractionConfig()
        ext_cfg = self._param_panel.get_config().get("extraction", {})
        cfg.image_path = Path(image_path)
        cfg.tile_size = int(ext_cfg.get("tile_size", 2048))
        cfg.tile_overlap = int(ext_cfg.get("tile_overlap", 128))
        cfg.skip_black_tile = bool(ext_cfg.get("skip_black_tile", True))
        cfg.valid_pixel_ratio_threshold = float(
            ext_cfg.get("valid_pixel_ratio_threshold", 0.10)
        )
        cfg.black_threshold = int(ext_cfg.get("black_threshold", 10))
        cfg.resume_from_existing_tiles = bool(
            ext_cfg.get("resume_from_existing_tiles", True)
        )
        cfg.roi_polygons = self._canvas.get_enabled_roi_polygon_points()
        if self._large_image_project is not None:
            cfg.output_dir = Path(
                os.path.join(self._large_image_project.project_dir, "formal_extraction")
            )
        return cfg

    def _show_roi_covered_tiles(self):
        if self._canvas.get_enabled_roi_count() == 0:
            self._on_extract_roi()
            return
        if self._roi_tile_overlay_visible:
            self._canvas.clear_roi_tile_overlay()
            self._roi_tile_overlay_visible = False
            self._status_bar.show_message("已隐藏 ROI 覆盖 tile。")
            return
        cfg = self._build_roi_estimate_config()
        estimate = self._estimate_roi_tiles(cfg) if cfg is not None else None
        if not estimate:
            QMessageBox.warning(self, "预估失败", "无法计算 ROI 覆盖 tile。")
            return
        self._canvas.show_roi_tile_overlay(estimate.get("tile_entries", []))
        self._layer_manager.show_layer("layer_roi")
        self._roi_tile_overlay_visible = True
        self._param_panel.update_roi_status(
            self._canvas.get_enabled_roi_count(),
            estimate["roi_tiles"],
            estimate["need_infer_tiles"],
        )
        self._status_bar.show_message(
            f"ROI 覆盖 {estimate['roi_tiles']} 个 tile，"
            f"待推理 {estimate['need_infer_tiles']} 个 tile。"
        )

    def _start_formal_extraction(self, mode: str):
        """启动正式道路提取（ROI 或全图）。"""
        if self._formal_extraction_thread is not None and self._formal_extraction_thread.isRunning():
            QMessageBox.information(self, "提取运行中", "正式道路提取已在运行。")
            return
        if self._segmentation_thread is not None and self._segmentation_thread.isRunning():
            QMessageBox.information(self, "分割运行中", "请等待传统分割完成。")
            return

        image_path = self._layer_manager.image_path
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "提示", "请先打开影像。")
            return

        is_large = self._layer_manager.is_large_image_mode
        if not is_large:
            QMessageBox.information(self, "小图模式",
                                    "小图建议使用「工具 → 运行 SAM-Road 单图初提取」。\n"
                                    "正式 tile 提取主要针对大图。")
            return

        config_data = self._param_panel.get_config()
        ext_cfg = config_data.get("extraction", {})

        # 构建 FormalExtractionConfig
        from roadnet.formal_extraction_config import (
            FormalExtractionConfig, EXTRACTION_MODE_ROI, EXTRACTION_MODE_FULL,
            INFER_MODE_PERSISTENT, INFER_MODE_SUBPROCESS,
        )
        from roadnet.samroad_single_runner import (
            SAMRoadSingleRunConfig, dict_to_runconfig, load_config as load_samroad_config,
        )

        # 从 UI 配置构建基础 config
        samroad_cfg = dict_to_runconfig(config_data)

        # ── 关键修复：如果 python_executable 为空/无效，从持久化YAML配置加载 ──
        if not samroad_cfg.python_executable or str(samroad_cfg.python_executable) == ".":
            saved = load_samroad_config()
            samroad_cfg = dict_to_runconfig({"samroad_single": saved.get("samroad_single", {})})
            if not samroad_cfg.python_executable or str(samroad_cfg.python_executable) == ".":
                # 最后回退：使用当前 Python
                import sys as _sys
                samroad_cfg.python_executable = Path(_sys.executable)
        cfg = FormalExtractionConfig()
        cfg.extraction_mode = EXTRACTION_MODE_ROI if mode == "roi" else EXTRACTION_MODE_FULL
        cfg.infer_mode = str(ext_cfg.get("infer_mode", INFER_MODE_PERSISTENT))
        cfg.tile_size = int(ext_cfg.get("tile_size", 2048))
        cfg.tile_overlap = int(ext_cfg.get("tile_overlap", 128))
        cfg.tile_batch_size = min(int(ext_cfg.get("tile_batch_size", 1)), 4)
        cfg.max_tiles = int(ext_cfg.get("max_tiles", 0)) or None
        cfg.max_tiles_for_test = int(ext_cfg.get("max_tiles_for_test", 0))
        cfg.model_load_timeout_seconds = int(ext_cfg.get("model_load_timeout_seconds", 180))
        cfg.heartbeat_interval_seconds = float(ext_cfg.get("heartbeat_interval_seconds", 2.0))
        cfg.skip_black_tile = bool(ext_cfg.get("skip_black_tile", True))
        cfg.skip_black_ratio_threshold = float(ext_cfg.get("skip_black_ratio_threshold", 0.80))
        cfg.black_threshold = int(ext_cfg.get("black_threshold", 10))
        cfg.valid_pixel_ratio_threshold = float(ext_cfg.get("valid_pixel_ratio_threshold", 0.10))
        cfg.merge_method = str(ext_cfg.get("merge_method", "max"))
        cfg.resume_from_existing_tiles = bool(ext_cfg.get("resume_from_existing_tiles", True))
        cfg.debug_mode = bool(ext_cfg.get("debug_mode", False))
        cfg.device = samroad_cfg.device
        cfg.image_path = Path(image_path)
        cfg.project_dir = samroad_cfg.project_dir
        cfg.python_executable = samroad_cfg.python_executable
        cfg.infer_script = samroad_cfg.infer_script
        cfg.config_path = samroad_cfg.config_path
        cfg.sam_backbone_ckpt_path = samroad_cfg.sam_backbone_ckpt_path
        cfg.samroad_model_ckpt_path = samroad_cfg.samroad_model_ckpt_path

        # ── 推理后端选择：默认 SAMRoadPlus Portable，覆盖旧 SAM-Road 单图路径 ──
        from roadnet.portable_samroadplus_runner import (
            apply_portable_config, resolve_portable_project_dir,
            is_portable_project,
        )
        backend = str(ext_cfg.get("inference_backend", "samroadplus_portable")).strip()
        portable_dir_cfg = ext_cfg.get("portable_project_dir", "")
        resolved_portable_dir = resolve_portable_project_dir(portable_dir_cfg)
        use_portable = (
            backend == "samroadplus_portable"
            or is_portable_project(cfg.project_dir)
            or is_portable_project(resolved_portable_dir)
        )
        if use_portable:
            cfg.adapter_type = apply_portable_config(cfg, portable_dir_cfg)
            # Portable 第一版强制 subprocess_per_tile，不使用旧持久化 worker
            cfg.infer_mode = INFER_MODE_SUBPROCESS
        else:
            cfg.adapter_type = "old_samroad"
        self._current_adapter_type = cfg.adapter_type

        if self._large_image_project is not None:
            tile_index = getattr(self._large_image_project, "tile_index_path", "")
            if tile_index and os.path.isfile(tile_index):
                cfg.tile_index_path = Path(tile_index)

        # 比赛模式：覆盖参数
        competition = bool(config_data.get("segment", {}).get("competition_fast_mode", False))
        if competition:
            cfg.competition_fast_mode = True
            cfg.apply_competition_mode()

        # ROI 多边形
        if mode == "roi":
            roi_polygons = self._canvas.get_enabled_roi_polygon_points()
            if not roi_polygons:
                self._status_bar.show_message("未设置 ROI，已取消 ROI 正式提取。")
                QMessageBox.warning(
                    self, "需要 ROI 区域",
                    "当前没有启用的 ROI 区域，ROI 正式提取已取消。"
                )
                return
            cfg.roi_polygons = roi_polygons

        # 输出目录 — 使用安全唯一目录，不强删旧目录
        from roadnet.formal_extraction_worker import safe_create_output_dir
        if self._large_image_project is not None:
            base_dir = os.path.join(self._large_image_project.project_dir, "formal_extraction")
        else:
            base_dir = os.path.join("outputs", "formal_extraction")
        cfg.output_dir = Path(safe_create_output_dir(base_dir, mode))

        # ── ROI tile 预估（仅 ROI 模式）──
        if mode == "roi":
            est_info = self._estimate_roi_tiles(cfg)
            if est_info:
                self._param_panel.update_roi_status(
                    self._canvas.get_enabled_roi_count(),
                    est_info["roi_tiles"],
                    est_info["need_infer_tiles"],
                )
                self._status_bar.show_message(
                    f"ROI 覆盖 {est_info['roi_tiles']} 个 tile，"
                    f"待推理 {est_info['need_infer_tiles']} 个 tile。"
                )
                extra = (
                    "\n\nROI 范围较大，预计耗时较长，建议缩小 ROI。"
                    if est_info["need_infer_tiles"] > 30 else ""
                )
                dialog = QMessageBox(self)
                dialog.setIcon(QMessageBox.Icon.Information)
                dialog.setWindowTitle("ROI Tile 预估")
                dialog.setText(
                    f"ROI 覆盖 {est_info['roi_tiles']} 个 tile，"
                    f"其中 {est_info['cache_hit_tiles']} 个已缓存，"
                    f"实际需要推理 {est_info['need_infer_tiles']} 个 tile。"
                    f"\n跳过黑边 tile：{est_info['skipped_black_tiles']} / "
                    f"总 tile：{est_info['total_tiles']}"
                    + extra
                )
                start_button = dialog.addButton(
                    "开始提取", QMessageBox.ButtonRole.AcceptRole
                )
                adjust_button = dialog.addButton(
                    "调整 ROI", QMessageBox.ButtonRole.ActionRole
                )
                dialog.addButton("取消", QMessageBox.ButtonRole.RejectRole)
                dialog.exec()
                if dialog.clickedButton() is adjust_button:
                    self._begin_roi_drawing()
                    return
                if dialog.clickedButton() is not start_button:
                    return

        # ── 启动前路径检查（任何一项失败则弹窗，不启动 worker）──
        from roadnet.formal_extraction_worker import (
            FormalExtractionWorker, validate_paths_for_extraction,
        )
        from roadnet.portable_samroadplus_runner import is_portable_project

        path_errors = validate_paths_for_extraction(cfg)
        if path_errors:
            portable = is_portable_project(cfg.project_dir)
            title = "Portable 路径检查失败" if portable else "路径检查失败"
            QMessageBox.critical(
                self, title,
                "正式提取无法启动，请修复以下问题：\n\n  " + "\n  ".join(path_errors),
            )
            self._status_bar.show_message("正式提取路径检查失败，已取消。")
            return

        # 创建 Worker 和 Thread
        from PySide6.QtCore import QThread

        thread = QThread(self)
        worker = FormalExtractionWorker(cfg)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        worker.log.connect(self._on_formal_extraction_log)
        worker.stage_progress.connect(self._on_formal_extraction_stage)
        worker.heartbeat.connect(self._on_formal_extraction_heartbeat)
        worker.progress.connect(self._on_formal_extraction_progress)
        worker.finished.connect(self._on_formal_extraction_finished)
        worker.failed.connect(self._on_formal_extraction_failed)
        worker.cancelled.connect(self._on_formal_extraction_cancelled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(self._on_formal_extraction_thread_finished)
        thread.finished.connect(thread.deleteLater)

        self._formal_extraction_thread = thread
        self._formal_extraction_worker = worker

        mode_label = "ROI" if mode == "roi" else "全图"
        self._param_panel.set_formal_extraction_running(True, mode)
        self._status_bar.show_message(
            f"正式道路提取 ({mode_label}) 已启动: tile={cfg.tile_size}, "
            f"infer={cfg.infer_mode}"
        )
        thread.start()

    def _on_formal_extraction_progress(self, percent, current, total, detail):
        stage = detail.get("stage", "")
        stage_label = {
            "infer_tiles": "推理",
            "load_model": "加载模型",
            "launch_worker": "启动进程",
            "select_tiles": "选择 tile",
        }.get(stage, "")
        prefix = f"[{stage_label}] " if stage_label else ""
        tile_id = detail.get("tile_id", "")
        self._param_panel.update_formal_extraction_progress(
            percent, current, total, detail
        )
        if stage in ("load_model", "launch_worker") and total == 0:
            self._status_bar.show_message(
                f"{prefix}{detail.get('stage_label', stage_label)} · "
                f"已用 {detail.get('elapsed_seconds', 0):.0f}s"
            )
            return
        msg = f"{prefix}提取进度 {percent}% · tile {current}/{total}"
        if tile_id:
            msg += f" ({tile_id})"
        self._status_bar.show_message(msg)

    def _on_formal_extraction_stage(self, stage: str, label: str,
                                     elapsed: float, detail):
        """阶段变更处理。"""
        self._status_bar.show_message(
            f"[正式提取] 阶段: {label} · 已用 {elapsed:.0f}s"
        )
        # 在模型加载阶段显示明确信息
        if stage in ("launch_worker", "load_model"):
            self._param_panel.update_formal_extraction_progress(
                0, 0, 0, {
                    "stage": stage,
                    "stage_label": label,
                    "elapsed_seconds": elapsed,
                }
            )
        # 记录阶段耗时到 detail（传递给后续处理）

    def _on_formal_extraction_heartbeat(self, elapsed: float, step: str, message: str):
        """模型加载心跳处理。"""
        self._status_bar.show_message(
            f"正在加载模型，已用时 {elapsed:.1f}s — {step}"
        )

    def _estimate_roi_tiles(self, cfg):
        """预估 ROI 覆盖的 tile 数量和缓存命中数。"""
        try:
            from roadnet.segmentation_worker import generate_tile_grid
            from roadnet.formal_extraction_worker import _tile_intersects_roi
            from roadnet.valid_image import analyze_valid_image_mask
            from roadnet.large_image_project import ImageRegionReader

            if not os.path.isfile(str(cfg.image_path)):
                return None

            reader = ImageRegionReader(str(cfg.image_path))
            width, height = reader.size

            preview = reader.read_preview(3000)
            valid_mask, _ = analyze_valid_image_mask(
                preview, cfg.black_threshold,
                max(64, int(cfg.min_black_component_area * (preview.shape[1] / width) ** 2)),
            )
            valid_mask = cv2.resize(valid_mask, (width, height),
                                     interpolation=cv2.INTER_NEAREST)

            candidates = generate_tile_grid(width, height, cfg.tile_size, cfg.tile_overlap)
            roi_tiles = []
            skipped_black_tiles = []
            for tile in candidates:
                x0, y0, x1, y1 = tile
                if not cfg.roi_polygons or not _tile_intersects_roi(tile, cfg.roi_polygons):
                    continue
                if cfg.skip_black_tile:
                    tile_valid = valid_mask[y0:y1, x0:x1]
                    tile_valid_ratio = float(np.count_nonzero(tile_valid)) / float(tile_valid.size)
                    if tile_valid_ratio < cfg.valid_pixel_ratio_threshold:
                        skipped_black_tiles.append(tile)
                        continue
                roi_tiles.append(tile)

            # 检查缓存
            tiles_root = os.path.join(
                str(cfg.output_dir) if str(cfg.output_dir) != "." else "outputs",
                "tiles"
            )
            cache_hits = 0
            tile_entries = [
                {"tile": tile, "category": "skipped_black"}
                for tile in skipped_black_tiles
            ]
            for i, tile in enumerate(roi_tiles, 1):
                cache_mask_path = os.path.join(tiles_root, f"tile_{i:06d}_mask.png")
                if cfg.resume_from_existing_tiles and os.path.isfile(cache_mask_path):
                    cache_hits += 1
                    category = "cached"
                else:
                    category = "need_infer"
                tile_entries.append({"tile": tile, "category": category})

            return {
                "total_tiles": len(candidates),
                "roi_tiles": len(roi_tiles),
                "skipped_black_tiles": len(skipped_black_tiles),
                "cache_hit_tiles": cache_hits,
                "need_infer_tiles": len(roi_tiles) - cache_hits,
                "tile_entries": tile_entries,
            }
        except Exception:
            return None

    def _show_tile_failure_dialog(self, failed_details: list, processed: int, failed: int):
        """显示 tile 推理失败诊断（含 command/cwd/stdout/stderr 路径）。"""
        first = failed_details[0] if failed_details else {}
        command = first.get("command", [])
        command_str = " ".join(str(c) for c in command) if command else "(无)"
        adapter_type = getattr(self, "_current_adapter_type", "unknown")
        lines = [
            f"当前推理后端: {adapter_type}",
            "",
            f"处理成功 {processed} 个，失败 {failed} 个 tile。",
            "",
            f"tile_id: {first.get('tile_id', '')}",
            f"return_code: {first.get('return_code', 'N/A')}",
            f"cwd: {first.get('cwd', '')}",
            f"stdout: {first.get('stdout_path', '')}",
            f"stderr: {first.get('stderr_path', '')}",
        ]
        candidates = first.get("mask_candidates")
        if candidates is not None:
            lines.append(f"mask 候选: {candidates if candidates else '未找到任何候选 mask'}")
        lines.append("")
        lines.append(f"command:\n{command_str}")

        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle("tile 推理失败")
        dialog.setText("\n".join(lines))
        err = first.get("error", "")
        if err:
            dialog.setDetailedText(str(err))
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.exec()

    def _on_formal_extraction_log(self, message: str):
        self._status_bar.show_message(message[:200])

    def _on_formal_extraction_finished(self, report: dict):
        """正式提取完成后：加载 global_road_mask 但不自动 skeleton。"""
        elapsed = report.get("elapsed_seconds", 0)
        mode = report.get("mode", "")
        mode_label = {"roi": "ROI", "full": "全图", "fast_preview": "快速预览"}.get(mode, mode)
        processed = report.get("processed_tiles", 0)
        failed = report.get("failed_tiles", 0)
        cache_hits = report.get("cache_hit_tiles", 0)
        non_zero_ratio = report.get("global_mask_nonzero_ratio", 0)

        mask_path = report.get("output_global_mask_path", "")
        preview_path = report.get("output_preview_path", "")

        # ── tile 推理失败：显示完整诊断（command/cwd/stdout/stderr）──
        failed_details = report.get("failed_tile_details", []) or []
        if failed_details and (processed == 0 or failed > 0):
            self._show_tile_failure_dialog(failed_details, processed, failed)
            if processed == 0:
                self._status_bar.show_message(
                    f"正式提取失败：{failed} 个 tile 全部失败，未生成 road_mask。"
                )
                return

        if mask_path and os.path.isfile(mask_path):
            try:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                # 加载 preview
                preview_mask = None
                if preview_path and os.path.isfile(preview_path):
                    preview_mask = cv2.imread(preview_path, cv2.IMREAD_GRAYSCALE)

                # SAMRoadPlus/SAM-Road 模型正式提取结果。
                self._formal_mask_meta = {
                    "mask_type": "formal_model",
                    "formal_ready": True,
                    "preview_only": False,
                    "coordinate_system": "original_image_pixel",
                    "global_mask_path": mask_path,
                }
                self._layer_manager.set_layer_data("mask", mask, preview_data=preview_mask)
                self._layer_manager.show_layer("mask")

                # 更新大图项目
                if self._large_image_project is not None:
                    self._large_image_project.global_mask_path = mask_path
                    self._large_image_project.save()
                    if hasattr(self, '_project_manager') and self._project_manager is not None:
                        self._project_manager.data.global_mask_path = mask_path
                        self._project_manager.mark_dirty()

                # ★ 不自动执行 skeleton / graph
                self._clear_skeleton_state(clear_layer=True)
                self._canvas.refresh_scene()
                self._canvas.viewport().update()
                total = mask.size
                road_px = int((mask > 0).sum())
                ratio = road_px / total if total else 0.0
                self._status_bar.update_road_ratio(ratio)

                # 显示报告路径
                report_dir = os.path.dirname(mask_path)
                report_file = os.path.join(report_dir, "formal_extraction_report.json")
                report_tip = ""
                if os.path.isfile(report_file):
                    report_tip = f"\n报告: {report_file}"

                self._status_bar.show_message(
                    f"正式提取完成 ({mode_label}): {processed} tiles, "
                    f"失败 {failed}, 缓存命中 {cache_hits}, 用时 {elapsed:.0f}s"
                )

                msg = QMessageBox(self)
                msg.setWindowTitle("正式道路提取完成")
                msg.setText(
                    f"正式道路提取 ({mode_label}) 已完成。\n\n"
                    f"处理 tile: {processed} 个\n"
                    f"失败 tile: {failed} 个\n"
                    f"缓存命中: {cache_hits} 个\n"
                    f"道路像素占比: {non_zero_ratio*100:.1f}%\n"
                    f"总用时: {elapsed:.0f}s\n"
                    f"平均每 tile: {report.get('avg_time_per_tile', 0):.1f}s"
                    f"{report_tip}"
                )
                msg.setInformativeText(
                    "Road Mask 已加载到画布。\n\n"
                    "下一步：进入「编辑」阶段检查 mask 质量，\n"
                    "确认后点击「进入骨架生成」手动执行后续流程。"
                )
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec()

            except Exception as e:
                QMessageBox.critical(self, "加载 mask 失败", f"无法加载 road_mask: {e}")
        else:
            QMessageBox.warning(self, "提取完成但无 mask",
                                f"正式提取完成，但未生成 global_road_mask。\n"
                                f"路径: {mask_path}")

    def _on_formal_extraction_failed(self, message: str, log_path: str):
        self._status_bar.show_message(f"正式提取失败: {message}")

        # 确保不污染状态：不加载 mask、不进入 skeleton
        self._param_panel.set_formal_extraction_running(False)
        self._formal_extraction_worker = None
        self._formal_extraction_thread = None

        # 构建增强错误弹窗
        adapter_type = getattr(self, "_current_adapter_type", "unknown")
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("正式道路提取失败")
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(
            f"当前推理后端: {adapter_type}\n"
            f"（若显示 old_samroad 表示未切换到 Portable）\n\n"
            f"正式道路提取过程中发生错误：\n\n{message}"
        )
        msg_box.setInformativeText(
            f"错误日志: {log_path}\n\n"
            "常见原因:\n"
            "1. Python 解释器路径无效（指向了目录而非 exe）\n"
            "2. 输出目录无写权限或被占用\n"
            "3. SAM-Road 模型文件缺失\n"
            "4. 杀毒软件拦截了子进程启动"
        )

        # 按钮
        open_log_btn = msg_box.addButton("打开错误日志", QMessageBox.ActionRole)
        open_dir_btn = msg_box.addButton("打开输出目录", QMessageBox.ActionRole)
        copy_btn = msg_box.addButton("复制错误信息", QMessageBox.ActionRole)
        close_btn = msg_box.addButton("关闭", QMessageBox.RejectRole)

        msg_box.setDefaultButton(close_btn)
        msg_box.exec()

        clicked = msg_box.clickedButton()
        if clicked == open_log_btn:
            if os.path.isfile(log_path):
                os.startfile(log_path)
            else:
                QMessageBox.warning(self, "日志不存在", f"日志文件未找到: {log_path}")
        elif clicked == open_dir_btn:
            log_dir = os.path.dirname(log_path)
            if os.path.isdir(log_dir):
                os.startfile(log_dir)
            else:
                QMessageBox.warning(self, "目录不存在", f"输出目录未找到: {log_dir}")
        elif clicked == copy_btn:
            from PySide6.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            clipboard.setText(f"正式提取错误:\n{message}\n\n日志: {log_path}")

    def _on_formal_extraction_cancelled(self, message: str):
        self._status_bar.show_message(f"正式提取已取消: {message}")

    def _on_formal_extraction_thread_finished(self):
        # 安全重置，可能已被 failed 处理过
        if self._formal_extraction_worker is not None or self._formal_extraction_thread is not None:
            self._param_panel.set_formal_extraction_running(False)
        self._formal_extraction_worker = None
        self._formal_extraction_thread = None

    def _on_apply_roi(self):
        mask = self._layer_manager.get_layer_data("mask")
        if mask is None:
            QMessageBox.warning(self, "应用 ROI", "当前没有可编辑的 Road Mask。")
            return
        from roadnet.region_edit import ensure_mask_image_size
        try:
            image_size = self._layer_manager.original_size
            if image_size[0] <= 0 or image_size[1] <= 0:
                image_size = self._layer_manager.image_size
            mask = ensure_mask_image_size(mask, image_size)
            self._layer_manager.set_layer_data("mask", mask)
        except (TypeError, ValueError) as exc:
            QMessageBox.critical(self, "应用 ROI 失败", str(exc))
            return
        roi_regions = self._canvas.get_roi_regions()
        if not roi_regions:
            QMessageBox.information(self, "应用 ROI", "没有 ROI 区域")
            return
        self._history.push_state("apply_roi")
        from roadnet.region_edit import apply_roi_regions
        try:
            new_mask, affected = apply_roi_regions(mask, roi_regions)
        except (TypeError, ValueError) as exc:
            QMessageBox.critical(self, "应用 ROI 失败", str(exc))
            return
        self._layer_manager.set_layer_data("mask", new_mask)
        total = new_mask.size
        road_px = (new_mask > 0).sum()
        self._status_bar.update_road_ratio(road_px / total)
        self._status_bar.show_message(
            f"ROI 已应用：{len(roi_regions)} 个区域，清除 {affected} 个道路像素"
        )

    def _on_apply_ignore(self):
        """应用 Ignore 区域 — 使用 cv2.fillPoly 从 mask 中删除多边形内部区域"""
        mask = self._layer_manager.get_layer_data("mask")
        if mask is None:
            QMessageBox.warning(self, "提示", "请先执行分割。")
            return
        from roadnet.region_edit import ensure_mask_image_size
        try:
            image_size = self._layer_manager.original_size
            if image_size[0] <= 0 or image_size[1] <= 0:
                image_size = self._layer_manager.image_size
            mask = ensure_mask_image_size(mask, image_size)
            self._layer_manager.set_layer_data("mask", mask)
        except (TypeError, ValueError) as exc:
            QMessageBox.critical(self, "应用 Ignore 失败", str(exc))
            return

        # ★ 统一使用 get_ignore_polygons()（兼容多边形 + 旧矩形自动转多边形）
        ignore_regions = self._canvas.get_ignore_regions()
        if not ignore_regions:
            QMessageBox.information(self, "应用 Ignore", "没有 Ignore 区域")
            return

        self._history.push_state("apply_ignore")
        from roadnet.region_edit import apply_ignore_regions
        try:
            new_mask, affected = apply_ignore_regions(mask, ignore_regions)
        except (TypeError, ValueError) as exc:
            QMessageBox.critical(self, "应用 Ignore 失败", str(exc))
            return
        self._layer_manager.set_layer_data("mask", new_mask)
        total = new_mask.size
        road_px = (new_mask > 0).sum()
        self._status_bar.update_road_ratio(road_px / total)
        self._status_bar.show_message(
            f"Ignore 已应用：{len(ignore_regions)} 个区域，删除 {affected} 个道路像素"
        )

    # ===================================================================
    # 参数应用
    # ===================================================================

    def _clear_mask_candidate_items(self):
        scene = self._canvas.scene() if self._canvas is not None else None
        if scene is not None:
            for item in self._mask_candidate_items:
                safe_remove_scene_item(scene, item)
        self._mask_candidate_items.clear()

    def _render_mask_candidates(self):
        self._clear_mask_candidate_items()
        scene = self._canvas.scene()
        if scene is None:
            return
        visualization = self._param_panel.get_config().get("visualization", {})
        show_numbers = bool(visualization.get("show_mask_candidate_numbers", True))
        show_reasons = bool(visualization.get("show_mask_candidate_reasons", False))
        # Clean mode is intended for an uncluttered review/export view.
        show_numbers = show_numbers and not self._clean_mode
        visible_candidates = [
            candidate for candidate in self._mask_ignore_candidates
            if candidate.get("status") != "rejected"
        ]
        visible_candidates.sort(
            key=lambda item: (
                item.get("status") == "applied",
                not bool(item.get("auto_apply_eligible", False)),
                float(item.get("area_ratio", 0)),
                float(item.get("confidence", 0)),
            ),
            reverse=True,
        )
        # Thousands of one-pixel candidates can otherwise freeze QGraphicsScene.
        # The full set remains available in JSON and the review dialog.
        visible_candidates = visible_candidates[:200]
        for candidate_index, candidate in enumerate(visible_candidates):
            applied = candidate.get("status") == "applied"
            color = QColor(80, 210, 90, 55) if applied else QColor(255, 55, 55, 45)
            outline = QColor(70, 220, 90) if applied else QColor(255, 70, 70)
            first_preview = None
            for points in candidate.get("polygons", [candidate.get("polygon", [])]):
                if len(points) < 3:
                    continue
                preview = [
                    QPointF(*self._layer_manager.global_to_preview_f(p[0], p[1]))
                    for p in points
                ]
                first_preview = first_preview or preview[0]
                item = QGraphicsPolygonItem(QPolygonF(preview))
                if applied or candidate.get("auto_apply_eligible", False):
                    item.setBrush(QBrush(color))
                else:
                    item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                item.setPen(QPen(outline, 2, Qt.PenStyle.DashLine))
                item.setZValue(self._canvas.ZVAL_IGNORE + 2)
                item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                item.setToolTip(
                    f"{candidate.get('id')} | {candidate.get('reason')} | "
                    f"confidence={candidate.get('confidence', 0):.2f}"
                )
                scene.addItem(item)
                self._mask_candidate_items.append(item)
            if show_numbers and candidate_index < 20 and first_preview is not None:
                label = str(candidate.get("id", "Ignore"))
                if show_reasons:
                    label += " " + str(candidate.get("reason", ""))
                text = QGraphicsSimpleTextItem(label)
                text.setBrush(QBrush(QColor(255, 255, 255)))
                text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                text.setPos(first_preview)
                text.setZValue(self._canvas.ZVAL_IGNORE + 3)
                scene.addItem(text)
                self._mask_candidate_items.append(text)

    def _on_analyze_mask_quality(self):
        mask = self._layer_manager.get_layer_data("mask")
        if mask is None:
            QMessageBox.warning(self, "Mask 自动筛选", "当前没有 road_mask / processed_mask。")
            return
        from datetime import datetime
        from roadnet.mask_quality_filter import filter_mask_quality
        output_dir = os.path.join(
            os.getcwd(), "outputs", "mask_filter",
            "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        try:
            result = filter_mask_quality(
                mask, processed_mask=mask,
                valid_image_mask=self._valid_image_mask,
                roi_polygons=self._canvas.get_roi_polygons(),
                final_graph=self._graph_editor,
                planned_path=self.planned_path_pixel,
                task_points=self._task_points,
                snapped_task_points=self._snapped_points,
                output_dir=output_dir,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Mask 自动筛选失败", str(exc))
            return
        self._mask_ignore_candidates = result.candidate_ignore_regions
        self._mask_candidate_signature = self._mask_data_signature(mask)
        self._last_mask_filter_output_dir = output_dir
        self._last_mask_filter_report = result.report
        self._mask_before_auto_ignore = np.asarray(mask).copy()
        self._render_mask_candidates()
        high_count = int(result.report.get("high_confidence_count", 0))
        blocked = bool(result.report.get("auto_apply_blocked", False))
        block_reasons = result.report.get("auto_apply_block_reasons", [])
        if "high_confidence_candidate_count_exceeded" in block_reasons:
            blocked_reason = "高置信候选数量过多，疑似碎片噪声，禁止批量应用；请逐个确认。"
        else:
            blocked_reason = "高置信 Ignore 总面积过大，疑似误判，请降低自动应用范围或逐个确认。"
        self._param_panel.set_mask_candidate_apply_enabled(
            high_count > 0 and not blocked,
            blocked_reason if blocked else
            "仅应用安全检查通过且 confidence >= 0.90 的候选；支持 Ctrl+Z 撤销",
        )
        QMessageBox.information(
            self, "Mask 自动诊断完成",
            f"共分析 {result.report['component_count']} 个连通域。\n"
            f"疑似误检/警告：{len(self._mask_ignore_candidates)} 个\n"
            f"高置信 Ignore：{high_count} 个\n\n"
            f"结果目录：\n{output_dir}\n\n"
            "红色区域仅为候选，尚未修改 Mask；画面最多显示 200 个候选轮廓。",
        )

    def _apply_mask_candidates(self, selected_ids=None):
        if self._region_edit_stable_mode:
            print("[RegionEdit] automatic high-confidence Ignore is disabled in stable mode")
            return 0
        if not self._mask_ignore_candidates:
            QMessageBox.information(self, "应用 Ignore 候选", "请先执行「自动筛选 Mask 误检」。")
            return 0
        if selected_ids is None:
            selected = [item for item in self._mask_ignore_candidates
                        if item.get("status") == "pending"
                        and bool(item.get("auto_apply_eligible", False))
                        and float(item.get("confidence", 0)) >= 0.90]
        else:
            selected_ids = set(selected_ids)
            selected = [item for item in self._mask_ignore_candidates
                        if item.get("status") == "pending" and item.get("id") in selected_ids]
        if not selected:
            QMessageBox.information(self, "应用 Ignore 候选", "没有符合条件的待处理候选。")
            return 0
        mask = self._layer_manager.get_layer_data("mask")
        if mask is None:
            return 0
        if (self._mask_candidate_signature is not None
                and self._mask_candidate_signature != self._mask_data_signature(mask)):
            QMessageBox.warning(
                self, "Ignore 候选已失效",
                "Mask 在诊断后已发生变化，请重新执行「自动筛选 Mask 误检」。",
            )
            return 0
        from roadnet.mask_quality_filter import (
            MaskQualityFilterConfig,
            add_candidate_runs_to_mask,
            update_mask_filter_apply_outputs,
        )
        cfg = MaskQualityFilterConfig()
        current = np.asarray(mask)
        current_binary = current[..., 0] if current.ndim == 3 else current
        ignore_mask = np.zeros(current_binary.shape, dtype=np.uint8)
        for candidate in selected:
            add_candidate_runs_to_mask(ignore_mask, candidate)
        # Absolute safety rule: only road-mask foreground pixels can change.
        ignore_mask[current_binary <= 0] = 0
        affected_pixels = int(np.count_nonzero(ignore_mask))
        total_area_ratio = affected_pixels / float(max(1, current_binary.size))
        near_graph = sum(bool(item.get("near_final_graph") or item.get("near_planned_path")) for item in selected)
        near_tasks = sum(bool(item.get("near_task_points")) for item in selected)
        automatic_batch = selected_ids is None
        exceeds = automatic_batch and (
            total_area_ratio > cfg.max_total_ignore_area_ratio
            or len(selected) > cfg.max_high_confidence_candidate_count
        )
        preview_text = (
            f"即将应用 Ignore：{len(selected)} 个\n"
            f"总面积占比：{total_area_ratio * 100:.3f}%（上限 8%）\n"
            f"影响 road_mask 像素：{affected_pixels}\n"
            f"接近 final_graph / planned_path：{near_graph} 个\n"
            f"接近 task_points：{near_tasks} 个\n"
            f"候选数量：{len(selected)}（批量上限 {cfg.max_high_confidence_candidate_count}）\n"
            f"安全阈值：{'超过，禁止批量应用' if exceeds else '通过'}"
        )
        if exceeds:
            safety_message = (
                "高置信候选数量过多，疑似碎片噪声，请缩小自动应用范围或逐个确认。"
                if len(selected) > cfg.max_high_confidence_candidate_count else
                "高置信 Ignore 总面积过大，疑似误判，请降低自动应用范围或逐个确认。"
            )
            self._param_panel.set_mask_candidate_apply_enabled(
                False, safety_message
            )
            QMessageBox.warning(
                self, "高置信 Ignore 安全检查未通过",
                preview_text + "\n\n" + safety_message,
            )
            return 0
        reply = QMessageBox.question(
            self, "应用高置信 Ignore - 预检",
            preview_text + "\n\n确认只从当前 road_mask 中清除这些候选像素？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return 0

        self._history.push_state("apply_auto_ignore_candidates")
        before = current.copy()
        updated = current.copy()
        updated[ignore_mask > 0] = 0
        applied_count = 0
        for candidate in selected:
            polygons = candidate.get("polygons", [candidate.get("polygon", [])])
            filled_polygon_area = sum(
                abs(cv2.contourArea(np.asarray(points, dtype=np.float32)))
                for points in polygons if len(points) >= 3
            )
            # Only persist a formal polygon when filling it later cannot erase
            # a large hole/background region. The current application always
            # uses exact component runs regardless of this display decision.
            persist_as_polygon = filled_polygon_area <= max(1.0, float(candidate.get("area", 0)) * 1.20)
            for points in polygons:
                if persist_as_polygon and len(points) >= 3:
                    self._canvas._ignore_polygons.append(
                        QPolygonF([QPointF(float(x), float(y)) for x, y in points])
                    )
            candidate["status"] = "applied"
            applied_count += 1
        self._layer_manager.set_layer_data("mask", updated)
        self._mask_candidate_signature = self._mask_data_signature(updated)
        self._clear_skeleton_state(clear_layer=True)
        self._canvas.refresh_scene()
        self._render_mask_candidates()
        self._param_panel.update_counts(ignore=len(self._canvas.get_ignore_polygons()))
        self._status_bar.update_road_ratio(float(np.count_nonzero(updated)) / max(1, updated.size))
        if self._last_mask_filter_output_dir:
            self._last_mask_filter_report = update_mask_filter_apply_outputs(
                self._last_mask_filter_output_dir,
                before, updated, self._mask_ignore_candidates,
                self._last_mask_filter_report, applied_count,
            )
        return applied_count

    def _on_apply_high_confidence_ignore(self):
        if self._region_edit_stable_mode:
            QMessageBox.information(
                self, "区域修正稳定模式",
                "自动应用高置信 Ignore 已暂停。请查看候选后使用手工 Ignore 多边形确认。",
            )
            return
        count = self._apply_mask_candidates()
        if count:
            QMessageBox.information(
                self, "高置信 Ignore 已应用",
                f"已应用 {count} 个候选并转换为正式 Ignore polygon。\n可按 Ctrl+Z 撤销。",
            )

    def _on_view_mask_candidates(self):
        pending = [item for item in self._mask_ignore_candidates if item.get("status") == "pending"]
        if not pending:
            QMessageBox.information(self, "Ignore 候选", "当前没有待确认候选。")
            return
        labels = [f"{item['id']} | {item['reason']} | confidence={item['confidence']:.2f}" for item in pending]
        selected, ok = QInputDialog.getItem(self, "查看 Ignore 候选", "选择候选：", labels, 0, False)
        if not ok:
            return
        candidate = pending[labels.index(selected)]
        box = QMessageBox(self)
        box.setWindowTitle(candidate["id"])
        box.setText(
            f"原因：{candidate['reason']}\n置信度：{candidate['confidence']:.2f}\n"
            f"bbox：{candidate.get('bbox')}"
        )
        apply_button = box.addButton("确认并应用", QMessageBox.ButtonRole.AcceptRole)
        reject_button = box.addButton("拒绝候选", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is apply_button:
            self._apply_mask_candidates({candidate["id"]})
        elif box.clickedButton() is reject_button:
            candidate["status"] = "rejected"
            self._render_mask_candidates()

    def _clear_graph_repair_items(self):
        scene = self._canvas.scene() if self._canvas is not None else None
        if scene is not None:
            for item in self._graph_repair_items:
                safe_remove_scene_item(scene, item)
        self._graph_repair_items.clear()

    @staticmethod
    def _mask_data_signature(mask):
        import zlib
        array = np.asarray(mask)
        if array.ndim == 3:
            array = array[..., 0]
        sy = max(1, array.shape[0] // 128)
        sx = max(1, array.shape[1] // 128)
        sample = np.ascontiguousarray(array[::sy, ::sx])
        return array.shape, int(np.count_nonzero(array)), int(zlib.crc32(sample.tobytes()))

    def _current_graph_signature(self):
        if self._graph_editor is None:
            return None
        nodes = tuple(sorted(repr(node.get("id")) for node in self._graph_editor.nodes))
        edges = tuple(sorted(
            (repr(edge.get("id")), repr(edge.get("start")), repr(edge.get("end")), bool(edge.get("enabled", True)))
            for edge in self._graph_editor.edges
        ))
        return nodes, edges

    def _render_graph_repair_candidates(self):
        self._clear_graph_repair_items()
        scene = self._canvas.scene()
        ge = self._graph_editor
        if scene is None or ge is None:
            return
        nodes_by_id = {node.get("id"): node for node in ge.nodes}
        edges_by_id = {edge.get("id"): edge for edge in ge.edges}
        from PySide6.QtGui import QPainterPath
        for candidate in self._graph_repair_candidates:
            if candidate.get("status") == "rejected":
                continue
            kind = candidate.get("type")
            points = []
            if kind in ("connect_endpoints", "merge_nodes"):
                left = nodes_by_id.get(candidate.get("node_a"))
                right = nodes_by_id.get(candidate.get("node_b"))
                if left is not None and right is not None:
                    points = [[left["x"], left["y"]], [right["x"], right["y"]]]
            elif kind == "delete_short_spur":
                edge = edges_by_id.get(candidate.get("edge_id"))
                if edge is not None:
                    points = edge.get("points_pixel", [])
            if len(points) < 2:
                continue
            path = QPainterPath()
            x, y = self._layer_manager.global_to_preview_f(points[0][0], points[0][1])
            path.moveTo(x, y)
            for point in points[1:]:
                x, y = self._layer_manager.global_to_preview_f(point[0], point[1])
                path.lineTo(x, y)
            item = QGraphicsPathItem(path)
            if candidate.get("status") == "applied":
                color = QColor(60, 220, 90)
            elif kind == "delete_short_spur":
                color = QColor(155, 155, 155)
            else:
                color = QColor(255, 215, 55)
            pen = QPen(color, 4, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setZValue(self._canvas.ZVAL_SELECTED + 2)
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            item.setToolTip(
                f"{candidate.get('id')} | {candidate.get('reason')} | "
                f"confidence={candidate.get('confidence', 0):.2f}"
            )
            scene.addItem(item)
            self._graph_repair_items.append(item)

    def _on_diagnose_graph(self, show_dialog=True):
        if self._graph_editor is None or not self._graph_editor.nodes:
            if show_dialog:
                QMessageBox.warning(self, "分析路网问题", "当前没有可分析的 final_graph。")
            return None
        import json
        from datetime import datetime
        from roadnet.graph_diagnostics import GraphDiagnosticsConfig, analyze_graph
        from roadnet.graph_auto_repair import generate_repair_candidates
        graph_config = self._param_panel.get_config().get("graph", {})
        endpoint_distance = float(graph_config.get("endpoint_connect_distance", 25))
        diagnostics = analyze_graph(
            self._graph_editor.nodes,
            self._graph_editor.edges,
            GraphDiagnosticsConfig(close_endpoint_distance=max(50.0, endpoint_distance)),
            task_points=self._snapped_points,
        )
        candidates = generate_repair_candidates(
            self._graph_editor.nodes,
            self._graph_editor.edges,
            diagnostics,
            road_mask=self._layer_manager.get_layer_data("mask"),
            endpoint_distance=max(50.0, endpoint_distance),
            junction_merge_distance=float(graph_config.get("node_merge_distance", 8)),
        )
        output_dir = os.path.join(
            os.getcwd(), "outputs", "auto_repair",
            "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        os.makedirs(output_dir, exist_ok=True)
        json_default = lambda value: value.item() if isinstance(value, np.generic) else value.tolist()
        with open(os.path.join(output_dir, "graph_diagnostics_report.json"), "w", encoding="utf-8") as stream:
            json.dump(diagnostics, stream, ensure_ascii=False, indent=2, default=json_default)
        with open(os.path.join(output_dir, "repair_candidates.json"), "w", encoding="utf-8") as stream:
            json.dump(candidates, stream, ensure_ascii=False, indent=2, default=json_default)
        self._last_graph_diagnostics = diagnostics
        self._graph_repair_candidates = candidates
        self._graph_repair_signature = self._current_graph_signature()
        self._last_auto_repair_output_dir = output_dir
        self._render_graph_repair_candidates()
        if show_dialog:
            components = diagnostics["connected_components"]
            message = (
                f"节点：{diagnostics['node_count']}，边：{diagnostics['edge_count']}\n"
                f"连通分量：{components}\n"
                f"degree=1 端点：{len(diagnostics['degree_1_endpoints'])}\n"
                f"短毛刺：{len(diagnostics['short_spurs'])}\n"
                f"修复建议：{len(candidates)}\n\n输出目录：\n{output_dir}"
            )
            if components > 1:
                message += "\n\n当前 final_graph 不连通，请优先检查黄色端点补边建议。"
            QMessageBox.warning(self, "路网诊断完成", message) if components > 1 else QMessageBox.information(self, "路网诊断完成", message)
        return diagnostics

    def _apply_graph_repairs(self, selected_ids=None, rerun_plan=False):
        if not self._graph_repair_candidates:
            if self._on_diagnose_graph(show_dialog=False) is None:
                return 0
        elif (self._graph_repair_signature is not None
              and self._graph_repair_signature != self._current_graph_signature()):
            QMessageBox.warning(
                self, "修复建议已失效",
                "final_graph 在诊断后已发生变化，软件将重新分析修复建议。",
            )
            if self._on_diagnose_graph(show_dialog=False) is None:
                return 0
        from roadnet.graph_auto_repair import apply_repair_candidates, save_auto_repair_bundle
        from roadnet.graph_diagnostics import analyze_graph
        before_nodes = copy.deepcopy(list(self._graph_editor.nodes))
        before_edges = copy.deepcopy(list(self._graph_editor.edges))
        diagnostics_before = analyze_graph(before_nodes, before_edges)
        self._history.push_state("auto_graph_repair")
        after_nodes, after_edges, apply_report = apply_repair_candidates(
            before_nodes, before_edges, self._graph_repair_candidates,
            confidence_threshold=0.80,
            selected_ids=set(selected_ids) if selected_ids is not None else None,
        )
        applied_count = len(apply_report["applied_candidate_ids"])
        if not applied_count:
            # Discard the no-op history snapshot.
            if self._history._undo_stack and self._history._undo_stack[-1].get("__action__") == "auto_graph_repair":
                self._history._undo_stack.pop()
            QMessageBox.information(self, "应用路网修复", "没有符合条件的待处理修复建议。")
            return 0
        ge = self._graph_editor
        ge._nodes = copy.deepcopy(after_nodes)
        ge._edges = copy.deepcopy(after_edges)
        ge._next_node_id = max((node.get("id") for node in ge.nodes if isinstance(node.get("id"), int)), default=-1) + 1
        ge._next_edge_id = max((edge.get("id") for edge in ge.edges if isinstance(edge.get("id"), int)), default=-1) + 1
        diagnostics_after = analyze_graph(after_nodes, after_edges)
        output_dir = self._last_auto_repair_output_dir
        if not output_dir:
            from datetime import datetime
            output_dir = os.path.join(os.getcwd(), "outputs", "auto_repair", "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        image = self._layer_manager.full_image_rgb
        if image is not None and image.ndim == 3 and image.shape[2] >= 3:
            image = cv2.cvtColor(np.asarray(image)[..., :3], cv2.COLOR_RGB2BGR)
        save_auto_repair_bundle(
            output_dir, before_nodes, before_edges, after_nodes, after_edges,
            diagnostics_before, diagnostics_after,
            self._graph_repair_candidates, apply_report, image=image,
        )
        self._last_graph_diagnostics = diagnostics_after
        self._graph_repair_signature = self._current_graph_signature()
        self.refresh_graph_layer()
        self._render_graph_repair_candidates()
        self._status_bar.show_message(
            f"自动修复完成：连通分量 {diagnostics_before['connected_components']} → "
            f"{diagnostics_after['connected_components']}，可按 Ctrl+Z 撤销"
        )
        if rerun_plan:
            QTimer.singleShot(0, lambda: self._on_run_global_plan("astar"))
        return applied_count

    def _on_apply_high_confidence_graph_repairs(self):
        count = self._apply_graph_repairs()
        if count:
            QMessageBox.information(
                self, "高置信修复已应用",
                f"已应用 {count} 项修复。\n"
                f"当前连通分量：{self._last_graph_diagnostics.get('connected_components', 0)}\n"
                "可按 Ctrl+Z 撤销，Ctrl+Y 重做。",
            )

    def _on_view_graph_repairs(self):
        if not self._graph_repair_candidates:
            if self._on_diagnose_graph(show_dialog=False) is None:
                return
        pending = [item for item in self._graph_repair_candidates if item.get("status") == "pending"]
        if not pending:
            QMessageBox.information(self, "修复建议", "当前没有待确认修复建议。")
            return
        labels = [f"{item['id']} | {item['type']} | {item['confidence']:.2f} | {item['reason']}" for item in pending]
        selected, ok = QInputDialog.getItem(self, "查看修复建议", "选择建议：", labels, 0, False)
        if not ok:
            return
        candidate = pending[labels.index(selected)]
        box = QMessageBox(self)
        box.setWindowTitle(candidate["id"])
        box.setText(
            f"类型：{candidate['type']}\n置信度：{candidate['confidence']:.2f}\n"
            f"原因：{candidate['reason']}\n距离：{candidate.get('distance_px', '--')} px"
        )
        apply_button = box.addButton("确认修复", QMessageBox.ButtonRole.AcceptRole)
        reject_button = box.addButton("拒绝建议", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is apply_button:
            self._apply_graph_repairs({candidate["id"]})
        elif box.clickedButton() is reject_button:
            candidate["status"] = "rejected"
            self._render_graph_repair_candidates()

    def _on_param_changed(self, key: str, value):
        """参数面板 SpinBox/CheckBox 值变化时同步到对应组件"""
        if key == "edit.brush_radius" and self._canvas is not None:
            self._canvas.brush_radius = int(value)
            self._status_bar.show_message(f"画笔半径 = {int(value)} px")
        elif key == "segment.preview_seg_alpha":
            # 实时更新 preview_segmentation 图层透明度
            alpha_int = max(1, min(255, int(float(value) * 255)))
            self._layer_manager.set_layer_opacity("layer_preview_segmentation", alpha_int)
            layer = self._layer_manager.layers().get("layer_preview_segmentation")
            if layer and layer.visible and layer.data is not None:
                self._canvas.update_overlay("layer_preview_segmentation")
                self._canvas.viewport().update()
        elif key in {
            "visualization.planned_path_width",
            "visualization.arrow_spacing_px",
            "visualization.arrow_size_px",
        } and len(self.planned_path_pixel) > 1:
            self._render_planned_path_to_scene()
        elif key in {
            "visualization.show_mask_candidate_numbers",
            "visualization.show_mask_candidate_reasons",
        }:
            self._render_mask_candidates()

    def _on_apply_params(self, category: str):
        # ★ 支持 :: 参数化命令
        if "::" in category:
            prefix, value = category.split("::", 1)
            if prefix == "cal_set_method":
                self._on_calibration_set_method(value)
                return

        actions = {
            "segment":         self._on_run_segment,
            "segment_preview": self._on_run_preview_segmentation,
            "competition_fast_roadnet": self._on_competition_fast_roadnet,
            "lowres_formal_mask": self._on_lowres_formal_mask,
            "emergency_hand_draw_graph": self._on_emergency_hand_draw_graph,
            "extract_roi_opencv": self._on_extract_roi_opencv,
            "extract_full_opencv": self._on_extract_full_opencv,
            "extract_roi":     self._on_extract_roi,
            "extract_full":    self._on_extract_full,
            "roi_draw":        self._begin_roi_drawing,
            "roi_use_view":    self._use_current_view_as_roi,
            "roi_clear":       self._clear_roi_regions,
            "roi_show_tiles":  self._show_roi_covered_tiles,
            "cancel_segment":  self._cancel_segmentation,
            "postprocess":     self._on_run_postprocess,
            "seed_draw":       lambda: self._begin_seed_drawing("freehand"),
            "seed_draw_two_point": self._begin_seed_drawing_two_point,
            "seed_draw_polyline": self._begin_seed_drawing_polyline,
            "seed_undo_last":  self._undo_last_seed_stroke,
            "seed_use_view":   self._seed_use_view,
            "seed_use_tasks":  self._seed_use_tasks,
            "seed_clear":      self._clear_main_road_seeds,
            "seed_rebuild_mask": self._on_seed_rebuild_mask,
            "seed_view_corridor": self._show_main_road_corridor,
            "seed_clean_mask": self._on_seed_clean_mask,
            "seed_clean_compare": self._on_seed_clean_compare,
            "seed_clean_accept": self._on_seed_clean_accept,
            "seed_clean_rollback": self._on_seed_clean_rollback,
            "ribbon_hole_gap_fill": self._on_ribbon_hole_gap_fill,
            "ribbon_hole_gap_view": self._on_ribbon_hole_gap_view,
            "ribbon_hole_gap_accept": self._on_ribbon_hole_gap_accept,
            "ribbon_hole_gap_rollback": self._on_ribbon_hole_gap_rollback,
            "refine_preview_bridges": self._on_preview_bridges,
            "refine_main_road": self._on_refine_main_road,
            "refine_accept":   self._on_refine_accept,
            "refine_rollback": self._on_refine_rollback,
            "restore_mask":    self._on_restore_pre_postprocess,
            "restore_original_roadmask": self._on_restore_original_roadmask,
            "skeleton":        self._on_run_skeleton,
            "skeleton_direct": self._on_go_skeleton_direct,
            "skel_show_raw":   self._on_large_skeleton_show_raw,
            "skel_show_cleaned": self._on_large_skeleton_show_cleaned,
            "skel_view_bridges": self._on_view_skeleton_bridges,
            "skel_view_input_mask": self._on_view_skeleton_input_mask,
            "skel_accept":     self._on_accept_skeleton_result,
            "skel_rollback":   self._on_rollback_skeleton_result,
            "optimize":        self._on_run_optimize,
            "graph":           self._on_run_graph,
            "apply_roi":       self._on_apply_roi,
            "apply_ignore":    self._on_apply_ignore,
            "analyze_mask_quality": self._on_analyze_mask_quality,
            "apply_mask_candidates": self._on_apply_high_confidence_ignore,
            "view_mask_candidates": self._on_view_mask_candidates,
            "graph_save":      self._on_save_graph,
            "diagnose_graph":  self._on_diagnose_graph,
            "apply_graph_repairs": self._on_apply_high_confidence_graph_repairs,
            "view_graph_repairs": self._on_view_graph_repairs,
            "clear_graph_edits": lambda: self._clear_graph_edits(),
            "export_graph":    lambda: self._on_save_graph(),
            "undo_graph":      self.undo_graph_edit,
            "redo_graph":      self.redo_graph_edit,
            "graph_polyline_repair": self._on_graph_polyline_repair,
            "graph_delete_edge_tool": self._on_graph_delete_edge_tool,
            "graph_merge_junctions": self._on_graph_merge_junctions,
            "graph_local_rebuild": self._on_graph_local_rebuild,
            "graph_locate_jump": self._on_graph_locate_jump,
            "save_mask":       self._on_save_current_mask,
            "save_skeleton":   self._on_export_skeleton,
            "view_compare":    self._on_view_compare,
            "clear_samples":   self._on_clear_samples,
            "import_task_points": self._on_import_task_points,
            "manual_set_start": lambda: self.set_tool("set_start"),
            "manual_set_goal": lambda: self.set_tool("set_end"),
            "manual_add_via": lambda: self.set_tool("add_task"),
            "clear_task_points": self._on_clear_task_points,
            "validate_task_points": self._on_validate_task_point_coordinates,
            "snap_task_points": self._on_snap_task_points,
            "plan":            lambda: self._on_run_global_plan("astar"),
            # Official vehicle-waypoint pipeline (唯一主线)
            "vwp_generate_dense_path": self._on_vwp_generate_dense_path,
            "vwp_generate_vehicle_csv": self._on_vwp_generate_vehicle_csv,
            "vwp_validate_vehicle_csv": self._on_vwp_validate_vehicle_csv,
            "vwp_repair_vehicle_csv": self._on_vwp_repair_vehicle_csv,
            "vwp_export_yaml": self._on_vwp_export_yaml,
            "vwp_run_full_pipeline": self._on_vwp_run_full_pipeline,
            # deprecated UI entries kept only for old project scripts / hotkeys
            "generate_vehicle_waypoints": self._on_vwp_generate_vehicle_csv,  # deprecated
            "validate_vehicle_waypoints": self._on_vwp_validate_vehicle_csv,  # deprecated
            "layered_path_diagnosis": self._on_layered_path_diagnosis_deprecated,
            "show_vehicle_waypoints": self._on_show_vehicle_waypoints,
            "show_waypoint_validation": self._on_show_waypoint_validation,
            "export":          self._on_vwp_export_yaml,  # deprecated alias → new YAML export
            "fix_sparse_cutting_corners": self._on_fix_sparse_cutting_corners_deprecated,
            "export_competition": self._on_export_competition_roadnet,
            "export_debug":    self._on_save_graph_debug,
            "open":            self._on_open_image,
            "new_project":     self._on_new_project,
            "save_project":    lambda: self._on_save_project(),
            # ★ 坐标校准
            "cal_import_txt":        self._on_calibration_import_txt,
            "cal_import_vertex_txt": self._on_calibration_import_vertex_txt,
            "cal_import_other":      self._on_calibration_import_other,
            "cal_import_corners":    self._on_calibration_import_corners_json,
            "cal_import_cp_file":    self._on_calibration_import_cp_file,
            "cal_start_map_click":   self._on_calibration_start_map_click,
            "cal_preset_1":          lambda: self._on_calibration_corner_preset(["top_left", "top_right", "bottom_left"]),
            "cal_preset_2":          lambda: self._on_calibration_corner_preset(["top_left", "top_right", "bottom_right"]),
            "cal_preset_3":          lambda: self._on_calibration_corner_preset(["top_left", "bottom_left", "bottom_right"]),
            "cal_preset_4":          lambda: self._on_calibration_corner_preset(["top_right", "bottom_left", "bottom_right"]),
            "cal_preset_all":        lambda: self._on_calibration_corner_preset(["top_left", "top_right", "bottom_left", "bottom_right"]),
            "cal_compute":           self._on_calibration_compute,
            "cal_apply_graph":       self._on_calibration_apply_graph,
            "cal_save":              self._on_calibration_save,
            "cal_clear":             self._on_calibration_clear,
            "cal_update_settings":   self._on_calibration_update_settings,
        }
        action = actions.get(category)
        if action:
            action()

    def _on_go_skeleton_direct(self):
        """直接使用当前 mask 进入骨架生成阶段（不强制后处理）"""
        # ★ 使用 get_current_mask_array 统一解析，兼容 dict/ndarray
        mask, mask_meta = self.get_current_mask_array(prefer_processed=False,
                                                       for_skeleton=False)
        if mask is None:
            err_msg = mask_meta.get("error", "未找到有效的 Road Mask 数据。")
            QMessageBox.warning(self, "提示",
                f"无法进入骨架阶段：{err_msg}\n\n"
                "请确保已加载正式 Road Mask（ndarray 或 global_road_mask.png）。")
            return
        self.mark_stage_done("edit")
        self.set_stage("skeleton")
        self._status_bar.show_message("已跳过可选后处理，进入骨架生成阶段")

    def _clear_graph_edits(self):
        """清空人工修改，恢复到 draft 状态"""
        if self._graph_editor is None:
            return
        self._history.push_state("clear_graph_edits")
        skeleton = self._layer_manager.get_layer_data("skeleton")
        if skeleton is not None:
            self._on_run_graph()
            self._status_bar.show_message("已恢复到草稿路网")
        else:
            QMessageBox.warning(self, "提示", "未找到 skeleton 数据")

    def _on_global_undo(self):
        """全局撤销（Ctrl+Z）"""
        action = self._history.undo()
        if action:
            print(f"[DEBUG][History] undo action={action}")
            # history.undo() 内部已调用 restore_state → _refresh_all
            # 这里做补充刷新确保 graph 和 task_points 渲染正确
            self._render_graph_to_scene()
            self._render_task_points_to_scene()
            self._update_graph_stats()
            self._sync_skeleton_state_from_layer()
            self._status_bar.show_message(f"已撤销: {action}")
        else:
            self._status_bar.show_message("没有可撤销的操作")
        # 更新菜单禁用状态
        if hasattr(self, '_act_undo'):
            self._act_undo.setEnabled(self._history.can_undo)
        if hasattr(self, '_act_redo'):
            self._act_redo.setEnabled(self._history.can_redo)

    def _on_global_redo(self):
        """全局重做（Ctrl+Y / Ctrl+Shift+Z）"""
        action = self._history.redo()
        if action:
            print(f"[DEBUG][History] redo action={action}")
            self._render_graph_to_scene()
            self._render_task_points_to_scene()
            self._update_graph_stats()
            self._sync_skeleton_state_from_layer()
            self._status_bar.show_message(f"已重做: {action}")
        else:
            self._status_bar.show_message("没有可重做的操作")
        # 更新菜单禁用状态
        if hasattr(self, '_act_undo'):
            self._act_undo.setEnabled(self._history.can_undo)
        if hasattr(self, '_act_redo'):
            self._act_redo.setEnabled(self._history.can_redo)

    def undo_graph_edit(self):
        """撤销按钮（供右侧面板按钮调用）— 路由到全局撤销"""
        self._on_global_undo()

    def redo_graph_edit(self):
        """重做按钮（供右侧面板按钮调用）— 路由到全局重做"""
        self._on_global_redo()

    def mark_stage_done(self, stage: str):
        self._stage_completed.add(stage)
        if stage in self._nav_buttons:
            btn = self._nav_buttons[stage]
            btn.setProperty("stageStatus", "done")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _ensure_validated_vehicle_waypoints_for_export(self):
        """Ensure in-memory vehicle waypoints + validation report for submission/YAML.

        Prefer official vehicle_waypoint_pipeline outputs when available.
        Returns (waypoints, report, bad_segments, error_message).
        error_message is None when preparation succeeded (gate may still fail).
        """
        if (
            self._vwp_result is not None
            and self._vwp_result.vehicle_waypoints_repaired
            and (self._vwp_result.validation_report or {}).get("export_ready")
        ):
            return (
                list(self._vwp_result.vehicle_waypoints_repaired),
                dict(self._vwp_result.validation_report or {}),
                [],
                None,
            )

        from roadnet.path_export import vehicle_export_gate_ok

        result = self.planning_result
        img_w = int(getattr(self._geo_calibration, "image_width", 0) or 0) if self._geo_calibration else 0
        img_h = int(getattr(self._geo_calibration, "image_height", 0) or 0) if self._geo_calibration else 0
        if img_w <= 0 and hasattr(self._layer_manager, "full_image_rgb"):
            arr = self._layer_manager.full_image_rgb
            if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                img_h, img_w = arr.shape[:2]

        if not self.sparse_waypoints:
            if (result is None
                    or not bool(getattr(result, "success", False))
                    or len(self.planned_path_pixel) < 2):
                return [], {}, [], (
                    "当前没有车辆航点，且无法从规划路径生成。"
                    "请先规划路径并生成/验证车辆航点。"
                )
            if self._geo_calibration is None:
                return [], {}, [], "请先完成坐标校准后再导出正式小车航点。"
            try:
                from roadnet.adaptive_waypoint_resampler import (
                    generate_vehicle_waypoints_adaptive,
                )
                adaptive = generate_vehicle_waypoints_adaptive(
                    self.planned_path_pixel,
                    self.planned_path_geo,
                    graph=self._graph_editor,
                    task_points=self.snapped_task_points or self._task_points,
                    road_mask=self._layer_manager.get_layer_data("mask"),
                    config=self._build_adaptive_waypoint_config(),
                    geo_calibration=self._geo_calibration,
                    path_node_sequence=self._path_node_sequence_from_result(result),
                    path_edge_sequence=self.planned_path_edges,
                    image_width=img_w,
                    image_height=img_h,
                )
                self.sparse_waypoints = list(adaptive.waypoints)
                self.sparse_waypoints_pixel = list(adaptive.sparse_waypoints_pixel)
                self.sparse_waypoints_geo = list(adaptive.sparse_waypoints_geo)
                self._dense_path_for_waypoints = list(
                    adaptive.dense_path_pixel or self.planned_path_pixel
                )
            except Exception as exc:
                return [], {}, [], f"自动生成车辆航点失败：{exc}"

        report = dict(self._waypoint_validation_report or {})
        bad = list(getattr(self, "_waypoint_bad_segments", None) or [])
        gate_ok, _ = vehicle_export_gate_ok(report)
        if not gate_ok or not report:
            try:
                from roadnet.waypoint_validator import validate_vehicle_waypoints
                validation = validate_vehicle_waypoints(
                    self.sparse_waypoints,
                    dense_path_pixel=self._dense_path_for_waypoints or self.planned_path_pixel,
                    geo_calibration=self._geo_calibration,
                    final_graph=self._graph_editor,
                    task_points=self.snapped_task_points or self._task_points,
                    road_mask=self._layer_manager.get_layer_data("mask"),
                    path_node_sequence=(
                        self._path_node_sequence_from_result(result) if result else None
                    ),
                    config=self._build_validation_config(),
                    adaptive_config=self._build_adaptive_waypoint_config(),
                    image_width=img_w,
                    image_height=img_h,
                )
                self.sparse_waypoints = list(validation.waypoints)
                self.sparse_waypoints_pixel = [
                    [float(wp["x_pixel"]), float(wp["y_pixel"])]
                    for wp in self.sparse_waypoints
                ]
                self.sparse_waypoints_geo = []
                for wp in self.sparse_waypoints:
                    lon = wp.get("longitude_deg", wp.get("longitude"))
                    lat = wp.get("latitude_deg", wp.get("latitude"))
                    alt = wp.get("altitude_m", wp.get("altitude", 0.0))
                    if lon is not None and lat is not None:
                        self.sparse_waypoints_geo.append(
                            [float(lon), float(lat), float(alt or 0.0)]
                        )
                report = dict(validation.report or {})
                bad = list(validation.bad_segments or [])
                self._waypoint_validation_report = report
                self._waypoint_bad_segments = bad
            except Exception as exc:
                return list(self.sparse_waypoints), report, bad, f"车辆航点验收失败：{exc}"

        return list(self.sparse_waypoints), report, bad, None

    # ------------------------------------------------------------------
    # Official vehicle-waypoint pipeline (唯一主线)
    # ------------------------------------------------------------------

    def _vwp_pipeline_config(self):
        from roadnet.vehicle_waypoint_pipeline import PipelineConfig
        alt = 21.741
        try:
            wp_cfg = self._param_panel.get_config().get("waypoints", {})
            alt = float(wp_cfg.get("default_altitude_m", alt))
        except (TypeError, ValueError):
            pass
        return PipelineConfig(default_altitude_m=alt)

    def _vwp_ensure_output_dir(self) -> Optional[str]:
        from roadnet.vehicle_waypoint_pipeline import default_pipeline_output_dir
        if self._vwp_output_dir and os.path.isdir(self._vwp_output_dir):
            return self._vwp_output_dir
        suggested = default_pipeline_output_dir(os.getcwd())
        selected = QFileDialog.getExistingDirectory(
            self, "小车航点主流程 - 选择输出文件夹", suggested,
            QFileDialog.Option.ShowDirsOnly,
        )
        if not selected:
            return None
        self._vwp_output_dir = os.path.abspath(selected)
        os.makedirs(self._vwp_output_dir, exist_ok=True)
        return self._vwp_output_dir

    def _vwp_refresh_status(self, status=None, error: str = "", suggestion: str = ""):
        if status is not None and hasattr(self._param_panel, "update_vwp_status"):
            self._param_panel.update_vwp_status(status)
        if error:
            tip = error
            if suggestion:
                tip = f"{error}\n\n建议：{suggestion}"
            QMessageBox.warning(self, "小车航点主流程", tip)

    def _vwp_set_waypoint_display(
        self,
        waypoints: list,
        *,
        layer_name: str,
        csv_path: Optional[str],
    ):
        """Bind canvas sparse layer to the exact exported CSV content."""
        self._vwp_waypoint_layer_name = layer_name
        self._vwp_waypoint_csv_path = csv_path
        self.sparse_waypoints = list(waypoints or [])
        self.sparse_waypoints_pixel = [
            [float(wp["x_pixel"]), float(wp["y_pixel"])] for wp in self.sparse_waypoints
        ]
        self.sparse_waypoints_geo = []
        for wp in self.sparse_waypoints:
            lon = wp.get("longitude_deg")
            lat = wp.get("latitude_deg")
            alt = wp.get("altitude_m", 0.0)
            if lon is not None and lat is not None:
                self.sparse_waypoints_geo.append(
                    [float(lon), float(lat), float(alt or 0.0)]
                )
        # Update checkbox label to match active CSV kind
        try:
            cb = getattr(self._tool_panel, "_layer_checkboxes", {}).get(
                "layer_sparse_waypoints"
            )
            if cb is not None:
                cb.setText(layer_name)
        except Exception:
            pass

    def _vwp_status_bar_layer(self, layer_name: str, count: int, path: Optional[str]):
        path_txt = path or "(未绑定文件)"
        msg = f"{layer_name} | {count} 点 | {path_txt}"
        try:
            self._status_bar.show_message(msg)
        except Exception:
            pass

    def _vwp_apply_result_to_ui(self, result):
        """Sync pipeline outputs into canvas memory for display."""
        self._vwp_result = result
        self._vwp_output_dir = result.output_dir
        if result.dense_path:
            self._dense_path_for_waypoints = [
                [float(r["x_pixel"]), float(r["y_pixel"])] for r in result.dense_path
            ]
            self.planned_path_pixel = list(self._dense_path_for_waypoints)
            if result.output_dir:
                self._vwp_dense_path_csv_path = os.path.join(
                    result.output_dir, "dense_path.csv",
                )
        repaired = list(result.vehicle_waypoints_repaired or [])
        wps = repaired or list(result.vehicle_waypoints or [])
        if wps:
            if repaired:
                layer = "repaired_vehicle_waypoints"
                csv_name = "vehicle_waypoints_repaired.csv"
            else:
                layer = "vehicle_waypoints"
                csv_name = "vehicle_waypoints.csv"
            csv_path = (
                os.path.join(result.output_dir, csv_name) if result.output_dir else None
            )
            self._vwp_set_waypoint_display(wps, layer_name=layer, csv_path=csv_path)
            self._waypoint_validation_report = dict(result.validation_report or {})
            try:
                self._layer_manager.show_layer("layer_sparse_waypoints")
                self._tool_panel.set_layer_checkbox_state("layer_sparse_waypoints", True)
                self._render_sparse_waypoints_to_scene()
                self._vwp_status_bar_layer(layer, len(wps), csv_path)
            except Exception:
                pass
        self._vwp_refresh_status(result.status)

    def _on_vwp_generate_dense_path(self):
        """① snap → plan → expand → dense_path.csv"""
        self._normalize_task_points("before_vwp_dense")
        if self._graph_editor is None or not self._graph_editor.nodes:
            QMessageBox.warning(self, "生成 dense_path", "final_graph 不存在，请先生成路网。")
            return
        if len(self._task_points or []) < 2:
            QMessageBox.warning(
                self, "生成 dense_path",
                "未生成 dense_path：请先输入任务点并规划路径",
            )
            return
        if self._geo_calibration is None:
            QMessageBox.warning(self, "生成 dense_path", "请先完成坐标校准。")
            return
        output_dir = self._vwp_ensure_output_dir()
        if not output_dir:
            return

        from roadnet.vehicle_waypoint_pipeline import (
            PipelineResult, PipelineStatus,
            classify_dense_path_zones, expand_route_edges_to_dense_path,
            export_dense_path_csv, export_dense_path_labeled_csv,
            plan_route_by_task_sequence, snap_task_points_to_graph, _mpp,
        )

        cfg = self._vwp_pipeline_config()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            snapped = snap_task_points_to_graph(
                self._task_points, self._graph_editor, self._geo_calibration,
                output_dir=output_dir,
            )
            if any(r.get("status") == "failed" for r in snapped):
                self._vwp_refresh_status(
                    error="任务点吸附失败",
                    suggestion="检查任务点是否落在路网附近",
                )
                return
            self.snapped_task_points = [
                r["_snapped_obj"] for r in snapped if r.get("_snapped_obj") is not None
            ]
            segments, ok = plan_route_by_task_sequence(
                snapped, self._graph_editor, output_dir=output_dir,
                metres_per_pixel=_mpp(self._geo_calibration),
            )
            if not ok:
                self._vwp_refresh_status(
                    error="dense_path 为空：任务点之间路径规划失败",
                    suggestion="请检查任务点连通性",
                )
                return
            dense = expand_route_edges_to_dense_path(
                segments, self._graph_editor, self._geo_calibration,
                snapped_rows=snapped,
                default_altitude_m=cfg.default_altitude_m,
            )
            if len(dense) < 2:
                self._vwp_refresh_status(
                    error="dense_path 为空：任务点之间路径规划失败",
                )
                return
            from roadnet.vehicle_waypoint_pipeline import densify_dense_path_rows
            dense = densify_dense_path_rows(dense, step_m=1.0)
            export_dense_path_csv(dense, os.path.join(output_dir, "dense_path.csv"))
            labeled = classify_dense_path_zones(
                dense, self._graph_editor, snapped, cfg,
                metres_per_pixel=_mpp(self._geo_calibration),
            )
            export_dense_path_labeled_csv(
                labeled, os.path.join(output_dir, "dense_path_labeled.csv"),
            )
            # Keep planned path display in sync — this is dense_path, NOT vehicle WPs
            self.planned_path_pixel = [
                [float(r["x_pixel"]), float(r["y_pixel"])] for r in dense
            ]
            self._dense_path_for_waypoints = list(self.planned_path_pixel)
            self.planned_path_edges = []
            for seg in segments:
                self.planned_path_edges.extend(seg.get("edge_ids") or [])
            # Clear vehicle waypoint layer so dense_path is not mistaken for WPs
            self.sparse_waypoints = []
            self.sparse_waypoints_pixel = []
            self.sparse_waypoints_geo = []
            self._vwp_waypoint_csv_path = None
            self._vwp_waypoint_layer_name = "vehicle_waypoints"
            self._clear_sparse_waypoint_items()
            dense_csv = os.path.join(output_dir, "dense_path.csv")
            self._vwp_dense_path_csv_path = dense_csv

            status = PipelineStatus(
                graph_valid=True,
                task_points_loaded=True,
                dense_path_generated=True,
                message=f"dense_path 已生成：{len(dense)} 点",
            )
            result = PipelineResult(
                output_dir=output_dir, status=status,
                snapped_points=snapped, route_segments=segments,
                dense_path=dense, dense_path_labeled=labeled,
            )
            self._vwp_result = result
            self._vwp_output_dir = output_dir
            try:
                self._layer_manager.show_layer("planned_path")
                self._tool_panel.set_layer_checkbox_state("layer_planned_path", True)
                self._render_planned_path_to_scene()
                self._vwp_status_bar_layer("dense_path", len(dense), dense_csv)
            except Exception:
                pass
            self._vwp_refresh_status(result.status)

            # Mode counts for user feedback
            mode_counts = {"straight": 0, "curve": 0, "junction": 0, "task": 0}
            for r in labeled:
                m = str(r.get("spacing_mode") or "straight")
                if m not in mode_counts:
                    m = "straight"
                mode_counts[m] += 1
            QMessageBox.information(
                self, "生成 dense_path",
                f"已生成 dense_path.csv（{len(dense)} 点）\n"
                f"目录：\n{output_dir}\n\n"
                f"dense_path_labeled spacing_mode 统计：\n"
                f"  straight={mode_counts['straight']}\n"
                f"  curve={mode_counts['curve']}\n"
                f"  junction={mode_counts['junction']}\n"
                f"  task={mode_counts['task']}\n\n"
                "注意：当前图上 dense_path 图层显示的是 dense_path，\n"
                "不是最终小车航点。请继续「② 生成小车航点 CSV」。",
            )
        except Exception as exc:
            QMessageBox.critical(self, "生成 dense_path 失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _on_vwp_generate_vehicle_csv(self):
        """② sample vehicle_waypoints.csv from dense_path_labeled."""
        from roadnet.vehicle_waypoint_pipeline import (
            sample_vehicle_waypoints_from_dense_path,
            cleanup_anchor_aware_duplicates,
            export_vehicle_waypoints_csv,
            build_vehicle_waypoint_summary,
            export_vehicle_waypoint_summary,
            validate_vehicle_waypoints_csv,
        )
        result = self._vwp_result
        if result is None or not result.dense_path_labeled:
            QMessageBox.warning(
                self, "生成小车航点 CSV",
                "未生成 dense_path：请先输入任务点并规划路径",
            )
            return
        output_dir = result.output_dir or self._vwp_ensure_output_dir()
        if not output_dir:
            return
        cfg = self._vwp_pipeline_config()
        try:
            wps = sample_vehicle_waypoints_from_dense_path(
                result.dense_path_labeled, result.snapped_points, cfg,
            )
            wps, dup_warnings = cleanup_anchor_aware_duplicates(wps, cfg)
            export_vehicle_waypoints_csv(
                wps, os.path.join(output_dir, "vehicle_waypoints.csv"),
            )
            summary = build_vehicle_waypoint_summary(wps, result.dense_path_labeled)
            if dup_warnings:
                summary["anchor_duplicate_warnings"] = list(dup_warnings)
            export_vehicle_waypoint_summary(
                summary, os.path.join(output_dir, "vehicle_waypoint_summary.json"),
            )
            report, _ = validate_vehicle_waypoints_csv(
                wps,
                dense_path_labeled=result.dense_path_labeled,
                snapped_rows=result.snapped_points,
                config=cfg,
                output_dir=output_dir,
            )
            if dup_warnings:
                report.setdefault("warnings", [])
                report["warnings"] = list(report.get("warnings") or []) + list(dup_warnings)
            result.validation_report = report
            result.status.waypoints_checked = True
            result.status.export_ready = bool(report.get("export_ready"))
            result.vehicle_waypoints = wps
            result.vehicle_waypoints_repaired = []
            result.status.vehicle_waypoints_generated = True
            tip = ""
            if summary.get("straight_waypoints_too_dense"):
                tip = "\n\n⚠ straight_waypoints_too_dense=true（直线平均间距 < 10m）"
            if dup_warnings:
                tip += f"\n\nanchor 近距 warning：{len(dup_warnings)} 条（keep-keep 已保留）"
            result.status.message = (
                f"vehicle_waypoints.csv：{len(wps)} 点；"
                f"dup={report.get('duplicate_consecutive_count')}；"
                f"straight_avg="
                f"{summary.get('straight_average_spacing_m')}m"
                f"{tip}"
            )
            csv_path = os.path.join(output_dir, "vehicle_waypoints.csv")
            self._vwp_set_waypoint_display(
                wps, layer_name="vehicle_waypoints", csv_path=csv_path,
            )
            self._vwp_result = result
            try:
                self._layer_manager.show_layer("layer_sparse_waypoints")
                self._tool_panel.set_layer_checkbox_state("layer_sparse_waypoints", True)
                self._render_sparse_waypoints_to_scene()
                self._vwp_status_bar_layer("vehicle_waypoints", len(wps), csv_path)
            except Exception:
                pass
            self._vwp_refresh_status(result.status)
            QMessageBox.information(
                self, "生成小车航点 CSV",
                f"已生成 vehicle_waypoints.csv（{len(wps)} 点）\n\n"
                f"straight={summary.get('straight_waypoint_count')}  "
                f"curve={summary.get('curve_waypoint_count')}  "
                f"junction={summary.get('junction_waypoint_count')}  "
                f"task={summary.get('task_waypoint_count')}\n"
                f"average_spacing_m={summary.get('average_spacing_m')}\n"
                f"max_spacing_m={summary.get('max_spacing_m')}\n"
                f"straight_max_spacing_m="
                f"{summary.get('straight_max_spacing_m')}\n"
                f"curve_max_spacing_m={summary.get('curve_max_spacing_m')}\n"
                f"junction_max_spacing_m={summary.get('junction_max_spacing_m')}\n"
                f"task_max_spacing_m={summary.get('task_max_spacing_m')}\n"
                f"duplicate_consecutive_count="
                f"{report.get('duplicate_consecutive_count')}\n"
                f"aba_backtrack_count={report.get('aba_backtrack_count')}"
                f"{tip}\n\n"
                f"图层 vehicle_waypoints 点数 = CSV 行数 = {len(wps)}\n"
                f"{csv_path}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "生成小车航点 CSV 失败", str(exc))

    def _on_vwp_validate_vehicle_csv(self):
        """③ validate vehicle_waypoints.csv."""
        from roadnet.vehicle_waypoint_pipeline import validate_vehicle_waypoints_csv
        result = self._vwp_result
        if result is None or not result.vehicle_waypoints:
            QMessageBox.warning(
                self, "检查航点 CSV",
                "请先生成小车航点 CSV",
            )
            return
        cfg = self._vwp_pipeline_config()
        report, _ = validate_vehicle_waypoints_csv(
            result.vehicle_waypoints_repaired or result.vehicle_waypoints,
            dense_path_labeled=result.dense_path_labeled,
            snapped_rows=result.snapped_points,
            config=cfg,
            output_dir=result.output_dir,
        )
        result.validation_report = report
        result.status.waypoints_checked = True
        result.status.export_ready = bool(report.get("export_ready"))
        result.status.message = (
            "检查通过" if report.get("export_ready") else "航点点距不符合：请点击自动修复航点 CSV"
        )
        self._vwp_refresh_status(result.status)
        if report.get("export_ready"):
            QMessageBox.information(
                self, "检查航点 CSV",
                f"通过\nmax_spacing_m={report.get('max_spacing_m')}\n"
                f"count={report.get('waypoint_count')}",
            )
        else:
            errs = "\n".join(f"- {e}" for e in (report.get("errors") or [])[:8])
            QMessageBox.warning(
                self, "检查航点 CSV",
                f"未通过。请点击「自动修复航点 CSV」。\n\n{errs}",
            )

    def _on_vwp_repair_vehicle_csv(self):
        """④ repair using dense_path → vehicle_waypoints_repaired.csv."""
        from roadnet.vehicle_waypoint_pipeline import (
            repair_vehicle_waypoints_using_dense_path,
            validate_vehicle_waypoints_csv,
        )
        result = self._vwp_result
        if result is None or not result.vehicle_waypoints or not result.dense_path_labeled:
            QMessageBox.warning(
                self, "自动修复航点 CSV",
                "请先生成 dense_path 与小车航点 CSV",
            )
            return
        cfg = self._vwp_pipeline_config()
        try:
            repaired, repair_report = repair_vehicle_waypoints_using_dense_path(
                result.vehicle_waypoints,
                result.dense_path_labeled,
                result.validation_report,
                cfg,
                output_dir=result.output_dir,
            )
            report2, _ = validate_vehicle_waypoints_csv(
                repaired,
                dense_path_labeled=result.dense_path_labeled,
                snapped_rows=result.snapped_points,
                config=cfg,
                output_dir=result.output_dir,
            )
            result.vehicle_waypoints_repaired = repaired
            result.repair_report = repair_report
            result.validation_report = report2
            result.status.waypoints_repaired = True
            result.status.waypoints_checked = True
            result.status.export_ready = bool(report2.get("export_ready"))
            self._vwp_apply_result_to_ui(result)
            if report2.get("export_ready"):
                QMessageBox.information(
                    self, "自动修复航点 CSV",
                    f"修复完成并通过检查（{len(repaired)} 点）",
                )
            else:
                fail_path = os.path.join(
                    result.output_dir or "", "waypoint_repair_failed_report.json",
                )
                reasons = "\n".join(
                    f"- {r}" for r in (repair_report.get("failure_reasons") or [])[:8]
                )
                QMessageBox.warning(
                    self, "自动修复航点 CSV",
                    "自动修复后仍未通过验证。\n\n"
                    f"{reasons or '请查看 bad_waypoint_segments_after_repair.csv'}\n\n"
                    f"失败报告：\n{fail_path}",
                )
        except Exception as exc:
            QMessageBox.critical(self, "自动修复失败", str(exc))

    def _on_vwp_export_yaml(self):
        """⑤ export subject1_waypoints.yaml from repaired CSV only."""
        from roadnet.vehicle_waypoint_pipeline import (
            export_subject1_yaml_from_vehicle_csv,
            evaluate_usable_for_vehicle,
        )
        result = self._vwp_result
        if result is None:
            QMessageBox.warning(self, "导出小车 YAML", "请先运行主流程。")
            return
        repaired = result.vehicle_waypoints_repaired
        report = result.validation_report or {}
        if not repaired:
            QMessageBox.warning(
                self, "导出小车 YAML",
                "YAML 未生成：航点 CSV 检查未通过\n请先自动修复并检查通过。",
            )
            return
        if not report.get("export_ready"):
            QMessageBox.warning(
                self, "导出小车 YAML",
                "YAML 未生成：航点 CSV 检查未通过",
            )
            return
        path = os.path.join(result.output_dir, "subject1_waypoints.yaml")
        try:
            export_subject1_yaml_from_vehicle_csv(
                repaired, path,
                default_altitude_m=self._vwp_pipeline_config().default_altitude_m,
            )
            result.status.yaml_exported = True
            result.status.usable_for_vehicle = evaluate_usable_for_vehicle(
                result.status, report, path,
            )
            result.status.message = (
                "可用于小车" if result.status.usable_for_vehicle else "YAML 已导出"
            )
            self._vwp_refresh_status(result.status)
            QMessageBox.information(
                self, "导出小车 YAML",
                f"已生成：\n{path}\n\n"
                f"{'★ 可用于小车' if result.status.usable_for_vehicle else ''}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "导出小车 YAML 失败", str(exc))

    def _on_vwp_run_full_pipeline(self):
        """一键执行完整主流程。"""
        self._normalize_task_points("before_vwp_full")
        if self._graph_editor is None or not self._graph_editor.nodes:
            QMessageBox.warning(self, "一键主流程", "final_graph 不存在。")
            return
        if len(self._task_points or []) < 2:
            QMessageBox.warning(
                self, "一键主流程",
                "未生成 dense_path：请先输入任务点并规划路径",
            )
            return
        if self._geo_calibration is None:
            QMessageBox.warning(self, "一键主流程", "请先完成坐标校准。")
            return

        from roadnet.vehicle_waypoint_pipeline import (
            default_pipeline_output_dir,
            run_vehicle_waypoint_pipeline,
        )
        suggested = default_pipeline_output_dir(os.getcwd())
        selected = QFileDialog.getExistingDirectory(
            self, "一键执行小车航点主流程 - 选择输出文件夹", suggested,
            QFileDialog.Option.ShowDirsOnly,
        )
        if not selected:
            return
        output_dir = os.path.abspath(selected)
        os.makedirs(output_dir, exist_ok=True)

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = run_vehicle_waypoint_pipeline(
                self._graph_editor,
                self._task_points,
                self._geo_calibration,
                output_dir,
                config=self._vwp_pipeline_config(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "一键主流程失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        if result.snapped_points:
            self.snapped_task_points = [
                r["_snapped_obj"] for r in result.snapped_points
                if r.get("_snapped_obj") is not None
            ]
        self._vwp_apply_result_to_ui(result)
        if result.error:
            self._vwp_refresh_status(
                result.status, error=result.error,
                suggestion=result.suggestion or "",
            )
            return
        msg = (
            f"完成。\n输出目录：\n{output_dir}\n\n"
            f"dense={len(result.dense_path)}  "
            f"wp={len(result.vehicle_waypoints_repaired or result.vehicle_waypoints)}\n"
            f"{result.status.message}"
        )
        QMessageBox.information(self, "一键主流程", msg)

    def _on_layered_path_diagnosis_deprecated(self):
        """deprecated: 分层路径诊断已停用，请使用小车航点主流程。"""
        QMessageBox.information(
            self, "功能已停用",
            "分层路径诊断已停用。\n请使用「小车航点主流程」按钮。",
        )

    def _on_fix_sparse_cutting_corners_deprecated(self):
        """deprecated: 请使用「自动修复航点 CSV」。"""
        QMessageBox.information(
            self, "功能已停用",
            "请改用「④ 自动修复航点 CSV」。",
        )

    def _on_export_competition_roadnet(self):
        """导出适合比赛提交的 clean/debug 路网影像和配套数据。"""
        self._normalize_task_points("before_submission_export")
        if self._graph_editor is None or (
                not self._graph_editor.nodes and not self._graph_editor.edges):
            QMessageBox.warning(
                self, "导出比赛路网图",
                "当前没有 final_graph，请先生成或编辑路网。",
            )
            return
        full_image = self._layer_manager.full_image_rgb
        if full_image is None:
            QMessageBox.warning(self, "导出比赛路网图", "当前没有可导出的原始影像。")
            return
        if len(self.planned_path_pixel) < 2:
            QMessageBox.information(
                self, "导出比赛路网图",
                "当前未检测到规划路径，仅导出 final_graph 路网叠加图。",
            )

        from roadnet.submission_export import (
            default_submission_dir, export_competition_submission,
        )
        output_dir = default_submission_dir(os.getcwd())
        base_output_dir = output_dir
        suffix = 2
        while os.path.exists(output_dir):
            output_dir = f"{base_output_dir}_{suffix}"
            suffix += 1
        try:
            os.makedirs(output_dir, exist_ok=False)
        except Exception as exc:
            QMessageBox.critical(
                self, "导出比赛路网图 - 无法创建目录",
                f"文件路径：\n{output_dir}\n\n失败原因：\n{exc}",
            )
            return

        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "导出比赛路网图 - 选择输出文件夹",
            output_dir,
            QFileDialog.Option.ShowDirsOnly,
        )
        if not selected_dir:
            try:
                os.rmdir(output_dir)
            except OSError:
                pass
            return
        selected_dir = os.path.abspath(selected_dir)
        if os.path.normcase(selected_dir) != os.path.normcase(os.path.abspath(output_dir)):
            try:
                os.rmdir(output_dir)
            except OSError:
                pass

        config = self._param_panel.get_config().get("visualization", {})
        skeleton = None
        for candidate in (
                self.optimized_skeleton,
                self.current_skeleton,
                self.raw_skeleton):
            if candidate is not None:
                skeleton = candidate
                break
        if skeleton is None:
            skeleton = self._layer_manager.get_layer_data("skeleton")
        if skeleton is None and self._canvas.skeleton_result is not None:
            skeleton = self._canvas.skeleton_result
        project_name = self._project_manager.data.project_name
        if not project_name:
            project_name = os.path.splitext(os.path.basename(
                self._layer_manager.image_path or "RoadNet"
            ))[0]

        prep_err = None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            vehicle_wps, val_report, bad_segs, prep_err = (
                self._ensure_validated_vehicle_waypoints_for_export()
            )
            default_alt = 21.741
            try:
                wp_cfg = self._param_panel.get_config().get("waypoints", {})
                default_alt = float(wp_cfg.get("default_altitude_m", default_alt))
            except (TypeError, ValueError):
                pass

            export_result = export_competition_submission(
                selected_dir,
                full_image,
                [dict(node) for node in self._graph_editor.nodes],
                [dict(edge) for edge in self._graph_editor.edges],
                planned_path_pixel=self.planned_path_pixel,
                sparse_waypoints=vehicle_wps or self.sparse_waypoints_pixel or [],
                vehicle_waypoints=vehicle_wps,
                waypoint_validation_report=val_report,
                waypoint_bad_segments=bad_segs,
                task_points=self._task_points,
                snapped_points=self.snapped_task_points,
                road_mask=self._layer_manager.get_layer_data("mask"),
                skeleton=skeleton,
                geo_calibration=self._geo_calibration,
                image_path=self._layer_manager.image_path,
                project_name=project_name,
                arrow_spacing_px=float(config.get("arrow_spacing_px", 80.0)),
                arrow_size_px=float(config.get("arrow_size_px", 12.0)),
                default_altitude_m=default_alt,
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "比赛路网图导出失败",
                f"输出目录：\n{selected_dir}\n\n失败原因：\n{exc}",
            )
            import traceback
            traceback.print_exc()
            return
        finally:
            QApplication.restoreOverrideCursor()

        try:
            display_dir = os.path.relpath(selected_dir, os.getcwd())
        except ValueError:
            display_dir = selected_dir
        message_box = QMessageBox(self)
        message_box.setWindowTitle("导出比赛路网图")
        report = export_result.get("report", {}) or {}
        track_n = int(report.get("track_waypoint_count", 0))
        crs = report.get("coordinate_system", "image_pixel")
        yaml_ok = bool(
            export_result.get("yaml_export_valid") or report.get("yaml_export_valid")
        )
        official = (
            export_result.get("official_vehicle_yaml")
            or report.get("official_vehicle_yaml")
        )
        lines = [
            "比赛路网图导出完成。",
            "",
            f"输出目录：\n{display_dir}",
            "",
            f"坐标系：{crs}",
            f"图上航迹点：{track_n} 个"
            + ("（WGS84 lon/lat）" if "WGS84" in str(crs) else ""),
            f"车辆航点数：{report.get('vehicle_waypoint_count', 0)}",
            f"max_spacing_m：{report.get('max_spacing_m')}",
            f"geometry_valid：{report.get('geometry_valid')}",
            f"export_valid：{report.get('export_valid')}",
            "",
        ]
        if yaml_ok and official:
            lines.extend([
                "已生成正式小车航点：",
                f"  ✅ {official}",
                "  （waypoints.yaml 为兼容副本，内容一致）",
            ])
            message_box.setIcon(QMessageBox.Icon.Information)
        else:
            reason = (
                report.get("subject1_block_reason")
                or prep_err
                or "验收未通过"
            )
            lines.extend([
                f"未生成 subject1_waypoints.yaml，原因：{reason}",
                "请先完成「生成车辆航点」→「验证车辆航点」，确认 export_valid=true。",
            ])
            message_box.setIcon(QMessageBox.Icon.Warning)
        lines.append("\n已生成：")
        lines.extend(f"- {name}" for name in export_result.get("exported_files") or [])
        message_box.setText("\n".join(lines))
        skipped_layers = report.get("skipped_layers", [])
        export_warnings = report.get("warnings", [])
        if skipped_layers or export_warnings:
            details = []
            if skipped_layers:
                details.append("调试图层被跳过：" + ", ".join(skipped_layers))
            details.extend(str(w) for w in export_warnings[:8])
            message_box.setInformativeText("\n".join(details))
        open_button = message_box.addButton(
            "打开输出目录", QMessageBox.ButtonRole.ActionRole
        )
        message_box.addButton(QMessageBox.StandardButton.Close)
        message_box.exec()
        if message_box.clickedButton() is open_button:
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(selected_dir)):
                QMessageBox.warning(self, "打开输出目录失败", selected_dir)
        self._status_bar.show_message(
            f"比赛路网图已导出: {selected_dir}"
            + (f" | {official}" if yaml_ok and official else " | 未生成 subject1_waypoints.yaml")
        )

    def _on_fix_sparse_cutting_corners(self):
        """deprecated: use vwp_repair_vehicle_csv."""
        return self._on_fix_sparse_cutting_corners_deprecated()

    def _on_fix_sparse_cutting_corners_legacy_body(self):
        """修复稀疏航点切弯：LOS 检查 → 插入 dense_path 中间点 → 可选重导出 YAML。"""
        if len(self.planned_path_pixel) < 2:
            QMessageBox.warning(
                self, "修复稀疏航点切弯",
                "当前没有 dense path，请先完成路径规划。",
            )
            return
        if not self.sparse_waypoints and not self.sparse_waypoints_pixel:
            QMessageBox.warning(
                self, "修复稀疏航点切弯",
                "当前没有稀疏航点。请先导出路径生成 sparse waypoints，或直接重新导出。",
            )
            return

        from roadnet.adaptive_waypoint_resampler import (
            AdaptiveWaypointConfig, fix_sparse_cutting_corners,
        )
        from roadnet.path_export import _subject1_yaml_text, _write_text_atomically

        waypoint_values = self._param_panel.get_config().get("waypoints", {})
        allowed = set(AdaptiveWaypointConfig.__dataclass_fields__)
        config = AdaptiveWaypointConfig(**{
            key: value for key, value in waypoint_values.items() if key in allowed
        })
        road_mask = self._layer_manager.get_layer_data("mask")
        source_wps = self.sparse_waypoints or self.sparse_waypoints_pixel

        try:
            repaired = fix_sparse_cutting_corners(
                self.planned_path_pixel,
                source_wps,
                self._geo_calibration,
                road_mask=road_mask,
                final_graph=self._graph_editor,
                config=config,
            )
        except Exception as exc:
            QMessageBox.critical(self, "修复稀疏航点切弯失败", str(exc))
            import traceback
            traceback.print_exc()
            return

        if not repaired.get("ok", False):
            QMessageBox.warning(
                self, "修复稀疏航点切弯",
                repaired.get("error") or "修复失败",
            )
            return

        pixels = repaired["sparse_waypoints_pixel"]
        self.sparse_waypoints_pixel = [list(p) for p in pixels]

        # Rebuild waypoint dicts for canvas / optional YAML
        geo = self._geo_calibration
        new_waypoints = []
        new_geo = []
        for index, (x, y) in enumerate(pixels):
            lon = lat = None
            if geo is not None and geo.is_valid:
                try:
                    lon, lat = geo.pixel_to_wgs84(float(x), float(y))
                    new_geo.append([float(lon), float(lat), 0.0])
                except Exception:
                    pass
            tag = (repaired.get("tags") or ["curve"])[index] if index < len(repaired.get("tags") or []) else "curve"
            new_waypoints.append({
                "seq": index + 1,
                "x_pixel": float(x),
                "y_pixel": float(y),
                "longitude": lon,
                "latitude": lat,
                "altitude": 0.0,
                "tag": tag,
                "forced": True,
                "path_distance_m": (repaired.get("sparse_s") or [None])[index]
                if index < len(repaired.get("sparse_s") or []) else None,
            })
        self.sparse_waypoints = new_waypoints
        self.sparse_waypoints_geo = new_geo
        self._layer_manager.show_layer("layer_sparse_waypoints")
        self._tool_panel.set_layer_checkbox_state("layer_sparse_waypoints", True)
        self._render_sparse_waypoints_to_scene()

        geometry_valid = bool(repaired.get("geometry_valid"))
        inserted = int(repaired.get("inserted_midpoint_count", 0))
        suspicious = int(repaired.get("suspicious_chord_count", 0))

        yaml_note = ""
        if geometry_valid and any(wp.get("longitude") is not None for wp in new_waypoints):
            out_dir = os.path.join(os.getcwd(), "outputs", "path_fix")
            os.makedirs(out_dir, exist_ok=True)
            s1_text, _s1_stats = _subject1_yaml_text(
                new_waypoints, default_altitude_m=0.0,
            )
            yaml_path = os.path.join(out_dir, "subject1_waypoints.yaml")
            _write_text_atomically(yaml_path, s1_text)
            yaml_note = f"\n\ngeometry_valid=true，已生成：\n{yaml_path}"
        elif not geometry_valid:
            yaml_note = (
                "\n\ngeometry_valid=false，未生成正式 subject1_waypoints.yaml。"
                "请检查 road mask / dense path，或再次导出。"
            )

        QMessageBox.information(
            self, "修复稀疏航点切弯",
            f"插入中间点：{inserted}\n"
            f"仍可疑弦段：{suspicious}\n"
            f"geometry_valid：{geometry_valid}"
            f"{yaml_note}",
        )
        self._status_bar.show_message(
            f"稀疏航点切弯修复完成: inserted={inserted}, valid={geometry_valid}"
        )

    def _build_adaptive_waypoint_config(self):
        from roadnet.adaptive_waypoint_resampler import AdaptiveWaypointConfig
        waypoint_values = self._param_panel.get_config().get("waypoints", {})
        allowed_config_keys = set(AdaptiveWaypointConfig.__dataclass_fields__)
        return AdaptiveWaypointConfig(**{
            key: value for key, value in waypoint_values.items()
            if key in allowed_config_keys
        })

    def _path_node_sequence_from_result(self, result):
        path_node_sequence = []
        for segment in (getattr(result, "segments", None) or []):
            for node_id in (getattr(segment, "node_path", None) or []):
                if not path_node_sequence or node_id != path_node_sequence[-1]:
                    path_node_sequence.append(node_id)
        return path_node_sequence

    def _on_generate_vehicle_waypoints(self):
        """deprecated: use _on_vwp_generate_vehicle_csv / full pipeline instead."""
        return self._on_vwp_generate_vehicle_csv()

    def _build_validation_config(self):
        from roadnet.waypoint_validator import WaypointValidationConfig
        wp = self._param_panel.get_config().get("waypoints", {})
        return WaypointValidationConfig(
            max_allowed_spacing_m=float(wp.get("max_waypoint_spacing_m", 12.0)),
            hard_fail_spacing_m=float(wp.get("hard_fail_spacing_m", 20.0)),
            allow_long_straight=bool(wp.get("allow_long_straight", False)),
            curve_angle_threshold_deg=float(wp.get("corner_angle_threshold_deg", 15.0)),
            curve_buffer_m=float(wp.get("corner_buffer_m", 5.0)),
            curve_spacing_m=float(wp.get("curve_spacing_m", 2.0)),
            junction_buffer_m=float(wp.get("intersection_buffer_m", 8.0)),
            junction_spacing_m=float(wp.get("intersection_spacing_m", 2.0)),
            task_buffer_m=float(wp.get("task_point_buffer_m", 5.0)),
            task_spacing_m=float(wp.get("task_point_spacing_m", 2.0)),
            min_mask_support_ratio=float(wp.get("min_mask_support_ratio", 0.75)),
            max_chord_error_m=float(wp.get("max_chord_error_m", 1.0)),
            max_insert_iterations=int(wp.get("max_insert_iterations", 6)),
        )

    def _on_layered_path_diagnosis(self):
        """deprecated: UI hidden; use vehicle waypoint pipeline."""
        return self._on_layered_path_diagnosis_deprecated()

    def _on_layered_path_diagnosis_legacy_body(self):
        """deprecated legacy body."""
        result = self.planning_result
        if result is None or not bool(getattr(result, "success", False)):
            QMessageBox.warning(
                self, "分层路径诊断",
                "当前没有成功的规划结果，请先「规划路径」。",
            )
            return
        ge = self._graph_editor
        if ge is None:
            QMessageBox.warning(self, "分层路径诊断", "当前没有 final_graph。")
            return
        nodes = list(getattr(ge, "nodes", None) or [])
        edges = list(getattr(ge, "edges", None) or [])
        if not nodes or not edges:
            QMessageBox.warning(self, "分层路径诊断", "final_graph 缺少 nodes/edges。")
            return

        def _as_dict(obj):
            if isinstance(obj, dict):
                return obj
            if hasattr(obj, "__dict__"):
                return dict(obj.__dict__)
            return obj

        nodes_d = [_as_dict(n) for n in nodes]
        edges_d = [_as_dict(e) for e in edges]

        img_w = int(getattr(self._geo_calibration, "image_width", 0) or 0) if self._geo_calibration else 0
        img_h = int(getattr(self._geo_calibration, "image_height", 0) or 0) if self._geo_calibration else 0
        if img_w <= 0 and hasattr(self._layer_manager, "full_image_rgb"):
            arr = self._layer_manager.full_image_rgb
            if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                img_h, img_w = arr.shape[:2]

        from roadnet.path_export import default_path_export_dir
        out_dir = default_path_export_dir(os.getcwd())
        mpp = 0.5
        if self._geo_calibration is not None:
            try:
                mpp = float(getattr(self._geo_calibration, "pixel_resolution_estimated_m", 0.5) or 0.5)
            except (TypeError, ValueError):
                mpp = 0.5

        try:
            from roadnet.path_layer_diagnostics import (
                PathLayerDiagConfig,
                classify_aba_source,
                run_layered_path_diagnostics,
            )
            diag = run_layered_path_diagnostics(
                result,
                nodes_d,
                edges_d,
                snapped_task_points=self.snapped_task_points or self._snapped_points,
                geo_calibration=self._geo_calibration,
                config=PathLayerDiagConfig(metres_per_pixel=mpp),
                output_dir=out_dir,
                preview_image=getattr(self._layer_manager, "full_image_rgb", None),
                image_width=img_w,
                image_height=img_h,
            )
        except Exception as exc:
            QMessageBox.critical(self, "分层路径诊断失败", str(exc))
            return

        vehicle_ok = None
        aba_source = None
        if self.sparse_waypoints:
            try:
                from roadnet.waypoint_validator import validate_vehicle_waypoints
                validation = validate_vehicle_waypoints(
                    self.sparse_waypoints,
                    dense_path_pixel=diag.dense_path_points or self.planned_path_pixel,
                    geo_calibration=self._geo_calibration,
                    final_graph=ge,
                    task_points=self.snapped_task_points or self._task_points,
                    road_mask=self._layer_manager.get_layer_data("mask"),
                    path_node_sequence=self._path_node_sequence_from_result(result),
                    config=self._build_validation_config(),
                    adaptive_config=self._build_adaptive_waypoint_config(),
                    image_width=img_w,
                    image_height=img_h,
                )
                vehicle_ok = bool(validation.export_valid)
                aba_source = classify_aba_source(diag.dense_report, validation.report or {})
            except Exception:
                vehicle_ok = False

        diag.vehicle_waypoints_valid = vehicle_ok
        diag.aba_source = aba_source
        self._path_layer_diag_result = diag
        self._path_layer_diag_dir = out_dir

        ff = diag.first_failure or {}
        dense_resolved = bool(
            (not diag.dense_path_valid)
            and vehicle_ok
            and any(
                str(b.get("reason") or "") == "step_distance_too_large"
                for b in (diag.dense_bad_segments or [])
            )
        )
        lines = [
            f"planned_segments_valid = {diag.planned_segments_valid}",
            f"dense_path_raw_valid = {diag.dense_path_valid}",
            f"vehicle_waypoints_valid = {vehicle_ok}",
            "",
            f"输出目录：{out_dir}",
            "  - planned_segments_debug.csv",
            "  - dense_path_debug.csv",
            "  - dense_path_validation_report.json",
            "  - dense_path_validation_overlay.png",
            "  - virtual_node_split_debug.csv",
        ]
        if dense_resolved:
            lines.extend([
                "",
                "⚠ dense_path_raw_valid=false（中间层 warning）",
                "  dense_path_warning_resolved_by_resampling = true",
                "  原始 dense_path 存在较长步长，但最终车辆航点已重采样并通过验证。",
                "",
                "最终结论：车辆航点通过，可导出",
            ])
        elif vehicle_ok and diag.planned_segments_valid:
            lines.append("\n最终结论：车辆航点通过，可导出")
        if aba_source:
            lines.append(f"\naba_source = {aba_source}")
        if ff and not dense_resolved:
            lines.extend([
                "",
                "第一个失败位置：",
                f"  layer = {ff.get('layer')}",
                f"  segment_index = {ff.get('segment_index')}",
                f"  edge_id = {ff.get('edge_id')}",
                f"  from_wp = {ff.get('from_wp')}",
                f"  to_wp = {ff.get('to_wp')}",
                f"  reason = {ff.get('reason')}",
                f"  debug_overlay_path = {ff.get('debug_overlay_path') or diag.artifact_paths.get('dense_path_validation_overlay.png')}",
            ])
        elif ff and dense_resolved:
            lines.extend([
                "",
                "dense_path 中间层提示（已不阻断导出）：",
                f"  reason = {ff.get('reason')}",
                f"  segment_index = {ff.get('segment_index')}",
                f"  edge_id = {ff.get('edge_id')}",
            ])
        elif not ff:
            lines.append("\n三层前置检查未发现失败（vehicle 层取决于是否已生成航点）。")

        box = QMessageBox.information if (
            vehicle_ok is not False and diag.planned_segments_valid
        ) else QMessageBox.warning
        box(self, "分层路径诊断", "\n".join(lines))
        self._status_bar.show_message(
            f"分层诊断: planned={diag.planned_segments_valid} "
            f"dense_raw={diag.dense_path_valid} vehicle={vehicle_ok}"
        )

    def _on_validate_vehicle_waypoints(self):
        """导出前自动验收：重复点 / 点距 / 加密 / LOS / ABA。"""
        if not self.sparse_waypoints:
            QMessageBox.warning(
                self, "验证车辆航点",
                "尚无车辆航点，请先「生成车辆航点」。",
            )
            return
        result = self.planning_result
        img_w = int(getattr(self._geo_calibration, "image_width", 0) or 0) if self._geo_calibration else 0
        img_h = int(getattr(self._geo_calibration, "image_height", 0) or 0) if self._geo_calibration else 0
        if img_w <= 0 and hasattr(self._layer_manager, "full_image_rgb"):
            arr = self._layer_manager.full_image_rgb
            if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                img_h, img_w = arr.shape[:2]
        try:
            from roadnet.waypoint_validator import validate_vehicle_waypoints
            validation = validate_vehicle_waypoints(
                self.sparse_waypoints,
                dense_path_pixel=self._dense_path_for_waypoints or self.planned_path_pixel,
                geo_calibration=self._geo_calibration,
                final_graph=self._graph_editor,
                task_points=self.snapped_task_points or self._task_points,
                road_mask=self._layer_manager.get_layer_data("mask"),
                path_node_sequence=self._path_node_sequence_from_result(result) if result else None,
                config=self._build_validation_config(),
                adaptive_config=self._build_adaptive_waypoint_config(),
                image_width=img_w,
                image_height=img_h,
            )
        except Exception as exc:
            QMessageBox.critical(self, "验证车辆航点失败", str(exc))
            return

        self.sparse_waypoints = list(validation.waypoints)
        self.sparse_waypoints_pixel = [
            [float(wp["x_pixel"]), float(wp["y_pixel"])] for wp in self.sparse_waypoints
        ]
        self.sparse_waypoints_geo = []
        for wp in self.sparse_waypoints:
            lon = wp.get("longitude_deg", wp.get("longitude"))
            lat = wp.get("latitude_deg", wp.get("latitude"))
            alt = wp.get("altitude_m", wp.get("altitude", 0.0))
            if lon is not None and lat is not None:
                self.sparse_waypoints_geo.append([float(lon), float(lat), float(alt or 0.0)])
        self._waypoint_validation_report = dict(validation.report or {})
        self._waypoint_bad_segments = list(validation.bad_segments or [])

        report = validation.report
        lines = [
            f"export_valid：{report.get('export_valid')}",
            f"geometry_valid：{report.get('geometry_valid')}",
            f"航点数：{report.get('waypoint_count')}",
            f"平均间距：{report.get('average_spacing_m')} m",
            f"最大间距：{report.get('max_spacing_m')} m",
            f"删除重复点：{report.get('duplicate_removed_count')}",
            f"残留连续重复：{report.get('duplicate_consecutive_count')}",
            f"ABA 回跳：{report.get('aba_backtrack_count')}",
            f"异常段：{report.get('bad_segment_count')}",
            f"LOS 失败：{report.get('line_of_sight_failed_count')}",
        ]
        failures = list(report.get("failure_reasons") or [])
        if failures:
            lines.append("\n失败原因：")
            lines.extend(f"- {r}" for r in failures[:12])
        warnings = list(report.get("warnings") or [])
        if warnings:
            lines.append("\n警告：")
            lines.extend(f"- {w}" for w in warnings[:8])

        self._on_show_vehicle_waypoints()
        if self._waypoint_bad_segments:
            self._on_show_waypoint_validation()
        box = QMessageBox.information if report.get("export_valid") else QMessageBox.warning
        box(self, "验证车辆航点", "\n".join(lines))
        self._status_bar.show_message(
            f"航点验收完成: export_valid={report.get('export_valid')}"
        )

    def _on_show_vehicle_waypoints(self):
        """显示 vehicle_waypoints / repaired_vehicle_waypoints（非 dense_path）。"""
        if not self.sparse_waypoints and not self.sparse_waypoints_pixel:
            QMessageBox.warning(
                self, "显示车辆航点",
                "尚无 vehicle_waypoints。\n"
                "请先「② 生成小车航点 CSV」。\n\n"
                "注意：dense_path 图层上的折线不是小车航点。",
            )
            return
        layer = getattr(self, "_vwp_waypoint_layer_name", None) or "vehicle_waypoints"
        path = getattr(self, "_vwp_waypoint_csv_path", None)
        self._layer_manager.show_layer("layer_sparse_waypoints")
        self._tool_panel.set_layer_checkbox_state("layer_sparse_waypoints", True)
        try:
            cb = self._tool_panel._layer_checkboxes.get("layer_sparse_waypoints")
            if cb is not None:
                cb.setText(layer)
        except Exception:
            pass
        self._render_sparse_waypoints_to_scene()
        self._vwp_status_bar_layer(layer, len(self.sparse_waypoints), path)

    def _on_show_waypoint_validation(self):
        """显示异常段图层。"""
        if not self._waypoint_bad_segments:
            QMessageBox.information(
                self, "显示异常段",
                "当前没有异常段。请先「验证车辆航点」，或验收已全部通过。",
            )
            return
        self._layer_manager.show_layer("layer_waypoint_validation")
        self._tool_panel.set_layer_checkbox_state("layer_waypoint_validation", True)
        self._render_waypoint_validation_to_scene()
        self._status_bar.show_message(
            f"已显示异常段：{len(self._waypoint_bad_segments)} 段"
        )

    def _on_export_planned_path(self):
        """导出无人车规划路径数据；与 Mask PNG 导出严格分离。"""
        self._normalize_task_points("before_path_export")
        result = self.planning_result
        if (result is None
                or not bool(getattr(result, "success", False))
                or len(self.planned_path_pixel) < 2):
            QMessageBox.warning(
                self, "导出路径",
                "当前没有可导出的规划路径，请先点击规划路径。",
            )
            return

        missing_ll = [
            tp for tp in self._task_points
            if tp.longitude is None or tp.latitude is None
        ]
        if missing_ll:
            QMessageBox.warning(
                self, "导出路径",
                "手动任务点缺少经纬度，请先完成坐标校准。",
            )
            return
        failed_snaps = [
            sp for sp in (self.snapped_task_points or self._snapped_points or [])
            if getattr(sp, "status", "") == "failed"
        ]
        if failed_snaps:
            QMessageBox.warning(
                self, "导出路径",
                f"{len(failed_snaps)} 个任务点吸附失败，禁止正式导出。\n"
                "请调整任务点或增大路网覆盖后重新吸附。",
            )
            return

        try:
            from roadnet.path_export import (
                InvalidGeoCalibrationError,
                PathOutOfBoundsError,
                convert_pixel_path_to_geo,
                default_path_export_dir,
                export_planned_path,
            )
            # 在选择目录前先验证完整路径可转换，避免创建半成品输出目录。
            convert_pixel_path_to_geo(self.planned_path_pixel, self._geo_calibration)
        except InvalidGeoCalibrationError as exc:
            QMessageBox.critical(
                self, "导出路径 - 坐标标定无效",
                f"无法导出经纬度路径和无人车 waypoints。\n\n{exc}\n\n"
                "请先完成有效的 geo_calibration，并确认 pixel_to_wgs84 可用。",
            )
            return
        except Exception as exc:
            QMessageBox.critical(self, "导出路径 - 前置检查失败", str(exc))
            return

        # Resolve image dimensions for bounds checking.
        img_w = int(getattr(self._geo_calibration, "image_width", 0) or 0)
        img_h = int(getattr(self._geo_calibration, "image_height", 0) or 0)
        if img_w <= 0 and hasattr(self._layer_manager, "full_image_rgb"):
            arr = self._layer_manager.full_image_rgb
            if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                img_h, img_w = arr.shape[:2]

        # ---- Stage A: Dense path bounds pre-check ----
        oob_indices = []
        for idx, (x, y) in enumerate(self.planned_path_pixel):
            if not (0.0 <= x < float(img_w) and 0.0 <= y < float(img_h)):
                oob_indices.append(idx + 1)
        if oob_indices and img_w > 0 and img_h > 0:
            QMessageBox.critical(
                self, "导出路径 - 规划路径越界",
                f"原始规划路径已经越界，请检查 final_graph 或路径规划结果。\n\n"
                f"图像尺寸：{img_w}×{img_h}\n"
                f"越界点数：{len(oob_indices)} / {len(self.planned_path_pixel)}\n"
                f"越界点序列号示例：{oob_indices[:10]}\n\n"
                f"请修复路径规划结果后重新导出。",
            )
            return

        # ---- Stage B: Pre-resample for stats preview ----
        adaptive_config = self._build_adaptive_waypoint_config()
        path_node_sequence = self._path_node_sequence_from_result(result)

        try:
            from roadnet.adaptive_waypoint_resampler import (
                generate_vehicle_waypoints_adaptive,
            )
            adaptive_preview = generate_vehicle_waypoints_adaptive(
                self.planned_path_pixel,
                self.planned_path_geo,
                graph=self._graph_editor,
                task_points=self.snapped_task_points or self._task_points,
                road_mask=self._layer_manager.get_layer_data("mask"),
                config=adaptive_config,
                geo_calibration=self._geo_calibration,
                path_node_sequence=path_node_sequence,
                path_edge_sequence=self.planned_path_edges,
                image_width=img_w,
                image_height=img_h,
            )
        except Exception as exc:
            QMessageBox.critical(self, "导出路径 - 航点重采样失败", str(exc))
            return

        preview_report = adaptive_preview.report
        dense_count = int(preview_report.get("dense_point_count",
                           preview_report.get("dense_path_point_count", 0)))
        sparse_count = int(preview_report.get("vehicle_waypoint_count",
                            preview_report.get("sparse_waypoint_count", 0)))
        average_spacing = float(preview_report.get("average_spacing_m", 0.0) or 0.0)
        max_spacing = float(preview_report.get("max_spacing_m", 0.0) or 0.0)
        path_length = float(preview_report.get("total_length_m",
                             preview_report.get("path_length_m", 0.0)) or 0.0)
        ob_count = int(preview_report.get("out_of_bounds_count", 0))

        stats_text = (
            f"dense_path 点数：{dense_count}\n"
            f"车辆航点数：{sparse_count}\n"
            f"平均间距：{average_spacing:.2f} m\n"
            f"最大间距：{max_spacing:.2f} m\n"
            f"直线路段间距：{adaptive_config.straight_spacing_m:.1f} m\n"
            f"弯道间距：{adaptive_config.curve_spacing_m:.1f} m\n"
            f"路口间距：{adaptive_config.intersection_spacing_m:.1f} m\n"
            f"路径总长：{path_length:.2f} m\n"
            f"geometry_valid：{preview_report.get('geometry_valid')}"
        )
        if max_spacing > 12.0:
            stats_text += "\n\n⚠ 存在过稀疏航点，请检查重采样。"
        if sparse_count > 500:
            stats_text += (
                "\n\n重采样后航点仍较多，建议在高级参数中增大 "
                "straight_spacing_m 或 curve_spacing_m。"
            )
        # Warn about out-of-bounds before user confirms.
        if ob_count > 0:
            stats_text += (
                f"\n\n⚠ 发现 {ob_count} 个航点越界（图像 {img_w}×{img_h}），"
                "导出后将以 INVALID 文件保存，不会生成可用的车辆 YAML 文件。"
            )
        if not preview_report.get("geometry_valid", True):
            stats_text += (
                "\n\n⚠ geometry_valid=false：相邻航点切弯未通过，"
                "不会生成正式 subject1_waypoints.yaml。"
            )

        answer = QMessageBox.question(
            self,
            "导出路径 - 自适应航点预检",
            stats_text + "\n\n是否继续导出？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        output_dir = default_path_export_dir(os.getcwd())
        base_output_dir = output_dir
        suffix = 2
        while os.path.exists(output_dir):
            output_dir = f"{base_output_dir}_{suffix}"
            suffix += 1
        try:
            os.makedirs(output_dir, exist_ok=False)
        except Exception as exc:
            QMessageBox.critical(
                self, "导出路径 - 无法创建目录",
                f"文件路径：\n{output_dir}\n\n失败原因：\n{exc}",
            )
            return

        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "导出规划路径 - 选择输出文件夹",
            output_dir,
            QFileDialog.Option.ShowDirsOnly,
        )
        if not selected_dir:
            try:
                os.rmdir(output_dir)  # 仅清理本次创建且仍为空的默认目录
            except OSError:
                pass
            return
        selected_dir = os.path.abspath(selected_dir)
        if os.path.normcase(selected_dir) != os.path.normcase(os.path.abspath(output_dir)):
            try:
                os.rmdir(output_dir)
            except OSError:
                pass

        # ---- Stage C: Actual export ----
        try:
            export_result = export_planned_path(
                self.planned_path_pixel,
                selected_dir,
                self._geo_calibration,
                planning_result=result,
                task_point_count=len(self._task_points),
                planned_path_edges=self.planned_path_edges,
                adaptive_config=adaptive_config,
                final_graph=self._graph_editor,
                snapped_task_points=self.snapped_task_points or self._task_points,
                path_node_sequence=path_node_sequence,
                preview_image=self._layer_manager.full_image_rgb,
                image_width=img_w,
                image_height=img_h,
                road_mask=self._layer_manager.get_layer_data("mask"),
            )
            self.planned_path_geo = convert_pixel_path_to_geo(
                self.planned_path_pixel, self._geo_calibration
            )
            self.sparse_waypoints_pixel = list(export_result["sparse_waypoints_pixel"])
            self.sparse_waypoints_geo = list(export_result["sparse_waypoints_geo"])
            self.sparse_waypoints = list(export_result["waypoints"])
            self._waypoint_validation_report = dict(
                export_result.get("waypoint_validation_report") or {}
            )
            self._waypoint_bad_segments = list(export_result.get("bad_segments") or [])
            self._path_layer_diag_dir = selected_dir
            self._dense_path_for_waypoints = list(
                export_result.get("dense_path_pixel") or self.planned_path_pixel
            )
            self._layer_manager.show_layer("layer_sparse_waypoints")
            self._tool_panel.set_layer_checkbox_state("layer_sparse_waypoints", True)
            self._render_sparse_waypoints_to_scene()
            if self._waypoint_bad_segments:
                self._layer_manager.show_layer("layer_waypoint_validation")
                self._tool_panel.set_layer_checkbox_state("layer_waypoint_validation", True)
                self._render_waypoint_validation_to_scene()
            self._project_manager.data.planned_path_file = os.path.join(
                selected_dir, "global_path_dense_pixel.json"
            )
        except PathOutOfBoundsError as exc:
            # Dense path bounds failure — a partial report was generated.
            QMessageBox.critical(
                self, "导出路径 - 规划路径越界",
                str(exc) + f"\n\n调试文件已写入：\n{selected_dir}",
            )
            return
        except InvalidGeoCalibrationError as exc:
            QMessageBox.critical(
                self, "导出路径 - 坐标标定无效", str(exc)
            )
            return
        except Exception as exc:
            QMessageBox.critical(
                self, "路径导出失败",
                f"文件路径：\n{selected_dir}\n\n"
                f"失败原因：\n{exc}\n\n"
                "请检查目录是否存在、写入权限、planned_path 和 geo_calibration。",
            )
            return

        # ---- Stage D: Result display ----
        export_valid = bool(export_result.get("export_valid", True))
        planning_report = export_result.get("planning_report", {})
        resample_report = export_result.get("waypoint_resample_report", {})
        wp_bounds = export_result.get("waypoint_bounds", {})
        waypoint_ob_count = int(wp_bounds.get("out_of_bounds_count", 0))
        recommended = export_result.get("recommended_vehicle_file")
        jump_report = export_result.get("suspicious_jump_report", {})
        suspicious_jump_count = int(jump_report.get("suspicious_jump_count", 0))
        long_seg_count = int(jump_report.get("long_segment_candidate_count", 0))
        semantic_valid = resample_report.get("semantic_valid", False)
        task_order_valid = resample_report.get("task_order_valid", False)
        repeated_vns = resample_report.get("repeated_task_virtual_nodes", [])
        goal_early = resample_report.get("goal_appears_before_final_segment", False)
        roundtrip_bad = int(resample_report.get("bad_roundtrip_count", 0))
        task_order_matches = planning_report.get("task_order_matches", None)
        actual_order = planning_report.get("actual_task_visit_order", [])
        expected_order = planning_report.get("expected_task_visit_order", [])
        planned_segs = planning_report.get("planned_segments", [])
        seg_val_errors = planning_report.get("segment_validation_errors", [])
        seg_isolation_valid = planning_report.get("segment_isolation_valid", False)
        geometry_valid = planning_report.get("geometry_valid", False)
        coordinate_valid_plan = planning_report.get("coordinate_valid", False)

        try:
            display_dir = os.path.relpath(selected_dir, os.getcwd())
        except ValueError:
            display_dir = selected_dir
        file_names = [os.path.basename(path) for path in export_result["exported_files"]]

        # ---- Post-export layered diagnostics ----
        if not export_valid:
            # Build layered error message
            lines = ["导出验证失败，以下是分层诊断：", ""]

            # Layer -1: planned_segments → dense_path → vehicle_waypoints
            val_report = export_result.get("waypoint_validation_report") or {}
            planned_ok = export_result.get(
                "planned_segments_valid",
                planning_report.get("planned_segments_valid"),
            )
            dense_raw_ok = export_result.get(
                "dense_path_raw_valid",
                export_result.get(
                    "dense_path_valid",
                    planning_report.get("dense_path_raw_valid",
                                        planning_report.get("dense_path_valid")),
                ),
            )
            vehicle_ok = export_result.get(
                "vehicle_waypoints_valid",
                val_report.get("vehicle_waypoints_valid"),
            )
            dense_resolved = bool(
                export_result.get("dense_path_warning_resolved_by_resampling")
                or planning_report.get("dense_path_warning_resolved_by_resampling")
                or val_report.get("dense_path_warning_resolved_by_resampling")
            )
            ff = export_result.get("path_layer_first_failure") or planning_report.get(
                "path_layer_first_failure"
            ) or {}
            aba_src = export_result.get("aba_source") or planning_report.get("aba_source")
            lines.append("── 分层路径诊断（planned → dense → vehicle）──")
            lines.append(f"  planned_segments_valid = {planned_ok}")
            lines.append(f"  dense_path_raw_valid = {dense_raw_ok}")
            lines.append(f"  vehicle_waypoints_valid = {vehicle_ok}")
            lines.append(f"  export_valid = {export_valid}")
            if dense_raw_ok is False:
                lines.append("  ⚠ dense_path 为中间层 warning（不单独阻断 YAML）")
            if dense_resolved:
                lines.append("  dense_path_warning_resolved_by_resampling = true")
                lines.append(
                    "  原始 dense_path 存在较长步长，但最终车辆航点已重采样并通过验证。"
                )
            if aba_src:
                lines.append(f"  aba_source = {aba_src}")
            if ff:
                lines.append(
                    f"  dense/planned 提示: layer={ff.get('layer')} "
                    f"segment={ff.get('segment_index')} "
                    f"edge={ff.get('edge_id')} reason={ff.get('reason')}"
                )
                if ff.get("debug_overlay_path"):
                    lines.append(f"  overlay: {ff.get('debug_overlay_path')}")
            lines.append(
                "  详见 planned_segments_debug.csv / dense_path_debug.csv / "
                "dense_path_validation_report.json"
            )
            lines.append("")

            # Layer 0: Waypoint validation gate
            lines.append("── 车辆航点验收 ──")
            lines.append(
                f"  max_spacing_m={val_report.get('max_spacing_m')}  "
                f"bad_segments={val_report.get('bad_segment_count')}  "
                f"geometry_valid={val_report.get('geometry_valid')}"
            )
            for reason in (val_report.get("failure_reasons") or [])[:10]:
                lines.append(f"  ❌ {reason}")
            if not val_report.get("failure_reasons"):
                lines.append("  （详见 waypoint_validation_report.json / bad_segments.csv）")
            lines.append("")

            # Layer 1: Coordinate validity
            lines.append("── 坐标合法性 ──")
            oob_ok = (waypoint_ob_count == 0)
            rt_ok = (roundtrip_bad == 0)
            if oob_ok and rt_ok:
                lines.append("  ✅ 坐标检查通过。")
            else:
                if not oob_ok:
                    lines.append(f"  ❌ 越界航点数: {waypoint_ob_count}")
                if not rt_ok:
                    lines.append(f"  ❌ 往返误差异常 (>2px) 航点数: {roundtrip_bad}")
            lines.append("")

            # Layer 2: Segment-level task virtual node isolation
            lines.append("── 分段任务点隔离检查 ──")
            if seg_val_errors:
                for err in seg_val_errors[:5]:
                    lines.append(f"  ❌ {err}")
                    if "意外经过" in err:
                        lines.append("     → 这通常说明所有任务点 virtual node 被全局插入了 planning_graph。")
                        lines.append("     → 请确保每段只插入当前 from/to virtual node。")
            else:
                lines.append(f"  ✅ segment_isolation_valid={seg_isolation_valid}")
            lines.append("")

            # Layer 3: Task visit order (★ SINGLE AUTHORITATIVE SOURCE)
            lines.append("── 任务点访问顺序 ──")
            if task_order_matches:
                lines.append(f"  ✅ actual_task_visit_order = [{', '.join(map(str, actual_order))}]")
                lines.append(f"  ✅ expected_task_visit_order = [{', '.join(map(str, expected_order))}]")
                lines.append(f"  ✅ task_order_valid=true (来源: planned_segments)")
            else:
                lines.append(f"  ❌ 实际访问顺序: {actual_order}")
                lines.append(f"     期望顺序:      {expected_order}")
                lines.append(f"  ❌ task_order_valid=false")
            lines.append("")

            # Layer 4: Semantic checks (★ uses authoritative semantic_valid from resample)
            lines.append("── 语义综合检查 ──")
            if semantic_valid:
                lines.append("  ✅ semantic_valid=true")
                lines.append(f"     - task_order_valid={task_order_valid} (来源: {resample_report.get('task_order_source', 'unknown')})")
                lines.append(f"     - segment_isolation_valid={seg_isolation_valid}")
                if repeated_vns:
                    lines.append(f"     ⚠ 注意: 存在重复虚拟节点 {repeated_vns}（但未影响 overall valid）")
                if goal_early:
                    lines.append("     ⚠ 注意: goal 在最终 segment 前出现（但未影响 overall valid）")
                lines.append(f"     - start_count={resample_report.get('start_count', '?')}, goal_count={resample_report.get('goal_count', '?')}")
            else:
                # Break down WHY semantic_valid is false
                if not task_order_valid and task_order_matches:
                    # ★ CRITICAL: task_order_matches says True but resample says False
                    #    → data discrepancy between planning_report and resample_report.
                    lines.append(f"  ❌ semantic_valid=false — INTERNAL DATA MISMATCH")
                    lines.append(f"     planning_report.task_order_matches = {task_order_matches}")
                    lines.append(f"     resample_report.task_order_valid   = {task_order_valid}")
                    lines.append(f"     task_order_source                  = {resample_report.get('task_order_source', '?')}")
                    lines.append(f"     actual_task_visit_order            = {actual_order}")
                    lines.append(f"     expected_task_visit_order          = {expected_order}")
                else:
                    lines.append("  ❌ semantic_valid=false")
                    if not task_order_valid:
                        lines.append(f"     ❌ task_order_valid=false (source: {resample_report.get('task_order_source', '?')})")
                        lines.append(f"        actual  = {actual_order}")
                        lines.append(f"        expected= {expected_order}")
                    if not seg_isolation_valid:
                        lines.append("     ❌ segment_isolation_valid=false")
                    if repeated_vns:
                        lines.append(f"     ❌ 重复的 task virtual 节点: {repeated_vns}")
                    if goal_early:
                        lines.append("     ❌ goal 任务点在最终 segment 之前提前出现。")
                    sc = resample_report.get("start_count", 1)
                    gc = resample_report.get("goal_count", 1)
                    if sc != 1:
                        lines.append(f"     ❌ start_count={sc}(期望=1)")
                    if gc != 1:
                        lines.append(f"     ❌ goal_count={gc}(期望=1)")
            lines.append("")

            # Layer 5: Path geometry validity
            lines.append("── 路径几何检查 ──")
            if suspicious_jump_count == 0:
                if long_seg_count > 0:
                    lines.append(f"  ⚠ 检测到 {long_seg_count} 个长直路段，经道路合法性验证均通过。")
                    lines.append(f"  ✅ geometry_valid={geometry_valid} (长直路段已通过道路合法性验证，不影响导出)")
                else:
                    lines.append(f"  ✅ geometry_valid={geometry_valid}")
            else:
                lines.append(f"  ❌ geometry_valid=false")
                lines.append(f"     真正异常跳边数: {suspicious_jump_count}")
                lines.append(f"     长直路候选数: {long_seg_count}（其中 {long_seg_count - suspicious_jump_count} 个验证通过）")
            lines.append("")

            # Per-segment detail
            lines.append("── 各段详情 ──")
            for seg in planned_segs:
                sidx = seg.get("segment_index", "?")
                fseq = seg.get("from_seq", "?")
                tseq = seg.get("to_seq", "?")
                fvn = seg.get("from_virtual_node", "")
                tvn = seg.get("to_virtual_node", "")
                ok = seg.get("success", False)
                unexpected = seg.get("unexpected_task_virtual_nodes", [])
                err = seg.get("error", "")
                icon = "✅" if ok else "❌"
                lines.append(f"  {icon} 段{sidx}: seq={fseq}→{tseq}")
                lines.append(f"     from={fvn}, to={tvn}")
                if unexpected:
                    lines.append(f"     ❌ 意外 task virtual 节点: {unexpected}")
                if err and not ok:
                    lines.append(f"     错误: {err[:120]}")
            lines.append("")

            lines.append("── 结论 ──")
            lines.append("  路径验证失败，未生成 subject1_waypoints.yaml。")
            lines.append("  请查看：")
            lines.append("    - waypoint_resample_report.json")
            lines.append("    - planning_report.json")
            lines.append("    - waypoints_sparse_10m_INVALID.csv")
            lines.append(f"  详情见: {display_dir}")


            # ★ 失败时直接显示诊断弹窗并返回，不再弹出第二个对话框
            failure_box = QMessageBox(self)
            failure_box.setWindowTitle("导出路径 - 验证失败")
            failure_box.setIcon(QMessageBox.Icon.Warning)
            failure_box.setText("\n".join(lines))
            failure_box.addButton(QMessageBox.StandardButton.Close)
            failure_box.exec()
            return

        # ---- Success path: show single result dialog ----
        # Separate file categories for clarity
        _subject1_key = "subject1_waypoints.yaml"
        _vehicle_files = [n for n in file_names if n in ("subject1_waypoints.yaml",)]
        _check_files = [n for n in file_names
                        if n in ("waypoints_sparse_10m.csv", "waypoints_sparse_10m.yaml")]
        _report_files = [n for n in file_names
                         if n in ("waypoint_resample_report.json", "planning_report.json")]
        _debug_files = [n for n in file_names
                        if n not in _vehicle_files and n not in _check_files
                        and n not in _report_files]

        # Build success message.
        message_text = (
            "路径导出完成。\n\n"
            f"输出目录：\n{display_dir}\n\n"
            f"原始路径点数：{dense_count}\n"
            f"稀疏航点数：{sparse_count}\n"
            f"平均间距：{average_spacing:.2f} m\n\n"
            "已生成：\n"
        )
        if _vehicle_files:
            message_text += "\n".join(f"  ✅ {name}  ← 小车使用" for name in _vehicle_files) + "\n"
        if _check_files:
            message_text += "\n".join(f"  - {name}  ← 人工检查" for name in _check_files) + "\n"
        if _report_files:
            message_text += "\n".join(f"  - {name}  ← 验证报告" for name in _report_files) + "\n"
        for name in _debug_files:
            message_text += f"  - {name}  ← 调试\n"

        if export_valid and recommended:
            message_text += (
                "\n✅ 最终结论：车辆航点通过，可导出。\n"
                f"已生成正式小车航点：\n  ✅ {recommended}\n"
                "  （waypoints.yaml 为兼容副本，内容一致）"
            )
            if (
                export_result.get("dense_path_warning_resolved_by_resampling")
                or planning_report.get("dense_path_warning_resolved_by_resampling")
            ):
                message_text += (
                    "\n\n⚠ dense_path_raw_valid=false（中间层 warning）\n"
                    "原始 dense_path 存在较长步长，但最终车辆航点已重采样并通过验证。"
                )
            elif export_result.get("dense_path_raw_valid") is False or (
                planning_report.get("dense_path_raw_valid") is False
            ):
                message_text += (
                    "\n\n⚠ dense_path_raw_valid=false（中间层 warning，未阻断导出）"
                )
        elif not export_valid:
            reason = (
                (export_result.get("planning_report") or {}).get("invalid_reason")
                or "验收未通过"
            )
            message_text += (
                f"\n未生成 subject1_waypoints.yaml，原因：{reason}"
            )

        message_box = QMessageBox(self)
        message_box.setWindowTitle("导出路径")
        message_box.setIcon(QMessageBox.Icon.Information)
        message_box.setText(message_text)
        open_button = message_box.addButton(
            "打开输出目录", QMessageBox.ButtonRole.ActionRole
        )
        message_box.addButton(QMessageBox.StandardButton.Close)
        message_box.exec()
        if message_box.clickedButton() is open_button:
            opened = QDesktopServices.openUrl(QUrl.fromLocalFile(selected_dir))
            if not opened:
                QMessageBox.warning(self, "打开输出目录失败", selected_dir)
        self._status_bar.show_message(f"路径已导出: {selected_dir}")

    def _on_export_mask(self):
        mask = self._layer_manager.get_layer_data("mask")
        if mask is None:
            QMessageBox.warning(self, "提示", "没有可导出的 Mask。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 Mask", "road_mask_export.png", "PNG 文件 (*.png);;所有文件 (*)")
        if path:
            import cv2
            cv2.imwrite(path, mask)
            self._status_bar.show_message(f"Mask 已导出: {os.path.basename(path)}")

    def _on_save_current_mask(self):
        """Save the editable mask to a timestamped artifact + large-image working/final."""
        mask = self._layer_manager.get_layer_data("mask")
        if mask is None:
            QMessageBox.warning(self, "保存当前 Mask", "当前没有可保存的 Road Mask。")
            return None
        from datetime import datetime
        from roadnet.region_edit import ensure_mask_uint8, save_mask_png_verified

        # 大图：拒绝把 preview 尺寸当成正式 mask 保存
        if self._layer_manager.is_large_image_mode and self._large_image_project is not None:
            ow = int(self._large_image_project.image_width or 0)
            oh = int(self._large_image_project.image_height or 0)
            if ow > 0 and oh > 0 and tuple(mask.shape[:2]) != (oh, ow):
                QMessageBox.warning(
                    self, "保存当前 Mask",
                    f"当前图层尺寸为 {mask.shape[1]}×{mask.shape[0]}，"
                    f"与原图 {ow}×{oh} 不一致，不能保存为正式 mask。\n"
                    "请先重新加载 working_road_mask 或重新生成低像素正式 Mask。",
                )
                return None

        try:
            normalized = ensure_mask_uint8(mask, copy=False)
            if normalized is not mask:
                self._layer_manager.set_layer_data("mask", normalized)
            run_dir = (
                Path(__file__).resolve().parents[1]
                / "outputs" / "masks"
                / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            output_path = run_dir / "current_mask.png"
            output_path = save_mask_png_verified(normalized, output_path)
        except Exception as exc:
            attempted = locals().get("output_path", "尚未创建输出路径")
            QMessageBox.critical(
                self, "保存当前 Mask 失败",
                f"文件路径：{attempted}\n失败原因：{exc}",
            )
            return None

        # ★ 大图：保存 working，并在手修后保存 final_edited_mask
        saved_final = None
        if (self._layer_manager.is_large_image_mode
                and self._large_image_project is not None):
            try:
                saved_final = self._persist_working_mask(normalized, save_as_final=True)
            except Exception as exc:
                QMessageBox.critical(
                    self, "保存当前 Mask 失败",
                    f"写入 final_edited_mask / working_road_mask 失败：\n{exc}",
                )
                return None

        detail = f"文件路径：\n{output_path}"
        if saved_final:
            detail += f"\n\n最终编辑 mask（骨架将优先使用）：\n{saved_final}"
        QMessageBox.information(self, "保存当前 Mask 成功", detail)
        self._update_large_mask_status_bar(
            f"已保存 {os.path.basename(saved_final or str(output_path))}"
        )
        return str(output_path)

    def _persist_working_mask(
        self,
        mask: np.ndarray,
        *,
        clear_cleaned: bool = False,
        save_as_final: bool = False,
        clear_final: bool = False,
    ):
        """将当前 working mask 写入项目目录；显式保存时另存 final_edited_mask。

        生命周期：
          cleaned_working_mask = 中间结果（保留）
          working_road_mask = 当前可编辑副本
          final_edited_mask = 用户点击「保存当前 Mask」后的最终骨架输入

        Args:
            save_as_final: True 时写出 final_edited_mask.png（仅「保存当前 Mask」应传 True）
            clear_final: True 时删除过期 final_edited_mask（新正式 mask 生成后调用）
        """
        import cv2
        project = self._large_image_project
        masks_dir = Path(project.project_dir) / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        # ★ 强制要求 full-size，禁止把 preview 尺寸写成正式 mask
        ow = int(getattr(project, "image_width", 0) or 0)
        oh = int(getattr(project, "image_height", 0) or 0)
        mh, mw = int(mask.shape[0]), int(mask.shape[1])
        if ow > 0 and oh > 0 and (mw, mh) != (ow, oh):
            raise ValueError(
                f"拒绝写入非正式尺寸 mask：got {mw}x{mh}, expected {ow}x{oh}。"
                "请确认图层是 original image pixel 的 working_road_mask。"
            )

        pw = self._layer_manager.preview_width
        ph = self._layer_manager.preview_height
        if pw and ph:
            preview = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST)
        else:
            preview = mask

        working_path = masks_dir / "working_road_mask.png"
        working_prev = masks_dir / "working_road_mask_preview.png"
        cv2.imwrite(str(working_path), mask)
        cv2.imwrite(str(working_prev), preview)
        self._layer_manager.set_layer_data("mask", mask, preview_data=preview)

        project.working_road_mask_path = str(working_path)
        project.edited_global_road_mask_path = str(working_path)
        project.working_road_mask_preview_path = str(working_prev)
        if project.global_mask_path and not project.global_road_mask_path:
            project.global_road_mask_path = project.global_mask_path

        src = getattr(self, "_working_mask_source", "") or ""
        base = getattr(self, "_mask_edit_base", "") or ""
        final_path_str = None

        if clear_final and not save_as_final:
            self._final_edited_mask_path = None
            self._final_edited_mask_preview_path = None
            project.final_edited_mask_path = ""
            project.final_edited_mask_preview_path = ""
            for old_name in ("final_edited_mask.png", "final_edited_mask_preview.png"):
                old_p = masks_dir / old_name
                if old_p.is_file():
                    try:
                        old_p.unlink()
                    except Exception:
                        pass
            try:
                self._layer_manager.set_layer_data("layer_final_edited_mask", None)
            except Exception:
                pass

        if save_as_final:
            final_path = masks_dir / "final_edited_mask.png"
            final_prev = masks_dir / "final_edited_mask_preview.png"
            cv2.imwrite(str(final_path), mask)
            cv2.imwrite(str(final_prev), preview)
            final_path_str = str(final_path)
            self._final_edited_mask_path = final_path_str
            self._final_edited_mask_preview_path = str(final_prev)
            project.final_edited_mask_path = final_path_str
            project.final_edited_mask_preview_path = str(final_prev)
            project.mask_source = "final_edited_mask"
            self._working_mask_source = "final_edited_mask"
            if not base:
                if src == "manual_after_cleaned" or "cleaned" in src:
                    base = "cleaned_working_mask"
                else:
                    base = "global_road_mask"
            project.mask_edit_base = base
            self._mask_edit_base = base
            self._layer_manager.set_layer_data(
                "layer_final_edited_mask", mask, preview_data=preview,
            )
            # 默认只显示 Working（已与 final 同内容），避免多层叠乱
            self._layer_manager.hide_layer("layer_final_edited_mask")
            self._layer_manager.hide_layer("layer_cleaned_road_mask")
            self._layer_manager.show_layer("layer_road_mask")
        else:
            project.mask_source = src or "working_road_mask"
            project.mask_edit_base = base

        if clear_cleaned:
            project.cleaned_working_mask_path = ""
            project.cleaned_working_mask_preview_path = ""
            self._cleaned_working_mask_path = None
            self._cleaned_working_mask_preview_path = None
            self._cleaned_mask_pending = None

        project.mask_dirty = False
        project.formal_ready = True
        project.preview_only = False
        project.save()

        self._working_road_mask_path = str(working_path)
        self._working_road_mask_preview_path = str(working_prev)
        self._working_mask_dirty = False
        self._working_mask_formal_ready = True
        self._working_mask_preview_only = False
        self._formal_mask_meta = {
            **(self._formal_mask_meta or {}),
            "mask_source": project.mask_source,
            "mask_edit_base": project.mask_edit_base,
            "working_mask_path": str(working_path),
            "working_mask_preview_path": str(working_prev),
            "cleaned_mask_path": self._cleaned_working_mask_path or project.cleaned_working_mask_path,
            "cleaned_mask_preview_path": (
                self._cleaned_working_mask_preview_path
                or project.cleaned_working_mask_preview_path
            ),
            "final_edited_mask_path": self._final_edited_mask_path or "",
            "final_edited_mask_preview_path": self._final_edited_mask_preview_path or "",
            "formal_ready": True,
            "preview_only": False,
        }

        key_path = final_path_str or str(working_path)
        skel_dir = os.path.join(project.project_dir, "skeleton")
        self._invalidate_large_skeleton_cache_if_needed(
            skel_dir,
            {
                "selected_mask_path": key_path,
                "checksum": self._mask_file_fingerprint(key_path).get("checksum"),
            },
        )
        self._sync_layer_checkboxes()
        return final_path_str or str(working_path)

    def _on_export_overlay(self):
        if not self._layer_manager.has_image():
            QMessageBox.warning(self, "提示", "没有可导出的叠加图。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出叠加图", "road_overlay_export.png", "PNG 文件 (*.png);;所有文件 (*)")
        if path:
            pixmap = self._canvas.grab()
            pixmap.save(path)
            self._status_bar.show_message(f"叠加图已导出: {os.path.basename(path)}")

    def _on_export_skeleton(self):
        result = self._canvas.skeleton_result
        if result is None:
            QMessageBox.warning(self, "提示", "请先优化骨架。")
            return
        import cv2
        path, _ = QFileDialog.getSaveFileName(self, "保存优化骨架", "road_skeleton_optimized.png", "PNG 文件 (*.png);;所有文件 (*)")
        if path:
            cv2.imwrite(path, result["optimized_skeleton"])
            self._status_bar.show_message(f"骨架已保存: {os.path.basename(path)}")

    def _on_view_compare(self):
        compare_path = os.path.join(os.getcwd(), "outputs", "road_skeleton_optimized_overlay.png")
        if not os.path.exists(compare_path):
            QMessageBox.warning(self, "提示", "对比图不存在，请先优化骨架。")
            return
        try:
            os.startfile(os.path.abspath(compare_path))
        except Exception:
            QMessageBox.information(self, "提示", f"对比图位置:\n{os.path.abspath(compare_path)}")
        self._status_bar.show_message("已打开优化对比图")

    def _on_clear_samples(self):
        self._history.push_state("clear_samples")
        self._canvas.clear_samples()
        self._status_bar.show_message("样本已清空")

    # ===================================================================
    # 鼠标移动（含校准坐标）
    # ===================================================================

    def _on_mouse_moved(self, x: int, y: int):
        """鼠标移动：更新像素坐标，若已校准则同时更新经纬度。"""
        self._status_bar.update_coords(x, y)
        if self._geo_calibration.enabled and self._geo_calibration.pixel_to_world_matrix is not None:
            try:
                lon, lat = self._geo_calibration.pixel_to_lonlat(x, y)
                x_m, y_m = self._geo_calibration.pixel_to_world(x, y)
                self._status_bar.update_geo_coords(lon, lat, x_m, y_m)
            except Exception:
                self._status_bar.update_geo_coords(None, None, None, None)
        else:
            self._status_bar.update_geo_coords(None, None, None, None)

    # ===================================================================
    # 坐标校准操作
    # ===================================================================

    def _on_calibration_corner_preset(self, corner_names: list):
        """快捷勾选顶点组合 — 从 GeoCalibration 保留已有 lon/lat，不触碰控件内部。"""
        # 大图模式下使用原图尺寸进行校准
        original_w, original_h = self._layer_manager.original_size
        w, h = original_w, original_h
        if w <= 0 or h <= 0:
            QMessageBox.warning(self, "提示", "请先打开影像。")
            return

        # 从 GeoCalibration 中保留已存在的 lon/lat
        existing = {}
        for cp in (self._geo_calibration.control_points or []):
            existing[cp.get("name", "")] = cp

        from roadnet.gcp_io import infer_pixel_from_corner_name
        new_cps = []
        for cname in corner_names:
            old = existing.get(cname, {})
            lon = float(old.get("lon", 0.0))
            lat = float(old.get("lat", 0.0))
            px = infer_pixel_from_corner_name(cname, w, h)
            new_cps.append({
                "name": cname,
                "pixel": [px[0], px[1]] if px else [0, 0],
                "lon": lon,
                "lat": lat,
            })

        self._geo_calibration.set_control_points(new_cps)

        # 同步面板（勾选角点 + 填入已有经纬度）
        if hasattr(self._param_panel, 'set_calibration_corners_from_data'):
            self._param_panel.set_calibration_corners_from_data(new_cps, (w, h))

        # 更新画布角点标记
        self._update_calibration_corner_markers()

        names_str = "-".join([c[:2].upper() for c in corner_names])
        self._status_bar.show_message(
            f"已选择 {len(corner_names)} 个顶点: {names_str}，请输入经纬度后计算。")

    def _on_calibration_import_vertex_txt(self):
        """导入比赛图片顶点校准文件（序号;经度;纬度;高程），自动完成三点仿射校准。"""
        original_w, original_h = self._layer_manager.original_size
        w, h = int(original_w), int(original_h)
        if w <= 0 or h <= 0:
            QMessageBox.warning(self, "提示", "请先打开影像。")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "导入图片顶点校准文件", "",
            "校准坐标 TXT (*.txt);;所有文件 (*)"
        )
        if not path:
            return
        self._apply_corner_vertex_calibration_file(path, w, h)

    def _apply_corner_vertex_calibration_file(self, path: str, w: int, h: int) -> bool:
        """解析顶点校准文件 → 绑像素 → ENU 仿射 → 保存 calibration.json。"""
        from roadnet.gcp_io import (
            parse_corner_calibration_txt,
            build_control_points_from_corner_records,
            validate_gcp_points,
        )

        parsed = parse_corner_calibration_txt(path)
        if not parsed.get("ok"):
            QMessageBox.warning(
                self, "导入失败",
                parsed.get("error") or "无法解析顶点校准文件。",
            )
            return False

        swap = False
        if parsed.get("swap_suspect"):
            reply = QMessageBox.question(
                self, "经纬度可能填反",
                "检测到经纬度可能填反，是否交换经纬度？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            swap = reply == QMessageBox.StandardButton.Yes

        # 仅用 1/2/3 做仿射；4 若存在也一并绑定（四点更稳健）
        cps = build_control_points_from_corner_records(
            parsed["records"], w, h, swap_lon_lat=swap,
        )
        # 三点校准至少需要 id 1/2/3 对应点
        needed = {"bottom_left", "bottom_right", "top_left"}
        have = {cp["name"] for cp in cps}
        if not needed.issubset(have):
            QMessageBox.warning(
                self, "导入失败",
                "顶点校准文件必须包含 1=左下角、2=右下角、3=左上角。",
            )
            return False

        # 三点仿射只用 1/2/3；文件若含 4 可参与也可不参与——按需求三点完成
        cps_fit = [cp for cp in cps if cp["name"] in needed]
        ok, err = validate_gcp_points(cps_fit, w, h)
        if not ok:
            QMessageBox.warning(self, "控制点验证失败", err)
            return False

        geo = self._geo_calibration
        geo.reset_state()
        geo.set_control_points(cps_fit)
        geo.set_calibration_metadata("image_corner_3point_affine", w, h)
        geo.calibration_mode = "image_corner_3point_affine"
        geo.source_file = os.path.basename(path)
        geo.method = "image_corner_3point_affine"

        if not geo.setup_projection(force_local_enu=True):
            QMessageBox.warning(self, "校准失败", "无法初始化局部 ENU 投影。")
            return False
        try:
            if not geo.compute_affine():
                QMessageBox.warning(self, "校准失败", "三点仿射变换计算失败。")
                self._param_panel.update_calibration_ui(geo)
                return False
        except Exception as exc:
            QMessageBox.critical(self, "校准失败", f"仿射计算出错：\n{exc}")
            self._param_panel.update_calibration_ui(geo)
            return False

        # 右上角自动推算
        rt_px, rt_py = w - 1, 0
        try:
            rt_lon, rt_lat = geo.pixel_to_lonlat(rt_px, rt_py)
        except Exception:
            rt_lon, rt_lat = 0.0, 0.0

        corner_points = []
        id_by_name = {
            "bottom_left": "1", "bottom_right": "2",
            "top_left": "3", "top_right": "4",
        }
        corner_label = {
            "bottom_left": "left_bottom", "bottom_right": "right_bottom",
            "top_left": "left_top", "top_right": "right_top",
        }
        for cp in cps_fit:
            name = cp["name"]
            corner_points.append({
                "id": id_by_name.get(name, cp.get("corner_id", "")),
                "corner": corner_label.get(name, name),
                "longitude": cp["lon"],
                "latitude": cp["lat"],
                "altitude": cp.get("altitude", 0),
                "pixel_x": int(cp["pixel"][0]),
                "pixel_y": int(cp["pixel"][1]),
            })
        inferred = [{
            "id": "4",
            "corner": "right_top",
            "pixel_x": rt_px,
            "pixel_y": rt_py,
            "longitude": round(float(rt_lon), 8),
            "latitude": round(float(rt_lat), 8),
        }]
        geo.corner_points = corner_points
        geo.inferred_corners = inferred
        geo.coordinate_system = "WGS84_to_local_ENU"

        # 同步 UI（含右上角推算值，便于手动查看）
        ui_cps = list(cps_fit) + [{
            "name": "top_right",
            "pixel": [rt_px, rt_py],
            "lon": round(float(rt_lon), 8),
            "lat": round(float(rt_lat), 8),
        }]
        if hasattr(self._param_panel, "set_calibration_method_combo"):
            self._param_panel.set_calibration_method_combo("image_corner_3point_affine")
        if hasattr(self._param_panel, "set_calibration_corners_from_data"):
            self._param_panel.set_calibration_corners_from_data(ui_cps, (w, h))
        self._param_panel.update_calibration_ui(geo)
        if hasattr(self._param_panel, "set_vertex_calibration_summary"):
            by_id = {c["id"]: c for c in corner_points}
            c1, c2, c3 = by_id.get("1"), by_id.get("2"), by_id.get("3")
            if c1 and c2 and c3:
                summary = (
                    "校准模式：三点图片顶点校准\n"
                    f"1 左下角：{c1['longitude']:.8f}, {c1['latitude']:.8f}, "
                    f"pixel=({c1['pixel_x']},{c1['pixel_y']})\n"
                    f"2 右下角：{c2['longitude']:.8f}, {c2['latitude']:.8f}, "
                    f"pixel=({c2['pixel_x']},{c2['pixel_y']})\n"
                    f"3 左上角：{c3['longitude']:.8f}, {c3['latitude']:.8f}, "
                    f"pixel=({c3['pixel_x']},{c3['pixel_y']})\n"
                    f"4 右上角：自动推算 {rt_lon:.8f}, {rt_lat:.8f}, "
                    f"pixel=({rt_px},{rt_py})\n"
                    "已根据图片顶点坐标自动完成三点仿射校准。"
                )
                self._param_panel.set_vertex_calibration_summary(summary)

        self._update_calibration_corner_markers()
        self._status_bar.update_resolution(
            geo.pixel_resolution_estimated_m
            or self._project_manager.data.pixel_resolution_m,
            calibrated=True,
        )

        # 保存 calibration.json（项目目录优先）
        save_paths = []
        if self._large_image_project is not None:
            proj_cal = os.path.join(
                self._large_image_project.project_dir, "calibration.json"
            )
            save_paths.append(proj_cal)
            try:
                self._large_image_project.geo_calibration_path = proj_cal
                self._large_image_project.save()
            except Exception as exc:
                print(f"[Calibration] 更新大图项目 geo_calibration_path 失败: {exc}")
        outputs_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        save_paths.append(os.path.join(outputs_dir, "calibration.json"))
        # 影像同目录
        img_path = self._layer_manager.image_path or ""
        if img_path:
            save_paths.append(
                os.path.join(os.path.dirname(os.path.abspath(img_path)), "calibration.json")
            )

        saved = []
        for sp in save_paths:
            try:
                if geo.save(sp):
                    saved.append(sp)
            except Exception as exc:
                print(f"[Calibration] 保存失败 {sp}: {exc}")

        if not geo.is_valid:
            QMessageBox.warning(self, "校准失败", "校准后 is_valid 仍为 False，请检查控制点。")
            return False

        # 同步到项目数据，任务点导入可直接复用
        try:
            self._project_manager.data.geo_calibration = geo.to_dict()
            self._project_manager.mark_dirty()
        except Exception:
            pass

        msg = (
            "已根据图片顶点坐标自动完成三点仿射校准。\n\n"
            f"模式：三点图片顶点校准\n"
            f"影像：{w} × {h}\n"
            f"右上角（自动推算）：{rt_lon:.8f}, {rt_lat:.8f}\n"
            f"is_valid = {geo.is_valid}\n"
        )
        if saved:
            msg += f"\n已保存：\n" + "\n".join(saved)
        QMessageBox.information(self, "顶点校准完成", msg)
        self._status_bar.show_message(
            f"三点图片顶点校准完成（valid={geo.is_valid}）"
        )
        return True

    def _on_calibration_import_txt(self):
        """导入 TXT 坐标文件。"""
        # 大图模式下使用原图尺寸进行校准
        original_w, original_h = self._layer_manager.original_size
        w, h = original_w, original_h
        if w <= 0 or h <= 0:
            QMessageBox.warning(self, "提示", "请先打开影像。")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "导入 GCP 坐标文件", "",
            "坐标文件 (*.txt *.csv *.json);;TXT 文件 (*.txt);;CSV 文件 (*.csv);;JSON 文件 (*.json);;所有文件 (*)"
        )
        if not path:
            return

        # 若是比赛顶点格式，走自动三点校准
        from roadnet.gcp_io import looks_like_corner_calibration_txt
        if looks_like_corner_calibration_txt(path):
            self._apply_corner_vertex_calibration_file(path, int(w), int(h))
            return

        from roadnet.gcp_io import load_gcp_file, validate_gcp_points
        cps = load_gcp_file(path, w, h)

        if not cps:
            QMessageBox.warning(self, "导入失败", "未能从文件中解析出有效控制点，请检查文件格式。")
            return

        valid, err_msg = validate_gcp_points(cps, w, h)
        if not valid:
            QMessageBox.warning(self, "控制点验证失败", err_msg)
            return

        # 更新 GeoCalibration（★ 设置标定元数据）
        self._geo_calibration.set_control_points(cps)
        self._geo_calibration.set_calibration_metadata("corner_manual", w, h)

        # ★ 同步标定方式下拉框
        if hasattr(self._param_panel, 'set_calibration_method_combo'):
            self._param_panel.set_calibration_method_combo("corner_manual")

        # 更新面板（勾选对应角点 + 填入经纬度）
        if hasattr(self._param_panel, 'set_calibration_corners_from_data'):
            self._param_panel.set_calibration_corners_from_data(cps, (w, h))

        # 通知画布更新角点标记
        self._update_calibration_corner_markers()

        corner_names = [cp["name"] for cp in cps]
        self._status_bar.show_message(
            f"成功导入 {len(cps)} 个顶点坐标: {', '.join(corner_names)}"
        )

    def _on_calibration_import_corners_json(self):
        """导入四角坐标 JSON 文件 (corners.json)。

        格式:
        {
          "image": "test_001.jpg",
          "width": 6000, "height": 6000,
          "crs": "EPSG:4326",
          "corners_wgs84": {
            "top_left":     [117.123456, 38.123456],
            "top_right":    [117.124456, 38.123456],
            "bottom_right": [117.124456, 38.122456],
            "bottom_left":  [117.123456, 38.122456]
          }
        }

        注意：corners_wgs84 中 [lon, lat]，不是 [lat, lon]。
        导入后自动填入 TL/TR/BR/BL 并勾选四个点，可直接计算整体变换。
        """
        original_w, original_h = self._layer_manager.original_size
        w, h = original_w, original_h
        if w <= 0 or h <= 0:
            QMessageBox.warning(self, "提示", "请先打开影像。")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "导入四角坐标 JSON", "",
            "JSON 文件 (*.json);;所有文件 (*)"
        )
        if not path:
            return

        try:
            from roadnet.gcp_io import load_gcp_json

            cps = load_gcp_json(path, w, h)
            if not cps:
                QMessageBox.warning(self, "导入失败",
                    "未能从 JSON 中解析出有效控制点。\n\n"
                    "请确认文件包含 'corners_wgs84' 字段，\n"
                    "或 'control_points' 数组。")
                return

            # ★ 设置控制点到 GeoCalibration
            self._geo_calibration.set_control_points(cps)
            self._geo_calibration.set_calibration_metadata("corner_file", w, h)

            # ★ 同步标定方式下拉框
            if hasattr(self._param_panel, 'set_calibration_method_combo'):
                self._param_panel.set_calibration_method_combo("corner_file")

            # ★ 自动勾选所有导入的角点 + 填入经纬度
            if hasattr(self._param_panel, 'set_calibration_corners_from_data'):
                self._param_panel.set_calibration_corners_from_data(cps, (w, h))

            # ★ 更新画布角点标记
            self._update_calibration_corner_markers()

            corner_names = [cp["name"] for cp in cps]
            lonlats = ", ".join(
                f"{cp['name']}=({cp['lon']:.6f}, {cp['lat']:.6f})"
                for cp in cps
            )
            self._status_bar.show_message(
                f"已导入 {len(cps)} 个四角坐标: {', '.join(corner_names)}"
            )
            print(f"[Calibration] 导入 corners.json: {len(cps)} 个控制点")
            for cp in cps:
                print(f"  {cp['name']}: pixel=({cp['pixel'][0]},{cp['pixel'][1]}), "
                      f"lon={cp['lon']:.6f}, lat={cp['lat']:.6f}")

            QMessageBox.information(
                self, "导入成功",
                f"已导入 {len(cps)} 个顶点坐标。\n\n"
                f"{lonlats}\n\n"
                f"请点击「计算整体变换」完成标定。"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "导入失败", f"解析 corners.json 失败:\n{e}")

    def _on_calibration_set_method(self, method: str):
        """设置标定方式（来自面板下拉框变化）。"""
        self._geo_calibration.method = method
        print(f"[Calibration] 标定方式: {method}")

    def _on_calibration_import_cp_file(self):
        """导入控制点文件（高级模式）。

        支持两种格式：
        1. 完整格式：点号,像素X,像素Y,经度,纬度 → 直接计算
        2. 仅坐标：点号,经度,纬度 → 进入图上配准模式
        """
        original_w, original_h = self._layer_manager.original_size
        w, h = original_w, original_h
        if w <= 0 or h <= 0:
            QMessageBox.warning(self, "提示", "请先打开影像。")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "导入控制点文件", "",
            "坐标文件 (*.txt *.csv);;TXT 文件 (*.txt);;CSV 文件 (*.csv);;所有文件 (*)"
        )
        if not path:
            return

        try:
            from roadnet.gcp_io import load_generic_gcp_file
            cps = load_generic_gcp_file(path)
            if not cps:
                QMessageBox.warning(self, "导入失败",
                    "未能从文件中解析出有效控制点。\n\n"
                    "支持的格式：\n"
                    "  • 点号,像素X,像素Y,经度,纬度\n"
                    "  • 点号,经度,纬度\n"
                    "  • CSV 带表头 (name,lon,lat)")
                return

            # ★ 关键：检查所有控制点是否有像素坐标
            has_all_pixels = all(
                cp.get("pixel") is not None
                and cp["pixel"][0] is not None
                and cp["pixel"][1] is not None
                for cp in cps
            )

            has_any_pixel = any(
                cp.get("pixel") is not None
                and cp["pixel"][0] is not None
                and cp["pixel"][1] is not None
                for cp in cps
            )

            self._geo_calibration.set_control_points(cps)
            self._geo_calibration.set_calibration_metadata("control_points_file", w, h)

            # 同步到面板
            self._param_panel.update_calibration_ui(self._geo_calibration)
            self._update_calibration_corner_markers()

            if has_all_pixels and len(cps) >= 3:
                # 有完整像素坐标 → 可以直接计算标定
                msg = (f"已导入 {len(cps)} 个控制点（含像素坐标）。\n\n"
                       f"可直接点击「计算坐标变换」完成标定。")
                QMessageBox.information(self, "导入成功", msg)
                self._status_bar.show_message(
                    f"已导入 {len(cps)} 个控制点（含像素坐标）")
            elif not has_any_pixel:
                # 仅有 lon/lat，无像素坐标 → 需要图上点击配准
                msg = (f"已导入 {len(cps)} 个控制点（仅经纬度，无像素坐标）。\n\n"
                       f"当前已进入「控制点图上配准」模式。\n"
                       f"请点击「开始图上点击配准」按钮，\n"
                       f"在图像上依次点击每个控制点的位置。")
                QMessageBox.information(self, "导入成功", msg)
                self._status_bar.show_message(
                    f"已导入 {len(cps)} 个控制点（仅经纬度），需要图上配准")
                # 自动进入配准模式
                self._start_map_click_registration(cps)
            else:
                # 部分有像素，部分没有
                missing = sum(1 for cp in cps
                              if not (cp.get("pixel") and cp["pixel"][0] is not None))
                msg = (f"已导入 {len(cps)} 个控制点，其中 {missing} 个缺少像素坐标。\n\n"
                       f"有像素坐标的点可直接计算，\n"
                       f"缺少像素坐标的需要图上配准补充。")
                QMessageBox.warning(self, "部分控制点不完整", msg)

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "导入失败", f"解析控制点文件失败:\n{e}")

    def _on_calibration_start_map_click(self):
        """开始图上点击配准（从 B 区按钮触发）。"""
        cps = self._geo_calibration.control_points
        if not cps:
            QMessageBox.warning(self, "提示",
                "请先导入控制点文件（含经纬度），再开始图上配准。")
            return
        # 检查是否已有像素坐标
        no_pixel = [cp for cp in cps
                    if not (cp.get("pixel") and cp["pixel"][0] is not None)]
        if not no_pixel:
            QMessageBox.information(self, "提示",
                "所有控制点已有像素坐标，无需再次配准。")
            return
        self._start_map_click_registration(cps)

    def _start_map_click_registration(self, cps: list):
        """启动图上点击配准模式。"""
        self._map_click_calibration_mode = True
        self._map_click_calibration_queue = list(cps)  # 待配准队列（全部）
        # 找到第一个无像素坐标的点
        self._map_click_calibration_index = 0
        for i, cp in enumerate(self._map_click_calibration_queue):
            px = cp.get("pixel")
            if px and px[0] is not None and px[1] is not None:
                self._map_click_calibration_index = i + 1
            else:
                self._map_click_calibration_index = i
                break

        total = len(cps)
        done = self._map_click_calibration_index

        # 更新 B 区状态
        if hasattr(self._param_panel, '_cal_map_click_status_label'):
            if done < total:
                next_cp = cps[done]
                name = next_cp.get("name", f"CP{done+1}")
                self._param_panel._cal_map_click_status_label.setText(
                    f"状态：配准中 ({done}/{total}) — 请在图上点击「{name}」")
            else:
                self._param_panel._cal_map_click_status_label.setText(
                    "状态：所有控制点已配准")

        self._geo_calibration.set_calibration_metadata("control_points_manual",
            self._layer_manager.original_size[0], self._layer_manager.original_size[1])
        self._param_panel.set_calibration_method_combo("control_points_manual")

        # 更新路口标记
        self._update_calibration_corner_markers()

        # ★ 设置 canvas 工具为 calibrate_map_click（mousePressEvent 会发 calibration_map_clicked 信号）
        if self._canvas:
            self._canvas.set_tool("calibrate_map_click")
            self._canvas.setCursor(Qt.CursorShape.CrossCursor)

        self._status_bar.show_message(
            f"图上配准模式：请在图像上点击各控制点位置（{done}/{total} 完成）")

    def _handle_map_click_calibration(self, px, py):
        """处理图上点击配准的单次点击。"""
        if not self._map_click_calibration_mode:
            return

        cps = self._map_click_calibration_queue
        idx = self._map_click_calibration_index

        if idx >= len(cps):
            self._exit_map_click_calibration()
            QMessageBox.information(self, "配准完成",
                "所有控制点已配准完毕。\n\n请点击「计算坐标变换」完成标定。")
            return

        cp = cps[idx]
        name = cp.get("name", f"CP{idx+1}")
        cp["pixel"] = [px, py]

        print(f"[Calibration] 图上配准: {name} → pixel=({px},{py}), "
              f"lon={cp.get('lon',0):.6f}, lat={cp.get('lat',0):.6f}")

        idx += 1
        self._map_click_calibration_index = idx

        # 同步到 GeoCalibration
        self._geo_calibration.set_control_points(cps)

        # 更新 B 区 UI
        self._param_panel.update_calibration_ui(self._geo_calibration)
        self._update_calibration_corner_markers()

        if idx < len(cps):
            next_cp = cps[idx]
            name = next_cp.get("name", f"CP{idx+1}")
            if hasattr(self._param_panel, '_cal_map_click_status_label'):
                self._param_panel._cal_map_click_status_label.setText(
                    f"状态：配准中 ({idx}/{len(cps)}) — 请在图上点击「{name}」")
            self._status_bar.show_message(
                f"图上配准: 已注册 {idx}/{len(cps)}，下一个: {name}")
        else:
            self._exit_map_click_calibration()
            QMessageBox.information(self, "配准完成",
                f"所有 {len(cps)} 个控制点已配准完毕。\n\n"
                f"请点击「计算坐标变换」完成标定。")

    def _exit_map_click_calibration(self):
        """退出图上点击配准模式。"""
        self._map_click_calibration_mode = False
        self._map_click_calibration_queue = []
        self._map_click_calibration_index = 0
        if hasattr(self._param_panel, '_cal_map_click_status_label'):
            self._param_panel._cal_map_click_status_label.setText("状态：就绪")
        if self._canvas:
            # ★ 恢复工具为 pan + 正常光标
            self._canvas.set_tool("pan")
            self._canvas.setCursor(Qt.CursorShape.ArrowCursor)
        self._status_bar.show_message("图上配准模式已退出")

    def _on_calibration_import_other(self):
        """导入 CSV / JSON 坐标文件（复用 TXT 逻辑）。"""
        self._on_calibration_import_txt()

    def _on_calibration_compute(self):
        """计算坐标变换。

        每次点击都从 UI 重新读取原始 lon/lat，重置旧状态后重新计算。
        确保重复计算稳定一致，不会因残留状态导致第二次失败。
        
        ★ 坐标校准绝不修改 final_graph 的 pixel 坐标。

        ★★ 控制点来源合并策略：
        1. 从四角点 widget 读取勾选的控制点（corner cps）
        2. 从 geo.control_points 读取非角点的控制点（B区导入/图上配准的 cps）
        3. 合并后统一进入 pixel ↔ ENU 拟合管道
        """
        geo = self._geo_calibration
        # 大图模式下使用原图尺寸进行校准
        w, h = self._layer_manager.original_size

        # ★ 记录计算前的 graph_hash（确保校准不修改 graph）
        hash_before = self._hash_graph_geometry()

        # ★ Step 0: 重置旧状态，确保每次计算从干净环境开始
        geo.reset_state()

        # ★ Step 1: 合并控制点来源
        # 来源 A: 四角点 widget（TL/TR/BR/BL）
        from roadnet.gcp_io import CORNERS_ORDER
        corner_cps = self._param_panel.get_calibration_control_points((w, h)) or []
        corner_names = set(cp.get("name", "") for cp in corner_cps)

        # 来源 B: geo.control_points 中的非角点控制点（B区导入/图上配准）
        existing_cps = geo.control_points or []
        non_corner_cps = [
            cp for cp in existing_cps
            if cp.get("name", "") not in CORNERS_ORDER
            and cp.get("pixel") is not None
            and cp["pixel"][0] is not None
            and cp["pixel"][1] is not None
        ]

        # 合并 → 四角点优先（corner 可覆盖 B 区同名点）
        # 反过来，B区也有的话四角点中不存在再加入
        merged_map = {}
        for cp in corner_cps:
            merged_map[cp.get("name", "")] = cp
        for cp in non_corner_cps:
            name = cp.get("name", "")
            if name and name not in merged_map:
                merged_map[name] = cp

        cp_data = list(merged_map.values())
        if len(cp_data) < 3:
            self._param_panel.update_calibration_ui(geo)  # 显示"未校准"
            QMessageBox.warning(self, "提示",
                f"至少需要 3 个控制点进行坐标校准，当前仅有 {len(cp_data)} 个。\n\n"
                f"请在四角点区勾选角点并填写经纬度，\n"
                f"或在 B 区导入控制点文件/图上点击配准。")
            return

        # ──── 调试：打印面板返回的控制点 ────
        print("[DEBUG][GCP] control points from panel (before compute):")
        all_zero = True
        for cp in cp_data:
            px = cp.get("pixel", [None, None])
            lon = cp.get("lon", float('nan'))
            lat = cp.get("lat", float('nan'))
            print(f"  name={cp.get('name', '?')}, pixel=({px[0]},{px[1]}), "
                  f"lon={lon:.8f}, lat={lat:.8f}")
            if abs(lon) > 0.0001 or abs(lat) > 0.0001:
                all_zero = False

        # ──── 经纬度全零检查 ────
        if all_zero:
            self._param_panel.update_calibration_ui(geo)
            QMessageBox.warning(self, "数据错误",
                "所有控制点经纬度均为 0.0，请先在右侧面板输入经纬度值，"
                "或通过「导入坐标 TXT」文件录入坐标。")
            return

        # ──── lon/lat 填反检测 ────
        swap_warning = self._param_panel.detect_lon_lat_swap(cp_data)
        if swap_warning:
            reply = QMessageBox.warning(
                self, "经纬度填反警告",
                f"{swap_warning}\n\n如果确实是填反了，请返回修改后重新计算。\n"
                f"是否忽略此警告，继续计算？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # ──── 像素坐标检查 ────
        for cp in cp_data:
            px = cp.get("pixel", [None, None])
            if px[0] is None or px[1] is None:
                self._param_panel.update_calibration_ui(geo)
                QMessageBox.warning(self, "数据错误",
                    f"控制点 {cp.get('name', '?')} 像素坐标未设置。")
                return

        # ★ Step 2: 设置控制点（自动清除旧 x_meter/y_meter）
        geo.set_control_points(cp_data)
        print(f"[DEBUG][GCP] geo lon0={geo.lon0:.8f}, lat0={geo.lat0:.8f}, "
              f"points={len(geo.control_points)}")

        # 验证基本格式和范围
        valid, err_msg = geo.validate_control_points()
        if not valid:
            self._param_panel.update_calibration_ui(geo)  # 显示"未校准"
            QMessageBox.warning(self, "控制点验证失败", err_msg)
            return

        # ★ Step 3: 设置投影（计算每个控制点的 x_meter/y_meter）
        if not geo.setup_projection():
            self._param_panel.update_calibration_ui(geo)
            QMessageBox.warning(self, "投影设置失败", "无法初始化坐标投影。")
            return

        # ★ Step 4: 计算仿射变换（带 RMS 残差）
        try:
            rms_error = None
            if not geo.compute_affine():
                # ★ 失败时重置状态，确保 UI 显示"未校准"
                geo.reset_state()
                self._param_panel.update_calibration_ui(geo)
                QMessageBox.warning(self, "仿射变换计算失败", "请检查控制点数据。")
                return
            # 计算 RMS 残差
            rms_error = self._compute_calibration_rms(geo)
            geo.rms_error = rms_error
        except ValueError as e:
            geo.reset_state()
            self._param_panel.update_calibration_ui(geo)
            QMessageBox.warning(self, "控制点错误", str(e))
            return

        # 检查分辨率
        res_ok, res_msg = geo.check_resolution()
        if not res_ok:
            reply = QMessageBox.question(
                self, "分辨率警告",
                f"{res_msg}\n\n是否继续使用当前校准结果？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                geo.reset_state()
                self._param_panel.update_calibration_ui(geo)
                return

        # ★ 成功：设置标定元数据
        # 从 UI 当前选中的模式获取 method
        if hasattr(self._param_panel, 'get_selected_calibration_method'):
            geo.method = self._param_panel.get_selected_calibration_method()
        elif not geo.method:
            # 根据控制点来源自动推断
            has_corners = any(cp.get("name", "") in CORNERS_ORDER for cp in cp_data)
            geo.method = "corner_manual" if has_corners else "control_points_file"
        geo.image_width = w
        geo.image_height = h

        # ★ 成功：更新 UI 为"已校准"
        self._param_panel.update_calibration_ui(geo)
        self._update_calibration_corner_markers()
        self._status_bar.update_resolution(
            geo.pixel_resolution_estimated_m or self._project_manager.data.pixel_resolution_m,
            calibrated=True
        )
        rms_str = f", RMS={rms_error:.3f}m" if rms_error is not None else ""
        self._status_bar.show_message(
            f"坐标变换计算成功: {geo.transform_mode}, "
            f"分辨率 {geo.pixel_resolution_estimated_m:.4f} m/px, "
            f"{len(cp_data)} 个控制点{rms_str}"
        )
        self.mark_stage_done("calibrate")

        # ★ 验证校准计算未修改 graph 数据
        hash_after = self._hash_graph_geometry()
        if hash_before != hash_after:
            print(
                f"[BUG] final_graph changed during calibration compute: "
                f"hash_before={hash_before}, hash_after={hash_after}"
            )

    @staticmethod
    def _compute_calibration_rms(geo) -> float:
        """计算校准 RMS 残差（米）。"""
        import numpy as np
        if geo.pixel_to_world_matrix is None or not geo.control_points:
            return 0.0
        errors = []
        for cp in geo.control_points:
            u, v = cp["pixel"]
            pred_x, pred_y = geo.pixel_to_world(u, v)
            actual_x = cp.get("x_meter", 0)
            actual_y = cp.get("y_meter", 0)
            err = np.sqrt((pred_x - actual_x) ** 2 + (pred_y - actual_y) ** 2)
            errors.append(err)
        return float(np.sqrt(np.mean(np.array(errors) ** 2)))

    def _on_calibration_apply_graph(self):
        """将校准应用到 final_graph（生成地理版本文件，不修改 pixel 坐标）。

        ★ 关键原则：
        1. apply_to_graph 是只读操作，创建新字典，绝不修改 graph_editor
        2. final_graph 永远保存 pixel 坐标（image_pixel 坐标系）
        3. 地理版本另存为 final_graph_geo.json
        4. transform 信息保存在 calibration.json 中
        5. 应用后保持 geo_calibration.is_valid = True
        """
        geo = self._geo_calibration
        if not geo.is_valid:
            QMessageBox.warning(self, "提示", "请先完成坐标校准计算。")
            return
        if self._graph_editor is None or not self._graph_editor.nodes:
            QMessageBox.warning(self, "提示", "没有可校准的路网数据。")
            return

        # ★ 记录校准前的 graph_hash
        hash_before = self._hash_graph_geometry()

        try:
            # ★ apply_to_graph 只读取 node.x/node.y 和 edge.points_pixel，
            #   通过 transform 计算 lon/lat/meter 坐标，返回新的字典列表。
            #   绝不修改 graph_editor._nodes 或 graph_editor._edges。
            calibrated = geo.apply_to_graph(self._graph_editor)

            # 更新状态栏统计
            total_len_m = sum(e.get("length_meter", 0) for e in calibrated["edges"])
            total_len_px = sum(e.get("length_pixel", 0) for e in calibrated["edges"])
            self._status_bar.show_message(
                f"校准已应用: {len(calibrated['nodes'])} 节点, "
                f"{len(calibrated['edges'])} 边, "
                f"总长度 {total_len_px:.0f} px / {total_len_m:.1f} m"
            )

            # 更新右侧统计
            stats = self._graph_editor.get_stats()
            stats["total_length_m_calibrated"] = round(total_len_m, 1)
            stats["calibrated"] = True
            self._param_panel.update_graph_stats(stats)

            # 保存校准后路网文件（保存为 final_graph_geo.json，不覆盖 final_graph.json）
            outputs_dir = os.path.join(os.getcwd(), "outputs")
            geo.save_calibrated_graph(
                self._graph_editor,
                outputs_dir,
                image_rgb=self._layer_manager.image_rgb,
                image_size=self._layer_manager.image_size
            )

            # ★ 验证 graph 未被修改
            hash_after = self._hash_graph_geometry()
            if hash_before != hash_after:
                print(
                    f"[BUG] final_graph changed during calibration apply: "
                    f"hash_before={hash_before}, hash_after={hash_after}"
                )
            else:
                print(f"[OK] final_graph intact after calibration apply, hash={hash_before}")

            QMessageBox.information(self, "校准完成",
                f"已保存校准路网到 outputs/\n"
                f"  - final_graph_geo.json  (地理版本，含 lon/lat/meter)\n"
                f"  - final_nodes_geo.csv\n"
                f"  - final_edges_geo.csv\n"
                f"  - calibration.json      (坐标变换参数)\n\n"
                f"总长度: {total_len_m:.1f} m\n\n"
                f"注意: final_graph.json 仍保留 pixel 坐标，未修改。"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "校准失败", f"应用校准时出错:\n{e}")

    def _on_calibration_save(self):
        """保存校准配置。"""
        geo = self._geo_calibration
        if not geo.is_valid:
            QMessageBox.warning(self, "提示", "请先完成坐标校准计算。")
            return
        saved = []
        if self._large_image_project is not None:
            proj_cal = os.path.join(
                self._large_image_project.project_dir, "calibration.json"
            )
            if geo.save(proj_cal):
                saved.append(proj_cal)
                try:
                    self._large_image_project.geo_calibration_path = proj_cal
                    self._large_image_project.save()
                except Exception:
                    pass
        outputs_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        out_cal = os.path.join(outputs_dir, "calibration.json")
        if geo.save(out_cal):
            saved.append(out_cal)
        if saved:
            self._status_bar.show_message(f"校准配置已保存：{saved[0]}")
        else:
            QMessageBox.warning(self, "保存失败", "无法写入 calibration.json")

    def _on_calibration_clear(self):
        """清空校准数据。"""
        self._geo_calibration = GeoCalibration(mode="auto")
        self._param_panel.update_calibration_ui(self._geo_calibration)
        if hasattr(self._param_panel, "set_vertex_calibration_summary"):
            self._param_panel.set_vertex_calibration_summary("")
        self._clear_calibration_corner_markers()
        self._status_bar.update_resolution(
            self._project_manager.data.pixel_resolution_m, calibrated=False
        )
        self._status_bar.show_message("校准数据已清空")

    def _on_calibration_update_settings(self):
        """当用户修改控制点经纬度时，更新 GeoCalibration 对象。"""
        # 大图模式下使用原图尺寸进行校准
        w, h = self._layer_manager.original_size
        cp_data = self._param_panel.get_calibration_control_points((w, h))
        if cp_data is not None:
            self._geo_calibration.set_control_points(cp_data)
        # 同步画布上的角点标记（实心/空心）
        self._update_calibration_corner_markers()

    def _update_calibration_status(self):
        """更新校准面板状态。"""
        # 大图模式下使用原图尺寸进行校准
        w, h = self._layer_manager.original_size
        self._param_panel.update_calibration_ui(self._geo_calibration)
        if hasattr(self._param_panel, '_sync_corner_widgets_from_geo'):
            self._param_panel._sync_corner_widgets_from_geo(self._geo_calibration)
        self._update_calibration_corner_markers()

    # 角点标记项缓存
    _cal_corner_marker_items: list = []

    def _update_calibration_corner_markers(self):
        """在画布上显示四角标记 (TL/TR/BL/BR)。"""
        self._clear_calibration_corner_markers()

        scene = self._canvas.scene()
        # 使用原图尺寸计算角点位置，再转换为预览图坐标显示
        original_w, original_h = self._layer_manager.original_size
        preview_w, preview_h = self._layer_manager.image_size
        if original_w <= 0 or original_h <= 0:
            return

        # 角点颜色和标签
        CORNER_CONFIG = {
            "top_left":     {"label": "TL", "color": QColor(255, 85, 85)},    # 红色
            "top_right":    {"label": "TR", "color": QColor(85, 136, 255)},    # 蓝色
            "bottom_left":  {"label": "BL", "color": QColor(255, 204, 0)},     # 黄色
            "bottom_right": {"label": "BR", "color": QColor(204, 85, 255)},    # 紫色
        }

        # 检查哪些角点已输入经纬度
        has_data = set()
        if self._geo_calibration.control_points:
            for cp in self._geo_calibration.control_points:
                name = cp.get("name", "")
                lon = cp.get("lon", 0)
                lat = cp.get("lat", 0)
                if name and not (lon == 0 and lat == 0):
                    has_data.add(name)

        from roadnet.gcp_io import infer_pixel_from_corner_name
        for cname, cfg in CORNER_CONFIG.items():
            # 计算原图尺寸下的角点位置
            px = infer_pixel_from_corner_name(cname, original_w, original_h)
            if px is None:
                continue
            # 转换为预览图坐标进行显示
            u_original, v_original = px
            u, v = self._layer_manager.global_to_preview(u_original, v_original)
            color = cfg["color"]
            label = cfg["label"]

            in_use = cname in has_data

            # 圆圈标记
            radius = 10
            circle = QGraphicsEllipseItem(u - radius, v - radius, radius * 2, radius * 2)
            circle.setPen(QPen(color, 2 if in_use else 1))
            if in_use:
                circle.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 100)))
            else:
                circle.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            circle.setZValue(self._canvas.ZVAL_SELECTED + 10)
            circle.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            scene.addItem(circle)
            self._cal_corner_marker_items.append(circle)

            # 文字标签
            from PySide6.QtWidgets import QGraphicsTextItem
            text_item = QGraphicsTextItem(label)
            text_item.setDefaultTextColor(color if in_use else QColor(120, 120, 120))
            font = QFont("Arial", 10, QFont.Weight.Bold)
            text_item.setFont(font)
            # 根据位置调整文字偏移
            offset_x = -20 if cname in ("top_right", "bottom_right") else 14
            offset_y = -20 if cname in ("top_left", "top_right") else 8
            text_item.setPos(u + offset_x, v + offset_y)
            text_item.setZValue(self._canvas.ZVAL_SELECTED + 11)
            text_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            scene.addItem(text_item)
            self._cal_corner_marker_items.append(text_item)

    def _clear_calibration_corner_markers(self):
        """清除角点标记。"""
        scene = self._canvas.scene()
        for item in self._cal_corner_marker_items:
            safe_remove_scene_item(scene, item)
        self._cal_corner_marker_items.clear()

    # ===================================================================
    # 路网导出（展示型 vs 调试型）
    # ===================================================================

    def export_display_image(self, output_path: str = None) -> bool:
        """
        展示型导出：渲染当前场景所见即所得的图像。

        保存当前界面显示的所有内容：
        - 当前可见图层（Final Graph / Draft Graph 等）
        - 当前配色（黄色边、绿色节点等）
        - 不显示节点编号
        - 保持简洁模式/调试模式的显示状态

        Returns:
            bool: 导出是否成功
        """
        try:
            from PySide6.QtGui import QImage, QPainter
            from PySide6.QtCore import QRectF

            scene = self._canvas.scene()
            if scene is None:
                print("[ERROR][Export] scene is None")
                return False

            # 获取图像尺寸
            if self._layer_manager.has_image():
                w, h = self._layer_manager.image_size
            else:
                rect = scene.sceneRect().toRect()
                w, h = rect.width(), rect.height()

            if w <= 0 or h <= 0:
                print("[ERROR][Export] invalid image size")
                return False

            # 创建图像（RGB32 格式，RGB 排序）
            image = QImage(w, h, QImage.Format.Format_RGB32)
            image.fill(0)  # 黑色填充（无图像区域）

            # 渲染场景到图像
            painter = QPainter(image)
            try:
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                # 渲染场景内容（不包括 UI 控件）
                scene.render(painter, QRectF(0, 0, w, h), QRectF(0, 0, w, h))
            finally:
                painter.end()

            # 保存图像
            if output_path is None:
                output_dir = os.path.join(os.getcwd(), "outputs")
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, "final_graph_overlay.png")

            # 转换为 BGR 并保存（OpenCV 使用 BGR）
            img_rgb = np.array(image)
            if img_rgb.size > 0:
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, img_bgr)
                print(f"[Export] 展示图已保存: {output_path}")
                return True
            else:
                print("[ERROR][Export] failed to convert QImage to numpy array")
                return False

        except Exception as e:
            import traceback
            print(f"[ERROR][Export] 展示型导出失败: {e}")
            traceback.print_exc()
            return False

    def export_debug_graph_image(self, output_path: str = None) -> bool:
        """
        调试型导出：使用 OpenCV 绘制调试风格图像。

        坐标处理：
        - 大图模式下，graph 坐标是全局像素，需要按 preview_scale 缩放后才能画到预览图上
        - 普通模式，preview_scale=1.0，无需转换

        Returns:
            bool: 导出是否成功
        """
        try:
            if self._graph_editor is None:
                print("[ERROR][Export] graph_editor is None")
                return False

            if output_path is None:
                output_dir = os.path.join(os.getcwd(), "outputs")
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, "final_graph_debug.png")

            # ★ 使用显示图像（大图模式下是预览图）
            image_rgb = self._layer_manager.display_image_rgb
            if image_rgb is None:
                print("[ERROR][Export] display_image_rgb is None")
                return False

            # ★ 在显示图像上用缩放的坐标绘制
            preview_scale = self._layer_manager.preview_scale
            img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            self._graph_editor._draw_overlay(img, scale=preview_scale)
            cv2.imwrite(output_path, img)
            print(f"[Export] 调试图已保存: {output_path} (scale={preview_scale:.4f})")
            return True

        except Exception as e:
            import traceback
            print(f"[ERROR][Export] 调试型导出失败: {e}")
            traceback.print_exc()
            return False

    # ===================================================================
    # 关于
    # ===================================================================

    def _on_about(self):
        QMessageBox.about(
            self, "关于 RoadNet Studio",
            "<h2>RoadNet Studio v2.0</h2>"
            "<p>无人车比赛 — 半自动路网生成与编辑工具</p>"
            "<p>基于色彩空间分割 + Skeleton 骨架化 + Qt Graph 编辑</p>"
            "<hr>"
            "<p>Powered by PySide6, OpenCV, scikit-image</p>"
        )

    # ===================================================================
    # 关闭事件
    # ===================================================================

    def closeEvent(self, event):
        background_running = (
            (self._segmentation_thread is not None and self._segmentation_thread.isRunning())
            or (self._pipeline_thread is not None and self._pipeline_thread.isRunning())
        )
        if background_running:
            self._cancel_segmentation()
            self._cancel_pipeline()
            QMessageBox.information(
                self, "后台任务正在取消",
                "已请求取消后台任务。请等待取消完成后再关闭窗口。"
            )
            event.ignore()
            return
        if self._project_manager.is_dirty and self._layer_manager.has_image():
            reply = QMessageBox.question(
                self, "未保存的更改",
                "项目有未保存的更改，是否保存后再退出？",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Save:
                self._on_save_project()
                event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
