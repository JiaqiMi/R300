"""
右侧参数面板：按阶段显示简洁参数，高级设置可折叠。
"""
from __future__ import annotations

from typing import Dict, Any, Optional

from PySide6.QtCore import Qt, Signal
try:
    from shiboken6 import isValid as _shiboken_is_valid
except ImportError:
    def _shiboken_is_valid(obj) -> bool:
        return True
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QLineEdit,
    QPushButton, QScrollArea, QSizePolicy, QFrame, QProgressBar,
)

from roadnet.utils import load_config
from roadnet.gcp_io import CORNERS_ORDER, CORNERS_DEF, infer_pixel_from_corner_name


class ParameterPanel(QWidget):
    """右侧参数面板 — 简单模式 + 高级设置折叠"""

    param_changed = Signal(str, object)
    apply_requested = Signal(str)

    def __init__(self, config: Optional[Dict] = None, parent=None):
        super().__init__(parent)
        self._config = config or load_config()
        self._widgets: Dict[str, QWidget] = {}
        self._advanced_widgets: list[QWidget] = []
        self._is_advanced_open = False
        self._mask_candidate_apply_allowed = True
        self._mask_candidate_apply_reason = ""

        # 提前初始化阶段统计标签（避免切换阶段时属性缺失）
        self._graph_stat_labels: Dict[str, QLabel] = {}
        self._skel_stat_labels: Dict[str, QLabel] = {}

        self._setup_ui()
        # 默认显示导入阶段
        self.set_stage("import")

    # ===================================================================
    # UI 构建
    # ===================================================================

    def _setup_ui(self):
        self.setMinimumWidth(240)
        self.setMaximumWidth(320)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # 标题
        title = QLabel("📋 参数设置")
        title.setStyleSheet("font-weight: bold; color: #89b4fa; font-size: 14px; padding: 4px 0;")
        main_layout.addWidget(title)

        # ★ 全局撤销/重做按钮（始终可见）
        undo_layout = QHBoxLayout()
        self._global_undo_btn = QPushButton("↩ 撤销 (Ctrl+Z)")
        self._global_undo_btn.setToolTip("全局撤销上一步操作")
        self._global_undo_btn.setStyleSheet("font-size: 10px; padding: 3px 6px;")
        self._global_undo_btn.clicked.connect(lambda: self.apply_requested.emit("undo_graph"))
        undo_layout.addWidget(self._global_undo_btn)

        self._global_redo_btn = QPushButton("↪ 重做 (Ctrl+Y)")
        self._global_redo_btn.setToolTip("全局重做撤销的操作")
        self._global_redo_btn.setStyleSheet("font-size: 10px; padding: 3px 6px;")
        self._global_redo_btn.clicked.connect(lambda: self.apply_requested.emit("redo_graph"))
        undo_layout.addWidget(self._global_redo_btn)
        main_layout.addLayout(undo_layout)

        # 滚动区域
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._page = QWidget()
        self._page_layout = QVBoxLayout(self._page)
        self._page_layout.setContentsMargins(0, 0, 0, 0)
        self._page_layout.setSpacing(6)

        self._scroll.setWidget(self._page)
        main_layout.addWidget(self._scroll)

    def _clear_page(self):
        """清空页面"""
        while self._page_layout.count():
            item = self._page_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._widgets.clear()
        self._advanced_widgets.clear()
        self._is_advanced_open = False
        # 重置阶段统计标签（旧 QLabel 已被 deleteLater）
        self._graph_stat_labels = {}
        self._skel_stat_labels = {}
        # 清除校准角点控件引用
        if hasattr(self, '_cal_corner_widgets'):
            self._cal_corner_widgets.clear()

    # ===================================================================
    # 阶段切换
    # ===================================================================

    def set_stage(self, stage: str):
        """按阶段显示对应参数"""
        self._clear_page()

        if stage == "import":
            self._build_import_page()
        elif stage == "segment":
            self._build_segment_page()
        elif stage == "edit":
            self._build_edit_page()
        elif stage == "skeleton":
            self._build_skeleton_page()
        elif stage == "graph":
            self._build_graph_page()
        elif stage == "calibrate":
            self._build_calibration_page()
        elif stage == "export":
            self._build_export_page()

        self._page_layout.addStretch()

    # ===================================================================
    # 各阶段页面
    # ===================================================================

    def _build_import_page(self):
        g = self._add_group("项目操作")
        btn_open = QPushButton("📂 打开影像")
        btn_open.setObjectName("primary")
        btn_open.clicked.connect(lambda: self.apply_requested.emit("open"))
        g.addWidget(btn_open)

        btn_new = QPushButton("📋 新建项目")
        btn_new.clicked.connect(lambda: self.apply_requested.emit("new_project"))
        g.addWidget(btn_new)

        btn_save = QPushButton("💾 保存项目")
        btn_save.clicked.connect(lambda: self.apply_requested.emit("save_project"))
        g.addWidget(btn_save)

    def _build_segment_page(self):
        # --- 样本信息 ---
        g1 = self._add_group("样本信息")
        self._pos_count_label = QLabel("正样本：0")
        self._pos_count_label.setStyleSheet("color: #50fa7b;")
        g1.addWidget(self._pos_count_label)

        self._neg_count_label = QLabel("负样本：0")
        self._neg_count_label.setStyleSheet("color: #ff5555;")
        g1.addWidget(self._neg_count_label)

        # --- ROI 状态与管理 ---
        g_roi = self._add_group("ROI 状态")
        self._seg_roi_count_label = QLabel("ROI 数量：0")
        self._seg_roi_count_label.setStyleSheet("color: #89b4fa;")
        g_roi.addWidget(self._seg_roi_count_label)

        self._seg_roi_tile_label = QLabel("覆盖 tile：-")
        self._seg_roi_tile_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        g_roi.addWidget(self._seg_roi_tile_label)

        self._seg_roi_infer_label = QLabel("待推理 tile：-")
        self._seg_roi_infer_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        g_roi.addWidget(self._seg_roi_infer_label)

        btn_draw_roi = QPushButton("✏ 绘制 ROI")
        btn_draw_roi.clicked.connect(lambda: self.apply_requested.emit("roi_draw"))
        g_roi.addWidget(btn_draw_roi)

        btn_view_roi = QPushButton("📐 使用当前视野作为 ROI")
        btn_view_roi.setToolTip("将当前画布可见区域转为矩形 ROI（比赛现场最快方式）")
        btn_view_roi.clicked.connect(lambda: self.apply_requested.emit("roi_use_view"))
        g_roi.addWidget(btn_view_roi)

        btn_clear_roi = QPushButton("🗑 清空 ROI")
        btn_clear_roi.clicked.connect(lambda: self.apply_requested.emit("roi_clear"))
        g_roi.addWidget(btn_clear_roi)

        btn_show_tiles = QPushButton("🔲 查看 ROI 覆盖 tile")
        btn_show_tiles.setToolTip("在画布上显示 ROI 覆盖的 tile 边框及缓存/待推理分类")
        btn_show_tiles.clicked.connect(lambda: self.apply_requested.emit("roi_show_tiles"))
        g_roi.addWidget(btn_show_tiles)

        # ── 大图道路提取模式（默认 OpenCV，SAMRoadPlus 为高级选项）──
        g_mode = self._add_group("大图道路提取模式")

        self._segment_running = False
        self._preview_seg_running = False
        self._formal_extraction_running = False
        self._competition_fast_running = False
        self._lowres_formal_running = False

        # 1. 快速预览提取（仅预览）
        self._preview_segment_btn = QPushButton("⚡ 快速预览提取（OpenCV，仅预览）")
        self._preview_segment_btn.setObjectName("primary")
        self._preview_segment_btn.setToolTip(
            "只处理 preview.png / 缩略图，几秒内返回。\n"
            "preview_only：仅用于显示，不能生成正式骨架。"
        )
        self._preview_segment_btn.clicked.connect(
            lambda: self.apply_requested.emit("segment_preview")
        )
        g_mode.addWidget(self._preview_segment_btn)

        # 2. ★ 推荐：低像素快速生成正式 Mask
        self._lowres_formal_btn = QPushButton("🏆 低像素快速生成正式 Mask")
        self._lowres_formal_btn.setObjectName("primary")
        self._lowres_formal_btn.setStyleSheet(
            "QPushButton { background-color: #e6a700; color: #111; "
            "font-weight: bold; padding: 10px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #ffc928; }"
        )
        self._lowres_formal_btn.setToolTip(
            "比赛推荐：在低分辨率图上快速生成道路 mask，\n"
            "再用 INTER_NEAREST 放大为正式 working_road_mask。\n"
            "不直接生成 graph；后续可区域修正 → 骨架 → graph。\n"
            "目标耗时约 30～90 秒（max_side=2500）。"
        )
        self._lowres_formal_btn.clicked.connect(
            lambda: self.apply_requested.emit("lowres_formal_mask")
        )
        g_mode.addWidget(self._lowres_formal_btn)

        self._lowres_formal_status = QLabel(
            "当前 Mask 来源：-\n"
            "正式 Mask：否\n"
            "工作分辨率：-\n"
            "原图分辨率：-\n"
            "scale_x / scale_y：-"
        )
        self._lowres_formal_status.setWordWrap(True)
        self._lowres_formal_status.setStyleSheet(
            "color: #cdd6f4; font-size: 11px; padding: 6px; "
            "background: #313244; border-radius: 4px;"
        )
        g_mode.addWidget(self._lowres_formal_status)

        # 3. ROI 正式提取（赛前）
        self._roi_extract_btn = QPushButton("🎯 大图精细提取（赛前，不推荐比赛现场）")
        self._roi_extract_btn.setObjectName("primary")
        self._roi_extract_btn.setToolTip(
            "ROI / 全图 OpenCV 正式 tile 提取，耗时长（可达数分钟）。\n"
            "仅建议赛前准备，比赛现场请用「低像素快速生成正式 Mask」。"
        )
        self._roi_extract_btn.clicked.connect(
            lambda: self.apply_requested.emit("extract_roi_opencv")
        )
        g_mode.addWidget(self._roi_extract_btn)

        # 4. 全图正式提取（赛前）
        self._full_extract_btn = QPushButton("🗺 全图正式提取（OpenCV，赛前）")
        self._full_extract_btn.setObjectName("primary")
        self._full_extract_btn.setToolTip(
            "处理全部 valid tile，使用 OpenCV 样本/颜色分割生成正式 "
            "global_road_mask.png，适合赛前批量处理。"
        )
        self._full_extract_btn.clicked.connect(
            lambda: self.apply_requested.emit("extract_full_opencv")
        )
        g_mode.addWidget(self._full_extract_btn)

        # 5. 应急手绘
        self._emergency_draw_btn = QPushButton("⚠ 应急手绘中心线（可能扣分）")
        self._emergency_draw_btn.setToolTip(
            "应急手绘模式，可能扣分。\n"
            "仅当自动路网失败且时间不够时使用；\n"
            "切换到手动画边工具进行少量补线。"
        )
        self._emergency_draw_btn.clicked.connect(
            lambda: self.apply_requested.emit("emergency_hand_draw_graph")
        )
        g_mode.addWidget(self._emergency_draw_btn)

        # 旧：比赛快速路网（直接 graph，效果较差，保留为高级选项）
        self._competition_fast_btn = QPushButton("大图比赛快速路网生成（直接 graph，不推荐）")
        self._competition_fast_btn.setToolTip(
            "旧方案：低分辨率直接生成 graph，路口/支路不稳定。\n"
            "比赛推荐改用「低像素快速生成正式 Mask」。"
        )
        self._competition_fast_btn.clicked.connect(
            lambda: self.apply_requested.emit("competition_fast_roadnet")
        )
        g_mode.addWidget(self._competition_fast_btn)

        self._competition_status = QLabel(
            "（旧）Competition Fast Roadnet：直接生成 graph，一般不推荐。"
        )
        self._competition_status.setWordWrap(True)
        self._competition_status.setStyleSheet(
            "color: #a6adc8; font-size: 10px; padding: 4px;"
        )
        g_mode.addWidget(self._competition_status)

        self._add_double(
            g_mode, "低像素 Mask 最长边", "segment.lowres_formal_max_side",
            2500.0, 1000.0, 4096.0, 100.0, 0,
        )
        self._add_double(
            g_mode, "比赛工作图最长边（旧）", "segment.competition_preview_max_side",
            1500.0, 1000.0, 4096.0, 100.0, 0,
        )
        self._add_check(
            g_mode, "启用比赛快速路网模式标记", "segment.competition_fast_mode", False
        )

        # 4/5. SAMRoadPlus 模型提取（高级选项，非比赛默认流程）
        g_model = self._add_group("模型提取（SAMRoadPlus，高级）")
        self._roi_model_btn = QPushButton("🧠 ROI 模型提取（SAMRoadPlus）")
        self._roi_model_btn.setToolTip(
            "高级选项：使用 SAMRoadPlus 模型对 ROI tile 推理。\n"
            "当前影像 OpenCV 效果更稳定，模型提取仅在需要时使用。"
        )
        self._roi_model_btn.clicked.connect(
            lambda: self.apply_requested.emit("extract_roi")
        )
        g_model.addWidget(self._roi_model_btn)

        self._full_model_btn = QPushButton("🧠 全图模型提取（SAMRoadPlus）")
        self._full_model_btn.setToolTip(
            "高级选项：使用 SAMRoadPlus 模型对全图 tile 推理。"
        )
        self._full_model_btn.clicked.connect(
            lambda: self.apply_requested.emit("extract_full")
        )
        g_model.addWidget(self._full_model_btn)

        # ── 兼容旧版分割按钮（折叠到高级设置）──
        g_legacy = self._add_group("传统色彩分割")
        self._segment_btn = QPushButton("🚀 传统分割（tile，后台）")
        self._segment_btn.setObjectName("primary")
        self._segment_btn.setToolTip("使用色彩分割 tile 分块，生成 road_mask（传统方式）")
        self._segment_btn.clicked.connect(self._on_segment_button_clicked)
        g_legacy.addWidget(self._segment_btn)

        # ── 进度条（共用）──
        self._segment_progress = QProgressBar()
        self._segment_progress.setRange(0, 100)
        self._segment_progress.setValue(0)
        self._segment_progress.setTextVisible(True)
        self._segment_progress.hide()
        g_legacy.addWidget(self._segment_progress)

        self._segment_progress_label = QLabel("")
        self._segment_progress_label.setWordWrap(True)
        self._segment_progress_label.setStyleSheet("color: #a0a0a0; font-size: 10px;")
        self._segment_progress_label.hide()
        g_legacy.addWidget(self._segment_progress_label)

        # 比赛快速模式
        self._add_check(g_legacy, "比赛快速预览缩略", "segment.competition_preview_shrink", False)

        self._clear_samples_btn = QPushButton("🗑 清空样本")
        self._clear_samples_btn.clicked.connect(
            lambda: self.apply_requested.emit("clear_samples")
        )
        g_legacy.addWidget(self._clear_samples_btn)

        # --- 高级设置（可折叠） ---
        self._add_advanced_toggle()
        self._start_advanced()

        # ── 大图 OpenCV 正式提取参数（比赛默认流程）──
        ag_ocv = self._add_group("大图 OpenCV 正式提取参数")
        self._add_combo(ag_ocv, "颜色空间", "opencv_extraction.color_space",
                        ["Lab+HSV", "hsv", "lab"], "Lab+HSV")
        self._add_spin(ag_ocv, "模糊核", "opencv_extraction.blur_kernel", 3, 0, 31)
        self._add_spin(ag_ocv, "开运算核", "opencv_extraction.open_kernel", 3, 0, 31)
        self._add_spin(ag_ocv, "闭运算核", "opencv_extraction.close_kernel", 5, 0, 31)
        self._add_spin(ag_ocv, "最小面积", "opencv_extraction.min_area", 100, 0, 100000)
        self._add_check(ag_ocv, "填充孔洞（大图默认关闭）",
                        "opencv_extraction.fill_holes", False)
        self._add_check(ag_ocv, "使用 ROI", "opencv_extraction.use_roi", True)
        self._add_check(ag_ocv, "使用 Ignore", "opencv_extraction.use_ignore", True)
        self._add_check(ag_ocv, "Debug 模式", "opencv_extraction.debug_mode", False)
        self._add_spin(ag_ocv, "Tile 大小", "opencv_extraction.tile_size", 1024, 256, 4096)
        self._add_spin(ag_ocv, "Tile 重叠", "opencv_extraction.overlap", 64, 0, 1023)

        # ── 正式提取参数 ──
        ag_formal = self._add_group("模型提取参数（SAMRoadPlus / SAM-Road）")
        self._add_combo(
            ag_formal, "推理后端", "extraction.inference_backend",
            ["samroadplus_portable", "old_samroad"], "samroadplus_portable",
        )
        self._add_line(
            ag_formal, "Portable 目录", "extraction.portable_project_dir",
            r"D:\samroadplus_portable_infer",
        )
        self._add_spin(ag_formal, "Tile 大小", "extraction.tile_size", 2048, 256, 4096)
        self._add_spin(ag_formal, "Tile 重叠", "extraction.tile_overlap", 128, 0, 1023)
        self._add_combo(ag_formal, "推理方式", "extraction.infer_mode",
                        ["persistent_worker", "subprocess_per_tile"], "persistent_worker")
        self._add_spin(ag_formal, "批量大小", "extraction.tile_batch_size", 1, 1, 4)
        self._add_combo(ag_formal, "合并方式", "extraction.merge_method",
                        ["max", "average"], "max")
        self._add_check(ag_formal, "断点续跑", "extraction.resume_from_existing_tiles", True)
        self._add_check(ag_formal, "Debug 模式", "extraction.debug_mode", False)
        self._add_spin(ag_formal, "最大 tile 数", "extraction.max_tiles", 0, 0, 99999)
        self._add_check(ag_formal, "跳过黑色 tile", "extraction.skip_black_tile", True)
        self._add_double(ag_formal, "跳过黑色阈值", "extraction.skip_black_ratio_threshold",
                         0.80, 0.10, 1.00, 0.05)
        self._add_spin(ag_formal, "黑色像素阈值", "extraction.black_threshold", 10, 0, 32)
        self._add_double(ag_formal, "Tile 最小有效比例", "extraction.valid_pixel_ratio_threshold",
                         0.10, 0.0, 1.0, 0.05)

        ag = self._add_group("色彩分割（传统）")
        self._add_combo(ag, "模式", "segment.mode",
                        ["combined", "hsv", "lab"], "combined")
        self._add_combo(ag, "组合方式", "segment.combine_method",
                        ["and", "or"], "and")
        self._add_spin(ag, "采样半径", "segment.sample_radius", 3, 1, 20)
        self._add_spin(ag, "H 容差", "segment.h_margin", 6, 1, 50)
        self._add_spin(ag, "S 容差", "segment.s_margin", 25, 1, 100)
        self._add_spin(ag, "V 容差", "segment.v_margin", 30, 1, 100)
        self._add_spin(ag, "Lab 容差", "segment.lab_margin", 12, 1, 50)
        self._add_check(ag, "使用负样本", "segment.use_negative_samples", True)
        self._add_spin(ag, "正样本距离阈值", "segment.positive_distance_threshold", 20, 1, 100)
        self._add_spin(ag, "负样本容差", "segment.negative_margin", 6, 1, 50)
        self._add_spin(ag, "Tile 大小", "segment.tile_size", 1024, 256, 4096)
        self._add_spin(ag, "Tile 重叠", "segment.overlap", 64, 0, 1023)
        self._add_double(ag, "预览缩放", "segment.preview_scale", 0.25, 0.10, 0.50, 0.05)
        self._add_check(ag, "ROI 优先", "segment.use_roi_only", True)
        self._add_check(ag, "跳过黑色无效区", "segment.skip_black_area", True)
        self._add_spin(ag, "黑色阈值", "segment.black_threshold", 10, 0, 32)
        self._add_spin(ag, "边界黑区最小面积", "segment.min_black_component_area", 4096, 1, 100000000)
        self._add_double(ag, "Tile 最小有效比例", "segment.valid_pixel_ratio_threshold", 0.10, 0.0, 1.0, 0.05)

        self._end_advanced()

    def _on_segment_button_clicked(self):
        self.apply_requested.emit(
            "cancel_segment" if self._segment_running else "segment"
        )

    def set_segmentation_running(self, running: bool, cancelling: bool = False):
        self._segment_running = bool(running)
        self._clear_samples_btn.setEnabled(not running)
        self._preview_segment_btn.setEnabled(not running)
        if running:
            self._segment_btn.setText("正在取消…" if cancelling else "取消分割")
            self._segment_btn.setEnabled(not cancelling)
            self._segment_progress.show()
            self._segment_progress_label.show()
        else:
            self._segment_btn.setText("🚀 正式道路提取（tile，后台）")
            self._segment_btn.setEnabled(True)
            self._segment_progress.hide()
            self._segment_progress_label.hide()

    def set_preview_seg_running(self, running: bool):
        """快速预览分割 running 状态。"""
        self._preview_seg_running = bool(running)
        self._segment_btn.setEnabled(not running)
        if running:
            self._preview_segment_btn.setText("正在预览…")
            self._preview_segment_btn.setEnabled(False)
            self._segment_progress.show()
            self._segment_progress_label.show()
        else:
            self._preview_segment_btn.setText("⚡ 快速预览分割（低清，仅预览）")
            self._preview_segment_btn.setEnabled(True)
            self._segment_btn.setEnabled(True)
            self._segment_progress.hide()
            self._segment_progress_label.hide()

    def update_preview_seg_progress(self, percent: int, message: str):
        """更新快速预览分割进度（简化版，无 tile 计数）。"""
        self._segment_progress.setValue(max(0, min(100, int(percent))))
        self._segment_progress.setFormat(f"{int(percent)}%")
        self._segment_progress_label.setText(message)

    def set_competition_fast_running(self, running: bool):
        """运行时只禁用相关提取按钮；取消按钮必须可点，不禁用整个主窗口。"""
        self._competition_fast_running = bool(running)
        btn = getattr(self, "_competition_fast_btn", None)
        if btn is not None:
            # 保持可点击，用于「取消快速路网生成」
            btn.setEnabled(True)
            btn.setText(
                "取消快速路网生成" if running else "大图比赛快速路网生成（直接 graph，不推荐）"
            )
        if hasattr(self, "_preview_segment_btn"):
            self._preview_segment_btn.setEnabled(not running)
        if hasattr(self, "_lowres_formal_btn"):
            self._lowres_formal_btn.setEnabled(not running)
        if hasattr(self, "_roi_extract_btn"):
            self._roi_extract_btn.setEnabled(not running)
        if hasattr(self, "_full_extract_btn"):
            self._full_extract_btn.setEnabled(not running)
        if hasattr(self, "_emergency_draw_btn"):
            self._emergency_draw_btn.setEnabled(not running)
        if running:
            self._segment_progress.show()
            self._segment_progress_label.show()
        else:
            self._segment_progress.hide()
            self._segment_progress_label.hide()

    def set_lowres_formal_running(self, running: bool):
        """低像素正式 Mask 运行状态；取消按钮保持可点。"""
        self._lowres_formal_running = bool(running)
        btn = getattr(self, "_lowres_formal_btn", None)
        if btn is not None:
            btn.setEnabled(True)
            btn.setText(
                "取消低像素 Mask 生成" if running else "🏆 低像素快速生成正式 Mask"
            )
        if hasattr(self, "_preview_segment_btn"):
            self._preview_segment_btn.setEnabled(not running)
        if hasattr(self, "_competition_fast_btn"):
            self._competition_fast_btn.setEnabled(not running)
        if hasattr(self, "_roi_extract_btn"):
            self._roi_extract_btn.setEnabled(not running)
        if hasattr(self, "_full_extract_btn"):
            self._full_extract_btn.setEnabled(not running)
        if hasattr(self, "_emergency_draw_btn"):
            self._emergency_draw_btn.setEnabled(not running)
        if running:
            self._segment_progress.show()
            self._segment_progress_label.show()
        else:
            self._segment_progress.hide()
            self._segment_progress_label.hide()

    def update_lowres_formal_progress(self, percent: int, message: str):
        self._segment_progress.setValue(max(0, min(100, int(percent))))
        self._segment_progress.setFormat(f"{int(percent)}%")
        self._segment_progress_label.setText(message)

    def update_lowres_formal_status(
        self,
        *,
        mask_source: str = "-",
        formal_ready: bool = False,
        lowres_w: int = 0,
        lowres_h: int = 0,
        original_w: int = 0,
        original_h: int = 0,
        scale_x: float = 0.0,
        scale_y: float = 0.0,
    ):
        label = getattr(self, "_lowres_formal_status", None)
        if label is None:
            return
        lw = f"{lowres_w}×{lowres_h}" if lowres_w and lowres_h else "-"
        ow = f"{original_w}×{original_h}" if original_w and original_h else "-"
        sx = f"{scale_x:.4f}" if scale_x else "-"
        sy = f"{scale_y:.4f}" if scale_y else "-"
        label.setText(
            f"当前 Mask 来源：{mask_source or '-'}\n"
            f"正式 Mask：{'是' if formal_ready else '否'}\n"
            f"工作分辨率：{lw}\n"
            f"原图分辨率：{ow}\n"
            f"scale_x / scale_y：{sx} / {sy}"
        )

    def update_competition_fast_progress(self, percent: int, message: str):
        self._segment_progress.setValue(max(0, min(100, int(percent))))
        self._segment_progress.setFormat(f"{int(percent)}%")
        self._segment_progress_label.setText(message)

    def update_competition_fast_status(self, *, work_w=0, work_h=0, max_side=1500,
                                       elapsed_s=None, ready=True):
        label = getattr(self, "_competition_status", None)
        if label is None:
            return
        elapsed = f"{elapsed_s:.1f} s" if elapsed_s is not None else "约 1～2 min"
        label.setText(
            "（旧）Competition Fast Roadnet\n"
            f"工作分辨率：{work_w}×{work_h} (max side ≤{int(max_side)})\n"
            f"耗时：{elapsed} · ready={ready}"
        )

    def set_formal_extraction_running(self, running: bool, mode: str = ""):
        """设置正式提取（ROI/全图）的运行状态。"""
        self._formal_extraction_running = bool(running)
        self._preview_segment_btn.setEnabled(not running)
        self._roi_extract_btn.setEnabled(not running)
        self._full_extract_btn.setEnabled(not running)
        for attr in ("_roi_model_btn", "_full_model_btn"):
            btn = getattr(self, attr, None)
            if btn is not None and _shiboken_is_valid(btn):
                btn.setEnabled(not running)
        self._segment_btn.setEnabled(not running)
        self._clear_samples_btn.setEnabled(not running)
        if running:
            mode_label = "ROI" if mode == "roi" else ("全图" if mode == "full" else mode)
            self._segment_btn.setText(f"取消提取 ({mode_label})")
            self._segment_btn.setEnabled(True)  # 允许取消
            self._segment_progress.show()
            self._segment_progress_label.show()
        else:
            self._segment_btn.setText("🚀 传统分割（tile，后台）")
            self._segment_btn.setEnabled(True)
            self._segment_progress.hide()
            self._segment_progress_label.hide()

    def update_formal_extraction_progress(
        self, percent: int, current: int, total: int, detail: dict
    ):
        """更新正式提取进度（含详细信息）。"""
        self._segment_progress.setValue(max(0, min(100, int(percent))))
        self._segment_progress.setFormat(f"{int(percent)}% ({current}/{total})")
        cache_hits = detail.get("cache_hit_tiles", 0)
        failed = detail.get("failed_tiles", 0)
        success = detail.get("success_tiles", 0)
        elapsed = detail.get("elapsed_seconds", 0)
        remaining = detail.get("estimated_remaining_seconds", 0)
        avg_time = detail.get("avg_time_per_tile", 0)
        mode = detail.get("mode", "")
        infer_mode = detail.get("infer_mode", "")
        stage = detail.get("stage", "")
        tile_id = detail.get("tile_id", "")
        mode_label = {"roi": "ROI", "full": "全图", "fast_preview": "快速预览"}.get(mode, mode)
        stage_label = {
            "load_model": "加载模型",
            "launch_worker": "启动进程",
            "infer_tiles": "tile 推理",
            "select_tiles": "选择 tile",
        }.get(stage, stage)

        lines = [
            f"阶段: {stage_label} | 模式: {mode_label} | 推理: {infer_mode}",
        ]
        if total > 0:
            lines.append(f"当前 tile: {current}/{total}" + (f" ({tile_id})" if tile_id else ""))
        lines.extend([
            f"成功: {success} 失败: {failed} 缓存命中: {cache_hits}",
            f"已用: {elapsed:.0f}s 预计剩余: {remaining:.0f}s (平均 {avg_time:.1f}s/tile)",
        ])
        self._segment_progress_label.setText("\n".join(lines))

    def update_roi_status(self, count: int, covered_tiles=None, need_infer=None):
        """更新道路分割页 ROI 状态面板。"""
        lbl = getattr(self, "_seg_roi_count_label", None)
        if lbl is not None and _shiboken_is_valid(lbl):
            lbl.setText(f"ROI 数量：{count}")
        tile_lbl = getattr(self, "_seg_roi_tile_label", None)
        if tile_lbl is not None and _shiboken_is_valid(tile_lbl):
            tile_lbl.setText(
                f"覆盖 tile：{covered_tiles}" if covered_tiles is not None else "覆盖 tile：-"
            )
        infer_lbl = getattr(self, "_seg_roi_infer_label", None)
        if infer_lbl is not None and _shiboken_is_valid(infer_lbl):
            infer_lbl.setText(
                f"待推理 tile：{need_infer}" if need_infer is not None else "待推理 tile：-"
            )

    def highlight_roi_extract_btn(self, highlight: bool = True):
        """高亮 ROI 正式提取按钮（引导用户继续操作）。"""
        btn = getattr(self, "_roi_extract_btn", None)
        if btn is None or not _shiboken_is_valid(btn):
            return
        if highlight:
            btn.setStyleSheet(
                "QPushButton { background-color: #ffc928; color: #111; "
                "font-weight: bold; padding: 9px; border-radius: 4px; "
                "border: 2px solid #ff79c6; }"
                "QPushButton:hover { background-color: #ffe66d; }"
            )
        else:
            btn.setStyleSheet(
                "QPushButton { background-color: #e6a700; color: #111; "
                "font-weight: bold; padding: 9px; border-radius: 4px; }"
                "QPushButton:hover { background-color: #ffc928; }"
            )

    def update_segmentation_progress(
        self, percent: int, current: int, total: int, message: str
    ):
        self._segment_progress.setValue(max(0, min(100, int(percent))))
        self._segment_progress.setFormat(f"{int(percent)}% ({current}/{total})")
        self._segment_progress_label.setText(message)

    def _build_edit_page(self):
        g1 = self._add_group("状态")
        stable_tip = QLabel("区域修正稳定模式：仅修改当前 Road Mask")
        stable_tip.setWordWrap(True)
        stable_tip.setStyleSheet("color: #a6e3a1; font-weight: bold; padding: 4px;")
        g1.addWidget(stable_tip)
        self._roi_count_label = QLabel("ROI 数量：0")
        self._roi_count_label.setStyleSheet("color: #89b4fa;")
        g1.addWidget(self._roi_count_label)

        self._ignore_count_label = QLabel("Ignore 数量：0")
        self._ignore_count_label.setStyleSheet("color: #ff5555;")
        g1.addWidget(self._ignore_count_label)

        g2 = self._add_group("Mask 精修")
        self._add_spin(g2, "画笔半径", "edit.brush_radius", 8, 1, 100)
        self._add_spin(g2, "最大撤销步数", "edit.max_undo_steps", 20, 5, 100)

        # ── 主要操作 ──
        g_main = self._add_group("主要操作")
        btn_roi = QPushButton("🔄 应用 ROI")
        btn_roi.clicked.connect(lambda: self.apply_requested.emit("apply_roi"))
        g_main.addWidget(btn_roi)

        btn_ignore = QPushButton("🚫 应用 Ignore")
        btn_ignore.clicked.connect(lambda: self.apply_requested.emit("apply_ignore"))
        g_main.addWidget(btn_ignore)

        btn_mask_filter = QPushButton("🔎 自动筛选 Mask 误检")
        btn_mask_filter.setToolTip("分析连通域并显示疑似孤立噪声、过宽区域和块状误检")
        btn_mask_filter.clicked.connect(lambda: self.apply_requested.emit("analyze_mask_quality"))
        btn_mask_filter.setVisible(False)
        g_main.addWidget(btn_mask_filter)

        btn_apply_candidates = QPushButton("✅ 应用高置信 Ignore")
        btn_apply_candidates.setToolTip("只应用 confidence > 0.8 的候选；操作可通过 Ctrl+Z 撤销")
        btn_apply_candidates.clicked.connect(lambda: self.apply_requested.emit("apply_mask_candidates"))
        self._apply_mask_candidates_btn = btn_apply_candidates
        btn_apply_candidates.setEnabled(False)
        btn_apply_candidates.setVisible(False)
        btn_apply_candidates.setToolTip(
            self._mask_candidate_apply_reason
            or "仅应用安全检查通过且 confidence >= 0.90 的候选；支持 Ctrl+Z 撤销"
        )
        g_main.addWidget(btn_apply_candidates)

        btn_view_candidates = QPushButton("📋 查看 Ignore 候选")
        btn_view_candidates.clicked.connect(lambda: self.apply_requested.emit("view_mask_candidates"))
        g_main.addWidget(btn_view_candidates)

        btn_save_mask = QPushButton("💾 保存当前 Mask")
        btn_save_mask.setToolTip("保存当前编辑的 mask，不执行后处理")
        btn_save_mask.clicked.connect(lambda: self.apply_requested.emit("save_mask"))
        g_main.addWidget(btn_save_mask)

        # ── 大图主路优先修复（种子 / ROI / 任务点约束，仅大图模式）──
        g_mainroad = self._add_group("大图主路优先（种子约束半自动修复）")
        mainroad_tip = QLabel(
            "在主路种子线 / ROI / 任务点约束的 corridor 内修复主路：\n"
            "保留主路连通、删除孤立误检、有上限地桥接断点。\n"
            "不做全图自由修复，避免把误检碎片越连越乱。"
        )
        mainroad_tip.setStyleSheet("color: #a0a0a0; font-size: 9px; padding: 2px 4px;")
        mainroad_tip.setWordWrap(True)
        g_mainroad.addWidget(mainroad_tip)

        self._seed_count_label = QLabel("主路种子线：0 笔")
        self._seed_count_label.setStyleSheet("color: #ff5cff; font-weight: bold;")
        g_mainroad.addWidget(self._seed_count_label)

        width_row = QWidget()
        width_layout = QHBoxLayout(width_row)
        width_layout.setContentsMargins(0, 0, 0, 0)
        width_layout.addWidget(QLabel("道路宽度"))
        self._seed_width_combo = QComboBox()
        self._seed_width_combo.addItem("普通道路 8m", "normal")
        self._seed_width_combo.addItem("主路 12m", "main_road")
        self._seed_width_combo.addItem("路口/环岛 16m", "junction")
        self._seed_width_combo.addItem("自定义", "custom")
        width_layout.addWidget(self._seed_width_combo)
        g_mainroad.addWidget(width_row)

        custom_row = QWidget()
        custom_layout = QHBoxLayout(custom_row)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        custom_layout.addWidget(QLabel("自定义宽(m)"))
        self._seed_width_m_spin = QDoubleSpinBox()
        self._seed_width_m_spin.setRange(1.0, 40.0)
        self._seed_width_m_spin.setValue(8.0)
        self._seed_width_m_spin.setSingleStep(1.0)
        custom_layout.addWidget(self._seed_width_m_spin)
        custom_layout.addWidget(QLabel("或半径(px)"))
        self._seed_radius_px_spin = QDoubleSpinBox()
        self._seed_radius_px_spin.setRange(0.0, 200.0)
        self._seed_radius_px_spin.setValue(0.0)
        self._seed_radius_px_spin.setSpecialValueText("自动")
        self._seed_radius_px_spin.setToolTip("0=按宽度与 GSD 自动换算；无 GSD 时可直接填像素半径")
        custom_layout.addWidget(self._seed_radius_px_spin)
        g_mainroad.addWidget(custom_row)

        self._seed_continuous_cb = QCheckBox("连续两点画线")
        self._seed_continuous_cb.setChecked(True)
        self._seed_continuous_cb.setToolTip("两点模式：每点两下生成一条线后继续等待下一条起点")
        g_mainroad.addWidget(self._seed_continuous_cb)

        btn_seed_two = QPushButton("📏 绘制两点主路线")
        btn_seed_two.setObjectName("primary")
        btn_seed_two.setToolTip("点击起点 → 点击终点，立即生成直线种子并预览 road ribbon")
        btn_seed_two.clicked.connect(lambda: self.apply_requested.emit("seed_draw_two_point"))
        g_mainroad.addWidget(btn_seed_two)

        btn_seed_poly = QPushButton("✏ 绘制多点主路线")
        btn_seed_poly.setToolTip("连续点击加点，双击/右键结束；适合弯道、环岛")
        btn_seed_poly.clicked.connect(lambda: self.apply_requested.emit("seed_draw_polyline"))
        g_mainroad.addWidget(btn_seed_poly)

        btn_seed_draw = QPushButton("🖌 自由绘主路种子线")
        btn_seed_draw.setToolTip("按住左键拖动绘制（旧模式，仍可用）")
        btn_seed_draw.clicked.connect(lambda: self.apply_requested.emit("seed_draw"))
        g_mainroad.addWidget(btn_seed_draw)

        btn_seed_undo = QPushButton("↩ 撤销上一条种子线")
        btn_seed_undo.clicked.connect(lambda: self.apply_requested.emit("seed_undo_last"))
        g_mainroad.addWidget(btn_seed_undo)

        btn_seed_clear = QPushButton("🧹 清空主路种子线")
        btn_seed_clear.clicked.connect(lambda: self.apply_requested.emit("seed_clear"))
        g_mainroad.addWidget(btn_seed_clear)

        btn_seed_rebuild = QPushButton("🛠 根据种子线重建 Mask")
        btn_seed_rebuild.setObjectName("primary")
        btn_seed_rebuild.setToolTip(
            "将 seed strokes 膨胀为 road ribbon，合并进 working_road_mask，\n"
            "删除远离 ribbon 的误检，输出 final_edited_mask"
        )
        btn_seed_rebuild.clicked.connect(lambda: self.apply_requested.emit("seed_rebuild_mask"))
        g_mainroad.addWidget(btn_seed_rebuild)

        # ── 道路带约束补洞 / 补缺口（仅大图）──
        g_ribbon_fill = self._add_group("道路带内补洞补缺口")
        ribbon_fill_tip = QLabel(
            "在 road_ribbon_mask 附近补内部小孔洞与道路带内缺口。\n"
            "不重跑分割；不修改影像；只改 Road Mask。\n"
            "请先绘制主路种子线并生成道路带。"
        )
        ribbon_fill_tip.setStyleSheet("color: #a0a0a0; font-size: 9px; padding: 2px 4px;")
        ribbon_fill_tip.setWordWrap(True)
        g_ribbon_fill.addWidget(ribbon_fill_tip)

        self._add_spin(
            g_ribbon_fill, "最大孔洞面积", "ribbon_fill.max_hole_area_px",
            500, 50, 20000, step=50,
        )
        self._add_spin(
            g_ribbon_fill, "最大缺口面积", "ribbon_fill.max_gap_area_px",
            800, 50, 20000, step=50,
        )
        self._add_spin(
            g_ribbon_fill, "最大孔洞直径", "ribbon_fill.max_hole_diameter_px",
            25, 5, 200, step=1,
        )
        self._add_spin(
            g_ribbon_fill, "最大缺口直径", "ribbon_fill.max_gap_diameter_px",
            35, 5, 200, step=1,
        )
        self._add_spin(
            g_ribbon_fill, "道路带外扩距离", "ribbon_fill.ribbon_buffer_px",
            10, 0, 80, step=1,
        )
        self._add_double(
            g_ribbon_fill, "孔洞周围道路占比", "ribbon_fill.min_surround_ratio_for_hole",
            0.70, 0.0, 1.0, step=0.05, decimals=2,
        )
        self._add_double(
            g_ribbon_fill, "缺口周围道路占比", "ribbon_fill.min_surround_ratio_for_gap",
            0.45, 0.0, 1.0, step=0.05, decimals=2,
        )
        self._add_spin(
            g_ribbon_fill, "缺口最大距道路", "ribbon_fill.max_gap_distance_to_mask_px",
            8, 0, 50, step=1,
        )
        self._add_check(
            g_ribbon_fill, "只在 road ribbon 内补",
            "ribbon_fill.require_inside_ribbon", True,
        )

        btn_ribbon_fill = QPushButton("🩹 道路带内补洞补缺口")
        btn_ribbon_fill.setObjectName("primary")
        btn_ribbon_fill.setToolTip(
            "仅大图：在 road_ribbon_mask 约束下填充内部孔洞与带内缺口"
        )
        btn_ribbon_fill.clicked.connect(
            lambda: self.apply_requested.emit("ribbon_hole_gap_fill")
        )
        g_ribbon_fill.addWidget(btn_ribbon_fill)

        btn_ribbon_view = QPushButton("👁 查看补洞补缺口结果")
        btn_ribbon_view.clicked.connect(
            lambda: self.apply_requested.emit("ribbon_hole_gap_view")
        )
        g_ribbon_fill.addWidget(btn_ribbon_view)

        btn_ribbon_accept = QPushButton("✅ 接受结果")
        btn_ribbon_accept.clicked.connect(
            lambda: self.apply_requested.emit("ribbon_hole_gap_accept")
        )
        g_ribbon_fill.addWidget(btn_ribbon_accept)

        btn_ribbon_rollback = QPushButton("↩ 回滚")
        btn_ribbon_rollback.clicked.connect(
            lambda: self.apply_requested.emit("ribbon_hole_gap_rollback")
        )
        g_ribbon_fill.addWidget(btn_ribbon_rollback)

        # ── 种子线一键清理 Mask（省事方案）──
        g_clean = self._add_group("大图种子线清理 Mask（推荐）")
        clean_tip = QLabel(
            "画几笔主路种子线后，一键清理当前 Road Mask：\n"
            "保留主路相关连通域，删除远离种子线的屋顶/草地误检。\n"
            "清理后再生成骨架，避免直接对脏 mask skeletonize。"
        )
        clean_tip.setStyleSheet("color: #a0a0a0; font-size: 9px; padding: 2px 4px;")
        clean_tip.setWordWrap(True)
        g_clean.addWidget(clean_tip)

        btn_seed_clean = QPushButton("🧹 根据种子线清理当前 Mask")
        btn_seed_clean.setObjectName("primary")
        btn_seed_clean.setToolTip("仅大图：用主路种子线筛选 working mask，输出 cleaned_working_mask")
        btn_seed_clean.clicked.connect(lambda: self.apply_requested.emit("seed_clean_mask"))
        g_clean.addWidget(btn_seed_clean)

        btn_seed_compare = QPushButton("👁 查看清理前后对比")
        btn_seed_compare.clicked.connect(lambda: self.apply_requested.emit("seed_clean_compare"))
        g_clean.addWidget(btn_seed_compare)

        btn_seed_accept = QPushButton("✅ 接受 cleaned mask")
        btn_seed_accept.clicked.connect(lambda: self.apply_requested.emit("seed_clean_accept"))
        g_clean.addWidget(btn_seed_accept)

        btn_seed_rollback = QPushButton("↩ 回滚到 working mask")
        btn_seed_rollback.clicked.connect(lambda: self.apply_requested.emit("seed_clean_rollback"))
        g_clean.addWidget(btn_seed_rollback)

        btn_seed_view = QPushButton("🖼 使用当前视野作为修复范围")
        btn_seed_view.clicked.connect(lambda: self.apply_requested.emit("seed_use_view"))
        g_mainroad.addWidget(btn_seed_view)

        btn_seed_tasks = QPushButton("📍 使用任务点生成 corridor")
        btn_seed_tasks.clicked.connect(lambda: self.apply_requested.emit("seed_use_tasks"))
        g_mainroad.addWidget(btn_seed_tasks)

        btn_view_corridor = QPushButton("👁 查看主路 corridor")
        btn_view_corridor.clicked.connect(lambda: self.apply_requested.emit("seed_view_corridor"))
        g_mainroad.addWidget(btn_view_corridor)

        btn_preview_bridges = QPushButton("🔎 预览桥接候选")
        btn_preview_bridges.setToolTip("绿=接受，红=拒绝，黄=待人工确认；不直接改动 mask")
        btn_preview_bridges.clicked.connect(lambda: self.apply_requested.emit("refine_preview_bridges"))
        g_mainroad.addWidget(btn_preview_bridges)

        btn_refine_mainroad = QPushButton("🛣 执行主路修复")
        btn_refine_mainroad.setObjectName("primary")
        btn_refine_mainroad.setToolTip(
            "仅大图模式：在 corridor 内修复主路，映射回原图，\n"
            "注册为正式 Road Mask（formal_opencv_mainroad_refined）"
        )
        btn_refine_mainroad.clicked.connect(lambda: self.apply_requested.emit("refine_main_road"))
        g_mainroad.addWidget(btn_refine_mainroad)

        btn_refine_accept = QPushButton("✅ 接受修复结果")
        btn_refine_accept.clicked.connect(lambda: self.apply_requested.emit("refine_accept"))
        g_mainroad.addWidget(btn_refine_accept)

        btn_refine_rollback = QPushButton("↩ 回滚修复结果")
        btn_refine_rollback.clicked.connect(lambda: self.apply_requested.emit("refine_rollback"))
        g_mainroad.addWidget(btn_refine_rollback)

        btn_to_skeleton = QPushButton("🦴 进入骨架生成")
        btn_to_skeleton.setObjectName("primary")
        btn_to_skeleton.setToolTip("使用当前 mask 直接生成骨架（跳过后处理）")
        btn_to_skeleton.clicked.connect(lambda: self.apply_requested.emit("skeleton_direct"))
        btn_to_skeleton.setVisible(False)
        g_main.addWidget(btn_to_skeleton)

        # ── 高级操作（折叠）──
        self._add_advanced_toggle()
        self._start_advanced()

        ag = self._add_group("高级操作（可选后处理）")
        warn = QLabel("⚠ 后处理可能改变道路形态，非必须操作")
        warn.setStyleSheet("color: #f9e2af; font-size: 10px; padding: 4px;")
        warn.setWordWrap(True)
        ag.addWidget(warn)

        # ★ Ignore 使用建议
        ignore_tip = QLabel(
            "💡 城市小区/校园/园区影像建议先用 Ignore 删除：\n"
            "   大草坪、建筑屋顶、水体、停车场、非路径区域"
        )
        ignore_tip.setStyleSheet("color: #a0a0a0; font-size: 9px; padding: 2px 4px;")
        ignore_tip.setWordWrap(True)
        ag.addWidget(ignore_tip)

        self._add_check(
            ag, "显示候选编号", "visualization.show_mask_candidate_numbers", True
        )
        self._add_check(
            ag, "显示候选原因", "visualization.show_mask_candidate_reasons", False
        )

        btn_post = QPushButton("⚙ 可选后处理")
        btn_post.setToolTip("执行形态学后处理（可能改变道路形态）")
        btn_post.clicked.connect(lambda: self.apply_requested.emit("postprocess"))
        ag.addWidget(btn_post)

        btn_restore = QPushButton("↩ 恢复后处理前 Mask")
        btn_restore.setToolTip("还原后处理之前的 mask")
        btn_restore.clicked.connect(lambda: self.apply_requested.emit("restore_mask"))
        ag.addWidget(btn_restore)

        # ★ 恢复原始 SAM-Road road_mask
        btn_restore_orig = QPushButton("🔄 恢复原始 road_mask")
        btn_restore_orig.setToolTip(
            "从 SAM-Road 输出目录重新加载原始 road_mask.png，\n"
            "覆盖当前被后处理/编辑污染的 mask"
        )
        btn_restore_orig.setStyleSheet("color: #ff8800;")
        btn_restore_orig.clicked.connect(lambda: self.apply_requested.emit("restore_original_roadmask"))
        ag.addWidget(btn_restore_orig)

        ag2 = self._add_group("形态学参数")
        self._add_spin(ag2, "开运算核", "postprocess.open_kernel_size", 3, 1, 21)
        # ★ close_kernel 风险提示
        close_spin = self._add_spin(ag2, "闭运算核 ⚠", "postprocess.close_kernel_size", 5, 1, 21)
        close_spin.setToolTip("城市密集区建议 3~5，>=9 容易大面积粘连！")
        close_spin.valueChanged.connect(self._check_close_kernel_risk)
        close_warn = QLabel("  ⚠ >=9 易大面积粘连")
        close_warn.setStyleSheet("color: #888888; font-size: 9px; padding-left: 80px;")
        ag2.addWidget(close_warn)

        self._add_check(ag2, "孔洞填充（全局→不推荐）", "postprocess.fill_holes", False)
        self._add_check(ag2, "填充小孔洞（推荐）", "postprocess.fill_small_holes", False)
        fill_spin = self._add_spin(ag2, "  小孔洞最大面积", "postprocess.max_hole_area", 500, 100, 5000, step=100)
        fill_spin.setToolTip("只填充面积 <= 此值的小孔洞，大孔洞（如草地）保留")
        self._add_spin(ag2, "最小面积", "postprocess.min_area", 500, 0, 10000)
        self._add_spin(ag2, "平滑核", "postprocess.smooth_kernel_size", 0, 0, 15)

        self._end_advanced()
        # 稳定模式暂停后处理/骨架等高级入口，避免误触发非基础流程。
        self._adv_btn.setVisible(False)

    def _build_skeleton_page(self):
        g1 = self._add_group("骨架操作")
        btn_gen = QPushButton("🦴 生成骨架")
        btn_gen.setObjectName("primary")
        btn_gen.clicked.connect(lambda: self.apply_requested.emit("skeleton"))
        g1.addWidget(btn_gen)

        btn_opt = QPushButton("✨ 优化骨架")
        btn_opt.clicked.connect(lambda: self.apply_requested.emit("optimize"))
        g1.addWidget(btn_opt)

        # ★ 查看优化对比
        btn_compare = QPushButton("🔍 查看优化对比")
        btn_compare.setToolTip("打开 road_skeleton_optimized_overlay.png")
        btn_compare.clicked.connect(lambda: self.apply_requested.emit("view_compare"))
        g1.addWidget(btn_compare)

        btn_save = QPushButton("💾 保存骨架")
        btn_save.clicked.connect(lambda: self.apply_requested.emit("save_skeleton"))
        g1.addWidget(btn_save)

        # ---- 大图专用：主路约束骨架 ----
        g_large = self._add_group("大图骨架（主路约束清理）")
        btn_view_input = QPushButton("🔎 查看骨架输入 Mask")
        btn_view_input.setToolTip(
            "显示当前将用于正式骨架生成的 full-size mask 路径与诊断信息"
        )
        btn_view_input.clicked.connect(lambda: self.apply_requested.emit("skel_view_input_mask"))
        g_large.addWidget(btn_view_input)

        self._large_skel_use_constraint = QCheckBox("使用主路约束生成骨架")
        self._large_skel_use_constraint.setChecked(True)
        g_large.addWidget(self._large_skel_use_constraint)

        self._large_skel_keep_seed = QCheckBox("仅保留种子/ROI/任务点相关骨架")
        self._large_skel_keep_seed.setChecked(True)
        g_large.addWidget(self._large_skel_keep_seed)

        self._large_skel_show_raw_cb = QCheckBox("显示 raw skeleton（噪声多）")
        self._large_skel_show_raw_cb.setChecked(False)
        g_large.addWidget(self._large_skel_show_raw_cb)

        btn_seed = QPushButton("✏ 绘制主路种子线")
        btn_seed.clicked.connect(lambda: self.apply_requested.emit("seed_draw"))
        g_large.addWidget(btn_seed)

        btn_raw = QPushButton("👁 显示 Raw Skeleton")
        btn_raw.clicked.connect(lambda: self.apply_requested.emit("skel_show_raw"))
        g_large.addWidget(btn_raw)

        btn_cleaned = QPushButton("✨ 显示 Cleaned Skeleton")
        btn_cleaned.clicked.connect(lambda: self.apply_requested.emit("skel_show_cleaned"))
        g_large.addWidget(btn_cleaned)

        btn_bridges = QPushButton("🔎 查看桥接候选")
        btn_bridges.clicked.connect(lambda: self.apply_requested.emit("skel_view_bridges"))
        g_large.addWidget(btn_bridges)

        btn_accept = QPushButton("✅ 接受骨架结果")
        btn_accept.clicked.connect(lambda: self.apply_requested.emit("skel_accept"))
        g_large.addWidget(btn_accept)

        btn_rollback = QPushButton("↩ 回滚骨架结果")
        btn_rollback.clicked.connect(lambda: self.apply_requested.emit("skel_rollback"))
        g_large.addWidget(btn_rollback)

        # ---- 统计 ----
        g_stats = self._add_group("优化统计")
        font = QFont()
        font.setPointSize(9)
        self._skel_stat_labels = {}
        stat_names = [
            "raw_pixels", "optimized_pixels",
            "optimized_endpoints", "raw_junction_pixels",
            "junction_cluster_count", "removed_spur_count",
            "connected_gap_count",
        ]
        stat_display = {
            "raw_pixels": "原始骨架像素：--",
            "optimized_pixels": "优化后像素：--",
            "optimized_endpoints": "端点数量：--",
            "raw_junction_pixels": "交叉点像素：--",
            "junction_cluster_count": "聚类路口数量：--",
            "removed_spur_count": "删除毛刺：--",
            "connected_gap_count": "连接断点：--",
        }
        for key in stat_names:
            label = QLabel(stat_display.get(key, key))
            label.setFont(font)
            label.setWordWrap(True)
            g_stats.addWidget(label)
            self._skel_stat_labels[key] = label

        # ---- 高级设置（折叠） ----
        self._add_advanced_toggle()
        self._start_advanced()

        ag = self._add_group("骨架化参数")
        self._add_double(ag, "最小中心距离", "skeleton.min_center_dist", 2.0, 0.5, 20.0, 0.5)
        self._add_spin(ag, "边界留白", "skeleton.border_margin", 10, 0, 50)
        self._add_spin(ag, "最小分支长度", "skeleton.min_branch_length", 20, 5, 500)

        ag2 = self._add_group("断线连接")
        self._add_spin(ag2, "最大连接距离", "skeleton.max_connect_dist", 25, 5, 200)
        self._add_spin(ag2, "最大连接角度", "skeleton.max_connect_angle", 45, 5, 90)
        self._add_double(ag2, "最小重合比例", "skeleton.min_line_mask_overlap", 0.65, 0.1, 1.0, 0.05)

        self._end_advanced()

    def large_skeleton_use_constraint(self) -> bool:
        cb = getattr(self, "_large_skel_use_constraint", None)
        return bool(cb.isChecked()) if cb is not None else True

    def large_skeleton_keep_seed_only(self) -> bool:
        cb = getattr(self, "_large_skel_keep_seed", None)
        return bool(cb.isChecked()) if cb is not None else True

    def large_skeleton_show_raw(self) -> bool:
        cb = getattr(self, "_large_skel_show_raw_cb", None)
        return bool(cb.isChecked()) if cb is not None else False

    def update_skeleton_stats(self, stats: dict):
        """更新骨架优化统计"""
        if not hasattr(self, "_skel_stat_labels"):
            self._skel_stat_labels = {}
        if not self._skel_stat_labels:
            return
        import numpy as np
        key_map = {
            "raw_pixels": "原始骨架像素：{}",
            "optimized_pixels": "优化后像素：{}",
            "optimized_endpoints": "端点数量：{}",
            "raw_junction_pixels": "交叉点像素：{}",
            "junction_cluster_count": "聚类路口数量：{}",
            "removed_spur_count": "删除毛刺：{}",
            "connected_gap_count": "连接断点：{}",
        }
        for key, label in list(self._skel_stat_labels.items()):
            val = stats.get(key, "--")
            fmt = key_map.get(key, "{}: {}")
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            try:
                label.setText(fmt.format(val))
            except RuntimeError:
                continue

    def _build_graph_page(self):
        # ---- 路网操作 ----
        g1 = self._add_group("路网操作")
        btn_draft = QPushButton("🔗 生成草稿路网")
        btn_draft.setObjectName("primary")
        btn_draft.clicked.connect(lambda: self.apply_requested.emit("graph"))
        g1.addWidget(btn_draft)

        btn_diagnose = QPushButton("🔎 分析路网问题")
        btn_diagnose.setToolTip("生成诊断报告与自动修复建议（黄色虚线预览）")
        btn_diagnose.clicked.connect(lambda: self.apply_requested.emit("diagnose_graph"))
        g1.addWidget(btn_diagnose)

        btn_issues = QPushButton("⚠ 分析路网并高亮异常")
        btn_issues.setObjectName("primary")
        btn_issues.setToolTip(
            "诊断 final_graph，在图上用红/橙/黄高亮异常，并打开异常列表。"
            "不修改 final_graph.json。"
        )
        btn_issues.clicked.connect(lambda: self.apply_requested.emit("highlight_graph_issues"))
        g1.addWidget(btn_issues)

        btn_show_issues = QPushButton("📋 显示路网异常列表")
        btn_show_issues.setToolTip("打开上次分析的异常列表；若路网已改动需重新分析")
        btn_show_issues.clicked.connect(lambda: self.apply_requested.emit("show_graph_issue_list"))
        g1.addWidget(btn_show_issues)

        repair_layout = QHBoxLayout()
        btn_apply_repair = QPushButton("✅ 应用高置信修复")
        btn_apply_repair.setToolTip("应用 confidence > 0.8 的建议；支持 Ctrl+Z/Ctrl+Y")
        btn_apply_repair.clicked.connect(lambda: self.apply_requested.emit("apply_graph_repairs"))
        repair_layout.addWidget(btn_apply_repair)
        btn_view_repair = QPushButton("📋 查看修复建议")
        btn_view_repair.clicked.connect(lambda: self.apply_requested.emit("view_graph_repairs"))
        repair_layout.addWidget(btn_view_repair)
        g1.addLayout(repair_layout)

        # ★ 全局撤销/重做按钮
        undo_layout = QHBoxLayout()
        btn_undo = QPushButton("↩ 撤销 (Ctrl+Z)")
        btn_undo.setToolTip("全局撤销上一步操作")
        btn_undo.setStyleSheet("font-size: 10px;")
        btn_undo.clicked.connect(lambda: self.apply_requested.emit("undo_graph"))
        undo_layout.addWidget(btn_undo)

        btn_redo = QPushButton("↪ 重做 (Ctrl+Y)")
        btn_redo.setToolTip("全局重做撤销的操作")
        btn_redo.setStyleSheet("font-size: 10px;")
        btn_redo.clicked.connect(lambda: self.apply_requested.emit("redo_graph"))
        undo_layout.addWidget(btn_redo)
        g1.addLayout(undo_layout)

        btn_save = QPushButton("💾 保存 Final Graph")
        btn_save.clicked.connect(lambda: self.apply_requested.emit("graph_save"))
        g1.addWidget(btn_save)

        # ---- 大图局部快速修路网 ----
        g_repair = self._add_group("局部快速修路网（大图）")
        tip = QLabel(
            "自动路网后的局部修正：折线补路 / 删错边 / 路口合并 / 局部重建。\n"
            "折线补路：沿道路中心点击 → 双击或 Enter 结束；自动吸附节点。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #a6adc8; font-size: 10px;")
        g_repair.addWidget(tip)

        btn_poly = QPushButton("✏️ 折线补路")
        btn_poly.setObjectName("primary")
        btn_poly.setToolTip("Polyline Edge Repair：沿中心线点击，双击结束")
        btn_poly.clicked.connect(lambda: self.apply_requested.emit("graph_polyline_repair"))
        g_repair.addWidget(btn_poly)

        btn_del = QPushButton("❌ 删除错误边")
        btn_del.clicked.connect(lambda: self.apply_requested.emit("graph_delete_edge_tool"))
        g_repair.addWidget(btn_del)

        btn_merge_j = QPushButton("🔀 合并路口节点")
        btn_merge_j.setToolTip("自动合并过近的路口节点簇")
        btn_merge_j.clicked.connect(lambda: self.apply_requested.emit("graph_merge_junctions"))
        g_repair.addWidget(btn_merge_j)

        btn_local = QPushButton("🧩 局部重建路网")
        btn_local.setToolTip("框选/绘制局部 ROI，仅在 ROI 内重建 graph")
        btn_local.clicked.connect(lambda: self.apply_requested.emit("graph_local_rebuild"))
        g_repair.addWidget(btn_local)

        btn_jump = QPushButton("🎯 定位异常跳边")
        btn_jump.clicked.connect(lambda: self.apply_requested.emit("graph_locate_jump"))
        g_repair.addWidget(btn_jump)

        btn_replan = QPushButton("🗺 重新规划路径")
        btn_replan.clicked.connect(lambda: self.apply_requested.emit("plan"))
        g_repair.addWidget(btn_replan)

        self._add_spin(g_repair, "节点吸附距离", "graph.node_snap_distance_px", 25, 5, 80)
        self._add_spin(g_repair, "路口合并距离", "graph.junction_merge_distance_px", 30, 5, 100)
        self._add_spin(g_repair, "端点吸附距离", "graph.endpoint_snap_distance_px", 25, 5, 80)
        self._add_spin(g_repair, "路口聚类半径", "graph.junction_cluster_radius_px", 30, 10, 80)

        btn_clear = QPushButton("🗑 清空人工修改")
        btn_clear.clicked.connect(lambda: self.apply_requested.emit("clear_graph_edits"))
        g1.addWidget(btn_clear)

        # ---- 统计 ----
        g2 = self._add_group("统计")
        font = QFont()
        font.setPointSize(9)
        self._graph_stat_labels = {}
        stat_items = [
            ("node_count",       "节点数量：--"),
            ("edge_count",       "边数量：--"),
            ("auto_edge_count",  "自动边数量：--"),
            ("manual_edge_count","人工边数量：--"),
            ("components",       "连通分量数量：--"),
            ("total_length_px",  "总长度(像素)：--"),
            ("total_length_m",   "总长度(米)：--"),
        ]
        for key, text in stat_items:
            label = QLabel(text)
            label.setFont(font)
            label.setWordWrap(True)
            g2.addWidget(label)
            self._graph_stat_labels[key] = label

        # ---- 高级设置（折叠） ----
        self._add_advanced_toggle()
        self._start_advanced()

        ag = self._add_group("图提取参数")
        self._add_spin(ag, "路口聚类半径", "graph.junction_cluster_radius", 10, 3, 50)
        self._add_spin(ag, "端点合并距离", "graph.endpoint_merge_distance", 12, 3, 50)
        self._add_spin(ag, "端点连接距离", "graph.endpoint_connect_distance", 25, 10, 100)
        self._add_spin(ag, "节点合并距离", "graph.node_merge_distance", 8, 1, 50)
        self._add_spin(ag, "最小边长度", "graph.min_edge_length", 8, 2, 50)
        self._add_spin(ag, "死端修剪长度", "graph.prune_length", 15, 0, 100)
        self._add_double(ag, "RDP简化容差", "graph.rdp_epsilon", 2.0, 0.5, 10.0, 0.5)
        self._add_check(ag, "启用线形优化", "graph.enable_graph_line_optimizer", False)

        ag2 = self._add_group("编辑参数")
        self._add_spin(ag2, "吸附距离", "graph.snap_distance", 10, 1, 50)
        self._add_double(ag2, "重复边阈值", "graph.duplicate_edge_threshold", 5.0, 1.0, 30.0, 1.0)

        self._end_advanced()

    def update_graph_stats(self, stats: dict):
        """更新路网统计"""
        if not hasattr(self, "_graph_stat_labels"):
            self._graph_stat_labels = {}
        if not self._graph_stat_labels:
            return
        key_map = {
            "node_count":       "节点数量：{}",
            "edge_count":       "边数量：{}",
            "auto_edge_count":  "自动边数量：{}",
            "manual_edge_count":"人工边数量：{}",
            "components":       "连通分量数量：{}",
            "total_length_px":  "总长度(像素)：{:.1f}",
            "total_length_m":   "总长度(米)：{:.1f}",
        }
        for key, label in list(self._graph_stat_labels.items()):
            val = stats.get(key, "--")
            fmt = key_map.get(key, "{}: {}")
            try:
                if val == "--":
                    label.setText(fmt.replace("{}", "--").replace("{:.1f}", "--"))
                else:
                    label.setText(fmt.format(val))
            except RuntimeError:
                # QLabel C++ 对象已被删除（阶段切换后）
                continue

    def _build_export_page(self):
        g_tp = self._add_group("任务点")
        btn_import_tp = QPushButton("📄 导入任务点文件")
        btn_import_tp.setObjectName("primary")
        btn_import_tp.setToolTip(
            "导入比赛任务点 txt：序号;经度;纬度;高程;属性\n"
            "0=起点 1=终点 2=必经点；按 seq 顺序规划"
        )
        btn_import_tp.clicked.connect(lambda: self.apply_requested.emit("import_task_points"))
        g_tp.addWidget(btn_import_tp)

        row_manual = QHBoxLayout()
        btn_start = QPushButton("设置起点")
        btn_start.setToolTip("进入起点输入模式；可连续点击，Esc 退出。已有起点时会确认是否替换。")
        btn_start.clicked.connect(lambda: self.apply_requested.emit("manual_set_start"))
        btn_goal = QPushButton("设置终点")
        btn_goal.setToolTip("进入终点输入模式；已有终点时会确认是否替换。")
        btn_goal.clicked.connect(lambda: self.apply_requested.emit("manual_set_goal"))
        row_manual.addWidget(btn_start)
        row_manual.addWidget(btn_goal)
        g_tp.addLayout(row_manual)

        btn_via = QPushButton("添加必经点 / 连续添加")
        btn_via.setObjectName("primary")
        btn_via.setToolTip(
            "连续添加任务点，不限数量。\n"
            "若尚无起点，第 1 个点自动为起点；之后均为必经点。\n"
            "Esc 退出输入模式。"
        )
        btn_via.clicked.connect(lambda: self.apply_requested.emit("manual_add_via"))
        g_tp.addWidget(btn_via)

        btn_clear_tp = QPushButton("清空任务点")
        btn_clear_tp.setToolTip("清空任务点数据与图层，并失效旧路径/航点结果")
        btn_clear_tp.clicked.connect(lambda: self.apply_requested.emit("clear_task_points"))
        g_tp.addWidget(btn_clear_tp)

        btn_validate_tp = QPushButton("验证任务点")
        btn_validate_tp.clicked.connect(lambda: self.apply_requested.emit("validate_task_points"))
        g_tp.addWidget(btn_validate_tp)

        btn_snap = QPushButton("吸附到路网")
        btn_snap.setObjectName("primary")
        btn_snap.clicked.connect(lambda: self.apply_requested.emit("snap_task_points"))
        g_tp.addWidget(btn_snap)

        self._task_points_table = QLabel("暂无任务点")
        self._task_points_table.setWordWrap(True)
        self._task_points_table.setStyleSheet(
            "color: #cdd6f4; font-size: 10px; font-family: Consolas, monospace; "
            "padding: 4px; background: #313244; border-radius: 4px;"
        )
        g_tp.addWidget(self._task_points_table)

        g1 = self._add_group("小车航点主流程")
        btn_dense = QPushButton("① 生成 dense_path")
        btn_dense.setObjectName("primary")
        btn_dense.setToolTip(
            "吸附任务点 → 按任务顺序规划 edge path → 展开 edge.polyline → dense_path.csv\n"
            "不从 final_graph 全图直接生成航点"
        )
        btn_dense.clicked.connect(lambda: self.apply_requested.emit("vwp_generate_dense_path"))
        g1.addWidget(btn_dense)

        btn_csv = QPushButton("② 生成小车航点 CSV")
        btn_csv.setToolTip("从 dense_path_labeled 采样：直线 15m / 弯道·路口 2m → vehicle_waypoints.csv")
        btn_csv.clicked.connect(lambda: self.apply_requested.emit("vwp_generate_vehicle_csv"))
        g1.addWidget(btn_csv)

        btn_check = QPushButton("③ 检查航点 CSV")
        btn_check.setToolTip("检查重复点 / ABA / 点距 / dense_index / s_m，写 validation report")
        btn_check.clicked.connect(lambda: self.apply_requested.emit("vwp_validate_vehicle_csv"))
        g1.addWidget(btn_check)

        btn_repair = QPushButton("④ 自动修复航点 CSV")
        btn_repair.setToolTip("仅基于 dense_path 区间插点修复 → vehicle_waypoints_repaired.csv")
        btn_repair.clicked.connect(lambda: self.apply_requested.emit("vwp_repair_vehicle_csv"))
        g1.addWidget(btn_repair)

        btn_yaml = QPushButton("⑤ 导出小车 YAML")
        btn_yaml.setToolTip("仅从 vehicle_waypoints_repaired.csv 格式转换 → subject1_waypoints.yaml")
        btn_yaml.clicked.connect(lambda: self.apply_requested.emit("vwp_export_yaml"))
        g1.addWidget(btn_yaml)

        btn_all = QPushButton("一键执行以上全部流程")
        btn_all.setObjectName("primary")
        btn_all.setStyleSheet(
            "QPushButton { background-color: #e6a700; color: #111; "
            "font-weight: bold; padding: 9px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #ffc928; }"
        )
        btn_all.clicked.connect(lambda: self.apply_requested.emit("vwp_run_full_pipeline"))
        g1.addWidget(btn_all)

        self._vwp_status_label = QLabel("主流程状态：未开始")
        self._vwp_status_label.setWordWrap(True)
        self._vwp_status_label.setStyleSheet(
            "color: #cdd6f4; font-size: 10px; font-family: Consolas, monospace; "
            "padding: 6px; background: #313244; border-radius: 4px;"
        )
        g1.addWidget(self._vwp_status_label)

        g2 = self._add_group("其他导出")
        btn_competition = QPushButton("🏆 导出比赛路网图")
        btn_competition.setToolTip("叠加图导出；正式 YAML 仍走上方主流程")
        btn_competition.clicked.connect(
            lambda: self.apply_requested.emit("export_competition")
        )
        g2.addWidget(btn_competition)

        btn_judge = QPushButton("突出显示 final_graph（裁判查看）")
        btn_judge.setCheckable(True)
        btn_judge.setToolTip(
            "ON：只显示原始影像 + final_graph\n"
            "OFF：恢复进入前的图层显示状态\n"
            "不修改 final_graph.json 等任何数据"
        )
        btn_judge.clicked.connect(
            lambda: self.apply_requested.emit("judge_view_toggle")
        )
        g2.addWidget(btn_judge)
        self._judge_view_btn = btn_judge

        btn_export_judge = QPushButton("导出裁判查看图")
        btn_export_judge.setToolTip(
            "导出 judge_final_graph_overlay.png（影像 + 高亮 final_graph）"
        )
        btn_export_judge.clicked.connect(
            lambda: self.apply_requested.emit("export_judge_overlay")
        )
        g2.addWidget(btn_export_judge)

        btn_debug = QPushButton("🧪 导出调试图")
        btn_debug.clicked.connect(lambda: self.apply_requested.emit("export_debug"))
        g2.addWidget(btn_debug)

        self._add_advanced_toggle()
        self._start_advanced()

        ag = self._add_group("导出选项")
        self._add_check(ag, "保存中间结果", "visualization.save_intermediate", True)
        self._add_check(ag, "显示车辆航点", "visualization.show_vehicle_waypoints", True)
        self._add_double(ag, "叠加透明度", "visualization.overlay_alpha", 0.45, 0.1, 1.0, 0.05)
        self._add_double(ag, "规划路径线宽(px)", "visualization.planned_path_width", 5.0, 1.0, 12.0, 1.0, 0)
        self._add_double(ag, "方向箭头间隔(px)", "visualization.arrow_spacing_px", 80.0, 20.0, 500.0, 10.0, 0)
        self._add_double(ag, "方向箭头大小(px)", "visualization.arrow_size_px", 12.0, 4.0, 40.0, 1.0, 0)

        wp = self._add_group("自适应航点重采样")
        self._add_double(wp, "直线航点间距(m)", "waypoints.straight_spacing_m", 10.0, 2.0, 30.0, 1.0, 1)
        self._add_double(wp, "弯道航点间距(m)", "waypoints.curve_spacing_m", 2.0, 1.0, 15.0, 0.5, 1)
        self._add_double(wp, "急弯间距(m)", "waypoints.sharp_turn_spacing_m", 2.0, 1.0, 10.0, 0.5, 1)
        self._add_double(wp, "路口航点间距(m)", "waypoints.intersection_spacing_m", 2.0, 1.0, 10.0, 0.5, 1)
        self._add_double(wp, "任务点附近间距(m)", "waypoints.task_point_spacing_m", 2.0, 1.0, 10.0, 0.5, 1)
        self._add_double(wp, "弯道角度阈值(°)", "waypoints.corner_angle_threshold_deg", 15.0, 5.0, 90.0, 1.0, 1)
        self._add_double(wp, "急弯阈值(°)", "waypoints.sharp_turn_angle_threshold_deg", 35.0, 10.0, 150.0, 1.0, 1)
        self._add_double(wp, "弯道加密缓冲(m)", "waypoints.corner_buffer_m", 5.0, 1.0, 30.0, 1.0, 1)
        self._add_double(wp, "路口加密半径(m)", "waypoints.intersection_buffer_m", 8.0, 1.0, 40.0, 1.0, 1)
        self._add_double(wp, "任务点缓冲(m)", "waypoints.task_point_buffer_m", 5.0, 1.0, 30.0, 1.0, 1)
        self._add_double(wp, "最大弦误差(m)", "waypoints.max_chord_error_m", 1.0, 0.3, 10.0, 0.1, 1)
        self._add_double(wp, "最小道路支撑比", "waypoints.min_mask_support_ratio", 0.75, 0.4, 1.0, 0.05, 2)
        self._add_double(wp, "最小航点间距(m)", "waypoints.min_waypoint_spacing_m", 1.0, 0.5, 10.0, 0.5, 1)
        self._add_double(wp, "最大允许点距(m)", "waypoints.max_waypoint_spacing_m", 12.0, 2.0, 40.0, 1.0, 1)
        self._add_double(wp, "硬失败点距(m)", "waypoints.hard_fail_spacing_m", 20.0, 12.0, 100.0, 1.0, 1)
        self._add_check(wp, "允许长直线(>12m)", "waypoints.allow_long_straight", False)

        self._end_advanced()

    def update_task_points_table(self, task_points, snapped_points=None):
        """刷新路径规划页任务点摘要表。"""
        if not hasattr(self, "_task_points_table"):
            return
        points = list(task_points or [])
        if not points:
            self._task_points_table.setText("暂无任务点")
            return
        snap_map = {}
        for sp in snapped_points or []:
            snap_map[int(getattr(sp, "seq", -1))] = sp
        type_name = {0: "START", 1: "GOAL", 2: "VIA"}
        lines = [
            "seq | type | lon | lat | alt | px | py | snap | dist"
        ]
        for tp in sorted(points, key=lambda p: int(p.seq)):
            sp = snap_map.get(int(tp.seq))
            snap_st = getattr(tp, "snap_status", "") or (
                getattr(sp, "status", "-") if sp else "-"
            )
            snap_d = getattr(tp, "snap_distance", None)
            if snap_d is None and sp is not None:
                snap_d = getattr(sp, "snap_distance", None)
            dist_s = f"{float(snap_d):.1f}" if snap_d is not None else "-"
            lon = tp.longitude
            lat = tp.latitude
            lon_s = f"{lon:.5f}" if lon is not None else "-"
            lat_s = f"{lat:.5f}" if lat is not None else "-"
            px = tp.pixel_x
            py = tp.pixel_y
            px_s = f"{px:.0f}" if px is not None else "-"
            py_s = f"{py:.0f}" if py is not None else "-"
            lines.append(
                f"{tp.seq} | {type_name.get(int(tp.point_type), '?')} | "
                f"{lon_s} | {lat_s} | {float(tp.altitude):.1f} | "
                f"{px_s} | {py_s} | {snap_st} | {dist_s}"
            )
        self._task_points_table.setText("\n".join(lines))

    def update_vwp_status(self, status) -> None:
        """Update vehicle-waypoint pipeline status panel."""
        if not hasattr(self, "_vwp_status_label"):
            return
        if status is None:
            self._vwp_status_label.setText("主流程状态：未开始")
            return
        if isinstance(status, dict):
            d = status
        else:
            d = status.to_dict() if hasattr(status, "to_dict") else {}
        flags = [
            ("graph_valid", d.get("graph_valid")),
            ("task_points_loaded", d.get("task_points_loaded")),
            ("dense_path_generated", d.get("dense_path_generated")),
            ("vehicle_waypoints_generated", d.get("vehicle_waypoints_generated")),
            ("waypoints_checked", d.get("waypoints_checked")),
            ("waypoints_repaired", d.get("waypoints_repaired")),
            ("yaml_exported", d.get("yaml_exported")),
        ]
        lines = ["主流程状态："]
        for name, ok in flags:
            mark = "✓" if ok else "·"
            lines.append(f"  {mark} {name}")
        if d.get("usable_for_vehicle"):
            lines.append("")
            lines.append("★ 可用于小车")
        elif d.get("message"):
            lines.append("")
            lines.append(str(d.get("message")))
        self._vwp_status_label.setText("\n".join(lines))

    # 角点颜色
    _CORNER_COLORS = {
        "top_left":     "#ff5555",  # 红色
        "top_right":    "#5588ff",  # 蓝色
        "bottom_left":  "#ffcc00",  # 黄色
        "bottom_right": "#cc55ff",  # 紫色
    }

    def _build_calibration_page(self):
        """坐标校准页面 — 多种标定模式。

        结构：
          - 校准状态区（模式选择 + 状态标签）
          - A 区：快速四角标定区（保留原有四角输入）
          - B 区：高级控制点标定区（文件导入 + 图上配准）
          - 校准操作区（计算 / 应用 / 保存 / 清空）
        """
        # ===================================================================
        # 校准状态
        # ===================================================================
        g_status = self._add_group("校准状态")

        # 标定方式选择
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("标定方式："))
        self._cal_method_combo = QComboBox()
        self._cal_method_combo.addItems([
            "手动输入四角坐标",
            "导入四角坐标文件",
            "三点图片顶点校准",
            "导入控制点文件",
            "控制点图上配准",
        ])
        self._cal_method_combo.setToolTip(
            "选择标定方式。切换方式不会清空已完成的标定结果。"
        )
        self._cal_method_combo.currentTextChanged.connect(self._on_cal_method_changed)
        mode_row.addWidget(self._cal_method_combo)
        g_status.addLayout(mode_row)

        self._cal_status_label = QLabel("⚠ 未校准")
        self._cal_status_label.setStyleSheet("color: #f9e2af; font-weight: bold;")
        g_status.addWidget(self._cal_status_label)

        self._cal_vertex_summary = QLabel("")
        self._cal_vertex_summary.setWordWrap(True)
        self._cal_vertex_summary.setStyleSheet(
            "color: #a6e3a1; font-size: 10px; padding: 4px;"
        )
        self._cal_vertex_summary.setVisible(False)
        g_status.addWidget(self._cal_vertex_summary)

        self._cal_mode_label = QLabel("转换模式：--")
        self._cal_mode_label.setStyleSheet("color: #89b4fa;")
        g_status.addWidget(self._cal_mode_label)

        self._cal_cp_count_label = QLabel("控制点数量：0")
        g_status.addWidget(self._cal_cp_count_label)

        self._cal_res_label = QLabel("估计分辨率：--")
        g_status.addWidget(self._cal_res_label)

        self._cal_rms_label = QLabel("RMS 误差：--")
        g_status.addWidget(self._cal_rms_label)

        self._cal_image_size_label = QLabel("图像尺寸：--")
        g_status.addWidget(self._cal_image_size_label)

        # ===================================================================
        # A 区：快速四角标定区（★ 保留原有全部功能）
        # ===================================================================
        g_area_a = self._add_group("A 区：快速四角标定")

        # ---- 导入坐标文件 ----	
        g_import = QGroupBox("导入坐标文件")
        g_import.setStyleSheet("QGroupBox { font-weight: bold; color: #89b4fa; margin-top: 6px; }")
        g_import_lay = QVBoxLayout(g_import)

        btn_import_vertex = QPushButton("🏁 导入图片顶点校准文件")
        btn_import_vertex.setObjectName("primary")
        btn_import_vertex.setToolTip(
            "导入比赛校准文件（序号;经度;纬度;高程）\n"
            "1=左下 2=右下 3=左上，自动绑定像素并完成三点仿射校准"
        )
        btn_import_vertex.clicked.connect(
            lambda: self.apply_requested.emit("cal_import_vertex_txt")
        )
        g_import_lay.addWidget(btn_import_vertex)

        btn_import_txt = QPushButton("📄 导入坐标 TXT")
        btn_import_txt.setObjectName("primary")
        btn_import_txt.setToolTip("导入包含 3~4 个顶点经纬度的 TXT 文件（推荐）")
        btn_import_txt.clicked.connect(lambda: self.apply_requested.emit("cal_import_txt"))
        g_import_lay.addWidget(btn_import_txt)

        btn_import_csv = QPushButton("📊 导入 CSV / JSON")
        btn_import_csv.setToolTip("也可导入 CSV 或 JSON 格式的控制点文件")
        btn_import_csv.clicked.connect(lambda: self.apply_requested.emit("cal_import_other"))
        g_import_lay.addWidget(btn_import_csv)

        btn_import_corners = QPushButton("📍 导入 corners.json")
        btn_import_corners.setObjectName("primary")
        btn_import_corners.setToolTip("导入四角坐标 JSON，自动填入 TL/TR/BR/BL 经纬度并勾选")
        btn_import_corners.clicked.connect(lambda: self.apply_requested.emit("cal_import_corners"))
        g_import_lay.addWidget(btn_import_corners)

        g_area_a.addWidget(g_import)

        # ---- 快捷选择按钮（★ 保留原有）----	
        g_quick = QGroupBox("快捷选择顶点组合")
        g_quick.setStyleSheet("QGroupBox { font-weight: bold; color: #89b4fa; margin-top: 6px; }")
        g_quick_lay = QVBoxLayout(g_quick)
        quick_presets = [
            ("1", "左上-右上-左下", "TL+TR+BL"),
            ("2", "左上-右上-右下", "TL+TR+BR"),
            ("3", "左上-左下-右下", "TL+BL+BR"),
            ("4", "右上-左下-右下", "TR+BL+BR"),
            ("all", "全部四个顶点", "TL+TR+BL+BR"),
        ]
        for pid, text, tooltip in quick_presets:
            btn = QPushButton(f"📍 {text}")
            btn.setToolTip(tooltip)
            btn.setStyleSheet("font-size: 10px; padding: 2px 6px;")
            btn.clicked.connect(lambda checked=False, p=pid: self.apply_requested.emit(f"cal_preset_{p}"))
            g_quick_lay.addWidget(btn)
        g_area_a.addWidget(g_quick)

        # ---- 四个顶点勾选 + 经纬度输入（★ 保留原有，代码不变）----	
        g_corners = QGroupBox("图片顶点选择（至少勾选 3 个）")
        g_corners.setStyleSheet("QGroupBox { font-weight: bold; color: #f9e2af; margin-top: 6px; }")
        g_corners_lay = QVBoxLayout(g_corners)
        self._cal_corner_widgets = {}
        corner_short = {
            "top_left": "TL", "top_right": "TR",
            "bottom_left": "BL", "bottom_right": "BR",
        }

        for cname in CORNERS_ORDER:
            info = CORNERS_DEF[cname]
            ws = {}
            self._cal_corner_widgets[cname] = ws

            # 带分隔线的角点行
            row_frame = QFrame()
            row_frame.setFrameShape(QFrame.Shape.StyledPanel)
            row_layout = QVBoxLayout(row_frame)
            row_layout.setContentsMargins(4, 2, 4, 2)
            row_layout.setSpacing(2)

            # 勾选框 + 标签 + 像素坐标
            top_row = QHBoxLayout()
            cb = QCheckBox(f"{info['label']}  [{corner_short.get(cname, '?')}]")
            cb.setToolTip(f"图片{cname}，像素坐标由图像尺寸自动确定")
            cb.stateChanged.connect(lambda state, cn=cname: self._on_corner_check_changed(cn))
            ws["check"] = cb
            top_row.addWidget(cb)
            top_row.addStretch()
            row_layout.addLayout(top_row)

            # 像素坐标（自动显示）
            pix_label = QLabel("pixel: (--, --)")
            pix_label.setStyleSheet(f"color: {self._CORNER_COLORS.get(cname, '#aaa')}; font-size: 9px; padding-left: 20px;")
            ws["pixel"] = pix_label
            row_layout.addWidget(pix_label)

            # 经度 lon（-180 ~ 180，支持 117.xxx 三位整数）
            lon_layout = QHBoxLayout()
            lon_layout.addWidget(QLabel("经度 lon:"))
            lon_spin = QDoubleSpinBox()
            lon_spin.setRange(-180.0, 180.0)
            lon_spin.setDecimals(8)
            lon_spin.setSingleStep(0.000001)
            lon_spin.setValue(0.0)
            lon_spin.setMinimumWidth(140)
            lon_spin.setToolTip("经度 Longitude，范围 -180 ~ 180，如 117.12345678")
            lon_spin.setEnabled(False)
            lon_spin.valueChanged.connect(lambda v: self.apply_requested.emit("cal_update_settings"))
            lon_layout.addWidget(lon_spin)
            lon_layout.addStretch()
            ws["lon"] = lon_spin
            row_layout.addLayout(lon_layout)

            # 纬度 lat（-90 ~ 90）
            lat_layout = QHBoxLayout()
            lat_layout.addWidget(QLabel("纬度 lat:"))
            lat_spin = QDoubleSpinBox()
            lat_spin.setRange(-90.0, 90.0)
            lat_spin.setDecimals(8)
            lat_spin.setSingleStep(0.000001)
            lat_spin.setValue(0.0)
            lat_spin.setMinimumWidth(140)
            lat_spin.setToolTip("纬度 Latitude，范围 -90 ~ 90，如 31.12345678")
            lat_spin.setEnabled(False)
            lat_spin.valueChanged.connect(lambda v: self.apply_requested.emit("cal_update_settings"))
            lat_layout.addWidget(lat_spin)
            lat_layout.addStretch()
            ws["lat"] = lat_spin
            row_layout.addLayout(lat_layout)

            g_corners_lay.addWidget(row_frame)

        # ---- 坐标输入示例提示 ----
        hint_label = QLabel(
            "💡 中国区域常见格式示例：\n"
            "   经度 lon：117.12345678\n"
            "   纬度 lat：31.12345678\n"
            "⚠ 注意：117.xx 是经度，不是纬度。请勿填反！"
        )
        hint_label.setStyleSheet(
            "color: #f9e2af; font-size: 10px; padding: 6px 4px; "
            "background-color: rgba(255,255,255,0.05); border-radius: 4px;"
        )
        hint_label.setWordWrap(True)
        g_corners_lay.addWidget(hint_label)

        g_area_a.addWidget(g_corners)

        # ===================================================================
        # B 区：高级控制点标定区（★ 新增）
        # ===================================================================
        g_area_b = self._add_group("B 区：高级控制点标定")
        self._cal_area_b = g_area_b  # 用于后续动态更新

        # 导入控制点文件（含仅 lon/lat 的模式）
        g_b_import = QGroupBox("控制点文件导入")
        g_b_import.setStyleSheet("QGroupBox { font-weight: bold; color: #89b4fa; margin-top: 6px; }")
        g_b_import_lay = QVBoxLayout(g_b_import)

        btn_import_cp = QPushButton("📁 导入控制点文件")
        btn_import_cp.setObjectName("primary")
        btn_import_cp.setToolTip(
            "支持格式：\n"
            "• 完整格式：点号,像素X,像素Y,经度,纬度 → 直接计算\n"
            "• 仅坐标：点号,经度,纬度 → 进入图上点击配准模式"
        )
        btn_import_cp.clicked.connect(lambda: self.apply_requested.emit("cal_import_cp_file"))
        g_b_import_lay.addWidget(btn_import_cp)

        g_area_b.addWidget(g_b_import)

        # 图上点击配准
        g_b_click = QGroupBox("控制点图上配准")
        g_b_click.setStyleSheet("QGroupBox { font-weight: bold; color: #89b4fa; margin-top: 6px; }")
        g_b_click_lay = QVBoxLayout(g_b_click)

        btn_start_click = QPushButton("🎯 开始图上点击配准")
        btn_start_click.setToolTip(
            "进入图上点击模式后，在图像上依次点击控制点位置。\n"
            "需先导入 lon/lat 控制点文件，或手动输入点号+经纬度。"
        )
        btn_start_click.clicked.connect(lambda: self.apply_requested.emit("cal_start_map_click"))
        g_b_click_lay.addWidget(btn_start_click)

        self._cal_map_click_status_label = QLabel("状态：就绪")
        self._cal_map_click_status_label.setStyleSheet("color: #89b4fa; font-size: 10px;")
        g_b_click_lay.addWidget(self._cal_map_click_status_label)

        g_area_b.addWidget(g_b_click)

        # 控制点列表（滚动区域）
        g_b_list = QGroupBox("控制点列表")
        g_b_list.setStyleSheet("QGroupBox { font-weight: bold; color: #f9e2af; margin-top: 6px; }")
        g_b_list_lay = QVBoxLayout(g_b_list)

        self._cal_cp_list_scroll = QScrollArea()
        self._cal_cp_list_scroll.setWidgetResizable(True)
        self._cal_cp_list_scroll.setMaximumHeight(200)
        self._cal_cp_list_container = QWidget()
        self._cal_cp_list_layout = QVBoxLayout(self._cal_cp_list_container)
        self._cal_cp_list_layout.setContentsMargins(2, 2, 2, 2)
        self._cal_cp_list_scroll.setWidget(self._cal_cp_list_container)
        g_b_list_lay.addWidget(self._cal_cp_list_scroll)

        g_area_b.addWidget(g_b_list)

        # residual 汇总显示
        self._cal_cp_residual_label = QLabel("")
        self._cal_cp_residual_label.setStyleSheet("color: #89b4fa; font-size: 10px; padding: 4px;")
        self._cal_cp_residual_label.setWordWrap(True)
        g_area_b.addWidget(self._cal_cp_residual_label)

        # ===================================================================
        # 校准操作（共用）
        # ===================================================================
        g_actions = self._add_group("校准操作")
        btn_compute = QPushButton("📐 计算坐标变换")
        btn_compute.setObjectName("primary")
        btn_compute.clicked.connect(lambda: self.apply_requested.emit("cal_compute"))
        g_actions.addWidget(btn_compute)

        btn_apply = QPushButton("🗺 应用到路网")
        btn_apply.setToolTip("将校准结果应用到 final_graph，生成校准后路网文件")
        btn_apply.clicked.connect(lambda: self.apply_requested.emit("cal_apply_graph"))
        g_actions.addWidget(btn_apply)

        btn_save = QPushButton("💾 保存校准")
        btn_save.clicked.connect(lambda: self.apply_requested.emit("cal_save"))
        g_actions.addWidget(btn_save)

        btn_clear = QPushButton("🗑 清空校准")
        btn_clear.clicked.connect(lambda: self.apply_requested.emit("cal_clear"))
        g_actions.addWidget(btn_clear)

    def _on_corner_check_changed(self, corner_name: str):
        """角点勾选变化时，启用/禁用对应的 lon/lat 输入框，并更新像素坐标显示。"""
        ws = self._cal_corner_widgets.get(corner_name)
        if not ws:
            return
        checked = ws["check"].isChecked()
        ws["lon"].setEnabled(checked)
        ws["lat"].setEnabled(checked)

        # 立即更新像素坐标标签（从图像尺寸自动推断）
        if checked:
            w, h = 0, 0
            try:
                parent = self.window()
                if hasattr(parent, '_layer_manager'):
                    w, h = parent._layer_manager.image_size
            except Exception:
                pass
            if w > 0 and h > 0:
                from roadnet.gcp_io import infer_pixel_from_corner_name
                px = infer_pixel_from_corner_name(corner_name, w, h)
                if px:
                    ws["pixel"].setText(f"pixel: ({px[0]}, {px[1]})")
                else:
                    ws["pixel"].setText("pixel: (--, --)")
            else:
                ws["pixel"].setText("pixel: (--, --) [无影像]")
        else:
            ws["pixel"].setText("pixel: (--, --)")

        self.apply_requested.emit("cal_update_settings")

    def _on_cal_method_changed(self, text: str):
        """标定方式下拉变化时，同步设置 GeoCalibration.method 元数据。"""
        method_map = {
            "手动输入四角坐标": "corner_manual",
            "导入四角坐标文件": "corner_file",
            "三点图片顶点校准": "image_corner_3point_affine",
            "导入控制点文件": "control_points_file",
            "控制点图上配准": "control_points_manual",
        }
        method = method_map.get(text, "")
        # 通知主窗口更新 method
        self.apply_requested.emit(f"cal_set_method::{method}")

    def set_calibration_method_combo(self, method: str):
        """根据 method 字符串设置下拉框（不触发 signal）。"""
        if not hasattr(self, '_cal_method_combo'):
            return
        method_labels = {
            "corner_manual": "手动输入四角坐标",
            "corner_file": "导入四角坐标文件",
            "image_corner_3point_affine": "三点图片顶点校准",
            "control_points_file": "导入控制点文件",
            "control_points_manual": "控制点图上配准",
        }
        label = method_labels.get(method, "手动输入四角坐标")
        self._cal_method_combo.blockSignals(True)
        self._cal_method_combo.setCurrentText(label)
        self._cal_method_combo.blockSignals(False)

    def get_selected_calibration_method(self) -> str:
        """获取当前选中的标定方式字符串。"""
        if not hasattr(self, '_cal_method_combo'):
            return "corner_manual"
        method_map = {
            "手动输入四角坐标": "corner_manual",
            "导入四角坐标文件": "corner_file",
            "三点图片顶点校准": "image_corner_3point_affine",
            "导入控制点文件": "control_points_file",
            "控制点图上配准": "control_points_manual",
        }
        return method_map.get(self._cal_method_combo.currentText(), "corner_manual")

    def set_vertex_calibration_summary(self, text: str):
        """显示三点图片顶点校准摘要。"""
        if not hasattr(self, "_cal_vertex_summary"):
            return
        if text:
            self._cal_vertex_summary.setText(text)
            self._cal_vertex_summary.setVisible(True)
        else:
            self._cal_vertex_summary.setText("")
            self._cal_vertex_summary.setVisible(False)

    def update_calibration_ui(self, geo_calibration):
        """根据 GeoCalibration 对象更新校准面板。"""
        if not hasattr(self, '_cal_status_label'):
            return

        if geo_calibration is None:
            return

        # 状态
        if geo_calibration.enabled:
            mode_label = geo_calibration.get_mode_label()
            self._cal_status_label.setText(f"✅ {mode_label}")
            self._cal_status_label.setStyleSheet("color: #50fa7b; font-weight: bold;")
        else:
            self._cal_status_label.setText("⚠ 未校准")
            self._cal_status_label.setStyleSheet("color: #f9e2af; font-weight: bold;")

        # 标定方式（同步 combo box）
        if geo_calibration.method:
            self.set_calibration_method_combo(geo_calibration.method)

        # 转换模式
        mode_text = geo_calibration.transform_mode or "--"
        self._cal_mode_label.setText(f"转换模式：{mode_text}")

        # 控制点数量
        cp_count = len(geo_calibration.control_points) if geo_calibration.control_points else 0
        self._cal_cp_count_label.setText(f"控制点数量：{cp_count}")

        # 估计分辨率
        if geo_calibration.pixel_resolution_estimated_m is not None:
            self._cal_res_label.setText(f"估计分辨率：{geo_calibration.pixel_resolution_estimated_m:.4f} m/px")
        else:
            self._cal_res_label.setText("估计分辨率：--")

        # RMS 误差
        rms_val = getattr(geo_calibration, 'rms_error', None)
        if rms_val is not None:
            self._cal_rms_label.setText(f"RMS 误差：{rms_val:.3f} m")
        else:
            self._cal_rms_label.setText("RMS 误差：--")

        # 图像尺寸（从主窗口获取）
        w, h = 0, 0
        try:
            parent = self.window()
            if hasattr(parent, '_layer_manager'):
                lm = parent._layer_manager
                w, h = lm.image_size
                if w > 0:
                    self._cal_image_size_label.setText(f"图像尺寸：{w} × {h}")
        except Exception:
            pass

        # 确保所有角点像素标签都根据图像尺寸更新（无论是否有控制点数据）
        self._init_corner_pixel_labels(w, h)

        # 刷新四角点勾选和经纬度
        self._sync_corner_widgets_from_geo(geo_calibration)

        # 刷新控制点列表（B区）
        self._refresh_cp_list(geo_calibration)

        # 三点图片顶点校准摘要（从已保存 metadata 恢复）
        if getattr(geo_calibration, "calibration_mode", "") == "image_corner_3point_affine" or (
            getattr(geo_calibration, "method", "") == "image_corner_3point_affine"
        ):
            cps = getattr(geo_calibration, "corner_points", None) or []
            inferred = getattr(geo_calibration, "inferred_corners", None) or []
            by_id = {str(c.get("id")): c for c in cps}
            c1, c2, c3 = by_id.get("1"), by_id.get("2"), by_id.get("3")
            rt = inferred[0] if inferred else None
            if c1 and c2 and c3:
                lines = [
                    "校准模式：三点图片顶点校准",
                    f"1 左下角：{c1.get('longitude')}, {c1.get('latitude')}, "
                    f"pixel=({c1.get('pixel_x')},{c1.get('pixel_y')})",
                    f"2 右下角：{c2.get('longitude')}, {c2.get('latitude')}, "
                    f"pixel=({c2.get('pixel_x')},{c2.get('pixel_y')})",
                    f"3 左上角：{c3.get('longitude')}, {c3.get('latitude')}, "
                    f"pixel=({c3.get('pixel_x')},{c3.get('pixel_y')})",
                ]
                if rt:
                    lines.append(
                        f"4 右上角：自动推算 {rt.get('longitude')}, {rt.get('latitude')}, "
                        f"pixel=({rt.get('pixel_x')},{rt.get('pixel_y')})"
                    )
                else:
                    lines.append("4 右上角：自动推算")
                lines.append("已根据图片顶点坐标自动完成三点仿射校准。")
                self.set_vertex_calibration_summary("\n".join(lines))
        elif hasattr(self, "_cal_vertex_summary"):
            # 非该模式时不强制清空（避免手动校准时闪烁）；仅在未校准时清空
            if not geo_calibration.enabled:
                self.set_vertex_calibration_summary("")

    def _refresh_cp_list(self, geo_calibration):
        """刷新 B 区控制点列表显示。"""
        if not hasattr(self, '_cal_cp_list_layout'):
            return

        # 清除旧列表
        while self._cal_cp_list_layout.count():
            item = self._cal_cp_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        cps = geo_calibration.control_points if geo_calibration else []
        if not cps:
            placeholder = QLabel("暂无控制点，请导入文件或图上点击配准。")
            placeholder.setStyleSheet("color: #6c7086; font-size: 10px; padding: 4px;")
            self._cal_cp_list_layout.addWidget(placeholder)
            self._cal_cp_residual_label.setText("")
            return

        # 计算每个控制点的 residual（如果已有矩阵）
        residuals = []
        if geo_calibration.pixel_to_world_matrix is not None:
            import numpy as np
            for cp in cps:
                u, v = cp.get("pixel", [0, 0])
                pred_x, pred_y = geo_calibration.pixel_to_world(u, v)
                actual_x = cp.get("x_meter", 0)
                actual_y = cp.get("y_meter", 0)
                err = np.sqrt((pred_x - actual_x) ** 2 + (pred_y - actual_y) ** 2)
                residuals.append((cp.get("name", "?"), err))

        for i, cp in enumerate(cps):
            name = cp.get("name", f"CP{i+1}")
            px = cp.get("pixel", [None, None])
            lon = cp.get("lon", 0)
            lat = cp.get("lat", 0)

            has_pixel = px[0] is not None and px[1] is not None

            if has_pixel:
                rms_str = ""
                for rn, re in residuals:
                    if rn == name:
                        rms_str = f"  |  残差: {re:.3f} m"
                        break
                text = f"• {name}: px=({px[0]:.0f},{px[1]:.0f}), lon={lon:.6f}, lat={lat:.6f}{rms_str}"
            else:
                text = f"• {name}: ⚠ 待配准 (lon={lon:.6f}, lat={lat:.6f})"

            label = QLabel(text)
            label.setStyleSheet("color: #cdd6f4; font-size: 10px; padding: 1px 4px;")
            self._cal_cp_list_layout.addWidget(label)

        # 显示 RMS 汇总
        if geo_calibration.rms_error is not None:
            self._cal_cp_residual_label.setText(
                f"RMS 误差（米）：{geo_calibration.rms_error:.4f}"
            )
        elif residuals:
            import numpy as np
            rms = float(np.sqrt(np.mean(np.array([e for _, e in residuals]) ** 2)))
            self._cal_cp_residual_label.setText(
                f"RMS 误差（米）：{rms:.4f}"
            )
        else:
            self._cal_cp_residual_label.setText("")

    def _init_corner_pixel_labels(self, w: int, h: int):
        """根据图像尺寸初始化所有角点的像素坐标标签（无论是否勾选）。"""
        if not hasattr(self, '_cal_corner_widgets') or w <= 0 or h <= 0:
            return
        from roadnet.gcp_io import infer_pixel_from_corner_name
        for cname in CORNERS_ORDER:
            ws = self._cal_corner_widgets.get(cname)
            if not ws:
                continue
            px = infer_pixel_from_corner_name(cname, w, h)
            if px:
                ws["pixel"].setText(f"pixel: ({px[0]}, {px[1]})")

    def _sync_corner_widgets_from_geo(self, geo_calibration):
        """将 GeoCalibration 中的控制点同步到四角点 UI。"""
        if not hasattr(self, '_cal_corner_widgets'):
            return

        w, h = 0, 0
        try:
            parent = self.window()
            if hasattr(parent, '_layer_manager'):
                w, h = parent._layer_manager.image_size
        except Exception:
            pass

        # 建立 name -> cp 映射
        cp_map = {}
        for cp in (geo_calibration.control_points or []):
            name = cp.get("name", "")
            if name:
                cp_map[name] = cp

        for cname in CORNERS_ORDER:
            ws = self._cal_corner_widgets.get(cname)
            if not ws:
                continue

            # 像素坐标（自动从图像尺寸推断）
            if w > 0 and h > 0:
                px = infer_pixel_from_corner_name(cname, w, h)
                if px:
                    ws["pixel"].setText(f"pixel: ({px[0]}, {px[1]})")

            cp = cp_map.get(cname)
            if cp:
                # 有数据 → 勾选 + 填入经纬度
                ws["check"].blockSignals(True)
                ws["check"].setChecked(True)
                ws["check"].blockSignals(False)
                ws["lon"].blockSignals(True)
                ws["lon"].setEnabled(True)
                ws["lon"].setValue(cp.get("lon", 0))
                ws["lon"].blockSignals(False)
                ws["lat"].blockSignals(True)
                ws["lat"].setEnabled(True)
                ws["lat"].setValue(cp.get("lat", 0))
                ws["lat"].blockSignals(False)
            else:
                # 无数据 → 取消勾选
                ws["check"].blockSignals(True)
                ws["check"].setChecked(False)
                ws["check"].blockSignals(False)
                ws["lon"].blockSignals(True)
                ws["lon"].setEnabled(False)
                ws["lon"].setValue(0.0)
                ws["lon"].blockSignals(False)
                ws["lat"].blockSignals(True)
                ws["lat"].setEnabled(False)
                ws["lat"].setValue(0.0)
                ws["lat"].blockSignals(False)

    def get_calibration_control_points(self, image_size: tuple = None) -> list:
        """从面板获取用户勾选的顶点控制点。

        Returns:
            control_points list（仅包含勾选且填入经纬度的顶点），或 None（如果选不够 3 个）
        """
        if not hasattr(self, '_cal_corner_widgets'):
            return None

        w, h = 0, 0
        try:
            parent = self.window()
            if hasattr(parent, '_layer_manager'):
                w, h = parent._layer_manager.image_size
        except Exception:
            pass
        if image_size:
            w, h = image_size

        cps = []
        for cname in CORNERS_ORDER:
            ws = self._cal_corner_widgets.get(cname)
            if not ws:
                continue
            if not ws["check"].isChecked():
                continue
            lon = ws["lon"].value()
            lat = ws["lat"].value()

            # 像素坐标自动推断
            if w > 0 and h > 0:
                px = infer_pixel_from_corner_name(cname, w, h)
            else:
                px = (0, 0)

            cps.append({
                "name": cname,
                "pixel": [px[0], px[1]] if px else [0, 0],
                "lon": round(lon, 8),
                "lat": round(lat, 8),
            })

        if len(cps) < 3:
            return None

        return cps

    def detect_lon_lat_swap(self, control_points: list = None) -> str:
        """检测控制点的经纬度是否存在填反嫌疑。

        Args:
            control_points: 控制点列表，如果为 None 则自动从面板读取。

        Returns:
            警告信息字符串，如果无异常则返回空字符串。
        """
        if control_points is None:
            control_points = self.get_calibration_control_points()
        if not control_points:
            return ""

        swap_suspects = []
        for cp in control_points:
            lon = cp.get("lon", 0)
            lat = cp.get("lat", 0)
            name = cp.get("name", "?")

            # 纬度超出范围
            if lat > 90 or lat < -90:
                swap_suspects.append(
                    f"控制点 {name} 的纬度 lat={lat:.6f} 超出合法范围 [-90, 90]，"
                    f"请检查是否把经度 lon 和纬度 lat 填反。"
                )
                continue

            # 疑似填反：lon 在 -90~90（像纬度），lat 在 100~130（像经度，中国区域）
            if -90 <= lon <= 90 and 100 <= abs(lat) <= 130:
                swap_suspects.append(
                    f"控制点 {name} 坐标疑似经纬度填反："
                    f"lon={lon:.6f}（范围在 -90~90，更像纬度），"
                    f"lat={lat:.6f}（范围在 100~130，更像经度）。\n"
                    f"    → 如果您的坐标是 {lat:.6f}, {lon:.6f}，请将 {lat:.6f} 填入「经度 lon」，"
                    f"{lon:.6f} 填入「纬度 lat」。"
                )

        if swap_suspects:
            return "⚠ 检测到以下经纬度填反嫌疑：\n\n" + "\n\n".join(swap_suspects)

        return ""

    def set_calibration_corners_from_data(self, control_points: list, image_size: tuple = None):
        """从导入的数据填充四角点 UI（由 TXT 导入等触发）。

        Args:
            control_points: 控制点列表 [{"name": "top_left", "lon": ..., "lat": ...}]
            image_size: (w, h)
        """
        if not hasattr(self, '_cal_corner_widgets'):
            return

        w, h = 0, 0
        if image_size:
            w, h = image_size
        else:
            try:
                parent = self.window()
                if hasattr(parent, '_layer_manager'):
                    w, h = parent._layer_manager.image_size
            except Exception:
                pass

        cp_map = {}
        for cp in control_points:
            name = cp.get("name", "")
            if name:
                cp_map[name] = cp

        for cname in CORNERS_ORDER:
            ws = self._cal_corner_widgets.get(cname)
            if not ws:
                continue

            # 像素坐标
            if w > 0 and h > 0:
                px = infer_pixel_from_corner_name(cname, w, h)
                if px:
                    ws["pixel"].setText(f"pixel: ({px[0]}, {px[1]})")

            cp = cp_map.get(cname)
            if cp:
                ws["check"].blockSignals(True)
                ws["check"].setChecked(True)
                ws["check"].blockSignals(False)
                ws["lon"].blockSignals(True)
                ws["lon"].setEnabled(True)
                ws["lon"].setValue(cp.get("lon", 0))
                ws["lon"].blockSignals(False)
                ws["lat"].blockSignals(True)
                ws["lat"].setEnabled(True)
                ws["lat"].setValue(cp.get("lat", 0))
                ws["lat"].blockSignals(False)
            else:
                ws["check"].blockSignals(True)
                ws["check"].setChecked(False)
                ws["check"].blockSignals(False)
                ws["lon"].blockSignals(True)
                ws["lon"].setEnabled(False)
                ws["lon"].setValue(0.0)
                ws["lon"].blockSignals(False)
                ws["lat"].blockSignals(True)
                ws["lat"].setEnabled(False)
                ws["lat"].setValue(0.0)
                ws["lat"].blockSignals(False)

    # ===================================================================
    # 高级设置折叠
    # ===================================================================

    def _add_advanced_toggle(self):
        self._adv_btn = QPushButton("▸ 高级设置")
        self._adv_btn.setObjectName("advanced-toggle")
        self._adv_btn.clicked.connect(self._toggle_advanced)
        self._page_layout.addWidget(self._adv_btn)

    def _start_advanced(self):
        """高级设置开始标记"""
        pass

    def _end_advanced(self):
        """标记之后添加的 widget 为高级设置，默认隐藏"""
        # 获取 _adv_btn 之后的所有 widget
        idx = self._page_layout.indexOf(self._adv_btn)
        for i in range(idx + 1, self._page_layout.count()):
            item = self._page_layout.itemAt(i)
            if item and item.widget():
                item.widget().setVisible(False)
                self._advanced_widgets.append(item.widget())

    def _toggle_advanced(self):
        self._is_advanced_open = not self._is_advanced_open
        for w in self._advanced_widgets:
            try:
                w.setVisible(self._is_advanced_open)
            except RuntimeError:
                pass
        self._adv_btn.setText("▾ 高级设置" if self._is_advanced_open else "▸ 高级设置")

    # ===================================================================
    # Widget 构建辅助
    # ===================================================================

    def _add_group(self, title: str = None, parent_layout=None) -> QVBoxLayout:
        """添加分组框，返回内部布局"""
        if parent_layout is None:
            parent_layout = self._page_layout
        group = QGroupBox(title) if title else QGroupBox()
        if title:
            group.setTitle(title)
        gl = QVBoxLayout(group)
        gl.setSpacing(4)
        parent_layout.addWidget(group)
        return gl

    def _add_spin(self, layout, label: str, key: str, default: int,
                  min_val: int = 0, max_val: int = 9999, step: int = 1):
        h = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(80)
        h.addWidget(lbl)
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setValue(self._get_config(key, default))
        spin.valueChanged.connect(lambda v, k=key: self._on_value_changed(k, v))
        h.addWidget(spin)
        h.addStretch()
        layout.addLayout(h)
        self._widgets[key] = spin
        return spin

    def _add_double(self, layout, label: str, key: str, default: float,
                    min_val: float = 0.0, max_val: float = 100.0, step: float = 0.1,
                    decimals: int = 2):
        h = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(80)
        h.addWidget(lbl)
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(self._get_config(key, default))
        spin.valueChanged.connect(lambda v, k=key: self._on_value_changed(k, v))
        h.addWidget(spin)
        h.addStretch()
        layout.addLayout(h)
        self._widgets[key] = spin
        return spin

    def _add_check(self, layout, label: str, key: str, default: bool = True):
        cb = QCheckBox(label)
        cb.setChecked(self._get_config(key, default))
        cb.stateChanged.connect(lambda state, k=key: self._on_value_changed(k, bool(state)))
        layout.addWidget(cb)
        self._widgets[key] = cb
        return cb

    def _add_line(self, layout, label: str, key: str, default: str = ""):
        h = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(80)
        h.addWidget(lbl)
        edit = QLineEdit()
        edit.setText(str(self._get_config(key, default)))
        edit.textChanged.connect(lambda v, k=key: self._on_value_changed(k, v))
        h.addWidget(edit)
        layout.addLayout(h)
        self._widgets[key] = edit
        return edit

    def _add_combo(self, layout, label: str, key: str, options: list,
                   default: str = ""):
        h = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(80)
        h.addWidget(lbl)
        combo = QComboBox()
        combo.addItems(options)
        current = self._get_config(key, default)
        if current in options:
            combo.setCurrentText(current)
        combo.currentTextChanged.connect(lambda v, k=key: self._on_value_changed(k, v))
        h.addWidget(combo)
        h.addStretch()
        layout.addLayout(h)
        self._widgets[key] = combo
        return combo

    # ===================================================================
    # 公共接口
    # ===================================================================

    def update_counts(self, pos: int = None, neg: int = None,
                      roi: int = None, ignore: int = None,
                      nodes: int = None, edges: int = None):
        """更新面板上的计数"""
        if pos is not None:
            lbl = getattr(self, '_pos_count_label', None)
            if lbl is not None and _shiboken_is_valid(lbl):
                lbl.setText(f"正样本：{pos}")
        if neg is not None:
            lbl = getattr(self, '_neg_count_label', None)
            if lbl is not None and _shiboken_is_valid(lbl):
                lbl.setText(f"负样本：{neg}")
        if roi is not None:
            lbl = getattr(self, '_roi_count_label', None)
            if lbl is not None and _shiboken_is_valid(lbl):
                lbl.setText(f"ROI 数量：{roi}")
        if ignore is not None:
            lbl = getattr(self, '_ignore_count_label', None)
            if lbl is not None and _shiboken_is_valid(lbl):
                lbl.setText(f"Ignore 数量：{ignore}")
        if nodes is not None:
            lbl = getattr(self, '_node_label', None)
            if lbl is not None and _shiboken_is_valid(lbl):
                lbl.setText(f"节点数量：{nodes}")
        if edges is not None:
            lbl = getattr(self, '_edge_label', None)
            if lbl is not None and _shiboken_is_valid(lbl):
                lbl.setText(f"边数量：{edges}")

    def update_main_road_seed_count(self, count: int):
        """更新主路种子线数量显示。"""
        lbl = getattr(self, '_seed_count_label', None)
        if lbl is not None and _shiboken_is_valid(lbl):
            lbl.setText(f"主路种子线：{count} 笔")

    def get_seed_width_settings(self) -> dict:
        """读取主路种子道路宽度 UI 设置。"""
        from roadnet.main_road_seed import PRESET_WIDTHS_M
        mode = "normal"
        if hasattr(self, "_seed_width_combo"):
            mode = self._seed_width_combo.currentData() or "normal"
        width_m = float(PRESET_WIDTHS_M.get(mode, 8.0))
        if mode == "custom" and hasattr(self, "_seed_width_m_spin"):
            width_m = float(self._seed_width_m_spin.value())
        elif mode in PRESET_WIDTHS_M:
            width_m = float(PRESET_WIDTHS_M[mode])
        radius_px = None
        if hasattr(self, "_seed_radius_px_spin"):
            val = float(self._seed_radius_px_spin.value())
            if val > 0:
                radius_px = val
        continuous = True
        if hasattr(self, "_seed_continuous_cb"):
            continuous = bool(self._seed_continuous_cb.isChecked())
        return {
            "width_mode": mode,
            "road_width_m": width_m,
            "road_radius_px": radius_px,
            "continuous_two_point": continuous,
        }

    def get_config(self) -> Dict:
        return self._config

    def set_mask_candidate_apply_enabled(self, enabled: bool, reason: str = ""):
        # 区域修正稳定模式下自动应用始终禁用；候选仍可查看。
        self._mask_candidate_apply_allowed = False
        self._mask_candidate_apply_reason = (
            "区域修正稳定模式已暂停自动应用；请使用手工 Ignore 多边形。"
        )
        button = getattr(self, "_apply_mask_candidates_btn", None)
        if button is None:
            return
        button.setEnabled(False)
        button.setVisible(False)
        button.setToolTip(self._mask_candidate_apply_reason)

    def update_config(self, config: Dict):
        self._config = config
        for key, widget in self._widgets.items():
            val = self._get_config(key, None)
            if val is None:
                continue
            if isinstance(widget, QSpinBox):
                widget.blockSignals(True)
                widget.setValue(int(val))
                widget.blockSignals(False)
            elif isinstance(widget, QDoubleSpinBox):
                widget.blockSignals(True)
                widget.setValue(float(val))
                widget.blockSignals(False)
            elif isinstance(widget, QCheckBox):
                widget.blockSignals(True)
                widget.setChecked(bool(val))
                widget.blockSignals(False)
            elif isinstance(widget, QComboBox):
                widget.blockSignals(True)
                if val in [widget.itemText(i) for i in range(widget.count())]:
                    widget.setCurrentText(str(val))
                widget.blockSignals(False)

    # ===================================================================
    # 配置读写
    # ===================================================================

    def _get_config(self, key: str, default: Any) -> Any:
        parts = key.split(".")
        val = self._config
        for p in parts:
            if isinstance(val, dict):
                val = val.get(p, default)
            else:
                return default
        return val if val is not None else default

    def _set_config(self, key: str, value: Any):
        parts = key.split(".")
        d = self._config
        for p in parts[:-1]:
            if p not in d:
                d[p] = {}
            d = d[p]
        d[parts[-1]] = value

    def _on_value_changed(self, key: str, value: Any):
        self._set_config(key, value)
        self.param_changed.emit(key, value)

    def _check_close_kernel_risk(self, value: int):
        """当闭运算核 >= 9 时弹出风险提示"""
        if value >= 9:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "闭运算核风险提示",
                f"闭运算核 = {value}，较大，容易导致道路大面积粘连！\n\n"
                "建议设置为 3~5，特别是在城市小区、校园等密集区域。\n"
                "如果必须使用大核，请同时设置 fill_small_holes=False 并\n"
                "开启 Ignore 区域排除非道路区域。"
            )
