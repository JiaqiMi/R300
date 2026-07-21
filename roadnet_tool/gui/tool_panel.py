"""
左侧工具栏面板。
"""
from __future__ import annotations

from typing import Dict, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QButtonGroup,
    QScrollArea, QLabel, QSizePolicy, QCheckBox,
)


# 工具定义：(id, 显示名, enabled_for_v1)
TOOL_DEFS: List[tuple] = [
    ("pan",                "🖐 平移/导航",      True),
    ("separator1",          None,               True),
    ("open",               "📂 打开影像",        True),
    ("separator2",          None,               True),
    ("positive_sample",    "➕ 道路样本",        True),
    ("negative_sample",    "➖ 非道路样本",      True),
    ("separator3",          None,               True),
    ("roi",                "🔲 ROI 保留区",      True),
    ("ignore",             "🚫 Ignore 删除区",   True),
    ("mask_brush",         "🖌 Mask 画笔",       True),
    ("mask_eraser",        "⌫ Mask 橡皮",       True),
    ("polyline",           "📏 折线补路",        True),
    ("separator4",          None,               True),
    ("postprocess",        "⚙ 后处理",          True),
    ("separator5",          None,               True),
    ("skeleton",           "🦴 生成骨架",        True),
    ("optimize",           "✨ 优化骨架",        True),
    ("separator6",          None,               True),
    ("graph",              "🔗 生成草稿路网",    True),
    ("separator7",          None,               True),
    ("graph_add_node",     "🔵 添加节点",        True),
    ("graph_add_edge",     "🔗 添加边",          True),
    ("graph_delete_node",  "❌ 删除节点",        True),
    ("graph_delete_edge",  "❌ 删除错误边",      True),
    ("graph_move_node",    "↕ 移动节点",        True),
    ("graph_merge_nodes",  "🔀 合并路口节点",    True),
    ("graph_draw_edge",    "✏️ 折线补路",        True),
    ("graph_local_rebuild","🧩 局部重建路网",    True),
    ("graph_locate_jump",  "🎯 定位异常跳边",    True),
    ("separator8",          None,               True),
    ("graph_save",         "💾 保存 Final Graph", True),
    ("separator9",          None,               True),
    ("set_start",          "🚩 设置起点",        True),
    ("set_end",            "🏁 设置终点",        True),
    ("add_task",           "📍 添加任务点",      True),
    ("plan",               "🗺 重新规划路径",    True),
    ("export",             "💾 导出路径",        True),
]


class ToolPanel(QWidget):
    """左侧工具栏"""

    tool_changed = Signal(str)
    tool_selected = Signal(str)

    # 新增信号
    mode_toggled = Signal(str)          # "clean" / "debug"
    clean_display_requested = Signal()
    debug_display_requested = Signal()

    def __init__(self, parent=None, v1_only: bool = True):
        super().__init__(parent)
        self._buttons: Dict[str, QPushButton] = {}
        self._separators: Dict[str, QLabel] = {}
        self._current_tool: str = "pan"
        self._v1_only = v1_only

        self._setup_ui()

    @property
    def current_tool(self) -> str:
        return self._current_tool

    def _setup_ui(self):
        self.setObjectName("ToolPanel")
        self.setMinimumWidth(155)
        self.setMaximumWidth(210)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(3)

        # 标题
        title = QLabel("🛠 工具")
        title.setObjectName("section-title")
        title.setStyleSheet("font-weight: bold; color: #89b4fa; padding: 4px 0;")
        layout.addWidget(title)

        # QButtonGroup 互斥
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)

        # 工具按钮
        for tool_id, label, v1_enabled in TOOL_DEFS:
            if tool_id.startswith("separator"):
                sep = QLabel("")
                sep.setFixedHeight(6)
                sep.setObjectName(f"sep_{tool_id}")
                layout.addWidget(sep)
                self._separators[tool_id] = sep
                continue

            btn = QPushButton(label)
            btn.setObjectName("tool-btn")
            btn.setProperty("tool_id", tool_id)
            btn.setCheckable(True)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

            if not v1_enabled:
                btn.setEnabled(False)
                btn.setToolTip(f"{label}（将在后续版本中提供）")

            btn.clicked.connect(lambda checked, tid=tool_id: self._on_tool_clicked(tid))
            layout.addWidget(btn)
            self._buttons[tool_id] = btn
            self._button_group.addButton(btn)

        # ──── 模式切换 ────
        sep_mode = QLabel("")
        sep_mode.setFixedHeight(10)
        layout.addWidget(sep_mode)

        mode_title = QLabel("🎯 显示模式")
        mode_title.setStyleSheet("font-weight: bold; color: #f9e2af; padding: 4px 0; font-size: 11px;")
        layout.addWidget(mode_title)

        self._mode_btn = QPushButton("✨ 简洁模式")
        self._mode_btn.setObjectName("mode-btn")
        self._mode_btn.setCheckable(True)
        self._mode_btn.setChecked(True)
        self._mode_btn.setToolTip("当前：简洁模式，点击切换为调试模式")
        self._mode_btn.clicked.connect(self._on_mode_toggle)
        layout.addWidget(self._mode_btn)

        # ──── 快捷操作按钮 ────
        btn_clean = QPushButton("🧹 清爽显示")
        btn_clean.setObjectName("quick-btn")
        btn_clean.setToolTip("隐藏调试图层，只显示路网")
        btn_clean.clicked.connect(lambda: self.clean_display_requested.emit())
        layout.addWidget(btn_clean)

        btn_debug = QPushButton("🔍 调试显示")
        btn_debug.setObjectName("quick-btn")
        btn_debug.setToolTip("显示全部中间图层")
        btn_debug.clicked.connect(lambda: self.debug_display_requested.emit())
        layout.addWidget(btn_debug)

        # ──── 图层显隐快速控制区 ────
        sep_layer = QLabel("")
        sep_layer.setFixedHeight(10)
        layout.addWidget(sep_layer)

        layer_title = QLabel("👁 图层显示")
        layer_title.setStyleSheet("font-weight: bold; color: #f9e2af; padding: 4px 0; font-size: 11px;")
        layout.addWidget(layer_title)

        self._layer_checkboxes: Dict[str, QCheckBox] = {}
        # 完整图层复选框（按统一命名）
        check_defs = [
            ("layer_sample_points",  "样本点",       "#c8c8c8"),
            ("layer_roi",            "ROI 区域",     "#4499ff"),
            ("layer_ignore",         "Ignore 区域",  "#ff5555"),
            ("layer_road_mask",      "Working Road Mask", "#50fa7b"),
            ("layer_cleaned_road_mask", "Cleaned Road Mask", "#00dcb4"),
            ("layer_final_edited_mask", "Final Edited Mask", "#ffaa50"),
            ("layer_preview_segmentation", "快速预览", "#50fa7b"),
            ("layer_raw_skeleton",   "Raw Skeleton", "#c8c8c8"),
            ("layer_center_filtered_skeleton", "Center Filtered Skeleton", "#b4dcff"),
            ("layer_skeleton",       "Cleaned Skeleton", "#f1fa8c"),
            ("layer_draft_graph",    "Draft Graph",  "#ffb86c"),
            ("layer_final_graph",    "Final Graph",  "#89b4fa"),
            ("layer_planned_path",   "dense_path",     "#cba6f7"),
            ("layer_sparse_waypoints", "vehicle_waypoints",   "#ffd166"),
            ("layer_waypoint_validation", "航点验收", "#ff5555"),
            ("layer_main_road_seed", "主路种子线", "#ff00ff"),
            ("layer_road_ribbon_preview", "Road Ribbon", "#00dcdc"),
            ("layer_skeleton_nodes", "调试图层",     "#ff69b4"),
        ]

        self._layer_cb_container = QWidget()
        cb_layout = QVBoxLayout(self._layer_cb_container)
        cb_layout.setContentsMargins(4, 2, 4, 2)
        cb_layout.setSpacing(1)

        for lname, ltext, lcolor in check_defs:
            cb = QCheckBox(ltext)
            cb.setChecked(False)
            cb.setStyleSheet(f"""
                QCheckBox {{
                    color: {lcolor};
                    font-size: 10px;
                    spacing: 4px;
                }}
                QCheckBox::indicator {{
                    width: 12px; height: 12px;
                }}
            """)
            cb.toggled.connect(lambda checked, n=lname: self._on_layer_toggle(n, checked))
            cb_layout.addWidget(cb)
            self._layer_checkboxes[lname] = cb

        layout.addWidget(self._layer_cb_container)
        layout.addStretch()

        scroll.setWidget(container)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

        # 默认选中平移
        self.set_current_tool("pan")

    # ===================================================================
    # 图层回调
    # ===================================================================
    _layer_toggle_callbacks: dict = {}

    def set_layer_toggle_callback(self, name: str, callback):
        """设置图层显隐回调"""
        self._layer_toggle_callbacks[name] = callback

    def _on_layer_toggle(self, name: str, checked: bool):
        cb = self._layer_toggle_callbacks.get(name)
        if cb:
            cb(name, checked)

    def set_layer_checkbox_state(self, name: str, checked: bool):
        """外部设置 checkbox 状态（不触发信号）"""
        if name in self._layer_checkboxes:
            self._layer_checkboxes[name].blockSignals(True)
            self._layer_checkboxes[name].setChecked(checked)
            self._layer_checkboxes[name].blockSignals(False)

    def sync_all_checkboxes(self, layer_manager):
        """根据 LayerManager 同步所有 checkbox 状态"""
        for lname, cb in self._layer_checkboxes.items():
            visible = layer_manager.is_layer_visible(lname)
            cb.blockSignals(True)
            cb.setChecked(visible)
            cb.blockSignals(False)

    # ===================================================================
    # 模式切换
    # ===================================================================
    def _on_mode_toggle(self):
        is_clean = self._mode_btn.isChecked()
        if is_clean:
            self._mode_btn.setText("✨ 简洁模式")
            self._mode_btn.setToolTip("当前：简洁模式，点击切换为调试模式")
            self.mode_toggled.emit("clean")
        else:
            self._mode_btn.setText("🔬 调试模式")
            self._mode_btn.setToolTip("当前：调试模式，点击切换为简洁模式")
            self.mode_toggled.emit("debug")

    def set_mode_button_state(self, mode: str):
        """外部设置模式按钮状态"""
        self._mode_btn.blockSignals(True)
        if mode == "clean":
            self._mode_btn.setChecked(True)
            self._mode_btn.setText("✨ 简洁模式")
            self._mode_btn.setToolTip("当前：简洁模式，点击切换为调试模式")
        else:
            self._mode_btn.setChecked(False)
            self._mode_btn.setText("🔬 调试模式")
            self._mode_btn.setToolTip("当前：调试模式，点击切换为简洁模式")
        self._mode_btn.blockSignals(False)

    # ===================================================================
    # 工具管理
    # ===================================================================
    def _on_tool_clicked(self, tool_id: str):
        print(f"[DEBUG][ToolPanel] clicked: {tool_id}")
        self.set_current_tool(tool_id)
        self.tool_changed.emit(tool_id)
        self.tool_selected.emit(tool_id)

    def set_enabled_tools(self, enabled_set: set):
        for tid, btn in self._buttons.items():
            if tid == "pan":
                continue
            btn.setEnabled(tid in enabled_set)

    def set_current_tool(self, tool_id: str):
        if tool_id in self._buttons:
            self._buttons[tool_id].blockSignals(True)
            self._buttons[tool_id].setChecked(True)
            self._buttons[tool_id].blockSignals(False)
            self._current_tool = tool_id

    def set_tool_enabled(self, tool_id: str, enabled: bool):
        if tool_id in self._buttons:
            self._buttons[tool_id].setEnabled(enabled)

    def set_visible_tools(self, tool_ids: set):
        visible = set(tool_ids)
        visible.add("pan")
        for tid, btn in self._buttons.items():
            btn.setVisible(tid in visible)
        for sid, sep in self._separators.items():
            sep.setVisible(True)
