"""
SAM-Road 单图推理包运行对话框。

提供完整的参数配置界面，支持：
- 路径浏览/验证
- 参数保存/恢复
- QProcess 异步运行（不阻塞 GUI）
- 实时日志显示
- dry-run / mock 模式
- 取消运行
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QSettings, QThread, QTimer, Signal, QUrl
from PySide6.QtGui import QDesktopServices, QImageReader
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QLabel, QFileDialog, QMessageBox,
    QGroupBox, QProgressBar, QScrollArea, QSizePolicy, QApplication,
    QWidget,
)

from roadnet.samroad_single_runner import (
    SAMRoadSingleRunConfig, load_config, save_config,
    dict_to_runconfig, runconfig_to_dict,
    validate_config, build_command, create_output_dir,
    SAMRoadSingleRunResult,
    prepare_runtime_env, prepare_project_import_paths,
)
from roadnet.samroadplus_runner import (
    MODEL_TYPE as SAMROADPLUS_MODEL_TYPE,
    SAMRoadPlusConfig,
    build_samroadplus_bridge_command,
    create_samroadplus_output_dir,
    load_samroadplus_config,
    run_samroadplus_preflight,
    save_samroadplus_config,
    scan_samroadplus_project,
    sam_backbone_required,
    validate_samroadplus_config,
)


DEFAULT_CONFIG_PATH = "config/samroad_single_config.yaml"


class SAMRoadSingleRunDialog(QDialog):
    """SAM-Road 单图推理包运行参数对话框。"""

    # 信号：运行完成
    finished = Signal(object)  # SAMRoadSingleRunResult

    def __init__(self, image_path: str = "", parent=None):
        super().__init__(parent)
        self._image_path = image_path
        self._config: Optional[SAMRoadSingleRunConfig] = None
        self._plus_config: SAMRoadPlusConfig = SAMRoadPlusConfig()
        self._active_model_type = "samroad_single_image"
        self._process: Optional[QProcess] = None
        self._tile_thread: Optional[QThread] = None
        self._tile_worker = None
        self._start_time: float = 0.0
        self._stdout_lines: list[str] = []
        self._stderr_lines: list[str] = []

        self.setWindowTitle("运行 SAM-Road 单图初提取")
        self.setMinimumSize(760, 560)
        self.resize(900, 720)
        self.setModal(False)
        self._window_settings = QSettings("RoadNetStudio", "SAMRoadSingleRunDialog")
        self._had_saved_geometry = False

        self._setup_ui()
        self._load_persisted_config()
        self._update_form_from_config()
        self._restore_window_geometry()
        QTimer.singleShot(0, self._fit_to_available_screen)

    def set_inference_mode(self, mode: str, force_auto_import: bool = False):
        """Preselect a mode for callers such as the one-click pipeline."""
        index = self._combo_inference_mode.findData(mode)
        if index >= 0:
            self._combo_inference_mode.setCurrentIndex(index)
        if force_auto_import:
            self._chk_auto_import.setChecked(True)

    # ===================================================================
    # UI 构建
    # ===================================================================

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # 参数配置放入独立滚动区；日志和底部操作栏不参与滚动。
        self._settings_scroll = QScrollArea()
        self._settings_scroll.setWidgetResizable(True)
        self._settings_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._settings_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        settings_content = QWidget()
        settings_content.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        settings_layout = QVBoxLayout(settings_content)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(8)
        self._settings_scroll.setWidget(settings_content)

        # ── 模型类型 ──
        model_group = QGroupBox("🧠 模型类型")
        model_form = QFormLayout(model_group)
        self._combo_model_type = QComboBox()
        self._combo_model_type.addItem("SAM-Road 单图（原入口）", "samroad_single_image")
        self._combo_model_type.addItem("SAM-RoadPlus Portable（新训练结果）", SAMROADPLUS_MODEL_TYPE)
        model_form.addRow("model_type:", self._combo_model_type)
        settings_layout.addWidget(model_group)

        # ── 路径配置 ──
        self._legacy_path_group = QGroupBox("📂 SAM-Road 单图推理包 路径配置")
        path_form = QFormLayout(self._legacy_path_group)

        self._edit_project_dir = self._create_path_row(
            path_form, "推理包目录:", self._on_browse_project_dir,
            hint="D:/sam_road_single_image_share"
        )
        self._edit_python = self._create_path_row(
            path_form, "Python 解释器:", self._on_browse_python,
            hint="例如: D:/Anaconda/envs/samroad/python.exe"
        )
        self._edit_infer_script = self._create_path_row(
            path_form, "infer_single.py:", self._on_browse_infer_script,
            hint="D:/sam_road_single_image_share/infer_single.py"
        )
        self._edit_sam_backbone = self._create_path_row(
            path_form, "SAM backbone 权重:", self._on_browse_sam_backbone,
            hint="D:/sam_road_single_image_share/sam_ckpts/sam_vit_b_01ec64.pth"
        )
        self._edit_samroad_ckpt = self._create_path_row(
            path_form, "SAM-Road 模型权重:", self._on_browse_samroad_ckpt,
            hint="D:/sam_road_single_image_share/checkpoints/model.ckpt"
        )
        self._edit_config_file = self._create_path_row(
            path_form, "Config 文件:", self._on_browse_config,
            hint="YAML 配置文件"
        )

        # 匹配检查按钮行
        match_row = QHBoxLayout()
        self._btn_check_match = QPushButton("🔍 检查 config/checkpoint 匹配")
        self._btn_check_match.setToolTip("快速检查当前 config 和 checkpoint 的 shape 是否匹配")
        self._btn_check_match.clicked.connect(self._on_check_match)
        match_row.addWidget(self._btn_check_match)

        self._btn_scan_match = QPushButton("🔎 扫描匹配组合")
        self._btn_scan_match.setToolTip("扫描项目目录中所有 config/*.yaml 和 checkpoints/*.ckpt，自动找匹配组合")
        self._btn_scan_match.clicked.connect(self._on_scan_match)
        match_row.addWidget(self._btn_scan_match)
        path_form.addRow("", match_row)

        settings_layout.addWidget(self._legacy_path_group)

        # SAM-RoadPlus 使用独立路径字段，切换模型不会覆盖旧 SAM-Road 配置。
        self._plus_path_group = QGroupBox("📂 SAM-RoadPlus Portable 路径配置")
        plus_form = QFormLayout(self._plus_path_group)
        self._edit_plus_project_dir = self._create_path_row(
            plus_form, "Portable 工程目录:", self._on_browse_plus_project_dir,
            hint="D:/samroadplus_portable_infer",
        )
        self._edit_plus_python = self._create_path_row(
            plus_form, "Python 解释器:", self._on_browse_python,
            hint="C:/Users/小马/.conda/envs/samroad/python.exe",
        )
        self._edit_plus_infer_script = self._create_path_row(
            plus_form, "推理入口脚本:", self._on_browse_plus_infer_script,
            hint="infer.py / infer_single.py / run_infer.py / predict.py",
        )
        self._edit_plus_config_file = self._create_path_row(
            plus_form, "训练对应 Config:", self._on_browse_plus_config,
            hint="D:/samroadplus_portable_infer/config.yaml",
        )
        self._edit_plus_model_ckpt = self._create_path_row(
            plus_form, "新训练模型权重:", self._on_browse_plus_model_ckpt,
            hint="D:/samroadplus_portable_infer/model_state_dict.pth",
        )
        self._edit_plus_sam_backbone = self._create_path_row(
            plus_form, "SAM backbone（按 config 可选）:", self._on_browse_sam_backbone,
            hint="当前 Portable config 设置 SKIP_SAM_CKPT_LOAD=true，可留空",
        )
        plus_buttons = QHBoxLayout()
        self._btn_scan_plus = QPushButton("🔎 扫描 SAM-RoadPlus 工程")
        self._btn_scan_plus.clicked.connect(self._on_scan_samroadplus_project)
        plus_buttons.addWidget(self._btn_scan_plus)
        self._btn_check_plus = QPushButton("🔍 检查 checkpoint/config 匹配")
        self._btn_check_plus.clicked.connect(self._on_check_samroadplus_match)
        plus_buttons.addWidget(self._btn_check_plus)
        plus_form.addRow("", plus_buttons)
        settings_layout.addWidget(self._plus_path_group)
        self._plus_path_group.hide()
        self._combo_model_type.currentIndexChanged.connect(self._on_model_type_changed)

        # ── 图像确认 ──
        img_group = QGroupBox("📷 输入图像")
        img_layout = QFormLayout(img_group)
        self._lbl_image = QLabel(self._image_path or "(无)")
        self._lbl_image.setWordWrap(True)
        self._lbl_image.setStyleSheet("font-weight: bold; color: #333;")
        img_layout.addRow("当前图像:", self._lbl_image)
        settings_layout.addWidget(img_group)

        # ── 运行参数 ──
        param_group = QGroupBox("⚙️ 运行参数")
        param_form = QFormLayout(param_group)

        device_row = QHBoxLayout()
        self._combo_device = QComboBox()
        self._combo_device.addItems(["cuda", "cpu"])
        device_row.addWidget(QLabel("Device:"))
        device_row.addWidget(self._combo_device)
        device_row.addStretch()
        param_form.addRow("", device_row)

        self._edit_output_dir = self._create_path_row(
            param_form, "输出目录:", self._on_browse_output_dir,
            hint="留空则自动创建: outputs/samroad_single_runs/samroad_single_<image>_<timestamp>/"
        )

        self._combo_inference_mode = QComboBox()
        self._combo_inference_mode.addItem("自动（大图使用 tile）", "auto")
        self._combo_inference_mode.addItem("小图整图推理", "whole")
        self._combo_inference_mode.addItem("大图 tile 推理", "tile")
        param_form.addRow("推理模式:", self._combo_inference_mode)

        tile_row = QHBoxLayout()
        self._spin_tile_size = QSpinBox()
        self._spin_tile_size.setRange(256, 4096)
        self._spin_tile_size.setValue(1024)
        self._spin_overlap = QSpinBox()
        self._spin_overlap.setRange(0, 1023)
        self._spin_overlap.setValue(128)
        tile_row.addWidget(QLabel("Tile:"))
        tile_row.addWidget(self._spin_tile_size)
        tile_row.addWidget(QLabel("Overlap:"))
        tile_row.addWidget(self._spin_overlap)
        param_form.addRow("Tile 参数:", tile_row)

        valid_row = QHBoxLayout()
        self._chk_skip_black_tile = QCheckBox("跳过黑色无效 tile")
        self._chk_skip_black_tile.setChecked(True)
        self._spin_black_threshold = QSpinBox()
        self._spin_black_threshold.setRange(0, 32)
        self._spin_black_threshold.setValue(10)
        valid_row.addWidget(self._chk_skip_black_tile)
        valid_row.addWidget(QLabel("黑阈值:"))
        valid_row.addWidget(self._spin_black_threshold)
        param_form.addRow("有效区域:", valid_row)

        self._spin_min_black_area = QSpinBox()
        self._spin_min_black_area.setRange(1, 100000000)
        self._spin_min_black_area.setValue(4096)
        self._spin_min_black_area.setToolTip(
            "只有与图像边界连通且面积达到该值的黑色区域才会被跳过"
        )
        param_form.addRow("边界黑区最小面积:", self._spin_min_black_area)

        merge_row = QHBoxLayout()
        self._spin_valid_ratio = QDoubleSpinBox()
        self._spin_valid_ratio.setRange(0.0, 1.0)
        self._spin_valid_ratio.setSingleStep(0.05)
        self._spin_valid_ratio.setValue(0.10)
        self._combo_merge_method = QComboBox()
        self._combo_merge_method.addItems(["max", "average"])
        merge_row.addWidget(QLabel("最小有效比例:"))
        merge_row.addWidget(self._spin_valid_ratio)
        merge_row.addWidget(QLabel("融合:"))
        merge_row.addWidget(self._combo_merge_method)
        param_form.addRow("Tile 融合:", merge_row)
        settings_layout.addWidget(param_group)

        # ── 选项 ──
        option_group = QGroupBox("📋 选项")
        option_form = QFormLayout(option_group)
        self._chk_auto_import = QCheckBox("运行完成后自动导入结果")
        self._chk_auto_import.setChecked(True)
        option_form.addRow(self._chk_auto_import)

        self._chk_ignore_plus_graph = QCheckBox(
            "只导入 road_mask，忽略 SAM-RoadPlus graph（推荐）"
        )
        self._chk_ignore_plus_graph.setChecked(True)
        self._chk_ignore_plus_graph.setToolTip(
            "SAM-RoadPlus graph 默认只作参考；final_graph 始终由 mask → skeleton → skeleton_to_graph 生成。"
        )
        option_form.addRow(self._chk_ignore_plus_graph)

        self._chk_dry_run = QCheckBox("Dry-run / Mock 测试模式（不调用真实推理）")
        self._chk_dry_run.setToolTip(
            "仅检查路径和参数，不执行真实 SAM-Road 推理。\n"
            "所有输出都会标注为 DRY-RUN / MOCK OUTPUT。"
        )
        option_form.addRow(self._chk_dry_run)

        self._chk_partial_load = QCheckBox(
            "⚠️ mask-only partial load 调试模式（仅加载可匹配权重，graph 不可靠）"
        )
        self._chk_partial_load.setToolTip(
            "仅适用于 config/checkpoint 不匹配时的临时调试。\n"
            "road_mask 和 keypoint_mask 仍然可用；\n"
            "拓扑 graph (draft_graph.json) 不可靠！"
        )
        option_form.addRow(self._chk_partial_load)

        self._checkpoint_hint = QLabel(
            "提示：当前模型若为早期 epoch checkpoint（例如 epoch=1-step=5292.ckpt），"
            "建议对比 epoch_5 / epoch_10 模型；仍允许继续使用当前权重。"
        )
        self._checkpoint_hint.setWordWrap(True)
        self._checkpoint_hint.setStyleSheet(
            "color:#d9822b; background:#fff4e5; padding:6px; border-radius:3px;"
        )
        option_form.addRow(self._checkpoint_hint)
        settings_layout.addWidget(option_group)
        settings_layout.addStretch()
        main_layout.addWidget(self._settings_scroll, stretch=3)

        # ── 日志 ──
        log_group = QGroupBox("📜 运行日志")
        log_layout = QVBoxLayout(log_group)
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #d4d4d4; "
            "font-family: Consolas, monospace; font-size: 12px; }"
        )
        self._log_view.setMinimumHeight(100)
        log_layout.addWidget(self._log_view)
        self._log_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        main_layout.addWidget(log_group, stretch=2)

        # ── 底部按钮 ──
        btn_layout = QHBoxLayout()

        self._btn_save = QPushButton("💾 保存配置")
        self._btn_save.clicked.connect(self._on_save_config)
        btn_layout.addWidget(self._btn_save)

        btn_layout.addStretch()

        self._btn_cancel = QPushButton("取消")
        self._btn_cancel.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self._btn_cancel)

        self._btn_run = QPushButton("▶ 运行 SAM-Road 单图初提取")
        self._btn_run.setStyleSheet(
            "QPushButton { background-color: #0078d4; color: white; font-weight: bold; "
            "padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #106ebe; }"
            "QPushButton:disabled { background-color: #ccc; }"
        )
        self._btn_run.clicked.connect(self._on_run)
        btn_layout.addWidget(self._btn_run)

        # ── 进度条 ──
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.hide()
        main_layout.addWidget(self._progress_bar)

        # 操作栏最后加入主布局，始终固定在窗口最底部。
        main_layout.addLayout(btn_layout)

    def _create_path_row(self, form: QFormLayout, label: str, browse_handler,
                         hint: str = "") -> QLineEdit:
        edit = QLineEdit()
        if hint:
            edit.setPlaceholderText(hint)
        edit.setMinimumWidth(350)

        row = QHBoxLayout()
        row.addWidget(edit, stretch=1)
        btn = QPushButton("浏览...")
        btn.clicked.connect(lambda: browse_handler(edit))
        row.addWidget(btn)

        form.addRow(label, row)
        return edit

    # ===================================================================
    # 配置持久化
    # ===================================================================

    def _load_persisted_config(self):
        data = load_config(DEFAULT_CONFIG_PATH)
        if not data:
            self._config = SAMRoadSingleRunConfig()
        else:
            self._config = dict_to_runconfig(data)
        self._plus_config = load_samroadplus_config()

    def _update_form_from_config(self):
        if self._config is None:
            return
        c = self._config
        self._edit_project_dir.setText(str(c.project_dir) if c.project_dir != Path() else "")
        self._edit_python.setText(str(c.python_executable) if c.python_executable != Path() else "")
        self._edit_infer_script.setText(str(c.infer_script) if c.infer_script != Path() else "")
        self._edit_sam_backbone.setText(str(c.sam_backbone_ckpt_path) if c.sam_backbone_ckpt_path != Path() else "")
        self._edit_samroad_ckpt.setText(str(c.samroad_model_ckpt_path) if c.samroad_model_ckpt_path != Path() else "")
        self._edit_config_file.setText(str(c.config_path) if c.config_path != Path() else "")
        idx = self._combo_device.findText(c.device)
        if idx >= 0:
            self._combo_device.setCurrentIndex(idx)
        self._chk_auto_import.setChecked(c.auto_import_after_run)
        self._chk_dry_run.setChecked(c.dry_run)
        self._chk_partial_load.setChecked(c.mask_only_partial_load)
        mode_index = self._combo_inference_mode.findData(c.inference_mode)
        self._combo_inference_mode.setCurrentIndex(max(0, mode_index))
        self._spin_tile_size.setValue(c.tile_size)
        self._spin_overlap.setValue(c.overlap)
        self._chk_skip_black_tile.setChecked(c.skip_black_tile)
        self._spin_black_threshold.setValue(c.black_threshold)
        self._spin_min_black_area.setValue(c.min_black_component_area)
        self._spin_valid_ratio.setValue(c.valid_pixel_ratio_threshold)
        self._combo_merge_method.setCurrentText(c.merge_method)
        plus = self._plus_config
        self._edit_plus_project_dir.setText(str(plus.project_dir) if plus.project_dir else "")
        self._edit_plus_python.setText(str(plus.python_executable) if plus.python_executable else "")
        self._edit_plus_infer_script.setText(str(plus.infer_script) if plus.infer_script else "")
        self._edit_plus_config_file.setText(str(plus.config_path) if plus.config_path else "")
        self._edit_plus_model_ckpt.setText(str(plus.model_ckpt_path) if plus.model_ckpt_path else "")
        self._edit_plus_sam_backbone.setText(
            str(plus.sam_backbone_ckpt_path)
            if plus.sam_backbone_ckpt_path != Path() else ""
        )
        self._chk_ignore_plus_graph.setChecked(plus.ignore_graph)
        model_index = self._combo_model_type.findData(plus.model_type)
        if model_index < 0:
            model_index = 0
        self._combo_model_type.setCurrentIndex(model_index)
        self._on_model_type_changed()

    def _update_config_from_form(self) -> SAMRoadSingleRunConfig:
        c = self._config or SAMRoadSingleRunConfig()
        if self._edit_project_dir.text():
            c.project_dir = Path(self._edit_project_dir.text())
        if self._edit_python.text():
            c.python_executable = Path(self._edit_python.text())
        if self._edit_infer_script.text():
            c.infer_script = Path(self._edit_infer_script.text())
        if self._edit_sam_backbone.text():
            c.sam_backbone_ckpt_path = Path(self._edit_sam_backbone.text())
        if self._edit_samroad_ckpt.text():
            c.samroad_model_ckpt_path = Path(self._edit_samroad_ckpt.text())
        if self._edit_config_file.text():
            c.config_path = Path(self._edit_config_file.text())
        if self._image_path:
            c.input_image = Path(self._image_path)
        # 确保 config_path 为绝对路径
        if c.config_path != Path() and not c.config_path.is_absolute():
            c.config_path = (c.project_dir / c.config_path).resolve()
        if c.sam_backbone_ckpt_path != Path() and not c.sam_backbone_ckpt_path.is_absolute():
            c.sam_backbone_ckpt_path = (c.project_dir / c.sam_backbone_ckpt_path).resolve()
        if c.samroad_model_ckpt_path != Path() and not c.samroad_model_ckpt_path.is_absolute():
            c.samroad_model_ckpt_path = (c.project_dir / c.samroad_model_ckpt_path).resolve()
        c.device = self._combo_device.currentText()
        c.auto_import_after_run = self._chk_auto_import.isChecked()
        c.dry_run = self._chk_dry_run.isChecked()
        c.mask_only_partial_load = self._chk_partial_load.isChecked()
        c.inference_mode = str(self._combo_inference_mode.currentData())
        c.tile_size = self._spin_tile_size.value()
        c.overlap = self._spin_overlap.value()
        c.skip_black_tile = self._chk_skip_black_tile.isChecked()
        c.black_threshold = self._spin_black_threshold.value()
        c.min_black_component_area = self._spin_min_black_area.value()
        c.valid_pixel_ratio_threshold = self._spin_valid_ratio.value()
        c.merge_method = self._combo_merge_method.currentText()
        self._config = c
        return c

    def _update_plus_config_from_form(self) -> SAMRoadPlusConfig:
        c = self._plus_config or SAMRoadPlusConfig()
        c.model_type = SAMROADPLUS_MODEL_TYPE
        c.project_dir = Path(self._edit_plus_project_dir.text().strip())
        c.python_executable = Path(self._edit_plus_python.text().strip())
        c.infer_script = Path(self._edit_plus_infer_script.text().strip())
        c.config_path = Path(self._edit_plus_config_file.text().strip())
        c.model_ckpt_path = Path(self._edit_plus_model_ckpt.text().strip())
        backbone_text = self._edit_plus_sam_backbone.text().strip()
        c.sam_backbone_ckpt_path = Path(backbone_text) if backbone_text else Path()
        c.input_image = Path(self._image_path) if self._image_path else Path()
        c.device = self._combo_device.currentText()
        c.auto_import_after_run = self._chk_auto_import.isChecked()
        c.ignore_graph = self._chk_ignore_plus_graph.isChecked()
        c.inference_mode = str(self._combo_inference_mode.currentData())
        c.tile_size = self._spin_tile_size.value()
        c.overlap = self._spin_overlap.value()
        c.skip_black_tile = self._chk_skip_black_tile.isChecked()
        self._plus_config = c
        return c

    def _on_model_type_changed(self, *_args):
        model_type = self._combo_model_type.currentData()
        is_plus = model_type == SAMROADPLUS_MODEL_TYPE
        self._active_model_type = str(model_type)
        self._legacy_path_group.setVisible(not is_plus)
        self._plus_path_group.setVisible(is_plus)
        self._chk_ignore_plus_graph.setVisible(is_plus)
        self._chk_partial_load.setVisible(not is_plus)
        self._chk_dry_run.setVisible(not is_plus)
        self._checkpoint_hint.setVisible(not is_plus)
        self._btn_run.setText(
            "▶ 运行 SAM-RoadPlus 推理" if is_plus
            else "▶ 运行 SAM-Road 单图初提取"
        )
        if is_plus:
            self.setWindowTitle("运行 SAM-RoadPlus Portable 单图初提取")
            plus = self._plus_config
            self._combo_device.setCurrentText(plus.device)
            plus_mode = self._combo_inference_mode.findData(plus.inference_mode)
            self._combo_inference_mode.setCurrentIndex(max(0, plus_mode))
            self._spin_tile_size.setValue(plus.tile_size)
            self._spin_overlap.setValue(plus.overlap)
            self._chk_skip_black_tile.setChecked(plus.skip_black_tile)
            self._chk_auto_import.setChecked(plus.auto_import_after_run)
            self._chk_ignore_plus_graph.setChecked(plus.ignore_graph)
            if self._plus_config.output_dir != Path():
                self._edit_output_dir.setText(str(self._plus_config.output_dir))
            else:
                self._edit_output_dir.clear()
        else:
            self.setWindowTitle("运行 SAM-Road 单图初提取")
            legacy = self._config or SAMRoadSingleRunConfig()
            self._combo_device.setCurrentText(legacy.device)
            legacy_mode = self._combo_inference_mode.findData(legacy.inference_mode)
            self._combo_inference_mode.setCurrentIndex(max(0, legacy_mode))
            self._spin_tile_size.setValue(legacy.tile_size)
            self._spin_overlap.setValue(legacy.overlap)
            self._chk_skip_black_tile.setChecked(legacy.skip_black_tile)
            self._chk_auto_import.setChecked(legacy.auto_import_after_run)
            self._edit_output_dir.clear()

    # ===================================================================
    # 浏览按钮
    # ===================================================================

    def _browse_dir(self, edit: QLineEdit, title: str):
        path = QFileDialog.getExistingDirectory(self, title, edit.text())
        if path:
            edit.setText(path)

    def _browse_file(self, edit: QLineEdit, title: str, filter_str: str,
                     start_dir: str = ""):
        d = start_dir or edit.text() or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, title, d, filter_str)
        if path:
            edit.setText(path)

    def _on_browse_project_dir(self, edit: QLineEdit):
        self._browse_dir(edit, "选择 SAM-Road 单图推理包目录")

    def _on_browse_plus_project_dir(self, edit: QLineEdit):
        self._browse_dir(edit, "选择 SAM-RoadPlus Portable 工程目录")

    def _on_browse_python(self, edit: QLineEdit):
        filter_str = "python.exe (python.exe);;所有文件 (*)"
        self._browse_file(edit, "选择 Python 解释器", filter_str)

    def _on_browse_infer_script(self, edit: QLineEdit):
        filter_str = "Python 脚本 (*.py);;所有文件 (*)"
        self._browse_file(edit, "选择 infer_single.py", filter_str)

    def _on_browse_plus_infer_script(self, edit: QLineEdit):
        self._browse_file(
            edit, "选择 SAM-RoadPlus 推理入口", "Python 脚本 (*.py);;所有文件 (*)",
            self._edit_plus_project_dir.text(),
        )

    def _on_browse_plus_config(self, edit: QLineEdit):
        self._browse_file(
            edit, "选择 SAM-RoadPlus 训练对应 Config",
            "配置文件 (*.yaml *.yml *.py);;所有文件 (*)",
            self._edit_plus_project_dir.text(),
        )

    def _on_browse_plus_model_ckpt(self, edit: QLineEdit):
        self._browse_file(
            edit, "选择 SAM-RoadPlus 新训练权重",
            "模型权重 (*.ckpt *.pth *.pt);;所有文件 (*)",
            self._edit_plus_project_dir.text(),
        )

    def _on_browse_sam_backbone(self, edit: QLineEdit):
        filter_str = "PyTorch 权重 (*.pth *.pt);;所有文件 (*)"
        self._browse_file(edit, "选择 SAM backbone 权重文件", filter_str)

    def _on_browse_samroad_ckpt(self, edit: QLineEdit):
        filter_str = "Checkpoint 文件 (*.ckpt);;所有文件 (*)"
        self._browse_file(edit, "选择 SAM-Road 模型权重文件", filter_str)

    @staticmethod
    def _is_likely_sam_backbone(path_str: str) -> bool:
        """判断路径是否很可能是 SAM backbone 权重而非 SAM-Road model checkpoint。"""
        if not path_str:
            return False
        lower = path_str.lower()
        # 文件名包含 sam_vit_b → 很可能是 backbone
        if "sam_vit_b" in lower:
            return True
        # 路径包含 sam_ckpts → 很可能是 backbone
        if "sam_ckpts" in lower.replace("\\", "/"):
            return True
        return False

    def _on_browse_config(self, edit: QLineEdit):
        filter_str = "YAML 文件 (*.yaml *.yml);;所有文件 (*)"
        start_dir = self._edit_project_dir.text()
        self._browse_file(edit, "选择配置文件", filter_str, start_dir)

    def _on_browse_output_dir(self, edit: QLineEdit):
        self._browse_dir(edit, "选择输出目录")

    def _on_save_config(self):
        try:
            selected = str(self._combo_model_type.currentData())
            if selected == SAMROADPLUS_MODEL_TYPE:
                # Keep the legacy object untouched while editing Plus paths.
                legacy = self._config or SAMRoadSingleRunConfig()
                save_config(DEFAULT_CONFIG_PATH, runconfig_to_dict(legacy))
                plus = self._update_plus_config_from_form()
            else:
                legacy = self._update_config_from_form()
                save_config(DEFAULT_CONFIG_PATH, runconfig_to_dict(legacy))
                plus = self._plus_config
            plus.model_type = selected
            save_samroadplus_config(plus)
            self._log(
                "[信息] 配置已保存到 config/samroad_single_config.yaml 和 "
                "config/samroadplus_config.yaml"
            )
            QMessageBox.information(self, "保存成功",
                                    "配置已保存到:\n"
                                    f"{DEFAULT_CONFIG_PATH}\nconfig/samroadplus_config.yaml")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def _on_scan_samroadplus_project(self):
        project = self._edit_plus_project_dir.text().strip()
        result = scan_samroadplus_project(project)
        if not result.project_dir.is_dir():
            QMessageBox.warning(self, "扫描失败", f"工程目录不存在:\n{project}")
            return
        if result.preferred_infer_script:
            self._edit_plus_infer_script.setText(str(result.preferred_infer_script))
        if result.preferred_config:
            self._edit_plus_config_file.setText(str(result.preferred_config))
        if result.preferred_checkpoint:
            self._edit_plus_model_ckpt.setText(str(result.preferred_checkpoint))
        if result.preferred_sam_backbone:
            self._edit_plus_sam_backbone.setText(str(result.preferred_sam_backbone))
        lines = [
            f"工程目录: {result.project_dir}",
            f"推理脚本: {len(result.infer_scripts)}",
            *[f"  - {path.name}" for path in result.infer_scripts[:5]],
            f"Config: {len(result.config_files)}",
            *[f"  - {path.name}" for path in result.config_files[:5]],
            f"模型权重: {len(result.model_checkpoints)}",
            *[f"  - {path.name}" for path in result.model_checkpoints[:5]],
            f"SAM backbone: {len(result.sam_backbones)}",
        ]
        config_path = result.preferred_config
        if config_path and not sam_backbone_required(config_path):
            lines.append("  - 当前 config 不需要独立 SAM backbone（已包含在模型 state_dict）")
        message = "\n".join(lines)
        self._log("[SAM-RoadPlus 扫描]\n" + message)
        QMessageBox.information(self, "SAM-RoadPlus 工程扫描完成", message)

    def _on_check_samroadplus_match(self):
        config = self._update_plus_config_from_form()
        errors = validate_samroadplus_config(config)
        # 匹配检查不要求已有真实输入图像之外的输出目录。
        if errors:
            QMessageBox.critical(self, "SAM-RoadPlus 配置错误", "\n".join(errors))
            return
        if not config.output_dir:
            config.output_dir = Path("outputs/samroadplus_preflight")
        self._btn_check_plus.setEnabled(False)
        self._btn_scan_plus.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._log("[SAM-RoadPlus 预检] 正在构建真实模型并严格检查 state_dict key/shape...")
            result = run_samroadplus_preflight(config)
        except Exception as exc:
            result = {"success": False, "message": str(exc)}
        finally:
            QApplication.restoreOverrideCursor()
            self._btn_check_plus.setEnabled(True)
            self._btn_scan_plus.setEnabled(True)
        if result.get("success"):
            message = (
                "✅ SAM-RoadPlus config/checkpoint 严格匹配。\n\n"
                f"模型: {result.get('model_class', '')}\n"
                f"state_dict keys: {result.get('state_dict_key_count', 0)}"
            )
            self._log("[SAM-RoadPlus 预检] " + message.replace("\n", " | "))
            QMessageBox.information(self, "匹配检查结果", message)
        else:
            message = result.get("message") or result.get("stderr") or "未知预检错误"
            self._log("[SAM-RoadPlus 预检失败] " + str(message))
            QMessageBox.critical(self, "checkpoint/config 不匹配", str(message))

    # ===================================================================
    # Config/Checkpoint 匹配检查
    # ===================================================================

    def _run_inspect_tool(self, match_mode: bool = False) -> str:
        """后台运行 inspect_samroad_checkpoint.py，返回 output 文本。

        重要：匹配检查只使用 SAM-Road 模型权重 (samroad_model_ckpt_path)，
        不使用 SAM backbone 权重 (sam_backbone_ckpt_path)。
        """
        import subprocess

        config = self._update_config_from_form()
        tools_script = Path(__file__).resolve().parent.parent / "tools" / "inspect_samroad_checkpoint.py"

        # ★ 匹配检查只使用 samroad_model_ckpt_path
        samroad_ckpt = str(config.samroad_model_ckpt_path)

        cmd = [
            str(config.python_executable) if config.python_executable != Path() else sys.executable,
            str(tools_script),
            "--checkpoint", samroad_ckpt,
            "--config", str(config.config_path),
            "--project-dir", str(config.project_dir),
        ]
        if match_mode:
            cmd.append("--match")

        self._log(f"[匹配检查] 运行: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=120,
                cwd=str(config.project_dir),
                env={**os.environ, "PYTHONPATH": prepare_project_import_paths(str(config.project_dir)),
                     "PYTHONUNBUFFERED": "1", "WANDB_MODE": "disabled"},
            )
            output = proc.stdout + "\n" + proc.stderr
            return output
        except Exception as e:
            return f"[ERROR] 匹配检查失败: {e}"

    def _on_check_match(self):
        """检查当前 config 和 checkpoint 的 shape 是否匹配。"""
        self._log("=" * 40)
        self._log("[匹配检查] 正在检查 config/checkpoint 匹配...")

        # ★ 检查用户是否误选了 SAM backbone 作为 SAM-Road 模型权重
        samroad_path = self._edit_samroad_ckpt.text()
        if self._is_likely_sam_backbone(samroad_path):
            msg = (
                "当前选择的是 SAM backbone 权重，不是 SAM-Road model.ckpt。\n\n"
                f"当前 SAM-Road 模型权重字段: {samroad_path}\n\n"
                "请在「SAM-Road 模型权重」字段选择:\n"
                "  D:/sam_road_single_image_share/checkpoints/model.ckpt"
            )
            self._log(f"[匹配检查] ⚠️  {msg}")
            QMessageBox.warning(self, "权重路径错误", msg)
            return

        self._btn_check_match.setEnabled(False)
        self._btn_scan_match.setEnabled(False)

        output = self._run_inspect_tool(match_mode=False)

        self._btn_check_match.setEnabled(True)
        self._btn_scan_match.setEnabled(True)

        # 解析结果
        if "SHAPE MISMATCHES" in output:
            # 提取关键不匹配
            lines = output.split("\n")
            mismatches = []
            in_mismatch = False
            for line in lines:
                if "SHAPE MISMATCHES" in line:
                    in_mismatch = True
                    continue
                if in_mismatch and line.strip().startswith("!"):
                    break
                if in_mismatch and "model:" in line:
                    mismatches.append(line.strip())
                elif in_mismatch and "checkpoint:" in line:
                    mismatches[-1] = mismatches[-1] + "; " + line.strip()

            msg = "❌ Config 与 Checkpoint 不匹配！\n\n"
            msg += f"Config: {self._edit_config_file.text()}\n"
            msg += f"SAM-Road 模型权重: {self._edit_samroad_ckpt.text()}\n\n"
            for m in mismatches[:5]:
                msg += f"  {m}\n"
            if len(mismatches) > 5:
                msg += f"  ... 和其他 {len(mismatches) - 5} 个参数\n"
            msg += "\n💡 建议：\n"
            msg += "  1. 点击「扫描匹配组合」自动找匹配的 config/ckpt\n"
            msg += "  2. 或启用「mask-only partial load」调试 road_mask\n"
            self._log(msg)
            QMessageBox.warning(self, "匹配检查结果", msg)
        elif "OK: All shapes match" in output:
            msg = "✅ Config 和 Checkpoint 完全匹配！可以安全运行。"
            self._log(msg)
            QMessageBox.information(self, "匹配检查结果", msg)
        else:
            self._log("[匹配检查] 输出如下：")
            for line in output.split("\n"):
                self._log(f"  {line}")
            QMessageBox.information(self, "匹配检查结果",
                                    f"检查完成。详见日志。\n\n最后 500 字符:\n{output[-500:]}")

    def _on_scan_match(self):
        """扫描项目目录中的所有 config 和 checkpoint 组合。"""
        import sys
        self._log("=" * 40)
        self._log("[扫描匹配] 正在扫描所有 config/checkpoint 组合...")
        self._btn_check_match.setEnabled(False)
        self._btn_scan_match.setEnabled(False)

        output = self._run_inspect_tool(match_mode=True)

        self._btn_check_match.setEnabled(True)
        self._btn_scan_match.setEnabled(True)

        # 解析结果
        if "Found matching combination" in output or "MATCH!" in output:
            # 提取匹配行
            lines = output.split("\n")
            configs = []
            checkpoints = []
            for i, line in enumerate(lines):
                if "Config:" in line and "MATCH" in output:
                    # Parse config and checkpoint from output
                    pass

            msg = "✅ 找到匹配组合！详见运行日志。"
            self._log(msg)
            QMessageBox.information(self, "扫描结果", msg)
        elif "NO matching" in output:
            msg = (
                "❌ 未找到匹配的 config/checkpoint 组合。\n\n"
                "当前 checkpoint 与现有 config 均不匹配。\n"
                "请提供与 checkpoint 匹配的 config，或更换 checkpoint。\n\n"
                "💡 可启用「mask-only partial load」临时调试 road_mask。\n"
                "⚠️ partial 模式下 graph 不可靠。"
            )
            self._log(msg)
            QMessageBox.warning(self, "扫描结果", msg)
        else:
            self._log("[扫描匹配] 输出如下：")
            for line in output.split("\n")[-40:]:
                self._log(f"  {line}")
            QMessageBox.information(self, "扫描结果",
                                    f"扫描完成。详见日志。\n\n最后 500 字符:\n{output[-500:]}")

    # ===================================================================
    # 运行
    # ===================================================================

    def _on_run(self):
        if self._combo_model_type.currentData() == SAMROADPLUS_MODEL_TYPE:
            self._on_run_samroadplus()
            return
        config = self._update_config_from_form()

        # ── 验证 ──
        errors = validate_config(config)
        if errors:
            QMessageBox.critical(self, "配置错误", "\n".join(errors))
            return

        image_size = QImageReader(str(config.input_image)).size()
        image_width = image_size.width() if image_size.isValid() else 0
        image_height = image_size.height() if image_size.isValid() else 0
        is_large_image = max(image_width, image_height) > 2048
        resolved_mode = config.inference_mode
        if resolved_mode == "auto":
            resolved_mode = "tile" if is_large_image else "whole"
            if is_large_image:
                QMessageBox.information(
                    self, "SAM-Road 大图 tile 推理",
                    "当前为大图，建议使用 tile 推理。整图推理会压缩道路细节，效果可能很差。\n\n"
                    "已自动选择后台 tile 推理。",
                )
        elif resolved_mode == "whole" and is_large_image:
            reply = QMessageBox.warning(
                self, "大图整图推理警告",
                "当前为大图，建议使用 tile 推理。整图推理会压缩道路细节，效果可能很差。\n\n"
                "是否仍继续整图推理？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # ── Partial load 模式警告 ──
        if config.mask_only_partial_load and not config.dry_run:
            reply = QMessageBox.warning(
                self, "⚠️ Partial Load 调试模式",
                "您启用了 mask-only partial load 调试模式。\n\n"
                "仅加载 shape 匹配的模型权重，其余参数将被跳过。\n\n"
                "road_mask 和 keypoint_mask 仍然可用，\n"
                "但拓扑 graph (draft_graph.json) 不可靠！\n\n"
                "确定要继续吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # ── 输出目录 ──
        output_dir_text = self._edit_output_dir.text().strip()
        if output_dir_text:
            config.output_dir = Path(output_dir_text)
        else:
            config.output_dir = create_output_dir(
                base_dir="outputs/samroad_single_runs",
                image_path=str(config.input_image),
            )
        config.output_dir = config.output_dir.expanduser().resolve()
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # ── 保存配置 ──
        try:
            data = runconfig_to_dict(config)
            save_config(DEFAULT_CONFIG_PATH, data)
            self._plus_config.model_type = "samroad_single_image"
            save_samroadplus_config(self._plus_config)
        except Exception:
            pass

        # ── 日志 ──
        self._log(f"[信息] 输出目录: {config.output_dir}")
        self._log(
            f"[信息] 推理模式: {resolved_mode}，影像尺寸: {image_width}x{image_height}"
        )

        if config.dry_run:
            self._log("=" * 60)
            self._log("  DRY RUN / MOCK MODE — 不会调用真实 SAM-Road 推理")
            self._log("=" * 60)
        else:
            self._log("=" * 60)
            self._log(f"  开始 SAM-Road 单图推理")
            self._log(f"  Python: {config.python_executable}")
            self._log(f"  脚本: {config.infer_script}")
            self._log(f"  图像: {config.input_image}")
            self._log(f"  SAM-Road 模型权重: {config.samroad_model_ckpt_path}")
            self._log("=" * 60)

        # ── 禁用按钮，显示进度 ──
        self._btn_run.setEnabled(False)
        self._btn_run.setText("⏳ 正在运行...")
        self._btn_cancel.setText("取消运行")
        self._progress_bar.show()

        self._start_time = time.time()
        self._stdout_lines = []
        self._stderr_lines = []

        if resolved_mode == "tile" and not config.dry_run:
            tile_command = build_command(config, "<tile_output_dir>")
            self._log(f"[SamRoadRun] project_dir = {config.project_dir}")
            self._log(f"[SamRoadRun] python_exe = {config.python_executable}")
            self._log(f"[SamRoadRun] infer_script = {config.infer_script}")
            self._log(f"[SamRoadRun] input_image = {config.input_image}")
            self._log(f"[SamRoadRun] output_dir = {config.output_dir}")
            self._log(f"[SamRoadRun] command = {' '.join(tile_command)}")
            self._log(f"[SamRoadRun] stdout_path = {config.output_dir / 'samroad_stdout.log'}")
            self._log(f"[SamRoadRun] stderr_path = {config.output_dir / 'samroad_stderr.log'}")
            self._start_tile_inference(config)
            return

        # ── 启动 QProcess ──
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self._process.readyReadStandardOutput.connect(self._on_stdout_ready)
        self._process.readyReadStandardError.connect(self._on_stderr_ready)
        self._process.finished.connect(self._on_process_finished)

        if config.dry_run:
            cmd = ["python", "-c",
                   f"import sys; print('DRY RUN / MOCK MODE — NOT REAL SAM-ROAD INFERENCE'); "
                   f"print(f'Image: {config.input_image}'); "
                   f"print(f'Config: {config.config_path}'); "
                   f"print(f'Checkpoint: {config.samroad_model_ckpt_path}'); "
                   f"print(f'Output: {config.output_dir}'); "
                   f"print('[DRY-RUN] 检查完成，没有执行真实 SAM-Road 推理。')"]
            python = "python"
            self._log(f"[DRY-RUN] 命令: python -c ...")
            self._process.start(python, cmd[1:])
            return

        # 真实运行：构造命令
        # infer_single.py 的 --output_dir 参数是 save/ 下的子目录名
        # 但如果传入绝对路径，os.path.join("save", abs_path) 会返回 abs_path
        # 所以直接传目标输出目录的绝对路径即可
        output_name = str(config.output_dir.absolute())
        cmd = build_command(config, output_name)

        self._log(f"[SamRoadRun] project_dir = {config.project_dir}")
        self._log(f"[SamRoadRun] python_exe = {config.python_executable}")
        self._log(f"[SamRoadRun] infer_script = {config.infer_script}")
        self._log(f"[SamRoadRun] input_image = {config.input_image}")
        self._log(f"[SamRoadRun] output_dir = {config.output_dir}")
        self._log(f"[SamRoadRun] command = {' '.join(cmd)}")
        self._log(f"[SamRoadRun] stdout_path = {config.output_dir / 'samroad_stdout.log'}")
        self._log(f"[SamRoadRun] stderr_path = {config.output_dir / 'samroad_stderr.log'}")

        # ── 运行时环境变量：解决 Windows 中文用户名 + matplotlib home 目录问题 ──

        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")

        # 注入 matplotlib / home 目录安全环境变量
        runtime_vars = prepare_runtime_env(output_dir=str(config.output_dir))
        for key, value in runtime_vars.items():
            env.insert(key, value)

        # 注入 PYTHONPATH：确保 segment_anything 等 SAM-Road 内部包可被导入
        #   D:/sam_road_single_image_share + D:/sam_road_single_image_share/sam
        project_pythonpath = prepare_project_import_paths(str(config.project_dir))
        old_pythonpath = env.value("PYTHONPATH", "")
        if old_pythonpath:
            new_pythonpath = project_pythonpath + os.pathsep + old_pythonpath
        else:
            new_pythonpath = project_pythonpath
        env.insert("PYTHONPATH", new_pythonpath)

        self._process.setProcessEnvironment(env)
        # 设置工作目录为推理包目录（因为脚本可能依赖相对路径资源）
        self._process.setWorkingDirectory(str(config.project_dir))

        self._log(f"[命令] {' '.join(cmd)}")
        self._process.start(cmd[0], cmd[1:])

    def _on_run_samroadplus(self):
        """Start the Portable bridge through QProcess without blocking Qt."""
        config = self._update_plus_config_from_form()
        errors = validate_samroadplus_config(config)
        if errors:
            QMessageBox.critical(self, "SAM-RoadPlus 配置错误", "\n".join(errors))
            return

        image_size = QImageReader(str(config.input_image)).size()
        image_width = image_size.width() if image_size.isValid() else 0
        image_height = image_size.height() if image_size.isValid() else 0
        if max(image_width, image_height) > 2048:
            reply = QMessageBox.warning(
                self, "SAM-RoadPlus 大图提示",
                "当前为大图，整图推理可能导致道路细节被压缩，建议使用 tile 推理。\n\n"
                "当前 Portable 工程自身采用 patch 滑窗推理，但 RoadNet Studio 外层 tile "
                "模式目前仅作为参数预留。是否继续运行？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        output_text = self._edit_output_dir.text().strip()
        if output_text:
            config.output_dir = Path(output_text)
        else:
            config.output_dir = create_samroadplus_output_dir(
                "outputs/samroadplus_runs", str(config.input_image)
            )
        # QProcess cwd is the Portable project. Always pass an absolute path,
        # otherwise the model writes into <project_dir>/outputs while Studio
        # checks <roadnet_tool>/outputs.
        config.output_dir = config.output_dir.expanduser().resolve()
        config.output_dir.mkdir(parents=True, exist_ok=True)
        save_samroadplus_config(config)

        self._plus_config = config
        self._config = config  # process-finished path/log handling is shared
        self._active_model_type = SAMROADPLUS_MODEL_TYPE
        self._log("=" * 60)
        self._log("  开始 SAM-RoadPlus Portable 推理")
        self._log(f"  工程: {config.project_dir}")
        self._log(f"  Python: {config.python_executable}")
        self._log(f"  脚本: {config.infer_script}")
        self._log(f"  Config: {config.config_path}")
        self._log(f"  新训练权重: {config.model_ckpt_path}")
        self._log(
            f"  SAM backbone: {config.sam_backbone_ckpt_path}"
            if config.sam_backbone_ckpt_path != Path()
            else "  SAM backbone: (config 不要求)"
        )
        self._log(f"  输出: {config.output_dir}")
        self._log(f"  Graph 策略: {'忽略' if config.ignore_graph else '仅参考层'}")
        self._log("=" * 60)

        self._btn_run.setEnabled(False)
        self._btn_run.setText("⏳ 正在运行 SAM-RoadPlus...")
        self._btn_cancel.setText("取消运行")
        self._progress_bar.setRange(0, 0)
        self._progress_bar.show()
        self._start_time = time.time()
        self._stdout_lines = []
        self._stderr_lines = []

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self._process.readyReadStandardOutput.connect(self._on_stdout_ready)
        self._process.readyReadStandardError.connect(self._on_stderr_ready)
        self._process.finished.connect(self._on_process_finished)

        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        for key, value in prepare_runtime_env(str(config.output_dir)).items():
            environment.insert(key, value)
        project_pythonpath = prepare_project_import_paths(str(config.project_dir))
        old_pythonpath = environment.value("PYTHONPATH", "")
        environment.insert(
            "PYTHONPATH",
            project_pythonpath + (os.pathsep + old_pythonpath if old_pythonpath else ""),
        )
        self._process.setProcessEnvironment(environment)
        self._process.setWorkingDirectory(str(config.project_dir))
        command = build_samroadplus_bridge_command(config)
        self._log(f"[SamRoadRun] project_dir = {config.project_dir}")
        self._log(f"[SamRoadRun] python_exe = {config.python_executable}")
        self._log(f"[SamRoadRun] infer_script = {config.infer_script}")
        self._log(f"[SamRoadRun] input_image = {config.input_image}")
        self._log(f"[SamRoadRun] output_dir = {config.output_dir}")
        self._log(f"[SamRoadRun] command = {' '.join(command)}")
        self._log(f"[SamRoadRun] stdout_path = {config.output_dir / 'samroadplus_stdout.log'}")
        self._log(f"[SamRoadRun] stderr_path = {config.output_dir / 'samroadplus_stderr.log'}")
        self._process.start(command[0], command[1:])

    def _start_tile_inference(self, config):
        from roadnet.samroad_tile_worker import SAMRoadTileWorker

        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        thread = QThread(self)
        worker = SAMRoadTileWorker(
            config=config,
            output_dir=str(config.output_dir),
            tile_size=config.tile_size,
            overlap=config.overlap,
            skip_black_tile=config.skip_black_tile,
            black_threshold=config.black_threshold,
            min_black_component_area=config.min_black_component_area,
            valid_pixel_ratio_threshold=config.valid_pixel_ratio_threshold,
            merge_method=config.merge_method,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self._log)
        worker.progress.connect(self._on_tile_progress)
        worker.finished.connect(self._on_tile_finished)
        worker.failed.connect(self._on_tile_failed)
        worker.cancelled.connect(self._on_tile_cancelled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(self._on_tile_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._tile_thread = thread
        self._tile_worker = worker
        thread.start()

    def _on_tile_progress(self, percent, current, total, message):
        self._progress_bar.setValue(int(percent))
        self._progress_bar.setFormat(f"{message} · {percent}%")

    def _restore_run_controls(self):
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(True)
        self._btn_run.setText(
            "▶ 运行 SAM-RoadPlus 推理"
            if self._active_model_type == SAMROADPLUS_MODEL_TYPE
            else "▶ 运行 SAM-Road 单图初提取"
        )
        self._btn_cancel.setText("取消")
        self._progress_bar.hide()
        self._progress_bar.setRange(0, 0)

    def _on_tile_finished(self, result):
        self._restore_run_controls()
        self._log(f"[结果] SAM-Road tile 推理成功，耗时 {result.duration_seconds:.1f}s")
        self.finished.emit(result)
        self._show_success_summary(result)

    def _on_tile_failed(self, message, error_log_path):
        self._restore_run_controls()
        self._log(f"[结果] SAM-Road tile 推理失败: {message}")
        QMessageBox.critical(
            self, "SAM-Road tile 推理失败",
            f"错误：{message}\n\n错误日志：\n{error_log_path}\n\n原有 Road Mask 未被覆盖。",
        )

    def _on_tile_cancelled(self, message):
        self._restore_run_controls()
        self._log(f"[结果] {message}；原有 Road Mask 未被覆盖")

    def _on_tile_thread_finished(self):
        self._tile_worker = None
        self._tile_thread = None

    def _on_stdout_ready(self):
        if self._process is None:
            return
        data = self._process.readAllStandardOutput()
        try:
            text = data.data().decode("utf-8", errors="replace")
        except Exception:
            text = str(data)
        self._stdout_lines.append(text)
        self._log(text.rstrip())

    def _on_stderr_ready(self):
        if self._process is None:
            return
        data = self._process.readAllStandardError()
        try:
            text = data.data().decode("utf-8", errors="replace")
        except Exception:
            text = str(data)
        self._stderr_lines.append(text)
        self._log(f"[STDERR] {text.rstrip()}")

    def _on_process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        elapsed = time.time() - self._start_time

        # 清缓冲区残留
        if self._process:
            data_out = self._process.readAllStandardOutput()
            if data_out:
                try:
                    self._stdout_lines.append(
                        data_out.data().decode("utf-8", errors="replace"))
                except Exception:
                    pass
            data_err = self._process.readAllStandardError()
            if data_err:
                try:
                    self._stderr_lines.append(
                        data_err.data().decode("utf-8", errors="replace"))
                except Exception:
                    pass

        stdout = "\n".join(self._stdout_lines)
        stderr = "\n".join(self._stderr_lines)

        # 保存日志
        config = self._config or SAMRoadSingleRunConfig()
        output_dir = config.output_dir
        is_plus = self._active_model_type == SAMROADPLUS_MODEL_TYPE
        if output_dir != Path() and output_dir.is_dir():
            try:
                stdout_name = "samroadplus_stdout.log" if is_plus else "samroad_stdout.log"
                stderr_name = "samroadplus_stderr.log" if is_plus else "samroad_stderr.log"
                (output_dir / stdout_name).write_text(
                    stdout, encoding="utf-8", errors="replace")
                (output_dir / stderr_name).write_text(
                    stderr, encoding="utf-8", errors="replace")
            except Exception:
                pass

        is_dry = self._chk_dry_run.isChecked() and not is_plus
        process_success = (exit_code == 0)
        diagnostics = {}
        if not is_dry and output_dir != Path():
            try:
                from roadnet.samroad_output_diagnostics import diagnose_and_standardize_samroad_outputs
                diagnostics = diagnose_and_standardize_samroad_outputs(
                    output_dir,
                    project_dir=getattr(config, "project_dir", None),
                    started_at=self._start_time,
                    ignore_graph=bool(getattr(config, "ignore_graph", False)),
                )
                mapped = diagnostics.get("output_mapping", {}).get("road_mask")
                if mapped:
                    self._log(
                        "[SamRoadRun] 未找到标准 road_mask.png，已映射: "
                        f"{mapped.get('source')} -> {mapped.get('standard')}"
                    )
            except Exception as exc:
                diagnostics = {
                    "road_mask_exists": False,
                    "files_found": [],
                    "all_files_found": [],
                    "warnings": [f"输出诊断失败: {exc}"],
                }
                self._log(f"[SamRoadRun] 输出诊断失败: {exc}")
        road_mask_exists = bool(diagnostics.get("road_mask_exists")) if not is_dry else True
        success = bool(process_success and road_mask_exists)
        missing_mask = bool(not is_dry and not road_mask_exists)

        self._log("")
        self._log(f"[SamRoadRun] return_code = {exit_code}")
        self._log(f"[结果] 进程已结束，返回码: {exit_code}，耗时: {elapsed:.1f}s")
        if success:
            run_name = "SAM-RoadPlus Portable" if is_plus else ("DRY-RUN" if is_dry else "SAM-Road 单图推理")
            self._log(f"[结果] {run_name} 运行成功 ✅")
        else:
            self._log(f"[结果] 运行失败 ❌")

        # 输出文件
        if output_dir != Path() and output_dir.is_dir():
            files = diagnostics.get("all_files_found") or sorted(
                str(path.relative_to(output_dir)).replace("\\", "/")
                for path in output_dir.rglob("*") if path.is_file()
            )
            if files:
                self._log(f"[输出文件] ({len(files)} 个):")
                for f in files:
                    self._log(f"   - {f}")

        # 构造结果
        result = SAMRoadSingleRunResult.from_process_result(
            return_code=exit_code,
            output_dir=output_dir,
            stdout=stdout,
            stderr=stderr,
            is_dry_run=is_dry,
        )
        result.duration_seconds = elapsed
        result.success = success
        result.model_type = SAMROADPLUS_MODEL_TYPE if is_plus else "samroad_single_image"
        result.ignore_graph = bool(getattr(config, "ignore_graph", False))
        result.output_diagnostics = diagnostics
        result.found_files = list(diagnostics.get("all_files_found", result.found_files))
        if missing_mask:
            result.error_message = "进程结束，但没有生成 road_mask 或其他候选 mask 文件"

        # 恢复 UI
        self._btn_run.setEnabled(True)
        self._btn_run.setText(
            "▶ 运行 SAM-RoadPlus 推理" if is_plus
            else "▶ 运行 SAM-Road 单图初提取"
        )
        self._btn_cancel.setText("取消")
        self._progress_bar.hide()

        # 发射信号
        if success:
            self.finished.emit(result)

        # 结果弹窗
        if success:
            self._show_success_summary(result)
        elif missing_mask:
            self._show_missing_mask_diagnostic(
                result, config, diagnostics,
                "samroadplus_stdout.log" if is_plus else "samroad_stdout.log",
                "samroadplus_stderr.log" if is_plus else "samroad_stderr.log",
            )
        else:
            err_msg = stderr.strip() or "未知错误"
            if len(err_msg) > 2000:
                err_msg = err_msg[:2000] + "\n... (已截断，完整日志见输出目录)"

            # 特定错误提示
            extra_hint = ""
            if "Could not determine home directory" in (stdout + stderr):
                extra_hint = (
                    "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "matplotlib 无法确定 home 目录，\n"
                    "已尝试设置 MPLCONFIGDIR / HOME / USERPROFILE。\n"
                    "请检查 QProcess 环境变量是否正确传入。\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                )
            elif "No module named 'segment_anything'" in (stdout + stderr):
                extra_hint = (
                    "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "无法导入 segment_anything 模块。\n"
                    "已尝试将 D:/sam_road_single_image_share/sam\n"
                    "加入 PYTHONPATH（通过 QProcess 环境变量）。\n"
                    "请确认:\n"
                    "  1) D:/sam_road_single_image_share/sam/segment_anything 目录存在\n"
                    "  2) PYTHONPATH 环境变量已正确传入 QProcess\n"
                    "  3) infer_single.py 脚本路径正确\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                )
            if "RuntimeError" in (stdout + stderr) and "home" in (stdout + stderr).lower():
                extra_hint = (
                    "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "运行时错误可能与 home 目录有关。\n"
                    "已尝试设置安全的环境变量。\n"
                    "请确认 Windows 用户名不含特殊字符，\n"
                    "或手动设置 HOME / USERPROFILE 环境变量。\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                )

            QMessageBox.critical(
                self, "SAM-Road 单图推理运行失败",
                f"返回码: {exit_code}\n\n"
                f"错误信息:\n{err_msg}\n{extra_hint}\n"
                f"完整日志已保存到:\n{output_dir}"
            )

    def _show_missing_mask_diagnostic(self, result, config, diagnostics,
                                      stdout_name, stderr_name):
        output_dir = Path(result.output_dir).resolve()
        files = diagnostics.get("all_files_found") or diagnostics.get("files_found") or []
        listed = "\n".join(f"- {name}" for name in files[:100]) or "（目录中没有输出文件）"
        if len(files) > 100:
            listed += f"\n... 另有 {len(files) - 100} 个文件"
        candidates = diagnostics.get("candidate_mask_files") or []
        candidate_text = "\n".join(f"- {name}" for name in candidates) or "（未发现候选 mask）"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("SAM-Road 输出缺少 road_mask")
        box.setText(
            "SAM-Road 推理结束，但没有生成 road_mask 或其他候选 mask 文件。\n\n"
            "已停止自动导入、mask 后处理、skeleton 和 final_graph 流程。"
        )
        box.setInformativeText(
            "请检查：\n"
            "1. 推理脚本是否支持输出 mask；\n"
            "2. output_dir 参数是否传入成功；\n"
            "3. checkpoint/config 是否匹配；\n"
            "4. samroad_stderr.log 是否有报错；\n"
            "5. SAM-RoadPlus 是否使用了不同输出目录。"
        )
        box.setDetailedText(
            f"输出目录：{output_dir}\n\n实际文件：\n{listed}\n\n候选 mask：\n{candidate_text}\n\n"
            + "\n".join(diagnostics.get("warnings") or [])
        )
        open_output = box.addButton("打开输出目录", QMessageBox.ButtonRole.ActionRole)
        open_stdout = box.addButton("打开 stdout", QMessageBox.ButtonRole.ActionRole)
        open_stderr = box.addButton("打开 stderr", QMessageBox.ButtonRole.ActionRole)
        open_project = box.addButton("打开 SAM-RoadPlus 工程目录", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        selected = box.clickedButton()
        target = None
        if selected is open_output:
            target = output_dir
        elif selected is open_stdout:
            target = output_dir / stdout_name
        elif selected is open_stderr:
            target = output_dir / stderr_name
        elif selected is open_project:
            target = Path(getattr(config, "project_dir", output_dir))
        if target is not None:
            if not target.exists():
                QMessageBox.warning(self, "无法打开", f"文件或目录不存在：\n{target}")
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _show_success_summary(self, result: SAMRoadSingleRunResult):
        is_dry = result.is_dry_run
        is_partial = result.is_partial_load
        is_plus = getattr(result, "model_type", "") == SAMROADPLUS_MODEL_TYPE
        if is_plus:
            title = "SAM-RoadPlus Portable 推理完成 ✅"
        elif is_dry:
            title = "DRY-RUN 完成 ✅"
        elif is_partial:
            title = "Partial Load 完成 ⚠️"
        else:
            title = "SAM-Road 单图推理完成 ✅"

        parts = []
        parts.append(f"耗时: {result.duration_seconds:.1f} 秒")
        if is_dry:
            parts.append("\n⚠️ MOCK OUTPUT — 非真实 SAM-Road 推理结果")
        if is_partial:
            parts.append("\n⚠️ PARTIAL LOAD — road_mask 可用，graph 不可靠！")
        if result.node_count > 0:
            parts.append(f"节点数: {result.node_count}")
            parts.append(f"边数: {result.edge_count}")
        if result.found_files:
            parts.append(f"\n输出文件 ({len(result.found_files)}):")
            for f in result.found_files:
                parts.append(f"  • {f}")
        mapping = getattr(result, "output_diagnostics", {}).get("output_mapping", {}).get("road_mask")
        if mapping and Path(mapping.get("source", "")).resolve() != Path(mapping.get("standard", "")).resolve():
            parts.append(
                "\n未找到 road_mask.png，已使用 "
                f"{Path(mapping['source']).name} 作为 road_mask。"
            )

        QMessageBox.information(
            self, title,
            f"推理完成。输出目录:\n{result.output_dir}\n\n" + "\n".join(parts)
        )

    # ===================================================================
    # 取消/关闭
    # ===================================================================

    def _restore_window_geometry(self):
        geometry = self._window_settings.value("window_geometry")
        if geometry is not None:
            self._had_saved_geometry = bool(self.restoreGeometry(geometry))

    def _available_screen_geometry(self):
        parent = self.parentWidget()
        screen = parent.screen() if parent is not None else QApplication.primaryScreen()
        return screen.availableGeometry() if screen is not None else None

    def _fit_to_available_screen(self):
        """限制窗口到屏幕可用区 90%，必要时自动居中。"""
        available = self._available_screen_geometry()
        if available is None:
            return
        max_width = max(640, int(available.width() * 0.90))
        max_height = max(480, int(available.height() * 0.90))

        # 极小屏幕时动态降低 minimum，避免 minimumSize 反而把按钮挤出屏幕。
        self.setMinimumSize(min(760, max_width), min(560, max_height))
        target_width = min(self.width(), max_width)
        target_height = min(self.height(), max_height)
        constrained = target_width != self.width() or target_height != self.height()
        if constrained:
            self.resize(target_width, target_height)

        frame = self.frameGeometry()
        outside = not available.intersects(frame)
        if constrained or outside or not self._had_saved_geometry:
            frame.moveCenter(available.center())
            self.move(frame.topLeft())

    def _save_window_geometry(self):
        self._window_settings.setValue("window_geometry", self.saveGeometry())
        self._window_settings.sync()

    def done(self, result):
        self._save_window_geometry()
        super().done(result)

    def _on_cancel(self):
        if self._tile_thread is not None and self._tile_thread.isRunning():
            self._log("[操作] 正在取消 SAM-Road tile 推理...")
            self._tile_worker.cancel()
            self._btn_cancel.setEnabled(False)
        elif self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._log("[操作] 正在终止进程...")
            self._process.kill()
            self._process.waitForFinished(3000)
            self._log("[操作] 进程已终止")
            self._btn_run.setEnabled(True)
            self._btn_run.setText("▶ 运行 SAM-Road 单图初提取")
            self._btn_cancel.setText("关闭")
            self._progress_bar.hide()
        else:
            self.reject()

    def closeEvent(self, event):
        tile_running = self._tile_thread is not None and self._tile_thread.isRunning()
        process_running = (
            self._process and self._process.state() != QProcess.ProcessState.NotRunning
        )
        if tile_running or process_running:
            reply = QMessageBox.question(
                self, "确认",
                "SAM-Road 单图推理正在运行，确定要终止并关闭吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                if tile_running:
                    self._tile_worker.cancel()
                    self._tile_thread.quit()
                    if not self._tile_thread.wait(5000):
                        QMessageBox.information(
                            self, "正在取消", "后台 tile 进程仍在退出，请稍后再次关闭窗口。"
                        )
                        event.ignore()
                        return
                if process_running:
                    self._process.kill()
                    self._process.waitForFinished(1000)
                self._save_window_geometry()
                event.accept()
            else:
                event.ignore()
        else:
            self._save_window_geometry()
            event.accept()

    # ===================================================================
    # 工具方法
    # ===================================================================

    def _log(self, text: str):
        self._log_view.append(text)
        self._log_view.ensureCursorVisible()

    @property
    def config(self) -> Optional[SAMRoadSingleRunConfig]:
        return self._config
