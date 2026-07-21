"""
SAM-Road 运行参数对话框。

提供完整的参数配置界面，支持：
- 路径浏览/扫描
- 参数保存/恢复
- QProcess 异步运行（不阻塞 GUI）
- 实时日志显示
- dry-run / mock 模式
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QProcess, Signal, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QCheckBox, QComboBox,
    QTextEdit, QLabel, QFileDialog, QMessageBox,
    QGroupBox, QProgressBar, QDialogButtonBox,
    QSpinBox, QWidget, QSplitter, QListWidget, QListWidgetItem,
)

from roadnet.samroad_runner import (
    SAMRoadRunConfig, load_config, save_config,
    dict_to_runconfig, runconfig_to_dict,
    validate_config, build_command, create_output_dir,
    scan_entry_scripts, SAMRoadRunResult,
)


# 默认配置文件路径
DEFAULT_CONFIG_PATH = "config/samroad_config.yaml"


class SAMRoadRunDialog(QDialog):
    """SAM-Road 运行参数对话框。"""

    # 信号：运行完成时发射
    finished = Signal(object)  # SAMRoadRunResult

    def __init__(self, image_path: str = "", parent=None):
        super().__init__(parent)
        self._image_path = image_path
        self._config: Optional[SAMRoadRunConfig] = None
        self._process: Optional[QProcess] = None
        self._start_time: float = 0.0
        self._stdout_lines: list[str] = []
        self._stderr_lines: list[str] = []

        self.setWindowTitle("运行 SAM-Road 初提取")
        self.setMinimumWidth(750)
        self.setMinimumHeight(600)
        self.setModal(False)  # 非模态，避免完全阻塞

        self._setup_ui()
        self._load_persisted_config()
        self._update_form_from_config()

    # ===================================================================
    # UI 构建
    # ===================================================================

    def _setup_ui(self):
        """构建完整 UI 布局。"""
        main_layout = QVBoxLayout(self)

        # ── 上半部分：参数表单 ──
        form_scroll = QWidget()
        form_layout = QVBoxLayout(form_scroll)

        # --- 路径组 ---
        path_group = QGroupBox("📂 SAM-Road 路径配置")
        path_form = QFormLayout(path_group)

        self._edit_project_dir = self._create_path_row(
            path_form, "SAM-Road 项目目录:", self._on_browse_project_dir
        )
        self._edit_python = self._create_path_row(
            path_form, "Python 解释器:", self._on_browse_python,
            hint="例如: D:/Anaconda/envs/samroad/python.exe"
        )
        self._edit_bridge = self._create_path_row(
            path_form, "推理脚本:", self._on_browse_bridge,
            hint="留空可扫描 SAM-Road 项目目录自动查找"
        )

        # 扫描项目目录按钮
        btn_row = QHBoxLayout()
        btn_scan_scripts = QPushButton("🔍 扫描项目目录查找入口脚本")
        btn_scan_scripts.clicked.connect(self._on_scan_entry_scripts)
        btn_row.addWidget(btn_scan_scripts)
        btn_scan_checkpoints = QPushButton("🔍 扫描查找 checkpoint")
        btn_scan_checkpoints.clicked.connect(self._on_scan_checkpoints)
        btn_row.addWidget(btn_scan_checkpoints)
        path_form.addRow("", btn_row)

        self._edit_checkpoint = self._create_path_row(
            path_form, "SAM-Road 模型权重:", self._on_browse_samroad_ckpt,
            hint="D:/sam_road_single_image_share/checkpoints/model.ckpt"
        )
        self._edit_sam_backbone = self._create_path_row(
            path_form, "SAM backbone 权重:", self._on_browse_sam_backbone,
            hint="D:/sam_road_single_image_share/sam_ckpts/sam_vit_b_01ec64.pth"
        )
        self._edit_config_file = self._create_path_row(
            path_form, "Config 文件:", self._on_browse_config,
            hint="相对于 SAM-Road 项目目录的 yaml 配置文件"
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

        form_layout.addWidget(path_group)

        # --- 图像确认 ---
        img_group = QGroupBox("📷 输入图像")
        img_layout = QFormLayout(img_group)
        self._lbl_image = QLabel(self._image_path or "(无)")
        self._lbl_image.setWordWrap(True)
        self._lbl_image.setStyleSheet("font-weight: bold; color: #333;")
        img_layout.addRow("当前图像:", self._lbl_image)
        form_layout.addWidget(img_group)

        # --- 运行参数组 ---
        param_group = QGroupBox("⚙️ 运行参数")
        param_form = QFormLayout(param_group)

        spin_row = QHBoxLayout()
        self._spin_tile_size = QSpinBox()
        self._spin_tile_size.setRange(256, 4096)
        self._spin_tile_size.setValue(1024)
        self._spin_tile_size.setSuffix(" px")
        spin_row.addWidget(QLabel("Tile Size:"))
        spin_row.addWidget(self._spin_tile_size)

        self._spin_overlap = QSpinBox()
        self._spin_overlap.setRange(0, 512)
        self._spin_overlap.setValue(128)
        self._spin_overlap.setSuffix(" px")
        spin_row.addWidget(QLabel("Overlap:"))
        spin_row.addWidget(self._spin_overlap)
        spin_row.addStretch()
        param_form.addRow("", spin_row)

        device_row = QHBoxLayout()
        self._combo_device = QComboBox()
        self._combo_device.addItems(["cuda", "cpu"])
        device_row.addWidget(QLabel("Device:"))
        device_row.addWidget(self._combo_device)
        device_row.addStretch()
        param_form.addRow("", device_row)

        self._edit_output_dir = self._create_path_row(
            param_form, "输出目录:", self._on_browse_output_dir,
            hint="留空则自动创建: outputs/samroad_<image>_<timestamp>/"
        )

        form_layout.addWidget(param_group)

        # --- 选项组 ---
        option_group = QGroupBox("📋 选项")
        option_form = QFormLayout(option_group)
        self._chk_auto_import = QCheckBox("运行完成后自动导入结果")
        self._chk_auto_import.setChecked(True)
        option_form.addRow(self._chk_auto_import)

        self._chk_dry_run = QCheckBox("Dry-run / Mock 测试模式（不调用真实推理）")
        self._chk_dry_run.setToolTip(
            "仅生成 mock 数据测试 UI 流程，不调用真实 SAM-Road 模型。\n"
            "所有输出都会标注为 MOCK OUTPUT。"
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
        form_layout.addWidget(option_group)

        main_layout.addWidget(form_scroll)

        # ── 下半部分：日志输出 ──
        log_group = QGroupBox("📜 运行日志")
        log_layout = QVBoxLayout(log_group)
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(self._log_view.document().defaultFont())
        self._log_view.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas, monospace; font-size: 12px; }"
        )
        self._log_view.setMinimumHeight(120)
        log_layout.addWidget(self._log_view)
        main_layout.addWidget(log_group)

        # ── 底部按钮栏 ──
        btn_layout = QHBoxLayout()

        self._btn_save_config = QPushButton("💾 保存配置")
        self._btn_save_config.clicked.connect(self._on_save_config)
        btn_layout.addWidget(self._btn_save_config)

        btn_layout.addStretch()

        self._btn_cancel = QPushButton("取消")
        self._btn_cancel.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self._btn_cancel)

        self._btn_run = QPushButton("▶ 运行 SAM-Road 初提取")
        self._btn_run.setStyleSheet(
            "QPushButton { background-color: #0078d4; color: white; font-weight: bold; "
            "padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #106ebe; }"
            "QPushButton:disabled { background-color: #ccc; }"
        )
        self._btn_run.clicked.connect(self._on_run)
        btn_layout.addWidget(self._btn_run)

        main_layout.addLayout(btn_layout)

        # ── 进度条（运行中显示） ──
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # 不确定进度
        self._progress_bar.hide()
        main_layout.addWidget(self._progress_bar)

    def _create_path_row(
        self, form: QFormLayout, label: str, browse_handler,
        hint: str = ""
    ) -> QLineEdit:
        """创建统一风格的行：标签 + [输入框 + 浏览按钮]。"""
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
        """加载持久化的 YAML 配置。"""
        data = load_config(DEFAULT_CONFIG_PATH)
        if not data:
            self._config = SAMRoadRunConfig()
            return
        self._config = dict_to_runconfig(data)
        # 确保 bridge_script 存在（可能是项目中相对路径）
        if not self._config.bridge_script.is_absolute():
            # 相对路径 → 绝对路径（相对于项目根目录）
            # 这里不强制转换，由后续验证处理
            pass

    def _update_form_from_config(self):
        """将配置对象的值同步到表单控件。"""
        if self._config is None:
            return
        c = self._config
        self._edit_project_dir.setText(str(c.project_dir) if c.project_dir != Path() else "")
        self._edit_python.setText(str(c.python_executable) if c.python_executable != Path() else "")
        self._edit_bridge.setText(str(c.bridge_script) if c.bridge_script != Path() else "")
        self._edit_checkpoint.setText(str(c.samroad_model_ckpt_path) if c.samroad_model_ckpt_path != Path() else "")
        self._edit_sam_backbone.setText(str(c.sam_backbone_ckpt_path) if c.sam_backbone_ckpt_path != Path() else "")
        self._edit_config_file.setText(str(c.config_file) if c.config_file != Path() else "")
        self._spin_tile_size.setValue(c.tile_size)
        self._spin_overlap.setValue(c.overlap)
        idx = self._combo_device.findText(c.device)
        if idx >= 0:
            self._combo_device.setCurrentIndex(idx)
        self._chk_auto_import.setChecked(c.auto_import_after_run)
        self._chk_dry_run.setChecked(c.dry_run)
        self._chk_partial_load.setChecked(c.mask_only_partial_load)

    def _update_config_from_form(self) -> SAMRoadRunConfig:
        """从表单控件同步值到配置对象。"""
        c = self._config or SAMRoadRunConfig()
        c.project_dir = Path(self._edit_project_dir.text()) if self._edit_project_dir.text() else c.project_dir
        c.python_executable = Path(self._edit_python.text()) if self._edit_python.text() else c.python_executable
        c.bridge_script = Path(self._edit_bridge.text()) if self._edit_bridge.text() else c.bridge_script
        c.sam_backbone_ckpt_path = Path(self._edit_sam_backbone.text()) if self._edit_sam_backbone.text() else c.sam_backbone_ckpt_path
        c.samroad_model_ckpt_path = Path(self._edit_checkpoint.text()) if self._edit_checkpoint.text() else c.samroad_model_ckpt_path
        c.config_file = Path(self._edit_config_file.text()) if self._edit_config_file.text() else c.config_file
        c.input_image = Path(self._image_path) if self._image_path else c.input_image
        # 确保 config_file 为绝对路径
        if c.config_file != Path() and not c.config_file.is_absolute():
            c.config_file = (c.project_dir / c.config_file).resolve()
        if c.sam_backbone_ckpt_path != Path() and not c.sam_backbone_ckpt_path.is_absolute():
            c.sam_backbone_ckpt_path = (c.project_dir / c.sam_backbone_ckpt_path).resolve()
        if c.samroad_model_ckpt_path != Path() and not c.samroad_model_ckpt_path.is_absolute():
            c.samroad_model_ckpt_path = (c.project_dir / c.samroad_model_ckpt_path).resolve()
        c.tile_size = self._spin_tile_size.value()
        c.overlap = self._spin_overlap.value()
        c.device = self._combo_device.currentText()
        c.auto_import_after_run = self._chk_auto_import.isChecked()
        c.dry_run = self._chk_dry_run.isChecked()
        c.mask_only_partial_load = self._chk_partial_load.isChecked()
        self._config = c
        return c

    # ===================================================================
    # 浏览按钮回调
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
        self._browse_dir(edit, "选择 SAM-Road 项目目录")

    def _on_browse_python(self, edit: QLineEdit):
        filter_str = "python.exe (python.exe);;所有文件 (*)"
        self._browse_file(edit, "选择 Python 解释器", filter_str)

    def _on_browse_bridge(self, edit: QLineEdit):
        filter_str = "Python 脚本 (*.py);;所有文件 (*)"
        self._browse_file(edit, "选择推理脚本", filter_str)

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
        if "sam_vit_b" in lower:
            return True
        if "sam_ckpts" in lower.replace("\\", "/"):
            return True
        return False

    def _on_browse_config(self, edit: QLineEdit):
        filter_str = "YAML 文件 (*.yaml *.yml);;所有文件 (*)"
        project_dir = self._edit_project_dir.text()
        start_dir = project_dir if project_dir and os.path.isdir(project_dir) else ""
        self._browse_file(edit, "选择配置文件", filter_str, start_dir)

    def _on_browse_output_dir(self, edit: QLineEdit):
        self._browse_dir(edit, "选择输出目录")

    # ===================================================================
    # 扫描
    # ===================================================================

    def _on_scan_entry_scripts(self):
        """扫描 SAM-Road 项目目录中的推理入口脚本。"""
        project_dir = self._edit_project_dir.text()
        if not project_dir or not os.path.isdir(project_dir):
            QMessageBox.warning(self, "提示", "请先选择有效的 SAM-Road 项目目录。")
            return

        scripts = scan_entry_scripts(project_dir)
        if not scripts:
            QMessageBox.information(self, "扫描结果", "未在项目目录中找到已知推理入口脚本。")
            return

        # 弹窗让用户选择
        dialog = QDialog(self)
        dialog.setWindowTitle("选择推理入口脚本")
        dialog.setMinimumWidth(500)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"在 {project_dir} 中找到以下脚本:"))
        list_widget = QListWidget()
        for s in scripts:
            item = QListWidgetItem(str(s.relative_to(project_dir)))
            item.setData(Qt.ItemDataRole.UserRole, str(s))
            list_widget.addItem(item)
        layout.addWidget(list_widget)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        if dialog.exec() == QDialog.DialogCode.Accepted and list_widget.currentItem():
            selected = list_widget.currentItem().data(Qt.ItemDataRole.UserRole)
            self._edit_bridge.setText(selected)
            self._log(f"[信息] 已选择入口脚本: {selected}")

    def _on_scan_checkpoints(self):
        """扫描 SAM-Road 项目目录中的 checkpoint 文件。"""
        from roadnet.samroad_runner import scan_checkpoints

        project_dir = self._edit_project_dir.text()
        if not project_dir or not os.path.isdir(project_dir):
            QMessageBox.warning(self, "提示", "请先选择有效的 SAM-Road 项目目录。")
            return

        checkpoints = scan_checkpoints(project_dir)
        if not checkpoints:
            QMessageBox.information(
                self, "扫描结果",
                f"未在 {project_dir} 中找到 checkpoint 文件。\n\n"
                "请从 huggingface 下载模型权重：\n"
                "https://huggingface.co/congrui/sam_road"
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("选择 Checkpoint 文件")
        dialog.setMinimumWidth(500)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"在 {project_dir} 中找到以下 checkpoint:"))
        list_widget = QListWidget()
        for cp in checkpoints:
            item = QListWidgetItem(str(cp.relative_to(project_dir)))
            item.setData(Qt.ItemDataRole.UserRole, str(cp))
            list_widget.addItem(item)
        layout.addWidget(list_widget)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        if dialog.exec() == QDialog.DialogCode.Accepted and list_widget.currentItem():
            selected = list_widget.currentItem().data(Qt.ItemDataRole.UserRole)
            self._edit_checkpoint.setText(selected)
            self._log(f"[信息] 已选择 checkpoint: {selected}")

    # ===================================================================
    # 配置保存
    # ===================================================================

    def _on_save_config(self):
        """保存当前配置到 YAML 文件。"""
        try:
            config = self._update_config_from_form()
            data = runconfig_to_dict(config)
            save_config(DEFAULT_CONFIG_PATH, data)
            self._log("[信息] 配置已保存到 config/samroad_config.yaml")
            QMessageBox.information(self, "保存成功",
                                    f"配置已保存到:\n{DEFAULT_CONFIG_PATH}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

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
        project_dir = str(config.project_dir)

        # ★ 匹配检查只使用 samroad_model_ckpt_path
        samroad_ckpt = str(config.samroad_model_ckpt_path)

        cmd = [
            str(config.python_executable) if config.python_executable != Path() else sys.executable,
            str(tools_script),
            "--checkpoint", samroad_ckpt,
            "--config", str(config.config_file),
            "--project-dir", project_dir,
        ]
        if match_mode:
            cmd.append("--match")

        # 注入 PYTHONPATH
        import os
        from roadnet.samroad_single_runner import prepare_project_import_paths
        env = {**os.environ, "PYTHONPATH": prepare_project_import_paths(project_dir),
               "PYTHONUNBUFFERED": "1", "WANDB_MODE": "disabled"}

        self._log(f"[匹配检查] 运行: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=120,
                cwd=project_dir,
                env=env,
            )
            return proc.stdout + "\n" + proc.stderr
        except Exception as e:
            return f"[ERROR] 匹配检查失败: {e}"

    def _on_check_match(self):
        """检查当前 config 和 checkpoint 的 shape 是否匹配。"""
        self._log("=" * 40)
        self._log("[匹配检查] 正在检查 config/checkpoint 匹配...")

        # ★ 检查用户是否误选了 SAM backbone 作为 SAM-Road 模型权重
        samroad_path = self._edit_checkpoint.text()
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

        if "SHAPE MISMATCHES" in output:
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

            msg = "Config 与 Checkpoint 不匹配！\n\n"
            msg += f"Config: {self._edit_config_file.text()}\n"
            msg += f"SAM-Road 模型权重: {self._edit_checkpoint.text()}\n\n"
            for m in mismatches[:5]:
                msg += f"  {m}\n"
            if len(mismatches) > 5:
                msg += f"  ... 和其他 {len(mismatches) - 5} 个参数\n"
            msg += "\n请点击「扫描匹配组合」或启用「mask-only partial load」。"
            self._log(msg)
            QMessageBox.warning(self, "匹配检查结果", msg)
        elif "OK: All shapes match" in output:
            msg = "Config 和 Checkpoint 完全匹配！可以安全运行。"
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

        if "Found matching combination" in output or "MATCH!" in output:
            msg = "找到匹配组合！详见运行日志。"
            self._log(msg)
            QMessageBox.information(self, "扫描结果", msg)
        elif "NO matching" in output:
            msg = (
                "未找到匹配的 config/checkpoint 组合。\n\n"
                "当前 checkpoint 与现有 config 均不匹配。\n"
                "请提供与 checkpoint 匹配的 config，或更换 checkpoint。\n\n"
                "可启用「mask-only partial load」临时调试 road_mask。\n"
                "partial 模式下 graph 不可靠。"
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
        """点击运行按钮。"""
        config = self._update_config_from_form()
        config.auto_import_after_run = self._chk_auto_import.isChecked()

        if not config.dry_run:
            # 验证路径
            errors = validate_config(config)
            if errors:
                QMessageBox.critical(self, "配置错误", "\n".join(errors))
                return

            # Partial load 模式警告
            if config.mask_only_partial_load:
                reply = QMessageBox.warning(
                    self, "Partial Load 调试模式",
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

        # 确定输出目录
        output_dir_text = self._edit_output_dir.text().strip()
        if output_dir_text:
            config.output_dir = Path(output_dir_text)
        else:
            config.output_dir = create_output_dir(
                base_dir="outputs/samroad_runs",
                image_path=str(config.input_image),
            )

        config.output_dir = config.output_dir.expanduser().resolve()
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        try:
            data = runconfig_to_dict(config)
            save_config(DEFAULT_CONFIG_PATH, data)
        except Exception:
            pass  # 静默保存

        self._log(f"[信息] 输出目录: {config.output_dir}")

        if config.dry_run:
            self._log("=" * 60)
            self._log("  DRY RUN / MOCK MODE — 不会调用真实 SAM-Road 推理")
            self._log("=" * 60)
        else:
            self._log("=" * 60)
            self._log(f"  开始 SAM-Road 推理")
            self._log(f"  Python: {config.python_executable}")
            self._log(f"  SAM backbone 权重: {config.sam_backbone_ckpt_path}")
            self._log(f"  SAM-Road 模型权重: {config.samroad_model_ckpt_path}")
            self._log(f"  图像: {config.input_image}")
            self._log("=" * 60)

        # 禁用运行按钮，显示进度
        self._btn_run.setEnabled(False)
        self._btn_run.setText("⏳ 正在运行...")
        self._btn_cancel.setText("取消运行")
        self._progress_bar.show()

        import time
        self._start_time = time.time()
        self._stdout_lines = []
        self._stderr_lines = []

        # 启动 QProcess
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self._process.readyReadStandardOutput.connect(self._on_stdout_ready)
        self._process.readyReadStandardError.connect(self._on_stderr_ready)
        self._process.finished.connect(self._on_process_finished)

        cmd = build_command(config)
        from PySide6.QtCore import QProcessEnvironment

        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")

        # 注入 matplotlib / home 目录安全环境变量
        from roadnet.samroad_single_runner import prepare_runtime_env, prepare_project_import_paths
        runtime_vars = prepare_runtime_env(output_dir=str(config.output_dir))
        for key, value in runtime_vars.items():
            env.insert(key, value)

        # 注入 PYTHONPATH：确保 segment_anything 等 SAM-Road 内部包可被导入
        project_pythonpath = prepare_project_import_paths(str(config.project_dir))
        old_pythonpath = env.value("PYTHONPATH", "")
        if old_pythonpath:
            new_pythonpath = project_pythonpath + os.pathsep + old_pythonpath
        else:
            new_pythonpath = project_pythonpath
        env.insert("PYTHONPATH", new_pythonpath)

        self._process.setProcessEnvironment(env)
        # 设置工作目录为推理包目录（确保 config 中相对路径能在正确上下文中解析）
        self._process.setWorkingDirectory(str(config.project_dir))

        self._log(f"[SamRoadRun] project_dir = {config.project_dir}")
        self._log(f"[SamRoadRun] python_exe = {config.python_executable}")
        self._log(f"[SamRoadRun] infer_script = {config.bridge_script}")
        self._log(f"[SamRoadRun] input_image = {config.input_image}")
        self._log(f"[SamRoadRun] output_dir = {config.output_dir}")
        self._log(f"[SamRoadRun] command = {' '.join(cmd)}")
        self._log(f"[SamRoadRun] stdout_path = {config.output_dir / 'samroad_stdout.log'}")
        self._log(f"[SamRoadRun] stderr_path = {config.output_dir / 'samroad_stderr.log'}")
        self._process.start(cmd[0], cmd[1:])

    def _on_stdout_ready(self):
        """接收进程 stdout 输出。"""
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
        """接收进程 stderr 输出。"""
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
        """进程结束回调。"""
        import time
        elapsed = time.time() - self._start_time

        # 清掉缓冲区残留
        if self._process:
            data_out = self._process.readAllStandardOutput()
            if data_out:
                try:
                    self._stdout_lines.append(data_out.data().decode("utf-8", errors="replace"))
                except Exception:
                    pass
            data_err = self._process.readAllStandardError()
            if data_err:
                try:
                    self._stderr_lines.append(data_err.data().decode("utf-8", errors="replace"))
                except Exception:
                    pass

        stdout = "\n".join(self._stdout_lines)
        stderr = "\n".join(self._stderr_lines)

        # 保存日志到文件
        config = self._config or SAMRoadRunConfig()
        output_dir = config.output_dir
        if output_dir != Path():
            try:
                log_dir = Path(output_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / "samroad_stdout.log").write_text(stdout, encoding="utf-8", errors="replace")
                (log_dir / "samroad_stderr.log").write_text(stderr, encoding="utf-8", errors="replace")
            except Exception:
                pass

        is_dry = self._chk_dry_run.isChecked()
        process_success = (exit_code == 0)
        diagnostics = {}
        if not is_dry and output_dir != Path():
            try:
                from roadnet.samroad_output_diagnostics import diagnose_and_standardize_samroad_outputs
                diagnostics = diagnose_and_standardize_samroad_outputs(
                    output_dir,
                    project_dir=getattr(config, "project_dir", None),
                    started_at=self._start_time,
                )
            except Exception as exc:
                diagnostics = {"road_mask_exists": False, "warnings": [str(exc)]}
        missing_mask = bool(not is_dry and not diagnostics.get("road_mask_exists"))
        success = bool(process_success and (is_dry or not missing_mask))

        self._log("")
        self._log(f"[结果] 进程已结束，返回码: {exit_code}，耗时: {elapsed:.1f}s")
        if success:
            self._log(f"[结果] {'DRY-RUN' if is_dry else 'SAM-Road'} 运行成功 ✅")
        else:
            self._log(f"[结果] 运行失败 ❌")

        # 检查输出文件
        if output_dir != Path() and output_dir.is_dir():
            files = sorted([f.name for f in output_dir.iterdir() if f.is_file()])
            if files:
                self._log(f"[输出文件] ({len(files)} 个):")
                for f in files:
                    self._log(f"   - {f}")
            else:
                self._log("[输出文件] 无")

        # 构造结果
        result = SAMRoadRunResult.from_process_result(
            return_code=exit_code,
            output_dir=output_dir,
            stdout=stdout,
            stderr=stderr,
            is_dry_run=is_dry,
        )
        result.duration_seconds = elapsed
        result.success = success
        result.output_diagnostics = diagnostics
        if diagnostics.get("all_files_found"):
            result.found_files = list(diagnostics["all_files_found"])

        # 恢复 UI
        self._btn_run.setEnabled(True)
        self._btn_run.setText("▶ 运行 SAM-Road 初提取")
        self._btn_cancel.setText("取消")
        self._progress_bar.hide()

        # 发射信号
        if success:
            self.finished.emit(result)

        # 显示结果弹窗
        if success:
            self._show_success_summary(result)
        elif missing_mask:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle("SAM-Road 输出缺少 road_mask")
            box.setText(
                "SAM-Road 推理结束，但没有生成 road_mask 或其他候选 mask 文件。\n"
                "已停止自动导入和后续处理。"
            )
            files = diagnostics.get("all_files_found") or []
            box.setDetailedText("实际输出文件：\n" + (
                "\n".join(f"- {name}" for name in files) or "（无）"
            ))
            open_output = box.addButton("打开输出目录", QMessageBox.ButtonRole.ActionRole)
            open_stderr = box.addButton("打开 stderr", QMessageBox.ButtonRole.ActionRole)
            box.addButton(QMessageBox.StandardButton.Close)
            box.exec()
            target = None
            if box.clickedButton() is open_output:
                target = output_dir
            elif box.clickedButton() is open_stderr:
                target = Path(output_dir) / "samroad_stderr.log"
            if target is not None and Path(target).exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(target).resolve())))
        else:
            # 失败弹窗
            err_msg = stderr.strip() or "未知错误"
            if len(err_msg) > 2000:
                err_msg = err_msg[:2000] + "\n... (已截断，完整日志见输出目录)"
            QMessageBox.critical(
                self, "SAM-Road 运行失败",
                f"返回码: {exit_code}\n\n"
                f"错误信息:\n{err_msg}\n\n"
                f"完整日志已保存到:\n{output_dir}"
            )

    def _show_success_summary(self, result: SAMRoadRunResult):
        """显示成功运行摘要。"""
        is_dry = result.is_dry_run
        is_partial = result.is_partial_load
        if is_dry:
            title = "DRY-RUN 完成 ✅"
        elif is_partial:
            title = "Partial Load 完成 ⚠️"
        else:
            title = "SAM-Road 运行完成 ✅"

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
        if is_dry:
            parts.append(
                "\n💡 Dry-run 模式完成。"
                "如需真实推理，请取消勾选 dry-run 并提供有效的 checkpoint 路径后重新运行。"
            )

        QMessageBox.information(
            self, title,
            f"推理完成。输出目录:\n{result.output_dir}\n\n" + "\n".join(parts)
        )

    # ===================================================================
    # 取消/关闭
    # ===================================================================

    def _on_cancel(self):
        """取消运行或关闭对话框。"""
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            # 正在运行，终止进程
            self._log("[操作] 正在终止进程...")
            self._process.kill()
            self._process.waitForFinished(3000)
            self._log("[操作] 进程已终止")
            self._btn_run.setEnabled(True)
            self._btn_run.setText("▶ 运行 SAM-Road 初提取")
            self._btn_cancel.setText("关闭")
            self._progress_bar.hide()
        else:
            self.reject()

    def closeEvent(self, event):
        """关闭窗口时确认是否终止进程。"""
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            reply = QMessageBox.question(
                self, "确认",
                "SAM-Road 正在运行，确定要终止并关闭吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._process.kill()
                self._process.waitForFinished(1000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    # ===================================================================
    # 工具方法
    # ===================================================================

    def _log(self, text: str):
        """追加日志到日志视图。"""
        self._log_view.append(text)
        # 自动滚动到底部
        self._log_view.ensureCursorVisible()

    @property
    def config(self) -> Optional[SAMRoadRunConfig]:
        return self._config
