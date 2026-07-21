"""
SAM-Road mask 后处理参数对话框。

打开时默认使用比赛固定参数（或用户显式「保存为默认参数」后的值），
不读取项目里的旧 postprocess 配置覆盖。
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QSlider, QSpinBox, QCheckBox, QPushButton, QLabel,
    QGroupBox, QMessageBox, QFileDialog, QTextEdit,
)
from PySide6.QtCore import Qt, Signal
import numpy as np
from pathlib import Path

from roadnet.mask_postprocess import (
    MaskPostprocessConfig,
    process_mask_from_layer,
    load_dialog_defaults,
    save_user_defaults,
)


class MaskPostprocessDialog(QDialog):
    """SAM-Road mask 后处理参数对话框"""

    preview_requested = Signal(object, str)
    apply_requested = Signal(object)

    def __init__(
        self,
        mask_data: np.ndarray,
        roi_data: np.ndarray | None = None,
        ignore_data: np.ndarray | None = None,
        output_dir: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._mask_data = mask_data
        self._roi_data = roi_data
        self._ignore_data = ignore_data
        self._output_dir = output_dir
        self._processed = np.asarray(mask_data).copy()
        self._steps = []
        self._log_lines: list[str] = []

        # 打开窗口：比赛默认，或用户曾点击「保存为默认参数」的值；不读项目旧参数
        self._startup_config = load_dialog_defaults()

        self.setWindowTitle("SAM-Road Mask 后处理")
        self.setMinimumWidth(520)
        self._build_ui()
        self._apply_config_to_widgets(self._startup_config)
        self._update_preview()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        params_group = QGroupBox("后处理参数（比赛默认：threshold=240 / blur=3 / close=5 / open=3 / min_area=500）")
        form = QFormLayout()

        self._threshold_slider = QSlider(Qt.Horizontal)
        self._threshold_slider.setRange(0, 255)
        self._threshold_spin = QSpinBox()
        self._threshold_spin.setRange(0, 255)
        self._threshold_slider.valueChanged.connect(self._threshold_spin.setValue)
        self._threshold_spin.valueChanged.connect(self._threshold_slider.setValue)
        row = QHBoxLayout()
        row.addWidget(self._threshold_slider)
        row.addWidget(self._threshold_spin)
        form.addRow("阈值 (threshold):", row)

        self._blur_spin = QSpinBox()
        self._blur_spin.setRange(0, 15)
        self._blur_spin.setSingleStep(2)
        self._blur_spin.setToolTip("高斯模糊核大小（奇数，0=不模糊）")
        form.addRow("模糊核 (blur):", self._blur_spin)

        self._close_spin = QSpinBox()
        self._close_spin.setRange(0, 31)
        self._close_spin.setToolTip("闭运算核大小（连接断裂，0=不闭合）")
        form.addRow("闭运算核 (close):", self._close_spin)

        self._open_spin = QSpinBox()
        self._open_spin.setRange(0, 31)
        self._open_spin.setToolTip("开运算核大小（去毛刺噪声，0=不开启）")
        form.addRow("开运算核 (open):", self._open_spin)

        self._min_area_spin = QSpinBox()
        self._min_area_spin.setRange(0, 100000)
        self._min_area_spin.setSingleStep(50)
        self._min_area_spin.setToolTip("删除面积小于此值的连通域")
        form.addRow("最小面积 (min_area):", self._min_area_spin)

        self._fill_holes_cb = QCheckBox("填充孔洞（默认关闭）")
        form.addRow("", self._fill_holes_cb)

        self._use_roi_cb = QCheckBox("使用 ROI 约束")
        form.addRow("", self._use_roi_cb)

        self._use_ignore_cb = QCheckBox("使用 Ignore 屏蔽")
        form.addRow("", self._use_ignore_cb)

        params_group.setLayout(form)
        layout.addWidget(params_group)

        self._info_label = QLabel("就绪")
        self._info_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._info_label)

        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumHeight(140)
        self._log_view.setPlaceholderText("预览/应用日志…")
        layout.addWidget(self._log_view)

        btn_layout = QHBoxLayout()

        self._preview_btn = QPushButton("预览")
        self._preview_btn.clicked.connect(self._on_preview)
        btn_layout.addWidget(self._preview_btn)

        self._apply_btn = QPushButton("应用")
        self._apply_btn.setDefault(True)
        self._apply_btn.clicked.connect(self._on_apply)
        btn_layout.addWidget(self._apply_btn)

        self._save_btn = QPushButton("保存 processed_mask.png")
        self._save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(self._save_btn)

        self._save_defaults_btn = QPushButton("保存为默认参数")
        self._save_defaults_btn.setToolTip(
            "仅在此显式保存后，下次打开才会使用你的参数；"
            "否则始终回到比赛默认参数。不会被项目旧配置覆盖。"
        )
        self._save_defaults_btn.clicked.connect(self._on_save_defaults)
        btn_layout.addWidget(self._save_defaults_btn)

        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

        for w in [self._threshold_slider, self._blur_spin, self._close_spin,
                  self._open_spin, self._min_area_spin,
                  self._fill_holes_cb, self._use_roi_cb, self._use_ignore_cb]:
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self._on_params_changed)
            elif hasattr(w, "toggled"):
                w.toggled.connect(self._on_params_changed)

    def _apply_config_to_widgets(self, cfg: MaskPostprocessConfig) -> None:
        self._threshold_slider.blockSignals(True)
        self._threshold_spin.blockSignals(True)
        self._threshold_slider.setValue(int(cfg.threshold))
        self._threshold_spin.setValue(int(cfg.threshold))
        self._threshold_slider.blockSignals(False)
        self._threshold_spin.blockSignals(False)

        self._blur_spin.setValue(int(cfg.blur_kernel))
        self._close_spin.setValue(int(cfg.close_kernel))
        self._open_spin.setValue(int(cfg.open_kernel))
        self._min_area_spin.setValue(int(cfg.min_area))
        self._fill_holes_cb.setChecked(bool(cfg.fill_holes))
        self._use_roi_cb.setChecked(bool(cfg.use_roi))
        self._use_ignore_cb.setChecked(bool(cfg.use_ignore))

    def _get_config(self) -> MaskPostprocessConfig:
        return MaskPostprocessConfig(
            threshold=self._threshold_slider.value(),
            blur_kernel=self._blur_spin.value(),
            close_kernel=self._close_spin.value(),
            open_kernel=self._open_spin.value(),
            min_area=self._min_area_spin.value(),
            fill_holes=self._fill_holes_cb.isChecked(),
            keep_largest=0,
            use_roi=self._use_roi_cb.isChecked(),
            use_ignore=self._use_ignore_cb.isChecked(),
        )

    def _append_log(self, msg: str) -> None:
        self._log_lines.append(msg)
        self._log_view.append(msg)

    def _on_params_changed(self, _=None):
        # 参数改动后仅更新摘要，避免每次拖动都重跑；预览/应用时正式计算
        cfg = self._get_config()
        self._info_label.setText(
            f"待预览: threshold={cfg.threshold}, blur={cfg.blur_kernel}, "
            f"close={cfg.close_kernel}, open={cfg.open_kernel}, "
            f"min_area={cfg.min_area}, fill_holes={cfg.fill_holes}, "
            f"use_roi={cfg.use_roi}, use_ignore={cfg.use_ignore}"
        )

    def _update_preview(self):
        try:
            config = self._get_config()
            self._log_lines.clear()
            self._log_view.clear()
            self._append_log("—— 开始后处理 ——")
            self._append_log(f"threshold={config.threshold}")
            self._append_log(f"blur={config.blur_kernel}")
            self._append_log(f"close={config.close_kernel}")
            self._append_log(f"open={config.open_kernel}")
            self._append_log(f"min_area={config.min_area}")
            self._append_log(f"fill_holes={config.fill_holes}")
            self._append_log(f"use_roi={config.use_roi}")
            self._append_log(f"use_ignore={config.use_ignore}")

            self._processed, self._steps = process_mask_from_layer(
                self._mask_data, config,
                roi_data=self._roi_data if config.use_roi else None,
                ignore_data=self._ignore_data if config.use_ignore else None,
                log_fn=self._append_log,
            )
            road_px = int((self._processed > 0).sum())
            total = self._processed.size
            ratio = road_px / total * 100 if total else 0
            self._info_label.setText(
                f"处理后: {road_px} 道路像素 ({ratio:.2f}%), "
                f"{len(self._steps)} 步处理 | dtype={self._processed.dtype}"
            )
        except Exception as e:
            self._info_label.setText(f"错误: {e}")
            self._append_log(f"ERROR: {e}")
            raise

    def _on_preview(self):
        try:
            self._update_preview()
            config = self._get_config()
            info = (
                f"threshold={config.threshold}, blur={config.blur_kernel}, "
                f"close={config.close_kernel}, open={config.open_kernel}, "
                f"min_area={config.min_area}, fill_holes={config.fill_holes}, "
                f"use_roi={config.use_roi}, use_ignore={config.use_ignore}"
            )
            self.preview_requested.emit(self._processed, info)
        except Exception as e:
            QMessageBox.warning(self, "预览失败", str(e))

    def _on_apply(self):
        try:
            self._update_preview()
            self.apply_requested.emit(self._processed)
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "应用失败", str(e))

    def _on_save(self):
        try:
            self._update_preview()
            default_dir = self._output_dir or str(Path.cwd() / "outputs")
            path, _ = QFileDialog.getSaveFileName(
                self, "保存 processed mask",
                str(Path(default_dir) / "processed_mask.png"),
                "PNG (*.png);;All (*.*)",
            )
            if not path:
                return
            import cv2
            ok = cv2.imwrite(path, self._processed)
            if not ok:
                raise RuntimeError(f"cv2.imwrite 失败: {path}")
            QMessageBox.information(self, "已保存", f"已保存到:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))

    def _on_save_defaults(self):
        try:
            path = save_user_defaults(self._get_config())
            QMessageBox.information(
                self, "已保存为默认参数",
                f"已写入：\n{path}\n\n下次打开此窗口将使用这些参数。\n"
                f"未保存前，打开窗口始终回到比赛默认参数。",
            )
        except Exception as e:
            QMessageBox.warning(self, "保存默认参数失败", str(e))

    def get_result(self) -> np.ndarray:
        return self._processed
