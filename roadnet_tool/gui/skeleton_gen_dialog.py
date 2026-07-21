"""
Skeleton 生成对话框：从当前 mask 生成 clean_skeleton。

提供：
- 骨架化方法选择
- 短枝剪除长度
- 端点连接参数（可选，默认关闭）
- 生成后写入 skeleton 图层
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QPushButton, QLabel, QGroupBox, QMessageBox,
    QApplication,
)
from PySide6.QtCore import Qt, Signal
import numpy as np

from roadnet.skeleton_gen import SkeletonConfig
from roadnet.optimized_skeleton import skeletonize_medial_axis, skeletonize_thin


class SkeletonGenDialog(QDialog):
    """从 mask 生成 skeleton 对话框"""

    # 信号：生成完成 (skeleton, 统计信息)
    skeleton_generated = Signal(object, str)

    def __init__(
        self,
        mask_data: np.ndarray,
        parent=None,
    ):
        super().__init__(parent)
        self._mask_data = mask_data
        self._skeleton = None

        self.setWindowTitle("从 Mask 生成 Skeleton")
        self.setMinimumWidth(420)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── 方法选择 ──
        method_group = QGroupBox("骨架化方法")
        method_layout = QFormLayout()
        self._method_combo = QComboBox()
        self._method_combo.addItem("skeletonize (Zhang-Suen, 推荐)", "thin")
        self._method_combo.addItem("medial_axis (中轴变换)", "medial_axis")
        method_layout.addRow("方法:", self._method_combo)
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # ── 剪枝参数 ──
        prune_group = QGroupBox("原始骨架（生成阶段不剪枝）")
        prune_form = QFormLayout()
        self._prune_spin = QSpinBox()
        self._prune_spin.setRange(0, 500)
        self._prune_spin.setValue(0)
        self._prune_spin.setEnabled(False)
        self._prune_spin.setSingleStep(5)
        self._prune_spin.setToolTip("删除短于此长度（像素）的毛刺分支，0=不剪枝")
        prune_form.addRow("最小分支长度 (px):", self._prune_spin)
        self._border_spin = QSpinBox()
        self._border_spin.setRange(0, 50)
        self._border_spin.setValue(0)
        self._border_spin.setEnabled(False)
        self._border_spin.setToolTip("图像边界留白（像素）")
        prune_form.addRow("边界留白 (px):", self._border_spin)
        prune_group.setLayout(prune_form)
        layout.addWidget(prune_group)

        # ── 端点连接（可选） ──
        connect_group = QGroupBox("端点自动连接（可选，默认关闭）")
        connect_form = QFormLayout()
        self._connect_enable_cb = QCheckBox("启用自动连接")
        self._connect_enable_cb.setChecked(False)
        self._connect_enable_cb.setEnabled(False)
        connect_form.addRow("", self._connect_enable_cb)
        self._connect_dist_spin = QSpinBox()
        self._connect_dist_spin.setRange(0, 200)
        self._connect_dist_spin.setValue(10)
        self._connect_dist_spin.setToolTip("两个端点距离小于此值才连接")
        connect_form.addRow("最大连接距离 (px):", self._connect_dist_spin)
        self._connect_angle_spin = QDoubleSpinBox()
        self._connect_angle_spin.setRange(0, 90)
        self._connect_angle_spin.setValue(30.0)
        self._connect_angle_spin.setToolTip("两个端点方向夹角小于此值才连接（度）")
        connect_form.addRow("最大方向夹角 (°):", self._connect_angle_spin)
        self._connect_overlap_spin = QDoubleSpinBox()
        self._connect_overlap_spin.setRange(0.0, 1.0)
        self._connect_overlap_spin.setValue(0.65)
        self._connect_overlap_spin.setSingleStep(0.05)
        self._connect_overlap_spin.setToolTip("连线经过 mask 的最小比例")
        connect_form.addRow("Mask 重叠率:", self._connect_overlap_spin)
        connect_group.setLayout(connect_form)
        layout.addWidget(connect_group)

        # ── 信息 ──
        self._info_label = QLabel('就绪，点击「生成」开始')
        self._info_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._info_label)

        # ── 按钮 ──
        btn_layout = QHBoxLayout()
        self._generate_btn = QPushButton("🦴 生成 Skeleton")
        self._generate_btn.setDefault(True)
        self._generate_btn.clicked.connect(self._on_generate)
        btn_layout.addWidget(self._generate_btn)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _get_config(self) -> SkeletonConfig:
        if self._connect_enable_cb.isChecked():
            return SkeletonConfig(
                method=self._method_combo.currentData(),
                prune_length=self._prune_spin.value(),
                border_margin=self._border_spin.value(),
                connect_endpoint_distance=self._connect_dist_spin.value(),
                connect_angle_threshold=self._connect_angle_spin.value(),
                connect_mask_overlap=self._connect_overlap_spin.value(),
            )
        else:
            return SkeletonConfig(
                method=self._method_combo.currentData(),
                prune_length=self._prune_spin.value(),
                border_margin=self._border_spin.value(),
                connect_endpoint_distance=0,
            )

    def _on_generate(self):
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            config = self._get_config()
            if config.method == "medial_axis":
                self._skeleton = skeletonize_medial_axis(self._mask_data)
            else:
                self._skeleton = skeletonize_thin(self._mask_data)
        except Exception as e:
            QMessageBox.critical(self, "生成失败", str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()

        skel_px = int((self._skeleton > 0).sum())
        info = (
            f"Skeleton 生成完成: {skel_px} 像素, "
            f"方法={config.method}, 状态=raw（未剪枝）"
        )
        if config.connect_endpoint_distance > 0:
            info += f", 端点连接≤{config.connect_endpoint_distance}px"
        self._info_label.setText(info)
        self.skeleton_generated.emit(self._skeleton, info)
        self.accept()

    def get_result(self) -> np.ndarray | None:
        return self._skeleton
