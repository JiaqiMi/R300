"""
骨架生成/优化对话框：从当前 road mask 生成优化后的骨架。

提供：
- 骨架化方法选择（medial_axis / skeletonize / thin）
- 边界过滤开关 + 边距
- 距离变换过滤开关 + min_center_dist（支持 auto）
- 短毛刺删除开关 + prune_length
- Junction 检测/聚类开关 + cluster_radius
- 端点连接开关 + 连接距离/夹角/重叠率
- 输出选项（overlay / stats JSON）
- 运行后自动写入 skeleton 图层
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QPushButton, QLabel, QGroupBox, QMessageBox,
    QApplication, QPlainTextEdit,
)
from PySide6.QtCore import Qt, Signal
import numpy as np

from roadnet.skeleton_optimizer_adapter import (
    SkeletonOptimizeConfig,
    SkeletonOptimizeResult,
    run_skeleton_optimization,
)


class SkeletonOptimizeDialog(QDialog):
    """骨架生成/优化参数对话框。"""

    # 信号：(optimized_skeleton, result, info_text)
    skeleton_optimized = Signal(object, object, str)

    def __init__(
        self,
        mask_data: np.ndarray,
        image_rgb: np.ndarray | None = None,
        output_base_dir: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._mask_data = mask_data
        self._image_rgb = image_rgb
        self._output_base_dir = output_base_dir
        self._result: SkeletonOptimizeResult | None = None

        self.setWindowTitle("生成/优化道路骨架")
        self.setMinimumWidth(460)
        self._build_ui()
        self._connect_signals()

    # ===================================================================
    # UI
    # ===================================================================

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── 骨架化方法 ──
        method_group = QGroupBox("骨架化方法")
        method_form = QFormLayout()
        self._method_combo = QComboBox()
        self._method_combo.addItem("medial_axis（中轴变换，推荐）", "medial_axis")
        self._method_combo.addItem("skeletonize（Zhang-Suen 细化）", "skeletonize")
        self._method_combo.addItem("thin（薄化）", "thin")
        method_form.addRow("方法:", self._method_combo)
        method_group.setLayout(method_form)
        layout.addWidget(method_group)

        # ── 边界过滤 ──
        border_group = QGroupBox("边界过滤")
        border_form = QFormLayout()
        self._border_enable_cb = QCheckBox("启用")
        self._border_enable_cb.setChecked(True)
        border_form.addRow("", self._border_enable_cb)
        self._border_spin = QSpinBox()
        self._border_spin.setRange(0, 100)
        self._border_spin.setValue(10)
        self._border_spin.setToolTip("删除靠近图像边界此距离内的骨架点（像素）")
        border_form.addRow("边界留白 (px):", self._border_spin)
        border_group.setLayout(border_form)
        layout.addWidget(border_group)

        # ── 距离变换过滤 ──
        dist_group = QGroupBox("距离变换过滤（中心线质量）")
        dist_form = QFormLayout()
        self._dist_enable_cb = QCheckBox("启用")
        self._dist_enable_cb.setChecked(True)
        dist_form.addRow("", self._dist_enable_cb)
        h_dist = QHBoxLayout()
        self._center_dist_spin = QDoubleSpinBox()
        self._center_dist_spin.setRange(0.5, 50.0)
        self._center_dist_spin.setValue(2.0)
        self._center_dist_spin.setSingleStep(0.5)
        self._center_dist_spin.setToolTip("最小道路中心距离。auto 模式会自动适应细线道路。")
        h_dist.addWidget(self._center_dist_spin)
        self._center_auto_cb = QCheckBox("auto（自适应）")
        self._center_auto_cb.setChecked(True)
        self._center_auto_cb.setToolTip("根据 mask 道路宽度自动调整阈值，避免删除细道路")
        h_dist.addWidget(self._center_auto_cb)
        dist_form.addRow("min_center_dist:", h_dist)
        dist_group.setLayout(dist_form)
        layout.addWidget(dist_group)

        # ── 短毛刺删除 ──
        spur_group = QGroupBox("短毛刺删除")
        spur_form = QFormLayout()
        self._spur_enable_cb = QCheckBox("启用")
        self._spur_enable_cb.setChecked(True)
        spur_form.addRow("", self._spur_enable_cb)
        self._prune_spin = QSpinBox()
        self._prune_spin.setRange(0, 500)
        self._prune_spin.setValue(20)
        self._prune_spin.setSingleStep(5)
        self._prune_spin.setToolTip("短于此像素数的分支将被删除（0=不剪枝）")
        spur_form.addRow("最小分支长度 (px):", self._prune_spin)
        spur_group.setLayout(spur_form)
        layout.addWidget(spur_group)

        # ── Junction 检测与聚类 ──
        junc_group = QGroupBox("Junction 检测与聚类")
        junc_form = QFormLayout()
        self._junc_enable_cb = QCheckBox("启用")
        self._junc_enable_cb.setChecked(True)
        junc_form.addRow("", self._junc_enable_cb)
        self._junc_radius_spin = QSpinBox()
        self._junc_radius_spin.setRange(0, 100)
        self._junc_radius_spin.setValue(10)
        self._junc_radius_spin.setToolTip("Junction 像素聚类半径（像素）")
        junc_form.addRow("聚类半径 (px):", self._junc_radius_spin)
        junc_group.setLayout(junc_form)
        layout.addWidget(junc_group)

        # ── 端点自动连接 ──
        connect_group = QGroupBox("端点自动连接（可选，默认关闭）")
        connect_form = QFormLayout()
        self._connect_enable_cb = QCheckBox("启用自动连接")
        self._connect_enable_cb.setChecked(False)
        connect_form.addRow("", self._connect_enable_cb)
        self._connect_dist_spin = QDoubleSpinBox()
        self._connect_dist_spin.setRange(0, 200.0)
        self._connect_dist_spin.setValue(25.0)
        self._connect_dist_spin.setToolTip("两个端点距离小于此值才尝试连接")
        connect_form.addRow("最大连接距离 (px):", self._connect_dist_spin)
        self._connect_angle_spin = QDoubleSpinBox()
        self._connect_angle_spin.setRange(0.0, 90.0)
        self._connect_angle_spin.setValue(45.0)
        self._connect_angle_spin.setToolTip("两个端点方向夹角小于此值才连接（度）")
        connect_form.addRow("最大方向夹角 (°):", self._connect_angle_spin)
        self._connect_overlap_spin = QDoubleSpinBox()
        self._connect_overlap_spin.setRange(0.0, 1.0)
        self._connect_overlap_spin.setValue(0.65)
        self._connect_overlap_spin.setSingleStep(0.05)
        self._connect_overlap_spin.setToolTip("连线经过 road mask 的最小比例")
        connect_form.addRow("Mask 重叠率:", self._connect_overlap_spin)
        connect_group.setLayout(connect_form)
        layout.addWidget(connect_group)

        # ── 输出选项 ──
        output_group = QGroupBox("输出选项")
        output_form = QFormLayout()
        self._overlay_cb = QCheckBox("生成 skeleton_overlay.png")
        self._overlay_cb.setChecked(True)
        output_form.addRow("", self._overlay_cb)
        self._stats_cb = QCheckBox("保存 skeleton_stats.json")
        self._stats_cb.setChecked(True)
        output_form.addRow("", self._stats_cb)
        output_group.setLayout(output_form)
        layout.addWidget(output_group)

        # ── 日志 ──
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(500)
        self._log_view.setMinimumHeight(100)
        self._log_view.setStyleSheet(
            "QPlainTextEdit { font-family: Consolas, monospace; font-size: 10px; "
            "background-color: #1e1e1e; color: #d4d4d4; }"
        )
        log_layout.addWidget(self._log_view)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # ── 状态 ──
        self._status_label = QLabel("就绪 — 点击「生成/优化骨架」开始")
        self._status_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._status_label)

        # ── 按钮 ──
        btn_layout = QHBoxLayout()
        self._run_btn = QPushButton("🦴 生成/优化骨架")
        self._run_btn.setDefault(True)
        self._run_btn.setToolTip("执行 mask 标准化 → 骨架生成 → 骨架优化 → 保存输出")
        btn_layout.addWidget(self._run_btn)
        cancel_btn = QPushButton("取消")
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _connect_signals(self):
        self._run_btn.clicked.connect(self._on_run)
        # 关联控件状态
        self._border_enable_cb.toggled.connect(self._border_spin.setEnabled)
        self._dist_enable_cb.toggled.connect(self._center_dist_spin.setEnabled)
        self._dist_enable_cb.toggled.connect(self._center_auto_cb.setEnabled)
        self._spur_enable_cb.toggled.connect(self._prune_spin.setEnabled)
        self._junc_enable_cb.toggled.connect(self._junc_radius_spin.setEnabled)
        self._connect_enable_cb.toggled.connect(self._connect_dist_spin.setEnabled)
        self._connect_enable_cb.toggled.connect(self._connect_angle_spin.setEnabled)
        self._connect_enable_cb.toggled.connect(self._connect_overlap_spin.setEnabled)

    # ===================================================================
    # 逻辑
    # ===================================================================

    def _get_config(self) -> SkeletonOptimizeConfig:
        return SkeletonOptimizeConfig(
            skeleton_method=self._method_combo.currentData(),
            enable_border_filter=self._border_enable_cb.isChecked(),
            border_margin=self._border_spin.value(),
            enable_distance_filter=self._dist_enable_cb.isChecked(),
            min_center_dist=self._center_dist_spin.value(),
            enable_spur_removal=self._spur_enable_cb.isChecked(),
            prune_length=self._prune_spin.value(),
            enable_junction_cluster=self._junc_enable_cb.isChecked(),
            junction_cluster_radius=self._junc_radius_spin.value(),
            enable_endpoint_connect=self._connect_enable_cb.isChecked(),
            endpoint_connect_distance=self._connect_dist_spin.value(),
            endpoint_connect_angle=self._connect_angle_spin.value(),
            endpoint_connect_overlap=self._connect_overlap_spin.value(),
            output_overlay=self._overlay_cb.isChecked(),
            save_stats_json=self._stats_cb.isChecked(),
        )

    def _append_log(self, text: str):
        self._log_view.appendPlainText(text)
        # 自动滚动到底部
        bar = self._log_view.verticalScrollBar()
        if bar:
            bar.setValue(bar.maximum())

    def _on_run(self):
        config = self._get_config()

        # 验证
        if self._mask_data is None or self._mask_data.size == 0:
            QMessageBox.warning(self, "提示", "Mask 数据为空，请先加载 road mask。")
            return

        # 禁用按钮防重入
        self._run_btn.setEnabled(False)
        self._run_btn.setText("运行中...")
        self._status_label.setText("正在执行骨架生成/优化...")
        QApplication.processEvents()

        self._append_log("=" * 50)
        self._append_log(f"骨架生成/优化开始")
        self._append_log(f"  方法: {config.skeleton_method}")
        self._append_log(f"  min_center_dist: {config.min_center_dist}"
                         f" {'(auto)' if self._center_auto_cb.isChecked() else ''}")
        self._append_log(f"  prune_length: {config.prune_length}")
        self._append_log(f"  border_margin: {config.border_margin}")
        self._append_log(f"  端点连接: {'启用' if config.enable_endpoint_connect else '关闭'}")
        if config.enable_endpoint_connect:
            self._append_log(f"    最大距离={config.endpoint_connect_distance}px, "
                             f"夹角<={config.endpoint_connect_angle}°, "
                             f"重叠率>={config.endpoint_connect_overlap}")
        self._append_log(f"  junction 聚类: {'启用' if config.enable_junction_cluster else '关闭'} "
                         f"(radius={config.junction_cluster_radius})")
        self._append_log(f"  输出目录: {self._output_base_dir}/skeleton_outputs/")

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._result = run_skeleton_optimization(
                mask=self._mask_data,
                config=config,
                output_base_dir=self._output_base_dir,
                image_rgb=self._image_rgb,
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self._append_log(f"\n[ERROR] {tb}")
            QMessageBox.critical(self, "骨架优化失败", str(e))
            self._result = None
        finally:
            QApplication.restoreOverrideCursor()

        if self._result is None or not self._result.success:
            self._append_log(f"\n[FAILED] {self._result.error if self._result else '未知错误'}")
            self._status_label.setText(f"失败: {self._result.error if self._result else '未知错误'}")
            self._run_btn.setEnabled(True)
            self._run_btn.setText("🦴 重试")
            return

        r = self._result
        stats = r.stats
        self._append_log(f"\n[OK] 优化完成 (耗时 {r.elapsed_seconds:.1f}s)")
        self._append_log(f"  原始骨架像素: {stats.get('raw_pixels', '?')}")
        self._append_log(f"  优化后像素:   {stats.get('optimized_pixels', '?')}")
        self._append_log(f"  原始端点:     {stats.get('raw_endpoints', '?')}")
        self._append_log(f"  优化后端点:   {stats.get('optimized_endpoints', '?')}")
        self._append_log(f"  删除毛刺:     {stats.get('removed_spur_count', '?')}px")
        self._append_log(f"  连接断点:     {stats.get('connected_gap_count', '?')}px")
        self._append_log(f"  有效中心距离: {stats.get('effective_min_center_dist', '?')}px")
        self._append_log(f"  Junction 聚类: {stats.get('junction_cluster_count', '?')} 个")
        self._append_log(f"\n保存文件:")
        for name, path in r.saved_files.items():
            self._append_log(f"  {name}: {path}")

        # 构建信息文本
        info_parts = [
            f"骨架优化完成: {stats.get('optimized_pixels', 0)}px",
            f"方法={config.skeleton_method}",
            f"端点={stats.get('optimized_endpoints', '?')}",
            f"路口={stats.get('junction_cluster_count', '?')}",
        ]
        if stats.get('connected_gap_count', 0) > 0:
            info_parts.append(f"连接断点+{stats['connected_gap_count']}px")
        info = ", ".join(info_parts)

        self._status_label.setText(info)
        self._run_btn.setText("✓ 已完成")
        self.skeleton_optimized.emit(
            r.optimized_skeleton, r, info,
        )
        self.accept()

    def get_result(self) -> SkeletonOptimizeResult | None:
        return self._result
