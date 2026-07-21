"""
Graph 线形优化对话框。

提供参数调整 UI：
- RDP 简化容差
- 直线判定最大偏离
- 最小拉直长度
- 平滑窗口大小
- 最大平滑偏移
- Mask 容差
- 是否使用 mask 校验
- 是否保存对比图

点击「优化」后触发优化，结果通过信号传递。
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QDoubleSpinBox, QSpinBox, QCheckBox,
    QPushButton, QLabel, QGroupBox, QMessageBox,
    QApplication,
)
from PySide6.QtCore import Qt, Signal
import numpy as np

from roadnet.graph_line_optimizer import (
    GraphLineOptimizeConfig, optimize_graph_lines,
    save_optimization_results, DEFAULT_CONFIG,
)


class GraphLineOptimizeDialog(QDialog):
    """Graph 线形优化参数对话框。"""

    # 信号：优化完成 (optimized_edges, report_dict)
    optimization_finished = Signal(list, dict)

    def __init__(
        self,
        edges: list,
        processed_mask: np.ndarray | None = None,
        image_rgb: np.ndarray | None = None,
        output_dir: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._edges = list(edges)  # 浅拷贝，只读引用
        self._processed_mask = processed_mask
        self._image_rgb = image_rgb
        self._output_dir = output_dir

        self._optimized_edges: list = []
        self._report: dict = {}

        self.setWindowTitle("优化 Graph 线形")
        self.setMinimumWidth(480)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── 参数组 ──
        params_group = QGroupBox("线形优化参数")
        form = QFormLayout()

        # RDP 简化容差
        self._rdp_spin = QDoubleSpinBox()
        self._rdp_spin.setRange(0.0, 20.0)
        self._rdp_spin.setValue(DEFAULT_CONFIG.rdp_epsilon)
        self._rdp_spin.setSingleStep(0.5)
        self._rdp_spin.setToolTip(
            "Douglas-Peucker 折线简化容差。\n"
            "越大简化越激进，越小保留越多细节。\n"
            "建议值: 1.5~5.0"
        )
        form.addRow("RDP 简化容差 (px):", self._rdp_spin)

        # 直线判定最大偏离
        self._straight_dev_spin = QDoubleSpinBox()
        self._straight_dev_spin.setRange(0.0, 50.0)
        self._straight_dev_spin.setValue(DEFAULT_CONFIG.straight_max_deviation)
        self._straight_dev_spin.setSingleStep(0.5)
        self._straight_dev_spin.setToolTip(
            "判断一条道路是否接近直线的最大允许偏移距离。\n"
            "中间所有点到首尾连线的垂直距离 <= 此值 → 拉直。\n"
            "建议值: 3.0~8.0"
        )
        form.addRow("直线判定最大偏离 (px):", self._straight_dev_spin)

        # 最小拉直长度
        self._min_straight_spin = QDoubleSpinBox()
        self._min_straight_spin.setRange(5.0, 500.0)
        self._min_straight_spin.setValue(DEFAULT_CONFIG.min_straight_edge_length)
        self._min_straight_spin.setSingleStep(5.0)
        self._min_straight_spin.setToolTip(
            "长度大于此值的 edge 才参与拉直判定。\n"
            "太短的 edge 不拉直，避免误判。"
        )
        form.addRow("最小拉直长度 (px):", self._min_straight_spin)

        # 平滑窗口
        self._smooth_window_spin = QSpinBox()
        self._smooth_window_spin.setRange(1, 21)
        self._smooth_window_spin.setValue(DEFAULT_CONFIG.smooth_window)
        self._smooth_window_spin.setSingleStep(2)
        self._smooth_window_spin.setToolTip(
            "弯路平滑的 moving average 窗口大小。\n"
            "越大越平滑，但细节损失越多。\n"
            "建议值: 3~7（奇数）"
        )
        form.addRow("平滑窗口:", self._smooth_window_spin)

        # 最大平滑偏移
        self._max_smooth_spin = QDoubleSpinBox()
        self._max_smooth_spin.setRange(0.0, 30.0)
        self._max_smooth_spin.setValue(DEFAULT_CONFIG.max_smooth_offset)
        self._max_smooth_spin.setSingleStep(0.5)
        self._max_smooth_spin.setToolTip(
            "平滑后每个点相对原始位置的最大偏移距离。\n"
            "防止过度平滑导致线偏移。\n"
            "建议值: 2.0~6.0"
        )
        form.addRow("最大平滑偏移 (px):", self._max_smooth_spin)

        # Mask 容差
        self._mask_tol_spin = QDoubleSpinBox()
        self._mask_tol_spin.setRange(0.0, 50.0)
        self._mask_tol_spin.setValue(DEFAULT_CONFIG.mask_tolerance)
        self._mask_tol_spin.setSingleStep(1.0)
        self._mask_tol_spin.setToolTip(
            "Mask 校验距离容差。\n"
            "优化后点不在 mask 内，但距最近道路区域 <= 此值 → 通过。\n"
            "建议值: 3.0~8.0"
        )
        form.addRow("Mask 容差 (px):", self._mask_tol_spin)

        params_group.setLayout(form)
        layout.addWidget(params_group)

        # ── 选项组 ──
        options_group = QGroupBox("选项")
        opts_layout = QVBoxLayout()

        self._use_mask_check = QCheckBox("使用 processed_mask 校验（推荐）")
        self._use_mask_check.setChecked(DEFAULT_CONFIG.validate_with_mask)
        self._use_mask_check.setToolTip(
            "启用后，优化后的 polyline 会与 mask 比对，\n"
            "偏离道路区域的 edge 将回退原始 polyline。"
        )
        opts_layout.addWidget(self._use_mask_check)

        self._save_preview_check = QCheckBox("保存优化前后对比图")
        self._save_preview_check.setChecked(True)
        opts_layout.addWidget(self._save_preview_check)

        if self._processed_mask is None:
            self._use_mask_check.setChecked(False)
            self._use_mask_check.setEnabled(False)
            self._use_mask_check.setText("使用 processed_mask 校验（无可用 mask）")

        options_group.setLayout(opts_layout)
        layout.addWidget(options_group)

        # ── 信息 ──
        edge_count = len([e for e in self._edges if e.get("enabled", True)])
        self._info_label = QLabel(
            f"就绪 · 当前 final_graph: {edge_count} 条 enabled edge\n"
            f"点击「优化」开始线形优化"
        )
        self._info_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._info_label)

        # ── 按钮 ──
        btn_layout = QHBoxLayout()

        self._optimize_btn = QPushButton("📐 优化线形")
        self._optimize_btn.setDefault(True)
        self._optimize_btn.clicked.connect(self._on_optimize)
        btn_layout.addWidget(self._optimize_btn)

        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

    def _build_config(self) -> GraphLineOptimizeConfig:
        return GraphLineOptimizeConfig(
            rdp_epsilon=self._rdp_spin.value(),
            straight_max_deviation=self._straight_dev_spin.value(),
            min_straight_edge_length=self._min_straight_spin.value(),
            smooth_window=self._smooth_window_spin.value(),
            max_smooth_offset=self._max_smooth_spin.value(),
            mask_tolerance=self._mask_tol_spin.value(),
            preserve_junctions=True,
            validate_with_mask=self._use_mask_check.isChecked(),
        )

    def _on_optimize(self):
        """执行优化并发射信号。"""
        enabled_edges = [e for e in self._edges if e.get("enabled", True)]
        if not enabled_edges:
            QMessageBox.warning(self, "提示", "当前 final_graph 中没有 enabled edge。")
            return

        config = self._build_config()
        use_mask = config.validate_with_mask and self._processed_mask is not None

        self._info_label.setText("正在优化 line geometry...")
        self._info_label.setStyleSheet("color: #FFA; font-size: 11px;")
        QApplication.processEvents()

        try:
            self._optimized_edges, self._report = optimize_graph_lines(
                self._edges,
                processed_mask=self._processed_mask if use_mask else None,
                config=config,
            )
        except Exception as e:
            QMessageBox.critical(self, "优化失败", f"线形优化出错:\n{e}")
            self._info_label.setText("优化失败")
            self._info_label.setStyleSheet("color: #F55; font-size: 11px;")
            return

        # ── 保存结果 ──
        if self._output_dir:
            save_dir = self._output_dir
            try:
                save_optimization_results(
                    edges_before=self._edges,
                    edges_after=self._optimized_edges,
                    report=self._report,
                    output_dir=save_dir,
                    image_rgb=self._image_rgb if self._save_preview_check.isChecked() else None,
                )
            except Exception as e:
                print(f"[GraphLineOpt] 保存结果时出错: {e}")

        # ── 构建用户可见摘要 ──
        s = self._report["summary"]
        summary_lines = []
        summary_lines.append("─────────── 线形优化完成 ───────────")
        summary_lines.append(f"  总 edge 数:        {s['total_edges']}")
        summary_lines.append(f"  成功优化:          {s['successful_edges']}")
        summary_lines.append(f"    ├ 拉直 (直线):    {s['straightened_edges']}")
        summary_lines.append(f"    └ 平滑 (弯路):    {s['smoothed_edges']}")
        summary_lines.append(f"  Mask 回退:         {s['mask_rollback_edges']}")
        summary_lines.append(f"  太短跳过:          {s['skipped_short_edges']}")
        summary_lines.append("")
        summary_lines.append(f"  优化前总点数:      {s['total_points_before']}")
        summary_lines.append(f"  优化后总点数:      {s['total_points_after']}")
        summary_lines.append(f"  点数减少:          {s['points_reduction_pct']}%")
        summary_lines.append(f"  最大偏离记录:      {s['max_deviation_recorded']} px")
        summary_lines.append("")
        summary_lines.append("  ❗ 交叉口节点和端点未移动")
        summary_lines.append("  ❗ graph 拓扑结构未改变")
        summary_lines.append("  ❗ 仅修改 edge.polyline 中间点")
        summary_lines.append("")
        summary_lines.append("  结果已保存至:")
        summary_lines.append(f"    {self._output_dir}")
        summary_text = "\n".join(summary_lines)

        # 更新 UI
        self._info_label.setText(f"优化完成 · {s['successful_edges']}/{s['total_edges']} edges 成功")
        self._info_label.setStyleSheet("color: #5F5; font-size: 11px;")

        # 弹窗摘要
        QMessageBox.information(self, "线形优化完成", summary_text)

        # 发射信号
        self.optimization_finished.emit(self._optimized_edges, self._report)
        self.accept()
