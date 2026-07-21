"""
Skeleton → Graph 对话框：从 clean_skeleton 生成 graph。

提供：
- 节点合并距离
- 最小边长度
- 折线简化容差
- 生成后加载到 final_graph
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QLabel, QGroupBox, QMessageBox,
    QApplication, QFileDialog,
)
from PySide6.QtCore import Qt, Signal
import numpy as np
from pathlib import Path

from roadnet.skeleton_to_graph import (
    SkeletonToGraphConfig, skeleton_to_graph,
    save_graph_from_skeleton,
)


class SkeletonToGraphDialog(QDialog):
    """Skeleton → Graph 转换对话框"""

    # 信号：graph 已生成 (nodes, edges, 统计信息)
    graph_generated = Signal(list, list, str)

    def __init__(
        self,
        skeleton_data: np.ndarray,
        output_dir: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._skeleton = skeleton_data
        self._output_dir = output_dir
        self._nodes = []
        self._edges = []

        self.setWindowTitle("从 Skeleton 生成 Graph")
        self.setMinimumWidth(400)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── 参数 ──
        params_group = QGroupBox("转换参数")
        form = QFormLayout()

        def int_spin(value, low=0, high=100):
            spin = QSpinBox()
            spin.setRange(low, high)
            spin.setValue(value)
            return spin

        self._junction_spin = int_spin(10)
        form.addRow("路口聚类半径 (px):", self._junction_spin)
        self._endpoint_merge_spin = int_spin(12)
        form.addRow("端点合并距离 (px):", self._endpoint_merge_spin)
        self._endpoint_connect_spin = int_spin(25, 0, 200)
        self._endpoint_connect_spin.setToolTip("常用值：20 / 25 / 35")
        form.addRow("端点连接距离 (px):", self._endpoint_connect_spin)
        self._node_merge_spin = int_spin(8)
        form.addRow("节点合并距离 (px):", self._node_merge_spin)

        self._min_edge_spin = QDoubleSpinBox()
        self._min_edge_spin.setRange(0, 500)
        self._min_edge_spin.setValue(8.0)
        self._min_edge_spin.setToolTip("短于此长度的边被删除（像素）")
        form.addRow("最小边长度 (px):", self._min_edge_spin)

        self._simplify_spin = QDoubleSpinBox()
        self._simplify_spin.setRange(0, 20)
        self._simplify_spin.setValue(2.0)
        self._simplify_spin.setSingleStep(0.5)
        self._simplify_spin.setToolTip("Douglas-Peucker 折线简化容差，越大越简")
        form.addRow("折线简化容差 (px):", self._simplify_spin)

        self._prune_spin = QDoubleSpinBox()
        self._prune_spin.setRange(0, 500)
        self._prune_spin.setValue(15.0)
        form.addRow("死端修剪长度 (px):", self._prune_spin)

        self._short_filter_check = QCheckBox("启用短边过滤（取消后保留全部短边）")
        self._short_filter_check.setChecked(True)
        form.addRow(self._short_filter_check)
        self._optimizer_check = QCheckBox("生成 raw 后执行 graph_line_optimizer")
        self._optimizer_check.setChecked(False)
        form.addRow(self._optimizer_check)
        self._raw_debug_check = QCheckBox("Raw graph 调试模式（不优化、不规划、不转换坐标）")
        self._raw_debug_check.setChecked(False)
        self._raw_debug_check.toggled.connect(
            lambda checked: self._optimizer_check.setEnabled(not checked)
        )
        form.addRow(self._raw_debug_check)

        params_group.setLayout(form)
        layout.addWidget(params_group)

        # ── 信息 ──
        self._info_label = QLabel('就绪，点击「生成」从 skeleton 提取 graph')
        self._info_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._info_label)

        # ── 统计 ──
        statue_str = ""
        if self._skeleton is not None:
            skel_px = int((self._skeleton > 0).sum())
            statue_str = f"当前 skeleton: {skel_px} 像素"
        self._stat_label = QLabel(statue_str)
        self._stat_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self._stat_label)

        # ── 按钮 ──
        btn_layout = QHBoxLayout()

        self._generate_btn = QPushButton("📊 生成 Graph")
        self._generate_btn.setDefault(True)
        self._generate_btn.clicked.connect(self._on_generate)
        btn_layout.addWidget(self._generate_btn)

        self._save_btn = QPushButton("💾 保存 graph_from_skeleton.json")
        self._save_btn.clicked.connect(self._on_save)
        self._save_btn.setEnabled(False)
        btn_layout.addWidget(self._save_btn)

        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

    def _get_config(self) -> SkeletonToGraphConfig:
        return SkeletonToGraphConfig(
            junction_cluster_radius=self._junction_spin.value(),
            endpoint_merge_distance=self._endpoint_merge_spin.value(),
            endpoint_connect_distance=self._endpoint_connect_spin.value(),
            node_merge_distance=self._node_merge_spin.value(),
            min_edge_length=self._min_edge_spin.value(),
            prune_length=self._prune_spin.value(),
            rdp_epsilon=self._simplify_spin.value(),
            enable_short_edge_filter=self._short_filter_check.isChecked(),
            enable_prune=self._prune_spin.value() > 0,
        )

    def get_build_options(self):
        cfg = self._get_config()
        return {
            "graph": {
                "junction_cluster_radius": cfg.junction_cluster_radius,
                "endpoint_merge_distance": cfg.endpoint_merge_distance,
                "endpoint_connect_distance": cfg.endpoint_connect_distance,
                "node_merge_distance": cfg.node_merge_distance,
                "min_edge_length": cfg.min_edge_length,
                "prune_length": cfg.prune_length,
                "rdp_epsilon": cfg.rdp_epsilon,
                "enable_short_edge_filter": cfg.enable_short_edge_filter,
                "enable_prune": cfg.enable_prune,
                "enable_graph_line_optimizer": self._optimizer_check.isChecked(),
            }
        }, self._raw_debug_check.isChecked(), self._optimizer_check.isChecked()

    def _on_generate(self):
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            config = self._get_config()
            self._nodes, self._edges = skeleton_to_graph(self._skeleton, config)
        except Exception as e:
            QMessageBox.critical(self, "生成失败", str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()

        endpoint_count = sum(1 for n in self._nodes if n["type"] == "endpoint")
        junction_count = sum(1 for n in self._nodes if n["type"] == "junction")
        info = (
            f"Graph 生成完成: {len(self._nodes)} 节点 "
            f"(端点={endpoint_count}, 交叉口={junction_count}), "
            f"{len(self._edges)} 条边"
        )
        self._info_label.setText(info)
        self._save_btn.setEnabled(True)

        self.graph_generated.emit(self._nodes, self._edges, info)
        self.accept()

    def _on_save(self):
        if not self._nodes:
            QMessageBox.warning(self, "提示", "请先生成 Graph。")
            return
        try:
            default_dir = self._output_dir or str(Path.cwd() / "outputs")
            path, _ = QFileDialog.getSaveFileName(
                self, "保存 Graph JSON", default_dir,
                "JSON (*.json);;All (*.*)",
            )
            if not path:
                return
            import os
            output_dir = os.path.dirname(path)
            save_graph_from_skeleton(self._nodes, self._edges, output_dir)
            QMessageBox.information(self, "已保存", f"已保存到:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))

    def get_result(self):
        return self._nodes, self._edges
