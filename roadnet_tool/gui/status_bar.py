"""
底部状态栏：显示鼠标坐标、缩放比例、当前工具、分辨率、经纬度等。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QStatusBar, QLabel, QWidget, QHBoxLayout


class RoadNetStatusBar(QStatusBar):
    """自定义状态栏"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # 样式
        self.setStyleSheet("""
            QStatusBar {
                background-color: #181825;
                color: #a6adc8;
                border-top: 1px solid #313244;
                font-size: 12px;
            }
        """)

        # ★ 阶段
        self._stage_label = QLabel("① 导入影像")
        self.addPermanentWidget(self._stage_label)

        sep0 = QLabel("│")
        sep0.setStyleSheet("color: #313244; padding: 0 4px;")
        self.addPermanentWidget(sep0)

        # 鼠标坐标 (像素)
        self._coord_label = QLabel("  📍 (0, 0)")
        self._coord_label.setMinimumWidth(140)
        self.addPermanentWidget(self._coord_label)

        # ★ 经纬度（校准后显示）
        self._geo_label = QLabel("")
        self._geo_label.setMinimumWidth(180)
        self._geo_label.setVisible(False)  # 默认隐藏
        self.addPermanentWidget(self._geo_label)

        sep1 = QLabel("│")
        sep1.setStyleSheet("color: #313244; padding: 0 4px;")
        self.addPermanentWidget(sep1)

        # 缩放
        self._zoom_label = QLabel("🔍 100%")
        self.addPermanentWidget(self._zoom_label)

        sep_res = QLabel("│")
        sep_res.setStyleSheet("color: #313244; padding: 0 4px;")
        self.addPermanentWidget(sep_res)

        # ★ 分辨率
        self._resolution_label = QLabel("📏 0.50 m/px")
        self.addPermanentWidget(self._resolution_label)

        sep2 = QLabel("│")
        sep2.setStyleSheet("color: #313244; padding: 0 4px;")
        self.addPermanentWidget(sep2)

        # 当前工具
        self._tool_label = QLabel("🖐 平移")
        self.addPermanentWidget(self._tool_label)

        sep3 = QLabel("│")
        sep3.setStyleSheet("color: #313244; padding: 0 4px;")
        self.addPermanentWidget(sep3)

        # 道路像素占比
        self._road_ratio_label = QLabel("🛣 --")
        self.addPermanentWidget(self._road_ratio_label)

        sep4 = QLabel("│")
        sep4.setStyleSheet("color: #313244; padding: 0 4px;")
        self.addPermanentWidget(sep4)

        # 节点数
        self._nodes_label = QLabel("🔵 节点: --")
        self.addPermanentWidget(self._nodes_label)

        sep5 = QLabel("│")
        sep5.setStyleSheet("color: #313244; padding: 0 4px;")
        self.addPermanentWidget(sep5)

        # 边数
        self._edges_label = QLabel("🔗 边: --")
        self.addPermanentWidget(self._edges_label)

        # 临时消息（左侧）
        self._message_label = QLabel("")
        self.addWidget(self._message_label)

        # ★ 校准状态存储
        self._calibrated = False
        self._pixel_resolution_m = 0.5

    # ===================================================================
    # 更新方法
    # ===================================================================

    def update_coords(self, x: int, y: int):
        self._coord_label.setText(f"  📍 ({x}, {y})")

    def update_geo_coords(self, lon: float = None, lat: float = None,
                          x_m: float = None, y_m: float = None):
        """更新经纬度和平面坐标（校准后）。"""
        if lon is not None and lat is not None and x_m is not None and y_m is not None:
            self._geo_label.setText(f"🌐 lon={lon:.6f} lat={lat:.6f}  x={x_m:.1f}m y={y_m:.1f}m")
            self._geo_label.setVisible(True)
            self._geo_label.setMinimumWidth(380)
        elif lon is not None and lat is not None:
            self._geo_label.setText(f"🌐 ({lon:.6f}, {lat:.6f})")
            self._geo_label.setVisible(True)
            self._geo_label.setMinimumWidth(180)
        else:
            self._geo_label.setText("")
            self._geo_label.setVisible(False)
            self._geo_label.setMinimumWidth(180)

    def update_zoom(self, zoom: float):
        self._zoom_label.setText(f"🔍 {zoom*100:.0f}%")

    def update_tool(self, tool_name: str):
        tool_names = {
            "pan":              "🖐 平移",
            "open":             "📂 打开影像",
            "positive_sample":  "➕ 道路样本",
            "negative_sample":  "➖ 非道路样本",
            "roi":              "🔲 ROI 保留区",
            "ignore":           "🚫 Ignore 删除区",
            "mask_refine":      "🖌 Mask 精修",
            "polyline":         "📏 折线补路",
            "postprocess":      "⚙ 后处理",
            "skeleton":         "🦴 骨架",
            "optimize":         "✨ 优化骨架",
            "graph":            "🔗 路网",
            "plan":             "🗺 路径",
            "export":           "💾 导出",
            "graph_add_node":    "🔵 添加节点",
            "graph_add_edge":    "🔗 添加边",
            "graph_delete_node": "❌ 删除节点",
            "graph_delete_edge": "❌ 删除边",
            "graph_move_node":   "↕ 移动节点",
            "graph_merge_nodes": "🔀 合并节点",
            "graph_draw_edge":   "✏ 手动画边",
            "graph_save":        "💾 保存路网",
            "set_start":        "🚩 起点",
            "set_end":          "🏁 终点",
            "add_task":         "📍 任务点",
        }
        self._tool_label.setText(tool_names.get(tool_name, f"🔧 {tool_name}"))

    def update_stage(self, stage: str):
        stage_names = {
            "import":    "① 导入影像",
            "segment":   "② 道路分割",
            "edit":      "③ 区域修正",
            "skeleton":  "④ 骨架优化",
            "graph":     "⑤ 路网编辑",
            "calibrate": "⑥ 坐标校准",
            "export":    "⑦ 路径规划/导出",
        }
        self._stage_label.setText(stage_names.get(stage, stage))

    def update_road_ratio(self, ratio: float = None):
        if ratio is not None:
            self._road_ratio_label.setText(f"🛣 {ratio*100:.1f}%")
        else:
            self._road_ratio_label.setText("🛣 --")

    def update_nodes(self, count: int = None):
        if count is not None:
            self._nodes_label.setText(f"🔵 节点: {count}")
        else:
            self._nodes_label.setText("🔵 节点: --")

    def update_edges(self, count: int = None):
        if count is not None:
            self._edges_label.setText(f"🔗 边: {count}")
        else:
            self._edges_label.setText("🔗 边: --")

    def update_resolution(self, pixel_resolution_m: float = None, calibrated: bool = False):
        """更新分辨率显示"""
        self._pixel_resolution_m = pixel_resolution_m or 0.5
        self._calibrated = calibrated
        if calibrated:
            self._resolution_label.setText("📏 已校准")
            self._resolution_label.setStyleSheet("color: #50fa7b;")
        else:
            self._resolution_label.setText(f"📏 {self._pixel_resolution_m:.2f} m/px")
            self._resolution_label.setStyleSheet("color: #a6adc8;")

    def show_message(self, msg: str, timeout: int = 5000):
        self._message_label.setText(msg)

    def clear_message(self):
        self._message_label.setText("")
