"""
V3.2 人工 graph 编辑器

功能：
1. 加载 draft_graph.json 作为初稿
2. 在原图上显示节点和边
3. 支持人工编辑：
   - 左键点击添加节点
   - 连续点击两个节点添加边
   - 鼠标拖动移动节点
   - 选中边后按 d 删除边
   - 选中节点后按 d 删除节点
   - 选中两个节点后按 m 合并节点
   - 选中一条边后按 s 拆分边
   - 按 u 撤销
   - 按 Ctrl+S / s 保存
4. 支持手动画 polyline edge：
   - 按 p 进入 polyline 模式
   - 左键依次点击道路中心点
   - Enter 确认生成边
5. 保存 final_nodes.csv / final_edges.csv / final_graph.json / overlay
"""

import csv
import copy
import gc
import json
import math
import os
import numpy as np
from roadnet.graph_utils import has_points
from typing import List, Tuple, Set, Dict, Optional, Any

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import cv2


# ===========================================================================
# 中文字体支持
# ===========================================================================

def _setup_cjk_font() -> None:
    """设置中文字体，使 matplotlib 能显示中文。"""
    try:
        from matplotlib.font_manager import FontProperties
        import platform
        system = platform.system()
        if system == "Windows":
            _fonts = [
                "Microsoft YaHei", "SimHei", "SimSun",
                "KaiTi", "FangSong", "Arial",
            ]
        elif system == "Darwin":
            _fonts = [
                "PingFang SC", "Heiti SC", "STHeiti",
                "Apple LiGothic", "Arial Unicode MS",
            ]
        else:
            _fonts = [
                "WenQuanYi Micro Hei", "Noto Sans CJK SC",
                "Droid Sans Fallback", "DejaVu Sans",
            ]
        for _f in _fonts:
            try:
                FontProperties(family=_f)
                plt.rcParams["font.family"] = _f
                return
            except Exception:
                continue
    except Exception:
        pass
    plt.rcParams["font.family"] = "sans-serif"


def _force_close_window(plt_module, fig):
    """强制关闭 matplotlib 窗口。"""
    try:
        plt_module.close(fig)
    except Exception:
        pass
    try:
        plt_module.close("all")
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass


# ===========================================================================
# 图数据
# ===========================================================================

class GraphState:
    """路网图的状态快照（用于撤销）。"""

    def __init__(
        self,
        nodes: List[Dict],
        edges: List[Dict],
    ):
        self.nodes = copy.deepcopy(nodes)
        self.edges = copy.deepcopy(edges)

    def snapshot(self) -> Tuple[List[Dict], List[Dict]]:
        return copy.deepcopy(self.nodes), copy.deepcopy(self.edges)


# ===========================================================================
# 图编辑器核心
# ===========================================================================

class GraphEditor:
    """
    交互式路网图编辑器。

    操作说明：
    - 左键点击空白处: 添加节点
    - 左键点击已有节点: 选中
    - 连续选中两个节点: 自动添加边
    - 左键拖动节点: 移动节点
    - 左键点击边附近: 选中边
    - d 键: 删除选中的节点或边
    - m 键: 合并选中的两个节点
    - s 键: 拆分选中的边（在中间位置插入节点）
    - p 键: 进入 polyline 画线模式
      - 左键点击: 添加路径点
      - Enter: 确认生成边（连接最近的两个节点）
    - u 键: 撤销
    - Ctrl+S / s: 保存
    - Esc: 退出
    """

    def __init__(
        self,
        image_rgb: np.ndarray,
        draft_nodes: List[Dict],
        draft_edges: List[Dict],
        output_dir: str,
        max_undo_steps: int = 50,
    ):
        """
        Args:
            image_rgb:    原始 RGB 图像
            draft_nodes:  初稿节点列表
            draft_edges:  初稿边列表
            output_dir:   输出目录
            max_undo_steps: 最大撤销步数
        """
        self._image_rgb = image_rgb
        self._output_dir = output_dir
        self._max_undo_steps = max_undo_steps

        # 初始化数据
        self._nodes = copy.deepcopy(draft_nodes)
        self._edges = copy.deepcopy(draft_edges)
        self._next_node_id = max((n["id"] for n in self._nodes), default=-1) + 1
        self._next_edge_id = max((e["id"] for e in self._edges), default=-1) + 1

        # 选中状态
        self._selected_nodes: List[int] = []      # 最多 2 个 node_id
        self._selected_edge: Optional[int] = None  # edge_id
        self._hovered_node: Optional[int] = None
        self._hovered_edge: Optional[int] = None

        # 拖动状态
        self._dragging_node: Optional[int] = None
        self._drag_start: Optional[Tuple[int, int]] = None

        # Polyline 模式
        self._polyline_mode = False
        self._polyline_pts: List[Tuple[int, int]] = []

        # 撤销栈
        self._undo_stack: List[GraphState] = []
        self._push_undo()

        # matplotlib 状态
        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[plt.Axes] = None
        self._plt: Any = plt
        self._is_cancelled = False
        self._is_confirmed = False

        # 渲染缓存
        self._node_scatter = None
        self._edge_lines: List = []
        self._polyline_plot = None
        self._selected_node_artists: List = []
        self._selected_edge_artist = None
        self._status_text = None

        # 命中范围
        self._node_hit_radius = 12
        self._edge_hit_distance = 8

    # ------ 属性 ------
    @property
    def final_nodes(self) -> List[Dict]:
        return self._nodes

    @property
    def final_edges(self) -> List[Dict]:
        return self._edges

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled

    @property
    def is_confirmed(self) -> bool:
        return self._is_confirmed

    # ------ 撤销管理 ------
    def _push_undo(self):
        state = GraphState(self._nodes, self._edges)
        self._undo_stack.append(state)
        if len(self._undo_stack) > self._max_undo_steps:
            self._undo_stack.pop(0)

    def undo(self):
        if len(self._undo_stack) <= 1:
            print("[GRAPH EDIT] 无法撤销（已达最早状态）")
            return False
        # 丢弃当前状态，回到上一个
        self._undo_stack.pop()
        state = self._undo_stack[-1]
        self._nodes, self._edges = state.snapshot()
        self._selected_nodes.clear()
        self._selected_edge = None
        self._next_node_id = max((n["id"] for n in self._nodes), default=-1) + 1
        self._next_edge_id = max((e["id"] for e in self._edges), default=-1) + 1
        self._refresh()
        print(f"[GRAPH EDIT] 已撤销（剩余 {len(self._undo_stack) - 1} 步）")
        return True

    # ------ 节点/边操作 ------
    def _add_node(self, y: int, x: int):
        """添加新节点。"""
        nid = self._next_node_id
        self._next_node_id += 1
        node = {"id": nid, "y": y, "x": x, "type": "endpoint"}
        self._nodes.append(node)
        self._push_undo()
        print(f"[GRAPH EDIT] 添加节点 {nid} at ({y}, {x})")

    def _add_edge(self, from_id: int, to_id: int):
        """在两个节点之间添加边。"""
        if from_id == to_id:
            return
        # 检查是否已存在边
        for e in self._edges:
            if {e["from"], e["to"]} == {from_id, to_id}:
                print(f"[GRAPH EDIT] 边 {from_id}↔{to_id} 已存在")
                return

        # 计算路径和长度
        node_map = {n["id"]: n for n in self._nodes}
        n1 = node_map[from_id]
        n2 = node_map[to_id]
        dy = n2["y"] - n1["y"]
        dx = n2["x"] - n1["x"]
        dist = math.sqrt(dy * dy + dx * dx)

        eid = self._next_edge_id
        self._next_edge_id += 1
        edge = {
            "id": eid,
            "from": from_id,
            "to": to_id,
            "length_px": round(dist, 2),
            "path": [[n1["y"], n1["x"]], [n2["y"], n2["x"]]],
        }
        self._edges.append(edge)
        # 更新节点类型
        self._update_node_types()
        self._push_undo()
        print(f"[GRAPH EDIT] 添加边 {eid}: {from_id}→{to_id} (长度={dist:.1f}px)")

    def _delete_node(self, node_id: int):
        """删除节点及其所有关联边。"""
        if node_id not in {n["id"] for n in self._nodes}:
            return
        self._nodes = [n for n in self._nodes if n["id"] != node_id]
        self._edges = [e for e in self._edges
                       if e["from"] != node_id and e["to"] != node_id]
        self._update_node_types()
        self._push_undo()
        print(f"[GRAPH EDIT] 删除节点 {node_id}")

    def _delete_edge(self, edge_id: int):
        """删除边。"""
        if edge_id not in {e["id"] for e in self._edges}:
            return
        self._edges = [e for e in self._edges if e["id"] != edge_id]
        self._update_node_types()
        self._push_undo()
        print(f"[GRAPH EDIT] 删除边 {edge_id}")

    def _merge_nodes(self, id1: int, id2: int):
        """合并两个节点为质心。"""
        node_map = {n["id"]: n for n in self._nodes}
        if id1 not in node_map or id2 not in node_map:
            return
        n1, n2 = node_map[id1], node_map[id2]

        # 计算质心
        new_y = int(round((n1["y"] + n2["y"]) / 2))
        new_x = int(round((n1["x"] + n2["x"]) / 2))
        new_id = self._next_node_id
        self._next_node_id += 1

        # 合并类型：任意一个是 junction 则为 junction
        new_type = "junction" if (n1["type"] == "junction" or n2["type"] == "junction") else "endpoint"

        # 移除旧节点
        self._nodes = [n for n in self._nodes if n["id"] not in (id1, id2)]
        self._nodes.append({"id": new_id, "y": new_y, "x": new_x, "type": new_type})

        # 更新边中引用
        for e in self._edges:
            if e["from"] in (id1, id2):
                e["from"] = new_id
            if e["to"] in (id1, id2):
                e["to"] = new_id

        # 去除重复边
        dedup: Dict[Tuple[int, int], Dict] = {}
        for e in self._edges:
            a, b = sorted([e["from"], e["to"]])
            key = (a, b)
            if key not in dedup or e.get("length_px", 0) > dedup[key].get("length_px", 0):
                dedup[key] = e
        self._edges = list(dedup.values())

        self._update_node_types()
        self._push_undo()
        print(f"[GRAPH EDIT] 合并节点 {id1}+{id2} → 新节点 {new_id}")

    def _split_edge(self, edge_id: int):
        """在边的中点拆分边。"""
        edge_map = {e["id"]: e for e in self._edges}
        if edge_id not in edge_map:
            return
        edge = edge_map[edge_id]

        path = edge.get("path", [])
        if len(path) < 2:
            return

        # 找到路径中点
        mid_idx = len(path) // 2
        mid_y, mid_x = path[mid_idx]

        # 创建新节点
        new_node_id = self._next_node_id
        self._next_node_id += 1
        new_node = {"id": new_node_id, "y": int(mid_y), "x": int(mid_x), "type": "endpoint"}
        self._nodes.append(new_node)

        # 创建两条新边
        eid1 = self._next_edge_id
        self._next_edge_id += 1
        eid2 = self._next_edge_id
        self._next_edge_id += 1

        path1 = path[:mid_idx + 1]
        path2 = path[mid_idx:]

        len1 = _path_length(path1)
        len2 = _path_length(path2)

        self._edges.append({
            "id": eid1, "from": edge["from"], "to": new_node_id,
            "length_px": round(len1, 2), "path": path1,
        })
        self._edges.append({
            "id": eid2, "from": new_node_id, "to": edge["to"],
            "length_px": round(len2, 2), "path": path2,
        })

        # 删除原始边
        self._edges = [e for e in self._edges if e["id"] != edge_id]

        self._update_node_types()
        self._push_undo()
        print(f"[GRAPH EDIT] 拆分边 {edge_id} → 新节点 {new_node_id} + 边 {eid1}, {eid2}")

    def _move_node(self, node_id: int, new_y: int, new_x: int):
        """移动节点到新位置。"""
        for n in self._nodes:
            if n["id"] == node_id:
                n["y"] = new_y
                n["x"] = new_x
                break
        # 更新相关边的 path 起点/终点
        for e in self._edges:
            if e["from"] == node_id and has_points(e.get("path")):
                e["path"][0] = [new_y, new_x]
                e["length_px"] = round(_path_length(e["path"]), 2)
            if e["to"] == node_id and has_points(e.get("path")):
                e["path"][-1] = [new_y, new_x]
                e["length_px"] = round(_path_length(e["path"]), 2)

    def _add_polyline_edge(self, polyline_pts: List[Tuple[int, int]]):
        """将手动绘制的 polyline 变成图的一条边。"""
        if len(polyline_pts) < 2:
            return

        # 找最近的两个端点
        node_map = {n["id"]: n for n in self._nodes}
        start_pt = polyline_pts[0]
        end_pt = polyline_pts[-1]

        def _dist(pid, pt):
            n = node_map[pid]
            return math.sqrt((n["y"] - pt[0]) ** 2 + (n["x"] - pt[1]) ** 2)

        # 为起点和终点分别找最近的节点（距离 <= 30px 则复用）
        best_start_id = None
        best_start_dist = 30
        best_end_id = None
        best_end_dist = 30

        for n in self._nodes:
            d_start = math.sqrt((n["y"] - start_pt[0]) ** 2 + (n["x"] - start_pt[1]) ** 2)
            d_end = math.sqrt((n["y"] - end_pt[0]) ** 2 + (n["x"] - end_pt[1]) ** 2)
            if d_start < best_start_dist:
                best_start_dist = d_start
                best_start_id = n["id"]
            if d_end < best_end_dist:
                best_end_dist = d_end
                best_end_id = n["id"]

        if best_start_id is None:
            best_start_id = self._next_node_id
            self._next_node_id += 1
            self._nodes.append({
                "id": best_start_id, "y": int(start_pt[0]),
                "x": int(start_pt[1]), "type": "endpoint",
            })
            # 以新节点作为 path 起点
            polyline_pts = [(int(start_pt[0]), int(start_pt[1]))] + list(polyline_pts[1:])
        else:
            polyline_pts = [(node_map[best_start_id]["y"], node_map[best_start_id]["x"])] + list(polyline_pts[1:])

        if best_end_id is None or best_end_id == best_start_id:
            best_end_id = self._next_node_id
            self._next_node_id += 1
            self._nodes.append({
                "id": best_end_id, "y": int(end_pt[0]),
                "x": int(end_pt[1]), "type": "endpoint",
            })
            polyline_pts = list(polyline_pts[:-1]) + [(int(end_pt[0]), int(end_pt[1]))]
        else:
            polyline_pts = list(polyline_pts[:-1]) + [(node_map[best_end_id]["y"], node_map[best_end_id]["x"])]

        path = [[y, x] for y, x in polyline_pts]
        plen = _path_length(path)

        eid = self._next_edge_id
        self._next_edge_id += 1
        edge = {
            "id": eid, "from": best_start_id, "to": best_end_id,
            "length_px": round(plen, 2), "path": path,
        }
        self._edges.append(edge)

        self._update_node_types()
        self._push_undo()
        print(f"[GRAPH EDIT] 添加 polyline 边 {eid}: {best_start_id}→{best_end_id} "
              f"(长度={plen:.1f}px, 顶点数={len(path)})")

    def _update_node_types(self):
        """根据边的连接情况更新节点类型。"""
        degree = {n["id"]: 0 for n in self._nodes}
        for e in self._edges:
            degree[e["from"]] = degree.get(e["from"], 0) + 1
            degree[e["to"]] = degree.get(e["to"], 0) + 1
        for n in self._nodes:
            deg = degree.get(n["id"], 0)
            if deg <= 1:
                n["type"] = "endpoint"
            else:
                n["type"] = "junction"

    # ------ 命中检测 ------
    def _find_node_at(self, x: float, y: float) -> Optional[int]:
        """查找 (x, y) 附近最近的节点（显示屏坐标）。"""
        if self._ax is None:
            return None
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        # 将显示屏坐标转为数据坐标
        disp_to_data = self._ax.transData.inverted()
        dx, dy = disp_to_data.transform([(0, 0), (self._node_hit_radius, self._node_hit_radius)])
        radius_x = abs(dx[1] - dx[0])
        radius_y = abs(dy[1] - dy[0])
        radius = max(radius_x, radius_y)

        best_id = None
        best_dist = float("inf")
        for n in self._nodes:
            d = math.sqrt((n["x"] - x) ** 2 + (n["y"] - y) ** 2)
            if d < radius and d < best_dist:
                best_dist = d
                best_id = n["id"]
        return best_id

    def _find_edge_at(self, x: float, y: float) -> Optional[int]:
        """查找 (x, y) 附近最近的边。"""
        if self._ax is None:
            return None
        disp_to_data = self._ax.transData.inverted()
        dx, dy = disp_to_data.transform([(0, 0), (self._edge_hit_distance, self._edge_hit_distance)])
        radius = max(abs(dx[1] - dx[0]), abs(dy[1] - dy[0]))

        best_id = None
        best_dist = float("inf")
        for e in self._edges:
            path = e.get("path", [])
            for i in range(len(path) - 1):
                d = _point_to_segment_distance(x, y,
                                                path[i][1], path[i][0],
                                                path[i + 1][1], path[i + 1][0])
                if d < radius and d < best_dist:
                    best_dist = d
                    best_id = e["id"]
        return best_id

    # ------ 事件处理 ------
    def _on_press(self, event):
        if event.inaxes != self._ax or self._ax is None:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        button = event.button

        # ---- Polyline 模式 ----
        if self._polyline_mode:
            if button == MouseButton.LEFT:
                self._polyline_pts.append((int(round(y)), int(round(x))))
                self._draw_polyline()
                print(f"[GRAPH EDIT] Polyline 添加点: ({int(round(y))}, {int(round(x))})")
            return

        # ---- 普通模式 ----
        if button == MouseButton.LEFT:
            # 先检查是否在节点上
            node_id = self._find_node_at(x, y)

            if node_id is not None:
                # 点击了节点
                if node_id in self._selected_nodes:
                    # 取消选中
                    self._selected_nodes.remove(node_id)
                else:
                    self._selected_nodes.append(node_id)
                    # 选中两个节点 → 自动添加边
                    if len(self._selected_nodes) == 2:
                        self._add_edge(self._selected_nodes[0], self._selected_nodes[1])
                        self._selected_nodes.clear()
                    elif len(self._selected_nodes) > 2:
                        self._selected_nodes = self._selected_nodes[-1:]

                self._selected_edge = None

                # 开始拖动
                self._dragging_node = node_id
                self._drag_start = (int(round(y)), int(round(x)))
            else:
                # 检查是否在边上
                edge_id = self._find_edge_at(x, y)
                if edge_id is not None:
                    self._selected_edge = edge_id
                    self._selected_nodes.clear()
                else:
                    # 空白区域添加节点
                    self._add_node(int(round(y)), int(round(x)))
                    self._selected_nodes.clear()
                    self._selected_edge = None
                self._dragging_node = None
            self._refresh()

    def _on_release(self, event):
        if event.inaxes != self._ax:
            return

        if self._dragging_node is not None and event.xdata is not None and event.ydata is not None:
            new_y = int(round(event.ydata))
            new_x = int(round(event.xdata))
            start_y, start_x = self._drag_start
            if abs(new_y - start_y) > 2 or abs(new_x - start_x) > 2:
                self._move_node(self._dragging_node, new_y, new_x)
                self._push_undo()
                print(f"[GRAPH EDIT] 移动节点 {self._dragging_node} → ({new_y}, {new_x})")

        self._dragging_node = None
        self._drag_start = None
        self._refresh()

    def _on_motion(self, event):
        if event.inaxes != self._ax:
            return

        if self._dragging_node is not None and event.xdata is not None and event.ydata is not None:
            new_y = int(round(event.ydata))
            new_x = int(round(event.xdata))
            self._move_node(self._dragging_node, new_y, new_x)
            self._refresh()

        # 悬停检测
        if event.xdata is not None and event.ydata is not None:
            hovered_n = self._find_node_at(event.xdata, event.ydata)
            hovered_e = self._find_edge_at(event.xdata, event.ydata)
            if hovered_n != self._hovered_node or hovered_e != self._hovered_edge:
                self._hovered_node = hovered_n
                self._hovered_edge = hovered_e
                self._update_status()

    def _on_key(self, event):
        key = event.key
        ctrl = event.key.startswith("ctrl+") if event.key else False

        # Esc — 退出
        if key in ("escape", "esc", "\x1b"):
            self._is_cancelled = True
            _force_close_window(self._plt, self._fig)

        # Ctrl+S / s — 保存
        elif key == "ctrl+s" or (key == "s" and not self._polyline_mode):
            self.save()
            self._is_confirmed = True

        # d — 删除选中
        elif key == "d":
            if self._selected_edge is not None:
                self._delete_edge(self._selected_edge)
                self._selected_edge = None
            elif self._selected_nodes:
                for nid in list(self._selected_nodes):
                    self._delete_node(nid)
                self._selected_nodes.clear()
            self._refresh()

        # m — 合并选中的两个节点
        elif key == "m":
            if len(self._selected_nodes) >= 2:
                self._merge_nodes(self._selected_nodes[0], self._selected_nodes[1])
                self._selected_nodes.clear()
                self._refresh()

        # s — 拆分选中的边
        elif key == "s" and self._selected_edge is not None:
            self._split_edge(self._selected_edge)
            self._selected_edge = None
            self._refresh()

        # p — 切换 polyline 模式
        elif key == "p":
            self._polyline_mode = not self._polyline_mode
            if self._polyline_mode:
                self._polyline_pts.clear()
                print("[GRAPH EDIT] 进入 Polyline 画线模式（左键添加点，Enter 确认，Esc 取消）")
            else:
                self._polyline_pts.clear()
                print("[GRAPH EDIT] 退出 Polyline 画线模式")
            self._refresh()

        # Enter — 在 polyline 模式下确认
        elif key == "enter":
            if self._polyline_mode and len(self._polyline_pts) >= 2:
                self._add_polyline_edge(self._polyline_pts)
                self._polyline_pts.clear()
                self._polyline_mode = False
                self._refresh()
            elif self._polyline_mode:
                self._polyline_pts.clear()
                self._polyline_mode = False
                print("[GRAPH EDIT] Polyline 点数不足，取消")
                self._refresh()

        # u — 撤销
        elif key == "u":
            self.undo()
            self._refresh()

        # c — 清空选择
        elif key == "c":
            self._selected_nodes.clear()
            self._selected_edge = None
            self._refresh()

    # ------ 渲染 ------
    def _render(self):
        """完整重绘。"""
        if self._ax is None:
            return

        self._ax.clear()
        self._ax.imshow(self._image_rgb)

        h, w = self._image_rgb.shape[:2]
        self._ax.set_xlim(0, w)
        self._ax.set_ylim(h, 0)
        self._ax.axis("off")

        # 绘制边
        for e in self._edges:
            path = e.get("path", [])
            if len(path) < 2:
                continue
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            color = "cyan" if e["id"] == self._selected_edge else "blue"
            lw = 3 if e["id"] == self._selected_edge else 1.5
            self._ax.plot(xs, ys, color=color, linewidth=lw, alpha=0.9, picker=False)

        # 绘制 hovered edge
        if self._hovered_edge is not None and self._hovered_edge != self._selected_edge:
            for e in self._edges:
                if e["id"] == self._hovered_edge:
                    path = e.get("path", [])
                    if len(path) >= 2:
                        xs = [p[1] for p in path]
                        ys = [p[0] for p in path]
                        self._ax.plot(xs, ys, color="lightblue", linewidth=2.5, alpha=0.6)

        # 绘制节点
        for n in self._nodes:
            color = "red" if n["type"] == "junction" else "lime"
            size = 80 if n["id"] in self._selected_nodes else 50
            ec = "yellow" if n["id"] in self._selected_nodes else "white"
            lw = 2 if n["id"] in self._selected_nodes else 1
            self._ax.scatter(n["x"], n["y"], c=color, s=size, edgecolors=ec,
                             linewidths=lw, zorder=5)

            # 标签
            self._ax.annotate(str(n["id"]), (n["x"] + 8, n["y"] - 4),
                              color="white", fontsize=7, weight="bold",
                              bbox=dict(boxstyle="round,pad=0.2",
                                        facecolor="black", alpha=0.6))

        # 绘制 hovered node
        if self._hovered_node is not None and self._hovered_node not in self._selected_nodes:
            for n in self._nodes:
                if n["id"] == self._hovered_node:
                    self._ax.scatter(n["x"], n["y"], c="none", s=100, edgecolors="yellow",
                                     linewidths=2, zorder=4)

        # 绘制 polyline 点
        if self._polyline_mode and self._polyline_pts:
            xs = [p[1] for p in self._polyline_pts]
            ys = [p[0] for p in self._polyline_pts]
            self._ax.scatter(xs, ys, c="magenta", s=30, zorder=6)
            if len(self._polyline_pts) >= 2:
                self._ax.plot(xs, ys, "magenta", linewidth=2, linestyle="--", alpha=0.8)

        self._fig.canvas.draw_idle()

    def _draw_polyline(self):
        """仅更新 polyline 渲染（增量更新）。"""
        self._refresh()

    def _refresh(self):
        """刷新显示。"""
        self._render()
        self._update_status()

    def _update_status(self):
        """更新标题栏状态信息。"""
        if self._ax is None:
            return
        parts = [f"节点: {len(self._nodes)}", f"边: {len(self._edges)}"]
        if self._polyline_mode:
            parts.append(f"[POLYLINE] 点数={len(self._polyline_pts)} | Enter=确认 | Esc=取消")
        if self._selected_nodes:
            parts.append(f"选中节点: {self._selected_nodes}")
        if self._selected_edge is not None:
            parts.append(f"选中边: {self._selected_edge}")
        if self._hovered_node is not None:
            parts.append(f"悬停节点: {self._hovered_node}")
        elif self._hovered_edge is not None:
            parts.append(f"悬停边: {self._hovered_edge}")

        status = " | ".join(parts)
        self._ax.set_title(status, fontsize=9, pad=5,
                           color="white", backgroundcolor="black")

    # ------ 保存 ------
    def save(self):
        """保存 final graph 到文件。"""
        # 更新 degree
        degree = {n["id"]: 0 for n in self._nodes}
        for e in self._edges:
            degree[e["from"]] = degree.get(e["from"], 0) + 1
            degree[e["to"]] = degree.get(e["to"], 0) + 1
        for n in self._nodes:
            n["degree"] = degree.get(n["id"], 0)

        # ---- final_nodes.csv ----
        nodes_path = os.path.join(self._output_dir, "final_nodes.csv")
        with open(nodes_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["node_id", "y", "x", "type", "degree"])
            for n in self._nodes:
                writer.writerow([n["id"], n["y"], n["x"], n["type"], n.get("degree", 0)])
        print(f"[GRAPH EDIT] 已保存: {nodes_path} ({len(self._nodes)} 个)")

        # ---- final_edges.csv ----
        edges_path = os.path.join(self._output_dir, "final_edges.csv")
        with open(edges_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["edge_id", "from_node", "to_node", "length_px", "path_points"])
            for e in self._edges:
                path_pts = len(e.get("path", []))
                writer.writerow([e["id"], e["from"], e["to"], e["length_px"], path_pts])
        print(f"[GRAPH EDIT] 已保存: {edges_path} ({len(self._edges)} 条)")

        # ---- final_graph.json ----
        graph_path = os.path.join(self._output_dir, "final_graph.json")
        graph = {
            "nodes": self._nodes,
            "edges": self._edges,
            "metadata": {
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
                "image_size": {
                    "width": self._image_rgb.shape[1],
                    "height": self._image_rgb.shape[0],
                },
            },
        }
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)
        print(f"[GRAPH EDIT] 已保存: {graph_path}")

        # ---- final_graph_overlay.png ----
        self._save_overlay()

        print(f"\n[GRAPH EDIT] ✅ 全部保存完成！共 {len(self._nodes)} 节点, {len(self._edges)} 边")

    def _save_overlay(self):
        """保存叠加图。"""
        img = cv2.cvtColor(self._image_rgb, cv2.COLOR_RGB2BGR)

        # 边 — 蓝色
        for e in self._edges:
            path = e.get("path", [])
            for i in range(len(path) - 1):
                p1 = (path[i][1], path[i][0])
                p2 = (path[i + 1][1], path[i + 1][0])
                cv2.line(img, p1, p2, (255, 0, 0), 2, cv2.LINE_AA)

        # 端点 — 绿色
        for n in self._nodes:
            if n["type"] == "endpoint":
                cv2.circle(img, (n["x"], n["y"]), 6, (0, 255, 0), -1)
                cv2.circle(img, (n["x"], n["y"]), 6, (0, 180, 0), 2)

        # 交叉点 — 红色
        for n in self._nodes:
            if n["type"] == "junction":
                cv2.circle(img, (n["x"], n["y"]), 7, (0, 0, 255), -1)
                cv2.circle(img, (n["x"], n["y"]), 7, (0, 0, 180), 2)

        # 标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        for n in self._nodes:
            cv2.putText(img, str(n["id"]), (n["x"] + 10, n["y"] - 6),
                        font, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

        overlay_path = os.path.join(self._output_dir, "final_graph_overlay.png")
        cv2.imwrite(overlay_path, img)
        print(f"[GRAPH EDIT] 已保存: {overlay_path}")

    # ------ 运行 ------
    def run(self):
        """启动交互编辑器。"""
        _setup_cjk_font()

        self._fig, self._ax = plt.subplots(figsize=(14, 10))
        self._fig.canvas.manager.set_window_title("Graph Editor V3.2 — 路网图编辑器")

        # 注册事件
        self._fig.canvas.mpl_connect("button_press_event", self._on_press)
        self._fig.canvas.mpl_connect("button_release_event", self._on_release)
        self._fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        # 初始渲染
        self._refresh()

        print("\n" + "=" * 60)
        print("[GRAPH EDIT] 路网图编辑器 V3.2")
        print("[GRAPH EDIT] 操作说明:")
        print("  左键空白处: 添加节点       d: 删除选中")
        print("  左键点击节点: 选中          m: 合并两个节点")
        print("  左键拖动节点: 移动          s: 拆分边")
        print("  连续选两个节点: 添加边      p: Polyline 画线模式")
        print("  左键点击边: 选中             u: 撤销")
        print("  c: 清空选择                 Ctrl+S: 保存并退出")
        print("  Esc: 退出（不保存）")
        print("=" * 60)
        print(f"[GRAPH EDIT] 已加载 {len(self._nodes)} 个节点, {len(self._edges)} 条边")

        plt.tight_layout()
        plt.show()


# ===========================================================================
# 辅助函数
# ===========================================================================

def _path_length(path: List[List]) -> float:
    """计算路径总长度。"""
    total = 0.0
    for i in range(len(path) - 1):
        dy = path[i + 1][0] - path[i][0]
        dx = path[i + 1][1] - path[i][1]
        total += math.sqrt(dy * dy + dx * dx)
    return total


def _point_to_segment_distance(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float,
) -> float:
    """计算点到线段的距离。"""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def load_draft_graph(graph_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    从 JSON 加载 draft graph。

    Returns:
        (nodes, edges)
    """
    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("nodes", []), data.get("edges", [])
