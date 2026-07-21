"""
V4 Qt 原生路网图编辑器

将原来基于 matplotlib 的 GraphEditor 迁移为纯数据结构操作层，
由 main_window 在 QGraphicsScene 上渲染节点/边图形。

功能：
- 添加/删除/移动/合并节点
- 添加/删除/手动画边
- 撤销/重做 (Ctrl+Z / Ctrl+Y)
- 保存 final_graph 到文件
- 连通分量计算
"""

import copy
import csv
import json
import math
import os
from typing import List, Dict, Tuple, Set, Optional

import cv2
import numpy as np
from roadnet.graph_utils import polyline_to_list, ensure_python_types


class UndoStack:
    """简单的撤销/重做栈"""
    def __init__(self, max_steps: int = 50):
        self._stack: List[Tuple[List[Dict], List[Dict]]] = []
        self._index: int = -1
        self._max_steps = max_steps

    def push(self, nodes: List[Dict], edges: List[Dict]):
        """保存当前状态"""
        # 丢弃 index 之后的 redo 状态
        self._stack = self._stack[:self._index + 1]
        self._stack.append((copy.deepcopy(nodes), copy.deepcopy(edges)))
        if len(self._stack) > self._max_steps:
            self._stack.pop(0)
        self._index = len(self._stack) - 1

    def undo(self) -> Optional[Tuple[List[Dict], List[Dict]]]:
        if self._index > 0:
            self._index -= 1
            nodes, edges = self._stack[self._index]
            return copy.deepcopy(nodes), copy.deepcopy(edges)
        return None

    def redo(self) -> Optional[Tuple[List[Dict], List[Dict]]]:
        if self._index < len(self._stack) - 1:
            self._index += 1
            nodes, edges = self._stack[self._index]
            return copy.deepcopy(nodes), copy.deepcopy(edges)
        return None

    def can_undo(self) -> bool:
        return self._index > 0

    def can_redo(self) -> bool:
        return self._index < len(self._stack) - 1


class GraphEditorQt:
    """Qt 原生的路网图编辑器数据层"""

    def __init__(
        self,
        image_size: Tuple[int, int] = (0, 0),
        pixel_resolution_m: float = 0.5,
    ):
        self._image_w, self._image_h = image_size
        self._pixel_resolution_m = pixel_resolution_m

        self._nodes: List[Dict] = []     # {"id","x","y","type","source"}
        self._edges: List[Dict] = []     # {"id","start","end","length_pixel","points_pixel","source","enabled"}
        self._next_node_id: int = 0
        self._next_edge_id: int = 0
        self._pixel_resolution_m: float = pixel_resolution_m

        # 选中状态
        self._selected_nodes: List[int] = []
        self._selected_edges: List[int] = []
        self._hovered_node: Optional[int] = None
        self._hovered_edge: Optional[int] = None

        # 交互状态
        self._dragging_node: Optional[int] = None
        self._edge_start_node: Optional[int] = None  # add_edge 的起点
        self._manual_edge_points: List[Tuple[int, int]] = []  # 手动画边的中间点
        self._merge_node_candidates: List[int] = []  # merge_nodes 的候选

        # 参数（可配置）
        self.merge_node_distance: int = 15
        self.min_edge_length: float = 40.0
        self.snap_distance: int = 10
        self.simplify_tolerance: float = 2.0

        # 选中/吸附命中范围
        self.node_select_radius: int = 12       # 节点选择半径
        self.node_hit_radius: int = 12          # 别名，兼容旧代码
        self.edge_select_radius: int = 8        # 边选择距离
        self.edge_hit_distance: int = 8         # 别名，兼容旧代码
        self.snap_node_distance: int = 15       # 手动画边吸附距离
        self.min_manual_edge_length: int = 5    # 最小人工边长度
        # 大图局部修路网默认吸附（可被 configure_large_repair_snaps 覆盖）
        self.node_snap_distance_px: int = 25
        self.junction_merge_distance_px: int = 30
        self.endpoint_snap_distance_px: int = 25
        self.edge_split_snap_distance_px: int = 20
        self.junction_cluster_radius_px: int = 30
        self.remove_internal_branch_length_px: int = 40
        self._last_validation_warnings: List[str] = []

        # 撤销（全局撤销由 GlobalHistoryManager 接管，此处仅做内部回退）
        self._undo_stack = UndoStack(max_steps=50)
        self._undo_stack.push(self._nodes, self._edges)

    # ------ 属性 ------
    @property
    def nodes(self) -> List[Dict]:
        return self._nodes

    @property
    def edges(self) -> List[Dict]:
        return self._edges

    @property
    def selected_nodes(self) -> List[int]:
        return self._selected_nodes

    @property
    def selected_edges(self) -> List[int]:
        return self._selected_edges

    @property
    def hovered_node(self) -> Optional[int]:
        return self._hovered_node

    @property
    def hovered_edge(self) -> Optional[int]:
        return self._hovered_edge

    @property
    def manual_edge_points(self) -> List[Tuple[int, int]]:
        return self._manual_edge_points

    @property
    def is_manual_edge_mode(self) -> bool:
        return len(self._manual_edge_points) > 0

    # ------ 节点查找 ------
    def find_node_at(self, x: float, y: float, radius: int = None) -> Optional[int]:
        """查找 (x, y) 附近的节点，使用 node_select_radius"""
        r = radius if radius is not None else self.node_select_radius
        best_id, best_dist = None, r
        for n in self._nodes:
            d = math.sqrt((n["x"] - x) ** 2 + (n["y"] - y) ** 2)
            if d < r and d < best_dist:
                best_dist = d
                best_id = n["id"]
        return best_id

    def find_node_near(self, x: float, y: float, snap_distance: int = None) -> Optional[Tuple[int, float]]:
        """查找 (x, y) 附近节点，返回 (node_id, distance)，用于吸附"""
        sd = snap_distance if snap_distance is not None else self.snap_node_distance
        best_id, best_dist = None, sd
        for n in self._nodes:
            d = math.sqrt((n["x"] - x) ** 2 + (n["y"] - y) ** 2)
            if d < sd and d < best_dist:
                best_dist = d
                best_id = n["id"]
        if best_id is not None:
            return (best_id, best_dist)
        return None

    def find_edge_at(self, x: float, y: float) -> Optional[int]:
        """查找 (x, y) 附近的边"""
        hit = self.find_edge_near(x, y, self.edge_select_radius)
        return hit[0] if hit else None

    def find_edge_near(
        self, x: float, y: float, max_dist: float = None
    ) -> Optional[Tuple[int, float, float, float]]:
        """查找附近边，返回 (edge_id, dist, proj_x, proj_y)。"""
        limit = float(max_dist if max_dist is not None else self.edge_select_radius)
        best = None
        best_dist = limit
        for e in self._edges:
            if e.get("enabled") is False:
                continue
            pts = e.get("points_pixel") or e.get("polyline") or []
            if len(pts) < 2:
                continue
            for i in range(len(pts) - 1):
                d, cx, cy = self._point_to_segment_proj(
                    x, y, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]
                )
                if d < best_dist:
                    best_dist = d
                    best = (int(e["id"]), float(d), float(cx), float(cy))
        return best

    @staticmethod
    def _point_to_segment(px, py, x1, y1, x2, y2) -> float:
        d, _, _ = GraphEditorQt._point_to_segment_proj(px, py, x1, y1, x2, y2)
        return d

    @staticmethod
    def _point_to_segment_proj(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2), float(x1), float(y1)
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        cx, cy = x1 + t * dx, y1 + t * dy
        return math.sqrt((px - cx) ** 2 + (py - cy) ** 2), float(cx), float(cy)

    def configure_large_repair_snaps(
        self,
        *,
        node_snap: int = 25,
        junction_merge: int = 30,
        endpoint_snap: int = 25,
        edge_split: int = 20,
        junction_cluster: int = 30,
    ):
        """大图局部修路网吸附参数。"""
        self.node_snap_distance_px = int(node_snap)
        self.junction_merge_distance_px = int(junction_merge)
        self.endpoint_snap_distance_px = int(endpoint_snap)
        self.edge_split_snap_distance_px = int(edge_split)
        self.junction_cluster_radius_px = int(junction_cluster)
        self.snap_node_distance = int(max(self.snap_node_distance, node_snap, endpoint_snap))
        self.node_select_radius = int(max(self.node_select_radius, node_snap))

    # ------ 编辑操作 ------
    def _add_node_raw(self, x: int, y: int, node_type: str = "manual") -> int:
        """添加节点（不推撤销栈，供内部使用）"""
        nid = self._next_node_id
        self._next_node_id += 1
        self._nodes.append({"id": nid, "x": x, "y": y, "type": node_type, "source": "manual"})
        print(f"[DEBUG][Graph] _add_node_raw id={nid} image=({x},{y}) node count={len(self._nodes)}")
        return nid

    def add_node(self, x: int, y: int, node_type: str = "manual") -> int:
        """添加节点，返回 node_id"""
        # 检查是否靠近已有节点
        near = self.find_node_at(x, y)
        if near is not None:
            print(f"[DEBUG][Graph] node already exists near ({x},{y}), id={near}")
            return near  # 返回已有节点
        nid = self._next_node_id
        self._next_node_id += 1
        self._nodes.append({"id": nid, "x": x, "y": y, "type": node_type, "source": "manual"})
        print(f"[DEBUG][Graph] add node id={nid} image=({x},{y})")
        print(f"[DEBUG][Graph] node count={len(self._nodes)}")
        return nid

    def delete_node(self, node_id: int) -> list:
        """删除节点及其关联边，返回被删除的关联边 id 列表"""
        if not any(n["id"] == node_id for n in self._nodes):
            return []
        related_edges = [e["id"] for e in self._edges
                         if e["start"] == node_id or e["end"] == node_id]
        self._nodes = [n for n in self._nodes if n["id"] != node_id]
        self._edges = [e for e in self._edges
                       if e["start"] != node_id and e["end"] != node_id]
        print(f"[DEBUG][Graph] delete node id={node_id}, related_edges={related_edges}")
        return related_edges

    def move_node(self, node_id: int, new_x: int, new_y: int):
        """移动节点"""
        for n in self._nodes:
            if n["id"] == node_id:
                n["x"] = new_x
                n["y"] = new_y
                break
        # 更新相关边
        for e in self._edges:
            pts = e.get("points_pixel", [])
            # ★ 使用 len() 显式判断，避免 numpy.ndarray 布尔歧义
            if len(pts) == 0:
                continue
            if e["start"] == node_id:
                pts[0] = [new_x, new_y]
            if e["end"] == node_id:
                pts[-1] = [new_x, new_y]
            e["length_pixel"] = round(self._path_length(pts), 2)

    def add_edge(self, start_id: int, end_id: int, source: str = "manual") -> Optional[int]:
        """在两个节点之间添加直线边"""
        if start_id == end_id:
            print(f"[DEBUG][Graph] add edge rejected: start==end={start_id}")
            return None
        # 检查重复
        for e in self._edges:
            if {e["start"], e["end"]} == {start_id, end_id}:
                print(f"[DEBUG][Graph] add edge rejected: duplicate edge {start_id}<->{end_id}")
                return None
        nm = {n["id"]: n for n in self._nodes}
        n1, n2 = nm.get(start_id), nm.get(end_id)
        if n1 is None or n2 is None:
            print(f"[DEBUG][Graph] add edge rejected: node not found start={start_id} end={end_id}")
            return None
        eid = self._next_edge_id
        self._next_edge_id += 1
        pts = [[n1["x"], n1["y"]], [n2["x"], n2["y"]]]
        length = round(self._path_length(pts), 2)
        edge = {
            "id": eid, "start": start_id, "end": end_id,
            "length_pixel": length,
            "points_pixel": pts, "source": source, "enabled": True,
        }
        self._edges.append(edge)
        print(f"[DEBUG][Graph] add edge id={eid} start={start_id} end={end_id} length={length}")
        return eid

    def delete_edge(self, edge_id: int):
        self._edges = [e for e in self._edges if e["id"] != edge_id]
        print(f"[DEBUG][Graph] delete edge id={edge_id}")

    def mark_edge_invalid(self, edge_id: int, reason: str = "manual_invalid"):
        for e in self._edges:
            if e["id"] == edge_id:
                e["enabled"] = False
                e["invalid"] = True
                e["invalid_reason"] = str(reason)
                e["source"] = e.get("source") or "manual_repair"
                return True
        return False

    def _resolve_endpoint(
        self, x: int, y: int, *, allow_edge_split: bool = True
    ) -> int:
        """吸附到已有节点，或在边上插入节点并 split，否则新建节点。"""
        snap = self.find_node_near(
            x, y, snap_distance=max(self.node_snap_distance_px, self.endpoint_snap_distance_px)
        )
        if snap is not None:
            return int(snap[0])
        if allow_edge_split:
            hit = self.find_edge_near(x, y, self.edge_split_snap_distance_px)
            if hit is not None:
                eid, _d, cx, cy = hit
                nid = self.split_edge_at_point(eid, int(round(cx)), int(round(cy)))
                if nid is not None:
                    return int(nid)
        return self._add_node_raw(int(x), int(y), node_type="manual")

    def split_edge_at_point(self, edge_id: int, x: int, y: int) -> Optional[int]:
        """在边上插入节点并拆成两条边，保留各自 polyline。返回新节点 id。"""
        edge = next((e for e in self._edges if e["id"] == edge_id), None)
        if edge is None:
            return None
        pts = polyline_to_list(edge.get("points_pixel") or edge.get("polyline") or [])
        if len(pts) < 2:
            return None

        # 找最近点所在线段
        best_i, best_d, best_xy = 0, 1e18, (float(x), float(y))
        for i in range(len(pts) - 1):
            d, cx, cy = self._point_to_segment_proj(
                x, y, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]
            )
            if d < best_d:
                best_d, best_i, best_xy = d, i, (cx, cy)

        nx, ny = int(round(best_xy[0])), int(round(best_xy[1]))
        # 若已落在端点附近，直接返回端点
        start_id, end_id = int(edge["start"]), int(edge["end"])
        for nid in (start_id, end_id):
            n = next((nn for nn in self._nodes if nn["id"] == nid), None)
            if n and math.hypot(n["x"] - nx, n["y"] - ny) <= 2:
                return nid

        new_id = self._add_node_raw(nx, ny, node_type="junction")
        left = pts[: best_i + 1] + [[nx, ny]]
        right = [[nx, ny]] + pts[best_i + 1 :]
        if len(left) < 2 or len(right) < 2:
            return new_id

        src = str(edge.get("source", "auto"))
        self._edges = [e for e in self._edges if e["id"] != edge_id]
        for a, b, poly in (
            (start_id, new_id, left),
            (new_id, end_id, right),
        ):
            eid = self._next_edge_id
            self._next_edge_id += 1
            self._edges.append({
                "id": eid,
                "start": a,
                "end": b,
                "length_pixel": round(self._path_length(poly), 2),
                "points_pixel": poly,
                "polyline": poly,
                "source": src,
                "enabled": True,
            })
        return new_id

    def add_manual_edge(self, points: List[Tuple[int, int]], source: str = "manual_repair"):
        """折线补路：完整 polyline 边，首尾吸附节点或 split 已有边。"""
        if len(points) < 2:
            print(f"[DEBUG][Graph] manual edge rejected: only {len(points)} points")
            return None
        start_pt = points[0]
        end_pt = points[-1]

        start_id = self._resolve_endpoint(int(start_pt[0]), int(start_pt[1]))
        end_id = self._resolve_endpoint(int(end_pt[0]), int(end_pt[1]))
        if start_id == end_id:
            print(f"[DEBUG][Graph] manual edge rejected: start==end={start_id}")
            self._manual_edge_points.clear()
            return None

        # 端点坐标对齐到节点
        nm = {n["id"]: n for n in self._nodes}
        pts = [[int(x), int(y)] for x, y in points]
        if start_id in nm:
            pts[0] = [int(nm[start_id]["x"]), int(nm[start_id]["y"])]
        if end_id in nm:
            pts[-1] = [int(nm[end_id]["x"]), int(nm[end_id]["y"])]

        length = round(self._path_length(pts), 2)
        if length < self.min_manual_edge_length:
            print(f"[DEBUG][Graph] manual edge rejected: too short length={length}")
            self._manual_edge_points.clear()
            return None

        eid = self._next_edge_id
        self._next_edge_id += 1
        edge = {
            "id": eid,
            "start": start_id,
            "end": end_id,
            "length_pixel": length,
            "points_pixel": pts,
            "polyline": pts,
            "source": source,
            "enabled": True,
        }
        self._edges.append(edge)
        print(
            f"[DEBUG][Graph] repair polyline edge id={eid} "
            f"start={start_id} end={end_id} length={length} pts={len(pts)}"
        )
        self._manual_edge_points.clear()
        warnings = self.validate_edge(edge)
        self._last_validation_warnings = warnings
        return eid

    def merge_nodes(self, id1: int, id2: int):
        """合并两个节点，并同步相关边 polyline 端点。"""
        nm = {n["id"]: n for n in self._nodes}
        if id1 not in nm or id2 not in nm:
            return None
        n1, n2 = nm[id1], nm[id2]
        new_x = int(round((n1["x"] + n2["x"]) / 2))
        new_y = int(round((n1["y"] + n2["y"]) / 2))
        new_type = "junction" if (n1["type"] == "junction" or n2["type"] == "junction") else n1["type"]

        new_id = self._next_node_id
        self._next_node_id += 1
        self._nodes = [n for n in self._nodes if n["id"] not in (id1, id2)]
        self._nodes.append({
            "id": new_id, "x": new_x, "y": new_y,
            "type": new_type, "source": "manual_repair",
        })

        for e in self._edges:
            pts = polyline_to_list(e.get("points_pixel") or e.get("polyline") or [])
            changed = False
            if e["start"] in (id1, id2):
                e["start"] = new_id
                if pts:
                    pts[0] = [new_x, new_y]
                    changed = True
            if e["end"] in (id1, id2):
                e["end"] = new_id
                if pts:
                    pts[-1] = [new_x, new_y]
                    changed = True
            if changed:
                e["points_pixel"] = pts
                e["polyline"] = pts
                e["length_pixel"] = round(self._path_length(pts), 2)

        # 去重 + 去自环
        seen = {}
        for e in self._edges:
            a, b = sorted([e["start"], e["end"]])
            if a == b:
                continue
            key = (a, b)
            if key not in seen or e.get("length_pixel", 0) > seen[key].get("length_pixel", 0):
                seen[key] = e
        self._edges = list(seen.values())
        print(f"[DEBUG][Graph] merge nodes id={id1} and id={id2} -> id={new_id}")
        return new_id

    def merge_nearby_junctions(self, radius_px: Optional[int] = None) -> int:
        """合并距离过近的路口节点簇，返回合并次数。"""
        radius = float(radius_px if radius_px is not None else self.junction_cluster_radius_px)
        merges = 0
        changed = True
        while changed:
            changed = False
            junctions = [
                n for n in self._nodes
                if n.get("type") in ("junction", "endpoint", "manual") or True
            ]
            # 优先合并 degree>=3 的节点
            degrees = {}
            for e in self._edges:
                if e.get("enabled") is False:
                    continue
                degrees[e["start"]] = degrees.get(e["start"], 0) + 1
                degrees[e["end"]] = degrees.get(e["end"], 0) + 1
            high = [n for n in junctions if degrees.get(n["id"], 0) >= 3]
            pool = high if len(high) >= 2 else junctions
            best = None
            best_d = radius
            for i, a in enumerate(pool):
                for b in pool[i + 1 :]:
                    d = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
                    if d < best_d:
                        best_d = d
                        best = (a["id"], b["id"])
            if best is None:
                break
            self.merge_nodes(best[0], best[1])
            merges += 1
            changed = True
            if merges > 500:
                break
        return merges

    def validate_edge(self, edge: Dict, road_mask: Optional[np.ndarray] = None) -> List[str]:
        """局部边验证（不阻塞编辑）。"""
        warnings: List[str] = []
        pts = polyline_to_list(edge.get("points_pixel") or edge.get("polyline") or [])
        if len(pts) < 2:
            warnings.append(f"edge {edge.get('id')}: edge_geometry_missing")
            return warnings
        length = float(edge.get("length_pixel") or self._path_length(pts))
        chord = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
        if length > 1 and chord > 1 and length / chord > 8.0:
            warnings.append(f"edge {edge.get('id')}: 长度异常(折线过弯或过长)")
        node_ids = {n["id"] for n in self._nodes}
        if edge.get("start") not in node_ids or edge.get("end") not in node_ids:
            warnings.append(f"edge {edge.get('id')}: 端点未连接到 graph")
        if road_mask is not None and road_mask.ndim >= 2:
            h, w = road_mask.shape[:2]
            miss = 0
            total = 0
            step = max(1, len(pts) // 40)
            for i in range(0, len(pts), step):
                x, y = int(pts[i][0]), int(pts[i][1])
                total += 1
                if not (0 <= x < w and 0 <= y < h) or road_mask[y, x] == 0:
                    miss += 1
            if total and miss / total > 0.35:
                warnings.append(f"edge {edge.get('id')}: polyline 偏离 road mask")
        return warnings

    def validate_graph_local(self, road_mask: Optional[np.ndarray] = None) -> List[str]:
        warnings: List[str] = []
        for e in self._edges:
            if e.get("enabled") is False:
                continue
            warnings.extend(self.validate_edge(e, road_mask=road_mask))
        degrees = {}
        for e in self._edges:
            if e.get("enabled") is False:
                continue
            degrees[e["start"]] = degrees.get(e["start"], 0) + 1
            degrees[e["end"]] = degrees.get(e["end"], 0) + 1
        isolated = [n["id"] for n in self._nodes if degrees.get(n["id"], 0) == 0]
        if isolated:
            warnings.append(f"孤立节点: {isolated[:8]}{'...' if len(isolated) > 8 else ''}")
        self._last_validation_warnings = warnings
        return warnings

    def clear_selection(self):
        self._selected_nodes.clear()
        self._selected_edges.clear()

    def select_node(self, node_id: int):
        if node_id not in self._selected_nodes:
            self._selected_nodes.append(node_id)

    def select_edge(self, edge_id: int):
        if edge_id not in self._selected_edges:
            self._selected_edges.append(edge_id)

    def set_dragging(self, node_id: Optional[int]):
        self._dragging_node = node_id

    def set_hover(self, node_id: Optional[int] = None, edge_id: Optional[int] = None):
        self._hovered_node = node_id
        self._hovered_edge = edge_id

    # ------ 手动画边 ------
    def start_manual_edge_mode(self):
        self._manual_edge_points.clear()

    def add_manual_point(self, x: int, y: int):
        self._manual_edge_points.append((x, y))

    def confirm_manual_edge(self):
        if len(self._manual_edge_points) >= 2:
            return self.add_manual_edge(self._manual_edge_points, source="manual_repair")
        self._manual_edge_points.clear()
        return None

    def cancel_manual_edge(self):
        self._manual_edge_points.clear()

    # ------ 撤销 ------
    def _push_undo(self):
        self._undo_stack.push(self._nodes, self._edges)

    def undo(self) -> bool:
        result = self._undo_stack.undo()
        if result:
            self._nodes, self._edges = result
            self.clear_selection()
            self._next_node_id = max((n["id"] for n in self._nodes), default=-1) + 1
            self._next_edge_id = max((e["id"] for e in self._edges), default=-1) + 1
            return True
        return False

    def redo(self) -> bool:
        result = self._undo_stack.redo()
        if result:
            self._nodes, self._edges = result
            self.clear_selection()
            self._next_node_id = max((n["id"] for n in self._nodes), default=-1) + 1
            self._next_edge_id = max((e["id"] for e in self._edges), default=-1) + 1
            return True
        return False

    def can_undo(self) -> bool:
        return self._undo_stack.can_undo()

    def can_redo(self) -> bool:
        return self._undo_stack.can_redo()

    # ------ 从 draft graph 加载 ------
    def load_draft(self, nodes: List[Dict], edges: List[Dict]):
        """从 extract_graph_from_skeleton 的输出加载。

        自动将 numpy 类型转为纯 Python 类型，确保后续 JSON 序列化安全。
        """
        self._nodes = []
        self._edges = []
        for n in nodes:
            self._nodes.append({
                "id": int(n["id"]),
                "x": int(n.get("x", 0)),
                "y": int(n.get("y", 0)),
                "type": str(n.get("type", "junction")),
                "source": str(n.get("source", "auto")),
            })
        for e in edges:
            path = e.get("path", [])
            # ★ 确保 path 是纯 Python list，再转换为 points_pixel 格式 [[x,y], ...]
            # path 原始格式是 [[y,x], ...]，转为 [[x,y], ...]
            cleaned_path = polyline_to_list(path)
            if cleaned_path:
                points = [[int(p[1]), int(p[0])] for p in cleaned_path]
            else:
                points = []
            self._edges.append({
                "id": int(e["id"]),
                "start": int(e.get("from", e.get("start", 0))),
                "end": int(e.get("to", e.get("end", 0))),
                "length_pixel": float(round(self._path_length(points), 2)),
                "points_pixel": points, "source": "auto", "enabled": True,
            })
        self._next_node_id = max((n["id"] for n in self._nodes), default=-1) + 1
        self._next_edge_id = max((e["id"] for e in self._edges), default=-1) + 1
        self._undo_stack = UndoStack()
        self._undo_stack.push(self._nodes, self._edges)

    # ------ 统计 ------
    def get_stats(self) -> Dict:
        auto_edges = sum(1 for e in self._edges if e.get("source") == "auto")
        manual_edges = sum(1 for e in self._edges if e.get("source") == "manual")
        total_len_px = sum(e.get("length_pixel", 0) for e in self._edges)
        components = self._count_components()
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "auto_edge_count": auto_edges,
            "manual_edge_count": manual_edges,
            "components": components,
            "total_length_px": total_len_px,
            "total_length_m": total_len_px * self._pixel_resolution_m,
        }

    def _count_components(self) -> int:
        """计算连通分量数量（BFS）"""
        if not self._nodes:
            return 0
        nids = {n["id"] for n in self._nodes}
        adj = {nid: set() for nid in nids}
        for e in self._edges:
            if e["start"] in adj and e["end"] in adj:
                adj[e["start"]].add(e["end"])
                adj[e["end"]].add(e["start"])
        visited = set()
        components = 0
        for nid in nids:
            if nid not in visited:
                components += 1
                stack = [nid]
                while stack:
                    v = stack.pop()
                    if v in visited:
                        continue
                    visited.add(v)
                    for nb in adj.get(v, set()):
                        if nb not in visited:
                            stack.append(nb)
        return components

    # ------ 保存 ------
    def save(self, output_dir: str, image_rgb: Optional[np.ndarray] = None,
             image_size: Tuple[int, int] = None,
             pixel_resolution_m: float = 0.5, calibrated: bool = False):
        """保存 final_graph 到文件。

        ★ 保存前自动将所有 numpy 类型转为纯 Python 类型，确保 JSON 序列化成功。
        """
        os.makedirs(output_dir, exist_ok=True)

        w, h = image_size or (self._image_w, self._image_h)

        # ---- final_nodes.csv ----
        nodes_path = os.path.join(output_dir, "final_nodes.csv")
        with open(nodes_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["node_id", "x_pixel", "y_pixel", "type", "source"])
            for n in self._nodes:
                writer.writerow([
                    int(n["id"]), int(n["x"]), int(n["y"]),
                    n.get("type", ""), n.get("source", "")
                ])

        # ---- final_edges.csv ----
        edges_path = os.path.join(output_dir, "final_edges.csv")
        with open(edges_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["edge_id", "start_node", "end_node", "length_pixel", "source", "enabled", "point_count"])
            for e in self._edges:
                writer.writerow([
                    int(e["id"]), int(e["start"]), int(e["end"]),
                    float(e.get("length_pixel", 0)),
                    e.get("source", ""), e.get("enabled", True),
                    len(e.get("points_pixel", []))
                ])

        # ---- final_graph.json（★ 确保所有值都是 Python 原生类型）----
        # ★ coordinate_system: "image_pixel" — 所有坐标均为图像像素坐标
        #    需要经纬度时，使用 calibration.json 中的 transform 另行转换。
        #    地理版路网请参见 final_graph_geo.json（由坐标校准模块生成）。
        graph_path = os.path.join(output_dir, "final_graph.json")
        graph = {
            "coordinate_system": "image_pixel",
            "metadata": {
                "image_width": int(w), "image_height": int(h),
                "pixel_resolution_m": float(pixel_resolution_m),
                "coordinate_calibrated": bool(calibrated),
                "node_count": int(len(self._nodes)),
                "edge_count": int(len(self._edges)),
            },
            "nodes": [
                {
                    "id": int(n["id"]),
                    "x_pixel": int(n["x"]),
                    "y_pixel": int(n["y"]),
                    "type": str(n.get("type", "")),
                    "source": str(n.get("source", "auto")),
                }
                for n in self._nodes
            ],
            "edges": [
                {
                    "id": int(e["id"]),
                    "start": int(e["start"]),
                    "end": int(e["end"]),
                    "length_pixel": float(e.get("length_pixel", 0)),
                    "points_pixel": polyline_to_list(e.get("points_pixel", [])),
                    "polyline": polyline_to_list(
                        e.get("polyline") or e.get("points_pixel", [])
                    ),
                    "source": str(e.get("source", "auto")),
                    "enabled": bool(e.get("enabled", True)),
                    **({"invalid": True, "invalid_reason": str(e.get("invalid_reason", ""))}
                       if e.get("invalid") else {}),
                }
                for e in self._edges
            ],
        }
        # ★ 最终一次全量转换，确保没有任何遗漏的 numpy 类型
        graph = ensure_python_types(graph)
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)

        # ---- final_graph_overlay_original.png（原图 + 全局坐标 graph）----
        if image_rgb is not None:
            overlay_original = os.path.join(output_dir, "final_graph_overlay_original.png")
            img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            self._draw_overlay(img, scale=1.0)
            cv2.imwrite(overlay_original, img)

        print(f"[GraphEditor] 已保存: {len(self._nodes)} 节点, {len(self._edges)} 边 → {output_dir}")
        return graph_path

    # ------ 加载 ------

    def load_from_dict(self, data: dict) -> int:
        """从字典加载节点和边数据，返回恢复的边数。

        支持两种格式：
        1. final_graph.json 格式（x_pixel/y_pixel, points_pixel）
        2. 项目内存格式（x/y, points_pixel）

        Returns:
            成功加载的边数量。
        """
        self._nodes.clear()
        self._edges.clear()
        self._next_node_id = 0
        self._next_edge_id = 0

        raw_nodes = data.get("nodes", [])
        raw_edges = data.get("edges", [])
        metadata = data.get("metadata", {})
        pixel_res = metadata.get("pixel_resolution_m", 0.5)
        if pixel_res and pixel_res > 0:
            self._pixel_resolution_m = float(pixel_res)

        # 加载节点（支持 x_pixel/y_pixel 和 x/y 两种字段名）
        max_nid = 0
        for n in raw_nodes:
            nid = int(n["id"])
            x = float(n.get("x_pixel", n.get("x", 0)))
            y = float(n.get("y_pixel", n.get("y", 0)))
            node_type = str(n.get("type", "manual"))
            source = str(n.get("source", "manual"))
            self._nodes.append({
                "id": nid, "x": x, "y": y,
                "type": node_type, "source": source,
            })
            if nid >= max_nid:
                max_nid = nid

        self._next_node_id = max_nid + 1 if raw_nodes else 0
        print(f"[GraphEditor] 已加载 {len(self._nodes)} 个节点, next_node_id={self._next_node_id}")

        # 加载边（支持 points_pixel 和 path 两种字段名）
        max_eid = 0
        for e in raw_edges:
            eid = int(e["id"])
            start = int(e.get("start", e.get("from", 0)))
            end = int(e.get("end", e.get("to", 0)))
            pts = e.get("points_pixel", e.get("path", []))
            # 确保 pts 是 [[x,y],...] 格式
            if pts and isinstance(pts[0], (list, tuple)):
                pts = [[float(p[0]), float(p[1])] for p in pts]
            else:
                pts = []
            # 如果 polyline 为空，用节点坐标补齐
            if not pts or len(pts) < 2:
                src_n = self._find_node_by_id(start)
                dst_n = self._find_node_by_id(end)
                if src_n and dst_n:
                    pts = [[src_n["x"], src_n["y"]], [dst_n["x"], dst_n["y"]]]
            length = round(self._path_length(pts), 2)
            self._edges.append({
                "id": eid, "start": start, "end": end,
                "length_pixel": float(e.get("length_pixel", length)),
                "points_pixel": pts,
                "source": str(e.get("source", "manual")),
                "enabled": bool(e.get("enabled", True)),
            })
            if eid >= max_eid:
                max_eid = eid

        self._next_edge_id = max_eid + 1 if raw_edges else 0
        print(f"[GraphEditor] 已加载 {len(self._edges)} 条边, next_edge_id={self._next_edge_id}")

        self.clear_selection()
        self._edge_start_node = None
        self._manual_edge_points.clear()
        self._dragging_node = None

        return len(self._edges)

    def load_from_file(self, file_path: str) -> bool:
        """从 final_graph.json 文件加载节点和边数据。

        Returns:
            True 表示加载成功，False 表示失败。
        """
        if not os.path.exists(file_path):
            print(f"[GraphEditor] 文件不存在: {file_path}")
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[GraphEditor] 读取文件失败: {e}")
            return False

        if "nodes" not in data or "edges" not in data:
            print(f"[GraphEditor] 文件格式不正确: 缺少 nodes 或 edges 字段")
            return False

        edge_count = self.load_from_dict(data)
        print(f"[GraphEditor] 已从文件加载: {len(self._nodes)} 节点, {edge_count} 边 ← {file_path}")
        return True

    def _find_node_by_id(self, node_id: int) -> Optional[Dict]:
        """按 ID 查找节点"""
        for n in self._nodes:
            if n["id"] == node_id:
                return n
        return None

    def to_dict(self) -> dict:
        """导出为字典（与 save 格式一致）"""
        return {
            "coordinate_system": "image_pixel",
            "metadata": {
                "image_width": int(self._image_w),
                "image_height": int(self._image_h),
                "pixel_resolution_m": float(self._pixel_resolution_m),
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
            },
            "nodes": [
                {
                    "id": int(n["id"]),
                    "x_pixel": int(n["x"]),
                    "y_pixel": int(n["y"]),
                    "type": str(n.get("type", "")),
                    "source": str(n.get("source", "manual")),
                }
                for n in self._nodes
            ],
            "edges": [
                {
                    "id": int(e["id"]),
                    "start": int(e["start"]),
                    "end": int(e["end"]),
                    "length_pixel": float(e.get("length_pixel", 0)),
                    "points_pixel": polyline_to_list(e.get("points_pixel", [])),
                    "source": str(e.get("source", "manual")),
                    "enabled": bool(e.get("enabled", True)),
                }
                for e in self._edges
            ],
        }

    def _draw_overlay(self, img: np.ndarray, scale: float = 1.0):
        """在图像上绘制节点和边（BGR 格式）。

        Args:
            img: 目标图像 (H, W, 3) BGR
            scale: 图坐标到图像像素的缩放比例。
                   1.0 = 直绘全局坐标到原图
                   <1.0 = 大图模式下全局坐标→预览图坐标
        """
        # ---- 边 ----
        for e in self._edges:
            pts = e.get("points_pixel", [])
            color = (255, 184, 108) if e.get("source") == "auto" else (137, 180, 250)
            for i in range(len(pts) - 1):
                p1 = (int(pts[i][0] * scale), int(pts[i][1] * scale))
                p2 = (int(pts[i+1][0] * scale), int(pts[i+1][1] * scale))
                cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)

        # ---- 节点 ----
        for n in self._nodes:
            cx = int(n["x"] * scale)
            cy = int(n["y"] * scale)
            color = (0, 0, 255) if n.get("type") == "junction" else (0, 255, 0)
            cv2.circle(img, (cx, cy), max(6, int(6 * scale)), color, -1)
            if n.get("type") == "junction":
                cv2.circle(img, (cx, cy), max(6, int(6 * scale)), (0, 0, 180), 2)

        # ---- 节点编号 ----
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.25, 0.4 * scale) if scale < 1.0 else 0.4
        for n in self._nodes:
            cx = int(n["x"] * scale)
            cy = int(n["y"] * scale)
            cv2.putText(img, str(n["id"]), (cx + 10, cy - 6),
                        font, font_scale, (255, 255, 0), 1, cv2.LINE_AA)

    def save_overlay_preview(self, preview_rgb: np.ndarray, output_dir: str,
                              preview_scale: float = 1.0):
        """保存预览图 overlay（global→preview 坐标转换）。

        Args:
            preview_rgb: 预览图 (H, W, 3) RGB
            output_dir: 输出目录
            preview_scale: 预览缩放比例
        """
        os.makedirs(output_dir, exist_ok=True)
        overlay_path = os.path.join(output_dir, "final_graph_overlay_preview.png")
        img = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
        self._draw_overlay(img, scale=preview_scale)
        cv2.imwrite(overlay_path, img)
        print(f"[GraphEditor] 预览叠加图已保存: {overlay_path} (scale={preview_scale:.4f})")
        return overlay_path

    @staticmethod
    def _path_length(points) -> float:
        """安全计算 polyline 总长度（兼容 list 和 numpy array 输入）。"""
        pts = polyline_to_list(points)
        if len(pts) < 2:
            return 0.0
        total = 0.0
        for i in range(len(pts) - 1):
            dx = pts[i + 1][0] - pts[i][0]
            dy = pts[i + 1][1] - pts[i][1]
            total += math.sqrt(dx * dx + dy * dy)
        return total
