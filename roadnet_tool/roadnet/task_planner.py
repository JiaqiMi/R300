"""
任务点与路径规划接口模块

功能：
- 添加任务点（像素坐标 / 经纬度）
- 任务点吸附到最近道路边
- 路径规划（当前为预留接口，V1 使用 Dijkstra 简单实现）
- 路径导出 CSV

后续完整实现可集成 networkx、osmnx 等图算法库。
"""

from __future__ import annotations

import csv
import math
import os
from typing import List, Dict, Tuple, Optional

import numpy as np


class TaskPoint:
    """任务点数据结构"""
    def __init__(self, task_id: str, x_pixel: float = 0, y_pixel: float = 0,
                 lon: Optional[float] = None, lat: Optional[float] = None,
                 tp_type: str = "task"):
        self.id = task_id
        self.x_pixel = x_pixel
        self.y_pixel = y_pixel
        self.lon = lon
        self.lat = lat
        self.type = tp_type  # "start" / "goal" / "task"
        self.snapped_edge_id: Optional[int] = None
        self.snapped_x: float = x_pixel
        self.snapped_y: float = y_pixel
        self.snapped_dist_px: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "x_pixel": self.x_pixel,
            "y_pixel": self.y_pixel,
            "lon": self.lon,
            "lat": self.lat,
            "type": self.type,
            "snapped_edge_id": self.snapped_edge_id,
            "snapped_x": self.snapped_x,
            "snapped_y": self.snapped_y,
        }


class TaskPlanner:
    """任务点管理与路径规划器（V1 预留接口）"""

    def __init__(self, pixel_resolution_m: float = 0.5):
        self._task_points: List[TaskPoint] = []
        self._pixel_resolution_m = pixel_resolution_m
        self._snap_max_distance_m: float = 10.0  # 默认吸附阈值 10m
        self._planned_path: List[List[float]] = []  # [[x,y], ...]

    @property
    def task_points(self) -> List[TaskPoint]:
        return self._task_points

    @property
    def planned_path(self) -> List[List[float]]:
        return self._planned_path

    @property
    def snap_max_distance_m(self) -> float:
        return self._snap_max_distance_m

    @snap_max_distance_m.setter
    def snap_max_distance_m(self, val: float):
        self._snap_max_distance_m = val

    # ------ 添加任务点 ------
    def add_task_point_lonlat(self, task_id: str, lon: float, lat: float,
                               tp_type: str = "task") -> TaskPoint:
        """通过经纬度添加任务点（需要后续用校准器转像素坐标）"""
        tp = TaskPoint(task_id=task_id, lon=lon, lat=lat, tp_type=tp_type)
        self._task_points.append(tp)
        return tp

    def add_task_point_pixel(self, task_id: str, x_pixel: float, y_pixel: float,
                              tp_type: str = "task") -> TaskPoint:
        """通过像素坐标添加任务点"""
        tp = TaskPoint(task_id=task_id, x_pixel=x_pixel, y_pixel=y_pixel, tp_type=tp_type)
        tp.snapped_x = x_pixel
        tp.snapped_y = y_pixel
        self._task_points.append(tp)
        return tp

    def add_task_point(self, task_id: str, x_pixel: float, y_pixel: float,
                        lon: Optional[float] = None, lat: Optional[float] = None,
                        tp_type: str = "task") -> TaskPoint:
        """通用添加"""
        tp = TaskPoint(task_id=task_id, x_pixel=x_pixel, y_pixel=y_pixel,
                        lon=lon, lat=lat, tp_type=tp_type)
        tp.snapped_x = x_pixel
        tp.snapped_y = y_pixel
        self._task_points.append(tp)
        return tp

    def clear_task_points(self):
        self._task_points.clear()
        self._planned_path.clear()

    # ------ 吸附到路网 ------
    def snap_task_point_to_graph(self, tp: TaskPoint,
                                   nodes: List[Dict], edges: List[Dict]) -> bool:
        """
        将任务点吸附到最近的道路边上。

        Args:
            tp: 任务点
            nodes: 节点列表 [{"id":int, "x":float, "y":float}, ...]
            edges: 边列表 [{"id":int, "points_pixel":[[x,y],...]}, ...]

        Returns:
            是否成功吸附（距离在阈值内）
        """
        snap_dist_px = self._snap_max_distance_m / max(self._pixel_resolution_m, 1e-6)
        best_dist = snap_dist_px
        best_edge = None
        best_px, best_py = tp.x_pixel, tp.y_pixel

        for e in edges:
            pts = e.get("points_pixel", [])
            for i in range(len(pts) - 1):
                px, py, d = self._project_to_segment(
                    tp.x_pixel, tp.y_pixel,
                    pts[i][0], pts[i][1],
                    pts[i+1][0], pts[i+1][1],
                )
                if d < best_dist:
                    best_dist = d
                    best_edge = e["id"]
                    best_px, best_py = px, py

        if best_edge is not None:
            tp.snapped_edge_id = best_edge
            tp.snapped_x = best_px
            tp.snapped_y = best_py
            tp.snapped_dist_px = best_dist
            return True
        else:
            print(f"[TaskPlanner] 警告: 任务点 {tp.id} 距离道路 > {self._snap_max_distance_m}m，需手动处理")
            return False

    @staticmethod
    def _project_to_segment(px, py, x1, y1, x2, y2) -> Tuple[float, float, float]:
        """将点投影到线段上，返回 (proj_x, proj_y, distance)"""
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            d = math.sqrt((px - x1)**2 + (py - y1)**2)
            return x1, y1, d
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)))
        cx, cy = x1 + t * dx, y1 + t * dy
        d = math.sqrt((px - cx)**2 + (py - cy)**2)
        return cx, cy, d

    # ------ 路径规划（V1 Dijkstra） ------
    def plan_path(self, nodes: List[Dict], edges: List[Dict],
                   start: TaskPoint, goal: TaskPoint,
                   via_tasks: Optional[List[TaskPoint]] = None) -> bool:
        """
        简单 Dijkstra 路径规划。

        按 start → tasks → goal 顺序（order_fixed=True）分段规划。

        Returns:
            是否成功规划路径
        """
        via = via_tasks or []
        checkpoints = [start] + via + [goal]
        self._planned_path = []

        for i in range(len(checkpoints) - 1):
            src = checkpoints[i]
            dst = checkpoints[i + 1]

            # 将任务点吸附位置添加到图作为临时节点
            temp_node_ids = []
            all_nodes = list(nodes)
            all_edges = list(edges)

            # 为起点和终点创建临时节点并连接到最近边
            for tp in [src, dst]:
                snap_edge_id = tp.snapped_edge_id
                nid = -1 - len(temp_node_ids)  # 负 ID
                temp_node_ids.append(nid)
                all_nodes.append({"id": nid, "x": tp.snapped_x, "y": tp.snapped_y})

                if snap_edge_id is not None:
                    edge = next((e for e in all_edges if e["id"] == snap_edge_id), None)
                    if edge:
                        # 拆边：在原边上插入临时节点
                        all_edges = [e for e in all_edges if e["id"] != snap_edge_id]
                        pts = edge.get("points_pixel", [])
                        # 找到最近段
                        best_idx = 0
                        best_d = float("inf")
                        for j in range(len(pts) - 1):
                            _, _, d = self._project_to_segment(
                                tp.snapped_x, tp.snapped_y,
                                pts[j][0], pts[j][1], pts[j+1][0], pts[j+1][1])
                            if d < best_d:
                                best_d = d
                                best_idx = j
                        # 拆成两条边
                        pts1 = pts[:best_idx + 1] + [[tp.snapped_x, tp.snapped_y]]
                        pts2 = [[tp.snapped_x, tp.snapped_y]] + pts[best_idx + 1:]
                        len1 = self._path_len(pts1)
                        len2 = self._path_len(pts2)
                        eid1 = -1 - len(all_edges)
                        eid2 = -2 - len(all_edges)
                        all_edges.append({
                            "id": eid1, "start": edge["start"], "end": nid,
                            "length_pixel": round(len1, 2), "points_pixel": pts1,
                        })
                        all_edges.append({
                            "id": eid2, "start": nid, "end": edge["end"],
                            "length_pixel": round(len2, 2), "points_pixel": pts2,
                        })

            # 运行 Dijkstra
            path_ids = self._dijkstra(all_nodes, all_edges,
                                       temp_node_ids[0], temp_node_ids[1])
            if path_ids is None:
                print(f"[TaskPlanner] 无法从 {src.id} 到达 {dst.id}")
                return False

            # 提取路径坐标
            id_to_pos = {n["id"]: (n["x"], n["y"]) for n in all_nodes}
            for nid in path_ids:
                pos = id_to_pos.get(nid)
                if pos:
                    self._planned_path.append(list(pos))

        return len(self._planned_path) > 0

    def _dijkstra(self, nodes: List[Dict], edges: List[Dict],
                   start_id: int, goal_id: int) -> Optional[List[int]]:
        """Dijkstra 最短路径"""
        nids = {n["id"] for n in nodes}
        if start_id not in nids or goal_id not in nids:
            return None

        adj = {nid: [] for nid in nids}
        for e in edges:
            s, t = e["start"], e["end"]
            w = e.get("length_pixel", 1)
            if s in adj and t in adj:
                adj[s].append((t, w))
                adj[t].append((s, w))

        import heapq
        dist = {nid: float("inf") for nid in nids}
        prev = {nid: None for nid in nids}
        dist[start_id] = 0
        pq = [(0, start_id)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == goal_id:
                break
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if dist[goal_id] == float("inf"):
            return None

        # 回溯路径
        path = []
        u = goal_id
        while u is not None:
            path.append(u)
            u = prev[u]
        path.reverse()
        return path

    @staticmethod
    def _path_len(points: List[List]) -> float:
        total = 0.0
        for i in range(len(points) - 1):
            dx = points[i+1][0] - points[i][0]
            dy = points[i+1][1] - points[i][1]
            total += math.sqrt(dx*dx + dy*dy)
        return total

    # ------ 导出 ------
    def export_path_csv(self, path: List[List[float]], output_path: str,
                         pixel_resolution_m: float = 0.5) -> str:
        """导出路径点 CSV"""
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["point_id", "x_pixel", "y_pixel", "distance_pixel", "distance_m"])
            cum_dist = 0.0
            for i, pt in enumerate(path):
                if i > 0:
                    dx = pt[0] - path[i-1][0]
                    dy = pt[1] - path[i-1][1]
                    cum_dist += math.sqrt(dx*dx + dy*dy)
                writer.writerow([i, round(pt[0], 2), round(pt[1], 2),
                                  round(cum_dist, 2),
                                  round(cum_dist * pixel_resolution_m, 2)])
        print(f"[TaskPlanner] 路径已导出: {output_path} ({len(path)} 个点, {cum_dist:.1f}px)")
        return output_path
