"""
任务点吸附模块

核心策略：优先吸附到最近 graph edge（折线投影），
而不是最近 node。支持多候选、扩大半径保底、skeleton/mask 保底。
"""

from __future__ import annotations

import math
import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np


# ===================================================================
# 数据类
# ===================================================================

@dataclass
class SnapConfig:
    """吸附配置"""
    max_snap_distance_px: float = 250.0        # 超过则 failed，禁止正式导出
    warning_distance_px: float = 50.0          # 超过则 warning
    search_radius_px: float = 150.0             # 搜索半径
    top_k: int = 5                              # 候选数量
    prefer_edge: bool = True                    # 优先吸附到 edge
    allow_virtual_node: bool = True             # 允许在 edge 中间插入虚拟节点
    allow_disabled_edges: bool = False          # 是否允许吸附到已禁用的边
    node_proximity_px: float = 15.0             # 若投影点距离 node 很近，直接吸附到 node
    expand_radius_on_fail: bool = True          # 失败时扩大半径重试
    expanded_radius_px: float = 250.0           # 扩大后的半径
    prefer_larger_component: bool = True        # 优先选择更大连通分量的候选
    use_skeleton_fallback: bool = True          # skeleton 保底
    use_mask_fallback: bool = True              # mask 保底（预留）


@dataclass
class SnapCandidate:
    """候选吸附点"""
    snapped_x: float
    snapped_y: float
    distance: float
    edge_id: Optional[int] = None
    node_id: Optional[int] = None
    segment_index: int = -1
    t: float = 0.0               # 投影在线段上的比例 (0~1)
    edge_enabled: bool = True
    component_size: int = 0

    @property
    def is_on_node(self) -> bool:
        """投影点是否非常接近已有 node"""
        return self.t < 0.01 or self.t > 0.99

    @property
    def score(self) -> float:
        """综合评分（越小越好）：距离 + 惩罚项"""
        s = self.distance
        if not self.edge_enabled:
            s += 1000  # 禁用的边大幅惩罚
        if self.component_size > 0:
            s -= min(self.component_size * 5, 100)  # 大连通分量奖励
        return s


@dataclass
class SnappedTaskPoint:
    """吸附后的任务点"""
    seq: int
    point_type: int
    original_x: float
    original_y: float
    snapped_x: float
    snapped_y: float
    snap_distance: float
    edge_id: Optional[int] = None
    node_id: Optional[int] = None
    virtual_node_id: Optional[str] = None
    snap_method: str = "none"
    status: str = "pending"
    warning: str = ""
    candidates: List[SnapCandidate] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "point_type": self.point_type,
            "original": [round(self.original_x, 2), round(self.original_y, 2)],
            "snapped": [round(self.snapped_x, 2), round(self.snapped_y, 2)],
            "snap_distance": round(self.snap_distance, 2),
            "edge_id": self.edge_id,
            "node_id": self.node_id,
            "virtual_node_id": self.virtual_node_id,
            "snap_method": self.snap_method,
            "status": self.status,
            "warning": self.warning,
        }


# ===================================================================
# 核心投影算法：点到 polyline 最近投影
# ===================================================================

def project_point_to_polyline(
    px: float, py: float, polyline: List[List[float]]
) -> Tuple[float, float, float, int, float]:
    """将点投影到折线上，返回最近投影点。

    Args:
        px, py: 待投影点
        polyline: [[x1,y1], [x2,y2], ...] 折线顶点列表

    Returns:
        (sx, sy, distance, segment_index, t)
        - sx, sy: 投影点坐标
        - distance: 点到投影点距离
        - segment_index: 投影在哪一段 (0 ~ len-2)
        - t: 该段上的比例 (0~1)，0=段起点, 1=段终点
    """
    if len(polyline) < 1:
        return px, py, float("inf"), -1, 0.0
    if len(polyline) == 1:
        dx = px - polyline[0][0]
        dy = py - polyline[0][1]
        return polyline[0][0], polyline[0][1], math.sqrt(dx*dx + dy*dy), 0, 0.0

    best_dist = float("inf")
    best_sx, best_sy = px, py
    best_seg = 0
    best_t = 0.0

    for i in range(len(polyline) - 1):
        x1, y1 = polyline[i][0], polyline[i][1]
        x2, y2 = polyline[i+1][0], polyline[i+1][1]
        dx, dy = x2 - x1, y2 - y1
        seg_len2 = dx*dx + dy*dy

        if seg_len2 < 1e-12:
            d = math.sqrt((px - x1)**2 + (py - y1)**2)
            if d < best_dist:
                best_dist = d
                best_sx, best_sy = x1, y1
                best_seg = i
                best_t = 0.0
            continue

        t = ((px - x1) * dx + (py - y1) * dy) / seg_len2
        t = max(0.0, min(1.0, t))
        sx = x1 + t * dx
        sy = y1 + t * dy
        d = math.sqrt((px - sx)**2 + (py - sy)**2)

        if d < best_dist:
            best_dist = d
            best_sx, best_sy = sx, sy
            best_seg = i
            best_t = t

    return best_sx, best_sy, best_dist, best_seg, best_t


# ===================================================================
# 图预处理：计算连通分量大小
# ===================================================================

def _node_x(n: Dict) -> float:
    """获取节点 x 坐标，兼容 x / x_pixel 字段名"""
    return n.get("x", n.get("x_pixel", 0.0))

def _node_y(n: Dict) -> float:
    """获取节点 y 坐标，兼容 y / y_pixel 字段名"""
    return n.get("y", n.get("y_pixel", 0.0))

def _compute_component_sizes(nodes: List[Dict], edges: List[Dict]) -> Dict[int, int]:
    """计算每个边所属连通分量的大小（边数）。"""
    # 构建 node → edges indice 映射
    node_to_edges: Dict[int, List[int]] = {}
    for i, e in enumerate(edges):
        s, t = e["start"], e["end"]
        node_to_edges.setdefault(s, []).append(i)
        node_to_edges.setdefault(t, []).append(i)

    visited = set()
    component_size: Dict[int, int] = {}

    for e_idx in range(len(edges)):
        if e_idx in visited:
            continue
        # BFS
        component_edges = set()
        queue = [e_idx]
        visited.add(e_idx)
        while queue:
            cur = queue.pop(0)
            component_edges.add(cur)
            e = edges[cur]
            for nid in (e["start"], e["end"]):
                for nei in node_to_edges.get(nid, []):
                    if nei not in visited:
                        visited.add(nei)
                        queue.append(nei)

        size = len(component_edges)
        for ce in component_edges:
            component_size[ce] = size

    return component_size


# ===================================================================
# 候选生成
# ===================================================================

def get_snap_candidates(
    px: float, py: float,
    nodes: List[Dict], edges: List[Dict],
    config: SnapConfig,
    component_sizes: Optional[Dict[int, int]] = None,
) -> List[SnapCandidate]:
    """为一个任务点生成 top_k 个候选吸附点。

    排序优先级：距离近 > edge enabled > 大连通分量
    """
    if component_sizes is None:
        component_sizes = _compute_component_sizes(nodes, edges)

    candidates: List[SnapCandidate] = []

    for e_idx, e in enumerate(edges):
        enabled = e.get("enabled", True)
        if not config.allow_disabled_edges and not enabled:
            continue

        pts = e.get("points_pixel", [])
        if len(pts) < 2:
            # 边没有 polyline，退化为端点到端点
            s_node = next((n for n in nodes if n["id"] == e["start"]), None)
            e_node = next((n for n in nodes if n["id"] == e["end"]), None)
            if s_node and e_node:
                pts = [[_node_x(s_node), _node_y(s_node)], [_node_x(e_node), _node_y(e_node)]]
            else:
                continue

        sx, sy, dist, seg_idx, t = project_point_to_polyline(px, py, pts)

        if dist > config.search_radius_px:
            continue

        edge_id = e.get("id") if "id" in e else e_idx
        comp_size = component_sizes.get(e_idx if "id" not in e else e.get("id", e_idx), 0)

        cand = SnapCandidate(
            snapped_x=sx,
            snapped_y=sy,
            distance=dist,
            edge_id=edge_id,
            segment_index=seg_idx,
            t=t,
            edge_enabled=enabled,
            component_size=comp_size,
        )

        # 检查是否靠近已有 node
        for nid in (e["start"], e["end"]):
            nd = next((n for n in nodes if n["id"] == nid), None)
            if nd:
                nx, ny = _node_x(nd), _node_y(nd)
                nd_dist = math.sqrt((sx - nx)**2 + (sy - ny)**2)
                if nd_dist < config.node_proximity_px:
                    cand.node_id = nid
                    cand.snapped_x = nx
                    cand.snapped_y = ny
                    cand.distance = math.sqrt((px - nx)**2 + (py - ny)**2)
                    break

        candidates.append(cand)

    # 排序：按综合评分
    candidates.sort(key=lambda c: c.score)

    return candidates[:config.top_k]


# ===================================================================
# 主吸附函数
# ===================================================================

def snap_task_point_to_graph(
    px: float, py: float,
    seq: int, point_type: int,
    nodes: List[Dict], edges: List[Dict],
    config: SnapConfig,
    component_sizes: Optional[Dict[int, int]] = None,
) -> SnappedTaskPoint:
    """将单个任务点吸附到路网图。

    Args:
        px, py: 任务点像素坐标
        seq: 任务点序号
        point_type: 点类型 (0=start, 1=goal, 2=via)
        nodes: 节点列表
        edges: 边列表
        config: 吸附配置

    Returns:
        SnappedTaskPoint 吸附结果
    """
    if component_sizes is None:
        component_sizes = _compute_component_sizes(nodes, edges)

    # 第一轮：默认半径
    candidates = get_snap_candidates(px, py, nodes, edges, config, component_sizes)

    # 第二轮：扩大半径
    if len(candidates) == 0 and config.expand_radius_on_fail:
        expanded_config = SnapConfig(
            max_snap_distance_px=config.expanded_radius_px,
            search_radius_px=config.expanded_radius_px,
            top_k=config.top_k,
            prefer_edge=config.prefer_edge,
            allow_virtual_node=config.allow_virtual_node,
            allow_disabled_edges=config.allow_disabled_edges,
            node_proximity_px=config.node_proximity_px,
            expand_radius_on_fail=False,
        )
        candidates = get_snap_candidates(
            px, py, nodes, edges, expanded_config, component_sizes
        )

    if len(candidates) == 0:
        return SnappedTaskPoint(
            seq=seq, point_type=point_type,
            original_x=px, original_y=py,
            snapped_x=px, snapped_y=py,
            snap_distance=float("inf"),
            snap_method="none",
            status="failed",
            warning=f"未找到 {config.search_radius_px}px 内的道路边",
        )

    best = candidates[0]

    # 确定吸附方法
    if best.node_id is not None:
        method = "node"
    else:
        method = "edge_projection"

    # ★ 每个任务点必须有唯一 virtual_node_id，即使吸附到了已有 node
    # 这样 path_node_sequence 中可以可靠地包含 task_N_virtual 标记。
    vnid = f"task_{seq}_virtual" if config.allow_virtual_node else None

    # 状态判断：warning_distance → warning；max_snap → failed（禁止正式导出）
    status = "ok"
    warning = ""
    warn_d = float(getattr(config, "warning_distance_px", 50.0) or 50.0)
    max_d = float(config.max_snap_distance_px)
    if best.distance > max_d:
        status = "failed"
        warning = f"吸附距离 {best.distance:.1f}px 超过最大阈值 {max_d:.1f}px"
    elif best.distance > warn_d:
        status = "warning"
        warning = f"吸附距离 {best.distance:.1f}px 超过警告阈值 {warn_d:.1f}px"

    return SnappedTaskPoint(
        seq=seq, point_type=point_type,
        original_x=px, original_y=py,
        snapped_x=best.snapped_x, snapped_y=best.snapped_y,
        snap_distance=best.distance,
        edge_id=best.edge_id,
        node_id=best.node_id,
        virtual_node_id=vnid,
        snap_method=method,
        status=status,
        warning=warning,
        candidates=candidates,
    )


def snap_all_task_points(
    task_points: List,  # TaskPoint list
    nodes: List[Dict], edges: List[Dict],
    config: SnapConfig,
    component_sizes: Optional[Dict[int, int]] = None,
) -> List[SnappedTaskPoint]:
    """批量吸附所有任务点。

    Args:
        task_points: TaskPoint 列表
        nodes: 路网节点
        edges: 路网边
        config: 吸附配置

    Returns:
        吸附结果列表
    """
    if component_sizes is None:
        component_sizes = _compute_component_sizes(nodes, edges)

    results = []
    for tp in task_points:
        px = tp.pixel_x
        py = tp.pixel_y
        if px is None or py is None:
            results.append(SnappedTaskPoint(
                seq=tp.seq, point_type=tp.point_type,
                original_x=0, original_y=0,
                snapped_x=0, snapped_y=0,
                snap_distance=float("inf"),
                snap_method="none",
                status="failed",
                warning="任务点像素坐标未知，请先完成坐标标定",
            ))
            continue

        result = snap_task_point_to_graph(
            px, py, tp.seq, tp.point_type,
            nodes, edges, config, component_sizes,
        )
        results.append(result)

    return results


# ===================================================================
# Skeleton 保底
# ===================================================================

def snap_task_point_to_skeleton(
    px: float, py: float,
    skeleton: np.ndarray,
    max_distance_px: float = 120.0,
) -> Optional[Tuple[float, float, float]]:
    """在 skeleton 图像上查找最近非零点。

    Args:
        px, py: 像素坐标
        skeleton: 骨架图像 (0/1 或 0/255)
        max_distance_px: 最大搜索距离

    Returns:
        (sx, sy, distance) 或 None
    """
    if skeleton is None or skeleton.size == 0:
        return None

    h, w = skeleton.shape
    px_i, py_i = int(round(px)), int(round(py))

    if px_i < 0 or px_i >= w or py_i < 0 or py_i >= h:
        return None

    search_r = int(math.ceil(max_distance_px))
    min_x = max(0, px_i - search_r)
    max_x = min(w, px_i + search_r + 1)
    min_y = max(0, py_i - search_r)
    max_y = min(h, py_i + search_r + 1)

    region = skeleton[min_y:max_y, min_x:max_x]
    ys, xs = np.where(region > 0)

    if len(ys) == 0:
        return None

    # 找最近点
    dx = xs + min_x - px
    dy = ys + min_y - py
    dists = np.sqrt(dx**2 + dy**2)
    min_idx = np.argmin(dists)

    if dists[min_idx] > max_distance_px:
        return None

    return (
        float(xs[min_idx] + min_x),
        float(ys[min_idx] + min_y),
        float(dists[min_idx]),
    )


def snap_task_point_to_mask(
    px: float, py: float,
    mask: np.ndarray,
    max_distance_px: float = 120.0,
) -> Optional[Tuple[float, float, float]]:
    """在 mask 图像上查找最近的道路像素。

    与 skeleton 保底逻辑相同，但在整个 mask 上搜索。
    """
    return snap_task_point_to_skeleton(px, py, mask, max_distance_px)


# ===================================================================
# 虚拟节点插入到图副本
# ===================================================================

def insert_virtual_nodes(
    nodes: List[Dict], edges: List[Dict],
    snapped_points: List[SnappedTaskPoint],
) -> Tuple[List[Dict], List[Dict]]:
    """在图副本中插入虚拟节点，不修改原图。

    对于一个 edge 上多个虚拟节点的情况，按 polyline 方向依次切开。

    Args:
        nodes: 原始节点列表
        edges: 原始边列表
        snapped_points: 吸附结果列表

    Returns:
        (new_nodes, new_edges): 插入了虚拟节点的图副本
    """
    new_nodes = [dict(n) for n in nodes]
    new_edges = []

    # 找出需要拆分的边和对应的虚拟节点
    # edge_id → [(snapped_point, projected_x, projected_y, seg_idx)]
    edge_splits: Dict[int, List[Tuple[SnappedTaskPoint, float, float, int]]] = {}

    for sp in snapped_points:
        if sp.status == "failed":
            continue
        if sp.virtual_node_id is None:
            continue

        # ★ 所有任务点都创建 virtual node
        # - edge_projection: 拆边插入 virtual node
        # - node-snapped: 创建 virtual node 并用零长边连接到 graph node
        if sp.edge_id is not None and sp.node_id is None:
            eid = sp.edge_id
            edge_splits.setdefault(eid, []).append(
                (sp, sp.snapped_x, sp.snapped_y, -1)
            )

    for e in edges:
        eid = e.get("id")
        if eid not in edge_splits or len(edge_splits[eid]) == 0:
            new_edges.append(dict(e))
            continue

        pts = e.get("points_pixel", [])
        splits = edge_splits[eid]

        # 对每个切割点，找到在 polyline 上的最近投影位置
        split_info = []  # (seg_idx, t, snapped_point)
        for sp, sx, sy, _ in splits:
            _, _, _, seg_idx, t = project_point_to_polyline(sx, sy, pts)
            split_info.append((seg_idx, t, sp))

        # 按 polyline 方向排序
        split_info.sort(key=lambda x: (x[0], x[1]))

        # 分段切割
        current_start_idx = 0
        prev_id = e["start"]
        last_cut_xy = None

        for split_i, (seg_idx, t, sp) in enumerate(split_info):
            vnid = sp.virtual_node_id
            new_nodes.append({
                "id": vnid,
                "x": sp.snapped_x,
                "y": sp.snapped_y,
                "type": "virtual",
                "source": "task_snap",
            })

            # 构建前半段 polyline：从上一切点/起点 → 当前 virtual
            seg_pts = []
            if last_cut_xy is not None:
                seg_pts.append([float(last_cut_xy[0]), float(last_cut_xy[1])])
                start_i = current_start_idx + 1
            else:
                start_i = 0
            for i in range(start_i, seg_idx + 1):
                if 0 <= i < len(pts):
                    seg_pts.append(list(pts[i]))
            seg_pts.append([float(sp.snapped_x), float(sp.snapped_y)])
            # 去重相邻重复点
            cleaned = [seg_pts[0]]
            for pt in seg_pts[1:]:
                if abs(cleaned[-1][0] - pt[0]) > 1e-6 or abs(cleaned[-1][1] - pt[1]) > 1e-6:
                    cleaned.append(pt)
            seg_pts = cleaned
            if len(seg_pts) < 2:
                seg_pts = [
                    [float(pts[0][0]), float(pts[0][1])] if last_cut_xy is None else [float(last_cut_xy[0]), float(last_cut_xy[1])],
                    [float(sp.snapped_x), float(sp.snapped_y)],
                ]

            new_edge_id = -1 - len(new_edges)
            new_edges.append({
                "id": new_edge_id,
                "start": prev_id,
                "end": vnid,
                "length_pixel": round(_polyline_length(seg_pts), 2),
                "points_pixel": seg_pts,
                "polyline": seg_pts,
                "source": "task_split",
                "enabled": True,
                "parent_edge_id": eid,
            })

            prev_id = vnid
            current_start_idx = seg_idx
            last_cut_xy = [float(sp.snapped_x), float(sp.snapped_y)]

        # 最后一段：从最后一个切割点到 e["end"]
        if last_cut_xy is not None:
            seg_pts = [[float(last_cut_xy[0]), float(last_cut_xy[1])]]
            for i in range(current_start_idx + 1, len(pts)):
                seg_pts.append(list(pts[i]))
            # ensure end node coordinate
            end_node = next((n for n in new_nodes if n.get("id") == e["end"]), None)
            if end_node is not None:
                ex = float(end_node.get("x", end_node.get("x_pixel", pts[-1][0])))
                ey = float(end_node.get("y", end_node.get("y_pixel", pts[-1][1])))
                if abs(seg_pts[-1][0] - ex) > 1e-3 or abs(seg_pts[-1][1] - ey) > 1e-3:
                    seg_pts.append([ex, ey])
            cleaned = [seg_pts[0]]
            for pt in seg_pts[1:]:
                if abs(cleaned[-1][0] - pt[0]) > 1e-6 or abs(cleaned[-1][1] - pt[1]) > 1e-6:
                    cleaned.append(pt)
            seg_pts = cleaned
            if len(seg_pts) >= 2:
                new_edge_id = -1 - len(new_edges)
                new_edges.append({
                    "id": new_edge_id,
                    "start": prev_id,
                    "end": e["end"],
                    "length_pixel": round(_polyline_length(seg_pts), 2),
                    "points_pixel": seg_pts,
                    "polyline": seg_pts,
                    "source": "task_split",
                    "enabled": True,
                    "parent_edge_id": eid,
                })

    # ★ 为 node-snapped 的任务点创建 virtual node（零长边连接）
    for sp in snapped_points:
        if sp.status == "failed":
            continue
        if sp.virtual_node_id is None:
            continue
        if sp.node_id is None:
            continue  # edge_projection 已由上面处理

        # node-snapped: 创建 virtual node 并通过零长边连接到 graph node
        vnid = sp.virtual_node_id
        new_nodes.append({
            "id": vnid,
            "x": sp.snapped_x,
            "y": sp.snapped_y,
            "type": "virtual",
            "source": "node_snap",
        })
        # 双向零长边：graph node ↔ virtual node
        new_edge_id = -1 - len(new_edges)
        new_edges.append({
            "id": new_edge_id,
            "start": sp.node_id,
            "end": vnid,
            "length_pixel": 0.001,  # 极小值，避免除零
            "points_pixel": [[sp.snapped_x, sp.snapped_y], [sp.snapped_x, sp.snapped_y]],
            "source": "node_snap_virtual",
            "enabled": True,
        })

    return new_nodes, new_edges


def _polyline_length(pts: List[List[float]]) -> float:
    """计算折线总长度"""
    total = 0.0
    for i in range(len(pts) - 1):
        dx = pts[i+1][0] - pts[i][0]
        dy = pts[i+1][1] - pts[i][1]
        total += math.sqrt(dx*dx + dy*dy)
    return total
