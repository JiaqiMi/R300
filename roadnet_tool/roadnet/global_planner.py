"""
全局路径规划模块

支持 A* 和 Dijkstra 分段规划：起点 → 必经点1 → ... → 终点。
路径展开为连续 polyline，含重采样和 yaw 计算。
"""

from __future__ import annotations

import heapq
import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ===================================================================
# 数据类
# ===================================================================

@dataclass
class PlannerConfig:
    """规划配置"""
    algorithm: str = "astar"          # "astar" / "dijkstra"
    pixel_resolution_m: float = 0.5
    resample_spacing_px: float = 20.0  # 路径点重采样间距
    turn_penalty_factor: float = 0.0   # 转弯惩罚（预留）
    confidence_penalty: float = 0.0    # 置信度惩罚（预留）


@dataclass
class SegmentPlanResult:
    """单段规划结果"""
    from_seq: int
    to_seq: int
    from_name: str                    # 起始点类型名
    to_name: str                      # 终点类型名
    status: str                       # "ok" / "unreachable"
    length_px: float
    algorithm: str
    node_path: List                   # 节点 ID 列表
    edge_path: List                   # 边 ID 列表
    path_points: List[List[float]]    # [[x,y], ...] 展开的连续 polyline
    error: str = ""
    from_virtual_node: str = ""       # ★ 段起点的 task virtual node ID
    to_virtual_node: str = ""         # ★ 段终点的 task virtual node ID
    unexpected_task_virtual_nodes: List[str] = field(default_factory=list)  # ★ 意外出现的其他 task virtual node


@dataclass
class GlobalPlanResult:
    """全局规划结果"""
    coordinate_type: str = "pixel"
    task_sequence: List[int] = field(default_factory=list)
    segments: List[SegmentPlanResult] = field(default_factory=list)
    global_path_points: List[List[float]] = field(default_factory=list)
    total_length_px: float = 0.0
    success: bool = False

    def to_dict(self) -> dict:
        return {
            "coordinate_type": self.coordinate_type,
            "task_sequence": self.task_sequence,
            "total_length_px": round(self.total_length_px, 2),
            "segments": [
                {
                    "from_seq": seg.from_seq,
                    "to_seq": seg.to_seq,
                    "from_name": seg.from_name,
                    "to_name": seg.to_name,
                    "status": seg.status,
                    "algorithm": seg.algorithm,
                    "length_px": round(seg.length_px, 2),
                    "node_path": seg.node_path,
                    "edge_path": seg.edge_path,
                    "error": seg.error,
                }
                for seg in self.segments
            ],
            "path_points": [[round(p[0], 2), round(p[1], 2)] for p in self.global_path_points],
        }


# ===================================================================
# 图构建
# ===================================================================

def _node_x(n: Dict) -> float:
    """获取节点 x 坐标，兼容 x / x_pixel"""
    return n.get("x", n.get("x_pixel", 0.0))

def _node_y(n: Dict) -> float:
    """获取节点 y 坐标，兼容 y / y_pixel"""
    return n.get("y", n.get("y_pixel", 0.0))


def _build_adjacency(nodes: List[Dict], edges: List[Dict]) -> Dict:
    """构建邻接表。

    Returns:
        adjacency: {node_id: [(neighbor_id, edge_id, edge_length_px, edge_points_pixel), ...]}
    """
    adj: Dict = {}
    for n in nodes:
        adj[n["id"]] = []

    for e in edges:
        s, t = e["start"], e["end"]
        w = e.get("length_pixel", 1.0)
        pts = e.get("points_pixel", [])
        eid = e.get("id")

        if s in adj:
            adj[s].append((t, eid, w, pts))
        if t in adj:
            direction_pts = pts if len(pts) > 0 else pts
            reverse_pts = list(reversed(direction_pts)) if len(direction_pts) > 0 else []
            adj[t].append((s, eid, w, reverse_pts))

    return adj


# ===================================================================
# A* 算法
# ===================================================================

def astar(
    nodes: List[Dict], edges: List[Dict],
    start_id, goal_id,
    config: PlannerConfig = None,
    forbidden_nodes: Optional[set] = None,
) -> Optional[Tuple[List, List, List[List[float]]]]:
    """A* 路径规划。

    Returns:
        (node_path, edge_path, polyline_points) 或 None
    """
    if forbidden_nodes is None:
        forbidden_nodes = set()
    adj = _build_adjacency(nodes, edges)
    if start_id not in adj or goal_id not in adj:
        return None

    # node id → position
    id_to_pos: Dict = {n["id"]: (_node_x(n), _node_y(n)) for n in nodes}
    goal_x, goal_y = id_to_pos.get(goal_id, (0, 0))

    open_set = []
    heapq.heappush(open_set, (0, 0, start_id))  # (f_score, tiebreaker, node_id)

    g_score = {start_id: 0.0}
    came_from: Dict = {}       # node_id → (prev_node_id, edge_id, edge_points)

    closed = set()

    while open_set:
        f, _, current = heapq.heappop(open_set)

        if current in closed:
            continue
        closed.add(current)

        if current == goal_id:
            return _reconstruct_path(came_from, current, id_to_pos, start_id)

        cur_x, cur_y = id_to_pos.get(current, (0, 0))

        for neighbor, eid, weight, pts in adj.get(current, []):
            if neighbor in closed:
                continue
            # ★ 禁止经过 forbidden 节点（除非是起点或终点）
            if neighbor in forbidden_nodes and neighbor not in (start_id, goal_id):
                continue

            tentative_g = g_score[current] + weight

            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = (current, eid, pts)

                # 启发式：欧几里得距离
                nx, ny = id_to_pos.get(neighbor, (0, 0))
                h = math.sqrt((goal_x - nx)**2 + (goal_y - ny)**2)

                f_score = tentative_g + h
                heapq.heappush(open_set, (f_score, neighbor, neighbor))

    return None


# ===================================================================
# Dijkstra 算法
# ===================================================================

def dijkstra(
    nodes: List[Dict], edges: List[Dict],
    start_id, goal_id,
    config: PlannerConfig = None,
    forbidden_nodes: Optional[set] = None,
) -> Optional[Tuple[List, List, List[List[float]]]]:
    """Dijkstra 最短路径。

    Returns:
        (node_path, edge_path, polyline_points) 或 None
    """
    if forbidden_nodes is None:
        forbidden_nodes = set()
    adj = _build_adjacency(nodes, edges)
    if start_id not in adj or goal_id not in adj:
        return None

    id_to_pos: Dict = {n["id"]: (_node_x(n), _node_y(n)) for n in nodes}

    dist = {nid: float("inf") for nid in adj}
    prev: Dict = {}       # node_id → (prev_node_id, edge_id, edge_points)

    dist[start_id] = 0
    pq = [(0, start_id)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == goal_id:
            break

        for v, eid, w, pts in adj.get(u, []):
            # ★ 禁止经过 forbidden 节点（除非是起点或终点）
            if v in forbidden_nodes and v not in (start_id, goal_id):
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = (u, eid, pts)
                heapq.heappush(pq, (nd, v))

    if dist[goal_id] == float("inf"):
        return None

    return _reconstruct_path(prev, goal_id, id_to_pos, start_id)


def _reconstruct_path(
    came_from: Dict,
    goal_id,
    id_to_pos: Dict,
    start_id,
) -> Tuple[List, List, List[List[float]]]:
    """回溯路径，展开 edge polyline。

    Returns:
        (node_path, edge_path, polyline_points)
    """
    node_path = []
    edge_path = []

    current = goal_id
    while current in came_from:
        node_path.append(current)
        prev_node, edge_id, pts = came_from[current]
        edge_path.append(edge_id)
        current = prev_node
    node_path.append(current)  # start_id

    node_path.reverse()
    edge_path.reverse()

    # 展开 polyline：按 edge_path 顺序拼接 points_pixel
    polyline = []
    for i, nid in enumerate(node_path):
        pos = id_to_pos.get(nid)
        if pos is None:
            continue
        polyline.append([pos[0], pos[1]])

    return node_path, edge_path, polyline


# ===================================================================
# 路径展开与重采样
# ===================================================================

class EdgeGeometryMissingError(RuntimeError):
    """边缺少可用 polyline，禁止用节点直线替代。"""

    def __init__(self, edge_id=None, detail: str = ""):
        self.edge_id = edge_id
        msg = "edge_geometry_missing"
        if edge_id is not None:
            msg = f"edge_geometry_missing: edge_id={edge_id}"
        if detail:
            msg = f"{msg} ({detail})"
        super().__init__(msg)


def _edge_polyline_points(edge: Dict) -> List:
    """读取边几何；只接受 points_pixel / polyline，不自动生成直线。"""
    pts = edge.get("points_pixel")
    if pts is None or len(pts) < 2:
        pts = edge.get("polyline")
    if pts is None:
        return []
    return list(pts)


def expand_edge_path_to_polyline(
    node_path: List, edge_path: List,
    nodes: List[Dict], edges: List[Dict],
    *,
    endpoint_tolerance_px: float = 8.0,
) -> List[List[float]]:
    """将 node_path + edge_path 按 edge polyline 展开为连续 dense path。

    禁止用 node-to-node 直线替代缺失几何；缺失时抛出
    ``EdgeGeometryMissingError``（消息含 ``edge_geometry_missing``）。
    方向与 polyline 首尾不一致时抛出 ``edge_polyline_endpoint_mismatch``。
    """
    if len(node_path) < 2:
        return []

    from roadnet.path_layer_diagnostics import _index_by_id, orient_polyline_for_travel

    id_to_node = _index_by_id(nodes)
    id_to_edge = _index_by_id(edges)

    result: List[List[float]] = []
    for i in range(len(edge_path)):
        eid = edge_path[i]
        e = id_to_edge.get(eid)
        if e is None and eid is not None:
            e = id_to_edge.get(str(eid))
        if e is None:
            raise EdgeGeometryMissingError(eid, "edge not found in graph")

        current_node = node_path[i]
        next_node = node_path[i + 1]
        oriented, _direction, err = orient_polyline_for_travel(
            e, current_node, next_node, id_to_node,
            tolerance_px=endpoint_tolerance_px,
        )
        if err is not None or len(oriented) < 2:
            raise EdgeGeometryMissingError(eid, err or "points_pixel/polyline missing")

        if not result:
            result.extend(oriented)
        else:
            start_k = 1 if (
                abs(result[-1][0] - oriented[0][0]) < 0.5
                and abs(result[-1][1] - oriented[0][1]) < 0.5
            ) else 0
            result.extend(oriented[start_k:])

    return result


def resample_polyline(
    polyline: List[List[float]],
    spacing_px: float = 20.0,
) -> List[List[float]]:
    """等距重采样折线。

    Args:
        polyline: [[x,y], ...]
        spacing_px: 采样间距（像素）

    Returns:
        重采样后的折线
    """
    if len(polyline) < 2:
        return list(polyline)

    # 计算累计弧长
    seg_lens = []
    total_len = 0.0
    for i in range(len(polyline) - 1):
        dx = polyline[i+1][0] - polyline[i][0]
        dy = polyline[i+1][1] - polyline[i][1]
        sl = math.sqrt(dx*dx + dy*dy)
        seg_lens.append(sl)
        total_len += sl

    if total_len < spacing_px:
        return list(polyline)

    result = [[polyline[0][0], polyline[0][1]]]
    sampled_dist = spacing_px

    cum_dist = 0.0
    for i in range(len(polyline) - 1):
        seg_len = seg_lens[i]
        x1, y1 = polyline[i][0], polyline[i][1]
        x2, y2 = polyline[i+1][0], polyline[i+1][1]

        while sampled_dist <= cum_dist + seg_len:
            t = (sampled_dist - cum_dist) / seg_len if seg_len > 0 else 0
            sx = x1 + t * (x2 - x1)
            sy = y1 + t * (y2 - y1)
            result.append([sx, sy])
            sampled_dist += spacing_px

        cum_dist += seg_len

    # 确保包含终点
    last = polyline[-1]
    if result[-1][0] != last[0] or result[-1][1] != last[1]:
        result.append([last[0], last[1]])

    return result


def compute_yaw_for_waypoints(points: List[List[float]]) -> List[float]:
    """计算路径点序列的朝向角（弧度）。

    Args:
        points: [[x,y], ...]

    Returns:
        [yaw1, yaw2, ...] 每个点 forward 方向的 yaw 角
    """
    if len(points) < 2:
        return [0.0] * len(points)

    yaws = []
    for i in range(len(points)):
        if i < len(points) - 1:
            dx = points[i+1][0] - points[i][0]
            dy = points[i+1][1] - points[i][1]
        else:
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
        yaw = math.atan2(dy, dx)  # x 轴方向为 0
        yaws.append(yaw)

    return yaws


# ===================================================================
# 全局规划（分段）
# ===================================================================

def plan_global_path(
    snapped_points: List,    # SnappedTaskPoint list
    nodes: List[Dict],
    edges: List[Dict],
    config: PlannerConfig = None,
) -> GlobalPlanResult:
    """按任务点顺序分段规划全局路径。

    关键设计：
    1. **分段隔离虚拟节点**：每一段（task_i → task_{i+1}）只插入该段端点
       的虚拟节点到原始图副本，防止路径穿过其他任务点的虚拟节点。
    2. 按 seq 排序后严格依序 segment 规划。
    3. 拼接时仅去掉相邻段重合的连接点，完整保留每段的语义。

    Args:
        snapped_points: 吸附后的任务点列表
        nodes: 原始节点列表（不会被修改）
        edges: 原始边列表（不会被修改）
        config: 规划配置

    Returns:
        GlobalPlanResult
    """
    if config is None:
        config = PlannerConfig()

    from roadnet.task_snapping import insert_virtual_nodes

    # 按 seq 排序（不允许按 point_type 或 snapped node 距离重新排序）
    sorted_sp = sorted(snapped_points, key=lambda x: x.seq)

    # 过滤失败点
    failed = [sp for sp in sorted_sp if sp.status == "failed"]
    if failed:
        seqs = [sp.seq for sp in failed]
        return GlobalPlanResult(
            success=False,
            segments=[SegmentPlanResult(
                from_seq=0, to_seq=0,
                from_name="", to_name="",
                status="unreachable", length_px=0,
                algorithm=config.algorithm,
                node_path=[], edge_path=[], path_points=[],
                error=f"任务点吸附失败 (seq={seqs})，无法规划",
                from_virtual_node="", to_virtual_node="",
            )],
        )

    # 预先为每个任务点解析其 graph node ID
    # ★ 优先使用 virtual_node_id（现在所有任务点都有），以保证
    #    path_node_sequence 中始终包含 task_N_virtual 标记。
    def _resolve_node_id(sp, extra_nodes: list):
        """解析任务点在当前图副本中的 graph node ID"""
        if sp.virtual_node_id is not None:
            return sp.virtual_node_id
        if sp.node_id is not None:
            return sp.node_id
        # 孤儿点：临时创建节点
        vnid = f"task_{sp.seq}_orphan"
        extra_nodes.append({
            "id": vnid, "x": sp.snapped_x, "y": sp.snapped_y,
            "type": "virtual", "source": "orphan",
        })
        return vnid

    # ★ 规划结果
    result = GlobalPlanResult(
        coordinate_type="pixel",
        task_sequence=[sp.seq for sp in sorted_sp],
        success=True,
    )

    global_points = []
    total_length = 0.0

    # ★ 所有任务 virtual node ID 集合（用于 forbidden_nodes）
    all_task_vnids = set()
    for sp in sorted_sp:
        vnid = sp.virtual_node_id or f"task_{sp.seq}_virtual"
        all_task_vnids.add(vnid)

    for i in range(len(sorted_sp) - 1):
        src = sorted_sp[i]
        dst = sorted_sp[i + 1]

        # ★ 核心修改：每段使用独立图副本，只插入本段的两个端点虚拟节点
        extra_nodes: list = []
        seg_sp_list = [src, dst]
        seg_nodes, seg_edges = insert_virtual_nodes(nodes, edges, seg_sp_list)

        src_node_id = _resolve_node_id(src, extra_nodes)
        dst_node_id = _resolve_node_id(dst, extra_nodes)
        for en in extra_nodes:
            seg_nodes.append(en)

        # ★ 构建 forbidden_nodes：本段之外的 task virtual node 禁止作为中间节点
        current_vnids = {src_node_id, dst_node_id}
        forbidden = all_task_vnids - current_vnids

        # 运行规划（在隔离图副本上，禁止经过非本段 virtual node）
        if config.algorithm == "astar":
            planner_result = astar(seg_nodes, seg_edges, src_node_id, dst_node_id,
                                   config, forbidden_nodes=forbidden)
        else:
            planner_result = dijkstra(seg_nodes, seg_edges, src_node_id, dst_node_id,
                                      config, forbidden_nodes=forbidden)

        if planner_result is None:
            # Dijkstra 保底（同样禁止非本段 virtual node）
            backup_result = dijkstra(seg_nodes, seg_edges, src_node_id, dst_node_id,
                                     config, forbidden_nodes=forbidden)
            if backup_result is None:
                result.success = False
                result.segments.append(SegmentPlanResult(
                    from_seq=src.seq, to_seq=dst.seq,
                    from_name=_point_type_name(src.point_type), to_name=_point_type_name(dst.point_type),
                    status="unreachable", length_px=0,
                    algorithm=config.algorithm,
                    node_path=[], edge_path=[], path_points=[],
                    error=f"从 seq={src.seq} 到 seq={dst.seq} 不可达",
                    from_virtual_node=str(src.virtual_node_id or ""),
                    to_virtual_node=str(dst.virtual_node_id or ""),
                ))
                continue
            node_path, edge_path, polyline = backup_result
            algo_used = "dijkstra_fallback"
        else:
            node_path, edge_path, polyline = planner_result
            algo_used = config.algorithm

        # ★ 路径完整性检查：当前段不应出现非本段的 task virtual node
        unexpected = set()
        for nid in node_path:
            nid_str = str(nid)
            if nid_str in all_task_vnids and nid_str not in current_vnids:
                unexpected.add(nid_str)

        # ★ 构建 segment 诊断信息
        seg_error = ""
        from_vn = src.virtual_node_id
        to_vn = dst.virtual_node_id

        if unexpected:
            result.success = False
            seg_error = (
                f"段 {src.seq}→{dst.seq} 的路径中意外经过"
                f" 非本段 task virtual 节点: {sorted(unexpected)}。"
                f"预期只应包含 from={from_vn} 和 to={to_vn}。"
            )

        # 展开完整 polyline（必须使用 edge.polyline，禁止直线替代）
        try:
            expanded = expand_edge_path_to_polyline(
                node_path, edge_path, seg_nodes, seg_edges,
            )
        except EdgeGeometryMissingError as exc:
            result.success = False
            result.segments.append(SegmentPlanResult(
                from_seq=src.seq, to_seq=dst.seq,
                from_name=_point_type_name(src.point_type),
                to_name=_point_type_name(dst.point_type),
                status="unreachable",
                length_px=0,
                algorithm=algo_used,
                node_path=[str(n) for n in node_path],
                edge_path=[str(e) for e in edge_path],
                path_points=[],
                error=str(exc),
                from_virtual_node=str(from_vn or ""),
                to_virtual_node=str(to_vn or ""),
                unexpected_task_virtual_nodes=sorted(unexpected),
            ))
            continue

        seg_len = _compute_path_length(expanded)

        result.segments.append(SegmentPlanResult(
            from_seq=src.seq, to_seq=dst.seq,
            from_name=_point_type_name(src.point_type),
            to_name=_point_type_name(dst.point_type),
            status="ok" if not unexpected else "unreachable",
            length_px=seg_len if not unexpected else 0,
            algorithm=algo_used,
            node_path=[str(n) for n in node_path],
            edge_path=[str(e) for e in edge_path],
            path_points=[[p[0], p[1]] for p in expanded] if not unexpected else [],
            error=seg_error,
            from_virtual_node=str(from_vn or ""),
            to_virtual_node=str(to_vn or ""),
            unexpected_task_virtual_nodes=sorted(unexpected),
        ))

        if unexpected:
            continue

        # 拼接全局路径（去掉段首以避免相邻段重复连接点）
        if i == 0:
            global_points.extend(expanded)
        else:
            # 仅去重：如果上一段的终点就是本段起点，则跳过第一个点
            if global_points and len(expanded) > 0:
                last = global_points[-1]
                first = expanded[0]
                if abs(last[0] - first[0]) < 0.5 and abs(last[1] - first[1]) < 0.5:
                    global_points.extend(expanded[1:])
                else:
                    global_points.extend(expanded)
            else:
                global_points.extend(expanded)

        total_length += seg_len

    result.global_path_points = [[p[0], p[1]] for p in global_points]
    result.total_length_px = total_length

    return result


def _point_type_name(ptype: int) -> str:
    """点类型名称"""
    return {0: "start", 1: "goal", 2: "via"}.get(ptype, "task")


def _compute_path_length(points: List[List[float]]) -> float:
    """计算路径总长"""
    total = 0.0
    for i in range(len(points) - 1):
        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]
        total += math.sqrt(dx*dx + dy*dy)
    return total


# ===================================================================
# 保存
# ===================================================================

def save_global_path(result: GlobalPlanResult, output_dir: str) -> str:
    """保存全局规划结果到 JSON（兼容旧版，像素坐标）"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "global_path.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"[GlobalPlanner] 规划结果已保存: {path}")
    return path


def save_global_path_pixel(result: GlobalPlanResult, output_dir: str) -> str:
    """保存全局规划路径（像素坐标）到 global_path_pixel.json。

    包含完整的 path_points [[x_pixel, y_pixel], ...] 和每段详情。
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "global_path_pixel.json")

    output = {
        "coordinate_system": "image_pixel",
        "total_length_px": round(result.total_length_px, 2),
        "task_sequence": result.task_sequence,
        "segments": [],
        "path_points": [[round(p[0], 2), round(p[1], 2)]
                         for p in result.global_path_points],
        "point_count": len(result.global_path_points),
    }

    for seg in result.segments:
        output["segments"].append({
            "from_seq": seg.from_seq,
            "to_seq": seg.to_seq,
            "from_name": seg.from_name,
            "to_name": seg.to_name,
            "status": seg.status,
            "algorithm": seg.algorithm,
            "length_px": round(seg.length_px, 2),
            "node_path": seg.node_path,
            "edge_path": seg.edge_path,
            "path_points": [[round(p[0], 2), round(p[1], 2)]
                             for p in seg.path_points],
            "point_count": len(seg.path_points),
            "error": seg.error,
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[GlobalPlanner] 像素路径已保存: {path} ({len(result.global_path_points)} 点)")
    return path


def save_global_path_geo(
    result: GlobalPlanResult,
    output_dir: str,
    geo_calibration=None,
) -> str:
    """保存全局规划路径（经纬度坐标）到 global_path_geo.json。

    Args:
        result: 规划结果（像素坐标）
        output_dir: 输出目录
        geo_calibration: GeoCalibration 实例，用于 pixel→lon/lat 转换

    Returns:
        保存的文件路径，或空字符串（无校准器）
    """
    if geo_calibration is None:
        print("[GlobalPlanner] 无坐标校准器，跳过 geo 导出")
        return ""

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "global_path_geo.json")

    # 转换路径点为经纬度
    path_lonlat = []
    for px, py in result.global_path_points:
        try:
            lon, lat = geo_calibration.pixel_to_lonlat(px, py)
            path_lonlat.append([round(lon, 8), round(lat, 8)])
        except Exception as e:
            print(f"[GlobalPlanner] pixel_to_lonlat 失败 ({px}, {py}): {e}")
            path_lonlat.append([0.0, 0.0])

    # 计算地理总长度（米）
    total_len_m = 0.0
    if hasattr(geo_calibration, 'pixel_resolution_estimated_m'):
        total_len_m = result.total_length_px * (geo_calibration.pixel_resolution_estimated_m or 0.5)

    output = {
        "coordinate_system": "wgs84",
        "total_length_px": round(result.total_length_px, 2),
        "total_length_m": round(total_len_m, 1),
        "pixel_resolution_m": round(geo_calibration.pixel_resolution_estimated_m or 0.5, 4),
        "task_sequence": result.task_sequence,
        "path_points_lonlat": path_lonlat,
        "point_count": len(path_lonlat),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[GlobalPlanner] 地理路径已保存: {path} ({len(path_lonlat)} 点)")
    return path


def save_waypoints_yaml(
    result: GlobalPlanResult,
    output_dir: str,
    geo_calibration=None,
    *,
    default_altitude_m: float = 21.741,
) -> str:
    """deprecated: formal YAML must use vehicle_waypoint_pipeline.

    Do not call this for competition/official export. Kept for legacy debug only.

    格式：
        subject1_waypoints:
          waypoints:
            - name: wp_001
              latitude_deg: ...
              longitude_deg: ...
              altitude_m: ...

    Args:
        result: 规划结果
        output_dir: 输出目录
        geo_calibration: GeoCalibration 实例

    Returns:
        保存的文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 重采样路径点（减少航点密度）
    resample_spacing = 10  # 约 10px 间距采样一个航点
    points = result.global_path_points

    # 简单等距采样
    sampled_points = []
    if len(points) > 0:
        sampled_points.append(points[0])
        cum = 0.0
        for i in range(len(points) - 1):
            dx = points[i+1][0] - points[i][0]
            dy = points[i+1][1] - points[i][1]
            seg_len = math.sqrt(dx*dx + dy*dy)
            cum += seg_len
            if cum >= resample_spacing:
                sampled_points.append(points[i+1])
                cum = 0.0
        # 确保包含终点
        if len(points) > 1 and sampled_points[-1] != points[-1]:
            sampled_points.append(points[-1])

    from roadnet.path_export import export_subject1_waypoints_yaml

    vehicle_wps = []
    for i, pt in enumerate(sampled_points, 1):
        px, py = pt[0], pt[1]
        lon = lat = None
        if geo_calibration is not None:
            try:
                lon, lat = geo_calibration.pixel_to_lonlat(px, py)
            except Exception:
                lon = lat = None
        if lon is None or lat is None:
            # 无标定则跳过经纬度写出（正式 YAML 需要 lon/lat）
            continue
        vehicle_wps.append({
            "seq": i,
            "name": f"wp_{i:03d}",
            "longitude": float(lon),
            "latitude": float(lat),
            "longitude_deg": float(lon),
            "latitude_deg": float(lat),
            "altitude_m": float(default_altitude_m),
            "x_pixel": float(px),
            "y_pixel": float(py),
        })

    if not vehicle_wps:
        raise RuntimeError(
            "无法生成 subject1_waypoints.yaml：缺少有效经纬度（请先完成坐标标定）"
        )

    yaml_path = os.path.join(output_dir, "subject1_waypoints.yaml")
    export_subject1_waypoints_yaml(
        vehicle_wps, yaml_path, default_altitude_m=float(default_altitude_m),
    )
    # Compatibility copy — identical subject1 content
    compat = os.path.join(output_dir, "waypoints.yaml")
    with open(yaml_path, encoding="utf-8") as src:
        text = src.read()
    with open(compat, "w", encoding="utf-8") as dst:
        dst.write(text)
    print(f"[GlobalPlanner] subject1_waypoints.yaml 已保存: {yaml_path} ({len(vehicle_wps)} 航点)")
    return yaml_path
