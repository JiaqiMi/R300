"""
从 clean_skeleton 生成 graph 模块。

功能：
1. 检测端点和交叉口
2. 合并邻近节点
3. 追踪边 polyline
4. 过滤短边
5. Douglas-Peucker 折线简化
6. 输出格式兼容 GraphEditorQt

graph 格式：
  nodes: [{"id": int, "x": int, "y": int, "type": "junction"|"endpoint"}, ...]
  edges: [{"id": int, "from": int, "to": int, "length_px": float,
           "path": [[y,x], ...], "source": "auto"}, ...]

与 SAM-Road 原始 graph 的关键区别：
- node 数量少（只在端点和交叉口创建节点）
- edge 保存完整 polyline
- 不产生三角网状误连
"""

from __future__ import annotations

import math
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import cv2
from roadnet.graph_utils import as_bool, ensure_graph_python_types, ensure_python_types


@dataclass
class SkeletonToGraphConfig:
    """Skeleton → Graph 转换配置（按阶段顺序排列）

    阶段顺序:
      1. detect_nodes:   degree=1 endpoint, degree>=3 junction
      2. cluster_junctions:   junction_cluster_radius
      3. merge_endpoints:     endpoint_merge_distance
      4. merge_nodes:         node_merge_distance
      5. trace_edges:         沿骨架追踪
      6. filter_short_edges:  min_edge_length
      7. prune_dead_ends:     prune_length (仅修剪无出路的端点分支)
      8. connect_endpoints:   endpoint_connect_distance (自动补边连接 degree=1 端点)
      9. simplify_edges:      rdp_epsilon
    """
    # ── 路口聚类 ──
    junction_cluster_radius: int = 10    # 将邻近的 degree>=3 像素聚类为一个 junction node
    # ── 端点合并 ──
    endpoint_merge_distance: int = 12    # 将过于靠近的端点合并
    # ── 节点合并 ──
    node_merge_distance: int = 8         # 合并所有节点（junction + endpoint）
    # ── 短边过滤 ──
    min_edge_length: float = 8.0         # 删除长度 < 此值的边（不含 junction-junction 边）
    # ── 死端修剪 ──
    prune_length: float = 15.0           # 删除 degree=1 且长度 < 此值的死端分支
    # ── 端点自动连接 ──
    endpoint_connect_distance: float = 25.0  # 自动连接距离 < 此值的两个 degree=1 端点
    # ── 折线简化 ──
    rdp_epsilon: float = 2.0             # Douglas-Peucker 简化容差
    enable_short_edge_filter: bool = True
    enable_prune: bool = True
    # ── 兼容旧参数名 ──
    merge_node_distance: int = 15        # 兼容旧接口
    simplify_tolerance: float = 2.0      # 兼容旧接口（rdp_epsilon 别名）


# ===========================================================================
# 8 邻域常量
# ===========================================================================

_NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _count_neighbors(binary: np.ndarray, y: int, x: int) -> int:
    h, w = binary.shape
    cnt = 0
    for dy, dx in _NEIGHBORS_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and as_bool(binary[ny, nx]):
            cnt += 1
    return cnt


def _get_neighbors(binary: np.ndarray, y: int, x: int) -> List[Tuple[int, int]]:
    h, w = binary.shape
    result = []
    for dy, dx in _NEIGHBORS_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and binary[ny, nx]:
            result.append((int(ny), int(nx)))
    return result


# ===========================================================================
# 节点检测
# ===========================================================================

def detect_nodes(
    skeleton: np.ndarray,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    检测骨架上的端点和交叉口。

    Returns:
        (endpoints, normals, junctions)
        - endpoints: degree=1 的端点 [(y,x), ...]
        - normals:   degree=2 的普通点 [(y,x), ...]
        - junctions: degree>=3 的交叉口 [(y,x), ...]
    """
    binary = skeleton > 0
    ys, xs = np.where(binary)
    endpoints, normals, junctions = [], [], []
    for y, x in zip(ys, xs):
        deg = _count_neighbors(binary, int(y), int(x))
        if deg == 1:
            endpoints.append((int(y), int(x)))
        elif deg == 2:
            normals.append((int(y), int(x)))
        else:
            junctions.append((int(y), int(x)))
    return endpoints, normals, junctions


# ===========================================================================
# 节点聚类与合并
# ===========================================================================

def _cluster_pixels(
    pixels: List[Tuple[int, int]],
    distance: int = 15,
) -> List[List[Tuple[int, int]]]:
    """BFS 聚类，将距离 <= distance 的像素归为一组。"""
    if not pixels:
        return []
    pixel_set = set(pixels)
    visited = set()
    clusters = []
    for py, px in pixels:
        if (py, px) in visited:
            continue
        cluster = []
        stack = [(py, px)]
        visited.add((py, px))
        while stack:
            cy, cx = stack.pop()
            cluster.append((cy, cx))
            for dy in range(-distance, distance + 1):
                for dx in range(-distance, distance + 1):
                    ny, nx = cy + dy, cx + dx
                    if (ny, nx) in pixel_set and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        stack.append((ny, nx))
        clusters.append(cluster)
    return clusters


def _snap_to_skeleton(
    skeleton: np.ndarray, y: int, x: int, radius: int = 15,
) -> Optional[Tuple[int, int]]:
    """将坐标吸附到最近的骨架像素。"""
    binary = skeleton > 0
    h, w = binary.shape
    best_d = float("inf")
    best_pt = None
    y0, x0 = max(0, y - radius), max(0, x - radius)
    y1, x1 = min(h, y + radius + 1), min(w, x + radius + 1)
    for sy in range(y0, y1):
        for sx in range(x0, x1):
            if binary[sy, sx]:
                d = (sy - y) ** 2 + (sx - x) ** 2
                if d < best_d:
                    best_d = d
                    best_pt = (sy, sx)
    return best_pt


def merge_nodes(
    endpoints: List[Tuple[int, int]],
    junctions: List[Tuple[int, int]],
    merge_distance: int = 15,
    skeleton: Optional[np.ndarray] = None,
) -> Tuple[List[Dict], Dict[Tuple[int, int], int]]:
    """
    合并邻近的端点和交叉口为统一节点。

    策略：
    1. junction 相互靠近的聚类成一个节点（取质心，吸附到骨架）
    2. endpoint 靠近任何节点的被吸收合并
    3. endpoint 之间靠近的聚类合并

    Returns:
        (nodes, pixel_to_node)
        nodes: [{"id": int, "y": int, "x": int, "type": str, "degree": int}, ...]
        pixel_to_node: {(y,x) -> node_id}
    """
    all_special = endpoints + junctions
    clusters = _cluster_pixels(all_special, merge_distance)

    nodes = []
    pixel_to_node = {}
    node_id = 0

    for cluster in clusters:
        avg_y = int(round(sum(p[0] for p in cluster) / len(cluster)))
        avg_x = int(round(sum(p[1] for p in cluster) / len(cluster)))

        if skeleton is not None:
            snap_y, snap_x = _snap_to_skeleton(skeleton, avg_y, avg_x, radius=merge_distance)
            if snap_y is not None:
                avg_y, avg_x = snap_y, snap_x

        node_type = "endpoint"
        for py, px in cluster:
            if (py, px) in junctions:
                node_type = "junction"
                break

        node = {"id": node_id, "y": avg_y, "x": avg_x, "type": node_type}
        nodes.append(node)
        for py, px in cluster:
            pixel_to_node[(py, px)] = node_id
        node_id += 1

    for node in nodes:
        pos = (node["y"], node["x"])
        if pos not in pixel_to_node:
            pixel_to_node[pos] = node["id"]

    return nodes, pixel_to_node


# ===========================================================================
# 边追踪
# ===========================================================================

def trace_edges(
    skeleton: np.ndarray,
    nodes: List[Dict],
    pixel_to_node: Dict[Tuple[int, int], int],
) -> List[Dict]:
    """
    从节点出发沿骨架追踪，生成边。

    Returns:
        edges: [{"id": int, "from": int, "to": int, "length_px": float,
                  "path": [[y,x],...]}, ...]
    """
    binary = skeleton > 0
    h, w = binary.shape
    edges_raw: Dict[Tuple, Dict] = {}

    for node in nodes:
        ny, nx = node["y"], node["x"]
        start_id = node["id"]

        for dy, dx in _NEIGHBORS_8:
            sy, sx = ny + dy, nx + dx
            if not (0 <= sy < h and 0 <= sx < w):
                continue
            if not as_bool(binary[sy, sx]):
                continue
            if (sy, sx) in pixel_to_node:
                continue

            path = [(ny, nx), (sy, sx)]
            visited = {(ny, nx), (sy, sx)}
            cy, cx = sy, sx

            while True:
                next_pts = []
                for d2y, d2x in _NEIGHBORS_8:
                    n2y, n2x = cy + d2y, cx + d2x
                    if not (0 <= n2y < h and 0 <= n2x < w):
                        continue
                    if not as_bool(binary[n2y, n2x]):
                        continue
                    if (n2y, n2x) in visited:
                        continue
                    next_pts.append((n2y, n2x))

                if len(next_pts) == 0:
                    break
                elif len(next_pts) == 1:
                    cy, cx = next_pts[0]
                    visited.add((cy, cx))
                    path.append((cy, cx))
                    if (cy, cx) in pixel_to_node:
                        end_id = pixel_to_node[(cy, cx)]
                        if start_id != end_id:
                            a, b = sorted([start_id, end_id])
                            key = (a, b)
                            plen = sum(
                                math.sqrt(
                                    (path[i + 1][0] - path[i][0]) ** 2 +
                                    (path[i + 1][1] - path[i][1]) ** 2
                                )
                                for i in range(len(path) - 1)
                            )
                            if key not in edges_raw or plen > edges_raw[key].get("length_px", 0):
                                edges_raw[key] = {
                                    "from": a, "to": b,
                                    "length_px": round(plen, 2),
                                    "path": [[p[0], p[1]] for p in path],
                                }
                        break
                else:
                    break

    edges_out = []
    for eid, key in enumerate(sorted(edges_raw.keys())):
        edge = edges_raw[key]
        edge["id"] = eid
        edges_out.append(edge)

    return edges_out


# ===========================================================================
# 边过滤
# ===========================================================================

def filter_short_edges(edges: List[Dict], min_length: float = 20.0) -> List[Dict]:
    """删除长度 < min_length 的短边。"""
    return [e for e in edges if e["length_px"] >= min_length]


def filter_short_edges_smart(
    edges: List[Dict],
    nodes: List[Dict],
    min_length: float = 8.0,
) -> List[Dict]:
    """智能短边过滤。

    规则：
    - 如果 edge 连两个 junction 节点，即使短也保留
    - 普通短边 (< min_length) 删除
    """
    node_type_map = {n["id"]: n.get("type", "endpoint") for n in nodes}
    result = []
    for e in edges:
        f_type = node_type_map.get(e["from"], "endpoint")
        t_type = node_type_map.get(e["to"], "endpoint")
        is_junction_pair = (f_type == "junction" and t_type == "junction")
        if is_junction_pair:
            result.append(e)
        elif e["length_px"] >= min_length:
            result.append(e)
    return result


def prune_dead_ends(
    edges: List[Dict],
    nodes: List[Dict],
    min_length: float = 15.0,
) -> List[Dict]:
    """修剪死端分支（degree=1 且长度 < min_length 的边）。

    只删除纯端点（非 junction）的短分支。
    保留所有 junction 节点相关的边。

    Returns:
        修剪后的 edges 列表
    """
    if not edges or min_length <= 0:
        return edges

    def _compute_degrees(edges_list):
        deg = {}
        for e in edges_list:
            deg[e["from"]] = deg.get(e["from"], 0) + 1
            deg[e["to"]] = deg.get(e["to"], 0) + 1
        return deg

    node_type_map = {n["id"]: n.get("type", "endpoint") for n in nodes}
    working = list(edges)

    # 迭代修剪，因为修剪后其他节点可能变成 degree=1
    changed = True
    while changed:
        changed = False
        deg = _compute_degrees(working)
        new_edges = []
        for e in working:
            if e["length_px"] >= min_length:
                new_edges.append(e)
                continue
            # 短边：检查是否连接的是纯端点
            f_type = node_type_map.get(e["from"], "endpoint")
            t_type = node_type_map.get(e["to"], "endpoint")
            f_deg = deg.get(e["from"], 0)
            t_deg = deg.get(e["to"], 0)

            # A standalone endpoint-to-endpoint segment is a complete small
            # road component, not a dangling spur.  Keep it intact.
            if f_deg == 1 and t_deg == 1:
                new_edges.append(e)
                continue

            # 保留 junction-junction 连接
            if f_type == "junction" and t_type == "junction":
                new_edges.append(e)
                continue
            # 如果任一端是 degree=1 的非 junction 节点，则删除
            if (f_deg == 1 and f_type != "junction") or (t_deg == 1 and t_type != "junction"):
                changed = True
                continue
            # 其他情况保留
            new_edges.append(e)
        working = new_edges

    return working


def connect_endpoints(
    edges: List[Dict],
    nodes: List[Dict],
    skeleton: np.ndarray,
    connect_distance: float = 25.0,
    road_mask: Optional[np.ndarray] = None,
    support_radius: int = 4,
    min_support_ratio: float = 0.70,
) -> int:
    """自动连接距离近的 degree=1 端点。

    流程：
    1. 找到当前所有 degree=1 的端点
    2. 计算端点间欧氏距离
    3. 对距离 < connect_distance 的端点对：
       a. 检查端点间直线是否沿骨架（采样点靠近骨架像素）
       b. 如果通过，创建新边连接
    4. 每个端点最多连接一次（最近的那个）

    Returns:
        新增的边数量
    """
    if not nodes or connect_distance <= 0:
        return 0

    # 计算 degree
    degree = {n["id"]: 0 for n in nodes}
    for e in edges:
        degree[e["from"]] = degree.get(e["from"], 0) + 1
        degree[e["to"]] = degree.get(e["to"], 0) + 1

    # 找 degree=1 的端点
    endpoints = [n for n in nodes if degree.get(n["id"], 0) == 1]
    if len(endpoints) < 2:
        return 0

    # 计算端点对距离
    pairs = []
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            ni, nj = endpoints[i], endpoints[j]
            dx = ni["x"] - nj["x"]
            dy = ni["y"] - nj["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= connect_distance:
                pairs.append((dist, ni, nj))

    if not pairs:
        return 0

    # 按距离排序（优先连接最近的）
    pairs.sort(key=lambda x: x[0])
    used = set()
    added = 0
    max_edge_id = max((e.get("id", 0) for e in edges), default=0)
    binary = skeleton > 0
    h, w = binary.shape
    support = binary
    if road_mask is not None:
        rm = np.asarray(road_mask)
        if rm.ndim == 3:
            rm = rm[:, :, 0]
        if rm.shape == binary.shape:
            support = rm > 0
    kernel_size = support_radius * 2 + 1
    support_near = cv2.dilate(
        support.astype(np.uint8),
        np.ones((kernel_size, kernel_size), np.uint8),
    ) > 0

    for dist, na, nb in pairs:
        if na["id"] in used or nb["id"] in used:
            continue

        # 检查连接线是否沿骨架
        y1, x1 = na["y"], na["x"]
        y2, x2 = nb["y"], nb["x"]

        # 对连线采样，检查是否靠近骨架
        num_samples = max(3, int(dist / 3))
        supported = 0
        longest_unsupported = 0
        unsupported_run = 0
        for k in range(1, num_samples):
            t = k / num_samples
            sy = int(round(y1 + t * (y2 - y1)))
            sx = int(round(x1 + t * (x2 - x1)))
            if 0 <= sy < h and 0 <= sx < w:
                if support_near[sy, sx]:
                    supported += 1
                    unsupported_run = 0
                else:
                    unsupported_run += 1
                    longest_unsupported = max(longest_unsupported, unsupported_run)

        # 至少 60% 的采样点靠近骨架才允许连接
        ratio = supported / max(num_samples - 1, 1)
        max_run = max(1, int(math.ceil((num_samples - 1) * 0.25)))
        if ratio < min_support_ratio or longest_unsupported > max_run:
            continue

        # 创建新边
        path = [[y1, x1]]
        for k in range(1, num_samples):
            t = k / num_samples
            py = int(round(y1 + t * (y2 - y1)))
            px = int(round(x1 + t * (x2 - x1)))
            path.append([py, px])
        path.append([y2, x2])

        plen = float(dist)
        new_edge = {
            "id": max_edge_id + added + 1,
            "from": na["id"],
            "to": nb["id"],
            "length_px": round(plen, 2),
            "path": path,
            "source": "auto_endpoint_connect",
        }
        edges.append(new_edge)
        degree[na["id"]] = degree.get(na["id"], 0) + 1
        degree[nb["id"]] = degree.get(nb["id"], 0) + 1
        used.add(na["id"])
        used.add(nb["id"])
        added += 1

        print(f"[EndpointConnect] connected endpoint {na['id']} <-> {nb['id']} "
              f"dist={dist:.1f}px, road_support={ratio:.1%}")

    if not added:
        print(f"[EndpointConnect] no suitable pairs within {connect_distance}px")

    return added


# ===========================================================================
# Douglas-Peucker 简化
# ===========================================================================

def _douglas_peucker(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) <= 2:
        return points
    dmax, index = 0, 0
    start, end = points[0], points[-1]
    vec = end - start
    vec_norm = np.linalg.norm(vec)
    for i in range(1, len(points) - 1):
        if vec_norm < 1e-8:
            d = np.linalg.norm(points[i] - start)
        else:
            cross = abs(vec[0] * (start[1] - points[i][1]) - vec[1] * (start[0] - points[i][0]))
            d = cross / vec_norm
        if d > dmax:
            dmax, index = d, i
    if dmax > epsilon:
        left = _douglas_peucker(points[:index + 1], epsilon)
        right = _douglas_peucker(points[index:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.array([start, end])


def simplify_edges(edges: List[Dict], tolerance: float = 1.5) -> List[Dict]:
    """Douglas-Peucker 简化边 polyline。"""
    for edge in edges:
        path = edge.get("path", [])
        if len(path) <= 2:
            continue
        pts = np.array([[p[0], p[1]] for p in path], dtype=np.float64)
        simplified = _douglas_peucker(pts, tolerance)
        edge["path"] = [[int(round(p[0])), int(round(p[1]))] for p in simplified]
        plen = sum(
            math.sqrt(
                (edge["path"][i + 1][0] - edge["path"][i][0]) ** 2 +
                (edge["path"][i + 1][1] - edge["path"][i][1]) ** 2
            )
            for i in range(len(edge["path"]) - 1)
        )
        edge["length_px"] = round(plen, 2)
    return edges


def _build_clustered_node_regions(
    skeleton: np.ndarray,
    endpoints: List[Tuple[int, int]],
    junctions: List[Tuple[int, int]],
    junction_cluster_radius: int,
    endpoint_merge_distance: int,
    node_merge_distance: int,
) -> Tuple[List[Dict], Dict[Tuple[int, int], int], Dict[int, Set[Tuple[int, int]]]]:
    """Build graph nodes while retaining every skeleton pixel owned by a node.

    Edge tracing must start from the boundary of a junction cluster, not from its
    centroid.  Losing this ownership map was the main cause of visually connected
    skeleton branches disappearing from the graph.
    """
    nodes: List[Dict] = []
    pixel_to_node: Dict[Tuple[int, int], int] = {}
    node_pixels: Dict[int, Set[Tuple[int, int]]] = {}

    def add_cluster(cluster: List[Tuple[int, int]], node_type: str) -> int:
        nid = len(nodes)
        cy = int(round(sum(p[0] for p in cluster) / len(cluster)))
        cx = int(round(sum(p[1] for p in cluster) / len(cluster)))
        snap = _snap_to_skeleton(skeleton, cy, cx, radius=max(1, node_merge_distance))
        if snap is not None:
            cy, cx = snap
        nodes.append({"id": nid, "y": cy, "x": cx, "type": node_type})
        owned = set(cluster)
        node_pixels[nid] = owned
        for p in owned:
            pixel_to_node[p] = nid
        return nid

    for cluster in _cluster_pixels(junctions, max(0, junction_cluster_radius)):
        add_cluster(cluster, "junction")

    # Merge nearby endpoints only when they belong to different skeleton
    # components.  Merging both ends of one short segment into the same node
    # turns a real edge into a self-loop and was another source of lost roads.
    _, component_labels = cv2.connectedComponents(
        (skeleton > 0).astype(np.uint8), connectivity=8
    )
    endpoint_clusters: List[List[Tuple[int, int]]] = []
    cluster_components: List[Set[int]] = []
    for endpoint in sorted(endpoints):
        ey, ex = endpoint
        component = int(component_labels[ey, ex])
        chosen = None
        for index, cluster in enumerate(endpoint_clusters):
            if component in cluster_components[index]:
                continue
            if any(math.hypot(ey - py, ex - px) <= endpoint_merge_distance
                   for py, px in cluster):
                chosen = index
                break
        if chosen is None:
            endpoint_clusters.append([endpoint])
            cluster_components.append({component})
        else:
            endpoint_clusters[chosen].append(endpoint)
            cluster_components[chosen].add(component)
    for cluster in endpoint_clusters:
        cy = sum(p[0] for p in cluster) / len(cluster)
        cx = sum(p[1] for p in cluster) / len(cluster)
        nearest = None
        nearest_d = float("inf")
        for node in nodes:
            d = math.hypot(node["y"] - cy, node["x"] - cx)
            if d <= node_merge_distance and d < nearest_d:
                nearest, nearest_d = node["id"], d
        if nearest is not None:
            node_pixels[nearest].update(cluster)
            for p in cluster:
                pixel_to_node[p] = nearest
        else:
            add_cluster(cluster, "endpoint")

    return nodes, pixel_to_node, node_pixels


def _trace_edges_from_regions(
    skeleton: np.ndarray,
    nodes: List[Dict],
    pixel_to_node: Dict[Tuple[int, int], int],
    node_pixels: Dict[int, Set[Tuple[int, int]]],
) -> List[Dict]:
    """Trace every skeleton branch from junction-region boundary to boundary."""
    binary = skeleton > 0
    node_by_id = {n["id"]: n for n in nodes}
    used_links: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    traced: List[Dict] = []

    def link(a, b):
        return tuple(sorted((a, b)))

    for start in nodes:
        sid = start["id"]
        for boundary in sorted(node_pixels.get(sid, {(start["y"], start["x"])})):
            for nxt in _get_neighbors(binary, boundary[0], boundary[1]):
                if pixel_to_node.get(nxt) == sid or link(boundary, nxt) in used_links:
                    continue
                path = [(start["y"], start["x"]), boundary]
                prev, cur = boundary, nxt
                local_seen = {boundary}
                end_id = None
                while True:
                    used_links.add(link(prev, cur))
                    if cur != path[-1]:
                        path.append(cur)
                    owner = pixel_to_node.get(cur)
                    if owner is not None:
                        end_id = owner
                        break
                    if cur in local_seen:
                        break
                    local_seen.add(cur)
                    candidates = [p for p in _get_neighbors(binary, cur[0], cur[1])
                                  if p != prev and p not in local_seen]
                    if not candidates:
                        break
                    # Non-node skeleton pixels should have degree 2.  The stable
                    # ordering is only a guard for diagonal pixel artifacts.
                    candidates.sort(key=lambda p: (p[0], p[1]))
                    prev, cur = cur, candidates[0]

                if end_id is None or end_id == sid:
                    continue
                end = node_by_id[end_id]
                if path[-1] != (end["y"], end["x"]):
                    path.append((end["y"], end["x"]))
                plen = sum(math.hypot(path[i + 1][0] - path[i][0],
                                      path[i + 1][1] - path[i][1])
                           for i in range(len(path) - 1))
                traced.append({
                    "id": len(traced), "from": sid, "to": end_id,
                    "length_px": round(plen, 2),
                    "path": [[int(y), int(x)] for y, x in path],
                    "source": "auto",
                })

    # The same physical path can be entered from both ends before all boundary
    # links are consumed.  Deduplicate by endpoints and normalized pixel path.
    unique = []
    signatures = set()
    for edge in traced:
        forward = tuple(tuple(p) for p in edge["path"])
        reverse = tuple(reversed(forward))
        sig = (min(edge["from"], edge["to"]), max(edge["from"], edge["to"]),
               min(forward, reverse))
        if sig in signatures:
            continue
        signatures.add(sig)
        edge["id"] = len(unique)
        unique.append(edge)
    return unique


# ===========================================================================
# 主入口
# ===========================================================================

def skeleton_to_graph(
    skeleton: np.ndarray,
    config: Optional[SkeletonToGraphConfig] = None,
    road_mask: Optional[np.ndarray] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """从 skeleton 提取路网图（完整流程）。

    Args:
        skeleton: 二值骨架 (H, W) uint8, 0/255
        config:   转换配置

    Returns:
        (nodes, edges)
        nodes: [{"id": int, "y": int, "x": int, "type": str, "degree": int}, ...]
        edges: [{"id": int, "from": int, "to": int, "length_px": float,
                 "path": [[y,x], ...], "source": str}, ...]
    """
    if config is None:
        config = SkeletonToGraphConfig()

    # ── 兼容旧参数名 ──
    jcr = getattr(config, 'junction_cluster_radius', config.merge_node_distance)
    emd = getattr(config, 'endpoint_merge_distance', config.node_merge_distance)
    nmd = getattr(config, 'node_merge_distance', config.merge_node_distance)
    mel = getattr(config, 'min_edge_length', 8.0)
    pl = getattr(config, 'prune_length', 15.0)
    ecd = getattr(config, 'endpoint_connect_distance', 25.0)
    rdp = getattr(config, 'rdp_epsilon', config.simplify_tolerance)

    h, w = skeleton.shape
    print(f"[SkeletonToGraph] skeleton={skeleton.shape} nonzero={(skeleton>0).sum()}")
    print(f"[SkeletonToGraph] junction_cluster_radius={jcr}, endpoint_merge={emd}, "
          f"node_merge={nmd}, min_edge={mel}, prune={pl}, endpoint_connect={ecd}, rdp={rdp}")

    # Step 1: 检测节点
    endpoints, _, junctions_list = detect_nodes(skeleton)
    print(f"[SkeletonToGraph] detected: {len(endpoints)} endpoints, {len(junctions_list)} junctions")

    # Steps 2-5: retain complete junction regions and trace from their boundary.
    nodes, pixel_to_node, node_pixels = _build_clustered_node_regions(
        skeleton, endpoints, junctions_list, jcr, emd, nmd
    )
    print(f"[SkeletonToGraph] clustered/merged nodes: {len(nodes)}")
    edges = _trace_edges_from_regions(
        skeleton, nodes, pixel_to_node, node_pixels
    )
    print(f"[SkeletonToGraph] traced edges: {len(edges)}")

    # Step 6: 过滤短边（但保留 junction-junction 连接）
    if getattr(config, "enable_short_edge_filter", True) and mel > 0:
        edges = filter_short_edges_smart(edges, nodes, min_length=mel)
    print(f"[SkeletonToGraph] after smart short-edge filter: {len(edges)}")

    # Step 7: 修剪死端
    if getattr(config, "enable_prune", True) and pl > 0:
        edges = prune_dead_ends(edges, nodes, min_length=pl)
    print(f"[SkeletonToGraph] after prune dead-ends: {len(edges)}")

    # Step 8: 端点自动连接
    added = connect_endpoints(
        edges, nodes, skeleton, connect_distance=ecd, road_mask=road_mask
    )
    print(f"[SkeletonToGraph] endpoint_connect added {added} edges")

    # Step 9: 简化折线
    edges = simplify_edges(edges, tolerance=rdp)

    # Step 10: 补全 metadata
    for e in edges:
        e.setdefault("source", "auto")

    degree = {n["id"]: 0 for n in nodes}
    for e in edges:
        start = e.get("from", e.get("start"))
        end = e.get("to", e.get("end"))
        if start in degree:
            degree[start] += 1
        if end in degree:
            degree[end] += 1
    for n in nodes:
        n["degree"] = degree[n["id"]]

    # ★ 确保所有坐标和 polyline 都是 Python 原生类型
    nodes, edges = ensure_graph_python_types(nodes, edges)

    print(f"[SkeletonToGraph] final: {len(nodes)} nodes, {len(edges)} edges")
    return nodes, edges


# ===========================================================================
# 保存
# ===========================================================================

def save_graph_from_skeleton(
    nodes: List[Dict],
    edges: List[Dict],
    output_dir: str,
    image_rgb: Optional[np.ndarray] = None,
    skeleton: Optional[np.ndarray] = None,
    filename: str = "graph_from_skeleton.json",
) -> str:
    """
    保存 graph JSON 文件。

    格式兼容 GraphEditorQt 的 load_draft()。

    Args:
        nodes: 节点列表
        edges: 边列表
        output_dir: 输出目录
        image_rgb: 可选 RGB 图像
        skeleton: 可选骨架
        filename: 输出文件名（默认 graph_from_skeleton.json）
    """
    # 为 JSON 同步 degree
    degree = {n["id"]: 0 for n in nodes}
    for e in edges:
        start = e.get("from", e.get("start"))
        end = e.get("to", e.get("end"))
        if start in degree:
            degree[start] += 1
        if end in degree:
            degree[end] += 1
    for n in nodes:
        n["degree"] = degree[n["id"]]

    # ★ 转换 nodes 为 image_pixel 坐标格式 (x, y)
    # skeleton_to_graph 生成的节点使用 {"id", "y", "x", "type", "degree"}
    # 确保兼容 GraphEditorQt 期望的 x/y/pixel_x/pixel_y 格式
    nodes_out = []
    for n in nodes:
        n_out = dict(n)
        if "x" not in n_out and "pixel_x" in n_out:
            n_out["x"] = n_out["pixel_x"]
        if "y" not in n_out and "pixel_y" in n_out:
            n_out["y"] = n_out["pixel_y"]
        if "x" in n_out and "pixel_x" not in n_out:
            n_out["pixel_x"] = n_out["x"]
        if "y" in n_out and "pixel_y" not in n_out:
            n_out["pixel_y"] = n_out["y"]
        nodes_out.append(n_out)

    # ★ edges: 将 [y,x] 的 path 转换为 points_pixel [[x,y],...]
    edges_out = []
    for e in edges:
        e_out = dict(e)
        # 兼容 from/to → start/end
        if "from" in e_out and "start" not in e_out:
            e_out["start"] = e_out.pop("from")
        if "to" in e_out and "end" not in e_out:
            e_out["end"] = e_out.pop("to")
        # path [y,x] → points_pixel [x,y]
        if "path" in e_out and "points_pixel" not in e_out:
            e_out["points_pixel"] = [[p[1], p[0]] for p in e_out["path"]]
        # length_px → length_pixel
        if "length_px" in e_out and "length_pixel" not in e_out:
            e_out["length_pixel"] = e_out["length_px"]
        edges_out.append(e_out)

    graph = {
        "coordinate_system": "image_pixel",
        "nodes": nodes_out,
        "edges": edges_out,
        "metadata": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "source": "skeleton_to_graph",
        },
    }

    # ★ 最终一次全量转换，确保 JSON 可序列化
    graph = ensure_python_types(graph)

    json_path = os.path.join(output_dir, filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"[GRAPH] 已保存: {json_path} ({len(nodes)} 节点, {len(edges)} 边)")

    return json_path
