"""
V3.1 自动 draft graph 提取模块

功能：
1. 从优化骨架提取节点和边
2. 8邻域统计节点类型（endpoint / normal / junction）
3. 合并距离小于 merge_node_distance 的节点
4. 从节点追踪骨架生成边
5. 删除长度小于 min_edge_length 的边
6. 对 edge polyline 做 Douglas-Peucker 简化

输出：
- draft_nodes.csv
- draft_edges.csv
- draft_graph.json
- draft_graph_overlay.png

推荐配置:
  graph:
    merge_node_distance: 15
    min_edge_length: 40
    simplify_tolerance: 2.0
"""

import csv
import json
import math
import os
import numpy as np
from typing import List, Tuple, Set, Dict, Optional
import cv2
from roadnet.graph_utils import as_bool, ensure_graph_python_types


# ===========================================================================
# 8邻域
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
        if 0 <= ny < h and 0 <= nx < w and as_bool(binary[ny, nx]):
            result.append((ny, nx))
    return result


# ===========================================================================
# 节点检测
# ===========================================================================

def detect_all_nodes(
    skeleton: np.ndarray,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    检测骨架上的所有节点类型。

    Args:
        skeleton: 二值骨架 (H, W) uint8, 0/255

    Returns:
        (endpoints, normals, junctions)
        - endpoints: degree=1 的像素 [(y,x), ...]
        - normals:   degree=2 的像素 [(y,x), ...]
        - junctions: degree>=3 的像素 [(y,x), ...]
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
# 节点合并
# ===========================================================================

def _cluster_nodes(
    pixels: List[Tuple[int, int]],
    distance: int = 15,
) -> List[List[Tuple[int, int]]]:
    """
    将距离 <= distance 的节点聚类。

    Returns:
        聚类列表 [[(y,x), ...], ...]
    """
    if not pixels:
        return []

    # 使用 BFS 聚类
    pixel_set = set(pixels)
    visited = set()
    clusters = []

    # 构建快速查找邻域的结构（距离 <= distance）
    for py, px in pixels:
        if (py, px) in visited:
            continue
        cluster = []
        stack = [(py, px)]
        visited.add((py, px))
        while stack:
            cy, cx = stack.pop()
            cluster.append((cy, cx))
            # 搜索 distance 范围内的其他节点
            for dy in range(-distance, distance + 1):
                for dx in range(-distance, distance + 1):
                    ny, nx = cy + dy, cx + dx
                    if (ny, nx) in pixel_set and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        stack.append((ny, nx))
        clusters.append(cluster)

    return clusters


def merge_nodes(
    endpoints: List[Tuple[int, int]],
    junctions: List[Tuple[int, int]],
    merge_node_distance: int = 15,
    skeleton: Optional[np.ndarray] = None,
) -> Tuple[List[Dict], Dict[Tuple[int, int], int]]:
    """
    合并距离较近的节点，创建统一的节点集。

    策略：
    1. junction 相互靠近的聚类为同一个节点（取质心，并吸附到骨架像素上）
    2. endpoint 与任何节点距离 <= merge_node_distance 则吸收合并
    3. endpoint 与 endpoint 距离 <= merge_node_distance 的聚类合并

    Args:
        endpoints: 端点列表
        junctions: 交叉点列表
        merge_node_distance: 合并距离
        skeleton: 二值骨架，用于吸附质心（可选）

    Returns:
        (nodes, pixel_to_node)
        nodes: [{"id": int, "y": int, "x": int, "type": str, "degree": int}, ...]
        pixel_to_node: {(y,x) -> node_id}
    """
    all_special = endpoints + junctions
    clusters = _cluster_nodes(all_special, merge_node_distance)

    nodes = []
    pixel_to_node = {}
    node_id = 0

    for cluster in clusters:
        # 计算质心
        avg_y = int(round(sum(p[0] for p in cluster) / len(cluster)))
        avg_x = int(round(sum(p[1] for p in cluster) / len(cluster)))

        # 吸附到最近的骨架像素上
        if skeleton is not None:
            snap_y, snap_x = _snap_to_skeleton(skeleton, avg_y, avg_x, radius=merge_node_distance)
            if snap_y is not None:
                avg_y, avg_x = snap_y, snap_x

        # 确定类型：如果有 junction 则为 junction
        node_type = "endpoint"
        for py, px in cluster:
            if (py, px) in junctions:
                node_type = "junction"
                break

        node = {"id": node_id, "y": avg_y, "x": avg_x, "type": node_type}
        nodes.append(node)

        # 将该聚类中的所有原始像素都映射到这个节点
        for py, px in cluster:
            pixel_to_node[(py, px)] = node_id

        node_id += 1

    # 将节点位置也注册到 pixel_to_node（确保在骨架上的位置被映射）
    for node in nodes:
        pos = (node["y"], node["x"])
        if pos not in pixel_to_node:
            pixel_to_node[pos] = node["id"]

    return nodes, pixel_to_node


def _snap_to_skeleton(
    skeleton: np.ndarray, y: int, x: int, radius: int = 15
) -> Optional[Tuple[int, int]]:
    """将坐标吸附到最近的骨架像素上。"""
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

    算法：
    1. 对每个节点，找出其所有 8 邻域的骨架邻居
    2. 从每个邻居出发沿骨架追踪
    3. 遇到节点时记录边
    4. 遇到分支点时停止

    Args:
        skeleton:      二值骨架 (H, W) uint8, 0/255
        nodes:         节点列表
        pixel_to_node: 像素到节点的映射

    Returns:
        edges: [{"id": int, "from": int, "to": int, "length_px": float,
                  "path": [[y,x],...]}, ...]
    """
    binary = skeleton > 0
    h, w = binary.shape

    edges_raw: Dict[Tuple[int, int], Dict] = {}

    for node in nodes:
        ny, nx = node["y"], node["x"]
        start_id = node["id"]

        # 从节点出发的所有方向
        for dy, dx in _NEIGHBORS_8:
            sy, sx = ny + dy, nx + dx
            if not (0 <= sy < h and 0 <= sx < w):
                continue
            if not binary[sy, sx]:
                continue
            if (sy, sx) in pixel_to_node:
                continue  # 直接相邻的节点，后续单独处理

            # 开始追踪
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
                    break  # 死路
                elif len(next_pts) == 1:
                    cy, cx = next_pts[0]
                    visited.add((cy, cx))
                    path.append((cy, cx))

                    if (cy, cx) in pixel_to_node:
                        end_id = pixel_to_node[(cy, cx)]
                        if start_id != end_id:
                            a, b = sorted([start_id, end_id])
                            key = (a, b)

                            # 计算欧氏路径长度
                            plen = 0
                            for i in range(len(path) - 1):
                                dy2 = path[i + 1][0] - path[i][0]
                                dx2 = path[i + 1][1] - path[i][1]
                                plen += math.sqrt(dy2 * dy2 + dx2 * dx2)

                            if key not in edges_raw or plen > edges_raw[key].get("length_px", 0):
                                edges_raw[key] = {
                                    "from": a,
                                    "to": b,
                                    "length_px": round(plen, 2),
                                    "path": [[p[0], p[1]] for p in path],
                                }
                        break
                else:
                    break  # 多分支 → 停止

    # 整理输出
    edges_out = []
    edge_id = 0
    for key in sorted(edges_raw.keys()):
        edge = edges_raw[key]
        edge["id"] = edge_id
        edges_out.append(edge)
        edge_id += 1

    return edges_out


# ===========================================================================
# 边过滤与简化
# ===========================================================================

def filter_short_edges(
    edges: List[Dict],
    min_edge_length: float = 40.0,
) -> List[Dict]:
    """删除长度 < min_edge_length 的边。"""
    return [e for e in edges if e["length_px"] >= min_edge_length]


def simplify_edges(
    edges: List[Dict],
    tolerance: float = 2.0,
) -> List[Dict]:
    """
    使用 Douglas-Peucker 算法简化边的 polyline。

    Args:
        edges:     边列表
        tolerance: 简化容差（像素）

    Returns:
        简化后的边列表
    """
    for edge in edges:
        path = edge.get("path", [])
        if len(path) <= 2:
            continue
        pts = np.array([[p[0], p[1]] for p in path], dtype=np.float64)
        simplified = _douglas_peucker(pts, tolerance)
        edge["path"] = [[int(round(p[0])), int(round(p[1]))] for p in simplified]
        # 重新计算长度
        plen = 0
        for i in range(len(edge["path"]) - 1):
            dy = edge["path"][i + 1][0] - edge["path"][i][0]
            dx = edge["path"][i + 1][1] - edge["path"][i][1]
            plen += math.sqrt(dy * dy + dx * dx)
        edge["length_px"] = round(plen, 2)
    return edges


def _douglas_peucker(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Douglas-Peucker 折线简化算法。"""
    if len(points) <= 2:
        return points

    # 找到距离最远的点
    dmax = 0
    index = 0
    start = points[0]
    end = points[-1]
    vec = end - start
    vec_norm = np.linalg.norm(vec)

    for i in range(1, len(points) - 1):
        if vec_norm < 1e-8:
            d = np.linalg.norm(points[i] - start)
        else:
            # 点到直线的距离
            cross = abs(vec[0] * (start[1] - points[i][1]) - vec[1] * (start[0] - points[i][0]))
            d = cross / vec_norm
        if d > dmax:
            dmax = d
            index = i

    if dmax > epsilon:
        left = _douglas_peucker(points[:index + 1], epsilon)
        right = _douglas_peucker(points[index:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.array([start, end])


# ===========================================================================
# 向后兼容：从 skeleton 直接提取图的便捷函数
# ===========================================================================

def extract_graph_from_skeleton(
    skeleton: np.ndarray,
    merge_node_distance: int = 15,
    min_edge_length: float = 40.0,
    simplify_tolerance: float = 2.0,
) -> Tuple[List[Dict], List[Dict]]:
    """
    从骨架提取路网图（完整流程）。

    Args:
        skeleton:            二值骨架 (H, W) uint8, 0/255
        merge_node_distance: 节点合并距离
        min_edge_length:     最小边长度
        simplify_tolerance:  折线简化容差

    Returns:
        (nodes, edges)
    """
    # Step 1: 检测所有节点
    endpoints, _, junctions = detect_all_nodes(skeleton)

    # Step 2: 合并节点
    nodes, pixel_to_node = merge_nodes(endpoints, junctions, merge_node_distance, skeleton)

    # Step 3: 追踪边
    edges = trace_edges(skeleton, nodes, pixel_to_node)

    # Step 4: 过滤短边
    edges = filter_short_edges(edges, min_edge_length)

    # Step 5: 简化边折线
    edges = simplify_edges(edges, simplify_tolerance)

    # Step 6: 更新节点 degree
    degree = {n["id"]: 0 for n in nodes}
    for e in edges:
        degree[e["from"]] += 1
        degree[e["to"]] += 1
    for n in nodes:
        n["degree"] = degree[n["id"]]

    # ★ 确保所有坐标和 polyline 都是 Python 原生类型，不含 numpy 对象
    nodes, edges = ensure_graph_python_types(nodes, edges)

    return nodes, edges


# ===========================================================================
# 保存函数
# ===========================================================================

def save_draft_graph(
    nodes: List[Dict],
    edges: List[Dict],
    image_rgb: np.ndarray,
    skeleton: np.ndarray,
    output_dir: str,
) -> None:
    """
    保存 draft graph 输出文件。

    生成: draft_nodes.csv, draft_edges.csv, draft_graph.json, draft_graph_overlay.png
    """
    # ---- draft_nodes.csv ----
    nodes_path = os.path.join(output_dir, "draft_nodes.csv")
    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "y", "x", "type", "degree"])
        for n in nodes:
            writer.writerow([n["id"], n["y"], n["x"], n["type"], n.get("degree", 0)])
    print(f"[GRAPH] 已保存 draft 节点: {nodes_path} ({len(nodes)} 个)")

    # ---- draft_edges.csv ----
    edges_path = os.path.join(output_dir, "draft_edges.csv")
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["edge_id", "from_node", "to_node", "length_px", "path_points"])
        for e in edges:
            path_pts = len(e.get("path", []))
            writer.writerow([e["id"], e["from"], e["to"], e["length_px"], path_pts])
    print(f"[GRAPH] 已保存 draft 边: {edges_path} ({len(edges)} 条)")

    # ---- draft_graph.json ----
    graph_path = os.path.join(output_dir, "draft_graph.json")
    # 为 JSON 同步 degree
    degree = {n["id"]: 0 for n in nodes}
    for e in edges:
        degree[e["from"]] += 1
        degree[e["to"]] += 1
    for n in nodes:
        n["degree"] = degree[n["id"]]

    graph = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "image_size": {"width": image_rgb.shape[1], "height": image_rgb.shape[0]},
        },
    }
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"[GRAPH] 已保存 draft 路网图: {graph_path}")

    # ---- draft_graph_overlay.png ----
    _draw_draft_overlay(image_rgb, nodes, edges, skeleton, output_dir)


def _draw_draft_overlay(
    image_rgb: np.ndarray,
    nodes: List[Dict],
    edges: List[Dict],
    skeleton: np.ndarray,
    output_dir: str,
) -> None:
    """绘制 draft graph 叠加原图的可视化。"""
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # 骨架浅灰
    img[skeleton > 0] = (200, 200, 200)

    # 边 — 蓝色线
    for e in edges:
        path = e.get("path", [])
        for i in range(len(path) - 1):
            p1 = (path[i][1], path[i][0])
            p2 = (path[i + 1][1], path[i + 1][0])
            cv2.line(img, p1, p2, (255, 0, 0), 2, cv2.LINE_AA)

    # 端点 — 绿色圆
    for n in nodes:
        if n["type"] == "endpoint":
            cv2.circle(img, (n["x"], n["y"]), 6, (0, 255, 0), -1)
            cv2.circle(img, (n["x"], n["y"]), 6, (0, 180, 0), 2)

    # 交叉点 — 红色圆
    for n in nodes:
        if n["type"] == "junction":
            cv2.circle(img, (n["x"], n["y"]), 7, (0, 0, 255), -1)
            cv2.circle(img, (n["x"], n["y"]), 7, (0, 0, 180), 2)

    # 标注 node_id
    font = cv2.FONT_HERSHEY_SIMPLEX
    for n in nodes:
        cv2.putText(img, str(n["id"]), (n["x"] + 10, n["y"] - 6),
                    font, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    overlay_path = os.path.join(output_dir, "draft_graph_overlay.png")
    cv2.imwrite(overlay_path, img)
    print(f"[GRAPH] 已保存 draft 路网叠加图: {overlay_path}")


# ===========================================================================
# 主入口
# ===========================================================================

def run_draft_graph_extract(
    image_rgb: np.ndarray,
    skeleton: np.ndarray,
    output_dir: str,
    config: Optional[Dict] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    运行完整的 draft graph 提取流水线。

    Args:
        image_rgb:  原始 RGB 图像
        skeleton:   优化后的骨架 (H, W) uint8
        output_dir: 输出目录
        config:     配置字典

    Returns:
        (nodes, edges)
    """
    if config is None:
        config = {}
    graph_cfg = config.get("graph", {})

    merge_node_distance = graph_cfg.get("merge_node_distance", 15)
    min_edge_length = graph_cfg.get("min_edge_length", 40)
    simplify_tolerance = graph_cfg.get("simplify_tolerance", 2.0)

    print(f"[GRAPH] 节点合并距离={merge_node_distance}px, "
          f"最小边长度={min_edge_length}px, "
          f"折线简化容差={simplify_tolerance}px")

    # 提取图
    nodes, edges = extract_graph_from_skeleton(
        skeleton,
        merge_node_distance=merge_node_distance,
        min_edge_length=min_edge_length,
        simplify_tolerance=simplify_tolerance,
    )

    print(f"[GRAPH] 提取完成: {len(nodes)} 个节点, {len(edges)} 条边")

    # 分类统计
    endpoint_count = sum(1 for n in nodes if n["type"] == "endpoint")
    junction_count = sum(1 for n in nodes if n["type"] == "junction")
    print(f"[GRAPH] 节点类型: 端点={endpoint_count}, 交叉点={junction_count}")

    # 保存
    save_draft_graph(nodes, edges, image_rgb, skeleton, output_dir)

    return nodes, edges
