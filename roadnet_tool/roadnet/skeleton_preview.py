"""
骨架化 + 路网图提取模块 V3：骨架化、毛刺删除、节点检测、图构建。

功能：
1. skimage.morphology.skeletonize 生成单像素骨架
2. 删除短毛刺
3. 检测端点(endpoints)和交叉点(junctions)
4. 从骨架中提取路网图（节点+边），输出 nodes.csv / edges.csv / road_graph.json

输出文件：
- road_skeleton_raw.png / road_skeleton_pruned.png
- road_skeleton_overlay.png
- skeleton_nodes_preview.png
- nodes.csv / edges.csv / road_graph.json
"""

import csv
import json
import os
import numpy as np
from typing import Tuple, List, Set, Dict, Optional
import cv2


# ===========================================================================
# 骨架化
# ===========================================================================

def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """
    对二值 mask 进行骨架化。

    Args:
        mask: 二值 mask (H, W) uint8, 0/255

    Returns:
        skeleton: 二值骨架 (H, W) uint8, 0/255
    """
    from skimage.morphology import skeletonize

    binary = mask > 0
    skel = skeletonize(binary)
    result = (skel * 255).astype(np.uint8)
    return result


# ===========================================================================
# 毛刺删除
# ===========================================================================

def prune_skeleton(
    skeleton: np.ndarray,
    min_branch_length: int = 30,
) -> np.ndarray:
    """
    删除骨架上的短毛刺分支。

    算法：
    1. 检测所有端点
    2. 从每个端点出发，沿骨架追踪直到遇到交叉点或达到 min_branch_length
    3. 如果追踪长度 < min_branch_length，删除该分支

    Args:
        skeleton:         二值骨架 (H, W) uint8, 0/255
        min_branch_length: 最小保留分支长度（像素）

    Returns:
        pruned: 删除短毛刺后的骨架
    """
    binary = skeleton > 0
    # 找到端点（邻居数=1）
    endpoints = _find_endpoints(binary)
    # 找到交叉点（邻居数>=3）
    junctions = _find_junctions(binary)

    # 构建删除标记
    to_remove = np.zeros_like(binary, dtype=bool)

    h, w = binary.shape
    # 8 邻域偏移
    _neighbors_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for ey, ex in endpoints:
        # BFS 追踪从端点出发的分支
        branch_pixels = []
        visited = set()
        cy, cx = ey, ex

        while True:
            if (cy, cx) in visited:
                break
            visited.add((cy, cx))
            branch_pixels.append((cy, cx))

            # 找到下一个未访问的邻居
            next_pixels = []
            for dy, dx in _neighbors_offsets:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if binary[ny, nx] and (ny, nx) not in visited:
                        next_pixels.append((ny, nx))

            if len(next_pixels) == 0:
                break
            elif len(next_pixels) == 1:
                cy, cx = next_pixels[0]
            else:
                # 遇到分支点，检查是否到达交叉点
                current_is_junction = (cy, cx) in junctions
                # 已经走了足够远 → 保留
                if current_is_junction or len(branch_pixels) >= min_branch_length:
                    break
                # 还没走够 → 需要在多条路径中选择，简化：走第一条
                cy, cx = next_pixels[0]

        # 如果分支长度不足，标记删除
        if 0 < len(branch_pixels) < min_branch_length:
            for py, px in branch_pixels:
                to_remove[py, px] = True

    # 应用删除
    binary[to_remove] = False
    pruned = (binary * 255).astype(np.uint8)
    return pruned


# ===========================================================================
# 节点检测
# ===========================================================================

def detect_nodes(
    skeleton: np.ndarray,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    检测骨架的端点和交叉点。

    使用 8 邻域统计每个骨架像素的邻居数：
    - 邻居数 = 1：endpoint
    - 邻居数 = 2：普通道路点
    - 邻居数 >= 3：junction

    Args:
        skeleton: 二值骨架 (H, W) uint8, 0/255

    Returns:
        (endpoints, junctions)
        endpoints: List[(y, x), ...] 像素坐标
        junctions: List[(y, x), ...] 像素坐标
    """
    binary = skeleton > 0
    endpoints = _find_endpoints(binary)
    junctions = _find_junctions(binary)
    return endpoints, junctions


def _count_neighbors(binary: np.ndarray, y: int, x: int) -> int:
    """统计 8 邻域中骨架像素的数量。"""
    h, w = binary.shape
    cnt = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and binary[ny, nx]:
                cnt += 1
    return cnt


def _find_endpoints(binary: np.ndarray) -> List[Tuple[int, int]]:
    """找到所有端点（邻居数=1 的骨架点）。"""
    ys, xs = np.where(binary)
    endpoints = []
    for y, x in zip(ys, xs):
        if _count_neighbors(binary, int(y), int(x)) == 1:
            endpoints.append((int(y), int(x)))
    return endpoints


def _find_junctions(binary: np.ndarray) -> Set[Tuple[int, int]]:
    """找到所有交叉点（邻居数>=3 的骨架点）。"""
    ys, xs = np.where(binary)
    junctions: Set[Tuple[int, int]] = set()
    for y, x in zip(ys, xs):
        if _count_neighbors(binary, int(y), int(x)) >= 3:
            junctions.add((int(y), int(x)))
    return junctions


# ===========================================================================
# 路网图构建（骨架 → nodes + edges）
# ===========================================================================

_NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _cluster_nearby_pixels(
    pixels: List[Tuple[int, int]],
    radius: int = 3,
) -> List[Tuple[int, int]]:
    """
    将相邻的像素聚类为单个代表点（取质心）。

    用于将互相邻接的 junction 像素合并为一个逻辑路口节点。

    Args:
        pixels: 需要聚类的像素列表 [(y, x), ...]
        radius:  合并半径（像素）

    Returns:
        聚类后的代表点列表
    """
    if not pixels:
        return []

    # BFS 连通分量聚类
    pixel_set = set(pixels)
    clusters = []
    visited = set()

    for py, px in pixels:
        if (py, px) in visited:
            continue
        cluster = []
        stack = [(py, px)]
        visited.add((py, px))
        while stack:
            cy, cx = stack.pop()
            cluster.append((cy, cx))
            for dy, dx in _NEIGHBORS_8:
                ny, nx = cy + dy, cx + dx
                if (ny, nx) in pixel_set and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    stack.append((ny, nx))

        # 取质心
        if cluster:
            avg_y = int(round(sum(p[0] for p in cluster) / len(cluster)))
            avg_x = int(round(sum(p[1] for p in cluster) / len(cluster)))
            clusters.append((avg_y, avg_x))

    return clusters


def build_road_graph(
    skeleton: np.ndarray,
) -> Tuple[List[Dict], List[Dict]]:
    """
    从骨架中提取路网图结构。

    算法：
    1. 检测所有端点 + 交叉点作为节点
    2. 对相邻 junction 像素做聚类合并
    3. 为每个节点分配唯一 ID
    4. 从每个端点出发沿骨架追踪 → 遇到节点时记录一条边
    5. 从每个交叉点出发，沿每个方向追踪 → 遇到其他节点时记录边
    6. 去重（无向边只保留一条）

    Args:
        skeleton: 二值骨架 (H, W) uint8, 0/255

    Returns:
        (nodes, edges)
        nodes: [{"id": int, "y": int, "x": int, "type": "endpoint"|"junction"}, ...]
        edges: [{"id": int, "from": int, "to": int, "length_px": float, "path": [[y,x],...]}, ...]
    """
    binary = skeleton > 0
    h, w = binary.shape

    # ----- 1. 找出所有节点候选 -----
    endpoint_pixels = _find_endpoints(binary)
    junction_pixels = list(_find_junctions(binary))

    # ----- 2. 聚类邻近的 junction -----
    junction_clusters = _cluster_nearby_pixels(junction_pixels, radius=4)

    # ----- 3. 合并：创建节点集（端点 + 聚类后 junction）-----
    # 构建一个集合，标记哪些像素是节点
    node_pixels: Set[Tuple[int, int]] = set()
    node_info: Dict[Tuple[int, int], Dict] = {}

    node_id = 0
    for ey, ex in endpoint_pixels:
        node_pixels.add((ey, ex))
        node_info[(ey, ex)] = {"id": node_id, "y": ey, "x": ex, "type": "endpoint"}
        node_id += 1

    for jy, jx in junction_clusters:
        node_pixels.add((jy, jx))
        node_info[(jy, jx)] = {"id": node_id, "y": jy, "x": jx, "type": "junction"}
        node_id += 1

    # Pixel → node lookup
    pixel_to_node: Dict[Tuple[int, int], int] = {
        (info["y"], info["x"]): info["id"] for info in node_info.values()
    }

    # ----- 4. 从每个节点出发，沿所有方向追踪边 -----
    edges_raw: Dict[Tuple[int, int], Dict] = {}  # key=(min_id, max_id) → edge dict

    all_nodes = [(info["y"], info["x"]) for info in node_info.values()]

    for ny, nx in all_nodes:
        start_node_id = pixel_to_node[(ny, nx)]

        # 找到从该节点出发的所有方向（相邻骨架像素）
        for dy, dx in _NEIGHBORS_8:
            sy, sx = ny + dy, nx + dx
            if not (0 <= sy < h and 0 <= sx < w):
                continue
            if not binary[sy, sx]:
                continue
            if (sy, sx) in node_pixels:
                continue  # 直接相邻的另一个节点，单独处理

            # 从 (sy, sx) 出发追踪
            path = [(ny, nx), (sy, sx)]
            visited_in_trace = {(ny, nx), (sy, sx)}
            cy, cx = sy, sx

            while True:
                # 找下一个未访问的骨架邻居
                next_candidates = []
                for d2y, d2x in _NEIGHBORS_8:
                    n2y, n2x = cy + d2y, cx + d2x
                    if not (0 <= n2y < h and 0 <= n2x < w):
                        continue
                    if not binary[n2y, n2x]:
                        continue
                    if (n2y, n2x) in visited_in_trace:
                        continue
                    next_candidates.append((n2y, n2x))

                if len(next_candidates) == 0:
                    break  # 死胡同
                elif len(next_candidates) == 1:
                    cy, cx = next_candidates[0]
                    visited_in_trace.add((cy, cx))
                    path.append((cy, cx))

                    # 碰到节点了 → 记录边
                    if (cy, cx) in node_pixels:
                        end_node_id = pixel_to_node[(cy, cx)]
                        if start_node_id != end_node_id:
                            a, b = sorted([start_node_id, end_node_id])
                            key = (a, b)
                            # 计算路径长度（欧氏距离累积）
                            path_length = sum(
                                ((path[i + 1][0] - path[i][0]) ** 2 +
                                 (path[i + 1][1] - path[i][1]) ** 2) ** 0.5
                                for i in range(len(path) - 1)
                            )
                            # 保留较长的路径（有时同一对节点间有2条路）
                            if key not in edges_raw or path_length > edges_raw[key].get("length_px", 0):
                                edges_raw[key] = {
                                    "from": a,
                                    "to": b,
                                    "length_px": round(path_length, 2),
                                    "path": [[p[0], p[1]] for p in path],
                                }
                        break  # 到达节点，停止追踪
                else:
                    # 多分支 → 应该是遇到了未标记的交叉区域，停止
                    break

    # ----- 5. 整理输出 -----
    nodes_out = sorted(node_info.values(), key=lambda n: n["id"])

    edges_out = []
    edge_id = 0
    for key, edge in sorted(edges_raw.items(), key=lambda x: x[0]):
        edge["id"] = edge_id
        edges_out.append(edge)
        edge_id += 1

    return nodes_out, edges_out


# ===========================================================================
# 保存函数
# ===========================================================================

def save_graph_outputs(
    nodes: List[Dict],
    edges: List[Dict],
    image_rgb: np.ndarray,
    skeleton: np.ndarray,
    output_dir: str,
) -> None:
    """
    保存路网图输出：nodes.csv, edges.csv, road_graph.json, road_graph_overlay.png.

    Args:
        nodes:    节点列表
        edges:    边列表
        image_rgb: 原图
        skeleton: 骨架
        output_dir: 输出目录
    """
    # ---- nodes.csv ----
    nodes_path = os.path.join(output_dir, "nodes.csv")
    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "y", "x", "type", "degree"])
        # 计算每个节点的 degree
        degree = {n["id"]: 0 for n in nodes}
        for e in edges:
            degree[e["from"]] += 1
            degree[e["to"]] += 1
        for n in nodes:
            writer.writerow([n["id"], n["y"], n["x"], n["type"], degree[n["id"]]])
    print(f"[GRAPH] 已保存节点: {nodes_path} ({len(nodes)} 个节点)")

    # ---- edges.csv ----
    edges_path = os.path.join(output_dir, "edges.csv")
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["edge_id", "from_node", "to_node", "length_px"])
        for e in edges:
            writer.writerow([e["id"], e["from"], e["to"], e["length_px"]])
    print(f"[GRAPH] 已保存边: {edges_path} ({len(edges)} 条边)")

    # ---- road_graph.json ----
    graph_path = os.path.join(output_dir, "road_graph.json")
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
    print(f"[GRAPH] 已保存路网图: {graph_path}")

    # ---- road_graph_overlay.png ----
    _draw_graph_overlay(image_rgb, nodes, edges, skeleton, output_dir)


def _draw_graph_overlay(
    image_rgb: np.ndarray,
    nodes: List[Dict],
    edges: List[Dict],
    skeleton: np.ndarray,
    output_dir: str,
) -> None:
    """绘制路网图叠加原图的可视化。"""
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # 骨架浅灰
    img[skeleton > 0] = (200, 200, 200)

    # 边 — 蓝色
    for e in edges:
        path = e.get("path", [])
        for i in range(len(path) - 1):
            p1 = (path[i][1], path[i][0])
            p2 = (path[i + 1][1], path[i + 1][0])
            cv2.line(img, p1, p2, (255, 0, 0), 1, cv2.LINE_AA)  # BGR blue

    # 端点 — 绿色圆
    for n in nodes:
        if n["type"] == "endpoint":
            cv2.circle(img, (n["x"], n["y"]), 5, (0, 255, 0), -1)

    # 交叉点 — 红色圆
    for n in nodes:
        if n["type"] == "junction":
            cv2.circle(img, (n["x"], n["y"]), 6, (0, 0, 255), -1)

    # 标注 node_id
    font = cv2.FONT_HERSHEY_SIMPLEX
    for n in nodes:
        cv2.putText(img, str(n["id"]), (n["x"] + 8, n["y"] - 4),
                    font, 0.35, (255, 255, 0), 1, cv2.LINE_AA)

    overlay_path = os.path.join(output_dir, "road_graph_overlay.png")
    cv2.imwrite(overlay_path, img)
    print(f"[GRAPH] 已保存路网图叠加: {overlay_path}")


# ===========================================================================
# 可视化函数
# ===========================================================================

def draw_skeleton_overlay(
    image_rgb: np.ndarray,
    skeleton: np.ndarray,
    skeleton_color: tuple = (0, 255, 255),
) -> np.ndarray:
    """
    将骨架叠加到原图上。

    Args:
        image_rgb:      原始 RGB 图像
        skeleton:       二值骨架 (H, W) uint8
        skeleton_color: 骨架颜色 (R, G, B)

    Returns:
        叠加上骨架的 RGB 图像
    """
    overlay = image_rgb.copy()
    overlay[skeleton > 0] = skeleton_color
    result = cv2.addWeighted(image_rgb, 0.4, overlay, 0.6, 0)
    return result


def draw_nodes_overlay(
    image_rgb: np.ndarray,
    skeleton: np.ndarray,
    endpoints: List[Tuple[int, int]],
    junctions: List[Tuple[int, int]],
    skeleton_color: tuple = (200, 200, 200),
    endpoint_color: tuple = (0, 255, 0),
    junction_color: tuple = (255, 0, 0),
    marker_radius: int = 4,
) -> np.ndarray:
    """
    将骨架和节点叠加到原图上。

    Args:
        image_rgb:    原始 RGB 图像
        skeleton:     二值骨架
        endpoints:    端点列表 [(y, x), ...]
        junctions:    交叉点列表 [(y, x), ...]
        skeleton_color: 骨架线颜色 (R, G, B)
        endpoint_color: 端点标记颜色 (R, G, B)
        junction_color: 交叉点标记颜色 (R, G, B)
        marker_radius:  标记半径

    Returns:
        叠加后的 RGB 图像
    """
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # 绘制骨架线（浅灰色）
    img_bgr[skeleton > 0] = skeleton_color[::-1]

    # 绘制端点（绿色圆）
    for y, x in endpoints:
        cv2.circle(img_bgr, (x, y), marker_radius, endpoint_color[::-1], -1)

    # 绘制交叉点（红色圆）
    for y, x in junctions:
        cv2.circle(img_bgr, (x, y), marker_radius + 1, junction_color[::-1], -1)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ===========================================================================
# 主入口函数（供 main.py 调用）
# ===========================================================================

def run_skeleton_preview(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    output_dir: str,
    min_branch_length: int = 30,
    build_graph: bool = False,
) -> None:
    """
    运行骨架化预览流水线，生成所有预览文件。

    Args:
        image_rgb:        原始 RGB 图像
        mask:             道路二值 mask (H, W) uint8
        output_dir:       输出目录
        min_branch_length: 最小保留分支长度
        build_graph:      是否生成路网图（nodes.csv / edges.csv / road_graph.json）
    """
    # Step 1: 骨架化
    print("[SKEL] 正在生成骨架...")
    skeleton_raw = skeletonize_mask(mask)

    skel_path = os.path.join(output_dir, "road_skeleton_raw.png")
    cv2.imwrite(skel_path, skeleton_raw)
    print(f"[SKEL] 已保存原始骨架: {skel_path}")

    # Step 2: 毛刺删除
    print(f"[SKEL] 正在修剪短毛刺（最小分支长度={min_branch_length}）...")
    skeleton_pruned = prune_skeleton(skeleton_raw, min_branch_length=min_branch_length)

    pruned_path = os.path.join(output_dir, "road_skeleton_pruned.png")
    cv2.imwrite(pruned_path, skeleton_pruned)
    print(f"[SKEL] 已保存修剪后骨架: {pruned_path}")

    # Step 3: 骨架叠加图
    overlay = draw_skeleton_overlay(image_rgb, skeleton_pruned)
    overlay_path = os.path.join(output_dir, "road_skeleton_overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"[SKEL] 已保存骨架叠加图: {overlay_path}")

    # Step 4: 节点检测 + 可视化
    endpoints, junctions = detect_nodes(skeleton_pruned)
    print(f"[SKEL] 检测到 {len(endpoints)} 个端点, {len(junctions)} 个交叉点")

    nodes_img = draw_nodes_overlay(image_rgb, skeleton_pruned, endpoints, junctions)
    nodes_path = os.path.join(output_dir, "skeleton_nodes_preview.png")
    cv2.imwrite(nodes_path, cv2.cvtColor(nodes_img, cv2.COLOR_RGB2BGR))
    print(f"[SKEL] 已保存节点预览图: {nodes_path}")
    print(f"[SKEL]   - 绿色圆点 = 端点 (endpoint)")
    print(f"[SKEL]   - 红色圆点 = 交叉点 (junction)")
    print(f"[SKEL]   - 灰色线 = 骨架中心线")

    # 骨架统计
    skel_pixels = int(skeleton_raw.sum() / 255)
    pruned_pixels = int(skeleton_pruned.sum() / 255)
    print(f"[SKEL] 骨架统计: 原始骨架 {skel_pixels} px, "
          f"修剪后 {pruned_pixels} px, "
          f"修剪掉 {skel_pixels - pruned_pixels} px ({100 * (skel_pixels - pruned_pixels) / max(skel_pixels, 1):.1f}%)")

    # Step 5: 路网图构建（可选）
    if build_graph:
        print("\n[GRAPH] 正在从骨架构建路网图...")
        nodes, edges = build_road_graph(skeleton_pruned)
        print(f"[GRAPH] 提取到 {len(nodes)} 个节点, {len(edges)} 条边")
        save_graph_outputs(nodes, edges, image_rgb, skeleton_pruned, output_dir)
