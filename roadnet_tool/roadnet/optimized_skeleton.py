"""
V3.1 自动 skeleton 优化模块

功能：
1. distance transform 过滤靠边点
2. 边界边距裁剪
3. 短毛刺删除
4. 邻近 junction 合并
5. 自动连接近距离断点
6. 保存 road_skeleton_raw.png / road_skeleton_optimized.png / overlay

推荐配置:
  skeleton:
    method: "medial_axis"
    min_center_dist: 4
    border_margin: 10
    min_branch_length: 50
    max_connect_dist: 45
    max_connect_angle: 45
    min_line_mask_overlap: 0.65
"""

"""
V3.2 自动 skeleton 优化模块

来源：D:/skeleton_fix_package_20260711/roadnet/optimized_skeleton.py
本文件是从独立的"道路骨架生成/优化修复包"集成进来的核心算法。

功能：
1. Mask 标准化（normalize_road_mask）
2. 多种骨架化方法（medial_axis / skeletonize / thin）
3. Distance transform 过滤靠边点 + 自适应中心距离阈值
4. 边界边距裁剪
5. 短毛刺删除
6. 邻近 junction 合并
7. 自动连接近距离断点
8. Junction 像素聚类
9. 可视化叠加输出
10. 完整优化流水线 + 统计输出

推荐配置:
  skeleton:
    method: "medial_axis"
    min_center_dist: 3.0
    border_margin: 10
    min_branch_length: 40
    max_connect_dist: 45
    max_connect_angle: 45
    min_line_mask_overlap: 0.65
    junction_cluster_radius: 10
"""

import csv
import json
import os
import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from collections import deque
import cv2


# ===========================================================================
# Mask 标准化
# ===========================================================================

def normalize_road_mask(mask: np.ndarray) -> np.ndarray:
    """
    将道路 mask 标准化为 0/255 二值图。

    支持输入：
    - bool mask → 0/255 uint8
    - 0/1 float → 0/255 uint8
    - 0/255 uint8 → 保持不变
    - 0-255 灰度概率图 → OTSU 自动阈值二值化
    - 3 通道彩色图 → 转灰度后二值化
    - NaN/Inf 安全处理

    暗噪声不会被误判为道路（OTSU 阈值会排除极暗区域）。
    如果 OTSU 结果道路像素占比 < 0.01%，回退到 >0 二值化。
    """
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * 255

    if arr.dtype != np.uint8:
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    unique_vals = np.unique(arr)
    if len(unique_vals) <= 2:
        return (arr > 0).astype(np.uint8) * 255

    if int(arr.max()) == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    road_ratio = float((binary > 0).sum()) / max(binary.size, 1)
    if road_ratio <= 0.0001:
        binary = (arr > 0).astype(np.uint8) * 255

    return binary.astype(np.uint8)


# ===========================================================================
# 骨架化
# ===========================================================================

def skeletonize_medial_axis(mask: np.ndarray) -> np.ndarray:
    """
    使用 medial_axis 生成骨架。

    注意：medial_axis 可能产生非单像素宽度的骨架。
    强烈建议使用 skeletonize_thin（skeletonize）代替，保证单像素宽度。
    此方法内部做了 thin 后处理尽力保证单像素。

    Args:
        mask: 二值 mask (H, W) uint8, 0/255

    Returns:
        skeleton: 二值骨架 (H, W) uint8, 0/255
    """
    from skimage.morphology import medial_axis, thin
    binary = mask > 0
    # 不使用 return_distance，直接获取骨架
    skel = medial_axis(binary, return_distance=False)
    # 确保单像素宽度
    try:
        skel = thin(skel)
    except Exception:
        pass
    return (skel * 255).astype(np.uint8)


def skeletonize_thin(mask: np.ndarray) -> np.ndarray:
    """
    使用 skeletonize（Zhang-Suen 细化）生成骨架，保证单像素宽度。

    Args:
        mask: 二值 mask (H, W) uint8, 0/255

    Returns:
        skeleton: 单像素宽二值骨架 (H, W) uint8, 0/255
    """
    from skimage.morphology import skeletonize
    binary = mask > 0
    skel = skeletonize(binary)
    return (skel * 255).astype(np.uint8)


# ===========================================================================
# 距离变换
# ===========================================================================

def compute_distance_transform(mask: np.ndarray) -> np.ndarray:
    """
    计算道路 mask 的距离变换（每个道路像素到最近边界的距离）。

    Args:
        mask: 二值 mask (H, W) uint8, 0/255

    Returns:
        dist: float32 距离图 (H, W)，非道路区域为 0
    """
    binary = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dist


# ===========================================================================
# 边界过滤
# ===========================================================================

def filter_boundary_points(
    skeleton: np.ndarray,
    dist_transform: np.ndarray,
    min_center_dist: float = 4.0,
) -> np.ndarray:
    """
    删除距离道路边界太近的 skeleton 点（即不在道路中心线上的点）。

    Args:
        skeleton:     二值骨架 (H, W) uint8, 0/255
        dist_transform: 距离变换图 (H, W) float32
        min_center_dist: 最小中心距离阈值

    Returns:
        过滤后的骨架
    """
    binary = skeleton > 0
    mask = dist_transform >= min_center_dist
    binary = binary & mask
    return (binary * 255).astype(np.uint8)


def resolve_center_distance_threshold(
    skeleton: np.ndarray,
    dist_transform: np.ndarray,
    requested_min_center_dist: float,
    min_keep_ratio: float = 0.55,
) -> float:
    """
    根据当前 mask 的道路宽度自适应中心距离阈值。

    当输入是细线式道路 mask 时，骨架点到边界的距离本来就只有 1-3px。
    固定使用 4px 会把真实道路大量删掉；宽道路面 mask 才需要严格中心过滤。

    Args:
        skeleton:                  二值骨架 (H, W) uint8
        dist_transform:            距离变换图 (H, W) float32
        requested_min_center_dist: 用户请求的最小中心距离
        min_keep_ratio:            允许保留的最小骨架像素比例

    Returns:
        自适应调整后的 min_center_dist
    """
    requested = float(requested_min_center_dist)
    if requested <= 1.0:
        return max(0.0, requested)

    vals = dist_transform[skeleton > 0]
    vals = vals[vals > 0]
    if vals.size == 0:
        return requested

    median_dist = float(np.percentile(vals, 50))
    if median_dist < requested * 1.2:
        return 1.0

    keep_ratio = float((vals >= requested).sum()) / max(vals.size, 1)
    if keep_ratio >= min_keep_ratio:
        return requested

    adaptive = float(np.percentile(vals, 100.0 * (1.0 - min_keep_ratio)))
    return max(1.0, min(requested, adaptive))


def filter_border_points(
    skeleton: np.ndarray,
    border_margin: int = 10,
) -> np.ndarray:
    """
    删除靠近图像边界的 skeleton 点。

    Args:
        skeleton:     二值骨架 (H, W) uint8, 0/255
        border_margin: 边界边距（像素）

    Returns:
        过滤后的骨架
    """
    binary = skeleton > 0
    h, w = binary.shape
    binary[:border_margin, :] = False
    binary[h - border_margin:, :] = False
    binary[:, :border_margin] = False
    binary[:, w - border_margin:] = False
    return (binary * 255).astype(np.uint8)


# ===========================================================================
# 毛刺删除（改进版）
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
        if 0 <= ny < h and 0 <= nx < w and binary[ny, nx]:
            cnt += 1
    return cnt


def _find_endpoints(binary: np.ndarray) -> List[Tuple[int, int]]:
    ys, xs = np.where(binary)
    endpoints = []
    for y, x in zip(ys, xs):
        if _count_neighbors(binary, int(y), int(x)) == 1:
            endpoints.append((int(y), int(x)))
    return endpoints


def _find_junctions(binary: np.ndarray) -> Set[Tuple[int, int]]:
    ys, xs = np.where(binary)
    junctions: Set[Tuple[int, int]] = set()
    for y, x in zip(ys, xs):
        if _count_neighbors(binary, int(y), int(x)) >= 3:
            junctions.add((int(y), int(x)))
    return junctions


def prune_short_branches(
    skeleton: np.ndarray,
    min_branch_length: int = 50,
) -> np.ndarray:
    """
    删除骨架上的短毛刺分支。

    从每个端点出发 BFS，追踪直到遇到交叉点或达到最小长度。

    Args:
        skeleton:         二值骨架 (H, W) uint8, 0/255
        min_branch_length: 最小保留分支长度（像素）

    Returns:
        删除短毛刺后的骨架
    """
    binary = skeleton > 0
    h, w = binary.shape

    endpoints = _find_endpoints(binary)
    junctions = _find_junctions(binary)

    to_remove = np.zeros_like(binary, dtype=bool)

    for ey, ex in endpoints:
        branch = []
        visited = set()
        cy, cx = ey, ex

        while True:
            if (cy, cx) in visited:
                break
            visited.add((cy, cx))
            branch.append((cy, cx))

            # 检查是否到达交叉点
            if (cy, cx) in junctions:
                break

            # 找下一个未访问邻居
            next_pts = []
            for dy, dx in _NEIGHBORS_8:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if binary[ny, nx] and (ny, nx) not in visited:
                        next_pts.append((ny, nx))

            if len(next_pts) == 0:
                break
            elif len(next_pts) == 1:
                cy, cx = next_pts[0]
            else:
                # 多分支，选第一个继续
                if len(branch) >= min_branch_length:
                    break
                cy, cx = next_pts[0]

        if 0 < len(branch) < min_branch_length:
            for py, px in branch:
                to_remove[py, px] = True

    binary[to_remove] = False
    return (binary * 255).astype(np.uint8)


# ===========================================================================
# Junction 合并
# ===========================================================================

def merge_nearby_junctions(
    skeleton: np.ndarray,
    merge_distance: int = 8,
) -> np.ndarray:
    """
    合并距离较近的 junction 节点。

    将互相邻接（距离 <= merge_distance）的 junction 像素用骨架线连接，
    使得后续图构建时能被聚类为一个节点。

    Args:
        skeleton:       二值骨架 (H, W) uint8, 0/255
        merge_distance: 合并距离（像素）

    Returns:
        合并后的骨架
    """
    binary = skeleton > 0
    junctions = list(_find_junctions(binary))

    if len(junctions) < 2:
        return skeleton

    # 找出所有需要连接的 junction 对
    j_set = set(junctions)

    # BFS 分组：距离 <= merge_distance 的 junction 归为一组
    visited = set()
    groups = []
    for j in junctions:
        if j in visited:
            continue
        group = []
        stack = [j]
        visited.add(j)
        while stack:
            cy, cx = stack.pop()
            group.append((cy, cx))
            # 搜索 merge_distance 范围内的其他 junction
            for dy in range(-merge_distance, merge_distance + 1):
                for dx in range(-merge_distance, merge_distance + 1):
                    ny, nx = cy + dy, cx + dx
                    if (ny, nx) in j_set and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        stack.append((ny, nx))
        groups.append(group)

    # 对每组 junction，用 Bresenham 线连接它们
    result = binary.copy()
    h, w = result.shape
    for group in groups:
        if len(group) <= 1:
            continue
        centroid_y = int(round(sum(p[0] for p in group) / len(group)))
        centroid_x = int(round(sum(p[1] for p in group) / len(group)))

        for py, px in group:
            # 画 Bresenham 线从 junction 到组质心
            rr, cc = _bresenham_line(py, px, centroid_y, centroid_x)
            for r, c in zip(rr, cc):
                if 0 <= r < h and 0 <= c < w:
                    result[r, c] = True

    return (result * 255).astype(np.uint8)


def _bresenham_line(y0: int, x0: int, y1: int, x1: int) -> Tuple[list, list]:
    """Bresenham 直线算法，返回 (row_indices, col_indices)。"""
    rows, cols = [], []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        rows.append(y0)
        cols.append(x0)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return rows, cols


# ===========================================================================
# 端点自动连接
# ===========================================================================

def auto_connect_endpoints(
    skeleton: np.ndarray,
    mask: np.ndarray,
    max_connect_dist: float = 45.0,
    max_connect_angle: float = 45.0,
    min_line_mask_overlap: float = 0.65,
) -> np.ndarray:
    """
    自动连接距离较近的断点（端点）。

    条件：
    1. 两个端点距离 <= max_connect_dist
    2. 两个端点方向夹角 <= max_connect_angle 度
    3. 连线经过 road mask 的比例 >= min_line_mask_overlap

    Args:
        skeleton:           二值骨架 (H, W) uint8, 0/255
        mask:               道路 mask (H, W) uint8, 0/255
        max_connect_dist:   最大连接距离
        max_connect_angle:  最大方向夹角（度）
        min_line_mask_overlap: 连线经过 mask 的最小比例

    Returns:
        连接后的骨架
    """
    binary = skeleton > 0
    h, w = binary.shape
    road_binary = mask > 0

    endpoints = _find_endpoints(binary)
    if len(endpoints) < 2:
        return skeleton

    # 计算每个端点的方向向量
    endpoint_dirs = {}
    for ey, ex in endpoints:
        d = _compute_endpoint_direction(binary, ey, ex)
        endpoint_dirs[(ey, ex)] = d

    # 构建 kd-tree 加速近邻搜索（简单版：距离预筛选）
    result = binary.copy()
    connected_pairs = set()

    max_cos_dev = np.cos(np.radians(max_connect_angle))

    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            if (i, j) in connected_pairs:
                continue

            p1 = endpoints[i]
            p2 = endpoints[j]
            dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            if dist > max_connect_dist:
                continue

            # 方向夹角检查
            dir1 = endpoint_dirs[p1]
            dir2 = endpoint_dirs[p2]
            if dir1 is not None and dir2 is not None:
                # 两方向向内连接时需反向其中一个方向
                cos_angle = abs(np.dot(dir1, dir2))
                if cos_angle < max_cos_dev:
                    continue

            # mask 重叠率检查
            rr, cc = _bresenham_line(p1[0], p1[1], p2[0], p2[1])
            if len(rr) == 0:
                continue
            mask_vals = []
            for r, c in zip(rr, cc):
                if 0 <= r < h and 0 <= c < w:
                    mask_vals.append(road_binary[r, c])
                else:
                    mask_vals.append(False)
            overlap = sum(mask_vals) / len(mask_vals) if mask_vals else 0
            if overlap < min_line_mask_overlap:
                continue

            # 通过检查，连接两个端点
            for r, c in zip(rr, cc):
                if 0 <= r < h and 0 <= c < w:
                    result[r, c] = True

            connected_pairs.add((i, j))
            connected_pairs.add((j, i))

    return (result * 255).astype(np.uint8)


def _compute_endpoint_direction(
    binary: np.ndarray, y: int, x: int
) -> Optional[np.ndarray]:
    """
    计算端点在骨架上的切线方向（从端点指向其唯一邻居）。

    Args:
        binary: 骨架二值数组
        y, x:   端点坐标

    Returns:
        归一化方向向量 (dy, dx) 或 None
    """
    h, w = binary.shape
    neighbors = []
    for dy, dx in _NEIGHBORS_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and binary[ny, nx]:
            neighbors.append((ny, nx))

    if not neighbors:
        return None

    # 取第一个邻居方向
    ny, nx = neighbors[0]
    vec = np.array([ny - y, nx - x], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return None
    return vec / norm


# ===========================================================================
# 可视化
# ===========================================================================

def draw_optimized_overlay(
    image_rgb: np.ndarray,
    skeleton: np.ndarray,
    endpoints: Optional[List[Tuple[int, int]]] = None,
    junctions: Optional[List[Tuple[int, int]]] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    将优化后的骨架叠加到原图上。

    Args:
        image_rgb: 原始 RGB 图像
        skeleton:  二值骨架
        endpoints: 端点列表
        junctions: 交叉点列表
        mask:      道路 mask（可选，用于显示边界验证对齐）

    Returns:
        叠加后的 RGB 图像
    """
    from skimage.morphology import skeletonize as skel_skimage

    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # ---- 半透明 mask 边界（青色虚线效果） ----
    if mask is not None:
        binary = (mask > 0).astype(np.uint8)
        # 找到 mask 边界
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.drawContours(img, cnt, -1, (255, 255, 0), 1)  # 青色细线

    # 骨架：亮黄色实线（比浅灰更显眼）
    binary_skel = skeleton > 0
    # 细化骨架线以生成更锐利的线
    try:
        thin_skel = skel_skimage(binary_skel)
    except Exception:
        thin_skel = binary_skel
    img[thin_skel] = (0, 255, 255)  # 亮黄色

    # 端点绿色
    if endpoints:
        for y, x in endpoints:
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(img, (x, y), 7, (0, 200, 0), 1)

    # 交叉点红色
    if junctions:
        for y, x in junctions:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.circle(img, (x, y), 7, (0, 0, 200), 1)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ===========================================================================
# Junction 像素聚类
# ===========================================================================

def _cluster_junction_pixels(
    junction_pixels: List[Tuple[int, int]],
    cluster_radius: int = 10,
) -> List[dict]:
    """
    将距离 <= cluster_radius 的 junction 像素聚类为 junction cluster。

    Args:
        junction_pixels: [(y,x), ...] 所有 junction 像素
        cluster_radius: 聚类半径（像素）

    Returns:
        [{"centroid_y": int, "centroid_x": int, "pixel_count": int}, ...]
    """
    if not junction_pixels:
        return []

    j_set = set(junction_pixels)
    visited = set()
    clusters = []

    for py, px in junction_pixels:
        if (py, px) in visited:
            continue
        group = []
        stack = [(py, px)]
        visited.add((py, px))
        while stack:
            cy, cx = stack.pop()
            group.append((cy, cx))
            for dy in range(-cluster_radius, cluster_radius + 1):
                for dx in range(-cluster_radius, cluster_radius + 1):
                    ny, nx = cy + dy, cx + dx
                    if (ny, nx) in j_set and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        stack.append((ny, nx))

        if group:
            centroid_y = int(round(sum(p[0] for p in group) / len(group)))
            centroid_x = int(round(sum(p[1] for p in group) / len(group)))
            clusters.append({
                "centroid_y": centroid_y,
                "centroid_x": centroid_x,
                "pixel_count": len(group),
                "pixels": group,
            })

    # 按像素数降序排列
    clusters.sort(key=lambda c: c["pixel_count"], reverse=True)
    return clusters


# ===========================================================================
# 优化对比图
# ===========================================================================

def draw_skeleton_compare(
    image_rgb: np.ndarray,
    raw_skeleton: np.ndarray,
    optimized_skeleton: np.ndarray,
    junction_clusters: Optional[List[dict]] = None,
    endpoints: Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray:
    """
    绘制 raw skeleton（左）与 optimized skeleton（右）并排对比图。

    Args:
        image_rgb:          原始 RGB 图像
        raw_skeleton:       原始骨架
        optimized_skeleton: 优化后骨架
        junction_clusters:  路口聚类列表 [{"centroid_y","centroid_x",...},...]
        endpoints:          优化后的端点列表

    Returns:
        对比拼接 RGB 图像 (H, 2W, 3)
    """
    from skimage.morphology import skeletonize as skel_skimage

    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    # 左图：原始骨架
    left = img.copy()
    try:
        thin_raw = skel_skimage(raw_skeleton > 0)
    except Exception:
        thin_raw = raw_skeleton > 0
    left[thin_raw] = (0, 255, 255)

    # 右图：优化后骨架
    right = img.copy()
    try:
        thin_opt = skel_skimage(optimized_skeleton > 0)
    except Exception:
        thin_opt = optimized_skeleton > 0
    right[thin_opt] = (0, 255, 255)

    # 端点绿色
    if endpoints:
        for y, x in endpoints:
            cv2.circle(right, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(right, (x, y), 7, (0, 200, 0), 1)

    # 路口红色
    if junction_clusters:
        for jc in junction_clusters:
            cv2.circle(right, (jc["centroid_x"], jc["centroid_y"]),
                       7, (0, 0, 255), -1)
            cv2.circle(right, (jc["centroid_x"], jc["centroid_y"]),
                       9, (0, 0, 200), 1)

    # 拼接
    compare = np.hstack([left, right])
    return cv2.cvtColor(compare, cv2.COLOR_BGR2RGB)


# ===========================================================================
# V4.1: optimize_skeleton — 完整的骨架优化函数
# ===========================================================================

def optimize_skeleton(
    mask: np.ndarray,
    skeleton: np.ndarray,
    min_center_dist: float = 2.0,
    border_margin: int = 10,
    min_branch_length: int = 20,
    max_connect_dist: float = 25.0,
    max_connect_angle: float = 45.0,
    min_line_mask_overlap: float = 0.65,
    junction_cluster_radius: int = 10,
) -> dict:
    """
    对现有 raw skeleton 执行完整优化流水线。

    Args:
        mask:                    道路 mask (H,W) uint8 0/255
        skeleton:                原始骨架 (H,W) uint8 0/255
        min_center_dist:         最小道路中心距离
        border_margin:           图像边界留白
        min_branch_length:       最小分支长度
        max_connect_dist:        断点自动连接的最大距离
        max_connect_angle:       断点连接的最大方向夹角
        min_line_mask_overlap:   连线经过 mask 的最小比例
        junction_cluster_radius: 路口像素聚类半径

    Returns:
        dict
    """
    # 1. 输入标准化
    mask = normalize_road_mask(mask)
    skeleton = ((skeleton if skeleton.dtype == np.uint8 else skeleton.astype(np.uint8)) > 0).astype(np.uint8) * 255

    raw_skeleton = skeleton.copy()
    raw_binary = raw_skeleton > 0
    raw_pixels = int(raw_binary.sum())
    raw_endpoints_count = len(_find_endpoints(raw_binary))
    raw_junction_pixels_count = len(_find_junctions(raw_binary))

    print(f"[OPTIMIZE] raw: pixels={raw_pixels}, endpoints={raw_endpoints_count}, "
          f"junction_pixels={raw_junction_pixels_count}")

    # 2. 删除图像边界附近 skeleton
    print(f"[OPTIMIZE] Step 1/7: 删除图像边界 {border_margin}px 内的骨架点...")
    skeleton = filter_border_points(skeleton, border_margin)

    # 3. distance transform 过滤
    print(f"[OPTIMIZE] Step 2/7: distance transform 过滤 (min_center_dist={min_center_dist}px)...")
    dist = compute_distance_transform(mask)
    effective_min_center_dist = resolve_center_distance_threshold(
        skeleton, dist, min_center_dist
    )
    if effective_min_center_dist != min_center_dist:
        print(f"[OPTIMIZE]   当前 mask 较细，中心距离阈值自适应为 "
              f"{effective_min_center_dist:.2f}px")
    skeleton = filter_boundary_points(skeleton, dist, effective_min_center_dist)

    # 4. 删除短毛刺 (第1轮)
    print(f"[OPTIMIZE] Step 3/7: 删除短毛刺 第1轮 (min_branch_length={min_branch_length}px)...")
    before_prune = (skeleton > 0).sum()
    skeleton = prune_short_branches(skeleton, min_branch_length)
    after_prune = (skeleton > 0).sum()
    spur_count_1 = int(before_prune - after_prune)
    print(f"[OPTIMIZE]   第1轮删除毛刺像素数: {spur_count_1}")

    # 5. junction 像素检测
    print(f"[OPTIMIZE] Step 4/7: 检测 junction 像素...")
    junction_pixels = list(_find_junctions(skeleton > 0))
    jp_count = len(junction_pixels)
    print(f"[OPTIMIZE]   junction 像素数: {jp_count}")

    # 6. junction 像素聚类
    print(f"[OPTIMIZE] Step 5/7: 聚类 junction 像素 (radius={junction_cluster_radius}px)...")
    junction_clusters = _cluster_junction_pixels(junction_pixels, junction_cluster_radius)
    cluster_count = len(junction_clusters)
    print(f"[OPTIMIZE]   聚类后路口数: {cluster_count} (从 {jp_count} 个像素)")

    # 7. 自动连接断点
    print(f"[OPTIMIZE] Step 6/7: 自动连接断点 "
          f"(max_dist={max_connect_dist}px, max_angle={max_connect_angle}°, "
          f"mask_overlap>={min_line_mask_overlap})...")
    before_connect = (skeleton > 0).sum()
    skeleton = auto_connect_endpoints(
        skeleton, mask,
        max_connect_dist=max_connect_dist,
        max_connect_angle=max_connect_angle,
        min_line_mask_overlap=min_line_mask_overlap,
    )
    after_connect = (skeleton > 0).sum()
    connected_gap_count = int(after_connect - before_connect)
    print(f"[OPTIMIZE]   连接增加的像素数: {connected_gap_count}")

    # 8. 再次删除短毛刺 (第2轮)
    print(f"[OPTIMIZE] Step 7/7: 删除短毛刺 第2轮...")
    before_prune2 = (skeleton > 0).sum()
    skeleton = prune_short_branches(skeleton, min_branch_length)
    after_prune2 = (skeleton > 0).sum()
    spur_count_2 = int(before_prune2 - after_prune2)
    print(f"[OPTIMIZE]   第2轮删除毛刺像素数: {spur_count_2}")

    # 最终统计
    optimized_binary = skeleton > 0
    optimized_pixels = int(optimized_binary.sum())
    optimized_endpoints = _find_endpoints(optimized_binary)
    optimized_junction_pixels = _find_junctions(optimized_binary)
    opt_jp_count = len(optimized_junction_pixels)
    opt_ep_count = len(optimized_endpoints)

    stats = {
        "raw_pixels": raw_pixels,
        "optimized_pixels": optimized_pixels,
        "raw_endpoints": raw_endpoints_count,
        "optimized_endpoints": opt_ep_count,
        "raw_junction_pixels": raw_junction_pixels_count,
        "optimized_junction_pixels": opt_jp_count,
        "junction_cluster_count": cluster_count,
        "removed_spur_count": spur_count_1 + spur_count_2,
        "connected_gap_count": connected_gap_count,
        "effective_min_center_dist": round(float(effective_min_center_dist), 2),
    }

    print(f"[OPTIMIZE] === 优化完成 ===")
    for k, v in stats.items():
        print(f"[OPTIMIZE]   {k}: {v}")

    return {
        "raw_skeleton": raw_skeleton,
        "optimized_skeleton": skeleton,
        "distance_map": dist,
        "endpoints": list(optimized_endpoints),
        "junction_pixels": junction_pixels,
        "junction_clusters": junction_clusters,
        "stats": stats,
    }


# ===========================================================================
# 主入口（向后兼容）
# ===========================================================================

def run_optimized_skeleton(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    output_dir: str,
    config: Optional[Dict] = None,
) -> np.ndarray:
    """
    运行完整的 skeleton 优化流水线。

    Args:
        image_rgb:  原始 RGB 图像
        mask:       道路 mask (H, W) uint8
        output_dir: 输出目录
        config:     配置字典

    Returns:
        优化后的骨架 (H, W) uint8
    """
    if config is None:
        config = {}
    skel_cfg = config.get("skeleton", {})

    method = skel_cfg.get("method", "medial_axis")
    min_center_dist = skel_cfg.get("min_center_dist", 2.0)
    border_margin = skel_cfg.get("border_margin", 10)
    min_branch_length = skel_cfg.get("min_branch_length", 20)
    max_connect_dist = skel_cfg.get("max_connect_dist", 25)
    max_connect_angle = skel_cfg.get("max_connect_angle", 45)
    min_line_mask_overlap = skel_cfg.get("min_line_mask_overlap", 0.65)

    mask = normalize_road_mask(mask)

    # Step 1: 距离变换
    print("[SKEL] Step 1/7: 计算距离变换...")
    dist = compute_distance_transform(mask)

    # Step 2: 骨架化
    print(f"[SKEL] Step 2/7: 生成初始骨架（方法={method}）...")
    if method == "medial_axis":
        skeleton_raw = skeletonize_medial_axis(mask)
    else:
        skeleton_raw = skeletonize_thin(mask)

    raw_path = os.path.join(output_dir, "road_skeleton_raw.png")
    cv2.imwrite(raw_path, skeleton_raw)
    print(f"[SKEL] 已保存原始骨架: {raw_path}")

    # Step 3: 过滤靠近边界的点
    print(f"[SKEL] Step 3/7: 过滤距离中心 < {min_center_dist}px 的骨架点...")
    skeleton = filter_boundary_points(skeleton_raw, dist, min_center_dist)

    # Step 4: 过滤靠近图像边界的点
    print(f"[SKEL] Step 4/7: 裁剪图像边界 {border_margin}px 内的骨架点...")
    skeleton = filter_border_points(skeleton, border_margin)

    # Step 5: 删除短毛刺
    print(f"[SKEL] Step 5/7: 删除短毛刺（最小分支长度={min_branch_length}px）...")
    skeleton = prune_short_branches(skeleton, min_branch_length)

    # Step 6: 合并邻近 junction 节点
    print(f"[SKEL] Step 6/7: 合并邻近 junction 节点...")
    skeleton = merge_nearby_junctions(skeleton, merge_distance=8)

    # Step 7: 自动连接近距离断点
    print(f"[SKEL] Step 7/7: 自动连接断点 "
          f"(最大距离={max_connect_dist}px, 最大夹角={max_connect_angle}°, "
          f"mask重叠率>={min_line_mask_overlap})...")
    skeleton = auto_connect_endpoints(
        skeleton, mask,
        max_connect_dist=max_connect_dist,
        max_connect_angle=max_connect_angle,
        min_line_mask_overlap=min_line_mask_overlap,
    )

    # ---- 保存 ---- 
    opt_path = os.path.join(output_dir, "road_skeleton_optimized.png")
    cv2.imwrite(opt_path, skeleton)
    print(f"[SKEL] 已保存优化骨架: {opt_path}")

    # 统计信息
    binary = skeleton > 0
    endpoints = _find_endpoints(binary)
    junctions_list = list(_find_junctions(binary))
    skel_pixels = int(skeleton.sum() / 255)
    print(f"[SKEL] 骨架统计: 像素数={skel_pixels}px, "
          f"端点={len(endpoints)}个, 交叉点={len(junctions_list)}个")

    # 保存叠加图
    overlay = draw_optimized_overlay(image_rgb, skeleton, endpoints, junctions_list, mask=mask)
    overlay_path = os.path.join(output_dir, "road_skeleton_optimized_overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"[SKEL] 已保存优化骨架叠加图: {overlay_path}")
    print(f"[SKEL]   - 黄色线 = 骨架中心线")
    print(f"[SKEL]   - 青色线 = mask 边界")
    print(f"[SKEL]   - 绿色圆点 = 端点 (endpoint)")
    print(f"[SKEL]   - 红色圆点 = 交叉点 (junction)")

    return skeleton
