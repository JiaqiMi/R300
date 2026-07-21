"""
从 processed_mask 生成 clean_skeleton 模块。

功能：
1. skeletonize/thinning 骨架化
2. 删除短毛刺
3. 可选端点连接（含方向约束，默认不连接避免乱连）
4. 边界留白处理

输出：clean_skeleton (0/255 单像素宽二值图)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2


@dataclass
class SkeletonConfig:
    """Skeleton 生成配置"""
    # 骨架化方法: "thin" (skeletonize, 推荐) 或 "medial_axis"
    method: str = "thin"
    # 短枝剪除最小长度（像素，0 = 不剪枝）
    prune_length: int = 20
    # 端点自动连接最大距离（像素，0 = 不连接）
    connect_endpoint_distance: int = 0
    # 端点自动连接最大方向夹角（度）
    connect_angle_threshold: float = 30.0
    # 连接线 mask 重叠最小比例（0-1）
    connect_mask_overlap: float = 0.65
    # 图像边界留白（像素）
    border_margin: int = 5


# ===========================================================================
# 骨架化
# ===========================================================================

def skeletonize(mask: np.ndarray, method: str = "thin") -> np.ndarray:
    """
    对二值 mask 进行骨架化。

    Args:
        mask:   二值 mask (H, W) uint8, 0/255
        method: "thin" 或 "medial_axis"

    Returns:
        skeleton (H, W) uint8, 0/255
    """
    from skimage.morphology import skeletonize as skel_func

    binary = mask > 0
    if method == "medial_axis":
        from skimage.morphology import medial_axis, thin
        skel = medial_axis(binary, return_distance=False)
        try:
            skel = thin(skel)
        except Exception:
            pass
        return (skel * 255).astype(np.uint8)
    else:
        skel = skel_func(binary)
        return (skel * 255).astype(np.uint8)


# ===========================================================================
# 短枝剪除
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


def _find_junctions(binary: np.ndarray) -> set:
    ys, xs = np.where(binary)
    junctions = set()
    for y, x in zip(ys, xs):
        if _count_neighbors(binary, int(y), int(x)) >= 3:
            junctions.add((int(y), int(x)))
    return junctions


def prune_branches(
    skeleton: np.ndarray,
    min_branch_length: int = 20,
) -> np.ndarray:
    """
    删除骨架上的短毛刺分支。

    从每个端点出发 BFS，追踪直到遇到交叉点或达到最小长度。

    Args:
        skeleton:         二值骨架 (H, W) uint8, 0/255
        min_branch_length: 最小保留分支长度（像素）

    Returns:
        剪枝后的骨架
    """
    if min_branch_length <= 0:
        return skeleton

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

            if (cy, cx) in junctions:
                break

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
                if len(branch) >= min_branch_length:
                    break
                cy, cx = next_pts[0]

        if 0 < len(branch) < min_branch_length:
            for py, px in branch:
                to_remove[py, px] = True

    binary[to_remove] = False
    return (binary * 255).astype(np.uint8)


# ===========================================================================
# 端点连接（方向约束）
# ===========================================================================

def _compute_endpoint_direction(
    binary: np.ndarray, y: int, x: int,
) -> Optional[np.ndarray]:
    """计算端点在骨架上的切线方向（从端点指向邻居）。"""
    h, w = binary.shape
    for dy, dx in _NEIGHBORS_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and binary[ny, nx]:
            vec = np.array([ny - y, nx - x], dtype=np.float64)
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                return vec / norm
    return None


def _bresenham_line(y0: int, x0: int, y1: int, x1: int) -> Tuple[List[int], List[int]]:
    """Bresenham 直线算法。"""
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


def connect_endpoints(
    skeleton: np.ndarray,
    mask: np.ndarray,
    max_dist: float = 10.0,
    max_angle: float = 30.0,
    min_mask_overlap: float = 0.65,
) -> np.ndarray:
    """
    自动连接近距离断点（带方向和 mask 约束）。

    只连接同时满足的端点对：
    1. 距离 <= max_dist
    2. 方向夹角 <= max_angle（单位：度）
    3. 连线经过 mask 的比例 >= min_mask_overlap

    Args:
        skeleton:         二值骨架
        mask:             道路 mask (用于验证连线)
        max_dist:         最大连接距离（像素）
        max_angle:        最大方向夹角（度）
        min_mask_overlap: 连线 mask 重叠率下限

    Returns:
        连接后的骨架
    """
    if max_dist <= 0:
        return skeleton

    binary = skeleton > 0
    h, w = binary.shape
    road_binary = mask > 0
    endpoints = _find_endpoints(binary)
    if len(endpoints) < 2:
        return skeleton

    # 计算端点方向
    dirs = {}
    for e in endpoints:
        d = _compute_endpoint_direction(binary, e[0], e[1])
        dirs[e] = d

    max_cos = np.cos(np.radians(max_angle))
    result = binary.copy()
    connected = set()

    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            if (i, j) in connected:
                continue
            p1, p2 = endpoints[i], endpoints[j]
            dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            if dist > max_dist:
                continue

            d1, d2 = dirs.get(p1), dirs.get(p2)
            if d1 is not None and d2 is not None:
                cos_a = abs(np.dot(d1, d2))
                if cos_a < max_cos:
                    continue

            rr, cc = _bresenham_line(p1[0], p1[1], p2[0], p2[1])
            if not rr:
                continue

            valid = 0
            for r, c in zip(rr, cc):
                if 0 <= r < h and 0 <= c < w and road_binary[r, c]:
                    valid += 1
            if valid / len(rr) < min_mask_overlap:
                continue

            for r, c in zip(rr, cc):
                if 0 <= r < h and 0 <= c < w:
                    result[r, c] = True
            connected.add((i, j))
            connected.add((j, i))

    return (result * 255).astype(np.uint8)


# ===========================================================================
# 完整流水线
# ===========================================================================

def generate_skeleton(
    mask: np.ndarray,
    config: Optional[SkeletonConfig] = None,
) -> np.ndarray:
    """
    从 mask 生成 clean skeleton 的完整流水线。

    Args:
        mask:   二值 mask (H, W) uint8, 0/255
        config: Skeleton 配置

    Returns:
        clean_skeleton (H, W) uint8, 0/255
    """
    if config is None:
        config = SkeletonConfig()

    # Step 1: 骨架化
    skeleton = skeletonize(mask, method=config.method)

    # Step 2: 边界留白
    if config.border_margin > 0:
        m = config.border_margin
        h, w = skeleton.shape
        skeleton[:m, :] = 0
        skeleton[h - m:, :] = 0
        skeleton[:, :m] = 0
        skeleton[:, w - m:] = 0

    # Step 3: 短枝剪除
    skeleton = prune_branches(skeleton, min_branch_length=config.prune_length)

    # Step 4: 端点连接（可选，默认关闭）
    if config.connect_endpoint_distance > 0:
        skeleton = connect_endpoints(
            skeleton, mask,
            max_dist=config.connect_endpoint_distance,
            max_angle=config.connect_angle_threshold,
            min_mask_overlap=config.connect_mask_overlap,
        )
        # 连接后可能有新毛刺，再剪一次
        skeleton = prune_branches(skeleton, min_branch_length=config.prune_length)

    return skeleton
