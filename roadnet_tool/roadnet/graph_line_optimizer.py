"""
Graph polyline 线形优化模块。

对 final_graph 中每条 edge 的 polyline（points_pixel）进行：
- 重复点/过近点去除
- RDP (Ramer-Douglas-Peucker) 折线简化
- 近似直线拉直（保留首尾点，移除中间点）
- 弯路轻微平滑（moving average，首尾点不动）
- 基于 processed_mask 的合法性校验（distanceTransform）

工作于 GraphEditorQt 的边数据格式：
    edge = {"id", "start", "end", "length_pixel", "points_pixel": [[x,y],...],
            "source", "enabled"}

只修改 edge["points_pixel"] 和 edge["length_pixel"]。
保持 start/end 节点不变，保持 graph 拓扑不变。
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from roadnet.graph_utils import as_bool, polyline_to_list, ensure_python_types


# ===========================================================================
# 配置
# ===========================================================================

@dataclass
class GraphLineOptimizeConfig:
    """Graph 线形优化配置。"""
    rdp_epsilon: float = 3.0
    straight_max_deviation: float = 6.0
    min_straight_edge_length: float = 30.0
    smooth_window: int = 5
    max_smooth_offset: float = 4.0
    mask_tolerance: float = 5.0
    preserve_junctions: bool = True
    validate_with_mask: bool = True


DEFAULT_CONFIG = GraphLineOptimizeConfig()


# ===========================================================================
# RDP 简化 (Ramer-Douglas-Peucker)
# ===========================================================================

def _rdp_simplify(points, epsilon: float):
    """
    Ramer-Douglas-Peucker 折线简化。

    保留首尾点，保留关键转折点。
    不会把真实弯道简化为直线（因为弯道中间会有满足 epsilon 的关键点）。

    Args:
        points: [[x, y], ...] 格式的点列表（至少 2 个点）
        epsilon: 最大允许垂直距离

    Returns:
        [[x, y], ...] 简化后的点列表
    """
    if len(points) <= 2:
        return list(points)

    pts = np.array(points, dtype=np.float64)
    start = pts[0]
    end = pts[-1]
    vec = end - start
    vec_norm = float(np.linalg.norm(vec))

    if vec_norm < 1e-8:
        # 首尾重合 — 保留首尾点
        return [points[0], points[-1]]

    # 找所有中间点到首尾连线的最大垂直距离
    dmax = 0.0
    index = 0
    for i in range(1, len(pts) - 1):
        # 叉积法计算点到线段的垂直距离
        cross = abs(float(vec[0]) * float(start[1] - pts[i][1])
                    - float(vec[1]) * float(start[0] - pts[i][0]))
        d = cross / vec_norm
        if d > dmax:
            dmax = d
            index = i

    if dmax > epsilon:
        # 该点是关键拐点，递归拆分
        left = _rdp_simplify(points[:index + 1], epsilon)
        right = _rdp_simplify(points[index:], epsilon)
        return left[:-1] + right
    else:
        # 所有中间点都在容差内 — 保留首尾点
        return [points[0], points[-1]]


# ===========================================================================
# 点清理
# ===========================================================================

def _remove_duplicates_and_close_points(points, min_dist: float = 1.0):
    """
    去除重复点和距离过近的点。
    保留首尾点不动，只清理中间点。

    Args:
        points: [[x, y], ...]
        min_dist: 两点之间的最小距离阈值（像素）

    Returns:
        [[x, y], ...]
    """
    if len(points) <= 2:
        return list(points)

    cleaned = [points[0]]
    for i in range(1, len(points) - 1):
        prev = cleaned[-1]
        dx = points[i][0] - prev[0]
        dy = points[i][1] - prev[1]
        if math.sqrt(dx * dx + dy * dy) >= min_dist:
            cleaned.append(list(points[i]))

    # 始终保留最后一个点
    if len(points) > 1:
        cleaned.append(list(points[-1]))

    return cleaned


# ===========================================================================
# 直线判断与拉直
# ===========================================================================

def _compute_polyline_length(points) -> float:
    """计算 polyline 的总长度（像素）。"""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        total += math.sqrt(dx * dx + dy * dy)
    return total


def _is_straight_line(points, max_deviation: float = 6.0, min_length: float = 30.0):
    """
    判断 polyline 是否可以视为近似直线。

    计算所有中间点到首尾连线的最大垂直距离。

    Args:
        points: [[x, y], ...]
        max_deviation: 最大允许偏离距离（像素）
        min_length: 最小 edge 长度，太短不拉直

    Returns:
        (is_straight: bool, max_dev: float, length: float)
    """
    if len(points) <= 2:
        return True, 0.0, _compute_polyline_length(points)

    start = points[0]
    end = points[-1]

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.sqrt(dx * dx + dy * dy)

    if length < min_length:
        return False, 0.0, length

    # 计算所有中间点的最大垂直偏离
    max_dev = 0.0
    for i in range(1, len(points) - 1):
        cross = abs(dx * (start[1] - points[i][1]) - dy * (start[0] - points[i][0]))
        d = cross / length
        if d > max_dev:
            max_dev = d

    return max_dev <= max_deviation, max_dev, length


def _straighten_edge(points):
    """
    拉直 polyline：保留首尾点，移除所有中间点。
    不移动交叉口节点和端点。
    """
    return [list(points[0]), list(points[-1])]


# ===========================================================================
# 弯路平滑 (Moving Average)
# ===========================================================================

def _smooth_polyline(points, window: int = 5, max_offset: float = 4.0):
    """
    对 polyline 做轻微平滑（moving average）。

    首尾点不动。
    平滑后每个点相对原始位置的偏移不超过 max_offset。
    如果偏移超限，则 clamp 到 max_offset 方向上的点。

    Args:
        points: [[x, y], ...]
        window: 平滑窗口大小
        max_offset: 最大允许偏移（像素）

    Returns:
        [[x, y], ...] 平滑后的点列表
    """
    if len(points) <= 2:
        return list(points)

    n = len(points)
    smoothed = [list(points[0])]  # 首点不动

    for i in range(1, n - 1):
        # 确定平滑窗口范围
        half = window // 2
        start_idx = max(1, i - half)
        end_idx = min(n - 1, i + half + 1)

        # 窗口内取平均
        sum_x = sum(points[j][0] for j in range(start_idx, end_idx))
        sum_y = sum(points[j][1] for j in range(start_idx, end_idx))
        count = end_idx - start_idx
        avg_x = sum_x / count
        avg_y = sum_y / count

        # 检查偏移约束
        ox = points[i][0]
        oy = points[i][1]
        offset = math.sqrt((avg_x - ox) ** 2 + (avg_y - oy) ** 2)

        if offset > max_offset and offset > 1e-8:
            # Clamp 到 max_offset
            ratio = max_offset / offset
            avg_x = ox + (avg_x - ox) * ratio
            avg_y = oy + (avg_y - oy) * ratio

        smoothed.append([avg_x, avg_y])

    smoothed.append(list(points[-1]))  # 尾点不动
    return smoothed


# ===========================================================================
# Mask 安全检查 (distanceTransform)
# ===========================================================================

def _validate_with_mask(points, mask, dist_map, mask_tolerance: float = 5.0) -> bool:
    """
    检查优化后的 polyline 是否仍在道路区域附近。

    对每个点：
    - 如果点在 mask 内（道路区域）→ 通过
    - 如果点不在 mask 内，但距最近道路区域 <= mask_tolerance → 通过
    - 否则 → 失败

    Args:
        points: [[x, y], ...] 优化后的点列表
        mask: (H, W) bool ndarray，True=道路
        dist_map: (H, W) float32，每个像素到最近道路区域的欧氏距离
        mask_tolerance: 距离容差（像素）

    Returns:
        bool: True 表示所有采样点合法
    """
    h, w = mask.shape

    for pt in points:
        x = int(round(pt[0]))
        y = int(round(pt[1]))

        # Clamp 到图像边界
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))

        if as_bool(mask[y, x]):
            continue  # 在道路区域内 → 通过

        # 不在道路区域内，检查到最近道路的距离
        if as_bool(dist_map[y, x] > mask_tolerance):
            return False

    return True


# ===========================================================================
# 主优化流程
# ===========================================================================

def optimize_graph_lines(
    edges: List[Dict],
    processed_mask: Optional[np.ndarray] = None,
    config: Optional[GraphLineOptimizeConfig] = None,
) -> Tuple[List[Dict], Dict]:
    """
    对 final_graph 中每条 enabled edge 的 polyline 进行线形优化。

    每条 edge 的处理流程：
    1. 去掉重复点和过近点
    2. 计算 edge 长度，太短跳过
    3. RDP 简化折线
    4. 判断是否接近直线 → 是则拉直，否则轻微平滑
    5. Mask 安全检查 → 失败则回退原 polyline
    6. 记录统计数据

    **关键约束**：
    - 不移动 edge 的起点和终点
    - 不改变 edge 的 start / end 节点引用
    - 不改变 graph 拓扑
    - 只替换 edge.points_pixel

    Args:
        edges: GraphEditorQt 格式的边列表
        processed_mask: (H, W) bool ndarray（True=道路），用于校验，可选
        config: 优化参数

    Returns:
        (optimized_edges, report)
        - optimized_edges: List[Dict] — 深复制的新边列表
        - report: Dict — 统计报告
    """
    if config is None:
        config = DEFAULT_CONFIG

    # ── 准备 mask 校验资源 ──
    dist_map = None
    mask_bool = None
    if config.validate_with_mask and processed_mask is not None:
        mask_bool = processed_mask.astype(bool) if processed_mask.dtype != bool else processed_mask
        # distanceTransform: 对非道路区域计算到最近道路区域的欧氏距离
        non_road = (~mask_bool).astype(np.uint8)
        dist_map = cv2.distanceTransform(non_road, cv2.DIST_L2, 3)

    # ── 深复制边数据 ──
    optimized = copy.deepcopy(edges)

    # ── 统计计数器 ──
    total_edges = 0
    successful = 0
    straightened = 0
    smoothed_edges = 0
    mask_rollback = 0
    skipped_short = 0
    total_points_before = 0
    total_points_after = 0
    max_deviation_recorded = 0.0
    per_edge_stats = []

    for edge in optimized:
        # 只处理 enabled 边
        if not edge.get("enabled", True):
            continue

        total_edges += 1
        original_points = copy.deepcopy(edge.get("points_pixel", []))

        if len(original_points) < 2:
            total_points_before += len(original_points)
            total_points_after += len(original_points)
            continue

        n_before = len(original_points)
        total_points_before += n_before

        # ── Step 1: 去除重复点和过近点 ──
        cleaned = _remove_duplicates_and_close_points(original_points, min_dist=1.0)
        if len(cleaned) < 2:
            # 清理后不够 2 个点，保留原始
            cleaned = list(original_points)

        # ── Step 2: 计算 edge 长度，太短跳过 ──
        edge_len = _compute_polyline_length(cleaned)
        if edge_len < config.min_straight_edge_length * 0.3:
            skipped_short += 1
            total_points_after += n_before
            per_edge_stats.append({
                "edge_id": edge["id"],
                "action": "skipped_short",
                "points_before": n_before,
                "points_after": n_before,
                "max_deviation": 0.0,
                "edge_length": round(edge_len, 2),
            })
            continue

        # ── Step 3: RDP 简化 ──
        simplified = _rdp_simplify(cleaned, config.rdp_epsilon)

        # ── Step 4: 判断是否接近直线 ──
        is_straight, max_dev, full_length = _is_straight_line(
            simplified,
            max_deviation=config.straight_max_deviation,
            min_length=config.min_straight_edge_length,
        )
        max_deviation_recorded = max(max_deviation_recorded, max_dev)

        # ── Step 5: 拉直 或 平滑 ──
        if is_straight:
            optimized_points = _straighten_edge(simplified)
            action = "straightened"
            straightened += 1
        else:
            optimized_points = _smooth_polyline(
                simplified,
                window=config.smooth_window,
                max_offset=config.max_smooth_offset,
            )
            action = "smoothed"
            smoothed_edges += 1

        # ── Step 6: Mask 安全检查 ──
        passed_mask_check = True
        if config.validate_with_mask and mask_bool is not None and dist_map is not None:
            passed_mask_check = _validate_with_mask(
                optimized_points, mask_bool, dist_map, config.mask_tolerance
            )

        # ── Step 7: 失败则回退 ──
        if not passed_mask_check:
            optimized_points = original_points
            mask_rollback += 1
            action = "mask_rollback"

        # ── Step 8: 写回边数据，确保 points_pixel 是纯 Python list ──
        edge["points_pixel"] = polyline_to_list(optimized_points)
        edge["length_pixel"] = round(_compute_polyline_length(optimized_points), 2)

        n_after = len(optimized_points)
        total_points_after += n_after

        if action != "mask_rollback":
            successful += 1

        per_edge_stats.append({
            "edge_id": edge["id"],
            "action": action,
            "points_before": n_before,
            "points_after": n_after,
            "max_deviation": round(max_dev, 2),
            "edge_length": round(full_length, 2),
        })

    # ── 构建报告 ──
    points_reduction = 0.0
    if total_points_before > 0:
        points_reduction = round(
            (1.0 - total_points_after / max(total_points_before, 1)) * 100, 1
        )

    report = {
        "summary": {
            "total_edges": total_edges,
            "successful_edges": successful,
            "straightened_edges": straightened,
            "smoothed_edges": smoothed_edges,
            "mask_rollback_edges": mask_rollback,
            "skipped_short_edges": skipped_short,
            "total_points_before": total_points_before,
            "total_points_after": total_points_after,
            "points_reduction_pct": points_reduction,
            "max_deviation_recorded": round(max_deviation_recorded, 2),
        },
        "per_edge_stats": per_edge_stats,
        "config": {
            "rdp_epsilon": config.rdp_epsilon,
            "straight_max_deviation": config.straight_max_deviation,
            "min_straight_edge_length": config.min_straight_edge_length,
            "smooth_window": config.smooth_window,
            "max_smooth_offset": config.max_smooth_offset,
            "mask_tolerance": config.mask_tolerance,
        },
    }

    # Topology is immutable in a line optimizer.  Keep an explicit guard here so
    # future geometry changes cannot silently remove or reconnect graph edges.
    if len(optimized) != len(edges):
        return copy.deepcopy(edges), report
    before_topology = {
        e.get("id"): (e.get("start", e.get("from")), e.get("end", e.get("to")))
        for e in edges
    }
    after_topology = {
        e.get("id"): (e.get("start", e.get("from")), e.get("end", e.get("to")))
        for e in optimized
    }
    if before_topology != after_topology:
        return copy.deepcopy(edges), report
    return optimized, report


# ===========================================================================
# 保存 / 预览图
# ===========================================================================

def save_optimization_results(
    edges_before: List[Dict],
    edges_after: List[Dict],
    report: Dict,
    output_dir: str,
    image_rgb: Optional[np.ndarray] = None,
) -> Dict:
    """
    保存优化前/后 graph JSON、报告 JSON 和对比预览图。

    输出目录结构：
        graph_line_optimize_outputs/
        ├── final_graph_before_line_opt.json
        ├── final_graph_after_line_opt.json
        ├── graph_line_optimize_report.json
        └── graph_line_optimize_preview.png

    Args:
        edges_before: 优化前的边列表
        edges_after: 优化后的边列表
        report: optimize_graph_lines 返回的 report
        output_dir: 输出目录
        image_rgb: (H, W, 3) uint8 RGB 原图，用于生成对比预览

    Returns:
        {"before_json": str, "after_json": str, "report_json": str, "preview_png": str|None}
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 优化前 graph JSON ──
    before_path = os.path.join(output_dir, "final_graph_before_line_opt.json")
    with open(before_path, "w", encoding="utf-8") as f:
        json.dump({"edges": edges_before, "edge_count": len(edges_before)},
                  f, ensure_ascii=False, indent=2)

    # ── 优化后 graph JSON ──
    after_path = os.path.join(output_dir, "final_graph_after_line_opt.json")
    with open(after_path, "w", encoding="utf-8") as f:
        json.dump({"edges": edges_after, "edge_count": len(edges_after)},
                  f, ensure_ascii=False, indent=2)

    # ── 报告 JSON ──
    report_path = os.path.join(output_dir, "graph_line_optimize_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── 对比预览图 ──
    preview_path = None
    if image_rgb is not None:
        preview_path = os.path.join(output_dir, "graph_line_optimize_preview.png")
        _generate_preview_image(edges_before, edges_after, report, image_rgb, preview_path)

    print(f"[GraphLineOpt] 结果已保存: {output_dir}")
    print(f"  - before: {before_path}")
    print(f"  - after:  {after_path}")
    print(f"  - report: {report_path}")
    if preview_path:
        print(f"  - preview: {preview_path}")

    return {
        "before_json": before_path,
        "after_json": after_path,
        "report_json": report_path,
        "preview_png": preview_path,
    }


def _generate_preview_image(edges_before, edges_after, report, image_rgb, output_path):
    """生成优化前/后 左右并排对比预览图。"""
    h, w = image_rgb.shape[:2]
    margin = 20
    canvas_w = w * 2 + margin
    canvas_h = h + 50  # 底部文字区域
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 30  # 深灰背景

    # 左边：原图 + 优化前 polyline（橙色）
    canvas[0:h, 0:w] = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for edge in edges_before:
        pts = edge.get("points_pixel", [])
        if len(pts) >= 2:
            pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_arr], False, (0, 180, 255), 2)

    # 右边：原图 + 优化后 polyline（绿色）
    canvas[0:h, w + margin:w + margin + w] = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for edge in edges_after:
        pts = edge.get("points_pixel", [])
        if len(pts) >= 2:
            pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_arr], False, (0, 255, 100), 2)

    # 标注
    cv2.putText(canvas, "BEFORE", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
    cv2.putText(canvas, "AFTER", (w + margin + 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

    # 底部统计
    s = report["summary"]
    y0 = h + 30
    text = (f"Straight: {s['straightened_edges']}  Smooth: {s['smoothed_edges']}  "
            f"Rollback: {s['mask_rollback_edges']}  "
            f"Pts: {s['total_points_before']} → {s['total_points_after']} "
            f"({s['points_reduction_pct']}%)")
    cv2.putText(canvas, text, (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.imwrite(output_path, canvas)
