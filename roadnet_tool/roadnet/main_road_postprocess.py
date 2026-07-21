"""大图主路优先后处理 / Main Road Refinement（种子 / ROI / 任务点约束下的半自动修复）。

设计原则（本轮重写）：
  - 不再做“全图自由修复”。所有连通域筛选、方向闭运算、骨架剪枝、端点桥接、
    回膨胀都必须限制在 main_road_corridor 内。
  - corridor 由主路种子线 / ROI / 任务点 buffer（可选当前视野）生成。
  - 连通域保留改为 seed-connected：不接触 seed/ROI/task 的碎片即使面积很大也删除。
  - 桥接受严格约束并有全局上限，输出桥接候选供人工确认。
  - 先 skeleton → graph edge 评分与剪枝，再回绘 skeleton → dilate 得到 main_road_mask。

只用于 large_image_mode；小图流程不调用本模块。
本模块不依赖 Qt，可独立测试。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from roadnet.optimized_skeleton import (
    skeletonize_thin,
    compute_distance_transform,
    _find_endpoints,
    _find_junctions,
    _count_neighbors,
    _NEIGHBORS_8,
)


# ---------------------------------------------------------------------------
# 默认参数（合并需求第四/六/七/十/十六条）
# ---------------------------------------------------------------------------
DEFAULT_MAIN_ROAD_CONFIG: Dict[str, Any] = {
    "use_preview_level": True,
    "preview_max_side": 2000,

    # corridor 生成
    "seed_corridor_width_preview": 60,
    "task_buffer_preview": 50,
    "roi_corridor_enabled": True,

    # 约束开关：无 seed / ROI / task 时禁止全图修复
    "require_seed_or_roi_or_task": True,
    "advanced_allow_unseeded": False,
    "keep_unseeded_top_k": 0,

    # 连通域保留
    "min_component_area_preview": 80,
    "min_skeleton_length_preview": 40,
    "inside_corridor_ratio_keep": 0.5,

    # 方向闭运算
    "line_close_length_preview": 15,

    # skeleton graph edge 评分 / 剪枝
    "min_edge_length_preview": 40,
    "edge_score_threshold": 1.0,
    "remove_branch_length_preview": 50,
    "preserve_task_nearby_branch": True,
    "preserve_seed_branch": True,

    # 桥接（严格约束）
    "max_bridge_gap_preview": 20,
    "angle_threshold_deg": 25,
    "line_sample_step_px": 2,
    "min_road_support_ratio": 0.65,
    "bridge_count_limit": 20,
    "only_bridge_inside_corridor": True,
    "require_seed_connected_for_bridge": True,
    "auto_accept_bridges": False,

    # 回膨胀
    "road_radius_preview": "auto",
    "road_radius_min": 5,
    "road_radius_max": 8,
    "fallback_road_radius_preview": 6,

    # 孔洞（大图仍禁止全局填充）
    "fill_holes": False,
    "max_hole_area_preview": 200,

    # road support
    "support_radius_preview": 6,
    "color_support_threshold": 40.0,
    "seed_line_thickness": 3,
}

_ANGLES = (0, 45, 90, 135)


# ---------------------------------------------------------------------------
# 掩膜辅助
# ---------------------------------------------------------------------------
def _polygons_to_mask(shape, polygons) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for poly in polygons or []:
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
    return mask


def _points_to_mask(shape, points, radius: int) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for pt in points or []:
        x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            cv2.circle(mask, (x, y), max(1, int(radius)), 255, -1)
    return mask


def _strokes_to_mask(shape, strokes, thickness: int) -> np.ndarray:
    """将主路种子线（折线列表）绘制为掩膜。"""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for stroke in strokes or []:
        pts = np.asarray(stroke, dtype=np.int32).reshape(-1, 2)
        if len(pts) == 1:
            x, y = int(pts[0][0]), int(pts[0][1])
            cv2.circle(mask, (x, y), max(1, thickness), 255, -1)
        elif len(pts) >= 2:
            cv2.polylines(mask, [pts.reshape(-1, 1, 2)], False, 255,
                          max(1, int(thickness)))
    return mask


def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    length = max(1, int(length))
    if length % 2 == 0:
        length += 1
    kernel = np.zeros((length, length), dtype=np.uint8)
    c = (length - 1) / 2.0
    theta = np.deg2rad(angle_deg)
    dx, dy = np.cos(theta), np.sin(theta)
    x0 = int(round(c - dx * c)); y0 = int(round(c - dy * c))
    x1 = int(round(c + dx * c)); y1 = int(round(c + dy * c))
    cv2.line(kernel, (x0, y0), (x1, y1), 1, 1)
    if kernel.sum() == 0:
        kernel[int(c), int(c)] = 1
    return kernel


def _ellipse(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _skeleton_length(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    return int(np.count_nonzero(skeletonize_thin(mask)))


# ---------------------------------------------------------------------------
# 第四条：main_road_corridor 生成
# ---------------------------------------------------------------------------
def build_main_road_corridor(
    shape,
    seed_strokes: Optional[Sequence] = None,
    roi_polygons: Optional[Sequence] = None,
    task_points: Optional[Sequence] = None,
    view_rect: Optional[Sequence] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """根据种子线 / ROI / 任务点（可选当前视野）生成主路修复走廊。"""
    cfg = dict(DEFAULT_MAIN_ROAD_CONFIG)
    cfg.update(config or {})
    corridor = np.zeros(shape[:2], dtype=np.uint8)
    info = {"has_seed": False, "has_roi": False, "has_task": False, "has_view": False}

    thickness = int(cfg["seed_line_thickness"])
    seed_mask = _strokes_to_mask(shape, seed_strokes, thickness)
    if np.any(seed_mask):
        info["has_seed"] = True
        half = max(1, int(cfg["seed_corridor_width_preview"]) // 2)
        corridor = np.maximum(corridor, cv2.dilate(seed_mask, _ellipse(half)))

    if cfg.get("roi_corridor_enabled", True) and roi_polygons:
        roi_mask = _polygons_to_mask(shape, roi_polygons)
        if np.any(roi_mask):
            info["has_roi"] = True
            corridor = np.maximum(corridor, roi_mask)

    if task_points:
        task_mask = _points_to_mask(shape, task_points, int(cfg["task_buffer_preview"]))
        if np.any(task_mask):
            info["has_task"] = True
            corridor = np.maximum(corridor, task_mask)

    if view_rect is not None and len(view_rect) == 4:
        x0, y0, x1, y1 = [int(round(v)) for v in view_rect]
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(shape[1], x1); y1 = min(shape[0], y1)
        if x1 > x0 and y1 > y0:
            corridor[y0:y1, x0:x1] = 255
            info["has_view"] = True

    info["corridor_nonzero_ratio"] = round(
        float(np.count_nonzero(corridor)) / float(shape[0] * shape[1]), 6
    )
    return corridor, info


# ---------------------------------------------------------------------------
# 第六条：连通域保留（seed-connected）
# ---------------------------------------------------------------------------
def filter_seed_connected_components(
    binary: np.ndarray,
    corridor: np.ndarray,
    seed_mask: np.ndarray,
    roi_mask: np.ndarray,
    task_mask: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """保留与 seed / ROI / task 相连的连通域，删除孤立误检（即使面积很大）。"""
    min_area = int(config["min_component_area_preview"])
    min_skel = int(config["min_skeleton_length_preview"])
    inside_keep = float(config["inside_corridor_ratio_keep"])
    allow_unseeded = bool(config.get("advanced_allow_unseeded", False))
    top_k = int(config.get("keep_unseeded_top_k", 0))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    info: Dict[str, Any] = {
        "component_count_before": int(num - 1),
        "kept_component_ids": [],
        "removed_component_count": 0,
        "removed_unseeded_components": 0,
    }
    if num <= 1:
        return np.zeros_like(binary), info

    skel_all = skeletonize_thin(binary)
    skel_counts = np.bincount(labels[skel_all > 0], minlength=num)
    skel_counts[0] = 0

    seed_hits = set(np.unique(labels[seed_mask > 0]).tolist()) if seed_mask is not None else set()
    roi_hits = set(np.unique(labels[roi_mask > 0]).tolist()) if roi_mask is not None else set()
    task_hits = set(np.unique(labels[task_mask > 0]).tolist()) if task_mask is not None else set()
    for s in (seed_hits, roi_hits, task_hits):
        s.discard(0)

    corridor_bin = (corridor > 0)
    order = sorted(range(1, num), key=lambda k: int(skel_counts[k]), reverse=True)
    top_set = set(order[:max(0, top_k)]) if allow_unseeded else set()

    kept = np.zeros_like(binary)
    kept_ids: List[int] = []
    removed = 0
    removed_unseeded = 0
    for k in range(1, num):
        area = int(stats[k, cv2.CC_STAT_AREA])
        skel_len = int(skel_counts[k])
        comp = (labels == k)
        inside_ratio = (float(np.count_nonzero(comp & corridor_bin)) / float(area)
                        if area else 0.0)
        intersects_seed = k in seed_hits
        intersects_task = k in task_hits
        intersects_roi = k in roi_hits

        keep = False
        if intersects_seed:
            keep = True
        elif intersects_task and skel_len >= min_skel:
            keep = True
        elif intersects_roi and inside_ratio >= inside_keep:
            keep = True
        elif allow_unseeded and k in top_set and skel_len >= min_skel:
            keep = True

        # 面积再大，只要不接触 seed/ROI/task 就删除（第六条第 4 点）。
        if not (intersects_seed or intersects_task or intersects_roi):
            if not (allow_unseeded and k in top_set):
                keep = False

        # 太小太短也删除。
        if area < min_area and skel_len < min_skel and not intersects_seed:
            keep = False

        if keep:
            kept[comp] = 255
            kept_ids.append(k)
        else:
            removed += 1
            if not (intersects_seed or intersects_task or intersects_roi):
                removed_unseeded += 1

    info["kept_component_ids"] = kept_ids
    info["removed_component_count"] = removed
    info["removed_unseeded_components"] = removed_unseeded
    return kept, info


# ---------------------------------------------------------------------------
# 方向形态学闭运算（仅 corridor 内）
# ---------------------------------------------------------------------------
def directional_close(
    mask: np.ndarray,
    length: int,
    allowed_region: Optional[np.ndarray] = None,
) -> np.ndarray:
    base = (mask > 0).astype(np.uint8) * 255
    acc = base.copy()
    for angle in _ANGLES:
        kernel = _line_kernel(length, angle)
        acc = np.maximum(acc, cv2.morphologyEx(base, cv2.MORPH_CLOSE, kernel))
    extra = cv2.bitwise_and(acc, cv2.bitwise_not(base))
    if allowed_region is not None:
        extra = cv2.bitwise_and(extra, (allowed_region > 0).astype(np.uint8) * 255)
    return np.maximum(base, extra)


# ---------------------------------------------------------------------------
# 第九条：skeleton graph edge 提取与评分
# ---------------------------------------------------------------------------
def extract_skeleton_edges(skeleton: np.ndarray):
    """将骨架在 junction（度>=3）处断开，返回 [(edge_pixels, ...)] 与 junction 集合。"""
    binary = skeleton > 0
    junctions = _find_junctions(binary)
    seg = binary.copy()
    for (y, x) in junctions:
        seg[y, x] = False
    num, labels = cv2.connectedComponents(seg.astype(np.uint8), connectivity=8)
    edges = []
    for k in range(1, num):
        ys, xs = np.where(labels == k)
        edges.append(list(zip(ys.tolist(), xs.tolist())))
    return edges, junctions


def score_and_filter_edges(
    skeleton: np.ndarray,
    corridor: np.ndarray,
    seed_dil: np.ndarray,
    task_mask: np.ndarray,
    color_support: Optional[np.ndarray],
    config: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """对 skeleton graph 的每条 edge 评分并剪枝，回绘保留的 edge。"""
    min_len = int(config["min_edge_length_preview"])
    threshold = float(config["edge_score_threshold"])
    binary = skeleton > 0

    edges, junctions = extract_skeleton_edges(skeleton)
    kept = np.zeros_like(skeleton)
    for (jy, jx) in junctions:      # 保留 junction 像素连接性
        kept[jy, jx] = 255

    info = {
        "edge_count_before": len(edges),
        "edge_count_after": 0,
        "removed_short_branch_count": 0,
    }
    corridor_bin = corridor > 0
    seed_bin = seed_dil > 0
    task_bin = (task_mask > 0) if task_mask is not None else None
    color_bin = (color_support > 0) if color_support is not None else None

    def _is_endpoint(y, x):
        return _count_neighbors(binary, y, x) == 1

    kept_count = 0
    removed_short = 0
    for pts in edges:
        n = len(pts)
        if n == 0:
            continue
        length = n
        inside = sum(1 for (y, x) in pts if corridor_bin[y, x]) / n
        seed_overlap = sum(1 for (y, x) in pts if seed_bin[y, x]) / n
        task_overlap = (sum(1 for (y, x) in pts if task_bin[y, x]) / n) if task_bin is not None else 0.0
        color_sup = (sum(1 for (y, x) in pts if color_bin[y, x]) / n) if color_bin is not None else 0.5
        endpoints = sum(1 for (y, x) in pts if _is_endpoint(y, x))
        is_spur = endpoints >= 1

        # 强制删除：corridor 外的 edge。
        if inside < 0.5:
            removed_short += 1 if length < min_len else 0
            continue
        # 强制删除：短毛刺，且不接触 seed / task。
        if length < min_len and seed_overlap == 0 and task_overlap == 0 and is_spur:
            removed_short += 1
            continue

        length_score = min(1.0, length / float(max(1, min_len)))
        score = (length_score * 1.0 + seed_overlap * 2.0 + inside * 1.0
                 + task_overlap * 1.5 + color_sup * 0.5)
        if is_spur and length < min_len:
            score -= 1.0

        if score >= threshold:
            for (y, x) in pts:
                kept[y, x] = 255
            kept_count += 1
        else:
            if length < min_len:
                removed_short += 1

    info["edge_count_after"] = kept_count
    info["removed_short_branch_count"] = removed_short
    return kept, info


# ---------------------------------------------------------------------------
# 端点方向 / 角度
# ---------------------------------------------------------------------------
def _endpoint_direction(binary: np.ndarray, ep: Tuple[int, int], steps: int = 6) -> np.ndarray:
    h, w = binary.shape
    y, x = ep
    visited = {(y, x)}
    cy, cx = y, x
    far = (y, x)
    for _ in range(steps):
        nxt = None
        for dy, dx in _NEIGHBORS_8:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and (ny, nx) not in visited:
                nxt = (ny, nx)
                break
        if nxt is None:
            break
        visited.add(nxt)
        cy, cx = nxt
        far = nxt
    v = np.array([x - far[1], y - far[0]], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([0.0, 0.0])


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 180.0
    c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


# ---------------------------------------------------------------------------
# 第七条：受严格约束的端点桥接（含候选记录）
# ---------------------------------------------------------------------------
def constrained_bridge_endpoints(
    skeleton: np.ndarray,
    corridor: np.ndarray,
    anchor_mask: np.ndarray,
    support_region: np.ndarray,
    ignore_mask: Optional[np.ndarray],
    config: Dict[str, Any],
    color_support: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    """只在 corridor 内、seed-connected 约束下桥接，且有全局数量上限。

    返回 (桥接后骨架, 统计, 候选记录列表)。候选记录含 status: accepted/rejected/pending。
    """
    max_gap = float(config["max_bridge_gap_preview"])
    angle_thr = float(config["angle_threshold_deg"])
    step = max(1, int(config["line_sample_step_px"]))
    min_support = float(config["min_road_support_ratio"])
    limit = int(config["bridge_count_limit"])
    only_in_corridor = bool(config["only_bridge_inside_corridor"])
    require_anchor = bool(config["require_seed_connected_for_bridge"])
    auto_accept = bool(config["auto_accept_bridges"])

    binary = (skeleton > 0)
    endpoints = _find_endpoints(binary)
    stats = {
        "endpoint_count_before": len(endpoints),
        "bridge_candidate_count": 0,
        "accepted_bridge_count": 0,
        "rejected_bridge_count": 0,
        "pending_bridge_count": 0,
    }
    candidates: List[Dict[str, Any]] = []
    if len(endpoints) < 2:
        stats["endpoint_count_after"] = len(endpoints)
        return (binary.astype(np.uint8) * 255), stats, candidates

    corridor_bin = corridor > 0

    # 每个 skeleton 连通域是否接触 anchor（seed/ROI/task）。
    num, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    anchored_labels = set(np.unique(labels[anchor_mask > 0]).tolist())
    anchored_labels.discard(0)

    def _anchored(ep):
        return int(labels[ep[0], ep[1]]) in anchored_labels

    dirs = [_endpoint_direction(binary, ep) for ep in endpoints]
    result = (binary.astype(np.uint8) * 255)
    used = set()
    accepted = 0

    pairs = []
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            (y1, x1), (y2, x2) = endpoints[i], endpoints[j]
            d = float(np.hypot(x2 - x1, y2 - y1))
            if 0 < d <= max_gap:
                pairs.append((d, i, j))
    pairs.sort(key=lambda t: t[0])

    for d, i, j in pairs:
        if i in used or j in used:
            continue
        (y1, x1), (y2, x2) = endpoints[i], endpoints[j]
        rec = {"p1": [int(x1), int(y1)], "p2": [int(x2), int(y2)],
               "distance": round(d, 2), "status": "rejected", "reason": ""}

        # 1) 两端点都在 corridor 内
        if only_in_corridor and not (corridor_bin[y1, x1] and corridor_bin[y2, x2]):
            rec["reason"] = "outside_corridor"
            candidates.append(rec); stats["rejected_bridge_count"] += 1
            continue
        # 2) 至少一端与 seed/ROI/task 网络相连
        if require_anchor and not (_anchored(endpoints[i]) or _anchored(endpoints[j])):
            rec["reason"] = "not_seed_connected"
            candidates.append(rec); stats["rejected_bridge_count"] += 1
            continue
        # 4) 方向夹角
        line_vec = np.array([x2 - x1, y2 - y1], dtype=float)
        a1 = _angle_between(dirs[i], line_vec)
        a2 = _angle_between(dirs[j], -line_vec)
        if a1 > angle_thr or a2 > angle_thr:
            rec["reason"] = "angle"
            candidates.append(rec); stats["rejected_bridge_count"] += 1
            continue
        # 5/6) road support + 不穿越 ignore
        n_samples = max(2, int(d / step))
        supported = 0
        crosses_ignore = False
        for t in np.linspace(0.0, 1.0, n_samples):
            sx = int(round(x1 + (x2 - x1) * t))
            sy = int(round(y1 + (y2 - y1) * t))
            if 0 <= sy < support_region.shape[0] and 0 <= sx < support_region.shape[1]:
                if ignore_mask is not None and ignore_mask[sy, sx] > 0:
                    crosses_ignore = True
                    break
                if support_region[sy, sx] > 0:
                    supported += 1
                elif color_support is not None and color_support[sy, sx] > 0:
                    supported += 1
        if crosses_ignore:
            rec["reason"] = "crosses_ignore"
            candidates.append(rec); stats["rejected_bridge_count"] += 1
            continue
        ratio = supported / float(n_samples)
        rec["road_support_ratio"] = round(ratio, 3)
        if ratio < min_support:
            rec["reason"] = "low_support"
            candidates.append(rec); stats["rejected_bridge_count"] += 1
            continue

        # 通过所有约束：是否达到全局上限
        if accepted >= limit:
            rec["status"] = "pending"
            rec["reason"] = "bridge_count_limit"
            candidates.append(rec); stats["pending_bridge_count"] += 1
            continue

        if auto_accept:
            cv2.line(result, (x1, y1), (x2, y2), 255, 1)
            rec["status"] = "accepted"
            rec["reason"] = "auto_accept"
            accepted += 1
            used.add(i); used.add(j)
            stats["accepted_bridge_count"] += 1
        else:
            # 默认不自动接受：标记为待人工确认（高置信仍记录）。
            rec["status"] = "pending"
            rec["reason"] = "await_confirm"
            stats["pending_bridge_count"] += 1
        candidates.append(rec)

    stats["bridge_candidate_count"] = len(candidates)
    bridged = (result > 0).astype(np.uint8) * 255
    stats["endpoint_count_after"] = len(_find_endpoints(bridged > 0))
    return bridged, stats, candidates


def apply_bridges(skeleton: np.ndarray, candidates: List[Dict[str, Any]],
                  statuses=("accepted",)) -> np.ndarray:
    """按状态把桥接候选画到骨架上（供人工确认后应用）。"""
    result = (skeleton > 0).astype(np.uint8) * 255
    for rec in candidates:
        if rec.get("status") in statuses:
            (x1, y1), (x2, y2) = rec["p1"], rec["p2"]
            cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
    return result


# ---------------------------------------------------------------------------
# 第十条：受保护短支路剪枝
# ---------------------------------------------------------------------------
def prune_short_branches_protected(
    skeleton: np.ndarray,
    min_branch_length: int,
    protect_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    binary = skeleton > 0
    h, w = binary.shape
    endpoints = _find_endpoints(binary)
    junctions = _find_junctions(binary)
    to_remove = np.zeros_like(binary, dtype=bool)

    for ey, ex in endpoints:
        if protect_mask is not None and protect_mask[ey, ex] > 0:
            continue
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
            nxt = []
            for dy, dx in _NEIGHBORS_8:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and (ny, nx) not in visited:
                    nxt.append((ny, nx))
            if len(nxt) == 0:
                break
            elif len(nxt) == 1:
                cy, cx = nxt[0]
            else:
                if len(branch) >= min_branch_length:
                    break
                cy, cx = nxt[0]
        if 0 < len(branch) < min_branch_length:
            if protect_mask is not None and any(protect_mask[py, px] > 0 for py, px in branch):
                continue
            for py, px in branch:
                to_remove[py, px] = True

    binary[to_remove] = False
    return (binary * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 第十一条：道路半宽估计
# ---------------------------------------------------------------------------
def estimate_road_radius(binary: np.ndarray, config: Dict[str, Any]) -> int:
    setting = config.get("road_radius_preview", "auto")
    r_min = int(config.get("road_radius_min", 5))
    r_max = int(config.get("road_radius_max", 8))
    fallback = int(config.get("fallback_road_radius_preview", 6))
    if isinstance(setting, (int, float)) and str(setting) != "auto":
        return int(np.clip(int(setting), r_min, r_max))
    if not np.any(binary):
        return fallback
    dist = compute_distance_transform((binary > 0).astype(np.uint8))
    vals = dist[binary > 0]
    vals = vals[vals > 0]
    if vals.size == 0:
        return fallback
    return int(np.clip(round(float(np.median(vals))), r_min, r_max))


def _fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    inverse = (mask == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    result = mask.copy()
    h, w = mask.shape[:2]
    for label in range(1, count):
        x, y, ww, hh, area = stats[label]
        touches_border = x == 0 or y == 0 or x + ww >= w or y + hh >= h
        if not touches_border and int(area) <= int(max_area):
            result[labels == label] = 255
    return result


def _build_color_support(image_bgr: np.ndarray, binary: np.ndarray,
                         cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    try:
        img = np.asarray(image_bgr)
        if img.ndim != 3:
            return None
        road_px = img[binary > 0]
        if road_px.size == 0:
            return None
        ref = np.median(road_px.reshape(-1, 3), axis=0).astype(np.float32)
        diff = img.astype(np.float32) - ref[None, None, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        thr = float(cfg.get("color_support_threshold", 40.0))
        return (dist <= thr).astype(np.uint8) * 255
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def refine_main_road_mask(
    raw_mask: np.ndarray,
    image_bgr: Optional[np.ndarray] = None,
    roi_polygons: Optional[Sequence] = None,
    ignore_polygons: Optional[Sequence] = None,
    task_points: Optional[Sequence] = None,
    seed_strokes: Optional[Sequence] = None,
    corridor_mask: Optional[np.ndarray] = None,
    view_rect: Optional[Sequence] = None,
    config: Optional[Dict[str, Any]] = None,
    stages_out: Optional[Dict[str, Any]] = None,
):
    """主路种子 / ROI / 任务点约束下的半自动主路修复。

    所有几何输入（seed/roi/ignore/task/corridor）必须与 raw_mask 处于同一像素坐标系
    （通常为 preview 像素）。report 中记录 coordinate_space。

    Returns:
        (refined_mask uint8 0/255, report dict)
    """
    started = time.perf_counter()
    cfg = dict(DEFAULT_MAIN_ROAD_CONFIG)
    cfg.update(config or {})

    raw = np.asarray(raw_mask)
    if raw.ndim != 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    binary = (raw > 0).astype(np.uint8) * 255
    shape = binary.shape
    total_px = float(shape[0] * shape[1])

    report: Dict[str, Any] = {
        "coordinate_space": "preview_pixel",
        "input_mask_shape": [int(shape[0]), int(shape[1])],
        "used_preview_level": bool(cfg.get("use_preview_level", True)),
        "seed_stroke_count": len(seed_strokes or []),
        "roi_count": len(roi_polygons or []),
        "task_point_count": len(task_points or []),
        "warnings": [],
        "mask_nonzero_ratio_before": round(float(np.count_nonzero(binary)) / total_px, 6),
        "skeleton_length_before": _skeleton_length(binary),
    }

    # 掩膜化约束。
    seed_mask = _strokes_to_mask(shape, seed_strokes, int(cfg["seed_line_thickness"]))
    roi_mask = _polygons_to_mask(shape, roi_polygons) if roi_polygons else np.zeros(shape, np.uint8)
    task_mask = (_points_to_mask(shape, task_points, int(cfg["task_buffer_preview"]))
                 if task_points else np.zeros(shape, np.uint8))
    ignore_mask = _polygons_to_mask(shape, ignore_polygons) if ignore_polygons else None

    has_constraint = bool(np.any(seed_mask) or np.any(roi_mask) or np.any(task_mask))

    # corridor：外部传入优先，否则由约束生成。
    if corridor_mask is not None:
        corridor = (np.asarray(corridor_mask) > 0).astype(np.uint8) * 255
        corr_info = {"source": "external"}
    else:
        corridor, corr_info = build_main_road_corridor(
            shape, seed_strokes, roi_polygons, task_points, view_rect, cfg
        )
    report["used_corridor"] = bool(np.any(corridor))
    report["corridor_info"] = corr_info

    # 第四/十三条：无约束禁止全图修复。
    if (cfg.get("require_seed_or_roi_or_task", True)
            and not has_constraint and corridor_mask is None):
        report["warnings"].append("未提供主路约束，已拒绝执行全图主路修复。")
        report["refused"] = True
        report["component_count_before"] = 0
        report["component_count_after"] = 0
        report["mask_nonzero_ratio_after"] = 0.0
        report["skeleton_length_after"] = 0
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        empty = np.zeros_like(binary)
        if stages_out is not None:
            stages_out["raw_mask_preview"] = binary
            stages_out["main_road_corridor_mask"] = corridor
            stages_out["main_road_mask_preview"] = empty
        return empty, report
    report["refused"] = False

    if not np.any(binary):
        report["warnings"].append("输入 mask 全为空，无法修复主路。")
        report["component_count_before"] = 0
        report["component_count_after"] = 0
        report["mask_nonzero_ratio_after"] = 0.0
        report["skeleton_length_after"] = 0
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        if stages_out is not None:
            stages_out["raw_mask_preview"] = binary
            stages_out["main_road_corridor_mask"] = corridor
            stages_out["main_road_mask_preview"] = binary
        return binary, report

    # ── 第六条：seed-connected 连通域筛选 ──
    component_filtered, comp_info = filter_seed_connected_components(
        binary, corridor, seed_mask, roi_mask, task_mask, cfg
    )
    report.update(comp_info)
    report["component_count_after"] = len(comp_info.get("kept_component_ids", []))

    # 第五条：只在 corridor 内修复。
    cleaned = cv2.bitwise_and(component_filtered, corridor)
    seed_connected_components = component_filtered  # 记录 seed-connected 保留结果

    # ── 方向闭运算（仅 corridor 内）──
    line_len = int(cfg["line_close_length_preview"])
    directional_closed = directional_close(cleaned, line_len, corridor)

    # ── 第九条：skeleton → graph edge 评分 / 剪枝 ──
    skeleton_raw = skeletonize_thin(directional_closed)
    seed_dil = cv2.dilate(seed_mask, _ellipse(max(2, int(cfg["seed_line_thickness"]))))
    color_support = _build_color_support(image_bgr, binary, cfg) if image_bgr is not None else None
    edge_kept, edge_info = score_and_filter_edges(
        skeleton_raw, corridor, seed_dil, task_mask, color_support, cfg
    )
    report.update(edge_info)
    skeleton_kept = skeletonize_thin(edge_kept)

    # ── 第十条：受保护短支路剪枝 ──
    protect_mask = np.zeros(shape, dtype=np.uint8)
    if bool(cfg.get("preserve_seed_branch", True)):
        protect_mask = np.maximum(protect_mask, seed_dil)
    if bool(cfg.get("preserve_task_nearby_branch", True)):
        protect_mask = np.maximum(protect_mask, task_mask)
    skeleton_pruned = prune_short_branches_protected(
        skeleton_kept, int(cfg["remove_branch_length_preview"]), protect_mask
    )

    # ── 第七/八条：受约束端点桥接（默认不自动接受）──
    support_radius = int(cfg["support_radius_preview"])
    support_region = cv2.dilate(binary, _ellipse(support_radius))
    anchor_mask = np.maximum(np.maximum(seed_dil, roi_mask), task_mask)
    endpoint_bridged, bridge_stats, bridge_candidates = constrained_bridge_endpoints(
        skeleton_pruned, corridor, anchor_mask, support_region, ignore_mask,
        cfg, color_support
    )
    report.update(bridge_stats)
    report["bridge_candidates"] = bridge_candidates

    # ── 回绘 skeleton（接受的桥接已在 endpoint_bridged 内）──
    repaired_skeleton = skeletonize_thin(endpoint_bridged)
    repaired_skeleton = prune_short_branches_protected(
        repaired_skeleton, max(5, int(cfg["remove_branch_length_preview"]) // 2), protect_mask
    )

    # ── 第十一条：估计半宽并回膨胀（限制在 corridor 内）──
    road_radius = estimate_road_radius(binary, cfg)
    report["road_radius_preview"] = int(road_radius)
    refined = cv2.dilate((repaired_skeleton > 0).astype(np.uint8) * 255, _ellipse(road_radius))
    refined = cv2.bitwise_and(refined, corridor)

    if bool(cfg.get("fill_holes", False)):
        refined = _fill_small_holes(refined, int(cfg["max_hole_area_preview"]))

    # ── 第十二条相关中间产物 ──
    if ignore_mask is not None:
        refined[ignore_mask > 0] = 0

    report["skeleton_length_after"] = int(np.count_nonzero(repaired_skeleton))
    report["mask_nonzero_ratio_after"] = round(float(np.count_nonzero(refined)) / total_px, 6)
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)

    if stages_out is not None:
        stages_out["raw_mask_preview"] = binary
        stages_out["main_road_corridor_mask"] = corridor
        stages_out["seed_connected_components"] = seed_connected_components
        stages_out["component_filtered_mask"] = cleaned
        stages_out["directional_closed_mask"] = directional_closed
        stages_out["skeleton_raw_preview"] = skeleton_raw
        stages_out["edge_kept_skeleton"] = skeleton_kept
        stages_out["pruned_skeleton_preview"] = repaired_skeleton
        stages_out["main_road_mask_preview"] = refined
        stages_out["seed_mask"] = seed_mask
        stages_out["bridge_candidates"] = bridge_candidates

    return refined, report
