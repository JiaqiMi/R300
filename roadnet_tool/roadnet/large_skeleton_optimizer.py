"""大图模式专用：干净骨架生成流水线。

仅用于 large_image_mode。小图流程不要调用本模块。

流水线：
  final_edited_mask (或其他优先级 mask)
  → mask_preclean
  → skeletonize / medial_axis
  → distance_transform_centerline_filter
  → skeleton_pixel / component filter
  → skeleton_to_graph
  → graph_edge_prune (+ junction clustering via skeleton_to_graph)
  → conservative_endpoint_bridge
  → smooth_polyline
  → cleaned_skeleton / final_graph
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from roadnet.main_road_postprocess import (
    apply_bridges,
    build_main_road_corridor,
    constrained_bridge_endpoints,
    filter_seed_connected_components,
    _polygons_to_mask,
    _points_to_mask,
    _strokes_to_mask,
    _ellipse,
)
from roadnet.optimized_skeleton import (
    compute_distance_transform,
    filter_boundary_points,
    skeletonize_thin,
)
from roadnet.skeleton_to_graph import (
    SkeletonToGraphConfig,
    skeleton_to_graph,
    simplify_edges,
)


DEFAULT_LARGE_CLEAN_SKELETON_CONFIG: Dict[str, Any] = {
    "use_preview_level": True,
    "preview_max_side": 2000,

    # mask_preclean
    "min_component_area_preview": 120,
    "min_skeleton_length_preview": 100,
    "close_kernel": 5,
    "open_kernel": 3,
    "fill_holes": False,
    "max_hole_area_preview": 200,
    "seed_line_thickness": 3,
    "seed_corridor_width_preview": 60,
    "task_buffer_preview": 50,
    "roi_corridor_enabled": True,
    "keep_top_k_without_constraints": 3,
    "remove_isolated_components": True,
    "inside_corridor_ratio_keep": 0.5,
    "advanced_allow_unseeded": True,
    "keep_unseeded_top_k": 3,

    # centerline distance filter
    "center_dist_percentile": 60,
    "center_dist_ratio": 0.25,
    "min_center_dist_floor": 2.0,

    # skeleton component filter
    "min_skeleton_component_length_preview": 80,
    "min_mean_center_distance_px": 2.0,
    "keep_top_k_skel_without_constraints": 3,

    # graph / edge prune
    "junction_cluster_radius_preview": 10,
    "min_edge_length_preview": 50,
    "remove_branch_length_preview": 60,
    "min_edge_mean_distance_px": 2.0,
    "preserve_task_nearby_edge": True,
    "preserve_seed_edge": True,
    "endpoint_merge_distance": 12,
    "node_merge_distance": 8,
    # disable skeleton_to_graph internal auto-bridge; we do conservative bridge after prune
    "endpoint_connect_distance": 0.0,

    # conservative bridge
    "max_bridge_gap_preview": 18,
    "angle_threshold_deg": 25,
    "line_sample_step_px": 2,
    "min_road_support_ratio": 0.70,
    "bridge_count_limit": 10,
    "bridge_count_limit_without_constraint": 3,
    "only_bridge_inside_corridor": True,
    "require_seed_connected_for_bridge": True,
    "auto_accept_bridges": True,
    "support_radius_preview": 6,

    # polyline smooth
    "rdp_epsilon_preview": 2.0,
    "smooth_window": 5,
    "max_smooth_offset": 3,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _binarize(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        return np.zeros((1, 1), dtype=np.uint8)
    arr = mask
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return (arr > 0).astype(np.uint8) * 255


def _downscale_for_preview(mask: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = mask.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return mask, 1.0
    scale = max_side / float(side)
    pw = max(1, int(round(w * scale)))
    ph = max(1, int(round(h * scale)))
    return cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST), scale


def _scale_polys(polys, scale: float):
    if not polys or scale == 1.0:
        return polys
    return [[(float(x) * scale, float(y) * scale) for x, y in poly] for poly in polys]


def _scale_points(points, scale: float):
    if not points or scale == 1.0:
        return points
    return [(float(x) * scale, float(y) * scale) for x, y in points]


def _scale_strokes(strokes, scale: float):
    if not strokes or scale == 1.0:
        return strokes
    return [[(float(x) * scale, float(y) * scale) for x, y in stroke] for stroke in strokes]


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return None
    return obj


def _fill_small_holes(binary: np.ndarray, max_area: int) -> np.ndarray:
    """仅填充面积很小的孔洞；默认关闭大面积 fill。"""
    if max_area <= 0:
        return binary
    inv = cv2.bitwise_not(_binarize(binary))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    out = _binarize(binary).copy()
    h, w = out.shape[:2]
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        # 跳过贴边的“外部背景”
        x, y, bw, bh = (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
        )
        touches_border = x <= 0 or y <= 0 or (x + bw) >= w or (y + bh) >= h
        if touches_border:
            continue
        if area <= max_area:
            out[labels == i] = 255
    return out


def _remove_small_components(binary: np.ndarray, min_area: int) -> Tuple[np.ndarray, int, int]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    kept = np.zeros_like(binary)
    before = max(0, num - 1)
    after = 0
    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            kept[labels == i] = 255
            after += 1
    return kept, before, after


def _filter_components_by_skel_length(
    binary: np.ndarray, min_skel: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    info = {"component_count_before": int(num - 1), "kept": 0, "removed": 0}
    if num <= 1:
        return np.zeros_like(binary), info
    skel = skeletonize_thin(binary)
    skel_counts = np.bincount(labels[skel > 0], minlength=num)
    skel_counts[0] = 0
    kept = np.zeros_like(binary)
    for i in range(1, num):
        if int(skel_counts[i]) >= min_skel or int(stats[i, cv2.CC_STAT_AREA]) >= min_skel * 3:
            kept[labels == i] = 255
            info["kept"] += 1
        else:
            info["removed"] += 1
    return kept, info


# ---------------------------------------------------------------------------
# 1) mask_preclean
# ---------------------------------------------------------------------------

def mask_preclean(
    mask: np.ndarray,
    roi_polygons: Optional[Sequence] = None,
    ignore_polygons: Optional[Sequence] = None,
    main_road_seed_strokes: Optional[Sequence] = None,
    task_points: Optional[Sequence] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """清理 mask，返回 (cleaned_mask_full, cleaned_preview, report)。"""
    cfg = dict(DEFAULT_LARGE_CLEAN_SKELETON_CONFIG)
    cfg.update(config or {})

    full = _binarize(mask)
    oh, ow = full.shape[:2]
    report: Dict[str, Any] = {
        "input_shape": [oh, ow],
        "seed_stroke_count": len(main_road_seed_strokes or []),
        "roi_count": len(roi_polygons or []),
        "task_point_count": len(task_points or []),
        "ignore_count": len(ignore_polygons or []),
    }

    scale = 1.0
    work = full
    if cfg.get("use_preview_level", True):
        work, scale = _downscale_for_preview(full, int(cfg["preview_max_side"]))
    report["preview_scale"] = float(scale)
    report["used_preview_level"] = scale < 1.0

    roi_s = _scale_polys(roi_polygons, scale)
    ign_s = _scale_polys(ignore_polygons, scale)
    seed_s = _scale_strokes(main_road_seed_strokes, scale)
    task_s = _scale_points(task_points, scale)

    # Ignore
    if ign_s:
        ign_m = _polygons_to_mask(work.shape, ign_s)
        work = cv2.bitwise_and(work, cv2.bitwise_not(ign_m))

    # ROI crop (optional soft: keep intersection when ROI exists)
    if roi_s:
        roi_m = _polygons_to_mask(work.shape, roi_s)
        if np.any(roi_m):
            work = cv2.bitwise_and(work, roi_m)

    # remove tiny components
    work, c_before, c_after_area = _remove_small_components(
        work, int(cfg["min_component_area_preview"])
    )
    report["component_count_before"] = int(c_before)

    # remove short-skeleton components
    work, skel_info = _filter_components_by_skel_length(
        work, int(cfg["min_skeleton_length_preview"])
    )
    report["short_skel_component_filter"] = skel_info

    has_constraint = bool(seed_s or roi_s or task_s)
    corridor, corridor_info = build_main_road_corridor(
        work.shape, seed_s, roi_s, task_s, view_rect=None, config=cfg,
    )
    report["corridor"] = corridor_info

    if has_constraint:
        seed_mask = _strokes_to_mask(work.shape, seed_s, int(cfg["seed_line_thickness"]))
        roi_mask = _polygons_to_mask(work.shape, roi_s) if roi_s else np.zeros_like(work)
        task_mask = (
            _points_to_mask(work.shape, task_s, int(cfg["task_buffer_preview"]))
            if task_s else np.zeros_like(work)
        )
        work = cv2.bitwise_and(work, corridor)
        local_cfg = dict(cfg)
        local_cfg["advanced_allow_unseeded"] = False
        local_cfg["keep_unseeded_top_k"] = 0
        work, comp_info = filter_seed_connected_components(
            work, corridor, seed_mask, roi_mask, task_mask, local_cfg,
        )
        report["seed_component_filter"] = comp_info
    else:
        # 无主路约束时：默认曾砍成最长 top-k，会丢掉用户已保存的正式 mask。
        # trust_input_mask / keep_top_k<=0 / keep_top_k>=999 → 保留全部显著连通域。
        top_k = int(cfg.get("keep_top_k_without_constraints", 3) or 0)
        trust = bool(cfg.get("trust_input_mask", False)) or top_k <= 0 or top_k >= 999
        if trust:
            corridor = np.full(work.shape, 255, dtype=np.uint8)
            report["warning"] = (
                "未提供主路约束：信任输入正式 mask，保留全部显著连通域"
                "（仅已剔除过小/过短分量）。"
            )
            report["kept_all_components"] = True
            report["keep_top_k_without_constraints"] = top_k
        else:
            # keep longest top-k by skeleton length
            num, labels, stats, _ = cv2.connectedComponentsWithStats(
                (work > 0).astype(np.uint8), connectivity=8
            )
            skel = skeletonize_thin(work)
            counts = np.bincount(labels[skel > 0], minlength=num)
            counts[0] = 0
            order = sorted(range(1, num), key=lambda k: int(counts[k]), reverse=True)
            kept = np.zeros_like(work)
            for k in order[:max(0, top_k)]:
                kept[labels == k] = 255
            work = kept
            corridor = np.full(work.shape, 255, dtype=np.uint8)
            report["warning"] = (
                "未提供主路约束，仅保留最长 top-k 连通域；建议绘制种子线或设置 ROI。"
            )
            report["keep_top_k_without_constraints"] = top_k
            report["kept_all_components"] = False

    # light close then open
    ck = int(cfg["close_kernel"])
    ok = int(cfg["open_kernel"])
    if ck > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, ker)
        if np.any(corridor):
            work = cv2.bitwise_and(work, corridor)
    if ok > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
        work = cv2.morphologyEx(work, cv2.MORPH_OPEN, ker)

    if cfg.get("fill_holes"):
        work = _fill_small_holes(work, int(cfg["max_hole_area_preview"]))

    _, _, c_after = _remove_small_components(work, max(1, int(cfg["min_component_area_preview"]) // 2))
    del c_after
    report["component_count_after"] = int(
        cv2.connectedComponents((work > 0).astype(np.uint8), connectivity=8)[0] - 1
    )

    if scale < 1.0:
        cleaned_full = cv2.resize(work, (ow, oh), interpolation=cv2.INTER_NEAREST)
        corridor_full = cv2.resize(corridor, (ow, oh), interpolation=cv2.INTER_NEAREST)
    else:
        cleaned_full = work
        corridor_full = corridor

    report["cleaned_nonzero"] = int(np.count_nonzero(cleaned_full))
    report["stages_preview"] = {
        "cleaned_mask_preview": work,
        "corridor_preview": corridor,
        "corridor_full": corridor_full,
    }
    return cleaned_full, work, report


# ---------------------------------------------------------------------------
# 2) distance centerline filter
# ---------------------------------------------------------------------------

def distance_transform_centerline_filter(
    cleaned_mask: np.ndarray,
    raw_skeleton: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """删除 distance 过小的边缘骨架点。"""
    cfg = dict(DEFAULT_LARGE_CLEAN_SKELETON_CONFIG)
    cfg.update(config or {})
    dist = compute_distance_transform(cleaned_mask)
    skel_bin = raw_skeleton > 0
    vals = dist[skel_bin]
    if vals.size == 0:
        return (
            np.zeros_like(raw_skeleton),
            dist,
            float(cfg["min_center_dist_floor"]),
            {"road_radius_est": 0.0, "min_center_dist": float(cfg["min_center_dist_floor"]),
             "raw_pixels": 0, "kept_pixels": 0},
        )
    pct = float(cfg["center_dist_percentile"])
    road_radius_est = float(np.percentile(vals, pct))
    min_center = max(
        float(cfg["min_center_dist_floor"]),
        road_radius_est * float(cfg["center_dist_ratio"]),
    )
    filtered = filter_boundary_points(raw_skeleton, dist, min_center)
    info = {
        "road_radius_est": round(road_radius_est, 3),
        "min_center_dist": round(min_center, 3),
        "raw_pixels": int(np.count_nonzero(skel_bin)),
        "kept_pixels": int(np.count_nonzero(filtered)),
    }
    return filtered, dist, min_center, info


# ---------------------------------------------------------------------------
# 3) skeleton component filter
# ---------------------------------------------------------------------------

def filter_skeleton_components(
    skeleton: np.ndarray,
    dist: np.ndarray,
    seed_mask: np.ndarray,
    roi_mask: np.ndarray,
    task_mask: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    cfg = dict(DEFAULT_LARGE_CLEAN_SKELETON_CONFIG)
    cfg.update(config or {})
    min_len = int(cfg["min_skeleton_component_length_preview"])
    min_mean = float(cfg["min_mean_center_distance_px"])
    top_k = int(cfg["keep_top_k_skel_without_constraints"])

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (skeleton > 0).astype(np.uint8), connectivity=8
    )
    info = {
        "component_count_before": int(max(0, num - 1)),
        "kept": 0,
        "removed_short": 0,
        "removed_low_distance": 0,
        "removed_isolated": 0,
    }
    if num <= 1:
        return np.zeros_like(skeleton), info

    seed_b = seed_mask > 0 if seed_mask is not None else None
    roi_b = roi_mask > 0 if roi_mask is not None else None
    task_b = task_mask > 0 if task_mask is not None else None
    has_constraint = bool(
        (seed_b is not None and np.any(seed_b))
        or (roi_b is not None and np.any(roi_b))
        or (task_b is not None and np.any(task_b))
    )

    scored: List[Tuple[int, int, float, bool]] = []  # id, length, mean_dist, touches
    for i in range(1, num):
        ys, xs = np.where(labels == i)
        length = int(ys.size)
        mean_d = float(dist[ys, xs].mean()) if length else 0.0
        touches = False
        if seed_b is not None and np.any(seed_b[ys, xs]):
            touches = True
        if roi_b is not None and np.any(roi_b[ys, xs]):
            touches = True
        if task_b is not None and np.any(task_b[ys, xs]):
            touches = True
        scored.append((i, length, mean_d, touches))

    kept_ids = set()
    if has_constraint:
        for i, length, mean_d, touches in scored:
            if length < min_len:
                info["removed_short"] += 1
                continue
            if mean_d < min_mean and not touches:
                info["removed_low_distance"] += 1
                continue
            if not touches and length < min_len * 2:
                info["removed_isolated"] += 1
                continue
            kept_ids.add(i)
    else:
        order = sorted(scored, key=lambda t: t[1], reverse=True)
        trust = bool(cfg.get("trust_input_mask", False)) or top_k <= 0 or top_k >= 999
        for i, length, mean_d, _ in order:
            if length < min_len or mean_d < min_mean:
                if length < min_len:
                    info["removed_short"] += 1
                else:
                    info["removed_low_distance"] += 1
                continue
            if (not trust) and len(kept_ids) >= top_k:
                info["removed_isolated"] += 1
                continue
            kept_ids.add(i)
        info["trust_input_mask"] = trust
        info["keep_top_k_skel_without_constraints"] = top_k

    out = np.zeros_like(skeleton)
    for i in kept_ids:
        out[labels == i] = 255
    info["kept"] = len(kept_ids)
    info["component_count_after"] = len(kept_ids)
    return out, info


# ---------------------------------------------------------------------------
# 4) graph edge prune with distance / corridor scoring
# ---------------------------------------------------------------------------

def _path_stats(
    path: Sequence,
    dist: np.ndarray,
    corridor: np.ndarray,
    seed: np.ndarray,
    task: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    if not path:
        return {
            "edge_length": 0.0,
            "mean_center_distance": 0.0,
            "inside_roi_ratio": 0.0,
            "inside_seed_corridor_ratio": 0.0,
            "near_task_buffer": 0.0,
            "mask_support_ratio": 0.0,
        }
    h, w = dist.shape[:2]
    pts = []
    for p in path:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            y, x = int(p[0]), int(p[1])
        else:
            continue
        if 0 <= y < h and 0 <= x < w:
            pts.append((y, x))
    n = len(pts)
    if n == 0:
        return {
            "edge_length": 0.0,
            "mean_center_distance": 0.0,
            "inside_roi_ratio": 0.0,
            "inside_seed_corridor_ratio": 0.0,
            "near_task_buffer": 0.0,
            "mask_support_ratio": 0.0,
        }
    length = 0.0
    for i in range(1, n):
        length += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
    if length <= 0:
        length = float(n)
    mean_d = float(np.mean([dist[y, x] for y, x in pts]))
    corridor_b = corridor > 0
    seed_b = seed > 0
    task_b = task > 0
    mask_b = mask > 0
    return {
        "edge_length": float(length),
        "mean_center_distance": mean_d,
        "inside_roi_ratio": sum(1 for y, x in pts if corridor_b[y, x]) / n,
        "inside_seed_corridor_ratio": sum(1 for y, x in pts if seed_b[y, x]) / n,
        "near_task_buffer": sum(1 for y, x in pts if task_b[y, x]) / n,
        "mask_support_ratio": sum(1 for y, x in pts if mask_b[y, x]) / n,
    }


def prune_graph_edges(
    nodes: List[Dict],
    edges: List[Dict],
    dist: np.ndarray,
    corridor: np.ndarray,
    seed_mask: np.ndarray,
    task_mask: np.ndarray,
    cleaned_mask: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    cfg = dict(DEFAULT_LARGE_CLEAN_SKELETON_CONFIG)
    cfg.update(config or {})
    min_len = float(cfg["min_edge_length_preview"])
    branch_len = float(cfg["remove_branch_length_preview"])
    min_mean = float(cfg["min_edge_mean_distance_px"])
    preserve_task = bool(cfg["preserve_task_nearby_edge"])
    preserve_seed = bool(cfg["preserve_seed_edge"])

    degree = {n["id"]: 0 for n in nodes}
    for e in edges:
        a = e.get("from", e.get("start"))
        b = e.get("to", e.get("end"))
        if a in degree:
            degree[a] += 1
        if b in degree:
            degree[b] += 1

    kept_edges: List[Dict] = []
    removed_short = 0
    removed_low_d = 0
    removed_noise = 0
    for e in edges:
        path = e.get("path") or []
        st = _path_stats(path, dist, corridor, seed_mask, task_mask, cleaned_mask)
        a = e.get("from", e.get("start"))
        b = e.get("to", e.get("end"))
        deg_a = degree.get(a, 0)
        deg_b = degree.get(b, 0)
        is_short_branch = (deg_a == 1 or deg_b == 1) and st["edge_length"] < branch_len
        seed_hit = st["inside_seed_corridor_ratio"] > 0.05
        task_hit = st["near_task_buffer"] > 0.05
        preserve = (preserve_seed and seed_hit) or (preserve_task and task_hit)

        if st["edge_length"] < min_len and not preserve:
            removed_short += 1
            continue
        if st["mean_center_distance"] < min_mean and not preserve:
            removed_low_d += 1
            continue
        if st["inside_roi_ratio"] < 0.35 and not preserve and not seed_hit:
            removed_noise += 1
            continue
        if is_short_branch and not preserve:
            removed_short += 1
            continue

        e2 = dict(e)
        e2.update(st)
        e2["degree_start"] = int(deg_a)
        e2["degree_end"] = int(deg_b)
        e2["is_short_branch"] = bool(is_short_branch)
        kept_edges.append(e2)

    used = set()
    for e in kept_edges:
        used.add(e.get("from", e.get("start")))
        used.add(e.get("to", e.get("end")))
    kept_nodes = [n for n in nodes if n["id"] in used]
    info = {
        "raw_graph_nodes": len(nodes),
        "raw_graph_edges": len(edges),
        "pruned_graph_nodes": len(kept_nodes),
        "pruned_graph_edges": len(kept_edges),
        "removed_short_edges": removed_short,
        "removed_low_distance_edges": removed_low_d,
        "removed_noise_edges": removed_noise,
    }
    return kept_nodes, kept_edges, info


# ---------------------------------------------------------------------------
# 5) smooth polyline (preserve junctions / endpoints)
# ---------------------------------------------------------------------------

def _smooth_path(
    path: Sequence,
    window: int,
    max_offset: float,
    mask: np.ndarray,
    fixed_ends: bool = True,
) -> List[List[int]]:
    if not path or len(path) < 3 or window < 3:
        return [[int(p[0]), int(p[1])] for p in path]
    pts = [(float(p[0]), float(p[1])) for p in path]
    n = len(pts)
    half = max(1, window // 2)
    out = []
    h, w = mask.shape[:2]
    for i in range(n):
        if fixed_ends and (i == 0 or i == n - 1):
            out.append([int(round(pts[i][0])), int(round(pts[i][1]))])
            continue
        y0, x0 = pts[i]
        ys = [pts[j][0] for j in range(max(0, i - half), min(n, i + half + 1))]
        xs = [pts[j][1] for j in range(max(0, i - half), min(n, i + half + 1))]
        y1, x1 = float(np.mean(ys)), float(np.mean(xs))
        if abs(y1 - y0) > max_offset or abs(x1 - x0) > max_offset:
            y1 = y0 + float(np.clip(y1 - y0, -max_offset, max_offset))
            x1 = x0 + float(np.clip(x1 - x0, -max_offset, max_offset))
        yi, xi = int(round(y1)), int(round(x1))
        if not (0 <= yi < h and 0 <= xi < w and mask[yi, xi] > 0):
            # 偏离道路 → 回退
            yi, xi = int(round(y0)), int(round(x0))
        out.append([yi, xi])
    return out


def smooth_graph_polylines(
    nodes: List[Dict],
    edges: List[Dict],
    cleaned_mask: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    cfg = dict(DEFAULT_LARGE_CLEAN_SKELETON_CONFIG)
    cfg.update(config or {})
    rdp = float(cfg["rdp_epsilon_preview"])
    window = int(cfg["smooth_window"])
    max_off = float(cfg["max_smooth_offset"])

    # RDP first
    edges = simplify_edges(edges, tolerance=rdp)

    # degree: endpoints/junctions are fixed by path ends already
    out_edges = []
    for e in edges:
        path = e.get("path") or []
        smoothed = _smooth_path(path, window, max_off, cleaned_mask, fixed_ends=True)
        e2 = dict(e)
        e2["path"] = smoothed
        # recompute length
        length = 0.0
        for i in range(1, len(smoothed)):
            length += math.hypot(
                smoothed[i][0] - smoothed[i - 1][0],
                smoothed[i][1] - smoothed[i - 1][1],
            )
        e2["length_px"] = float(length)
        out_edges.append(e2)
    return nodes, out_edges


def graph_to_skeleton(shape: Tuple[int, int], edges: List[Dict]) -> np.ndarray:
    skel = np.zeros(shape, dtype=np.uint8)
    for e in edges:
        path = e.get("path") or []
        for i in range(len(path) - 1):
            y0, x0 = int(path[i][0]), int(path[i][1])
            y1, x1 = int(path[i + 1][0]), int(path[i + 1][1])
            cv2.line(skel, (x0, y0), (x1, y1), 255, 1)
        for p in path:
            y, x = int(p[0]), int(p[1])
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                skel[y, x] = 255
    if np.count_nonzero(skel) == 0:
        return skel
    return skeletonize_thin(skel)


def _scale_graph(nodes: List[Dict], edges: List[Dict], scale: float, full_shape: Tuple[int, int]):
    """将 preview 级 graph 坐标映射回全分辨率。"""
    if scale >= 0.999:
        return nodes, edges
    inv = 1.0 / scale
    oh, ow = full_shape
    new_nodes = []
    id_map = {}
    for n in nodes:
        nn = dict(n)
        y = int(round(float(n["y"]) * inv))
        x = int(round(float(n["x"]) * inv))
        nn["y"] = int(np.clip(y, 0, oh - 1))
        nn["x"] = int(np.clip(x, 0, ow - 1))
        new_nodes.append(nn)
        id_map[n["id"]] = nn["id"]
    new_edges = []
    for e in edges:
        ee = dict(e)
        path = []
        for p in e.get("path") or []:
            y = int(round(float(p[0]) * inv))
            x = int(round(float(p[1]) * inv))
            path.append([int(np.clip(y, 0, oh - 1)), int(np.clip(x, 0, ow - 1))])
        ee["path"] = path
        length = 0.0
        for i in range(1, len(path)):
            length += math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        ee["length_px"] = float(length)
        new_edges.append(ee)
    return new_nodes, new_edges


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------

def _save_artifacts(
    output_dir: str,
    stages: Dict[str, np.ndarray],
    graph_raw: Dict[str, Any],
    graph_pruned: Dict[str, Any],
    cleaned_skeleton_full: np.ndarray,
    raw_skeleton_full: np.ndarray,
    report: Dict[str, Any],
) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}

    def _w(name: str, arr: np.ndarray):
        path = out / name
        cv2.imwrite(str(path), arr)
        saved[name] = str(path)

    for name in (
        "raw_skeleton_preview.png",
        "center_filtered_skeleton_preview.png",
        "pruned_skeleton_preview.png",
        "cleaned_mask_preview.png",
    ):
        if name.replace(".png", "") in stages or name in stages:
            key = name if name in stages else name.replace(".png", "")
            # stages keys without .png
            pass
    mapping = {
        "raw_skeleton_preview.png": "raw_skeleton_preview",
        "center_filtered_skeleton_preview.png": "center_filtered_skeleton_preview",
        "pruned_skeleton_preview.png": "pruned_skeleton_preview",
        "cleaned_mask_preview.png": "cleaned_mask_preview",
    }
    for fname, key in mapping.items():
        arr = stages.get(key)
        if arr is not None:
            _w(fname, arr)

    _w("optimized_skeleton.png", (cleaned_skeleton_full > 0).astype(np.uint8) * 255)
    _w("raw_skeleton.png", (raw_skeleton_full > 0).astype(np.uint8) * 255)
    if stages.get("pruned_skeleton_preview") is not None:
        _w("optimized_skeleton_preview.png", stages["pruned_skeleton_preview"])

    raw_path = out / "skeleton_graph_raw.json"
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(graph_raw), f, ensure_ascii=False, indent=2)
    saved["skeleton_graph_raw.json"] = str(raw_path)

    pruned_path = out / "skeleton_graph_pruned.json"
    with pruned_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(graph_pruned), f, ensure_ascii=False, indent=2)
    saved["skeleton_graph_pruned.json"] = str(pruned_path)

    report_path = out / "large_skeleton_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(report), f, ensure_ascii=False, indent=2)
    saved["large_skeleton_report.json"] = str(report_path)
    return saved


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def generate_large_clean_skeleton(
    mask: np.ndarray,
    image_bgr: Optional[np.ndarray] = None,
    roi_polygons: Optional[Sequence] = None,
    ignore_polygons: Optional[Sequence] = None,
    main_road_seed_strokes: Optional[Sequence] = None,
    task_points: Optional[Sequence] = None,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    input_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """大图干净骨架生成。

    Returns:
        cleaned_skeleton (full-res), graph dict {nodes, edges}, report
    """
    cfg = dict(DEFAULT_LARGE_CLEAN_SKELETON_CONFIG)
    cfg.update(config or {})
    t0 = time.time()
    input_meta = input_meta or {}

    # 1) mask preclean
    cleaned_full, cleaned_preview, prep = mask_preclean(
        mask,
        roi_polygons=roi_polygons,
        ignore_polygons=ignore_polygons,
        main_road_seed_strokes=main_road_seed_strokes,
        task_points=task_points,
        config=cfg,
    )
    stages_p = prep.pop("stages_preview", {})
    corridor_preview = stages_p.get("corridor_preview")
    if corridor_preview is None:
        corridor_preview = np.full(cleaned_preview.shape, 255, dtype=np.uint8)
    scale = float(prep.get("preview_scale", 1.0))

    seed_s = _scale_strokes(main_road_seed_strokes, scale)
    task_s = _scale_points(task_points, scale)
    roi_s = _scale_polys(roi_polygons, scale)
    ign_s = _scale_polys(ignore_polygons, scale)

    seed_mask = _strokes_to_mask(
        cleaned_preview.shape, seed_s, int(cfg["seed_line_thickness"])
    )
    task_mask = (
        _points_to_mask(cleaned_preview.shape, task_s, int(cfg["task_buffer_preview"]))
        if task_s else np.zeros_like(cleaned_preview)
    )
    roi_mask = (
        _polygons_to_mask(cleaned_preview.shape, roi_s) if roi_s else np.zeros_like(cleaned_preview)
    )
    ignore_mask = _polygons_to_mask(cleaned_preview.shape, ign_s) if ign_s else None
    has_constraint = bool(seed_s or roi_s or task_s)
    if not has_constraint:
        cfg = dict(cfg)
        cfg["bridge_count_limit"] = int(cfg["bridge_count_limit_without_constraint"])
        cfg["require_seed_connected_for_bridge"] = False
        cfg["only_bridge_inside_corridor"] = False

    # 2) skeletonize
    raw_skel = skeletonize_thin(cleaned_preview)

    # 3) distance centerline filter
    center_skel, dist, min_center, center_info = distance_transform_centerline_filter(
        cleaned_preview, raw_skel, cfg,
    )

    # 4) component-level skeleton filter
    seed_dil = cv2.dilate(
        seed_mask,
        _ellipse(max(1, int(cfg["seed_corridor_width_preview"]) // 4)),
    )
    comp_skel, comp_info = filter_skeleton_components(
        center_skel, dist, seed_dil, roi_mask, task_mask, cfg,
    )

    # 5) skeleton_to_graph (+ junction clustering)
    stg_cfg = SkeletonToGraphConfig(
        junction_cluster_radius=int(cfg["junction_cluster_radius_preview"]),
        endpoint_merge_distance=int(cfg["endpoint_merge_distance"]),
        node_merge_distance=int(cfg["node_merge_distance"]),
        min_edge_length=float(cfg["min_edge_length_preview"]) * 0.5,  # light prefilter
        prune_length=float(cfg["remove_branch_length_preview"]) * 0.5,
        endpoint_connect_distance=float(cfg["endpoint_connect_distance"]),
        rdp_epsilon=float(cfg["rdp_epsilon_preview"]),
        enable_short_edge_filter=True,
        enable_prune=True,
    )
    nodes_raw, edges_raw = skeleton_to_graph(
        comp_skel, config=stg_cfg, road_mask=cleaned_preview,
    )
    graph_raw = {"nodes": nodes_raw, "edges": edges_raw}
    junction_cluster_count = sum(1 for n in nodes_raw if n.get("type") == "junction")

    # 6) graph edge prune
    nodes_pruned, edges_pruned, edge_info = prune_graph_edges(
        nodes_raw, edges_raw, dist, corridor_preview, seed_dil, task_mask,
        cleaned_preview, cfg,
    )

    # rasterize for bridge
    pruned_skel = graph_to_skeleton(cleaned_preview.shape, edges_pruned)

    # 7) conservative bridge (after prune)
    support_region = cv2.dilate(
        cleaned_preview, _ellipse(int(cfg["support_radius_preview"]))
    )
    protect = np.maximum(seed_mask, task_mask)
    protect = np.maximum(protect, roi_mask)
    anchor = np.maximum(protect, (corridor_preview > 0).astype(np.uint8) * 255)
    bridged, bridge_stats, bridge_candidates = constrained_bridge_endpoints(
        pruned_skel, corridor_preview, anchor, support_region, ignore_mask, cfg,
        color_support=None,
    )
    if not cfg.get("auto_accept_bridges", True):
        bridged = apply_bridges(pruned_skel, bridge_candidates, statuses=("accepted",))

    # rebuild graph from bridged skeleton (light) then smooth
    stg_cfg2 = SkeletonToGraphConfig(
        junction_cluster_radius=int(cfg["junction_cluster_radius_preview"]),
        endpoint_merge_distance=int(cfg["endpoint_merge_distance"]),
        node_merge_distance=int(cfg["node_merge_distance"]),
        min_edge_length=max(8.0, float(cfg["min_edge_length_preview"]) * 0.4),
        prune_length=max(10.0, float(cfg["remove_branch_length_preview"]) * 0.4),
        endpoint_connect_distance=0.0,
        rdp_epsilon=float(cfg["rdp_epsilon_preview"]),
    )
    nodes_b, edges_b = skeleton_to_graph(bridged, config=stg_cfg2, road_mask=cleaned_preview)
    nodes_final, edges_final = smooth_graph_polylines(
        nodes_b, edges_b, cleaned_preview, cfg,
    )
    final_preview = graph_to_skeleton(cleaned_preview.shape, edges_final)

    oh, ow = cleaned_full.shape[:2]
    if scale < 1.0:
        cleaned_skeleton = cv2.resize(final_preview, (ow, oh), interpolation=cv2.INTER_NEAREST)
        raw_full = cv2.resize(raw_skel, (ow, oh), interpolation=cv2.INTER_NEAREST)
        center_full = cv2.resize(center_skel, (ow, oh), interpolation=cv2.INTER_NEAREST)
        nodes_full, edges_full = _scale_graph(nodes_final, edges_final, scale, (oh, ow))
        cleaned_skeleton = skeletonize_thin(cleaned_skeleton)
    else:
        cleaned_skeleton = final_preview
        raw_full = raw_skel
        center_full = center_skel
        nodes_full, edges_full = nodes_final, edges_final

    graph = {"nodes": nodes_full, "edges": edges_full}
    graph_pruned = {"nodes": nodes_pruned, "edges": edges_pruned}

    elapsed = time.time() - t0
    nz_in = int(np.count_nonzero(_binarize(mask)))
    report: Dict[str, Any] = {
        "pipeline": "generate_large_clean_skeleton",
        "input_mask_path": input_meta.get("selected_mask_path") or input_meta.get("input_mask_path"),
        "input_mask_hash": input_meta.get("checksum") or input_meta.get("input_mask_hash"),
        "input_mask_mtime": input_meta.get("file_modified_time") or input_meta.get("input_mask_mtime"),
        "mask_source": input_meta.get("mask_source"),
        "mask_edit_base": input_meta.get("mask_edit_base"),
        "mask_nonzero_ratio": (
            round(nz_in / float(max(1, mask.shape[0] * mask.shape[1])), 6)
            if isinstance(mask, np.ndarray) else None
        ),
        "component_count_before": prep.get("component_count_before"),
        "component_count_after": prep.get("component_count_after"),
        "raw_skeleton_pixels": int(np.count_nonzero(raw_skel)),
        "center_filtered_skeleton_pixels": int(np.count_nonzero(center_skel)),
        "pruned_skeleton_pixels": int(np.count_nonzero(final_preview)),
        "raw_graph_nodes": edge_info.get("raw_graph_nodes", len(nodes_raw)),
        "raw_graph_edges": edge_info.get("raw_graph_edges", len(edges_raw)),
        "pruned_graph_nodes": len(nodes_full),
        "pruned_graph_edges": len(edges_full),
        "removed_short_edges": edge_info.get("removed_short_edges", 0),
        "removed_low_distance_edges": edge_info.get("removed_low_distance_edges", 0),
        "junction_cluster_count": junction_cluster_count,
        "bridge_candidate_count": bridge_stats.get("bridge_candidate_count", 0),
        "accepted_bridge_count": bridge_stats.get("accepted_bridge_count", 0),
        "rejected_bridge_count": bridge_stats.get("rejected_bridge_count", 0),
        "centerline_filter": center_info,
        "skeleton_component_filter": comp_info,
        "min_center_dist": min_center,
        "has_constraint": has_constraint,
        "preview_scale": scale,
        "elapsed_seconds": round(elapsed, 3),
        "warning": prep.get("warning"),
    }

    stages = {
        "cleaned_mask_preview": cleaned_preview,
        "raw_skeleton_preview": (raw_skel > 0).astype(np.uint8) * 255,
        "center_filtered_skeleton_preview": (center_skel > 0).astype(np.uint8) * 255,
        "pruned_skeleton_preview": (final_preview > 0).astype(np.uint8) * 255,
        "corridor_preview": corridor_preview,
    }

    saved = {}
    if output_dir:
        saved = _save_artifacts(
            output_dir, stages, graph_raw, graph_pruned,
            cleaned_skeleton, raw_full, report,
        )

    return (
        (cleaned_skeleton > 0).astype(np.uint8) * 255,
        graph,
        {
            "report": report,
            "stages": stages,
            "saved_files": saved,
            "raw_skeleton": (raw_full > 0).astype(np.uint8) * 255,
            "center_filtered_skeleton": (center_full > 0).astype(np.uint8) * 255,
            "cleaned_mask": cleaned_full,
            "bridge_candidates": bridge_candidates,
            "graph_raw": graph_raw,
            "graph_pruned": {"nodes": nodes_full, "edges": edges_full},
        },
    )
