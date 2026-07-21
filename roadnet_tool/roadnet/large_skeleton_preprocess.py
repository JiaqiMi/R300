"""大图骨架生成前的 mask 清理 / 主路约束预处理。

只用于 large_image_mode。小图流程不要调用本模块。

流水线：
  working_road_mask
  → binarize
  → apply ROI / Ignore
  → remove small / short / unseeded components
  → directional close（corridor 内）
  → skeletonize
  → prune short branches
  → constrained endpoint bridge
  → graph edge scoring
  → cleaned_skeleton

复杂分析默认在 preview 尺寸上执行，再上采样回全分辨率。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from roadnet.main_road_postprocess import (
    apply_bridges,
    build_main_road_corridor,
    constrained_bridge_endpoints,
    directional_close,
    filter_seed_connected_components,
    prune_short_branches_protected,
    score_and_filter_edges,
    _polygons_to_mask,
    _points_to_mask,
    _strokes_to_mask,
    _ellipse,
    _build_color_support,
)
from roadnet.optimized_skeleton import skeletonize_thin


DEFAULT_LARGE_SKELETON_CONFIG: Dict[str, Any] = {
    "use_preview_level": True,
    "preview_max_side": 2000,

    # corridor
    "seed_corridor_width_preview": 60,
    "task_buffer_preview": 50,
    "roi_corridor_enabled": True,
    "seed_line_thickness": 3,

    # component filtering
    "min_component_area_preview": 80,
    "min_skeleton_length_preview": 80,
    "keep_top_k_components_without_seed": 3,
    "remove_isolated_components": True,
    "inside_corridor_ratio_keep": 0.5,
    "advanced_allow_unseeded": True,  # 无约束时保留最长 top-k
    "keep_unseeded_top_k": 3,

    # directional close
    "line_close_length_preview": 11,

    # skeleton prune / edge score
    "remove_branch_length_preview": 50,
    "min_edge_length_preview": 40,
    "edge_score_threshold": 1.0,
    "preserve_task_nearby_branch": True,
    "preserve_seed_branch": True,

    # bridge（收紧）
    "max_bridge_gap_preview": 20,
    "angle_threshold_deg": 25,
    "line_sample_step_px": 2,
    "min_road_support_ratio": 0.65,
    "bridge_count_limit": 20,
    "bridge_count_limit_without_constraint": 5,
    "only_bridge_inside_corridor": True,
    "require_seed_connected_for_bridge": True,
    "auto_accept_bridges": True,  # 骨架流水线内自动接受通过约束的桥

    "support_radius_preview": 6,
    "color_support_threshold": 40.0,
}


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
    out = []
    for poly in polys:
        pts = [(float(x) * scale, float(y) * scale) for x, y in poly]
        out.append(pts)
    return out


def _scale_points(points, scale: float):
    if not points or scale == 1.0:
        return points
    return [(float(x) * scale, float(y) * scale) for x, y in points]


def _scale_strokes(strokes, scale: float):
    if not strokes or scale == 1.0:
        return strokes
    return [[(float(x) * scale, float(y) * scale) for x, y in stroke] for stroke in strokes]


def _filter_top_k_by_skeleton(
    binary: np.ndarray, top_k: int, min_skel: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """无 seed/ROI/task 时只保留骨架最长的前 top_k 个连通域。"""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    info = {"component_count_before": int(num - 1), "kept_component_ids": [],
            "removed_component_count": 0}
    if num <= 1:
        return np.zeros_like(binary), info
    skel = skeletonize_thin(binary)
    skel_counts = np.bincount(labels[skel > 0], minlength=num)
    skel_counts[0] = 0
    order = sorted(range(1, num), key=lambda k: int(skel_counts[k]), reverse=True)
    kept = np.zeros_like(binary)
    kept_ids = []
    for k in order[:max(0, top_k)]:
        if int(skel_counts[k]) < min_skel and int(stats[k, cv2.CC_STAT_AREA]) < min_skel:
            continue
        kept[labels == k] = 255
        kept_ids.append(k)
    info["kept_component_ids"] = kept_ids
    info["removed_component_count"] = max(0, (num - 1) - len(kept_ids))
    return kept, info


def prepare_mask_for_skeleton_large(
    mask: np.ndarray,
    image_bgr: Optional[np.ndarray] = None,
    roi_polygons: Optional[Sequence] = None,
    ignore_polygons: Optional[Sequence] = None,
    main_road_seed_strokes: Optional[Sequence] = None,
    task_points: Optional[Sequence] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """清理脏 mask，返回适合 skeletonize 的 cleaned_mask（与输入同尺寸）及报告。"""
    cfg = dict(DEFAULT_LARGE_SKELETON_CONFIG)
    cfg.update(config or {})
    t0 = time.time()

    full = _binarize(mask)
    orig_h, orig_w = full.shape[:2]
    report: Dict[str, Any] = {
        "input_shape": [orig_h, orig_w],
        "seed_stroke_count": len(main_road_seed_strokes or []),
        "roi_count": len(roi_polygons or []),
        "task_point_count": len(task_points or []),
        "ignore_count": len(ignore_polygons or []),
        "warning": None,
        "used_preview_level": False,
        "preview_scale": 1.0,
    }

    use_preview = bool(cfg.get("use_preview_level", True))
    scale = 1.0
    work = full
    if use_preview:
        work, scale = _downscale_for_preview(full, int(cfg["preview_max_side"]))
        report["used_preview_level"] = scale < 1.0
        report["preview_scale"] = float(scale)

    roi_s = _scale_polys(roi_polygons, scale)
    ign_s = _scale_polys(ignore_polygons, scale)
    seed_s = _scale_strokes(main_road_seed_strokes, scale)
    task_s = _scale_points(task_points, scale)

    # ROI / Ignore
    if roi_s:
        roi_m = _polygons_to_mask(work.shape, roi_s)
        if np.any(roi_m):
            work = cv2.bitwise_and(work, roi_m)
    if ign_s:
        ign_m = _polygons_to_mask(work.shape, ign_s)
        work = cv2.bitwise_and(work, cv2.bitwise_not(ign_m))

    has_constraint = bool(seed_s or roi_s or task_s)
    if not has_constraint:
        report["warning"] = (
            "未提供主路约束，骨架可能包含大量误检。建议先绘制主路种子线或设置 ROI。"
        )

    corridor, corridor_info = build_main_road_corridor(
        work.shape, seed_s, roi_s, task_s, view_rect=None, config=cfg,
    )
    report["corridor"] = corridor_info

    # 无任何约束时：corridor 退化为全图，但只保留最长 top-k
    if not has_constraint:
        corridor = np.full(work.shape, 255, dtype=np.uint8)
        cleaned, comp_info = _filter_top_k_by_skeleton(
            work,
            int(cfg["keep_top_k_components_without_seed"]),
            int(cfg["min_skeleton_length_preview"]),
        )
    else:
        seed_mask = _strokes_to_mask(work.shape, seed_s, int(cfg["seed_line_thickness"]))
        roi_mask = _polygons_to_mask(work.shape, roi_s) if roi_s else np.zeros_like(work)
        task_mask = (
            _points_to_mask(work.shape, task_s, int(cfg["task_buffer_preview"]))
            if task_s else np.zeros_like(work)
        )
        # 先裁到 corridor，再 seed-connected 过滤
        work = cv2.bitwise_and(work, corridor)
        local_cfg = dict(cfg)
        local_cfg["advanced_allow_unseeded"] = False
        local_cfg["keep_unseeded_top_k"] = 0
        cleaned, comp_info = filter_seed_connected_components(
            work, corridor, seed_mask, roi_mask, task_mask, local_cfg,
        )

    report["components"] = comp_info

    # 方向闭运算（仅 corridor 内）
    closed = directional_close(
        cleaned, int(cfg["line_close_length_preview"]), allowed_region=corridor,
    )
    report["closed_nonzero"] = int(np.count_nonzero(closed))
    report["cleaned_nonzero_preview"] = int(np.count_nonzero(closed))

    # 上采样回全分辨率
    if scale < 1.0:
        cleaned_full = cv2.resize(closed, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        corridor_full = cv2.resize(corridor, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    else:
        cleaned_full = closed
        corridor_full = corridor

    report["elapsed_sec"] = round(time.time() - t0, 3)
    report["stages"] = {
        "cleaned_mask_preview": closed,
        "corridor_preview": corridor,
        "corridor_full": corridor_full,
    }
    return cleaned_full, report


def generate_cleaned_skeleton_large(
    mask: np.ndarray,
    image_bgr: Optional[np.ndarray] = None,
    roi_polygons: Optional[Sequence] = None,
    ignore_polygons: Optional[Sequence] = None,
    main_road_seed_strokes: Optional[Sequence] = None,
    task_points: Optional[Sequence] = None,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """完整大图骨架流水线：清理 mask → skeletonize → prune → bridge → edge score。

    返回 dict，含 cleaned_skeleton / raw_skeleton / report / stages / saved_files。
    """
    cfg = dict(DEFAULT_LARGE_SKELETON_CONFIG)
    cfg.update(config or {})
    t0 = time.time()

    cleaned_mask, prep_report = prepare_mask_for_skeleton_large(
        mask,
        image_bgr=image_bgr,
        roi_polygons=roi_polygons,
        ignore_polygons=ignore_polygons,
        main_road_seed_strokes=main_road_seed_strokes,
        task_points=task_points,
        config=cfg,
    )

    # 在 preview 级做骨架后续，避免全图慢
    stages_preview = prep_report.pop("stages", {})
    cleaned_preview = stages_preview.get("cleaned_mask_preview")
    corridor_preview = stages_preview.get("corridor_preview")
    scale = float(prep_report.get("preview_scale", 1.0))
    if cleaned_preview is None:
        cleaned_preview, scale = _downscale_for_preview(
            cleaned_mask, int(cfg["preview_max_side"])
        )
        corridor_preview = np.full(cleaned_preview.shape, 255, dtype=np.uint8)

    seed_s = _scale_strokes(main_road_seed_strokes, scale)
    task_s = _scale_points(task_points, scale)
    ign_s = _scale_polys(ignore_polygons, scale)
    roi_s = _scale_polys(roi_polygons, scale)

    seed_mask = _strokes_to_mask(
        cleaned_preview.shape, seed_s, int(cfg["seed_line_thickness"])
    )
    task_mask = (
        _points_to_mask(cleaned_preview.shape, task_s, int(cfg["task_buffer_preview"]))
        if task_s else np.zeros_like(cleaned_preview)
    )
    roi_mask = _polygons_to_mask(cleaned_preview.shape, roi_s) if roi_s else np.zeros_like(cleaned_preview)
    ignore_mask = _polygons_to_mask(cleaned_preview.shape, ign_s) if ign_s else None
    protect = np.maximum(seed_mask, task_mask)
    protect = np.maximum(protect, roi_mask)

    has_constraint = bool(seed_s or roi_s or task_s)
    if not has_constraint:
        cfg = dict(cfg)
        cfg["bridge_count_limit"] = int(cfg["bridge_count_limit_without_constraint"])
        cfg["require_seed_connected_for_bridge"] = False
        cfg["only_bridge_inside_corridor"] = False

    # skeletonize
    raw_skel = skeletonize_thin(cleaned_preview)

    # color support（可选）
    color_support = None
    if image_bgr is not None:
        img = image_bgr
        if img.shape[:2] != cleaned_preview.shape[:2]:
            img = cv2.resize(
                img, (cleaned_preview.shape[1], cleaned_preview.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        color_support = _build_color_support(img, cleaned_preview, cfg)

    seed_dil = cv2.dilate(seed_mask, _ellipse(max(1, int(cfg["seed_corridor_width_preview"]) // 4)))
    edge_kept, edge_info = score_and_filter_edges(
        raw_skel, corridor_preview, seed_dil, task_mask, color_support, cfg,
    )

    pruned = prune_short_branches_protected(
        edge_kept, int(cfg["remove_branch_length_preview"]), protect_mask=protect,
    )

    support_region = cv2.dilate(cleaned_preview, _ellipse(int(cfg["support_radius_preview"])))
    anchor = np.maximum(protect, (corridor_preview > 0).astype(np.uint8) * 255)
    bridged, bridge_stats, bridge_candidates = constrained_bridge_endpoints(
        pruned, corridor_preview, anchor, support_region, ignore_mask, cfg,
        color_support=color_support,
    )
    # 若默认 auto_accept，bridged 已含 accepted；否则把 pending 中高置信也画上？保持 auto_accept
    if not cfg.get("auto_accept_bridges", True):
        bridged = apply_bridges(pruned, bridge_candidates, statuses=("accepted",))

    final_skel = prune_short_branches_protected(
        bridged, max(10, int(cfg["remove_branch_length_preview"]) // 2),
        protect_mask=protect,
    )
    # 再 skeletonize 一次保证单像素
    final_skel = skeletonize_thin(final_skel)

    # 上采样到全分辨率
    oh, ow = cleaned_mask.shape[:2]
    if final_skel.shape[:2] != (oh, ow):
        cleaned_skeleton = cv2.resize(final_skel, (ow, oh), interpolation=cv2.INTER_NEAREST)
        raw_skeleton_full = cv2.resize(raw_skel, (ow, oh), interpolation=cv2.INTER_NEAREST)
    else:
        cleaned_skeleton = final_skel
        raw_skeleton_full = raw_skel

    report = dict(prep_report)
    report.update({
        "edge_count_before": edge_info.get("edge_count_before", 0),
        "edge_count_after": edge_info.get("edge_count_after", 0),
        "removed_short_branch_count": edge_info.get("removed_short_branch_count", 0),
        "bridge_candidate_count": bridge_stats.get("bridge_candidate_count", 0),
        "accepted_bridge_count": bridge_stats.get("accepted_bridge_count", 0),
        "raw_skeleton_pixels_preview": int(np.count_nonzero(raw_skel)),
        "cleaned_skeleton_pixels_preview": int(np.count_nonzero(final_skel)),
        "elapsed_sec": round(time.time() - t0, 3),
        "has_constraint": has_constraint,
    })

    stages = {
        "skeleton_input_mask_preview": _binarize(
            cv2.resize(_binarize(mask), (cleaned_preview.shape[1], cleaned_preview.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
            if mask.shape[:2] != cleaned_preview.shape[:2] else _binarize(mask)
        ),
        "skeleton_cleaned_mask_preview": cleaned_preview,
        "raw_skeleton_preview": (raw_skel > 0).astype(np.uint8) * 255,
        "pruned_skeleton_preview": (final_skel > 0).astype(np.uint8) * 255,
        "corridor_preview": corridor_preview,
        "bridge_candidates": bridge_candidates,
        "edge_kept_preview": (edge_kept > 0).astype(np.uint8) * 255,
    }

    saved: Dict[str, str] = {}
    if output_dir:
        saved = _save_skeleton_artifacts(output_dir, stages, cleaned_skeleton, raw_skeleton_full, report)

    return {
        "cleaned_mask": cleaned_mask,
        "cleaned_skeleton": (cleaned_skeleton > 0).astype(np.uint8) * 255,
        "raw_skeleton": (raw_skeleton_full > 0).astype(np.uint8) * 255,
        "report": report,
        "stages": stages,
        "saved_files": saved,
        "bridge_candidates": bridge_candidates,
    }


def _make_bridge_overlay(base: np.ndarray, candidates: List[Dict[str, Any]]) -> np.ndarray:
    if base.ndim == 2:
        rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        rgb = base.copy()
    # 底图：corridor/skeleton 灰底
    for rec in candidates or []:
        p1 = tuple(int(v) for v in rec["p1"])
        p2 = tuple(int(v) for v in rec["p2"])
        status = rec.get("status", "pending")
        if status == "accepted":
            color = (0, 255, 0)
        elif status == "rejected":
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)
        cv2.line(rgb, p1, p2, color, 2)
    return rgb


def _make_edge_score_overlay(raw: np.ndarray, kept: np.ndarray, corridor: np.ndarray) -> np.ndarray:
    h, w = raw.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[corridor > 0] = (40, 20, 40)
    overlay[raw > 0] = (0, 0, 200)       # raw = red
    overlay[kept > 0] = (0, 220, 0)      # kept = green
    return overlay


def _save_skeleton_artifacts(
    output_dir: str,
    stages: Dict[str, Any],
    cleaned_skeleton: np.ndarray,
    raw_skeleton: np.ndarray,
    report: Dict[str, Any],
) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}

    def _w(name: str, arr: np.ndarray):
        path = out / name
        cv2.imwrite(str(path), arr)
        saved[name] = str(path)

    _w("skeleton_input_mask_preview.png", stages.get("skeleton_input_mask_preview", np.zeros((1, 1), np.uint8)))
    _w("skeleton_cleaned_mask_preview.png", stages.get("skeleton_cleaned_mask_preview", np.zeros((1, 1), np.uint8)))
    _w("raw_skeleton_preview.png", stages.get("raw_skeleton_preview", np.zeros((1, 1), np.uint8)))
    _w("pruned_skeleton_preview.png", stages.get("pruned_skeleton_preview", np.zeros((1, 1), np.uint8)))

    bridge_ov = _make_bridge_overlay(
        stages.get("pruned_skeleton_preview", np.zeros((1, 1), np.uint8)),
        stages.get("bridge_candidates") or [],
    )
    _w("bridge_candidates_overlay.png", bridge_ov)

    edge_ov = _make_edge_score_overlay(
        stages.get("raw_skeleton_preview", np.zeros((1, 1), np.uint8)),
        stages.get("edge_kept_preview", np.zeros((1, 1), np.uint8)),
        stages.get("corridor_preview", np.zeros((1, 1), np.uint8)),
    )
    _w("graph_edge_score_overlay.png", edge_ov)

    _w("optimized_skeleton.png", (cleaned_skeleton > 0).astype(np.uint8) * 255)
    # preview-sized cleaned
    prev = stages.get("pruned_skeleton_preview")
    if prev is not None:
        _w("optimized_skeleton_preview.png", prev)

    # serializable report
    report_path = out / "large_skeleton_preprocess_report.json"
    serial = {k: v for k, v in report.items() if not isinstance(v, np.ndarray)}
    # nested dicts may contain numpy ints
    def _jsonable(obj):
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
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(serial), f, ensure_ascii=False, indent=2)
    saved["large_skeleton_preprocess_report.json"] = str(report_path)

    # bridge candidates json
    cand_path = out / "bridge_candidates.json"
    with cand_path.open("w", encoding="utf-8") as f:
        json.dump(stages.get("bridge_candidates") or [], f, ensure_ascii=False, indent=2)
    saved["bridge_candidates.json"] = str(cand_path)
    return saved
