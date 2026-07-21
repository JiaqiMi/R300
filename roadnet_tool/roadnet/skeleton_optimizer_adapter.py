"""
骨架优化适配器：连接当前软件与 skeleton_fix_package 的算法。

来源：适配 D:/skeleton_fix_package_20260711/roadnet/optimized_skeleton.py

职责：
1. 从当前软件的 mask 图层读取 road mask
2. 调用 normalize_road_mask() 标准化
3. 调用骨架化函数（medial_axis / skeletonize / thin）
4. 调用 optimize_skeleton() 执行完整优化
5. 将结果写入当前软件图层
6. 保存验证输出文件到 skeleton_outputs/

注意：
- 本模块是"薄适配层"，不重新实现算法逻辑
- 所有核心算法在 roadnet/optimized_skeleton.py 中
- 不对 skeleton 做二次加工
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from roadnet.optimized_skeleton import (
    normalize_road_mask,
    skeletonize_medial_axis,
    skeletonize_thin,
    optimize_skeleton,
    draw_optimized_overlay,
    _find_endpoints,
    _find_junctions,
)


# ===========================================================================
# 配置
# ===========================================================================

@dataclass
class SkeletonOptimizeConfig:
    """骨架生成/优化参数配置。"""

    # ── 骨架化方法 ──
    skeleton_method: str = "medial_axis"  # "medial_axis" | "skeletonize" | "thin"

    # ── 边界过滤 ──
    enable_border_filter: bool = True
    border_margin: int = 10

    # ── 距离变换过滤 ──
    enable_distance_filter: bool = True
    min_center_dist: float = 2.0  # resolve_center_distance_threshold 会自动适配细路

    # ── 短毛刺删除 ──
    enable_spur_removal: bool = True
    prune_length: int = 20

    # ── Junction 聚类 ──
    enable_junction_cluster: bool = True
    junction_cluster_radius: int = 10

    # ── 端点连接 ──
    enable_endpoint_connect: bool = False
    endpoint_connect_distance: float = 25.0
    endpoint_connect_angle: float = 45.0
    endpoint_connect_overlap: float = 0.65

    # ── 输出 ──
    output_overlay: bool = True
    save_stats_json: bool = True

    @staticmethod
    def defaults() -> SkeletonOptimizeConfig:
        return SkeletonOptimizeConfig()


# ===========================================================================
# 执行结果
# ===========================================================================

@dataclass
class SkeletonOptimizeResult:
    """骨架优化的完整结果。"""

    # 标准化后的 mask
    normalized_mask: Optional[np.ndarray] = None

    # 原始骨架（优化前）
    raw_skeleton: Optional[np.ndarray] = None

    # 优化后骨架
    optimized_skeleton: Optional[np.ndarray] = None

    # 距离变换图
    distance_map: Optional[np.ndarray] = None

    # 端点 / junction / clusters
    endpoints: List[Tuple[int, int]] = field(default_factory=list)
    junction_pixels: List[Tuple[int, int]] = field(default_factory=list)
    junction_clusters: List[dict] = field(default_factory=list)

    # 统计信息
    stats: Dict[str, Any] = field(default_factory=dict)
    stats_path: Optional[str] = None

    # 保存的文件路径
    saved_files: Dict[str, str] = field(default_factory=dict)

    # 错误信息
    error: Optional[str] = None
    success: bool = False

    # 耗时
    elapsed_seconds: float = 0.0


# ===========================================================================
# 主入口
# ===========================================================================

def run_skeleton_optimization(
    mask: np.ndarray,
    config: SkeletonOptimizeConfig,
    output_base_dir: str | Path,
    image_rgb: Optional[np.ndarray] = None,
) -> SkeletonOptimizeResult:
    """
    从 road mask 生成并优化骨架。

    完整流水线：
    mask → normalize_road_mask → skeletonize → optimize_skeleton → 保存输出

    Args:
        mask:            道路 mask (H, W) 任意格式
        config:          优化参数
        output_base_dir: 输出文件根目录（在此基础上创建 skeleton_outputs/ 子目录）
        image_rgb:       原始 RGB 图像（用于 overlay 叠加图），可选

    Returns:
        SkeletonOptimizeResult — 包含所有输出数据和统计
    """
    t0 = time.perf_counter()
    result = SkeletonOptimizeResult()
    base_dir = Path(output_base_dir)
    out_dir = base_dir / "skeleton_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── 1. Mask 标准化 ──
        normalized = normalize_road_mask(mask)
        result.normalized_mask = normalized
        if normalized.sum() == 0:
            result.error = "标准化后 road mask 为空（无道路像素）"
            result.elapsed_seconds = time.perf_counter() - t0
            return result

        # 保存 processed_mask.png
        mask_path = out_dir / "processed_mask.png"
        cv2.imwrite(str(mask_path), normalized)
        result.saved_files["processed_mask"] = str(mask_path)

        # ── 2. 骨架化 ──
        if config.skeleton_method in ("skeletonize", "thin"):
            raw_skel = skeletonize_thin(normalized)
        else:
            raw_skel = skeletonize_medial_axis(normalized)
        result.raw_skeleton = raw_skel

        # ── 3. 优化 ──
        opt_result = optimize_skeleton(
            normalized, raw_skel,
            min_center_dist=config.min_center_dist if config.enable_distance_filter else 0.0,
            border_margin=config.border_margin if config.enable_border_filter else 0,
            min_branch_length=config.prune_length if config.enable_spur_removal else 0,
            max_connect_dist=config.endpoint_connect_distance if config.enable_endpoint_connect else 0,
            max_connect_angle=config.endpoint_connect_angle,
            min_line_mask_overlap=config.endpoint_connect_overlap,
            junction_cluster_radius=config.junction_cluster_radius if config.enable_junction_cluster else 0,
        )

        result.optimized_skeleton = opt_result["optimized_skeleton"]
        result.distance_map = opt_result["distance_map"]
        result.endpoints = opt_result["endpoints"]
        result.junction_pixels = opt_result["junction_pixels"]
        result.junction_clusters = opt_result["junction_clusters"]
        result.stats = opt_result["stats"]

        # Canonical state/report artifacts used by every skeleton entry point.
        from roadnet.skeleton_artifacts import (
            build_skeleton_optimize_report, save_skeleton_artifacts,
        )
        canonical_report = build_skeleton_optimize_report(
            normalized,
            result.raw_skeleton,
            result.optimized_skeleton,
            min_branch_length=config.prune_length if config.enable_spur_removal else 0,
            min_center_dist=result.stats.get(
                "effective_min_center_dist",
                config.min_center_dist if config.enable_distance_filter else 0.0,
            ),
            endpoint_connect_distance=(
                config.endpoint_connect_distance if config.enable_endpoint_connect else 0.0
            ),
            skeleton_state_input="raw",
        )
        result.stats.update(canonical_report)
        canonical_files = save_skeleton_artifacts(
            base_dir / "skeleton",
            normalized,
            result.raw_skeleton,
            result.optimized_skeleton,
            canonical_report,
            image_rgb=image_rgb,
        )
        result.saved_files.update({
            f"canonical_{key}": value for key, value in canonical_files.items()
        })

        # ── 4. 保存 optimized_skeleton.png ──
        skel_path = out_dir / "optimized_skeleton.png"
        cv2.imwrite(str(skel_path), result.optimized_skeleton)
        result.saved_files["optimized_skeleton"] = str(skel_path)

        # ── 5. 保存 overlay（如果有原图） ──
        if config.output_overlay and image_rgb is not None:
            overlay = draw_optimized_overlay(
                image_rgb,
                result.optimized_skeleton,
                endpoints=result.endpoints,
                junctions=result.junction_pixels[:500],  # 限制绘制数量
                mask=normalized,
            )
            overlay_path = out_dir / "skeleton_overlay.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            result.saved_files["skeleton_overlay"] = str(overlay_path)

        # ── 6. 保存统计 ──
        if config.save_stats_json:
            full_stats = {
                "mask_width": int(normalized.shape[1]),
                "mask_height": int(normalized.shape[0]),
                "raw_skeleton_pixels": result.stats.get("raw_pixels", 0),
                "optimized_skeleton_pixels": result.stats.get("optimized_pixels", 0),
                "raw_endpoints": result.stats.get("raw_endpoints", 0),
                "optimized_endpoints": result.stats.get("optimized_endpoints", 0),
                "connected_gap_pixels": result.stats.get("connected_gap_count", 0),
                "effective_min_center_dist": result.stats.get("effective_min_center_dist", config.min_center_dist),
                "skeleton_method": config.skeleton_method,
                "prune_length": config.prune_length,
                "border_margin": config.border_margin,
                "endpoint_connect_distance": config.endpoint_connect_distance,
                "endpoint_connect_enabled": config.enable_endpoint_connect,
                "junction_cluster_radius": config.junction_cluster_radius,
                "removed_spur_count": result.stats.get("removed_spur_count", 0),
                "junction_cluster_count": result.stats.get("junction_cluster_count", 0),
                "elapsed_seconds": round(time.perf_counter() - t0, 2),
            }
            stats_path = out_dir / "skeleton_stats.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(full_stats, f, ensure_ascii=False, indent=2)
            result.stats_path = str(stats_path)
            result.saved_files["skeleton_stats"] = str(stats_path)

        result.success = True

    except Exception as e:
        import traceback
        traceback.print_exc()
        result.error = f"{type(e).__name__}: {e}"
        result.success = False

    result.elapsed_seconds = time.perf_counter() - t0
    return result
