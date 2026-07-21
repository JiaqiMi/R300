"""
统一 Graph 生成流水线（分阶段、可调试、保底输出）。

将 "从 skeleton 生成 graph" 拆分为明确阶段：
  1. 读取并验证 skeleton
  2. skeleton_to_graph 生成 raw graph
  3. 检查 raw graph 的 node / edge 数量
  4. 保存 final_graph_raw.json
  5. 加载到 graph_editor
  6. 渲染 Final Graph 图层
  7. 可选执行 graph_line_optimizer
  8. 如果优化成功，保存 final_graph_optimized.json
  9. 如果优化失败，保留 raw graph

每个阶段都有 [GraphBuild] 前缀的日志输出。
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

from roadnet.graph_utils import (
    as_bool,
    ensure_python_types,
    is_empty_array_like,
    has_points,
    polyline_to_list,
)

# ===========================================================================
# 日志系统
# ===========================================================================

@dataclass
class GraphBuildLog:
    """Graph 生成阶段日志记录器。

    同时输出到：
    - Python logging (控制台)
    - 内部日志列表 (可用于弹窗 / 状态栏)
    """
    stage: str = "init"
    messages: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def log(self, msg: str):
        print(f"[GraphBuild] {msg}")
        self.messages.append(msg)

    def error(self, stage: str, msg: str):
        full = f"[GraphBuild] [{stage}] {msg}"
        print(full, file=sys.stderr)
        self.errors.append(full)


# 全局日志实例（每次构建时重建）
_build_log: Optional[GraphBuildLog] = None


def get_build_log() -> Optional[GraphBuildLog]:
    return _build_log


# ===========================================================================
# Stage 1: Skeleton 读取与验证
# ===========================================================================

def _read_and_validate_skeleton(
    skeleton: Any, log: GraphBuildLog
) -> Tuple[Optional[np.ndarray], str]:
    """读取并验证 skeleton 输入。

    返回 (skeleton_2d_binary, error_code)。
    - 成功: (binary_skeleton, "")
    - 失败: (None, error_code)
    """
    # ── 1.1 读取 ──
    if skeleton is None:
        log.error("read_skeleton_failed", "skeleton 数据为 None")
        return None, "read_skeleton_failed"

    # 如果是 dict（常见于 layer manager），提取真正的数组
    if isinstance(skeleton, dict):
        for key in ["optimized_skeleton", "skeleton_optimized", "skeleton", "raw_skeleton"]:
            if key in skeleton and skeleton[key] is not None:
                skeleton = skeleton[key]
                break
        else:
            log.error("read_skeleton_failed",
                       f"skeleton dict 中没有有效数据，keys={list(skeleton.keys())}")
            return None, "read_skeleton_failed"

    # ── 1.2 转换为 numpy ──
    try:
        skeleton = np.asarray(skeleton)
    except Exception as e:
        log.error("read_skeleton_failed", f"无法转换为 numpy array: {e}")
        return None, "read_skeleton_failed"

    if skeleton is None or skeleton.size == 0:
        log.error("read_skeleton_failed", "skeleton array 为空 (size=0)")
        return None, "read_skeleton_failed"

    log.log(f"skeleton shape = {skeleton.shape}, dtype={skeleton.dtype}")

    # ── 1.3 维度和颜色处理 ──
    if skeleton.ndim == 3:
        # RGB/RGBA → 取第一个通道（通常骨架在单一通道）
        log.log("skeleton 是 3 通道，取第一个通道")
        skeleton = skeleton[:, :, 0]

    if skeleton.ndim != 2:
        log.error("read_skeleton_failed", f"skeleton 维度异常: ndim={skeleton.ndim}")
        return None, "read_skeleton_failed"

    # ── 1.4 转为二值图 ──
    # 兼容 bool, uint8(0/1), uint8(0/255), float 等格式
    if skeleton.dtype == bool:
        binary = skeleton.astype(np.uint8) * 255
    else:
        binary = (skeleton > 0).astype(np.uint8) * 255

    # ── 1.5 检查非零像素 ──
    nonzero_pixels = int((binary > 0).sum())
    log.log(f"skeleton nonzero pixels = {nonzero_pixels}")

    if nonzero_pixels == 0:
        log.error(
            "empty_skeleton",
            "当前 skeleton 为空或过少（0 像素），请先生成/优化骨架。"
        )
        return None, "empty_skeleton"

    min_pixels = 10
    if nonzero_pixels < min_pixels:
        log.error(
            "empty_skeleton",
            f"当前 skeleton 非零像素过少（{nonzero_pixels} < {min_pixels}），"
            "请先生成/优化骨架。"
        )
        return None, "empty_skeleton"

    return binary, ""


# ===========================================================================
# Stage 2: 生成 raw graph
# ===========================================================================

def _generate_raw_graph(
    skeleton: np.ndarray,
    config: Optional[Dict] = None,
    road_mask: Optional[np.ndarray] = None,
    log: Optional[GraphBuildLog] = None,
) -> Tuple[Optional[List[Dict]], Optional[List[Dict]], str]:
    """执行 skeleton_to_graph 生成原始 graph。

    Returns:
        (nodes, edges, error_code)
        - 成功: (nodes, edges, "")
        - 失败: (None, None, "skeleton_to_graph_failed")
    """
    if config is None:
        config = {}
    graph_cfg = config.get("graph", {})

    # ★ 新参数（优先读取新参数名，回退旧参数名）
    junction_cluster_radius = int(graph_cfg.get("junction_cluster_radius",
                                                graph_cfg.get("merge_node_distance", 10)))
    endpoint_merge_distance = int(graph_cfg.get("endpoint_merge_distance", 12))
    node_merge_distance = int(graph_cfg.get("node_merge_distance",
                                             graph_cfg.get("merge_node_distance", 8)))
    min_edge_length = float(graph_cfg.get("min_edge_length", 8))
    prune_length = float(graph_cfg.get("prune_length", 15))
    endpoint_connect_distance = float(graph_cfg.get("endpoint_connect_distance", 25))
    rdp_epsilon = float(graph_cfg.get("rdp_epsilon",
                                      graph_cfg.get("simplify_tolerance", 2.0)))
    enable_short_edge_filter = bool(graph_cfg.get("enable_short_edge_filter", True))
    enable_prune = bool(graph_cfg.get("enable_prune", True))

    try:
        from roadnet.skeleton_to_graph import skeleton_to_graph, SkeletonToGraphConfig

        s2g_config = SkeletonToGraphConfig(
            junction_cluster_radius=junction_cluster_radius,
            endpoint_merge_distance=endpoint_merge_distance,
            node_merge_distance=node_merge_distance,
            min_edge_length=min_edge_length,
            prune_length=prune_length,
            endpoint_connect_distance=endpoint_connect_distance,
            rdp_epsilon=rdp_epsilon,
            enable_short_edge_filter=enable_short_edge_filter,
            enable_prune=enable_prune,
            merge_node_distance=node_merge_distance,  # 兼容旧字段
            simplify_tolerance=rdp_epsilon,            # 兼容旧字段
        )
        nodes, edges = skeleton_to_graph(
            skeleton, config=s2g_config, road_mask=road_mask
        )
    except Exception as e:
        if log:
            log.error("skeleton_to_graph_failed", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return None, None, "skeleton_to_graph_failed"

    if log:
        log.log(f"raw nodes = {len(nodes)}")
        log.log(f"raw edges = {len(edges)}")

    # ── 检查空 graph ──
    if len(nodes) == 0 and len(edges) == 0:
        if log:
            log.error("raw_graph_empty", "skeleton_to_graph 生成了空的 graph (0 nodes, 0 edges)")
        return None, None, "raw_graph_empty"

    return nodes, edges, ""


# ===========================================================================
# Stage 3: 保存 raw graph
# ===========================================================================

def _save_raw_graph(
    nodes: List[Dict],
    edges: List[Dict],
    output_dir: str,
    log: Optional[GraphBuildLog] = None,
) -> Tuple[Optional[str], str]:
    """保存 final_graph_raw.json。

    Returns:
        (json_path, error_code)
        - 成功: (path_str, "")
        - 失败: (None, "save_raw_graph_failed")
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        from roadnet.skeleton_to_graph import save_graph_from_skeleton
        json_path = save_graph_from_skeleton(
            nodes, edges, output_dir,
            filename="final_graph_raw.json",
        )
        if log:
            log.log(f"saved final_graph_raw.json = {json_path}")
        return json_path, ""
    except Exception as e:
        if log:
            log.error("save_raw_graph_failed", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return None, "save_raw_graph_failed"


# ===========================================================================
# 连通性分析
# ===========================================================================

def analyze_graph_connectivity(
    nodes: List[Dict],
    edges: List[Dict],
    log: Optional[GraphBuildLog] = None,
    endpoint_connect_distance: float = 25.0,
) -> Dict:
    """分析 graph 连通性并生成报告。

    Returns:
        {
            "node_count": int,
            "edge_count": int,
            "connected_components": int,
            "largest_component_node_ratio": float,
            "isolated_nodes": int,
            "degree_1_endpoints": int,
            "failed_endpoint_pairs": int,    # 距离 < endpoint_connect_distance * 2 但无法连接的端点对
            "average_edge_length": float,
            "component_sizes": [int, ...],
        }
    """
    if not nodes or not edges:
        result = {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "connected_components": 0 if not nodes else len(nodes),
            "largest_component_node_ratio": 1.0 if len(nodes) == 1 else 0.0,
            "isolated_nodes": len(nodes),
            "degree_1_endpoints": 0,
            "failed_endpoint_pairs": 0,
            "average_edge_length": 0.0,
            "component_sizes": [],
        }
        return result

    # 计算度数
    degree = {n["id"]: 0 for n in nodes}
    total_len = 0.0
    for e in edges:
        fid = e.get("from", e.get("start", 0))
        tid = e.get("to", e.get("end", 0))
        degree[fid] = degree.get(fid, 0) + 1
        degree[tid] = degree.get(tid, 0) + 1
        total_len += e.get("length_pixel",
                           e.get("length_px",
                                 e.get("length", 0)))

    degree_1_count = sum(1 for v in degree.values() if v == 1)
    isolated = sum(1 for v in degree.values() if v == 0)
    avg_len = total_len / max(len(edges), 1)

    # BFS 连通分量
    adj = {n["id"]: [] for n in nodes}
    for e in edges:
        fid = e.get("from", e.get("start", 0))
        tid = e.get("to", e.get("end", 0))
        adj.setdefault(fid, []).append(tid)
        adj.setdefault(tid, []).append(fid)

    visited = set()
    components = []
    for n in nodes:
        nid = n["id"]
        if nid in visited:
            continue
        # BFS
        queue = [nid]
        visited.add(nid)
        comp = []
        while queue:
            v = queue.pop(0)
            comp.append(v)
            for nb in adj.get(v, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(comp)

    num_components = len(components)
    largest_comp_size = max(len(c) for c in components) if components else 0
    largest_ratio = largest_comp_size / max(len(nodes), 1)

    # 找出距离 < endpoint_connect_distance*2 但不能连接的端点对
    # （简单估算：哪些 degree=1 端点在同一分量内但距离很近却不在同一个小分量中？实际我们按不同分量中端点间的距离算）
    # 这里统计不同分量间存在距离较近的端点对
    failed_pairs = 0
    degree_1_nodes = [n for n in nodes if degree.get(n["id"], 0) == 1]
    if num_components > 1 and len(degree_1_nodes) >= 2:
        # 找不同分量中最接近的端点对
        from math import sqrt
        component_of = {}
        for i, comp in enumerate(components):
            for nid in comp:
                component_of[nid] = i
        node_map = {n["id"]: n for n in nodes}
        for i in range(len(degree_1_nodes)):
            for j in range(i + 1, len(degree_1_nodes)):
                na = degree_1_nodes[i]
                nb = degree_1_nodes[j]
                if component_of.get(na["id"]) == component_of.get(nb["id"]):
                    continue
                dx = na["x"] - nb["x"]
                dy = na["y"] - nb["y"]
                d = sqrt(dx * dx + dy * dy)
                if d <= endpoint_connect_distance:
                    failed_pairs += 1

    result = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "connected_components": num_components,
        "largest_component_node_ratio": round(largest_ratio, 4),
        "isolated_nodes": isolated,
        "degree_1_endpoints": degree_1_count,
        "failed_endpoint_pairs": failed_pairs,
        "average_edge_length": round(avg_len, 2),
        "component_sizes": [len(c) for c in components],
    }

    if log:
        log.log(f"connectivity analysis: {num_components} components, "
                f"largest={largest_comp_size}/{len(nodes)} ({largest_ratio:.1%}), "
                f"deg1={degree_1_count}, isolated={isolated}, "
                f"avg_edge_len={avg_len:.1f}px")
        if num_components > 1:
            log.log(f"Graph 不连通: {num_components} 个分量，建议增大 endpoint_connect_distance 或手动补边")
            sizes = [len(c) for c in components]
            sizes.sort(reverse=True)
            for i, s in enumerate(sizes[:5]):
                log.log(f"  分量 {i + 1}: {s} 个节点")

    return result


def save_graph_build_report(
    connectivity: Dict,
    output_dir: str,
    config: Optional[Dict] = None,
    log: Optional[GraphBuildLog] = None,
) -> Optional[str]:
    """保存 graph_build_report.json 连通性分析报告。

    Returns:
        json_path 或 None
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        report = {
            "connectivity": connectivity,
            "parameters": config or {},
            "warnings": [],
        }
        if connectivity["connected_components"] > 1:
            report["warnings"].append(
                f"当前 final_graph 不连通，共 {connectivity['connected_components']} 个连通分量。"
                f"建议增大 endpoint_connect_distance 或手动补边。"
            )
        report_path = os.path.join(output_dir, "graph_build_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        if log:
            log.log(f"saved graph_build_report.json")
        return report_path
    except Exception as e:
        if log:
            log.error("save_report_failed", f"{type(e).__name__}: {e}")
        return None


# ===========================================================================
# Stage 4: 加载到 graph_editor 并渲染
# ===========================================================================

def _load_to_graph_editor(
    graph_editor,
    nodes: List[Dict],
    edges: List[Dict],
    log: Optional[GraphBuildLog] = None,
) -> str:
    """将 nodes/edges 加载到 GraphEditorQt 实例。

    Returns:
        error_code (空字符串表示成功)
    """
    try:
        graph_editor.load_draft(nodes, edges)
        if log:
            log.log("render final_graph success")
        return ""
    except Exception as e:
        if log:
            log.error("render_graph_failed", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return "render_graph_failed"


# ===========================================================================
# Stage 5: 线形优化（可选）
# ===========================================================================

def _run_line_optimization(
    edges: List[Dict],
    processed_mask: Optional[np.ndarray],
    log: Optional[GraphBuildLog] = None,
) -> Tuple[Optional[List[Dict]], Optional[Dict], str]:
    """对 raw edges 执行线形优化。

    Returns:
        (optimized_edges, report, error_code)
        - 成功: (edges, report, "")
        - 失败: (None, None, "graph_line_optimizer_failed")
    """
    try:
        from roadnet.graph_line_optimizer import optimize_graph_lines, GraphLineOptimizeConfig

        config = GraphLineOptimizeConfig(
            rdp_epsilon=3.0,
            straight_max_deviation=6.0,
            min_straight_edge_length=30.0,
            smooth_window=5,
            max_smooth_offset=4.0,
            mask_tolerance=5.0,
            validate_with_mask=(processed_mask is not None),
        )
        optimizer_input = []
        for edge in edges:
            converted = dict(edge)
            converted["start"] = edge.get("start", edge.get("from"))
            converted["end"] = edge.get("end", edge.get("to"))
            if "points_pixel" not in converted:
                converted["points_pixel"] = [
                    [int(p[1]), int(p[0])] for p in edge.get("path", [])
                ]
            converted.setdefault("enabled", True)
            optimizer_input.append(converted)
        optimized_edges, report = optimize_graph_lines(
            optimizer_input,
            processed_mask=processed_mask,
            config=config,
        )
        if log:
            s = report["summary"]
            log.log(
                f"graph_line_optimizer success: {s['straightened_edges']} straightened, "
                f"{s['smoothed_edges']} smoothed, "
                f"{s['mask_rollback_edges']} rollback"
            )
        return optimized_edges, report, ""
    except Exception as e:
        if log:
            log.error("graph_line_optimizer_failed", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return None, None, "graph_line_optimizer_failed"


def _save_optimized_graph(
    nodes: List[Dict],
    optimized_edges: List[Dict],
    output_dir: str,
    log: Optional[GraphBuildLog] = None,
) -> Tuple[Optional[str], str]:
    """保存 final_graph_optimized.json。

    Returns:
        (json_path, error_code)
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        from roadnet.skeleton_to_graph import save_graph_from_skeleton
        json_path = save_graph_from_skeleton(
            nodes, optimized_edges, output_dir,
            filename="final_graph_optimized.json",
        )
        if log:
            log.log(f"saved final_graph_optimized.json = {json_path}")
        return json_path, ""
    except Exception as e:
        if log:
            log.error("save_optimized_failed", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return None, "save_optimized_failed"


# ===========================================================================
# 主入口：统一 Graph 构建流水线
# ===========================================================================

@dataclass
class GraphBuildResult:
    """Graph 构建结果。"""
    success: bool = False
    stage: str = "init"

    raw_nodes: List[Dict] = field(default_factory=list)
    raw_edges: List[Dict] = field(default_factory=list)
    raw_graph_path: Optional[str] = None

    connectivity: Optional[Dict] = None         # ★ 连通性分析结果
    report_path: Optional[str] = None            # ★ graph_build_report.json 路径

    optimized_edges: Optional[List[Dict]] = None
    optimized_graph_path: Optional[str] = None
    optimization_report: Optional[Dict] = None

    errors: List[str] = field(default_factory=list)
    log: GraphBuildLog = field(default_factory=GraphBuildLog)


def build_graph_from_skeleton(
    skeleton: Any,
    graph_editor=None,
    output_dir: Optional[str] = None,
    config: Optional[Dict] = None,
    processed_mask: Optional[np.ndarray] = None,
    run_optimization: bool = False,
) -> GraphBuildResult:
    """统一的 Graph 生成流水线（分阶段执行，每阶段有日志）。

    流程：
      1. 读取并验证 skeleton
      2. skeleton_to_graph 生成 raw graph
      3. 检查 raw graph 的 node / edge 数量
      4. 保存 final_graph_raw.json
      5. 加载到 graph_editor
      6. 渲染 Final Graph 图层（由调用方负责）
      7. 可选执行 graph_line_optimizer
      8. 如果优化成功，保存 final_graph_optimized.json
      9. 如果优化失败，保留 raw graph

    Args:
        skeleton: skeleton 数据（numpy array、dict 或 None）
        graph_editor: GraphEditorQt 实例（可选）
        output_dir: 输出目录（默认 outputs/）
        config: 配置字典
        processed_mask: 道路 mask（用于线形优化校验）
        run_optimization: 是否在生成后自动运行线形优化

    Returns:
        GraphBuildResult
    """
    global _build_log
    _build_log = GraphBuildLog()
    log = _build_log

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "outputs")

    result = GraphBuildResult(log=log)

    # =================================================================
    # Stage 1: 读取并验证 skeleton
    # =================================================================
    log.stage = "read_skeleton"
    binary, err = _read_and_validate_skeleton(skeleton, log)
    if err:
        result.errors.append(err)
        result.stage = err
        return result

    # =================================================================
    # Stage 2: skeleton_to_graph 生成 raw graph
    # =================================================================
    log.stage = "skeleton_to_graph"
    nodes, edges, err = _generate_raw_graph(
        binary, config=config, road_mask=processed_mask, log=log
    )
    if err:
        result.errors.append(err)
        result.stage = err
        return result

    result.raw_nodes = nodes
    result.raw_edges = edges

    # Persist raw topology before any analysis, rendering, or optimization.
    log.stage = "save_raw_graph"
    raw_path, save_err = _save_raw_graph(nodes, edges, output_dir, log=log)
    result.raw_graph_path = raw_path
    if save_err:
        result.errors.append(save_err)

    # =================================================================
    # ★ 连通性分析（无论如何都执行）
    # =================================================================
    log.stage = "connectivity_analysis"
    graph_cfg = config.get("graph", {}) if config else {}
    connectivity = analyze_graph_connectivity(
        nodes, edges, log=log,
        endpoint_connect_distance=float(graph_cfg.get("endpoint_connect_distance", 25)),
    )

    # ── 保存报告 ──
    report_path = save_graph_build_report(
        connectivity, output_dir, config=graph_cfg, log=log
    )
    result.connectivity = connectivity
    result.report_path = report_path

    # =================================================================
    # Stage 3: 保存 raw graph（★ 无论如何都保存）
    # =================================================================
    log.stage = "save_raw_graph"
    if result.raw_graph_path is None:
        raw_path, err = _save_raw_graph(nodes, edges, output_dir, log=log)
        if err:
            log.error("save_raw_graph_failed_but_continue",
                      "raw graph 保存失败，但 graph 已生成，继续后续流程")
        result.raw_graph_path = raw_path

    # =================================================================
    # Stage 4: 加载到 graph_editor
    # =================================================================
    log.stage = "load_to_editor"
    if graph_editor is not None:
        err = _load_to_graph_editor(graph_editor, nodes, edges, log=log)
        if err:
            # ★ 渲染失败也不致命，graph 数据已保存
            result.errors.append(err)
            log.error("render_graph_failed_but_continue",
                       "Graph 渲染失败，但 final_graph_raw.json 已保存")
        else:
            # 已加载到 graph_editor 成功
            pass
    else:
        log.log("graph_editor 未提供，跳过加载到编辑器")

    # =================================================================
    # Stage 5: 线形优化（可选）
    # =================================================================
    if run_optimization:
        log.stage = "line_optimization"
        optimized_edges, report, err = _run_line_optimization(
            edges, processed_mask, log=log
        )
        if err:
            # ★ 优化失败不是致命错误，保留 raw graph
            result.errors.append(err)
            log.log("graph_line_optimizer skipped due to error, keeping raw graph")
        else:
            result.optimized_edges = optimized_edges
            result.optimization_report = report

            # 保存优化后的 graph
            opt_path, save_err = _save_optimized_graph(
                nodes, optimized_edges, output_dir, log=log
            )
            if save_err:
                result.errors.append(save_err)
                log.log("graph_line_optimizer 成功，但保存失败，已保留已优化的 edges")
            result.optimized_graph_path = opt_path

            # ★ 如果提供了 graph_editor，更新优化后的 edges
            if graph_editor is not None and optimized_edges is not None:
                try:
                    # 备份原始 edges
                    edges_before = [dict(e) for e in graph_editor._edges]
                    graph_editor._edges = optimized_edges
                    graph_editor._undo_stack.push(graph_editor._nodes, graph_editor._edges)
                    log.log("graph_line_optimizer 优化后的 edges 已加载到编辑器")
                except Exception as e:
                    log.error("graph_line_optimizer_failed",
                               f"line optimizer succeeded but update editor failed: {e}")
                    # 回退
                    try:
                        graph_editor._edges = edges_before
                    except Exception:
                        pass
    else:
        log.stage = "done"
        log.log("graph_line_optimizer skipped (run_optimization=False)")

    # =================================================================
    # 完成
    # =================================================================
    result.success = True
    result.stage = "done"
    log.stage = "done"

    summary_parts = [
        f"raw nodes={len(nodes)}, edges={len(edges)}",
        f"raw graph saved={result.raw_graph_path is not None}",
    ]
    if connectivity:
        summary_parts.append(
            f"components={connectivity['connected_components']}"
        )
    if run_optimization and result.optimized_edges is not None:
        summary_parts.append(f"optimized graph saved={result.optimized_graph_path is not None}")
    log.log(" | ".join(summary_parts))

    return result


# ===========================================================================
# 便捷函数：从 skeleton 生成 raw graph（最小运行）
# ===========================================================================

def generate_raw_graph_minimal(
    skeleton: Any,
    output_dir: Optional[str] = None,
) -> GraphBuildResult:
    """最小化运行：只生成 raw graph，不做优化和渲染。

    用于 CI / 调试 / 测试。
    """
    return build_graph_from_skeleton(
        skeleton=skeleton,
        graph_editor=None,
        output_dir=output_dir,
        run_optimization=False,
    )


# ===========================================================================
# 调试模式：只生成 raw graph + 连通性报告，不做后续处理
# ===========================================================================

def build_graph_debug_mode(
    skeleton: Any,
    output_dir: Optional[str] = None,
    config: Optional[Dict] = None,
    road_mask: Optional[np.ndarray] = None,
) -> GraphBuildResult:
    """调试模式：只生成 final_graph_raw + 连通性报告 + 叠加图。

    不执行：
    - graph_line_optimizer
    - 路径规划
    - 坐标转换
    - 加载到 graph_editor

    输出到 outputs/graph_build/：
      - final_graph_raw.json
      - final_graph_raw_overlay.png (叠加可视化)
      - graph_build_report.json
    """
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "outputs", "graph_build")

    global _build_log
    _build_log = GraphBuildLog()
    log = _build_log

    result = GraphBuildResult(log=log)

    # Stage 1: 验证 skeleton
    log.stage = "read_skeleton"
    binary, err = _read_and_validate_skeleton(skeleton, log)
    if err:
        result.errors.append(err)
        result.stage = err
        return result

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "optimized_skeleton_input.png"), binary)
    log.log("saved optimized_skeleton_input.png")

    # Stage 2: 生成 raw graph
    log.stage = "skeleton_to_graph"
    nodes, edges, err = _generate_raw_graph(
        binary, config=config, road_mask=road_mask, log=log
    )
    if err:
        result.errors.append(err)
        result.stage = err
        return result

    result.raw_nodes = nodes
    result.raw_edges = edges

    # Raw JSON is the first durable artifact in debug mode as well.
    log.stage = "save_raw_graph"
    raw_path, save_err = _save_raw_graph(nodes, edges, output_dir, log=log)
    result.raw_graph_path = raw_path
    if save_err:
        result.errors.append(save_err)

    # Stage 3: 连通性分析
    log.stage = "connectivity_analysis"
    graph_cfg = config.get("graph", {}) if config else {}
    connectivity = analyze_graph_connectivity(
        nodes, edges, log=log,
        endpoint_connect_distance=float(graph_cfg.get("endpoint_connect_distance", 25)),
    )
    report_path = save_graph_build_report(
        connectivity, output_dir, config=graph_cfg, log=log
    )
    result.connectivity = connectivity
    result.report_path = report_path

    # Stage 4: 保存 raw graph
    log.stage = "save_raw_graph"
    if result.raw_graph_path is None:
        raw_path, err = _save_raw_graph(nodes, edges, output_dir, log=log)
        if err:
            result.errors.append(err)
        result.raw_graph_path = raw_path

    # Stage 5: 生成叠加可视化
    log.stage = "overlay_preview"
    try:
        from roadnet.skeleton_to_graph import save_graph_from_skeleton
        # 使用 skeleton 作为底图生成叠加图
        overlay_img = None
        if binary is not None:
            overlay_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            # 在 skeleton 上绘制 graph
            for e in edges:
                path = e.get("path", e.get("points_pixel", []))
                if not path:
                    continue
                # Internal raw paths are always [y, x].
                pts = [[int(p[1]), int(p[0])] for p in path if len(p) >= 2]
                for i in range(len(pts) - 1):
                    cv2.line(overlay_img, tuple(pts[i]), tuple(pts[i + 1]),
                            (0, 255, 100), 2)
            degree = {n["id"]: 0 for n in nodes}
            for e in edges:
                degree[e["from"]] += 1
                degree[e["to"]] += 1
            # degree=1 endpoints are red; other nodes are blue.
            for n in nodes:
                color = (0, 0, 255) if degree.get(n["id"], 0) == 1 else (255, 120, 0)
                cv2.circle(overlay_img, (n.get("x", 0), n.get("y", 0)),
                          4, color, -1)
            preview_path = os.path.join(output_dir, "final_graph_raw_overlay.png")
            cv2.imwrite(preview_path, overlay_img)
            log.log(f"saved final_graph_raw_overlay.png")
    except Exception as e:
        log.log(f"overlay preview failed: {e}")

    result.success = True
    result.stage = "done"
    log.stage = "done"

    log.log(f"debug mode done: nodes={len(nodes)}, edges={len(edges)}, "
            f"components={connectivity.get('connected_components', '?')}")

    return result


# ===========================================================================
# 工具函数：安全构建 final_graph_raw.json（JSON 格式检查）
# ===========================================================================

def validate_final_graph_raw(json_path: str) -> Dict:
    """验证 final_graph_raw.json 的格式是否正确。

    Returns:
        {"valid": bool, "node_count": int, "edge_count": int,
         "has_coordinate_system": bool, "errors": [...]}
    """
    result = {
        "valid": False,
        "node_count": 0,
        "edge_count": 0,
        "has_coordinate_system": False,
        "errors": [],
    }
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            graph = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        result["errors"].append(str(e))
        return result

    result["has_coordinate_system"] = "coordinate_system" in graph
    if not result["has_coordinate_system"]:
        result["errors"].append("缺少 coordinate_system 字段")

    result["node_count"] = len(graph.get("nodes", []))
    result["edge_count"] = len(graph.get("edges", []))

    # 检查每个 node 的格式
    for i, n in enumerate(graph.get("nodes", [])):
        if "id" not in n:
            result["errors"].append(f"node[{i}] 缺少 id")
        if "x" not in n:
            result["errors"].append(f"node[{i}] 缺少 x")
        if "y" not in n:
            result["errors"].append(f"node[{i}] 缺少 y")

    # 检查每个 edge 的格式
    for i, e in enumerate(graph.get("edges", [])):
        if "id" not in e:
            result["errors"].append(f"edge[{i}] 缺少 id")
        if "source" not in e and "start" not in e:
            result["errors"].append(f"edge[{i}] 缺少 source/start")
        if "target" not in e and "end" not in e and "to" not in e:
            result["errors"].append(f"edge[{i}] 缺少 target/end/to")

    result["valid"] = len(result["errors"]) == 0
    return result
