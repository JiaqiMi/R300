"""Non-destructive graph repair candidate generation and application."""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from roadnet.graph_diagnostics import (
    GraphDiagnosticsConfig,
    analyze_graph,
    graph_components,
    node_xy,
)


def _json_native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_native(item) for item in value]
    return value


def _line_support_ratio(mask, a, b, tolerance_px: int = 3) -> float:
    if mask is None:
        return 0.5
    binary = np.asarray(mask)
    if binary.ndim == 3:
        binary = binary[..., 0]
    binary = (binary > 0).astype(np.uint8)
    if tolerance_px > 0:
        size = tolerance_px * 2 + 1
        binary = cv2.dilate(binary, np.ones((size, size), np.uint8))
    length = max(2, int(round(math.hypot(b[0] - a[0], b[1] - a[1]))) + 1)
    xs = np.rint(np.linspace(a[0], b[0], length)).astype(int)
    ys = np.rint(np.linspace(a[1], b[1], length)).astype(int)
    inside = (xs >= 0) & (ys >= 0) & (xs < binary.shape[1]) & (ys < binary.shape[0])
    if not np.any(inside):
        return 0.0
    return float(np.mean(binary[ys[inside], xs[inside]] > 0))


def generate_repair_candidates(
    nodes: Iterable[dict],
    edges: Iterable[dict],
    diagnostics: Optional[dict] = None,
    road_mask=None,
    endpoint_distance: float = 50.0,
    min_mask_support: float = 0.60,
    junction_merge_distance: float = 8.0,
) -> list[dict]:
    nodes, edges = list(nodes), list(edges)
    nodes_by_id = {node.get("id"): node for node in nodes}
    diagnostics = diagnostics or analyze_graph(
        nodes, edges, GraphDiagnosticsConfig(close_endpoint_distance=endpoint_distance)
    )
    candidates = []
    for pair in diagnostics.get("close_unconnected_endpoint_pairs", []):
        if pair.get("same_component"):
            continue
        node_a, node_b = pair["node_a"], pair["node_b"]
        a, b = node_xy(nodes_by_id[node_a]), node_xy(nodes_by_id[node_b])
        support = _line_support_ratio(road_mask, a, b)
        distance = float(pair["distance_px"])
        distance_score = max(0.0, 1.0 - distance / max(1.0, endpoint_distance))
        confidence = 0.58 + 0.22 * distance_score + 0.20 * support
        if road_mask is not None and support < min_mask_support:
            confidence = min(confidence, 0.74)
            reason = "端点距离较近，但连接线道路支撑不足，需人工确认"
        else:
            reason = "两个端点距离近且连接线经过道路 mask"
        candidates.append({
            "id": f"repair_{len(candidates) + 1:03d}",
            "type": "connect_endpoints",
            "node_a": node_a,
            "node_b": node_b,
            "distance_px": round(distance, 3),
            "mask_support_ratio": round(support, 4),
            "confidence": round(min(0.99, confidence), 4),
            "reason": reason,
            "status": "pending",
        })
    for spur in diagnostics.get("short_spurs", []):
        candidates.append({
            "id": f"repair_{len(candidates) + 1:03d}",
            "type": "delete_short_spur",
            "edge_id": spur["edge_id"],
            "length_px": spur["length_px"],
            "confidence": 0.82,
            "reason": "degree=1 端点上的短毛刺边",
            "status": "pending",
        })
    # Small non-largest components are review candidates. They are deliberately
    # below the default auto-apply threshold because an isolated road can be real.
    components, component_of, adjacency = graph_components(nodes, edges)
    largest_index = max(range(len(components)), key=lambda i: len(components[i]), default=-1)
    for index, component in enumerate(components):
        if index == largest_index or len(component) > 2:
            continue
        candidates.append({
            "id": f"repair_{len(candidates) + 1:03d}",
            "type": "remove_small_component",
            "node_ids": list(component),
            "confidence": 0.65,
            "reason": "孤立小连通分量，建议人工确认后删除",
            "status": "pending",
        })
    # Nearby junction-like nodes often represent one split intersection.
    for i, left in enumerate(nodes):
        left_id = left.get("id")
        if len(adjacency.get(left_id, ())) < 2:
            continue
        lx, ly = node_xy(left)
        for right in nodes[i + 1:]:
            right_id = right.get("id")
            if len(adjacency.get(right_id, ())) < 2:
                continue
            rx, ry = node_xy(right)
            distance = math.hypot(rx - lx, ry - ly)
            if distance <= junction_merge_distance:
                candidates.append({
                    "id": f"repair_{len(candidates) + 1:03d}",
                    "type": "merge_nodes",
                    "node_a": left_id,
                    "node_b": right_id,
                    "distance_px": round(distance, 3),
                    "confidence": 0.72,
                    "reason": "路口附近节点距离过近，可能属于同一路口",
                    "status": "pending",
                })
    return candidates


def _next_numeric_id(items):
    values = [item.get("id") for item in items if isinstance(item.get("id"), int)]
    return max(values, default=-1) + 1


def apply_repair_candidates(
    nodes: Iterable[dict],
    edges: Iterable[dict],
    candidates: Iterable[dict],
    confidence_threshold: float = 0.80,
    selected_ids: Optional[set[str]] = None,
) -> tuple[list[dict], list[dict], dict]:
    """Return repaired copies; input graph is never mutated."""
    new_nodes, new_edges = copy.deepcopy(list(nodes)), copy.deepcopy(list(edges))
    nodes_by_id = {node.get("id"): node for node in new_nodes}
    existing_pairs = {frozenset((edge.get("start"), edge.get("end"))) for edge in new_edges}
    edge_by_id = {edge.get("id"): edge for edge in new_edges}
    next_edge_id = _next_numeric_id(new_edges)
    applied, skipped = [], []
    added_edges = deleted_spurs = 0
    for candidate in candidates:
        cid = str(candidate.get("id"))
        should_apply = cid in selected_ids if selected_ids is not None else float(candidate.get("confidence", 0)) > confidence_threshold
        if not should_apply or candidate.get("status") == "rejected":
            skipped.append(cid)
            continue
        kind = candidate.get("type")
        if kind == "connect_endpoints":
            a, b = candidate.get("node_a"), candidate.get("node_b")
            pair = frozenset((a, b))
            if a not in nodes_by_id or b not in nodes_by_id or pair in existing_pairs or a == b:
                skipped.append(cid)
                continue
            ax, ay = node_xy(nodes_by_id[a])
            bx, by = node_xy(nodes_by_id[b])
            new_edges.append({
                "id": next_edge_id,
                "start": a,
                "end": b,
                "length_pixel": round(math.hypot(bx - ax, by - ay), 3),
                "points_pixel": [[ax, ay], [bx, by]],
                "source": "auto_repair",
                "enabled": True,
                "repair_candidate_id": cid,
            })
            next_edge_id += 1
            existing_pairs.add(pair)
            added_edges += 1
        elif kind == "delete_short_spur":
            edge_id = candidate.get("edge_id")
            if edge_id not in edge_by_id:
                skipped.append(cid)
                continue
            new_edges = [edge for edge in new_edges if edge.get("id") != edge_id]
            edge_by_id.pop(edge_id, None)
            deleted_spurs += 1
        elif kind == "remove_small_component":
            remove_ids = set(candidate.get("node_ids", []))
            new_nodes = [node for node in new_nodes if node.get("id") not in remove_ids]
            new_edges = [edge for edge in new_edges
                         if edge.get("start") not in remove_ids and edge.get("end") not in remove_ids]
            nodes_by_id = {node.get("id"): node for node in new_nodes}
            edge_by_id = {edge.get("id"): edge for edge in new_edges}
        elif kind == "merge_nodes":
            keep_id, remove_id = candidate.get("node_a"), candidate.get("node_b")
            if keep_id not in nodes_by_id or remove_id not in nodes_by_id:
                skipped.append(cid)
                continue
            keep, remove = nodes_by_id[keep_id], nodes_by_id[remove_id]
            kx, ky = node_xy(keep)
            rx, ry = node_xy(remove)
            keep["x"], keep["y"] = round((kx + rx) / 2), round((ky + ry) / 2)
            for edge in new_edges:
                if edge.get("start") == remove_id:
                    edge["start"] = keep_id
                if edge.get("end") == remove_id:
                    edge["end"] = keep_id
            new_edges = [edge for edge in new_edges if edge.get("start") != edge.get("end")]
            deduplicated = []
            seen = set()
            for edge in new_edges:
                pair = frozenset((edge.get("start"), edge.get("end")))
                if pair in seen:
                    continue
                seen.add(pair)
                deduplicated.append(edge)
            new_edges = deduplicated
            new_nodes = [node for node in new_nodes if node.get("id") != remove_id]
            nodes_by_id = {node.get("id"): node for node in new_nodes}
            edge_by_id = {edge.get("id"): edge for edge in new_edges}
        else:
            skipped.append(cid)
            continue
        candidate["status"] = "applied"
        applied.append(cid)
    return new_nodes, new_edges, {
        "applied_candidate_ids": applied,
        "skipped_candidate_ids": skipped,
        "added_edge_count": added_edges,
        "deleted_spur_count": deleted_spurs,
    }


def render_repair_overlay(image, nodes, edges, candidates, applied_ids=()):
    """Render repair suggestions for the audit bundle (BGR uint8)."""
    canvas = np.asarray(image).copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif canvas.shape[2] == 4:
        canvas = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_BGRA2BGR)
    else:
        canvas = np.ascontiguousarray(canvas.astype(np.uint8))
    nodes_by_id = {node.get("id"): node for node in nodes}
    for edge in edges:
        points = edge.get("points_pixel", []) or []
        if len(points) >= 2:
            cv2.polylines(canvas, [np.rint(points).astype(np.int32).reshape(-1, 1, 2)],
                          False, (180, 180, 180), 2, cv2.LINE_AA)
    applied_ids = set(applied_ids)
    for candidate in candidates:
        if candidate.get("type") != "connect_endpoints":
            continue
        a = nodes_by_id.get(candidate.get("node_a"))
        b = nodes_by_id.get(candidate.get("node_b"))
        if a is None or b is None:
            continue
        color = (50, 210, 60) if candidate.get("id") in applied_ids else (0, 220, 255)
        p1, p2 = tuple(map(lambda v: int(round(v)), node_xy(a))), tuple(map(lambda v: int(round(v)), node_xy(b)))
        cv2.line(canvas, p1, p2, color, 3, cv2.LINE_AA)
        cv2.circle(canvas, p1, 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 6, (0, 0, 255), -1, cv2.LINE_AA)
    return canvas


def suggest_component_bridge(
    nodes: Iterable[dict], edges: Iterable[dict], node_a, node_b,
    max_distance: float = 50.0, road_mask=None,
) -> Optional[dict]:
    nodes, edges = list(nodes), list(edges)
    nodes_by_id = {node.get("id"): node for node in nodes}
    components, component_of, adjacency = graph_components(nodes, edges)
    comp_a, comp_b = component_of.get(node_a), component_of.get(node_b)
    if comp_a is None or comp_b is None or comp_a == comp_b:
        return None
    endpoints_a = [nid for nid in components[comp_a] if len(adjacency[nid]) <= 1]
    endpoints_b = [nid for nid in components[comp_b] if len(adjacency[nid]) <= 1]
    best = None
    for left in endpoints_a:
        ax, ay = node_xy(nodes_by_id[left])
        for right in endpoints_b:
            bx, by = node_xy(nodes_by_id[right])
            distance = math.hypot(bx - ax, by - ay)
            if best is None or distance < best[0]:
                best = (distance, left, right, (ax, ay), (bx, by))
    if best is None or best[0] > max_distance:
        return None
    support = _line_support_ratio(road_mask, best[3], best[4])
    confidence = min(0.99, 0.65 + 0.20 * (1.0 - best[0] / max_distance) + 0.15 * support)
    return {
        "id": "planning_bridge_001",
        "type": "connect_endpoints",
        "node_a": best[1], "node_b": best[2],
        "distance_px": round(best[0], 3),
        "mask_support_ratio": round(support, 4),
        "confidence": round(confidence, 4),
        "reason": "规划失败的两个连通分量之间最近端点",
        "status": "pending",
    }


def save_auto_repair_bundle(
    output_dir: str | Path,
    before_nodes, before_edges, after_nodes, after_edges,
    diagnostics_before: dict, diagnostics_after: dict,
    candidates: list[dict], apply_report: dict, image=None,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    def graph(nodes, edges):
        return {"coordinate_system": "image_pixel", "nodes": nodes, "edges": edges}
    files = {
        "graph_diagnostics_report.json": diagnostics_before,
        "repair_candidates.json": candidates,
        "final_graph_before_repair.json": graph(before_nodes, before_edges),
        "final_graph_after_repair.json": graph(after_nodes, after_edges),
    }
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "connected_components_before": diagnostics_before.get("connected_components", 0),
        "connected_components_after": diagnostics_after.get("connected_components", 0),
        "endpoints_before": len(diagnostics_before.get("degree_1_endpoints", [])),
        "endpoints_after": len(diagnostics_after.get("degree_1_endpoints", [])),
        "applied_connection_count": apply_report.get("added_edge_count", 0),
        "deleted_spur_count": apply_report.get("deleted_spur_count", 0),
        "remaining_manual_review_count": sum(1 for item in candidates if item.get("status") == "pending"),
        "apply_report": apply_report,
    }
    files["auto_repair_report.json"] = report
    for name, payload in files.items():
        (out / name).write_text(
            json.dumps(_json_native(payload), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if image is not None:
        overlay = render_repair_overlay(
            image, before_nodes, before_edges, candidates,
            apply_report.get("applied_candidate_ids", []),
        )
        if not cv2.imwrite(str(out / "repair_overlay.png"), overlay):
            raise IOError(f"无法保存 {out / 'repair_overlay.png'}")
    return report
