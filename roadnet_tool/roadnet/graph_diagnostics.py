"""Connectivity and structural diagnostics for RoadNet final graphs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class GraphDiagnosticsConfig:
    short_spur_length: float = 15.0
    suspicious_long_edge_length: float = 500.0
    close_endpoint_distance: float = 50.0


def node_xy(node: dict) -> tuple[float, float]:
    return float(node.get("x", node.get("x_pixel", 0.0))), float(node.get("y", node.get("y_pixel", 0.0)))


def edge_length(edge: dict, nodes_by_id: dict) -> float:
    value = edge.get("length_pixel")
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    points = edge.get("points_pixel", edge.get("polyline", [])) or []
    if len(points) >= 2:
        return sum(math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])) for a, b in zip(points, points[1:]))
    a, b = nodes_by_id.get(edge.get("start")), nodes_by_id.get(edge.get("end"))
    if a is None or b is None:
        return 0.0
    ax, ay = node_xy(a)
    bx, by = node_xy(b)
    return math.hypot(bx - ax, by - ay)


def graph_components(nodes: Iterable[dict], edges: Iterable[dict]):
    nodes = list(nodes)
    node_ids = [node.get("id") for node in nodes]
    adjacency = {node_id: set() for node_id in node_ids}
    for edge in edges:
        if not edge.get("enabled", True):
            continue
        a, b = edge.get("start"), edge.get("end")
        if a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)
    components = []
    component_of = {}
    for node_id in node_ids:
        if node_id in component_of:
            continue
        index = len(components)
        stack = [node_id]
        component = []
        while stack:
            current = stack.pop()
            if current in component_of:
                continue
            component_of[current] = index
            component.append(current)
            stack.extend(adjacency[current])
        components.append(component)
    return components, component_of, adjacency


def analyze_graph(
    nodes: Iterable[dict],
    edges: Iterable[dict],
    config: Optional[GraphDiagnosticsConfig] = None,
    task_points: Optional[Iterable] = None,
    output_path: Optional[str | Path] = None,
) -> dict:
    cfg = config or GraphDiagnosticsConfig()
    nodes, edges = list(nodes), list(edges)
    nodes_by_id = {node.get("id"): node for node in nodes}
    enabled_edges = [edge for edge in edges if edge.get("enabled", True)]
    components, component_of, adjacency = graph_components(nodes, enabled_edges)
    degrees = {node_id: len(neighbors) for node_id, neighbors in adjacency.items()}
    endpoints = [node_id for node_id, degree in degrees.items() if degree == 1]
    isolated = [node_id for node_id, degree in degrees.items() if degree == 0]
    lengths = {edge.get("id"): edge_length(edge, nodes_by_id) for edge in enabled_edges}
    short_spurs = []
    suspicious = []
    for edge in enabled_edges:
        eid = edge.get("id")
        length = lengths[eid]
        a, b = edge.get("start"), edge.get("end")
        if length < cfg.short_spur_length and (degrees.get(a) == 1 or degrees.get(b) == 1):
            short_spurs.append({"edge_id": eid, "start": a, "end": b, "length_px": round(length, 3)})
        if length > cfg.suspicious_long_edge_length:
            suspicious.append({"edge_id": eid, "start": a, "end": b, "length_px": round(length, 3)})

    close_pairs = []
    for index, node_a in enumerate(endpoints):
        ax, ay = node_xy(nodes_by_id[node_a])
        for node_b in endpoints[index + 1:]:
            if node_b in adjacency[node_a]:
                continue
            bx, by = node_xy(nodes_by_id[node_b])
            distance = math.hypot(bx - ax, by - ay)
            if distance <= cfg.close_endpoint_distance:
                close_pairs.append({
                    "node_a": node_a,
                    "node_b": node_b,
                    "distance_px": round(distance, 3),
                    "same_component": component_of.get(node_a) == component_of.get(node_b),
                    "component_a": component_of.get(node_a),
                    "component_b": component_of.get(node_b),
                })
    close_pairs.sort(key=lambda item: item["distance_px"])

    component_sizes = sorted((len(component) for component in components), reverse=True)
    task_reachability = []
    if task_points is not None:
        previous = None
        for point in sorted(task_points, key=lambda p: int(getattr(p, "seq", p.get("seq", 0)) if isinstance(p, dict) else getattr(p, "seq", 0))):
            node_id = point.get("node_id") if isinstance(point, dict) else getattr(point, "node_id", None)
            seq = point.get("seq") if isinstance(point, dict) else getattr(point, "seq", None)
            current = {"seq": seq, "node_id": node_id, "component": component_of.get(node_id)}
            if previous is not None:
                task_reachability.append({
                    "from_seq": previous["seq"], "to_seq": seq,
                    "reachable": previous["component"] is not None and previous["component"] == current["component"],
                })
            previous = current

    report = {
        "node_count": len(nodes),
        "edge_count": len(enabled_edges),
        "connected_components": len(components),
        "component_sizes": component_sizes,
        "largest_component_ratio": round(component_sizes[0] / len(nodes), 6) if nodes and component_sizes else 0.0,
        "isolated_nodes": isolated,
        "degree_1_endpoints": endpoints,
        "short_spurs": short_spurs,
        "suspicious_long_edges": suspicious,
        "close_unconnected_endpoint_pairs": close_pairs,
        "task_point_reachability": task_reachability,
        "average_edge_length": round(sum(lengths.values()) / len(lengths), 3) if lengths else 0.0,
        "node_component": {str(key): value for key, value in component_of.items()},
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
