"""Graph issue detection + report export for referee / repair visualization.

Display-only: never mutates final_graph.json.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class GraphIssueConfig:
    short_spur_px: float = 10.0
    long_straight_px: float = 100.0
    long_straight_max_points: int = 2
    road_support_ratio_min: float = 0.6
    road_sample_step_px: float = 8.0
    complex_junction_degree: int = 4
    bounds_margin_px: float = 100.0


ISSUE_SUGGESTIONS = {
    "isolated_node": "检查是否误建节点；若不需要请删除该节点。",
    "degree1_endpoint": "建议检查是否为真实道路尽头；如果不是，请使用折线补路连接到附近道路。",
    "short_spur_edge": "建议删除短毛刺边。",
    "long_straight_edge": "建议检查是否为错误直连；必要时删除并沿道路中心重新绘制。",
    "edge_missing_polyline": "建议删除该边或用折线补路工具重新绘制。",
    "edge_out_of_bounds": "建议检查 graph 坐标是否使用了 preview pixel，而不是 original image pixel。",
    "graph_bbox_mismatch": "建议检查是否加载了正确影像或是否存在坐标缩放错误。",
    "non_main_component": "建议确认是否为独立道路；如果不是，请删除或连接到主路网。",
    "low_road_support_edge": "建议检查该边是否跳过非道路区域；必要时删除并重新绘制。",
    "complex_junction": "不是错误，仅提示人工确认路口连接是否合理。",
}


def _node_xy(node: dict) -> Tuple[float, float]:
    return float(node.get("x", node.get("x_pixel", 0.0))), float(
        node.get("y", node.get("y_pixel", 0.0))
    )


def _edge_poly(edge: dict) -> List[List[float]]:
    pts = edge.get("points_pixel") or edge.get("polyline") or edge.get("path") or []
    out = []
    for p in pts:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append([float(p[0]), float(p[1])])
    return out


def _poly_length(pts: Sequence[Sequence[float]]) -> float:
    if len(pts) < 2:
        return 0.0
    return sum(
        math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
        for a, b in zip(pts, pts[1:])
    )


def _poly_midpoint(pts: Sequence[Sequence[float]]) -> Tuple[float, float]:
    if not pts:
        return 0.0, 0.0
    if len(pts) == 1:
        return float(pts[0][0]), float(pts[0][1])
    total = _poly_length(pts)
    if total <= 1e-9:
        return float(pts[0][0]), float(pts[0][1])
    half = total * 0.5
    acc = 0.0
    for a, b in zip(pts, pts[1:]):
        seg = math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
        if acc + seg >= half:
            t = 0.0 if seg <= 1e-12 else (half - acc) / seg
            return (
                float(a[0]) + t * (float(b[0]) - float(a[0])),
                float(a[1]) + t * (float(b[1]) - float(a[1])),
            )
        acc += seg
    return float(pts[-1][0]), float(pts[-1][1])


def _poly_bbox(pts: Sequence[Sequence[float]]) -> List[float]:
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def _components(nodes: Sequence[dict], edges: Sequence[dict]):
    ids = [n.get("id") for n in nodes]
    adj = {i: set() for i in ids}
    for e in edges:
        if not e.get("enabled", True):
            continue
        a, b = e.get("start"), e.get("end")
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    comps = []
    of = {}
    for nid in ids:
        if nid in of:
            continue
        idx = len(comps)
        stack = [nid]
        group = []
        while stack:
            cur = stack.pop()
            if cur in of:
                continue
            of[cur] = idx
            group.append(cur)
            stack.extend(adj[cur])
        comps.append(group)
    return comps, of, adj


def _sample_road_support(
    pts: Sequence[Sequence[float]],
    mask: np.ndarray,
    step_px: float,
) -> Optional[float]:
    if mask is None or len(pts) < 2:
        return None
    h, w = mask.shape[:2]
    samples = []
    for a, b in zip(pts, pts[1:]):
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        seg = math.hypot(bx - ax, by - ay)
        n = max(1, int(math.ceil(seg / max(step_px, 1.0))))
        for i in range(n + 1):
            t = i / n
            x = ax + t * (bx - ax)
            y = ay + t * (by - ay)
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < w and 0 <= iy < h:
                samples.append(1 if mask[iy, ix] > 0 else 0)
    if not samples:
        return None
    return float(sum(samples) / len(samples))


def detect_graph_issues(
    nodes: Sequence[dict],
    edges: Sequence[dict],
    *,
    image_width: int,
    image_height: int,
    road_mask: Optional[np.ndarray] = None,
    config: Optional[GraphIssueConfig] = None,
) -> Dict[str, Any]:
    """Detect structural / geometric issues. Returns report dict with issues list."""
    cfg = config or GraphIssueConfig()
    nodes = list(nodes or [])
    edges = [e for e in (edges or []) if e.get("enabled", True)]
    nodes_by_id = {n.get("id"): n for n in nodes}
    comps, comp_of, adj = _components(nodes, edges)
    degrees = {nid: len(neis) for nid, neis in adj.items()}

    # Main component = largest by node count
    main_idx = 0
    if comps:
        main_idx = max(range(len(comps)), key=lambda i: len(comps[i]))
    main_nodes = set(comps[main_idx]) if comps else set()
    main_edges = [
        e for e in edges
        if e.get("start") in main_nodes and e.get("end") in main_nodes
    ]

    issues: List[dict] = []
    issue_n = 0

    def _prefix_for(issue_type: str, object_type: str) -> str:
        if issue_type == "complex_junction":
            return "C"
        if object_type == "component":
            return "C"
        if object_type == "bbox":
            return "B"
        if object_type == "edge":
            return "E"
        if object_type == "node":
            return "N"
        return "X"

    def _add(
        severity: str,
        issue_type: str,
        object_type: str,
        object_id: Any,
        message: str,
        center: Tuple[float, float],
        bbox: Optional[List[float]] = None,
        extra: Optional[dict] = None,
    ):
        nonlocal issue_n
        issue_n += 1
        prefix = _prefix_for(issue_type, object_type)
        issue_id = f"{prefix}{issue_n:03d}"
        cx, cy = center
        item = {
            "issue_id": issue_id,
            "severity": severity,
            "issue_type": issue_type,
            "object_type": object_type,
            "object_id": object_id,
            "message": message,
            "suggestion": ISSUE_SUGGESTIONS.get(issue_type, "请人工确认。"),
            "center_x": round(float(cx), 3),
            "center_y": round(float(cy), 3),
            "bbox": [float(v) for v in (bbox or [cx - 20, cy - 20, cx + 20, cy + 20])],
        }
        if extra:
            item.update(extra)
        issues.append(item)

    # Global bbox
    all_xy = []
    for n in nodes:
        all_xy.append(_node_xy(n))
    for e in edges:
        all_xy.extend(_edge_poly(e))
    graph_bbox = None
    if all_xy:
        xs = [p[0] for p in all_xy]
        ys = [p[1] for p in all_xy]
        graph_bbox = [min(xs), min(ys), max(xs), max(ys)]
        m = cfg.bounds_margin_px
        if (
            image_width > 0
            and image_height > 0
            and (
                graph_bbox[0] < -m
                or graph_bbox[1] < -m
                or graph_bbox[2] > image_width + m
                or graph_bbox[3] > image_height + m
            )
        ):
            cx = 0.5 * (graph_bbox[0] + graph_bbox[2])
            cy = 0.5 * (graph_bbox[1] + graph_bbox[3])
            _add(
                "error",
                "graph_bbox_mismatch",
                "bbox",
                "graph",
                "final_graph 整体坐标范围与影像不匹配",
                (cx, cy),
                graph_bbox,
            )

    # Nodes
    for n in nodes:
        nid = n.get("id")
        xy = _node_xy(n)
        deg = int(degrees.get(nid, 0))
        if deg == 0:
            _add("error", "isolated_node", "node", nid, f"孤立节点 degree=0", xy)
        elif deg == 1:
            _add("warning", "degree1_endpoint", "node", nid, f"可疑断头点 degree=1", xy)
        if deg >= cfg.complex_junction_degree:
            _add(
                "info",
                "complex_junction",
                "node",
                nid,
                f"复杂路口 degree={deg}",
                xy,
                extra={"degree": deg},
            )

    # Non-main connected components (one issue per component)
    if len(comps) > 1:
        for ci, group in enumerate(comps):
            if ci == main_idx:
                continue
            # Skip pure isolated-node components (already reported as isolated_node)
            if all(int(degrees.get(nid, 0)) == 0 for nid in group):
                continue
            pts = []
            for nid in group:
                n = nodes_by_id.get(nid)
                if n:
                    pts.append(_node_xy(n))
            if not pts:
                continue
            bbox = _poly_bbox(pts)
            cx = 0.5 * (bbox[0] + bbox[2])
            cy = 0.5 * (bbox[1] + bbox[3])
            member_edges = [
                e.get("id") for e in edges
                if e.get("start") in group and e.get("end") in group
            ]
            _add(
                "error",
                "non_main_component",
                "component",
                f"component_{ci}",
                f"非主路网分量 nodes={len(group)} edges={len(member_edges)}",
                (cx, cy),
                bbox,
                extra={
                    "component": ci,
                    "node_ids": list(group),
                    "edge_ids": member_edges,
                },
            )

    # Edges
    for e in edges:
        eid = e.get("id")
        a, b = e.get("start"), e.get("end")
        na, nb = nodes_by_id.get(a), nodes_by_id.get(b)
        pts = _edge_poly(e)
        if len(pts) < 2:
            # synthesize from nodes for center
            if na and nb:
                pts_fallback = [_node_xy(na), _node_xy(nb)]
            else:
                pts_fallback = [[0.0, 0.0], [1.0, 1.0]]
            mid = _poly_midpoint(pts_fallback)
            _add(
                "error",
                "edge_missing_polyline",
                "edge",
                eid,
                "边缺少 polyline / points_pixel",
                mid,
                _poly_bbox(pts_fallback),
            )
            continue

        mid = _poly_midpoint(pts)
        bbox = _poly_bbox(pts)
        length = _poly_length(pts)

        # out of bounds
        if image_width > 0 and image_height > 0:
            oob = any(
                px < -1 or py < -1 or px > image_width + 1 or py > image_height + 1
                for px, py in pts
            )
            if oob:
                _add(
                    "error",
                    "edge_out_of_bounds",
                    "edge",
                    eid,
                    "polyline 超出影像范围",
                    mid,
                    bbox,
                )

        # short spur
        if length < cfg.short_spur_px and (
            degrees.get(a) == 1 or degrees.get(b) == 1
        ):
            _add(
                "warning",
                "short_spur_edge",
                "edge",
                eid,
                f"短毛刺边 length={length:.1f}px",
                mid,
                bbox,
                extra={"length_px": round(length, 3)},
            )

        # long straight with few points
        if na and nb:
            ax, ay = _node_xy(na)
            bx, by = _node_xy(nb)
            chord = math.hypot(bx - ax, by - ay)
            if chord > cfg.long_straight_px and len(pts) <= cfg.long_straight_max_points:
                _add(
                    "warning",
                    "long_straight_edge",
                    "edge",
                    eid,
                    f"超长直连边 chord={chord:.1f}px points={len(pts)}",
                    mid,
                    bbox,
                    extra={"chord_px": round(chord, 3), "point_count": len(pts)},
                )

        # road support
        if road_mask is not None:
            ratio = _sample_road_support(pts, road_mask, cfg.road_sample_step_px)
            if ratio is not None and ratio < cfg.road_support_ratio_min:
                if ratio < 0.35:
                    sev = "error"
                elif ratio < 0.5:
                    sev = "warning"
                else:
                    sev = "info"  # mild mismatch → yellow tip
                _add(
                    sev,
                    "low_road_support_edge",
                    "edge",
                    eid,
                    f"疑似跳边 / 不贴合道路 support={ratio:.2f}",
                    mid,
                    bbox,
                    extra={"road_support_ratio": round(ratio, 4)},
                )

    serious = sum(1 for i in issues if i["severity"] == "error")
    warning = sum(1 for i in issues if i["severity"] == "warning")
    info = sum(1 for i in issues if i["severity"] == "info")

    return {
        "image_width": int(image_width),
        "image_height": int(image_height),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "component_count": len(comps),
        "main_component_index": main_idx,
        "main_component_node_count": len(main_nodes),
        "main_component_edge_count": len(main_edges),
        "issue_count": len(issues),
        "serious_issue_count": serious,
        "warning_issue_count": warning,
        "info_issue_count": info,
        "graph_bbox": graph_bbox,
        "issues": issues,
        "stale": False,
    }


def export_graph_issue_reports(report: dict, output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "graph_issue_report.json")
    csv_path = os.path.join(output_dir, "graph_issue_list.csv")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    fields = [
        "issue_id", "severity", "issue_type", "object_type", "object_id",
        "message", "suggestion", "center_x", "center_y",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in report.get("issues") or []:
            writer.writerow(row)
    return {"json": json_path, "csv": csv_path}


def render_graph_issue_overlay_png(
    image_rgb: np.ndarray,
    nodes: Sequence[dict],
    edges: Sequence[dict],
    report: dict,
    output_path: str,
    *,
    coord_scale: float = 1.0,
) -> str:
    """Export judge-style overlay with issue highlights (BGR write)."""
    img = np.asarray(image_rgb)
    if img.ndim == 2:
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        base = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
    out = base.copy()
    scale = float(coord_scale) if coord_scale > 0 else 1.0
    nodes_by_id = {n.get("id"): n for n in nodes}
    edges_by_id = {e.get("id"): e for e in edges if e.get("enabled", True)}

    def _pt(x, y):
        return int(round(float(x) * scale)), int(round(float(y) * scale))

    colors = {
        "error": (0, 0, 255),
        "warning": (0, 140, 255),
        "info": (0, 220, 255),
    }
    for issue in report.get("issues") or []:
        sev = issue.get("severity", "warning")
        color = colors.get(sev, (0, 0, 255))
        ot = issue.get("object_type")
        oid = issue.get("object_id")
        if ot == "edge":
            e = edges_by_id.get(oid)
            pts = _edge_poly(e) if e else []
            if len(pts) < 2:
                na = nodes_by_id.get((e or {}).get("start"))
                nb = nodes_by_id.get((e or {}).get("end"))
                if na and nb:
                    pts = [_node_xy(na), _node_xy(nb)]
            if len(pts) >= 2:
                arr = np.array([_pt(p[0], p[1]) for p in pts], np.int32).reshape(-1, 1, 2)
                cv2.polylines(out, [arr], False, (0, 0, 0), 6, cv2.LINE_AA)
                cv2.polylines(out, [arr], False, color, 3, cv2.LINE_AA)
        elif ot == "node":
            n = nodes_by_id.get(oid)
            if n:
                cx, cy = _pt(*_node_xy(n))
                cv2.circle(out, (cx, cy), 10, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.circle(out, (cx, cy), 8, color, -1, cv2.LINE_AA)
        else:
            bbox = issue.get("bbox") or []
            if len(bbox) == 4:
                x1, y1 = _pt(bbox[0], bbox[1])
                x2, y2 = _pt(bbox[2], bbox[3])
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cx, cy = _pt(issue.get("center_x", 0), issue.get("center_y", 0))
        cv2.putText(
            out, str(issue.get("issue_id", "")),
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    if not cv2.imwrite(output_path, out):
        raise RuntimeError(f"failed to write {output_path}")
    return output_path
