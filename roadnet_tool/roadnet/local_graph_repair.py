"""大图局部路网快速修正：ROI 内重建 graph 并与边界外路网拼接。

仅用于 large_image_mode；不跑全图 OpenCV 分割。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from roadnet.graph_utils import polyline_to_list


def _point_in_poly(x: float, y: float, poly: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly.astype(np.float32), (float(x), float(y)), False) >= 0


def _bbox_of_poly(poly: Sequence[Sequence[float]], margin: int, w: int, h: int):
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    x0 = max(0, int(math.floor(min(xs))) - margin)
    y0 = max(0, int(math.floor(min(ys))) - margin)
    x1 = min(w, int(math.ceil(max(xs))) + margin)
    y1 = min(h, int(math.ceil(max(ys))) + margin)
    return x0, y0, x1, y1


def rebuild_graph_in_roi(
    road_mask: np.ndarray,
    nodes: List[Dict],
    edges: List[Dict],
    roi_polygon: Sequence[Sequence[float]],
    *,
    margin_px: int = 40,
    connect_distance_px: float = 35.0,
    max_roi_side: int = 2500,
) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    """在 ROI 内基于 mask 重建 skeleton→graph，并拼回外部路网。

    Returns:
        (new_nodes, new_edges, report)
    """
    report: Dict[str, Any] = {
        "ok": False,
        "source": "local_repair",
        "warnings": [],
    }
    if road_mask is None or road_mask.size == 0:
        report["error"] = "road_mask 为空"
        return nodes, edges, report
    if not roi_polygon or len(roi_polygon) < 3:
        report["error"] = "ROI 多边形无效"
        return nodes, edges, report

    h, w = road_mask.shape[:2]
    poly = np.asarray([[float(p[0]), float(p[1])] for p in roi_polygon], dtype=np.float32)
    x0, y0, x1, y1 = _bbox_of_poly(poly, margin_px, w, h)
    bw, bh = x1 - x0, y1 - y0
    if bw <= 8 or bh <= 8:
        report["error"] = "ROI 太小"
        return nodes, edges, report
    if max(bw, bh) > max_roi_side:
        report["error"] = (
            f"ROI 过大 ({bw}x{bh})，请缩小选区（max_side≤{max_roi_side}）。"
            "局部重建不允许全图。"
        )
        return nodes, edges, report

    report["roi_bbox"] = [x0, y0, x1, y1]

    # ROI mask crop
    crop = (road_mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
    local_poly = poly.copy()
    local_poly[:, 0] -= x0
    local_poly[:, 1] -= y0
    roi_m = np.zeros_like(crop)
    cv2.fillPoly(roi_m, [local_poly.astype(np.int32)], 255)
    crop = cv2.bitwise_and(crop, roi_m)
    if int(np.count_nonzero(crop)) < 30:
        report["error"] = "ROI 内 road mask 几乎为空"
        return nodes, edges, report

    from roadnet.optimized_skeleton import skeletonize_thin
    from roadnet.skeleton_to_graph import skeleton_to_graph, SkeletonToGraphConfig

    skel = skeletonize_thin(crop)
    cfg = SkeletonToGraphConfig(
        junction_cluster_radius=12,
        endpoint_merge_distance=10,
        node_merge_distance=8,
        min_edge_length=8.0,
        prune_length=20.0,
    )
    local_nodes, local_edges = skeleton_to_graph(skel, config=cfg, road_mask=crop)

    # 映射到原图坐标
    mapped_nodes: List[Dict] = []
    id_map: Dict[int, int] = {}
    next_nid = max((int(n["id"]) for n in nodes), default=0) + 1
    for n in local_nodes:
        old_id = int(n["id"])
        gx = int(n.get("x", 0)) + x0
        gy = int(n.get("y", 0)) + y0
        new_id = next_nid
        next_nid += 1
        id_map[old_id] = new_id
        mapped_nodes.append({
            "id": new_id,
            "x": gx,
            "y": gy,
            "type": str(n.get("type", "junction")),
            "source": "local_repair",
        })

    next_eid = max((int(e["id"]) for e in edges), default=0) + 1
    mapped_edges: List[Dict] = []
    for e in local_edges:
        path = e.get("path") or []
        pts = []
        raw_pts = e.get("points_pixel")
        if raw_pts and len(raw_pts) >= 2:
            pts = [[int(p[0]) + x0, int(p[1]) + y0] for p in raw_pts]
        else:
            # skeleton_to_graph: path 为 [[y,x], ...]
            for p in path:
                if p is None or len(p) < 2:
                    continue
                yy, xx = float(p[0]), float(p[1])
                pts.append([int(xx) + x0, int(yy) + y0])
        if len(pts) < 2:
            continue
        sid = id_map.get(int(e.get("from", e.get("start", -1))))
        eid_n = id_map.get(int(e.get("to", e.get("end", -1))))
        if sid is None or eid_n is None:
            continue
        length = float(e.get("length_px") or e.get("length_pixel") or 0.0)
        if length <= 0:
            length = 0.0
            for i in range(1, len(pts)):
                length += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        mapped_edges.append({
            "id": next_eid,
            "start": sid,
            "end": eid_n,
            "length_pixel": round(length, 2),
            "points_pixel": pts,
            "polyline": pts,
            "source": "local_repair",
            "enabled": True,
        })
        next_eid += 1

    # 删除 ROI 内旧边/旧节点
    keep_nodes = []
    removed_node_ids = set()
    for n in nodes:
        if _point_in_poly(n["x"], n["y"], poly):
            removed_node_ids.add(int(n["id"]))
        else:
            keep_nodes.append(dict(n))

    keep_edges = []
    boundary_stubs: List[Tuple[int, float, float]] = []  # node_id, x, y near boundary
    for e in edges:
        pts = polyline_to_list(e.get("points_pixel") or e.get("polyline") or [])
        if not pts:
            if int(e["start"]) in removed_node_ids or int(e["end"]) in removed_node_ids:
                continue
            keep_edges.append(dict(e))
            continue
        inside_flags = [_point_in_poly(p[0], p[1], poly) for p in pts]
        if all(inside_flags):
            continue  # 完全在 ROI 内 → 删除
        if any(inside_flags) and not all(inside_flags):
            # 跨边界边：截断，保留外侧部分，记录边界 stub
            # 简化：整边删除，在外侧端点处记 stub
            for nid_key, end_idx in (("start", 0), ("end", -1)):
                nid = int(e[nid_key])
                if nid not in removed_node_ids:
                    px, py = pts[end_idx]
                    boundary_stubs.append((nid, float(px), float(py)))
            continue
        keep_edges.append(dict(e))

    # 拼接：把局部边界节点连到外部 stub
    for mn in mapped_nodes:
        if not _point_in_poly(mn["x"], mn["y"], poly):
            # 节点在 margin 外
            pass
        # 找最近外部 stub
        best = None
        best_d = connect_distance_px
        for sid, sx, sy in boundary_stubs:
            d = math.hypot(mn["x"] - sx, mn["y"] - sy)
            if d < best_d:
                best_d = d
                best = (sid, sx, sy)
        if best is None:
            continue
        sid, sx, sy = best
        # 若已有同端点边则跳过
        exists = any(
            {ee["start"], ee["end"]} == {mn["id"], sid} for ee in keep_edges + mapped_edges
        )
        if exists:
            continue
        pts = [[mn["x"], mn["y"]], [int(sx), int(sy)]]
        mapped_edges.append({
            "id": next_eid,
            "start": mn["id"],
            "end": sid,
            "length_pixel": round(best_d, 2),
            "points_pixel": pts,
            "polyline": pts,
            "source": "local_repair",
            "enabled": True,
        })
        next_eid += 1

    out_nodes = keep_nodes + mapped_nodes
    out_edges = keep_edges + mapped_edges
    report.update({
        "ok": True,
        "removed_nodes": len(removed_node_ids),
        "added_nodes": len(mapped_nodes),
        "added_edges": len(mapped_edges),
        "kept_edges": len(keep_edges),
        "boundary_stubs": len(boundary_stubs),
    })
    return out_nodes, out_edges, report


def load_jump_debug_rows(project_dir: str = "", outputs_dir: str = "") -> List[Dict]:
    """从最近的 path_jump_debug.csv 读取异常跳边。"""
    import csv
    import os
    from pathlib import Path

    candidates: List[Path] = []
    for root in (project_dir, outputs_dir, os.path.join(os.getcwd(), "outputs")):
        if not root:
            continue
        p = Path(root)
        if not p.exists():
            continue
        candidates.extend(p.rglob("path_jump_debug.csv"))
    if not candidates:
        return []
    newest = max(candidates, key=lambda f: f.stat().st_mtime)
    rows: List[Dict] = []
    with newest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            suspicious = str(row.get("is_suspicious", "")).lower() in ("1", "true", "yes")
            if suspicious or row.get("reason"):
                rows.append(dict(row))
    for r in rows:
        r["_source_csv"] = str(newest)
    return rows
