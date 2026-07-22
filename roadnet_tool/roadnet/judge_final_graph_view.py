"""Referee / judge view helpers for final_graph overlay on the original image.

Display-only utilities: never mutate final_graph.json / mask / skeleton / waypoints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class JudgeViewStyle:
    outer_line_width: int = 8
    inner_line_width: int = 4
    outer_color_bgr: Tuple[int, int, int] = (0, 0, 0)          # black
    inner_color_bgr: Tuple[int, int, int] = (0, 255, 255)      # yellow (BGR)
    node_color_bgr: Tuple[int, int, int] = (0, 0, 255)         # red
    node_outline_bgr: Tuple[int, int, int] = (255, 255, 255)   # white
    node_radius: int = 5
    alpha: float = 0.95
    show_nodes: bool = True
    show_legend: bool = True
    show_title: bool = True
    show_direction_arrows: bool = False


JUDGE_HIDE_LAYERS = (
    "layer_road_mask", "mask", "road_mask",
    "layer_skeleton", "skeleton",
    "layer_raw_skeleton", "raw_skeleton",
    "layer_center_filtered_skeleton",
    "layer_draft_graph", "draft_graph",
    "layer_planned_path", "planned_path", "dense_path",
    "layer_sparse_waypoints", "sparse_waypoints", "vehicle_waypoints",
    "layer_waypoint_validation", "waypoint_validation",
    "layer_task_points", "task_points",
    "layer_roi", "roi",
    "layer_ignore", "ignore",
    "layer_preview_segmentation", "preview_segmentation",
    "layer_debug", "debug",
    "layer_skeleton_nodes", "skeleton_nodes",
    "layer_sample_points", "sample_points",
    "layer_reference_graph", "reference_graph", "samroad_raw_graph",
    "layer_main_road_seed", "main_road_seed",
    "layer_road_ribbon_preview", "road_ribbon_preview",
)

JUDGE_SHOW_LAYERS = (
    "layer_image", "image",
    "layer_final_graph", "final_graph",
)


def edge_polyline(edge: dict) -> List[List[float]]:
    pts = edge.get("points_pixel") or edge.get("polyline") or edge.get("path") or []
    out = []
    for p in pts:
        if p is None:
            continue
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append([float(p[0]), float(p[1])])
        elif isinstance(p, dict):
            x = p.get("x", p.get("x_pixel"))
            y = p.get("y", p.get("y_pixel"))
            if x is not None and y is not None:
                out.append([float(x), float(y)])
    return out


def graph_bounds(nodes: Sequence[dict], edges: Sequence[dict]) -> Optional[Tuple[float, float, float, float]]:
    xs: List[float] = []
    ys: List[float] = []
    for n in nodes or []:
        try:
            xs.append(float(n.get("x", n.get("x_pixel"))))
            ys.append(float(n.get("y", n.get("y_pixel"))))
        except (TypeError, ValueError):
            continue
    for e in edges or []:
        for x, y in edge_polyline(e):
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def validate_final_graph_for_judge(
    *,
    has_image: bool,
    image_width: int,
    image_height: int,
    nodes: Sequence[dict],
    edges: Sequence[dict],
    margin_px: float = 100.0,
) -> Dict[str, Any]:
    """Pre-checks before entering judge view. Never raises for mismatch."""
    errors: List[str] = []
    warnings: List[str] = []

    if not has_image or image_width <= 0 or image_height <= 0:
        errors.append("尚未加载影像图")
    if not nodes:
        errors.append("final_graph 不包含 nodes")
    if not edges:
        errors.append("final_graph 不包含 edges")

    edges_with_poly = 0
    for e in edges or []:
        if len(edge_polyline(e)) >= 2:
            edges_with_poly += 1
    if edges and edges_with_poly == 0:
        errors.append("final_graph 的 edges 缺少有效 polyline / points_pixel")

    bounds = graph_bounds(nodes, edges)
    range_ok = True
    if bounds is not None and image_width > 0 and image_height > 0:
        x0, y0, x1, y1 = bounds
        if (
            x0 < -margin_px
            or y0 < -margin_px
            or x1 > image_width + margin_px
            or y1 > image_height + margin_px
        ):
            range_ok = False
            warnings.append(
                "final_graph 坐标范围与当前影像图不匹配，请检查是否加载了正确影像或是否存在坐标缩放错误。"
            )

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "range_ok": range_ok,
        "bounds": bounds,
        "node_count": len(nodes or []),
        "edge_count": len(edges or []),
        "edges_with_polyline": edges_with_poly,
        "image_width": int(image_width),
        "image_height": int(image_height),
    }


def _blend_overlay(base_bgr: np.ndarray, overlay_bgr: np.ndarray, alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    if a >= 0.999:
        return overlay_bgr
    return cv2.addWeighted(overlay_bgr, a, base_bgr, 1.0 - a, 0)


def render_judge_overlay_bgr(
    image_rgb_or_bgr: np.ndarray,
    nodes: Sequence[dict],
    edges: Sequence[dict],
    style: Optional[JudgeViewStyle] = None,
    *,
    title: str = "裁判查看：原始影像 + final_graph",
    coord_scale: float = 1.0,
    assume_rgb: bool = True,
) -> np.ndarray:
    """Rasterize high-contrast final_graph onto an image copy (display/export only).

    Graph coordinates are original-image pixels; if drawing on a preview image,
    pass coord_scale = preview_scale so points map correctly (single transform).
    """
    style = style or JudgeViewStyle()
    img = np.asarray(image_rgb_or_bgr)
    if img.ndim == 2:
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif assume_rgb:
        base = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        base = img.copy()
    if base.dtype != np.uint8:
        base = np.clip(base, 0, 255).astype(np.uint8)

    overlay = base.copy()
    scale = float(coord_scale) if coord_scale and coord_scale > 0 else 1.0

    def _pt(x: float, y: float) -> Tuple[int, int]:
        return (int(round(x * scale)), int(round(y * scale)))

    # Edges: black outer then yellow inner
    for e in edges or []:
        if e.get("enabled", True) is False:
            continue
        pts = edge_polyline(e)
        if len(pts) < 2:
            continue
        arr = np.array([_pt(p[0], p[1]) for p in pts], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [arr], False, style.outer_color_bgr, style.outer_line_width, cv2.LINE_AA)
        cv2.polylines(overlay, [arr], False, style.inner_color_bgr, style.inner_line_width, cv2.LINE_AA)

    if style.show_nodes:
        for n in nodes or []:
            try:
                x = float(n.get("x", n.get("x_pixel")))
                y = float(n.get("y", n.get("y_pixel")))
            except (TypeError, ValueError):
                continue
            cx, cy = _pt(x, y)
            cv2.circle(overlay, (cx, cy), style.node_radius + 1, style.node_outline_bgr, -1, cv2.LINE_AA)
            cv2.circle(overlay, (cx, cy), style.node_radius, style.node_color_bgr, -1, cv2.LINE_AA)

    out = _blend_overlay(base, overlay, style.alpha)

    h, w = out.shape[:2]
    if style.show_title or style.show_legend:
        # Semi-transparent banner
        banner_h = 72 if style.show_legend else 40
        cv2.rectangle(out, (0, 0), (w, banner_h), (20, 20, 20), -1)
        if style.show_title:
            cv2.putText(
                out, title, (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )
        if style.show_legend:
            cv2.putText(
                out,
                f"nodes={len(nodes or [])}  edges={len(edges or [])}  "
                f"image={w}x{h}",
                (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA,
            )
    return out


def export_judge_overlay_png(
    output_path: str,
    image_rgb_or_bgr: np.ndarray,
    nodes: Sequence[dict],
    edges: Sequence[dict],
    style: Optional[JudgeViewStyle] = None,
    *,
    coord_scale: float = 1.0,
    assume_rgb: bool = True,
    title: str = "裁判查看：原始影像 + final_graph",
) -> str:
    rendered = render_judge_overlay_bgr(
        image_rgb_or_bgr, nodes, edges, style,
        title=title, coord_scale=coord_scale, assume_rgb=assume_rgb,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    ok = cv2.imwrite(output_path, rendered)
    if not ok:
        raise RuntimeError(f"failed to write {output_path}")
    return output_path


def load_image_for_judge_export(
    layer_manager,
    *,
    max_side: int = 8192,
) -> Tuple[np.ndarray, float, str]:
    """Return (image_rgb, coord_scale, note).

    Prefer full-resolution original image in original pixel space (coord_scale=1).
    Fall back to preview with preview_scale as coord_scale (single transform).
    """
    full = getattr(layer_manager, "full_image_rgb", None)
    if callable(full):
        full = layer_manager.full_image_rgb
    if full is not None:
        return np.asarray(full), 1.0, "original"

    path = getattr(layer_manager, "image_path", "") or ""
    ow, oh = layer_manager.original_size
    if path and os.path.isfile(path) and ow > 0 and oh > 0:
        if max(ow, oh) <= max_side:
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return rgb, 1.0, "original_from_disk"

    preview = layer_manager.display_image_rgb
    if preview is None:
        raise ValueError("无法获取影像用于导出")
    scale = float(getattr(layer_manager, "preview_scale", 1.0) or 1.0)
    if not getattr(layer_manager, "is_large_image_mode", False):
        scale = 1.0
    return np.asarray(preview), scale, "preview"
