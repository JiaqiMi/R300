"""Competition Fast Roadnet Mode — 大图比赛快速路网生成。

在低分辨率 working image 上生成 mask / skeleton / graph，再将 graph
映射回 original image pixel，供后续任务点吸附与路径规划使用。

约束：
- 仅用于 large_image_mode + competition_fast_mode
- 不依赖任务点 / task corridor
- 不跑 full-size ROI OpenCV tile 正式提取
- 不破坏小图流程
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from roadnet.opencv_road_segmenter import segment_road_by_samples
from roadnet.valid_image import analyze_valid_image_mask, apply_valid_image_mask


DEFAULT_COMPETITION_PREVIEW_MAX_SIDE = 1500
DEFAULT_OPEN_KERNEL = 3
DEFAULT_CLOSE_KERNEL = 5
DEFAULT_MIN_COMPONENT_AREA = 80
DEFAULT_MAX_TOTAL_SECONDS = 120.0
DEFAULT_MAX_SEGMENTATION_SECONDS = 60.0
DEFAULT_MAX_SKELETON_GRAPH_SECONDS = 60.0


@dataclass
class CompetitionFastConfig:
    competition_preview_max_side: int = DEFAULT_COMPETITION_PREVIEW_MAX_SIDE
    process_full_resolution: bool = False
    use_preview_as_formal_graph_source: bool = True
    debug_mode: bool = False
    open_kernel: int = DEFAULT_OPEN_KERNEL
    close_kernel: int = DEFAULT_CLOSE_KERNEL
    remove_small_components: bool = True
    min_component_area: int = DEFAULT_MIN_COMPONENT_AREA
    fill_holes: bool = False
    valid_area_only: bool = True
    black_threshold: int = 10
    max_total_seconds: float = DEFAULT_MAX_TOTAL_SECONDS
    max_segmentation_seconds: float = DEFAULT_MAX_SEGMENTATION_SECONDS
    max_skeleton_graph_seconds: float = DEFAULT_MAX_SKELETON_GRAPH_SECONDS
    # opencv segmentation knobs (passed through)
    h_margin: int = 6
    s_margin: int = 25
    v_margin: int = 30
    lab_margin: int = 12
    use_negative_samples: bool = True
    blur_kernel: int = 3
    mode: str = "combined"
    combine_method: str = "and"
    sample_radius: int = 3


@dataclass
class CompetitionFastResult:
    ok: bool
    timed_out: bool = False
    output_dir: str = ""
    final_graph_path: str = ""
    report: Dict[str, Any] = field(default_factory=dict)
    nodes_original: List[Dict] = field(default_factory=list)
    edges_original: List[Dict] = field(default_factory=list)  # skeleton path [[y,x]]
    warning: str = ""
    error: str = ""


class CompetitionFastTimeout(RuntimeError):
    """快速路网生成超时。"""


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_png(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        ok = cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    else:
        ok = cv2.imwrite(str(path), arr)
    if not ok:
        raise IOError(f"无法写入图像: {path}")
    return str(path)


def _write_json(path: Path, payload: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return str(path)


def build_competition_work_image(
    image_path: str,
    max_side: int = DEFAULT_COMPETITION_PREVIEW_MAX_SIDE,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """从原始大图生成低分辨率 working image（不加载全图 RGB 到内存）。"""
    from roadnet.large_image_project import ImageRegionReader

    reader = ImageRegionReader(image_path)
    work = reader.read_preview(int(max_side))
    oh, ow = int(reader.height), int(reader.width)
    wh, ww = int(work.shape[0]), int(work.shape[1])
    scale = {
        "original_width": ow,
        "original_height": oh,
        "work_width": ww,
        "work_height": wh,
        "scale_x": float(ow) / float(max(1, ww)),
        "scale_y": float(oh) / float(max(1, wh)),
        "competition_preview_max_side": int(max_side),
    }
    return work, scale


def upscale_graph_to_original(
    graph_work: Dict[str, Any],
    scale_x: float,
    scale_y: float,
    *,
    original_width: int,
    original_height: int,
) -> Dict[str, Any]:
    """将 working-pixel graph 映射到 original image pixel。

    支持 nodes.x/y + edges.path([[y,x],...]) 或 edges.points_pixel([[x,y],...])。
    """
    sx = float(scale_x)
    sy = float(scale_y)
    ow = max(1, int(original_width))
    oh = max(1, int(original_height))

    def clip_x(v: float) -> int:
        return int(np.clip(int(round(v)), 0, ow - 1))

    def clip_y(v: float) -> int:
        return int(np.clip(int(round(v)), 0, oh - 1))

    nodes_out: List[Dict] = []
    for node in graph_work.get("nodes") or []:
        nn = dict(node)
        x = float(node.get("x", node.get("x_pixel", 0)))
        y = float(node.get("y", node.get("y_pixel", 0)))
        nn["x"] = clip_x(x * sx)
        nn["y"] = clip_y(y * sy)
        nn["x_pixel"] = nn["x"]
        nn["y_pixel"] = nn["y"]
        nodes_out.append(nn)

    edges_out: List[Dict] = []
    for edge in graph_work.get("edges") or []:
        ee = dict(edge)
        path = edge.get("path")
        points = edge.get("points_pixel")
        new_path = []
        new_points = []
        if path:
            for p in path:
                if p is None or len(p) < 2:
                    continue
                # skeleton path: [y, x]
                y = clip_y(float(p[0]) * sy)
                x = clip_x(float(p[1]) * sx)
                new_path.append([y, x])
                new_points.append([x, y])
        elif points:
            for p in points:
                if p is None or len(p) < 2:
                    continue
                x = clip_x(float(p[0]) * sx)
                y = clip_y(float(p[1]) * sy)
                new_points.append([x, y])
                new_path.append([y, x])
        ee["path"] = new_path
        ee["points_pixel"] = new_points
        length = 0.0
        for i in range(1, len(new_points)):
            length += math.hypot(
                new_points[i][0] - new_points[i - 1][0],
                new_points[i][1] - new_points[i - 1][1],
            )
        ee["length_pixel"] = float(length)
        edges_out.append(ee)

    return {
        "nodes": nodes_out,
        "edges": edges_out,
        "coordinate_system": "original_image_pixel",
        "source_mode": "competition_fast_lowres",
        "scale_x": sx,
        "scale_y": sy,
        "original_width": ow,
        "original_height": oh,
    }


def _light_clean_mask(
    mask: np.ndarray,
    *,
    open_kernel: int,
    close_kernel: int,
    remove_small: bool,
    min_area: int,
    fill_holes: bool,
) -> np.ndarray:
    out = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if open_kernel and open_kernel > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(open_kernel), int(open_kernel))
        )
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if close_kernel and close_kernel > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(close_kernel), int(close_kernel))
        )
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    if remove_small and min_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            (out > 0).astype(np.uint8), connectivity=8
        )
        keep = np.zeros_like(out)
        for i in range(1, num):
            if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_area):
                keep[labels == i] = 255
        out = keep
    if fill_holes:
        # 轻量孔洞填充：仅在工作分辨率上执行
        inv = cv2.bitwise_not(out)
        h, w = inv.shape[:2]
        flood = inv.copy()
        cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        out = cv2.bitwise_or(out, holes)
    return out


def _valid_area_polygon(valid_mask: np.ndarray) -> Dict[str, Any]:
    mask = (np.asarray(valid_mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 32:
            continue
        pts = [[int(p[0][0]), int(p[0][1])] for p in cnt]
        if len(pts) >= 3:
            polygons.append(pts)
    return {
        "coordinate_system": "working_preview_pixel",
        "polygon_count": len(polygons),
        "polygons": polygons,
    }


def _budget_check(t0: float, budget: float, stage: str) -> None:
    if budget <= 0:
        return
    elapsed = time.perf_counter() - t0
    if elapsed > budget:
        raise CompetitionFastTimeout(
            f"快速路网生成超时（阶段={stage}, elapsed={elapsed:.1f}s > {budget:.1f}s）。"
            "请降低 preview_max_side 或切换应急手绘模式。"
        )


def run_competition_fast_roadnet(
    image_path: str,
    pos_samples_rgb: np.ndarray,
    neg_samples_rgb: np.ndarray,
    output_dir: str,
    *,
    config: Optional[CompetitionFastConfig] = None,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
) -> CompetitionFastResult:
    """同步执行比赛快速路网流水线。"""
    cfg = config or CompetitionFastConfig()
    if cfg.process_full_resolution:
        # 明确禁止 full-size 作为本模式主路径
        cfg = CompetitionFastConfig(**{**asdict(cfg), "process_full_resolution": False})

    def progress(pct: int, msg: str):
        if progress_cb:
            progress_cb(int(pct), str(msg))

    def cancelled() -> bool:
        return bool(cancelled_cb and cancelled_cb())

    t_total0 = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []
    report: Dict[str, Any] = {
        "source_mode": "competition_fast_lowres",
        "mask_type": "competition_fast_mask",
        "preview_only": False,
        "formal_ready": True,
        "coordinate_system_work": "working_preview_pixel",
        "coordinate_system_final": "original_image_pixel",
        "warnings": warnings,
    }

    try:
        if cancelled():
            report["cancelled"] = True
            return CompetitionFastResult(
                ok=False, error="用户取消", output_dir=str(out), report=report
            )

        # ── 1) working image ──
        progress(5, "生成低分辨率 working image…")
        work_rgb, scale = build_competition_work_image(
            image_path, cfg.competition_preview_max_side
        )
        report.update(scale)
        report["input_image_shape"] = [
            int(scale["original_height"]), int(scale["original_width"]), 3
        ]
        _write_png(out / "competition_work_image.png", work_rgb)
        _write_json(out / "competition_scale.json", scale)
        _budget_check(t_total0, cfg.max_total_seconds, "work_image")
        if cancelled():
            report["cancelled"] = True
            return CompetitionFastResult(
                ok=False, error="用户取消", output_dir=str(out), report=report
            )

        # ── 2) valid area ──
        progress(12, "识别有效影像区域…")
        min_black = max(
            64,
            int(4096 * (scale["work_width"] / max(1.0, scale["original_width"])) ** 2),
        )
        valid_mask, valid_report = analyze_valid_image_mask(
            work_rgb,
            black_threshold=cfg.black_threshold,
            min_black_component_area=min_black,
        )
        if cfg.debug_mode:
            _write_png(out / "valid_image_mask_preview.png", valid_mask)
            poly = _valid_area_polygon(valid_mask)
            _write_json(out / "valid_area_polygon.json", poly)
        report["valid_area_ratio"] = valid_report.get("valid_area_ratio", 0.0)
        report["valid_mask_report"] = valid_report
        _budget_check(t_total0, cfg.max_total_seconds, "valid_mask")
        if cancelled():
            report["cancelled"] = True
            return CompetitionFastResult(ok=False, error="用户取消", output_dir=str(out), report=report)

        # ── 3) fast segmentation on work image ──
        progress(25, "低分辨率道路提取…")
        t_seg0 = time.perf_counter()
        seg_cfg = {
            "mode": cfg.mode,
            "combine_method": cfg.combine_method,
            "h_margin": cfg.h_margin,
            "s_margin": cfg.s_margin,
            "v_margin": cfg.v_margin,
            "lab_margin": cfg.lab_margin,
            "use_negative_samples": cfg.use_negative_samples,
            "blur_kernel": cfg.blur_kernel,
            "open_kernel": 0,
            "close_kernel": 0,
            "min_area": 0,
            "fill_holes": False,
            "sample_radius": cfg.sample_radius,
        }
        pos = np.asarray(pos_samples_rgb, dtype=np.uint8)
        neg = np.asarray(neg_samples_rgb, dtype=np.uint8)
        if pos.size == 0:
            return CompetitionFastResult(
                ok=False, error="正样本为空，请先添加道路正样本。", output_dir=str(out)
            )
        if cfg.use_negative_samples and neg.size == 0:
            return CompetitionFastResult(
                ok=False, error="负样本为空，请先添加非道路负样本。", output_dir=str(out)
            )

        raw_mask = segment_road_by_samples(work_rgb, pos, neg, seg_cfg)
        if cfg.valid_area_only:
            raw_mask = apply_valid_image_mask(raw_mask, valid_mask)
        _write_png(out / "competition_fast_mask_preview.png", raw_mask)
        if cfg.debug_mode:
            _write_png(out / "competition_road_mask_preview.png", raw_mask)
        _budget_check(t_seg0, cfg.max_segmentation_seconds, "segmentation")
        _budget_check(t_total0, cfg.max_total_seconds, "segmentation")
        if cancelled():
            report["cancelled"] = True
            return CompetitionFastResult(ok=False, error="用户取消", output_dir=str(out), report=report)

        progress(40, "轻量 mask 后处理…")
        cleaned = _light_clean_mask(
            raw_mask,
            open_kernel=cfg.open_kernel,
            close_kernel=cfg.close_kernel,
            remove_small=cfg.remove_small_components,
            min_area=cfg.min_component_area,
            fill_holes=cfg.fill_holes,
        )
        if cfg.valid_area_only:
            cleaned = apply_valid_image_mask(cleaned, valid_mask)
        if cfg.debug_mode:
            _write_png(out / "competition_road_mask_cleaned.png", cleaned)
        nz = int(np.count_nonzero(cleaned))
        report["mask_nonzero_ratio"] = round(
            nz / float(max(1, cleaned.size)), 6
        )
        report["elapsed_segmentation_seconds"] = round(time.perf_counter() - t_seg0, 3)
        if nz < 50:
            return CompetitionFastResult(
                ok=False,
                error="道路 mask 几乎为空，请检查正负样本或降低阈值。",
                output_dir=str(out),
                report=report,
            )

        # ── 4) skeleton + graph on work mask (no task corridor) ──
        progress(55, "低分辨率骨架与路网…")
        t_sk0 = time.perf_counter()
        from roadnet.large_skeleton_optimizer import generate_large_clean_skeleton

        skel_cfg = {
            "preview_max_side": max(
                int(cfg.competition_preview_max_side),
                int(max(cleaned.shape[:2])),
            ),
            "use_preview_level": True,
            "bridge_count_limit_without_constraint": 40,
            "require_seed_connected_for_bridge": False,
            "only_bridge_inside_corridor": False,
        }
        skel_out = str(out / "skeleton_work") if cfg.debug_mode else None
        cleaned_skel, graph_work, skel_report = generate_large_clean_skeleton(
            cleaned,
            image_bgr=None,
            roi_polygons=None,
            ignore_polygons=None,
            main_road_seed_strokes=None,
            task_points=None,
            config=skel_cfg,
            output_dir=skel_out,
            input_meta={
                "mask_source": "competition_fast_mask",
                "mask_type": "competition_fast_mask",
                "preview_only": False,
                "formal_ready": True,
            },
        )
        _budget_check(t_sk0, cfg.max_skeleton_graph_seconds, "skeleton_graph")
        _budget_check(t_total0, cfg.max_total_seconds, "skeleton_graph")
        if cancelled():
            report["cancelled"] = True
            return CompetitionFastResult(ok=False, error="用户取消", output_dir=str(out), report=report)

        _write_png(
            out / "competition_skeleton_preview.png",
            (cleaned_skel > 0).astype(np.uint8) * 255,
        )
        if cfg.debug_mode:
            _write_png(out / "competition_skeleton_raw.png", (cleaned_skel > 0).astype(np.uint8) * 255)
            _write_png(out / "competition_skeleton_cleaned.png", (cleaned_skel > 0).astype(np.uint8) * 255)
            work_graph_payload = {
                "coordinate_system": "working_preview_pixel",
                "source_mode": "competition_fast_lowres",
                "work_width": scale["work_width"],
                "work_height": scale["work_height"],
                "nodes": graph_work.get("nodes", []),
                "edges": graph_work.get("edges", []),
            }
            _write_json(out / "competition_graph_work.json", work_graph_payload)
            overlay = work_rgb.copy()
            for e in graph_work.get("edges") or []:
                path = e.get("path") or []
                pts = []
                for p in path:
                    if p is None or len(p) < 2:
                        continue
                    pts.append([int(p[1]), int(p[0])])
                if len(pts) >= 2:
                    arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(overlay, [arr], False, (0, 220, 255), 2, cv2.LINE_AA)
            for n in graph_work.get("nodes") or []:
                cv2.circle(
                    overlay,
                    (int(n.get("x", 0)), int(n.get("y", 0))),
                    3, (0, 255, 80), -1, cv2.LINE_AA,
                )
            _write_png(out / "competition_graph_overlay.png", overlay)

        report["skeleton_pixel_count"] = int(np.count_nonzero(cleaned_skel))
        report["elapsed_skeleton_seconds"] = round(time.perf_counter() - t_sk0, 3)
        report["skeleton_report"] = {
            k: skel_report.get(k)
            for k in (
                "pruned_graph_nodes", "pruned_graph_edges",
                "raw_graph_nodes", "raw_graph_edges", "elapsed_seconds",
            )
            if k in skel_report
        }

        # ── 5) upscale graph to original ──
        progress(85, "映射 graph 到原图像素坐标…")
        upscaled = upscale_graph_to_original(
            graph_work,
            scale["scale_x"],
            scale["scale_y"],
            original_width=int(scale["original_width"]),
            original_height=int(scale["original_height"]),
        )
        upscaled["work_width"] = scale["work_width"]
        upscaled["work_height"] = scale["work_height"]
        progress(92, "保存 final_graph…")
        # save final_graph.json (original pixels)
        final_nodes = []
        for n in upscaled["nodes"]:
            final_nodes.append({
                "id": int(n["id"]),
                "x_pixel": int(n["x"]),
                "y_pixel": int(n["y"]),
                "type": str(n.get("type", "")),
                "source": str(n.get("source", "auto")),
            })
        final_edges = []
        for e in upscaled["edges"]:
            pts = e.get("points_pixel") or []
            final_edges.append({
                "id": int(e["id"]),
                "start": int(e.get("from", e.get("start", 0))),
                "end": int(e.get("to", e.get("end", 0))),
                "length_pixel": float(e.get("length_pixel", 0)),
                "points_pixel": [[int(p[0]), int(p[1])] for p in pts],
                "source": str(e.get("source", "auto")),
                "enabled": bool(e.get("enabled", True)),
            })

        # connectivity
        from collections import defaultdict, deque
        adj = defaultdict(set)
        for e in final_edges:
            adj[e["start"]].add(e["end"])
            adj[e["end"]].add(e["start"])
        visited = set()
        components = 0
        largest = 0
        for nid in [n["id"] for n in final_nodes]:
            if nid in visited:
                continue
            components += 1
            q = deque([nid])
            size = 0
            while q:
                v = q.popleft()
                if v in visited:
                    continue
                visited.add(v)
                size += 1
                for nb in adj.get(v, ()):
                    if nb not in visited:
                        q.append(nb)
            largest = max(largest, size)
        n_nodes = max(1, len(final_nodes))

        final_graph = {
            "coordinate_system": "original_image_pixel",
            "source_mode": "competition_fast_lowres",
            "work_width": int(scale["work_width"]),
            "work_height": int(scale["work_height"]),
            "original_width": int(scale["original_width"]),
            "original_height": int(scale["original_height"]),
            "scale_x": float(scale["scale_x"]),
            "scale_y": float(scale["scale_y"]),
            "metadata": {
                "image_width": int(scale["original_width"]),
                "image_height": int(scale["original_height"]),
                "node_count": len(final_nodes),
                "edge_count": len(final_edges),
                "mask_type": "competition_fast_mask",
                "preview_only": False,
                "formal_ready": True,
            },
            "nodes": final_nodes,
            "edges": final_edges,
        }
        final_path = _write_json(out / "final_graph.json", final_graph)

        report["graph_node_count"] = len(final_nodes)
        report["graph_edge_count"] = len(final_edges)
        report["connected_components"] = components
        report["largest_component_ratio"] = round(largest / float(n_nodes), 4)
        report["elapsed_total_seconds"] = round(time.perf_counter() - t_total0, 3)
        report["final_graph_path"] = final_path
        _write_json(out / "competition_fast_roadnet_report.json", report)

        progress(100, "比赛快速路网完成")
        return CompetitionFastResult(
            ok=True,
            timed_out=False,
            output_dir=str(out),
            final_graph_path=final_path,
            report=report,
            nodes_original=upscaled["nodes"],
            edges_original=upscaled["edges"],
            warning="; ".join(warnings),
        )

    except CompetitionFastTimeout as exc:
        report["elapsed_total_seconds"] = round(time.perf_counter() - t_total0, 3)
        report["timed_out"] = True
        warnings.append(str(exc))
        report["warnings"] = warnings
        _write_json(out / "competition_fast_roadnet_report.json", report)
        return CompetitionFastResult(
            ok=False,
            timed_out=True,
            output_dir=str(out),
            report=report,
            warning=str(exc),
            error=str(exc),
        )
    except Exception as exc:
        report["elapsed_total_seconds"] = round(time.perf_counter() - t_total0, 3)
        report["error"] = str(exc)
        if "取消" in str(exc):
            report["cancelled"] = True
        try:
            _write_json(out / "competition_fast_roadnet_report.json", report)
        except Exception:
            pass
        return CompetitionFastResult(
            ok=False, output_dir=str(out), report=report, error=str(exc)
        )


# Qt Worker 见 roadnet/competition_fast_roadnet_worker.py
