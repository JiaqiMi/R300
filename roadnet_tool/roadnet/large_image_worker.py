"""Background workers for large-image project creation and mask processing."""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from roadnet.large_image_project import (
    ImageRegionReader, LargeImageProject, build_tile_rects, create_large_image_project,
    generate_tile_index, load_tile_index,
    regenerate_project_preview,
)


class LargeImageCancelled(RuntimeError):
    pass


class LargeImageRefused(RuntimeError):
    """主路修复因缺少 seed/ROI/task 约束被拒绝执行。"""
    pass


class LargeImageProjectWorker(QObject):
    progress = Signal(int, int, int, str)
    finished = Signal(object, object)
    failed = Signal(str, str)
    cancelled = Signal(str)

    def __init__(self, *, action: str, image_path: str = "", output_root: str = "",
                 project: Optional[LargeImageProject] = None,
                 tile_size: int = 2048, overlap: int = 256,
                 preview_max_side: int = 3000, black_threshold: int = 10,
                 black_ratio_threshold: float = 0.8, parent=None):
        super().__init__(parent)
        self.action = action
        self.image_path = image_path
        self.output_root = output_root
        self.project = project
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.preview_max_side = int(preview_max_side)
        self.black_threshold = int(black_threshold)
        self.black_ratio_threshold = float(black_ratio_threshold)
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def _check(self):
        if self._cancel.is_set():
            raise LargeImageCancelled("用户取消大图项目任务")

    @Slot()
    def run(self):
        try:
            self._check()
            if self.action == "create":
                self.progress.emit(5, 0, 1, "读取影像尺寸并生成预览…")
                project = create_large_image_project(
                    self.image_path, self.output_root,
                    tile_size=self.tile_size, tile_overlap=self.overlap,
                    preview_max_side=self.preview_max_side,
                )
                self._check()
                self.progress.emit(35, 0, 1, "生成 tile index…")
                index = generate_tile_index(
                    project,
                    black_threshold=self.black_threshold,
                    black_ratio_threshold=self.black_ratio_threshold,
                    progress=lambda current, total: self.progress.emit(
                        35 + int(64 * current / max(1, total)), current, total,
                        f"分析 tile {current} / {total}",
                    ),
                    cancelled=self._cancel.is_set,
                )
                self.progress.emit(100, index["tile_count"], index["tile_count"], "大图项目已创建")
                self.finished.emit(project, index)
            elif self.action == "index":
                if self.project is None:
                    raise ValueError("缺少 large image project")
                index = generate_tile_index(
                    self.project,
                    black_threshold=self.black_threshold,
                    black_ratio_threshold=self.black_ratio_threshold,
                    progress=lambda current, total: self.progress.emit(
                        int(100 * current / max(1, total)), current, total,
                        f"分析 tile {current} / {total}",
                    ),
                    cancelled=self._cancel.is_set,
                )
                self.finished.emit(self.project, index)
            elif self.action == "preview":
                if self.project is None:
                    raise ValueError("missing large image project")
                self.progress.emit(10, 0, 1, "Generating bounded large-image preview")
                regenerate_project_preview(self.project, self.preview_max_side)
                self._check()
                index = (
                    load_tile_index(self.project.tile_index_path)
                    if self.project.tile_index_path and os.path.isfile(self.project.tile_index_path)
                    else {}
                )
                self.progress.emit(100, 1, 1, "Large-image preview regenerated")
                self.finished.emit(self.project, index)
            else:
                raise ValueError(f"未知大图任务: {self.action}")
        except LargeImageCancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            project_dir = Path(self.project.project_dir) if self.project else Path(self.output_root or ".")
            error_path = project_dir / "reports" / "large_image_project_error.log"
            try:
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
                if self.project:
                    self.project.last_error_log = str(error_path)
                    self.project.save()
            except OSError:
                pass
            self.failed.emit(str(exc), str(error_path))


@dataclass
class LargeMaskPostprocessResult:
    output_dir: str
    mask_path: str
    preview_path: str
    report: dict


def _normalize_polygons(polygons: Optional[Sequence]) -> list[np.ndarray]:
    result = []
    for polygon in polygons or []:
        arr = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
        if len(arr) >= 3:
            result.append(arr)
    return result


def _bounds(polygon: np.ndarray):
    return int(polygon[:, 0].min()), int(polygon[:, 1].min()), int(polygon[:, 0].max()), int(polygon[:, 1].max())


def _intersects(rect, bounds):
    return not (rect[2] <= bounds[0] or bounds[2] < rect[0] or rect[3] <= bounds[1] or bounds[3] < rect[1])


def _tile_polygon_mask(polygons, rect):
    x0, y0, x1, y1 = rect
    result = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for polygon in polygons:
        if not _intersects(rect, _bounds(polygon)):
            continue
        local = polygon.copy()
        local[:, 0] -= x0
        local[:, 1] -= y0
        cv2.fillPoly(result, [local.reshape((-1, 1, 2))], 255)
    return result


def _fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    inverse = (mask == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    result = mask.copy()
    height, width = mask.shape
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
        if not touches_border and int(area) <= int(max_area):
            result[labels == label] = 255
    return result


def _process_tile(mask, config):
    current = (np.asarray(mask) > int(config.get("threshold", 0))).astype(np.uint8) * 255
    blur = int(config.get("blur", config.get("blur_kernel", 0)))
    if blur > 0:
        blur += 1 - blur % 2
        current = cv2.GaussianBlur(current, (blur, blur), 0)
        current = (current > 127).astype(np.uint8) * 255
    close = int(config.get("close_kernel_size", config.get("close_kernel", 0)))
    if close > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
        current = cv2.morphologyEx(current, cv2.MORPH_CLOSE, kernel)
    opening = int(config.get("open_kernel_size", config.get("open_kernel", 0)))
    if opening > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening, opening))
        current = cv2.morphologyEx(current, cv2.MORPH_OPEN, kernel)
    if bool(config.get("fill_small_holes", False) or config.get("fill_holes", False)):
        current = _fill_small_holes(current, int(config.get("max_hole_area", 500)))
    minimum = int(config.get("min_area", 0))
    if minimum > 1:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(current, connectivity=8)
        cleaned = np.zeros_like(current)
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            touches_tile_edge = (
                x == 0 or y == 0
                or x + w >= current.shape[1]
                or y + h >= current.shape[0]
            )
            if touches_tile_edge or int(area) >= minimum:
                cleaned[labels == label] = 255
        current = cleaned
    return current


class LargeMaskPostprocessWorker(QObject):
    progress = Signal(int, int, int, str)
    finished = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal(str)

    def __init__(self, mask: np.ndarray, output_dir: str, *, config: dict,
                 tile_size: int = 2048, overlap: int = 256,
                 roi_polygons=None, ignore_polygons=None,
                 valid_image_mask: Optional[np.ndarray] = None, parent=None):
        super().__init__(parent)
        self.mask = mask
        self.output_dir = Path(output_dir).resolve()
        self.config = dict(config)
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.rois = _normalize_polygons(roi_polygons)
        self.ignores = _normalize_polygons(ignore_polygons)
        self.valid_image_mask = valid_image_mask
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def _check(self):
        if self._cancel.is_set():
            raise LargeImageCancelled("用户取消大图 Mask 后处理")

    @Slot()
    def run(self):
        started = time.perf_counter()
        try:
            source = np.asarray(self.mask)
            if source.ndim != 2 or source.dtype == object:
                raise TypeError(f"global mask 必须是单通道数组，实际 shape={source.shape}, dtype={source.dtype}")
            height, width = source.shape
            self.output_dir.mkdir(parents=True, exist_ok=True)
            before_path = self.output_dir / "global_mask_before.png"
            if not cv2.imwrite(str(before_path), source.astype(np.uint8)):
                raise IOError(f"无法保存 {before_path}")

            rects = build_tile_rects(width, height, self.tile_size, self.overlap)
            if self.rois:
                rects = [rect for rect in rects if any(_intersects(rect, _bounds(poly)) for poly in self.rois)]
            if not rects:
                raise ValueError("ROI 未覆盖任何可处理 tile")

            working_path = self.output_dir / "global_mask_after.working.uint8"
            output = np.memmap(working_path, mode="w+", dtype=np.uint8, shape=(height, width))
            output[:] = source
            for index, rect in enumerate(rects, 1):
                self._check()
                x0, y0, x1, y1 = rect
                tile = source[y0:y1, x0:x1]
                processed = _process_tile(tile, self.config)
                if self.valid_image_mask is not None:
                    processed[np.asarray(self.valid_image_mask)[y0:y1, x0:x1] == 0] = 0
                margin = max(0, self.overlap // 2)
                lx0 = 0 if x0 == 0 else min(margin, processed.shape[1])
                ly0 = 0 if y0 == 0 else min(margin, processed.shape[0])
                lx1 = processed.shape[1] if x1 == width else max(lx0, processed.shape[1] - margin)
                ly1 = processed.shape[0] if y1 == height else max(ly0, processed.shape[0] - margin)
                gx0, gy0 = x0 + lx0, y0 + ly0
                gx1, gy1 = x0 + lx1, y0 + ly1
                if self.rois:
                    roi = _tile_polygon_mask(self.rois, rect)
                    current = output[gy0:gy1, gx0:gx1]
                    local_roi = roi[ly0:ly1, lx0:lx1]
                    local_processed = processed[ly0:ly1, lx0:lx1]
                    current[local_roi > 0] = local_processed[local_roi > 0]
                else:
                    output[gy0:gy1, gx0:gx1] = processed[ly0:ly1, lx0:lx1]
                if self.ignores:
                    ignore = _tile_polygon_mask(self.ignores, rect)
                    output[y0:y1, x0:x1][ignore > 0] = 0
                self.progress.emit(
                    int(round(index * 100 / len(rects))), index, len(rects),
                    f"大图 Mask 后处理 tile {index} / {len(rects)}",
                )
            output.flush()
            self._check()
            after_path = self.output_dir / "global_mask_after.png"
            if not cv2.imwrite(str(after_path), output):
                raise IOError(f"无法保存 {after_path}")

            preview_max = 3000
            scale = min(1.0, preview_max / max(width, height))
            preview_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            before_preview = cv2.resize(source, preview_size, interpolation=cv2.INTER_NEAREST)
            after_preview = cv2.resize(output, preview_size, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(self.output_dir / "global_mask_preview_before.png"), before_preview)
            preview_path = self.output_dir / "global_mask_preview_after.png"
            cv2.imwrite(str(preview_path), after_preview)
            elapsed = time.perf_counter() - started
            report = {
                "image_width": width, "image_height": height,
                "coordinate_system": "image_pixel",
                "tile_size": self.tile_size, "overlap": self.overlap,
                "tile_count": len(rects), "roi_used": bool(self.rois),
                "ignore_used": bool(self.ignores), "fill_holes": bool(self.config.get("fill_holes", False)),
                "max_hole_area": int(self.config.get("max_hole_area", 500)),
                "seam_strategy": "overlap_half_core_crop",
                "elapsed_seconds": round(elapsed, 3),
                "before_nonzero": int(np.count_nonzero(source)),
                "after_nonzero": int(np.count_nonzero(output)),
            }
            report_path = self.output_dir / "mask_postprocess_large_report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            self.finished.emit(LargeMaskPostprocessResult(
                output_dir=str(self.output_dir), mask_path=str(after_path),
                preview_path=str(preview_path), report=report,
            ))
        except LargeImageCancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            error_path = self.output_dir / "error.log"
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
            except OSError:
                pass
            self.failed.emit(str(exc), str(error_path))


@dataclass
class LargeSegmentationResult:
    raw_mask: np.ndarray
    processed_mask: np.ndarray
    valid_image_mask: np.ndarray
    valid_mask_report: dict
    report: dict
    output_dir: str
    preview_mask: np.ndarray
    tile_status: list = field(default_factory=list)
    tile_status_report_path: str = ""
    tile_status_overlay_path: str = ""
    failed_roi_tile_count: int = 0


# tile 状态可视化配色（BGR，供 cv2 使用）
_TILE_STATUS_COLORS = {
    "success": (0, 200, 0),      # 绿色：成功
    "cache": (0, 220, 220),      # 黄色：缓存复用
    "skipped": (140, 140, 140),  # 灰色：跳过
    "failed": (0, 0, 235),       # 红色：失败
    "pending": (235, 120, 0),    # 蓝色：待处理
}


def _tile_status_label(record: dict) -> str:
    """根据 tile 记录返回状态标签，用于配色。"""
    if record.get("failed"):
        return "failed"
    if record.get("cache_hit"):
        return "cache"
    if record.get("success"):
        return "success"
    if record.get("skipped_black"):
        return "skipped"
    return "pending"


def render_tile_status_overlay(width: int, height: int, tile_records: list,
                               max_side: int = 3000) -> np.ndarray:
    """渲染 tile 状态 overlay（BGR）。

    绿色=成功 黄色=缓存 灰色=跳过 红色=失败 蓝色=待处理。
    """
    scale = min(1.0, float(max_side) / max(1, max(width, height)))
    ow = max(1, int(width * scale))
    oh = max(1, int(height * scale))
    base = np.zeros((oh, ow, 3), dtype=np.uint8)
    fill = base.copy()
    for record in tile_records:
        color = _TILE_STATUS_COLORS[_tile_status_label(record)]
        p0 = (int(record["x0"] * scale), int(record["y0"] * scale))
        p1 = (max(p0[0] + 1, int(record["x1"] * scale) - 1),
              max(p0[1] + 1, int(record["y1"] * scale) - 1))
        cv2.rectangle(fill, p0, p1, color, thickness=-1)
        cv2.rectangle(base, p0, p1, color, thickness=1)
    overlay = cv2.addWeighted(fill, 0.45, base, 1.0, 0.0)
    return overlay


class LargeImageSegmentationWorker(QObject):
    """Tile segmentation that never keeps the original full RGB image."""

    progress = Signal(int, int, int, str)
    finished = Signal(object)
    failed = Signal(str, str, str)
    cancelled = Signal(str)

    def __init__(self, image_path: str, positive_samples_rgb: np.ndarray,
                 negative_samples_rgb: np.ndarray, config: dict, output_dir: str,
                 *, roi_polygons=None, ignore_polygons=None,
                 tile_size: int = 1024, overlap: int = 64,
                 skip_black_area: bool = True, black_threshold: int = 10,
                 valid_pixel_ratio_threshold: float = 0.1,
                 extraction_label: str = "large_image_tile",
                 roi_required: bool = False,
                 mask_type: str = "formal_opencv", parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.positive_samples_rgb = np.asarray(positive_samples_rgb, dtype=np.uint8)
        self.negative_samples_rgb = np.asarray(negative_samples_rgb, dtype=np.uint8)
        self.config = dict(config)
        self.output_dir = Path(output_dir).resolve()
        self.rois = _normalize_polygons(roi_polygons)
        self.ignores = _normalize_polygons(ignore_polygons)
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.skip_black_area = bool(skip_black_area)
        self.black_threshold = int(black_threshold)
        self.valid_pixel_ratio_threshold = float(valid_pixel_ratio_threshold)
        self.extraction_label = str(extraction_label)
        self.roi_required = bool(roi_required)
        self.mask_type = str(mask_type)
        self._cancel = threading.Event()
        self._logs = []

    def cancel(self):
        self._cancel.set()

    def _check(self):
        if self._cancel.is_set():
            raise LargeImageCancelled("用户取消大图分割")

    def _log(self, message):
        line = f"[Segmentation] {message}"
        self._logs.append(line)
        print(line)

    @Slot()
    def run(self):
        started = time.perf_counter()
        stage = "validate_input"
        try:
            if not len(self.positive_samples_rgb):
                raise ValueError("正样本为空")
            if self.config.get("require_negative_samples", False) and not len(self.negative_samples_rgb):
                raise ValueError("负样本为空")
            reader = ImageRegionReader(self.image_path)
            width, height = reader.size
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._log(f"image size = {width} x {height}")
            self._log(f"tile_size = {self.tile_size}")
            self._log(f"overlap = {self.overlap}")

            # Valid-area analysis is run on a bounded preview, then mapped to
            # original pixels.  The analyzer only removes large black regions
            # connected to the four image borders.
            from roadnet.valid_image import analyze_valid_image_mask
            preview = reader.read_preview(3000)
            preview_valid, valid_report = analyze_valid_image_mask(
                preview, self.black_threshold, max(64, int(4096 * (preview.shape[1] / width) ** 2))
            )
            valid = cv2.resize(preview_valid, (width, height), interpolation=cv2.INTER_NEAREST)
            valid_report["analysis_mode"] = "border_connected_preview_mapped_to_original"
            valid_report["original_image_width"] = width
            valid_report["original_image_height"] = height

            stage = "select_tiles"
            all_rects = build_tile_rects(width, height, self.tile_size, self.overlap)
            # ROI 模式必须有 ROI，绝不退化为全图。
            if self.roi_required and not self.rois:
                raise ValueError("ROI 正式提取需要至少一个 ROI 区域，已阻止全图退化。")
            # 处理范围：ROI 模式只取与 ROI 相交的 tile；全图模式取全部。
            if self.rois:
                scope_rects = [rect for rect in all_rects
                               if any(_intersects(rect, _bounds(poly)) for poly in self.rois)]
            else:
                scope_rects = list(all_rects)
            if not scope_rects:
                raise ValueError("ROI/有效区域没有可处理 tile")

            # 建立每个 tile 的状态记录（含黑边跳过判定）。
            tile_records = []
            process_items = []  # (record_index, rect)
            for i, rect in enumerate(scope_rects):
                x0, y0, x1, y1 = rect
                area = float((x1 - x0) * (y1 - y0))
                valid_ratio = (
                    float(np.count_nonzero(valid[y0:y1, x0:x1])) / area if area else 0.0
                )
                skipped_black = bool(
                    self.skip_black_area and valid_ratio < self.valid_pixel_ratio_threshold
                )
                record = {
                    "tile_id": f"tile_{i:06d}",
                    "x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1),
                    "intersects_roi": bool(self.rois),
                    "skipped_black": skipped_black,
                    "cache_hit": False,
                    "processed": False,
                    "success": False,
                    "failed": False,
                    "mask_nonzero_ratio": 0.0,
                    "error_message": "",
                    "output_mask_path": "",
                }
                tile_records.append(record)
                if not skipped_black:
                    process_items.append((i, rect))

            skipped_black_count = sum(1 for r in tile_records if r["skipped_black"])
            if not process_items:
                raise ValueError("ROI/有效区域内所有 tile 均为黑边，无可处理 tile")
            self._log(f"scope tiles = {len(tile_records)}")
            self._log(f"process tiles = {len(process_items)}")
            self._log(f"skipped black tiles = {skipped_black_count}")

            stage = "segment_tiles"
            raw_path = self.output_dir / "road_mask_raw.working.uint8"
            raw = np.memmap(raw_path, mode="w+", dtype=np.uint8, shape=(height, width))
            raw[:] = 0
            from roadnet.opencv_road_segmenter import segment_road_by_samples
            failed_tiles = []
            total_proc = len(process_items)
            for order, (rec_idx, rect) in enumerate(process_items, 1):
                self._check()
                x0, y0, x1, y1 = rect
                record = tile_records[rec_idx]
                record["processed"] = True
                area = float((x1 - x0) * (y1 - y0))
                self._log(f"processing tile {order} / {total_proc} ({record['tile_id']})")
                try:
                    tile = reader.read_region(x0, y0, x1, y1)
                    tile_mask = segment_road_by_samples(
                        tile, self.positive_samples_rgb,
                        self.negative_samples_rgb, self.config,
                    )
                    tile_mask = np.asarray(tile_mask, dtype=np.uint8)
                    if tile_mask.shape != (y1 - y0, x1 - x0):
                        raise ValueError(f"输出尺寸错误: {tile_mask.shape}")
                    tile_mask[valid[y0:y1, x0:x1] == 0] = 0
                    if self.rois:
                        roi = _tile_polygon_mask(self.rois, rect)
                        tile_mask[roi == 0] = 0
                    target = raw[y0:y1, x0:x1]
                    np.maximum(target, tile_mask, out=target)
                    nonzero_ratio = (
                        float(np.count_nonzero(tile_mask)) / area if area else 0.0
                    )
                    record["success"] = True
                    record["mask_nonzero_ratio"] = round(nonzero_ratio, 6)
                except Exception as tile_error:
                    record["failed"] = True
                    record["error_message"] = str(tile_error)
                    failed_tiles.append({
                        "tile_id": record["tile_id"], "index": order,
                        "rect": list(rect), "error": str(tile_error),
                    })
                    self._log(f"tile {record['tile_id']} failed, continuing: {tile_error}")
                self.progress.emit(
                    int(round(order * 100 / total_proc)), order, total_proc,
                    f"OpenCV 正式提取 tile {order} / {total_proc}",
                )
            if len(failed_tiles) == total_proc:
                raise RuntimeError("所有 tile 分割均失败")

            stage = "merge_and_save"
            for _, rect in process_items:
                if self.ignores:
                    ignore = _tile_polygon_mask(self.ignores, rect)
                    x0, y0, x1, y1 = rect
                    raw[y0:y1, x0:x1][ignore > 0] = 0
            raw[valid == 0] = 0
            raw.flush()
            processed = raw
            road_raw_png = self.output_dir / "road_mask_raw.png"
            road_processed_png = self.output_dir / "road_mask_processed.png"
            global_mask_png = self.output_dir / "global_road_mask.png"
            for path in (road_raw_png, road_processed_png, global_mask_png):
                if not cv2.imwrite(str(path), processed):
                    raise IOError(f"无法保存 {path}")
            valid_path = self.output_dir / "valid_image_mask.png"
            cv2.imwrite(str(valid_path), valid)
            (self.output_dir / "valid_mask_report.json").write_text(
                json.dumps(valid_report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            scale = min(1.0, 3000.0 / max(width, height))
            preview_mask = cv2.resize(
                processed, (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_NEAREST,
            )
            preview_path = self.output_dir / "global_road_mask_preview.png"
            cv2.imwrite(str(preview_path), preview_mask)

            # ── tile 状态诊断：报告 + 可视化 overlay ──
            failed_roi_tile_count = sum(
                1 for r in tile_records if r["failed"] and r["intersects_roi"]
            )
            success_tile_count = sum(1 for r in tile_records if r["success"])
            status_report = {
                "image_width": width, "image_height": height,
                "coordinate_system": "original_image_pixel",
                "extraction_label": self.extraction_label,
                "mask_type": self.mask_type,
                "tile_size": self.tile_size, "overlap": self.overlap,
                "roi_used": bool(self.rois),
                "scope_tile_count": len(tile_records),
                "processed_tile_count": total_proc,
                "success_tile_count": success_tile_count,
                "failed_tile_count": len(failed_tiles),
                "failed_roi_tile_count": failed_roi_tile_count,
                "skipped_black_tile_count": skipped_black_count,
                "tiles": tile_records,
            }
            status_report_path = self.output_dir / "tile_status_report.json"
            status_report_path.write_text(
                json.dumps(status_report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            overlay_img = render_tile_status_overlay(width, height, tile_records)
            overlay_path = self.output_dir / "tile_status_overlay.png"
            cv2.imwrite(str(overlay_path), overlay_img)

            elapsed = time.perf_counter() - started
            report = {
                "image_width": width, "image_height": height,
                "coordinate_system": "original_image_pixel",
                "mode": self.extraction_label,
                "mask_type": self.mask_type,
                "formal_ready": True,
                "preview_only": False,
                "tile_size": self.tile_size, "overlap": self.overlap,
                "tile_count": total_proc,
                "scope_tile_count": len(tile_records),
                "success_tile_count": success_tile_count,
                "failed_tiles": failed_tiles,
                "failed_tile_count": len(failed_tiles),
                "failed_roi_tile_count": failed_roi_tile_count,
                "skipped_black_tile_count": skipped_black_count,
                "positive_sample_count": len(self.positive_samples_rgb),
                "negative_sample_count": len(self.negative_samples_rgb),
                "roi_used": bool(self.rois), "ignore_used": bool(self.ignores),
                "tile_status_report_path": str(status_report_path),
                "tile_status_overlay_path": str(overlay_path),
                "elapsed_seconds": round(elapsed, 3),
            }
            (self.output_dir / "segmentation_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (self.output_dir / "formal_extraction_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._log(f"elapsed = {elapsed:.3f}s")
            self._log("saved global_road_mask.png")
            (self.output_dir / "segmentation_log.txt").write_text("\n".join(self._logs) + "\n", encoding="utf-8")
            self.finished.emit(LargeSegmentationResult(
                raw_mask=raw, processed_mask=processed,
                valid_image_mask=valid, valid_mask_report=valid_report,
                report=report, output_dir=str(self.output_dir),
                preview_mask=preview_mask,
                tile_status=tile_records,
                tile_status_report_path=str(status_report_path),
                tile_status_overlay_path=str(overlay_path),
                failed_roi_tile_count=failed_roi_tile_count,
            ))
        except LargeImageCancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            error_path = self.output_dir / "segmentation_error.log"
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                error_path.write_text(
                    f"stage={stage}\nerror={exc}\n\n{traceback.format_exc()}", encoding="utf-8"
                )
            except OSError:
                pass
            self.failed.emit(stage, str(exc), str(error_path))


@dataclass
class MainRoadRefineResult:
    output_dir: str
    mask_path: str            # main_road_mask.png (full-size)
    preview_path: str         # main_road_mask_preview.png
    corridor_path: str        # main_road_corridor_mask.png (full-size)
    report: dict


class MainRoadRefineWorker(QObject):
    """大图主路优先修复 worker（种子 / ROI / 任务点约束下的半自动修复）。

    仅用于大图模式。在 preview 尺寸上生成 corridor、做连通域筛选、骨架 graph
    评分、受约束端点桥接，再映射回 original image size（仅 resize + dilate + Ignore）。
    无 seed/ROI/task 约束时拒绝执行（不做全图自由修复）。
    """

    progress = Signal(int, int, int, str)
    finished = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal(str)

    def __init__(self, mask: np.ndarray, output_dir: str, *,
                 config: Optional[dict] = None,
                 image_path: str = "",
                 roi_polygons=None, ignore_polygons=None, task_points=None,
                 seed_strokes=None, view_rect=None,
                 source_is_preview: bool = False,
                 parent=None):
        super().__init__(parent)
        self.mask = np.asarray(mask)
        self.output_dir = Path(output_dir).resolve()
        self.config = dict(config or {})
        self.image_path = str(image_path or "")
        self.roi_polygons = [list(p) for p in (roi_polygons or [])]
        self.ignore_polygons = [list(p) for p in (ignore_polygons or [])]
        self.task_points = [list(p) for p in (task_points or [])]
        self.seed_strokes = [list(s) for s in (seed_strokes or [])]
        self.view_rect = list(view_rect) if view_rect else None
        self.source_is_preview = bool(source_is_preview)
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def _check(self):
        if self._cancel.is_set():
            raise LargeImageCancelled("用户取消大图主路优先修复")

    def _has_constraint(self) -> bool:
        return bool(self.seed_strokes or self.roi_polygons or self.task_points
                    or self.view_rect)

    @staticmethod
    def _scale_polys(polys, scale):
        out = []
        for poly in polys or []:
            arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2) * float(scale)
            out.append(arr.tolist())
        return out

    @staticmethod
    def _scale_points(points, scale):
        return [[float(p[0]) * scale, float(p[1]) * scale] for p in (points or [])]

    def _save(self, name, arr):
        path = self.output_dir / name
        if not cv2.imwrite(str(path), np.asarray(arr, dtype=np.uint8)):
            raise IOError(f"无法保存 {path}")
        return path

    @Slot()
    def run(self):
        from roadnet.main_road_postprocess import (
            refine_main_road_mask, DEFAULT_MAIN_ROAD_CONFIG,
        )
        started = time.perf_counter()
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            cfg = dict(DEFAULT_MAIN_ROAD_CONFIG)
            cfg.update(self.config)

            # 无约束直接拒绝（第四/十三条），绝不退化为全图自由修复。
            if cfg.get("require_seed_or_roi_or_task", True) and not self._has_constraint():
                raise LargeImageRefused(
                    "请先画主路种子线或设置 ROI，否则全图修复会误连大量噪声。"
                )

            source = self.mask
            if source.ndim != 2:
                source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            source = (source > 0).astype(np.uint8) * 255
            orig_h, orig_w = source.shape

            self.progress.emit(5, 0, 100, "准备 preview 工作图…")
            self._check()

            # ── preview 工作分辨率 ──
            preview_max = int(cfg.get("preview_max_side", 2000))
            if bool(cfg.get("use_preview_level", True)) and not self.source_is_preview:
                scale = min(1.0, preview_max / max(orig_w, orig_h))
            else:
                scale = 1.0
            work_w = max(1, int(round(orig_w * scale)))
            work_h = max(1, int(round(orig_h * scale)))
            work_mask = (cv2.resize(source, (work_w, work_h), interpolation=cv2.INTER_NEAREST)
                         if scale < 1.0 else source)

            # ── 可选颜色支持图（preview 尺寸）──
            image_bgr = None
            if self.image_path and os.path.exists(self.image_path):
                try:
                    reader = ImageRegionReader(self.image_path)
                    prev_rgb = reader.read_preview(max(work_w, work_h))
                    prev_bgr = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2BGR)
                    image_bgr = cv2.resize(prev_bgr, (work_w, work_h),
                                           interpolation=cv2.INTER_AREA)
                except Exception:
                    image_bgr = None

            roi_prev = self._scale_polys(self.roi_polygons, scale)
            ignore_prev = self._scale_polys(self.ignore_polygons, scale)
            task_prev = self._scale_points(self.task_points, scale)
            seed_prev = self._scale_polys(self.seed_strokes, scale)
            view_prev = ([v * scale for v in self.view_rect] if self.view_rect else None)

            self.progress.emit(20, 0, 100, "生成 corridor / 连通域筛选 / 桥接…")
            self._check()

            stages: dict = {}
            refined_preview, report = refine_main_road_mask(
                work_mask, image_bgr=image_bgr,
                roi_polygons=roi_prev or None,
                ignore_polygons=ignore_prev or None,
                task_points=task_prev or None,
                seed_strokes=seed_prev or None,
                view_rect=view_prev,
                config=cfg, stages_out=stages,
            )
            report["input_mask_shape"] = [int(orig_h), int(orig_w)]
            report["preview_shape"] = [int(work_h), int(work_w)]
            report["preview_scale"] = round(float(scale), 6)
            report["source_is_preview"] = self.source_is_preview

            if report.get("refused"):
                raise LargeImageRefused(
                    "; ".join(report.get("warnings", []))
                    or "未提供主路约束，已拒绝执行全图主路修复。"
                )

            self.progress.emit(70, 0, 100, "保存中间结果 / overlay…")
            self._check()

            # ── 第十二条：保存 preview 级别中间产物 ──
            for name, key in (
                ("raw_mask_preview.png", "raw_mask_preview"),
                ("main_road_corridor_mask.png", "main_road_corridor_mask"),
                ("seed_connected_components.png", "seed_connected_components"),
                ("component_filtered_mask.png", "component_filtered_mask"),
                ("skeleton_raw_preview.png", "skeleton_raw_preview"),
                ("pruned_skeleton_preview.png", "pruned_skeleton_preview"),
                ("main_road_mask_preview.png", "main_road_mask_preview"),
            ):
                if key in stages:
                    self._save(name, stages[key])
            preview_path = self.output_dir / "main_road_mask_preview.png"

            self._make_edge_score_overlay(stages, self.output_dir / "graph_edge_score_overlay.png")
            self._make_bridge_overlay(stages, "all",
                                      self.output_dir / "bridge_candidates_overlay.png")
            self._make_bridge_overlay(stages, "accepted",
                                      self.output_dir / "accepted_bridges_overlay.png")

            # ── 映射回 original size：仅 resize + dilate + Ignore（第十一/十五条）──
            self.progress.emit(85, 0, 100, "映射回原图尺寸…")
            self._check()
            if scale < 1.0:
                refined_full = cv2.resize(refined_preview, (orig_w, orig_h),
                                          interpolation=cv2.INTER_NEAREST)
                k = max(3, int(round(1.0 / max(scale, 1e-6))) | 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                refined_full = cv2.morphologyEx(refined_full, cv2.MORPH_CLOSE, kernel)
                corridor_full = cv2.resize(
                    stages.get("main_road_corridor_mask", refined_preview),
                    (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                # 全图仍限制在 corridor 内。
                refined_full = cv2.bitwise_and(refined_full, corridor_full)
            else:
                refined_full = refined_preview
                corridor_full = stages.get("main_road_corridor_mask", refined_preview)

            # 全图坐标下强制应用 Ignore。
            if self.ignore_polygons:
                ig = _normalize_polygons(self.ignore_polygons)
                ig_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                for poly in ig:
                    cv2.fillPoly(ig_mask, [poly.reshape(-1, 1, 2)], 255)
                refined_full[ig_mask > 0] = 0

            mask_path = self._save("main_road_mask.png", refined_full)
            corridor_path = self._save("main_road_corridor_mask_full.png", corridor_full)

            report["input_mask_path"] = str(mask_path)
            report["output_dir"] = str(self.output_dir)
            report["refined_mask_path"] = str(mask_path)
            report["refined_preview_path"] = str(preview_path)
            report["corridor_path"] = str(corridor_path)
            report["mask_type"] = "formal_opencv_mainroad_refined"
            report["formal_ready"] = True
            report["preview_only"] = False
            report["coordinate_system"] = "original_image_pixel"
            report["total_elapsed_seconds"] = round(time.perf_counter() - started, 3)
            # bridge_candidates 是列表，单独存文件，report json 内只留计数与路径。
            bridge_candidates = report.pop("bridge_candidates", [])
            (self.output_dir / "bridge_candidates.json").write_text(
                json.dumps(bridge_candidates, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            report["bridge_candidates_path"] = str(self.output_dir / "bridge_candidates.json")

            (self.output_dir / "main_road_refine_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.progress.emit(100, 0, 100, "主路优先修复完成")
            self.finished.emit(MainRoadRefineResult(
                output_dir=str(self.output_dir), mask_path=str(mask_path),
                preview_path=str(preview_path), corridor_path=str(corridor_path),
                report=report,
            ))
        except LargeImageCancelled as exc:
            self.cancelled.emit(str(exc))
        except LargeImageRefused as exc:
            self.failed.emit(str(exc), "")
        except Exception as exc:
            error_path = self.output_dir / "main_road_refine_error.log"
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
            except OSError:
                pass
            self.failed.emit(str(exc), str(error_path))

    def _make_edge_score_overlay(self, stages: dict, path: Path):
        raw = stages.get("raw_mask_preview")
        if raw is None:
            return
        base = (cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR) * 0.30).astype(np.uint8)
        corridor = stages.get("main_road_corridor_mask")
        if corridor is not None:
            base[corridor > 0] = np.clip(base[corridor > 0] + (25, 25, 0), 0, 255)
        kept = stages.get("edge_kept_skeleton")
        skel_raw = stages.get("skeleton_raw_preview")
        if skel_raw is not None:
            base[skel_raw > 0] = (0, 0, 200)          # 原始骨架（含被删边）→ 红
        if kept is not None:
            base[kept > 0] = (0, 220, 0)              # 保留 edge → 绿
        cv2.imwrite(str(path), base)

    def _make_bridge_overlay(self, stages: dict, which: str, path: Path):
        raw = stages.get("raw_mask_preview")
        if raw is None:
            return
        base = (cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR) * 0.30).astype(np.uint8)
        skel = stages.get("pruned_skeleton_preview")
        if skel is not None:
            base[skel > 0] = (180, 180, 180)
        colors = {"accepted": (0, 220, 0), "rejected": (0, 0, 220), "pending": (0, 220, 220)}
        for rec in stages.get("bridge_candidates", []) or []:
            status = rec.get("status", "rejected")
            if which != "all" and status != which:
                continue
            (x1, y1), (x2, y2) = rec["p1"], rec["p2"]
            cv2.line(base, (int(x1), int(y1)), (int(x2), int(y2)),
                     colors.get(status, (0, 0, 220)), 2)
        cv2.imwrite(str(path), base)
