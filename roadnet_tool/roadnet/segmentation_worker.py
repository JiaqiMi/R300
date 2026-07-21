"""Background, cancellable tile segmentation for RoadNet Studio.

The worker owns computation and file output only.  It never touches QWidget,
QGraphicsScene, QPixmap, or any other GUI object.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot


DEFAULT_TILE_SIZE = 1024
DEFAULT_OVERLAP = 64
LARGE_IMAGE_THRESHOLD = 4096


class SegmentationCancelled(RuntimeError):
    pass


class SegmentationStageError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def _axis_starts(length: int, tile_size: int, overlap: int) -> List[int]:
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    starts = list(range(0, max(1, length - tile_size + 1), step))
    last = length - tile_size
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def generate_tile_grid(
    width: int, height: int, tile_size: int, overlap: int
) -> List[Tuple[int, int, int, int]]:
    """Return unique covering tiles as ``(x0, y0, x1, y1)``."""
    if tile_size <= 0:
        raise ValueError("tile_size 必须大于 0")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap 必须满足 0 <= overlap < tile_size")
    return [
        (x, y, min(width, x + tile_size), min(height, y + tile_size))
        for y in _axis_starts(height, tile_size, overlap)
        for x in _axis_starts(width, tile_size, overlap)
    ]


def _polygon_bounds(polygon: np.ndarray) -> Tuple[int, int, int, int]:
    return (
        int(np.min(polygon[:, 0])), int(np.min(polygon[:, 1])),
        int(np.max(polygon[:, 0])), int(np.max(polygon[:, 1])),
    )


def _rect_intersects(a, b) -> bool:
    return not (a[2] < b[0] or b[2] <= a[0] or a[3] < b[1] or b[3] <= a[1])


def _normalize_polygons(polygons: Optional[Sequence]) -> List[np.ndarray]:
    result = []
    for polygon in polygons or []:
        arr = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
        if len(arr) >= 3:
            result.append(arr)
    return result


def _scale_polygons(polygons: List[np.ndarray], scale: float) -> List[np.ndarray]:
    if scale == 1.0:
        return [polygon.copy() for polygon in polygons]
    return [np.rint(polygon.astype(np.float64) * scale).astype(np.int32)
            for polygon in polygons]


def _region_mask_for_tile(
    polygons: List[np.ndarray], tile: Tuple[int, int, int, int]
) -> Optional[np.ndarray]:
    if not polygons:
        return None
    x0, y0, x1, y1 = tile
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    tile_rect = (x0, y0, x1, y1)
    for polygon in polygons:
        if not _rect_intersects(tile_rect, _polygon_bounds(polygon)):
            continue
        local = polygon.copy()
        local[:, 0] -= x0
        local[:, 1] -= y0
        cv2.fillPoly(mask, [local.reshape(-1, 1, 2)], 255)
    return mask


@dataclass
class SegmentationResult:
    raw_mask: np.ndarray
    processed_mask: np.ndarray
    valid_image_mask: np.ndarray
    valid_mask_report: Dict
    report: Dict
    output_dir: str


class SegmentationWorker(QObject):
    progress = Signal(int, int, int, str)  # percent, current, total, message
    finished = Signal(object)
    failed = Signal(str, str, str)         # stage, message, error_log_path
    cancelled = Signal(str)

    def __init__(
        self,
        image_rgb: np.ndarray,
        positive_samples_rgb: np.ndarray,
        negative_samples_rgb: np.ndarray,
        config: Dict,
        output_dir: str,
        roi_polygons: Optional[Sequence] = None,
        ignore_polygons: Optional[Sequence] = None,
        tile_size: int = DEFAULT_TILE_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        preview_scale: float = 1.0,
        skip_black_area: bool = True,
        black_threshold: int = 10,
        min_black_component_area: int = 4096,
        valid_pixel_ratio_threshold: float = 0.1,
        parent=None,
    ):
        super().__init__(parent)
        self.image_rgb = image_rgb  # shared read-only reference; intentionally no full copy
        self.positive_samples_rgb = np.asarray(positive_samples_rgb, dtype=np.uint8)
        self.negative_samples_rgb = np.asarray(negative_samples_rgb, dtype=np.uint8)
        self.config = dict(config)
        self.output_dir = output_dir
        self.roi_polygons = _normalize_polygons(roi_polygons)
        self.ignore_polygons = _normalize_polygons(ignore_polygons)
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.preview_scale = float(preview_scale)
        self.skip_black_area = bool(skip_black_area)
        self.black_threshold = int(black_threshold)
        self.min_black_component_area = int(min_black_component_area)
        self.valid_pixel_ratio_threshold = float(valid_pixel_ratio_threshold)
        self._cancel_event = threading.Event()
        self._log_lines: List[str] = []

    def cancel(self):
        """Thread-safe cancellation request; safe to call from the GUI thread."""
        self._cancel_event.set()

    def _check_cancelled(self):
        if self._cancel_event.is_set():
            raise SegmentationCancelled("用户取消")

    def _log(self, message: str):
        line = f"[Segmentation] {message}"
        self._log_lines.append(line)
        print(line)

    def _write_text_log(self, filename: str, extra: str = "") -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        lines = list(self._log_lines)
        if extra:
            lines.append(extra)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        return path

    @Slot()
    def run(self):
        started = time.perf_counter()
        stage = "validate_input"
        try:
            image = self.image_rgb
            if image is None or not isinstance(image, np.ndarray) or image.size == 0:
                raise SegmentationStageError(stage, "图像为空")
            if image.ndim != 3 or image.shape[2] < 3:
                raise SegmentationStageError(stage, f"图像格式无效: shape={image.shape}")
            if len(self.positive_samples_rgb) == 0:
                raise SegmentationStageError(stage, "正样本为空")
            if self.config.get("require_negative_samples", False) and len(self.negative_samples_rgb) == 0:
                raise SegmentationStageError(stage, "负样本为空")
            self._check_cancelled()

            height, width = image.shape[:2]
            self._log(f"image size = {width} x {height}")
            from roadnet.valid_image import (
                analyze_valid_image_mask, apply_valid_image_mask,
                save_valid_mask_outputs, valid_area_ratio,
            )
            valid_image_mask, valid_mask_report = analyze_valid_image_mask(
                image,
                self.black_threshold,
                self.min_black_component_area,
            )
            valid_ratio = valid_area_ratio(valid_image_mask)
            self._check_cancelled()
            self._log(f"valid area ratio = {valid_ratio:.4f}")
            self._log(f"tile_size = {self.tile_size}")
            self._log(f"overlap = {self.overlap}")

            work_scale = min(1.0, max(0.05, self.preview_scale))
            if work_scale < 1.0:
                work_width = max(1, int(round(width * work_scale)))
                work_height = max(1, int(round(height * work_scale)))
                work_image = cv2.resize(image, (work_width, work_height),
                                        interpolation=cv2.INTER_AREA)
                work_valid = cv2.resize(valid_image_mask, (work_width, work_height),
                                        interpolation=cv2.INTER_NEAREST)
                work_rois = _scale_polygons(self.roi_polygons, work_scale)
                work_ignores = _scale_polygons(self.ignore_polygons, work_scale)
                self._log(f"preview_scale = {work_scale:.2f} ({work_width} x {work_height})")
            else:
                work_width, work_height = width, height
                work_image = image
                work_valid = valid_image_mask
                work_rois = self.roi_polygons
                work_ignores = self.ignore_polygons

            stage = "build_tile_grid"
            all_tiles = generate_tile_grid(work_width, work_height, self.tile_size, self.overlap)
            if work_rois:
                roi_bounds = [_polygon_bounds(p) for p in work_rois]
                tiles = [
                    tile for tile in all_tiles
                    if any(_rect_intersects(tile, bounds) for bounds in roi_bounds)
                ]
            else:
                tiles = all_tiles
            candidate_tile_count = len(tiles)
            if self.skip_black_area:
                tiles = [
                    tile for tile in tiles
                    if float(np.count_nonzero(work_valid[tile[1]:tile[3], tile[0]:tile[2]]))
                    / float((tile[3] - tile[1]) * (tile[2] - tile[0]))
                    >= self.valid_pixel_ratio_threshold
                ]
            skipped_black_tiles = candidate_tile_count - len(tiles)
            self._log(f"total_tiles = {len(tiles)}")
            self._log(f"skipped black tiles = {skipped_black_tiles}")
            if not tiles:
                raise SegmentationStageError(stage, "ROI/有效影像区域未覆盖任何可处理 tile")

            stage = "allocate_mask"
            try:
                work_raw_mask = np.zeros((work_height, work_width), dtype=np.uint8)
            except MemoryError as exc:
                raise SegmentationStageError(stage, "内存不足，无法创建完整 uint8 mask") from exc

            from roadnet.color_segment import segment_road

            stage = "tile_segmentation"
            total = len(tiles)
            for index, tile in enumerate(tiles, 1):
                self._check_cancelled()
                x0, y0, x1, y1 = tile
                self._log(f"processing tile {index} / {total}")
                try:
                    # Basic slicing is a view; segment_road owns only tile-sized arrays.
                    tile_rgb = work_image[y0:y1, x0:x1]
                    tile_mask = segment_road(
                        tile_rgb,
                        self.positive_samples_rgb,
                        self.negative_samples_rgb,
                        self.config,
                    )
                except MemoryError as exc:
                    raise SegmentationStageError(
                        stage, f"内存不足（tile {index}/{total}, rect={tile}）"
                    ) from exc
                except Exception as exc:
                    raise SegmentationStageError(
                        stage, f"tile 处理失败 {index}/{total}, rect={tile}: {exc}"
                    ) from exc

                if tile_mask is None or tile_mask.shape != (y1 - y0, x1 - x0):
                    shape = None if tile_mask is None else tile_mask.shape
                    raise SegmentationStageError(
                        stage, f"tile {index}/{total} 输出尺寸错误: {shape}"
                    )
                tile_mask = np.asarray(tile_mask, dtype=np.uint8)
                valid_tile = work_valid[y0:y1, x0:x1]
                tile_mask[valid_tile == 0] = 0
                roi_tile = _region_mask_for_tile(work_rois, tile)
                if roi_tile is not None:
                    tile_mask = cv2.bitwise_and(tile_mask, roi_tile)

                # Max merge is stable for binary masks and smooths overlap by
                # retaining positive evidence from either neighboring tile.
                target = work_raw_mask[y0:y1, x0:x1]
                np.maximum(target, tile_mask, out=target)
                percent = int(round(index * 100 / total))
                self.progress.emit(
                    percent, index, total, f"正在处理 tile {index} / {total}"
                )
                self._check_cancelled()

            stage = "restore_full_resolution"
            if work_scale < 1.0:
                raw_mask = cv2.resize(work_raw_mask, (width, height),
                                      interpolation=cv2.INTER_NEAREST)
            else:
                raw_mask = work_raw_mask
            raw_mask = apply_valid_image_mask(raw_mask, valid_image_mask)

            stage = "apply_ignore"
            processed_mask = raw_mask.copy()
            if self.ignore_polygons:
                for tile in generate_tile_grid(width, height, self.tile_size, self.overlap):
                    ignore_tile = _region_mask_for_tile(self.ignore_polygons, tile)
                    if ignore_tile is None or not np.any(ignore_tile):
                        continue
                    x0, y0, x1, y1 = tile
                    view = processed_mask[y0:y1, x0:x1]
                    view[ignore_tile > 0] = 0

            self._check_cancelled()
            stage = "save_outputs"
            os.makedirs(self.output_dir, exist_ok=True)
            raw_path = os.path.join(self.output_dir, "road_mask_raw.png")
            processed_path = os.path.join(self.output_dir, "road_mask_processed.png")
            if not cv2.imwrite(raw_path, raw_mask):
                raise SegmentationStageError(stage, f"无法保存 {raw_path}")
            if not cv2.imwrite(processed_path, processed_mask):
                raise SegmentationStageError(stage, f"无法保存 {processed_path}")
            valid_mask_report["removed_road_pixels_estimate"] = int(
                np.count_nonzero((raw_mask > 0) & (valid_image_mask == 0))
            )
            save_valid_mask_outputs(
                self.output_dir, valid_image_mask, valid_mask_report
            )

            elapsed = time.perf_counter() - started
            report = {
                "image_width": int(width),
                "image_height": int(height),
                "tile_size": self.tile_size,
                "overlap": self.overlap,
                "tile_count": total,
                "candidate_tile_count": candidate_tile_count,
                "skipped_black_tile_count": skipped_black_tiles,
                "valid_area_ratio": round(valid_ratio, 6),
                "black_threshold": self.black_threshold,
                "min_black_component_area": self.min_black_component_area,
                "valid_pixel_ratio_threshold": self.valid_pixel_ratio_threshold,
                "skip_black_area": self.skip_black_area,
                "preview_scale": work_scale,
                "positive_sample_count": int(len(self.positive_samples_rgb)),
                "negative_sample_count": int(len(self.negative_samples_rgb)),
                "elapsed_seconds": round(elapsed, 3),
                "roi_used": bool(self.roi_polygons),
                "ignore_used": bool(self.ignore_polygons),
            }
            with open(os.path.join(self.output_dir, "segmentation_report.json"),
                      "w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
            self._log(f"elapsed = {elapsed:.3f}s")
            self._log("saved road_mask.png")
            self._write_text_log("segmentation_log.txt")
            self.progress.emit(100, total, total, f"分割完成，用时 {elapsed:.1f} 秒")
            self.finished.emit(SegmentationResult(
                raw_mask=raw_mask,
                processed_mask=processed_mask,
                valid_image_mask=valid_image_mask,
                valid_mask_report=valid_mask_report,
                report=report,
                output_dir=self.output_dir,
            ))
        except SegmentationCancelled as exc:
            elapsed = time.perf_counter() - started
            self._log(f"cancelled after {elapsed:.3f}s")
            self._write_text_log("segmentation_log.txt", "status = cancelled")
            self.cancelled.emit(str(exc))
        except Exception as exc:
            if isinstance(exc, SegmentationStageError):
                stage = exc.stage
            details = traceback.format_exc()
            self._log(f"failed at {stage}: {exc}")
            error_path = self._write_text_log(
                "segmentation_error.log",
                f"stage = {stage}\nerror = {exc}\n\n{details}",
            )
            self.failed.emit(stage, str(exc), error_path)
