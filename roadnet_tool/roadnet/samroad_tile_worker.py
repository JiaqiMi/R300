"""Background SAM-Road tile inference orchestrator.

The external single-image script is invoked once per valid tile.  The worker
never touches Qt widgets and only publishes progress/results through signals.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from roadnet.samroad_single_runner import (
    SAMRoadSingleRunResult, build_command, prepare_project_import_paths,
    prepare_runtime_env,
)
from roadnet.segmentation_worker import generate_tile_grid
from roadnet.valid_image import (
    analyze_valid_image_mask, save_valid_mask_outputs, valid_area_ratio,
)
from roadnet.large_image_project import ImageRegionReader


class SAMRoadTileCancelled(RuntimeError):
    pass


def _read_rgb(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"无法读取输入影像: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_gray(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取 SAM-Road tile mask: {path}")
    return image


def _write_image(path: str, image: np.ndarray):
    extension = Path(path).suffix or ".png"
    source = image
    if image.ndim == 3:
        source = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(extension, source)
    if not ok:
        raise IOError(f"无法编码图像: {path}")
    encoded.tofile(path)


class SAMRoadTileWorker(QObject):
    progress = Signal(int, int, int, str)
    log = Signal(str)
    finished = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal(str)

    def __init__(self, config, output_dir: str, *, tile_size: int = 1024,
                 overlap: int = 128, skip_black_tile: bool = True,
                 black_threshold: int = 10,
                 min_black_component_area: int = 4096,
                 valid_pixel_ratio_threshold: float = 0.1,
                 merge_method: str = "max", parent=None):
        super().__init__(parent)
        self.config = config
        self.output_dir = Path(output_dir).resolve()
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.skip_black_tile = bool(skip_black_tile)
        self.black_threshold = int(black_threshold)
        self.min_black_component_area = int(min_black_component_area)
        self.valid_pixel_ratio_threshold = float(valid_pixel_ratio_threshold)
        self.merge_method = str(merge_method).lower()
        self._cancel = threading.Event()
        self._process = None
        self._process_lock = threading.Lock()
        self._logs = []

    def cancel(self):
        self._cancel.set()
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass

    def _check_cancelled(self):
        if self._cancel.is_set():
            raise SAMRoadTileCancelled("用户取消 SAM-Road tile 推理")

    def _log(self, message):
        line = f"[SAM-Road Tile] {message}"
        self._logs.append(line)
        self.log.emit(line)

    def _run_process(self, command, env):
        self._check_cancelled()
        process = subprocess.Popen(
            command,
            cwd=str(self.config.project_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        with self._process_lock:
            self._process = process
        try:
            while True:
                self._check_cancelled()
                try:
                    stdout, stderr = process.communicate(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            with self._process_lock:
                self._process = None
        return process.returncode, stdout, stderr

    @Slot()
    def run(self):
        started = time.perf_counter()
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            reader = ImageRegionReader(str(self.config.input_image))
            width, height = reader.size
            preview = reader.read_preview(3000)
            valid_mask, valid_mask_report = analyze_valid_image_mask(
                preview, self.black_threshold,
                max(64, int(self.min_black_component_area * (preview.shape[1] / width) ** 2)),
            )
            valid_mask = cv2.resize(
                valid_mask, (width, height), interpolation=cv2.INTER_NEAREST
            )
            valid_mask_report["analysis_mode"] = "border_connected_preview_mapped_to_original"
            ratio = valid_area_ratio(valid_mask)
            self._log(f"image size = {width} x {height}")
            self._log(f"valid area ratio = {ratio:.4f}")
            self._log(f"tile_size = {self.tile_size}")
            self._log(f"overlap = {self.overlap}")
            save_valid_mask_outputs(
                self.output_dir, valid_mask, valid_mask_report
            )

            candidates = generate_tile_grid(width, height, self.tile_size, self.overlap)
            tiles = []
            for tile in candidates:
                x0, y0, x1, y1 = tile
                tile_valid = valid_mask[y0:y1, x0:x1]
                tile_ratio = float(np.count_nonzero(tile_valid)) / float(tile_valid.size)
                if self.skip_black_tile and tile_ratio < self.valid_pixel_ratio_threshold:
                    continue
                tiles.append((tile, tile_ratio))
            skipped = len(candidates) - len(tiles)
            self._log(f"total_tiles = {len(tiles)}")
            self._log(f"skipped black tiles = {skipped}")
            if not tiles:
                raise ValueError("没有有效 tile；请检查黑色阈值或输入影像")

            if self.merge_method not in ("max", "average"):
                raise ValueError("merge_method 必须为 max 或 average")
            merged = np.zeros((height, width), dtype=np.uint8)
            sum_mask = None
            weights = None
            if self.merge_method == "average":
                sum_mask = np.zeros((height, width), dtype=np.float32)
                weights = np.zeros((height, width), dtype=np.uint16)

            tiles_root = self.output_dir / "tiles"
            tiles_root.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(prepare_runtime_env(str(self.output_dir)))
            env["PYTHONUNBUFFERED"] = "1"
            project_path = prepare_project_import_paths(str(self.config.project_dir))
            env["PYTHONPATH"] = project_path + (
                os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
            )

            stdout_all, stderr_all, failed_tiles = [], [], []
            for index, (tile, tile_ratio) in enumerate(tiles, 1):
                self._check_cancelled()
                x0, y0, x1, y1 = tile
                self._log(f"processing tile {index} / {len(tiles)} rect={tile} valid={tile_ratio:.3f}")
                tile_rgb = reader.read_region(x0, y0, x1, y1)
                tile_valid = valid_mask[y0:y1, x0:x1]
                tile_rgb[tile_valid == 0] = 0
                tile_path = tiles_root / f"tile_{index:04d}.png"
                tile_output = tiles_root / f"tile_{index:04d}_output"
                tile_output.mkdir(parents=True, exist_ok=True)
                _write_image(str(tile_path), tile_rgb)

                original_input = self.config.input_image
                self.config.input_image = tile_path
                try:
                    command = build_command(self.config, str(tile_output.resolve()))
                finally:
                    self.config.input_image = original_input
                code, stdout, stderr = self._run_process(command, env)
                stdout_all.append(stdout)
                stderr_all.append(stderr)
                if code != 0:
                    failed_tiles.append({
                        "tile": index, "rect": list(tile), "return_code": code,
                        "error": stderr[-1000:] if stderr else "无错误输出",
                    })
                    self._log(f"tile {index}/{len(tiles)} failed; continuing")
                    self.progress.emit(
                        int(round(index * 100 / len(tiles))), index, len(tiles),
                        f"SAM-Road tile {index} / {len(tiles)} (failed)",
                    )
                    continue
                mask_path = tile_output / "road_mask.png"
                if not mask_path.is_file():
                    failed_tiles.append({
                        "tile": index, "rect": list(tile), "return_code": code,
                        "error": f"未生成 road_mask.png: {mask_path}",
                    })
                    self._log(f"tile {index}/{len(tiles)} missing mask; continuing")
                    self.progress.emit(
                        int(round(index * 100 / len(tiles))), index, len(tiles),
                        f"SAM-Road tile {index} / {len(tiles)} (missing mask)",
                    )
                    continue
                tile_mask = _read_gray(str(mask_path))
                if tile_mask.shape != (y1 - y0, x1 - x0):
                    tile_mask = cv2.resize(tile_mask, (x1 - x0, y1 - y0),
                                           interpolation=cv2.INTER_LINEAR)
                tile_mask[tile_valid == 0] = 0
                _write_image(str(tiles_root / f"tile_{index:04d}_mask.png"), tile_mask)
                if self.merge_method == "max":
                    target = merged[y0:y1, x0:x1]
                    np.maximum(target, tile_mask, out=target)
                else:
                    sum_mask[y0:y1, x0:x1] += tile_mask.astype(np.float32)
                    weights[y0:y1, x0:x1] += 1
                percent = int(round(index * 100 / len(tiles)))
                self.progress.emit(percent, index, len(tiles),
                                   f"SAM-Road tile {index} / {len(tiles)}")

            if len(failed_tiles) == len(tiles):
                raise RuntimeError("所有 SAM-RoadPlus tile 推理均失败")
            if self.merge_method == "average":
                nonzero = weights > 0
                merged[nonzero] = np.clip(
                    sum_mask[nonzero] / weights[nonzero], 0, 255
                ).astype(np.uint8)
            merged[valid_mask == 0] = 0
            _write_image(str(self.output_dir / "road_mask.png"), merged)
            _write_image(str(self.output_dir / "road_mask_raw.png"), merged)
            _write_image(str(self.output_dir / "global_road_mask.png"), merged)
            preview_scale = min(1.0, 3000.0 / max(width, height))
            merged_preview = cv2.resize(
                merged,
                (max(1, int(width * preview_scale)), max(1, int(height * preview_scale))),
                interpolation=cv2.INTER_NEAREST,
            )
            _write_image(str(self.output_dir / "global_road_mask_preview.png"), merged_preview)
            elapsed = time.perf_counter() - started
            report = {
                "mode": "tile",
                "image_width": width,
                "image_height": height,
                "tile_size": self.tile_size,
                "overlap": self.overlap,
                "candidate_tile_count": len(candidates),
                "tile_count": len(tiles),
                "skipped_black_tile_count": skipped,
                "valid_area_ratio": round(ratio, 6),
                "black_threshold": self.black_threshold,
                "min_black_component_area": self.min_black_component_area,
                "valid_pixel_ratio_threshold": self.valid_pixel_ratio_threshold,
                "merge_method": self.merge_method,
                "failed_tiles": failed_tiles,
                "failed_tile_count": len(failed_tiles),
                "elapsed_seconds": round(elapsed, 3),
            }
            (self.output_dir / "tile_infer_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (self.output_dir / "samroad_tile_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (self.output_dir / "metadata.json").write_text(
                json.dumps({"original_width": width, "original_height": height,
                            "node_count": 0, "edge_count": 0,
                            "inference_mode": "tile"}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (self.output_dir / "samroad_stdout.log").write_text(
                "\n".join(stdout_all), encoding="utf-8", errors="replace"
            )
            (self.output_dir / "samroad_stderr.log").write_text(
                "\n".join(stderr_all), encoding="utf-8", errors="replace"
            )
            (self.output_dir / "samroadplus_stdout.log").write_text(
                "\n".join(stdout_all), encoding="utf-8", errors="replace"
            )
            (self.output_dir / "samroadplus_stderr.log").write_text(
                "\n".join(stderr_all), encoding="utf-8", errors="replace"
            )
            self._log(f"elapsed = {elapsed:.3f}s")
            self._log("saved road_mask.png")
            (self.output_dir / "samroad_tile_log.txt").write_text(
                "\n".join(self._logs) + "\n", encoding="utf-8"
            )
            result = SAMRoadSingleRunResult.from_process_result(
                0, self.output_dir, "\n".join(stdout_all), "\n".join(stderr_all)
            )
            result.duration_seconds = elapsed
            result.success = True
            self.finished.emit(result)
        except SAMRoadTileCancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            try:
                (self.output_dir / "samroad_tile_error.log").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
            except OSError:
                pass
            self.failed.emit(str(exc), str(self.output_dir / "samroad_tile_error.log"))
