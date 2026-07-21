"""Lowres Formal Mask Qt Worker — 全部耗时工作在后台线程。

只生成正式 working_road_mask，不跑 skeleton / graph。
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal

from roadnet.lowres_formal_mask import (
    LowresFormalMaskConfig,
    LowresFormalMaskResult,
    LowresFormalMaskTimeout,
    generate_lowres_formal_mask,
)


def _thread_id() -> int:
    return int(threading.get_ident())


def _read_sample_colors(
    image_path: str,
    points: Sequence[Sequence[float]],
) -> np.ndarray:
    """在 worker 线程内读取样本色（不整图 convert）。"""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    values: List[np.ndarray] = []
    with Image.open(image_path) as image:
        width, height = image.size
        for point in points or []:
            x = max(0, min(width - 1, int(round(float(point[0])))))
            y = max(0, min(height - 1, int(round(float(point[1])))))
            crop = image.crop((x, y, x + 1, y + 1)).convert("RGB")
            values.append(np.asarray(crop, dtype=np.uint8)[0, 0])
    if not values:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.asarray(values, dtype=np.uint8).reshape((-1, 3))


class LowresFormalMaskWorker(QObject):
    progress_changed = Signal(int, str)
    preview_ready = Signal(str)
    mask_ready = Signal(str)
    error_occurred = Signal(str)
    finished = Signal(object)
    # 兼容别名
    progress = progress_changed
    failed = error_occurred

    def __init__(
        self,
        image_path: str,
        pos_points: Sequence[Sequence[float]],
        neg_points: Sequence[Sequence[float]],
        output_dir: str,
        config: Optional[LowresFormalMaskConfig] = None,
        *,
        roi_polygons=None,
        ignore_polygons=None,
        main_thread_id: int = 0,
        button_clicked_time: str = "",
    ):
        super().__init__()
        self._image_path = str(image_path)
        self._pos_points = [tuple(map(float, p[:2])) for p in (pos_points or [])]
        self._neg_points = [tuple(map(float, p[:2])) for p in (neg_points or [])]
        self._output_dir = str(output_dir)
        self._config = config or LowresFormalMaskConfig()
        self._roi_polygons = roi_polygons or []
        self._ignore_polygons = ignore_polygons or []
        self._main_thread_id = int(main_thread_id or 0)
        self._button_clicked_time = button_clicked_time or datetime.now().isoformat(
            timespec="seconds"
        )
        self.cancel_requested = False
        self._stage = "precheck"
        self._progress = 0
        self._t0 = 0.0
        self._log_path = ""
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def request_cancel(self):
        self.cancel_requested = True
        self._log("cancel_requested=true")

    def cancel(self):
        self.request_cancel()

    def _log(self, line: str):
        if not self._log_path:
            return
        try:
            with open(self._log_path, "a", encoding="utf-8") as stream:
                stream.write(
                    f"{datetime.now().isoformat(timespec='seconds')} | {line}\n"
                )
        except OSError:
            pass

    def _set_stage(self, stage: str, percent: int, message: str):
        with self._lock:
            self._stage = stage
            self._progress = int(percent)
        self.progress_changed.emit(int(percent), message)
        self._log(f"stage={stage} percent={percent} msg={message}")

    def _emit_heartbeat(self):
        with self._lock:
            payload = {
                "stage": self._stage,
                "elapsed_seconds": round(time.perf_counter() - self._t0, 2)
                if self._t0 else 0.0,
                "progress": self._progress,
            }
        self._log(
            f"heartbeat stage={payload['stage']} elapsed={payload['elapsed_seconds']}"
        )

    def _heartbeat_loop(self):
        while not self._heartbeat_stop.wait(1.0):
            try:
                self._emit_heartbeat()
            except Exception:
                break

    def _start_heartbeat(self):
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="LowresFormalMaskHeartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self):
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        self._heartbeat_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)

    def run(self):
        self._t0 = time.perf_counter()
        out = Path(self._output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._log_path = str(out / "lowres_formal_mask_runtime.log")
        self._start_heartbeat()

        self._log(f"button_clicked_time={self._button_clicked_time}")
        self._log(f"worker_started_time={datetime.now().isoformat(timespec='seconds')}")
        self._log(f"main_thread_id={self._main_thread_id}")
        self._log(f"worker_thread_id={_thread_id()}")
        self._log(f"max_side={self._config.max_side}")
        self._log(f"image_path={self._image_path}")

        try:
            self._set_stage("precheck", 2, "预检查…")
            if self.cancel_requested:
                raise RuntimeError("用户取消")
            if not self._image_path or not os.path.isfile(self._image_path):
                raise FileNotFoundError(f"影像不存在: {self._image_path}")
            if not self._pos_points:
                raise ValueError("正样本为空，请先添加道路正样本。")
            if self._config.use_negative_samples and not self._neg_points:
                raise ValueError("负样本为空，请先添加非道路负样本。")

            self._set_stage("precheck", 5, "读取样本颜色…")
            pos_rgb = _read_sample_colors(self._image_path, self._pos_points)
            neg_rgb = _read_sample_colors(self._image_path, self._neg_points)
            self._log(f"sample_colors pos={len(pos_rgb)} neg={len(neg_rgb)}")

            def progress_cb(percent: int, message: str):
                stage = "segmentation"
                if "工作" in message or "working" in message.lower():
                    stage = "build_work_image"
                elif "有效" in message:
                    stage = "valid_area"
                elif "清理" in message:
                    stage = "mask_clean"
                elif "放大" in message:
                    stage = "upscale"
                elif "preview" in message.lower() or "显示" in message:
                    stage = "preview"
                elif "完成" in message:
                    stage = "finished"
                self._set_stage(stage, percent, message)
                if self.cancel_requested:
                    raise RuntimeError("用户取消")

            result = generate_lowres_formal_mask(
                self._image_path,
                self._output_dir,
                max_side=self._config.max_side,
                positive_samples=pos_rgb,
                negative_samples=neg_rgb,
                roi_polygons=self._roi_polygons,
                ignore_polygons=self._ignore_polygons,
                config=self._config,
                progress_cb=progress_cb,
                cancelled_cb=lambda: self.cancel_requested,
            )

            if result.timed_out or not result.ok:
                self.error_occurred.emit(result.error or result.warning or "失败")
            else:
                if result.working_mask_preview_path:
                    self.preview_ready.emit(result.working_mask_preview_path)
                if result.working_mask_path:
                    self.mask_ready.emit(result.working_mask_path)
                self._set_stage("finished", 100, "完成")

            self._log(f"finished_time={datetime.now().isoformat(timespec='seconds')}")
            self._log(f"ok={result.ok} error={result.error}")
            self.finished.emit(result)

        except Exception as exc:
            tb = traceback.format_exc()
            self._log(f"exception={exc}")
            self._log(tb)
            try:
                partial = {
                    "ok": False,
                    "cancelled": bool(self.cancel_requested),
                    "error": str(exc),
                    "stage": self._stage,
                    "elapsed_seconds": round(time.perf_counter() - self._t0, 3),
                    "traceback": tb,
                }
                with open(
                    out / "lowres_formal_mask_report.json",
                    "w",
                    encoding="utf-8",
                ) as stream:
                    json.dump(partial, stream, ensure_ascii=False, indent=2)
            except Exception:
                pass
            self.error_occurred.emit(str(exc))
            self.finished.emit(
                LowresFormalMaskResult(
                    ok=False,
                    timed_out=isinstance(exc, LowresFormalMaskTimeout),
                    output_dir=str(out),
                    error=str(exc),
                    warning=str(exc),
                    report={"stage": self._stage, "cancelled": self.cancel_requested},
                )
            )
        finally:
            self._stop_heartbeat()
