"""Competition Fast Roadnet Qt Worker — 全部耗时工作在后台线程。

主线程禁止：imread 大图、resize、分割、骨架、graph、大图像素图转换、wait。
Worker 禁止直接操作 UI，只通过 Signal 回报进度。
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import QObject, Signal

from roadnet.competition_fast_roadnet import (
    CompetitionFastConfig,
    CompetitionFastResult,
    CompetitionFastTimeout,
    run_competition_fast_roadnet,
)


def _thread_id() -> int:
    return int(threading.get_ident())


def _read_sample_colors(
    image_path: str,
    points: Sequence[Sequence[float]],
) -> np.ndarray:
    """在 worker 线程内一次性打开影像读取样本色（不整图 convert）。"""
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


class CompetitionFastRoadnetWorker(QObject):
    """后台 Worker：load → work image → mask → skeleton → graph → upscale → save。"""

    progress_changed = Signal(int, str)          # percent, message
    stage_changed = Signal(str)                  # stage name
    heartbeat = Signal(dict)                     # stage/elapsed/progress/...
    preview_ready = Signal(str)                  # preview path
    graph_ready = Signal(str)                    # final_graph path
    error_occurred = Signal(str)                 # error message
    finished = Signal(object)                    # CompetitionFastResult
    # 兼容旧连接名
    progress = progress_changed
    failed = error_occurred

    def __init__(
        self,
        image_path: str,
        pos_points: Sequence[Sequence[float]],
        neg_points: Sequence[Sequence[float]],
        output_dir: str,
        config: Optional[CompetitionFastConfig] = None,
        *,
        main_thread_id: int = 0,
        button_clicked_time: str = "",
    ):
        super().__init__()
        self._image_path = str(image_path)
        self._pos_points = [tuple(map(float, p[:2])) for p in (pos_points or [])]
        self._neg_points = [tuple(map(float, p[:2])) for p in (neg_points or [])]
        self._output_dir = str(output_dir)
        self._config = config or CompetitionFastConfig()
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
        self.stage_changed.emit(stage)
        self.progress_changed.emit(int(percent), message)
        self._log(f"stage={stage} percent={percent} msg={message}")

    def _emit_heartbeat(self):
        with self._lock:
            payload = {
                "stage": self._stage,
                "elapsed_seconds": round(time.perf_counter() - self._t0, 2)
                if self._t0 else 0.0,
                "progress": self._progress,
                "current_step": self._stage,
                "message": f"{self._stage} {self._progress}%",
                "cancel_requested": bool(self.cancel_requested),
                "worker_thread_id": _thread_id(),
            }
        self.heartbeat.emit(payload)
        self._log(
            f"heartbeat stage={payload['stage']} elapsed={payload['elapsed_seconds']} "
            f"progress={payload['progress']}"
        )

    def _heartbeat_loop(self):
        """独立线程心跳：不依赖 worker 事件循环（run 阻塞时 QTimer 不会触发）。"""
        while not self._heartbeat_stop.wait(1.0):
            try:
                self._emit_heartbeat()
            except Exception:
                break

    def _start_heartbeat(self):
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="CompetitionFastHeartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self):
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        self._heartbeat_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)

    def _check_cancel(self):
        if self.cancel_requested:
            raise RuntimeError("用户取消快速路网生成")

    def run(self):
        """在 QThread 中执行；禁止任何 UI 调用。"""
        self._t0 = time.perf_counter()
        out = Path(self._output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._log_path = str(out / "competition_fast_runtime.log")
        self._start_heartbeat()

        worker_started = datetime.now().isoformat(timespec="seconds")
        self._log(f"button_clicked_time={self._button_clicked_time}")
        self._log(f"worker_started_time={worker_started}")
        self._log(f"main_thread_id={self._main_thread_id}")
        self._log(f"worker_thread_id={_thread_id()}")
        self._log(f"preview_max_side={self._config.competition_preview_max_side}")
        self._log(f"image_path={self._image_path}")
        self._log(f"pos_points={len(self._pos_points)} neg_points={len(self._neg_points)}")

        result: Optional[CompetitionFastResult] = None
        try:
            self._set_stage("precheck", 2, "预检查…")
            self._check_cancel()
            if not self._image_path or not os.path.isfile(self._image_path):
                raise FileNotFoundError(f"影像不存在: {self._image_path}")
            if not self._pos_points:
                raise ValueError("正样本为空，请先添加道路正样本。")
            if self._config.use_negative_samples and not self._neg_points:
                raise ValueError("负样本为空，请先添加非道路负样本。")

            self._set_stage("precheck", 5, "读取样本颜色…")
            self._check_cancel()
            pos_rgb = _read_sample_colors(self._image_path, self._pos_points)
            neg_rgb = _read_sample_colors(self._image_path, self._neg_points)
            self._log(f"sample_colors pos={len(pos_rgb)} neg={len(neg_rgb)}")

            def progress_cb(percent: int, message: str):
                # map pipeline messages to stages
                stage = self._stage
                low = (message or "").lower()
                if "working" in low or "工作" in message:
                    stage = "build_work_image"
                elif "有效" in message or "valid" in low:
                    stage = "valid_area"
                elif "道路提取" in message or "分割" in message:
                    stage = "fast_segmentation"
                elif "后处理" in message or "clean" in low:
                    stage = "mask_clean"
                elif "骨架" in message or "skeleton" in low:
                    stage = "skeleton"
                elif "路网" in message or "graph" in low:
                    stage = "graph_build"
                elif "映射" in message or "upscale" in low:
                    stage = "graph_upscale"
                elif "完成" in message:
                    stage = "finished"
                self._set_stage(stage, percent, message)
                self._check_cancel()

            self._set_stage("build_work_image", 8, "生成低分辨率 working image…")
            result = run_competition_fast_roadnet(
                self._image_path,
                pos_rgb,
                neg_rgb,
                self._output_dir,
                config=self._config,
                progress_cb=progress_cb,
                cancelled_cb=lambda: self.cancel_requested,
            )

            if result.timed_out:
                self.error_occurred.emit(result.error or result.warning or "超时")
            elif not result.ok:
                self.error_occurred.emit(result.error or result.warning or "失败")
            else:
                # lightweight preview paths for UI
                work_preview = os.path.join(
                    result.output_dir, "competition_work_image.png"
                )
                if os.path.isfile(work_preview):
                    self.preview_ready.emit(work_preview)
                if result.final_graph_path:
                    self.graph_ready.emit(result.final_graph_path)
                self._set_stage("finished", 100, "完成")

            self._log(f"finished_time={datetime.now().isoformat(timespec='seconds')}")
            self._log(f"ok={result.ok} timed_out={result.timed_out} error={result.error}")
            self.finished.emit(result)

        except Exception as exc:
            tb = traceback.format_exc()
            self._log(f"exception={exc}")
            self._log(tb)
            # partial report
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
                    out / "competition_fast_roadnet_report.json",
                    "w",
                    encoding="utf-8",
                ) as stream:
                    json.dump(partial, stream, ensure_ascii=False, indent=2)
            except Exception:
                pass
            self.error_occurred.emit(str(exc))
            self.finished.emit(
                CompetitionFastResult(
                    ok=False,
                    timed_out=isinstance(exc, CompetitionFastTimeout),
                    output_dir=str(out),
                    error=str(exc),
                    warning=str(exc),
                    report={"stage": self._stage, "cancelled": self.cancel_requested},
                )
            )
        finally:
            self._stop_heartbeat()
