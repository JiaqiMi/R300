"""轻量级快速预览分割 Worker。

只处理 preview.png，不读取原始大图。

要求：
- 不调用 cv2.imread(original_image)
- 不做 full-size connectedComponents / findContours
- 不做 full-size QPixmap overlay
- 不做 fill_holes / skeleton / graph / 任务点
- 不在大图上生成 global_road_mask.png
- 1~5 秒内返回
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict

import cv2
import numpy as np

from PySide6.QtCore import QObject, Signal

# ============================================================================
# 默认参数
# ============================================================================

DEFAULT_PREVIEW_MAX_SIDE = 1500
DEFAULT_COMPETITION_MAX_SIDE = 1200
DEFAULT_BLUR_KERNEL = 3
DEFAULT_OPEN_KERNEL = 3
DEFAULT_CLOSE_KERNEL = 3
DEFAULT_MIN_AREA_PREVIEW = 50


# ============================================================================
# 缓存工具
# ============================================================================

def _compute_cache_key(
    preview_path: str,
    preview_max_side: int,
    config: dict,
) -> str:
    """计算快速预览分割缓存键。

    基于 preview 文件 mtime + 参数生成唯一哈希。
    """
    try:
        mtime = os.path.getmtime(preview_path)
    except OSError:
        mtime = 0.0

    param_str = json.dumps(
        {
            "preview_path": str(preview_path),
            "preview_mtime": mtime,
            "preview_max_side": preview_max_side,
            "mode": config.get("mode", "combined"),
            "combine_method": config.get("combine_method", "and"),
            "h_margin": config.get("h_margin", 6),
            "s_margin": config.get("s_margin", 25),
            "v_margin": config.get("v_margin", 30),
            "lab_margin": config.get("lab_margin", 12),
            "use_negative": config.get("use_negative_samples", True),
            "blur_kernel": config.get("preview_blur_kernel", DEFAULT_BLUR_KERNEL),
            "open_kernel": config.get("preview_open_kernel", DEFAULT_OPEN_KERNEL),
            "close_kernel": config.get("preview_close_kernel", DEFAULT_CLOSE_KERNEL),
            "sample_radius": config.get("sample_radius", 3),
        },
        sort_keys=True,
    )
    return hashlib.sha256(param_str.encode("utf-8")).hexdigest()


def _load_cache(output_dir: str, cache_key: str) -> Optional[np.ndarray]:
    """尝试从缓存目录加载已缓存的 preview_mask。"""
    cache_dir = Path(output_dir) / ".preview_cache"
    mask_path = cache_dir / f"{cache_key}_mask.png"
    if mask_path.is_file():
        cached = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if cached is not None:
            return cached
    return None


def _load_cache_overlay(output_dir: str, cache_key: str) -> Optional[np.ndarray]:
    """尝试从缓存目录加载已缓存的 overlay。"""
    cache_dir = Path(output_dir) / ".preview_cache"
    overlay_path = cache_dir / f"{cache_key}_overlay.png"
    if overlay_path.is_file():
        cached = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        if cached is not None:
            return cv2.cvtColor(cached, cv2.COLOR_BGR2RGB)
    return None


def _save_cache(output_dir: str, cache_key: str, mask: np.ndarray, overlay: Optional[np.ndarray] = None):
    """保存 preview_mask 和 overlay 到缓存目录。清理旧缓存（保留最近 5 个）。"""
    cache_dir = Path(output_dir) / ".preview_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    mask_path = cache_dir / f"{cache_key}_mask.png"
    cv2.imwrite(str(mask_path), mask)

    if overlay is not None:
        overlay_path = cache_dir / f"{cache_key}_overlay.png"
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(overlay_path), overlay_bgr)

    # 清理旧缓存：保留最近 5 个
    mask_files = sorted(cache_dir.glob("*_mask.png"), key=lambda f: f.stat().st_mtime)
    overlay_files = sorted(cache_dir.glob("*_overlay.png"), key=lambda f: f.stat().st_mtime)
    for old in mask_files[:-5]:
        try:
            old.unlink()
        except OSError:
            pass
    for old in overlay_files[:-5]:
        try:
            old.unlink()
        except OSError:
            pass


# ============================================================================
# 轻量级快速预览分割算法
# ============================================================================

def generate_preview_segmentation(
    preview_rgb: np.ndarray,
    pos_samples_rgb: np.ndarray,
    neg_samples_rgb: np.ndarray,
    config: dict,
    preview_max_side: int = DEFAULT_PREVIEW_MAX_SIDE,
) -> np.ndarray:
    """在 preview 图像上执行轻量级颜色分割。

    严格限制：
    - 只处理 preview (已是缩略图)
    - 如果 preview 仍过大，再次缩放到 preview_max_side
    - 不做 tile 分块（足够小，一次处理）
    - 不做 connectedComponents / findContours / fill_holes
    - 不做 full-size overlay

    Args:
        preview_rgb:      preview RGB 图像 (H, W, 3)
        pos_samples_rgb:  正样本 RGB (N, 3)
        neg_samples_rgb:  负样本 RGB (M, 3)
        config:           分割配置
        preview_max_side: 如果 preview 任一边超过此值，先缩放

    Returns:
        二值 mask (H, W), dtype uint8, 255=道路, 0=非道路
    """
    h, w = preview_rgb.shape[:2]
    scale = 1.0
    work_image = preview_rgb

    # Step 1: 如果 preview 仍过大，再次缩放
    max_dim = max(w, h)
    if max_dim > preview_max_side:
        scale = preview_max_side / max_dim
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        work_image = cv2.resize(preview_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Step 2 + 3: 颜色空间分割 + 轻量形态学。
    # 复用公共模块 segment_road_by_samples，保证“快速预览”与“OpenCV 正式提取”
    # 使用完全一致的核心分割算法。preview_* 形态学键映射到公共形态学键。
    from roadnet.opencv_road_segmenter import segment_road_by_samples

    seg_cfg = dict(config)
    seg_cfg["blur_kernel"] = config.get("preview_blur_kernel", DEFAULT_BLUR_KERNEL)
    seg_cfg["open_kernel"] = config.get("preview_open_kernel", DEFAULT_OPEN_KERNEL)
    seg_cfg["close_kernel"] = config.get("preview_close_kernel", DEFAULT_CLOSE_KERNEL)
    # 预览不做 min_area / fill_holes（保持轻量、快速）。
    seg_cfg["min_area"] = 0
    seg_cfg["fill_holes"] = False

    work_mask = segment_road_by_samples(
        work_image, pos_samples_rgb, neg_samples_rgb, seg_cfg
    )

    # Step 4: 如果做了缩放，恢复到 preview 原始尺寸
    if scale < 1.0:
        work_mask = cv2.resize(work_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return work_mask


# ============================================================================
# 分割结果数据类
# ============================================================================

@dataclass
class PreviewSegmentationResult:
    preview_mask: np.ndarray                     # 预览 mask (preview 尺寸)
    overlay_rgb: Optional[np.ndarray] = None     # 叠加图 (preview 尺寸)
    report: dict = None                          # 性能报告
    output_dir: str = ""                         # 输出目录
    cache_used: bool = False
    preview_only: bool = True


# ============================================================================
# PreviewSegmentationWorker
# ============================================================================

class PreviewSegmentationWorker(QObject):
    """快速预览分割后台 Worker。

    只处理 preview.png，不访问原始大图。

    用法:
        thread = QThread()
        worker = PreviewSegmentationWorker(...)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(on_finished)
        thread.start()
    """

    progress = Signal(int, str)           # percent, message
    finished = Signal(object)             # emits PreviewSegmentationResult
    failed = Signal(str, str, str)        # stage, message, error_log_path
    cancelled_signal = Signal(str)        # message

    def __init__(
        self,
        preview_path: str,
        pos_samples_rgb: np.ndarray,
        neg_samples_rgb: np.ndarray,
        output_dir: str,
        config: dict,
        *,
        preview_max_side: int = DEFAULT_PREVIEW_MAX_SIDE,
        competition_fast_mode: bool = False,
        save_debug_files: bool = False,
    ):
        super().__init__()
        self._preview_path = str(preview_path)
        self._pos_rgb = np.asarray(pos_samples_rgb, dtype=np.uint8)
        self._neg_rgb = np.asarray(neg_samples_rgb, dtype=np.uint8)
        self._output_dir = str(output_dir)
        self._config = dict(config)
        self._preview_max_side = int(preview_max_side)
        self._competition_mode = bool(competition_fast_mode)
        self._save_debug = bool(save_debug_files)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        """主执行体（运行在 QThread 中）。"""
        t0 = time.perf_counter()
        steps = []
        cache_used = False

        try:
            # Step 1: 计算缓存 key
            steps.append("compute_cache_key")
            if self._cancelled:
                self.cancelled_signal.emit("用户取消")
                return

            cache_key = _compute_cache_key(
                self._preview_path, self._preview_max_side, self._config,
            )
            self.progress.emit(5, "检查缓存…")

            # Step 2: 尝试从缓存加载
            steps.append("check_cache")
            if self._cancelled:
                self.cancelled_signal.emit("用户取消")
                return

            output_dir = self._output_dir or os.path.join(
                os.path.dirname(self._preview_path), "mask_preview",
            )
            os.makedirs(output_dir, exist_ok=True)

            cached_mask = _load_cache(output_dir, cache_key)
            if cached_mask is not None:
                cache_used = True
                steps.append("cache_hit")
                self.progress.emit(20, "使用缓存的快速预览分割结果。")

                # 尝试加载缓存的 overlay；若无则加载 preview 重建
                cached_overlay = _load_cache_overlay(output_dir, cache_key)
                if cached_overlay is None:
                    self.progress.emit(40, "加载预览图以重建叠加…")
                    preview_for_overlay = cv2.imread(self._preview_path, cv2.IMREAD_COLOR)
                    if preview_for_overlay is not None:
                        preview_for_overlay = cv2.cvtColor(preview_for_overlay, cv2.COLOR_BGR2RGB)
                        if preview_for_overlay.shape[:2] == cached_mask.shape[:2]:
                            cached_overlay = self._build_overlay(preview_for_overlay, cached_mask)
                            # 也保存 overlay 到磁盘和缓存
                            overlay_path = os.path.join(output_dir, "preview_seg_overlay.png")
                            if cached_overlay is not None:
                                cv2.imwrite(overlay_path, cv2.cvtColor(cached_overlay, cv2.COLOR_RGB2BGR))
                                _save_cache(output_dir, cache_key, cached_mask, cached_overlay)
                        else:
                            cached_overlay = None
                    else:
                        cached_overlay = None

                self.progress.emit(95, "完成。")

                # ★ 缓存命中也要做输出验证
                self._verify_outputs(output_dir,
                                     cached_overlay.shape[1] if cached_overlay is not None else cached_mask.shape[1],
                                     cached_overlay.shape[0] if cached_overlay is not None else cached_mask.shape[0])

                elapsed = time.perf_counter() - t0
                report = self._build_report(
                    preview_size=cached_mask.shape[:2][::-1],
                    elapsed=elapsed,
                    steps=steps,
                    cache_used=True,
                )
                self._write_report(output_dir, report)

                result = PreviewSegmentationResult(
                    preview_mask=cached_mask,
                    overlay_rgb=cached_overlay,
                    report=report,
                    output_dir=output_dir,
                    cache_used=True,
                    preview_only=True,
                )
                self.progress.emit(100, "预览完成（缓存）")
                self.finished.emit(result)
                return

            # Step 3: 加载 preview.png
            steps.append("load_preview")
            if self._cancelled:
                self.cancelled_signal.emit("用户取消")
                return
            self.progress.emit(10, "加载预览图…")

            preview_rgb = cv2.imread(self._preview_path, cv2.IMREAD_COLOR)
            if preview_rgb is None:
                raise RuntimeError(f"无法读取预览图: {self._preview_path}")
            preview_rgb = cv2.cvtColor(preview_rgb, cv2.COLOR_BGR2RGB)
            preview_h, preview_w = preview_rgb.shape[:2]

            # Step 4: 映射样本点到 preview 坐标（如果样本点是原始分辨率坐标）
            # 此时样本点 RGB 值已经是调用方从原始图像读出的值，直接使用
            steps.append("map_samples")

            # Step 5: 轻量分割
            steps.append("segment")
            if self._cancelled:
                self.cancelled_signal.emit("用户取消")
                return
            self.progress.emit(30, "执行轻量颜色分割…")

            actual_max_side = (
                DEFAULT_COMPETITION_MAX_SIDE if self._competition_mode
                else self._preview_max_side
            )
            mask = generate_preview_segmentation(
                preview_rgb,
                self._pos_rgb,
                self._neg_rgb,
                self._config,
                preview_max_side=actual_max_side,
            )

            if self._cancelled:
                self.cancelled_signal.emit("用户取消")
                return
            self.progress.emit(70, "保存预览 mask…")

            # Step 6: 保存 preview_mask.png
            steps.append("save_mask")
            mask_path = os.path.join(output_dir, "preview_mask.png")
            cv2.imwrite(mask_path, mask)

            # Step 7: 生成叠加图 —— 始终保存到磁盘（必须在缓存写入前完成）
            steps.append("build_overlay")
            self.progress.emit(85, "生成预览叠加…")
            overlay = self._build_overlay(preview_rgb, mask)

            # ★ 始终保存 overlay 到磁盘，不依赖 save_debug_files
            overlay_path = os.path.join(output_dir, "preview_seg_overlay.png")
            if overlay is not None:
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                cv2.imwrite(overlay_path, overlay_bgr)
                steps.append("save_overlay")

            # Step 8: 写入缓存（含 overlay，必须在 overlay 生成后调用）
            steps.append("save_cache")
            _save_cache(output_dir, cache_key, mask, overlay)

            if self._save_debug:
                steps.append("save_debug_files")
                self.progress.emit(90, "保存调试文件…")

            # Step 9: 输出验证
            steps.append("verify_outputs")
            self._verify_outputs(output_dir, preview_w, preview_h)

            # Step 10: 写报告
            steps.append("write_report")
            elapsed = time.perf_counter() - t0
            report = self._build_report(
                preview_size=(preview_w, preview_h),
                elapsed=elapsed,
                steps=steps,
                cache_used=False,
            )
            self._write_report(output_dir, report)

            result = PreviewSegmentationResult(
                preview_mask=mask,
                overlay_rgb=overlay,
                report=report,
                output_dir=output_dir,
                cache_used=False,
                preview_only=True,
            )
            self.progress.emit(100, "快速预览完成")
            self.finished.emit(result)

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            elapsed = time.perf_counter() - t0

            # 写错误日志
            error_log = ""
            try:
                error_dir = self._output_dir or os.path.join(
                    os.path.dirname(self._preview_path), "mask_preview",
                )
                os.makedirs(error_dir, exist_ok=True)
                error_path = os.path.join(error_dir, "preview_segmentation_error.log")
                with open(error_path, "w", encoding="utf-8") as f:
                    f.write(f"=== Preview Segmentation Error ===\n")
                    f.write(f"preview_path: {self._preview_path}\n")
                    f.write(f"elapsed_seconds: {elapsed:.3f}\n")
                    f.write(f"steps_completed: {steps}\n")
                    f.write(f"error: {exc}\n")
                    f.write(f"traceback:\n{tb}\n")
                error_log = error_path
            except Exception:
                pass

            stage = steps[-1] if steps else "unknown"
            self.failed.emit(stage, str(exc), error_log)

    def _build_overlay(
        self, preview_rgb: np.ndarray, mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        """生成道路 mask 在 preview 图上的半透明叠加图。preview 尺寸。"""
        try:
            if preview_rgb.ndim != 3:
                return None
            h, w = preview_rgb.shape[:2]
            if mask.shape[:2] != (h, w):
                return None
            overlay = preview_rgb.copy()
            road_mask_3ch = np.stack([mask] * 3, axis=-1)
            green = np.zeros_like(overlay)
            green[:, :, 1] = 255
            alpha = 0.4
            np.copyto(overlay, green, where=(road_mask_3ch > 0))
            blended = cv2.addWeighted(preview_rgb, 1 - alpha, overlay, alpha, 0)
            return blended
        except Exception:
            return None

    def _build_report(
        self,
        preview_size: tuple,
        elapsed: float,
        steps: list,
        cache_used: bool,
    ) -> dict:
        """构建性能报告。"""
        working_side = (
            DEFAULT_COMPETITION_MAX_SIDE if self._competition_mode
            else self._preview_max_side
        )
        report = {
            "preview_size": list(preview_size),
            "working_max_side": working_side,
            "competition_fast_mode": self._competition_mode,
            "elapsed_seconds": round(elapsed, 3),
            "cache_used": cache_used,
            "operation_steps": steps,
            "preview_only": True,
            "preview_path": self._preview_path,
            "output_dir": self._output_dir,
        }
        if elapsed > 5.0:
            report["warning"] = (
                f"快速预览耗时过长 ({elapsed:.1f}s > 5s)，"
                f"请降低 preview_max_side 或关闭 debug。"
            )
        return report

    def _write_report(self, output_dir: str, report: dict):
        """写性能报告 JSON。"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, "preview_segmentation_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
                f.write("\n")
        except Exception:
            pass

    def _verify_outputs(self, output_dir: str, preview_w: int, preview_h: int):
        """验证输出文件完整性并在日志中报告。"""
        mask_path = os.path.join(output_dir, "preview_mask.png")
        overlay_path = os.path.join(output_dir, "preview_seg_overlay.png")
        report_path = os.path.join(output_dir, "preview_segmentation_report.json")
        error_log_path = os.path.join(output_dir, "preview_segmentation_error.log")

        mask_exists = os.path.isfile(mask_path)
        overlay_exists = os.path.isfile(overlay_path)
        report_exists = os.path.isfile(report_path)
        error_log_exists = os.path.isfile(error_log_path)

        print(f"[PreviewSeg] output_dir = {output_dir}")
        print(f"[PreviewSeg] preview_mask exists = {mask_exists}")
        print(f"[PreviewSeg] preview_overlay exists = {overlay_exists}")
        print(f"[PreviewSeg] preview_segmentation_report.json exists = {report_exists}")
        print(f"[PreviewSeg] preview_segmentation_error.log exists = {error_log_exists}")
        print(f"[PreviewSeg] preview image size = {preview_w} x {preview_h}")

        if mask_exists:
            try:
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    mh, mw = mask_img.shape[:2]
                    print(f"[PreviewSeg] preview_mask size = {mw} x {mh}")
                    if (mw, mh) != (preview_w, preview_h):
                        print(f"[PreviewSeg] ⚠ WARNING: preview_mask size ({mw}x{mh}) "
                              f"!= expected ({preview_w}x{preview_h})")
                else:
                    print(f"[PreviewSeg] ⚠ preview_mask.png 存在但无法读取（cv2.imread 返回 None）")
            except Exception as e:
                print(f"[PreviewSeg] ⚠ 读取 preview_mask.png 失败: {e}")

        if overlay_exists:
            try:
                overlay_img = cv2.imread(overlay_path, cv2.IMREAD_COLOR)
                if overlay_img is not None:
                    oh, ow = overlay_img.shape[:2]
                    print(f"[PreviewSeg] overlay size = {ow} x {oh}")
                    if (ow, oh) != (preview_w, preview_h):
                        print(f"[PreviewSeg] ⚠ WARNING: overlay size ({ow}x{oh}) "
                              f"!= expected ({preview_w}x{preview_h})")
                else:
                    print(f"[PreviewSeg] ⚠ preview_seg_overlay.png 存在但无法读取")
            except Exception as e:
                print(f"[PreviewSeg] ⚠ 读取 preview_seg_overlay.png 失败: {e}")
