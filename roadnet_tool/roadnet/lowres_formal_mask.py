"""Low-res formal road mask — 大图低像素快速生成正式 working Road Mask。

流程：
  original image → lowres_work_image → OpenCV 分割 → lowres clean
  → INTER_NEAREST 放大 → working_road_mask.png

只生成正式 mask，不跑 skeleton / graph / 路径规划。
不影响小图流程。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from roadnet.opencv_road_segmenter import segment_road_by_samples
from roadnet.valid_image import analyze_valid_image_mask, apply_valid_image_mask


DEFAULT_LOWRES_MAX_SIDE = 2500
DEFAULT_OPEN_KERNEL = 3
DEFAULT_CLOSE_KERNEL = 5
DEFAULT_MIN_COMPONENT_AREA = 80
DEFAULT_MAX_TOTAL_SECONDS = 120.0


@dataclass
class LowresFormalMaskConfig:
    max_side: int = DEFAULT_LOWRES_MAX_SIDE
    open_kernel: int = DEFAULT_OPEN_KERNEL
    close_kernel: int = DEFAULT_CLOSE_KERNEL
    remove_small_components: bool = True
    min_component_area: int = DEFAULT_MIN_COMPONENT_AREA
    fill_holes: bool = False
    valid_area_only: bool = True
    black_threshold: int = 10
    max_total_seconds: float = DEFAULT_MAX_TOTAL_SECONDS
    # OpenCV 分割参数（与快速预览共用 segment_road_by_samples）
    h_margin: int = 6
    s_margin: int = 25
    v_margin: int = 30
    lab_margin: int = 12
    use_negative_samples: bool = True
    blur_kernel: int = 3
    mode: str = "combined"
    combine_method: str = "and"
    sample_radius: int = 3
    use_roi: bool = True
    use_ignore: bool = True


@dataclass
class LowresFormalMaskResult:
    ok: bool
    output_dir: str = ""
    working_mask_path: str = ""
    working_mask_preview_path: str = ""
    lowres_work_image_path: str = ""
    lowres_road_mask_path: str = ""
    report: Dict[str, Any] = field(default_factory=dict)
    working_mask: Optional[np.ndarray] = None
    warning: str = ""
    error: str = ""
    timed_out: bool = False


class LowresFormalMaskTimeout(RuntimeError):
    """低像素正式 Mask 生成超时。"""


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


def build_lowres_work_image(
    image_path: str,
    max_side: int = DEFAULT_LOWRES_MAX_SIDE,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """从原始大图生成低分辨率 working image（不加载全图 RGB）。"""
    from roadnet.large_image_project import ImageRegionReader

    reader = ImageRegionReader(image_path)
    work = reader.read_preview(int(max_side))
    oh, ow = int(reader.height), int(reader.width)
    wh, ww = int(work.shape[0]), int(work.shape[1])
    scale = {
        "original_width": ow,
        "original_height": oh,
        "lowres_width": ww,
        "lowres_height": wh,
        "work_width": ww,
        "work_height": wh,
        "scale_x": float(ow) / float(max(1, ww)),
        "scale_y": float(oh) / float(max(1, wh)),
        "max_side": int(max_side),
    }
    return work, scale


def _scale_polygons_to_lowres(
    polygons: Optional[Sequence[Sequence[Sequence[float]]]],
    scale_x: float,
    scale_y: float,
) -> List[np.ndarray]:
    """将 original-pixel 多边形缩放到 lowres 坐标。"""
    out: List[np.ndarray] = []
    sx = float(scale_x) if scale_x else 1.0
    sy = float(scale_y) if scale_y else 1.0
    for poly in polygons or []:
        pts = []
        for p in poly:
            if p is None or len(p) < 2:
                continue
            pts.append([float(p[0]) / sx, float(p[1]) / sy])
        if len(pts) >= 3:
            out.append(np.asarray(pts, dtype=np.float32))
    return out


def _apply_roi_ignore(
    mask: np.ndarray,
    roi_polys: List[np.ndarray],
    ignore_polys: List[np.ndarray],
) -> np.ndarray:
    out = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if roi_polys:
        roi_mask = np.zeros(out.shape, dtype=np.uint8)
        for poly in roi_polys:
            pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
            if len(pts) >= 3:
                cv2.fillPoly(roi_mask, [pts], 255)
        out = cv2.bitwise_and(out, roi_mask)
    if ignore_polys:
        ignore_mask = np.zeros(out.shape, dtype=np.uint8)
        for poly in ignore_polys:
            pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
            if len(pts) >= 3:
                cv2.fillPoly(ignore_mask, [pts], 255)
        out = cv2.bitwise_and(out, cv2.bitwise_not(ignore_mask))
    return out


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
        inv = cv2.bitwise_not(out)
        h, w = inv.shape[:2]
        flood = inv.copy()
        cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        out = cv2.bitwise_or(out, holes)
    return out


def upscale_mask_nearest(
    lowres_mask: np.ndarray,
    original_width: int,
    original_height: int,
) -> np.ndarray:
    """INTER_NEAREST 放大到原图尺寸，保持 uint8 0/255 二值。"""
    ow = max(1, int(original_width))
    oh = max(1, int(original_height))
    binary = (np.asarray(lowres_mask) > 0).astype(np.uint8) * 255
    up = cv2.resize(binary, (ow, oh), interpolation=cv2.INTER_NEAREST)
    return (up > 0).astype(np.uint8) * 255


def generate_lowres_formal_mask(
    image_path: str,
    output_dir: str,
    *,
    max_side: int = DEFAULT_LOWRES_MAX_SIDE,
    positive_samples=None,
    negative_samples=None,
    roi_polygons=None,
    ignore_polygons=None,
    config: Optional[LowresFormalMaskConfig] = None,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    cancelled_cb: Optional[Callable[[], bool]] = None,
) -> LowresFormalMaskResult:
    """同步生成低像素正式 working road mask。"""
    cfg = config or LowresFormalMaskConfig(max_side=int(max_side))
    if config is None:
        cfg.max_side = int(max_side)

    def progress(pct: int, msg: str):
        if progress_cb:
            progress_cb(int(pct), str(msg))

    def cancelled() -> bool:
        return bool(cancelled_cb and cancelled_cb())

    t0 = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []
    report: Dict[str, Any] = {
        "mask_source": "lowres_formal_mask",
        "preview_only": False,
        "formal_ready": True,
        "mask_dirty": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "warnings": warnings,
    }

    try:
        if cancelled():
            report["cancelled"] = True
            return LowresFormalMaskResult(
                ok=False, error="用户取消", output_dir=str(out), report=report
            )

        progress(5, "生成低分辨率 working image…")
        work_rgb, scale = build_lowres_work_image(image_path, int(cfg.max_side))
        report.update(scale)
        report["input_image_shape"] = [
            int(scale["original_height"]), int(scale["original_width"]), 3
        ]
        lowres_work_path = _write_png(out / "lowres_work_image.png", work_rgb)
        _write_json(out / "lowres_scale.json", scale)
        if cancelled():
            report["cancelled"] = True
            return LowresFormalMaskResult(
                ok=False, error="用户取消", output_dir=str(out), report=report
            )

        elapsed = time.perf_counter() - t0
        if cfg.max_total_seconds > 0 and elapsed > cfg.max_total_seconds:
            raise LowresFormalMaskTimeout(
                f"低像素正式 Mask 超时（elapsed={elapsed:.1f}s）。请降低 max_side。"
            )

        # valid area
        progress(15, "识别有效影像区域…")
        valid_mask = None
        if cfg.valid_area_only:
            min_black = max(
                64,
                int(4096 * (scale["lowres_width"] / max(1.0, scale["original_width"])) ** 2),
            )
            valid_mask, valid_report = analyze_valid_image_mask(
                work_rgb,
                black_threshold=cfg.black_threshold,
                min_black_component_area=min_black,
            )
            report["valid_area_ratio"] = valid_report.get("valid_area_ratio", 0.0)
            report["valid_mask_report"] = valid_report

        if cancelled():
            report["cancelled"] = True
            return LowresFormalMaskResult(
                ok=False, error="用户取消", output_dir=str(out), report=report
            )

        # OpenCV 分割（复用快速预览核心）
        progress(30, "低像素道路分割…")
        pos = np.asarray(positive_samples if positive_samples is not None else [], dtype=np.uint8)
        neg = np.asarray(negative_samples if negative_samples is not None else [], dtype=np.uint8)
        if pos.size == 0:
            return LowresFormalMaskResult(
                ok=False,
                error="正样本为空，请先添加道路正样本。",
                output_dir=str(out),
                report=report,
            )
        if cfg.use_negative_samples and neg.size == 0:
            return LowresFormalMaskResult(
                ok=False,
                error="负样本为空，请先添加非道路负样本。",
                output_dir=str(out),
                report=report,
            )

        # 分割阶段不做形态学，留给后续统一 clean
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
        raw_mask = segment_road_by_samples(work_rgb, pos, neg, seg_cfg)

        roi_low = _scale_polygons_to_lowres(
            roi_polygons if cfg.use_roi else None,
            scale["scale_x"],
            scale["scale_y"],
        )
        ignore_low = _scale_polygons_to_lowres(
            ignore_polygons if cfg.use_ignore else None,
            scale["scale_x"],
            scale["scale_y"],
        )
        raw_mask = _apply_roi_ignore(raw_mask, roi_low, ignore_low)
        if valid_mask is not None:
            raw_mask = apply_valid_image_mask(raw_mask, valid_mask)

        raw_path = _write_png(out / "lowres_road_mask_raw.png", raw_mask)
        if cancelled():
            report["cancelled"] = True
            return LowresFormalMaskResult(
                ok=False, error="用户取消", output_dir=str(out), report=report
            )

        progress(55, "低像素 mask 轻量清理…")
        cleaned = _light_clean_mask(
            raw_mask,
            open_kernel=cfg.open_kernel,
            close_kernel=cfg.close_kernel,
            remove_small=cfg.remove_small_components,
            min_area=cfg.min_component_area,
            fill_holes=cfg.fill_holes,
        )
        if valid_mask is not None:
            cleaned = apply_valid_image_mask(cleaned, valid_mask)
        cleaned = _apply_roi_ignore(cleaned, roi_low, ignore_low)

        nz = int(np.count_nonzero(cleaned))
        report["mask_nonzero_ratio"] = round(nz / float(max(1, cleaned.size)), 6)
        if nz < 50:
            return LowresFormalMaskResult(
                ok=False,
                error="道路 mask 几乎为空，请检查正负样本或降低阈值。",
                output_dir=str(out),
                report=report,
            )

        lowres_cleaned_path = _write_png(out / "lowres_road_mask_cleaned.png", cleaned)
        if cancelled():
            report["cancelled"] = True
            return LowresFormalMaskResult(
                ok=False, error="用户取消", output_dir=str(out), report=report
            )

        # INTER_NEAREST 放大到原图
        progress(75, "放大 mask 到原图尺寸（INTER_NEAREST）…")
        working = upscale_mask_nearest(
            cleaned,
            int(scale["original_width"]),
            int(scale["original_height"]),
        )
        assert working.shape[1] == int(scale["original_width"])
        assert working.shape[0] == int(scale["original_height"])
        assert working.dtype == np.uint8

        working_path = _write_png(out / "working_road_mask.png", working)

        # 显示用 preview（缩放到约 3000 边长以内，仅显示）
        progress(88, "生成显示用 preview…")
        preview_max = 3000
        oh, ow = working.shape[:2]
        max_dim = max(oh, ow)
        if max_dim > preview_max:
            s = preview_max / float(max_dim)
            pw = max(1, int(round(ow * s)))
            ph = max(1, int(round(oh * s)))
            preview = cv2.resize(working, (pw, ph), interpolation=cv2.INTER_NEAREST)
        else:
            preview = working
        preview_path = _write_png(out / "working_road_mask_preview.png", preview)

        elapsed_total = round(time.perf_counter() - t0, 3)
        report.update({
            "elapsed_total_seconds": elapsed_total,
            "working_road_mask_path": working_path,
            "working_road_mask_preview_path": preview_path,
            "lowres_work_image_path": lowres_work_path,
            "lowres_road_mask_path": lowres_cleaned_path,
            "lowres_road_mask_raw_path": raw_path,
            "formal_ready": True,
            "preview_only": False,
            "mask_dirty": False,
            "mask_source": "lowres_formal_mask",
            "interpolation": "INTER_NEAREST",
        })
        if elapsed_total > 120 and int(cfg.max_side) >= 3000:
            warnings.append(
                f"耗时 {elapsed_total:.1f}s 超过 2 分钟，建议降低 max_side（当前 {cfg.max_side}）。"
            )
            report["warnings"] = warnings

        report_path = _write_json(out / "lowres_formal_mask_report.json", report)
        report["report_path"] = report_path

        progress(100, "低像素正式 Mask 完成")
        return LowresFormalMaskResult(
            ok=True,
            output_dir=str(out),
            working_mask_path=working_path,
            working_mask_preview_path=preview_path,
            lowres_work_image_path=lowres_work_path,
            lowres_road_mask_path=lowres_cleaned_path,
            report=report,
            working_mask=working,
            warning="; ".join(warnings),
        )

    except LowresFormalMaskTimeout as exc:
        report["elapsed_total_seconds"] = round(time.perf_counter() - t0, 3)
        report["timed_out"] = True
        warnings.append(str(exc))
        report["warnings"] = warnings
        _write_json(out / "lowres_formal_mask_report.json", report)
        return LowresFormalMaskResult(
            ok=False,
            timed_out=True,
            output_dir=str(out),
            report=report,
            warning=str(exc),
            error=str(exc),
        )
    except Exception as exc:
        report["elapsed_total_seconds"] = round(time.perf_counter() - t0, 3)
        report["error"] = str(exc)
        if "取消" in str(exc):
            report["cancelled"] = True
        try:
            _write_json(out / "lowres_formal_mask_report.json", report)
        except Exception:
            pass
        return LowresFormalMaskResult(
            ok=False, output_dir=str(out), report=report, error=str(exc)
        )
