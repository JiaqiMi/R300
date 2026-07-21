"""
SAM-Road mask 后处理模块：将 SAM-Road 生成的 road score / mask 转为干净二值 mask。

固定比赛默认流水线：
1. 读取原始 mask
2. 转单通道 uint8（保留灰度供阈值）
3. threshold = 240
4. blur = 3
5. close = 5
6. open = 3
7. ensure_binary_uint8_mask → connectedComponentsWithStats(min_area=500)
8. use_roi = true → 只保留 ROI
9. use_ignore = true → 删除 Ignore
10. fill_holes = false（默认不填孔）
11. 输出 processed_mask.png
"""

from __future__ import annotations

import json
import cv2
import numpy as np
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Callable


# 比赛固定默认参数（打开后处理窗口时使用）
COMPETITION_MASK_POSTPROCESS_DEFAULTS = {
    "threshold": 240,
    "blur_kernel": 3,
    "close_kernel": 5,
    "open_kernel": 3,
    "min_area": 500,
    "fill_holes": False,
    "use_roi": True,
    "use_ignore": True,
    "keep_largest": 0,
}


@dataclass
class MaskPostprocessConfig:
    """Mask 后处理配置（默认值 = 比赛固定参数）"""
    threshold: int = 240
    blur_kernel: int = 3
    close_kernel: int = 5
    open_kernel: int = 3
    min_area: int = 500
    fill_holes: bool = False
    keep_largest: int = 0
    use_roi: bool = True
    use_ignore: bool = True

    @classmethod
    def competition_defaults(cls) -> "MaskPostprocessConfig":
        return cls(**COMPETITION_MASK_POSTPROCESS_DEFAULTS)


def ensure_binary_uint8_mask(mask) -> np.ndarray:
    """Ensure mask is single-channel uint8 binary (0 / 255) for OpenCV CC APIs.

    Raises ValueError if mask is None.
    """
    if mask is None:
        raise ValueError("mask is None")

    arr = np.asarray(mask)
    if arr.size == 0:
        raise ValueError("mask is empty")

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            if arr.dtype == np.bool_ or arr.dtype == bool:
                arr = arr.astype(np.uint8) * 255
            elif arr.dtype != np.uint8:
                arr_f = arr.astype(np.float64)
                mx = float(np.nanmax(arr_f)) if arr_f.size else 0.0
                if mx <= 1.0 + 1e-9:
                    arr = np.clip(arr_f * 255.0, 0, 255).astype(np.uint8)
                else:
                    arr = np.clip(arr_f, 0, 255).astype(np.uint8)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    if arr.ndim != 2:
        raise ValueError(f"mask must be H×W after conversion, got shape={arr.shape}")

    if arr.dtype == np.bool_ or arr.dtype == bool:
        out = arr.astype(np.uint8) * 255
    else:
        out = (np.asarray(arr) > 0).astype(np.uint8) * 255

    if out.dtype != np.uint8:
        out = out.astype(np.uint8)
    if out.ndim != 2:
        raise ValueError(f"ensure_binary_uint8_mask produced invalid shape={out.shape}")
    return out


def ensure_gray_uint8_mask(mask) -> np.ndarray:
    """Convert mask to HxW uint8 grayscale (0–255), preserving score levels for threshold."""
    if mask is None:
        raise ValueError("mask is None")
    arr = np.asarray(mask)
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            if arr.dtype == np.bool_:
                arr = arr.astype(np.uint8) * 255
            arr = cv2.cvtColor(
                arr if arr.dtype == np.uint8 else np.clip(arr, 0, 255).astype(np.uint8),
                cv2.COLOR_BGR2GRAY,
            )
    if arr.ndim != 2:
        raise ValueError(f"mask must be H×W, got shape={arr.shape}")

    if arr.dtype == np.bool_ or arr.dtype == bool:
        return arr.astype(np.uint8) * 255
    if arr.dtype == np.uint8:
        return arr.copy()

    arr_f = arr.astype(np.float64)
    mx = float(np.nanmax(arr_f)) if arr_f.size else 0.0
    if mx <= 1.0 + 1e-9:
        return np.clip(arr_f * 255.0, 0, 255).astype(np.uint8)
    if mx <= 255.0 + 1e-6:
        return np.clip(arr_f, 0, 255).astype(np.uint8)
    # uint16 / large ints: scale to 0–255
    mn = float(np.nanmin(arr_f))
    if mx > mn:
        scaled = (arr_f - mn) / (mx - mn) * 255.0
    else:
        scaled = np.zeros_like(arr_f)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _log_mask_stats(log_fn: Optional[Callable[[str], None]], prefix: str, mask: np.ndarray) -> None:
    if log_fn is None:
        return
    uniq = np.unique(mask)
    uniq_preview = uniq[:16].tolist()
    if len(uniq) > 16:
        uniq_preview.append("...")
    log_fn(f"{prefix} dtype={mask.dtype} shape={mask.shape} unique={uniq_preview}")


def postprocess_samroad_mask(
    score_or_mask: np.ndarray,
    config: Optional[MaskPostprocessConfig] = None,
    roi_polygons: Optional[List[np.ndarray]] = None,
    ignore_polygons: Optional[List[np.ndarray]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
    """
    SAM-Road mask 后处理主入口。

    Returns:
        (processed_mask, steps)
        processed_mask: 处理后二值 mask (H, W) uint8 0/255
        steps:          List of (step_name, intermediate_mask)
    """
    if config is None:
        config = MaskPostprocessConfig.competition_defaults()

    def _log(msg: str) -> None:
        line = f"[SAM-Road Mask后处理] {msg}"
        print(line)
        if log_fn is not None:
            log_fn(line)

    steps: List[Tuple[str, np.ndarray]] = []

    raw = score_or_mask
    _log(f"mask dtype before={getattr(raw, 'dtype', None)}")
    _log(f"mask shape before={getattr(raw, 'shape', None)}")

    # 1–2: 转单通道 uint8 灰度（保留 score 供 threshold）
    current = ensure_gray_uint8_mask(raw)
    _log(f"mask dtype after gray_uint8={current.dtype}")
    _log(f"mask shape after gray_uint8={current.shape}")
    steps.append(("00_input", current.copy()))

    _log(
        f"params threshold={config.threshold} blur={config.blur_kernel} "
        f"close={config.close_kernel} open={config.open_kernel} "
        f"min_area={config.min_area} fill_holes={config.fill_holes} "
        f"use_roi={config.use_roi} use_ignore={config.use_ignore}"
    )

    # 3: threshold
    _, current = cv2.threshold(
        current, int(config.threshold), 255, cv2.THRESH_BINARY,
    )
    current = ensure_binary_uint8_mask(current)
    steps.append(("01_threshold", current.copy()))

    # 4: blur
    if config.blur_kernel > 0:
        k = int(config.blur_kernel)
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(current.astype(np.float32), (k, k), 0)
        _, current = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        current = ensure_binary_uint8_mask(current)
        steps.append(("02_blur", current.copy()))

    # 5: close
    if config.close_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.close_kernel, config.close_kernel)
        )
        current = cv2.morphologyEx(current, cv2.MORPH_CLOSE, kernel)
        current = ensure_binary_uint8_mask(current)
        steps.append(("03_close", current.copy()))

    # 6: open
    if config.open_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.open_kernel, config.open_kernel)
        )
        current = cv2.morphologyEx(current, cv2.MORPH_OPEN, kernel)
        current = ensure_binary_uint8_mask(current)
        steps.append(("04_open", current.copy()))

    # 7: connectedComponentsWithStats 过滤 min_area
    if config.min_area > 1:
        current = _remove_small_components(current, config.min_area, log_fn=_log)
        steps.append(("05_remove_small", current.copy()))

    # keep_largest（可选，默认 0）
    if config.keep_largest > 0:
        current = _keep_largest_n(current, config.keep_largest, log_fn=_log)
        steps.append(("06_keep_largest", current.copy()))

    # 8: ROI
    if config.use_roi and roi_polygons:
        current = _apply_roi(current, roi_polygons)
        current = ensure_binary_uint8_mask(current)
        steps.append(("07_roi", current.copy()))

    # 9: Ignore
    if config.use_ignore and ignore_polygons:
        current = _apply_ignore(current, ignore_polygons)
        current = ensure_binary_uint8_mask(current)
        steps.append(("08_ignore", current.copy()))

    # 10: fill_holes（默认关闭）
    if config.fill_holes:
        current = _fill_holes(current)
        current = ensure_binary_uint8_mask(current)
        steps.append(("09_fill_holes", current.copy()))

    current = ensure_binary_uint8_mask(current)
    _log(f"mask dtype after ensure_binary_uint8_mask={current.dtype}")
    _log(f"mask shape after={current.shape}")
    _log(f"unique values after={np.unique(current).tolist()}")

    return current, steps


# ===========================================================================
# 内部实现
# ===========================================================================

def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """填充 mask 内部孔洞。"""
    mask = ensure_binary_uint8_mask(mask)
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    original_padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    original_padded[1:-1, 1:-1] = mask
    result = cv2.bitwise_or(original_padded, flood_inv)
    return ensure_binary_uint8_mask(result[1:-1, 1:-1])


def _remove_small_components(
    mask: np.ndarray,
    min_area: int,
    log_fn: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """删除面积小于 min_area 的连通分量。"""
    binary = ensure_binary_uint8_mask(mask)
    if log_fn:
        log_fn(
            f"before connectedComponentsWithStats dtype={binary.dtype} "
            f"shape={binary.shape} unique={np.unique(binary).tolist()}"
        )
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return ensure_binary_uint8_mask(cleaned)


def _keep_largest_n(
    mask: np.ndarray,
    n: int,
    log_fn: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """只保留面积最大的前 n 个连通域。"""
    binary = ensure_binary_uint8_mask(mask)
    if log_fn:
        log_fn(
            f"before connectedComponentsWithStats(keep_largest) dtype={binary.dtype} "
            f"shape={binary.shape}"
        )
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    areas.sort(key=lambda x: x[1], reverse=True)
    keep_labels = {lb for lb, _ in areas[:n]}
    cleaned = np.zeros_like(binary)
    for lb in keep_labels:
        cleaned[labels == lb] = 255
    return ensure_binary_uint8_mask(cleaned)


def _apply_roi(mask: np.ndarray, roi_polygons: List[np.ndarray]) -> np.ndarray:
    """用 ROI 多边形约束 mask（只保留 ROI 内的部分）。"""
    mask = ensure_binary_uint8_mask(mask)
    roi_mask = np.zeros(mask.shape, dtype=np.uint8)
    for poly in roi_polygons:
        pts = _to_int_points(poly)
        if len(pts) >= 3:
            cv2.fillPoly(roi_mask, [pts], 255)
    roi_mask = ensure_binary_uint8_mask(roi_mask)
    return ensure_binary_uint8_mask(cv2.bitwise_and(mask, roi_mask))


def _apply_ignore(mask: np.ndarray, ignore_polygons: List[np.ndarray]) -> np.ndarray:
    """用 Ignore 多边形屏蔽 mask（删除 Ignore 内的部分）。"""
    mask = ensure_binary_uint8_mask(mask)
    ignore_mask = np.zeros(mask.shape, dtype=np.uint8)
    for poly in ignore_polygons:
        pts = _to_int_points(poly)
        if len(pts) >= 3:
            cv2.fillPoly(ignore_mask, [pts], 255)
    ignore_mask = ensure_binary_uint8_mask(ignore_mask)
    return ensure_binary_uint8_mask(cv2.bitwise_and(mask, cv2.bitwise_not(ignore_mask)))


def _to_int_points(poly: np.ndarray) -> np.ndarray:
    """将多边形顶点转为 int32。"""
    if poly.dtype != np.int32:
        poly = poly.astype(np.int32)
    return poly.reshape(-1, 1, 2)


# ===========================================================================
# 用户“保存为默认参数”（显式保存后下次打开才覆盖比赛默认）
# ===========================================================================

def user_defaults_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "config" / "samroad_mask_postprocess_user_defaults.json"


def load_dialog_defaults() -> MaskPostprocessConfig:
    """打开对话框时使用的参数：有用户显式保存则用保存值，否则比赛默认。"""
    path = user_defaults_path()
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            base = asdict(MaskPostprocessConfig.competition_defaults())
            base.update({k: data[k] for k in base if k in data})
            return MaskPostprocessConfig(**base)
        except Exception as exc:
            print(f"[SAM-Road Mask后处理] 读取用户默认参数失败，回退比赛默认: {exc}")
    return MaskPostprocessConfig.competition_defaults()


def save_user_defaults(config: MaskPostprocessConfig) -> str:
    """用户点击「保存为默认参数」时写入。"""
    path = user_defaults_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold": int(config.threshold),
        "blur_kernel": int(config.blur_kernel),
        "close_kernel": int(config.close_kernel),
        "open_kernel": int(config.open_kernel),
        "min_area": int(config.min_area),
        "fill_holes": bool(config.fill_holes),
        "use_roi": bool(config.use_roi),
        "use_ignore": bool(config.use_ignore),
        "keep_largest": int(config.keep_largest),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


# ===========================================================================
# 便捷函数：从已有图层获取 mask 并后处理
# ===========================================================================

def process_mask_from_layer(
    layer_data: np.ndarray,
    config: Optional[MaskPostprocessConfig] = None,
    roi_data: Optional[np.ndarray] = None,
    ignore_data: Optional[np.ndarray] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
    """
    从图层数据读取 mask 并执行后处理。
    """
    if config is None:
        config = MaskPostprocessConfig.competition_defaults()

    roi_polys = None
    ignore_polys = None
    if config.use_roi and roi_data is not None:
        roi_polys = _mask_to_polygons(roi_data)
    if config.use_ignore and ignore_data is not None:
        ignore_polys = _mask_to_polygons(ignore_data)

    return postprocess_samroad_mask(
        layer_data, config,
        roi_polygons=roi_polys,
        ignore_polygons=ignore_polys,
        log_fn=log_fn,
    )


def _mask_to_polygons(mask: Optional[np.ndarray]) -> List[np.ndarray]:
    """二值 mask → 多边形轮廓列表。ROI/Ignore 先转 uint8 单通道二值。"""
    if mask is None:
        return []
    binary = ensure_binary_uint8_mask(mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt.reshape(-1, 2) for cnt in contours if len(cnt) >= 3]
