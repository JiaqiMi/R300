"""公共 OpenCV 样本/颜色道路分割模块。

大图“快速预览提取”与“OpenCV 正式提取”共用本模块的核心函数
``segment_road_by_samples``，从而保证预览结果与正式结果一致。

设计要点：
- 核心颜色分割复用 ``roadnet.color_segment.segment_road``（RGB 输入）。
- 形态学后处理（blur/open/close/min_area/fill_holes）仅在 config 中显式
  提供对应键且为正值时才执行；因此旧调用方（不传这些键）行为保持不变。
- ``min_area`` 过滤会保留触碰图块边缘的连通域，避免删除跨 tile 的道路。
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from roadnet.color_segment import segment_road

# 大图 OpenCV 正式提取默认参数（见需求第九条）。
DEFAULT_LARGE_OPENCV_CONFIG: Dict[str, Any] = {
    "color_space": "Lab+HSV",
    "blur_kernel": 3,
    "open_kernel": 3,
    "close_kernel": 5,
    "min_area": 100,
    "fill_holes": False,
    "use_roi": True,
    "use_ignore": True,
    "debug_mode": False,
}

# color_space -> color_segment.segment_road 的 mode 映射。
_COLOR_SPACE_TO_MODE = {
    "lab+hsv": "combined",
    "hsv+lab": "combined",
    "combined": "combined",
    "hsv": "hsv",
    "lab": "lab",
}


def color_space_to_mode(color_space) -> str | None:
    """将 color_space 字符串映射为 segment_road 的 mode，无法识别返回 None。"""
    if not color_space:
        return None
    return _COLOR_SPACE_TO_MODE.get(str(color_space).strip().lower())


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """按面积移除小连通域，但保留触碰图块边缘的连通域（跨 tile 道路）。"""
    if min_area <= 1:
        return mask
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(mask)
    h, w = mask.shape[:2]
    for label in range(1, count):
        x, y, ww, hh, area = stats[label]
        touches_edge = x == 0 or y == 0 or x + ww >= w or y + hh >= h
        if touches_edge or int(area) >= int(min_area):
            cleaned[labels == label] = 255
    return cleaned


def _fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    """填充不触碰边界的小孔洞。大图默认禁用（fill_holes=False）。"""
    inverse = (mask == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    result = mask.copy()
    h, w = mask.shape[:2]
    for label in range(1, count):
        x, y, ww, hh, area = stats[label]
        touches_border = x == 0 or y == 0 or x + ww >= w or y + hh >= h
        if not touches_border and int(area) <= int(max_area):
            result[labels == label] = 255
    return result


def apply_mask_morphology(mask: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """按 config 对二值 mask 应用形态学后处理。

    只有当对应键存在且为正值时才执行相应操作，保证向后兼容。
    """
    cfg = dict(config or {})
    out = np.asarray(mask, dtype=np.uint8)
    out = (out > 0).astype(np.uint8) * 255

    blur = int(cfg.get("blur_kernel", 0) or 0)
    if blur > 0:
        out = cv2.blur(out, (blur, blur))
        out = (out > 128).astype(np.uint8) * 255

    open_k = int(cfg.get("open_kernel", 0) or 0)
    if open_k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)

    close_k = int(cfg.get("close_kernel", 0) or 0)
    if close_k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

    if bool(cfg.get("fill_holes", False)):
        out = _fill_small_holes(out, int(cfg.get("max_hole_area", 500)))

    min_area = int(cfg.get("min_area", 0) or 0)
    if min_area > 1:
        out = _remove_small_components(out, min_area)

    return out


def segment_road_by_samples(
    image: np.ndarray,
    positive_samples,
    negative_samples,
    config: Dict[str, Any] | None = None,
) -> np.ndarray:
    """基于正/负颜色样本的道路分割（预览与正式提取共用核心）。

    Args:
        image: (H, W, 3) uint8 RGB 图像（与 color_segment.segment_road 一致）。
        positive_samples: (N, 3) 道路颜色样本。
        negative_samples: (M, 3) 非道路颜色样本，可为空。
        config: 分割配置。支持 color_space / mode / combine_method /
            h_margin / s_margin / v_margin / lab_margin /
            use_negative_samples / positive_distance_threshold /
            negative_margin，以及形态学键
            blur_kernel / open_kernel / close_kernel / min_area / fill_holes。

    Returns:
        (H, W) uint8 二值 mask，255=道路，0=非道路。
    """
    cfg = dict(config or {})
    seg_cfg = dict(cfg)

    mode = color_space_to_mode(cfg.get("color_space"))
    if mode is not None:
        seg_cfg["mode"] = mode

    image_rgb = np.asarray(image)
    pos = np.asarray(positive_samples, dtype=np.uint8).reshape(-1, 3) \
        if len(positive_samples) else np.zeros((0, 3), dtype=np.uint8)
    neg = np.asarray(negative_samples, dtype=np.uint8).reshape(-1, 3) \
        if len(negative_samples) else np.zeros((0, 3), dtype=np.uint8)

    mask = segment_road(image_rgb, pos, neg, seg_cfg)
    mask = np.asarray(mask, dtype=np.uint8)

    mask = apply_mask_morphology(mask, cfg)
    return mask
