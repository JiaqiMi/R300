"""Tests for the shared OpenCV road segmenter and tile-status overlay."""

import numpy as np

from roadnet.opencv_road_segmenter import (
    segment_road_by_samples,
    apply_mask_morphology,
    color_space_to_mode,
    DEFAULT_LARGE_OPENCV_CONFIG,
)


def _road_image():
    """构造一张含明显道路色带的 RGB 图。"""
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    img[:] = (30, 120, 40)          # 背景绿色（草地）
    img[25:35, :] = (180, 180, 180)  # 道路灰色横带
    return img


def test_color_space_to_mode_mapping():
    assert color_space_to_mode("Lab+HSV") == "combined"
    assert color_space_to_mode("HSV+Lab") == "combined"
    assert color_space_to_mode("hsv") == "hsv"
    assert color_space_to_mode("lab") == "lab"
    assert color_space_to_mode("") is None
    assert color_space_to_mode("unknown") is None


def test_segment_road_by_samples_detects_road():
    img = _road_image()
    pos = np.array([[180, 180, 180]], dtype=np.uint8)
    neg = np.array([[30, 120, 40]], dtype=np.uint8)
    cfg = dict(DEFAULT_LARGE_OPENCV_CONFIG)
    mask = segment_road_by_samples(img, pos, neg, cfg)
    assert mask.shape == img.shape[:2]
    assert mask.dtype == np.uint8
    # 道路带中心应被检出
    assert mask[30, 30] > 0
    # 背景不应大面积检出
    assert mask[5, 5] == 0


def test_segment_road_by_samples_backward_compatible_no_morphology():
    """不传形态学键时结果应等价于纯 segment_road（不做额外清理）。"""
    from roadnet.color_segment import segment_road
    img = _road_image()
    pos = np.array([[180, 180, 180]], dtype=np.uint8)
    neg = np.array([[30, 120, 40]], dtype=np.uint8)
    cfg = {"mode": "combined"}
    a = segment_road_by_samples(img, pos, neg, cfg)
    b = segment_road(img, pos, neg, cfg)
    assert np.array_equal(a, b)


def test_apply_mask_morphology_min_area_preserves_edge_components():
    mask = np.zeros((50, 50), dtype=np.uint8)
    # 小的中心斑块（应被 min_area 移除）
    mask[24:26, 24:26] = 255
    # 触碰边缘的小斑块（应保留）
    mask[0:2, 0:2] = 255
    out = apply_mask_morphology(mask, {"min_area": 100})
    assert out[0, 0] > 0          # 边缘斑块保留
    assert out[24, 24] == 0       # 中心小斑块移除


def test_apply_mask_morphology_noop_when_no_keys():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:8, 5:8] = 255
    out = apply_mask_morphology(mask, {})
    assert np.array_equal(out, (mask > 0).astype(np.uint8) * 255)


def test_render_tile_status_overlay_colors():
    from roadnet.large_image_worker import render_tile_status_overlay, _TILE_STATUS_COLORS
    records = [
        {"x0": 0, "y0": 0, "x1": 100, "y1": 100,
         "success": True, "failed": False, "skipped_black": False, "cache_hit": False},
        {"x0": 100, "y0": 0, "x1": 200, "y1": 100,
         "success": False, "failed": True, "skipped_black": False, "cache_hit": False},
        {"x0": 0, "y0": 100, "x1": 100, "y1": 200,
         "success": False, "failed": False, "skipped_black": True, "cache_hit": False},
    ]
    overlay = render_tile_status_overlay(200, 200, records, max_side=200)
    assert overlay.shape == (200, 200, 3)
    assert overlay.dtype == np.uint8
    # overlay 应包含非零像素（绘制了 tile 边框/填充）
    assert overlay.sum() > 0


def test_tile_status_label_priority():
    from roadnet.large_image_worker import _tile_status_label
    assert _tile_status_label({"failed": True, "success": True}) == "failed"
    assert _tile_status_label({"cache_hit": True, "success": True}) == "cache"
    assert _tile_status_label({"success": True}) == "success"
    assert _tile_status_label({"skipped_black": True}) == "skipped"
    assert _tile_status_label({}) == "pending"
