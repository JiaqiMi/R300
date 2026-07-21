"""
可视化模块 V2.1：叠加图 + 样本点标注 + 前后对比图。
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def overlay_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
    color: tuple = (0, 255, 0),
) -> np.ndarray:
    """
    将二值 mask 以半透明方式叠加在原图上。

    Args:
        rgb:   原始 RGB 图像 (H, W, 3), dtype uint8
        mask:  二值 mask (H, W), dtype uint8, 值为 0 或 255
        alpha: mask 叠加透明度 (0.0~1.0)
        color: mask 着色 (R, G, B)，默认绿色

    Returns:
        RGB 叠加结果 (H, W, 3), dtype uint8
    """
    overlay = rgb.copy()
    overlay[mask > 0] = color
    blended = cv2.addWeighted(rgb, 1.0 - alpha, overlay, alpha, 0)
    return blended


def draw_sample_markers(
    image_rgb: np.ndarray,
    pos_points: List[Tuple[int, int]],
    neg_points: List[Tuple[int, int]],
    marker_radius: int = 5,
) -> np.ndarray:
    """
    在图像上直接绘制正负样本点标记（使用 OpenCV，无需 matplotlib）。

    Args:
        image_rgb:    原始 RGB 图像 (H, W, 3)
        pos_points:   正样本像素坐标列表 [(x, y), ...]
        neg_points:   负样本像素坐标列表 [(x, y), ...]
        marker_radius: 标记圆圈半径

    Returns:
        标注后的 RGB 图像 (H, W, 3)
    """
    # 转为 BGR 以便 OpenCV 绘制
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    result = img_bgr.copy()

    # 画正样本（绿色空心圆）
    for (px, py) in pos_points:
        cv2.circle(result, (px, py), marker_radius, (0, 255, 0), 2)

    # 画负样本（红色叉号 = 两条交叉线）
    for (px, py) in neg_points:
        r = marker_radius
        cv2.line(result, (px - r, py - r), (px + r, py + r), (0, 0, 255), 2)
        cv2.line(result, (px + r, py - r), (px - r, py + r), (0, 0, 255), 2)

    # 转回 RGB
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def draw_roi_polygon_on_image(
    image_rgb: np.ndarray,
    vertices: List[Tuple[float, float]],
    color: tuple = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    在图像上绘制 ROI 多边形（用于保存 roi_visual.png）。
    支持单个多边形顶点列表，向后兼容。

    Args:
        image_rgb: 原始 RGB 图像
        vertices:  多边形顶点 [(x, y), ...] 像素坐标（单区域）
        color:     线条颜色 (R, G, B)
        thickness: 线条粗细

    Returns:
        标注后的 RGB 图像
    """
    if vertices is None or len(vertices) == 0:
        return image_rgb.copy()

    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    pts = np.array(vertices, dtype=np.int32)
    cv2.polylines(img_bgr, [pts], isClosed=True, color=color[::-1], thickness=thickness)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def draw_roi_regions_on_image(
    image_rgb: np.ndarray,
    regions: List[List[Tuple[float, float]]],
    colors: Optional[List[tuple]] = None,
    fill_alpha: float = 0.2,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    在图像上绘制多个 ROI 多边形区域（V2.3 多区域支持）。

    Args:
        image_rgb:    原始 RGB 图像
        regions:      多个多边形区域的顶点列表，每个区域为 [(x, y), ...] 像素坐标
        colors:       各区域线条颜色列表 (R, G, B)，默认全部红色
        fill_alpha:   区域填充透明度 (0.0~1.0)
        line_thickness: 线条粗细

    Returns:
        标注后的 RGB 图像
    """
    if regions is None or len(regions) == 0:
        return image_rgb.copy()

    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = img_bgr.copy()

    if colors is None:
        # 默认使用不同色相的红色系/蓝色系
        import colorsys
        n = len(regions)
        colors = []
        for i in range(n):
            hue = (i * 0.618) % 1.0  # 黄金比例分布色相
            r, g, b = colorsys.hsv_to_rgb(hue % 1.0, 0.8, 1.0)
            # RGB 顺序（opencv 是 BGR，所以后面会 reverse）
            colors.append((int(r * 255), int(g * 255), int(b * 255)))

    for i, region in enumerate(regions):
        if len(region) < 3:
            continue
        pts = np.array(region, dtype=np.int32).reshape((-1, 1, 2))
        color_bgr = colors[i % len(colors)][::-1]  # RGB -> BGR

        # 半透明填充
        cv2.fillPoly(overlay, [pts], color_bgr)
        # 边框线条
        cv2.polylines(overlay, [pts], isClosed=True, color=color_bgr,
                      thickness=line_thickness)

    # 混合填充
    result = cv2.addWeighted(img_bgr, 1.0 - fill_alpha, overlay, fill_alpha, 0)
    # 重绘线条使其不被混合冲淡
    for i, region in enumerate(regions):
        if len(region) < 3:
            continue
        pts = np.array(region, dtype=np.int32).reshape((-1, 1, 2))
        color_bgr = colors[i % len(colors)][::-1]
        cv2.polylines(result, [pts], isClosed=True, color=color_bgr,
                      thickness=line_thickness)

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def save_postprocess_compare(
    image_rgb: np.ndarray,
    raw_mask: np.ndarray,
    clean_mask: np.ndarray,
    output_path: str,
    alpha: float = 0.45,
) -> None:
    """
    保存后处理前后对比图（并排 2x2 网格）。

    布局：
        [raw_mask]           [clean_mask]
        [raw_overlay]        [clean_overlay]

    Args:
        image_rgb:   原始 RGB 图像
        raw_mask:    V1.5 原始分割 mask
        clean_mask:  V2.1 清理后 mask
        output_path: 保存路径
        alpha:       叠加透明度
    """
    import os

    # 转为 BGR 以便拼接
    raw_mask_disp = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
    clean_mask_disp = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)

    # 叠加图
    raw_overlay_bgr = cv2.cvtColor(
        overlay_mask(image_rgb, raw_mask, alpha=alpha), cv2.COLOR_RGB2BGR
    )
    clean_overlay_bgr = cv2.cvtColor(
        overlay_mask(image_rgb, clean_mask, alpha=alpha), cv2.COLOR_RGB2BGR
    )

    # 并排拼接：上排=mask, 下排=overlay
    top = np.hstack([raw_mask_disp, clean_mask_disp])
    bottom = np.hstack([raw_overlay_bgr, clean_overlay_bgr])

    # 添加标签文本
    h, w = raw_mask_disp.shape[:2]
    # 上半部分加上标签
    cv2.putText(top, "Raw Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 255), 2)
    cv2.putText(top, "Clean Mask", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2)

    combined = np.vstack([top, bottom])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, combined)


def resize_image_to_max(image: np.ndarray, max_dim: int = 800) -> np.ndarray:
    """
    按最大边长等比例缩放图像，用于显示大图时节省内存。

    Args:
        image:  输入图像
        max_dim: 最大边长（像素）

    Returns:
        缩放后的图像
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image

    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
