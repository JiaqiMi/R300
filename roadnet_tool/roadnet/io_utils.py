"""
IO 工具：图像读取、保存。
"""

import os
import cv2
import numpy as np


def read_image_rgb(path: str) -> np.ndarray:
    """
    读取图像并转换为 RGB 格式（OpenCV 默认 BGR）。

    Args:
        path: 图像文件路径

    Returns:
        RGB 格式的 numpy 数组 (H, W, 3)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"图像文件不存在: {path}")

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"无法读取图像（文件可能损坏或格式不支持）: {path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def save_image(path: str, img: np.ndarray) -> None:
    """
    保存图像到指定路径。自动将 RGB 转为 BGR 再写入。

    Args:
        path: 输出文件路径
        img:  RGB 或灰度图像 (H, W) 或 (H, W, 3)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if img.ndim == 3 and img.shape[2] == 3:
        # RGB -> BGR
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        out = img

    cv2.imwrite(path, out)
