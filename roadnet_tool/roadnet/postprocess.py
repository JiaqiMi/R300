"""
后处理增强模块 V2.1：对二值 mask 进行全面的清理和优化。

处理流水线：
1. 开运算：去除小噪声和细小毛刺
2. 闭运算：连接小断裂
3. 孔洞填充
4. 连通域筛选：删除小面积区域
5. 保留最大的前 N 个连通域
6. 边界平滑
7. 删除细长孤立误检区域
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


def clean_pipeline(
    mask: np.ndarray,
    config: Dict[str, Any],
    save_intermediate: bool = False,
    output_dir: str = "",
    return_stats: bool = False,
) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
    """
    后处理流水线主入口。

    Args:
        mask:              输入二值 mask (H, W), dtype uint8
        config:            后处理配置字典
        save_intermediate: 是否保存中间结果
        output_dir:        中间结果输出目录
        return_stats:      是否返回后处理统计信息（用于异常检测）

    Returns:
        (cleaned_mask, intermediates)
        cleaned_mask:  清理后的 mask (H, W) uint8
        intermediates: List of (step_name, mask) for debugging
    """
    intermediates: List[Tuple[str, np.ndarray]] = []
    current = mask.copy()

    open_ks = config.get("open_kernel_size", 5)
    close_ks = config.get("close_kernel_size", 3)
    # ★ 兼容旧配置：fill_holes 为 True 时启用小孔洞填充
    fill_holes = config.get("fill_holes", False)
    fill_small = config.get("fill_small_holes", fill_holes)
    max_hole_area = config.get("max_hole_area", 500)
    min_area = config.get("min_area", 800)
    keep_largest = config.get("keep_largest_components", 5)
    smooth_ks = config.get("smooth_kernel_size", 5)
    remove_thin = config.get("remove_thin_noise", True)

    # 保存填充统计
    fill_stats = {"fill_added_area": 0, "filled_small_holes": 0, "skipped_large_holes": 0}

    # Step 1: 开运算 - 去除小噪声和毛刺
    current = morphological_open(current, kernel_size=open_ks)
    _record(current, "01_open", intermediates, save_intermediate, output_dir)

    # Step 2: 闭运算 - 连接小断裂（默认保护：不超过 5）
    current = morphological_close(current, kernel_size=close_ks)
    _record(current, "02_close", intermediates, save_intermediate, output_dir)

    # Step 3: 孔洞填充（★ 只填充小孔洞，不超过 max_hole_area）
    if fill_small:
        before_fill = current.copy()
        current, fill_stats = fill_small_mask_holes(current, max_hole_area=max_hole_area)
        after_fill = current.copy()
        fill_stats["fill_added_area"] = int((after_fill > 0).sum()) - int((before_fill > 0).sum())
        _record(current, "03_fill_holes", intermediates, save_intermediate, output_dir)
    elif fill_holes:
        # 旧行为（不推荐）—— 留着兼容但有警告
        import warnings
        warnings.warn("fill_holes=True 将填充所有孔洞（风险操作）。建议改用 fill_small_holes=True 并设置 max_hole_area。")
        current = fill_mask_holes(current)
        _record(current, "03_fill_holes", intermediates, save_intermediate, output_dir)

    # Step 4: 连通域筛选 - 删除小面积
    current = remove_small_components(current, min_area=min_area)
    _record(current, "04_remove_small", intermediates, save_intermediate, output_dir)

    # Step 5: 保留最大的前 N 个连通域
    if keep_largest > 0:
        current = keep_largest_n_components(current, n=keep_largest)
        _record(current, "05_keep_largest", intermediates, save_intermediate, output_dir)

    # Step 6: 边界平滑
    if smooth_ks > 0:
        current = smooth_boundary(current, kernel_size=smooth_ks)
        _record(current, "06_smooth", intermediates, save_intermediate, output_dir)

    # Step 7: 删除细长孤立误检
    if remove_thin:
        current = remove_thin_blobs(current)
        _record(current, "07_remove_thin", intermediates, save_intermediate, output_dir)

    return current, intermediates


# ===========================================================================
# 各后处理步骤实现
# ===========================================================================

def morphological_open(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    开运算（先腐蚀后膨胀）：去除小噪声和细小毛刺。

    Args:
        mask:       二值 mask (H, W), 0/255
        kernel_size: 结构元素大小
        iterations:  迭代次数

    Returns:
        处理后的 mask
    """
    if kernel_size <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return result


def morphological_close(
    mask: np.ndarray,
    kernel_size: int = 9,
    iterations: int = 1,
) -> np.ndarray:
    """
    闭运算（先膨胀后腐蚀）：连接小断裂。

    Args:
        mask:       二值 mask (H, W), 0/255
        kernel_size: 结构元素大小
        iterations:  迭代次数

    Returns:
        处理后的 mask
    """
    if kernel_size <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return result


def morphological_clean(
    mask: np.ndarray,
    kernel_size: int = 3,
    open_iter: int = 1,
    close_iter: int = 2,
) -> np.ndarray:
    """
    形态学清理（兼容旧接口）：
    1. 开运算去噪
    2. 闭运算填孔

    兼容 V1.x 调用方式。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    return mask


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """
    孔洞填充：将 mask 内部封闭的空洞区域填充为道路。

    原理：从图像边界 flood-fill 背景，未被填充的 0 区域即为孔洞。

    ⚠ 注意：此函数填充所有孔洞，包括大面积草地/空地。
    对城市影像建议使用 fill_small_mask_holes() 只填充小孔洞。

    Args:
        mask: 二值 mask (H, W), 0/255

    Returns:
        填充孔洞后的 mask
    """
    h, w = mask.shape
    # 创建比原图大 2 像素的边框（确保边界像素可连通）
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask

    # flood-fill 从 (0,0) 开始填充背景
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 255)

    # 反转 flood：背景=0，孔洞+前景=255
    flood_inv = cv2.bitwise_not(flood)

    # 原始 mask（扩展到 padded 尺寸）
    original_padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    original_padded[1:-1, 1:-1] = mask

    # 孔洞 = 前景 OR (flood 未填充的区域)
    result = cv2.bitwise_or(original_padded, flood_inv)

    # 裁回原尺寸
    return result[1:-1, 1:-1]


def fill_small_mask_holes(mask: np.ndarray, max_hole_area: int = 500) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    只填充面积小于 max_hole_area 的小孔洞，保留大孔洞（如草地、空地）。

    原理：
    1. 检测所有孔洞（通过 flood-fill 找到背景中的"孤岛"）
    2. 仅将面积 <= max_hole_area 的孔洞填充为道路
    3. 大孔洞保持原样

    Args:
        mask:          二值 mask (H, W), 0/255
        max_hole_area: 孔洞面积阈值（像素），大于此值的孔洞不填充

    Returns:
        (filled_mask, stats)
        stats 字典包含:
            filled_small_holes: 已填充的小孔洞数量
            skipped_large_holes: 跳过的大孔洞数量
    """
    h, w = mask.shape

    # 创建带边框的 padded 图像用于 flood-fill
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask

    # flood-fill 从 (0,0) 填充背景 → 背景=255，孔洞+前景=0
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 255)

    # 孔洞区域 = 原图为0 且 flood 也为0 的区域（即被道路包围的封闭区域）
    hole_region = np.zeros((h + 2, w + 2), dtype=np.uint8)
    hole_region[(padded == 0) & (flood == 0)] = 255

    # 连通域分析，找出每个独立孔洞
    num_labels, labels, stats_array, _ = cv2.connectedComponentsWithStats(
        hole_region, connectivity=8
    )

    # 构建结果：初始 = 原 mask
    result = mask.copy()

    filled_count = 0
    skipped_count = 0

    # label 0 是背景，从 1 开始遍历每个孔洞
    for i in range(1, num_labels):
        area = stats_array[i, cv2.CC_STAT_AREA]
        if area <= max_hole_area:
            # 小孔洞 → 填充为道路
            # labels 中此孔洞对应的像素位置
            hole_mask = (labels == i)
            result[hole_mask[1:-1, 1:-1]] = 255
            filled_count += 1
        else:
            # 大孔洞 → 跳过
            skipped_count += 1

    return result, {
        "filled_small_holes": filled_count,
        "skipped_large_holes": skipped_count,
    }


def analyze_mask_anomalies(
    mask: np.ndarray,
    original_mask: np.ndarray = None,
    max_road_ratio: float = 0.25,
    max_largest_ratio: float = 0.10,
    max_fill_added_ratio: float = 0.05,
) -> Dict[str, Any]:
    """
    分析 mask 是否存在大面积误填异常。

    Args:
        mask:              当前 mask (H, W), uint8
        original_mask:     原始 mask（用于计算 fill 新增面积），可选
        max_road_ratio:    道路像素占比上限
        max_largest_ratio: 最大连通域占比上限
        max_fill_added_ratio: 填洞新增面积占比上限

    Returns:
        分析结果字典:
            road_mask_area_ratio: float          — 道路占整图比例
            largest_component_area_ratio: float  — 最大连通域占整图比例
            fill_added_area_ratio: float         — 填洞新增面积比例
            is_anomalous: bool                   — 是否存在异常
            warnings: list[str]                   — 警告信息列表
    """
    total = mask.size
    road_px = int((mask > 0).sum())
    road_mask_area_ratio = road_px / total if total else 0

    # 最大连通域分析
    num_labels, labels, stats_array, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    largest_area = 0
    for i in range(1, num_labels):
        area = stats_array[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
    largest_component_area_ratio = largest_area / total if total else 0

    # 填洞新增面积
    fill_added_area_ratio = 0.0
    if original_mask is not None:
        orig_road = int((original_mask > 0).sum())
        added = max(0, road_px - orig_road)
        fill_added_area_ratio = added / total if total else 0

    # 判断异常
    warnings = []
    is_anomalous = False

    if road_mask_area_ratio > max_road_ratio:
        is_anomalous = True
        warnings.append(
            f"道路占比 {road_mask_area_ratio*100:.1f}% > {max_road_ratio*100:.0f}%（阈值）"
        )
    if largest_component_area_ratio > max_largest_ratio:
        is_anomalous = True
        warnings.append(
            f"最大连通域占比 {largest_component_area_ratio*100:.1f}% > {max_largest_ratio*100:.0f}%（阈值）"
        )
    if fill_added_area_ratio > max_fill_added_ratio and original_mask is not None:
        is_anomalous = True
        warnings.append(
            f"填洞新增面积占比 {fill_added_area_ratio*100:.1f}% > {max_fill_added_ratio*100:.0f}%（阈值）"
        )

    return {
        "road_mask_area_ratio": road_mask_area_ratio,
        "largest_component_area_ratio": largest_component_area_ratio,
        "fill_added_area_ratio": fill_added_area_ratio,
        "is_anomalous": is_anomalous,
        "warnings": warnings,
    }


def remove_small_components(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """
    移除面积小于 min_area 的连通分量。

    Args:
        mask:     二值 mask
        min_area: 最小保留面积（像素数）

    Returns:
        过滤后的 mask
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):  # label 0 是背景
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def keep_largest_n_components(mask: np.ndarray, n: int = 5) -> np.ndarray:
    """
    只保留面积最大的前 n 个连通域。

    Args:
        mask: 二值 mask
        n:    保留的连通域数量

    Returns:
        过滤后的 mask
    """
    if n <= 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # 背景 label=0 跳过，收集各连通域面积
    areas = []
    for i in range(1, num_labels):
        areas.append((i, stats[i, cv2.CC_STAT_AREA]))

    # 按面积降序排序，取前 n 个
    areas.sort(key=lambda x: x[1], reverse=True)
    keep_labels = set(lb for lb, _ in areas[:n])

    cleaned = np.zeros_like(mask)
    for lb in keep_labels:
        cleaned[labels == lb] = 255

    return cleaned


def smooth_boundary(
    mask: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    边界平滑：通过高斯模糊 + 二值化来平滑 mask 边缘。

    Args:
        mask:       二值 mask
        kernel_size: 高斯核大小（奇数）

    Returns:
        平滑后的 mask
    """
    if kernel_size <= 0:
        return mask

    # 确保核是奇数
    if kernel_size % 2 == 0:
        kernel_size += 1

    # 高斯模糊
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)

    # 二值化阈值 127（半白）
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return smoothed.astype(np.uint8)


def remove_thin_blobs(mask: np.ndarray) -> np.ndarray:
    """
    删除细长孤立误检区域。

    通过分析连通域的宽高比和面积比来判断是否为细长噪声。
    细长区域判定标准：包围盒长宽比 > 5 且填充率 < 0.3

    Args:
        mask: 二值 mask

    Returns:
        处理后的 mask
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if w == 0 or h == 0:
            cleaned[labels == i] = 255
            continue

        # 宽高比
        aspect_ratio = max(w, h) / min(w, h)

        # 填充率 = 面积 / 包围盒面积
        fill_ratio = area / (w * h)

        # 细长且稀疏 → 视为噪声
        if aspect_ratio > 5 and fill_ratio < 0.3:
            continue

        cleaned[labels == i] = 255

    return cleaned


# ===========================================================================
# 辅助函数
# ===========================================================================

def _record(
    mask: np.ndarray,
    step_name: str,
    intermediates: List[Tuple[str, np.ndarray]],
    save_intermediate: bool,
    output_dir: str,
) -> None:
    """记录中间结果（内存 + 可选文件保存）。"""
    intermediates.append((step_name, mask))
    if save_intermediate and output_dir:
        import os
        path = os.path.join(output_dir, f"postprocess_{step_name}.png")
        cv2.imwrite(path, mask)
        print(f"  [POST] 已保存: {path}")
