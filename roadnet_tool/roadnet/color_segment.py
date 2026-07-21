"""
颜色分割模块 V1.5：正负样本交互采样 + 颜色距离约束的道路分割。

核心改进：
- 支持正样本（道路）和负样本（非道路）分别采样
- 在 Lab 色彩空间中计算像素到正/负样本集的距离，做精细化排除
- Combined 模式默认 AND，避免误检膨胀
"""

import gc
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


def _force_close_window(plt_module, fig):
    """强制关闭 matplotlib 窗口（TkAgg 等后端兼容）。"""
    try:
        plt_module.close(fig)
    except Exception:
        pass
    try:
        plt_module.close("all")
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass


def _setup_cjk_font() -> None:
    """配置 matplotlib 中文字体，消除 CJK glyph missing 警告。"""
    import matplotlib
    # Windows 常见中文字体，按优先级尝试
    _cjk_candidates = [
        "Microsoft YaHei", "SimHei", "SimSun",
        "FangSong", "KaiTi", "Noto Sans CJK SC",
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "sans-serif",
    ]
    for _font in _cjk_candidates:
        try:
            matplotlib.rcParams["font.family"] = _font
            # 验证字体是否可用
            from matplotlib.font_manager import FontProperties
            FontProperties(family=_font)
            break
        except Exception:
            continue


# ===========================================================================
# 交互采样器（matplotlib 窗口）
# ===========================================================================

class ColorSampler:
    """
    基于 matplotlib 的交互采样器 V1.5。

    操作方式：
        左键点击  → 添加道路正样本（绿色圆圈）
        右键点击  → 添加非道路负样本（红色叉号）
        c        → 清空所有样本点
        u        → 撤销上一个样本点
        Enter    → 确认并开始分割
        Esc      → 取消退出

    使用方式：
        sampler = ColorSampler(image_rgb, sample_radius=3)
        sampler.run()
        pos_samples = sampler.positive_samples   # (N, 3) numpy 数组
        neg_samples = sampler.negative_samples   # (M, 3) numpy 数组
    """

    def __init__(self, image_rgb: np.ndarray, sample_radius: int = 3):
        self.image = image_rgb
        self.radius = sample_radius

        # 正/负采样点列表：存储 matplolib 坐标 (x, y)
        self._pos_points: List[Tuple[float, float]] = []
        self._neg_points: List[Tuple[float, float]] = []

        # 计算结果
        self._pos_samples: np.ndarray = np.array([])
        self._neg_samples: np.ndarray = np.array([])

        self._cancelled: bool = False
        self._confirmed: bool = False

        # matplotlib 对象引用
        self._fig = None
        self._ax = None
        self._pos_scatter = None
        self._neg_scatter = None
        self._plt = None

    # ---- 属性 ----
    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    @property
    def positive_samples(self) -> np.ndarray:
        """正样本颜色均值 (N, 3), float32"""
        return self._pos_samples

    @property
    def negative_samples(self) -> np.ndarray:
        """负样本颜色均值 (M, 3), float32"""
        return self._neg_samples

    @property
    def pos_pixel_points(self) -> List[Tuple[int, int]]:
        """正样本像素坐标 [(col, row), ...]"""
        return [(int(round(x)), int(round(y))) for (x, y) in self._pos_points]

    @property
    def neg_pixel_points(self) -> List[Tuple[int, int]]:
        """负样本像素坐标 [(col, row), ...]"""
        return [(int(round(x)), int(round(y))) for (x, y) in self._neg_points]

    def has_samples(self) -> bool:
        """是否有至少一个正样本"""
        return len(self._pos_samples) > 0

    # ---- 主入口 ----
    def run(self) -> None:
        """启动交互窗口，阻塞直到用户确认或取消。"""
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        self._plt = plt

        # 配置中文字体，避免 CJK glyph missing 警告
        _setup_cjk_font()

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(self.image)
        title = (
            "左键=道路(绿) | 右键=非道路(红) | c=清空 | u=撤销 | Enter=确认 | Esc=退出"
        )
        ax.set_title(title, fontsize=9)
        ax.axis("off")

        # 正样本散点（绿色空心圆）
        (pos_sc,) = ax.plot([], [], "o", color="lime", markersize=9,
                            markerfacecolor="none", markeredgewidth=2.5, label="正样本(道路)")
        # 负样本散点（红色叉号）
        (neg_sc,) = ax.plot([], [], "x", color="red", markersize=9,
                            markeredgewidth=2.5, label="负样本(非道路)")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.7)

        self._fig = fig
        self._ax = ax
        self._pos_scatter = pos_sc
        self._neg_scatter = neg_sc

        # 绑定事件
        fig.canvas.mpl_connect("button_press_event", self._on_click)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        fig.canvas.mpl_connect("close_event", self._on_close)

        plt.tight_layout()
        plt.show()  # 阻塞

        self._compute_all_samples()

    # ---- 事件回调 ----
    def _on_click(self, event) -> None:
        """鼠标事件：左键→正样本，右键→负样本。"""
        if event.inaxes != self._ax:
            return
        x, y = event.xdata, event.ydata

        if event.button == 1:  # 左键：正样本
            self._pos_points.append((x, y))
            xs, ys = zip(*self._pos_points) if self._pos_points else ([], [])
            self._pos_scatter.set_data(xs, ys)
        elif event.button == 3:  # 右键：负样本
            self._neg_points.append((x, y))
            xs, ys = zip(*self._neg_points) if self._neg_points else ([], [])
            self._neg_scatter.set_data(xs, ys)
        else:
            return

        self._fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        """键盘回调。"""
        if event.key == "enter":
            self._confirmed = True
            self._plt.close("all")
        elif event.key in ("escape", "esc", "\x1b"):
            self._cancelled = True
            _force_close_window(self._plt, self._fig)
        elif event.key == "c":
            # 清空所有样本点
            self._pos_points.clear()
            self._neg_points.clear()
            self._pos_scatter.set_data([], [])
            self._neg_scatter.set_data([], [])
            self._fig.canvas.draw_idle()
        elif event.key == "u":
            # 撤销最后一个样本点（先撤负样本，再撤正样本）
            if self._neg_points:
                self._neg_points.pop()
                xs, ys = zip(*self._neg_points) if self._neg_points else ([], [])
                self._neg_scatter.set_data(xs, ys)
            elif self._pos_points:
                self._pos_points.pop()
                xs, ys = zip(*self._pos_points) if self._pos_points else ([], [])
                self._pos_scatter.set_data(xs, ys)
            self._fig.canvas.draw_idle()

    def _on_close(self, event) -> None:
        """窗口关闭按钮。"""
        if not self._confirmed:
            self._cancelled = True

    # ---- 采样点计算 ----
    def _compute_all_samples(self) -> None:
        """从记录的坐标点计算邻域颜色均值。"""
        self._pos_samples = self._extract_colors(self._pos_points)
        self._neg_samples = self._extract_colors(self._neg_points)

    def _extract_colors(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """从坐标列表中提取邻域颜色均值 (K, 3)。"""
        if points is None or len(points) == 0:
            return np.array([])

        h, w = self.image.shape[:2]
        r = self.radius
        values = []

        for (mx, my) in points:
            col, row = int(round(mx)), int(round(my))
            c1 = max(0, col - r)
            c2 = min(w, col + r + 1)
            r1 = max(0, row - r)
            r2 = min(h, row + r + 1)
            patch = self.image[r1:r2, c1:c2]
            mean_rgb = patch.mean(axis=(0, 1))
            values.append(mean_rgb)

        return np.array(values, dtype=np.float32) if values else np.array([])

    # ---- 可视化辅助 ----
    def get_sample_points_image(self) -> np.ndarray:
        """
        生成一张标注了所有采样点的 RGB 图像（用于保存 sample_points.png）。
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(self.image)
        ax.set_title("Sample Points (green=road, red=non-road)")
        ax.axis("off")

        if self._pos_points:
            xs, ys = zip(*self._pos_points)
            ax.plot(xs, ys, "o", color="lime", markersize=9,
                    markerfacecolor="none", markeredgewidth=2.5, label="Positive")
        if self._neg_points:
            xs, ys = zip(*self._neg_points)
            ax.plot(xs, ys, "x", color="red", markersize=9,
                    markeredgewidth=2.5, label="Negative")
        if self._pos_points or self._neg_points:
            ax.legend(loc="lower right", fontsize=8)

        plt.tight_layout()
        fig.canvas.draw()
        # 将 matplotlib figure 转为 numpy RGB 数组
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return data


# ===========================================================================
# 颜色空间转换工具
# ===========================================================================

def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 → HSV (OpenCV 格式：H:[0,180], S:[0,255], V:[0,255])"""
    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 → Lab (L:[0,255], a:[0,255], b:[0,255])"""
    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)


# ===========================================================================
# 阈值估计
# ===========================================================================

def _estimate_bounds(
    samples: np.ndarray,
    margins: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据采样点均值和标准差估计阈值上下界。

    lower = mean - margin - std
    upper = mean + margin + std
    """
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    lower = mean - np.array(margins) - std
    upper = mean + np.array(margins) + std
    return lower, upper


# ===========================================================================
# 正负样本距离约束（核心新增功能）
# ===========================================================================

def _lab_distance_map(image_rgb: np.ndarray, ref_lab: np.ndarray) -> np.ndarray:
    """
    计算图像每个像素到参考 Lab 颜色集的最小欧氏距离。

    Args:
        image_rgb:  原始 RGB 图像 (H, W, 3)
        ref_lab:    参考 Lab 颜色集 (K, 3)

    Returns:
        距离图 (H, W)，每个元素是该像素到 ref_lab 中最近颜色的距离
    """
    if len(ref_lab) == 0:
        return np.full(image_rgb.shape[:2], np.inf, dtype=np.float32)

    image_lab = rgb_to_lab(image_rgb).astype(np.float32)  # (H, W, 3)
    ref = ref_lab.astype(np.float32)                       # (K, 3)

    # 对每个参考颜色计算整图距离，取逐像素最小值
    h, w = image_rgb.shape[:2]
    min_dist = np.full((h, w), np.inf, dtype=np.float32)

    for k in range(len(ref)):
        diff = image_lab - ref[k]                          # (H, W, 3)
        dist = np.linalg.norm(diff, axis=2)                # (H, W)
        min_dist = np.minimum(min_dist, dist)

    return min_dist


def apply_positive_negative_constraint(
    image_rgb: np.ndarray,
    pos_samples_rgb: np.ndarray,
    neg_samples_rgb: np.ndarray,
    pos_threshold: float = 22,
    neg_margin: float = 5,
) -> np.ndarray:
    """
    基于正负样本 Lab 距离的约束 mask。

    规则：
        1. 计算每个像素到正样本集的最小 Lab 距离 d_pos
        2. 计算每个像素到负样本集的最小 Lab 距离 d_neg
        3. 如果 d_neg <= neg_margin（非常接近负样本颜色）→ 排除
        4. 如果 d_pos > pos_threshold（距离正样本太远）→ 排除
        5. 如果 d_pos < d_neg（更接近正样本）→ 保留

    Returns:
        二值 mask (H, W), dtype uint8, 255=保留, 0=排除
    """
    h, w = image_rgb.shape[:2]

    # 没有负样本时只做正样本距离约束
    if len(neg_samples_rgb) == 0:
        if len(pos_samples_rgb) == 0:
            return np.zeros((h, w), dtype=np.uint8)
        pos_lab = rgb_to_lab(pos_samples_rgb.reshape(1, -1, 3).astype(np.uint8)).reshape(-1, 3)
        d_pos = _lab_distance_map(image_rgb, pos_lab)
        mask = (d_pos <= pos_threshold).astype(np.uint8) * 255
        return mask

    # 转换正负样本到 Lab
    pos_lab = rgb_to_lab(pos_samples_rgb.reshape(1, -1, 3).astype(np.uint8)).reshape(-1, 3)
    neg_lab = rgb_to_lab(neg_samples_rgb.reshape(1, -1, 3).astype(np.uint8)).reshape(-1, 3)

    d_pos = _lab_distance_map(image_rgb, pos_lab)
    d_neg = _lab_distance_map(image_rgb, neg_lab)

    # 组合约束
    mask = np.zeros((h, w), dtype=np.uint8)

    # 条件1：负样本排除区（太接近负样本颜色直接排除）
    neg_exclude = d_neg <= neg_margin

    # 条件2：正样本有效区（距离正样本不太远）
    pos_valid = d_pos <= pos_threshold

    # 条件3：更接近正样本而非负样本
    closer_to_pos = d_pos < d_neg

    # 最终保留：在正样本有效区内、且不在负样本排除区内、且更接近正样本
    mask = (pos_valid & (~neg_exclude) & closer_to_pos).astype(np.uint8) * 255
    return mask


# ===========================================================================
# 各模式分割函数
# ===========================================================================

def segment_hsv(
    image_rgb: np.ndarray,
    pos_samples_rgb: np.ndarray,
    h_margin: float = 6,
    s_margin: float = 25,
    v_margin: float = 30,
) -> np.ndarray:
    """
    HSV 色彩空间阈值分割（基于正样本）。
    返回二值 mask (H,W), 255=道路。
    """
    if len(pos_samples_rgb) == 0:
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    samples_hsv = rgb_to_hsv(pos_samples_rgb.reshape(1, -1, 3).astype(np.uint8))
    samples_hsv = samples_hsv.reshape(-1, 3).astype(np.float32)

    lower, upper = _estimate_bounds(samples_hsv, (h_margin, s_margin, v_margin))
    lower = np.clip(lower, [0, 0, 0], [179, 255, 255])
    upper = np.clip(upper, [0, 0, 0], [179, 255, 255])

    image_hsv = rgb_to_hsv(image_rgb)
    mask = cv2.inRange(image_hsv, lower.astype(np.uint8), upper.astype(np.uint8))
    return mask


def segment_lab(
    image_rgb: np.ndarray,
    pos_samples_rgb: np.ndarray,
    lab_margin: float = 12,
) -> np.ndarray:
    """
    Lab 色彩空间阈值分割（基于正样本）。
    返回二值 mask (H,W), 255=道路。
    """
    if len(pos_samples_rgb) == 0:
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    samples_lab = rgb_to_lab(pos_samples_rgb.reshape(1, -1, 3).astype(np.uint8))
    samples_lab = samples_lab.reshape(-1, 3).astype(np.float32)

    lower, upper = _estimate_bounds(samples_lab, (lab_margin, lab_margin, lab_margin))
    lower = np.clip(lower, [0, 0, 0], [255, 255, 255])
    upper = np.clip(upper, [0, 0, 0], [255, 255, 255])

    image_lab = rgb_to_lab(image_rgb)
    mask = cv2.inRange(image_lab, lower.astype(np.uint8), upper.astype(np.uint8))
    return mask


def segment_combined(
    image_rgb: np.ndarray,
    pos_samples_rgb: np.ndarray,
    neg_samples_rgb: np.ndarray,
    combine_method: str = "and",
    h_margin: float = 6,
    s_margin: float = 25,
    v_margin: float = 30,
    lab_margin: float = 12,
    use_negative: bool = True,
    pos_threshold: float = 22,
    neg_exclude_margin: float = 5,
) -> np.ndarray:
    """
    Combined 模式：组合 HSV 阈值、Lab 阈值、正负样本距离约束。

    combine_method:
        "and" → HSV mask AND Lab mask AND 距离约束（默认，更严格）
        "or"  → HSV mask OR  Lab mask AND 距离约束

    返回二值 mask (H,W), 255=道路。
    """
    h, w = image_rgb.shape[:2]

    if len(pos_samples_rgb) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    mask_hsv = segment_hsv(image_rgb, pos_samples_rgb, h_margin, s_margin, v_margin)
    mask_lab = segment_lab(image_rgb, pos_samples_rgb, lab_margin)

    # 合并 HSV + Lab
    if combine_method == "and":
        base_mask = cv2.bitwise_and(mask_hsv, mask_lab)
    else:
        base_mask = cv2.bitwise_or(mask_hsv, mask_lab)

    # 正负样本距离约束
    if use_negative:
        dist_mask = apply_positive_negative_constraint(
            image_rgb, pos_samples_rgb, neg_samples_rgb,
            pos_threshold=pos_threshold, neg_margin=neg_exclude_margin,
        )
        # AND：必须同时满足颜色阈值和距离约束
        final_mask = cv2.bitwise_and(base_mask, dist_mask)
    else:
        final_mask = base_mask

    return final_mask


# ===========================================================================
# 统一入口
# ===========================================================================

def segment_road(
    image_rgb: np.ndarray,
    pos_samples_rgb: np.ndarray,
    neg_samples_rgb: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    """
    根据配置执行道路分割（V1.5）。

    Args:
        image_rgb:        原始 RGB 图像 (H, W, 3)
        pos_samples_rgb:  正样本（道路）RGB 颜色 (N, 3)
        neg_samples_rgb:  负样本（非道路）RGB 颜色 (M, 3)
        config:           配置字典中的 segment 项

    Returns:
        二值 mask (H, W), dtype uint8, 255=道路, 0=非道路
    """
    mode = config.get("mode", "combined")
    combine_method = config.get("combine_method", "and")
    h_margin = config.get("h_margin", 6)
    s_margin = config.get("s_margin", 25)
    v_margin = config.get("v_margin", 30)
    lab_margin = config.get("lab_margin", 12)
    use_negative = config.get("use_negative_samples", True)
    pos_threshold = config.get("positive_distance_threshold", 22)
    neg_margin_cfg = config.get("negative_margin", 5)

    if mode == "hsv":
        mask = segment_hsv(image_rgb, pos_samples_rgb, h_margin, s_margin, v_margin)
        if use_negative and len(neg_samples_rgb) > 0:
            dist_mask = apply_positive_negative_constraint(
                image_rgb, pos_samples_rgb, neg_samples_rgb,
                pos_threshold=pos_threshold, neg_margin=neg_margin_cfg,
            )
            mask = cv2.bitwise_and(mask, dist_mask)

    elif mode == "lab":
        mask = segment_lab(image_rgb, pos_samples_rgb, lab_margin)
        if use_negative and len(neg_samples_rgb) > 0:
            dist_mask = apply_positive_negative_constraint(
                image_rgb, pos_samples_rgb, neg_samples_rgb,
                pos_threshold=pos_threshold, neg_margin=neg_margin_cfg,
            )
            mask = cv2.bitwise_and(mask, dist_mask)

    elif mode == "combined":
        mask = segment_combined(
            image_rgb, pos_samples_rgb, neg_samples_rgb,
            combine_method=combine_method,
            h_margin=h_margin, s_margin=s_margin, v_margin=v_margin,
            lab_margin=lab_margin,
            use_negative=use_negative,
            pos_threshold=pos_threshold,
            neg_exclude_margin=neg_margin_cfg,
        )

    else:
        raise ValueError(f"不支持的分割模式: {mode}，可选值: hsv / lab / combined")

    return mask
