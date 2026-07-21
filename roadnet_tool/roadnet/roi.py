"""
ROI 区域绘制模块 V2.3：交互式多区域多边形 ROI 绘制 + mask 生成。

功能：
- 左键点击：添加当前 ROI 多边形顶点
- Enter：闭合当前多边形（支持多区域）
- c：清空所有 ROI 区域
- u：撤销上一个 ROI 区域
- Esc：取消退出

返回二值 roi_mask（所有 ROI 区域并集，区域内为 255）。
"""

import gc
import numpy as np
from typing import List, Tuple, Optional
from matplotlib.patches import Polygon as MplPolygon


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
    _cjk_candidates = [
        "Microsoft YaHei", "SimHei", "SimSun",
        "FangSong", "KaiTi", "sans-serif",
    ]
    for _font in _cjk_candidates:
        try:
            matplotlib.rcParams["font.family"] = _font
            from matplotlib.font_manager import FontProperties
            FontProperties(family=_font)
            break
        except Exception:
            continue


class RoiDrawer:
    """
    基于 matplotlib 的交互式多区域 ROI 多边形绘制器。

    操作方式：
        左键点击  → 添加当前多边形顶点
        Enter     → 闭合当前多边形（至少3个顶点），可继续绘制下一个
                      再次按 Enter（无活跃顶点时）确认所有区域并退出
        c         → 清空所有 ROI 区域
        u         → 撤销上一个 ROI 区域
        Esc       → 取消退出

    使用方式：
        drawer = RoiDrawer(image_rgb)
        drawer.run()
        roi_mask = drawer.roi_mask  # (H, W) uint8, 255=ROI内

    属性：
        roi_regions: 所有已确认的多边形区域，每个区域为顶点列表 [(x,y), ...]
        vertices:    兼容旧API，返回所有区域的合并顶点列表
    """

    # 已完成区域填充色 (RGBA, 半透明红)
    _FILL_COLOR = (1.0, 0.0, 0.0, 0.25)
    # 当前正在绘制的多边形边框色
    _EDGE_COLOR = "#ff0000"

    def __init__(self, image_rgb: np.ndarray):
        self.image = image_rgb
        self.h, self.w = image_rgb.shape[:2]

        # 已确认的多边形区域列表: List[List[Tuple[float,float]]]
        self._roi_regions: List[List[Tuple[float, float]]] = []

        # 当前正在绘制的多边形顶点
        self._current_vertices: List[Tuple[float, float]] = []

        # 计算结果
        self._roi_mask: Optional[np.ndarray] = None

        self._cancelled: bool = False
        self._confirmed: bool = False

        # matplotlib 对象引用
        self._fig = None
        self._ax = None
        self._poly_line = None  # 当前多边形连线 Line2D
        self._plt = None
        self._patches: List[MplPolygon] = []  # 已完成区域的填充 Patch

    # ---- 属性 ----
    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    @property
    def is_confirmed(self) -> bool:
        return self._confirmed

    @property
    def roi_mask(self) -> Optional[np.ndarray]:
        """返回 ROI 二值 mask (H, W), uint8, 255=ROI内（所有区域并集）"""
        return self._roi_mask

    @property
    def roi_regions(self) -> List[List[Tuple[float, float]]]:
        """返回所有 ROI 区域的顶点列表"""
        return self._roi_regions

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        """兼容旧 API：返回所有区域的合并顶点列表（主要用于可视化边界绘制）"""
        result = []
        for region in self._roi_regions:
            result.extend(region)
        result.extend(self._current_vertices)
        return result

    def region_count(self) -> int:
        """返回当前已确认的区域数量。"""
        return len(self._roi_regions)

    # ---- 主入口 ----
    def run(self) -> None:
        """启动交互窗口，阻塞直到用户确认或取消。"""
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        self._plt = plt

        # 配置中文字体
        _setup_cjk_font()

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(self.image)

        title = (
            "左键=添加顶点 | Enter=闭合当前区域(再按Enter确认全部) | c=清空 | u=撤销 | Esc=退出"
        )
        count = self.region_count()
        if count > 0:
            title += f"  [已绘制 {count} 个区域]"
        ax.set_title(title, fontsize=9)
        ax.axis("off")

        # 当前多边形连线（空初始）
        (poly_line,) = ax.plot([], [], "r-", linewidth=2, marker="o",
                               markersize=6, markerfacecolor="red")

        self._fig = fig
        self._ax = ax
        self._poly_line = poly_line

        # 重绘已有区域
        self._redraw_regions()

        # 绑定事件
        fig.canvas.mpl_connect("button_press_event", self._on_click)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        fig.canvas.mpl_connect("close_event", self._on_close)

        plt.tight_layout()
        plt.show()

        # 窗口关闭后，如果已确认则生成 roi_mask
        if self._confirmed:
            if len(self._roi_regions) == 0 and len(self._current_vertices) >= 3:
                # 用户在 Enter 关闭窗口前还有一个未闭合的多边形
                self._roi_regions.append(list(self._current_vertices))
            if len(self._roi_regions) > 0:
                self._compute_roi_mask()

    # ---- 事件回调 ----
    def _on_click(self, event) -> None:
        """左键添加顶点到当前多边形。"""
        if event.inaxes != self._ax:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata
        self._current_vertices.append((x, y))
        self._redraw_polygon()

    def _on_key(self, event) -> None:
        """键盘回调。"""
        if event.key == "enter":
            # 如果有正在绘制的多边形（>=3顶点），先闭合它
            if len(self._current_vertices) >= 3:
                self._roi_regions.append(list(self._current_vertices))
                self._current_vertices.clear()
                self._poly_line.set_data([], [])
                self._redraw_regions()
                self._update_title()
                self._fig.canvas.draw_idle()
                print(f"[ROI] 已添加第 {self.region_count()} 个区域，"
                      f"可继续绘制更多区域，或按 Enter 确认全部。")
            elif len(self._current_vertices) > 0:
                print("[ROI] 当前多边形顶点不足3个，无法闭合，请继续点击添加顶点。")
            elif self.region_count() > 0:
                # 无活跃多边形且有已确认区域 → 确认全部
                self._confirmed = True
                self._plt.close("all")
            else:
                print("[ROI] 请至少绘制一个区域（>=3个顶点）再按 Enter 确认。")

        elif event.key in ("escape", "esc", "\x1b"):
            self._cancelled = True
            _force_close_window(self._plt, self._fig)

        elif event.key == "c":
            self._current_vertices.clear()
            self._roi_regions.clear()
            self._poly_line.set_data([], [])
            # 移除所有 patch
            for p in self._patches:
                p.remove()
            self._patches.clear()
            self._update_title()
            self._fig.canvas.draw_idle()
            print("[ROI] 已清空所有区域。")

        elif event.key == "u":
            if self._current_vertices:
                # 撤销当前多边形的最后一个顶点
                self._current_vertices.pop()
                if self._current_vertices:
                    self._redraw_polygon()
                else:
                    self._poly_line.set_data([], [])
                    self._fig.canvas.draw_idle()
                print(f"[ROI] 已撤销当前多边形的顶点，剩余 {len(self._current_vertices)} 个。")
            elif self._patches:
                # 撤销上一个已确认的区域
                removed = self._roi_regions.pop()
                patch_to_remove = self._patches.pop()
                patch_to_remove.remove()
                self._update_title()
                self._fig.canvas.draw_idle()
                print(f"[ROI] 已撤销第 {self.region_count() + 1} 个区域"
                      f"（{len(removed)} 个顶点），剩余 {self.region_count()} 个区域。")
            else:
                print("[ROI] 没有可撤销的内容。")

    def _on_close(self, event) -> None:
        """窗口关闭按钮。"""
        if not self._confirmed:
            self._cancelled = True

    # ---- 绘制辅助 ----
    def _redraw_polygon(self) -> None:
        """更新当前多边形连线显示。"""
        if len(self._current_vertices) < 1:
            return
        xs, ys = zip(*self._current_vertices)
        xs_closed = list(xs) + [xs[0]]
        ys_closed = list(ys) + [ys[0]]
        self._poly_line.set_data(xs_closed, ys_closed)
        self._fig.canvas.draw_idle()

    def _redraw_regions(self) -> None:
        """重绘所有已完成区域的半透明填充。"""
        for region in self._roi_regions[len(self._patches):]:
            if len(region) < 3:
                continue
            patch = MplPolygon(
                region,
                closed=True,
                facecolor=self._FILL_COLOR,
                edgecolor=self._EDGE_COLOR,
                linewidth=2,
                linestyle="-",
            )
            self._ax.add_patch(patch)
            self._patches.append(patch)

    def _update_title(self) -> None:
        """更新窗口标题，显示区域数量。"""
        title = (
            "左键=添加顶点 | Enter=闭合当前区域(再按Enter确认全部) | c=清空 | u=撤销 | Esc=退出"
        )
        count = self.region_count()
        if count > 0:
            title += f"  [已绘制 {count} 个区域]"
        self._ax.set_title(title, fontsize=9)

    # ---- mask 生成 ----
    def _compute_roi_mask(self) -> None:
        """
        将所有 matplolib 坐标的多边形顶点转为像素坐标，
        生成二值 ROI mask（所有区域并集）。
        """
        import cv2

        mask = np.zeros((self.h, self.w), dtype=np.uint8)

        for region in self._roi_regions:
            if len(region) < 3:
                continue
            # matplotlib 坐标 (x,y) → 像素坐标 (col, row)
            pixel_pts = [(int(round(x)), int(round(y))) for (x, y) in region]
            pts_array = np.array(pixel_pts, dtype=np.int32)
            cv2.fillPoly(mask, [pts_array], 255)

        self._roi_mask = mask


def load_roi_mask(roi_path: str) -> np.ndarray:
    """
    从文件加载 ROI mask（用于跳过 ROI 绘制直接使用已有 mask）。

    Args:
        roi_path: ROI mask 文件路径

    Returns:
        二值 mask (H, W), dtype uint8
    """
    import cv2
    import os

    if not os.path.isfile(roi_path):
        raise FileNotFoundError(f"ROI mask 文件不存在: {roi_path}")

    mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取 ROI mask: {roi_path}")
    return mask
