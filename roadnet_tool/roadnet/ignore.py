"""
Ignore 区域屏蔽模块 V2.2：交互式绘制矩形/多边形屏蔽区域。

功能：
- 左键拖拽：绘制矩形 ignore 区域（起点→终点矩形）
- p 键：切换为多边形模式（逐点点击 → Enter 闭合）
- r 键：切回矩形模式（默认）
- c 键：清空所有 ignore 区域
- u 键：撤销上一个 ignore 区域
- Enter：确认所有 ignore 区域
- Esc：取消退出（不屏蔽任何区域）

ignore_mask 中，屏蔽区域为 255（这些区域会被 AND NOT 操作清零）。
"""

import gc
import numpy as np
from typing import List, Tuple, Optional, Union


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


# 绘图模式
MODE_RECT = "rect"
MODE_POLYGON = "polygon"


class IgnoreDrawer:
    """
    基于 matplotlib 的交互式 ignore 区域绘制器。

    操作方式：
        默认矩形模式（红色半透明框）：
            左键按下拖拽 → 矩形 ignore 区域
            释放鼠标 → 确认矩形

        多边形模式（按 p 切换）：
            左键点击 → 添加多边形顶点
            Enter     → 闭合当前多边形

        r 键 → 切回矩形模式
        p 键 → 切换为多边形模式
        c 键 → 清空所有 ignore 区域
        u 键 → 撤销上一个区域
        Enter → 确认所有屏蔽区域
        Esc   → 退出（不屏蔽）

    使用方式：
        drawer = IgnoreDrawer(image_rgb, mask)
        drawer.run()
        ignore_mask = drawer.ignore_mask  # (H, W) uint8, 255=屏蔽区域
    """

    def __init__(self, image_rgb: np.ndarray, mask: np.ndarray):
        self.image = image_rgb
        self.h, self.w = image_rgb.shape[:2]
        # 输入 mask（用于叠加显示参考）
        self._input_mask = (mask > 127).astype(np.uint8) * 255

        # 已确认的 ignore 区域列表
        # 每个元素: ("rect", (x1, y1, x2, y2)) 或 ("polygon", [(x,y),...])
        self._ignore_regions: List[Tuple[str, Union[Tuple, List]]] = []

        # 当前正在拖拽的矩形（未确认）
        self._drag_start: Optional[Tuple[float, float]] = None
        self._drag_end: Optional[Tuple[float, float]] = None

        # 当前模式
        self._mode: str = MODE_RECT

        # 多边形模式下的当前顶点
        self._poly_vertices: List[Tuple[float, float]] = []

        # 计算结果
        self._ignore_mask: Optional[np.ndarray] = None

        self._cancelled: bool = False
        self._confirmed: bool = False

        # matplotlib 引用
        self._fig = None
        self._ax = None
        self._overlay_img = None
        self._plt = None
        # 已确认的 ignore 矩形 patches
        self._confirmed_rects: list = []
        # 已确认的 ignore 多边形 patches
        self._confirmed_polys: list = []
        # 当前拖拽矩形
        self._drag_rect = None
        # 当前多边形连线
        self._poly_line = None

    # ---- 属性 ----
    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    @property
    def is_confirmed(self) -> bool:
        return self._confirmed

    @property
    def ignore_mask(self) -> Optional[np.ndarray]:
        """返回 ignore 二值 mask (H,W), 255=屏蔽区域"""
        return self._ignore_mask

    # ---- 主入口 ----
    def run(self) -> None:
        """启动交互窗口，阻塞直到用户确认或取消。"""
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        self._plt = plt
        self._mpatches = mpatches

        # 配置中文字体
        _setup_cjk_font()

        fig, ax = plt.subplots(figsize=(10, 8))

        # 生成叠加图（显示原图 + 当前 mask，方便参考）
        overlay = self._make_overlay()
        self._overlay_img = ax.imshow(overlay)

        ax.set_title(
            "左键拖拽=矩形屏蔽 | p=多边形 | r=矩形 | c=清空 | u=撤销 | Enter=确认 | Esc=退出",
            fontsize=9,
        )
        ax.axis("off")

        self._fig = fig
        self._ax = ax

        # 绑定事件
        fig.canvas.mpl_connect("button_press_event", self._on_press)
        fig.canvas.mpl_connect("button_release_event", self._on_release)
        fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        fig.canvas.mpl_connect("close_event", self._on_close)

        plt.tight_layout()
        plt.show()

        # 窗口关闭后，生成 ignore_mask
        if self._confirmed:
            self._compute_ignore_mask()
        else:
            # 取消了，生成全 0 mask
            self._ignore_mask = np.zeros((self.h, self.w), dtype=np.uint8)

    # ---- 叠加显示 ----
    def _make_overlay(self) -> np.ndarray:
        """生成带有当前 mask 半透明叠加和已确认 ignore 区域的显示图像。"""
        import cv2

        overlay = self.image.copy()
        # 道路区域（绿色）
        overlay[self._input_mask > 0] = (0, 255, 0)

        # 已确认的 ignore 区域（红色半透明）
        for region in self._ignore_regions:
            rtype, data = region
            if rtype == "rect":
                x1, y1, x2, y2 = data
                x1c = max(0, int(round(x1)))
                y1c = max(0, int(round(y1)))
                x2c = min(self.w, int(round(x2)) + 1)
                y2c = min(self.h, int(round(y2)) + 1)
                overlay[y1c:y2c, x1c:x2c] = (255, 0, 0)
            elif rtype == "polygon":
                pts = np.array(
                    [(int(round(x)), int(round(y))) for (x, y) in data],
                    dtype=np.int32,
                )
                cv2.fillPoly(overlay, [pts], (255, 0, 0))

        blended = cv2.addWeighted(self.image, 0.55, overlay, 0.45, 0)
        return blended

    def _redraw_confirmed(self) -> None:
        """重绘已确认的 ignore 区域 patches（matplotlib 矩形/多边形）。"""
        import matplotlib.patches as mpatches

        # 清除旧 patches
        for p in self._confirmed_rects:
            p.remove()
        self._confirmed_rects.clear()
        for p in self._confirmed_polys:
            p.remove()
        self._confirmed_polys.clear()

        for region in self._ignore_regions:
            rtype, data = region
            if rtype == "rect":
                x1, y1, x2, y2 = data
                w_rect = x2 - x1
                h_rect = y2 - y1
                rect = mpatches.Rectangle(
                    (x1, y1), w_rect, h_rect,
                    linewidth=2, edgecolor="red", facecolor="red",
                    alpha=0.3,
                )
                self._ax.add_patch(rect)
                self._confirmed_rects.append(rect)
            elif rtype == "polygon":
                poly = mpatches.Polygon(
                    data, closed=True,
                    linewidth=2, edgecolor="red", facecolor="red",
                    alpha=0.3,
                )
                self._ax.add_patch(poly)
                self._confirmed_polys.append(poly)

        if self._drag_rect is not None:
            self._drag_rect.remove()
            self._drag_rect = None
        if self._poly_line is not None:
            self._poly_line.remove()
            self._poly_line = None

    # ---- 事件回调 ----
    def _on_press(self, event) -> None:
        """鼠标按下。"""
        if event.inaxes != self._ax:
            return
        if event.button != 1:
            return

        if self._mode == MODE_RECT:
            # 矩形模式：记录拖拽起点
            self._drag_start = (event.xdata, event.ydata)
            self._drag_end = (event.xdata, event.ydata)
        elif self._mode == MODE_POLYGON:
            # 多边形模式：不在这里处理，在 _on_release 中处理
            pass

    def _on_release(self, event) -> None:
        """鼠标释放。"""
        if event.inaxes != self._ax:
            return
        if event.button != 1:
            return

        if self._mode == MODE_RECT and self._drag_start is not None:
            # 矩形模式：确认拖拽矩形
            x1 = min(self._drag_start[0], event.xdata)
            y1 = min(self._drag_start[1], event.ydata)
            x2 = max(self._drag_start[0], event.xdata)
            y2 = max(self._drag_start[1], event.ydata)

            # 最小矩形尺寸检查（至少 5 像素）
            if abs(x2 - x1) >= 5 and abs(y2 - y1) >= 5:
                self._ignore_regions.append(("rect", (x1, y1, x2, y2)))

            self._drag_start = None
            self._drag_end = None

            # 清除拖拽矩形
            if self._drag_rect is not None:
                self._drag_rect.remove()
                self._drag_rect = None

            self._redraw_confirmed()
            self._fig.canvas.draw_idle()

        elif self._mode == MODE_POLYGON:
            # 多边形模式：记录点击（在 press 中也触发，避免重复）
            # 这里用 press 处理点击，release 忽略
            pass

    def _on_motion(self, event) -> None:
        """鼠标移动：更新拖拽预览。"""
        import matplotlib.patches as mpatches

        if event.inaxes != self._ax:
            return

        if self._mode == MODE_RECT and self._drag_start is not None:
            self._drag_end = (event.xdata, event.ydata)
            x1 = min(self._drag_start[0], self._drag_end[0])
            y1 = min(self._drag_start[1], self._drag_end[1])
            w = abs(self._drag_end[0] - self._drag_start[0])
            h = abs(self._drag_end[1] - self._drag_start[1])

            # 更新或创建拖拽矩形
            if self._drag_rect is None:
                self._drag_rect = mpatches.Rectangle(
                    (x1, y1), w, h,
                    linewidth=2, edgecolor="yellow", facecolor="yellow",
                    alpha=0.25,
                )
                self._ax.add_patch(self._drag_rect)
            else:
                self._drag_rect.set_xy((x1, y1))
                self._drag_rect.set_width(w)
                self._drag_rect.set_height(h)

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
            # 清空所有 ignore 区域
            self._ignore_regions.clear()
            self._poly_vertices.clear()
            self._drag_start = None
            self._drag_end = None
            if self._drag_rect is not None:
                self._drag_rect.remove()
                self._drag_rect = None
            if self._poly_line is not None:
                self._poly_line.remove()
                self._poly_line = None
            self._redraw_confirmed()
            self._fig.canvas.draw_idle()
            print("[IGNORE] 已清空所有屏蔽区域。")
        elif event.key == "u":
            # 撤销上一个区域
            if self._ignore_regions:
                removed = self._ignore_regions.pop()
                rtype = removed[0]
                print(f"[IGNORE] 已撤销一个 {rtype} 屏蔽区域（剩余 {len(self._ignore_regions)} 个）。")
            self._redraw_confirmed()
            self._fig.canvas.draw_idle()
        elif event.key == "r":
            # 切回矩形模式
            self._mode = MODE_RECT
            self._poly_vertices.clear()
            if self._poly_line is not None:
                self._poly_line.remove()
                self._poly_line = None
            self._fig.canvas.draw_idle()
            print("[IGNORE] 切换到矩形模式（左键拖拽绘制屏蔽矩形）。")
        elif event.key == "p":
            # 切换到多边形模式
            self._mode = MODE_POLYGON
            self._drag_start = None
            self._drag_end = None
            if self._drag_rect is not None:
                self._drag_rect.remove()
                self._drag_rect = None
            self._poly_vertices.clear()
            self._fig.canvas.draw_idle()
            print("[IGNORE] 切换到多边形模式（左键点击顶点，Enter 闭合）。")

    def _on_close(self, event) -> None:
        """窗口关闭。"""
        if not self._confirmed:
            self._cancelled = True

    # ---- mask 生成 ----
    def _compute_ignore_mask(self) -> None:
        """根据已记录的所有 ignore 区域生成组合 ignore mask。"""
        import cv2

        mask = np.zeros((self.h, self.w), dtype=np.uint8)

        for region in self._ignore_regions:
            rtype, data = region
            if rtype == "rect":
                x1, y1, x2, y2 = data
                x1c = max(0, int(round(x1)))
                y1c = max(0, int(round(y1)))
                x2c = min(self.w, int(round(x2)) + 1)
                y2c = min(self.h, int(round(y2)) + 1)
                mask[y1c:y2c, x1c:x2c] = 255
            elif rtype == "polygon":
                pts = np.array(
                    [(int(round(x)), int(round(y))) for (x, y) in data],
                    dtype=np.int32,
                )
                cv2.fillPoly(mask, [pts], 255)

        self._ignore_mask = mask
        print(f"[IGNORE] 已生成 ignore mask（屏蔽区域数: {len(self._ignore_regions)}）。")
