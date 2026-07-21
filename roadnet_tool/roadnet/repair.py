"""
道路折线补线模块 V2.4：交互式 polyline 补线编辑器。

功能：
- 左键点击：添加道路中心点
- Enter：闭合折线，按 road_width 画成道路区域加入 mask
- u：撤销上一个点
- c：清空当前折线
- s：保存当前补线结果到 mask
- Esc：退出

用途：补充漏检道路、断裂道路、细路缺口。
"""

import gc
import numpy as np
from typing import List, Tuple, Optional
import cv2


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
    # 确保事件循环退出（TkAgg 有时需要额外触发）
    try:
        gc.collect()
    except Exception:
        pass


def _setup_cjk_font() -> None:
    """配置 matplotlib 中文字体。"""
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


class PolylineRepairEditor:
    """
    基于 matplotlib 的交互式道路折线补线编辑器。

    操作方式：
        左键点击  → 添加道路中心点
        Enter     → 将当前点序列按 road_width 画为道路区域加入 mask
        u         → 撤销当前折线的最后一个点
        c         → 清空当前折线
        s         → 保存补线结果
        Esc       → 退出

    使用方式：
        editor = PolylineRepairEditor(image_rgb, mask, road_width=25)
        editor.run()
        repaired_mask = editor.repaired_mask
    """

    def __init__(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        road_width: int = 25,
    ):
        self.image = image_rgb
        self.h, self.w = image_rgb.shape[:2]
        self._mask = (mask > 127).astype(np.uint8) * 255
        self.road_width = road_width

        # 当前折线的顶点 (matplotlib 坐标 x, y)
        self._vertices: List[Tuple[float, float]] = []

        # 已补线的折线列表：每条是 (vertices, road_width)
        self._polylines: List[Tuple[List[Tuple[float, float]], int]] = []

        self._modified: bool = False

        # matplotlib 引用
        self._fig = None
        self._ax = None
        self._overlay_img = None
        self._poly_line = None
        self._plt = None

    # ---- 属性 ----
    @property
    def repaired_mask(self) -> np.ndarray:
        """返回补线后的最终 mask"""
        return self._mask.copy()

    @property
    def is_modified(self) -> bool:
        return self._modified

    # ---- 主入口 ----
    def run(self) -> None:
        """启动交互窗口，阻塞直到用户退出。"""
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        self._plt = plt

        _setup_cjk_font()

        fig, ax = plt.subplots(figsize=(10, 8))

        # 显示叠加图
        overlay = self._make_overlay()
        self._overlay_img = ax.imshow(overlay)

        title = (
            f"折线补线 | 左键=添加中心点 | Enter=画道路(宽度={self.road_width}) | "
            "+/-=调宽度 | c=清空当前折线 | u=撤销点 | s=保存结果 | Esc=退出"
        )
        ax.set_title(title, fontsize=9)
        ax.axis("off")

        # 当前折线连线（初始为空）
        (poly_line,) = ax.plot(
            [], [], "c-", linewidth=3, marker="o",
            markersize=8, markerfacecolor="cyan", markeredgecolor="blue",
        )
        self._fig = fig
        self._ax = ax
        self._poly_line = poly_line

        # 绑定事件
        fig.canvas.mpl_connect("button_press_event", self._on_click)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        fig.canvas.mpl_connect("close_event", self._on_close)

        plt.tight_layout()
        plt.show()

    # ---- 叠加图 ----
    def _make_overlay(self) -> np.ndarray:
        """生成 mask 半透明叠加 + 已有补线的可视化。"""
        overlay = self.image.copy()
        overlay[self._mask > 0] = (0, 255, 0)
        blended = cv2.addWeighted(self.image, 0.55, overlay, 0.45, 0)
        return blended

    def _update_display(self) -> None:
        """刷新显示。"""
        overlay = self._make_overlay()
        self._overlay_img.set_data(overlay)
        title = (
            f"折线补线 | 左键=添加中心点 | Enter=画道路(宽度={self.road_width}) | "
            "+/-=调宽度 | c=清空当前折线 | u=撤销点 | s=保存结果 | Esc=退出"
        )
        if self._polylines:
            title += f"  [已补 {len(self._polylines)} 段]"
        self._ax.set_title(title, fontsize=9)
        self._fig.canvas.draw_idle()

    # ---- 事件回调 ----
    def _on_click(self, event) -> None:
        """左键添加顶点。"""
        if event.inaxes != self._ax:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata
        self._vertices.append((x, y))
        self._redraw_polyline()

    def _on_key(self, event) -> None:
        """键盘回调。"""
        if event.key == "enter":
            if len(self._vertices) >= 2:
                # 将当前折线画入 mask
                self._paint_polyline()
                self._polylines.append((list(self._vertices), self.road_width))
                self._vertices.clear()
                self._poly_line.set_data([], [])
                self._modified = True
                self._update_display()
                print(f"[REPAIR] 已补线第 {len(self._polylines)} 段，可继续补线或按 s 保存。")
            elif len(self._vertices) == 1:
                print("[REPAIR] 请至少点击 2 个点（起点+终点）再按 Enter。")
            else:
                print("[REPAIR] 当前没有折线点，请先左键点击添加中心点。")

        elif event.key in ("escape", "esc", "\x1b"):
            _force_close_window(self._plt, self._fig)

        elif event.key == "c":
            self._vertices.clear()
            self._poly_line.set_data([], [])
            self._fig.canvas.draw_idle()
            print("[REPAIR] 已清空当前折线。")

        elif event.key == "u":
            if self._vertices:
                removed = self._vertices.pop()
                if self._vertices:
                    self._redraw_polyline()
                else:
                    self._poly_line.set_data([], [])
                    self._fig.canvas.draw_idle()
                print(f"[REPAIR] 已撤销点 ({removed[0]:.0f}, {removed[1]:.0f})，"
                      f"剩余 {len(self._vertices)} 个点。")
            else:
                print("[REPAIR] 当前折线没有可撤销的点。")

        elif event.key == "s":
            self._modified = True
            print("[REPAIR] 已将当前补线结果保存到 mask。")

        elif event.key == "+" or event.key == "=":
            old = self.road_width
            self.road_width = min(old + 5, 100)
            self._update_display()
            print(f"[REPAIR] 道路宽度: {old} → {self.road_width} px")

        elif event.key == "-":
            old = self.road_width
            self.road_width = max(old - 5, 5)
            self._update_display()
            print(f"[REPAIR] 道路宽度: {old} → {self.road_width} px")

    def _on_close(self, event) -> None:
        pass

    # ---- 绘图逻辑 ----
    def _redraw_polyline(self) -> None:
        """更新当前折线显示。"""
        if len(self._vertices) < 1:
            return
        xs, ys = zip(*self._vertices)
        self._poly_line.set_data(xs, ys)
        self._fig.canvas.draw_idle()

    def _paint_polyline(self) -> None:
        """在 mask 上沿当前折线按 road_width 画出道路区域。"""
        if len(self._vertices) < 2:
            return

        pixel_pts = [(int(round(x)), int(round(y))) for (x, y) in self._vertices]
        pts_array = np.array(pixel_pts, dtype=np.int32)

        # 在临时图层上绘制粗折线
        temp = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.polylines(temp, [pts_array], isClosed=False,
                      color=255, thickness=self.road_width,
                      lineType=cv2.LINE_AA)
        # 端点画圆帽
        for px, py in (pixel_pts[0], pixel_pts[-1]):
            cv2.circle(temp, (px, py), self.road_width // 2, 255, -1,
                       lineType=cv2.LINE_AA)

        # 合并到主 mask（OR 操作）
        self._mask = cv2.bitwise_or(self._mask, temp)
