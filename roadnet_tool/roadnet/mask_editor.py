"""
人工编辑 mask 模块 V2.1：画笔补充 + 橡皮擦删除。

功能：
- 左键拖动：画笔补充道路区域（白色，255）
- 右键拖动：橡皮擦删除非道路区域（黑色，0）
- + / =：增大画笔半径
- -：减小画笔半径
- s：保存当前编辑结果
- u：撤销上一步
- Esc：退出编辑

输出编辑后的 mask 和叠加图。
"""

import gc
import numpy as np
from typing import List, Optional, Tuple
import copy


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


class MaskEditor:
    """
    基于 matplotlib 的交互式 mask 编辑器。

    操作方式：
        左键拖动  → 画刷补充（涂白，标记为道路）
        右键拖动  → 橡皮擦删除（涂黑，标记为非道路）
        +/-       → 调整画笔半径
        s         → 保存当前结果
        u         → 撤销上一步
        Esc       → 退出编辑

    使用方式：
        editor = MaskEditor(image_rgb, mask)
        editor.run()
        edited_mask = editor.edited_mask
    """

    def __init__(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        brush_radius: int = 8,
        max_undo_steps: int = 20,
    ):
        self.image = image_rgb
        self.h, self.w = image_rgb.shape[:2]

        # 确保 mask 是 uint8 二值格式
        self._mask = (mask > 127).astype(np.uint8) * 255

        self.brush_radius = brush_radius
        self.max_undo_steps = max_undo_steps

        # 撤销历史
        self._undo_stack: List[np.ndarray] = []

        # 当前修改的临时状态
        self._modified: bool = False

        # 画笔颜色：左键=白(255)，右键=黑(0)
        self._current_color: int = 255

        # matplotlib 引用
        self._fig = None
        self._ax = None
        self._overlay_img = None
        self._cursor_circle = None
        self._is_drawing: bool = False
        self._plt = None

    # ---- 属性 ----
    @property
    def edited_mask(self) -> np.ndarray:
        """返回编辑后的 mask"""
        return self._mask.copy()

    @property
    def is_modified(self) -> bool:
        return self._modified

    # ---- 主入口 ----
    def run(self) -> None:
        """启动交互编辑窗口，阻塞直到用户退出。"""
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        self._plt = plt

        # 配置中文字体
        _setup_cjk_font()

        fig, ax = plt.subplots(figsize=(10, 8))

        # 生成半透明叠加图
        overlay = self._make_overlay()
        self._overlay_img = ax.imshow(overlay)

        title_fmt = (
            "画笔编辑 | 左键=补道路(白) | 右键=擦除(黑) | "
            "+/-=画笔大小(%d) | s=保存 | u=撤销 | Esc=退出"
        )
        ax.set_title(title_fmt % self.brush_radius, fontsize=9)
        ax.axis("off")

        # 画笔光标圆
        (cursor,) = ax.plot(
            [], [], "o",
            color="cyan", markersize=self.brush_radius * 2,
            markerfacecolor="none", markeredgewidth=1.5, alpha=0.8,
        )
        self._fig = fig
        self._ax = ax
        self._cursor_circle = cursor

        # 绑定事件
        fig.canvas.mpl_connect("button_press_event", self._on_press)
        fig.canvas.mpl_connect("button_release_event", self._on_release)
        fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        fig.canvas.mpl_connect("close_event", self._on_close)

        plt.tight_layout()
        plt.show()

    # ---- 叠加图生成 ----
    def _make_overlay(self) -> np.ndarray:
        """生成 mask 半透明叠加图像"""
        import cv2

        overlay = self.image.copy()
        # 道路区域涂绿
        overlay[self._mask > 0] = (0, 255, 0)
        blended = cv2.addWeighted(self.image, 0.55, overlay, 0.45, 0)
        return blended

    def _update_display(self) -> None:
        """更新叠加图显示"""
        overlay = self._make_overlay()
        self._overlay_img.set_data(overlay)
        title_fmt = (
            "画笔编辑 | 左键=补道路(白) | 右键=擦除(黑) | "
            "+/-=画笔大小(%d) | s=保存 | u=撤销 | Esc=退出"
        )
        self._ax.set_title(title_fmt % self.brush_radius, fontsize=9)
        self._fig.canvas.draw_idle()

    # ---- 事件回调 ----
    def _on_press(self, event) -> None:
        """鼠标按下：开始绘制。"""
        if event.inaxes != self._ax:
            return
        if event.button == 1:
            # 保存撤销快照
            self._push_undo()
            self._current_color = 255  # 白色 = 补道路
            self._is_drawing = True
            self._paint_at(event.xdata, event.ydata)
        elif event.button == 3:
            self._push_undo()
            self._current_color = 0  # 黑色 = 擦除
            self._is_drawing = True
            self._paint_at(event.xdata, event.ydata)

    def _on_release(self, event) -> None:
        """鼠标释放：停止绘制。"""
        self._is_drawing = False

    def _on_motion(self, event) -> None:
        """鼠标移动：绘制/光标。"""
        if event.inaxes != self._ax:
            self._cursor_circle.set_data([], [])
            self._fig.canvas.draw_idle()
            return

        # 更新光标位置
        self._cursor_circle.set_data([event.xdata], [event.ydata])

        if self._is_drawing:
            self._paint_at(event.xdata, event.ydata)

        self._fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        """键盘回调。"""
        if event.key in ("escape", "esc", "\x1b"):
            _force_close_window(self._plt, self._fig)
        elif event.key in ("+", "="):
            self.brush_radius = min(50, self.brush_radius + 2)
            self._cursor_circle.set_markersize(self.brush_radius * 2)
            self._update_display()
            print(f"[EDIT] 画笔半径: {self.brush_radius}")
        elif event.key == "-":
            self.brush_radius = max(1, self.brush_radius - 2)
            self._cursor_circle.set_markersize(self.brush_radius * 2)
            self._update_display()
            print(f"[EDIT] 画笔半径: {self.brush_radius}")
        elif event.key == "s":
            self._modified = True
            print("[EDIT] 已保存当前编辑结果（按 Esc 退出）。")
        elif event.key == "u":
            self._undo()
            self._update_display()

    def _on_close(self, event) -> None:
        """窗口关闭。"""
        pass

    # ---- 绘制逻辑 ----
    def _paint_at(self, mx: float, my: float) -> None:
        """在指定坐标处用当前画笔颜色绘制圆形。"""
        import cv2

        col = int(round(mx))
        row = int(round(my))
        r = self.brush_radius

        # 边界检查
        c1 = max(0, col - r)
        c2 = min(self.w, col + r + 1)
        r1 = max(0, row - r)
        r2 = min(self.h, row + r + 1)

        # 创建圆形 mask
        patch_h = r2 - r1
        patch_w = c2 - c1
        circle_mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
        cv2.circle(
            circle_mask,
            (col - c1, row - r1), r,
            255, -1,
        )

        # 只在圆形范围内修改
        if self._current_color == 255:
            self._mask[r1:r2, c1:c2] = np.where(
                circle_mask > 0, 255, self._mask[r1:r2, c1:c2]
            )
        else:
            self._mask[r1:r2, c1:c2] = np.where(
                circle_mask > 0, 0, self._mask[r1:r2, c1:c2]
            )

        self._update_display()

    # ---- 撤销逻辑 ----
    def _push_undo(self) -> None:
        """保存当前 mask 状态到撤销栈。"""
        self._undo_stack.append(self._mask.copy())
        # 限制最大撤销步数
        if len(self._undo_stack) > self.max_undo_steps:
            self._undo_stack.pop(0)

    def _undo(self) -> bool:
        """撤销上一步。"""
        if not self._undo_stack:
            print("[EDIT] 没有可撤销的操作。")
            return False
        self._mask = self._undo_stack.pop()
        print(f"[EDIT] 已撤销（剩余可撤销步数: {len(self._undo_stack)}）")
        return True
