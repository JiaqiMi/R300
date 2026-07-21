"""Local original-pixel tile editor for a global large-image mask."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from roadnet.large_image_project import ImageRegionReader


def _qimage_rgb(array: np.ndarray) -> QImage:
    source = np.ascontiguousarray(array, dtype=np.uint8)
    h, w = source.shape[:2]
    return QImage(source.data, w, h, source.strides[0], QImage.Format.Format_RGB888).copy()


class TileMaskCanvas(QWidget):
    def __init__(self, rgb: np.ndarray, mask: np.ndarray, parent=None):
        super().__init__(parent)
        self.rgb = np.asarray(rgb, dtype=np.uint8)
        self.mask = np.asarray(mask, dtype=np.uint8)
        self.mode = "add"
        self.radius = 5
        self._drawing = False
        self._last = None
        self._undo = []
        self._cursor = None
        self.setMouseTracking(True)
        self.setMinimumSize(600, 420)

    def _display_rect(self) -> QRectF:
        h, w = self.rgb.shape[:2]
        scale = min(self.width() / max(1, w), self.height() / max(1, h))
        dw, dh = w * scale, h * scale
        return QRectF((self.width() - dw) / 2, (self.height() - dh) / 2, dw, dh)

    def _to_tile(self, position: QPointF):
        rect = self._display_rect()
        if not rect.contains(position):
            return None
        x = int((position.x() - rect.left()) * self.rgb.shape[1] / rect.width())
        y = int((position.y() - rect.top()) * self.rgb.shape[0] / rect.height())
        return (
            max(0, min(self.rgb.shape[1] - 1, x)),
            max(0, min(self.rgb.shape[0] - 1, y)),
        )

    def _paint_segment(self, start, end):
        cv2.line(
            self.mask, start, end, 0 if self.mode == "erase" else 255,
            thickness=max(1, self.radius * 2), lineType=cv2.LINE_8,
        )
        cv2.circle(
            self.mask, end, self.radius, 0 if self.mode == "erase" else 255,
            thickness=-1, lineType=cv2.LINE_8,
        )
        self.update()

    def undo(self):
        if self._undo:
            self.mask[:] = self._undo.pop()
            self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self._to_tile(event.position())
        if point is None:
            return
        self._undo.append(self.mask.copy())
        if len(self._undo) > 20:
            self._undo.pop(0)
        self._drawing = True
        self._last = point
        self._paint_segment(point, point)

    def mouseMoveEvent(self, event):
        point = self._to_tile(event.position())
        self._cursor = point
        if self._drawing and point is not None:
            self._paint_segment(self._last, point)
            self._last = point
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drawing = False
            self._last = None

    def paintEvent(self, _event):
        painter = QPainter(self)
        rect = self._display_rect()
        painter.drawImage(rect, _qimage_rgb(self.rgb))
        rgba = np.zeros((*self.mask.shape, 4), dtype=np.uint8)
        active = self.mask > 0
        rgba[active] = (60, 240, 100, 115)
        overlay = QImage(
            np.ascontiguousarray(rgba).data, rgba.shape[1], rgba.shape[0],
            rgba.strides[0], QImage.Format.Format_RGBA8888,
        ).copy()
        painter.drawImage(rect, overlay)
        if self._cursor is not None:
            x, y = self._cursor
            sx = rect.left() + x * rect.width() / self.mask.shape[1]
            sy = rect.top() + y * rect.height() / self.mask.shape[0]
            radius = self.radius * rect.width() / self.mask.shape[1]
            painter.setPen(QPen(QColor("red") if self.mode == "erase" else QColor("white"), 1))
            painter.drawEllipse(QPointF(sx, sy), radius, radius)


class LargeTileEditDialog(QDialog):
    def __init__(self, image_path: str, global_mask: np.ndarray, center,
                 tile_size: int = 2048, parent=None):
        super().__init__(parent)
        self.setWindowTitle("大图局部 Tile Mask 精修")
        self.resize(980, 760)
        self.setMinimumSize(720, 560)
        reader = ImageRegionReader(image_path)
        cx, cy = int(center[0]), int(center[1])
        half = max(128, int(tile_size) // 2)
        x0 = max(0, min(reader.width - min(tile_size, reader.width), cx - half))
        y0 = max(0, min(reader.height - min(tile_size, reader.height), cy - half))
        x1, y1 = min(reader.width, x0 + tile_size), min(reader.height, y0 + tile_size)
        self.rect_original = (x0, y0, x1, y1)
        self.before_patch = np.asarray(global_mask[y0:y1, x0:x1], dtype=np.uint8).copy()
        rgb = reader.read_region(x0, y0, x1, y1)
        self.canvas = TileMaskCanvas(rgb, self.before_patch.copy(), self)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"Original pixel tile: x=[{x0},{x1}), y=[{y0},{y1})  |  {x1-x0} x {y1-y0}"
        ))
        controls = QHBoxLayout()
        add = QPushButton("画笔（补道路）")
        erase = QPushButton("橡皮擦（删道路）")
        undo = QPushButton("撤销本地一笔")
        radius = QSpinBox()
        radius.setRange(1, 100)
        radius.setValue(5)
        add.clicked.connect(lambda: setattr(self.canvas, "mode", "add"))
        erase.clicked.connect(lambda: setattr(self.canvas, "mode", "erase"))
        undo.clicked.connect(self.canvas.undo)
        radius.valueChanged.connect(lambda value: setattr(self.canvas, "radius", int(value)))
        controls.addWidget(add)
        controls.addWidget(erase)
        controls.addWidget(QLabel("半径"))
        controls.addWidget(radius)
        controls.addWidget(undo)
        controls.addStretch(1)
        layout.addLayout(controls)
        layout.addWidget(self.canvas, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("应用到全局 Mask")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def after_patch(self):
        return self.canvas.mask.copy()

    @property
    def changed(self):
        return not np.array_equal(self.before_patch, self.canvas.mask)
