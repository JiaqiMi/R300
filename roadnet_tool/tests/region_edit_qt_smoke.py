"""Manual/offscreen Qt smoke check for the region-edit minimum loop."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from gui.main_window import MainWindow


def _mouse_event(event_type, point, button, buttons):
    return QMouseEvent(
        event_type, QPointF(*point), button, buttons,
        Qt.KeyboardModifier.NoModifier,
    )


def main():
    app = QApplication.instance() or QApplication([])
    # The smoke check must never wait for a modal dialog.
    QMessageBox.information = staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    QMessageBox.warning = staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    QMessageBox.critical = staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Ok)

    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = Path(temp_dir) / "image.png"
        cv2.imwrite(str(image_path), np.full((100, 120, 3), 160, dtype=np.uint8))

        window = MainWindow()
        window._layer_manager.load_image(str(image_path))
        window._layer_manager.set_layer_data(
            "mask", np.full((100, 120), 255, dtype=np.uint8)
        )
        window.set_stage("edit")
        canvas = window._canvas

        # ROI add/apply/undo.
        canvas._add_roi_point(10, 10)
        canvas._add_roi_point(80, 10)
        canvas._add_roi_point(10, 80)
        canvas._finalize_roi()
        assert len(canvas.get_roi_regions()) == 1
        window._on_global_undo()
        assert len(canvas.get_roi_regions()) == 0
        window._on_global_redo()
        assert len(canvas.get_roi_regions()) == 1
        window._on_apply_roi()
        assert window._layer_manager.get_layer_data("mask")[99, 119] == 0
        window._on_global_undo()
        assert window._layer_manager.get_layer_data("mask")[99, 119] == 255

        # Ignore add/apply/undo.
        canvas._add_ignore_point(20, 20)
        canvas._add_ignore_point(50, 20)
        canvas._add_ignore_point(20, 50)
        canvas._finalize_ignore()
        assert len(canvas.get_ignore_regions()) == 1
        window._on_global_undo()
        assert len(canvas.get_ignore_regions()) == 0
        window._on_global_redo()
        assert len(canvas.get_ignore_regions()) == 1
        window._on_apply_ignore()
        assert window._layer_manager.get_layer_data("mask")[25, 25] == 0
        window._on_global_undo()
        assert window._layer_manager.get_layer_data("mask")[25, 25] == 255

        # Brush stroke/undo using the same handlers as real mouse input.
        window._layer_manager.set_layer_data(
            "mask", np.zeros((100, 120), dtype=np.uint8)
        )
        window._history.clear()
        canvas.set_tool("mask_brush")
        press = _mouse_event(
            QEvent.Type.MouseButtonPress, (60, 60),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        )
        release = _mouse_event(
            QEvent.Type.MouseButtonRelease, (60, 60),
            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
        )
        canvas.handle_mask_refine_press(press, QPointF(60, 60))
        canvas.handle_mask_refine_release(release)
        assert np.count_nonzero(window._layer_manager.get_layer_data("mask")) > 0
        window._on_global_undo()
        assert np.count_nonzero(window._layer_manager.get_layer_data("mask")) == 0

        # Eraser stroke/undo.
        window._layer_manager.set_layer_data(
            "mask", np.full((100, 120), 255, dtype=np.uint8)
        )
        window._history.clear()
        canvas.set_tool("mask_eraser")
        canvas.handle_mask_refine_press(press, QPointF(60, 60))
        canvas.handle_mask_refine_release(release)
        assert np.count_nonzero(window._layer_manager.get_layer_data("mask")) < 12000
        window._on_global_undo()
        assert np.count_nonzero(window._layer_manager.get_layer_data("mask")) == 12000

        visible = window._layer_manager.get_visible_layers()
        assert all(name in visible for name in (
            "layer_road_mask", "layer_roi", "layer_ignore"
        ))
        report = window._on_region_edit_self_check()
        assert report["mask_exists"] is True
        assert report["mask_dtype"] == "uint8"
        window.close()
        app.processEvents()

    print("REGION_EDIT_QT_SMOKE_OK")


if __name__ == "__main__":
    main()
