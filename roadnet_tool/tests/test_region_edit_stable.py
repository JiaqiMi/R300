import unittest
import tempfile
from pathlib import Path

import numpy as np

from roadnet.region_edit import (
    PolygonRegion,
    apply_ignore_regions,
    apply_roi_regions,
    ensure_mask_uint8,
    paint_mask_segment,
    save_mask_png_verified,
)


ROOT = Path(__file__).resolve().parents[1]


class RegionEditPrimitiveTests(unittest.TestCase):
    def test_roi_keeps_only_enabled_roi(self):
        mask = np.full((20, 20), 255, dtype=np.uint8)
        roi = PolygonRegion.create("roi", [(3, 3), (16, 3), (16, 16), (3, 16)])
        edited, affected = apply_roi_regions(mask, [roi])
        self.assertEqual(edited.dtype, np.uint8)
        self.assertEqual(edited.ndim, 2)
        self.assertEqual(int(edited[0, 0]), 0)
        self.assertEqual(int(edited[10, 10]), 255)
        self.assertGreater(affected, 0)
        np.testing.assert_array_equal(mask, np.full((20, 20), 255, dtype=np.uint8))

    def test_ignore_only_removes_polygon_in_mask(self):
        mask = np.full((20, 20), 255, dtype=np.uint8)
        ignore = PolygonRegion.create(
            "ignore", [(5, 5), (14, 5), (14, 14), (5, 14)]
        )
        edited, affected = apply_ignore_regions(mask, [ignore])
        self.assertEqual(int(edited[0, 0]), 255)
        self.assertEqual(int(edited[10, 10]), 0)
        self.assertEqual(affected, 100)

    def test_brush_and_eraser_write_mask_array(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        paint_mask_segment(mask, (5, 20), (30, 20), radius=3, erase=False)
        self.assertGreater(int(np.count_nonzero(mask)), 0)
        self.assertEqual(set(np.unique(mask)).issubset({0, 255}), True)
        before = int(np.count_nonzero(mask))
        paint_mask_segment(mask, (10, 20), (25, 20), radius=3, erase=True)
        self.assertLess(int(np.count_nonzero(mask)), before)

    def test_mask_normalization_rejects_object(self):
        with self.assertRaisesRegex(TypeError, "dtype=object"):
            ensure_mask_uint8(np.array([[object()]], dtype=object))

    def test_saved_mask_can_be_reopened_with_same_pixels(self):
        mask = np.zeros((31, 37), dtype=np.uint8)
        paint_mask_segment(mask, (3, 15), (30, 15), radius=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_mask_png_verified(mask, Path(temp_dir) / "current_mask.png")
            self.assertTrue(path.is_file())
            import cv2
            reopened = cv2.imdecode(
                np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE
            )
            np.testing.assert_array_equal(reopened, mask)


class RegionEditIntegrationSourceTests(unittest.TestCase):
    def test_history_imports_qpolygonf_from_qtgui(self):
        source = (ROOT / "gui" / "history_manager.py").read_text(encoding="utf-8")
        self.assertIn("from PySide6.QtGui import QPolygonF", source)
        self.assertNotIn("from PySide6.QtCore import QPointF, QPolygonF", source)

    def test_edit_stage_keeps_core_layers_visible(self):
        source = (ROOT / "gui" / "layer_manager.py").read_text(encoding="utf-8")
        self.assertIn('if stage == "edit":', source)
        self.assertIn('("layer_road_mask", "layer_roi", "layer_ignore")', source)

    def test_stable_mode_has_separate_brush_and_eraser(self):
        tool_source = (ROOT / "gui" / "tool_panel.py").read_text(encoding="utf-8")
        main_source = (ROOT / "gui" / "main_window.py").read_text(encoding="utf-8")
        self.assertIn('("mask_brush",', tool_source)
        self.assertIn('("mask_eraser",', tool_source)
        self.assertIn('"mask_brush", "mask_eraser"', main_source)

    def test_high_confidence_auto_apply_is_blocked(self):
        source = (ROOT / "gui" / "main_window.py").read_text(encoding="utf-8")
        panel = (ROOT / "gui" / "parameter_panel.py").read_text(encoding="utf-8")
        self.assertIn("automatic high-confidence Ignore is disabled in stable mode", source)
        self.assertIn("button.setEnabled(False)", panel)
        self.assertIn("button.setVisible(False)", panel)

    def test_self_check_and_verified_save_are_exposed(self):
        source = (ROOT / "gui" / "main_window.py").read_text(encoding="utf-8")
        self.assertIn("def _on_region_edit_self_check", source)
        self.assertIn("def _on_save_current_mask", source)
        self.assertIn('"save_mask":       self._on_save_current_mask', source)
        self.assertIn("save_mask_png_verified", source)


if __name__ == "__main__":
    unittest.main()
