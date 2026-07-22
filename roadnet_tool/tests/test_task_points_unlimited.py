"""Tests: unlimited via task points + clear/persist helpers (no hard 3-point cap)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from roadnet.task_points import TaskPoint, normalize_task_point_sequence
from roadnet.large_image_project import LargeImageProject


def _manual(seq, ptype, x, y, order=None):
    return TaskPoint(
        seq=seq,
        longitude=None,
        latitude=None,
        altitude=0.0,
        point_type=ptype,
        pixel_x=float(x),
        pixel_y=float(y),
        created_order=order if order is not None else seq,
        source="manual_click",
    )


class UnlimitedTaskPointTests(unittest.TestCase):
    def test_can_hold_more_than_three_vias(self):
        points = [
            _manual(1, 0, 10, 10, 0),
            _manual(2, 2, 20, 20, 1),
            _manual(3, 2, 30, 30, 2),
            _manual(4, 2, 40, 40, 3),
            _manual(5, 2, 50, 50, 4),
            _manual(6, 2, 60, 60, 5),
            _manual(7, 1, 70, 70, 6),
        ]
        normalize_task_point_sequence(points)
        self.assertEqual(len(points), 7)
        self.assertEqual(sum(1 for p in points if p.point_type == 0), 1)
        self.assertEqual(sum(1 for p in points if p.point_type == 1), 1)
        self.assertEqual(sum(1 for p in points if p.point_type == 2), 5)

    def test_default_add_pattern_start_then_vias(self):
        """Simulate continuous add_task: first→start, rest→via, no 3-cap."""
        points = []
        for i in range(6):
            has_start = any(int(p.point_type) == 0 for p in points)
            ptype = 0 if not has_start else 2
            points.append(_manual(i + 1, ptype, 10 * (i + 1), 10, i))
        normalize_task_point_sequence(points)
        self.assertEqual(len(points), 6)
        self.assertEqual(points[0].point_type, 0)
        self.assertTrue(all(p.point_type == 2 for p in points[1:]))

    def test_to_dict_has_unified_pixel_aliases(self):
        tp = _manual(1, 0, 123.5, 456.5)
        d = tp.to_dict()
        self.assertEqual(d["pixel_x"], 123.5)
        self.assertEqual(d["x_pixel"], 123.5)
        self.assertEqual(d["y_pixel"], 456.5)
        self.assertEqual(d["coordinate_system"], "original_image_pixel")

    def test_large_image_project_persists_task_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = LargeImageProject(
                image_path="img.tif",
                image_width=10000,
                image_height=8000,
                preview_path=os.path.join(tmp, "preview.png"),
                preview_scale=0.1,
                project_dir=tmp,
                task_points=[
                    {
                        "seq": 1,
                        "point_type": 0,
                        "x_pixel": 100.0,
                        "y_pixel": 200.0,
                        "pixel_x": 100.0,
                        "pixel_y": 200.0,
                        "source": "manual_click",
                        "coordinate_system": "original_image_pixel",
                    }
                ],
            )
            path = project.save()
            self.assertTrue(os.path.isfile(path))
            loaded = LargeImageProject.load(path)
            self.assertEqual(len(loaded.task_points), 1)
            self.assertEqual(loaded.task_points[0]["point_type"], 0)

            # Clear and save empty — reopen must stay empty
            loaded.task_points = []
            loaded.save()
            again = LargeImageProject.load(path)
            self.assertEqual(again.task_points, [])


if __name__ == "__main__":
    unittest.main()
