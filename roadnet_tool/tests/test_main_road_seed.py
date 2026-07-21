"""Tests for large-image main-road seed strokes / ribbon rebuild."""

from __future__ import annotations

import unittest

import numpy as np

from roadnet.main_road_seed import (
    apply_angle_constraint,
    build_road_ribbon_mask,
    compute_road_radius_px,
    deserialize_seed_strokes,
    make_seed_stroke,
    next_seed_id,
    rebuild_mask_from_seed_ribbons,
    serialize_seed_strokes,
    snap_point_to_candidates,
)


class MainRoadSeedTests(unittest.TestCase):
    def test_two_point_stroke_is_line_type(self):
        s = make_seed_stroke(
            [(10, 10), (100, 10)],
            stroke_id="seed_001",
            road_width_m=8.0,
            gsd_m_per_px=0.5,
            source="two_point_click",
        )
        self.assertEqual(s["type"], "line")
        self.assertEqual(s["source"], "two_point_click")
        self.assertEqual(len(s["points"]), 2)
        self.assertAlmostEqual(s["road_radius_px"], 8.0)  # 8/2/0.5

    def test_radius_from_gsd_and_custom_px(self):
        self.assertAlmostEqual(compute_road_radius_px(12.0, gsd_m_per_px=0.5), 12.0)
        self.assertAlmostEqual(compute_road_radius_px(12.0, road_radius_px=7.0), 7.0)

    def test_ribbon_writes_mask_pixels(self):
        stroke = make_seed_stroke(
            [(20, 50), (180, 50)],
            stroke_id="seed_001",
            road_width_m=8.0,
            road_radius_px=5.0,
            source="two_point_click",
        )
        ribbon = build_road_ribbon_mask((100, 200), [stroke])
        self.assertGreater(int(np.count_nonzero(ribbon)), 500)
        # centerline row should be mostly covered
        self.assertGreater(int(np.count_nonzero(ribbon[50, 20:180])), 100)

    def test_rebuild_merges_ribbon_into_working(self):
        working = np.zeros((120, 200), dtype=np.uint8)
        working[5:25, 5:25] = 255  # far blob (top-left)
        working[48:52, 80:100] = 255  # near road fragment
        stroke = make_seed_stroke(
            [(10, 50), (190, 50)],
            stroke_id="seed_001",
            road_radius_px=6.0,
            source="two_point_click",
        )
        repaired, report = rebuild_mask_from_seed_ribbons(
            working, [stroke], far_component_distance_px=15.0
        )
        self.assertGreater(report["ribbon_nonzero"], 0)
        self.assertGreater(int(np.count_nonzero(repaired[50, 10:190])), 50)
        # far blob should be removed
        self.assertEqual(int(np.count_nonzero(repaired[5:25, 5:25])), 0)

    def test_serialize_roundtrip(self):
        strokes = [
            make_seed_stroke([(0, 0), (10, 0)], stroke_id="seed_001", source="two_point_click"),
            make_seed_stroke([(0, 0), (5, 5), (10, 0)], stroke_id="seed_002", source="polyline_click"),
        ]
        payload = serialize_seed_strokes(strokes)
        back = deserialize_seed_strokes(payload)
        self.assertEqual(len(back), 2)
        self.assertEqual(back[0]["id"], "seed_001")
        self.assertEqual(back[1]["type"], "polyline")

    def test_legacy_point_list_load(self):
        payload = {
            "coordinate_system": "original_image_pixel",
            "strokes": [[[0, 0], [20, 0]]],
        }
        back = deserialize_seed_strokes(payload)
        self.assertEqual(len(back), 1)
        self.assertEqual(len(back[0]["points"]), 2)

    def test_next_seed_id_and_snap(self):
        self.assertEqual(next_seed_id([{"id": "seed_001"}, {"id": "seed_003"}]), "seed_004")
        x, y, ok = snap_point_to_candidates(10, 10, [(12, 11), (50, 50)], 5)
        self.assertTrue(ok)
        self.assertAlmostEqual(x, 12)
        ex, ey = apply_angle_constraint((0, 0), (10, 1))
        self.assertAlmostEqual(ey, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
