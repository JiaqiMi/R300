#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.util
import math
import pathlib
import unittest


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts" / "parking_safety.py"
)
MANAGER_PATH = MODULE_PATH.with_name("parking_manager.py")
SPEC = importlib.util.spec_from_file_location("parking_safety", MODULE_PATH)
parking_safety = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parking_safety)


class Detection:
    def __init__(
            self, class_id, center_u=320, center_v=240,
            depth_valid=True, depth_m=2.0,
            x_min=270, y_min=180, x_max=370, y_max=300):
        self.class_id = class_id
        self.center_u = center_u
        self.center_v = center_v
        self.depth_valid = depth_valid
        self.depth_m = depth_m
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max


class ParkingSafetyTest(unittest.TestCase):
    def test_occupied_class_is_never_navigation_target(self):
        self.assertFalse(parking_safety.is_navigation_target(0, 1, 2))

    def test_occupied_wins_same_frame(self):
        matched, _, _ = parking_safety.occupied_visual_match(
            Detection(1), Detection(0), 100, 0.20, 1.50)
        self.assertTrue(matched)

    def test_occupied_invalid_depth_still_matches(self):
        matched, method, _ = parking_safety.occupied_visual_match(
            Detection(1), Detection(0, depth_valid=False, depth_m=float("nan")),
            100, 0.20, 1.50)
        self.assertTrue(matched)
        self.assertIn(method, ("BBOX_IOU", "PIXEL_CENTER"))

    def test_insufficient_free_confirmation(self):
        self.assertFalse(parking_safety.confirmation_ready(2, 3))

    def test_search_tolerance_must_be_smaller_than_steps(self):
        valid, _ = parking_safety.validate_turn_tolerances(28, 8, 0.05)
        self.assertTrue(valid)
        invalid, _ = parking_safety.validate_turn_tolerances(28, 8, 0.12)
        self.assertFalse(invalid)

    def test_latest_goal_change_requires_refresh(self):
        changed, _, _ = parking_safety.goal_change_exceeds(
            0, 0, 0, 0.25, 0, math.radians(1), 0.20, 3.0)
        self.assertTrue(changed)
        source = MANAGER_PATH.read_text(encoding="utf-8")
        entry = source[source.index("    def execute_free_slot_entry"):]
        self.assertLess(
            entry.index("_latest_free_slot_is_valid(reference_slot)"),
            entry.index("_goal_from_latest_slot(latest_slot)"),
        )
        self.assertIn("_goal_from_latest_slot(refreshed_slot)", entry)

    def test_costmap_unknown_occupied_and_outside_block(self):
        self.assertTrue(parking_safety.occupancy_value_is_blocked(-1, 80, True)[0])
        self.assertTrue(parking_safety.occupancy_value_is_blocked(90, 80, True)[0])
        self.assertIsNone(parking_safety.grid_index(10, 10, 10, 5))

    def test_reset_invalidates_old_execution(self):
        self.assertFalse(parking_safety.execution_is_current(4, 5, True))

    def test_completed_repetition_does_not_start(self):
        self.assertFalse(parking_safety.should_auto_start(
            "COMPLETED", "COMPLETED", True))

    def test_final_motion_requires_linear_and_yaw_stability(self):
        self.assertFalse(parking_safety.final_motion_stable(
            0.0, 0.10, 0.05, 0.05))
        self.assertTrue(parking_safety.final_motion_stable(
            0.0, 0.01, 0.05, 0.05))


if __name__ == "__main__":
    unittest.main()
