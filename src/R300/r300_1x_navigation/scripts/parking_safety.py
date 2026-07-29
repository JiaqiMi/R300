#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure safety helpers shared by parking_manager and ROS-free regression tests."""

import math


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def validate_turn_tolerances(
        search_step_deg, alignment_step_deg, tolerance_rad,
        safety_margin=1.5):
    smallest_step = min(
        abs(math.radians(float(search_step_deg))),
        abs(math.radians(float(alignment_step_deg))),
    )
    tolerance = abs(float(tolerance_rad))
    if tolerance <= 0.0:
        return False, "search yaw tolerance must be positive"
    if smallest_step <= tolerance:
        return False, "turn step must exceed search yaw tolerance"
    if smallest_step < tolerance * float(safety_margin):
        return False, "turn step lacks yaw tolerance safety margin"
    return True, "OK"


def should_auto_start(previous_state, current_state, armed):
    return (
        previous_state != "COMPLETED"
        and current_state == "COMPLETED"
        and bool(armed)
    )


def bbox_iou(first, second):
    left = max(float(first.x_min), float(second.x_min))
    top = max(float(first.y_min), float(second.y_min))
    right = min(float(first.x_max), float(second.x_max))
    bottom = min(float(first.y_max), float(second.y_max))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first.x_max) - float(first.x_min)) * max(
        0.0, float(first.y_max) - float(first.y_min))
    second_area = max(0.0, float(second.x_max) - float(second.x_min)) * max(
        0.0, float(second.y_max) - float(second.y_min))
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def occupied_visual_match(
        reference, occupied, center_threshold_px, iou_threshold,
        depth_threshold_m):
    center_distance = math.hypot(
        float(reference.center_u) - float(occupied.center_u),
        float(reference.center_v) - float(occupied.center_v),
    )
    iou = bbox_iou(reference, occupied)
    if center_distance > float(center_threshold_px) and iou < float(iou_threshold):
        return False, "NONE", center_distance
    reference_depth_valid = bool(reference.depth_valid) and math.isfinite(
        float(reference.depth_m))
    occupied_depth_valid = bool(occupied.depth_valid) and math.isfinite(
        float(occupied.depth_m))
    if (reference_depth_valid and occupied_depth_valid
            and abs(float(reference.depth_m) - float(occupied.depth_m))
            > float(depth_threshold_m)):
        return False, "DEPTH_REJECT", center_distance
    method = "BBOX_IOU" if iou >= float(iou_threshold) else "PIXEL_CENTER"
    return True, method, center_distance


def occupancy_value_is_blocked(value, threshold, unknown_is_blocked):
    value = int(value)
    if value < 0:
        return bool(unknown_is_blocked), "UNKNOWN"
    if value >= int(threshold):
        return True, "OCCUPIED_%d" % value
    return False, "FREE_%d" % value


def grid_index(width, height, cell_x, cell_y):
    if cell_x < 0 or cell_y < 0 or cell_x >= width or cell_y >= height:
        return None
    return cell_y * width + cell_x


def goal_change_exceeds(
        old_x, old_y, old_yaw, new_x, new_y, new_yaw,
        distance_threshold_m, bearing_threshold_deg):
    distance = math.hypot(float(new_x) - float(old_x),
                          float(new_y) - float(old_y))
    bearing_change = abs(math.degrees(wrap_angle(
        float(new_yaw) - float(old_yaw))))
    return (
        distance > float(distance_threshold_m)
        or bearing_change > float(bearing_threshold_deg)
    ), distance, bearing_change


def confirmation_ready(count, required):
    return int(count) >= int(required)


def is_navigation_target(class_id, free_slot_class_id, parking_sign_class_id):
    return int(class_id) in (
        int(free_slot_class_id), int(parking_sign_class_id))


def execution_is_current(expected, current, active):
    return int(expected) == int(current) and bool(active)


def final_motion_stable(
        linear_speed_mps, yaw_rate_radps,
        linear_tolerance_mps, yaw_rate_tolerance_radps):
    return (
        math.isfinite(float(linear_speed_mps))
        and math.isfinite(float(yaw_rate_radps))
        and abs(float(linear_speed_mps)) <= float(linear_tolerance_mps)
        and abs(float(yaw_rate_radps)) <= float(yaw_rate_tolerance_radps)
    )
