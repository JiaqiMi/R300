"""Geometry shared by planned-path UI rendering and submission exports."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence


def _value(item, name, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def validate_task_sequence(task_points: Iterable) -> List[str]:
    """Validate the START → via points → GOAL sequence controlled by ``seq``."""
    points = sorted(list(task_points or []), key=lambda p: int(_value(p, "seq", 0)))
    errors: List[str] = []
    if len(points) < 2:
        return ["至少需要 2 个任务点（起点和终点）"]
    seqs = [int(_value(point, "seq", 0)) for point in points]
    if len(set(seqs)) != len(seqs):
        errors.append("任务点 seq 存在重复值")
    types = [int(_value(point, "point_type", 2)) for point in points]
    if types.count(0) != 1:
        errors.append(f"起点数量必须为 1，当前为 {types.count(0)}")
    if types.count(1) != 1:
        errors.append(f"终点数量必须为 1，当前为 {types.count(1)}")
    if types and types[0] != 0:
        errors.append(f"最小 seq={seqs[0]} 的任务点必须是起点（属性 0）")
    if types and types[-1] != 1:
        errors.append(f"最大 seq={seqs[-1]} 的任务点必须是终点（属性 1）")
    invalid_middle = [seqs[i] for i in range(1, len(points) - 1) if types[i] != 2]
    if invalid_middle:
        errors.append(f"中间任务点必须是必经点（属性 2），异常 seq={invalid_middle}")
    return errors


def ordered_task_markers(task_points: Iterable, snapped_points: Iterable = ()) -> List[dict]:
    """Return marker metadata ordered only by task ``seq``."""
    snapped_by_seq = {
        int(_value(point, "seq", 0)): point for point in (snapped_points or [])
    }
    result = []
    via_index = 0
    for point in sorted(list(task_points or []), key=lambda p: int(_value(p, "seq", 0))):
        seq = int(_value(point, "seq", 0))
        point_type = int(_value(point, "point_type", 2))
        snapped = snapped_by_seq.get(seq)
        if snapped is not None:
            x = _value(snapped, "snapped_x")
            y = _value(snapped, "snapped_y")
            status = str(_value(snapped, "status", "pending"))
        else:
            x = _value(point, "pixel_x")
            y = _value(point, "pixel_y")
            status = "unsnapped"
        if x is None or y is None:
            continue
        if point_type == 0:
            role, label = "start", f"START · seq={seq}"
        elif point_type == 1:
            role, label = "goal", f"GOAL · seq={seq}"
        else:
            via_index += 1
            role, label = "waypoint", f"P{via_index} · seq={seq}"
        result.append({
            "seq": seq,
            "point_type": point_type,
            "role": role,
            "label": label,
            "x": float(x),
            "y": float(y),
            "status": status,
        })
    return result


def sample_direction_arrows(
    points: Iterable[Sequence[float]],
    spacing_px: float = 80.0,
    size_px: float = 12.0,
) -> List[dict]:
    """Sample oriented arrow triangles along a path, preserving point order."""
    path = []
    for point in points or []:
        if point is None or len(point) < 2:
            continue
        xy = (float(point[0]), float(point[1]))
        if not path or xy != path[-1]:
            path.append(xy)
    if len(path) < 2 or spacing_px <= 0 or size_px <= 0:
        return []

    lengths = [
        math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]
    total = sum(lengths)
    arrows = []
    target = float(spacing_px)
    traversed = 0.0
    segment_index = 0
    while target < total and segment_index < len(lengths):
        while segment_index < len(lengths) and traversed + lengths[segment_index] < target:
            traversed += lengths[segment_index]
            segment_index += 1
        if segment_index >= len(lengths):
            break
        seg_len = lengths[segment_index]
        if seg_len <= 1e-9:
            segment_index += 1
            continue
        a, b = path[segment_index], path[segment_index + 1]
        ratio = (target - traversed) / seg_len
        x = a[0] + ratio * (b[0] - a[0])
        y = a[1] + ratio * (b[1] - a[1])
        dx, dy = (b[0] - a[0]) / seg_len, (b[1] - a[1]) / seg_len
        nx, ny = -dy, dx
        half = float(size_px) * 0.5
        tip = [x + dx * half, y + dy * half]
        rear_x, rear_y = x - dx * half, y - dy * half
        left = [rear_x + nx * half * 0.65, rear_y + ny * half * 0.65]
        right = [rear_x - nx * half * 0.65, rear_y - ny * half * 0.65]
        arrows.append({
            "center": [x, y],
            "direction": [dx, dy],
            "triangle": [tip, left, right],
        })
        target += float(spacing_px)
    return arrows
