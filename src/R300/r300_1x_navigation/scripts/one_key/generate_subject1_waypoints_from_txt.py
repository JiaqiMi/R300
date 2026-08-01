#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert the competition semicolon-delimited TXT task file to ROS waypoint YAML.

Input columns (one task point per line):
    sequence;longitude;latitude;altitude;attribute

Attribute semantics:
    0 = competition start
    1 = competition finish
    2 = mandatory waypoint

The output always contains every TXT point exactly once.  The unique start is
placed first, the unique finish is placed last, and all remaining points keep
their original line order from the TXT file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import List, Sequence


@dataclass(frozen=True)
class TaskPoint:
    line_number: int
    sequence: int
    longitude_text: str
    latitude_text: str
    altitude_text: str
    attribute: int

    @property
    def role(self) -> str:
        return {0: "start", 1: "finish", 2: "mandatory"}[self.attribute]


def decimal_in_range(text: str, minimum: Decimal, maximum: Decimal, field: str, line: int) -> None:
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"第 {line} 行 {field} 不是合法数字：{text!r}") from exc
    if not value.is_finite() or value < minimum or value > maximum:
        raise ValueError(
            f"第 {line} 行 {field} 超出范围 [{minimum}, {maximum}]：{text!r}"
        )


def parse_task_file(path: Path) -> List[TaskPoint]:
    if not path.is_file():
        raise ValueError(f"TXT 文件不存在：{path}")

    points: List[TaskPoint] = []
    seen_sequences = set()

    # utf-8-sig accepts both normal UTF-8 and UTF-8 files with a BOM.
    with path.open("r", encoding="utf-8-sig", newline=None) as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue

            columns = [column.strip() for column in line.split(";")]
            if len(columns) != 5:
                raise ValueError(
                    f"第 {line_number} 行应有 5 列，实际为 {len(columns)} 列：{line!r}"
                )

            sequence_text, longitude_text, latitude_text, altitude_text, attribute_text = columns

            try:
                sequence = int(sequence_text)
            except ValueError as exc:
                raise ValueError(
                    f"第 {line_number} 行序号不是整数：{sequence_text!r}"
                ) from exc
            if sequence < 1:
                raise ValueError(f"第 {line_number} 行序号必须从 1 开始：{sequence}")
            if sequence in seen_sequences:
                raise ValueError(f"第 {line_number} 行序号重复：{sequence}")
            seen_sequences.add(sequence)

            decimal_in_range(longitude_text, Decimal("-180"), Decimal("180"), "经度", line_number)
            decimal_in_range(latitude_text, Decimal("-90"), Decimal("90"), "纬度", line_number)
            decimal_in_range(
                altitude_text,
                Decimal("-100000"),
                Decimal("100000"),
                "高程",
                line_number,
            )

            try:
                attribute = int(attribute_text)
            except ValueError as exc:
                raise ValueError(
                    f"第 {line_number} 行属性不是整数：{attribute_text!r}"
                ) from exc
            if attribute not in (0, 1, 2):
                raise ValueError(
                    f"第 {line_number} 行属性只能是 0、1 或 2，实际为：{attribute}"
                )

            points.append(
                TaskPoint(
                    line_number=line_number,
                    sequence=sequence,
                    longitude_text=longitude_text,
                    latitude_text=latitude_text,
                    altitude_text=altitude_text,
                    attribute=attribute,
                )
            )

    if len(points) < 2:
        raise ValueError("任务文件至少需要包含起点和终点两个任务点")

    starts = [point for point in points if point.attribute == 0]
    finishes = [point for point in points if point.attribute == 1]
    if len(starts) != 1:
        raise ValueError(f"任务文件必须且只能有 1 个属性 0 起点，当前有 {len(starts)} 个")
    if len(finishes) != 1:
        raise ValueError(f"任务文件必须且只能有 1 个属性 1 终点，当前有 {len(finishes)} 个")

    return points


def ordered_points(points: Sequence[TaskPoint]) -> List[TaskPoint]:
    start = next(point for point in points if point.attribute == 0)
    finish = next(point for point in points if point.attribute == 1)
    middle = [point for point in points if point.attribute not in (0, 1)]
    return [start, *middle, finish]


def yaml_name(point: TaskPoint, output_index: int, total: int) -> str:
    width = max(2, len(str(total)), len(str(point.sequence)))
    if point.attribute == 0:
        return f"start_{point.sequence:0{width}d}"
    if point.attribute == 1:
        return f"finish_{point.sequence:0{width}d}"
    return f"wp_{output_index:0{width}d}"


def render_yaml(points: Sequence[TaskPoint], source_path: Path) -> str:
    generated_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# 自动生成文件，请不要手工编辑。",
        f"# 来源 TXT: {source_path}",
        f"# 生成时间: {generated_at}",
        "# 排序规则：属性0起点置于首位，属性1终点置于末位，其余点保持TXT原行顺序。",
        "subject1_waypoints:",
        "  waypoints:",
    ]

    total = len(points)
    for output_index, point in enumerate(points, start=1):
        role_text = {0: "起点", 1: "终点", 2: "必经点"}[point.attribute]
        lines.extend(
            [
                f"    # TXT序号={point.sequence}，属性={point.attribute}（{role_text}）",
                f"    - name: {yaml_name(point, output_index, total)}",
                f"      latitude_deg: {point.latitude_text}",
                f"      longitude_deg: {point.longitude_text}",
                f"      altitude_m: {point.altitude_text}",
            ]
        )
        if output_index != total:
            lines.append("")

    return "\n".join(lines) + "\n"


def default_output_path() -> Path:
    # .../r300_1x_navigation/scripts/one_key/this_file.py -> package root is parents[2].
    package_root = Path(__file__).resolve().parents[2]
    return package_root / "config" / "subject1_waypoints.yaml"


def atomic_write(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=str(output_path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary_path), str(output_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将比赛 TXT 任务文件转换并覆盖 subject1_waypoints.yaml"
    )
    parser.add_argument("input_txt", type=Path, help="比赛下发的 UTF-8 TXT 任务文件")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="输出 YAML；默认覆盖本功能包 config/subject1_waypoints.yaml",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="覆盖前不备份原 subject1_waypoints.yaml",
    )
    args = parser.parse_args()

    input_path = args.input_txt.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    try:
        parsed = parse_task_file(input_path)
        arranged = ordered_points(parsed)
        content = render_yaml(arranged, input_path)

        backup_path = None
        if output_path.exists() and not args.no_backup:
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = output_path.with_name(f"{output_path.name}.bak.{timestamp}")
            shutil.copy2(output_path, backup_path)

        atomic_write(output_path, content)
    except (OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    start = arranged[0]
    finish = arranged[-1]
    print(f"[ OK ] 已读取 {len(parsed)} 个任务点")
    print(
        f"[ OK ] 起点：TXT序号={start.sequence}，lon={start.longitude_text}，lat={start.latitude_text}"
    )
    print(
        f"[ OK ] 终点：TXT序号={finish.sequence}，lon={finish.longitude_text}，lat={finish.latitude_text}"
    )
    if backup_path is not None:
        print(f"[ OK ] 原 YAML 已备份：{backup_path}")
    print(f"[ OK ] 已覆盖生成：{output_path}")
    print("[INFO] 航点 YAML 在导航 launch 启动时加载；若导航已运行，请重启导航后生效。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
