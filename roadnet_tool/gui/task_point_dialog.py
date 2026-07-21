"""Editable task-point list used by RoadNet Studio."""

from __future__ import annotations

import copy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from roadnet.task_points import normalize_task_point_sequence


class TaskPointManagerDialog(QDialog):
    HEADERS = ["seq", "point_type", "lon", "lat", "pixel_x", "pixel_y", "status", "操作"]

    def __init__(self, points, parent=None):
        super().__init__(parent)
        self.setWindowTitle("任务点管理")
        self.resize(1050, 520)
        self.setMinimumSize(760, 400)
        self._points = copy.deepcopy(list(points or []))

        root = QVBoxLayout(self)
        self.table = QTableWidget(0, len(self.HEADERS), self)
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, 1)

        actions = QHBoxLayout()
        for label, callback in (
            ("上移", lambda: self._move(-1)), ("下移", lambda: self._move(1)),
            ("设置为起点", lambda: self._set_type(0)),
            ("设置为终点", lambda: self._set_type(1)),
            ("设置为必经点", lambda: self._set_type(2)),
            ("删除", self._delete), ("重新编号", self._renumber),
        ):
            button = QPushButton(label, self)
            button.clicked.connect(callback)
            actions.addWidget(button)
        actions.addStretch(1)
        root.addLayout(actions)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        box.button(QDialogButtonBox.StandardButton.Save).setText("保存修改")
        box.accepted.connect(self._accept_changes)
        box.rejected.connect(self.reject)
        root.addWidget(box)
        self._refresh()

    def _refresh(self, selected_row=0):
        self.table.setRowCount(len(self._points))
        for row, point in enumerate(self._points):
            values = [
                point.seq, point.point_type, point.longitude, point.latitude,
                point.pixel_x, point.pixel_y, point.status, "选择此行后使用下方操作",
            ]
            for column, value in enumerate(values):
                text = "" if value is None else (f"{value:.8f}" if isinstance(value, float) else str(value))
                item = QTableWidgetItem(text)
                if column not in (0, 1):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        if self._points:
            self.table.selectRow(max(0, min(selected_row, len(self._points) - 1)))

    def _sync_editable_cells(self):
        for row, point in enumerate(self._points):
            try:
                point.seq = int(self.table.item(row, 0).text())
                point_type = int(self.table.item(row, 1).text())
            except (AttributeError, ValueError) as exc:
                raise ValueError(f"第 {row + 1} 行 seq/point_type 必须是整数") from exc
            if point_type not in (0, 1, 2):
                raise ValueError(f"第 {row + 1} 行 point_type 必须是 0、1 或 2")
            point.point_type = point_type

    def _selected_row(self):
        rows = self.table.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def _move(self, delta):
        try:
            self._sync_editable_cells()
        except ValueError as exc:
            QMessageBox.warning(self, "输入错误", str(exc))
            return
        row = self._selected_row()
        target = row + delta
        if row < 0 or target < 0 or target >= len(self._points):
            return
        self._points[row], self._points[target] = self._points[target], self._points[row]
        for seq, point in enumerate(self._points, 1):
            point.seq = seq
        self._refresh(target)

    def _set_type(self, point_type):
        row = self._selected_row()
        if row < 0:
            return
        self._sync_editable_cells()
        selected = self._points[row]
        if point_type in (0, 1):
            for index, point in enumerate(self._points):
                if index != row and point.point_type == point_type:
                    point.point_type = 2
        selected.point_type = point_type
        normalize_task_point_sequence(self._points)
        self._refresh(next((i for i, point in enumerate(self._points) if point is selected), 0))

    def _delete(self):
        row = self._selected_row()
        if row < 0:
            return
        self._sync_editable_cells()
        del self._points[row]
        normalize_task_point_sequence(self._points)
        self._refresh(row)

    def _renumber(self):
        self._sync_editable_cells()
        self._points.sort(key=lambda point: int(point.seq))
        normalize_task_point_sequence(self._points)
        self._refresh(0)

    def _accept_changes(self):
        try:
            self._sync_editable_cells()
            # Explicit seq edits define via order; roles are normalized only now.
            self._points.sort(key=lambda point: int(point.seq))
            normalize_task_point_sequence(self._points)
        except ValueError as exc:
            QMessageBox.warning(self, "输入错误", str(exc))
            return
        self.accept()

    def task_points(self):
        return copy.deepcopy(self._points)
