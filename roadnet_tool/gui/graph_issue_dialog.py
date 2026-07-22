"""Dialog: graph issue list with click-to-focus."""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QHeaderView,
    QAbstractItemView,
)


_SEV_LABEL = {"error": "严重", "warning": "可疑", "info": "提示"}
_SEV_COLOR = {
    "error": QColor(255, 80, 80),
    "warning": QColor(255, 160, 60),
    "info": QColor(240, 210, 60),
}


class GraphIssueDialog(QDialog):
    issue_activated = Signal(str)  # issue_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("路网异常列表")
        self.resize(900, 480)
        self.setModal(False)
        self._issues = []

        layout = QVBoxLayout(self)
        self._summary = QLabel("尚未分析")
        layout.addWidget(self._summary)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["编号", "严重程度", "类型", "对象ID", "简短说明", "建议操作"]
        )
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.cellClicked.connect(self._on_cell)
        self._table.cellDoubleClicked.connect(self._on_cell)
        layout.addWidget(self._table)

        row = QHBoxLayout()
        self._btn_export = QPushButton("重新导出报告")
        self._btn_close = QPushButton("关闭")
        row.addStretch(1)
        row.addWidget(self._btn_export)
        row.addWidget(self._btn_close)
        layout.addLayout(row)
        self._btn_close.clicked.connect(self.hide)
        self.on_export: Optional[Callable[[], None]] = None
        self._btn_export.clicked.connect(self._export)

    def _export(self):
        if callable(self.on_export):
            self.on_export()

    def set_report(self, report: Optional[dict]):
        self._issues = list((report or {}).get("issues") or [])
        stale = bool((report or {}).get("stale"))
        n = int((report or {}).get("issue_count") or len(self._issues))
        se = int((report or {}).get("serious_issue_count") or 0)
        wa = int((report or {}).get("warning_issue_count") or 0)
        if stale:
            self._summary.setText("路网已修改，请重新分析异常。")
            self._summary.setStyleSheet("color: #ffaa00;")
        else:
            self._summary.setText(
                f"共 {n} 项异常（严重 {se} / 可疑 {wa} / 提示 {n - se - wa}）"
            )
            self._summary.setStyleSheet("")

        self._table.setRowCount(0)
        if stale:
            return
        for issue in self._issues:
            r = self._table.rowCount()
            self._table.insertRow(r)
            sev = issue.get("severity", "warning")
            vals = [
                str(issue.get("issue_id", "")),
                _SEV_LABEL.get(sev, sev),
                str(issue.get("issue_type", "")),
                str(issue.get("object_id", "")),
                str(issue.get("message", "")),
                str(issue.get("suggestion", "")),
            ]
            color = _SEV_COLOR.get(sev)
            for c, text in enumerate(vals):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, issue.get("issue_id"))
                if color and c == 1:
                    item.setForeground(color)
                self._table.setItem(r, c, item)

    def _on_cell(self, row: int, _col: int):
        item = self._table.item(row, 0)
        if item is None:
            return
        issue_id = item.data(Qt.UserRole) or item.text()
        if issue_id:
            self.issue_activated.emit(str(issue_id))
