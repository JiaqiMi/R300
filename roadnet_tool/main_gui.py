#!/usr/bin/env python3
"""
RoadNet Studio — 无人车比赛半自动路网生成与编辑工具
GUI 入口

用法:
    python main_gui.py
    python main_gui.py -i <image_path>
"""

from __future__ import annotations

import sys
import os
import argparse

# 确保项目根目录在 sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from gui.main_window import MainWindow


def parse_args():
    parser = argparse.ArgumentParser(description="RoadNet Studio GUI")
    parser.add_argument(
        "-i", "--image",
        type=str, default="",
        help="启动时打开的影像路径"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 高 DPI 支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("RoadNet Studio")
    app.setApplicationDisplayName("RoadNet Studio - 无人车路网生成与编辑系统")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("RoadNet")

    # 默认字体
    font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    # 创建主窗口
    window = MainWindow()
    window.show()

    # 如果指定了影像，用 QTimer 延迟加载（等待窗口首次布局完成）
    if args.image and os.path.exists(args.image):
        image_path = os.path.abspath(args.image)

        def _delayed_load():
            print(f"[INFO] 自动加载影像: {image_path}")
            window._layer_manager.load_image(image_path)
            window._project_manager.data.image_path = image_path
            # load_image 已通过信号触发 refresh_scene + fit_to_window
            # 再确保一次
            window._canvas.viewport().update()
            window._try_autoload_mask()

        QTimer.singleShot(100, _delayed_load)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
