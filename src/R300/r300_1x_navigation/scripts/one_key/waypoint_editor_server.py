#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""航点补点编辑器 —— 独立小工具（考试档, 2026-08-04）。

用途: 考卷 TXT 航点太稀(长腿/缺角)时, 在白底网格画布上人工补充属性2引导点,
     确认后走【现有生成器】产出 config/subject1_waypoints.yaml。

设计红线:
  - 独立项目: 零 ROS 依赖、零第三方库(纯 Python 标准库 + 原生 JS)、零外网、
    与 r300_web_dashboard 完全解耦, 独立端口(默认 8099)。
  - 考卷原件只读: 确认时写"合并 TXT"(renwu1_edited_时间戳.txt), 原 TXT 一字不动。
  - 序号连续重排: 起点恒 1、终点恒 N、行顺序=序号顺序(生成器按行序排中间点,
    重排后两者恒一致); 新点一律属性 2。
  - yaml 只经生成器产出(校验/备份/原子写/精度单一来源), 本工具绝不直接写 yaml。

运行(在 one_key 目录):
  python3 waypoint_editor_server.py [--txt 考卷路径] [--port 8099] [--yaml-out 测试用输出]
  浏览器打开 http://<主机>:8099
"""

import argparse
import datetime as dt
import importlib.util
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent            # .../scripts/one_key
GENERATOR = HERE / "generate_subject1_waypoints_from_txt.py"   # 同目录
DEFAULT_TXT = HERE.parents[1] / "waypoints" / "renwu1.txt"     # 包内 waypoints/

TXT_PATH = DEFAULT_TXT
YAML_OUT = None  # None = 生成器默认(config/subject1_waypoints.yaml)


def load_generator_parser():
    """复用生成器的 parse_task_file(5列校验/唯一序号/起终点唯一), 零重复代码。"""
    spec = importlib.util.spec_from_file_location("wp_gen", str(GENERATOR))
    module = importlib.util.module_from_spec(spec)
    sys.modules["wp_gen"] = module  # dataclasses 需要模块已注册才能解析注解
    spec.loader.exec_module(module)
    return module


GEN = load_generator_parser()


def read_points():
    points = GEN.parse_task_file(TXT_PATH)
    ordered = GEN.ordered_points(points)
    return [
        {
            "seq": p.sequence,
            "lon": p.longitude_text,
            "lat": p.latitude_text,
            "alt": p.altitude_text,
            "attr": p.attribute,
        }
        for p in ordered
    ]


def validate_and_renumber(points):
    """校验前端送来的完整序列并重排序号 1..N。返回合并 TXT 文本。"""
    if len(points) < 2:
        raise ValueError("至少需要起点和终点两个点")
    if int(points[0]["attr"]) != 0:
        raise ValueError("第一个点必须是属性0起点")
    if int(points[-1]["attr"]) != 1:
        raise ValueError("最后一个点必须是属性1终点")
    for p in points[1:-1]:
        if int(p["attr"]) != 2:
            raise ValueError("中间点属性必须为2")
    lines = []
    for i, p in enumerate(points, start=1):
        lon, lat = str(p["lon"]).strip(), str(p["lat"]).strip()
        alt = str(p.get("alt", "0")).strip() or "0"
        # 基本数值防呆(生成器还会再全量校验一遍)
        float(lon), float(lat), float(alt)
        lines.append(f"{i};{lon};{lat};{alt};{int(p['attr'])}")
    return "\n".join(lines) + "\n"


def run_generate(points):
    content = validate_and_renumber(points)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    merged = TXT_PATH.with_name(f"{TXT_PATH.stem}_edited_{stamp}.txt")
    merged.write_text(content, encoding="utf-8")
    cmd = [sys.executable, str(GENERATOR), str(merged)]
    if YAML_OUT:
        cmd += ["--output", str(YAML_OUT)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return {
        "ok": proc.returncode == 0,
        "merged_txt": str(merged),
        "generator_output": (proc.stdout + proc.stderr).strip(),
    }


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (HERE / "waypoint_editor.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/points":
            try:
                self._json({"ok": True, "txt": str(TXT_PATH),
                            "points": read_points()})
            except Exception as exc:  # noqa: BLE001
                self._json({"ok": False, "error": str(exc)}, 400)
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/api/generate":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = run_generate(payload["points"])
            self._json(result)
        except Exception as exc:  # noqa: BLE001
            self._json({"ok": False, "error": str(exc)}, 400)

    def log_message(self, fmt, *args):  # 安静点
        sys.stderr.write("[editor] " + fmt % args + "\n")


def main():
    global TXT_PATH, YAML_OUT
    ap = argparse.ArgumentParser(description="航点补点编辑器(独立, 8099)")
    ap.add_argument("--txt", type=Path, default=DEFAULT_TXT,
                    help=f"考卷 TXT 路径, 默认 {DEFAULT_TXT}")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--yaml-out", type=Path, default=None,
                    help="自定义 yaml 输出(默认走生成器: config/subject1_waypoints.yaml)")
    args = ap.parse_args()
    TXT_PATH = args.txt.expanduser().resolve()
    YAML_OUT = args.yaml_out
    if not TXT_PATH.is_file():
        sys.exit(f"TXT 不存在: {TXT_PATH}")
    if not GENERATOR.is_file():
        sys.exit(f"找不到生成器: {GENERATOR}")
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"航点补点编辑器: http://0.0.0.0:{args.port}  考卷: {TXT_PATH}")
    print("原 TXT 只读; 确认后生成 合并TXT + yaml(自动备份旧文件)。Ctrl+C 退出。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
