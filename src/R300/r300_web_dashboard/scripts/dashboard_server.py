#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R300 Web 上位机静态文件服务器 + 轻量节点启动接口。

说明：
- 静态网页由本节点提供；
- ROS 话题通讯由浏览器通过 rosbridge 完成；
- 相机/视觉、1X惯导、纯实车导航、视觉避障导航/代价地图、MID-360点云/高程图可从网页按钮分别启动；
- 仅用于本机局域网/实验环境，不建议暴露到公网。
"""

from __future__ import print_function

import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
try:
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
except ImportError:  # Python2 fallback, normally not used on Noetic
    from SimpleHTTPServer import SimpleHTTPRequestHandler
    from SocketServer import ThreadingMixIn
    from BaseHTTPServer import HTTPServer
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

import rospy

PROC_LOCK = threading.RLock()
PROCS = {}
LOGS = {
    "camera": deque(maxlen=160),
    "ins": deque(maxlen=220),
    "real_nav": deque(maxlen=300),
    "costmap": deque(maxlen=260),
    "lidar": deque(maxlen=320),
}
TARGET_RECORD_LOCK = threading.RLock()
TARGET_RECORDING = {"enabled": False, "path": None, "rows": 0}

# 点云/代价地图脚本内部按现有方式处理 sudo；网页服务只负责分开启动进程。


def _package_www_dir():
    """Return package www directory; support both installed and source-tree use."""
    try:
        import rospkg
        pkg_dir = rospkg.RosPack().get_path("r300_web_dashboard")
        www_dir = os.path.join(pkg_dir, "www")
        if os.path.isdir(www_dir):
            return www_dir
    except Exception:
        pass
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "www"))


def _append_log(name, line):
    with PROC_LOCK:
        LOGS.setdefault(name, deque(maxlen=200)).append(
            time.strftime("%H:%M:%S ") + line.rstrip()
        )


def _reader_thread(name, proc):
    try:
        for raw in iter(proc.stdout.readline, b""):
            try:
                line = raw.decode("utf-8", "replace")
            except Exception:
                line = str(raw)
            if line.strip():
                _append_log(name, line)
    except Exception as exc:
        _append_log(name, "日志读取异常：%s" % exc)
    finally:
        code = proc.poll()
        _append_log(name, "进程结束，returncode=%s" % code)


def _is_running(proc):
    return proc is not None and proc.poll() is None


def _start_process(name, command, needs_password=False):
    """Start a long-running child process in its own process group.

    needs_password is kept for API compatibility; password handling is now
    performed inside the wrapper shell script. This avoids the fragile pattern
    of writing sudo password to a background process stdin from Python.
    """
    with PROC_LOCK:
        old = PROCS.get(name)
        if _is_running(old):
            return False, "%s 已在运行，pid=%s" % (name, old.pid)

        _append_log(name, "启动命令：%s" % command)
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("ROS_MASTER_URI", "http://localhost:11311")
        env.setdefault("ROS_HOSTNAME", "localhost")
        try:
            proc = subprocess.Popen(
                ["/bin/bash", "-lc", command],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=os.path.expanduser("~/r300_ws"),
                env=env,
                preexec_fn=os.setsid,
                bufsize=0,
            )
            PROCS[name] = proc
        except Exception as exc:
            _append_log(name, "启动失败：%s" % exc)
            return False, "启动失败：%s" % exc

        threading.Thread(target=_reader_thread, args=(name, proc), daemon=True).start()
        return True, "%s 已启动，pid=%s" % (name, proc.pid)

def _stop_process(name):
    with PROC_LOCK:
        proc = PROCS.get(name)
        if not _is_running(proc):
            return False, "%s 未运行" % name
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            _append_log(name, "已发送 SIGINT")
            return True, "%s 正在停止" % name
        except Exception as exc:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
            _append_log(name, "停止异常：%s" % exc)
            return False, "停止异常：%s" % exc



def _target_record_dir():
    path = os.path.expanduser("~/.ros/r300_web_dashboard/targets")
    os.makedirs(path, exist_ok=True)
    return path


def _target_record_status():
    with TARGET_RECORD_LOCK:
        return dict(TARGET_RECORDING)


def _start_target_recording():
    with TARGET_RECORD_LOCK:
        if TARGET_RECORDING["enabled"] and TARGET_RECORDING["path"]:
            return True, "目标本地记录已在运行", dict(TARGET_RECORDING)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(_target_record_dir(), "r300_targets_%s.csv" % stamp)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow([
                "time", "source_topic", "frame_id", "selected", "class", "confidence",
                "x_right_m", "y_down_m", "z_forward_m",
                "vehicle_lat", "vehicle_lon", "vehicle_alt", "heading_deg",
                "forward_m", "right_m", "north_m", "east_m",
                "target_lat", "target_lon"
            ])
        TARGET_RECORDING.update({"enabled": True, "path": path, "rows": 0})
        return True, "目标本地记录已开始", dict(TARGET_RECORDING)


def _stop_target_recording():
    with TARGET_RECORD_LOCK:
        was_enabled = TARGET_RECORDING["enabled"]
        TARGET_RECORDING["enabled"] = False
        msg = "目标本地记录已停止" if was_enabled else "目标本地记录未运行"
        return True, msg, dict(TARGET_RECORDING)


def _append_target_records(records):
    if not isinstance(records, list):
        return False, "records 必须是数组", _target_record_status()
    with TARGET_RECORD_LOCK:
        if not TARGET_RECORDING["enabled"] or not TARGET_RECORDING["path"]:
            return False, "目标本地记录未启动", dict(TARGET_RECORDING)
        rows = []
        for item in records[:200]:
            if not isinstance(item, dict):
                continue
            rows.append([
                item.get("time", ""), item.get("source_topic", ""), item.get("frame_id", ""),
                item.get("selected", ""), item.get("class", ""), item.get("confidence", ""),
                item.get("x", ""), item.get("y", ""), item.get("z", ""),
                item.get("vehicle_lat", ""), item.get("vehicle_lon", ""), item.get("vehicle_alt", ""),
                item.get("heading_deg", ""), item.get("forward_m", ""), item.get("right_m", ""),
                item.get("north_m", ""), item.get("east_m", ""),
                item.get("target_lat", ""), item.get("target_lon", "")
            ])
        if rows:
            with open(TARGET_RECORDING["path"], "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerows(rows)
            TARGET_RECORDING["rows"] += len(rows)
        return True, "已写入 %d 条目标" % len(rows), dict(TARGET_RECORDING)

def _process_status():
    with PROC_LOCK:
        out = {}
        for name in ("camera", "ins", "real_nav", "costmap", "lidar"):
            proc = PROCS.get(name)
            out[name] = {
                "running": bool(_is_running(proc)),
                "pid": None if proc is None else proc.pid,
                "returncode": None if proc is None else proc.poll(),
                "logs": list(LOGS.get(name, []))[-30:],
            }
        return out


class DashboardHandler(SimpleHTTPRequestHandler):
    """Static file server plus /api/start_* endpoints."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        SimpleHTTPRequestHandler.end_headers(self)

    def log_message(self, fmt, *args):
        rospy.loginfo("dashboard_server: " + fmt, *args)

    def _send_json(self, payload, code=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self, max_bytes=2 * 1024 * 1024):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            length = 0
        if length < 0 or length > max_bytes:
            raise ValueError("请求体过大")
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        if self.path.startswith("/api/process_status"):
            self._send_json({"ok": True, "processes": _process_status()})
            return
        if self.path.startswith("/api/target_record/status"):
            self._send_json({"ok": True, "recording": _target_record_status()})
            return
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/start_camera":
            cmd = "bash ~/r300_ws/src/R300/r300_web_dashboard/scripts/web_start_camera.sh"
            ok, msg = _start_process("camera", cmd, needs_password=False)
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path == "/api/start_ins":
            cmd = "bash ~/r300_ws/src/R300/r300_web_dashboard/scripts/web_start_ins.sh"
            ok, msg = _start_process("ins", cmd, needs_password=False)
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path == "/api/start_real_nav":
            with PROC_LOCK:
                visual_proc = PROCS.get("costmap")
                visual_running = _is_running(visual_proc)
            if visual_running:
                ok, msg = False, "视觉避障导航/代价地图正在运行，请先停止后再启动纯实车导航。"
            else:
                cmd = "bash ~/r300_ws/src/R300/r300_web_dashboard/scripts/web_start_real_nav.sh"
                ok, msg = _start_process("real_nav", cmd, needs_password=True)
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path in ("/api/start_costmap", "/api/start_nav"):
            with PROC_LOCK:
                real_proc = PROCS.get("real_nav")
                real_running = _is_running(real_proc)
            if real_running:
                ok, msg = False, "纯实车导航正在运行，请先停止后再启动视觉避障导航/代价地图。"
            else:
                cmd = "bash ~/r300_ws/src/R300/r300_web_dashboard/scripts/web_start_nav.sh"
                ok, msg = _start_process("costmap", cmd, needs_password=True)
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path == "/api/start_lidar":
            cmd = "bash ~/r300_ws/src/R300/r300_web_dashboard/scripts/web_start_lidar_elevation.sh"
            ok, msg = _start_process("lidar", cmd, needs_password=False)
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path == "/api/stop_camera":
            ok, msg = _stop_process("camera")
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path == "/api/stop_ins":
            ok, msg = _stop_process("ins")
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path == "/api/stop_real_nav":
            ok, msg = _stop_process("real_nav")
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path in ("/api/stop_costmap", "/api/stop_nav"):
            ok, msg = _stop_process("costmap")
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path == "/api/stop_lidar":
            ok, msg = _stop_process("lidar")
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path == "/api/target_record/start":
            ok, msg, status = _start_target_recording()
            self._send_json({"ok": ok, "message": msg, "recording": status})
            return
        if path == "/api/target_record/stop":
            ok, msg, status = _stop_target_recording()
            self._send_json({"ok": ok, "message": msg, "recording": status})
            return
        if path == "/api/target_record/append":
            try:
                body = self._read_json_body()
                ok, msg, status = _append_target_records(body.get("records", []))
                self._send_json({"ok": ok, "message": msg, "recording": status}, code=200 if ok else 409)
            except Exception as exc:
                self._send_json({"ok": False, "message": str(exc), "recording": _target_record_status()}, code=400)
            return
        self._send_json({"ok": False, "message": "unknown api: " + path}, code=404)


def main():
    rospy.init_node("r300_web_dashboard_server", anonymous=False)
    port = int(rospy.get_param("~port", 8090))
    bind = rospy.get_param("~bind", "0.0.0.0")
    www_dir = rospy.get_param("~www_dir", _package_www_dir())

    if not os.path.isdir(www_dir):
        rospy.logerr("Web directory not found: %s", www_dir)
        sys.exit(1)

    os.chdir(www_dir)
    httpd = ThreadingHTTPServer((bind, port), DashboardHandler)

    def serve():
        rospy.loginfo("R300 Web dashboard serving %s at http://%s:%d", www_dir, bind, port)
        httpd.serve_forever()

    t = threading.Thread(target=serve)
    t.daemon = True
    t.start()

    rospy.loginfo("Open browser: http://<robot-ip>:%d", port)
    rospy.spin()
    try:
        _stop_process("camera")
        _stop_process("costmap")
        _stop_process("real_nav")
        _stop_process("ins")
        _stop_process("lidar")
    except Exception:
        pass
    httpd.shutdown()


if __name__ == "__main__":
    main()
