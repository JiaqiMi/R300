#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R300 Web 上位机静态文件服务器 + 轻量节点启动接口。

说明：
- 静态网页由本节点提供；
- ROS 话题通讯由浏览器通过 rosbridge 完成；
- 相机/视觉和导航/costmap 可从网页按钮启动；
- 仅用于本机局域网/实验环境，不建议暴露到公网。
"""

from __future__ import print_function

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
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
    "nav": deque(maxlen=220),
}

# 用户要求：点云/导航脚本需要 sudo 密码。仅保存在本地网页服务进程内。
# 更安全的长期方案是配置 sudoers NOPASSWD 只允许指定脚本。
NAV_SUDO_PASSWORD = "1234\n"


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


def _process_status():
    with PROC_LOCK:
        out = {}
        for name in ("camera", "nav"):
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

    def do_GET(self):
        if self.path.startswith("/api/process_status"):
            self._send_json({"ok": True, "processes": _process_status()})
            return
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/start_camera":
            cmd = "bash ~/r300_ws/src/R300/r300_web_dashboard/scripts/web_start_camera.sh"
            ok, msg = _start_process("camera", cmd, needs_password=False)
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path == "/api/start_nav":
            cmd = "bash ~/r300_ws/src/R300/r300_web_dashboard/scripts/web_start_nav.sh"
            ok, msg = _start_process("nav", cmd, needs_password=True)
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path == "/api/stop_camera":
            ok, msg = _stop_process("camera")
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path == "/api/stop_nav":
            ok, msg = _stop_process("nav")
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
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
        _stop_process("nav")
    except Exception:
        pass
    httpd.shutdown()


if __name__ == "__main__":
    main()
