#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R300 Web 上位机静态文件服务器 + 轻量节点启动接口。

说明：
- 静态网页由本节点提供；
- ROS 话题通讯由浏览器通过 rosbridge 完成；
- 相机/视觉、1X惯导、纯实车导航、视觉避障导航、雷达避障导航、MID-360点云/高程图可从网页按钮分别启动；
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
from urllib.parse import parse_qs, urlparse
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
    "lidar_nav": deque(maxlen=300),
    "lidar": deque(maxlen=320),
    "lidar_display": deque(maxlen=220),
    "target_recorder": deque(maxlen=260),
}

# R300_LIDAR_LOG_TAIL_FALLBACK_V1
# Web 内存中仍只保留固定条数；当雷达导航 stdout 管道丢失时，
# 从磁盘日志尾部恢复显示，不参与任何导航或节点启动逻辑。
LOG_LAST_UPDATE = {}
LIDAR_NAV_LOG_FILE = os.path.expanduser(
    "~/.ros/r300_web_dashboard/web_start_lidar_nav.log"
)
LIDAR_NAV_LOG_STALE_S = 3.0
LIDAR_NAV_LOG_TAIL_LINES = 300
LIDAR_NAV_LOG_TAIL_BYTES = 512 * 1024

TARGET_RECORD_LOCK = threading.RLock()
TARGET_RECORD_ROOT = os.path.expanduser(os.environ.get("R300_TARGET_RECORD_DIR", "~/r300_target_records"))

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
        LOG_LAST_UPDATE[name] = time.time()


def _tail_log_file(path, max_lines=300, max_bytes=512 * 1024):
    # 只读取日志文件末尾，避免大日志文件整体读入内存。
    if not path or not os.path.isfile(path):
        return []

    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            start = max(0, size - max_bytes)
            handle.seek(start, os.SEEK_SET)
            raw = handle.read()

        lines = raw.decode("utf-8", "replace").splitlines()
        # 从文件中间开始时，第一行可能是不完整行。
        if start > 0 and lines:
            lines = lines[1:]
        return lines[-max_lines:]
    except Exception:
        return []


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




def _ros_nodes_snapshot():
    """Return current ROS node names without trusting Web child-process memory.

    Browser refresh does not restart ROS nodes, and the dashboard server may also
    be restarted independently.  PROCS therefore cannot be the sole source of
    truth for subsystem status.
    """
    try:
        result = subprocess.run(
            ["rosnode", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=1.5,
            universal_newlines=True,
        )
        if result.returncode != 0:
            return set()
        return {line.strip() for line in (result.stdout or "").splitlines() if line.strip()}
    except Exception:
        return set()


def _runtime_flags(nodes=None):
    """Infer subsystem state from actual ROS nodes.

    This is intentionally read-only: it never starts, stops, or restarts nodes.
    """
    nodes = _ros_nodes_snapshot() if nodes is None else set(nodes)

    camera_nodes = {
        "/r300_dual_yolo_depth_node",
        "/camera/realsense2_camera",
        "/camera/realsense2_camera_manager",
    }
    lidar_sensor_nodes = {
        "/livox_lidar_publisher2",
        "/laserMapping",
        "/elevation_mapping",
    }

    return {
        "camera": bool(nodes & camera_nodes),
        "ins": "/one_x_serial_driver" in nodes,
        "real_nav": False,
        "costmap": "/move_base" in nodes and "/vision_obstacle_layer_node" in nodes,
        "lidar_nav": "/move_base" in nodes and "/lidar_obstacle_scan_node" in nodes,
        "sign_guidance": "/direction_sign_local_goal" in nodes,
        "lidar": bool(nodes & lidar_sensor_nodes),
        "lidar_display": "/r300_lidar_web_adapter" in nodes,
        "target_recorder": "/r300_target_snapshot_recorder" in nodes,
    }


def _runtime_running(name):
    return bool(_runtime_flags().get(name, False))


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


def _run_stop_script(name, script_path, success_message, timeout_s=30):
    """Run the subsystem's official stop script, then clear its Web wrapper.

    Some R300 launch scripts start child processes that are not reliably stopped
    by signalling only the Web wrapper process group.  For 1X and pure real
    navigation, always use the official one_key stop scripts.
    """
    script_path = os.path.expanduser(script_path)
    if not os.path.isfile(script_path):
        msg = "未找到停止脚本：%s" % script_path
        _append_log(name, msg)
        return False, msg

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("ROS_MASTER_URI", "http://localhost:11311")
    env.setdefault("ROS_HOSTNAME", "localhost")
    _append_log(name, "执行停止脚本：%s" % script_path)

    ok = False
    message = ""
    try:
        result = subprocess.run(
            ["/bin/bash", script_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(script_path),
            env=env,
            timeout=timeout_s,
            universal_newlines=True,
        )
        output = result.stdout or ""
        for line in output.splitlines():
            if line.strip():
                _append_log(name, line)
        ok = result.returncode == 0
        if ok:
            message = success_message
        else:
            message = "停止脚本返回错误码 %s" % result.returncode
    except subprocess.TimeoutExpired:
        message = "停止脚本执行超时（%ss）" % timeout_s
        _append_log(name, message)
    except Exception as exc:
        message = "停止脚本执行异常：%s" % exc
        _append_log(name, message)

    # The official stop script handles ROS nodes.  This only cleans the tracked
    # Web wrapper/tee process so that the button can be used again immediately.
    with PROC_LOCK:
        proc = PROCS.get(name)
    if _is_running(proc):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        except ProcessLookupError:
            pass
        except Exception as exc:
            _append_log(name, "清理 Web 包装进程异常：%s" % exc)
    with PROC_LOCK:
        if PROCS.get(name) is proc:
            PROCS[name] = None

    return ok, message



def _target_record_paths():
    root = os.path.abspath(os.path.expanduser(TARGET_RECORD_ROOT))
    candidate_dir = os.path.join(root, "candidate_records")
    submit_dir = os.path.join(root, "submit_results")
    return {
        "root": root,
        "candidate_dir": candidate_dir,
        "submit_dir": submit_dir,
        "candidate_index": os.path.join(candidate_dir, "index.json"),
        "candidate_summary": os.path.join(candidate_dir, "summary.csv"),
        "submit_index": os.path.join(submit_dir, "index.json"),
        "submit_summary": os.path.join(submit_dir, "summary.csv"),
    }


def _read_target_index(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        records = payload.get("records", [])
        return records if isinstance(records, list) else []
    except Exception:
        return []


def _target_record_status():
    paths = _target_record_paths()
    candidate_records = _read_target_index(paths["candidate_index"])
    submit_records = _read_target_index(paths["submit_index"])
    enabled = _runtime_running("target_recorder")

    display_records = []
    for rank, record in enumerate(submit_records[:10], start=1):
        if not isinstance(record, dict):
            continue
        item = {
            "rank": rank,
            "class_id": record.get("class_id"),
            "class_name": record.get("class_name", ""),
            "confidence": record.get("confidence"),
            "target_latitude": record.get("target_latitude"),
            "target_longitude": record.get("target_longitude"),
            "gps_valid": bool(record.get("gps_valid", False)),
            "heading_valid": bool(record.get("heading_valid", False)),
            "geolocation_valid": bool(record.get("geolocation_valid", False)),
            "depth_m": record.get("depth_m"),
            "record_id": record.get("record_id", ""),
            "image_file": record.get("image_file", ""),
        }
        if item["image_file"]:
            item["image_url"] = (
                "/api/target_record/image?scope=submit&file="
                + str(item["image_file"])
            )
        display_records.append(item)

    return {
        "enabled": enabled,
        "path": paths["submit_summary"],
        "rows": len(candidate_records),
        "output_dir": paths["root"],
        "candidate_dir": paths["candidate_dir"],
        "submit_dir": paths["submit_dir"],
        "candidate_summary": paths["candidate_summary"],
        "submit_summary": paths["submit_summary"],
        "candidate_count": len(candidate_records),
        "submit_count": len(submit_records),
        "submit_records": display_records,
    }


def _start_target_recording():
    if _runtime_running("target_recorder"):
        return True, "比赛目标图片记录器已在运行", _target_record_status()
    cmd = (
        "bash ~/r300_ws/src/R300/r300_web_dashboard/"
        "scripts/web_start_target_recorder.sh"
    )
    ok, msg = _start_process("target_recorder", cmd, needs_password=False)
    if ok:
        msg = "比赛目标图片记录器正在启动；按 YAML 规则保存图片、JSON、候选库和 Top10"
    return ok, msg, _target_record_status()


def _stop_target_recording():
    if not _runtime_running("target_recorder") and not _is_running(PROCS.get("target_recorder")):
        return True, "比赛目标图片记录器未运行", _target_record_status()
    ok, msg = _run_stop_script(
        "target_recorder",
        "~/r300_ws/src/R300/r300_web_dashboard/scripts/web_stop_target_recorder.sh",
        "比赛目标图片记录器已停止；已保存结果保留在磁盘",
        timeout_s=20,
    )
    return ok, msg, _target_record_status()


def _append_target_records(records):
    # 旧版浏览器逐帧 CSV 录制已停用。正式本地记录由
    # target_snapshot_recorder.py 负责，避免每帧重复写入数千条。
    return False, "浏览器逐帧CSV本地录制已停用，请使用比赛目标图片记录器", _target_record_status()


def _safe_target_image_path(scope, filename):
    paths = _target_record_paths()
    base = paths["submit_dir"] if scope == "submit" else paths["candidate_dir"]
    base = os.path.abspath(base)
    requested = os.path.abspath(os.path.join(base, str(filename)))
    if requested != base and not requested.startswith(base + os.sep):
        return None
    if not os.path.isfile(requested):
        return None
    return requested


def _process_status():
    nodes = _ros_nodes_snapshot()
    runtime = _runtime_flags(nodes)
    result = {}
    with PROC_LOCK:
        for name in ("camera", "ins", "real_nav", "costmap", "lidar_nav", "lidar", "lidar_display", "target_recorder"):
            proc = PROCS.get(name)
            tracked_running = bool(_is_running(proc))
            runtime_running = bool(runtime.get(name, False))
            memory_logs = list(LOGS.get(name, []))
            logs = memory_logs
            log_source = "memory"

            # 雷达导航 ROS 节点仍在运行，但 Web 的启动包装进程/stdout
            # 已丢失或超过阈值没有新日志时，读取磁盘文件最后若干行。
            # 这只影响网页日志显示，不启动、不停止任何 ROS 节点。
            if name == "lidar_nav" and runtime_running:
                last_update = LOG_LAST_UPDATE.get(name, 0.0)
                memory_stale = (time.time() - last_update) > LIDAR_NAV_LOG_STALE_S
                if (not tracked_running) or (not memory_logs) or memory_stale:
                    disk_logs = _tail_log_file(
                        LIDAR_NAV_LOG_FILE,
                        max_lines=LIDAR_NAV_LOG_TAIL_LINES,
                        max_bytes=LIDAR_NAV_LOG_TAIL_BYTES,
                    )
                    if disk_logs:
                        logs = disk_logs
                        log_source = "file"

            result[name] = {
                "running": tracked_running or runtime_running,
                "tracked_running": tracked_running,
                "runtime_detected": runtime_running,
                "pid": proc.pid if tracked_running else None,
                "returncode": None if proc is None else proc.poll(),
                "logs": logs,
                "log_source": log_source,
            }

        # 指示牌引导节点由雷达导航 launch 按开关条件启动，不是独立 Web 子进程。
        # 单独返回运行状态，页面刷新后也能准确显示本次雷达导航是否启用路牌功能。
        result["sign_guidance"] = {
            "running": bool(runtime.get("sign_guidance", False)),
            "tracked_running": False,
            "runtime_detected": bool(runtime.get("sign_guidance", False)),
            "pid": None,
            "returncode": None,
            "logs": [],
        }
    return result


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

    def _send_binary_file(self, path, content_type="application/octet-stream"):
        try:
            size = os.path.getsize(path)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 256)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except Exception as exc:
            self._send_json({"ok": False, "message": str(exc)}, code=500)

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
        parsed = urlparse(self.path)
        if parsed.path == "/api/process_status":
            self._send_json({"ok": True, "processes": _process_status()})
            return
        if parsed.path == "/api/target_record/status":
            self._send_json({"ok": True, "recording": _target_record_status()})
            return
        if parsed.path == "/api/target_record/image":
            query = parse_qs(parsed.query)
            scope = str((query.get("scope") or ["submit"])[0])
            filename = str((query.get("file") or [""])[0])
            path = _safe_target_image_path(scope, filename)
            if not path:
                self._send_json({"ok": False, "message": "目标图片不存在或路径无效"}, code=404)
                return
            self._send_binary_file(path, content_type="image/jpeg")
            return
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/start_camera":
            if _runtime_running("camera"):
                ok, msg = False, "相机/视觉 ROS 节点已在运行，未重复启动"
            else:
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
                visual_running = _is_running(PROCS.get("costmap"))
                lidar_nav_running = _is_running(PROCS.get("lidar_nav"))
            if visual_running:
                ok, msg = False, "视觉避障导航/代价地图正在运行，请先停止后再启动纯实车导航。"
            elif lidar_nav_running:
                ok, msg = False, "雷达避障导航/代价地图正在运行，请先停止后再启动纯实车导航。"
            else:
                cmd = "bash ~/r300_ws/src/R300/r300_web_dashboard/scripts/web_start_real_nav.sh"
                ok, msg = _start_process("real_nav", cmd, needs_password=True)
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path in ("/api/start_costmap", "/api/start_nav"):
            with PROC_LOCK:
                real_running = _is_running(PROCS.get("real_nav"))
                lidar_nav_running = _is_running(PROCS.get("lidar_nav"))
            if real_running:
                ok, msg = False, "纯实车导航正在运行，请先停止后再启动视觉避障导航/代价地图。"
            elif lidar_nav_running:
                ok, msg = False, "雷达避障导航/代价地图正在运行，请先停止后再启动视觉避障导航。"
            else:
                cmd = "bash ~/r300_ws/src/R300/r300_web_dashboard/scripts/web_start_nav.sh"
                ok, msg = _start_process("costmap", cmd, needs_password=True)
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path == "/api/start_lidar_nav":
            try:
                body = self._read_json_body()
            except Exception as exc:
                self._send_json(
                    {"ok": False, "message": "雷达导航启动参数无效：%s" % exc,
                     "processes": _process_status()},
                    code=400,
                )
                return

            sign_guidance = body.get("sign_guidance", True)
            if not isinstance(sign_guidance, bool):
                self._send_json(
                    {"ok": False, "message": "sign_guidance 必须为 true 或 false",
                     "processes": _process_status()},
                    code=400,
                )
                return

            with PROC_LOCK:
                real_running = _is_running(PROCS.get("real_nav"))
                visual_running = _is_running(PROCS.get("costmap"))
            if real_running:
                ok, msg = False, "纯实车导航正在运行，请先停止后再启动雷达避障导航。"
            elif visual_running:
                ok, msg = False, "视觉避障导航/代价地图正在运行，请先停止后再启动雷达避障导航。"
            elif sign_guidance and not _runtime_running("camera"):
                ok, msg = False, (
                    "已勾选视觉指示牌临时转向，但相机/YOLO未运行。"
                    "请先点击“启动相机/视觉”，或取消勾选后启动纯雷达避障。"
                )
            else:
                mode = "true" if sign_guidance else "false"
                cmd = (
                    "SIGN_GUIDANCE_ENABLED=%s "
                    "bash ~/r300_ws/src/R300/r300_web_dashboard/scripts/web_start_lidar_nav.sh"
                    % mode
                )
                ok, msg = _start_process("lidar_nav", cmd, needs_password=True)
                if ok:
                    msg += "；视觉指示牌临时转向=%s" % ("开启" if sign_guidance else "关闭")
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path == "/api/start_lidar":
            if _runtime_running("lidar"):
                ok, msg = False, "雷达感知/高程 ROS 节点已在运行，未重复启动"
            else:
                cmd = "bash ~/r300_ws/src/R300/r300_web_dashboard/scripts/web_start_lidar_elevation.sh"
                ok, msg = _start_process("lidar", cmd, needs_password=False)
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path == "/api/start_lidar_display":
            if _runtime_running("lidar_display"):
                ok, msg = False, "点云/高程 Web 适配节点已在运行，未重复启动"
            else:
                cmd = "bash ~/r300_ws/src/R300/r300_web_dashboard/scripts/web_start_lidar_display.sh"
                ok, msg = _start_process("lidar_display", cmd, needs_password=False)
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path == "/api/stop_camera":
            ok, msg = _stop_process("camera")
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path == "/api/stop_ins":
            ok, msg = _run_stop_script(
                "ins",
                "~/r300_ws/src/R300/r300_web_dashboard/scripts/web_stop_ins.sh",
                "1X 惯导已通过 stop_1x.sh 停止",
            )
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path == "/api/stop_real_nav":
            ok, msg = _run_stop_script(
                "real_nav",
                "~/r300_ws/src/R300/r300_web_dashboard/scripts/web_stop_real_nav.sh",
                "纯实车导航已通过 stop_r300_nav.sh 停止，1X 保持运行",
            )
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path in ("/api/stop_costmap", "/api/stop_nav"):
            ok, msg = _stop_process("costmap")
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path == "/api/stop_lidar_nav":
            with PROC_LOCK:
                lidar_nav_running = _is_running(PROCS.get("lidar_nav"))
            if not lidar_nav_running:
                ok, msg = False, "雷达避障导航未运行"
            else:
                ok, msg = _run_stop_script(
                    "lidar_nav",
                    "~/r300_ws/src/R300/r300_web_dashboard/scripts/web_stop_lidar_nav.sh",
                    "雷达避障导航已通过 stop_r300_nav.sh 停止，1X 和雷达感知保持运行",
                )
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path == "/api/stop_lidar":
            ok, msg = _stop_process("lidar")
            self._send_json({"ok": ok, "message": msg, "processes": _process_status()})
            return
        if path == "/api/stop_lidar_display":
            ok, msg = _stop_process("lidar_display")
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
        _stop_process("lidar_nav")
        _stop_process("real_nav")
        _stop_process("ins")
        _stop_process("lidar")
        _stop_process("lidar_display")
        _stop_process("target_recorder")
    except Exception:
        pass
    httpd.shutdown()


if __name__ == "__main__":
    main()
