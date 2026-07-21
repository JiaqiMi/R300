"""
正式道路提取后台 Worker。

支持两种推理方式：
1. persistent_worker — 启动一次 SAM-Road 子进程，模型只加载一次，循环处理 tile
2. subprocess_per_tile — 每个 tile 启动一次 infer_single.py（兼容旧版）

支持三种提取模式：
1. fast_preview — 快速预览
2. roi — ROI 正式提取
3. full — 全图正式提取

功能：
- ROI 过滤 tile
- 跳过黑边/无效 tile
- 断点续跑（缓存 tile mask）
- 取消支持（含模型加载阶段）
- 分阶段进度报告
- 模型加载心跳 + 超时
- 管道阻塞防护
- 详细进度报告
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from roadnet.formal_extraction_config import (
    FormalExtractionConfig,
    EXTRACTION_MODE_FAST_PREVIEW,
    EXTRACTION_MODE_ROI,
    EXTRACTION_MODE_FULL,
    INFER_MODE_PERSISTENT,
    INFER_MODE_SUBPROCESS,
)
from roadnet.samroad_single_runner import (
    SAMRoadSingleRunResult,
    build_command,
    prepare_project_import_paths,
    prepare_runtime_env,
)
from roadnet.portable_samroadplus_runner import (
    is_portable_project,
    resolve_portable_paths,
    validate_portable_paths,
    build_portable_env,
    build_portable_command,
    normalize_output_mask,
)
from roadnet.segmentation_worker import generate_tile_grid
from roadnet.valid_image import analyze_valid_image_mask, save_valid_mask_outputs, valid_area_ratio
from roadnet.large_image_project import ImageRegionReader


# ===================================================================
# 提取阶段枚举
# ===================================================================

STAGE_PRECHECK = "precheck"
STAGE_SELECT_TILES = "select_tiles"
STAGE_LAUNCH_WORKER = "launch_worker"
STAGE_LOAD_MODEL = "load_model"
STAGE_MODEL_READY = "model_ready"
STAGE_INFER_TILES = "infer_tiles"
STAGE_MERGE_MASK = "merge_mask"
STAGE_REGISTER_MASK = "register_mask"
STAGE_FINISHED = "finished"

STAGE_ORDER = [
    STAGE_PRECHECK,
    STAGE_SELECT_TILES,
    STAGE_LAUNCH_WORKER,
    STAGE_LOAD_MODEL,
    STAGE_MODEL_READY,
    STAGE_INFER_TILES,
    STAGE_MERGE_MASK,
    STAGE_REGISTER_MASK,
    STAGE_FINISHED,
]

STAGE_LABELS = {
    STAGE_PRECHECK: "参数检查",
    STAGE_SELECT_TILES: "选择 tile",
    STAGE_LAUNCH_WORKER: "启动推理进程",
    STAGE_LOAD_MODEL: "加载模型",
    STAGE_MODEL_READY: "模型就绪",
    STAGE_INFER_TILES: "tile 推理",
    STAGE_MERGE_MASK: "合并 mask",
    STAGE_REGISTER_MASK: "注册结果",
    STAGE_FINISHED: "完成",
}


# ===================================================================
# Helpers
# ===================================================================

class ExtractionCancelled(RuntimeError):
    pass


def _read_rgb(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"无法读取影像: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_gray(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像: {path}")
    return image


def _write_image(path: str, image: np.ndarray):
    ext = Path(path).suffix or ".png"
    source = image
    if image.ndim == 3:
        source = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(ext, source)
    if not ok:
        raise IOError(f"无法编码图像: {path}")
    encoded.tofile(path)


def _point_in_polygon(px: float, py: float, polygon: list) -> bool:
    """判断点是否在多边形内部（射线法）。"""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / max(yj - yi, 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _tile_intersects_roi(tile: tuple, roi_polygons: list) -> bool:
    """判断 tile 矩形是否与任一 ROI 多边形相交。"""
    x0, y0, x1, y1 = tile
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for poly in roi_polygons:
        if not poly or len(poly) < 3:
            continue
        # 任一 tile 角点在 ROI 内
        for cx, cy in corners:
            if _point_in_polygon(cx, cy, poly):
                return True
        # ROI 角点在 tile 内
        for px, py in poly:
            if x0 <= px <= x1 and y0 <= py <= y1:
                return True
    return False


def _tile_hash_key(image_path: str, tile_bbox: tuple, model_path: str,
                   config_path: str, backbone_path: str,
                   tile_size: int, overlap: int) -> str:
    """计算 tile 缓存 hash key。"""
    parts = [
        os.path.abspath(image_path),
        f"{tile_bbox[0]}_{tile_bbox[1]}_{tile_bbox[2]}_{tile_bbox[3]}",
        os.path.abspath(model_path) if os.path.exists(str(model_path)) else str(model_path),
        os.path.abspath(config_path) if os.path.exists(str(config_path)) else str(config_path),
        os.path.abspath(backbone_path) if os.path.exists(str(backbone_path)) else str(backbone_path),
        str(tile_size),
        str(overlap),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def normalize_tile_id(value, fallback_index=None) -> str:
    """将各种 tile 标识安全归一化为 tile_XXXXXX 字符串。"""
    if isinstance(value, int):
        return f"tile_{value:06d}"
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            pass
        elif stripped.startswith("tile_"):
            return stripped
        elif stripped.isdigit():
            return f"tile_{int(stripped):06d}"
        else:
            return stripped
    if fallback_index is not None:
        try:
            return f"tile_{int(fallback_index):06d}"
        except (TypeError, ValueError):
            pass
    return "tile_unknown"


def _load_tile_id_lookup(tile_index_path: str) -> dict:
    """从 tile_index.json 构建 bbox -> tile_id 查找表。"""
    lookup = {}
    if not tile_index_path or not os.path.isfile(tile_index_path):
        return lookup
    try:
        with open(tile_index_path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        for entry in payload.get("tiles", []):
            tile_id = entry.get("tile_id")
            x0, y0, x1, y1 = (
                entry.get("x0"), entry.get("y0"),
                entry.get("x1"), entry.get("y1"),
            )
            if tile_id and None not in (x0, y0, x1, y1):
                lookup[(int(x0), int(y0), int(x1), int(y1))] = str(tile_id)
    except Exception:
        pass
    return lookup


def _make_tile_work_item(
    bbox,
    tile_idx: int,
    tile_id_lookup: Optional[dict] = None,
) -> dict:
    """将 bbox 转为带 tile_id 的工作项。"""
    raw_id = None
    if isinstance(bbox, dict):
        raw_id = bbox.get("tile_id")
        bbox_tuple = bbox.get("bbox") or (
            bbox.get("x0"), bbox.get("y0"), bbox.get("x1"), bbox.get("y1")
        )
    else:
        bbox_tuple = bbox
    x0, y0, x1, y1 = bbox_tuple
    if tile_id_lookup and raw_id is None:
        raw_id = tile_id_lookup.get((int(x0), int(y0), int(x1), int(y1)))
    return {
        "bbox": (int(x0), int(y0), int(x1), int(y1)),
        "tile_id": normalize_tile_id(raw_id, fallback_index=tile_idx),
    }


# ===================================================================
# Safety helpers — path validation, unique run_dir, writability check
# ===================================================================

def ensure_python_executable(python_exe) -> str:
    """确保 python_exe 是可执行文件，否则回退到 sys.executable。"""
    if python_exe and isinstance(python_exe, (str, Path)):
        exe_str = str(python_exe)
        if exe_str and os.path.isfile(exe_str):
            return exe_str
    # 回退到当前 Python
    fallback = sys.executable
    if fallback and os.path.isfile(fallback):
        return fallback
    raise RuntimeError(
        f"无法找到有效的 Python 解释器。\n"
        f"提供的路径: {python_exe}\n"
        f"当前 Python: {sys.executable}"
    )


def check_output_dir_writable(output_dir) -> Tuple[bool, str]:
    """全面检查输出目录是否可写。

    检查：
    1. 目录是否存在（不存在则创建）
    2. 是否能写入临时文件
    3. 是否能读取临时文件
    4. 是否能删除临时文件
    5. 是否能创建子目录
    6. 是否能删除子目录

    返回 (是否可用, 错误消息)
    """
    dir_path = Path(output_dir)
    try:
        os.makedirs(str(dir_path), exist_ok=True)
    except PermissionError as e:
        return False, f"无法创建输出目录 {dir_path}: 权限不足 ({e})"
    except OSError as e:
        return False, f"无法创建输出目录 {dir_path}: {e}"

    try:
        # 1. 写入文件
        test_file = dir_path / "_writable_test_.tmp"
        with open(str(test_file), "w", encoding="utf-8") as f:
            f.write("test")
        # 2. 读取文件
        with open(str(test_file), "r", encoding="utf-8") as f:
            if f.read() != "test":
                return False, f"读取验证失败: {test_file}"
        # 3. 删除文件
        test_file.unlink()

        # 4. 创建子目录
        test_subdir = dir_path / "_subdir_test_"
        test_subdir.mkdir(exist_ok=True)
        # 5. 删除子目录
        test_subdir.rmdir()

        return True, ""

    except PermissionError as e:
        return False, f"输出目录权限不足 ({dir_path}): {e}"
    except OSError as e:
        return False, f"输出目录 I/O 错误 ({dir_path}): {e}"


def safe_create_output_dir(base_dir, mode: str) -> str:
    """创建唯一输出目录，如已存在则递增编号，不强制删除旧目录。

    命名规则: <base_dir>/formal_<mode>_<timestamp>
    已存在时: formal_<mode>_<timestamp>_001, _002, ...

    返回最终目录路径。
    """
    base = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base / f"formal_{mode}_{timestamp}"

    # 如果不存在，直接使用
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return str(candidate)

    # 存在则递增编号
    for i in range(1, 1000):
        candidate = base / f"formal_{mode}_{timestamp}_{i:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return str(candidate)

    raise RuntimeError(
        f"无法创建唯一输出目录，已尝试 1000 个编号。\n"
        f"基础目录: {base}\n"
        f"建议清理旧输出目录或更换输出位置。"
    )


def validate_paths_for_extraction(cfg) -> list[str]:
    """正式提取启动前验证所有关键路径，返回错误列表。"""
    # Portable 工程使用独立的校验逻辑（不依赖旧版 infer_script / backbone）。
    if is_portable_project(cfg.project_dir):
        python_exe = ensure_python_executable(cfg.python_executable)
        errors = validate_portable_paths(
            python_exe, cfg.project_dir, str(cfg.output_dir)
        )
        image_path = str(cfg.image_path) if cfg.image_path else ""
        if not image_path or not os.path.isfile(image_path):
            errors.append(f"输入图像不存在: {cfg.image_path}")
        return errors

    errors: list[str] = []

    python_exe = ensure_python_executable(cfg.python_executable)
    if not python_exe or not os.path.isfile(python_exe):
        errors.append(f"Python 解释器无效: {cfg.python_executable}")

    infer_script = str(cfg.infer_script) if cfg.infer_script else ""
    if not infer_script or not os.path.isfile(infer_script):
        errors.append(f"推理脚本不存在: {cfg.infer_script}")

    project_dir = str(cfg.project_dir) if cfg.project_dir else ""
    if not project_dir or not os.path.isdir(project_dir):
        errors.append(f"SAM-Road 项目目录不存在: {cfg.project_dir}")

    config_path = str(cfg.config_path) if cfg.config_path else ""
    if config_path and not os.path.isfile(config_path):
        errors.append(f"配置文件不存在: {cfg.config_path}")

    ckpt_path = str(cfg.samroad_model_ckpt_path) if cfg.samroad_model_ckpt_path else ""
    if ckpt_path and not os.path.isfile(ckpt_path):
        errors.append(f"模型权重不存在: {cfg.samroad_model_ckpt_path}")

    backbone = str(cfg.sam_backbone_ckpt_path) if cfg.sam_backbone_ckpt_path else ""
    if backbone and not os.path.isfile(backbone):
        errors.append(f"SAM Backbone 不存在: {cfg.sam_backbone_ckpt_path}")

    image_path = str(cfg.image_path) if cfg.image_path else ""
    if not image_path or not os.path.isfile(image_path):
        errors.append(f"输入图像不存在: {cfg.image_path}")

    # 检查 python_exe 不是目录
    if python_exe and os.path.isdir(python_exe):
        errors.append(f"Python 解释器路径是目录而非文件: {python_exe}")

    return errors


def write_error_log(output_dir: str, exc_info, stage: str = "",
                    current_tile_id: str = "", current_tile_idx: Optional[int] = None,
                    operation: str = "", failed_path: str = "",
                    worker_mode: str = "", model_load_state: str = "",
                    original_error_type: str = "",
                    original_error_message: str = "",
                    original_traceback: str = "") -> str:
    """写入结构化错误日志到 formal_extraction_error.log。

    返回日志文件路径。日志写入本身不得抛异常。
    """
    log_path = os.path.join(output_dir, "formal_extraction_error.log")
    tb = original_traceback or traceback.format_exc()
    etype, e, _ = exc_info if isinstance(exc_info, tuple) else (type(exc_info), exc_info, None)
    err_type = original_error_type or str(
        etype.__name__ if hasattr(etype, "__name__") else type(etype)
    )
    err_message = original_error_message or str(e)

    safe_tile_id = normalize_tile_id(
        current_tile_id,
        fallback_index=current_tile_idx if isinstance(current_tile_idx, int) else None,
    )

    report = {
        "original_error_type": err_type,
        "original_error_message": err_message,
        "original_traceback": tb,
        "error_type": err_type,
        "error_message": err_message,
        "failed_stage": stage or "unknown",
        "failed_path": failed_path or "",
        "output_dir": str(output_dir),
        "current_tile_id": safe_tile_id,
        "current_tile_idx": current_tile_idx,
        "current_operation": operation or "",
        "worker_mode": worker_mode or "",
        "model_load_state": model_load_state or "",
        "python_executable": sys.executable,
        "platform": sys.platform,
        "timestamp": datetime.now().isoformat(),
        "traceback": tb,
    }
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n\n--- RAW TRACEBACK ---\n")
            f.write(tb)
    except Exception as write_err:
        print(f"[FATAL] 无法写入错误日志: {write_err}", file=sys.stderr)
        print(tb, file=sys.stderr)

    return log_path


def write_startup_log(output_dir: str, startup_info: dict) -> str:
    """写入启动日志 formal_extraction_startup.log。"""
    log_path = os.path.join(output_dir, "formal_extraction_startup.log")
    os.makedirs(output_dir, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        for key, value in startup_info.items():
            f.write(f"{key}: {value}\n")
    return log_path


# ===================================================================
# Persistent Infer Process Manager (with heartbeat + timeout)
# ===================================================================

class PersistentInferProcess:
    """管理一个长期运行的 SAM-Road 子进程。

    特性：
    - 独立线程读取 stdout/stderr，防止管道阻塞
    - 模型加载心跳 + 超时检测
    - 支持在加载阶段取消
    """

    def __init__(self, python_exe: str, script_path: str, config_path: str,
                 checkpoint_path: str, device: str = "cuda",
                 cancel_event: Optional[threading.Event] = None,
                 log_callback=None, heartbeat_callback=None):
        self._python = ensure_python_executable(python_exe)
        self._script = script_path
        self._config = config_path
        self._checkpoint = checkpoint_path
        self._device = device
        self._process = None
        self._ready = threading.Event()
        self._error_event = threading.Event()
        self._model_load_time = 0.0
        self._cancel = cancel_event or threading.Event()
        self._log = log_callback
        self._heartbeat = heartbeat_callback

        # stdout/stderr 读取线程
        self._stdout_thread = None
        self._stderr_thread = None

        # 消息队列（从 stdout 线程到主逻辑）
        self._msg_queue: queue.Queue = queue.Queue()

        # 收集的 stderr 内容
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def start(self, timeout_seconds: float = 180.0, heartbeat_interval: float = 2.0):
        """启动持久化推理进程，等待模型加载完成。

        Args:
            timeout_seconds: 模型加载超时（秒），0 表示无限等待
            heartbeat_interval: 心跳检查间隔
        """
        # ── 路径预检 ──
        checks = []
        if not os.path.isfile(self._python):
            checks.append(f"python_exe 不存在或不是文件: {self._python}")
        if not os.path.isfile(self._script):
            checks.append(f"持久化推理脚本不存在: {self._script}")
        if self._config and not os.path.isfile(self._config):
            checks.append(f"配置文件不存在: {self._config}")
        if self._checkpoint and not os.path.isfile(self._checkpoint):
            checks.append(f"模型权重不存在: {self._checkpoint}")
        if checks:
            raise RuntimeError(
                "持久化推理进程启动失败 — 路径无效:\n  " + "\n  ".join(checks)
            )

        t0 = time.perf_counter()
        cmd = [
            self._python,
            "-u", self._script,
            "--config", self._config,
            "--checkpoint", self._checkpoint,
            "--device", self._device,
        ]

        if self._log:
            self._log(f"启动命令: {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except PermissionError:
            raise RuntimeError(
                f"无法启动持久化推理进程 (权限不足):\n"
                f"  Python: {self._python}\n"
                f"  Script: {self._script}\n"
                f"  Config: {self._config}\n"
                f"  Checkpoint: {self._checkpoint}\n"
                f"  请检查 Python 路径是否为文件、杀毒软件是否拦截。"
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"无法启动持久化推理进程 (文件未找到):\n"
                f"  Python: {self._python}\n"
                f"  Script: {self._script}"
            )

        # 启动 stdout/stderr 读取线程（防管道阻塞）
        self._stdout_thread = threading.Thread(
            target=self._read_stdout, daemon=True, name="persist_stdout_reader"
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, daemon=True, name="persist_stderr_reader"
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        # ── 等待 ready 信号（带心跳和超时）──
        last_heartbeat = time.perf_counter()
        last_log_time = time.perf_counter()

        while not self._ready.is_set() and not self._error_event.is_set():
            # 检查取消
            if self._cancel.is_set():
                self._emit_stderr_diag()
                raise ExtractionCancelled("模型加载阶段被用户取消")

            # 检查进程是否意外退出
            if self._process.poll() is not None:
                rc = self._process.returncode
                with self._stderr_lock:
                    stderr_tail = "\n".join(self._stderr_lines[-20:]) if self._stderr_lines else "(无 stderr)"
                raise RuntimeError(
                    f"持久化推理进程意外退出 (exit code={rc}).\n"
                    f"最后 stderr 输出:\n{stderr_tail}"
                )

            # 尝试从队列获取消息
            try:
                msg = self._msg_queue.get(timeout=heartbeat_interval)
                if msg.get("type") == "ready":
                    self._model_load_time = msg.get("model_load_time_s",
                                                    time.perf_counter() - t0)
                    self._ready.set()
                    break
                elif msg.get("type") == "error":
                    self._error_event.set()
                    raise RuntimeError(
                        f"Persistent infer failed: {msg.get('message', 'unknown error')}"
                    )
                elif msg.get("type") == "status":
                    # 心跳消息
                    now = time.perf_counter()
                    elapsed = now - t0
                    step = msg.get("step", "?")
                    message = msg.get("message", "")
                    if self._log:
                        self._log(f"[{step}] {message} (已用时 {elapsed:.1f}s)")
                    if self._heartbeat:
                        self._heartbeat(elapsed, step, message)
                    last_heartbeat = now

            except queue.Empty:
                pass  # 超时，检查心跳

            # 心跳超时检查
            now = time.perf_counter()
            elapsed = now - t0

            if timeout_seconds > 0 and elapsed > timeout_seconds:
                self._emit_stderr_diag()
                raise TimeoutError(
                    f"模型加载超时 ({timeout_seconds}s).\n"
                    f"进程状态: {'运行中' if self._process.poll() is None else f'已退出(code={self._process.returncode})'}\n"
                    f"请检查 Python 环境、CUDA、checkpoint 或日志。"
                )

            # 定期心跳日志（即使子进程没有输出）
            if now - last_log_time >= heartbeat_interval:
                if self._log:
                    self._log(f"等待模型加载... 已用时 {elapsed:.1f}s")
                if self._heartbeat:
                    self._heartbeat(elapsed, "waiting", "等待模型加载中...")
                last_log_time = now

    @property
    def model_load_time(self) -> float:
        return self._model_load_time

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    def infer_tile(self, image_path: str, output_dir: str, request_id: str = "",
                   timeout: float = 300.0) -> dict:
        """发送推理请求并读取响应。"""
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("Persistent infer process is not running")

        request = {
            "action": "infer",
            "image_path": image_path,
            "output_dir": output_dir,
            "request_id": request_id or os.path.basename(image_path),
        }
        self._process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self._process.stdin.flush()

        t0 = time.perf_counter()
        while True:
            if self._cancel.is_set():
                raise ExtractionCancelled("tile 推理被取消")

            if self._process.poll() is not None:
                self._emit_stderr_diag()
                raise RuntimeError(
                    f"Persistent infer process exited (code={self._process.returncode}) "
                    f"during inference of {request_id}"
                )

            try:
                msg = self._msg_queue.get(timeout=1.0)
                if msg.get("type") in ("result", "error"):
                    return msg
            except queue.Empty:
                if timeout > 0 and (time.perf_counter() - t0) > timeout:
                    raise TimeoutError(f"tile 推理超时 ({timeout}s): {request_id}")

    def ping(self) -> bool:
        """发送 ping 检查进程是否存活。"""
        if self._process is None or self._process.poll() is not None:
            return False
        try:
            self._process.stdin.write(json.dumps({"action": "ping"}) + "\n")
            self._process.stdin.flush()
        except Exception:
            return False

        # 等待 pong
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < 10:
            try:
                msg = self._msg_queue.get(timeout=1.0)
                if msg.get("type") == "pong":
                    return True
            except queue.Empty:
                if self._process.poll() is not None:
                    return False
        return False

    def shutdown(self):
        """发送 shutdown 命令并等待进程退出。安全处理各种异常。"""
        if self._process is None:
            return
        proc = self._process
        self._process = None
        try:
            if proc.poll() is not None:
                return  # 已退出
            # 发送 shutdown 命令
            try:
                proc.stdin.write(json.dumps({"action": "shutdown"}) + "\n")
                proc.stdin.flush()
                proc.stdin.close()
            except Exception:
                pass
            # 先温和等待
            try:
                proc.wait(timeout=10)
                return
            except subprocess.TimeoutExpired:
                pass
            # 强制终止
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        finally:
            # 确保 stdio 关闭
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    # ------------------------------------------------------------------
    # 内部 — stdio 读取线程
    # ------------------------------------------------------------------

    def _read_stdout(self):
        """持续读取子进程 stdout 到消息队列。"""
        try:
            for line in self._process.stdout:
                if self._cancel.is_set():
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    self._msg_queue.put(msg)
                except json.JSONDecodeError:
                    # 非 JSON 行记录为原始日志
                    self._msg_queue.put({
                        "type": "raw_stdout",
                        "line": line[:500],
                    })
        except (ValueError, OSError):
            pass  # stream closed

    def _read_stderr(self):
        """持续读取子进程 stderr。"""
        try:
            for line in self._process.stderr:
                if self._cancel.is_set():
                    break
                line_stripped = line.strip()
                if line_stripped:
                    with self._stderr_lock:
                        self._stderr_lines.append(line_stripped)
                    # 限制 stderr 历史行数
                    with self._stderr_lock:
                        if len(self._stderr_lines) > 200:
                            self._stderr_lines[:] = self._stderr_lines[-200:]
        except (ValueError, OSError):
            pass

    def _emit_stderr_diag(self):
        """输出 stderr 诊断信息到 log。"""
        with self._stderr_lock:
            lines = self._stderr_lines[-30:] if self._stderr_lines else []
        if lines and self._log:
            self._log(f"[stderr 最近输出]\n" + "\n".join(lines))


# ===================================================================
# Main Worker
# ===================================================================

class FormalExtractionWorker(QObject):
    """正式道路提取后台 Worker。

    Signals:
        stage_progress — (stage_str, stage_label, elapsed_seconds, detail_dict)
        heartbeat — (elapsed_seconds, step, message)  模型加载心跳
        progress — (percent, current_tile, total_tiles, detail_dict)
        log — (message_str)
        finished — (result_dict)
        failed — (error_message, error_log_path)
        cancelled — (message)
    """

    # 阶段信号: stage, label, elapsed, detail
    stage_progress = Signal(str, str, float, object)

    # 模型加载心跳: elapsed, step, message
    heartbeat = Signal(float, str, str)

    # tile 进度: percent, current, total, detail
    progress = Signal(int, int, int, object)

    log = Signal(str)
    finished = Signal(object)    # result dict
    failed = Signal(str, str)    # error message, log path
    cancelled = Signal(str)

    def __init__(self, config: FormalExtractionConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._cancel_flag = threading.Event()
        self._persistent_proc = None
        self._logs = []
        self._start_time = 0.0
        self._model_load_time = 0.0
        self._total_infer_time = 0.0
        self._io_time = 0.0

        # 阶段计时
        self._stage_times: dict[str, float] = {}
        self._current_stage = ""
        self._heartbeat_count = 0
        self.current_tile_idx: int = 0
        self.current_tile_id: str = ""
        self._model_load_state: str = "not_started"
        self._portable_mode: bool = False

    def cancel(self):
        self._cancel_flag.set()
        if self._persistent_proc and self._persistent_proc.is_alive():
            self._persistent_proc.shutdown()

    def _check_cancelled(self):
        if self._cancel_flag.is_set():
            raise ExtractionCancelled("用户取消正式道路提取")

    def _emit_log(self, message: str):
        line = f"[正式提取] {message}"
        self._logs.append(line)
        self.log.emit(line)

    def _emit_stage(self, stage: str, detail: Optional[dict] = None):
        """发射阶段变更信号。"""
        self._current_stage = stage
        elapsed = time.perf_counter() - self._start_time
        label = STAGE_LABELS.get(stage, stage)
        self.stage_progress.emit(stage, label, round(elapsed, 2),
                                 detail or {})

    def _stage_start(self, stage: str) -> float:
        """记录阶段开始时间，返回 start_time。"""
        t = time.perf_counter()
        self._stage_times[f"_stage_{stage}_start"] = t
        self._emit_stage(stage)
        return t

    def _stage_end(self, stage: str, start_t: float):
        """记录阶段结束时间并存储耗时。"""
        elapsed = time.perf_counter() - start_t
        self._stage_times[stage] = round(elapsed, 3)
        self._emit_log(f"阶段 [{STAGE_LABELS.get(stage, stage)}] 完成，耗时 {elapsed:.1f}s")

    def _on_persistent_heartbeat(self, elapsed: float, step: str, message: str):
        """持久化进程模型加载心跳回调。"""
        self._heartbeat_count += 1
        self.heartbeat.emit(elapsed, step, message)

    # ===================================================================
    # Main run
    # ===================================================================

    @Slot()
    def run(self):
        self._start_time = time.perf_counter()
        cfg = self.config
        result_base = {
            "mode": cfg.extraction_mode,
            "infer_mode": cfg.infer_mode,
            "tile_size": cfg.tile_size,
            "overlap": cfg.tile_overlap,
            "total_tiles": 0,
            "processed_tiles": 0,
            "skipped_tiles": 0,
            "cache_hit_tiles": 0,
            "failed_tiles": 0,
            "cancelled": False,
            "warmup_only": cfg.warmup_only,
            "test_mode": cfg.max_tiles_for_test > 0,
            "stage_times": {},
            "worker_mode": cfg.infer_mode,
            "model_ready": False,
            "heartbeat_count": 0,
            "timeout": False,
        }

        # ══════════════════════════════════════════════════════════════
        # 阶段 precheck — 输出目录 + 路径验证
        # ══════════════════════════════════════════════════════════════
        t_precheck = self._stage_start(STAGE_PRECHECK)

        ok, err_msg = check_output_dir_writable(str(cfg.output_dir))
        if not ok:
            error_log = write_error_log(
                str(cfg.output_dir),
                RuntimeError(err_msg),
                stage=STAGE_PRECHECK,
                operation="check_output_dir_writable",
                failed_path=str(cfg.output_dir),
            )
            self.failed.emit(
                f"输出目录不可写: {err_msg}\n日志: {error_log}",
                error_log,
            )
            return

        path_errors = validate_paths_for_extraction(cfg)
        if path_errors:
            err_full = "正式提取无法启动 — 关键路径无效:\n  " + "\n  ".join(path_errors)
            error_log = write_error_log(
                str(cfg.output_dir),
                RuntimeError(err_full),
                stage=STAGE_PRECHECK,
                operation="validate_paths_for_extraction",
            )
            self.failed.emit(err_full, error_log)
            return

        # ── 适配器类型判定（Portable vs 旧版 SAM-Road）──
        self._portable_mode = is_portable_project(cfg.project_dir)
        adapter_type = "samroadplus_portable" if self._portable_mode else "old_samroad"
        result_base["adapter_type"] = adapter_type
        # Portable 第一版强制 subprocess_per_tile，禁止走旧持久化 worker。
        if self._portable_mode and cfg.infer_mode == INFER_MODE_PERSISTENT:
            cfg.infer_mode = INFER_MODE_SUBPROCESS

        # 写入启动日志
        startup_info = {
            "adapter_type": adapter_type,
            "python_exe": ensure_python_executable(cfg.python_executable),
            "project_dir": cfg.project_dir,
            "infer_script": cfg.infer_script,
            "config_file": cfg.config_path,
            "checkpoint_file": cfg.samroad_model_ckpt_path,
            "sam_backbone_ckpt_path": cfg.sam_backbone_ckpt_path,
            "device": cfg.device,
            "cwd": cfg.project_dir,
            "output_dir": cfg.output_dir,
            "extraction_mode": cfg.extraction_mode,
            "infer_mode": cfg.infer_mode,
            "tile_size": cfg.tile_size,
            "tile_overlap": cfg.tile_overlap,
            "max_tiles": cfg.max_tiles,
            "max_tiles_for_test": cfg.max_tiles_for_test,
            "model_load_timeout_seconds": cfg.model_load_timeout_seconds,
            "warmup_only": cfg.warmup_only,
            "timestamp": datetime.now().isoformat(),
        }
        write_startup_log(str(cfg.output_dir), startup_info)

        # 在实时日志中打印后端关键信息（便于确认未回退旧版）
        self._emit_log("=" * 50)
        self._emit_log(f"adapter_type = {adapter_type}")
        self._emit_log(f"project_dir = {cfg.project_dir}")
        self._emit_log(f"infer_script = {cfg.infer_script}")
        self._emit_log(f"config_file = {cfg.config_path}")
        self._emit_log(f"checkpoint_file = {cfg.samroad_model_ckpt_path}")
        self._emit_log(f"device = {cfg.device}")
        self._emit_log(f"cwd = {cfg.project_dir}")
        self._emit_log("=" * 50)
        if not self._portable_mode:
            self._emit_log(
                "警告：未检测到 SAMRoad++ Portable 工程，将使用旧版 SAM-Road 入口。"
            )

        self._stage_end(STAGE_PRECHECK, t_precheck)

        try:
            os.makedirs(cfg.output_dir, exist_ok=True)

            # ── 快速预览模式 ──
            if cfg.extraction_mode == EXTRACTION_MODE_FAST_PREVIEW:
                self._run_fast_preview()
                result_base["processed_tiles"] = 1
                result_base["elapsed_seconds"] = round(time.perf_counter() - self._start_time, 3)
                result_base["stage_times"] = self._stage_times
                self.finished.emit(result_base)
                return

            # ══════════════════════════════════════════════════════════
            # 阶段 select_tiles — 读取图像 + 生成 tile 网格
            # ══════════════════════════════════════════════════════════
            t_select = self._stage_start(STAGE_SELECT_TILES)

            reader = ImageRegionReader(str(cfg.image_path))
            width, height = reader.size
            cfg.image_width, cfg.image_height = width, height
            self._emit_log(f"图像尺寸: {width} x {height}")
            self._emit_log(f"提取模式: {cfg.extraction_mode}")
            self._emit_log(f"推理方式: {cfg.infer_mode}")
            self._emit_log(f"tile_size={cfg.tile_size}, overlap={cfg.tile_overlap}")

            # ROI 模式必须显式提供 ROI，绝不允许退化为全图提取。
            if cfg.extraction_mode == EXTRACTION_MODE_ROI and not cfg.roi_polygons:
                raise ValueError("ROI 正式提取需要至少一个启用的 ROI 区域")

            # ── 有效区域分析 ──
            preview = reader.read_preview(3000)
            valid_mask, valid_mask_report = analyze_valid_image_mask(
                preview, cfg.black_threshold,
                max(64, int(cfg.min_black_component_area * (preview.shape[1] / width) ** 2)),
            )
            valid_mask = cv2.resize(valid_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            ratio = valid_area_ratio(valid_mask)
            self._emit_log(f"有效区域比例: {ratio:.4f}")

            if cfg.debug_mode:
                save_valid_mask_outputs(str(cfg.output_dir), valid_mask, valid_mask_report)

            # ── 生成 tile 网格 ──
            candidates = generate_tile_grid(width, height, cfg.tile_size, cfg.tile_overlap)
            self._emit_log(f"候选 tile 数量: {len(candidates)}")

            # ── 过滤 tile ──
            tiles = []
            skipped_black = 0
            skipped_roi = 0

            for candidate in candidates:
                x0, y0, x1, y1 = candidate

                # 1. 黑色 tile 过滤
                if cfg.skip_black_tile:
                    tile_valid = valid_mask[y0:y1, x0:x1]
                    tile_valid_ratio = float(np.count_nonzero(tile_valid)) / float(tile_valid.size)
                    if tile_valid_ratio < cfg.valid_pixel_ratio_threshold:
                        skipped_black += 1
                        continue

                # 2. ROI 过滤
                if cfg.extraction_mode == EXTRACTION_MODE_ROI:
                    if not _tile_intersects_roi(candidate, cfg.roi_polygons):
                        skipped_roi += 1
                        continue

                tiles.append(candidate)

            # max_tiles / max_tiles_for_test 限制
            effective_max = None
            if cfg.max_tiles_for_test > 0:
                effective_max = cfg.max_tiles_for_test
            elif cfg.max_tiles and cfg.max_tiles > 0:
                effective_max = cfg.max_tiles

            if effective_max is not None and len(tiles) > effective_max:
                self._emit_log(f"限制 tile 数量: {len(tiles)} → {effective_max}" +
                               (" (测试模式)" if cfg.max_tiles_for_test > 0 else ""))
                tiles = tiles[:effective_max]

            tile_index_path = str(cfg.tile_index_path) if getattr(cfg, "tile_index_path", None) else ""
            tile_id_lookup = _load_tile_id_lookup(tile_index_path)
            if tile_id_lookup:
                self._emit_log(f"已加载 tile_index.json，复用 {len(tile_id_lookup)} 个 tile_id")

            tile_work_items = [
                _make_tile_work_item(bbox, tile_idx, tile_id_lookup)
                for tile_idx, bbox in enumerate(tiles, start=1)
            ]

            result_base["total_tiles"] = len(tile_work_items)
            result_base["skipped_candidate_tiles"] = len(candidates) - len(tiles)

            self._emit_log(
                f"有效 tile 数量: {len(tile_work_items)} "
                f"(跳过 黑色={skipped_black}, ROI外={skipped_roi})"
            )

            self._stage_end(STAGE_SELECT_TILES, t_select)

            if len(tile_work_items) == 0:
                raise ValueError("没有有效 tile；请检查 ROI 或黑色阈值")

            # ══════════════════════════════════════════════════════════
            # 阶段 launch_worker + load_model
            # ══════════════════════════════════════════════════════════

            # Portable 工程：第一版强制走 subprocess_per_tile，不启动旧版
            # persistent_samroad_infer.py（避免 from model import SAMRoad）。
            # self._portable_mode 已在 precheck 阶段完成判定。
            if self._portable_mode:
                self._emit_log(
                    "检测到 SAMRoad++ Portable 工程，使用 infer.py 子进程模式"
                )
                self._model_load_state = "portable_subprocess"
                self._emit_stage(STAGE_MODEL_READY)
            elif cfg.infer_mode == INFER_MODE_PERSISTENT:
                # 启动持久化进程（带心跳和超时）
                self._start_persistent_process_with_heartbeat()
            else:
                self._emit_stage(STAGE_MODEL_READY)

            result_base["model_ready"] = True
            result_base["heartbeat_count"] = self._heartbeat_count

            # ── 预热模式：加载完模型后直接返回 ──
            if cfg.warmup_only:
                self._emit_log("模型预热完成，worker 保持常驻供后续复用。")
                if self._persistent_proc:
                    self._persistent_proc.shutdown()
                elapsed = time.perf_counter() - self._start_time
                result_base["elapsed_seconds"] = round(elapsed, 3)
                result_base["stage_times"] = self._stage_times
                result_base["model_load_time_seconds"] = round(self._model_load_time, 3)
                self._emit_stage(STAGE_FINISHED)
                self.finished.emit(result_base)
                return

            # ══════════════════════════════════════════════════════════
            # 阶段 infer_tiles
            # ══════════════════════════════════════════════════════════
            t_infer_stage = self._stage_start(STAGE_INFER_TILES)

            # ── 准备输出目录 ──
            tiles_root = os.path.join(cfg.output_dir, "tiles")
            os.makedirs(tiles_root, exist_ok=True)

            # ── 环境变量 ──
            env = os.environ.copy()
            env.update(prepare_runtime_env(str(cfg.output_dir)))
            env["PYTHONUNBUFFERED"] = "1"
            if self._portable_mode:
                # Portable：PYTHONPATH 指向 portable 工程与 sam 目录，
                # 严禁从 roadnet_tool 目录查找 infer.py / model.py。
                env = build_portable_env(env, cfg.project_dir)
            else:
                project_path = prepare_project_import_paths(str(cfg.project_dir))
                env["PYTHONPATH"] = project_path + (
                    os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else ""
                )

            # ── 初始化合并数组 ──
            merged = np.zeros((height, width), dtype=np.uint8)
            sum_mask = None
            weights = None
            if cfg.merge_method == "average":
                sum_mask = np.zeros((height, width), dtype=np.float32)
                weights = np.zeros((height, width), dtype=np.uint16)

            # ── 逐 tile 处理 ──
            processed = 0
            cache_hits = 0
            failed_tile_indices = []
            failed_tile_details = []

            total = len(tile_work_items)
            infer_start_offset = time.perf_counter()

            for tile_idx, tile_item in enumerate(tile_work_items, start=1):
                self._check_cancelled()
                self.current_tile_idx = tile_idx
                tile_id = tile_item["tile_id"]
                self.current_tile_id = tile_id
                x0, y0, x1, y1 = tile_item["bbox"]
                cache_key = _tile_hash_key(
                    str(cfg.image_path), (x0, y0, x1, y1),
                    str(cfg.samroad_model_ckpt_path), str(cfg.config_path),
                    str(cfg.sam_backbone_ckpt_path), cfg.tile_size, cfg.tile_overlap,
                )

                # ── 缓存检查 ──
                cache_mask_path = os.path.join(tiles_root, f"{tile_id}_mask.png")
                cache_meta_path = os.path.join(tiles_root, f"{tile_id}_metadata.json")
                cache_hit = False

                if cfg.resume_from_existing_tiles and os.path.isfile(cache_mask_path):
                    try:
                        tile_mask = _read_gray(cache_mask_path)
                        if tile_mask.shape == (y1 - y0, x1 - x0):
                            cache_hit = True
                    except Exception:
                        pass

                if cache_hit:
                    cache_hits += 1
                    self._emit_log(f"复用已有 tile 结果: {tile_id}")
                    tile_mask = _read_gray(cache_mask_path)
                else:
                    # ── 读取 tile 图像 ──
                    t_io = time.perf_counter()
                    tile_rgb = reader.read_region(x0, y0, x1, y1)
                    tile_valid = valid_mask[y0:y1, x0:x1]
                    tile_rgb[tile_valid == 0] = 0

                    if self._portable_mode:
                        # Portable 每 tile 一个独立子目录（含 image/mask/日志/元数据）
                        tile_output = os.path.join(tiles_root, tile_id)
                        os.makedirs(tile_output, exist_ok=True)
                        tile_path = os.path.join(tile_output, "tile_image.png")
                    else:
                        tile_path = os.path.join(tiles_root, f"{tile_id}_input.png")
                        tile_output = os.path.join(tiles_root, f"{tile_id}_output")
                        os.makedirs(tile_output, exist_ok=True)
                    _write_image(tile_path, tile_rgb)

                    self._io_time += time.perf_counter() - t_io

                    # ── 推理 ──
                    t_infer = time.perf_counter()

                    if self._portable_mode:
                        success = self._infer_tile_portable(
                            cfg, env, tile_id, tile_idx, tile_path, tile_output,
                            [x0, y0, x1, y1], failed_tile_details,
                        )
                    elif cfg.infer_mode == INFER_MODE_PERSISTENT:
                        try:
                            result = self._persistent_proc.infer_tile(
                                str(tile_path), str(tile_output), request_id=tile_id
                            )
                            success = result.get("success", False)
                            if not success:
                                raise RuntimeError(result.get("message", "未知错误"))
                        except Exception as e:
                            success = False
                            failed_tile_details.append({
                                "tile": tile_idx, "tile_id": tile_id, "rect": [x0, y0, x1, y1],
                                "error": str(e),
                            })
                    else:
                        # subprocess_per_tile（兼容旧版）
                        temp_config = type('_Temp', (), {})()
                        temp_config.project_dir = cfg.project_dir
                        temp_config.infer_script = cfg.infer_script
                        temp_config.config_path = cfg.config_path
                        temp_config.samroad_model_ckpt_path = cfg.samroad_model_ckpt_path
                        temp_config.input_image = Path(tile_path)
                        temp_config.device = cfg.device
                        temp_config.mask_only_partial_load = cfg.mask_only_partial_load
                        temp_config.dry_run = False

                        command = build_command(temp_config, str(tile_output))
                        try:
                            process = subprocess.run(
                                command, cwd=str(cfg.project_dir),
                                env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace",
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                            )
                            success = (process.returncode == 0)
                            if not success:
                                failed_tile_details.append({
                                    "tile": tile_idx, "tile_id": tile_id, "rect": [x0, y0, x1, y1],
                                    "return_code": process.returncode,
                                    "error": process.stderr[-500:] if process.stderr else "无错误输出",
                                })
                        except Exception as e:
                            success = False
                            failed_tile_details.append({
                                "tile": tile_idx, "tile_id": tile_id, "rect": [x0, y0, x1, y1],
                                "error": str(e),
                            })

                    infer_time = time.perf_counter() - t_infer
                    self._total_infer_time += infer_time

                    if not success:
                        failed_tile_indices.append(tile_idx)
                        self._emit_log(f"tile {tile_id} 推理失败")
                        self._report_progress(tile_idx, total, processed, cache_hits,
                                              len(failed_tile_indices), infer_start_offset,
                                              tile_id=tile_id)
                        continue

                    # ── 读取 tile mask ──
                    mask_file = os.path.join(tile_output, "road_mask.png")
                    if not os.path.isfile(mask_file):
                        failed_tile_indices.append(tile_idx)
                        failed_tile_details.append({
                            "tile": tile_idx, "tile_id": tile_id, "rect": [x0, y0, x1, y1],
                            "error": "未生成 road_mask.png",
                        })
                        self._emit_log(f"tile {tile_id} 缺少 road_mask")
                        self._report_progress(tile_idx, total, processed, cache_hits,
                                              len(failed_tile_indices), infer_start_offset,
                                              tile_id=tile_id)
                        continue

                    tile_mask = _read_gray(mask_file)
                    if tile_mask.shape != (y1 - y0, x1 - x0):
                        tile_mask = cv2.resize(tile_mask, (x1 - x0, y1 - y0),
                                               interpolation=cv2.INTER_LINEAR)

                    # ── 保存 tile mask（缓存）──
                    tile_mask[tile_valid == 0] = 0
                    _write_image(cache_mask_path, tile_mask)
                    meta = {
                        "tile_id": tile_id,
                        "bbox": [x0, y0, x1, y1],
                        "cache_key": cache_key,
                        "infer_time_s": round(infer_time, 3),
                        "timestamp": datetime.now().isoformat(),
                    }
                    with open(cache_meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)

                # ── 合并到 global mask ──
                t_merge = time.perf_counter()
                if cfg.merge_method == "max":
                    target = merged[y0:y1, x0:x1]
                    np.maximum(target, tile_mask, out=target)
                else:
                    sum_mask[y0:y1, x0:x1] += tile_mask.astype(np.float32)
                    weights[y0:y1, x0:x1] += 1

                if not cache_hit:
                    self._io_time += time.perf_counter() - t_merge

                processed += 1
                self._report_progress(tile_idx, total, processed, cache_hits,
                                      len(failed_tile_indices), infer_start_offset,
                                      tile_id=tile_id)

            self._stage_end(STAGE_INFER_TILES, t_infer_stage)

            # ══════════════════════════════════════════════════════════
            # 阶段 merge_mask
            # ══════════════════════════════════════════════════════════
            t_merge_stage = self._stage_start(STAGE_MERGE_MASK)
            self._emit_log("正在生成 global_road_mask...")

            if cfg.merge_method == "average" and sum_mask is not None:
                nz = weights > 0
                merged[nz] = np.clip(sum_mask[nz] / weights[nz], 0, 255).astype(np.uint8)

            merged[valid_mask == 0] = 0

            # ── 保存 global mask ──
            mask_path = os.path.join(cfg.output_dir, "global_road_mask.png")
            _write_image(mask_path, merged)
            _write_image(os.path.join(cfg.output_dir, "road_mask.png"), merged)

            # ── 生成 preview ──
            preview_scale = min(1.0, 3000.0 / max(width, height))
            merged_preview = cv2.resize(
                merged,
                (max(1, int(width * preview_scale)), max(1, int(height * preview_scale))),
                interpolation=cv2.INTER_NEAREST,
            )
            preview_path = os.path.join(cfg.output_dir, "global_road_mask_preview.png")
            _write_image(preview_path, merged_preview)

            self._stage_end(STAGE_MERGE_MASK, t_merge_stage)

            # ══════════════════════════════════════════════════════════
            # 阶段 register_mask + finished
            # ══════════════════════════════════════════════════════════
            t_register = self._stage_start(STAGE_REGISTER_MASK)

            # ── 生成报告 ──
            elapsed = time.perf_counter() - self._start_time
            non_zero = int(np.count_nonzero(merged))
            total_px = merged.size

            report = {
                "mode": cfg.extraction_mode,
                "infer_mode": cfg.infer_mode,
                "tile_size": cfg.tile_size,
                "overlap": cfg.tile_overlap,
                "total_tiles": total,
                "processed_tiles": processed,
                "skipped_tiles": result_base.get("skipped_candidate_tiles", 0),
                "cache_hit_tiles": cache_hits,
                "failed_tiles": len(failed_tile_indices),
                "failed_tile_details": failed_tile_details,
                "elapsed_seconds": round(elapsed, 3),
                "avg_time_per_tile": round(
                    (elapsed - self._model_load_time) / max(1, total), 3
                ),
                "model_load_time_seconds": round(self._model_load_time, 3),
                "total_infer_time_seconds": round(self._total_infer_time, 3),
                "io_time_seconds": round(self._io_time, 3),
                "global_mask_nonzero_ratio": round(non_zero / max(1, total_px), 6),
                "output_global_mask_path": mask_path,
                "output_preview_path": preview_path,
                "image_width": width,
                "image_height": height,
                "cancelled": False,
                "worker_mode": cfg.infer_mode,
                "model_ready": True,
                "heartbeat_count": self._heartbeat_count,
                "timeout": False,
                "test_mode": cfg.max_tiles_for_test > 0,
                "warmup_only": cfg.warmup_only,
                "stage_times": self._stage_times,
                "timestamp": datetime.now().isoformat(),
            }

            report_path = os.path.join(cfg.output_dir, "formal_extraction_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            self._emit_log(f"正式提取完成: {processed} tiles, 用时 {elapsed:.1f}s")
            self._emit_log(f"道路占比: {report['global_mask_nonzero_ratio']*100:.1f}%")

            self._stage_end(STAGE_REGISTER_MASK, t_register)
            self._emit_stage(STAGE_FINISHED)

            # 清理持久化进程
            if self._persistent_proc:
                self._persistent_proc.shutdown()

            self.finished.emit(report)

        except ExtractionCancelled:
            # 保存部分结果
            _merged = locals().get("merged")
            _valid_mask = locals().get("valid_mask")
            _width = locals().get("width", 0)
            _height = locals().get("height", 0)
            self._save_partial(_merged, _valid_mask, _width, _height)
            if self._persistent_proc:
                self._persistent_proc.shutdown()
            partial_report = {**result_base, "cancelled": True,
                              "elapsed_seconds": round(time.perf_counter() - self._start_time, 3),
                              "stage_times": self._stage_times}
            report_path = os.path.join(cfg.output_dir, "formal_extraction_report.json")
            try:
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(partial_report, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            self.cancelled.emit("正式道路提取已取消")

        except TimeoutError as exc:
            if self._persistent_proc:
                self._persistent_proc.shutdown()

            result_base["timeout"] = True
            self._model_load_state = "timeout"
            log_path = ""
            try:
                log_path = write_error_log(
                    str(cfg.output_dir),
                    sys.exc_info(),
                    stage=self._current_stage or STAGE_LOAD_MODEL,
                    operation=f"模型加载超时 (>{cfg.model_load_timeout_seconds}s)",
                    worker_mode=cfg.infer_mode,
                    model_load_state=self._model_load_state,
                )
            except Exception as log_err:
                print(f"[FATAL] 错误日志写入失败: {log_err}", file=sys.stderr)

            self.failed.emit(
                f"模型加载超时 (>{cfg.model_load_timeout_seconds}s):\n{exc}\n日志: {log_path}",
                log_path,
            )

        except Exception as exc:
            original_error_type = type(exc).__name__
            original_error_message = str(exc)
            original_traceback = traceback.format_exc()

            log_path = ""
            try:
                if self._persistent_proc:
                    self._persistent_proc.shutdown()

                tile_idx, tile_id = self._resolve_error_tile_context(locals())
                log_path = write_error_log(
                    str(cfg.output_dir),
                    sys.exc_info(),
                    stage=self._current_stage or "run",
                    current_tile_id=tile_id,
                    current_tile_idx=tile_idx,
                    operation="FormalExtractionWorker.run",
                    failed_path=str(cfg.output_dir),
                    worker_mode=cfg.infer_mode,
                    model_load_state=self._model_load_state,
                    original_error_type=original_error_type,
                    original_error_message=original_error_message,
                    original_traceback=original_traceback,
                )
            except Exception as log_err:
                print(f"[FATAL] 错误日志写入失败: {log_err}", file=sys.stderr)
                print(original_traceback, file=sys.stderr)

            self.failed.emit(
                f"正式提取异常: {original_error_message}\n"
                f"阶段: {self._current_stage}\n"
                f"tile_idx: {getattr(self, 'current_tile_idx', 0)} "
                f"tile_id: {getattr(self, 'current_tile_id', '')}\n"
                f"日志: {log_path}",
                log_path,
            )

    # ===================================================================
    # Internal methods
    # ===================================================================

    def _start_persistent_process_with_heartbeat(self):
        """启动持久化推理进程（带心跳和超时）。"""
        cfg = self.config
        t_launch = self._stage_start(STAGE_LAUNCH_WORKER)

        script = os.path.join(os.path.dirname(__file__), "persistent_samroad_infer.py")
        if not os.path.isfile(script):
            raise RuntimeError(
                f"持久化推理脚本不存在: {script}\n"
                f"请确保 persistent_samroad_infer.py 与 formal_extraction_worker.py 在同一目录。"
            )

        python_exe = ensure_python_executable(cfg.python_executable)
        self._emit_log(f"Python 解释器: {python_exe}")
        self._emit_log(f"推理脚本: {script}")
        self._emit_log(f"配置文件: {cfg.config_path}")
        self._emit_log(f"模型权重: {cfg.samroad_model_ckpt_path}")
        self._emit_log(f"设备: {cfg.device}")
        self._emit_log(f"模型加载超时: {cfg.model_load_timeout_seconds}s")

        self._emit_log("启动持久化推理进程...")
        self._persistent_proc = PersistentInferProcess(
            python_exe=python_exe,
            script_path=script,
            config_path=str(cfg.config_path),
            checkpoint_path=str(cfg.samroad_model_ckpt_path),
            device=cfg.device,
            cancel_event=self._cancel_flag,
            log_callback=self._emit_log,
            heartbeat_callback=self._on_persistent_heartbeat,
        )

        self._stage_end(STAGE_LAUNCH_WORKER, t_launch)

        # ── 加载模型（带心跳和超时）──
        t_load = self._stage_start(STAGE_LOAD_MODEL)
        self._model_load_state = "loading"
        self._emit_log("等待模型加载（可随时取消）...")

        try:
            self._persistent_proc.start(
                timeout_seconds=cfg.model_load_timeout_seconds,
                heartbeat_interval=cfg.heartbeat_interval_seconds,
            )
        except Exception as exc:
            self._model_load_state = "failed"
            # 附加路径诊断
            diag_lines = [
                f"Python: {python_exe} (存在: {os.path.isfile(python_exe)})",
                f"Script: {script} (存在: {os.path.isfile(script)})",
                f"Config: {cfg.config_path} (存在: {os.path.isfile(str(cfg.config_path)) if cfg.config_path else 'N/A'})",
                f"Checkpoint: {cfg.samroad_model_ckpt_path} (存在: {os.path.isfile(str(cfg.samroad_model_ckpt_path)) if cfg.samroad_model_ckpt_path else 'N/A'})",
            ]
            raise RuntimeError(
                f"持久化推理进程启动/加载失败:\n  {exc}\n\n路径诊断:\n  " + "\n  ".join(diag_lines)
            ) from exc

        self._model_load_time = self._persistent_proc.model_load_time
        self._model_load_state = "ready"
        self._stage_end(STAGE_LOAD_MODEL, t_load)
        self._emit_stage(STAGE_MODEL_READY)
        self._emit_log(f"模型加载完成，耗时 {self._model_load_time:.1f}s")

    def _infer_tile_portable(self, cfg, env, tile_id, tile_idx, tile_path,
                             tile_output, bbox, failed_tile_details) -> bool:
        """使用 SAMRoad++ Portable infer.py 子进程处理单个 tile。

        产物写入 tile_output/：tile_image.png, road_mask.png,
        stdout.log, stderr.log, metadata.json。
        返回 success 布尔值。
        """
        python_exe = ensure_python_executable(cfg.python_executable)
        cwd = str(resolve_portable_paths(cfg.project_dir)["project_dir"])
        stdout_path = os.path.join(tile_output, "stdout.log")
        stderr_path = os.path.join(tile_output, "stderr.log")
        meta_path = os.path.join(tile_output, "metadata.json")

        devices = [cfg.device]
        if str(cfg.device).lower() == "cuda":
            devices.append("cpu")  # cuda 失败允许回退 cpu

        last_command = []
        last_return_code = -1
        stdout_text = ""
        stderr_text = ""
        selected_mask = None
        mask_candidates: list[str] = []
        used_device = cfg.device

        for device in devices:
            command = build_portable_command(
                python_exe, cfg.project_dir, tile_path, tile_output, device
            )
            last_command = command
            used_device = device
            try:
                process = subprocess.run(
                    command, cwd=cwd, env=env,
                    capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                last_return_code = process.returncode
                stdout_text = process.stdout or ""
                stderr_text = process.stderr or ""
            except Exception as exc:
                last_return_code = -1
                stderr_text = f"subprocess 启动失败: {exc}"

            # 写日志（不因日志写入失败而中断）
            try:
                with open(stdout_path, "w", encoding="utf-8") as f:
                    f.write(stdout_text)
                with open(stderr_path, "w", encoding="utf-8") as f:
                    f.write(stderr_text)
            except Exception:
                pass

            selected_mask, mask_candidates = normalize_output_mask(tile_output)
            success = (last_return_code == 0) and (selected_mask is not None)

            if success:
                break
            if device == "cuda" and len(devices) > 1:
                self._emit_log(f"tile {tile_id} cuda 推理失败，尝试 cpu 回退")

        success = (last_return_code == 0) and (selected_mask is not None)

        meta = {
            "tile_id": tile_id,
            "tile_idx": tile_idx,
            "tile_bbox": bbox,
            "command": last_command,
            "cwd": cwd,
            "device": used_device,
            "return_code": last_return_code,
            "mask_candidates": mask_candidates,
            "selected_mask": selected_mask,
            "success": success,
        }
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        if not success:
            failed_tile_details.append({
                "tile": tile_idx, "tile_id": tile_id, "rect": bbox,
                "return_code": last_return_code,
                "command": last_command,
                "cwd": cwd,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "mask_candidates": mask_candidates,
                "error": (stderr_text[-500:] if stderr_text else "未生成 mask"),
            })

        return success

    def _resolve_error_tile_context(self, local_vars: Optional[dict] = None):
        """从 worker 状态或局部变量安全解析 tile_idx / tile_id。"""
        local_vars = local_vars or {}
        tile_idx = getattr(self, "current_tile_idx", None)
        if not isinstance(tile_idx, int) or tile_idx <= 0:
            raw_idx = local_vars.get("tile_idx")
            if isinstance(raw_idx, int):
                tile_idx = raw_idx
            elif isinstance(local_vars.get("index"), int):
                tile_idx = local_vars["index"]
            else:
                tile_idx = None

        tile_id = normalize_tile_id(
            getattr(self, "current_tile_id", None) or local_vars.get("tile_id"),
            fallback_index=tile_idx,
        )
        return tile_idx, tile_id

    def _report_progress(self, tile_idx: int, total: int, processed: int,
                         cache_hits: int, failed: int, start_time: float,
                         tile_id: str = ""):
        """计算并发送进度信号。tile_idx 用于百分比，tile_id 用于日志/文件。"""
        percent = int(round(tile_idx * 100 / max(1, total)))
        safe_tile_id = normalize_tile_id(tile_id or getattr(self, "current_tile_id", None),
                                        fallback_index=tile_idx)

        elapsed = time.perf_counter() - start_time
        if processed > 0:
            avg = elapsed / processed
            remaining = avg * (total - tile_idx)
        else:
            avg = 0
            remaining = 0

        detail = {
            "success_tiles": processed - cache_hits,
            "failed_tiles": failed,
            "skipped_tiles": cache_hits,
            "cache_hit_tiles": cache_hits,
            "elapsed_seconds": round(time.perf_counter() - self._start_time, 1),
            "estimated_remaining_seconds": round(remaining, 1),
            "avg_time_per_tile": round(avg, 3),
            "mode": self.config.extraction_mode,
            "infer_mode": self.config.infer_mode,
            "stage": self._current_stage,
            "tile_idx": tile_idx,
            "tile_id": safe_tile_id,
        }

        self._emit_log(
            f"进度 {tile_idx}/{total} ({safe_tile_id}) | 成功={processed} 失败={failed} "
            f"缓存命中={cache_hits} | 已用 {detail['elapsed_seconds']:.0f}s "
            f"预计剩余 {remaining:.0f}s"
        )

        self.progress.emit(percent, tile_idx, total, detail)

    def _save_partial(self, merged: np.ndarray, valid_mask: np.ndarray,
                      width: int, height: int):
        """取消时保存部分结果（允许后续 resume）。"""
        cfg = self.config
        try:
            if merged is not None and merged.size > 0:
                partial_path = os.path.join(cfg.output_dir, "partial_global_road_mask.png")
                _write_image(partial_path, merged)
            self._emit_log("已保存部分结果，下次可继续")
        except Exception:
            pass

    def _run_fast_preview(self):
        """快速预览提取。"""
        cfg = self.config
        self._emit_log("快速预览提取")
        reader = ImageRegionReader(str(cfg.image_path))
        preview = reader.read_preview(3000)
        mask_path = os.path.join(cfg.output_dir, "global_road_mask_preview.png")
        h, w = preview.shape[:2]
        dummy_mask = np.zeros((h, w), dtype=np.uint8)
        _write_image(mask_path, dummy_mask)
        self._emit_log("快速预览完成（未生成正式 road_mask）")
