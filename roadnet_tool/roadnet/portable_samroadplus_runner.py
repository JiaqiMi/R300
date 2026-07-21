"""SAMRoad++ Portable 单图推理适配器。

用于 samroadplus_portable_infer 工程结构：

    <project_dir>/
      ├── infer.py
      ├── model_portable.py
      ├── config.yaml
      ├── model_state_dict.pth
      ├── sam/
      └── outputs/

infer.py 的参数格式：

    python infer.py --image IMAGE [--output_dir OUTPUT_DIR]
                    [--config CONFIG] [--checkpoint CHECKPOINT]
                    [--device {cuda,cpu}]

该模块只负责 subprocess_per_tile 流程（第一版）。
持久化 worker 后续单独实现。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional


# 默认 Portable 工程目录（新训练模型）
DEFAULT_PORTABLE_PROJECT_DIR = r"D:\samroadplus_portable_infer"

# 判定 Portable 工程所需的标志文件
PORTABLE_MARKER_FILES = (
    "infer.py",
    "model_portable.py",
    "config.yaml",
    "model_state_dict.pth",
)

# infer.py 可能生成的候选 mask 文件名（按优先级）
MASK_CANDIDATE_NAMES = (
    "road_mask.png",
    "mask.png",
    "pred_mask.png",
    "road_pred.png",
    "road_prediction.png",
    "seg.png",
    "segmentation.png",
    "binary_mask.png",
    "output_mask.png",
)


def is_portable_project(project_dir) -> bool:
    """检测 project_dir 是否为 samroadplus_portable_infer 结构。"""
    if not project_dir:
        return False
    base = Path(project_dir)
    if not base.is_dir():
        return False
    return all((base / name).is_file() for name in PORTABLE_MARKER_FILES)


def resolve_portable_paths(project_dir) -> dict:
    """根据 project_dir 推导 Portable 工程的固定路径。"""
    base = Path(project_dir).resolve()
    return {
        "project_dir": base,
        "infer_script": base / "infer.py",
        "model_portable": base / "model_portable.py",
        "config": base / "config.yaml",
        "checkpoint": base / "model_state_dict.pth",
        "sam_dir": base / "sam",
    }


def resolve_portable_project_dir(configured_dir=None):
    """解析要使用的 Portable 工程目录。

    优先级：
      1. 显式传入且为合法 Portable 工程的目录；
      2. 显式传入的目录（即使暂时不完整，也按用户意图使用）；
      3. 默认目录 DEFAULT_PORTABLE_PROJECT_DIR。
    """
    if configured_dir:
        candidate = str(configured_dir).strip()
        if candidate and candidate not in (".", "None"):
            return Path(candidate)
    return Path(DEFAULT_PORTABLE_PROJECT_DIR)


def apply_portable_config(cfg, configured_dir=None) -> str:
    """将正式提取 cfg 覆盖为 Portable 工程路径，返回 adapter_type。

    覆盖字段：project_dir / infer_script / config_path / samroad_model_ckpt_path。
    python_executable / device 保持外部配置不变。
    """
    base = resolve_portable_project_dir(configured_dir)
    paths = resolve_portable_paths(base)
    cfg.project_dir = paths["project_dir"]
    cfg.infer_script = paths["infer_script"]
    cfg.config_path = paths["config"]
    cfg.samroad_model_ckpt_path = paths["checkpoint"]
    return "samroadplus_portable"


def validate_portable_paths(python_exe, project_dir, output_dir) -> list[str]:
    """启动前验证 Portable 关键路径，返回错误列表（空列表表示通过）。"""
    errors: list[str] = []

    if not python_exe or not os.path.isfile(str(python_exe)):
        errors.append(f"Python 解释器无效: {python_exe}")

    if not project_dir or not os.path.isdir(str(project_dir)):
        errors.append(f"Portable 工程目录不存在: {project_dir}")
        return errors

    paths = resolve_portable_paths(project_dir)
    checks = [
        ("infer_script", "infer.py"),
        ("model_portable", "model_portable.py"),
        ("config", "config.yaml"),
        ("checkpoint", "model_state_dict.pth"),
    ]
    for key, label in checks:
        if not paths[key].is_file():
            errors.append(f"缺少 {label}: {paths[key]}")

    if not paths["sam_dir"].is_dir():
        errors.append(f"缺少 sam 目录: {paths['sam_dir']}")

    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            probe = os.path.join(str(output_dir), ".write_probe")
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
        except Exception as exc:
            errors.append(f"输出目录不可写: {output_dir} ({exc})")

    return errors


def build_portable_env(base_env: dict, project_dir) -> dict:
    """构建 Portable 子进程环境变量，注入 PYTHONPATH。"""
    env = dict(base_env)
    paths = resolve_portable_paths(project_dir)
    pythonpath_parts = [str(paths["project_dir"])]
    if paths["sam_dir"].is_dir():
        pythonpath_parts.append(str(paths["sam_dir"]))
    existing = env.get("PYTHONPATH", "")
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def build_portable_command(
    python_exe,
    project_dir,
    tile_image_path,
    tile_output_dir,
    device: str = "cuda",
) -> list[str]:
    """构造 infer.py 子进程命令（不使用 shell）。"""
    paths = resolve_portable_paths(project_dir)
    return [
        str(python_exe),
        str(paths["infer_script"]),
        "--image", str(tile_image_path),
        "--output_dir", str(tile_output_dir),
        "--config", str(paths["config"]),
        "--checkpoint", str(paths["checkpoint"]),
        "--device", str(device),
    ]


def find_mask_candidates(tile_output_dir) -> list[str]:
    """在 tile_output_dir 中递归搜索候选 mask 文件，按优先级排序。"""
    base = Path(tile_output_dir)
    if not base.is_dir():
        return []
    found: list[str] = []
    priority = {name: i for i, name in enumerate(MASK_CANDIDATE_NAMES)}
    for path in base.rglob("*.png"):
        if path.name.lower() in priority:
            found.append(str(path))
    found.sort(key=lambda p: (priority.get(Path(p).name.lower(), 999), len(p)))
    return found


def normalize_output_mask(tile_output_dir) -> tuple[Optional[str], list[str]]:
    """将候选 mask 统一为 tile_output_dir/road_mask.png。

    返回 (selected_mask_path 或 None, 所有候选列表)。
    """
    candidates = find_mask_candidates(tile_output_dir)
    if not candidates:
        return None, []

    target = os.path.join(str(tile_output_dir), "road_mask.png")
    selected = candidates[0]

    if os.path.abspath(selected) == os.path.abspath(target):
        return target, candidates

    try:
        shutil.copyfile(selected, target)
    except Exception:
        return selected, candidates
    return target, candidates
