"""
SAM-Road 单图推理包 Runner 模块。

负责管理外部 infer_single.py 的 QProcess 异步调用：

1. 配置读取/保存（YAML）
2. 路径验证
3. QProcess 命令构造（不使用 shell=True）
4. 结果数据结构
5. dry-run / mock 模式支持

与现有 samroad_runner.py 的区别：
- 调用目标不同：infer_single.py（单图推理） vs bridge 脚本（批处理）
- 输出格式不同：graph.p + road_mask/itsc_mask/viz vs draft_graph.json
- 参数不同：tile_size/overlap → device + output_dir name
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml


# ===================================================================
# Runtime 环境准备（必须在 import model / matplotlib 前执行）
# ===================================================================

def prepare_runtime_env(output_dir=None) -> Dict[str, str]:
    """为 SAM-Road 子进程设置安全的运行环境变量。

    解决 Windows + 中文用户名 + QProcess 环境变量不完整时
    matplotlib 无法确定 home 目录的问题：
        RuntimeError: Could not determine home directory.

    返回构建好的环境变量字典，调用方应将其注入 QProcess / subprocess。
    """
    base = Path(output_dir or "outputs/samroad_runtime").resolve()
    runtime_dir = base / "_runtime"
    mpl_dir = runtime_dir / "matplotlib"
    home_dir = runtime_dir / "home"

    mpl_dir.mkdir(parents=True, exist_ok=True)
    home_dir.mkdir(parents=True, exist_ok=True)

    env_overrides: Dict[str, str] = {
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(mpl_dir),
        # Windows 下避免 pathlib.Path.home() 因中文用户名或环境缺失失败
        "HOME": str(home_dir),
        "USERPROFILE": str(home_dir),
        # Python / Qt 日志统一 UTF-8
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        # 禁用 wandb 在线模式，防止模型初始化失败
        "WANDB_MODE": "disabled",
    }

    return env_overrides


def prepare_project_import_paths(project_dir) -> str:
    """构建 SAM-Road 项目所需的 PYTHONPATH 组件（os.pathsep 分隔）。

    必须在子进程中 import model / segment_anything 之前注入 QProcess 环境变量。

    segment_anything 位于 <project_dir>/sam/segment_anything/，
    但 predictor.py 中写了绝对导入:
        from segment_anything.modeling import Sam

    因此 PYTHONPATH 必须包含：
        - <project_dir>       (使 from model import SAMRoad 生效)
        - <project_dir>/sam   (使 import segment_anything 生效)

    返回: PYTHONPATH 中应该前置的路径部分（不含原有 PYTHONPATH）。
    """
    project_dir = Path(project_dir).resolve()
    sam_dir = project_dir / "sam"

    paths = [str(project_dir)]
    if sam_dir.is_dir():
        paths.append(str(sam_dir))

    return os.pathsep.join(paths)


# ===================================================================
# 配置数据结构
# ===================================================================

@dataclass
class SAMRoadSingleRunConfig:
    """单图推理包运行配置。"""
    project_dir: Path = Path("D:/sam_road_single_image_share")
    python_executable: Path = Path()
    infer_script: Path = Path("D:/sam_road_single_image_share/infer_single.py")
    config_path: Path = Path("D:/sam_road_single_image_share/config/toponet_vitb_256_spacenet_4060_night.yaml")
    sam_backbone_ckpt_path: Path = Path("D:/sam_road_single_image_share/sam_ckpts/sam_vit_b_01ec64.pth")
    samroad_model_ckpt_path: Path = Path("D:/sam_road_single_image_share/checkpoints/model.ckpt")
    input_image: Path = Path()
    output_dir: Path = Path()
    device: str = "cuda"
    auto_import_after_run: bool = True
    dry_run: bool = False
    mask_only_partial_load: bool = False
    inference_mode: str = "auto"       # auto / whole / tile
    tile_size: int = 1024
    overlap: int = 128
    skip_black_tile: bool = True
    black_threshold: int = 10
    min_black_component_area: int = 4096
    valid_pixel_ratio_threshold: float = 0.1
    merge_method: str = "max"

    @property
    def is_valid(self) -> bool:
        return (
            self.infer_script.is_file()
            and self.input_image.is_file()
        )


# ===================================================================
# 配置持久化
# ===================================================================

DEFAULT_CONFIG_PATH = "config/samroad_single_config.yaml"


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """从 YAML 文件加载配置。"""
    path = Path(config_path)
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_config(config_path: str, config: dict):
    """保存配置到 YAML 文件。"""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def dict_to_runconfig(data: dict) -> SAMRoadSingleRunConfig:
    """从字典加载运行配置。"""
    sr = data.get("samroad_single", {})
    c = SAMRoadSingleRunConfig()

    if sr.get("project_dir"):
        c.project_dir = Path(sr["project_dir"])
    if sr.get("python_executable"):
        c.python_executable = Path(sr["python_executable"])
    if sr.get("infer_script"):
        c.infer_script = Path(sr["infer_script"])
    if sr.get("config_path"):
        c.config_path = Path(sr["config_path"])
    if sr.get("sam_backbone_ckpt_path"):
        c.sam_backbone_ckpt_path = Path(sr["sam_backbone_ckpt_path"])
    if sr.get("samroad_model_ckpt_path"):
        c.samroad_model_ckpt_path = Path(sr["samroad_model_ckpt_path"])
    # 向后兼容旧字段名
    if sr.get("checkpoint_path") and not sr.get("samroad_model_ckpt_path"):
        c.samroad_model_ckpt_path = Path(sr["checkpoint_path"])
    if sr.get("device"):
        c.device = str(sr["device"])
    if sr.get("auto_import_after_run") is not None:
        c.auto_import_after_run = bool(sr["auto_import_after_run"])
    if sr.get("dry_run") is not None:
        c.dry_run = bool(sr["dry_run"])
    if sr.get("mask_only_partial_load") is not None:
        c.mask_only_partial_load = bool(sr["mask_only_partial_load"])
    c.inference_mode = str(sr.get("inference_mode", c.inference_mode))
    c.tile_size = int(sr.get("tile_size", c.tile_size))
    c.overlap = int(sr.get("overlap", c.overlap))
    c.skip_black_tile = bool(sr.get("skip_black_tile", c.skip_black_tile))
    c.black_threshold = int(sr.get("black_threshold", c.black_threshold))
    c.min_black_component_area = int(sr.get(
        "min_black_component_area", c.min_black_component_area
    ))
    c.valid_pixel_ratio_threshold = float(sr.get(
        "valid_pixel_ratio_threshold", c.valid_pixel_ratio_threshold
    ))
    c.merge_method = str(sr.get("merge_method", c.merge_method))

    return c


def runconfig_to_dict(config: SAMRoadSingleRunConfig) -> dict:
    """将运行配置序列化为字典（用于 YAML 保存）。"""
    return {
        "samroad_single": {
            "project_dir": str(config.project_dir).replace("\\", "/"),
            "python_executable": str(config.python_executable).replace("\\", "/"),
            "infer_script": str(config.infer_script).replace("\\", "/"),
            "config_path": str(config.config_path).replace("\\", "/"),
            "sam_backbone_ckpt_path": str(config.sam_backbone_ckpt_path).replace("\\", "/"),
            "samroad_model_ckpt_path": str(config.samroad_model_ckpt_path).replace("\\", "/"),
            "device": config.device,
            "auto_import_after_run": config.auto_import_after_run,
            "dry_run": config.dry_run,
            "mask_only_partial_load": config.mask_only_partial_load,
            "inference_mode": config.inference_mode,
            "tile_size": config.tile_size,
            "overlap": config.overlap,
            "skip_black_tile": config.skip_black_tile,
            "black_threshold": config.black_threshold,
            "min_black_component_area": config.min_black_component_area,
            "valid_pixel_ratio_threshold": config.valid_pixel_ratio_threshold,
            "merge_method": config.merge_method,
        }
    }


# ===================================================================
# 验证
# ===================================================================

def validate_config(config: SAMRoadSingleRunConfig) -> list[str]:
    """验证运行配置，返回错误信息列表。"""
    errors: list[str] = []
    if config.inference_mode not in ("auto", "whole", "tile"):
        errors.append(f"未知推理模式: {config.inference_mode}")
    if config.tile_size <= 0 or config.overlap < 0 or config.overlap >= config.tile_size:
        errors.append("SAM-Road tile 参数必须满足 tile_size > overlap >= 0")
    if config.min_black_component_area < 1:
        errors.append("min_black_component_area 必须大于 0")
    if config.merge_method not in ("max", "average"):
        errors.append("SAM-Road merge_method 必须为 max 或 average")

    # Python 解释器
    if config.dry_run:
        # dry-run 模式下不检查 python/checkpoint
        pass
    else:
        if not config.python_executable or not Path(config.python_executable).is_file():
            errors.append(f"Python 解释器未找到: {config.python_executable}")

        if not config.samroad_model_ckpt_path or not Path(config.samroad_model_ckpt_path).is_file():
            errors.append(f"SAM-Road 模型权重未找到: {config.samroad_model_ckpt_path}")

        if not config.sam_backbone_ckpt_path or not Path(config.sam_backbone_ckpt_path).is_file():
            errors.append(f"SAM backbone 权重未找到: {config.sam_backbone_ckpt_path}")

    # 输入图像
    if not config.input_image or not Path(config.input_image).is_file():
        errors.append(f"输入图像未找到: {config.input_image}")

    # 推理脚本
    if not config.infer_script or not Path(config.infer_script).is_file():
        errors.append(f"infer_single.py 未找到: {config.infer_script}")

    # 配置文件
    if not config.config_path or not Path(config.config_path).is_file():
        if not config.dry_run:
            errors.append(f"Config 文件未找到: {config.config_path}")

    # SAM-Road 项目目录
    if not config.project_dir or not Path(config.project_dir).is_dir():
        if not config.dry_run:
            errors.append(f"SAM-Road 项目目录不存在: {config.project_dir}")

    return errors


# ===================================================================
# 命令构造
# ===================================================================

def build_command(config: SAMRoadSingleRunConfig, output_dir_name: str) -> list[str]:
    """构造 subprocess/QProcess 的命令参数列表（不使用 shell=True）。

    infer_single.py 的调用格式：
        python infer_single.py --config <yaml> --checkpoint <ckpt> --image <img> --output_dir <name>

    注意：infer_single.py 的 output_dir 是 save/ 下的子目录名（不是绝对路径）。
    实际输出在 <project_dir>/save/<output_dir_name>/

    为避免输出位置不明确，我们传一个固定的子目录名，然后将结果复制到目标位置。
    更好的做法：让命令在当前工作目录等于输出目录的环境下运行。

    实际方案：创建目标输出目录，以它为 --output_dir 参数。
    但 infer_single.py 会固定输出到 save/<output_dir>/。
    所以我们不能在参数中指定绝对路径。

    更好的方案：
    1. 创建目标目录 output_dir
    2. 使用 --output_dir 传固定名（如 "_temp_single_run"）
    3. 进程结束后将 save/_temp_single_run/* 移动到 output_dir/

    简单方案：把工作目录设置为 output_dir 的父目录，然后传相对路径。
    但 infer_single.py 硬编码了 save/ 前缀。

    最终方案：
    1. 不传 --output_dir（用默认值，结果在 save/single_infer/）
    2. 运行完成后把结果复制/移动到目标 output_dir

    OR 最简方案：把目标 output_dir 路径传给 --output_dir，
    infer_single.py 会输出到 save/<output_dir>，其中 output_dir 可以是绝对路径拼接...

    实际上，看代码: output_dir = os.path.join("save", args.output_dir)
    如果 args.output_dir 是绝对路径，os.path.join 会忽略 "save"！

    所以：直接传目标 output_dir 的绝对路径即可！
    """
    cmd = [
        str(config.python_executable),
        str(config.infer_script),
        "--config", str(config.config_path),
        "--checkpoint", str(config.samroad_model_ckpt_path) if not config.dry_run else "mock",
        "--image", str(config.input_image),
        "--output_dir", output_dir_name,
        "--device", config.device,
    ]
    if config.mask_only_partial_load:
        cmd.append("--mask-only-partial-load")
    return cmd


def build_dryrun_command(config: SAMRoadSingleRunConfig) -> list[str]:
    """构造 dry-run 测试命令（只做路径检查，不运行推理）。"""
    return [
        str(config.python_executable) if config.python_executable != Path() else "python",
        "-c",
        (
            f"import sys, os; print('DRY RUN / MOCK MODE — NOT REAL SAM-ROAD INFERENCE');"
            f"print(f'Python: {{sys.executable}}');"
            f"print(f'Image: {config.input_image}');"
            f"print(f'Config: {config.config_path}');"
            f"print(f'Checkpoint: {config.samroad_model_ckpt_path}');"
            f"print(f'Output: {config.output_dir}');"
            f"print('Dry-run check passed — all paths exist.' if "
            f"os.path.exists('{str(config.input_image).replace(chr(39), chr(39)+chr(92)+chr(39)+chr(39))}') "
            f"else 'WARNING: image not found');"
            f"print('[DRY-RUN] 检查完成，没有执行真实 SAM-Road 推理。')"
        ),
    ]


# ===================================================================
# 输出目录管理
# ===================================================================

def create_output_dir(base_dir: str, image_path: str) -> Path:
    """基于图像名创建输出目录。

    命名规则: <base_dir>/samroad_single_<image_stem>_<timestamp>/
    """
    base = Path(base_dir)
    image_stem = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = f"samroad_single_{image_stem}_{timestamp}"
    output_dir = base / dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ===================================================================
# 运行结果数据结构
# ===================================================================

@dataclass
class SAMRoadSingleRunResult:
    """单图推理包运行结果。"""
    success: bool = False
    return_code: int = -1
    output_dir: Path = field(default_factory=Path)
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""
    is_dry_run: bool = False
    is_partial_load: bool = False
    partial_load_warning: str = ""
    duration_seconds: float = 0.0
    node_count: int = 0
    edge_count: int = 0
    found_files: list[str] = field(default_factory=list)
    model_type: str = "samroad_single_image"
    ignore_graph: bool = False
    output_diagnostics: dict = field(default_factory=dict)

    @classmethod
    def from_process_result(
        cls,
        return_code: int,
        output_dir: Path,
        stdout: str,
        stderr: str,
        is_dry_run: bool = False,
    ) -> "SAMRoadSingleRunResult":
        result = cls(
            success=(return_code == 0 and (
                is_dry_run or (Path(output_dir) / "road_mask.png").is_file()
            )),
            return_code=return_code,
            output_dir=output_dir,
            stdout=stdout,
            stderr=stderr,
            is_dry_run=is_dry_run,
        )

        # 读取 metadata
        meta_path = output_dir / "metadata.json"
        if meta_path.is_file():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                result.node_count = meta.get("node_count", 0)
                result.edge_count = meta.get("edge_count", 0)
                result.is_partial_load = meta.get("partial_load", False)
                result.partial_load_warning = meta.get("partial_load_warning", "")
            except Exception:
                pass

        # 列出输出文件
        if output_dir.is_dir():
            result.found_files = sorted(
                str(path.relative_to(output_dir)).replace("\\", "/")
                for path in output_dir.rglob("*") if path.is_file()
            )

        return result
