"""
工具函数：YAML 加载、目录创建、路径拼接等通用能力。
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载配置文件。优先使用用户指定的路径，否则加载默认 config/default.yaml。

    Args:
        config_path: 用户指定的配置文件路径（可选）

    Returns:
        配置字典
    """
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # 回退到默认配置
    default_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    with open(default_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> str:
    """
    确保目录存在，若不存在则递归创建。

    Args:
        path: 目标目录路径

    Returns:
        标准化后的目录路径
    """
    os.makedirs(path, exist_ok=True)
    return path
