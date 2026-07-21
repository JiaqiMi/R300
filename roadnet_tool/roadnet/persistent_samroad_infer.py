"""
SAM-Road persistent inference server.

This script is designed to be run as a long-lived subprocess that:
1. Loads the SAM-Road model ONCE at startup
2. Reads tile inference requests from stdin (JSON lines)
3. Processes each tile and writes results to stdout (JSON lines)
4. Continues running until stdin closes or it receives a shutdown command

Protocol (JSON lines over stdin/stdout):

Request (stdin):
    {\"action\": \"infer\", \"image_path\": \"...\", \"output_dir\": \"...\", \"request_id\": \"...\"}
    {\"action\": \"shutdown\"}

Response (stdout):
    {\"type\": \"ready\", \"device\": \"cuda\"}
    {\"type\": \"status\", \"message\": \"...\"}
    {\"type\": \"result\", \"request_id\": \"...\", \"success\": true, \"output_dir\": \"...\", \"duration_s\": 1.23}
    {\"type\": \"error\", \"request_id\": \"...\", \"message\": \"...\"}
    {\"type\": \"shutdown_ok\"}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch


# ===================================================================
# Global model holder (loaded once)
# ===================================================================

_net = None
_config = None
_device = None


def _emit(msg: dict):
    """安全输出 JSON 行到 stdout."""
    print(json.dumps(msg, ensure_ascii=False), flush=True)


def load_model(config_path: str, checkpoint_path: str, device_str: str = "cuda"):
    """Load the SAM-Road model. Called once at startup."""
    global _net, _config, _device

    t_start = time.perf_counter()

    # Add project directory to path for imports
    project_dir = str(Path(__file__).resolve().parent.parent)
    sam_dir = os.path.join(project_dir, "sam")
    for p in (project_dir, sam_dir):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    _emit({"type": "status", "stage": "load_model", "step": "import_modules",
           "message": "Importing SAM-Road modules...",
           "elapsed_s": round(time.perf_counter() - t_start, 2)})

    from utils import load_config as _load_config
    from model import SAMRoad

    _emit({"type": "status", "stage": "load_model", "step": "load_config",
           "message": f"Loading config: {config_path}",
           "elapsed_s": round(time.perf_counter() - t_start, 2)})

    _config = _load_config(config_path)
    _device = torch.device(device_str) if device_str == "cuda" and torch.cuda.is_available() else torch.device("cpu")

    _emit({"type": "status", "stage": "load_model", "step": "device",
           "message": f"Device: {_device}",
           "elapsed_s": round(time.perf_counter() - t_start, 2)})

    if _device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        _emit({"type": "status", "stage": "load_model", "step": "cuda_config",
               "message": "CUDA configured (benchmark=True)",
               "elapsed_s": round(time.perf_counter() - t_start, 2)})

    _emit({"type": "status", "stage": "load_model", "step": "init_model",
           "message": "Initializing SAMRoad model...",
           "elapsed_s": round(time.perf_counter() - t_start, 2)})

    _net = SAMRoad(_config)

    _emit({"type": "status", "stage": "load_model", "step": "load_checkpoint",
           "message": f"Loading checkpoint: {checkpoint_path}",
           "elapsed_s": round(time.perf_counter() - t_start, 2)})

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    _emit({"type": "status", "stage": "load_model", "step": "load_state_dict",
           "message": "Loading state dict into model...",
           "elapsed_s": round(time.perf_counter() - t_start, 2)})

    _net.load_state_dict(checkpoint["state_dict"], strict=True)
    _net.eval()

    _emit({"type": "status", "stage": "load_model", "step": "to_device",
           "message": f"Moving model to {_device}...",
           "elapsed_s": round(time.perf_counter() - t_start, 2)})

    _net.to(_device)

    _emit({"type": "ready", "device": str(_device),
           "model_load_time_s": round(time.perf_counter() - t_start, 2)})


def pad_to_square(img: np.ndarray, min_side: int):
    h, w = img.shape[:2]
    side = max(h, w, min_side)
    if h == side and w == side:
        return img, 0, 0
    padded = np.zeros((side, side, img.shape[2]), dtype=img.dtype)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    padded[y0:y0 + h, x0:x0 + w] = img
    return padded, x0, y0


def crop_to_original(img: np.ndarray, x0: int, y0: int, h: int, w: int):
    return img[y0:y0 + h, x0:x0 + w]


def process_single_tile(image_path: str, output_dir: str) -> dict:
    """Process a single tile image and save road_mask.png to output_dir."""
    global _net, _config, _device

    from dataset import read_rgb_img
    from inferencer import infer_one_img

    t0 = time.perf_counter()
    original_img = read_rgb_img(image_path)
    original_h, original_w = original_img.shape[:2]
    img, pad_x, pad_y = pad_to_square(original_img, _config.PATCH_SIZE)

    # Inference
    _, _, itsc_mask, road_mask = infer_one_img(_net, img, _config, _device)

    # Crop back
    road_mask = crop_to_original(road_mask, pad_x, pad_y, original_h, original_w)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "road_mask.png"), road_mask)

    # Metadata
    metadata = {
        "image_path": os.path.abspath(image_path),
        "original_width": original_w,
        "original_height": original_h,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return {
        "success": True,
        "output_dir": output_dir,
        "duration_s": round(time.perf_counter() - t0, 3),
        "tile_size": f"{original_w}x{original_h}",
    }


def run_server(config_arg_path: str, checkpoint_path: str, device_str: str):
    """Main server loop: read requests from stdin, write results to stdout."""
    load_model(config_arg_path, checkpoint_path, device_str)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            _emit({"type": "error", "request_id": "", "message": f"JSON decode error: {e}"})
            continue

        action = request.get("action", "")
        request_id = request.get("request_id", "")

        if action == "ping":
            # 健康检查 / 心跳响应
            _emit({"type": "pong", "request_id": request_id})

        elif action == "shutdown":
            _emit({"type": "shutdown_ok"})
            break

        elif action == "infer":
            image_path = request.get("image_path", "")
            output_dir = request.get("output_dir", "")
            try:
                result = process_single_tile(image_path, output_dir)
                result["request_id"] = request_id
                result["type"] = "result"
                _emit(result)
            except Exception as e:
                _emit({
                    "type": "error",
                    "request_id": request_id,
                    "message": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc()[-500:],
                })

        else:
            _emit({"type": "error", "request_id": request_id,
                   "message": f"Unknown action: {action}"})


def main():
    parser = argparse.ArgumentParser(description="SAM-Road persistent inference server")
    parser.add_argument("--config", required=True, help="config yaml path")
    parser.add_argument("--checkpoint", required=True, help="model checkpoint path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    run_server(args.config, args.checkpoint, args.device)


if __name__ == "__main__":
    main()
