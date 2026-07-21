"""Background worker for the mask-to-final-graph one-click pipeline."""
from __future__ import annotations

import copy
import os
import threading
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import cv2
from PySide6.QtCore import QObject, Signal, Slot


@dataclass
class PipelineResult:
    processed_mask: np.ndarray
    raw_skeleton: np.ndarray
    optimized_skeleton: np.ndarray
    skeleton_report: Dict
    nodes: List[Dict]
    edges: List[Dict]
    connectivity: Dict
    warnings: List[str]
    mask_preview: Optional[np.ndarray] = None
    skeleton_preview: Optional[np.ndarray] = None


class PipelineWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str, str, str)
    cancelled = Signal(str)

    def __init__(self, mask: np.ndarray, config: Dict, output_root: str,
                 valid_image_mask: np.ndarray = None, parent=None):
        super().__init__(parent)
        # Do not copy a global mask in the GUI thread.  The private copy is
        # created at the start of run(), after this object enters QThread.
        self.mask = mask
        self.config = copy.deepcopy(config)
        self.output_root = output_root
        self.valid_image_mask = (
            None if valid_image_mask is None
            else valid_image_mask
        )
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def _check_cancelled(self):
        if self._cancel.is_set():
            raise InterruptedError("用户取消")

    @Slot()
    def run(self):
        stage = "validate_mask"
        try:
            self.mask = np.asarray(self.mask, dtype=np.uint8).copy()
            if self.valid_image_mask is not None:
                self.valid_image_mask = np.asarray(
                    self.valid_image_mask, dtype=np.uint8,
                ).copy()
            if self.valid_image_mask is not None:
                from roadnet.valid_image import apply_valid_image_mask
                self.mask = apply_valid_image_mask(self.mask, self.valid_image_mask)
            if self.mask.size == 0 or not np.any(self.mask):
                raise ValueError("road_mask 为空")
            self._check_cancelled()

            stage = "mask_postprocess"
            self.progress.emit(10, "阶段 1/4：mask 后处理")
            from roadnet.postprocess import clean_pipeline, analyze_mask_anomalies
            post_cfg = copy.deepcopy(self.config.get("postprocess", {}))
            post_cfg["close_kernel_size"] = min(post_cfg.get("close_kernel_size", 3), 5)
            post_cfg["fill_holes"] = False
            post_cfg["fill_small_holes"] = False
            clean_mask, _ = clean_pipeline(
                self.mask, post_cfg, save_intermediate=False, output_dir=""
            )
            if self.valid_image_mask is not None:
                clean_mask = apply_valid_image_mask(clean_mask, self.valid_image_mask)
            if clean_mask is None or clean_mask.size == 0 or not np.any(clean_mask):
                raise ValueError("后处理输出了空 mask")
            if np.all(clean_mask > 0):
                raise ValueError("后处理输出全白 mask")
            anomaly = analyze_mask_anomalies(
                clean_mask,
                original_mask=self.mask,
                max_road_ratio=post_cfg.get("max_road_ratio_warn", 0.25),
                max_largest_ratio=post_cfg.get("max_largest_ratio_warn", 0.10),
                max_fill_added_ratio=post_cfg.get("max_fill_added_ratio_warn", 0.05),
            )
            if anomaly.get("is_anomalous"):
                warnings = anomaly.get("warnings", [])
                raise ValueError("后处理面积异常，已停止后台流程：" + "; ".join(warnings))
            self._check_cancelled()

            stage = "skeleton_generation"
            self.progress.emit(35, "阶段 2/4：生成骨架")
            from roadnet.optimized_skeleton import skeletonize_medial_axis, skeletonize_thin
            skel_cfg = self.config.get("skeleton", {})
            if skel_cfg.get("method", "medial_axis") == "medial_axis":
                raw_skeleton = skeletonize_medial_axis(clean_mask)
            else:
                raw_skeleton = skeletonize_thin(clean_mask)
            self._check_cancelled()

            stage = "skeleton_optimization"
            self.progress.emit(60, "阶段 3/4：优化骨架")
            from roadnet.optimized_skeleton import optimize_skeleton
            opt = optimize_skeleton(
                clean_mask, raw_skeleton,
                min_center_dist=skel_cfg.get("min_center_dist", 2.0),
                border_margin=skel_cfg.get("border_margin", 10),
                min_branch_length=skel_cfg.get("min_branch_length", 20),
                max_connect_dist=skel_cfg.get("max_connect_dist", 25),
                max_connect_angle=skel_cfg.get("max_connect_angle", 45),
                min_line_mask_overlap=skel_cfg.get("min_line_mask_overlap", 0.65),
                junction_cluster_radius=skel_cfg.get("junction_cluster_radius", 10),
            )
            optimized = opt["optimized_skeleton"] if isinstance(opt, dict) else opt
            if self.valid_image_mask is not None:
                optimized = apply_valid_image_mask(optimized, self.valid_image_mask)
            from roadnet.skeleton_artifacts import (
                build_skeleton_optimize_report, save_skeleton_artifacts,
            )
            skeleton_report = build_skeleton_optimize_report(
                clean_mask,
                raw_skeleton,
                optimized,
                min_branch_length=skel_cfg.get("min_branch_length", 20),
                min_center_dist=opt.get("stats", {}).get(
                    "effective_min_center_dist", skel_cfg.get("min_center_dist", 2.0)
                ),
                endpoint_connect_distance=skel_cfg.get("max_connect_dist", 25),
                skeleton_state_input="raw",
            )
            save_skeleton_artifacts(
                os.path.join(self.output_root, "skeleton"),
                clean_mask,
                raw_skeleton,
                optimized,
                skeleton_report,
            )
            self._check_cancelled()

            stage = "graph_generation"
            self.progress.emit(82, "阶段 4/4：生成并分析 graph")
            from roadnet.graph_build import build_graph_from_skeleton
            graph_cfg = self.config.get("graph", {})
            graph_result = build_graph_from_skeleton(
                skeleton=optimized,
                graph_editor=None,
                output_dir=os.path.join(self.output_root, "graph_build"),
                config=self.config,
                processed_mask=clean_mask,
                run_optimization=graph_cfg.get("enable_graph_line_optimizer", False),
            )
            if not graph_result.success:
                raise RuntimeError(
                    f"graph_build failed at {graph_result.stage}: "
                    + "; ".join(graph_result.errors)
                )
            self._check_cancelled()
            final_edges = graph_result.optimized_edges or graph_result.raw_edges
            self.progress.emit(100, "一键生成路网完成")
            pipeline_warnings = list(anomaly.get("warnings", []))
            if skeleton_report["removed_ratio"] > 0.60:
                pipeline_warnings.append("骨架优化删除比例超过 60%，可能发生过度剪枝")
            preview_scale = min(1.0, 3000.0 / max(clean_mask.shape))
            preview_size = (
                max(1, int(clean_mask.shape[1] * preview_scale)),
                max(1, int(clean_mask.shape[0] * preview_scale)),
            )
            mask_preview = cv2.resize(
                clean_mask, preview_size, interpolation=cv2.INTER_NEAREST,
            ) if preview_scale < 1.0 else clean_mask
            skeleton_preview = cv2.resize(
                optimized, preview_size, interpolation=cv2.INTER_NEAREST,
            ) if preview_scale < 1.0 else optimized
            self.finished.emit(PipelineResult(
                processed_mask=clean_mask,
                raw_skeleton=raw_skeleton,
                optimized_skeleton=optimized,
                skeleton_report=skeleton_report,
                nodes=graph_result.raw_nodes,
                edges=final_edges,
                connectivity=graph_result.connectivity or {},
                warnings=pipeline_warnings,
                mask_preview=mask_preview,
                skeleton_preview=skeleton_preview,
            ))
        except InterruptedError as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            self.failed.emit(stage, str(exc), traceback.format_exc())
