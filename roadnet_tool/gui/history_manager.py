"""
全局撤销/重做管理器

覆盖所有阶段的操作：道路分割、ROI/Ignore、Mask 编辑、Skeleton、Graph 编辑、路径规划。
每次会修改项目状态的操作前调用 push_state(action_name) 保存全局快照。
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPen, QColor


class GlobalHistoryManager:
    """全局撤销/重做管理器"""

    def __init__(self, max_steps: int = 50):
        self._undo_stack: List[Dict[str, Any]] = []
        self._redo_stack: List[Dict[str, Any]] = []
        self._max_steps = max_steps
        self._canvas = None        # 由 MainWindow 注入
        self._layer_manager = None
        self._graph_editor = None

    # ---- 注入依赖 ----
    def inject(self, canvas, layer_manager, graph_editor, main_window):
        self._canvas = canvas
        self._layer_manager = layer_manager
        self._graph_editor = graph_editor
        self._mw = main_window

    # ---- 属性 ----
    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    @property
    def undo_count(self) -> int:
        return len(self._undo_stack)

    @property
    def redo_count(self) -> int:
        return len(self._redo_stack)

    # ---- 核心 API ----

    def push_state(self, action_name: str):
        """保存当前全局状态快照到撤销栈"""
        state = self.capture_state()
        state["__action__"] = action_name
        self._undo_stack.append(state)
        if len(self._undo_stack) > self._max_steps:
            self._undo_stack.pop(0)
        # 新操作后清空 redo 栈
        self._redo_stack.clear()
        print(f"[DEBUG][History] push_state action={action_name} undo_depth={len(self._undo_stack)}")

    def push_mask_patch(self, action_name: str, rect, before, after):
        """Store one local mask edit without copying a multi-gigapixel mask."""
        state = {
            "__kind__": "mask_patch",
            "__action__": action_name,
            "rect": tuple(int(value) for value in rect),
            "before": before.copy(),
            "after": after.copy(),
        }
        self._undo_stack.append(state)
        if len(self._undo_stack) > self._max_steps:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        print(f"[DEBUG][History] push_mask_patch action={action_name} rect={state['rect']}")

    def push_mask_files(self, action_name: str, before_path: str, after_path: str,
                        before_preview: str = "", after_preview: str = ""):
        state = {
            "__kind__": "mask_file",
            "__action__": action_name,
            "before_path": before_path,
            "after_path": after_path,
            "before_preview": before_preview,
            "after_preview": after_preview,
        }
        self._undo_stack.append(state)
        if len(self._undo_stack) > self._max_steps:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _restore_mask_file(self, state, use_after: bool):
        import cv2
        suffix = "after" if use_after else "before"
        path = state[f"{suffix}_path"]
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Unable to restore mask history file: {path}")
        preview_path = state.get(f"{suffix}_preview", "")
        preview = cv2.imread(preview_path, cv2.IMREAD_GRAYSCALE) if preview_path else None
        self._layer_manager.set_layer_data("mask", mask, preview_data=preview)
        project = getattr(self._mw, "_large_image_project", None)
        if project is not None:
            project.global_mask_path = path
            project.save()
        if self._canvas is not None:
            self._canvas.refresh_scene()
            self._canvas.viewport().update()

    def _restore_mask_patch(self, state, use_after: bool):
        lm = self._layer_manager
        if lm is None:
            return
        mask = lm.get_layer_data("mask")
        if mask is None:
            return
        x0, y0, x1, y1 = state["rect"]
        mask[y0:y1, x0:x1] = state["after" if use_after else "before"]
        lm.update_layer_preview_region("mask", state["rect"])
        lm.invalidate_layer_cache("mask")
        lm.layer_changed.emit("layer_road_mask")
        if self._canvas is not None:
            self._canvas.viewport().update()

    def undo(self) -> Optional[str]:
        """撤销一步，返回操作名称；栈空返回 None"""
        if not self._undo_stack:
            return None
        state = self._undo_stack.pop()
        action_name = state.get("__action__", "unknown")
        if state.get("__kind__") == "mask_patch":
            self._restore_mask_patch(state, use_after=False)
            self._redo_stack.append(state)
            return action_name
        if state.get("__kind__") == "mask_file":
            self._restore_mask_file(state, use_after=False)
            self._redo_stack.append(state)
            return action_name
        # 当前状态保存到 redo 栈
        current = self.capture_state()
        current["__action__"] = action_name
        self._redo_stack.append(current)
        # 恢复上一个状态
        print(f"[DEBUG][History] undo action={action_name} undo_depth={len(self._undo_stack)}")
        self.restore_state(state)
        return action_name

    def redo(self) -> Optional[str]:
        """重做一步，返回操作名称；栈空返回 None"""
        if not self._redo_stack:
            return None
        state = self._redo_stack.pop()
        action_name = state.get("__action__", "unknown")
        if state.get("__kind__") == "mask_patch":
            self._restore_mask_patch(state, use_after=True)
            self._undo_stack.append(state)
            return action_name
        if state.get("__kind__") == "mask_file":
            self._restore_mask_file(state, use_after=True)
            self._undo_stack.append(state)
            return action_name
        # 当前状态保存回 undo 栈
        current = self.capture_state()
        current["__action__"] = action_name
        self._undo_stack.append(current)
        # 恢复 redo 状态
        print(f"[DEBUG][History] redo action={action_name} redo_depth={len(self._redo_stack)}")
        self.restore_state(state)
        return action_name

    def clear(self):
        self._undo_stack.clear()
        self._redo_stack.clear()

    # ---- 状态捕获与恢复 ----

    def capture_state(self) -> Dict[str, Any]:
        """捕获当前所有项目状态"""
        canvas = self._canvas
        lm = self._layer_manager
        ge = self._graph_editor
        mw = self._mw

        state: Dict[str, Any] = {}

        # Canvas 数据
        if canvas is not None:
            state["positive_points"] = copy.deepcopy(list(canvas.positive_points))
            state["negative_points"] = copy.deepcopy(list(canvas.negative_points))
            # ROI 数据 — 转换为可序列化的点列表
            from PySide6.QtCore import QRectF
            roi_poly_data = []
            for poly in canvas._roi_polygons:
                pts = [(p.x(), p.y()) for p in poly]
                roi_poly_data.append(pts)
            state["roi_polygons"] = roi_poly_data
            # Ignore 数据 — 统一保存为多边形点列表
            ig_polys = []
            for poly in canvas._ignore_polygons:
                try:
                    pts = [(p.x(), p.y()) for p in poly]
                    ig_polys.append(pts)
                except (RuntimeError, AttributeError):
                    pass
            state["ignore_polys"] = ig_polys

            # Ignore 兼容旧矩形（_ignore_rects_deprecated）
            ig_rects_raw = getattr(canvas, '_ignore_rects_deprecated', None)
            state["ignore_rects"] = list(ig_rects_raw) if ig_rects_raw else []

            # Mask
            # current_mask aliases LayerManager's mask; storing it again used
            # to duplicate every large global mask in each history snapshot.

            # Skeleton
            skel = canvas.skeleton_result
            state["skeleton_result"] = copy.deepcopy(skel) if skel is not None else None

            # Main-road seed strokes (large-image)
            if hasattr(canvas, "get_main_road_seed_stroke_dicts"):
                state["main_road_seed_strokes"] = copy.deepcopy(
                    canvas.get_main_road_seed_stroke_dicts()
                )
            elif hasattr(canvas, "get_main_road_seed_strokes"):
                state["main_road_seed_strokes"] = copy.deepcopy(
                    canvas.get_main_road_seed_strokes()
                )
            else:
                state["main_road_seed_strokes"] = []

        # LayerManager 数据
        if lm is not None:
            # Skeleton layer data
            skel_data = lm.get_layer_data("skeleton")
            state["skeleton_layer_data"] = None if skel_data is None else (
                skel_data.copy() if not isinstance(skel_data, dict) else copy.deepcopy(skel_data)
            )
            # Mask layer data
            mask_data = lm.get_layer_data("mask")
            state["mask_layer_data"] = None if mask_data is None else mask_data.copy()

            # 图层可见性
            vis = {}
            for name, layer in lm.layers().items():
                vis[name] = layer.visible
            state["layer_visibility"] = vis

        # GraphEditor 数据
        if ge is not None:
            state["graph_nodes"] = copy.deepcopy(list(ge.nodes))
            state["graph_edges"] = copy.deepcopy(list(ge.edges))
            state["graph_next_node_id"] = getattr(ge, '_next_node_id', 0)
            state["graph_next_edge_id"] = getattr(ge, '_next_edge_id', 0)

        # ★ TaskPoints 数据
        if mw is not None:
            tps = getattr(mw, '_task_points', None)
            if tps:
                state["task_points"] = copy.deepcopy(list(tps))
            else:
                state["task_points"] = []
            sps = getattr(mw, '_snapped_points', None)
            if sps:
                state["snapped_points"] = copy.deepcopy(list(sps))
            else:
                state["snapped_points"] = []

        # MainWindow 状态
        if mw is not None:
            state["current_stage"] = getattr(mw, '_current_stage', 'import')
            state["current_tool"] = canvas.current_tool if canvas else 'pan'
            state["stage_completed"] = copy.deepcopy(getattr(mw, '_stage_completed', set()))
            state["clean_mode"] = getattr(mw, '_clean_mode', True)
            for name in ("raw_skeleton", "optimized_skeleton", "current_skeleton"):
                value = getattr(mw, name, None)
                state[name] = None if value is None else value.copy()
            state["skeleton_state"] = getattr(mw, "skeleton_state", "none")

        return state

    def restore_state(self, state: Dict[str, Any]):
        """从快照恢复所有项目状态并刷新全部显示"""
        from PySide6.QtCore import QPointF, QRectF
        from PySide6.QtGui import QPolygonF
        from roadnet.region_edit import PolygonRegion

        canvas = self._canvas
        lm = self._layer_manager
        ge = self._graph_editor
        mw = self._mw

        # ---- Canvas 数据恢复 ----
        if canvas is not None:
            canvas._positive_points = copy.deepcopy(state.get("positive_points", []))
            canvas._negative_points = copy.deepcopy(state.get("negative_points", []))

            # ROI 多边形恢复
            canvas._roi_polygons = []
            for poly_data in state.get("roi_polygons", []):
                if isinstance(poly_data, QPolygonF):
                    canvas._roi_polygons.append(QPolygonF(poly_data))
                elif isinstance(poly_data, list):
                    poly = QPolygonF([QPointF(p[0], p[1]) for p in poly_data])
                    canvas._roi_polygons.append(poly)
            canvas._roi_regions = [
                PolygonRegion.create("roi", [(p.x(), p.y()) for p in poly])
                for poly in canvas._roi_polygons if len(poly) >= 3
            ]

            # ROI items 会在 refresh_scene 中重建
            canvas._roi_items.clear()

            # Ignore 矩形数据（兼容旧存档 → 转存 _ignore_rects_deprecated）
            canvas._ignore_rects_deprecated = []
            for rdata in state.get("ignore_rects", []):
                if isinstance(rdata, (list, tuple)) and len(rdata) == 4:
                    canvas._ignore_rects_deprecated.append(
                        (float(rdata[0]), float(rdata[1]), float(rdata[2]), float(rdata[3]))
                    )

            # Ignore 多边形数据（★ 统一为 _ignore_polygons）
            canvas._ignore_polygons = []
            canvas._ignore_items = []
            for pdata in state.get("ignore_polys", []):
                if isinstance(pdata, QPolygonF):
                    poly = QPolygonF(pdata)
                elif isinstance(pdata, list):
                    poly = QPolygonF([QPointF(p[0], p[1]) for p in pdata])
                else:
                    continue
                canvas._ignore_polygons.append(poly)
                # QGraphicsPolygonItem 将在 refresh_scene 中重建
            canvas._ignore_regions = [
                PolygonRegion.create("ignore", [(p.x(), p.y()) for p in poly])
                for poly in canvas._ignore_polygons if len(poly) >= 3
            ]

        # ---- LayerManager 数据恢复 ----
        if lm is not None:
            # Mask
            mask_data = state.get("mask_layer_data")
            if mask_data is not None:
                lm.set_layer_data("mask", mask_data)
            else:
                # Only clear if currently has data
                d = lm.get_layer_data("mask")
                if d is not None:
                    lm.clear_layer("mask")

            # Skeleton
            skel_data = state.get("skeleton_layer_data")
            if skel_data is not None:
                lm.set_layer_data("skeleton", skel_data)
            else:
                d = lm.get_layer_data("skeleton")
                if d is not None:
                    lm.clear_layer("skeleton")

            # Skeleton result on canvas
            if canvas is not None:
                sr = state.get("skeleton_result")
                canvas.skeleton_result = sr
                if "main_road_seed_strokes" in state and hasattr(
                    canvas, "set_main_road_seed_stroke_dicts"
                ):
                    canvas.set_main_road_seed_stroke_dicts(
                        state.get("main_road_seed_strokes") or [], emit=False
                    )

            # 图层可见性
            vis = state.get("layer_visibility", {})
            for name, layer in lm.layers().items():
                should_show = vis.get(name, False)
                if should_show:
                    lm.show_layer(name)
                else:
                    lm.hide_layer(name)

        # ---- GraphEditor 数据恢复 ----
        if ge is not None:
            ge._nodes = copy.deepcopy(state.get("graph_nodes", []))
            ge._edges = copy.deepcopy(state.get("graph_edges", []))
            ge._next_node_id = state.get("graph_next_node_id", 0)
            ge._next_edge_id = state.get("graph_next_edge_id", 0)
            ge.clear_selection()
            ge._edge_start_node = None
            ge._manual_edge_points.clear()
            ge._dragging_node = None

        # ---- TaskPoints 数据恢复 ----
        if mw is not None:
            if "task_points" in state:
                mw._task_points = copy.deepcopy(state["task_points"])
            if "snapped_points" in state:
                mw._snapped_points = copy.deepcopy(state["snapped_points"])

        # ---- MainWindow 状态恢复 ----
        if mw is not None:
            if "current_stage" in state:
                mw._current_stage = state["current_stage"]
            if "stage_completed" in state:
                mw._stage_completed = state["stage_completed"]
            if "clean_mode" in state:
                mw._clean_mode = state["clean_mode"]
            for name in ("raw_skeleton", "optimized_skeleton", "current_skeleton"):
                value = state.get(name)
                setattr(mw, name, None if value is None else value.copy())
            mw.skeleton_state = state.get("skeleton_state", "none")

        # ---- 全景刷新 ----
        self._refresh_all()

    def _refresh_all(self):
        """恢复状态后刷新所有图层和面板"""
        mw = self._mw
        if mw is None:
            return

        canvas = self._canvas

        # 1. 刷新 scene
        if canvas is not None:
            canvas.refresh_scene()
            # 同步左侧复选框
            if hasattr(mw, '_sync_layer_checkboxes'):
                mw._sync_layer_checkboxes()

        # 2. 渲染 graph
        if hasattr(mw, '_render_graph_to_scene'):
            mw._render_graph_to_scene()

        # 3. ★ 渲染 task_points
        if hasattr(mw, '_render_task_points_to_scene'):
            mw._render_task_points_to_scene()

        # 4. 更新统计
        if hasattr(mw, '_update_graph_stats'):
            mw._update_graph_stats()

        # 4. 采样点计数
        if canvas is not None:
            canvas.sample_points_changed.emit(
                len(canvas._positive_points), len(canvas._negative_points)
            )

        # 5. 更新右侧面板
        if hasattr(mw, '_param_panel') and mw._param_panel is not None:
            mw._param_panel.update_counts(
                pos=len(canvas._positive_points) if canvas else 0,
                neg=len(canvas._negative_points) if canvas else 0,
                roi=len(canvas.get_roi_regions()) if canvas else 0,
                ignore=len(canvas.get_ignore_regions()) if canvas else 0,
            )

        # 区域修正稳定模式的核心层在 undo/redo 后仍必须可见。
        if getattr(mw, "_current_stage", None) == "edit" and self._layer_manager is not None:
            for name in ("layer_road_mask", "layer_roi", "layer_ignore"):
                self._layer_manager.show_layer(name)

        # 6. 更新状态栏
        if hasattr(mw, '_status_bar') and mw._status_bar is not None:
            if mw._graph_editor is not None:
                stats = mw._graph_editor.get_stats()
                mw._status_bar.update_nodes(stats["node_count"])
                mw._status_bar.update_edges(stats["edge_count"])

        # 7. 刷新视口
        if canvas is not None:
            canvas.viewport().update()
