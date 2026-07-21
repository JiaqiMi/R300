"""
路网生成工具 - 主入口 V2.4 + V3.1 + V3.2

V0:   基础命令行 + 图像读取
V1:   交互采样 + 色彩空间道路分割
V1.5: 正负样本交互式分割 + 距离约束 + AND 组合
V2.1: ROI 区域限制 + 人工编辑 mask + 后处理增强
V2.2: ignore 区域屏蔽 + 参数调优 + 负样本增强
V2.3: ROI 多区域支持
V2.4: 折线补线修复 + 温和后处理
V3:   骨架化 + 路网图构建（nodes.csv / edges.csv / road_graph.json）
V3.1: 自动 skeleton 优化 + draft graph 提取
V3.2: 人工 graph 编辑

完整流水线示例：
python main.py -i data/sample_images/002.png -o outputs/test_run ^
    --draw-roi --draw-ignore --edit-mask --repair-mask ^
    --postprocess --optimize-skeleton --extract-draft-graph ^
    --edit-graph --save-intermediate
"""

import argparse
import os
import sys

import cv2

from roadnet.utils import load_config, ensure_dir
from roadnet.io_utils import read_image_rgb, save_image
from roadnet.color_segment import ColorSampler, segment_road
from roadnet.visualization import (
    overlay_mask,
    draw_sample_markers,
    save_postprocess_compare,
    draw_roi_regions_on_image,
)
from roadnet.postprocess import clean_pipeline

# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="路网生成工具 V2.4+V3 — 折线补线 + 骨架化预览 + 温和后处理",
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="输入图像路径（卫星图/航拍图）",
    )
    parser.add_argument(
        "--output", "-o",
        default="./outputs",
        help="输出目录（默认 ./outputs）",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="配置文件路径（默认使用 config/default.yaml）",
    )

    # ---- V2.x 参数 ----
    parser.add_argument(
        "--draw-roi",
        action="store_true",
        help="开启 ROI 多边形绘制，限制道路检测区域",
    )
    parser.add_argument(
        "--draw-ignore",
        action="store_true",
        help="开启 ignore 区域屏蔽（矩形/多边形），排除误检区域",
    )
    parser.add_argument(
        "--edit-mask",
        action="store_true",
        help="开启人工编辑 mask（画笔补线/橡皮擦删除）",
    )

    # ---- V2.4 参数 ----
    parser.add_argument(
        "--repair-mask",
        action="store_true",
        help="开启折线补线修复（左键点击中心点，Enter 画道路）",
    )
    parser.add_argument(
        "--road-width",
        type=int,
        default=25,
        help="折线补线的默认道路宽度，单位像素（默认 25）",
    )

    # ---- 后处理 ----
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="开启后处理增强流水线",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="保存后处理各步骤中间结果",
    )

    # ---- V3 preview 参数 ----
    parser.add_argument(
        "--skeleton-preview",
        action="store_true",
        help="开启骨架化预览（骨架+毛刺删除+节点检测）",
    )
    parser.add_argument(
        "--min-branch-length",
        type=int,
        default=30,
        help="骨架最短保留分支长度，小于此值的毛刺会被删除（默认 30）",
    )

    # ---- V3 路网图生成 ----
    parser.add_argument(
        "--build-graph",
        action="store_true",
        help="从骨架生成路网图（nodes.csv + edges.csv + road_graph.json）",
    )

    # ---- V3.1 skeleton 优化 ----
    parser.add_argument(
        "--optimize-skeleton",
        action="store_true",
        help="开启自动 skeleton 优化（距离变换过滤+边界裁剪+毛刺删除+节点合并+自动连接）",
    )

    # ---- V3.1 draft graph 提取 ----
    parser.add_argument(
        "--extract-draft-graph",
        action="store_true",
        help="从优化骨架提取 draft graph（节点+边提取+合并+简化）",
    )

    # ---- V3.2 graph 编辑器 ----
    parser.add_argument(
        "--edit-graph",
        action="store_true",
        help="开启人工 graph 编辑（添加/删除/移动/合并/拆分节点和边）",
    )
    parser.add_argument(
        "--manual-graph",
        action="store_true",
        help="同 --edit-graph（向后兼容别名）",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# V1.5 阶段：交互采样 + 分割
# ---------------------------------------------------------------------------


def stage_v15_sample_and_segment(image_rgb, output_dir, config):
    """
    V2.2 Stage 1：正负样本交互采样 + 色彩空间分割（负样本增强版）。

    负样本采集指导：草坪、跑道、建筑屋顶、树林、文字、田地等。
    返回 mask。
    """
    seg_cfg = config.get("segment", {})
    print(f"[INFO] 分割模式: {seg_cfg.get('mode', 'combined')}")
    print(f"[INFO] 组合方式: {seg_cfg.get('combine_method', 'and')}")
    use_neg = seg_cfg.get("use_negative_samples", True)
    print(f"[INFO] 正负样本约束: {'启用' if use_neg else '关闭'}")
    print(f"[INFO] 正样本距离阈值: {seg_cfg.get('positive_distance_threshold', 20)}")
    print(f"[INFO] 负样本排除边距: {seg_cfg.get('negative_margin', 6)}")

    # ---- 交互采样 ----
    print("\n[INFO] 操作说明:")
    print("        左键点击: 添加道路正样本（绿色圆圈）")
    print("        右键点击: 添加非道路负样本（红色叉号）")
    print("        c 键:     清空所有样本点")
    print("        u 键:     撤销上一个样本点")
    print("        Enter:    确认并开始分割")
    print("        Esc / 关闭窗口: 退出程序")
    print("\n[提示] V2.2 负样本增强 — 建议采集方式：")
    print("       - 正样本: 点击路面（沥青/水泥）不同光照位置 8~12 个点")
    print("       - 负样本: 点击 草坪、跑道、建筑屋顶、树林、文字、田地 等")
    print("                  每类至少 2~3 个点，共 10~20 个负样本点")

    sampler = ColorSampler(
        image_rgb=image_rgb,
        sample_radius=seg_cfg.get("sample_radius", 3),
    )
    sampler.run()

    if sampler.is_cancelled or not sampler.has_samples():
        print("[INFO] 用户取消操作或未采集正样本，退出。")
        sys.exit(0)

    pos_samples = sampler.positive_samples
    neg_samples = sampler.negative_samples
    print(f"\n[INFO] 正样本数: {len(pos_samples)}, 负样本数: {len(neg_samples)}")

    # ---- 保存 sample_points.png ----
    sample_img = draw_sample_markers(
        image_rgb,
        pos_points=sampler.pos_pixel_points,
        neg_points=sampler.neg_pixel_points,
    )
    sample_path = os.path.join(output_dir, "sample_points.png")
    save_image(sample_path, sample_img)
    print(f"[INFO] 已保存采样点图: {sample_path}")

    # ---- 执行分割 ----
    print("[INFO] 正在执行道路分割...")
    mask = segment_road(image_rgb, pos_samples, neg_samples, seg_cfg)
    road_ratio = mask.mean() / 255 * 100
    print(f"[INFO] 道路像素占比: {road_ratio:.2f}%")

    if road_ratio < 1:
        print("[WARN] 道路像素占比过低，请检查采样点是否准确。")

    # ---- 保存 raw 结果 ----
    mask_path = os.path.join(output_dir, "road_mask_raw.png")
    save_image(mask_path, mask)
    print(f"[INFO] 已保存 mask: {mask_path}")

    overlay_alpha = seg_cfg.get("overlay_alpha", 0.45)
    overlay = overlay_mask(image_rgb, mask, alpha=overlay_alpha)
    overlay_path = os.path.join(output_dir, "road_overlay_raw.png")
    save_image(overlay_path, overlay)
    print(f"[INFO] 已保存 overlay: {overlay_path}")

    return mask


# ---------------------------------------------------------------------------
# V2.1 阶段 ROI：绘制 + 应用
# ---------------------------------------------------------------------------


def stage_roi(image_rgb, mask, output_dir, config):
    """
    V2.3 ROI 阶段：交互式绘制多区域 ROI 多边形，应用后生成受限 mask。

    返回 (roi_applied_mask) —— 如果取消则返回原 mask。
    """
    from roadnet.roi import RoiDrawer

    print("\n" + "=" * 60)
    print("[ROI] 进入 ROI 区域绘制模式（支持多区域）")
    print("[ROI] 左键点击添加顶点 | Enter 闭合当前区域 | 再按 Enter 确认全部 | c 清空 | u 撤销 | Esc 退出")
    print("=" * 60)

    drawer = RoiDrawer(image_rgb)
    drawer.run()

    if drawer.is_cancelled or not drawer.is_confirmed:
        print("[ROI] 已取消 ROI 绘制，跳过 ROI 限制。")
        return mask

    roi_mask = drawer.roi_mask
    if roi_mask is None:
        print("[ROI] ROI mask 生成失败，跳过 ROI 限制。")
        return mask

    regions = drawer.roi_regions
    print(f"[ROI] 已绘制 {len(regions)} 个 ROI 区域")

    # 保存 roi_mask.png
    if config.get("roi", {}).get("save_roi", True):
        roi_path = os.path.join(output_dir, "roi_mask.png")
        save_image(roi_path, roi_mask)
        print(f"[ROI] 已保存 ROI mask: {roi_path}")

    # 应用 ROI：road_mask_roi = road_mask_raw & roi_mask
    roi_applied = cv2.bitwise_and(mask, roi_mask)
    roi_mask_path = os.path.join(output_dir, "road_mask_roi.png")
    save_image(roi_mask_path, roi_applied)
    print(f"[ROI] 已保存 ROI 限制后 mask: {roi_mask_path}")

    alpha = config.get("visualization", {}).get("overlay_alpha", 0.45)
    roi_overlay = overlay_mask(image_rgb, roi_applied, alpha=alpha)
    roi_overlay_path = os.path.join(output_dir, "road_overlay_roi.png")
    save_image(roi_overlay_path, roi_overlay)
    print(f"[ROI] 已保存 ROI 限制后 overlay: {roi_overlay_path}")

    # 保存 ROI 边界可视化图（多区域）
    if regions:
        pixel_regions = []
        for region in regions:
            pixel_verts = [(int(round(x)), int(round(y))) for (x, y) in region]
            pixel_regions.append(pixel_verts)
        roi_vis = draw_roi_regions_on_image(image_rgb, pixel_regions)
        roi_vis_path = os.path.join(output_dir, "roi_visual.png")
        save_image(roi_vis_path, roi_vis)
        print(f"[ROI] 已保存 ROI 可视化图: {roi_vis_path}")

    road_ratio = roi_applied.mean() / 255 * 100
    print(f"[ROI] ROI 限制后道路像素占比: {road_ratio:.2f}%")

    return roi_applied


# ---------------------------------------------------------------------------
# V2.2 阶段 IGNORE：ignore 区域屏蔽
# ---------------------------------------------------------------------------


def stage_ignore(image_rgb, mask, output_dir, config):
    """
    V2.2 Ignore 阶段：交互式绘制矩形/多边形屏蔽区域。

    返回 ignore_applied_mask —— 应用屏蔽后的 mask。
    """
    from roadnet.ignore import IgnoreDrawer

    print("\n" + "=" * 60)
    print("[IGNORE] 进入 ignore 区域屏蔽模式")
    print("[IGNORE] 左键拖拽=矩形屏蔽 | p=多边形 | r=矩形 | c=清空 | u=撤销 | Enter=确认 | Esc=取消")
    print("=" * 60)

    drawer = IgnoreDrawer(image_rgb, mask)
    drawer.run()

    if drawer.is_cancelled or not drawer.is_confirmed:
        print("[IGNORE] 已取消 ignore 屏蔽，跳过。")
        return mask

    ignore_mask = drawer.ignore_mask
    if ignore_mask is None or ignore_mask.max() == 0:
        print("[IGNORE] 未绘制任何屏蔽区域，跳过。")
        return mask

    # 保存 ignore_mask.png
    if config.get("ignore", {}).get("save_ignore", True):
        ignore_path = os.path.join(output_dir, "ignore_mask.png")
        save_image(ignore_path, ignore_mask)
        print(f"[IGNORE] 已保存 ignore mask: {ignore_path}")

    # 应用屏蔽：road_mask_ignore = mask & ~ignore_mask
    ignore_applied = cv2.bitwise_and(mask, cv2.bitwise_not(ignore_mask))
    ignore_mask_path = os.path.join(output_dir, "road_mask_ignore.png")
    save_image(ignore_mask_path, ignore_applied)
    print(f"[IGNORE] 已保存 ignore 屏蔽后 mask: {ignore_mask_path}")

    alpha = config.get("visualization", {}).get("overlay_alpha", 0.45)
    ignore_overlay = overlay_mask(image_rgb, ignore_applied, alpha=alpha)
    ignore_overlay_path = os.path.join(output_dir, "road_overlay_ignore.png")
    save_image(ignore_overlay_path, ignore_overlay)
    print(f"[IGNORE] 已保存 ignore 屏蔽后 overlay: {ignore_overlay_path}")

    road_ratio = ignore_applied.mean() / 255 * 100
    print(f"[IGNORE] ignore 屏蔽后道路像素占比: {road_ratio:.2f}%")

    return ignore_applied


# ---------------------------------------------------------------------------
# V2.1 阶段 EDIT：人工编辑 mask
# ---------------------------------------------------------------------------


def stage_edit(image_rgb, mask, output_dir, config):
    """
    V2.1 编辑阶段：画笔补道路 / 橡皮擦删误检。

    返回 edited_mask。
    """
    from roadnet.mask_editor import MaskEditor

    print("\n" + "=" * 60)
    print("[EDIT] 进入人工编辑模式")
    print("[EDIT] 左键拖动=补道路 | 右键拖动=擦除 | +/-=画笔大小 | s=保存 | u=撤销 | Esc=退出")
    print("=" * 60)

    edit_cfg = config.get("edit", {})
    brush_radius = edit_cfg.get("brush_radius", 8)
    max_undo = edit_cfg.get("max_undo_steps", 20)

    editor = MaskEditor(
        image_rgb=image_rgb,
        mask=mask,
        brush_radius=brush_radius,
        max_undo_steps=max_undo,
    )
    editor.run()

    edited = editor.edited_mask

    # 保存
    edited_path = os.path.join(output_dir, "road_mask_edited.png")
    save_image(edited_path, edited)
    print(f"[EDIT] 已保存编辑后 mask: {edited_path}")

    alpha = config.get("visualization", {}).get("overlay_alpha", 0.45)
    edited_overlay = overlay_mask(image_rgb, edited, alpha=alpha)
    edited_overlay_path = os.path.join(output_dir, "road_overlay_edited.png")
    save_image(edited_overlay_path, edited_overlay)
    print(f"[EDIT] 已保存编辑后 overlay: {edited_overlay_path}")

    road_ratio = edited.mean() / 255 * 100
    print(f"[EDIT] 编辑后道路像素占比: {road_ratio:.2f}%")

    return edited


# ---------------------------------------------------------------------------
# V2.4 阶段 REPAIR：折线补线修复
# ---------------------------------------------------------------------------


def stage_repair(image_rgb, mask, output_dir, config, road_width=25):
    """
    V2.4 折线补线阶段：交互式点击道路中心点，按道路宽度画线补充漏检。

    返回 repaired_mask。
    """
    from roadnet.repair import PolylineRepairEditor

    road_width_cfg = config.get("repair", {}).get("road_width", road_width)

    print("\n" + "=" * 60)
    print("[REPAIR] 进入折线补线修复模式")
    print(f"[REPAIR] 道路宽度={road_width_cfg}px | 左键=添加中心点 | Enter=画入mask | "
          "c=清空 | u=撤销 | s=保存 | Esc=退出")
    print("=" * 60)

    editor = PolylineRepairEditor(
        image_rgb=image_rgb,
        mask=mask,
        road_width=road_width_cfg,
    )
    editor.run()

    repaired = editor.repaired_mask

    # 保存
    repaired_path = os.path.join(output_dir, "road_mask_repaired.png")
    save_image(repaired_path, repaired)
    print(f"[REPAIR] 已保存补线后 mask: {repaired_path}")

    alpha = config.get("visualization", {}).get("overlay_alpha", 0.45)
    repaired_overlay = overlay_mask(image_rgb, repaired, alpha=alpha)
    repaired_overlay_path = os.path.join(output_dir, "road_overlay_repaired.png")
    save_image(repaired_overlay_path, repaired_overlay)
    print(f"[REPAIR] 已保存补线后 overlay: {repaired_overlay_path}")

    road_ratio = repaired.mean() / 255 * 100
    print(f"[REPAIR] 补线后道路像素占比: {road_ratio:.2f}%")

    return repaired


# ---------------------------------------------------------------------------
# V3 preview 阶段 SKELETON：骨架化预览
# ---------------------------------------------------------------------------


def stage_skeleton_preview(image_rgb, mask, output_dir, config,
                            min_branch_length=30, build_graph=False):
    """
    V3 骨架化阶段：骨架化 + 毛刺修剪 + 节点检测 + 可选路网图构建。

    无返回值（仅生成预览文件/图文件）。
    """
    from roadnet.skeleton_preview import run_skeleton_preview

    skel_cfg = config.get("skeleton", {})
    branch_len = skel_cfg.get("min_branch_length", min_branch_length)

    print("\n" + "=" * 60)
    print(f"[SKEL] 进入骨架化 + {'路网图构建' if build_graph else '预览'}模式")
    print(f"[SKEL] 最短分支长度={branch_len}px")
    print("=" * 60)

    run_skeleton_preview(
        image_rgb=image_rgb,
        mask=mask,
        output_dir=output_dir,
        min_branch_length=branch_len,
        build_graph=build_graph,
    )


# ---------------------------------------------------------------------------
# V3.1 阶段 OPTIMIZE SKELETON：自动 skeleton 优化
# ---------------------------------------------------------------------------


def stage_optimize_skeleton(image_rgb, mask, output_dir, config):
    """
    V3.1 骨架优化阶段：距离变换过滤 + 边界裁剪 + 毛刺删除 + 节点合并 + 自动连接。

    返回优化后的 skeleton。
    """
    from roadnet.optimized_skeleton import run_optimized_skeleton

    print("\n" + "=" * 60)
    print("[SKEL OPT] 进入 V3.1 自动 skeleton 优化模式")
    print("=" * 60)

    skeleton = run_optimized_skeleton(image_rgb, mask, output_dir, config)
    return skeleton


# ---------------------------------------------------------------------------
# V3.1 阶段 DRAFT GRAPH EXTRACT：draft graph 提取
# ---------------------------------------------------------------------------


def stage_draft_graph_extract(image_rgb, skeleton, output_dir, config):
    """
    V3.1 Draft graph 提取阶段：节点检测 + 合并 + 边追踪 + 精简。

    返回 (nodes, edges)。
    """
    from roadnet.draft_graph_extract import run_draft_graph_extract

    print("\n" + "=" * 60)
    print("[GRAPH] 进入 V3.1 draft graph 提取模式")
    print("=" * 60)

    nodes, edges = run_draft_graph_extract(image_rgb, skeleton, output_dir, config)
    return nodes, edges


# ---------------------------------------------------------------------------
# V3.2 阶段 GRAPH EDITOR：人工 graph 编辑
# ---------------------------------------------------------------------------


def stage_graph_editor(image_rgb, output_dir):
    """
    V3.2 Graph 编辑阶段：加载 draft graph，交互式编辑路网图。

    保存 final_graph.json / final_nodes.csv / final_edges.csv / final_graph_overlay.png
    """
    from roadnet.graph_editor import GraphEditor, load_draft_graph

    print("\n" + "=" * 60)
    print("[GRAPH EDIT] 进入 V3.2 人工 graph 编辑模式")
    print("=" * 60)

    draft_path = os.path.join(output_dir, "draft_graph.json")
    if not os.path.exists(draft_path):
        print(f"[GRAPH EDIT] 未找到 draft_graph.json ({draft_path})")
        print("[GRAPH EDIT] 请先运行 --extract-draft-graph 生成初稿")
        return

    draft_nodes, draft_edges = load_draft_graph(draft_path)

    editor = GraphEditor(
        image_rgb=image_rgb,
        draft_nodes=draft_nodes,
        draft_edges=draft_edges,
        output_dir=output_dir,
    )
    editor.run()

    if editor.is_cancelled:
        print("[GRAPH EDIT] 已取消编辑，不保存。")
    else:
        print("[GRAPH EDIT] 编辑完成。")
        if not editor.is_confirmed:
            # 用户通过 Esc 退出但未保存，询问式自动保存
            editor.save()


# ---------------------------------------------------------------------------
# V2.1 阶段 POSTPROCESS：后处理增强
# ---------------------------------------------------------------------------


def stage_postprocess(image_rgb, mask, output_dir, config, save_intermediate=False):
    """
    V2.1 后处理阶段：开运算/闭运算/孔洞填充/连通域筛选/边界平滑/细长噪声删除。

    返回 clean_mask。
    """
    print("\n" + "=" * 60)
    print("[POST] 进入后处理增强流水线")
    print("=" * 60)

    post_cfg = config.get("postprocess", {})

    print(f"[POST] 配置: open_kernel={post_cfg.get('open_kernel_size', 3)}, "
          f"close_kernel={post_cfg.get('close_kernel_size', 9)}, "
          f"fill_holes={post_cfg.get('fill_holes', True)}, "
          f"min_area={post_cfg.get('min_area', 800)}, "
          f"keep_largest={post_cfg.get('keep_largest_components', 5)}, "
          f"smooth_kernel={post_cfg.get('smooth_kernel_size', 5)}, "
          f"remove_thin={post_cfg.get('remove_thin_noise', True)}")

    clean_mask, intermediates = clean_pipeline(
        mask,
        post_cfg,
        save_intermediate=save_intermediate,
        output_dir=output_dir,
    )

    # 保存最终结果
    clean_path = os.path.join(output_dir, "road_mask_clean.png")
    save_image(clean_path, clean_mask)
    print(f"[POST] 已保存清理后 mask: {clean_path}")

    alpha = config.get("visualization", {}).get("overlay_alpha", 0.45)
    clean_overlay = overlay_mask(image_rgb, clean_mask, alpha=alpha)
    clean_overlay_path = os.path.join(output_dir, "road_overlay_clean.png")
    save_image(clean_overlay_path, clean_overlay)
    print(f"[POST] 已保存清理后 overlay: {clean_overlay_path}")

    road_ratio = clean_mask.mean() / 255 * 100
    print(f"[POST] 清理后道路像素占比: {road_ratio:.2f}%")

    # 保存前后对比图
    compare_path = os.path.join(output_dir, "postprocess_compare.png")
    save_postprocess_compare(image_rgb, mask, clean_mask, compare_path, alpha=alpha)
    print(f"[POST] 已保存后处理对比图: {compare_path}")

    road_ratio = clean_mask.mean() / 255 * 100
    print(f"[POST] 清理后道路像素占比: {road_ratio:.2f}%")

    return clean_mask


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # ---- V0: 读取图像 & 准备输出目录 ----
    print(f"[INFO] 读取图像: {args.image}")
    image_rgb = read_image_rgb(args.image)
    print(f"[INFO] 图像尺寸: {image_rgb.shape[1]} x {image_rgb.shape[0]}")

    output_dir = ensure_dir(args.output)
    print(f"[INFO] 输出目录: {output_dir}")

    # ---- 加载配置 ----
    config = load_config(args.config)

    # ===================================================================
    # Stage 1: V1.5 采样 + 分割（必须执行）— V2.2 负样本增强
    # ===================================================================
    print("\n" + "=" * 60)
    print("  STAGE 1: V2.2 正负样本交互采样 + 道路分割（负样本增强）")
    print("=" * 60)

    current_mask = stage_v15_sample_and_segment(image_rgb, output_dir, config)

    # ===================================================================
    # Stage 2: ROI 区域限制（可选）
    # ===================================================================
    if args.draw_roi:
        print("\n" + "=" * 60)
        print("  STAGE 2: ROI 区域限制")
        print("=" * 60)
        current_mask = stage_roi(image_rgb, current_mask, output_dir, config)
    else:
        print("\n[INFO] 未开启 --draw-roi，跳过 ROI 区域限制。")

    # ===================================================================
    # Stage 3: Ignore 区域屏蔽（V2.2 新增，可选）
    # ===================================================================
    if args.draw_ignore:
        print("\n" + "=" * 60)
        print("  STAGE 3: Ignore 区域屏蔽（矩形/多边形）")
        print("=" * 60)
        current_mask = stage_ignore(image_rgb, current_mask, output_dir, config)
    else:
        print("\n[INFO] 未开启 --draw-ignore，跳过 ignore 区域屏蔽。")

    # ===================================================================
    # Stage 4: 人工编辑 mask（可选）
    # ===================================================================
    if args.edit_mask:
        print("\n" + "=" * 60)
        print("  STAGE 4: 人工编辑 mask")
        print("=" * 60)
        current_mask = stage_edit(image_rgb, current_mask, output_dir, config)
    else:
        print("\n[INFO] 未开启 --edit-mask，跳过人工编辑。")

    # ===================================================================
    # Stage 5: 折线补线修复（V2.4 新增，可选）
    # ===================================================================
    if args.repair_mask:
        print("\n" + "=" * 60)
        print("  STAGE 5: 折线补线修复")
        print("=" * 60)
        current_mask = stage_repair(
            image_rgb, current_mask, output_dir, config,
            road_width=args.road_width,
        )
    else:
        print("\n[INFO] 未开启 --repair-mask，跳过折线补线修复。")

    # ---- 保存后处理前的 mask（用于 skeleton 生成，避免后处理膨胀影响） ----
    mask_before_postprocess = current_mask.copy()

    # ===================================================================
    # Stage 6: 后处理增强（可选）
    # ===================================================================
    if args.postprocess:
        print("\n" + "=" * 60)
        print("  STAGE 6: 后处理增强")
        print("=" * 60)
        current_mask = stage_postprocess(
            image_rgb, current_mask, output_dir, config,
            save_intermediate=args.save_intermediate,
        )
    else:
        print("\n[INFO] 未开启 --postprocess，跳过后处理增强。")

    # ===================================================================
    # Stage 7: 骨架化预览 + 路网图构建（V3，需要 --skeleton-preview）
    # ===================================================================
    if args.skeleton_preview:
        print("\n" + "=" * 60)
        print(f"  STAGE 7: {'骨架化预览 + 路网图构建' if args.build_graph else '骨架化预览'}（V3）")
        print("=" * 60)
        stage_skeleton_preview(
            image_rgb, current_mask, output_dir, config,
            min_branch_length=args.min_branch_length,
            build_graph=args.build_graph,
        )
    else:
        print("\n[INFO] 未开启 --skeleton-preview，跳过骨架化。")

    # ===================================================================
    # Stage 8: 自动 skeleton 优化（V3.1，需要 --optimize-skeleton）
    #           使用后处理前的 mask，避免 close 膨胀导致中心线偏移
    # ===================================================================
    optimized_skeleton = None
    if args.optimize_skeleton:
        print("\n" + "=" * 60)
        print("  STAGE 8: 自动 skeleton 优化（V3.1）")
        print("=" * 60)
        # 优先使用后处理前的 mask（人工编辑/补线的原始结果）
        mask_for_skeleton = mask_before_postprocess
        pre_px = int(mask_for_skeleton.sum() / 255)
        post_px = int(current_mask.sum() / 255)
        print(f"[INFO] Skeleton 使用后处理前 mask ({pre_px} px)，非后处理后 mask ({post_px} px)")
        print(f"[INFO] 后处理变化: {post_px - pre_px:+d} px ({(post_px/pre_px - 1)*100:+.1f}%)")
        optimized_skeleton = stage_optimize_skeleton(
            image_rgb, mask_for_skeleton, output_dir, config
        )
    else:
        print("\n[INFO] 未开启 --optimize-skeleton，跳过 skeleton 优化。")

    # ===================================================================
    # Stage 9: Draft graph 提取（V3.1，需要 --extract-draft-graph）
    # ===================================================================
    draft_nodes = None
    draft_edges = None
    if args.extract_draft_graph:
        print("\n" + "=" * 60)
        print("  STAGE 9: Draft graph 提取（V3.1）")
        print("=" * 60)
        # 优先使用优化骨架，否则尝试加载已有的
        if optimized_skeleton is not None:
            skel_for_graph = optimized_skeleton
        else:
            skel_path = os.path.join(output_dir, "road_skeleton_optimized.png")
            if os.path.exists(skel_path):
                skel_for_graph = cv2.imread(skel_path, cv2.IMREAD_GRAYSCALE)
                print(f"[INFO] 加载已有优化骨架: {skel_path}")
            else:
                skel_path = os.path.join(output_dir, "road_skeleton_pruned.png")
                if os.path.exists(skel_path):
                    skel_for_graph = cv2.imread(skel_path, cv2.IMREAD_GRAYSCALE)
                    print(f"[WARN] 未找到优化骨架，使用修剪骨架: {skel_path}")
                else:
                    print("[ERROR] 未找到任何骨架文件！请先运行 --optimize-skeleton 或 --skeleton-preview")
                    skel_for_graph = None

        if skel_for_graph is not None:
            draft_nodes, draft_edges = stage_draft_graph_extract(
                image_rgb, skel_for_graph, output_dir, config
            )
    else:
        print("\n[INFO] 未开启 --extract-draft-graph，跳过 draft graph 提取。")

    # ===================================================================
    # Stage 10: 人工 graph 编辑（V3.2，需要 --edit-graph 或 --manual-graph）
    # ===================================================================
    edit_graph_requested = args.edit_graph or args.manual_graph
    if edit_graph_requested:
        print("\n" + "=" * 60)
        print("  STAGE 10: 人工 graph 编辑（V3.2）")
        print("=" * 60)
        stage_graph_editor(image_rgb, output_dir)
    else:
        print("\n[INFO] 未开启 --edit-graph / --manual-graph，跳过 graph 编辑。")

    # ===================================================================
    # 总结
    # ===================================================================
    print("\n" + "=" * 60)
    print("  V2.4 + V3.1 + V3.2 完成！输出文件列表:")
    print("=" * 60)
    for root, dirs, files in sorted(os.walk(output_dir)):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            fsize = os.path.getsize(fpath)
            size_str = f"{fsize / 1024:.1f} KB" if fsize > 1024 else f"{fsize} B"
            print(f"  {fpath}  ({size_str})")

    print("\n[INFO] V2.4 + V3.1 + V3.2 全部流程结束。")


if __name__ == "__main__":
    main()
