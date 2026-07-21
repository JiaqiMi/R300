# 路网生成工具 (RoadNet Tool)

从卫星图/航拍图中提取道路区域，为无人车自主导航比赛提供路网数据。

当前版本：**V2.4** — SAM-Road 深度学习 + 骨架化 + 路网图生成 + 线形优化 + 路径规划

## 项目结构

```
roadnet_tool/
├── main.py                  # 主入口，命令行 + V2.2 完整流水线
├── config/
│   └── default.yaml         # 默认参数配置（含 V2.2 新增配置）
├── roadnet/
│   ├── __init__.py          # 包初始化
│   ├── io_utils.py          # 图像读写工具 (RGB 统一)
│   ├── color_segment.py     # 核心: 正负样本采样 + HSV/Lab/距离约束分割
│   ├── roi.py               # V2.1: ROI 多边形绘制与 mask 生成
│   ├── ignore.py            # V2.2: ignore 矩形/多边形区域屏蔽
│   ├── mask_editor.py       # V2.1: 人工画笔补线/橡皮擦删除
│   ├── postprocess.py       # V2.2: 后处理增强流水线（参数调优）
│   ├── visualization.py     # 叠加显示 + 样本标注 + 对比图
│   └── utils.py             # 配置加载、目录创建
├── data/
│   └── sample_images/       # 放置测试图片
├── outputs/                 # 输出目录
├── requirements.txt
└── README.md
```

## 各模块职责

| 文件 | 职责 |
|------|------|
| `main.py` | 命令行入口，编排完整流程（采样→分割→ROI→ignore→编辑→后处理） |
| `config/default.yaml` | 全流程参数配置 |
| `roadnet/color_segment.py` | `ColorSampler` 正负样本交互采样 + `segment_*` 分割函数 |
| `roadnet/roi.py` | `RoiDrawer` 多边形 ROI 绘制 + mask 生成 |
| `roadnet/ignore.py` | `IgnoreDrawer` 矩形/多边形 ignore 区域屏蔽 |
| `roadnet/mask_editor.py` | `MaskEditor` 画笔补线 / 橡皮擦删除 + 撤销 |
| `roadnet/postprocess.py` | `clean_pipeline()` 开/闭运算/孔洞填充/连通域筛选/平滑/细长噪声删除 |
| `roadnet/visualization.py` | `overlay_mask()`, `draw_sample_markers()`, `save_postprocess_compare()` |
| `roadnet/io_utils.py` | `read_image_rgb()` / `save_image()` |
| `roadnet/utils.py` | `load_config()` / `ensure_dir()` |

## 环境安装

**要求**: Python >= 3.8

```bash
pip install -r requirements.txt
```

依赖项：
- `opencv-python-headless` — 图像处理
- `numpy` — 数值计算
- `matplotlib` — 交互窗口（采样/ROI/ignore/编辑）
- `PyYAML` — 配置文件解析

## 运行方式

### V2.2 完整流水线（推荐）

```bash
cd roadnet_tool
python main.py -i data/sample_images/002.png -o outputs/test_run --draw-roi --draw-ignore --postprocess --save-intermediate
```

### V2.2 精简模式（仅分割 + 后处理）

```bash
python main.py -i data/sample_images/002.png -o outputs --postprocess
```

### V2.2 ROI + Ignore + 后处理（跳过手动编辑）

```bash
python main.py -i data/sample_images/002.png -o outputs --draw-roi --draw-ignore --postprocess
```

## 命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--image` | `-i` | 输入图像路径（必填） | — |
| `--output` | `-o` | 输出目录 | `./outputs` |
| `--config` | `-c` | 配置文件路径 | `config/default.yaml` |
| `--draw-roi` | — | 开启 ROI 多边形绘制 | 关闭 |
| `--draw-ignore` | — | 开启 ignore 区域屏蔽（矩形/多边形） | 关闭 |
| `--edit-mask` | — | 开启人工编辑 mask | 关闭 |
| `--postprocess` | — | 开启后处理增强 | 关闭 |
| `--save-intermediate` | — | 保存后处理各步骤中间结果 | 关闭 |

## V2.2 完整工作流程

```
Stage 1: V2.2 采样 + 分割（负样本增强版）
    └→ road_mask_raw.png, road_overlay_raw.png, sample_points.png

Stage 2: ROI 区域限制 (--draw-roi)
    └→ roi_mask.png, road_mask_roi.png, road_overlay_roi.png

Stage 3: Ignore 区域屏蔽 (--draw-ignore)  ← V2.2 新增
    └→ ignore_mask.png, road_mask_ignore.png, road_overlay_ignore.png

Stage 4: 人工编辑 mask (--edit-mask)
    └→ road_mask_edited.png, road_overlay_edited.png

Stage 5: 后处理增强 (--postprocess)
    └→ road_mask_clean.png, road_overlay_clean.png
        postprocess_compare.png
        [若 --save-intermediate] postprocess_01_open.png ~ 07_remove_thin.png
```

## V2.2 各阶段操作说明

### Stage 1: V2.2 负样本增强采样分割

| 操作 | 效果 |
|------|------|
| **左键点击** | 添加道路正样本（绿色圆圈） |
| **右键点击** | 添加非道路负样本（红色叉号） |
| **c 键** | 清空所有样本点 |
| **u 键** | 撤销上一个样本点 |
| **Enter** | 确认并开始分割 |
| **Esc / 关闭窗口** | 取消退出 |

**V2.2 负样本增强建议**：对草坪、跑道、建筑屋顶、树林、文字、田地等非道路区域各采集 2~3 个负样本点（共 10~20 个），可大幅减少误检。

### Stage 2: ROI 区域绘制 (--draw-roi)

| 操作 | 效果 |
|------|------|
| **左键点击** | 添加 ROI 多边形顶点 |
| **Enter** | 闭合并确认 ROI |
| **c 键** | 清空当前多边形 |
| **Esc / 关闭窗口** | 跳过 ROI 限制 |

### Stage 3: Ignore 区域屏蔽 (--draw-ignore) ← V2.2 新增

| 操作 | 效果 |
|------|------|
| **左键拖拽** | 绘制矩形 ignore 区域（红色） |
| **p 键** | 切换到多边形模式 |
| **r 键** | 切回矩形模式 |
| **多边形模式下左键点击** | 添加多边形顶点 |
| **多边形模式下 Enter** | 闭合当前多边形 |
| **c 键** | 清空所有屏蔽区域 |
| **u 键** | 撤销上一个屏蔽区域 |
| **Enter** | 确认所有屏蔽区域 |
| **Esc / 关闭窗口** | 取消屏蔽 |

**用途**：屏蔽底部文字、建筑屋顶、右侧树林/田地、左侧空地等明显误检区域。

### Stage 4: 人工编辑 mask (--edit-mask)

| 操作 | 效果 |
|------|------|
| **左键拖动** | 画笔添加道路区域（涂白） |
| **右键拖动** | 橡皮擦删除误检（涂黑） |
| **+ / = 键** | 增大画笔半径 |
| **- 键** | 减小画笔半径 |
| **s 键** | 保存当前编辑结果 |
| **u 键** | 撤销上一步 |
| **Esc / 关闭窗口** | 退出编辑 |

### Stage 5: 后处理增强 (--postprocess)

自动执行以下步骤：
1. **开运算** (kernel=5) — 去除小噪声和细小毛刺
2. **闭运算** (kernel=9) — 连接小断裂
3. **孔洞填充** — 填充道路区域内部空洞
4. **连通域筛选** (min_area=1000) — 删除小面积碎片
5. **保留最大 N 个连通域** (N=6) — 只保留主道路区域
6. **边界平滑** (kernel=5) — 高斯模糊+二值化平滑边缘
7. **删除细长噪声** — 移除宽高比异常的细长误检

## 分割算法原理 (V1.5/V2.2)

### 三层 AND 约束

```
最终 mask = HSV阈值mask AND Lab阈值mask AND 正负距离约束mask
```

1. **HSV 阈值**：基于正样本在 HSV 空间的均值±(margin+std) 做 inRange 过滤
2. **Lab 阈值**：同上但在 Lab 色彩空间
3. **正负距离约束（核心，V2.2 增强）**：
   - 每个像素计算到正样本集和负样本集的 Lab 颜色距离
   - `d_pos > positive_distance_threshold(20)` → 排除
   - `d_neg <= negative_margin(6)` → 排除（V2.2 收紧）
   - `d_pos >= d_neg` → 排除

## 配置说明 (`config/default.yaml`)

```yaml
# V2.2 分割参数（负样本增强，参数调优）
segment:
  mode: "combined"                    # hsv / lab / combined
  combine_method: "and"               # and(交集) / or(并集)
  sample_radius: 3
  h_margin: 6
  s_margin: 25
  v_margin: 30
  lab_margin: 12
  use_negative_samples: true
  positive_distance_threshold: 20     # V2.2: 22→20 更严格
  negative_margin: 6                  # V2.2: 5→6 更严格
  overlay_alpha: 0.45

# V2.2 ROI
roi:
  enable: true
  save_roi: true

# V2.2 ignore 区域屏蔽（新增）
ignore:
  enable: true
  save_ignore: true

# V2.2 编辑
edit:
  enable: true
  brush_radius: 8
  max_undo_steps: 20

# V2.2 后处理（参数调优）
postprocess:
  enable: true
  open_kernel_size: 5              # V2.2: 3→5 更彻底去毛刺
  close_kernel_size: 9
  fill_holes: true
  min_area: 1000                   # V2.2: 800→1000 删除更多小碎片
  keep_largest_components: 6       # V2.2: 5→6 保留更多主路
  smooth_kernel_size: 5
  remove_thin_noise: true

# V2.2 可视化
visualization:
  overlay_alpha: 0.45
  save_intermediate: true
```

## 调参指南

### V2.2 分割阶段

| 现象 | 调整方向 |
|------|----------|
| mask 覆盖过大（过分割） | 减小 h/s/v/lab_margin，减小 positive_distance_threshold，增加负样本点 |
| 道路漏检严重（欠分割） | 增大 margin，增大 positive_distance_threshold |
| 草/跑道/建筑/树林/文字误检 | 添加更多该类负样本点（每类 2~3 个），增大 negative_margin |
| 道路断裂不连续 | 增大 positive_distance_threshold，多采正样本 |

### V2.2 后处理阶段

| 现象 | 调整方向 |
|------|----------|
| 道路边缘毛刺多 | 增大 open_kernel_size（如 7） |
| 道路断裂未连接 | 增大 close_kernel_size（如 15） |
| 小碎片噪声残留 | 增大 min_area（如 2000） |
| 保留道路区域太少 | 增大 keep_largest_components（如 8） |
| 道路边缘过于光滑 | 减小 smooth_kernel_size（如 3） |
| 长条噪声未移除 | 确认 remove_thin_noise: true |

## V2.2 输出文件

### 完整流水线输出

```
outputs/
├── sample_points.png           # Stage 1: 采样点标注
├── road_mask_raw.png            # Stage 1: 原始分割 mask
├── road_overlay_raw.png         # Stage 1: 原始叠加图
├── roi_mask.png                 # Stage 2: 二值 ROI mask
├── road_mask_roi.png            # Stage 2: ROI 限制后 mask
├── road_overlay_roi.png         # Stage 2: ROI 限制后叠加图
├── ignore_mask.png              # Stage 3: 二值 ignore mask（V2.2 新增）
├── road_mask_ignore.png         # Stage 3: ignore 屏蔽后 mask（V2.2 新增）
├── road_overlay_ignore.png      # Stage 3: ignore 屏蔽后叠加图（V2.2 新增）
├── road_mask_edited.png         # Stage 4: 编辑后 mask
├── road_overlay_edited.png      # Stage 4: 编辑后叠加图
├── road_mask_clean.png          # Stage 5: 最终清理 mask
├── road_overlay_clean.png       # Stage 5: 最终清理叠加图
├── postprocess_compare.png      # Stage 5: raw vs clean 对比
└── (若 --save-intermediate)
    ├── postprocess_01_open.png
    ├── postprocess_02_close.png
    ├── postprocess_03_fill_holes.png
    ├── postprocess_04_remove_small.png
    ├── postprocess_05_keep_largest.png
    ├── postprocess_06_smooth.png
    └── postprocess_07_remove_thin.png
```

## V2.2 设计原则

1. **宁可道路少分，也不把非道路大区域分进来** — 道路断裂可以人工补线/闭运算连接，误检多会导致骨架化和路网生成完全混乱
2. **多模式屏蔽** — ROI（粗筛）+ Ignore（精细屏蔽）+ 编辑（像素级修正），三层递进
3. 各步骤可独立开关，灵活组合
4. 每步检查文件存在性，避免程序崩溃
5. 所有输出路径由 `--output` 统一控制

## 验收标准 (V2.2)

1. ✅ 右侧树林/田地的细线误检通过 ignore 屏蔽明显减少
2. ✅ 左侧建筑和底部文字区域通过 ignore 屏蔽基本被删除
3. ✅ 体育场外圈主道路仍然保留
4. ✅ 上方、右侧、底部主路基本连续
5. ✅ `road_mask_clean.png` 可作为后续 skeleton 和 graph_extract 的输入

## GUI 推荐工作流（RoadNet Studio）

### Startup

```bash
cd roadnet_tool
python main_gui.py
```

### Recommended Pipeline (using SAM-Road)

```
Step 1: 文件 → 打开影像...
        
Step 2: 工具 → 运行 SAM-Road 单图初提取...
        ├→ 配置 Python 解释器、推理包目录、权重
        ├→ 点击「检查 config/checkpoint 匹配」
        ├→ 运行完成后自动导入:
        │   ├─ road_mask.png  → Mask 图层（绿色，用于后续所有处理）
        │   ├─ itsc_mask.png  → ITSC 参考叠加层（蓝紫色半透明）
        │   ├─ viz.png        → 可视化参考叠加层
        │   └─ graph.p        → 参考 Graph（金色，不作为 final_graph）
        └─ 状态栏提示: Graph.p 已作为参考 graph 导入，不作为 final_graph

Step 3: 工具 → SAM-Road mask 后处理...
        └→ 调整阈值、形态学参数、面积过滤，优化 road mask

Step 4: 工具 → 生成/优化道路骨架...
        └→ 从 road mask 生成 skeleton → 优化（去毛刺、连接断裂）

Step 5: 工具 → 从 skeleton 生成 graph
        └→ 从优化后 skeleton 提取节点和边 → final_graph

Step 5.5: 工具 → 优化 graph 线形...  ← ★ 新增
        ├→ RDP 折线简化：去除冗余中间点
        ├→ 近似直线拉直：直路只保留首尾点
        ├→ 弯路轻微平滑：moving average + 偏移约束
        ├→ processed_mask 校验：偏离道路区域则回退
        ├→ 保存 graph_line_optimize_outputs/ 目录
        └→ 自动更新 final_graph 渲染

Step 6: ⑤ 路网编辑 阶段 → 手动修图、移动节点、增删边

Step 7: ⑥ 坐标校准 → 校正地理坐标

Step 8: 📍 任务点 → 导入任务点 → 自动吸附 → 全局路径规划 → 导出
```

### 重要说明

| 项目 | 说明 |
|------|------|
| **graph.p** | SAM-Road 模型直接输出的原始拓扑图，**仅作为参考图层**（金色半透明），不参与路径规划 |
| **final_graph** | 从 skeleton 重新提取并手动编辑后的**正式路网图**（蓝线），是路径规划的输入 |
| **旧版入口** | `工具 → 高级/旧版功能 → 旧版 SAM-Road 初提取` 保留作为历史兼容，不推荐使用 |

### 菜单入口说明

| 菜单项 | 状态 | 用途 |
|--------|------|------|
| `运行 SAM-Road 单图初提取...` | ✅ 推荐 | 调用 infer_single.py 推理 + 自动导入结果 |
| `导入 SAM-Road 单图结果...` | ✅ 推荐 | 仅导入已有结果目录，不运行推理 |
| `优化 graph 线形...` | ✅ 推荐 | RDP 简化 + 直线拉直 + 弯路平滑 + mask 校验 |
| `高级/旧版功能 → 旧版 SAM-Road 初提取...` | ⚠️ 已废弃 | 旧版批量推理入口（不再维护） |
| `高级/旧版功能 → 旧版 导入 SAM-Road 结果...` | ⚠️ 已废弃 | 旧版 draft_graph.json 格式导入 |

### 线形优化默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| rdp_epsilon | 3.0 px | RDP 折线简化容差，越大简化越激进 |
| straight_max_deviation | 6.0 px | 判定为近似直线的最大偏移距离 |
| min_straight_edge_length | 30.0 px | 参与拉直判定的最小 edge 长度 |
| smooth_window | 5 | 弯路平滑的 moving average 窗口大小 |
| max_smooth_offset | 4.0 px | 平滑后每个点相对原始位置的最大偏移 |
| mask_tolerance | 5.0 px | mask 校验时允许偏离道路的距离容差 |

### 图层一览

| 图层 | 颜色 | 来源 | 用途 |
|------|------|------|------|
| Road Mask | 绿色半透明 | SAM-Road 推理或手动分割 | 道路区域，后续骨架化输入 |
| ITSC Mask | 蓝紫色半透明 | SAM-Road 推理 | 交叉口检测参考 |
| Viz | 彩色半透明 | SAM-Road 推理 | 可视化参考 |
| Skeleton | 黄色 | 骨架化/优化 | 道路骨架线 |
| Draft Graph | 橙色 | 骨架→Graph 提取 | 草案路网 |
| Final Graph | 蓝色 | 骨架→Graph 提取 + 手动编辑 | 正式路网，路径规划输入 |
| Reference Graph (graph.p) | 金色 | SAM-Road 推理 | 参考层，不作为 final_graph |
| 规划路径 | 紫色 | 全局规划 | 最终行驶路径 |

## 后续扩展方向

- V3: 道路骨架提取（形态学骨架化 / Zhang-Suen 算法）✅ 已实现
- V4: 节点与边生成（从骨架构建图结构）✅ 已实现
- V5: 路网导出（GeoJSON / GraphML 等格式供路径规划使用）✅ 已实现
- 任务点吸附与全局路径规划 ✅ 已实现
- SAM-Road 深度学习模型集成 ✅ 已实现
- Graph 线形优化（RDP 简化 + 直线拉直 + 弯路平滑 + mask 校验）✅ 已实现
