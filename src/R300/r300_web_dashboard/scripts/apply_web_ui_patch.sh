#!/usr/bin/env bash
set -euo pipefail

PKG_DIR="${PKG_DIR:-/home/explorer/r300_ws/src/R300/r300_web_dashboard}"
WWW_DIR="$PKG_DIR/www"
TS="$(date +%Y%m%d_%H%M%S)"

echo "[R300 Web UI v11] package: $PKG_DIR"

if [ ! -d "$WWW_DIR" ]; then
  echo "[ERROR] 未找到 $WWW_DIR"
  echo "请确认 r300_web_dashboard 路径正确，或这样指定："
  echo "  PKG_DIR=/你的路径/r300_web_dashboard $0"
  exit 1
fi

for f in style.css app.js config.json index.html; do
  if [ -f "$WWW_DIR/$f" ]; then
    cp "$WWW_DIR/$f" "$WWW_DIR/${f}.bak_v11_${TS}"
    echo "[backup] $WWW_DIR/${f}.bak_v11_${TS}"
  fi
done

python3 - <<'PY'
from pathlib import Path
import json
import re

pkg = Path('/home/explorer/r300_ws/src/R300/r300_web_dashboard')
www = pkg / 'www'

# 1) 视频参数恢复为清晰版：不再使用 v10 的低分辨率/低 quality
cfg_path = www / 'config.json'
if cfg_path.exists():
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    cfg.setdefault('video', {})
    cfg['video']['host'] = cfg['video'].get('host', 'auto')
    cfg['video']['quality'] = 70
    cfg['video']['width'] = 640
    cfg['video']['height'] = 480
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('[config] video restored: quality=70, width=640, height=480')
else:
    print('[warn] config.json 不存在，跳过视频参数修改')

# 2) index：雷达卡片保留统计标题，不恢复大雷达图
idx_path = www / 'index.html'
if idx_path.exists():
    s = idx_path.read_text(encoding='utf-8')
    s = s.replace('<h2>激光 / 视觉障碍俯视图</h2>', '<h2>雷达 / 视觉障碍统计</h2>')
    s = s.replace('<span id="scanInfo">等待 /scan</span>', '<span id="scanInfo">等待雷达/视觉障碍数据</span>')
    idx_path.write_text(s, encoding='utf-8')
    print('[index] scan title kept as statistics')

# 3) app.js：继续使用轻量雷达统计，避免大 Canvas 拖慢页面；日志保持较多行
app_path = www / 'app.js'
if app_path.exists():
    s = app_path.read_text(encoding='utf-8')

    def replace_function(src, name, new_body):
        key = f'function {name}('
        start = src.find(key)
        if start < 0:
            return src, False
        brace = src.find('{', start)
        if brace < 0:
            return src, False
        depth = 0
        end = None
        for i in range(brace, len(src)):
            ch = src[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            return src, False
        return src[:start] + new_body + src[end:], True

    new_draw_scan = r'''function drawScan() {
  // v11：保留雷达/视觉障碍统计，不绘制雷达大图，减轻浏览器压力。
  const rawN = scanData && scanData.ranges ? scanData.ranges.length : 0;
  const rawFinite = countFiniteScan(scanData);
  const visionFinite = countFiniteScan(visionScanData);
  const activeFinite = countFiniteScan(activeVisionScanData);

  let nearest = Infinity;
  [scanData, visionScanData, activeVisionScanData].forEach(scan => {
    if (!scan || !scan.ranges) return;
    for (const r of scan.ranges) {
      if (Number.isFinite(r) && r >= scan.range_min && r <= scan.range_max && r < nearest) nearest = r;
    }
  });
  const nearestText = Number.isFinite(nearest) ? `${fmt(nearest, 2)} m` : "--";

  if ($("scanInfo")) {
    $("scanInfo").textContent = rawN
      ? `/scan beams=${rawN}, finite=${rawFinite}, vision=${visionFinite}, active=${activeFinite}, nearest=${nearestText}`
      : `等待 /scan，vision=${visionFinite}, active=${activeFinite}, nearest=${nearestText}`;
  }
}'''
    s, ok = replace_function(s, 'drawScan', new_draw_scan)
    print('[app] drawScan statistics mode:', 'ok' if ok else 'not found')

    s = re.sub(r'\(cam\.logs \|\| \[\]\)\.slice\(-\d+\)', '(cam.logs || []).slice(-30)', s)
    s = re.sub(r'\(nav\.logs \|\| \[\]\)\.slice\(-\d+\)', '(nav.logs || []).slice(-60)', s)
    app_path.write_text(s, encoding='utf-8')

# 4) CSS：第一行只放视频和点云/costmap；卫星轨迹放第二行。雷达大图继续隐藏，只保留统计。
css_path = www / 'style.css'
patch = r'''

/* =========================================================
 * R300 Web Dashboard v11-video-map-firstrow
 * 目标：第一行只保留 摄像头 + local_costmap/DWA，视频恢复清晰参数；
 *      卫星轨迹放第二行；雷达大图隐藏，仅保留统计信息。
 * ========================================================= */

.dashboard {
  grid-template-columns: minmax(620px, 1fr) minmax(620px, 1fr) !important;
  grid-auto-flow: row dense !important;
  gap: 14px !important;
  align-items: stretch !important;
}

/* 第一行：只放两个最重要的大图 */
.video-card {
  grid-column: 1 !important;
  grid-row: 1 !important;
  min-height: 660px !important;
}
.map-card {
  grid-column: 2 !important;
  grid-row: 1 !important;
  min-height: 660px !important;
}
.video-wrap,
#costmapCanvas {
  height: 560px !important;
  min-height: 560px !important;
}

/* 第二行：卫星轨迹占左侧，状态和雷达统计等放右侧 */
.satellite-card {
  grid-column: 1 !important;
  grid-row: 2 / span 2 !important;
  min-height: 650px !important;
}
.status-card {
  grid-column: 2 !important;
  grid-row: 2 !important;
  min-height: 360px !important;
}
.scan-card {
  grid-column: 2 !important;
  grid-row: 3 !important;
  min-height: 170px !important;
}
.control-card {
  grid-column: 1 !important;
  grid-row: 4 !important;
  min-height: 360px !important;
}
.detection-card {
  grid-column: 2 !important;
  grid-row: 4 !important;
  min-height: 360px !important;
}

/* 卫星地图改为第二行的大卡片，不再挤在第一行 */
.satellite-layout {
  grid-template-columns: minmax(640px, 1fr) 320px !important;
  gap: 14px !important;
  align-items: stretch !important;
}
#satelliteMap {
  height: 560px !important;
  min-height: 560px !important;
}
.satellite-side {
  display: block !important;
  padding: 12px !important;
}
.satellite-side .mini-title,
.satellite-side .map-buttons,
.satellite-side .map-note {
  grid-column: auto !important;
}

/* 雷达大图隐藏，只保留统计文本，减轻页面渲染 */
#scanCanvas,
.scan-card .toolbar {
  display: none !important;
}
.scan-card .card-title {
  align-items: flex-start !important;
  flex-direction: column !important;
  gap: 8px !important;
}
#scanInfo {
  display: block !important;
  width: 100% !important;
  white-space: normal !important;
  font-family: Consolas, "JetBrains Mono", monospace !important;
  font-size: 14px !important;
  line-height: 1.6 !important;
  color: #d7f549 !important;
  background: rgba(6, 23, 7, .82) !important;
  border: 1px solid rgba(151,219,72,.22) !important;
  border-radius: 10px !important;
  padding: 10px !important;
}

/* 日志显示更多行 */
#serviceLog { max-height: 260px !important; min-height: 120px !important; }
#nodeLog { max-height: 360px !important; min-height: 220px !important; }
#detections, #targetPoint { max-height: 260px !important; min-height: 110px !important; }

@media (max-width: 1500px) {
  .dashboard { grid-template-columns: 1fr !important; }
  .video-card,
  .map-card,
  .satellite-card,
  .scan-card,
  .status-card,
  .control-card,
  .detection-card {
    grid-column: 1 !important;
    grid-row: auto !important;
  }
  .satellite-layout { grid-template-columns: 1fr !important; }
  #satelliteMap { height: 480px !important; min-height: 480px !important; }
}
'''
if css_path.exists():
    s = css_path.read_text(encoding='utf-8')
    # 直接追加 v11 覆盖 v8/v9/v10 样式，后写优先级更高
    s += patch
    css_path.write_text(s, encoding='utf-8')
    print('[css] v11 layout patch appended')

print('\n完成：v11 已应用。请重启 Web，并在浏览器 Ctrl+F5 强制刷新。')
PY

echo "[OK] v11 patch done."
