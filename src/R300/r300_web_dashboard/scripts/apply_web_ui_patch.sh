#!/bin/bash
set -e
PKG_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP="$PKG_DIR/www/app.js"
CSS="$PKG_DIR/www/style.css"

if [ ! -f "$APP" ]; then
  echo "[ERROR] 找不到 app.js: $APP"
  exit 1
fi

python3 - "$APP" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text(encoding='utf-8')
old = s
# 节点日志原来只显示 camera 后 8 行、nav 后 12 行；改为更多行，便于外场排查。
s = s.replace('(cam.logs || []).slice(-8).forEach(x => lines.push(x));',
              '(cam.logs || []).slice(-30).forEach(x => lines.push(x));')
s = s.replace('(nav.logs || []).slice(-12).forEach(x => lines.push(x));',
              '(nav.logs || []).slice(-60).forEach(x => lines.push(x));')
# 服务调用日志做一个软限制，避免长时间运行后页面文本无限增长。
s = s.replace('$("serviceLog").textContent = line + "\\n" + $("serviceLog").textContent;',
              '$("serviceLog").textContent = (line + "\\n" + $("serviceLog").textContent).split("\\n").slice(0, 120).join("\\n");')
s = s.replace('$("serviceLog").textContent = line + "\\n" + $("serviceLog").textContent;',
              '$("serviceLog").textContent = (line + "\\n" + $("serviceLog").textContent).split("\\n").slice(0, 120).join("\\n");')
if s != old:
    p.write_text(s, encoding='utf-8')
    print('[OK] app.js 已更新：节点日志显示 camera 30 行、nav 60 行，服务日志保留 120 行。')
else:
    print('[WARN] app.js 没有匹配到旧写法，可能已经改过；style.css 仍然已替换。')
PY

echo "[OK] Web UI patch 完成。"
echo "[INFO] 建议重启 roslaunch r300_web_dashboard，并在浏览器 Ctrl+F5 强制刷新。"
