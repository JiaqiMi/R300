/* R300 Web 上位机 v3。
 * 设计原则：浏览器直接通过 rosbridge JSON 协议订阅 ROS1 话题；
 * 视频由已有 web_video_server 提供；不直接发布 /cmd_vel。
 */

let cfg = null;
let ws = null;
let reconnectTimer = null;
let lastCostmap = null;
let costmapCanvasCache = null;
let costmapCacheKey = "";
let globalPlan = null;
let localPlan = null;
let scanData = null;
let visionScanData = null;
let activeVisionScanData = null;
let robotPose = null;  // {x, y, yaw, stampMs}
let headingDeg = null;
let safetyState = {limit: null, estop: null};
let msgCounter = 0;
let lastBadgeUpdateMs = 0;
let lastRx = {};

const viewState = {
  costmapCanvas: {scale: 1, tx: 0, ty: 0, dragging: false, lastX: 0, lastY: 0},
  scanCanvas: {scale: 1, tx: 0, ty: 0, dragging: false, lastX: 0, lastY: 0}
};

const $ = (id) => document.getElementById(id);
const fmt = (v, n=2) => (Number.isFinite(v) ? Number(v).toFixed(n) : "--");
const ageSec = (k) => lastRx[k] ? (Date.now() - lastRx[k]) / 1000.0 : 999.0;

function nowTime() { return new Date().toLocaleTimeString(); }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function quaternionToYaw(q) {
  if (!q) return 0;
  const x = q.x || 0, y = q.y || 0, z = q.z || 0, w = q.w || 1;
  const siny = 2.0 * (w * z + x * y);
  const cosy = 1.0 - 2.0 * (y * y + z * z);
  return Math.atan2(siny, cosy);
}

function transformBaseToWorld(bx, by) {
  if (!robotPose) return null;
  const c = Math.cos(robotPose.yaw);
  const s = Math.sin(robotPose.yaw);
  return {
    x: robotPose.x + c * bx - s * by,
    y: robotPose.y + s * bx + c * by
  };
}

async function loadConfig() {
  const res = await fetch("config.json?ts=" + Date.now(), {cache: "no-store"});
  cfg = await res.json();
  const host = location.hostname || "127.0.0.1";
  if (cfg.rosbridge.host === "auto") cfg.rosbridge.host = host;
  if (cfg.video.host === "auto") cfg.video.host = host;
  setVideoUrl();
}

function setVideoUrl() {
  const v = cfg.video;
  const url = `http://${v.host}:${v.port}/stream?topic=${v.topic}&type=mjpeg&quality=${v.quality}&width=${v.width}&height=${v.height}`;
  $("video").src = url;
  $("videoTopic").textContent = v.topic;
}

function connectRosbridge() {
  const url = `ws://${cfg.rosbridge.host}:${cfg.rosbridge.port}`;
  ws = new WebSocket(url);
  ws.onopen = () => {
    $("rosStatus").textContent = "ROSBridge 已连接";
    $("rosStatus").className = "badge good";
    logLast("已连接 " + url);
    subscribeAll();
  };
  ws.onclose = () => {
    $("rosStatus").textContent = "ROSBridge 断开，重连中";
    $("rosStatus").className = "badge bad";
    if (!reconnectTimer) {
      reconnectTimer = setTimeout(() => { reconnectTimer = null; connectRosbridge(); }, 1500);
    }
  };
  ws.onerror = () => { $("rosStatus").textContent = "ROSBridge 错误"; $("rosStatus").className = "badge bad"; };
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.op === "publish") handleTopic(msg.topic, msg.msg);
      if (msg.op === "service_response") handleServiceResponse(msg);
    } catch (e) { console.warn("Bad websocket message", e); }
  };
}

function send(obj) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return false;
  ws.send(JSON.stringify(obj));
  return true;
}
function sub(topic, type, throttle=200) {
  if (!topic) return;
  send({op: "subscribe", topic: topic, type: type, throttle_rate: throttle});
}

function subscribeAll() {
  const t = cfg.topics;
  sub(t.odom, "nav_msgs/Odometry", 80);
  sub(t.fix, "sensor_msgs/NavSatFix", 1000);
  sub(t.gps_fix, "sensor_msgs/NavSatFix", 1000);
  sub(t.heading_deg, "std_msgs/Float64", 200);
  sub(t.cmd_vel, "geometry_msgs/Twist", 100);
  sub(t.global_plan, "nav_msgs/Path", 350);
  sub(t.local_plan, "nav_msgs/Path", 150);
  sub(t.current_goal, "geometry_msgs/PoseStamped", 500);
  sub(t.costmap, "nav_msgs/OccupancyGrid", 800);
  sub(t.scan, "sensor_msgs/LaserScan", 180);
  sub(t.vision_scan, "sensor_msgs/LaserScan", 180);
  sub(t.active_vision_scan, "sensor_msgs/LaserScan", 180);
  sub(t.detections, "r300_vision_msgs/DetectedObjectArray", 500);
  sub(t.target_point, "geometry_msgs/PointStamped", 500);
  sub(t.dynamic_state, "std_msgs/String", 250);
  sub(t.speed_limit, "std_msgs/Float32", 250);
  sub(t.emergency_stop, "std_msgs/Bool", 250);
}

function handleTopic(topic, msg) {
  const t = cfg.topics;
  lastRx[topic] = Date.now();
  logLast(topic);
  if (topic === t.odom) updateOdom(msg);
  else if (topic === t.heading_deg) { headingDeg = Number(msg.data); updateHeading(); }
  else if (topic === t.fix || topic === t.gps_fix) updateGps(msg, topic === t.gps_fix ? "GPS" : "FIX");
  else if (topic === t.cmd_vel) updateCmdVel(msg);
  else if (topic === t.current_goal) updateGoal(msg);
  else if (topic === t.global_plan) { globalPlan = msg; drawCostmap(); updatePlanStats(); }
  else if (topic === t.local_plan) { localPlan = msg; drawCostmap(); updatePlanStats(); }
  else if (topic === t.costmap) { lastCostmap = msg; costmapCanvasCache = null; drawCostmap(); updatePlanStats(); }
  else if (topic === t.scan) { scanData = msg; drawScan(); drawCostmap(); updatePlanStats(); }
  else if (topic === t.vision_scan) { visionScanData = msg; drawScan(); drawCostmap(); updatePlanStats(); }
  else if (topic === t.active_vision_scan) { activeVisionScanData = msg; drawScan(); drawCostmap(); updatePlanStats(); }
  else if (topic === t.detections) updateDetections(msg);
  else if (topic === t.target_point) updateTargetPoint(msg);
  else if (topic === t.dynamic_state) $("dynState").textContent = msg.data;
  else if (topic === t.speed_limit) updateSafety("limit", msg.data);
  else if (topic === t.emergency_stop) updateSafety("estop", msg.data);
}

function logLast(text) {
  msgCounter += 1;
  const now = performance.now();
  if (text && text.startsWith("已连接")) { $("lastMsg").textContent = text; return; }
  if (now - lastBadgeUpdateMs > 1500) {
    lastBadgeUpdateMs = now;
    $("lastMsg").textContent = `${nowTime()} 数据接收中`;
  }
}

function updateOdom(m) {
  const p = m.pose.pose.position;
  const q = m.pose.pose.orientation;
  const v = m.twist.twist;
  robotPose = {x: Number(p.x), y: Number(p.y), yaw: quaternionToYaw(q), stampMs: Date.now()};
  $("poseXY").textContent = `x=${fmt(p.x)} m, y=${fmt(p.y)} m`;
  $("vel").textContent = `vx=${fmt(v.linear.x)} m/s, wz=${fmt(v.angular.z)} rad/s`;
  updateHeading();
  drawCostmap();
}

function updateHeading() {
  if (Number.isFinite(headingDeg)) $("heading").textContent = `${fmt(headingDeg, 1)}°`;
  else if (robotPose) $("heading").textContent = `${fmt(robotPose.yaw * 180 / Math.PI, 1)}° (odom yaw)`;
}

function updateGps(m, label) {
  if (!Number.isFinite(m.latitude) || !Number.isFinite(m.longitude)) return;
  $("gps").textContent = `${label}: ${fmt(m.latitude, 7)}, ${fmt(m.longitude, 7)}`;
}
function updateCmdVel(m) { $("vel").textContent = `cmd vx=${fmt(m.linear.x)} m/s, wz=${fmt(m.angular.z)} rad/s`; }
function updateGoal(m) {
  const p = m.pose.position;
  $("goal").textContent = `x=${fmt(p.x)} y=${fmt(p.y)} frame=${m.header.frame_id || "--"}`;
}
function updateSafety(k, v) {
  safetyState[k] = v;
  const limit = Number.isFinite(safetyState.limit) ? `${fmt(safetyState.limit)} m/s` : "--";
  const estop = safetyState.estop === null ? "--" : (safetyState.estop ? "急停" : "正常");
  $("safety").textContent = `${limit} / ${estop}`;
}

function updateDetections(m) {
  const arr = m.objects || m.detections || [];
  $("detectionCount").textContent = `${arr.length} 个目标`;
  if (!arr.length) { $("detections").textContent = "当前无检测目标"; return; }
  $("detections").textContent = arr.slice(0, 12).map((o, i) => {
    const cls = o.class_name || o.label || o.name || o.class_id || "object";
    const conf = o.confidence !== undefined ? fmt(o.confidence, 2) : "--";
    let pos = "";
    if (o.position) pos = ` pos=(${fmt(o.position.x)}, ${fmt(o.position.y)}, ${fmt(o.position.z)})`;
    if (o.center) pos = ` center=(${fmt(o.center.x)}, ${fmt(o.center.y)}, ${fmt(o.center.z)})`;
    return `${i}: ${cls}  conf=${conf}${pos}`;
  }).join("\n");
}
function updateTargetPoint(m) {
  const p = m.point;
  $("targetPoint").textContent = `/r300_vision/target_point\nframe=${m.header.frame_id}\nx=${fmt(p.x)} y=${fmt(p.y)} z=${fmt(p.z)}`;
}

function setupInteractiveCanvas(canvasId, redrawFn) {
  const c = $(canvasId), st = viewState[canvasId];
  if (!c || !st) return;
  c.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = c.getBoundingClientRect();
    const x = (e.clientX - rect.left) * c.width / rect.width;
    const y = (e.clientY - rect.top) * c.height / rect.height;
    const old = st.scale;
    const next = clamp(old * (e.deltaY < 0 ? 1.15 : 1 / 1.15), 0.35, 18);
    st.tx = x - (x - st.tx) * (next / old);
    st.ty = y - (y - st.ty) * (next / old);
    st.scale = next;
    redrawFn();
  }, {passive: false});
  c.addEventListener("mousedown", (e) => { st.dragging = true; st.lastX = e.clientX; st.lastY = e.clientY; });
  window.addEventListener("mousemove", (e) => {
    if (!st.dragging) return;
    const rect = c.getBoundingClientRect();
    st.tx += (e.clientX - st.lastX) * c.width / rect.width;
    st.ty += (e.clientY - st.lastY) * c.height / rect.height;
    st.lastX = e.clientX; st.lastY = e.clientY;
    redrawFn();
  });
  window.addEventListener("mouseup", () => { st.dragging = false; });
  c.addEventListener("dblclick", () => resetCanvasView(canvasId));
}
function resetCanvasView(canvasId) { const st = viewState[canvasId]; if (!st) return; st.scale = 1; st.tx = 0; st.ty = 0; drawCostmap(); drawScan(); }
function applyView(ctx, canvasId) { const st = viewState[canvasId]; ctx.translate(st.tx, st.ty); ctx.scale(st.scale, st.scale); }
function drawHint(ctx, canvasId) {
  const st = viewState[canvasId];
  ctx.save(); ctx.font = "12px Consolas"; ctx.fillStyle = "rgba(226,232,240,.88)";
  ctx.fillText(`滚轮缩放 / 拖拽平移 / 双击复位 / zoom=${st.scale.toFixed(2)}x`, 14, 20); ctx.restore();
}

function mapWorldToPixel(x, y, map, canvas) {
  const info = map.info;
  const ox = info.origin.position.x;
  const oy = info.origin.position.y;
  const sx = (x - ox) / (info.resolution * info.width) * canvas.width;
  const sy = canvas.height - (y - oy) / (info.resolution * info.height) * canvas.height;
  return [sx, sy];
}

function buildCostmapImage(map, canvas) {
  const key = `${map.header.seq || 0}:${map.info.width}:${map.info.height}:${map.info.resolution}:${map.data.length}:${Date.now()}`;
  const off = document.createElement("canvas");
  off.width = canvas.width; off.height = canvas.height;
  const octx = off.getContext("2d");
  const img = octx.createImageData(canvas.width, canvas.height);
  const info = map.info;
  for (let py = 0; py < canvas.height; py++) {
    for (let px = 0; px < canvas.width; px++) {
      const mx = Math.floor(px / canvas.width * info.width);
      const my = Math.floor((canvas.height - 1 - py) / canvas.height * info.height);
      const val = map.data[my * info.width + mx];
      const idx = (py * canvas.width + px) * 4;
      let r=239, g=244, b=250;
      if (val < 0) { r=185; g=193; b=204; }
      else if (val === 0) { r=245; g=247; b=250; }
      else if (val >= 90) { r=15; g=18; b=22; }
      else { const d = Math.round(245 - val * 1.8); r=d; g=d; b=d; }
      img.data[idx]=r; img.data[idx+1]=g; img.data[idx+2]=b; img.data[idx+3]=255;
    }
  }
  octx.putImageData(img, 0, 0);
  costmapCacheKey = key;
  costmapCanvasCache = off;
  return off;
}

function pathLength(path) {
  if (!path || !path.poses || path.poses.length < 2) return 0;
  let len = 0;
  for (let i=1; i<path.poses.length; i++) {
    const a = path.poses[i-1].pose.position, b = path.poses[i].pose.position;
    len += Math.hypot(b.x-a.x, b.y-a.y);
  }
  return len;
}

function drawCostmap() {
  const c = $("costmapCanvas"), ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.fillStyle = "#020617"; ctx.fillRect(0, 0, c.width, c.height);
  if (!lastCostmap) { ctx.fillStyle = "#94a3b8"; ctx.fillText("等待 /move_base/local_costmap/costmap ...", 18, 30); return; }

  const map = lastCostmap, info = map.info;
  const vst = viewState.costmapCanvas;
  $("costmapInfo").textContent = `${info.width}×${info.height}, res=${fmt(info.resolution,3)}m, zoom=${vst.scale.toFixed(2)}x`;
  const off = buildCostmapImage(map, c);

  ctx.save();
  applyView(ctx, "costmapCanvas");
  ctx.drawImage(off, 0, 0);

  if ($("showGlobal").checked) drawPathOnCostmap(ctx, globalPlan, map, c, "#22c55e", 3);
  if ($("showLocal").checked) drawPathOnCostmap(ctx, localPlan, map, c, "#38bdf8", 5);
  if ($("showCostLaser").checked) drawScanOnCostmap(ctx, scanData, map, c, "rgba(239,68,68,.72)", 1.8);
  if ($("showCostVision").checked) {
    drawScanOnCostmap(ctx, visionScanData, map, c, "rgba(249,115,22,.90)", 3.4);
    drawScanOnCostmap(ctx, activeVisionScanData, map, c, "rgba(168,85,247,.95)", 3.8);
  }
  if ($("showCostRobot").checked) drawRobotArrowOnCostmap(ctx, map, c);
  ctx.restore();
  drawHint(ctx, "costmapCanvas");
}

function drawPathOnCostmap(ctx, path, map, canvas, color, width) {
  if (!path || !path.poses || path.poses.length === 0) return;
  ctx.save(); ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width; ctx.lineJoin = "round"; ctx.lineCap = "round";
  ctx.beginPath();
  path.poses.forEach((ps, i) => {
    const p = ps.pose.position;
    const [x,y] = mapWorldToPixel(p.x, p.y, map, canvas);
    if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  ctx.stroke();
  const end = path.poses[path.poses.length - 1].pose.position;
  const [ex,ey] = mapWorldToPixel(end.x, end.y, map, canvas);
  ctx.beginPath(); ctx.arc(ex, ey, width+2, 0, Math.PI*2); ctx.fill();
  ctx.restore();
}

function scanPointBase(scan, i) {
  const r = scan.ranges[i];
  if (!Number.isFinite(r) || r < scan.range_min || r > scan.range_max || r > 25) return null;
  const a = scan.angle_min + i * scan.angle_increment;
  return {x: r * Math.cos(a), y: r * Math.sin(a)};
}

function drawScanOnCostmap(ctx, scan, map, canvas, color, radius) {
  if (!scan || !scan.ranges || !robotPose) return;
  ctx.save(); ctx.fillStyle = color;
  for (let i=0; i<scan.ranges.length; i++) {
    const p = scanPointBase(scan, i); if (!p) continue;
    const w = transformBaseToWorld(p.x, p.y); if (!w) continue;
    const [sx, sy] = mapWorldToPixel(w.x, w.y, map, canvas);
    if (sx < -50 || sy < -50 || sx > canvas.width+50 || sy > canvas.height+50) continue;
    ctx.beginPath(); ctx.arc(sx, sy, radius, 0, Math.PI*2); ctx.fill();
  }
  ctx.restore();
}

function drawRobotArrowOnCostmap(ctx, map, canvas) {
  if (!robotPose) return;
  const [x, y] = mapWorldToPixel(robotPose.x, robotPose.y, map, canvas);
  drawRobotArrow(ctx, x, y, -robotPose.yaw, 30, "#2563eb", "#dbeafe");
}

function drawRobotArrow(ctx, x, y, canvasYaw, size, fill, stroke) {
  ctx.save();
  ctx.translate(x, y); ctx.rotate(canvasYaw);
  ctx.beginPath();
  ctx.moveTo(size, 0);
  ctx.lineTo(-size * 0.62, -size * 0.52);
  ctx.lineTo(-size * 0.34, 0);
  ctx.lineTo(-size * 0.62, size * 0.52);
  ctx.closePath();
  ctx.fillStyle = fill; ctx.fill();
  ctx.lineWidth = 2; ctx.strokeStyle = stroke; ctx.stroke();
  ctx.restore();
}

function drawScan() {
  const c = $("scanCanvas"), ctx = c.getContext("2d");
  ctx.clearRect(0,0,c.width,c.height);
  ctx.fillStyle = "#020617"; ctx.fillRect(0,0,c.width,c.height);
  const origin = {x: c.width/2, y: c.height*0.77};
  const meters = 12.0;
  const scale = Math.min(c.width, c.height) * 0.42 / meters;

  ctx.save(); applyView(ctx, "scanCanvas");
  drawGrid(ctx, c, origin, scale, meters);
  if ($("showScanRaw").checked) drawLaser(ctx, scanData, origin, scale, "#ef4444", 1.5);
  if ($("showScanVision").checked) drawLaser(ctx, visionScanData, origin, scale, "#f97316", 2.8);
  if ($("showScanActive").checked) drawLaser(ctx, activeVisionScanData, origin, scale, "#a855f7", 3.0);
  drawRobotArrow(ctx, origin.x, origin.y, -Math.PI/2, 24, "#2563eb", "#dbeafe");
  ctx.restore();
  drawHint(ctx, "scanCanvas");
  const rawN = scanData && scanData.ranges ? scanData.ranges.length : 0;
  const finite = countFiniteScan(scanData);
  $("scanInfo").textContent = rawN ? `/scan beams=${rawN}, finite=${finite}, zoom=${viewState.scanCanvas.scale.toFixed(2)}x` : "等待 /scan ...";
}

function drawGrid(ctx, canvas, origin, scale, meters) {
  ctx.strokeStyle = "rgba(148,163,184,.18)"; ctx.lineWidth = 1;
  ctx.font = "12px Consolas"; ctx.fillStyle = "#94a3b8";
  for (let r=2; r<=meters; r+=2) { ctx.beginPath(); ctx.arc(origin.x, origin.y, r*scale, 0, Math.PI*2); ctx.stroke(); ctx.fillText(`${r}m`, origin.x + 5, origin.y - r*scale - 3); }
  ctx.strokeStyle = "rgba(148,163,184,.35)";
  ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(origin.x, 18); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(18, origin.y); ctx.lineTo(canvas.width-18, origin.y); ctx.stroke();
  ctx.fillStyle = "#cbd5e1"; ctx.fillText("前方 x+", origin.x + 8, 34); ctx.fillText("左 y+", 24, origin.y - 8);
}
function drawLaser(ctx, scan, origin, scale, color, radius) {
  if (!scan || !scan.ranges) return;
  ctx.fillStyle = color;
  for (let i=0; i<scan.ranges.length; i++) {
    const p = scanPointBase(scan, i); if (!p) continue;
    const sx = origin.x - p.y * scale;
    const sy = origin.y - p.x * scale;
    if (sx < -30 || sy < -30 || sx > ctx.canvas.width+30 || sy > ctx.canvas.height+30) continue;
    ctx.beginPath(); ctx.arc(sx, sy, radius, 0, Math.PI*2); ctx.fill();
  }
}
function countFiniteScan(scan) {
  if (!scan || !scan.ranges) return 0;
  let n = 0;
  for (const r of scan.ranges) if (Number.isFinite(r) && r >= scan.range_min && r <= scan.range_max && r < 25) n++;
  return n;
}

function updatePlanStats() {
  const gN = globalPlan && globalPlan.poses ? globalPlan.poses.length : 0;
  const lN = localPlan && localPlan.poses ? localPlan.poses.length : 0;
  $("globalStat").textContent = `${gN} 点, ${fmt(pathLength(globalPlan))} m`;
  $("localStat").textContent = `${lN} 点, ${fmt(pathLength(localPlan))} m`;
  if (lastCostmap) $("mapStat").textContent = `${lastCostmap.info.width}×${lastCostmap.info.height}, age=${fmt(ageSec(cfg.topics.costmap),1)}s`;
  const obs = countFiniteScan(visionScanData) + countFiniteScan(activeVisionScanData);
  $("obsStat").textContent = `视觉=${obs}, scan=${countFiniteScan(scanData)}`;
  $("dataAge").textContent = `odom ${fmt(ageSec(cfg.topics.odom),1)}s / map ${fmt(ageSec(cfg.topics.costmap),1)}s`;
}

function callService(name) {
  const service = cfg.services[name]; if (!service) return;
  const id = `svc:${name}:${Date.now()}`;
  const ok = send({op: "call_service", service: service, args: {}, id: id});
  const line = `${nowTime()} call ${service} ${ok ? "sent" : "failed: websocket not connected"}`;
  $("serviceLog").textContent = line + "\n" + $("serviceLog").textContent;
}
function handleServiceResponse(m) {
  const line = `${nowTime()} response ${m.service || ""} result=${m.result}`;
  $("serviceLog").textContent = line + "\n" + $("serviceLog").textContent;
}

async function main() {
  await loadConfig();
  setupInteractiveCanvas("costmapCanvas", drawCostmap);
  setupInteractiveCanvas("scanCanvas", drawScan);
  connectRosbridge();
  drawCostmap(); drawScan();
  setInterval(() => { updatePlanStats(); drawScan(); }, 500);
  setInterval(drawCostmap, 1000);
}

main().catch(e => { console.error(e); $("rosStatus").textContent = "配置加载失败"; $("rosStatus").className = "badge bad"; });
