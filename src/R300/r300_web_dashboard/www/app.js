/* R300 Web 上位机 v4 kiwi。
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
let satMap = null;
let satTrack = [];
let satPolyline = null;
let satMarker = null;
let satStartMarker = null;
let satLastPoint = null;
let satTotalDistance = 0;
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
  // 卫星地图默认使用 /one_x/fix；如果 fix 不发布，也接受 gps_fix 作为兜底。
  const fixTopic = (cfg.satellite_map && cfg.satellite_map.fix_topic) || cfg.topics.fix;
  const topicKey = label === "GPS" ? cfg.topics.gps_fix : cfg.topics.fix;
  if (topicKey === fixTopic || (!lastRx[fixTopic] && topicKey === cfg.topics.gps_fix)) {
    updateSatelliteMap(m);
  }
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


function initSatelliteMap() {
  if (!$("satelliteMap")) return;
  const mapCfg = cfg.satellite_map || {};
  if (typeof L === "undefined") {
    $("satStatus").textContent = "Leaflet 未加载，检查浏览器网络";
    return;
  }
  const center = mapCfg.default_center || [38.9866, 117.3418];
  satMap = L.map("satelliteMap", {zoomControl: true, attributionControl: true}).setView(center, mapCfg.default_zoom || 18);
  L.tileLayer(mapCfg.tile_url || "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    maxZoom: 21,
    attribution: mapCfg.attribution || "Tiles © Esri"
  }).addTo(satMap);
  satPolyline = L.polyline([], {color: "#a8ff34", weight: 4, opacity: 0.95}).addTo(satMap);
  $("satStatus").textContent = `等待 ${mapCfg.fix_topic || cfg.topics.fix} ...`;
}

function updateSatelliteMap(m) {
  if (!Number.isFinite(m.latitude) || !Number.isFinite(m.longitude)) return;
  const lat = Number(m.latitude), lon = Number(m.longitude);
  if (Math.abs(lat) < 1e-9 && Math.abs(lon) < 1e-9) return;
  const alt = Number.isFinite(m.altitude) ? Number(m.altitude) : NaN;
  const yawDeg = Number.isFinite(headingDeg) ? headingDeg : (robotPose ? robotPose.yaw * 180 / Math.PI : 0);
  const point = {lat, lon, alt, yawDeg, t: new Date().toISOString()};

  const minD = cfg.satellite_map && Number.isFinite(cfg.satellite_map.min_distance_m) ? cfg.satellite_map.min_distance_m : 0.2;
  let shouldAppend = true;
  if (satLastPoint) {
    const d = haversineMeters(satLastPoint.lat, satLastPoint.lon, lat, lon);
    shouldAppend = d >= minD;
    if (shouldAppend) satTotalDistance += d;
  }
  if (shouldAppend || satTrack.length === 0) {
    satTrack.push(point);
    satLastPoint = point;
  }

  $("satLat").textContent = fmt(lat, 8);
  $("satLon").textContent = fmt(lon, 8);
  $("satAlt").textContent = Number.isFinite(alt) ? `${fmt(alt, 2)} m` : "--";
  $("satCount").textContent = String(satTrack.length);
  $("satDistance").textContent = `${fmt(satTotalDistance, 2)} m`;
  $("satUpdate").textContent = nowTime();
  $("satStatus").textContent = `${fmt(lat, 7)}, ${fmt(lon, 7)} | ${satTrack.length} 点`;

  if (!satMap || typeof L === "undefined") return;
  const ll = [lat, lon];
  if (!satStartMarker && satTrack.length > 0) {
    satStartMarker = L.marker([satTrack[0].lat, satTrack[0].lon], {
      icon: L.divIcon({className: "", html: '<div class="start-marker">起</div>', iconSize: [28, 28], iconAnchor: [14, 14]})
    }).addTo(satMap);
  }
  const html = `<div class="vehicle-marker" style="transform: rotate(${yawDeg}deg)"></div>`;
  const icon = L.divIcon({className: "", html: html, iconSize: [28, 34], iconAnchor: [14, 22]});
  if (!satMarker) satMarker = L.marker(ll, {icon}).addTo(satMap);
  else { satMarker.setLatLng(ll); satMarker.setIcon(icon); }
  if (satPolyline) satPolyline.setLatLngs(satTrack.map(p => [p.lat, p.lon]));
  if (satTrack.length <= 2) satMap.setView(ll, cfg.satellite_map?.default_zoom || 18);
}

function centerSatelliteMap() {
  if (!satMap || !satLastPoint) return;
  satMap.setView([satLastPoint.lat, satLastPoint.lon], Math.max(satMap.getZoom(), 18));
}

function clearSatelliteTrack() {
  satTrack = [];
  satLastPoint = null;
  satTotalDistance = 0;
  if (satPolyline) satPolyline.setLatLngs([]);
  if (satStartMarker) { satMap.removeLayer(satStartMarker); satStartMarker = null; }
  $("satCount").textContent = "0";
  $("satDistance").textContent = "0.00 m";
  $("satStatus").textContent = "轨迹已清空，等待新定位";
}

function haversineMeters(lat1, lon1, lat2, lon2) {
  const R = 6371000.0;
  const toRad = d => d * Math.PI / 180.0;
  const dLat = toRad(lat2 - lat1), dLon = toRad(lon2 - lon1);
  const a = Math.sin(dLat/2)**2 + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon/2)**2;
  return 2 * R * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}

function downloadTrackCsv() {
  if (!satTrack.length) return;
  const rows = ["time,lat,lon,alt,heading_deg"];
  satTrack.forEach(p => rows.push(`${p.t},${p.lat},${p.lon},${Number.isFinite(p.alt)?p.alt:""},${Number.isFinite(p.yawDeg)?p.yawDeg:""}`));
  downloadText(`r300_track_${timestampName()}.csv`, rows.join("\n"));
}

function downloadTrackKml() {
  if (!satTrack.length) return;
  const coords = satTrack.map(p => `${p.lon},${p.lat},${Number.isFinite(p.alt)?p.alt:0}`).join(" ");
  const kml = `<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>R300 Track</name><Style id="track"><LineStyle><color>ff34ffa8</color><width>4</width></LineStyle></Style><Placemark><name>R300 trajectory</name><styleUrl>#track</styleUrl><LineString><tessellate>1</tessellate><coordinates>${coords}</coordinates></LineString></Placemark></Document></kml>`;
  downloadText(`r300_track_${timestampName()}.kml`, kml);
}

function timestampName() {
  return new Date().toISOString().replace(/[:.]/g, "-").replace("T", "_").slice(0, 19);
}

function downloadText(filename, text) {
  const blob = new Blob([text], {type: "text/plain;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
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
  ctx.fillStyle = "#031207"; ctx.fillRect(0, 0, c.width, c.height);
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
}

function drawGrid(ctx, canvas, origin, scale, meters) {
  ctx.strokeStyle = "rgba(151,234,34,.16)"; ctx.lineWidth = 1;
  ctx.font = "12px Consolas"; ctx.fillStyle = "#b8df65";
  for (let r=2; r<=meters; r+=2) { ctx.beginPath(); ctx.arc(origin.x, origin.y, r*scale, 0, Math.PI*2); ctx.stroke(); ctx.fillText(`${r}m`, origin.x + 5, origin.y - r*scale - 3); }
  ctx.strokeStyle = "rgba(151,234,34,.36)";
  ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(origin.x, 18); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(18, origin.y); ctx.lineTo(canvas.width-18, origin.y); ctx.stroke();
  ctx.fillStyle = "#eaffc0"; ctx.fillText("前方 x+", origin.x + 8, 34); ctx.fillText("左 y+", 24, origin.y - 8);
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
  $("serviceLog").textContent = (line + "\n" + $("serviceLog").textContent).split("\n").slice(0, 120).join("\n");
}
function handleServiceResponse(m) {
  const line = `${nowTime()} response ${m.service || ""} result=${m.result}`;
  $("serviceLog").textContent = (line + "\n" + $("serviceLog").textContent).split("\n").slice(0, 120).join("\n");
}

async function postApi(path) {
  try {
    const res = await fetch(path, {method: "POST", cache: "no-store"});
    const data = await res.json();
    renderProcessStatus(data.processes, data.message || "");
    return data;
  } catch (e) {
    appendNodeLog(`${nowTime()} API 调用失败：${e}`);
    return null;
  }
}

async function startProcess(name) {
  if (name === "camera") await postApi("/api/start_camera");
  else if (name === "nav") await postApi("/api/start_nav");
}

async function stopProcess(name) {
  if (name === "camera") await postApi("/api/stop_camera");
  else if (name === "nav") await postApi("/api/stop_nav");
}

async function refreshProcessStatus() {
  try {
    const res = await fetch("/api/process_status?ts=" + Date.now(), {cache: "no-store"});
    const data = await res.json();
    renderProcessStatus(data.processes, "");
  } catch (e) {
    // 页面刚打开时服务可能正在启动，安静失败即可。
  }
}

function renderProcessStatus(processes, message) {
  if (!processes) return;
  const cam = processes.camera || {};
  const nav = processes.nav || {};
  if ($("cameraProcState")) {
    $("cameraProcState").textContent = cam.running ? `相机节点：运行中 pid=${cam.pid}` : "相机节点：未运行";
  }
  if ($("navProcState")) {
    $("navProcState").textContent = nav.running ? `点云/导航：运行中 pid=${nav.pid}` : "点云/导航：未运行";
  }
  const lines = [];
  if (message) lines.push(`${nowTime()} ${message}`);
  lines.push("[camera]");
  (cam.logs || []).slice(-30).forEach(x => lines.push(x));
  lines.push("[nav]");
  (nav.logs || []).slice(-60).forEach(x => lines.push(x));
  if ($("nodeLog")) $("nodeLog").textContent = lines.join("\n") || "节点启动日志...";
}

function appendNodeLog(line) {
  if (!$('nodeLog')) return;
  $('nodeLog').textContent = line + "\n" + $('nodeLog').textContent;
}

async function main() {
  await loadConfig();
  initSatelliteMap();
  setupInteractiveCanvas("costmapCanvas", drawCostmap);
  setupInteractiveCanvas("scanCanvas", drawScan);
  connectRosbridge();
  drawCostmap(); drawScan();
  setInterval(() => { updatePlanStats(); drawScan(); }, 500);
  setInterval(drawCostmap, 1000);
  refreshProcessStatus();
  setInterval(refreshProcessStatus, 2000);
}

main().catch(e => { console.error(e); $("rosStatus").textContent = "配置加载失败"; $("rosStatus").className = "badge bad"; });
