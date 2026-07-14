/* R300 Web 上位机前端。
 * 不依赖 roslibjs，直接使用 rosbridge websocket JSON 协议。
 * 这样在比赛现场没有外网时也能运行。
 */

let cfg = null;
let ws = null;
let lastCostmap = null;
let globalPlan = null;
let localPlan = null;
let scanData = null;
let visionScanData = null;
let activeVisionScanData = null;
let reconnectTimer = null;

const $ = (id) => document.getElementById(id);
const fmt = (v, n=2) => (Number.isFinite(v) ? v.toFixed(n) : "--");

function nowTime() {
  return new Date().toLocaleTimeString();
}

async function loadConfig() {
  const res = await fetch("config.json?ts=" + Date.now());
  cfg = await res.json();
  const host = location.hostname || "127.0.0.1";
  if (cfg.rosbridge.host === "auto") cfg.rosbridge.host = host;
  if (cfg.video.host === "auto") cfg.video.host = host;
  setVideoUrl();
}

function setVideoUrl() {
  const v = cfg.video;
  const url = `http://${v.host}:${v.port}/stream?topic=${encodeURIComponent(v.topic)}&type=mjpeg&quality=${v.quality}&width=${v.width}&height=${v.height}`;
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
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connectRosbridge();
      }, 1500);
    }
  };

  ws.onerror = () => {
    $("rosStatus").textContent = "ROSBridge 错误";
    $("rosStatus").className = "badge bad";
  };

  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.op === "publish") handleTopic(msg.topic, msg.msg);
      if (msg.op === "service_response") handleServiceResponse(msg);
    } catch (e) {
      console.warn("Bad websocket message", e);
    }
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
  sub(t.odom, "nav_msgs/Odometry", 100);
  sub(t.fix, "sensor_msgs/NavSatFix", 1000);
  sub(t.gps_fix, "sensor_msgs/NavSatFix", 1000);
  sub(t.heading_deg, "std_msgs/Float64", 100);
  sub(t.cmd_vel, "geometry_msgs/Twist", 100);
  sub(t.global_plan, "nav_msgs/Path", 500);
  sub(t.local_plan, "nav_msgs/Path", 200);
  sub(t.current_goal, "geometry_msgs/PoseStamped", 500);
  sub(t.costmap, "nav_msgs/OccupancyGrid", 1000);
  sub(t.scan, "sensor_msgs/LaserScan", 200);
  sub(t.vision_scan, "sensor_msgs/LaserScan", 200);
  sub(t.active_vision_scan, "sensor_msgs/LaserScan", 200);
  sub(t.detections, "r300_vision_msgs/DetectedObjectArray", 500);
  sub(t.target_point, "geometry_msgs/PointStamped", 500);
  sub(t.dynamic_state, "std_msgs/String", 200);
  sub(t.speed_limit, "std_msgs/Float32", 200);
  sub(t.emergency_stop, "std_msgs/Bool", 200);
}

function handleTopic(topic, msg) {
  const t = cfg.topics;
  logLast(topic);
  if (topic === t.odom) updateOdom(msg);
  else if (topic === t.heading_deg) $("heading").textContent = fmt(msg.data, 1) + "°";
  else if (topic === t.fix || topic === t.gps_fix) updateGps(msg, topic === t.gps_fix ? "GPS" : "FIX");
  else if (topic === t.cmd_vel) updateCmdVel(msg);
  else if (topic === t.current_goal) updateGoal(msg);
  else if (topic === t.global_plan) { globalPlan = msg; drawCostmap(); }
  else if (topic === t.local_plan) { localPlan = msg; drawCostmap(); }
  else if (topic === t.costmap) { lastCostmap = msg; drawCostmap(); }
  else if (topic === t.scan) { scanData = msg; drawScan(); }
  else if (topic === t.vision_scan) { visionScanData = msg; drawScan(); }
  else if (topic === t.active_vision_scan) { activeVisionScanData = msg; drawScan(); }
  else if (topic === t.detections) updateDetections(msg);
  else if (topic === t.target_point) updateTargetPoint(msg);
  else if (topic === t.dynamic_state) $("dynState").textContent = msg.data;
  else if (topic === t.speed_limit) updateSafety("limit", msg.data);
  else if (topic === t.emergency_stop) updateSafety("estop", msg.data);
}

function logLast(topic) {
  $("lastMsg").textContent = `${nowTime()}  ${topic}`;
}

function updateOdom(m) {
  const p = m.pose.pose.position;
  const v = m.twist.twist;
  $("poseXY").textContent = `x=${fmt(p.x)} m, y=${fmt(p.y)} m`;
  $("vel").textContent = `vx=${fmt(v.linear.x)} m/s, wz=${fmt(v.angular.z)} rad/s`;
}

function updateGps(m, label) {
  if (!Number.isFinite(m.latitude) || !Number.isFinite(m.longitude)) return;
  $("gps").textContent = `${label}: ${fmt(m.latitude, 7)}, ${fmt(m.longitude, 7)}`;
}

function updateCmdVel(m) {
  $("vel").textContent = `cmd vx=${fmt(m.linear.x)} m/s, wz=${fmt(m.angular.z)} rad/s`;
}

function updateGoal(m) {
  const p = m.pose.position;
  $("goal").textContent = `x=${fmt(p.x)} y=${fmt(p.y)} frame=${m.header.frame_id || "--"}`;
}

let safetyState = {limit: null, estop: null};
function updateSafety(k, v) {
  safetyState[k] = v;
  const limit = Number.isFinite(safetyState.limit) ? `${fmt(safetyState.limit)} m/s` : "--";
  const estop = safetyState.estop === null ? "--" : (safetyState.estop ? "急停" : "正常");
  $("safety").textContent = `${limit} / ${estop}`;
}

function updateDetections(m) {
  // 兼容不同 DetectedObjectArray 字段命名：objects / detections。
  const arr = m.objects || m.detections || [];
  if (!arr.length) {
    $("detections").textContent = "当前无检测目标";
    return;
  }
  const lines = arr.slice(0, 12).map((o, i) => {
    const cls = o.class_name || o.label || o.name || o.class_id || "object";
    const conf = o.confidence !== undefined ? fmt(o.confidence, 2) : "--";
    let pos = "";
    if (o.position) pos = ` pos=(${fmt(o.position.x)}, ${fmt(o.position.y)}, ${fmt(o.position.z)})`;
    if (o.center) pos = ` center=(${fmt(o.center.x)}, ${fmt(o.center.y)}, ${fmt(o.center.z)})`;
    return `${i}: ${cls} conf=${conf}${pos}`;
  });
  $("detections").textContent = lines.join("\n");
}

function updateTargetPoint(m) {
  const p = m.point;
  $("targetPoint").textContent = `/r300_vision/target_point\nframe=${m.header.frame_id}\nx=${fmt(p.x)} y=${fmt(p.y)} z=${fmt(p.z)}`;
}

function worldToMap(x, y, map, canvas) {
  const info = map.info;
  const ox = info.origin.position.x;
  const oy = info.origin.position.y;
  const res = info.resolution;
  const mx = (x - ox) / res;
  const my = (y - oy) / res;
  const sx = mx / info.width * canvas.width;
  const sy = canvas.height - my / info.height * canvas.height;
  return [sx, sy];
}

function drawCostmap() {
  const c = $("costmapCanvas");
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.fillStyle = "#020617";
  ctx.fillRect(0, 0, c.width, c.height);
  if (!lastCostmap) {
    ctx.fillStyle = "#94a3b8";
    ctx.fillText("等待 /move_base/local_costmap/costmap ...", 20, 30);
    return;
  }

  const map = lastCostmap;
  const info = map.info;
  $("costmapInfo").textContent = `${info.width}x${info.height}, res=${fmt(info.resolution, 3)}m`;

  const img = ctx.createImageData(c.width, c.height);
  for (let py = 0; py < c.height; py++) {
    for (let px = 0; px < c.width; px++) {
      const mx = Math.floor(px / c.width * info.width);
      const my = Math.floor((c.height - 1 - py) / c.height * info.height);
      const idxMap = my * info.width + mx;
      const val = map.data[idxMap];
      const idx = (py * c.width + px) * 4;
      let r=8, g=13, b=25;
      if (val < 0) { r=60; g=60; b=70; }
      else if (val === 0) { r=15; g=23; b=42; }
      else { const q = Math.min(255, 30 + val * 2.25); r=q; g=Math.max(0, 120-val); b=Math.max(0, 120-val); }
      img.data[idx]=r; img.data[idx+1]=g; img.data[idx+2]=b; img.data[idx+3]=255;
    }
  }
  ctx.putImageData(img, 0, 0);

  drawPath(ctx, globalPlan, map, c, "#22c55e", 2);
  drawPath(ctx, localPlan, map, c, "#38bdf8", 3);

  // 画车辆近似位置：局部 costmap 中心附近，仅作参考。
  ctx.fillStyle = "#f8fafc";
  ctx.beginPath(); ctx.arc(c.width/2, c.height/2, 5, 0, Math.PI*2); ctx.fill();
}

function drawPath(ctx, path, map, canvas, color, width) {
  if (!path || !path.poses || path.poses.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  let started = false;
  for (const ps of path.poses) {
    const p = ps.pose.position;
    const [sx, sy] = worldToMap(p.x, p.y, map, canvas);
    if (!started) { ctx.moveTo(sx, sy); started = true; }
    else ctx.lineTo(sx, sy);
  }
  ctx.stroke();
}

function drawScan() {
  const c = $("scanCanvas");
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.fillStyle = "#020617";
  ctx.fillRect(0, 0, c.width, c.height);

  // 坐标：车在底部偏中，前方朝上；默认显示 12 m 半径。
  const origin = {x: c.width/2, y: c.height*0.78};
  const meters = 12.0;
  const scale = Math.min(c.width, c.height) * 0.42 / meters;

  drawGrid(ctx, c, origin, scale, meters);
  drawLaser(ctx, scanData, origin, scale, "#ef4444", 1.4);
  drawLaser(ctx, visionScanData, origin, scale, "#f97316", 2.2);
  drawLaser(ctx, activeVisionScanData, origin, scale, "#a855f7", 2.6);

  // 车辆图标
  ctx.fillStyle = "#e5e7eb";
  ctx.fillRect(origin.x-7, origin.y-12, 14, 24);
  ctx.beginPath(); ctx.moveTo(origin.x, origin.y-22); ctx.lineTo(origin.x-10, origin.y-8); ctx.lineTo(origin.x+10, origin.y-8); ctx.closePath(); ctx.fill();

  const n = scanData && scanData.ranges ? scanData.ranges.length : 0;
  $("scanInfo").textContent = n ? `/scan beams=${n}` : "等待 /scan ...";
}

function drawGrid(ctx, canvas, origin, scale, meters) {
  ctx.strokeStyle = "rgba(148,163,184,.18)";
  ctx.lineWidth = 1;
  ctx.font = "12px Consolas";
  ctx.fillStyle = "#94a3b8";
  for (let r=2; r<=meters; r+=2) {
    ctx.beginPath(); ctx.arc(origin.x, origin.y, r*scale, 0, Math.PI*2); ctx.stroke();
    ctx.fillText(`${r}m`, origin.x + 5, origin.y - r*scale - 3);
  }
  ctx.strokeStyle = "rgba(148,163,184,.35)";
  ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(origin.x, 20); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(20, origin.y); ctx.lineTo(canvas.width-20, origin.y); ctx.stroke();
}

function drawLaser(ctx, scan, origin, scale, color, radius) {
  if (!scan || !scan.ranges) return;
  ctx.fillStyle = color;
  const maxRange = 50.0;
  for (let i=0; i<scan.ranges.length; i++) {
    const r = scan.ranges[i];
    if (!Number.isFinite(r) || r <= scan.range_min || r >= Math.min(scan.range_max, maxRange)) continue;
    const a = scan.angle_min + i * scan.angle_increment;
    // ROS base_link: x 前, y 左。画布：前方为 -Y，左为 -X。
    const bx = r * Math.cos(a);
    const by = r * Math.sin(a);
    const sx = origin.x - by * scale;
    const sy = origin.y - bx * scale;
    if (sx < 0 || sy < 0 || sx > ctx.canvas.width || sy > ctx.canvas.height) continue;
    ctx.beginPath(); ctx.arc(sx, sy, radius, 0, Math.PI*2); ctx.fill();
  }
}

function callService(name) {
  const service = cfg.services[name];
  if (!service) return;
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
  connectRosbridge();
  drawCostmap();
  drawScan();
  setInterval(drawCostmap, 1000);
  setInterval(drawScan, 300);
}

main().catch(e => {
  console.error(e);
  $("rosStatus").textContent = "配置加载失败";
  $("rosStatus").className = "badge bad";
});
