/* R300 Web 上位机 v26：雷达导航可选视觉指示牌临时转向。
 * 设计原则：浏览器直接通过 rosbridge JSON 协议订阅 ROS1 话题；
 * 视频由已有 web_video_server 提供；不直接发布 /cmd_vel。
 */

let cfg = null;
let ws = null;
let reconnectTimer = null;
let wsGeneration = 0;
const subscribedTopics = new Set();
let lidarDisplayEnabled = false;
let videoRetryTimer = null;
let videoRecoveryToken = 0;
let latestProcesses = {};
let lastNavigationRebindMs = 0;
let navigationRecoveryTimers = [];
let lastCostmap = null;
let costmapCanvasCache = null;
let costmapCacheKey = "";
let costmapRevision = 0;
let globalPlan = null;
let localPlan = null;
let scanData = null;
let visionScanData = null;
let activeVisionScanData = null;
let lidarScanData = null;
let activeLidarScanData = null;
let directionSignState = "未启用 / 等待节点";
let directionSignSelected = "NONE";
let directionSignGoal = null;
let lidarCloudData = null;
let elevationData = null;
let robotPose = null;  // {x, y, yaw, stampMs}
let headingDeg = null;
let latestGps = null;
let latestDetections = [];
let latestTargetFeedback = null;
let lastTargetFeedbackMs = 0;
let latestTargetPointMsg = null;
let targetRecords = [];
let targetLocalRecording = false;
let targetLocalStatus = {enabled: false, path: null, rows: 0};
let pendingTargetRecords = [];
let targetFlushTimer = null;
let safetyState = {limit: null, estop: null};
let insGpsFirstValidMs = null;
let insGpsLastMsg = null;
let latestInsStatus = null;
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
  scanCanvas: {scale: 1, tx: 0, ty: 0, dragging: false, lastX: 0, lastY: 0},
  elevationCanvas: {scale: 1, tx: 0, ty: 0, dragging: false, lastX: 0, lastY: 0}
};
const cloudView = {yaw: -0.70, pitch: 0.62, zoom: 1.0, dragging: false, lastX: 0, lastY: 0};
// 固定米制显示范围，避免每帧点云边界变化导致画面自动忽大忽小。
// 只影响浏览器绘图，不改变 PointCloud2、FAST-LIO 或任何 ROS 坐标系。
const CLOUD_DISPLAY_HALF_RANGE_M = 8.0;

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
}

function videoStreamUrl() {
  const v = cfg.video;
  return `http://${v.host}:${v.port}/stream?topic=${v.topic}&type=mjpeg&quality=${v.quality}&width=${v.width}&height=${v.height}`;
}

function setVideoUrl(force=false) {
  if (!cfg || !cfg.video || !$("video")) return;
  const base = videoStreamUrl();
  const url = force ? `${base}&_reconnect=${Date.now()}` : base;
  if (force || $("video").src !== url) $("video").src = url;
  $("videoTopic").textContent = `${cfg.video.topic}（HTTP ${cfg.video.port}）`;
}

function scheduleVideoReconnect(delay=1800) {
  if (videoRetryTimer) return;
  videoRetryTimer = setTimeout(() => {
    videoRetryTimer = null;
    if (document.visibilityState === "visible") setVideoUrl(true);
  }, delay);
}

function setupVideoReconnect() {
  const img = $("video");
  if (!img) return;
  img.addEventListener("error", () => scheduleVideoReconnect());
  img.addEventListener("load", () => {
    if (videoRetryTimer) clearTimeout(videoRetryTimer);
    videoRetryTimer = null;
  });
}

// 相机节点通常在点击按钮后数秒才建立 8080 MJPEG 服务。页面最初加载时如果
// 8080 尚未监听，浏览器不会因为后端后来启动而自动重新请求原 URL。这里仅重载
// <img> 地址，不启动/停止任何 ROS 节点，也不改变原相机启动脚本。
function recoverVideoStream(maxAttempts=30, intervalMs=1000) {
  const token = ++videoRecoveryToken;
  let attempt = 0;

  const retry = () => {
    if (token !== videoRecoveryToken) return;
    const img = $("video");
    if (!img) return;

    if (img.naturalWidth > 0 && img.naturalHeight > 0) {
      if (videoRetryTimer) clearTimeout(videoRetryTimer);
      videoRetryTimer = null;
      return;
    }

    setVideoUrl(true);
    attempt += 1;
    if (attempt < maxAttempts) setTimeout(retry, intervalMs);
  };
  retry();
}

function renewSubscription(topic, type, throttle) {
  if (!topic || !ws || ws.readyState !== WebSocket.OPEN) return;
  unsub(topic);
  setTimeout(() => sub(topic, type, throttle), 80);
}

// move_base、costmap 和障碍话题是在点击导航按钮后才出现。部分 rosbridge 版本
// 对“订阅时尚不存在的动态话题”恢复不稳定；刷新页面之所以能恢复，是刷新触发了
// 一次完整重新订阅。这里在不刷新页面、不重启节点的前提下主动重绑这些订阅。
function renewNavigationSubscriptions() {
  if (!cfg || !ws || ws.readyState !== WebSocket.OPEN) return;
  const t = cfg.topics;
  renewSubscription(t.odom, "nav_msgs/Odometry", 80);
  renewSubscription(t.cmd_vel, "geometry_msgs/Twist", 100);
  renewSubscription(t.global_plan, "nav_msgs/Path", 350);
  renewSubscription(t.local_plan, "nav_msgs/Path", 150);
  renewSubscription(t.current_goal, "geometry_msgs/PoseStamped", 500);
  renewSubscription(t.costmap, "nav_msgs/OccupancyGrid", 800);
  renewSubscription(t.scan, "sensor_msgs/LaserScan", 180);
  renewSubscription(t.vision_scan, "sensor_msgs/LaserScan", 180);
  renewSubscription(t.active_vision_scan, "sensor_msgs/LaserScan", 180);
  if (t.lidar_scan) renewSubscription(t.lidar_scan, "sensor_msgs/LaserScan", 180);
  if (t.active_lidar_scan) renewSubscription(t.active_lidar_scan, "sensor_msgs/LaserScan", 180);
  if (t.direction_sign_state) renewSubscription(t.direction_sign_state, "std_msgs/String", 150);
  if (t.direction_sign_selected) renewSubscription(t.direction_sign_selected, "std_msgs/String", 150);
  if (t.direction_sign_goal) renewSubscription(t.direction_sign_goal, "geometry_msgs/PoseStamped", 250);
  lastNavigationRebindMs = Date.now();
}

function scheduleNavigationRecovery() {
  navigationRecoveryTimers.forEach(clearTimeout);
  navigationRecoveryTimers = [0, 900, 2500, 6000].map(delay =>
    setTimeout(renewNavigationSubscriptions, delay)
  );
}

function anyNavigationRunning() {
  return Boolean(
    (latestProcesses.real_nav || {}).running ||
    (latestProcesses.costmap || {}).running ||
    (latestProcesses.lidar_nav || {}).running
  );
}

function scheduleRosReconnect(delay=1500) {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connectRosbridge();
  }, delay);
}

function connectRosbridge() {
  if (!cfg) return;
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

  const url = `ws://${cfg.rosbridge.host}:${cfg.rosbridge.port}`;
  const socket = new WebSocket(url);
  const generation = ++wsGeneration;
  ws = socket;

  socket.onopen = () => {
    if (ws !== socket || generation !== wsGeneration) return;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    reconnectTimer = null;
    subscribedTopics.clear();
    $("rosStatus").textContent = `ROSBridge 已连接 :${cfg.rosbridge.port}`;
    $("rosStatus").className = "badge good";
    logLast("已连接 " + url);
    subscribeAll();
    if (lidarDisplayEnabled) subscribeLidarDisplayTopics();
  };
  socket.onclose = () => {
    if (ws !== socket || generation !== wsGeneration) return;
    ws = null;
    subscribedTopics.clear();
    $("rosStatus").textContent = "ROSBridge 断开，重连中";
    $("rosStatus").className = "badge bad";
    scheduleRosReconnect();
  };
  socket.onerror = () => {
    if (ws !== socket) return;
    $("rosStatus").textContent = "ROSBridge 错误，准备重连";
    $("rosStatus").className = "badge bad";
    try { socket.close(); } catch (e) { /* ignore */ }
  };
  socket.onmessage = (ev) => {
    if (ws !== socket) return;
    try {
      const msg = JSON.parse(ev.data);
      if (msg.op === "publish") handleTopic(msg.topic, msg.msg);
      if (msg.op === "service_response") handleServiceResponse(msg);
    } catch (e) { console.warn("Bad websocket message", e); }
  };
}

function ensureRosbridgeConnected() {
  if (!ws || (ws.readyState !== WebSocket.OPEN && ws.readyState !== WebSocket.CONNECTING)) {
    scheduleRosReconnect(0);
  }
}

function send(obj) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return false;
  ws.send(JSON.stringify(obj));
  return true;
}
function sub(topic, type, throttle=200) {
  if (!topic || subscribedTopics.has(topic)) return;
  if (send({op: "subscribe", id: `sub:${topic}`, topic: topic, type: type, throttle_rate: throttle, queue_length: 1})) {
    subscribedTopics.add(topic);
  }
}

function unsub(topic) {
  if (!topic || !subscribedTopics.has(topic)) return;
  send({op: "unsubscribe", id: `sub:${topic}`, topic: topic});
  subscribedTopics.delete(topic);
}

function subscribeLidarDisplayTopics() {
  if (!lidarDisplayEnabled || !cfg) return;
  const t = cfg.topics;
  if (t.lidar_points_json) sub(t.lidar_points_json, "std_msgs/String", 900);
  if (t.elevation_json) sub(t.elevation_json, "std_msgs/String", 900);
}

function unsubscribeLidarDisplayTopics() {
  if (!cfg) return;
  const t = cfg.topics;
  if (t.lidar_points_json) unsub(t.lidar_points_json);
  if (t.elevation_json) unsub(t.elevation_json);
}

function setLidarDisplayEnabled(enabled, clearData=false) {
  const next = Boolean(enabled);
  if (lidarDisplayEnabled === next && !clearData) {
    if (next) subscribeLidarDisplayTopics();
    return;
  }
  lidarDisplayEnabled = next;
  if (next) {
    subscribeLidarDisplayTopics();
  } else {
    unsubscribeLidarDisplayTopics();
    if (clearData) {
      lidarCloudData = null;
      elevationData = null;
      if ($("cloudInfo")) $("cloudInfo").textContent = "显示传输已关闭；雷达感知可继续运行";
      if ($("elevationInfo")) $("elevationInfo").textContent = "显示传输已关闭；高程计算可继续运行";
      if ($("elevationRange")) $("elevationRange").textContent = "--";
      drawPointCloud();
      drawElevationMap();
    }
  }
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
  if (t.lidar_scan) sub(t.lidar_scan, "sensor_msgs/LaserScan", 180);
  if (t.active_lidar_scan) sub(t.active_lidar_scan, "sensor_msgs/LaserScan", 180);
  if (t.direction_sign_state) sub(t.direction_sign_state, "std_msgs/String", 150);
  if (t.direction_sign_selected) sub(t.direction_sign_selected, "std_msgs/String", 150);
  if (t.direction_sign_goal) sub(t.direction_sign_goal, "geometry_msgs/PoseStamped", 250);
  // 订阅本身不会启动适配器；适配器未运行时没有任何大数据传输。
  // 这样页面刷新后，只要后台适配器仍在运行，点云/高程可立即恢复。
  if (t.lidar_points_json) sub(t.lidar_points_json, "std_msgs/String", 900);
  if (t.elevation_json) sub(t.elevation_json, "std_msgs/String", 900);
  sub(t.detections, "r300_vision_msgs/DetectedObjectArray", 500);
  if (t.target_feedback) sub(t.target_feedback, "std_msgs/String", 250);
  sub(t.target_point, "geometry_msgs/PointStamped", 500);
  sub(t.dynamic_state, "std_msgs/String", 250);
  sub(t.speed_limit, "std_msgs/Float32", 250);
  sub(t.emergency_stop, "std_msgs/Bool", 250);
  if (t.ins_status) sub(t.ins_status, "r300_1x_navigation/InsStatus", 500);
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
  else if (topic === t.costmap) { lastCostmap = msg; costmapRevision += 1; costmapCanvasCache = null; drawCostmap(); updatePlanStats(); }
  else if (topic === t.scan) { scanData = msg; drawScan(); drawCostmap(); updatePlanStats(); }
  else if (topic === t.vision_scan) { visionScanData = msg; drawScan(); drawCostmap(); updatePlanStats(); }
  else if (topic === t.active_vision_scan) { activeVisionScanData = msg; drawScan(); drawCostmap(); updatePlanStats(); }
  else if (topic === t.lidar_scan) { lidarScanData = msg; drawScan(); drawCostmap(); updatePlanStats(); }
  else if (topic === t.active_lidar_scan) { activeLidarScanData = msg; drawScan(); drawCostmap(); updatePlanStats(); }
  else if (topic === t.direction_sign_state) updateDirectionSignState(msg);
  else if (topic === t.direction_sign_selected) updateDirectionSignSelected(msg);
  else if (topic === t.direction_sign_goal) updateDirectionSignGoal(msg);
  else if (topic === t.lidar_points_json) { lidarDisplayEnabled = true; updateLidarCloud(msg); }
  else if (topic === t.elevation_json) { lidarDisplayEnabled = true; updateElevationMap(msg); }
  else if (topic === t.detections) updateDetections(msg);
  else if (topic === t.target_feedback) updateTargetFeedback(msg);
  else if (topic === t.target_point) updateTargetPoint(msg);
  else if (topic === t.dynamic_state) $("dynState").textContent = msg.data;
  else if (topic === t.speed_limit) updateSafety("limit", msg.data);
  else if (topic === t.emergency_stop) updateSafety("estop", msg.data);
  else if (topic === t.ins_status) updateInsStatus(msg);
}

function renderDirectionSignStatus() {
  if ($("directionSignState")) $("directionSignState").textContent = directionSignState || "--";
  if ($("directionSignSelected")) $("directionSignSelected").textContent = directionSignSelected || "NONE";
  if ($("directionSignGoal")) {
    if (directionSignGoal && directionSignGoal.pose && directionSignGoal.pose.position) {
      const p = directionSignGoal.pose.position;
      const frame = (directionSignGoal.header && directionSignGoal.header.frame_id) || "--";
      $("directionSignGoal").textContent = `frame=${frame}, x=${fmt(Number(p.x), 2)} m, y=${fmt(Number(p.y), 2)} m`;
    } else {
      $("directionSignGoal").textContent = "--";
    }
  }
}

function updateDirectionSignState(msg) {
  const raw = String((msg && msg.data) || "--");
  const fields = {};
  raw.split(/\s+/).forEach(token => {
    const idx = token.indexOf("=");
    if (idx > 0) fields[token.slice(0, idx)] = token.slice(idx + 1);
  });

  const names = {
    WAITING: "等待识别",
    CONFIRMING: "多帧确认中",
    PAUSING_WAYPOINTS: "正在暂停GPS航点",
    EXECUTING_LOCAL_TURN: "正在执行临时转向",
    COMPLETED: "临时转向已完成",
    ERROR: "临时转向失败",
    DISABLED: "未启用"
  };
  const state = fields.state || raw;
  const details = [];
  if (fields.candidate && fields.candidate !== "NONE") details.push(`候选=${fields.candidate}`);
  if (fields.count && !fields.count.startsWith("0/")) details.push(`确认=${fields.count}`);
  if (fields.waypoint_state) details.push(`航点=${fields.waypoint_state}`);
  if (fields.error && fields.error !== "NONE") details.push(`错误=${fields.error.replaceAll("_", " ")}`);
  directionSignState = `${names[state] || state}${details.length ? `（${details.join("，")}）` : ""}`;

  if (fields.selected && fields.selected !== "NONE") {
    directionSignSelected = fields.selected;
  }
  renderDirectionSignStatus();
}

function updateDirectionSignSelected(msg) {
  directionSignSelected = String((msg && msg.data) || "NONE");
  renderDirectionSignStatus();
}

function updateDirectionSignGoal(msg) {
  directionSignGoal = msg || null;
  renderDirectionSignStatus();
}

function setupSignGuidanceToggle() {
  const toggle = $("lidarSignGuidance");
  if (!toggle) return;
  const saved = localStorage.getItem("r300_lidar_sign_guidance");
  if (saved === "true" || saved === "false") toggle.checked = saved === "true";
  toggle.addEventListener("change", () => {
    localStorage.setItem("r300_lidar_sign_guidance", String(toggle.checked));
    const mode = toggle.checked ? "开启：启动雷达导航前需先运行相机/YOLO" : "关闭：本次为纯雷达避障，不依赖摄像头";
    if ($("signGuidanceProcState") && !((latestProcesses.lidar_nav || {}).running)) {
      $("signGuidanceProcState").textContent = `路牌引导：${mode}`;
    }
  });
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
  renderTargetPoint();
}

function updateGps(m, label) {
  if (!Number.isFinite(m.latitude) || !Number.isFinite(m.longitude)) return;
  if (Math.abs(Number(m.latitude)) < 1e-9 && Math.abs(Number(m.longitude)) < 1e-9) return;
  latestGps = {
    lat: Number(m.latitude),
    lon: Number(m.longitude),
    alt: Number.isFinite(Number(m.altitude)) ? Number(m.altitude) : 0.0,
    label: label,
    receivedMs: Date.now()
  };
  $("gps").textContent = `${label}: ${fmt(m.latitude, 7)}, ${fmt(m.longitude, 7)}`;
  updateInsGpsPanel(m, label);
  renderTargetPoint();
  // 卫星地图默认使用 /one_x/fix；如果 fix 不发布，也接受 gps_fix 作为兜底。
  const fixTopic = (cfg.satellite_map && cfg.satellite_map.fix_topic) || cfg.topics.fix;
  const topicKey = label === "GPS" ? cfg.topics.gps_fix : cfg.topics.fix;
  if (topicKey === fixTopic || (!lastRx[fixTopic] && topicKey === cfg.topics.gps_fix)) {
    updateSatelliteMap(m);
  }
}

function isValidNavSat(m) {
  return Number.isFinite(m.latitude) && Number.isFinite(m.longitude) &&
         Math.abs(Number(m.latitude)) > 1e-9 && Math.abs(Number(m.longitude)) > 1e-9;
}

function updateInsGpsPanel(m, label) {
  if (!$('insGpsValid')) return;
  insGpsLastMsg = m;
  const valid = isValidNavSat(m);
  if (valid && insGpsFirstValidMs === null) insGpsFirstValidMs = Date.now();
  const lat = Number(m.latitude);
  const lon = Number(m.longitude);
  const alt = Number.isFinite(m.altitude) ? Number(m.altitude) : NaN;
  $('insGpsValid').textContent = valid ? `${label} 有效` : `${label} 无效/零值`;
  $('insGpsLla').textContent = valid
    ? `lat=${fmt(lat, 8)}, lon=${fmt(lon, 8)}, h=${Number.isFinite(alt) ? fmt(alt, 2) : '--'} m`
    : '--';
  updateInsTimer();
  if ($('insPanelStatus')) $('insPanelStatus').textContent = valid ? 'GPS已接收，等待/显示惯导状态' : '等待有效GPS';
}

function updateInsTimer() {
  if (!$('insGpsTimer')) return;
  if (insGpsFirstValidMs === null) { $('insGpsTimer').textContent = '未开始'; return; }
  const sec = Math.max(0, Math.floor((Date.now() - insGpsFirstValidMs) / 1000));
  const mm = String(Math.floor(sec / 60)).padStart(2, '0');
  const ss = String(sec % 60).padStart(2, '0');
  $('insGpsTimer').textContent = `${mm}:${ss}`;
}

function updateInsStatus(m) {
  latestInsStatus = m;
  if ($('insWorkState')) $('insWorkState').textContent = m.work_state || '--';
  if ($('insNavMode')) $('insNavMode').textContent = m.navigation_mode || '--';
  if ($('insRefMode')) $('insRefMode').textContent = `${m.position_reference || '--'} / ${m.velocity_reference || '--'}`;
  if ($('insHealth')) $('insHealth').textContent = `${m.ins_data_valid ? '有效' : '无效'} / ${m.fault ? '故障' : '正常'}`;
  if ($('insStatusRaw')) $('insStatusRaw').textContent = m.summary || JSON.stringify(m, null, 2);
  if ($('insPanelStatus')) $('insPanelStatus').textContent = m.summary || '已接收 /one_x/ins_status';
}

function sendInsResetCommand() {
  const topic = (cfg.topics && cfg.topics.ins_command) || '/one_x/command_hex';
  const hex = '55 AA 55 AA 5A A5 5A A5 BB 78 56 34 12 78 56 34 12';
  const okAdv = send({op: 'advertise', topic: topic, type: 'std_msgs/String'});
  const okPub = send({op: 'publish', topic: topic, msg: {data: hex}});
  const text = `${nowTime()} 发送惯导复位命令到 ${topic}: ${okPub ? 'sent' : 'failed: ROSBridge not connected'}`;
  if ($('insResetState')) $('insResetState').textContent = text;
  if ($('serviceLog')) $('serviceLog').textContent = (text + '\n' + $('serviceLog').textContent).split('\n').slice(0, 120).join('\n');
  setTimeout(() => send({op: 'unadvertise', topic: topic}), 500);
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

function normalizeDeg(v) {
  let x = Number(v) % 360;
  if (x < 0) x += 360;
  return x;
}

function getObjPosition(o) {
  const p = o.position || o.position_camera || o.center_3d || o.center || o.point ||
            (o.pose && o.pose.position) || (o.pose && o.pose.pose && o.pose.pose.position) || null;
  if (!p) return null;
  const out = {x: Number(p.x), y: Number(p.y), z: Number(p.z)};
  return Number.isFinite(out.x) && Number.isFinite(out.y) && Number.isFinite(out.z) ? out : null;
}

function resolveVehicleHeadingDeg() {
  const geoCfg = cfg.target_geolocation || {};
  const maxAge = Number.isFinite(Number(geoCfg.max_heading_age_s)) ? Number(geoCfg.max_heading_age_s) : 5.0;
  if (Number.isFinite(headingDeg) && ageSec(cfg.topics.heading_deg) <= maxAge) {
    return {deg: normalizeDeg(headingDeg), source: cfg.topics.heading_deg};
  }
  if (geoCfg.allow_odom_yaw_fallback !== false && robotPose && (Date.now() - robotPose.stampMs) / 1000 <= maxAge) {
    // ROS ENU yaw：东为0°、逆时针为正；转换为北0°、顺时针为正。
    return {deg: normalizeDeg(90.0 - robotPose.yaw * 180.0 / Math.PI), source: `${cfg.topics.odom} yaw换算`};
  }
  return null;
}

function estimateTargetLatLon(pos) {
  const geoCfg = cfg.target_geolocation || {};
  if (geoCfg.enabled === false) return {ok: false, reason: '目标经纬度计算已关闭'};
  if (!latestGps || !pos) return {ok: false, reason: '等待有效惯导经纬度'};
  const maxGpsAge = Number.isFinite(Number(geoCfg.max_gps_age_s)) ? Number(geoCfg.max_gps_age_s) : 5.0;
  if ((Date.now() - latestGps.receivedMs) / 1000 > maxGpsAge) return {ok: false, reason: '惯导经纬度已超时'};
  const headingInfo = resolveVehicleHeadingDeg();
  if (!headingInfo) return {ok: false, reason: '等待有效航向'};

  // 检测节点发布 camera optical 坐标：x向右、y向下、z向前。
  const forward = Number(pos.z) + Number(geoCfg.camera_forward_offset_m || 0.0);
  const right = Number(pos.x) + Number(geoCfg.camera_right_offset_m || 0.0);
  if (!Number.isFinite(forward) || !Number.isFinite(right) || forward <= 0) {
    return {ok: false, reason: '目标三维深度无效'};
  }

  const h = headingInfo.deg * Math.PI / 180.0;
  const north = forward * Math.cos(h) - right * Math.sin(h);
  const east = forward * Math.sin(h) + right * Math.cos(h);

  // WGS84局部切平面小距离换算，比固定111320更适合米级目标定位。
  const lat0 = latestGps.lat * Math.PI / 180.0;
  const a = 6378137.0;
  const e2 = 6.69437999014e-3;
  const sinLat = Math.sin(lat0);
  const w = Math.sqrt(1.0 - e2 * sinLat * sinLat);
  const rn = a / w;
  const rm = a * (1.0 - e2) / (w * w * w);
  const alt = Number.isFinite(latestGps.alt) ? latestGps.alt : 0.0;
  const cosLat = Math.max(1e-8, Math.abs(Math.cos(lat0))) * Math.sign(Math.cos(lat0) || 1);
  const lat = latestGps.lat + north / (rm + alt) * 180.0 / Math.PI;
  const lon = latestGps.lon + east / ((rn + alt) * cosLat) * 180.0 / Math.PI;

  return {
    ok: true, lat, lon, north, east, forward, right,
    headingDeg: headingInfo.deg, headingSource: headingInfo.source,
    vehicleLat: latestGps.lat, vehicleLon: latestGps.lon, vehicleAlt: latestGps.alt
  };
}

function targetClassName(o) {
  return String(o.class_name || o.label || o.name ||
                (o.class_id !== undefined ? `class_${o.class_id}` : 'object'));
}

function targetConfidence(o) {
  const v = o.confidence !== undefined ? o.confidence :
            (o.score !== undefined ? o.score : o.probability);
  return Number(v);
}

function matchTargetObject(point) {
  if (!point || !latestDetections.length) return null;
  let best = null;
  let bestD = Infinity;
  latestDetections.forEach(o => {
    const p = getObjPosition(o);
    if (!p) return;
    const d = Math.hypot(p.x - point.x, p.y - point.y, p.z - point.z);
    if (d < bestD) { bestD = d; best = o; }
  });
  const tol = Number(cfg.target_geolocation?.target_match_tolerance_m || 0.30);
  return best && bestD <= tol ? {object: best, error: bestD} : null;
}

function updateDetections(m) {
  const arr = m.objects || m.detections || m.targets || [];
  latestDetections = arr;

  // 统一回传节点在2秒内有数据时，Web只显示后端计算结果；原始检测
  // 仍保留用于 /target_point 匹配。节点未运行时才回退到浏览器计算。
  if (lastTargetFeedbackMs && Date.now() - lastTargetFeedbackMs < 2000) {
    renderTargetPoint();
    return;
  }

  if ($('detectionCount')) $('detectionCount').textContent = `${arr.length} 个目标（浏览器回退计算）`;
  if (!arr.length) {
    latestTargetPointMsg = null;
    if ($('detections')) $('detections').textContent = '当前无检测目标';
    renderTargetPoint();
    return;
  }

  const lines = [];
  const frameRecords = [];
  const frameId = (m.header && m.header.frame_id) || '';
  const targetPoint = latestTargetPointMsg && (latestTargetPointMsg.point || latestTargetPointMsg.position);
  const matched = matchTargetObject(targetPoint);

  arr.slice(0, 32).forEach((o, i) => {
    const cls = targetClassName(o);
    const conf = targetConfidence(o);
    const pos = getObjPosition(o);
    const isSelected = matched && matched.object === o;
    if (!pos || o.depth_valid === false) {
      lines.push(`${i + 1}. ${isSelected ? '[当前目标] ' : ''}${cls} | conf=${fmt(conf, 2)} | 无有效三维深度，无法计算经纬度`);
      return;
    }
    const geo = estimateTargetLatLon(pos);
    const cameraText = `相机(x右/y下/z前)=(${fmt(pos.x, 2)}, ${fmt(pos.y, 2)}, ${fmt(pos.z, 2)}) m`;
    const geoText = geo.ok
      ? `目标经纬度=(${fmt(geo.lat, 8)}, ${fmt(geo.lon, 8)})`
      : `目标经纬度=--（${geo.reason}）`;
    lines.push(`${i + 1}. ${isSelected ? '[当前目标] ' : ''}${cls} | conf=${fmt(conf, 2)} | ${cameraText} | ${geoText}`);

    const rec = {
      time: new Date().toISOString(), source_topic: cfg.topics.detections, frame_id: frameId,
      selected: !!isSelected, class: cls, confidence: conf,
      x: pos.x, y: pos.y, z: pos.z,
      vehicle_lat: geo.ok ? geo.vehicleLat : (latestGps ? latestGps.lat : ''),
      vehicle_lon: geo.ok ? geo.vehicleLon : (latestGps ? latestGps.lon : ''),
      vehicle_alt: geo.ok ? geo.vehicleAlt : (latestGps ? latestGps.alt : ''),
      heading_deg: geo.ok ? geo.headingDeg : '',
      forward_m: geo.ok ? geo.forward : '', right_m: geo.ok ? geo.right : '',
      north_m: geo.ok ? geo.north : '', east_m: geo.ok ? geo.east : '',
      target_lat: geo.ok ? geo.lat : '', target_lon: geo.ok ? geo.lon : ''
    };
    targetRecords.push(rec);
    frameRecords.push(rec);
  });

  if (targetRecords.length > 10000) targetRecords = targetRecords.slice(-10000);
  if ($('detections')) $('detections').textContent = `frame=${frameId || '--'}\n` + lines.join('\n');
  renderTargetRecordStatus();
  if (frameRecords.length) queueTargetRecords(frameRecords);
  renderTargetPoint();
}

function updateTargetFeedback(msg) {
  let payload = null;
  try {
    payload = JSON.parse(String((msg && msg.data) || "{}"));
  } catch (error) {
    console.warn("目标统一回传JSON解析失败", error);
    return;
  }

  latestTargetFeedback = payload;
  lastTargetFeedbackMs = Date.now();
  const targets = Array.isArray(payload.targets) ? payload.targets : [];
  if ($('detectionCount')) $('detectionCount').textContent = `${targets.length} 个目标（后端统一回传）`;

  const lines = [];
  const frameRecords = [];
  targets.slice(0, 32).forEach((target, index) => {
    const classIdText = target.class_id === undefined || target.class_id === null ? "?" : String(target.class_id);
    const cls = String(target.class_name || `class_${classIdText}`);
    const conf = Number(target.confidence);
    const cameraValid = [target.camera_x_m, target.camera_y_m, target.camera_z_m]
      .map(Number).every(Number.isFinite);
    const bodyValid = [target.body_x_m, target.body_y_m, target.body_z_m]
      .map(Number).every(Number.isFinite);
    const geoValid = Boolean(target.geolocation_valid) &&
      Number.isFinite(Number(target.target_latitude)) &&
      Number.isFinite(Number(target.target_longitude));

    const cameraText = cameraValid
      ? `相机(x右/y下/z前)=(${fmt(Number(target.camera_x_m), 2)}, ${fmt(Number(target.camera_y_m), 2)}, ${fmt(Number(target.camera_z_m), 2)}) m`
      : '相机三维坐标无效';
    const bodyText = bodyValid
      ? `车体(x前/y右/z下)=(${fmt(Number(target.body_x_m), 2)}, ${fmt(Number(target.body_y_m), 2)}, ${fmt(Number(target.body_z_m), 2)}) m`
      : '车体坐标无效';
    const geoText = geoValid
      ? `目标经纬度=(${fmt(Number(target.target_latitude), 8)}, ${fmt(Number(target.target_longitude), 8)})`
      : `目标经纬度=--（${target.reason || 'GPS/航向/深度无效'}）`;
    lines.push(`${index + 1}. ${cls} | conf=${fmt(conf, 2)} | ${cameraText} | ${bodyText} | ${geoText}`);

    const rec = {
      time: new Date().toISOString(),
      source_topic: cfg.topics.target_feedback,
      frame_id: payload.frame_id || '',
      selected: false,
      class: cls,
      confidence: conf,
      x: cameraValid ? Number(target.camera_x_m) : '',
      y: cameraValid ? Number(target.camera_y_m) : '',
      z: cameraValid ? Number(target.camera_z_m) : '',
      vehicle_lat: Number.isFinite(Number(payload.vehicle_latitude)) ? Number(payload.vehicle_latitude) : '',
      vehicle_lon: Number.isFinite(Number(payload.vehicle_longitude)) ? Number(payload.vehicle_longitude) : '',
      vehicle_alt: Number.isFinite(Number(payload.vehicle_altitude)) ? Number(payload.vehicle_altitude) : '',
      heading_deg: Number.isFinite(Number(payload.heading_deg)) ? Number(payload.heading_deg) : '',
      forward_m: bodyValid ? Number(target.body_x_m) : '',
      right_m: bodyValid ? Number(target.body_y_m) : '',
      north_m: Number.isFinite(Number(target.north_offset_m)) ? Number(target.north_offset_m) : '',
      east_m: Number.isFinite(Number(target.east_offset_m)) ? Number(target.east_offset_m) : '',
      target_lat: geoValid ? Number(target.target_latitude) : '',
      target_lon: geoValid ? Number(target.target_longitude) : ''
    };
    targetRecords.push(rec);
    frameRecords.push(rec);
  });

  if (targetRecords.length > 10000) targetRecords = targetRecords.slice(-10000);
  if ($('detections')) {
    $('detections').textContent = lines.length
      ? `统一回传=${cfg.topics.target_feedback}\nframe=${payload.frame_id || '--'}\n` + lines.join('\n')
      : `统一回传=${cfg.topics.target_feedback}\n当前无检测目标`;
  }
  if ($('targetGeoStatus')) {
    if (payload.geolocation_valid) {
      $('targetGeoStatus').textContent = `后端统一计算正常：GPS有效、航向有效，目标=${targets.length}`;
    } else {
      const gps = payload.gps_valid ? 'GPS有效' : 'GPS无效/超时';
      const heading = payload.heading_valid ? '航向有效' : '航向无效/超时';
      $('targetGeoStatus').textContent = `统一回传已连接：${gps}，${heading}`;
    }
  }
  renderTargetRecordStatus();
  if (frameRecords.length) queueTargetRecords(frameRecords);
  renderTargetPoint();
}

function updateTargetPoint(m) {
  latestTargetPointMsg = m;
  renderTargetPoint();
}

function renderTargetPoint() {
  if (!$('targetPoint')) return;
  if (!latestTargetPointMsg) {
    $('targetPoint').textContent = `等待 ${cfg?.topics?.target_point || '/r300_vision/target_point'} ...`;
    if ($('targetGeoStatus')) $('targetGeoStatus').textContent = '等待三维目标点';
    return;
  }
  const pRaw = latestTargetPointMsg.point || latestTargetPointMsg.position || {};
  const p = {x: Number(pRaw.x), y: Number(pRaw.y), z: Number(pRaw.z)};
  const frame = (latestTargetPointMsg.header && latestTargetPointMsg.header.frame_id) || '--';
  if (![p.x, p.y, p.z].every(Number.isFinite)) {
    $('targetPoint').textContent = `${cfg.topics.target_point}\nframe=${frame}\n目标点三维坐标无效`;
    if ($('targetGeoStatus')) $('targetGeoStatus').textContent = '目标三维坐标无效';
    return;
  }
  const matched = matchTargetObject(p);
  const cls = matched ? targetClassName(matched.object) : '未匹配到检测类型';
  const conf = matched ? targetConfidence(matched.object) : NaN;
  const geo = estimateTargetLatLon(p);
  const distance = Math.hypot(p.x, p.y, p.z);
  const lines = [
    `${cfg.topics.target_point}`,
    `类型=${cls}${Number.isFinite(conf) ? `  conf=${fmt(conf, 2)}` : ''}`,
    `frame=${frame}`,
    `相机坐标 x右=${fmt(p.x, 3)}  y下=${fmt(p.y, 3)}  z前=${fmt(p.z, 3)} m`,
    `三维距离=${fmt(distance, 3)} m`
  ];
  if (geo.ok) {
    lines.push(`车辆经纬度 lat=${fmt(geo.vehicleLat, 8)}  lon=${fmt(geo.vehicleLon, 8)}`);
    lines.push(`目标经纬度 lat=${fmt(geo.lat, 8)}  lon=${fmt(geo.lon, 8)}`);
    lines.push(`局部位移 北=${fmt(geo.north, 2)} m  东=${fmt(geo.east, 2)} m`);
    lines.push(`航向=${fmt(geo.headingDeg, 2)}°（${geo.headingSource}）`);
    if ($('targetGeoStatus')) $('targetGeoStatus').textContent = `已计算：${cls}，lat=${fmt(geo.lat, 8)}，lon=${fmt(geo.lon, 8)}`;
  } else {
    lines.push(`目标经纬度=--（${geo.reason}）`);
    if ($('targetGeoStatus')) $('targetGeoStatus').textContent = geo.reason;
  }
  $('targetPoint').textContent = lines.join('\n');
}

function clearTargetRecords() {
  targetRecords = [];
  renderTargetRecordStatus();
}

function csvCell(v) {
  const text = v === null || v === undefined ? '' : String(v);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function downloadTargetsCsv() {
  const rows = ['time,source_topic,frame_id,selected,class,confidence,x_right_m,y_down_m,z_forward_m,vehicle_lat,vehicle_lon,vehicle_alt,heading_deg,forward_m,right_m,north_m,east_m,target_lat,target_lon'];
  targetRecords.forEach(r => rows.push([
    r.time, r.source_topic, r.frame_id, r.selected, r.class, r.confidence,
    r.x, r.y, r.z, r.vehicle_lat, r.vehicle_lon, r.vehicle_alt, r.heading_deg,
    r.forward_m, r.right_m, r.north_m, r.east_m, r.target_lat, r.target_lon
  ].map(csvCell).join(',')));
  downloadText(`r300_targets_${timestampName()}.csv`, rows.join('\n'));
}

function renderTargetRecordStatus() {
  const localText = targetLocalStatus.enabled ? `本地记录中 ${targetLocalStatus.rows || 0} 条` : `本地已停 ${targetLocalStatus.rows || 0} 条`;
  if ($('targetRecordInfo')) $('targetRecordInfo').textContent = `浏览器 ${targetRecords.length} 条；${localText}`;
  if ($('targetRecordFile')) $('targetRecordFile').textContent = targetLocalStatus.path || '--';
}

async function targetRecordApi(path, body = null) {
  const options = {method: 'POST', headers: {'Content-Type': 'application/json'}};
  if (body !== null) options.body = JSON.stringify(body);
  const res = await fetch(path, options);
  const data = await res.json();
  if (data.recording) {
    targetLocalStatus = data.recording;
    targetLocalRecording = !!data.recording.enabled;
    renderTargetRecordStatus();
  }
  if (!data.ok) throw new Error(data.message || '目标记录接口失败');
  appendNodeLog(`${nowTime()} ${data.message || path}`);
  return data;
}

async function startTargetRecording() {
  try { await targetRecordApi('/api/target_record/start'); }
  catch (e) { appendNodeLog(`${nowTime()} 启动目标记录失败：${e}`); }
}

async function stopTargetRecording() {
  try { await flushTargetRecords(); await targetRecordApi('/api/target_record/stop'); }
  catch (e) { appendNodeLog(`${nowTime()} 停止目标记录失败：${e}`); }
}

function queueTargetRecords(records) {
  if (!targetLocalRecording) return;
  pendingTargetRecords.push(...records);
  if (pendingTargetRecords.length > 500) pendingTargetRecords = pendingTargetRecords.slice(-500);
  if (!targetFlushTimer) targetFlushTimer = setTimeout(flushTargetRecords, 500);
}

async function flushTargetRecords() {
  if (targetFlushTimer) { clearTimeout(targetFlushTimer); targetFlushTimer = null; }
  if (!targetLocalRecording || !pendingTargetRecords.length) return;
  const batch = pendingTargetRecords.splice(0, 200);
  try { await targetRecordApi('/api/target_record/append', {records: batch}); }
  catch (e) {
    pendingTargetRecords.unshift(...batch);
    appendNodeLog(`${nowTime()} 写入目标记录失败：${e}`);
  }
  if (pendingTargetRecords.length && targetLocalRecording) targetFlushTimer = setTimeout(flushTargetRecords, 700);
}

async function refreshTargetRecordStatus() {
  try {
    const res = await fetch('/api/target_record/status?ts=' + Date.now(), {cache: 'no-store'});
    const data = await res.json();
    if (data.recording) {
      targetLocalStatus = data.recording;
      targetLocalRecording = !!data.recording.enabled;
      renderTargetRecordStatus();
    }
  } catch (e) {}
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


function parseJsonTopicMessage(msg, label) {
  try {
    const data = typeof msg.data === "string" ? JSON.parse(msg.data) : msg.data;
    if (!data || typeof data !== "object") throw new Error("empty payload");
    return data;
  } catch (e) {
    console.warn(`${label} JSON parse failed`, e);
    return null;
  }
}

function updateLidarCloud(msg) {
  const data = parseJsonTopicMessage(msg, "lidar cloud");
  if (!data || !Array.isArray(data.points)) return;
  lidarCloudData = data;
  const bounds = Array.isArray(data.bounds) ? data.bounds : [0,0,0,0,0,0];
  if ($("cloudInfo")) {
    $("cloudInfo").textContent = `${data.frame_id || "--"}，显示 ${data.count || 0}/${data.source_points || 0} 点，z=${fmt(bounds[4],2)}~${fmt(bounds[5],2)} m`;
  }
  drawPointCloud();
}

function updateElevationMap(msg) {
  const data = parseJsonTopicMessage(msg, "elevation map");
  if (!data || !Array.isArray(data.values)) return;
  elevationData = data;
  if ($("elevationInfo")) {
    $("elevationInfo").textContent = `${data.frame_id || "--"}，${data.source_rows || data.rows}×${data.source_cols || data.cols}，res=${fmt(data.resolution,3)} m，valid=${data.valid_count || 0}`;
  }
  if ($("elevationRange")) {
    $("elevationRange").textContent = `${fmt(data.min,2)} m → ${fmt(data.max,2)} m`;
  }
  drawElevationMap();
}

function setupPointCloudCanvas() {
  const c = $("cloudCanvas");
  if (!c) return;
  c.addEventListener("mousedown", (e) => {
    cloudView.dragging = true;
    cloudView.lastX = e.clientX;
    cloudView.lastY = e.clientY;
  });
  window.addEventListener("mouseup", () => { cloudView.dragging = false; });
  window.addEventListener("mousemove", (e) => {
    if (!cloudView.dragging) return;
    const dx = e.clientX - cloudView.lastX;
    const dy = e.clientY - cloudView.lastY;
    cloudView.lastX = e.clientX;
    cloudView.lastY = e.clientY;
    cloudView.yaw += dx * 0.008;
    cloudView.pitch = clamp(cloudView.pitch - dy * 0.006, 0.08, 1.45);
    drawPointCloud();
  });
  c.addEventListener("wheel", (e) => {
    e.preventDefault();
    cloudView.zoom = clamp(cloudView.zoom * (e.deltaY < 0 ? 1.12 : 1 / 1.12), 0.25, 8.0);
    drawPointCloud();
  }, {passive: false});
  c.addEventListener("dblclick", resetCloudView);
  ["showElevationGrid", "showElevationRobot"].forEach(id => {
    const el = $(id);
    if (el) el.addEventListener("change", drawElevationMap);
  });
}

function resetCloudView() {
  cloudView.yaw = -0.70;
  cloudView.pitch = 0.62;
  cloudView.zoom = 1.0;
  drawPointCloud();
}

function cloudColor(z, zMin, zMax) {
  let t = (z - zMin) / Math.max(0.05, zMax - zMin);
  t = clamp(t, 0, 1);
  const hue = 220 - 190 * t;
  return `hsl(${hue}, 92%, ${48 + 12*t}%)`;
}

function projectCloudPoint(x, y, z, scale, canvas) {
  const cyaw = Math.cos(cloudView.yaw), syaw = Math.sin(cloudView.yaw);
  const forward = cyaw * x - syaw * y;
  const left = syaw * x + cyaw * y;
  const sp = Math.sin(cloudView.pitch), cp = Math.cos(cloudView.pitch);
  return {
    x: canvas.width * 0.50 - left * scale,
    y: canvas.height * 0.68 - (forward * sp + z * cp) * scale,
    depth: forward * cp - z * sp
  };
}

function drawCloudAxis(ctx, canvas, scale) {
  const axes = [
    {p:[2,0,0], color:"#ff5a5f", label:"前 x+"},
    {p:[0,2,0], color:"#6ee7ff", label:"左 y+"},
    {p:[0,0,1.5], color:"#d7f549", label:"上 z+"}
  ];
  const o = projectCloudPoint(0,0,0,scale,canvas);
  axes.forEach(a => {
    const p = projectCloudPoint(a.p[0],a.p[1],a.p[2],scale,canvas);
    ctx.strokeStyle = a.color;
    ctx.fillStyle = a.color;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(o.x,o.y); ctx.lineTo(p.x,p.y); ctx.stroke();
    ctx.beginPath(); ctx.arc(p.x,p.y,3,0,Math.PI*2); ctx.fill();
    ctx.font = "12px Consolas";
    ctx.fillText(a.label,p.x+5,p.y-5);
  });
}

function drawPointCloud() {
  const c = $("cloudCanvas");
  if (!c) return;
  const ctx = c.getContext("2d");
  ctx.clearRect(0,0,c.width,c.height);
  ctx.fillStyle = "#020b05";
  ctx.fillRect(0,0,c.width,c.height);
  ctx.strokeStyle = "rgba(151,234,34,.16)";
  ctx.lineWidth = 1;
  for (let y=40; y<c.height; y+=40) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(c.width,y); ctx.stroke(); }
  for (let x=40; x<c.width; x+=40) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,c.height); ctx.stroke(); }

  if (!lidarCloudData || !Array.isArray(lidarCloudData.points) || lidarCloudData.points.length < 3) {
    ctx.fillStyle = "#b7cf9b";
    ctx.font = "18px Microsoft YaHei";
    ctx.fillText("等待 MID-360 / FAST-LIO 配准点云...", 28, 42);
    drawCloudAxis(ctx,c,35);
    return;
  }

  const d = lidarCloudData;
  const arr = d.points;
  const unit = Number(d.scale) || 0.001;
  const b = Array.isArray(d.bounds) ? d.bounds : [-5,5,-5,5,-1,1];
  // 不再按每一帧 bounds 自动适配缩放。MID-360 非重复扫描和抽样点数变化
  // 会让 bounds 轻微变化，旧逻辑因此产生跳大跳小。这里固定为 ±8 m。
  const horizontal = CLOUD_DISPLAY_HALF_RANGE_M;
  const scale = Math.min(c.width * 0.42 / horizontal, c.height * 0.48 / horizontal) * cloudView.zoom;
  const zMin = Number.isFinite(Number(b[4])) ? Number(b[4]) : -1;
  const zMax = Number.isFinite(Number(b[5])) ? Number(b[5]) : 1;
  const projected = [];
  for (let i=0; i+2<arr.length; i+=3) {
    const x = Number(arr[i]) * unit;
    const y = Number(arr[i+1]) * unit;
    const z = Number(arr[i+2]) * unit;
    if (!Number.isFinite(x+y+z)) continue;
    const p = projectCloudPoint(x,y,z,scale,c);
    if (p.x < -20 || p.x > c.width+20 || p.y < -20 || p.y > c.height+20) continue;
    projected.push({x:p.x,y:p.y,z:z,depth:p.depth});
  }
  projected.sort((a,bp) => bp.depth - a.depth);
  const radius = clamp(1.2 * Math.sqrt(cloudView.zoom), 1.0, 3.0);
  for (const p of projected) {
    ctx.fillStyle = cloudColor(p.z,zMin,zMax);
    ctx.fillRect(p.x-radius,p.y-radius,radius*2,radius*2);
  }
  drawCloudAxis(ctx,c,scale);
  ctx.fillStyle = "#d7f549";
  ctx.font = "13px Consolas";
  ctx.fillText(`points=${projected.length}  yaw=${fmt(cloudView.yaw*180/Math.PI,0)}°  pitch=${fmt(cloudView.pitch*180/Math.PI,0)}°  zoom=${fmt(cloudView.zoom,2)}x`, 18, c.height-18);
}

function elevationColor(t) {
  t = clamp(t,0,1);
  const stops = [
    [0.00, 24, 68, 170],
    [0.25, 24, 180, 220],
    [0.50, 52, 211, 153],
    [0.75, 245, 210, 72],
    [1.00, 239, 68, 68]
  ];
  for (let i=1; i<stops.length; i++) {
    if (t <= stops[i][0]) {
      const a=stops[i-1], b=stops[i], u=(t-a[0])/(b[0]-a[0]);
      return [Math.round(a[1]+(b[1]-a[1])*u),Math.round(a[2]+(b[2]-a[2])*u),Math.round(a[3]+(b[3]-a[3])*u)];
    }
  }
  return [239,68,68];
}

function drawElevationMap() {
  const c = $("elevationCanvas");
  if (!c) return;
  const ctx = c.getContext("2d");
  ctx.clearRect(0,0,c.width,c.height);
  ctx.fillStyle = "#020b05";
  ctx.fillRect(0,0,c.width,c.height);
  if (!elevationData || !Array.isArray(elevationData.values)) {
    ctx.fillStyle = "#b7cf9b";
    ctx.font = "18px Microsoft YaHei";
    ctx.fillText("等待 GPU 高程图...", 28, 42);
    return;
  }

  const d=elevationData, rows=Number(d.rows)||0, cols=Number(d.cols)||0;
  if (rows<=0 || cols<=0 || d.values.length < rows*cols) return;
  const off=document.createElement("canvas"); off.width=cols; off.height=rows;
  const octx=off.getContext("2d"), img=octx.createImageData(cols,rows);
  const unit=Number(d.scale)||0.001, invalid=Number(d.invalid);
  const lo=Number(d.color_min), hi=Number(d.color_max), denom=Math.max(0.02,hi-lo);
  for (let i=0;i<rows*cols;i++) {
    const raw=Number(d.values[i]), k=i*4;
    if (!Number.isFinite(raw) || raw===invalid) {
      img.data[k]=5; img.data[k+1]=18; img.data[k+2]=9; img.data[k+3]=255;
      continue;
    }
    const value=raw*unit, rgb=elevationColor((value-lo)/denom);
    img.data[k]=rgb[0]; img.data[k+1]=rgb[1]; img.data[k+2]=rgb[2]; img.data[k+3]=255;
  }
  octx.putImageData(img,0,0);

  const margin=28, availW=c.width-2*margin, availH=c.height-2*margin;
  const lengthX=Math.max(0.1,Number(d.length_x)||rows), lengthY=Math.max(0.1,Number(d.length_y)||cols);
  const aspect=lengthY/lengthX;
  let drawW=availW, drawH=drawW/aspect;
  if (drawH>availH) { drawH=availH; drawW=drawH*aspect; }
  const x0=(c.width-drawW)/2, y0=(c.height-drawH)/2;

  // 车头朝上视图（146d0d0 修复，勿删）：高程图网格是 odom 轴对齐的（不随车旋转），
  // 适配器经 FAST-LIO 树 TF 提供 robot_yaw 后，把整幅图绕图心旋转即可让"上=车头"。
  // 画面映射：上=odom +x、左=odom +y ⇒ canvas rotate(+yaw) 恰好把车头(odom 方位角 yaw)转回正上方。
  // 注意 JSON null：typeof null === "object"，不能用 Number() 判断（Number(null)===0）。
  const headingUp = (typeof d.robot_yaw === "number") && Number.isFinite(d.robot_yaw);
  const yaw = headingUp ? d.robot_yaw : 0;
  const cx=x0+drawW/2, cy=y0+drawH/2, clipR=Math.min(drawW,drawH)/2;

  ctx.save();
  applyView(ctx,"elevationCanvas");
  ctx.imageSmoothingEnabled=false;
  if (headingUp) {
    ctx.save();
    ctx.beginPath(); ctx.arc(cx,cy,clipR,0,Math.PI*2); ctx.clip(); // 圆形视窗：旋转时四角不越出面板
    ctx.translate(cx,cy); ctx.rotate(yaw); ctx.translate(-cx,-cy);
  }
  ctx.drawImage(off,x0,y0,drawW,drawH);

  if ($("showElevationGrid") && $("showElevationGrid").checked) {
    ctx.strokeStyle="rgba(255,255,255,.22)"; ctx.lineWidth=1;
    const sx=drawW/lengthY, sy=drawH/lengthX;
    for (let m=1; m<lengthY/2; m+=1) {
      [x0+drawW/2-m*sx,x0+drawW/2+m*sx].forEach(x=>{ctx.beginPath();ctx.moveTo(x,y0);ctx.lineTo(x,y0+drawH);ctx.stroke();});
    }
    for (let m=1; m<lengthX/2; m+=1) {
      [y0+drawH/2-m*sy,y0+drawH/2+m*sy].forEach(y=>{ctx.beginPath();ctx.moveTo(x0,y);ctx.lineTo(x0+drawW,y);ctx.stroke();});
    }
  }
  if (headingUp) {
    ctx.restore(); // 结束旋转与圆形裁剪
    ctx.strokeStyle="rgba(215,245,73,.65)"; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(cx,cy,clipR,0,Math.PI*2); ctx.stroke();
    // odom +x 方位角标（随车转动，供与 rviz/航向对照）：
    // 未旋转画面中 odom+x 指向"上"(0,-1)，rotate(yaw) 后变为 (sin yaw, -cos yaw)。
    const rr=clipR-12;
    ctx.fillStyle="#9fd0ff"; ctx.font="11px Consolas";
    ctx.fillText("x+", cx + rr*Math.sin(yaw) - 6, cy - rr*Math.cos(yaw) + 4);
  } else {
    ctx.strokeStyle="rgba(215,245,73,.65)"; ctx.lineWidth=1.5; ctx.strokeRect(x0,y0,drawW,drawH);
  }
  if ($("showElevationRobot") && $("showElevationRobot").checked) {
    if (headingUp) {
      drawRobotArrow(ctx,cx,cy,-Math.PI/2,20,"#2563eb","#dbeafe"); // 车头朝上视图：箭头即车头
    } else {
      // 朝向未知（TF 未就绪）：只画位置点，避免固定箭头误导方向
      ctx.beginPath(); ctx.arc(cx,cy,6,0,Math.PI*2);
      ctx.fillStyle="#2563eb"; ctx.fill();
      ctx.lineWidth=2; ctx.strokeStyle="#dbeafe"; ctx.stroke();
    }
  }
  ctx.restore();

  ctx.fillStyle="#eaffc0"; ctx.font="13px Microsoft YaHei";
  if (headingUp) {
    ctx.fillText("车头朝上",c.width/2-28,18);
    ctx.fillText(`yaw ${(yaw*180/Math.PI).toFixed(1)}°`,8,c.height/2);
  } else {
    ctx.fillText("odom x+ 朝上（车辆朝向待 TF）",c.width/2-96,18);
  }
  ctx.fillStyle="#d7f549"; ctx.font="12px Consolas";
  ctx.fillText(`${fmt(lengthX,1)}m × ${fmt(lengthY,1)}m  center=(${fmt(d.center_x,1)}, ${fmt(d.center_y,1)})`,18,c.height-10);
  drawHint(ctx,"elevationCanvas");
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
function resetCanvasView(canvasId) { const st = viewState[canvasId]; if (!st) return; st.scale = 1; st.tx = 0; st.ty = 0; drawCostmap(); drawScan(); drawElevationMap(); }
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
  const info = map.info;
  const key = `${costmapRevision}:${info.width}:${info.height}:${info.resolution}:${map.data.length}`;
  if (costmapCanvasCache && costmapCacheKey === key) return costmapCanvasCache;

  // 直接按 costmap 原始网格生成图像，而不是每次重算 760×520 个画布像素。
  // 该缓存仅在收到新 OccupancyGrid 时更新，odom/path/scan 重绘时直接复用。
  const off = document.createElement("canvas");
  off.width = info.width;
  off.height = info.height;
  const octx = off.getContext("2d");
  const img = octx.createImageData(info.width, info.height);
  for (let py = 0; py < info.height; py++) {
    const my = info.height - 1 - py;
    for (let px = 0; px < info.width; px++) {
      const val = map.data[my * info.width + px];
      const idx = (py * info.width + px) * 4;
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
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, c.width, c.height);

  if ($("showGlobal").checked) drawPathOnCostmap(ctx, globalPlan, map, c, "#22c55e", 3);
  if ($("showLocal").checked) drawPathOnCostmap(ctx, localPlan, map, c, "#38bdf8", 5);
  if ($("showCostLaser").checked) drawScanOnCostmap(ctx, scanData, map, c, "rgba(239,68,68,.72)", 1.8);
  if ($("showCostVision").checked) {
    drawScanOnCostmap(ctx, visionScanData, map, c, "rgba(249,115,22,.90)", 3.4);
    drawScanOnCostmap(ctx, activeVisionScanData, map, c, "rgba(168,85,247,.95)", 3.8);
  }
  if ($("showCostLidar") && $("showCostLidar").checked) {
    drawScanOnCostmap(ctx, lidarScanData, map, c, "rgba(255,82,82,.94)", 3.4);
    drawScanOnCostmap(ctx, activeLidarScanData, map, c, "rgba(255,224,87,.98)", 3.8);
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
  // 保留轻量统计模式；雷达/视觉虚拟 LaserScan 主要叠加到 costmap 画布。
  const rawN = scanData && scanData.ranges ? scanData.ranges.length : 0;
  const rawFinite = countFiniteScan(scanData);
  const visionFinite = countFiniteScan(visionScanData);
  const visionActiveFinite = countFiniteScan(activeVisionScanData);
  const lidarFinite = countFiniteScan(lidarScanData);
  const lidarActiveFinite = countFiniteScan(activeLidarScanData);

  let nearest = Infinity;
  [scanData, visionScanData, activeVisionScanData, lidarScanData, activeLidarScanData].forEach(scan => {
    if (!scan || !scan.ranges) return;
    for (const r of scan.ranges) {
      if (Number.isFinite(r) && r >= scan.range_min && r <= scan.range_max && r < nearest) nearest = r;
    }
  });
  const nearestText = Number.isFinite(nearest) ? `${fmt(nearest, 2)} m` : "--";

  if ($("scanInfo")) {
    $("scanInfo").textContent = rawN
      ? `/scan finite=${rawFinite}；视觉=${visionFinite}/${visionActiveFinite}；雷达=${lidarFinite}/${lidarActiveFinite}；nearest=${nearestText}`
      : `等待 /scan；视觉=${visionFinite}/${visionActiveFinite}；雷达=${lidarFinite}/${lidarActiveFinite}；nearest=${nearestText}`;
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
  const visionObs = countFiniteScan(visionScanData) + countFiniteScan(activeVisionScanData);
  const lidarObs = countFiniteScan(lidarScanData) + countFiniteScan(activeLidarScanData);
  $("obsStat").textContent = `视觉=${visionObs}, 雷达=${lidarObs}, scan=${countFiniteScan(scanData)}`;
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

async function postApi(path, logGroup="nav", body=null) {
  try {
    const options = {method: "POST", cache: "no-store"};
    if (body !== null) {
      options.headers = {"Content-Type": "application/json"};
      options.body = JSON.stringify(body);
    }
    const res = await fetch(path, options);
    const data = await res.json();
    renderProcessStatus(data.processes, data.message || "", logGroup);
    return data;
  } catch (e) {
    appendSubsystemLog(logGroup, `${nowTime()} API 调用失败：${e}`);
    return null;
  }
}

async function startProcess(name) {
  if (name === "camera") {
    const data = await postApi("/api/start_camera", "camera");
    if (data && (data.ok || ((data.processes || {}).camera || {}).running)) {
      recoverVideoStream();
    }
  }
  else if (name === "ins") await postApi("/api/start_ins", "nav");
  else if (name === "real_nav") {
    const data = await postApi("/api/start_real_nav", "nav");
    if (data && (data.ok || ((data.processes || {}).real_nav || {}).running)) scheduleNavigationRecovery();
  }
  else if (name === "costmap") {
    const data = await postApi("/api/start_costmap", "nav");
    if (data && (data.ok || ((data.processes || {}).costmap || {}).running)) scheduleNavigationRecovery();
  }
  else if (name === "lidar_nav") {
    const signGuidance = Boolean($("lidarSignGuidance") && $("lidarSignGuidance").checked);
    const cameraRunning = Boolean((latestProcesses.camera || {}).running);
    if (signGuidance && !cameraRunning) {
      appendSubsystemLog(
        "nav",
        `${nowTime()} 路牌引导已勾选，正在由后端再次确认相机/YOLO运行状态。`
      );
    }
    const data = await postApi("/api/start_lidar_nav", "nav", {sign_guidance: signGuidance});
    if (data && (data.ok || ((data.processes || {}).lidar_nav || {}).running)) scheduleNavigationRecovery();
  }
  else if (name === "lidar") await postApi("/api/start_lidar", "lidar");
  else if (name === "lidar_display") {
    const data = await postApi("/api/start_lidar_display", "lidar");
    if (data && data.ok) setLidarDisplayEnabled(true);
  }
}

async function stopProcess(name) {
  if (name === "camera") await postApi("/api/stop_camera", "camera");
  else if (name === "ins") await postApi("/api/stop_ins", "nav");
  else if (name === "real_nav") await postApi("/api/stop_real_nav", "nav");
  else if (name === "costmap") await postApi("/api/stop_costmap", "nav");
  else if (name === "lidar_nav") await postApi("/api/stop_lidar_nav", "nav");
  else if (name === "lidar") await postApi("/api/stop_lidar", "lidar");
  else if (name === "lidar_display") {
    const data = await postApi("/api/stop_lidar_display", "lidar");
    if (data && (data.ok || !data.processes || !(data.processes.lidar_display || {}).running)) {
      setLidarDisplayEnabled(false, true);
    }
  }
}

async function refreshProcessStatus() {
  try {
    const res = await fetch("/api/process_status?ts=" + Date.now(), {cache: "no-store"});
    const data = await res.json();
    renderProcessStatus(data.processes, "", "");
  } catch (e) {
    // 页面刚打开时服务可能正在启动，安静失败即可。
  }
}

function stripAnsi(text) {
  return String(text || "").replace(/\x1B\[[0-?]*[ -\/]*[@-~]/g, "");
}

function renderLogBox(id, sections, message="") {
  const el = $(id);
  if (!el) return;
  const lines = [];
  if (message) lines.push(`${nowTime()} ${stripAnsi(message)}`);
  sections.forEach(section => {
    lines.push(`[${section.title}]`);
    (section.logs || []).forEach(line => lines.push(stripAnsi(line)));
  });
  el.textContent = lines.join("\n") || "暂无日志";
  el.scrollTop = el.scrollHeight;
}

function renderProcessStatus(processes, message, messageGroup="") {
  if (!processes) return;
  latestProcesses = processes;
  const cam = processes.camera || {};
  const ins = processes.ins || {};
  const realNav = processes.real_nav || {};
  const costmap = processes.costmap || {};
  const lidarNav = processes.lidar_nav || {};
  const signGuidance = processes.sign_guidance || {};
  const lidar = processes.lidar || {};
  const lidarDisplay = processes.lidar_display || {};

  if ($("cameraProcState")) {
    $("cameraProcState").textContent = cam.running ? `相机节点：运行中${cam.pid ? ` pid=${cam.pid}` : "（ROS节点检测）"}` : "相机节点：未运行";
  }
  if ($("insProcState")) {
    $("insProcState").textContent = ins.running ? `1X 惯导：运行中 pid=${ins.pid}` : "1X 惯导：未运行";
  }
  if ($("realNavProcState")) {
    $("realNavProcState").textContent = realNav.running ? `纯实车导航：运行中 pid=${realNav.pid}` : "纯实车导航：未运行";
  }
  if ($("costmapProcState")) {
    $("costmapProcState").textContent = costmap.running ? `视觉避障 / 代价地图：运行中 pid=${costmap.pid}` : "视觉避障 / 代价地图：未运行";
  }
  if ($("lidarNavProcState")) {
    $("lidarNavProcState").textContent = lidarNav.running ? `雷达避障 / 代价地图：运行中${lidarNav.pid ? ` pid=${lidarNav.pid}` : "（ROS节点检测）"}` : "雷达避障 / 代价地图：未运行";
  }

  const signToggle = $("lidarSignGuidance");
  if (signToggle) {
    signToggle.disabled = Boolean(lidarNav.running);
    if (lidarNav.running) signToggle.checked = Boolean(signGuidance.running);
  }
  if ($("signGuidanceProcState")) {
    if (lidarNav.running && signGuidance.running) {
      $("signGuidanceProcState").textContent = "路牌引导：已启用";
    } else if (lidarNav.running) {
      $("signGuidanceProcState").textContent = "路牌引导：已关闭（纯雷达）";
      directionSignState = "DISABLED";
      directionSignSelected = "NONE";
      directionSignGoal = null;
      renderDirectionSignStatus();
    } else {
      const requested = Boolean(signToggle && signToggle.checked);
      $("signGuidanceProcState").textContent = requested
        ? "路牌引导：待启动（需要相机/YOLO）"
        : "路牌引导：待启动（纯雷达）";
    }
  }

  const lidarText = lidar.running ? `雷达感知：运行中${lidar.pid ? ` pid=${lidar.pid}` : "（ROS节点检测）"}` : "雷达感知：未运行";
  const displayText = lidarDisplay.running ? `点云 / 高程传输：运行中${lidarDisplay.pid ? ` pid=${lidarDisplay.pid}` : "（ROS节点检测）"}（rosbridge 9090）` : "点云 / 高程传输：未运行";
  if ($("lidarProcState")) $("lidarProcState").textContent = lidarText;
  if ($("lidarProcStateMirror")) $("lidarProcStateMirror").textContent = lidarText;
  if ($("lidarDisplayProcState")) $("lidarDisplayProcState").textContent = displayText;
  if ($("lidarDisplayProcStateMirror")) $("lidarDisplayProcStateMirror").textContent = displayText;

  // 只在确认适配器运行时启用；不能因 Web 包装进程状态缺失而自动退订。
  // 显式点击“关闭点云/高程显示”仍会正常退订并清空画布。
  if (lidarDisplay.running) setLidarDisplayEnabled(true);

  renderLogBox("cameraNodeLog", [
    {title: "camera-vision", logs: cam.logs || []}
  ], messageGroup === "camera" ? message : "");

  renderLogBox("navNodeLog", [
    {title: "1X-INS", logs: ins.logs || []},
    {title: "pure-real-navigation", logs: realNav.logs || []},
    {title: "vision-navigation-costmap", logs: costmap.logs || []},
    {title: "lidar-navigation-costmap", logs: lidarNav.logs || []}
  ], messageGroup === "nav" ? message : "");

  renderLogBox("lidarNodeLog", [
    {title: "lidar-sensing-elevation", logs: lidar.logs || []},
    {title: "lidar-web-display", logs: lidarDisplay.logs || []}
  ], messageGroup === "lidar" ? message : "");
}

function appendSubsystemLog(group, line) {
  const id = group === "camera" ? "cameraNodeLog" : (group === "lidar" ? "lidarNodeLog" : "navNodeLog");
  const el = $(id);
  if (!el) return;
  el.textContent = stripAnsi(line) + "\n" + el.textContent;
}

async function main() {
  await loadConfig();
  setupSignGuidanceToggle();
  renderDirectionSignStatus();
  setupVideoReconnect();
  setVideoUrl(true);
  initSatelliteMap();
  setupInteractiveCanvas("costmapCanvas", drawCostmap);
  setupInteractiveCanvas("scanCanvas", drawScan);
  setupInteractiveCanvas("elevationCanvas", drawElevationMap);
  setupPointCloudCanvas();
  connectRosbridge();
  drawCostmap(); drawScan(); drawPointCloud(); drawElevationMap();
  setInterval(() => { updatePlanStats(); drawScan(); updateInsTimer(); }, 500);
  setInterval(() => { drawCostmap(); }, 1200);
  setInterval(ensureRosbridgeConnected, 2500);
  // 动态节点启动后无需刷新页面：视频主动重载，导航话题在长期无数据时主动重绑。
  setInterval(() => {
    const camRunning = Boolean((latestProcesses.camera || {}).running);
    const video = $("video");
    if (camRunning && video && video.naturalWidth === 0 && !videoRetryTimer) {
      recoverVideoStream(8, 1200);
    }

    if (anyNavigationRunning() && cfg && ageSec(cfg.topics.costmap) > 5.0 &&
        Date.now() - lastNavigationRebindMs > 5000) {
      renewNavigationSubscriptions();
    }
  }, 2500);
  refreshTargetRecordStatus();
  refreshProcessStatus();
  setInterval(refreshProcessStatus, 2500);

  window.addEventListener("online", () => { ensureRosbridgeConnected(); setVideoUrl(true); });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      ensureRosbridgeConnected();
      if ($("video") && !$("video").naturalWidth) setVideoUrl(true);
      refreshProcessStatus();
    }
  });
}

main().catch(e => { console.error(e); $("rosStatus").textContent = "配置加载失败"; $("rosStatus").className = "badge bad"; });
