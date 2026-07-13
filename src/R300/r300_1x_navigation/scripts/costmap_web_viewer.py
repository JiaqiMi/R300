#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R300 ROS1 局部代价地图 / DWA 路径 Web 查看器

新增功能
--------
1. 实时计算并绘制每个视觉障碍物到 base_link 中心的最近距离；
2. 实时绘制 DWA 局部规划路径，并显示路径点数、路径长度和末端距离；
3. 合并 /costmap 与 /costmap_updates，避免网页地图停在初始帧；
4. 使用当前 shell 中默认的 python3，可直接在 yolo26 虚拟环境运行；
5. 不依赖 Flask、OpenCV、Pillow、NumPy。

默认订阅
--------
/move_base/local_costmap/costmap
/move_base/local_costmap/costmap_updates
/move_base/DWAPlannerROS/local_plan
/r300_vision/costmap_scan
/subject1/cmd_vel_raw
"""

import base64
import json
import math
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple


def add_ros_python_paths() -> None:
    """保留当前虚拟环境，只补充 ROS Noetic Python 包路径。"""
    distro = os.environ.get("ROS_DISTRO", "noetic")
    candidates = [
        f"/opt/ros/{distro}/lib/python3/dist-packages",
        "/usr/lib/python3/dist-packages",
    ]
    for candidate in candidates:
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.append(candidate)


add_ros_python_paths()

try:
    import rospy
    import tf2_ros
    from geometry_msgs.msg import Twist
    from map_msgs.msg import OccupancyGridUpdate
    from nav_msgs.msg import OccupancyGrid, Path
    from sensor_msgs.msg import LaserScan
except ImportError as exc:
    raise SystemExit(
        "ROS Python 模块导入失败：{}\n"
        "请保持当前虚拟环境，并先执行：\n"
        "  source /opt/ros/noetic/setup.bash\n"
        "  source ~/r300_ws/devel/setup.bash\n".format(exc)
    )


Point2D = Tuple[float, float]
Transform2D = Tuple[float, float, float]


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def transform_point(x: float, y: float, transform: Transform2D) -> Point2D:
    tx, ty, yaw = transform
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return (
        tx + cosine * x - sine * y,
        ty + sine * x + cosine * y,
    )


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.RLock()

        self.map_frame = ""
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0.0
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.map_data: Optional[List[int]] = None

        self.full_map_rx_time = 0.0
        self.latest_map_rx_time = 0.0
        self.update_count = 0
        self.invalid_update_count = 0
        self.update_rx_times: Deque[float] = deque(maxlen=100)
        self.full_map_rx_times: Deque[float] = deque(maxlen=100)

        self.global_plan: Optional[Path] = None
        self.global_plan_rx_time = 0.0

        self.local_plan: Optional[Path] = None
        self.local_plan_rx_time = 0.0

        self.obstacle_scan: Optional[LaserScan] = None
        self.obstacle_scan_rx_time = 0.0

        self.cmd_vel: Optional[Twist] = None
        self.cmd_vel_rx_time = 0.0


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>R300 Costmap & DWA</title>
<style>
:root {
  color-scheme: dark;
  --bg:#111418;
  --panel:#1d2228;
  --border:#353d46;
  --muted:#a2adba;
  --ok:#2fc879;
  --warn:#f4af3d;
  --bad:#ef5350;
}
* { box-sizing:border-box; }
body {
  margin:0;
  background:var(--bg);
  color:#edf2f7;
  font-family:Arial,"Microsoft YaHei",sans-serif;
}
header {
  padding:12px 18px;
  background:#1a1f25;
  border-bottom:1px solid var(--border);
  position:sticky;
  top:0;
  z-index:2;
}
h2 { margin:0 0 6px; font-size:20px; }
#status { color:var(--muted); font-size:13px; }
main {
  display:grid;
  grid-template-columns:minmax(560px,1fr) 410px;
  gap:14px;
  padding:14px;
}
.panel {
  background:var(--panel);
  border:1px solid var(--border);
  border-radius:9px;
  padding:12px;
}
.canvas-wrap {
  width:100%;
  display:flex;
  justify-content:center;
}
canvas {
  width:min(76vw,840px);
  height:min(76vw,840px);
  max-width:100%;
  background:#eee;
  border:1px solid #555;
  image-rendering:pixelated;
}
.controls {
  display:flex;
  flex-wrap:wrap;
  gap:14px;
  margin-top:10px;
  font-size:13px;
}
.controls select {
  margin-left:5px;
  padding:3px 6px;
  color:#edf2f7;
  background:#2a3037;
  border:1px solid #4b5561;
  border-radius:4px;
}
table {
  width:100%;
  border-collapse:collapse;
  font-size:14px;
}
td {
  padding:7px 8px;
  border-bottom:1px solid var(--border);
}
td:first-child {
  width:57%;
  color:var(--muted);
}
.ok { color:var(--ok); font-weight:bold; }
.warn { color:var(--warn); font-weight:bold; }
.bad { color:var(--bad); font-weight:bold; }
.legend { margin-top:14px; font-size:14px; }
.legend div { margin:8px 0; }
.sw {
  display:inline-block;
  width:22px;
  height:5px;
  vertical-align:middle;
  margin-right:8px;
}
.obstacle-title {
  margin:16px 0 6px;
  color:#ffd166;
  font-weight:bold;
}
@media (max-width:1050px) {
  main { grid-template-columns:1fr; }
  canvas { width:92vw; height:92vw; }
}
</style>
</head>
<body>
<header>
  <h2>R300 局部代价地图与 DWA 避障监视</h2>
  <div id="status">等待 ROS 数据……</div>
</header>

<main>
  <section class="panel">
    <div class="canvas-wrap">
      <canvas id="map" width="840" height="840"></canvas>
    </div>
    <div class="controls">
      <label><input id="showScan" type="checkbox" checked> 红色视觉扫描</label>
      <label><input id="showDistance" type="checkbox" checked> 障碍距离</label>
      <label><input id="showGlobalPlan" type="checkbox" checked> Navfn 全局路径</label>
      <label><input id="showPlan" type="checkbox" checked> DWA 局部路径</label>
      <label><input id="showRobot" type="checkbox" checked> base_link</label>
      <label>视图：
        <select id="viewMode">
          <option value="local" selected>车辆局部放大</option>
          <option value="full">完整 24 m 地图</option>
        </select>
      </label>
      <label>局部半径：
        <select id="viewRadius">
          <option value="3">3 m</option>
          <option value="4" selected>4 m</option>
          <option value="5">5 m</option>
          <option value="6">6 m</option>
          <option value="8">8 m</option>
          <option value="10">10 m</option>
        </select>
      </label>
    </div>
  </section>

  <aside class="panel">
    <table id="metrics"></table>
    <div id="obstacleList"></div>

    <div class="legend">
      <div><span class="sw" style="background:#f5c542"></span>黄色：Navfn 全局路径</div>
      <div><span class="sw" style="background:#19c76b"></span>绿色：DWA local plan</div>
      <div><span class="sw" style="background:#ef3434"></span>红色：视觉 obstacle_scan</div>
      <div><span class="sw" style="background:#ff9f1c"></span>橙色：base_link 到障碍最近点</div>
      <div><span class="sw" style="background:#2675ff"></span>蓝色：base_link 中心及朝向</div>
      <div><span class="sw" style="height:13px;background:#080808"></span>黑色：致命障碍</div>
      <div><span class="sw" style="height:13px;background:#777"></span>灰色：膨胀代价</div>
    </div>
  </aside>
</main>

<script>
const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const metricsEl = document.getElementById("metrics");
const obstacleListEl = document.getElementById("obstacleList");

function getView(snapshot) {
  const map = snapshot.map;
  const mode = document.getElementById("viewMode").value;

  if (mode === "local" && snapshot.robot) {
    const radius = parseFloat(
      document.getElementById("viewRadius").value
    );

    // 车辆略偏后放置，为车头前方留出更多显示空间。
    const forwardShift = radius * 0.18;
    const centerX = snapshot.robot.x
      + Math.cos(snapshot.robot.yaw) * forwardShift;
    const centerY = snapshot.robot.y
      + Math.sin(snapshot.robot.yaw) * forwardShift;

    return {
      min_x: centerX - radius,
      max_x: centerX + radius,
      min_y: centerY - radius,
      max_y: centerY + radius,
      mode: "local",
      radius: radius
    };
  }

  return {
    min_x: map.origin_x,
    max_x: map.origin_x + map.width * map.resolution,
    min_y: map.origin_y,
    max_y: map.origin_y + map.height * map.resolution,
    mode: "full",
    radius: null
  };
}

function worldToPixel(x, y, view) {
  return [
    (x - view.min_x) / (view.max_x - view.min_x) * canvas.width,
    canvas.height
      - (y - view.min_y) / (view.max_y - view.min_y) * canvas.height
  ];
}

function drawMapImage(offscreen, map, view) {
  const sxRaw = (view.min_x - map.origin_x) / map.resolution;
  const syBottomRaw = (view.min_y - map.origin_y) / map.resolution;
  const swRaw = (view.max_x - view.min_x) / map.resolution;
  const shRaw = (view.max_y - view.min_y) / map.resolution;

  // offscreen 已经上下翻转，因此源图 y 从 view.max_y 开始。
  let sx = sxRaw;
  let sy = map.height - (syBottomRaw + shRaw);
  let sw = swRaw;
  let sh = shRaw;

  let dx = 0;
  let dy = 0;
  let dw = canvas.width;
  let dh = canvas.height;

  if (sx < 0) {
    const ratio = -sx / sw;
    dx += ratio * dw;
    dw *= (1 - ratio);
    sw += sx;
    sx = 0;
  }
  if (sy < 0) {
    const ratio = -sy / sh;
    dy += ratio * dh;
    dh *= (1 - ratio);
    sh += sy;
    sy = 0;
  }
  if (sx + sw > map.width) {
    const ratio = (map.width - sx) / sw;
    sw = map.width - sx;
    dw *= Math.max(0, ratio);
  }
  if (sy + sh > map.height) {
    const ratio = (map.height - sy) / sh;
    sh = map.height - sy;
    dh *= Math.max(0, ratio);
  }

  ctx.fillStyle = "#d0d0d0";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (sw > 0 && sh > 0 && dw > 0 && dh > 0) {
    ctx.drawImage(
      offscreen,
      sx, sy, sw, sh,
      dx, dy, dw, dh
    );
  }
}

function row(name, value, cls="") {
  return `<tr><td>${name}</td><td class="${cls}">${value}</td></tr>`;
}

function ageClass(age, okLimit, warnLimit) {
  if (age <= okLimit) return "ok";
  if (age <= warnLimit) return "warn";
  return "bad";
}

function drawPolyline(points, view, color, width) {
  if (!points || points.length === 0) return;

  if (points.length === 1) {
    const q = worldToPixel(points[0][0], points[0][1], view);
    ctx.beginPath();
    ctx.arc(q[0], q[1], 5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    return;
  }

  ctx.beginPath();
  points.forEach((point, index) => {
    const q = worldToPixel(point[0], point[1], view);
    if (index === 0) ctx.moveTo(q[0], q[1]);
    else ctx.lineTo(q[0], q[1]);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.stroke();

  const end = worldToPixel(
    points[points.length - 1][0],
    points[points.length - 1][1],
    view
  );
  ctx.beginPath();
  ctx.arc(end[0], end[1], 5, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

function drawDistanceMeasurement(robot, obstacle, view) {
  const start = worldToPixel(robot.x, robot.y, view);
  const end = worldToPixel(
    obstacle.nearest_map_x,
    obstacle.nearest_map_y,
    view
  );

  ctx.save();
  ctx.strokeStyle = "#ff9f1c";
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 6]);
  ctx.beginPath();
  ctx.moveTo(start[0], start[1]);
  ctx.lineTo(end[0], end[1]);
  ctx.stroke();
  ctx.setLineDash([]);

  const label = `${obstacle.distance_m.toFixed(2)} m`;
  const mx = (start[0] + end[0]) * 0.5;
  const my = (start[1] + end[1]) * 0.5;

  ctx.font = "bold 14px Arial";
  const width = ctx.measureText(label).width + 12;
  ctx.fillStyle = "rgba(20,20,20,0.82)";
  ctx.fillRect(mx - width/2, my - 18, width, 22);
  ctx.fillStyle = "#ffd166";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, mx, my - 7);
  ctx.restore();
}

function drawSnapshot(snapshot) {
  const map = snapshot.map;
  const view = getView(snapshot);
  const raw = atob(map.data_b64);
  const image = ctx.createImageData(map.width, map.height);

  for (let y = 0; y < map.height; y++) {
    for (let x = 0; x < map.width; x++) {
      const sourceIndex = y * map.width + x;
      const encoded = raw.charCodeAt(sourceIndex);

      let color;
      if (encoded === 0) {
        color = 205;
      } else {
        const value = encoded - 1;
        if (value === 0) color = 248;
        else if (value >= 90) color = 8;
        else color = Math.max(45, 238 - Math.round(value * 1.9));
      }

      const destinationY = map.height - 1 - y;
      const destinationIndex = (
        destinationY * map.width + x
      ) * 4;

      image.data[destinationIndex] = color;
      image.data[destinationIndex + 1] = color;
      image.data[destinationIndex + 2] = color;
      image.data[destinationIndex + 3] = 255;
    }
  }

  const offscreen = document.createElement("canvas");
  offscreen.width = map.width;
  offscreen.height = map.height;
  offscreen.getContext("2d").putImageData(image, 0, 0);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingEnabled = false;
  drawMapImage(offscreen, map, view);

  if (document.getElementById("showGlobalPlan").checked) {
    drawPolyline(snapshot.global_plan, view, "#f5c542", 3);
  }

  if (document.getElementById("showPlan").checked) {
    drawPolyline(snapshot.local_plan, view, "#19c76b", 7);
  }

  if (
    document.getElementById("showScan").checked
    && snapshot.scan_points
  ) {
    ctx.fillStyle = "#ef3434";
    snapshot.scan_points.forEach(point => {
      const q = worldToPixel(point[0], point[1], view);
      ctx.beginPath();
      ctx.arc(q[0], q[1], 3.6, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  if (
    document.getElementById("showDistance").checked
    && snapshot.robot
  ) {
    snapshot.obstacles.forEach(obstacle => {
      drawDistanceMeasurement(snapshot.robot, obstacle, view);
    });
  }

  if (
    document.getElementById("showRobot").checked
    && snapshot.robot
  ) {
    const q = worldToPixel(
      snapshot.robot.x,
      snapshot.robot.y,
      view
    );
    const scale = 26;

    ctx.save();
    ctx.translate(q[0], q[1]);
    ctx.rotate(-snapshot.robot.yaw);
    ctx.beginPath();
    ctx.moveTo(scale, 0);
    ctx.lineTo(-scale * 0.68, -scale * 0.58);
    ctx.lineTo(-scale * 0.68, scale * 0.58);
    ctx.closePath();
    ctx.fillStyle = "#2675ff";
    ctx.fill();
    ctx.restore();
  }

  const mapLive = snapshot.ages.costmap < 1.0;
  statusEl.innerHTML =
    `<span class="${mapLive ? "ok" : "bad"}">` +
    `${mapLive ? "LIVE" : "STALE"}</span>` +
    ` | frame=${map.frame}` +
    ` | ${map.width}×${map.height}` +
    ` | ${map.resolution.toFixed(3)} m/cell` +
    ` | 视图=${view.mode === "local" ? "局部±" + view.radius + "m" : "完整"}` +
    ` | ${new Date().toLocaleTimeString()}`;

  let metrics = "";
  metrics += row(
    "最新 costmap 年龄",
    snapshot.ages.costmap.toFixed(3) + " s",
    ageClass(snapshot.ages.costmap, 0.8, 2.0)
  );
  metrics += row(
    "完整 costmap 频率",
    snapshot.stats.full_map_rate.toFixed(2) + " Hz",
    snapshot.stats.full_map_rate >= 2.0 ? "ok" : "warn"
  );
  metrics += row(
    "costmap_updates 频率",
    snapshot.stats.update_rate.toFixed(2) + " Hz",
    ""
  );
  metrics += row(
    "有效地图刷新频率",
    snapshot.stats.effective_map_rate.toFixed(2) + " Hz",
    snapshot.stats.effective_map_rate >= 2.0 ? "ok" : "warn"
  );
  metrics += row("已合并增量更新", snapshot.stats.update_count);
  metrics += row("致命障碍格", snapshot.stats.lethal);
  metrics += row("膨胀/非零代价格", snapshot.stats.nonzero);
  metrics += row(
    "视觉扫描年龄",
    snapshot.ages.scan.toFixed(3) + " s",
    ageClass(snapshot.ages.scan, 0.5, 1.0)
  );
  metrics += row("视觉有限扫描束", snapshot.stats.scan_finite);
  metrics += row("视觉障碍簇数量", snapshot.stats.obstacle_count);

  const closest = snapshot.stats.closest_obstacle_m;
  metrics += row(
    "最近障碍距 base_link",
    closest === null ? "无" : closest.toFixed(2) + " m",
    closest === null ? "" :
      (closest < 1.0 ? "bad" : (closest < 2.0 ? "warn" : "ok"))
  );

  metrics += row(
    "Navfn 全局路径年龄",
    snapshot.ages.global_plan.toFixed(3) + " s",
    snapshot.stats.global_plan_points > 0
      ? ageClass(snapshot.ages.global_plan, 1.5, 4.0)
      : ""
  );
  metrics += row("全局路径点数", snapshot.stats.global_plan_points);
  metrics += row(
    "全局路径长度",
    snapshot.stats.global_plan_length_m.toFixed(2) + " m"
  );

  metrics += row(
    "DWA local plan 年龄",
    snapshot.ages.plan.toFixed(3) + " s",
    snapshot.stats.plan_points > 0
      ? ageClass(snapshot.ages.plan, 0.8, 2.0)
      : ""
  );
  metrics += row("local plan 点数", snapshot.stats.plan_points);
  metrics += row(
    "local plan 长度",
    snapshot.stats.plan_length_m.toFixed(2) + " m"
  );
  metrics += row(
    "local plan 末端距车辆",
    snapshot.stats.plan_endpoint_distance_m === null
      ? "无"
      : snapshot.stats.plan_endpoint_distance_m.toFixed(2) + " m"
  );
  metrics += row(
    "cmd linear.x",
    snapshot.cmd.linear_x.toFixed(3) + " m/s"
  );
  metrics += row(
    "cmd angular.z",
    snapshot.cmd.angular_z.toFixed(3) + " rad/s"
  );
  metricsEl.innerHTML = metrics;

  if (snapshot.obstacles.length === 0) {
    obstacleListEl.innerHTML =
      `<div class="obstacle-title">障碍物距离</div>` +
      `<div style="color:#9eabb8">当前没有有效视觉障碍</div>`;
  } else {
    let html = `<div class="obstacle-title">障碍物距离（按近到远）</div>`;
    html += `<table>`;
    snapshot.obstacles.forEach((obstacle, index) => {
      html += row(
        `障碍 ${index + 1}`,
        `${obstacle.distance_m.toFixed(2)} m` +
        `，方位 ${obstacle.bearing_deg.toFixed(1)}°`
      );
    });
    html += `</table>`;
    obstacleListEl.innerHTML = html;
  }
}

async function poll() {
  try {
    const response = await fetch(
      "/snapshot?ts=" + Date.now(),
      {cache:"no-store"}
    );
    if (!response.ok) {
      throw new Error(await response.text());
    }
    drawSnapshot(await response.json());
  } catch (error) {
    statusEl.innerHTML =
      `<span class="bad">获取失败：${error}</span>`;
  }
}

setInterval(poll, 350);
poll();
</script>
</body>
</html>
"""


class CostmapWebViewer:
    def __init__(self) -> None:
        self.state = SharedState()

        self.host = str(rospy.get_param("~host", "0.0.0.0"))
        self.port = int(rospy.get_param("~port", 8088))

        self.costmap_topic = str(rospy.get_param(
            "~costmap_topic",
            "/move_base/local_costmap/costmap",
        ))
        self.costmap_updates_topic = str(rospy.get_param(
            "~costmap_updates_topic",
            "/move_base/local_costmap/costmap_updates",
        ))
        self.global_plan_topic = str(rospy.get_param(
            "~global_plan_topic",
            "/move_base/NavfnROS/plan",
        ))
        self.local_plan_topic = str(rospy.get_param(
            "~local_plan_topic",
            "/move_base/DWAPlannerROS/local_plan",
        ))
        self.scan_topic = str(rospy.get_param(
            "~obstacle_scan_topic",
            "/r300_vision/active_obstacle_scan",
        ))
        self.cmd_topic = str(rospy.get_param(
            "~cmd_vel_topic",
            "/subject1/cmd_vel_raw",
        ))
        self.robot_frame = str(rospy.get_param(
            "~robot_frame",
            "base_link",
        ))
        # /r300_vision/costmap_scan 中 10.4 m 是清除射线端点，
        # 网页只绘制 obstacle_range 内的真实障碍束。
        self.display_obstacle_range_m = float(rospy.get_param(
            "~display_obstacle_range_m",
            10.0,
        ))

        self.cluster_gap_m = float(rospy.get_param(
            "~obstacle_cluster_gap_m",
            0.60,
        ))
        self.min_cluster_beams = max(1, int(rospy.get_param(
            "~min_obstacle_cluster_beams",
            1,
        )))
        self.max_obstacle_labels = max(1, int(rospy.get_param(
            "~max_obstacle_labels",
            10,
        )))

        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rospy.Duration(15.0)
        )
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.Subscriber(
            self.costmap_topic,
            OccupancyGrid,
            self.full_map_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.costmap_updates_topic,
            OccupancyGridUpdate,
            self.map_update_callback,
            queue_size=50,
        )
        rospy.Subscriber(
            self.global_plan_topic,
            Path,
            self.global_plan_callback,
            queue_size=2,
        )
        rospy.Subscriber(
            self.local_plan_topic,
            Path,
            self.local_plan_callback,
            queue_size=2,
        )
        rospy.Subscriber(
            self.scan_topic,
            LaserScan,
            self.scan_callback,
            queue_size=2,
        )
        rospy.Subscriber(
            self.cmd_topic,
            Twist,
            self.cmd_callback,
            queue_size=2,
        )

        rospy.loginfo(
            "Web查看器订阅：global plan=%s；local plan=%s；costmap scan=%s",
            self.global_plan_topic,
            self.local_plan_topic,
            self.scan_topic,
        )

    def full_map_callback(self, message: OccupancyGrid) -> None:
        expected = int(message.info.width) * int(message.info.height)
        if len(message.data) != expected:
            rospy.logerr_throttle(
                2.0,
                "完整costmap长度错误：期望%d，实际%d",
                expected,
                len(message.data),
            )
            return

        now = time.monotonic()
        with self.state.lock:
            self.state.map_frame = message.header.frame_id
            self.state.map_width = int(message.info.width)
            self.state.map_height = int(message.info.height)
            self.state.map_resolution = float(message.info.resolution)
            self.state.map_origin_x = float(
                message.info.origin.position.x
            )
            self.state.map_origin_y = float(
                message.info.origin.position.y
            )
            self.state.map_data = list(message.data)
            self.state.full_map_rx_time = now
            self.state.latest_map_rx_time = now
            self.state.full_map_rx_times.append(now)
            # 不再清空增量统计。always_send_full_costmap=true 时，
            # costmap_updates=0 Hz 是正常的，应查看完整地图频率。

    def map_update_callback(
        self,
        message: OccupancyGridUpdate,
    ) -> None:
        now = time.monotonic()

        with self.state.lock:
            if self.state.map_data is None:
                self.state.invalid_update_count += 1
                rospy.logwarn_throttle(
                    2.0,
                    "尚未收到完整costmap，暂时忽略增量更新。",
                )
                return

            map_width = self.state.map_width
            map_height = self.state.map_height

            x = int(message.x)
            y = int(message.y)
            width = int(message.width)
            height = int(message.height)

            valid = (
                x >= 0
                and y >= 0
                and width > 0
                and height > 0
                and x + width <= map_width
                and y + height <= map_height
                and len(message.data) == width * height
            )
            if not valid:
                self.state.invalid_update_count += 1
                rospy.logwarn_throttle(
                    2.0,
                    "非法costmap更新：x=%d y=%d w=%d h=%d data=%d",
                    x,
                    y,
                    width,
                    height,
                    len(message.data),
                )
                return

            source = 0
            for row in range(height):
                destination = (y + row) * map_width + x
                self.state.map_data[
                    destination:destination + width
                ] = list(message.data[source:source + width])
                source += width

            self.state.latest_map_rx_time = now
            self.state.update_count += 1
            self.state.update_rx_times.append(now)

    def global_plan_callback(self, message: Path) -> None:
        with self.state.lock:
            self.state.global_plan = message
            self.state.global_plan_rx_time = time.monotonic()

    def local_plan_callback(self, message: Path) -> None:
        with self.state.lock:
            self.state.local_plan = message
            self.state.local_plan_rx_time = time.monotonic()

    def scan_callback(self, message: LaserScan) -> None:
        with self.state.lock:
            self.state.obstacle_scan = message
            self.state.obstacle_scan_rx_time = time.monotonic()

    def cmd_callback(self, message: Twist) -> None:
        with self.state.lock:
            self.state.cmd_vel = message
            self.state.cmd_vel_rx_time = time.monotonic()

    def lookup_transform_2d(
        self,
        target_frame: str,
        source_frame: str,
        stamp: Optional[rospy.Time] = None,
    ) -> Optional[Transform2D]:
        if not source_frame:
            return None
        if target_frame == source_frame:
            return (0.0, 0.0, 0.0)

        lookup_time = stamp
        if lookup_time is None or lookup_time == rospy.Time():
            lookup_time = rospy.Time(0)

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                lookup_time,
                rospy.Duration(0.12),
            )
        except Exception as exact_exc:
            # 只作为显示兜底；正常情况下应使用消息时间戳，
            # 这样车辆原地旋转时红线才会与 costmap 中的黑团对齐。
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rospy.Time(0),
                    rospy.Duration(0.08),
                )
                rospy.logwarn_throttle(
                    3.0,
                    "Web查看器无法按消息时间变换，临时使用最新TF：%s <- %s：%s",
                    target_frame,
                    source_frame,
                    str(exact_exc),
                )
            except Exception as latest_exc:
                rospy.logwarn_throttle(
                    3.0,
                    "Web查看器TF失败：%s <- %s：%s",
                    target_frame,
                    source_frame,
                    str(latest_exc),
                )
                return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return (
            float(translation.x),
            float(translation.y),
            quaternion_to_yaw(
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
            ),
        )

    def path_to_map_points(
        self,
        path: Optional[Path],
        map_frame: str,
    ) -> List[List[float]]:
        if path is None or not path.poses:
            return []

        default_source_frame = path.header.frame_id
        transform_cache: Dict[str, Optional[Transform2D]] = {}
        points: List[List[float]] = []

        for pose_stamped in path.poses:
            source_frame = (
                pose_stamped.header.frame_id
                or default_source_frame
            )
            if source_frame not in transform_cache:
                transform_cache[source_frame] = self.lookup_transform_2d(
                    map_frame,
                    source_frame,
                )

            transform = transform_cache[source_frame]
            if transform is None:
                continue

            point = transform_point(
                float(pose_stamped.pose.position.x),
                float(pose_stamped.pose.position.y),
                transform,
            )
            points.append([point[0], point[1]])

        return points

    def scan_to_points_and_obstacles(
        self,
        scan: Optional[LaserScan],
        map_frame: str,
    ) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        if scan is None:
            return [], []

        map_from_scan = self.lookup_transform_2d(
            map_frame,
            scan.header.frame_id,
            scan.header.stamp,
        )
        base_from_scan = self.lookup_transform_2d(
            self.robot_frame,
            scan.header.frame_id,
            scan.header.stamp,
        )
        if map_from_scan is None or base_from_scan is None:
            return [], []

        scan_points: List[List[float]] = []
        clusters: List[List[Dict[str, float]]] = []
        current_cluster: List[Dict[str, float]] = []
        previous_base_point: Optional[Point2D] = None

        angle = float(scan.angle_min)

        def finish_cluster() -> None:
            nonlocal current_cluster
            if len(current_cluster) >= self.min_cluster_beams:
                clusters.append(current_cluster)
            current_cluster = []

        for value in scan.ranges:
            finite = (
                math.isfinite(value)
                and scan.range_min <= value
                <= min(scan.range_max, self.display_obstacle_range_m)
            )

            if not finite:
                finish_cluster()
                previous_base_point = None
                angle += float(scan.angle_increment)
                continue

            local_x = float(value) * math.cos(angle)
            local_y = float(value) * math.sin(angle)

            map_point = transform_point(
                local_x,
                local_y,
                map_from_scan,
            )
            base_point = transform_point(
                local_x,
                local_y,
                base_from_scan,
            )

            if previous_base_point is not None:
                gap = math.hypot(
                    base_point[0] - previous_base_point[0],
                    base_point[1] - previous_base_point[1],
                )
                if gap > self.cluster_gap_m:
                    finish_cluster()

            item = {
                "map_x": map_point[0],
                "map_y": map_point[1],
                "base_x": base_point[0],
                "base_y": base_point[1],
                "distance": math.hypot(base_point[0], base_point[1]),
            }
            current_cluster.append(item)
            scan_points.append([map_point[0], map_point[1]])
            previous_base_point = base_point
            angle += float(scan.angle_increment)

        finish_cluster()

        obstacles: List[Dict[str, float]] = []
        for cluster in clusters:
            nearest = min(cluster, key=lambda item: item["distance"])
            center_base_x = sum(item["base_x"] for item in cluster) / len(cluster)
            center_base_y = sum(item["base_y"] for item in cluster) / len(cluster)

            obstacles.append({
                "distance_m": float(nearest["distance"]),
                "bearing_deg": math.degrees(
                    math.atan2(center_base_y, center_base_x)
                ),
                "nearest_map_x": float(nearest["map_x"]),
                "nearest_map_y": float(nearest["map_y"]),
                "beam_count": float(len(cluster)),
            })

        obstacles.sort(key=lambda item: item["distance_m"])
        return scan_points, obstacles[:self.max_obstacle_labels]

    @staticmethod
    def path_length(points: Sequence[Sequence[float]]) -> float:
        length = 0.0
        for index in range(1, len(points)):
            length += math.hypot(
                points[index][0] - points[index - 1][0],
                points[index][1] - points[index - 1][1],
            )
        return length

    @staticmethod
    def message_age(now: float, received_at: float) -> float:
        if received_at <= 0.0:
            return 999.0
        return max(0.0, now - received_at)

    @staticmethod
    def update_rate(times: Sequence[float]) -> float:
        if len(times) < 2:
            return 0.0
        duration = times[-1] - times[0]
        if duration <= 0.0:
            return 0.0
        return float(len(times) - 1) / duration

    def snapshot(self) -> Dict[str, Any]:
        now = time.monotonic()

        with self.state.lock:
            if self.state.map_data is None:
                raise RuntimeError(
                    "尚未收到完整costmap：{}".format(
                        self.costmap_topic
                    )
                )

            map_frame = self.state.map_frame
            map_width = self.state.map_width
            map_height = self.state.map_height
            map_resolution = self.state.map_resolution
            map_origin_x = self.state.map_origin_x
            map_origin_y = self.state.map_origin_y
            map_data = list(self.state.map_data)

            full_map_rx_time = self.state.full_map_rx_time
            latest_map_rx_time = self.state.latest_map_rx_time
            update_count = self.state.update_count
            invalid_update_count = self.state.invalid_update_count
            update_times = list(self.state.update_rx_times)
            full_map_times = list(self.state.full_map_rx_times)

            global_plan = self.state.global_plan
            global_plan_rx_time = self.state.global_plan_rx_time
            local_plan = self.state.local_plan
            local_plan_rx_time = self.state.local_plan_rx_time
            obstacle_scan = self.state.obstacle_scan
            obstacle_scan_rx_time = self.state.obstacle_scan_rx_time
            cmd_vel = self.state.cmd_vel
            cmd_vel_rx_time = self.state.cmd_vel_rx_time

        robot_transform = self.lookup_transform_2d(
            map_frame,
            self.robot_frame,
        )
        robot = None
        if robot_transform is not None:
            robot = {
                "x": robot_transform[0],
                "y": robot_transform[1],
                "yaw": robot_transform[2],
            }

        global_plan_points = self.path_to_map_points(
            global_plan,
            map_frame,
        )
        plan_points = self.path_to_map_points(
            local_plan,
            map_frame,
        )
        scan_points, obstacles = self.scan_to_points_and_obstacles(
            obstacle_scan,
            map_frame,
        )

        global_plan_length_m = self.path_length(global_plan_points)
        plan_length_m = self.path_length(plan_points)

        plan_endpoint_distance_m: Optional[float] = None
        if robot is not None and plan_points:
            endpoint = plan_points[-1]
            plan_endpoint_distance_m = math.hypot(
                endpoint[0] - robot["x"],
                endpoint[1] - robot["y"],
            )

        closest_obstacle_m: Optional[float] = None
        if obstacles:
            closest_obstacle_m = obstacles[0]["distance_m"]

        encoded_map = bytes(
            0 if value < 0 else min(100, int(value)) + 1
            for value in map_data
        )

        lethal_count = sum(
            1 for value in map_data if value >= 90
        )
        nonzero_count = sum(
            1 for value in map_data if value > 0
        )

        return {
            "map": {
                "frame": map_frame,
                "width": map_width,
                "height": map_height,
                "resolution": map_resolution,
                "origin_x": map_origin_x,
                "origin_y": map_origin_y,
                "data_b64": base64.b64encode(
                    encoded_map
                ).decode("ascii"),
            },
            "robot": robot,
            "global_plan": global_plan_points,
            "local_plan": plan_points,
            "scan_points": scan_points,
            "obstacles": obstacles,
            "cmd": {
                "linear_x": (
                    0.0
                    if cmd_vel is None
                    else float(cmd_vel.linear.x)
                ),
                "angular_z": (
                    0.0
                    if cmd_vel is None
                    else float(cmd_vel.angular.z)
                ),
            },
            "ages": {
                "costmap": self.message_age(
                    now,
                    latest_map_rx_time,
                ),
                "full_map": self.message_age(
                    now,
                    full_map_rx_time,
                ),
                "global_plan": self.message_age(
                    now,
                    global_plan_rx_time,
                ),
                "plan": self.message_age(
                    now,
                    local_plan_rx_time,
                ),
                "scan": self.message_age(
                    now,
                    obstacle_scan_rx_time,
                ),
                "cmd": self.message_age(
                    now,
                    cmd_vel_rx_time,
                ),
            },
            "stats": {
                "update_count": update_count,
                "invalid_update_count": invalid_update_count,
                "update_rate": self.update_rate(update_times),
                "full_map_rate": self.update_rate(full_map_times),
                "effective_map_rate": max(
                    self.update_rate(update_times),
                    self.update_rate(full_map_times),
                ),
                "lethal": lethal_count,
                "nonzero": nonzero_count,
                "scan_finite": len(scan_points),
                "obstacle_count": len(obstacles),
                "closest_obstacle_m": closest_obstacle_m,
                "global_plan_points": len(global_plan_points),
                "global_plan_length_m": global_plan_length_m,
                "plan_points": len(plan_points),
                "plan_length_m": plan_length_m,
                "plan_endpoint_distance_m": plan_endpoint_distance_m,
            },
        }

    def start_http_server(self) -> ThreadingHTTPServer:
        viewer = self

        class RequestHandler(BaseHTTPRequestHandler):
            def log_message(
                self,
                format_string: str,
                *args: Any,
            ) -> None:
                rospy.logdebug(format_string, *args)

            def send_body(
                self,
                status_code: int,
                body: bytes,
                content_type: str,
            ) -> None:
                self.send_response(status_code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                request_path = self.path.split("?", 1)[0]

                if request_path in ("/", "/index.html"):
                    self.send_body(
                        200,
                        HTML_PAGE.encode("utf-8"),
                        "text/html; charset=utf-8",
                    )
                    return

                if request_path == "/snapshot":
                    try:
                        payload = json.dumps(
                            viewer.snapshot(),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        self.send_body(
                            200,
                            payload,
                            "application/json; charset=utf-8",
                        )
                    except Exception as exc:
                        self.send_body(
                            503,
                            str(exc).encode("utf-8"),
                            "text/plain; charset=utf-8",
                        )
                    return

                if request_path == "/health":
                    self.send_body(
                        200,
                        b"ok\n",
                        "text/plain; charset=utf-8",
                    )
                    return

                self.send_body(
                    404,
                    b"not found\n",
                    "text/plain; charset=utf-8",
                )

        server = ThreadingHTTPServer(
            (self.host, self.port),
            RequestHandler,
        )
        server.daemon_threads = True

        thread = threading.Thread(
            target=server.serve_forever,
            name="costmap_web_http",
            daemon=True,
        )
        thread.start()

        rospy.loginfo(
            "Costmap Web查看器：http://%s:%d",
            self.host,
            self.port,
        )
        rospy.loginfo(
            "DWA局部路径话题：%s",
            self.local_plan_topic,
        )
        return server


def main() -> None:
    rospy.init_node("costmap_web_viewer")

    viewer = CostmapWebViewer()
    server = viewer.start_http_server()

    try:
        rospy.spin()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()