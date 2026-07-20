#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R300 ROS1 DWA web tuning laboratory.

The node is intentionally dependency-light: only ROS Python packages and the
standard-library HTTP server are used.  It can:

* display local costmap, global plan, DWA local plan, robot trail and cmd_vel;
* send a move_base goal by clicking the canvas;
* add/remove synthetic circular obstacles that are published as the same
  /r300_vision/obstacle_scan consumed by VisionSnapshotLayer;
* read and update DWAPlannerROS dynamic-reconfigure parameters at runtime;
* clear costmaps, cancel goals and reset the lightweight simulator.

Use it with subject1_dwa_web_sim.launch.  Do not run the synthetic scan output
on top of the real vision pipeline unless explicitly intended.
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
    distro = os.environ.get("ROS_DISTRO", "noetic")
    for candidate in (
        f"/opt/ros/{distro}/lib/python3/dist-packages",
        "/usr/lib/python3/dist-packages",
    ):
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.append(candidate)


add_ros_python_paths()

import actionlib  # noqa: E402
import rospy  # noqa: E402
import tf2_ros  # noqa: E402
from dynamic_reconfigure.client import Client as DynamicClient  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from map_msgs.msg import OccupancyGridUpdate  # noqa: E402
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal  # noqa: E402
from nav_msgs.msg import OccupancyGrid, Odometry, Path  # noqa: E402
from sensor_msgs.msg import LaserScan  # noqa: E402
from std_srvs.srv import Empty  # noqa: E402


Point = Tuple[float, float]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def yaw_from_quaternion(q: Any) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


def path_length(points: Sequence[Point]) -> float:
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(points, points[1:])
    )


def turn_sign_changes(points: Sequence[Point], deadband: float = 0.01) -> int:
    if len(points) < 4:
        return 0
    headings: List[float] = []
    for a, b in zip(points, points[1:]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        if math.hypot(dx, dy) > 1.0e-4:
            headings.append(math.atan2(dy, dx))
    signs: List[int] = []
    for h0, h1 in zip(headings, headings[1:]):
        delta = math.atan2(math.sin(h1 - h0), math.cos(h1 - h0))
        if abs(delta) < deadband:
            continue
        signs.append(1 if delta > 0.0 else -1)
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.RLock()

        self.map_frame = "odom"
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0.0
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.map_data: Optional[List[int]] = None
        self.map_rx = 0.0
        self.map_update_count = 0

        self.global_plan: Optional[Path] = None
        self.global_plan_rx = 0.0
        self.local_plan: Optional[Path] = None
        self.local_plan_rx = 0.0

        self.odom: Optional[Odometry] = None
        self.odom_rx = 0.0
        self.trail: Deque[Point] = deque(maxlen=2500)

        self.cmd: Optional[Twist] = None
        self.cmd_rx = 0.0
        self.cmd_history: Deque[Tuple[float, float]] = deque(maxlen=500)

        self.goal: Optional[Dict[str, float]] = None
        self.obstacles: List[Dict[str, float]] = []
        self.next_obstacle_id = 1


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>R300 DWA Web Tuner</title>
<style>
:root{color-scheme:dark;--bg:#0f141a;--panel:#18212a;--panel2:#202b35;--border:#344451;--text:#eef4f8;--muted:#9fb0bd;--accent:#4ea1ff;--ok:#43d17d;--warn:#ffb547;--bad:#ff6262}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Arial,"Microsoft YaHei",sans-serif}header{padding:11px 16px;border-bottom:1px solid var(--border);background:#141c23;position:sticky;top:0;z-index:3}h1{font-size:20px;margin:0 0 5px}h2{font-size:16px;margin:0 0 10px}small,.muted{color:var(--muted)}main{padding:12px;display:grid;grid-template-columns:minmax(520px,1fr) 410px;gap:12px}.panel{border:1px solid var(--border);background:var(--panel);border-radius:10px;padding:11px}.metrics{display:grid;grid-template-columns:repeat(4,minmax(100px,1fr));gap:8px;margin-bottom:10px}.metric{background:var(--panel2);border:1px solid var(--border);border-radius:8px;padding:8px}.metric b{display:block;font-size:18px;margin-top:4px}.canvasWrap{position:relative;width:100%;aspect-ratio:1.25;background:#111;border:1px solid var(--border);border-radius:8px;overflow:hidden}canvas{width:100%;height:100%;display:block;cursor:crosshair}.toolbar{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}.toolbar label{display:flex;align-items:center;gap:5px}.toolbar select,.toolbar input,button,.params input{background:#111820;color:var(--text);border:1px solid #465b6b;border-radius:6px;padding:7px 8px}button{cursor:pointer}button.primary{background:#2368a2;border-color:#3a86c1}button.danger{background:#7b2929;border-color:#a94040}.hint{padding:8px;border-left:3px solid var(--accent);background:#172532;color:#cfe5f7;font-size:13px;margin-top:8px}.params{display:grid;grid-template-columns:1fr 104px;gap:6px 8px;align-items:center}.params label{font-size:13px;color:#dce6ec}.params input{width:100%}.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:12px;color:var(--muted);margin-top:8px}.sw{width:18px;height:4px;display:inline-block;margin-right:4px;vertical-align:middle}.row{display:flex;gap:8px;flex-wrap:wrap}.status{font-size:13px;color:var(--muted)}#message{min-height:20px;margin-top:8px;font-size:13px}.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}@media(max-width:1050px){main{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<header>
  <h1>R300 DWA 可视化调参台</h1>
  <div class="status" id="status">等待 ROS 数据……</div>
</header>
<main>
<section>
  <div class="metrics">
    <div class="metric"><span class="muted">线速度指令</span><b id="mV">--</b></div>
    <div class="metric"><span class="muted">角速度指令</span><b id="mW">--</b></div>
    <div class="metric"><span class="muted">10秒角速度换向</span><b id="mFlip">--</b></div>
    <div class="metric"><span class="muted">局部路径转向换向</span><b id="mPlanFlip">--</b></div>
  </div>
  <div class="panel">
    <div class="canvasWrap"><canvas id="map" width="1000" height="800"></canvas></div>
    <div class="toolbar">
      <label>点击模式<select id="clickMode"><option value="goal">设置目标</option><option value="add">添加障碍</option><option value="remove">删除最近障碍</option></select></label>
      <label>障碍半径<input id="obsRadius" type="number" min="0.1" max="2" step="0.1" value="0.35" style="width:74px"></label>
      <label>视野半径<select id="viewRadius"><option>5</option><option selected>8</option><option>12</option></select>m</label>
      <label><input id="showCost" type="checkbox" checked>costmap</label>
      <label><input id="showTrail" type="checkbox" checked>实际轨迹</label>
      <label><input id="showGlobal" type="checkbox" checked>全局路径</label>
      <label><input id="showLocal" type="checkbox" checked>局部路径</label>
    </div>
    <div class="row">
      <button id="cancelGoal">取消目标</button>
      <button id="clearMap">清除costmap</button>
      <button id="clearObs">清空障碍</button>
      <button id="clearTrail">清空轨迹</button>
      <button id="resetSim" class="danger">复位仿真</button>
    </div>
    <div class="legend">
      <span><i class="sw" style="background:#ffd84d"></i>全局路径</span>
      <span><i class="sw" style="background:#35e27c"></i>DWA局部路径</span>
      <span><i class="sw" style="background:#42a5ff"></i>实际轨迹</span>
      <span><i class="sw" style="background:#ff4b4b"></i>人工视觉障碍</span>
      <span><i class="sw" style="background:#d778ff"></i>目标点</span>
    </div>
    <div class="hint">理想仿真中局部路径和实际轨迹都直，而实车仍走 S 形，说明主要问题在定位反馈或底盘动态，不应继续只调 DWA 权重。</div>
  </div>
</section>
<aside class="panel">
  <h2>视觉导航 DWA 参数（运行时生效）</h2>
  <div class="params" id="params"></div>
  <div class="row" style="margin-top:10px">
    <button id="loadParams">读取当前参数</button>
    <button id="presetStable">载入抗S形基线</button>
    <button id="applyParams" class="primary">应用参数</button>
  </div>
  <div id="message" aria-live="polite"></div>
  <hr style="border:0;border-top:1px solid var(--border);margin:13px 0">
  <h2>当前状态</h2>
  <div id="details" class="status"></div>
</aside>
</main>
<script>
const canvas=document.getElementById('map'),ctx=canvas.getContext('2d');
let snapshot=null,currentView=null;
const paramDefs=[
 ['max_vel_x','最大前进速度 m/s',0.01],['max_vel_trans','最大平移速度 m/s',0.01],
 ['max_vel_theta','最大角速度 rad/s',0.01],['min_vel_theta','最小角速度 rad/s',0.01],
 ['acc_lim_x','前进加速度 m/s²',0.05],['acc_lim_theta','角加速度 rad/s²',0.05],
 ['sim_time','轨迹预测时间 s',0.1],['vx_samples','线速度采样数',1],['vth_samples','角速度采样数',1],
 ['path_distance_bias','路径权重',1],['goal_distance_bias','目标权重',1],['occdist_scale','障碍权重',0.01],
 ['forward_point_distance','前视点距离 m',0.05],['stop_time_buffer','停车缓冲 s',0.05]
];
const pbox=document.getElementById('params');
for(const [key,label,step] of paramDefs){const l=document.createElement('label');l.textContent=label;const i=document.createElement('input');i.type='number';i.step=step;i.id='p_'+key;i.dataset.key=key;pbox.append(l,i)}
function msg(text,cls='ok'){const e=document.getElementById('message');e.className=cls;e.textContent=text}
async function api(path,body=null){const opt=body===null?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)};const r=await fetch(path,opt);const t=await r.text();let d;try{d=JSON.parse(t)}catch(_){d={ok:false,error:t}}if(!r.ok||d.ok===false)throw new Error(d.error||('HTTP '+r.status));return d}
async function loadParams(){try{const d=await api('/api/params');for(const [k] of paramDefs){const e=document.getElementById('p_'+k);if(d.params[k]!==undefined)e.value=d.params[k]}msg('已读取当前动态参数')}catch(e){msg(e.message,'bad')}}
function preset(){const p={max_vel_x:.6,max_vel_trans:.6,max_vel_theta:.28,min_vel_theta:0,acc_lim_x:.8,acc_lim_theta:.6,sim_time:1.8,vx_samples:10,vth_samples:30,path_distance_bias:32,goal_distance_bias:8,occdist_scale:.12,forward_point_distance:.25,stop_time_buffer:.8};for(const[k,v]of Object.entries(p))document.getElementById('p_'+k).value=v;msg('已载入抗S形基线，点击“应用参数”才会生效','warn')}
async function applyParams(){const p={};for(const[k]of paramDefs){const e=document.getElementById('p_'+k);if(e.value!=='')p[k]=Number(e.value)}try{const d=await api('/api/params',p);msg('参数已应用；建议重新设置一次目标观察路径变化');await loadParams()}catch(e){msg(e.message,'bad')}}
function mapBytes(s){const raw=atob(s.map.data_b64),a=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)a[i]=raw.charCodeAt(i);return a}
function viewFor(s){const r=Number(document.getElementById('viewRadius').value);const c=s.robot||{x:0,y:0};return{minX:c.x-r,maxX:c.x+r,minY:c.y-r,maxY:c.y+r}}
function projection(v){const W=canvas.width,H=canvas.height,scale=Math.min(W/(v.maxX-v.minX),H/(v.maxY-v.minY));const usedW=(v.maxX-v.minX)*scale,usedH=(v.maxY-v.minY)*scale;const left=(W-usedW)/2,top=(H-usedH)/2;return{scale,left,top,w2c:(x,y)=>[left+(x-v.minX)*scale,top+(v.maxY-y)*scale],c2w:(x,y)=>[v.minX+(x-left)/scale,v.maxY-(y-top)/scale]}}
function line(points,proj,color,width=3,dash=[]){if(!points||points.length<2)return;ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();let p=proj.w2c(points[0][0],points[0][1]);ctx.moveTo(...p);for(const q of points.slice(1)){p=proj.w2c(q[0],q[1]);ctx.lineTo(...p)}ctx.stroke();ctx.restore()}
function drawGrid(v,p){ctx.save();ctx.strokeStyle='rgba(150,170,185,.18)';ctx.lineWidth=1;for(let x=Math.ceil(v.minX);x<=v.maxX;x++){const a=p.w2c(x,v.minY),b=p.w2c(x,v.maxY);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}for(let y=Math.ceil(v.minY);y<=v.maxY;y++){const a=p.w2c(v.minX,y),b=p.w2c(v.maxX,y);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}ctx.restore()}
function drawCost(s,v,p){if(!document.getElementById('showCost').checked)return;const m=s.map,data=mapBytes(s),res=m.resolution;const x0=Math.max(0,Math.floor((v.minX-m.origin_x)/res)),x1=Math.min(m.width-1,Math.ceil((v.maxX-m.origin_x)/res));const y0=Math.max(0,Math.floor((v.minY-m.origin_y)/res)),y1=Math.min(m.height-1,Math.ceil((v.maxY-m.origin_y)/res));const px=Math.max(1,res*p.scale+0.5);for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){const enc=data[y*m.width+x];if(enc===1)continue;const val=enc===0?-1:enc-1;if(val<0)ctx.fillStyle='rgba(70,76,82,.25)';else if(val>=90)ctx.fillStyle='rgba(5,5,5,.96)';else{const a=.12+.55*val/100;ctx.fillStyle=`rgba(110,110,110,${a})`}const w=m.origin_x+x*res,wy=m.origin_y+(y+1)*res;const c=p.w2c(w,wy);ctx.fillRect(c[0],c[1],px,px)}}
function circle(x,y,r,p,color,fill=false){const c=p.w2c(x,y);ctx.beginPath();ctx.arc(c[0],c[1],Math.max(3,r*p.scale),0,Math.PI*2);ctx.strokeStyle=color;ctx.lineWidth=3;if(fill){ctx.fillStyle=color;ctx.globalAlpha=.22;ctx.fill();ctx.globalAlpha=1}ctx.stroke()}
function drawRobot(robot,p){if(!robot)return;const c=p.w2c(robot.x,robot.y),L=.9*p.scale,W=.7*p.scale;ctx.save();ctx.translate(c[0],c[1]);ctx.rotate(-robot.yaw);ctx.strokeStyle='#4ea1ff';ctx.fillStyle='rgba(78,161,255,.20)';ctx.lineWidth=3;ctx.beginPath();ctx.rect(-L/2,-W/2,L,W);ctx.fill();ctx.stroke();ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(L*.75,0);ctx.stroke();ctx.restore()}
function render(s){snapshot=s;const v=viewFor(s),p=projection(v);currentView={v,p};ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#e7ebee';ctx.fillRect(0,0,canvas.width,canvas.height);drawCost(s,v,p);drawGrid(v,p);if(document.getElementById('showTrail').checked)line(s.trail,p,'#318de4',2);if(document.getElementById('showGlobal').checked)line(s.global_plan,p,'#e5b900',4);if(document.getElementById('showLocal').checked)line(s.local_plan,p,'#18c967',4);for(const o of s.obstacles)circle(o.x,o.y,o.radius,p,'#ef3e3e',true);if(s.goal){circle(s.goal.x,s.goal.y,.18,p,'#bc5cff',false);const c=p.w2c(s.goal.x,s.goal.y);ctx.strokeStyle='#bc5cff';ctx.beginPath();ctx.moveTo(c[0]-10,c[1]);ctx.lineTo(c[0]+10,c[1]);ctx.moveTo(c[0],c[1]-10);ctx.lineTo(c[0],c[1]+10);ctx.stroke()}drawRobot(s.robot,p);document.getElementById('mV').textContent=s.cmd.linear_x.toFixed(3)+' m/s';document.getElementById('mW').textContent=s.cmd.angular_z.toFixed(3)+' rad/s';document.getElementById('mFlip').textContent=s.metrics.cmd_sign_changes;document.getElementById('mPlanFlip').textContent=s.metrics.local_plan_turn_changes;document.getElementById('details').innerHTML=`机器人: x=${s.robot?s.robot.x.toFixed(2):'--'}, y=${s.robot?s.robot.y.toFixed(2):'--'}, yaw=${s.robot?(s.robot.yaw*180/Math.PI).toFixed(1):'--'}°<br>里程计: vx=${s.odom.linear_x.toFixed(3)}, vy=${s.odom.linear_y.toFixed(3)}, wz=${s.odom.angular_z.toFixed(3)}<br>全局路径: ${s.global_plan.length}点 / ${s.metrics.global_length_m.toFixed(2)}m<br>局部路径: ${s.local_plan.length}点 / ${s.metrics.local_length_m.toFixed(2)}m<br>人工障碍: ${s.obstacles.length}个；costmap 更新: ${s.metrics.map_updates}<br>数据年龄: odom ${s.ages.odom.toFixed(2)}s，costmap ${s.ages.map.toFixed(2)}s`;document.getElementById('status').textContent=`ROS在线｜frame=${s.map.frame}｜目标=${s.goal?'已设置':'无'}｜${new Date().toLocaleTimeString()}`}
async function poll(){try{render(await api('/api/state'))}catch(e){document.getElementById('status').textContent='等待数据：'+e.message}finally{setTimeout(poll,300)}}
canvas.addEventListener('click',async ev=>{if(!currentView)return;const rect=canvas.getBoundingClientRect(),cx=(ev.clientX-rect.left)*canvas.width/rect.width,cy=(ev.clientY-rect.top)*canvas.height/rect.height,[x,y]=currentView.p.c2w(cx,cy),mode=document.getElementById('clickMode').value;try{if(mode==='goal')await api('/api/goal',{x,y});else if(mode==='add')await api('/api/obstacle',{action:'add',x,y,radius:Number(document.getElementById('obsRadius').value)});else await api('/api/obstacle',{action:'remove_nearest',x,y});msg(`${mode==='goal'?'目标':'障碍操作'}完成：(${x.toFixed(2)}, ${y.toFixed(2)})`)}catch(e){msg(e.message,'bad')}});
document.getElementById('loadParams').addEventListener('click',loadParams);document.getElementById('presetStable').addEventListener('click',preset);document.getElementById('applyParams').addEventListener('click',applyParams);
for(const [id,action] of [['cancelGoal','cancel_goal'],['clearMap','clear_costmaps'],['clearObs','clear_obstacles'],['clearTrail','clear_trail'],['resetSim','reset_sim']])document.getElementById(id).addEventListener('click',async()=>{try{await api('/api/action',{action});msg('操作完成：'+action)}catch(e){msg(e.message,'bad')}});
loadParams();poll();
</script>
</body>
</html>
"""


class DwaWebTuner:
    PARAM_SPEC: Dict[str, Tuple[type, float, float]] = {
        "max_vel_x": (float, 0.0, 3.0),
        "max_vel_trans": (float, 0.0, 3.0),
        "max_vel_theta": (float, 0.0, 3.0),
        "min_vel_theta": (float, 0.0, 1.0),
        "acc_lim_x": (float, 0.01, 20.0),
        "acc_lim_theta": (float, 0.01, 20.0),
        "sim_time": (float, 0.2, 8.0),
        "vx_samples": (int, 1, 100),
        "vth_samples": (int, 1, 200),
        "path_distance_bias": (float, 0.0, 200.0),
        "goal_distance_bias": (float, 0.0, 200.0),
        "occdist_scale": (float, 0.0, 10.0),
        "forward_point_distance": (float, 0.0, 3.0),
        "stop_time_buffer": (float, 0.0, 5.0),
    }

    def __init__(self) -> None:
        self.state = SharedState()

        # 监听所有网卡。实际浏览器访问地址由启动脚本自动识别。
        self.host = str(rospy.get_param("~host", "0.0.0.0"))
        self.port = int(rospy.get_param("~port", 8070))
        self.robot_frame = str(rospy.get_param("~robot_frame", "base_link"))
        self.goal_frame = str(rospy.get_param("~goal_frame", "map"))

        self.costmap_topic = str(rospy.get_param(
            "~costmap_topic", "/move_base/local_costmap/costmap"
        ))
        self.costmap_update_topic = str(rospy.get_param(
            "~costmap_update_topic", "/move_base/local_costmap/costmap_updates"
        ))
        self.global_plan_topic = str(rospy.get_param(
            "~global_plan_topic", "/move_base/NavfnROS/plan"
        ))
        self.local_plan_topic = str(rospy.get_param(
            "~local_plan_topic", "/move_base/DWAPlannerROS/local_plan"
        ))
        self.odom_topic = str(rospy.get_param("~odom_topic", "/one_x/odom"))
        self.cmd_topic = str(rospy.get_param(
            "~cmd_topic", "/subject1/cmd_vel_raw"
        ))

        self.dynamic_server = str(rospy.get_param(
            "~dynamic_server", "/move_base/DWAPlannerROS"
        ))
        self.move_base_action = str(rospy.get_param(
            "~move_base_action", "/move_base"
        ))
        self.clear_costmaps_service = str(rospy.get_param(
            "~clear_costmaps_service", "/move_base/clear_costmaps"
        ))
        self.reset_sim_service = str(rospy.get_param(
            "~reset_sim_service", "/sim/reset_pose"
        ))

        self.enable_synthetic_scan = bool(rospy.get_param(
            "~enable_synthetic_scan", True
        ))
        self.scan_topic = str(rospy.get_param(
            "~scan_topic", "/r300_vision/obstacle_scan"
        ))
        self.scan_rate_hz = float(rospy.get_param("~scan_rate_hz", 10.0))
        self.scan_min_angle = math.radians(float(rospy.get_param(
            "~scan_angle_min_deg", -27.0
        )))
        self.scan_max_angle = math.radians(float(rospy.get_param(
            "~scan_angle_max_deg", 27.0
        )))
        self.scan_increment = math.radians(float(rospy.get_param(
            "~scan_angle_increment_deg", 0.5
        )))
        self.scan_min_range = float(rospy.get_param("~scan_range_min_m", 0.2))
        self.scan_max_range = float(rospy.get_param("~scan_range_max_m", 10.5))

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(20.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.dynamic_client: Optional[DynamicClient] = None
        self.action_client = actionlib.SimpleActionClient(
            self.move_base_action, MoveBaseAction
        )

        rospy.Subscriber(
            self.costmap_topic, OccupancyGrid, self.costmap_cb, queue_size=2
        )
        rospy.Subscriber(
            self.costmap_update_topic,
            OccupancyGridUpdate,
            self.costmap_update_cb,
            queue_size=20,
        )
        rospy.Subscriber(
            self.global_plan_topic, Path, self.global_plan_cb, queue_size=2
        )
        rospy.Subscriber(
            self.local_plan_topic, Path, self.local_plan_cb, queue_size=5
        )
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=50)
        rospy.Subscriber(self.cmd_topic, Twist, self.cmd_cb, queue_size=50)

        self.scan_pub = rospy.Publisher(self.scan_topic, LaserScan, queue_size=2)
        if self.enable_synthetic_scan:
            rospy.Timer(
                rospy.Duration(1.0 / max(1.0, self.scan_rate_hz)),
                self.publish_scan,
            )

        rospy.logwarn(
            "DWA web tuner ready: http://%s:%d; dynamic=%s; synthetic_scan=%s",
            self.host,
            self.port,
            self.dynamic_server,
            self.enable_synthetic_scan,
        )

    def costmap_cb(self, msg: OccupancyGrid) -> None:
        with self.state.lock:
            self.state.map_frame = msg.header.frame_id or "odom"
            self.state.map_width = int(msg.info.width)
            self.state.map_height = int(msg.info.height)
            self.state.map_resolution = float(msg.info.resolution)
            self.state.map_origin_x = float(msg.info.origin.position.x)
            self.state.map_origin_y = float(msg.info.origin.position.y)
            self.state.map_data = list(msg.data)
            self.state.map_rx = time.time()
            self.state.map_update_count += 1

    def costmap_update_cb(self, msg: OccupancyGridUpdate) -> None:
        with self.state.lock:
            data = self.state.map_data
            width = self.state.map_width
            height = self.state.map_height
            if data is None or width <= 0 or height <= 0:
                return
            if msg.x < 0 or msg.y < 0:
                return
            if msg.x + msg.width > width or msg.y + msg.height > height:
                return
            if len(msg.data) != msg.width * msg.height:
                return
            for row in range(msg.height):
                src = row * msg.width
                dst = (msg.y + row) * width + msg.x
                data[dst:dst + msg.width] = msg.data[src:src + msg.width]
            self.state.map_rx = time.time()
            self.state.map_update_count += 1

    def global_plan_cb(self, msg: Path) -> None:
        with self.state.lock:
            self.state.global_plan = msg
            self.state.global_plan_rx = time.time()

    def local_plan_cb(self, msg: Path) -> None:
        with self.state.lock:
            self.state.local_plan = msg
            self.state.local_plan_rx = time.time()

    def odom_cb(self, msg: Odometry) -> None:
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        with self.state.lock:
            self.state.odom = msg
            self.state.odom_rx = time.time()
            if not self.state.trail or math.hypot(
                x - self.state.trail[-1][0], y - self.state.trail[-1][1]
            ) >= 0.015:
                self.state.trail.append((x, y))

    def cmd_cb(self, msg: Twist) -> None:
        now = time.time()
        with self.state.lock:
            self.state.cmd = msg
            self.state.cmd_rx = now
            self.state.cmd_history.append((now, float(msg.angular.z)))

    def lookup_2d(self, target: str, source: str) -> Optional[Tuple[float, float, float]]:
        if not source or source == target:
            return (0.0, 0.0, 0.0)
        try:
            tfm = self.tf_buffer.lookup_transform(
                target, source, rospy.Time(0), rospy.Duration(0.05)
            )
            t = tfm.transform.translation
            yaw = yaw_from_quaternion(tfm.transform.rotation)
            return float(t.x), float(t.y), yaw
        except Exception:
            return None

    @staticmethod
    def apply_transform(point: Point, tfm: Tuple[float, float, float]) -> Point:
        tx, ty, yaw = tfm
        c = math.cos(yaw)
        s = math.sin(yaw)
        return tx + c * point[0] - s * point[1], ty + s * point[0] + c * point[1]

    def path_points(self, msg: Optional[Path], target_frame: str) -> List[Point]:
        if msg is None:
            return []
        source = msg.header.frame_id or target_frame
        tfm = self.lookup_2d(target_frame, source)
        if tfm is None:
            return []
        poses = msg.poses
        stride = max(1, int(math.ceil(len(poses) / 1500.0)))
        return [
            self.apply_transform(
                (float(p.pose.position.x), float(p.pose.position.y)), tfm
            )
            for p in poses[::stride]
        ]

    def robot_in_frame(self, odom: Optional[Odometry], target: str) -> Optional[Dict[str, float]]:
        if odom is None:
            return None
        source = odom.header.frame_id or "odom"
        point = (float(odom.pose.pose.position.x), float(odom.pose.pose.position.y))
        yaw = yaw_from_quaternion(odom.pose.pose.orientation)
        tfm = self.lookup_2d(target, source)
        if tfm is None:
            return None
        x, y = self.apply_transform(point, tfm)
        return {"x": x, "y": y, "yaw": yaw + tfm[2]}

    def dynamic(self) -> DynamicClient:
        if self.dynamic_client is None:
            self.dynamic_client = DynamicClient(self.dynamic_server, timeout=2.0)
        return self.dynamic_client

    def get_params(self) -> Dict[str, Any]:
        config = self.dynamic().get_configuration(timeout=2.0)
        return {key: config[key] for key in self.PARAM_SPEC if key in config}

    def set_params(self, request: Dict[str, Any]) -> Dict[str, Any]:
        clean: Dict[str, Any] = {}
        for key, value in request.items():
            if key not in self.PARAM_SPEC:
                continue
            typ, lower, upper = self.PARAM_SPEC[key]
            try:
                converted = typ(value)
            except (TypeError, ValueError):
                raise ValueError(f"参数 {key} 不是有效数值")
            if converted < lower or converted > upper:
                raise ValueError(f"参数 {key} 超出允许范围 [{lower}, {upper}]")
            clean[key] = converted

        if "max_vel_x" in clean and "max_vel_trans" not in clean:
            clean["max_vel_trans"] = clean["max_vel_x"]
        if not clean:
            raise ValueError("没有可应用的DWA参数")

        updated = self.dynamic().update_configuration(clean)
        return {key: updated[key] for key in self.PARAM_SPEC if key in updated}

    def send_goal(self, x: float, y: float, yaw: Optional[float]) -> None:
        if yaw is None:
            with self.state.lock:
                odom = self.state.odom
            if odom is not None:
                rx = float(odom.pose.pose.position.x)
                ry = float(odom.pose.pose.position.y)
                yaw = math.atan2(y - ry, x - rx)
            else:
                yaw = 0.0

        if not self.action_client.wait_for_server(rospy.Duration(2.0)):
            raise RuntimeError("move_base action server尚未就绪")

        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = self.goal_frame
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.x = qx
        goal.target_pose.pose.orientation.y = qy
        goal.target_pose.pose.orientation.z = qz
        goal.target_pose.pose.orientation.w = qw
        self.action_client.send_goal(goal)

        with self.state.lock:
            self.state.goal = {"x": x, "y": y, "yaw": yaw}

    def obstacle_action(self, request: Dict[str, Any]) -> None:
        action = str(request.get("action", ""))
        with self.state.lock:
            if action == "add":
                x = float(request["x"])
                y = float(request["y"])
                radius = clamp(float(request.get("radius", 0.35)), 0.10, 2.0)
                self.state.obstacles.append({
                    "id": float(self.state.next_obstacle_id),
                    "x": x,
                    "y": y,
                    "radius": radius,
                })
                self.state.next_obstacle_id += 1
            elif action == "remove_nearest":
                if not self.state.obstacles:
                    return
                x = float(request["x"])
                y = float(request["y"])
                index = min(
                    range(len(self.state.obstacles)),
                    key=lambda i: math.hypot(
                        self.state.obstacles[i]["x"] - x,
                        self.state.obstacles[i]["y"] - y,
                    ),
                )
                self.state.obstacles.pop(index)
            elif action == "clear":
                self.state.obstacles.clear()
            else:
                raise ValueError("未知障碍操作")

    def publish_scan(self, _event: Any) -> None:
        with self.state.lock:
            odom = self.state.odom
            obstacles = [dict(item) for item in self.state.obstacles]
        if odom is None:
            return

        pose = odom.pose.pose
        robot_x = float(pose.position.x)
        robot_y = float(pose.position.y)
        robot_yaw = yaw_from_quaternion(pose.orientation)
        c = math.cos(robot_yaw)
        s = math.sin(robot_yaw)

        beam_count = int(round(
            (self.scan_max_angle - self.scan_min_angle) / self.scan_increment
        )) + 1
        ranges = [float("inf")] * beam_count

        for obstacle in obstacles:
            dx = obstacle["x"] - robot_x
            dy = obstacle["y"] - robot_y
            bx = c * dx + s * dy
            by = -s * dx + c * dy
            radius = obstacle["radius"]
            d2 = bx * bx + by * by
            if d2 <= radius * radius:
                for i in range(beam_count):
                    ranges[i] = self.scan_min_range
                continue

            center_angle = math.atan2(by, bx)
            distance = math.sqrt(d2)
            half_angle = math.asin(clamp(radius / distance, 0.0, 1.0))
            start = max(0, int(math.floor(
                (center_angle - half_angle - self.scan_min_angle)
                / self.scan_increment
            )))
            end = min(beam_count - 1, int(math.ceil(
                (center_angle + half_angle - self.scan_min_angle)
                / self.scan_increment
            )))
            if end < 0 or start >= beam_count:
                continue

            for index in range(start, end + 1):
                angle = self.scan_min_angle + index * self.scan_increment
                projection = bx * math.cos(angle) + by * math.sin(angle)
                perpendicular2 = d2 - projection * projection
                discriminant = radius * radius - perpendicular2
                if projection <= 0.0 or discriminant < 0.0:
                    continue
                hit = projection - math.sqrt(discriminant)
                if self.scan_min_range <= hit <= self.scan_max_range:
                    ranges[index] = min(ranges[index], hit)

        scan = LaserScan()
        scan.header.stamp = rospy.Time.now()
        scan.header.frame_id = self.robot_frame
        scan.angle_min = self.scan_min_angle
        scan.angle_max = self.scan_max_angle
        scan.angle_increment = self.scan_increment
        scan.time_increment = 0.0
        scan.scan_time = 1.0 / max(1.0, self.scan_rate_hz)
        scan.range_min = self.scan_min_range
        scan.range_max = self.scan_max_range
        scan.ranges = ranges
        self.scan_pub.publish(scan)

    def action(self, name: str) -> None:
        if name == "cancel_goal":
            self.action_client.cancel_all_goals()
            with self.state.lock:
                self.state.goal = None
        elif name == "clear_costmaps":
            rospy.wait_for_service(self.clear_costmaps_service, timeout=2.0)
            rospy.ServiceProxy(self.clear_costmaps_service, Empty)()
        elif name == "reset_sim":
            self.action_client.cancel_all_goals()
            rospy.wait_for_service(self.reset_sim_service, timeout=2.0)
            rospy.ServiceProxy(self.reset_sim_service, Empty)()
            with self.state.lock:
                self.state.goal = None
                self.state.trail.clear()
                self.state.cmd_history.clear()
        elif name == "clear_obstacles":
            with self.state.lock:
                self.state.obstacles.clear()
        elif name == "clear_trail":
            with self.state.lock:
                self.state.trail.clear()
        else:
            raise ValueError("未知操作")

    @staticmethod
    def age(now: float, received: float) -> float:
        return 1.0e9 if received <= 0.0 else max(0.0, now - received)

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        with self.state.lock:
            if self.state.map_data is None:
                raise RuntimeError(f"尚未收到 {self.costmap_topic}")
            map_frame = self.state.map_frame
            map_width = self.state.map_width
            map_height = self.state.map_height
            map_resolution = self.state.map_resolution
            map_origin_x = self.state.map_origin_x
            map_origin_y = self.state.map_origin_y
            map_data = list(self.state.map_data)
            map_rx = self.state.map_rx
            map_updates = self.state.map_update_count
            global_plan = self.state.global_plan
            local_plan = self.state.local_plan
            odom = self.state.odom
            odom_rx = self.state.odom_rx
            cmd = self.state.cmd
            cmd_rx = self.state.cmd_rx
            trail = list(self.state.trail)
            cmd_history = list(self.state.cmd_history)
            goal = None if self.state.goal is None else dict(self.state.goal)
            obstacles = [dict(item) for item in self.state.obstacles]

        robot = self.robot_in_frame(odom, map_frame)
        global_points = self.path_points(global_plan, map_frame)
        local_points = self.path_points(local_plan, map_frame)

        trail_tfm = self.lookup_2d(map_frame, odom.header.frame_id or "odom") if odom else None
        if trail_tfm is not None:
            trail = [self.apply_transform(point, trail_tfm) for point in trail]
        else:
            trail = []

        encoded = bytes(
            0 if value < 0 else min(100, max(0, int(value))) + 1
            for value in map_data
        )

        recent_signs: List[int] = []
        for stamp, angular in cmd_history:
            if now - stamp > 10.0 or abs(angular) < 0.02:
                continue
            recent_signs.append(1 if angular > 0.0 else -1)
        cmd_changes = sum(
            1 for a, b in zip(recent_signs, recent_signs[1:]) if a != b
        )

        if odom is None:
            odom_values = {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
        else:
            twist = odom.twist.twist
            odom_values = {
                "linear_x": float(twist.linear.x),
                "linear_y": float(twist.linear.y),
                "angular_z": float(twist.angular.z),
            }

        return {
            "ok": True,
            "map": {
                "frame": map_frame,
                "width": map_width,
                "height": map_height,
                "resolution": map_resolution,
                "origin_x": map_origin_x,
                "origin_y": map_origin_y,
                "data_b64": base64.b64encode(encoded).decode("ascii"),
            },
            "robot": robot,
            "odom": odom_values,
            "cmd": {
                "linear_x": 0.0 if cmd is None else float(cmd.linear.x),
                "angular_z": 0.0 if cmd is None else float(cmd.angular.z),
            },
            "global_plan": global_points,
            "local_plan": local_points,
            "trail": trail,
            "goal": goal,
            "obstacles": obstacles,
            "ages": {
                "map": self.age(now, map_rx),
                "odom": self.age(now, odom_rx),
                "cmd": self.age(now, cmd_rx),
            },
            "metrics": {
                "map_updates": map_updates,
                "cmd_sign_changes": cmd_changes,
                "local_plan_turn_changes": turn_sign_changes(local_points),
                "global_plan_turn_changes": turn_sign_changes(global_points),
                "global_length_m": path_length(global_points),
                "local_length_m": path_length(local_points),
            },
        }

    @staticmethod
    def json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
        length = int(handler.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1024 * 1024:
            raise ValueError("请求体为空或过大")
        body = handler.rfile.read(length)
        value = json.loads(body.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON请求必须是对象")
        return value

    def start_http(self) -> ThreadingHTTPServer:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                rospy.logdebug(fmt, *args)

            def reply(self, status: int, value: Any, content_type: str = "application/json; charset=utf-8") -> None:
                if isinstance(value, (dict, list)):
                    body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                elif isinstance(value, bytes):
                    body = value
                else:
                    body = str(value).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                try:
                    if path in ("/", "/index.html"):
                        self.reply(200, HTML_PAGE, "text/html; charset=utf-8")
                    elif path == "/api/state":
                        self.reply(200, owner.snapshot())
                    elif path == "/api/params":
                        self.reply(200, {"ok": True, "params": owner.get_params()})
                    elif path == "/health":
                        self.reply(200, "ok\n", "text/plain; charset=utf-8")
                    else:
                        self.reply(404, {"ok": False, "error": "not found"})
                except Exception as exc:
                    self.reply(503, {"ok": False, "error": str(exc)})

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0]
                try:
                    request = owner.json_body(self)
                    if path == "/api/params":
                        result = owner.set_params(request)
                        self.reply(200, {"ok": True, "params": result})
                    elif path == "/api/goal":
                        owner.send_goal(
                            float(request["x"]),
                            float(request["y"]),
                            None if "yaw" not in request else float(request["yaw"]),
                        )
                        self.reply(200, {"ok": True})
                    elif path == "/api/obstacle":
                        owner.obstacle_action(request)
                        self.reply(200, {"ok": True})
                    elif path == "/api/action":
                        owner.action(str(request.get("action", "")))
                        self.reply(200, {"ok": True})
                    else:
                        self.reply(404, {"ok": False, "error": "not found"})
                except Exception as exc:
                    self.reply(400, {"ok": False, "error": str(exc)})

        server = ThreadingHTTPServer((self.host, self.port), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server


def main() -> None:
    rospy.init_node("dwa_web_tuner")
    node = DwaWebTuner()
    server = node.start_http()
    try:
        rospy.spin()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
