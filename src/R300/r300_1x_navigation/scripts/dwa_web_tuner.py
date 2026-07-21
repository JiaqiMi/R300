#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R300 visual-navigation DWA web simulation and tuning console.

Important design rule:
  The simulator does NOT use a separate DWA configuration.  move_base loads
  subject1_dwa_vision.yaml and subject1_local_costmap_vision.yaml, exactly the
  same files used by the visual-obstacle navigation launch.  This node only
  provides a browser UI, synthetic vision obstacles, multi-waypoint execution,
  runtime dynamic-reconfigure access, and configuration export/import.
"""

import base64
import json
import math
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse


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
from actionlib_msgs.msg import GoalStatus  # noqa: E402
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


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return json.dumps(str(value), ensure_ascii=False)
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def yaml_dump(value: Any, indent: int = 0) -> str:
    pad = " " * indent
    lines: List[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(yaml_dump(item, indent + 2))
            else:
                lines.append(f"{pad}{key}: {yaml_scalar(item)}")
    elif isinstance(value, list):
        if not value:
            return f"{pad}[]"
        for item in value:
            if isinstance(item, dict):
                first = True
                for key, sub in item.items():
                    prefix = "- " if first else "  "
                    if isinstance(sub, (dict, list)):
                        lines.append(f"{pad}{prefix}{key}:")
                        lines.append(yaml_dump(sub, indent + 4))
                    else:
                        lines.append(f"{pad}{prefix}{key}: {yaml_scalar(sub)}")
                    first = False
            elif isinstance(item, list):
                lines.append(f"{pad}-")
                lines.append(yaml_dump(item, indent + 2))
            else:
                lines.append(f"{pad}- {yaml_scalar(item)}")
    else:
        lines.append(f"{pad}{yaml_scalar(value)}")
    return "\n".join(lines)


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
        self.local_plan: Optional[Path] = None
        self.global_plan_rx = 0.0
        self.local_plan_rx = 0.0

        self.odom: Optional[Odometry] = None
        self.odom_rx = 0.0
        self.cmd: Optional[Twist] = None
        self.cmd_rx = 0.0

        self.trail: Deque[Point] = deque(maxlen=12000)
        self.cmd_history: Deque[Tuple[float, float]] = deque(maxlen=3000)

        self.goal: Optional[Dict[str, Any]] = None
        self.obstacles: List[Dict[str, Any]] = []
        self.next_obstacle_id = 1

        self.waypoints: List[Dict[str, Any]] = []
        self.next_waypoint_id = 1
        self.route_running = False
        self.route_loop = False
        self.route_index = -1
        self.route_run_id = 0
        self.route_message = "未启动"


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>R300 视觉导航 DWA 仿真调参</title>
<style>
:root{color-scheme:dark;--bg:#0f141a;--panel:#18212a;--panel2:#202b35;--border:#344451;--text:#eef4f8;--muted:#9fb0bd;--accent:#4ea1ff;--ok:#43d17d;--warn:#ffb547;--bad:#ff6262}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Arial,"Microsoft YaHei",sans-serif}header{padding:11px 16px;border-bottom:1px solid var(--border);background:#141c23;position:sticky;top:0;z-index:3}h1{font-size:20px;margin:0 0 5px}h2{font-size:16px;margin:0 0 10px}h3{font-size:14px;margin:14px 0 8px;color:#d9e6ee}small,.muted{color:var(--muted)}main{padding:12px;display:grid;grid-template-columns:minmax(540px,1fr) 470px;gap:12px}.panel{border:1px solid var(--border);background:var(--panel);border-radius:10px;padding:11px}.metrics{display:grid;grid-template-columns:repeat(5,minmax(95px,1fr));gap:8px;margin-bottom:10px}.metric{background:var(--panel2);border:1px solid var(--border);border-radius:8px;padding:8px}.metric b{display:block;font-size:17px;margin-top:4px}.canvasWrap{position:relative;width:100%;aspect-ratio:1.25;background:#111;border:1px solid var(--border);border-radius:8px;overflow:hidden}canvas{width:100%;height:100%;display:block;cursor:crosshair}.toolbar{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}.toolbar label{display:flex;align-items:center;gap:5px}.toolbar select,.toolbar input,button,.params input,.params select,textarea{background:#111820;color:var(--text);border:1px solid #465b6b;border-radius:6px;padding:7px 8px}button{cursor:pointer}button.primary{background:#2368a2;border-color:#3a86c1}button.danger{background:#7b2929;border-color:#a94040}button.good{background:#1f6b43;border-color:#369466}.hint{padding:8px;border-left:3px solid var(--accent);background:#172532;color:#cfe5f7;font-size:13px;margin-top:8px}.params{display:grid;grid-template-columns:1fr 112px;gap:6px 8px;align-items:center}.params label{font-size:13px;color:#dce6ec}.params input,.params select{width:100%}.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:12px;color:var(--muted);margin-top:8px}.sw{width:18px;height:4px;display:inline-block;margin-right:4px;vertical-align:middle}.row{display:flex;gap:8px;flex-wrap:wrap}.status{font-size:13px;color:var(--muted)}#message{min-height:20px;margin-top:8px;font-size:13px}.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}.tab{padding:6px 10px}.tab.active{background:#2368a2}.tabPage{display:none}.tabPage.active{display:block}.wpList{max-height:210px;overflow:auto;border:1px solid var(--border);border-radius:7px;margin-top:8px}.wp{display:grid;grid-template-columns:34px 1fr auto;gap:7px;padding:6px 8px;border-bottom:1px solid #2e3b45;align-items:center;font-size:12px}.wp:last-child{border:0}.wp.active{background:#213d53}.wp.done{color:#74df9e}.wp.failed{color:#ff7f7f}.wp button{padding:3px 6px}.two{display:grid;grid-template-columns:1fr 1fr;gap:8px}hr{border:0;border-top:1px solid var(--border);margin:13px 0}@media(max-width:1100px){main{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<header><h1>R300 视觉导航 DWA 仿真调参台</h1><div class="status" id="status">等待 ROS 数据……</div></header>
<main>
<section>
  <div class="metrics">
    <div class="metric"><span class="muted">线速度指令</span><b id="mV">--</b></div>
    <div class="metric"><span class="muted">实际线速度</span><b id="mOdomV">--</b></div>
    <div class="metric"><span class="muted">角速度指令</span><b id="mW">--</b></div>
    <div class="metric"><span class="muted">10秒换向</span><b id="mFlip">--</b></div>
    <div class="metric"><span class="muted">多点进度</span><b id="mRoute">--</b></div>
  </div>
  <div class="panel">
    <div class="canvasWrap"><canvas id="map" width="1100" height="850"></canvas></div>
    <div class="toolbar">
      <label>点击模式<select id="clickMode"><option value="goal">单目标</option><option value="waypoint" selected>添加多点航点</option><option value="add">添加障碍</option><option value="remove">删除最近障碍</option></select></label>
      <label>障碍半径<input id="obsRadius" type="number" min="0.1" max="3" step="0.1" value="0.35" style="width:75px"></label>
      <label>视野半径<select id="viewRadius"><option>5</option><option selected>8</option><option>12</option><option>20</option></select>m</label>
      <label><input id="showCost" type="checkbox" checked>costmap</label>
      <label><input id="showTrail" type="checkbox" checked>轨迹</label>
      <label><input id="showGlobal" type="checkbox" checked>全局路径</label>
      <label><input id="showLocal" type="checkbox" checked>局部路径</label>
      <label><input id="showWaypoints" type="checkbox" checked>多点路线</label>
    </div>
    <div class="row">
      <button id="cancelGoal">取消当前目标</button><button id="clearMap">清除costmap</button><button id="clearObs">清空障碍</button><button id="clearTrail">清空轨迹</button><button id="resetSim" class="danger">复位仿真</button>
    </div>
    <div class="legend"><span><i class="sw" style="background:#ffd84d"></i>全局路径</span><span><i class="sw" style="background:#35e27c"></i>DWA局部路径</span><span><i class="sw" style="background:#42a5ff"></i>实际轨迹</span><span><i class="sw" style="background:#ff4b4b"></i>人工视觉障碍</span><span><i class="sw" style="background:#d778ff"></i>单目标</span><span><i class="sw" style="background:#ff9f43"></i>多点路线</span></div>
    <div class="hint">本仿真直接加载实车视觉导航使用的 <b>subject1_dwa_vision.yaml</b> 与 <b>subject1_local_costmap_vision.yaml</b>，没有独立的“仿真DWA参数文件”。网页修改的是当前 move_base 运行参数。</div>
  </div>
</section>
<aside class="panel">
  <div class="tabs"><button class="tab active" data-tab="paramsPage">参数</button><button class="tab" data-tab="routePage">多点目标</button><button class="tab" data-tab="exportPage">导入导出</button><button class="tab" data-tab="statePage">状态</button></div>

  <div id="paramsPage" class="tabPage active">
    <h2>视觉导航运行参数</h2>
    <div id="paramSections"></div>
    <div class="hint">仅显示当前 ROS Noetic 动态服务器真正支持的运行时参数。<code>latch_xy_goal_tolerance</code> 是启动时参数，已移到 <code>/move_base/latch_xy_goal_tolerance</code>，修改后需要重启导航；<code>penalize_negative_x</code> 不是 DWAPlannerROS 参数。</div>
    <div class="row" style="margin-top:10px"><button id="loadParams">读取当前参数</button><button id="restoreParams">恢复启动值</button><button id="applyParams" class="primary">应用全部参数</button></div>
  </div>

  <div id="routePage" class="tabPage">
    <h2>多点目标队列</h2>
    <div class="row"><label><input type="checkbox" id="routeLoop">循环执行</label><button id="startRoute" class="good">开始</button><button id="pauseRoute">暂停</button><button id="skipRoute">跳过当前</button><button id="clearRoute" class="danger">清空</button></div>
    <div class="status" id="routeStatus" style="margin-top:8px"></div>
    <div id="waypointList" class="wpList"></div>
    <div class="hint">选择“添加多点航点”后依次点击地图。开始后，当前航点成功到达才发送下一个；失败时自动暂停，便于观察DWA和障碍状态。</div>
  </div>

  <div id="exportPage" class="tabPage">
    <h2>配置导入导出</h2>
    <div class="row"><button id="exportYaml" class="primary">导出 YAML</button><button id="exportJson">导出 JSON</button><label style="display:inline-block"><input id="importFile" type="file" accept="application/json,.json" style="display:none"><button id="importJson" type="button">导入 JSON</button></label></div>
    <div class="hint">导出内容包括当前DWA、move_base、局部costmap、膨胀层参数，以及多点目标和人工障碍。导入JSON默认只恢复路线与障碍；勾选下方选项后同时应用参数。</div>
    <label style="display:block;margin-top:10px"><input id="importApplyParams" type="checkbox">导入时同时应用参数</label>
  </div>

  <div id="statePage" class="tabPage"><h2>当前状态</h2><div id="details" class="status"></div></div>
  <div id="message" aria-live="polite"></div>
</aside>
</main>
<script>
const canvas=document.getElementById('map'),ctx=canvas.getContext('2d');let snapshot=null,currentView=null;
const groups=[
 {id:'dwa',title:'DWA 速度与动力学',defs:[
  ['max_vel_x','最大前进速度 m/s','number',.01],['min_vel_x','最小前进速度 m/s','number',.01],['max_vel_trans','最大平移速度 m/s','number',.01],['min_vel_trans','最小平移速度 m/s','number',.01],['max_vel_theta','最大角速度 rad/s','number',.01],['min_vel_theta','最小角速度 rad/s','number',.01],['acc_lim_x','前进加速度 m/s²','number',.05],['acc_lim_y','横向加速度 m/s²','number',.05],['acc_lim_trans','平移加速度 m/s²','number',.05],['acc_lim_theta','角加速度 rad/s²','number',.05]
 ]},
 {id:'dwa',title:'DWA 到点、预测与采样',defs:[
  ['xy_goal_tolerance','位置容差 m','number',.05],['yaw_goal_tolerance','航向容差 rad','number',.05],['trans_stopped_vel','平移停止阈值 m/s','number',.01],['theta_stopped_vel','角速度停止阈值 rad/s','number',.01],['sim_time','轨迹预测时间 s','number',.1],['sim_granularity','线性离散间隔 m','number',.01],['angular_sim_granularity','角度离散间隔 rad','number',.005],['vx_samples','线速度采样数','number',1],['vy_samples','横向采样数','number',1],['vth_samples','角速度采样数','number',1]
 ]},
 {id:'dwa',title:'DWA 评分与振荡',defs:[
  ['path_distance_bias','路径权重','number',1],['goal_distance_bias','目标权重','number',1],['occdist_scale','障碍权重','number',.01],['forward_point_distance','前视点距离 m','number',.05],['stop_time_buffer','停车缓冲 s','number',.05],['scaling_speed','轮廓缩放起始速度','number',.05],['max_scaling_factor','最大轮廓缩放比例','number',.05],['oscillation_reset_dist','振荡复位距离 m','number',.05],['oscillation_reset_angle','振荡复位角度 rad','number',.05],['prune_plan','裁剪全局路径','bool',1],['twirling_scale','旋转惩罚','number',.05]
 ]},
 {id:'move_base',title:'MoveBase',defs:[
  ['controller_frequency','控制频率 Hz','number',.5],['planner_frequency','全局规划频率 Hz','number',.1],['planner_patience','规划耐心 s','number',.5],['controller_patience','控制耐心 s','number',.5],['oscillation_timeout','振荡超时 s','number',.5],['oscillation_distance','振荡检测距离 m','number',.05]
 ]},
 {id:'local_costmap',title:'局部 Costmap',defs:[
  ['update_frequency','更新频率 Hz','number',.5],['publish_frequency','发布频率 Hz','number',.5],['transform_tolerance','TF容差 s','number',.05],['width','宽度 m','number',.5],['height','高度 m','number',.5],['resolution','分辨率 m/格','number',.01]
 ]},
 {id:'inflation',title:'局部膨胀层',defs:[
  ['inflation_radius','膨胀半径 m','number',.05],['cost_scaling_factor','代价衰减系数','number',.1],['enabled','启用膨胀层','bool',1]
 ]}
];
const sectionBox=document.getElementById('paramSections');
for(const g of groups){const h=document.createElement('h3');h.textContent=g.title;sectionBox.appendChild(h);const box=document.createElement('div');box.className='params';for(const [key,label,type,step] of g.defs){const l=document.createElement('label');l.textContent=label;let i;if(type==='bool'){i=document.createElement('select');i.innerHTML='<option value="true">true</option><option value="false">false</option>'}else{i=document.createElement('input');i.type='number';i.step=step}i.id=`p_${g.id}_${key}`;i.dataset.group=g.id;i.dataset.key=key;i.dataset.type=type;box.append(l,i)}sectionBox.appendChild(box)}
function msg(text,cls='ok'){const e=document.getElementById('message');e.className=cls;e.textContent=text}
async function api(path,body=null){const opt=body===null?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)};const r=await fetch(path,opt);const t=await r.text();let d;try{d=JSON.parse(t)}catch(_){d={ok:false,error:t}}if(!r.ok||d.ok===false)throw new Error(d.error||('HTTP '+r.status));return d}
function collectParams(){const result={};for(const e of document.querySelectorAll('[data-group][data-key]')){if(!result[e.dataset.group])result[e.dataset.group]={};if(e.dataset.type==='bool')result[e.dataset.group][e.dataset.key]=(e.value==='true');else if(e.value!=='')result[e.dataset.group][e.dataset.key]=Number(e.value)}return result}
function fillParams(params){for(const e of document.querySelectorAll('[data-group][data-key]')){const v=params?.[e.dataset.group]?.[e.dataset.key];if(v!==undefined)e.value=(e.dataset.type==='bool'?(v?'true':'false'):v)}}
async function loadParams(){try{const d=await api('/api/params');fillParams(d.params);msg('已读取当前视觉导航运行参数')}catch(e){msg(e.message,'bad')}}
async function applyParams(){try{const d=await api('/api/params',collectParams());fillParams(d.params);const w=d.warnings||[];msg(w.length?('已应用支持的参数；'+w.join('；')):'参数已应用到当前 move_base',w.length?'warn':'ok')}catch(e){msg(e.message,'bad')}}
async function restoreParams(){try{const d=await api('/api/action',{action:'restore_params'});fillParams(d.params);msg('已恢复本次启动时从视觉导航YAML加载的参数','warn')}catch(e){msg(e.message,'bad')}}
function mapBytes(s){const raw=atob(s.map.data_b64),a=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)a[i]=raw.charCodeAt(i);return a}
function viewFor(s){const r=Number(document.getElementById('viewRadius').value);const c=s.robot||{x:0,y:0};return{minX:c.x-r,maxX:c.x+r,minY:c.y-r,maxY:c.y+r}}
function projection(v){const W=canvas.width,H=canvas.height,scale=Math.min(W/(v.maxX-v.minX),H/(v.maxY-v.minY));const usedW=(v.maxX-v.minX)*scale,usedH=(v.maxY-v.minY)*scale;const left=(W-usedW)/2,top=(H-usedH)/2;return{scale,left,top,w2c:(x,y)=>[left+(x-v.minX)*scale,top+(v.maxY-y)*scale],c2w:(x,y)=>[v.minX+(x-left)/scale,v.maxY-(y-top)/scale]}}
function line(points,p,color,width=3,dash=[]){if(!points||points.length<2)return;ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();let q=p.w2c(points[0][0],points[0][1]);ctx.moveTo(...q);for(const pt of points.slice(1)){q=p.w2c(pt[0],pt[1]);ctx.lineTo(...q)}ctx.stroke();ctx.restore()}
function drawGrid(v,p){ctx.save();ctx.strokeStyle='rgba(150,170,185,.18)';ctx.lineWidth=1;for(let x=Math.ceil(v.minX);x<=v.maxX;x++){const a=p.w2c(x,v.minY),b=p.w2c(x,v.maxY);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}for(let y=Math.ceil(v.minY);y<=v.maxY;y++){const a=p.w2c(v.minX,y),b=p.w2c(v.maxX,y);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}ctx.restore()}
function drawCost(s,v,p){if(!document.getElementById('showCost').checked)return;const m=s.map,data=mapBytes(s),res=m.resolution;if(!res)return;const x0=Math.max(0,Math.floor((v.minX-m.origin_x)/res)),x1=Math.min(m.width-1,Math.ceil((v.maxX-m.origin_x)/res));const y0=Math.max(0,Math.floor((v.minY-m.origin_y)/res)),y1=Math.min(m.height-1,Math.ceil((v.maxY-m.origin_y)/res));const px=Math.max(1,res*p.scale+0.5);for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){const enc=data[y*m.width+x];if(enc===1)continue;const val=enc===0?-1:enc-1;if(val<0)ctx.fillStyle='rgba(70,76,82,.25)';else if(val>=90)ctx.fillStyle='rgba(5,5,5,.96)';else ctx.fillStyle=`rgba(110,110,110,${.12+.55*val/100})`;const w=m.origin_x+x*res,wy=m.origin_y+(y+1)*res,c=p.w2c(w,wy);ctx.fillRect(c[0],c[1],px,px)}}
function circle(x,y,r,p,color,fill=false){const c=p.w2c(x,y);ctx.beginPath();ctx.arc(c[0],c[1],Math.max(3,r*p.scale),0,Math.PI*2);ctx.strokeStyle=color;ctx.lineWidth=3;if(fill){ctx.fillStyle=color;ctx.globalAlpha=.22;ctx.fill();ctx.globalAlpha=1}ctx.stroke()}
function drawRobot(robot,p){if(!robot)return;const c=p.w2c(robot.x,robot.y),L=.9*p.scale,W=.7*p.scale;ctx.save();ctx.translate(c[0],c[1]);ctx.rotate(-robot.yaw);ctx.strokeStyle='#4ea1ff';ctx.fillStyle='rgba(78,161,255,.20)';ctx.lineWidth=3;ctx.beginPath();ctx.rect(-L/2,-W/2,L,W);ctx.fill();ctx.stroke();ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(L*.7,0);ctx.stroke();ctx.restore()}
function drawWaypoints(s,p){if(!document.getElementById('showWaypoints').checked||!s.route?.waypoints)return;const pts=s.route.waypoints.map(w=>[w.x,w.y]);line(pts,p,'#ff9f43',2,[7,5]);for(let i=0;i<s.route.waypoints.length;i++){const w=s.route.waypoints[i],c=p.w2c(w.x,w.y);let color='#ff9f43';if(w.status==='active')color='#4ea1ff';else if(w.status==='done')color='#43d17d';else if(w.status==='failed')color='#ff6262';ctx.beginPath();ctx.arc(c[0],c[1],10,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.fillStyle='#10161c';ctx.font='bold 11px Arial';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(i+1),c[0],c[1])}}
function renderWaypointList(route){const box=document.getElementById('waypointList');box.innerHTML='';for(let i=0;i<route.waypoints.length;i++){const w=route.waypoints[i],row=document.createElement('div');row.className='wp '+(w.status||'');row.innerHTML=`<b>${i+1}</b><span>x=${w.x.toFixed(2)}, y=${w.y.toFixed(2)}<br><span class="muted">${w.status||'pending'}</span></span><button data-remove-wp="${w.id}">删除</button>`;box.appendChild(row)}if(!route.waypoints.length)box.innerHTML='<div class="wp"><span></span><span class="muted">尚未添加航点</span><span></span></div>';document.getElementById('routeStatus').textContent=`${route.message}｜${route.running?'运行中':'未运行'}｜当前 ${route.current_index>=0?route.current_index+1:'--'} / ${route.waypoints.length}`;document.getElementById('mRoute').textContent=`${route.current_index>=0?route.current_index+1:0}/${route.waypoints.length}`;for(const b of box.querySelectorAll('[data-remove-wp]'))b.onclick=async()=>{try{await api('/api/route',{action:'remove',id:Number(b.dataset.removeWp)});msg('航点已删除')}catch(e){msg(e.message,'bad')}}}
function render(s){snapshot=s;const v=viewFor(s),p=projection(v);currentView={v,p};ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#e7ebee';ctx.fillRect(0,0,canvas.width,canvas.height);drawCost(s,v,p);drawGrid(v,p);if(document.getElementById('showTrail').checked)line(s.trail,p,'#318de4',2);if(document.getElementById('showGlobal').checked)line(s.global_plan,p,'#e5b900',4);if(document.getElementById('showLocal').checked)line(s.local_plan,p,'#18c967',4);drawWaypoints(s,p);for(const o of s.obstacles)circle(o.x,o.y,o.radius,p,'#ef3e3e',true);if(s.goal){circle(s.goal.x,s.goal.y,.18,p,'#bc5cff',false)}drawRobot(s.robot,p);document.getElementById('mV').textContent=s.cmd.linear_x.toFixed(3)+' m/s';document.getElementById('mOdomV').textContent=s.odom.linear_x.toFixed(3)+' m/s';document.getElementById('mW').textContent=s.cmd.angular_z.toFixed(3)+' rad/s';document.getElementById('mFlip').textContent=s.metrics.cmd_sign_changes;renderWaypointList(s.route);document.getElementById('details').innerHTML=`机器人: x=${s.robot?s.robot.x.toFixed(2):'--'}, y=${s.robot?s.robot.y.toFixed(2):'--'}, yaw=${s.robot?(s.robot.yaw*180/Math.PI).toFixed(1):'--'}°<br>里程计: vx=${s.odom.linear_x.toFixed(3)}, vy=${s.odom.linear_y.toFixed(3)}, wz=${s.odom.angular_z.toFixed(3)}<br>全局路径: ${s.global_plan.length}点 / ${s.metrics.global_length_m.toFixed(2)}m<br>局部路径: ${s.local_plan.length}点 / ${s.metrics.local_length_m.toFixed(2)}m<br>人工障碍: ${s.obstacles.length}个；costmap更新: ${s.metrics.map_updates}<br>数据年龄: odom ${s.ages.odom.toFixed(2)}s，costmap ${s.ages.map.toFixed(2)}s`;document.getElementById('status').textContent=`ROS在线｜frame=${s.map.frame}｜视觉导航同源参数｜${new Date().toLocaleTimeString()}`}
async function poll(){try{render(await api('/api/state'))}catch(e){document.getElementById('status').textContent='等待数据：'+e.message}finally{setTimeout(poll,350)}}
canvas.addEventListener('click',async ev=>{if(!currentView)return;const rect=canvas.getBoundingClientRect(),cx=(ev.clientX-rect.left)*canvas.width/rect.width,cy=(ev.clientY-rect.top)*canvas.height/rect.height,[x,y]=currentView.p.c2w(cx,cy),mode=document.getElementById('clickMode').value;try{if(mode==='goal')await api('/api/goal',{x,y});else if(mode==='waypoint')await api('/api/route',{action:'add',x,y});else if(mode==='add')await api('/api/obstacle',{action:'add',x,y,radius:Number(document.getElementById('obsRadius').value)});else await api('/api/obstacle',{action:'remove_nearest',x,y});msg(`操作完成：(${x.toFixed(2)}, ${y.toFixed(2)})`)}catch(e){msg(e.message,'bad')}});
for(const [id,action] of [['cancelGoal','cancel_goal'],['clearMap','clear_costmaps'],['clearObs','clear_obstacles'],['clearTrail','clear_trail'],['resetSim','reset_sim']])document.getElementById(id).addEventListener('click',async()=>{try{await api('/api/action',{action});msg('操作完成：'+action)}catch(e){msg(e.message,'bad')}});
document.getElementById('startRoute').onclick=async()=>{try{await api('/api/route',{action:'start',loop:document.getElementById('routeLoop').checked});msg('多点路线已开始')}catch(e){msg(e.message,'bad')}};
document.getElementById('pauseRoute').onclick=async()=>{try{await api('/api/route',{action:'pause'});msg('多点路线已暂停','warn')}catch(e){msg(e.message,'bad')}};
document.getElementById('skipRoute').onclick=async()=>{try{await api('/api/route',{action:'skip'});msg('已跳过当前航点','warn')}catch(e){msg(e.message,'bad')}};
document.getElementById('clearRoute').onclick=async()=>{try{await api('/api/route',{action:'clear'});msg('多点路线已清空','warn')}catch(e){msg(e.message,'bad')}};
document.getElementById('loadParams').onclick=loadParams;document.getElementById('applyParams').onclick=applyParams;document.getElementById('restoreParams').onclick=restoreParams;
function download(url){const a=document.createElement('a');a.href=url;a.click()}
document.getElementById('exportYaml').onclick=()=>download('/api/export?format=yaml');document.getElementById('exportJson').onclick=()=>download('/api/export?format=json');document.getElementById('importJson').onclick=()=>document.getElementById('importFile').click();document.getElementById('importFile').onchange=async ev=>{const f=ev.target.files[0];if(!f)return;try{const data=JSON.parse(await f.text());await api('/api/import',{data,apply_params:document.getElementById('importApplyParams').checked});msg('JSON配置已导入')}catch(e){msg('导入失败：'+e.message,'bad')}ev.target.value=''};
for(const b of document.querySelectorAll('.tab'))b.onclick=()=>{for(const x of document.querySelectorAll('.tab'))x.classList.remove('active');for(const x of document.querySelectorAll('.tabPage'))x.classList.remove('active');b.classList.add('active');document.getElementById(b.dataset.tab).classList.add('active')};
loadParams();poll();
</script>
</body></html>"""


class DwaWebTuner:
    PARAM_SPECS: Dict[str, Dict[str, Tuple[type, float, float]]] = {
        "dwa": {
            "max_vel_x": (float, -1.0, 5.0),
            "min_vel_x": (float, -1.0, 5.0),
            "max_vel_y": (float, -1.0, 1.0),
            "min_vel_y": (float, -1.0, 1.0),
            "max_vel_trans": (float, 0.0, 5.0),
            "min_vel_trans": (float, 0.0, 2.0),
            "max_vel_theta": (float, 0.0, 3.0),
            "min_vel_theta": (float, 0.0, 2.0),
            "acc_lim_x": (float, 0.0, 20.0),
            "acc_lim_y": (float, 0.0, 20.0),
            "acc_lim_trans": (float, 0.0, 20.0),
            "acc_lim_theta": (float, 0.0, 20.0),
            "xy_goal_tolerance": (float, 0.0, 20.0),
            "yaw_goal_tolerance": (float, 0.0, 6.4),
            "trans_stopped_vel": (float, 0.0, 2.0),
            "theta_stopped_vel": (float, 0.0, 2.0),
            "sim_time": (float, 0.1, 10.0),
            "sim_granularity": (float, 0.005, 2.0),
            "angular_sim_granularity": (float, 0.001, 1.0),
            "vx_samples": (int, 1, 200),
            "vy_samples": (int, 1, 50),
            "vth_samples": (int, 1, 300),
            "path_distance_bias": (float, 0.0, 300.0),
            "goal_distance_bias": (float, 0.0, 300.0),
            "occdist_scale": (float, 0.0, 20.0),
            "forward_point_distance": (float, 0.0, 10.0),
            "stop_time_buffer": (float, 0.0, 10.0),
            "scaling_speed": (float, 0.0, 5.0),
            "max_scaling_factor": (float, 0.0, 5.0),
            "oscillation_reset_dist": (float, 0.0, 10.0),
            "oscillation_reset_angle": (float, 0.0, 6.4),
            "prune_plan": (bool, 0.0, 1.0),
            "twirling_scale": (float, 0.0, 100.0),
        },
        "move_base": {
            "controller_frequency": (float, 0.1, 100.0),
            "planner_frequency": (float, 0.0, 100.0),
            "planner_patience": (float, 0.0, 300.0),
            "controller_patience": (float, 0.0, 300.0),
            "oscillation_timeout": (float, 0.0, 300.0),
            "oscillation_distance": (float, 0.0, 20.0),
        },
        "local_costmap": {
            "update_frequency": (float, 0.0, 100.0),
            "publish_frequency": (float, 0.0, 100.0),
            "transform_tolerance": (float, 0.0, 10.0),
            "width": (float, 1.0, 200.0),
            "height": (float, 1.0, 200.0),
            "resolution": (float, 0.01, 2.0),
        },
        "inflation": {
            "inflation_radius": (float, 0.0, 20.0),
            "cost_scaling_factor": (float, 0.0, 100.0),
            "enabled": (bool, 0.0, 1.0),
        },
    }

    def __init__(self) -> None:
        self.state = SharedState()
        self.host = str(rospy.get_param("~host", "0.0.0.0"))
        self.port = int(rospy.get_param("~port", 8070))
        self.robot_frame = str(rospy.get_param("~robot_frame", "base_link"))
        self.goal_frame = str(rospy.get_param("~goal_frame", "map"))

        self.costmap_topic = str(rospy.get_param("~costmap_topic", "/move_base/local_costmap/costmap"))
        self.costmap_update_topic = str(rospy.get_param("~costmap_update_topic", "/move_base/local_costmap/costmap_updates"))
        self.global_plan_topic = str(rospy.get_param("~global_plan_topic", "/move_base/NavfnROS/plan"))
        self.local_plan_topic = str(rospy.get_param("~local_plan_topic", "/move_base/DWAPlannerROS/local_plan"))
        self.odom_topic = str(rospy.get_param("~odom_topic", "/one_x/odom"))
        self.cmd_topic = str(rospy.get_param("~cmd_topic", "/subject1/cmd_vel_raw"))

        self.dynamic_servers = {
            "dwa": str(rospy.get_param("~dwa_dynamic_server", "/move_base/DWAPlannerROS")),
            "move_base": str(rospy.get_param("~move_base_dynamic_server", "/move_base")),
            "local_costmap": str(rospy.get_param("~local_costmap_dynamic_server", "/move_base/local_costmap")),
            "inflation": str(rospy.get_param("~inflation_dynamic_server", "/move_base/local_costmap/inflation_layer")),
        }
        self.dynamic_clients: Dict[str, DynamicClient] = {}
        self.startup_params: Optional[Dict[str, Dict[str, Any]]] = None
        self.last_param_warnings: List[str] = []

        self.move_base_action = str(rospy.get_param("~move_base_action", "/move_base"))
        self.clear_costmaps_service = str(rospy.get_param("~clear_costmaps_service", "/move_base/clear_costmaps"))
        self.reset_sim_service = str(rospy.get_param("~reset_sim_service", "/sim/reset_pose"))

        self.enable_synthetic_scan = bool(rospy.get_param("~enable_synthetic_scan", True))
        self.scan_topic = str(rospy.get_param("~scan_topic", "/r300_vision/obstacle_scan"))
        self.scan_rate_hz = float(rospy.get_param("~scan_rate_hz", 10.0))
        self.scan_min_angle = math.radians(float(rospy.get_param("~scan_angle_min_deg", -27.0)))
        self.scan_max_angle = math.radians(float(rospy.get_param("~scan_angle_max_deg", 27.0)))
        self.scan_increment = math.radians(float(rospy.get_param("~scan_angle_increment_deg", 0.5)))
        self.scan_min_range = float(rospy.get_param("~scan_range_min_m", 0.2))
        self.scan_max_range = float(rospy.get_param("~scan_range_max_m", 10.5))

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(20.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.action_client = actionlib.SimpleActionClient(self.move_base_action, MoveBaseAction)

        rospy.Subscriber(self.costmap_topic, OccupancyGrid, self.costmap_cb, queue_size=2)
        rospy.Subscriber(self.costmap_update_topic, OccupancyGridUpdate, self.costmap_update_cb, queue_size=3)
        rospy.Subscriber(self.global_plan_topic, Path, self.global_plan_cb, queue_size=2)
        rospy.Subscriber(self.local_plan_topic, Path, self.local_plan_cb, queue_size=3)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.cmd_topic, Twist, self.cmd_cb, queue_size=1, tcp_nodelay=True)

        self.scan_pub = rospy.Publisher(self.scan_topic, LaserScan, queue_size=1)
        if self.enable_synthetic_scan:
            rospy.Timer(rospy.Duration(1.0 / max(1.0, self.scan_rate_hz)), self.publish_scan)

        rospy.logwarn(
            "DWA web tuner: same visual-nav config; http://%s:%d; synthetic_scan=%s",
            self.host, self.port, self.enable_synthetic_scan,
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
            if msg.x < 0 or msg.y < 0 or msg.x + msg.width > width or msg.y + msg.height > height:
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
            if not self.state.trail or math.hypot(x - self.state.trail[-1][0], y - self.state.trail[-1][1]) >= 0.015:
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
            tfm = self.tf_buffer.lookup_transform(target, source, rospy.Time(0), rospy.Duration(0.05))
            t = tfm.transform.translation
            return float(t.x), float(t.y), yaw_from_quaternion(tfm.transform.rotation)
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
        stride = max(1, int(math.ceil(len(msg.poses) / 1500.0)))
        return [
            self.apply_transform((float(p.pose.position.x), float(p.pose.position.y)), tfm)
            for p in msg.poses[::stride]
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

    def dynamic(self, group: str) -> DynamicClient:
        if group not in self.dynamic_clients:
            self.dynamic_clients[group] = DynamicClient(self.dynamic_servers[group], timeout=2.5)
        return self.dynamic_clients[group]

    @staticmethod
    def convert_param(key: str, value: Any, spec: Tuple[type, float, float]) -> Any:
        typ, lower, upper = spec
        if typ is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        try:
            converted = typ(value)
        except (TypeError, ValueError):
            raise ValueError(f"参数 {key} 不是有效数值")
        if converted < lower or converted > upper:
            raise ValueError(f"参数 {key} 超出允许范围 [{lower}, {upper}]")
        return converted

    def read_group(self, group: str) -> Dict[str, Any]:
        config = self.dynamic(group).get_configuration(timeout=2.5)
        return {key: config[key] for key in self.PARAM_SPECS[group] if key in config}

    def get_params(self, capture_startup: bool = True) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        errors: Dict[str, str] = {}
        for group in self.PARAM_SPECS:
            try:
                result[group] = self.read_group(group)
            except Exception as exc:
                result[group] = {}
                errors[group] = str(exc)
        if capture_startup and self.startup_params is None:
            self.startup_params = json.loads(json.dumps(result))
        if errors:
            result["_errors"] = errors  # type: ignore[assignment]
        return result

    def set_params(self, request: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        if not isinstance(request, dict):
            raise ValueError("参数请求格式错误")

        changed = False
        warnings: List[str] = []

        for group, group_values in request.items():
            if group not in self.PARAM_SPECS or not isinstance(group_values, dict):
                continue

            # Different ROS/navigation package builds may expose slightly
            # different dynamic_reconfigure fields.  Intersect the Web request
            # with the server's live schema instead of letting one unsupported
            # field reject the complete "apply all" request.
            current = self.dynamic(group).get_configuration(timeout=2.5)
            supported = set(current.keys())
            clean: Dict[str, Any] = {}

            for key, value in group_values.items():
                if key not in self.PARAM_SPECS[group]:
                    warnings.append(f"{group}.{key} 不是本页面支持的参数，已跳过")
                    continue
                if key not in supported:
                    warnings.append(f"{group}.{key} 不支持动态修改，已跳过")
                    continue
                clean[key] = self.convert_param(
                    key, value, self.PARAM_SPECS[group][key]
                )

            if (
                group == "dwa"
                and "max_vel_x" in clean
                and "max_vel_trans" not in clean
                and "max_vel_trans" in supported
            ):
                clean["max_vel_trans"] = max(0.0, float(clean["max_vel_x"]))

            if clean:
                self.dynamic(group).update_configuration(clean)
                changed = True

        self.last_param_warnings = warnings
        if not changed:
            if warnings:
                return self.get_params(capture_startup=False)
            raise ValueError("没有可应用的参数")
        return self.get_params(capture_startup=False)

    def restore_startup_params(self) -> Dict[str, Dict[str, Any]]:
        if self.startup_params is None:
            self.get_params(capture_startup=True)
        assert self.startup_params is not None
        return self.set_params(self.startup_params)

    def build_goal(self, x: float, y: float, yaw: Optional[float] = None) -> MoveBaseGoal:
        if yaw is None:
            with self.state.lock:
                odom = self.state.odom
            if odom is not None:
                yaw = math.atan2(y - float(odom.pose.pose.position.y), x - float(odom.pose.pose.position.x))
            else:
                yaw = 0.0
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
        return goal

    def ensure_action_server(self) -> None:
        if not self.action_client.wait_for_server(rospy.Duration(2.5)):
            raise RuntimeError("move_base action server尚未就绪")

    def send_goal(self, x: float, y: float, yaw: Optional[float]) -> None:
        self.ensure_action_server()
        self.pause_route(cancel=False, message="已切换为单目标")
        goal = self.build_goal(x, y, yaw)
        self.action_client.send_goal(goal)
        with self.state.lock:
            self.state.goal = {"x": x, "y": y, "yaw": yaw, "source": "single"}

    def add_waypoint(self, x: float, y: float, yaw: Optional[float] = None) -> None:
        with self.state.lock:
            self.state.waypoints.append({
                "id": self.state.next_waypoint_id,
                "x": float(x),
                "y": float(y),
                "yaw": yaw,
                "status": "pending",
            })
            self.state.next_waypoint_id += 1
            self.state.route_message = "已添加航点"

    def remove_waypoint(self, waypoint_id: int) -> None:
        with self.state.lock:
            if self.state.route_running:
                raise RuntimeError("路线运行中不能删除航点，请先暂停")
            self.state.waypoints = [w for w in self.state.waypoints if int(w["id"]) != waypoint_id]
            self.state.route_index = -1
            self.state.route_message = "已删除航点"

    def start_route(self, loop: bool) -> None:
        self.ensure_action_server()
        with self.state.lock:
            if not self.state.waypoints:
                raise RuntimeError("尚未添加多点航点")
            self.state.route_run_id += 1
            self.state.route_running = True
            self.state.route_loop = bool(loop)
            self.state.route_index = 0
            for w in self.state.waypoints:
                w["status"] = "pending"
            self.state.route_message = "路线开始"
            run_id = self.state.route_run_id
        self.action_client.cancel_all_goals()
        self.send_route_index(run_id, 0)

    def pause_route(self, cancel: bool = True, message: str = "路线已暂停") -> None:
        with self.state.lock:
            self.state.route_run_id += 1
            self.state.route_running = False
            self.state.route_message = message
            idx = self.state.route_index
            if 0 <= idx < len(self.state.waypoints) and self.state.waypoints[idx]["status"] == "active":
                self.state.waypoints[idx]["status"] = "pending"
            self.state.goal = None
        if cancel:
            self.action_client.cancel_all_goals()

    def send_route_index(self, run_id: int, index: int) -> None:
        with self.state.lock:
            if not self.state.route_running or run_id != self.state.route_run_id:
                return
            if index >= len(self.state.waypoints):
                if self.state.route_loop and self.state.waypoints:
                    for w in self.state.waypoints:
                        w["status"] = "pending"
                    index = 0
                else:
                    self.state.route_running = False
                    self.state.route_index = len(self.state.waypoints) - 1
                    self.state.route_message = "全部航点已完成"
                    self.state.goal = None
                    return
            self.state.route_index = index
            waypoint = dict(self.state.waypoints[index])
            self.state.waypoints[index]["status"] = "active"
            self.state.route_message = f"前往航点 {index + 1}/{len(self.state.waypoints)}"
            self.state.goal = {"x": waypoint["x"], "y": waypoint["y"], "yaw": waypoint.get("yaw"), "source": "route"}

        goal = self.build_goal(float(waypoint["x"]), float(waypoint["y"]), waypoint.get("yaw"))
        waypoint_id = int(waypoint["id"])
        self.action_client.send_goal(
            goal,
            done_cb=lambda status, result: self.route_done(run_id, waypoint_id, status),
        )

    def route_done(self, run_id: int, waypoint_id: int, status: int) -> None:
        next_index: Optional[int] = None
        with self.state.lock:
            if run_id != self.state.route_run_id or not self.state.route_running:
                return
            index = next((i for i, w in enumerate(self.state.waypoints) if int(w["id"]) == waypoint_id), -1)
            if index < 0:
                return
            if status == GoalStatus.SUCCEEDED:
                self.state.waypoints[index]["status"] = "done"
                self.state.route_message = f"航点 {index + 1} 已到达"
                next_index = index + 1
            else:
                self.state.waypoints[index]["status"] = "failed"
                self.state.route_running = False
                self.state.route_message = f"航点 {index + 1} 失败，状态码 {status}，路线已暂停"
                self.state.goal = None
        if next_index is not None:
            rospy.Timer(rospy.Duration(0.15), lambda _event: self.send_route_index(run_id, next_index), oneshot=True)

    def skip_route(self) -> None:
        with self.state.lock:
            if not self.state.route_running or self.state.route_index < 0:
                raise RuntimeError("当前没有运行中的路线")
            # Invalidate the done callback of the goal being cancelled, while
            # keeping the route running under a new generation id.
            self.state.route_run_id += 1
            run_id = self.state.route_run_id
            index = self.state.route_index
            if index < len(self.state.waypoints):
                self.state.waypoints[index]["status"] = "skipped"
            next_index = index + 1
            self.state.route_message = "已跳过当前航点"
        self.action_client.cancel_all_goals()
        rospy.Timer(rospy.Duration(0.15), lambda _event: self.send_route_index(run_id, next_index), oneshot=True)

    def route_action(self, request: Dict[str, Any]) -> None:
        action = str(request.get("action", ""))
        if action == "add":
            self.add_waypoint(float(request["x"]), float(request["y"]), None if "yaw" not in request else float(request["yaw"]))
        elif action == "remove":
            self.remove_waypoint(int(request["id"]))
        elif action == "start":
            self.start_route(bool(request.get("loop", False)))
        elif action == "pause":
            self.pause_route()
        elif action == "skip":
            self.skip_route()
        elif action == "clear":
            self.pause_route()
            with self.state.lock:
                self.state.waypoints.clear()
                self.state.route_index = -1
                self.state.route_message = "路线已清空"
        else:
            raise ValueError("未知多点路线操作")

    def obstacle_action(self, request: Dict[str, Any]) -> None:
        action = str(request.get("action", ""))
        with self.state.lock:
            if action == "add":
                self.state.obstacles.append({
                    "id": self.state.next_obstacle_id,
                    "x": float(request["x"]),
                    "y": float(request["y"]),
                    "radius": clamp(float(request.get("radius", 0.35)), 0.10, 3.0),
                })
                self.state.next_obstacle_id += 1
            elif action == "remove_nearest":
                if not self.state.obstacles:
                    return
                x, y = float(request["x"]), float(request["y"])
                index = min(range(len(self.state.obstacles)), key=lambda i: math.hypot(self.state.obstacles[i]["x"] - x, self.state.obstacles[i]["y"] - y))
                self.state.obstacles.pop(index)
            elif action == "clear":
                self.state.obstacles.clear()
            else:
                raise ValueError("未知障碍操作")

    def publish_scan(self, _event: Any) -> None:
        with self.state.lock:
            obstacles = [dict(item) for item in self.state.obstacles]
            odom = self.state.odom
        if odom is None:
            return
        robot_x = float(odom.pose.pose.position.x)
        robot_y = float(odom.pose.pose.position.y)
        robot_yaw = yaw_from_quaternion(odom.pose.pose.orientation)
        count = max(1, int(round((self.scan_max_angle - self.scan_min_angle) / self.scan_increment)) + 1)
        ranges = [float("inf")] * count
        c = math.cos(robot_yaw)
        s = math.sin(robot_yaw)
        for obstacle in obstacles:
            dx = float(obstacle["x"]) - robot_x
            dy = float(obstacle["y"]) - robot_y
            bx = c * dx + s * dy
            by = -s * dx + c * dy
            radius = float(obstacle["radius"])
            distance = math.hypot(bx, by)
            if distance - radius > self.scan_max_range or distance + radius < self.scan_min_range:
                continue
            center_angle = math.atan2(by, bx)
            angular_radius = math.asin(clamp(radius / max(radius, distance), -1.0, 1.0)) if distance > radius else math.pi
            start = max(0, int(math.floor((center_angle - angular_radius - self.scan_min_angle) / self.scan_increment)))
            end = min(count - 1, int(math.ceil((center_angle + angular_radius - self.scan_min_angle) / self.scan_increment)))
            d2 = bx * bx + by * by
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
        scan.scan_time = 1.0 / max(1.0, self.scan_rate_hz)
        scan.range_min = self.scan_min_range
        scan.range_max = self.scan_max_range
        scan.ranges = ranges
        self.scan_pub.publish(scan)

    def action(self, name: str) -> Optional[Dict[str, Any]]:
        if name == "cancel_goal":
            self.action_client.cancel_all_goals()
            self.pause_route(cancel=False, message="当前目标已取消")
        elif name == "clear_costmaps":
            rospy.wait_for_service(self.clear_costmaps_service, timeout=2.0)
            rospy.ServiceProxy(self.clear_costmaps_service, Empty)()
        elif name == "reset_sim":
            self.action_client.cancel_all_goals()
            self.pause_route(cancel=False, message="仿真已复位")
            rospy.wait_for_service(self.reset_sim_service, timeout=2.0)
            rospy.ServiceProxy(self.reset_sim_service, Empty)()
            with self.state.lock:
                self.state.trail.clear()
                self.state.cmd_history.clear()
                for w in self.state.waypoints:
                    w["status"] = "pending"
                self.state.route_index = -1
        elif name == "clear_obstacles":
            with self.state.lock:
                self.state.obstacles.clear()
        elif name == "clear_trail":
            with self.state.lock:
                self.state.trail.clear()
        elif name == "restore_params":
            return {"params": self.restore_startup_params()}
        else:
            raise ValueError("未知操作")
        return None

    @staticmethod
    def age(now: float, received: float) -> float:
        return 1.0e9 if received <= 0.0 else max(0.0, now - received)

    def route_snapshot(self) -> Dict[str, Any]:
        with self.state.lock:
            return {
                "running": self.state.route_running,
                "loop": self.state.route_loop,
                "current_index": self.state.route_index,
                "message": self.state.route_message,
                "waypoints": [dict(w) for w in self.state.waypoints],
            }

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
        trail = [self.apply_transform(point, trail_tfm) for point in trail] if trail_tfm is not None else []
        encoded = bytes(0 if value < 0 else min(100, max(0, int(value))) + 1 for value in map_data)
        recent_signs: List[int] = []
        for stamp, angular in cmd_history:
            if now - stamp > 10.0 or abs(angular) < 0.02:
                continue
            recent_signs.append(1 if angular > 0.0 else -1)
        cmd_changes = sum(1 for a, b in zip(recent_signs, recent_signs[1:]) if a != b)
        if odom is None:
            odom_values = {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
        else:
            twist = odom.twist.twist
            odom_values = {"linear_x": float(twist.linear.x), "linear_y": float(twist.linear.y), "angular_z": float(twist.angular.z)}
        return {
            "ok": True,
            "map": {"frame": map_frame, "width": map_width, "height": map_height, "resolution": map_resolution, "origin_x": map_origin_x, "origin_y": map_origin_y, "data_b64": base64.b64encode(encoded).decode("ascii")},
            "robot": robot,
            "odom": odom_values,
            "cmd": {"linear_x": 0.0 if cmd is None else float(cmd.linear.x), "angular_z": 0.0 if cmd is None else float(cmd.angular.z)},
            "global_plan": global_points,
            "local_plan": local_points,
            "trail": trail,
            "goal": goal,
            "obstacles": obstacles,
            "route": self.route_snapshot(),
            "ages": {"map": self.age(now, map_rx), "odom": self.age(now, odom_rx), "cmd": self.age(now, cmd_rx)},
            "metrics": {"map_updates": map_updates, "cmd_sign_changes": cmd_changes, "local_plan_turn_changes": turn_sign_changes(local_points), "global_plan_turn_changes": turn_sign_changes(global_points), "global_length_m": path_length(global_points), "local_length_m": path_length(local_points)},
        }

    def export_payload(self) -> Dict[str, Any]:
        with self.state.lock:
            waypoints = [{k: v for k, v in w.items() if k in ("x", "y", "yaw")} for w in self.state.waypoints]
            obstacles = [{k: v for k, v in o.items() if k in ("x", "y", "radius")} for o in self.state.obstacles]
            loop = self.state.route_loop
        return {
            "format": "r300_dwa_web_config_v2",
            "exported_at": datetime.now().astimezone().isoformat(),
            "note": "DWA and costmap parameters are read from the same visual-navigation move_base instance.",
            "parameters": self.get_params(capture_startup=False),
            "startup_only_parameters": {
                "move_base": {
                    "latch_xy_goal_tolerance": bool(rospy.get_param(
                        "/move_base/latch_xy_goal_tolerance", False
                    )),
                },
                "note": "These parameters are not dynamic; edit YAML and restart move_base.",
            },
            "route": {"loop": loop, "waypoints": waypoints},
            "obstacles": obstacles,
            "synthetic_vision_scan": {
                "topic": self.scan_topic,
                "rate_hz": self.scan_rate_hz,
                "angle_min_deg": math.degrees(self.scan_min_angle),
                "angle_max_deg": math.degrees(self.scan_max_angle),
                "angle_increment_deg": math.degrees(self.scan_increment),
                "range_min_m": self.scan_min_range,
                "range_max_m": self.scan_max_range,
            },
        }

    def import_payload(self, request: Dict[str, Any]) -> None:
        data = request.get("data")
        if not isinstance(data, dict):
            raise ValueError("导入数据必须是JSON对象")
        route = data.get("route", {})
        waypoints = route.get("waypoints", []) if isinstance(route, dict) else []
        obstacles = data.get("obstacles", [])
        self.pause_route()
        with self.state.lock:
            self.state.waypoints.clear()
            self.state.obstacles.clear()
            self.state.route_loop = bool(route.get("loop", False)) if isinstance(route, dict) else False
            for item in waypoints:
                if not isinstance(item, dict):
                    continue
                self.state.waypoints.append({
                    "id": self.state.next_waypoint_id,
                    "x": float(item["x"]),
                    "y": float(item["y"]),
                    "yaw": None if item.get("yaw") is None else float(item["yaw"]),
                    "status": "pending",
                })
                self.state.next_waypoint_id += 1
            for item in obstacles:
                if not isinstance(item, dict):
                    continue
                self.state.obstacles.append({
                    "id": self.state.next_obstacle_id,
                    "x": float(item["x"]),
                    "y": float(item["y"]),
                    "radius": clamp(float(item.get("radius", 0.35)), 0.1, 3.0),
                })
                self.state.next_obstacle_id += 1
            self.state.route_message = "已从JSON导入路线"
        if bool(request.get("apply_params", False)) and isinstance(data.get("parameters"), dict):
            self.set_params(data["parameters"])

    @staticmethod
    def json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
        length = int(handler.headers.get("Content-Length", "0"))
        if length <= 0 or length > 5 * 1024 * 1024:
            raise ValueError("请求体为空或过大")
        value = json.loads(handler.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON请求必须是对象")
        return value

    def start_http(self) -> ThreadingHTTPServer:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                rospy.logdebug(fmt, *args)

            def reply(self, status: int, value: Any, content_type: str = "application/json; charset=utf-8", filename: Optional[str] = None) -> None:
                if isinstance(value, (dict, list)):
                    body = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
                elif isinstance(value, bytes):
                    body = value
                else:
                    body = str(value).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                if filename:
                    self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                try:
                    if path in ("/", "/index.html"):
                        self.reply(200, HTML_PAGE, "text/html; charset=utf-8")
                    elif path == "/api/state":
                        self.reply(200, owner.snapshot())
                    elif path == "/api/params":
                        self.reply(200, {"ok": True, "params": owner.get_params()})
                    elif path == "/api/export":
                        fmt = parse_qs(parsed.query).get("format", ["yaml"])[0].lower()
                        payload = owner.export_payload()
                        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        if fmt == "json":
                            self.reply(200, payload, "application/json; charset=utf-8", f"r300_dwa_config_{stamp}.json")
                        else:
                            self.reply(200, yaml_dump(payload) + "\n", "application/x-yaml; charset=utf-8", f"r300_dwa_config_{stamp}.yaml")
                    elif path == "/health":
                        self.reply(200, "ok\n", "text/plain; charset=utf-8")
                    else:
                        self.reply(404, {"ok": False, "error": "not found"})
                except Exception as exc:
                    self.reply(503, {"ok": False, "error": str(exc)})

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                try:
                    request = owner.json_body(self)
                    if path == "/api/params":
                        params = owner.set_params(request)
                        self.reply(200, {
                            "ok": True,
                            "params": params,
                            "warnings": list(owner.last_param_warnings),
                        })
                    elif path == "/api/goal":
                        owner.send_goal(float(request["x"]), float(request["y"]), None if "yaw" not in request else float(request["yaw"]))
                        self.reply(200, {"ok": True})
                    elif path == "/api/route":
                        owner.route_action(request)
                        self.reply(200, {"ok": True})
                    elif path == "/api/obstacle":
                        owner.obstacle_action(request)
                        self.reply(200, {"ok": True})
                    elif path == "/api/action":
                        extra = owner.action(str(request.get("action", ""))) or {}
                        self.reply(200, dict({"ok": True}, **extra))
                    elif path == "/api/import":
                        owner.import_payload(request)
                        self.reply(200, {"ok": True})
                    else:
                        self.reply(404, {"ok": False, "error": "not found"})
                except Exception as exc:
                    self.reply(400, {"ok": False, "error": str(exc)})

        server = ThreadingHTTPServer((self.host, self.port), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
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
