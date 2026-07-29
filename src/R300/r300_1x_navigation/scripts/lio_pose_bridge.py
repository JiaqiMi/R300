#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""室内测试模式：用 FAST-LIO 位姿桥接出 odom->base_link 与 DWA 里程计。

背景（2026-07-29 实测）：1X 组合惯导室内无 GPS 时 odom 冻结，导航闭环
无法测试。本桥用 FAST-LIO 的 odom->body 复合固定安装外参，发布：
  1) TF odom->base_link（50Hz）
  2) nav_msgs/Odometry 到 ~odom_topic（默认 /subject1/dwa_odom，
     twist 用位姿差分+一阶低通，室内低速测试足够）

使用红线：
- 本模式下【不要】给 1X 设原点（否则它也发 odom->base_link，双父帧冲突）；
  建议室内测试直接不启 1X。
- 本模式下全系统单树（都在 FAST-LIO 的 odom 语义里），rviz 中高程图与
  代价地图天然对齐；GPS 航点(waypoint_executor)在本模式下无意义，只用 rviz 点目标。
- 外参约定与 lidar_obstacle_scan_node 完全一致（ext_x/ext_y/ext_yaw_deg，
  launch 里复用 subject1_lidar_obstacles.yaml 加载）。
"""
import math

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry

rospy.init_node('lio_pose_bridge')
ext_x = float(rospy.get_param('~ext_x', 0.40))
ext_y = float(rospy.get_param('~ext_y', 0.0))
ext_yaw = math.radians(float(rospy.get_param('~ext_yaw_deg', 0.0)))
sensor_h = float(rospy.get_param('~sensor_height_above_ground', 0.48))
lio_map = rospy.get_param('~lio_map_frame', 'odom')
lio_body = rospy.get_param('~lio_body_frame', 'body')
out_frame = rospy.get_param('~output_base_frame', 'base_link')
odom_topic = rospy.get_param('~odom_topic', '/subject1/dwa_odom')
rate_hz = float(rospy.get_param('~rate', 50.0))

buf = tf2_ros.Buffer()
tf2_ros.TransformListener(buf)
br = tf2_ros.TransformBroadcaster()
pub = rospy.Publisher(odom_topic, Odometry, queue_size=10)
state = {'t': None, 'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'v': 0.0, 'w': 0.0}
ALPHA = 0.35  # twist 一阶低通


def tick(_):
    try:
        tr_full = buf.lookup_transform(lio_map, lio_body, rospy.Time(0),
                                       rospy.Duration(0.05))
    except tf2_ros.TransformException:
        rospy.logwarn_throttle(5.0, 'lio_pose_bridge: FAST-LIO TF(%s->%s) 不可用，桥暂停',
                               lio_map, lio_body)
        return
    tr = tr_full.transform
    tf_stamp = tr_full.header.stamp.to_sec()
    q = tr.rotation
    yaw_body = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    yaw = yaw_body - ext_yaw
    c, s = math.cos(yaw), math.sin(yaw)
    x = tr.translation.x - (c * ext_x - s * ext_y)
    y = tr.translation.y - (s * ext_x + c * ext_y)
    z = tr.translation.z - sensor_h

    now = rospy.Time.now()
    # 2026-07-29 修复(对抗性审查 P0-1): 差分必须以 TF 时戳为准, 且只在 TF 真正更新
    # (~10Hz)时进行。旧实现按 50Hz 定时器差分——4/5 个 tick 得 0、1/5 得 5 倍真速,
    # 低通后速度估计在 0.35~2 倍真值间振荡, DWA 加速度采样窗被垃圾锚点带偏,
    # 是室内 "failed to produce path"×45 与掉头爬行的头号放大器。
    new_sample = False
    if state['t'] is None:
        state.update(t=tf_stamp, x=x, y=y, yaw=yaw)
        new_sample = True
    elif tf_stamp > state['t'] + 1e-6:
        dt = tf_stamp - state['t']
        dyaw = (yaw - state['yaw'] + math.pi) % (2 * math.pi) - math.pi
        vx = ((x - state['x']) * c + (y - state['y']) * s) / dt
        w = dyaw / dt
        state['v'] += ALPHA * (vx - state['v'])
        state['w'] += ALPHA * (w - state['w'])
        state.update(t=tf_stamp, x=x, y=y, yaw=yaw)
        new_sample = True

    tfm = TransformStamped()
    tfm.header.stamp = now
    tfm.header.frame_id = lio_map
    tfm.child_frame_id = out_frame
    tfm.transform.translation.x = x
    tfm.transform.translation.y = y
    tfm.transform.translation.z = z
    tfm.transform.rotation.z = math.sin(yaw / 2)
    tfm.transform.rotation.w = math.cos(yaw / 2)
    br.sendTransform(tfm)
    # 2026-07-30 旋转判撞修复配套: 障碍 scan 现以 FAST-LIO TF 时刻打戳, 快照层按该
    # 过去时刻查本桥的 odom->base_link。FAST-LIO 每出一个新样本(~10Hz)就补发一帧
    # "精确采样时刻"的 TF, 使该查询命中真实样本——只有 50Hz now 时戳帧(阶梯保持位姿)
    # 时, 插值会残留 0~100ms 切向错位。now 帧保留(move_base 按当前时刻查询依赖它)。
    if new_sample:
        tfe = TransformStamped()
        tfe.header.stamp = tr_full.header.stamp
        tfe.header.frame_id = lio_map
        tfe.child_frame_id = out_frame
        tfe.transform = tfm.transform
        br.sendTransform(tfe)

    od = Odometry()
    od.header.stamp = now
    od.header.frame_id = lio_map
    od.child_frame_id = out_frame
    od.pose.pose.position.x = x
    od.pose.pose.position.y = y
    od.pose.pose.position.z = z
    od.pose.pose.orientation = tfm.transform.rotation
    od.twist.twist.linear.x = state['v']
    od.twist.twist.angular.z = state['w']
    pub.publish(od)


rospy.loginfo('lio_pose_bridge(室内测试模式): %s->%s @%.0fHz, odom_topic=%s, '
              'ext=(%.2f, %.2f, %.1f°) —— 1X 请勿设原点!',
              lio_map, out_frame, rate_hz, odom_topic, ext_x, ext_y, math.degrees(ext_yaw))
rospy.Timer(rospy.Duration(1.0 / rate_hz), tick)
rospy.spin()
