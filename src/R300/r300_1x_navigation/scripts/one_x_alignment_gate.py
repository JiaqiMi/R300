#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1X 对准门检：拒绝"静默假定位"进入导航。

2026-07-29 实锤教训：室内无 GPS 时 1X 的 /one_x/odom 以 100Hz 发布"逐字节相同"
的冻结位姿（heading_deg 可能非零但位姿仍冻结！GPS status=10 无定位时控制位置流
不解算）——车物理旋转而软件位姿纹丝不动 → DWA 无限原地转圈、目标永不可达。

判据：采样窗口内位姿极差精确为 0 = 冻结（活的惯导即使静止也有微噪声，
实测冻结样本 61 帧 ptp=0.000000）。本节点 required=true：
- 门检失败 → exit(1) → 整个导航 launch 退出并打印醒目错误；
- 门检通过 → 常驻空转（required 节点不能退出）。
室内测试请改用 indoor_test_mode:=true（FAST-LIO 位姿桥, 不经过本门检）。
"""
import time

import numpy as np
import rospy
from nav_msgs.msg import Odometry

rospy.init_node('one_x_alignment_gate')
window_s = float(rospy.get_param('~window_s', 3.0))

buf = []
rospy.Subscriber(
    '/one_x/odom', Odometry,
    lambda m: buf.append((m.pose.pose.position.x, m.pose.pose.position.y,
                          m.pose.pose.orientation.z, m.pose.pose.orientation.w)),
    queue_size=200)

t0 = time.time()
while (time.time() - t0 < window_s + 8.0 and len(buf) < max(30, int(window_s * 50))
       and not rospy.is_shutdown()):
    time.sleep(0.05)

if len(buf) < 20:
    rospy.logfatal('【1X 门检失败】/one_x/odom 无数据——1X 未启动或未设原点。'
                   '室外: 先 start_1x.sh 并 set_current_origin; 室内测试: indoor_test_mode:=true')
    raise SystemExit(1)

a = np.array(buf)
if float(np.ptp(a, axis=0).max()) == 0.0:
    rospy.logfatal('【1X 门检失败】位姿输出完全冻结（%d 帧逐字节相同）——典型原因: 室内无 GPS,'
                   '控制位置流未解算(pos_compare 里 GPS lat=0)。禁止进入导航(会原地无限转圈)。'
                   '处置: 去室外重设原点, 或室内用 indoor_test_mode:=true。', len(a))
    raise SystemExit(1)

rospy.loginfo('1X 门检通过: %d 帧位姿有正常噪声/变化, 定位源可信。', len(a))
rospy.spin()
