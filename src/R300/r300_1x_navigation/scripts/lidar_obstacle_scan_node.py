#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""雷达高程图 → 虚拟 LaserScan 避障适配节点（方案A：替代视觉避障的观测源）。

链路位置（对照原视觉链路，详见 LIDAR_AVOIDANCE_VS_VISION_README_CN.md）：
  原: /r300_vision/detections → vision_obstacle_layer_node → /r300_vision/obstacle_scan
  新: /elevation_mapping/elevation_map_raw → 本节点 → /r300_lidar/obstacle_scan
下游 VisionSnapshotLayer / InflationLayer / move_base / DWA 完全不变。

坐标系红线（single_lidar_elevation/bringup/README.md "共跑必读"）：
  高程图活在 FAST-LIO 的 odom 树(odom→camera_init→body)，导航活在 1X 的
  odom 树(odom→base_link)，两树同名不同源，禁止跨树查 TF。本节点只在
  FAST-LIO 树内查 odom→body，再用【固定安装外参】把障碍换算成车体系
  (base_link 语义)的相对坐标输出——输出 LaserScan 的 frame_id=base_link，
  由 VisionSnapshotLayer 用 1X 自己的 TF 落图，两树各管各的。

安装外参约定（只需 3 个平面量，其余已被感知栈内部消化）：
  ext_x/ext_y  = 雷达原点在车体系(base_link, x前y左)下的水平位置，米
  ext_yaw_deg  = 雷达安装航向相对车头的偏角，度（逆时针为正）
  默认全 0（雷达近似装在车体中心且朝前）。俯仰 -42.3°/横滚 -2.1°/离地
  0.48m 无需在此配置：高程图本身是重力对齐的水平格网，高度语义由
  节点内"局部地面自参考"处理。

算法移植自 mevius2_nav/elevation_obstacle_node.py v3（真机验证版）：
  1) 局部地面自参考：车周环带高程中位数为地面基准（抗 z 漂移，v2 曾因
     TF 地面基准在新场地爆 6000+ 假负障碍）；
  2) 三通道：正障碍(h>pos)/负障碍(h<-drop, 坑沿虚拟墙)/几何陡坡(>30°)；
  3) 世界格 N 帧去抖（负3帧/正2帧）；
  4) 移除了 v3 的 /move_base/clear_costmaps 调用——VisionSnapshotLayer
     的每障碍 TTL 自动过期，无需主动清图。

第三招·地面确认限速（enable_ground_speed_limit 开关）：
  坑的探测距离是几何定死的(d≈3.2×坑宽)，"更早看见坑"无解，正解是
  "不要开得比验证过的地面更快"：把车前走廊按纵向切片，统计每片内
  |h|≤阈值 的有效地面格，得到连续验证地面距离 D（负障碍/陡坡/正障碍/
  无数据空洞都会截断 D），再按完整制动方程
      v = -a·t_r + sqrt((a·t_r)² + 2a·max(0, D - 余量))
  经 dynamic_reconfigure 动态压 DWA 的 max_vel_x/max_vel_trans。
  前方 7m 验证平地→放开跑；3m 处出现空洞→自动降速；扫清后自动恢复。
  speed_limit_apply_to_dwa=false 时只发布 /r300_lidar/ground_speed_limit
  观测值、不干预 DWA（实车首跑建议先观测一圈再打开执行）。
"""

import math

import numpy as np
import rospy
import tf2_ros
from grid_map_msgs.msg import GridMap
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pcl2
from std_msgs.msg import Float32, Header
from geometry_msgs.msg import Twist
from actionlib_msgs.msg import GoalStatusArray

try:
    from dynamic_reconfigure.client import Client as DRClient
except ImportError:  # dynamic_reconfigure 缺失时限速降级为纯观测
    DRClient = None

CLOUD_FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]


class LidarObstacleScan:
    def __init__(self):
        # ---- 输入/输出 ----
        self.elevation_topic = rospy.get_param('~elevation_topic', '/elevation_mapping/elevation_map_raw')
        self.scan_topic = rospy.get_param('~scan_topic', '/r300_lidar/obstacle_scan')
        self.debug_cloud_topic = rospy.get_param('~debug_cloud_topic', '/r300_lidar/obstacle_points')
        self.publish_debug_cloud = bool(rospy.get_param('~publish_debug_cloud', True))
        # ---- FAST-LIO 树内的 frame（勿改成 1X 树的帧！）----
        self.lio_map_frame = rospy.get_param('~lio_map_frame', 'odom')
        self.lio_body_frame = rospy.get_param('~lio_body_frame', 'body')
        # ---- 输出 frame（1X 树语义，仅作为 frame_id 字符串，本节点不查它的 TF）----
        self.output_frame = rospy.get_param('~output_frame', 'base_link')
        # ---- 安装外参（待实车标定，先给 0）----
        self.ext_x = float(rospy.get_param('~ext_x', 0.0))
        self.ext_y = float(rospy.get_param('~ext_y', 0.0))
        self.ext_yaw = math.radians(float(rospy.get_param('~ext_yaw_deg', 0.0)))
        # ---- 地形判定（v3 语义）----
        self.drop = float(rospy.get_param('~drop_thresh', 0.15))
        self.pos = float(rospy.get_param('~pos_thresh', 0.15))
        self.max_h = float(rospy.get_param('~max_height', 1.5))
        self.slope_deg = float(rospy.get_param('~slope_thresh_deg', 30.0))
        self.min_range = float(rospy.get_param('~min_range', 0.60))
        self.max_range = float(rospy.get_param('~max_range', 5.5))
        self.ring_min = float(rospy.get_param('~ground_ring_min', 0.60))
        self.ring_max = float(rospy.get_param('~ground_ring_max', 1.60))
        self.ring_min_cells = int(rospy.get_param('~ground_ring_min_cells', 30))
        self.sensor_height = float(rospy.get_param('~sensor_height_above_ground', 0.48))
        self.neg_confirm = int(rospy.get_param('~neg_confirm', 3))
        self.pos_confirm = int(rospy.get_param('~pos_confirm', 2))
        # 2026-07-30 室外2m/s实测"发现晚"根因之二: MID360非重复扫描下远处4cm格
        # 每帧命中率仅~50%, 原"未命中-1"衰减使计数器 +1,-1,+1,-1 永远到不了阈值
        # ——确认被结构性推迟到近处。未命中改衰减0.5: 50%命中率~0.5s内可确认。
        self.confirm_miss_decay = float(rospy.get_param('~confirm_miss_decay', 0.5))
        # 数据接缝伪障守卫：高程图数据前沿 3~4 格是"少点数不可靠带"（掠射单点建格），
        # 会产生环状伪坡/伪正障碍（2026-07-29 实测：侵蚀3层伪坡-59%、伪正障碍-91%）。
        # 障碍判定只在"边界侵蚀 N 层后的可靠区"内做，代价是障碍轮廓缩 N*4cm（核心保留）。
        self.edge_erosion = int(rospy.get_param('~edge_erosion', 3))
        # ---- 虚拟扫描布局（对齐 vision_obstacle_layer_node 的输出契约）----
        fov_deg = float(rospy.get_param('~scan_fov_deg', 120.0))
        inc_deg = float(rospy.get_param('~scan_angle_increment_deg', 0.5))
        self.angle_min = -math.radians(fov_deg) / 2.0
        self.angle_inc = math.radians(inc_deg)
        self.n_beams = int(round(math.radians(fov_deg) / self.angle_inc)) + 1
        self.range_min = float(rospy.get_param('~scan_range_min_m', 0.20))
        self.range_max = float(rospy.get_param('~scan_range_max_m', 10.5))
        self.safety_margin = float(rospy.get_param('~range_safety_margin_m', 0.15))
        self.input_timeout = float(rospy.get_param('~input_timeout_s', 2.0))
        # ---- 第三招：地面确认限速 ----
        self.enable_speed_limit = bool(rospy.get_param('~enable_ground_speed_limit', True))
        self.sl_apply = bool(rospy.get_param('~speed_limit_apply_to_dwa', True))
        self.sl_v_max = float(rospy.get_param('~speed_limit_v_max', 1.5))
        self.sl_v_min = float(rospy.get_param('~speed_limit_v_min', 0.3))
        self.sl_acc = max(0.1, float(rospy.get_param('~speed_limit_brake_acc', 1.0)))
        self.sl_react = float(rospy.get_param('~speed_limit_react_time', 0.5))
        self.sl_margin = float(rospy.get_param('~speed_limit_margin', 0.75))
        self.cor_half_w = float(rospy.get_param('~corridor_half_width', 0.45))
        self.cor_start = float(rospy.get_param('~corridor_start', 0.7))
        self.cor_bin = max(0.1, float(rospy.get_param('~corridor_bin', 0.25)))
        self.cor_frac = float(rospy.get_param('~corridor_min_ground_frac', 0.3))
        self.cor_min_cells = int(rospy.get_param('~corridor_min_ground_cells', 8))
        # ---- 幽灵污染根治两件套（2026-07-29 实测: 车周42%致命格为行人残影）----
        # 记忆过期: 高程图 time 层=每格距上次真实观测的秒数, 超龄障碍不再进 scan。
        # 真墙在视场内持续刷新不受影响; 转弯所需的秒级侧后记忆保留。
        self.pos_memory_ttl = float(rospy.get_param('~pos_memory_ttl_s', 20.0))
        # 2026-07-30 分段记忆(审查): 近车环带(min_range~near_ring_m)结构性无法被再次
        # 观测(前向最近真实回波~0.94m), 障碍前静止>TTL 会"过期蒸发→起步撞", 给长记忆;
        # 环带外保持短 TTL 防行人残影幽灵(20s 是室内实测调定的值, 勿整体放大)。
        self.pos_memory_ttl_near = float(rospy.get_param('~pos_memory_ttl_near_s', 60.0))
        self.near_ring = float(rospy.get_param('~near_ring_m', 1.2))
        self.neg_memory_ttl = float(rospy.get_param('~neg_memory_ttl_s', 60.0))
        # 自动脱困: move_base 已放弃(ABORTED)+车头堵+车尾走廊经高程图验证干净+静止
        # → 自动低速倒车 0.4m（DWA 在 footprint 陷入致命区后连倒车轨迹都会拒绝, 见死锁分析）
        self.enable_auto_unstick = bool(rospy.get_param('~enable_auto_unstick', True))
        self.unstick_cooldown = float(rospy.get_param('~unstick_cooldown_s', 15.0))

        self._neg_hist = {}
        self._pos_hist = {}
        self._last_msg_time = rospy.Time(0)
        self._dr_client = None
        self._dr_retry_after = 0.0
        self._last_sent_v = None
        self._last_hb = 0.0
        self._last_abort = 0.0
        self._last_unstick = 0.0
        self._abort_goal_id = None
        self._handled_goal_id = None
        self._mb_active = False
        self._rear_free = False
        self._pose_hist = []
        self._time_layer_warned = False
        self.buf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.buf)

        self.pub_scan = rospy.Publisher(self.scan_topic, LaserScan, queue_size=1)
        self.pub_cloud = rospy.Publisher(self.debug_cloud_topic, PointCloud2, queue_size=1)
        self.pub_vlimit = rospy.Publisher('/r300_lidar/ground_speed_limit', Float32, queue_size=1)
        rospy.Subscriber(self.elevation_topic, GridMap, self.cb, queue_size=1,
                         buff_size=64 * 1024 * 1024)
        self.pub_cmd = rospy.Publisher('/subject1/cmd_vel_raw', Twist, queue_size=1)
        rospy.Subscriber('/move_base/status', GoalStatusArray, self._mb_status_cb, queue_size=2)
        rospy.Timer(rospy.Duration(1.0), self._watchdog)
        rospy.loginfo(
            'lidar_obstacle_scan: %s -> %s | 负<-%.2f 正>+%.2f 坡>%.0f° | 视场±%.0f° 量程%.1fm | '
            '外参 x=%.2f y=%.2f yaw=%.1f°(0=待标定)',
            self.elevation_topic, self.scan_topic, self.drop, self.pos, self.slope_deg,
            fov_deg / 2.0, self.max_range, self.ext_x, self.ext_y, math.degrees(self.ext_yaw))

    def _watchdog(self, _):
        age = (rospy.Time.now() - self._last_msg_time).to_sec()
        if age > self.input_timeout:
            rospy.logwarn_throttle(
                5.0, 'lidar_obstacle_scan: 高程图输入超时 %.1fs，已停发 scan（下游 TTL 会在 3s 内清空障碍）', age)
            self._fail_safe_slow('高程图断流')
            return
        # —— 自动脱困反射: move_base 已放弃(4s内) + 冷却期外 + 车头堵 + 车尾走廊
        # 经高程图验证干净 + 静止。ABORTED 后 move_base 不再发 cmd_vel, 无话题争抢。
        now = rospy.get_time()
        # 2026-07-30 修盲区: 原条件含"车头堵(D<1.2)", 但角卡(footprint角压致命格)时
        # 前方开阔照样全轨迹违规——move_base放弃+静止本身就是被困充分证据。
        # 改按目标ID边沿触发: 每个被放弃的目标只自救一次, 防止陈旧ABORTED状态无限重试。
        # 2026-07-30 审查修订三守卫: ①每个被放弃的目标只救一次(成败都标 handled)——
        # "失败即重试"无法区分急停与"车尾被顶住"(尾后0.45~0.6m环带在 min_range 盲区内,
        # 细障碍不可见), 重试=以0.25反复撞击; ②要求当前无 ACTIVE 目标, 防倒车插进
        # 下一目标的原地对位期与 DWA 抢 cmd_vel_raw(原地旋转也判"静止"); ③要求位姿流
        # 新鲜(<1s), 防 TF 异常期间拿冻结的"静止+车尾净"素材盲发倒车。
        if (self.enable_auto_unstick
                and self._abort_goal_id is not None
                and self._abort_goal_id != self._handled_goal_id
                and not self._mb_active
                and now - self._last_abort < 30.0
                and now - self._last_unstick > self.unstick_cooldown
                and self._pose_hist and now - self._pose_hist[-1][0] < 1.0
                and self._rear_free and self._stationary()):
            goal_id = self._abort_goal_id
            self._handled_goal_id = goal_id
            start = self._pose_hist[-1]
            rospy.logwarn('lidar_obstacle_scan: 【自动脱困】DWA已放弃(角卡/围困), '
                          '车尾1.3m走廊已验证干净 → 倒车0.4m')
            self._last_unstick = now
            tw = Twist()
            # 2026-07-30 -0.15→-0.25: 实测底盘死区≈0.2m/s, -0.15 属"指令在发轮子不转",
            # 唯一自救路径空转。16 拍×0.1s×0.25=0.4m(固件坡道下实际约 0.2~0.35m)。
            tw.linear.x = -0.25
            for _ in range(16):
                if rospy.is_shutdown():
                    break
                if self._mb_active:
                    # 审查P1: 倒车中途出现新 ACTIVE 目标(executor 重试/跳过后发出),
                    # 立即让位停车, 避免与 DWA 前进指令在 cmd_vel_raw 上 10Hz/5Hz 交替争抢
                    rospy.logwarn('lidar_obstacle_scan: 【自动脱困】倒车中检测到新导航目标, 提前让位停车')
                    break
                self.pub_cmd.publish(tw)
                rospy.sleep(0.1)
            self.pub_cmd.publish(Twist())
            rospy.sleep(0.3)  # 等 cb 线程刷入倒车后的位姿
            # 位移复核(仅用于告警, 不重试): 终点位姿必须比倒车开始时刻新才可信
            end = self._pose_hist[-1] if self._pose_hist else None
            if end is not None and end[0] > start[0] + 0.5:
                moved = math.hypot(end[1] - start[1], end[2] - start[2])
                if moved >= 0.10:
                    rospy.logwarn('lidar_obstacle_scan: 【自动脱困】完成(实际倒车 %.2fm), '
                                  '请重新下发目标', moved)
                else:
                    rospy.logerr('lidar_obstacle_scan: 【自动脱困】指令已发但位移仅 %.2fm'
                                 '——疑似急停/被顶住/底盘未使能, 本目标不再自动重试, 请人工处理',
                                 moved)
            else:
                rospy.logwarn('lidar_obstacle_scan: 【自动脱困】已执行, 但位姿流中断无法复核位移, 请人工确认')

    def _fail_safe_slow(self, reason):
        """感知不可信时把限速压到下限（地面无法验证 = 不允许快跑）。"""
        if not self.enable_speed_limit:
            return
        self.pub_vlimit.publish(Float32(data=self.sl_v_min))
        self._send_dwa_limit(self.sl_v_min, force=True)
        rospy.logwarn_throttle(5.0, 'lidar_obstacle_scan: %s，限速压至 %.1fm/s', reason, self.sl_v_min)

    def _send_dwa_limit(self, v, force=False):
        """经 dynamic_reconfigure 写 DWA 的 max_vel_x/max_vel_trans。
        量化 0.05m/s 去抖 + 2s 心跳重发；move_base 未就绪时静默退避重试，
        speed_limit_apply_to_dwa=false 时本函数不生效（纯观测模式）。"""
        if not (self.sl_apply and self.enable_speed_limit) or DRClient is None:
            return
        now = rospy.get_time()
        vq = round(v / 0.05) * 0.05
        if (not force and self._last_sent_v is not None
                and abs(vq - self._last_sent_v) < 0.049
                and now - self._last_hb < 2.0):
            return
        if self._dr_client is None:
            if now < self._dr_retry_after:
                return
            try:
                self._dr_client = DRClient('/move_base/DWAPlannerROS', timeout=0.5)
                rospy.loginfo('lidar_obstacle_scan: 已连接 DWA dynamic_reconfigure，限速接管 max_vel_x/max_vel_trans')
            except Exception:
                self._dr_retry_after = now + 5.0
                rospy.logwarn_throttle(
                    30.0, 'lidar_obstacle_scan: DWA dynamic_reconfigure 未就绪（move_base 未启动?），限速暂为纯观测')
                return
        try:
            self._dr_client.update_configuration({'max_vel_x': vq, 'max_vel_trans': vq})
            self._last_sent_v = vq
            self._last_hb = now
        except Exception as exc:
            self._dr_client = None
            self._dr_retry_after = now + 5.0
            rospy.logwarn_throttle(10.0, 'lidar_obstacle_scan: 下发限速失败(%s)，重连中', exc)

    def _mb_status_cb(self, msg):
        active = False
        for st in msg.status_list:
            if st.status in (0, 1):  # PENDING/ACTIVE = 正在执行新目标, 禁止脱困插话
                active = True
            if st.status == 4:  # ABORTED = move_base 已放弃且不再发 cmd_vel
                self._last_abort = rospy.get_time()
                self._abort_goal_id = st.goal_id.id  # 边沿触发: 每个放弃的目标只救一次
        self._mb_active = active

    def _stationary(self):
        """近 1.2 秒位移 < 5cm 视为静止（脱困前置条件, 防与运动中的 DWA 抢话题）"""
        if len(self._pose_hist) < 2:
            return False
        tn, xn, yn = self._pose_hist[-1]
        for t0, x0, y0 in self._pose_hist:
            if tn - t0 >= 1.2:
                return math.hypot(xn - x0, yn - y0) < 0.05
        return False

    @staticmethod
    def _layer_matrix(msg, name):
        """GridMap 层 → [x_idx, y_idx] 矩阵，含环形缓存还原（对齐 lidar_web_adapter 的解析）。"""
        layers = list(msg.layers)
        if name not in layers:
            return None
        arr = msg.data[layers.index(name)]
        dims = list(arr.layout.dim)
        if len(dims) < 2:
            return None
        size0, size1 = int(dims[0].size), int(dims[1].size)
        data = np.asarray(arr.data, dtype=np.float32)
        if data.size < size0 * size1:
            return None
        data = data[:size0 * size1]
        if str(dims[0].label) == 'row_index':
            mat = data.reshape((size0, size1), order='C')
        else:  # grid_map 默认列主序，dim[0]=column_index
            mat = data.reshape((size1, size0), order='F')
        outer = int(msg.outer_start_index) % max(1, mat.shape[0])
        inner = int(msg.inner_start_index) % max(1, mat.shape[1])
        return np.roll(mat, shift=(-outer, -inner), axis=(0, 1))

    def cb(self, m):
        self._last_msg_time = rospy.Time.now()
        # 1) FAST-LIO 树内取车辆位姿（唯一一次 TF 查询，绝不触碰 1X 树）
        try:
            t = self.buf.lookup_transform(self.lio_map_frame, self.lio_body_frame,
                                          rospy.Time(0), rospy.Duration(0.3))
        except tf2_ros.TransformException as exc:
            rospy.logwarn_throttle(5.0, 'lidar_obstacle_scan: TF %s->%s 不可用: %s',
                                   self.lio_map_frame, self.lio_body_frame, exc)
            self._fail_safe_slow('FAST-LIO TF 不可用')
            return
        # 2026-07-30 审查P0配套: scan 现以 TF 时刻打戳, 而 lookup(Time(0)) 只要缓存里有
        # 任意旧数据就"成功"不抛异常——TF 陈旧时若照发 scan, 下游快照层会按消息龄(>1.0s)
        # 静默丢帧, TTL 3s 后代价地图整层清空=盲驶。这里显式卡 TF 龄: 陈旧即按感知失效
        # 处理(停发 scan → 快照层 stop_on_stale 1s 后令 move_base 停车, 与"TF 不可用"
        # 走同一条失效保护链)。
        tf_age = (rospy.Time.now() - t.header.stamp).to_sec()
        if tf_age > 0.35:
            rospy.logwarn_throttle(2.0, 'lidar_obstacle_scan: FAST-LIO TF 陈旧 %.2fs, 停发 scan', tf_age)
            self._fail_safe_slow('FAST-LIO TF 陈旧')
            return
        q = t.transform.rotation
        yaw_body = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        heading = yaw_body - self.ext_yaw            # 车头在 lio-odom 中的航向
        ch, sh = math.cos(heading), math.sin(heading)
        # 车体原点 = 雷达原点 - Rz(heading)·(ext_x, ext_y)（水平近似，俯仰已被重力对齐消化）
        bx = t.transform.translation.x - (ch * self.ext_x - sh * self.ext_y)
        by = t.transform.translation.y - (sh * self.ext_x + ch * self.ext_y)

        elev = self._layer_matrix(m, 'elevation')
        if elev is None:
            return
        res = float(m.info.resolution)
        cx, cy = m.info.pose.position.x, m.info.pose.position.y
        nx, ny = elev.shape
        xs = cx + (0.5 * nx * res - (np.arange(nx) + 0.5) * res)
        ys = cy + (0.5 * ny * res - (np.arange(ny) + 0.5) * res)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        d2 = (X - bx) ** 2 + (Y - by) ** 2
        fin = np.isfinite(elev)

        # 2) 局部地面自参考（v3 核心）：车周环带高程中位数
        ring = fin & (d2 > self.ring_min ** 2) & (d2 < self.ring_max ** 2)
        if int(ring.sum()) >= self.ring_min_cells:
            ground = float(np.median(elev[ring]))
        else:
            # 环带数据不足：退回"雷达高度-离地高"的几何地面估计
            ground = float(t.transform.translation.z) - self.sensor_height
            rospy.logwarn_throttle(
                10.0, 'lidar_obstacle_scan: 地面环带格数不足(%d<%d)，退用几何地面 %.2f',
                int(ring.sum()), self.ring_min_cells, ground)

        rng = (d2 < self.max_range ** 2) & (d2 > self.min_range ** 2)
        h = elev - ground

        # 可靠区 = 有效区做 N 层 3x3 侵蚀（见 edge_erosion 注释）；环带地面仍用全有效区（中位数抗噪）
        rel = fin
        if self.edge_erosion > 0:
            rel = fin
            for _ in range(self.edge_erosion):
                p = np.pad(rel, 1, constant_values=False)
                rel = (p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1]
                       & p[1:-1, :-2] & p[1:-1, 2:]
                       & p[:-2, :-2] & p[:-2, 2:] & p[2:, :-2] & p[2:, 2:])

        # 3) 三通道地形判定（v3 语义 + 可靠区守卫）
        pos_mask = rel & (h > self.pos) & (h < self.max_h) & rng
        neg_mask = rel & (h < -self.drop) & rng
        k = 3  # 24cm 基线中心差分，抗单格噪声
        gx = np.full_like(elev, np.nan)
        gy = np.full_like(elev, np.nan)
        gx[k:-k, :] = (elev[2 * k:, :] - elev[:-2 * k, :]) / (2 * k * res)
        gy[:, k:-k] = (elev[:, 2 * k:] - elev[:, :-2 * k]) / (2 * k * res)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        neg_mask |= (np.isfinite(slope) & (slope > self.slope_deg)
                     & rel & (np.abs(h) <= self.pos) & rng)

        # 3.5) 障碍记忆过期（幽灵污染根治）: 调试/作业时人员走动的残影会被高程图
        # 永久记住并经 360° scan 永续喂给 costmap（实测 42% 车周致命格为幽灵,
        # 且每 ~30 分钟复发一次把车"软件围死"）。按 time 层给记忆设寿命。
        tlay = self._layer_matrix(m, 'time')
        if tlay is not None:
            pos_ttl = np.where(d2 < self.near_ring ** 2,
                               self.pos_memory_ttl_near, self.pos_memory_ttl)
            pos_mask &= (tlay <= pos_ttl)
            neg_mask &= (tlay <= self.neg_memory_ttl)
        elif not self._time_layer_warned:
            self._time_layer_warned = True
            rospy.logwarn('lidar_obstacle_scan: 高程图未发布 time 层, 障碍记忆过期未生效'
                          '(请在 elevation_single.yaml publishers.layers 加 time)')

        # 4) 世界格 N 帧去抖（v3 原样）
        def debounce(mask, hist, need):
            keys = set(zip((X[mask] / res).astype(np.int32).tolist(),
                           (Y[mask] / res).astype(np.int32).tolist()))
            for kk in list(hist.keys()):
                if kk in keys:
                    hist[kk] = min(hist[kk] + 1, need + 2)
                else:
                    hist[kk] -= self.confirm_miss_decay
                    if hist[kk] <= 0:
                        del hist[kk]
            for kk in keys:
                hist.setdefault(kk, 1)
            ok = {kk for kk, v in hist.items() if v >= need}
            conf = np.zeros_like(mask)
            if ok:
                Xi = (X / res).astype(np.int64)
                Yi = (Y / res).astype(np.int64)
                flat = Xi * 1000000 + Yi
                okf = np.array([a * 1000000 + b for a, b in ok], dtype=np.int64)
                conf = np.isin(flat, okf) & mask
            return conf

        neg_mask = debounce(neg_mask, self._neg_hist, self.neg_confirm)
        pos_mask = debounce(pos_mask, self._pos_hist, self.pos_confirm)

        # 5) 障碍格 → 车体系虚拟 LaserScan（对齐视觉链契约：空束=inf）
        obst = pos_mask | neg_mask
        ranges = np.full(self.n_beams, np.inf, dtype=np.float32)
        n_obst = int(obst.sum())
        if n_obst:
            dx = X[obst] - bx
            dy = Y[obst] - by
            c, s = math.cos(-heading), math.sin(-heading)
            fx = c * dx - s * dy
            fy = s * dx + c * dy
            r = np.hypot(fx, fy) - self.safety_margin
            ang = np.arctan2(fy, fx)
            idx = np.round((ang - self.angle_min) / self.angle_inc).astype(np.int64)
            keep = (idx >= 0) & (idx < self.n_beams) & \
                   (r >= self.range_min) & (r <= self.range_max)
            np.minimum.at(ranges, idx[keep], r[keep].astype(np.float32))

        # 2026-07-30 旋转判撞修复: stamp 必须=位姿采样时刻(回调入口 TF 的时间), 不能用 now()。
        # 此前 now() 比 TF 采样晚 0.1~0.2s(numpy 处理耗时+消息龄), 下游快照层按 stamp 查
        # 1X 树 TF 把点反变换回世界系——旋转 0.75rad/s 时两次采样差=4~9° 切向错位
        # (2m 处 15~30cm), 每帧方向随机、又被 hold_time 3s 锁存成致命弧, 恰好填进
        # 0.36~0.52m 的旋转扫掠环 = "现实能转/软件判撞"的主因。改用 TF 时刻后快照层
        # 查的是同一时刻的位姿; 且查过去时刻的 TF 缓存里已有, 不再需要等 1X TF 追上 now。
        stamp = t.header.stamp
        if stamp.is_zero():
            stamp = rospy.Time.now()
        scan = LaserScan()
        scan.header = Header(stamp=stamp, frame_id=self.output_frame)
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_min + (self.n_beams - 1) * self.angle_inc
        scan.angle_increment = self.angle_inc
        scan.time_increment = 0.0
        scan.scan_time = 0.1  # 随高程图 10Hz
        scan.range_min = self.range_min
        scan.range_max = self.range_max
        scan.ranges = ranges.tolist()
        scan.intensities = []
        self.pub_scan.publish(scan)

        # ---- 走廊系底料（限速与脱困共用）----
        # 2026-07-30 与限速开关解耦: 此前 _rear_free/_pose_hist 只在限速分支内更新,
        # 关掉 enable_ground_speed_limit 会让自动脱困静默失效(永远判非静止)。
        csl, ssl = math.cos(-heading), math.sin(-heading)
        fxa = csl * (X - bx) - ssl * (Y - by)
        fya = ssl * (X - bx) + csl * (Y - by)
        ground_ok = rel & (h > -self.drop) & (h < self.pos)  # 可靠区内±阈值=已验证地面
        bad = neg_mask | pos_mask                            # 已去抖的障碍
        # —— 自动脱困素材（1Hz watchdog 消费）——
        in_rear = (np.abs(fya) <= self.cor_half_w) & (fxa <= -0.42) & (fxa >= -1.7)
        n_r = int(in_rear.sum())
        self._rear_free = bool(
            n_r > 40
            and int((in_rear & bad).sum()) == 0
            and int((in_rear & ground_ok).sum()) >= max(20, 0.3 * n_r))
        tnow = rospy.get_time()
        self._pose_hist.append((tnow, bx, by))
        self._pose_hist = [p for p in self._pose_hist if tnow - p[0] < 3.0]

        # ---- 第三招：地面确认限速（不要开得比验证过的地面更快）----
        # 车前走廊按纵向切片，逐片要求"有效地面格达标且无障碍"，得到连续验证
        # 距离 D；负障碍/陡坡/正障碍/无数据空洞任何一种都会截断 D。
        if self.enable_speed_limit:
            in_cor = (np.abs(fya) <= self.cor_half_w) & \
                     (fxa >= self.cor_start) & (fxa <= self.max_range)
            n_bins = max(1, int((self.max_range - self.cor_start) / self.cor_bin))
            bidx = np.clip(((fxa - self.cor_start) / self.cor_bin).astype(np.int32),
                           0, n_bins - 1)
            cnt_all = np.bincount(bidx[in_cor], minlength=n_bins)
            cnt_gnd = np.bincount(bidx[in_cor & ground_ok], minlength=n_bins)
            cnt_bad = np.bincount(bidx[in_cor & bad], minlength=n_bins)
            D = self.cor_start
            for i in range(n_bins):
                need = max(self.cor_min_cells, self.cor_frac * cnt_all[i])
                if cnt_bad[i] > 0 or cnt_gnd[i] < need:
                    break
                D = self.cor_start + (i + 1) * self.cor_bin
            # 完整制动方程: v·t_r + v²/(2a) ≤ D - 余量  →  解 v
            usable = max(0.0, D - self.sl_margin)
            at = self.sl_acc * self.sl_react
            v = -at + math.sqrt(at * at + 2.0 * self.sl_acc * usable)
            v = max(self.sl_v_min, min(self.sl_v_max, v))
            self.pub_vlimit.publish(Float32(data=v))
            self._send_dwa_limit(v)
            rospy.loginfo_throttle(
                5.0, 'lidar_obstacle_scan: 验证地面 D=%.2fm → 限速 %.2fm/s', D, v)

        if self.publish_debug_cloud:
            c, s = math.cos(-heading), math.sin(-heading)

            def cloud(mask, z_val=None):
                if not int(mask.sum()):
                    return np.zeros((0, 3), np.float32)
                dx, dy = X[mask] - bx, Y[mask] - by
                z = h[mask] if z_val is None else np.full(int(mask.sum()), z_val, np.float32)
                return np.column_stack([c * dx - s * dy, s * dx + c * dy, z])

            pts = np.vstack([cloud(pos_mask), cloud(neg_mask, 0.5)])
            self.pub_cloud.publish(pcl2.create_cloud(
                Header(stamp=stamp, frame_id=self.output_frame), CLOUD_FIELDS, pts.tolist()))

        if int(neg_mask.sum()):
            rospy.loginfo_throttle(5.0, 'lidar_obstacle_scan: 负障碍/陡坡格 %d', int(neg_mask.sum()))
        if int(pos_mask.sum()):
            rospy.loginfo_throttle(10.0, 'lidar_obstacle_scan: 正障碍格 %d', int(pos_mask.sum()))


if __name__ == '__main__':
    rospy.init_node('lidar_obstacle_scan_node')
    LidarObstacleScan()
    rospy.spin()
