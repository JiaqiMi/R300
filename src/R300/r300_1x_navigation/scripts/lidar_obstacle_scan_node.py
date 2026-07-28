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
     的每障碍 TTL(3s) 自动过期，无需主动清图。
"""

import math

import numpy as np
import rospy
import tf2_ros
from grid_map_msgs.msg import GridMap
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pcl2
from std_msgs.msg import Header

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

        self._neg_hist = {}
        self._pos_hist = {}
        self._last_msg_time = rospy.Time(0)
        self.buf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.buf)

        self.pub_scan = rospy.Publisher(self.scan_topic, LaserScan, queue_size=1)
        self.pub_cloud = rospy.Publisher(self.debug_cloud_topic, PointCloud2, queue_size=1)
        rospy.Subscriber(self.elevation_topic, GridMap, self.cb, queue_size=1,
                         buff_size=64 * 1024 * 1024)
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

        # 3) 三通道地形判定（v3 语义原样移植）
        pos_mask = fin & (h > self.pos) & (h < self.max_h) & rng
        neg_mask = fin & (h < -self.drop) & rng
        k = 3  # 24cm 基线中心差分，抗单格噪声
        gx = np.full_like(elev, np.nan)
        gy = np.full_like(elev, np.nan)
        gx[k:-k, :] = (elev[2 * k:, :] - elev[:-2 * k, :]) / (2 * k * res)
        gy[:, k:-k] = (elev[:, 2 * k:] - elev[:, :-2 * k]) / (2 * k * res)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        neg_mask |= (np.isfinite(slope) & (slope > self.slope_deg)
                     & fin & (np.abs(h) <= self.pos) & rng)

        # 4) 世界格 N 帧去抖（v3 原样）
        def debounce(mask, hist, need):
            keys = set(zip((X[mask] / res).astype(np.int32).tolist(),
                           (Y[mask] / res).astype(np.int32).tolist()))
            for kk in list(hist.keys()):
                if kk in keys:
                    hist[kk] = min(hist[kk] + 1, need + 2)
                else:
                    hist[kk] -= 1
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
