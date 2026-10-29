#!/usr/bin/env python3
"""YOLO + point-cloud pose estimation (low-lag, base-frame accurate).

For every YOLO box, the depth pixels inside it are deprojected into a small point
cloud (camera colour frame), cleaned and fitted:
    ball -> fixed-radius sphere fit      -> sphere CENTRE
    box  -> centroid of the visible face

Published (both carry the IMAGE timestamp, not the publish time):
    detected_objects       frame = global_frame (map)   -> Nav2 / far targets
    detected_objects_base  frame = base_frame (base_link) -> fine align + pick

The base_link pose is ONE direct TF hop camera -> base_link at the image time,
so it does not depend on map / odom at all. An optional rigid correction
(corr_R, corr_t) fixes a wrongly measured camera mount; see the calibration
snippet in the chat answer.

Lag fixes compared with the old version:
  * the sync callback only stores the NEWEST frame pair; a worker thread
    processes it, so the node never works through a queue of old frames;
  * subscription queue depth is 1;
  * inference size / device are parameters (smaller imgsz = faster);
  * the debug image is only published when somebody subscribes;
  * every few seconds the log prints processing time and image age.

Requires depth ALIGNED to colour (RealSense: align_depth.enable:=true).
"""

import math
import threading
import time
import traceback

import cv2
import image_geometry
import message_filters
import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transforms)
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from object_interfaces.msg import DetectedObject
from object_interfaces.msg import DetectedObjectArray
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (HistoryPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from ultralytics import YOLO

BALL_RADIUS_M = 0.030
BALL_SHRINK = 0.6
BOX_SHRINK = 0.6
BALL_Z_BAND = 0.05
BOX_Z_BAND = 0.15
DEPTH_RANGE_M = (0.10, 5.0)     # depth below the camera's real minimum range is noise
MIN_BALL_POINTS = 30
MIN_BOX_POINTS = 100
SPHERE_MAX_RMS = 0.008


def cloud_from_bbox(depth, bbox, shrink, intr):
    """Deproject the central `shrink` fraction of a bbox -> Nx3 points (camera frame)."""
    x1, x2, y1, y2 = bbox
    h, w = depth.shape[:2]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = (x2 - x1) / 2.0 * shrink, (y2 - y1) / 2.0 * shrink
    x0, x3 = max(0, int(cx - hw)), min(w, int(cx + hw) + 1)
    y0, y3 = max(0, int(cy - hh)), min(h, int(cy + hh) + 1)

    d = depth[y0:y3, x0:x3].astype(np.float32)
    if depth.dtype == np.uint16:
        d *= 0.001
    us, vs = np.meshgrid(np.arange(x0, x3), np.arange(y0, y3))
    ok = np.isfinite(d) & (d > DEPTH_RANGE_M[0]) & (d < DEPTH_RANGE_M[1])
    z = d[ok]
    fx, fy, ppx, ppy = intr
    return np.column_stack(((us[ok] - ppx) * z / fx, (vs[ok] - ppy) * z / fy, z))


def clean_cloud(pts, z_band):
    """Pass-through on depth around the median, then statistical outlier removal."""
    if len(pts) == 0:
        return pts
    pts = pts[np.abs(pts[:, 2] - np.median(pts[:, 2])) < z_band]
    if len(pts) < 10:
        return pts
    dist = np.linalg.norm(pts - np.median(pts, axis=0), axis=1)
    med = np.median(dist)
    mad = np.median(np.abs(dist - med))
    return pts[dist <= med + 3.0 * 1.4826 * mad + 1e-6]


def fit_sphere(pts, radius, iters=10):
    """Gauss-Newton fit of a sphere centre with KNOWN radius. Returns (centre, rms)."""
    mean = pts.mean(axis=0)
    c = mean + mean / np.linalg.norm(mean) * radius
    for _ in range(iters):
        d = pts - c
        n = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
        step, *_ = np.linalg.lstsq(-d / n[:, None], -(n - radius), rcond=None)
        c = c + step
        if np.linalg.norm(step) < 1e-5:
            break
    res = np.linalg.norm(pts - c, axis=1) - radius
    return c, float(np.sqrt(np.mean(res ** 2)))


def estimate_center(kind, depth, bbox, intr):
    """3-D position (camera frame) of the object, or None if the cloud is too thin."""
    if kind == "ball":
        pts = clean_cloud(cloud_from_bbox(depth, bbox, BALL_SHRINK, intr), BALL_Z_BAND)
        if len(pts) < MIN_BALL_POINTS:
            return None
        centre, rms = fit_sphere(pts, BALL_RADIUS_M)
        if rms > SPHERE_MAX_RMS:
            m = pts.mean(axis=0)
            centre = m + m / np.linalg.norm(m) * BALL_RADIUS_M
        return centre
    pts = clean_cloud(cloud_from_bbox(depth, bbox, BOX_SHRINK, intr), BOX_Z_BAND)
    if len(pts) < MIN_BOX_POINTS:
        return None
    return pts.mean(axis=0)


class PoseEstimation(Node):

    def __init__(self):
        super().__init__("pose_estimation_3d")

        self.declare_parameter("rgb_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("yolo_model_path", "/home/startrace1/best.pt")
        self.declare_parameter("conf_threshold", 0.6)
        self.declare_parameter("dedup_dist", 0.10)      # same-class detections closer than this = one object
        self.declare_parameter("ball_classes", ["ball"])
        self.declare_parameter("box_classes", ["box"])
        self.declare_parameter("imgsz", 416)            # smaller = faster (320 / 416 / 640)
        self.declare_parameter("device", "cpu")         # "cpu" or "0" for a CUDA GPU
        self.declare_parameter("max_det", 10)
        self.declare_parameter("publish_debug", True)
        # optional rigid correction applied to base_link points: p = R @ p + t
        self.declare_parameter("corr_R", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("corr_t", [0.0, 0.0, 0.0])

        self.global_frame = self.get_parameter("global_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.conf = self.get_parameter("conf_threshold").value
        self.dedup_dist = self.get_parameter("dedup_dist").value
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.device = str(self.get_parameter("device").value)
        self.max_det = int(self.get_parameter("max_det").value)
        self.publish_debug = bool(self.get_parameter("publish_debug").value)
        self.corr_R = np.array(self.get_parameter("corr_R").value, dtype=float).reshape(3, 3)
        self.corr_t = np.array(self.get_parameter("corr_t").value, dtype=float)

        self.class_kinds = {}
        for kind in ("ball", "box"):
            for name in self.get_parameter(f"{kind}_classes").value:
                self.class_kinds[name.lower()] = kind

        self.bridge = CvBridge()
        self.cam_model = None
        self.model = YOLO(self.get_parameter("yolo_model_path").value, task="detect")

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.objects_pub = self.create_publisher(DetectedObjectArray, "detected_objects", 10)
        self.base_pub = self.create_publisher(DetectedObjectArray, "detected_objects_base", 10)
        self.debug_pub = self.create_publisher(Image, "detection_image", 10)
        self.create_subscription(
            CameraInfo, self.get_parameter("camera_info_topic").value,
            self._info_cb, qos_profile_sensor_data)

        # depth-1 queues: never keep old images around
        img_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        rgb_sub = message_filters.Subscriber(
            self, Image, self.get_parameter("rgb_topic").value, qos_profile=img_qos)
        depth_sub = message_filters.Subscriber(
            self, Image, self.get_parameter("depth_topic").value, qos_profile=img_qos)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=5, slop=0.05)
        self.sync.registerCallback(self._frame_cb)

        # newest frame pair + worker thread
        self._cond = threading.Condition()
        self._latest = None
        threading.Thread(target=self._worker, daemon=True).start()

        self.get_logger().info(f"Model classes: {self.model.names}")
        self.get_logger().info(
            f"PoseEstimation ready. Class map: {self.class_kinds} "
            f"imgsz={self.imgsz} device={self.device}")

    # ------------------------------------------------------------------
    def _info_cb(self, msg: CameraInfo):
        if self.cam_model is None:
            model = image_geometry.PinholeCameraModel()
            model.fromCameraInfo(msg)
            self.cam_model = model

    def _frame_cb(self, rgb_msg: Image, depth_msg: Image):
        """Only remember the newest pair; never do heavy work here."""
        with self._cond:
            self._latest = (rgb_msg, depth_msg)
            self._cond.notify()

    def _worker(self):
        while rclpy.ok():
            with self._cond:
                while self._latest is None and rclpy.ok():
                    self._cond.wait(timeout=0.5)
                if self._latest is None:
                    continue
                rgb_msg, depth_msg = self._latest
                self._latest = None
            if self.cam_model is None:
                continue
            t0 = time.monotonic()
            try:
                self._process(rgb_msg, depth_msg)
            except Exception as e:
                self.get_logger().error(
                    f"Frame processing failed: {e!r}\n{traceback.format_exc()}")
                continue
            dt = time.monotonic() - t0
            age = (self.get_clock().now() - Time.from_msg(rgb_msg.header.stamp)).nanoseconds * 1e-9
            self.get_logger().info(
                f"processing {dt * 1000:.0f} ms | image age at publish {age * 1000:.0f} ms",
                throttle_duration_sec=3.0)

    def class_name(self, cls_id: int) -> str:
        names = self.model.names
        if isinstance(names, dict):
            return names.get(cls_id, f"class_{cls_id}")
        return names[cls_id] if cls_id < len(names) else f"class_{cls_id}"

    def transform_point(self, p, src, stamp, dst):
        """ONE TF hop of point `p` from `src` to `dst`, evaluated at `stamp`."""
        pt = PointStamped()
        pt.header.frame_id = src
        pt.header.stamp = stamp
        pt.point.x, pt.point.y, pt.point.z = float(p[0]), float(p[1]), float(p[2])
        try:
            out = self.tf_buffer.transform(pt, dst, timeout=Duration(seconds=0.2))
        except TransformException as e:
            self.get_logger().warn(f"TF {src}->{dst} failed: {e}", throttle_duration_sec=2.0)
            return None
        return np.array([out.point.x, out.point.y, out.point.z])

    @staticmethod
    def make_obj(name, conf, p):
        obj = DetectedObject()
        obj.class_name = name
        obj.confidence = conf
        obj.pose.position.x, obj.pose.position.y, obj.pose.position.z = map(float, p)
        obj.pose.orientation.w = 1.0
        return obj

    def _process(self, rgb_msg: Image, depth_msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except CvBridgeError as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return
        if depth.shape[:2] != bgr.shape[:2]:
            self.get_logger().error(
                f"Depth {depth.shape[:2]} != RGB {bgr.shape[:2]}: use aligned_depth_to_color",
                throttle_duration_sec=5.0)
            return

        m = self.cam_model
        intr = (m.fx(), m.fy(), m.cx(), m.cy())
        frame = rgb_msg.header.frame_id
        stamp = rgb_msg.header.stamp               # IMAGE time
        result = self.model(bgr, imgsz=self.imgsz, conf=self.conf, device=self.device,
                            max_det=self.max_det, verbose=False)[0]

        out_map = DetectedObjectArray()
        out_map.header.stamp = stamp
        out_map.header.frame_id = self.global_frame
        out_base = DetectedObjectArray()
        out_base.header.stamp = stamp
        out_base.header.frame_id = self.base_frame

        for box in result.boxes:
            name = self.class_name(int(box.cls[0]))
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            kind = self.class_kinds.get(name.lower(), "other")

            centre = estimate_center(kind, depth, (x1, x2, y1, y2), intr)
            label = f"{name} {conf:.2f}"
            color = (0, 255, 0)
            if centre is None:
                color, label = (0, 0, 255), label + " no depth"
            else:
                label += f" z={centre[2]:.2f}m"
            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(bgr, label, (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if centre is None:
                continue

            # camera -> base_link: one hop at the image time, then optional correction
            p_base = self.transform_point(centre, frame, stamp, self.base_frame)
            if p_base is None:
                continue
            p_base = self.corr_R @ p_base + self.corr_t

            # base_link -> map (same image time) so both topics describe the same point
            p_map = self.transform_point(p_base, self.base_frame, stamp, self.global_frame)
            if p_map is None:
                continue

            if any(o.class_name == name and
                   math.hypot(o.pose.position.x - p_base[0],
                              o.pose.position.y - p_base[1]) < self.dedup_dist
                   for o in out_base.objects):
                continue

            out_base.objects.append(self.make_obj(name, conf, p_base))
            out_map.objects.append(self.make_obj(name, conf, p_map))
            if kind == "ball":
                self.get_logger().info(
                    f"BALL cam=({centre[0]:.3f},{centre[1]:.3f},{centre[2]:.3f}) "
                    f"base=({p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f})",
                    throttle_duration_sec=1.0)

        if out_base.objects:
            self.base_pub.publish(out_base)
            self.objects_pub.publish(out_map)
        if self.publish_debug and self.debug_pub.get_subscription_count() > 0:
            debug = self.bridge.cv2_to_imgmsg(bgr, encoding="passthrough")
            debug.header = rgb_msg.header
            self.debug_pub.publish(debug)


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimation()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()