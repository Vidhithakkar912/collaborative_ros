#!/usr/bin/env python3


import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
import cv2
import numpy as np
import threading
import json
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose, PointStamped
from cv_bridge import CvBridge, CvBridgeError
import image_geometry
import math
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from rclpy.duration import Duration
from ultralytics import YOLO
from sensor_msgs.msg import Image as RosImage
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from object_interfaces.msg import DetectedObject
from object_interfaces.msg import DetectedObjectArray


class DetectionInfo:

    def __init__(self, cls_name, confidence, bbox):
        self.cls_name = cls_name
        self.confidence = confidence
        self.bbox = bbox


def detect_rectangles_hsv(image: np.ndarray,
                           min_area: int = 5000) -> list[list[int]]:

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_brown = np.array([10, 80, 40])
    upper_brown = np.array([20, 200, 180])

    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return []

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area < min_area:
        return []

    x, y, w, h = cv2.boundingRect(cnt)
    return [[x, x + w, y, y + h]]


def detect_rectangles_yolo(model, image: np.ndarray):

    results = model(image, conf=0.6, verbose=False)

    detections = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "class": class_name,
                "confidence": conf,
                "bbox": [x1, x2, y1, y2],
            })

    return detections


class PoseEstimation(Node):

    def __init__(self):
        super().__init__("pose_estimation_depth")
        self.get_logger().info("=== PoseEstimation node initialising ===")

        self.declare_parameter("to_meter", 1.0)
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("min_bbox_area", 5000)
        self.declare_parameter("approx_epsilon_ratio", 0.02)
      
        self.declare_parameter("dedup_dist_thresh", 0.4)

        self.to_meter_ = self.get_parameter("to_meter").value
        self.global_frame_ = self.get_parameter("global_frame").value
        self.min_bbox_area_ = self.get_parameter("min_bbox_area").value
        self.dedup_dist_thresh_ = self.get_parameter("dedup_dist_thresh").value

        self.model = YOLO(
            "/home/einfochips/Downloads/cardbox/runs/detect/box_detector/best.pt")
        self.latest_rgb = None

        ns = self.get_namespace().strip("/")
        prefix = f"/{ns}" if ns else ""

        depth_topic = f"{prefix}/camera/camera/depth/image_rect_raw"
        rgb_topic = f"{prefix}/camera/camera/color/image_raw"
        caminfo_topic = f"{prefix}/camera/camera/depth/camera_info"
        pose_topic = f"{prefix}/box_poses"

        self.optical_frame_ = "camera_depth_optical_frame"
        self.get_logger().info(f"  depth topic    = {depth_topic}")
        self.get_logger().info(f"  rgb topic      = {rgb_topic}")
        self.get_logger().info(f"  caminfo topic  = {caminfo_topic}")
        self.get_logger().info(f"  pose pub topic = {pose_topic}")
        self.get_logger().info(f"  global_frame   = {self.global_frame_}")
        self.get_logger().info(f"  optical_frame  = {self.optical_frame_}")

        self._cbg_depth = MutuallyExclusiveCallbackGroup()
        self._cbg_rgb = MutuallyExclusiveCallbackGroup()

        self._bridge = CvBridge()
        self._detections = []
        self._mtx = threading.Lock()
       
        self.target_class = None
        self._tf_buffer = Buffer(cache_time=Duration(seconds=10))
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._debug_img_pub = self.create_publisher(
            Image, f"{prefix}/detection_image", 10)
        self.detected_object_pub = self.create_publisher(
            DetectedObjectArray, "detected_objects", 10)

        tf_ready = False
        for _ in range(20):
            if self._tf_buffer.can_transform(
                    self.global_frame_, self.optical_frame_,
                    rclpy.time.Time(), timeout=Duration(seconds=0.5)):
                tf_ready = True
                break
            self.get_logger().warn("Waiting for TF tree...")
            rclpy.spin_once(self, timeout_sec=0.2)

        if tf_ready:
            self.get_logger().info(
                f"TF chain OK ✓ '{self.optical_frame_}'→'{self.global_frame_}'")
        else:
            self.get_logger().error(
                f"TF chain STILL MISSING: "
                f"'{self.optical_frame_}'→'{self.global_frame_}'"
            )

        self.get_logger().info(
            "TF buffer feeds: /tf + /tf_static (transient) + "
            "/tb1/tf + /tb1/tf_static (transient)")

        self.get_logger().info(f"Waiting for CameraInfo on '{caminfo_topic}' …")
        cam_info_msg = self._wait_for_camera_info(caminfo_topic)

        self._cam_model = image_geometry.PinholeCameraModel()
        self._cam_model.fromCameraInfo(cam_info_msg)
        self.get_logger().info(
            f"Camera model OK: fx={self._cam_model.fx():.2f} "
            f"fy={self._cam_model.fy():.2f} "
            f"cx={self._cam_model.cx():.2f} cy={self._cam_model.cy():.2f}"
        )

        self.get_logger().info(
            f"Verifying TF chain: '{self.optical_frame_}' → '{self.global_frame_}' …")
        if self._tf_buffer.can_transform(
                self.global_frame_, self.optical_frame_,
                rclpy.time.Time(), timeout=Duration(seconds=5.0)):
            self.get_logger().info(
                f"TF chain OK ✓  '{self.optical_frame_}'→'{self.global_frame_}'")
        else:
            self.get_logger().error(
                f"TF chain STILL MISSING: '{self.optical_frame_}'→'{self.global_frame_}'\n"
                f"Run:  ros2 run tf2_tools view_frames\n"
                f"and check that map→odom is published on /tf by Nav2/AMCL."
            )

        self._rgb_sub = self.create_subscription(
            Image, rgb_topic, self._rgb_callback, 10,
            callback_group=self._cbg_rgb)
        self._depth_sub = self.create_subscription(
            Image, depth_topic, self._depth_img_callback, 10,
            callback_group=self._cbg_depth)
        self.create_subscription(
            String, "object_target", self.target_callback, 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, "detected_box_markers", 10)

        self.get_logger().info("=== PoseEstimation node ready ===")

    def target_callback(self, msg):
        self.target_class = msg.data.lower()
        self.get_logger().info(f"Target object set to {self.target_class}")

    def _wait_for_camera_info(self, topic: str) -> CameraInfo:
        event = threading.Event()
        received = {}

        def _cb(msg):
            received["msg"] = msg
            event.set()

        sub = self.create_subscription(CameraInfo, topic, _cb, 1)
        tmp_executor = rclpy.executors.SingleThreadedExecutor()
        tmp_executor.add_node(self)

        while rclpy.ok() and not event.is_set():
            tmp_executor.spin_once(timeout_sec=0.1)

        tmp_executor.remove_node(self)
        tmp_executor.shutdown()
        self.destroy_subscription(sub)

        if "msg" not in received:
            raise RuntimeError("CameraInfo never received")
        self.get_logger().info("CameraInfo received successfully.")
        return received["msg"]

    def _rgb_callback(self, msg: Image):
        self.get_logger().info("rgb image is received")
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_rgb = cv_img.copy()
        except CvBridgeError as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        self.rgb_h, self.rgb_w = cv_img.shape[:2]

        try:
            detections = detect_rectangles_yolo(self.model, cv_img)
        except Exception as e:
            self.get_logger().error(f"detect_rectangles failed: {e}")
            return

        for det in detections:
            xmin, xmax, ymin, ymax = det["bbox"]
            cls_name = det["class"]
            conf = det["confidence"]

            cv2.rectangle(cv_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(
                cv_img, f"{cls_name} {conf:.2f}", (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        try:
            annotated_msg = self._bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
            annotated_msg.header = msg.header
            self._debug_img_pub.publish(annotated_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Publish debug image failed: {e}")

        with self._mtx:
            self._detections = detections if len(detections) > 0 else []

    def _depth_img_callback(self, msg: Image):
        self.get_logger().info("depth callback is received")
        try:
            cv_depth = self._bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            self.get_logger().error(f"Depth conversion failed: {e}")
            return

        with self._mtx:
            if not self._detections:
                return
            detection_snapshot = list(self._detections)

        scaled_bboxes = detection_snapshot
        if not hasattr(self, "rgb_w"):
            return

        depth_h, depth_w = cv_depth.shape[:2]

        detectedmsg = DetectedObjectArray()
        detectedmsg.header.stamp = self.get_clock().now().to_msg()
        detectedmsg.header.frame_id = self.global_frame_

        for det in scaled_bboxes:

            xmin, xmax, ymin, ymax = det["bbox"]
            class_name = det["class"]

            cx_pix = int((xmin + xmax) / 2)
            cy_pix = int((ymin + ymax) / 2)

            self.get_logger().info(f"cx_pix{cx_pix}::cypix::{cy_pix}")

            roi = cv_depth[ymin:ymax, xmin:xmax]
            valid_depths = roi[np.isfinite(roi) & (roi > 0)]

            if valid_depths.size == 0:
                self.get_logger().warn(f"BBox[{det}] no valid depth — skipping")
                continue

            depth = float(np.median(valid_depths)) / 1000.0

            fx = self._cam_model.fx()
            fy = self._cam_model.fy()
            ppx = self._cam_model.cx()
            ppy = self._cam_model.cy()

            cx = (cx_pix - ppx) * depth / fx
            cy = (cy_pix - ppy) * depth / fy
            cz = depth

            self.get_logger().info(
                f"BBox[{det}] centroid optical frame: "
                f"x={cx:.3f} y={cy:.3f} z={cz:.3f}"
            )

            pt = PointStamped()
            pt.header.stamp = msg.header.stamp
            pt.header.frame_id = self.optical_frame_
            pt.point.x = cx
            pt.point.y = cy
            pt.point.z = cz

            try:
                if not self._tf_buffer.can_transform(
                        self.global_frame_, self.optical_frame_,
                        msg.header.stamp, timeout=Duration(seconds=0.5)):
                    self.get_logger().warn("TF unavailable")
                    continue

                transform = self._tf_buffer.lookup_transform(
                    self.global_frame_, self.optical_frame_,
                    msg.header.stamp, timeout=Duration(seconds=0.5))

                pt_out = tf2_geometry_msgs.do_transform_point(pt, transform)

            except Exception as e:
                self.get_logger().warn(f"TF transform failed: {e}")
                continue

            self.get_logger().info(
                f"BBox[{det}] in map: "
                f"x={pt_out.point.x:.3f} y={pt_out.point.y:.3f} z={pt_out.point.z:.3f}"
            )

            obj = DetectedObject()
            obj.class_name = class_name
            obj.confidence = float(det["confidence"])
            obj.pose.position.x = pt_out.point.x
            obj.pose.position.y = pt_out.point.y
            obj.pose.position.z = pt_out.point.z
            obj.pose.orientation.w = 1.0

            duplicate = False
            for existing in detectedmsg.objects:
                if existing.class_name != obj.class_name:
                    continue
                dist = math.sqrt(
                    (existing.pose.position.x - obj.pose.position.x) ** 2 +
                    (existing.pose.position.y - obj.pose.position.y) ** 2
                )
                if dist < self.dedup_dist_thresh_:
                    duplicate = True
                    break

            if duplicate:
                continue

            detectedmsg.objects.append(obj)

        if len(detectedmsg.objects) > 0:
            self.get_logger().info(
                f"Publishing {len(detectedmsg.objects)} object pose(s): "
                f"{[o.class_name for o in detectedmsg.objects]}"
            )
            self.detected_object_pub.publish(detectedmsg)

        if self.latest_rgb is not None:
            debug_msg = self._bridge.cv2_to_imgmsg(
                self.latest_rgb, encoding="bgr8")
            debug_msg.header = msg.header
            self._debug_img_pub.publish(debug_msg)


def main(args=None):
    print("=== pose_estimation_depth: entering main ===")
    rclpy.init(args=args)
    node = PoseEstimation()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        print("Spinning … (Ctrl-C to exit)")
        executor.spin()
    except KeyboardInterrupt:
        print("KeyboardInterrupt — shutting down")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("pose_estimation_depth: shutdown complete")


if __name__ == "__main__":
    main()