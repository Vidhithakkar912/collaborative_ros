#!/usr/bin/env python3


import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import cv2
import numpy as np
import threading

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
from collections import defaultdict


def detect_rectangles_hsv(image: np.ndarray,
                      min_area: int = 5000) -> list[list[int]]:

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # brown / cardboard color range
    lower_brown = np.array([10, 80, 40])
    upper_brown = np.array([20, 200, 180])

    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # remove noise
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # no contour found
    if len(contours) == 0:
        return []

    # take largest contour only
    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)

    if area < min_area:
        return []

    x, y, w, h = cv2.boundingRect(cnt)

    return [[x, x + w, y, y + h]]
 
def detect_rectangles_yolo(model,
                      image: np.ndarray) -> list[list[int]]:
    
    results = model(image,conf=0.6,verbose=False)
   
    bboxes = []

    for r in results:

        

        for box in r.boxes:


            conf = float(box.conf[0])
            print(f"confidence is :{conf}")
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            bboxes.append([x1, x2, y1, y2])

    return bboxes


class PoseEstimation(Node):

    def __init__(self):
        super().__init__("pose_estimation_depth")
        self.get_logger().info("=== PoseEstimation node initialising ===")

        self.declare_parameter("to_meter",             1.0)
        self.declare_parameter("global_frame",         "map")
        self.declare_parameter("min_bbox_area",        5000)
        self.declare_parameter("approx_epsilon_ratio", 0.02)
        
        self.to_meter_         = self.get_parameter("to_meter").value
        self.global_frame_     = self.get_parameter("global_frame").value
        self.min_bbox_area_    = self.get_parameter("min_bbox_area").value
        self.model = YOLO("/home/einfochips/Downloads/cardbox/runs/detect/box_detector/yolov8s_box/weights/best.pt")
        self.latest_rgb = None
        self.track_history = defaultdict(list)

        self.STABLE_FRAMES = 5

        self.MAX_POSITION_JUMP = 0.20      # meters

        self.MAX_HISTORY = 10
                

        ns = self.get_namespace().strip("/")

        if ns:
            prefix = f"/{ns}"
        else:
            prefix = ""

        depth_topic   = f"{prefix}/camera/camera/depth/image_rect_raw"
        rgb_topic     = f"{prefix}/camera/camera/color/image_raw"
        caminfo_topic = f"{prefix}/camera/camera/depth/camera_info"
        pose_topic    = f"{prefix}/box_poses"
        

        
        self.optical_frame_ = "camera_depth_optical_frame"
        self.get_logger().info(f"  depth topic    = {depth_topic}")
        self.get_logger().info(f"  rgb topic      = {rgb_topic}")
        self.get_logger().info(f"  caminfo topic  = {caminfo_topic}")
        self.get_logger().info(f"  pose pub topic = {pose_topic}")
        self.get_logger().info(f"  global_frame   = {self.global_frame_}")
        self.get_logger().info(f"  optical_frame  = {self.optical_frame_}")

        self._cbg_depth = MutuallyExclusiveCallbackGroup()
        self._cbg_rgb   = MutuallyExclusiveCallbackGroup()

        self._bridge = CvBridge()
        self._bboxes = []
        self._mtx    = threading.Lock()

        self._tf_buffer   = Buffer(cache_time=Duration(seconds=10))
        self._tf_listener = TransformListener(
             self._tf_buffer,
             self
             )
        

        self.detected_objects = []
        self.current_target = None
        self._debug_img_pub = self.create_publisher(
             Image, f"{prefix}/detection_image", 10
         )
        tf_ready = False

        for _ in range(20):

            if self._tf_buffer.can_transform(
                    self.global_frame_,
                    self.optical_frame_,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.5)):

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
                rclpy.time.Time(),
                timeout=Duration(seconds=5.0)):
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
        self.marker_pub = self.create_publisher(
        MarkerArray,
        "detected_box_markers",
        10
    )
        self._pose_pub = self.create_publisher(PoseArray, pose_topic, 10)
        self.get_logger().info("=== PoseEstimation node ready ===")

    def _wait_for_camera_info(self, topic: str) -> CameraInfo:
        event    = threading.Event()
        received = {}

        def _cb(msg):
            received["msg"] = msg
            event.set()

        sub          = self.create_subscription(CameraInfo, topic, _cb, 1)
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
              #new_bboxes = detect_rectangles_yolo(self.model,cv_img)
              new_bboxes = detect_rectangles_hsv(cv_img, min_area=self.min_bbox_area_)
        except Exception as e:
            self.get_logger().error(f"detect_rectangles failed: {e}")
            return
        for xmin, xmax, ymin, ymax in new_bboxes:
            cv2.rectangle(
                cv_img,
                (xmin, ymin),
                (xmax, ymax),
                (0,255,0),
                2
            )
        
        try:
            annotated_msg = self._bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
            annotated_msg.header = msg.header
            self._debug_img_pub.publish(annotated_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Publish debug image failed: {e}")
           
        with self._mtx:
            if len(new_bboxes) > 0:
                self._bboxes = new_bboxes
            else:
                self._bboxes = []

    def _depth_img_callback(self, msg: Image):
            self.get_logger().info("depth callback is received")
            try:
                cv_depth = self._bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="passthrough"
                )

            except CvBridgeError as e:
                self.get_logger().error(f"Depth conversion failed: {e}")
                return

           
            with self._mtx:
                if not self._bboxes:
                    return

                bboxes_snapshot = list(self._bboxes)

            scaled_bboxes = bboxes_snapshot
            if not hasattr(self, "rgb_w"):
                return

            depth_h, depth_w = cv_depth.shape[:2]

            
            

           

            pose_array = PoseArray()
            pose_array.header.stamp = msg.header.stamp
            pose_array.header.frame_id = self.global_frame_

            for idx, (xmin, xmax, ymin, ymax) in enumerate(scaled_bboxes):

                
                cx_pix = int((xmin + xmax) / 2)
                cy_pix = int((ymin + ymax) / 2)
            

                    
                self.get_logger().info(f"cx_pix{cx_pix}::cypix::{cy_pix}")
                window = 5

                depths = []

                for i in range(cx_pix - window, cx_pix + window):

                    for j in range(cy_pix - window, cy_pix + window):

                        
                        if i < 0 or j < 0 or i >= depth_w or j >= depth_h:
                            continue

                        depth = cv_depth[j, i]

                       
                        if np.isnan(depth) or np.isinf(depth) or depth <= 0.0:
                            continue

                        depths.append(depth)

               
                if not depths:
                    self.get_logger().warn(
                        f"BBox[{idx}] no valid depth — skipping"
                    )
                    continue

               
                # depth = float(np.median(depths)) / 1000.0
                # self.get_logger().info(f'depth is received {depth}')
               
                # ray = self._cam_model.projectPixelTo3dRay(
                #     (float(cx_pix), float(cy_pix))
                # )

                # cx = ray[0] * depth
                # cy = ray[1] * depth
                # cz = depth
                
                roi = cv_depth[ymin:ymax, xmin:xmax]
                valid_depths = roi[np.isfinite(roi) & (roi > 0)]
                depth = np.median(valid_depths) / 1000.0

                fx = self._cam_model.fx()
                fy = self._cam_model.fy()
                ppx = self._cam_model.cx()
                ppy = self._cam_model.cy()

                cx = (cx_pix - ppx) * depth / fx
                cy = (cy_pix - ppy) * depth / fy
                cz = depth
                self.get_logger().info(
                    f"BBox[{idx}] centroid optical frame: "
                    f"x={cx:.3f} "
                    f"y={cy:.3f} "
                    f"z={cz:.3f}"
                )

                pt = PointStamped()

                
                pt.header.stamp = msg.header.stamp

                pt.header.frame_id = self.optical_frame_

                pt.point.x = cx
                pt.point.y = cy
                pt.point.z = cz
                
                try:

                
                    if not self._tf_buffer.can_transform(
                            self.global_frame_,
                            self.optical_frame_,
                            msg.header.stamp,
                            timeout=Duration(seconds=0.5)
                            ):

                        self.get_logger().warn("TF unavailable")
                        continue

                    transform = self._tf_buffer.lookup_transform(
                        self.global_frame_,
                        self.optical_frame_,
                        msg.header.stamp,
                        timeout=Duration(seconds=0.5)
                    )

                    pt_out = tf2_geometry_msgs.do_transform_point(pt, transform)
                    current = np.array([
                        pt_out.point.x,
                        pt_out.point.y
                    ])
                    matched_id = None

                    for obj_id, history in self.track_history.items():

                        last = history[-1]

                        dist = np.linalg.norm(current - last)

                        if dist < self.MAX_POSITION_JUMP:

                            matched_id = obj_id
                            break
                    if matched_id is None:

                        matched_id = len(self.track_history)
                        self.track_history[matched_id] = []
                    self.track_history[matched_id].append(current)

                    if len(self.track_history[matched_id]) > self.MAX_HISTORY:

                        self.track_history[matched_id].pop(0)

                       

                except Exception as e:

                    self.get_logger().warn(
                        f"TF transform failed: {e}"
                    )

                    continue
                
                self.get_logger().info(
                    f"BBox[{idx}] in map: "
                    f"x={pt_out.point.x:.3f} "
                    f"y={pt_out.point.y:.3f} "
                    f"z={pt_out.point.z:.3f}"
                )
                
                pose = Pose()

                pose.position.x = pt_out.point.x
                pose.position.y = pt_out.point.y
                pose.position.z = pt_out.point.z

                pose.orientation.w = 1.0
                duplicate = False

                for existing in pose_array.poses:

                    dist = math.sqrt(
                        (existing.position.x - pose.position.x) ** 2 +
                        (existing.position.y - pose.position.y) ** 2
                    )

                    if dist < 1.0:
                        duplicate = True
                        break

                if duplicate:
                    continue
                if len(self.track_history[matched_id]) >= self.STABLE_FRAMES:

                    pose_array.poses.append(pose)
           
            if pose_array.poses:

                self.get_logger().info(
                    f"Publishing {len(pose_array.poses)} pose(s)"
                )

                self._pose_pub.publish(pose_array)
            if self.latest_rgb is not None:
                debug_msg = self._bridge.cv2_to_imgmsg(
                    self.latest_rgb,
                    encoding="bgr8"
                )

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