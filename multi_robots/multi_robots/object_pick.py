#!/usr/bin/env python3


import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, PoseStamped
from image_geometry import PinholeCameraModel
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
#from std_srvs.srv import Trigger

import tf2_geometry_msgs  
import tf2_ros

from pymoveit2 import MoveIt2
from pymoveit2.gripper_command import GripperCommand


RGB_TOPIC = '/manipulator/camera/image_raw'
DEPTH_TOPIC = '/manipulator/camera/depth/image_raw'
CAMERA_INFO_TOPIC = '/manipulator/camera/camera_info'

CAMERA_OPTICAL_FRAME = 'manipulator/camera_depth_optical_frame'
ARM_BASE_FRAME = 'manipulator/base_link'
PLANNING_GROUP = 'arm'
END_EFFECTOR_LINK = 'manipulator/end_effector_link'
GRIPPER_GROUP = 'gripper'

GRASP_Z_OFFSET_PREGRASP = 0.12
GRASP_Z_OFFSET_LIFT = 0.10
DEPTH_ROI_MIN_VALID_PIXELS = 5


TOPDOWN_QUAT = (1.0, 0.0, 0.0, 0.0)

MIN_BBOX_AREA = 5000


def detect_rectangles_hsv(image: np.ndarray,
                      min_area: int = 5000) -> list[list[int]]:

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 100, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 | mask2

    
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    
    if len(contours) == 0:
        return []

    
    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)

    if area < min_area:
        return []

    x, y, w, h = cv2.boundingRect(cnt)

    return [[x, x + w, y, y + h]]


class PickFromDetection(Node):

    def __init__(self):
        super().__init__('pick_from_detection')

        self.bridge = CvBridge()
        self.cam_model = PinholeCameraModel()
        self.camera_info_received = False
        self.lock = threading.Lock()
        self.pick_lock = threading.Lock()
        self.pick_in_progress = False
        self.object_detected = False

        
        cb_group = ReentrantCallbackGroup()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_cb, 10,
            callback_group=cb_group,
        )
        self.create_subscription(
            Image, RGB_TOPIC, self.rgb_cb, 10, callback_group=cb_group,
        )
        self.create_subscription(
            Image, DEPTH_TOPIC, self.depth_cb, 10, callback_group=cb_group,
        )
        self.pick_lock = threading.Lock()
        self.debug_img_pub = self.create_publisher(
            Image, '/manipulator/detection_debug_image', 10
        )

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=[
                'manipulator/joint1', 'manipulator/joint2',
                'manipulator/joint3', 'manipulator/joint4',
            ],
            base_link_name=ARM_BASE_FRAME,
            end_effector_name=END_EFFECTOR_LINK,
            group_name=PLANNING_GROUP,
            callback_group=cb_group,
        )
        self.gripper = GripperCommand(
            node=self,
            gripper_joint_names=['manipulator/gripper_left_joint'],
            open_gripper_joint_positions=[0.019],   
            closed_gripper_joint_positions=[-0.010],  
            max_effort=0.0,
            gripper_command_action_name='/manipulator/gripper_controller/gripper_cmd',
            callback_group=cb_group,
        )

        # self.create_service(
        #     Trigger, 'trigger_pick', self.trigger_pick_cb, callback_group=cb_group
        # )

        self.latest_rgb_msg = None
        self.latest_depth_msg = None
        self.get_logger().info('pick_from_detection ready, waiting for camera_info + images')
    
    def rgb_cb(self, msg: Image):
            with self.lock:
                self.latest_rgb_msg = msg
            if self.pick_in_progress:
                return

            
            

            try:
                cv_rgb = self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding='bgr8'
                )
                self.get_logger().info(f'cv_rgb shape: {cv_rgb.shape}')
                bboxes = detect_rectangles_hsv(cv_rgb)

                if not bboxes:
                    self.object_detected = False
                    return
                self.get_logger().info(f'bboxes found: {bboxes}') 
                if self.object_detected:
                    return
                self.object_detected = True
            
               

                self.get_logger().info(
                    'Object detected automatically - starting pick'
                )

                threading.Thread(
                    target=self.auto_pick,
                    daemon=True
                ).start()

            except Exception as e:
                self.get_logger().error(
                    f'Automatic detection failed: {e}'
                )
    def auto_pick(self):
        with self.pick_lock:

            if self.pick_in_progress:
                return

            self.pick_in_progress = True
            try:
                with self.lock:
                    rgb_msg = self.latest_rgb_msg
                    depth_msg = self.latest_depth_msg

                if rgb_msg is None or depth_msg is None:
                    self.get_logger().warn(
                        'Waiting for RGB + depth images'
                    )
                    return

                if not self.camera_info_received:
                    self.get_logger().warn(
                        'Waiting for camera info'
                    )
                    return

                cv_rgb = self.bridge.imgmsg_to_cv2(
                    rgb_msg,
                    desired_encoding='bgr8'
                )

                cv_depth = self.bridge.imgmsg_to_cv2(
                    depth_msg,
                    desired_encoding='passthrough'
                )

                bboxes = detect_rectangles_hsv(cv_rgb)

                if not bboxes:
                    self.get_logger().info(
                        'Object is not found'
                    )
                    return

                x1, x2, y1, y2 = bboxes[0]

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                self.get_logger().info(
                    f'Detected object, pixel center ({cx}, {cy})'
                )

                self.get_logger().info(
                    'Calculating 3D position'
                )

                pt_cam = self.deproject_roi(
                    cv_depth,
                    x1,
                    x2,
                    y1,
                    y2,
                    rgb_msg.header.stamp
                )

                self.get_logger().info(
                    f'Object in camera frame: '
                    f'x={pt_cam.point.x:.3f}, '
                    f'y={pt_cam.point.y:.3f}, '
                    f'z={pt_cam.point.z:.3f}'
                )

                pt_base = self.transform_to_base(pt_cam)

                self.get_logger().info(
                    f'Object in arm base frame: '
                    f'x={pt_base.point.x:.3f}, '
                    f'y={pt_base.point.y:.3f}, '
                    f'z={pt_base.point.z:.3f}'
                )

                self.run_pick_sequence(pt_base)

            except Exception as e:
                self.get_logger().error(
                    f'Automatic pick failed: {e}'
                )

            finally:
            
                import time
                time.sleep(2.0)

            

                self.get_logger().info(
                    'Ready for next object'
                )
    def camera_info_cb(self, msg: CameraInfo):
        if not self.camera_info_received:
            self.cam_model.fromCameraInfo(msg)
            self.camera_info_received = True
            self.get_logger().info('Camera intrinsics loaded')

  

    def depth_cb(self, msg: Image):
        with self.lock:
            self.latest_depth_msg = msg

    # def trigger_pick_cb(self, request, response):
    #     """ros2 service call /trigger_pick std_srvs/srv/Trigger {}"""
    #     with self.lock:
    #         rgb_msg = self.latest_rgb_msg
    #         depth_msg = self.latest_depth_msg

    #     if rgb_msg is None:
    #         response.success = False
    #         response.message = 'No RGB frame received yet'
    #         return response
    #     if depth_msg is None:
    #         response.success = False
    #         response.message = 'No depth frame received yet'
    #         return response
    #     if not self.camera_info_received:
    #         response.success = False
    #         response.message = 'Camera intrinsics not received yet'
    #         return response

    #     try:
    #         cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
    #     except CvBridgeError as e:
    #         response.success = False
    #         response.message = f'RGB conversion failed: {e}'
    #         return response

    #     bboxes = detect_rectangles_hsv(cv_rgb)

        
    #     debug_img = cv_rgb.copy()
    #     for x1, x2, y1, y2 in bboxes:
    #         cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     try:
    #         debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
    #         debug_msg.header = rgb_msg.header
    #         self.debug_img_pub.publish(debug_msg)
    #     except CvBridgeError as e:
    #         self.get_logger().warn(f'Could not publish debug image: {e}')

    #     if not bboxes:
    #         response.success = False
    #         response.message = 'No object detected in latest frame'
    #         return response

    #     x1, x2, y1, y2 = bboxes[0]
    #     cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    #     self.get_logger().info(f'Detected object, pixel center ({cx},{cy})')

    #     try:
    #         cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
    #     except CvBridgeError as e:
    #         response.success = False
    #         response.message = f'Depth conversion failed: {e}'
    #         return response

    #     try:
    #         pt_cam = self.deproject_roi(cv_depth, x1, x2, y1, y2, rgb_msg.header.stamp)
    #         pt_base = self.transform_to_base(pt_cam)
    #         ok = self.run_pick_sequence(pt_base)
    #     except Exception as e:
    #         self.get_logger().error(f'Pick failed: {e}')
    #         response.success = False
    #         response.message = f'Pick failed: {e}'
    #         return response

    #     response.success = bool(ok)
    #     response.message = 'Pick sequence complete' if ok else 'Pick sequence failed'
    #     return response

    def deproject_roi(self, depth_image: np.ndarray, x1, x2, y1, y2, stamp) -> PointStamped:
        """Median depth over the bbox ROI (more robust than a single pixel),
        deprojected to a 3D point in the camera optical frame."""
        roi = depth_image[y1:y2, x1:x2].astype(np.float32)

        if depth_image.dtype == np.uint16:
            roi = roi / 1000.0  

        valid = roi[np.isfinite(roi) & (roi > 0.0)]
        if valid.size < DEPTH_ROI_MIN_VALID_PIXELS:
            raise ValueError(
                f'Not enough valid depth pixels in ROI ({valid.size} found)'
            )

        z = float(np.median(valid))
        cx_pix, cy_pix = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        ray = self.cam_model.projectPixelTo3dRay((cx_pix, cy_pix))
        scale = z / ray[2]
        x, y = ray[0] * scale, ray[1] * scale

        pt = PointStamped()
        pt.header.frame_id = CAMERA_OPTICAL_FRAME
        pt.header.stamp = stamp
        pt.point.x, pt.point.y, pt.point.z = x, y, z
        return pt

    def transform_to_base(self, pt_cam: PointStamped) -> PointStamped:
        try:
            transform = self.tf_buffer.lookup_transform(
                ARM_BASE_FRAME,
                pt_cam.header.frame_id,
                rclpy.time.Time(), 
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(
                f'TF lookup {pt_cam.header.frame_id} -> {ARM_BASE_FRAME} failed: {e}'
            )
            raise

        return tf2_geometry_msgs.do_transform_point(pt_cam, transform)

    def build_grasp_pose(self, pt_base: PointStamped) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = ARM_BASE_FRAME
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = pt_base.point.x
        pose.pose.position.y = pt_base.point.y
        pose.pose.position.z = pt_base.point.z
        (pose.pose.orientation.x, pose.pose.orientation.y,
         pose.pose.orientation.z, pose.pose.orientation.w) = TOPDOWN_QUAT
        return pose

    
    def run_pick_sequence(self, pt_base: PointStamped) -> bool:
        grasp_pose = self.build_grasp_pose(pt_base)

        pre_grasp = PoseStamped()
        pre_grasp.header = grasp_pose.header
        pre_grasp.pose.position.x = grasp_pose.pose.position.x
        pre_grasp.pose.position.y = grasp_pose.pose.position.y
        pre_grasp.pose.position.z = grasp_pose.pose.position.z + GRASP_Z_OFFSET_PREGRASP
        pre_grasp.pose.orientation = grasp_pose.pose.orientation

        self.get_logger().info('Opening gripper')
        self.gripper.open()
        self.gripper.wait_until_executed()

        self.get_logger().info('Moving to pre-grasp')
        self.moveit2.move_to_pose(pose=pre_grasp, cartesian=False)
        self.moveit2.wait_until_executed()

        self.get_logger().info('Cartesian descent to grasp pose')
        self.moveit2.move_to_pose(pose=grasp_pose, cartesian=True)
        self.moveit2.wait_until_executed()

        self.get_logger().info('Closing gripper')
        self.gripper.close()
        self.gripper.wait_until_executed()

        retreat = PoseStamped()
        retreat.header = grasp_pose.header
        retreat.pose = pre_grasp.pose
        retreat.pose.position.z = grasp_pose.pose.position.z + GRASP_Z_OFFSET_LIFT

        self.get_logger().info('Cartesian retreat')
        self.moveit2.move_to_pose(pose=retreat, cartesian=True)
        self.moveit2.wait_until_executed()

        self.get_logger().info('Pick sequence complete')
        return True


def main():
    rclpy.init()
    node = PickFromDetection()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()