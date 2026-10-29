#!/usr/bin/env python3
"""Pick-and-place driven by your YOLO detection node.

Input : DetectedObjectArray on `detected_objects` (map-frame poses from pose_estimation_3d;
        ball pose = sphere centre). No camera code lives in this node.

Ball cycle (ball is placed in front of the robot, no searching):
    wait for detection
      -> ball > BALL_NAV_THRESHOLD (50 cm) away : Nav2 to a standoff
         ball <= 50 cm                          : skip Nav2
      -> cmd_vel fine align to ARM_STANDOFF_DIST (30 cm)
      -> PICK

Box cycle (only while a ball is held):
    search box -> Nav2 to standoff -> PLACE

Arm motion uses MoveIt only (no own IK / own planning code):
    move_to()       -> /move_action (MoveGroup) with a POSITION-ONLY goal.
                       MoveIt does IK + OMPL planning + execution.
    move_straight() -> /compute_cartesian_path + /execute_trajectory
                       (straight-line approach / lift / retreat).

IMPORTANT: the 4-joint arm cannot reach a full 6D pose, so MoveIt's IK solver
must be position-only. In moveit_config/config/kinematics.yaml either use
    arm:
      kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
      position_only_ik: true
or pick_ik with `rotation_scale: 0.0`. Restart move_group afterwards.
"""

import math
import threading
import time
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
import numpy as np
import rclpy
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transforms)
from control_msgs.action import GripperCommand
from geometry_msgs.msg import Pose, PointStamped, PoseStamped, TwistStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (Constraints, MotionPlanRequest, MoveItErrorCodes,
                             PositionConstraint, RobotState)
from moveit_msgs.srv import GetCartesianPath
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from object_interfaces.msg import DetectedObjectArray
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros import TransformException


DETECTION_MAX_AGE_SEC = 3.0    
LOOK_WAIT_SEC = 3.0             
LATEST_MAX_AGE_SEC = 10.0       
WAIT_FOR_BALL_SEC = 15.0        


BALL_NAV_THRESHOLD = 0.50    
NAV_STANDOFF_BALL=0.25   
NAV_STANDOFF_BOX = 0.50
NAV_MIN_MOVE = 0.25           

SEARCH_SPIN_SPEED = 0.3         
SEARCH_TIMEOUT = 25.0

ARM_STANDOFF_DIST = 0.30       
ARM_STANDOFF_TOL = 0.02
FINE_MAX_STEPS = 25
FINE_MAX_MISSES = 3
FINE_PULSE_SEC = 0.25
FINE_SETTLE_SEC = 0.5
FINE_YAW_TOL = 0.04
FINE_ANGULAR_GAIN = 1.0
FINE_LINEAR_GAIN = 0.5
FINE_ANGULAR_LIMITS = (0.06, 0.30)
FINE_LINEAR_LIMITS = (0.02, 0.06)

PLANNING_GROUP = "arm"
EE_LINK = "end_effector_link"
ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4")
PLAN_TIME_SEC = 8.0
EXECUTE_TIMEOUT_SEC = 30.0
GOAL_TOLERANCES = (0.01, 0.02, 0.03)   
CARTESIAN_STEP = 0.005                 
CARTESIAN_MIN_FRACTION = 0.95
GRASP_CONFIRM_TOL = 0.02
GRASP_PRE_HEIGHT = -0.07
GRASP_LIFT_HEIGHT = 0.05
PLACE_DROP_HEIGHT = 0.08
PLACE_PRE_HEIGHT = 0.15
GRASP_Z_OFFSET = 0.0  
GRIPPER_OPEN = 0.019
GRIPPER_CLOSE = 0.00
GRIPPER_MAX_EFFORT = 5.0


HOME_JOINTS_DEG = [0.0, -57.0, 40.0, 17.0]

def limit(value, lo, hi):
    """Clamp |value| into [lo, hi], keeping the sign."""
    return math.copysign(min(max(abs(value), lo), hi), value)


def err_name(code: int) -> str:
    """MoveItErrorCodes value -> readable name, e.g. 'NO_IK_SOLUTION(-31)'."""
    for k, v in vars(MoveItErrorCodes).items():
        if k.isupper() and isinstance(v, int) and not isinstance(v, bool) and v == code:
            return f"{k}({code})"
    return str(code)


class PickAndPlaceNode(Node):

    def __init__(self):
        super().__init__("pick_and_place_node")

        self.declare_parameter("global_frame", "map")           # must match the detection node
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("arm_base_frame", "base_link")   # frame of the arm's first joint
        self.declare_parameter("ball_classes", ["ball"])
        self.declare_parameter("box_classes", ["box"])
        self.global_frame = self.get_parameter("global_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.arm_base_frame = self.get_parameter("arm_base_frame").value

        self.last_move_target = None
        self.class_kinds = {}
        for kind in ("ball", "box"):
            for name in self.get_parameter(f"{kind}_classes").value:
                self.class_kinds[name.lower()] = kind

        self._lock = threading.Lock()
        self._seen = {}                       
        self._seq = {"ball": 0, "box": 0}     
        self.joint_state = None
        self.holding_ball = False

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        cbg = ReentrantCallbackGroup()
        self.create_subscription(
            DetectedObjectArray, "detected_objects",
            self._objects_cb, 10, callback_group=cbg)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10,
                                 callback_group=cbg)
        self.cmd_vel_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)

        
        self.move_client = ActionClient(self, MoveGroup, "/move_action",
                                        callback_group=cbg)
        self.cartesian_client = self.create_client(
            GetCartesianPath, "/compute_cartesian_path", callback_group=cbg)
        self.execute_client = ActionClient(self, ExecuteTrajectory, "/execute_trajectory",
                                           callback_group=cbg)
        self.gripper = ActionClient(self, GripperCommand, "/gripper_controller/gripper_cmd",
                                    callback_group=cbg)
        self.navigator = BasicNavigator()
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik", callback_group=cbg)
        self.cycles = {
            "ball": [
                
                ("Wait for ball", lambda:self.wait_for_ball),
                #("Find ball", lambda: self.search_for("ball")),
                ("Nav2 if ball > 50 cm", self.approach_ball),
                ("Fine align (cmd_vel)", self.fine_align),
                ("Pick", self.pick),
            ],
            "box": [
                ("Find box", lambda: self.search_for("box")),
                ("Nav2 to box standoff", lambda: self.approach("box", NAV_STANDOFF_BOX)),
                ("Place", self.place),
            ],
        }
        self.get_logger().info(
            f"Pick-and-place node ready. Class map: {self.class_kinds}")
    
    def settle_and_log(self, target) -> bool:
        time.sleep(1.0)                      # let arm sag/settle
        real = self.ee_pos()
        if real is not None:
            err = float(np.linalg.norm(np.array(target) - real))
            self.get_logger().info(
                f"REACH real={np.round(real, 3)} err={err * 100:.1f} cm (info only)")
        return True                         
    def move_home(self) -> bool:
        """Move arm to the specified home joint configuration."""
        if self.joint_state is None:
            self.get_logger().error("No joint states available")
            return False

        req = MotionPlanRequest()
        req.group_name = PLANNING_GROUP
        req.pipeline_id = "ompl"
        req.start_state = self.current_state()
        req.allowed_planning_time = PLAN_TIME_SEC
        req.num_planning_attempts = 10
        req.max_velocity_scaling_factor = 0.2
        req.max_acceleration_scaling_factor = 0.2

        from moveit_msgs.msg import JointConstraint

        constraints = Constraints()
        for name, deg in zip(ARM_JOINTS, HOME_JOINTS_DEG):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = math.radians(deg)
            jc.tolerance_above = math.radians(2.0)
            jc.tolerance_below = math.radians(2.0)
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        req.goal_constraints.append(constraints)

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        send = self.move_client.send_goal_async(goal)
        if not self.wait_for(send, 5.0):
            return False

        handle = send.result()
        if handle is None or not handle.accepted:
            self.get_logger().error("Home goal rejected")
            return False

        result_future = handle.get_result_async()
        if not self.wait_for(result_future, PLAN_TIME_SEC + EXECUTE_TIMEOUT_SEC):
            self.get_logger().error("Home movement timed out")
            return False

        result = result_future.result().result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"Home movement failed: {err_name(result.error_code.val)}"
            )
            return False

        self.log_joints("HOME reached")
        return True
    def ee_pos(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.arm_base_frame, EE_LINK, rclpy.time.Time())
        except TransformException as e:
            self.get_logger().warn(f"EE pose unavailable: {e}")
            return None
        p = t.transform.translation
        return np.array([p.x, p.y, p.z])

   
    def probe_reach(self, bx, by, bz):
        """Diagnostic only: does MoveIt's IK ignore orientation?"""
        points = []
        try:
            t = self.tf_buffer.lookup_transform(
                self.arm_base_frame, EE_LINK, rclpy.time.Time())
            c, q = t.transform.translation, t.transform.rotation
            js = dict(zip(self.joint_state.name, self.joint_state.position))
            pitch_joints = sum(js.get(j, 0.0) for j in ("joint2", "joint3", "joint4"))
            self.get_logger().info(
                f"PROBE EE quat=({q.x:.3f},{q.y:.3f},{q.z:.3f},{q.w:.3f}) "
                f"pitch_from_quat={2.0 * math.atan2(q.y, q.w):.3f} "
                f"sum(j2..j4)={pitch_joints:.3f}")
            points.append(("EE pos + ACTUAL orientation", c.x, c.y, c.z,
                        (q.x, q.y, q.z, q.w)))
            points.append(("EE pos + identity orientation", c.x, c.y, c.z,
                        (0.0, 0.0, 0.0, 1.0)))
            points.append(("EE pos + roll-90deg orientation", c.x, c.y, c.z,
                        (0.7071, 0.0, 0.0, 0.7071)))
        except TransformException as e:
            self.get_logger().warn(f"PROBE: EE pose unavailable: {e}")
        for dz in (0.12, 0.0):
            points.append((f"ball z{dz:+.2f}", bx, by, bz + dz, (0.0, 0.0, 0.0, 1.0)))

        for label, x, y, z, quat in points:
            req = GetPositionIK.Request()
            req.ik_request.group_name = PLANNING_GROUP
            req.ik_request.robot_state = self.current_state()
            req.ik_request.ik_link_name = EE_LINK
            req.ik_request.avoid_collisions = False
            req.ik_request.timeout = Duration(seconds=1.0).to_msg()
            ps = PoseStamped()
            ps.header.frame_id = self.arm_base_frame
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = x, y, z
            (ps.pose.orientation.x, ps.pose.orientation.y,
            ps.pose.orientation.z, ps.pose.orientation.w) = quat
            req.ik_request.pose_stamped = ps
            fut = self.ik_client.call_async(req)
            if not self.wait_for(fut, 3.0):
                self.get_logger().warn(f"PROBE {label}: service timed out")
                continue
            code = fut.result().error_code.val
            self.get_logger().info(
                f"PROBE {label}: ({x:.3f}, {y:.3f}, {z:.3f}) -> {err_name(code)}")
   
    def _objects_cb(self, msg: DetectedObjectArray):
        now = time.monotonic()
        by_kind = {}
        for obj in msg.objects:
            kind = self.class_kinds.get(obj.class_name.lower())
            if kind is None:
                continue
            p = obj.pose.position
            by_kind.setdefault(kind, []).append(np.array([p.x, p.y, p.z]))
        with self._lock:
            for kind, poses in by_kind.items():
                self._seen[kind] = (now, poses)
                self._seq[kind] += 1

    def _joint_cb(self, msg: JointState):
        self.joint_state = msg

    def visible(self, kind) -> bool:
        with self._lock:
            entry = self._seen.get(kind)
        return entry is not None and time.monotonic() - entry[0] < DETECTION_MAX_AGE_SEC

    def robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.global_frame, self.base_frame, rclpy.time.Time())
        except TransformException as e:
            self.get_logger().warn(f"Robot pose unavailable: {e}")
            return None
        return t.transform.translation.x, t.transform.translation.y

    def nearest(self, poses):
        robot = self.robot_xy()
        if robot is None:
            return poses[0]
        return min(poses, key=lambda p: math.hypot(p[0] - robot[0], p[1] - robot[1]))

    def to_frame(self, p, frame):
        """Map-frame xyz -> `frame` (latest TF)."""
        if frame != self.global_frame:
            pt = PointStamped()
            pt.header.frame_id = self.global_frame     
            pt.point.x, pt.point.y, pt.point.z = map(float, p)
            try:
                out = self.tf_buffer.transform(pt, frame, timeout=Duration(seconds=0.5))
            except TransformException as e:
                self.get_logger().warn(f"TF {self.global_frame}->{frame} failed: {e}")
                return None
            p = np.array([out.point.x, out.point.y, out.point.z])
        return np.array(p, dtype=float)

    def latest(self, kind, frame):
        """Most recent pose of `kind` (PointStamped in `frame`) WITHOUT waiting.
        Good for far targets / Nav2 goals: the object is static in the map."""
        with self._lock:
            entry = self._seen.get(kind)
        if entry is None or time.monotonic() - entry[0] > LATEST_MAX_AGE_SEC:
            return None
        p = self.to_frame(self.nearest(entry[1]), frame)
        if p is None:
            return None
        out = PointStamped()
        out.header.frame_id = frame
        out.point.x, out.point.y, out.point.z = map(float, p)
        return out

    def locate(self, kind, frame, samples=1, min_valid=1, timeout=10.0):
        """Median position (PointStamped, in `frame`) of the nearest `kind` object,
        using only detection messages that arrive AFTER this call. Call it while
        the robot is stationary."""
        with self._lock:
            last_seq = self._seq[kind]
        pts = []
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline and len(pts) < samples:
            with self._lock:
                seq, entry = self._seq[kind], self._seen.get(kind)
            if seq != last_seq and entry is not None:
                last_seq = seq
                p = self.to_frame(self.nearest(entry[1]), frame)
                if p is not None:
                    pts.append(p)
                    time.sleep(0.1)
                    continue
            time.sleep(0.02)

        if len(pts) < min_valid:
            return None
        med = np.median(np.array(pts), axis=0)
        out = PointStamped()
        out.header.frame_id = frame
        out.point.x, out.point.y, out.point.z = map(float, med)
        return out

    def stop_base(self):
        self.cmd_vel_pub.publish(TwistStamped())

    def pulse(self, linear, angular, duration):
        twist = TwistStamped()
        twist.twist.linear.x = float(linear)
        twist.twist.angular.z = float(angular)
        end = time.monotonic() + duration
        while time.monotonic() < end and rclpy.ok():
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.05)
        self.stop_base()

    def wait_for_ball(self, timeout=WAIT_FOR_BALL_SEC) -> bool:
        """Robot stays still and just waits for a ball detection (no spinning)."""
        self.stop_base()
        end = time.monotonic() + timeout
        while time.monotonic() < end and rclpy.ok():
            if self.visible("ball"):
                self.get_logger().info("Ball detected, continuing")
                return True
            time.sleep(0.1)
        self.get_logger().warn("No ball detected (is /detected_objects publishing?)")
        return False

    def search_for(self, kind, timeout=SEARCH_TIMEOUT) -> bool:
        """Stand still briefly (detections are slow), then spin until `kind` is detected.
        Used for the box only."""
        self.stop_base()
        look_until = time.monotonic() + LOOK_WAIT_SEC
        while time.monotonic() < look_until and rclpy.ok():
            if self.visible(kind):
                return True
            time.sleep(0.05)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and rclpy.ok():
            if self.visible(kind):
                self.stop_base()
                return True
            twist = TwistStamped()
            twist.twist.angular.z = SEARCH_SPIN_SPEED
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)
        self.stop_base()
        self.get_logger().warn(f"No '{kind}' detected while searching")
        return False

    def navigate_to(self, x, y, standoff, min_move=NAV_MIN_MOVE) -> bool:
        """Nav2 goal `standoff` metres short of map point (x, y), facing it."""
        robot = self.robot_xy()
        if robot is None:
            return False
        dx, dy = x - robot[0], y - robot[1]
        dist = math.hypot(dx, dy)
        self.get_logger().info(f"Target is {dist:.2f} m away")
        if dist - standoff < min_move:
            return True                      

        theta = math.atan2(dy, dx)
        goal = PoseStamped()
        goal.header.frame_id = self.global_frame
        goal.header.stamp = self.navigator.get_clock().now().to_msg()
        goal.pose.position.x = x - standoff * math.cos(theta)
        goal.pose.position.y = y - standoff * math.sin(theta)
        goal.pose.orientation.z = math.sin(theta / 2.0)
        goal.pose.orientation.w = math.cos(theta / 2.0)

        self.get_logger().info(
            f"Nav goal: x={goal.pose.position.x:.2f} y={goal.pose.position.y:.2f}")
        self.stop_base()
        self.navigator.goToPose(goal)
        while not self.navigator.isTaskComplete():
            time.sleep(0.1)
        result = self.navigator.getResult()
        self.get_logger().info(f"Nav result: {result}")
        return result == TaskResult.SUCCEEDED

    def approach(self, kind, standoff) -> bool:
        """Nav2 to a standoff from the detected object, then re-acquire it (box)."""
        target = self.latest(kind, self.global_frame)
        if target is None:
            self.get_logger().warn(f"Could not get a map pose for '{kind}'")
            return False
        if not self.navigate_to(target.point.x, target.point.y, standoff):
            return False
        return self.search_for(kind)         

    def approach_ball(self) -> bool:
        """Ball > BALL_NAV_THRESHOLD from the arm base -> Nav2 goal.
        Otherwise skip Nav2 and let fine_align bring it to ARM_STANDOFF_DIST."""
        ball = self.latest("ball", self.arm_base_frame)
        if ball is None:
            self.get_logger().warn("approach_ball: no ball pose available")
            return False

        dist = math.hypot(ball.point.x, ball.point.y)
        self.get_logger().info(f"Ball is {dist:.2f} m from arm base")

        if dist <= BALL_NAV_THRESHOLD:
            self.get_logger().info(
                f"Ball within {BALL_NAV_THRESHOLD:.2f} m -> skip Nav2, fine align only")
            return True

        target = self.latest("ball", self.global_frame)
        if target is None:
            return False
        self.get_logger().info(
            f"Ball farther than {BALL_NAV_THRESHOLD:.2f} m -> sending Nav2 goal")
        if not self.navigate_to(target.point.x, target.point.y, NAV_STANDOFF_BALL,
                                min_move=0.0):
            return False

        
        if not self.wait_for_ball(timeout=5.0):
            return self.search_for("ball")
        return True

    def fine_align(self) -> bool:
        """Place the base so the ball is straight ahead at ARM_STANDOFF_DIST from the
        arm base: rotate until the ball is centred, then drive the radial error."""
        misses = 0
        for step in range(FINE_MAX_STEPS):
            pt = self.locate("ball", self.arm_base_frame)
            if pt is None:
                misses += 1
                if misses > FINE_MAX_MISSES:
                    break
                continue
            misses = 0

            x, y = pt.point.x, pt.point.y
            yaw_err = math.atan2(y, x)
            dist_err = math.hypot(x, y) - ARM_STANDOFF_DIST
            self.get_logger().info(
                f"FineAlign[{step}] x={x:.3f} y={y:.3f} "
                f"yaw_err={yaw_err:.3f} dist_err={dist_err:.3f}")

            if abs(yaw_err) > FINE_YAW_TOL:
                self.pulse(0.0, limit(FINE_ANGULAR_GAIN * yaw_err, *FINE_ANGULAR_LIMITS),
                           FINE_PULSE_SEC)
            elif abs(dist_err) > ARM_STANDOFF_TOL:
                self.pulse(limit(FINE_LINEAR_GAIN * dist_err, *FINE_LINEAR_LIMITS), 0.0,
                           FINE_PULSE_SEC)
            else:
                self.stop_base()
                self.get_logger().info("Base placed at golden distance")
                return True
            time.sleep(FINE_SETTLE_SEC)

        self.stop_base()
        self.get_logger().warn("Fine alignment failed")
        return False

    @staticmethod
    def wait_for(future, timeout) -> bool:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done():
            if time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return future.done()

    def current_state(self) -> RobotState:
        state = RobotState()
        state.joint_state = self.joint_state
        return state

    def log_joints(self, tag):
        if self.joint_state is None:
            return
        cur = {n: round(p, 3) for n, p in zip(self.joint_state.name,
                                              self.joint_state.position)
               if n in ARM_JOINTS}
        self.get_logger().info(f"{tag} joints={cur}")

    def build_position_goal(self, x, y, z, tol) -> MoveGroup.Goal:
        """MoveGroup goal: end effector inside a sphere of radius `tol` around (x, y, z)
        in arm_base_frame. Position only (no orientation constraint), because the
        arm has 4 joints and cannot satisfy a full 6D pose."""
        pc = PositionConstraint()
        pc.header.frame_id = self.arm_base_frame
        pc.link_name = EE_LINK
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [float(tol)]                 # radius
        pc.constraint_region.primitives.append(sphere)
        centre = Pose()
        centre.position.x, centre.position.y, centre.position.z = float(x), float(y), float(z)
        centre.orientation.w = 1.0
        pc.constraint_region.primitive_poses.append(centre)
        pc.weight = 1.0

        goal_c = Constraints()
        goal_c.position_constraints.append(pc)

        req = MotionPlanRequest()
        req.group_name = PLANNING_GROUP
        req.pipeline_id = "ompl"
        req.start_state = self.current_state()
        req.goal_constraints.append(goal_c)
        req.num_planning_attempts = 10
        req.allowed_planning_time = PLAN_TIME_SEC
        req.max_velocity_scaling_factor = 0.3
        req.max_acceleration_scaling_factor = 0.3

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = False          
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        return goal

    def move_to(self, x, y, z) -> bool:
        """OMPL move of the end effector to (x, y, z) in arm_base_frame.
        MoveIt does IK + planning + execution."""
        if self.joint_state is None:
            self.get_logger().error("No /joint_states received")
            return False
        self.get_logger().info(
            f"MOVE_TO ({self.arm_base_frame}) x={x:.3f} y={y:.3f} z={z:.3f} "
            f"| radial={math.hypot(x, y):.3f} m")
        self.log_joints("  start")

        for tol in GOAL_TOLERANCES:
            self.get_logger().info(f"  MoveGroup goal, tolerance={tol * 100:.0f} cm")
            send = self.move_client.send_goal_async(self.build_position_goal(x, y, z, tol))
            if not self.wait_for(send, 5.0):
                self.get_logger().error("  MoveGroup: goal send timed out")
                return False
            handle = send.result()
            if handle is None or not handle.accepted:
                self.get_logger().error("  MoveGroup: goal rejected")
                return False
            result_future = handle.get_result_async()
            if not self.wait_for(result_future, PLAN_TIME_SEC + EXECUTE_TIMEOUT_SEC):
                self.get_logger().error("  MoveGroup: result timed out")
                return False
            code = result_future.result().result.error_code.val
            if code == MoveItErrorCodes.SUCCESS:
                self.log_joints("  MoveGroup SUCCESS, now")
                return True
            self.get_logger().warn(f"  MoveGroup failed: {err_name(code)}")

        self.get_logger().error(
            f"No plan found for ({x:.3f}, {y:.3f}, {z:.3f}) with any tolerance")
        return False

    def move_straight(self, x, y, z, avoid_collisions=True,
                      min_fraction=CARTESIAN_MIN_FRACTION) -> bool:
        """Straight-line move to (x, y, z), keeping the current gripper orientation."""
        if self.joint_state is None:
            self.get_logger().error("No /joint_states received")
            return False
        try:
            t = self.tf_buffer.lookup_transform(
                self.arm_base_frame, EE_LINK, rclpy.time.Time())
        except TransformException as e:
            self.get_logger().error(f"EE pose unavailable: {e}")
            return False

        self.get_logger().info(
            f"MOVE_STRAIGHT x={x:.3f} y={y:.3f} z={z:.3f} "
            f"| from z={t.transform.translation.z:.3f}")

        wp = Pose()
        wp.position.x, wp.position.y, wp.position.z = float(x), float(y), float(z)
        wp.orientation = t.transform.rotation

        req = GetCartesianPath.Request()
        req.header.frame_id = self.arm_base_frame
        req.start_state = self.current_state()
        req.group_name = PLANNING_GROUP
        req.link_name = EE_LINK
        req.waypoints = [wp]
        req.max_step = CARTESIAN_STEP
        req.jump_threshold = 0.0
        req.avoid_collisions = avoid_collisions

        future = self.cartesian_client.call_async(req)
        if not self.wait_for(future, 5.0):
            self.get_logger().error("  CARTESIAN: service call timed out")
            return False
        res = future.result()
        if res is None:
            self.get_logger().error("  CARTESIAN: no response")
            return False
        self.get_logger().info(
            f"  CARTESIAN: fraction={res.fraction:.2f} code={err_name(res.error_code.val)}")
        if res.fraction < min_fraction:
            self.get_logger().warn(
                f"  CARTESIAN: only {res.fraction:.0%} of the path is reachable")
            return False
        return self.execute(res.solution)

    def execute(self, trajectory) -> bool:
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        send = self.execute_client.send_goal_async(goal)
        if not self.wait_for(send, 5.0):
            self.get_logger().error("  EXECUTE: goal send timed out")
            return False
        handle = send.result()
        if handle is None or not handle.accepted:
            self.get_logger().error("  EXECUTE: trajectory goal rejected")
            return False
        result_future = handle.get_result_async()
        if not self.wait_for(result_future, EXECUTE_TIMEOUT_SEC):
            self.get_logger().error("  EXECUTE: timed out")
            return False
        result = result_future.result().result
        if result.error_code.val != result.error_code.SUCCESS:
            self.get_logger().error(
                f"  EXECUTE failed: {err_name(result.error_code.val)}")
            return False
        self.get_logger().info("  EXECUTE ok")
        return True

    def set_gripper(self, position, closing=False) -> bool:
        """Command the gripper. When closing on the ball a stall counts as success."""
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = GRIPPER_MAX_EFFORT

        send = self.gripper.send_goal_async(goal)
        if not self.wait_for(send, 5.0):
            return False
        handle = send.result()
        if handle is None or not handle.accepted:
            self.get_logger().error("Gripper goal rejected")
            return False
        result_future = handle.get_result_async()
        if not self.wait_for(result_future, 5.0):
            return False
        result = result_future.result().result
        self.last_gripper_pos = result.position  
        self.get_logger().info(
            f"Gripper: pos={result.position:.4f} stalled={result.stalled} "
            f"reached={result.reached_goal}")
        time.sleep(0.3)
        return result.reached_goal or (closing and result.stalled)

   
    def run_steps(self, steps) -> bool:
        for name, fn in steps:
            self.get_logger().info(f"--- {name}")
            if not fn():
                self.get_logger().error(f"--- FAILED: {name}")
                return False
        return True

    def pick(self) -> bool:
        ball = self.locate("ball", self.arm_base_frame, samples=3, min_valid=2)
        if ball is None:
            self.get_logger().error("Could not localise ball in base frame")
            return False
        x, y, z = ball.point.x, ball.point.y, ball.point.z
        
        self.get_logger().info(
            f"PICK ball in '{self.arm_base_frame}': x={x:.3f} y={y:.3f} z={z:.3f} "
            f"| yaw={math.degrees(math.atan2(y, x)):.1f} deg "
            f"| radial={math.hypot(x, y):.3f} m")
        self.probe_reach(x, y, z)   
        ok = self.run_steps([
            ("Open gripper", lambda: self.set_gripper(GRIPPER_OPEN)),
            ("Pre-grasp (OMPL)", lambda: self.move_to(x, y, z + GRASP_PRE_HEIGHT)),
           
            ("Settle + log", lambda: self.settle_and_log((x, y, z + GRASP_Z_OFFSET))),
            ("Close gripper", lambda: self.set_gripper(GRIPPER_CLOSE, closing=True)),
            
            # ("Lift (straight up)",
            #  lambda: self.move_straight(x, y, z + GRASP_LIFT_HEIGHT,
            #                             avoid_collisions=False)),
            ("Move to home pose",self.move_home),
        ])
        self.holding_ball = ok
        return ok

    def place(self) -> bool:
        box = self.locate("box", self.arm_base_frame, samples=3, min_valid=2)
        if box is None:
            self.get_logger().error("Could not localise box in base frame")
            return False
        x, y = box.point.x, box.point.y
        z_drop = box.point.z + PLACE_DROP_HEIGHT
        z_pre = z_drop + PLACE_PRE_HEIGHT
        self.get_logger().info(
            f"PLACE box in '{self.arm_base_frame}': x={x:.3f} y={y:.3f} "
            f"z_drop={z_drop:.3f} z_pre={z_pre:.3f} | radial={math.hypot(x, y):.3f} m")
        ok = self.run_steps([
            ("Pre-place (OMPL)", lambda: self.move_to(x, y, z_pre)),
            ("Drop pose (straight down)",
             lambda: self.move_straight(x, y, z_drop, avoid_collisions=False)),
            ("Release", lambda: self.set_gripper(GRIPPER_OPEN)),
            ("Retreat (straight up)",
             lambda: self.move_straight(x, y, z_pre, avoid_collisions=False)),
        ])
        if ok:
            self.holding_ball = False
        return ok

    def wait_for_interfaces(self):
        self.get_logger().info("Waiting for MoveIt, gripper and joint states...")
        while rclpy.ok() and not self.cartesian_client.wait_for_service(timeout_sec=1.0):
            pass
        for client in (self.move_client, self.execute_client, self.gripper):
            while rclpy.ok() and not client.wait_for_server(timeout_sec=1.0):
                pass
        while rclpy.ok() and self.joint_state is None:
            time.sleep(0.2)
        self.get_logger().info("All interfaces ready.")

    def run_mission(self):
        self.wait_for_interfaces()
        while rclpy.ok():
            
            kind = "box" if self.holding_ball else "ball"
            try:
                ok = self.run_steps(self.cycles[kind])
            except Exception as e:  
                self.get_logger().error(f"Mission step crashed: {e}")
                ok = False
            if not ok:
                self.stop_base()
                time.sleep(1.0)
            elif kind == "box":
                self.get_logger().info("Cycle complete, waiting for next ball.")


def main():
    rclpy.init()
    node = PickAndPlaceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=node.run_mission, daemon=True).start()
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_base()
        node.navigator.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()