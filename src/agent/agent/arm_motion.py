"""Low-level arm + gripper primitives (MoveIt only), cancel-aware.
Shared by pick.py and place.py. No perception, no navigation here."""

import math
import threading
import time

import numpy as np
import rclpy
from control_msgs.action import GripperCommand
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (Constraints, JointConstraint, MotionPlanRequest,
                             MoveItErrorCodes, PositionConstraint, RobotState)
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros import TransformException

PLANNING_GROUP = "arm"
EE_LINK = "end_effector_link"
ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4")
PLAN_TIME_SEC = 8.0
EXECUTE_TIMEOUT_SEC = 30.0
GOAL_TOLERANCES = (0.01, 0.02, 0.03)
CARTESIAN_STEP = 0.005
CARTESIAN_MIN_FRACTION = 0.95

GRIPPER_OPEN = 0.019
GRIPPER_CLOSE = 0.00
GRIPPER_MAX_EFFORT = 5.0

HOME_JOINTS_DEG = [0.0, -57.0, 40.0, 17.0]


def err_name(code: int) -> str:
    for k, v in vars(MoveItErrorCodes).items():
        if k.isupper() and isinstance(v, int) and not isinstance(v, bool) and v == code:
            return f"{k}({code})"
    return str(code)


def run_steps(logger, steps, cancel=None) -> bool:
    """steps = [(name, callable_returning_bool), ...]. Stops at first failure or cancel."""
    for name, fn in steps:
        if cancel is not None and cancel.is_set():
            logger.warn(f"--- CANCELLED before: {name}")
            return False
        logger.info(f"--- {name}")
        if not fn():
            logger.error(f"--- FAILED: {name}")
            return False
    return True


class ArmMotion:
    def __init__(self, node, tf_buffer, arm_base_frame="base_link", cancel_event=None):
        self.node = node
        self.tf_buffer = tf_buffer
        self.arm_base_frame = arm_base_frame
        self.cancel_event = cancel_event or threading.Event()
        self.joint_state = None
        self._handles = set()          # goal handles currently running (for cancel)

        cbg = ReentrantCallbackGroup()
        node.create_subscription(JointState, "/joint_states", self._joint_cb, 10,
                                 callback_group=cbg)
        self.move_client = ActionClient(node, MoveGroup, "/move_action", callback_group=cbg)
        self.cartesian_client = node.create_client(
            GetCartesianPath, "/compute_cartesian_path", callback_group=cbg)
        self.execute_client = ActionClient(node, ExecuteTrajectory, "/execute_trajectory",
                                           callback_group=cbg)
        self.gripper = ActionClient(node, GripperCommand,
                                    "/gripper_controller/gripper_cmd", callback_group=cbg)
        self.ik_client = node.create_client(GetPositionIK, "/compute_ik",
                                            callback_group=cbg)

    # ------------------------------------------------------------ helpers
    @property
    def log(self):
        return self.node.get_logger()

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def _joint_cb(self, msg: JointState):
        self.joint_state = msg

    def wait_for(self, future, timeout, abort_on_cancel=True) -> bool:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done():
            if (abort_on_cancel and self.cancelled) or time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return future.done()

    def request_cancel(self):
        """Call from any thread: cancels the MoveGroup / ExecuteTrajectory goal in flight."""
        for h in list(self._handles):
            try:
                h.cancel_goal_async()
            except Exception as e:  # noqa: BLE001
                self.log.warn(f"cancel_goal_async failed: {e}")

    def wait_for_interfaces(self, timeout=None) -> bool:
        self.log.info("Waiting for MoveIt, gripper and joint states...")
        end = None if timeout is None else time.monotonic() + timeout

        def expired():
            return (end is not None and time.monotonic() > end) or self.cancelled

        while rclpy.ok() and not self.cartesian_client.wait_for_service(timeout_sec=0.5):
            if expired():
                return False
        for client in (self.move_client, self.execute_client, self.gripper):
            while rclpy.ok() and not client.wait_for_server(timeout_sec=0.5):
                if expired():
                    return False
        while rclpy.ok() and self.joint_state is None:
            if expired():
                return False
            time.sleep(0.1)
        self.log.info("Arm interfaces ready.")
        return True

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
        self.log.info(f"{tag} joints={cur}")

    def ee_pos(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.arm_base_frame, EE_LINK, rclpy.time.Time())
        except TransformException as e:
            self.log.warn(f"EE pose unavailable: {e}")
            return None
        p = t.transform.translation
        return np.array([p.x, p.y, p.z])

    def settle_and_log(self, target) -> bool:
        time.sleep(1.0)
        real = self.ee_pos()
        if real is not None:
            err = float(np.linalg.norm(np.array(target) - real))
            self.log.info(f"REACH real={np.round(real, 3)} err={err * 100:.1f} cm (info only)")
        return True

    # ------------------------------------------------------------ MoveIt
    def _wrap_goal(self, req) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        return goal

    def _send_move_goal(self, goal, label) -> bool:
        send = self.move_client.send_goal_async(goal)
        if not self.wait_for(send, 5.0, abort_on_cancel=False):
            self.log.error(f"{label}: goal send timed out")
            return False
        handle = send.result()
        if handle is None or not handle.accepted:
            self.log.error(f"{label}: goal rejected")
            return False
        self._handles.add(handle)
        try:
            if self.cancelled:
                handle.cancel_goal_async()
                return False
            rf = handle.get_result_async()
            if not self.wait_for(rf, PLAN_TIME_SEC + EXECUTE_TIMEOUT_SEC):
                handle.cancel_goal_async()
                self.log.error(f"{label}: cancelled or timed out")
                return False
            code = rf.result().result.error_code.val
            if code != MoveItErrorCodes.SUCCESS:
                self.log.warn(f"{label} failed: {err_name(code)}")
                return False
            return True
        finally:
            self._handles.discard(handle)

    def build_position_goal(self, x, y, z, tol) -> MoveGroup.Goal:
        pc = PositionConstraint()
        pc.header.frame_id = self.arm_base_frame
        pc.link_name = EE_LINK
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [float(tol)]
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
        return self._wrap_goal(req)

    def move_to(self, x, y, z) -> bool:
        if self.joint_state is None:
            self.log.error("No /joint_states received")
            return False
        self.log.info(f"MOVE_TO ({self.arm_base_frame}) x={x:.3f} y={y:.3f} z={z:.3f} "
                      f"| radial={math.hypot(x, y):.3f} m")
        self.log_joints("  start")
        for tol in GOAL_TOLERANCES:
            self.log.info(f"  MoveGroup goal, tolerance={tol * 100:.0f} cm")
            if self._send_move_goal(self.build_position_goal(x, y, z, tol), "  MoveGroup"):
                self.log_joints("  MoveGroup SUCCESS, now")
                return True
            if self.cancelled:
                return False
        self.log.error(f"No plan found for ({x:.3f}, {y:.3f}, {z:.3f}) with any tolerance")
        return False

    def move_home(self) -> bool:
        if self.joint_state is None:
            self.log.error("No joint states available")
            return False
        req = MotionPlanRequest()
        req.group_name = PLANNING_GROUP
        req.pipeline_id = "ompl"
        req.start_state = self.current_state()
        req.allowed_planning_time = PLAN_TIME_SEC
        req.num_planning_attempts = 10
        req.max_velocity_scaling_factor = 0.2
        req.max_acceleration_scaling_factor = 0.2

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

        if not self._send_move_goal(self._wrap_goal(req), "Home"):
            return False
        self.log_joints("HOME reached")
        return True

    def move_straight(self, x, y, z, avoid_collisions=True,
                      min_fraction=CARTESIAN_MIN_FRACTION) -> bool:
        if self.joint_state is None:
            self.log.error("No /joint_states received")
            return False
        try:
            t = self.tf_buffer.lookup_transform(
                self.arm_base_frame, EE_LINK, rclpy.time.Time())
        except TransformException as e:
            self.log.error(f"EE pose unavailable: {e}")
            return False
        self.log.info(f"MOVE_STRAIGHT x={x:.3f} y={y:.3f} z={z:.3f} "
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
            self.log.error("  CARTESIAN: service call timed out/cancelled")
            return False
        res = future.result()
        if res is None:
            self.log.error("  CARTESIAN: no response")
            return False
        self.log.info(f"  CARTESIAN: fraction={res.fraction:.2f} "
                      f"code={err_name(res.error_code.val)}")
        if res.fraction < min_fraction:
            self.log.warn(f"  CARTESIAN: only {res.fraction:.0%} of the path is reachable")
            return False
        return self.execute(res.solution)

    def execute(self, trajectory) -> bool:
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        send = self.execute_client.send_goal_async(goal)
        if not self.wait_for(send, 5.0, abort_on_cancel=False):
            self.log.error("  EXECUTE: goal send timed out")
            return False
        handle = send.result()
        if handle is None or not handle.accepted:
            self.log.error("  EXECUTE: trajectory goal rejected")
            return False
        self._handles.add(handle)
        try:
            if self.cancelled:
                handle.cancel_goal_async()
                return False
            rf = handle.get_result_async()
            if not self.wait_for(rf, EXECUTE_TIMEOUT_SEC):
                handle.cancel_goal_async()
                self.log.error("  EXECUTE: cancelled or timed out")
                return False
            result = rf.result().result
            if result.error_code.val != result.error_code.SUCCESS:
                self.log.error(f"  EXECUTE failed: {err_name(result.error_code.val)}")
                return False
            self.log.info("  EXECUTE ok")
            return True
        finally:
            self._handles.discard(handle)

    # ------------------------------------------------------------ gripper
    def set_gripper(self, position, closing=False) -> bool:
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = GRIPPER_MAX_EFFORT
        send = self.gripper.send_goal_async(goal)
        if not self.wait_for(send, 5.0):
            return False
        handle = send.result()
        if handle is None or not handle.accepted:
            self.log.error("Gripper goal rejected")
            return False
        rf = handle.get_result_async()
        if not self.wait_for(rf, 5.0):
            return False
        result = rf.result().result
        self.log.info(f"Gripper: pos={result.position:.4f} stalled={result.stalled} "
                      f"reached={result.reached_goal}")
        time.sleep(0.3)
        return result.reached_goal or (closing and result.stalled)

    # ------------------------------------------------------------ diagnostics
    def probe_reach(self, bx, by, bz):
        """Diagnostic only: does MoveIt's IK accept these points?"""
        for dz in (0.12, 0.0):
            req = GetPositionIK.Request()
            req.ik_request.group_name = PLANNING_GROUP
            req.ik_request.robot_state = self.current_state()
            req.ik_request.ik_link_name = EE_LINK
            req.ik_request.avoid_collisions = False
            req.ik_request.timeout = Duration(seconds=1.0).to_msg()
            ps = PoseStamped()
            ps.header.frame_id = self.arm_base_frame
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = bx, by, bz + dz
            ps.pose.orientation.w = 1.0
            req.ik_request.pose_stamped = ps
            fut = self.ik_client.call_async(req)
            if not self.wait_for(fut, 3.0):
                self.log.warn(f"PROBE z{dz:+.2f}: service timed out")
                continue
            code = fut.result().error_code.val
            self.log.info(f"PROBE ball z{dz:+.2f}: ({bx:.3f}, {by:.3f}, {bz + dz:.3f}) "
                          f"-> {err_name(code)}")