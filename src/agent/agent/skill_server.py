#!/usr/bin/env python3
"""Per-robot SKILL SERVER (one per robot, runs persistently, started with the robot bringup).

Contract with the planner (all relative -> become /<robot>/... when run with __ns:=/<robot>):
    skill_cmd    (String JSON)  {"id": "...", "skill": "pick"|"place"|"find",
                                 "object": "ball", "target": "box"}
    skill_cancel (String)       anything -> cancel the running skill NOW
    skill_status (String JSON)  {"id","skill","state","msg","holding"}
                                 state: idle | running | succeeded | failed | cancelled
                                 (published every 1 s as heartbeat + once at the end)
    detected_objects (DetectedObjectArray, map frame) -> input from your detection node

Skills
    find  : wait for object -> Nav2 to FIND_STANDOFF -> stop
    pick  : wait for object -> Nav2 if far -> fine align -> PICK -> stop
    place : find box -> Nav2 to standoff -> PLACE -> stop

CANCEL = set flag -> Nav2 goal cancelled (by the worker thread) + MoveGroup/Execute goal
cancelled + zero cmd_vel. The base is ALWAYS stopped when a skill ends, whatever the result.
"""

import json
import math
import threading
import time
import traceback
from collections import defaultdict

import numpy as np
import rclpy
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transforms)
from geometry_msgs.msg import PointStamped, PoseStamped, Twist, TwistStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from object_interfaces.msg import DetectedObjectArray
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import TransformException


from agent.arm_motion import ArmMotion
from agent.object_pick import Picker
from agent.object_place import Placer
from agent.process_manager import ProcessManager


DETECTION_MAX_AGE_SEC = 3.0
LOOK_WAIT_SEC = 3.0
LATEST_MAX_AGE_SEC = 10.0

BALL_NAV_THRESHOLD = 0.50
NAV_STANDOFF_BALL = 0.25
NAV_STANDOFF_BOX = 0.50
FIND_STANDOFF = 0.50
NAV_MIN_MOVE = 0.25
NAV_TIMEOUT_SEC = 120.0

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


def limit(value, lo, hi):
    return math.copysign(min(max(abs(value), lo), hi), value)


class SkillServer(Node):

    def __init__(self):
        super().__init__("skill_server")
        for name, default in (("global_frame", "map"), ("base_frame", "base_link"),
                              ("arm_base_frame", "base_link"),
                              ("cmd_vel_topic", "/cmd_vel_"), ("cmd_vel_stamped", True),
                              ("nav_namespace", ""), ("enable_arm", True),
                              ("detect_wait_sec", 30.0),
                              ("manage_detection", False),
                              ("detection_pkg", "agent"),
                              ("detection_exec", "multi_robot_detection")):
            self.declare_parameter(name, default)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.global_frame = gp("global_frame")
        self.base_frame = gp("base_frame")
        self.arm_base_frame = gp("arm_base_frame")
        self._stamped = bool(gp("cmd_vel_stamped"))
        self._detect_wait = float(gp("detect_wait_sec"))
        self._manage_det = bool(gp("manage_detection"))
        det_ns = self.get_namespace().rstrip("/")
        self._det_cmd = ["ros2", "run", gp("detection_pkg"), gp("detection_exec"),
                         "--ros-args"]
        if det_ns:
            self._det_cmd += ["-r", f"__ns:={det_ns}", "-r", f"/tf:={det_ns}/tf",
                              "-r", f"/tf_static:={det_ns}/tf_static"]
        self._procs = ProcessManager(self.get_logger())

        self._lock = threading.Lock()
        self._seen = {}
        self._seq = defaultdict(int)
        self._cancel = threading.Event()
        self._busy = threading.Lock()         
        self._current = None
        self._fail_step = ""
        self.holding_ball = False

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        cbg = ReentrantCallbackGroup()
        self.create_subscription(DetectedObjectArray, "detected_objects",
                                 self._objects_cb, 10, callback_group=cbg)
        self.create_subscription(String, "skill_cmd", self._cmd_cb, 10, callback_group=cbg)
        self.create_subscription(String, "skill_cancel", self._cancel_cb, 10,
                                 callback_group=cbg)
        self.create_subscription(String, "detection_cmd", self._det_cb, 10,
                                 callback_group=cbg)
        self.status_pub = self.create_publisher(String, "skill_status", 10)
        msg_type = TwistStamped if self._stamped else Twist
        self.cmd_vel_pub = self.create_publisher(msg_type, gp("cmd_vel_topic"), 10)
        self.create_timer(1.0, self._heartbeat, callback_group=cbg)

        ns = gp("nav_namespace")
        nav_kwargs = {"node_name": f"{self.get_name()}_nav"}
        if ns:
            nav_kwargs["namespace"] = ns
        self.navigator = BasicNavigator(**nav_kwargs)

        self.arm = self.picker = self.placer = None
        if gp("enable_arm"):
            self.arm = ArmMotion(self, self.tf_buffer, self.arm_base_frame, self._cancel)
            self.picker = Picker(self.arm)
            self.placer = Placer(self.arm)

        self._skills = {"pick": self.skill_pick, "place": self.skill_place,
                        "find": self.skill_find}
        self.get_logger().info("SkillServer ready (idle).")

    
    def _publish_status(self, state, msg="", sid=None, skill=None):
        cur = self._current or {}
        d = {"id": sid if sid is not None else cur.get("id", ""),
             "skill": skill if skill is not None else cur.get("skill", ""),
             "state": state, "msg": msg, "holding": self.holding_ball}
        self.status_pub.publish(String(data=json.dumps(d)))

    def _det_cb(self, msg: String):
        cmd = msg.data.strip().lower()
        if not self._manage_det:
            self.get_logger().warn("detection_cmd ignored (manage_detection:=false)")
            return
        if cmd == "start":
            self._procs.start("det", self._det_cmd)
        elif cmd == "stop":                      # kill blocks a few s -> own thread
            threading.Thread(target=self._procs.stop, args=("det",), daemon=True).start()

    def _heartbeat(self):
        self._publish_status("running" if self._current else "idle")

    def _cmd_cb(self, msg: String):
        try:
            spec = json.loads(msg.data)
        except json.JSONDecodeError:
            self._publish_status("failed", "bad json", sid="", skill="")
            return
        sid, skill = spec.get("id", ""), spec.get("skill", "")
        if not self._busy.acquire(blocking=False):
            self._publish_status("failed", "skill server busy", sid=sid, skill=skill)
            return
        self._cancel.clear()
        threading.Thread(target=self._run, args=(spec,), daemon=True).start()

    def _cancel_cb(self, _msg):
        self.get_logger().warn("CANCEL requested")
        self._cancel.set()
        if self.arm is not None:
            self.arm.request_cancel()          # MoveGroup / ExecuteTrajectory goal
        self.stop_base()                       # 0 cmd_vel right now
        # Nav2 goal is cancelled by the worker thread (see navigate_to) -- a second thread
        # must not spin the BasicNavigator at the same time.

    def _run(self, spec):
        sid, skill = spec.get("id", ""), spec.get("skill", "")
        self._current = {"id": sid, "skill": skill}
        self._fail_step = ""
        self._publish_status("running", sid=sid, skill=skill)
        ok, msg = False, ""
        try:
            fn = self._skills.get(skill)
            if fn is None:
                msg = f"unknown skill '{skill}'"
            else:
                ok = fn(spec)
                if not ok and self._fail_step:
                    msg = f"failed at: {self._fail_step}"
        except Exception as e:  # noqa: BLE001
            msg = f"crashed: {e!r}"
            self.get_logger().error(traceback.format_exc())
        finally:
            self.stop_base()
        state = "cancelled" if self._cancel.is_set() else ("succeeded" if ok else "failed")
        self._publish_status(state, msg, sid=sid, skill=skill)
        self._current = None
        self._busy.release()

    def _steps(self, steps) -> bool:
        for name, fn in steps:
            if self._cancel.is_set():
                self._fail_step = f"cancelled before {name}"
                return False
            self.get_logger().info(f"--- {name}")
            if not fn():
                self._fail_step = name
                self.get_logger().error(f"--- FAILED: {name}")
                return False
        return True

    # ================================================================= skills
    def skill_find(self, spec) -> bool:
        obj = spec.get("object") or "ball"
        return self._steps([
            (f"Wait for {obj}", lambda: self.wait_for_kind(obj)),
            (f"Nav2 to {obj}", lambda: self.approach(obj, FIND_STANDOFF, reacquire=False)),
        ])

    def skill_pick(self, spec) -> bool:
        obj = spec.get("object") or "ball"
        return self._steps([
            (f"Wait for {obj}", lambda: self.wait_for_kind(obj)),
            ("Nav2 if far", lambda: self.approach_ball(obj)),
            ("Fine align (cmd_vel)", lambda: self.fine_align(obj)),
            ("Pick", lambda: self.do_pick(obj)),
        ])

    def skill_place(self, spec) -> bool:
        target = spec.get("target") or "box"
        return self._steps([
            (f"Find {target}", lambda: self.search_for(target)),
            (f"Nav2 to {target}", lambda: self.approach(target, NAV_STANDOFF_BOX)),
            ("Place", lambda: self.do_place(target)),
        ])

    def ensure_arm(self) -> bool:
        if self.arm is None:
            self.get_logger().error("This skill server runs with enable_arm:=false")
            return False
        return self.arm.wait_for_interfaces(timeout=20.0)

    def do_pick(self, obj) -> bool:
        if not self.ensure_arm():
            return False
        pt = self.locate(obj, self.arm_base_frame, samples=3, min_valid=2)
        if pt is None:
            self.get_logger().error(f"Could not localise {obj} in arm base frame")
            return False
        ok = self.picker.pick(pt.point.x, pt.point.y, pt.point.z)
        self.holding_ball = ok
        return ok

    def do_place(self, target) -> bool:
        if not self.ensure_arm():
            return False
        if not self.holding_ball:
            self.get_logger().warn("Not holding a ball according to the last pick; placing anyway")
        pt = self.locate(target, self.arm_base_frame, samples=3, min_valid=2)
        if pt is None:
            self.get_logger().error(f"Could not localise {target} in arm base frame")
            return False
        ok = self.placer.place(pt.point.x, pt.point.y, pt.point.z)
        if ok:
            self.holding_ball = False
        return ok

    # ================================================================= perception cache
    def _objects_cb(self, msg: DetectedObjectArray):
        now = time.monotonic()
        by_kind = defaultdict(list)
        for obj in msg.objects:
            p = obj.pose.position
            by_kind[obj.class_name.lower()].append(np.array([p.x, p.y, p.z]))
        with self._lock:
            for kind, poses in by_kind.items():
                self._seen[kind] = (now, poses)
                self._seq[kind] += 1

    def visible(self, kind) -> bool:
        with self._lock:
            entry = self._seen.get(kind)
        return entry is not None and time.monotonic() - entry[0] < DETECTION_MAX_AGE_SEC

    def nap(self, sec) -> bool:
        """Cancel-aware sleep. False if cancelled."""
        end = time.monotonic() + sec
        while time.monotonic() < end:
            if self._cancel.is_set() or not rclpy.ok():
                return False
            time.sleep(min(0.05, max(0.0, end - time.monotonic())))
        return not self._cancel.is_set()

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
        """Median of `samples` NEW detections (robot must be stationary)."""
        with self._lock:
            last_seq = self._seq[kind]
        pts = []
        deadline = time.monotonic() + timeout
        while (rclpy.ok() and not self._cancel.is_set()
               and time.monotonic() < deadline and len(pts) < samples):
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

    # ================================================================= base control
    def _publish_twist(self, lin=0.0, ang=0.0):
        if self._stamped:
            m = TwistStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.twist.linear.x, m.twist.angular.z = float(lin), float(ang)
        else:
            m = Twist()
            m.linear.x, m.angular.z = float(lin), float(ang)
        self.cmd_vel_pub.publish(m)

    def stop_base(self):
        self._publish_twist()

    def pulse(self, linear, angular, duration):
        end = time.monotonic() + duration
        while time.monotonic() < end and rclpy.ok() and not self._cancel.is_set():
            self._publish_twist(linear, angular)
            time.sleep(0.05)
        self.stop_base()

    def wait_for_kind(self, kind, timeout=None) -> bool:
        """Stand still and wait for a detection of `kind`."""
        timeout = self._detect_wait if timeout is None else timeout
        self.stop_base()
        end = time.monotonic() + timeout
        while time.monotonic() < end and rclpy.ok() and not self._cancel.is_set():
            if self.visible(kind):
                self.get_logger().info(f"'{kind}' detected")
                return True
            time.sleep(0.1)
        self.get_logger().warn(f"No '{kind}' detected within {timeout:.0f} s "
                               f"(is detection running / publishing?)")
        return False

    def search_for(self, kind, timeout=SEARCH_TIMEOUT) -> bool:
        """Stand still briefly, then spin until `kind` is detected."""
        self.stop_base()
        look_until = time.monotonic() + LOOK_WAIT_SEC
        while time.monotonic() < look_until and rclpy.ok() and not self._cancel.is_set():
            if self.visible(kind):
                return True
            time.sleep(0.05)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and rclpy.ok() and not self._cancel.is_set():
            if self.visible(kind):
                self.stop_base()
                return True
            self._publish_twist(0.0, SEARCH_SPIN_SPEED)
            time.sleep(0.1)
        self.stop_base()
        self.get_logger().warn(f"No '{kind}' detected while searching")
        return False

    # ================================================================= navigation
    def navigate_to(self, x, y, standoff, min_move=NAV_MIN_MOVE) -> bool:
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
        t0 = time.monotonic()
        while not self.navigator.isTaskComplete():
            if self._cancel.is_set() or time.monotonic() - t0 > NAV_TIMEOUT_SEC:
                self.get_logger().warn("Cancelling Nav2 goal")
                self.navigator.cancelTask()
                self.stop_base()
                return False
            time.sleep(0.1)
        result = self.navigator.getResult()
        self.get_logger().info(f"Nav result: {result}")
        self.stop_base()
        return result == TaskResult.SUCCEEDED

    def approach(self, kind, standoff, reacquire=True) -> bool:
        target = self.latest(kind, self.global_frame)
        if target is None:
            self.get_logger().warn(f"No map pose for '{kind}'")
            return False
        if not self.navigate_to(target.point.x, target.point.y, standoff):
            return False
        return self.search_for(kind) if reacquire else True

    def approach_ball(self, kind) -> bool:
        obj = self.latest(kind, self.arm_base_frame)
        if obj is None:
            self.get_logger().warn(f"approach: no {kind} pose available")
            return False
        dist = math.hypot(obj.point.x, obj.point.y)
        self.get_logger().info(f"{kind} is {dist:.2f} m from arm base")
        if dist <= BALL_NAV_THRESHOLD:
            self.get_logger().info("Within threshold -> skip Nav2, fine align only")
            return True
        target = self.latest(kind, self.global_frame)
        if target is None:
            return False
        if not self.navigate_to(target.point.x, target.point.y, NAV_STANDOFF_BALL,
                                min_move=0.0):
            return False
        if not self.wait_for_kind(kind, timeout=5.0):
            return self.search_for(kind)
        return True

    def fine_align(self, kind) -> bool:
        misses = 0
        for step in range(FINE_MAX_STEPS):
            if self._cancel.is_set():
                return False
            pt = self.locate(kind, self.arm_base_frame)
            if pt is None:
                misses += 1
                if misses > FINE_MAX_MISSES:
                    break
                continue
            misses = 0
            x, y = pt.point.x, pt.point.y
            yaw_err = math.atan2(y, x)
            dist_err = math.hypot(x, y) - ARM_STANDOFF_DIST
            self.get_logger().info(f"FineAlign[{step}] x={x:.3f} y={y:.3f} "
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
            if not self.nap(FINE_SETTLE_SEC):
                return False
        self.stop_base()
        self.get_logger().warn("Fine alignment failed")
        return False


def main():
    rclpy.init()
    node = SkillServer()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_base()
        node._procs.stop_all()
        node.navigator.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()