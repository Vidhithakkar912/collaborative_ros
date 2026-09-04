#!/usr/bin/env python3
"""
multi_robot_planner.py
======================
Four planning modes with RAG-based command resolution.

RAG Pipeline:
  1. At startup: embed all KB entries from planning_config.yaml → FAISS index
  2. At query:
     a. Fast regex pre-parse (instant, covers 95% of commands)
     b. RAG retrieve → focused prompt → LLM (fallback for unknown commands)

Planning Modes:
  Mode 1  auto_location       "go to room1"
  Mode 2  waypoint_chain      "tb1 go to room1 then room2"
  Mode 3  auto_object         "go to red object"
  Mode 4  named_robot_object  "tb1 go to red object"
"""

import os, yaml, json, math, time, threading, regex, re
from collections import deque
from threading import Lock, Thread

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration

from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseArray, Pose, PoseStamped
from nav_msgs.msg import Odometry

import numpy as np

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from transforms3d.euler import euler2quat
from ament_index_python.packages import get_package_share_directory

import subprocess
import signal




def load_config(pkg_share: str) -> dict:
    """
    Load planning_config.yaml — single file containing both
    system config AND rag_knowledge_base entries.
    """
    path = os.path.join(pkg_share, "config", "planner.yaml")
    with open(path) as f:
        return yaml.safe_load(f)



class NavigationWorker:
    """Wraps BasicNavigator for one robot. Non-blocking via daemon Thread."""

    def __init__(self, namespace: str, cmd_vel_pub, cfg: dict, logger):
        self._ns        = namespace.rstrip("/")
        self._vel_pub   = cmd_vel_pub
        self._cfg       = cfg
        self._log       = logger
        self._nav       = BasicNavigator(namespace=self._ns)
        self._lock      = Lock()
        self._busy      = False
        self.visited_objects: list = []
        self._stop_dist = cfg["object_tracking"]["stop_offset_m"]
        self._visit_r   = cfg["object_tracking"]["visited_radius_m"]

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._busy

    def _set_busy(self, v: bool):
        with self._lock:
            self._busy = v

    def stop(self):
        self._vel_pub.publish(Twist())

    def is_visited(self, pose: Pose) -> bool:
        return any(
            math.sqrt((pose.position.x - vx)**2 +
                      (pose.position.y - vy)**2) < self._visit_r
            for vx, vy in self.visited_objects
        )

    def send_goal_async(self, goal_pose: PoseStamped, on_complete=None):
        if self.is_busy:
            self._log.warn(f"[{self._ns}] Already navigating, ignoring goal")
            return
        Thread(target=self._navigate, args=(goal_pose, on_complete),
               daemon=True).start()

    def _navigate(self, goal_pose: PoseStamped, on_complete):
        self._set_busy(True)
        self.stop()
        time.sleep(0.3)
        try:
            self._nav.goToPose(goal_pose)
            while not self._nav.isTaskComplete():
                time.sleep(0.1)
            result = self._nav.getResult()
            status = {TaskResult.SUCCEEDED: "succeeded",
                      TaskResult.CANCELED:  "canceled",
                      TaskResult.FAILED:    "failed"}.get(result, "unknown")
            self._log.info(f"[{self._ns}] Nav result: {status}")
            if on_complete:
                on_complete(result, goal_pose)
        finally:
            self._set_busy(False)

    def send_goal_pose(self, target: Pose, robot_x: float, robot_y: float,
                       map_frame: str = "map", add_offset: bool = True,
                       on_complete=None):
        theta = math.atan2(target.position.y - robot_y,
                           target.position.x - robot_x)
        d = self._stop_dist if add_offset else 0.0

        gp = PoseStamped()
        gp.header.frame_id    = map_frame
        gp.header.stamp       = self._nav.get_clock().now().to_msg()
        gp.pose.position.x    = target.position.x - d * math.cos(theta)
        gp.pose.position.y    = target.position.y - d * math.sin(theta)
        gp.pose.position.z    = 0.0
        w, _, _, z            = euler2quat(0, 0, theta)
        gp.pose.orientation.w = w
        gp.pose.orientation.z = z

        self.send_goal_async(gp, on_complete)


# ══════════════════════════════════════════════════════════════════════════════
# MultiRobotPlanner
# ══════════════════════════════════════════════════════════════════════════════

class MultiRobotPlanner(Node):
    """
    Central planner. Command resolution pipeline:

      1. Fast regex   → instant,  covers ~95% of real commands
      2. RAG + LLM    → fallback for unknown/ambiguous commands

    Planning modes:
      1. auto_location       → nearest idle robot to named location
      2. waypoint_chain      → explicit robot, sequential locations
      3. auto_object         → nearest robot to any detected object
      4. named_robot_object  → explicit robot to its detected object
    """

    def __init__(self, cfg: dict, locations: dict, robot_names: list):
        super().__init__("multi_robot_planner")
        self._cfg       = cfg
        self._locations = locations
        self._robots    = robot_names
        self._map_frame = "map"
        self._object_detection_processes = {}
        self._object_tracking_processes = {}
        rp = cfg["robot_selection"]
        self._arrival_thresh = rp["arrival_threshold_m"]
        self._goal_timeout   = rp["goal_timeout_s"]

        self._poses       = {r: {"x": 0.0, "y": 0.0} for r in self._robots}
        self._status      = {r: "idle"               for r in self._robots}
        self._goal_queues = {r: deque()              for r in self._robots}
        self._active_goals: dict = {}
        self._box_poses: dict    = {r: [] for r in self._robots}
        self._box_lock  = Lock()

        # publishers & workers
        self._cmd_vel_pubs = {}
        self._workers: dict = {}
        for robot in self._robots:
            pub = self.create_publisher(Twist, f"/{robot}/cmd_vel", 10)
            self._cmd_vel_pubs[robot] = pub
            self._workers[robot] = NavigationWorker(
                robot, pub, cfg, self.get_logger())

        self._goal_pubs = {
            r: self.create_publisher(PoseStamped, f"/{r}/goal_pose", 10)
            for r in self._robots
        }
        self._object_launch_pub = self.create_publisher(
            String,
            "/start_object_nodes",
            10
            )
        self._decision_pub = self.create_publisher(String, "/robot_assignment", 10)
        self._tracker_pub = self.create_publisher(
            String,
            "/tracker_command",
            10
        )

        # subscriptions
        for robot in self._robots:
            self.create_subscription(
                Odometry, f"/{robot}/odom",
                lambda msg, r=robot: self._odom_cb(msg, r), 10)
            self.create_subscription(
                PoseArray, f"/{robot}/box_poses",
                lambda msg, r=robot: self._box_poses_cb(msg, r), 1)

        self.create_subscription(String, "/chat_input", self._chat_cb, 10)

        # timers
        self.create_timer(0.5, self._publish_queued_goals)
        self.create_timer(1.0, self._check_arrivals)

      

        self.get_logger().info("MultiRobotPlanner READY")

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry, robot: str):
        self._poses[robot]["x"] = msg.pose.pose.position.x
        self._poses[robot]["y"] = msg.pose.pose.position.y

    def _box_poses_cb(self, msg: PoseArray, robot: str):
        with self._box_lock:
            self._box_poses[robot] = list(msg.poses)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _dist_robot_xy(self, robot: str, tx: float, ty: float) -> float:
        rx = self._poses[robot]["x"]
        ry = self._poses[robot]["y"]
        return math.sqrt((rx - tx)**2 + (ry - ty)**2)
    def _start_object_detection(self, robot: str):

        if robot in self._object_detection_processes:
            self.get_logger().info(
                f"[{robot}] object_detection already running")
            return

        self.get_logger().info(
            f"[{robot}] Launching object_detection")

        proc = subprocess.Popen([
        "ros2", "run", "multi_robots", "object_detection",
        "--ros-args",
        "-r", "/tf:=/" + robot + "/tf",
        "-r", "/tf_static:=/" + robot + "/tf_static",
        "-r", "__ns:=/" + robot,
    ])

        self._object_detection_processes[robot] = proc
    def _start_object_tracking(self, robot: str):

        if robot in self._object_tracking_processes:
            self.get_logger().info(
                f"[{robot}] object_tracking already running")
            return

        self.get_logger().info(
            f"[{robot}] Launching object_tracking")

        proc = subprocess.Popen([
        "ros2", "run", "multi_robots", "object_tracking",
        "--ros-args",
        "-r", "/tf:=/" + robot + "/tf",
        "-r", "/tf_static:=/" + robot + "/tf_static",
        "-r", "__ns:=/" + robot,
    ])

        self._object_tracking_processes[robot] = proc
    def _stop_object_tracking(self, robot: str):

        proc = self._object_tracking_processes.get(robot)

        if proc is not None:
            self.get_logger().info(
                f"[{robot}] Stopping object_tracking")

            proc.send_signal(signal.SIGINT)
            proc.wait()

            del self._object_tracking_processes[robot]
    def _nearest_robot(self, tx: float, ty: float,
                        prefer_idle: bool = True) -> str | None:
        idle = [r for r in self._robots if self._status[r] == "idle"]
        candidates = idle if (prefer_idle and idle) else self._robots
        if not candidates:
            return None
        nearest = min(candidates, key=lambda r: self._dist_robot_xy(r, tx, ty))
        self.get_logger().info(
            f"[DECISION] nearest to ({tx:.2f},{ty:.2f}): {nearest} "
            f"dist={self._dist_robot_xy(nearest, tx, ty):.2f}m")
        return nearest

    def _check_arrivals(self):
        for robot, goal in list(self._active_goals.items()):
            dist    = self._dist_robot_xy(robot, goal["x"], goal["y"])
            elapsed = time.time() - goal["sent_time"]
            if dist < self._arrival_thresh:
                self.get_logger().info(f"[{robot}] Arrived → idle")
                self._status[robot] = "idle"
                del self._active_goals[robot]
            elif elapsed > self._goal_timeout:
                self.get_logger().warn(f"[{robot}] Goal timeout → idle")
                self._status[robot] = "idle"
                del self._active_goals[robot]

    # ── planning modes ────────────────────────────────────────────────────────

    def _mode_auto_location(self, location_name: str):
        if location_name not in self._locations:
            self.get_logger().warn(f"Unknown location: {location_name}"); return
        lx, ly, _ = self._locations[location_name]
        robot = self._nearest_robot(lx, ly)
        if robot is None:
            self.get_logger().error("No robot available"); return
        self._enqueue_location_goal(robot, location_name)
        self._publish_decision(robot, location_name, auto=True)

    def _mode_waypoint_chain(self, robot: str, goals: list):
        if robot not in self._robots:
            self.get_logger().warn(f"Unknown robot: {robot}"); return
        unknown = [g for g in goals if g not in self._locations]
        if unknown:
            self.get_logger().warn(f"Unknown locations: {unknown}"); return
        for loc in goals:
            self._enqueue_location_goal(robot, loc)
        self.get_logger().info(
            f"[{robot}] Waypoint chain: {' → '.join(goals)}")

    def _enqueue_location_goal(self, robot: str, location_name: str):
        x, y, yaw = self._locations[location_name]
        self._goal_queues[robot].append(
            {"x": x, "y": y, "yaw": yaw, "name": location_name})
        self._status[robot] = "busy"

    def _publish_queued_goals(self):
        for robot, q in self._goal_queues.items():
            if not q or robot in self._active_goals:
                continue
            if self._workers[robot].is_busy:
                continue
            goal = q.popleft()
            x, y, yaw = goal["x"], goal["y"], goal["yaw"]
            msg = PoseStamped()
            msg.header.frame_id    = self._map_frame
            msg.header.stamp       = self.get_clock().now().to_msg()
            msg.pose.position.x    = x
            msg.pose.position.y    = y
            msg.pose.position.z    = 0.0
            msg.pose.orientation.z = math.sin(yaw / 2.0)
            msg.pose.orientation.w = math.cos(yaw / 2.0)
            self._goal_pubs[robot].publish(msg)
            self.get_logger().info(
                f"[{robot}] Goal → {goal['name']} ({x:.2f},{y:.2f})")
            self._active_goals[robot] = {
                "x": x, "y": y, "name": goal["name"],
                "sent_time": time.time(),
            }



    def _publish_decision(self, robot: str, location: str, auto: bool):
        x, y = (self._locations[location][:2]
                 if location in self._locations else (0.0, 0.0))
        msg = String()
        msg.data = json.dumps({"robot": robot, "location": location,
                                "x": x, "y": y, "auto": auto})
        self._decision_pub.publish(msg)

 
    def _chat_cb(self, msg: String):
        Thread(target=self._process_chat, args=(msg.data,), daemon=True).start()

    def _process_chat(self, text: str):
        user_text = text.replace("human|", "").strip().lower()
        self.get_logger().info(f"[CHAT] '{user_text}'")

       
        parsed = self._fast_parse(user_text)
        if parsed:
            self.get_logger().info("[STAGE 1] Regex matched — no LLM needed")
            for cmd in parsed.get("commands", []):
                self._dispatch(cmd)
            return

        self.get_logger().warn(
            f"[STAGE 1] No regex match for '{user_text}' → RAG + LLM fallback")

     

        if not parsed:
            self.get_logger().error("[STAGE 2] RAG+LLM failed to resolve command")
            return

        self.get_logger().info("[STAGE 2] RAG+LLM resolved command")
        for cmd in parsed.get("commands", []):
            self._dispatch(cmd)

  
    def _fast_parse(self, text: str) -> dict | None:
        t         = text.strip().lower()
        robot_pat = "|".join(re.escape(r) for r in self._robots)
        loc_pat   = "|".join(re.escape(l) for l in self._locations)
        obj_words = r"(?:red\s+)?(?:object|box|ball)"

        m = re.search(rf'\b({robot_pat})\b', t)
        if m:
            robot     = m.group(1)
            remainder = t[m.end():]
            if re.search(obj_words, remainder):
                self.get_logger().info(f"[FAST] Mode 4: {robot} → object")
                return {"commands": [{"mode": "named_robot_object",
                                      "robot": robot}]}
            move_match = re.search(
                rf'\b({robot_pat})\b.*\b(left|right|forward|backward|stop)\b',
                t
            )

            if move_match:

                robot = move_match.group(1)
                direction = move_match.group(2)

                return {
                    "commands": [{
                        "mode": "movement",
                        "robot": robot,
                        "direction": direction
                    }]
                }
            coord_match = re.search(
                rf'\b({robot_pat})\b.*x\s*(-?\d+\.?\d*).*y\s*(-?\d+\.?\d*)(?:.*yaw\s*(-?\d+\.?\d*))?',
                t
            )

            if coord_match:

                robot = coord_match.group(1)

                x = float(coord_match.group(2))
                y = float(coord_match.group(3))

                yaw = 0.0

                if coord_match.group(4):
                    yaw = float(coord_match.group(4))

                return {
                    "commands": [{
                        "mode": "coordinate",
                        "robot": robot,
                        "x": x,
                        "y": y,
                        "yaw": yaw
                    }]
                }
            stop_match = re.search(
                rf'(stop|cancel|halt)\s*({robot_pat})|({robot_pat})\s*(stop|cancel|halt)',
                t
            )

            if stop_match:

                robot = next(
                    g for g in stop_match.groups()
                    if g in self._robots
                )

                return {
                    "commands": [{
                        "mode": "stop",
                        "robot": robot
                    }]
                }
            locs = re.findall(rf'\b({loc_pat})\b', remainder)
            if locs:
                self.get_logger().info(f"[FAST] Mode 2: {robot} → {locs}")
                return {"commands": [{"mode": "waypoint_chain",
                                      "robot": robot,
                                      "goals": [l.lower() for l in locs]}]}

    
        if re.search(obj_words, t):
            self.get_logger().info("[FAST] Mode 3: auto object")
            return {"commands": [{"mode": "auto_object"}]}

      
        locs = re.findall(rf'\b({loc_pat})\b', t)
        nav  = r'\b(?:go|goto|navigate|send|move)\b'
        if locs and re.search(nav, t):
            self.get_logger().info(f"[FAST] Mode 1: auto location → {locs}")
            return {"commands": [{"mode": "auto_location",
                                  "goals": [l.lower() for l in locs]}]}

        return None

  

    def _dispatch(self, cmd: dict):
        mode  = cmd.get("mode", "")
        robot = cmd.get("robot", "").lower()
        goals = [g.lower() for g in cmd.get("goals", [])]

        self.get_logger().info(
            f"[DISPATCH] mode={mode} robot={robot} goals={goals}")

        if mode == "auto_location":
            for loc in goals:
                self._mode_auto_location(loc)

        elif mode == "waypoint_chain":
            if not robot:
                self.get_logger().warn("waypoint_chain requires robot name"); return
            self._mode_waypoint_chain(robot, goals)

        elif mode == "auto_object":

            nearest_robot = self._nearest_robot(0.0, 0.0)

            if nearest_robot is None:
                self.get_logger().warn("No robot available")
                return

            self._start_object_detection(nearest_robot)

            time.sleep(2.0)

            self._start_object_tracking(nearest_robot)

            self.get_logger().info(
                f"[{nearest_robot}] Started autonomous object search")

        elif mode == "named_robot_object":

            if robot not in self._robots:
                self.get_logger().warn(f"Unknown robot: {robot}")
                return

            self._start_object_detection(robot)

            time.sleep(2.0)

            self._start_object_tracking(robot)

            self.get_logger().info(
                f"[{robot}] Started object search")

        

        elif mode == "stop":

            if robot not in self._robots:
                return

            self.get_logger().warn(f"[{robot}] STOP requested")

            self._workers[robot]._nav.cancelTask()

            self._cmd_vel_pubs[robot].publish(Twist())

            self._status[robot] = "idle"

            if robot in self._active_goals:
                del self._active_goals[robot]
        elif mode == "coordinate":
            if robot in self._robots:
                try:
                    x   = float(cmd.get("x",   0.0))
                    y   = float(cmd.get("y",   0.0))
                    yaw = float(cmd.get("yaw", 0.0))
                    self._goal_queues[robot].append(
                        {"x": x, "y": y, "yaw": yaw, "name": f"({x},{y})"})
                    self._status[robot] = "busy"
                except (TypeError, ValueError) as e:
                    self.get_logger().error(f"Bad coordinates: {e}")

        elif mode == "movement":
            direction = cmd.get("direction", "").lower()
            if robot in self._robots and direction in (
                    "forward", "backward", "left", "right"):
                twist = Twist()
                if   direction == "forward":  twist.linear.x  =  0.3
                elif direction == "backward": twist.linear.x  = -0.3
                elif direction == "left":     twist.angular.z =  0.5
                elif direction == "right":    twist.angular.z = -0.5
                end = time.time() + 2.0
                def _pub_until():
                    while time.time() < end:
                        self._cmd_vel_pubs[robot].publish(twist)
                        time.sleep(0.1)
                    self._cmd_vel_pubs[robot].publish(Twist())
                Thread(target=_pub_until, daemon=True).start()

        else:
            self.get_logger().warn(f"Unknown mode: {mode}")



def main(args=None):
    rclpy.init(args=args)

    pkg_share = get_package_share_directory("multi_robots")
    cfg       = load_config(pkg_share)

    
    robots_path = os.path.join(pkg_share, "config", "robots.yaml")
    with open(robots_path) as f:
        robots_data = yaml.safe_load(f)
    robot_names = [r["name"] for r in robots_data["robots"]
                   if r.get("enabled", True)]
    print(f"Active robots: {robot_names}")

    
    locs_path = os.path.join(pkg_share, "config", "locations.yaml")
    with open(locs_path) as f:
        locs_data = yaml.safe_load(f)
    locations = {
        name.lower(): (float(p["x"]), float(p["y"]), float(p["yaw"]))
        for name, p in locs_data["locations"].items()
    }
    print(f"Locations: {list(locations.keys())}")

    planner  = MultiRobotPlanner(cfg, locations, robot_names)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(planner)

    try:
        print("Spinning ... (Ctrl-C to exit)")
        executor.spin()
    except KeyboardInterrupt:
        print("Shutting down ...")
    finally:
        planner.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()