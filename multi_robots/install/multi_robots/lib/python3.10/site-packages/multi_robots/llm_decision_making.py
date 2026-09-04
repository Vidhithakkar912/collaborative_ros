#!/usr/bin/env python3
import os
import yaml
import json
from collections import deque
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from ament_index_python.packages import get_package_share_directory
from llama_cpp import Llama
import regex


class MultiRobotLLMNavigator(Node):

    def __init__(self):
        super().__init__("multi_robot_llm_navigator")

        # Load robot names
        pkg_share = get_package_share_directory("multi_robots")
        robots_path = os.path.join(pkg_share, "config", "robots.yaml")
        with open(robots_path, "r") as f:
            robots_data = yaml.safe_load(f)
        self.robot_names = [r["name"] for r in robots_data["robots"] if r.get("enabled", True)]
        self.get_logger().info(f"Active robots: {self.robot_names}")

        # Load locations
        locations_path = os.path.join(pkg_share, "config", "locations.yaml")
        with open(locations_path, "r") as f:
            locations_data = yaml.safe_load(f)
        self.locations = {
            name.lower(): (float(p["x"]), float(p["y"]), float(p["yaw"]))
            for name, p in locations_data["locations"].items()
        }
        self.get_logger().info(f"Loaded locations: {list(self.locations.keys())}")

        # ── Robot pose tracking (updated by odom subscribers) ─────────────────
        # { robot_name: {"x": float, "y": float} }
        self.robot_poses = {robot: {"x": 0.0, "y": 0.0} for robot in self.robot_names}

        # ── Robot status tracking ─────────────────────────────────────────────
        # "idle" or "busy" — busy robots are skipped during auto-assignment
        self.robot_status = {robot: "idle" for robot in self.robot_names}

        # ── Odometry subscribers — one per robot ──────────────────────────────
        self.odom_subscribers = {}
        for robot in self.robot_names:
            topic = f"/{robot}/odom"
            self.odom_subscribers[robot] = self.create_subscription(
                Odometry,
                topic,
                lambda msg, r=robot: self._odom_callback(msg, r),
                10
            )
            self.get_logger().info(f"Subscribed to {topic}")

        # Goal queues for each robot
        self.goal_queues = {robot: deque() for robot in self.robot_names}

        # Movement command queues for each robot (for cmd_vel)
        self.movement_queues = {robot: deque() for robot in self.robot_names}

        # Goal publishers (PoseStamped)
        self.goal_publishers = {}
        for robot in self.robot_names:
            topic_name = f"/{robot}/goal_pose"
            self.goal_publishers[robot] = self.create_publisher(PoseStamped, topic_name, 10)

        # Velocity publishers (Twist)
        self.cmd_vel_publishers = {}
        for robot in self.robot_names:
            topic_name = f"/{robot}/cmd_vel"
            self.cmd_vel_publishers[robot] = self.create_publisher(Twist, topic_name, 10)

        # ── Decision result publisher ──────────────────────────────────────────
        # Publishes which robot was assigned to which location
        self.decision_pub = self.create_publisher(String, "/robot_assignment", 10)

        # Chat input subscriber
        self.sub = self.create_subscription(
            String,
            '/chat_input',
            self.chat_callback,
            10
        )
        self.get_logger().info("Subscriber /chat_input created")

        # Timer for publishing goals
        self.create_timer(0.5, self.publish_goals)

        # Timer for publishing movement commands
        self.create_timer(0.1, self.publish_movements)

        # ── Timer to check if robots reached goal and mark idle ───────────────
        self.create_timer(1.0, self._check_goal_completion)

        # Movement parameters
        self.linear_speed = 0.3
        self.angular_speed = 0.5
        self.movement_duration = 2.0

        # Track active movements
        self.active_movements = {}

        # ── Track active goals per robot (to detect arrival) ──────────────────
        # { robot: {"x": float, "y": float, "sent_time": float} }
        self.active_goals = {}

        # Distance threshold to consider robot has "arrived" (metres)
        self.arrival_threshold = 0.3

        # Load LLM
        self.get_logger().info("Loading LLM model...")
        start = time.time()
        self.llm = Llama(
            model_path="/home/einfochips/models/Phi-3-mini-4k-instruct-q4.gguf",
            n_ctx=512,
            n_threads=6,
            n_gpu_layers=0,
            verbose=False
        )
        self.get_logger().info(f"LLM loaded in {time.time() - start:.2f} seconds")
        self.get_logger().info("Multi-Robot LLM Navigator READY")

    # ═══════════════════════════════════════════════════════════════════════════
    # Odometry callback — update robot pose
    # ═══════════════════════════════════════════════════════════════════════════
    def _odom_callback(self, msg: Odometry, robot: str):
        """Update robot's current position from odometry."""
        self.robot_poses[robot]["x"] = msg.pose.pose.position.x
        self.robot_poses[robot]["y"] = msg.pose.pose.position.y

    # ═══════════════════════════════════════════════════════════════════════════
    # Core decision making
    # ═══════════════════════════════════════════════════════════════════════════
    def _distance(self, robot: str, location_name: str) -> float:
        """Calculate Euclidean distance from robot to a location."""
        rx = self.robot_poses[robot]["x"]
        ry = self.robot_poses[robot]["y"]
        lx, ly, _ = self.locations[location_name]
        return math.sqrt((rx - lx) ** 2 + (ry - ly) ** 2)

    def _find_nearest_robot(self, location_name: str,
                             prefer_idle: bool = True) -> str | None:
        """
        Find the nearest robot to a given location.

        Args:
            location_name: target location key
            prefer_idle:   if True, only consider idle robots first;
                           fall back to all robots if none are idle
        Returns:
            robot name string, or None if no robots available
        """
        if location_name not in self.locations:
            self.get_logger().warn(f"Unknown location: {location_name}")
            return None

        candidates = (
            [r for r in self.robot_names if self.robot_status[r] == "idle"]
            if prefer_idle else self.robot_names
        )

        # If no idle robots, fall back to all robots
        if not candidates:
            self.get_logger().warn("No idle robots — assigning to closest busy robot")
            candidates = self.robot_names

        if not candidates:
            return None

        nearest = min(candidates, key=lambda r: self._distance(r, location_name))
        dist    = self._distance(nearest, location_name)

        self.get_logger().info(
            f"[DECISION] Nearest robot to '{location_name}': {nearest} "
            f"(distance={dist:.2f}m, status={self.robot_status[nearest]})"
        )

        # Log all distances for transparency
        for robot in self.robot_names:
            d = self._distance(robot, location_name)
            self.get_logger().info(
                f"  {robot}: {d:.2f}m [{self.robot_status[robot]}]"
            )

        return nearest

    def _assign_robot_to_location(self, location_name: str,
                                   forced_robot: str | None = None):
        """
        Assign the best robot to a location.

        If forced_robot is given, use that robot directly (explicit command).
        Otherwise auto-select the nearest idle robot.
        """
        if forced_robot:
            robot = forced_robot
            if robot not in self.robot_names:
                self.get_logger().warn(f"Unknown robot: {robot}")
                return
        else:
            robot = self._find_nearest_robot(location_name)
            if robot is None:
                self.get_logger().error("No robot available for assignment.")
                return

        if location_name not in self.locations:
            self.get_logger().warn(f"Unknown location: {location_name}")
            return

        x, y, yaw = self.locations[location_name]
        self.goal_queues[robot].append({
            "x": x, "y": y, "yaw": yaw, "name": location_name
        })
        self.robot_status[robot] = "busy"
        self.get_logger().info(
            f"[ASSIGNED] {robot} → '{location_name}' ({x:.2f}, {y:.2f})")

        # Publish assignment decision
        decision_msg      = String()
        decision_msg.data = json.dumps({
            "robot":    robot,
            "location": location_name,
            "x":        x,
            "y":        y,
            "auto":     forced_robot is None,
        })
        self.decision_pub.publish(decision_msg)

    def _check_goal_completion(self):
        """
        Periodically check if robots have arrived at their goals
        and mark them idle again.
        """
        for robot, goal in list(self.active_goals.items()):
            dist = math.sqrt(
                (self.robot_poses[robot]["x"] - goal["x"]) ** 2 +
                (self.robot_poses[robot]["y"] - goal["y"]) ** 2
            )
            elapsed = time.time() - goal["sent_time"]

            if dist < self.arrival_threshold:
                self.get_logger().info(
                    f"[{robot}] Arrived at goal (dist={dist:.2f}m) → idle")
                self.robot_status[robot] = "idle"
                del self.active_goals[robot]

            elif elapsed > 60.0:
                # Timeout fallback — mark idle after 60 s regardless
                self.get_logger().warn(
                    f"[{robot}] Goal timeout after 60s → marking idle")
                self.robot_status[robot] = "idle"
                del self.active_goals[robot]

    # ═══════════════════════════════════════════════════════════════════════════
    # Chat callback
    # ═══════════════════════════════════════════════════════════════════════════
    def chat_callback(self, msg: String):
        user_text = msg.data.strip().lower()
        self.get_logger().info(f"Received command: {user_text}")

        prompt = f"""
You are a multi-robot navigation planner.

Robots: {', '.join(self.robot_names)}
Locations: {', '.join(self.locations.keys())}

You can handle FOUR types of commands:

1. LOCATION-BASED (explicit robot): User specifies which robot goes where
   Format: {{"commands":[{{"robot":"tb1","type":"location","goals":["kitchen"]}}]}}

2. AUTO-ASSIGN: User says "go to kitchen" or "send a robot to kitchen" without
   specifying which robot — system picks the nearest idle one automatically
   Format: {{"commands":[{{"robot":"auto","type":"location","goals":["kitchen"]}}]}}

3. COORDINATE-BASED: Send robot to specific x,y coordinates
   Format: {{"commands":[{{"robot":"tb1","type":"coordinate","x":1.5,"y":2.0,"yaw":0.0}}]}}

4. MOVEMENT: Move robot in a direction (forward, backward, left, right)
   Format: {{"commands":[{{"robot":"tb1","type":"movement","direction":"forward"}}]}}

Respond with **only JSON**. No explanation.

Examples:
- "tb1 go to kitchen"          → explicit robot, location type
- "send a robot to kitchen"    → robot:"auto", location type
- "go to kitchen"              → robot:"auto", location type
- "which robot is nearest"     → robot:"auto", location type
- "send tb2 to 3.5, 2.0"       → coordinate type
- "tb1 move forward"           → movement type

User request: {user_text}
"""
        result     = self.llm(prompt, max_tokens=200, temperature=0.0, top_p=1.0)
        raw_output = result["choices"][0]["text"].strip()
        self.get_logger().info(f"LLM Output:\n{raw_output}")

        match = regex.search(r'\{(?:[^{}]|(?0))*\}', raw_output)
        if not match:
            self.get_logger().error("No JSON detected from LLM.")
            return

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            self.get_logger().error("Invalid JSON from LLM.")
            return

        for cmd in parsed.get("commands", []):
            robot    = cmd.get("robot")
            cmd_type = cmd.get("type", "location")

            if cmd_type == "location":
                goals = cmd.get("goals", [])
                for goal in goals:
                    goal = goal.lower()
                    if robot == "auto":
                        # ── AUTO DECISION: pick nearest idle robot ────────────
                        self._assign_robot_to_location(goal)
                    else:
                        # ── EXPLICIT: user named a specific robot ─────────────
                        if robot not in self.robot_names:
                            self.get_logger().warn(f"Unknown robot: {robot}")
                            continue
                        self._assign_robot_to_location(goal, forced_robot=robot)

            elif cmd_type == "coordinate":
                if robot not in self.robot_names:
                    self.get_logger().warn(f"Unknown robot: {robot}")
                    continue
                self._handle_coordinate_command(robot, cmd)

            elif cmd_type == "movement":
                if robot not in self.robot_names:
                    self.get_logger().warn(f"Unknown robot: {robot}")
                    continue
                self._handle_movement_command(robot, cmd)

            else:
                self.get_logger().warn(f"Unknown command type: {cmd_type}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Existing handlers
    # ═══════════════════════════════════════════════════════════════════════════
    def _handle_coordinate_command(self, robot: str, cmd: dict):
        try:
            x   = float(cmd.get("x",   0.0))
            y   = float(cmd.get("y",   0.0))
            yaw = float(cmd.get("yaw", 0.0))
            self.goal_queues[robot].append({
                "x": x, "y": y, "yaw": yaw, "name": f"({x},{y})"
            })
            self.robot_status[robot] = "busy"
            self.get_logger().info(f"[{robot}] Queued coordinates → ({x},{y})")
        except (TypeError, ValueError) as e:
            self.get_logger().error(f"Invalid coordinates: {e}")

    def _handle_movement_command(self, robot: str, cmd: dict):
        direction = cmd.get("direction", "").lower()
        if direction not in ["forward", "backward", "left", "right"]:
            self.get_logger().warn(f"Invalid direction: {direction}")
            return
        end_time = time.time() + self.movement_duration
        self.active_movements[robot] = {
            "direction": direction,
            "end_time":  end_time
        }
        self.get_logger().info(
            f"[{robot}] Movement → {direction} for {self.movement_duration}s")

    def publish_goals(self):
        """Publish the next goal from queue for each robot."""
        for robot, queue in self.goal_queues.items():
            if not queue:
                continue

            next_goal = queue.popleft()
            x    = next_goal["x"]
            y    = next_goal["y"]
            yaw  = next_goal["yaw"]
            name = next_goal["name"]

            goal_msg = PoseStamped()
            goal_msg.header.frame_id       = "map"
            goal_msg.header.stamp          = self.get_clock().now().to_msg()
            goal_msg.pose.position.x       = x
            goal_msg.pose.position.y       = y
            goal_msg.pose.position.z       = 0.0
            goal_msg.pose.orientation.z    = math.sin(yaw / 2.0)
            goal_msg.pose.orientation.w    = math.cos(yaw / 2.0)

            self.goal_publishers[robot].publish(goal_msg)
            self.get_logger().info(f"[{robot}] Published goal → {name}")

            # Track this goal for arrival detection
            self.active_goals[robot] = {
                "x":         x,
                "y":         y,
                "name":      name,
                "sent_time": time.time()
            }

    def publish_movements(self):
        current_time  = time.time()
        robots_to_stop = []

        for robot, movement in self.active_movements.items():
            if current_time >= movement["end_time"]:
                self._publish_stop(robot)
                robots_to_stop.append(robot)
                self.get_logger().info(f"[{robot}] Movement complete → stopped")
            else:
                self._publish_velocity(robot, movement["direction"])

        for robot in robots_to_stop:
            del self.active_movements[robot]

    def _publish_velocity(self, robot: str, direction: str):
        twist = Twist()
        if   direction == "forward":  twist.linear.x  =  self.linear_speed
        elif direction == "backward": twist.linear.x  = -self.linear_speed
        elif direction == "left":     twist.angular.z =  self.angular_speed
        elif direction == "right":    twist.angular.z = -self.angular_speed
        self.cmd_vel_publishers[robot].publish(twist)

    def _publish_stop(self, robot: str):
        self.cmd_vel_publishers[robot].publish(Twist())
        self.robot_status[robot] = "idle"


def main():
    rclpy.init()
    node = MultiRobotLLMNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()