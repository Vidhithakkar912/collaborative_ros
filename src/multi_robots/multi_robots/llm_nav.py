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
from ament_index_python.packages import get_package_share_directory
from llama_cpp import Llama
import regex


class MultiRobotLLMNavigator(Node):

    def __init__(self):
        super().__init__("multi_robot_llm_navigator")

        
        pkg_share = get_package_share_directory("multi_robots")
        robots_path = os.path.join(pkg_share, "config", "robots.yaml")
        with open(robots_path, "r") as f:
            robots_data = yaml.safe_load(f)
        self.robot_names = [r["name"] for r in robots_data["robots"] if r.get("enabled", True)]
        self.get_logger().info(f"Active robots: {self.robot_names}")

    
        locations_path = os.path.join(pkg_share, "config", "locations.yaml")
        with open(locations_path, "r") as f:
            locations_data = yaml.safe_load(f)
        self.locations = {
            name.lower(): (float(p["x"]), float(p["y"]), float(p["yaw"]))
            for name, p in locations_data["locations"].items()
        }
        self.get_logger().info(f"Loaded locations: {list(self.locations.keys())}")

        
        self.goal_queues = {robot: deque() for robot in self.robot_names}

        
        self.movement_queues = {robot: deque() for robot in self.robot_names}

        
        self.goal_publishers = {}
        for robot in self.robot_names:
            topic_name = f"/{robot}/goal_pose"
            self.goal_publishers[robot] = self.create_publisher(PoseStamped, topic_name, 10)

        
        self.cmd_vel_publishers = {}
        for robot in self.robot_names:
            topic_name = f"/{robot}/cmd_vel"
            self.cmd_vel_publishers[robot] = self.create_publisher(Twist, topic_name, 10)

        
        self.sub = self.create_subscription(
            String,
            '/chat_input',
            self.chat_callback,
            10
        )
        self.get_logger().info("Subscriber /chat_input created")

        
        self.create_timer(0.5, self.publish_goals)

        
        self.create_timer(0.1, self.publish_movements)

        
        self.linear_speed = 0.3   
        self.angular_speed = 0.5  
        self.movement_duration = 2.0  

        
        self.active_movements = {}  

      
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

    def chat_callback(self, msg: String):
                raw_text  = msg.data.strip()
                
               
                for prefix in ["human|robot", "human|", "user:", "user |", "robot:"]:
                    if raw_text.lower().startswith(prefix.lower()):
                        raw_text = raw_text[len(prefix):].strip()
                        break

                user_text = raw_text.lower()
                self.get_logger().info(f"Cleaned command: '{user_text}'")

                
                self.get_logger().info("=== Current robot positions ===")
                for robot in self.robot_names:
                    rx = self.robot_poses[robot]["x"]
                    ry = self.robot_poses[robot]["y"]
                    status = self.robot_status[robot]
                    self.get_logger().info(
                        f"  {robot}: pos=({rx:.2f}, {ry:.2f}) status={status}")

              
                self.get_logger().info("=== Distances to locations ===")
                for loc in self.locations:
                    lx, ly, _ = self.locations[loc]
                    self.get_logger().info(f"  Location '{loc}': ({lx:.2f}, {ly:.2f})")
                    for robot in self.robot_names:
                        d = self._distance(robot, loc)
                        self.get_logger().info(f"    {robot} → {d:.2f}m")

                prompt = f"""
            You are a multi-robot navigation planner.

            Robots: {', '.join(self.robot_names)}
            Locations: {', '.join(self.locations.keys())}

            IMPORTANT RULES:
            - If the user says "goto room2" or "go to room2" WITHOUT specifying a robot name → use robot:"auto"
            - If the user says "tb1 goto room2" or "send tb1 to room2" → use robot:"tb1"
            - "auto" means the system will pick the nearest robot automatically
            - Only use a specific robot name if the user EXPLICITLY mentions it

            Command types:

            1. LOCATION (explicit robot named by user):
            Format: {{"commands":[{{"robot":"tb1","type":"location","goals":["room2"]}}]}}

            2. LOCATION (no robot specified — AUTO select nearest):
            Format: {{"commands":[{{"robot":"auto","type":"location","goals":["room2"]}}]}}

            3. COORDINATE:
            Format: {{"commands":[{{"robot":"tb1","type":"coordinate","x":1.5,"y":2.0,"yaw":0.0}}]}}

            4. MOVEMENT:
            Format: {{"commands":[{{"robot":"tb1","type":"movement","direction":"forward"}}]}}

            Respond with ONLY the JSON. No explanation. No extra text.

            Examples:
            - "goto room2"              → {{"commands":[{{"robot":"auto","type":"location","goals":["room2"]}}]}}
            - "go to room2"             → {{"commands":[{{"robot":"auto","type":"location","goals":["room2"]}}]}}
            - "robot goto room2"        → {{"commands":[{{"robot":"auto","type":"location","goals":["room2"]}}]}}
            - "tb1 goto room2"          → {{"commands":[{{"robot":"tb1","type":"location","goals":["room2"]}}]}}
            - "send tb3 to room2"       → {{"commands":[{{"robot":"tb3","type":"location","goals":["room2"]}}]}}
            - "tb3 move forward"        → {{"commands":[{{"robot":"tb3","type":"movement","direction":"forward"}}]}}

            User request: {user_text}
            JSON:"""

                result     = self.llm(prompt, max_tokens=100, temperature=0.0, top_p=1.0)
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

                self.get_logger().info(f"Parsed command: {parsed}")

                for cmd in parsed.get("commands", []):
                    robot    = cmd.get("robot")
                    cmd_type = cmd.get("type", "location")

                    if cmd_type == "location":
                        goals = cmd.get("goals", [])
                        for goal in goals:
                            goal = goal.lower()
                            if robot == "auto":
                                
                                self.get_logger().info(
                                    f"AUTO mode → finding nearest robot to '{goal}'")
                                self._assign_robot_to_location(goal)
                            else:
                               
                                if robot not in self.robot_names:
                                    self.get_logger().warn(f"Unknown robot: {robot}")
                                    continue
                                self.get_logger().info(
                                    f"EXPLICIT mode → sending {robot} to '{goal}'")
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
    def _handle_location_command(self, robot: str, cmd: dict):
        """Handle location-based navigation commands."""
        goals = cmd.get("goals", [])
        for goal in goals:
            goal = goal.lower()
            if goal not in self.locations:
                self.get_logger().warn(f"Unknown location: {goal}")
                continue
            x, y, yaw = self.locations[goal]
            self.goal_queues[robot].append({"x": x, "y": y, "yaw": yaw, "name": goal})
            self.get_logger().info(f"[{robot}] Queued location → {goal}")

    def _handle_coordinate_command(self, robot: str, cmd: dict):
        """Handle coordinate-based navigation commands."""
        try:
            x = float(cmd.get("x", 0.0))
            y = float(cmd.get("y", 0.0))
            yaw = float(cmd.get("yaw", 0.0))
            
            self.goal_queues[robot].append({"x": x, "y": y, "yaw": yaw, "name": f"({x}, {y})"})
            self.get_logger().info(f"[{robot}] Queued coordinates → ({x}, {y}, yaw={yaw})")
        except (TypeError, ValueError) as e:
            self.get_logger().error(f"Invalid coordinates: {e}")

    def _handle_movement_command(self, robot: str, cmd: dict):
        """Handle movement commands (forward, backward, left, right)."""
        direction = cmd.get("direction", "").lower()
        valid_directions = ["forward", "backward", "left", "right"]
        
        if direction not in valid_directions:
            self.get_logger().warn(f"Invalid direction: {direction}. Use: {valid_directions}")
            return

        
        end_time = time.time() + self.movement_duration
        self.active_movements[robot] = {
            "direction": direction,
            "end_time": end_time
        }
        self.get_logger().info(f"[{robot}] Movement started → {direction} for {self.movement_duration}s")

    def publish_goals(self):
        """Publish the next goal from queue for each robot as PoseStamped."""
        for robot, queue in self.goal_queues.items():
            if not queue:
                continue

            next_goal = queue.popleft()
            x = next_goal["x"]
            y = next_goal["y"]
            yaw = next_goal["yaw"]
            name = next_goal["name"]

            goal_msg = PoseStamped()
            goal_msg.header.frame_id = "map"
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            goal_msg.pose.position.x = x
            goal_msg.pose.position.y = y
            goal_msg.pose.position.z = 0.0
            goal_msg.pose.orientation.z = math.sin(yaw / 2.0)
            goal_msg.pose.orientation.w = math.cos(yaw / 2.0)

            self.goal_publishers[robot].publish(goal_msg)
            self.get_logger().info(f"[{robot}] Published goal → {name}")

    def publish_movements(self):
        """Publish velocity commands for active movements."""
        current_time = time.time()
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
        """Publish Twist message based on direction."""
        twist = Twist()

        if direction == "forward":
            twist.linear.x = self.linear_speed
        elif direction == "backward":
            twist.linear.x = -self.linear_speed
        elif direction == "left":
            twist.angular.z = self.angular_speed
        elif direction == "right":
            twist.angular.z = -self.angular_speed

        self.cmd_vel_publishers[robot].publish(twist)

    def _publish_stop(self, robot: str):
        """Publish zero velocity to stop the robot."""
        twist = Twist()
        self.cmd_vel_publishers[robot].publish(twist)


def main():
    rclpy.init()
    node = MultiRobotLLMNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
