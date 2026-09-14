#!/usr/bin/env python3
 
 
import os, yaml, json, math, time, signal
from collections import deque
from threading import Lock, Thread
 
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
 
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseArray, Pose, PoseStamped
from nav_msgs.msg import Odometry
 
import numpy as np
import subprocess
import re
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from transforms3d.euler import euler2quat
from ament_index_python.packages import get_package_share_directory
from llama_cpp import Llama
import time
 
# def load_config(pkg_share: str) -> dict:
#     path = os.path.join(pkg_share, "config", "planner.yaml")
#     with open(path) as f:
#         return yaml.safe_load(f)
 
 
class LLMCommandResolver:
    """
    Pure LLM-based command resolver.
    Builds a detailed prompt with all available robots, locations, modes,
    and JSON schemas, then asks the LLM to output a single JSON object.
    """
    _CHAT_PROMPT = """
You are CoMuRoS, a friendly robot assistant.
 
Rules:
- Talk naturally.
- Answer greetings.
- Answer general questions.
- Keep responses short.
- Do not return JSON.
- Do not explain that you are an AI.
 
Examples:
 
User: hello
Assistant: Hello! How can I help you today?
 
User: how are you
Assistant: I'm doing well. How can I assist you?
 
User: what is a fruit?
Assistant: A fruit is the edible part of a plant that usually contains seeds.
 
User: tell me a joke
Assistant: Why did the robot go to school? To improve its byte-sized knowledge!
"""
    _SYSTEM_PROMPT =  """
You are a robot command parser.

Available robots:
{robots}

Available locations:
{locations}

Convert the user command into this format.

Examples:

User: send sr1 to room1
ACTION=WAYPOINT
ROBOT=sr1
LOCATION=room1

User: sr1 go to room1
ACTION=WAYPOINT
ROBOT=sr1
LOCATION=room1

User: move sr1 left
ACTION=MOVE
ROBOT=sr1
DIRECTION=left

User: stop sr1
ACTION=STOP
ROBOT=sr1

User: sr1 find object
ACTION=OBJECT
ROBOT=sr1

Output only the fields.
"""
 
    def __init__(self, robots: list, locations: list, logger):
        self._robots    = [r.lower() for r in robots]
        self._locations = [l.lower() for l in locations]
        self._logger    = logger
    def _parse_command(self, text):

        text = text.lower()

        cmd = {}

      
        for r in self._robots:
            if r.lower() in text:
                cmd["robot"] = r
                break

        if "left" in text:
            return {
                "mode": "movement",
                "robot": cmd.get("robot"),
                "direction": "left"
            }

        if "right" in text:
            return {
                "mode": "movement",
                "robot": cmd.get("robot"),
                "direction": "right"
            }

        if "forward" in text:
            return {
                "mode": "movement",
                "robot": cmd.get("robot"),
                "direction": "forward"
            }

        if "backward" in text:
            return {
                "mode": "movement",
                "robot": cmd.get("robot"),
                "direction": "backward"
            }

        
        if "stop" in text:
            return {
                "mode": "stop",
                "robot": cmd.get("robot")
            }

       
        object_words = [
            "object",
            "find",
            "search",
            "detect",
            "track"
        ]

        if any(w in text for w in object_words):

            if cmd.get("robot"):
                return {
                    "mode": "named_robot_object",
                    "robot": cmd["robot"]
                }

            return {
                "mode": "auto_object"
            }
        m = re.search(
    r'(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)',
    text
)

        if m:
            robot = cmd.get("robot")

            if robot is None:
                idle = [r for r in self._robots]
                if idle:
                    robot = idle[0]

            return {
                "mode": "coordinate",
                "robot": robot,
                "x": float(m.group(1)),
                "y": float(m.group(2)),
                "yaw": float(m.group(3))
            }      
        
        for loc in self._locations:

            if loc.lower() in text:

                if cmd.get("robot"):
                    return {
                        "mode": "waypoint_chain",
                        "robot": cmd["robot"],
                        "goals": [loc]
                    }

                return {
                    "mode": "auto_location",
                    "goals": [loc]
                }

        return None
        
    def chat(self, user_text: str, llm) -> str:
 
        prompt = f"""
    You are CoMuRoS, a friendly robot assistant.
 
    User: {user_text}
 
    Assistant:
    """
 
        result = llm(
            prompt,
            max_tokens=100,
            temperature=0.7,
        )
 
        return result["choices"][0]["text"].strip()
   
    
    def resolve(self, user_text, llm, status):

        try:

            cmd = self._parse_command(user_text)

            if cmd:
                return {
                    "response": "",
                    "commands": [cmd]
                }

            return None

           

        except Exception as e:

            self._logger.error(str(e))

            return None
    
    
 
 
 
class MultiRobotPlanner(Node):
    """
    Central planner. All commands go directly to the LLM for resolution.
 
    Planning modes:
      1. auto_location       → nearest idle robot to named location
      2. waypoint_chain      → explicit robot, sequential locations
      3. auto_object         → nearest robot to any detected object
      4. named_robot_object  → explicit robot to its detected object
      5. stop                → cancel navigation for a robot
      6. movement            → velocity command in a direction
      7. coordinate          → absolute map coordinates
    """
    _ARRIVAL_THRESH = 0.3    
    _GOAL_TIMEOUT   = 60     
    def __init__(self,  locations: dict, robot_names: list):
        super().__init__("multi_robot_planner")
        
        self._locations = locations
        self._robots    = robot_names
        self._map_frame = "map"
        self._object_detection_processes = {}
        self._object_tracking_processes  = {}
 
        
        self._arrival_thresh = 0.3
        self._goal_timeout   = 60
 
        self._poses       = {r: {"x": 0.0, "y": 0.0} for r in self._robots}
        self._status      = {r: "idle"               for r in self._robots}
        self._goal_queues = {r: deque() for r in self._robots}
        
        
        self._active_goals: dict = {}
       
        
 
    
        self._chat_out_pub  = self.create_publisher(String, "/chat_output",  10)
        self._status_pub    = self.create_publisher(String, "/robot_status", 10)
        self._decision_pub = self.create_publisher(
            String,
            "/planner_decision",
            10
        )
        self._cmd_vel_pubs  = {}
        self._goal_pubs     = {}
        for robot in self._robots:
            
            self._cmd_vel_pubs[robot] = self.create_publisher(
                Twist,       f"/{robot}/cmd_vel",   10)
            self._goal_pubs[robot]    = self.create_publisher(
                PoseStamped, f"/{robot}/goal_pose", 10)
           
       
        
 
        
        for robot in self._robots:
            self.create_subscription(
                Odometry, f"/{robot}/odom",
                lambda msg, r=robot: self._odom_cb(msg, r), 10)
            
 
        self.create_subscription(String, "/chat_input", self._chat_cb, 10)
 
    
        self.create_timer(0.5, self._publish_queued_goals)
        self.create_timer(1.0, self._check_arrivals)
        self.create_timer(1.0, self._publish_robot_status)
        self._goal_queues = {r: deque() for r in self._robots}
        self.get_logger().info("Loading LLM ...")
        self._llm = Llama(
            model_path   = "/home/einfochips/models/qwen2.5-coder-0.5b-instruct-q4_0.gguf",
            n_ctx        = 1024 ,
            n_threads    = 8 ,
            n_gpu_layers = 0 ,
            verbose      = False,
        )
        self.get_logger().info("LLM loaded")
 
        self._llm_resolver = LLMCommandResolver(
            self._robots,
            list(self._locations.keys()),
            self.get_logger(),
        )
 
        self.get_logger().info("MultiRobotPlanner READY")
        self._feedback("system", "MultiRobotPlanner ready.")
        self.get_logger().info("MultiRobotPlanner READY")
 
   
 
    def _feedback(self, sender: str, text: str):
        """Publish a message that the GUI will display as a chat bubble."""
        msg      = String()
        msg.data = f"{sender}|{text}"
        self._chat_out_pub.publish(msg)
        self.get_logger().info(f"[FEEDBACK] {sender}: {text}")
    def _publish_robot_status(self):
        """Publish JSON robot status for the GUI status panel."""
        data = {}
        for r in self._robots:
            goal_name = "—"
            if r in self._active_goals:
                goal_name = self._active_goals[r].get("name", "—")
            data[r] = {
                "status": self._status[r],
                "x":      round(self._poses[r]["x"], 2),
                "y":      round(self._poses[r]["y"], 2),
                "goal":   goal_name,
            }
        msg      = String()
        msg.data = json.dumps(data)
        self._status_pub.publish(msg)
    def _odom_cb(self, msg: Odometry, robot: str):
        self._poses[robot]["x"] = msg.pose.pose.position.x
        self._poses[robot]["y"] = msg.pose.pose.position.y
 
    def _dist_robot_xy(self, robot: str, tx: float, ty: float) -> float:
        rx = self._poses[robot]["x"]
        ry = self._poses[robot]["y"]
        return math.sqrt((rx - tx)**2 + (ry - ty)**2)
 
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
 
    def _start_object_detection(self, robot: str):
        if robot in self._object_detection_processes:
            self.get_logger().info(f"[{robot}] object_detection already running")
            return
        self.get_logger().info(f"[{robot}] Launching object_detection")
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
            self.get_logger().info(f"[{robot}] object_tracking already running")
            return
        self.get_logger().info(f"[{robot}] Launching object_tracking")
        proc = subprocess.Popen([
            "ros2", "run", "multi_robots", "object_tracking",
            "--ros-args",
            "-r", "/tf:=/" + robot + "/tf",
            "-r", "/tf_static:=/" + robot + "/tf_static",
            "-r", "__ns:=/" + robot,
        ])
        self._object_tracking_processes[robot] = proc
 
    def _stop_object_tracking(self, robot):

        proc = self._object_tracking_processes.get(robot)

        if proc is None:
            return

        self.get_logger().info(
            f"[{robot}] Stopping object_tracking"
        )

        proc.terminate()

        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        del self._object_tracking_processes[robot]
    
    def _stop_object_detection(self, robot):

        proc = self._object_detection_processes.get(robot)

        if proc is None:
            return

        self.get_logger().info(
            f"[{robot}] Stopping object_detection"
        )

        proc.terminate()

        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        del self._object_detection_processes[robot]
    def _mode_auto_location(self, location_name: str):
        if location_name not in self._locations:
           self._feedback("planner","unknown location")
           return
        lx, ly, _ = self._locations[location_name]
        robot = self._nearest_robot(lx, ly)
        if robot is None:
           self._feedback("planner","no robot available")
           return
        self._enqueue_location_goal(robot, location_name)
        self._feedback("planner","auto assigned")
        self._publish_decision(robot, location_name, auto=True)
 
    def _mode_waypoint_chain(self, robot: str, goals: list):
        if robot.strip().lower() not in [r.lower() for r in self._robots]:
            self._feedback(
                "planner",
                f"robot is not known: {robot}"
            )
            return
        unknown = [g for g in goals if g not in self._locations]
        if unknown:
            self._feedback("planner","unknown location")
            return
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
 
        user_text = text.replace("human|", "").strip()
 
        self.get_logger().info(f"[CHAT] '{user_text}'")
 
        
        if not self.is_robot(user_text):
 
            reply = self.chat(user_text,self._llm)
 
            self._feedback("planner", reply)
 
            return
 
        parsed = self._llm_resolver.resolve(
        user_text,
        self._llm,
        self._status
    )
 
        if not parsed:
            self._feedback(
                "planner",
                "Sorry, I didn't understand that command."
            )
            return
 
        response = parsed.get("response", "")
 
        if response:
            self._feedback("planner", response)
 
        for cmd in parsed.get("commands", []):
            cmd["original_text"] = user_text
            self._dispatch(cmd)
    def is_robot(self, text: str) -> bool:
        text = text.lower()
 
        robot_words = self._robots
 
        command_words = [
            "go",
            "goto",
            "move",
            "stop",
            "navigate",
            "room",
            "object",
            "left",
            "right",
            "forward",
            "backward",
        ]
 
        if any(r in text for r in robot_words):
            return True
 
        if any(w in text for w in command_words):
            return True
 
        return False
    def chat(self, user_text: str, llm) -> str:
 
        prompt = f"""
    You are CoMuRoS, a friendly robot assistant.
 
    User: {user_text}
 
    Assistant:
    """
 
        result = llm(
            prompt,
            max_tokens=100,
            temperature=0.7,
        )
 
        return result["choices"][0]["text"].strip()
    def _dispatch(self, cmd: dict):
        mode  = cmd.get("mode", "")
        robot = cmd.get("robot")
        if robot:
            robot = robot.lower()
        else:
            robot = ""
        goals = [g.lower() for g in cmd.get("goals", [])]
 
        self.get_logger().info(
            f"[DISPATCH] mode={mode} robot={robot} goals={goals}")
        if mode == "auto_location":
 
            
 
            original_text = cmd.get("original_text", "").lower()
 
            explicit_robot = None
 
            for r in self._robots:
                if r.lower() in original_text:
                    explicit_robot = r
                    break
 
            if explicit_robot:
                self.get_logger().warn(
                    f"[SAFETY] Explicit robot '{explicit_robot}' detected "
                    f"-> forcing waypoint_chain"
                )
 
                self._mode_waypoint_chain(explicit_robot, goals)
 
            else:
                for loc in goals:
                    self._mode_auto_location(loc)
 
        elif mode == "waypoint_chain":
            if not robot:
                self.get_logger().warn("waypoint_chain requires robot name"); return
            self._mode_waypoint_chain(robot, goals)
 
        elif mode == "auto_object":
            nearest_robot = self._nearest_robot(0.0, 0.0)
            if nearest_robot is None:
                self.get_logger().warn("No robot available"); return
            self._start_object_detection(nearest_robot)
            time.sleep(2.0)
            self._start_object_tracking(nearest_robot)
            self.get_logger().info(f"[{nearest_robot}] Started autonomous object search")
            self._feedback("planner","searching for object")
        elif mode == "named_robot_object":
            if robot.strip().lower() not in [r.lower() for r in self._robots]:
                self.get_logger().warn(f"Unknown robot: {robot}")
                self._feedback("planner","unknown robot")
                return
            self._start_object_detection(robot)
            time.sleep(2.0)
            self._start_object_tracking(robot)
            self.get_logger().info(f"[{robot}] Started object search")
 
        elif mode == "stop":

            if robot:

                self._stop_object_detection(robot)
                self._stop_object_tracking(robot)

                self._cmd_vel_pubs[robot].publish(Twist())

                self._goal_queues[robot].clear()
                self._active_goals.pop(robot, None)

                self._status[robot] = "idle"

                self._feedback(
                    "planner",
                    f"{robot.upper()} stopped"
                )
           
 
        elif mode == "coordinate":
            if robot in self._robots:
                try:
                    x   = float(cmd.get("x",   0.0))
                    y   = float(cmd.get("y",   0.0))
                    yaw = float(cmd.get("yaw", 0.0))
                    self._goal_queues[robot].append(
                        {"x": x, "y": y, "yaw": yaw, "name": f"({x},{y})"})
                    self._status[robot] = "busy"
                    self._feedback("planner",f"{robot.upper()},({x}{y})")
                except (TypeError, ValueError) as e:
                    self.get_logger().error(f"Bad coordinates: {e}")
                    self._feedback("planner",f"bad coordinates")
 
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
 
                self._feedback("planner",f"{robot.upper()}moving")
                def _pub_until():
                    while time.time() < end:
                        self._cmd_vel_pubs[robot].publish(twist)
                        time.sleep(0.1)
                    self._cmd_vel_pubs[robot].publish(Twist())
                    self._feedback("planner","done")
                Thread(target=_pub_until, daemon=True).start()
        elif mode == "stop_detection":

            self._stop_object_detection(robot)
            self._feedback("planner", f"{robot.upper()} tracking stopped")   
        elif mode == "stop_tracking":

            self._stop_object_tracking(robot)
            self._feedback("planner", f"{robot.upper()} tracking stopped")
        elif mode == "stop_all":

            for r in self._robots:

                self._stop_object_detection(r)
                self._stop_object_tracking(r)

                self._cmd_vel_pubs[r].publish(Twist())
                self._goal_queues[r].clear()
                self._active_goals.pop(r, None)
                self._status[r] = "idle"

            self._feedback("planner", "All robots stopped")
        else:
            self.get_logger().warn(f"Unknown mode: {mode}")
            self._feedback("planner",f"unknown mode")
 
 
 
def main(args=None):
    rclpy.init(args=args)
 
    pkg_share = get_package_share_directory("multi_robots")
    
 
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
 
    planner  = MultiRobotPlanner( locations, robot_names)
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
 