#!/usr/bin/env python3
"""CoMuRoS planner = the AGENT.

chat text -> parse_command() -> intent {"mode": <tool>} -> _dispatch():
    1. look the tool up in the ToolRegistry
    2. robot named?  -> check it exists, has the required skills, skill server alive, not busy
       no robot?     -> choose the best FREE robot with the skills (by distance, see _auto_assign)
    3. run the tool handler

Skill tools (find / pick / place / pick_place) work like this:
    start detection process  ->  send JSON to /<robot>/skill_cmd  ->  skill_server does
    detect -> Nav2 -> pick/place -> 0 cmd_vel  ->  /<robot>/skill_status tells us the result.

STOP (per robot):  cancel Nav2 goal -> tell skill_server to cancel (MoveGroup goal + Nav2 +
0 cmd_vel) -> 0 cmd_vel here too -> kill detection / tracking process groups.
"""

import json
import math
import os
import re
import time
import uuid
from collections import deque
from threading import Event, Lock, Thread

import rclpy
import yaml
from action_msgs.srv import CancelGoal
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry
from object_interfaces.msg import DetectedObjectArray
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

try:
    from .agent import RobotRegistry, Tool, ToolRegistry, parse_command
    from .central_llm import CentralPlanner
    from .llm_utils import LocalLLM
    from .process_manager import ProcessManager
except ImportError:  # run as plain scripts
    from agent import RobotRegistry, Tool, ToolRegistry, parse_command
    from central_llm import CentralPlanner
    from llm_utils import LocalLLM
    from process_manager import ProcessManager

PKG = "multi_robots"
DETECT_PROBE_SEC = 15.0          # how long we look for an object with several candidate robots
OBJ_FRESH_SEC = 10.0
SKILL_SILENCE_SEC = 6.0          # skill_server heartbeat missing this long -> considered dead
KILL_DETECTION_AFTER_SKILL = True
TERMINAL = ("succeeded", "failed", "cancelled")


class MultiRobotPlanner(Node):

    def __init__(self, locations: dict, registry: RobotRegistry, locations_path: str):
        super().__init__("multi_robot_planner")
        self.declare_parameter("cmd_vel_stamped", True)   # type of /<robot>/cmd_vel
        self.declare_parameter("use_central_llm", True)    # LLM fallback for fuzzy requests
        self.declare_parameter("route_skills_via_agent", False)  # pick/place/find -> robot LLM
        self.declare_parameter(
            "llm_model_path",
            "/home/vidhi123/models/qwen2.5-coder-0.5b-instruct-q4_0.gguf")
        self._stamped = bool(self.get_parameter("cmd_vel_stamped").value)

        self.registry = registry
        self._robots = registry.names()
        self._locations = locations
        self._locations_path = locations_path
        self._locations_lock = Lock()
        self._map_frame = "map"
        self._arrival_thresh = 0.3
        self._goal_timeout = 60

        self._poses = {r: {"x": 0.0, "y": 0.0, "yaw": 0.0} for r in self._robots}
        self._status = {r: "idle" for r in self._robots}
        self._goal_queues = {r: deque() for r in self._robots}
        self._active_goals = {}
        self.target_object = {r: None for r in self._robots}
        self.detected_objects = {r: [] for r in self._robots}
        self._obj_seen = {r: {} for r in self._robots}     # robot -> class -> (t, x, y)
        self._holding = {r: False for r in self._robots}
        self._skill_active = {}                            # robot -> {"id","skill","start"}
        self._skill_alive = {}                             # robot -> last heartbeat time
        self._pending = {r: deque() for r in self._robots} # queued follow-up skills
        self._stop_flags = {r: Event() for r in self._robots}
        self._agent_active = {}                            # robot -> {"id","skill","start","step"}
        self._agent_alive = {}                             # robot -> last heartbeat time
        self._use_central = bool(self.get_parameter("use_central_llm").value)
        self._route_via_agent = bool(self.get_parameter("route_skills_via_agent").value)

        self.procs = ProcessManager(self.get_logger())

        # ---- publishers / subscribers
        self._chat_out_pub = self.create_publisher(String, "/chat_output", 10)
        self._status_pub = self.create_publisher(String, "/robot_status", 10)
        self._decision_pub = self.create_publisher(String, "/planner_decision", 10)
        self._cmd_vel_pubs, self._goal_pubs = {}, {}
        self._object_target_pubs, self.object_pose_pub = {}, {}
        self._skill_cmd_pubs, self._skill_cancel_pubs = {}, {}
        self._agent_task_pubs, self._agent_cancel_pubs = {}, {}
        self._nav_cancel_clients = {}
        msg_type = TwistStamped if self._stamped else Twist
        for r in self._robots:
            self._cmd_vel_pubs[r] = self.create_publisher(msg_type, f"/{r}/cmd_vel", 10)
            self._goal_pubs[r] = self.create_publisher(PoseStamped, f"/{r}/goal_pose", 10)
            self._object_target_pubs[r] = self.create_publisher(
                String, f"/{r}/object_target", 10)
            self.object_pose_pub[r] = self.create_publisher(
                DetectedObjectArray, f"/{r}/planner_object_pose", 10)
            self._skill_cmd_pubs[r] = self.create_publisher(String, f"/{r}/skill_cmd", 10)
            self._skill_cancel_pubs[r] = self.create_publisher(
                String, f"/{r}/skill_cancel", 10)
            self._agent_task_pubs[r] = self.create_publisher(String, f"/{r}/agent_task", 10)
            self._agent_cancel_pubs[r] = self.create_publisher(
                String, f"/{r}/agent_cancel", 10)
            self._nav_cancel_clients[r] = self.create_client(
                CancelGoal, f"/{r}/navigate_to_pose/_action/cancel_goal")

            self.create_subscription(Odometry, f"/{r}/odom",
                                     lambda m, r=r: self._odom_cb(m, r), 10)
            self.create_subscription(DetectedObjectArray, f"/{r}/detected_objects",
                                     lambda m, r=r: self.detected_objects_cb(m, r), 10)
            self.create_subscription(String, f"/{r}/skill_status",
                                     lambda m, r=r: self._skill_status_cb(m, r), 10)
            self.create_subscription(String, f"/{r}/agent_report",
                                     lambda m, r=r: self._agent_report_cb(m, r), 10)
        self.create_subscription(String, "/chat_input", self._chat_cb, 10)

        self.create_timer(0.5, self._publish_queued_goals)
        self.create_timer(1.0, self._check_arrivals)
        self.create_timer(1.0, self._publish_robot_status)

        self.tools = ToolRegistry()
        self._register_tools()

        self._llm, self._central = None, None
        self._load_llm()

        self.get_logger().info(f"Robots/skills: "
                               f"{ {r: sorted(registry.skills(r)) for r in self._robots} }")
        self.get_logger().info(f"Tools: {[t[0] for t in self.tools.describe()]}")
        self._feedback("system", "MultiRobotPlanner ready.")

    # ================================================================= tools
    def _register_tools(self):
        reg = self.tools.register
        reg(Tool("navigate", "go to saved location(s)", self._h_navigate,
                 ("navigate",), True, "location"))
        reg(Tool("coordinate", "go to x y yaw", self._h_coordinate,
                 ("navigate",), True, "xy"))
        reg(Tool("movement", "move left/right/forward/backward for 2 s", self._h_movement,
                 ("navigate",), True, None))
        reg(Tool("track", "continuously track an object (old object_tracking node)",
                 self._h_track, ("track",), True, "object"))
        reg(Tool("find", "detect object, navigate to it, stop", self._h_skill,
                 ("find",), True, "object", True))
        reg(Tool("pick", "detect, navigate, pick, stop", self._h_skill,
                 ("pick",), True, "object", True))
        reg(Tool("place", "detect box, navigate, place, stop", self._h_skill,
                 ("place",), True, "holding", True))
        reg(Tool("pick_place", "pick the ball then place it in the box", self._h_skill,
                 ("pick", "place"), True, "object", True))
        reg(Tool("delegate", "give an open-ended goal to the robot's own LLM agent",
                 self._h_delegate, ("agent",), True, "any", True, True))
        reg(Tool("stop", "stop one robot completely", self._h_stop, (), False))
        reg(Tool("stop_all", "stop every robot", self._h_stop_all, (), False))
        reg(Tool("stop_detection", "kill detection process", self._h_stop_detection,
                 (), False))
        reg(Tool("stop_tracking", "kill tracking process", self._h_stop_tracking,
                 (), False))

    # ================================================================= dispatch
    def _dispatch(self, cmd: dict):
        mode = cmd.get("mode", "")
        robot = (cmd.get("robot") or "").lower()
        self.get_logger().info(f"[DISPATCH] {cmd}")
        tool = self.tools.get(mode)
        if tool is None:
            self._feedback("planner", f"unknown command '{mode}'")
            return
        if not tool.needs_robot:
            tool.handler(robot, cmd)
            return
        if robot:
            err = self._check_robot(robot, tool)
            if err:
                self._feedback("planner", err)
                return
        else:
            robot = self._auto_assign(tool, cmd)
            if robot is None:
                return                                   # feedback already given
            cmd["_auto"] = True
        tool.handler(robot, cmd)

    def _check_robot(self, robot, tool):
        if not self.registry.known(robot):
            return f"robot is not known: {robot}"
        if not self.registry.has_skills(robot, tool.required_skills):
            miss = sorted(set(tool.required_skills) - self.registry.skills(robot))
            return f"{robot.upper()} cannot '{tool.name}' (missing skill: {', '.join(miss)})"
        if tool.uses_skill_server and not self._skill_is_alive(robot):
            return f"{robot.upper()}'s skill server is not running"
        if tool.uses_agent and not self._agent_is_alive(robot):
            return f"{robot.upper()}'s robot agent is not running"
        busy = self._skill_active.get(robot) or self._agent_active.get(robot)
        if busy:
            return f"{robot.upper()} is busy with {busy['skill']}; say 'stop {robot}' first"
        if tool.uses_skill_server and self._status[robot] == "busy":
            return f"{robot.upper()} is busy; say 'stop {robot}' first"
        return None

    def _agent_is_alive(self, robot) -> bool:
        return time.time() - self._agent_alive.get(robot, 0.0) < SKILL_SILENCE_SEC

    def _skill_is_alive(self, robot) -> bool:
        return time.time() - self._skill_alive.get(robot, 0.0) < SKILL_SILENCE_SEC

    def _is_free(self, robot, tool) -> bool:
        return self._check_robot(robot, tool) is None and self._status[robot] == "idle"

    # ---- robot choice when the user names none
    def _auto_assign(self, tool, cmd):
        cands = [r for r in self.registry.with_skills(tool.required_skills)
                 if self._is_free(r, tool)]
        if not cands:
            self._feedback("planner", f"no free robot with skill(s) "
                                      f"{', '.join(tool.required_skills)} for '{tool.name}'")
            return None
        kind = tool.auto_assign
        if kind == "location":
            goals = [g for g in cmd.get("goals", []) if g in self._locations]
            if not goals:
                self._feedback("planner", "unknown location")
                return None
            x, y, _ = self._locations[goals[0]]
            return self._nearest_to(cands, x, y)
        if kind == "xy":
            return self._nearest_to(cands, cmd.get("x", 0.0), cmd.get("y", 0.0))
        if kind == "holding":
            cands = [r for r in cands if self._holding[r]]
            if not cands:
                self._feedback("planner", "no robot is holding a ball; say 'sr1 place ...' "
                                          "to force a specific robot")
                return None
            if len(cands) == 1:
                return cands[0]
            return self._assign_by_object(cands, cmd.get("target") or "box")
        if kind == "object":
            obj = cmd.get("object")
            if not obj:
                self._feedback("planner", "which object? (ball / box / bottle / chair)")
                return None
            if len(cands) == 1:
                return cands[0]
            return self._assign_by_object(cands, obj)
        if kind == "any":
            return min(cands, key=lambda r: len(self._goal_queues[r]))
        self._feedback("planner", f"please name a robot for '{tool.name}'")
        return None

    def _nearest_to(self, cands, tx, ty):
        best = min(cands, key=lambda r: self._dist_robot_xy(r, tx, ty))
        self.get_logger().info(f"[DECISION] nearest to ({tx:.2f},{ty:.2f}) = {best} "
                               f"({self._dist_robot_xy(best, tx, ty):.2f} m)")
        return best

    def _fresh_object_pos(self, obj, cands):
        best = None
        for r in cands:
            e = self._obj_seen[r].get(obj)
            if e and time.time() - e[0] < OBJ_FRESH_SEC and (best is None or e[0] > best[0]):
                best = e
        return None if best is None else (best[1], best[2])

    def _assign_by_object(self, cands, obj):
        """Object position unknown? start detection on every candidate, wait until one
        sees it, pick the nearest robot, kill the detection of the others."""
        pos = self._fresh_object_pos(obj, cands)
        started = []
        if pos is None:
            self._feedback("planner", f"looking for {obj} with {', '.join(cands)} ...")
            for r in cands:
                if not self.procs.is_running(f"det:{r}"):
                    started.append(r)
                self._start_detection(r)
            end = time.time() + DETECT_PROBE_SEC
            while pos is None and time.time() < end:
                time.sleep(0.3)
                pos = self._fresh_object_pos(obj, cands)
        if pos is None:
            self._feedback("planner", f"could not see a {obj}")
            for r in started:
                self._stop_detection(r)
            return None
        robot = self._nearest_to(cands, pos[0], pos[1])
        for r in started:
            if r != robot:
                self._stop_detection(r)
        return robot

    # ================================================================= handlers
    def _h_navigate(self, robot, cmd):
        goals = [g.lower() for g in cmd.get("goals", [])]
        unknown = [g for g in goals if g not in self._locations]
        if unknown:
            self._feedback("planner", f"unknown location: {', '.join(unknown)}")
            return
        for loc in goals:
            self._enqueue_location_goal(robot, loc)
        self._publish_decision(robot, goals[0], bool(cmd.get("_auto")))
        self._feedback("planner", f"{robot.upper()} -> {' -> '.join(goals)}"
                       + (" (auto assigned)" if cmd.get("_auto") else ""))

    def _h_coordinate(self, robot, cmd):
        x, y, yaw = float(cmd.get("x", 0)), float(cmd.get("y", 0)), float(cmd.get("yaw", 0))
        self._goal_queues[robot].append({"x": x, "y": y, "yaw": yaw, "name": f"({x},{y})"})
        self._status[robot] = "busy"
        self._feedback("planner", f"{robot.upper()} -> ({x}, {y})")

    def _h_movement(self, robot, cmd):
        direction = cmd.get("direction", "")
        speeds = {"forward": (0.3, 0.0), "backward": (-0.3, 0.0),
                  "left": (0.0, 0.5), "right": (0.0, -0.5)}
        if direction not in speeds:
            return
        lin, ang = speeds[direction]
        flag = self._stop_flags[robot]
        flag.clear()
        self._feedback("planner", f"{robot.upper()} moving {direction}")

        def run():
            end = time.time() + 2.0
            while time.time() < end and not flag.is_set():
                self._publish_twist(robot, lin, ang)
                time.sleep(0.1)
            self._publish_twist(robot)
            self._feedback("planner", "done")
        Thread(target=run, daemon=True).start()

    def _h_track(self, robot, cmd):
        target = cmd.get("object")
        if not target:
            self._feedback("planner", "No target object specified")
            return
        self.target_object[robot] = target
        self._start_detection(robot)
        time.sleep(1.5)
        self._object_target_pubs[robot].publish(String(data=target))
        self._start_tracking(robot)
        self._feedback("planner", f"{robot.upper()} tracking {target}")

    @staticmethod
    def _goal_text(cmd):
        mode, obj = cmd["mode"], cmd.get("object") or "ball"
        target = cmd.get("target") or "box"
        return {"pick": f"pick the {obj}",
                "place": f"place the {obj} into the {target}",
                "pick_place": f"pick the {obj} and place it into the {target}",
                "find": f"find the {obj}"}[mode]

    def _h_delegate(self, robot, cmd):
        self._delegate(robot, cmd.get("goal") or "")

    def _delegate(self, robot, goal):
        """Level 1 -> level 2: hand a goal to the robot's own LLM agent."""
        self._start_detection(robot)
        tid = uuid.uuid4().hex[:8]
        self._agent_active[robot] = {"id": tid, "skill": "agent", "start": time.time(),
                                     "step": 0}
        self._status[robot] = "busy"
        self._agent_task_pubs[robot].publish(String(data=json.dumps(
            {"task_id": tid, "goal": goal})))
        self._feedback("planner", f"{robot.upper()} agent goal: '{goal}'")

    def _agent_report_cb(self, msg: String, robot):
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._agent_alive[robot] = time.time()
        act = self._agent_active.get(robot)
        if act is None or d.get("task_id") != act["id"]:
            return
        st = d.get("state")
        if st == "running" and d.get("action"):
            if d.get("step") != act["step"]:
                act["step"] = d.get("step")
                self._feedback(robot, f"step {d['step']}: {d['action']} "
                                      f"[{d.get('source')}] {d.get('reason', '')}")
        elif st in TERMINAL:
            self._finish_agent(robot, st, d.get("msg", ""))

    def _finish_agent(self, robot, state, text=""):
        if self._agent_active.pop(robot, None) is None:
            return
        self._feedback(robot, f"task {state}" + (f" ({text})" if text else ""))
        self._pending[robot].clear()
        self._status[robot] = "idle"
        Thread(target=self._after_skill, args=(robot,), daemon=True).start()

    def _h_skill(self, robot, cmd):
        if self._route_via_agent and self._agent_is_alive(robot):
            self._delegate(robot, self._goal_text(cmd))
            return
        mode, obj = cmd["mode"], cmd.get("object") or "ball"
        target = cmd.get("target") or "box"
        if mode == "pick_place":
            queue = [{"skill": "pick", "object": obj},
                     {"skill": "place", "object": obj, "target": target}]
        elif mode == "place":
            if not self._holding[robot]:
                self._feedback("planner", f"note: {robot.upper()} did not report a held ball")
            queue = [{"skill": "place", "object": obj, "target": target}]
        else:
            queue = [{"skill": mode, "object": obj}]
        self._pending[robot] = deque(queue[1:])
        self._start_skill(robot, queue[0])

    def _start_skill(self, robot, spec):
        self._start_detection(robot)           # idempotent
        spec = dict(spec, id=uuid.uuid4().hex[:8])
        self._skill_active[robot] = {"id": spec["id"], "skill": spec["skill"],
                                     "start": time.time()}
        self._status[robot] = "busy"
        self._skill_cmd_pubs[robot].publish(String(data=json.dumps(spec)))
        what = spec.get("object", "")
        self._feedback("planner", f"{robot.upper()} {spec['skill']} {what}: detection started")

    def _h_stop(self, robot, cmd):
        if not robot:
            self._h_stop_all("", cmd)
            return
        if not self.registry.known(robot):
            self._feedback("planner", f"robot is not known: {robot}")
            return
        self._stop_robot(robot)
        self._feedback("planner", f"{robot.upper()} stopped")

    def _h_stop_all(self, _robot, _cmd):
        for r in self._robots:
            self._stop_robot(r)
        self._feedback("planner", "All robots stopped")

    def _h_stop_detection(self, robot, _cmd):
        if robot and self.registry.known(robot):
            self._stop_detection(robot)
            self.target_object[robot] = None
            self._feedback("planner", f"{robot.upper()} detection stopped")

    def _h_stop_tracking(self, robot, _cmd):
        if robot and self.registry.known(robot):
            self.procs.stop(f"track:{robot}")
            self.target_object[robot] = None
            self._feedback("planner", f"{robot.upper()} tracking stopped")

    
    def _publish_twist(self, robot, lin=0.0, ang=0.0):
        if self._stamped:
            m = TwistStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.twist.linear.x, m.twist.angular.z = float(lin), float(ang)
        else:
            m = TwistStamped()
            m.twist.linear.x, m.twist.angular.z = float(lin), float(ang)
        self._cmd_vel_pubs[robot].publish(m)

    def _zero_cmd_vel(self, robot, n=5):
        for _ in range(n):
            self._publish_twist(robot)
            time.sleep(0.05)

    def _cancel_nav2(self, robot):
        """Cancel ALL goals on /<robot>/navigate_to_pose (empty request = cancel all)."""
        cli = self._nav_cancel_clients[robot]
        if cli.service_is_ready():
            cli.call_async(CancelGoal.Request())
        else:
            self.get_logger().warn(f"[{robot}] nav2 cancel service not ready "
                                   f"(/{robot}/navigate_to_pose/_action/cancel_goal)")

    def _stop_robot(self, robot):
        self.get_logger().warn(f"[{robot}] STOP")
        self._stop_flags[robot].set()                      # movement thread
        self._goal_queues[robot].clear()
        self._active_goals.pop(robot, None)
        self._pending[robot].clear()
        self._skill_active.pop(robot, None)
        self._agent_active.pop(robot, None)
        self._agent_cancel_pubs[robot].publish(String(data="cancel"))  # robot LLM stops first
        self._cancel_nav2(robot)                           # 1. Nav2 goal
        self._skill_cancel_pubs[robot].publish(String(data="cancel"))  # 2. MoveGroup etc.
        self._zero_cmd_vel(robot)                          # 3. zero velocity
        self.target_object[robot] = None
        self.procs.stop(f"det:{robot}")                    # 4. kill vision (process group)
        self.procs.stop(f"track:{robot}")
        self._zero_cmd_vel(robot, 3)
        self._status[robot] = "idle"

    # ================================================================= skill status
    def _skill_status_cb(self, msg: String, robot):
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._skill_alive[robot] = time.time()
        if "holding" in d:
            self._holding[robot] = bool(d["holding"])
        act = self._skill_active.get(robot)
        if act is None or d.get("id") != act["id"]:
            return
        if d.get("state") in TERMINAL:
            self._finish_skill(robot, d["state"], d.get("msg", ""))

    def _finish_skill(self, robot, state, text=""):
        act = self._skill_active.pop(robot, None)
        if act is None:
            return
        self._feedback(robot, f"{act['skill']} {state}" + (f" ({text})" if text else ""))
        if state == "succeeded" and self._pending[robot]:
            self._start_skill(robot, self._pending[robot].popleft())
            return
        self._pending[robot].clear()
        self._status[robot] = "idle"
        Thread(target=self._after_skill, args=(robot,), daemon=True).start()

    def _after_skill(self, robot):
        self._zero_cmd_vel(robot, 3)
        if KILL_DETECTION_AFTER_SKILL:
            self._stop_detection(robot)

    # ================================================================= detection / tracking procs
    def _proc_cmd(self, robot, exe):
        return ["ros2", "run", PKG, exe, "--ros-args",
                "-r", f"/tf:=/{robot}/tf", "-r", f"/tf_static:=/{robot}/tf_static",
                "-r", f"__ns:=/{robot}"]

    def _start_detection(self, robot):
        self.procs.start(f"det:{robot}", self._proc_cmd(robot, "multi_object_detection"))

    def _stop_detection(self, robot):
        self.procs.stop(f"det:{robot}")

    def _start_tracking(self, robot):
        self.procs.start(f"track:{robot}", self._proc_cmd(robot, "object_tracking"))

    # ================================================================= callbacks / timers
    def detected_objects_cb(self, msg, robot):
        self.detected_objects[robot] = msg.objects
        now = time.time()
        for o in msg.objects:
            self._obj_seen[robot][o.class_name.lower()] = (
                now, o.pose.position.x, o.pose.position.y)
        target = self.target_object.get(robot)
        if not target:
            return
        matches = DetectedObjectArray()
        matches.header.stamp = self.get_clock().now().to_msg()
        matches.header.frame_id = msg.header.frame_id or self._map_frame
        matches.objects = [o for o in msg.objects if o.class_name.lower() == target]
        if matches.objects:
            self.object_pose_pub[robot].publish(matches)

    def _odom_cb(self, msg: Odometry, robot):
        p = self._poses[robot]
        p["x"], p["y"] = msg.pose.pose.position.x, msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        p["yaw"] = math.atan2(2 * (q.w * q.z + q.x * q.y),
                              1 - 2 * (q.y * q.y + q.z * q.z))

    def _dist_robot_xy(self, robot, tx, ty):
        p = self._poses[robot]
        return math.hypot(p["x"] - tx, p["y"] - ty)

    def _check_arrivals(self):
        for robot, goal in list(self._active_goals.items()):
            dist = self._dist_robot_xy(robot, goal["x"], goal["y"])
            if dist < self._arrival_thresh:
                self.get_logger().info(f"[{robot}] Arrived -> idle")
            elif time.time() - goal["sent_time"] > self._goal_timeout:
                self.get_logger().warn(f"[{robot}] Goal timeout -> idle")
            else:
                continue
            self._status[robot] = "idle"
            self._active_goals.pop(robot, None)
        # watchdog: skill server died / disconnected
        for robot, act in list(self._skill_active.items()):
            if (time.time() - act["start"] > SKILL_SILENCE_SEC
                    and not self._skill_is_alive(robot)):
                self._finish_skill(robot, "failed", "lost contact with skill server")
        for robot, act in list(self._agent_active.items()):
            if (time.time() - act["start"] > SKILL_SILENCE_SEC
                    and not self._agent_is_alive(robot)):
                self._finish_agent(robot, "failed", "lost contact with robot agent")

    def _enqueue_location_goal(self, robot, location):
        x, y, yaw = self._locations[location]
        self._goal_queues[robot].append({"x": x, "y": y, "yaw": yaw, "name": location})
        self._status[robot] = "busy"

    def _publish_queued_goals(self):
        for robot, q in self._goal_queues.items():
            if not q or robot in self._active_goals:
                continue
            goal = q.popleft()
            msg = PoseStamped()
            msg.header.frame_id = self._map_frame
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.position.x, msg.pose.position.y = goal["x"], goal["y"]
            msg.pose.orientation.z = math.sin(goal["yaw"] / 2.0)
            msg.pose.orientation.w = math.cos(goal["yaw"] / 2.0)
            self._goal_pubs[robot].publish(msg)
            self.get_logger().info(f"[{robot}] Goal -> {goal['name']}")
            self._active_goals[robot] = {"x": goal["x"], "y": goal["y"],
                                         "name": goal["name"], "sent_time": time.time()}

    def _publish_decision(self, robot, location, auto):
        x, y = self._locations.get(location, (0.0, 0.0, 0.0))[:2]
        self._decision_pub.publish(String(data=json.dumps(
            {"robot": robot, "location": location, "x": x, "y": y, "auto": auto})))

    def _publish_robot_status(self):
        data = {}
        for r in self._robots:
            act = self._skill_active.get(r) or self._agent_active.get(r)
            goal = self._active_goals.get(r, {}).get("name", "—")
            data[r] = {"status": self._status[r],
                       "x": round(self._poses[r]["x"], 2), "y": round(self._poses[r]["y"], 2),
                       "goal": act["skill"] if act else goal,
                       "tracking": self.target_object.get(r),
                       "holding": self._holding[r]}
        self._status_pub.publish(String(data=json.dumps(data)))

    def _feedback(self, sender, text):
        self._chat_out_pub.publish(String(data=f"{sender}|{text}"))
        self.get_logger().info(f"[FEEDBACK] {sender}: {text}")

    # ================================================================= chat
    def _load_llm(self):
        path = self.get_parameter("llm_model_path").value
        if not path or not os.path.exists(path):
            self.get_logger().warn("LLM model not found -> chat replies disabled "
                                   "(commands still work)")
            return
        try:
            self.get_logger().info("Loading central LLM ...")
            self._llm = LocalLLM(path, n_ctx=2048, n_threads=8, logger=self.get_logger())
            self._central = CentralPlanner(self._llm)
            self.get_logger().info("Central LLM loaded")
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"LLM load failed: {e}")

    def _chat_reply(self, user_text) -> str:
        if self._llm is None:
            return "Chat is disabled, but I can run robot commands."
        return self._llm.chat(user_text)

    def _fleet_state(self):
        return [{"name": r, "skills": sorted(self.registry.skills(r)),
                 "status": self._status[r], "holding": self._holding[r]}
                for r in self._robots]

    def _chat_cb(self, msg: String):
        Thread(target=self._process_chat, args=(msg.data,), daemon=True).start()

    def is_robot_command(self, text):
        t = text.lower()
        words = ("go", "goto", "move", "stop", "navigate", "room", "object", "left", "right",
                 "forward", "backward", "find", "search", "track", "pick", "place", "put",
                 "grab", "drop", "halt", "abort")
        return any(r in t for r in self._robots) or any(w in t.split() for w in words)

    def _process_chat(self, text):
        try:
            user_text = text.replace("human|", "").strip()
            self.get_logger().info(f"[CHAT] '{user_text}'")
            if user_text.lower().startswith("save location"):
                self._handle_save_location(user_text)
                return
            if not self.is_robot_command(user_text):
                self._feedback("planner", self._chat_reply(user_text))
                return
            cmd = parse_command(user_text, self._robots, list(self._locations))
            named = [r for r in self._robots
                     if re.search(rf"\b{re.escape(r)}\b", user_text.lower())]
            if self._use_central and self._central is not None and (
                    cmd is None or len(named) >= 2):
                intents = self._central.plan(user_text, self._fleet_state(),
                                             list(self._locations))
                if intents:
                    self._feedback("planner", "central plan: " + "; ".join(
                        f"{i.get('robot') or 'auto'}:{i['mode']}" for i in intents))
                    for it in intents:
                        self._dispatch(it)
                    return
                if cmd is None:
                    self._feedback("planner", "Sorry, I could not turn that into a safe plan.")
                    return
            if cmd is None:
                self._feedback("planner", "Sorry, I didn't understand that command.")
                return
            self._dispatch(cmd)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"chat processing failed: {e!r}")
            self._feedback("planner", "internal error, see planner log")

    def _handle_save_location(self, user_text):
        try:
            payload = user_text[len("save location"):].strip()
            if "|" not in payload:
                self._feedback("planner", "usage: save location <name>|<robot>")
                return
            name_part, robot_part = payload.rsplit("|", 1)
            name, robot = name_part.strip().lower(), robot_part.strip().lower()
            if not name:
                self._feedback("planner", "location name cannot be empty")
                return
            if not self.registry.known(robot):
                self._feedback("planner", f"unknown robot: {robot}")
                return
            p = self._poses[robot]
            with self._locations_lock:
                self._locations[name] = (p["x"], p["y"], p["yaw"])
                data = {"locations": {n: {"x": x, "y": y, "yaw": yaw}
                                      for n, (x, y, yaw) in self._locations.items()}}
                with open(self._locations_path, "w") as f:
                    yaml.safe_dump(data, f, default_flow_style=False, sort_keys=True)
            self._feedback("planner", f"saved location '{name}' at "
                                      f"({p['x']:.2f},{p['y']:.2f}) from {robot.upper()}")
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"save location failed: {e}")
            self._feedback("planner", "failed to save location")


def main(args=None):
    rclpy.init(args=args)
    pkg_share = get_package_share_directory(PKG)
    registry = RobotRegistry(os.path.join(pkg_share, "config", "robots.yaml"))
    print(f"Active robots: {registry.names()}")

    locs_path = os.path.join(pkg_share, "config", "locations.yaml")
    with open(locs_path) as f:
        locs_data = yaml.safe_load(f)
    locations = {n.lower(): (float(p["x"]), float(p["y"]), float(p["yaw"]))
                 for n, p in locs_data["locations"].items()}
    print(f"Locations: {list(locations)}")

    planner = MultiRobotPlanner(locations, registry, locs_path)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(planner)
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Shutting down ...")
    finally:
        planner.procs.stop_all()
        planner.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()