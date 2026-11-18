#!/usr/bin/env python3
"""ROBOT AGENT (level 2): one per robot, runs ON that robot with its OWN local LLM.

Topics (relative -> /<robot>/... when started with __ns:=/<robot>):
    agent_task   (String JSON) {"task_id","goal"}          <- central planner
    agent_cancel (String)                                  <- central planner (stop)
    agent_report (String JSON) {"task_id","state","msg","step","action","source","reason"}
                                state: idle | running | succeeded | failed | cancelled
    skill_cmd / skill_status / skill_cancel                <-> skill_server.py (same robot)
    detected_objects                                       <- detection node (for state)

Loop:  observe state -> AgentBrain.decide (LLM, validated, rule fallback) -> send ONE skill
       to skill_server -> wait for its result -> repeat until done / give_up / cancel.
"""

import json
import threading
import time
import uuid

import rclpy
from object_interfaces.msg import DetectedObjectArray
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String


from agent.llm_utils import LocalLLM
from agent.robot_brain import AgentBrain


TERMINAL = ("succeeded", "failed", "cancelled")
VISIBLE_SEC = 3.0


class RobotAgent(Node):

    def __init__(self, llm=None):
        super().__init__("robot_agent")
        for n, d in (("llm_model_path", ""), ("skills", ["find", "pick", "place"]),
                     ("max_steps", 6), ("max_retries", 1), ("skill_timeout_sec", 300.0)):
            self.declare_parameter(n, d)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731

        path = gp("llm_model_path")
        if llm is None and path:
            try:
                llm = LocalLLM(path, n_ctx=1536, n_threads=4, logger=self.get_logger())
                self.get_logger().info("Robot LLM loaded")
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"Robot LLM load failed ({e}); using rules only")
        self.brain = AgentBrain(llm, gp("skills"), int(gp("max_steps")),
                                int(gp("max_retries")))
        self._skill_timeout = float(gp("skill_timeout_sec"))

        self._cancel = threading.Event()
        self._busy = threading.Lock()
        self._cv = threading.Condition()
        self._results = {}
        self._holding = False
        self._seen = {}
        self._task_id = None

        cbg = ReentrantCallbackGroup()
        self.create_subscription(String, "agent_task", self._task_cb, 10, callback_group=cbg)
        self.create_subscription(String, "agent_cancel", self._cancel_cb, 10,
                                 callback_group=cbg)
        self.create_subscription(String, "skill_status", self._status_cb, 10,
                                 callback_group=cbg)
        self.create_subscription(DetectedObjectArray, "detected_objects", self._objects_cb,
                                 10, callback_group=cbg)
        self.report_pub = self.create_publisher(String, "agent_report", 10)
        self.skill_pub = self.create_publisher(String, "skill_cmd", 10)
        self.skill_cancel_pub = self.create_publisher(String, "skill_cancel", 10)
        self.create_timer(1.0, lambda: self._report(
            "running" if self._task_id else "idle", self._task_id or ""), callback_group=cbg)
        self.get_logger().info(f"RobotAgent ready (llm={'yes' if llm else 'rules only'})")

    
    def _report(self, state, task_id, msg="", **extra):
        d = {"task_id": task_id, "state": state, "msg": msg}
        d.update(extra)
        self.report_pub.publish(String(data=json.dumps(d)))

    def _objects_cb(self, msg):
        now = time.monotonic()
        for o in msg.objects:
            self._seen[o.class_name.lower()] = now

    def _visible(self):
        now = time.monotonic()
        return sorted(k for k, t in self._seen.items() if now - t < VISIBLE_SEC)

    def _status_cb(self, msg):
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if "holding" in d:
            self._holding = bool(d["holding"])
        if d.get("state") in TERMINAL:
            with self._cv:
                self._results[d.get("id")] = d
                self._cv.notify_all()

    def _cancel_cb(self, _msg):
        self.get_logger().warn("agent CANCEL")
        self._cancel.set()
        with self._cv:
            self._cv.notify_all()

    def _task_cb(self, msg):
        try:
            spec = json.loads(msg.data)
            tid, goal = spec["task_id"], spec["goal"]
        except (json.JSONDecodeError, KeyError):
            self._report("failed", "", "bad task json")
            return
        if not self._busy.acquire(blocking=False):
            self._report("failed", tid, "agent busy")
            return
        self._cancel.clear()
        self._task_id = tid
        threading.Thread(target=self._run_task, args=(tid, goal), daemon=True).start()


    def _wait_result(self, sid):
        deadline = time.monotonic() + self._skill_timeout
        with self._cv:
            while (sid not in self._results and not self._cancel.is_set()
                   and time.monotonic() < deadline and rclpy.ok()):
                self._cv.wait(0.5)
            return self._results.pop(sid, None)

    def _run_task(self, tid, goal):
        history, final = [], ("failed", "stopped unexpectedly")
        try:
            self._report("running", tid, f"goal: {goal}")
            while True:
                if self._cancel.is_set():
                    final = ("cancelled", "")
                    break
                state = {"holding": self._holding, "visible": self._visible(),
                         "skills": sorted(self.brain.skills)}
                d = self.brain.decide(goal, state, history)
                if self._cancel.is_set():
                    final = ("cancelled", "")
                    break
                step = len(history) + 1
                self.get_logger().info(f"[step {step}] {d}")
                self._report("running", tid, step=step, action=d["action"],
                             source=d["source"], reason=d["reason"])
                if d["action"] == "done":
                    final = ("succeeded", d["reason"])
                    break
                if d["action"] == "give_up":
                    final = ("failed", d["reason"])
                    break
                sid = uuid.uuid4().hex[:8]
                self.skill_pub.publish(String(data=json.dumps(
                    {"id": sid, "skill": d["action"], "object": d["object"],
                     "target": d["target"]})))
                res = self._wait_result(sid)
                if res is None:
                    self.skill_cancel_pub.publish(String(data="cancel"))
                    if self._cancel.is_set():
                        final = ("cancelled", "")
                        break
                    res = {"state": "failed", "msg": "skill timeout"}
                if res["state"] == "cancelled":
                    final = ("cancelled", "")
                    break
                history.append({"action": d["action"], "object": d["object"],
                                "result": res["state"], "msg": res.get("msg", "")})
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"agent task crashed: {e!r}")
            final = ("failed", f"crashed: {e!r}")
        self._task_id = None
        self._report(final[0], tid, final[1])
        self._busy.release()


def main():
    rclpy.init()
    node = RobotAgent()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
