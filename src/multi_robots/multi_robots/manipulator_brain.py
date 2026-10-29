#!/usr/bin/env python3
"""
Manipulator executor.

Subscribes : /manipulator_task  (std_msgs/String, JSON from TaskManager)
Publishes  : /task_status       (std_msgs/String, JSON)
             {"id": "...", "status": "running" | "done" | "failed", "detail": "..."}

Task JSON example:
  {"id": "p1_t2", "action": "pick", "object": "ball", "location": null,
   "robot": "manipulator", "depends_on": ["p1_t1"]}
"""

import json
import queue
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TaskFailed(Exception):
    """Raise inside a handler to report a failed task with a reason."""


class ManipulatorExecutor(Node):
    ROBOT_NAME = "manipulator"

    def __init__(self):
        super().__init__("manipulator_executor")

        self.declare_parameter("sim_delay", 2.0)  
        self.sim_delay = float(self.get_parameter("sim_delay").value)

        self.holding = None          
        self.seen_ids = set()       
        self.task_queue = queue.Queue()
        self._stop_event = threading.Event()

        self.handlers = {
            "find": self.do_find,
            "pick": self.do_pick,
            "place": self.do_place,
            "handover": self.do_handover,
        }

        self.task_sub = self.create_subscription(
            String, "/manipulator_task", self.task_callback, 10
        )
        self.status_pub = self.create_publisher(String, "/task_status", 10)

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

        self.get_logger().info("Manipulator executor ready, waiting for tasks.")

   
    def task_callback(self, msg: String):
        try:
            task = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error(f"Bad task JSON: {msg.data}")
            return

        task_id = task.get("id")
        action = task.get("action")

        if not task_id or not action:
            self.get_logger().error(f"Task missing id/action: {task}")
            return

        if task.get("robot") not in (None, self.ROBOT_NAME):
            self.get_logger().warn(f"Task {task_id} not for me: {task.get('robot')}")
            return

        if task_id in self.seen_ids:
            self.get_logger().warn(f"Duplicate task ignored: {task_id}")
            return
        self.seen_ids.add(task_id)

        self.get_logger().info(f"Received {task_id}: {action} {task.get('object')}")
        self.task_queue.put(task)

   
    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            task_id = task["id"]
            action = task["action"]
            self._publish_status(task_id, "running")

            handler = self.handlers.get(action)
            if handler is None:
                self._publish_status(task_id, "failed", f"unsupported action '{action}'")
                continue

            try:
                handler(task)
                self._publish_status(task_id, "done")
            except TaskFailed as e:
                self.get_logger().warn(f"{task_id} failed: {e}")
                self._publish_status(task_id, "failed", str(e))
            except Exception as e:  # never let the worker die
                self.get_logger().error(f"{task_id} crashed: {e}")
                self._publish_status(task_id, "failed", f"exception: {e}")

    def _publish_status(self, task_id, status, detail=""):
        msg = String()
        msg.data = json.dumps({"id": task_id, "status": status, "detail": detail})
        self.status_pub.publish(msg)

   
    def do_find(self, task):
        obj = self._require(task, "object")
        if not self.arm_find(obj):
            raise TaskFailed(f"could not find '{obj}'")

    def do_pick(self, task):
        obj = self._require(task, "object")
        if self.holding is not None:
            raise TaskFailed(f"gripper already holding '{self.holding}'")
        if not self.arm_pick(obj):
            raise TaskFailed(f"pick of '{obj}' failed")
        self.holding = obj

    def do_place(self, task):
        obj = self._require(task, "object")
        target = self._require(task, "location")
        if self.holding is None:
            raise TaskFailed("nothing in gripper to place")
        if self.holding.lower() != obj.lower():
            raise TaskFailed(f"holding '{self.holding}', not '{obj}'")
        if not self.arm_place(obj, target):
            raise TaskFailed(f"place of '{obj}' at '{target}' failed")
        self.holding = None

    def do_handover(self, task):
        obj = self._require(task, "object")
        if self.holding is None:
            raise TaskFailed("nothing to hand over")
        if not self.arm_handover(obj):
            raise TaskFailed("handover failed")
        self.holding = None

    @staticmethod
    def _require(task, field):
        value = task.get(field)
        if not value:
            raise TaskFailed(f"task missing '{field}'")
        return value

   
    def arm_find(self, obj) -> bool:
        self.get_logger().info(f"[arm] looking for {obj}")
        time.sleep(self.sim_delay)
        return True

    def arm_pick(self, obj) -> bool:
        self.get_logger().info(f"[arm] picking {obj}")
        time.sleep(self.sim_delay)
        return True

    def arm_place(self, obj, target) -> bool:
        self.get_logger().info(f"[arm] placing {obj} at {target}")
        time.sleep(self.sim_delay)
        return True

    def arm_handover(self, obj) -> bool:
        self.get_logger().info(f"[arm] handing over {obj}")
        time.sleep(self.sim_delay)
        return True

    def destroy_node(self):
        self._stop_event.set()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ManipulatorExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()