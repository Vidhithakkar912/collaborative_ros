#!/usr/bin/env python3
"""
Generic replacement for TransporterBrain / ManipulatorBrain.

WHAT THIS ADDS
--------------
1. reason_about_task() now asks the local LLM for a structured JSON
   skill list, not a natural-language explanation. This is the
   "decentralized" half of the architecture: the central node only
   knows coarse actions (pick/place/transport), this node's own small
   LLM breaks that into the robot's actual skill primitives, using
   ONLY that robot's own skill list — so a transporter physically
   cannot be told to "pick" here, the schema won't allow it.

2. Skill execution is a clearly marked stub loop (`_execute_skills`)
   that logs+sleeps per skill. Replace each branch with your real
   calls (Nav2 BasicNavigator.goToPose for navigate_to, a gripper
   service for pick/place, etc). Since you already import
   nav2_simple_commander elsewhere, an example wire-up for
   navigate_to is included.

3. On completion (or on a skill-parse/execution failure), this node
   publishes to /task_status with the task's id, so the central
   TaskManager patch can unlock dependent tasks or trigger a replan.

4. One file, two robots: instantiate with ROBOT_NAME="transporter" or
   "manipulator" and the matching SKILLS list — no code duplication.
"""

import json
import time
import rclpy

from rclpy.node import Node
from std_msgs.msg import String
from llama_cpp import Llama


ROBOT_NAME = "manipulator"  
TASK_TOPIC = f"/{ROBOT_NAME}_task"
SKILLS = {
    "transporter": ["find_object", "report_object_position", "navigate_to",
                     "approach_object", "transport", "handover", "stop"],
    "manipulator": ["find_object", "report_object_position", "navigate_to",
                     "approach_object", "pick", "place", "handover", "stop"],
}[ROBOT_NAME]

SKILL_SCHEMA_PROMPT = """You are the local brain of the {robot} robot.

Your ONLY valid skills are: {skills}

You do not have any other capability. If the task from the central
manager asks for something outside this list (e.g. asking you to pick
when you have no "pick" skill), output an empty skills list and explain
why in "note".

Convert the task below into an ordered list of skill calls. Output ONLY
JSON of this exact shape, nothing else:

{{"skills": [{{"skill": "<one of your skills>", "args": {{"object": "<obj or null>", "location": "<loc or null>"}}}}],
  "note": "<short note, empty string if nothing unusual>"}}

TASK FROM CENTRAL MANAGER (id={task_id}):
{task}
"""


class RobotBrain(Node):
    def __init__(self):
        super().__init__(f"{ROBOT_NAME}_brain")

        model_path = "/home/vidhi123/models/qwen2.5-coder-0.5b-instruct-q4_0.gguf"
        self.llm = Llama(model_path=model_path, n_ctx=4096, n_threads=8, verbose=False)
        self.get_logger().info(f"{ROBOT_NAME} brain: Qwen loaded.")

        self.task_sub = self.create_subscription(String, TASK_TOPIC, self.task_callback, 10)
        self.status_pub = self.create_publisher(String, "/task_status", 10)

        # TODO once Nav2 is wired in per-robot namespace:
        # from nav2_simple_commander.robot_navigator import BasicNavigator
        # self.navigator = BasicNavigator(namespace=ROBOT_NAME)

        self.get_logger().info(f"{ROBOT_NAME} brain ready.")

    def task_callback(self, msg: String):
        try:
            task = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error(f"Bad task payload: {msg.data}")
            return

        task_id = task.get("id", "unknown")
        self.get_logger().info(f"Received task {task_id}: {task}")

        skill_plan = self._plan_skills(task)
        if skill_plan is None or not skill_plan.get("skills"):
            note = (skill_plan or {}).get("note", "no skill plan produced")
            self.get_logger().warn(f"Task {task_id} not executable here: {note}")
            self._report_status(task_id, "failed", detail=note)
            return

        ok, detail = self._execute_skills(task_id, skill_plan["skills"])
        self._report_status(task_id, "done" if ok else "failed", detail=detail)

   
    def _plan_skills(self, task):
        prompt = SKILL_SCHEMA_PROMPT.format(
            robot=ROBOT_NAME,
            skills=", ".join(SKILLS),
            task_id=task.get("id", "unknown"),
            task=json.dumps(task),
        )
        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": f"You are the local brain of the {ROBOT_NAME}."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            text = response["choices"][0]["message"]["content"]
            start, end = text.find("{"), text.rfind("}")
            parsed = json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.get_logger().error(f"Skill planning failed to parse: {e}")
            return None

        valid_skills = [
            s for s in parsed.get("skills", [])
            if isinstance(s, dict) and s.get("skill") in SKILLS
        ]
        parsed["skills"] = valid_skills
        return parsed

    def _execute_skills(self, task_id, skills):
        for step in skills:
            skill, args = step["skill"], step.get("args", {})
            self.get_logger().info(f"[{task_id}] executing skill: {skill}({args})")

            # TODO: replace each branch with a real call.
            if skill == "navigate_to":
                # Example once BasicNavigator is wired in:
                # pose = self._lookup_location_pose(args.get("location"))
                # self.navigator.goToPose(pose)
                # while not self.navigator.isTaskComplete():
                #     time.sleep(0.2)
                # result = self.navigator.getResult()
                # if result != TaskResult.SUCCEEDED:
                #     return False, f"navigate_to {args.get('location')} failed"
                time.sleep(0.5)
            elif skill in ("pick", "place", "handover", "transport",
                            "find_object", "approach_object",
                            "report_object_position", "stop"):
                time.sleep(0.3)
            else:
                return False, f"unhandled skill: {skill}"

        return True, f"completed {len(skills)} skill(s)"

    def _report_status(self, task_id, status, detail=""):
        out = String()
        out.data = json.dumps({"id": task_id, "robot": ROBOT_NAME, "status": status, "detail": detail})
        self.status_pub.publish(out)
        self.get_logger().info(f"[{task_id}] status -> {status} ({detail})")


def main(args=None):
    rclpy.init(args=args)
    node = RobotBrain()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()