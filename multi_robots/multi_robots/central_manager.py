#!/usr/bin/env python3


import json
import os
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from llama_cpp import Llama
from ament_index_python.packages import get_package_share_directory
import yaml
import re

ROBOT_TASK_TOPIC = {
    "transporter": "/transporter_task",
    "manipulator": "/manipulator_task",
}

TASK_SCHEMA_PROMPT = """
You are the high-level task planner for a heterogeneous
multi-robot system.

YOUR ROLE


Your job is to understand an arbitrary natural-language
command and determine the sequence of physical actions
required to accomplish the user's goal.

You are responsible for SEMANTIC REASONING and TASK
DECOMPOSITION.

You are NOT responsible for final robot assignment.

A separate deterministic module will assign each task to
a robot using the robot capability configuration.


ROBOT CONFIGURATION


Available robots and capabilities:

{capabilities}

You must understand these capabilities when deciding whether
a plan is physically achievable.

Do not assume that a robot can perform an action unless
that capability exists.


AVAILABLE ACTIONS


The robot system supports the following primitive actions:

- find
- navigate
- pick
- place
- transport
- handover

PLANNING REQUIREMENTS


For every user command:

1. Understand the overall intent.

2. Identify all relevant objects.

3. Identify the object that is being manipulated.

4. Identify destinations, targets, reference objects,
   and locations.

5. Infer actions that are logically necessary even if the
   user did not explicitly mention them.

6. Decompose the goal into atomic physical actions.

7. Each step must represent one physical action.

8. Preserve the correct logical order.

9. Do not invent objects that are not mentioned or
   logically required.

10. Do not add unnecessary actions.

11. Do not assign robots to tasks.

12. Do not output JSON.

13. Do not explain your reasoning.

14. Produce only the final numbered execution plan.


SEMANTIC REASONING


Understand the semantic meaning of the command rather than
matching keywords.

For example:

"Put the ball in the box"

means that:

- ball is the manipulated object
- box is the destination

Therefore the required plan is:

1. Find the ball.
2. Pick up the ball.
3. Find the box.
4. Place the ball inside the box.

Similarly:

"Take the box to the table"

means:

- box is the manipulated object
- table is the destination

Therefore:

1. Find the box.
2. Pick up the box.
3. Find the table.
4. Place the box at the table.


DEPENDENCY REASONING


Respect physical dependencies.

For example:

An object must normally be found before it can be picked.

A manipulated object must normally be picked before it
can be placed.

A target should normally be found before the object is
placed there.

Therefore:

Find object
Pick object
Find target
Place object at target

EXAMPLES


Example 1:

User:
Pick the ball and put it on the box.

Plan:
1. Find the ball.
2. Pick up the ball.
3. Find the box.
4. Place the ball on the box.


Example 2:

User:
Put the red ball inside the blue box.

Plan:
1. Find the red ball.
2. Pick up the red ball.
3. Find the blue box.
4. Place the red ball inside the blue box.


Example 3:

User:
Move the box to the table.

Plan:
1. Find the box.
2. Pick up the box.
3. Find the table.
4. Place the box at the table.


Example 4:

User:
Find the bottle next to the chair and put it on
the table.

Plan:
1. Find the chair.
2. Find the bottle next to the chair.
3. Pick up the bottle.
4. Find the table.
5. Place the bottle on the table.


Example 5:

User:
Take the ball from the table and put it inside the box.

Plan:
1. Find the ball.
2. Pick up the ball.
3. Find the box.
4. Place the ball inside the box.


Example 6:

User:
Move the box near the ball.

Plan:
1. Find the box.
2. Pick up the box.
3. Find the ball.
4. Place the box near the ball.

Every task MUST contain:
id
action
object
robot
depends_on

Never omit robot.

If a sequence of tasks performs one continuous physical task,
all dependent manipulation tasks must use the same assigned robot
unless explicitly reassigned.

For "pick up the ball", object must be "ball", not "up the ball".

OUTPUT FORMAT


Return ONLY a numbered plan.

Each line must contain exactly one atomic action.

Do not output JSON.

Do not output markdown.

Do not output explanations.

Do not assign robots.


USER COMMAND


{user_prompt}

Generate the execution plan.
"""


class TaskManager(Node):
    def __init__(self):
        super().__init__("task_manager_node")

        package_share = get_package_share_directory("multi_robots")
        config_path = os.path.join(package_share, "config", "robot_capability.yaml")
        with open(config_path, "r") as f:
            self.robot_config = yaml.safe_load(f)
        self.get_logger().info(f"Loaded robot configuration: {config_path}")

        model_path = "/home/vidhi123/models/qwen2.5-3b-instruct-q4_k_m.gguf"
        self.llm = Llama(model_path=model_path, n_ctx=4096, n_threads=8, verbose=False)
        self.get_logger().info("Qwen 2.5 3B loaded.")

        
        self.tasks = {}            
        self.task_status = {}      
        self.completed_summary = []  

        self.chat_sub = self.create_subscription(String, "chat_input", self.chat_input_callback, 10)
        self.status_sub = self.create_subscription(String, "/task_status", self.task_status_callback, 10)

        self.transporter_pub = self.create_publisher(String, "/transporter_task", 10)
        self.manipulator_pub = self.create_publisher(String, "/manipulator_task", 10)
        self.chat_pub = self.create_publisher(String, "/chat_output", 10)

        self.get_logger().info("Task Manager ready.")

    def chat_input_callback(self, msg: String):
        user_prompt = msg.data.strip()
        if not user_prompt:
            return
        self.get_logger().info(f"\n user prompt is : {user_prompt}")
        self._plan_and_dispatch(user_prompt, context="")
    def build_dependencies(self, plan):
        self.get_logger().info("dependencies")
        self.get_logger().info(f"plan is : {plan}")
        tasks = plan["tasks"]

        last_find = {}
        last_pick = {}

        for task in tasks:

            action = task["action"]
            obj = task.get("object")
            location = task.get("location")

            if action == "find":

                if obj:
                    last_find[obj.lower()] = task["id"]

            elif action == "pick":

                if obj:

                    find_id = last_find.get(
                        obj.lower()
                    )

                    if find_id:
                        task["depends_on"] = [find_id]

                    last_pick[obj.lower()] = task["id"]

            elif action == "navigate":

                task["depends_on"] = []

            elif action == "place":

                dependencies = []

                if obj:

                    pick_id = last_pick.get(
                        obj.lower()
                    )

                    if pick_id:
                        dependencies.append(
                            pick_id
                        )

                if location:

                    target_find_id = last_find.get(
                        location.lower()
                    )

                    if target_find_id:
                        dependencies.append(
                            target_find_id
                        )

                task["depends_on"] = list(
                    dict.fromkeys(dependencies)
                )

        return plan
    def _plan_and_dispatch(self, user_prompt, context):
        response = self.ask_to_model(user_prompt, context)
        self.get_logger().info("response done")
        if response is None:
            self.get_logger().error("Failed to generate a valid response.")
            return

        plan = self.parse_planning(response)
        self.get_logger().info("natural_plan_done")
        if plan is None:
            self.get_logger().error("Failed to generate a valid plan.")
            return

        plan=self.build_dependencies(plan)
        self.get_logger().info("build dependencies done")
        if plan is None:
                    self.get_logger().error("Failed to generate a valid plan.")
                    return
        plan = self.assign_robots(plan)
        self.get_logger().info("assign robots task done")
        if plan is None:
            self.get_logger().error(
                "No valid robot assignment."
            )
            return
        new_tasks={}
        for task in plan["tasks"]:
            if not self._validate_task(task):
                self.get_logger().error(
                f"Invalid task: {task}"
            )
                return

            new_tasks[task["id"]] = task

        self.tasks.update(new_tasks)

        for task_id in new_tasks:
            self.task_status.setdefault(
                task_id,
                "pending"
            )
        self._dispatch_ready_tasks()

    def _validate_task(self, task):
        if not isinstance(task, dict):
            return False
        required = [ "id","robot","action"]
        for field in required:
            if field not in task:
                self.get_logger().error(f"task missing '{field}':{task}")
                return False
        robot = task["robot"]
        action = task["action"]
        if robot not in self.robot_config.get("robots",{}):
            self.get_logger().error(
            f"Unknown robot: {robot}"
        )
        capabilities = self.robot_config[
        "robots"
        ][robot].get(
            "capabilities",
            []
        )
        if action not in capabilities:

            self.get_logger().error(
                f"INVALID PLAN: {robot} cannot "
                f"perform {action}"
            )

            return False

        task.setdefault(
            "depends_on",
            []
        )

        task.setdefault(
            "object",
            None
        )

        task.setdefault(
            "location",
            None
        )



        return True
    def _dispatch_ready_tasks(self):
        for tid, task in self.tasks.items():
            if self.task_status.get(tid) != "pending":
                continue
            deps = task.get("depends_on", [])
            if all(self.task_status.get(d) == "done" for d in deps):
                self.get_logger().info(f"task is {task}")
                self._dispatch_task(task)
    def parse_planning(self, response):

            tasks = []

            lines = response.splitlines()

            task_id = 1

            for line in lines:

                line = line.strip()

                if not line:
                    continue

                
                line = re.sub(
                    r"^\s*\d+[\.\)]\s*",
                    "",
                    line
                )

                line = line.strip(" .")

                lower = line.lower()


                if lower.startswith("find "):

                    obj = line[5:].strip(" .")

                    tasks.append({
                        "id": f"t{task_id}",
                        "action": "find",
                        "object": obj,
                        "depends_on": []
                    })

             

                elif lower.startswith("pick "):

                    obj = line[5:].strip(" .")

                    tasks.append({
                        "id": f"t{task_id}",
                        "action": "pick",
                        "object": obj,
                        "depends_on": []
                    })

                elif lower.startswith("pick up "):

                    obj = line[8:].strip(" .")

                    tasks.append({
                        "id": f"t{task_id}",
                        "action": "pick",
                        "object": obj,
                        "depends_on": []
                    })

               

                elif lower.startswith("navigate "):

                    location = line[9:].strip(" .")

                    tasks.append({
                        "id": f"t{task_id}",
                        "action": "navigate",
                        "location": location,
                        "depends_on": []
                    })

               

                elif lower.startswith("place "):

                    content = line[6:].strip(" .")

                    match = re.match(
                        r"(.+?)\s+(?:on|in|inside|at|near)\s+(.+)",
                        content,
                        re.IGNORECASE
                    )

                    if match:

                        obj = match.group(1).strip()
                        target = match.group(2).strip()

                        tasks.append({
                            "id": f"t{task_id}",
                            "action": "place",
                            "object": obj,
                            "location": target,
                            "depends_on": []
                        })

                task_id += 1

            if not tasks:

                self.get_logger().error(
                    "Could not parse any tasks from LLM plan."
                )

                return None

            return {
                "tasks": tasks
            }
    def _dispatch_task(self, task):
        topic = ROBOT_TASK_TOPIC[task["robot"]]
        pub = self.transporter_pub if task["robot"] == "transporter" else self.manipulator_pub
        msg = String()
        msg.data = json.dumps(task)
        pub.publish(msg)
        self.task_status[task["id"]] = "dispatched"
        self.get_logger().info(f"Dispatched {task['id']} -> {task['robot']} ({topic}): {task}")

    def task_status_callback(self, msg: String):
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error(f"Bad status message: {msg.data}")
            return

        tid = status.get("id")
        state = status.get("status")
        if tid not in self.tasks:
            return

        if state == "done":
            self.task_status[tid] = "done"
            self.completed_summary.append(
                f"{tid} ({self.tasks[tid]['robot']}: {self.tasks[tid]['action']} "
                f"{self.tasks[tid].get('object')}) completed successfully."
            )
            self.get_logger().info(f"{tid} done.")
            self._dispatch_ready_tasks()

        elif state == "failed":
            self.task_status[tid] = "failed"
            detail = status.get("detail", "no detail provided")
            self.get_logger().warn(f"{tid} failed: {detail}. Triggering replan.")
            self._replan_after_failure(tid, detail)

    def _replan_after_failure(self, failed_id, detail):
        failed_task = self.tasks[failed_id]
        remaining = [
            t for t_id, t in self.tasks.items()
            if self.task_status.get(t_id) in ("pending", "dispatched", "failed")
        ]
        context = (
            "Context: this is a replan after a failure mid-execution.\n"
            f"Already completed: {self.completed_summary}\n"
            f"Failed task: {failed_task} — reason: {detail}\n"
            f"Remaining/incomplete tasks before the failure: {remaining}\n"
            "Produce a new task list to accomplish the ORIGINAL remaining "
            "goal, working around the failure (e.g. reassign to a different "
            "capable robot if one exists, or pick a different approach). "
            "Do not repeat tasks already completed."
        )
        self._feedback(f"Replanning after {failed_id} failed: {detail}")
       
        self._plan_and_dispatch(
            user_prompt="(continue the original goal given the context above)",
            context=context,
        )

    def assign_robots(self, plan):

        robots = self.robot_config.get(
            "robots",
            {}
        )

        for task in plan["tasks"]:

            action = task["action"]

            capable_robots = []

            for robot_name, robot_info in robots.items():

                capabilities = robot_info.get(
                    "capabilities",
                    []
                )

                if action in capabilities:
                    capable_robots.append(
                        robot_name
                    )

            if not capable_robots:

                self.get_logger().error(
                    f"No robot has capability "
                    f"'{action}'"
                )

                return None

            
            task["robot"] = capable_robots[0]

            self.get_logger().info(
                f"{task['id']} | "
                f"{action} | "
                f"{task['robot']}"
            )

        return plan
    def _feedback(self, text):
        out = String()
        out.data = f"planner|{text}"
        self.chat_pub.publish(out)
        self.get_logger().info(f"[FEEDBACK] {text}")


    def ask_to_model(self, user_prompt, context):
        capabilities = yaml.safe_dump(self.robot_config, sort_keys=False)
        system_prompt = TASK_SCHEMA_PROMPT.format(
            capabilities=capabilities, context=context, user_prompt=user_prompt
        )
        if context:
            system_prompt+=f"""
            {context}
            """
        response = self.llm.create_chat_completion(
            messages=[{ "role": "system", "content": system_prompt} ,
                      {"role": "user","content":user_prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        natural_plan=response["choices"][0]["message"]["content"]
        self.get_logger().info(f"{natural_plan}")
        return natural_plan

def main(args=None):
    rclpy.init(args=args)
    node = TaskManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()