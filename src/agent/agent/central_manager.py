"""CENTRAL LLM (level 1). Used only when the rule parser cannot handle the request
(unknown phrasing, several robots in one sentence, open-ended goals).

The LLM proposes, this module disposes: the JSON is validated against the fleet, the
locations and the known objects. ANY invalid element -> the whole plan is rejected ([]),
so a hallucinated robot / place can never reach a robot.
Stop is intentionally NOT an LLM action: it stays a rule-based fast path.
"""

import json

try:
    from .agent import KNOWN_OBJECTS
except ImportError:  # run as plain scripts
    from agent import KNOWN_OBJECTS

ACTIONS = ("navigate", "find", "pick", "place", "pick_place", "track", "delegate")

SYSTEM = """You are the CENTRAL planner of a robot fleet. Turn the user request into tasks.
Output ONLY this JSON, nothing else:
{"tasks":[{"robot":null,"action":"find","object":null,"target":null,"goals":[],"goal_text":""}]}
Actions:
 navigate   - drive to saved locations (goals = location names)
 find       - detect an object and drive to it (object)
 pick       - pick an object (object)
 place      - place the held ball into a target (target)
 pick_place - pick the ball then place it into the target
 track      - follow an object (object)
 delegate   - open-ended goal; write it in goal_text, the robot reasons about the steps itself
Rules: robot = null when the user names no robot. Use ONLY robots, locations and objects from
the given lists. At most 4 tasks. If the request is unclear or unsafe output {"tasks":[]}.
Examples:
request "sr1 grab the ball then put it in the box" -> {"tasks":[{"robot":"sr1","action":"pick_place","object":"ball","target":"box"}]}
request "sr2 go to room1 and sr1 find the bottle" -> {"tasks":[{"robot":"sr2","action":"navigate","goals":["room1"]},{"robot":"sr1","action":"find","object":"bottle"}]}
request "bring the ball to the box" -> {"tasks":[{"robot":null,"action":"delegate","goal_text":"pick the ball and place it in the box"}]}"""


class CentralPlanner:
    def __init__(self, llm, max_tasks=4):
        self.llm = llm
        self.max_tasks = max_tasks

    def plan(self, text, fleet, locations):
        """fleet = [{"name","skills","status","holding"}]. Returns list of intent dicts."""
        user = json.dumps({"request": text, "robots": fleet,
                           "locations": list(locations), "objects": list(KNOWN_OBJECTS)})
        raw = self.llm.ask_json(SYSTEM, user, max_tokens=320)
        return self.validate(raw, text, [r["name"] for r in fleet], locations)

    def validate(self, raw, text, robots, locations):
        tasks = raw.get("tasks") if isinstance(raw, dict) else raw
        if not isinstance(tasks, list) or not tasks or len(tasks) > self.max_tasks:
            return []
        robots = [r.lower() for r in robots]
        locs = [l.lower() for l in locations]
        intents = []
        for t in tasks:
            if not isinstance(t, dict):
                return []
            action = str(t.get("action", "")).lower()
            if action not in ACTIONS:
                return []
            robot = t.get("robot")
            robot = None if robot in (None, "", "null", "none") else str(robot).lower()
            if robot is not None and robot not in robots:
                return []
            obj = t.get("object")
            obj = None if obj in (None, "", "null") else str(obj).lower()
            if obj is not None and obj not in KNOWN_OBJECTS:
                return []
            target = t.get("target")
            target = "box" if target in (None, "", "null") else str(target).lower()
            if target not in KNOWN_OBJECTS:
                return []

            if action == "navigate":
                goals = [str(g).lower() for g in (t.get("goals") or [])]
                if not goals or any(g not in locs for g in goals):
                    return []
                intents.append({"mode": "navigate", "robot": robot, "goals": goals})
            elif action in ("find", "track", "pick"):
                if obj is None:
                    return []
                intents.append({"mode": action, "robot": robot, "object": obj})
            elif action in ("place", "pick_place"):
                intents.append({"mode": action, "robot": robot, "object": "ball",
                                "target": target})
            else:  # delegate
                goal = str(t.get("goal_text") or text).strip()
                if not goal:
                    return []
                intents.append({"mode": "delegate", "robot": robot, "goal": goal})
        return intents