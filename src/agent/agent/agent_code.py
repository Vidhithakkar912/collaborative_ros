"""Agent core (no ROS here, so it is easy to unit-test):
  - RobotRegistry : which robots exist and which skills each one has (robots.yaml)
  - Tool/ToolRegistry : MCP-style tool table  (name, description, required skills, handler)
  - parse_command : text -> intent dict {"mode": <tool name>, ...}
"""

import re
from dataclasses import dataclass
from typing import Callable, Optional

import yaml

DEFAULT_SKILLS = ("navigate", "detect", "track", "find")   # used if robots.yaml has no `skills:`
KNOWN_OBJECTS = ("ball", "box", "bottle", "chair")


# --------------------------------------------------------------------------- robots
class RobotRegistry:
    """robots.yaml:
        robots:
          - {name: sr1, enabled: true, skills: [navigate, detect, track, find, pick, place]}
          - {name: sr2, enabled: true, skills: [navigate, detect, track, find]}
    Names are used in lower case everywhere (topics are /sr1/...)."""

    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        self._skills = {}
        for r in data["robots"]:
            if not r.get("enabled", True):
                continue
            self._skills[r["name"].lower()] = {
                s.lower() for s in r.get("skills", DEFAULT_SKILLS)}

    def names(self):
        return list(self._skills)

    def known(self, name) -> bool:
        return name in self._skills

    def skills(self, name):
        return self._skills.get(name, set())

    def has_skills(self, name, required) -> bool:
        return set(required) <= self.skills(name)

    def with_skills(self, required):
        return [r for r in self._skills if self.has_skills(r, required)]


# --------------------------------------------------------------------------- tools
@dataclass
class Tool:
    name: str                              # == intent["mode"]
    description: str
    handler: Callable                      # handler(robot: str, cmd: dict)
    required_skills: tuple = ()
    needs_robot: bool = True               # False for stop-style tools
    auto_assign: Optional[str] = None      # "location" | "xy" | "object" | "holding" | None
    uses_skill_server: bool = False        # True -> pick / place / find (skill_server.py)
    uses_agent: bool = False               # True -> needs the robot's own LLM agent


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name) -> Optional[Tool]:
        return self._tools.get(name)

    def describe(self):
        return [(t.name, t.description, t.required_skills) for t in self._tools.values()]


# --------------------------------------------------------------------------- parser
def _w(*words):
    return re.compile(r"\b(" + "|".join(words) + r")\b")


RE_STOP = _w("stop", "halt", "abort", "cancel")
RE_PICK = _w("pick", "grab", "grasp")
RE_PLACE = _w("place", "put", "drop", "release")
RE_TRACK = _w("track", "follow")
RE_FIND = _w("find", "search", "locate", "detect", "look for")
RE_GOTO = re.compile(r"\b(go|move|navigate|come|drive)\b.*\b(to|towards|toward)\b")
RE_DIR = _w("left", "right", "forward", "backward")
RE_XYZ = re.compile(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


def parse_command(text: str, robots, locations) -> Optional[dict]:
    """Rule-based on purpose: deterministic and instant. The LLM is only used for chat."""
    t = text.lower().strip()
    robot = next((r for r in robots if re.search(rf"\b{re.escape(r)}\b", t)), None)
    obj = next((o for o in KNOWN_OBJECTS if re.search(rf"\b{o}\b", t)), None)

    # ---- stop family (always first, safest)
    if "stop tracking" in t:
        return {"mode": "stop_tracking", "robot": robot}
    if "stop detection" in t:
        return {"mode": "stop_detection", "robot": robot}
    if RE_STOP.search(t):
        if robot is None or re.search(r"\ball\b", t):
            return {"mode": "stop_all"}
        return {"mode": "stop", "robot": robot}

    # ---- manipulation skills
    has_pick, has_place = RE_PICK.search(t), RE_PLACE.search(t)
    others = [o for o in KNOWN_OBJECTS if o != "ball" and re.search(rf"\b{o}\b", t)]
    target = others[0] if others else "box"
    if has_pick and has_place:
        return {"mode": "pick_place", "robot": robot, "object": "ball", "target": target}
    if has_pick:
        return {"mode": "pick", "robot": robot, "object": obj or "ball"}
    if has_place:
        return {"mode": "place", "robot": robot, "object": "ball", "target": target}

    # ---- track / find
    if RE_TRACK.search(t):
        return {"mode": "track", "robot": robot, "object": obj}
    if RE_FIND.search(t) or (RE_GOTO.search(t) and obj):
        return {"mode": "find", "robot": robot, "object": obj}

    # ---- plain movement / coordinates / saved locations
    m = RE_DIR.search(t)
    if m:
        return {"mode": "movement", "robot": robot, "direction": m.group(1)}

    m = RE_XYZ.search(t)
    if m:
        return {"mode": "coordinate", "robot": robot, "x": float(m.group(1)),
                "y": float(m.group(2)), "yaw": float(m.group(3))}

    found = sorted((t.find(loc.lower()), loc.lower()) for loc in locations
                   if loc.lower() in t)
    if found:
        return {"mode": "navigate", "robot": robot, "goals": [g for _, g in found]}

    if obj:                                   # bare "sr1 ball" -> treat as find
        return {"mode": "find", "robot": robot, "object": obj}
    return None