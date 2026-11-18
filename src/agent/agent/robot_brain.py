"""ROBOT-LEVEL reasoning (level 2), pure python (no ROS) so it can be tested anywhere.

Given a goal ("pick the ball and place it in the box"), the robot's own state and the
history of what it already tried, decide the NEXT action:
    find | pick | place | done | give_up
The local LLM proposes; _validate() checks the proposal against hard rules; if the LLM is
missing, answers garbage or breaks a rule, the deterministic _rule() decides instead.
"""

import json

from agent.agent_code import KNOWN_OBJECTS, parse_command


PLAN = {"pick": ["pick"], "place": ["place"], "pick_place": ["pick", "place"],
        "find": ["find"]}
ACTIONS = ("find", "pick", "place", "done", "give_up")

SYSTEM = """You are the onboard reasoning module of ONE robot. Choose the NEXT action toward the goal.
Output ONLY JSON: {"action":"find|pick|place|done|give_up","object":null,"target":null,"reason":"short"}
find = detect an object and drive to it. pick = detect, drive to and grasp the object. place = put the held object into the target.
Rules: pick only when holding=false. place only when holding=true. done only when the whole goal is achieved. If an action failed you may retry it once or use find first; after repeated failures use give_up.
Example: goal "pick the ball and place it in the box", holding=false, history=[] -> {"action":"pick","object":"ball","target":null,"reason":"ball not held yet"}
Example: same goal, holding=true, history=[pick succeeded] -> {"action":"place","object":"ball","target":"box","reason":"ball is held"}"""


class AgentBrain:
    def __init__(self, llm, skills, max_steps=6, max_retries=1):
        self.llm = llm                       # None -> rules only
        self.skills = set(skills)
        self.max_steps = max_steps
        self.max_retries = max_retries

    # ---------------------------------------------------------------- goal
    @staticmethod
    def parse_goal(goal):
        cmd = parse_command(goal, [], [])
        if not cmd or cmd["mode"] not in PLAN:
            return [], "ball", "box"
        return list(PLAN[cmd["mode"]]), cmd.get("object") or "ball", cmd.get("target") or "box"

    @staticmethod
    def _succeeded(history):
        return {h["action"] for h in history if h["result"] == "succeeded"}

    @staticmethod
    def _consecutive_fails(history, action):
        n = 0
        for h in reversed(history):
            if h["action"] == action and h["result"] == "failed":
                n += 1
            else:
                break
        return n

    # ---------------------------------------------------------------- decide
    def decide(self, goal, state, history):
        """state = {"holding": bool, "visible": [..], "skills": [..]}. Returns
        {"action","object","target","reason","source": "llm"|"rule"}."""
        required, obj, target = self.parse_goal(goal)
        if len(history) >= self.max_steps:
            return {"action": "give_up", "object": None, "target": None,
                    "reason": f"step limit ({self.max_steps}) reached", "source": "rule"}
        if self.llm is not None:
            user = json.dumps({"goal": goal, "state": state, "history": history[-4:]})
            proposal = self.llm.ask_json(SYSTEM, user, max_tokens=160)
            act = self._validate(proposal, required, obj, target, state, history)
            if act:
                act["source"] = "llm"
                return act
        act = self._rule(required, obj, target, state, history)
        act["source"] = "rule"
        return act

    def _validate(self, d, required, obj, target, state, history):
        if not isinstance(d, dict):
            return None
        a = str(d.get("action", "")).lower()
        if a not in ACTIONS:
            return None
        reason = str(d.get("reason", ""))[:120]
        done = self._succeeded(history)
        if a == "give_up":
            return {"action": a, "object": None, "target": None, "reason": reason}
        if a == "done":
            if required and all(r in done for r in required):
                return {"action": a, "object": None, "target": None, "reason": reason}
            return None
        if a not in self.skills:
            return None
        if self._consecutive_fails(history, a) > self.max_retries:
            return None
        if a == "pick" and state.get("holding"):
            return None
        if a == "place" and not state.get("holding") and "pick" in required:
            return None
        if a == "find" and "find" not in required and any(h["action"] == "find" for h in history):
            return None
        o = str(d.get("object") or "").lower()
        t = str(d.get("target") or "").lower()
        return {"action": a, "object": o if o in KNOWN_OBJECTS else obj,
                "target": t if t in KNOWN_OBJECTS else target, "reason": reason}

    def _rule(self, required, obj, target, state, history):
        def act(a, why):
            return {"action": a, "object": obj if a != "place" else "ball",
                    "target": target if a == "place" else None, "reason": why}
        if not required:
            return act("give_up", "goal not understood")
        done = self._succeeded(history)
        remaining = [r for r in required if r not in done]
        if state.get("holding"):
            remaining = [r for r in remaining if r != "pick"]
        if not remaining:
            return act("done", "all steps finished")
        nxt = remaining[0]
        if nxt not in self.skills:
            return act("give_up", f"robot has no '{nxt}' skill")
        if self._consecutive_fails(history, nxt) > self.max_retries:
            return act("give_up", f"{nxt} failed repeatedly")
        return act(nxt, "next required step" if not history else "continue plan / retry")