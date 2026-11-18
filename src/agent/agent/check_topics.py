#!/usr/bin/env python3
"""Topic-name doctor for the host <-> robot (zenoh bridge + namespace) link.

Run it on BOTH machines and read the verdict:
    HOST  : ros2 run agent check_topics --side host  --robots manipulator
    ROBOT : ros2 run agent check_topics --side robot

Design it checks (multi-robot, the standard way):
    robot nodes are NOT namespaced           /skill_status  /skill_cmd  /detected_objects ...
    robot bridge has namespace /<robot>      (zenoh_bridge_ros2dds -c zenoh_namespace.json5)
    host bridge has NO namespace             (zenoh_bridge_ros2dds -e tcp/<robot_ip>:7447)
    host (planner) sees                      /<robot>/skill_status  /<robot>/skill_cmd ...
"""

import argparse
import os
import time

FROM_ROBOT = ("skill_status", "detected_objects", "odom", "agent_report")   # robot -> host
TO_ROBOT = ("skill_cmd", "skill_cancel", "detection_cmd", "agent_task", "agent_cancel")
OPTIONAL = {"agent_report", "agent_task", "agent_cancel"}                    # robot_agent only


def _last(topic):
    return topic.rstrip("/").split("/")[-1]


def _alts(graph, expected, name, need):
    """Other topics with the same last segment that DO have what we need (pub / sub)."""
    idx = 0 if need == "pub" else 1
    return sorted(t for t, c in graph.items() if _last(t) == name and t != expected and c[idx] > 0)


def diagnose(side, robots, template, graph):
    """graph = {topic: (n_publishers, n_subscribers)} as seen on THIS machine.
    Returns list of (level, text); level in OK / INFO / FAIL."""
    out = []
    if side == "host":
        for r in robots:
            for name in FROM_ROBOT:
                exp = template.format(robot=r, name=name)
                if graph.get(exp, (0, 0))[0] > 0:
                    out.append(("OK", f"{exp}  (robot '{r}' publishes it, host receives it)"))
                    continue
                if name in OPTIONAL:
                    out.append(("INFO", f"{exp} missing (fine unless you use robot_agent)"))
                    continue
                others = {template.format(robot=o, name=name) for o in robots if o != r}
                alts = [a for a in _alts(graph, exp, name, "pub") if a not in others]
                if not alts:
                    out.append(("FAIL", f"{exp} missing and NOTHING called '{name}' arrives. "
                                        f"Bridge link / ROS_DOMAIN_ID / allow list problem, or the "
                                        f"robot node is not running (check 'ros2 topic list' on the robot)."))
                    continue
                for a in alts:
                    if a == "/" + name:
                        if len(robots) > 1:
                            fix = ("several robots: the ROBOT bridge must load a namespace "
                                   f"(namespace \"/{r}\" in zenoh_namespace.json5; its log prints the "
                                   "namespace it uses). Do NOT drop the namespace.")
                        else:
                            fix = ("fix the robot bridge namespace, OR (one robot only) run the planner "
                                   "with -p \"robot_topic_template:=/{name}\"")
                        out.append(("FAIL", f"host sees '{a}' (no robot prefix) but planner expects "
                                            f"'{exp}'. {fix}"))
                    elif a.endswith(exp) or a.endswith("/" + r + "/" + name):
                        out.append(("FAIL", f"host sees '{a}' = DOUBLE prefix. The robot node runs with "
                                            f"__ns AND the bridge adds the namespace. Start the robot "
                                            f"node WITHOUT __ns."))
                    else:
                        out.append(("FAIL", f"host sees '{a}' but planner expects '{exp}'. Use the same "
                                            f"name: -p \"robot_topic_template:=...\" must produce '{a}'."))
    else:  # robot side: local names must be un-namespaced when the bridge adds the namespace
        for name in FROM_ROBOT:
            exp = "/" + name
            if graph.get(exp, (0, 0))[0] > 0:
                out.append(("OK", f"{exp}  (published locally)"))
            elif name in OPTIONAL:
                out.append(("INFO", f"{exp} missing (fine unless you use robot_agent)"))
            else:
                alts = _alts(graph, exp, name, "pub")
                if alts:
                    out.append(("FAIL", f"robot publishes '{alts[0]}' instead of '{exp}': the node runs "
                                        f"inside a namespace. Start it WITHOUT __ns (the bridge adds it)."))
                else:
                    out.append(("FAIL", f"{exp} is not published on the robot: node not running "
                                        f"(skill_server / detection / base)."))
        for name in TO_ROBOT:
            exp = "/" + name
            if graph.get(exp, (0, 0))[1] > 0:
                out.append(("OK", f"{exp}  (a node on the robot listens)"))
            elif name in OPTIONAL:
                out.append(("INFO", f"{exp} not listened to (fine unless you use robot_agent)"))
            else:
                alts = _alts(graph, exp, name, "sub")
                if alts:
                    out.append(("FAIL", f"robot node listens on '{alts[0]}' not '{exp}': it runs inside "
                                        f"a namespace. Start it WITHOUT __ns."))
                else:
                    out.append(("FAIL", f"nobody on the robot listens on {exp}: skill_server not running."))
    return out


def format_report(results):
    mark = {"OK": "[ OK ]", "INFO": "[info]", "FAIL": "[FAIL]"}
    lines = [f"{mark[l]} {t}" for l, t in results]
    bad = sum(1 for l, _ in results if l == "FAIL")
    lines.append("")
    lines.append("ALL GOOD" if not bad else f"{bad} problem(s): fix the [FAIL] lines top to bottom.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=("host", "robot"), required=True)
    ap.add_argument("--robots", default="", help="host side: comma list, default = robots.yaml")
    ap.add_argument("--template", default="/{robot}/{name}")
    ap.add_argument("--wait", type=float, default=4.0, help="seconds for DDS discovery")
    a = ap.parse_args(argv)

    import rclpy
    from rclpy.node import Node
    rclpy.init()
    node = Node("check_topics")
    print(f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0 (unset)')}  side={a.side}  "
          f"template={a.template}")
    time.sleep(a.wait)
    graph = {}
    for topic, _types in node.get_topic_names_and_types():
        graph[topic] = (node.count_publishers(topic), node.count_subscribers(topic))

    robots = [r for r in a.robots.split(",") if r]
    if a.side == "host" and not robots:
        try:
            from ament_index_python.packages import get_package_share_directory
            from .agent import RobotRegistry
            robots = RobotRegistry(os.path.join(get_package_share_directory("agent"),
                                                "config", "robots.yaml")).names()
        except Exception as e:  # noqa: BLE001
            print(f"cannot read robots.yaml ({e}); pass --robots a,b")
            return 2
    print(format_report(diagnose(a.side, robots, a.template, graph)))
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())