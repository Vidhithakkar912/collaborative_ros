#!/usr/bin/env python3
"""Robot dashboard backend (FastAPI + optional rclpy bridge).

Run ONE per machine (robot, server), each with its own services yaml:
    python3 dashboard_server.py --config services_robot.yaml  --port 8080
    python3 dashboard_server.py --config services_server.yaml --port 8081

Features
  * start / stop / restart any node or launch file, start / stop whole groups
  * depends_on + wait_sec + ready_topic -> correct start order
  * every service runs in its OWN process group -> stop really kills `ros2 run` children
  * live logs over WebSocket
  * ROS bridge (only if rclpy is importable): topic health, per-robot skill/agent status,
    chat with the planner (/chat_input, /chat_output), STOP-robots button

Security: the browser only sends service NAMES. Commands come from the yaml file.
Optional token: export DASH_TOKEN=secret  and open  http://host:8080/?token=secret
"""

import argparse
import asyncio
import collections
import json
import os
import signal
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:                                   # ROS is optional: dashboard still runs without it
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    ROS_OK = True
except Exception:  # noqa: BLE001
    ROS_OK = False

STATIC = Path(__file__).parent / "static"
LOG_LINES = 1000
ALIVE_SEC = 6.0


# ============================================================ event hub (threads -> websockets)
class Hub:
    def __init__(self):
        self.loop = None
        self.clients = set()

    def publish(self, event: dict):
        if self.loop is None:
            return
        for q in list(self.clients):
            self.loop.call_soon_threadsafe(self._put, q, event)

    @staticmethod
    def _put(q, event):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


# ============================================================ ROS bridge
class RosBridge:
    def __init__(self, robots, watch_topics, hub):
        self.ok = False
        self.robots = robots
        self.watch_topics = watch_topics
        self.hub = hub
        self.chat = collections.deque(maxlen=200)
        self.skill = {r: {"ts": 0.0, "state": "", "skill": ""} for r in robots}
        self.agent = {r: {"ts": 0.0, "state": "", "msg": ""} for r in robots}
        self.planner_status = {}
        if not ROS_OK:
            return
        try:
            if not rclpy.ok():
                rclpy.init()
            self.node = Node("dashboard")
            self.chat_pub = self.node.create_publisher(String, "/chat_input", 10)
            self.node.create_subscription(String, "/chat_output", self._chat_cb, 10)
            self.node.create_subscription(String, "/robot_status", self._status_cb, 10)
            for r in robots:
                self.node.create_subscription(
                    String, f"/{r}/skill_status", lambda m, r=r: self._skill_cb(m, r), 10)
                self.node.create_subscription(
                    String, f"/{r}/agent_report", lambda m, r=r: self._agent_cb(m, r), 10)
            threading.Thread(target=self._spin, daemon=True).start()
            self.ok = True
        except Exception as e:  # noqa: BLE001
            print(f"[dashboard] ROS bridge disabled: {e!r}")

    def _spin(self):
        try:
            rclpy.spin(self.node)
        except Exception:  # noqa: BLE001
            pass

    def _chat_cb(self, msg):
        sender, _, text = msg.data.partition("|")
        if not text:
            sender, text = "system", msg.data
        item = {"type": "chat", "sender": sender.strip(), "text": text.strip(),
                "t": time.strftime("%H:%M:%S")}
        self.chat.append(item)
        self.hub.publish(item)

    def _status_cb(self, msg):
        try:
            self.planner_status = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def _skill_cb(self, msg, r):
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.skill[r] = {"ts": time.time(), "state": d.get("state", ""),
                         "skill": d.get("skill", ""), "holding": d.get("holding", False)}

    def _agent_cb(self, msg, r):
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.agent[r] = {"ts": time.time(), "state": d.get("state", ""),
                         "msg": d.get("msg", "")}

    def count_publishers(self, topic) -> int:
        if not self.ok:
            return 0
        try:
            return self.node.count_publishers(topic)
        except Exception:  # noqa: BLE001
            return 0

    def send_chat(self, text) -> bool:
        if not self.ok:
            return False
        self.chat_pub.publish(String(data=f"human|{text}"))
        item = {"type": "chat", "sender": "human", "text": text,
                "t": time.strftime("%H:%M:%S")}
        self.chat.append(item)
        self.hub.publish(item)
        return True

    def snapshot(self):
        now = time.time()
        robots = {}
        for r in self.robots:
            ps = self.planner_status.get(r, {})
            robots[r] = {
                "skill_alive": now - self.skill[r]["ts"] < ALIVE_SEC,
                "skill_state": self.skill[r]["state"],
                "agent_alive": now - self.agent[r]["ts"] < ALIVE_SEC,
                "holding": bool(self.skill[r].get("holding")),
                "planner_status": ps.get("status", "?"),
                "goal": ps.get("goal", ""),
            }
        return {"robots": robots,
                "topics": {t: self.count_publishers(t) for t in self.watch_topics}}

    def shutdown(self):
        if self.ok:
            try:
                self.node.destroy_node()
                rclpy.shutdown()
            except Exception:  # noqa: BLE001
                pass


# ============================================================ services
class Service:
    def __init__(self, name, d, setup):
        self.name = name
        self.group = d.get("group", "default")
        self.description = d.get("description", "")
        self.cmd = d["cmd"]
        self.cwd = os.path.expanduser(d["cwd"]) if d.get("cwd") else None
        self.env = {str(k): str(v) for k, v in (d.get("env") or {}).items()}
        self.depends_on = list(d.get("depends_on") or [])
        self.wait_sec = float(d.get("wait_sec", 0))
        self.ready_topic = d.get("ready_topic")
        self.ready_timeout = float(d.get("ready_timeout", 30))
        self.setup = setup
        self.proc = None
        self.started_at = 0.0
        self.exit_code = None
        self.stopped_by_us = False
        self.logs = collections.deque(maxlen=LOG_LINES)
        self.lock = threading.RLock()

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def status(self):
        if self.running:
            return "running"
        if self.proc is None:
            return "stopped"
        if self.exit_code in (0, None) or self.stopped_by_us:
            return "stopped"
        return "crashed"

    def info(self):
        return {"name": self.name, "group": self.group, "description": self.description,
                "status": self.status(), "pid": self.proc.pid if self.running else None,
                "uptime": int(time.time() - self.started_at) if self.running else 0,
                "exit_code": self.exit_code, "depends_on": self.depends_on,
                "cmd": self.cmd}


class ServiceManager:
    def __init__(self, cfg, hub, ros):
        self.hub, self.ros = hub, ros
        setup = cfg.get("setup") or []
        self.services = {}
        for name, d in (cfg.get("services") or {}).items():
            self.services[name] = Service(name, d, setup)
        for s in self.services.values():
            for dep in s.depends_on:
                if dep not in self.services:
                    raise ValueError(f"service '{s.name}' depends on unknown '{dep}'")
        self.groups = []
        for s in self.services.values():
            if s.group not in self.groups:
                self.groups.append(s.group)

    def get(self, name) -> Service:
        if name not in self.services:
            raise KeyError(name)
        return self.services[name]

    # ---- start / stop
    def _log(self, svc, line):
        svc.logs.append(line)
        self.hub.publish({"type": "log", "name": svc.name, "line": line})

    def _reader(self, svc, proc):
        for line in proc.stdout:
            self._log(svc, line.rstrip("\n"))
        proc.wait()
        svc.exit_code = proc.returncode
        self._log(svc, f"[dashboard] exited with code {proc.returncode}")
        self.hub.publish({"type": "svc", **svc.info()})

    def _spawn(self, svc):
        with svc.lock:
            if svc.running:
                return
            script = "; ".join([f"source {s}" for s in svc.setup] + [svc.cmd])
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env.update(svc.env)
            svc.exit_code, svc.stopped_by_us = None, False
            self._log(svc, f"[dashboard] starting: {svc.cmd}")
            svc.proc = subprocess.Popen(
                ["bash", "-c", script], cwd=svc.cwd, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                start_new_session=True)               # own process group
            svc.started_at = time.time()
            threading.Thread(target=self._reader, args=(svc, svc.proc), daemon=True).start()
        self.hub.publish({"type": "svc", **svc.info()})

    def _wait_ready(self, svc):
        if svc.wait_sec:
            time.sleep(svc.wait_sec)
        if svc.ready_topic and self.ros.ok:
            end = time.time() + svc.ready_timeout
            while time.time() < end and svc.running:
                if self.ros.count_publishers(svc.ready_topic) > 0:
                    return
                time.sleep(0.5)
            self._log(svc, f"[dashboard] ready_topic {svc.ready_topic} not seen "
                           f"after {svc.ready_timeout:.0f}s (continuing)")

    def start(self, name, _stack=()):
        svc = self.get(name)
        if name in _stack:
            raise ValueError(f"dependency cycle at {name}")
        for dep in svc.depends_on:
            d = self.get(dep)
            if not d.running:
                self.start(dep, _stack + (name,))
        if not svc.running:
            self._spawn(svc)
            self._wait_ready(svc)

    def stop(self, name):
        svc = self.get(name)
        with svc.lock:
            p = svc.proc
            if p is None or p.poll() is not None:
                return
            svc.stopped_by_us = True
            for sig, wait in ((signal.SIGINT, 3.0), (signal.SIGTERM, 2.0)):
                try:
                    os.killpg(p.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    p.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
            try:                                       # sweep survivors in the group
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
        self.hub.publish({"type": "svc", **svc.info()})

    def restart(self, name):
        self.stop(name)
        self.start(name)

    def start_group(self, group):
        for s in self.services.values():
            if s.group == group:
                self.start(s.name)

    def stop_group(self, group):
        for s in reversed(list(self.services.values())):
            if s.group == group:
                self.stop(s.name)

    def start_all(self):
        for s in self.services.values():          # yaml order; depends_on handled in start()
            self.start(s.name)

    def stop_all(self):
        for s in reversed(list(self.services.values())):
            self.stop(s.name)

    def state(self):
        return {"services": [s.info() for s in self.services.values()],
                "groups": self.groups}


# ============================================================ app
def create_app(config_path: str, static_dir: str = None) -> FastAPI:
    static = Path(static_dir).expanduser() if static_dir else STATIC
    if not (static / "index.html").exists():
        raise SystemExit(
            f"[dashboard] static folder not found: {static}\n"
            f"  Put the folder 'static' (index.html, style.css, app.js) next to this script,\n"
            f"  or run with  --static /path/to/static")
    cfg = yaml.safe_load(open(config_path)) or {}
    hub = Hub()
    ros = RosBridge(cfg.get("robots") or [], cfg.get("watch_topics") or [], hub)
    mgr = ServiceManager(cfg, hub, ros)
    token = os.environ.get("DASH_TOKEN")
    title = cfg.get("title", "Robot Dashboard")

    def auth(request: Request):
        if token and request.query_params.get("token") != token \
                and request.headers.get("x-token") != token:
            raise HTTPException(401, "bad token")

    def full_state():
        st = mgr.state()
        st.update(ros.snapshot())
        st.update({"ros": ros.ok, "title": title, "type": "state"})
        return st

    def bg(fn, *a):
        threading.Thread(target=fn, args=a, daemon=True).start()

    @asynccontextmanager
    async def lifespan(_app):
        hub.loop = asyncio.get_running_loop()

        async def ticker():
            while True:
                hub.publish(full_state())
                await asyncio.sleep(1.0)
        task = asyncio.create_task(ticker())
        for name in cfg.get("autostart") or []:
            bg(mgr.start, name)
        yield
        task.cancel()
        mgr.stop_all()
        ros.shutdown()

    app = FastAPI(title=title, lifespan=lifespan)
    app.state.mgr, app.state.ros, app.state.hub = mgr, ros, hub

    def need(name):
        try:
            return mgr.get(name)
        except KeyError:
            raise HTTPException(404, f"unknown service {name}") from None

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/api/state", dependencies=[Depends(auth)])
    def api_state():
        return full_state()

    @app.post("/api/services/{name}/{action}", dependencies=[Depends(auth)])
    def api_service(name: str, action: str):
        need(name)
        fn = {"start": mgr.start, "stop": mgr.stop, "restart": mgr.restart}.get(action)
        if fn is None:
            raise HTTPException(400, "action must be start|stop|restart")
        bg(fn, name)
        return {"ok": True}

    @app.post("/api/groups/{group}/{action}", dependencies=[Depends(auth)])
    def api_group(group: str, action: str):
        if group not in mgr.groups:
            raise HTTPException(404, f"unknown group {group}")
        fn = {"start": mgr.start_group, "stop": mgr.stop_group}.get(action)
        if fn is None:
            raise HTTPException(400, "action must be start|stop")
        bg(fn, group)
        return {"ok": True}

    @app.post("/api/start_all_services", dependencies=[Depends(auth)])
    def api_start_all_services():
        bg(mgr.start_all)
        return {"ok": True}

    @app.post("/api/stop_all_services", dependencies=[Depends(auth)])
    def api_stop_all_services():
        bg(mgr.stop_all)
        return {"ok": True}

    @app.get("/api/logs/{name}", dependencies=[Depends(auth)])
    def api_logs(name: str, n: int = 300):
        return {"name": name, "lines": list(need(name).logs)[-n:]}

    @app.post("/api/chat", dependencies=[Depends(auth)])
    def api_chat(body: dict):
        text = str(body.get("text", "")).strip()
        if not text:
            raise HTTPException(400, "empty text")
        if not ros.send_chat(text):
            raise HTTPException(503, "ROS bridge not available")
        return {"ok": True}

    @app.post("/api/robots/stop", dependencies=[Depends(auth)])
    def api_robots_stop(body: dict = None):
        robot = (body or {}).get("robot")
        if not ros.send_chat(f"stop {robot}" if robot else "stop"):
            raise HTTPException(503, "ROS bridge not available")
        return {"ok": True}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        if token and ws.query_params.get("token") != token:
            await ws.close(code=4401)
            return
        await ws.accept()
        q = asyncio.Queue(maxsize=3000)
        hub.clients.add(q)
        await ws.send_json(full_state())
        for item in list(ros.chat):
            await ws.send_json(item)

        async def sender():
            while True:
                await ws.send_json(await q.get())

        async def receiver():
            while True:
                await ws.receive_text()

        tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except WebSocketDisconnect:
            pass
        finally:
            for t in tasks:
                t.cancel()
            hub.clients.discard(q)

    app.mount("/static", StaticFiles(directory=static), name="static")
    return app


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="services.yaml")
    ap.add_argument("--host", default="127.0.0.1",
                    help="use 0.0.0.0 to open from another PC (set DASH_TOKEN!)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--static", default=None, help="folder with index.html, style.css, app.js")
    a = ap.parse_args()
    uvicorn.run(create_app(a.config, a.static), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
