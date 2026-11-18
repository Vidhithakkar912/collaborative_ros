"""Start / kill external ROS nodes reliably.

WHY YOUR VISION KEPT RUNNING: `ros2 run pkg exe` starts a wrapper process which starts the
real python node as a CHILD. proc.terminate() only signals the wrapper, so the child
(your YOLO loop) lives on. Fix: start every process in its OWN process group
(start_new_session=True) and signal the WHOLE group with os.killpg().
"""

import os
import signal
import subprocess
import threading


class ProcessManager:
    def __init__(self, logger):
        self._log = logger
        self._procs = {}
        self._lock = threading.Lock()

    def is_running(self, key) -> bool:
        with self._lock:
            p = self._procs.get(key)
        return p is not None and p.poll() is None

    def start(self, key, cmd) -> bool:
        """Returns True if started, False if it was already running."""
        if self.is_running(key):
            self._log.info(f"[proc] {key} already running")
            return False
        self._log.info(f"[proc] starting {key}: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, start_new_session=True)   # own process group
        with self._lock:
            self._procs[key] = proc
        return True

    def stop(self, key, grace=2.0) -> bool:
        """SIGINT (clean) -> SIGTERM -> SIGKILL on the whole group. Blocks <= ~6 s."""
        with self._lock:
            proc = self._procs.pop(key, None)
        if proc is None:
            return False
        self._log.info(f"[proc] stopping {key} (pgid {proc.pid})")
        for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 2.0)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                proc.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue
        try:                                   # sweep any survivor in the group
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        return True

    def stop_all(self):
        with self._lock:
            keys = list(self._procs)
        for k in keys:
            self.stop(k)