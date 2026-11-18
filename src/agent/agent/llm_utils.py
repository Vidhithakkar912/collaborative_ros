"""Small LLM helpers shared by the central planner and the per-robot agents.

Both LLMs expose the same two methods so they can be swapped / faked in tests:
    ask_json(system, user, max_tokens) -> dict | list | None   (None = bad / no output)
    chat(user, system, max_tokens)     -> str
"""

import json
import re
import threading


def extract_json(text):
    """First balanced JSON object/array in `text` (tolerates ```json fences and chatter)."""
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text).strip()
    starts = [i for i in (t.find("{"), t.find("[")) if i != -1]
    if not starts:
        return None
    start = min(starts)
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class LocalLLM:
    """llama.cpp model (GGUF). One lock -> safe to call from several threads."""

    def __init__(self, model_path, n_ctx=2048, n_threads=8, logger=None):
        from llama_cpp import Llama
        self._log = logger
        self._lock = threading.Lock()
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                          n_gpu_layers=0, verbose=False)

    def _chat(self, system, user, max_tokens, temperature, json_mode=False):
        kwargs = dict(messages=[{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                      max_tokens=max_tokens, temperature=temperature)
        with self._lock:
            if json_mode:
                try:
                    out = self._llm.create_chat_completion(
                        response_format={"type": "json_object"}, **kwargs)
                except Exception:  # noqa: BLE001  (older builds: no response_format)
                    out = self._llm.create_chat_completion(**kwargs)
            else:
                out = self._llm.create_chat_completion(**kwargs)
        return out["choices"][0]["message"]["content"].strip()

    def chat(self, user, system="You are CoMuRoS, a friendly robot assistant. "
                               "Keep answers short.", max_tokens=100):
        return self._chat(system, user, max_tokens, temperature=0.7)

    def ask_json(self, system, user, max_tokens=256):
        try:
            return extract_json(self._chat(system, user, max_tokens, 0.0, json_mode=True))
        except Exception as e:  # noqa: BLE001
            if self._log:
                self._log.error(f"LLM ask_json failed: {e!r}")
            return None


class FakeLLM:
    """Scripted replies for tests / dry runs. replies: list of dict | None."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def ask_json(self, system, user, max_tokens=256):
        self.prompts.append(user)
        return self.replies.pop(0) if self.replies else None

    def chat(self, user, system="", max_tokens=100):
        return "fake reply"