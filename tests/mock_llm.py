"""A local stand-in for a model server, speaking the real OpenAI and Anthropic wire formats.

Behind the HTTP API, a scripted policy agent decides each tool call. Its calls are
bridged over the network one at a time, so the built-in agents exercise the full
loop: tool definitions out, tool calls back, tool results in, many turns, usage.
"""

from __future__ import annotations

import json
import queue
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

DONE = object()


class _Bridge:
    def __init__(self, make_agent: Callable):
        self.make_agent = make_agent
        self.to_model: "queue.Queue" = queue.Queue()
        self.to_server: "queue.Queue" = queue.Queue()
        self.tool_names: set = set()

    def start(self) -> None:
        self.to_model = queue.Queue()
        self.to_server = queue.Queue()
        bridge = self

        class ProxyTask:
            def call(self, name, args=None, **kwargs):
                merged = dict(args or {})
                merged.update(kwargs)
                bridge.to_server.put((name, merged))
                return bridge.to_model.get()

        def run():
            try:
                self.make_agent().run(ProxyTask())
            finally:
                self.to_server.put(DONE)

        threading.Thread(target=run, daemon=True).start()

    def next_action(self, tool_result=None):
        if tool_result is not None:
            self.to_model.put(tool_result)
        return self.to_server.get(timeout=30)


class MockModelServer:
    def __init__(self, make_agent: Callable):
        self.bridge = _Bridge(make_agent)
        self.requests = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                server.requests += 1
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                if self.path.endswith("/chat/completions"):
                    out = server._openai(body)
                elif self.path.endswith("/v1/messages"):
                    if self.headers.get("x-api-key") != "test-key":
                        self.send_response(401)
                        self.end_headers()
                        return
                    out = server._anthropic(body)
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()

    def _openai(self, body: dict) -> dict:
        msgs = body["messages"]
        assert body["tools"] and all(t["type"] == "function" for t in body["tools"])
        if msgs[-1]["role"] == "user":
            self.bridge.start()
            action = self.bridge.next_action()
        else:
            assert msgs[-1]["role"] == "tool"
            action = self.bridge.next_action(json.loads(msgs[-1]["content"]))
        usage = {"prompt_tokens": 100 * len(msgs), "completion_tokens": 20}
        if action is DONE:
            message = {"role": "assistant", "content": "All items handled."}
        else:
            name, args = action
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_" + uuid.uuid4().hex[:8], "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}],
            }
        return {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}], "usage": usage}

    def _anthropic(self, body: dict) -> dict:
        msgs = body["messages"]
        assert body["tools"] and "input_schema" in body["tools"][0]
        last = msgs[-1]
        if len(msgs) == 1:
            self.bridge.start()
            action = self.bridge.next_action()
        else:
            block = last["content"][-1]
            assert block["type"] == "tool_result"
            action = self.bridge.next_action(json.loads(block["content"]))
        usage = {"input_tokens": 10, "cache_read_input_tokens": 90 * len(msgs), "output_tokens": 20}
        if action is DONE:
            return {"content": [{"type": "text", "text": "All items handled."}], "stop_reason": "end_turn", "usage": usage}
        name, args = action
        return {
            "content": [{"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:8], "name": name, "input": args}],
            "stop_reason": "tool_use",
            "usage": usage,
        }
