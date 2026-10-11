"""A tool-use agent for the Anthropic Messages API, with prompt caching.

Caching matters for long tasks: every step resends the whole conversation, and
cached input is billed at a fraction of the normal rate.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from ..task import Task
from ._http import ModelAPIError, post_json

SYSTEM = "You complete operational tasks by calling the provided tools. Follow the policy exactly. Keep working until every item is handled, then reply with a one-line summary and no tool calls."


class AnthropicAgent:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        timeout: float = 600,
        base_url: str = "https://api.anthropic.com",
        name: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self.__name__ = name or model

    def __call__(self, task: Task) -> Optional[str]:
        return self.run(task)

    @staticmethod
    def _mark_cache(messages: list) -> None:
        for m in messages:
            if isinstance(m["content"], list):
                for block in m["content"]:
                    block.pop("cache_control", None)
        last = messages[-1]
        if isinstance(last["content"], str):
            last["content"] = [{"type": "text", "text": last["content"]}]
        last["content"][-1]["cache_control"] = {"type": "ephemeral"}

    def run(self, task: Task) -> Optional[str]:
        if not self.api_key:
            raise ModelAPIError("ANTHROPIC_API_KEY is not set")
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        tools = task.anthropic_tools()
        if tools:
            tools[-1] = dict(tools[-1], cache_control={"type": "ephemeral"})
        messages: list = [{"role": "user", "content": task.instructions}]
        while True:
            self._mark_cache(messages)
            payload = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": [{"type": "text", "text": SYSTEM}],
                "tools": tools,
                "messages": messages,
            }
            data = post_json(f"{self.base_url}/v1/messages", payload, headers, self.timeout)
            usage = data.get("usage") or {}
            task.record_usage(
                usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0),
                usage.get("output_tokens", 0),
            )
            content = data.get("content", [])
            messages.append({"role": "assistant", "content": content})
            calls = [b for b in content if b.get("type") == "tool_use"]
            if data.get("stop_reason") != "tool_use" or not calls:
                return " ".join(b.get("text", "") for b in content if b.get("type") == "text")
            results = []
            for block in calls:
                out = task.call(block["name"], block.get("input") or {})
                results.append({"type": "tool_result", "tool_use_id": block["id"], "content": json.dumps(out)})
            messages.append({"role": "user", "content": results})
