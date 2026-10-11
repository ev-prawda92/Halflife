"""A tool-use agent for any OpenAI-compatible chat API.

Works with Ollama, vLLM, LM Studio, llama.cpp server, OpenRouter, Together,
Groq, and OpenAI itself: anything that serves /v1/chat/completions with tools.
"""

from __future__ import annotations

import json
from typing import Optional

from ..task import Task
from ._http import post_json


class OpenAICompatibleAgent:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 600,
        name: Optional[str] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.__name__ = name or model

    def __call__(self, task: Task) -> Optional[str]:
        return self.run(task)

    def run(self, task: Task) -> Optional[str]:
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        messages = [
            {"role": "system", "content": "You complete operational tasks by calling the provided tools. Follow the policy exactly. Keep working until every item is handled, then reply with a one-line summary and no tool calls."},
            {"role": "user", "content": task.instructions},
        ]
        tools = task.openai_tools()
        while True:
            payload = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            data = post_json(f"{self.base_url}/chat/completions", payload, headers, self.timeout)
            usage = data.get("usage") or {}
            task.record_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            msg = data["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            assistant = {"role": "assistant", "content": msg.get("content") or ""}
            if calls:
                assistant["tool_calls"] = calls
            messages.append(assistant)
            if not calls:
                return msg.get("content")
            for call in calls:
                fn = call.get("function", {})
                raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                    result = task.call(fn.get("name", ""), args)
                except json.JSONDecodeError:
                    result = {"error": f"arguments were not valid JSON: {raw[:200]}"}
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps(result)})
