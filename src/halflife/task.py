"""Tasks: what the agent sees, and what the grader checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .sandbox import Sandbox


@dataclass
class Checkpoint:
    """One verifiable step of a task, in the order the work should happen.

    check(sandbox) -> True when the step was done correctly.
    diagnose(sandbox) -> optional failure category, for finer failure modes.
    """

    name: str
    category: str
    check: Callable[[Any], bool]
    diagnose: Optional[Callable[[Any], str]] = None


@dataclass
class Constraint:
    """A rule that must hold for the whole run. check(sandbox) -> True if respected."""

    name: str
    check: Callable[[Any], bool]


@dataclass
class TaskSpec:
    """What a suite builds for one task: instructions, a sandbox, and the answer key."""

    instructions: str
    sandbox: Sandbox
    checkpoints: List[Checkpoint]
    constraints: List[Constraint] = field(default_factory=list)
    max_tool_calls: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Task:
    """The object your agent receives.

    task.instructions        the goal and the policy, as plain text
    task.call(name, args)    call a tool; returns a dict (errors come back as {"error": ...})
    task.tools               {name: callable} for keyword-style calls
    task.anthropic_tools()   tool definitions for the Anthropic Messages API
    task.openai_tools()      tool definitions for OpenAI-style function calling
    task.max_tool_calls      the budget; exceeding it ends the run
    """

    def __init__(self, id: str, suite: str, length: int, instructions: str, sandbox: Sandbox):
        self.id = id
        self.suite = suite
        self.length = length
        self.instructions = instructions
        self._sandbox = sandbox
        self._specs = sandbox.tool_specs()
        self.tools: Dict[str, Callable[..., Any]] = {s["name"]: self._bind(s["name"]) for s in self._specs}
        self.usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "model_calls": 0}

    def record_usage(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Agents call this after each model call so reports can show tokens and cost."""
        self.usage["input_tokens"] += int(input_tokens or 0)
        self.usage["output_tokens"] += int(output_tokens or 0)
        self.usage["model_calls"] += 1

    def _bind(self, name: str) -> Callable[..., Any]:
        def call(**kwargs: Any) -> Any:
            return self._sandbox.call(name, kwargs)

        call.__name__ = name
        return call

    def call(self, name: str, args: Optional[dict] = None, **kwargs: Any) -> Any:
        merged = dict(args or {})
        merged.update(kwargs)
        return self._sandbox.call(name, merged)

    @property
    def max_tool_calls(self) -> Optional[int]:
        return self._sandbox.max_calls

    @property
    def tool_calls(self) -> int:
        return len(self._sandbox.trace)

    def tool_specs(self) -> List[dict]:
        return [dict(s) for s in self._specs]

    def anthropic_tools(self) -> List[dict]:
        return [{"name": s["name"], "description": s["description"], "input_schema": s["parameters"]} for s in self._specs]

    def openai_tools(self) -> List[dict]:
        return [
            {"type": "function", "function": {"name": s["name"], "description": s["description"], "parameters": s["parameters"]}}
            for s in self._specs
        ]

    def __repr__(self) -> str:
        return f"<Task {self.id} length={self.length} tools={len(self._specs)}>"
