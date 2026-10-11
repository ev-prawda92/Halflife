"""Simulated systems that agents act on.

A Sandbox is a fake version of a real system (a claims platform, an AP ledger,
a ticket queue). Its methods marked with @tool are what the agent can call.
Every call is recorded, and after the run the sandbox's final state is graded
against an answer key. Agents never see the answer key.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional


class ToolError(Exception):
    """Raise inside a tool to send an error back to the agent. The run continues."""


class ToolBudgetExceeded(Exception):
    """Raised when an agent exceeds its tool-call budget. Ends the run."""


class param:
    """JSON-schema builders for tool parameters."""

    @staticmethod
    def _base(kind: str, description: str, optional: bool, enum=None) -> dict:
        schema: Dict[str, Any] = {"type": kind, "description": description}
        if enum is not None:
            schema["enum"] = list(enum)
        if optional:
            schema["x-optional"] = True
        return schema

    @staticmethod
    def string(description: str, *, enum=None, optional: bool = False) -> dict:
        return param._base("string", description, optional, enum)

    @staticmethod
    def number(description: str, *, optional: bool = False) -> dict:
        return param._base("number", description, optional)

    @staticmethod
    def integer(description: str, *, optional: bool = False) -> dict:
        return param._base("integer", description, optional)

    @staticmethod
    def boolean(description: str, *, optional: bool = False) -> dict:
        return param._base("boolean", description, optional)


def tool(description: str, **params: dict) -> Callable:
    """Mark a Sandbox method as a tool the agent can call.

    @tool("Look up a member", member_id=param.string("Member ID"))
    def get_member(self, member_id): ...
    """

    def deco(fn: Callable) -> Callable:
        fn.__halflife_tool__ = (description, params)
        return fn

    return deco


class Sandbox:
    """Base class for simulated systems. Subclasses define tools with @tool.

    Calling super().__init__() is optional.
    """

    @property
    def trace(self) -> List[dict]:
        return self.__dict__.setdefault("_hl_trace", [])

    @property
    def max_calls(self) -> Optional[int]:
        return self.__dict__.get("_hl_max_calls")

    @max_calls.setter
    def max_calls(self, value: Optional[int]) -> None:
        self.__dict__["_hl_max_calls"] = value

    @classmethod
    def _tool_methods(cls) -> Dict[str, Callable]:
        found: Dict[str, Callable] = {}
        for klass in reversed(cls.__mro__):
            for name, attr in vars(klass).items():
                if hasattr(attr, "__halflife_tool__"):
                    found[name] = attr
        return found

    def tool_specs(self) -> List[dict]:
        specs = []
        for name, fn in self._tool_methods().items():
            description, params = fn.__halflife_tool__
            properties = {}
            required = []
            for key, schema in params.items():
                clean = {k: v for k, v in schema.items() if k != "x-optional"}
                properties[key] = clean
                if not schema.get("x-optional"):
                    required.append(key)
            specs.append(
                {
                    "name": name,
                    "description": description,
                    "parameters": {"type": "object", "properties": properties, "required": required},
                }
            )
        return specs

    def call(self, name: str, args: Optional[dict] = None) -> Any:
        args = dict(args or {})
        limit = self.max_calls
        if limit is not None and len(self.trace) >= limit:
            raise ToolBudgetExceeded(f"tool-call budget of {limit} exceeded")

        entry: Dict[str, Any] = {"step": len(self.trace) + 1, "tool": name, "args": copy.deepcopy(args)}
        self.trace.append(entry)

        methods = self._tool_methods()
        if name not in methods:
            entry["error"] = f"unknown tool '{name}'. Available: {', '.join(methods)}"
            return {"error": entry["error"]}

        fn = methods[name]
        _, params = fn.__halflife_tool__
        unknown = sorted(set(args) - set(params))
        missing = [k for k, s in params.items() if not s.get("x-optional") and k not in args]
        if unknown or missing:
            parts = []
            if missing:
                parts.append(f"missing argument(s): {', '.join(missing)}")
            if unknown:
                parts.append(f"unknown argument(s): {', '.join(unknown)}")
            entry["error"] = "; ".join(parts)
            return {"error": entry["error"]}

        try:
            result = fn(self, **args)
        except ToolError as exc:
            entry["error"] = str(exc)
            return {"error": str(exc)}
        except (TypeError, ValueError) as exc:
            entry["error"] = f"invalid arguments: {exc}"
            return {"error": entry["error"]}

        result = copy.deepcopy(result)
        entry["result"] = result
        return result
