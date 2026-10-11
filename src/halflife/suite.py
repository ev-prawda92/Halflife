"""Suite registry. A suite builds tasks of a requested length from a seeded RNG."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from .task import TaskSpec

DEFAULT_LENGTHS: Tuple[int, ...] = (5, 10, 20, 40)

_REGISTRY: Dict[str, "Suite"] = {}


@dataclass
class Suite:
    name: str
    description: str
    build: Callable[[int, random.Random], TaskSpec]
    default_lengths: Tuple[int, ...] = DEFAULT_LENGTHS


def suite(name: str, description: str = "", default_lengths=DEFAULT_LENGTHS) -> Callable:
    """Register a task suite.

    @halflife.suite("ticket-routing")
    def build(length: int, rng: random.Random) -> TaskSpec: ...
    """

    def deco(fn: Callable[[int, random.Random], TaskSpec]) -> Callable:
        doc = (fn.__doc__ or "").strip().splitlines()
        _REGISTRY[name] = Suite(name, description or (doc[0] if doc else ""), fn, tuple(default_lengths))
        return fn

    return deco


def _load_builtins() -> None:
    from . import suites  # noqa: F401


def get_suite(name: str) -> Suite:
    _load_builtins()
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(f"unknown suite '{name}'. Available suites: {available}")
    return _REGISTRY[name]


def list_suites() -> List[Suite]:
    _load_builtins()
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]
