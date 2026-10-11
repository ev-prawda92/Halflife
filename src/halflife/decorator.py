"""The @benchmark decorator."""

from __future__ import annotations

import functools
from dataclasses import asdict, dataclass
from typing import Any, Callable, List, Optional, Sequence

from .report import Report

_REGISTERED: List[Callable[..., Any]] = []


@dataclass
class BenchmarkConfig:
    suite: str
    lengths: Optional[Sequence[int]] = None
    tasks: int = 3
    runs: int = 5
    seed: int = 0
    workers: int = 1


def benchmark(
    fn: Optional[Callable] = None,
    *,
    suite: str = "claims-intake",
    lengths: Optional[Sequence[int]] = None,
    tasks: int = 3,
    runs: int = 5,
    seed: int = 0,
    workers: int = 1,
) -> Callable:
    """Mark an agent function for benchmarking. The function is otherwise unchanged.

    @benchmark(suite="claims-intake")
    def my_agent(task):
        return agent.run(task)

    my_agent.evaluate()           run it from Python, returns a Report
    halflife run my_agent.py      run it from the command line
    """

    def deco(f: Callable) -> Callable:
        cfg = BenchmarkConfig(suite, lengths, tasks, runs, seed, workers)

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return f(*args, **kwargs)

        def evaluate(**overrides: Any) -> Report:
            from .runner import evaluate as _evaluate

            params = asdict(cfg)
            params.update({k: v for k, v in overrides.items() if v is not None})
            params.setdefault("agent_name", f.__name__)
            return _evaluate(f, **params)

        wrapper.__halflife__ = cfg
        wrapper.evaluate = evaluate
        _REGISTERED.append(wrapper)
        return wrapper

    if fn is not None and callable(fn):
        return deco(fn)
    return deco


def registered() -> List[Callable[..., Any]]:
    return list(_REGISTERED)


def assert_reliable(agent: Callable, min_half_life: float, quick: bool = True, **kwargs: Any) -> Report:
    """Fail a test if the agent's half-life is below a threshold. Returns the report.

    def test_agent_is_reliable():
        halflife.assert_reliable(my_agent, min_half_life=15)

    quick=True runs lengths 5,10,20 with 2 tasks and 2 runs, fast enough for every test run.
    The check is against the half-life backed by tested lengths, so test lengths that reach the threshold.
    """
    if quick:
        kwargs.setdefault("lengths", [5, 10, 20])
        kwargs.setdefault("tasks", 2)
        kwargs.setdefault("runs", 2)
    if hasattr(agent, "evaluate"):
        report = agent.evaluate(**kwargs)
    else:
        from .runner import evaluate

        report = evaluate(agent, **kwargs)
    if not report.passes(min_half_life):
        raise AssertionError(f"reliability half-life below {min_half_life:g}\n\n{report.summary()}")
    return report
