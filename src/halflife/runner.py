"""Runs an agent across tasks of increasing length and grades every run."""

from __future__ import annotations

import hashlib
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional, Sequence

from .report import Report
from .results import RunResult
from .sandbox import ToolBudgetExceeded
from .suite import Suite, get_suite
from .task import Task, TaskSpec

TRACE_TAIL = 12


def _task_seed(seed: int, suite: str, length: int, index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{suite}:{length}:{index}".encode()).hexdigest()
    return int(digest[:12], 16)


def build_task(suite: Suite, length: int, index: int = 0, seed: int = 0) -> TaskSpec:
    """Build one task deterministically. Same arguments, same task."""
    return suite.build(length, random.Random(_task_seed(seed, suite.name, length, index)))


def grade(spec: TaskSpec, error: Optional[str] = None) -> dict:
    sandbox = spec.sandbox
    first: Optional[dict] = None
    passed_count = 0
    for i, cp in enumerate(spec.checkpoints):
        try:
            ok = bool(cp.check(sandbox))
        except Exception:
            ok = False
        if ok:
            passed_count += 1
        elif first is None:
            category = cp.category
            if cp.diagnose is not None:
                try:
                    category = cp.diagnose(sandbox) or cp.category
                except Exception:
                    pass
            first = {"index": i, "name": cp.name, "category": category}

    violations = []
    for c in spec.constraints:
        try:
            if not c.check(sandbox):
                violations.append(c.name)
        except Exception:
            violations.append(c.name)

    length = len(spec.checkpoints)
    if error is not None:
        kind = "budget_exceeded" if "budget" in error else "agent_error"
        first = {"index": first["index"] if first else length, "name": error, "category": kind}
    elif first is None and violations:
        first = {"index": length, "name": violations[0], "category": "constraint_violation"}

    return {
        "passed": first is None,
        "checkpoints_passed": passed_count,
        "progress": first["index"] if first else length,
        "first_failure": first,
        "violations": violations,
    }


def run_once(
    agent: Callable[[Task], Any],
    suite: Suite,
    length: int,
    index: int,
    run: int,
    seed: int = 0,
    rules: Optional[Sequence[Any]] = None,
    trace_writer: Any = None,
) -> RunResult:
    spec = build_task(suite, length, index, seed)
    sandbox = spec.sandbox
    n_checkpoints = len(spec.checkpoints)
    sandbox.max_calls = spec.max_tool_calls or max(60, 10 * n_checkpoints)
    task = Task(f"{suite.name}/L{length}/t{index}", suite.name, n_checkpoints, spec.instructions, sandbox)

    error = None
    started = time.perf_counter()
    try:
        agent(task)
    except ToolBudgetExceeded as exc:
        error = str(exc)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - started

    graded = grade(spec, error)
    if rules:
        from .rules import check as check_rules

        broken = check_rules(rules, sandbox.trace)
        if broken:
            graded["violations"] = graded["violations"] + [f"{v.rule}: {v.message}" for v in broken]
            if graded["first_failure"] is None:
                v = broken[0]
                graded["first_failure"] = {"index": min(v.step, n_checkpoints), "name": f"{v.rule}: {v.message}", "category": v.category}
                graded["passed"] = False
    if trace_writer is not None:
        trace_writer.run(f"{task.id}/r{run}", sandbox.trace)
    tail = [] if graded["passed"] else sandbox.trace[-TRACE_TAIL:]
    return RunResult(
        task_id=task.id,
        requested_length=length,
        length=n_checkpoints,
        run=run,
        error=error,
        tool_calls=len(sandbox.trace),
        seconds=round(seconds, 4),
        trace_tail=tail,
        task_index=index,
        input_tokens=task.usage["input_tokens"],
        output_tokens=task.usage["output_tokens"],
        model_calls=task.usage["model_calls"],
        **graded,
    )


def evaluate(
    agent: Callable[[Task], Any],
    suite: str = "claims-intake",
    lengths: Optional[Sequence[int]] = None,
    tasks: int = 3,
    runs: int = 5,
    seed: int = 0,
    workers: int = 1,
    agent_name: Optional[str] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    cases: Optional[Sequence[Sequence[int]]] = None,
    workflow_length: Optional[int] = None,
    rules: Optional[Sequence[Any]] = None,
    save_traces: Optional[str] = None,
) -> Report:
    """Benchmark an agent and return a Report.

    lengths: task lengths to test, in checkpoints (verifiable steps)
    tasks:   distinct tasks per length
    runs:    repeated runs per task (agents are not deterministic)
    cases:   replay only these (length, task_index) pairs, e.g. from an eval pack
    workflow_length: your real workflow's length, for an autonomy recommendation
    """
    s = get_suite(suite)
    if cases:
        unique = sorted({(int(c[0]), int(c[1])) for c in cases})
        lengths = sorted({c[0] for c in unique})
        jobs = [(length, i, r) for (length, i) in unique for r in range(runs)]
    else:
        lengths = list(lengths or s.default_lengths)
        jobs = [(length, i, r) for length in lengths for i in range(tasks) for r in range(runs)]
    total = len(jobs)
    results: List[RunResult] = []
    writer = None
    if save_traces:
        from .traces import TraceWriter

        writer = TraceWriter(save_traces, mode="w")

    if workers <= 1:
        for done, (length, i, r) in enumerate(jobs, 1):
            results.append(run_once(agent, s, length, i, r, seed, rules, writer))
            if progress:
                progress(done, total)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_once, agent, s, length, i, r, seed, rules, writer) for (length, i, r) in jobs]
            for done, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                if progress:
                    progress(done, total)
        results.sort(key=lambda x: (x.requested_length, x.task_id, x.run))

    if writer is not None:
        writer.close()
    config = {"suite": suite, "lengths": lengths, "tasks": tasks, "runs": runs, "seed": seed, "workflow_length": workflow_length, "replay": bool(cases)}
    name = agent_name or getattr(agent, "__name__", "agent")
    return Report.build(name, suite, config, results)
