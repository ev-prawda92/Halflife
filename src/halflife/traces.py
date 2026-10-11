"""Reading and writing agent traces, and auditing them against rules.

Accepted formats (JSON Lines, one object per line):

  one line per tool call, grouped by a run id:
    {"run_id": "r1", "tool": "get_claim", "args": {"claim_id": "C1"}, "result": {...}}

  one line per run:
    {"run_id": "r1", "steps": [{"tool": ..., "args": ..., "result": ...}, ...]}

Field names are flexible: tool / name / tool_name / function, and
args / arguments / input / parameters (a JSON string is fine). The run id can be
run_id, trace_id, session_id or conversation_id.
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import rules as rules_mod
from .report import Report
from .results import RunResult

TOOL_KEYS = ("tool", "name", "tool_name", "function")
ARG_KEYS = ("args", "arguments", "input", "parameters")
RUN_KEYS = ("run_id", "trace_id", "session_id", "conversation_id", "task_id")


def _norm_step(raw: dict) -> dict:
    tool = next((raw[k] for k in TOOL_KEYS if k in raw and isinstance(raw[k], str)), None)
    if tool is None and isinstance(raw.get("function"), dict):
        tool = raw["function"].get("name")
        raw = dict(raw, arguments=raw["function"].get("arguments"))
    args: Any = next((raw[k] for k in ARG_KEYS if k in raw), {}) or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw": args}
    step = {"tool": tool or "?", "args": args if isinstance(args, dict) else {"_value": args}}
    if "result" in raw or "output" in raw:
        step["result"] = raw.get("result", raw.get("output"))
    err = raw.get("error")
    if not err and isinstance(step.get("result"), dict) and step["result"].get("error"):
        err = step["result"]["error"]
    if err:
        step["error"] = err
    return step


def read(path: str) -> "OrderedDict[str, List[dict]]":
    runs: "OrderedDict[str, List[dict]]" = OrderedDict()
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"halflife: {path} line {n} is not valid JSON ({exc.msg})")
            run_id = str(next((obj[k] for k in RUN_KEYS if k in obj), "run-1"))
            if isinstance(obj.get("steps"), list):
                runs.setdefault(run_id, []).extend(_norm_step(s) for s in obj["steps"])
            else:
                runs.setdefault(run_id, []).append(_norm_step(obj))
    return runs


class TraceWriter:
    """Record your agent's tool calls in the format `halflife audit` reads.

    with halflife.TraceWriter("traces.jsonl") as log:
        log.step(run_id, "get_claim", {"claim_id": "C1"}, result)
    """

    def __init__(self, path: str, mode: str = "a"):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        self._f = open(path, mode)
        self._lock = threading.Lock()

    def step(self, run_id: str, tool: str, args: Optional[dict] = None, result: Any = None, error: Optional[str] = None) -> None:
        rec = {"run_id": run_id, "tool": tool, "args": args or {}}
        if result is not None:
            rec["result"] = result
        if error:
            rec["error"] = error
        with self._lock:
            self._f.write(json.dumps(rec, default=str) + "\n")

    def run(self, run_id: str, steps: Sequence[dict]) -> None:
        with self._lock:
            self._f.write(json.dumps({"run_id": run_id, "steps": list(steps)}, default=str) + "\n")

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _buckets(lengths: List[int], n: int = 5) -> List[int]:
    """Upper bounds of up to n equal-count length buckets."""
    xs = sorted(lengths)
    if not xs:
        return []
    bounds = sorted({xs[min(len(xs) - 1, int(len(xs) * (i + 1) / n) - 1)] for i in range(n)})
    return bounds


def audit(
    path: str,
    rules: Sequence[rules_mod.Rule],
    name: Optional[str] = None,
    include_defaults: bool = True,
    workflow_length: Optional[int] = None,
) -> Report:
    """Check every run in a trace file against rules and fit a half-life over run length."""
    runs = read(path)
    if not runs:
        raise SystemExit(f"halflife: no runs found in {path}")
    all_rules = list(rules) + (rules_mod.default_rules() if include_defaults else [])
    lengths = [max(1, len(s)) for s in runs.values()]
    bounds = _buckets(lengths)
    results: List[RunResult] = []
    violation_counts: Dict[str, int] = {}
    for run_id, steps in runs.items():
        violations = rules_mod.check(all_rules, steps)
        for v in violations:
            violation_counts[v.rule] = violation_counts.get(v.rule, 0) + 1
        length = max(1, len(steps))
        bucket = next(b for b in bounds if length <= b)
        first = violations[0] if violations else None
        results.append(
            RunResult(
                task_id=run_id,
                requested_length=bucket,
                length=length,
                run=0,
                passed=not violations,
                checkpoints_passed=length if not violations else first.step,
                progress=length if not violations else min(first.step, length - 1),
                first_failure={"index": first.step, "name": f"{first.rule}: {first.message}", "category": first.category} if first else None,
                violations=[f"{v.rule}: {v.message}" for v in violations],
                error=None,
                tool_calls=len(steps),
                seconds=0.0,
                trace_tail=[] if not violations else steps[max(0, first.step - 6): first.step + 1],
            )
        )
    config = {
        "mode": "audit",
        "source": os.path.basename(path),
        "lengths": bounds,
        "unit": "tool calls",
        "rules": [r.name for r in all_rules],
        "violations_by_rule": violation_counts,
        "seed": 0,
        "workflow_length": workflow_length,
    }
    return Report.build(name or os.path.splitext(os.path.basename(path))[0], f"audit:{os.path.basename(path)}", config, results)
