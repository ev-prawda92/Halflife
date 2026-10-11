"""MCP server: lets coding agents (Claude Code, Cursor, and others) test and fix your agent.

    claude mcp add halflife -- halflife mcp

The loop it enables: run a benchmark → read what broke and how to fix it → edit
the agent's code → replay exactly the failing cases → repeat until it holds up.

Speaks MCP over stdio (newline-delimited JSON-RPC 2.0). Standard library only.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional

from . import __version__

SERVER_INSTRUCTIONS = (
    "halflife measures how long a task an AI agent can be trusted with. "
    "Typical loop: run_benchmark (or bench_model / audit_traces) → read the summary, failure modes and fixes → "
    "change the agent's code → replay_failures with the eval pack path from the last run → repeat."
)


def _report_payload(report, paths: Dict[str, str]) -> str:
    failed = [r for r in report.results if not r.passed][:3]
    examples = [
        {
            "task_id": r.task_id,
            "run": r.run,
            "first_failure": r.first_failure,
            "error": r.error,
            "last_tool_calls": [
                {k: t[k] for k in ("tool", "args", "error") if k in t} for t in r.trace_tail[-6:]
            ],
        }
        for r in failed
    ]
    return (
        report.summary()
        + "\n\nFiles:\n"
        + "\n".join(f"  {k}: {v}" for k, v in paths.items())
        + ("\n\nFailed run details (JSON):\n" + json.dumps(examples, indent=1, default=str) if examples else "")
    )


def _lengths(value: Any) -> Optional[List[int]]:
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        return [int(x) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def _evaluate_target(args: dict, cases=None, suite=None, seed=None):
    from .cli import _find_agents
    from .decorator import BenchmarkConfig
    from .rules import load as load_rules
    from .runner import evaluate

    agents = _find_agents(args["target"])
    name, fn = agents[0]
    cfg = getattr(fn, "__halflife__", None) or BenchmarkConfig(suite=args.get("suite") or "claims-intake")
    raw = getattr(fn, "__wrapped__", fn)
    return evaluate(
        raw,
        suite=suite or args.get("suite") or cfg.suite,
        lengths=_lengths(args.get("lengths")) or cfg.lengths,
        tasks=int(args.get("tasks") or cfg.tasks),
        runs=int(args.get("runs") or cfg.runs),
        seed=cfg.seed if seed is None else seed,
        agent_name=name,
        cases=cases,
        workflow_length=args.get("workflow_length"),
        rules=load_rules(args["rules"]) if args.get("rules") else None,
    )


def tool_list_suites(args: dict) -> str:
    from .suite import list_suites

    return "\n".join(f"{s.name}: {s.description} (default lengths {list(s.default_lengths)})" for s in list_suites())


def tool_show_task(args: dict) -> str:
    from .runner import build_task
    from .suite import get_suite

    s = get_suite(args.get("suite") or "claims-intake")
    spec = build_task(s, int(args.get("length") or 10), 0, int(args.get("seed") or 0))
    out = {"instructions": spec.instructions, "tools": spec.sandbox.tool_specs()}
    if args.get("answers"):
        out["answer_key"] = [c.name for c in spec.checkpoints]
    return json.dumps(out, indent=1)


def tool_run_benchmark(args: dict) -> str:
    report = _evaluate_target(args)
    return _report_payload(report, report.save(args.get("out") or ".halflife"))


def tool_bench_model(args: dict) -> str:
    from .agents import from_spec
    from .runner import evaluate

    agent = from_spec(args["model"], base_url=args.get("base_url"))
    report = evaluate(
        agent,
        suite=args.get("suite") or "claims-intake",
        lengths=_lengths(args.get("lengths")),
        tasks=int(args.get("tasks") or 2),
        runs=int(args.get("runs") or 2),
        agent_name=args["model"],
        workflow_length=args.get("workflow_length"),
    )
    return _report_payload(report, report.save(args.get("out") or ".halflife"))


def tool_replay_failures(args: dict) -> str:
    with open(args["pack"]) as f:
        pack = json.load(f)
    cases = [(c["length"], c["task_index"]) for c in pack["cases"]]
    report = _evaluate_target(args, cases=cases, suite=pack["suite"], seed=pack["seed"])
    head = f"Replayed {len(cases)} previously failing task(s). Now passing: {report.success_rate*100:.0f}% of runs.\n\n"
    return head + _report_payload(report, report.save(args.get("out") or ".halflife"))


def tool_audit_traces(args: dict) -> str:
    from .rules import load as load_rules
    from .traces import audit

    rules = load_rules(args["rules"]) if args.get("rules") else []
    report = audit(args["traces"], rules, workflow_length=args.get("workflow_length"))
    return _report_payload(report, report.save(args.get("out") or ".halflife"))


def _schema(props: Dict[str, dict], required: List[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


S = lambda d: {"type": "string", "description": d}
I = lambda d: {"type": "integer", "description": d}

TOOLS: Dict[str, Dict[str, Any]] = {
    "list_suites": {"fn": tool_list_suites, "description": "List the built-in task suites.", "inputSchema": _schema({}, [])},
    "show_task": {
        "fn": tool_show_task,
        "description": "Show a sample task: the instructions and tools an agent receives, optionally the answer key.",
        "inputSchema": _schema({"suite": S("Suite name"), "length": I("Task length"), "answers": {"type": "boolean", "description": "Include the answer key"}}, []),
    },
    "run_benchmark": {
        "fn": tool_run_benchmark,
        "description": "Benchmark an agent defined in a Python file (with @benchmark or file.py:function). Returns half-life, failure modes, fixes, file paths including the eval pack.",
        "inputSchema": _schema(
            {
                "target": S("path/to/agent.py or path/to/agent.py:function"),
                "suite": S("Suite name; defaults to the one in @benchmark"),
                "lengths": S("Comma-separated task lengths, e.g. 5,10,20"),
                "tasks": I("Distinct tasks per length"),
                "runs": I("Runs per task"),
                "workflow_length": I("Real workflow length, for an autonomy recommendation"),
                "rules": S("Python file with extra rules"),
            },
            ["target"],
        ),
    },
    "bench_model": {
        "fn": tool_bench_model,
        "description": "Benchmark a model directly with the built-in agent, e.g. ollama:qwen2.5:7b or anthropic:claude-sonnet-5-5.",
        "inputSchema": _schema(
            {"model": S("provider:model"), "suite": S("Suite name"), "lengths": S("Comma-separated lengths"), "tasks": I("Tasks per length"), "runs": I("Runs per task"), "base_url": S("Server URL for local providers"), "workflow_length": I("Real workflow length")},
            ["model"],
        ),
    },
    "replay_failures": {
        "fn": tool_replay_failures,
        "description": "Re-run only the tasks that failed before (from a *.evals.json eval pack) against the current agent code. Use after making a fix.",
        "inputSchema": _schema({"target": S("path/to/agent.py"), "pack": S("Path to the .evals.json file"), "runs": I("Runs per case")}, ["target", "pack"]),
    },
    "audit_traces": {
        "fn": tool_audit_traces,
        "description": "Check recorded agent logs (JSON Lines of tool calls) against rules and estimate the half-life from them.",
        "inputSchema": _schema({"traces": S("Path to a .jsonl trace file"), "rules": S("Python file defining RULES"), "workflow_length": I("Real workflow length")}, ["traces"]),
    },
}


def handle(message: dict) -> Optional[dict]:
    method = message.get("method")
    mid = message.get("id")
    if mid is None:
        return None
    try:
        if method == "initialize":
            requested = (message.get("params") or {}).get("protocolVersion") or "2025-06-18"
            result = {
                "protocolVersion": requested,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "halflife", "version": __version__},
                "instructions": SERVER_INSTRUCTIONS,
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": [{"name": n, "description": t["description"], "inputSchema": t["inputSchema"]} for n, t in TOOLS.items()]}
        elif method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name")
            if name not in TOOLS:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"unknown tool {name}"}}
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    text = TOOLS[name]["fn"](params.get("arguments") or {})
                is_error = False
            except SystemExit as exc:
                text, is_error = str(exc), True
            except Exception as exc:
                text, is_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}", True
            noise = buf.getvalue().strip()
            if noise:
                text += "\n\nOutput printed during the run:\n" + noise[-2000:]
            result = {"content": [{"type": "text", "text": text}], "isError": is_error}
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
    except Exception as exc:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(exc)}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def serve(stdin=None, stdout=None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            stdout.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}) + "\n")
            stdout.flush()
            continue
        batch = message if isinstance(message, list) else [message]
        for m in batch:
            response = handle(m)
            if response is not None:
                stdout.write(json.dumps(response, default=str) + "\n")
                stdout.flush()
    return 0
