"""Command line: halflife run | demo | show | suites"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from typing import Any, Callable, List, Optional, Tuple

from . import __version__
from .decorator import BenchmarkConfig
from .report import Report
from .runner import build_task, evaluate
from .suite import get_suite, list_suites


def _progress(enabled: bool) -> Optional[Callable[[int, int], None]]:
    if not enabled or not sys.stderr.isatty():
        return None

    def show(done: int, total: int) -> None:
        width = 30
        filled = int(width * done / total)
        sys.stderr.write(f"\r  running [{'█' * filled}{'░' * (width - filled)}] {done}/{total}")
        if done == total:
            sys.stderr.write("\r" + " " * (width + 30) + "\r")
        sys.stderr.flush()

    return show


def _load_module(path: str):
    if path.endswith(".py") or os.path.sep in path or os.path.exists(path):
        full = os.path.abspath(path)
        if not os.path.exists(full):
            raise SystemExit(f"halflife: no such file: {path}")
        folder = os.path.dirname(full)
        if folder not in sys.path:
            sys.path.insert(0, folder)
        name = os.path.splitext(os.path.basename(full))[0]
        spec = importlib.util.spec_from_file_location(name, full)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    return importlib.import_module(path)


def _find_agents(target: str) -> List[Tuple[str, Callable[..., Any]]]:
    path, sep, func = target.rpartition(":") if ":" in target and not os.path.exists(target) else (target, "", "")
    module = _load_module(path)
    if func:
        obj = getattr(module, func, None)
        if obj is None:
            raise SystemExit(f"halflife: '{func}' not found in {path}")
        return [(func, obj)]
    found = [(name, obj) for name, obj in vars(module).items() if callable(obj) and hasattr(obj, "__halflife__")]
    if not found:
        raise SystemExit(
            f"halflife: no @benchmark functions found in {target}.\n"
            "Decorate your agent with @benchmark(suite=...) or name it explicitly: file.py:function"
        )
    return found


def _lengths(text: Optional[str]) -> Optional[List[int]]:
    if not text:
        return None
    return [int(x) for x in text.replace(" ", "").split(",") if x]


def _finish(report: Report, args: argparse.Namespace) -> int:
    from .history import badge, delta_text, previous

    prev = None if report.config.get("replay") else previous(args.out, report.suite, report.agent)
    if not args.quiet:
        print()
        report.print_summary()
    paths = report.save(args.out, html_report=not args.no_html)
    if not report.config.get("replay"):
        b = badge(report)
        with open(os.path.join(args.out, "badge.json"), "w") as f:
            json.dump(b["endpoint"], f)
        with open(os.path.join(args.out, "badge.svg"), "w") as f:
            f.write(b["svg"])
    delta = delta_text(report, prev)
    if delta:
        print(f"\n  {delta}")
    if not args.quiet:
        print()
        print(f"  Results: {paths['json']}")
        if "html" in paths:
            print(f"  Report:  {paths['html']}")
        if "evals" in paths:
            print(f"  Evals:   {paths['evals']}  (replay with --replay)")
    if args.min_half_life is not None:
        ok = report.passes(args.min_half_life)
        verdict = "PASS" if ok else "FAIL"
        print(f"\n  Gate: half-life ≥ {args.min_half_life:g} steps → {verdict}")
        longest = max(r.length for r in report.results)
        if not ok and report.half_life is not None and report.half_life >= args.min_half_life:
            print(f"  Only tasks up to {longest} steps were tested. Add longer lengths to certify {args.min_half_life:g}.")
        return 0 if ok else 1
    return 0


def _replay(args: argparse.Namespace) -> Optional[dict]:
    if not getattr(args, "replay", None):
        return None
    with open(args.replay) as f:
        pack = json.load(f)
    if not pack.get("halflife_eval_pack"):
        raise SystemExit(f"halflife: {args.replay} is not an eval pack")
    return pack


def _rules(args: argparse.Namespace):
    if not getattr(args, "rules", None):
        return None
    from .rules import load

    return load(args.rules)


def cmd_audit(args: argparse.Namespace) -> int:
    from .rules import load
    from .traces import audit

    rules = load(args.rules) if args.rules else []
    report = audit(args.traces, rules, name=args.name, include_defaults=not args.no_default_rules, workflow_length=args.workflow_length)
    return _finish(report, args)


def _evaluate(agent, name: str, cfg: BenchmarkConfig, args: argparse.Namespace) -> Report:
    pack = _replay(args)
    suite = args.suite or cfg.suite
    seed = cfg.seed if args.seed is None else args.seed
    cases = None
    if pack:
        suite, seed = pack["suite"], pack["seed"]
        cases = [(c["length"], c["task_index"]) for c in pack["cases"]]
        if not args.quiet:
            print(f"Replaying {len(cases)} failing task(s) from {args.replay} (originally from {pack.get('source_agent')}).")
    quick = getattr(args, "quick", False)
    return evaluate(
        agent,
        suite=suite,
        lengths=_lengths(args.lengths) or ([5, 10, 20] if quick else cfg.lengths),
        tasks=args.tasks or (2 if quick else cfg.tasks),
        runs=args.runs or (2 if quick else cfg.runs),
        seed=seed,
        workers=args.workers or cfg.workers,
        agent_name=name,
        progress=_progress(not args.quiet),
        cases=cases,
        workflow_length=args.workflow_length,
        rules=_rules(args),
        save_traces=args.save_traces,
    )


def cmd_run(args: argparse.Namespace) -> int:
    code = 0
    for name, fn in _find_agents(args.target):
        cfg = getattr(fn, "__halflife__", None) or BenchmarkConfig(suite=args.suite or "claims-intake")
        raw = getattr(fn, "__wrapped__", fn)
        code = max(code, _finish(_evaluate(raw, name, cfg, args), args))
    return code


def cmd_bench(args: argparse.Namespace) -> int:
    from .agents import from_spec

    code = 0
    reports = []
    for spec in args.model:
        agent = from_spec(spec, base_url=args.base_url, temperature=args.temperature)
        if not args.quiet:
            print(f"\nBenchmarking {spec} on {args.suite or 'claims-intake'} …")
            if spec.startswith("ollama:") and not os.environ.get("OLLAMA_CONTEXT_LENGTH"):
                print("  Note: Ollama's default context window is short and silently drops older messages,")
                print("  which makes long tasks fail for the wrong reason. Start Ollama with a larger window:")
                print("  OLLAMA_CONTEXT_LENGTH=32768 ollama serve")
        cfg = BenchmarkConfig(suite=args.suite or "claims-intake", tasks=2, runs=3)
        report = _evaluate(agent, spec, cfg, args)
        reports.append(report)
        code = max(code, _finish(report, args))
    if len(reports) > 1:
        print()
        print(comparison_table(reports))
    return code


def comparison_table(reports: List[Report]) -> str:
    rows = sorted(reports, key=lambda r: -(r.certified_half_life or 0))
    lines = ["  Leaderboard", "  " + "-" * 92, f"  {'#':<3}{'Agent':<38}{'Half-life':<14}{'95% CI':<18}{'Shape':<10}{'Top failure'}"]
    for i, r in enumerate(rows, 1):
        top = r.failure_modes[0]["label"] if r.failure_modes else "none"
        hl = r.half_life_text().split(" (")[0]
        ci = r.ci_text().replace("95% CI ", "")
        lines.append(f"  {i:<3}{r.agent[:36]:<38}{hl:<14}{ci:<18}{r.shape_value():<10}{top}")
    return "\n".join(lines)


def cmd_compare(args: argparse.Namespace) -> int:
    rows = []
    for path in args.files:
        with open(path) as f:
            d = json.load(f)
        if "half_life" not in d:
            continue
        rows.append(d)
    if not rows:
        raise SystemExit("halflife: no result files given")
    suites = {d["suite"] for d in rows}
    def key(d):
        h = d["half_life"]
        return -(float("inf") if h == "inf" else (h or 0))
    print(f"  Leaderboard · {', '.join(sorted(suites))}")
    print("  " + "-" * 80)
    print(f"  {'#':<3}{'Agent':<38}{'Half-life':<12}{'Success':<10}{'Top failure'}")
    for i, d in enumerate(sorted(rows, key=key), 1):
        h = d["half_life"]
        hl = "> tested" if h == "inf" else ("n/a" if h is None else f"{h:.1f}")
        top = d["failure_modes"][0]["label"] if d.get("failure_modes") else "none"
        print(f"  {i:<3}{d['agent'][:36]:<38}{hl:<12}{d['success_rate']*100:>5.0f}%    {top}")
    if len(suites) > 1:
        print("\n  Note: these results come from different suites and aren't directly comparable.")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import demo_agent_for

    agent = demo_agent_for(args.suite)
    if not args.quiet:
        print(f"Running a simulated agent on '{args.suite}'. No API keys needed.")
        print("It follows the policy but slips more as tasks get longer, like real agents do.")
    report = evaluate(
        agent.run,
        suite=args.suite,
        lengths=_lengths(args.lengths) or ([5, 10, 20] if args.quick else None),
        tasks=args.tasks or (2 if args.quick else 4),
        runs=args.runs or (2 if args.quick else 5),
        seed=args.seed or 0,
        workers=args.workers or 1,
        agent_name="demo-agent",
        progress=_progress(not args.quiet),
        workflow_length=args.workflow_length,
    )
    return _finish(report, args)


def cmd_show(args: argparse.Namespace) -> int:
    s = get_suite(args.suite)
    spec = build_task(s, args.length, 0, args.seed)
    tools = spec.sandbox.tool_specs()
    if args.json:
        print(json.dumps({"instructions": spec.instructions, "tools": tools, "checkpoints": [c.name for c in spec.checkpoints]}, indent=2))
        return 0
    print(f"Suite: {s.name} · requested length {args.length} · {len(spec.checkpoints)} checkpoints\n")
    print(spec.instructions)
    print("\nTools")
    for t in tools:
        params = ", ".join(t["parameters"]["properties"])
        print(f"  {t['name']}({params})  {t['description']}")
    if args.answers:
        print("\nAnswer key (checkpoints, in order). The agent never sees this.")
        for i, c in enumerate(spec.checkpoints, 1):
            print(f"  {i:>3}. {c.name}")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    from .scaffold import init

    written = init(".", suite=args.suite, ci=args.ci)
    if not written:
        print("Nothing to do: the halflife files already exist here.")
        return 0
    for f in written:
        print(f"  created {f}")
    print("\nNext:")
    print("  halflife run halflife_agent.py --rules halflife_rules.py")
    print("Then replace the stand-in in halflife_agent.py with your agent.")
    return 0


def cmd_badge(args: argparse.Namespace) -> int:
    import glob

    from .history import badge

    src = args.result
    if not src:
        candidates = [p for p in glob.glob(os.path.join(".halflife", "*.json")) if not p.endswith((".evals.json", "badge.json"))]
        if not candidates:
            raise SystemExit("halflife: no results yet. Run halflife first, or pass a result .json")
        src = max(candidates, key=os.path.getmtime)
    with open(src) as f:
        b = badge(json.load(f), label=args.label)
    os.makedirs(args.out, exist_ok=True)
    svg = os.path.join(args.out, "halflife-badge.svg")
    with open(svg, "w") as f:
        f.write(b["svg"])
    with open(os.path.join(args.out, "halflife-badge.json"), "w") as f:
        json.dump(b["endpoint"], f)
    print(f"  {b['endpoint']['label']}: {b['endpoint']['message']}")
    print(f"  Badge: {svg}")
    print(f"  Add to your README:  ![{args.label}]({svg})")
    return 0


def cmd_suites(args: argparse.Namespace) -> int:
    for s in list_suites():
        print(f"{s.name:<20} {s.description}")
    return 0


def _run_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--quick", action="store_true", help="fast check for the dev loop: lengths 5,10,20, 2 tasks, 2 runs")
    p.add_argument("--lengths", help="comma-separated task lengths, e.g. 5,10,20,40")
    p.add_argument("--tasks", type=int, help="distinct tasks per length")
    p.add_argument("--runs", type=int, help="repeated runs per task")
    p.add_argument("--seed", type=int, help="task seed (same seed, same tasks)")
    p.add_argument("--workers", type=int, help="parallel runs")
    p.add_argument("--out", default=".halflife", help="output folder (default .halflife)")
    p.add_argument("--min-half-life", type=float, help="exit 1 if the half-life is below this (for CI)")
    p.add_argument("--workflow-length", type=int, help="your real workflow's length in steps, for an autonomy recommendation")
    p.add_argument("--rules", help="Python file with extra rules (invariants) to check on every run")
    p.add_argument("--save-traces", help="write every run's full tool-call trace to this .jsonl file")
    p.add_argument("--no-html", action="store_true", help="skip the HTML report")
    p.add_argument("--quiet", action="store_true")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="halflife", description="Measure how long a task you can trust an AI agent with.")
    parser.add_argument("--version", action="version", version=f"halflife {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="benchmark your agent")
    run.add_argument("target", help="file.py, file.py:function, or module:function")
    run.add_argument("--suite", help="override the suite set in @benchmark")
    run.add_argument("--replay", help="re-run only the failing tasks in an eval pack (*.evals.json)")
    _run_flags(run)
    run.set_defaults(func=cmd_run)

    bench = sub.add_parser("bench", help="benchmark a model directly, no code: --model ollama:qwen2.5:7b")
    bench.add_argument("--model", action="append", required=True, help="provider:model; repeat to compare models")
    bench.add_argument("--suite", help="suite to run (default claims-intake)")
    bench.add_argument("--base-url", help="server URL for vllm, lmstudio or compatible providers")
    bench.add_argument("--temperature", type=float, default=0.0)
    bench.add_argument("--replay", help="re-run only the failing tasks in an eval pack (*.evals.json)")
    _run_flags(bench)
    bench.set_defaults(func=cmd_bench)

    aud = sub.add_parser("audit", help="check your agent's logs against rules; no setup, any agent")
    aud.add_argument("traces", help="JSON Lines file of tool calls or runs")
    aud.add_argument("--rules", help="Python file defining RULES = [...]")
    aud.add_argument("--name", help="label for the report")
    aud.add_argument("--no-default-rules", action="store_true", help="skip the built-in loop and error-recovery rules")
    aud.add_argument("--out", default=".halflife")
    aud.add_argument("--min-half-life", type=float)
    aud.add_argument("--workflow-length", type=int)
    aud.add_argument("--no-html", action="store_true")
    aud.add_argument("--quiet", action="store_true")
    aud.set_defaults(func=cmd_audit)

    bdg = sub.add_parser("badge", help="make a README badge from your latest result")
    bdg.add_argument("result", nargs="?", help="result .json (default: latest in .halflife)")
    bdg.add_argument("--out", default=".")
    bdg.add_argument("--label", default="agent half-life")
    bdg.set_defaults(func=cmd_badge)

    ini = sub.add_parser("init", help="scaffold halflife_agent.py, rules, and optionally a CI workflow")
    ini.add_argument("--suite", default="claims-intake")
    ini.add_argument("--ci", action="store_true", help="also add .github/workflows/halflife.yml")
    ini.set_defaults(func=cmd_init)

    mcp = sub.add_parser("mcp", help="run as an MCP server for Claude Code, Cursor and other coding agents")
    mcp.set_defaults(func=lambda a: __import__("halflife.mcp", fromlist=["serve"]).serve())

    compare = sub.add_parser("compare", help="leaderboard from saved result files")
    compare.add_argument("files", nargs="+", help="result .json files")
    compare.set_defaults(func=cmd_compare)

    demo = sub.add_parser("demo", help="run a simulated agent, no API keys needed")
    demo.add_argument("--suite", default="claims-intake")
    _run_flags(demo)
    demo.set_defaults(func=cmd_demo)

    show = sub.add_parser("show", help="print a sample task: instructions, tools, answer key")
    show.add_argument("--suite", default="claims-intake")
    show.add_argument("--length", type=int, default=10)
    show.add_argument("--seed", type=int, default=0)
    show.add_argument("--answers", action="store_true", help="also print the answer key")
    show.add_argument("--json", action="store_true")
    show.set_defaults(func=cmd_show)

    suites = sub.add_parser("suites", help="list available suites")
    suites.set_defaults(func=cmd_suites)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
