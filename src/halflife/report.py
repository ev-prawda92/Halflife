"""Reports: terminal summary, JSON for machines, and a self-contained HTML page."""

from __future__ import annotations

import datetime as _dt
import html
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import remediation, stats
from .results import RunResult

CATEGORY_LABELS = {
    "agent_error": "Agent crashed",
    "budget_exceeded": "Ran out of tool-call budget (looping)",
    "constraint_violation": "Broke a hard rule",
    "skipped_step": "Skipped a required step",
    "wrong_reason": "Right action, wrong reason code",
    "false_denial": "Denied a valid line",
    "false_flag": "Held or rejected a clean invoice",
    "deductible_tracking": "Lost track of deductible across the batch",
    "payment_amount": "Wrong payment amount",
    "eligibility": "Missed an ineligible member",
    "network": "Missed an out-of-network provider",
    "coverage": "Paid an uncovered service",
    "prior_auth": "Paid without prior authorization",
    "finalize": "Didn't finalize a claim",
    "notification": "Wrong or missing member notice",
    "vendor": "Paid a blocked vendor",
    "duplicate_history": "Paid a previously paid invoice",
    "duplicate_in_batch": "Paid a duplicate from earlier in the batch",
    "price": "Missed a price variance",
    "quantity": "Paid more than was received",
    "receipt_tracking": "Lost track of quantities across the batch",
    "due_date": "Wrong due date",
    "missed_check": "Acted without a required check",
    "duplicate_action": "Did the same action twice",
    "ignored_error": "Repeated a call that had just failed",
    "rule_error": "A rule raised an error",
}


def label(category: str) -> str:
    return CATEGORY_LABELS.get(category, category.replace("_", " ").capitalize())


@dataclass
class Report:
    agent: str
    suite: str
    config: dict
    results: List[RunResult]
    created_at: str
    half_life: Optional[float]
    ci: Tuple[Optional[float], Optional[float]]
    shape: Optional[float]
    crossing: Tuple[str, float]
    by_length: List[dict]
    failure_modes: List[dict]
    remediation: List[dict] = field(default_factory=list)
    structural_advice: List[str] = field(default_factory=list)
    autonomy: Optional[dict] = None
    usage: Dict[str, int] = field(default_factory=dict)

    # ---------- construction ----------

    @classmethod
    def build(cls, agent: str, suite: str, config: dict, results: List[RunResult]) -> "Report":
        by_actual: Dict[float, List[bool]] = defaultdict(list)
        groups: Dict[int, List[RunResult]] = defaultdict(list)
        for r in results:
            by_actual[float(r.length)].append(r.passed)
            groups[r.requested_length].append(r)

        points = [(L, sum(v), len(v)) for L, v in by_actual.items()]
        h, k = stats.fit_decay(points)
        ci = stats.bootstrap_ci(by_actual, seed=config.get("seed", 0))

        rows = []
        for req in sorted(groups):
            rs = groups[req]
            n = len(rs)
            s = sum(r.passed for r in rs)
            lo, hi = stats.wilson(s, n)
            mean_len = sum(r.length for r in rs) / n
            rows.append(
                {
                    "requested_length": req,
                    "mean_length": round(mean_len, 2),
                    "runs": n,
                    "successes": s,
                    "rate": s / n,
                    "ci": [lo, hi],
                    "mean_progress": round(sum(r.progress / r.length for r in rs) / n, 3),
                    "mean_tool_calls": round(sum(r.tool_calls for r in rs) / n, 1),
                }
            )
        crossing = stats.observed_crossing([(row["mean_length"], row["rate"]) for row in rows])

        failed = [r for r in results if not r.passed]
        counts = Counter(r.first_failure["category"] for r in failed if r.first_failure)
        modes = [
            {"category": c, "label": label(c), "count": k, "share": k / len(failed)}
            for c, k in counts.most_common()
        ]

        fixes = []
        for m in modes:
            if m["share"] < 0.05 and fixes:
                continue
            example = next(r for r in failed if r.first_failure and r.first_failure["category"] == m["category"])
            guide = remediation.for_category(m["category"])
            fixes.append(
                {
                    "category": m["category"],
                    "label": m["label"],
                    "share": m["share"],
                    "why": guide["why"],
                    "fixes": guide["fixes"],
                    "example": {
                        "task_id": example.task_id,
                        "run": example.run,
                        "step": example.progress + 1,
                        "of": example.length,
                        "checkpoint": example.first_failure["name"],
                    },
                }
            )

        lengths_tested = [r.length for r in results] or [0]
        workflow = config.get("workflow_length")
        auto = None
        if workflow and h is not None:
            p = stats.success_at(float(workflow), h, k)
            auto = remediation.autonomy(p)
            auto.update(
                {
                    "workflow_length": workflow,
                    "predicted_success": p,
                    "extrapolated": workflow > max(lengths_tested),
                }
            )

        usage = {
            "input_tokens": sum(r.input_tokens for r in results),
            "output_tokens": sum(r.output_tokens for r in results),
            "model_calls": sum(r.model_calls for r in results),
        }

        return cls(
            agent=agent,
            suite=suite,
            config=config,
            results=results,
            created_at=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            half_life=h,
            ci=ci,
            shape=k,
            crossing=crossing,
            by_length=rows,
            failure_modes=modes,
            remediation=fixes,
            structural_advice=remediation.structural(k, h if h and not math.isinf(h) else None, workflow),
            autonomy=auto,
            usage=usage,
        )

    def eval_pack(self) -> dict:
        """The failing tasks, as a replayable eval set: halflife run agent.py --replay pack.json"""
        cases = {}
        for r in self.results:
            if not r.passed:
                key = (r.requested_length, r.task_index)
                entry = cases.setdefault(
                    key,
                    {"length": r.requested_length, "task_index": r.task_index, "task_id": r.task_id, "failed_runs": 0, "failures": []},
                )
                entry["failed_runs"] += 1
                if r.first_failure and r.first_failure["category"] not in entry["failures"]:
                    entry["failures"].append(r.first_failure["category"])
        return {
            "halflife_eval_pack": 1,
            "suite": self.suite,
            "seed": self.config.get("seed", 0),
            "source_agent": self.agent,
            "created_at": self.created_at,
            "cases": [cases[k] for k in sorted(cases)],
        }

    # ---------- queries ----------

    @property
    def runs(self) -> int:
        return len(self.results)

    @property
    def success_rate(self) -> float:
        return sum(r.passed for r in self.results) / max(1, len(self.results))

    @property
    def certified_half_life(self) -> Optional[float]:
        """The half-life backed by evidence: never more than the longest task actually tested."""
        if self.half_life is None:
            return None
        return min(self.half_life, float(self._lengths()[1]))

    def passes(self, min_half_life: float) -> bool:
        h = self.certified_half_life
        return h is not None and h >= min_half_life

    def _lengths(self) -> Tuple[float, float]:
        ls = [r.length for r in self.results] or [0]
        return min(ls), max(ls)

    def half_life_text(self) -> str:
        h = self.half_life
        lo, hi = self._lengths()
        if h is None:
            return "n/a"
        if math.isinf(h):
            return f"> {hi} {self.unit} (no failures observed; test longer tasks)"
        if h == 0:
            return f"< {lo} {self.unit} (every run failed)"
        if h > hi:
            if h <= 2 * hi:
                return f"~{h:.0f} {self.unit} (estimated beyond the longest tested, {hi})"
            return f"> {hi} {self.unit} (not reached in the tested range; test longer runs)"
        return f"{h:.1f} {self.unit}"

    def shape_value(self) -> str:
        if self.shape is None:
            return "n/a"
        if self.shape >= stats.K_BOUNDS[1] * 0.97:
            return f"k ≥ {stats.K_BOUNDS[1]:.0f}"
        return f"k = {self.shape:.2f}"

    @property
    def unit(self) -> str:
        return self.config.get("unit", "steps")

    @property
    def is_audit(self) -> bool:
        return self.config.get("mode") == "audit"

    def rules_broken(self) -> List[Tuple[str, float]]:
        counts: Dict[str, int] = {}
        for r in self.results:
            for name in {v.split(": ", 1)[0] for v in r.violations}:
                counts[name] = counts.get(name, 0) + 1
        n = max(1, len(self.results))
        return sorted(((k, v / n) for k, v in counts.items()), key=lambda x: -x[1])

    def ci_text(self) -> str:
        lo, hi = self.ci
        if lo is None or hi is None:
            return ""
        if math.isinf(lo):
            # no failures at all: there's no lower bound to report, and the half-life text already says so
            return ""
        longest = self._lengths()[1]
        fmt = lambda v: "inf" if math.isinf(v) else f"{v:.1f}"
        if math.isinf(hi) or hi > 3 * longest:
            return f"95% CI ≥ {fmt(lo)}"
        return f"95% CI {fmt(lo)}–{fmt(hi)}"

    def crossing_text(self) -> str:
        kind, value = self.crossing
        if kind == "at":
            return f"~{value:.0f} {self.unit}"
        if kind == "below":
            return f"below 50% already at {value:.0f} {self.unit}"
        return f"stays above 50% through {value:.0f} {self.unit}"

    # ---------- output ----------

    def summary(self) -> str:
        lines = []
        lines.append(f"halflife · {self.suite} · {self.agent}")
        c = self.config
        lines.append(
            f"{self.runs} recorded runs from {c.get('source')}, checked against {len(c.get('rules', []))} rules"
            if self.is_audit
            else f"{self.runs} runs ({len(c.get('lengths', []))} lengths × {c.get('tasks')} tasks × {c.get('runs')} runs)"
        )
        lines.append("")
        lines.append("  Length   Success rate               95% CI       Avg progress")
        for row in self.by_length:
            filled = int(round(row["rate"] * 20))
            bar = "█" * filled + "░" * (20 - filled)
            lo, hi = row["ci"]
            lines.append(
                f"  {row['mean_length']:>6.0f}   {bar} {row['rate']*100:>4.0f}%   "
                f"{lo*100:>3.0f}–{hi*100:<3.0f}%     {row['mean_progress']*100:>4.0f}%"
            )
        lines.append("")
        ci = self.ci_text()
        lines.append(f"  Reliability half-life:  {self.half_life_text()}" + (f"  ({ci})" if ci else ""))
        if self.shape is not None:
            lines.append(f"  Decay shape:            {self.shape_value()}, {stats.shape_text(self.shape)}")
        lines.append(f"  Observed 50% crossing:  {self.crossing_text()}")
        if self.failure_modes:
            lines.append("")
            lines.append("  Where runs broke (first failure)")
            for m in self.failure_modes[:6]:
                lines.append(f"    {m['share']*100:>4.0f}%  {m['label']}")
        if self.is_audit and self.config.get("violations_by_rule"):
            lines.append("")
            lines.append("  Rules broken (runs with at least one violation)")
            for name, share in self.rules_broken()[:8]:
                lines.append(f"    {share*100:>4.0f}%  {name}")
        if self.autonomy:
            a = self.autonomy
            lines.append("")
            note = " (extrapolated: test longer tasks to confirm)" if a["extrapolated"] else ""
            lines.append(
                f"  At your workflow length ({a['workflow_length']} {self.unit}): predicted success "
                f"{a['predicted_success']*100:.0f}% → {a['tier']}{note}"
            )
            lines.append(f"    {a['meaning']}")
        if self.remediation or self.structural_advice:
            lines.append("")
            lines.append("  What to fix first")
            for tip in self.structural_advice:
                lines.append(f"    • {tip}")
            for i, fx in enumerate(self.remediation[:3], 1):
                lines.append(f"    {i}. {fx['label']} ({fx['share']*100:.0f}% of failures)")
                for step in fx["fixes"][:2]:
                    lines.append(f"       - {step}")
                ex = fx["example"]
                lines.append(f"       e.g. {ex['task_id']} run {ex['run']}, step {ex['step']}/{ex['of']}: {ex['checkpoint']}")
        if self.usage.get("model_calls"):
            u = self.usage
            lines.append("")
            lines.append(f"  Model usage: {u['model_calls']} calls, {u['input_tokens']:,} input / {u['output_tokens']:,} output tokens")
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.summary())

    def to_dict(self) -> dict:
        def num(v):
            if v is None:
                return None
            if isinstance(v, float) and math.isinf(v):
                return "inf"
            return v

        return {
            "halflife_version": __import__("halflife").__version__,
            "agent": self.agent,
            "suite": self.suite,
            "created_at": self.created_at,
            "config": self.config,
            "half_life": num(self.half_life),
            "half_life_ci": [num(self.ci[0]), num(self.ci[1])],
            "certified_half_life": num(self.certified_half_life),
            "decay_shape": self.shape,
            "observed_crossing": {"kind": self.crossing[0], "length": self.crossing[1]},
            "success_rate": self.success_rate,
            "by_length": self.by_length,
            "failure_modes": self.failure_modes,
            "remediation": self.remediation,
            "structural_advice": self.structural_advice,
            "autonomy": self.autonomy,
            "usage": self.usage,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, out_dir: str = ".halflife", html_report: bool = True) -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        stamp = self.created_at.replace("-", "").replace(":", "").replace("T", "-").rstrip("Z")
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{self.suite}-{self.agent}-{stamp}")
        if self.config.get("replay"):
            slug += "-replay"
        base, n = slug, 2
        while os.path.exists(os.path.join(out_dir, slug + ".json")):
            slug, n = f"{base}-{n}", n + 1
        paths = {"json": os.path.join(out_dir, slug + ".json")}
        with open(paths["json"], "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        pack = self.eval_pack()
        if pack["cases"] and not self.is_audit:
            paths["evals"] = os.path.join(out_dir, slug + ".evals.json")
            with open(paths["evals"], "w") as f:
                json.dump(pack, f, indent=2)
        if html_report:
            paths["html"] = os.path.join(out_dir, slug + ".html")
            with open(paths["html"], "w") as f:
                f.write(self.to_html())
        return paths

    # ---------- html ----------

    def _chart_svg(self) -> str:
        W, H = 720, 320
        L, R, T, B = 56, 24, 20, 44
        lo_len, hi_len = self._lengths()
        xmax = max(10, hi_len * 1.15)
        X = lambda v: L + v / xmax * (W - L - R)
        Y = lambda r: T + (1 - r) * (H - T - B)
        parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Task success rate by task length" class="chart">']
        for r in (0, 0.25, 0.5, 0.75, 1):
            cls = "grid half" if r == 0.5 else "grid"
            parts.append(f'<line x1="{L}" y1="{Y(r):.1f}" x2="{W-R}" y2="{Y(r):.1f}" class="{cls}"/>')
            parts.append(f'<text x="{L-8}" y="{Y(r)+4:.1f}" class="tick" text-anchor="end">{int(r*100)}%</text>')
        step = 5 if xmax <= 40 else 10 if xmax <= 100 else 20
        v = 0
        while v <= xmax:
            parts.append(f'<text x="{X(v):.1f}" y="{H-B+18}" class="tick" text-anchor="middle">{v}</text>')
            v += step
        parts.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-6}" class="axis" text-anchor="middle">{"Run length (tool calls)" if self.is_audit else "Task length (verifiable steps)"}</text>')
        h = self.half_life
        if h is not None and 0 < h and not math.isinf(h):
            pts = " ".join(f"{X(x):.1f},{Y(stats.success_at(x, h, self.shape)):.1f}" for x in [i * xmax / 120 for i in range(121)])
            parts.append(f'<polyline points="{pts}" class="fit"/>')
            if h <= xmax:
                parts.append(f'<line x1="{X(h):.1f}" y1="{Y(0.5):.1f}" x2="{X(h):.1f}" y2="{Y(0):.1f}" class="hl"/>')
                parts.append(f'<text x="{X(h)+6:.1f}" y="{Y(0.5)-8:.1f}" class="hltext">half-life {h:.1f}</text>')
        for row in self.by_length:
            x = X(row["mean_length"])
            lo, hi = row["ci"]
            parts.append(f'<line x1="{x:.1f}" y1="{Y(lo):.1f}" x2="{x:.1f}" y2="{Y(hi):.1f}" class="err"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{Y(row["rate"]):.1f}" r="5" class="pt"/>')
        parts.append("</svg>")
        return "".join(parts)

    def to_html(self) -> str:
        e = html.escape
        rows_html = "".join(
            f"<tr><td>{row['mean_length']:.0f}</td><td>{row['successes']}/{row['runs']}</td>"
            f"<td>{row['rate']*100:.0f}%</td><td>{row['ci'][0]*100:.0f}–{row['ci'][1]*100:.0f}%</td>"
            f"<td>{row['mean_progress']*100:.0f}%</td><td>{row['mean_tool_calls']:.0f}</td></tr>"
            for row in self.by_length
        )
        modes_html = "".join(
            f'<div class="mode"><div class="mlabel">{e(m["label"])}</div>'
            f'<div class="mbar"><span style="width:{m["share"]*100:.1f}%"></span></div>'
            f'<div class="mval">{m["share"]*100:.0f}%</div></div>'
            for m in self.failure_modes
        ) or "<p>No failures.</p>"

        failed = [r for r in self.results if not r.passed][:8]
        samples = []
        for r in failed:
            ff = r.first_failure or {}
            calls = "".join(
                f"<li><code>{e(str(t.get('step')))}. {e(t.get('tool',''))}({e(json.dumps(t.get('args', {}))[:140])})</code>"
                + (f' <span class="err-t">→ {e(str(t["error"])[:120])}</span>' if "error" in t else "")
                + "</li>"
                for t in r.trace_tail[-6:]
            )
            samples.append(
                f"<details><summary><strong>{e(r.task_id)}</strong> · run {r.run} · broke at step "
                f"{r.progress + 1} of {r.length} · {e(label(ff.get('category','')))}</summary>"
                f"<p>First failed checkpoint: {e(ff.get('name',''))}</p>"
                + (f"<p>Error: {e(r.error)}</p>" if r.error else "")
                + f"<p>Last tool calls:</p><ol class='trace'>{calls}</ol></details>"
            )

        fix_cards = []
        if self.structural_advice:
            fix_cards.append("".join(f'<p class="struct">{e(t)}</p>' for t in self.structural_advice))
        for i, fx in enumerate(self.remediation, 1):
            ex = fx["example"]
            steps = "".join(f"<li>{e(st)}</li>" for st in fx["fixes"])
            fix_cards.append(
                f'<div class="fix"><h3>{i}. {e(fx["label"])} <span class="share">{fx["share"]*100:.0f}% of failures</span></h3>'
                f'<p class="sub" style="margin:4px 0 8px">{e(fx["why"])}</p><ol>{steps}</ol>'
                f'<p class="ex">Example: {e(ex["task_id"])} run {ex["run"]}, broke at step {ex["step"]} of {ex["of"]}: {e(ex["checkpoint"])}</p></div>'
            )
        fixes_html = "".join(fix_cards) or "<p>Nothing to fix: no failures observed.</p>"

        auto_html = ""
        if self.autonomy:
            a = self.autonomy
            note = " Extrapolated beyond tested lengths: run longer tasks to confirm." if a["extrapolated"] else ""
            auto_html = (
                f'<div class="card auto"><div class="l">Recommendation at your workflow length ({a["workflow_length"]} {e(self.unit)})</div>'
                f'<div class="v">{e(a["tier"])}</div><p class="sub" style="margin:4px 0 0">Predicted success {a["predicted_success"]*100:.0f}%. '
                f'{e(a["meaning"])}{e(note)}</p></div>'
            )
        pack_cases = 0 if self.is_audit else len(self.eval_pack()["cases"])
        rules_html = ""
        if self.is_audit:
            rules_html = '<div class="card"><h2>Rules broken</h2>' + "".join(
                f'<div class="mode"><div class="mlabel">{e(n)}</div><div class="mbar"><span style="width:{sh*100:.1f}%"></span></div><div class="mval">{sh*100:.0f}%</div></div>'
                for n, sh in self.rules_broken()
            ) + '<p class="sub" style="margin:8px 0 0">Share of runs with at least one violation of each rule.</p></div>'
        pack_html = (
            f"<p>{pack_cases} failing task(s) were saved as a replayable eval pack next to this report (<code>*.evals.json</code>). "
            f"After a fix, re-run exactly those cases: <code>halflife run agent.py --replay &lt;pack&gt;.evals.json</code></p>"
            if pack_cases else ""
        )

        ci = self.ci_text()
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Halflife report</title>
<style>
:root{{--bg:#F3F5F7;--card:#fff;--ink:#12161C;--muted:#4A5462;--line:#DCE1E7;--accent:#1F4E8C;--warn:#B4530A}}
@media (prefers-color-scheme: dark){{:root{{--bg:#0E1116;--card:#171B22;--ink:#E8ECF0;--muted:#A3ADB9;--line:#2A313B;--accent:#7FB0E8;--warn:#F0A060}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:960px;margin:0 auto;padding:32px 16px 64px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:17px;margin:0 0 12px}}
.sub{{color:var(--muted);margin:0 0 24px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px;margin-bottom:16px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:16px}}
.kpi .v{{font:600 28px ui-monospace,SFMono-Regular,Menlo,monospace}} .kpi .l{{color:var(--muted);font-size:13px}}
.chart{{width:100%;height:auto}} .grid{{stroke:var(--line)}} .grid.half{{stroke:var(--warn);stroke-dasharray:4 4}}
.tick{{fill:var(--muted);font:11px ui-monospace,monospace}} .axis{{fill:var(--muted);font-size:12px}}
.fit{{fill:none;stroke:var(--accent);stroke-width:2.5}} .pt{{fill:var(--accent);stroke:var(--card);stroke-width:2}}
.err{{stroke:var(--accent);stroke-width:1.5;opacity:.6}} .hl{{stroke:var(--warn);stroke-dasharray:2 3}} .hltext{{fill:var(--warn);font-size:12px}}
table{{width:100%;border-collapse:collapse}} th,td{{text-align:left;padding:8px;border-top:1px solid var(--line)}} th{{color:var(--muted);font-weight:500;font-size:13px;border-top:none}}
.tablewrap{{overflow-x:auto}}
.mode{{display:grid;grid-template-columns:minmax(0,2fr) minmax(0,2fr) 48px;gap:12px;align-items:center;margin:6px 0}}
.mbar{{background:var(--bg);border-radius:4px;height:10px}} .mbar span{{display:block;height:10px;border-radius:4px;background:var(--warn)}}
.mval{{font-family:ui-monospace,monospace;text-align:right}}
details{{border-top:1px solid var(--line);padding:10px 0}} summary{{cursor:pointer}} .trace{{font-size:13px}} .err-t{{color:var(--warn)}}
code{{font-size:12.5px;word-break:break-word}}
.fix{{border-top:1px solid var(--line);padding:12px 0}} .fix h3{{font-size:15px;margin:0}} .share{{color:var(--warn);font-weight:500;font-size:13px;margin-left:6px}}
.fix ol{{margin:0;padding-left:20px}} .fix li{{margin:4px 0}} .ex{{font-size:13px;color:var(--muted);margin:8px 0 0}}
.struct{{background:var(--bg);border-radius:8px;padding:10px 12px;margin:0 0 8px}} .auto .v{{font:600 22px ui-sans-serif,system-ui,sans-serif}}
</style></head><body><main>
<h1>Reliability report: {e(self.agent)}</h1>
<p class="sub">Suite <strong>{e(self.suite)}</strong> · {self.runs} runs · {e(self.created_at)}</p>
<div class="kpis">
<div class="card kpi"><div class="l">Reliability half-life</div><div class="v">{e(self.half_life_text().split(' (')[0])}</div><div class="l">{e(ci)}</div></div>
<div class="card kpi"><div class="l">Decay shape</div><div class="v">{e(self.shape_value())}</div><div class="l">{e(stats.shape_text(self.shape))}</div></div>
<div class="card kpi"><div class="l">Observed 50% crossing</div><div class="v">{e(self.crossing_text())}</div><div class="l">straight from the data</div></div>
</div>
{auto_html}
{rules_html}
<div class="card"><h2>What to fix first</h2>{fixes_html}{pack_html}</div>
<div class="card"><h2>Success rate by task length</h2>{self._chart_svg()}
<p class="sub" style="margin:8px 0 0">Dots are observed success rates with 95% intervals. The line is the fitted decay, success = 2^(−(length / half-life)^k).</p></div>
<div class="card"><h2>By length</h2><div class="tablewrap"><table><thead><tr><th>Length</th><th>Passed</th><th>Rate</th><th>95% CI</th><th>Avg progress before failure</th><th>Avg tool calls</th></tr></thead><tbody>{rows_html}</tbody></table></div></div>
<div class="card"><h2>Where runs broke</h2>{modes_html}</div>
<div class="card"><h2>Sample failed runs</h2>{''.join(samples) or '<p>No failed runs.</p>'}</div>
</main></body></html>"""
