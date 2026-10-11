"""Suggest rules from an agent's own logs, so `halflife audit` works without writing rules first.

The idea is specification mining: look for patterns the agent follows in most runs
(a lookup before an action, one notification per claim, every listed item handled)
and propose them as rules. Patterns that hold in every run guard future runs; patterns
that hold in most runs but not all point straight at the runs that went wrong.

Mined rules describe what the agent usually does, not what it should do. Review the
file before you rely on it: delete what's coincidence, keep what's policy.

    halflife audit logs.jsonl --suggest-rules
    halflife audit logs.jsonl --suggest-rules my_rules.py
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import rules as R

READ_PREFIXES = (
    "get", "list", "search", "read", "fetch", "find", "query", "lookup",
    "check", "view", "load", "retrieve", "describe", "show", "count", "verify", "validate",
)

MIN_CONFIDENCE = 0.8
MIN_SUPPORT = 3


@dataclass
class Suggestion:
    rule: R.Rule
    code: str
    holds: int
    applies: int
    note: str = ""
    kind: str = ""
    sort_key: tuple = field(default_factory=tuple)
    unit: str = "calls"

    @property
    def confidence(self) -> float:
        return self.holds / self.applies if self.applies else 0.0

    @property
    def breaks(self) -> int:
        return self.applies - self.holds


def is_read(tool: str) -> bool:
    """Lookups (get_claim, listClaims, search-users) as opposed to actions that change something."""
    import re

    first = re.split(r"[_\-.\s]|(?<=[a-z])(?=[A-Z])", tool, maxsplit=1)[0].lower()
    return first in READ_PREFIXES


def _ok(step: dict) -> bool:
    return R._ok(step)


def _scalar(v) -> bool:
    return isinstance(v, (str, int)) and not isinstance(v, bool)


def _id_keys(runs: Dict[str, List[dict]]) -> Dict[str, List[str]]:
    """Per tool, the argument names that look like identifiers (scalar, many distinct values)."""
    values: Dict[Tuple[str, str], set] = defaultdict(set)
    seen: Dict[Tuple[str, str], int] = defaultdict(int)
    floats = set()
    for steps in runs.values():
        for s in steps:
            for k, v in (s.get("args") or {}).items():
                key = (s["tool"], k)
                seen[key] += 1
                if _scalar(v):
                    values[key].add(v)
                else:
                    floats.add(key)
    out: Dict[str, List[str]] = defaultdict(list)
    for (tool, k), n in seen.items():
        if (tool, k) in floats or n == 0:
            continue
        distinct = len(values[(tool, k)])
        if distinct >= 2 or n == 1:
            out[tool].append(k)
    for tool in out:
        out[tool].sort(key=lambda k: -len(values[(tool, k)]))
    return out


def _cardinality(runs, tool, k) -> int:
    return len({(s.get("args") or {}).get(k) for st in runs.values() for s in st if s["tool"] == tool})


def _fmt_list(xs: Sequence[str]) -> str:
    return "[" + ", ".join(repr(x) for x in xs) + "]"


# ── miners ──────────────────────────────────────────────────────────────────


def _mine_requires(runs, tools, writes, ids) -> List[Suggestion]:
    out = []
    for action in writes:
        for before in tools:
            if before == action:
                continue
            for k in ids.get(action, []):
                if k not in ids.get(before, []):
                    continue
                holds = applies = 0
                for steps in runs.values():
                    seen = set()
                    for s in steps:
                        a = s.get("args") or {}
                        if s["tool"] == before and _ok(s):
                            seen.add(str(a.get(k)))
                        elif s["tool"] == action and _ok(s) and k in a:
                            applies += 1
                            holds += str(a.get(k)) in seen
                if applies >= MIN_SUPPORT and holds / applies >= MIN_CONFIDENCE:
                    rule = R.requires(action, before=before, match=[k])
                    out.append(Suggestion(rule, f"requires({action!r}, before={before!r}, match=[{k!r}])", holds, applies,
                                          f"{action} comes after {before} for the same {k}", "requires",
                                          (action, k, before)))
    return out


def _mine_at_most_once(runs, writes, ids) -> List[Suggestion]:
    out = []
    for action in writes:
        keys = ids.get(action, [])
        candidates: List[Tuple[str, ...]] = [(k,) for k in keys] + [tuple(p) for p in itertools.combinations(keys, 2)]
        best = None
        for key in candidates:
            holds = applies = 0
            for steps in runs.values():
                counts: Dict[tuple, int] = defaultdict(int)
                for s in steps:
                    if s["tool"] == action and _ok(s):
                        counts[R._key(s.get("args") or {}, key)] += 1
                for c in counts.values():
                    applies += 1
                    holds += c <= 1
            if applies >= MIN_SUPPORT and holds / applies >= MIN_CONFIDENCE:
                best = (key, holds, applies)
                break
        if best:
            key, holds, applies = best
            out.append(Suggestion(R.at_most_once(action, key=list(key)), f"at_most_once({action!r}, key={_fmt_list(key)})",
                                  holds, applies, f"{action} happens at most once per {', '.join(key)}", "at_most_once",
                                  (action,), ", ".join(key) + " values"))
    return out


def _mine_eventually(runs, writes, ids) -> List[Suggestion]:
    out = []
    for trigger in writes:
        for then in writes:
            if then == trigger:
                continue
            for k in ids.get(trigger, []):
                if k not in ids.get(then, []):
                    continue
                holds = applies = 0
                for steps in runs.values():
                    for i, s in enumerate(steps):
                        a = s.get("args") or {}
                        if s["tool"] != trigger or not _ok(s) or k not in a:
                            continue
                        applies += 1
                        want = str(a.get(k))
                        holds += any(t["tool"] == then and _ok(t) and str((t.get("args") or {}).get(k)) == want for t in steps[i + 1:])
                if applies >= MIN_SUPPORT and holds / applies >= MIN_CONFIDENCE:
                    out.append(Suggestion(R.eventually(trigger, then=then, match=[k]),
                                          f"eventually({trigger!r}, then={then!r}, match=[{k!r}])", holds, applies,
                                          f"{trigger} is followed by {then} for the same {k}", "eventually",
                                          (trigger, k, then)))
    return out


def _mine_covers(runs, writes, ids) -> List[Suggestion]:
    """A read returns a list of ids; every id should end up at one of a few final actions."""
    out = []
    sources: Dict[Tuple[str, str], int] = defaultdict(int)
    for steps in runs.values():
        for s in steps:
            res = s.get("result")
            if _ok(s) and isinstance(res, dict):
                for f, v in res.items():
                    if isinstance(v, list) and v and all(_scalar(x) for x in v):
                        sources[(s["tool"], f)] += 1
    for (source, fld), n in sources.items():
        if n < MIN_SUPPORT:
            continue
        # which argument of which actions carries these ids?
        hits: Dict[str, int] = defaultdict(int)
        for steps in runs.values():
            items = None
            for s in steps:
                if items is None and s["tool"] == source and _ok(s) and isinstance(s.get("result"), dict):
                    items = {str(x) for x in s["result"].get(fld, [])}
                elif items is not None and s["tool"] in writes:
                    for k, v in (s.get("args") or {}).items():
                        if str(v) in items:
                            hits[k] += 1
        if not hits:
            continue
        key = max(hits, key=hits.get)
        # the final action each item reached, per run
        finals: Dict[str, int] = defaultdict(int)
        total_items = 0
        for steps in runs.values():
            items, last = None, {}
            for s in steps:
                if items is None and s["tool"] == source and _ok(s) and isinstance(s.get("result"), dict):
                    items = [str(x) for x in s["result"].get(fld, [])]
                elif items is not None and s["tool"] in writes and _ok(s):
                    v = str((s.get("args") or {}).get(key))
                    if v in items:
                        last[v] = s["tool"]
            if items:
                total_items += len(items)
                for t in last.values():
                    finals[t] += 1
        if not total_items:
            continue
        actions = sorted(t for t, c in finals.items() if c >= max(1, 0.05 * total_items))
        if not actions:
            continue
        rule = R.covers(source, field=fld, actions=actions, key=key)
        holds = applies = 0
        for steps in runs.values():
            if any(s["tool"] == source and _ok(s) for s in steps):
                applies += 1
                holds += not rule.check(steps)
        if applies >= MIN_SUPPORT and holds / applies >= MIN_CONFIDENCE:
            out.append(Suggestion(rule, f"covers({source!r}, field={fld!r}, actions={_fmt_list(actions)}, key={key!r})",
                                  holds, applies, f"every {fld} item from {source} reaches {' or '.join(actions)}",
                                  "covers", (source, fld), "runs"))
    return out


def _prune_transitive(sugs: List[Suggestion], kind: str) -> List[Suggestion]:
    """Drop A→C when A→B and B→C on the same key are both suggested: it's implied."""
    edges = {s.sort_key for s in sugs if s.kind == kind}
    keep = []
    for s in sugs:
        if s.kind != kind:
            keep.append(s)
            continue
        a, k, c = s.sort_key
        implied = any((a, k, b) in edges and (b, k, c) in edges for (_, kk, b) in edges if kk == k and b not in (a, c))
        if not implied:
            keep.append(s)
    return keep


def suggest(runs: Dict[str, List[dict]], min_confidence: float = MIN_CONFIDENCE) -> List[Suggestion]:
    global MIN_CONFIDENCE
    saved, MIN_CONFIDENCE = MIN_CONFIDENCE, min_confidence
    try:
        tools = sorted({s["tool"] for st in runs.values() for s in st if s.get("tool") and s["tool"] != "?"})
        writes = [t for t in tools if not is_read(t)]
        ids = _id_keys(runs)
        sugs = (
            _mine_covers(runs, writes, ids)
            + _mine_requires(runs, tools, writes, ids)
            + _mine_eventually(runs, writes, ids)
            + _mine_at_most_once(runs, writes, ids)
        )
        sugs = _prune_transitive(sugs, "requires")
        sugs = _prune_transitive(sugs, "eventually")
        return sugs
    finally:
        MIN_CONFIDENCE = saved


def render(sugs: Sequence[Suggestion], source: str) -> str:
    lines = [
        f'"""Rules suggested by halflife from {source}.',
        "",
        "Each rule is a pattern your agent followed in most runs. Review before relying on them:",
        "delete anything that's coincidence, keep what's policy, add what's missing.",
        "",
        "A rule that held in every run guards future runs. A rule that broke in a few runs",
        "is pointing at those runs: check whether the agent or the rule is wrong.",
        "",
        f"    halflife audit {source} --rules <this file>",
        '"""',
        "",
        "from halflife.rules import at_most_once, covers, eventually, requires",
        "",
        "RULES = [",
    ]
    sections = [
        ("covers", "Every listed item gets handled"),
        ("requires", "Look before you act"),
        ("eventually", "Finish what you start"),
        ("at_most_once", "Don't do it twice"),
    ]
    for kind, title in sections:
        group = [s for s in sugs if s.kind == kind]
        if not group:
            continue
        lines.append(f"    # {title}")
        for s in group:
            verdict = "always held" if s.breaks == 0 else f"BROKE {s.breaks}×, check those runs"
            lines.append(f"    # {s.note}: {s.holds} of {s.applies} {s.unit} ({verdict})")
            lines.append(f"    {s.code},")
        lines.append("")
    if lines[-1] == "":
        lines.pop()
    lines.append("]")
    return "\n".join(lines) + "\n"


def print_summary(sugs: Sequence[Suggestion], path: str) -> None:
    breaking = [s for s in sugs if s.breaks]
    print(f"  Suggested {len(sugs)} rule(s) from your logs → {path}")
    if breaking:
        print(f"  {len(breaking)} of them broke in some runs. Those runs are the ones to look at first.")
    print("  These describe what your agent usually does. Review the file and delete anything that's coincidence.")


def write(traces: str, out: str, min_confidence: float = MIN_CONFIDENCE, overwrite: bool = False):
    """Mine rules from a trace file, write them as a rules file, and return (rules, suggestions)."""
    import os

    from .traces import read

    if os.path.exists(out) and not overwrite:
        raise SystemExit(
            f"halflife: {out} already exists. It may hold rules you've reviewed, so it wasn't overwritten. "
            f"Pass a different path: --suggest-rules other_rules.py"
        )
    runs = read(traces)
    if not runs:
        raise SystemExit(f"halflife: no runs found in {traces}")
    sugs = suggest(runs, min_confidence=min_confidence)
    folder = os.path.dirname(out)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(out, "w") as f:
        f.write(render(sugs, traces))
    return [s.rule for s in sugs], sugs
