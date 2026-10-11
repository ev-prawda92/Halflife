"""Rules: invariants that must hold for any run of an agent, checked on its tool-call trace.

Rules need no answer key. They describe what must always be true, whatever the
inputs: a check happens before the action it guards, nothing is done twice,
every item that was started gets finished, the agent doesn't loop.

Use them three ways:
  - `halflife audit traces.jsonl --rules rules.py` on logs you already have
  - `halflife run agent.py --rules rules.py` as extra checks during a benchmark
  - from Python: `check(rules, steps)`

A step is a dict: {"tool": name, "args": {...}, "result": ..., "error": ...}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

Step = Dict[str, Any]
Match = Union[Sequence[str], Dict[str, str], None]
Pred = Optional[Callable[[dict], bool]]


@dataclass
class Violation:
    rule: str
    category: str
    step: int
    message: str

    def to_dict(self) -> dict:
        return {"rule": self.rule, "category": self.category, "step": self.step, "message": self.message}


def _pairs(match: Match) -> Dict[str, str]:
    if not match:
        return {}
    if isinstance(match, dict):
        return dict(match)
    return {k: k for k in match}


def _key(args: dict, keys: Iterable[str]) -> tuple:
    return tuple(str(args.get(k)) for k in keys)


def _ok(step: Step) -> bool:
    return "error" not in step or not step.get("error")


class Rule:
    category = "constraint_violation"

    def __init__(self, name: Optional[str] = None, category: Optional[str] = None):
        if name:
            self.name = name
        if category:
            self.category = category

    name = "rule"

    def check(self, steps: List[Step]) -> List[Violation]:
        raise NotImplementedError

    def _v(self, step: int, message: str) -> Violation:
        return Violation(self.name, self.category, step, message)

    def __repr__(self) -> str:
        return f"<rule {self.name}>"


class requires(Rule):
    """Every successful `action` call must come after a successful `before` call with matching args.

    requires("finalize_claim", before="get_claim", match=["claim_id"])
    requires("approve_invoice", before="get_receipts", match={"invoice_id": "invoice_id"})
    """

    category = "missed_check"

    def __init__(self, action: str, before: str, match: Match = None, when: Pred = None, name: Optional[str] = None):
        super().__init__(name or f"{before} before {action}")
        self.action, self.before, self.match, self.when = action, before, _pairs(match), when

    def check(self, steps):
        seen = set()
        out = []
        for i, s in enumerate(steps):
            args = s.get("args") or {}
            if s.get("tool") == self.before and _ok(s):
                seen.add(_key(args, self.match.values()))
            elif s.get("tool") == self.action and _ok(s) and (self.when is None or self.when(args)):
                if _key(args, self.match.keys()) not in seen:
                    detail = ", ".join(f"{k}={args.get(k)}" for k in self.match) or "this call"
                    out.append(self._v(i, f"{self.action} ({detail}) without a prior {self.before}"))
        return out


class at_most_once(Rule):
    """`action` may succeed at most once per key: at_most_once("notify_member", key=["claim_id"])"""

    category = "duplicate_action"

    def __init__(self, action: str, key: Sequence[str] = (), name: Optional[str] = None):
        super().__init__(name or f"{action} at most once per {', '.join(key) or 'run'}")
        self.action, self.key = action, list(key)

    def check(self, steps):
        seen = {}
        out = []
        for i, s in enumerate(steps):
            if s.get("tool") == self.action and _ok(s):
                k = _key(s.get("args") or {}, self.key)
                if k in seen:
                    out.append(self._v(i, f"{self.action} repeated for {dict(zip(self.key, k)) or 'this run'} (first at step {seen[k] + 1})"))
                else:
                    seen[k] = i
        return out


class eventually(Rule):
    """Every successful `trigger` must be followed later by a successful `then` with matching args.

    eventually("finalize_claim", then="notify_member", match=["claim_id"])
    """

    category = "skipped_step"

    def __init__(self, trigger: str, then: Union[str, Sequence[str]], match: Match = None, name: Optional[str] = None):
        thens = [then] if isinstance(then, str) else list(then)
        super().__init__(name or f"{' or '.join(thens)} after {trigger}")
        self.trigger, self.then, self.match = trigger, thens, _pairs(match)

    def check(self, steps):
        out = []
        for i, s in enumerate(steps):
            if s.get("tool") == self.trigger and _ok(s):
                want = _key(s.get("args") or {}, self.match.keys())
                if not any(
                    t.get("tool") in self.then and _ok(t) and _key(t.get("args") or {}, self.match.values()) == want
                    for t in steps[i + 1:]
                ):
                    detail = ", ".join(f"{k}={(s.get('args') or {}).get(k)}" for k in self.match) or "this call"
                    out.append(self._v(i, f"{self.trigger} ({detail}) never followed by {' or '.join(self.then)}"))
        return out


class covers(Rule):
    """Every id returned by `source` must later reach one of `actions`.

    covers("list_claims", field="claims", actions=["finalize_claim", "flag_for_review"], key="claim_id")
    """

    category = "skipped_step"

    def __init__(self, source: str, field: str, actions: Sequence[str], key: str, name: Optional[str] = None):
        super().__init__(name or f"every {field} item reaches {' or '.join(actions)}")
        self.source, self.field, self.actions, self.key = source, field, list(actions), key

    def check(self, steps):
        out = []
        for i, s in enumerate(steps):
            if s.get("tool") != self.source or not _ok(s):
                continue
            result = s.get("result") or {}
            items = result.get(self.field, []) if isinstance(result, dict) else []
            done = {
                str((t.get("args") or {}).get(self.key))
                for t in steps[i + 1:]
                if t.get("tool") in self.actions and _ok(t)
            }
            missing = [str(x) for x in items if str(x) not in done]
            if missing:
                shown = ", ".join(missing[:5]) + (" …" if len(missing) > 5 else "")
                out.append(self._v(len(steps), f"{len(missing)} of {len(items)} {self.field} never reached {' or '.join(self.actions)}: {shown}"))
            break
        return out


class never(Rule):
    """A call that must never succeed: never("approve_invoice", when=lambda a: a["amount"] > 10000)"""

    category = "constraint_violation"

    def __init__(self, action: str, when: Pred = None, name: Optional[str] = None, message: Optional[str] = None):
        super().__init__(name or f"never {action}")
        self.action, self.when, self.message = action, when, message

    def check(self, steps):
        return [
            self._v(i, self.message or f"forbidden call: {self.action}({s.get('args')})")
            for i, s in enumerate(steps)
            if s.get("tool") == self.action and _ok(s) and (self.when is None or self.when(s.get("args") or {}))
        ]


class no_loops(Rule):
    """The same call (tool and args) may not repeat more than `max_repeats` times in a row."""

    category = "budget_exceeded"

    def __init__(self, max_repeats: int = 3, name: Optional[str] = None):
        super().__init__(name or f"no identical call repeated more than {max_repeats}×")
        self.max_repeats = max_repeats

    def check(self, steps):
        out, run, prev = [], 0, None
        for i, s in enumerate(steps):
            sig = (s.get("tool"), repr(sorted((s.get("args") or {}).items())))
            run = run + 1 if sig == prev else 1
            prev = sig
            if run == self.max_repeats + 1:
                out.append(self._v(i, f"{s.get('tool')} called {run}× in a row with the same arguments"))
        return out


class recovers_from_errors(Rule):
    """After a tool error, the agent must not repeat the identical failing call."""

    category = "ignored_error"

    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "doesn't repeat a call that just failed")

    def check(self, steps):
        out = []
        for i in range(1, len(steps)):
            a, b = steps[i - 1], steps[i]
            if a.get("error") and a.get("tool") == b.get("tool") and (a.get("args") or {}) == (b.get("args") or {}):
                out.append(self._v(i, f"repeated {b.get('tool')} right after it failed: {str(a.get('error'))[:80]}"))
        return out


class custom(Rule):
    """Wrap any function(steps) -> list of (step_index, message), or a bool for the whole run."""

    def __init__(self, fn: Callable[[List[Step]], Any], name: Optional[str] = None, category: str = "constraint_violation"):
        super().__init__(name or fn.__name__, category)
        self.fn = fn

    def check(self, steps):
        res = self.fn(steps)
        if res is True or res is None:
            return []
        if res is False:
            return [self._v(len(steps), f"{self.name} failed")]
        return [self._v(int(i), str(msg)) for i, msg in res]


def rule(name: Optional[str] = None, category: str = "constraint_violation") -> Callable:
    """Decorator form of custom():

    @rule("totals reconcile")
    def totals(steps): ...
    """

    def deco(fn):
        return custom(fn, name or fn.__name__, category)

    return deco


def default_rules() -> List[Rule]:
    """Rules that hold for almost any agent."""
    return [no_loops(), recovers_from_errors()]


def check(rules: Iterable[Rule], steps: List[Step]) -> List[Violation]:
    out: List[Violation] = []
    for r in rules:
        try:
            out.extend(r.check(steps))
        except Exception as exc:
            out.append(Violation(getattr(r, "name", "rule"), "rule_error", len(steps), f"rule raised {type(exc).__name__}: {exc}"))
    return sorted(out, key=lambda v: v.step)


def load(path: str) -> List[Rule]:
    """Load rules from a Python file: a RULES list, or every Rule defined at module level."""
    from .cli import _load_module

    mod = _load_module(path)
    found = getattr(mod, "RULES", None)
    if found is None:
        found = [v for v in vars(mod).values() if isinstance(v, Rule)]
    if not found:
        raise SystemExit(f"halflife: no rules found in {path}. Define RULES = [...]")
    return list(found)
