import json
import os

import halflife
from halflife.cli import main
from halflife.demo import ClaimsDemoAgent
from halflife.rules import at_most_once, check, covers, custom, eventually, never, no_loops, recovers_from_errors, requires, rule
from halflife.traces import TraceWriter, audit, read

HERE = os.path.dirname(__file__)
RULES_FILE = os.path.join(HERE, "..", "examples", "rules_claims.py")


def s(tool, **args):
    return {"tool": tool, "args": args}


def test_requires_and_when():
    r = requires("pay", before="check", match=["id"], when=lambda a: a.get("amount", 0) > 0)
    assert not check([r], [s("check", id=1), s("pay", id=1, amount=5)])
    assert check([r], [s("check", id=2), s("pay", id=1, amount=5)])
    assert not check([r], [s("pay", id=1, amount=0)])


def test_requires_ignores_failed_checks():
    r = requires("pay", before="check", match=["id"])
    failed = dict(s("check", id=1), error="timeout")
    assert check([r], [failed, s("pay", id=1)])


def test_requires_with_renamed_keys():
    r = requires("approve", before="lookup", match={"invoice": "inv_id"})
    assert not check([r], [s("lookup", inv_id="A"), s("approve", invoice="A")])


def test_at_most_once_and_eventually():
    assert check([at_most_once("notify", key=["id"])], [s("notify", id=1), s("notify", id=1)])
    assert not check([at_most_once("notify", key=["id"])], [s("notify", id=1), s("notify", id=2)])
    ev = eventually("finalize", then=["notify", "flag"], match=["id"])
    assert check([ev], [s("finalize", id=1)])
    assert not check([ev], [s("finalize", id=1), s("flag", id=1)])


def test_covers_items_from_a_listing():
    listing = {"tool": "list", "args": {}, "result": {"items": ["a", "b", "c"]}}
    r = covers("list", field="items", actions=["close"], key="id")
    v = check([r], [listing, s("close", id="a"), s("close", id="c")])
    assert len(v) == 1 and "b" in v[0].message


def test_never_loops_errors_custom():
    assert check([never("delete")], [s("delete", id=1)])
    assert check([no_loops(2)], [s("get", id=1)] * 3)
    assert not check([no_loops(2)], [s("get", id=1)] * 2)
    failed = dict(s("get", id=1), error="bad id")
    assert check([recovers_from_errors()], [failed, s("get", id=1)])

    @rule("short runs")
    def short(steps):
        return len(steps) < 3

    assert check([short], [s("a")] * 4)
    assert check([custom(lambda st: 1 / 0, name="broken")], [s("a")])[0].category == "rule_error"


def test_read_accepts_common_log_shapes(tmp_path):
    p = tmp_path / "logs.jsonl"
    lines = [
        {"session_id": "x", "name": "get", "arguments": json.dumps({"id": 1})},
        {"session_id": "x", "function": {"name": "put", "arguments": "{\"id\": 1}"}, "output": {"error": "nope"}},
        {"trace_id": "y", "steps": [{"tool_name": "get", "input": {"id": 2}}]},
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n\n")
    runs = read(str(p))
    assert list(runs) == ["x", "y"]
    assert runs["x"][0] == {"tool": "get", "args": {"id": 1}}
    assert runs["x"][1]["tool"] == "put" and runs["x"][1]["error"] == "nope"
    assert runs["y"][0]["args"] == {"id": 2}


def test_trace_writer_round_trip(tmp_path):
    p = str(tmp_path / "t.jsonl")
    with TraceWriter(p) as log:
        log.step("r1", "get", {"id": 1}, {"ok": True})
        log.step("r1", "get", {"id": 1}, error="boom")
        log.run("r2", [{"tool": "x", "args": {}}])
    runs = read(p)
    assert len(runs["r1"]) == 2 and runs["r1"][1]["error"] == "boom" and runs["r2"][0]["tool"] == "x"


def test_audit_end_to_end(tmp_path):
    traces = str(tmp_path / "claims.jsonl")
    halflife.evaluate(ClaimsDemoAgent(seed=3).run, suite="claims-intake", lengths=[5, 20], tasks=2, runs=2, save_traces=traces)
    perfect = str(tmp_path / "perfect.jsonl")
    halflife.evaluate(ClaimsDemoAgent(base_error=0, fatigue=0).run, suite="claims-intake", lengths=[5, 20], tasks=3, runs=1, save_traces=perfect)
    rules = halflife.rules.load(RULES_FILE)
    clean = audit(perfect, rules)
    assert clean.success_rate == 1.0, clean.summary()
    assert clean.is_audit and clean.unit == "tool calls"

    out = str(tmp_path / "out")
    assert main(["audit", traces, "--rules", RULES_FILE, "--quiet", "--out", out, "--workflow-length", "40"]) == 0
    result = json.load(open(os.path.join(out, next(f for f in os.listdir(out) if f.endswith(".json")))))
    assert result["config"]["mode"] == "audit" and result["autonomy"]


def test_rules_during_benchmark():
    def skips_notices(task):
        ClaimsDemoAgent(base_error=0, fatigue=0).run(task)

    report = halflife.evaluate(skips_notices, suite="claims-intake", lengths=[5], tasks=1, runs=1, rules=[never("notify_member")])
    r = report.results[0]
    assert not r.passed and any("never notify_member" in v for v in r.violations)
