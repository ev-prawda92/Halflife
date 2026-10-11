import os

import pytest

from halflife.cli import main
from halflife.mcp import tool_audit_traces
from halflife.rules import check, load
from halflife.suggest import is_read, suggest, write
from halflife.traces import read

HERE = os.path.dirname(__file__)
TRACES = os.path.join(HERE, "..", "examples", "traces", "claims.jsonl")


def s(tool, result=None, **args):
    step = {"tool": tool, "args": args}
    if result is not None:
        step["result"] = result
    return step


def codes(sugs):
    return {x.code for x in sugs}


def test_reads_and_actions():
    assert is_read("get_claim") and is_read("listClaims") and is_read("search-users") and is_read("check_provider")
    assert not is_read("notify_member") and not is_read("finalize_claim") and not is_read("getter")


def test_recovers_the_hand_written_claims_rules():
    sugs = suggest(read(TRACES))
    found = codes(sugs)
    assert "requires('decide_line', before='get_claim', match=['claim_id'])" in found
    assert "eventually('finalize_claim', then='notify_member', match=['claim_id'])" in found
    assert "at_most_once('notify_member', key=['claim_id'])" in found
    assert "at_most_once('decide_line', key=['claim_id', 'line'])" in found
    assert any(c.startswith("covers('list_claims', field='claims'") for c in found)


def test_rules_that_break_point_at_the_failing_runs():
    sugs = suggest(read(TRACES))
    notify = next(x for x in sugs if x.code.startswith("eventually('finalize_claim', then='notify_member'"))
    assert 0 < notify.breaks < notify.applies


def test_needs_enough_evidence_and_consistency():
    # one run is too little to call anything a pattern
    runs = {"r1": [s("get", id=1), s("pay", id=1)]}
    assert not suggest(runs)
    # a pattern that holds half the time isn't suggested at the default confidence
    runs = {f"r{i}": [s("get", id=i), s("pay", id=i)] if i % 2 else [s("pay", id=i)] for i in range(10)}
    assert not [x for x in suggest(runs) if x.kind == "requires"]
    assert [x for x in suggest(runs, min_confidence=0.5) if x.kind == "requires"]


def test_implied_rules_are_pruned():
    runs = {f"r{i}": [s("open", id=i), s("approve", id=i), s("close", id=i)] for i in range(5)}
    ev = [x.code for x in suggest(runs) if x.kind == "eventually"]
    assert "eventually('open', then='approve', match=['id'])" in ev
    assert "eventually('approve', then='close', match=['id'])" in ev
    assert "eventually('open', then='close', match=['id'])" not in ev


def test_written_file_loads_and_matches(tmp_path):
    out = str(tmp_path / "rules.py")
    rules, sugs = write(TRACES, out)
    loaded = load(out)
    assert len(loaded) == len(rules) == len(sugs)
    runs = read(TRACES)
    for steps in runs.values():
        assert len(check(loaded, steps)) == len(check(rules, steps))


def test_never_overwrites_a_reviewed_file(tmp_path):
    out = tmp_path / "rules.py"
    out.write_text("RULES = []  # reviewed by hand\n")
    with pytest.raises(SystemExit):
        write(TRACES, str(out))
    assert "reviewed by hand" in out.read_text()


def test_cli_suggest_and_audit(tmp_path, capsys):
    out = str(tmp_path / "suggested.py")
    assert main(["audit", TRACES, "--suggest-rules", out, "--out", str(tmp_path / ".halflife"), "--no-html"]) == 0
    assert os.path.exists(out)
    text = capsys.readouterr().out
    assert "Suggested" in text and "Review" in text


def test_mcp_suggest(tmp_path):
    out = str(tmp_path / "suggested.py")
    payload = tool_audit_traces({"traces": TRACES, "suggest_rules": True, "rules_out": out, "out": str(tmp_path / ".halflife")})
    assert "Suggested" in payload and "RULES = [" in payload and os.path.exists(out)
