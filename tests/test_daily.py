import json
import os

import pytest

import halflife
from halflife.cli import main
from halflife.demo import ClaimsDemoAgent
from halflife.history import badge

AGENT = os.path.join(os.path.dirname(__file__), "..", "examples", "my_agent.py")


def test_init_scaffold_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["init", "--ci"]) == 0
    for f in ("halflife_agent.py", "halflife_rules.py", os.path.join(".github", "workflows", "halflife.yml")):
        assert os.path.exists(f)
    assert main(["run", "halflife_agent.py", "--rules", "halflife_rules.py", "--quick", "--quiet"]) == 0
    before = open("halflife_agent.py").read()
    main(["init"])
    assert open("halflife_agent.py").read() == before


def test_second_run_shows_change(tmp_path, capsys):
    out = str(tmp_path)
    args = ["run", AGENT, "--quick", "--quiet", "--out", out]
    main(args)
    assert "Since last run" not in capsys.readouterr().out
    main(args)
    assert "Since last run" in capsys.readouterr().out
    assert os.path.exists(os.path.join(out, "badge.svg"))
    endpoint = json.load(open(os.path.join(out, "badge.json")))
    assert endpoint["schemaVersion"] == 1 and endpoint["label"] == "agent half-life"


def test_badge_messages():
    assert badge({"half_life": "inf", "certified_half_life": 20.0, "config": {}})["endpoint"]["message"] == "≥ 20 steps"
    assert badge({"half_life": 18.4, "certified_half_life": 18.4, "config": {}})["endpoint"]["message"] == "18 steps"
    assert "<svg" in badge({"half_life": None, "config": {}})["svg"]


def test_badge_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["run", AGENT, "--quick", "--quiet"])
    assert main(["badge", "--out", "docs"]) == 0
    assert os.path.exists(os.path.join("docs", "halflife-badge.svg"))


def test_assert_reliable():
    perfect = ClaimsDemoAgent(base_error=0, fatigue=0)
    assert halflife.assert_reliable(perfect.run, min_half_life=15, suite="claims-intake").success_rate == 1.0
    weak = ClaimsDemoAgent(base_error=0.1, fatigue=0.05, seed=1)
    with pytest.raises(AssertionError, match="half-life below"):
        halflife.assert_reliable(weak.run, min_half_life=15, suite="claims-intake")
    with pytest.raises(AssertionError):
        halflife.assert_reliable(perfect.run, min_half_life=50, suite="claims-intake")
