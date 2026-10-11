import os
import random

import pytest

import halflife
from halflife import Checkpoint, Sandbox, TaskSpec, benchmark, param, tool
from halflife import stats
from halflife.cli import main
from halflife.demo import ClaimsDemoAgent, InvoiceDemoAgent

HERE = os.path.dirname(__file__)
EXAMPLE = os.path.join(HERE, "..", "examples", "my_agent.py")


@pytest.mark.parametrize("suite,agent", [("claims-intake", ClaimsDemoAgent), ("invoice-matching", InvoiceDemoAgent)])
def test_answer_key_matches_policy(suite, agent):
    """An agent that follows the written policy exactly must pass every task."""
    perfect = agent(base_error=0, fatigue=0)
    report = halflife.evaluate(perfect.run, suite=suite, lengths=[5, 10, 20, 40, 80], tasks=12, runs=1, seed=123)
    failed = [r for r in report.results if not r.passed]
    assert not failed, failed[:3]
    assert report.half_life == float("inf")


def test_tasks_are_deterministic():
    s = halflife.get_suite("claims-intake")
    a = halflife.build_task(s, 20, index=3, seed=9)
    b = halflife.build_task(s, 20, index=3, seed=9)
    c = halflife.build_task(s, 20, index=4, seed=9)
    assert [x.name for x in a.checkpoints] == [x.name for x in b.checkpoints]
    assert [x.name for x in a.checkpoints] != [x.name for x in c.checkpoints]


@pytest.mark.parametrize("h,k", [(25.0, 1.0), (25.0, 1.8), (12.0, 0.7)])
def test_fit_recovers_known_decay(h, k):
    rng = random.Random(0)
    points = []
    for length in (5, 10, 20, 40, 60):
        n = 400
        p = stats.success_at(length, h, k)
        s = sum(rng.random() < p for _ in range(n))
        points.append((length, s, n))
    fit_h, fit_k = stats.fit_decay(points)
    assert abs(fit_h - h) / h < 0.12
    assert abs(fit_k - k) / k < 0.25


def test_fit_edge_cases():
    assert stats.fit_decay([(10, 5, 5), (20, 5, 5)])[0] == float("inf")
    assert stats.fit_decay([(10, 0, 5), (20, 0, 5)])[0] == 0.0
    assert stats.observed_crossing([(10, 1.0), (20, 0.5)]) == ("at", 20.0)
    assert stats.observed_crossing([(10, 0.4), (20, 0.1)])[0] == "below"


class Counter(Sandbox):
    def __init__(self):
        self.value = 0

    @tool("Add to the counter.", amount=param.integer("How much to add"))
    def add(self, amount):
        self.value += int(amount)
        return {"value": self.value}


@halflife.suite("counter-test", "test suite")
def _counter_suite(length, rng):
    return TaskSpec(
        instructions=f"Add 1, {length} times.",
        sandbox=Counter(),
        checkpoints=[Checkpoint(f"reach {i}", "count", lambda sb, i=i: sb.value >= i) for i in range(1, length + 1)],
        max_tool_calls=length + 5,
    )


def test_tool_errors_come_back_to_the_agent():
    sb = Counter()
    assert "error" in sb.call("nope")
    assert "missing" in sb.call("add", {})["error"]
    assert "unknown" in sb.call("add", {"amount": 1, "extra": 2})["error"]
    assert sb.call("add", {"amount": 2}) == {"value": 2}
    assert len(sb.trace) == 4


def test_custom_suite_and_failure_modes():
    def loops_forever(task):
        while True:
            task.call("add", amount=0)

    def crashes(task):
        task.call("add", amount=1)
        raise RuntimeError("boom")

    def stops_halfway(task):
        for _ in range(task.length // 2):
            task.call("add", amount=1)

    r = halflife.evaluate(loops_forever, suite="counter-test", lengths=[4], tasks=1, runs=1)
    assert r.results[0].first_failure["category"] == "budget_exceeded"
    r = halflife.evaluate(crashes, suite="counter-test", lengths=[4], tasks=1, runs=1)
    assert r.results[0].first_failure["category"] == "agent_error"
    r = halflife.evaluate(stops_halfway, suite="counter-test", lengths=[10], tasks=1, runs=1)
    assert r.results[0].progress == 5


def test_decorator_is_transparent():
    @benchmark(suite="counter-test", lengths=[3], tasks=1, runs=2)
    def agent(task):
        for _ in range(task.length):
            task.call("add", amount=1)
        return "done"

    assert agent.__halflife__.suite == "counter-test"
    report = agent.evaluate()
    assert report.success_rate == 1.0
    assert report.agent == "agent"


def test_task_exposes_provider_tool_formats():
    s = halflife.get_suite("claims-intake")
    spec = halflife.build_task(s, 10)
    task = halflife.Task("t", s.name, len(spec.checkpoints), spec.instructions, spec.sandbox)
    anthropic_tools = task.anthropic_tools()
    openai_tools = task.openai_tools()
    assert {"name", "description", "input_schema"} <= set(anthropic_tools[0])
    assert openai_tools[0]["type"] == "function"
    decide = next(t for t in anthropic_tools if t["name"] == "decide_line")
    assert "reason_code" not in decide["input_schema"]["required"]


def test_cli_gate(tmp_path):
    out = str(tmp_path)
    assert main(["run", EXAMPLE, "--lengths", "5,10", "--tasks", "1", "--runs", "2", "--quiet", "--out", out, "--min-half-life", "1"]) == 0
    assert main(["run", EXAMPLE, "--lengths", "5,10", "--tasks", "1", "--runs", "2", "--quiet", "--out", out, "--min-half-life", "100000"]) == 1
    files = os.listdir(out)
    assert any(f.endswith(".json") for f in files) and any(f.endswith(".html") for f in files)
