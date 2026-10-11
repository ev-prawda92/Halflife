import json
import os

import pytest

import halflife
from halflife.agents import AnthropicAgent, OpenAICompatibleAgent, from_spec
from halflife.cli import main
from halflife.demo import ClaimsDemoAgent, InvoiceDemoAgent
from mock_llm import MockModelServer


@pytest.fixture
def perfect_claims_server():
    srv = MockModelServer(lambda: ClaimsDemoAgent(base_error=0, fatigue=0))
    yield srv
    srv.close()


def test_openai_compatible_agent_full_loop(perfect_claims_server):
    agent = OpenAICompatibleAgent("mock", perfect_claims_server.url + "/v1")
    report = halflife.evaluate(agent, suite="claims-intake", lengths=[5, 20], tasks=2, runs=1, seed=4)
    assert report.success_rate == 1.0, report.summary()
    assert report.usage["model_calls"] > 40
    assert report.usage["input_tokens"] > 0


def test_anthropic_agent_full_loop(perfect_claims_server):
    agent = AnthropicAgent("mock", api_key="test-key", base_url=perfect_claims_server.url)
    report = halflife.evaluate(agent, suite="claims-intake", lengths=[5, 20], tasks=2, runs=1, seed=4)
    assert report.success_rate == 1.0, report.summary()
    assert report.usage["model_calls"] > 40


def test_invoice_suite_over_http():
    srv = MockModelServer(lambda: InvoiceDemoAgent(base_error=0, fatigue=0))
    try:
        agent = OpenAICompatibleAgent("mock", srv.url + "/v1")
        report = halflife.evaluate(agent, suite="invoice-matching", lengths=[6, 20], tasks=2, runs=1, seed=4)
        assert report.success_rate == 1.0, report.summary()
    finally:
        srv.close()


def test_unreachable_server_is_reported_as_agent_error(monkeypatch):
    monkeypatch.setattr("halflife.agents._http.time.sleep", lambda s: None)
    agent = OpenAICompatibleAgent("mock", "http://127.0.0.1:9/v1")
    report = halflife.evaluate(agent, suite="claims-intake", lengths=[5], tasks=1, runs=1)
    assert report.results[0].first_failure["category"] == "agent_error"
    assert "cannot reach" in report.results[0].error


def test_from_spec(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    a = from_spec("ollama:qwen2.5:7b")
    assert a.model == "qwen2.5:7b" and a.base_url == "http://localhost:11434/v1"
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    assert from_spec("openrouter:meta-llama/llama-3.3-70b-instruct").base_url.startswith("https://openrouter.ai")
    assert isinstance(from_spec("anthropic:claude-sonnet-5-5"), AnthropicAgent)
    with pytest.raises(ValueError):
        from_spec("qwen")


def test_bench_cli_compare_and_replay(tmp_path, monkeypatch):
    noisy = MockModelServer(lambda: ClaimsDemoAgent(base_error=0.02, fatigue=0.01, seed=1))
    perfect = MockModelServer(lambda: ClaimsDemoAgent(base_error=0, fatigue=0))
    try:
        out = str(tmp_path)
        args = ["bench", "--suite", "claims-intake", "--lengths", "5,10,20", "--tasks", "2", "--runs", "2", "--quiet", "--out", out]
        assert main(args + ["--model", "compatible:noisy", "--base-url", noisy.url + "/v1", "--workflow-length", "15"]) == 0
        packs = [f for f in os.listdir(out) if f.endswith(".evals.json")]
        assert packs, "noisy model should produce failing cases"
        pack = json.load(open(os.path.join(out, packs[0])))
        assert pack["cases"] and pack["suite"] == "claims-intake"
        result = json.load(open(os.path.join(out, [f for f in os.listdir(out) if f.endswith(".json") and not f.endswith(".evals.json")][0])))
        assert result["remediation"] and result["autonomy"]["tier"]

        replay_out = str(tmp_path / "replay")
        code = main(["bench", "--model", "compatible:fixed", "--base-url", perfect.url + "/v1", "--replay", os.path.join(out, packs[0]), "--runs", "1", "--quiet", "--out", replay_out])
        assert code == 0
        result_file = next(f for f in os.listdir(replay_out) if f.endswith(".json") and not f.endswith(".evals.json"))
        replayed = json.load(open(os.path.join(replay_out, result_file)))
        assert replayed["config"]["replay"] is True
        assert replayed["success_rate"] == 1.0
        assert len(replayed["results"]) == len(pack["cases"])

        files = [os.path.join(out, f) for f in os.listdir(out) if f.endswith(".json") and not f.endswith(".evals.json")]
        files += [os.path.join(replay_out, f) for f in os.listdir(replay_out) if f.endswith(".json") and not f.endswith(".evals.json")]
        assert main(["compare"] + files) == 0
    finally:
        noisy.close()
        perfect.close()
