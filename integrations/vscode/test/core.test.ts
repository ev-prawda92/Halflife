import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { test } from "node:test";
import { evalPackPath, findAgents, fixPrompt, halfLifeText, latest, parseResult, Result, runMarkdown, shellQuote } from "../src/core";

function result(over: Partial<Result> = {}): Result {
  return {
    agent: "my_agent",
    suite: "claims-intake",
    created_at: "2026-10-11T10:00:00Z",
    config: { unit: "steps" },
    half_life: 32.7,
    half_life_ci: [28.1, 39.0],
    certified_half_life: 32.7,
    success_rate: 0.7,
    by_length: [{ requested_length: 10, mean_length: 10, runs: 5, successes: 4, rate: 0.8, ci: [0.4, 0.96], mean_progress: 0.9 }],
    failure_modes: [{ category: "skipped_step", label: "Skipped a required step", count: 3, share: 1 }],
    remediation: [{ category: "skipped_step", label: "Skipped a required step", share: 1, why: "It stopped early.", fixes: ["Add a completion check."] }],
    structural_advice: [],
    results: [
      { task_id: "claims-intake/L10/t0", run: 0, length: 10, passed: true, first_failure: null },
      {
        task_id: "claims-intake/L40/t1",
        run: 2,
        length: 40,
        passed: false,
        first_failure: { index: 7, name: "CLM-1: member notified", category: "skipped_step" },
        trace_tail: [{ step: 7, tool: "finalize_claim", args: { claim_id: "CLM-1" } }, { step: 8, tool: "get_claim", args: { claim_id: "CLM-2" }, error: "timeout" }],
      },
    ],
    ...over,
  };
}

test("half-life wording never claims more than was tested", () => {
  assert.equal(halfLifeText(result()), "32.7 steps");
  assert.equal(halfLifeText(result(), true), "32.7");
  assert.equal(halfLifeText(result({ half_life: "inf" }), true), "> 40");
  assert.match(halfLifeText(result({ half_life: 60 })), /^~60 steps \(estimated beyond the longest tested, 40\)/);
  assert.match(halfLifeText(result({ half_life: 500 })), /^> 40 steps/);
  assert.match(halfLifeText(result({ half_life: 0 })), /^< 10 steps/);
  assert.equal(halfLifeText(result({ half_life: null })), "n/a");
});

test("finds @benchmark functions", () => {
  const src = [
    "from halflife import benchmark",
    "import halflife",
    "@benchmark(suite='claims-intake')",
    "def my_agent(task):",
    "    pass",
    "",
    "@halflife.benchmark",
    "@other",
    "async def second(task):",
    "    pass",
    "def not_this(task): pass",
  ].join("\n");
  assert.deepEqual(findAgents(src), [
    { name: "my_agent", line: 2, defLine: 3 },
    { name: "second", line: 6, defLine: 8 },
  ]);
});

test("latest picks the newest non-replay result, per agent", () => {
  const a = result({ created_at: "2026-10-11T09:00:00Z" });
  const b = result({ created_at: "2026-10-11T11:00:00Z", agent: "other" });
  const c = result({ created_at: "2026-10-11T12:00:00Z", config: { replay: "x.evals.json" } });
  const items: Array<[string, Result]> = [["a.json", a], ["b.json", b], ["c.json", c]];
  assert.equal(latest(items)?.[0], "b.json");
  assert.equal(latest(items, "my_agent")?.[0], "a.json");
  assert.equal(latest(items, "missing"), undefined);
});

test("fix prompt carries the evidence and how to prove the fix", () => {
  const p = fixPrompt(result(), ".halflife/run.json", "agents/claims.py");
  assert.match(p, /Reliability half-life: 32\.7 steps \(95% CI 28\.1–39\.0\)/);
  assert.match(p, /Top problem: Skipped a required step/);
  assert.match(p, /Add a completion check\./);
  assert.match(p, /get_claim\(\{"claim_id":"CLM-2"\}\)  → error: timeout/);
  assert.match(p, /show it to me as a diff before applying it/);
  assert.match(p, /halflife run agents\/claims\.py:my_agent --replay \.halflife\/run\.evals\.json/);
});

test("run view marks the breaking step", () => {
  const r = result();
  const md = runMarkdown(r, r.results[1]);
  assert.match(md, /\*\*First failure:\*\* CLM-1: member notified/);
  assert.match(md, /\*\*→ `8\. get_claim/);
});

test("parses a real result file from the CLI", () => {
  const dir = path.join(__dirname, "..", "..", "test", "fixtures");
  for (const f of fs.readdirSync(dir)) {
    const r = parseResult(fs.readFileSync(path.join(dir, f), "utf8"));
    assert.ok(r, f);
    assert.ok(halfLifeText(r!).length > 0);
    assert.ok(fixPrompt(r!, f).includes(r!.agent));
  }
  assert.equal(parseResult("{not json"), undefined);
  assert.equal(parseResult('{"agent": 1}'), undefined);
});

test("helpers", () => {
  assert.equal(evalPackPath(".halflife/x-2026.json"), ".halflife/x-2026.evals.json");
  assert.equal(shellQuote("examples/my_agent.py:my_agent"), "examples/my_agent.py:my_agent");
  assert.equal(shellQuote("my logs/a b.jsonl"), '"my logs/a b.jsonl"');
});
