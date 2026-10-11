---
name: halflife
description: Test how reliable an AI agent is on long, multi-step tasks, find where and why it breaks, and fix it. Use whenever the user is building, changing, debugging, or about to ship a tool-using AI agent, asks whether an agent is reliable or ready, or shares agent logs or traces. Also use after editing an agent's prompt, tools, or code, to check the change didn't make it less reliable.
---

# halflife: agent reliability testing

halflife measures an agent's **reliability half-life**: the task length at which it succeeds half the time. It also reports where runs broke, ranked fixes, a recommended level of autonomy, and an eval pack of the failing cases that can be replayed after a fix.

## Setup

Check `halflife --version`. If it's missing, install it in this order, stopping at the first that works:

1. `pip install halflife-ai`
2. `pip install` the wheel bundled in this skill's folder: `pip install <this skill's folder>/halflife_ai-*.whl` (add `--break-system-packages` if pip refuses to install into the system Python)
3. `pip install git+https://github.com/ev-prawda92/Halflife`

halflife has no dependencies beyond Python 3.9+.

## Where it can run

- **In a chat with code execution:** audit uploaded logs, benchmark uploaded agent code, run the demo, write rules or a custom suite, and read result files the user uploads. Hosted model APIs and local servers like Ollama usually aren't reachable from here; if a model benchmark is needed, give the user the exact `halflife bench` command to run on their own machine.
- **In Claude Code or a terminal on the user's machine:** everything, including `halflife bench` against local or hosted models.

## Pick the entry point

1. **The user has agent logs** (JSON Lines of tool calls): audit them. No setup needed.
   - `halflife audit their_logs.jsonl --suggest-rules` mines rules from the logs into `rules_suggested.py` and audits with them.
   - Show the user the suggested rules and ask which are real policy. Mined rules describe what the agent usually does, not what it should do: delete coincidences, add what's missing (see Rules below), then `halflife audit their_logs.jsonl --rules rules_suggested.py`.
   - A suggested rule that broke in a few runs is the first thing to investigate: either the agent failed in those runs or the rule is wrong.
2. **The user has agent code**: benchmark it.
   - Run `halflife init` to scaffold `halflife_agent.py` and `halflife_rules.py`.
   - Edit `halflife_agent.py` so `my_agent(task)` calls the user's agent. The task provides `task.instructions`, `task.openai_tools()` or `task.anthropic_tools()`, and `task.call(name, args)` to execute tools. Adapt the user's tool-use loop to these; don't rewrite their agent.
   - Run `halflife run halflife_agent.py --quick` while iterating. Use full runs (no `--quick`) before shipping.
3. **The user wants to compare models**: `halflife bench --model PROVIDER:MODEL --quick`, repeating `--model` to compare. Providers: ollama, lmstudio, vllm, openai, openrouter, together, groq, anthropic, compatible (with `--base-url`). Remind the user that Ollama needs `OLLAMA_CONTEXT_LENGTH=32768 ollama serve` or long tasks fail for the wrong reason.

Use `halflife show --suite claims-intake --length 10` to see exactly what an agent receives.

## The fix loop

1. Run the benchmark. Read the summary: half-life, decay shape, where runs broke, and **What to fix first**.
2. Explain the result to the user in plain terms. Lead with the half-life and the top failure mode.
3. Propose the top fix. Prefer structural fixes (move a rule into a tool or code, keep running totals out of the conversation, add a completion check in the harness, split long work into chunks under the half-life) over prompt wording. Make the change only with the user's agreement.
4. Replay exactly the cases that failed: `halflife run halflife_agent.py --replay .halflife/<name>.evals.json`. The eval pack path is printed after each run.
5. Re-run the full benchmark. The output shows the change since the last run. Report whether the half-life improved.

## Rules

Rules are invariants checked on every run's tool calls, with no answer key needed:

```python
from halflife.rules import at_most_once, covers, eventually, never, requires

RULES = [
    covers("list_items", field="items", actions=["close_item"], key="item_id"),
    requires("close_item", before="get_item", match=["item_id"]),
    eventually("close_item", then="notify", match=["item_id"]),
    at_most_once("notify", key=["item_id"]),
    never("refund", when=lambda a: a["amount"] > 500),
]
```

Base rules on the user's real policies. Ask what must never happen and what must always be finished.

## Shipping

- To gate merges: `halflife run halflife_agent.py --min-half-life 20` exits 1 below the threshold. `halflife init --ci` adds a GitHub Actions workflow.
- To recommend autonomy for a real workflow: add `--workflow-length N`.
- For a README badge: `halflife badge`.

## Honesty rules

- The gate only certifies lengths that were tested. If the user's workflow is longer than the longest tested task, say so and test longer lengths.
- Simulated demo agents (`halflife demo`) are for trying the tool. Never present their numbers as results about a real model.
- Small samples give wide intervals. Report the 95% interval alongside the half-life.
