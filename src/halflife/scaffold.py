"""`halflife init`: scaffold an agent test, rules, and optionally a CI workflow."""

from __future__ import annotations

import os
from typing import List

AGENT = '''"""Your agent, under test.

Run it:      halflife run halflife_agent.py
In CI:       halflife run halflife_agent.py --min-half-life 20

Replace the stand-in with your own agent. The task gives you everything a tool-use loop needs:
    task.instructions        the goal and policy
    task.openai_tools()      tool definitions (OpenAI / Ollama / vLLM format)
    task.anthropic_tools()   tool definitions (Anthropic format)
    task.call(name, args)    execute a tool and get the result
"""

from halflife import benchmark
from halflife.demo import ClaimsDemoAgent


@benchmark(suite="{suite}", lengths=[5, 10, 20, 40], tasks=3, runs=3)
def my_agent(task):
    return ClaimsDemoAgent().run(task)
'''

RULES = '''"""Rules your agent must never break, on any input. Checked on every run's tool calls.

    halflife run halflife_agent.py --rules halflife_rules.py
    halflife audit your_logs.jsonl --rules halflife_rules.py
"""

from halflife.rules import at_most_once, covers, eventually, never, requires

RULES = [
{rules}
]
'''

SUITE_RULES = {
    "claims-intake": [
        'covers("list_claims", field="claims", actions=["finalize_claim", "flag_for_review"], key="claim_id")',
        'requires("finalize_claim", before="get_claim", match=["claim_id"])',
        'eventually("finalize_claim", then="notify_member", match=["claim_id"])',
        'at_most_once("notify_member", key=["claim_id"])',
    ],
    "invoice-matching": [
        'requires("approve_invoice", before="get_invoice", match=["invoice_id"])',
        'at_most_once("approve_invoice", key=["invoice_id"])',
        'never("approve_invoice", when=lambda a: float(a.get("amount", 0)) <= 0)',
    ],
}

WORKFLOW = """name: agent reliability

on:
  pull_request:

jobs:
  halflife:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install halflife-ai
      - run: halflife run halflife_agent.py --rules halflife_rules.py --min-half-life 20
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: halflife-report
          path: .halflife/
"""


def init(folder: str = ".", suite: str = "claims-intake", ci: bool = False) -> List[str]:
    files = {
        "halflife_agent.py": AGENT.format(suite=suite),
        "halflife_rules.py": RULES.format(rules="\n".join(f"    {r}," for r in SUITE_RULES.get(suite, []))),
    }
    if ci:
        files[os.path.join(".github", "workflows", "halflife.yml")] = WORKFLOW
    written = []
    for rel, content in files.items():
        path = os.path.join(folder, rel)
        if os.path.exists(path):
            continue
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        written.append(rel)
    return written
