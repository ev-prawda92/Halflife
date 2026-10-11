"""Benchmark a real Claude tool-use agent.

    pip install "halflife-ai[anthropic]"
    export ANTHROPIC_API_KEY=...
    halflife run examples/claude_agent.py --lengths 5,10,20 --tasks 2 --runs 3

The task hands you everything a tool-use loop needs: instructions, tool
definitions in Anthropic's format, and task.call() to execute a tool.
"""

import json
import os

import anthropic

from halflife import benchmark

client = anthropic.Anthropic()
MODEL = os.environ.get("HALFLIFE_MODEL", "claude-sonnet-5-5")


@benchmark(suite="claims-intake", lengths=[5, 10, 20, 40], tasks=2, runs=3)
def claude_agent(task):
    messages = [{"role": "user", "content": task.instructions}]
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            tools=task.anthropic_tools(),
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = task.call(block.name, block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(output)})
        messages.append({"role": "user", "content": results})
