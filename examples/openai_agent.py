"""Benchmark an OpenAI-compatible tool-calling agent (works with any provider
that speaks the Chat Completions tools format).

    pip install "halflife-ai[openai]"
    export OPENAI_API_KEY=...
    halflife run examples/openai_agent.py --lengths 5,10,20 --tasks 2 --runs 3
"""

import json
import os

from openai import OpenAI

from halflife import benchmark

client = OpenAI()
MODEL = os.environ.get("HALFLIFE_MODEL", "gpt-4.1")


@benchmark(suite="invoice-matching", lengths=[6, 10, 20, 40], tasks=2, runs=3)
def openai_agent(task):
    messages = [{"role": "user", "content": task.instructions}]
    while True:
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=task.openai_tools())
        msg = response.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return
        for call in msg.tool_calls:
            output = task.call(call.function.name, json.loads(call.function.arguments or "{}"))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(output)})
