"""Quickstart. Run it with:

    halflife run examples/my_agent.py

Swap the demo agent for your own. Anything with a .run(task) method, or any
function that takes a task, works.
"""

from halflife import benchmark
from halflife.demo import ClaimsDemoAgent

agent = ClaimsDemoAgent()


@benchmark(suite="claims-intake")
def my_agent(task):
    return agent.run(task)
