"""halflife: measure how long a task you can trust an AI agent with.

    from halflife import benchmark

    @benchmark(suite="claims-intake")
    def my_agent(task):
        return agent.run(task)

Then run `halflife run my_agent.py`, or `my_agent.evaluate()` from Python.
"""

__version__ = "0.3.0"

from .decorator import assert_reliable, benchmark
from .report import Report
from .runner import build_task, evaluate, grade
from .sandbox import Sandbox, ToolBudgetExceeded, ToolError, param, tool
from .suite import get_suite, list_suites, suite
from .task import Checkpoint, Constraint, Task, TaskSpec
from .traces import TraceWriter, audit
from . import rules

__all__ = [
    "benchmark",
    "assert_reliable",
    "evaluate",
    "Report",
    "Task",
    "TaskSpec",
    "Checkpoint",
    "Constraint",
    "Sandbox",
    "tool",
    "param",
    "ToolError",
    "ToolBudgetExceeded",
    "suite",
    "get_suite",
    "list_suites",
    "build_task",
    "grade",
    "rules",
    "audit",
    "TraceWriter",
]
