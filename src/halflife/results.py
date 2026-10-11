from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class RunResult:
    """The outcome of one agent run on one task."""

    task_id: str
    requested_length: int
    length: int
    run: int
    passed: bool
    checkpoints_passed: int
    progress: int
    first_failure: Optional[dict]
    violations: List[str]
    error: Optional[str]
    tool_calls: int
    seconds: float
    trace_tail: List[dict] = field(default_factory=list)
    task_index: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
