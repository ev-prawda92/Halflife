"""What to fix: concrete remediation steps for each failure mode.

Each failure mode maps to the change most likely to remove it. The ordering of
fixes is deliberate: structural fixes (move the rule into code or a tool) come
before prompt fixes, because prompts are the part that decays with length.
"""

from __future__ import annotations

from typing import Dict, List, Optional

STATE_TRACKING = {
    "why": "The agent has to remember a running total from many steps earlier, and that memory degrades as the task grows.",
    "fixes": [
        "Keep running totals in code or a ledger tool, not in the conversation. Give the agent a tool that returns the current remaining amount.",
        "Have the agent re-read authoritative state right before each action that depends on it.",
        "Group work by entity (member, purchase order) so dependent items are processed together, close in context.",
        "Add a verification pass before finalizing that recomputes totals from the ledger and corrects mismatches.",
    ],
}

MISSED_RULE = {
    "why": "A required check was skipped or forgotten, so an item went through that the policy should have stopped.",
    "fixes": [
        "Enforce the check in the tool: make the action tool refuse unless the prerequisite check was called for that item.",
        "Give the agent a per-item checklist in rule order and require it to state which rule applied before acting.",
        "Add targeted evals for this rule (export the failing cases with --export) and re-run them after each change.",
    ],
}

REMEDIES: Dict[str, dict] = {
    "deductible_tracking": STATE_TRACKING,
    "receipt_tracking": STATE_TRACKING,
    "duplicate_in_batch": {
        "why": "The agent forgot an item it had already seen earlier in the batch.",
        "fixes": [
            "Track seen identifiers in code, or expose a check_duplicate tool backed by the batch so far.",
            "Run a batch-level duplicate scan before working items one by one.",
            "Add a final audit step that lists all approved items and flags repeats.",
        ],
    },
    "skipped_step": {
        "why": "The agent stopped early or skipped a required step, most often a closing step like finalizing or notifying.",
        "fixes": [
            "Add a completion contract in the harness: after the agent stops, check every item reached its required end state and send it back with the gaps.",
            "Split long queues into chunks shorter than the half-life, each in a fresh context.",
            "Ask for a plan listing every item and required step up front, then tick items off as it goes.",
        ],
    },
    "budget_exceeded": {
        "why": "The agent looped: it kept calling tools without making progress.",
        "fixes": [
            "Detect repeated identical tool calls in the harness and interrupt with a corrective message.",
            "Cap retries per item and move on, flagging the item for review.",
            "Make tool error messages say exactly what to change, so the agent can recover instead of retrying.",
        ],
    },
    "agent_error": {
        "why": "The agent crashed, usually on an API error, a timeout, or malformed tool arguments.",
        "fixes": [
            "Add retries with backoff around model calls and handle rate limits.",
            "Validate and repair tool-call JSON before executing it.",
            "Check the failed runs' error messages in the report for the specific cause.",
        ],
    },
    "constraint_violation": {
        "why": "The agent broke a hard rule that must never be broken.",
        "fixes": [
            "Enforce hard rules in the tools themselves so the action is impossible, not just discouraged.",
            "Treat any violation as a release blocker in CI.",
        ],
    },
    "payment_amount": {
        "why": "The arithmetic was wrong.",
        "fixes": [
            "Move calculation into a calculator tool or code; let the agent supply inputs, not results.",
            "Validate amounts in the action tool and reject values that don't reconcile.",
        ],
    },
    "due_date": {
        "why": "Date arithmetic was wrong.",
        "fixes": [
            "Compute dates in code or a tool rather than in the model.",
            "Return the computed due date from the tool so the agent only confirms it.",
        ],
    },
    "wrong_reason": {
        "why": "The action was right but the reason code was wrong, which breaks downstream reporting and appeals.",
        "fixes": [
            "Constrain reason codes with an enum in the tool schema.",
            "Map each policy rule to exactly one reason code in the instructions.",
        ],
    },
    "false_denial": {
        "why": "The agent was over-cautious and denied a valid item.",
        "fixes": [
            "Add worked examples of clean items that must be approved.",
            "Require the agent to cite the specific rule that triggered a denial; deny only if it can.",
        ],
    },
    "false_flag": {
        "why": "The agent held or rejected a clean item.",
        "fixes": [
            "State tolerances explicitly (for example, the 2% price band) and give the comparison to a tool.",
            "Require a cited rule for every hold or rejection.",
        ],
    },
    "notification": {
        "why": "The closing notice was missing or had the wrong total.",
        "fixes": [
            "Have the finalize tool return the total and pass it straight through to the notice.",
            "Add the notice to the per-item completion contract.",
        ],
    },
    "finalize": {
        "why": "Items were left open.",
        "fixes": ["Add finalizing to the per-item completion contract, checked by the harness."],
    },
}

for _cat in ("eligibility", "network", "coverage", "prior_auth", "vendor", "duplicate_history", "price", "quantity", "missed_check"):
    REMEDIES[_cat] = MISSED_RULE

REMEDIES["duplicate_action"] = REMEDIES["duplicate_in_batch"]
REMEDIES["ignored_error"] = {
    "why": "A tool returned an error and the agent sent the identical call again instead of changing course.",
    "fixes": [
        "Make tool errors say what to change, in one line the model can act on.",
        "In the harness, block an identical retry of a failed call and ask the agent to adjust.",
        "Add a recovery instruction: after an error, re-read the relevant state before trying again.",
    ],
}

FALLBACK = {
    "why": "See the failed runs in the report for the pattern.",
    "fixes": ["Export the failing cases with --export, inspect the traces, and re-run them after each change."],
}


def for_category(category: str) -> dict:
    return REMEDIES.get(category, FALLBACK)


def structural(shape: Optional[float], half_life: Optional[float], workflow_length: Optional[int]) -> List[str]:
    """Advice that follows from the shape of the curve rather than one failure mode."""
    tips: List[str] = []
    if shape is not None and shape > 1.15:
        tips.append(
            "Errors compound with length. The highest-leverage fix is shortening the horizon: split work into chunks "
            "well under the half-life, each with fresh context and state passed in explicitly."
        )
    if shape is not None and shape < 0.85:
        tips.append("Failures come early. Look at setup steps and the first items: unclear instructions or tool errors are likely causes.")
    if half_life and workflow_length and half_life < workflow_length:
        chunk = max(1, int(half_life * 0.4))
        tips.append(
            f"Your workflow ({workflow_length} steps) is longer than the half-life ({half_life:.0f}). "
            f"Chunks of about {chunk} steps would keep each piece in the reliable range."
        )
    return tips


TIERS = [
    (0.95, "Act autonomously", "Reliable at your workflow length. Keep monitoring and sample-audit."),
    (0.80, "Act with approval", "Usually right; a person approves before actions take effect."),
    (0.50, "Suggest only", "Right more often than not, but a person should make each decision."),
    (0.00, "Observe only (shadow mode)", "Run alongside people without acting until reliability improves."),
]


def autonomy(success_at_workflow: float) -> dict:
    for threshold, tier, meaning in TIERS:
        if success_at_workflow >= threshold:
            return {"tier": tier, "meaning": meaning, "threshold": threshold}
    return {"tier": TIERS[-1][1], "meaning": TIERS[-1][2], "threshold": 0.0}
