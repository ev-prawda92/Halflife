"""Rules for a claims-processing agent. No answer key needed: these must hold on every run.

    halflife audit examples/traces/claims.jsonl --rules examples/rules_claims.py
    halflife bench --model ollama:qwen2.5:7b --rules examples/rules_claims.py
"""

from halflife.rules import at_most_once, covers, eventually, never, requires

RULES = [
    covers("list_claims", field="claims", actions=["finalize_claim", "flag_for_review"], key="claim_id"),
    requires("decide_line", before="get_claim", match=["claim_id"]),
    requires("finalize_claim", before="get_claim", match=["claim_id"]),
    eventually("finalize_claim", then="notify_member", match=["claim_id"]),
    at_most_once("notify_member", key=["claim_id"]),
    at_most_once("finalize_claim", key=["claim_id"]),
    never("decide_line", when=lambda a: a.get("decision") == "approve" and float(a.get("paid_amount", 0)) < 0, message="negative payment"),
]
