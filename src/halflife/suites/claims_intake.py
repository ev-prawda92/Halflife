"""claims-intake: a health plan's claims queue.

The agent works a batch of medical claims against a simulated claims system:
check eligibility and network, apply coverage and prior-authorization rules,
compute payment with the member's deductible, then finalize and notify.

Why it gets harder with length: members recur across claims, and the deductible
shown is as of the start of the batch, so the agent must carry state across the
whole task. That is exactly where long-horizon agents slip.
"""

from __future__ import annotations

import random
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, List

from ..sandbox import Sandbox, ToolError, param, tool
from ..suite import suite
from ..task import Checkpoint, Constraint, TaskSpec
from ._money import money, same_money

CODES = {
    "99213": ("Office visit, established patient", 140.00),
    "99214": ("Office visit, moderate complexity", 210.00),
    "93000": ("Electrocardiogram", 95.00),
    "80053": ("Comprehensive metabolic panel", 60.00),
    "71046": ("Chest X-ray, two views", 120.00),
    "97110": ("Physical therapy, therapeutic exercise", 85.00),
    "J1885": ("Ketorolac injection", 45.00),
    "73721": ("MRI knee without contrast", 1450.00),
    "70553": ("MRI brain with and without contrast", 2600.00),
    "27447": ("Total knee replacement", 32000.00),
}
CODE_WEIGHTS = {"99213": 6, "99214": 4, "93000": 3, "80053": 4, "71046": 3, "97110": 3, "J1885": 2, "73721": 2, "70553": 1, "27447": 1}
PRIOR_AUTH = ["73721", "70553", "27447"]

PLANS = {
    "GOLD-PPO": {c: 90 for c in CODES},
    "SILVER-PPO": {c: 80 for c in CODES if c != "70553"},
    "BRONZE-HMO": {c: 70 for c in CODES if c not in ("97110", "70553", "27447")},
}

FIRST = ["Ana", "Ben", "Chloe", "Dev", "Elena", "Femi", "Grace", "Hiro", "Isla", "Jon", "Kemi", "Liam", "Maya", "Noor", "Owen", "Priya"]
LAST = ["Alvarez", "Brooks", "Chen", "Diallo", "Evans", "Fischer", "Gupta", "Hughes", "Ibrahim", "Jensen", "Kowalski", "Lopez"]
PRACTICES = ["Riverside Family Medicine", "Northgate Imaging", "Summit Orthopedics", "Lakeview Cardiology", "Elm Street Physical Therapy", "Harbor Labs", "Westside Urgent Care"]

INSTRUCTIONS = """You are a claims intake agent for a health plan. Work every claim in the queue, in queue order, using the tools provided. Within a claim, handle lines in order.

Policy
1. Eligibility. Look up the member. If the member's status is "terminated", deny every line on the claim with reason_code "INELIGIBLE" and paid_amount 0, then finalize and notify (total 0).
2. Network. Otherwise check the provider. If the provider is out of network, do not decide any lines: call flag_for_review with reason "OUT_OF_NETWORK" and move on. Do not finalize or notify that claim.
3. Coverage. Otherwise, for each line: if the member's plan does not cover the procedure code, deny with reason_code "NOT_COVERED".
4. Prior authorization. If the code is in the plan's prior_auth_required list and check_prior_auth does not return approved, deny with reason_code "NO_PRIOR_AUTH".
5. Payment. Otherwise approve. allowed = billed × coverage_pct / 100, rounded to cents. Apply the deductible: deductible portion = the smaller of the member's remaining deductible and the allowed amount; paid_amount = allowed − deductible portion.
   The member record shows the deductible remaining at the START of this batch. Subtract any deductible you already applied to that member's earlier lines and claims in this batch.
6. When every line on a claim is decided, call finalize_claim, then notify_member with the claim's total paid amount.

Round all money to 2 decimal places. When the queue is done, stop."""


class ClaimsSystem(Sandbox):
    def __init__(self, world: dict):
        self.w = world
        self.decisions: Dict[str, Dict[int, dict]] = defaultdict(dict)
        self.finalized: Dict[str, float] = {}
        self.flags: Dict[str, str] = {}
        self.notifications: List[dict] = []

    def _claim(self, claim_id: str) -> dict:
        claim = self.w["claims"].get(claim_id)
        if claim is None:
            raise ToolError(f"no claim with id {claim_id}")
        return claim

    @tool("List the claim ids in the queue, in the order they must be worked.")
    def list_claims(self):
        return {"claims": list(self.w["queue"])}

    @tool("Get a claim and its service lines.", claim_id=param.string("Claim id, e.g. CLM-10422"))
    def get_claim(self, claim_id):
        c = self._claim(claim_id)
        return {
            "claim_id": claim_id,
            "member_id": c["member_id"],
            "provider_npi": c["npi"],
            "service_date": c["service_date"],
            "lines": [
                {"line": i + 1, "code": ln["code"], "description": CODES[ln["code"]][0], "billed": ln["billed"]}
                for i, ln in enumerate(c["lines"])
            ],
        }

    @tool("Look up a member: plan, status, and deductible remaining at the start of this batch.", member_id=param.string("Member id"))
    def get_member(self, member_id):
        m = self.w["members"].get(member_id)
        if m is None:
            raise ToolError(f"no member with id {member_id}")
        return {
            "member_id": member_id,
            "name": m["name"],
            "plan_id": m["plan"],
            "status": "active" if m["active"] else "terminated",
            "deductible_remaining_at_batch_start": m["deductible"],
        }

    @tool("Get a plan's covered codes (with coverage percent) and codes requiring prior authorization.", plan_id=param.string("Plan id"))
    def get_plan(self, plan_id):
        p = PLANS.get(plan_id)
        if p is None:
            raise ToolError(f"no plan with id {plan_id}")
        return {"plan_id": plan_id, "covered": {c: {"coverage_pct": pct} for c, pct in p.items()}, "prior_auth_required": [c for c in PRIOR_AUTH if c in p]}

    @tool("Check whether a provider is in network.", npi=param.string("Provider NPI"))
    def check_provider(self, npi):
        pr = self.w["providers"].get(npi)
        if pr is None:
            raise ToolError(f"no provider with NPI {npi}")
        return {"npi": npi, "name": pr["name"], "in_network": pr["in_network"]}

    @tool("Check for an approved prior authorization.", member_id=param.string("Member id"), code=param.string("Procedure code"))
    def check_prior_auth(self, member_id, code):
        return {"member_id": member_id, "code": code, "approved": (member_id, code) in self.w["auths"]}

    @tool(
        "Record the decision for one claim line.",
        claim_id=param.string("Claim id"),
        line=param.integer("Line number, starting at 1"),
        decision=param.string("approve or deny", enum=["approve", "deny"]),
        paid_amount=param.number("Amount paid for this line; 0 when denied"),
        reason_code=param.string("Required when denying: INELIGIBLE, NOT_COVERED or NO_PRIOR_AUTH", optional=True),
    )
    def decide_line(self, claim_id, line, decision, paid_amount, reason_code=None):
        c = self._claim(claim_id)
        line = int(line)
        if claim_id in self.finalized:
            raise ToolError(f"{claim_id} is already finalized")
        if not 1 <= line <= len(c["lines"]):
            raise ToolError(f"{claim_id} has no line {line}")
        if decision not in ("approve", "deny"):
            raise ToolError("decision must be approve or deny")
        self.decisions[claim_id][line] = {"decision": decision, "paid": float(paid_amount), "reason": reason_code}
        return {"ok": True, "claim_id": claim_id, "line": line}

    @tool("Finalize a claim once every line is decided. Returns the total paid.", claim_id=param.string("Claim id"))
    def finalize_claim(self, claim_id):
        c = self._claim(claim_id)
        if claim_id in self.finalized:
            raise ToolError(f"{claim_id} is already finalized")
        missing = [i + 1 for i in range(len(c["lines"])) if (i + 1) not in self.decisions[claim_id]]
        if missing:
            raise ToolError(f"cannot finalize {claim_id}: undecided line(s) {missing}")
        total = money(sum(d["paid"] for d in self.decisions[claim_id].values()))
        self.finalized[claim_id] = total
        return {"ok": True, "claim_id": claim_id, "total_paid": total}

    @tool(
        "Send the member their claim notice.",
        member_id=param.string("Member id"),
        claim_id=param.string("Claim id"),
        total_paid=param.number("Total paid on the claim"),
    )
    def notify_member(self, member_id, claim_id, total_paid):
        self._claim(claim_id)
        if claim_id not in self.finalized:
            raise ToolError(f"{claim_id} is not finalized")
        self.notifications.append({"member_id": member_id, "claim_id": claim_id, "total": float(total_paid)})
        return {"ok": True}

    @tool("Route a claim to human review instead of deciding it.", claim_id=param.string("Claim id"), reason=param.string("Reason code, e.g. OUT_OF_NETWORK"))
    def flag_for_review(self, claim_id, reason):
        self._claim(claim_id)
        if claim_id in self.finalized:
            raise ToolError(f"{claim_id} is already finalized")
        self.flags[claim_id] = reason
        return {"ok": True}


def _allowed(billed: float, pct: int) -> float:
    return float((Decimal(str(billed)) * pct / 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def solve(world: dict) -> List[dict]:
    """The answer key: what a perfect agent does, claim by claim."""
    remaining = {mid: m["deductible"] for mid, m in world["members"].items()}
    out = []
    for cid in world["queue"]:
        c = world["claims"][cid]
        m = world["members"][c["member_id"]]
        if not m["active"]:
            lines = [{"decision": "deny", "paid": 0.0, "reason": "INELIGIBLE", "category": "eligibility"} for _ in c["lines"]]
            out.append({"claim_id": cid, "member_id": c["member_id"], "kind": "decide", "lines": lines, "total": 0.0})
            continue
        if not world["providers"][c["npi"]]["in_network"]:
            out.append({"claim_id": cid, "member_id": c["member_id"], "kind": "flag"})
            continue
        plan = PLANS[m["plan"]]
        lines = []
        for ln in c["lines"]:
            code = ln["code"]
            if code not in plan:
                lines.append({"decision": "deny", "paid": 0.0, "reason": "NOT_COVERED", "category": "coverage"})
            elif code in PRIOR_AUTH and (c["member_id"], code) not in world["auths"]:
                lines.append({"decision": "deny", "paid": 0.0, "reason": "NO_PRIOR_AUTH", "category": "prior_auth"})
            else:
                allowed = _allowed(ln["billed"], plan[code])
                ded = min(remaining[c["member_id"]], allowed)
                remaining[c["member_id"]] = money(remaining[c["member_id"]] - ded)
                stale = money(allowed - min(m["deductible"], allowed))
                lines.append({"decision": "approve", "paid": money(allowed - ded), "reason": None, "category": "payment", "deductible": ded, "stale": stale})
        out.append({"claim_id": cid, "member_id": c["member_id"], "kind": "decide", "lines": lines, "total": money(sum(x["paid"] for x in lines))})
    return out


def _generate(length: int, rng: random.Random) -> dict:
    n_members = max(2, min(12, length // 6 + 2))
    members = {}
    for _ in range(n_members):
        mid = f"M{rng.randint(10000, 99999)}"
        members[mid] = {
            "name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
            "plan": rng.choice(list(PLANS)),
            "active": rng.random() > 0.12,
            "deductible": rng.choice([0.0, 0.0, 0.0, 50.0, 100.0, 200.0, 400.0]),
        }
    providers = {}
    for name in rng.sample(PRACTICES, 5):
        providers[str(rng.randint(10**9, 10**10 - 1))] = {"name": name, "in_network": rng.random() > 0.15}
    auths = {(mid, code) for mid in members for code in PRIOR_AUTH if rng.random() < 0.55}

    claims, queue = {}, []
    world = {"members": members, "providers": providers, "auths": auths, "claims": claims, "queue": queue}
    count = 0
    codes, weights = list(CODE_WEIGHTS), list(CODE_WEIGHTS.values())
    while count < length:
        cid = f"CLM-{rng.randint(10000, 99999)}"
        if cid in claims:
            continue
        mid = rng.choice(list(members))
        npi = rng.choice(list(providers))
        n_lines = rng.choice([1, 1, 2, 2, 3])
        lines = []
        for code in rng.choices(codes, weights, k=n_lines):
            base = CODES[code][1]
            lines.append({"code": code, "billed": money(base * rng.uniform(0.85, 1.2))})
        claims[cid] = {"member_id": mid, "npi": npi, "service_date": f"2026-09-{rng.randint(1, 28):02d}", "lines": lines}
        queue.append(cid)
        count += 1 if (members[mid]["active"] and not providers[npi]["in_network"]) else n_lines + 2
    return world


def _line_checkpoint(cid: str, n: int, exp: dict) -> Checkpoint:
    def check(sb: ClaimsSystem) -> bool:
        got = sb.decisions.get(cid, {}).get(n)
        if not got or got["decision"] != exp["decision"]:
            return False
        if exp["decision"] == "deny":
            return got["reason"] == exp["reason"] and same_money(got["paid"], 0)
        return same_money(got["paid"], exp["paid"])

    def diagnose(sb: ClaimsSystem) -> str:
        got = sb.decisions.get(cid, {}).get(n)
        if not got:
            return "skipped_step"
        if exp["decision"] == "deny" and got["decision"] == "approve":
            return exp["category"]
        if exp["decision"] == "approve" and got["decision"] == "deny":
            return "false_denial"
        if exp["decision"] == "deny":
            return "wrong_reason"
        if exp.get("deductible", 0) > 0 or same_money(got["paid"], exp.get("stale")):
            return "deductible_tracking"
        return "payment_amount"

    what = f"deny {exp['reason']}" if exp["decision"] == "deny" else f"approve ${exp['paid']:.2f}"
    return Checkpoint(f"{cid} line {n}: {what}", exp["category"], check, diagnose)


def _checkpoints(expected: List[dict]) -> List[Checkpoint]:
    cps: List[Checkpoint] = []
    for e in expected:
        cid = e["claim_id"]
        if e["kind"] == "flag":
            cps.append(
                Checkpoint(
                    f"{cid}: flagged OUT_OF_NETWORK, not decided",
                    "network",
                    lambda sb, cid=cid: sb.flags.get(cid) == "OUT_OF_NETWORK" and not sb.decisions.get(cid) and cid not in sb.finalized,
                    lambda sb, cid=cid: "skipped_step" if cid not in sb.flags and not sb.decisions.get(cid) else "network",
                )
            )
            continue
        for n, exp in enumerate(e["lines"], 1):
            cps.append(_line_checkpoint(cid, n, exp))
        cps.append(Checkpoint(f"{cid}: finalized", "finalize", lambda sb, cid=cid: cid in sb.finalized, lambda sb: "skipped_step"))
        cps.append(
            Checkpoint(
                f"{cid}: member notified, total ${e['total']:.2f}",
                "notification",
                lambda sb, cid=cid, e=e: any(
                    x["claim_id"] == cid and x["member_id"] == e["member_id"] and same_money(x["total"], e["total"]) for x in sb.notifications
                ),
                lambda sb, cid=cid: "notification" if any(x["claim_id"] == cid for x in sb.notifications) else "skipped_step",
            )
        )
    return cps


def _constraints(world: dict) -> List[Constraint]:
    inactive = {mid for mid, m in world["members"].items() if not m["active"]}

    def no_dup_notices(sb: ClaimsSystem) -> bool:
        seen = [x["claim_id"] for x in sb.notifications]
        return len(seen) == len(set(seen))

    def no_pay_terminated(sb: ClaimsSystem) -> bool:
        for cid, lines in sb.decisions.items():
            if world["claims"][cid]["member_id"] in inactive and any(d["decision"] == "approve" for d in lines.values()):
                return False
        return True

    return [Constraint("No duplicate member notices", no_dup_notices), Constraint("Never pay a terminated member", no_pay_terminated)]


@suite("claims-intake", "Health plan claims queue: eligibility, network, coverage, prior auth, deductibles.", default_lengths=(5, 10, 20, 40))
def build(length: int, rng: random.Random) -> TaskSpec:
    world = _generate(length, rng)
    expected = solve(world)
    return TaskSpec(
        instructions=INSTRUCTIONS,
        sandbox=ClaimsSystem(world),
        checkpoints=_checkpoints(expected),
        constraints=_constraints(world),
        metadata={"claims": len(world["queue"])},
    )
