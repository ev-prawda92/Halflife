"""A small claims-processing agent to try halflife on. It has bugs on purpose.

It works through a batch of insurance claims with tools (look up the claim, the
member and their plan, decide each line, finalize, notify the member). On short
batches it looks fine. On long ones it quietly gets things wrong, the way real
agents do. Run it, see where it breaks, fix it, and watch the half-life go up.

    halflife run halflife_demo_agent.py --quick

This one is plain Python so it runs without an API key. Your real agent would
call a model in a loop; halflife tests it the same way.
"""

from halflife import benchmark

MAX_CLAIMS = 8  # keep runs fast


def money(x):
    return round(x + 1e-9, 2)


@benchmark(suite="claims-intake")
def claims_agent(task):
    members, plans, providers = {}, {}, {}

    for claim_id in task.call("list_claims")["claims"][:MAX_CLAIMS]:
        claim = task.call("get_claim", claim_id=claim_id)
        member_id = claim["member_id"]
        if member_id not in members:
            members[member_id] = task.call("get_member", member_id=member_id)
        member = members[member_id]

        if member["status"] == "terminated":
            for line in claim["lines"]:
                task.call("decide_line", claim_id=claim_id, line=line["line"], decision="deny", paid_amount=0, reason_code="INELIGIBLE")
            task.call("finalize_claim", claim_id=claim_id)
            task.call("notify_member", member_id=member_id, claim_id=claim_id, total_paid=0)
            continue

        npi = claim["provider_npi"]
        if npi not in providers:
            providers[npi] = task.call("check_provider", npi=npi)
        if not providers[npi]["in_network"]:
            task.call("flag_for_review", claim_id=claim_id, reason="OUT_OF_NETWORK")
            continue

        plan_id = member["plan_id"]
        if plan_id not in plans:
            plans[plan_id] = task.call("get_plan", plan_id=plan_id)
        plan = plans[plan_id]

        total_paid = 0.0
        deductible_used = 0.0
        for line in claim["lines"]:
            coverage = plan["covered"].get(line["code"])
            if coverage is None:
                task.call("decide_line", claim_id=claim_id, line=line["line"], decision="deny", paid_amount=0, reason_code="NOT_COVERED")
                continue
            if line["code"] in plan["prior_auth_required"]:
                if not task.call("check_prior_auth", member_id=member_id, code=line["code"])["approved"]:
                    task.call("decide_line", claim_id=claim_id, line=line["line"], decision="deny", paid_amount=0, reason_code="NO_PRIOR_AUTH")
                    continue
            allowed = money(line["billed"] * coverage["coverage_pct"] / 100)
            deductible_left = member["deductible_remaining_at_batch_start"] - deductible_used
            deductible = min(max(deductible_left, 0), allowed)
            deductible_used += deductible
            paid = money(allowed - deductible)
            total_paid += paid
            task.call("decide_line", claim_id=claim_id, line=line["line"], decision="approve", paid_amount=paid)

        task.call("finalize_claim", claim_id=claim_id)
        task.call("notify_member", member_id=member_id, claim_id=claim_id, total_paid=money(total_paid))

    return "done"
