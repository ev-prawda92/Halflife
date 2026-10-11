"""Simulated agents for trying halflife without an API key.

These agents follow each suite's policy through the tools, like a real agent
would, but slip with a probability that grows as the task gets longer. That
mimics how real agents degrade over long horizons: forgetting state from
earlier in the task, skipping a check, or quietly stopping.

They are stand-ins. Swap in your real agent to measure it.
"""

from __future__ import annotations

import datetime as dt
import random
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

from .task import Task


def _money(x) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


class _Fatigue:
    def __init__(self, base: float, growth: float, seed: Optional[int]):
        self.rng = random.Random(seed)
        self.base = base
        self.growth = growth
        self.step = 0

    def slip(self) -> bool:
        self.step += 1
        return self.rng.random() < self.base + self.growth * self.step


class ClaimsDemoAgent:
    """Works the claims-intake policy, with simulated long-horizon slips."""

    def __init__(self, base_error: float = 0.003, fatigue: float = 0.002, seed: Optional[int] = None):
        self.base_error = base_error
        self.fatigue = fatigue
        self.seed = seed

    def run(self, task: Task) -> str:
        f = _Fatigue(self.base_error, self.fatigue, self.seed)
        used = defaultdict(float)
        members, plans, providers = {}, {}, {}
        for cid in task.call("list_claims")["claims"]:
            if f.slip() and f.rng.random() < 0.25:
                return "stopped early"
            claim = task.call("get_claim", claim_id=cid)
            mid = claim["member_id"]
            if mid not in members:
                members[mid] = task.call("get_member", member_id=mid)
            m = members[mid]
            if m["status"] == "terminated":
                for ln in claim["lines"]:
                    task.call("decide_line", claim_id=cid, line=ln["line"], decision="deny", paid_amount=0, reason_code="INELIGIBLE")
                task.call("finalize_claim", claim_id=cid)
                task.call("notify_member", member_id=mid, claim_id=cid, total_paid=0)
                continue
            npi = claim["provider_npi"]
            if npi not in providers:
                providers[npi] = task.call("check_provider", npi=npi)
            if not providers[npi]["in_network"]:
                task.call("flag_for_review", claim_id=cid, reason="OUT_OF_NETWORK")
                continue
            pid = m["plan_id"]
            if pid not in plans:
                plans[pid] = task.call("get_plan", plan_id=pid)
            plan = plans[pid]
            total = 0.0
            for ln in claim["lines"]:
                code = ln["code"]
                cov = plan["covered"].get(code)
                if cov is None:
                    task.call("decide_line", claim_id=cid, line=ln["line"], decision="deny", paid_amount=0, reason_code="NOT_COVERED")
                    continue
                if code in plan["prior_auth_required"]:
                    approved = True if f.slip() else task.call("check_prior_auth", member_id=mid, code=code)["approved"]
                    if not approved:
                        task.call("decide_line", claim_id=cid, line=ln["line"], decision="deny", paid_amount=0, reason_code="NO_PRIOR_AUTH")
                        continue
                allowed = _money(ln["billed"] * cov["coverage_pct"] / 100)
                start = m["deductible_remaining_at_batch_start"]
                remaining = start if f.slip() else start - used[mid]
                ded = min(max(remaining, 0), allowed)
                used[mid] += min(max(start - used[mid], 0), allowed)
                paid = _money(allowed - ded)
                total += paid
                task.call("decide_line", claim_id=cid, line=ln["line"], decision="approve", paid_amount=paid)
            task.call("finalize_claim", claim_id=cid)
            if not f.slip():
                task.call("notify_member", member_id=mid, claim_id=cid, total_paid=_money(total))
        return "done"


class InvoiceDemoAgent:
    """Works the invoice-matching policy, with simulated long-horizon slips."""

    def __init__(self, base_error: float = 0.004, fatigue: float = 0.005, seed: Optional[int] = None):
        self.base_error = base_error
        self.fatigue = fatigue
        self.seed = seed

    def run(self, task: Task) -> str:
        f = _Fatigue(self.base_error, self.fatigue, self.seed)
        approved = defaultdict(int)
        seen = set()
        vendors, history, pos, receipts = {}, {}, {}, {}
        for iid in task.call("list_invoices")["invoices"]:
            if f.slip() and f.rng.random() < 0.2:
                return "stopped early"
            inv = task.call("get_invoice", invoice_id=iid)
            vid, number = inv["vendor_id"], inv["invoice_number"]
            if vid not in vendors:
                vendors[vid] = task.call("get_vendor", vendor_id=vid)
                history[vid] = set(task.call("get_payment_history", vendor_id=vid)["paid_invoice_numbers"])
            key = (vid, number)
            remembers = not f.slip()
            if vendors[vid]["status"] == "blocked":
                task.call("reject_invoice", invoice_id=iid, reason="VENDOR_BLOCKED")
            elif number in history[vid] or (remembers and key in seen):
                task.call("reject_invoice", invoice_id=iid, reason="DUPLICATE")
            else:
                po_id = inv["po_id"]
                if po_id not in pos:
                    pos[po_id] = {ln["po_line"]: ln for ln in task.call("get_purchase_order", po_id=po_id)["lines"]}
                    receipts[po_id] = {r["po_line"]: r["qty_received"] for r in task.call("get_receipts", po_id=po_id)["receipts"]}
                variance = any(abs(ln["unit_price"] - pos[po_id][ln["po_line"]]["unit_price"]) / pos[po_id][ln["po_line"]]["unit_price"] > 0.02 for ln in inv["lines"])
                tracks = not f.slip()
                over = any(
                    ln["qty"] > receipts[po_id][ln["po_line"]] - (approved[(po_id, ln["po_line"])] if tracks else 0)
                    for ln in inv["lines"]
                )
                if variance:
                    task.call("hold_invoice", invoice_id=iid, reason="PRICE_VARIANCE")
                elif over:
                    task.call("hold_invoice", invoice_id=iid, reason="QTY_EXCEEDS_RECEIPT")
                else:
                    for ln in inv["lines"]:
                        approved[(po_id, ln["po_line"])] += ln["qty"]
                    amount = _money(sum(ln["qty"] * ln["unit_price"] for ln in inv["lines"]))
                    due = dt.date.fromisoformat(inv["invoice_date"]) + dt.timedelta(days=vendors[vid]["payment_terms_days"])
                    task.call("approve_invoice", invoice_id=iid, amount=amount, due_date=due.isoformat())
            seen.add(key)
        return "done"


def demo_agent_for(suite: str):
    if suite == "claims-intake":
        return ClaimsDemoAgent()
    if suite == "invoice-matching":
        return InvoiceDemoAgent()
    raise KeyError(f"no demo agent for suite '{suite}'")
