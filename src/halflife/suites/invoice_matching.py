"""invoice-matching: accounts payable three-way match.

The agent works a batch of vendor invoices against a simulated AP system:
screen out blocked vendors and duplicates, match each line to the purchase
order (price) and goods receipt (quantity), then approve with the right amount
and due date, or hold or reject with the right reason.

Why it gets harder with length: duplicates can repeat an invoice from earlier
in the batch, and receipts are consumed by invoices approved earlier in the
batch, so the agent has to carry state across the whole task.
"""

from __future__ import annotations

import datetime as dt
import math
import random
from collections import defaultdict
from typing import Dict, List

from ..sandbox import Sandbox, ToolError, param, tool
from ..suite import suite
from ..task import Checkpoint, Constraint, TaskSpec
from ._money import money, same_money

SKUS = {
    "STL-0412": ("Steel bracket, 4 in", 3.75),
    "CBL-1100": ("Cat6 cable, 100 ft", 42.00),
    "PLT-4048": ("Pallet, 40 x 48", 18.50),
    "LBL-9000": ("Shipping labels, 1000 ct", 27.00),
    "GLV-0300": ("Nitrile gloves, case", 64.00),
    "TNR-7700": ("Toner cartridge", 89.00),
    "BRG-6205": ("Ball bearing 6205", 6.20),
    "FLT-1416": ("Air filter 14 x 16", 11.40),
}
VENDOR_NAMES = ["Allegheny Supply Co", "Brightline Packaging", "Cedar Industrial", "Delta Office Products", "Everfield Logistics", "Foundry Fasteners", "Granite Safety", "Hudson Electrical"]
PRICE_TOLERANCE = 0.02

INSTRUCTIONS = """You are an accounts payable agent. Work every invoice in the queue, in queue order, and make exactly one decision per invoice using the tools provided.

Policy (apply in this order; the first rule that applies decides the invoice)
1. Blocked vendor. If the vendor's status is "blocked", reject with reason "VENDOR_BLOCKED".
2. Duplicate. If the vendor's payment history already contains this invoice_number, or an invoice with the same vendor and invoice_number appeared earlier in this batch, reject with reason "DUPLICATE".
3. Price. If any line's unit_price differs from the purchase order's unit_price by more than 2%, hold with reason "PRICE_VARIANCE".
4. Quantity. For each line, the quantity available to bill is qty_received on that PO line minus the quantity on that PO line you already approved earlier in this batch. If any line's qty exceeds what is available, hold with reason "QTY_EXCEEDS_RECEIPT".
5. Otherwise approve. amount = sum of qty × unit_price over the invoice lines (invoice prices), rounded to cents. due_date = invoice_date + the vendor's payment_terms_days, formatted YYYY-MM-DD.

When the queue is done, stop."""


class APSystem(Sandbox):
    def __init__(self, world: dict):
        self.w = world
        self.decisions: Dict[str, dict] = {}

    def _invoice(self, invoice_id: str) -> dict:
        inv = self.w["invoices"].get(invoice_id)
        if inv is None:
            raise ToolError(f"no invoice with id {invoice_id}")
        return inv

    def _decide(self, invoice_id: str, record: dict) -> dict:
        self._invoice(invoice_id)
        if invoice_id in self.decisions:
            raise ToolError(f"{invoice_id} already has a decision")
        self.decisions[invoice_id] = record
        return {"ok": True, "invoice_id": invoice_id}

    @tool("List the invoice ids in the queue, in the order they must be worked.")
    def list_invoices(self):
        return {"invoices": list(self.w["queue"])}

    @tool("Get an invoice and its lines.", invoice_id=param.string("Invoice id, e.g. AP-30211"))
    def get_invoice(self, invoice_id):
        inv = self._invoice(invoice_id)
        return {
            "invoice_id": invoice_id,
            "vendor_id": inv["vendor_id"],
            "invoice_number": inv["number"],
            "po_id": inv["po_id"],
            "invoice_date": inv["date"],
            "lines": [{"po_line": ln["po_line"], "sku": ln["sku"], "qty": ln["qty"], "unit_price": ln["price"]} for ln in inv["lines"]],
        }

    @tool("Look up a vendor: status and payment terms.", vendor_id=param.string("Vendor id"))
    def get_vendor(self, vendor_id):
        v = self.w["vendors"].get(vendor_id)
        if v is None:
            raise ToolError(f"no vendor with id {vendor_id}")
        return {"vendor_id": vendor_id, "name": v["name"], "status": v["status"], "payment_terms_days": v["terms"]}

    @tool("Get a purchase order and its lines.", po_id=param.string("Purchase order id"))
    def get_purchase_order(self, po_id):
        po = self.w["pos"].get(po_id)
        if po is None:
            raise ToolError(f"no purchase order {po_id}")
        return {
            "po_id": po_id,
            "vendor_id": po["vendor_id"],
            "lines": [{"po_line": i + 1, "sku": ln["sku"], "description": SKUS[ln["sku"]][0], "qty_ordered": ln["qty"], "unit_price": ln["price"]} for i, ln in enumerate(po["lines"])],
        }

    @tool("Get goods receipts for a purchase order.", po_id=param.string("Purchase order id"))
    def get_receipts(self, po_id):
        po = self.w["pos"].get(po_id)
        if po is None:
            raise ToolError(f"no purchase order {po_id}")
        return {"po_id": po_id, "receipts": [{"po_line": i + 1, "qty_received": ln["received"]} for i, ln in enumerate(po["lines"])]}

    @tool("Get invoice numbers previously paid to a vendor.", vendor_id=param.string("Vendor id"))
    def get_payment_history(self, vendor_id):
        if vendor_id not in self.w["vendors"]:
            raise ToolError(f"no vendor with id {vendor_id}")
        return {"vendor_id": vendor_id, "paid_invoice_numbers": list(self.w["history"][vendor_id])}

    @tool(
        "Approve an invoice for payment.",
        invoice_id=param.string("Invoice id"),
        amount=param.number("Amount to pay"),
        due_date=param.string("Due date, YYYY-MM-DD"),
    )
    def approve_invoice(self, invoice_id, amount, due_date):
        return self._decide(invoice_id, {"action": "approve", "amount": float(amount), "due_date": str(due_date)})

    @tool("Put an invoice on hold.", invoice_id=param.string("Invoice id"), reason=param.string("PRICE_VARIANCE or QTY_EXCEEDS_RECEIPT"))
    def hold_invoice(self, invoice_id, reason):
        return self._decide(invoice_id, {"action": "hold", "reason": reason})

    @tool("Reject an invoice.", invoice_id=param.string("Invoice id"), reason=param.string("VENDOR_BLOCKED or DUPLICATE"))
    def reject_invoice(self, invoice_id, reason):
        return self._decide(invoice_id, {"action": "reject", "reason": reason})


def solve(world: dict) -> List[dict]:
    approved_qty: Dict[tuple, int] = defaultdict(int)
    seen = set()
    out = []
    for iid in world["queue"]:
        inv = world["invoices"][iid]
        vid = inv["vendor_id"]
        v = world["vendors"][vid]
        po = world["pos"][inv["po_id"]]
        key = (vid, inv["number"])
        if v["status"] == "blocked":
            exp = {"action": "reject", "reason": "VENDOR_BLOCKED", "category": "vendor"}
        elif inv["number"] in world["history"][vid]:
            exp = {"action": "reject", "reason": "DUPLICATE", "category": "duplicate_history"}
        elif key in seen:
            exp = {"action": "reject", "reason": "DUPLICATE", "category": "duplicate_in_batch"}
        elif any(abs(ln["price"] - po["lines"][ln["po_line"] - 1]["price"]) / po["lines"][ln["po_line"] - 1]["price"] > PRICE_TOLERANCE for ln in inv["lines"]):
            exp = {"action": "hold", "reason": "PRICE_VARIANCE", "category": "price"}
        else:
            over = None
            for ln in inv["lines"]:
                received = po["lines"][ln["po_line"] - 1]["received"]
                available = received - approved_qty[(inv["po_id"], ln["po_line"])]
                if ln["qty"] > available:
                    over = "quantity" if ln["qty"] > received else "receipt_tracking"
                    break
            if over:
                exp = {"action": "hold", "reason": "QTY_EXCEEDS_RECEIPT", "category": over}
            else:
                for ln in inv["lines"]:
                    approved_qty[(inv["po_id"], ln["po_line"])] += ln["qty"]
                due = dt.date.fromisoformat(inv["date"]) + dt.timedelta(days=v["terms"])
                exp = {"action": "approve", "amount": money(sum(ln["qty"] * ln["price"] for ln in inv["lines"])), "due_date": due.isoformat(), "category": "payment"}
        seen.add(key)
        exp["invoice_id"] = iid
        out.append(exp)
    return out


def _generate(length: int, rng: random.Random) -> dict:
    n_invoices = max(2, math.ceil(length / 2))
    vendors, pos, history = {}, {}, {}
    for name in rng.sample(VENDOR_NAMES, min(len(VENDOR_NAMES), max(3, n_invoices // 4 + 2))):
        vid = f"V-{rng.randint(1000, 9999)}"
        vendors[vid] = {"name": name, "status": "blocked" if rng.random() < 0.12 else "active", "terms": rng.choice([15, 30, 45, 60])}
        history[vid] = [f"INV-{rng.randint(10000, 99999)}" for _ in range(rng.randint(1, 3))]
        for _ in range(rng.randint(1, 2)):
            po_id = f"PO-{rng.randint(40000, 49999)}"
            lines = []
            for sku in rng.sample(list(SKUS), rng.randint(2, 3)):
                qty = rng.randint(4, 30) * 5
                received = qty if rng.random() > 0.3 else qty - rng.randint(1, qty // 5) * 5
                lines.append({"sku": sku, "qty": qty, "price": money(SKUS[sku][1] * rng.uniform(0.95, 1.05)), "received": received})
            pos[po_id] = {"vendor_id": vid, "lines": lines}

    planned: Dict[tuple, int] = defaultdict(int)
    invoices, queue, numbers = {}, [], []
    po_ids = list(pos)
    while len(queue) < n_invoices:
        iid = f"AP-{rng.randint(30000, 39999)}"
        if iid in invoices:
            continue
        po_id = rng.choice(po_ids)
        po = pos[po_id]
        vid = po["vendor_id"]
        number = f"INV-{rng.randint(10000, 99999)}"
        chosen = rng.sample(range(len(po["lines"])), rng.randint(1, min(2, len(po["lines"]))))
        lines = []
        for i in chosen:
            pl = po["lines"][i]
            left = pl["received"] - planned[(po_id, i + 1)]
            qty = rng.randint(1, max(1, left // 5)) * 5 if left >= 5 else rng.randint(1, 4) * 5
            lines.append({"po_line": i + 1, "sku": pl["sku"], "qty": qty, "price": pl["price"]})
        scenario = rng.random()
        if scenario < 0.10:
            number = rng.choice(history[vid])
        elif scenario < 0.20:
            earlier = [n for (v2, n) in numbers if v2 == vid]
            if earlier:
                number = rng.choice(earlier)
        elif scenario < 0.31:
            ln = rng.choice(lines)
            ln["price"] = money(ln["price"] * rng.uniform(1.04, 1.10))
        elif scenario < 0.41:
            ln = rng.choice(lines)
            ln["qty"] = po["lines"][ln["po_line"] - 1]["received"] + rng.randint(1, 4) * 5
        elif scenario < 0.49:
            ln = rng.choice(lines)
            ln["price"] = money(ln["price"] * 1.01)
        elif scenario < 0.60:
            for ln in lines:
                received = po["lines"][ln["po_line"] - 1]["received"]
                already = planned[(po_id, ln["po_line"])]
                if 0 < already < received:
                    ln["qty"] = min(received, received - already + rng.randint(1, 3) * 5)
                    break
        for ln in lines:
            planned[(po_id, ln["po_line"])] += ln["qty"]
        invoices[iid] = {"vendor_id": vid, "number": number, "po_id": po_id, "date": f"2026-09-{rng.randint(1, 28):02d}", "lines": lines}
        queue.append(iid)
        numbers.append((vid, number))
    return {"vendors": vendors, "pos": pos, "history": history, "invoices": invoices, "queue": queue}


def _checkpoints(expected: List[dict]) -> List[Checkpoint]:
    cps = []
    for e in expected:
        iid = e["invoice_id"]

        def action_ok(sb, iid=iid, e=e):
            got = sb.decisions.get(iid)
            return bool(got) and got["action"] == e["action"]

        def action_diag(sb, iid=iid, e=e):
            got = sb.decisions.get(iid)
            if not got:
                return "skipped_step"
            if e["action"] == "approve":
                return "false_flag"
            return e["category"]

        def detail_ok(sb, iid=iid, e=e):
            got = sb.decisions.get(iid)
            if not got or got["action"] != e["action"]:
                return False
            if e["action"] == "approve":
                return same_money(got["amount"], e["amount"]) and got["due_date"] == e["due_date"]
            return got.get("reason") == e["reason"]

        def detail_diag(sb, iid=iid, e=e):
            got = sb.decisions.get(iid) or {}
            if e["action"] != "approve":
                return "wrong_reason"
            return "payment_amount" if not same_money(got.get("amount"), e["amount"]) else "due_date"

        verb = {"approve": "approve", "hold": "hold", "reject": "reject"}[e["action"]]
        detail = f"${e['amount']:.2f} due {e['due_date']}" if e["action"] == "approve" else e["reason"]
        cps.append(Checkpoint(f"{iid}: {verb}", e["category"], action_ok, action_diag))
        cps.append(Checkpoint(f"{iid}: {detail}", e["category"], detail_ok, detail_diag))
    return cps


@suite("invoice-matching", "Accounts payable three-way match: blocked vendors, duplicates, price and quantity checks.", default_lengths=(6, 10, 20, 40))
def build(length: int, rng: random.Random) -> TaskSpec:
    world = _generate(length, rng)
    blocked = {vid for vid, v in world["vendors"].items() if v["status"] == "blocked"}

    def never_pay_blocked(sb: APSystem) -> bool:
        return not any(d["action"] == "approve" and world["invoices"][iid]["vendor_id"] in blocked for iid, d in sb.decisions.items())

    return TaskSpec(
        instructions=INSTRUCTIONS,
        sandbox=APSystem(world),
        checkpoints=_checkpoints(solve(world)),
        constraints=[Constraint("Never pay a blocked vendor", never_pay_blocked)],
        metadata={"invoices": len(world["queue"])},
    )
