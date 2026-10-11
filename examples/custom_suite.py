"""Write your own suite in about 40 lines: a support-ticket router.

    halflife run examples/custom_suite.py

A suite needs three things:
  1. a Sandbox: the fake system, with @tool methods the agent can call
  2. the answer key: Checkpoints that check the sandbox's final state
  3. a build(length, rng) function registered with @halflife.suite
"""

import random

import halflife
from halflife import Checkpoint, Sandbox, TaskSpec, ToolError, benchmark, param, tool

TEAMS = {"refund": "billing", "invoice": "billing", "password": "tech", "crash": "tech", "pricing": "sales", "demo": "sales"}


class Helpdesk(Sandbox):
    def __init__(self, tickets):
        self.tickets = tickets
        self.routed = {}

    @tool("List open ticket ids.")
    def list_tickets(self):
        return {"tickets": list(self.tickets)}

    @tool("Read a ticket.", ticket_id=param.string("Ticket id"))
    def get_ticket(self, ticket_id):
        if ticket_id not in self.tickets:
            raise ToolError(f"no ticket {ticket_id}")
        return {"ticket_id": ticket_id, "text": self.tickets[ticket_id]}

    @tool("Route a ticket to a team.", ticket_id=param.string("Ticket id"), team=param.string("Team", enum=["billing", "tech", "sales"]))
    def route(self, ticket_id, team):
        self.routed[ticket_id] = team
        return {"ok": True}


@halflife.suite("ticket-routing", "Route support tickets to billing, tech or sales.")
def build(length: int, rng: random.Random) -> TaskSpec:
    tickets, answers = {}, {}
    for i in range(length):
        topic = rng.choice(list(TEAMS))
        tid = f"T-{1000 + i}"
        tickets[tid] = f"Customer writes about a {topic} issue."
        answers[tid] = TEAMS[topic]
    return TaskSpec(
        instructions="Route every open ticket: refunds and invoices to billing, passwords and crashes to tech, pricing and demos to sales.",
        sandbox=Helpdesk(tickets),
        checkpoints=[
            Checkpoint(f"{tid} → {team}", "routing", lambda sb, tid=tid, team=team: sb.routed.get(tid) == team)
            for tid, team in answers.items()
        ],
    )


@benchmark(suite="ticket-routing", lengths=[5, 10, 20], tasks=2, runs=2)
def keyword_router(task):
    for tid in task.call("list_tickets")["tickets"]:
        text = task.call("get_ticket", ticket_id=tid)["text"]
        team = next(team for word, team in TEAMS.items() if word in text)
        task.call("route", ticket_id=tid, team=team)
