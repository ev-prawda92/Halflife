# The halflife specification, v0.1

This document defines the reliability half-life metric and the file formats halflife reads and writes, so that any tool, framework, or platform can produce comparable numbers. The reference implementation is this repository.

## 1. Terms

- **Run**: one attempt by an agent at one task.
- **Length**: the size of a run. For benchmark tasks, the number of verifiable steps (checkpoints) in the task. For recorded logs, the number of tool calls in the run. Results state their unit (`steps` or `tool calls`); the two are not interchangeable.
- **Pass**: a run passes when every checkpoint is met, no rule is violated, and the agent did not crash or exceed its tool-call budget.

## 2. The metric

Success probability at length L is modeled as

    S(L) = 2 ^ ( −(L / h) ^ k )

- **h, the reliability half-life**: the length at which S = 0.5.
- **k, the decay shape**: k ≈ 1 is constant risk per step; k > 1 means errors compound as length grows; k < 1 means failures concentrate early. k at the upper fitting bound (6) indicates a cliff.

h and k are fit jointly by maximum likelihood over all runs, each run contributing a Bernoulli outcome at its own length. A 95% interval for h is obtained by bootstrap: resample runs with replacement within each length group, refit, and take the 2.5th and 97.5th percentiles.

Reports also give the **observed 50% crossing**, linear interpolation between the observed success rates of adjacent length groups. It is a model-free check on h.

### Certified half-life

A half-life is only backed by evidence up to the longest length tested. The **certified half-life** is

    min(h, longest length tested)

Gates, badges, and comparisons use the certified value. When h exceeds the longest length tested, results must say so (for example, "≥ 40 steps") rather than report the extrapolated h as a measurement.

### Minimum reporting

A result claiming a half-life must state: h, its 95% interval, k, the unit, the lengths tested, the number of runs, and the task source (suite name and seed, or trace file).

## 3. Trace format

JSON Lines. Either one object per tool call:

```json
{"run_id": "r1", "tool": "get_claim", "args": {"claim_id": "C1"}, "result": {"...": "..."}}
{"run_id": "r1", "tool": "decide_line", "args": {"claim_id": "C1", "line": 1}, "error": "claim is finalized"}
```

or one object per run:

```json
{"run_id": "r1", "steps": [{"tool": "get_claim", "args": {"claim_id": "C1"}}]}
```

Readers must accept these aliases: tool name as `tool`, `name`, `tool_name`, or `function` (string, or object with `name` and `arguments`); arguments as `args`, `arguments`, `input`, or `parameters` (object or JSON string); run id as `run_id`, `trace_id`, `session_id`, `conversation_id`, or `task_id`; result as `result` or `output`. A step is an error when it has a non-empty `error`, or its result is an object with a non-empty `error`.

## 4. Rules

A rule is an invariant over a run's ordered steps. Only successful steps (no error) count toward satisfying or triggering a rule. Built-in rule semantics:

| Rule | Violated when |
|---|---|
| `requires(A, before=B, match)` | an A call has no earlier B call with matching argument values |
| `at_most_once(A, key)` | A succeeds more than once for the same key values |
| `eventually(A, then=B, match)` | an A call is never followed by a B call with matching values |
| `covers(S, field, actions, key)` | an id listed in S's result field never reaches one of the actions |
| `never(A, when)` | A succeeds with arguments satisfying the predicate |
| `no_loops(n)` | the same tool and arguments repeat more than n times consecutively |
| `recovers_from_errors()` | a call identical to one that just failed is sent again |

Default rules applied to recorded logs: `no_loops(3)` and `recovers_from_errors()`.

## 5. Result file

A JSON object with at least: `agent`, `suite`, `created_at`, `config` (including `lengths`, `seed`, and `unit` when not steps), `half_life` (number, `"inf"` if no failures, or null), `half_life_ci`, `certified_half_life`, `decay_shape`, `observed_crossing`, `success_rate`, `by_length`, `failure_modes`, and `results` (one entry per run).

## 6. Eval pack

Failing benchmark tasks, reproducible because tasks are generated from a seed:

```json
{"halflife_eval_pack": 1, "suite": "claims-intake", "seed": 0,
 "cases": [{"length": 20, "task_index": 3, "task_id": "claims-intake/L20/t3", "failed_runs": 2, "failures": ["deductible_tracking"]}]}
```

Replaying a pack runs exactly those (suite, seed, length, task index) tasks.

## 7. Badge

`label: "agent half-life"`, message the certified half-life rounded to an integer with its unit, prefixed by `≥` when the half-life exceeds the longest length tested.

## Changes

Proposals to this spec are welcome as issues. Versions change when a definition would change a reported number.
