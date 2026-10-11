# Contributing

Thanks for helping make agent reliability measurable.

## Setup

```
git clone https://github.com/ev-prawda92/halflife
cd halflife
pip install -e ".[dev]"
pytest
```

## Most wanted

- **New suites.** A realistic workflow (support, IT ops, procurement, scheduling…) with a simulated system and an answer key. See `examples/custom_suite.py` and `src/halflife/suites/`.
- **Trace readers** for agent frameworks and observability tools, so `halflife audit` reads their logs directly.
- **Rules** that hold for many agents.
- **Real-model results.** Run `halflife bench` on models you have access to and share the result JSON in an issue.

## Ground rules for suites

1. Every suite needs a test showing that an agent following the written instructions exactly passes 100% of tasks across lengths and seeds. Without it, failures can't be trusted.
2. Tasks must be generated from the seeded RNG passed to `build`, so the same seed always gives the same tasks.
3. The answer key must never reach the agent. Only the instructions and tool results do.
4. Make it harder with length for a real reason: state carried across items, items that interact, not just more of the same.

## Changes to the metric

Anything that would change a reported number goes through [SPEC.md](SPEC.md) first. Open an issue describing the change and why.

## Pull requests

Keep them focused, add tests, and run `pytest` before opening. Plain, specific commit messages.
