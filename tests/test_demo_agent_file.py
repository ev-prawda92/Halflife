"""The sample agent the VS Code extension drops into a folder must stay broken in a fixable way."""

import importlib.util
import os

AGENT = os.path.join(os.path.dirname(__file__), "..", "integrations", "vscode", "media", "halflife_demo_agent.py")


def load(source, name):
    path = os.path.join(os.path.dirname(AGENT), f"_{name}.py")
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(source, path, "exec"), mod.__dict__)
    return mod.claims_agent


def fixed(source):
    s = source.replace("[:MAX_CLAIMS]", "")
    s = s.replace("        deductible_used = 0.0\n", "")
    s = s.replace("    members, plans, providers = {}, {}, {}\n", "    members, plans, providers = {}, {}, {}\n    deductible_used = {}\n")
    s = s.replace(
        'member["deductible_remaining_at_batch_start"] - deductible_used\n',
        'member["deductible_remaining_at_batch_start"] - deductible_used.get(member_id, 0.0)\n',
    )
    s = s.replace("            deductible_used += deductible\n", "            deductible_used[member_id] = deductible_used.get(member_id, 0.0) + deductible\n")
    assert s != source
    return s


def test_sample_agent_breaks_on_long_tasks_and_the_fix_works():
    source = open(AGENT).read()
    buggy = load(source, "buggy").evaluate(lengths=[5, 40], tasks=3, runs=1, seed=0)
    rates = {row["requested_length"]: row["rate"] for row in buggy.by_length}
    assert rates[5] == 1.0 and rates[40] < 0.5
    labels = {m["label"] for m in buggy.failure_modes}
    assert "Lost track of deductible across the batch" in labels

    good = load(fixed(source), "good").evaluate(lengths=[5, 40], tasks=3, runs=1, seed=0)
    assert good.success_rate == 1.0
