# Halflife for VS Code

**Your agent works on 5-step tasks. Does it work on 50?**

Halflife measures your AI agent's **reliability half-life**: the task length at which it succeeds half the time. This extension puts that number next to your code, shows where runs broke and why, and hands the fix to your coding agent for you to approve.

Works in VS Code, Cursor, Windsurf and other editors built on VS Code.

## Get started

1. Install the CLI (Python 3.9+): `pip install halflife-ai`, or run **Halflife: Install or Update the Halflife CLI** from the Command Palette.
2. Open a folder, open the Halflife sidebar and click **Try the Demo**. It adds `halflife_demo_agent.py`, a small claims agent with bugs on purpose, and runs it. Click **✦ Fix with coding agent** above it, approve the change, and watch the half-life go up.

Then pick the way in that fits what you have.

**You have agent code.** Put `@benchmark` above the function that runs your agent:

```python
from halflife import benchmark

@benchmark(suite="claims-intake")
def my_agent(task):
    return agent.run(task)
```

A line appears above it: **▶ Run quick · Run full**. After a run it shows the half-life and pass rate, plus **Replay failures** and **Fix with coding agent** when something broke.

**You have agent logs.** Right-click a `.jsonl` log file in the Explorer and choose **Halflife: Audit Agent Logs**. Pick **Suggest rules from these logs** and halflife finds the patterns your agent follows (look before acting, finish what you start, no duplicates, every item handled), writes them to a file next to your logs, and shows which runs broke them. Review the file: delete what's coincidence, keep what's policy.

## What you see

- **Above each agent:** half-life and pass rate from its latest run.
- **Sidebar:** success by task length, where runs broke (with the fixes for each), every failing run with the last tool calls before it broke, and earlier results.
- **Status bar:** the current half-life. Click it to open the sidebar.

## The fix loop

1. **Run quick** while you work.
2. **Fix with coding agent** sends what broke, an example failing run, the suggested fixes and the replay command to your editor's chat, and copies it to the clipboard for Claude Code or any other agent. The agent proposes a change; you approve it.
3. **Replay failures** re-runs exactly the cases that failed.
4. **Run full** confirms the half-life went up.

Results refresh on their own, including runs you start from the terminal or from your coding agent through the halflife MCP server (`claude mcp add halflife -- halflife mcp`).

## Settings

| Setting | Default | What it does |
|---|---|---|
| `halflife.command` | `halflife` | How to start the CLI, e.g. `python -m halflife` or a full path |
| `halflife.outputFolder` | `.halflife` | Where results are saved, relative to the workspace |
| `halflife.codeLens` | `true` | Show half-life and Run actions above `@benchmark` functions |

## More

The metric, the rules library and the CLI are documented in the [halflife repository](https://github.com/ev-prawda92/halflife).
