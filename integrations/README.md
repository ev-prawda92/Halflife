# Use halflife inside the tools you already use

One engine, many entry points. Every surface reports the same metric.

## Claude Code

```
claude mcp add halflife -- halflife mcp
```

Then ask Claude Code to test your agent. It can run benchmarks, read where runs broke, change your code, and replay exactly the failing cases.

## Claude (skill)

`claude-skill/halflife/` is a skill that teaches Claude when and how to use halflife: whenever you're building, changing, or shipping an agent. Zip the `halflife` folder and add it as a skill in Claude.

## Cursor and other MCP clients

Add to your MCP configuration (for Cursor, `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "halflife": { "command": "halflife", "args": ["mcp"] }
  }
}
```

Any assistant that supports MCP servers over stdio can use the same command.

## MCP tools

| Tool | What it does |
|---|---|
| `run_benchmark` | Benchmark an agent file; returns half-life, failure modes, fixes, and the eval pack path |
| `replay_failures` | Re-run only the previously failing cases after a fix |
| `bench_model` | Benchmark a model directly, e.g. `ollama:qwen2.5:7b` |
| `audit_traces` | Check recorded logs against rules, or suggest rules from the logs first |
| `show_task` | Show what an agent receives, optionally with the answer key |
| `list_suites` | List built-in suites |

## Tests and CI

```python
import halflife

def test_agent_is_reliable():
    halflife.assert_reliable(my_agent, min_half_life=15)
```

```
halflife init --ci
```
