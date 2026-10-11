# Changelog

## 0.3.0

- `halflife audit`: check any agent's logs against rules and measure half-life in tool calls, no setup
- Rules library: `requires`, `at_most_once`, `eventually`, `covers`, `never`, `no_loops`, `recovers_from_errors`, and custom rules
- `--rules` and `--save-traces` for benchmark runs
- MCP server (`halflife mcp`) for Claude Code, Cursor and other coding agents
- Claude skill in `integrations/`
- Change since last run, printed after every run
- `--quick` mode for the dev loop
- `halflife init` scaffolding, `halflife badge`, `halflife.assert_reliable`
- SPEC.md: the metric and file formats
- Honest reporting when the half-life lies beyond the tested range

## 0.2.0

- Built-in agents for Ollama, LM Studio, vLLM, OpenAI, OpenRouter, Together, Groq and Anthropic (`halflife bench`)
- Remediation steps, eval packs with `--replay`, autonomy recommendation with `--workflow-length`
- `halflife compare` leaderboards, token usage tracking

## 0.1.0

- `@benchmark`, `halflife run`, claims-intake and invoice-matching suites, half-life and decay-shape fitting, HTML and JSON reports, CI gate
