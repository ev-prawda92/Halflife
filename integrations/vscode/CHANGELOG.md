# Changelog

## 0.1.0

- Try the Demo adds a small sample agent with bugs on purpose, so the whole fix loop can be tried in a few minutes
- Half-life, pass rate, and Run quick / Run full above every `@benchmark` function
- Reliability sidebar: success by task length, where runs broke, the fixes for each, and every failing run with the breaking step marked
- Fix with your coding agent: sends what broke, the evidence and the replay command to your editor's chat (and the clipboard), so the agent proposes a change for you to approve
- Replay only the failing cases after a change
- Audit agent logs from the sidebar or the Explorer, with rules suggested from the logs themselves
- Half-life in the status bar; results refresh whenever a run finishes, from the editor or the terminal
