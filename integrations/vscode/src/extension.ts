import * as cp from "child_process";
import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";
import {
  AgentFunction,
  Result,
  RunRecord,
  ciText,
  evalPackPath,
  failingRuns,
  findAgents,
  fixPrompt,
  halfLifeText,
  isAudit,
  latest,
  parseResult,
  pct,
  runMarkdown,
  shellQuote,
  unit,
} from "./core";

const RUN_SCHEME = "halflife-run";

// ── settings and paths ─────────────────────────────────────────────────────

function root(): string | undefined {
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
}

function cfg() {
  const c = vscode.workspace.getConfiguration("halflife");
  return {
    command: c.get<string>("command", "halflife").trim() || "halflife",
    outputFolder: c.get<string>("outputFolder", ".halflife"),
    codeLens: c.get<boolean>("codeLens", true),
  };
}

function outDir(): string | undefined {
  const r = root();
  return r ? path.resolve(r, cfg().outputFolder) : undefined;
}

// ── results on disk ────────────────────────────────────────────────────────

class Store {
  private items: Array<[string, Result]> = [];
  selected: string | undefined;
  private emitter = new vscode.EventEmitter<void>();
  readonly onDidChange = this.emitter.event;

  load(): void {
    const dir = outDir();
    const items: Array<[string, Result]> = [];
    if (dir && fs.existsSync(dir)) {
      for (const name of fs.readdirSync(dir)) {
        if (!name.endsWith(".json") || name.endsWith(".evals.json")) continue;
        const p = path.join(dir, name);
        try {
          const r = parseResult(fs.readFileSync(p, "utf8"));
          if (r) items.push([p, r]);
        } catch {
          // a half-written file during a run; the watcher will fire again
        }
      }
    }
    items.sort((a, b) => (a[1].created_at < b[1].created_at ? 1 : -1));
    this.items = items;
    if (this.selected && !items.some(([p]) => p === this.selected)) this.selected = undefined;
    this.emitter.fire();
  }

  all(): Array<[string, Result]> {
    return this.items;
  }

  get(p: string): Result | undefined {
    return this.items.find(([q]) => q === p)?.[1];
  }

  current(): [string, Result] | undefined {
    if (this.selected) {
      const r = this.get(this.selected);
      if (r) return [this.selected, r];
    }
    return latest(this.items);
  }

  forAgent(name: string): [string, Result] | undefined {
    return latest(this.items, name);
  }

  select(p: string | undefined): void {
    this.selected = p;
    this.emitter.fire();
  }
}

// ── running the CLI ────────────────────────────────────────────────────────

class Runner {
  private busy = false;
  constructor(private out: vscode.OutputChannel, private store: Store) {}

  async run(title: string, args: string[]): Promise<boolean> {
    const cwd = root();
    if (!cwd) {
      vscode.window.showErrorMessage("Halflife: open a folder first, so results have somewhere to go.");
      return false;
    }
    if (this.busy) {
      vscode.window.showInformationMessage("Halflife is already running. Wait for it to finish, or cancel it from the notification.");
      return false;
    }
    this.busy = true;
    const cmdline = [cfg().command, ...args.map(shellQuote)].join(" ");
    this.out.appendLine(`\n$ ${cmdline}`);
    try {
      return await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: `Halflife: ${title}`, cancellable: true },
        (_progress, token) =>
          new Promise<boolean>((resolve) => {
            const child = cp.spawn(cmdline, { cwd, shell: true, env: { ...process.env, PYTHONUNBUFFERED: "1" } });
            token.onCancellationRequested(() => child.kill());
            child.stdout.on("data", (d) => this.out.append(d.toString()));
            child.stderr.on("data", (d) => this.out.append(d.toString()));
            child.on("error", (err) => {
              this.out.appendLine(String(err));
              resolve(false);
            });
            child.on("close", async (code) => {
              this.store.load();
              if (token.isCancellationRequested) {
                this.out.appendLine("Cancelled.");
                return resolve(false);
              }
              // 127 / 9009: the shell couldn't find the command
              if (code === 127 || code === 9009) {
                this.out.appendLine(`Couldn't find "${cfg().command}".`);
                const pick = await vscode.window.showErrorMessage(
                  "Halflife: the halflife command isn't installed in this environment.",
                  "Install it",
                  "Change the command"
                );
                if (pick === "Install it") vscode.commands.executeCommand("halflife.install");
                if (pick === "Change the command") vscode.commands.executeCommand("workbench.action.openSettings", "halflife.command");
                return resolve(false);
              }
              if (code !== 0) {
                const pick = await vscode.window.showWarningMessage(
                  `Halflife: ${title} finished with exit code ${code}. The output says why.`,
                  "Show output"
                );
                if (pick) this.out.show(true);
                return resolve(false);
              }
              resolve(true);
            });
          })
      );
    } finally {
      this.busy = false;
    }
  }
}

// ── sidebar ────────────────────────────────────────────────────────────────

type Node =
  | { kind: "result"; path: string; result: Result; current: boolean }
  | { kind: "info"; label: string; description?: string; tooltip?: string; icon?: string; command?: vscode.Command }
  | { kind: "group"; id: string; label: string; path: string; description?: string; icon?: string }
  | { kind: "mode"; path: string; category: string; label: string; share: number }
  | { kind: "run"; path: string; index: number; run: RunRecord }
  | { kind: "earlier" };

class ResultsView implements vscode.TreeDataProvider<Node> {
  private emitter = new vscode.EventEmitter<Node | undefined>();
  readonly onDidChangeTreeData = this.emitter.event;

  constructor(private store: Store) {
    store.onDidChange(() => this.emitter.fire(undefined));
  }

  getChildren(node?: Node): Node[] {
    if (!node) {
      const cur = this.store.current();
      if (!cur) return [];
      const nodes: Node[] = [{ kind: "result", path: cur[0], result: cur[1], current: true }];
      if (this.store.all().length > 1) nodes.push({ kind: "earlier" });
      return nodes;
    }
    if (node.kind === "earlier") {
      const cur = this.store.current()?.[0];
      return this.store
        .all()
        .filter(([p]) => p !== cur)
        .slice(0, 20)
        .map(([p, r]) => ({ kind: "result", path: p, result: r, current: false }) as Node);
    }
    if (node.kind === "result") {
      const r = node.result;
      const out: Node[] = [];
      out.push({
        kind: "info",
        label: `Half-life: ${halfLifeText(r)}`,
        tooltip: [ciText(r), "The task length at which the agent succeeds half the time."].filter(Boolean).join("\n"),
        icon: "pulse",
      });
      out.push({ kind: "info", label: `Success: ${pct(r.success_rate)} of ${r.results.length} runs`, icon: "pass" });
      out.push({ kind: "group", id: "lengths", label: "Success by task length", path: node.path, icon: "graph-line" });
      if (r.failure_modes.length) out.push({ kind: "group", id: "modes", label: "Where runs broke", path: node.path, icon: "debug-breakpoint-log" });
      const fails = failingRuns(r);
      if (fails.length) out.push({ kind: "group", id: "runs", label: "Failing runs", description: String(fails.length), path: node.path, icon: "error" });
      if (r.structural_advice.length) out.push({ kind: "group", id: "advice", label: "Biggest lever", path: node.path, icon: "lightbulb" });
      if (!node.current) {
        out.unshift({
          kind: "info",
          label: "Show this result first",
          icon: "pin",
          command: { command: "halflife.selectResult", title: "Show", arguments: [node.path] },
        });
      }
      return out;
    }
    if (node.kind === "group") {
      const r = this.store.get(node.path);
      if (!r) return [];
      if (node.id === "lengths") {
        return r.by_length.map((b) => ({
          kind: "info",
          label: `${Math.round(b.mean_length)} ${unit(r)}`,
          description: `${pct(b.rate)}  (${b.successes}/${b.runs})`,
          tooltip: `95% CI ${pct(b.ci[0])}–${pct(b.ci[1])} · average progress ${pct(b.mean_progress)}`,
          icon: b.rate >= 0.8 ? "circle-filled" : b.rate >= 0.5 ? "circle-large-outline" : "circle-slash",
        }));
      }
      if (node.id === "modes") {
        return r.failure_modes.map((m) => ({ kind: "mode", path: node.path, category: m.category, label: m.label, share: m.share }));
      }
      if (node.id === "advice") {
        return r.structural_advice.map((a) => ({ kind: "info", label: a, tooltip: a, icon: "lightbulb" }));
      }
      if (node.id === "runs") {
        const runs: Node[] = [];
        r.results.forEach((run, index) => {
          if (!run.passed) runs.push({ kind: "run", path: node.path, index, run });
        });
        return runs.slice(0, 50);
      }
    }
    if (node.kind === "mode") {
      const r = this.store.get(node.path);
      const rem = r?.remediation.find((x) => x.category === node.category);
      if (!rem) return [];
      const out: Node[] = [];
      if (rem.why) out.push({ kind: "info", label: rem.why, tooltip: rem.why, icon: "question" });
      for (const f of rem.fixes) out.push({ kind: "info", label: f, tooltip: f, icon: "tools" });
      out.push({
        kind: "info",
        label: "Fix with your coding agent",
        icon: "sparkle",
        command: { command: "halflife.fix", title: "Fix", arguments: [node.path] },
      });
      return out;
    }
    return [];
  }

  getTreeItem(node: Node): vscode.TreeItem {
    const C = vscode.TreeItemCollapsibleState;
    switch (node.kind) {
      case "result": {
        const r = node.result;
        const item = new vscode.TreeItem(r.agent, node.current ? C.Expanded : C.Collapsed);
        item.description = `half-life ${halfLifeText(r, true)} ${unit(r)}`;
        item.tooltip = `${r.suite}\n${new Date(r.created_at).toLocaleString()}\n${path.basename(node.path)}`;
        item.iconPath = new vscode.ThemeIcon(isAudit(r) ? "checklist" : "beaker");
        const replayable = !isAudit(r) && fs.existsSync(evalPackPath(node.path));
        item.contextValue = replayable ? "result replayable" : "result";
        (item as vscode.TreeItem & { resultPath?: string }).resultPath = node.path;
        return item;
      }
      case "earlier": {
        const item = new vscode.TreeItem("Earlier results", C.Collapsed);
        item.iconPath = new vscode.ThemeIcon("history");
        return item;
      }
      case "info": {
        const item = new vscode.TreeItem(node.label, C.None);
        item.description = node.description;
        item.tooltip = node.tooltip;
        if (node.icon) item.iconPath = new vscode.ThemeIcon(node.icon);
        item.command = node.command;
        return item;
      }
      case "group": {
        const open = node.id === "modes" || node.id === "advice";
        const item = new vscode.TreeItem(node.label, open ? C.Expanded : C.Collapsed);
        item.description = node.description;
        if (node.icon) item.iconPath = new vscode.ThemeIcon(node.icon);
        return item;
      }
      case "mode": {
        const item = new vscode.TreeItem(node.label, C.Collapsed);
        item.description = pct(node.share);
        item.iconPath = new vscode.ThemeIcon("warning");
        return item;
      }
      case "run": {
        const ff = node.run.first_failure;
        const item = new vscode.TreeItem(ff?.name || node.run.error || "failed", C.None);
        const audited = isAudit(this.store.get(node.path) ?? ({ config: {} } as Result));
        item.description = audited ? node.run.task_id : `${node.run.task_id} · run ${node.run.run}`;
        item.tooltip = `${node.run.task_id} (${node.run.length} steps)\n${ff?.name ?? ""}`;
        item.iconPath = new vscode.ThemeIcon("circle-slash");
        item.command = { command: "halflife.showRun", title: "Show run", arguments: [node.path, node.index] };
        return item;
      }
    }
  }
}

// ── code lenses above @benchmark functions ─────────────────────────────────

class Lenses implements vscode.CodeLensProvider {
  private emitter = new vscode.EventEmitter<void>();
  readonly onDidChangeCodeLenses = this.emitter.event;

  constructor(private store: Store) {
    store.onDidChange(() => this.emitter.fire());
  }

  provideCodeLenses(doc: vscode.TextDocument): vscode.CodeLens[] {
    if (!cfg().codeLens) return [];
    const lenses: vscode.CodeLens[] = [];
    for (const fn of findAgents(doc.getText())) {
      const range = new vscode.Range(fn.line, 0, fn.line, 0);
      const found = this.store.forAgent(fn.name);
      if (found) {
        const [p, r] = found;
        lenses.push(
          new vscode.CodeLens(range, {
            title: `half-life ${halfLifeText(r, true)} ${unit(r)} · ${pct(r.success_rate)} pass`,
            tooltip: "Show where runs broke",
            command: "halflife.selectResult",
            arguments: [p, true],
          })
        );
      }
      lenses.push(new vscode.CodeLens(range, { title: "▶ Run quick", command: "halflife.runQuick", arguments: [doc.uri, fn.name] }));
      lenses.push(new vscode.CodeLens(range, { title: "Run full", command: "halflife.runFull", arguments: [doc.uri, fn.name] }));
      if (found) {
        const [p, r] = found;
        if (failingRuns(r).length) {
          if (fs.existsSync(evalPackPath(p))) {
            lenses.push(new vscode.CodeLens(range, { title: "Replay failures", command: "halflife.replay", arguments: [p, doc.uri] }));
          }
          lenses.push(new vscode.CodeLens(range, { title: "✦ Fix with coding agent", command: "halflife.fix", arguments: [p, doc.uri] }));
        }
      }
    }
    return lenses;
  }
}

// ── activation ─────────────────────────────────────────────────────────────

export function activate(context: vscode.ExtensionContext): void {
  const output = vscode.window.createOutputChannel("Halflife");
  const store = new Store();
  const runner = new Runner(output, store);
  const view = new ResultsView(store);
  const tree = vscode.window.createTreeView("halflife.results", { treeDataProvider: view, showCollapseAll: true });
  const status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 50);
  status.command = "halflife.results.focus";

  const runDocs = new Map<string, string>();
  const runDocEmitter = new vscode.EventEmitter<vscode.Uri>();

  function updateStatus(): void {
    const cur = store.current();
    if (!cur) {
      status.hide();
      return;
    }
    const r = cur[1];
    status.text = `$(pulse) half-life ${halfLifeText(r, true)} ${unit(r)}`;
    status.tooltip = `${r.agent} · ${r.suite}\n${pct(r.success_rate)} of ${r.results.length} runs passed\nClick to see where runs broke`;
    status.show();
  }

  function updateContext(): void {
    const ed = vscode.window.activeTextEditor;
    const has = !!ed && ed.document.languageId === "python" && findAgents(ed.document.getText()).length > 0;
    vscode.commands.executeCommand("setContext", "halflife.hasAgents", has);
  }

  store.onDidChange(updateStatus);

  function rel(uri: vscode.Uri): string {
    const r = root();
    return r ? path.relative(r, uri.fsPath).split(path.sep).join("/") : uri.fsPath;
  }

  async function pickAgent(uri?: vscode.Uri, name?: string): Promise<[vscode.Uri, string] | undefined> {
    if (uri && name) return [uri, name];
    const target = uri ?? vscode.window.activeTextEditor?.document.uri;
    if (!target || !target.fsPath.endsWith(".py")) {
      vscode.window.showInformationMessage("Halflife: open the Python file with your agent, or click Run quick above an @benchmark function.");
      return undefined;
    }
    const doc = await vscode.workspace.openTextDocument(target);
    const agents: AgentFunction[] = findAgents(doc.getText());
    if (!agents.length) {
      const pick = await vscode.window.showInformationMessage(
        "Halflife: no @benchmark functions in this file. Decorate your agent function with @benchmark(suite=\"claims-intake\") to benchmark it.",
        "Show how"
      );
      if (pick) vscode.env.openExternal(vscode.Uri.parse("https://github.com/ev-prawda92/halflife#you-have-agent-code--benchmark-it"));
      return undefined;
    }
    if (agents.length === 1) return [target, agents[0].name];
    const chosen = await vscode.window.showQuickPick(agents.map((a) => a.name), { placeHolder: "Which agent?" });
    return chosen ? [target, chosen] : undefined;
  }

  async function runAgent(quick: boolean, uri?: vscode.Uri, name?: string): Promise<void> {
    const picked = await pickAgent(uri, name);
    if (!picked) return;
    const [file, fn] = picked;
    await vscode.workspace.saveAll(false);
    const args = ["run", `${rel(file)}:${fn}`, "--out", cfg().outputFolder];
    if (quick) args.push("--quick");
    const ok = await runner.run(`${quick ? "quick" : "full"} benchmark of ${fn}`, args);
    if (ok) afterRun(fn);
  }

  function afterRun(agent?: string): void {
    const found = agent ? store.forAgent(agent) : store.current();
    store.select(undefined);
    if (!found) return;
    const [, r] = found;
    const fails = failingRuns(r).length;
    const msg = `Halflife: ${r.agent} half-life ${halfLifeText(r)}. ${fails ? `${fails} of ${r.results.length} runs failed.` : "Every run passed."}`;
    const actions = fails ? ["Fix with coding agent", "Open report"] : ["Open report"];
    vscode.window.showInformationMessage(msg, ...actions).then((pick) => {
      if (pick === "Fix with coding agent") vscode.commands.executeCommand("halflife.fix", found[0]);
      if (pick === "Open report") vscode.commands.executeCommand("halflife.openReport", found[0]);
    });
  }

  function resolveResult(arg?: unknown): [string, Result] | undefined {
    if (typeof arg === "string") {
      const r = store.get(arg);
      if (r) return [arg, r];
    }
    const fromItem = (arg as { resultPath?: string; path?: string } | undefined);
    const p = fromItem?.resultPath ?? (fromItem && (fromItem as { kind?: string }).kind === "result" ? fromItem.path : undefined);
    if (p) {
      const r = store.get(p);
      if (r) return [p, r];
    }
    return store.current();
  }

  const nextFree = (dir: string, base: string): string => {
    let p = path.join(dir, `${base}.py`);
    for (let i = 2; fs.existsSync(p); i++) p = path.join(dir, `${base}_${i}.py`);
    return p;
  };

  context.subscriptions.push(
    output,
    tree,
    status,
    runDocEmitter,
    vscode.languages.registerCodeLensProvider({ language: "python" }, new Lenses(store)),
    vscode.workspace.registerTextDocumentContentProvider(RUN_SCHEME, {
      onDidChange: runDocEmitter.event,
      provideTextDocumentContent: (uri) => runDocs.get(uri.toString()) ?? "",
    }),

    vscode.commands.registerCommand("halflife.runQuick", (uri?: vscode.Uri, name?: string) => runAgent(true, uri, name)),
    vscode.commands.registerCommand("halflife.runFull", (uri?: vscode.Uri, name?: string) => runAgent(false, uri, name)),

    vscode.commands.registerCommand("halflife.replay", async (arg?: unknown, fileUri?: vscode.Uri) => {
      const found = resolveResult(arg);
      if (!found) return void vscode.window.showInformationMessage("Halflife: no results to replay yet. Run a benchmark first.");
      const [p, r] = found;
      const pack = evalPackPath(p);
      if (isAudit(r) || !fs.existsSync(pack)) {
        return void vscode.window.showInformationMessage("Halflife: this result has no failing cases saved to replay.");
      }
      let file = fileUri;
      if (!file) {
        const ed = vscode.window.activeTextEditor;
        if (ed && findAgents(ed.document.getText()).some((a) => a.name === r.agent)) file = ed.document.uri;
      }
      if (!file) {
        const hits = await vscode.workspace.findFiles("**/*.py", "**/{.venv,venv,node_modules,site-packages}/**", 500);
        for (const h of hits) {
          const text = fs.readFileSync(h.fsPath, "utf8");
          if (findAgents(text).some((a) => a.name === r.agent)) {
            file = h;
            break;
          }
        }
      }
      if (!file) return void vscode.window.showInformationMessage(`Halflife: couldn't find the @benchmark function ${r.agent}. Open its file and try again.`);
      await vscode.workspace.saveAll(false);
      const started = new Date(Date.now() - 1000).toISOString().replace(/\.\d+Z$/, "Z");
      const ok = await runner.run(`replaying ${failingRuns(r).length} failing case(s) of ${r.agent}`, [
        "run", `${rel(file)}:${r.agent}`, "--replay", path.relative(root() ?? "", pack), "--out", cfg().outputFolder,
      ]);
      if (ok) {
        const newest = store.all().find(([, x]) => x.agent === r.agent && (x.config as { replay?: unknown }).replay && x.created_at >= started);
        if (newest) {
          const nr = newest[1];
          const fixed = nr.results.filter((x) => x.passed).length;
          vscode.window.showInformationMessage(
            `Halflife: replayed the cases that failed. ${fixed} of ${nr.results.length} runs pass now.${fixed === nr.results.length ? " Run a full benchmark to confirm the half-life went up." : ""}`,
            "Run full"
          ).then((pick) => pick && runAgent(false, file, r.agent));
        }
      }
    }),

    vscode.commands.registerCommand("halflife.audit", async (uri?: vscode.Uri) => {
      let logs = uri;
      if (!logs) {
        const picked = await vscode.window.showOpenDialog({
          canSelectMany: false,
          openLabel: "Audit these logs",
          filters: { "Agent logs (JSON Lines)": ["jsonl", "json"] },
          defaultUri: root() ? vscode.Uri.file(root()!) : undefined,
        });
        logs = picked?.[0];
      }
      if (!logs) return;
      const choice = await vscode.window.showQuickPick(
        [
          { label: "$(sparkle) Suggest rules from these logs", detail: "Halflife finds the patterns your agent follows and writes them to a file you can review.", id: "suggest" },
          { label: "$(file-code) Use a rules file…", detail: "A Python file with RULES = [...]", id: "file" },
          { label: "$(check) Only the built-in checks", detail: "Loops and repeating a call that just failed.", id: "none" },
        ],
        { placeHolder: "How should halflife check these logs?" }
      );
      if (!choice) return;
      const args = ["audit", rel(logs), "--out", cfg().outputFolder];
      let rulesOut: string | undefined;
      if (choice.id === "suggest") {
        rulesOut = nextFree(path.dirname(logs.fsPath), "rules_suggested");
        args.push("--suggest-rules", rel(vscode.Uri.file(rulesOut)));
      } else if (choice.id === "file") {
        const f = await vscode.window.showOpenDialog({ canSelectMany: false, filters: { Python: ["py"] }, openLabel: "Use these rules" });
        if (!f?.[0]) return;
        args.push("--rules", rel(f[0]));
      }
      const ok = await runner.run(`auditing ${path.basename(logs.fsPath)}`, args);
      if (ok) {
        afterRun();
        if (rulesOut && fs.existsSync(rulesOut)) {
          const doc = await vscode.workspace.openTextDocument(rulesOut);
          await vscode.window.showTextDocument(doc, { preview: false, viewColumn: vscode.ViewColumn.Beside });
          vscode.window.showInformationMessage(
            "Halflife: these rules describe what your agent usually does. Delete anything that's coincidence, keep what's policy, then audit with this file."
          );
        }
      }
    }),

    vscode.commands.registerCommand("halflife.fix", async (arg?: unknown, fileUri?: vscode.Uri) => {
      const found = resolveResult(arg);
      if (!found) return void vscode.window.showInformationMessage("Halflife: run a benchmark or audit first.");
      const [p, r] = found;
      let file: string | undefined = fileUri ? rel(fileUri) : undefined;
      if (!file) {
        const ed = vscode.window.activeTextEditor;
        if (ed && findAgents(ed.document.getText()).some((a) => a.name === r.agent)) file = rel(ed.document.uri);
      }
      const prompt = fixPrompt(r, path.relative(root() ?? "", p), file);
      let copied = true;
      try {
        await vscode.env.clipboard.writeText(prompt);
      } catch {
        copied = false;
      }
      // Hand it to the editor's chat when there is one (VS Code with Copilot Chat, and others that support this command).
      let opened = false;
      try {
        await vscode.commands.executeCommand("workbench.action.chat.open", { query: prompt });
        opened = true;
      } catch {
        opened = false;
      }
      if (!opened) {
        const doc = await vscode.workspace.openTextDocument({ content: prompt, language: "markdown" });
        await vscode.window.showTextDocument(doc, { preview: true, viewColumn: vscode.ViewColumn.Beside });
      }
      vscode.window.showInformationMessage(
        opened
          ? "Halflife: sent the fix request to chat (and copied it). Your coding agent will propose a change for you to approve."
          : `Halflife: ${copied ? "copied" : "opened"} a fix request for your coding agent. Paste it into Claude Code, Cursor or Copilot Chat; it will propose a change for you to approve.`
      );
    }),

    vscode.commands.registerCommand("halflife.openReport", (arg?: unknown) => {
      const found = resolveResult(arg);
      if (!found) return;
      const html = found[0].replace(/\.json$/, ".html");
      if (!fs.existsSync(html)) return void vscode.window.showInformationMessage("Halflife: this result was saved without an HTML report.");
      vscode.env.openExternal(vscode.Uri.file(html));
    }),

    vscode.commands.registerCommand("halflife.showRun", async (p: string, index: number) => {
      const r = store.get(p);
      const run = r?.results[index];
      if (!r || !run) return;
      const uri = vscode.Uri.parse(`${RUN_SCHEME}:/${encodeURIComponent(run.task_id)}-run${run.run}.md?${encodeURIComponent(p)}#${index}`);
      runDocs.set(uri.toString(), runMarkdown(r, run));
      runDocEmitter.fire(uri);
      await vscode.commands.executeCommand("markdown.showPreview", uri);
    }),

    vscode.commands.registerCommand("halflife.selectResult", (p: string, focus?: boolean) => {
      store.select(p);
      if (focus) vscode.commands.executeCommand("halflife.results.focus");
    }),

    vscode.commands.registerCommand("halflife.demo", async () => {
      const r = root();
      if (!r) return void vscode.window.showErrorMessage("Halflife: open a folder first, so the demo has somewhere to go.");
      const target = path.join(r, "halflife_demo_agent.py");
      const created = !fs.existsSync(target);
      if (created) fs.copyFileSync(path.join(context.extensionPath, "media", "halflife_demo_agent.py"), target);
      const uri = vscode.Uri.file(target);
      const doc = await vscode.workspace.openTextDocument(uri);
      await vscode.window.showTextDocument(doc, { preview: false });
      if (created) {
        vscode.window.showInformationMessage(
          "Halflife: added halflife_demo_agent.py, a small claims agent with bugs on purpose. Running it now. Then click ✦ Fix with coding agent above it."
        );
      }
      await runAgent(true, uri, "claims_agent");
    }),

    vscode.commands.registerCommand("halflife.install", () => {
      const term = vscode.window.createTerminal("Halflife install");
      term.show();
      term.sendText("python -m pip install -U halflife-ai");
    }),

    vscode.commands.registerCommand("halflife.refresh", () => store.load()),

    vscode.window.onDidChangeActiveTextEditor(updateContext),
    vscode.workspace.onDidChangeTextDocument((e) => {
      if (e.document === vscode.window.activeTextEditor?.document) updateContext();
    }),
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("halflife")) {
        setupWatcher();
        store.load();
      }
    })
  );

  let watcher: vscode.FileSystemWatcher | undefined;
  function setupWatcher(): void {
    watcher?.dispose();
    const r = root();
    if (!r) return;
    const pattern = new vscode.RelativePattern(r, `${cfg().outputFolder.replace(/\\/g, "/")}/*.json`);
    watcher = vscode.workspace.createFileSystemWatcher(pattern);
    let t: NodeJS.Timeout | undefined;
    const reload = () => {
      if (t) clearTimeout(t);
      t = setTimeout(() => store.load(), 300);
    };
    watcher.onDidCreate(reload);
    watcher.onDidChange(reload);
    watcher.onDidDelete(reload);
  }
  setupWatcher();
  context.subscriptions.push({ dispose: () => watcher?.dispose() });

  store.load();
  updateContext();
}

export function deactivate(): void {}
