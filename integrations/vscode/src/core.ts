// Pure logic: reading halflife result files and turning them into what the editor shows.
// No `vscode` import here, so it can be unit-tested with plain Node.

export type Num = number | "inf" | null;

export interface ByLength {
  requested_length: number;
  mean_length: number;
  runs: number;
  successes: number;
  rate: number;
  ci: [number, number];
  mean_progress: number;
}

export interface FailureMode {
  category: string;
  label: string;
  count: number;
  share: number;
}

export interface Remedy {
  category: string;
  label: string;
  share: number;
  why?: string;
  fixes: string[];
  example?: { task_id: string; run: number; step: number; of: number; checkpoint: string };
}

export interface TraceStep {
  step?: number;
  tool?: string;
  args?: Record<string, unknown>;
  result?: unknown;
  error?: unknown;
}

export interface RunRecord {
  task_id: string;
  run: number;
  length: number;
  passed: boolean;
  first_failure: { index: number; name: string; category: string } | null;
  violations?: string[];
  error?: string | null;
  trace_tail?: TraceStep[];
}

export interface Result {
  halflife_version?: string;
  agent: string;
  suite: string;
  created_at: string;
  config: { mode?: string; unit?: string; lengths?: number[]; source?: string; [k: string]: unknown };
  half_life: Num;
  half_life_ci: [Num, Num];
  certified_half_life: Num;
  success_rate: number;
  by_length: ByLength[];
  failure_modes: FailureMode[];
  remediation: Remedy[];
  structural_advice: string[];
  results: RunRecord[];
}

export function isResult(x: unknown): x is Result {
  const r = x as Result;
  return !!r && typeof r === "object" && typeof r.agent === "string" && typeof r.suite === "string" && Array.isArray(r.by_length);
}

export function parseResult(text: string): Result | undefined {
  try {
    const obj = JSON.parse(text);
    return isResult(obj) ? obj : undefined;
  } catch {
    return undefined;
  }
}

export function unit(r: Result): string {
  return r.config.unit || "steps";
}

export function isAudit(r: Result): boolean {
  return r.config.mode === "audit";
}

function longestTested(r: Result): number {
  const ls = r.results.map((x) => x.length);
  return ls.length ? Math.max(...ls) : 0;
}

function shortestTested(r: Result): number {
  const ls = r.results.map((x) => x.length);
  return ls.length ? Math.min(...ls) : 0;
}

/** Same wording as the CLI: never claims more than the tested range supports. */
export function halfLifeText(r: Result, short = false): string {
  const h = r.half_life;
  const hi = longestTested(r);
  const u = unit(r);
  if (h === null || h === undefined) return "n/a";
  if (h === "inf") return short ? `> ${hi}` : `> ${hi} ${u} (no failures observed; test longer tasks)`;
  if (h === 0) return short ? `< ${shortestTested(r)}` : `< ${shortestTested(r)} ${u} (every run failed)`;
  if (h > hi) {
    if (h <= 2 * hi) return short ? `~${Math.round(h)}` : `~${Math.round(h)} ${u} (estimated beyond the longest tested, ${hi})`;
    return short ? `> ${hi}` : `> ${hi} ${u} (not reached in the tested range; test longer runs)`;
  }
  return short ? h.toFixed(1) : `${h.toFixed(1)} ${u}`;
}

export function ciText(r: Result): string {
  const [lo, hi] = r.half_life_ci || [null, null];
  if (typeof lo !== "number" || typeof hi !== "number" || lo === hi) return "";
  return `95% CI ${lo.toFixed(1)}–${hi.toFixed(1)}`;
}

export function pct(x: number): string {
  return `${Math.round(x * 100)}%`;
}

export function failingRuns(r: Result): RunRecord[] {
  return r.results.filter((x) => !x.passed);
}

/** Pick the newest result, optionally for one agent. Results are [path, result] pairs. */
export function latest(items: Array<[string, Result]>, agent?: string): [string, Result] | undefined {
  let best: [string, Result] | undefined;
  for (const item of items) {
    const [, r] = item;
    if (agent && r.agent !== agent) continue;
    if (r.config && (r.config as { replay?: unknown }).replay) continue;
    if (!best || r.created_at > best[1].created_at) best = item;
  }
  return best;
}

/** The eval pack saved next to a result, if the run produced one. */
export function evalPackPath(resultPath: string): string {
  return resultPath.replace(/\.json$/, ".evals.json");
}

export interface AgentFunction {
  name: string;
  line: number; // zero-based line of the decorator
  defLine: number; // zero-based line of `def`
}

/** Find functions decorated with @benchmark / @halflife.benchmark in Python source. */
export function findAgents(source: string): AgentFunction[] {
  const lines = source.split(/\r?\n/);
  const out: AgentFunction[] = [];
  for (let i = 0; i < lines.length; i++) {
    if (!/^\s*@(?:halflife\.)?benchmark\b/.test(lines[i])) continue;
    for (let j = i + 1; j < Math.min(lines.length, i + 40); j++) {
      const m = /^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/.exec(lines[j]);
      if (m) {
        out.push({ name: m[1], line: i, defLine: j });
        break;
      }
    }
  }
  return out;
}

function stepLine(s: TraceStep): string {
  const args = JSON.stringify(s.args ?? {});
  const err = s.error ? `  → error: ${String(s.error).slice(0, 160)}` : "";
  return `${s.step ?? "?"}. ${s.tool ?? "?"}(${args.length > 200 ? args.slice(0, 200) + "…" : args})${err}`;
}

/** A prompt for the developer's coding agent: what broke, the evidence, and how to prove a fix. */
export function fixPrompt(r: Result, resultPath: string, agentFile?: string): string {
  const lines: string[] = [];
  const top = r.remediation[0];
  const fails = failingRuns(r);
  lines.push(`My agent \`${r.agent}\` was tested with halflife (agent reliability on long tasks).`);
  lines.push("");
  lines.push(`Reliability half-life: ${halfLifeText(r)}${ciText(r) ? ` (${ciText(r)})` : ""}. Success rate ${pct(r.success_rate)} over ${r.results.length} runs.`);
  if (r.failure_modes.length) {
    lines.push("");
    lines.push("Where runs broke (first failure):");
    for (const m of r.failure_modes.slice(0, 4)) lines.push(`- ${pct(m.share)} ${m.label}`);
  }
  if (r.structural_advice.length) {
    lines.push("");
    for (const tip of r.structural_advice) lines.push(`Note: ${tip}`);
  }
  if (top) {
    lines.push("");
    lines.push(`Top problem: ${top.label} (${pct(top.share)} of failures).`);
    if (top.why) lines.push(top.why);
    lines.push("Suggested fixes, structural first:");
    for (const f of top.fixes) lines.push(`- ${f}`);
  }
  const ex = fails.find((f) => top && f.first_failure?.category === top.category) || fails[0];
  if (ex) {
    lines.push("");
    lines.push(`Example failing run ${ex.task_id} (run ${ex.run}): ${ex.first_failure?.name ?? ex.error ?? "failed"}`);
    if (ex.trace_tail?.length) {
      lines.push("Last tool calls before it broke:");
      for (const s of ex.trace_tail.slice(-6)) lines.push(`  ${stepLine(s)}`);
    }
  }
  lines.push("");
  lines.push(`Please propose the smallest change${agentFile ? ` to ${agentFile}` : ""} that addresses the top problem, and show it to me as a diff before applying it.`);
  if (!isAudit(r)) {
    const target = agentFile ? `${agentFile}:${r.agent}` : r.agent;
    lines.push(`After I approve it, prove it worked: \`halflife run ${target} --replay ${evalPackPath(resultPath)}\`, then a full \`halflife run ${target}\` to check the half-life went up.`);
  } else {
    lines.push(`After I approve it, re-run the agent to record new logs and audit them again to check the broken rules no longer break.`);
  }
  return lines.join("\n");
}

/** A readable Markdown view of one failing run, with the breaking step marked. */
export function runMarkdown(r: Result, run: RunRecord): string {
  const out: string[] = [];
  out.push(`# ${run.task_id} · run ${run.run}`);
  out.push("");
  out.push(`**Agent:** ${r.agent} · **Length:** ${run.length} ${unit(r)} · **Result:** ${run.passed ? "passed" : "failed"}`);
  if (run.first_failure) {
    out.push("");
    out.push(`**First failure:** ${run.first_failure.name}`);
  }
  if (run.error) out.push("", `**Error:** ${run.error}`);
  if (run.violations?.length) {
    out.push("", "**Rules broken:**");
    for (const v of run.violations) out.push(`- ${v}`);
  }
  if (run.trace_tail?.length) {
    out.push("", "## Last tool calls before it broke", "");
    const tail = run.trace_tail;
    tail.forEach((s, i) => {
      const last = i === tail.length - 1;
      out.push(`${last ? "**→ " : ""}\`${stepLine(s)}\`${last ? "**" : ""}`);
      out.push("");
    });
  }
  return out.join("\n");
}

/** Quote an argument for a shell command line (POSIX and PowerShell-safe enough for paths). */
export function shellQuote(s: string): string {
  return /^[A-Za-z0-9_./:\\-]+$/.test(s) ? s : `"${s.replace(/(["\\$`])/g, "\\$1")}"`;
}
