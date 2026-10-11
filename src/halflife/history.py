"""Run history: compare each run with the previous one, and make badges.

Every run is saved under .halflife/. The next run for the same agent and suite
shows what changed, so halflife is useful every time you touch your agent.
"""

from __future__ import annotations

import glob
import json
import math
import os
from typing import Optional

from .report import Report


def _num(v) -> Optional[float]:
    if v is None:
        return None
    if v == "inf":
        return math.inf
    return float(v)


def previous(out_dir: str, suite: str, agent: str, exclude: Optional[str] = None) -> Optional[dict]:
    """The most recent saved result for this agent and suite, if any."""
    best = None
    for path in glob.glob(os.path.join(out_dir, "*.json")):
        if path.endswith(".evals.json") or path.endswith("badge.json") or (exclude and os.path.abspath(path) == os.path.abspath(exclude)):
            continue
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("suite") != suite or d.get("agent") != agent or "half_life" not in d:
            continue
        if best is None or d.get("created_at", "") > best.get("created_at", ""):
            best = d
    return best


def delta_text(report: Report, prev: Optional[dict]) -> Optional[str]:
    if not prev:
        return None
    old = _num(prev.get("certified_half_life", prev.get("half_life")))
    old_capped = (_num(prev.get("half_life")) or 0) > (old or 0)
    new = report.certified_half_life
    new_capped = report.half_life is not None and new is not None and report.half_life > new
    unit = report.unit
    if old is None or new is None:
        return None
    rate_old, rate_new = prev.get("success_rate", 0.0), report.success_rate

    def fmt(v, capped=False):
        return ("≥ " if capped else "") + ("∞" if math.isinf(v) else f"{v:.1f}")

    if math.isinf(old) or math.isinf(new) or old == 0 or (old_capped and new_capped):
        change = ""
    else:
        pct = (new - old) / old * 100
        if abs(pct) < 3:
            change = " (no meaningful change)"
        else:
            change = f" ({'+' if pct > 0 else '−'}{abs(pct):.0f}%{', better' if pct > 0 else ', worse'})"
    return (
        f"Since last run ({prev.get('created_at', '?')}): half-life {fmt(old, old_capped)} → {fmt(new, new_capped)} {unit}{change}; "
        f"success {rate_old*100:.0f}% → {rate_new*100:.0f}%"
    )


def badge(report_or_dict, label: str = "agent half-life") -> dict:
    """Shields.io endpoint JSON plus a standalone SVG for a README badge."""
    d = report_or_dict.to_dict() if isinstance(report_or_dict, Report) else report_or_dict
    h = _num(d.get("certified_half_life", d.get("half_life")))
    raw = _num(d.get("half_life"))
    unit = d.get("config", {}).get("unit", "steps")
    if h is None:
        message, color = "n/a", "lightgrey"
    elif raw is not None and raw > h:
        message, color = f"≥ {h:.0f} {unit}", "1F4E8C"
    else:
        message, color = f"{h:.0f} {unit}", "1F4E8C"
    endpoint = {"schemaVersion": 1, "label": label, "message": message, "color": color}
    lw, rw = 7 * len(label) + 12, 7 * len(message) + 12
    w = lw + rw
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="20" role="img" aria-label="{label}: {message}">'
        f'<title>{label}: {message}</title><clipPath id="r"><rect width="{w}" height="20" rx="3"/></clipPath>'
        f'<g clip-path="url(#r)"><rect width="{lw}" height="20" fill="#3B424C"/><rect x="{lw}" width="{rw}" height="20" fill="#{color if color != "lightgrey" else "9F9F9F"}"/></g>'
        f'<g fill="#fff" text-anchor="middle" font-family="Verdana,DejaVu Sans,sans-serif" font-size="11">'
        f'<text x="{lw/2:.1f}" y="14">{label}</text><text x="{lw + rw/2:.1f}" y="14">{message}</text></g></svg>'
    )
    return {"endpoint": endpoint, "svg": svg}
