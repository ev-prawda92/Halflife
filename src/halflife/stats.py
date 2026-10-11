"""Statistics: fitting a reliability half-life from success rates at different task lengths.

Model: success(L) = 2 ** (-(L / h) ** k)

  h, the half-life: the task length at which success falls to 50%.
  k, the decay shape:
     k = 1   constant risk per step (success = p ** L)
     k > 1   compounding: each step gets riskier as the task grows
     k < 1   front-loaded: failures come early; runs that survive keep going

Both are fit by maximum likelihood over every run. The interval comes from
bootstrapping runs within each length. We also report where the observed
success rate crosses 50%, straight from the data, as a model-free check.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, int, int]

K_BOUNDS = (0.2, 6.0)


def _loglik(h: float, k: float, points: Iterable[Point]) -> float:
    total = 0.0
    for length, successes, n in points:
        q = 2.0 ** (-((length / h) ** k))
        q = min(max(q, 1e-12), 1 - 1e-12)
        total += successes * math.log(q) + (n - successes) * math.log(1 - q)
    return total


def _best_h(k: float, points: Sequence[Point], iters: int) -> Tuple[float, float]:
    a, b = math.log(0.05), math.log(1e6)
    for _ in range(iters):
        m1 = a + (b - a) / 3
        m2 = b - (b - a) / 3
        if _loglik(math.exp(m1), k, points) < _loglik(math.exp(m2), k, points):
            a = m1
        else:
            b = m2
    h = math.exp((a + b) / 2)
    return h, _loglik(h, k, points)


def fit_decay(points: Sequence[Point], shape: Optional[float] = None, iters: int = 70) -> Tuple[Optional[float], Optional[float]]:
    """Maximum-likelihood (half_life, shape).

    Returns (inf, None) if every run succeeded, (0.0, None) if every run failed.
    Pass shape=1.0 to fit the constant-risk model only.
    """
    n_total = sum(n for _, _, n in points)
    if n_total == 0:
        return (None, None)
    s_total = sum(s for _, s, _ in points)
    if s_total == n_total:
        return (math.inf, None)
    if s_total == 0:
        return (0.0, None)
    if shape is not None or len({p[0] for p in points}) < 2:
        k = 1.0 if shape is None else shape
        return (_best_h(k, points, iters)[0], k)

    a, b = math.log(K_BOUNDS[0]), math.log(K_BOUNDS[1])
    for _ in range(40):
        m1 = a + (b - a) / 3
        m2 = b - (b - a) / 3
        if _best_h(math.exp(m1), points, iters)[1] < _best_h(math.exp(m2), points, iters)[1]:
            a = m1
        else:
            b = m2
    k = math.exp((a + b) / 2)
    return (_best_h(k, points, iters)[0], k)


def fit_half_life(points: Sequence[Point]) -> Optional[float]:
    return fit_decay(points)[0]


def success_at(length: float, half_life: float, shape: Optional[float]) -> float:
    if math.isinf(half_life):
        return 1.0
    if half_life <= 0:
        return 0.0
    return 2.0 ** (-((length / half_life) ** (shape or 1.0)))


def shape_text(shape: Optional[float]) -> str:
    if shape is None:
        return "n/a"
    if shape >= K_BOUNDS[1] * 0.97:
        return "cliff: reliable up to a point, then breaks"
    if shape < 0.85:
        return "front-loaded: failures come early"
    if shape <= 1.15:
        return "steady: constant risk per step"
    return "compounding: errors grow with length"


def bootstrap_ci(
    outcomes: Dict[float, List[bool]], samples: int = 150, seed: int = 0
) -> Tuple[Optional[float], Optional[float]]:
    """95% interval for the half-life, by resampling runs within each length."""
    if not outcomes:
        return (None, None)
    rng = random.Random(seed)
    estimates: List[float] = []
    for _ in range(samples):
        points = []
        for length, runs in outcomes.items():
            draw = [runs[rng.randrange(len(runs))] for _ in runs]
            points.append((length, sum(draw), len(draw)))
        h = fit_decay(points, iters=50)[0]
        if h is not None:
            estimates.append(h)
    if not estimates:
        return (None, None)
    estimates.sort()
    lo = estimates[int(0.025 * len(estimates))]
    hi = estimates[min(len(estimates) - 1, int(0.975 * len(estimates)))]
    return (lo, hi)


def wilson(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def observed_crossing(rows: Sequence[Tuple[float, float]]) -> Tuple[str, float]:
    """Where the observed success rate crosses 50%, by linear interpolation.

    Returns ("at", L), ("below", first_length) if already under 50% at the shortest
    length, or ("above", last_length) if it never drops below 50%.
    """
    rows = sorted(rows)
    if not rows:
        return ("above", 0.0)
    if rows[0][1] < 0.5:
        return ("below", rows[0][0])
    if rows[0][1] == 0.5:
        return ("at", rows[0][0])
    for (l1, r1), (l2, r2) in zip(rows, rows[1:]):
        if r1 > 0.5 >= r2:
            return ("at", l1 + (r1 - 0.5) / (r1 - r2) * (l2 - l1))
    return ("above", rows[-1][0])
