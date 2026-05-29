"""Agent evaluation metrics, computed over collected TaskTraces.

These are the headline numbers, chosen to dodge the traps the course flagged:

  - success_at_budget: the project's primary metric. Fraction solved when each
    task is capped at a fixed budget in one currency. Reported as a curve over
    budgets and per currency, because the whole thesis is that the optimal policy
    differs by currency.
  - pass_hat_k (pass^k): reliability -- did ALL k attempts succeed. The honest
    consistency metric (tau-bench's). Distinct from pass@k.
  - pass_at_k: coverage -- did ANY of k succeed. Kept only as a *diagnostic*
    ceiling, since it conflates generation with selection and ignores cost.
  - risk_coverage: from STOP/abstention. Sort answered tasks by confidence; plot
    accuracy vs coverage. A good Allocator's STOP arm should bend this upward.
  - cvar / p95: tail cost. A policy can win on mean cost and still be unshippable
    if its p95 blows the budget; CVaR captures that.
  - generation_verification_gap: process-score vs terminal-reward divergence --
    measures how much selection (not generation) is leaving on the table.
  - metr_horizon: stub for the task-horizon-at-50%-reliability headline.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from wdp.loop.trace import TaskTrace


@dataclass
class CurvePoint:
    x: float
    y: float


def success_at_budget(traces: list[TaskTrace], budgets: list[float],
                      currency: str = "dollars") -> list[CurvePoint]:
    """Fraction of tasks solved within each budget cap (in `currency`)."""
    pts: list[CurvePoint] = []
    for b in budgets:
        if not traces:
            pts.append(CurvePoint(b, 0.0))
            continue
        hits = sum(
            1 for t in traces
            if t.solved and t.total_cost.get(currency, math.inf) <= b
        )
        pts.append(CurvePoint(b, hits / len(traces)))
    return pts


def pass_hat_k(per_task_successes: list[list[bool]]) -> float:
    """pass^k: fraction of tasks where ALL attempts succeeded (reliability)."""
    if not per_task_successes:
        return 0.0
    return statistics.fmean(1.0 if all(a) and a else 0.0 for a in per_task_successes)


def pass_at_k(per_task_successes: list[list[bool]]) -> float:
    """pass@k: fraction of tasks where ANY attempt succeeded (coverage ceiling)."""
    if not per_task_successes:
        return 0.0
    return statistics.fmean(1.0 if any(a) else 0.0 for a in per_task_successes)


def risk_coverage(answered: list[tuple[float, bool]]) -> list[CurvePoint]:
    """Risk-coverage curve. `answered` = (confidence, correct) for non-abstained
    tasks. Returns (coverage, accuracy) sorted by descending confidence."""
    if not answered:
        return []
    ranked = sorted(answered, key=lambda x: x[0], reverse=True)
    pts: list[CurvePoint] = []
    correct = 0
    for i, (_, ok) in enumerate(ranked, start=1):
        correct += 1 if ok else 0
        pts.append(CurvePoint(x=i / len(ranked), y=correct / i))
    return pts


def cvar(costs: list[float], alpha: float = 0.95) -> float:
    """Conditional value-at-risk: mean of the worst (1-alpha) tail of costs."""
    if not costs:
        return 0.0
    ranked = sorted(costs)
    cutoff = int(math.ceil(alpha * len(ranked)))
    tail = ranked[cutoff:] or ranked[-1:]
    return float(statistics.fmean(tail))


def percentile(costs: list[float], p: float = 95.0) -> float:
    if not costs:
        return 0.0
    ranked = sorted(costs)
    idx = min(len(ranked) - 1, int(math.ceil(p / 100.0 * len(ranked))) - 1)
    return float(ranked[max(0, idx)])


def generation_verification_gap(traces: list[TaskTrace]) -> float:
    """Mean |best process score seen - terminal reward| across tasks. High gap =
    the verifier the Allocator acts on disagrees with ground truth = selection,
    not generation, is the limiter."""
    gaps: list[float] = []
    for t in traces:
        best_ps = max((d.process_score_after for d in t.decisions), default=0.0)
        gaps.append(abs(best_ps - t.terminal_reward))
    return float(statistics.fmean(gaps)) if gaps else 0.0


def metr_horizon(task_minutes: list[float], successes: list[bool],
                target_reliability: float = 0.5) -> float:
    """METR task-horizon stub: the human-time-length at which the agent crosses
    `target_reliability`. Bins tasks by their human-time estimate and returns the
    longest bin still at/above the target. Returns 0.0 if it never clears it."""
    if not task_minutes or len(task_minutes) != len(successes):
        return 0.0
    paired = sorted(zip(task_minutes, successes), key=lambda x: x[0])
    horizon = 0.0
    window: list[bool] = []
    for minutes, ok in paired:
        window.append(ok)
        if statistics.fmean(1.0 if w else 0.0 for w in window) >= target_reliability:
            horizon = minutes
    return float(horizon)


def summarize_round(traces: list[TaskTrace], currency: str = "dollars") -> dict:
    """One-line-per-round scoreboard used to plot the self-improvement curve."""
    costs = [t.total_cost.get(currency, 0.0) for t in traces]
    solved = [t for t in traces if t.solved]
    return {
        "n": len(traces),
        "solve_rate": (len(solved) / len(traces)) if traces else 0.0,
        "mean_cost": float(statistics.fmean(costs)) if costs else 0.0,
        "p95_cost": percentile(costs, 95.0),
        "cvar95_cost": cvar(costs, 0.95),
        "gen_verif_gap": generation_verification_gap(traces),
        "currency": currency,
    }
