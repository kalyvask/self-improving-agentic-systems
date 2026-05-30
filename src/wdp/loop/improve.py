"""The self-improvement driver: the project's headline experiment.

This is where "self-improving" stops being a label and becomes a measured curve.
The Allocator improves its own compute policy from its own logged traces:

  round 0   BanditAllocator cold-starts (no data) and collects traces.
  round r   fit a fresh BC or DPO policy on ALL accumulated traces, run it to
            collect more traces, and evaluate it on a held-out task set.

The output is one scoreboard per round (solve rate, mean / p95 / CVaR cost,
generation-verification gap). Plotting eval solve-rate-per-cost across rounds is
the self-improvement curve; if BC/DPO is working it bends up and to the left
(more solves, less spend) relative to the round-0 bandit. The same accumulated
traces are what the end-of-project GRPO estimate extrapolates from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from wdp.allocator.policy import BanditAllocator
from wdp.allocator.bc import BCAllocator
from wdp.allocator.dpo import DPOAllocator
from wdp.allocator.kto import KTOAllocator
from wdp.executor.react import Executor, Task
from wdp.loop.runner import RunConfig, run_round
from wdp.loop.trace import TaskTrace
from wdp.metrics import summarize_round


@dataclass
class RoundReport:
    round: int
    policy: str
    eval: dict
    train: dict
    n_accumulated_traces: int


def _make_learner(kind: str, keep_fraction: float, seed: int | None):
    if kind == "bc":
        return BCAllocator(keep_fraction=keep_fraction, seed=seed)
    if kind == "dpo":
        return DPOAllocator(keep_fraction=keep_fraction, seed=seed)
    if kind == "kto":
        return KTOAllocator(keep_fraction=keep_fraction, seed=seed)
    raise ValueError(f"unknown learner {kind!r}; expected 'bc', 'dpo', or 'kto'")


def self_improve(
    train_tasks: list[Task],
    eval_tasks: list[Task],
    executor: Executor,
    verifier,
    terminal,
    *,
    planner=None,
    learner: str = "bc",
    rounds: int = 3,
    cfg: RunConfig | None = None,
    keep_fraction: float = 0.3,
    seed: int | None = 0,
    fit_window: int | None = None,
    trace_log=None,
    eval_trace_log=None,
    difficulty_fn=None,
) -> list[RoundReport]:
    cfg = cfg or RunConfig()
    accumulated: list[TaskTrace] = []
    reports: list[RoundReport] = []

    def _fit_set() -> list[TaskTrace]:
        # Fit on the most recent `fit_window` rounds of traces (on-policy-ish),
        # or on everything when no window is set. A window stops stale round-0
        # bandit traces from diluting a policy that has since moved on.
        if not fit_window:
            return accumulated
        return accumulated[-fit_window * len(train_tasks):]

    def _eval(alloc, name, rnd) -> dict:
        # Eval is greedy (explore=False): we measure the policy we would deploy,
        # not the exploring data-collection policy. Eval traces are tagged with the
        # round (name@rN) so persisted eval files keep each round distinct -- without
        # the tag, every round writes the same policy label and grouping by policy in
        # analysis silently overwrites/mixes intermediate rounds.
        ev = run_round(eval_tasks, alloc, executor, verifier, terminal,
                       planner=planner, cfg=cfg, policy_name=f"{name}@r{rnd}",
                       explore=False, update=False, difficulty_fn=difficulty_fn,
                       trace_log=eval_trace_log)
        return summarize_round(ev, cfg.currency)

    # Round 0: bandit cold start.
    bandit = BanditAllocator(seed=seed)
    train_traces = run_round(train_tasks, bandit, executor, verifier, terminal,
                             planner=planner, cfg=cfg, policy_name="bandit",
                             explore=True, trace_log=trace_log, difficulty_fn=difficulty_fn)
    accumulated += train_traces
    reports.append(RoundReport(
        round=0, policy="bandit",
        train=summarize_round(train_traces, cfg.currency),
        eval=_eval(bandit, "bandit", 0),
        n_accumulated_traces=len(accumulated),
    ))

    # Rounds 1..R: refit the learner on accumulated traces, collect, evaluate.
    for r in range(1, rounds + 1):
        alloc = _make_learner(learner, keep_fraction, seed)
        alloc.fit(_fit_set())
        train_traces = run_round(train_tasks, alloc, executor, verifier, terminal,
                                 planner=planner, cfg=cfg, policy_name=learner,
                                 explore=True, trace_log=trace_log, difficulty_fn=difficulty_fn)
        accumulated += train_traces
        reports.append(RoundReport(
            round=r, policy=learner,
            train=summarize_round(train_traces, cfg.currency),
            eval=_eval(alloc, learner, r),
            n_accumulated_traces=len(accumulated),
        ))

    return reports


def format_curve(reports: list[RoundReport]) -> str:
    """One line per round: the self-improvement scoreboard, plain text."""
    lines = [f"{'round':>5} {'policy':>7} {'solve':>6} {'mean_cost':>10} "
             f"{'p95_cost':>10} {'gap':>6} {'traces':>7}"]
    for rep in reports:
        e = rep.eval
        lines.append(
            f"{rep.round:>5} {rep.policy:>7} {e['solve_rate']:>6.2f} "
            f"{e['mean_cost']:>10.5f} {e['p95_cost']:>10.5f} "
            f"{e['gen_verif_gap']:>6.2f} {rep.n_accumulated_traces:>7}"
        )
    return "\n".join(lines)
