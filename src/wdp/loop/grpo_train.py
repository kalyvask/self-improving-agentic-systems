"""GRPO on-policy training loop for the Allocator.

Each step: sample `prompts_per_step` prompts, roll each out `group_size` times with
the CURRENT policy, score each rollout, turn the group-relative reward into an
advantage, and take one GRPO update (a few cheap inner epochs over that fresh
batch). Total rollouts = num_steps * prompts_per_step * group_size, the exact unit
the GRPO cost estimator prices.

The rollout reward is outcome times cost-efficiency,
    R = terminal_reward * exp(-cost_weight * spent/budget),
the same cost-aware signal `assign_credit` uses, so GRPO optimizes the project's
thesis metric (cheap solves) rather than raw success. Advantage is the
mean-centered group-relative reward within each prompt's G rollouts -- GRPO's
critic-free baseline, with the std division dropped (Dr. GRPO; dividing by a tiny
all-solve-group std amplified cost jitter into spurious advantages) -- and every
decision in a rollout inherits its rollout's advantage.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from wdp.allocator.grpo import GRPOAllocator
from wdp.allocator.bc import _INDEX
from wdp.executor.react import Executor, Task
from wdp.loop.runner import RunConfig, run_round, run_task
from wdp.metrics import summarize_round


@dataclass
class GRPOReport:
    step: int
    policy: str
    eval: dict
    n_rollouts: int


def _rollout_reward(trace, budget: float, cost_weight: float) -> float:
    spent = (trace.total_cost or {}).get(trace.currency, 0.0)
    eff = math.exp(-cost_weight * (spent / budget)) if budget and budget > 0 else 1.0
    return float(trace.terminal_reward) * eff


def grpo_train(
    seed_traces,
    train_tasks: list[Task],
    eval_tasks: list[Task],
    executor: Executor,
    verifier,
    terminal,
    *,
    planner=None,
    cfg: RunConfig | None = None,
    group_size: int = 8,
    prompts_per_step: int = 16,
    num_steps: int = 40,
    inner_epochs: int = 5,
    beta_kl: float = 0.05,
    cost_weight: float = 0.5,
    dynamic_sampling: bool = True,
    eval_every: int = 10,
    seed: int = 0,
    trace_log=None,
):
    """Run GRPO from a BC warm start; return (reports, allocator).

    `seed_traces` are existing traces used only to fit the BC reference / warm
    start (no extra spend). The live spend is the on-policy rollouts.
    """
    cfg = cfg or RunConfig()
    rng = random.Random(seed)
    alloc = GRPOAllocator(seed=seed)
    alloc.warm_start(seed_traces)

    def _eval(step: int, n_rollouts: int) -> GRPOReport:
        ev = run_round(eval_tasks, alloc, executor, verifier, terminal,
                       planner=planner, cfg=cfg, policy_name="grpo",
                       explore=False, update=False)
        return GRPOReport(step=step, policy="grpo",
                          eval=summarize_round(ev, cfg.currency), n_rollouts=n_rollouts)

    reports = [_eval(0, 0)]          # step 0 = the warm-started BC policy, greedy
    rollouts = 0
    for step in range(1, num_steps + 1):
        if prompts_per_step <= len(train_tasks):
            prompts = rng.sample(train_tasks, prompts_per_step)
        else:
            prompts = [rng.choice(train_tasks) for _ in range(prompts_per_step)]

        X, actions, advs = [], [], []
        kept_groups = 0
        for task in prompts:
            group = [run_task(task, alloc, executor, verifier, terminal,
                              planner=planner, cfg=cfg, policy_name="grpo",
                              explore=True, update=False)
                     for _ in range(group_size)]
            rollouts += len(group)
            if trace_log is not None:
                for t in group:
                    trace_log.append(t)
            # DAPO dynamic sampling (arXiv:2503.14476): a group whose rollouts ALL
            # solve or ALL fail carries no outcome signal -- its only within-group
            # variation is cost jitter, which (even mean-centered) imparts a small
            # but consistent pull toward the cheapest action and suppresses the
            # necessary-but-expensive one (DECOMPOSE), collapsing the policy. Skip
            # those groups so only outcome-varying groups train; the surviving
            # mixed groups still carry cost signal via the reward magnitude.
            solved = [1 if (t.solved or t.terminal_reward >= 0.99) else 0 for t in group]
            if dynamic_sampling and (sum(solved) == 0 or sum(solved) == len(group)):
                continue
            kept_groups += 1
            R = [_rollout_reward(t, cfg.budget, cost_weight) for t in group]
            mean = sum(R) / len(R)
            # Advantage is mean-centered only -- NOT divided by the group std.
            # Dividing by std (vanilla GRPO) over-weights low-variance groups: when
            # all G rollouts solve, the only variation is cost jitter (std ~ 0.02),
            # and 1/std amplifies that jitter into advantages LARGER than the real
            # solve-vs-fail signal from mixed groups (std ~ 0.32). That drove the
            # policy toward the cheapest action (WIDER) on amplified noise and
            # collapsed solve rate. Dropping the std division (Dr. GRPO, Liu et al.
            # arXiv:2503.20783) removes that bias: mixed groups, with the largest
            # reward gap, correctly dominate the gradient.
            for t, r in zip(group, R):
                a = r - mean                         # group-relative advantage
                for d in t.decisions:
                    if d.action in _INDEX and d.features:
                        X.append(d.features)
                        actions.append(_INDEX[d.action])
                        advs.append(a)

        if X:
            alloc.policy.grpo_update(X, actions, advs,
                                     reference=alloc.reference.policy,
                                     beta_kl=beta_kl, inner_epochs=inner_epochs)
        if step % eval_every == 0 or step == num_steps:
            reports.append(_eval(step, rollouts))

    return reports, alloc


def format_grpo_curve(reports: list[GRPOReport]) -> str:
    # 'util' = solved-or-correctly-abstained (the metric that tracks frontier quality).
    # We deliberately do NOT surface gen_verif_gap here: completed traces now use the
    # exact terminal reward as their process score, so that "gap" no longer measures a
    # cheap-verifier disagreement and would mislead. It stays computed in agent_metrics
    # for partial-trace diagnostics only.
    lines = [f"{'step':>5} {'policy':>6} {'solve':>6} {'mean_cost':>10} "
             f"{'p95_cost':>10} {'util':>6} {'rollouts':>9}"]
    for r in reports:
        e = r.eval
        lines.append(
            f"{r.step:>5} {r.policy:>6} {e['solve_rate']:>6.2f} "
            f"{e['mean_cost']:>10.5f} {e['p95_cost']:>10.5f} "
            f"{e.get('utility_rate', 0.0):>6.2f} {r.n_rollouts:>9}"
        )
    return "\n".join(lines)
