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
group-relative z-score within each prompt's G rollouts -- GRPO's critic-free
baseline -- and every decision in a rollout inherits its rollout's advantage.
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
        for task in prompts:
            group = [run_task(task, alloc, executor, verifier, terminal,
                              planner=planner, cfg=cfg, policy_name="grpo",
                              explore=True, update=False)
                     for _ in range(group_size)]
            rollouts += len(group)
            if trace_log is not None:
                for t in group:
                    trace_log.append(t)
            R = [_rollout_reward(t, cfg.budget, cost_weight) for t in group]
            mean = sum(R) / len(R)
            var = sum((r - mean) ** 2 for r in R) / len(R)
            std = math.sqrt(var)
            for t, r in zip(group, R):
                a = (r - mean) / (std + 1e-6)        # group-relative advantage
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
    lines = [f"{'step':>5} {'policy':>6} {'solve':>6} {'mean_cost':>10} "
             f"{'p95_cost':>10} {'gap':>6} {'rollouts':>9}"]
    for r in reports:
        e = r.eval
        lines.append(
            f"{r.step:>5} {r.policy:>6} {e['solve_rate']:>6.2f} "
            f"{e['mean_cost']:>10.5f} {e['p95_cost']:>10.5f} "
            f"{e['gen_verif_gap']:>6.2f} {r.n_rollouts:>9}"
        )
    return "\n".join(lines)
