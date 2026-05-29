"""GRPO cost estimator: the locked end-of-project deliverable.

The project trains the Allocator with BC then DPO, both of which learn from a
*fixed* set of already-collected traces -- so their generation cost is paid once,
up front, and the policy update itself is free. GRPO is different: it is on-policy,
so every gradient step must regenerate fresh rollouts with the *current* policy
and score them with the verifier. That on-policy rollout requirement is the entire
cost delta, and it is what makes GRPO expensive on an API-credits budget.

This module makes the estimate measured, not guessed. We read the real per-task
cost out of the collected traces (tokens, wall-seconds, dollars), then extrapolate
to a GRPO training run of a given size. Because a "rollout" in this system is one
full task episode (the Executor + verifier calls the Allocator drives), the
measured mean task cost IS the per-rollout cost. No new assumptions.

The headline number is the ratio: GRPO generation spend / BC+DPO collection spend.
It is typically two to three orders of magnitude, which frames the verdict --
GRPO buys on-policy data, and the question is only whether the accuracy ceiling it
unlocks is worth that multiple.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from wdp.cost.accounting import CURRENCIES
from wdp.loop.trace import TaskTrace


@dataclass
class GRPOConfig:
    """A GRPO training run sized in the usual terms.

    total on-policy rollouts = num_steps * prompts_per_step * group_size
    """
    group_size: int = 8          # G rollouts per prompt for the group-relative baseline
    prompts_per_step: int = 16   # prompts (tasks) sampled per gradient step
    num_steps: int = 200         # gradient steps (on-policy: fresh rollouts each step)
    epochs_per_batch: int = 1    # reuse factor; >1 amortizes generation across updates

    @property
    def total_rollouts(self) -> int:
        gen_steps = max(1, self.num_steps // max(1, self.epochs_per_batch))
        return gen_steps * self.prompts_per_step * self.group_size


@dataclass
class RolloutCost:
    """Mean and p95 cost of one task episode, measured from traces."""
    n: int
    mean: dict = field(default_factory=dict)   # per-currency mean
    p95: dict = field(default_factory=dict)     # per-currency p95


@dataclass
class GRPOEstimate:
    rollout_cost: RolloutCost
    total_rollouts: int
    grpo_cost: dict                 # extrapolated per-currency GRPO generation cost
    bc_dpo_collection_cost: dict    # what we actually paid to collect the BC/DPO data
    cost_ratio: float               # grpo_dollars / bc_dpo_dollars
    stability_notes: list[str]
    verdict: str


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    import math
    idx = min(len(s) - 1, max(0, int(math.ceil(p / 100.0 * len(s))) - 1))
    return float(s[idx])


def measure_rollout_cost(traces: list[TaskTrace]) -> RolloutCost:
    """Per-currency mean and p95 cost of one task episode."""
    if not traces:
        return RolloutCost(n=0, mean={c: 0.0 for c in CURRENCIES},
                           p95={c: 0.0 for c in CURRENCIES})
    cols = {c: [float(t.total_cost.get(c, 0.0)) for t in traces] for c in CURRENCIES}
    mean = {c: (sum(v) / len(v)) for c, v in cols.items()}
    p95 = {c: _percentile(v, 95.0) for c, v in cols.items()}
    return RolloutCost(n=len(traces), mean=mean, p95=p95)


def estimate_grpo(
    traces: list[TaskTrace],
    grpo: GRPOConfig | None = None,
    *,
    bc_dpo_collection_attempts: int | None = None,
) -> GRPOEstimate:
    """Extrapolate GRPO training cost from measured per-rollout cost.

    `bc_dpo_collection_attempts` is the number of task episodes that were run to
    collect the BC/DPO training data (e.g. rounds * |train_tasks|). If omitted we
    use the number of traces provided, since those ARE the collected episodes.
    """
    grpo = grpo or GRPOConfig()
    rc = measure_rollout_cost(traces)
    total = grpo.total_rollouts

    grpo_cost = {c: total * rc.mean[c] for c in CURRENCIES}

    n_collect = bc_dpo_collection_attempts or rc.n
    bc_dpo_cost = {c: n_collect * rc.mean[c] for c in CURRENCIES}

    ratio = (grpo_cost["dollars"] / bc_dpo_cost["dollars"]) if bc_dpo_cost["dollars"] else float("inf")

    stability = [
        "Policy here is a small feature model, so GRPO reduces to a contextual-"
        "bandit policy gradient with a group-relative baseline -- the LLM-scale "
        "instabilities DAPO targets (clip-higher, dynamic sampling, token-level "
        "loss, overlong reward shaping) are largely N/A; the linear policy update "
        "is stable.",
        "The real risk is variance, not divergence: with a small group_size the "
        "group-relative advantage is noisy. Increasing G cuts variance but scales "
        "rollout cost linearly -- the dominant cost lever.",
        "Verifier noise propagates directly into the advantage (no critic to "
        "smooth it), so the generation-verification gap measured in the BC/DPO "
        "runs is a leading indicator of how hard GRPO advantages will be to trust.",
    ]

    verdict = (
        f"GRPO would cost about {ratio:,.0f}x the BC+DPO data-collection spend "
        f"(~${grpo_cost['dollars']:.2f} vs ~${bc_dpo_cost['dollars']:.2f} in "
        f"generation, for {total:,} on-policy rollouts). It is worth running only "
        f"if the BC->DPO self-improvement curve is still climbing at its end "
        f"(headroom to the pass@k coverage ceiling) AND the generation-verification "
        f"gap is small enough to trust on-policy advantages. If the DPO curve has "
        f"flattened, GRPO buys little for a large multiple; if it is still rising, "
        f"a short GRPO run (smaller num_steps) is the cheapest way to probe the "
        f"ceiling before committing."
    )

    return GRPOEstimate(
        rollout_cost=rc, total_rollouts=total, grpo_cost=grpo_cost,
        bc_dpo_collection_cost=bc_dpo_cost, cost_ratio=ratio,
        stability_notes=stability, verdict=verdict,
    )


def format_estimate(est: GRPOEstimate) -> str:
    rc = est.rollout_cost
    lines = [
        "GRPO cost estimate (measured per-rollout cost, extrapolated)",
        "=" * 60,
        f"measured rollouts (task episodes): n={rc.n}",
        f"  mean  per rollout: ${rc.mean['dollars']:.5f}  "
        f"{rc.mean['tokens']:.0f} tok  {rc.mean['latency']:.2f} s",
        f"  p95   per rollout: ${rc.p95['dollars']:.5f}  "
        f"{rc.p95['tokens']:.0f} tok  {rc.p95['latency']:.2f} s",
        "",
        f"GRPO run: {est.total_rollouts:,} on-policy rollouts",
        f"  est dollars : ${est.grpo_cost['dollars']:.2f}",
        f"  est tokens  : {est.grpo_cost['tokens']:,.0f}",
        f"  est latency : {est.grpo_cost['latency'] / 3600.0:.2f} h (serial)",
        "",
        f"BC+DPO collection: ${est.bc_dpo_collection_cost['dollars']:.2f}",
        f"cost ratio (GRPO / BC+DPO generation): {est.cost_ratio:,.0f}x",
        "",
        "stability notes:",
    ]
    lines += [f"  - {n}" for n in est.stability_notes]
    lines += ["", "verdict:", f"  {est.verdict}"]
    return "\n".join(lines)
