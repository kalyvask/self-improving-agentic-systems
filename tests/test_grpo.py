"""Offline tests for the GRPO cost estimator."""
from __future__ import annotations

from wdp.loop.trace import TaskTrace
from wdp.grpo import GRPOConfig, estimate_grpo, measure_rollout_cost, format_estimate


def _traces(costs):
    out = []
    for i, (tok, lat, doll) in enumerate(costs):
        out.append(TaskTrace(task_id=f"t{i}", currency="dollars", policy="bandit",
                             total_cost={"tokens": tok, "latency": lat, "dollars": doll}))
    return out


def test_measure_rollout_cost():
    rc = measure_rollout_cost(_traces([(100, 1.0, 0.001), (300, 3.0, 0.003)]))
    assert rc.n == 2
    assert abs(rc.mean["dollars"] - 0.002) < 1e-9
    assert abs(rc.mean["tokens"] - 200.0) < 1e-9
    assert rc.p95["dollars"] == 0.003


def test_total_rollouts_and_scaling():
    grpo = GRPOConfig(group_size=8, prompts_per_step=16, num_steps=200)
    assert grpo.total_rollouts == 200 * 16 * 8
    est = estimate_grpo(_traces([(100, 1.0, 0.002)] * 10), grpo)
    # per-rollout dollars 0.002, total rollouts 25600
    assert abs(est.grpo_cost["dollars"] - 0.002 * grpo.total_rollouts) < 1e-6
    # collection defaults to #traces = 10
    assert abs(est.bc_dpo_collection_cost["dollars"] - 0.002 * 10) < 1e-9
    assert est.cost_ratio == grpo.total_rollouts / 10


def test_epochs_amortize_generation():
    base = GRPOConfig(num_steps=200, epochs_per_batch=1)
    amort = GRPOConfig(num_steps=200, epochs_per_batch=4)
    assert amort.total_rollouts == base.total_rollouts // 4


def test_format_runs():
    est = estimate_grpo(_traces([(100, 1.0, 0.002)] * 5))
    text = format_estimate(est)
    assert "GRPO cost estimate" in text
    assert "verdict:" in text
    assert "cost ratio" in text
