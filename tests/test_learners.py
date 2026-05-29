"""Offline tests for the trainable Allocators (BC, DPO).

We build a synthetic world with a known, linearly-separable optimal policy: each
action is favored by one feature. Decisions take a random action and are credited
value-per-cost 1.0 when that action was optimal, 0.1 otherwise. A correct learner
should recover the optimal action far better than the 0.25 random baseline, both
from cloning (BC) and from preferences (DPO).
"""
from __future__ import annotations

import numpy as np

from wdp.allocator import BCAllocator, DPOAllocator, ACTIONS, Action, NodeFeatures
from wdp.loop.trace import TaskTrace, DecisionRecord

_NAMES = NodeFeatures.names()
_IDX = {n: i for i, n in enumerate(_NAMES)}


def _optimal(feat: np.ndarray) -> int:
    """argmax over four linearly-separable action affinities."""
    scores = [
        feat[_IDX["budget_remaining_frac"]],   # WIDER
        feat[_IDX["score_max"]],                # DEEPER
        feat[_IDX["decomposability"]],          # DECOMPOSE
        1.0 - feat[_IDX["budget_remaining_frac"]],  # STOP
    ]
    return int(np.argmax(scores))


def _make_traces(n: int, seed: int = 0) -> list[TaskTrace]:
    rng = np.random.default_rng(seed)
    traces: list[TaskTrace] = []
    for i in range(n):
        feat = np.zeros(len(_NAMES))
        feat[_IDX["budget_remaining_frac"]] = rng.uniform(0, 1)
        feat[_IDX["score_max"]] = rng.uniform(0, 1)
        feat[_IDX["score_mean"]] = feat[_IDX["score_max"]] * rng.uniform(0.5, 1)
        feat[_IDX["decomposability"]] = rng.uniform(0, 1)
        opt = _optimal(feat)
        taken = int(rng.integers(0, len(ACTIONS)))
        vpc = 1.0 if taken == opt else 0.1
        rec = DecisionRecord(step=0, features=feat.tolist(),
                             action=ACTIONS[taken].value, value_per_cost=vpc)
        traces.append(TaskTrace(task_id=f"t{i}", currency="dollars", policy="gen",
                               decisions=[rec]))
    return traces


def _accuracy(alloc, n: int = 400, seed: int = 99) -> float:
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        feat = np.zeros(len(_NAMES))
        feat[_IDX["budget_remaining_frac"]] = rng.uniform(0, 1)
        feat[_IDX["score_max"]] = rng.uniform(0, 1)
        feat[_IDX["decomposability"]] = rng.uniform(0, 1)
        opt = ACTIONS[_optimal(feat)]
        nf = NodeFeatures(
            score_max=feat[_IDX["score_max"]],
            budget_remaining_frac=feat[_IDX["budget_remaining_frac"]],
            decomposability=feat[_IDX["decomposability"]],
        )
        if alloc.decide(nf, "dollars").action == opt:
            hits += 1
    return hits / n


def test_bc_recovers_policy_above_baseline():
    traces = _make_traces(800, seed=1)
    bc = BCAllocator(keep_fraction=0.4, seed=0)
    bc.fit(traces)
    acc = _accuracy(bc)
    assert acc > 0.6, f"BC accuracy {acc:.2f} not above baseline"


def test_bc_decide_returns_normalized_scores():
    traces = _make_traces(400, seed=2)
    bc = BCAllocator(keep_fraction=0.4, seed=0)
    bc.fit(traces)
    d = bc.decide(NodeFeatures(score_max=0.95, budget_remaining_frac=0.9), "dollars")
    assert d.action in ACTIONS
    assert abs(sum(d.scores.values()) - 1.0) < 1e-6


def test_dpo_recovers_policy_above_baseline():
    traces = _make_traces(800, seed=3)
    dpo = DPOAllocator(keep_fraction=0.4, beta=0.1, seed=0)
    dpo.fit(traces)
    acc = _accuracy(dpo)
    assert acc > 0.6, f"DPO accuracy {acc:.2f} not above baseline"


def test_dpo_falls_back_to_reference_when_no_pairs():
    # All decisions take the same action -> no contrastable pairs.
    traces = _make_traces(50, seed=4)
    for tr in traces:
        for d in tr.decisions:
            d.action = Action.WIDER.value
    dpo = DPOAllocator(keep_fraction=0.5, seed=0)
    dpo.fit(traces)  # must not raise; degenerates to BC reference
    d = dpo.decide(NodeFeatures(budget_remaining_frac=0.9), "dollars")
    assert d.action in ACTIONS


def test_unfitted_raises():
    import pytest
    with pytest.raises(RuntimeError):
        BCAllocator().decide(NodeFeatures(), "dollars")
