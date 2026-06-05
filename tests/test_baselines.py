"""Offline tests for the non-learned and online-contextual baselines.

These need no API: ConstantAllocator is trivial, and LinUCBAllocator is exercised
on a synthetic world with a known context->best-arm rule, so we can assert it
concentrates on the right arm purely from simulated rewards.
"""
from __future__ import annotations

import numpy as np

from wdp.allocator import (
    Action,
    ConstantAllocator,
    LinUCBAllocator,
    NodeFeatures,
)
from wdp.allocator.policy import BANDIT_ARMS
from wdp.loop.trace import DecisionRecord, TaskTrace


# --------------------------------------------------------------------------- #
# ConstantAllocator
# --------------------------------------------------------------------------- #
def test_constant_allocator_returns_its_action():
    for a in (Action.WIDER, Action.DEEPER, Action.STOP, Action.ESCALATE):
        alloc = ConstantAllocator(a)
        d = alloc.decide(NodeFeatures(score_max=0.5, budget_remaining_frac=0.5), "dollars")
        assert d.action == a
        # Every action must appear in scores so the runner's mask-and-repick works.
        assert set(d.scores.keys()) == set(Action)


# --------------------------------------------------------------------------- #
# LinUCBAllocator
# --------------------------------------------------------------------------- #
def test_linucb_decide_is_valid_and_scores_all_arms():
    alloc = LinUCBAllocator(seed=0)
    d = alloc.decide(NodeFeatures(score_max=0.9, budget_remaining_frac=0.2), "dollars")
    assert d.action in set(Action)
    # Scores must cover the contextual arms + STOP (so masking can re-pick).
    for a in BANDIT_ARMS:
        assert a in d.scores
    assert Action.STOP in d.scores


def test_linucb_learns_feature_conditioned_arm():
    """Online world: DEEPER is optimal when score_max > budget_remaining_frac,
    else WIDER. Reward 1.0 for the optimal arm, 0.1 otherwise. After online
    collection, the greedy (eval) policy should recover the rule well above the
    1/len(arms) random baseline."""
    rng = np.random.default_rng(0)
    alloc = LinUCBAllocator(alpha=1.0, seed=0)

    def optimal(smax: float, brem: float) -> Action:
        return Action.DEEPER if smax > brem else Action.WIDER

    # Collect online with exploration on.
    for _ in range(800):
        smax, brem = rng.uniform(0, 1), rng.uniform(0, 1)
        nf = NodeFeatures(score_max=smax, budget_remaining_frac=brem)
        d = alloc.decide(nf, "dollars", explore=True)
        reward = 1.0 if d.action == optimal(smax, brem) else 0.1
        alloc.update(d.action, reward)

    # Greedy eval accuracy on fresh contexts.
    hits, n = 0, 400
    for _ in range(n):
        smax, brem = rng.uniform(0, 1), rng.uniform(0, 1)
        nf = NodeFeatures(score_max=smax, budget_remaining_frac=brem)
        if alloc.decide(nf, "dollars", explore=False).action == optimal(smax, brem):
            hits += 1
    acc = hits / n
    assert acc > 0.6, f"LinUCB accuracy {acc:.2f} not above the random/avg baseline"


def test_linucb_warm_start_from_traces_runs():
    """fit() should replay logged decisions through the online update without error
    and leave a usable policy."""
    rng = np.random.default_rng(1)
    traces = []
    for i in range(20):
        recs = []
        for _ in range(3):
            nf = NodeFeatures(score_max=rng.uniform(0, 1),
                              budget_remaining_frac=rng.uniform(0, 1))
            recs.append(DecisionRecord(
                step=0, features=nf.vector().tolist(),
                action=Action.DEEPER.value, value_per_cost=float(rng.uniform(0, 1))))
        traces.append(TaskTrace(task_id=f"t{i}", currency="dollars", policy="gen",
                                decisions=recs))
    alloc = LinUCBAllocator(seed=0)
    alloc.fit(traces)  # must not raise
    d = alloc.decide(NodeFeatures(score_max=0.8, budget_remaining_frac=0.3), "dollars")
    assert d.action in set(Action)


def test_linucb_snapshot_restore_roundtrip():
    a1 = LinUCBAllocator(seed=0)
    for _ in range(50):
        nf = NodeFeatures(score_max=0.7, budget_remaining_frac=0.3)
        d = a1.decide(nf, "dollars", explore=True)
        a1.update(d.action, 0.8)
    a2 = LinUCBAllocator(seed=0)
    a2.restore(a1.snapshot())
    # The LEARNED state (A/b -> theta -> value estimates) must round-trip exactly.
    # The exploration RNG is deliberately not persisted (you restore knowledge, not
    # the random stream), so compare the deterministic greedy value estimates, not
    # the tie-broken action.
    nf = NodeFeatures(score_max=0.6, budget_remaining_frac=0.4)
    s1 = a1.decide(nf, "dollars", explore=False).scores
    s2 = a2.decide(nf, "dollars", explore=False).scores
    for a in BANDIT_ARMS:
        assert abs(s1[a] - s2[a]) < 1e-9


def test_linucb_save_load_file_roundtrip(tmp_path):
    """save() then load() into a fresh allocator reproduces the value estimates
    (the 'keeps improving session to session' persistence story)."""
    a1 = LinUCBAllocator(seed=0)
    for _ in range(40):
        nf = NodeFeatures(score_max=0.7, budget_remaining_frac=0.3)
        d = a1.decide(nf, "dollars", explore=True)
        a1.update(d.action, 0.8)
    path = tmp_path / "linucb_state.json"
    a1.save(path)
    a2 = LinUCBAllocator(seed=0)
    a2.load(path)
    nf = NodeFeatures(score_max=0.6, budget_remaining_frac=0.4)
    s1 = a1.decide(nf, "dollars", explore=False).scores
    s2 = a2.decide(nf, "dollars", explore=False).scores
    for a in BANDIT_ARMS:
        assert abs(s1[a] - s2[a]) < 1e-9


def test_linear_policy_save_load_roundtrip(tmp_path):
    """A fitted BC/DPO/KTO core round-trips through save/load (so an allocator can
    be reloaded across runs without re-fitting)."""
    from wdp.allocator import BCAllocator, NodeFeatures as NF
    from wdp.loop.trace import DecisionRecord, TaskTrace
    rng = np.random.default_rng(2)
    traces = []
    for i in range(60):
        nf = NF(score_max=rng.uniform(0, 1), budget_remaining_frac=rng.uniform(0, 1))
        traces.append(TaskTrace(task_id=f"t{i}", currency="dollars", policy="gen", solved=True,
                                decisions=[DecisionRecord(step=0, features=nf.vector().tolist(),
                                                          action="deeper", value_per_cost=0.9)]))
    bc = BCAllocator(keep_fraction=0.5, seed=0)
    bc.fit(traces)
    path = tmp_path / "bc_policy.json"
    bc.policy.save(path)
    bc2 = BCAllocator(keep_fraction=0.5, seed=0)
    bc2.policy.load(path)
    nf = NF(score_max=0.8, budget_remaining_frac=0.2)
    p1 = bc.policy.probs(nf.vector())
    p2 = bc2.policy.probs(nf.vector())
    assert np.allclose(p1, p2)
