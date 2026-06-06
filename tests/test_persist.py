"""Offline tests for policy persistence (save/load + schema-drift guard)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from wdp.allocator.bc import ACTIONS
from wdp.allocator.linear import LinearSoftmaxPolicy
from wdp.allocator.persist import FrozenLinearAllocator, load_policy, save_policy
from wdp.allocator.policy import Action, BanditAllocator, NodeFeatures


def _fitted_policy() -> LinearSoftmaxPolicy:
    p = LinearSoftmaxPolicy(n_features=len(NodeFeatures.names()), n_actions=len(ACTIONS), seed=0)
    rng = np.random.default_rng(0)
    p.W = rng.normal(size=p.W.shape)
    p.b = rng.normal(size=p.b.shape)
    p._fitted = True
    return p


def test_linear_policy_roundtrip_preserves_decision(tmp_path):
    p = _fitted_policy()
    alloc = FrozenLinearAllocator(p)
    feats = NodeFeatures(score_max=0.3, n_children=1, difficulty=0.6)
    before = alloc.decide(feats, "dollars").action
    path = tmp_path / "pol.json"
    save_policy(alloc, path, meta={"learner": "dpo"})
    loaded = load_policy(path)
    assert loaded.decide(feats, "dollars").action == before
    assert np.allclose(loaded.policy.probs(feats.vector()), p.probs(feats.vector()))


def test_bandit_roundtrip_restores_posteriors(tmp_path):
    b = BanditAllocator(seed=0)
    b.update(Action.WIDER, 1.0)
    b.update(Action.DEEPER, 0.0)
    path = tmp_path / "b.json"
    save_policy(b, path)
    lb = load_policy(path)
    assert isinstance(lb, BanditAllocator)
    assert lb._alpha[Action.WIDER] == b._alpha[Action.WIDER]
    assert lb._beta[Action.DEEPER] == b._beta[Action.DEEPER]


def test_feature_schema_drift_refuses_to_load(tmp_path):
    path = tmp_path / "p.json"
    save_policy(FrozenLinearAllocator(_fitted_policy()), path)
    doc = json.loads(path.read_text())
    doc["feature_names"] = doc["feature_names"] + ["ghost_feature"]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        load_policy(path)


def test_refuses_to_save_unfitted_policy(tmp_path):
    p = LinearSoftmaxPolicy(n_features=len(NodeFeatures.names()), n_actions=len(ACTIONS))
    with pytest.raises(ValueError):
        save_policy(FrozenLinearAllocator(p), tmp_path / "x.json")
