"""Save / load a trained allocator so it can be deployed without retraining.

A learned policy lives only in memory and is refit from traces each session. For
deployment (train once, serve frozen) we persist the fitted weights (or the bandit
posteriors) plus the EXACT action vocabulary and feature names they were trained
against. On load we refuse to deserialize if either has drifted from the current
code -- a policy mis-indexed against a changed feature vector or action list would
be a silent, damaging bug (we have fixed several of that family), so we fail loud.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wdp.allocator.bc import ACTIONS
from wdp.allocator.linear import LinearSoftmaxPolicy
from wdp.allocator.policy import (
    Action, Allocator, BanditAllocator, BANDIT_ARMS, Decision, NodeFeatures,
)

SCHEMA = 1


class FrozenLinearAllocator(Allocator):
    """A deployable, no-training allocator wrapping a fitted LinearSoftmaxPolicy.
    `decide` is greedy argmax over the fixed ACTIONS (the deployed eval behaviour)."""

    def __init__(self, policy: LinearSoftmaxPolicy) -> None:
        self._policy = policy

    @property
    def policy(self) -> LinearSoftmaxPolicy:
        return self._policy

    def decide(self, feats: NodeFeatures, currency: str, *, explore: bool = False) -> Decision:
        p = self._policy.probs(feats.vector())
        scores = {a: float(p[i]) for i, a in enumerate(ACTIONS)}
        return Decision(action=ACTIONS[int(np.argmax(p))], scores=scores)

    def fit(self, traces) -> None:                     # pragma: no cover
        raise RuntimeError("FrozenLinearAllocator is deploy-only; train then save_policy().")


def save_policy(alloc: Allocator, path: str | Path, *, meta: dict | None = None) -> None:
    """Persist a BanditAllocator or any learner exposing a fitted `.policy`."""
    if isinstance(alloc, BanditAllocator):
        kind, body, vocab = "bandit", alloc.to_dict(), [a.value for a in BANDIT_ARMS]
    else:
        pol = getattr(alloc, "policy", None)
        if not isinstance(pol, LinearSoftmaxPolicy):
            raise TypeError(f"don't know how to save {type(alloc).__name__}")
        if not pol._fitted:
            raise ValueError("refusing to save an unfitted policy")
        kind, body, vocab = "linear", pol.to_dict(), [a.value for a in ACTIONS]
    doc = {"schema": SCHEMA, "kind": kind, "action_vocab": vocab,
           "feature_names": NodeFeatures.names(), "policy": body, "meta": meta or {}}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def load_policy(path: str | Path) -> Allocator:
    """Load a saved policy, validating the action/feature schema against this code."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"policy schema {doc.get('schema')} != {SCHEMA}; incompatible")
    kind = doc.get("kind")
    if kind == "bandit":
        want = [a.value for a in BANDIT_ARMS]
        if doc.get("action_vocab") != want:
            raise ValueError(f"bandit arm drift: saved {doc.get('action_vocab')} != {want}")
        return BanditAllocator.from_dict(doc["policy"])
    if kind == "linear":
        if doc.get("feature_names") != NodeFeatures.names():
            raise ValueError("feature-schema drift since training; refusing to load "
                             "(a mis-indexed feature vector would silently corrupt decisions)")
        if doc.get("action_vocab") != [a.value for a in ACTIONS]:
            raise ValueError("action-vocab drift since training; refusing to load")
        return FrozenLinearAllocator(LinearSoftmaxPolicy.from_dict(doc["policy"]))
    raise ValueError(f"unknown policy kind {kind!r}")
