"""BCAllocator: behavior-clone the good allocation decisions from logged traces.

This is the first self-improvement rung above the bandit. It does NOT invent new
behavior; it distills the decisions that worked. Two filters make "worked"
concrete:

  1. Trace-level: keep only the top `keep_fraction` of task traces, ranked by
     mean realized value-per-cost. This is the STaR / rejection-sampling move --
     learn from successes, drop the rest.
  2. Decision-level: weight each kept (features -> action) example by its own
     value-per-cost, so a cheap solve teaches more than a lucky expensive one.

Correct STOPs survive both filters (a correct abstention scores high in
assign_credit), so BC learns *when not to spend*, not just how to spend.
"""
from __future__ import annotations

import math

import numpy as np

from wdp.allocator.policy import Action, Allocator, Decision, NodeFeatures
from wdp.allocator.linear import LinearSoftmaxPolicy

# Fixed, ordered action vocabulary shared by every trainable policy.
ACTIONS: list[Action] = [Action.WIDER, Action.DEEPER, Action.DECOMPOSE, Action.STOP]
_INDEX = {a.value: i for i, a in enumerate(ACTIONS)}


class BCAllocator(Allocator):
    def __init__(
        self,
        *,
        keep_fraction: float = 0.3,
        weight_floor: float = 0.05,
        l2: float = 1e-3,
        lr: float = 0.2,
        epochs: int = 400,
        seed: int | None = None,
    ) -> None:
        self.keep_fraction = keep_fraction
        self.weight_floor = weight_floor
        self._policy = LinearSoftmaxPolicy(
            n_features=len(NodeFeatures.names()), n_actions=len(ACTIONS),
            l2=l2, lr=lr, epochs=epochs, seed=seed,
        )

    @property
    def policy(self) -> LinearSoftmaxPolicy:
        return self._policy

    def fit(self, traces) -> None:
        kept = _filter_traces(traces, self.keep_fraction)
        X, y, w = [], [], []
        for tr in kept:
            for d in tr.decisions:
                if d.action not in _INDEX or not d.features:
                    continue
                X.append(d.features)
                y.append(_INDEX[d.action])
                w.append(max(self.weight_floor, float(d.value_per_cost)))
        if not X:
            raise ValueError("BCAllocator.fit: no usable decisions in traces")
        self._policy.fit_bc(np.asarray(X), np.asarray(y), np.asarray(w))

    def decide(self, feats: NodeFeatures, currency: str) -> Decision:
        if not self._policy._fitted:
            raise RuntimeError("BCAllocator.decide called before fit()")
        p = self._policy.probs(feats.vector())
        scores = {a: float(p[i]) for i, a in enumerate(ACTIONS)}
        best_i = int(np.argmax(p))
        return Decision(action=ACTIONS[best_i], scores=scores)


def _filter_traces(traces, keep_fraction: float):
    """Keep the top `keep_fraction` traces by mean realized value-per-cost."""
    scored = []
    for tr in traces:
        vals = [d.value_per_cost for d in tr.decisions] or [0.0]
        scored.append((float(np.mean(vals)), tr))
    scored.sort(key=lambda x: x[0], reverse=True)
    k = max(1, math.ceil(keep_fraction * len(scored)))
    return [tr for _, tr in scored[:k]]
