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

# Fixed, ordered action vocabulary shared by every trainable policy. ESCALATE is
# appended last so existing 4-action policy weights/indices are unchanged; a policy
# trained without escalation data simply never learns to prefer index 4.
ACTIONS: list[Action] = [Action.WIDER, Action.DEEPER, Action.DECOMPOSE,
                         Action.STOP, Action.ESCALATE]
_INDEX = {a.value: i for i, a in enumerate(ACTIONS)}


class BCAllocator(Allocator):
    def __init__(
        self,
        *,
        keep_fraction: float = 0.3,
        weight_floor: float = 0.05,
        explore_eps: float = 0.15,
        l2: float = 1e-3,
        lr: float = 0.2,
        epochs: int = 400,
        seed: int | None = None,
    ) -> None:
        self.keep_fraction = keep_fraction
        self.weight_floor = weight_floor
        self.explore_eps = explore_eps
        self._rng = np.random.default_rng(seed)
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

    def decide(self, feats: NodeFeatures, currency: str, *, explore: bool = False) -> Decision:
        if not self._policy._fitted:
            raise RuntimeError("BCAllocator.decide called before fit()")
        p = self._policy.probs(feats.vector())
        scores = {a: float(p[i]) for i, a in enumerate(ACTIONS)}
        i = _choose(p, self._rng, self.explore_eps) if explore else int(np.argmax(p))
        return Decision(action=ACTIONS[i], scores=scores)


def _choose(p: np.ndarray, rng: np.random.Generator, eps: float) -> int:
    """Sample an action index during data collection.

    Greedy (argmax) collection is what collapses self-improvement: the policy
    only ever logs its current best action, so the next fit has no counter-
    examples for the other actions and narrows to a constant. We instead sample
    from the softmax mixed with a uniform floor (epsilon), guaranteeing every
    action keeps appearing in the traces. Eval still uses argmax.
    """
    uniform = np.ones(len(p)) / len(p)
    mix = (1.0 - eps) * p + eps * uniform
    mix = mix / mix.sum()
    return int(rng.choice(len(p), p=mix))


def _filter_traces(traces, keep_fraction: float):
    """Keep the SUCCESSFUL traces (a solve, or a correct abstention) -- STaR-style
    rejection sampling -- rather than only the globally cheapest by mean value-per-cost.

    The old top-`keep_fraction`-by-mean-vpc filter kept only the cheap atomic solves
    (highest value-per-cost) and discarded the expensive-but-correct DECOMPOSE solves
    and the correct STOPs. That starved the BC reference -- and the DPO/KTO policies
    warm-started from it -- of the very actions that help multi-part and unsolvable
    tasks, so the controller could never learn to use them. Keeping every success
    preserves those examples; per-decision value-per-cost weighting (with solve_floor)
    still emphasizes cheaper solves without erasing the necessary-expensive ones.
    Falls back to the old top-fraction rule only when nothing has succeeded yet."""
    successful = [
        tr for tr in traces
        if tr.solved or (tr.abstention_reward >= 0.5
                         and any(d.action == "stop" for d in tr.decisions))
    ]
    if successful:
        return successful
    scored = sorted(
        traces,
        key=lambda tr: float(np.mean([d.value_per_cost for d in tr.decisions] or [0.0])),
        reverse=True,
    )
    k = max(1, math.ceil(keep_fraction * len(scored)))
    return scored[:k]
