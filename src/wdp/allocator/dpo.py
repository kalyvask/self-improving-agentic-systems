"""DPOAllocator: prefer the allocation actions that paid off, over those that did not.

BC clones good decisions but treats every kept example as equally "correct." DPO
goes further: it learns from *contrasts*. The standard DPO recipe is (1) an SFT
reference, here the BC policy, then (2) preference pairs that say "in this kind of
state, action A beat action B." We mine those pairs from realized value-per-cost.

The honest caveat, documented because it matters: true DPO pairs share one state
with two ranked responses, but our logs have no counterfactuals (we never see two
actions at the identical node). So we approximate -- we bucket decisions into
coarse state-neighborhoods (similar budget / best-score / decomposability), and
within a bucket pair the higher value-per-cost decision's action over a lower
one's, anchored at the winner's state. It is an approximation, but a faithful one
for a contextual bandit, and it keeps the BC-vs-DPO comparison on equal footing.
"""
from __future__ import annotations

import numpy as np

from wdp.allocator.policy import Action, Allocator, Decision, NodeFeatures
from wdp.allocator.linear import LinearSoftmaxPolicy
from wdp.allocator.bc import ACTIONS, _INDEX, BCAllocator

# Feature indices used to bucket decisions into state-neighborhoods.
_F = {n: i for i, n in enumerate(NodeFeatures.names())}


class DPOAllocator(Allocator):
    def __init__(
        self,
        *,
        keep_fraction: float = 0.3,
        beta: float = 0.1,
        min_gap: float = 0.05,
        max_pairs: int = 5000,
        l2: float = 1e-3,
        lr: float = 0.2,
        epochs: int = 400,
        seed: int | None = None,
    ) -> None:
        self.keep_fraction = keep_fraction
        self.beta = beta
        self.min_gap = min_gap
        self.max_pairs = max_pairs
        self._reference = BCAllocator(keep_fraction=keep_fraction, l2=l2, lr=lr,
                                      epochs=epochs, seed=seed)
        self._policy = LinearSoftmaxPolicy(
            n_features=len(NodeFeatures.names()), n_actions=len(ACTIONS),
            l2=l2, lr=lr, epochs=epochs, seed=seed,
        )

    @property
    def policy(self) -> LinearSoftmaxPolicy:
        return self._policy

    @property
    def reference(self) -> BCAllocator:
        return self._reference

    def fit(self, traces) -> None:
        # 1) SFT reference = BC on the same traces.
        self._reference.fit(traces)
        # 2) Mine preference pairs from realized value-per-cost.
        X, a_pref, a_rej = _mine_pairs(traces, self.min_gap, self.max_pairs)
        if len(X) == 0:
            # No contrastable pairs -> DPO degenerates to the reference.
            ref = self._reference.policy
            self._policy.mu, self._policy.sigma = ref.mu.copy(), ref.sigma.copy()
            self._policy.W, self._policy.b = ref.W.copy(), ref.b.copy()
            self._policy._fitted = True
            return
        # 3) DPO update against the frozen reference.
        self._policy.fit_dpo(np.asarray(X), np.asarray(a_pref), np.asarray(a_rej),
                             reference=self._reference.policy, beta=self.beta)

    def decide(self, feats: NodeFeatures, currency: str) -> Decision:
        if not self._policy._fitted:
            raise RuntimeError("DPOAllocator.decide called before fit()")
        p = self._policy.probs(feats.vector())
        scores = {a: float(p[i]) for i, a in enumerate(ACTIONS)}
        best_i = int(np.argmax(p))
        return Decision(action=ACTIONS[best_i], scores=scores)


def _bucket_key(feat: list[float]) -> tuple:
    """Coarse state-neighborhood: round the features the allocation hinges on."""
    return (
        round(feat[_F["budget_remaining_frac"]], 1),
        round(feat[_F["score_max"]], 1),
        round(feat[_F["decomposability"]], 1),
        round(feat[_F["executor_stalled"]], 0),
    )


def _mine_pairs(traces, min_gap: float, max_pairs: int):
    """Within each state-bucket, prefer the higher value-per-cost action over a
    lower one (different action, gap >= min_gap). Anchored at the winner state."""
    buckets: dict[tuple, list] = {}
    for tr in traces:
        for d in tr.decisions:
            if d.action not in _INDEX or not d.features:
                continue
            buckets.setdefault(_bucket_key(d.features), []).append(d)

    X, a_pref, a_rej = [], [], []
    for decisions in buckets.values():
        if len(decisions) < 2:
            continue
        ranked = sorted(decisions, key=lambda d: d.value_per_cost, reverse=True)
        for i, win in enumerate(ranked):
            for lose in ranked[i + 1:]:
                if win.action == lose.action:
                    continue
                if win.value_per_cost - lose.value_per_cost < min_gap:
                    continue
                X.append(win.features)
                a_pref.append(_INDEX[win.action])
                a_rej.append(_INDEX[lose.action])
                if len(X) >= max_pairs:
                    return X, a_pref, a_rej
    return X, a_pref, a_rej
