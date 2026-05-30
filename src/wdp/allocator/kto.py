"""KTOAllocator: learn from unpaired good/bad decisions, no preference pairs.

DPO needs contrastive *pairs* sharing a state, which our logs never truly give
us -- the DPOAllocator approximates them by bucketing decisions and pairing the
higher value-per-cost action over a lower one. KTO (Kahneman-Tversky
Optimization, arXiv:2402.01306) removes that approximation: it learns from
*unpaired* examples each tagged desirable or undesirable. Our traces are
naturally that shape -- a decision's realized value-per-cost thresholds cleanly
into good/bad -- so KTO uses every decision instead of only the ones that happen
to share a bucket with a differing action. In the tiny-trace regime that is
strictly more data-efficient, which is exactly where we operate.

Kept on the same LinearSoftmaxPolicy and the same BC reference as DPO so the
BC-vs-DPO-vs-KTO comparison isolates the objective, not the model or the data.
"""
from __future__ import annotations

import numpy as np

from wdp.allocator.policy import Allocator, Decision, NodeFeatures
from wdp.allocator.linear import LinearSoftmaxPolicy
from wdp.allocator.bc import ACTIONS, _INDEX, BCAllocator, _choose


class KTOAllocator(Allocator):
    def __init__(
        self,
        *,
        keep_fraction: float = 0.3,
        beta: float = 0.1,
        desirable_threshold: float = 0.5,
        lambda_d: float = 1.0,
        lambda_u: float = 1.0,
        explore_eps: float = 0.15,
        l2: float = 1e-3,
        lr: float = 0.2,
        epochs: int = 400,
        seed: int | None = None,
    ) -> None:
        self.keep_fraction = keep_fraction
        self.beta = beta
        self.desirable_threshold = desirable_threshold
        self.lambda_d = lambda_d
        self.lambda_u = lambda_u
        self.explore_eps = explore_eps
        self._rng = np.random.default_rng(seed)
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
        # 1) SFT reference = BC on the same traces (shared with DPO).
        self._reference.fit(traces)
        # 2) Tag every decision desirable/undesirable by realized value-per-cost.
        X, actions, desirable = [], [], []
        for tr in traces:
            for d in tr.decisions:
                if d.action not in _INDEX or not d.features:
                    continue
                X.append(d.features)
                actions.append(_INDEX[d.action])
                desirable.append(d.value_per_cost >= self.desirable_threshold)
        # Degenerate to the reference if there's nothing to contrast (no examples,
        # or every example one class -> KTO has no signal either way).
        if not X or all(desirable) or not any(desirable):
            ref = self._reference.policy
            self._policy.mu, self._policy.sigma = ref.mu.copy(), ref.sigma.copy()
            self._policy.W, self._policy.b = ref.W.copy(), ref.b.copy()
            self._policy._fitted = True
            return
        self._policy.fit_kto(
            np.asarray(X), np.asarray(actions), np.asarray(desirable),
            reference=self._reference.policy, beta=self.beta,
            lambda_d=self.lambda_d, lambda_u=self.lambda_u,
        )

    def decide(self, feats: NodeFeatures, currency: str, *, explore: bool = False) -> Decision:
        if not self._policy._fitted:
            raise RuntimeError("KTOAllocator.decide called before fit()")
        p = self._policy.probs(feats.vector())
        scores = {a: float(p[i]) for i, a in enumerate(ACTIONS)}
        i = _choose(p, self._rng, self.explore_eps) if explore else int(np.argmax(p))
        return Decision(action=ACTIONS[i], scores=scores)
