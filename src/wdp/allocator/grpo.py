"""GRPOAllocator: the on-policy rung above BC/DPO/KTO.

BC, DPO, and KTO all learn from a *fixed* set of already-collected traces. GRPO is
on-policy: every gradient step regenerates fresh rollouts with the *current*
policy, groups G rollouts per prompt, and uses the group-relative reward as the
advantage (no learned critic). That on-policy rollout requirement is the entire
cost delta the GRPO estimator quantifies, and it is why GRPO is run as a small
probe rather than the default.

This allocator is the policy container and decision rule; the on-policy loop
(collect rollouts -> group-relative advantage -> update) lives in
`wdp.loop.grpo_train`. We warm-start from the same BC reference as DPO/KTO and
keep the same LinearSoftmaxPolicy, so a GRPO probe is comparable to the other
learners on equal footing (same model, features, and reference).
"""
from __future__ import annotations

import numpy as np

from wdp.allocator.policy import Allocator, Decision, NodeFeatures
from wdp.allocator.linear import LinearSoftmaxPolicy
from wdp.allocator.bc import ACTIONS, BCAllocator, _choose


class GRPOAllocator(Allocator):
    def __init__(
        self,
        *,
        keep_fraction: float = 0.3,
        explore_eps: float = 0.15,
        l2: float = 1e-3,
        lr: float = 0.2,
        epochs: int = 400,
        seed: int | None = None,
    ) -> None:
        self.keep_fraction = keep_fraction
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

    def warm_start(self, traces) -> None:
        """Fit the BC reference on seed traces and start the GRPO policy there, so
        on-policy updates refine a sane policy and the KL anchor is meaningful."""
        self._reference.fit(traces)
        ref = self._reference.policy
        self._policy.mu, self._policy.sigma = ref.mu.copy(), ref.sigma.copy()
        self._policy.W, self._policy.b = ref.W.copy(), ref.b.copy()
        self._policy._fitted = True

    def decide(self, feats: NodeFeatures, currency: str, *, explore: bool = False) -> Decision:
        if not self._policy._fitted:
            raise RuntimeError("GRPOAllocator.decide called before warm_start()")
        p = self._policy.probs(feats.vector())
        scores = {a: float(p[i]) for i, a in enumerate(ACTIONS)}
        i = _choose(p, self._rng, self.explore_eps) if explore else int(np.argmax(p))
        return Decision(action=ACTIONS[i], scores=scores)
