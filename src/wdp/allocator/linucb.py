"""LinUCBAllocator: a contextual-bandit baseline over the SAME features and
actions the trained policies use -- the online, contextual control the offline
BC/DPO/GRPO stack must beat to justify its cost.

Where it sits in the allocator family:

  - v0 ``BanditAllocator``: online but NON-contextual (a Beta posterior per
    action, ignoring ``NodeFeatures``). It can learn "DEEPER is best on average,"
    never "DEEPER is best *when the score is climbing and budget is low*."
  - BC / DPO / KTO / GRPO: contextual (they read the 12 features) but OFFLINE
    (trained once on logged traces).
  - LinUCB (this file): the empty cell -- contextual AND online. It reads the
    exact same ``NodeFeatures`` as BC/DPO and updates per decision from the live
    reward, with no offline training step.

So the comparison is clean: if BC->DPO->GRPO does not beat this cheap online
contextual bandit on the same features and actions, the offline machinery has not
earned its cost -- and that is the honest headline. If it does, the gain is
attributable to offline preference learning, not to context per se.

Algorithm -- disjoint LinUCB (Li et al. 2010, "A Contextual-Bandit Approach to
Personalized News Article Recommendation"): per arm ``a`` keep
``A_a = I + sum x xT`` and ``b_a = sum r x``; for context ``x`` the value estimate
is ``theta_a . x`` with ``theta_a = A_a^-1 b_a`` and the optimistic score adds the
UCB bonus ``alpha * sqrt(x . A_a^-1 . x)``. Pull the argmax (UCB while collecting
data, the plain mean at eval -- mirroring how BC/DPO argmax a frozen policy at
eval). The reward ``r`` is the SAME [0,1] value-per-cost the v0 bandit consumes,
so the two share a currency and the runner's update path
(``runner._maybe_update`` calls ``update(action, vpc)`` with no context, so the
context from the most recent ``decide`` is cached and reused). STOP and the
DECOMPOSE gate mirror ``BanditAllocator`` exactly, so the only thing that differs
from v0 is "contextual vs average-case."
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wdp.allocator.policy import (
    Action,
    Allocator,
    BANDIT_ARMS,
    Decision,
    NodeFeatures,
)


def _context_from_vector(vec) -> np.ndarray:
    """Unit-normalize the raw feature vector and append a bias term.

    ``NodeFeatures`` mixes scales (counts like ``steps_taken`` vs fractions like
    ``budget_remaining_frac``). LinUCB has no fitted scaler the online setting can
    persist, so we L2-normalize the context -- which bounds the UCB bonus and keeps
    the per-arm ridge well-conditioned -- and add a bias feature so an arm can
    learn a context-independent baseline value."""
    x = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(x)) or 1.0
    return np.append(x / norm, 1.0)


class LinUCBAllocator(Allocator):
    """Disjoint LinUCB over ``BANDIT_ARMS`` (cheap spends + ESCALATE), with STOP
    chosen via the same value threshold the v0 bandit uses."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        stop_threshold: float = 0.02,
        seed: int | None = None,
    ) -> None:
        self.alpha = alpha
        self.stop_threshold = stop_threshold
        self._rng = np.random.default_rng(seed)
        self._dim = len(NodeFeatures.names()) + 1  # +1 for the bias feature
        self._A = {a: np.identity(self._dim) for a in BANDIT_ARMS}
        self._b = {a: np.zeros(self._dim) for a in BANDIT_ARMS}
        # The runner's _maybe_update(allocator, action, vpc) passes no context, so
        # we cache the context of the most recent decide() and use it on update().
        self._last_x: np.ndarray | None = None

    # -- selection --------------------------------------------------------- #
    def decide(self, feats: NodeFeatures, currency: str, *, explore: bool = False) -> Decision:
        x = _context_from_vector(feats.vector())
        self._last_x = x
        scores: dict[Action, float] = {}
        for a in BANDIT_ARMS:
            A_inv = np.linalg.inv(self._A[a])
            theta = A_inv @ self._b[a]
            mean = float(theta @ x)
            # UCB bonus drives exploration while collecting data; at eval (explore
            # =False) we report the plain mean, the deployed-greedy value, so the
            # comparison against a frozen BC/DPO argmax is apples-to-apples.
            bonus = self.alpha * float(np.sqrt(max(0.0, float(x @ A_inv @ x)))) if explore else 0.0
            scores[a] = mean + bonus
        # Gate DECOMPOSE on the cheap decomposability probe (mirror BanditAllocator)
        # so we don't plan an atomic task; the runner masks it too if no planner.
        scores[Action.DECOMPOSE] *= max(feats.decomposability, 1e-3)

        best_score = max(scores[a] for a in BANDIT_ARMS)
        # Random tie-break: with A=I and b=0 every arm's score is identical at cold
        # start, so a deterministic argmax would pull only the first arm forever and
        # never explore the rest. Break ties uniformly (the Thompson v0 gets this for
        # free from its random draws).
        candidates = [a for a in BANDIT_ARMS if scores[a] == best_score]
        best = candidates[0] if len(candidates) == 1 else candidates[int(self._rng.integers(len(candidates)))]
        chosen = Action.STOP if scores[best] < self.stop_threshold else best
        scores[Action.STOP] = self.stop_threshold
        return Decision(action=chosen, scores=scores)

    # -- online update ----------------------------------------------------- #
    def update(self, action: Action, value_per_cost_norm: float, context=None) -> None:
        """Rank-1 LinUCB update on the cached (or supplied) context.

        Signature matches the v0 bandit's ``update(action, vpc)`` so the runner
        drives it unchanged; ``context`` is optional for callers (warm-start) that
        pass ``x`` directly. STOP and unavailable actions have no contextual arm,
        so they are no-ops (exactly like the v0 bandit)."""
        if action not in self._A:
            return
        x = context if context is not None else self._last_x
        if x is None:
            return
        x = np.asarray(x, dtype=np.float64)
        r = float(np.clip(value_per_cost_norm, 0.0, 1.0))
        self._A[action] = self._A[action] + np.outer(x, x)
        self._b[action] = self._b[action] + r * x

    # -- optional offline warm-start --------------------------------------- #
    def fit(self, traces) -> None:
        """Replay logged (features, action, value_per_cost) decisions through the
        online update so the bandit starts from the same evidence the trace-trained
        policies see. Purely additive -- it stays fully online-updatable afterward,
        so a warm-started LinUCB models 'deploy with a prior, keep adapting.'"""
        for tr in traces:
            for d in tr.decisions:
                try:
                    action = Action(d.action)
                except ValueError:
                    continue
                if action not in self._A or not d.features:
                    continue
                self.update(action, float(d.value_per_cost),
                            context=_context_from_vector(d.features))

    # -- persistence (mirrors the design's save/load story) ---------------- #
    def snapshot(self) -> dict:
        return {
            "alpha": self.alpha,
            "stop_threshold": self.stop_threshold,
            "A": {a.value: self._A[a].tolist() for a in BANDIT_ARMS},
            "b": {a.value: self._b[a].tolist() for a in BANDIT_ARMS},
        }

    def restore(self, state: dict) -> None:
        self.alpha = state.get("alpha", self.alpha)
        self.stop_threshold = state.get("stop_threshold", self.stop_threshold)
        for a in BANDIT_ARMS:
            if a.value in state.get("A", {}):
                self._A[a] = np.asarray(state["A"][a.value], dtype=np.float64)
                self._b[a] = np.asarray(state["b"][a.value], dtype=np.float64)

    def save(self, path: str | Path) -> None:
        """Persist the learned per-arm A/b to disk so the policy keeps improving
        session to session: deploy, update online from live rewards, save, reload
        next run with the evidence intact."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.snapshot(), indent=2))

    def load(self, path: str | Path) -> None:
        self.restore(json.loads(Path(path).read_text()))
