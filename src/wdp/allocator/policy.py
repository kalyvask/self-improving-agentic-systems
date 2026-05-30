"""The Allocator: the controller that decides how to spend the next unit of compute.

At each decision node it chooses one of four actions:
  - WIDER:     spawn a fresh parallel Executor attempt from the current state
  - DEEPER:    continue / refine the current trajectory on tool feedback
  - DECOMPOSE: hand the task to the Planner -> sub-task DAG -> sub-Executors
  - STOP:      stop spending and escalate (abstain) -- a *safe non-attempt*

Design choice (set by the API-credits-only constraint): the policy is a small,
CPU-trainable model over cheap numeric features -- NOT a fine-tuned LLM. The
expensive part of the project is collecting traces (frontier-model Executors);
the policy update itself is cheap. This makes BC and DPO laptop-runnable and
makes the GRPO cost estimate clean (GRPO trains the same small policy, but the
on-policy rollout requirement is the cost delta).

This module ships the v0 BanditAllocator (works with no training data). BC and
DPO subclass `Allocator` and learn from logged traces -- see wdp.loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Action(str, Enum):
    WIDER = "wider"
    DEEPER = "deeper"
    DECOMPOSE = "decompose"
    STOP = "stop"


SPEND_ACTIONS = (Action.WIDER, Action.DEEPER, Action.DECOMPOSE)


@dataclass
class NodeFeatures:
    """Cheap numeric features describing a decision node. These are the inputs to
    the trainable policy; keep them model-agnostic and cheap to compute."""
    # Verifier-score statistics over the children/attempts seen so far at this node.
    score_mean: float = 0.0
    score_max: float = 0.0
    score_std: float = 0.0
    n_children: int = 0
    # Fraction of the chosen-currency budget remaining (0..1).
    budget_remaining_frac: float = 1.0
    # Executor depth / steps already taken on this trajectory.
    depth: int = 0
    steps_taken: int = 0
    # Cheap probe from the Planner: how decomposable does this task look (0..1)?
    decomposability: float = 0.0
    # Has the current Executor self-reported a stall/failure?
    executor_stalled: float = 0.0
    # Structural distress/progress signals, free to compute from the trajectories
    # (no extra model call). They separate the WIDER vs DEEPER vs STOP decision in
    # ways the score stats wash out:
    #   - tool_error_rate: fraction of steps so far whose tool/env call errored. A
    #     flailing attempt (high errors) is a poor candidate for more DEEPER.
    #   - attempts_done_frac: fraction of attempts that actually reached a final
    #     answer. Low with budget left => attempts are truncating => DEEPER may
    #     finish them; high but unsolved => a fresh WIDER is likelier to help.
    tool_error_rate: float = 0.0
    attempts_done_frac: float = 0.0
    # Task difficulty proxy in [0,1], pinned to the FIRST attempt's process score
    # (difficulty = 1 - first_score), so it stays roughly fixed across the run
    # while score_max climbs. This is the lever that lets the policy condition the
    # WIDER-vs-DEEPER choice on difficulty rather than picking a fixed mix:
    # Snell et al. (arXiv:2408.03314) show the compute-optimal parallel/sequential
    # split flips with difficulty. 0.5 = unknown (no attempt scored yet).
    difficulty: float = 0.5

    def vector(self) -> np.ndarray:
        return np.array(
            [
                self.score_mean,
                self.score_max,
                self.score_std,
                float(self.n_children),
                self.budget_remaining_frac,
                float(self.depth),
                float(self.steps_taken),
                self.decomposability,
                self.executor_stalled,
                self.tool_error_rate,
                self.attempts_done_frac,
                self.difficulty,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def names() -> list[str]:
        return [
            "score_mean", "score_max", "score_std", "n_children",
            "budget_remaining_frac", "depth", "steps_taken",
            "decomposability", "executor_stalled",
            "tool_error_rate", "attempts_done_frac",
            "difficulty",
        ]


@dataclass
class Decision:
    action: Action
    # Per-action predicted value-per-unit-cost (for logging / risk-coverage).
    scores: dict[Action, float] = field(default_factory=dict)


class Allocator:
    """Base policy interface. Subclasses implement `decide`; trainable ones
    (BC, DPO) also implement `fit(traces)`."""

    def decide(self, feats: NodeFeatures, currency: str, *, explore: bool = False) -> Decision:  # pragma: no cover
        raise NotImplementedError

    def fit(self, traces) -> None:  # pragma: no cover - bandit needs no offline fit
        raise NotImplementedError


class BanditAllocator(Allocator):
    """v0 controller: Thompson sampling over per-action value-per-cost.

    Keeps a Beta posterior per action (success-per-cost is normalized to [0,1] as
    the Bernoulli mean). With no data it explores; as outcomes arrive via
    `update`, it concentrates on the action with the best value-per-cost. STOP is
    chosen when every spend-action's sampled value falls below `stop_threshold`.
    This generalizes AB-MCTS's wider/deeper Thompson rule by (a) adding the
    decompose and stop arms and (b) scoring per unit cost rather than per sample.
    """

    def __init__(self, stop_threshold: float = 0.02, seed: int | None = None) -> None:
        self.stop_threshold = stop_threshold
        self._rng = np.random.default_rng(seed)
        # Beta(alpha, beta) per spend-action.
        self._alpha = {a: 1.0 for a in SPEND_ACTIONS}
        self._beta = {a: 1.0 for a in SPEND_ACTIONS}

    def decide(self, feats: NodeFeatures, currency: str, *, explore: bool = False) -> Decision:
        # Thompson sampling already explores, so `explore` is a no-op here; it
        # exists only to share the Allocator.decide signature with BC/DPO.
        samples: dict[Action, float] = {}
        for a in SPEND_ACTIONS:
            samples[a] = float(self._rng.beta(self._alpha[a], self._beta[a]))
        # Gate decompose on the cheap decomposability probe so we don't waste a
        # plan on an atomic task (the policy still learns to refine this).
        samples[Action.DECOMPOSE] *= max(feats.decomposability, 1e-3)

        best = max(SPEND_ACTIONS, key=lambda a: samples[a])
        if samples[best] < self.stop_threshold:
            chosen = Action.STOP
        else:
            chosen = best
        samples[Action.STOP] = self.stop_threshold
        return Decision(action=chosen, scores=samples)

    def update(self, action: Action, value_per_cost_norm: float) -> None:
        """Bandit posterior update. `value_per_cost_norm` in [0,1]: 1.0 = cheap
        success, 0.0 = expensive failure."""
        if action not in self._alpha:
            return
        v = float(np.clip(value_per_cost_norm, 0.0, 1.0))
        self._alpha[action] += v
        self._beta[action] += (1.0 - v)
