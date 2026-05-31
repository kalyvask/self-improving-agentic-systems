"""Trace records: the training data the self-improving Allocator learns from.

This is the spine of the "self-improving" claim. Every allocation decision is
logged as a (features, action, predicted-scores, cost-incurred) tuple; when the
task terminates we attach the terminal reward and a credit-assigned value to each
decision. That gives us exactly the supervision the three learners need:

  - BC   wants (features -> action) pairs from *good* tasks (filtered traces).
  - DPO  wants (features, preferred-action, rejected-action) pairs, which we mine
         from sibling decisions with differing realized value-per-cost.
  - GRPO would want the same tuples but generated on-policy each step; we log the
         per-call token/wall cost so the GRPO cost estimate is measured, not guessed.

Everything is plain dataclasses + JSONL so a trace file is portable and the
trainers in wdp.loop can read it without importing the live agent stack.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from wdp.allocator.policy import Action, NodeFeatures


@dataclass
class DecisionRecord:
    """One Allocator decision and what it cost / earned."""
    step: int
    features: list[float]                 # NodeFeatures.vector()
    feature_names: list[float] = field(default_factory=list)
    action: str = ""                      # Action.value chosen
    scores: dict = field(default_factory=dict)   # per-action predicted value
    currency: str = "dollars"
    cost_before: float = 0.0              # ledger.amount(currency) before the action
    cost_after: float = 0.0               # ... after
    process_score_after: float = 0.0      # best process score visible after acting
    # For ESCALATE only: how the strong model was invoked -- "live_handoff" (it
    # resumed the cheap model's unfinished env/conversation) or "fresh_retry" (it
    # started the task over). None for every non-ESCALATE action. Recorded because the
    # "true handoff" behaviour is part of the cascade claim and should be auditable.
    escalate_mode: str | None = None
    # Filled in by credit assignment once the task terminates:
    terminal_reward: float = 0.0
    value_per_cost: float = 0.0           # credited value / marginal cost (normalized)

    @property
    def marginal_cost(self) -> float:
        return max(self.cost_after - self.cost_before, 0.0)


@dataclass
class TaskTrace:
    """All decisions for one task attempt, plus the outcome."""
    task_id: str
    currency: str
    policy: str                           # "bandit" | "bc" | "dpo" | ...
    decisions: list[DecisionRecord] = field(default_factory=list)
    solved: bool = False
    terminal_reward: float = 0.0
    # Ground-truth quality of abstaining on this task: 1.0 if the task was
    # genuinely unsolvable (STOP was the right call), 0.0 otherwise. Kept
    # separate from terminal_reward so a correct abstention never retroactively
    # credits the spend actions that ran before the STOP.
    abstention_reward: float = 0.0
    total_cost: dict = field(default_factory=dict)   # full per-currency snapshot
    wall_started: float = field(default_factory=time.time)

    def add(self, rec: DecisionRecord) -> None:
        self.decisions.append(rec)

    def to_json(self) -> dict:
        return asdict(self)


class TraceLog:
    """Append-only JSONL writer/reader for TaskTraces."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, trace: TaskTrace) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace.to_json()) + "\n")

    def read(self) -> list[TaskTrace]:
        traces: list[TaskTrace] = []
        if not self.path.exists():
            return traces
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                decisions = [DecisionRecord(**dr) for dr in d.pop("decisions", [])]
                traces.append(TaskTrace(decisions=decisions, **d))
        return traces


def assign_credit(trace: TaskTrace, *, gamma: float = 1.0,
                  budget: float | None = None,
                  advantage_floor: float = 0.5,
                  cost_weight: float = 0.5,
                  abstention_credit: float = 0.5,
                  solve_floor: float = 0.6) -> None:
    """Attach terminal reward + value-per-cost to each decision (in place).

    Spend decisions are credited by outcome *times* cost-efficiency *times* the
    decision's own contribution, all in [0,1], so the Allocator is pushed toward
    cheap solves rather than just any solve, and toward the *decisions that did
    the work* rather than every decision on a winning task equally.

      - outcome: `trace.terminal_reward`, the graded quality of the best answer
        (not a binary solved flag, so a partial reward still trains).
      - cost-efficiency: `exp(-cost_weight * spent/budget)`, a smooth decay in
        (0, 1]. A solve that used little budget keeps almost all its reward; a
        pricey one keeps less, but it is *never* zero, so a solved task always
        trains as a win. Two earlier forms each failed: `1 - spent/budget` drove
        efficiency to 0 at budget and erased expensive-but-winning WIDER traces;
        its replacement `1 - cost_weight*min(1, spent/budget)` fixed that but the
        `min(1, .)` cap flattened *every* over-budget trace to the same floor, so
        a solve at 1.1x budget and one at 3x budget got identical credit -- no
        gradient against runaway spend, which let the policy balloon its cost on
        expensive benchmarks. The exponential removes the cap: it keeps decaying
        past budget, so gross overspend is penalized strictly more than marginal
        overspend while staying positive. Without a budget this term is 1.0.
      - contribution (advantage): how much this decision *raised the best process
        score seen so far*. A WIDER that spun a dead-end attempt and a DEEPER that
        finished the winner used to get identical credit on a solved task; now the
        decision that moved the verifier signal earns more. We blend a uniform
        floor (`advantage_floor`) with the normalized per-step advantage so set-up
        moves still get partial credit and aren't zeroed. When no decision moved
        the score (no process signal at all), this collapses to uniform weights --
        i.e. the previous behavior -- so the mechanics tests still hold.

    STOP decisions are credited by `abstention_credit * trace.abstention_reward`:
    the ground-truth quality of the abstention (1.0 only when the task was genuinely
    unsolvable, 0.0 for a premature give-up), scaled DOWN by `abstention_credit`
    (default 0.5) so a correct abstention is worth less than a correct solve. This
    matters because a correct solve is cost-discounted to ~0.6-0.8 while an
    un-scaled correct STOP would be the maximum 1.0 -- making abstaining look
    *better* than solving. That asymmetry let STOP become the single
    highest-weighted example in the BC reference's value-weighted clone, which
    every learner (and the warm-started KTO/GRPO policies) then inherited, drifting
    the whole controller toward STOP across self-improvement rounds. Scaling
    abstention below the solve scale keeps the abstention arm useful (a correct STOP
    still beats a premature one and beats failing) without letting it dominate.
    """
    # Guard the credit ordering: a correct abstention must be worth less than the
    # cheapest possible solve, or we re-create the STOP-over-solve asymmetry that
    # drifted the controller to STOP. Required: 0 <= abstention_credit < solve_floor <= 1.
    if not (0.0 <= abstention_credit < solve_floor <= 1.0):
        raise ValueError(
            f"need 0 <= abstention_credit ({abstention_credit}) < solve_floor "
            f"({solve_floor}) <= 1 so solves out-value correct abstentions")
    spent = (trace.total_cost or {}).get(trace.currency)
    if spent is None:
        spent = sum(rec.marginal_cost for rec in trace.decisions)
    efficiency = 1.0
    if budget and budget > 0:
        # Solve floor: a real solve keeps at least `solve_floor` of its outcome
        # credit no matter how expensive, so the cost term ranks cheap > expensive
        # solves WITHOUT pushing a necessary-but-expensive solve (a DECOMPOSE that
        # ran over budget, or a future ESCALATE to a pricier model) below a correct
        # abstention (abstention_credit) or KTO's desirability threshold. Without
        # this floor, exp(-cost_weight*spent/budget) drove expensive solves to
        # ~0.2-0.3 -- under the 0.5 abstention credit -- so the objective taught
        # "cheap mediocre beats necessary expensive" and suppressed DECOMPOSE.
        # Keep solve_floor > abstention_credit so every solve out-values abstaining.
        efficiency = solve_floor + (1.0 - solve_floor) * math.exp(-cost_weight * (spent / budget))

    # Per-step advantage = how much each decision raised the running-best process
    # score. Only positive moves count (a decision can't be blamed for noise dips).
    advantages: list[float] = []
    best_so_far = 0.0
    for rec in trace.decisions:
        adv = max(0.0, rec.process_score_after - best_so_far)
        advantages.append(adv)
        best_so_far = max(best_so_far, rec.process_score_after)
    total_adv = sum(advantages)

    n = len(trace.decisions)
    for i, rec in enumerate(trace.decisions):
        rec.terminal_reward = trace.terminal_reward
        if rec.action == Action.STOP.value:
            # Credit STOP by whether abstaining was actually correct (ground-truth
            # abstention reward, not 1 - terminal_reward, so a failed task isn't a
            # good place to give up), scaled below the solve scale so a correct
            # abstention can't out-value an actual solve and dominate the clone.
            # (We tried also discounting by cost-efficiency to reward EARLY
            # abstention, but it pushed every correct STOP below KTO's desirability
            # threshold, neutralizing the abstention arm for zero gain on the actual
            # collapse -- so STOP credit is left on the flat abstention scale.)
            rec.value_per_cost = abstention_credit * trace.abstention_reward
            continue
        discount = gamma ** (n - 1 - i)
        if total_adv > 1e-9:
            # contrib_i sums to 1 over the trace; * n makes the mean weight 1.0 so
            # magnitudes stay comparable to the uniform scheme (a no-op on average).
            contrib = advantage_floor / n + (1.0 - advantage_floor) * (advantages[i] / total_adv)
            weight = contrib * n
        else:
            weight = 1.0
        rec.value_per_cost = float(min(1.0, discount * trace.terminal_reward * efficiency * weight))


def feature_names() -> list[str]:
    return NodeFeatures.names()
