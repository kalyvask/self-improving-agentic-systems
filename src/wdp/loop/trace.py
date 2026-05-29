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


def assign_credit(trace: TaskTrace, *, gamma: float = 1.0) -> None:
    """Attach terminal reward + value-per-cost to each decision (in place).

    We use the simplest defensible scheme: every spend decision on a solved task
    shares the terminal reward, discounted toward the end of the trajectory, and
    normalized by that decision's marginal cost so the credited quantity is
    value-PER-COST (the thing the Allocator actually optimizes). STOP decisions
    are credited by the *avoided* cost when the task was genuinely unsolvable --
    i.e. a correct abstention scores well. This is intentionally close to STaR /
    SWiRL outcome credit so the BC/DPO/GRPO comparison stays clean.
    """
    n = len(trace.decisions)
    for i, rec in enumerate(trace.decisions):
        rec.terminal_reward = trace.terminal_reward
        discount = gamma ** (n - 1 - i)
        if rec.action == Action.STOP.value:
            # Reward a correct stop (unsolved & cheap-to-have-stopped) near 1; a
            # premature stop on a solvable task is penalized via terminal_reward~0.
            rec.value_per_cost = (1.0 - trace.terminal_reward) if not trace.solved else 0.0
        else:
            mc = rec.marginal_cost or 1e-9
            rec.value_per_cost = float(min(1.0, (discount * trace.terminal_reward) / mc)) \
                if trace.solved else 0.0


def feature_names() -> list[str]:
    return NodeFeatures.names()
