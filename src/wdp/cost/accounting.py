"""Cost accounting across the three currencies the Allocator can optimize.

The whole project's thesis is that the optimal allocation policy depends on which
currency you spend in, so every LLM call is logged in all three at once:

  - tokens:   prompt + completion tokens (parallel branches still cost full tokens)
  - latency:  wall-clock SECONDS, with a parallelism model -- concurrent branches
              (wider / decompose) cost the MAX of their children, not the sum
  - dollars:  OpenRouter usage cost when available, else priced from a table

We also log everything GRPO would need (per-call token counts, wall time) so the
end-of-project GRPO cost estimate is an extrapolation, not a guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CURRENCIES = ("tokens", "latency", "dollars")


@dataclass
class Spend:
    """A single billable LLM call."""
    model: str
    prompt_tokens: int
    completion_tokens: int
    wall_seconds: float
    dollars: float
    # True if this call ran concurrently with siblings (wider / decompose branch).
    parallel_group: str | None = None

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class CostLedger:
    """Accumulates Spend records and reports cost per currency.

    Latency is the subtle one: serial calls add, but calls sharing a
    ``parallel_group`` are billed as the max within the group (they ran at once).
    """
    spends: list[Spend] = field(default_factory=list)

    def add(self, spend: Spend) -> None:
        self.spends.append(spend)

    @property
    def tokens(self) -> int:
        return sum(s.tokens for s in self.spends)

    @property
    def dollars(self) -> float:
        return sum(s.dollars for s in self.spends)

    @property
    def latency(self) -> float:
        serial = sum(s.wall_seconds for s in self.spends if s.parallel_group is None)
        groups: dict[str, float] = {}
        for s in self.spends:
            if s.parallel_group is not None:
                groups[s.parallel_group] = max(groups.get(s.parallel_group, 0.0), s.wall_seconds)
        return serial + sum(groups.values())

    def amount(self, currency: str) -> float:
        if currency == "tokens":
            return float(self.tokens)
        if currency == "latency":
            return self.latency
        if currency == "dollars":
            return self.dollars
        raise ValueError(f"unknown currency {currency!r}; expected one of {CURRENCIES}")

    def remaining(self, currency: str, budget: float) -> float:
        return budget - self.amount(currency)

    def snapshot(self) -> dict[str, float]:
        return {c: self.amount(c) for c in CURRENCIES}
