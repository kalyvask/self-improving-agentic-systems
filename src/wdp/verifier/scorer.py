"""Verifiers: the part the whole project leans on.

The course's recurring lesson is that the generation-verification gap, not raw
generation, is the bottleneck -- "the verifier is the product." The Allocator
only ever sees the world through verifier scores, so we keep two kinds:

  - TerminalVerifier: outcome reward at the end of a trajectory. On a benchmark
    with a checkable answer (tau-bench DB state, SWE-bench tests) this is the
    ground-truth env reward in [0,1]. It is what credit assignment ultimately
    trusts.
  - ProcessVerifier: a cheap per-step / per-attempt score in [0,1] used *during*
    a run to feed NodeFeatures (score_mean/max/std). This is the weak signal the
    Allocator acts on before the terminal reward exists -- a PRM-style estimate,
    here produced by a cheap scorer model rather than a trained PRM.

Keeping the two separate is deliberate: the Allocator is trained against the
terminal reward (so it cannot be fooled by a miscalibrated process score), but it
*acts* on the process score (so it can decide before the answer is known). The
gap between the two is itself logged as the generation-verification gap metric.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class Score:
    """A verifier judgement in [0,1] plus optional rationale for logging."""
    value: float
    rationale: str = ""

    def __post_init__(self) -> None:
        self.value = float(min(1.0, max(0.0, self.value)))


@runtime_checkable
class TerminalVerifier(Protocol):
    """Ground-truth outcome reward. Implemented per-benchmark."""

    def score_final(self, task: object, answer: str) -> Score: ...


@runtime_checkable
class ProcessVerifier(Protocol):
    """Cheap mid-run quality estimate over a partial trajectory or attempt."""

    def score_step(self, task: object, partial_trajectory: str) -> Score: ...


class LLMProcessVerifier:
    """Default ProcessVerifier: ask a cheap scorer model to rate progress 0..1.

    This is the weak verifier the Allocator reads at every decision node. It is
    intentionally cheap (haiku-class) -- the design bets that a noisy-but-cheap
    process score, aggregated across attempts, is enough for the *allocation*
    decision even when it is too noisy to be a final selector.
    """

    def __init__(self, client, model: str, ledger=None, parallel_group=None) -> None:
        self._client = client
        self._model = model
        self._ledger = ledger
        self._parallel_group = parallel_group

    _PROMPT = (
        "You are a strict progress grader. Given a task and an agent's partial "
        "work, output ONLY a number in [0,1] estimating the probability this "
        "trajectory is on track to fully solve the task. 0 = clearly failing/"
        "stalled, 1 = essentially solved. Output the number and nothing else."
    )

    def score_step(self, task: object, partial_trajectory: str) -> Score:
        messages = [
            {"role": "system", "content": self._PROMPT},
            {"role": "user", "content": f"TASK:\n{task}\n\nPARTIAL WORK:\n{partial_trajectory}"},
        ]
        resp = self._client.chat(
            self._model,
            messages,
            ledger=self._ledger,
            parallel_group=self._parallel_group,
            temperature=0.0,
            max_tokens=8,
        )
        return Score(value=_parse_float(resp.text), rationale=resp.text.strip())


def _parse_float(text: str) -> float:
    """Pull the first float out of a model response; default 0.0 on garbage."""
    import re

    m = re.search(r"[-+]?\d*\.?\d+", text or "")
    if not m:
        return 0.0
    try:
        return float(m.group(0))
    except ValueError:
        return 0.0
