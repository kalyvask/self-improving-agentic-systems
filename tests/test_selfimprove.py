"""Offline tests for the self-improvement driver and the local benchmark.

The driver test uses a scripted fake client (no key, no network) and checks the
*mechanics*: round 0 is the bandit, later rounds fit the named learner, every
round emits a well-formed scoreboard, and the curve formats. It does not assert
the curve goes up -- that needs real model variance and is what the live run is
for. The benchmark test checks the locally-checkable verifier directly.
"""
from __future__ import annotations

import json

from wdp.cost import Spend
from wdp.llm.openrouter import LLMResponse
from wdp.executor.react import Executor, Task
from wdp.planner.decompose import Planner
from wdp.verifier.scorer import LLMProcessVerifier, Score
from wdp.loop import RunConfig, self_improve, format_curve
from wdp.benchmarks import ArithmeticBenchmark, safe_eval


class FakeClient:
    def chat(self, model, messages, *, ledger=None, parallel_group=None,
             temperature=0.7, max_tokens=None, **kwargs) -> LLMResponse:
        system = messages[0]["content"] if messages else ""
        if "tool-using agent" in system:
            text = json.dumps({"thought": "", "action": "FINISH",
                               "action_input": {"answer": "42"}})
        elif "progress grader" in system:
            text = "0.9"
        elif "how much this task benefits" in system:
            text = "0.3"
        elif "Decompose the task" in system:
            text = json.dumps([{"id": "s1", "prompt": "p", "depends_on": []}])
        else:
            text = "ok"
        spend = Spend(model=model, prompt_tokens=50, completion_tokens=10,
                      wall_seconds=0.2, dollars=0.0005, parallel_group=parallel_group)
        if ledger is not None:
            ledger.add(spend)
        return LLMResponse(text=text, model=model, spend=spend, raw={})


class GoldVerifier:
    def score_final(self, task, answer: str) -> Score:
        return Score(value=1.0 if "42" in (answer or "") else 0.0)


def _stack():
    c = FakeClient()
    return (Executor(c, "fake", tools={}, max_steps=4),
            Planner(c, "fake"), LLMProcessVerifier(c, "fake"), GoldVerifier())


def test_self_improve_bc_runs_rounds():
    ex, pl, vf, tm = _stack()
    train = [Task(id=f"tr{i}", prompt="q") for i in range(6)]
    eval_ = [Task(id=f"ev{i}", prompt="q") for i in range(3)]
    reports = self_improve(train, eval_, ex, vf, tm, planner=pl, learner="bc",
                           rounds=2, cfg=RunConfig(max_decisions=3), seed=0)
    assert len(reports) == 3
    assert reports[0].policy == "bandit"
    assert reports[1].policy == "bc" and reports[2].policy == "bc"
    assert reports[2].n_accumulated_traces > reports[0].n_accumulated_traces
    for rep in reports:
        assert {"solve_rate", "mean_cost", "p95_cost", "gen_verif_gap"} <= set(rep.eval)
    assert "round" in format_curve(reports)


def test_self_improve_dpo_runs():
    ex, pl, vf, tm = _stack()
    train = [Task(id=f"tr{i}", prompt="q") for i in range(6)]
    eval_ = [Task(id=f"ev{i}", prompt="q") for i in range(2)]
    reports = self_improve(train, eval_, ex, vf, tm, planner=pl, learner="dpo",
                           rounds=1, cfg=RunConfig(max_decisions=3), seed=0)
    assert len(reports) == 2
    assert reports[1].policy == "dpo"


def test_arithmetic_benchmark_offline():
    b = ArithmeticBenchmark(n_atomic=3, n_multi=2, n_underspecified=1, seed=0)
    tasks = b.tasks()
    assert len(tasks) == 6
    v = b.terminal_verifier()

    atomic = next(t for t in tasks if t.metadata["kind"] == "atomic")
    assert v.score_final(atomic, str(atomic.metadata["gold"])).value == 1.0
    assert v.score_final(atomic, "definitely 99999 wrong").value == 0.0

    under = next(t for t in tasks if t.metadata["kind"] == "underspecified")
    assert v.score_final(under, "anything").value == 0.0

    calc = b.tools()["calc"]
    assert calc(expr="2*(3+4)") == "14.0"
    assert safe_eval("2 * (3 + 4)") == 14.0
