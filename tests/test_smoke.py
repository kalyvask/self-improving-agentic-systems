"""Offline smoke test: exercises the full stack with a scripted fake LLM client.

Runs with no API key and no network. It proves the wiring is sound -- cost
accounting, the bandit allocator, the ReAct executor loop, the planner, the
verifiers, the round runner, trace logging, credit assignment, and the metrics
-- all fit together and produce a well-formed TaskTrace. The live OpenRouter
path is checked separately by scripts/smoke_live.py once a key is set.
"""
from __future__ import annotations

import json

from wdp.cost import CostLedger, Spend
from wdp.llm.openrouter import LLMResponse
from wdp.executor.react import Executor, Task
from wdp.planner.decompose import Planner
from wdp.verifier.scorer import LLMProcessVerifier, Score
from wdp.allocator.policy import BanditAllocator, Action
from wdp.loop.runner import RunConfig, run_task, run_round
from wdp.loop.trace import TaskTrace, TraceLog, assign_credit, DecisionRecord
from wdp.metrics import (
    success_at_budget, pass_hat_k, pass_at_k, risk_coverage, cvar,
    generation_verification_gap, summarize_round,
)


class FakeClient:
    """Scripted stand-in for OpenRouterClient. Branches on the system prompt so a
    single client can serve executor turns, process scores, and plans -- while
    still folding a realistic Spend into the ledger like the real client does."""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, model, messages, *, ledger=None, parallel_group=None,
             temperature=0.7, max_tokens=None, **kwargs) -> LLMResponse:
        self.calls += 1
        system = messages[0]["content"] if messages else ""

        if "tool-using agent" in system:
            text = json.dumps({"thought": "I can answer directly.",
                               "action": "FINISH", "action_input": {"answer": "42"}})
        elif "progress grader" in system:
            text = "0.95"
        elif "how much this task benefits" in system:
            text = "0.2"
        elif "Decompose the task" in system:
            text = json.dumps([
                {"id": "s1", "prompt": "part one", "depends_on": []},
                {"id": "s2", "prompt": "part two", "depends_on": ["s1"]},
            ])
        else:
            text = "ok"

        spend = Spend(model=model, prompt_tokens=100, completion_tokens=20,
                      wall_seconds=0.5, dollars=0.001, parallel_group=parallel_group)
        if ledger is not None:
            ledger.add(spend)
        return LLMResponse(text=text, model=model, spend=spend, raw={})


class GoldVerifier:
    """Terminal verifier: the gold answer is '42'."""

    def score_final(self, task, answer: str) -> Score:
        return Score(value=1.0 if "42" in (answer or "") else 0.0)


def _stack():
    client = FakeClient()
    executor = Executor(client, "fake/model", tools={}, max_steps=5)
    planner = Planner(client, "fake/cheap")
    verifier = LLMProcessVerifier(client, "fake/cheap")
    terminal = GoldVerifier()
    allocator = BanditAllocator(seed=0)
    return client, executor, planner, verifier, terminal, allocator


def test_cost_ledger_parallel_is_max_not_sum():
    led = CostLedger()
    led.add(Spend("m", 10, 10, 2.0, 0.0, parallel_group="g"))
    led.add(Spend("m", 10, 10, 3.0, 0.0, parallel_group="g"))
    led.add(Spend("m", 10, 10, 1.0, 0.0, parallel_group=None))
    # serial 1.0 + max(2,3) for the group = 4.0
    assert led.latency == 4.0
    assert led.tokens == 60


def test_bandit_decides_and_updates():
    alloc = BanditAllocator(seed=1)
    from wdp.allocator.policy import NodeFeatures
    d = alloc.decide(NodeFeatures(decomposability=0.5), "dollars")
    assert d.action in (Action.WIDER, Action.DEEPER, Action.DECOMPOSE, Action.STOP)
    alloc.update(Action.WIDER, 0.9)
    assert alloc._alpha[Action.WIDER] > 1.0


def test_executor_finishes():
    client, executor, *_ = _stack()
    traj = executor.run(Task(id="t0", prompt="what is the answer?"))
    assert traj.final_answer == "42"
    assert traj.done


def test_run_task_produces_wellformed_trace():
    _, executor, planner, verifier, terminal, allocator = _stack()
    cfg = RunConfig(currency="dollars", budget=1.0, max_decisions=6)
    trace = run_task(Task(id="t1", prompt="solve it"), allocator, executor,
                     verifier, terminal, planner=planner, cfg=cfg)
    assert isinstance(trace, TaskTrace)
    assert trace.decisions
    assert trace.solved  # gold answer 42 is reachable in one WIDER
    assert trace.terminal_reward == 1.0
    # every decision has credit assigned
    for d in trace.decisions:
        assert "dollars" in trace.total_cost
        assert d.value_per_cost >= 0.0


def test_round_and_trace_log(tmp_path):
    _, executor, planner, verifier, terminal, allocator = _stack()
    tasks = [Task(id=f"t{i}", prompt="q") for i in range(3)]
    log = TraceLog(tmp_path / "traces.jsonl")
    traces = run_round(tasks, allocator, executor, verifier, terminal,
                       planner=planner, cfg=RunConfig(max_decisions=4), trace_log=log)
    assert len(traces) == 3
    # round-trips through JSONL
    reloaded = log.read()
    assert len(reloaded) == 3
    assert reloaded[0].task_id == "t0"


def test_metrics_smoke():
    successes = [[True, True], [True, False], [False, False]]
    assert pass_at_k(successes) == 2 / 3
    assert pass_hat_k(successes) == 1 / 3
    rc = risk_coverage([(0.9, True), (0.5, False), (0.8, True)])
    assert rc[0].y == 1.0  # most confident answer is correct
    assert cvar([1, 2, 3, 100], 0.75) >= 3.0

    t = TaskTrace(task_id="x", currency="dollars", policy="bandit",
                  solved=True, terminal_reward=1.0, total_cost={"dollars": 0.5})
    t.decisions.append(DecisionRecord(step=0, features=[], process_score_after=0.6))
    assert 0.0 <= generation_verification_gap([t]) <= 1.0
    s = summarize_round([t])
    assert s["solve_rate"] == 1.0
