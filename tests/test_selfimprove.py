"""Offline tests for the self-improvement driver and the local benchmark.

The driver test uses a scripted fake client (no key, no network) and checks the
*mechanics*: round 0 is the bandit, later rounds fit the named learner, every
round emits a well-formed scoreboard, and the curve formats. It does not assert
the curve goes up -- that needs real model variance and is what the live run is
for. The benchmark test checks the locally-checkable verifier directly.
"""
from __future__ import annotations

import json

from wdp.cost import Spend, CostLedger
from wdp.llm.openrouter import LLMResponse
from wdp.executor.react import Executor, Task, Trajectory, Step
from wdp.planner.decompose import Planner
from wdp.verifier.scorer import LLMProcessVerifier, Score
from wdp.loop import RunConfig, self_improve, format_curve
from wdp.loop.runner import _features
from wdp.loop.trace import DecisionRecord, TaskTrace, assign_credit
from wdp.allocator.policy import NodeFeatures
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


def test_self_improve_kto_runs():
    ex, pl, vf, tm = _stack()
    train = [Task(id=f"tr{i}", prompt="q") for i in range(6)]
    eval_ = [Task(id=f"ev{i}", prompt="q") for i in range(2)]
    reports = self_improve(train, eval_, ex, vf, tm, planner=pl, learner="kto",
                           rounds=1, cfg=RunConfig(max_decisions=3), seed=0)
    assert len(reports) == 2
    assert reports[1].policy == "kto"


def test_kto_prefers_desirable_action_over_reference():
    # KTO needs no pairs: tag each decision good/bad by value_per_cost and it
    # should push probability toward the desirable action and away from the
    # undesirable one, relative to the BC reference it warm-starts from.
    import numpy as np
    from wdp.allocator.kto import KTOAllocator
    from wdp.allocator.bc import _INDEX
    from wdp.allocator.policy import Action

    feats = [1.0] + [0.0] * 11   # a single fixed state (12-dim feature vector)
    good = TaskTrace(task_id="g", currency="dollars", policy="bandit",
                     solved=True, terminal_reward=1.0)
    good.add(DecisionRecord(step=0, features=feats, action="wider",
                            value_per_cost=0.9))
    bad = TaskTrace(task_id="b", currency="dollars", policy="bandit")
    bad.add(DecisionRecord(step=0, features=feats, action="deeper",
                           value_per_cost=0.0))

    kto = KTOAllocator(keep_fraction=1.0, seed=0)
    kto.fit([good, bad])
    p = kto.policy.probs(np.asarray(feats))
    ref = kto.reference.policy.probs(np.asarray(feats))
    w, d = _INDEX[Action.WIDER.value], _INDEX[Action.DEEPER.value]
    assert p[w] > ref[w]   # desirable action gains probability
    assert p[d] < ref[d]   # undesirable action loses it


def test_grpo_update_shifts_probability_toward_high_advantage_action():
    # One fixed state. WIDER rollouts get positive group-relative advantage,
    # DEEPER rollouts negative. A GRPO update must raise p(WIDER) and lower
    # p(DEEPER) relative to the BC reference it warm-starts from.
    import numpy as np
    from wdp.allocator.linear import LinearSoftmaxPolicy
    from wdp.allocator.bc import _INDEX
    from wdp.allocator.policy import Action, NodeFeatures

    F = len(NodeFeatures.names())
    feats = [1.0] + [0.0] * (F - 1)
    ref = LinearSoftmaxPolicy(n_features=F, n_actions=4, seed=0)
    ref.fit_bc(np.array([feats, feats]), np.array([0, 1]))   # sets scaler, _fitted
    pol = LinearSoftmaxPolicy(n_features=F, n_actions=4, seed=0)
    pol.mu, pol.sigma = ref.mu.copy(), ref.sigma.copy()
    pol.W, pol.b = ref.W.copy(), ref.b.copy()

    w, d = _INDEX[Action.WIDER.value], _INDEX[Action.DEEPER.value]
    X = [feats] * 6
    actions = [w, w, w, d, d, d]
    advs = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]
    pol.grpo_update(np.array(X), np.array(actions), np.array(advs),
                    reference=ref, beta_kl=0.0, inner_epochs=20)
    p, p_ref = pol.probs(np.array(feats)), ref.probs(np.array(feats))
    assert p[w] > p_ref[w]
    assert p[d] < p_ref[d]


def test_grpo_train_runs_with_fake_stack():
    from wdp.loop.runner import run_round
    from wdp.allocator.policy import BanditAllocator
    from wdp.loop.grpo_train import grpo_train, format_grpo_curve

    ex, pl, vf, tm = _stack()
    train = [Task(id=f"tr{i}", prompt="q") for i in range(4)]
    eval_ = [Task(id=f"ev{i}", prompt="q") for i in range(2)]
    cfg = RunConfig(max_decisions=3)
    seed = run_round(train, BanditAllocator(seed=0), ex, vf, tm, planner=pl,
                     cfg=cfg, policy_name="bandit", explore=True)
    reports, alloc = grpo_train(
        seed, train, eval_, ex, vf, tm, planner=pl, cfg=cfg,
        group_size=2, prompts_per_step=2, num_steps=2, eval_every=1, seed=0)
    assert reports[0].step == 0 and reports[-1].step == 2
    assert reports[-1].n_rollouts == 2 * 2 * 2          # steps*prompts*group
    for r in reports:
        assert {"solve_rate", "mean_cost", "p95_cost", "gen_verif_gap"} <= set(r.eval)
    assert "step" in format_grpo_curve(reports)


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


def test_score_abstention_tracks_solvability():
    b = ArithmeticBenchmark(n_atomic=1, n_multi=0, n_underspecified=1, seed=0)
    tasks = b.tasks()
    v = b.terminal_verifier()
    under = next(t for t in tasks if t.metadata["kind"] == "underspecified")
    atomic = next(t for t in tasks if t.metadata["kind"] == "atomic")
    assert v.score_abstention(under).value == 1.0   # abstaining was right
    assert v.score_abstention(atomic).value == 0.0  # gave up on a solvable task


def _stop_trace(abstention_reward: float) -> TaskTrace:
    tr = TaskTrace(task_id="t", currency="dollars", policy="bc",
                   abstention_reward=abstention_reward)
    tr.add(DecisionRecord(step=0, features=[0.0], action="stop",
                          cost_before=0.0, cost_after=0.0))
    return tr


def _solved_trace(spend: float) -> TaskTrace:
    tr = TaskTrace(task_id="t", currency="dollars", policy="bc",
                   solved=True, terminal_reward=1.0,
                   total_cost={"dollars": spend})
    tr.add(DecisionRecord(step=0, features=[0.0], action="deeper",
                          cost_before=0.0, cost_after=spend))
    return tr


def test_cheaper_solve_earns_higher_credit():
    # The thesis fix: with a budget, an equally-correct but cheaper solve must
    # earn strictly higher value-per-cost, so the learner becomes cost-aware
    # instead of treating every solve as equally good.
    cheap = _solved_trace(0.02)
    pricey = _solved_trace(0.18)
    assign_credit(cheap, budget=0.2)
    assign_credit(pricey, budget=0.2)
    assert cheap.decisions[0].value_per_cost > pricey.decisions[0].value_per_cost

    # Without a budget the cost term is neutral (back-compat: pure outcome).
    flat = _solved_trace(0.18)
    assign_credit(flat)
    assert flat.decisions[0].value_per_cost == 1.0

    # Regression: a SOLVED task that hit/exceeded its budget must still train as a
    # win, not get zeroed. The old `1 - spent/budget` drove credit to 0 for any
    # solve at/over budget, which erased exactly the expensive-but-winning WIDER
    # traces and taught the policy to avoid the action with the most headroom.
    overspent = _solved_trace(0.25)        # spent > budget
    assign_credit(overspent, budget=0.2)
    assert overspent.decisions[0].value_per_cost > 0.0

    # Regression: the smooth decay must keep discriminating PAST the budget. The
    # old `1 - cost_weight*min(1, spent/budget)` cap flattened every over-budget
    # solve to the same floor, so a 3x-budget blowout trained identically to a
    # marginal 1.1x overspend -- no gradient against runaway spend. A gross
    # overspend must now earn strictly less credit than a marginal one.
    marginal = _solved_trace(0.22)         # 1.1x budget
    gross = _solved_trace(0.60)            # 3x budget
    assign_credit(marginal, budget=0.2)
    assign_credit(gross, budget=0.2)
    assert gross.decisions[0].value_per_cost < marginal.decisions[0].value_per_cost
    assert gross.decisions[0].value_per_cost > 0.0


def test_stop_credit_comes_from_abstention_reward():
    # The bug we fixed: STOP must NOT be credited just because the task went
    # unsolved. A premature stop (abstention_reward 0) earns 0; only a correct
    # abstention (reward 1) earns credit.
    wrong = _stop_trace(0.0)
    assign_credit(wrong)
    assert wrong.decisions[0].value_per_cost == 0.0

    right = _stop_trace(1.0)
    assign_credit(right)
    assert right.decisions[0].value_per_cost == 1.0


def _multi_trace(process_scores: list[float]) -> TaskTrace:
    tr = TaskTrace(task_id="t", currency="dollars", policy="bc",
                   solved=True, terminal_reward=1.0)
    for i, ps in enumerate(process_scores):
        tr.add(DecisionRecord(step=i, features=[0.0], action="deeper",
                              cost_before=0.0, cost_after=0.0,
                              process_score_after=ps))
    return tr


def test_advantage_weighting_credits_the_decision_that_moved_the_score():
    # Lever #4: on a solved task, the decision that actually raised the process
    # score should earn more credit than the one that did nothing. Here only the
    # last decision moves the verifier signal (0.2 -> 0.2 -> 0.9).
    tr = _multi_trace([0.2, 0.2, 0.9])
    assign_credit(tr)  # no budget => efficiency neutral; isolates the advantage term
    v = [d.value_per_cost for d in tr.decisions]
    assert v[2] > v[0] > v[1]   # mover > set-up > the do-nothing middle step


def test_advantage_weighting_falls_back_to_uniform_without_signal():
    # When no decision moves the process score, credit collapses to uniform (the
    # pre-lever behavior), so flat-signal traces stay well-behaved.
    tr = _multi_trace([0.0, 0.0, 0.0])
    assign_credit(tr)
    v = [d.value_per_cost for d in tr.decisions]
    assert v[0] == v[1] == v[2] == 1.0


def test_structural_features_tool_error_and_done_frac():
    # Lever #3: tool_error_rate and attempts_done_frac are computed for free from
    # the trajectories. Two attempts, three steps, two of which errored; one
    # attempt finished and one did not.
    done = Trajectory(task_id="a", final_answer="done")
    done.steps = [Step(thought="", observation="ERROR: bad tool"),
                  Step(thought="", observation="ok result")]
    truncated = Trajectory(task_id="a")
    truncated.steps = [Step(thought="", observation="Error: env blew up")]

    feats = _features([done, truncated], [0.5, 0.7], ledger=CostLedger(),
                      cfg=RunConfig(budget=1.0), decomposability=0.0)
    assert abs(feats.tool_error_rate - 2 / 3) < 1e-9
    assert feats.attempts_done_frac == 0.5
    # Difficulty is pinned to the FIRST attempt's process score (1 - 0.5), not the
    # best-so-far, so it reads intrinsic hardness even as score_max climbs.
    assert abs(feats.difficulty - 0.5) < 1e-9
    # Before any attempt is scored, difficulty is the neutral 0.5 prior.
    blank = _features([], [], ledger=CostLedger(), cfg=RunConfig(budget=1.0),
                      decomposability=0.0)
    assert blank.difficulty == 0.5
    # vector() and names() stay in lockstep so the policy auto-sizes.
    assert len(feats.vector()) == len(NodeFeatures.names()) == 12
