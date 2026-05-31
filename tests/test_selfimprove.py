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


def test_kto_does_not_inflate_a_mostly_undesirable_action():
    # Regression for the double-beta bug: the sigmoid argument must be (r - z), not
    # beta*(r - z). With the extra beta the argument pinned near 0, sigmoid never
    # saturated, and constant per-example gradients drained probability from
    # down-weighted undesirable spends onto the rarely-used STOP action -- which
    # collapsed the live policy to ~50% STOP. A mostly-undesirable action must NOT
    # gain probability over the reference.
    import numpy as np
    from wdp.allocator.linear import LinearSoftmaxPolicy
    from wdp.allocator.bc import _INDEX
    from wdp.allocator.policy import Action, NodeFeatures

    F = len(NodeFeatures.names())
    feats = [1.0] + [0.0] * (F - 1)
    w, d, s = (_INDEX[Action.WIDER.value], _INDEX[Action.DEEPER.value],
               _INDEX[Action.STOP.value])
    ref = LinearSoftmaxPolicy(n_features=F, n_actions=4, seed=0)
    ref.fit_bc(np.array([feats] * 3), np.array([w, d, d]))      # STOP rare in ref
    pol = LinearSoftmaxPolicy(n_features=F, n_actions=4, seed=0)
    pol.mu, pol.sigma = ref.mu.copy(), ref.sigma.copy()
    pol.W, pol.b = ref.W.copy(), ref.b.copy()

    # 12 desirable DEEPER, 8 undesirable STOP (premature). KTO must push STOP down.
    X = np.array([feats] * 20)
    actions = np.array([d] * 12 + [s] * 8)
    desirable = np.array([True] * 12 + [False] * 8)
    pol.fit_kto(X, actions, desirable, reference=ref, beta=0.1)
    p, p_ref = pol.probs(np.array(feats)), ref.probs(np.array(feats))
    assert p[s] <= p_ref[s] + 1e-3          # mostly-undesirable STOP not inflated
    assert p[s] < 0.2                        # nowhere near the ~0.5 collapse
    assert p[d] > p_ref[d]                   # desirable action still gains


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
    # dynamic_sampling off here: the fake stack always solves, so every group is
    # all-solve and would be skipped; off keeps the update path exercised.
    reports, alloc = grpo_train(
        seed, train, eval_, ex, vf, tm, planner=pl, cfg=cfg,
        group_size=2, prompts_per_step=2, num_steps=2, eval_every=1, seed=0,
        dynamic_sampling=False)
    assert reports[0].step == 0 and reports[-1].step == 2
    assert reports[-1].n_rollouts == 2 * 2 * 2          # steps*prompts*group
    for r in reports:
        assert {"solve_rate", "mean_cost", "p95_cost", "gen_verif_gap"} <= set(r.eval)
    assert "step" in format_grpo_curve(reports)


def test_rollout_difficulty_grounds_and_caches():
    # Math-Shepherd-style difficulty: fork N fresh attempts, grade with the free
    # terminal verifier, difficulty = 1 - solve fraction. The fake stack always
    # answers "42" which the GoldVerifier accepts, so every fork solves -> the
    # easy task gets difficulty 0; results are cached (no re-forking).
    from wdp.verifier.rollout import RolloutProcessVerifier
    from wdp.cost import CostLedger
    ex, pl, vf, tm = _stack()
    rpv = RolloutProcessVerifier(ex, tm, n_rollouts=4)
    task = Task(id="t0", prompt="q")
    led = CostLedger()
    assert rpv.difficulty(task, ledger=led) == 0.0      # all forks solve -> easy
    spent_after_first = led.amount("dollars")
    assert rpv.difficulty(task, ledger=led) == 0.0      # cached
    assert led.amount("dollars") == spent_after_first   # no extra spend on cache hit


def test_self_improve_accepts_rollout_difficulty():
    from wdp.verifier.rollout import RolloutProcessVerifier
    ex, pl, vf, tm = _stack()
    rpv = RolloutProcessVerifier(ex, tm, n_rollouts=2)
    train = [Task(id=f"tr{i}", prompt="q") for i in range(4)]
    eval_ = [Task(id=f"ev{i}", prompt="q") for i in range(2)]
    reports = self_improve(train, eval_, ex, vf, tm, planner=pl, learner="bc",
                           rounds=1, cfg=RunConfig(max_decisions=3), seed=0,
                           difficulty_fn=rpv.difficulty)
    assert len(reports) == 2 and reports[1].policy == "bc"


def test_bc_keeps_decompose_solves_not_just_cheap_atomic():
    # Bug: BC kept only the top-fraction by mean value-per-cost (cheap atomic solves)
    # and discarded expensive-but-correct DECOMPOSE solves, starving the reference of
    # the action that helps multi tasks. Now it keeps all successes.
    from wdp.allocator.bc import _filter_traces
    cheap = TaskTrace(task_id="a", currency="dollars", policy="bandit", solved=True, terminal_reward=1.0)
    cheap.add(DecisionRecord(step=0, features=[0.0], action="wider", value_per_cost=0.95))
    pricey = TaskTrace(task_id="m", currency="dollars", policy="bandit", solved=True, terminal_reward=1.0)
    pricey.add(DecisionRecord(step=0, features=[0.0], action="decompose", value_per_cost=0.6))
    failed = TaskTrace(task_id="f", currency="dollars", policy="bandit", solved=False)
    failed.add(DecisionRecord(step=0, features=[0.0], action="wider", value_per_cost=0.0))
    kept = _filter_traces([cheap, pricey, failed], keep_fraction=0.3)
    actions = {d.action for t in kept for d in t.decisions}
    assert "decompose" in actions and "wider" in actions   # both successes survive
    assert failed not in kept                               # failure excluded


def test_stop_after_failed_attempts_abstains_on_hopeless_task():
    # The hopeless-task rule: on a non-decomposable task (planner=None -> decomp 0)
    # with no progress after k attempts, abstain. Gives STOP a path into the data.
    from wdp.loop.runner import run_task
    from wdp.allocator.policy import Action, Decision

    class WiderAlloc:
        def decide(self, f, c, *, explore=False):
            return Decision(action=Action.WIDER, scores={Action.WIDER: 1.0})

    class FailExec:
        def run(self, task, *, ledger=None, parallel_group=None):
            return Trajectory(task_id=task.id, final_answer="nope", parallel_group=parallel_group)

    class NeverSolve:
        def score_final(self, task, ans):
            return Score(value=0.0)

    class V:
        def score_step(self, task, partial, *, ledger=None):
            return Score(value=0.0)

    trace = run_task(Task(id="u", prompt="q"), WiderAlloc(), FailExec(), verifier=V(),
                     terminal=NeverSolve(), planner=None,
                     cfg=RunConfig(max_decisions=6, stop_after_failed_attempts=2))
    assert trace.decisions[-1].action == "stop" and not trace.solved
    # Disabled by default (back-compat): no forced STOP.
    t2 = run_task(Task(id="u", prompt="q"), WiderAlloc(), FailExec(), verifier=V(),
                  terminal=NeverSolve(), planner=None, cfg=RunConfig(max_decisions=3))
    assert all(d.action != "stop" for d in t2.decisions)


def test_decompose_synthesizes_parent_answer():
    # Bug #1: DECOMPOSE must SYNTHESIZE a parent answer, not concatenate sub-answers.
    # Before the fix parent.final_answer was "[s1] 6\n[s2] 20", graded on the last
    # number (20) not the sum (26), so DECOMPOSE could never solve a multi-part task.
    from wdp.loop.runner import run_task
    from wdp.planner.decompose import SubTaskDAG, SubTask
    from wdp.benchmarks.arithmetic import ArithmeticVerifier
    from wdp.allocator.policy import Action, Decision

    class DecomposeAllocator:
        def decide(self, feats, currency, *, explore=False):
            return Decision(action=Action.DECOMPOSE, scores={Action.DECOMPOSE: 1.0})

    class FixedPlanner:
        def probe(self, task, *, parallel_group=None, ledger=None):
            return 1.0
        def decompose(self, task, *, parallel_group=None, ledger=None):
            return SubTaskDAG(parent_id=task.id, subtasks=[
                SubTask(task=Task(id="p::s1", prompt="Compute 2 * 3.", metadata={"sub_id": "s1"})),
                SubTask(task=Task(id="p::s2", prompt="Compute 4 * 5.", metadata={"sub_id": "s2"})),
            ])

    class FixedExecutor:
        def run(self, task, *, ledger=None, parallel_group=None):
            ans = "26" if "synthesis" in task.id else (
                "6" if "2 * 3" in task.prompt else "20" if "4 * 5" in task.prompt else "0")
            return Trajectory(task_id=task.id, final_answer=ans, parallel_group=parallel_group)

    class ConstVerifier:
        def score_step(self, task, partial, *, ledger=None):
            return Score(value=1.0)

    task = Task(id="p", prompt="Compute 2 * 3 and 4 * 5, then sum.", metadata={"gold": 26.0})
    trace = run_task(task, DecomposeAllocator(), FixedExecutor(), verifier=ConstVerifier(),
                     terminal=ArithmeticVerifier(), planner=FixedPlanner(),
                     cfg=RunConfig(max_decisions=1))
    assert trace.solved                      # synthesized 26 -> graded correct (was 20)


def test_decompose_masked_when_no_planner():
    # Bug #2: with planner=None, DECOMPOSE cannot run; it must be re-picked to the
    # best available action instead of being logged as a zero-cost no-op.
    from wdp.loop.runner import run_task
    from wdp.allocator.policy import Action, Decision

    class DecomposePreferring:
        def decide(self, feats, currency, *, explore=False):
            return Decision(action=Action.DECOMPOSE,
                            scores={Action.DECOMPOSE: 0.6, Action.WIDER: 0.3, Action.DEEPER: 0.1})

    ex, pl, vf, tm = _stack()
    trace = run_task(Task(id="t", prompt="q"), DecomposePreferring(), ex, vf, tm,
                     planner=None, cfg=RunConfig(max_decisions=1))
    assert "decompose" not in [d.action for d in trace.decisions]


def test_escalate_masked_when_no_strong_executor():
    # Like the DECOMPOSE-without-planner mask: an ESCALATE preference with no strong
    # executor wired in must be re-picked to the best CHEAP spend, never logged as a
    # no-op and never silently turned into a STOP (fake abstention).
    from wdp.loop.runner import run_task
    from wdp.allocator.policy import Action, Decision

    class EscalatePreferring:
        def decide(self, feats, currency, *, explore=False):
            return Decision(action=Action.ESCALATE,
                            scores={Action.ESCALATE: 0.7, Action.WIDER: 0.2,
                                    Action.STOP: 0.5, Action.DEEPER: 0.1})

    ex, pl, vf, tm = _stack()
    trace = run_task(Task(id="t", prompt="q"), EscalatePreferring(), ex, vf, tm,
                     planner=None, cfg=RunConfig(max_decisions=1), strong_executor=None)
    acts = [d.action for d in trace.decisions]
    assert "escalate" not in acts and "stop" not in acts
    assert acts == ["wider"]            # re-picked to the best available cheap spend


def test_escalate_is_a_rescue_after_a_failed_cheap_attempt():
    # Cascade semantics: ESCALATE is masked at step 0 (n_children==0) even with a strong
    # executor wired in -- the cheap model must attempt first. So the controller tries
    # cheap (step 0), and only after that miss does ESCALATE run the strong model (step
    # 1), solving the task and billing the strong call's higher cost. This is the gate
    # that stops the policy from collapsing to always-escalate-at-step-0.
    from wdp.loop.runner import run_task
    from wdp.allocator.policy import Action, Decision
    from wdp.cost import Spend
    from wdp.benchmarks.arithmetic import ArithmeticVerifier

    class AlwaysEscalate:            # wants ESCALATE every step; WIDER is the fallback
        def decide(self, feats, currency, *, explore=False):
            return Decision(action=Action.ESCALATE,
                            scores={Action.ESCALATE: 0.9, Action.WIDER: 0.5, Action.DEEPER: 0.1})

    class CheapFails:                # cheap model attempts and gets it WRONG (gold=42)
        def run(self, task, *, ledger=None, parallel_group=None):
            ledger.add(Spend(model="cheap", prompt_tokens=10, completion_tokens=5,
                             wall_seconds=0.1, dollars=0.0002, parallel_group=parallel_group))
            return Trajectory(task_id=task.id, final_answer="0", parallel_group=parallel_group)

    class StrongSolves:              # strong model rescues it and bills a pricey call
        def run(self, task, *, ledger=None, parallel_group=None):
            ledger.add(Spend(model="strong", prompt_tokens=100, completion_tokens=50,
                             wall_seconds=1.0, dollars=0.01, parallel_group=parallel_group))
            return Trajectory(task_id=task.id, final_answer="42", parallel_group=parallel_group)

    class DummyVerifier:
        def score_step(self, task, partial, *, ledger=None):
            return Score(value=0.0)

    task = Task(id="t", prompt="q", metadata={"gold": 42.0})
    trace = run_task(task, AlwaysEscalate(), CheapFails(), verifier=DummyVerifier(),
                     terminal=ArithmeticVerifier(),
                     cfg=RunConfig(max_decisions=2, currency="dollars"),
                     strong_executor=StrongSolves())
    acts = [d.action for d in trace.decisions]
    assert acts == ["wider", "escalate"]          # step 0 masked to cheap; step 1 rescues
    assert trace.decisions[-1].escalate_mode == "fresh_retry"   # no live env -> fresh
    assert trace.solved                            # strong solved after the cheap miss
    assert (trace.total_cost or {}).get("dollars", 0.0) >= 0.0102   # cheap + strong billed


def test_deeper_with_no_unfinished_target_is_relabeled_wider():
    # DEEPER at step 0 (or whenever nothing is unfinished) executes as a fresh attempt,
    # so the trace must record WIDER -- not DEEPER -- otherwise the policy is trained to
    # think "DEEPER first" works when a fresh attempt actually ran.
    from wdp.loop.runner import run_task
    from wdp.allocator.policy import Action, Decision

    class DeeperPreferring:
        def decide(self, feats, currency, *, explore=False):
            return Decision(action=Action.DEEPER,
                            scores={Action.DEEPER: 0.7, Action.WIDER: 0.3})

    ex, pl, vf, tm = _stack()
    trace = run_task(Task(id="t", prompt="q"), DeeperPreferring(), ex, vf, tm,
                     planner=None, cfg=RunConfig(max_decisions=1))
    assert [d.action for d in trace.decisions] == ["wider"]


def test_escalate_hands_off_a_live_env_via_continue_from():
    # True handoff: when the cheap model leaves an UNFINISHED trajectory with a live env
    # (e.g. tau-bench ran out of steps mid-conversation), ESCALATE resumes THAT env with
    # the strong model (continue_from), not a fresh attempt. Arithmetic has no env so it
    # always takes the fresh path -- this exercises the env-bearing handoff branch.
    from wdp.loop.runner import run_task
    from wdp.allocator.policy import Action, Decision
    from wdp.cost import Spend
    from wdp.benchmarks.arithmetic import ArithmeticVerifier

    class AlwaysEscalate:
        def decide(self, feats, currency, *, explore=False):
            return Decision(action=Action.ESCALATE,
                            scores={Action.ESCALATE: 0.9, Action.WIDER: 0.5})

    class CheapUnfinished:           # leaves an unfinished trajectory holding a live env
        def run(self, task, *, ledger=None, parallel_group=None):
            ledger.add(Spend(model="cheap", prompt_tokens=10, completion_tokens=5,
                             wall_seconds=0.1, dollars=0.0002, parallel_group=parallel_group))
            return Trajectory(task_id=task.id, final_answer=None, env=object(),
                              parallel_group=parallel_group)

    class StrongHandoff:
        def __init__(self):
            self.ran = self.continued = False
        def run(self, task, *, ledger=None, parallel_group=None):
            self.ran = True
            return Trajectory(task_id=task.id, final_answer="42", parallel_group=parallel_group)
        def continue_from(self, task, traj, *, ledger=None, parallel_group=None, extra_steps=None):
            self.continued = True
            ledger.add(Spend(model="strong", prompt_tokens=100, completion_tokens=50,
                             wall_seconds=1.0, dollars=0.01, parallel_group=parallel_group))
            traj.final_answer = "42"          # MUTATE in place + return same (real semantics)
            return traj

    class DummyVerifier:
        def score_step(self, task, partial, *, ledger=None):
            return Score(value=0.3)

    strong = StrongHandoff()
    trace = run_task(Task(id="t", prompt="q", metadata={"gold": 42.0}),
                     AlwaysEscalate(), CheapUnfinished(), verifier=DummyVerifier(),
                     terminal=ArithmeticVerifier(),
                     cfg=RunConfig(max_decisions=2, currency="dollars"),
                     strong_executor=strong)
    assert [d.action for d in trace.decisions] == ["wider", "escalate"]
    assert strong.continued and not strong.ran    # resumed the env, did NOT start fresh
    assert trace.decisions[-1].escalate_mode == "live_handoff"
    assert trace.solved


def test_deeper_targets_unfinished_and_does_not_revise_a_completed_answer():
    # DEEPER semantics contract. The runner only continues a trajectory that has
    # genuine unfinished work (or a self-reported stall); on an all-completed set
    # there is nothing to continue, so _deepest_unfinished returns None and the
    # runner falls back to a fresh attempt (WIDER-equivalent) rather than a no-op
    # "revise". This locks the behavior so the learner's DEEPER distribution is not
    # silently misleading. A true review-and-revise mode (needs executor support)
    # would change THIS assertion -- the test is the tripwire for that work.
    from wdp.loop.runner import _deepest_unfinished

    completed = Trajectory(task_id="t", steps=[Step(thought="x")], final_answer="42")
    # all answered, none stalled -> no continuation target (runner does a fresh attempt)
    assert _deepest_unfinished([completed]) is None

    unfinished = Trajectory(task_id="t", steps=[Step(thought="x")], final_answer=None)
    assert _deepest_unfinished([completed, unfinished]) is unfinished

    # a self-reported give-up (stalled) IS revisable even though it has an answer
    stalled = Trajectory(task_id="t", steps=[Step(thought="x")], final_answer="42", stalled=True)
    assert _deepest_unfinished([stalled]) is stalled

    # among several unfinished, DEEPER continues the deepest (most progressed) one
    shallow = Trajectory(task_id="t", steps=[Step(thought="a")], final_answer=None)
    deep = Trajectory(task_id="t", steps=[Step(thought="a"), Step(thought="b")], final_answer=None)
    assert _deepest_unfinished([shallow, deep]) is deep


def test_rollout_difficulty_bills_a_labeling_ledger():
    # Bug #3: forked difficulty rollouts must be billed somewhere (visible), not vanish.
    from wdp.verifier.rollout import RolloutProcessVerifier
    ex, pl, vf, tm = _stack()
    rpv = RolloutProcessVerifier(ex, tm, n_rollouts=3)
    rpv.difficulty(Task(id="t0", prompt="q"))           # no ledger -> labeling ledger
    assert rpv.labeling_ledger.amount("dollars") > 0


def test_subtasks_exclude_parent_gold():
    # Bug #4: subtasks must not inherit the parent's gold (would mis-grade a subtask).
    pl = Planner(FakeClient(), "fake")
    task = Task(id="p", prompt="q", metadata={"gold": 26.0, "kind": "multi"})
    dag = pl.decompose(task)
    assert dag.subtasks
    for st in dag.subtasks:
        assert "gold" not in st.task.metadata
        assert st.task.metadata.get("parent_gold") == 26.0


def test_credit_ordering_guard():
    # Bug #5: abstention_credit must stay below solve_floor or solves lose to STOP.
    import pytest
    with pytest.raises(ValueError):
        assign_credit(_solved_trace(0.02), budget=0.2,
                      abstention_credit=0.6, solve_floor=0.5)


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

    # A correct abstention earns positive credit, but SCALED below a solve
    # (abstention_credit=0.5) so it can't out-value actually solving the task --
    # the asymmetry that drove the controller to drift toward STOP across rounds.
    right = _stop_trace(1.0)
    assign_credit(right)
    assert right.decisions[0].value_per_cost == 0.5
    # A correct, reasonably cheap solve must out-value a correct abstention.
    solve = _solved_trace(0.02)
    assign_credit(solve, budget=0.2)
    assert solve.decisions[0].value_per_cost > right.decisions[0].value_per_cost


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
