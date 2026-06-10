"""Round runner: execute one task under the Allocator, logging a TaskTrace.

`run_task` is the control loop the whole project orbits. Given a task and a
budget in one currency, it repeatedly asks the Allocator for the next action,
executes it (spawning/continuing Executors or invoking the Planner), folds the
cost into the shared ledger, recomputes the cheap NodeFeatures from the verifier
scores it can see, and stops when the Allocator says STOP or the budget runs out.
Each decision is logged; at the end we score the best answer and assign credit.

`run_round` drives a batch of tasks and is the unit of the self-improvement
curve: collect traces with the current policy, then a trainer (BC/DPO) fits the
next policy from those traces. The improvement across rounds is the headline
result.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from wdp.allocator.policy import Action, Decision, NodeFeatures
from wdp.cost import CostLedger
from wdp.executor.react import Executor, Task, Trajectory
from wdp.loop.trace import DecisionRecord, TaskTrace, assign_credit
from wdp.planner.decompose import Planner
from wdp.verifier.scorer import ProcessVerifier, TerminalVerifier


@dataclass
class RunConfig:
    currency: str = "dollars"
    budget: float = 1.0
    solved_threshold: float = 0.99       # terminal score at/above which a task is "solved"
    max_decisions: int = 12              # hard cap on allocation steps per task
    deeper_extra_steps: int = 6
    cost_weight: float = 0.5             # credit cost-efficiency steepness (thesis knob)
    abstention_credit: float = 0.5       # scale on correct-STOP credit (< solve scale)
    solve_floor: float = 0.6             # min cost-efficiency for a solve (> abstention_credit)
    stop_after_failed_attempts: int = 0  # >0: abstain (STOP) on a non-decomposable task after
                                         # this many attempts with no progress (0 = disabled)
    escalate_after: int = 1              # ESCALATE is masked until the cheap model has made
                                         # this many attempts. 1 = escalate after a single miss;
                                         # 2 = require a cheap RETRY first (cuts over-escalation:
                                         # tasks the cheap model can do on a 2nd try cost a cheap
                                         # call, not a strong one).


def _features(
    trajectories: list[Trajectory],
    process_scores: list[float],
    *,
    ledger: CostLedger,
    cfg: RunConfig,
    decomposability: float,
    difficulty_override: float | None = None,
) -> NodeFeatures:
    scores = process_scores or [0.0]
    spent = ledger.amount(cfg.currency)
    frac = max(0.0, 1.0 - spent / cfg.budget) if cfg.budget else 0.0
    depth = max((t.depth for t in trajectories), default=0)
    steps = sum(t.depth for t in trajectories)
    stalled = 1.0 if trajectories and all(t.stalled for t in trajectories) else 0.0
    # Structural signals, free from the trajectories we already hold.
    all_steps = [s for t in trajectories for s in t.steps]
    n_err = sum(1 for s in all_steps
                if str(s.observation).strip().lower().startswith("error"))
    tool_error_rate = n_err / len(all_steps) if all_steps else 0.0
    n_done = sum(1 for t in trajectories if t.final_answer is not None)
    attempts_done_frac = n_done / len(trajectories) if trajectories else 0.0
    # Difficulty: prefer a rollout-grounded estimate (Math-Shepherd style, passed in
    # as difficulty_override and fixed for the whole task) over the noisy
    # 1 - first_process_score proxy. The proxy is the fallback when no grounded
    # estimate is available; 0.5 = unknown before any attempt has been scored. Unlike
    # score_max (which rises as we improve) difficulty stays put, so the policy reads
    # intrinsic task hardness and conditions WIDER-vs-DEEPER on it.
    if difficulty_override is not None:
        difficulty = difficulty_override
    else:
        difficulty = (1.0 - process_scores[0]) if process_scores else 0.5
    return NodeFeatures(
        score_mean=float(statistics.fmean(scores)),
        score_max=float(max(scores)),
        score_std=float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0,
        n_children=len(trajectories),
        budget_remaining_frac=frac,
        depth=depth,
        steps_taken=steps,
        decomposability=decomposability,
        executor_stalled=stalled,
        tool_error_rate=tool_error_rate,
        attempts_done_frac=attempts_done_frac,
        difficulty=difficulty,
    )


def run_task(
    task: Task,
    allocator,
    executor: Executor,
    verifier: ProcessVerifier,
    terminal: TerminalVerifier,
    *,
    planner: Planner | None = None,
    cfg: RunConfig | None = None,
    policy_name: str = "bandit",
    explore: bool = False,
    update: bool = True,
    difficulty_fn=None,
    strong_executor: Executor | None = None,
) -> TaskTrace:
    cfg = cfg or RunConfig()
    ledger = CostLedger()
    trace = TaskTrace(task_id=task.id, currency=cfg.currency, policy=policy_name)

    trajectories: list[Trajectory] = []
    process_scores: list[float] = []
    decomposability = planner.probe(task, parallel_group=None, ledger=ledger) if planner else 0.0
    # Rollout-grounded difficulty (cached per task), fixed for the whole task; falls
    # back to the 1-first_process_score proxy inside _features when not provided.
    task_difficulty = difficulty_fn(task) if difficulty_fn is not None else None

    best_terminal = 0.0

    for step in range(cfg.max_decisions):
        feats = _features(trajectories, process_scores, ledger=ledger, cfg=cfg,
                           decomposability=decomposability,
                           difficulty_override=task_difficulty)
        # Hopeless-task abstention rule (gives the STOP arm a path into the data and
        # saves budget on unsolvable tasks): on a structurally non-decomposable task
        # (decomposability==0) with no terminal progress and ~0 process scores after
        # `stop_after_failed_attempts` attempts, abstain. Correct on underspecified
        # tasks (earns abstention credit); a premature give-up on a hard-but-solvable
        # task earns 0, so credit still distinguishes the two and the policy can learn.
        if (cfg.stop_after_failed_attempts and feats.decomposability <= 0.0
                and len(trajectories) >= cfg.stop_after_failed_attempts
                and best_terminal <= 0.0 and feats.score_max <= 0.05):
            trace.add(DecisionRecord(
                step=step, features=feats.vector().tolist(), action=Action.STOP.value,
                scores={}, currency=cfg.currency, cost_before=ledger.amount(cfg.currency),
                cost_after=ledger.amount(cfg.currency), process_score_after=feats.score_max,
            ))
            break

        decision = allocator.decide(feats, cfg.currency, explore=explore)
        # Mask unavailable actions: DECOMPOSE cannot run without a planner (a logged
        # no-op), and is structurally pointless on a non-decomposable task
        # (decomposability == 0, e.g. atomic/underspecified). Allowing it there lets
        # an exploratory decompose "succeed" on an atomic task and enter BC's kept
        # set, teaching decompose-on-atomic. Re-pick the best available action.
        if decision.action == Action.DECOMPOSE and (planner is None or feats.decomposability <= 0.0):
            # Fall back to the best SPEND action (exclude STOP): otherwise a masked
            # DECOMPOSE whose next-highest score is STOP silently becomes an abstention,
            # manufacturing fake "learned STOP" (premature stops on atomic tasks). STOP
            # must come from the policy genuinely ranking it first, or the evidence rule.
            avail = {a: v for a, v in decision.scores.items()
                     if a not in (Action.DECOMPOSE, Action.STOP)}
            # Hard fallback to WIDER if the policy somehow scored only DECOMPOSE/STOP,
            # so a masked action can never survive as a zero-cost no-op.
            nxt = max(avail, key=avail.get) if avail else Action.WIDER
            decision = Decision(action=nxt, scores=decision.scores)
        # Mask ESCALATE when (a) no stronger model is wired in, or (b) the cheap model
        # has not attempted yet (n_children == 0). (b) is the cascade SEMANTICS:
        # ESCALATE is a RESCUE after the cheap model fails, not a step-0 shortcut.
        # Without it the bandit discovers escalate-at-step-0 is a sure solve (one strong
        # call always works), the successful-trace set fills with escalate-first, and
        # BC/DPO clone "always escalate" -- collapsing the cascade to strong-only with no
        # cost saving. Forcing one cheap attempt first makes the policy learn to escalate
        # only the tasks the cheap model actually missed (selective). Re-pick the best
        # cheap spend so a masked ESCALATE never silently becomes a STOP (fake abstention).
        if decision.action == Action.ESCALATE and (strong_executor is None
                                                    or len(trajectories) < cfg.escalate_after):
            avail = {a: v for a, v in decision.scores.items()
                     if a not in (Action.ESCALATE, Action.STOP, Action.DECOMPOSE)}
            nxt = max(avail, key=avail.get) if avail else Action.WIDER
            decision = Decision(action=nxt, scores=decision.scores)
        cost_before = ledger.amount(cfg.currency)

        if decision.action == Action.STOP:
            trace.add(DecisionRecord(
                step=step, features=feats.vector().tolist(),
                action=Action.STOP.value, scores={a.value: v for a, v in decision.scores.items()},
                currency=cfg.currency, cost_before=cost_before, cost_after=cost_before,
                process_score_after=feats.score_max,
            ))
            break

        pg = f"{task.id}:step{step}"
        new_traj: Trajectory | None = None
        escalate_mode: str | None = None

        if decision.action == Action.WIDER:
            new_traj = executor.run(task, ledger=ledger, parallel_group=pg)
            trajectories.append(new_traj)

        elif decision.action == Action.DEEPER:
            # DEEPER should refine an UNFINISHED trajectory. The old code, when none
            # was unfinished, ran a fresh attempt and then continue_from on it -- but
            # continue_from is a no-op on an already-done trajectory (react stops on a
            # final answer), so DEEPER silently did "one fresh attempt + nothing", i.e.
            # it was not actually deeper. Make it honest: continue genuine unfinished
            # work; otherwise fall back to a fresh attempt (WIDER-equivalent) instead of
            # the wasted no-op. (Future: a true "review and revise the completed answer"
            # mode would let DEEPER lift hard-atomic solve -- needs executor support.)
            target = _deepest_unfinished(trajectories)
            if target is not None:
                new_traj = executor.continue_from(
                    task, target, ledger=ledger, parallel_group=pg,
                    extra_steps=cfg.deeper_extra_steps)
            else:
                # No unfinished trajectory to deepen -> this executes as a FRESH attempt
                # (WIDER-equivalent). RE-LABEL the decision to WIDER so the trace and
                # credit match what actually ran; logging it as "deeper" would teach the
                # policy that DEEPER-at-step-0 works when it was really a fresh attempt.
                decision = Decision(action=Action.WIDER, scores=decision.scores)
                new_traj = executor.run(task, ledger=ledger, parallel_group=pg)
                trajectories.append(new_traj)

        elif decision.action == Action.DECOMPOSE and planner is not None:
            new_traj = _run_decompose(task, planner, executor, ledger, pg)
            if new_traj is not None:
                trajectories.append(new_traj)

        elif decision.action == Action.ESCALATE and strong_executor is not None:
            # Hand the step to the stronger model, billed into the SAME ledger at its
            # (higher) price. TRUE HANDOFF when the cheap model left an unfinished
            # trajectory with a LIVE env (e.g. tau-bench ran out of steps mid-task): the
            # strong model resumes that same env/conversation rather than starting over.
            # Otherwise (no live env to resume -- always the case on stateless arithmetic,
            # where completed-wrong attempts have no env) it is a fresh strong attempt.
            # Gating on a live env keeps arithmetic behavior identical (always fresh).
            target = _deepest_unfinished(trajectories)
            if target is not None and getattr(target, "env", None) is not None:
                # continue_from MUTATES `target` in place and returns it, so it is
                # ALREADY in `trajectories` -- do NOT append (that double-counts the
                # same attempt: inflated n_children / attempts_done_frac). Mirrors DEEPER.
                escalate_mode = "live_handoff"
                new_traj = strong_executor.continue_from(
                    task, target, ledger=ledger, parallel_group=pg,
                    extra_steps=cfg.deeper_extra_steps)
            else:
                escalate_mode = "fresh_retry"
                new_traj = strong_executor.run(task, ledger=ledger, parallel_group=pg)
                trajectories.append(new_traj)

        # Score whatever we just produced. For a COMPLETED trajectory the terminal
        # grade is exact (and free on arithmetic; env-carried on tau-bench), so use
        # it as the process score rather than paying the cheap LLM scorer -- which
        # is near-noise (it rated unsolvable tasks 0.95 while terminal=0), corrupting
        # the score features and wasting spend. The LLM scorer is used only for
        # genuinely partial trajectories where no terminal grade exists yet.
        ps = feats.score_max
        if new_traj is not None:
            if new_traj.final_answer is not None:
                tv = (new_traj.reward if new_traj.reward is not None
                      else terminal.score_final(task, new_traj.final_answer).value)
                ps = tv
                if tv >= best_terminal:
                    best_terminal = tv
            else:
                ps = verifier.score_step(task, new_traj.transcript(), ledger=ledger).value
            process_scores.append(ps)

        cost_after = ledger.amount(cfg.currency)
        trace.add(DecisionRecord(
            step=step, features=feats.vector().tolist(),
            action=decision.action.value,
            scores={a.value: v for a, v in decision.scores.items()},
            currency=cfg.currency, cost_before=cost_before, cost_after=cost_after,
            process_score_after=ps, escalate_mode=escalate_mode,
        ))

        # Bandit online update (if the policy supports it). Skipped at eval so
        # the baseline isn't trained on the held-out tasks it's measured on.
        if update:
            _maybe_update(allocator, decision.action, ps, cost_before, cost_after)

        if best_terminal >= cfg.solved_threshold or cost_after >= cfg.budget:
            break

    trace.solved = best_terminal >= cfg.solved_threshold
    trace.terminal_reward = best_terminal
    # Ground-truth quality of abstaining on this task (1.0 only if it was
    # genuinely unsolvable). Verifiers without score_abstention default to 0.0,
    # so STOP earns credit only on benchmarks that mark unsolvable tasks.
    score_abstention = getattr(terminal, "score_abstention", None)
    if not trace.solved and score_abstention is not None:
        trace.abstention_reward = float(score_abstention(task).value)
    trace.total_cost = ledger.snapshot()
    assign_credit(trace, budget=cfg.budget, cost_weight=cfg.cost_weight,
                  abstention_credit=cfg.abstention_credit, solve_floor=cfg.solve_floor)
    return trace


def _deepest_unfinished(trajectories: list[Trajectory]) -> Trajectory | None:
    candidates = [t for t in trajectories if not t.done or t.stalled]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t.depth)


def _run_decompose(task, planner, executor, ledger, parallel_group) -> Trajectory | None:
    """Run a sub-task DAG; stitch sub-answers into one synthetic parent Trajectory.

    Each topological layer runs as one parallel_group so the latency currency
    bills the layer's max, not its sum (SPRINT-style parallel sub-agents)."""
    dag = planner.decompose(task, parallel_group=parallel_group, ledger=ledger)
    if not dag.subtasks:
        return None
    answers: list[str] = []
    parent = Trajectory(task_id=task.id, parallel_group=parallel_group)
    for li, layer in enumerate(dag.ready_layers()):
        lpg = f"{parallel_group}:layer{li}"
        for st in layer:
            sub_traj = executor.run(st.task, ledger=ledger, parallel_group=lpg)
            ans = sub_traj.final_answer or "(no answer)"
            answers.append(f"[{st.task.metadata.get('sub_id', st.task.id)}] {ans}")
            parent.steps.extend(sub_traj.steps)
    # Synthesis step: combine the sub-answers into the PARENT answer. Without this
    # the parent answer was just the concatenation of sub-answers, so the terminal
    # verifier graded it on the last sub-result (e.g. read 20 from "[s1] 6\n[s2] 20"
    # instead of the sum 26) -- DECOMPOSE could NEVER solve a multi-part task, so the
    # learner correctly suppressed it (0 solves across the calibrated runs) and the
    # multi-task headroom never materialized. Run one more executor pass over the
    # original task plus the sub-results; bill it to the same ledger.
    sub_block = "\n".join(answers)
    synth_task = Task(
        id=f"{task.id}::synthesis",
        prompt=(f"{task.prompt}\n\nSub-results already computed:\n{sub_block}\n\n"
                "Using ONLY these sub-results, combine them and FINISH with the single "
                "final answer."),
        metadata=task.metadata,
    )
    synth = executor.run(synth_task, ledger=ledger, parallel_group=f"{parallel_group}:synth")
    parent.steps.extend(synth.steps)
    parent.final_answer = synth.final_answer if synth.final_answer is not None else sub_block
    return parent


def _maybe_update(allocator, action, process_score, cost_before, cost_after) -> None:
    update = getattr(allocator, "update", None)
    if update is None:
        return
    marginal = max(cost_after - cost_before, 0.0)
    # Normalize value-per-cost into [0,1] with a soft squash so the Beta update
    # stays well-behaved regardless of currency magnitude.
    vpc = process_score / (1.0 + marginal) if marginal else process_score
    update(action, float(max(0.0, min(1.0, vpc))))


def run_round(
    tasks: list[Task],
    allocator,
    executor: Executor,
    verifier: ProcessVerifier,
    terminal: TerminalVerifier,
    *,
    planner: Planner | None = None,
    cfg: RunConfig | None = None,
    policy_name: str = "bandit",
    explore: bool = False,
    update: bool = True,
    trace_log=None,
    difficulty_fn=None,
    strong_executor: Executor | None = None,
) -> list[TaskTrace]:
    """Run every task once; optionally append each trace to a TraceLog."""
    traces: list[TaskTrace] = []
    for task in tasks:
        tr = run_task(task, allocator, executor, verifier, terminal,
                      planner=planner, cfg=cfg, policy_name=policy_name,
                      explore=explore, update=update, difficulty_fn=difficulty_fn,
                      strong_executor=strong_executor)
        traces.append(tr)
        if trace_log is not None:
            trace_log.append(tr)
    return traces
