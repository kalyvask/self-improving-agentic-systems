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

import math
import statistics
from dataclasses import dataclass

from wdp.allocator.policy import Action, NodeFeatures
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


def _features(
    trajectories: list[Trajectory],
    process_scores: list[float],
    *,
    ledger: CostLedger,
    cfg: RunConfig,
    decomposability: float,
) -> NodeFeatures:
    scores = process_scores or [0.0]
    spent = ledger.amount(cfg.currency)
    frac = max(0.0, 1.0 - spent / cfg.budget) if cfg.budget else 0.0
    depth = max((t.depth for t in trajectories), default=0)
    steps = sum(t.depth for t in trajectories)
    stalled = 1.0 if trajectories and all(t.stalled for t in trajectories) else 0.0
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
) -> TaskTrace:
    cfg = cfg or RunConfig()
    ledger = CostLedger()
    trace = TaskTrace(task_id=task.id, currency=cfg.currency, policy=policy_name)

    trajectories: list[Trajectory] = []
    process_scores: list[float] = []
    decomposability = planner.probe(task, parallel_group=None) if planner else 0.0

    best_answer: str | None = None
    best_terminal = 0.0

    for step in range(cfg.max_decisions):
        feats = _features(trajectories, process_scores, ledger=ledger, cfg=cfg,
                           decomposability=decomposability)
        decision = allocator.decide(feats, cfg.currency)
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

        if decision.action == Action.WIDER:
            new_traj = executor.run(task, ledger=ledger, parallel_group=pg)
            trajectories.append(new_traj)

        elif decision.action == Action.DEEPER:
            target = _deepest_unfinished(trajectories) or (
                executor.run(task, ledger=ledger, parallel_group=pg))
            if target not in trajectories:
                trajectories.append(target)
            new_traj = executor.continue_from(
                task, target, ledger=ledger, parallel_group=pg,
                extra_steps=cfg.deeper_extra_steps)

        elif decision.action == Action.DECOMPOSE and planner is not None:
            new_traj = _run_decompose(task, planner, executor, ledger, pg)
            if new_traj is not None:
                trajectories.append(new_traj)

        # Score whatever we just produced (cheap process verifier).
        ps = feats.score_max
        if new_traj is not None:
            ps = verifier.score_step(task, new_traj.transcript()).value
            process_scores.append(ps)
            # If it produced a final answer, get the ground-truth terminal score.
            if new_traj.final_answer is not None:
                tv = terminal.score_final(task, new_traj.final_answer).value
                if tv >= best_terminal:
                    best_terminal, best_answer = tv, new_traj.final_answer

        cost_after = ledger.amount(cfg.currency)
        trace.add(DecisionRecord(
            step=step, features=feats.vector().tolist(),
            action=decision.action.value,
            scores={a.value: v for a, v in decision.scores.items()},
            currency=cfg.currency, cost_before=cost_before, cost_after=cost_after,
            process_score_after=ps,
        ))

        # Bandit online update (if the policy supports it).
        _maybe_update(allocator, decision.action, ps, cost_before, cost_after)

        if best_terminal >= cfg.solved_threshold or cost_after >= cfg.budget:
            break

    trace.solved = best_terminal >= cfg.solved_threshold
    trace.terminal_reward = best_terminal
    trace.total_cost = ledger.snapshot()
    assign_credit(trace)
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
    dag = planner.decompose(task, parallel_group=parallel_group)
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
    parent.final_answer = "\n".join(answers)
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
    trace_log=None,
) -> list[TaskTrace]:
    """Run every task once; optionally append each trace to a TraceLog."""
    traces: list[TaskTrace] = []
    for task in tasks:
        tr = run_task(task, allocator, executor, verifier, terminal,
                      planner=planner, cfg=cfg, policy_name=policy_name)
        traces.append(tr)
        if trace_log is not None:
            trace_log.append(tr)
    return traces
