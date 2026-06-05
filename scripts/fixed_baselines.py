"""Honest baselines: does the learned controller beat the best FIXED action?

The credibility question for a learned allocator is not "is it better than nothing"
but "is it better than the best you could do WITHOUT learning at all." This script
runs, on a fresh held-out split, the fixed-policy baselines (always-WIDER /
always-DEEPER / always-STOP -- and DECOMPOSE/ESCALATE when a planner / strong
executor is wired), the v0 non-contextual bandit, the online contextual bandit
(LinUCB), and a learned policy (BC/DPO). For each it reports solve rate with a
Wilson 95% CI and cost-per-solved, then tests whether the learned controller's
outcome SEPARATES from the best fixed action.

If it does not -- if "always-DEEPER" ties the controller within CIs -- that is the
honest headline, and this script surfaces it: McNemar on paired solves plus a
percentile-bootstrap CI on the paired cost difference (the lower-variance metric a
small eval can actually resolve).

Like eval_ab.py this is a SINGLE-model harness (no --escalate-model cascade path).
Training data is reused from --traces; the only credits spent are the eval
conversations (one per task per policy). Usage:

    python scripts/fixed_baselines.py --traces traces/taubench_dpo.jsonl \
        --env retail --eval-start 12 --n-eval 15 --budget 0.20 --learner dpo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wdp.config import load_config, require_openrouter_key
from wdp.llm.openrouter import OpenRouterClient
from wdp.verifier.scorer import LLMProcessVerifier
from wdp.allocator import (
    Action,
    BanditAllocator,
    BCAllocator,
    ConstantAllocator,
    DPOAllocator,
    LinUCBAllocator,
)
from wdp.benchmarks import TauBenchBenchmark, TauReActExecutor
from wdp.loop import RunConfig, TraceLog
from wdp.loop.runner import run_task, _maybe_update
from wdp.metrics.reliability import mcnemar, paired_diff_ci, wilson_ci


def _replay_bandit(traces, seed):
    """Rebuild the deployed v0 bandit by replaying its logged online updates
    (mirrors eval_ab.py so the two scripts compare the SAME round-0 policy)."""
    b = BanditAllocator(seed=seed)
    for tr in traces:
        if tr.policy != "bandit":
            continue
        for d in tr.decisions:
            if d.action == Action.STOP.value:
                continue
            _maybe_update(b, Action(d.action), d.process_score_after,
                          d.cost_before, d.cost_after)
    return b


def _cost(trace, currency: str) -> float:
    return float((trace.total_cost or {}).get(currency, 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="trace file from a prior run")
    ap.add_argument("--env", default="retail")
    ap.add_argument("--tb-split", default="test")
    ap.add_argument("--eval-start", type=int, default=12,
                    help="first task index of the fresh eval split (past training indices)")
    ap.add_argument("--n-eval", type=int, default=15)
    ap.add_argument("--currency", choices=["tokens", "latency", "dollars"], default="dollars")
    ap.add_argument("--budget", type=float, default=0.20)
    ap.add_argument("--max-decisions", type=int, default=4)
    ap.add_argument("--learner", choices=["bc", "dpo"], default="dpo",
                    help="the learned controller to put on trial against the fixed baselines")
    ap.add_argument("--agent-model", default=None)
    ap.add_argument("--user-model", default="openai/gpt-4o-mini")
    ap.add_argument("--user-provider", default="openrouter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="traces/fixed_baselines.jsonl")
    args = ap.parse_args()

    require_openrouter_key()
    cfg = load_config()
    models = cfg["models"]
    run_cfg = RunConfig(currency=args.currency, budget=args.budget,
                        max_decisions=args.max_decisions)

    accumulated = TraceLog(args.traces).read()
    if not accumulated:
        raise SystemExit(f"no traces in {args.traces!r}; run the sweep first")

    # Learned controller (fit on all accumulated traces) + the two cheap online
    # learners, warm-started/replayed from the SAME traces so the comparison is
    # apples-to-apples. Fixed-action constants need no data. Single-model harness:
    # the runnable fixed baselines are WIDER / DEEPER / STOP (DECOMPOSE needs a
    # planner, ESCALATE a strong executor -- masked here, so we omit them rather
    # than log a constant that silently falls back to WIDER).
    learned_cls = DPOAllocator if args.learner == "dpo" else BCAllocator
    learned = learned_cls(keep_fraction=cfg["loop"]["bc_keep_fraction"], seed=args.seed)
    learned.fit(accumulated)
    linucb = LinUCBAllocator(seed=args.seed)
    linucb.fit(accumulated)
    bandit = _replay_bandit(accumulated, args.seed)

    policies: dict[str, object] = {
        "always-wider": ConstantAllocator(Action.WIDER),
        "always-deeper": ConstantAllocator(Action.DEEPER),
        "always-stop": ConstantAllocator(Action.STOP),
        "bandit": bandit,
        "linucb": linucb,
        args.learner: learned,
    }

    indices = list(range(args.eval_start, args.eval_start + args.n_eval))
    bench = TauBenchBenchmark(env_name=args.env, split=args.tb_split, task_indices=indices)
    eval_tasks = bench.tasks()
    terminal = bench.terminal_verifier()

    with OpenRouterClient() as client:
        executor = TauReActExecutor(
            client=client, model=args.agent_model or models["executor"],
            env_name=args.env, split=args.tb_split,
            user_model=args.user_model, user_provider=args.user_provider,
            max_steps=cfg["executor"]["max_steps"], temperature=cfg["executor"]["temperature"],
        )
        verifier = LLMProcessVerifier(client, models["scorer"])
        trace_log = TraceLog(args.out)

        print(f"fixed-baseline eval on {args.env}/{args.tb_split} | "
              f"{len(eval_tasks)} held-out tasks (idx {indices[0]}..{indices[-1]})")
        print(f"fit/replay from {len(accumulated)} accumulated traces\n")

        # Interleave per task (all policies on the same task) so a mid-run credit
        # cutoff still leaves a set of PAIRED tasks every policy attempted.
        results: dict[str, list] = {name: [] for name in policies}
        for task in eval_tasks:
            staged = {}
            try:
                for name, alloc in policies.items():
                    staged[name] = run_task(
                        task, alloc, executor, verifier, terminal, planner=None,
                        cfg=run_cfg, policy_name=name, explore=False, update=False)
                    trace_log.append(staged[name])
            except httpx.HTTPStatusError as e:
                print(f"stopped on task {task.id}: {e.response.status_code} "
                      f"(likely out of credits); reporting paired tasks collected so far\n")
                break
            for name in policies:
                results[name].append(staged[name])

    n = len(results["always-wider"])
    if n == 0:
        raise SystemExit("no paired tasks completed; top up credits and re-run")

    # Per-policy report: solve (Wilson CI), mean cost, cost-per-solved.
    print(f"paired held-out tasks: {n}  (budget {args.budget} {args.currency})\n")
    print(f"{'policy':>14} {'solve':>7} {'wilson95':>15} {'mean_cost':>11} {'cost/solved':>12}")
    rep: dict[str, dict] = {}
    for name, trs in results.items():
        k = sum(1 for t in trs if t.solved)
        ci = wilson_ci(k, n)
        costs = [_cost(t, args.currency) for t in trs]
        solved_costs = [c for t, c in zip(trs, costs) if t.solved]
        cps = (sum(solved_costs) / len(solved_costs)) if solved_costs else float("inf")
        rep[name] = {"k": k, "ci": ci, "mean_cost": sum(costs) / n, "cps": cps, "trs": trs, "costs": costs}
        cps_s = f"{cps:.5f}" if cps != float("inf") else "  inf"
        print(f"{name:>14} {k:>3}/{n:<3} [{ci.lo:.2f}, {ci.hi:.2f}] "
              f"{rep[name]['mean_cost']:>11.5f} {cps_s:>12}")

    # The honest test: does the learned controller separate from the BEST fixed action?
    fixed = [k for k in rep if k.startswith("always-")]
    best_fixed = max(fixed, key=lambda k: rep[k]["k"])
    ctrl = args.learner
    pairs = [(rep[best_fixed]["trs"][i].solved, rep[ctrl]["trs"][i].solved) for i in range(n)]
    mc = mcnemar(pairs)
    cost_deltas = [rep[ctrl]["costs"][i] - rep[best_fixed]["costs"][i] for i in range(n)]
    dci = paired_diff_ci(cost_deltas)

    print(f"\nstrongest fixed action: {best_fixed} ({rep[best_fixed]['k']}/{n})")
    print(f"solve: {ctrl} {rep[ctrl]['k']}/{n} vs {best_fixed} {rep[best_fixed]['k']}/{n}  "
          f"(McNemar p={mc['p_value']:.3f}, discordant={mc['discordant']})")
    print(f"paired cost delta ({ctrl} - {best_fixed}): {dci}  (negative => controller cheaper)")
    separates = (rep[ctrl]["ci"].lo > rep[best_fixed]["ci"].hi) or (dci.hi < 0) or (mc["p_value"] < 0.05)
    if separates:
        print(f"\n=> the controller SEPARATES from the best fixed action on this suite (n={n}).")
    else:
        print(f"\n=> the controller does NOT separate from '{best_fixed}' on this suite "
              f"(n={n}). Honest null: at this n the learned policy is a tie with the best "
              f"fixed action -- report it, and note the eval is likely under-powered (see "
              f"metrics.reliability.tasks_needed).")


if __name__ == "__main__":
    main()
