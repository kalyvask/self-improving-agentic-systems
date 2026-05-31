"""Frozen-policy eval A/B: round-0 bandit vs the fitted DPO policy.

SCOPE / WARNING: this is a pre-ESCALATE A/B harness. It builds a SINGLE
TauReActExecutor and has NO --escalate-model path, so it CANNOT evaluate the
weak->strong cascade -- do not use it for any cascade cost/solve claim. For
cascades, run the arms via `run_selfimprove.py --escalate-model ...` and compare
the eval files with `analyze_eval.py --ab2 FILE_A FILE_B` (cross-file paired,
reports mean and p95). This script remains valid only for the 2-policy
(bandit vs learner) single-model comparison it was written for.

The per-round self-improvement curve is starved for resolution on a tiny eval
split (5 tasks => 0.20 granularity). This script isolates the comparison: it
takes the traces a prior `run_selfimprove` run already collected, reconstructs
the two policies it would deploy, freezes them, and runs both on a *fresh*
held-out task split that neither policy trained on.

  - bandit:  the deployed round-0 policy, rebuilt by replaying its logged online
             updates (same Beta posteriors it had at eval time). No re-collection.
  - dpo:     DPOAllocator fit on ALL accumulated traces, exactly as round R did.

Only credits spent are the 2 x N_eval evaluation conversations -- training data
is reused. Usage:

    python scripts/eval_ab.py --traces traces/taubench_dpo.jsonl \
        --env retail --eval-start 12 --n-eval 15 --budget 0.20
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
from wdp.allocator.policy import BanditAllocator, Action
from wdp.allocator.dpo import DPOAllocator
from wdp.benchmarks import TauBenchBenchmark, TauReActExecutor
from wdp.loop import RunConfig, TraceLog
from wdp.loop.runner import run_task, _maybe_update
from wdp.metrics import summarize_round


def _replay_bandit(traces, seed):
    """Rebuild the deployed round-0 bandit by replaying its logged updates.

    Mirrors runner._maybe_update over the bandit-policy traces so the posteriors
    match what the policy held when round 0 was evaluated. STOP is never an
    update target (the bandit only tracks spend arms), so it is skipped."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="trace file from a prior run")
    ap.add_argument("--env", default="retail")
    ap.add_argument("--tb-split", default="test")
    ap.add_argument("--eval-start", type=int, default=12,
                    help="first task index of the fresh eval split (must be past "
                         "the indices used for training)")
    ap.add_argument("--n-eval", type=int, default=15)
    ap.add_argument("--currency", choices=["tokens", "latency", "dollars"], default="dollars")
    ap.add_argument("--budget", type=float, default=0.20)
    ap.add_argument("--max-decisions", type=int, default=4)
    ap.add_argument("--agent-model", default=None)
    ap.add_argument("--user-model", default="openai/gpt-4o-mini")
    ap.add_argument("--user-provider", default="openrouter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="traces/eval_ab.jsonl",
                    help="where to log the eval traces for inspection")
    args = ap.parse_args()

    require_openrouter_key()
    cfg = load_config()
    models = cfg["models"]
    run_cfg = RunConfig(currency=args.currency, budget=args.budget,
                        max_decisions=args.max_decisions)

    accumulated = TraceLog(args.traces).read()
    if not accumulated:
        raise SystemExit(f"no traces in {args.traces!r}; run the sweep first")

    bandit = _replay_bandit(accumulated, args.seed)
    dpo = DPOAllocator(keep_fraction=cfg["loop"]["bc_keep_fraction"], seed=args.seed)
    dpo.fit(accumulated)

    indices = list(range(args.eval_start, args.eval_start + args.n_eval))
    bench = TauBenchBenchmark(env_name=args.env, split=args.tb_split,
                              task_indices=indices)
    eval_tasks = bench.tasks()
    terminal = bench.terminal_verifier()

    with OpenRouterClient() as client:
        executor = TauReActExecutor(
            client=client, model=args.agent_model or models["executor"],
            env_name=args.env, split=args.tb_split,
            user_model=args.user_model, user_provider=args.user_provider,
            max_steps=cfg["executor"]["max_steps"],
            temperature=cfg["executor"]["temperature"],
        )
        verifier = LLMProcessVerifier(client, models["scorer"])
        trace_log = TraceLog(args.out)

        print(f"frozen eval A/B on {args.env}/{args.tb_split} | "
              f"{len(eval_tasks)} held-out tasks (idx {indices[0]}..{indices[-1]})")
        print(f"fit from {len(accumulated)} accumulated traces in {args.traces}\n")

        # Interleave per task (bandit then dpo on the same task) so a mid-run
        # stop -- e.g. an OpenRouter 402 when credits run out -- still leaves a
        # set of *paired* tasks both policies attempted. A task only counts once
        # both policies finished it; an unpaired tail is dropped.
        paired = {"bandit": [], "dpo": []}
        for task in eval_tasks:
            staged = {}
            try:
                for name, alloc in (("bandit", bandit), ("dpo", dpo)):
                    staged[name] = run_task(
                        task, alloc, executor, verifier, terminal, planner=None,
                        cfg=run_cfg, policy_name=name, explore=False, update=False)
                    trace_log.append(staged[name])
            except httpx.HTTPStatusError as e:
                print(f"stopped on task {task.id}: {e.response.status_code} "
                      f"(likely out of credits); reporting {len(paired['bandit'])} "
                      f"paired tasks collected before the cutoff\n")
                break
            for name in ("bandit", "dpo"):
                paired[name].append(staged[name])

    n = len(paired["bandit"])
    if n == 0:
        raise SystemExit("no paired tasks completed; top up credits and re-run")
    reps = {name: summarize_round(paired[name], run_cfg.currency)
            for name in ("bandit", "dpo")}

    print(f"paired held-out tasks: {n}/{len(eval_tasks)}")
    print(f"{'policy':>7} {'solve':>6} {'mean_cost':>10} {'p95_cost':>10} {'gap':>6}")
    for name in ("bandit", "dpo"):
        e = reps[name]
        print(f"{name:>7} {e['solve_rate']:>6.2f} {e['mean_cost']:>10.5f} "
              f"{e['p95_cost']:>10.5f} {e['gen_verif_gap']:>6.2f}")

    d_solve = reps["dpo"]["solve_rate"] - reps["bandit"]["solve_rate"]
    d_cost = reps["dpo"]["mean_cost"] - reps["bandit"]["mean_cost"]
    print(f"\ndelta (dpo - bandit): solve {d_solve:+.2f}  mean_cost {d_cost:+.5f}")


if __name__ == "__main__":
    main()
