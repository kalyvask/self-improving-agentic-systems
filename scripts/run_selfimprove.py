"""Run the self-improvement loop against OpenRouter.

Costs real credits (one Executor run per task per round). Start small. Usage:

    python scripts/run_selfimprove.py --learner bc --rounds 3
    python scripts/run_selfimprove.py --learner dpo --rounds 3 --budget 0.15
    python scripts/run_selfimprove.py --benchmark taubench --env retail \
        --n-tasks 8 --rounds 2 --budget 0.50

Prints the per-round self-improvement curve and writes traces to traces/traces.jsonl.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wdp.config import load_config, require_openrouter_key
from wdp.llm.openrouter import OpenRouterClient
from wdp.executor.react import Executor
from wdp.planner.decompose import Planner
from wdp.verifier.scorer import LLMProcessVerifier
from wdp.allocator import BanditAllocator  # noqa: F401  (referenced in docs)
from wdp.benchmarks import ArithmeticBenchmark, split
from wdp.loop import RunConfig, TraceLog, self_improve, format_curve


def _build_arithmetic(args, cfg, client):
    """Local arithmetic suite: cheap, fully offline-gradable, DECOMPOSE applies."""
    models = cfg["models"]
    bench = ArithmeticBenchmark(n_atomic=args.atomic, n_multi=args.multi,
                                n_underspecified=args.underspecified, seed=args.seed)
    tasks = bench.tasks()
    train, eval_ = split(tasks, frac_train=args.frac_train, seed=args.seed)
    executor = Executor(client, models["cheap"], tools=bench.tools(),
                        max_steps=cfg["executor"]["max_steps"],
                        temperature=cfg["executor"]["temperature"])
    planner = Planner(client, models["cheap"])
    return bench, train, eval_, executor, planner


def _build_taubench(args, cfg, client):
    """tau-bench retail/airline: multi-turn, env-graded, real headroom.

    The agent's own model calls go through the wdp OpenRouter client (cost ledger);
    the LLM user simulator runs on tau-bench's litellm path against OpenRouter (the
    key is already in os.environ via require_openrouter_key). DECOMPOSE does not
    apply (sub-tasks have no env), so planner is None."""
    from wdp.benchmarks import TauBenchBenchmark, TauReActExecutor

    models = cfg["models"]
    bench = TauBenchBenchmark(env_name=args.env, split=args.tb_split,
                              task_indices=list(range(args.n_tasks)))
    tasks = bench.tasks()
    train, eval_ = split(tasks, frac_train=args.frac_train, seed=args.seed)
    executor = TauReActExecutor(
        client=client, model=args.agent_model or models["executor"],
        env_name=args.env, split=args.tb_split,
        user_model=args.user_model, user_provider=args.user_provider,
        max_steps=cfg["executor"]["max_steps"],
        temperature=cfg["executor"]["temperature"],
    )
    return bench, train, eval_, executor, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["arithmetic", "taubench"], default="arithmetic")
    ap.add_argument("--learner", choices=["bc", "dpo", "kto"], default="bc")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--currency", choices=["tokens", "latency", "dollars"], default="dollars")
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--max-decisions", type=int, default=4)
    ap.add_argument("--cost-weight", type=float, default=0.5,
                    help="credit cost-efficiency steepness exp(-cost_weight*spent/budget)")
    ap.add_argument("--abstention-credit", type=float, default=0.5,
                    help="scale on correct-STOP credit; keep below the solve scale")
    ap.add_argument("--rollout-difficulty", action="store_true",
                    help="estimate difficulty from forked rollout success (Math-Shepherd); "
                         "spends extra forked-attempt credits, cached per task")
    ap.add_argument("--diff-rollouts", type=int, default=4,
                    help="number of forked attempts per task for rollout difficulty")
    # Arithmetic-suite knobs.
    ap.add_argument("--atomic", type=int, default=8)
    ap.add_argument("--multi", type=int, default=6)
    ap.add_argument("--underspecified", type=int, default=2)
    # tau-bench knobs.
    ap.add_argument("--env", default="retail", help="tau-bench domain: retail | airline")
    ap.add_argument("--tb-split", default="test", help="tau-bench task split: train | dev | test")
    ap.add_argument("--n-tasks", type=int, default=8, help="number of tau-bench task indices")
    ap.add_argument("--agent-model", default=None,
                    help="model the tau-bench agent uses (default: config models.executor)")
    ap.add_argument("--user-model", default="openai/gpt-4o-mini",
                    help="model for tau-bench's LLM user simulator (via OpenRouter)")
    ap.add_argument("--user-provider", default="openrouter")
    ap.add_argument("--frac-train", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the benchmark split, learner init, and exploration")
    ap.add_argument("--fit-window", type=int, default=None,
                    help="fit each round on traces from the last N rounds only "
                         "(default: all accumulated traces)")
    ap.add_argument("--out", default=None,
                    help="trace file path (default traces/traces.jsonl); set per "
                         "run to avoid collisions when running learners concurrently")
    args = ap.parse_args()

    require_openrouter_key()
    cfg = load_config()

    run_cfg = RunConfig(currency=args.currency, budget=args.budget,
                        max_decisions=args.max_decisions,
                        cost_weight=args.cost_weight,
                        abstention_credit=args.abstention_credit)
    out = args.out or str(Path(cfg["paths"]["traces"]) / "traces.jsonl")
    trace_log = TraceLog(out)

    with OpenRouterClient() as client:
        if args.benchmark == "taubench":
            bench, train, eval_, executor, planner = _build_taubench(args, cfg, client)
        else:
            bench, train, eval_, executor, planner = _build_arithmetic(args, cfg, client)

        tasks = bench.tasks()
        print(f"benchmark: {args.benchmark} | {len(tasks)} tasks -> "
              f"{len(train)} train, {len(eval_)} eval")

        verifier = LLMProcessVerifier(client, cfg["models"]["scorer"])
        terminal = bench.terminal_verifier()

        # Optional Math-Shepherd rollout-grounded difficulty (spends forked-attempt
        # credits; cached per task). Replaces the noisy 1-first_process_score proxy.
        difficulty_fn = None
        if args.rollout_difficulty:
            from wdp.verifier.rollout import RolloutProcessVerifier
            difficulty_fn = RolloutProcessVerifier(
                executor, terminal, n_rollouts=args.diff_rollouts).difficulty

        eval_out = out.replace(".jsonl", "") + "_eval.jsonl"
        eval_trace_log = TraceLog(eval_out)
        reports = self_improve(
            train, eval_, executor, verifier, terminal,
            planner=planner, learner=args.learner, rounds=args.rounds,
            cfg=run_cfg, keep_fraction=cfg["loop"]["bc_keep_fraction"],
            seed=args.seed, fit_window=args.fit_window, trace_log=trace_log,
            eval_trace_log=eval_trace_log,
            difficulty_fn=difficulty_fn,
        )

    print(f"\nself-improvement curve ({args.benchmark}/{args.learner}, "
          f"currency={args.currency}):\n")
    print(format_curve(reports))


if __name__ == "__main__":
    main()
