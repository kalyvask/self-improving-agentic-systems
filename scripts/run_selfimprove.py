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
    # --cheap-model overrides the executor+planner model WITHOUT editing the config,
    # so capability-gap probes (weak cheap model -> strong ESCALATE target) can't
    # silently contaminate the committed config/default.yaml. Falls back to cheap.
    exec_model = args.cheap_model or models["cheap"]
    # --no-calc withholds the calculator tool for the WHOLE suite, but the atomic/multi
    # prompts say "Use the calc tool" -- mixing them produces incoherent tasks. Guard:
    # no-calc runs must be hard-tier-only.
    if args.no_calc and (args.atomic or args.multi):
        raise SystemExit("--no-calc requires --atomic 0 --multi 0 (the atomic/multi prompts "
                         "reference the calc tool; only the --hard tier is calculator-free).")
    bench = ArithmeticBenchmark(n_atomic=args.atomic, n_multi=args.multi,
                                n_underspecified=args.underspecified,
                                n_hard=args.hard, no_calc=args.no_calc, seed=args.seed)
    tasks = bench.tasks()
    train, eval_ = split(tasks, frac_train=args.frac_train, seed=args.seed)
    executor = Executor(client, exec_model, tools=bench.tools(),
                        max_steps=cfg["executor"]["max_steps"],
                        temperature=cfg["executor"]["temperature"])
    planner = Planner(client, exec_model)
    print(f"arithmetic executor model: {exec_model}")
    return bench, train, eval_, executor, planner


def _build_taubench(args, cfg, client):
    """tau-bench retail/airline: multi-turn, env-graded, real headroom.

    The agent's own model calls go through the wdp OpenRouter client (cost ledger);
    the LLM user simulator runs on tau-bench's litellm path against OpenRouter (the
    key is already in os.environ via require_openrouter_key). DECOMPOSE does not
    apply (sub-tasks have no env), so planner is None."""
    from wdp.benchmarks import TauBenchBenchmark, TauReActExecutor

    models = cfg["models"]
    # Guard #2: tau's default agent model is models["executor"] (Opus in config), so a
    # cascade run without an explicit, DIFFERENT --agent-model would silently be
    # strong->strong (or strong->weaker). Require both to be set and distinct.
    if args.escalate_model and (not args.agent_model or args.agent_model == args.escalate_model):
        raise SystemExit("tau cascade requires an explicit --agent-model (the cheap model) "
                         "that DIFFERS from --escalate-model; otherwise the 'cheap' model "
                         "defaults to the config executor (Opus) and the cascade is meaningless.")
    # Guard #5: tau tasks are all solvable, so TauTerminalVerifier.score_abstention is
    # always 0 -- a forced STOP can only buy cost by giving up. Warn against it on tau.
    if args.stop_after_failed_attempts:
        print("WARNING: --stop-after-failed-attempts on tau-bench abstains on solvable tasks "
              "(abstention reward is always 0 here); it can lower cost only by losing solves. "
              "Recommend 0 for tau.")
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
    ap.add_argument("--stop-after-failed-attempts", type=int, default=0,
                    help="abstain (STOP) on a non-decomposable task after N attempts with no "
                         "progress; gives the STOP arm data and saves budget (0 = off)")
    ap.add_argument("--overwrite", action="store_true",
                    help="truncate pre-existing trace outputs instead of refusing (append "
                         "to existing files is a contamination footgun)")
    # Arithmetic-suite knobs.
    ap.add_argument("--atomic", type=int, default=8)
    ap.add_argument("--multi", type=int, default=6)
    ap.add_argument("--underspecified", type=int, default=2)
    ap.add_argument("--hard", type=int, default=0,
                    help="number of hard multi-hop word problems (capability-limited; "
                         "non-decomposable, for the ESCALATE regime)")
    ap.add_argument("--no-calc", action="store_true",
                    help="withhold the calculator tool and use the small-number/long-chain "
                         "hard variant (controlled capability gap for the ESCALATE test bed)")
    ap.add_argument("--cheap-model", default=None,
                    help="override the arithmetic executor+planner model (OpenRouter slug) "
                         "without editing config; used for weak-cheap-model capability probes")
    ap.add_argument("--escalate-model", default=None,
                    help="enable the ESCALATE action by wiring a stronger executor model "
                         "(OpenRouter slug, e.g. anthropic/claude-haiku-4-5); billed at its "
                         "real price into the same ledger")
    ap.add_argument("--escalate-after", type=int, default=1,
                    help="cheap attempts required before ESCALATE is allowed (1=after one miss; "
                         "2=force a cheap retry first, cutting over-escalation)")
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
                        abstention_credit=args.abstention_credit,
                        stop_after_failed_attempts=args.stop_after_failed_attempts,
                        escalate_after=args.escalate_after)
    out = args.out or str(Path(cfg["paths"]["traces"]) / "traces.jsonl")
    # Append-only TraceLog is a contamination footgun: a re-run silently appends to
    # (or interleaves with) an existing file. Refuse pre-existing outputs unless
    # --overwrite, which truncates the train and eval logs up front.
    eval_out = out.replace(".jsonl", "") + "_eval.jsonl"
    existing = [p for p in (out, eval_out) if Path(p).exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"refusing to append to existing {existing}; pass --overwrite "
                         f"to truncate, or choose a fresh --out")
    for p in (out, eval_out):
        Path(p).unlink(missing_ok=True)
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

        # ESCALATE target: a stronger (pricier) executor the controller can hand a step
        # to, billed into the same ledger at its real price. It must be the SAME executor
        # CLASS as the cheap one (only the model differs) -- a generic Executor for
        # arithmetic, a TauReActExecutor (env + user simulator) for tau-bench. Absent =>
        # ESCALATE is masked in the runner and the loop is the usual 4-action one.
        strong_executor = None
        if args.escalate_model:
            if args.benchmark == "taubench":
                from wdp.benchmarks import TauReActExecutor
                strong_executor = TauReActExecutor(
                    client=client, model=args.escalate_model,
                    env_name=args.env, split=args.tb_split,
                    user_model=args.user_model, user_provider=args.user_provider,
                    max_steps=cfg["executor"]["max_steps"],
                    temperature=cfg["executor"]["temperature"])
                print("NOTE: on tau-bench, ESCALATE is a TRUE HANDOFF when the cheap attempt is "
                      "unfinished (the strong model resumes the same live env/conversation); it "
                      "falls back to a fresh strong attempt only when there is no live env to "
                      "resume. User-simulator cost runs through tau/litellm and is NOT in the "
                      "ledger, so reported cost is AGENT cost only.")
            else:
                strong_executor = Executor(client, args.escalate_model, tools=bench.tools(),
                                           max_steps=cfg["executor"]["max_steps"],
                                           temperature=cfg["executor"]["temperature"])
            print(f"ESCALATE target model: {args.escalate_model}")

        eval_trace_log = TraceLog(eval_out)   # eval_out defined above with the overwrite guard
        reports = self_improve(
            train, eval_, executor, verifier, terminal,
            planner=planner, learner=args.learner, rounds=args.rounds,
            cfg=run_cfg, keep_fraction=cfg["loop"]["bc_keep_fraction"],
            seed=args.seed, fit_window=args.fit_window, trace_log=trace_log,
            eval_trace_log=eval_trace_log,
            difficulty_fn=difficulty_fn,
            strong_executor=strong_executor,
        )

    print(f"\nself-improvement curve ({args.benchmark}/{args.learner}, "
          f"currency={args.currency}):\n")
    print(format_curve(reports))


if __name__ == "__main__":
    main()
