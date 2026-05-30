"""GRPO probe on the arithmetic benchmark. COSTS REAL CREDITS (on-policy rollouts).

GRPO is on-policy: it regenerates fresh rollouts every step, so it is the one
learner whose cost is the rollout count, not a fixed collected set. This runs a
small probe to see whether on-policy training climbs above the BC/DPO endpoint.

Defaults are the "tiny probe" (~$3 on arithmetic/Haiku): 20 steps x 16 prompts x
4 rollouts = 1,280 rollouts. Use --num-steps 40 --group-size 8 for the "small
probe" (~$12). GRPO on tau-bench is intentionally not exposed: ~35x the per-rollout
cost makes it impractical on an API budget.

Usage (gated; run only with explicit go-ahead):
    python scripts/run_grpo_probe.py --seed-traces traces/calib_bc.jsonl \
        --num-steps 20 --group-size 4 --budget 0.003 --out traces/grpo_probe.jsonl
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
from wdp.benchmarks import ArithmeticBenchmark, split
from wdp.loop import RunConfig, TraceLog
from wdp.loop.grpo_train import grpo_train, format_grpo_curve
from wdp.grpo.estimate import estimate_grpo, GRPOConfig, format_estimate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-traces", required=True,
                    help="trace file to fit the BC warm start (no extra spend)")
    ap.add_argument("--atomic", type=int, default=60)
    ap.add_argument("--multi", type=int, default=40)
    ap.add_argument("--underspecified", type=int, default=10)
    ap.add_argument("--frac-train", type=float, default=0.6)
    ap.add_argument("--budget", type=float, default=0.003)
    ap.add_argument("--max-decisions", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--inner-epochs", type=int, default=5)
    ap.add_argument("--beta-kl", type=float, default=0.05)
    ap.add_argument("--cost-weight", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="traces/grpo_probe.jsonl")
    args = ap.parse_args()

    require_openrouter_key()
    cfg = load_config()
    models = cfg["models"]

    bench = ArithmeticBenchmark(n_atomic=args.atomic, n_multi=args.multi,
                                n_underspecified=args.underspecified, seed=args.seed)
    tasks = bench.tasks()
    train, eval_ = split(tasks, frac_train=args.frac_train, seed=args.seed)
    seed_traces = TraceLog(args.seed_traces).read()
    run_cfg = RunConfig(budget=args.budget, max_decisions=args.max_decisions)

    planned = GRPOConfig(group_size=args.group_size,
                         prompts_per_step=args.prompts_per_step,
                         num_steps=args.num_steps)
    print(f"GRPO probe: {planned.total_rollouts:,} planned rollouts "
          f"({args.num_steps} steps x {args.prompts_per_step} prompts x "
          f"{args.group_size} group) | {len(train)} train, {len(eval_)} eval")
    if seed_traces:
        print(format_estimate(estimate_grpo(seed_traces, planned)))

    with OpenRouterClient() as client:
        executor = Executor(client, models["cheap"], tools=bench.tools(),
                            max_steps=cfg["executor"]["max_steps"],
                            temperature=cfg["executor"]["temperature"])
        planner = Planner(client, models["cheap"])
        verifier = LLMProcessVerifier(client, models["scorer"])
        terminal = bench.terminal_verifier()
        reports, _ = grpo_train(
            seed_traces, train, eval_, executor, verifier, terminal,
            planner=planner, cfg=run_cfg, group_size=args.group_size,
            prompts_per_step=args.prompts_per_step, num_steps=args.num_steps,
            inner_epochs=args.inner_epochs, beta_kl=args.beta_kl,
            cost_weight=args.cost_weight, eval_every=args.eval_every,
            seed=args.seed, trace_log=TraceLog(args.out),
        )

    print("\nGRPO probe curve (arithmetic, currency=dollars):\n")
    print(format_grpo_curve(reports))


if __name__ == "__main__":
    main()
