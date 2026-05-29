"""Run the self-improvement loop against OpenRouter on the local arithmetic suite.

Costs real credits (one Executor run per task per round). Start small. Usage:

    python scripts/run_selfimprove.py --learner bc --rounds 3
    python scripts/run_selfimprove.py --learner dpo --rounds 3 --budget 0.15

Prints the per-round self-improvement curve and writes traces to runs/traces.jsonl.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", choices=["bc", "dpo"], default="bc")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--currency", choices=["tokens", "latency", "dollars"], default="dollars")
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--max-decisions", type=int, default=4)
    args = ap.parse_args()

    require_openrouter_key()
    cfg = load_config()
    models = cfg["models"]
    bench = ArithmeticBenchmark()
    tasks = bench.tasks()
    train, eval_ = split(tasks, frac_train=0.6, seed=0)

    run_cfg = RunConfig(currency=args.currency, budget=args.budget,
                        max_decisions=args.max_decisions)
    trace_log = TraceLog(Path(cfg["paths"]["traces"]) / "traces.jsonl")

    with OpenRouterClient() as client:
        executor = Executor(client, models["cheap"], tools=bench.tools(),
                            max_steps=cfg["executor"]["max_steps"],
                            temperature=cfg["executor"]["temperature"])
        planner = Planner(client, models["cheap"])
        verifier = LLMProcessVerifier(client, models["scorer"])
        terminal = bench.terminal_verifier()

        reports = self_improve(
            train, eval_, executor, verifier, terminal,
            planner=planner, learner=args.learner, rounds=args.rounds,
            cfg=run_cfg, keep_fraction=cfg["loop"]["bc_keep_fraction"],
            seed=0, trace_log=trace_log,
        )

    print(f"\nself-improvement curve ({args.learner}, currency={args.currency}):\n")
    print(format_curve(reports))


if __name__ == "__main__":
    main()
