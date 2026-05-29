"""Estimate GRPO training cost from collected traces. Offline, no credits.

    python scripts/estimate_grpo.py [--traces runs/traces.jsonl] \
        [--group-size 8] [--prompts-per-step 16] [--num-steps 200]

Reads the traces produced by run_selfimprove.py and prints the GRPO cost report.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wdp.config import load_config
from wdp.loop.trace import TraceLog
from wdp.grpo import GRPOConfig, estimate_grpo, format_estimate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=None)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--num-steps", type=int, default=200)
    ap.add_argument("--collection-attempts", type=int, default=None,
                    help="task episodes run to collect BC/DPO data (defaults to #traces)")
    args = ap.parse_args()

    path = args.traces or str(Path(load_config()["paths"]["traces"]) / "traces.jsonl")
    traces = TraceLog(path).read()
    if not traces:
        raise SystemExit(f"no traces at {path}; run scripts/run_selfimprove.py first")

    grpo = GRPOConfig(group_size=args.group_size,
                      prompts_per_step=args.prompts_per_step,
                      num_steps=args.num_steps)
    est = estimate_grpo(traces, grpo, bc_dpo_collection_attempts=args.collection_attempts)
    print(format_estimate(est))


if __name__ == "__main__":
    main()
