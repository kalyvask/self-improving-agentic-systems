"""Live smoke test against OpenRouter. Costs a few cents; needs a key.

Run after pasting your key into wdp-controller/.env:

    python scripts/smoke_live.py

It runs ONE trivial task through the full Allocator loop with real models, then
prints the decision trace, the per-currency cost, and a round summary. Use it to
confirm the key works and real cost accounting flows end to end before spending
real credits on a benchmark batch.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/smoke_live.py` without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wdp.config import load_config, require_openrouter_key
from wdp.llm.openrouter import OpenRouterClient
from wdp.executor.react import Executor, Task
from wdp.planner.decompose import Planner
from wdp.verifier.scorer import LLMProcessVerifier, Score
from wdp.allocator.policy import BanditAllocator
from wdp.loop.runner import RunConfig, run_task
from wdp.metrics import summarize_round


class ContainsVerifier:
    """Terminal verifier: pass if the gold substring appears in the answer."""

    def __init__(self, gold: str) -> None:
        self._gold = gold.lower()

    def score_final(self, task, answer: str) -> Score:
        return Score(value=1.0 if self._gold in (answer or "").lower() else 0.0)


def main() -> None:
    require_openrouter_key()  # fail fast with a helpful message if empty
    cfg = load_config()
    models = cfg["models"]

    with OpenRouterClient() as client:
        executor = Executor(client, models["cheap"], tools={},
                            max_steps=cfg["executor"]["max_steps"],
                            temperature=cfg["executor"]["temperature"])
        planner = Planner(client, models["cheap"])
        verifier = LLMProcessVerifier(client, models["scorer"])
        terminal = ContainsVerifier("paris")
        allocator = BanditAllocator(seed=0)

        task = Task(id="live-0",
                    prompt="What is the capital of France? Answer with one word.")
        run_cfg = RunConfig(currency="dollars", budget=0.25, max_decisions=4)
        trace = run_task(task, allocator, executor, verifier, terminal,
                         planner=planner, cfg=run_cfg, policy_name="bandit")

    print(f"\nsolved={trace.solved}  terminal_reward={trace.terminal_reward:.2f}")
    print("cost:", {k: round(v, 6) for k, v in trace.total_cost.items()})
    print("\ndecisions:")
    for d in trace.decisions:
        print(f"  step {d.step}: {d.action:9s} "
              f"cost+={d.marginal_cost:.5f} proc={d.process_score_after:.2f} "
              f"vpc={d.value_per_cost:.3f}")
    print("\nround summary:", summarize_round([trace]))


if __name__ == "__main__":
    main()
