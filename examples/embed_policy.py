"""Embed a trained allocation policy in YOUR OWN agent loop — no API key, no
training, no benchmark. Runs offline against the policies shipped in this repo.

    pip install -e .
    python examples/embed_policy.py

The seam is three calls: load_policy() once, then per step build a NodeFeatures
from your agent's state and ask decide(). Your loop stays yours; the policy only
answers "what should this task get next: wider / deeper / decompose / stop /
escalate".

Two policies ship in artifacts/policies/:
  - arith_dpo_policy.json  — refit offline from the calibrated arithmetic run's
    training traces (calib4, k=3). Context-sensitive: watch the action change
    with the state below.
  - sql_dpo_policy.json    — learned on the text-to-SQL benchmark. Decisive:
    it answers DECOMPOSE nearly everywhere, because "ground the query in the
    schema before writing SQL" is the lesson it learned. A one-note policy is
    not a bug; it is what the traces supported.
"""
from __future__ import annotations

from pathlib import Path

from wdp import NodeFeatures, load_policy

POLICIES = Path(__file__).resolve().parents[1] / "artifacts" / "policies"

# Three moments in a hypothetical task's life. In your integration you would
# fill these from real state: best verifier score so far, attempts spawned,
# a difficulty estimate, and how much of the per-task budget is left.
MOMENTS = [
    ("fresh task, full budget", NodeFeatures(
        score_max=0.0, n_children=0, difficulty=0.5, budget_remaining_frac=1.0)),
    ("hard + decomposable, attempts failing", NodeFeatures(
        score_max=0.0, n_children=3, difficulty=0.9, decomposability=0.8,
        budget_remaining_frac=0.6)),
    ("hard task, budget nearly spent", NodeFeatures(
        score_max=0.0, n_children=1, difficulty=0.9, budget_remaining_frac=0.05)),
]


def main() -> None:
    for name in ("arith_dpo_policy.json", "sql_dpo_policy.json"):
        alloc = load_policy(POLICIES / name)
        print(f"{name} ({type(alloc).__name__}):")
        for label, feats in MOMENTS:
            d = alloc.decide(feats, "dollars")
            top = {a.value: round(v, 2) for a, v in
                   sorted(d.scores.items(), key=lambda kv: -kv[1])[:2]}
            print(f"  {label:40s} -> {d.action.value:9s} {top}")
        print()


if __name__ == "__main__":
    main()
