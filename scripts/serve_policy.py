"""Load a frozen, trained policy and make an allocation decision -- no training, no
trace store, no benchmark. This is the deployable unit: an integration computes the
cheap NodeFeatures for its agent's current state and calls `alloc.decide(...)`.

    python scripts/serve_policy.py --policy runs/dpo_policy.json \
        --score-max 0.0 --n-children 0 --difficulty 0.7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wdp.allocator.persist import load_policy
from wdp.allocator.policy import NodeFeatures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="JSON written by --save-policy")
    ap.add_argument("--score-max", type=float, default=0.0)
    ap.add_argument("--n-children", type=int, default=0)
    ap.add_argument("--difficulty", type=float, default=0.5)
    ap.add_argument("--budget-remaining", type=float, default=1.0)
    args = ap.parse_args()

    alloc = load_policy(args.policy)
    feats = NodeFeatures(score_max=args.score_max, n_children=args.n_children,
                         difficulty=args.difficulty, budget_remaining_frac=args.budget_remaining)
    d = alloc.decide(feats, "dollars")
    print(f"loaded {type(alloc).__name__} from {args.policy}")
    print(f"decision: {d.action.value}")
    print("scores:", {a.value: round(v, 3) for a, v in d.scores.items()})


if __name__ == "__main__":
    main()
