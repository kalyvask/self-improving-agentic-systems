"""`wdp-decide`: load a frozen, trained policy and make one allocation decision.

This is the deployable unit with no training, no trace store, and no benchmark
attached: an integration computes the cheap NodeFeatures for its agent's current
state and asks the policy what to spend next.

    wdp-decide --policy artifacts/policies/sql_dpo_policy.json \
        --score-max 0.0 --n-children 0 --difficulty 0.7
"""
from __future__ import annotations

import argparse

from wdp.allocator.persist import load_policy
from wdp.allocator.policy import NodeFeatures


def main() -> None:
    ap = argparse.ArgumentParser(prog="wdp-decide", description=__doc__)
    ap.add_argument("--policy", required=True, help="JSON written by --save-policy")
    ap.add_argument("--score-max", type=float, default=0.0,
                    help="best process score seen so far on this task [0,1]")
    ap.add_argument("--n-children", type=int, default=0,
                    help="attempts already spawned on this task")
    ap.add_argument("--difficulty", type=float, default=0.5,
                    help="task difficulty estimate [0,1]")
    ap.add_argument("--budget-remaining", type=float, default=1.0,
                    help="fraction of the per-task budget still unspent [0,1]")
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
