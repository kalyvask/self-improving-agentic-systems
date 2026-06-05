# Positioning

## What this is, stated plainly

- **A controller over a fixed action set, not a self-rewriting agent.** The Allocator makes one discrete choice per decision node from {WIDER, DEEPER, DECOMPOSE, ESCALATE, STOP}. It does not modify the Executor, the prompts, or its own code.
- **A contextual-bandit / offline-RL reduction, not deep sequential RL with a learned critic.** Each decision is treated as a contextual choice over cheap `NodeFeatures`. BC clones the good decisions; DPO/KTO learn from realized value-per-cost preferences; GRPO's group-relative advantage replaces a learned value function (there is no critic network). The policy is a small linear-softmax over 12 features, CPU-trainable in seconds.
- **The contribution is the composition, not the algorithms.** Behavior cloning, DPO, KTO, GRPO, LinUCB, and Thompson sampling are all textbook. What is new is putting a *learned, cost-aware policy at the compute-allocation layer* of a tool-using agent (WIDER vs DEEPER vs DECOMPOSE vs ESCALATE vs STOP, per step), judged on **cost-per-solved-task**, and improving it from the agent's own logged traces.

## Where it sits relative to prior work

| | per-step compute allocation | cost in the objective | learns from own traces | off-the-shelf model (no retrain) |
|---|:---:|:---:|:---:|:---:|
| **This controller** | yes | yes | yes | yes |
| Compute-optimal inference (Snell et al. 2024) | yes (parallel/sequential split) | partial (compute-aware, fixed rule) | no (analysis, not a learned policy) | yes |
| Thompson-sampling tree search (AB-MCTS) | yes (wider/deeper) | no (samples, not cost) | no (search, not a trained policy) | yes |
| Process-reward / verifier-guided search | partial (selection, not allocation) | no | partial | yes |
| Inference-time model routing | no (per-query model pick) | partial (accuracy/$) | partial | yes |

The closest neighbors are **compute-optimal inference** (Snell et al., arXiv:2408.03314 — which shows the optimal parallel-vs-sequential split *flips with difficulty*, the result the `difficulty` feature conditions on) and **Thompson-sampling tree search** (AB-MCTS, generalized here by the v0 `BanditAllocator` with added DECOMPOSE / ESCALATE / STOP arms and value-per-cost scoring). The distinguishing claims are: (a) the decision is a *learned* policy, not a fixed rule or a search procedure; (b) it optimizes *cost-per-solved*, not accuracy alone; (c) it improves from the agent's own traces and persists across runs.

## What this is NOT

- **Not a claim that one learner is best.** The honest finding is objective *robustness* and the credit/normalization bugs that break it (see the README results), not a single winning algorithm.
- **Not a general accuracy knob.** Better allocation of one model cannot exceed that model's capability ceiling; that is exactly what the ESCALATE cascade is for.
- **Not a powered claim on tau-bench.** The powered comparison is the calibrated arithmetic suite; tau-bench is the realism / transfer check.

## The baselines the controller is measured against

A learned policy is only interesting if it beats what you can do without learning. `scripts/fixed_baselines.py` runs the comparison and reports whether the controller *separates* from the best of them.

- **Fixed-action policies** (`ConstantAllocator`): always-WIDER / DEEPER / DECOMPOSE / ESCALATE / STOP. The strongest single fixed action is the bar to clear (Wilson CIs + McNemar + paired bootstrap cost-delta; honest null printed when it ties).
- **Non-contextual online bandit** (`BanditAllocator`, v0): Thompson over per-action value-per-cost, ignoring features.
- **Contextual online bandit** (`LinUCBAllocator`): the same 12 features and actions as the trained policies, learned online with no offline step. If offline BC -> DPO -> GRPO does not beat this, the offline machinery has not earned its cost.
