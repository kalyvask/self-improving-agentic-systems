"""wdp-controller: a self-improving controller for cost-optimal agent compute.

Layers:
  - executor: ReAct tool-using base agent (the unit of work).
  - planner:  decompose a task into a sub-task DAG.
  - verifier: terminal env reward + process scorer.
  - allocator: the controller that picks {wider, deeper, decompose, stop}.
  - loop:     trace logging, credit assignment, self-improvement rounds (BC -> DPO).
  - cost:     token / latency / dollar accounting.
  - metrics:  success@budget, pass^k, risk-coverage, CVaR, gen-verification gap, METR horizon.
"""

__version__ = "0.1.0"

# Public embed seam: load a trained policy, hand it your agent's NodeFeatures, get an
# allocation decision back. Everything else (executor, runner, benchmarks) is optional.
from wdp.allocator.persist import FrozenLinearAllocator, load_policy, save_policy  # noqa: E402
from wdp.allocator.policy import Action, Decision, NodeFeatures  # noqa: E402

__all__ = [
    "Action",
    "Decision",
    "FrozenLinearAllocator",
    "NodeFeatures",
    "load_policy",
    "save_policy",
    "__version__",
]
