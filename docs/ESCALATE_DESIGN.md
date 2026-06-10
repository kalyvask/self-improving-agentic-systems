<!-- Design note for the ESCALATE capstone (task #54). Offline reasoning only; no
     live spend was incurred writing this. -->

# ESCALATE capstone: offline-Opus-labeling design

## The problem this solves

ESCALATE is a 5th action: re-run the current step on a stronger model and pay its
higher per-token cost. It only earns its keep on tasks the base model **cannot**
solve at any allocation. The oracle-rescue diagnostic (`analyze_eval.py --oracle`)
shows the current benchmarks have **capability_ceiling = 0** -- arithmetic (pooled
over 10 Haiku policy-rounds: 1 solvable miss, recoverable) and tau-bench (7 tasks,
all recoverable). So two things are true:

1. We need a **harder benchmark tier** where Haiku genuinely fails ~20-40%, or
   ESCALATE has nothing to demonstrate.
2. We want to know, **per task and offline**, whether a miss is capability-limited
   (a stronger model would solve it) before paying for ESCALATE in the live loop.

The instinct "use an Opus LLM-judge before escalating" is right in spirit but wrong
in placement: an Opus judge *in the live loop* is pure cost where ground truth is
free (arithmetic exact grade, tau-bench env grade), and it inverts cascade economics
(judge tier > execution tier). The correct form is **Opus as an offline labeler**,
never in the live cost path.

## Principle: keep Opus out of the live ledger

- **Live inference cost** = what the deployed policy spends per task. Opus must NOT
  appear here unless it is the chosen ESCALATE *target* actually executing a step.
- **Training/labeling cost** = a one-time offline spend to produce supervision. It is
  reported separately and amortized, exactly like `rollout.py`'s `labeling_ledger`
  for difficulty labels. It is "self-improvement training spend," not inference.
- **Eval ground truth stays env/exact-graded.** Opus labels are *training signal*,
  never the definition of "solved." Do not let the labeler grade the benchmark.

## Pipeline

### Step 0 -- a benchmark with a real ceiling (prerequisite)
Add a hard tier so `capability_ceiling > 0`. Cheapest: extend `arithmetic.py` with
larger operands / deeper nesting / more parts sized so Haiku solves ~60-80%. More
compelling but pricier: a harder tau-bench split. Acceptance gate: a small Haiku-only
calibration run whose pooled oracle shows capability_ceiling in the 20-40% band.

### Step 1 -- collect cheap attempts (already have the machinery)
Run the existing Haiku policies on the hard tier, log traces as today. Use
`--oracle` to split misses into premature / recoverable / **capability_ceiling**.
Only the capability_ceiling set is a candidate for ESCALATE labeling.

### Step 2 -- offline Opus labeling (the one-time spend)
For each capability-ceiling task, run the stronger model ONCE offline:
```
escalation_value(task) = solved_by_strong(task) AND NOT solved_by_cheap(task)
```
Bill every strong-model call into a dedicated `escalation_label_ledger` (mirror
`RolloutProcessVerifier.labeling_ledger`). Persist `{task_id, solved_by_cheap,
solved_by_strong, strong_cost, escalation_value}`. This is the supervision set.

### Step 3 -- train the 5-action policy on cheap features
Add ESCALATE to `Action` + the `runner.py` branch (execute step on the strong model,
bill real cost) + the linear policy. Credit ESCALATE with the SAME
`value_per_cost = terminal_reward * cost_efficiency * advantage` rule, where the cost
is the realized strong-model cost -- so a successful-but-expensive escalation trains
as better than a wasted cheap loop but worse than a cheap solve. The policy learns
`p(escalate | cheap features)`: it never sees Opus at decision time, only structural
features (score_max, depth, decomposability, stall, prior failures).

### Step 4 -- validate the cheap escalation signal (alt-test, no new spend)
Run the existing alt-test: does the policy's escalate score (or any cheap verifier
feature) predict `escalation_value` better than always-guess-majority? If it fails
the alt-test, the cheap signal is not trustworthy and ESCALATE should stay
conservative (escalate rarely / only on the strongest capability evidence). This is
the same bar the process verifier had to clear, applied to the new decision.

### Step 5 -- live eval and the win condition
Run the learned 5-action policy live on the hard tier. Win condition:
**solve rises above the single-model (Haiku) ceiling at a cost the policy chose to
pay only on capability-limited tasks.** Compare cost-per-solved-task against:
  (i)  always-Haiku (no escalation)         -- lower solve, lower cost
  (ii) always-escalate (Opus on everything) -- the solve ceiling, high cost
  (iii) feature-only router WITHOUT Opus labels -- the baseline ESCALATE must beat
The claim is paired-cost + solve A/B, read the same way as calib4 (cost is the metric
with power at small n; solve read with its Wilson interval).

## What would make this INVALID (guardrails)
- Opus appearing in the live cost ledger when it is not executing the chosen step.
- Opus defining "solved" for the eval (must stay env/exact-graded).
- Skipping the alt-test and trusting the cheap escalate signal blindly.
- Reporting inference cost with the labeling spend folded in (must be separated).
- Declaring an ESCALATE win without beating the feature-only router (iii).

## Reuse map (existing code)
- `verifier/rollout.py` -- `labeling_ledger` + `difficulty()` pattern -> copy for the
  offline escalation labeler.
- `loop/runner.py` -- action branches; add ESCALATE next to WIDER/DEEPER/DECOMPOSE.
- `loop/trace.py` -- `assign_credit`; ESCALATE uses the existing value_per_cost rule.
- `metrics/alt_test.py` -- validate the cheap escalate signal vs Opus labels.
- `scripts/analyze_eval.py --oracle` -- gates Step 0 (is there a ceiling?) and reads
  the live result (did escalation move capability misses?).
