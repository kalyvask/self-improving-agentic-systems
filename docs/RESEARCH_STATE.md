# Research state and handoff

A complete working dump for a second model to reason about this project cold. Covers
what the system is, what we have measured, the bugs we found and fixed, the open
problems, our hypotheses, and the concrete improvement directions under review. It is
deliberately honest about what does NOT work yet.

Repo: https://github.com/kalyvask/self-improving-agentic-system
Local: `C:\Users\alexa\OneDrive\Documents\GSB\claude\wdp-controller`
Course: Stanford CS329A (Self-Improving AI Agents). Reading list:
`C:\Users\alexa\OneDrive\Documents\GSB\CS329A Self-Improving AI Agents\readings`

---

## 1. What the system is

A **self-improving, cost-aware compute-allocation controller** ("the Allocator"). On a
tool-using agent task, at each decision node it picks one of four actions for the next
unit of compute:

- `WIDER`  — spawn a fresh parallel Executor attempt (pass@k style coverage)
- `DEEPER` — continue / refine the current trajectory on tool feedback
- `DECOMPOSE` — hand the task to a Planner, producing a sub-task DAG
- `STOP`   — stop spending and abstain (a safe non-attempt)

The Allocator is a **tiny linear-softmax policy** over cheap numeric features, NOT a
fine-tuned LLM (the design constraint is "API credits, not GPUs": the policy must train
on a laptop in seconds). It **self-improves from its own logged traces**: round 0 is a
Thompson-sampling bandit cold start; later rounds refit the policy on accumulated traces
and re-collect. The headline is a self-improvement curve (solve rate and cost per round).

**Thesis metric is COST**, not raw solve rate: spend less for equal-or-better solve. The
whole point is that the best allocation policy depends on which currency you spend
(tokens / latency / dollars), so success is always paired with a budget and a currency.

### Architecture (files)
- `src/wdp/allocator/policy.py` — `Action`, `NodeFeatures`, `BanditAllocator`
- `src/wdp/allocator/linear.py` — shared `LinearSoftmaxPolicy` (fit_bc / fit_dpo / fit_kto / grpo_update)
- `src/wdp/allocator/{bc,dpo,kto,grpo}.py` — the four learners, all on the SAME policy core + SAME BC reference, so any difference is the objective, not capacity
- `src/wdp/loop/runner.py` — `run_task` (the control loop) + `run_round`; `_features`
- `src/wdp/loop/trace.py` — `TaskTrace`, `DecisionRecord`, `assign_credit`
- `src/wdp/loop/improve.py` — `self_improve` driver (BC/DPO/KTO, fit_window)
- `src/wdp/loop/grpo_train.py` — on-policy GRPO loop (separate, regenerates rollouts)
- `src/wdp/verifier/scorer.py` — `LLMProcessVerifier` (cheap), terminal verifier protocol
- `src/wdp/executor/react.py` — ReAct loop, Task/Trajectory
- `src/wdp/planner/decompose.py` — decomposability probe + sub-task DAG
- `src/wdp/metrics/{__init__,reliability,irt,alt_test}.py` — success@budget etc. + measurement layer
- `src/wdp/benchmarks/{arithmetic,taubench}.py` — benchmarks
- `src/wdp/grpo/estimate.py` — GRPO cost estimator
- `scripts/` — run_selfimprove, run_grpo_probe, analyze_eval, offline_ablations, run_calibrated_arith_sweep
- Tests: `tests/test_selfimprove.py`, `tests/test_measurement.py` (40 tests, all offline)

### The 12 features (NodeFeatures)
`score_mean, score_max, score_std, n_children, budget_remaining_frac, depth,
steps_taken, decomposability, executor_stalled, tool_error_rate, attempts_done_frac,
difficulty`. `difficulty = 1 - first_attempt_process_score` (pinned to the first attempt
so it reads intrinsic hardness, not best-so-far).

### Credit assignment (`assign_credit`)
Per spend decision: `value_per_cost = terminal_reward * cost_efficiency * advantage_weight`.
- `cost_efficiency = exp(-cost_weight * spent/budget)` (cost_weight=0.5). Smooth decay,
  always positive, keeps discriminating past budget.
- `advantage_weight` blends a uniform floor with the per-step process-score gain, so the
  decision that moved the verifier signal earns more.
- STOP decisions are credited by `abstention_reward` (1.0 only if the task was genuinely
  unsolvable), bypassing the efficiency/advantage terms.

### Benchmarks
- **Arithmetic** (local, exact + free grading): 110 tasks = 60 atomic (3 difficulty tiers
  by nesting depth), 40 multi-part (2-4 sub-results, DECOMPOSE has real payoff), 10
  underspecified (only STOP is right). This is where a POWERED comparison is affordable.
- **tau-bench** retail/airline adapter (multi-turn, env-graded, live LLM user sim). The
  realism check; ~0.8 solve on Haiku, headroom-limited, and ~$0.18/task so it cannot be
  run at the sample sizes needed for power.
- Models: Claude Haiku 4.5 for executor / planner / scorer (cheap path).
- **Calibrated budget = 0.003** (~2x the median arithmetic task cost), so spent/budget ~
  0.5 where the exp cost term is sharpest. At the old default 0.2 the cost term was inert.

---

## 2. Headline result (powered arithmetic sweep, n=66 train / 44 eval, budget 0.003)

Same data, same policy core, same BC reference; only the learning objective differs.

| objective | behavior | solve | cost vs bandit | verdict |
|---|---|---|---|---|
| **BC**   | clones the bandit            | 0.82 | ~equal | flat, stable; cloning can't exceed its data |
| **DPO**  | balanced (pairwise, bucketed) | 0.82 | no resolved change | **robust** — never collapses |
| **KTO**  | → STOP (abstain)             | 0.55 (collapsed) | resolved cheaper | collapses; see bug #2 + open problem |
| **GRPO** | → WIDER (one cheap try)      | 0.55 (collapsed) | resolved cheaper | collapsed via bug #3; re-run pending |

Note the powered eval is still underpowered for binary solve rate (minimum detectable
effect at n=44 is ~+0.24); **paired per-task COST is the metric with power**. Reported
cost deltas use a paired bootstrap CI; solve uses Wilson intervals.

**Honest current state: no learner yet shows a clean "cheaper at equal solve" win that
resolves.** BC is flat. DPO is a tie. KTO and GRPO collapsed (bugs now fixed; re-runs
pending). The KTO round-1/2 policy DID show 0.82 solve at ~20% lower cost before its
round-3 collapse — the closest thing to the thesis win we have.

---

## 3. Bugs found and fixed (this is much of the real work)

1. **Cost-credit cap-flattening** (`b30bc41`). `efficiency = 1 - cost_weight*min(1, spent/budget)`
   capped every over-budget trace to the same floor, so a 3x blowout trained identically
   to a marginal overspend — no gradient against runaway spend. Fixed with `exp(-cost_weight*spent/budget)`.
2. **KTO double-beta** (`876c670`). The sigmoid argument was `beta*(r - z)` but `r` already
   carries beta (`r = beta*log(pi/pi_ref)`), so the argument was ~`beta^2*logratio`, pinned
   near 0. That kept the sigmoid at ~0.5 for every example, **disabling KTO's
   self-normalizing saturation**: gradients became a constant per-example push that never
   turned off, draining probability mass from down-weighted undesirable spends onto the
   rarely-used STOP action. Fix: argument is `(r - z)`. Offline P(STOP) 0.485 -> 0.003.
3. **GRPO std-normalization bias** (`13dd5e2`). Group-relative advantage divided by
   per-group std. When all G rollouts solve, the only variation is cost jitter (std~0.02),
   and 1/std amplified it into advantages LARGER than the genuine solve/fail signal from
   mixed groups (std~0.32). In 95% of all-solve groups the amplified positive advantage
   landed on the cheapest rollout (a single WIDER) -> policy driven to WIDER by noise ->
   shed hard-task solves -> collapse 0.84->0.55. Fix: mean-center, drop std division
   (Dr. GRPO, arXiv:2503.20783). On probe traces, all-solve cost-jitter advantage |1.6|->|0.03|
   while mixed-group signal stays |0.52| (real signal now dominates ~15x).

Earlier fixes (pre-this-session): cost-credit normalization, billing verifier+planner into
the ledger, freezing the bandit during eval, fixing inverted STOP credit, fixing greedy
collection that collapsed exploration, seeds plumbing, recency-window fitting.

**Pattern across bugs #2 and #3:** a normalization that amplified noise and inverted the
intended optimization. Both have citable corrections in the reading list.

---

## 4. The measurement layer (a contribution in itself)

A small agent eval has almost no statistical power, so we built tooling to say what a
result can and cannot claim:
- `reliability.py` — Wilson CIs, minimum detectable effect, tasks-needed, McNemar (paired
  binary), paired bootstrap on per-task cost.
- `irt.py` — Rasch (1PL) fit for per-task difficulty + Fisher information (pick an
  informative small eval instead of a random one).
- `alt_test.py` — alternative-annotator test (Calderon et al. 2501.10970): a pass/fail
  verdict on whether the cheap process verifier is good enough to ACT on.

**Key measurement findings:**
- At n=10–44, binary solve-rate differences are mostly unresolvable (MDE +0.24 to +0.57).
  Cost (continuous, paired) is the metric with power. This reframes the whole eval.
- **The cheap ProcessVerifier FAILS the alt-test**: as a binary predictor of terminal
  success it agrees 0.60 vs a 0.69 majority-class baseline. It carries little actionable
  signal — which caps controller gains, since features lean partly on it.

---

## 5. Open problems, hypotheses, and concerns

### P1. The cheap process verifier is near-noise (the biggest ceiling)
The controller conditions on process scores and a difficulty feature derived from them,
but the verifier fails the alt-test. **Hypothesis:** generate *automatic* process labels
from rollout success rates (Math-Shepherd style) using our FREE exact terminal verifier,
and either (a) train a better cheap scorer or (b) replace the difficulty feature with a
rollout-based estimate. **Concern:** on arithmetic the terminal verifier is free, but on
tau-bench it isn't, so any verifier-improvement must transfer. (See §6 — verifier agent.)

### P2. KTO round-3 STOP drift, even after the beta fix
With the beta bug fixed, KTO rounds 1–2 were good (0.82, ~20% cheaper), then round 3
drifted to STOP as *correct-abstention* examples accumulated. **Hypothesis A (credit
asymmetry):** a correct STOP earns 1.0 while a cost-discounted correct solve earns ~0.7,
so KTO rationally prefers abstaining — cap abstention credit at/below the solve credit, or
treat STOP credit on the same efficiency scale. **Hypothesis B (representation):** the
linear policy can't separate "unsolvable" from "solvable" states, so STOP over-generalizes;
DPO avoids this because it compares within state buckets. **Concern:** fixes that suppress
STOP must not kill the legitimate abstention arm on the underspecified tasks.

### P3. Limited allocation headroom on arithmetic
On easy tasks a single WIDER suffices, so there's little to reallocate; that's partly why
BC/DPO look flat. **Hypothesis:** we need either a harder regime (more tasks that genuinely
need DEEPER/DECOMPOSE) or a stronger difficulty-conditioned policy so WIDER<->DEEPER flips
with difficulty (Snell). **Concern:** tau-bench is the realistic hard regime but is too
expensive to run at power.

### P4. The difficulty feature is weak
`difficulty = 1 - first_process_score` is derived from the failing verifier, and its
correlation with outcome/cost is weak and inconsistent. Tied to P1.

### P5. Small-eval power
Even the 110-task arithmetic suite gives MDE ~+0.24 on solve. We lean on cost + IRT-chosen
informative tasks, but a clean solve-rate claim would need many more tasks/arm.

### P6. GRPO KL-anchor strength and on-policy stability
beta_kl=0.05 was too weak to prevent the (now-fixed) collapse; even with the std fix, the
right KL strength / number of steps / group size is untuned. DAPO's tricks (dynamic
sampling, clip-higher) likely apply (see §6 — RL agent).

---

## 6. Academic improvement directions (from the CS329A reading list)

Distilled from a focused pass over the reading list, mapped to our open problems.
**Cross-cutting conclusion: the verifier (P1) is the prerequisite.** Three of four
allocation papers independently find that allocation/search gains are gated by verifier
quality (AB-MCTS gains shrink with weak feedback; Large Language Monkeys' selection
precision plateaus without a real verifier). So fix the verifier first, then allocation.

### Cluster A — verifier / generation-verification gap (the prerequisite, attacks P1, P4)
- **[HEADLINE] Math-Shepherd automatic process labels** (Math-Shepherd §3.3). Label each
  partial step by forking N=4–8 rollouts to termination and scoring with our FREE exact
  terminal grader; the fraction reaching a correct answer IS the process score. Replaces
  the 0.60 near-noise LLM verifier with a grounded signal — no model training, no judge
  calls beyond rollouts we can already run. Cost: API spend (rollouts), zero judge cost.
  Payoff: high — this is the headline fix.
- **Rollout-success as the difficulty feature** (Let's Verify App. G). Use the per-step
  correct-rollout fraction from the above as `difficulty`, replacing the verifier-derived
  estimate. Fixes P4 for free given the rollouts.
- **Binarize + discard the weak verifier** (Weaver §4.2.1). Our 0.60-vs-0.69 verifier is
  formally a "discard" candidate; drop it from features or binarize at a tuned threshold so
  it stops injecting noise. Free-offline.
- **Weaver weak-supervision ensemble** (Weaver §4.2). Run 3–6 cheap prompt variants,
  binarize, weight by agreement-estimated accuracy (calibrate on our ~10 free-labeled
  tasks). Lifted weak verifiers ~14–17% toward oracle. Cost: a few extra cheap calls/step.
- **Active labeling of "convincing wrong" traces** (Let's Verify §2.4, 2.6× data
  efficiency). Spend rollout-label budget only on high-verifier-score / wrong-terminal
  traces. Cost: API spend, but cheaper.
- **Per-step value > terminal scalar; cap best-of-N** (Cobbe §4.3, §5.1). Score per step,
  not one terminal score; best-of-N degrades past a point as search fools a weak verifier —
  caps how aggressively WIDER should expand until the verifier improves.
- _(Distill labels into a small cross-encoder verifier, Weaver §6, only if we need cheap
  live per-step scores — needs a one-time small train; otherwise use rollout labels offline.)_

### Cluster B — robust self-improving RL (attacks P2, P6, and the collapse bugs)
- **[COLLAPSE] DAPO Dynamic Sampling** (DAPO §3.2). Drop groups where all rollouts get the
  same reward (all-solve or all-fail) and over-sample until the batch has outcome variance.
  This is the standard cure for our all-solve-group problem — complementary to our Dr. GRPO
  mean-centering. (Note: DAPO/DeepSeek KEEP std normalization; our jitter pathology is
  specific to a sharp cost reward + low outcome variance, so mean-centering is right for us,
  but dynamic sampling is the more general fix.) Cost: API spend (re-roll), or free if we
  filter from an over-generated pool.
- **[COLLAPSE] Softer cost term** (DAPO §3.4 overlong-shaping lesson). A too-sharp cost
  reward is the same reward-noise pathology that nudges toward the cheapest action
  (STOP/WIDER). Lower `cost_weight`, or make cost a soft penalty only past a budget
  threshold; mask (don't reward-penalize) runs that hit the hard cap. Directly de-risks P2.
- **[COLLAPSE] STaR base-retraining + class-balancing** (STaR §3.1). Each round refit from
  the BC reference (we do) AND cap/down-weight the accumulating correct-STOP examples so the
  class balance can't drift — the direct fix for the KTO round-3 STOP drift (P2). Free.
- **Clip-higher** (DAPO §3.1) — decouple the PPO clip so rare actions (DECOMPOSE/DEEPER)
  can recover; monitor policy entropy per round as the collapse signature. Free.
- **k3 unbiased KL estimator + keep the BC anchor** (DeepSeekMath §4.1). Verify our KL is
  the positive low-variance k3 form; keep β·KL to BC as the main drift brake at tiny scale.
- **SWiRL per-step process credit** (SWiRL §2.1). Score each allocation decision by whether
  it was reasonable in its state (cheap judge or heuristic) and assign per-step advantages,
  rather than one terminal scalar for a whole WIDER/DEEPER/DECOMPOSE sequence. Strongest
  multi-step lever for tau-bench. Process-filter, don't outcome-filter, the trace pool.

### Cluster C — allocation / test-time compute (attacks P3, headroom; gated on Cluster A)
- **[HEADLINE] AB-MCTS Thompson WIDER↔DEEPER posterior** (Wider or Deeper, §3.2–3.4). Keep
  a Bayesian posterior over the score a fresh attempt (WIDER) vs a refinement (DEEPER) would
  earn; sample once from each, take the argmax. **Self-adjusts to difficulty with no
  difficulty label** — easy tasks let the DEEPER posterior dominate fast, hard tasks keep
  WIDER competitive. Add as a per-task posterior feeding the policy. Directly attacks P3/P4.
- **Snell difficulty-conditioned flip** (Scaling Test-Time Compute §3, §6.2). Easy → DEEPER,
  hard → balanced WIDER+DEEPER, monotone in difficulty. Encode as a `difficulty × (action==WIDER)`
  interaction feature. Free given a real difficulty signal (Cluster A).
- **Verifier-score-quintile difficulty + early-sample spread** (Snell §3.2, App. C). Bin the
  mean process score over the first k=2–4 samples into quintiles; high early-sample variance
  = headroom for WIDER. The warm-up samples double as real attempts. Cost: small API spend.
- **Large Language Monkeys power-law STOP rule** (§3.1). Fit coverage `log c ≈ a·k^b` online
  from the first few WIDER samples; STOP WIDER when predicted ΔP(solve) < cost. A principled
  WIDER budget AND an abstain trigger. Free-offline. (Precision plateaus ~100 samples without
  a real verifier — another reason Cluster A comes first.)
- **Archon offline BO over the policy/config space** (§3.3). Treat the policy weights or a
  per-difficulty-bin action table as a search space and Bayesian-optimize offline on the
  100-task suite instead of hand-tuning. Free-offline. Optimal config varies by task+budget.

### Suggested sequence (cheapest, highest-leverage first)
1. **Math-Shepherd rollout labels** → replace the verifier + fix the difficulty feature (A).
2. **DAPO dynamic sampling + softer cost term** → lock in non-collapsing GRPO (B).
3. **STaR class-balancing** → lock in non-drifting KTO (B).
4. **AB-MCTS Thompson WIDER↔DEEPER + Snell interaction feature** → create real allocation
   headroom now that the verifier signal is trustworthy (C).
5. **Power-law STOP rule** → principled cost lever / abstention (C).

---

## 7. How to reproduce / run
- Offline tests: `PYTHONPATH=src python -m pytest -q` (40 tests, no key/network).
- Offline analysis (no spend): `python scripts/analyze_eval.py --ab <traces> --irt <traces> --verifier <traces>`;
  `python scripts/offline_ablations.py --arith <traces>`.
- Calibrated sweep (spends credits): `scripts/run_calibrated_arith_sweep.sh` (bc/dpo/kto, ~$5-8).
- GRPO probe (spends credits): `python scripts/run_grpo_probe.py --seed-traces traces/calib_bc.jsonl --num-steps 20 --group-size 4 --budget 0.003`.
- Trace files live in `traces/` (gitignored). `.env` holds the OpenRouter key (never committed).

## 8. Constraints / ground rules
- Live runs cost real OpenRouter credits — gate before spending.
- Never commit `.env`, `traces/`, or `../pm-state/`.
- Plain technical writing in public docs; no overclaiming.
