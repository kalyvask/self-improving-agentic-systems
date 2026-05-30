<!-- TEMPORARY working handoff for a second model. DELETE this file (and drop it from
     git) once the collaboration is done and its contents are folded into the README /
     final writeup. Not intended to live in the repo long-term. -->

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
- STOP decisions are credited by `abstention_credit * abstention_reward` (abstention_credit
  default 0.5; abstention_reward is 1.0 only if the task was genuinely unsolvable), bypassing
  the efficiency/advantage terms. The 0.5 scaling keeps a correct abstention below a solve so
  it can't dominate the value-weighted clone (see bug #4).

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

| objective | behavior (round 3 eval) | solve | verdict |
|---|---|---|---|
| **BC**   | clones the bandit             | 0.82 | flat, stable; cloning can't exceed its data |
| **DPO**  | balanced (pairwise, bucketed) | 0.82 | **robust** — never collapses |
| **KTO**  | stable after credit fixes     | 0.80 | **FIXED** (was collapsing to 0.55; bugs #2 + #4) |
| **GRPO** | holds ~6 steps then drifts to WIDER | 0.84→0.52 | std fix #3 PARTIAL; still collapses, needs dynamic sampling |

Note the powered eval is still underpowered for binary solve rate (minimum detectable
effect at n=44 is ~+0.24); **paired per-task COST is the metric with power**. Reported
cost deltas use a paired bootstrap CI; solve uses Wilson intervals.

**Honest current state:** after fixing four normalization/credit bugs (below), BC, DPO,
and KTO are stable and competitive (~0.80-0.82), and the "collapses" we first saw were
bugs, not objective limitations. No learner yet shows a *resolved* cheaper-at-equal-solve
win, because one WIDER attempt already solves most arithmetic tasks (thin headroom). GRPO
remains the open case: the std fix doubled its time-to-collapse but it still drifts to
WIDER (see §3 bug #3 note + the residual-mechanism finding in §5 P6).

---

## 2b. Why we are not improving yet (plain words)

We have not yet moved the headline -- more solves, or clearly lower cost. The honest reason is
simple, and it points straight at what to do next.

**What the controller can and cannot do.** It only decides HOW to spend compute on a task:
another fresh attempt (WIDER), keep going on the current one (DEEPER), split it up (DECOMPOSE),
or give up (STOP). It does NOT make the underlying model (Haiku) any smarter. That is the key
limit, and it squeezes the controller from both sides:
- On EASY tasks one attempt already solves them, so there is nothing to allocate better.
- On HARD tasks Haiku simply cannot do, re-routing the same weak model more cleverly does not
  get you past a capability wall.
- That leaves only a thin middle band where smart allocation matters. On our arithmetic suite
  that band is small (atomic 0.94, multi 0.65, unsolvable 0.00), so the ceiling on any
  improvement is low to begin with.

**Three more things hold it down:**
1. The cheap progress-checker (process verifier) is near-noise: it fails the alt-test (agrees
   with the truth 0.60 vs 0.69 for just guessing the majority). The controller partly steers
   on a bad signal.
2. The eval is small (44 tasks), so even a real small gain would not show above the noise.
3. GRPO does not learn at this scale (tiny policy, ~5 useful training groups per step, flat
   reward) -- not a bug, just the wrong tool here. BC/DPO are stable; DPO is the robust pick.

**And the earlier "collapses" were bugs, now fixed** (four credit/normalization bugs, §3).
After fixing them the learners are stable and competitive -- just flat, for the reasons above.
So the work so far bought correctness and understanding, not yet a bigger number.

**What would actually move it, in order:**
1. **Add a lever that changes outcomes: ESCALATE to a stronger model (Opus).** Today no action
   can solve a task Haiku cannot. Escalation raises the ceiling AND creates a real
   cost-vs-accuracy choice the controller can optimize ("stay cheap on Haiku; escalate only
   when stuck"). Biggest lever for solve rate, and the most thesis-relevant. Nuance: on
   arithmetic the cheap winning move for multi-part tasks is often DECOMPOSE (split into atomic
   parts Haiku already solves), so the interesting decision is decompose-vs-escalate on cost;
   on tau-bench, where the gap is genuine reasoning, escalation is the only ceiling-raiser.
2. **Give it a trustworthy signal: rollout-grounded labels (Math-Shepherd) or a strong-model
   judge.** Replace the near-noise verifier so decisions (and the difficulty feature) rest on
   something real. Cheaper than escalation -- paid once at train time, deploy stays cheap. But
   it sharpens DECISIONS only; it cannot raise the executor's capability ceiling.
3. **Tasks where allocation matters** (partly done: tier-3 + 5-part). No headroom, no visible gain.
4. **Measure on paired cost with enough tasks for power.** Solve rate at n=44 cannot resolve
   small wins; cost can.

One-line summary: we are flat because the controller can only re-route a fixed-capability model,
on an easy benchmark, judged by a weak signal. Fix any of those -- above all, add a real
capability lever (escalation) and a trustworthy training signal -- and there is room to improve.

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
4. **Abstention-credit asymmetry** (`9e8c06d`). A correct abstention was credited the
   maximum 1.0, while a correct solve is cost-discounted to ~0.6-0.8 -- so abstaining looked
   *better* than solving. That made correct-STOP decisions the single highest-weighted
   examples in the BC reference's value-weighted clone; every learner and the warm-started
   KTO/GRPO policies inherited a STOP-heavy reference and drifted toward STOP across rounds.
   Diagnosed offline: BC fit on the KTO arm's accumulated traces gave P(STOP)=0.47 vs 0.003
   on the clean BC arm DESPITE identical ~5% STOP frequency -- so it was the credit WEIGHTING,
   not the STOP count. Fix: scale STOP credit by `abstention_credit=0.5` so a correct
   abstention can't out-value a solve. Offline: BC P(STOP) 0.47->0.004, KTO 0.51->0.002.
   This is the true root of the "KTO round-3 collapse" (supersedes the P2 hypothesis below).

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

### P2. KTO round-3 STOP drift — RESOLVED (was bug #4, the abstention-credit asymmetry)
With the beta bug fixed, KTO rounds 1–2 were good (0.82, ~20% cheaper), then round 3
drifted to STOP. **Resolved:** Hypothesis A (credit asymmetry) was correct and is now fixed
(`9e8c06d`, see bug #4) — a correct STOP earned the max 1.0 vs a cost-discounted solve ~0.7,
so correct-STOP decisions dominated the BC reference's weighted clone and the whole
controller drifted. Scaling STOP credit by `abstention_credit=0.5` fixes it offline
(P(STOP) 0.47→0.004). Hypothesis B (representation: the linear policy can't separate
unsolvable from solvable) is a contributing factor, not the primary cause — the harder-task
P3 refinement (feature-separable underspecified tasks) addresses it. Live confirmation of
the fixed KTO curve was running at the time of writing.

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

### P6. GRPO still collapses after the std fix — OPEN (mechanism verified)
The Dr. GRPO std fix (#3) roughly doubled time-to-collapse (held 0.80-0.86 through step 6,
vs broken collapsing by step 5) but GRPO still drifts to WIDER (0.86) and craters to 0.52
by step 8-10. Verified on the fixed-probe traces:
- 67% of groups carry NO outcome signal (76 all-solve + 31 all-fail of 160).
- Net mean-centered advantage: STOP -0.295 and DECOMPOSE -0.150 are strongly suppressed;
  DEEPER +0.045 and WIDER +0.016 are mildly up. So GRPO isn't preferring WIDER over DEEPER
  directly -- it SUPPRESSES STOP and DECOMPOSE, and the freed mass flows to the
  BC-warm-start-dominant WIDER.
- In all-solve groups the cheapest winning rollout is WIDER 69x vs DECOMPOSE/DEEPER -- so on
  a multi-part task that several actions solved, the cost term hands the win to cheap WIDER
  and suppresses the DECOMPOSE that is actually NECESSARY on the hard multi-part tasks ->
  solve craters there.
**Root cause (the through-line across all four learners):** a sharp per-rollout COST term
over-penalizes the necessary-but-expensive action (DECOMPOSE for GRPO, STOP credit for KTO),
and each objective amplifies it differently; on a thin-headroom benchmark this yields a
degenerate single-action policy.

**Dynamic sampling tried -- arrested the collapse but GRPO still does not learn (NO BUG).**
Added DAPO dynamic sampling (drop all-same-outcome groups). Result: the catastrophic
cheap-WIDER collapse is gone (solve no longer craters to 0.52; it holds ~0.66-0.70), and
DECOMPOSE survives so cost rises rather than cratering -- but GRPO still ends BELOW its 0.84
BC warm-start. Verified on the collection traces that this is not a sign error or eval bug:
the mean training reward is FLAT across steps (~0.45, never climbs), eval uses the updated
policy, warm-start copies BC correctly, and the reward rewards solving. Conclusion: on-policy
GRPO simply does not learn at this scale -- a tiny linear policy with only ~5 informative
groups per step gives too noisy a gradient to beat a good BC warm-start, and it costs 2-5x to
run. This is the wrong tool for this regime; DPO (offline, pairwise) is the robust choice.
Remaining GRPO knobs (softer cost_weight, clip-higher, more steps) are unlikely to change the
verdict and are not worth the spend.

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

### Suggested sequence (updated with what we have learned)
- [DONE] **DAPO dynamic sampling** — arrested the GRPO cheap-collapse but GRPO still does not
  beat its BC warm-start (does not learn at this scale, §5 P6). Not pursued further.
- [DONE] **Harder, separable tasks** — tier-3 atomic + 5-part multi added for headroom.
- **Step 3, verifier/signal:** Math-Shepherd rollout-grounded labels (or a strong-model judge)
  to replace the near-noise verifier + the difficulty feature. Cache by
  `task_id + transcript_hash`; swap in for `difficulty = 1 - first_process_score` at
  runner.py:~63. Sharpens DECISIONS; paid once at train time, deploy stays cheap; cannot raise
  the executor capability ceiling.
- **Step 4 (capstone), ESCALATE to a stronger model (Opus) as a 5th action.** The only lever
  that raises the capability ceiling and creates a real cost-vs-accuracy frontier the
  controller optimizes ("stay cheap on Haiku; decompose; escalate only when stuck"). Most
  thesis-relevant; likeliest path from flat to a resolved win. Policy auto-sizes to 5 actions;
  cost ledger already handles the ~10-15x price gap.
- **Step 5, tau-bench demonstration** with escalation + better verifier, measured on PAIRED
  COST — tau-bench is where the capability gap is genuine, so escalation pays there; worth the
  spend only after steps 3-4.
- (Also available: AB-MCTS Thompson WIDER/DEEPER router; power-law STOP rule.)

### Design note: better-judge vs escalate are two DIFFERENT ceilings (do both, in order)
A second model asked whether escalation or "a better judge that trains the simpler model" is
the optimal next move. They are not substitutes -- they lift different ceilings:
- **Better judge / teacher (train-time)** fixes the *decision* ceiling. A strong LLM as the
  process verifier (or Math-Shepherd rollout labels) gives the controller a trustworthy signal,
  so it learns *when* to DECOMPOSE / DEEPER / STOP. Paid once at train time; inference stays
  cheap. It CANNOT make the executor solve a task it is incapable of -- it only improves which
  action the controller picks. (Note: we cannot fine-tune Haiku via API, so "train the simpler
  model" here means train the linear CONTROLLER on better-judged credit, not fine-tune the LLM.)
- **Escalate to a stronger model (inference-time)** fixes the *capability* ceiling. It is the
  only action that can solve a task Haiku cannot, and it makes the cost metric meaningful
  (~10-15x price gap), turning the flat curve into a real cost-vs-accuracy frontier.
- **Our data says do the judge first.** The arithmetic gap is the multi-part tasks (0.65), and
  the controller rarely DECOMPOSES (0.04-0.10) -- so part of the gap is a *decision* problem a
  better judge can close (teach it to decompose), with no inference-time spend. Escalation then
  adds the capability lever on top, giving a 3-way cost choice (DEEPER vs DECOMPOSE cheap-Haiku
  vs ESCALATE expensive-Opus). The capstone result is the controller learning a cost-aware
  cascade: "solve cheaply with Haiku / decompose; escalate only when truly stuck." That hybrid
  (strong teacher at train, cheap student + escalation valve at inference) is also the standard
  production pattern. Sequence: **judge/verifier (step 3) -> escalate (step 4) -> tau-bench on
  paired cost (step 5).**

### Cross-check + refinements (independent second-tool analysis + our offline tests)
An independent analysis converged on the SAME diagnosis (verifier-first; STOP-credit
asymmetry; GRPO needs dynamic sampling; not enough allocation headroom; add AB-MCTS),
which is strong corroboration. Net-new, actionable refinements from it:
- **Cache Math-Shepherd labels by `task_id + transcript_hash`** (folded into step 1 above).
- **Harder benchmark designs (refines P3):** add multi-step expression tasks, tasks with
  *distractor tool errors*, tasks where DECOMPOSE is *structurally necessary*, and
  underspecified variants that are *feature-separable* from normal-hard tasks — the last
  gives the linear policy a real handle to gate STOP instead of conflating it.
- **REJECTED after offline test — late-STOP efficiency discount.** Idea: discount a correct
  abstention by `cost_efficiency(spent)` so early abstention beats late. Tested: it pushed
  EVERY correct STOP (even early) to 0.35–0.49, below KTO's 0.5 desirability threshold,
  neutralizing the abstention arm for ZERO gain on the collapse (BC 0.004 / KTO 0.002,
  identical to the flat 0.5). Not worth the complexity + downside; STOP credit stays flat
  at `abstention_credit * abstention_reward`. (Don't re-try without decoupling the threshold.)
- *Note:* "class-balance desirable-STOP tags to the ~9% prior" was also suggested, but our
  offline test showed removing the desirable-STOP TAGS changed nothing (P(STOP) stayed
  0.51) — the driver was BC's value-per-cost WEIGHTING, which the abstention_credit fix
  already addresses. So tag-balancing is unnecessary; credit-scaling is the right lever.

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
