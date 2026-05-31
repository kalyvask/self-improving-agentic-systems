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

## >>> RESULT OF calib4 (k-sweep DONE -- k=3 is the operating point, accuracy RECOVERED) <<<

The DPO STOP-selectivity k-sweep (`--stop-after-failed-attempts` {2,3,4}, mask->spend +
DEEPER-honest fixes in) COMPLETED (bg `bkklrfxmj`, exit 0). Eval = paired `dpo@r3` vs cold-start
`bandit@r0` on the same 44 tasks:

| k | solve (both arms) | mean cost bandit->dpo | paired cost CI | correct STOP | premature STOP |
|---|---|---|---|---|---|
| 2 | 0.795 = 0.795 | 0.0029 -> 0.0018 | -0.001 [-0.002,-0.000] resolved | 6 | 3 |
| **3** | **0.841 = 0.841** | **0.0030 -> 0.0018** | **-0.001 [-0.002,-0.001] resolved** | 6 | **1** |
| 4 | 0.841 = 0.841 | 0.0030 -> 0.0021 | -0.001 [-0.002,-0.000] resolved | 3 | 0 |

**k=3 wins decisively and is now the headline operating point.** It RECOVERS the calib3 raw-solve
sag (0.77 -> 0.84, equal to baseline, McNemar p=1.0 = accuracy not dropping), KEEPS the full ~40%
cost win (tightest CI of the three), and cuts premature stops 3 -> 1. k=4 over-conserves (only 3
STOPs, cost win shrinks to 0.0021). DONE: README headline+caveat rewritten to k=3 numbers;
`make_figures.fig_frontier` repointed to `calib4_dpo_k3_eval.jsonl` (calib3 fallback kept);
frontier PNG regenerated. The only remaining ceiling is capability (~0.84 Haiku) -> ESCALATE.

## >>> RESULT OF calib3 (superseded by calib4 k=3 -- kept for history) <<<

With all fixes + the STOP rule (`--stop-after-failed-attempts 2`), DPO and KTO now show a
RESOLVED cheaper-at-equal-solve win:
- **paired eval cost bandit@r0 0.0029 -> dpo@r3 0.0015, delta -0.001 [-0.002, -0.001] (excludes
  0 = resolved CHEAPER)**; solve 0.75 vs 0.77 (McNemar p=1.0, same); p95 cost 0.0087 -> 0.0026.
- KTO similar (~0.0013). Self-improvement curve bends DOWN-LEFT across rounds (dpo cost
  0.00287 r0 -> 0.00148 r3 at stable solve/util).
- Action mix healthy, NOT collapsed: dpo {deeper .43, wider .25, stop .20, decompose .12};
  STOP 0 -> 0.20 (the rule saved budget AND seeded learning). utility_rate 0.89 -> 0.91.
- Tradeoff to report honestly: raw solve is ~0.77 (down from calib2's 0.84) because the k=2
  STOP rule abstains on some solvable-but-hard tasks after 2 attempts; cost halved and utility
  rose in exchange. Tuning k higher trades cost back for solve. Solve itself is unresolved at
  n=44 (cost is the metric with power).
- **This is the thesis result:** a learned cost-aware policy that matches the baseline's solve
  at ~half the cost, with a sensible WIDER/DEEPER/DECOMPOSE/STOP mix. Next: refresh README
  Results + figures (`make_figures.py` -> read calib3*), then the ESCALATE capstone to push
  solve past the ~0.84 Haiku ceiling, then tau-bench on paired cost.

## >>> RESULT OF THE calib2 SWEEP (earlier; DECOMPOSE fix validated) <<<

The post-fix sweep COMPLETED. **DECOMPOSE fix validated end-to-end:** decompose now solves
50-81 multi-part tasks (bc=81, dpo=51, kto=52), vs **0 across every pre-fix run**. It is also
actively chosen (88-107x). So the structural chain works: synthesis + separable feature +
BC-keeps-successes.

**BUT the aggregate frontier is still flat at n=44:** bandit/bc/dpo/kto all ~0.84 solve;
DPO@r3 vs bandit@r0 paired cost delta -0.000 [-0.001, 0.001] (straddles 0, NOT resolved;
McNemar p=1.0). Mean cost actually ROSE (~0.0031) because DECOMPOSE is now used and the
synthesis pass costs more. **Diagnosis: the controller decomposes a lot but is not yet
SELECTIVE** -- it solves multi tasks but does not net a cost win. The next lever is teaching
selective decompose (only the hard multi) and/or ESCALATE; STOP-exploration still open.
Solve is capped ~0.84 (Haiku ceiling), so cost is the only axis with headroom.

## >>> PICK UP HERE (next session) -- calib4 DONE, choose the next lever <<<

calib4 is analyzed and folded in (see the calib4 RESULT block at the top): **k=3 is the operating
point**, accuracy recovered to 0.84=0.84, ~40% cost win resolved, premature stops down to 1.
README + frontier figure + this doc are updated and pushed. Step 2 below is NO LONGER FORCED:
premature stops at k=3 are already low (1 of 7) and solve is back at the Haiku ceiling, so the
evidence-gated-STOP / true-DEEPER-revise work is optional polish, not a blocker.

**KEY NEW FINDING (oracle-rescue diagnostic, `analyze_eval.py --oracle`, no spend): ESCALATE has
ZERO headroom on the arithmetic benchmark.** Pooling every Haiku arm (calib4 k2/k3/k4 + calib3
bc/dpo/kto = 10 policy-rounds), `dpo@r3` misses exactly **1 solvable task**, and it is a
premature_STOP / recoverable miss -- **capability_ceiling = 0 in every pooling**. solvable_solve is
0.97; the raw 0.84 only looks like a ceiling because 6 of 44 tasks are underspecified-abstain. So
some Haiku config solves every solvable arithmetic task: a stronger model has nothing to add here,
and a live ESCALATE run on arithmetic would produce a NULL result by construction. The oracle
diagnostic just saved that spend.

**Revised next lever: do NOT run ESCALATE on arithmetic OR the current tau-bench set.** Oracle
check is now run on BOTH: arithmetic capability_ceiling=0 (1 recoverable miss over 10 Haiku
policy-rounds) AND tau-bench capability_ceiling=0 (7 tasks, all recoverable; n=7 is also far too
small -- MDE +0.68). **No benchmark in the repo has a real capability ceiling**, so ESCALATE (and
any Opus judge in front of it) has nothing to demonstrate. The cost thesis is DONE on arithmetic
(k=3). The real PREREQUISITE for the capstone is **a harder benchmark tier where Haiku fails
~20-40%** (Step 0 in `docs/ESCALATE_DESIGN.md`), gated by re-running `--oracle` until
capability_ceiling lands in that band.

**ESCALATE PROGRESS (weak->strong cascade test bed -- GATE PASSED, building the action):**
After two failed attempts to make a ceiling via task difficulty (hard-calc and no-calc both
solved ~1.0 by Haiku-4.5), pivoted to the MODEL axis: weak cheap model -> Haiku-4.5 ESCALATE
target. Added `--cheap-model` override (no config edits) and graded the no-calc tier
(easy/medium/hard chain lengths). Probe ladder on identical no-calc tasks: Haiku-4.5 1.00,
claude-3-haiku 0.94, **llama-3.1-8b 0.56 (8-decision) / 0.17 (single-attempt)**. Reference runs
(`casc_llama`, `casc_haiku`, 60 graded tasks, single attempt): llama 0.17 vs Haiku **1.00**,
50/60 rescuable. **Integrity-checked**: direct llama outputs confirm genuine misses (echoes the
start number / stops mid-chain), NOT a grading artifact. Gradient is flat (llama weak across all
bands), so the escalation signal is POST-ATTEMPT FAILURE (cheap tries -> low score -> escalate),
learnable from the process-score feature, not task length. Phases 1+2 DONE (Haiku rescues 100%, so
the casc_haiku run IS the label set). NOW: Phase 3 -- build ESCALATE as the 5th action (runner
branch executes the step on the strong model + bills real cost; policy + credit via existing
value_per_cost). Then Phase 4 alt-test, Phase 5 live calib5 (llama-only DPO vs DPO+ESCALATE vs
Haiku-only). Target = Haiku-4.5 first (cheaper than Opus); Opus only if Haiku is not enough.
Cascade story: "small cheap model handles the easy share; learned controller escalates what it
can't solve to a frontier model -> near-Haiku solve at a fraction of Haiku-only cost." calib4
Haiku-only cost result stays SEPARATE (two experiments).

**PHASE 5 FINAL (CASCADE WIN -- DONE, in README + cascade_frontier.png):** weak->strong cascade
works. cheap=claude-3-haiku, strong=Haiku-4.5, no-calc graded tasks, budget 0.0008, cost-weight 1.5.
Learned cascade (dpo@r2): **solve 1.00 @ $0.00079, escalate_rate 0.42** vs Haiku-4.5-only 1.00 @
$0.00125 -> **~37% cheaper at equal solve, paired cost delta -0.000 [-0.001,-0.000] RESOLVED,
McNemar p=1.0**. claude-3-haiku-only = 0.88 @ $0.00038 (the cascade recovers the 0.12 gap by
escalating). TWO KEY FIXES en route: (1) naive ESCALATE collapses to always-escalate-at-step-0
(sure 1-call solve -> bandit locks on -> clone) -- fixed by gating ESCALATE on n_children>=1 (it is
a RESCUE after a cheap miss, not a step-0 shortcut); (2) cascade only saves cost when cheap model is
BOTH cheaper AND capable -- llama-8b (0.17 single-attempt, retry-cost ~= 1 Haiku call) gave NO
saving (always-escalate genuinely ~optimal, policy was right); claude-3-haiku (0.88) does. cost-weight
0.5->1.5 pushed escalate 0.50->0.42, saving 29%->37% at solve 1.0. Files: casc3_A_c3h (cheap-only),
casc5_B_cw15 (cascade), casc2_C_haikuonly (ceiling). This is a SEPARATE experiment from calib4
(different cheap/strong pair; not mixed into single-model numbers). Remaining optional: push
cost-weight higher / let policy retry cheap before escalating (more saving toward ~60% theoretical);
tau-bench transfer (credibility); Opus as strong target if a bigger lift is wanted. Phase 4 alt-test
is structurally satisfied (rescue gate conditions escalation on the cheap-attempt outcome).

**PHASE 5 RESULT v1 (uncalibrated budget -- cascade NOT selective; diagnosed + re-running):**
ESCALATE built (Phase 3, 52 tests). 3-arm eval (60 graded no-calc tasks, budget 0.02, dpo@r2):
A llama-only solve 0.62 @ $0.00103; **B cascade solve 1.00 @ $0.00124 but escalate_rate 1.00**;
C haiku-only solve 1.00 @ $0.00125. So B == C: the cascade lifts solve (0.62->1.00) but saves
NOTHING because it escalates every task. Root cause = budget too flat (0.02 >> ~0.001 task cost)
-> escalate barely penalized -> "always escalate" ties "selective". Compounded by llama burning
~5 retries (~$0.001, ~= one Haiku call) so a SINGLE llama attempt (~$0.0002, ~6x cheaper) is the
real edge. Cost-optimal = "try llama ONCE, escalate on failure" (~half the always-escalate cost),
which is feature-learnable (escalate when n_children>=1 & score_max~0). FIX = budget calibration:
re-running all 3 arms at **budget 0.0008** (~1.5x a Haiku call) so escalation genuinely costs while
solve_floor keeps a necessary escalate-solve worth it. Files: casc2_{A,B,C}_*. If B now escalates
SELECTIVELY (rate < 1, cost between A and C, solve ~1) that is the cascade win; if still
always-escalate, lower budget further / raise --cost-weight, else report honestly.

**PHASE 0 RESULT (hard-arith-v1 -- NO CEILING, blocked, awaiting direction):** Added a `hard`
tier (`--hard N`): multi-hop word problems, tangled prose, distractor numbers, a final conditional,
non-decomposable. Live Haiku probe (30 tasks, 2 rounds, `traces/hard_probe*`): eval solve **1.00**
for bandit AND dpo, oracle = **0 solvable misses**. So Haiku-4.5 + the calc tool is
capability-SATURATED on arithmetic -- the calc tool removes the only hard part, and prose
tangling/distractors/conditionals are not enough. The Haiku<->Opus gap is NOT in computation; it is
in reasoning-setup or only appears once the calculator crutch is gone. Manufacturing an honest
ceiling needs a different lever. Options put to the user (DISMISSED -- waiting for instruction):
(1) remove calc on hard tasks (in-context multi-step arithmetic; synthetic but reliable, cheap),
(2) bigger tau-bench split (realistic, more spend; n=7 earlier was too small to show a ceiling),
(3) derivation word problems (rate/mixture/systems; uncertain it breaks Haiku), (4) keep ESCALATE
design-only (no honest ceiling -> don't run live; the proven cost thesis stands alone).
Phase 0 code is committed (b62615c); the `--hard` tier + `_hard_problem` generator are in place and
reusable. Phases 1-5 are BLOCKED on the ceiling decision.

**ESCALATE design is written: `docs/ESCALATE_DESIGN.md`** -- the validated answer to the
"Opus-judge-before-escalation" question. Verdict: an Opus judge in the LIVE loop is invalid here
(free ground truth makes it pure cost; judge tier > execution tier inverts cascade economics). The
correct form is **Opus as an OFFLINE labeler** (never in the live ledger): label which
capability-ceiling tasks a stronger model actually solves -> train the 5-action policy on CHEAP
features -> alt-test the cheap escalate signal vs Opus labels -> live eval with the win condition
"solve above the single-model ceiling at a cost paid only on capability-limited tasks," beating a
feature-only router baseline. Reuses `rollout.py` labeling_ledger, `alt_test.py`, `--oracle`.

**Done this session (offline, pushed):** `analyze_eval.py` gained `--oracle` (miss classification)
and folded solvable_solve / utility / premature_stop into the `--ab` headline; a DEEPER-semantics
characterization test (`test_deeper_targets_unfinished...`) locks that DEEPER falls back to a fresh
attempt on all-completed sets (tripwire for the future true-revise mode); gen_verif_gap dropped
from the GRPO display (renamed to `util`) since completed traces use terminal reward as the process
score. **Optional (only if premature stops creep back later):** evidence-gated learned STOP +
true DEEPER review-revise mode (needs executor support).

## >>> earlier PICK UP notes (still relevant for context) <<<

Latest commit on main: **1cfa1a9** (all pushed, 48 offline tests pass, tree clean).

0. **calib2 is DONE and analyzed** (RESULT block above: DECOMPOSE fixed, frontier flat). No
   sweep is running (a premature calib3 was launched then killed -- decision: do the FIXES
   FIRST, then ONE consolidated rerun, rather than spend on an incremental mask-only calib3).

   **Plan = implement the offline fixes, THEN a single rerun, THEN escalate** (user-confirmed
   order; a 2nd model + ours agree on the fixes):
   - **(a) Teach STOP — DONE** (`00c4df5`): hopeless-task rule (decomp=0 & no progress after k
     attempts & scores~0 -> STOP), gated by `--stop-after-failed-attempts` (default off). Gives
     STOP data + saves budget; abstention_credit<solve_floor guards drift. (A learned version
     with a `zero_score_attempts` feature + STOP exploration is the future refinement.)
   - **(b) Selective DECOMPOSE — LIKELY ALREADY SUBSUMED, verify in rerun:** solve_floor makes an
     expensive solve (>=0.6) out-rank a cheap failure (0), and `_bucket_key` already pairs within
     the decomposability bucket, so DPO pair-mining is roughly lexicographic already. If the rerun
     still shows DECOMPOSE over-used / cost not improving, THEN add explicit lexicographic mining.
   - **(c) NEXT: one consolidated rerun** (`calib3_*`, --overwrite, single wrapper, ADD
     `--stop-after-failed-attempts 2`): `for L in bc dpo kto; do python scripts/run_selfimprove.py
     --learner $L --benchmark arithmetic --atomic 60 --multi 40 --underspecified 10 --budget 0.003
     --max-decisions 8 --rounds 3 --seed 0 --stop-after-failed-attempts 2 --out
     traces/calib3_${L}.jsonl --overwrite; done`. Expect ~0.84 solve at LOWER cost (mask) + STOP
     catching the ~6 underspecified eval tasks (utility up, cost down). Headline = PAIRED COST.
   - **(c2) POST-calib3 review fixes (do before ESCALATE -- STOP-selectivity is the near frontier):**
     * DONE: DECOMPOSE-mask fallback no longer falls back to STOP (was manufacturing fake
       "learned STOP" / premature atomic stops; now picks best SPEND action). Commit pending.
     * PARTIAL: **DEEPER semantics fixed** (runner) -- now continues a genuinely-unfinished
       trajectory, else does a fresh attempt (WIDER-equiv) instead of a wasted no-op continue_from.
       STILL TODO (the impactful half): a true "review+revise the completed answer" mode so DEEPER
       can fix a wrong final answer -- needs executor/react support; likely lifts hard-atomic solve.
     * TODO **gate learned STOP by evidence** (decomp==0 & n_children>=k & score_max<=eps & no
       progress) so the POLICY can only pick STOP when warranted -- removes premature DPO/KTO stops.
     * TODO **STOP-gating ablation** (offline-ish): estimate the upper bound -- replacing DPO's ~4
       premature atomic STOPs with successful sequences ~= solve 0.86 / utility 1.0 at ~same cost.
       Run this BEFORE escalate; it shows STOP selectivity, not capability, is the immediate gain.
       In the k={2,3,4} sweep, track CORRECT_STOP vs PREMATURE_STOP separately (not just best k or
       a single utility number): the goal is to cut premature stops while preserving correct
       underspecified abstention. Prefer STOP firing ONLY via the explicit evidence predicate over
       free learned/masked STOP.
     * Use **DPO** as the base policy (KTO dropped DECOMPOSE + over-stops); fix make_figures curve
       parser (reads `util` as cost on new logs) + repoint non-headline figs to calib3; update
       run_calibrated_arith_sweep.sh to calib3 + `--stop-after-failed-attempts 2`.
   - **(d) THEN ESCALATE capstone** (task #54): the only lever past the ~0.84 Haiku ceiling;
     learn cheap-Haiku-first, Opus-only-when-stuck; then tau-bench on paired cost.
   Validate each fix offline before the rerun (recompute/refit on existing traces where possible).
2. **Then run the decisive analysis** (the whole point of this sweep -- DECOMPOSE can now solve
   multi-part tasks, the feature separates them, and BC keeps DECOMPOSE successes):
   - Does DECOMPOSE now SOLVE multi tasks (was 0 across all prior runs)? grep decompose+solved.
   - DECOMPOSE usage by task kind; solve_rate / solvable_solve_rate / utility_rate per round.
   - Paired eval cost: `python scripts/analyze_eval.py --ab traces/calib2_dpo_eval.jsonl`
     (now compares bandit@r0 vs final learner round; prints CHEAPER/MORE EXPENSIVE direction).
   - Compare to the PRE-fix `calib_*` sweep. Early read: cost rose (DECOMPOSE now exercised =
     "up and right"); DPO showed ~28% cheaper-at-equal-solve in round 1 (noisy -- confirm w/ CI).
3. **Refresh** `scripts/make_figures.py` to read `calib2*` + eval traces; update README Results.
4. **Then the two deferred levers** (see Next steps section): STOP-exploration (top behavior
   gap -- utility==solve until STOP is explored), then the ESCALATE capstone (task #54), then
   tau-bench. Escalation stays OFF until the above shows whether the fixes moved the frontier.

## Current status (latest commit 1cfa1a9)

- **All 8 bugs fixed and committed; 47 offline tests pass.** Latest commits: cae1958 (DECOMPOSE
  synthesis + 4), 9c67b3b (utility/solvable metrics + eval-trace logging), 20220a8 (decomposability
  + verifier + eval round id). Math-Shepherd rollout-difficulty built and gated (b02d392).
- **A decisive arithmetic sweep is RUNNING** (bg id may be stale): bc/dpo/kto, 110 tasks,
  budget 0.003, escalation OFF, no --rollout-difficulty, outputs `traces/calib2_{bc,dpo,kto}.jsonl`
  (+ `_eval.jsonl`). Re-run via `scripts/run_selfimprove.py ... --out traces/calib2_<L>.jsonl`.
  It is the first run where DECOMPOSE can actually solve multi-part tasks AND the policy has a
  separable decomposability feature. **When it lands, check:** does DECOMPOSE now SOLVE multi
  tasks (was 0)? does DECOMPOSE usage rise by task kind? solvable_solve_rate / utility_rate;
  paired eval cost (on the `_eval.jsonl` traces) vs the pre-fix `calib_*` sweep.
- **Escalation (ESCALATE to Opus) is deliberately DEFERRED** (task #54) until we see whether the
  fixes alone move the frontier. It is the planned capstone (step 4); then tau-bench (step 5).
- **Deferred/known-open:** STOP barely explored (bandit threshold too strict; ~9% abstention
  arm); rollout-difficulty would mis-bill on eval if enabled (precompute + report separately).
- **Expected shape of the result:** likely "up and to the right" first (DECOMPOSE usable ->
  more multi solves, more cost), then DPO/KTO should learn WHEN decompose is worth it and bend
  the frontier back left. A flat-but-correct result is still possible; report honestly.

## Next steps, pending actions, and ideas

**Immediate (in flight / on landing of the calib2 sweep):**
1. Analyze `traces/calib2_{bc,dpo,kto}.jsonl` (+ `_eval.jsonl`): DECOMPOSE solve count on
   multi tasks (was 0), DECOMPOSE usage by task kind, solve_rate / solvable_solve_rate /
   utility_rate, and PAIRED eval cost (bandit@r0 vs learner@r3 from the tagged eval traces)
   vs the pre-fix `calib_*` sweep. Tools: `scripts/analyze_eval.py`, `scripts/offline_ablations.py`.
2. Update `scripts/make_figures.py` to read the `calib2` + eval-trace files and refresh the
   four `artifacts/*.png`; update README Results with the post-fix numbers.
3. If DECOMPOSE now earns its keep and the frontier moves, that is the first real win -- write
   it up honestly (likely "up and right" first, then DPO/KTO bend it back left).

**Sequenced build path (the capstone), only after the above:**
4. **ESCALATE to Opus as a 5th action** (task #54, deferred). Build, then run an escalation
   headroom probe, then a paired comparison: DPO-cascade vs Haiku-only vs Opus-only, on COST.
   Do this AFTER the solve_floor credit fix (already in) so the cost term doesn't suppress it.
5. **tau-bench demonstration** with escalation + a better verifier, measured on paired cost --
   tau-bench is where the capability gap is genuine (Opus >> Haiku). Expensive; gate.
6. **Math-Shepherd rollout-difficulty** live run (gated, `--rollout-difficulty`) to test whether
   a grounded difficulty signal raises DECOMPOSE-on-multi usage further. Isolate from #1.

**Deferred / open (lower priority):**
- STOP barely explored (bandit threshold too strict, policy.py) -- only matters if the ~9%
  abstention arm matters; revisit if utility_rate lags solve_rate.
- rollout-difficulty eval billing: if ever enabled on eval, precompute labels + report cost
  separately (don't let the labeling ledger make eval look cheaper).
- DPO pair-mining could be made explicitly lexicographic (solved > correct-abstention > failed,
  then cheaper within class); solve_floor already pushes this direction.
- GRPO: do NOT spend more -- it does not learn at this scale; keep as the estimated/contrast arm.
- **STOP has no path into the data (top remaining BEHAVIOR gap):** 0 STOP on underspecified
  tasks at cold start because the bandit only stops when all spend arms sample < 0.02. So
  utility_rate == solve_rate until STOP is explored. Next behavior fix after the frontier check:
  force some STOP exploration (epsilon over STOP, or stop when scores stay ~0 after k attempts).
- **Measurement caveat:** now that completed trajectories use the exact terminal grade as the
  process score, `gen_verif_gap` and `analyze_eval --verifier` collapse toward 0 for completed
  work -- they no longer measure the cheap LLM verifier. Run the alt-test on the OLD/dedicated
  verifier-vs-terminal traces, not the new ones, or they will overstate verifier quality.
- DECOMPOSE is not hard-masked at decomposability=0 for the trainable policies (only the bandit
  gates it); greedy eval picks WIDER anyway, so minor -- the controller should learn it.
- **Append footgun fixed:** run_selfimprove now refuses pre-existing outputs unless --overwrite.

**Research ideas (from the CS329A reading list, see Section 6):**
- AB-MCTS Thompson WIDER<->DEEPER per-task posterior (self-adjusts to difficulty, no label).
- Snell difficulty x action interaction feature; Large Language Monkeys power-law STOP rule.
- Scale arithmetic n for real power (cheap, free grading) once a real effect appears.
- Distill Math-Shepherd labels into a small cross-encoder verifier only if live per-step
  scoring is needed.

**The thesis result to aim for** (not "DPO beats KTO"): *a learned cost-aware cascade matches
most of the strong-model solve rate while spending much closer to the cheap-model baseline.*

**Housekeeping:** task #50 = delete this doc when the collaboration ends (it is marked TEMPORARY
at the top); README Results + figures need a refresh once calib2 lands; never commit .env / traces.

## 0. Lessons learned / mistakes to avoid (READ FIRST)

Hard-won, from finding ~8 real bugs. Most "the learner is bad / the design is weak"
conclusions turned out to be BUGS. Patterns to apply next time:

- **A result that "shouldn't happen" is a bug signal, not a finding.** Every collapse we
  saw (KTO->STOP, GRPO->WIDER, DECOMPOSE never solving) was a bug, not an objective
  limitation. If an optimizer moves to *lower* reward, or an action *never* succeeds, dig
  before theorizing. The user's "that doesn't make sense" caught 4+ bugs.
- **Verify each ACTION can succeed end-to-end before blaming the policy.** DECOMPOSE solved
  0 tasks for a structural reason: `_run_decompose` concatenated sub-answers and the
  verifier read the last number, so it could never produce the parent sum. The learner
  correctly suppressed a broken action. Check the mechanics of every arm.
- **Features must be SEPARABLE or no learner can use them.** The decomposability probe rated
  atomic 1.0 and multi 0.83 -- inseparable, so the policy could not learn "decompose multi."
  A miscalibrated feature silently caps everything downstream.
- **The cheap LLM signals are near-noise; prefer exact/free signals.** The process verifier
  fails the alt-test (0.60 vs 0.69) and rated unsolvable tasks 0.95 while terminal=0. Use the
  exact terminal grade as the process score whenever a trajectory is complete; only fall back
  to the LLM for genuinely partial work.
- **The credit chain has four interacting invariants -- keep all of them:**
  (a) budget must be ~2x the median task cost or the cost term is inert;
  (b) cost-efficiency must keep discriminating past budget (exp decay, not a capped linear);
  (c) a correct abstention must be worth LESS than a solve (abstention_credit < solve_floor);
  (d) a necessary-but-expensive solve must stay above the abstention/threshold (solve_floor).
  Violating any one silently teaches "cheap mediocre beats necessary expensive."
- **Normalization bugs hide in plain sight.** KTO double-beta (sigmoid arg scaled by beta
  twice) and GRPO std-division (amplified cost-jitter in all-solve groups) both inverted the
  optimization. When in doubt, dump the per-example gradient/advantage sign by group.
- **Small evals can't resolve small wins.** n<=44 has MDE ~+0.24 on solve rate. Report COST
  (paired bootstrap) + Wilson/McNemar; never headline a raw solve-rate delta.
- **Don't spend on a sweep over a known-broken setup.** Validate fixes offline on existing
  traces first (recompute credit, refit, check the property), THEN spend.
- **On-policy GRPO does not learn at this scale** (tiny linear policy, ~5 informative groups
  /step, flat reward). DPO (offline, pairwise) is the robust learner. Don't keep tuning GRPO.
- **Cost/measurement hygiene:** bill every probe (rollout difficulty -> separate labeling
  ledger, not invisible); persist eval traces (not just train) and tag them by round; don't
  copy parent gold into subtasks; mask unavailable actions (DECOMPOSE with no planner is a
  logged no-op). Each of these silently corrupts the solve/cost matrix.
- **Process note:** do not run two agents on the same working copy -- it caused churn and a
  red test. One editor at a time; commit/push frequently; never `git -c` the global config.

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

## 2a. UPDATE: DECOMPOSE was broken (likely the main flat-frontier cause)

A later review found a structural bug that reframes the "flat frontier" story. `_run_decompose`
set the parent answer to the CONCATENATION of sub-answers (`"[s1] 6\n[s2] 20"`), and the
arithmetic verifier reads the last number -> it graded 20, never the sum 26. So DECOMPOSE was
*incapable* of solving a multi-part task: it solved 0 tasks across every calibrated run. The
learners were therefore CORRECT to suppress it, and the multi-part headroom (the band where
allocation should matter most) never materialized. Fixed (commit cae1958) by adding a synthesis
pass that combines the sub-results into the parent answer. Four smaller correctness fixes shipped
with it: mask DECOMPOSE when planner=None (was a logged no-op on tau-bench); bill rollout-difficulty
to a separate labeling ledger (was invisible); drop parent gold from subtask metadata; and a
solve_floor on credit (a solve keeps >= solve_floor of its outcome credit so a necessary-but-
expensive DECOMPOSE/ESCALATE is not pushed below a correct abstention) with an enforced ordering
0 <= abstention_credit < solve_floor <= 1.

**Implication:** the §2b explanation below is still right about the controller's structural
limits, but part of the flatness was this bug, not just thin headroom. The next step is a fresh
arithmetic sweep (escalation still OFF) to see whether DECOMPOSE usage now rises on multi-part
tasks and the cost/solve frontier finally moves -- before any escalation.

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

6. **DECOMPOSE could never solve a multi-part task** (`cae1958`). `_run_decompose` set the
   parent answer to the CONCATENATION of sub-answers, so the verifier graded the last number
   (20, not the sum 26). DECOMPOSE solved 0 tasks across every calibrated run -> learners
   correctly suppressed it -> multi-task headroom never appeared. Fix: a synthesis pass
   combines sub-results into the parent answer. **Likely the main flat-frontier cause.**
   (Shipped with: mask DECOMPOSE when planner=None; rollout-difficulty -> separate labeling
   ledger; drop parent gold from subtask metadata; solve_floor + ordering guard.)
7. **Decomposability feature miscalibrated** (`20220a8`). The cheap-LLM probe saturated near
   1.0, rating atomic single-expression tasks 1.0 and multi 0.83 -- inseparable, so the policy
   could not condition DECOMPOSE on task type. Fix: benchmark supplies a structural
   decomposability in metadata (atomic 0 / multi graded 0.25-1.0 by part count / underspec 0);
   planner.probe uses it when present (also drops the LLM probe cost on arithmetic).
8. **Noisy process verifier used even when exact grading was free** (`20220a8`). The LLM scorer
   rated unsolvable tasks 0.95 while terminal=0, corrupting the score features and wasting
   spend. Fix: for a COMPLETED trajectory use the exact terminal grade as the process score;
   LLM scorer only for genuinely partial work. Also added eval-trace round tags (name@rN) and
   utility_rate / solvable_solve_rate metrics.

9. **BC discarded DECOMPOSE (training-selection blocker)** (`cca18f1`). After DECOMPOSE could
   solve, `_filter_traces` still kept only the top-fraction by mean value-per-cost -- the cheap
   atomic solves -- so the BC reference (and DPO/KTO warm start) had ~0 DECOMPOSE/STOP examples.
   Fix: keep all SUCCESSFUL traces (solve or correct abstention), STaR-style; vpc weighting still
   favors cheap. Also: analyze_eval now compares bandit@r0 vs final learner round on tagged eval.
   OPERATIONAL note: a duplicate sweep (a lingering bash wrapper kept respawning python and
   re-appending to calib2_*) contaminated the prior attempt -- killed all by command line and
   relaunched ONE clean sweep. Always confirm a single wrapper before/after launching.

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
