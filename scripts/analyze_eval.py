"""Offline measurement-science analysis of collected traces -- no API spend.

Answers the two questions a tiny eval cannot answer by eyeballing solve rates:

  1. Is a solve-rate difference real or just noise? -> Wilson CIs per arm, the
     minimum detectable effect at our n, the tasks needed to detect a target
     lift, McNemar on the paired binary outcomes, and -- the metric that actually
     has power at small n -- a paired bootstrap CI on per-task COST.
  2. How hard is each task, and which tasks discriminate? -> a Rasch (1PL) IRT
     fit over all responses, giving per-task difficulty and Fisher information at
     the agent's ability (the basis for choosing an informative small eval).

Usage:
    python scripts/analyze_eval.py --ab traces/eval_ab_haiku.jsonl
    python scripts/analyze_eval.py --ab traces/eval_ab_haiku.jsonl \
        --irt traces/taubench_haiku_dpo.jsonl traces/eval_ab_haiku.jsonl
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wdp.loop import TraceLog
from wdp.metrics.reliability import (
    wilson_ci, min_detectable_effect, tasks_needed, mcnemar, paired_diff_ci,
)
from wdp.metrics.irt import fit_from_responses
from wdp.metrics.alt_test import alt_test, best_threshold


def _by_policy(traces):
    out = defaultdict(dict)            # policy -> {task_id: trace}
    for t in traces:
        out[t.policy][t.task_id] = t
    return out


def _pick_ab(names):
    """Choose the two policies to compare. Eval traces are now round-tagged
    (bandit@r0, bc@r1, bc@r2, bc@r3), so when there are more than two we compare the
    cold-start baseline (round 0) against the final learner round (highest @rN)."""
    if len(names) == 2:
        return names[0], names[1]
    def _round(p):
        return int(p.split("@r")[1]) if "@r" in p else -1
    baseline = next((p for p in names if _round(p) == 0), None)
    final = max(names, key=_round)
    if baseline is not None and final != baseline:
        return baseline, final
    return None


def analyze_ab(path: str) -> None:
    traces = TraceLog(path).read()
    pols = _by_policy(traces)
    names = list(pols)
    pick = _pick_ab(names)
    if pick is None:
        print(f"[ab] could not pick 2 policies in {path}, found {names}; skipping A/B.")
        return
    a_name, b_name = pick
    a, b = pols[a_name], pols[b_name]
    shared = sorted(set(a) & set(b))
    n = len(shared)
    print(f"=== paired A/B: {a_name} vs {b_name} | {n} shared tasks ===\n")

    def solved(t):  # robust to missing solved flag
        return bool(t.solved or t.terminal_reward >= 0.99)

    def cost(t):
        return (t.total_cost or {}).get(t.currency, 0.0)

    def solvable(t):       # a real gold answer exists (not an underspecified-abstain task)
        return t.abstention_reward < 0.5

    def stopped(t):
        return any(d.action == "stop" for d in t.decisions)

    def utility(t):        # solved OR correctly abstained on an unsolvable task
        return solved(t) or ((not solved(t)) and t.abstention_reward >= 0.5)

    def premature_stop(t):  # gave up on a task that actually had an answer
        return stopped(t) and solvable(t) and not solved(t)

    ka = sum(solved(a[t]) for t in shared)
    kb = sum(solved(b[t]) for t in shared)
    print(f" solve rate {a_name:>7}: {wilson_ci(ka, n)}")
    print(f" solve rate {b_name:>7}: {wilson_ci(kb, n)}")

    # Frontier is now mostly controlled by STOP quality, so a cost win must be read
    # next to whether it was bought by giving up. Report both arms side by side.
    n_solvable = sum(solvable(a[t]) for t in shared)
    print(f"\n quality, beyond raw solve (n_solvable={n_solvable}):")
    print(f"   {'metric':<16}{a_name:>10}{b_name:>10}")
    for label, fn, denom in (
        ("solvable_solve", lambda t: solved(t) and solvable(t), n_solvable),
        ("utility",        utility,        n),
        ("premature_stop", premature_stop, n),
    ):
        ra = sum(fn(a[t]) for t in shared) / denom if denom else 0.0
        rb = sum(fn(b[t]) for t in shared) / denom if denom else 0.0
        print(f"   {label:<16}{ra:>10.2f}{rb:>10.2f}")

    mc = mcnemar([(solved(a[t]), solved(b[t])) for t in shared])
    print(f"\n McNemar (paired solve): {b_name} wins {mc['b_only']}, {a_name} wins "
          f"{mc['c_only']}, p={mc['p_value']:.3f}  "
          f"({'no significant difference' if mc['p_value'] > 0.05 else 'significant'})")

    p_pool = (ka + kb) / (2 * n) if n else 0.0
    print(f"\n power on binary solve rate (pooled p={p_pool:.2f}):")
    print(f"   min detectable lift @ n={n}: +{min_detectable_effect(p_pool, n):.2f}")
    for d in (0.10, 0.15, 0.20):
        print(f"   to detect +{d:.2f}: need ~{tasks_needed(p_pool, d):.0f} tasks/arm")

    deltas = [cost(b[t]) - cost(a[t]) for t in shared]   # b - a (negative = cheaper)
    ci = paired_diff_ci(deltas)
    mean_a = sum(cost(a[t]) for t in shared) / n
    mean_b = sum(cost(b[t]) for t in shared) / n
    if ci.lo <= 0 <= ci.hi:
        sig = "  (straddles 0: not resolved)"
    elif ci.hi < 0:
        sig = "  <-- resolved CHEAPER (cost decrease)"
    else:
        sig = "  <-- resolved MORE EXPENSIVE (cost increase)"
    print(f"\n COST is the low-variance, paired metric with power at small n:")
    print(f"   mean cost {a_name}: {mean_a:.4f} | {b_name}: {mean_b:.4f}")
    print(f"   paired delta ({b_name}-{a_name}): {ci}{sig}")


def analyze_irt(paths: list[str]) -> None:
    responses = []
    for p in paths:
        for t in TraceLog(p).read():
            solved = 1.0 if (t.solved or t.terminal_reward >= 0.99) else 0.0
            responses.append((t.task_id, t.policy, solved))
    if not responses:
        print("[irt] no responses found.")
        return
    fit = fit_from_responses(responses)
    print(f"\n=== Rasch IRT difficulty | {fit.n_responses} responses, "
          f"{len(fit.items)} tasks, respondents={fit.respondents} ===")
    theta = float(fit.ability.mean())
    info = fit.information(theta)
    order = sorted(range(len(fit.items)), key=lambda i: fit.difficulty[i], reverse=True)
    print(f" ability theta (mean respondent) = {theta:+.2f}\n")
    print(f" {'task':<18}{'difficulty':>11}{'P(solve)':>10}{'info@theta':>12}")
    for i in order:
        print(f" {fit.items[i]:<18}{fit.difficulty[i]:>11.2f}"
              f"{fit.solve_prob(theta)[i]:>10.2f}{info[i]:>12.3f}")
    top = sorted(range(len(fit.items)), key=lambda i: info[i], reverse=True)[:5]
    print(f"\n most informative tasks at this ability (pick these for a small eval):")
    print("   " + ", ".join(fit.items[i] for i in top))


def analyze_verifier(paths: list[str]) -> None:
    """Alt-test verdict on whether the cheap ProcessVerifier is good enough to act
    on: does its best process score predict terminal success better than always
    guessing the majority outcome? Ground truth = the env-graded solved flag."""
    scores, truth = [], []
    for p in paths:
        for t in TraceLog(p).read():
            ps = [d.process_score_after for d in t.decisions
                  if getattr(d, "process_score_after", None) is not None]
            if not ps:
                continue
            scores.append(max(ps))
            truth.append(1.0 if (t.solved or t.terminal_reward >= 0.99) else 0.0)
    if not scores:
        print("[verifier] no process scores found.")
        return
    print(f"\n=== ProcessVerifier alt-test | {len(scores)} traces, "
          f"solve rate {sum(truth)/len(truth):.2f} ===")
    print(" " + str(alt_test(scores, truth, threshold=0.5, epsilon=0.05)).replace("\n", "\n "))
    print(" best operating point over thresholds:")
    print(" " + str(best_threshold(scores, truth, epsilon=0.05)).replace("\n", "\n "))


def analyze_stops(paths: list[str], policy: str | None = None) -> None:
    """STOP selectivity: split STOP decisions into CORRECT (abstained on an unsolvable
    task) vs PREMATURE (gave up on a solvable one). The goal of STOP-gating is to cut
    premature stops while keeping correct underspecified abstention -- a single utility
    number hides this. Pass --stops-policy to filter to one round tag (e.g. dpo@r3)."""
    for p in paths:
        per = defaultdict(lambda: [0, 0, 0])   # pol -> [correct_stop, premature_stop, n_stop_traces]
        for t in TraceLog(p).read():
            if policy and t.policy != policy:
                continue
            stopped = any(d.action == "stop" for d in t.decisions)
            if not stopped:
                continue
            unsolvable = (not (t.solved or t.terminal_reward >= 0.99)) and t.abstention_reward >= 0.5
            per[t.policy][0 if unsolvable else 1] += 1
            per[t.policy][2] += 1
        print(f"=== STOP selectivity | {p} ===")
        for pol, (cor, prem, n) in sorted(per.items()):
            print(f"  {pol:>10}: correct_STOP={cor}  premature_STOP={prem}  (of {n} STOP traces)")
        if not per:
            print("  (no STOP traces)")


def analyze_oracle(paths: list[str], policy: str | None = None) -> None:
    """Oracle-rescue: WHY does the chosen policy miss the tasks it misses? For each
    SOLVABLE task the target policy failed, classify the miss so the right next lever
    is obvious instead of assumed:
      - premature_STOP : the policy abstained on a task that had a real answer.
      - recoverable    : some OTHER policy/round in the pool solved the same task,
                         so the model CAN do it -- the miss is allocation/training,
                         not capability. ESCALATE would NOT help here.
      - capability_ceiling : no policy/round ever solved it -> a stronger model
                         (ESCALATE) is the only lever that can.
    A high recoverable/premature share says tune STOP / train the policy; a high
    capability share is the green light for the ESCALATE capstone (live spend)."""
    traces = []
    for p in paths:
        traces.extend(TraceLog(p).read())
    if not traces:
        print("[oracle] no traces found."); return

    def solved(t):
        return bool(t.solved or t.terminal_reward >= 0.99)

    names = sorted({t.policy for t in traces})
    def _round(p):
        return int(p.split("@r")[1]) if "@r" in p else -1
    target = policy or max(names, key=_round)

    # a task is "ever solved" if ANY policy/round in the pool solved it
    ever_solved = {t.task_id for t in traces if solved(t)}
    tgt = {t.task_id: t for t in traces if t.policy == target}
    if not tgt:
        print(f"[oracle] target policy {target} not in pool {names}."); return

    cap, recov, prem = [], [], []
    for tid, t in tgt.items():
        if solved(t) or t.abstention_reward >= 0.5:   # only SOLVABLE misses
            continue
        stopped = any(d.action == "stop" for d in t.decisions)
        if stopped:
            prem.append(tid)
        elif tid in ever_solved:                       # someone else got it
            recov.append(tid)
        else:
            cap.append(tid)
    n_miss = len(cap) + len(recov) + len(prem)
    print(f"=== oracle rescue | target={target} | pooled policies={names} ===")
    print(f" solvable misses by {target}: {n_miss}")
    if not n_miss:
        print("  (no solvable misses -- policy is at the oracle frontier)"); return
    print(f"   premature_STOP     : {len(prem):>3}  (gave up; STOP-tuning / training lever)")
    print(f"   recoverable        : {len(recov):>3}  (another arm solved it; allocation/training, NOT ESCALATE)")
    print(f"   capability_ceiling : {len(cap):>3}  (nobody solved it; ESCALATE is the only lever)")
    verdict = ("ESCALATE is justified" if cap and len(cap) >= len(recov) + len(prem)
               else "tune STOP / train policy FIRST -- most misses are recoverable, not capability")
    print(f"   -> {verdict}")
    if cap:
        print(f"   capability-ceiling tasks: {', '.join(sorted(cap)[:12])}"
              + (" ..." if len(cap) > 12 else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", help="paired trace file with exactly two policies")
    ap.add_argument("--irt", nargs="*", default=[],
                    help="trace files to pool for the IRT difficulty fit")
    ap.add_argument("--verifier", nargs="*", default=[],
                    help="trace files to pool for the ProcessVerifier alt-test")
    ap.add_argument("--stops", nargs="*", default=[],
                    help="trace files: split STOP into correct vs premature")
    ap.add_argument("--stops-policy", default=None, help="filter --stops to one policy tag")
    ap.add_argument("--oracle", nargs="*", default=[],
                    help="trace files: classify a policy's solvable misses (premature/recoverable/capability)")
    ap.add_argument("--oracle-policy", default=None,
                    help="target policy tag for --oracle (default: highest @rN)")
    args = ap.parse_args()
    if args.ab:
        analyze_ab(args.ab)
    if args.irt:
        analyze_irt(args.irt)
    if args.verifier:
        analyze_verifier(args.verifier)
    if args.stops:
        analyze_stops(args.stops, args.stops_policy)
    if args.oracle:
        analyze_oracle(args.oracle, args.oracle_policy)
    if not (args.ab or args.irt or args.verifier or args.stops or args.oracle):
        ap.error("pass --ab, --irt, --verifier, --stops, and/or --oracle")


if __name__ == "__main__":
    main()
