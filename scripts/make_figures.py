"""Generate the README figures from already-collected traces and run logs.

Offline, no API spend. Reads the per-round/per-step curve tables out of the run
logs in traces/*.log (the exact eval numbers reported) and the per-decision
traces in traces/*.jsonl (for action-mix and confidence intervals), and writes
PNGs to artifacts/. The figures tell the honest story: the learners are stable
but flat, and the real content is the normalization/credit bugs that caused
apparent collapses (caught via action-mix drift) plus the measurement rigor.

    python scripts/make_figures.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wdp.metrics.reliability import wilson_ci  # noqa: E402

TR = ROOT / "traces"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

ACTIONS = ["wider", "deeper", "decompose", "stop"]
ACT_COLOR = {"wider": "#4C72B0", "deeper": "#55A868",
             "decompose": "#C44E52", "stop": "#8172B3"}

# A curve/step table row: "  3   kto   0.80   0.00229  0.00340  0.08   264"
_ROW = re.compile(r"^\s*(\d+)\s+([a-z]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d,]+\s*$")


def parse_segments(logfile: str):
    """Return [(label, [(step, solve, mean_cost), ...]), ...]; a new segment starts
    each time the step/round column resets to 0.

    Header-aware: locates the mean_cost column BY NAME from each header row, instead of
    assuming a fixed position. Older logs are `round policy solve mean_cost p95 ...`
    (mean_cost at index 3); newer logs inserted a `util` column
    (`round policy solve util mean_cost p95 ...`, mean_cost at index 4). A fixed-column
    parser silently read `util` as cost on the newer logs -- this avoids that."""
    segs, cur, label = [], [], None
    mean_idx = 3                          # sensible default for the oldest format
    for line in (TR / logfile).read_text().splitlines():
        toks = line.split()
        if "mean_cost" in toks:           # header row: relocate the cost column
            mean_idx = toks.index("mean_cost")
            continue
        if len(toks) <= mean_idx or not toks[0].isdigit():
            continue
        try:
            step, pol, solve, cost = int(toks[0]), toks[1], float(toks[2]), float(toks[mean_idx])
        except ValueError:
            continue
        if step == 0 and cur:
            segs.append((label, cur)); cur = []
        if pol != "bandit":
            label = pol
        cur.append((step, solve, cost))
    if cur:
        segs.append((label, cur))
    return segs


def _read(jsonl: str):
    return [json.loads(l) for l in (TR / jsonl).read_text().splitlines() if l.strip()]


def _solved(t):
    return 1 if (t.get("solved") or (t.get("terminal_reward") or 0) >= 0.99) else 0


def _cost(t):
    return (t.get("total_cost") or {}).get(t.get("currency", "dollars"), 0.0)


# ---- Figure 1: self-improvement curves (post-fix) --------------------------
def fig_curves():
    arms = {}
    for label, seg in parse_segments("calib_sweep.log"):
        if label in ("bc", "dpo"):
            arms[label] = seg
    kto = parse_segments("kto_fixed2.log")
    if kto:
        arms["kto"] = kto[0][1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, seg in arms.items():
        xs = [s for s, _, _ in seg]
        ax1.plot(xs, [v for _, v, _ in seg], marker="o", label=name)
        ax2.plot(xs, [c for _, _, c in seg], marker="o", label=name)
    ax1.set(title="Solve rate per round (44-task eval)", xlabel="round", ylabel="solve rate")
    ax1.set_ylim(0.4, 1.0); ax1.axhline(0.82, ls=":", c="gray", lw=0.8)
    ax2.set(title="Mean cost per round", xlabel="round", ylabel="mean cost ($)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Self-improvement curves (post-fix): stable, competitive, ~within noise", fontsize=11)
    fig.tight_layout(); fig.savefig(ART / "self_improvement_curves.png", dpi=130); plt.close(fig)


# ---- Figure 2: collapse-and-fix --------------------------------------------
def fig_collapse():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # GRPO: std-bug vs std-fixed (both still collapse, fix delays it)
    for logf, lbl, c in [("grpo_probe.log", "std bug", "#C44E52"),
                         ("grpo_probe_fixed.log", "std fixed", "#4C72B0")]:
        seg = parse_segments(logf)
        if seg:
            xs = [s for s, _, _ in seg[0][1]]
            ax1.plot(xs, [v for _, v, _ in seg[0][1]], marker="o", color=c, label=lbl)
    ax1.set(title="GRPO: still collapses (fix only delays)", xlabel="step", ylabel="solve rate")
    ax1.set_ylim(0.4, 1.0); ax1.grid(alpha=0.3); ax1.legend()
    # KTO: credit-bug vs credit-fixed
    kseg = {lbl: s for lbl, s in [("credit bug", parse_segments("calib_sweep.log")[-1][1]),
                                  ("credit fixed", parse_segments("kto_fixed2.log")[0][1])]}
    for lbl, c in [("credit bug", "#C44E52"), ("credit fixed", "#55A868")]:
        seg = kseg[lbl]
        xs = [s for s, _, _ in seg]
        ax2.plot(xs, [v for _, v, _ in seg], marker="o", color=c, label=lbl)
    ax2.set(title="KTO: credit fix resolves the collapse", xlabel="round", ylabel="solve rate")
    ax2.set_ylim(0.4, 1.0); ax2.grid(alpha=0.3); ax2.legend()
    fig.suptitle("Apparent 'collapses' were normalization / credit bugs", fontsize=11)
    fig.tight_layout(); fig.savefig(ART / "collapse_and_fix.png", dpi=130); plt.close(fig)


# ---- Figure 3: action-mix drift --------------------------------------------
def _mix_bins(traces, n_bins, per):
    bins = []
    for b in range(n_bins):
        chunk = traces[b * per:(b + 1) * per]
        c = {a: 0 for a in ACTIONS}
        for t in chunk:
            for d in t.get("decisions", []):
                if d.get("action") in c:
                    c[d["action"]] += 1
        tot = sum(c.values()) or 1
        bins.append({a: c[a] / tot for a in ACTIONS})
    return bins


def _stack(ax, bins, xlabel, title):
    x = np.arange(len(bins))
    bottom = np.zeros(len(bins))
    for a in ACTIONS:
        vals = np.array([b[a] for b in bins])
        ax.bar(x, vals, bottom=bottom, color=ACT_COLOR[a], label=a)
        bottom += vals
    ax.set(title=title, xlabel=xlabel, ylabel="action fraction", ylim=(0, 1))
    ax.set_xticks(x)


def fig_action_mix():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    kto = _read("calib_kto.jsonl")          # original (collapsing) KTO, 4 rounds x 66
    _stack(ax1, _mix_bins(kto, 4, 66), "round", "KTO (buggy): drifts to STOP")
    grpo = _read("grpo_probe_fixed.jsonl")  # std-fixed GRPO, ~10 step-bins of 64
    _stack(ax2, _mix_bins(grpo, 10, 64), "step", "GRPO (std-fixed): drifts to WIDER")
    h, l = ax1.get_legend_handles_labels()
    fig.suptitle("Collapse = one action taking over (the diagnostic that found each bug)",
                 y=0.99, fontsize=11)
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.94), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.88]); fig.savefig(ART / "action_mix_drift.png", dpi=130); plt.close(fig)


# ---- Figure 4: measurement rigor (CIs) -------------------------------------
def _final66(jsonl):
    return _read(jsonl)[-66:]


def _boot_mean_ci(xs, n=10000, seed=0):
    x = np.asarray(xs, float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def fig_cis():
    arms = {"bandit": _read("calib_bc.jsonl")[:66], "BC": _final66("calib_bc.jsonl"),
            "DPO": _final66("calib_dpo.jsonl"), "KTO": _final66("calib_kto_fixed2.jsonl")}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    names = list(arms); y = np.arange(len(names))
    # solve + Wilson CI
    for i, n in enumerate(names):
        ts = arms[n]; k = sum(_solved(t) for t in ts); ci = wilson_ci(k, len(ts))
        ax1.errorbar(ci.point, i, xerr=[[ci.point - ci.lo], [ci.hi - ci.point]],
                     fmt="o", color="#4C72B0", capsize=4)
    ax1.set(title="Solve rate (Wilson 95% CI, n=66 train)", xlabel="solve rate", yticks=y,
            yticklabels=names); ax1.set_xlim(0.4, 1.0); ax1.grid(alpha=0.3, axis="x")
    # cost + bootstrap CI
    for i, n in enumerate(names):
        m, lo, hi = _boot_mean_ci([_cost(t) for t in arms[n]])
        ax2.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color="#55A868", capsize=4)
    ax2.set(title="Mean cost (bootstrap 95% CI)", xlabel="mean cost ($)", yticks=y,
            yticklabels=names); ax2.grid(alpha=0.3, axis="x")
    fig.suptitle("At this n, learner differences sit within noise (cost is the metric with power)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(ART / "measurement_cis.png", dpi=130); plt.close(fig)


# ---- Figure 0 (headline): cost / solve frontier --------------------------
def _arm(evalfile, tag):
    ts = [t for t in _read(evalfile) if t.get("policy") == tag]
    k = sum(_solved(t) for t in ts)
    sc = wilson_ci(k, len(ts))
    m, lo, hi = _boot_mean_ci([_cost(t) for t in ts])
    return sc, (m, lo, hi)


def fig_frontier():
    """Headline: mean cost (x) vs solve rate (y). Lower-left-at-equal-height = better.
    Cold-start bandit vs the learned DPO policy at the chosen operating point
    (calib4 k=3: abstain-after-3, which holds solve at 0.84 while halving cost).
    Falls back to the calib3 BC/DPO/KTO frontier if calib4 traces are absent."""
    try:
        pts = {
            "bandit (cold start)": _arm("calib4_dpo_k3_eval.jsonl", "bandit@r0"),
            "DPO (learned, k=3)":  _arm("calib4_dpo_k3_eval.jsonl", "dpo@r3"),
        }
        dpo_key = "DPO (learned, k=3)"
        colors = {"bandit (cold start)": "#C44E52", dpo_key: "#4C72B0"}
        subtitle = "learned policy: same 0.84 solve, ~40% less cost (abstain-after-3)"
    except (FileNotFoundError, ZeroDivisionError):
        try:
            pts = {
                "bandit (cold start)": _arm("calib3_bc_eval.jsonl", "bandit@r0"),
                "BC":  _arm("calib3_bc_eval.jsonl", "bc@r3"),
                "DPO": _arm("calib3_dpo_eval.jsonl", "dpo@r3"),
                "KTO": _arm("calib3_kto_eval.jsonl", "kto@r3"),
            }
            dpo_key = "DPO"
            colors = {"bandit (cold start)": "#C44E52", "BC": "#8172B3",
                      "DPO": "#4C72B0", "KTO": "#55A868"}
            subtitle = "learned policy: ~half the cost at the same solve"
        except (FileNotFoundError, ZeroDivisionError):
            print("skip frontier: calib4/calib3 eval traces missing"); return
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (sc, (m, lo, hi)) in pts.items():
        ax.errorbar(m, sc.point, xerr=[[m - lo], [hi - m]],
                    yerr=[[sc.point - sc.lo], [sc.hi - sc.point]],
                    fmt="o", ms=9, capsize=4, color=colors[name], label=name)
        ax.annotate(name, (m, sc.point), textcoords="offset points", xytext=(8, 6), fontsize=9)
    b = pts["bandit (cold start)"]; d = pts[dpo_key]
    ax.annotate("", xy=(d[1][0], d[0].point), xytext=(b[1][0], b[0].point),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, ls="--"))
    ax.text(0.5, 0.06, subtitle,
            transform=ax.transAxes, ha="center", color="gray", fontsize=9)
    ax.set(xlabel="mean cost per task ($)", ylabel="solve rate (44-task eval)",
           title="Cost / solve frontier: learned cost-aware policy beats the cold start")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(ART / "cost_solve_frontier.png", dpi=130); plt.close(fig)


def fig_cascade():
    """Capstone: the weak->strong ESCALATE cascade. Cheap model (claude-3-haiku) only,
    vs the learned selective cascade (escalate the misses to Haiku-4.5), vs Haiku-4.5
    only. The cascade sits at the strong model's solve height but well left of it on
    cost -- the win is 'near-strong solve, far-below-strong cost, selectively'."""
    try:
        pts = {
            # cheap-only at its BEST achievable point (bandit@r0, 0.96) -- not the weaker
            # dpo@r2 (0.88) -- so the baseline is not cherry-picked low.
            "claude-3-haiku only":     _arm("casc3_A_c3h_eval.jsonl", "bandit@r0"),
            "learned cascade":         _arm("casc6_B_retry_eval.jsonl", "dpo@r2"),
            "Haiku-4.5 only (ceiling)": _arm("casc2_C_haikuonly_eval.jsonl", "dpo@r2"),
        }
    except (FileNotFoundError, ZeroDivisionError):
        print("skip cascade: calib5 eval traces missing"); return
    colors = {"claude-3-haiku only": "#C44E52", "learned cascade": "#4C72B0",
              "Haiku-4.5 only (ceiling)": "#55A868"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (sc, (m, lo, hi)) in pts.items():
        ax.errorbar(m, sc.point, xerr=[[m - lo], [hi - m]],
                    yerr=[[sc.point - sc.lo], [sc.hi - sc.point]],
                    fmt="o", ms=9, capsize=4, color=colors[name], label=name)
        ax.annotate(name, (m, sc.point), textcoords="offset points", xytext=(8, 6), fontsize=9)
    c = pts["Haiku-4.5 only (ceiling)"]; d = pts["learned cascade"]
    ax.annotate("", xy=(d[1][0], d[0].point), xytext=(c[1][0], c[0].point),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, ls="--"))
    ax.text(0.5, 0.06, "cascade: matches strong-model solve at 36-47% lower mean cost (2 seeds)",
            transform=ax.transAxes, ha="center", color="gray", fontsize=9)
    ax.set(xlabel="mean cost per task ($)", ylabel="solve rate (24-task eval)",
           title="Weak->strong cascade: selective ESCALATE reaches strong solve, cheaper")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(ART / "cascade_frontier.png", dpi=130); plt.close(fig)


def main():
    fig_cascade(); print("wrote cascade_frontier.png")
    fig_frontier(); print("wrote cost_solve_frontier.png")
    fig_curves(); print("wrote self_improvement_curves.png")
    fig_collapse(); print("wrote collapse_and_fix.png")
    fig_action_mix(); print("wrote action_mix_drift.png")
    fig_cis(); print("wrote measurement_cis.png")


if __name__ == "__main__":
    main()
