"""A self-contained, locally-checkable benchmark for end-to-end loop runs.

Real benchmarks cost credits and external setup; this one costs neither for its
verifier. Tasks are arithmetic word-problems with a `calc` tool and a gold answer
computed locally, so the TerminalVerifier is exact and free. The suite is built
to exercise every action:

  - atomic single-expression tasks reward WIDER/DEEPER over DECOMPOSE,
  - multi-part tasks ("compute A, then B, then combine") are genuinely
    decomposable, so DECOMPOSE should pay off there,
  - a few intentionally underspecified tasks have no checkable answer, so the
    only good move is STOP (a correct abstention).

The Executor still spends real model tokens to solve them; only the grading is
free. That keeps a full self-improvement run cheap enough to iterate on a laptop.
"""
from __future__ import annotations

import ast
import operator
import random
import re

from wdp.executor.react import Task, Tool, make_tool
from wdp.verifier.scorer import Score

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.USub: operator.neg, ast.Pow: operator.pow,
}


def safe_eval(expr: str) -> float:
    """Evaluate a pure-arithmetic expression with no names, calls, or attributes."""
    def _ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_ev(node.operand))
        raise ValueError("unsupported expression")
    return float(_ev(ast.parse(expr, mode="eval").body))


def _calc(expr: str = "") -> str:
    try:
        return str(safe_eval(expr))
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def _num(text: str) -> float | None:
    m = re.findall(r"[-+]?\d*\.?\d+", (text or "").replace(",", ""))
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None


def _hard_problem(rng: random.Random) -> tuple[str, int]:
    """A multi-hop word problem: tangled prose, distractors, a final conditional.
    Returns (prompt, exact-integer gold). The arithmetic is calc-trivial; the
    difficulty is reading it correctly. See ArithmeticBenchmark.tasks (hard kind)."""
    val = rng.randint(10, 40)
    lines = [f"A counter starts at {val}."]
    # varied phrasings for the same op so surface form can't be pattern-matched
    add_p = ["Then it increases by {n}.", "Next, add {n} to it.", "After that it grows by {n}."]
    sub_p = ["Then it decreases by {n}.", "Next, take away {n}.", "After that it drops by {n}."]
    mul_p = ["Then it is multiplied by {n}.", "Next, scale it by a factor of {n}."]
    dbl_p = ["Then the counter doubles.", "Next, it is twice as large."]
    tpl_p = ["Then the counter triples.", "After that it becomes three times as big."]
    distract = ["(A nearby sign reads {d}, but it has nothing to do with the counter.)",
                "(Ignore the {d} birds on the wire.)",
                "(There are {d} chairs in the room; this is irrelevant.)"]
    n_ops = rng.randint(4, 6)
    for _ in range(n_ops):
        k = rng.choice(["add", "sub", "mul", "double", "triple"])
        if k == "add":
            n = rng.randint(3, 20); lines.append(rng.choice(add_p).format(n=n)); val += n
        elif k == "sub":
            n = rng.randint(3, 20); lines.append(rng.choice(sub_p).format(n=n)); val -= n
        elif k == "mul":
            n = rng.randint(2, 4); lines.append(rng.choice(mul_p).format(n=n)); val *= n
        elif k == "double":
            lines.append(rng.choice(dbl_p)); val *= 2
        else:
            lines.append(rng.choice(tpl_p)); val *= 3
        if rng.random() < 0.5:
            lines.append(rng.choice(distract).format(d=rng.randint(5, 99)))
    t, h = rng.randint(50, 150), rng.randint(5, 20)
    lines.append(f"Finally, if the total is greater than {t}, subtract {h}; otherwise add {h}.")
    val = val - h if val > t else val + h
    prompt = (" ".join(lines) +
              " Use the calc tool for the arithmetic, then FINISH with the final integer.")
    return prompt, int(val)


# Chain-length bands for the no-calc tier. The difficulty is carrying an accurate
# running total over N steps with no scratchpad tool, so longer chain = harder. The
# bands are chosen so a weak cheap model (Llama-3.1-8B) spans roughly easy~0.9 /
# medium~0.6 / hard~0.25 while the strong target (Haiku-4.5) stays near-ceiling --
# giving the controller a LEARNABLE escalation signal (escalate the long ones) rather
# than a uniform coin-flip where escalation can only fire at random.
_NOCALC_BANDS = {"easy": (2, 3), "medium": (5, 6), "hard": (9, 12)}


def _hard_no_calc_problem(rng: random.Random,
                          difficulty: str = "hard") -> tuple[str, int, int]:
    """A chain of SMALL-number operations done WITHOUT a calculator, ending in a
    conditional. Every step is trivial; difficulty is tracking the running total over
    `difficulty`-many steps with no tool. Returns (prompt, exact-integer gold, n_ops)."""
    lo, hi = _NOCALC_BANDS[difficulty]
    val = rng.randint(5, 15)
    lines = [f"Start with the number {val}. Do the arithmetic in your head, "
             f"step by step, with no tools."]
    # bias heavily to +/- of small numbers; allow only occasional doubling so the
    # running total stays bounded (~under 150) and the load is tracking, not products.
    add_p = ["Add {n}.", "Increase it by {n}.", "Then add {n}.", "Now add {n} to it."]
    sub_p = ["Subtract {n}.", "Decrease it by {n}.", "Then take away {n}.", "Now subtract {n}."]
    dbl_p = ["Double it.", "Now multiply it by 2.", "Then double the result."]
    n_ops = rng.randint(lo, hi)
    for _ in range(n_ops):
        k = rng.choice(["add", "add", "add", "sub", "sub", "double"])
        if k == "add":
            n = rng.randint(2, 12); lines.append(rng.choice(add_p).format(n=n)); val += n
        elif k == "sub":
            n = rng.randint(2, 12); lines.append(rng.choice(sub_p).format(n=n)); val -= n
        else:
            lines.append(rng.choice(dbl_p)); val *= 2
    t, h = rng.randint(40, 120), rng.randint(3, 12)
    lines.append(f"Finally, if the result is greater than {t}, subtract {h}; otherwise add {h}.")
    val = val - h if val > t else val + h
    return " ".join(lines) + " FINISH with the final integer only.", int(val), n_ops


class ArithmeticVerifier:
    """Terminal verifier: pass when the answer's last number matches gold.

    Underspecified tasks (gold is None) are scored 0.0 for any answer, so the
    only way to do well on them is to STOP -- which assign_credit rewards."""

    def score_final(self, task: Task, answer: str) -> Score:
        gold = task.metadata.get("gold")
        if gold is None:
            return Score(value=0.0, rationale="underspecified: no checkable answer")
        got = _num(answer)
        if got is None:
            return Score(value=0.0, rationale="no number in answer")
        return Score(value=1.0 if abs(got - float(gold)) < 1e-6 else 0.0)

    def score_abstention(self, task: Task) -> Score:
        """Grade a STOP/abstention against ground-truth solvability. A correct
        abstention (task has no checkable answer) scores 1.0; giving up on a
        solvable task scores 0.0. This is the ground-truth signal assign_credit
        needs to tell a right abstention from a premature one."""
        unsolvable = task.metadata.get("gold") is None
        return Score(value=1.0 if unsolvable else 0.0,
                     rationale="correct abstention" if unsolvable
                     else "gave up on a solvable task")


class ArithmeticBenchmark:
    name = "arithmetic"

    def __init__(self, n_atomic: int = 8, n_multi: int = 6, n_underspecified: int = 2,
                 n_hard: int = 0, no_calc: bool = False, seed: int = 0) -> None:
        self.n_atomic = n_atomic
        self.n_multi = n_multi
        self.n_underspecified = n_underspecified
        self.n_hard = n_hard
        # no_calc: withhold the calculator tool AND switch the hard tier to the
        # small-number / long-chain variant. This manufactures a CONTROLLED capability
        # gap (careful step-tracking, not hard multiplication) to validate the ESCALATE
        # machinery -- a strong model tracks a long chain reliably; a weak one slips.
        # It is a mechanism test bed, not the headline thesis (tau-bench is that).
        self.no_calc = no_calc
        self.seed = seed

    def tools(self) -> dict[str, Tool]:
        if self.no_calc:
            return {}      # force in-context computation: the capability stressor
        calc = make_tool("calc", "Evaluate an arithmetic expression, e.g. "
                                 'calc with action_input {"expr": "12*(7+5)"}.', _calc)
        return {calc.name: calc}

    def terminal_verifier(self) -> ArithmeticVerifier:
        return ArithmeticVerifier()

    def tasks(self) -> list[Task]:
        rng = random.Random(self.seed)
        out: list[Task] = []

        # Atomic tasks span a difficulty gradient (tier 0-3): a flat eval looks the
        # same at every difficulty and tells the WIDER-vs-DEEPER split nothing, so
        # we vary nesting depth. Tier rises with i so a large suite is graded, not
        # uniform. Only +,-,* keep golds exact integers (no rounding ambiguity).
        # Tier 3 adds a third nesting level so one careless WIDER pass is unlikely to
        # track it -- that is where DEEPER refinement should start to pay off.
        for i in range(self.n_atomic):
            def r() -> int:
                return rng.randint(2, 20)
            tier = i % 4
            if tier == 0:                                   # easy: one operation
                expr = rng.choice([f"{r()} + {r()}", f"{r()} * {r()}"])
            elif tier == 1:                                 # medium: one nesting
                expr = f"{r()} * ({r()} + {r()})"
            elif tier == 2:                                 # hard: two sub-expressions
                expr = f"({r()} + {r()}) * ({r()} - {r()}) + {r()}"
            else:                                           # very hard: three levels
                expr = f"(({r()} + {r()}) * {r()} - {r()}) * ({r()} + {r()})"
            out.append(Task(
                id=f"atomic-{i}",
                prompt=f"Compute {expr}. Use the calc tool, then FINISH with the integer.",
                metadata={"gold": safe_eval(expr), "kind": "atomic", "tier": tier,
                          "decomposability": 0.0},   # single expression: do not decompose
            ))

        # Multi-part tasks vary in how many sub-results must be combined (2-5), so
        # DECOMPOSE has a real, graded payoff that grows with the part count. At 5
        # parts a single trajectory tends to drop or misadd one term, so decomposing
        # into independent sub-tasks becomes structurally the better move -- giving
        # the controller a regime where DECOMPOSE genuinely beats WIDER/DEEPER.
        for i in range(self.n_multi):
            n_parts = 2 + (i % 4)                           # 2, 3, 4, or 5 parts
            parts = [f"{rng.randint(2, 15)} * {rng.randint(2, 15)}"
                     for _ in range(n_parts)]
            gold = safe_eval(" + ".join(f"({p})" for p in parts))
            steps = "; ".join(f"compute {p}" for p in parts)
            out.append(Task(
                id=f"multi-{i}",
                prompt=(f"Separately {steps}. Then FINISH with the sum of all "
                        f"{n_parts} results."),
                metadata={"gold": gold, "kind": "multi", "n_parts": n_parts,
                          # graded by part count so the policy sees "more parts =
                          # more decomposable": 2->0.25 .. 5->1.0
                          "decomposability": round(min(1.0, (n_parts - 1) / 4.0), 2)},
            ))

        # HARD tasks: the arithmetic is trivial (calc does it), but the PROSE is
        # tangled -- a chain of 4-6 sequential transformations in varied natural
        # language, salted with distractor numbers that must be ignored, ending in a
        # conditional that depends on the running total. The capability bottleneck is
        # translation/reasoning, not computation, so a weak model mis-structures the
        # chain (drops a step, applies the wrong number, gets the conditional branch
        # wrong) where a stronger model does not. Marked decomposability 0.0: these
        # are NOT separable, so the ONLY lever that lifts them is a more capable model
        # (ESCALATE) -- which is exactly the regime the capstone needs. Gold is an
        # exact integer computed by the same sequential semantics described in prose.
        # no_calc hard tier is GRADED across easy/medium/hard chain lengths (cycled by
        # index) so the cheap model's solve rate correlates with a real difficulty axis
        # -- the basis for SELECTIVE escalation. The calc hard tier stays uniform.
        bands = ("easy", "medium", "hard")
        for i in range(self.n_hard):
            if self.no_calc:
                diff = bands[i % 3]
                prompt, gold, n_ops = _hard_no_calc_problem(rng, diff)
                meta = {"gold": gold, "kind": "hard", "decomposability": 0.0,
                        "no_calc": True, "difficulty": diff, "n_ops": n_ops}
            else:
                prompt, gold = _hard_problem(rng)
                meta = {"gold": gold, "kind": "hard", "decomposability": 0.0,
                        "no_calc": False}
            out.append(Task(id=f"hard-{i}", prompt=prompt, metadata=meta))

        for i in range(self.n_underspecified):
            out.append(Task(
                id=f"underspecified-{i}",
                prompt=("Compute the value of x. (No value of x is given anywhere.) "
                        "If it cannot be determined, STALL."),
                metadata={"gold": None, "kind": "underspecified", "decomposability": 0.0},
            ))

        rng.shuffle(out)
        return out


def split(tasks: list[Task], frac_train: float = 0.6, seed: int = 0):
    """Deterministic train/eval split."""
    rng = random.Random(seed)
    idx = list(range(len(tasks)))
    rng.shuffle(idx)
    cut = int(round(frac_train * len(tasks)))
    train = [tasks[i] for i in idx[:cut]]
    eval_ = [tasks[i] for i in idx[cut:]]
    return train, eval_
