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
                 seed: int = 0) -> None:
        self.n_atomic = n_atomic
        self.n_multi = n_multi
        self.n_underspecified = n_underspecified
        self.seed = seed

    def tools(self) -> dict[str, Tool]:
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
            r = lambda: rng.randint(2, 20)
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
