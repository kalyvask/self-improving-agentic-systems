"""Text-to-SQL benchmark with a FREE execution-match reward.

A real task (natural-language question -> SQL over a database) graded for $0 and
deterministically: run the predicted SQL and the gold SQL against the same SQLite
database and compare result sets. That is exactly the terminal signal the controller
needs, and the agent's only billable calls are its own model calls -- the SQL runs
locally, so the cost ledger is clean (no off-ledger user simulator like tau-bench).

v1 is SINGLE-DATABASE: all questions share one db, so the read-only SQL tools can be
bound globally (the generic Executor's tools are fixed at construction). Multi-db
suites (e.g. full BIRD) need per-task tool binding and are out of scope here.

The actions map naturally: list_tables/describe_table -> DECOMPOSE (explore schema,
then compose); refine a query after an error -> DEEPER; a fresh rewrite -> WIDER; a
hard schema -> ESCALATE.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from wdp.executor.react import Task, Tool, make_tool
from wdp.verifier.scorer import Score

_SELECT_RE = re.compile(r"^\s*select\b", re.IGNORECASE)
_ORDERBY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)


def _connect_ro(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    con.execute("PRAGMA query_only = 1")
    # abort runaway queries (~1e7 vdbe steps) so a bad SELECT can't hang the eval.
    steps = {"n": 0}
    def _guard():
        steps["n"] += 1
        return 1 if steps["n"] > 200_000 else 0
    con.set_progress_handler(_guard, 100_000)
    return con


def _run_select(db_path: str, sql: str, row_limit: int) -> list[tuple]:
    """Run a single read-only SELECT, returning up to row_limit rows. Raises on any
    non-SELECT, multiple statements, or execution error."""
    s = (sql or "").strip().rstrip(";")
    if not _SELECT_RE.match(s):
        raise ValueError("only a single SELECT statement is allowed")
    if ";" in s:
        raise ValueError("multiple statements are not allowed")
    con = _connect_ro(db_path)
    try:
        return list(con.execute(s).fetchmany(row_limit))
    finally:
        con.close()


def _canon(rows: list[tuple], ordered: bool) -> list[tuple]:
    norm = [tuple("" if c is None else str(c) for c in row) for row in rows]
    return norm if ordered else sorted(norm)


def _extract_sql(answer: str) -> str:
    """Pull the SQL out of the agent's final answer (strip code fences; take from the
    first SELECT to the end)."""
    if not answer:
        return ""
    a = answer.replace("```sql", "```").replace("```", " ")
    m = _SELECT_RE.search(a) or re.search(r"select\b", a, re.IGNORECASE)
    return a[m.start():].strip() if m else ""


def _schema_summary(db_path: str) -> str:
    con = _connect_ro(db_path)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        lines = []
        for t in tables:
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
            lines.append(f"{t}({', '.join(cols)})")
        return "\n".join(lines)
    finally:
        con.close()


def _render(rows: list[tuple], row_limit: int) -> str:
    if not rows:
        return "(0 rows)"
    shown = rows[:20]
    body = "\n".join(" | ".join("" if c is None else str(c) for c in r) for r in shown)
    more = f"\n... ({len(rows)} rows{'+' if len(rows) >= row_limit else ''})" if len(rows) > 20 else ""
    return body + more


class SqlVerifier:
    """Terminal verifier: execution match (EX). 1.0 iff the predicted SQL's result set
    equals the gold SQL's on the same db (order-sensitive only if gold has ORDER BY)."""

    def __init__(self, db_path: str, row_limit: int = 200) -> None:
        self.db_path = db_path
        self.row_limit = row_limit

    def score_final(self, task: Task, answer: str) -> Score:
        db = task.metadata.get("db_path", self.db_path)
        gold = task.metadata.get("gold_sql")
        pred = _extract_sql(answer)
        if not pred:
            return Score(value=0.0, rationale="no SQL in answer")
        try:
            g = _run_select(db, gold, self.row_limit)
        except Exception as e:                          # a bad gold is a dataset bug
            return Score(value=0.0, rationale=f"gold SQL failed: {e}")
        try:
            p = _run_select(db, pred, self.row_limit)
        except Exception as e:
            return Score(value=0.0, rationale=f"predicted SQL error: {e}")
        ordered = bool(_ORDERBY_RE.search(gold or ""))
        ok = _canon(p, ordered) == _canon(g, ordered)
        return Score(value=1.0 if ok else 0.0,
                     rationale="execution match" if ok else "result set mismatch")

    def score_abstention(self, task: Task) -> Score:
        return Score(value=0.0, rationale="SQL tasks are solvable; STOP is not correct")


class SqlBenchmark:
    """Single-database text-to-SQL suite. `specs` is a list of
    {question, gold_sql, difficulty?}; all share `db_path`."""

    name = "sql"

    def __init__(self, db_path: str, specs: list[dict], *, row_limit: int = 200,
                 seed: int = 0) -> None:
        self.db_path = str(db_path)
        self.specs = list(specs)
        self.row_limit = row_limit

    def tools(self) -> dict[str, Tool]:
        db, lim = self.db_path, self.row_limit

        def run_sql(query: str = "") -> str:
            try:
                return _render(_run_select(db, query, lim), lim)
            except Exception as e:
                return f"ERROR: {type(e).__name__}: {e}"

        def list_tables() -> str:
            return _schema_summary(db)

        def describe_table(table: str = "") -> str:
            con = _connect_ro(db)
            try:
                rows = con.execute(f"PRAGMA table_info({table})").fetchall()
                if not rows:
                    return f"no such table: {table}"
                return "\n".join(f"{r[1]} {r[2]}" for r in rows)
            except Exception as e:
                return f"ERROR: {e}"
            finally:
                con.close()

        ts = [
            make_tool("run_sql", 'Run a read-only SELECT, e.g. run_sql with '
                      '{"query": "SELECT count(*) FROM t"}.', run_sql),
            make_tool("list_tables", "List tables and their columns.", list_tables),
            make_tool("describe_table", 'Columns+types of a table, e.g. '
                      '{"table": "users"}.', describe_table),
        ]
        return {t.name: t for t in ts}

    def terminal_verifier(self) -> SqlVerifier:
        return SqlVerifier(self.db_path, self.row_limit)

    def tasks(self) -> list[Task]:
        schema = _schema_summary(self.db_path)
        out: list[Task] = []
        for i, s in enumerate(self.specs):
            prompt = (
                f"Database schema:\n{schema}\n\nQuestion: {s['question']}\n"
                "Write a single SQLite SELECT that answers the question. Use the "
                "run_sql / list_tables / describe_table tools to check it, then FINISH "
                "with the final SELECT statement as your answer.")
            out.append(Task(
                id=f"sql-{i}", prompt=prompt,
                metadata={"db_path": self.db_path, "gold_sql": s["gold_sql"],
                          "difficulty": s.get("difficulty", "?"),
                          # schema-exploration is a real first move here, so DECOMPOSE
                          # is plausible (probe value, not 0); not strongly separable.
                          "decomposability": 0.3}))
        return out
