"""Offline tests for the text-to-SQL benchmark (fixture db, no model calls)."""
from __future__ import annotations

import sqlite3

import pytest

from wdp.benchmarks.sql import SqlBenchmark, SqlVerifier, _run_select


class _T:  # minimal task stand-in carrying metadata
    def __init__(self, m):
        self.metadata = m


def _make_db(tmp_path) -> str:
    db = tmp_path / "shop.sqlite"
    con = sqlite3.connect(str(db))
    con.executescript(
        "CREATE TABLE items (id INTEGER, name TEXT, price INTEGER);"
        "INSERT INTO items VALUES (1,'a',10),(2,'b',20),(3,'c',20);")
    con.commit()
    con.close()
    return str(db)


def test_run_select_runs_selects_and_rejects_writes(tmp_path):
    db = _make_db(tmp_path)
    assert _run_select(db, "SELECT count(*) FROM items", 200) == [(3,)]
    for bad in ("DROP TABLE items", "UPDATE items SET price=0",
                "SELECT 1; SELECT 2", "DELETE FROM items"):
        with pytest.raises(Exception):
            _run_select(db, bad, 200)


def test_execution_match_grades_results_not_sql_text(tmp_path):
    db = _make_db(tmp_path)
    bench = SqlBenchmark(db, [{"question": "how many items cost 20?",
                               "gold_sql": "SELECT count(*) FROM items WHERE price=20"}])
    task = bench.tasks()[0]
    v = bench.terminal_verifier()
    # different SQL, same result set -> match
    assert v.score_final(task, "SELECT COUNT(id) FROM items WHERE price = 20").value == 1.0
    assert v.score_final(task, "```sql\nSELECT count(*) FROM items WHERE price=20\n```").value == 1.0
    # wrong result, erroring SQL, and a non-SELECT all score 0
    assert v.score_final(task, "SELECT count(*) FROM items WHERE price = 10").value == 0.0
    assert v.score_final(task, "SELECT * FROM nope").value == 0.0
    assert v.score_final(task, "DROP TABLE items").value == 0.0
    assert v.score_final(task, "no sql here").value == 0.0


def test_order_matters_only_when_gold_has_order_by(tmp_path):
    db = _make_db(tmp_path)
    v = SqlBenchmark(db, []).terminal_verifier()
    ordered = {"db_path": db, "gold_sql": "SELECT name FROM items ORDER BY price"}
    unordered = {"db_path": db, "gold_sql": "SELECT name FROM items"}

    class T:  # minimal task stand-in carrying metadata
        def __init__(self, m): self.metadata = m

    assert v.score_final(T(ordered), "SELECT name FROM items ORDER BY price DESC").value == 0.0
    assert v.score_final(T(ordered), "SELECT name FROM items ORDER BY price").value == 1.0
    # no ORDER BY in gold -> compared as a set, order-insensitive
    assert v.score_final(T(unordered), "SELECT name FROM items ORDER BY name DESC").value == 1.0


def test_grader_is_type_and_null_aware(tmp_path):
    db = _make_db(tmp_path)
    v = SqlBenchmark(db, []).terminal_verifier()
    # int 20 vs text '20' must NOT execution-match (the old str() collapse called them equal)
    ti = _T({"db_path": db, "gold_sql": "SELECT 20"})
    assert v.score_final(ti, "SELECT 20").value == 1.0
    assert v.score_final(ti, "SELECT '20'").value == 0.0
    # NULL vs '' must NOT match
    tn = _T({"db_path": db, "gold_sql": "SELECT NULL"})
    assert v.score_final(tn, "SELECT NULL").value == 1.0
    assert v.score_final(tn, "SELECT ''").value == 0.0


def test_row_limit_overflow_is_a_failure_not_a_prefix_match(tmp_path):
    db = _make_db(tmp_path)  # items has 3 rows
    v = SqlVerifier(db, row_limit=2)
    t = _T({"db_path": db, "gold_sql": "SELECT id FROM items"})  # 3 rows > limit
    assert v.score_final(t, "SELECT id FROM items").value == 0.0  # overflow -> graded failure
    with pytest.raises(Exception):
        _run_select(db, "SELECT id FROM items", 2)


def test_semicolon_inside_a_string_literal_is_allowed(tmp_path):
    db = _make_db(tmp_path)
    assert _run_select(db, "SELECT 'a;b'", 200) == [("a;b",)]


def test_tasks_carry_schema_and_gold(tmp_path):
    db = _make_db(tmp_path)
    bench = SqlBenchmark(db, [{"question": "q", "gold_sql": "SELECT 1", "difficulty": "simple"}])
    t = bench.tasks()[0]
    assert "items(" in t.prompt and "Question: q" in t.prompt
    assert t.metadata["gold_sql"] == "SELECT 1" and t.metadata["difficulty"] == "simple"
