"""Build a small, self-contained text-to-SQL dataset for the SQL benchmark.

Creates a deterministic e-commerce SQLite database and a graded question set
(simple / moderate / challenging) with gold SQL, then verifies every gold query
executes. Outputs:
  data/sql/shop.sqlite   (the database; gitignored, regenerate with this script)
  data/sql/tasks.json    (list of {question, gold_sql, difficulty})

Run on the suite with:
  python scripts/run_selfimprove.py --benchmark sql \
      --sql-db data/sql/shop.sqlite --sql-tasks data/sql/tasks.json \
      --learner dpo --rounds 2 --budget 0.03 --max-decisions 6
"""
from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

OUT_DIR = Path("data/sql")
DB = OUT_DIR / "shop.sqlite"
TASKS = OUT_DIR / "tasks.json"

CITIES = ["Athens", "London", "Luxembourg", "San Francisco", "Berlin"]
CATEGORIES = ["books", "games", "tools", "food"]
STATUSES = ["placed", "shipped", "delivered", "cancelled"]


def build_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = sqlite3.connect(str(path))
    con.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT, signup_date TEXT);
        CREATE TABLE products  (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price INTEGER);
        CREATE TABLE orders    (id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, status TEXT);
        CREATE TABLE order_items(id INTEGER PRIMARY KEY, order_id INTEGER, product_id INTEGER, quantity INTEGER);
        """
    )
    rng = random.Random(0)
    customers = [(i, f"cust{i:02d}", rng.choice(CITIES), f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}")
                 for i in range(1, 21)]                                   # 20 customers
    products = [(i, f"prod{i:02d}", CATEGORIES[(i - 1) % len(CATEGORIES)], rng.choice([5, 9, 12, 20, 35, 50]))
                for i in range(1, 16)]                                    # 15 products
    orders = [(i, rng.randint(1, 18),                                    # custs 19,20 never order
               f"2026-{rng.randint(1,5):02d}-{rng.randint(1,28):02d}", rng.choice(STATUSES))
              for i in range(1, 61)]                                     # 60 orders
    items, iid = [], 1
    for oid, *_ in orders:
        for _ in range(rng.randint(1, 4)):
            items.append((iid, oid, rng.randint(1, 13), rng.randint(1, 5)))  # prods 14,15 never ordered
            iid += 1
    con.executemany("INSERT INTO customers VALUES (?,?,?,?)", customers)
    con.executemany("INSERT INTO products VALUES (?,?,?,?)", products)
    con.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
    con.executemany("INSERT INTO order_items VALUES (?,?,?,?)", items)
    con.commit()
    con.close()


def questions() -> list[dict]:
    """Graded questions with gold SQL. Kept unambiguous (clear aggregates, explicit
    ORDER BY + LIMIT for 'top' questions) so execution-match is well-defined."""
    return [
        # --- simple: single table, filter/aggregate ---
        {"difficulty": "simple", "question": "How many customers are there?",
         "gold_sql": "SELECT count(*) FROM customers"},
        {"difficulty": "simple", "question": "How many products are in the 'books' category?",
         "gold_sql": "SELECT count(*) FROM products WHERE category='books'"},
        {"difficulty": "simple", "question": "List the names of customers in London.",
         "gold_sql": "SELECT name FROM customers WHERE city='London'"},
        {"difficulty": "simple", "question": "How many orders have status 'cancelled'?",
         "gold_sql": "SELECT count(*) FROM orders WHERE status='cancelled'"},
        {"difficulty": "simple", "question": "What is the most expensive product price?",
         "gold_sql": "SELECT max(price) FROM products"},
        {"difficulty": "simple", "question": "How many distinct cities do customers come from?",
         "gold_sql": "SELECT count(DISTINCT city) FROM customers"},
        {"difficulty": "simple", "question": "List product names that cost more than 20.",
         "gold_sql": "SELECT name FROM products WHERE price>20"},
        {"difficulty": "simple", "question": "How many products are there per category? Return category and the count.",
         "gold_sql": "SELECT category, count(*) FROM products GROUP BY category"},
        # --- moderate: joins / group-by ---
        {"difficulty": "moderate", "question": "How many orders did each customer place? Return customer name and order count, for customers who placed at least one order.",
         "gold_sql": "SELECT c.name, count(*) FROM customers c JOIN orders o ON o.customer_id=c.id GROUP BY c.id"},
        {"difficulty": "moderate", "question": "What is the total quantity sold per product? Return product name and total quantity.",
         "gold_sql": "SELECT p.name, sum(i.quantity) FROM products p JOIN order_items i ON i.product_id=p.id GROUP BY p.id"},
        {"difficulty": "moderate", "question": "How many orders are there for each status?",
         "gold_sql": "SELECT status, count(*) FROM orders GROUP BY status"},
        {"difficulty": "moderate", "question": "What is the average product price per category? Return category and average price.",
         "gold_sql": "SELECT category, avg(price) FROM products GROUP BY category"},
        {"difficulty": "moderate", "question": "Which city has the most customers? Return the city and the count.",
         "gold_sql": "SELECT city, count(*) FROM customers GROUP BY city ORDER BY count(*) DESC LIMIT 1"},
        {"difficulty": "moderate", "question": "How many order items belong to delivered orders?",
         "gold_sql": "SELECT count(*) FROM order_items i JOIN orders o ON o.id=i.order_id WHERE o.status='delivered'"},
        # --- challenging: multi-join revenue / subqueries / having ---
        {"difficulty": "challenging", "question": "Which product generated the most revenue (sum of quantity times price)? Return the product name.",
         "gold_sql": "SELECT p.name FROM products p JOIN order_items i ON i.product_id=p.id GROUP BY p.id ORDER BY sum(i.quantity*p.price) DESC LIMIT 1"},
        {"difficulty": "challenging", "question": "Which customer spent the most in total (sum of quantity times price across their order items)? Return the customer name.",
         "gold_sql": "SELECT c.name FROM customers c JOIN orders o ON o.customer_id=c.id JOIN order_items i ON i.order_id=o.id JOIN products p ON p.id=i.product_id GROUP BY c.id ORDER BY sum(i.quantity*p.price) DESC LIMIT 1"},
        {"difficulty": "challenging", "question": "List the names of products that were never ordered.",
         "gold_sql": "SELECT name FROM products WHERE id NOT IN (SELECT DISTINCT product_id FROM order_items)"},
        {"difficulty": "challenging", "question": "List the names of customers who never placed an order.",
         "gold_sql": "SELECT name FROM customers WHERE id NOT IN (SELECT DISTINCT customer_id FROM orders)"},
        {"difficulty": "challenging", "question": "Total revenue (sum of quantity times price) per category, ordered by revenue descending. Return category and revenue.",
         "gold_sql": "SELECT p.category, sum(i.quantity*p.price) FROM products p JOIN order_items i ON i.product_id=p.id GROUP BY p.category ORDER BY sum(i.quantity*p.price) DESC"},
        {"difficulty": "challenging", "question": "Which customers placed more than 3 orders? Return the customer name and order count.",
         "gold_sql": "SELECT c.name, count(*) FROM customers c JOIN orders o ON o.customer_id=c.id GROUP BY c.id HAVING count(*)>3"},
    ]


def main() -> None:
    build_db(DB)
    specs = questions()
    # verify every gold SQL executes against the db
    con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    bad = []
    for s in specs:
        try:
            con.execute(s["gold_sql"]).fetchall()
        except Exception as e:                                          # noqa: BLE001
            bad.append((s["question"], str(e)))
    con.close()
    if bad:
        for q, e in bad:
            print("BAD GOLD:", q, "->", e)
        raise SystemExit(f"{len(bad)} gold queries failed; fix before writing tasks")
    TASKS.write_text(json.dumps(specs, indent=2), encoding="utf-8")
    by = {}
    for s in specs:
        by[s["difficulty"]] = by.get(s["difficulty"], 0) + 1
    print(f"wrote {DB} and {TASKS}: {len(specs)} tasks {by}")


if __name__ == "__main__":
    main()
