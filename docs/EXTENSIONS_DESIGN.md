# Design: three extensions (SQL benchmark, response cache, policy persistence)

Design notes for three additions that (a) put the controller on a real task with a
free reward, (b) make repeated runs cheap and deterministic, and (c) let a trained
policy be saved and deployed without retraining. All three slot into the existing
interfaces (`Benchmark`, `Executor`, `TerminalVerifier`, `OpenRouterClient`,
`Allocator`) with no change to the loop driver.

---

## 1. Text-to-SQL benchmark with a free execution-match reward

### Why
Arithmetic is synthetic and tau-bench is expensive, noisy, and has no cheap grader.
A text-to-SQL suite (e.g. BIRD/Spider-style) is a **real task with a free,
deterministic reward**: run the predicted SQL and the gold SQL against the same
database and compare result sets (execution match, "EX"). That is exactly the
terminal signal the controller needs, computed locally for $0, and the agent's only
billable calls are its own model calls -- so the cost ledger is clean (no off-ledger
user simulator like tau). It also exercises every action naturally (schema
exploration -> DECOMPOSE, query refinement on an error -> DEEPER, a fresh rewrite ->
WIDER, abstain on an unanswerable schema -> STOP, hard schema -> ESCALATE).

### Interface (mirrors `ArithmeticBenchmark`)
`src/wdp/benchmarks/sql.py`:

```python
class SqlBenchmark:                       # implements the Benchmark protocol
    name = "sql"
    def __init__(self, data_dir, split="dev", n_tasks=40, difficulty=None, seed=0): ...
    def tasks(self) -> list[Task]: ...     # one Task per (question, db)
    def tools(self) -> dict[str, Tool]: ...# read-only SQL tools (below)
    def terminal_verifier(self) -> "SqlVerifier": ...

class SqlVerifier:                         # implements TerminalVerifier
    def score_final(self, task, answer) -> Score    # EX match: 1.0 / 0.0
    def score_abstention(self, task) -> Score       # solvable -> 0.0 (STOP rarely right)
```

`Task.metadata = {"db_id", "db_path", "gold_sql", "difficulty"}`; `Task.prompt` is the
natural-language question plus a compact schema summary (table/column names). The
agent's `final_answer` is the predicted **SQL string**.

### Tools (read-only, billed nowhere -- local SQLite)
- `run_sql(query)` -> first N rows or `ERROR: <message>`. Opened read-only
  (`file:<db>?mode=ro`, `PRAGMA query_only=1`); reject anything that is not a single
  `SELECT` (defense in depth, even on a read-only handle).
- `list_tables()` / `describe_table(name)` -> schema exploration (this is what makes
  DECOMPOSE meaningful: explore schema, then compose the query).

### Reward (execution match)
`score_final` executes the agent's SQL and the gold SQL on the **same** db and
compares result sets:
- run both read-only with a wall/time guard; the agent's SQL erroring -> 0.0.
- canonicalize rows before comparing: cast each row to a tuple of stringified cells,
  compare as **multisets** unless the gold query has a top-level `ORDER BY` (then
  compare as ordered lists). This is the standard EX rule and avoids spurious
  column-order / row-order mismatches.
- this is the only grading call and it is free + deterministic.

### Difficulty + diagnostics
BIRD ships `simple/moderate/challenging` labels; carry them in metadata so the
oracle-rescue diagnostic (`analyze_eval.py --oracle`) and IRT fit stratify by real
difficulty. This is the natural place to look for an actual capability ceiling
(a weak cheap model failing the `challenging` tier) -- the regime the cascade needs.

### Wiring
`run_selfimprove.py`: add `--benchmark sql` and `_build_sql(args, cfg, client)` that
builds `SqlBenchmark`, an `Executor` over `bench.tools()`, and `planner=None`
(sub-task DAGs over SQL are out of scope v1; DECOMPOSE still works as schema-explore
via the tools). Everything downstream (loop, credit, learners, ESCALATE, analysis)
is unchanged.

### Honest expectations
SQL selection/allocation wins are usually modest and often statistically tied at
small n -- treat this as a **credibility** benchmark (real task, clean cost, free
grader), not a place to expect a blowout. Report with the same paired-cost CI +
McNemar discipline as the rest.

### Build order / cost
Offline scaffold first (loader, tools, verifier, unit test on a tiny fixture db with
no model calls). The only live spend is when the agent generates SQL; a 40-task dev
run on a cheap model is inexpensive and fully cost-clean.

---

## 2. Response cache (free, deterministic re-runs)

### Why
We re-run sweeps constantly (calibration, seeds, ablations). Every re-run re-pays for
identical model calls, and the OpenRouter key has hit its spend limit. A cache keyed
on the exact request makes repeats free **and** keeps reported cost correct.

### Design
`src/wdp/llm/cache.py`:

```python
class ResponseCache:
    def __init__(self, path, mode="off"):      # mode: off | readwrite | replay
    def key(self, model, messages, temperature, max_tokens, tools) -> str  # sha256 of canonical JSON
    def get(self, key) -> LLMResponse | None
    def put(self, key, resp) -> None
```

Backed by a single SQLite file under `paths.cache` (one row per key: key, model,
response_text, prompt_tokens, completion_tokens, dollars, wall_seconds). SQLite gives
concurrent-reader safety for our parallel runs.

Hook into `OpenRouterClient.chat`:
1. compute `key` from `(model, messages, temperature, max_tokens, tools)`;
2. on a hit, **bill the STORED cost into the ledger** and return the cached response
   without calling the API. Billing the stored cost (not zero) is the key decision:
   it keeps every cost/solve number identical to a fresh run while making the re-run
   free to us. The measurement stays honest; only our wallet benefits.
3. on a miss, call the API, `put` the response with its real usage, return it.

### Determinism caveat (important)
A cache hit replays one fixed sample. That is correct for **eval** (greedy, intended
to be deterministic) and for **replaying a finished run** for analysis, but it is
WRONG for the exploration/collection passes, which use `temperature>0` specifically
to get variance across attempts. Rule:
- `mode=off` (default): no caching.
- `mode=readwrite`: cache reads+writes; **only serve hits when `temperature==0`**, so
  exploration variance is never frozen; temp>0 calls always go live (but are still
  written, so a later `replay` can reproduce them).
- `mode=replay`: serve every hit verbatim regardless of temperature (exact
  reproduction of a prior run for figures/analysis; errors if a needed key is
  missing).

Flag: `--cache-mode {off,readwrite,replay}` on `run_selfimprove.py`.

### Build order / cost
Pure offline infra; a unit test stubs the client and asserts a second identical call
makes no live request and reproduces the cost. Zero spend; immediate savings on the
next sweep.

---

## 3. Policy persistence (train once, deploy frozen)

### Why
Today a learned policy lives only in memory and is refit from traces every session.
For deployment (and for the productization question) we need: train once, save the
fitted policy, load it, and make decisions with no retraining and no trace store.

### Design
Add symmetric `to_dict()/from_dict()` (JSON-friendly) to the two stateful pieces:
- `LinearSoftmaxPolicy`: `W` (n_actions x n_features) and `b` -> lists.
- `BanditAllocator`: the per-arm Beta posteriors `_alpha/_beta`.

`src/wdp/allocator/persist.py`:

```python
def save_policy(alloc, path, *, meta=None): ...
def load_policy(path) -> Allocator: ...
```

The saved JSON records, besides the weights/posteriors:
- `allocator`: "bandit" | "bc" | "dpo" | "kto",
- `action_vocab`: the exact `ACTIONS` list at train time,
- `feature_names`: `NodeFeatures.names()` at train time,
- `meta`: benchmark, n_traces, seed, cost-weight, git sha, timestamp.

### Safety (load-time validation -- non-negotiable)
On `load_policy`, assert the saved `action_vocab` and `feature_names` **exactly match**
the current code, else refuse to load. We have already been bitten by silent
feature/action index drift; a persisted policy mis-indexed against a changed feature
vector would be a quiet, damaging bug. Fail loud instead.

### Wiring + a serve entrypoint
- `run_selfimprove.py --save-policy out.json`: after the final round, persist the
  fitted learner (or the round-0 bandit).
- `scripts/serve_policy.py --policy out.json`: load a frozen policy and expose
  `decide(features) -> action` (a tiny demo / the seam a real integration would call).
  No training, no trace log, no benchmark dependency -- this is the deployable unit.

### Build order / cost
Pure offline; unit test = fit a tiny policy, save, load, assert identical `probs()` on
a fixed feature vector and that a mismatched feature schema refuses to load. Zero spend.

---

## How they compose (the productization seam)
Together these turn the research loop into a deployable shape: **train** a policy on a
benchmark with a free reward (SQL) or your own reward, **replay** cached calls so
re-analysis costs nothing, **persist** the fitted policy, and **serve** it frozen
behind `decide(features)`. The honest framing stays the same: the controller learns
*your* cost-aware allocation from *your* traces and reward; it is not a pre-trained
drop-in that transfers a fixed win across task distributions (the tau result is the
standing evidence for why that caveat matters).

## Suggested build order
1. Response cache (offline, immediate spend savings, unblocks cheaper iteration).
2. Policy persistence (offline, small, unblocks the serve story).
3. SQL benchmark (offline scaffold + a small, cost-clean live run) -- the headline of
   the three, since it is the real-task credibility result.
