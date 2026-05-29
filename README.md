# wdp-controller

A self-improving controller that decides how to spend the next unit of compute
on a tool-using agent task. At each decision node it picks one of four actions:

- `WIDER`: spawn a fresh parallel Executor attempt from the current state
- `DEEPER`: continue and refine the current trajectory on tool feedback
- `DECOMPOSE`: hand the task to the Planner, producing a sub-task DAG
- `STOP`: stop spending and abstain (a safe non-attempt)

The controller (the Allocator) is a small, CPU-trainable policy over cheap
numeric features, not a fine-tuned LLM. Executors are frontier models called
through OpenRouter. The expensive part is collecting traces; the policy update is
cheap. The Allocator learns from its own logged traces, so the headline result is
a self-improvement curve across rounds.

## Layout

```
src/wdp/
  config.py            .env + YAML config loading
  cost/                per-call cost accounting in three currencies
  llm/                 OpenRouter chat client with usage-based cost
  allocator/           the policy: Action, NodeFeatures, BanditAllocator (v0)
  verifier/            terminal (ground-truth) and process (cheap) scorers
  executor/            ReAct loop, tool protocol, Task/Trajectory types
  planner/             decomposability probe + sub-task DAG
  loop/                trace logging, credit assignment, round runner
  metrics/             success@budget, pass^k, risk-coverage, CVaR, gen-verif gap
tests/test_smoke.py    offline end-to-end test (no key, no network)
scripts/smoke_live.py  one live task against OpenRouter
config/default.yaml    models, budgets, allocator and loop settings
```

## Cost currencies

Every LLM call is logged in three currencies at once, because the optimal
allocation policy depends on which one you are spending:

- tokens: prompt plus completion tokens
- latency: wall-clock seconds, where concurrent branches cost the max of their
  children, not the sum
- dollars: OpenRouter usage cost when available

## Setup

```bash
pip install -e ".[dev]"
```

Paste your OpenRouter key into `.env` (get one at https://openrouter.ai/keys):

```
OPENROUTER_API_KEY=sk-or-...
```

## Run

Offline test (no key needed):

```bash
python -m pytest tests/test_smoke.py -q
```

Live single-task check (costs a few cents):

```bash
python scripts/smoke_live.py
```

## Policy roadmap

- v0 `BanditAllocator`: Thompson sampling over per-action value-per-cost. Works
  with no training data.
- BC: behavior-cloning the action choices from filtered good traces.
- DPO: preference pairs mined from sibling decisions with differing realized
  value-per-cost.
- GRPO is estimated, not run. The loop logs the per-call token and wall cost GRPO
  would need, so the end-of-project GRPO cost and expected ceiling are an
  extrapolation from measured data rather than a guess.
