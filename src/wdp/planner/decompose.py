"""The Planner: turns one task into a small sub-task DAG for DECOMPOSE.

Two responsibilities, matching the two ways the rest of the system uses it:

  1. `probe(task)` -> decomposability in [0,1]. A *cheap* signal the Allocator
     reads at every node (NodeFeatures.decomposability) to gate the DECOMPOSE arm
     so it does not waste a plan on an atomic task. This is the only Planner call
     on the hot path, so it is one short cheap-model completion.

  2. `decompose(task)` -> SubTaskDAG. Only paid for once the Allocator actually
     chooses DECOMPOSE. Produces sub-tasks plus dependency edges so the loop can
     run independent branches in parallel (SPRINT-style) and bill them as a
     parallel_group, while respecting ordering where a sub-task needs an earlier
     result (ADaPT-style as-needed structure).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from wdp.executor.react import Task
from wdp.verifier.scorer import _parse_float


@dataclass
class SubTask:
    task: Task
    depends_on: list[str] = field(default_factory=list)  # SubTask.task.id values


@dataclass
class SubTaskDAG:
    parent_id: str
    subtasks: list[SubTask] = field(default_factory=list)

    def ready_layers(self) -> list[list[SubTask]]:
        """Topological layering: each layer's sub-tasks have all deps satisfied by
        earlier layers, so a layer can run as one parallel_group."""
        done: set[str] = set()
        remaining = list(self.subtasks)
        layers: list[list[SubTask]] = []
        while remaining:
            layer = [st for st in remaining if all(d in done for d in st.depends_on)]
            if not layer:  # cycle / dangling dep -> dump the rest serially
                layer = remaining
            layers.append(layer)
            done.update(st.task.id for st in layer)
            remaining = [st for st in remaining if st not in layer]
        return layers


_PROBE = (
    "Rate from 0 to 1 how much this task benefits from being split into "
    "independent sub-tasks. 0 = atomic/single-step, 1 = clearly several separable "
    "parts. Output ONLY the number."
)

_PLAN = (
    "Decompose the task into 2-5 sub-tasks. Respond with ONLY a JSON list; each "
    'item is {"id": "s1", "prompt": "...", "depends_on": ["s0", ...]}. Use '
    "depends_on to express ordering when a sub-task needs an earlier result; "
    "leave it empty for sub-tasks that can run in parallel."
)


class Planner:
    def __init__(self, client, model: str, ledger=None) -> None:
        self._client = client
        self._model = model
        self._ledger = ledger

    def probe(self, task: Task, *, parallel_group=None) -> float:
        resp = self._client.chat(
            self._model,
            [
                {"role": "system", "content": _PROBE},
                {"role": "user", "content": str(task)},
            ],
            ledger=self._ledger,
            parallel_group=parallel_group,
            temperature=0.0,
            max_tokens=8,
        )
        return min(1.0, max(0.0, _parse_float(resp.text)))

    def decompose(self, task: Task, *, parallel_group=None) -> SubTaskDAG:
        resp = self._client.chat(
            self._model,
            [
                {"role": "system", "content": _PLAN},
                {"role": "user", "content": str(task)},
            ],
            ledger=self._ledger,
            parallel_group=parallel_group,
            temperature=0.3,
        )
        items = _parse_plan(resp.text)
        subs: list[SubTask] = []
        for it in items:
            sid = str(it.get("id") or f"{task.id}.s{len(subs)}")
            subs.append(
                SubTask(
                    task=Task(id=f"{task.id}::{sid}", prompt=str(it.get("prompt", "")),
                              metadata={**task.metadata, "parent": task.id, "sub_id": sid}),
                    depends_on=[f"{task.id}::{d}" for d in (it.get("depends_on") or [])],
                )
            )
        return SubTaskDAG(parent_id=task.id, subtasks=subs)


def _parse_plan(text: str) -> list[dict]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except (json.JSONDecodeError, ValueError):
        pass
    return []
