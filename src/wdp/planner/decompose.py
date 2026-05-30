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

    def probe(self, task: Task, *, parallel_group=None, ledger=None) -> float:
        # A benchmark that knows its task structure can provide a calibrated
        # decomposability directly (NOT the gold answer -- just structure). We use
        # it when present because the cheap-LLM probe is badly miscalibrated: it
        # saturates near 1.0 even for atomic single-expression tasks, so it could
        # not separate "atomic, do not decompose" (rated 1.0) from "multi,
        # decompose" (rated 0.83). With the feature unable to tell them apart, no
        # learner can condition DECOMPOSE on task type. The structural value is the
        # signal the LLM probe is supposed to estimate; using it fixes the feature.
        meta = getattr(task, "metadata", None) or {}
        if meta.get("decomposability") is not None:
            return min(1.0, max(0.0, float(meta["decomposability"])))
        resp = self._client.chat(
            self._model,
            [
                {"role": "system", "content": _PROBE},
                {"role": "user", "content": str(task)},
            ],
            ledger=ledger if ledger is not None else self._ledger,
            parallel_group=parallel_group,
            temperature=0.0,
            max_tokens=8,
        )
        return min(1.0, max(0.0, _parse_float(resp.text)))

    def decompose(self, task: Task, *, parallel_group=None, ledger=None) -> SubTaskDAG:
        resp = self._client.chat(
            self._model,
            [
                {"role": "system", "content": _PLAN},
                {"role": "user", "content": str(task)},
            ],
            ledger=ledger if ledger is not None else self._ledger,
            parallel_group=parallel_group,
            temperature=0.3,
        )
        items = _parse_plan(resp.text)
        subs: list[SubTask] = []
        for it in items:
            sid = str(it.get("id") or f"{task.id}.s{len(subs)}")
            # Do NOT copy the parent's gold into subtasks: a subtask ("compute 2*3")
            # graded against the parent's gold (the final sum) would be scored wrong.
            # Keep the parent's gold under parent_gold for reference only.
            sub_meta = {k: v for k, v in task.metadata.items() if k != "gold"}
            sub_meta.update({"parent": task.id, "sub_id": sid,
                             "parent_gold": task.metadata.get("gold")})
            subs.append(
                SubTask(
                    task=Task(id=f"{task.id}::{sid}", prompt=str(it.get("prompt", "")),
                              metadata=sub_meta),
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
