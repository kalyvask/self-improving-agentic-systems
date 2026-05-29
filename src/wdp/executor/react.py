"""The Executor: a single frontier-model attempt at a task via a ReAct loop.

The Executor is deliberately dumb. It takes a task and a starting state and runs
think -> act -> observe until it emits a final answer, stalls, or hits a step
cap. All the *intelligence about how much to spend* lives one level up in the
Allocator; the Executor just burns one trajectory's worth of compute and reports
back a Trajectory (with its cost folded into the shared ledger).

This is the unit the three spend-actions manipulate:
  - WIDER     = run another Executor from the same start state (fresh attempt)
  - DEEPER    = call `continue_from` to extend an existing Trajectory
  - DECOMPOSE = run Executors on Planner-produced sub-tasks (see wdp.planner)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable


@dataclass
class Task:
    """A unit of work. `id` is stable for logging; `prompt` is the instruction;
    `metadata` carries benchmark-specific payload (env handle, gold tests, etc)."""
    id: str
    prompt: str
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.prompt


@runtime_checkable
class Tool(Protocol):
    """A callable tool the Executor can invoke. Name + JSON-able signature."""
    name: str
    description: str

    def __call__(self, **kwargs) -> str: ...


@dataclass
class Step:
    """One think/act/observe cycle."""
    thought: str
    action: str | None = None        # tool name, or None for a pure-reasoning step
    action_input: dict = field(default_factory=dict)
    observation: str = ""


@dataclass
class Trajectory:
    """An Executor attempt. `final_answer` is set once the loop terminates with a
    answer; `stalled` marks a self-reported give-up (feeds executor_stalled)."""
    task_id: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str | None = None
    stalled: bool = False
    parallel_group: str | None = None

    @property
    def done(self) -> bool:
        return self.final_answer is not None or self.stalled

    @property
    def depth(self) -> int:
        return len(self.steps)

    def transcript(self) -> str:
        """Flat text rendering used as scorer input and for DEEPER continuation."""
        out: list[str] = []
        for i, s in enumerate(self.steps):
            out.append(f"[{i}] THOUGHT: {s.thought}")
            if s.action:
                out.append(f"    ACTION: {s.action}({json.dumps(s.action_input)})")
                out.append(f"    OBSERVATION: {s.observation}")
        if self.final_answer is not None:
            out.append(f"FINAL: {self.final_answer}")
        return "\n".join(out)


# Signature of the structured turn we ask the model to emit each step.
_SYSTEM = (
    "You are a tool-using agent solving a task step by step. Each turn, respond "
    "with a single JSON object and nothing else, of the form:\n"
    '{"thought": "...", "action": "<tool name or FINISH>", "action_input": {..}}\n'
    "Use action FINISH with action_input {\"answer\": \"...\"} when done. Use action "
    "STALL with {} if you are stuck and cannot make progress. Only call tools from "
    "the provided list."
)


class Executor:
    """Runs ReAct trajectories against an OpenRouter model."""

    def __init__(
        self,
        client,
        model: str,
        tools: dict[str, Tool] | None = None,
        *,
        max_steps: int = 25,
        temperature: float = 0.7,
    ) -> None:
        self._client = client
        self._model = model
        self._tools = tools or {}
        self._max_steps = max_steps
        self._temperature = temperature

    def _tool_catalog(self) -> str:
        if not self._tools:
            return "(no tools available; reason to the answer directly)"
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())

    def _ask(self, messages, ledger, parallel_group) -> dict:
        resp = self._client.chat(
            self._model,
            messages,
            ledger=ledger,
            parallel_group=parallel_group,
            temperature=self._temperature,
        )
        return _parse_turn(resp.text)

    def run(
        self,
        task: Task,
        *,
        ledger=None,
        parallel_group: str | None = None,
    ) -> Trajectory:
        """Run a fresh attempt from scratch."""
        traj = Trajectory(task_id=task.id, parallel_group=parallel_group)
        return self._loop(task, traj, ledger, parallel_group)

    def continue_from(
        self,
        task: Task,
        traj: Trajectory,
        *,
        ledger=None,
        parallel_group: str | None = None,
        extra_steps: int | None = None,
    ) -> Trajectory:
        """DEEPER: resume an existing (unfinished) trajectory and push it further."""
        traj.stalled = False  # give it another shot
        return self._loop(task, traj, ledger, parallel_group, extra_steps=extra_steps)

    def _loop(self, task, traj, ledger, parallel_group, extra_steps=None) -> Trajectory:
        budget = self._max_steps if extra_steps is None else traj.depth + extra_steps
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"TASK:\n{task.prompt}\n\nTOOLS:\n{self._tool_catalog()}"},
        ]
        if traj.steps:
            messages.append({"role": "assistant", "content": traj.transcript()})

        while traj.depth < budget and not traj.done:
            turn = self._ask(messages, ledger, parallel_group)
            action = (turn.get("action") or "").strip()
            step = Step(
                thought=str(turn.get("thought", "")),
                action=action or None,
                action_input=turn.get("action_input") or {},
            )

            if action.upper() == "FINISH":
                traj.final_answer = str(step.action_input.get("answer", ""))
                traj.steps.append(step)
                break
            if action.upper() == "STALL":
                traj.stalled = True
                traj.steps.append(step)
                break

            tool = self._tools.get(action)
            if tool is None:
                step.observation = f"ERROR: unknown tool {action!r}. Available: {list(self._tools)}"
            else:
                try:
                    step.observation = str(tool(**step.action_input))
                except Exception as e:  # tool errors are observations, not crashes
                    step.observation = f"ERROR: {type(e).__name__}: {e}"

            traj.steps.append(step)
            messages.append({"role": "assistant", "content": json.dumps(turn)})
            messages.append({"role": "user", "content": f"OBSERVATION: {step.observation}"})

        return traj


def _parse_turn(text: str) -> dict:
    """Tolerant JSON-turn parser. Falls back to treating raw text as a FINISH."""
    text = (text or "").strip()
    # Strip markdown fences if the model wrapped its JSON.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    # Last resort: the model answered in prose -> treat it as the final answer.
    return {"thought": "", "action": "FINISH", "action_input": {"answer": text}}


def make_tool(name: str, description: str, fn: Callable[..., str]) -> Tool:
    """Wrap a plain function as a Tool (name/description + __call__)."""

    class _Fn:
        def __init__(self) -> None:
            self.name = name
            self.description = description

        def __call__(self, **kwargs) -> str:
            return str(fn(**kwargs))

    return _Fn()
