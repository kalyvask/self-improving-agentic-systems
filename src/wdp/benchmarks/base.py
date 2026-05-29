"""Benchmark adapter protocol.

A benchmark supplies three things the loop needs and nothing else: the tasks,
the tools the Executor may call, and a terminal verifier that scores a final
answer against ground truth. Real benchmarks (tau-bench, SWE-bench, ALFWorld)
implement this same surface; the local arithmetic suite implements it too, so the
self-improvement driver is benchmark-agnostic.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from wdp.executor.react import Task, Tool
from wdp.verifier.scorer import TerminalVerifier


@runtime_checkable
class Benchmark(Protocol):
    name: str

    def tasks(self) -> list[Task]: ...
    def tools(self) -> dict[str, Tool]: ...
    def terminal_verifier(self) -> TerminalVerifier: ...
