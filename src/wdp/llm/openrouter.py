"""Minimal OpenRouter chat client with built-in cost accounting.

OpenRouter exposes an OpenAI-compatible /chat/completions endpoint. We request
usage accounting so each call returns real token counts and dollar cost, which we
fold straight into a CostLedger. Wall-clock is measured locally.

Docs: https://openrouter.ai/docs
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from wdp.config import require_openrouter_key, load_env
from wdp.cost import CostLedger, Spend

import os

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class LLMResponse:
    text: str
    model: str
    spend: Spend
    raw: dict


class OpenRouterClient:
    """Thin synchronous client. One instance can be shared across executors;
    pass a per-task CostLedger to ``chat`` so spend is attributed correctly."""

    def __init__(self, timeout: float = 120.0) -> None:
        load_env()
        self._key = require_openrouter_key()
        self._app_url = os.environ.get("OPENROUTER_APP_URL", "")
        self._app_title = os.environ.get("OPENROUTER_APP_TITLE", "wdp-controller")
        self._client = httpx.Client(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        if self._app_url:
            h["HTTP-Referer"] = self._app_url
        if self._app_title:
            h["X-Title"] = self._app_title
        return h

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        ledger: CostLedger | None = None,
        parallel_group: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            # Ask OpenRouter to include real cost in the usage block.
            "usage": {"include": True},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        t0 = time.perf_counter()
        resp = self._client.post(BASE_URL, headers=self._headers(), json=payload)
        wall = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {}) or {}
        spend = Spend(
            model=model,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            wall_seconds=wall,
            dollars=float(usage.get("cost", 0.0)),
            parallel_group=parallel_group,
        )
        if ledger is not None:
            ledger.add(spend)
        return LLMResponse(text=text, model=model, spend=spend, raw=data)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenRouterClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
