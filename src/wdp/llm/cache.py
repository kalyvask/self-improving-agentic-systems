"""On-disk response cache for the OpenRouter client.

Repeated sweeps (calibration, seeds, ablations) re-pay for identical model calls.
This cache makes an exact repeat free to us while keeping the reported cost honest:
a cache HIT bills the STORED cost into the ledger (so every cost/solve number is
identical to a fresh run) but skips the API call (so we do not pay again).

Determinism caveat: a hit replays one fixed sample. That is correct for a
deterministic eval (temperature 0) or for replaying a finished run, but WRONG for
the exploration passes that use temperature > 0 to get variance across attempts.
Hence the three modes:
  - off       : no caching (default).
  - readwrite : write every call; serve a hit ONLY when temperature == 0, so
                exploration variance is never frozen (temp>0 calls still go live
                and are written, so a later `replay` can reproduce them).
  - replay    : serve every hit verbatim regardless of temperature (exact
                reproduction of a prior run for figures/analysis).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


class ResponseCache:
    def __init__(self, path: str | Path, mode: str = "off") -> None:
        if mode not in ("off", "readwrite", "replay"):
            raise ValueError(f"bad cache mode {mode!r}; want off|readwrite|replay")
        self.mode = mode
        self.path = Path(path)
        self._db: sqlite3.Connection | None = None
        if mode != "off":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self.path), check_same_thread=False)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS resp ("
                "key TEXT PRIMARY KEY, model TEXT, text TEXT, "
                "prompt_tokens INT, completion_tokens INT, dollars REAL, wall REAL)"
            )
            self._db.commit()

    @staticmethod
    def key(model: str, messages: list[dict], temperature: float,
            max_tokens: int | None, extra: dict | None = None) -> str:
        """Stable hash of the exact request. Any field that changes the response
        (model, messages, temperature, max_tokens, tools/extra) is part of the key."""
        blob = json.dumps(
            {"model": model, "messages": messages, "temperature": temperature,
             "max_tokens": max_tokens, "extra": extra or {}},
            sort_keys=True, separators=(",", ":"), default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def should_serve(self, temperature: float) -> bool:
        """Whether a hit may be served given the mode and call temperature."""
        if self.mode == "replay":
            return True
        if self.mode == "readwrite":
            return temperature == 0.0      # never freeze temp>0 exploration variance
        return False

    def get(self, key: str) -> dict | None:
        if self._db is None:
            return None
        row = self._db.execute(
            "SELECT model, text, prompt_tokens, completion_tokens, dollars, wall "
            "FROM resp WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return {"model": row[0], "text": row[1], "prompt_tokens": row[2],
                "completion_tokens": row[3], "dollars": row[4], "wall": row[5]}

    def put(self, key: str, *, model: str, text: str, prompt_tokens: int,
            completion_tokens: int, dollars: float, wall: float) -> None:
        if self._db is None:
            return
        self._db.execute(
            "INSERT OR REPLACE INTO resp VALUES (?,?,?,?,?,?,?)",
            (key, model, text, int(prompt_tokens), int(completion_tokens),
             float(dollars), float(wall)))
        self._db.commit()

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
