"""Configuration loading: .env secrets + YAML run config."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


def load_env() -> None:
    """Load .env from the repo root into the process environment."""
    load_dotenv(REPO_ROOT / ".env")


def require_openrouter_key() -> str:
    load_env()
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is empty. Open wdp-controller/.env and paste your "
            "key (get one at https://openrouter.ai/keys)."
        )
    return key


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
