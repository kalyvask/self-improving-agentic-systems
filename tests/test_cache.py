"""Offline tests for the response cache (no network, no key)."""
from __future__ import annotations

from wdp.llm.cache import ResponseCache

_MSGS = [{"role": "user", "content": "2+2?"}]


def test_key_is_stable_and_temperature_sensitive():
    a = ResponseCache.key("m", _MSGS, 0.0, None)
    b = ResponseCache.key("m", _MSGS, 0.0, None)
    c = ResponseCache.key("m", _MSGS, 0.7, None)
    assert a == b            # same request -> same key
    assert a != c            # temperature is part of the key


def test_put_get_roundtrip_preserves_cost(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite", mode="readwrite")
    k = ResponseCache.key("m", _MSGS, 0.0, None)
    assert cache.get(k) is None
    cache.put(k, model="m", text="4", prompt_tokens=3, completion_tokens=1,
              dollars=0.01, wall=0.2)
    hit = cache.get(k)
    assert hit["text"] == "4" and hit["dollars"] == 0.01 and hit["completion_tokens"] == 1
    cache.close()


def test_should_serve_respects_mode_and_temperature(tmp_path):
    rw = ResponseCache(tmp_path / "rw.sqlite", mode="readwrite")
    assert rw.should_serve(0.0) and not rw.should_serve(0.7)   # only deterministic hits
    rp = ResponseCache(tmp_path / "rp.sqlite", mode="replay")
    assert rp.should_serve(0.0) and rp.should_serve(0.7)       # exact reproduction
    off = ResponseCache(tmp_path / "off.sqlite", mode="off")
    assert not off.should_serve(0.0) and off.get("x") is None  # disabled
