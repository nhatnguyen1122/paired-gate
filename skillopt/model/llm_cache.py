"""Disk cache, run_id isolation, concurrency cap and USD accounting for LLM calls.

Used by :mod:`skillopt.model.openai_compatible_backend` (the Mistral path).

Everything here is opt-in: the backend uses it only when ``SKILLOPT_LLM_DIR`` is
set (``scripts/mistral_env.sh`` sets it); otherwise it behaves as upstream.

* Every call is cached on disk (SQLite) under
  ``sha256(model, temperature, max_tokens, messages, run_id)``.  The ``run_id``
  is part of the key, so re-running the *same* skill under a different
  ``run_id`` is a genuinely fresh stochastic sample and is never served from the
  cache.  ``set_run_id`` / ``run_id_scope`` change the active run id.
* Every *real* (non-cached) call appends its ``usage`` and USD cost to a ledger
  (``$SKILLOPT_LLM_DIR/ledger.jsonl``).  The total is re-read from the ledger
  before every real call, so the cap is shared by all processes that use the
  same ledger; once it reaches ``$SKILLOPT_MAX_USD``
  (default 10) every further real call raises :class:`BudgetExceeded`.
  ``BudgetExceeded`` derives from ``BaseException`` so the many
  ``except Exception`` blocks in the rollout code cannot swallow it and record a
  fake "wrong answer".
* At most ``$SKILLOPT_MAX_CONCURRENCY`` (default 8) requests are in flight per
  process.

Prices (USD per 1M tokens) were read from Mistral's model cards on 2026-11-26:
https://docs.mistral.ai/models/ministral-3-8b-25-12 etc.  ``mistral-medium`` has
no published price on its card, so it is absent; unknown models are charged at
the highest known price (the cap can only trigger early, never late).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Any

# input, output  (USD / 1M tokens), from docs.mistral.ai model cards, 2026-11-26
PRICES: dict[str, tuple[float, float]] = {
    "ministral-3b-latest": (0.10, 0.10),
    "ministral-3b-2512": (0.10, 0.10),
    "ministral-8b-latest": (0.15, 0.15),
    "ministral-8b-2512": (0.15, 0.15),
    "ministral-14b-latest": (0.20, 0.20),
    "ministral-14b-2512": (0.20, 0.20),
    "mistral-small-latest": (0.15, 0.60),
    "mistral-small-2603": (0.15, 0.60),
    "mistral-large-latest": (0.70, 2.10),
    "mistral-large-4-0": (0.70, 2.10),
}


def _force_ipv4() -> None:
    """On hosts with a broken IPv6 route every new HTTPS connection first waits
    for the IPv6 connect to time out (observed: 10-150 s stalls). Resolve IPv4
    only. Disable with SKILLOPT_FORCE_IPV4=0."""
    if os.environ.get("SKILLOPT_FORCE_IPV4", "1") == "0":
        return
    import socket

    if getattr(socket, "_skillopt_ipv4", False):
        return
    orig = socket.getaddrinfo

    def getaddrinfo(host, port, family=0, *a, **k):
        if family in (0, socket.AF_UNSPEC):
            family = socket.AF_INET
        return orig(host, port, family, *a, **k)

    socket.getaddrinfo = getaddrinfo
    socket._skillopt_ipv4 = True


_force_ipv4()


class BudgetExceeded(BaseException):
    """Raised when the USD cap is reached. BaseException on purpose."""


_LOCK = threading.RLock()
_RUN_ID = 0
_DB: sqlite3.Connection | None = None
_TOTAL_USD = 0.0
_LEDGER_OFFSET = 0  # bytes of the ledger already summed into _TOTAL_USD
_ABORTED = False
_SEM = threading.BoundedSemaphore(int(os.environ.get("SKILLOPT_MAX_CONCURRENCY", "8")))
_STATS = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}


def llm_dir() -> str:
    d = os.environ.get("SKILLOPT_LLM_DIR") or os.path.join(os.getcwd(), ".llm_cache")
    os.makedirs(d, exist_ok=True)
    return d


def max_usd() -> float:
    return float(os.environ.get("SKILLOPT_MAX_USD", "10"))


def experiment_tag() -> str:
    return os.environ.get("SKILLOPT_EXPERIMENT", "untagged")


# ── run id ────────────────────────────────────────────────────────────────
def set_run_id(run_id: Any) -> None:
    global _RUN_ID
    with _LOCK:
        _RUN_ID = run_id


def get_run_id() -> Any:
    return _RUN_ID


@contextlib.contextmanager
def run_id_scope(run_id: Any):
    prev = get_run_id()
    set_run_id(run_id)
    try:
        yield
    finally:
        set_run_id(prev)


# ── cache ─────────────────────────────────────────────────────────────────
def _db() -> sqlite3.Connection:
    global _DB
    with _LOCK:
        if _DB is None:
            _DB = sqlite3.connect(os.path.join(llm_dir(), "cache.sqlite"), check_same_thread=False,
                                  timeout=60)
            _DB.execute("PRAGMA journal_mode=WAL")
            _DB.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
            _DB.commit()
        return _DB


def cache_key(model: str, temperature: Any, max_tokens: int, messages: list, run_id: Any,
              extra: dict | None = None) -> str:
    payload = json.dumps(
        {"model": model, "temperature": temperature, "max_tokens": max_tokens,
         "messages": messages, "run_id": str(run_id), "extra": extra or {}},
        sort_keys=True, ensure_ascii=False, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_get(key: str) -> dict | None:
    with _LOCK:
        row = _db().execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def cache_put(key: str, value: dict) -> None:
    with _LOCK:
        _db().execute("INSERT OR REPLACE INTO c (k, v) VALUES (?, ?)", (key, json.dumps(value)))
        _db().commit()


# ── accounting ────────────────────────────────────────────────────────────
def enabled() -> bool:
    """Cache + ledger are opt-in: active only when SKILLOPT_LLM_DIR is set
    (``scripts/mistral_env.sh`` sets it). Otherwise the backend behaves as upstream."""
    return bool(os.environ.get("SKILLOPT_LLM_DIR"))


def price_for(model: str) -> tuple[float, float]:
    """Known Mistral price, else the most expensive known price (conservative:
    an unknown model can only make the cap trigger earlier, never later)."""
    if model in PRICES:
        return PRICES[model]
    return max(p[0] for p in PRICES.values()), max(p[1] for p in PRICES.values())


def _ledger_path() -> str:
    return os.path.join(llm_dir(), "ledger.jsonl")


def total_usd() -> float:
    """Total spent according to the shared ledger (all processes). Reads only the
    complete lines appended since the last call."""
    global _TOTAL_USD, _LEDGER_OFFSET
    with _LOCK:
        path = _ledger_path()
        if os.path.exists(path):
            with open(path, "rb") as f:
                f.seek(_LEDGER_OFFSET)
                chunk = f.read()
            end = chunk.rfind(b"\n") + 1  # ignore a partially written last line
            for line in chunk[:end].splitlines():
                try:
                    _TOTAL_USD += float(json.loads(line)["usd"])
                except Exception:  # noqa: BLE001
                    pass
            _LEDGER_OFFSET += end
        return _TOTAL_USD


def check_budget() -> None:
    global _ABORTED
    if _ABORTED or total_usd() >= max_usd():
        _ABORTED = True
        raise BudgetExceeded(f"USD cap reached: spent ${total_usd():.4f} >= --max_usd ${max_usd():.2f}")


def record_usage(model: str, stage: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = price_for(model)
    usd = prompt_tokens * pin / 1e6 + completion_tokens * pout / 1e6
    rec = {"t": round(time.time(), 2), "exp": experiment_tag(), "model": model, "stage": stage,
           "run_id": str(get_run_id()), "prompt_tokens": prompt_tokens,
           "completion_tokens": completion_tokens, "usd": usd}
    with _LOCK:
        # one write per record (O_APPEND), picked up by every process's total_usd()
        with open(_ledger_path(), "a") as f:
            f.write(json.dumps(rec) + "\n")
        _STATS["calls"] += 1
        _STATS["prompt_tokens"] += prompt_tokens
        _STATS["completion_tokens"] += completion_tokens
        _STATS["usd"] += usd
    return usd


def note_cache_hit() -> None:
    with _LOCK:
        _STATS["cache_hits"] += 1


def session_stats() -> dict:
    with _LOCK:
        return dict(_STATS)


@contextlib.contextmanager
def concurrency_slot():
    _SEM.acquire()
    try:
        yield
    finally:
        _SEM.release()


def message_text(content: Any) -> str:
    """Mistral reasoning models return ``content`` as a list of typed chunks
    (``thinking`` + ``text``). Keep only the visible text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for ch in content:
            if isinstance(ch, dict):
                if ch.get("type") == "text":
                    parts.append(str(ch.get("text", "")))
            else:
                if getattr(ch, "type", None) == "text":
                    parts.append(str(getattr(ch, "text", "")))
        return "".join(parts)
    return str(content)
