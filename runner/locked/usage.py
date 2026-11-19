# stdlib only; tested by tests/unit/test_locked_build_run_down.py
# Vendored verbatim from runner/mode_ci/llm_usage_summary.py (summarize and its
# helpers, docstring included); test_vendored_usage_summary_matches_mode_ci
# keeps the two in step. Locked `down` reads the usage volume with it.
"""Fold the apps' LLM usage JSONL into the lock's per-provider ``llm_usage`` map.

Each app (agentic-app, mcp-server, seed) appends one line per request via
``llm_usage.record`` (source: ``shared/llm_usage.py``). ``summarize`` reads every
``*.jsonl`` in the usage directory and returns, keyed by provider, the spec
section 6 shape:

    {"chat_model", "embedding_model", "chat_input_tokens", "chat_output_tokens",
     "embedding_tokens", "requests"}

plus ``embedding_provider`` for a provider whenever any of its records carries
one, i.e. its embeddings were served by another provider (always so for
``anthropic`` and ``groq``, which have no embeddings API). A model field is ``None`` when
the provider made no request of that kind; two distinct models of one kind are
joined with ", " in sorted order.

Nothing is guessed: a malformed line, an unknown kind, or a request whose
provider response carried no usage object raises ``ValueError`` naming the file
and line, because a lock that under-reports tokens is worse than no lock.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set

_REQUIRED = ("provider", "kind", "model", "input_tokens", "output_tokens", "requests")


def _iter_records(usage_dir: Path):
    for path in sorted(usage_dir.glob("*.jsonl")):
        for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            where = f"{path.name}:{lineno}"
            try:
                rec = json.loads(raw)
            except ValueError as exc:
                raise ValueError(f"{where}: not a JSON line ({exc})") from exc
            if not isinstance(rec, dict):
                raise ValueError(f"{where}: a usage line must be a JSON object")
            missing = [k for k in _REQUIRED if k not in rec]
            if missing:
                raise ValueError(f"{where}: usage line lacks {', '.join(missing)}")
            if rec["kind"] not in ("chat", "embedding"):
                raise ValueError(f"{where}: unknown usage kind {rec['kind']!r}")
            if rec.get("usage_missing"):
                raise ValueError(
                    f"{where}: a {rec['kind']} request to {rec['provider']} ({rec['model']}) "
                    "returned no token usage, so the totals cannot be trusted"
                )
            yield rec


def _joined(names: Set[str]):
    return ", ".join(sorted(names)) if names else None


def summarize(usage_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Per-provider usage totals from every ``*.jsonl`` under ``usage_dir``.

    Raises ``FileNotFoundError`` when ``usage_dir`` does not exist (no app ever
    recorded anything, which a CI run that made requests cannot produce).
    """
    usage_dir = Path(usage_dir)
    if not usage_dir.is_dir():
        raise FileNotFoundError(f"LLM usage directory {usage_dir} does not exist")

    totals: Dict[str, Dict[str, Any]] = {}
    for rec in _iter_records(usage_dir):
        provider = str(rec["provider"]).strip().lower()
        t = totals.setdefault(provider, {
            "chat_models": set(), "embedding_models": set(), "embedding_providers": set(),
            "chat_input_tokens": 0, "chat_output_tokens": 0, "embedding_tokens": 0, "requests": 0,
        })
        t["requests"] += int(rec["requests"])
        if rec["kind"] == "chat":
            t["chat_models"].add(str(rec["model"]))
            t["chat_input_tokens"] += int(rec["input_tokens"])
            t["chat_output_tokens"] += int(rec["output_tokens"])
        else:
            t["embedding_models"].add(str(rec["model"]))
            t["embedding_tokens"] += int(rec["input_tokens"])
            if rec.get("embedding_provider"):
                t["embedding_providers"].add(str(rec["embedding_provider"]).strip().lower())

    out: Dict[str, Dict[str, Any]] = {}
    for provider, t in totals.items():
        entry: Dict[str, Any] = {
            "chat_model": _joined(t["chat_models"]),
            "embedding_model": _joined(t["embedding_models"]),
        }
        if t["embedding_providers"]:
            entry["embedding_provider"] = _joined(t["embedding_providers"])
        entry["chat_input_tokens"] = t["chat_input_tokens"]
        entry["chat_output_tokens"] = t["chat_output_tokens"]
        entry["embedding_tokens"] = t["embedding_tokens"]
        entry["requests"] = t["requests"]
        out[provider] = entry
    return out


__all__: List[str] = ["summarize"]
