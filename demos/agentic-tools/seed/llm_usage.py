"""Per-app LLM usage accounting for the CaberDemos public preview. stdlib only.

Every chat completion and embedding request an app makes appends one JSON line
to ``<usage dir>/<app>.jsonl``:

    {"ts", "app", "provider", "kind", "model", "input_tokens", "output_tokens",
     "requests": 1}

plus ``embedding_provider`` on an embedding served by a provider other than the
run's chat provider (always so for Anthropic and Groq, which have no embeddings
API), and ``usage_missing: true`` when the provider's response carried no usage
object. The usage dir is ``CABER_DEMO_USAGE_DIR``, default
``/var/lib/caber-demo/usage`` (the ``caber-demo-usage`` named volume in
demos/agentic-tools/docker-compose.yaml).

Recording from the apps is opt-in: ``record_or_warn`` does nothing unless
``CABER_DEMO_USAGE_DIR`` is set in the process environment, which only the
agentic-tools compose file does. Other demos built from the same images record
nothing and log nothing.
``runner/mode_ci/llm_usage_summary.py`` folds the lines into the lock's
per-provider ``llm_usage`` map.

The source of truth is ``shared/llm_usage.py``. Byte-identical copies live in
``CaberApp/agentic-app/``, ``CaberApp/mcp-server/`` and
``demos/agentic-tools/seed/`` because each image's build context only reaches its
own directory; ``tests/unit/test_llm_usage.py`` asserts the four are identical.
Edit this file, then copy it over the other three.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

DEFAULT_USAGE_DIR = "/var/lib/caber-demo/usage"
USAGE_DIR_ENV = "CABER_DEMO_USAGE_DIR"
KINDS = frozenset({"chat", "embedding"})

_WRITE_LOCK = threading.Lock()


def usage_dir_from_env() -> Path:
    """The directory records go to: ``CABER_DEMO_USAGE_DIR`` or the default."""
    return Path((os.environ.get(USAGE_DIR_ENV) or "").strip() or DEFAULT_USAGE_DIR)


def selected_chat_provider() -> str:
    """The run's chat provider (``SELECTED_LLM_PROVIDER``), lowercased.

    Records are keyed by the chat provider the run selected, which is also the
    key of the lock's ``llm_usage`` map. Empty when the variable is unset.
    """
    return (os.environ.get("SELECTED_LLM_PROVIDER") or "").strip().lower()


def recording_enabled() -> bool:
    """Whether this process opted in to usage recording.

    Recording is opt-in per compose file: only a service whose environment sets
    ``CABER_DEMO_USAGE_DIR`` records. The agentic-app and mcp-server images are
    shared by other demos, which leave it unset and so record nothing and log
    nothing.
    """
    return bool((os.environ.get(USAGE_DIR_ENV) or "").strip())


def embedding_provider_for(chat_provider: str) -> Optional[str]:
    """``SELECTED_EMBEDDING_PROVIDER`` (lowercased) when it differs from the chat
    provider (Anthropic and Groq have no embeddings API; any provider may also
    embed elsewhere); ``None`` when it is unset or the same provider."""
    embedding = (os.environ.get("SELECTED_EMBEDDING_PROVIDER") or "").strip().lower()
    if not embedding or embedding == chat_provider.strip().lower():
        return None
    return embedding


def record(
    app: str,
    provider: str,
    kind: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    usage_dir: Optional[Path] = None,
    *,
    embedding_provider: Optional[str] = None,
    usage_missing: bool = False,
) -> None:
    """Append one usage line for one request to ``<usage_dir>/<app>.jsonl``.

    Raises ``ValueError`` for an unknown ``kind`` or an ``app`` that is not a
    plain file stem, and ``OSError`` when the directory cannot be written.
    Callers that must not fail a user request on accounting use
    ``record_or_warn``.
    """
    if kind not in KINDS:
        raise ValueError(f"usage kind must be one of {sorted(KINDS)}, not {kind!r}")
    if not app or "/" in app or "\\" in app or app.startswith("."):
        raise ValueError(f"usage app name must be a plain file stem, not {app!r}")
    if not (provider or "").strip():
        raise ValueError(
            f"usage provider is empty for a {kind} request ({model}); "
            "SELECTED_LLM_PROVIDER is not set in this process"
        )

    line = {
        "ts": time.time(),
        "app": app,
        "provider": (provider or "").strip().lower(),
        "kind": kind,
        "model": model,
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "requests": 1,
    }
    if embedding_provider:
        line["embedding_provider"] = embedding_provider.strip().lower()
    if usage_missing:
        line["usage_missing"] = True

    target_dir = Path(usage_dir) if usage_dir is not None else usage_dir_from_env()
    text = json.dumps(line, separators=(",", ":")) + "\n"
    with _WRITE_LOCK:
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(target_dir / f"{app}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(text)


def record_or_warn(logger: logging.Logger, *args, **kwargs) -> bool:
    """``record``, but a failure is logged as a WARNING instead of raised.

    For call sites inside a user request (a chat turn, a search): losing one
    usage line must not fail the request, and the warning says the count is
    now short. Returns whether the line was written.

    Does nothing (returns ``False``, logs nothing) unless ``recording_enabled()``:
    a process that did not set ``CABER_DEMO_USAGE_DIR`` did not ask for usage.
    """
    if not recording_enabled():
        return False
    try:
        record(*args, **kwargs)
        return True
    except Exception as exc:  # noqa: BLE001 -- accounting never fails the request
        logger.warning("LLM usage was NOT recorded, so the usage totals are short: %s", exc)
        return False
