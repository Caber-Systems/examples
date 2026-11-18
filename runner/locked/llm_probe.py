# stdlib only; tested by tests/unit/test_llm_usage.py and tests/unit/test_locked_env_build.py (urlopen monkeypatched).
"""Prove a provider answers one chat and one embedding request before a run.

``probe_with_dim(provider, env)`` sends, with a 20 s timeout each:

- one chat request of ``"Say ok"`` with ``max_tokens`` 5 to the OpenAI-compatible
  ``/chat/completions`` under the provider's base URL, with a Bearer token. That
  is the path ``runner.core.llm_provider`` encodes for every provider, Anthropic
  included, and the one the apps actually use;
- one embedding of ``"ok"`` to ``/embeddings`` under the embedding provider's
  base URL. The embedding provider is ``SELECTED_EMBEDDING_PROVIDER`` when it
  names a provider, else the chat provider itself. Anthropic and Groq have no
  embeddings API, so for them it must be set. ``LOCAL`` (the resolver's
  fallback to deterministic local vectors) makes no request, so the embedding
  probe is skipped and reports nothing (``providers.resolve_providers`` is what
  refuses LOCAL for the preview).

It returns ``(problems, dim)``: one plain sentence per problem (empty when both
requests answered) and the length of the returned embedding vector, or None when
no embedding came back. ``run`` hands that length to ``resolve_providers`` so
``SELECTED_EMBEDDING_DIM`` is set without the stranger knowing it. ``probe`` is
the problems half alone.

The embedding key is ``SELECTED_EMBEDDING_API_KEY`` when set, else
``ORIGINAL_OPENAI_API_KEY`` when the embedding provider is OpenAI and it is set
(the resolver keeps the real OpenAI key there after pointing ``OPENAI_API_KEY``
at the chat provider), else ``{PROVIDER}_API_KEY``.

Base URLs and env names come from ``runner.locked.providers``, whose table a
unit test holds equal to ``runner.core.llm_provider.LLM_PROVIDERS``;
this module keeps no second table. Keys are read by variable name and never
appear in a problem sentence. No provider SDK is imported.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Dict, List, Mapping, Optional, Tuple

from runner.locked.providers import (BY_NAME, LOCAL_EMBEDDINGS, NO_EMBEDDINGS_API, Provider,  # noqa: F401
                                     embedding_spec, error_summary)

TIMEOUT_S = 20
CHAT_PROMPT = "Say ok"
CHAT_MAX_TOKENS = 5
EMBED_INPUT = "ok"

_BY_NAME = BY_NAME
_embedding_spec = embedding_spec


def _post(url: str, headers: Dict[str, str], body: dict) -> Tuple[bool, str, str]:
    """POST JSON; ``(True, "", body)`` on 2xx, else ``(False, reason, "")`` with no secrets."""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "CaberDemos/1.0", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = getattr(resp, "status", 200)
            if 200 <= status < 300:
                return True, "", raw
            return False, error_summary(status, raw), ""
    except urllib.error.HTTPError as exc:
        raw = ""
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        return False, error_summary(exc.code, raw), ""
    except Exception as exc:  # noqa: BLE001 -- URLError, timeouts, TLS: all one sentence
        detail = str(exc).splitlines()[0][:160] if str(exc) else ""
        return False, (f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__), ""


def _setting(env: Mapping[str, str], var: str) -> str:
    return (env.get(var) or "").strip() if var else ""


def _probe_chat(spec: Provider, env: Mapping[str, str]) -> List[str]:
    key = _setting(env, spec.api_key_var)
    model = _setting(env, spec.model_var)
    missing = [v for v, val in ((spec.api_key_var, key), (spec.model_var, model)) if not val]
    if missing:
        return [f"{spec.name} chat cannot be checked: {' and '.join(missing)} not set."]

    url = f"{spec.base_url.rstrip('/')}/chat/completions"
    auth = {"Authorization": f"Bearer {key}"}
    messages = [{"role": "user", "content": CHAT_PROMPT}]
    ok, reason, _raw = _post(url, auth, {"model": model, "messages": messages, "max_tokens": CHAT_MAX_TOKENS})
    if not ok and reason.startswith("HTTP 400") and "max_tokens" in reason:
        # Newer OpenAI models refuse max_tokens and want max_completion_tokens.
        ok, reason, _raw = _post(
            url, auth, {"model": model, "messages": messages, "max_completion_tokens": CHAT_MAX_TOKENS}
        )
    if ok:
        return []
    return [f"{spec.name} chat request (model {model} at {url}) failed: {reason}."]


def _embedding_key(spec: Provider, env: Mapping[str, str]) -> Tuple[str, str]:
    """``(key, variable it came from)``; the variable names the expected one when empty."""
    for var in ("SELECTED_EMBEDDING_API_KEY",
                "ORIGINAL_OPENAI_API_KEY" if spec.name == "OPENAI" else "",
                spec.api_key_var):
        value = _setting(env, var)
        if value:
            return value, var
    return "", spec.api_key_var


def _vector_length(raw: str) -> Optional[int]:
    """``len(data[0].embedding)`` of an OpenAI-shaped embeddings response, or None."""
    try:
        vector = json.loads(raw)["data"][0]["embedding"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    return len(vector) if isinstance(vector, list) and vector else None


def _probe_embedding(spec: Provider, env: Mapping[str, str]) -> Tuple[List[str], Optional[int]]:
    key, key_var = _embedding_key(spec, env)
    model = _setting(env, spec.embedding_model_var) or _setting(env, "SELECTED_EMBEDDING_MODEL")
    missing = []
    if not key:
        missing.append(f"SELECTED_EMBEDDING_API_KEY or {key_var}")
    if not model:
        missing.append(f"{spec.embedding_model_var} or SELECTED_EMBEDDING_MODEL")
    if missing:
        return [f"{spec.name} embeddings cannot be checked: {' and '.join(missing)} not set."], None

    url = f"{spec.base_url.rstrip('/')}/embeddings"
    body = {"model": model, "input": EMBED_INPUT}
    if spec.name == "NVIDIA":
        # NVIDIA NIM's asymmetric embedding models require input_type
        # (shared/milvus_query/query.py sends the same pair).
        body.update({"input_type": "query", "truncate": "END"})
    ok, reason, raw = _post(url, {"Authorization": f"Bearer {key}"}, body)
    if ok:
        return [], _vector_length(raw)
    return [f"{spec.name} embedding request (model {model} at {url}) failed: {reason}."], None


def probe_with_dim(provider: str, env: Mapping[str, str]) -> Tuple[List[str], Optional[int]]:
    """``(problems, embedding vector length)``; no problems = both endpoints answered."""
    spec = _BY_NAME.get((provider or "").strip().upper())
    if spec is None:
        return [f"{provider!r} is not a supported LLM provider; use one of {', '.join(_BY_NAME)}."], None

    problems = _probe_chat(spec, env)
    dim = None
    embed_spec, problem = _embedding_spec(spec, env)
    if embed_spec is not None:
        embed_problems, dim = _probe_embedding(embed_spec, env)
        problems.extend(embed_problems)
    elif problem:
        problems.append(problem)
    return problems, dim


def probe(provider: str, env: dict) -> List[str]:
    """Problems with ``provider``'s chat and embedding endpoints; empty = both answered."""
    return probe_with_dim(provider, env)[0]
