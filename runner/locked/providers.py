# stdlib only; tested by tests/unit/test_locked_check.py and tests/unit/test_locked_env_build.py
"""The LLM provider table and resolver the locked ``check``, ``run`` and modes use.

A vendored copy of the fields of ``runner.core.llm_provider.LLM_PROVIDERS`` that
the locked runner needs, because ``runner.locked`` may import only the standard
library. ``tests/unit/test_locked_check.py`` holds it equal to the original.

``resolve_providers`` is the stdlib mirror of ``resolve_llm_provider(probe=False)``
followed by ``resolve_embedding_provider(probe=False)`` for ONE explicitly selected
provider: it sets exactly the variables the CI resolver sets, so the compose files
see the same environment a CI run gives them. ``tests/unit/test_locked_env_build.py``
holds the two equal. Every refusal is one sentence that names variables, never values.
"""

import json
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

LOCAL_EMBEDDINGS = "LOCAL"
# Providers with no embeddings API (runner/core/llm_provider.py's module
# docstring: the resolver skips them for embeddings).
NO_EMBEDDINGS_API = frozenset({"ANTHROPIC", "GROQ"})


class ProviderError(Exception):
    """The selected provider cannot be resolved; the message is one sentence."""


class EmbeddingDimMissing(ProviderError):
    """Everything resolved except the embedding dimension, which a probe or a running container can supply."""


@dataclass(frozen=True)
class Provider:
    name: str
    api_key_var: str
    model_var: str
    embedding_model_var: str
    base_url: str
    host: str

    @property
    def embedding_dim_var(self) -> str:
        return f"{self.name}_EMBEDDING_DIM"


PROVIDERS: tuple[Provider, ...] = (
    Provider("ANTHROPIC", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "ANTHROPIC_EMBEDDING_MODEL",
             "https://api.anthropic.com/v1", "api.anthropic.com"),
    Provider("NVIDIA", "NVIDIA_API_KEY", "NVIDIA_MODEL", "NVIDIA_EMBEDDING_MODEL",
             "https://integrate.api.nvidia.com/v1", "integrate.api.nvidia.com"),
    Provider("GEMINI", "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_EMBEDDING_MODEL",
             "https://generativelanguage.googleapis.com/v1beta/openai", "generativelanguage.googleapis.com"),
    Provider("OPENAI", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_EMBEDDING_MODEL",
             "https://api.openai.com/v1", "api.openai.com"),
    Provider("GROQ", "GROQ_API_KEY", "GROQ_MODEL", "GROQ_EMBEDDING_MODEL",
             "https://api.groq.com/openai/v1", "api.groq.com"),
)

BY_NAME: dict[str, Provider] = {p.name: p for p in PROVIDERS}
NAMES: tuple[str, ...] = tuple(p.name for p in PROVIDERS)


def _setting(env: Mapping[str, str], var: str) -> str:
    return (env.get(var) or "").strip() if var else ""


def embedding_spec(spec: Provider, env: Mapping[str, str]) -> Tuple[Optional[Provider], str]:
    """``(embedding provider, "")``, ``(None, problem)``, or ``(None, "")`` for LOCAL."""
    selected = _setting(env, "SELECTED_EMBEDDING_PROVIDER").upper()
    usable = ", ".join(n for n in BY_NAME if n not in NO_EMBEDDINGS_API)
    if selected == LOCAL_EMBEDDINGS:
        return None, ""
    if selected:
        target = BY_NAME.get(selected)
        if target is None:
            return None, (
                f"SELECTED_EMBEDDING_PROVIDER={selected} is not a provider; use one of {usable} or LOCAL."
            )
        if target.name in NO_EMBEDDINGS_API:
            return None, (
                f"SELECTED_EMBEDDING_PROVIDER={selected} cannot embed: {selected} has no embeddings API; "
                f"use one of {usable} or LOCAL."
            )
        return target, ""
    if spec.name in NO_EMBEDDINGS_API:
        return None, (
            f"{spec.name} has no embeddings API, so SELECTED_EMBEDDING_PROVIDER must name the provider "
            f"that embeds for it (one of {usable}) or LOCAL."
        )
    return spec, ""


def _unset(var: str, who: str) -> str:
    return f"{var} is not set in .env.local or the environment; {who} needs it."


def _select_provider(env: Mapping[str, str]) -> Provider:
    """Resolve the active chat provider from ``SELECTED_LLM_PROVIDER`` or the first configured ``<PROVIDER>_API_KEY``."""
    choices = ", ".join(NAMES)
    name = _setting(env, "SELECTED_LLM_PROVIDER").upper()
    if name:
        spec = BY_NAME.get(name)
        if spec is None:
            raise ProviderError(f"SELECTED_LLM_PROVIDER={name} is not one of {choices}.")
        return spec
    for candidate in PROVIDERS:
        if _setting(env, candidate.api_key_var):
            return candidate
    raise ProviderError(
        f"SELECTED_LLM_PROVIDER is not set in .env.local or the environment; choose one of {choices}.")


def resolve_providers(env: Mapping[str, str], *, embedding_dim: Optional[int] = None) -> dict[str, str]:
    """A NEW dict: ``env`` plus the variables the CI resolver sets for the selected provider.

    Raises ``ProviderError`` (one sentence) when the selection cannot be resolved,
    and ``EmbeddingDimMissing`` when only the embedding dimension is unknown: no
    ``<PROVIDER>_EMBEDDING_DIM`` and no ``embedding_dim`` argument.
    """
    spec = _select_provider(env)

    key = _setting(env, spec.api_key_var)
    if not key:
        raise ProviderError(_unset(spec.api_key_var, f"the selected provider {spec.name}"))
    model = _setting(env, spec.model_var)
    model_vars = spec.model_var
    if not model and spec.name == "OPENAI":
        model = _setting(env, "USE_LLM_MODEL")
        model_vars = "OPENAI_MODEL (or USE_LLM_MODEL)"
    if not model:
        raise ProviderError(_unset(model_vars, f"the selected provider {spec.name}"))

    usable = ", ".join(n for n in NAMES if n not in NO_EMBEDDINGS_API)
    if _setting(env, "SELECTED_EMBEDDING_PROVIDER").upper() == LOCAL_EMBEDDINGS:
        raise ProviderError(
            "SELECTED_EMBEDDING_PROVIDER=LOCAL is not offered in the preview, whose apps need real "
            f"embeddings; use one of {usable}.")
    emb, problem = embedding_spec(spec, env)
    if emb is None:
        raise ProviderError(problem)

    out = dict(env)
    original_openai = _setting(env, "OPENAI_API_KEY")
    if original_openai and "ORIGINAL_OPENAI_API_KEY" not in out:
        out["ORIGINAL_OPENAI_API_KEY"] = original_openai
    out.update({
        "SELECTED_LLM_PROVIDER_RESOLVED": "1",
        "SELECTED_LLM_PROVIDER": spec.name,
        "SELECTED_LLM_MODEL": model,
        "SELECTED_LLM_BASE_URL": spec.base_url,
        "SELECTED_LLM_HOST": spec.host,
        "OPENAI_API_KEY": key,
        "OPENAI_BASE_URL": spec.base_url,
        "OPENAI_MODEL": model,
        "USE_LLM_MODEL": model,
        "DATAFACTORY_OPENAI_MODEL": model,
    })

    if emb.name == "OPENAI" and spec.name != "OPENAI":
        # The chat provider's key now sits in OPENAI_API_KEY; OpenAI's own is kept aside.
        emb_key = _setting(out, "ORIGINAL_OPENAI_API_KEY")
    else:
        emb_key = _setting(env, emb.api_key_var)
    if not emb_key:
        raise ProviderError(_unset(emb.api_key_var, f"the embedding provider {emb.name}"))
    emb_model = _setting(env, emb.embedding_model_var)
    if not emb_model:
        raise ProviderError(_unset(emb.embedding_model_var, f"the embedding provider {emb.name}"))
    dim_raw = _setting(env, emb.embedding_dim_var)
    if dim_raw:
        if not dim_raw.isdigit() or int(dim_raw) <= 0:
            raise ProviderError(f"{emb.embedding_dim_var}={dim_raw!r} is not a positive integer.")
        dim = int(dim_raw)
    elif embedding_dim:
        dim = int(embedding_dim)
    else:
        raise EmbeddingDimMissing(
            f"{emb.embedding_dim_var} is not set and no embedding probe has measured the vector length; "
            f"set {emb.embedding_dim_var} in .env.local.")
    out.update({
        "SELECTED_EMBEDDING_RESOLVED": "1",
        "SELECTED_EMBEDDING_PROVIDER": emb.name,
        "SELECTED_EMBEDDING_API_KEY": emb_key,
        "SELECTED_EMBEDDING_BASE_URL": emb.base_url,
        "SELECTED_EMBEDDING_HOST": emb.host,
        "SELECTED_EMBEDDING_MODEL": emb_model,
        "SELECTED_EMBEDDING_DIM": str(dim),
    })
    return out


def error_summary(status_code: int, body_text: str) -> str:
    """Return a concise, single-line error reason without leaking secrets."""
    text = (body_text or "").strip()
    if text:
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                err = payload.get("error")
                if isinstance(err, dict):
                    msg = err.get("message") or err.get("type") or err.get("code")
                    if msg:
                        return f"HTTP {status_code}: {str(msg).splitlines()[0][:160]}"
                elif isinstance(err, str):
                    return f"HTTP {status_code}: {err.splitlines()[0][:160]}"
                detail = payload.get("detail") or payload.get("message") or payload.get("title")
                if detail:
                    return f"HTTP {status_code}: {str(detail).splitlines()[0][:160]}"
        except Exception:
            pass
        first_line = text.splitlines()[0][:160]
        return f"HTTP {status_code}: {first_line}"
    return f"HTTP {status_code}"
