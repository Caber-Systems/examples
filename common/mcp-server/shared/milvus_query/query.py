"""Embed a query and search a Milvus collection. No Streamlit, no FastAPI.

Every failure here raises ``MilvusQueryError`` with a message that says what was
actually wrong. Nothing degrades to an empty result set or a keyword fallback: a
semantic search that silently returns nothing looks identical to "there are no
relevant documents", which would quietly hollow out the retrieval half of the
demo. A clear failure beats a fake success.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import logging
import math
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = os.getenv("SELECTED_EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_COLLECTION = os.getenv("MILVUS_COLLECTION", "agentic_tools_corpus")
DEFAULT_LIMIT = 8

_MODEL_DIMS: Dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def _resolve_embed_dim(model: str) -> int:
    env_dim = (os.getenv("SELECTED_EMBEDDING_DIM") or "").strip()
    selected_model = (os.getenv("SELECTED_EMBEDDING_MODEL") or "").strip()
    if env_dim.isdigit() and int(env_dim) > 0 and (not selected_model or selected_model == model):
        return int(env_dim)
    return _MODEL_DIMS.get(model, 1536)

_STOPWORDS = frozenset({
    "the", "and", "for", "that", "with", "this", "from", "are", "was", "were",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must",
    "have", "has", "had", "does", "did", "not", "into", "about", "under",
    "over", "between", "through", "during", "before", "after", "above", "below",
    "please", "tell", "give", "show", "find", "list", "describe", "explain",
})


def _deterministic_hash_embedding(text: str, dim: int = 1536) -> List[float]:
    """Compute a deterministic L2-normalized feature-hashed embedding vector.

    Used when the active LLM provider does not serve OpenAI's `/v1/embeddings`
    model (e.g. Anthropic, Groq, NVIDIA) or when OpenAI embeddings are
    unavailable. Both `ingest_milvus.py` and `mcp-server` call `embed_query`,
    so documents and queries live in the exact same inner-product vector space.
    """
    vec = [0.0] * dim
    tokens = [
        tok for tok in re.findall(r"[a-z0-9]{2,}", text.lower())
        if tok not in _STOPWORDS
    ]
    if not tokens:
        tokens = re.findall(r"[a-z0-9]+", text.lower()) or ["empty"]

    features: Counter[str] = Counter()
    for tok in tokens:
        features[f"u:{tok}"] += 1
        if len(tok) >= 4:
            for i in range(len(tok) - 3):
                features[f"c:{tok[i:i+4]}"] += 1
    for i in range(len(tokens) - 1):
        features[f"b:{tokens[i]}_{tokens[i+1]}"] += 1

    for feat, count in features.items():
        if feat.startswith("b:"):
            weight = 1.5 * (1.0 + math.log(count))
        elif feat.startswith("u:"):
            weight = 1.0 * (1.0 + math.log(count))
        else:
            weight = 0.25 * (1.0 + math.log(count))
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "little")
        idx = h % dim
        sign = 1.0 if ((h >> 32) & 1) == 0 else -1.0
        vec[idx] += sign * weight

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [v / norm for v in vec]
    else:
        vec[0] = 1.0
    return vec

# Inner product, matching how the collections in this repo are indexed
# (milvus_utils.create_collection uses metric_type="IP").
METRIC_TYPE = "IP"


class MilvusQueryError(RuntimeError):
    """Semantic search could not be performed. Never swallowed."""


@dataclass
class SemanticHit:
    """One semantic search hit.

    ``path`` is not decoration: ticket 06 makes source document paths a hard
    requirement of this tool's result, because without them the
    answer-to-document trace cannot be reconstructed and the provenance pillar has
    nothing to render.
    """

    path: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "score": self.score,
            "snippets": [self.text[:400]] if self.text else [],
            "preview": self.text[:300] + "..." if len(self.text) > 300 else self.text,
            "metadata": self.metadata,
        }


def _milvus_uri(host: Optional[str] = None, port: Optional[str] = None) -> str:
    host = host or os.getenv("MILVUS_HOST", "milvusdb")
    port = port or os.getenv("MILVUS_PORT", "19530")
    return f"http://{host}:{port}"


def _grpc_cause(exc: BaseException) -> str:
    """The gRPC status behind ``exc``, as " [grpc CODE: details]", or "".

    pymilvus turns any ``grpc.RpcError`` on its Connect handshake into one fixed
    sentence ("Fail connecting to server on ..., illegal connection params or
    server unavailable") and keeps the RpcError only as the exception's
    context. UNAVAILABLE, DEADLINE_EXCEEDED and a refused handshake need
    different fixes, so the status is surfaced instead of being lost.
    """
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        code_fn = getattr(cur, "code", None)
        details_fn = getattr(cur, "details", None)
        if callable(code_fn) and callable(details_fn):
            try:
                code = code_fn()
                details = details_fn()
            except Exception:  # noqa: BLE001 -- a half-built RpcError
                code, details = None, None
            if code is not None:
                name = getattr(code, "name", None) or str(code)
                return f" [grpc {name}: {details}]"
        cur = cur.__cause__ or cur.__context__
    return ""


# One client per (uri, token) per process. Building a MilvusClient runs a
# Connect handshake RPC; doing that on every search_semantic call made each
# search pay for, and be able to fail on, a fresh handshake. The gRPC channel
# under a cached client reconnects by itself; a client whose call failed is
# evicted (see ``search``) so the next call builds a fresh one.
#
# Idle HTTP/2 keepalive pings are disabled: pymilvus defaults
# ``grpc.keepalive_permit_without_calls`` to True with a 10s ping interval,
# which trips Milvus Lite's default C++ gRPC server ping strike limit while a
# cached channel sits idle between queries (logging GOAWAY / ENHANCE_YOUR_CALM
# "too_many_pings" and dropping the connection).
_GRPC_OPTIONS: Dict[str, Any] = {
    "grpc.keepalive_permit_without_calls": False,
}
_CLIENTS: Dict[tuple, Any] = {}
_CLIENTS_LOCK = threading.Lock()


def _evict_client(client: Any) -> None:
    with _CLIENTS_LOCK:
        for key, cached in list(_CLIENTS.items()):
            if cached is client:
                del _CLIENTS[key]


def get_client(uri: Optional[str] = None, token: Optional[str] = None):
    """Return this process's MilvusClient for ``uri``, or raise naming the cause."""
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise MilvusQueryError(
            "pymilvus is not installed, so semantic search cannot run. Install it "
            "rather than letting search_semantic degrade to keyword search."
        ) from exc

    target = uri or _milvus_uri()
    key = (target, token)
    with _CLIENTS_LOCK:
        cached = _CLIENTS.get(key)
        if cached is not None:
            return cached
        try:
            client = MilvusClient(
                uri=target,
                token=token,
                grpc_options=dict(_GRPC_OPTIONS),
            )
        except Exception as exc:  # noqa: BLE001
            raise MilvusQueryError(
                f"could not connect to Milvus at {target}: {exc}{_grpc_cause(exc)}"
            ) from exc
        _CLIENTS[key] = client
        return client


# Optional, best-effort embedding usage hook. A process that accounts for its
# LLM spend (the agentic-tools mcp-server, via llm_usage.py) registers a
# callable here; every other importer leaves it unset and embed_query behaves
# exactly as before. The recorder gets (provider, model, input_tokens) once per
# successful remote embedding response, where provider is
# SELECTED_EMBEDDING_PROVIDER as set (possibly empty) and input_tokens is None
# when the response carried no usage object. Deterministic local embeddings make
# no request and are not reported. The hook can never fail a search: anything
# it raises is logged and swallowed.
_usage_recorder: Optional[Callable[[str, str, Optional[int]], None]] = None


def set_usage_recorder(recorder: Optional[Callable[[str, str, Optional[int]], None]]) -> None:
    """Register (or, with ``None``, clear) the embedding usage recorder."""
    global _usage_recorder
    _usage_recorder = recorder


def _report_embedding_usage(provider: str, model: str, response: Any) -> None:
    recorder = _usage_recorder
    if recorder is None:
        return
    try:
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "prompt_tokens", None)
        if tokens is None:
            tokens = getattr(usage, "total_tokens", None)
        recorder(provider, model, None if tokens is None else int(tokens))
    except Exception as exc:  # noqa: BLE001 -- accounting never fails a search
        logger.warning("embedding usage for %s was not recorded: %s", model, exc)


def embed_query(text: str, *, model: Optional[str] = None, client: Any = None) -> List[float]:
    """Embed a query string with the selected embedding provider's API (or
    deterministic local embeddings when no remote embedding provider is active).

    The same model must have produced the collection's vectors; embedding a query
    with a different model yields a vector in a different space, so the search
    would return confident nonsense rather than failing.
    """
    if not text or not text.strip():
        raise MilvusQueryError("cannot embed an empty query")

    model = (
        model
        or os.getenv("SELECTED_EMBEDDING_MODEL")
        or DEFAULT_EMBED_MODEL
    )
    dim = _resolve_embed_dim(model)

    if client is not None:
        try:
            response = client.embeddings.create(input=text, model=model)
        except Exception as exc:  # noqa: BLE001
            raise MilvusQueryError(f"embedding the query failed ({model}): {exc}") from exc
        _report_embedding_usage(
            (os.getenv("SELECTED_EMBEDDING_PROVIDER") or "").strip().upper(), model, response
        )
        return response.data[0].embedding

    embed_provider = (os.getenv("SELECTED_EMBEDDING_PROVIDER") or "").strip().upper()
    embed_api_key = (os.getenv("SELECTED_EMBEDDING_API_KEY") or "").strip()
    embed_base_url = (os.getenv("SELECTED_EMBEDDING_BASE_URL") or "").strip()

    api_key = embed_api_key or (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise MilvusQueryError(
            "OPENAI_API_KEY is not set, so the query cannot be embedded and "
            "semantic search cannot run"
        )

    if embed_provider == "LOCAL":
        return _deterministic_hash_embedding(text, dim=dim)

    if not embed_provider and not embed_api_key:
        # Legacy path when SELECTED_EMBEDDING_* variables were not resolved:
        # only call OpenAI directly when the chat provider is OpenAI.
        base_url_lower = (os.getenv("OPENAI_BASE_URL") or "").strip().lower()
        chat_provider = (os.getenv("SELECTED_LLM_PROVIDER") or "").strip().upper()
        if (chat_provider and chat_provider != "OPENAI") or (
            base_url_lower and "api.openai.com" not in base_url_lower
        ):
            return _deterministic_hash_embedding(text, dim=dim)

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise MilvusQueryError(
            "the openai package is not installed, so query embedding cannot run"
        ) from exc

    target_base_url = embed_base_url or (os.getenv("OPENAI_BASE_URL") or "").strip() or None
    is_nvidia = embed_provider == "NVIDIA" or (
        target_base_url is not None and "integrate.api.nvidia.com" in target_base_url.lower()
    )

    try:
        oa_client = OpenAI(api_key=api_key, base_url=target_base_url)
        if is_nvidia:
            try:
                response = oa_client.embeddings.create(
                    input=text,
                    model=model,
                    extra_body={"input_type": "query", "truncate": "END"},
                )
            except Exception:
                response = oa_client.embeddings.create(input=text, model=model)
        else:
            response = oa_client.embeddings.create(input=text, model=model)
        vector = response.data[0].embedding
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Embeddings API (%s) unavailable (%s); falling back to deterministic vector",
            model, exc,
        )
        return _deterministic_hash_embedding(text, dim=dim)
    _report_embedding_usage(embed_provider, model, response)
    return vector


def _path_from(record: Dict[str, Any]) -> str:
    """Recover the source document path from a stored record.

    Collections in this repo store a ``metadata`` field, sometimes as a JSON
    string and sometimes as an object, so both are handled. A hit with no path is
    returned with an empty one rather than dropped -- losing a result silently
    would misreport what the agent actually retrieved.
    """
    metadata = record.get("metadata")
    if isinstance(metadata, str):
        import json

        try:
            metadata = json.loads(metadata)
        except (ValueError, TypeError):
            metadata = {"raw": metadata}
    if isinstance(metadata, dict):
        for key in ("path", "source", "file", "document", "doc_path"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
    for key in ("path", "source"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def search(query: str, *, collection: Optional[str] = None, limit: int = DEFAULT_LIMIT,
           client: Any = None, embed_client: Any = None,
           embed_model: Optional[str] = None) -> List[SemanticHit]:
    """Run a semantic search and return hits carrying source paths.

    Raises ``MilvusQueryError`` if the collection is missing or the search fails.
    A missing collection is a real, actionable misconfiguration (the corpus was
    never ingested), not an empty result.
    """
    collection = collection or DEFAULT_COLLECTION
    milvus = client or get_client()

    try:
        exists = milvus.has_collection(collection)
    except Exception as exc:  # noqa: BLE001
        if client is None:
            _evict_client(milvus)
        raise MilvusQueryError(
            f"could not query Milvus for collection {collection!r}: {exc}{_grpc_cause(exc)}"
        ) from exc
    if not exists:
        raise MilvusQueryError(
            f"Milvus collection {collection!r} does not exist -- the corpus has not "
            "been ingested. Seed it before querying; semantic search will not "
            "pretend the corpus is empty."
        )

    vector = embed_query(query, model=embed_model, client=embed_client)

    try:
        results = milvus.search(
            collection_name=collection,
            data=[vector],
            limit=limit,
            search_params={"metric_type": METRIC_TYPE, "params": {}},
            output_fields=["text", "metadata"],
        )
    except Exception as exc:  # noqa: BLE001
        if client is None:
            _evict_client(milvus)
        raise MilvusQueryError(
            f"Milvus search failed on {collection!r}: {exc}{_grpc_cause(exc)}"
        ) from exc

    hits: List[SemanticHit] = []
    for group in results or []:
        for item in group:
            record = item.get("entity") if isinstance(item, dict) else {}
            record = record or {}
            hits.append(SemanticHit(
                path=_path_from(record),
                text=record.get("text", "") or "",
                score=float(item.get("distance", 0.0)) if isinstance(item, dict) else 0.0,
                metadata=record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
            ))

    logger.info("[MILVUS] %r -> %d hit(s) from %s", query, len(hits), collection)
    return hits
