"""Ingest the demo corpus into Milvus so `search_semantic` has something to find.

Why this exists
---------------
Milvus was previously outside the demo data path entirely: the (now archived)
`milvus-ragchat` demo seeds it by `docker exec`-ing a script against a corpus baked into the image. That
does not work here, because agentic-tools' corpus is the SAME corpus the other
three tools search -- and it has to be, or the agent's tool choice stops being a
choice between views of one document store and becomes a choice between different
stores with different contents.

Where it runs
-------------
This is a sibling of `seed.py` in `demos/agentic-tools/seed/`, so it runs in two
places with one implementation:

  - inside the one-shot `seed` compose service (the mcp-server image, this
    directory mounted read-only at `/seed`), where `MILVUS_HOST` is the compose
    network alias `milvusdb` and `shared.milvus_query` is `/app/shared/milvus_query`;
  - on the host, through the CLI `demos/agentic-tools/ingest_milvus.py`, which
    re-exports this module, defaults `MILVUS_HOST` to `localhost` (the published
    port) and resolves the embedding provider from the repo `.env`.

It imports only the stdlib at module load; `pymilvus` and `shared.milvus_query`
are imported inside `ingest()`, after the cheap checks. The caller puts the
directory holding `shared/` on `sys.path`.

The path contract (the important part)
--------------------------------------
Each chunk is stored with `metadata.path` set to the **same path string the
filesystem and BM25 tools return** for that document -- `<bucket>/<user>/<file>`,
matching how the MCP server addresses MinIO objects.

This is not cosmetic. Ticket 11's provenance-resolves invariant checks that every
path the audit cites is a document the agent actually retrieved, and the
citation-set outcome gate compares retrieved paths against ground truth. If
semantic hits came back under a different path spelling than BM25 hits, every
cross-check between them would report a mismatch that isn't real, and the
duplicate/stale pillar could not tell that two tools had returned the same
document.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Checkout root: seed -> agentic-tools -> demos -> root. Inside the container the
# file is /seed/ingest_milvus.py, whose parents are only /seed and /, so the root
# falls back to / there (the data dir then comes from DEMO_DATA_DIR_HOST).
_p = Path(__file__).resolve().parents
REPO_ROOT = _p[3] if len(_p) > 3 else Path("/")

DEFAULT_BUCKET = os.getenv("DEMO_DEFAULT_BUCKET", "contracts")
COLLECTION = os.getenv("MILVUS_COLLECTION", "agentic_tools_corpus")
EMBED_MODEL = os.getenv("SELECTED_EMBEDDING_MODEL", "text-embedding-3-small")
TEXT_EXTS = {".md", ".txt"}

# Default OpenAI embedding dimensions when SELECTED_EMBEDDING_DIM is not set.
EMBED_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

CHUNK_SIZE = int(os.getenv("MILVUS_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("MILVUS_CHUNK_OVERLAP", "100"))

_DEFAULT_DATA_DIR = REPO_ROOT / "data" / "agentgov-docs"


def log(msg: str) -> None:
    print(f"[milvus-ingest] {msg}", flush=True)


def iter_documents(data_path: Path) -> Iterable[Tuple[str, str]]:
    """Yield ``(path, text)`` for every corpus document.

    The layout logic mirrors the MinIO seed exactly -- bucketed
    (``<user>/<bucket>/<file>``) or flat (``<user>/<file>`` into the default
    bucket) -- so the emitted paths line up with the object keys the other tools
    see. Kept in step with `demos/agentgov-mimic/seed.py` (vendored here as
    `_agentgov_seed.py`); if that layout changes, this must change with it.
    """
    for user_dir in sorted(data_path.iterdir()):
        if not user_dir.is_dir():
            continue
        user = user_dir.name
        subdirs = [d for d in sorted(user_dir.iterdir()) if d.is_dir()]
        flat_files = [f for f in sorted(user_dir.iterdir())
                      if f.is_file() and f.suffix in TEXT_EXTS]

        if subdirs:
            for bucket_dir in subdirs:
                bucket = bucket_dir.name
                for filepath in sorted(f for f in bucket_dir.iterdir()
                                       if f.is_file() and f.suffix in TEXT_EXTS):
                    yield f"{bucket}/{user}/{filepath.name}", filepath.read_text(
                        encoding="utf-8", errors="replace")
        else:
            for filepath in flat_files:
                yield f"{DEFAULT_BUCKET}/{user}/{filepath.name}", filepath.read_text(
                    encoding="utf-8", errors="replace")


def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split on paragraph-ish boundaries, falling back to a hard cut.

    Deliberately dependency-free: pulling langchain's text splitters into the
    seed path would add a heavy import for a job that is a few lines of string
    handling.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Prefer a paragraph break, then a line break, near the end.
            for sep in ("\n\n", "\n", ". "):
                cut = text.rfind(sep, start + size // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def ingest(data_dir: Path, drop: bool = True, *, embed_model: Optional[str] = None) -> int:
    """Embed every corpus document into ``COLLECTION``. Returns 0 on success.

    ``embed_model`` defaults to this module's ``EMBED_MODEL``; the host CLI passes
    its own (resolved from the repo ``.env``) so an override set there reaches
    this function.
    """
    model = embed_model or EMBED_MODEL
    # Cheap, actionable checks FIRST, before importing pymilvus or touching the
    # network. A missing corpus or an unknown embedding model should say exactly
    # that, not fail with an unrelated ImportError that sends the reader looking
    # in the wrong place.
    if not data_dir.is_dir():
        log(f"ERROR: corpus directory not found at {data_dir}")
        return 1

    dim = EMBED_DIMS.get(model)
    if dim is None:
        env_dim = (os.getenv("SELECTED_EMBEDDING_DIM") or "").strip()
        selected_model = (os.getenv("SELECTED_EMBEDDING_MODEL") or "").strip()
        if selected_model and model == selected_model and env_dim.isdigit() and int(env_dim) > 0:
            dim = int(env_dim)
    if dim is None:
        log(f"ERROR: unknown embedding model {model!r}; cannot size the vector "
            f"field. Known: {sorted(EMBED_DIMS)}")
        return 1

    if not (os.getenv("SELECTED_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")):
        log("ERROR: No embedding API key (SELECTED_EMBEDDING_API_KEY or OPENAI_API_KEY) "
            "is set, so documents cannot be embedded. Refusing to create an empty "
            "collection that would make semantic search silently return nothing.")
        return 1

    documents = list(iter_documents(data_dir))
    if not documents:
        log(f"ERROR: no .md/.txt documents found under {data_dir}")
        return 1
    log(f"found {len(documents)} document(s) under {data_dir}")

    try:
        from pymilvus import DataType
    except ImportError:
        log("ERROR: pymilvus is not installed, so the corpus cannot be ingested.")
        log("       The mcp-server image (the `seed` compose service) carries it.")
        log("       For a host run (demo.py invokes seed.py directly) install it")
        log("       into the venv you run demo.py with:")
        log("           pip install 'pymilvus>=2.3,<3'")
        log("       Without it search_semantic has no collection to query and will")
        log("       report a missing collection on every call.")
        return 1
    from shared.milvus_query import MilvusQueryError, embed_query, get_client

    try:
        client = get_client()
    except MilvusQueryError as exc:
        log(f"ERROR: {exc}")
        return 1

    if drop and client.has_collection(COLLECTION):
        log(f"dropping existing collection {COLLECTION}")
        client.drop_collection(COLLECTION)

    if not client.has_collection(COLLECTION):
        schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("metadata", DataType.VARCHAR, max_length=65535)
        index_params = client.prepare_index_params()
        # Inner product, matching how the query side searches.
        index_params.add_index(field_name="vector", metric_type="IP")
        client.create_collection(
            collection_name=COLLECTION, schema=schema,
            index_params=index_params, consistency_level="Strong",
        )
        log(f"created collection {COLLECTION} (dim={dim}, metric=IP)")

    rows: List[Dict[str, object]] = []
    for path, text in documents:
        pieces = chunk(text)
        if not pieces:
            log(f"  WARNING: {path} is empty after stripping; skipped")
            continue
        for index, piece in enumerate(pieces):
            try:
                vector = embed_query(piece, model=model)
            except MilvusQueryError as exc:
                log(f"ERROR embedding {path} chunk {index}: {exc}")
                return 1
            rows.append({
                "vector": vector,
                "text": piece,
                # `path` here MUST match what read_file/list_files/search_files_bm25
                # report for the same document -- see the module docstring.
                "metadata": json.dumps({"path": path, "chunk": index,
                                        "chunks": len(pieces)}),
            })
        log(f"  {path}: {len(pieces)} chunk(s)")

    if not rows:
        log("ERROR: nothing to insert after chunking")
        return 1

    client.insert(collection_name=COLLECTION, data=rows)
    client.flush(COLLECTION)
    log(f"inserted {len(rows)} chunk(s) from {len(documents)} document(s) "
        f"into {COLLECTION}")
    return 0
