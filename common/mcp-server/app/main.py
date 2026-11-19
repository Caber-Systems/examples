#!/usr/bin/env python3
"""
Direct MCP Filesystem Server - HTTP Implementation

Provides direct filesystem access via the Model Context Protocol (MCP)
over HTTP using JSON-RPC 2.0. No tenant awareness -- serves ALL files
to ANY requester. PyCaber auto-instrumentation transparently intercepts
inbound requests via CaberASGIMiddleware.

Key Features:
- Direct filesystem or S3 compatible object store-backed file operations
- MCP JSON-RPC 2.0 protocol over HTTP
- Session management for MCP clients
- BM25 search index over all files
- PyCaber auto-instrumentation (ASGI middleware for inbound, patches for outbound)

Protocol conformance
--------------------
This is JSON-RPC 2.0 over a single HTTP POST -- **not** MCP Streamable HTTP,
which additionally requires an SSE channel on GET, resumability, server-initiated
messages, and real session lifecycle. `GET /` reports the accurate label.

A real MCP client *can* now drive it. It could not before: the ticket-02 audit
showed the official client died on the mandatory `notifications/initialized`
right after `initialize`, and every `tools/call` result failed client-side schema
validation. Ticket 12 chose the surgical fix over adopting the official SDK,
because ticket 15 moved Caber's interception to the client-side framework
tool-call wrapper -- so this server carries no Caber middleware and is a faithful
stand-in for a customer's own third-party MCP server.

Implemented: `202 Accepted` for notifications, spec-shaped `CallToolResult`,
JSON-RPC errors inside HTTP 200, `ping`, `protocolVersion` negotiation, and `405`
on the standalone GET.

Deliberately deferred, recorded as known gaps (ticket 12, not oversights):
the `GET /mcp` SSE channel (the demo has no server-initiated messages), session
validation, and `Origin` checking -- anti-bypass is the proxy's job per ticket 15.
`Origin`-unchecked is a conscious spec-MUST gap.

Gate: `tests/conformance/test_mcp_spec_conformance.py`, which drives this server
with the official SDK client and is green on both the `mcp` 1.x and 2.x lines.

Ported from ZeroTrustDemos/CiscoDuo/CaberApp/mcp-server/app/main.py
"""

import asyncio
import importlib.util
import io
import json
import logging
import math
import os
import re
import sys
import time
import uuid
import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
from urllib.parse import unquote

# ---------------------------------------------------------------------------
# Logging -- must be configured BEFORE importing PyCaber so that the
# pyo3-log bridge (established at import time) finds Python handlers.
# ---------------------------------------------------------------------------
VERBOSE_MODE = os.getenv("MCP_VERBOSE", "false").lower() == "true"

logging.basicConfig(
    level=logging.DEBUG if VERBOSE_MODE else logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
)
logger = logging.getLogger("mcp_server")
logger.setLevel(logging.DEBUG)

# Always enable pipeline and stream_process debug for OBO diagnostics
logging.getLogger("pycaber").setLevel(logging.INFO)
logging.getLogger("pycaber.auto").setLevel(logging.INFO)
# logging.getLogger("pycaber.config").setLevel(logging.DEBUG)
logging.getLogger("pycaber.pipeline").setLevel(logging.DEBUG)
# logging.getLogger("pycaber.intercept.redact_event").setLevel(logging.DEBUG)
# logging.getLogger("pycaber.path_redaction").setLevel(logging.DEBUG)

# OBO auth-split diagnostics: pyo3-log maps Rust TRACE -> Python DEBUG, which the
# INFO root handler drops -- so the per-edge "auth.allow" / AUTHORIZED|DENIED
# decisions never reach the log. Attach a dedicated DEBUG handler directly to the
# OBO logger (propagate=False to avoid duplicating the INFO summaries via root) so
# the edge-level authorization detail is captured. Toggle off with MCP_OBO_TRACE=false.
if os.getenv("MCP_OBO_TRACE", "true").lower() != "false":
    _obo_logger = logging.getLogger("pycaber.pipeline.obo")
    _obo_trace_handler = logging.StreamHandler()
    _obo_trace_handler.setLevel(logging.DEBUG)
    _obo_trace_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s"))
    _obo_logger.addHandler(_obo_trace_handler)
    _obo_logger.propagate = False

# ---------------------------------------------------------------------------
# PyCaber auto-instrumentation (must precede FastAPI import)
# Setting the configuration variables for PyCaber Auto must happen before
# importing pycaber.auto
# ---------------------------------------------------------------------------
os.environ.setdefault("CSI_CABER_HOST", "http://caber-core:5556")
os.environ.setdefault("CSI_API_KEY", "abc123")
os.environ.setdefault("CSI_GLOBAL_REDACT_ENABLED", "false")

try:
    import pycaber.auto  # noqa: F401 -- bootstrap: patches httpx/requests + auto-installs CaberASGIMiddleware on FastAPI
    _has_pycaber = True
except ImportError:
    _has_pycaber = False

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
import uvicorn

# MinIO client
try:
    from minio import Minio
    MINIO_AVAILABLE = True
except ImportError:
    Minio = None
    MINIO_AVAILABLE = False

# BM25 search functionality
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None
    BM25_AVAILABLE = False

if BM25_AVAILABLE:
    class LuceneIdfBM25(BM25Okapi):
        """BM25Okapi scoring with Lucene's always-positive IDF.

        rank_bm25's Okapi IDF, ``log((N - n + 0.5) / (n + 0.5))``, is exactly 0
        for a term in half the corpus and floored to a small epsilon above half.
        A demo corpus is a few chains of near-identical versions, so a chain's
        distinctive terms routinely sit in half the documents: every document
        then scores 0.0 and the ranking is argsort's tie order. Lucene (and so
        Elasticsearch/OpenSearch) uses ``log(1 + (N - n + 0.5) / (n + 0.5))``,
        positive for every term, while a document holding no query term still
        scores 0.
        """

        def _calc_idf(self, nd):
            for word, freq in nd.items():
                self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
else:
    LuceneIdfBM25 = None

# ---------------------------------------------------------------------------
# PyCaber globals
# ---------------------------------------------------------------------------

# Bucket discovery: MCP_MINIO_BUCKETS (comma-separated) takes priority.
# Falls back to legacy READ_MINIO_CONTRACTS/CUSTOMERS two-slot system.
MCP_MINIO_BUCKETS = os.getenv("MCP_MINIO_BUCKETS", "")

READ_MINIO_CONTRACTS = os.getenv("READ_MINIO_CONTRACTS", "true").lower() == "true"
READ_MINIO_CUSTOMERS = os.getenv("READ_MINIO_CUSTOMERS", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Server profile
#
# This source is SHARED with agentgov-mimic (and, before they were archived,
# -duo, -strongdm, -cloudflare and streamlit-mock-mcp), which ticket 05 requires
# be kept frozen and never put at risk. Those demos build the image from this
# same source, so pinning an image
# tag does not actually protect them -- compose rebuilds when the image is absent.
# A profile switch does, while keeping ONE implementation rather than a forked
# copy of the server (which would be the very stale-duplicate problem this demo
# sells against).
#
#   legacy         (default) -- the seven-tool catalog, BM25 falling back to
#                               substring search, /redaction demo routes on.
#                               Exactly what the existing demos have today.
#   agentic-tools            -- the four-tool catalog (ticket 03), BM25 that fails
#                               loud, search_semantic over Milvus, no /redaction
#                               back door.
#   graphrag                 -- the agentic-tools catalog with `search_graph` in
#                               the slot search_semantic occupies: a thin client
#                               of CaberApp/graphrag-retriever's policy-aware
#                               scoped-search over the Neo4j lineage graph. No
#                               Milvus, no embeddings. Everything else matches
#                               agentic-tools byte for byte.
# ---------------------------------------------------------------------------
SERVER_PROFILE = os.getenv("MCP_SERVER_PROFILE", "legacy").strip().lower()
AGENTIC_PROFILE = SERVER_PROFILE == "agentic-tools"
GRAPHRAG_PROFILE = SERVER_PROFILE == "graphrag"
# The agentic-tools behaviours (four-tool catalog, fail-loud BM25) that graphrag
# inherits unchanged. Only the fourth tool differs between the two profiles.
FOUR_TOOL_PROFILE = AGENTIC_PROFILE or GRAPHRAG_PROFILE

# GraphRAG retriever backing the `search_graph` tool under the graphrag profile.
# The retriever governs the candidate frontier before any content is expanded,
# so what comes back is already scoped to the OBO user this server forwards.
GRAPHRAG_URL = os.getenv("GRAPHRAG_URL", "http://graphrag-retriever:8084").rstrip("/")
GRAPHRAG_TIMEOUT = float(os.getenv("GRAPHRAG_TIMEOUT", "60"))

# Milvus collection backing the `search_semantic` tool. The corpus is ingested
# into this collection by the demo's seed step; querying a collection that does
# not exist is reported as an error, never as an empty result.
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "agentic_tools_corpus")

# The shared Milvus query module lives at <repo>/shared/milvus_query. In the
# container it is copied to /app/shared; when the server is run straight from a
# checkout, the repo root is two levels above this file. Adding the parent to
# sys.path keeps `from shared.milvus_query import ...` working in both.
_APP_DIR = Path(__file__).resolve().parent
for _candidate in (_APP_DIR.parent, _APP_DIR.parent.parent.parent):
    if (_candidate / "shared" / "milvus_query").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break


# save_file accepts only ai-notes/<name>; a name of only dots is refused, and
# ".." anywhere is also refused where it is used.
_SAVE_FILE_PATH_RE = re.compile(r"ai-notes/(?!\.+\Z)[A-Za-z0-9._-]{1,120}")


_DIRECTORY_LOOKUP = None  # loaded once, at first save_file


def _load_directory_lookup():
    """The agentic-app's stdlib-only ``directory_lookup``, loaded by path.

    The image copies it to /app/directory_lookup.py beside ``app/`` (with the
    seed at /app/zt_mimic_seed/directory.json); a checkout has it at
    CaberApp/agentic-app/directory_lookup.py. One module, never duplicated here.
    """
    global _DIRECTORY_LOOKUP
    if _DIRECTORY_LOOKUP is not None:
        return _DIRECTORY_LOOKUP
    for candidate in (
        _APP_DIR.parent / "directory_lookup.py",
        _APP_DIR.parent.parent / "agentic-app" / "directory_lookup.py",
    ):
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("directory_lookup", candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _DIRECTORY_LOOKUP = module
            return module
    raise RuntimeError("save_file cannot find directory_lookup.py")


def _register_embedding_usage_recorder() -> None:
    """Record search_semantic's embedding usage to the demo usage JSONL.

    ``llm_usage.py`` sits beside ``app/`` (copied to /app/llm_usage.py in the
    image; CaberApp/mcp-server/llm_usage.py in a checkout) and is loaded by path.
    When it or the shared query module is absent the server runs unchanged,
    with a log line saying usage is not recorded. Nothing is registered unless
    ``CABER_DEMO_USAGE_DIR`` is set (recording is opt-in per compose file).
    ``embedding_provider`` is written whenever the embedding provider differs
    from the chat provider. The recorder itself never
    fails a search (shared/milvus_query swallows and logs anything it raises).
    """
    usage_path = _APP_DIR.parent / "llm_usage.py"
    if not usage_path.is_file():
        logger.info("[USAGE] %s is absent; embedding usage is not recorded", usage_path)
        return
    try:
        spec = importlib.util.spec_from_file_location("llm_usage", usage_path)
        llm_usage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(llm_usage)
        from shared.milvus_query import set_usage_recorder
    except Exception as exc:  # noqa: BLE001 -- accounting never stops the server
        logger.warning("[USAGE] embedding usage recorder not registered: %s", exc)
        return
    if not llm_usage.recording_enabled():
        # Opt-in: only agentic-tools' compose sets CABER_DEMO_USAGE_DIR. Other
        # demos sharing this image register nothing and log nothing.
        return

    def _record(embed_provider: str, model: str, tokens: Optional[int]) -> None:
        chat_provider = llm_usage.selected_chat_provider() or embed_provider.lower()
        llm_usage.record_or_warn(
            logger, "mcp-server", chat_provider, "embedding", model, tokens or 0, 0,
            embedding_provider=llm_usage.embedding_provider_for(chat_provider),
            usage_missing=tokens is None,
        )

    set_usage_recorder(_record)
    logger.info("[USAGE] embedding usage is recorded to %s", llm_usage.usage_dir_from_env())

if not MCP_MINIO_BUCKETS and not READ_MINIO_CONTRACTS and not READ_MINIO_CUSTOMERS:
    READ_MINIO_CONTRACTS = True


# ---------------------------------------------------------------------------
# Timeout helper (threading-based, Streamlit compatible)
# ---------------------------------------------------------------------------
def with_timeout(func, timeout_seconds, *args, **kwargs):
    """Execute function with timeout using threading."""
    result_queue = queue.Queue()

    def target():
        try:
            result = func(*args, **kwargs)
            result_queue.put(("success", result))
        except Exception as e:
            result_queue.put(("error", e))

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")

    try:
        status, value = result_queue.get_nowait()
        if status == "error":
            raise value
        return value
    except queue.Empty:
        raise RuntimeError("Thread completed but no result was produced")


# ---------------------------------------------------------------------------
# MCP protocol versions
#
# The server previously hardcoded the reply to "2024-11-05" regardless of what
# the client asked for. That is the pre-Streamable-HTTP revision, and it worked
# only because the string is still in the official SDK's
# SUPPORTED_PROTOCOL_VERSIONS -- a time bomb that fails *silently at initialize*
# the moment it is dropped. Per the lifecycle spec the server MUST echo the
# requested version when it supports it, and otherwise MUST answer with a
# version it does support (SHOULD be its latest).
# ---------------------------------------------------------------------------
SUPPORTED_PROTOCOL_VERSIONS = ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"]
LATEST_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[-1]

# JSON-RPC 2.0 error codes (https://www.jsonrpc.org/specification#error_object)
JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
# The identity this server acts as when the request names no acting user.
MCP_SERVER_IDENTITY = "mcpserver_identity"


def baggage_member(baggage: str, name: str) -> str:
    """Return member ``name`` of a W3C ``baggage`` header value, decoded.

    Values are percent-encoded per the W3C Baggage spec; member properties
    (``;prop=x``) are dropped. Returns ``""`` when the header or the member is
    absent.
    """
    for member in (baggage or "").split(","):
        key, sep, value = member.partition("=")
        if sep and key.strip() == name:
            return unquote(value.split(";", 1)[0].strip())
    return ""


def baggage_user(baggage: str) -> str:
    """Return the acting user from a W3C ``baggage`` header value.

    The agent carries the acting user only in baggage ``caber.user``;
    X-Forwarded-User is not read. Returns ``""`` when absent.
    """
    return baggage_member(baggage, "caber.user")


def baggage_email(baggage: str) -> str:
    """Return the acting user's email from baggage ``caber.email``.

    PyCaber writes it next to ``caber.user`` when the host policy allows the
    destination. ``search_graph`` forwards it so the retriever's governance can
    take the email identity fallback; it is never trusted here. Returns ``""``
    when absent.
    """
    return baggage_member(baggage, "caber.email")


class MCPSession:
    """Represents an MCP client session (no tenant binding)."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.initialized = False
        self.client_info = {}
        self.capabilities = {}
        self.protocol_version = LATEST_PROTOCOL_VERSION
        # Identity the agent sent on the most recent request (baggage
        # `caber.user` and `caber.email`). `search_graph` hands these to the
        # retriever as the OBO principal and its acting email. The principal's
        # groups, roles and tenant are never taken from traffic: the
        # retriever's governance resolves them from the identity replica.
        self.acting_user = MCP_SERVER_IDENTITY
        self.acting_email = ""


# ---------------------------------------------------------------------------
# DirectMCPServer -- tenant-unaware, ObjStor-backed, PyCaber-enabled
# ---------------------------------------------------------------------------
class DirectMCPServer:
    """Direct MCP filesystem server with ObjStor support and PyCaber lineage."""

    def __init__(self, files_root: str = "/mcp-files"):
        self.files_root = Path(files_root)
        self.sessions: Dict[str, MCPSession] = {}
        self.request_id = 0

        # BM25 search index
        self.bm25_index = None
        self.indexed_documents = []
        self.doc_metadata = []
        self.bm25_initialized = False
        # Guards index rebuilds run by the eager background builder.
        self._index_lock = asyncio.Lock()

        # MinIO configuration
        self.use_minio = os.getenv("MCP_USE_MINIO", "false").lower() == "true"
        self.minio_clients = {}
        self.minio_buckets = {}

        if self.use_minio and MINIO_AVAILABLE:
            self._init_minio_clients()
        elif self.use_minio and not MINIO_AVAILABLE:
            raise ValueError("MinIO support requested but minio package not available")
        else:
            if not self.files_root.exists():
                raise ValueError(f"Files root directory does not exist: {files_root}")

        # Build BM25 index
        if BM25_AVAILABLE:
            try:
                self._build_search_index()
                logger.info("[BM25] startup index build complete: index_built=%s docs=%d",
                            self.bm25_initialized, len(self.indexed_documents))
            except Exception as e:
                logger.warning("[BM25] startup index build FAILED: %s -- queries will fall "
                               "back to exact-substring search", e)
        else:
            logger.warning("[BM25] rank_bm25 NOT INSTALLED -- all queries use exact-substring "
                           "search (_search_files); BM25 scoring unavailable")

    # ------------------------------------------------------------------
    # MinIO client initialization
    # ------------------------------------------------------------------

    def _init_minio_clients(self):
        """Initialize MinIO clients from MCP_MINIO_BUCKETS or legacy env vars."""
        try:
            logger.debug("[ObjStor] Initializing MinIO clients...")
            access_key = os.getenv("MCP_MINIO_ACCESS_KEY", "admin")
            secret_key = os.getenv("MCP_MINIO_SECRET_KEY", "caberdev")
            secure = os.getenv("MCP_MINIO_SECURE", "false").lower() == "true"
            logger.debug(f"[ObjStor] Using access_key={access_key}, secure={secure}")

            initialized_clients = []

            if MCP_MINIO_BUCKETS:
                endpoint = os.getenv("MCP_MINIO_ENDPOINT", "csi-dep-objstor:9000")
                for bucket_name in MCP_MINIO_BUCKETS.split(","):
                    bucket_name = bucket_name.strip()
                    if not bucket_name:
                        continue
                    logger.debug(f"[ObjStor] Bucket config: endpoint={endpoint}, bucket={bucket_name}")
                    self.minio_clients[bucket_name] = Minio(
                        endpoint,
                        access_key=access_key,
                        secret_key=secret_key,
                        secure=secure,
                    )
                    self.minio_buckets[bucket_name] = bucket_name
                    initialized_clients.append(f"{bucket_name}: {endpoint}/{bucket_name}")
            else:
                if READ_MINIO_CONTRACTS:
                    contracts_endpoint = os.getenv("MCP_MINIO_CONTRACTS_ENDPOINT", "csi-dep-objstor:9000")
                    contracts_bucket = os.getenv("MCP_MINIO_CONTRACTS_BUCKET", "contracts")
                    logger.debug(f"[ObjStor] Contracts config: endpoint={contracts_endpoint}, bucket={contracts_bucket}")
                    self.minio_clients["contracts"] = Minio(
                        contracts_endpoint,
                        access_key=access_key,
                        secret_key=secret_key,
                        secure=secure,
                    )
                    self.minio_buckets["contracts"] = contracts_bucket
                    initialized_clients.append(f"Contracts: {contracts_endpoint}/{contracts_bucket}")
                else:
                    logger.info("[ObjStor] Contracts data source disabled (READ_MINIO_CONTRACTS=false)")

                if READ_MINIO_CUSTOMERS:
                    customers_endpoint = os.getenv("MCP_MINIO_CUSTOMERS_ENDPOINT", "csi-dep-objstor:9000")
                    customers_bucket = os.getenv("MCP_MINIO_CUSTOMERS_BUCKET", "customers")
                    logger.debug(f"[ObjStor] Customers config: endpoint={customers_endpoint}, bucket={customers_bucket}")
                    self.minio_clients["customers"] = Minio(
                        customers_endpoint,
                        access_key=access_key,
                        secret_key=secret_key,
                        secure=secure,
                    )
                    self.minio_buckets["customers"] = customers_bucket
                    initialized_clients.append(f"Customers: {customers_endpoint}/{customers_bucket}")
                else:
                    logger.info("[ObjStor] Customers data source disabled (READ_MINIO_CUSTOMERS=false)")

            if initialized_clients:
                logger.info("MinIO clients initialized:")
                for client_info in initialized_clients:
                    logger.info(f"   {client_info}")
            else:
                logger.warning("[ObjStor] No MinIO clients initialized - all data sources disabled")

        except Exception as e:
            logger.error(f"[ObjStor] Failed to initialize MinIO clients: {e}")
            raise

    # ------------------------------------------------------------------
    # MinIO file operations
    # ------------------------------------------------------------------

    def _get_all_minio_files(self) -> List[str]:
        """Get all files from enabled MinIO buckets."""
        logger.debug("[ObjStor] Getting all files from MinIO buckets...")
        all_files = []

        for bucket_type in self.minio_clients:
            if bucket_type not in self.minio_clients:
                continue

            client = self.minio_clients[bucket_type]
            bucket_name = self.minio_buckets[bucket_type]
            logger.debug(f"[ObjStor] Listing objects from {bucket_type} bucket: {bucket_name}")

            try:
                objects = client.list_objects(bucket_name, recursive=True)
                file_count = 0
                for obj in objects:
                    if "mcpserver" in obj.object_name:
                        logger.debug(f"[ObjStor] Skipping duplicate mcpserver file: {obj.object_name}")
                        continue
                    file_path = f"{bucket_type}/{obj.object_name}"
                    all_files.append(file_path)
                    file_count += 1
                    logger.debug(f"[ObjStor] Found file: {file_path} (size: {obj.size} bytes)")
                logger.debug(f"[ObjStor] Listed {file_count} files from {bucket_type} bucket")
            except Exception as e:
                logger.warning(f"[ObjStor] Failed to list objects from {bucket_type} bucket: {e}")

        logger.debug(f"[ObjStor] Total files found across all buckets: {len(all_files)}")
        return all_files

    def _read_minio_file(self, file_path: str) -> str:
        """Read file content from MinIO."""
        logger.debug(f"[ObjStor] Reading file: {file_path}")

        if "/" not in file_path:
            raise ValueError(f"Invalid MinIO path format. Expected 'bucket_type/filename', got: {file_path}")

        bucket_type, object_name = file_path.split("/", 1)

        if bucket_type not in self.minio_clients:
            raise ValueError(f"Unknown bucket type: {bucket_type}")

        client = self.minio_clients[bucket_type]
        bucket_name = self.minio_buckets[bucket_type]

        try:
            response = client.get_object(bucket_name, object_name)
            content = response.read().decode("utf-8")
            response.close()
            response.release_conn()
            logger.debug(f"[ObjStor] Successfully read {file_path}: {len(content)} bytes")
            return content
        except Exception as e:
            logger.error(f"[ObjStor] Failed to read {file_path} from MinIO: {e}")
            raise ValueError(f"Failed to read {file_path} from MinIO: {e}")

    def _list_minio_objects(self, path: str = "") -> List[str]:
        """List objects in enabled MinIO buckets."""
        all_items = []

        if not path or path == "/":
            for bucket_type in self.minio_clients:
                all_items.append(f"{bucket_type}/")
            return all_items

        bucket_type = None
        prefix = ""
        for bt in self.minio_clients:
            if path == bt or path.startswith(bt + "/"):
                bucket_type = bt
                prefix = path[len(bt) + 1:] if len(path) > len(bt) + 1 else ""
                break

        if not bucket_type:
            available = [f"/{bt}/" for bt in self.minio_clients]
            raise FileNotFoundError(
                f"Path does not exist: {path}. Available top-level directories: {available}"
            )

        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"

        client = self.minio_clients[bucket_type]
        bucket_name = self.minio_buckets[bucket_type]

        try:
            objects = client.list_objects(bucket_name, prefix=prefix, recursive=False)
            for obj in objects:
                if obj.is_dir:
                    all_items.append(f"{bucket_type}/{obj.object_name}")
                else:
                    all_items.append(f"{bucket_type}/{obj.object_name}")
        except Exception as e:
            logger.warning(f"[ObjStor] Failed to list objects in {bucket_type}: {e}")

        return all_items

    # ------------------------------------------------------------------
    # File discovery and indexing
    # ------------------------------------------------------------------

    def _get_all_text_files(self) -> List[str]:
        """Get all text files (from filesystem or MinIO)."""
        text_extensions = [".txt", ".md", ".json", ".csv", ".py", ".yaml", ".yml"]

        if self.use_minio:
            all_files = self._get_all_minio_files()
            text_files = []
            for file_path in all_files:
                if "/" in file_path:
                    _, object_name = file_path.split("/", 1)
                    ext = "." + object_name.split(".")[-1].lower() if "." in object_name else ""
                    if ext in text_extensions:
                        text_files.append(file_path)
            return text_files
        else:
            text_files = []

            def scan_directory(dir_path: Path):
                try:
                    for item in dir_path.iterdir():
                        if item.is_file():
                            if item.suffix.lower() in text_extensions:
                                relative_path = str(item.relative_to(self.files_root))
                                text_files.append(relative_path)
                        elif item.is_dir():
                            scan_directory(item)
                except PermissionError:
                    pass

            scan_directory(self.files_root)
            return text_files

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize and normalize text for BM25."""
        text_lower = text.lower()
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text_lower)

        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "this", "that", "these", "those", "i", "you", "he",
            "she", "it", "we", "they", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "not", "no", "so", "if", "as",
        }

        tokens = [word for word in words if len(word) > 1 and word not in stopwords]
        return tokens

    def _build_search_index(self):
        """Build BM25 search index from all documents."""
        if not BM25_AVAILABLE:
            return

        documents = []
        metadata = []
        text_files = self._get_all_text_files()

        logger.debug(f"Building BM25 index from {len(text_files)} files")

        for file_path in text_files:
            try:
                if self.use_minio:
                    logger.debug(f"[ObjStor] Indexing file from MinIO: {file_path}")
                    content = self._read_minio_file(file_path)
                else:
                    full_path = self.files_root / file_path
                    content = full_path.read_text(encoding="utf-8")

                if VERBOSE_MODE:
                    logger.debug(f"Indexing {file_path}: {len(content)} bytes")

                tokenized = self._tokenize_text(content)
                documents.append(tokenized)
                metadata.append({"path": file_path, "content": content})

            except (UnicodeDecodeError, PermissionError, OSError, ValueError) as e:
                logger.warning(f"Skipping {file_path}: {e}")
                continue

        if documents:
            self.bm25_index = LuceneIdfBM25(documents)
            self.indexed_documents = documents
            self.doc_metadata = metadata
            self.bm25_initialized = True
            logger.info(f"BM25 index built successfully with {len(documents)} documents")
        else:
            logger.warning("No text documents found for BM25 indexing")

    async def eager_index_builder(self, poll_interval=3.0, idle_interval=15.0,
                                  stable_after=5, settle_after=4):
        """Build the BM25 index once the corpus appears, then settle.

        Replaces the old lazy build-on-first-query path: a background task polls
        the document source and rebuilds whenever the text-file count changes, so
        the index is ready BEFORE any query arrives. The blocking build runs in a
        worker thread so it never stalls request handling, and is serialized by
        ``_index_lock``. Polls fast until the corpus first appears and stops
        changing (``stable_after`` cycles), backs off to ``idle_interval`` for a
        short settle window (``settle_after`` cycles), then STOPS re-enumerating:
        a static corpus must not be listed forever. Set ``MCP_WATCH_INGEST=true``
        to keep watching indefinitely for demos with progressive/late ingest.
        """
        if not BM25_AVAILABLE:
            logger.warning("[BM25] eager builder disabled -- rank_bm25 not installed")
            return

        watch_forever = os.getenv("MCP_WATCH_INGEST", "").lower() in ("1", "true", "yes")
        logger.info("[BM25] eager index builder started (poll=%.1fs, idle=%.1fs, "
                    "watch_forever=%s)", poll_interval, idle_interval, watch_forever)
        last_count = len(self.indexed_documents)
        stable_cycles = 0
        built = last_count > 0
        while True:
            try:
                current = len(self._get_all_text_files())
            except Exception as e:  # noqa: BLE001 - keep the watcher alive
                logger.warning("[BM25] eager builder: doc enumeration failed: %s", e)
                await asyncio.sleep(poll_interval)
                continue

            if current > 0 and current != last_count:
                logger.info("[BM25] ingest detected: text-file count %d -> %d; "
                            "rebuilding index now", last_count, current)
                async with self._index_lock:
                    await asyncio.to_thread(self._build_search_index)
                last_count = len(self.indexed_documents)
                stable_cycles = 0
                built = True
            else:
                stable_cycles += 1

            # Once the corpus has been built and has stayed stable through the
            # fast phase plus the settle window, stop watching: continuing to
            # re-list an unchanging corpus every interval is pure waste.
            if built and not watch_forever and stable_cycles >= stable_after + settle_after:
                logger.info("[BM25] corpus settled at %d documents; index build "
                            "complete, stopping watcher (set MCP_WATCH_INGEST=true "
                            "to keep watching for late ingest)", last_count)
                return

            await asyncio.sleep(poll_interval if stable_cycles < stable_after
                                else idle_interval)

    def _extract_snippets(self, content: str, query_tokens: List[str], max_snippets: int = 3) -> List[str]:
        """Extract relevant text snippets containing query terms."""
        sentences = re.split(r"[.!?]\s+", content)
        relevant_sentences = []

        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue
            sentence_tokens = self._tokenize_text(sentence)
            if any(token in sentence_tokens for token in query_tokens):
                relevant_sentences.append(sentence.strip())
                if len(relevant_sentences) >= max_snippets:
                    break

        return relevant_sentences

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def get_or_create_session(self, session_id: Optional[str] = None) -> MCPSession:
        """Get existing session or create new one."""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        new_session_id = session_id or f"session-{uuid.uuid4().hex[:8]}"
        session = MCPSession(new_session_id)
        self.sessions[new_session_id] = session
        return session

    # ------------------------------------------------------------------
    # MCP protocol handlers
    # ------------------------------------------------------------------

    async def handle_initialize(self, params: Dict[str, Any], session: MCPSession) -> Dict[str, Any]:
        session.initialized = True
        session.client_info = params.get("clientInfo", {})
        session.capabilities = params.get("capabilities", {})

        requested = params.get("protocolVersion")
        if requested in SUPPORTED_PROTOCOL_VERSIONS:
            negotiated = requested
        else:
            negotiated = LATEST_PROTOCOL_VERSION
            logger.info(
                "[MCP] client requested unsupported protocolVersion %r; answering "
                "with our latest supported version %r",
                requested, negotiated,
            )
        session.protocol_version = negotiated

        return {
            "protocolVersion": negotiated,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "direct-filesystem-mcp-server", "version": "1.0.0"},
        }

    def tool_names(self) -> set:
        """The set of advertised tool names.

        Single source of truth shared by ``tools/list`` and the ``tools/call``
        dispatcher, so an unknown tool can be rejected as a protocol error
        (``-32602``) before any handler runs.
        """
        return {tool["name"] for tool in self._tool_definitions()}

    def _tool_definitions(self) -> List[Dict[str, Any]]:
        """The four-tool catalog: one tool per store (plus ``save_file`` under agentic-tools).

        Under ``agentic-tools`` the catalog is the four tools below plus
        ``save_file``, the one write tool (see ``_save_file_definition``). The
        graphrag catalog is the four tools only.

        Ticket 03 cut the seven-tool surface to four -- ``read_file`` and
        ``list_files`` (filesystem), ``search_files_bm25`` (BM25 index) and
        ``search_semantic`` (Milvus). Dropped: ``ask_about_files`` (advertised
        "comprehensive answers" but keyword-matched hardcoded demo strings -- the
        exact inauthenticity this demo exists to remove), ``list_all_files``
        (folded into ``list_files``), ``get_search_diagnostics`` (a debug route,
        not an agent affordance) and ``search_files`` (exact substring).

        Dropping exact-substring is the consistent choice, not a convenience: the
        corpus decision differentiates tools by *what the question is about*, so
        four tools map to three stores and the agent chooses by subject. Keeping a
        near-identical second lexical tool would invite wrong-but-reasonable picks
        that read as failed qualification.

        **The descriptions ARE the selection mechanism.** These are the exact
        strings validated in prototypes/06-tool-selection: 11/11 on gpt-4o after a
        one-line ``read_file`` fix, and 6/6 on the qualification-critical
        semantic-vs-BM25 split. The design principle that made it work was stating
        each tool's NEGATIVE boundary -- selection failures come from not knowing
        when NOT to pick a tool. Do not "tidy" this wording without re-running that
        harness against the agent's model; legibility is model-specific.

        Under the ``legacy`` profile the original seven-tool catalog is served
        unchanged, so the demos that share this source keep working as before.

        Under the ``graphrag`` profile the fourth slot is ``search_graph`` instead
        of ``search_semantic``: same catalog shape, same three other tools, but
        meaning-based retrieval is answered by the governed lineage graph rather
        than a vector index.
        """
        if not FOUR_TOOL_PROFILE:
            return self._legacy_tool_definitions()
        tools = [
            {
                "name": "list_files",
                "description": (
                    "List the file paths available in the document store. Use this to "
                    "discover what documents exist -- before reading or searching, or when "
                    "the user asks what is available. Returns paths only, never content."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Optional path prefix to scope the listing (defaults to root)",
                            "default": "/",
                        }
                    },
                },
            },
            {
                "name": "read_file",
                "description": (
                    "Read the full text of ONE specific document identified by its path or "
                    "filename. Use when the user names or points to a specific document and "
                    "you want its complete contents. It takes a document path or filename, "
                    "NOT a search query -- do not use it to explore or to find documents by "
                    "topic or keyword."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Document path or filename"}
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "search_files_bm25",
                "description": (
                    "Keyword search using BM25 lexical ranking. Best for finding documents "
                    "that contain specific TERMS, names, identifiers, error codes, or exact "
                    "phrases the user named. Matches literal words -- the query and document "
                    "must share vocabulary. Use for 'which documents mention X', exact IDs, "
                    "or known terminology. It will MISS paraphrases and synonyms."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Keyword query"},
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_semantic",
                "description": (
                    "Semantic (meaning-based) search using vector embeddings. Best for "
                    "conceptual or natural-language questions where the relevant documents "
                    "may NOT use the same words as the query -- finds by topic and meaning, "
                    "not shared vocabulary. Use for 'how does X work', 'documents about the "
                    "idea of Y', or anywhere a keyword search would miss paraphrases."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural-language query"},
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 8,
                        },
                    },
                    "required": ["query"],
                },
            },
        ]
        if GRAPHRAG_PROFILE:
            tools[3] = self._search_graph_definition()
        if AGENTIC_PROFILE:
            tools.append(self._save_file_definition())
        return tools

    @staticmethod
    def _save_file_definition() -> Dict[str, Any]:
        """The `save_file` tool, advertised only under the agentic-tools profile.

        It exists so a note the agent writes is a write Caber sees: the PUT goes
        through the minio client that pycaber's urllib3 patch instruments. The
        negative boundary is stated in the style of the other four tools.
        """
        return {
            "name": "save_file",
            "description": (
                "Save a NEW markdown note of your own into the user's document store, "
                "under the user's group folder, at ai-notes/<name>. Use it when the user "
                "asks you to write something down, such as a summary or draft, for later. "
                "Not for editing or replacing an existing document (it refuses if the "
                "file already exists), and not for answering a question; answer in the "
                "reply instead."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "ai-notes/<name>, where <name> is letters, digits, dot, dash or underscore",
                    },
                    "content": {"type": "string", "description": "The markdown text to save"},
                },
                "required": ["path", "content"],
            },
        }

    @staticmethod
    def _search_graph_definition() -> Dict[str, Any]:
        """The `search_graph` tool, advertised only under the graphrag profile.

        Written to the same rule as the other three: state the NEGATIVE boundary.
        The agent must be able to tell this apart from BM25 (literal terms) and
        from read_file (one named document), and must know that what it gets back
        is already policy-scoped -- an empty result can mean "denied", not only
        "nothing relevant".
        """
        return {
            "name": "search_graph",
            "description": (
                "Governed knowledge-graph search over the document lineage graph. "
                "Best for conceptual or natural-language questions where the relevant "
                "documents may NOT use the same words as the query -- finds by topic "
                "and meaning through the graph's content regions and their links, not "
                "shared vocabulary. Every candidate is checked against the access "
                "policy for the requesting user BEFORE its content is returned, so the "
                "results are already scoped to what this user may see; documents that "
                "exist but are denied are counted, never returned. Use for 'how does X "
                "work', 'documents about the idea of Y', or anywhere a keyword search "
                "would miss paraphrases. It is NOT for exact identifiers or a named "
                "file -- use search_files_bm25 or read_file for those."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 8,
                    },
                },
                "required": ["query"],
            },
        }

    def _legacy_tool_definitions(self) -> List[Dict[str, Any]]:
        """The original seven-tool catalog, served under the ``legacy`` profile.

        Kept verbatim so agentgov-mimic (and the archived -duo, -strongdm,
        -cloudflare and streamlit-mock-mcp) -- all of which build from this same source -- see
        exactly the surface they see today. Ticket 05 requires mimic's proven CI is
        never put at risk by this work, and those demos rebuild the image from
        source, so an image tag alone would not have protected them.

        `ask_about_files` is included here and is dead, fake (it keyword-matches
        hardcoded demo strings) and unreferenced by any client, test or narration.
        It is retired from the agentic-tools catalog by ticket 03 rather than
        deleted outright, purely because deleting it would change a frozen demo's
        `tools/list` output. Delete it when those demos are retired.
        """
        return [
            {
                "name": "list_files",
                "description": "List files and directories in a specific path",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to list (relative to files root, defaults to root)",
                            "default": "/",
                        }
                    },
                },
            },
            {
                "name": "list_all_files",
                "description": "Recursively list all files in the entire filesystem, regardless of directory structure",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "include_metadata": {
                            "type": "boolean",
                            "description": "Include file metadata (size, type, etc.)",
                            "default": True,
                        }
                    },
                },
            },
            {
                "name": "read_file",
                "description": "Read the contents of a text file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to read"}
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "search_files",
                "description": "Search for text content within files",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Text to search for"},
                        "path": {
                            "type": "string",
                            "description": "Path to search within (defaults to root)",
                            "default": "/",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_files_bm25",
                "description": "Advanced search using BM25 ranking algorithm - handles natural language queries and ranks results by relevance",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "ask_about_files",
                "description": "Ask questions about files and get comprehensive answers based on their content and metadata",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Question about the files or their contents",
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional context or specific files to focus on",
                            "default": "",
                        },
                    },
                    "required": ["question"],
                },
            },
            {
                "name": "get_search_diagnostics",
                "description": "Get diagnostic information about the search index including sample searchable content from each indexed document",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "max_samples": {
                            "type": "integer",
                            "description": "Maximum number of sample sentences per document",
                            "default": 5,
                        }
                    },
                },
            },
        ]


    async def handle_tools_list(self, session: MCPSession) -> Dict[str, Any]:
        return {"tools": self._tool_definitions()}

    async def handle_tools_call(self, params: Dict[str, Any], session: MCPSession) -> Dict[str, Any]:
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        # Only tools this profile advertises can reach here: the endpoint rejects
        # anything outside `tool_names()` with a JSON-RPC -32602 first.
        handlers = {
            "list_files": self._list_files,
            "read_file": self._read_file,
            "search_files_bm25": self._search_files_bm25,
            "search_semantic": self._search_semantic,
            # graphrag-profile only; not advertised under agentic-tools or legacy.
            "search_graph": self._search_graph,
            # agentic-tools only: the one write tool.
            "save_file": self._save_file,
            # Legacy-profile only; not advertised under agentic-tools.
            "search_files": self._search_files,
            "list_all_files": self._list_all_files,
            "ask_about_files": self._ask_about_files,
            "get_search_diagnostics": self._get_search_diagnostics,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        if tool_name == "search_graph":
            # The only tool that needs the caller's identity: the retriever
            # governs per OBO user, so the forwarded principal travels with it.
            return await self._search_graph(arguments, session)
        if tool_name == "save_file":
            # Needs the acting user (the login email), to resolve their group.
            return await self._save_file(arguments, session)
        return await handler(arguments)

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _save_file(
        self, arguments: Dict[str, Any], session: Optional["MCPSession"] = None
    ) -> Dict[str, Any]:
        """Write a new note to ``<group>/ai-notes/<name>`` in the first bucket.

        The acting user is the login email (baggage ``caber.user``). Filename to
        User identity grants permits from the key's first path segment as a
        GROUP, so the key starts with the user's directory group, resolved from
        the zt-mimic seed by ``directory_lookup``. A user in no group, or in
        several, is refused: nothing is written. Refusals raise ``ValueError``,
        which the endpoint returns as a tool result with ``isError: true``.
        The log carries the key and byte count, never the content.
        """
        path = arguments.get("path")
        content = arguments.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            raise ValueError("save_file needs a path and content, both strings")
        if not self.use_minio:
            raise ValueError("save_file needs the object store")
        if not _SAVE_FILE_PATH_RE.fullmatch(path) or ".." in path:
            raise ValueError("save_file only writes new notes under ai-notes/")
        if not self.minio_clients:
            raise RuntimeError("save_file has no object store bucket configured")

        acting_user = (session.acting_user if session else None) or ""
        login_email = acting_user
        if acting_user.startswith("iss_sub|"):
            parts = acting_user.split("|", 2)
            if len(parts) == 3 and parts[2]:
                login_email = parts[2]
        directory = _load_directory_lookup()
        try:
            seed = directory.load_seed()
        except directory.DirectoryLookupError as exc:
            # An operator fault, not the user's: say so apart from "unknown user".
            logger.error("[SAVE_FILE] refused: cannot read the user directory: %s", exc)
            raise ValueError("save_file cannot read the user directory") from exc
        try:
            _tenant, groups = directory.tenant_and_groups(login_email, seed)
        except directory.DirectoryLookupError as exc:
            logger.warning("[SAVE_FILE] refused: cannot resolve groups for %r: %s", acting_user, exc)
            groups = []
        if len(groups) != 1:
            logger.warning(
                "[SAVE_FILE] refused: %r has %d directory groups, needs exactly one",
                acting_user, len(groups),
            )
            raise ValueError("save_file needs the acting user to belong to exactly one directory group")

        client_key = next(iter(self.minio_clients))
        client = self.minio_clients[client_key]
        bucket = self.minio_buckets[client_key]
        key = f"{groups[0]}/{path}"

        try:
            client.stat_object(bucket, key)
        except Exception as exc:  # noqa: BLE001 -- only "not found" means the key is free
            if getattr(exc, "code", None) not in ("NoSuchKey", "NoSuchObject", "NotFound"):
                raise RuntimeError(f"save_file could not check {key}: {exc}") from exc
        else:
            raise ValueError(f"save_file will not replace an existing file: {path}")

        data = content.encode("utf-8")
        client.put_object(
            bucket, key, io.BytesIO(data), length=len(data), content_type="text/markdown"
        )
        logger.info("[SAVE_FILE] wrote %s/%s (%d bytes)", bucket, key, len(data))
        return {
            "content": f"Saved {bucket}/{key} ({len(data)} bytes)",
            "path": key,
            "size": len(data),
        }

    async def _list_files(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        rel_path = arguments.get("path", "/").strip("/")

        if self.use_minio:
            all_items = self._list_minio_objects(rel_path)
        else:
            full_path = self.files_root / rel_path if rel_path else self.files_root
            if not full_path.exists():
                raise FileNotFoundError(f"Path does not exist: {rel_path}")
            if not full_path.is_dir():
                raise ValueError(f"Path is not a directory: {rel_path}")

            files = []
            directories = []
            try:
                for item in sorted(full_path.iterdir()):
                    relative_item = str(item.relative_to(self.files_root))
                    if item.is_file():
                        files.append(relative_item)
                    elif item.is_dir():
                        directories.append(relative_item + "/")
            except PermissionError:
                raise PermissionError(f"Permission denied accessing: {rel_path}")
            all_items = directories + files

        return {"files": all_items, "path": rel_path or "/", "count": len(all_items)}

    def _lookup_file_freshness_meta(self, file_path: str) -> Dict[str, Any]:
        """Resolve optional `drift_report` / `regions` metadata attached to a file
        via S3 object metadata (`caber-drift-b64` / `caber-regions-b64`) or an
        optional `MCP_FRESHNESS_MANIFEST` JSON file."""
        import base64

        extra: Dict[str, Any] = {}
        if self.use_minio and "/" in file_path:
            bucket_type, object_name = file_path.split("/", 1)
            client = self.minio_clients.get(bucket_type)
            bucket_name = self.minio_buckets.get(bucket_type)
            if client is not None and bucket_name and hasattr(client, "stat_object"):
                try:
                    stat = client.stat_object(bucket_name, object_name)
                    raw_meta = {str(k).lower(): str(v) for k, v in (getattr(stat, "metadata", None) or {}).items()}
                    for header_suffix, out_key in (
                        ("caber-drift-b64", "drift_report"),
                        ("caber-regions-b64", "regions"),
                    ):
                        encoded = raw_meta.get(f"x-amz-meta-{header_suffix}") or raw_meta.get(header_suffix)
                        if encoded:
                            extra[out_key] = json.loads(base64.b64decode(encoded).decode("utf-8"))
                except Exception:  # noqa: BLE001 -- optional metadata lookup must never break read_file
                    pass
        manifest_path = os.getenv("MCP_FRESHNESS_MANIFEST", "").strip()
        if not extra and manifest_path and os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as fh:
                    manifest = json.load(fh)
                entry = (manifest.get("files") or {}).get(file_path) or (manifest.get("files") or {}).get(
                    file_path.rsplit("/", 1)[-1]
                )
                if isinstance(entry, dict):
                    if isinstance(entry.get("drift_report"), dict):
                        extra["drift_report"] = entry["drift_report"]
                    if isinstance(entry.get("regions"), list):
                        extra["regions"] = entry["regions"]
            except Exception:  # noqa: BLE001
                pass
        return extra

    async def _read_file(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        file_path = arguments.get("path", "").strip("/")
        if not file_path:
            raise ValueError("File path is required")

        if self.use_minio:
            content = self._read_minio_file(file_path)
        else:
            full_path = self.files_root / file_path
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            if not full_path.is_file():
                raise ValueError(f"Path is not a file: {file_path}")
            try:
                content = full_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raise ValueError(f"File is not text readable: {file_path}")
            except PermissionError:
                raise PermissionError(f"Permission denied reading file: {file_path}")

        out: Dict[str, Any] = {"content": content, "path": file_path, "size": len(content)}
        out.update(self._lookup_file_freshness_meta(file_path))
        return out

    async def _search_files(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        query = arguments.get("query", "")
        search_path = arguments.get("path", "/").strip("/")

        if not query:
            raise ValueError("Search query is required")

        query_lower = query.lower()
        query_tokens = []

        identifiers = re.findall(r"\b[A-Z]+-[A-Z]+-\d+|\b[A-Z]-\d+\b", query)
        query_tokens.extend([id_.lower() for id_ in identifiers])

        words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", query_lower)
        stopwords = {"the", "and", "for", "are", "what", "how", "they", "compare", "other", "with", "that", "this", "from"}
        significant_words = [w for w in words if w not in stopwords and w not in query_tokens]
        query_tokens.extend(significant_words[:10])

        results = []

        if self.use_minio:
            all_files = self._get_all_minio_files()
            for fp in all_files:
                try:
                    content = self._read_minio_file(fp)
                    content_lower = content.lower()
                    matching_tokens = [token for token in query_tokens if token in content_lower]
                    if matching_tokens:
                        lines = content.split("\n")
                        matching_lines = []
                        for i, line in enumerate(lines, 1):
                            if any(token in line.lower() for token in query_tokens):
                                matching_lines.append(f"Line {i}: {line[:100]}")
                                if len(matching_lines) >= 3:
                                    break
                        results.append({
                            "path": fp,
                            "matches": matching_lines,
                            "matching_tokens": matching_tokens,
                            "preview": content[:300] + "..." if len(content) > 300 else content,
                        })
                except (ValueError, UnicodeDecodeError):
                    continue
        else:
            full_path = self.files_root / search_path if search_path else self.files_root
            if not full_path.exists():
                raise FileNotFoundError(f"Search path does not exist: {search_path}")

            def search_directory(dir_path: Path):
                try:
                    for item in dir_path.iterdir():
                        if item.is_file():
                            try:
                                content = item.read_text(encoding="utf-8")
                                content_lower = content.lower()
                                matching_tokens = [token for token in query_tokens if token in content_lower]
                                if matching_tokens:
                                    line_list = content.split("\n")
                                    matching_lines = []
                                    for i, line in enumerate(line_list, 1):
                                        if any(token in line.lower() for token in query_tokens):
                                            matching_lines.append(f"Line {i}: {line[:100]}")
                                            if len(matching_lines) >= 3:
                                                break
                                    relative_path = str(item.relative_to(self.files_root))
                                    results.append({
                                        "path": relative_path,
                                        "matches": matching_lines,
                                        "matching_tokens": matching_tokens,
                                        "preview": content[:300] + "..." if len(content) > 300 else content,
                                    })
                            except (UnicodeDecodeError, PermissionError):
                                continue
                        elif item.is_dir():
                            search_directory(item)
                except PermissionError:
                    pass

            if full_path.is_dir():
                search_directory(full_path)

        logger.info(
            "[BM25] exact-substring search (_search_files) returned %d result(s) for "
            "tokens=%s (a doc is included only if a token appears verbatim)",
            len(results), query_tokens,
        )
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "search_tokens": query_tokens,
            "search_type": "token_based",
        }

    async def _search_files_bm25(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 10)

        if not query:
            raise ValueError("Search query is required")

        logger.info(
            "[BM25] search_files_bm25 called: query=%r bm25_available=%s "
            "index_built=%s indexed_docs=%d",
            query, BM25_AVAILABLE, self.bm25_initialized, len(self.indexed_documents),
        )

        # FAIL LOUD under the agentic-tools profile (ticket 03). This used to fall
        # back to exact-substring search whenever the index was missing or the
        # scoring raised -- silently, under a tool whose whole identity is "BM25
        # lexical ranking". The demo claims deterministic, NAMED algorithms;
        # quietly answering with a different algorithm than the one advertised
        # makes that claim false, and a substituted result is indistinguishable
        # from a real one.
        #
        # The legacy profile keeps the old fallback, because the demos sharing this
        # source have passing CI built on it and ticket 05 freezes them.
        #
        # There is no lazy build in either profile: the eager background builder
        # populates the index as soon as documents are ingested, and /healthz now
        # reports `index_ready` so the harness can WAIT for a queryable index
        # rather than discovering it by failing.
        if not FOUR_TOOL_PROFILE:
            if not BM25_AVAILABLE or not self.bm25_initialized:
                logger.warning(
                    "[BM25] index NOT READY at query time (bm25_available=%s "
                    "index_built=%s) -- FALLBACK to exact-substring search "
                    "(legacy profile).", BM25_AVAILABLE, self.bm25_initialized,
                )
                return await self._search_files({"query": query})
            legacy_tokens = self._tokenize_text(query)
            if not legacy_tokens:
                logger.warning(
                    "[BM25] FALLBACK to exact-substring search: query tokenized to "
                    "nothing (query=%r) -- BM25 scoring skipped (legacy profile).", query,
                )
                return await self._search_files({"query": query})
            try:
                return await self._bm25_rank(query, legacy_tokens, max_results)
            except Exception as e:  # noqa: BLE001 - legacy behaviour, preserved
                logger.error("[BM25] scoring raised -- FALLBACK to exact-substring "
                             "search (legacy profile): %s", e)
                return await self._search_files({"query": query})

        if not BM25_AVAILABLE:
            raise RuntimeError(
                "rank_bm25 is not installed, so search_files_bm25 cannot rank. It is a "
                "hard dependency; this tool will not substitute a different search."
            )
        if not self.bm25_initialized:
            raise RuntimeError(
                "the BM25 index is not ready yet (no documents indexed). Wait for "
                "/healthz to report index_ready before querying. This tool will not "
                "fall back to substring matching."
            )

        query_tokens = self._tokenize_text(query)
        if not query_tokens:
            raise ValueError(
                f"the query {query!r} contains no indexable terms after "
                "tokenisation (all stopwords or punctuation), so BM25 has nothing "
                "to rank. Rephrase with content words."
            )

        try:
            return await self._bm25_rank(query, query_tokens, max_results)
        except Exception as e:  # noqa: BLE001
            # A scoring bug is a bug. Report it; do not answer with a different
            # algorithm and let the agent believe it got BM25 results.
            logger.error("[BM25] scoring raised: %s", e)
            raise RuntimeError(f"BM25 scoring failed: {e}") from e

    async def _bm25_rank(self, query: str, query_tokens: List[str],
                         max_results: int) -> Dict[str, Any]:
        """Score the corpus with BM25 and shape the hits.

        Extracted so both profiles run the SAME ranking code -- they differ only
        in what happens when ranking is impossible (legacy substitutes substring
        search; agentic-tools raises).
        """
        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = scores.argsort()[-max_results:][::-1]

        results = []
        for idx in top_indices:
            if idx < len(self.doc_metadata):
                metadata = self.doc_metadata[idx]
                score = float(scores[idx])

                doc_tokens = self.indexed_documents[idx] if idx < len(self.indexed_documents) else []
                matching_tokens = [token for token in query_tokens if token in doc_tokens]
                has_token_match = len(matching_tokens) > 0

                include_result = score > 0 or score > -0.5 or has_token_match

                if include_result:
                    snippets = self._extract_snippets(metadata["content"], query_tokens)
                    results.append({
                        "path": metadata["path"],
                        "score": score,
                        "snippets": snippets,
                        "preview": metadata["content"][:300] + "..." if len(metadata["content"]) > 300 else metadata["content"],
                        "matches": [f"Snippet: {snippet[:100]}..." for snippet in snippets],
                    })

        top_score = float(scores[top_indices[0]]) if len(top_indices) else 0.0
        logger.info(
            "[BM25] used BM25 ranking: query_tokens=%s scored_docs=%d "
            "returned=%d top_score=%.3f",
            query_tokens, len(self.indexed_documents), len(results), top_score,
        )
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "search_type": "bm25",
            "tokens_used": query_tokens,
        }

    async def _search_semantic(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Meaning-based retrieval over the Milvus vector store.

        Embedded rather than proxied to the ragchat service (ticket 06): a proxy
        adds a network hop and a second service whose health can fail the demo,
        and buys nothing, because provenance is captured agent-side at the
        framework tool wrapper and does not care where the query runs.

        The result shape deliberately matches ``search_files_bm25`` -- same
        ``results`` list with ``path``/``score``/``snippets``/``preview`` -- so the
        agent sees the two searches identically and chooses between them on
        *subject*, which is the distinction ticket 03 designed for and ticket 06
        measured at 6/6.

        Source paths are mandatory here, not incidental: without them the
        answer-to-document trace cannot be reconstructed, so the provenance pillar
        has nothing to render and the citation-set outcome gate has nothing to
        compare against.

        Every failure raises. Semantic search that silently returns nothing is
        indistinguishable from "no relevant documents exist", which would hollow
        out the retrieval half of the demo without any signal that it had.
        """
        query = arguments.get("query", "")
        max_results = int(arguments.get("max_results", 8))

        if not query:
            raise ValueError("Search query is required")

        try:
            from shared.milvus_query import MilvusQueryError, search
        except ImportError as exc:
            raise RuntimeError(
                "the shared Milvus query module is not importable, so semantic "
                f"search cannot run: {exc}"
            ) from exc

        logger.info("[SEMANTIC] search_semantic called: query=%r max_results=%d",
                    query, max_results)

        try:
            hits = await asyncio.to_thread(
                search, query, collection=MILVUS_COLLECTION, limit=max_results
            )
        except MilvusQueryError as exc:
            # Surface the real cause (no collection, no API key, no connection) as
            # a tool error the agent can report, rather than an empty result set.
            raise RuntimeError(str(exc)) from exc

        results = [hit.as_dict() for hit in hits]
        logger.info("[SEMANTIC] returned %d hit(s); paths=%s",
                    len(results), [r["path"] for r in results])

        return {
            "query": query,
            "results": results,
            "count": len(results),
            "search_type": "semantic",
            "collection": MILVUS_COLLECTION,
        }

    async def _search_graph(self, arguments: Dict[str, Any],
                            session: Optional[MCPSession] = None) -> Dict[str, Any]:
        """Governed retrieval over the lineage graph via the GraphRAG retriever.

        A thin client of ``POST /graphrag/scoped-search`` on
        CaberApp/graphrag-retriever. The retriever discovers a content-free
        candidate frontier in Neo4j, has the shared governance evaluator decide
        access for the OBO user this server forwards, and hydrates content only
        for the permitted survivors. Nothing here re-implements any of that: the
        server translates the request, forwards the identity, and reshapes the
        reply.

        The result shape matches ``search_files_bm25`` and ``search_semantic`` --
        ``results`` with ``path``/``score``/``snippets``/``preview`` -- so the agent
        sees the three searches identically and chooses on *subject*. ``path`` is
        what the retriever derives from the Object node and is what ``read_file``
        and the provenance pillar key on; an item without one is reported rather
        than dropped, since a missing path is exactly the kind of silent
        degradation that hollows out the citation gate.

        Every failure raises. A retriever that is down, a scoped-search that
        returns a non-2xx, or a malformed body are all tool errors the agent can
        see and route around, never an empty result set.
        """
        query = arguments.get("query", "")
        max_results = int(arguments.get("max_results", 8))

        if not query:
            raise ValueError("Search query is required")

        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - httpx is a hard dependency
            raise RuntimeError(f"httpx is required for search_graph: {exc}") from exc

        obo_user = (session.acting_user if session else None) or MCP_SERVER_IDENTITY
        acting_email = (session.acting_email if session else "") or ""
        payload = {
            "query_terms": query,
            "max_results": max(1, min(max_results, 500)),
            "obo_user": obo_user,
            "acting_email": acting_email or None,
            "api_id": "api-graphrag",
            "agent_identity": "mcp-server-back-door",
            "action": "retrieve",
        }
        logger.info("[GRAPH] search_graph called: query=%r max_results=%d obo_user=%s email=%s",
                    query, max_results, obo_user, "yes" if acting_email else "no")

        def _post():
            # trust_env=False: the system proxy must never see an in-cluster hop.
            # (Not proxy=None: this image pins httpx 0.25, which has no `proxy`
            # kwarg, and on newer httpx proxy=None still reads the env proxies.)
            with httpx.Client(trust_env=False, timeout=GRAPHRAG_TIMEOUT) as client:
                return client.post(f"{GRAPHRAG_URL}/graphrag/scoped-search", json=payload)

        try:
            response = await asyncio.to_thread(_post)
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"graphrag retriever at {GRAPHRAG_URL} is unreachable, so graph "
                f"search cannot run: {exc}"
            ) from exc

        if response.status_code != 200:
            detail = response.text[:300]
            raise RuntimeError(
                f"graphrag retriever returned HTTP {response.status_code} for "
                f"scoped-search: {detail}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"graphrag retriever returned a non-JSON scoped-search body: {exc}"
            ) from exc

        items = body.get("items") or body.get("results") or []
        results = []
        missing_path = []
        primary_drift_report = None
        primary_superseded_from = None
        merged_regions: List[Dict[str, Any]] = []
        for item in items:
            path = (item.get("path") or "").strip()
            if not path:
                missing_path.append(item.get("object_id") or item.get("retrieval_item_id"))
            content = item.get("content") or ""
            item_meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            snippets = item.get("snippets")
            if not isinstance(snippets, list):
                snippets = [content[:300]] if content else []
            preview = item.get("preview") if isinstance(item.get("preview"), str) else content[:200]

            chosen_member_id = (
                item.get("chosen_member_id")
                if item.get("chosen_member_id") is not None
                else item_meta.get("chosen_member_id")
            )
            superseded_from = (
                item.get("superseded_from")
                if item.get("superseded_from") is not None
                else item_meta.get("superseded_from")
            )
            drift_report = (
                item.get("drift_report")
                if item.get("drift_report") is not None
                else item_meta.get("drift_report")
            )
            ordering_basis = (
                item.get("ordering_basis")
                or item_meta.get("ordering_basis")
                or (drift_report.get("stale_basis") if isinstance(drift_report, dict) else None)
            )
            resolution_basis = (
                item.get("resolution_basis")
                or item_meta.get("resolution_basis")
                or ordering_basis
            )
            regions = (
                item.get("regions")
                if item.get("regions") is not None
                else item_meta.get("regions")
            )

            entry_meta: Dict[str, Any] = {
                "retrieval_item_id": item.get("retrieval_item_id") or item_meta.get("retrieval_item_id"),
                "edge_id": item.get("edge_id") or item_meta.get("edge_id"),
                "region_id": item.get("region_id") or item_meta.get("region_id"),
                "object_id": item.get("object_id") or item_meta.get("object_id"),
                "object_name": item.get("object_name") or item_meta.get("object_name"),
                "chosen_member_id": chosen_member_id,
                "superseded_from": superseded_from,
                "ordering_basis": ordering_basis,
                "resolution_basis": resolution_basis,
                "drift_report": drift_report,
            }
            if isinstance(regions, list):
                entry_meta["regions"] = regions
                merged_regions.extend(r for r in regions if isinstance(r, dict))

            entry: Dict[str, Any] = {
                "path": path,
                "score": float(item.get("score") or 0.0),
                "snippets": snippets,
                "preview": preview,
                "chosen_member_id": chosen_member_id,
                "superseded_from": superseded_from,
                "ordering_basis": ordering_basis,
                "resolution_basis": resolution_basis,
                "drift_report": drift_report,
                "metadata": entry_meta,
            }
            if isinstance(regions, list):
                entry["regions"] = regions
            results.append(entry)

            if superseded_from and primary_superseded_from is None:
                primary_superseded_from = superseded_from
            if isinstance(drift_report, dict):
                if primary_drift_report is None or (
                    drift_report.get("is_drifted") and not primary_drift_report.get("is_drifted")
                ):
                    primary_drift_report = drift_report

        if missing_path:
            raise RuntimeError(
                "graphrag retriever returned items without a document path "
                f"({len(missing_path)} of {len(items)}: {missing_path[:5]}); the "
                "provenance trace cannot be reconstructed from them"
            )

        denied = body.get("denied_item_ids") or []
        logger.info("[GRAPH] returned %d permitted hit(s), %d denied; paths=%s",
                    len(results), len(denied), [r["path"] for r in results])

        reply: Dict[str, Any] = {
            "query": query,
            "results": results,
            "count": len(results),
            "search_type": "graph",
            "denied_count": len(denied),
            "clarifications": body.get("clarifications") or [],
            "audit_refs": body.get("audit_refs") or [],
            "obo_user": obo_user,
        }
        if primary_superseded_from is not None:
            reply["superseded_from"] = primary_superseded_from
        if primary_drift_report is not None:
            reply["drift_report"] = primary_drift_report
        if merged_regions:
            reply["regions"] = merged_regions
        return reply

    async def _list_all_files(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        include_metadata = arguments.get("include_metadata", True)
        all_files = []
        directories = []

        if self.use_minio:
            # _init_minio_clients creates a client only for each enabled bucket
            # (MCP_MINIO_BUCKETS, or the legacy READ_MINIO_CONTRACTS/CUSTOMERS
            # slots), so the client map is the set of buckets to list.
            for bucket_type in self.minio_clients:
                directories.append({"path": f"{bucket_type}/", "name": bucket_type, "type": "directory"})
                client = self.minio_clients[bucket_type]
                bucket_name = self.minio_buckets[bucket_type]
                try:
                    objects = client.list_objects(bucket_name, recursive=True)
                    for obj in objects:
                        file_path = f"{bucket_type}/{obj.object_name}"
                        file_info = {
                            "path": file_path,
                            "name": obj.object_name.split("/")[-1],
                            "type": "file",
                        }
                        if include_metadata:
                            file_info.update({
                                "size": obj.size,
                                "extension": "." + obj.object_name.split(".")[-1].lower() if "." in obj.object_name else "",
                                "directory": f"{bucket_type}/" + "/".join(obj.object_name.split("/")[:-1]) if "/" in obj.object_name else f"{bucket_type}/",
                                "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                                "bucket_type": bucket_type,
                            })
                        all_files.append(file_info)
                except Exception as e:
                    logger.warning(f"[ObjStor] Failed to list objects from {bucket_type} bucket: {e}")
        else:
            def scan_directory(dir_path: Path):
                try:
                    for item in sorted(dir_path.iterdir()):
                        relative_path = str(item.relative_to(self.files_root))
                        if item.is_file():
                            file_info = {"path": relative_path, "name": item.name, "type": "file"}
                            if include_metadata:
                                try:
                                    stat = item.stat()
                                    file_info.update({
                                        "size": stat.st_size,
                                        "extension": item.suffix.lower(),
                                        "directory": str(item.parent.relative_to(self.files_root)) if item.parent != self.files_root else "/",
                                    })
                                except (OSError, PermissionError):
                                    pass
                            all_files.append(file_info)
                        elif item.is_dir():
                            directories.append({"path": relative_path + "/", "name": item.name, "type": "directory"})
                            scan_directory(item)
                except PermissionError:
                    pass

            scan_directory(self.files_root)

        all_items = directories + all_files
        return {
            "items": all_items,
            "files": all_files,
            "directories": directories,
            "total_count": len(all_items),
            "file_count": len(all_files),
            "directory_count": len(directories),
            "include_metadata": include_metadata,
        }

    async def _ask_about_files(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        question = arguments.get("question", "")
        context = arguments.get("context", "")

        if not question:
            raise ValueError("Question is required")

        all_files_result = await self._list_all_files({"include_metadata": True})
        all_files = all_files_result["files"]

        response_parts = []
        response_parts.append(f"**File Inventory ({len(all_files)} files found):**")
        for file_info in all_files:
            path = file_info["path"]
            size = file_info.get("size", "unknown size")
            ext = file_info.get("extension", "no extension")
            directory = file_info.get("directory", "/")
            response_parts.append(f"- `{path}` ({size} bytes, {ext} file in {directory})")

        response_parts.append("")

        question_lower = question.lower()
        relevant_files = []
        for file_info in all_files:
            file_path = file_info["path"]
            file_name = file_info["name"].lower()
            if (file_name in question_lower or file_path.lower() in question_lower or
                    any(keyword in file_path.lower() for keyword in ["contract", "customer", "7742", "c-8821"])):
                relevant_files.append(file_info)

        if relevant_files:
            response_parts.append("**Relevant Files Based on Your Question:**")
            for file_info in relevant_files:
                path = file_info["path"]
                response_parts.append(f"\n**{path}:**")
                try:
                    content_result = await self._read_file({"path": path})
                    content = content_result["content"]
                    preview = content[:500] + "..." if len(content) > 500 else content
                    response_parts.append(f"```\n{preview}\n```")
                except Exception as e:
                    response_parts.append(f"Could not read file: {str(e)}")

        if "how many" in question_lower:
            response_parts.append(f"\n**Count Information:**")
            response_parts.append(f"- Total files: {len(all_files)}")
            response_parts.append(f"- Contract files: {len([f for f in all_files if 'contract' in f['path'].lower()])}")
            response_parts.append(f"- Customer files: {len([f for f in all_files if 'customer' in f['path'].lower()])}")

        answer = "\n".join(response_parts)
        return {
            "question": question,
            "context": context,
            "answer": answer,
            "file_count": len(all_files),
            "relevant_files": [f["path"] for f in relevant_files],
        }

    async def _get_search_diagnostics(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        max_samples = arguments.get("max_samples", 5)
        diagnostics = {
            "bm25_available": BM25_AVAILABLE,
            "bm25_initialized": self.bm25_initialized,
            "total_documents": len(self.indexed_documents) if self.bm25_initialized else 0,
            "documents": [],
        }
        if self.bm25_initialized:
            for i, metadata in enumerate(self.doc_metadata):
                doc_info = {
                    "index": i,
                    "path": metadata["path"],
                    "content_length": len(metadata["content"]),
                    "token_count": len(self.indexed_documents[i]) if i < len(self.indexed_documents) else 0,
                    "sample_tokens": self.indexed_documents[i][:20] if i < len(self.indexed_documents) else [],
                    "sample_sentences": [],
                }
                content = metadata["content"]
                sentences = re.split(r"[.!?]\s+", content)
                sample_count = 0
                for sentence in sentences:
                    if sample_count >= max_samples:
                        break
                    if len(sentence.strip()) > 30:
                        doc_info["sample_sentences"].append(sentence.strip()[:200])
                        sample_count += 1
                diagnostics["documents"].append(doc_info)
        else:
            diagnostics["message"] = "BM25 index not initialized"
        return diagnostics


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

mcp_server: Optional[DirectMCPServer] = None

health_check_count = 0


def _as_call_tool_result(result: Any, is_error: bool = False) -> Dict[str, Any]:
    """Wrap a tool result in a spec-valid MCP ``CallToolResult`` envelope.

    The MCP tool-result model is::

        content: list[ContentBlock]          # REQUIRED, must be a list
        structuredContent: dict | None
        isError: bool = False

    The previous version of this helper only added ``content`` when the key was
    absent, so ``read_file`` -- whose domain dict already carries a *string*
    ``content`` -- went out with ``content`` as a str and failed client-side
    pydantic validation on every call (``content Input should be a valid list``).
    That was Tier-A blocker A2 in the ticket-02 audit: *every* tool call failed
    against a real MCP client. ``content`` is now always a list.

    ``Result`` sets ``extra="allow"``, so the legacy top-level domain fields
    (``files``, ``results``, ``count``, ``path``, ...) are preserved alongside the
    spec fields and existing raw-httpx consumers keep working unchanged. The one
    unavoidable casualty is a domain ``content`` *string*: it cannot sit at the
    top level and be a list at the same time. Where a tool returns one (only
    ``read_file`` does) it becomes the text block and is NOT duplicated into
    ``structuredContent`` -- deliberately, because a second copy of the same
    document text is a second thing every redaction pass has to find.

    Any ``_meta`` already present (e.g. a projected governance decision) is
    carried through untouched.
    """
    if not isinstance(result, dict):
        text = str(result)
        wrapped = {"result": result}
        return {
            "content": [{"type": "text", "text": text}],
            "structuredContent": wrapped,
            "structured_content": wrapped,
            "isError": is_error,
        }

    # Copy-on-write throughout: never mutate the caller's dict.
    domain = dict(result)
    meta = domain.pop("_meta", None)
    domain.pop("isError", None)
    domain.pop("structuredContent", None)
    domain.pop("structured_content", None)

    raw_content = domain.pop("content", None)
    if isinstance(raw_content, str):
        # A document-text tool (read_file). The text IS the content block; the
        # remaining keys (path, size) are the structured view.
        text = raw_content
        structured = domain
    else:
        # A structured tool (list_files, search_*). Mirror the whole domain dict
        # into the text block so a text-only client still sees everything.
        structured = domain
        try:
            text = json.dumps(structured, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(structured)

    envelope = dict(domain)  # legacy top-level fields, minus any content string
    envelope["content"] = [{"type": "text", "text": text}]
    envelope["structuredContent"] = structured
    envelope["structured_content"] = structured
    envelope["isError"] = is_error
    if meta is not None:
        envelope["_meta"] = meta
    return envelope


def _jsonrpc_result(request_id: Any, result: Any) -> JSONResponse:
    """A JSON-RPC success envelope at HTTP 200."""
    return JSONResponse(content={"jsonrpc": "2.0", "id": request_id, "result": result})


def _jsonrpc_error(request_id: Any, code: int, message: str) -> JSONResponse:
    """A JSON-RPC error envelope at **HTTP 200**.

    Tier-A blocker A3: the server used to answer JSON-RPC-level errors with HTTP
    400/403/404/500. The official client special-cases only 202 and 404 and calls
    ``raise_for_status()`` on everything else, which kills ``post_writer`` and
    tears down both streams -- so a single missing file ended the session instead
    of returning a recoverable error. Transport status and protocol status are
    different things; protocol errors ride inside a 200.
    """
    return JSONResponse(
        content={"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
    )


def _tool_error_result(request_id: Any, message: str) -> JSONResponse:
    """A failed *tool execution*, per the spec's tool error-handling rules.

    Tool-level failures (file not found, bad path, permission denied) are NOT
    JSON-RPC errors -- the spec puts them in the result with ``isError: true`` so
    the model can see what went wrong and recover. JSON-RPC errors are reserved
    for protocol-level faults (unknown method, unknown tool, malformed request).
    """
    return _jsonrpc_result(
        request_id,
        _as_call_tool_result({"error": message}, is_error=True),
    )


class HealthCheckLogFilter(logging.Filter):
    """Filter to suppress uvicorn access logs for health check requests."""

    def filter(self, record):
        if VERBOSE_MODE:
            return True
        if hasattr(record, "args") and record.args:
            message = record.getMessage()
            if 'GET /healthz HTTP/1.1" 200' in message:
                return False
        return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server lifecycle."""
    global mcp_server
    files_root = os.getenv("MCP_FILES_PATH", "/mcp-files")
    mcp_server = DirectMCPServer(files_root)
    logger.info("MCP Server started with files root: %s", files_root)
    _register_embedding_usage_recorder()

    # Eagerly (re)build the BM25 index in the background as soon as docs are
    # ingested -- no lazy build-on-first-query.
    builder_task = asyncio.create_task(mcp_server.eager_index_builder())

    try:
        yield
    finally:
        logger.info("MCP Server shutting down")
        builder_task.cancel()
        try:
            await builder_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Direct MCP Filesystem Server",
    description="Model Context Protocol server with direct filesystem access and PyCaber lineage",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Demo-choreography side routes (NOT MCP)
#
# `PUT /redaction/{enabled}` and `GET /redaction` are not part of the MCP spec --
# a real third-party MCP server has no such endpoint. Ticket 12 called for
# removing them so this server is a faithful stand-in, with the on/off contrast
# relocating client-side (ticket 10 landed it on the `CSI_GLOBAL_REDACT_ENABLED` flag).
#
# They are NOT deleted, because they are load-bearing for agentgov-mimic (and the
# archived -duo, -strongdm, -cloudflare and streamlit-mock-mcp demos), which
# share this source and drive them from narration `shell_command` segments and
# `runner/demo_profiles/*.py`, and `tests/e2e/test_agentgov_mimic_e2e.py` asserts
# on `GET /redaction`. Deleting them would break the very CI ticket 05 exists to
# protect ("mimic's proven CI is never at risk").
#
# So they are opt-OUT instead: on by default (every existing demo unchanged),
# and disabled for agentic-tools, which has no back door and no server-side
# middleware to toggle.
# ---------------------------------------------------------------------------
ENABLE_REDACTION_ROUTES = os.getenv("MCP_ENABLE_REDACTION_ROUTES", "true").lower() != "false"


@app.put("/redaction/{enabled}")
async def set_redaction(enabled: str):
    """Toggle CSI_GLOBAL_REDACT_ENABLED at runtime.  PUT /redaction/true or /redaction/false."""
    if not ENABLE_REDACTION_ROUTES:
        raise HTTPException(status_code=404, detail="Not Found")
    value = enabled.lower() in ("true", "1", "yes")
    if _has_pycaber:
        pycaber.auto.set_redact_enabled(value)
    logger.info("CSI_GLOBAL_REDACT_ENABLED set to %s via API", value)
    return {"CSI_GLOBAL_REDACT_ENABLED": value}


def _live_redact_enabled():
    """Read the redaction kill-switch as the OBO processor actually sees it.

    Prefers PyCaber's in-process flag (set via set_redact_enabled / the PUT
    endpoint above); falls back to the CSI_GLOBAL_REDACT_ENABLED env var when
    PyCaber is not loaded. Returns None if the state cannot be read.
    """
    if _has_pycaber:
        return bool(pycaber.is_redact_enabled())
    env = os.environ.get("CSI_GLOBAL_REDACT_ENABLED")
    if env is None:
        return None
    return env.lower() != "false"


@app.get("/redaction")
async def get_redaction():
    """Report the live redaction kill-switch state for test/diagnostic use."""
    if not ENABLE_REDACTION_ROUTES:
        raise HTTPException(status_code=404, detail="Not Found")
    return {
        "CSI_GLOBAL_REDACT_ENABLED": _live_redact_enabled(),
        "env_CSI_GLOBAL_REDACT_ENABLED": os.environ.get("CSI_GLOBAL_REDACT_ENABLED"),
        "pycaber_loaded": _has_pycaber,
    }


@app.get("/healthz")
async def health_check():
    """Liveness plus BM25 index readiness.

    ``index_ready`` is the signal ticket 03 asked for. Now that
    ``search_files_bm25`` fails loud instead of silently substituting substring
    search, a query that arrives before the eager builder has settled returns an
    error -- so the demo harness and CI need a way to WAIT for a queryable index
    rather than discovering it by failing. The eager builder already polls until
    the corpus settles; this just exposes what it knows.

    ``status`` stays "healthy" while the index is still building: the process is
    up and serving, and gating container health on corpus ingestion would deadlock
    a compose graph where the seed step runs after the service is healthy.
    """
    global health_check_count
    health_check_count += 1
    if health_check_count == 10 and not VERBOSE_MODE:
        logger.info("Health check uvicorn access logging will now be suppressed (enable MCP_VERBOSE=true to see all)")
        uvicorn_access_logger = logging.getLogger("uvicorn.access")
        if not any(isinstance(f, HealthCheckLogFilter) for f in uvicorn_access_logger.filters):
            uvicorn_access_logger.addFilter(HealthCheckLogFilter())

    server = mcp_server
    return {
        "status": "healthy",
        "service": "direct-mcp-filesystem-server",
        "bm25_available": BM25_AVAILABLE,
        "index_ready": bool(server.bm25_initialized) if server else False,
        "indexed_documents": len(server.indexed_documents) if server else 0,
    }


@app.get("/")
async def service_info():
    return {
        "server": "direct-mcp-filesystem-server",
        "version": "1.0.0",
        "protocol": "JSON-RPC 2.0",
        "transport": "HTTP",
        "endpoints": {
            "mcp": "POST /mcp - Main JSON-RPC endpoint",
            "health": "GET /healthz - Health check",
        },
    }


_mcp_inflight = 0
_mcp_inflight_lock = asyncio.Lock()

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """Main MCP JSON-RPC 2.0 endpoint. PyCaber interception handled by pycaber.auto."""
    global _mcp_inflight
    start_time = time.time()
    async with _mcp_inflight_lock:
        _mcp_inflight += 1
        inflight = _mcp_inflight

    try:
        body = await request.json()
    except Exception:
        return _jsonrpc_error(None, JSONRPC_PARSE_ERROR, "Parse error: request body is not valid JSON")

    if not isinstance(body, dict) or body.get("jsonrpc") != "2.0":
        # A JSON array here is a JSON-RPC batch, removed in revision 2025-06-18 --
        # rejecting it is correct.
        return _jsonrpc_error(
            body.get("id") if isinstance(body, dict) else None,
            JSONRPC_INVALID_REQUEST,
            "Invalid Request: not a JSON-RPC 2.0 object",
        )

    # A JSON-RPC *notification* is defined by the ABSENCE of the `id` member --
    # not by a falsy one. `body.get("id")` cannot tell `{"id": 0}` (a legal
    # request id) from a notification, so membership is what we test.
    is_notification = "id" not in body
    request_id = body.get("id")
    method = body.get("method")
    params = body.get("params") or {}

    # Tier-A blocker A1: the mandatory `notifications/initialized` that every
    # client sends immediately after `initialize` used to get HTTP 400 "Unknown
    # method", which killed the transport before `tools/list` was ever reached.
    # The transport spec: "If the input is a JSON-RPC response or notification:
    # if the server accepts the input, the server MUST return HTTP status code
    # 202 Accepted with no body."
    if is_notification:
        logger.debug("[MCP] notification accepted: %s", method)
        return Response(status_code=202)

    session_id = request.headers.get("Mcp-Session-Id") or (str(request_id) if request_id else None)

    tool_name = (params.get("name", "unknown") if params else "unknown") if method == "tools/call" else method
    acting_user = baggage_user(request.headers.get("baggage", "")) or MCP_SERVER_IDENTITY
    _perf_label = f"{acting_user} {method}:{tool_name}"
    logger.info(f"[PERF] >> {_perf_label} inflight={inflight}")

    try:
        if method == "initialize":
            session = mcp_server.get_or_create_session()
            result = await mcp_server.handle_initialize(params, session)

            response = {"jsonrpc": "2.0", "id": request_id, "result": result}
            return JSONResponse(content=response, headers={"Mcp-Session-Id": session.session_id})

        else:
            if not session_id:
                session = mcp_server.get_or_create_session()
            else:
                session = mcp_server.get_or_create_session(session_id)

            if not session.initialized:
                session.initialized = True
                session.client_info = {"name": "auto-initialized", "version": "1.0.0"}
                session.capabilities = {}

            # Remember who the agent says is acting, for tools that govern per
            # principal (search_graph). Re-read on every request: one session can
            # serve several users over its lifetime.
            session.acting_user = acting_user
            session.acting_email = baggage_email(request.headers.get("baggage", ""))

            event_path = method.lower()
            if method == "tools/list":
                result = await mcp_server.handle_tools_list(session)

            elif method == "tools/call":
                tool_name = params.get("name", "unknown") if params else "unknown"
                event_path = f"tools/call/{tool_name}".lower()
                if tool_name not in mcp_server.tool_names():
                    # Unknown *tool* is a protocol fault, not a tool failure.
                    return _jsonrpc_error(
                        request_id, JSONRPC_INVALID_PARAMS, f"Unknown tool: {tool_name}"
                    )
                try:
                    result = await mcp_server.handle_tools_call(params, session)
                except RuntimeError as e:
                    # A tool that ran and failed. Spec: report in the tool result
                    # with isError: true, so the model can recover, rather than
                    # as a transport-level error that ends the session.
                    #
                    # RuntimeError is included deliberately -- it is what the
                    # fail-loud tools raise (BM25 index not ready, Milvus
                    # unreachable, no embeddings key). Those are tool *execution*
                    # failures, and surfacing them to the model is what lets the
                    # agent visibly choose a different tool instead of the server
                    # silently substituting one for it.
                    logger.error("[MCP] tool %s failed: %s", tool_name, e)
                    return _tool_error_result(request_id, str(e))
                except (FileNotFoundError, ValueError, PermissionError) as e:
                    logger.warning("[MCP] tool %s failed: %s", tool_name, e)
                    return _tool_error_result(request_id, str(e))
                # Present tool output as an MCP CallToolResult so the governance
                # layer (PyCaber interception on this governed host) has a
                # `content` array and `_meta` slot to project a decision onto
                # (clarification / block). Existing structured fields (e.g.
                # "results") are preserved for backward-compatible clients.
                result = _as_call_tool_result(result)

            elif method == "ping":
                # Spec (Utilities -> Ping): "The receiver MUST respond promptly
                # with an empty response." Clients that ping for liveness
                # (Claude Desktop, VS Code) mark a non-responder dead.
                result = {}

            else:
                return _jsonrpc_error(
                    request_id, JSONRPC_METHOD_NOT_FOUND, f"Method not found: {method}"
                )

            response = {"jsonrpc": "2.0", "id": request_id, "result": result}
            return JSONResponse(content=response)

    except HTTPException:
        raise

    except Exception as e:
        error_details = f"Internal error: {str(e)}"
        if VERBOSE_MODE:
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
        return _jsonrpc_error(request_id, JSONRPC_INTERNAL_ERROR, error_details)

    finally:
        async with _mcp_inflight_lock:
            _mcp_inflight -= 1
        total = time.time() - start_time
        logger.info(f"[PERF] << {_perf_label} total={total:.2f}s inflight={_mcp_inflight}")


@app.get("/mcp")
async def mcp_get_stream():
    """No server-to-client stream: answer the standalone GET with 405.

    Spec (Transports, Listening for Messages from the Server): "The server MUST
    either return `Content-Type: text/event-stream` in response to this HTTP GET,
    or else return HTTP 405 Method Not Allowed". This route used to return a
    `200 application/json` service banner, which made the official client's SSE
    reader raise (`Expected response header Content-Type to contain
    'text/event-stream'`), retry twice, and give up -- log spam for a channel
    that does not exist.

    405 is the honest answer: this demo has no server-initiated messages (no
    `notifications/tools/list_changed`, no progress, no sampling round-trips), so
    there is nothing to stream. `GET /` still serves the banner. Recorded as a
    deliberate deferral in ticket 12, not an oversight.
    """
    return Response(status_code=405, headers={"Allow": "POST"})


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    uvicorn_log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"},
            "access": {"format": "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"},
        },
        "handlers": {
            "default": {"formatter": "default", "class": "logging.StreamHandler", "stream": "ext://sys.stdout"},
            "access": {"formatter": "access", "class": "logging.StreamHandler", "stream": "ext://sys.stdout"},
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO"},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        },
    }

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("MCP_HTTP_PORT", "3010")),
        log_level="info",
        log_config=uvicorn_log_config,
    )
