"""
AgentGov Demo -- Agentic Chat Application

Streamlit chat app that:
1. Authenticates users (Amy/Bob/Admin) with tenant context
2. Calls GraphRAG retriever for authorization-scoped document list
3. Calls MCP server (with tenant header) for BM25 search filtered to authorized docs
4. Sends context to OpenAI LLM for answer generation
5. PyCaber auto-instrumentation transparently intercepts all HTTP/WS traffic
"""

import os
import json
import time
import logging
import random
from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# Logging -- must be configured BEFORE importing PyCaber so that the
# pyo3-log bridge (established at import time) finds Python handlers.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
)

# PyCaber per-module logging (standard Python loggers bridged from Rust)
logging.getLogger("pycaber").setLevel(logging.INFO)
logging.getLogger("pycaber.auto").setLevel(logging.INFO)
# logging.getLogger("pycaber.rbix_workspace").setLevel(logging.DEBUG)
logging.getLogger("pycaber.api_discovery").setLevel(logging.DEBUG)
# logging.getLogger("pycaber.pipeline").setLevel(logging.DEBUG)
# logging.getLogger("pycaber.protobuf").setLevel(logging.DEBUG)
# logging.getLogger("pycaber.interaction").setLevel(logging.DEBUG)
# logging.getLogger("pycaber.config").setLevel(logging.DEBUG)
# logging.getLogger("pycaber.caber_apis").setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# PyCaber auto-instrumentation (must precede httpx/openai imports)
# DREAM is disabled here (set in compose) -- the MCP server handles DREAM
# lookups and redaction.  The chat app only does OBO tracking and
# fingerprinting (RBIX).  CSI_GLOBAL_REDACT_ENABLED=false keeps redaction off.
# ---------------------------------------------------------------------------
os.environ.setdefault("CSI_CABER_HOST", "http://caber-core:5556")
os.environ.setdefault("CSI_API_KEY", "abc123")
os.environ.setdefault("CSI_GLOBAL_REDACT_ENABLED", "false")

import sys
if "pycaber.auto" not in sys.modules:
    try:
        import pycaber.auto  # noqa: F401 -- bootstrap + patches httpx + tornado WS
        logging.getLogger(__name__).info("pycaber.auto loaded -- transparent interception active")
    except ImportError:
        pass
import pycaber

import yaml
from yaml.loader import SafeLoader
import streamlit as st
import streamlit_authenticator as stauth
import httpx
from openai import OpenAI
from dotenv import load_dotenv

from ask_llm import get_llm_answer
from encoded_input import encoded_text_area
import governance
import purpose_client
# Imported AFTER pycaber.auto above. agent_loop itself imports langchain lazily
# inside its functions, so that ordering is preserved even if this line moves:
# if langchain_core were imported before pycaber.auto, BaseTool would never be
# patched and every tool result would reach the model unredacted, silently.
import agent_loop
import directory_lookup

logger = logging.getLogger(__name__)

# The issuer this app declares for the users it authenticates. Caber keys a
# user on issuer + subject (iss_sub), so this string must match the
# `agentic_chat_app` Identity_Sources entry (type issuer) in the demo configs.
CHAT_APP_ISSUER = "agentic-chat-app"
CHAT_APP_AUTHENTICATOR = "streamlit-authenticator"
CHAT_APP_ACCEPTOR = "agentic-chat-app"

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://mcp-server:3010")
GRAPHRAG_URL = os.getenv("GRAPHRAG_URL", "http://graphrag-retriever:8084")
OPENAI_LLM_MODEL = os.getenv("USE_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
OPENAI_BASE_URL = (os.getenv("OPENAI_BASE_URL") or "").strip() or None

# ---------------------------------------------------------------------------
# Agent mode (agentic-tools demo, ticket 09)
#
# "langgraph" replaces the legacy retrieval path -- a single hardcoded
# `search_files_bm25` call on the raw question -- with a real LangGraph ReAct
# agent that chooses among the MCP server's tools and formulates its own queries.
#
# Off by default so agentgov-mimic (which shares this image source) keeps its
# proven behaviour; the agentic-tools overlay sets AGENT_MODE=langgraph.
# ---------------------------------------------------------------------------
AGENT_MODE = os.getenv("AGENT_MODE", "legacy").strip().lower()
AGENT_ENABLED = AGENT_MODE == "langgraph"

# Purpose classification (GLiNER2). The client deployment boundary is trusted for
# classification in this MVP; there is NO keyword fallback -- if the classifier is
# unreachable the failure is surfaced, never faked into a guessed purpose.
# No built-in URL: a deployment opts in by setting PURPOSE_CLASSIFIER_URL (the
# agentic-tools --with-ner overlay, agentgov-mimic, agentgov-auth). Unset or
# empty means no classifier is deployed and prompts go unclassified.
PURPOSE_CLASSIFIER_URL = os.getenv("PURPOSE_CLASSIFIER_URL", "").strip()
# Set after the first "no classifier configured" log line (logged once).
_purpose_unset_logged = False
PURPOSE_MODEL_ID = os.getenv("PURPOSE_MODEL_ID", "fastino/gliner2-base-v1")
PURPOSE_TAXONOMY_REVISION = os.getenv("PURPOSE_TAXONOMY_REVISION", "caber-purpose-v1")
# The gliner2 model server authenticates every /gliner-2 call with an X-API-Key.
# Reuse the same key the ingest path uses (CSI_GLINER2_MODEL_API_KEY); the demo
# launcher (demo.py) starts the server with this key.
PURPOSE_CLASSIFIER_API_KEY = (
    os.getenv("PURPOSE_CLASSIFIER_API_KEY")
    or os.getenv("CSI_GLINER2_MODEL_API_KEY")
    or "dev-local-key"
)


def classify_purpose_context(prompt: str):
    """Classify the prompt's purpose via GLiNER2 and return its evidence context.

    Returns the ``PurposeContext`` on success. On classifier unavailability the
    failure is logged and ``None`` is returned so the caller can proceed with an
    unclassified request -- never a fabricated purpose. The taxonomy and
    enforcement are NOT reimplemented here; the returned label/evidence are
    attached as-is for the server-side governor to consume.

    With no ``PURPOSE_CLASSIFIER_URL`` configured it returns ``None`` without
    calling anything, and logs that once.
    """
    global _purpose_unset_logged
    if not PURPOSE_CLASSIFIER_URL:
        if not _purpose_unset_logged:
            _purpose_unset_logged = True
            logger.info(
                "PURPOSE_CLASSIFIER_URL is not set: no purpose classifier is deployed, "
                "so prompts are sent unclassified."
            )
        return None
    try:
        return purpose_client.classify_purpose(
            prompt,
            url=PURPOSE_CLASSIFIER_URL,
            model_id=PURPOSE_MODEL_ID,
            taxonomy_revision=PURPOSE_TAXONOMY_REVISION,
            api_key=PURPOSE_CLASSIFIER_API_KEY,
        )
    except purpose_client.PurposeUnavailable as exc:
        logger.warning("Purpose classifier unavailable (no fallback): %s", exc)
        return None

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AgentGov Demo", layout="wide")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for key, default in {
    "mcp_session_id": None,
    "mcp_request_id": 0,
    "mcp_tenant_id": None,
    "_rerun_count": 0,
    "_ws_pin_id": "",
    "gov_purpose_context": None,
    "_clar_session": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.session_state._rerun_count += 1
_rerun = st.session_state._rerun_count
_auth = st.session_state.get("authentication_status", None)
logger.info("[RERUN #%d] auth_status=%s user=%s", _rerun, _auth, st.session_state.get("username", "(none)"))


# ---------------------------------------------------------------------------
# MCP session management
# ---------------------------------------------------------------------------
def _streamlit_session_id() -> str:
    """The current Streamlit session id, or "" if it cannot be determined.

    Passed to ``set_ws_user`` so PyCaber can tie the authenticated WebSocket
    connection to this user. Best-effort: a missing session id degrades identity
    seeding to the direct trace-context route, which is what actually carries the
    acting user for in-process agent actions anyway.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return getattr(get_script_run_ctx(), "session_id", "") or ""
    except Exception:  # noqa: BLE001 - never let diagnostics break a run
        return ""


def get_next_request_id():
    """Increment and return the next JSON-RPC request ID for this Streamlit session.

    JSON-RPC requires each request to carry a unique ID so the client can
    correlate responses. The counter lives in session state so it survives
    Streamlit reruns without resetting to zero mid-session.

    Returns:
        int: Monotonically increasing request ID, starting at 1.
    """
    st.session_state.mcp_request_id += 1
    return st.session_state.mcp_request_id


def init_mcp_session(tenant_id: str) -> bool:
    """Initialize (or reuse) a stateful MCP session for the given tenant.

    MCP (Model Context Protocol) requires a session handshake before tool calls
    can be made. This function is idempotent: if a session already exists for the
    same tenant it returns immediately. If the tenant has changed (user switching
    tenants mid-session), the old session is discarded and a new one is
    established.

    The session ID returned by the server is stored in ``st.session_state`` and
    forwarded as the ``Mcp-Session-Id`` header on subsequent tool calls.  The
    ``X-Caber-Tenant`` header is sent on every request so the MCP server can
    apply per-tenant access controls. The acting user is not a header: it
    travels as baggage ``caber.user``, which ``pycaber.auto`` adds to every
    intercepted httpx request from the trace context.

    Args:
        tenant_id: Opaque tenant identifier read from ``user_config.yaml``.  Must
            match the value the MCP server and Caber gateway are configured to
            recognise.

    Returns:
        True if a session is ready (either pre-existing or freshly created).
        False if the HTTP request failed or the server returned a JSON-RPC error.
    """
    if st.session_state.mcp_session_id is not None and st.session_state.mcp_tenant_id == tenant_id:
        return True
    # Tenant changed or no session yet -- reset and create a new session
    st.session_state.mcp_session_id = None

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{MCP_SERVER_URL}/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": get_next_request_id(),
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "agentgov-chat", "version": "1.0.0"},
                    },
                },
                headers={
                    "Content-Type": "application/json",
                    "X-Caber-Tenant": tenant_id,
                },
            )
            resp.raise_for_status()
            result = resp.json()
            if "error" not in result:
                st.session_state.mcp_session_id = resp.headers.get(
                    "Mcp-Session-Id", f"session-{get_next_request_id()}"
                )
                st.session_state.mcp_tenant_id = tenant_id
                return True
            st.error(f"MCP init failed: {result['error']}")
            return False
    except Exception as e:
        st.error(f"MCP connection failed: {e}")
        return False


def call_mcp_tool(tool_name: str, arguments: Dict[str, Any], tenant_id: str):
    """Call an MCP tool on the remote MCP server and return its result.

    Sends a JSON-RPC 2.0 ``tools/call`` request over a single HTTP POST. This is
    **not** MCP's Streamable HTTP transport, despite what this docstring used to
    claim: Streamable HTTP has five moving parts (POST *and* GET on one endpoint,
    an SSE stream with event ids and resumability, server-to-client messages,
    real session lifecycle, `Origin`/protocol-version validation) and this client
    exercises one of them. The accurate label is what the server itself reports:
    JSON-RPC 2.0 over HTTP. Corrected per the ticket-02 audit, which called out
    this exact docstring.

    The call is made via ``httpx``, which is transparently intercepted by
    ``pycaber.auto`` so PyCaber can attribute the hop to the authenticated user
    and record it in the audit trail. Note that this buffered-httpx path is
    precisely what ticket 15 found does NOT survive a real MCP SDK client (which
    streams ``tools/call``); the agent loop therefore governs at the framework
    tool-call wrapper instead, not on the wire.

    The ``X-Caber-Tenant`` header scopes the tool's data access to the requesting
    tenant. The ``Mcp-Session-Id`` header (set by ``init_mcp_session``) maintains
    conversation state across multiple tool invocations.

    Args:
        tool_name: Name of the registered MCP tool to invoke (e.g.
            ``"search_files_bm25"``).
        arguments: Dict of keyword arguments to pass to the tool, as defined by
            the tool's JSON Schema.
        tenant_id: Tenant identifier forwarded as ``X-Caber-Tenant`` so the MCP
            server applies the correct access policy.

    Returns:
        The ``result`` field from the JSON-RPC response (tool-specific shape).

    Raises:
        Exception: If ``init_mcp_session`` fails, the HTTP response is non-2xx,
            or the server returns a JSON-RPC error object.
    """
    if not init_mcp_session(tenant_id):
        raise Exception("Failed to initialize MCP session")

    _t0 = time.time()
    mcp_request = {
        "jsonrpc": "2.0",
        "id": get_next_request_id(),
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }

    acting_user = st.session_state.get("username", "")
    # Attach the classified purpose context (label + evidence) to the outbound
    # governed request so the server-side governor can consume it. The app does
    # not reimplement the taxonomy or re-decide enforcement here.
    purpose_headers = governance.purpose_request_headers(
        st.session_state.get("gov_purpose_context")
    )
    with httpx.Client(timeout=30.0) as client:
        logger.info(f"[SEQUENCE STEP 3] MCP Query: user={acting_user} groups={tenant_id} "
                    f"purpose_attached={bool(purpose_headers)}")
        resp = client.post(
            f"{MCP_SERVER_URL}/mcp",
            json=mcp_request,
            headers={
                "Content-Type": "application/json",
                "Mcp-Session-Id": st.session_state.mcp_session_id,
                "X-Caber-Tenant": tenant_id,
                **purpose_headers,
            },
        )
        resp.raise_for_status()
        result = resp.json()

    if "error" in result:
        raise Exception(f"MCP tool error: {result['error']}")

    logger.info("[PERF] call_mcp_tool %s: %.2fs", tool_name, time.time() - _t0)
    return result.get("result", {})


# ---------------------------------------------------------------------------
# GraphRAG integration
# ---------------------------------------------------------------------------
def get_authorized_docs(tenant_groups: List[str]) -> List[Dict[str, Any]]:
    """Retired legacy GraphRAG ``Object.auth`` pre-filter (Task 13).

    The ``/graphrag/authorized-objects`` endpoint and its ``Object.auth`` allow/deny
    Cypher predicate were removed: authorization is now decided solely by the shared
    governance evaluator on the governed paths (``/graphrag/scoped-search`` and the
    MCP-server governance enforcement), not by a caller-side allow/deny pre-filter.
    This function is kept as a no-op returning an empty list so ``search_files``
    performs an unfiltered BM25 query that the governed MCP boundary then enforces.
    """
    return []


def get_provenance(object_id: str) -> List[Dict[str, Any]]:
    """Query the GraphRAG retriever for the provenance chain of a document.

    Provenance records describe how content was derived or shared: which source
    documents contributed chunks to the requested object, how many n-mer chunks
    overlap, and what the related object names are.  This information is displayed
    in the sidebar so users can trace the lineage of documents returned by search.

    If the GraphRAG service is unavailable the function logs a warning and returns
    an empty list; the sidebar silently shows nothing for that object.

    Args:
        object_id: Opaque identifier of the document to query (as returned in the
            ``id`` field of ``get_authorized_docs`` results).

    Returns:
        List of provenance record dicts.  Each record typically contains
        ``source_name``, ``related_name``, and ``shared_chunks`` fields.
        Returns an empty list if the service is unreachable or returns no data.
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{GRAPHRAG_URL}/graphrag/provenance",
                json={"object_id": object_id},
            )
            resp.raise_for_status()
            return resp.json().get("provenance", [])
    except Exception as e:
        logger.warning("GraphRAG provenance query failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Search with GraphRAG pre-filter
# ---------------------------------------------------------------------------
def search_files(query: str, tenant_id: str, tenant_groups: List[str]):
    """Run a two-stage authorization-aware search and return matching documents.

    Stage 1 -- GraphRAG pre-filter:
        Calls ``get_authorized_docs`` to obtain the set of object storage paths
        the user is permitted to access.  If GraphRAG is available and returns
        results, these paths are passed as ``filter_paths`` to the MCP tool so
        BM25 ranking is constrained to authorized objects only.

    Stage 2 -- MCP BM25 search:
        Calls the ``search_files_bm25`` MCP tool via ``call_mcp_tool``.  The
        ``X-Caber-Tenant`` header on the underlying HTTP request ensures the MCP
        server applies its own tenant-level access controls as a second defence.
        ``pycaber.auto`` intercepts both the GraphRAG and MCP HTTP calls to record
        them in the audit trail.

    Each raw search result is normalised into a document dict compatible with the
    LLM context builder:
    - ``page_content``: best available text (snippets > preview > content field).
    - ``metadata``: path, snippets, source label, and raw BM25 score.

    Performance timings for each stage are logged at INFO level.

    Args:
        query: Free-text search query entered by the user.
        tenant_id: Tenant identifier forwarded to the MCP server via
            ``X-Caber-Tenant``.
        tenant_groups: Group memberships used to scope the GraphRAG pre-filter
            query.

    Returns:
        Tuple of:
        - ``documents`` (List[dict]): Normalised document dicts ready for LLM
          context assembly.
        - ``authorized_objects`` (List[dict]): Raw GraphRAG result list, used to
          render the provenance sidebar.
        - ``mcp_result`` (dict): The raw MCP tool result, which may carry a
          governance decision projected by the interception layer
          (``_meta["ai.caber/governance"]``) for the caller to consume.
    """
    _t0 = time.time()

    # Step 1: Get authorized doc paths from GraphRAG
    authorized_objects = get_authorized_docs(tenant_groups)
    _t_graphrag = time.time() - _t0
    filter_paths = [obj.get("name") or obj.get("id", "") for obj in authorized_objects] or None

    # Step 2: MCP BM25 search (scoped by tenant header, optionally filtered by GraphRAG)
    arguments = {"query": query, "max_results": 8}
    if filter_paths:
        arguments["filter_paths"] = filter_paths

    _t1 = time.time()
    result = call_mcp_tool("search_files_bm25", arguments, tenant_id)
    _t_mcp = time.time() - _t1
    logger.info("[PERF] search_files: graphrag=%.2fs mcp_tool=%.2fs total=%.2fs",
                _t_graphrag, _t_mcp, time.time() - _t0)

    # The MCP interception may project a governance decision onto this result
    # (clarification / block). Surface the raw result so the caller can consume
    # the projection without the app re-deciding enforcement.
    _last_mcp_result = result

    # Convert to document-like dicts
    documents = []
    for res in result.get("results", []):
        if isinstance(res, str):
            # Withheld whole by a tool-boundary guard: render the marker.
            documents.append({
                "page_content": res,
                "metadata": {"path": "withheld", "snippets": [res],
                             "source": "mcp_bm25", "bm25_score": 0.0},
            })
            continue
        if not isinstance(res, dict):
            continue
        content_sources = [
            "\n".join(res.get("snippets", [])),
            res.get("preview", ""),
            res.get("content", ""),
        ]
        content = next((s for s in content_sources if s and s.strip()), "No content available")
        documents.append({
            "page_content": content,
            "metadata": {
                "path": res.get("path", "Unknown"),
                "snippets": res.get("snippets", []),
                "source": "mcp_bm25",
                "bm25_score": res.get("score", 0.0),
            },
        })

    return documents, authorized_objects, _last_mcp_result


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
with open(os.path.join(SCRIPT_DIR, "user_config.yaml")) as f:
    config = yaml.load(f, Loader=SafeLoader)

# Users log in by EMAIL only. Every login name must be a directory user's email,
# because the login name is the subject this app asserts on every hop. Checked
# before stauth sees the credentials; a bad config refuses to serve at all.
try:
    directory_lookup.validate_login_config(config["credentials"]["usernames"])
except directory_lookup.DirectoryLookupError as exc:
    logger.error("[AUTH] invalid login configuration: %s", exc)
    st.error(f"Invalid login configuration: {exc}")
    st.stop()

# USER_STATE_TRACKING
# This app tracks the user by calling pycaber.set_ws_user() after login. Do NOT add config["cookie"]["name"]
# to the userIdKeys parameter of the ApiMap for this app: the stauth cookie is a JWT carrying only
# {username, exp_date}, with no `iss` or `sub`, so PyCaber cannot build a canonical subject from it and logs
# an ERROR on every frame. Using both methods at the same time may also cause unexpected results.
authenticator = stauth.Authenticate(
    credentials=config["credentials"],
    cookie_name=config["cookie"]["name"],
    cookie_key=config["cookie"]["key"],
    cookie_expiry_days=config["cookie"]["expiry_days"],
    auto_hash=True,
)

# On page navigation with a valid cookie, auth takes 3-4 reruns to resolve:
# RERUN #1-2: auth_status=None (login form rendered briefly)
# RERUN #3-4: auth_status=True (cookie validated, authenticated view)
# Playwright tests must wait ~8s for the textarea, not react to the login form.
authenticator.login(location="main", max_login_attempts=2, fields={"Username": "Email"})

# Debug: log post-login state so we can see what the authenticator decided
_post_auth = st.session_state.get("authentication_status", None)
_post_user = st.session_state.get("username", "(none)")
_post_name = st.session_state.get("name", "(none)")
logger.info("[AUTH DEBUG] post-login: auth_status=%s user=%s name=%s cookie_name=%s",
            _post_auth, _post_user, _post_name, config["cookie"]["name"])


# ---------------------------------------------------------------------------
# Get tenant info for authenticated user
# ---------------------------------------------------------------------------
def get_tenant_info(uname: str) -> tuple:
    """Return the tenant and group memberships for a given login email.

    ``user_config.yaml`` holds credentials, display name and email only. The
    tenant (SCIM ``organization``) and groups come from the zt-mimic directory
    seed, the demo's single user source (see ``directory_lookup``). They feed
    the sidebar and the ``X-Caber-Tenant`` header; Caber itself never
    authorizes from them, it resolves groups from its directory replica.

    Args:
        uname: The authenticated login email from ``st.session_state["username"]``.

    Returns:
        Tuple of ``(tenant_id, tenant_groups)``.

    Raises:
        directory_lookup.DirectoryLookupError: the seed is missing or does not
            define ``uname``. Never defaulted: a guessed tenant would show the
            wrong context and send the wrong header.
    """
    return directory_lookup.tenant_and_groups(uname)


def get_user_email(uname: str) -> str:
    """The login user's email from ``user_config.yaml``; empty when none is set."""
    return config["credentials"]["usernames"].get(uname, {}).get("email", "") or ""


# ---------------------------------------------------------------------------
# UI Header
# ---------------------------------------------------------------------------
col1, col2 = st.columns([0.15, 3])
with col2:
    st.title("AgentGov Demo - Agentic Chat")

st.markdown(
    """<div style="font-size:16px; color:gray; margin-bottom:20px;">
    Tenant-isolated AI agent chat with MCP file search.  Caber provides 
    tenant isolation on data use, audit trails, and data lineage tracking.
    </div>""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main application (authenticated)
# ---------------------------------------------------------------------------
if st.session_state.get("authentication_status"):
    name = st.session_state["name"]
    username = st.session_state["username"]

    try:
        tenant_id, tenant_groups = get_tenant_info(username)
    except directory_lookup.DirectoryLookupError as exc:
        logger.error("[AUTH] directory lookup failed for user=%s: %s", username, exc)
        st.error(f"Cannot load directory context for {username!r}: {exc}")
        st.stop()
    user_email = get_user_email(username)
    logger.info("[AUTH DEBUG] AUTHENTICATED branch: user=%s tenant=%s groups=%s", username, tenant_id, tenant_groups)

    # -----------------------------------------------------------------------
    # Identity context propagation (set_ws_user)
    # Pin the human caller to this WS connection as issuer + subject. Tenant
    # and groups are NOT passed: traffic-asserted attributes never authorize,
    # Caber resolves them from the directory replica.
    # -----------------------------------------------------------------------
    from streamlit.runtime.scriptrunner import get_script_run_ctx as _get_ctx
    _st_session_id = getattr(_get_ctx(), "session_id", "") or ""

    logger.info("[AUTH DEBUG] calling set_ws_user: subject=%s issuer=%s session=%s",
                username, CHAT_APP_ISSUER, _st_session_id[:16])
    st.session_state._ws_pin_id = pycaber.set_ws_user(
        subject=username,
        issuer=CHAT_APP_ISSUER,
        email=user_email or None,
        authenticator=CHAT_APP_AUTHENTICATOR,
        acceptor=CHAT_APP_ACCEPTOR,
        session=_st_session_id,
    )
    logger.info("[AUTH DEBUG] set_ws_user returned pin=%s", st.session_state._ws_pin_id)

    def _on_logout(_params=None):
        """Streamlit authenticator logout callback: release the PyCaber WebSocket pin.

        ``pycaber.set_ws_user`` returns an opaque pin that ties the authenticated
        Streamlit WebSocket connection to the user's identity in PyCaber's session
        cache.  If the pin is not explicitly cleared on logout, PyCaber continues
        to attribute subsequent WebSocket traffic from the same connection to the
        previous user, producing incorrect audit attributions.

        Args:
            _params: Unused; present to satisfy the streamlit-authenticator
                callback signature which may pass a dict of button event data.
        """
        pin = st.session_state.get("_ws_pin_id", "")
        if pin:
            pycaber.clear_ws_user(pin)
            st.session_state._ws_pin_id = ""

    authenticator.logout(button_name=f"Logout :green[{name}]", location="main",
                         callback=_on_logout)

    # Sidebar: tenant info and provenance
    with st.sidebar:
        st.subheader("Tenant Context")
        st.markdown(f"**User:** {name}")
        st.markdown(f"**Tenant:** `{tenant_id}`")
        st.markdown(f"**Groups:** `{', '.join(tenant_groups)}`")
        st.markdown(f"**LLM Model:** `{OPENAI_LLM_MODEL}`")
        st.markdown("---")
        st.subheader("Data Provenance")
        provenance_container = st.container()

    openai_client = OpenAI(timeout=60.0, base_url=OPENAI_BASE_URL)

    # Load demo questions
    demo_questions = {}
    try:
        questions_path = os.environ.get(
            "DEMO_QUESTIONS_FILE",
            os.path.join(SCRIPT_DIR, "default_demo_questions.json"),
        )
        with open(questions_path, "r") as fin:
            demo_questions = json.load(fin)
    except Exception:
        pass

    # Questions are keyed by the login email (the session user's subject).
    questions_list = demo_questions.get(username, [])
    random_question = random.choice(questions_list) if questions_list else ""

    retrieved_lines_with_metadata = []
    # Initialised out here because the post-form blocks below read it whether or
    # not the form was submitted on this rerun.
    agent_run = None

    # -----------------------------------------------------------------------
    # Encoded Question Input (Avoid Pre-Populating OBO Index With Default Prompt)
    #
    # Problem: Streamlit normally sends the default textarea value as plaintext
    # over its WebSocket connection (server -> browser). PyCaber's pipeline
    # extracts n-mers (content-defined chunk hashes) from all traffic. When the
    # user submits the form with the same text, PyCaber sees matching n-mers in
    # the request. Because the WebSocket message is a principal-owned chain root
    # (obo.resolution = root), those matching n-mers do not overwrite the chain
    # principal, but they would pre-populate the OBO index with default-question
    # n-mers before the user ever submits a turn and could trigger false
    # cross-principal content findings if another user receives the same default
    # question render.
    #
    # Solution: The encoded_input custom component XOR-encrypts the default
    # question with a fresh random 32-byte key on every render. The server
    # sends (xor_encoded_bytes, key) as component args over the WebSocket.
    # The browser JavaScript decodes client-side and displays plaintext in a
    # textarea. PyCaber sees only random noise in the server->browser frame,
    # producing n-mers with zero overlap to the plaintext the user later
    # submits.
    #
    # Data flow:
    #   Server  --[XOR noise + key]--> Browser (WS frame, opaque to PyCaber)
    #   Browser --[plaintext]---------> Server  (form submit, user-originated)
    #
    # The XOR key rotates every render, so even repeated page loads produce
    # entirely different encoded byte sequences.
    # -----------------------------------------------------------------------
    with st.form("chat_form"):
        logger.info("[SEQUENCE STEP 0] Sending XOR-encoded random question to browser")

        question = encoded_text_area(
            "Enter your question:",
            default_value=random_question,
            height=100,
            key="encoded_question",
        )
        submitted = st.form_submit_button("Submit")

        if question and submitted:
            logger.info("[SEQUENCE STEP 1] Prompt Entered: user=%s tenant=%s prompt=%r", username, tenant_id, question)
            # Declare this question as the meaningful user request for OBO chain
            # tracking. The ClarificationSession registers it as a NEW request (a
            # fresh OBO root): if the previous turn was a clarification, this answer
            # is not a continuation of the prior interaction.
            clar_session = st.session_state._clar_session
            if clar_session is None:
                clar_session = governance.ClarificationSession(
                    register_request=pycaber.register_request,
                    register_response=pycaber.register_response,
                    api="localhost:8083",
                )
                st.session_state._clar_session = clar_session
            interaction = clar_session.consume_user_message(question)

            # Classify the purpose via GLiNER2 and attach its evidence context onto
            # the outbound governed request (see call_mcp_tool). No taxonomy or
            # enforcement is reimplemented here -- the label + evidence are attached
            # for the server-side governor to consume.
            st.session_state.gov_purpose_context = None
            purpose_ctx = classify_purpose_context(question)
            if purpose_ctx is not None:
                attached = governance.attach_purpose_context({}, purpose_ctx)
                st.session_state.gov_purpose_context = attached
                logger.info("[GOVERNANCE] purpose=%s confidence=%.2f model=%s digest=%s",
                            purpose_ctx.purpose, purpose_ctx.confidence,
                            purpose_ctx.model_id, purpose_ctx.prompt_digest[:12])

            _perf_start = time.time()
            _perf = {}  # stage -> seconds

            if AGENT_ENABLED:
                # No identity call here. run_agent drives its event loop on
                # this script thread, where PyCaber resolves the user pinned
                # by set_ws_user above from this Streamlit session's own
                # record, exactly as on the direct BM25/OpenAI path below.

                # -- Agent loop: the agent chooses its own tools --
                # Caber governs this at the LangChain BaseTool wrapper, so the
                # redaction decision lands on each tool result BEFORE the model
                # reads it -- not on the graph's return value afterwards, which
                # would be the wrong side of the trust boundary.
                _t = time.time()
                try:
                    agent_run = agent_loop.run_agent(
                        question,
                        tenant_id=tenant_id,
                        tenant_groups=tenant_groups,
                        extra_headers=governance.purpose_request_headers(
                            st.session_state.gov_purpose_context
                        ),
                    )
                except Exception as e:
                    # Surface the failure. Never fall back to the legacy hardcoded
                    # call and present it as an agent run -- that would fake a
                    # successful agentic retrieval.
                    logger.exception("[AGENT] run failed")
                    st.chat_message("user").write(question)
                    st.chat_message("assistant").error(
                        f"Agent run failed: {e}"
                    )
                    st.stop()
                _perf["agent"] = time.time() - _t

                mcp_docs = agent_run.documents
                mcp_result = agent_run.last_tool_result
                authorized_objects = []  # GraphRAG sidebar retired (ticket 10)
                logger.info(
                    "[AGENT] user=%s tools_called=%s sources=%s",
                    username,
                    [c.tool for c in agent_run.tool_call_log],
                    agent_run.source_paths,
                )
            else:
                # -- Legacy path: GraphRAG pre-filter + one hardcoded BM25 call --
                # pycaber.auto intercepts all httpx calls (hop 2 to MCP, GraphRAG)
                _t = time.time()
                mcp_docs, authorized_objects, mcp_result = search_files(
                    question, tenant_id, tenant_groups
                )
                _perf["search"] = time.time() - _t

            # -- Consume any governance decision the MCP interception projected --
            # The ClarificationSession registers the rendered text as the OBO
            # response and marks (for clarifications) that the next turn is a new
            # request. The decision is server-authoritative; the app does not
            # re-decide it.
            gov_plan = clar_session.on_tool_result(mcp_result, interaction=interaction)
            if gov_plan["skip_llm"]:
                # Clarification or block: render the EXACT policy-authored text.
                # Do NOT call the LLM.
                render_text = gov_plan["render_text"] or ""
                logger.info("[GOVERNANCE] state=%s decision_id=%s -- skipping LLM",
                            governance.governance_state(mcp_result), gov_plan.get("decision_id"))
                st.chat_message("user").write(question)
                st.chat_message("assistant").write(render_text)
                st.stop()

            # Show provenance in sidebar.
            #
            # In agent mode this renders the agent's own tool-selection trace
            # instead of the GraphRAG sidebar. Ticket 10 retires the chat app's
            # `get_provenance -> GRAPHRAG_URL` panel: the retriever is reached only
            # through the MCP server's `search_graph` tool (graphrag demo), and the
            # real provenance moment lives on /admin/audit, rendered from Neo4j +
            # recorded audit events. What belongs here is the thing this demo is
            # about -- which tools the agent chose, and what they returned.
            if AGENT_ENABLED and agent_run is not None:
                with provenance_container:
                    st.markdown(f"**{len(agent_run.tool_call_log)}** tool call(s)")
                    for call in agent_run.tool_call_log:
                        st.markdown(f"- `{call.tool}`")
                        if call.args:
                            preview = json.dumps(call.args)[:80]
                            st.caption(f"  {preview}")
                        for path in call.source_paths[:3]:
                            st.caption(f"  -> `{path}`")
                    if agent_run.hit_iteration_limit:
                        st.warning("Agent reached its iteration bound.")
                    st.caption(
                        "Tools available: " + ", ".join(agent_run.tool_names_available)
                    )
            else:
                with provenance_container:
                    if authorized_objects:
                        st.markdown(f"**{len(authorized_objects)}** authorized objects from GraphRAG")
                        for obj in authorized_objects[:5]:
                            obj_id = obj.get("id", "unknown")
                            st.markdown(f"- `{obj.get('name') or obj_id}`")
                            prov = get_provenance(obj_id)
                            if prov:
                                for p in prov[:2]:
                                    st.caption(
                                        f"  {p.get('source_name', '?')} "
                                        f"({p.get('shared_chunks', 0)} shared chunks) "
                                        f"-> {p.get('related_name', '?')}"
                                    )
                    else:
                        st.info("GraphRAG not available or no authorized objects found. MCP search used directly.")

            # Build the retrieved-documents panel from whatever the retrieval
            # stage produced (agent tool results, or the legacy BM25 hit list).
            if mcp_docs:
                retrieved_lines_with_metadata = [
                    (d["page_content"], d["metadata"]) for d in mcp_docs
                ]
                context = "\n".join(d["page_content"] for d in mcp_docs)
            else:
                context = "No relevant documents found."

            if AGENT_ENABLED and agent_run is not None:
                # The agent has already reasoned over its own tool results and
                # written the answer. Calling the LLM again here would be a
                # second, ungoverned summarisation pass over context the agent
                # already consumed -- and would re-send document text that the
                # tool-boundary redaction had just decided about.
                answer = agent_run.answer
                logger.info(
                    "[SEQUENCE STEP 4] Agent produced the answer directly: user=%s tenant=%s",
                    username, tenant_id,
                )
            else:
                # -- Call LLM --
                # pycaber.auto intercepts the OpenAI SDK httpx calls (hop 3)
                _t = time.time()
                try:
                    logger.info("[SEQUENCE STEP 4] Sending to LLM: user=%s tenant=%s", username, tenant_id)
                    answer = get_llm_answer(openai_client, context, question, model=OPENAI_LLM_MODEL)
                except Exception as e:
                    logger.error("LLM call failed: %s", e)
                    answer = f"LLM error: {e}"
                _perf["llm"] = time.time() - _t

            _perf["total"] = time.time() - _perf_start
            _stages = " ".join(f"{k}={v:.2f}s" for k, v in _perf.items())
            logger.info("[PERF] %s query: %s", username, _stages)

            # Escape special characters so they display correctly in streamlit.
            answer = answer.replace("%", "\\%").replace("$", "\\$")

            # Close the OBO interaction chain by declaring the final answer content.
            # Called BEFORE st.chat_message renders so the outgoing ForwardMsg can be
            # matched when it appears on the wire. Links back to the request via the
            # interaction handle. No need for mark_all_ws_responded() -- the registered
            # interaction model handles chain lifecycle explicitly per question/answer pair.
            pycaber.register_response(answer, interaction=interaction)

            # Display chat
            st.chat_message("user").write(question)
            logger.info("[SEQUENCE STEP 5] Returning answer to user: user=%s tenant=%s prompt=%r", username, tenant_id, question)
            st.chat_message("assistant").write(answer)

            # # Since Streamlit runs over Websockets there is no straightofrward way to associate requests and
            # # responses.  PyCaber automatically submits all "BackMsg" coming from the client browser to this server
            # # as "requests".  Since we just delivered the response, we can mark all open requests as responded.
            # pycaber.mark_all_ws_responded()

    # -----------------------------------------------------------------------
    # Expose the agent's tool_call_log as the INDEPENDENT record for CI.
    #
    # Ticket 11's invariant is a reconciliation between two records captured at
    # different layers: what the agent thinks it did (this log, from LangGraph's
    # message state) versus what Caber intercepted (rendered by /admin/audit from
    # the BaseTool-wrapper capture). Comparing the audit UI against the same
    # source the audit UI renders would prove nothing, so this has to be reachable
    # by the test independently.
    #
    # Deliberately omits `result_preview`: the tool names, arguments and source
    # paths are what the five invariants and the citation-set outcome gate compare,
    # and pushing document text back through the Streamlit WebSocket for no reason
    # would add fingerprintable noise to the very traffic under test.
    # -----------------------------------------------------------------------
    if AGENT_ENABLED and agent_run is not None:
        _ci_record = {
            "user": username,
            "tenant": tenant_id,
            "hit_iteration_limit": agent_run.hit_iteration_limit,
            "tools_available": agent_run.tool_names_available,
            "tool_calls": [
                {"tool": c.tool, "args": c.args, "source_paths": c.source_paths,
                 "is_error": c.is_error}
                for c in agent_run.tool_call_log
            ],
            "source_paths": agent_run.source_paths,
        }
        st.markdown(
            '<div id="agent-tool-call-log" data-testid="agent-tool-call-log" '
            'style="display:none">'
            f"{json.dumps(_ci_record)}"
            "</div>",
            unsafe_allow_html=True,
        )
        # Also dump to disk when asked, for drivers that never open a browser
        # (e.g. the deterministic multi-user CI driver).
        _log_path = os.getenv("AGENT_TOOL_CALL_LOG_PATH")
        if _log_path:
            try:
                with open(_log_path, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps(_ci_record) + "\n")
            except OSError as exc:
                logger.warning("could not write AGENT_TOOL_CALL_LOG_PATH=%s: %s",
                               _log_path, exc)

    # Display retrieved documents
    if retrieved_lines_with_metadata:
        st.markdown("---")
        st.subheader("Retrieved data sent to LLM:")
        for i, (text, metadata) in enumerate(retrieved_lines_with_metadata):
            clean_text = text[:300].replace("\n", " ").strip()
            try:
                formatted_metadata = json.dumps(metadata, indent=2, ensure_ascii=False)
                formatted_metadata = formatted_metadata.replace("\n", "<br>").replace("  ", "&nbsp;&nbsp;")
            except Exception:
                formatted_metadata = str(metadata)

            st.markdown(
                f"""<div style="margin-bottom:15px;" data-caber="Caber_Response_Element">
                <div style="font-weight:500;">Document[{i+1}]:</div>
                <div style="font-size:0.9em; font-weight:300;">{clean_text}...</div>
                <div style="font-size:0.85em; color:lightblue; padding-left:20px;">{formatted_metadata}</div>
                </div>""",
                unsafe_allow_html=True,
            )

elif st.session_state.get("authentication_status") is False:
    logger.info("[AUTH DEBUG] FAILED branch: auth_status=False -- rendering error message")
    _stale_pin = st.session_state.get("_ws_pin_id", "")
    if _stale_pin:
        pycaber.clear_ws_user(_stale_pin)
        st.session_state._ws_pin_id = ""
    st.error("Email/password is incorrect")

elif st.session_state.get("authentication_status") is None:
    logger.info("[AUTH DEBUG] NONE branch: auth_status=None -- rendering login form")
    _stale_pin = st.session_state.get("_ws_pin_id", "")
    if _stale_pin:
        pycaber.clear_ws_user(_stale_pin)
        st.session_state._ws_pin_id = ""
    st.warning("Please enter your email and password")
