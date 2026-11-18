"""The agent loop: a LangGraph ReAct agent over the MCP server's tool catalog.

This is what makes the demo's central claim true -- that an LLM *genuinely
chooses* which retrieval tool to use -- and it replaces the old back door, a
single hardcoded ``search_files_bm25`` call on the raw user question.

Why LangChain, specifically
---------------------------
Not a taste call. Ticket 15 established that Caber's interception point is the
**framework tool-call wrapper**, not the HTTP transport: ``pycaber.auto`` has no
stdio or MCP-SDK patch, and its httpx patch only sees buffered bodies while the
official MCP SDK *streams* every ``tools/call``. A raw ``ClientSession`` loop
would therefore be **invisible to Caber -- no redaction, no provenance**.

``langchain-mcp-adapters`` turns each MCP tool into a ``langchain_core.tools``
``BaseTool``, and ``BaseTool.invoke/ainvoke/run/arun`` is exactly what
``pycaber.auto`` patches. Redaction fires on the assembled ``CallToolResult``,
which is transport-agnostic -- the ticket-09 spike confirmed it works even over
stdio. Governing a real LangChain agent is also Caber's most mature integration,
so the demo shows the actual product at its strongest point rather than a
bespoke decorator path.

Two things this module must get right or the demo silently lies
--------------------------------------------------------------
1. **Import order.** ``pycaber.auto`` must be imported before ``langchain_core``,
   or ``BaseTool`` is never patched and every tool result reaches the model in
   plaintext -- with no error anywhere. ``app.py`` imports ``pycaber.auto`` at
   module top; this module additionally imports langchain lazily *inside*
   functions. If ``BaseTool`` is not fully patched, PyCaber itself reports it:
   a ``post_log`` error row, a WARNING, and an unhealthy SDK status.

2. **The acting user.** Agent actions are posted under the container host
   with framework selector (e.g. ``<host>:langchain``) and the OBO decision needs
   an acting principal. This module never sets one: the app pins the login with
   ``pycaber.set_ws_user()`` and PyCaber resolves it for every call made on the
   Streamlit script thread, which is where ``run_agent`` drives its event loop.
   Demo code uses only PyCaber's public API; it never builds or writes a trace
   context of its own.

Also note the ApiMap prerequisite, which lives in the demo config rather than
here: without the bare ``selfAC`` entry (which expands to the container hostname the
SDK posts agent actions under, scheme ``framework``), the agent host falls through
to the ``default`` entry (``function: "log"``), which ``is_drop()`` treats as a drop --
the body is emptied *before* fingerprinting, so no decision can exist at all.

This module deliberately imports no Streamlit. The CI driver needs to run the
same loop headlessly and compare its ``tool_call_log`` against ``/admin/audit``
(ticket 11), which only works if the loop is callable outside the UI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import llm_usage  # beside this file in /app; stdlib only, so import order is unaffected

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://mcp-server-back-door:3010")

# gpt-4o, not the gpt-4o-mini the legacy path defaulted to. Ticket 06 measured
# four-tool selection legibility at 11/11 on gpt-4o, and 6/6 on the
# qualification-critical semantic-vs-BM25 distinction. Selection legibility is
# model-specific, so changing this invalidates that measurement -- re-run
# prototypes/06-tool-selection/run.py --model <new> before switching.
AGENT_MODEL = os.getenv("USE_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o"

# Ticket 09 mechanic 2: reuse the archived streamlit-mock-mcp demo's 10-iteration cap as LangGraph's
# recursion_limit. An agent that genuinely chooses can also loop.
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))

# Temperature 0. Ticket 10 pins the agent at temp 0 for the video so tool
# selection is reproducible and narration segments can name specific tools; it
# costs CI nothing and makes failures reproducible.
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0"))

MAX_ITERATIONS_PREFIX = "[reached max iterations] "

SYSTEM_PROMPT = """You are a retrieval assistant with access to a document store.

Answer the user's question using ONLY information you retrieve with the tools,
unless the user explicitly asks you to answer without using any tools or gives
you an excerpt to summarize directly (in which case answer directly without
calling any tools).

Be decisive, concise, and efficient:
1. Keep every answer and summary concise (1 to 2 short paragraphs, 60 to 120
   words total).
2. Never call list_files unless the user explicitly asks to list files.
3. When the user names a specific document path (such as docs/... or
   contracts/...), call read_file on that path once.
4. For any other question about document contents, terms, or provisions, call
   at most ONE search tool (either search_files_bm25 OR search_semantic, never
   both) in a single turn, then write your final answer immediately from the
   returned snippets without calling read_file or running additional searches.
5. Only call save_file when the user explicitly asks you to save a file.

Always cite the source document paths you used, as a list at the end of your
answer, prefixed with "Sources:". If the tools return nothing relevant, say so
plainly -- never invent an answer or a citation."""


@dataclass
class ToolCall:
    """One observed tool call.

    This is the "logged, never asserted" observation record from ticket 09
    mechanic 3. CI must never assert *which* tool the agent picked -- that would
    gate the build on model behaviour and turn a legitimate alternative choice
    into a red build. It is recorded because it feeds two things that ARE
    checked: the citation-set outcome gate, and the ticket-11 reconciliation
    against ``/admin/audit``.
    """

    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    result_preview: str = ""
    # Required, not optional: without source paths the answer->document trace
    # cannot be reconstructed and the provenance pillar has nothing to render.
    source_paths: List[str] = field(default_factory=list)
    is_error: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "args": self.args,
            "result_preview": self.result_preview,
            "source_paths": self.source_paths,
            "is_error": self.is_error,
        }


@dataclass
class AgentRun:
    """The result of one agent turn."""

    answer: str
    tool_call_log: List[ToolCall] = field(default_factory=list)
    # Normalised documents for the "retrieved data sent to LLM" panel, in the same
    # shape the legacy search_files() returned so the UI code is unchanged.
    documents: List[Dict[str, Any]] = field(default_factory=list)
    # The last raw tool payload, so the caller can consume a governance decision
    # projected onto it without the app re-deciding enforcement.
    last_tool_result: Optional[Dict[str, Any]] = None
    hit_iteration_limit: bool = False
    tool_names_available: List[str] = field(default_factory=list)

    @property
    def source_paths(self) -> List[str]:
        """Every distinct source path the agent actually retrieved, in order.

        This is what the citation-set outcome gate compares against.
        """
        seen, out = set(), []
        for call in self.tool_call_log:
            for path in call.source_paths:
                if path not in seen:
                    seen.add(path)
                    out.append(path)
        return out

    def log_as_dicts(self) -> List[Dict[str, Any]]:
        return [c.as_dict() for c in self.tool_call_log]


# ---------------------------------------------------------------------------
# Extracting the observation record from LangGraph's message state
# ---------------------------------------------------------------------------

def _first_text(content: Any) -> str:
    """Flatten a LangChain message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return "" if content is None else str(content)


def _unwrap_structured(payload: Any) -> Optional[Dict[str, Any]]:
    """Unwrap a structured-content envelope, whichever spelling it arrives in.

    langchain-mcp-adapters 0.3.1 hands the MCP ``structuredContent`` field back on
    the ToolMessage's ``artifact`` as ``{"structured_content": {...}}`` -- note the
    **snake_case** key, which is the adapter's Python-side renaming of the
    protocol's camelCase field. Both spellings are accepted so a future adapter
    version that stops renaming does not silently drop every source path (and with
    it the provenance pillar and the citation-set gate).
    """
    if not isinstance(payload, dict):
        return None
    for key in ("structured_content", "structuredContent"):
        inner = payload.get(key)
        if isinstance(inner, dict):
            return inner
    return payload


def _structured_from_tool_message(message: Any) -> Optional[Dict[str, Any]]:
    """Recover the tool's structured payload from a ToolMessage.

    Two routes, because where the payload lands depends on the adapter's
    ``response_format`` (``content_and_artifact`` for MCP tools):
      1. ``message.artifact`` -- verified to carry
         ``{"structured_content": {...}}``. This is the route that works for
         ``read_file``, whose content block is the raw document text with no JSON
         to parse.
      2. the text content parsed as JSON -- our MCP server mirrors the structured
         payload into the text block for the search tools, so a text-only client
         still sees everything.
    """
    artifact = getattr(message, "artifact", None)
    unwrapped = _unwrap_structured(artifact)
    if isinstance(unwrapped, dict) and unwrapped:
        return unwrapped
    if isinstance(artifact, list):
        for item in artifact:
            candidate = _unwrap_structured(item)
            if isinstance(candidate, dict) and (
                "results" in candidate or "path" in candidate or "files" in candidate
            ):
                return candidate

    text = _first_text(getattr(message, "content", ""))
    if text:
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return None
        if isinstance(parsed, dict):
            return _unwrap_structured(parsed) or parsed
    return None


def _source_paths_from(structured: Optional[Dict[str, Any]]) -> List[str]:
    """Pull source document paths out of a tool's structured payload.

    Handles the shapes the four-tool catalog produces:
      - search tools:  {"results": [{"path": ...}, ...]}
      - read_file:     {"path": ...}
      - list_files:    {"files": ["a", "b"]} or [{"path": ...}]
    """
    if not isinstance(structured, dict):
        return []

    paths: List[str] = []
    for key in ("structured_content", "structuredContent"):
        inner = structured.get(key)
        if isinstance(inner, dict):
            paths.extend(_source_paths_from(inner))

    for hit in structured.get("results") or []:
        if isinstance(hit, dict) and hit.get("path"):
            paths.append(str(hit["path"]))

    if structured.get("path"):
        paths.append(str(structured["path"]))

    for entry in structured.get("files") or []:
        if isinstance(entry, str):
            paths.append(entry)
        elif isinstance(entry, dict) and entry.get("path"):
            paths.append(str(entry["path"]))

    seen, out = set(), []
    for path in paths:
        if path and path not in seen and _is_document_path(path):
            seen.add(path)
            out.append(path)
    return out


def _is_document_path(path: str) -> bool:
    """Whether a path names a DOCUMENT rather than a directory.

    `list_files` returns directory entries too -- `contracts`, `contracts/amy/`,
    `contracts/bob/` -- and a live run duly reported them as retrieved sources.
    They are not documents: nothing was retrieved from them, they cannot be
    read_file'd, and letting them into the source set corrupts three things at
    once. The citation gate would compare against non-documents, the
    provenance-resolves invariant would look for lineage that cannot exist, and
    the policy oracle would try to parse an owner out of a two-segment path.

    Deliberately does NOT require a fixed depth. In MinIO mode paths are
    `<bucket>/<user>/<file>`, but in filesystem mode the server returns paths
    relative to its files root, which can be any depth. Requiring three segments
    here silently discarded valid sources. The three-segment shape IS asserted,
    but in the smoke test, where MinIO mode makes it a real guarantee.
    """
    if not path or path.endswith("/"):
        return False
    return "." in path.rsplit("/", 1)[-1]


def _documents_from(
    structured: Optional[Dict[str, Any]],
    tool_name: str,
    fallback_text: str = "",
) -> List[Dict[str, Any]]:
    """Normalise a tool payload into the document dicts the UI already renders.

    Same shape the legacy ``search_files()`` returned, so the "Retrieved data
    sent to LLM" panel and the retrieved-docs E2E check need no changes.
    """
    if not isinstance(structured, dict):
        return []

    inner = structured.get("structuredContent")
    if isinstance(inner, dict) and (inner.get("results") or inner.get("path")):
        structured = inner

    docs: List[Dict[str, Any]] = []

    for hit in structured.get("results") or []:
        if isinstance(hit, str) and hit.strip():
            # A hit the tool-boundary guard withheld whole: Caber replaced the
            # entire result object with its marker string (the duplicate and
            # stale guards drop at item granularity, json_path `results.N`).
            # Skipping it would make the withholding invisible on the page.
            docs.append({
                "page_content": hit,
                "metadata": {
                    "path": "withheld",
                    "snippets": [hit],
                    "source": tool_name,
                    "bm25_score": 0.0,
                },
            })
            continue
        if not isinstance(hit, dict):
            continue
        candidates = [
            "\n".join(hit.get("snippets") or []),
            hit.get("preview") or "",
            "\n".join(hit.get("matches") or []),
        ]
        content = next((c for c in candidates if c and c.strip()), "No content available")
        docs.append({
            "page_content": content,
            "metadata": {
                "path": hit.get("path", "Unknown"),
                "snippets": hit.get("snippets", []),
                "source": tool_name,
                "bm25_score": hit.get("score", 0.0),
            },
        })

    raw_content = (
        structured["content"]
        if isinstance(structured.get("content"), str)
        else (fallback_text if isinstance(fallback_text, str) and fallback_text.strip() else None)
    )
    if not docs and structured.get("path") and isinstance(raw_content, str):
        content = raw_content.strip() or "No content available"
        docs.append({
            "page_content": content,
            "metadata": {
                "path": structured["path"],
                "snippets": [content[:400]],
                "source": tool_name,
                "bm25_score": 0.0,
            },
        })

    return docs


def extract_tool_calls(messages: List[Any]) -> List[ToolCall]:
    """Build the ``tool_call_log`` by pairing AI tool-call requests with results.

    Pairing is by ``tool_call_id`` rather than by order: a single assistant turn
    can request several tools, and their ToolMessages may come back in any order.
    """
    requested: Dict[str, Dict[str, Any]] = {}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if isinstance(call, dict):
                call_id = call.get("id") or ""
                requested[call_id] = {
                    "tool": call.get("name", "unknown"),
                    "args": call.get("args") or {},
                }

    log: List[ToolCall] = []
    for message in messages:
        if getattr(message, "type", "") != "tool" and message.__class__.__name__ != "ToolMessage":
            continue
        call_id = getattr(message, "tool_call_id", "") or ""
        meta = requested.get(call_id, {})
        name = meta.get("tool") or getattr(message, "name", "") or "unknown"
        structured = _structured_from_tool_message(message)
        text = _first_text(getattr(message, "content", ""))
        args = meta.get("args", {})
        is_error = bool(getattr(message, "status", "") == "error"
                        or (isinstance(structured, dict) and structured.get("isError")))

        source_paths = _source_paths_from(structured)
        if not source_paths and not is_error:
            # Fallback: a path-taking tool tells us its source in its own
            # arguments. `read_file(path=...)` retrieved exactly that document, so
            # a missing structured payload should not silently cost us the
            # citation -- provenance failing quietly is worse than provenance
            # being derived from the request.
            # NB: a distinct name. This used to be called `requested`, which
            # shadowed the tool-call lookup dict built above -- so as soon as one
            # tool produced no source paths, every LATER call in the same run
            # crashed on `requested.get(...)`. Caught by the two-tool-call test.
            requested_path = args.get("path")
            # The fallback must honour the SAME document filter as the payload
            # path, or it reintroduces exactly what that filter removes. A live
            # run reported '/contracts/' as a retrieved source: `list_files` takes
            # a directory in `path`, produces no source paths of its own, and so
            # fell straight through to here, where the directory was adopted
            # unchecked.
            if (isinstance(requested_path, str) and requested_path
                    and _is_document_path(requested_path)):
                source_paths = [requested_path]
                logger.debug(
                    "[AGENT] %s: no source path in the tool payload; using the "
                    "requested path %r", name, requested_path,
                )

        log.append(ToolCall(
            tool=name,
            args=args,
            result_preview=text[:500],
            source_paths=source_paths,
            is_error=is_error,
        ))
    return log


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def _connection_config(tenant_id: str,
                       extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """MCP connection config for langchain-mcp-adapters.

    ``streamable_http`` rather than stdio: the demo server is a long-lived
    container with a MinIO-backed corpus and a background BM25 builder, and stdio
    would require launching it as a subprocess. HTTP fully supports "an LLM
    queries this server directly" -- the claim was previously false because of
    the Tier-A protocol gaps, not because stdio was missing.

    The acting user is not a header here: ``pycaber.auto`` sends the user that
    ``set_ws_user`` pinned on every MCP request as baggage ``caber.user``.
    """
    headers = {
        # The MCP server scopes data access by tenant, as a second line of
        # defence independent of the OBO decision.
        "X-Caber-Tenant": tenant_id,
    }
    if extra_headers:
        headers.update(extra_headers)
    return {
        "caber-docs": {
            "transport": "streamable_http",
            "url": f"{MCP_SERVER_URL.rstrip('/')}/mcp",
            "headers": headers,
        }
    }


def _usage_from_llm_result(result: Any) -> Optional[tuple]:
    """``(input_tokens, output_tokens)`` from a LangChain ``LLMResult``, or None.

    ``ChatOpenAI`` puts the provider's usage on the generated message's
    ``usage_metadata``; older paths only fill ``llm_output["token_usage"]``.
    """
    for generations in getattr(result, "generations", None) or []:
        for generation in generations or []:
            meta = getattr(getattr(generation, "message", None), "usage_metadata", None)
            if meta:
                return int(meta.get("input_tokens") or 0), int(meta.get("output_tokens") or 0)
    token_usage = (getattr(result, "llm_output", None) or {}).get("token_usage") or {}
    if token_usage:
        return (int(token_usage.get("prompt_tokens") or 0),
                int(token_usage.get("completion_tokens") or 0))
    return None


def _usage_callback(model_name: str) -> Any:
    """A LangChain callback that records one usage line per chat request.

    The ReAct agent makes one chat request per reasoning step, so recording on
    ``on_llm_end`` counts every request, including those of a turn that later
    hits the recursion bound (whose messages are discarded below).
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class _UsageRecorder(BaseCallbackHandler):
        def on_llm_end(self, response: Any, **kwargs: Any) -> None:
            # Opt-in: only a compose file that sets CABER_DEMO_USAGE_DIR
            # (agentic-tools) records; other demos sharing this image do nothing.
            if not llm_usage.recording_enabled():
                return
            tokens = _usage_from_llm_result(response)
            llm_usage.record_or_warn(
                logger, "agentic-app", llm_usage.selected_chat_provider(), "chat", model_name,
                tokens[0] if tokens else 0, tokens[1] if tokens else 0,
                usage_missing=tokens is None,
            )

    return _UsageRecorder()


async def _run_agent_async(question: str, *, tenant_id: str,
                           tenant_groups: Optional[List[str]] = None,
                           extra_headers: Optional[Dict[str, str]] = None,
                           model: Optional[str] = None,
                           max_iterations: Optional[int] = None) -> AgentRun:
    # Imported here, not at module scope, so that `import agent_loop` cannot
    # accidentally pull langchain_core in ahead of pycaber.auto.
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.runnables import RunnableLambda
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_openai import ChatOpenAI
    from openai import DefaultAsyncHttpxClient
    # `create_react_agent` emits a LangGraphDeprecatedSinceV10 warning: it moved to
    # `langchain.agents.create_agent`, deprecated in LangGraph 1.0 for removal in
    # 2.0. Kept deliberately, not overlooked. Ticket 09 chose it, the ticket-09
    # redaction spike verified redaction survives THIS path (specifically
    # `ToolNode` invoking with a `ToolCall` and getting a `ToolMessage` back), and
    # that spike is the only evidence the demo's central claim holds. Switching to
    # an unverified constructor would discard that evidence and add the `langchain`
    # package as a new dependency. Migrate deliberately, with a re-run of the
    # spike, before LangGraph 2.0 -- not as a drive-by warning fix.
    from langgraph.prebuilt import create_react_agent

    model_name = model or AGENT_MODEL
    limit = max_iterations or AGENT_MAX_ITERATIONS

    client = MultiServerMCPClient(
        _connection_config(tenant_id, extra_headers)
    )
    tools = await client.get_tools()
    tool_names = [t.name for t in tools]
    logger.info("[AGENT] MCP catalog: %s", tool_names)
    if not tools:
        # Fail loudly. An agent with no tools will happily answer from parametric
        # memory, which is the opposite of retrieval qualification.
        raise RuntimeError(
            f"the MCP server at {MCP_SERVER_URL} advertised no tools -- refusing to "
            "run an agent that would answer without retrieving anything"
        )

    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    llm_kwargs: Dict[str, Any] = {
        "model": model_name,
    }
    provider = (os.getenv("SELECTED_LLM_PROVIDER") or "").strip().upper()
    lower_model = model_name.lower()
    if os.getenv("AGENT_TEMPERATURE") or not (
        provider == "ANTHROPIC"
        or "anthropic.com" in base_url.lower()
        or lower_model.startswith(("claude-", "o1", "o3", "o4", "gpt-5", "gpt-6"))
    ):
        llm_kwargs["temperature"] = AGENT_TEMPERATURE
    if lower_model.startswith(("gpt-5", "gpt-6")):
        llm_kwargs["use_responses_api"] = False
        llm_kwargs["reasoning_effort"] = "none"
    if base_url:
        llm_kwargs["base_url"] = base_url
    # One HTTP client per turn, owned by this turn's event loop. Without it,
    # ChatOpenAI falls back to langchain-openai's process-wide lru_cached
    # AsyncClient, whose pooled connections stay bound to the loop that opened
    # them. run_agent drives every turn with its own asyncio.run on the
    # session's script thread, so the next turn (or another user's concurrent
    # turn) picks up a connection from a closed loop and the OpenAI SDK reports
    # "Connection error." (cause: RuntimeError: Event loop is closed), or the
    # pool deadlocks once retries kick in.
    llm_http = DefaultAsyncHttpxClient()
    llm_kwargs["http_async_client"] = llm_http
    chat_model = ChatOpenAI(**llm_kwargs)
    tc_extra: Dict[str, Any] = {}
    orig_create = chat_model._create_chat_result
    orig_payload = chat_model._get_request_payload

    def _create_chat_result(response: Any, generation_info: Optional[Dict[str, Any]] = None) -> Any:
        res_dict = (
            response
            if isinstance(response, dict)
            else (response.model_dump(warnings=False) if hasattr(response, "model_dump") else {})
        )
        if isinstance(res_dict, dict):
            for choice in res_dict.get("choices") or []:
                msg = (choice or {}).get("message") or {}
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id") and "extra_content" in tc:
                        tc_extra[tc["id"]] = tc["extra_content"]
        return orig_create(response, generation_info=generation_info)

    def _get_request_payload(input_: Any, *, stop: Optional[List[str]] = None, **kwargs: Any) -> Dict[str, Any]:
        payload = orig_payload(input_, stop=stop, **kwargs)
        for msg in payload.get("messages") or []:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id") in tc_extra:
                        tc["extra_content"] = tc_extra[tc["id"]]
        return payload

    object.__setattr__(chat_model, "_create_chat_result", _create_chat_result)
    object.__setattr__(chat_model, "_get_request_payload", _get_request_payload)
    q_lower = question.lower()
    wants_no_tools = (
        "without using any tools" in q_lower
        or "summarize this excerpt:" in q_lower
    )
    wants_save = "save_file" in q_lower
    wants_list = (
        "list_files" in q_lower
        or "list files" in q_lower
        or "list the files" in q_lower
    )
    initial_tools = (
        list(tools)
        if wants_list
        else ([t for t in tools if t.name != "list_files"] or list(tools))
    )
    bound_all = chat_model.bind_tools(list(tools))
    bound_initial = chat_model.bind_tools(initial_tools)
    save_tools = [t for t in tools if t.name == "save_file"]
    bound_save = chat_model.bind_tools(save_tools) if save_tools else chat_model

    orig_agenerate = chat_model._agenerate

    async def _agenerate(messages_in: List[Any], stop: Optional[List[str]] = None,
                         run_manager: Any = None, **kwargs: Any) -> Any:
        cur_messages = list(messages_in)
        cur_kwargs = dict(kwargs)
        for attempt in range(1, 7):
            try:
                return await orig_agenerate(cur_messages, stop=stop, run_manager=run_manager, **cur_kwargs)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if ("429" in msg or "rate_limit" in msg.lower() or "rate limit" in msg.lower()) and attempt < 6:
                    await asyncio.sleep(2.5 * attempt)
                    continue
                if ("tool_use_failed" in msg or "Tool choice is none" in msg) and attempt < 4:
                    if attempt == 1 and "tools" not in cur_kwargs:
                        cur_messages = list(messages_in) + [
                            HumanMessage(
                                content="Do not call any tools. Write your final answer in plain text "
                                        "using only the tool output already above."
                            )
                        ]
                        continue
                    all_bound = getattr(bound_all, "kwargs", {}).get("tools")
                    if all_bound:
                        cur_kwargs["tools"] = all_bound
                        cur_messages = list(messages_in)
                        continue
                raise

    object.__setattr__(chat_model, "_agenerate", _agenerate)

    def _finish_after_save(msgs_or_state: Any) -> AIMessage:
        msgs = (
            msgs_or_state.get("messages", [])
            if isinstance(msgs_or_state, dict)
            else (
                getattr(msgs_or_state, "messages", None)
                if hasattr(msgs_or_state, "messages")
                else list(msgs_or_state or [])
            )
        )
        saved_text = ""
        for m in reversed(msgs):
            for tc in getattr(m, "tool_calls", None) or []:
                if isinstance(tc, dict) and tc.get("name") == "save_file":
                    saved_text = str((tc.get("args") or {}).get("content") or "").strip()
                    if saved_text:
                        break
            if saved_text:
                break
        prior_calls = extract_tool_calls(msgs)
        src_paths = []
        for c in prior_calls:
            for p in c.source_paths:
                if p not in src_paths:
                    src_paths.append(p)
        cite = "\n".join(f"- {p}" for p in src_paths) if src_paths else ""
        body = saved_text or "Saved summary."
        if cite and "Sources:" not in body:
            body = f"{body}\n\nSources:\n{cite}"
        return AIMessage(content=body)

    post_save_model = RunnableLambda(_finish_after_save)

    def _select_model(state: Any, _runtime: Any = None) -> Any:
        if wants_no_tools:
            return chat_model
        msgs = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        seen_tools = {
            getattr(m, "name", "")
            for m in msgs
            if getattr(m, "type", "") == "tool" or m.__class__.__name__ == "ToolMessage"
        }
        has_retrieved = bool(
            seen_tools & {"read_file", "search_files_bm25", "search_semantic", "list_files"}
        )
        if not has_retrieved:
            return bound_initial
        if wants_save and save_tools:
            if "save_file" not in seen_tools:
                return bound_save
            return post_save_model
        return chat_model

    agent = create_react_agent(
        _select_model,
        tools,
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]

    hit_limit = False
    try:
        # LangGraph counts one "step" per node visit, so a tool-using turn costs
        # roughly two. Doubling keeps `limit` meaning "tool iterations", which is
        # what the old loop's cap meant.
        state = await agent.ainvoke(
            {"messages": messages},
            config={
                "recursion_limit": max(2, limit * 2),
                "callbacks": [_usage_callback(model_name)],
            },
        )
        final_messages = state.get("messages", [])
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ != "GraphRecursionError":
            raise
        # Graceful termination, matching the salvaged pattern: surface whatever
        # the agent had rather than erroring out mid-video.
        logger.warning("[AGENT] hit the %d-iteration bound: %s", limit, exc)
        hit_limit = True
        final_messages = list(messages)
    finally:
        await llm_http.aclose()

    tool_calls = extract_tool_calls(final_messages)

    answer = ""
    for message in reversed(final_messages):
        if getattr(message, "type", "") == "ai" or message.__class__.__name__ == "AIMessage":
            text = _first_text(getattr(message, "content", ""))
            if text.strip():
                answer = text
                break
    if hit_limit:
        answer = MAX_ITERATIONS_PREFIX + (answer or "no answer was produced before the bound")
    if not answer:
        answer = "The agent produced no answer."

    documents: List[Dict[str, Any]] = []
    last_structured: Optional[Dict[str, Any]] = None
    for message in final_messages:
        if getattr(message, "type", "") != "tool" and message.__class__.__name__ != "ToolMessage":
            continue
        structured = _structured_from_tool_message(message)
        if structured is not None:
            last_structured = structured
        name = getattr(message, "name", "") or "mcp_tool"
        fallback_text = _first_text(getattr(message, "content", ""))
        documents.extend(_documents_from(structured, name, fallback_text=fallback_text))

    logger.info(
        "[AGENT] done: model=%s tools_called=%d distinct_sources=%d hit_limit=%s",
        model_name, len(tool_calls),
        len({p for c in tool_calls for p in c.source_paths}), hit_limit,
    )

    return AgentRun(
        answer=answer,
        tool_call_log=tool_calls,
        documents=documents,
        last_tool_result=last_structured,
        hit_iteration_limit=hit_limit,
        tool_names_available=tool_names,
    )


def run_agent(question: str, *, tenant_id: str,
              tenant_groups: Optional[List[str]] = None,
              extra_headers: Optional[Dict[str, str]] = None,
              model: Optional[str] = None,
              max_iterations: Optional[int] = None) -> AgentRun:
    """Run one agent turn. Synchronous wrapper for Streamlit and the CI driver.

    The loop is async (the MCP client and LangGraph both are); Streamlit's script
    runner is not. ``asyncio.run`` is correct here because each Streamlit rerun is
    a fresh synchronous call with no ambient event loop.
    """
    coro = _run_agent_async(
        question,
        tenant_id=tenant_id,
        tenant_groups=tenant_groups,
        extra_headers=extra_headers,
        model=model,
        max_iterations=max_iterations,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # An event loop is already running (some pytest contexts; a Streamlit
    # script thread runs none). Run the coroutine on a dedicated loop in a
    # worker thread rather than failing. The worker runs in a copy of this
    # thread's contextvars, which is how PyCaber's context reaches it: a new
    # thread does not inherit them.
    import concurrent.futures
    import contextvars

    ctx = contextvars.copy_context()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(ctx.run, asyncio.run, coro).result()
