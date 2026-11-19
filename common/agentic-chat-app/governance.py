"""Consume the governance decision PyCaber projects onto an MCP tool result.

PyCaber's MCP interception (Task 7) projects a governance decision as a
*successful* ``CallToolResult`` (``isError: false``) whose ``content`` carries
the exact policy-authored text and whose ``_meta["ai.caber/governance"]`` carries
the structured decision::

    {
      "state": "clarification_required" | "blocked" | "allowed",
      "decision_id": "<id>",
      "questions": [{"question_id", "text", "reason_code"}...],
      "target_references": ["edge-..."]
    }

The agentic app only *reads* this projection. It never re-implements the purpose
taxonomy or re-decides enforcement -- the server side already decided. On a
clarification it skips the LLM call, renders the exact text, registers the
response for the OBO event, and treats the user's next answer as a new request.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

GOVERNANCE_META_KEY = "ai.caber/governance"

# Header that carries the classified purpose context on the outbound governed
# request so the server-side governor can consume it.
PURPOSE_HEADER = "X-Caber-Purpose-Context"

STATE_CLARIFICATION = "clarification_required"
STATE_BLOCKED = "blocked"
STATE_ALLOWED = "allowed"


def extract_governance(mcp_result: Optional[dict]) -> Optional[dict]:
    """Return the projected governance decision, or ``None`` for a plain result."""
    meta = (mcp_result or {}).get("_meta") or {}
    gov = meta.get(GOVERNANCE_META_KEY)
    return gov if isinstance(gov, dict) else None


def governance_state(mcp_result: Optional[dict]) -> Optional[str]:
    gov = extract_governance(mcp_result)
    return gov.get("state") if gov else None


def clarification_text(mcp_result: Optional[dict]) -> str:
    """Join the ``content`` text blocks verbatim -- no rewriting or trimming."""
    blocks = (mcp_result or {}).get("content") or []
    return "\n".join(
        b.get("text", "") for b in blocks
        if isinstance(b, dict) and b.get("type") == "text"
    )


def plan_action(mcp_result: Optional[dict]) -> dict:
    """Decide what the app should do with a governed tool result.

    The projected decision is authoritative: the server (not the app) decided.
    A clarification or block is honoured regardless of any purpose the session
    already declared -- the app never overrides the server by re-deciding.

    Returns a plan dict with:
      * ``action``: ``"answer" | "clarify" | "block"``
      * ``skip_llm``: whether to bypass the LLM
      * ``render_text``: exact text to render (``None`` when answering)
      * ``register_response``: whether the rendered text is the OBO response
      * ``treat_answer_as_new_request``: whether the user's next turn is a fresh OBO root
      * ``decision_id`` / ``governance``: passthrough of the projected decision
    """
    gov = extract_governance(mcp_result)
    state = gov.get("state") if gov else None

    if state == STATE_CLARIFICATION:
        return {
            "action": "clarify",
            "skip_llm": True,
            "render_text": clarification_text(mcp_result),
            "register_response": True,
            "treat_answer_as_new_request": True,
            "decision_id": gov.get("decision_id"),
            "governance": gov,
        }
    if state == STATE_BLOCKED:
        return {
            "action": "block",
            "skip_llm": True,
            "render_text": clarification_text(mcp_result),
            "register_response": True,
            "treat_answer_as_new_request": False,
            "decision_id": gov.get("decision_id"),
            "governance": gov,
        }
    # Allowed / no projection: proceed normally to the LLM.
    return {
        "action": "answer",
        "skip_llm": False,
        "render_text": None,
        "register_response": False,
        "treat_answer_as_new_request": False,
        "decision_id": gov.get("decision_id") if gov else None,
        "governance": gov,
    }


def attach_purpose_context(base_context: dict, purpose_ctx: Any) -> dict:
    """Attach the classified purpose context WITHOUT re-deciding enforcement.

    The purpose label passes through unchanged (the app does not re-map the
    taxonomy); provenance marks it as ``classified`` so the server can tell it
    apart from a user-declared purpose. No ``access``/``decision`` field is
    authored client-side.
    """
    merged = dict(base_context)
    merged["purpose"] = purpose_ctx.as_context()
    return merged


def purpose_request_headers(attached_context: Optional[dict]) -> dict:
    """Build outbound headers that carry the attached purpose context.

    Returns ``{}`` when there is no purpose to attach so callers can splat it
    unconditionally. The header value is the classified purpose block (label +
    evidence), JSON-encoded; the server-side governor reads it -- the app does
    not re-decide anything.
    """
    if not attached_context:
        return {}
    purpose = attached_context.get("purpose")
    if not purpose:
        return {}
    return {PURPOSE_HEADER: json.dumps(purpose, sort_keys=True)}


class ClarificationSession:
    """Track the clarification lifecycle for OBO-correct request/response pairing.

    When a clarification is rendered, its text is the OBO *response* for the
    just-completed interaction. The user's subsequent answer is a brand-new
    request (a fresh OBO root), not a continuation of the prior interaction.
    """

    def __init__(self, register_request: Callable, register_response: Callable,
                 api: str = "localhost:8083"):
        self._register_request = register_request
        self._register_response = register_response
        self._api = api
        self.awaiting_answer = False
        self.last_decision_id: Optional[str] = None

    def on_tool_result(self, mcp_result: Optional[dict], interaction=None) -> dict:
        """Apply the projected decision. Registers the rendered text as the OBO
        response for clarifications/blocks. Returns the plan."""
        plan = plan_action(mcp_result)
        if plan["register_response"] and plan["render_text"] is not None:
            self._register_response(plan["render_text"], interaction=interaction)
        if plan["treat_answer_as_new_request"]:
            self.awaiting_answer = True
            self.last_decision_id = plan.get("decision_id")
        return plan

    def consume_user_message(self, text: str):
        """Register the user's message as a NEW request (fresh OBO root)."""
        self.awaiting_answer = False
        return self._register_request(text, api=self._api)
