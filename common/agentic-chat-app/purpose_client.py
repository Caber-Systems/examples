"""Client-side purpose classification via the supported GLiNER2 model server.

The demo trusts the client deployment boundary for purpose classification (MVP;
server/MCP-side classification is deferred hardening). The agentic app calls the
supported PyCaber GLiNER2 model server (``gliner2_model_server`` served by
``scripts/parser_service.py`` on the host, reachable at ``parser-service:8765``
via ``/gliner-2``) to classify the user's prompt into the governance purpose
taxonomy, and attaches the classified purpose plus its *evidence* (model id,
taxonomy revision, confidence, prompt digest) to the request.

Wire contract (the exact protocol the model server implements, matching the
``gliner2`` package's own ``GLiNER2API`` client -- see PyCaber
``crates/document_parser/orchestrator/python_scripts/gliner2_model_server.py``):

  POST /gliner-2
    headers: ``X-API-Key: <CSI_GLINER2_MODEL_API_KEY>``,
             ``User-Agent: CSI.CaberCore/0.1.0 (purpose-classifier)``
    body: {
      "task": "schema",
      "text": <prompt>,
      "schema": {"classifications": {"purpose": {"labels": [...], "multi_label": false}}},
      "threshold": 0.5, "include_confidence": true,
      "include_spans": false, "format_results": true
    }
  -> 200 {"result": [{"purpose": {"label": "<class>", "confidence": <float>}}]}

There is deliberately NO keyword fallback and NO LLM fallback: if the classifier
is unavailable, malformed, or unauthenticated, the call raises
:class:`PurposeUnavailable`. A fabricated classification would poison the
governance decision, so a clear failure is required instead of a guess.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)

# Set after the first "no classifier URL" log line, so a deployment without a
# classifier (the agentic-tools base stack, which ships without NER) says so
# once instead of on every prompt.
_unset_url_logged = False


# The caber-purpose-v1 taxonomy: the HIPAA-slice governance purpose classes
# (compliance_policy_enforcement_SPEC.md "Purpose taxonomy (HIPAA slice)":
# treatment / payment / operations / marketing) plus the explicit neutral class
# for prompts that carry no governance-relevant purpose. These are the candidate
# labels handed to the zero-shot classifier; the server picks one. The client
# does NOT re-decide enforcement -- it only supplies the taxonomy the revision
# names and passes the chosen label through as classifier-provenance evidence.
PURPOSE_TAXONOMIES = {
    "caber-purpose-v1": (
        "treatment",
        "payment",
        "operations",
        "marketing",
        "neutral",
    ),
}

# The classification task name under which the server keys the result.
_TASK_NAME = "purpose"

# The classifier call is Caber's own control traffic, not the agent's. The
# `CSI.` prefix is the one PyCaber's intercept treats as Caber-internal
# (base_utils CABER_USER_AGENT_PREFIXES), so the call is skipped rather than
# recorded and discovered as an application API. The version mirrors the
# `RecordModifiedBy: CSI.CaberCore/0.1.0 (...)` stamp; this app has no version
# constant of its own.
USER_AGENT = "CSI.CaberCore/0.1.0 (purpose-classifier)"


class PurposeUnavailable(RuntimeError):
    """Raised when the GLiNER2 classifier cannot produce a classification.

    Never swallowed into a fabricated purpose -- the caller must handle the
    failure explicitly (e.g. surface it, not silently proceed unclassified).
    """


@dataclass
class PurposeContext:
    """Classified purpose plus the evidence proving how it was derived."""

    purpose: str
    model_id: str
    taxonomy_revision: str
    confidence: float
    prompt_digest: str
    source_url: str

    def as_context(self) -> dict:
        """Governance-request purpose block, marked as classifier-provenance."""
        return {
            "purpose": self.purpose,
            "provenance": "classified",
            "confidence": self.confidence,
            "model_id": self.model_id,
            "taxonomy_revision": self.taxonomy_revision,
            "prompt_digest": self.prompt_digest,
            "classifier_url": self.source_url,
        }


def _default_post(url, json, headers, timeout):  # pragma: no cover - thin httpx shim
    import httpx

    return httpx.post(url, json=json, headers=headers, timeout=timeout)


def classify_purpose(
    prompt: str,
    *,
    url: Optional[str],
    model_id: str,
    taxonomy_revision: str,
    api_key: str = "",
    timeout: float = 10.0,
    labels: Optional[Sequence[str]] = None,
    http_post: Optional[Callable] = None,
) -> Optional[PurposeContext]:
    """Classify ``prompt`` into the governance purpose taxonomy via GLiNER2.

    Speaks the gliner2 model-server ``/gliner-2`` schema-classification contract
    (single-label over the ``taxonomy_revision`` label set). Retains full
    classifier evidence. Raises :class:`PurposeUnavailable` on any transport
    error, non-2xx status, unknown taxonomy, or malformed/absent classification
    in the response -- never a keyword or LLM fallback.

    An empty or ``None`` ``url`` means no classifier is deployed: the call
    returns ``None`` without contacting anything, never raises and never
    retries, and logs that once per process. That is a deployment choice, not
    a failure; the request proceeds unclassified, with no invented purpose.
    """
    global _unset_url_logged
    if not url:
        if not _unset_url_logged:
            _unset_url_logged = True
            logger.info(
                "No purpose classifier URL is configured (PURPOSE_CLASSIFIER_URL "
                "unset); prompts are not purpose-classified."
            )
        return None

    post = http_post or _default_post
    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    resolved_labels = labels if labels is not None else PURPOSE_TAXONOMIES.get(taxonomy_revision)
    if not resolved_labels:
        # Never invent a taxonomy: an unknown revision is a hard failure.
        raise PurposeUnavailable(
            f"no purpose taxonomy defined for revision {taxonomy_revision!r}"
        )

    body = {
        "task": "schema",
        "text": prompt,
        "schema": {
            "classifications": {
                _TASK_NAME: {"labels": list(resolved_labels), "multi_label": False}
            }
        },
        "threshold": 0.5,
        "include_confidence": True,
        "include_spans": False,
        "format_results": True,
    }
    headers = {"User-Agent": USER_AGENT}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        resp = post(url, json=body, headers=headers, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 -- any transport failure is fatal here
        raise PurposeUnavailable(f"purpose classifier unreachable: {exc}") from exc

    status = getattr(resp, "status_code", None)
    if status is None or not (200 <= status < 300):
        raise PurposeUnavailable(f"purpose classifier returned status {status}")

    try:
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise PurposeUnavailable(f"purpose classifier returned non-JSON: {exc}") from exc

    # The model server returns {"result": [ {<task>: {"label", "confidence"}} ]}
    # (a batch list, one entry per input text). Unwrap defensively without
    # guessing a shape: any deviation is a hard failure.
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, list) or not result:
        raise PurposeUnavailable("purpose classifier returned no result list")

    first = result[0]
    if not isinstance(first, dict):
        raise PurposeUnavailable("purpose classifier result entry is not an object")

    classification = first.get(_TASK_NAME)
    if isinstance(classification, dict):
        purpose = classification.get("label")
        confidence_raw = classification.get("confidence")
    else:
        # include_confidence=True guarantees the {"label","confidence"} object;
        # anything else (bare string, missing) is a protocol violation, not a
        # value to guess a confidence for.
        raise PurposeUnavailable(
            "purpose classifier returned no confidence-bearing classification"
        )

    if not purpose:
        raise PurposeUnavailable("purpose classifier returned no purpose label")

    # Confidence is required evidence; a missing/invalid value is a hard failure,
    # not a defaulted guess.
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError) as exc:
        raise PurposeUnavailable("purpose classifier returned no confidence") from exc

    return PurposeContext(
        purpose=purpose,
        model_id=model_id,
        taxonomy_revision=taxonomy_revision,
        confidence=confidence,
        prompt_digest=prompt_digest,
        source_url=url,
    )
