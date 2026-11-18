# stdlib only; tested by tests/unit/test_policy_plan_and_mode.py
"""Activate a demo's policies through the Dashboard, the locked counterpart of
``runner.core.activate_policies`` (which needs ``requests`` and so cannot ship).

The protocol is that module's, call for call:

  1. ``POST /v1/policies/import-config`` imports the config's policies and
     activates them (the Dashboard blocks until the runtimes adopt the digest).
  2. With flips: ``GET /v1/policies``, flip ``enabled`` on exactly the named
     ids, ``PUT /v1/policies`` with ``If-Match: <concurrency_token>``, then
     ``POST /v1/policies/activate`` for the new revision.
  3. ``GET /v1/policies/status`` until that revision is active with a digest.
  4. ``GET /v1/storage/customer/governance/current/manifest.json`` and require
     its ``policy_digest`` to equal the active one.

Every call sends ``Authorization: Bearer <key>``, the form the Dashboard's
``/v1`` gate prefers (``runner/core/dashboard_auth.py``). Nothing is retried
except a 409 ``activation_in_progress``, once, after the pending revision
resolves. Every failure is :class:`ActivationFailed` with one reason.
"""

import base64
import copy
import json
import time

TIMEOUT_S = 180.0
POLL_INTERVAL_S = 2.0
# import-config and activate block for the Dashboard's adoption deadline
# (up to about 120 s), so one request must be allowed to outlast it.
REQUEST_TIMEOUT_S = 150.0

GSS_CURRENT_MANIFEST = "customer/governance/current/manifest.json"


class ActivationFailed(Exception):
    """Activation did not reach a verified active state; the message says why."""


class _Client:
    """The Dashboard ``/v1/policies`` calls, with one deadline for every wait."""

    def __init__(self, dashboard_url: str, api_key: str, http, sleep, clock):
        self.base = dashboard_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.http = http
        self.sleep = sleep
        self.clock = clock
        self.deadline = clock() + TIMEOUT_S

    def _get(self, path: str) -> dict:
        try:
            return self.http.get_json(self.base + path, headers=self.headers,
                                      timeout=REQUEST_TIMEOUT_S)
        except self.http.HttpError as exc:
            raise ActivationFailed(f"GET {path} failed: {exc}") from exc

    def status(self) -> dict:
        body = self._get("/v1/policies/status")
        if not isinstance(body, dict):
            raise ActivationFailed("GET /v1/policies/status returned a non-object body")
        return body

    def wait_no_pending(self) -> None:
        while True:
            if self.status().get("pending") is None:
                return
            if self.clock() >= self.deadline:
                raise ActivationFailed("an earlier activation is still pending")
            self.sleep(POLL_INTERVAL_S)

    def activation_call(self, path: str, body: dict) -> dict:
        """POST an activating call; one retry after a 409 ``activation_in_progress``."""
        for attempt in (1, 2):
            try:
                payload = self.http.post_json(self.base + path, body, headers=self.headers,
                                              timeout=REQUEST_TIMEOUT_S)
            except self.http.HttpError as exc:
                detail = exc.json()
                if exc.status == 502 and isinstance(detail, dict):
                    raise ActivationFailed(
                        f"{path}: revision {detail.get('revision_id')} failed: "
                        f"{detail.get('reason') or exc}") from exc
                if exc.status == 409 and isinstance(detail, dict) \
                        and detail.get("error") == "activation_in_progress" and attempt == 1:
                    self.wait_no_pending()
                    continue
                raise ActivationFailed(str(exc)) from exc
            if isinstance(payload, dict) and payload.get("state") == "active":
                return payload
            state = payload.get("state") if isinstance(payload, dict) else None
            raise ActivationFailed(f"POST {path} did not report state active (state={state!r})")
        raise AssertionError("unreachable")

    def flip(self, enable: tuple, disable: tuple) -> str:
        """Flip ``enabled`` on exactly the named ids, save, and return the new revision id."""
        body = self._get("/v1/policies")
        policy_set = copy.deepcopy(body.get("policy_set"))
        token = body.get("concurrency_token")
        if not isinstance(policy_set, dict) or not token:
            raise ActivationFailed(
                "GET /v1/policies returned no policy_set or concurrency_token to edit")
        by_id = {p.get("id"): p for p in policy_set.get("policies") or [] if isinstance(p, dict)}
        missing = {pid for pid in enable + disable if pid not in by_id}
        if missing:
            raise ActivationFailed(f"unknown policy ids: {sorted(missing)}")
        for pid in enable:
            by_id[pid]["enabled"] = True
        for pid in disable:
            by_id[pid]["enabled"] = False
        headers = dict(self.headers)
        headers["If-Match"] = str(token)
        try:
            saved = self.http.put_json(self.base + "/v1/policies", {"policy_set": policy_set},
                                       headers=headers, timeout=REQUEST_TIMEOUT_S)
        except self.http.HttpError as exc:
            raise ActivationFailed(f"PUT /v1/policies failed: {exc}") from exc
        revision_id = saved.get("revision_id") if isinstance(saved, dict) else None
        if not revision_id:
            raise ActivationFailed("PUT /v1/policies returned no revision_id")
        return revision_id

    def wait_active(self, revision_id: str) -> dict:
        """Wait until ``revision_id`` is active with a digest; ``pending`` is ignored, as upstream."""
        while True:
            active = self.status().get("active")
            if isinstance(active, dict) and active.get("revision_id") == revision_id \
                    and active.get("policy_digest"):
                return active
            if self.clock() >= self.deadline:
                raise ActivationFailed(
                    f"revision {revision_id} did not become active within {TIMEOUT_S:.0f}s")
            self.sleep(POLL_INTERVAL_S)

    def manifest_digest(self):
        body = self._get(f"/v1/storage/{GSS_CURRENT_MANIFEST}")
        try:
            manifest = json.loads(base64.b64decode(body["data"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ActivationFailed(
                f"GSS {GSS_CURRENT_MANIFEST} is not a readable manifest: {exc}") from exc
        return manifest.get("policy_digest") if isinstance(manifest, dict) else None


def activate(dashboard_url: str, api_key: str, plan, http, *, sleep=time.sleep,
             clock=time.monotonic) -> None:
    """Activate the config's policy set with ``plan``'s flips and verify the stored manifest.

    ``http`` is :mod:`runner.locked.http` or a stand-in with its ``get_json``,
    ``post_json``, ``put_json`` and ``HttpError``. Raises :class:`ActivationFailed`.
    """
    enable, disable = tuple(plan.enable), tuple(plan.disable)
    both = sorted(set(enable) & set(disable))
    if both:
        raise ActivationFailed(f"policy ids in both enable and disable: {both}")
    client = _Client(dashboard_url, api_key, http, sleep, clock)
    result = client.activation_call("/v1/policies/import-config", {})
    revision_id = result.get("revision_id")
    if enable or disable:
        new_revision = client.flip(enable, disable)
        result = client.activation_call("/v1/policies/activate", {"revision_id": new_revision})
        revision_id = result.get("revision_id") or new_revision
    if not revision_id:
        raise ActivationFailed("POST /v1/policies/import-config returned no revision_id")
    active = client.wait_active(revision_id)
    if client.manifest_digest() != active["policy_digest"]:
        raise ActivationFailed("stored manifest digest does not match the active revision")
