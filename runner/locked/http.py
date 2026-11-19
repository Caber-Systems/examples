# stdlib only; tested by test_locked_package.py
"""JSON over HTTP with ``urllib.request`` (Plan Ruling 9).

Every failure, whether an HTTP status, an unreachable host or a body that is
not a JSON object, surfaces as one :class:`HttpError`. ``opener`` is
injectable so tests never open a socket.
"""

import json
import urllib.error
import urllib.request

_BODY_CHARS = 500


class HttpError(Exception):
    """An HTTP call failed; ``status`` is None when no response arrived.

    ``body`` is the whole response body; only the message is cut to
    ``_BODY_CHARS`` characters.
    """

    def __init__(self, message: str, *, status: "int | None", url: str, body: str = ""):
        super().__init__(message)
        self.status = status
        self.url = url
        self.body = body

    def json(self) -> "dict | None":
        """The body as a JSON object, or None when it is not one."""
        try:
            data = json.loads(self.body)
        except ValueError:
            return None
        return data if isinstance(data, dict) else None


def _decode(raw) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw or "")


def _send(request: urllib.request.Request, timeout: float, opener) -> dict:
    url = request.full_url
    method = request.get_method()
    try:
        with opener(request, timeout=timeout) as resp:
            status = getattr(resp, "status", None)
            text = _decode(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            body = _decode(exc.read())
        except (OSError, ValueError):
            body = ""
        raise HttpError(f"{method} {url} returned HTTP {exc.code}: {body[:_BODY_CHARS]}",
                        status=exc.code, url=url, body=body) from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        raise HttpError(f"{method} {url} failed: {reason}", status=None, url=url) from exc
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise HttpError(f"{method} {url} returned a non-JSON body: {text[:_BODY_CHARS]}",
                        status=status, url=url, body=text) from exc
    if not isinstance(payload, dict):
        raise HttpError(f"{method} {url} returned JSON that is not an object",
                        status=status, url=url, body=text)
    return payload


def get_json(url: str, headers: "dict | None" = None, timeout: float = 30, *,
             opener=urllib.request.urlopen) -> dict:
    """GET ``url`` and return its JSON object body."""
    request = urllib.request.Request(url, headers=dict(headers or {}), method="GET")
    return _send(request, timeout, opener)


def _json_request(method: str, url: str, body: dict, headers: "dict | None") -> urllib.request.Request:
    all_headers = dict(headers or {})
    all_headers["Content-Type"] = "application/json"
    return urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                  headers=all_headers, method=method)


def post_json(url: str, body: dict, headers: "dict | None" = None, timeout: float = 30, *,
              opener=urllib.request.urlopen) -> dict:
    """POST ``body`` as JSON to ``url`` and return its JSON object body."""
    return _send(_json_request("POST", url, body, headers), timeout, opener)


def put_json(url: str, body: dict, headers: "dict | None" = None, timeout: float = 30, *,
             opener=urllib.request.urlopen) -> dict:
    """PUT ``body`` as JSON to ``url`` and return its JSON object body."""
    return _send(_json_request("PUT", url, body, headers), timeout, opener)
