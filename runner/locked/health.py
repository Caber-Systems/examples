# stdlib only; tested by tests/unit/test_locked_build_run_down.py
"""Wait for a demo URL to answer, the health gate locked ``run`` applies per ``demo.json.health``.

Any answer below 500, a 4xx included, means the service is up (a login page
or a missing route still proves the server is serving). No answer, or a 5xx,
means not yet.
"""

import time
import urllib.error
import urllib.request

REQUEST_TIMEOUT_S = 10


def _answered(url: str, opener) -> bool:
    request = urllib.request.Request(url, method="GET")
    try:
        with opener(request, timeout=REQUEST_TIMEOUT_S) as resp:
            status = getattr(resp, "status", None)
    except urllib.error.HTTPError as exc:
        return exc.code < 500
    except (urllib.error.URLError, OSError, TimeoutError):
        return False
    return status is None or status < 500


def wait_for(url: str, timeout_s, *, opener=urllib.request.urlopen, sleep=time.sleep,
             clock=time.monotonic, interval_s=5) -> bool:
    """True once ``url`` answers below 500; False when ``timeout_s`` passes first."""
    deadline = clock() + timeout_s
    while True:
        if _answered(url, opener):
            return True
        if clock() >= deadline:
            return False
        sleep(interval_s)
