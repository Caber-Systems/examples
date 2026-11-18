# stdlib only; tested by test_locked_package.py
"""Read and validate ``preview.lock.json``, the exported tree's pinned-release record."""

import json
from pathlib import Path

SUPPORTED_SCHEMA = 1

# Spec section 6 top-level keys, in spec order.
REQUIRED_KEYS = ("schema", "produced_at", "demo", "pycaber", "notice",
                 "images", "sources", "llm_usage", "image_sizes_compressed_bytes",
                 "files_sha256", "durations_s", "ci")


class LockError(Exception):
    """``preview.lock.json`` is absent, unreadable, or not a schema this demo.py reads."""


def load_lock(path: Path) -> dict:
    """Return the parsed lock at ``path``, or raise :class:`LockError` saying why not.

    The schema is checked before the keys, so a newer lock is refused for its
    schema rather than for keys this demo.py does not know about.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise LockError(f"preview.lock.json not found at {path}") from None
    except OSError as exc:
        raise LockError(f"preview.lock.json at {path} cannot be read: {exc}") from exc
    try:
        lock = json.loads(text)
    except ValueError as exc:
        raise LockError(f"preview.lock.json is not valid JSON: {exc}") from exc
    if not isinstance(lock, dict):
        raise LockError(
            f"preview.lock.json must hold a JSON object, not {type(lock).__name__}")
    schema = lock.get("schema")
    if schema != SUPPORTED_SCHEMA:
        raise LockError(
            f"preview.lock.json schema {schema} is not supported by this demo.py "
            f"(supports {SUPPORTED_SCHEMA})")
    missing = [key for key in REQUIRED_KEYS if key not in lock]
    if missing:
        raise LockError("preview.lock.json is missing keys: " + ", ".join(sorted(missing)))
    return lock
