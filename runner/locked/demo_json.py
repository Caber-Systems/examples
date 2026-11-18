# stdlib only; tested by test_locked_package.py
"""Load and validate ``demos/<demo>/demo.json``, the per-demo descriptor (Plan Ruling 10).

Locked ``demo.py`` reads only this file to learn a demo's compose files,
service roles, ports, URLs, health checks and policy flips.
"""

import json
from pathlib import Path

SERVICE_ROLES = ("app", "mcp", "seed", "identity", "console", "dashboard", "init")


class DemoJsonError(Exception):
    """``demo.json`` is absent, unparsable, or has missing or mistyped keys."""


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_str_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _is_map(value, check_value) -> bool:
    return isinstance(value, dict) and all(
        isinstance(k, str) and check_value(v) for k, v in value.items())


def _services_ok(value) -> bool:
    return _is_map(value, lambda v: isinstance(v, str)) and all(
        k in SERVICE_ROLES for k in value)


def _policies_ok(value) -> bool:
    return (isinstance(value, dict)
            and _is_str_list(value.get("enable"))
            and _is_str_list(value.get("disable")))


def _health_ok(value) -> bool:
    return isinstance(value, list) and all(
        isinstance(h, dict) and isinstance(h.get("url"), str) and _is_int(h.get("timeout_s"))
        for h in value)


# key -> (check, expected shape named in the error)
_SCHEMA = (
    ("name", lambda v: isinstance(v, str), "str"),
    ("title", lambda v: isinstance(v, str), "str"),
    ("compose", _is_str_list, "list[str]"),
    ("platform_compose", _is_str_list, "list[str]"),
    ("services", _services_ok, "dict[str, str] with roles " + "/".join(SERVICE_ROLES)),
    ("ports", lambda v: _is_map(v, _is_int), "dict[str, int]"),
    ("urls", lambda v: _is_map(v, lambda u: isinstance(u, str)), "dict[str, str]"),
    ("what_to_try", _is_str_list, "list[str]"),
    ("policies", _policies_ok, "{enable: list[str], disable: list[str]}"),
    ("health", _health_ok, "list[{url: str, timeout_s: int}]"),
)


def load(path: Path) -> dict:
    """Return the descriptor at ``path``; one :class:`DemoJsonError` names every problem."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DemoJsonError(f"demo.json not found at {path}") from None
    except (OSError, ValueError) as exc:
        raise DemoJsonError(f"demo.json at {path} cannot be read: {exc}") from exc
    if not isinstance(data, dict):
        raise DemoJsonError(
            f"demo.json at {path} must hold a JSON object, not {type(data).__name__}")
    missing = [key for key, _check, _shape in _SCHEMA if key not in data]
    wrong = [f"{key} ({shape})" for key, check, shape in _SCHEMA
             if key in data and not check(data[key])]
    problems = []
    if missing:
        problems.append("missing keys: " + ", ".join(missing))
    if wrong:
        problems.append("wrong type: " + ", ".join(wrong))
    if problems:
        raise DemoJsonError("demo.json: " + "; ".join(problems))
    return data
