# stdlib only; tested by test_locked_package.py
"""Read a KEY=VALUE env file (``.env.local``) without python-dotenv.

Deliberately small: comments, blank lines, an optional ``export `` prefix and
one pair of matching quotes. No ``$VAR`` interpolation, no multi-line values.
This module only reads; it never writes an env file.
"""

from pathlib import Path


class EnvFileError(Exception):
    """A non-blank, non-comment line is not ``KEY=VALUE``."""


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def read_env(path: Path) -> dict[str, str]:
    """Return the ``KEY=VALUE`` pairs in ``path``; a missing file is ``{}``."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    env: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key:
            raise EnvFileError(f"{path}:{lineno}: expected KEY=VALUE")
        env[key] = _unquote(value.strip())
    return env
