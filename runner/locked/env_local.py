# stdlib only; tested by tests/unit/test_locked_build_run_down.py
"""Create ``.env.local`` from ``.env.template`` once, with fresh secrets.

The exported tree ships ``.env.template``; the first locked ``run`` copies it to
``.env.local`` and fills in each secret in :data:`GENERATED` with a newly
generated value. An existing ``.env.local`` is never rewritten or appended
to, so the stranger's provider key and the secrets the running stack was
started with stay put. The template itself is never edited.
"""

import os
import secrets
from pathlib import Path

# MOCK_IDP_SCIM_TOKEN (ruling P2-55): mock-idp in the exported tree has no
# default for MOCKIDP_API_TOKEN, so mock-idp and caber-core read this value.
GENERATED = ("CSI_HMAC_KEY", "CSI_SEED_API_KEY", "CSI_OBJECT_STORE_SECRET", "MOCK_IDP_SCIM_TOKEN")
SECRET_BYTES = 32


class EnvLocalError(Exception):
    """``.env.local`` does not exist and cannot be created from the template."""


def _key(line: str) -> str:
    head = line.strip()
    if head.startswith("export "):
        head = head[len("export "):]
    key, sep, _value = head.partition("=")
    return key.strip() if sep else ""


def ensure(root: Path, template: Path, generate=secrets.token_urlsafe) -> Path:
    """Return ``root/.env.local``, creating it from ``template`` only when it is absent."""
    target = Path(root) / ".env.local"
    if target.exists():
        return target
    template = Path(template)
    try:
        text = template.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise EnvLocalError(
            f"{template} is missing; the exported tree ships .env.template beside demo.py.") from None
    lines = []
    seen = set()
    for line in text.splitlines():
        key = _key(line)
        if key in GENERATED:
            lines.append(f"{key}={generate(SECRET_BYTES)}")
            seen.add(key)
        else:
            lines.append(line)
    for key in GENERATED:
        if key not in seen:
            lines.append(f"{key}={generate(SECRET_BYTES)}")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.chmod(target, 0o600)
    return target
