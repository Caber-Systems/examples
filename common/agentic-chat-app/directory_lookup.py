"""Display-side lookups against the zt-mimic directory seed.

``user_config.yaml`` holds login credentials, display name and email only.
A user's tenant and groups live in ONE place, the zt-mimic directory seed
(``CaberApp/zt-mimic/seed/directory.json``), which zt-mimic serves over SCIM
and Caber syncs as the authoritative MIMU directory.

The chat app reads the same seed for two things that are NOT authorization:
the sidebar's Tenant / Groups lines, and the ``X-Caber-Tenant`` header the MCP
server still forwards to graphrag-retriever. Caber never authorizes from
either: it resolves the caller's groups from its directory replica.

Users log in to the chat app by EMAIL only: every ``user_config.yaml`` login
name is an email address, and it resolves to the directory user holding that
address (case-insensitive, as zt-mimic's own login matches). A first name is
not a login. ``validate_login_config`` enforces that at app startup.

A missing seed, or a login user the seed does not define, raises
``DirectoryLookupError``. The app shows that error instead of inventing a
tenant.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Mapping, Optional, Tuple

SEED_ENV = "ZT_MIMIC_SEED_FILE"
# Where the Dockerfile copies the seed inside the image.
IMAGE_SEED_PATH = Path("/app/zt_mimic_seed/directory.json")
# The seed's place in a source checkout, relative to this file.
REPO_SEED_PATH = Path(__file__).resolve().parent.parent / "zt-mimic" / "seed" / "directory.json"


class DirectoryLookupError(RuntimeError):
    """The seed cannot be found or does not define the requested user."""


class LoginConfigError(DirectoryLookupError):
    """``user_config.yaml`` names a login that is not a directory user's email."""


def seed_path() -> Path:
    """The seed file to read: ``$ZT_MIMIC_SEED_FILE``, else the image copy, else the checkout."""
    override = os.environ.get(SEED_ENV, "").strip()
    if override:
        path = Path(override)
        if not path.is_file():
            raise DirectoryLookupError(f"{SEED_ENV}={override!r} is not a file")
        return path
    for candidate in (IMAGE_SEED_PATH, REPO_SEED_PATH):
        if candidate.is_file():
            return candidate
    raise DirectoryLookupError(
        f"zt-mimic directory seed not found at {IMAGE_SEED_PATH} or {REPO_SEED_PATH}; "
        f"set {SEED_ENV}"
    )


def load_seed(path: Optional[Path] = None) -> dict:
    target = path or seed_path()
    try:
        seed = json.loads(Path(target).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectoryLookupError(f"cannot read directory seed {target}: {exc}") from exc
    if not isinstance(seed, dict) or not isinstance(seed.get("users"), list):
        raise DirectoryLookupError(f"directory seed {target} has no 'users' list")
    return seed


def _emails(user: dict) -> List[str]:
    return [e["value"] for e in user.get("emails") or [] if isinstance(e, dict) and e.get("value")]


def user_by_email(login_email: str, seed: Optional[dict] = None) -> dict:
    """The one directory user holding ``login_email`` (case-insensitive).

    Raises ``DirectoryLookupError`` when no user, or more than one, holds it.
    """
    data = seed if seed is not None else load_seed()
    wanted = login_email.strip().lower()
    matches = [u for u in data["users"] if any(wanted == e.lower() for e in _emails(u))]
    if not matches:
        raise DirectoryLookupError(
            f"login {login_email!r} is not an email of any zt-mimic directory user")
    if len(matches) > 1:
        names = sorted(str(u.get("userName")) for u in matches)
        raise DirectoryLookupError(f"login {login_email!r} is an email of several users: {names}")
    return matches[0]


def tenant_and_groups(login_email: str, seed: Optional[dict] = None) -> Tuple[str, List[str]]:
    """Return ``(organization, groups)`` for the directory user holding ``login_email``."""
    user = user_by_email(login_email, seed)
    organization = user.get("organization")
    if not organization:
        raise DirectoryLookupError(f"directory user {user.get('userName')!r} has no organization")
    return organization, list(user.get("groups") or [])


def validate_login_config(usernames: Mapping[str, Mapping], seed: Optional[dict] = None) -> None:
    """Fail loudly unless every login name is a directory user's email.

    ``usernames`` is ``user_config.yaml``'s ``credentials.usernames``. Each key
    must be a lowercase email (streamlit-authenticator lowercases the keys and
    what the user types, so the session username, which is the asserted
    subject and the demo-questions key, is always the lowercase form), equal
    to its own ``email`` field, and held by exactly one directory user. Raises
    ``LoginConfigError`` naming every bad entry.
    """
    if not usernames:
        raise LoginConfigError("user_config.yaml defines no logins")
    data = seed if seed is not None else load_seed()
    problems = []
    for login, entry in usernames.items():
        if "@" not in login:
            problems.append(f"{login!r} is not an email; users log in by email only")
            continue
        if login != login.lower():
            problems.append(f"{login!r} is not lowercase; the session username always is")
        email = (entry or {}).get("email", "")
        if email.lower() != login.lower():
            problems.append(f"{login!r} has email {email!r}; the two must be the same address")
        try:
            user_by_email(login, data)
        except DirectoryLookupError as exc:
            problems.append(str(exc))
    if problems:
        raise LoginConfigError("user_config.yaml: " + "; ".join(problems))
