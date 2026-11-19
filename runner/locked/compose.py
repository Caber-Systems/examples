# stdlib only; tested by tests/unit/test_locked_build_run_down.py and tests/unit/test_locked_env_build.py
"""The ``docker compose`` command line locked ``build``, ``run``, ``down`` and the modes share.

Platform compose files (``shared/compose/``) come first, then the demo's own
(``demos/<name>/``), so a demo file overrides the platform where both speak.
No ``--project-directory`` is passed: compose takes the project directory
from the first ``-f`` file, as the developer runner does.

Interpolation reads ``--env-file <root>/.env`` (only when that file exists)
and then ``--env-file <root>/.env.local``; compose layers a later file over an
earlier one, so ``.env.local`` wins over ``.env``. The process env each caller
passes as ``env=`` (``env_build``) wins over both.

``docker_run`` and ``run_guarded`` turn a missing ``docker`` binary into one
sentence and exit 1 instead of a traceback, for every subcommand that runs
compose.
"""

import subprocess
from pathlib import Path

from runner.locked import demo_json
from runner.locked.lock import load_lock

DOCKER_MISSING = "docker is not installed or not on PATH; install Docker Desktop and retry."


class DockerMissing(Exception):
    """Launching ``docker`` failed with ``OSError`` (no binary, not executable)."""


def compose_cmd(root: Path, demo, extra, *, profiles=(), env_file: bool = True,
                exists=Path.exists) -> list[str]:
    """``docker compose -f ... [--profile p ...] [--env-file .env] [--env-file .env.local] <extra>``."""
    root = Path(root)
    cmd = ["docker", "compose"]
    if demo is not None:
        for name in demo["platform_compose"]:
            cmd += ["-f", str(root / "shared" / "compose" / name)]
        if demo["compose"]:
            demo_dir = root / "demos" / demo["name"]
            for name in demo["compose"]:
                cmd += ["-f", str(demo_dir / name)]
    for profile in profiles:
        cmd += ["--profile", profile]
    if env_file:
        if exists(root / ".env"):
            cmd += ["--env-file", str(root / ".env")]
        cmd += ["--env-file", str(root / ".env.local")]
    return cmd + list(extra)


def load_tree(root: Path) -> tuple[dict, dict]:
    """The lock and the demo descriptor it names; raises ``LockError`` or ``DemoJsonError``."""
    root = Path(root)
    lock = load_lock(root / "preview.lock.json")
    demo = demo_json.load(root / "demos" / lock["demo"] / "demo.json")
    return lock, demo


def docker_run(run=None):
    """``run`` (``subprocess.run`` looked up at call time) with an ``OSError`` at launch raised as ``DockerMissing``."""
    def call(cmd, **kw):
        try:
            return (run or subprocess.run)(cmd, **kw)
        except OSError as exc:
            raise DockerMissing(str(exc)) from exc
    return call


def run_guarded(fn, *, out=print) -> int:
    """``fn()``; ``DOCKER_MISSING`` and 1 when it raised ``DockerMissing``."""
    try:
        return fn()
    except DockerMissing:
        out(DOCKER_MISSING)
        return 1
