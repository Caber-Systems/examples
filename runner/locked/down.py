# stdlib only; tested by tests/unit/test_locked_build_run_down.py
"""``./demo.py down`` in locked mode: print the LLM usage, then tear the demo down.

Usage is read first, while the mcp container still exists, by copying the
``caber-demo-usage`` volume's JSONL out of it; ``compose down -v`` then
removes that volume with the rest. ``--keep-images`` keeps the pulled and
built images; ``--purge`` also deletes ``.env.local`` after a successful down.
"""

import functools
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from runner.locked import links, usage
from runner.locked.compose import compose_cmd, docker_run, load_tree, run_guarded
from runner.locked.demo_json import DemoJsonError
from runner.locked.env_build import merge_env_files
from runner.locked.env_file import EnvFileError
from runner.locked.lock import LockError

USAGE_DIR = "/var/lib/caber-demo/usage"


def _has_env_local(root: Path) -> bool:
    """``down`` before any ``run`` still tears down: no ``.env.local`` means no ``--env-file``."""
    return (Path(root) / ".env.local").exists()


def _read_usage(root: Path, demo: dict, run, env) -> "Path | None":
    """Copy the usage volume out of the mcp container; None when it cannot be read."""
    dest = Path(tempfile.mkdtemp(prefix="caber-demo-usage-"))
    cmd = compose_cmd(root, demo, ["cp", f"{demo['services']['mcp']}:{USAGE_DIR}/.", str(dest)],
                      env_file=_has_env_local(root))
    try:
        proc = run(cmd, env=env, capture_output=True, text=True, check=False)
    except (OSError, subprocess.SubprocessError):
        shutil.rmtree(dest, ignore_errors=True)
        return None
    if proc.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        return None
    return dest


def _print_usage(path, out) -> None:
    nothing = "LLM usage: nothing recorded, or no running container to read it from."
    if path is None or not Path(path).is_dir():
        out(nothing)
        return
    try:
        summary = usage.summarize(path)
    except (ValueError, OSError) as exc:
        out(f"LLM usage: the recorded usage file could not be parsed ({exc}).")
        return
    if not summary:
        out(nothing)
        return
    out("LLM usage this installation:")
    for provider in sorted(summary):
        t = summary[provider]
        out(f"  {provider}: {t.get('requests') or 0} requests, "
            f"{t.get('chat_input_tokens') or 0} prompt tokens, "
            f"{t.get('chat_output_tokens') or 0} completion tokens, "
            f"{t.get('embedding_tokens') or 0} embedding tokens")


def run_down(root: Path, *, demo, keep_images, purge, run=subprocess.run, out=print,
             read_usage=None, env=None) -> int:
    """Print usage, ``compose down -v``; 0 on success, 1 when compose fails.

    ``env`` defaults to ``.env``, ``.env.local`` and the shell merged
    (``env_build.merge_env_files``); ``down`` resolves no provider.
    """
    root = Path(root)
    if env is None:
        try:
            env = merge_env_files(root)
        except EnvFileError as exc:
            out(str(exc))
            return 1
    copied = read_usage is None
    if copied:
        read_usage = functools.partial(_read_usage, root, demo, run, env)
    path = read_usage()
    try:
        _print_usage(path, out)
    finally:
        if copied and path is not None:
            shutil.rmtree(path, ignore_errors=True)

    extra = ["down", "-v"] + ([] if keep_images else ["--rmi", "all"])
    cmd = compose_cmd(root, demo, extra, profiles=("seed", "console"), env_file=_has_env_local(root))
    rc = run(cmd, env=env, check=False).returncode
    if rc != 0:
        out(f"docker compose down failed (rc={rc}).")
        return 1
    if purge:
        (root / ".env.local").unlink(missing_ok=True)
        out(".env.local removed; the next run generates new secrets.")
    out(links.HELP)
    return 0


def run_down_cli(root: Path, args) -> int:
    """``./demo.py down [--keep-images] [--purge]``."""
    try:
        _lock, demo = load_tree(root)
    except (LockError, DemoJsonError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return run_guarded(lambda: run_down(root, demo=demo, keep_images=args.keep_images, purge=args.purge,
                                        run=docker_run()))
