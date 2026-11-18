# stdlib only; tested by tests/unit/test_policy_plan_and_mode.py
"""``./demo.py enforce`` and ``./demo.py observe`` in locked mode.

The mode is ``CSI_GLOBAL_REDACT_ENABLED`` in the chat app and the MCP server
(``true`` enforce, ``false`` observe). The demo compose file reads it from
``CSI_GLOBAL_REDACT_ENABLED`` for the app and from ``MCP_GLOBAL_REDACT_ENABLED``
for the MCP server, so both are set and both services are recreated together,
the same ``up -d --no-deps --force-recreate --pull never`` the developer
runner's ``docker_lifecycle.recreate_services`` uses. Each recreated container
is then asked for the value it got, the health URLs are waited on, and the
chat window is opened again.

Every compose call gets the env ``run`` gave the stack: ``.env``, ``.env.local``
and the shell merged (``env_build``) with the provider variables resolved, so a
mode switch from a new shell recreates the apps with their key. There is no
provider probe here; the embedding dimension is read back from the running MCP
container (the chat app carries no ``SELECTED_EMBEDDING_DIM``), and a stack that
is not running is refused before anything is recreated.
"""

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

from runner.locked.compose import compose_cmd, docker_run, load_tree, run_guarded
from runner.locked.demo_json import DemoJsonError
from runner.locked.env_build import merge_env_files
from runner.locked.env_file import EnvFileError, read_env
from runner.locked.lock import LockError
from runner.locked.providers import EmbeddingDimMissing, ProviderError, resolve_providers

STATE_KEY = "CSI_GLOBAL_REDACT_ENABLED"
MCP_COMPOSE_VAR = "MCP_GLOBAL_REDACT_ENABLED"
DIM_KEY = "SELECTED_EMBEDDING_DIM"


def _running_dim(root: Path, demo: dict, run, env: dict) -> "int | None":
    """``SELECTED_EMBEDDING_DIM`` of the running MCP container (the app's when no MCP), or None."""
    svc = demo["services"].get("mcp") or demo["services"]["app"]
    proc = run(compose_cmd(root, demo, ["exec", "-T", svc, "printenv", DIM_KEY]),
               env=env, capture_output=True, text=True, check=False)
    got = (proc.stdout or "").strip()
    if proc.returncode != 0 or not got.isdigit() or int(got) <= 0:
        return None
    return int(got)


def mode_env(root: Path, demo: dict, run, *, environ=None, read=read_env) -> tuple["dict | None", str]:
    """``(resolved env, "")`` for a mode switch, or ``(None, one sentence)``; recreates nothing.

    No probe: the embedding dimension is read back from the running stack, so a
    stack that is not running is refused here.
    """
    merged = merge_env_files(root, environ=os.environ if environ is None else environ, read=read)
    try:
        resolve_providers(merged)
    except EmbeddingDimMissing:
        pass
    except ProviderError as exc:
        return None, str(exc)
    dim = _running_dim(root, demo, run, merged)
    if dim is None:
        return None, (f"The demo stack is not running ({DIM_KEY} could not be read from "
                      f"{demo['services'].get('mcp') or demo['services']['app']}); start it with ./demo.py run.")
    try:
        return resolve_providers(merged, embedding_dim=dim), ""
    except ProviderError as exc:
        return None, str(exc)


def set_mode(root, demo, enforce: bool, run=subprocess.run, *, out=print, wait_for=None,
             open_url=webbrowser.open, env=None, environ=None, read=read_env) -> int:
    """Recreate the app (and MCP server) in enforce or observe; 0 when the switch took.

    ``env`` given is used as it stands (tests); otherwise ``mode_env`` builds it,
    and a refusal there is exit 1 with nothing recreated.
    """
    root = Path(root)
    if wait_for is None:
        from runner.locked.health import wait_for
    value = "true" if enforce else "false"
    services = [demo["services"]["app"]]
    if "mcp" in demo["services"]:
        services.append(demo["services"]["mcp"])

    if env is None:
        try:
            env, problem = mode_env(root, demo, run, environ=environ, read=read)
        except EnvFileError as exc:
            env, problem = None, str(exc)
        if env is None:
            out(problem)
            return 1
    compose_env = dict(env)
    compose_env[STATE_KEY] = value
    compose_env[MCP_COMPOSE_VAR] = value
    up = compose_cmd(root, demo, ["up", "-d", "--no-deps", "--force-recreate", "--pull", "never", *services])
    rc = run(up, env=compose_env, check=False).returncode
    if rc != 0:
        out(f"docker compose up failed (rc={rc}).")
        return 1

    for svc in services:
        probe = run(compose_cmd(root, demo, ["exec", "-T", svc, "printenv", STATE_KEY]),
                    env=compose_env, capture_output=True, text=True, check=False)
        got = probe.stdout.strip()
        if got != value:
            out(f"{svc} came back with {STATE_KEY}={got!r}, expected {value!r}.")
            return 1

    for check in demo.get("health", []):
        url, timeout_s = check["url"], check["timeout_s"]
        if not wait_for(url, timeout_s):
            out(f"{url} did not answer within {timeout_s}s.")
            return 1

    chat = demo.get("urls", {}).get("chat")
    if chat:
        open_url(chat)
    out(f"Mode: {'enforce' if enforce else 'observe'} ({STATE_KEY}={value}).")
    return 0


def _set_mode_cli(root: Path, enforce: bool) -> int:
    try:
        _lock, demo = load_tree(root)
    except (LockError, DemoJsonError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return run_guarded(lambda: set_mode(root, demo, enforce, run=docker_run()))


def run_enforce_cli(root: Path, args) -> int:
    """``./demo.py enforce``."""
    return _set_mode_cli(root, True)


def run_observe_cli(root: Path, args) -> int:
    """``./demo.py observe``."""
    return _set_mode_cli(root, False)
