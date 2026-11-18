# stdlib only; tested by tests/unit/test_locked_build_run_down.py
"""``./demo.py run`` in locked mode: start the stack, seed it, wait for health, open the browser.

Steps, in order, each failure one sentence and exit 1: the machine checks
``check`` runs (Docker, ports, memory, disk), before anything is written;
create ``.env.local`` if absent; merge ``.env``, ``.env.local`` and the shell
(``env_build``), resolve the selected provider and probe it, the probe
measuring the embedding vector length; ``compose up -d`` with every platform
and demo file (Console under its ``console`` profile) and that resolved env, in
observe for the chat app and the MCP server both;
run the one-shot ``seed`` service with its output shown live; activate the
demo's policies (none today); wait on each ``demo.json.health`` URL; open each
``demo.json.urls`` URL and print them with the ``what_to_try`` questions.
"""

import functools
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

from runner.locked import check, env_local, mode
from runner.locked.compose import compose_cmd, docker_run, load_tree, run_guarded
from runner.locked.env_build import merge_env_files
from runner.locked.demo_json import DemoJsonError
from runner.locked.env_file import EnvFileError, read_env
from runner.locked.lock import LockError


def _logs(root: Path, demo: dict, run, env, svc: str) -> str:
    """The last 20 compose log lines of ``svc``; empty when compose cannot say."""
    cmd = compose_cmd(root, demo, ["logs", "--no-color", "--tail", "20", svc])
    try:
        proc = run(cmd, env=env, capture_output=True, text=True, check=False)
    except OSError:
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


def run_run(root: Path, *, lock, demo, run=subprocess.run, http=None, open_url=webbrowser.open,
            out=print, logs=None, wait_for=None, policies=None, policy_plan=None,
            preflight=None, environ=None, read=read_env, probe=None) -> int:
    """Bring the demo up and open it; 0 when every URL is open, 1 at the first failure.

    ``policies`` (a module with ``activate`` and ``ActivationFailed``) and
    ``policy_plan`` default to ``runner.locked.policies`` and
    ``runner.core.policy_plan.policy_plan``, imported only when the demo has
    policies to flip. ``preflight`` (the machine checks), ``environ``, ``read``
    and ``probe`` (``llm_probe.probe_with_dim``) are test seams.
    """
    root = Path(root)
    if preflight is None:
        preflight = lambda: check.machine_problems(root, lock, demo)[0]  # noqa: E731
    problems = preflight()
    if problems:
        for line in problems:
            out(line)
        return 1

    try:
        env_local.ensure(root, root / ".env.template")
    except env_local.EnvLocalError as exc:
        out(str(exc))
        return 1

    try:
        merged = merge_env_files(root, environ=os.environ if environ is None else environ, read=read)
    except EnvFileError as exc:
        out(str(exc))
        return 1
    problems, compose_env = check.provider_problems(
        merged, **({} if probe is None else {"probe": probe}))
    if problems:
        for line in problems:
            out(line)
        return 1
    # P2-145: every run starts in observe, for the chat app and the MCP server alike.
    compose_env[mode.STATE_KEY] = "false"
    compose_env[mode.MCP_COMPOSE_VAR] = "false"

    if logs is None:
        logs = functools.partial(_logs, root, demo, run, compose_env)
    if wait_for is None:
        from runner.locked.health import wait_for

    rc = run(compose_cmd(root, demo, ["up", "-d", "--pull", "never"], profiles=("console",)),
             env=compose_env, check=False).returncode
    if rc != 0:
        out(f"docker compose up failed (rc={rc}).")
        return 1

    svc = demo["services"]["seed"]
    rc = run(compose_cmd(root, demo, ["--profile", "seed", "run", "--rm", svc]), env=compose_env,
             check=False).returncode
    if rc != 0:
        out(f"seed failed (rc={rc}); last 20 log lines of {svc}:")
        out("\n".join(logs(svc).splitlines()[-20:]))
        return 1

    plan = demo["policies"]
    if plan["enable"] or plan["disable"]:
        env_path = root / ".env.local"
        try:
            seed_key = read_env(env_path)["CSI_SEED_API_KEY"]
        except EnvFileError as exc:
            out(f"policy activation failed: {exc}.")
            return 1
        except KeyError:
            out(f"policy activation failed: {env_path} has no CSI_SEED_API_KEY.")
            return 1
        # Imported only here, so a demo with empty policy lists never loads them.
        _p = policies
        if _p is None:
            from runner.locked import policies as _p
        if policy_plan is None:
            from runner.core.policy_plan import policy_plan
        if http is None:
            from runner.locked import http
        try:
            _p.activate(demo["urls"]["dashboard"], seed_key,
                        policy_plan(demo["name"], root / "demos"), http)
        except _p.ActivationFailed as exc:
            out(f"policy activation failed: {exc}")
            return 1
    else:
        out("policies: nothing to activate for this demo.")

    for probe_url in demo["health"]:
        url, timeout_s = probe_url["url"], probe_url["timeout_s"]
        if not wait_for(url, timeout_s):
            out(f"{url} did not answer within {timeout_s}s.")
            return 1

    for url in demo["urls"].values():
        open_url(url)
    for label, url in demo["urls"].items():
        out(f"{label}: {url}")
    out("What to try:")
    for question in demo["what_to_try"]:
        out(f"  - {question}")
    return 0


def run_run_cli(root: Path, args) -> int:
    """``./demo.py run``."""
    try:
        lock, demo = load_tree(root)
    except (LockError, DemoJsonError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return run_guarded(lambda: run_run(root, lock=lock, demo=demo, run=docker_run()))
