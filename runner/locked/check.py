# stdlib only; tested by tests/unit/test_locked_check.py
"""``./demo.py check`` in locked mode: can this machine run the preview.

``run_check`` returns one plain sentence per problem (empty = pass). Sentences
that start with ``note: `` (any case, so the ``NOTE: ... since the lock`` lines
too) are reported but never fail the check. Apart from the
two port sentences, no sentence here contains the letters "port" (a test relies
on it), so wording is chosen with that in mind.
"""

import functools
import hashlib
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Mapping

from runner.locked import demo_json, links, llm_probe
from runner.locked.demo_json import DemoJsonError
from runner.locked.env_build import merge_env_files
from runner.locked.env_file import EnvFileError
from runner.locked.lock import LockError, load_lock
from runner.locked.providers import EmbeddingDimMissing, ProviderError, resolve_providers

GIB = 1024 ** 3
MIN_MEMORY = 16 * GIB
MIN_DISK = 20 * GIB
MIN_COMPOSE = (2, 20)
MIN_PYTHON = (3, 10)
ARCHES = ("x86_64", "arm64", "aarch64")
NOTICE_LABEL = "com.caber.notice.sha256"
NOTE = "note: "
SUBPROCESS_TIMEOUT_S = 60


class DockerUnavailable(Exception):
    """Docker is not installed, not running, or did not answer in time."""


def _run(cmd: list[str], timeout: int = SUBPROCESS_TIMEOUT_S) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DockerUnavailable(str(exc)) from exc


def _docker(refs: list[str]) -> dict:
    """Memory, compose version and notice labels of the images present locally."""
    info = _run(["docker", "info", "--format", "{{.MemTotal}}"])
    if info.returncode != 0:
        raise DockerUnavailable(info.stderr.strip())
    try:
        info_mem = int(info.stdout.strip())
    except ValueError as exc:
        raise DockerUnavailable(f"unreadable docker info output: {info.stdout!r}") from exc
    compose = _run(["docker", "compose", "version", "--short"])
    if compose.returncode != 0:
        raise DockerUnavailable(compose.stderr.strip())
    labels: dict = {}
    for ref in refs:
        res = _run(["docker", "image", "inspect", "--format",
                    '{{index .Config.Labels "' + NOTICE_LABEL + '"}}', ref])
        if res.returncode != 0:
            continue
        labels[ref] = res.stdout.strip() or None
    return {"info_mem": info_mem, "compose_version": compose.stdout.strip().lstrip("v"), "labels": labels}


def _port_owner(port: int) -> str:
    try:
        res = subprocess.run(["lsof", "-nP", "-i", f":{port}", "-sTCP:LISTEN", "-Fc"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    for line in res.stdout.splitlines():
        if line.startswith("c"):
            return line[1:]
    return ""


def _bind_fails(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


def _connect_succeeds(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _ports_in_use(ports: list[int]) -> dict[int, str]:
    """Busy ports mapped to the owning command ('' if unknown).

    A port is busy when a 127.0.0.1 bind fails or a 127.0.0.1 connect succeeds:
    on macOS the ``SO_REUSEADDR`` bind still succeeds beside a listener on
    0.0.0.0, which is where Docker publishes.
    """
    return {port: _port_owner(port) for port in ports if _bind_fails(port) or _connect_succeeds(port)}


def _compose_too_old(version: str) -> bool:
    try:
        major, minor = (int(part) for part in version.split(".")[:2])
    except ValueError:
        return True
    return (major, minor) < MIN_COMPOSE


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _notice_notes(root: Path, expected: str, labels: dict) -> list[str]:
    notes = []
    notice = root / "LICENSE-PREVIEW.md"
    if not notice.is_file():
        notes.append(f"{NOTE}LICENSE-PREVIEW.md is missing beside demo.py; the lock notice sha256 is {expected}.")
    else:
        actual = _file_sha256(notice)
        if actual != expected:
            notes.append(f"{NOTE}LICENSE-PREVIEW.md sha256 {actual} differs from the lock notice sha256 {expected}.")
    for ref, label in labels.items():
        if not label:
            notes.append(f"{NOTE}image {ref} has no {NOTICE_LABEL} label; the lock notice sha256 is {expected}.")
        elif label != expected:
            notes.append(f"{NOTE}image {ref} label {NOTICE_LABEL} {label} differs from the lock notice sha256 {expected}.")
    return notes


def _path_digest(path: Path) -> str:
    """The lock writer's convention: a file's sha256, or for a directory one sha256 over
    ``rel\\n`` + ``file sha256\\n`` of every file in sorted ``rglob`` order."""
    if not path.is_dir():
        return _file_sha256(path)
    running = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            running.update(f"{p.relative_to(path).as_posix()}\n".encode())
            running.update(f"{_file_sha256(p)}\n".encode())
    return running.hexdigest()


def _files_notes(root: Path, files: Mapping[str, str]) -> list[str]:
    """P2-146: one ``NOTE`` line per ``files_sha256`` entry edited or missing since the lock."""
    notes = []
    for rel, expected in files.items():
        target = Path(root) / rel
        if not target.exists():
            notes.append(f"NOTE: missing since the lock: {rel}")
        elif _path_digest(target) != expected:
            notes.append(f"NOTE: edited since the lock: {rel}")
    return notes


def _is_note(line: str) -> bool:
    return line.lower().startswith(NOTE)


def machine_problems(root: Path, lock: dict, demo: dict, *, system=platform.system, machine=platform.machine,
                     python_version=lambda: sys.version_info[:2],
                     disk_free=lambda p: shutil.disk_usage(p).free,
                     docker=None, ports_in_use=_ports_in_use) -> tuple[list[str], dict]:
    """``(one sentence per machine problem, image notice labels)``: OS, arch, Docker, Python, ports, disk.

    ``check`` reports these and ``run`` refuses on them before it writes anything.
    """
    root = Path(root)
    if docker is None:
        docker = functools.partial(_docker, [img["ref"] for img in lock.get("images", {}).values()])

    if system() == "Windows":
        return ["Windows cannot run this demo; use macOS or Linux."], {}
    out: list[str] = []

    arch = machine()
    if arch not in ARCHES:
        out.append(f"This machine's architecture {arch} is not x86_64 or arm64, "
                   "and the demo images are built for those two.")

    labels: dict = {}
    try:
        info = docker()
    except DockerUnavailable:
        out.append("Docker is not running or not installed; start Docker Desktop and run ./demo.py check again.")
    else:
        labels = info["labels"]
        if info["info_mem"] < MIN_MEMORY:
            out.append(f"Docker has {info['info_mem'] / GIB:.1f} GiB of memory; "
                       "the demo needs at least 16 GiB. Raise it in Docker Desktop settings.")
        if _compose_too_old(info["compose_version"]):
            out.append(f"Docker Compose {info['compose_version']} is older than the 2.20 the demo needs.")

    major, minor = python_version()
    if (major, minor) < MIN_PYTHON:
        out.append(f"Host Python {major}.{minor} is older than the 3.10 the demo needs.")

    busy = ports_in_use(list(demo["ports"].values()))
    for name, port in demo["ports"].items():
        if port in busy:
            owner = busy[port]
            out.append(f"Port {port} ({name}) is in use by {owner}." if owner else f"Port {port} ({name}) is in use.")

    free = disk_free(root)
    if free < MIN_DISK:
        out.append(f"Free disk at {root} is {free / GIB:.1f} GiB; the demo needs at least 20 GiB.")
    return out, labels


def provider_problems(env: Mapping[str, str], *, probe=llm_probe.probe_with_dim) -> tuple[list[str], "dict | None"]:
    """``(problems, resolved env)``: the resolver's sentence, else the probe's, else ``([], env)``.

    ``env`` is the merged, unresolved environment (``env_build.merge_env_files``).
    The selection is resolved without a dimension first, so a missing key or model
    is one sentence and nothing is sent; the probe then measures the embedding
    vector length that ``SELECTED_EMBEDDING_DIM`` takes when no
    ``<PROVIDER>_EMBEDDING_DIM`` is configured.
    """
    try:
        resolve_providers(env)
    except EmbeddingDimMissing:
        pass
    except ProviderError as exc:
        return [str(exc)], None
    problems, dim = probe((env.get("SELECTED_LLM_PROVIDER") or "").strip().upper(), env)
    if problems:
        return list(problems), None
    try:
        return [], resolve_providers(env, embedding_dim=dim)
    except ProviderError as exc:
        return [str(exc)], None


def run_check(root: Path, *, env: Mapping[str, str], system=platform.system, machine=platform.machine,
              python_version=lambda: sys.version_info[:2],
              disk_free=lambda p: shutil.disk_usage(p).free,
              docker=None, probe=llm_probe.probe_with_dim, ports_in_use=_ports_in_use) -> list[str]:
    """One sentence per problem found on this machine; empty means ready."""
    lock = load_lock(root / "preview.lock.json")
    demo = demo_json.load(root / "demos" / lock["demo"] / "demo.json")
    out, labels = machine_problems(root, lock, demo, system=system, machine=machine,
                                   python_version=python_version, disk_free=disk_free,
                                   docker=docker, ports_in_use=ports_in_use)
    if system() == "Windows":
        return out
    out.extend(provider_problems(env, probe=probe)[0])
    out.extend(_notice_notes(root, lock["notice"]["sha256"], labels))
    out.extend(_files_notes(root, lock.get("files_sha256") or {}))
    return out


def _registry_free(ref: str) -> bool:
    """True when ``ref`` names no registry host, so docker pulls it from docker.io."""
    if "/" not in ref:
        return True  # "postgres:14-alpine": the colon is a tag, not a registry port
    head = ref.split("/", 1)[0]
    return not ("." in head or ":" in head or head == "localhost")


def warnings(root: Path) -> list[str]:
    """Non-fatal advice; one sentence when any image pulls anonymously from docker.io."""
    lock = load_lock(root / "preview.lock.json")
    refs = [img["ref"] for img in lock["images"].values() if _registry_free(img["ref"])]
    if not refs:
        return []
    return [f"{len(refs)} image(s) in preview.lock.json pull anonymously from docker.io, "
            f"which rate-limits anonymous pulls: {', '.join(refs)}. "
            "If a pull fails, run docker login and try again."]


def run_check_cli(root: Path, args) -> int:
    """Print the check result; exit 1 when any sentence is not a note (``note: ``, any case).

    A failing check ends with the same Help line ``down`` prints (``links.HELP``).
    """
    try:
        env = merge_env_files(root)
        problems = run_check(root, env=env)
        warns = warnings(root)
    except (EnvFileError, LockError, DemoJsonError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for line in problems:
        print(line)
    for line in warns:
        print(f"warning: {line}")
    if any(not _is_note(line) for line in problems):
        print(links.HELP)
        return 1
    print("check passed.")
    return 0
