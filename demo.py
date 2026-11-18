#!/usr/bin/env python3
"""Caber Demo Catalog CLI -- Single entry point for all demos.

Human workflow (visible, interactive):
    demo.py up <demo>                         Launch + seed + keep browser open
    demo.py down <demo>                       Tear down

AI / automation workflow (headless, fast):
    demo.py ci <demo>                         Full lifecycle: launch -> seed -> test -> teardown
    demo.py test <demo>                       Run smoke tests on a running demo
    demo.py video <demo>                      Produce narrated demo video

Utilities:
    demo.py list                              List available demos
    demo.py info <demo>                       Show demo details
    demo.py check [<demo>]                    Verify prerequisites
    demo.py build <demo>                      Build demo app containers
    demo.py build --platform                  Build all csi-* platform containers
    demo.py build --platform --service X,Y    Build specific platform containers

Cloud:
    demo.py deploy <demo> --target aws        Deploy to AWS
    demo.py undeploy <demo> --target aws      Tear down AWS deployment
"""

import argparse
import json
import os
import re
import socket
import shutil
import subprocess
import sys
import platform
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_VENV_PY = ROOT / ".venv" / "bin" / "python3"
if _VENV_PY.exists() and sys.prefix == sys.base_prefix and os.environ.get("CABER_NO_VENV_REEXEC") != "1":
    os.execv(str(_VENV_PY), [str(_VENV_PY)] + sys.argv)


def is_locked(root: Path) -> bool:
    """Exported public-preview tree: lock present and no sibling PyCaber checkout."""
    return (root / "preview.lock.json").exists() and not (root.parent / "PyCaber").exists()


# Locked mode hands over BEFORE any third-party or runner.core import (Plan
# Ruling 8): a stranger's exported tree has only the standard library.
if __name__ == "__main__" and is_locked(ROOT):
    from runner.locked.cli import main as locked_main
    sys.exit(locked_main(sys.argv[1:]))

from runner.core.aws_auth import (
    explain_aws_or_ecr_auth_failure,
    format_ecr_login_command,
    is_ecr_image,
)
from runner.core.object_store_buckets import bucket_environment
from runner.core.catalog import NER_OVERLAY, NER_OVERLAY_DEMO, WITH_NER_ENV
from runner.core.llm_provider import CI_LLM_MODEL_ENV, ensure_llm_provider, resolve_llm_provider

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:
    _load_dotenv = None

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

import re
from datetime import datetime, timedelta, timezone


def _load_dotenv_file() -> None:
    """Load ROOT/.env (developer mode) when python-dotenv is installed and the file exists.

    Runs at import, BEFORE any docker compose invocation: compose resolves
    ${VAR} references (e.g. OPENAI_API_KEY) from the calling shell's
    environment at container creation time, so without this containers get
    empty strings.
    """
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    if _load_dotenv is not None:
        _load_dotenv(env_path)
        return
    # Stdlib fallback when python-dotenv is unavailable
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k.startswith("export "):
            k = k[7:].strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv_file()
CATALOG_YAML = ROOT / "catalog.yaml"
CATALOG_JSON = ROOT / "catalog.json"
DASHBOARD_UI_DIR = ROOT.parent / "CaberUserConsole"
DASHBOARD_UI_PID = ROOT / ".dashboard-ui.pid"
MVP_ROOT = Path(os.environ.get("CABER_MVP_ROOT", ROOT.parent / "mvp")).expanduser().resolve()

# --- GLiNER2 native GPU model-server automation (Apple MPS) -----------------
# When DocParser.gliner is enabled with engine gliner2 (GLINER_ENABLE/GLINER_ENGINE
# override the config), `up`/`ci` automate the full "models-only GPU serving"
# operator flow: start PyCaber's native gliner2 model server on the HOST (Apple MPS),
# then point the caber-core container at it via CSI_GLINER2_MODEL_URL so gliner2
# forward passes run on the Mac GPU. The server is a host process (containers can't
# see the GPU), and it is REQUIRED when that NER config is on -- caber-core installs
# the gliner2 client only, so there is no in-container model to fall back to;
# see PyCaber scripts/parser_service.py + orchestrator gliner2_model_server.py.
PYCABER_ROOT = Path(os.environ.get("CABER_PYCABER_ROOT", ROOT.parent / "PyCaber")).expanduser().resolve()
PARSER_SERVICE_PY = PYCABER_ROOT / "scripts" / "parser_service.py"
GLINER2_GPU_PORT = 8765
# 127.0.0.1 for the host-side health probe; containers reach it via
# host.docker.internal (Docker Desktop/macOS maps this to the host automatically).
GLINER2_GPU_HEALTH_URL = f"http://127.0.0.1:{GLINER2_GPU_PORT}/health"
GLINER2_GPU_CONTAINER_URL = f"http://host.docker.internal:{GLINER2_GPU_PORT}"
# The supported launcher (parser_service.py) records its server under this pid
# file. We use it to tell the launcher-managed native-venv server (torch built
# by `pull` into .parser_service_state/native_venv) apart from a FOREIGN server
# a developer may have hand-started on :8765 from a general venv such as
# PyCaber/.venv. A foreign server built on an older torch (2.8.x) crashes under
# Apple MPS with `MPSNDArray ... buffer is not large enough` (SIGABRT) on the
# first real forward pass, while /health -- a pure liveness probe that never
# loads the model -- still answers 200. Reusing it would silently break NER, so
# we detect and warn instead of trusting the bare health probe.
GLINER2_LAUNCHER_PID_FILE = (
    PARSER_SERVICE_PY.parent / ".parser_service_state" / "gliner2.pid"
)
# caber-core is configured from config.json (CSI_CONFIG_FILE), not from the
# GLINER_* container env vars the retired Python S3_Scanner read, so the
# model-server automation must gate on the same source of truth it uses.
CONFIG_BASE_JSON = ROOT / "shared" / "config" / "config-base.json"
# Must match parser_service.EXIT_REQUIREMENTS_NOT_MET: the host doesn't meet the
# resource requirements for native GPU serving -> fall back to CPU (not an error).
PARSER_SERVICE_REQUIREMENTS_EXIT = 3

PREVIEW_LIMITS_URL = "https://github.com/Caber-Systems/preview-license/blob/v1.2/PREVIEW_LIMITS.md"


def print_preview_edition_notice():
    """Print the Caber Community Preview Edition startup notice."""
    print("  [preview] Caber Community Preview Edition (limits: 20,000 events, 500 objects, 5 SDK instances)")
    print(f"  [preview] Evaluation only — reverse engineering & AI cloning prohibited. Terms & reset: {PREVIEW_LIMITS_URL}")



def _truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _configured_ner():
    """Resolve (enabled, engine) the way caber-core does: the DocParser block of
    the shared Env_Manager base config, with GLINER_ENABLE/GLINER_ENGINE retained
    as operator overrides."""
    enabled = False
    engine = "gliner"
    try:
        with open(CONFIG_BASE_JSON, encoding="utf-8") as fh:
            cfg = json.load(fh)
        # Every level is isinstance-checked rather than trusted: a schema-shaped
        # surprise (e.g. "gliner": "oops") must degrade to env-only resolution the
        # same way a missing file does, not crash `up`/`ci` with an AttributeError.
        doc_parser = cfg.get("DocParser") if isinstance(cfg, dict) else None
        gliner = doc_parser.get("gliner") if isinstance(doc_parser, dict) else None
        if isinstance(gliner, dict):
            if isinstance(gliner.get("enabled"), bool):
                enabled = gliner["enabled"]
            # DEPRECATED spelling (DocParser.gliner.engine), read first so the
            # canonical DocParser.nerEngine below wins when a config sets both
            # -- matches PyCaber sys_config registry.rs registration order.
            if isinstance(gliner.get("engine"), str) and gliner["engine"].strip():
                engine = gliner["engine"].strip().lower()
        if isinstance(doc_parser, dict):
            if isinstance(doc_parser.get("nerEngine"), str) and doc_parser["nerEngine"].strip():
                engine = doc_parser["nerEngine"].strip().lower()
    except (OSError, ValueError):
        # No/unreadable base config: fall back to env-only resolution.
        pass

    # Override order matters. caber-core resolves these from
    # CSI_DOC_PARSER_GLINER_ENABLED / CSI_DOC_PARSER_NER_ENGINE (PyCaber
    # crates/config/src/registry.rs), so those must win here too -- otherwise
    # an operator could set GLINER_ENABLE=0, disarm the host model server, and
    # leave caber-core with NER still on from the config file: zero entities
    # and no warning, the exact failure this gate exists to catch.
    # GLINER_ENABLE/GLINER_ENGINE are the retired Python scanner's names, kept as a
    # deprecated fallback for existing .env files.
    env_enabled = (os.environ.get("CSI_DOC_PARSER_GLINER_ENABLED", "").strip()
                   or os.environ.get("GLINER_ENABLE", "").strip())
    if env_enabled:
        enabled = _truthy(env_enabled)
    env_engine = (os.environ.get("CSI_DOC_PARSER_NER_ENGINE", "").strip()
                  or os.environ.get("GLINER_ENGINE", "").strip()).lower()
    if env_engine:
        engine = env_engine
    return enabled, engine


def gliner2_gpu_requested():
    """The GPU model-server flow is armed only when NER is enabled AND the enrich
    engine is gliner2 -- the exact condition to gate the automation on."""
    enabled, engine = _configured_ner()
    return enabled and engine == "gliner2"


def _gliner2_gpu_healthy():
    try:
        with urllib.request.urlopen(GLINER2_GPU_HEALTH_URL, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _gliner2_server_is_launcher_managed():
    """True iff the supported launcher (parser_service.py) currently owns a live
    server -- i.e. its pid file exists and names a running process. A healthy
    :8765 with NO such record is a FOREIGN server (e.g. hand-started from
    PyCaber/.venv), which may be built on a torch that crashes under MPS."""
    try:
        pid = int(GLINER2_LAUNCHER_PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _native_venv_ready(venv_python):
    """True iff the launcher's native venv exists AND can import the model-server
    deps. Guards against a PARTIAL venv (e.g. a `pull` that died mid-install on a
    full disk) being mistaken for a ready one."""
    if not venv_python.exists():
        return False
    try:
        return subprocess.run(
            [str(venv_python), "-c", "import torch, gliner2, uvicorn, fastapi"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except Exception:
        return False


def ensure_gliner2_gpu_server():
    """Idempotently start the native Apple-MPS gliner2 model server on the host and
    export CSI_GLINER2_MODEL_URL so the container routes gliner2 to it.

    No-op unless `gliner2_gpu_requested()`. On ANY failure (launcher missing, pull
    or start failed, never became healthy) prints a clear WARNING and leaves
    CSI_GLINER2_MODEL_URL unset, which means the gliner2 enrich stage is
    UNAVAILABLE -- caber-core installs the gliner2 client only, so there is no
    in-container CPU fallback. Returns True iff the container will use the GPU
    server; callers that need NER should use `require_gliner2_gpu_server()`, which
    turns that False into a loud abort rather than a silently entity-free demo."""
    if not gliner2_gpu_requested():
        return False

    api_key = os.environ.get("CSI_GLINER2_MODEL_API_KEY") or "dev-local-key"
    container_url = os.environ.get("CSI_GLINER2_MODEL_URL") or GLINER2_GPU_CONTAINER_URL
    # Keep the server's model aligned with the configured gliner2 model id so the
    # host server loads exactly what the pipeline expects.
    gliner_model = os.environ.get("GLINER_MODEL", "").strip()
    if gliner_model.startswith("fastino/gliner2") and not os.environ.get("CSI_GLINER2_MODEL"):
        os.environ["CSI_GLINER2_MODEL"] = gliner_model

    if _gliner2_gpu_healthy():
        if _gliner2_server_is_launcher_managed():
            print(f"  [gliner2-gpu] Native model server already running on "
                  f"127.0.0.1:{GLINER2_GPU_PORT}")
        else:
            # /health answered but the supported launcher does not own this
            # process: a foreign server (e.g. hand-started from PyCaber/.venv)
            # is on the port. It may be built on an older torch that aborts
            # under Apple MPS with "MPSNDArray ... buffer is not large enough"
            # on the first forward pass. Do not silently trust it.
            print(f"  [gliner2-gpu] WARNING: a gliner2 server is answering on "
                  f"127.0.0.1:{GLINER2_GPU_PORT} but was NOT started by the "
                  f"supported launcher (scripts/parser_service.py). If it was "
                  f"launched from a general venv such as PyCaber/.venv it can "
                  f"crash under Apple MPS (\"buffer is not large enough\") on the "
                  f"first request and NER will fail. Stop that process and re-run "
                  f"so demo.py manages the supported native-venv server "
                  f"(`python {PARSER_SERVICE_PY} stop`), or ensure it was started "
                  f"via parser_service.py.")
    else:
        if not PARSER_SERVICE_PY.exists():
            print(f"  [gliner2-gpu] WARNING: parser_service.py not found at "
                  f"{PARSER_SERVICE_PY} (set CABER_PYCABER_ROOT). The gliner2 enrich "
                  f"stage will be UNAVAILABLE (caber-core installs the gliner2 client "
                  f"only).")
            return False
        # One-time: create the native venv (torch + gliner2 + fastapi) and cache the
        # model. `pull` is idempotent but re-pips on every call, so only run it when
        # the venv is missing or incomplete.
        venv_python = (PARSER_SERVICE_PY.parent / ".parser_service_state" /
                       "native_venv" / "bin" / "python")
        if not _native_venv_ready(venv_python):
            print("  [gliner2-gpu] Preparing native GPU venv (one-time: downloads "
                  "torch + the gliner2 model, ~2GB)...")
            pull_rc = subprocess.run([sys.executable, str(PARSER_SERVICE_PY), "pull"]).returncode
            if pull_rc == PARSER_SERVICE_REQUIREMENTS_EXIT:
                print("  [gliner2-gpu] Host does not meet native GPU serving "
                      "requirements (see checks above). The gliner2 enrich stage will "
                      "be UNAVAILABLE (caber-core installs the gliner2 client only).")
                return False
            if pull_rc != 0:
                print("  [gliner2-gpu] WARNING: `parser_service.py pull` failed. "
                      "The gliner2 enrich stage will be UNAVAILABLE (caber-core "
                      "installs the gliner2 client only).")
                return False
            if not _native_venv_ready(venv_python):
                print("  [gliner2-gpu] WARNING: native venv still incomplete after "
                      "pull. The gliner2 enrich stage will be UNAVAILABLE (caber-core "
                      "installs the gliner2 client only).")
                return False
        # Clear any stale pid file (a previous crash leaves the launcher refusing to
        # start), then launch. On macOS force Apple MPS: the launcher's device
        # auto-detect imports torch, which is absent from THIS interpreter, so it
        # would otherwise resolve to CPU.
        subprocess.run([sys.executable, str(PARSER_SERVICE_PY), "stop"])
        start_cmd = [sys.executable, str(PARSER_SERVICE_PY), "start"]
        if platform.system() == "Darwin":
            start_cmd += ["--device", "mps"]
        print(f"  [gliner2-gpu] Starting native model server (Apple MPS) on "
              f"127.0.0.1:{GLINER2_GPU_PORT} ...")
        start_env = {**os.environ, "CSI_GLINER2_MODEL_API_KEY": api_key}
        start_rc = subprocess.run(start_cmd, env=start_env).returncode
        if start_rc == PARSER_SERVICE_REQUIREMENTS_EXIT:
            print("  [gliner2-gpu] Host does not meet native GPU serving "
                  "requirements (see checks above). The gliner2 enrich stage will be "
                  "UNAVAILABLE (caber-core installs the gliner2 client only).")
            return False
        if start_rc != 0 or not _gliner2_gpu_healthy():
            print("  [gliner2-gpu] WARNING: model server did not become healthy. "
                  "The gliner2 enrich stage will be UNAVAILABLE (caber-core installs "
                  "the gliner2 client only).")
            return False

    os.environ["CSI_GLINER2_MODEL_URL"] = container_url
    os.environ["CSI_GLINER2_MODEL_API_KEY"] = api_key
    os.environ["CSI_GLINER2_HOST_URL"] = f"http://127.0.0.1:{GLINER2_GPU_PORT}/gliner-2"
    os.environ.setdefault("PURPOSE_CLASSIFIER_URL", f"{container_url}/gliner-2")
    print(f"  [gliner2-gpu] Container gliner2 -> {container_url} (Mac GPU)")
    return True


def require_gliner2_gpu_server():
    """Start the model server and ABORT if NER is configured on without it.

    caber-core installs the gliner2 CLIENT ONLY; gliner2_runner.load_model raises
    without CSI_GLINER2_MODEL_URL, and WorkerSupervisorParseClient tolerantly SKIPS
    an enrich stage whose runner is unavailable. caber-core IS the parsing service
    now (WS8 flipped shared/compose/caber-platform.yaml to it), so continuing past
    a missing model server would yield a green demo that indexes ZERO named
    entities -- a silent false success. A hard abort is the honest outcome; run
    structure-only by setting DocParser.gliner.enabled=false instead."""
    if ensure_gliner2_gpu_server() or not gliner2_gpu_requested():
        return
    print(
        "ERROR: NER is enabled (DocParser.gliner.enabled=true, engine=gliner2) but\n"
        "       the native GLiNER2 model server is not reachable. caber-core is the\n"
        "       parsing service: without the model server the gliner2 enrich stage is\n"
        "       silently skipped and the demo indexes ZERO named entities. Fix the\n"
        "       model server (see the [gliner2-gpu] output above), or set\n"
        "       DocParser.gliner.enabled=false in shared/config/config-base.json to\n"
        "       run structure-only. Aborting.",
        file=sys.stderr,
    )
    sys.exit(1)


def gliner2_gpu_server_survived():
    """Re-probe the model server AFTER a run; True iff NER was actually available.

    `require_gliner2_gpu_server()` is a BRING-UP gate and cannot see a server that
    dies mid-run -- and dying mid-run produces the exact silent false success that
    gate exists to prevent: WorkerSupervisorParseClient tolerantly SKIPS an enrich
    stage whose runner has become unreachable, per document, so the run stays green
    while indexing ZERO named entities from the point of death onward. Observed
    here: the native server was killed partway through a `ci` run and the run still
    reported success. So the honest verdict needs a check at BOTH ends.

    A False here means the entity output of the run just completed is untrustworthy
    -- not necessarily empty, but truncated at an unknown document.
    """
    if not gliner2_gpu_requested():
        return True
    if _gliner2_gpu_healthy():
        return True
    print(
        "ERROR: the native GLiNER2 model server was reachable at start-up but is NOT\n"
        "       reachable now, so it died during this run. caber-core skips an enrich\n"
        "       stage whose runner is unavailable, which means every document parsed\n"
        "       after it died was indexed with ZERO named entities while the run kept\n"
        "       reporting success. Treat this run's entity output as INVALID. See the\n"
        f"       server log ({PARSER_SERVICE_PY.parent / '.parser_service_state' / 'gliner2.log'})\n"
        "       and re-run once the server stays up.",
        file=sys.stderr,
    )
    return False


def stop_gliner2_gpu_server():
    """Stop the native gliner2 GPU model server. No-op unless
    `gliner2_gpu_requested()` and the launcher is present."""
    if not gliner2_gpu_requested() or not PARSER_SERVICE_PY.exists():
        return
    print("  [gliner2-gpu] Stopping native model server...")
    subprocess.run([sys.executable, str(PARSER_SERVICE_PY), "stop"])


# --- docling-serve native GPU sidecar automation (Plan C section 2.5) -------
# When DOCLING_GPU_ENABLE=1, `up`/`ci` start PyCaber's native docling-serve
# sidecar (parser_service.py --service docling) on the HOST (Apple MPS or
# CUDA) and point the caber-core container at it via CSI_DOCLING_ENDPOINT.
# UNLIKE gliner2, this has NO in-container CPU fallback by policy (spec
# section 2.5): docling in a container with no real GPU and no attested
# remote endpoint reports the capability Unavailable and rejects parses at
# request time (Plan C Tasks 2/3's guards). So on ANY failure here we leave
# CSI_DOCLING_ENDPOINT unset and let those guards produce the loud failure --
# we do NOT silently degrade to CPU serving.
DOCLING_GPU_PORT = 8766
DOCLING_GPU_ATTEST_URL = f"http://127.0.0.1:{DOCLING_GPU_PORT}/caber/attest"
DOCLING_GPU_CONTAINER_URL = f"http://host.docker.internal:{DOCLING_GPU_PORT}"
# Same pid-file convention as gliner2's (parser_service.pid_file_path), just
# named for the "docling" service.
DOCLING_LAUNCHER_PID_FILE = (
    PARSER_SERVICE_PY.parent / ".parser_service_state" / "docling.pid"
)


def docling_gpu_requested():
    """The docling GPU sidecar automation is armed only via an explicit
    operator opt-in -- unlike gliner2 there is no config-derived signal to
    gate on, since docling parsing itself is unconditionally attempted by
    the parser; DOCLING_GPU_ENABLE is the operator's explicit request for
    native GPU serving instead of accepting the no-CPU-fallback failure."""
    return _truthy(os.environ.get("DOCLING_GPU_ENABLE", ""))


def _docling_gpu_attest():
    """Probe /caber/attest; return the parsed JSON payload on HTTP 200, else
    None. Never raises (unreachable/non-200 are both treated as unhealthy)."""
    try:
        with urllib.request.urlopen(DOCLING_GPU_ATTEST_URL, timeout=2) as resp:
            if resp.status == 200:
                return json.loads(resp.read())
    except Exception:
        pass
    return None


def _docling_gpu_device_ok(payload):
    """True iff the attest payload reports a real GPU device. A CPU (or
    missing/unknown) device is refused -- no CPU fallback by policy."""
    return isinstance(payload, dict) and payload.get("device") in ("mps", "cuda")


def _docling_server_is_launcher_managed():
    """True iff the supported launcher (parser_service.py) currently owns a
    live docling-serve process -- i.e. its pid file exists and names a
    running process. A healthy attest with NO such record is a FOREIGN
    server a developer may have hand-started."""
    try:
        pid = int(DOCLING_LAUNCHER_PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _docling_native_venv_ready(venv_python):
    """True iff the launcher's native venv exists AND can import the
    docling-serve deps. Guards against a PARTIAL venv (e.g. a `pull` that
    died mid-install on a full disk) being mistaken for a ready one."""
    if not venv_python.exists():
        return False
    try:
        return subprocess.run(
            [str(venv_python), "-c", "import docling, fastapi, uvicorn"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except Exception:
        return False


DOCLING_GPU_UNAVAILABLE_ERROR = (
    "ERROR: docling GPU serving unavailable -- docling parses in containers "
    "will FAIL LOUDLY (no CPU fallback by policy)"
)


def ensure_docling_gpu_server():
    """Idempotently start the native GPU (Apple MPS / CUDA) docling-serve
    sidecar on the host and export CSI_DOCLING_ENDPOINT so the container
    routes docling to it.

    No-op unless `docling_gpu_requested()`. On ANY failure (launcher missing,
    pull failed, attest unreachable or reporting a non-GPU device, start
    failed) prints DOCLING_GPU_UNAVAILABLE_ERROR and leaves
    CSI_DOCLING_ENDPOINT unset -- there is deliberately NO in-container CPU
    fallback (spec section 2.5); the container-side guards (Plan C Tasks 2/3)
    turn the missing endpoint into a loud parse-time rejection instead of a
    silent CPU degrade. Returns True iff the container will use the GPU
    sidecar."""
    if not docling_gpu_requested():
        return False

    container_url = os.environ.get("CSI_DOCLING_ENDPOINT") or DOCLING_GPU_CONTAINER_URL

    attest = _docling_gpu_attest()
    if attest is not None and _docling_gpu_device_ok(attest):
        if _docling_server_is_launcher_managed():
            print(f"  [docling-gpu] Native docling-serve sidecar already running on "
                  f"127.0.0.1:{DOCLING_GPU_PORT} (device={attest.get('device')})")
        else:
            # /caber/attest answered but the supported launcher does not own
            # this process: a foreign server is on the port. Do not silently
            # trust it -- it was not launched with the pinned docling-serve
            # triple this project relies on.
            print(f"  [docling-gpu] WARNING: a docling-serve sidecar is answering on "
                  f"127.0.0.1:{DOCLING_GPU_PORT} but was NOT started by the supported "
                  f"launcher (scripts/parser_service.py). Stop that process and re-run "
                  f"so demo.py manages the supported sidecar "
                  f"(`python {PARSER_SERVICE_PY} stop --service docling`), or ensure it "
                  f"was started via parser_service.py.")
    else:
        if not PARSER_SERVICE_PY.exists():
            print(DOCLING_GPU_UNAVAILABLE_ERROR)
            print(f"  [docling-gpu] launcher not found at {PARSER_SERVICE_PY} "
                  f"(set CABER_PYCABER_ROOT).")
            return False
        # One-time: pip-install docling-serve into the (shared) native venv
        # and warm-download docling's models. `pull` is idempotent but
        # re-pips on every call, so only run it when the venv is missing or
        # incomplete.
        venv_python = (PARSER_SERVICE_PY.parent / ".parser_service_state" /
                       "native_venv" / "bin" / "python")
        if not _docling_native_venv_ready(venv_python):
            print("  [docling-gpu] Preparing native docling-serve venv (one-time: "
                  "downloads docling-serve + models)...")
            pull_rc = subprocess.run(
                [sys.executable, str(PARSER_SERVICE_PY), "pull", "--service", "docling"]
            ).returncode
            if pull_rc != 0:
                print(DOCLING_GPU_UNAVAILABLE_ERROR)
                print("  [docling-gpu] `parser_service.py pull --service docling` failed.")
                return False
            if not _docling_native_venv_ready(venv_python):
                print(DOCLING_GPU_UNAVAILABLE_ERROR)
                print("  [docling-gpu] native venv still incomplete after pull.")
                return False
        # Clear any stale pid file (a previous crash leaves the launcher
        # refusing to start), then launch.
        subprocess.run([sys.executable, str(PARSER_SERVICE_PY), "stop", "--service", "docling"])
        # Force the device rather than letting parser_service.py's
        # resolve_launch_device() auto-detect: this outer interpreter (the
        # demo.py process) has no torch installed, so auto-detection would
        # resolve to "cpu" and start_docling would self-refuse even on a host
        # with a real GPU. parser_service's start_docling re-verifies actual
        # availability via the native venv's torch and fail-loud-refuses if
        # the forced device isn't really present, so forcing here is safe and
        # preserves the no-CPU-fallback property (mirrors gliner2's identical
        # forcing at ensure_gliner2_gpu_server, demo.py:257-259).
        start_cmd = [sys.executable, str(PARSER_SERVICE_PY), "start", "--service", "docling"]
        if platform.system() == "Darwin":
            start_cmd += ["--device", "mps"]
        elif platform.system() == "Linux":
            start_cmd += ["--device", "cuda"]
        print(f"  [docling-gpu] Starting native docling-serve sidecar on "
              f"127.0.0.1:{DOCLING_GPU_PORT} ...")
        start_rc = subprocess.run(start_cmd).returncode
        attest = _docling_gpu_attest()
        if start_rc != 0 or attest is None or not _docling_gpu_device_ok(attest):
            print(DOCLING_GPU_UNAVAILABLE_ERROR)
            print("  [docling-gpu] sidecar did not become healthy on a real GPU device "
                  f"(attest={attest!r}).")
            return False

    os.environ["CSI_DOCLING_ENDPOINT"] = container_url
    print(f"  [docling-gpu] Container docling -> {container_url} "
          f"(device={attest.get('device')})")
    return True


def stop_docling_gpu_server():
    """Stop the native docling-serve GPU sidecar. No-op unless
    `docling_gpu_requested()` and the launcher is present."""
    if not docling_gpu_requested() or not PARSER_SERVICE_PY.exists():
        return
    print("  [docling-gpu] Stopping native docling-serve sidecar...")
    subprocess.run([sys.executable, str(PARSER_SERVICE_PY), "stop", "--service", "docling"])


# The canonical platform compose file. It is the SINGLE source of truth for
# which csi-* services exist and which of them demo.py can build -- do not
# re-list them here. A hand-maintained PLATFORM_BUILDS dict used to duplicate
# it and drifted: csi-link-data was retired from the compose file while the
# dict kept naming it, so `docker compose build ... csi-link-data` aborted the
# whole platform build with "no such service: csi-link-data" and none of the
# surviving containers got built either.
PLATFORM_COMPOSE = ROOT / "shared" / "compose" / "caber-platform.yaml"

# Infra fragments composed ahead of PLATFORM_COMPOSE for `build --platform`.
# Compose `build` pulls none of their images, but the first --file sets the
# compose project directory that relative paths resolve from.
# infra-versitygw.yaml lives in shared/compose/ like every other fragment here,
# so that directory is shared/compose/ either way.
PLATFORM_BUILD_INFRA_FILES = [
    ROOT / "shared" / "compose" / "infra-versitygw.yaml",
    ROOT / "shared" / "compose" / "infra-elasticsearch.yaml",
]

# Platform images that are NOT built from the compose file, keyed by image name
# with a ${CSI_IMAGE_TAG} placeholder. WS8 cutover: caber-core (the single
# merged Rust ingest container that replaced csi-process-api / csi-s3-scanner /
# csi-kafka-scanner) is prebuilt by PyCaber scripts/local_build.sh -g caber_core,
# so it appears in compose with no build: block.
PLATFORM_PREBUILT_IMAGES = ["csimvp-caber-core:{tag}"]


def platform_build_services():
    """Platform services `demo.py build --platform` knows how to build.

    Derived from PLATFORM_COMPOSE: a service is buildable exactly when that
    file still gives it a `build:` block. Services retired from compose, or
    supplied as prebuilt images, drop out automatically instead of aborting the
    build. The build args live in the compose file too -- compose reads them
    straight from there, so nothing needs to mirror them.
    """
    if not HAS_YAML:
        print(f"  ERROR: PyYAML is required to read {PLATFORM_COMPOSE}")
        return []
    if not PLATFORM_COMPOSE.exists():
        print(f"  ERROR: Compose file not found at {PLATFORM_COMPOSE}")
        return []
    compose = yaml.safe_load(PLATFORM_COMPOSE.read_text()) or {}
    services = compose.get("services") or {}
    return [name for name, service in services.items()
            if isinstance(service, dict) and service.get("build")]


def platform_required_images(image_tag):
    """Platform images a demo needs present before it can start."""
    built = [f"csimvp-{service}:{image_tag}" for service in platform_build_services()]
    return built + [image.format(tag=image_tag) for image in PLATFORM_PREBUILT_IMAGES]


def load_catalog():
    """Load catalog from YAML (preferred) or JSON fallback."""
    if HAS_YAML and CATALOG_YAML.exists():
        with open(CATALOG_YAML) as f:
            return yaml.safe_load(f)
    elif CATALOG_JSON.exists():
        with open(CATALOG_JSON) as f:
            return json.load(f)
    else:
        print("ERROR: No catalog found. Need catalog.yaml (+ PyYAML) or catalog.json")
        sys.exit(1)


def load_buckets(demo_name):
    """Load bucket definitions from demos/<demo>/buckets.yaml.

    Returns a dict of logical_name -> {name, env, config_path, ...} or
    empty dict if the file doesn't exist or is empty.
    """
    buckets_file = ROOT / "demos" / demo_name / "buckets.yaml"
    if not buckets_file.exists():
        return {}
    if not HAS_YAML:
        print(f"  WARNING: PyYAML required to read {buckets_file}")
        return {}
    with open(buckets_file) as f:
        data = yaml.safe_load(f)
    return data or {}


def prepare_bucket_env(buckets, demo_name=None):
    """Build env var overrides from bucket definitions.

    Returns a dict with:
      - DEMO_BUCKETS: comma-separated list of all bucket names
      - Any per-bucket env vars defined in buckets.yaml (e.g. DEMO_BUCKET, OBJECTSTORE_S3_BUCKET)
    """
    return bucket_environment(ROOT, demo_name or "", buckets)


def prepare_bucket_env_for_demo(demo_name):
    """Load a demo's buckets and build its object-store environment."""
    return prepare_bucket_env(load_buckets(demo_name), demo_name)


def validate_config_buckets(demo_name, buckets):
    """Validate that config.json bucket arrays and Regex.customer_bucket match buckets.yaml.

    Returns a list of (level, message) tuples where level is 'WARN' or 'FAIL'.
    """
    config_file = ROOT / "demos" / demo_name / "config.json"
    if not config_file.exists():
        return []
    try:
        with open(config_file) as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return [("WARN", f"Could not parse config.json: {e}")]

    issues = []

    # Collect expected bucket names grouped by config_path
    path_to_names = {}
    all_bucket_names = set()
    for _key, spec in buckets.items():
        bname = spec.get("name", "")
        cpath = spec.get("config_path", "")
        if bname and cpath:
            path_to_names.setdefault(cpath, set()).add(bname)
        if bname:
            all_bucket_names.add(bname)

    # Check Object_Sources.*.buckets arrays
    obj_sources = config.get("Object_Sources", {})
    for cpath, expected_names in path_to_names.items():
        parts = cpath.split(".")
        if len(parts) >= 2 and parts[0] == "Object_Sources":
            source_key = parts[1]
            source = obj_sources.get(source_key, {})
            actual_buckets = set(source.get("buckets", []))
            missing = expected_names - actual_buckets
            extra = actual_buckets - expected_names
            if missing:
                issues.append(("WARN", f"config.json {source_key}.buckets missing: {', '.join(sorted(missing))}"))
            if extra:
                issues.append(("WARN", f"config.json {source_key}.buckets has extra: {', '.join(sorted(extra))}"))

    # Check Regex.customer_bucket pattern
    regex_section = config.get("Regex", {})
    customer_pattern = regex_section.get("customer_bucket", "")
    if customer_pattern and all_bucket_names:
        # Pattern looks like "(bucket1|bucket2)" -- extract names from alternation
        inner = customer_pattern.strip("()")
        pattern_names = set(inner.split("|")) if inner else set()
        missing = all_bucket_names - pattern_names
        extra = pattern_names - all_bucket_names
        if missing:
            issues.append(("WARN", f"Regex.customer_bucket missing buckets: {', '.join(sorted(missing))}"))
        if extra:
            issues.append(("WARN", f"Regex.customer_bucket has extra buckets: {', '.join(sorted(extra))}"))

    return issues


def resolve_compose_files(demo_cfg):
    """Resolve compose file paths relative to the repo root.

    When running on EC2 (CABER_EC2_SESSION=true), automatically appends the
    ECR overlay after caber-platform.yaml so session-tagged images are pulled
    from ECR instead of the local Docker cache.
    """
    files = []
    ecr_overlay = ROOT / "shared" / "compose" / "caber-platform-ecr.yaml"
    on_ec2 = os.environ.get("CABER_EC2_SESSION", "").lower() == "true"

    for f in demo_cfg.get("compose_files", []):
        path = ROOT / f
        if not path.exists():
            print(f"WARNING: compose file not found: {path}")
        files.append(str(path))

        # Insert ECR overlay immediately after caber-platform*.yaml
        if on_ec2 and ecr_overlay.exists() and "caber-platform" in f and "ecr" not in f:
            files.append(str(ecr_overlay))

    return files


def compose_files_for(demo_cfg, with_ner):
    """The demo's compose files as Paths, with the NER overlay last when asked.

    `with_ner` is the `--with-ner` flag of `up` and `ci`. The overlay
    (runner.core.catalog.NER_OVERLAY) belongs to agentic-tools, which ships
    without NER; callers refuse the flag for any other demo (see
    `ner_overlay_refused`). `ci` reaches the same list through
    runner.core.docker_lifecycle, keyed on runner.core.catalog.WITH_NER_ENV.
    """
    files = [Path(f) for f in resolve_compose_files(demo_cfg)]
    if with_ner:
        files.append(NER_OVERLAY)
    return files


def ner_overlay_refused(demo, with_ner):
    """Error text when `--with-ner` is given for a demo that has no NER overlay."""
    if with_ner and demo != NER_OVERLAY_DEMO:
        return (
            f"--with-ner applies only to {NER_OVERLAY_DEMO}; {demo} configures NER "
            "in its own compose files and config."
        )
    return None


def verdict_without_ner_refused(demo, verdict, with_ner):
    """Error text when `ci --verdict` targets agentic-tools without `--with-ner`.

    The verdict gate needs the NER pass, and the agentic-tools base stack ships
    without it (ruling P2-14: refused, not implied).
    """
    if demo == NER_OVERLAY_DEMO and verdict and not with_ner:
        return (
            f"the {NER_OVERLAY_DEMO} base stack ships without the purpose classifier "
            "and the verdict gate needs the NER pass; add --with-ner."
        )
    return None


def host_ner_server_required(demo, with_ner):
    """Whether launching `demo` needs the host GLiNER2 model server.

    agentic-tools needs it only with `--with-ner`, because its base stack ships
    without NER. Every other demo keeps the configured-NER check of
    `require_gliner2_gpu_server()` unchanged.
    """
    return with_ner or demo != NER_OVERLAY_DEMO


def check_port_conflicts(demo_cfg, catalog):
    """Check if any ports required by this demo are already in use."""
    ports = demo_cfg.get("ports", {})
    if not ports:
        return []

    conflicts = []
    for svc, port in ports.items():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                result = s.connect_ex(("127.0.0.1", int(port)))
                if result == 0:
                    conflicts.append((svc, port))
        except OSError:
            pass

    return conflicts


def compose_cmd(files, action_args, env=None):
    """Build and run a docker compose command."""
    file_args = []
    for f in files:
        file_args.extend(["-f", f])
    cmd = ["docker", "compose"] + file_args + action_args
    print(f"  Running: {' '.join(cmd)}")
    return subprocess.run(cmd, env=env or os.environ)


# mock-idp local fallback (until the GHCR image is published): shared with
# runner.core.docker_lifecycle so non-locked `demo.py ci` applies it too.
from runner.core.mock_idp_local import (  # noqa: E402,F401
    MOCK_IDP_DEMOS,
    MOCK_IDP_IDENTITY_DIR,
    MOCK_IDP_LOCAL_DIR,
    MOCK_IDP_LOCAL_REPO,
    mock_idp_local_tag,
    mock_idp_pin_is_placeholder,
    mock_idp_source_root,
    prepare_mock_idp_local_fallback,
    write_mock_idp_local_overlay,
)


def format_limits_status(status_payload):
    """Format preview limits status from platform status payload.

    Args:
        status_payload: Dict parsed from /v1/status or similar status payload.

    Returns:
        tuple[str, bool]: (status_line, is_ok)
    """
    if not isinstance(status_payload, dict):
        return ("[OK]   Preview limits: not configured", True)

    limits = status_payload.get("limits")
    if not limits or not isinstance(limits, dict):
        return ("[OK]   Preview limits: not configured", True)

    events = limits.get("events")
    if not isinstance(events, dict):
        return ("[OK]   Preview limits: not configured", True)

    state = events.get("state")
    count = events.get("count")
    max_val = events.get("max") or limits.get("values", {}).get("max_events")
    reset_instructions = (
        limits.get("reset_instructions")
        or events.get("reset_instructions")
        or ""
    ).strip()

    if state == "ok":
        return (f"[OK]   Preview limits: {count} of {max_val} events", True)
    elif state == "approaching":
        return (f"[WARN] Preview limits: {count} of {max_val} events approaching ceiling", True)
    elif state == "reached":
        prefix = f"[FAIL] Preview limits: ceiling reached ({count} of {max_val})."
        msg = f"{prefix} {reset_instructions}".rstrip()
        return (msg, False)
    elif state == "unknown":
        return ("[WARN] Preview limits: count unavailable", True)
    else:
        return ("[OK]   Preview limits: not configured", True)


def fetch_platform_status(url=None):
    """Fetch status payload from platform /v1/status. Returns parsed dict or None."""
    if not url:
        host = os.environ.get("CSI_CABER_HOST", "http://localhost:8050")
        url = f"{host.rstrip('/')}/v1/status"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        pass
    return None



# ── Commands ──────────────────────────────────────────────────────────

def cmd_list(args, catalog):
    """List all available demos."""
    demos = catalog.get("demos", {})
    print(f"\n{'Name':<28} {'Category':<18} {'Dashboard':<10} {'AI Features'}")
    print("-" * 80)
    for name, cfg in demos.items():
        ai = ", ".join(cfg.get("ai_features", [])) or "-"
        dash = "yes" if cfg.get("dashboard") else "-"
        cat = cfg.get("category", "")
        print(f"{name:<28} {cat:<18} {dash:<10} {ai}")
    print()


def cmd_info(args, catalog):
    """Show detailed info about a demo."""
    demos = catalog.get("demos", {})
    cfg = demos.get(args.demo)
    if not cfg:
        print(f"Unknown demo: {args.demo}")
        print(f"Available: {', '.join(demos.keys())}")
        return 1

    print(f"\n  {cfg.get('name', args.demo)}")
    print(f"  {cfg.get('description', '')}\n")
    print(f"  Category:       {cfg.get('category', '-')}")
    print(f"  Dashboard:      {'yes' if cfg.get('dashboard') else 'no'}")
    print(f"  AI Features:    {', '.join(cfg.get('ai_features', [])) or '-'}")
    print(f"  Startup Time:   {cfg.get('estimated_startup', 'unknown')}")
    print(f"  Docker Memory:  {cfg.get('docker_memory_gb', '?')} GB minimum")

    req_env = cfg.get("required_env", [])
    if req_env:
        print(f"  Required Env:   {', '.join(req_env)}")

    ports = cfg.get("ports", {})
    if ports:
        print(f"\n  Ports:")
        for svc, port in ports.items():
            print(f"    {svc:<30} :{port}")

    compose = cfg.get("compose_files", [])
    if compose:
        print(f"\n  Compose Files:")
        for f in compose:
            print(f"    {f}")
    print()
    return 0


def _run_seed(demo_name, visible=True, slow_mo=300, keep_open=True):
    """Run demos/<demo>/seed.py if it exists. Returns exit code or None."""
    seed_py = ROOT / "demos" / demo_name / "seed.py"
    if not seed_py.exists():
        return None

    seed_args = [sys.executable, str(seed_py)]
    if visible:
        seed_args.append("--visible")
    if slow_mo:
        seed_args.extend(["--slow-mo", str(slow_mo)])
    if keep_open:
        seed_args.append("--keep-open")
    from runner.core.generated_data import resolve_generated_data
    gen_overrides = resolve_generated_data(demo_name)
    if gen_overrides and gen_overrides.get("_generated_data_dir"):
        seed_args.extend(["--data-dir", gen_overrides["_generated_data_dir"]])

    result = subprocess.run(seed_args, cwd=str(ROOT))
    return result.returncode


def _start_dashboard_ui():
    """Start the CaberUserConsole Next.js dev server on the host."""
    if not DASHBOARD_UI_DIR.is_dir():
        print(f"  WARNING: CaberUserConsole not found at {DASHBOARD_UI_DIR}")
        return False
    if not (DASHBOARD_UI_DIR / "node_modules").is_dir():
        print(f"  WARNING: node_modules missing in {DASHBOARD_UI_DIR}, run 'npm install' first")
        return False

    # Kill any leftover process from a previous demo.py run
    _stop_dashboard_ui(quiet=True)

    # If an external `next dev` is already serving :3000, reuse it instead of
    # spawning a second instance that binds to :3001 and wipes `.next/server/`.
    from runner.core.health import wait_for_url
    if wait_for_url("http://localhost:3000", timeout_s=2, interval_s=1):
        print("  + Dashboard UI already running on http://localhost:3000 (reusing)")
        return True

    log_path = ROOT / ".dashboard-ui.log"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(DASHBOARD_UI_DIR),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    DASHBOARD_UI_PID.write_text(str(proc.pid))
    print(f"  + Dashboard UI started (pid {proc.pid}, log: {log_path.name})")
    print(f"    Open http://localhost:3000 once it's ready")
    return True


def _stop_dashboard_ui(quiet=False):
    """Stop the host-side dashboard UI process."""
    import signal
    if not DASHBOARD_UI_PID.exists():
        return
    try:
        pid = int(DASHBOARD_UI_PID.read_text().strip())
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        if not quiet:
            print(f"  Dashboard UI stopped (pid {pid})")
    except (ProcessLookupError, ValueError, PermissionError):
        if not quiet:
            print(f"  Dashboard UI process already gone")
    finally:
        DASHBOARD_UI_PID.unlink(missing_ok=True)


def cmd_up(args, catalog):
    """Launch a demo: compose up, wait for health, seed data."""
    print_preview_edition_notice()
    demos = catalog.get("demos", {})
    cfg = demos.get(args.demo)
    if not cfg:
        print(f"Unknown demo: {args.demo}")
        return 1

    with_ner = getattr(args, "with_ner", False)
    refused = ner_overlay_refused(args.demo, with_ner)
    if refused:
        print(f"ERROR: {refused}")
        return 1
    files = [str(f) for f in compose_files_for(cfg, with_ner)]
    if not files:
        print("ERROR: No compose files configured for this demo.")
        return 1

    # Select and validate the first working LLM provider (ANTHROPIC, NVIDIA,
    # GEMINI, OPENAI, GROQ) and populate OPENAI_API_KEY / OPENAI_BASE_URL /
    # USE_LLM_MODEL before checking required_env or launching containers.
    ensure_llm_provider()

    # Check required env vars
    missing = []
    for var in cfg.get("required_env", []):
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        print(f"\n  WARNING: Required environment variables not set: {', '.join(missing)}")
        print(f"  Copy .env.example and set them, or export them in your shell.\n")

    # A CI run that died hard can leave its 2 s heartbeat snippet behind. It is
    # CI-only, so a non-CI launch removes it before the stack reads snippets.
    from runner.core.ci_fast_pickup import host_snippet_dir, sweep_ci_snippet
    sweep_ci_snippet(host_snippet_dir())

    from runner.core.docker_lifecycle import provision_demo_artifacts
    artifact_result = provision_demo_artifacts(args.demo)
    if artifact_result is not None and artifact_result.returncode != 0:
        print("  ERROR: Required demo artifacts could not be provisioned.")
        return artifact_result.returncode

    # gliner2 models-only GPU serving: start the host Apple-MPS model server and
    # export CSI_GLINER2_MODEL_URL before run_env snapshots os.environ. No-op unless
    # DocParser.gliner is enabled with engine gliner2 (GLINER_ENABLE/GLINER_ENGINE
    # override); aborts if that NER config is on but the server is unreachable.
    # Left running for `up`; `down` stops it. agentic-tools ships without NER,
    # so it needs the server only with --with-ner.
    if host_ner_server_required(args.demo, with_ner):
        require_gliner2_gpu_server()

    # docling-serve native GPU sidecar (Plan C section 2.5): opt-in via
    # DOCLING_GPU_ENABLE=1. Unlike gliner2 there is NO CPU fallback -- on
    # failure CSI_DOCLING_ENDPOINT is left unset and the container-side
    # guards (Plan C Tasks 2/3) fail parses loudly rather than silently
    # degrading. Left running for `up`; `down` stops it.
    ensure_docling_gpu_server()

    # Check for port conflicts
    conflicts = check_port_conflicts(cfg, catalog)
    if conflicts:
        print(f"\n  PORT CONFLICTS DETECTED:")
        for svc, port in conflicts:
            print(f"    :{port} ({svc}) is already in use")
        print(f"\n  Another demo or process may be using these ports.")
        print(f"  Run 'demo.py down <other-demo>' first, or stop the conflicting process.")
        confirm = input("  Continue anyway? [y/N] ").strip().lower()
        if confirm != "y":
            print("  Aborted.")
            return 1

    # A demo shipping a frozen corpus seeds it unless a factory corpus was asked
    # for. Before the bucket env: activation changes which buckets resolve.
    if not _use_frozen_corpus(args.demo, explicit_dataset=getattr(args, "dataset", None)):
        return 1

    # Load bucket definitions and build env overrides
    bucket_env = prepare_bucket_env_for_demo(args.demo)
    run_env = dict(os.environ, **bucket_env) if bucket_env else None
    if bucket_env:
        print(f"\n  Buckets: {bucket_env.get('DEMO_BUCKETS', '')}")
        for var, val in bucket_env.items():
            if var != "DEMO_BUCKETS":
                print(f"    {var}={val}")

    # Resolve dataset: CLI --dataset flag > catalog default_dataset
    dataset_name = getattr(args, "dataset", None) or cfg.get("default_dataset")
    if dataset_name:
        dataset_result = _dataset_env(args.demo, dataset_name)
        if dataset_result:
            dataset_env_vars, _data_mount = dataset_result  # mount path; compose file wires the actual bind mount
            run_env = dict(run_env or os.environ, **dataset_env_vars)
            print(f"\n  Dataset: {dataset_name} -> env vars injected (mount wired in compose file)")

    # Check for factory-generated data
    from runner.core.generated_data import resolve_generated_data
    gen_overrides = resolve_generated_data(args.demo)
    if gen_overrides:
        print(f"\n  [generate] Using factory-generated data")
        generated_bucket_env = {
            key: value for key, value in gen_overrides.items()
            if key in {"DEMO_BUCKETS", "OBJECT_STORE_BUCKETS_JSON"}
        }
        if generated_bucket_env:
            run_env = dict(run_env or os.environ, **generated_bucket_env)
        from runner.core.docker_lifecycle import _generated_parser_cache_env
        cache_env = _generated_parser_cache_env(gen_overrides.get("_parser_cache_dir"))
        if cache_env:
            run_env = dict(run_env or os.environ, **cache_env)
        compose_override = gen_overrides.get("_compose_override")
        if compose_override and Path(compose_override).exists():
            files.append(compose_override)

    # `up` builds any missing image, so the mock-idp service needs the same local mock-idp
    # overlay `build` uses while the pinned digest is still the placeholder.
    try:
        mock_idp_overlay = prepare_mock_idp_local_fallback(args.demo, env=run_env)
    except RuntimeError as exc:
        print(f"\n  ERROR: {exc}")
        return 1
    if mock_idp_overlay:
        files.append(str(mock_idp_overlay))

    # Step 1: docker compose up
    print(f"\n  [compose] Launching: {cfg.get('name', args.demo)}")
    result = compose_cmd(files, ["up", "-d", "--pull", "never"], env=run_env)
    if result.returncode != 0:
        out = (getattr(result, "stdout", None) or "") + "\n" + (getattr(result, "stderr", None) or "")
        ecr_imgs = [img for img in compose_pull_images(files, env=run_env) if is_ecr_image(img)]
        explain_aws_or_ecr_auth_failure(out, ecr_images=ecr_imgs, env=run_env, probe_if_empty=True)
        return result.returncode

    # Step 1b: start dashboard UI on host if requested
    if args.dashboard:
        _start_dashboard_ui()

    # Step 2: wait for health
    healthcheck = cfg.get("healthcheck_url")
    if healthcheck:
        from runner.core.health import wait_for_url
        est = cfg.get("estimated_startup", "a few minutes")
        print(f"  [health]  Waiting for services (est. {est})...")
        ok = wait_for_url(healthcheck, timeout_s=360, interval_s=10)
        if not ok:
            print(f"  WARNING: Health check timed out: {healthcheck}")

    # Step 3: auto-seed
    if not args.no_seed:
        visible = not args.no_visible
        keep_open = not args.no_keep_open
        slow_mo = args.slow_mo
        seed_rc = _run_seed(args.demo, visible=visible, slow_mo=slow_mo,
                            keep_open=keep_open)
        if seed_rc is not None and seed_rc != 0:
            print(f"  WARNING: Seed exited with code {seed_rc}")

    # Print service URLs
    ports = cfg.get("ports", {})
    if ports:
        print(f"\n  Services available at:")
        for svc, port in ports.items():
            print(f"    {svc:<30} http://localhost:{port}")

    print(f"\n  Preview edition: limits (20,000 events, 500 objects, 5 SDK instances) and reset instructions at {PREVIEW_LIMITS_URL}")

    return 0


def cmd_down(args, catalog):
    """Tear down a demo."""
    demos = catalog.get("demos", {})
    cfg = demos.get(args.demo)
    if not cfg:
        print(f"Unknown demo: {args.demo}")
        return 1

    files = resolve_compose_files(cfg)

    # Stop host-side dashboard UI if it was running
    _stop_dashboard_ui()

    print(f"\n  Tearing down: {cfg.get('name', args.demo)}")
    result = compose_cmd(files, ["down", "-v"], env=_demo_env(args.demo))
    from runner.core.docker_lifecycle import clear_shared_storage_volume
    clear_shared_storage_volume()

    # Stop the host gliner2 GPU model server started by `up`/`ci` (no-op otherwise).
    stop_gliner2_gpu_server()
    # Stop the host docling-serve GPU sidecar started by `up`/`ci` (no-op otherwise).
    stop_docling_gpu_server()
    return result.returncode


def _demo_env(demo_name):
    """Load bucket env vars for a demo, returns env dict or None."""
    bucket_env = prepare_bucket_env_for_demo(demo_name)
    return dict(os.environ, **bucket_env) if bucket_env else None


def _dataset_env(demo_name, dataset_name, repo_root=None):
    """Resolve dataset path and return (env_overrides, bind_mount_str) or None.

    Args:
      demo_name: reserved for future per-demo dataset path overrides; currently unused.
      dataset_name: name of the subdirectory under data/ to resolve.
      repo_root: optional override for the repo root path; defaults to ROOT.

    Returns:
      (env_overrides dict, bind_mount_str) if dataset_name is valid and directory exists.
      None if dataset_name is falsy or directory not found.

    env_overrides keys: DEMO_DATASET, DEMO_DATA_DIR (/demo-data), DEMO_DATA_DIR_HOST (abs host path)
    bind_mount_str: "<abs_path>:/demo-data:ro" for future use in compose override files
    """
    if not dataset_name:
        return None
    root = Path(repo_root) if repo_root else ROOT
    dataset_path = root / "data" / dataset_name
    if not dataset_path.is_dir():
        print(f"  ERROR: Dataset not found: {dataset_path}")
        return None
    env_overrides = {
        "DEMO_DATASET": dataset_name,
        "DEMO_DATA_DIR": "/demo-data",
        "DEMO_DATA_DIR_HOST": str(dataset_path),
    }
    return env_overrides, f"{dataset_path}:/demo-data:ro"


def log_corpus_choice(chosen, reason, displaced=None, displaced_reason=None):
    """Emit the one line that says which corpus a run will seed, and why.

    Why this exists as its own function
    -----------------------------------
    Seeding the wrong corpus is the defect this line was written for: the run
    generated ``data/sensitive-docs/generated`` and ingested
    ``data/agentgov-docs`` instead, and nothing in the output said so. Every
    later symptom (an enforcement gate that cannot pass) pointed at governance.

    So the choice is stated as a FACT WITH ITS SOURCE, in one greppable line,
    at the point the choice is made, in every branch -- including the boring
    ones. A line that only appears when something unusual happens cannot be used
    to rule the usual case in or out.

    When one candidate displaces another, the loser is named too. "Chose A" and
    "chose A over B" answer different questions, and the second is the one asked
    after a surprise.

    Returns the emitted line, so tests can assert its content rather than
    re-deriving it.
    """
    line = f"  [corpus] seeding from {chosen} (source: {reason})"
    if displaced:
        line += (f"; OVERRIDES {displaced}"
                 f" ({displaced_reason or 'previously selected'})")
    print(line)
    return line


def _use_frozen_corpus(demo_name, **choice):
    """Activate the demo's committed corpus snapshot when the run should seed it.

    The decision is ``generated_data.choose_frozen_corpus``; this only logs it
    and reports failure. Returns False when the snapshot was chosen but does
    not verify, because seeding something other than the committed bytes
    would be a silent substitution.
    """
    from runner.core.generated_data import frozen_corpus_dir, use_frozen_corpus_if_chosen
    if frozen_corpus_dir(demo_name) is None:
        return True
    try:
        materialized, reason = use_frozen_corpus_if_chosen(demo_name, **choice)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  [corpus] ERROR: frozen corpus for {demo_name!r} unusable: {exc}")
        return False
    if materialized is None:
        print(f"\n  [corpus] frozen snapshot not used: {reason}")
    else:
        log_corpus_choice(materialized, reason)
    return True


def _sensitive_corpus_dataset_conflict(explicit_dataset):
    """Return an error message when ``--dataset`` and ``--verdict`` disagree.

    ``--verdict`` generates the sensitive_documents corpus and seeds THAT
    corpus: the verdict gate scores the deployed NER pass against its manifest,
    and the policy question map targets regions inside it. So an explicit
    ``--dataset`` naming a different corpus cannot be honoured, and both silent
    resolutions are traps: honouring it seeds one corpus while scoring another,
    and overriding it discards an explicit operator instruction.

    The catalog's ``default_dataset`` is NOT a conflict: it is a default, not an
    instruction, and the caller overrides it loudly (see
    :func:`_sensitive_corpus_env`).

    Returns:
        None when there is no conflict, otherwise the operator-facing message.
    """
    if not explicit_dataset:
        return None
    return (
        f"--dataset {explicit_dataset!r} cannot be combined with --verdict.\n"
        f"  --verdict generates and seeds the sensitive_documents corpus "
        f"(data/sensitive-docs/generated): its manifest is the ground truth the "
        f"verdict gate scores, and the policy question map targets regions "
        f"inside it.\n"
        f"  Drop --dataset to run over the sensitive corpus, or drop --verdict "
        f"to run {explicit_dataset!r}."
    )


def _generate_verdict_conflict(generate, verdict):
    """Return an error message when ``--generate`` and ``--verdict`` are combined.

    ``--generate`` writes the factory corpus under ``data/agentgov-docs/generated``
    and arms the IDF shape gate over its ``ground_truth.json``. ``--verdict``
    pins and seeds ``data/sensitive-docs/generated`` instead. Together, the
    shape gate would score a corpus the run never seeded, and its recall failure
    would read as an IDF regression.
    """
    if not (generate and verdict):
        return None
    return (
        "--generate cannot be combined with --verdict.\n"
        "  --generate arms the IDF shape gate over the factory corpus "
        "(data/agentgov-docs/generated), but --verdict seeds the "
        "sensitive_documents corpus (data/sensitive-docs/generated), so the "
        "gate would score documents this run never ingested.\n"
        "  Run them separately: `python demo.py ci <demo> --generate` and "
        "`python demo.py ci <demo> --verdict`."
    )


def _sensitive_corpus_via_dataset(explicit_dataset, verdict, repo_root=None):
    """Return an error message when ``--dataset`` names the sensitive corpus.

    ``--verdict`` is the one route to the sensitive corpus: it regenerates it,
    pins it (``DEMO_CORPUS_PINNED``) and enables the NER pass whose labels the
    SSN gate and the verdict gate read. ``--dataset`` pointing at the same
    directory would seed it with none of that, so the e2e sees the corpus
    seeded while the verdict pass is off. Refused with a pointer to
    ``--verdict`` rather than given a second, differently-behaving route.

    With ``--verdict`` set this returns None:
    :func:`_sensitive_corpus_dataset_conflict` already refuses any explicit
    ``--dataset`` in that case.
    """
    if not explicit_dataset or verdict:
        return None
    from runner.core.generated_data import sensitive_corpus_dir
    root = Path(repo_root) if repo_root else ROOT
    candidate = root / "data" / explicit_dataset
    sensitive_root = root / "data" / "sensitive-docs"
    try:
        same = candidate.resolve() == sensitive_corpus_dir(root).resolve()
    except OSError:
        same = False
    inside = candidate == sensitive_root or sensitive_root in candidate.parents
    if not (same or inside):
        return None
    return (
        f"--dataset {explicit_dataset!r} names the sensitive_documents corpus.\n"
        f"  That corpus is reached through --verdict, which regenerates it, pins "
        f"it and enables the NER pass the SSN gate and the verdict gate depend "
        f"on. Seeding it via --dataset would do none of that.\n"
        f"  Use: python demo.py ci <demo> --verdict"
    )


def _sensitive_corpus_env(corpus_dir):
    """Env overrides that point every corpus consumer at the sensitive corpus.

    ``run_generate_sensitive`` writes the corpus and activates it behind
    ``data/sensitive-docs/generated``; nothing else in the run knows that. Four
    consumers have to be told, and they are told through env because that is the
    channel docker compose interpolation, the host seed script and the CI
    coverage gate already share:

      ``DEMO_DATA_DIR_HOST``  the host path ``demos/<demo>/seed.py`` reads when
                              it uploads the corpus into the demo's buckets.
                              Repointing it means the seeded corpus is the one
                              the graph was built over rather than a stale one.
      ``DEMO_DATASET``        naming, for the run banner and artifacts.
      ``DEMO_DATA_DIR``       the container-side path, unchanged from the
                              ``--dataset`` path so nothing has to special-case it.
      ``DEMO_CORPUS_PINNED``  the run has chosen its corpus; do not let an
                              activated corpus elsewhere replace it. Without
                              this, ``resolve_generated_data`` adopts whatever
                              ``data/agentgov-docs/generated`` points at, passes
                              it to ``seed.py`` as ``--data-dir`` (which
                              outranks ``DEMO_DATA_DIR_HOST``) and swaps
                              csi-init's config for the generated one -- so the
                              sensitive corpus would be neither seeded nor
                              configured for.

    The path is resolved through the activation symlink deliberately: docker
    bind-mount sources are resolved by the daemon, and a symlink there is a
    portability question nobody should have to think about mid-demo.

    Returns:
        (env_overrides, error). ``error`` is non-None when the corpus is absent,
        which is fatal: the generator reported success, so a missing directory
        means the corpus was written somewhere this function cannot see and
        seeding would silently fall back to the default dataset.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        return None, (
            f"sensitive corpus not found at {corpus_dir}.\n"
            f"  Generation reported success, so this means the corpus was "
            f"written somewhere else. Seeding would fall back to "
            f"data/agentgov-docs (the amy/bob corpus), and the verdict gate "
            f"and every SSN assertion would be scored against documents that "
            f"carry no SSN, failing as though the pipeline were broken."
        )
    resolved = corpus_dir.resolve()
    return {
        "DEMO_DATASET": "sensitive-docs/generated",
        "DEMO_DATA_DIR": "/demo-data",
        "DEMO_DATA_DIR_HOST": str(resolved),
        "DEMO_CORPUS_PINNED": "1",
    }, None


PYCABER_WHEEL_DISTRIBUTIONS = (
    "pycaber",
    "pycaber_connector",
    "pycaber_graph",
    "pycaber_ingest",
    "pycaber_parser",
    "pycaber_wrap",
)

# Distributions PyCaber does not require in every release.
# - pycaber_connector: standalone connector SDK, only built when needed
# - pycaber_graph: retired distribution (govern_retrieval moved into pycaber, 2026-11)
# - pycaber_ingest: retired distribution (ingest moved into pure-Rust caber-core)
# - pycaber_wrap: retired distribution (folded into pycaber, 2026-11)
# They stay in PYCABER_WHEEL_DISTRIBUTIONS so stale copies are still recognised and pruned
# from a wheelhouse, but their absence must not condemn a release as incomplete.
PYCABER_OPTIONAL_DISTRIBUTIONS = frozenset({
    "pycaber_connector",
    "pycaber_graph",  # retired distribution: govern_retrieval moved into pycaber (2026-11); pruned, never required
    "pycaber_ingest",
    "pycaber_wrap",  # retired distribution: folded into pycaber (2026-11); pruned from stale wheelhouses, never required
})

PYCABER_REQUIRED_DISTRIBUTIONS = tuple(
    distribution
    for distribution in PYCABER_WHEEL_DISTRIBUTIONS
    if distribution not in PYCABER_OPTIONAL_DISTRIBUTIONS
)


def _pycaber_wheel_identity(path):
    """Return the managed distribution and version encoded in a wheel name."""
    name = path.name
    for distribution in sorted(PYCABER_WHEEL_DISTRIBUTIONS, key=len, reverse=True):
        prefix = f"{distribution}-"
        if name.startswith(prefix) and name.endswith(".whl"):
            return distribution, name[len(prefix):].split("-", 1)[0]
    return None


def _latest_complete_pycaber_wheels(src):
    """Select every managed wheel belonging to the newest complete release."""
    wheels = {}
    top_level_versions = set()
    for path in src.glob("*.whl"):
        identity = _pycaber_wheel_identity(path)
        if identity is None:
            continue
        wheels.setdefault(identity, []).append(path)
        if identity[0] == "pycaber":
            top_level_versions.add(identity[1])

    if not top_level_versions:
        return None, ("pycaber",)

    try:
        version = max(
            top_level_versions,
            key=lambda value: tuple(int(part) for part in value.split(".")),
        )
    except ValueError:
        return None, ("numeric PyCaber version",)

    missing = tuple(
        distribution
        for distribution in PYCABER_REQUIRED_DISTRIBUTIONS
        if (distribution, version) not in wheels
    )
    if missing:
        return None, missing

    selected = []
    for distribution in PYCABER_WHEEL_DISTRIBUTIONS:
        matching = wheels.get((distribution, version), ())
        if matching:
            selected.extend(matching)

    return tuple(selected), ()


def _sync_wheelhouse(src=None, dst=None):
    """Mirror only the newest complete PyCaber release into a wheelhouse."""

    src = ROOT / "wheelhouse" if src is None else src
    dst = MVP_ROOT / "Deploy" / "DockerComposeFiles" / "wheelhouse" if dst is None else dst
    dst.mkdir(parents=True, exist_ok=True)

    if not src.is_dir():
        print(f"  WARNING: wheelhouse not found at {src}")
        return False

    selected, missing = _latest_complete_pycaber_wheels(src)
    if selected is None:
        print(
            f"  WARNING: newest PyCaber wheel set in {src} is incomplete "
            f"(missing: {', '.join(missing)}); leaving {dst} unchanged"
        )
        return False

    copied = 0
    selected_names = {whl.name for whl in selected}
    for whl in selected:
        target = dst / whl.name
        source_stat = whl.stat()
        target_stat = target.stat() if target.exists() else None
        if (
            target_stat is None
            or target_stat.st_size != source_stat.st_size
            or target_stat.st_mtime_ns != source_stat.st_mtime_ns
        ):
            shutil.copy2(whl, target)
            copied += 1

    pruned = 0
    for whl in dst.glob("*.whl"):
        if _pycaber_wheel_identity(whl) is not None and whl.name not in selected_names:
            whl.unlink()
            pruned += 1

    if copied or pruned:
        print(f"  Synced {copied} and pruned {pruned} PyCaber wheel(s) in {dst}")
    return True


_DAEMON_NETWORK_ERROR = re.compile(
    r"context deadline exceeded|DeadlineExceeded|Client\.Timeout|i/o timeout|"
    r"TLS handshake timeout|proxyconnect|connection refused|connection reset|"
    r"no such host|network is unreachable|failed to fetch oauth token|: EOF",
    re.IGNORECASE,
)
_NETWORK_PROBE_IMAGE = "docker.io/docker/dockerfile:1"


def _explain_daemon_network_failure():
    """After a failed build, check whether the Docker daemon can reach a registry.

    The build output streams straight to the terminal, so instead of parsing it
    this pulls the BuildKit frontend every build needs first. If that pull fails
    with a network error, the build failed because Docker Desktop's VM cannot get
    out (its internal proxy http.docker.internal:3128 is stuck), not because of
    anything in the Dockerfile, and we print how to fix it.
    """
    print(f"\n  Checking whether the Docker daemon can reach Docker Hub "
          f"(docker pull {_NETWORK_PROBE_IMAGE}) ...")
    try:
        probe = subprocess.run(["docker", "pull", _NETWORK_PROBE_IMAGE],
                               capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        output = "docker pull timed out after 90s"
    else:
        if probe.returncode == 0:
            print("  Docker daemon network is OK; the build failed for another reason (see output above).")
            return
        output = (probe.stdout or "") + (probe.stderr or "")
        if not _DAEMON_NETWORK_ERROR.search(output) and "timed out" not in output:
            print(f"  Probe pull failed, but not with a network error:\n    {output.strip()}")
            return

    print(f"""
  The Docker daemon cannot reach the network:
    {output.strip()}

  Docker Desktop routes all daemon traffic (builds, pulls, logins) through an
  internal proxy (http.docker.internal:3128 / 192.168.65.1:3128). When that proxy
  gets stuck, usually after sleep, a Wi-Fi change, or a VPN connecting or
  disconnecting, every build fails like this even though the Mac itself is online.

  To fix:
    1. Confirm the Mac itself is online (an HTTP 401 means the registry is reachable):
         curl -sS -o /dev/null -w '%{{http_code}}\\n' https://registry-1.docker.io/v2/
    2. Restart Docker Desktop (whale menu > Restart, or: docker desktop restart).
       This stops all running containers, including any running demo.
    3. Confirm the daemon can get out again:
         docker pull {_NETWORK_PROBE_IMAGE}
    4. If step 3 still fails, check Docker Desktop > Settings > Resources > Proxies,
       and toggle any VPN or network security tool, then repeat step 3.
    5. Re-run the same demo.py build command.
""")


_COMPOSE_VAR_RE = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-((?:(?!\$\{)[^}])*)|-((?:(?!\$\{)[^}])*))?\}")
# One nesting level is all the compose files use; the cap only bounds a malformed value.
_COMPOSE_VAR_MAX_PASSES = 10


def _expand_compose_vars(value, env=None):
    """Expand Docker Compose ${VAR}, ${VAR:-default}, and ${VAR-default} syntax.

    A default may itself hold a ``${...}`` (``${A:-${B:-c}}``), as docker compose v2
    allows: the innermost reference matches first (a default may not contain
    ``${``), so each pass expands one level until no reference is left.
    """
    if not isinstance(value, str) or "${" not in value:
        return value
    lookup = env if env is not None else os.environ

    def _replace(match):
        var, colon_default, dash_default = match.group(1), match.group(2), match.group(3)
        if colon_default is not None:
            val = lookup.get(var)
            return val if val else colon_default
        if dash_default is not None:
            return lookup[var] if var in lookup else dash_default
        return lookup.get(var, "")

    for _ in range(_COMPOSE_VAR_MAX_PASSES):
        expanded = _COMPOSE_VAR_RE.sub(_replace, value)
        if expanded == value:
            break
        value = expanded
        if "${" not in value:
            break
    return value


def _normalize_image_ref(image):
    """Ensure a Docker image reference includes an explicit tag (default :latest)."""
    if "@" in image:
        return image
    last_part = image.rsplit("/", 1)[-1]
    if ":" not in last_part:
        return f"{image}:latest"
    return image


def _active_compose_profiles(env=None):
    """Compose profiles that are switched on, read from COMPOSE_PROFILES the way
    `docker compose` reads it (comma-separated; blanks ignored)."""
    raw = (env if env is not None else os.environ).get("COMPOSE_PROFILES", "")
    return {p.strip() for p in raw.split(",") if p.strip()}


def compose_pull_images(compose_files, env=None, services=None):
    """Return deduplicated non-buildable service image refs across compose_files.

    Follows Compose profile rules: a service that declares `profiles:` is part of
    the project only when one of its profiles is active (COMPOSE_PROFILES, or "*"
    for all) or when it is named explicitly in `services`. Without this, an
    opt-in service such as agentic-tools' `caber-console` (profile `console`) was
    pulled on every build.
    """
    if not HAS_YAML:
        return []
    merged = {}
    for path in compose_files:
        p = Path(path)
        if not p.exists():
            continue
        doc = yaml.safe_load(p.read_text()) or {}
        for name, svc in (doc.get("services") or {}).items():
            if not isinstance(svc, dict):
                continue
            entry = merged.setdefault(name, {})
            if "image" in svc:
                entry["image"] = svc["image"]
            if "build" in svc:
                entry["build"] = svc["build"]
            if "profiles" in svc:
                entry["profiles"] = svc["profiles"]

    allowed = set(services) if services is not None else None
    active = _active_compose_profiles(env)
    images = []
    seen = set()
    for name, svc in merged.items():
        if allowed is not None and name not in allowed:
            continue
        profiles = set(svc.get("profiles") or [])
        if profiles and allowed is None and "*" not in active and not (profiles & active):
            continue
        if svc.get("build"):
            continue
        raw_image = svc.get("image")
        if not raw_image:
            continue
        image = _expand_compose_vars(str(raw_image).strip(), env)
        norm = _normalize_image_ref(image)
        if norm not in seen:
            seen.add(norm)
            images.append(norm)
    return images


def _expected_pycaber_version():
    """Return the expected PyCaber version from local wheelhouse or checkout, or None."""
    wheelhouse = ROOT / "wheelhouse"
    if wheelhouse.is_dir():
        selected, _ = _latest_complete_pycaber_wheels(wheelhouse)
        if selected:
            identity = _pycaber_wheel_identity(selected[0])
            if identity:
                return identity[1]
    version, _ = _pycaber_version_from_checkout()
    return version


def _caber_core_image_is_stale(image_ref, expected_version=None):
    """Return True if a local caber-core image has an OCI version label that mismatches expected_version."""
    repo = image_ref.split("@", 1)[0].split(":", 1)[0]
    if repo != "caber-core" and not repo.endswith("/caber-core"):
        return False
    if expected_version is None:
        expected_version = _expected_pycaber_version()
    if not expected_version:
        return False
    result = subprocess.run(
        [
            "docker", "image", "inspect",
            "--format", '{{index .Config.Labels "org.opencontainers.image.version"}}',
            image_ref,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False
    local_version = result.stdout.strip()
    if local_version and local_version != "<no value>" and local_version != expected_version:
        print(
            f"  Local {image_ref} has version {local_version} "
            f"(expected {expected_version}); marking for pull."
        )
        return True
    return False


def _local_docker_images():
    """Return the set of locally cached Docker image repo:tag references.

    Excludes any local caber-core image whose `org.opencontainers.image.version`
    label mismatches the expected PyCaber version so mutable tags like `:demo`
    are re-pulled when wheels advance.
    """
    result = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return set()
    images = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    expected_version = None
    stale = set()
    for img in images:
        repo = img.split("@", 1)[0].split(":", 1)[0]
        if repo == "caber-core" or repo.endswith("/caber-core"):
            if expected_version is None:
                expected_version = _expected_pycaber_version()
            if _caber_core_image_is_stale(img, expected_version=expected_version):
                stale.add(img)
    return images - stale


def _pull_missing_images(compose_files, env=None, services=None):
    """Pull any non-buildable service images in compose_files that are missing locally."""
    needed = compose_pull_images(compose_files, env=env, services=services)
    if not needed:
        return 0
    available = _local_docker_images()
    missing = [img for img in needed if img not in available]
    if not missing:
        print(f"  All {len(needed)} required non-build image(s) already present locally.")
        return 0

    print(f"\n  Pulling {len(missing)} missing image(s): {', '.join(missing)}")
    for img in missing:
        cmd = ["docker", "pull", img]
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"\n  Failed to pull image: {img}")
            out = (getattr(result, "stdout", None) or "") + "\n" + (getattr(result, "stderr", None) or "")
            if not explain_aws_or_ecr_auth_failure(out, image=img, env=env, probe_if_empty=True):
                _explain_daemon_network_failure()
            return result.returncode
    return 0


def _build_platform(image_tag, services=None):
    """Build csi-* platform containers via docker compose build (legacy builder).

    Uses shared/compose/caber-platform.yaml with COMPOSE_DOCKER_CLI_BUILD=0
    to share layer cache across services and improve network resilience
    (matches deploy.py behaviour). Also pulls any missing non-buildable
    platform/infra images.
    """
    if not MVP_ROOT.is_dir():
        print(f"  ERROR: mvp repo not found at {MVP_ROOT}")
        return 1

    compose_file = PLATFORM_COMPOSE
    dockerfile = ROOT / "shared" / "docker" / "caber-service.dockerfile"
    if not compose_file.exists():
        print(f"  ERROR: Compose file not found at {compose_file}")
        return 1
    if not dockerfile.exists():
        print(f"  ERROR: Dockerfile not found at {dockerfile}")
        return 1

    local_src = ROOT / "wheelhouse"
    pycaber_src = PYCABER_ROOT / "wheelhouse" / "cloud-arm64"
    mvp_dst = MVP_ROOT / "Deploy" / "DockerComposeFiles" / "wheelhouse"

    _sync_wheelhouse(pycaber_src, local_src)
    _sync_wheelhouse(local_src, mvp_dst)

    infra_files = [
        ROOT / "shared" / "compose" / "infra-versitygw.yaml",
        ROOT / "shared" / "compose" / "infra-elasticsearch.yaml",
    ]
    all_files = infra_files + [compose_file]

    env = os.environ.copy()
    env["COMPOSE_DOCKER_CLI_BUILD"] = "0"
    env["CSI_IMAGE_TAG"] = image_tag
    env["PLATFORM_BUILD_CONTEXT"] = str(MVP_ROOT)
    env["PLATFORM_DOCKERFILE"] = str(dockerfile)

    buildable = platform_build_services()
    build_services = list(buildable)
    if services:
        pullable = compose_pull_images(all_files, env=env, services=services)
        unknown = [
            s for s in services
            if s not in buildable and not compose_pull_images(all_files, env=env, services=[s])
        ]
        if unknown:
            print(f"  WARNING: Not buildable from {compose_file.name} "
                  f"(retired, or a prebuilt image): {', '.join(unknown)}")
        build_services = [s for s in services if s in buildable]
        if not build_services and not pullable:
            print("  No matching platform services to build.")
            return 1

    pull_rc = _pull_missing_images(all_files, env=env, services=services)
    if pull_rc != 0:
        return pull_rc

    if not build_services:
        return 0

    print(f"\n  Building platform containers: {', '.join(build_services)}")
    print(f"  Image tag: {image_tag}")

    file_args = []
    for f in PLATFORM_BUILD_INFRA_FILES + [compose_file]:
        file_args.extend(["--file", str(f)])

    cmd = [
        "docker", "compose",
    ] + file_args + [
        "--project-name", "csimvp",
        "build", "--force-rm",
    ] + build_services

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\n  Platform build FAILED")
        out = (getattr(result, "stdout", None) or "") + "\n" + (getattr(result, "stderr", None) or "")
        ecr_imgs = [img for img in compose_pull_images(all_files, env=env, services=services) if is_ecr_image(img)]
        if not explain_aws_or_ecr_auth_failure(out, ecr_images=ecr_imgs, env=env, probe_if_empty=True):
            _explain_daemon_network_failure()
        return 1

    print(f"\n  All platform containers built with tag: {image_tag}")
    return 0


def cmd_build(args, catalog):
    """Build containers for a demo and/or platform services."""
    image_tag = os.environ.get("CSI_IMAGE_TAG", "demo")
    rc = 0

    # Build platform csi-* containers
    if args.platform:
        services = [s.strip() for s in args.service.split(",")] if args.service else None
        rc = _build_platform(image_tag, services)
        if rc != 0:
            return rc

    # Build demo app containers (if demo specified)
    if args.demo:
        demos = catalog.get("demos", {})
        cfg = demos.get(args.demo)
        if not cfg:
            print(f"Unknown demo: {args.demo}")
            return 1
        if not args.platform:
            local_src = ROOT / "wheelhouse"
            pycaber_src = PYCABER_ROOT / "wheelhouse" / "cloud-arm64"
            mvp_dst = MVP_ROOT / "Deploy" / "DockerComposeFiles" / "wheelhouse"
            _sync_wheelhouse(pycaber_src, local_src)
            if MVP_ROOT.is_dir():
                _sync_wheelhouse(local_src, mvp_dst)
        files = resolve_compose_files(cfg)
        build_env = _demo_env(args.demo) or dict(os.environ)
        build_env["COMPOSE_DOCKER_CLI_BUILD"] = "0"
        build_env["PLATFORM_BUILD_CONTEXT"] = str(MVP_ROOT)
        build_env["PLATFORM_DOCKERFILE"] = str(ROOT / "shared" / "docker" / "caber-service.dockerfile")
        try:
            mock_idp_overlay = prepare_mock_idp_local_fallback(args.demo, env=build_env)
        except RuntimeError as exc:
            print(f"\n  ERROR: {exc}")
            return 1
        if mock_idp_overlay:
            files.append(str(mock_idp_overlay))
        pull_rc = _pull_missing_images(files, env=build_env)
        if pull_rc != 0:
            return pull_rc
        result = compose_cmd(files, ["build", "--force-rm"], env=build_env)
        if result.returncode != 0:
            print(f"\n  Demo build FAILED")
            out = (getattr(result, "stdout", None) or "") + "\n" + (getattr(result, "stderr", None) or "")
            ecr_imgs = [img for img in compose_pull_images(files, env=build_env) if is_ecr_image(img)]
            if not explain_aws_or_ecr_auth_failure(out, ecr_images=ecr_imgs, env=build_env, probe_if_empty=True):
                _explain_daemon_network_failure()
            rc = result.returncode

    if not args.platform and not args.demo:
        print("  Specify --platform, a demo name, or both.")
        return 1

    return rc


def cmd_test(args, catalog):
    """Run smoke tests for a demo."""
    print_preview_edition_notice()
    ensure_llm_provider()
    # Look for pytest test file matching the demo name
    test_dir = ROOT / "tests" / "smoke"
    # Map demo names to test files
    test_name = args.demo.replace("-", "_")
    test_file = test_dir / f"test_{test_name}.py"
    if not test_file.exists():
        # Try generic match (e.g., agentgov-* -> test_agentgov.py)
        prefix = args.demo.split("-")[0]
        test_file = test_dir / f"test_{prefix}.py"
    if not test_file.exists():
        print(f"No smoke test found for {args.demo}")
        print(f"  Looked in: {test_dir}")
        return 1
    print(f"\n  Running smoke tests: {test_file.name}")
    result = subprocess.run([sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"])
    return result.returncode


def parse_ttl(ttl_str):
    """Parse a TTL string like '4h', '30m', '1d' into minutes."""
    match = re.match(r"^(\d+)([mhd])$", ttl_str.strip().lower())
    if not match:
        return None
    value, unit = int(match.group(1)), match.group(2)
    if unit == "m":
        return value
    elif unit == "h":
        return value * 60
    elif unit == "d":
        return value * 60 * 24
    return None


def schedule_ttl_teardown(demo_name, ttl_minutes, tf_dir, tfvars):
    """Schedule auto-teardown via AWS EventBridge + SSM Run Command.

    Creates a one-shot EventBridge rule that triggers at the expiry time.
    The target is an SSM Automation document that runs terraform destroy.
    """
    try:
        import json as _json
    except ImportError:
        pass

    expiry = datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)
    cron_expr = f"cron({expiry.minute} {expiry.hour} {expiry.day} {expiry.month} ? {expiry.year})"
    rule_name = f"caber-ttl-{demo_name}-{int(expiry.timestamp())}"

    print(f"\n  Scheduling auto-teardown:")
    print(f"    Rule:    {rule_name}")
    print(f"    Expires: {expiry.strftime('%Y-%m-%d %H:%M UTC')} ({ttl_minutes} min from now)")

    # Create EventBridge rule
    result = subprocess.run([
        "aws", "events", "put-rule",
        "--name", rule_name,
        "--schedule-expression", cron_expr,
        "--state", "ENABLED",
        "--description", f"Auto-teardown for {demo_name} demo deployment"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        err_text = (getattr(result, "stderr", None) or "").strip()
        out = (getattr(result, "stdout", None) or "") + "\n" + err_text
        print(f"  [WARN] Could not create EventBridge rule: {err_text}")
        explain_aws_or_ecr_auth_failure(out)
        print(f"         TTL not set. Remember to run: demo.py undeploy {demo_name} --target aws")
        return False

    rule_arn = ""
    try:
        rule_info = json.loads(result.stdout)
        rule_arn = rule_info.get("RuleArn", "")
    except (json.JSONDecodeError, KeyError):
        pass

    # Write a local marker file so undeploy can clean up the rule
    ttl_marker = tf_dir / ".ttl-rules" / f"{demo_name}.json"
    ttl_marker.parent.mkdir(parents=True, exist_ok=True)
    with open(ttl_marker, "w") as f:
        json.dump({
            "rule_name": rule_name,
            "rule_arn": rule_arn,
            "demo": demo_name,
            "expires_utc": expiry.isoformat(),
            "tfvars": str(tfvars),
        }, f, indent=2)

    print(f"    Marker:  {ttl_marker}")
    return True



def cmd_deploy(args, catalog):
    """Deploy a demo to AWS using Terraform (or destroy with --destroy)."""
    if getattr(args, "destroy", False):
        return cmd_undeploy(args, catalog)

    demos = catalog.get("demos", {})
    cfg = demos.get(args.demo)
    if not cfg:
        print(f"Unknown demo: {args.demo}")
        return 1

    tf_dir = ROOT / "cloud" / "terraform"
    # Map demo names to profile files
    profile_map = {
        "agentgov-mimic": "agentgov",
    }
    profile_name = profile_map.get(args.demo, args.demo)
    tfvars = tf_dir / "profiles" / f"{profile_name}.tfvars"

    if not tfvars.exists():
        print(f"No Terraform profile found: {tfvars}")
        return 1

    print(f"\n  Deploying {cfg.get('name', args.demo)} to AWS")
    print(f"  Profile: {tfvars.name}")
    print(f"  Terraform dir: {tf_dir}\n")

    # Init (with backend key per demo)
    init_result = subprocess.run(
        ["terraform", "init", f"-backend-config=key=demos/{args.demo}/terraform.tfstate"],
        cwd=str(tf_dir)
    )
    if init_result.returncode != 0:
        out = (getattr(init_result, "stdout", None) or "") + "\n" + (getattr(init_result, "stderr", None) or "")
        explain_aws_or_ecr_auth_failure(out, probe_sts_if_empty=True)
        return init_result.returncode

    # Plan
    plan_result = subprocess.run(
        ["terraform", "plan", f"-var-file={tfvars}"],
        cwd=str(tf_dir)
    )
    if plan_result.returncode != 0:
        out = (getattr(plan_result, "stdout", None) or "") + "\n" + (getattr(plan_result, "stderr", None) or "")
        explain_aws_or_ecr_auth_failure(out, probe_sts_if_empty=True)
        return plan_result.returncode

    # Prompt before apply
    confirm = input("\n  Apply this plan? [y/N] ").strip().lower()
    if confirm != "y":
        print("  Aborted.")
        return 0

    apply_result = subprocess.run(
        ["terraform", "apply", f"-var-file={tfvars}", "-auto-approve"],
        cwd=str(tf_dir)
    )
    if apply_result.returncode != 0:
        out = (getattr(apply_result, "stdout", None) or "") + "\n" + (getattr(apply_result, "stderr", None) or "")
        explain_aws_or_ecr_auth_failure(out, probe_sts_if_empty=True)
        return apply_result.returncode

    # Schedule TTL teardown if requested
    ttl_str = getattr(args, "ttl", None)
    if ttl_str:
        ttl_minutes = parse_ttl(ttl_str)
        if ttl_minutes:
            schedule_ttl_teardown(args.demo, ttl_minutes, tf_dir, tfvars)
        else:
            print(f"  [WARN] Invalid TTL format '{ttl_str}'. Use e.g. 4h, 30m, 1d")

    return 0


def cmd_undeploy(args, catalog):
    """Tear down an AWS deployment using Terraform."""
    demos = catalog.get("demos", {})
    cfg = demos.get(args.demo)
    if not cfg:
        print(f"Unknown demo: {args.demo}")
        return 1

    tf_dir = ROOT / "cloud" / "terraform"
    profile_map = {
        "agentgov-mimic": "agentgov",
    }
    profile_name = profile_map.get(args.demo, args.demo)
    tfvars = tf_dir / "profiles" / f"{profile_name}.tfvars"

    print(f"\n  Undeploying {cfg.get('name', args.demo)} from AWS")

    confirm = input("  Destroy all resources? [y/N] ").strip().lower()
    if confirm != "y":
        print("  Aborted.")
        return 0

    result = subprocess.run(
        ["terraform", "destroy", f"-var-file={tfvars}", "-auto-approve"],
        cwd=str(tf_dir)
    )
    if result.returncode != 0:
        out = (getattr(result, "stdout", None) or "") + "\n" + (getattr(result, "stderr", None) or "")
        explain_aws_or_ecr_auth_failure(out, probe_sts_if_empty=True)

    # Clean up any TTL EventBridge rule
    ttl_marker = tf_dir / ".ttl-rules" / f"{args.demo}.json"
    if ttl_marker.exists():
        try:
            with open(ttl_marker) as f:
                ttl_info = json.load(f)
            rule_name = ttl_info.get("rule_name", "")
            if rule_name:
                print(f"  Removing TTL rule: {rule_name}")
                del_res = subprocess.run(
                    ["aws", "events", "delete-rule", "--name", rule_name, "--force"],
                    capture_output=True, text=True
                )
                if del_res.returncode != 0:
                    del_out = (getattr(del_res, "stdout", None) or "") + "\n" + (getattr(del_res, "stderr", None) or "")
                    explain_aws_or_ecr_auth_failure(del_out)
            ttl_marker.unlink()
        except (json.JSONDecodeError, OSError):
            pass

    return result.returncode


def cmd_check(args, catalog):
    """Verify prerequisites for running demos."""
    ok = True
    warns = 0

    # Check for a specific demo if provided via --demo (optional positional)
    demo_name = getattr(args, "demo", None)
    demo_cfg = catalog.get("demos", {}).get(demo_name) if demo_name else None

    print("\n  Prerequisites Check")
    print("  " + "-" * 40)

    # Docker
    docker = shutil.which("docker")
    if docker:
        result = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  [OK]   Docker {result.stdout.strip()}")
        else:
            print("  [FAIL] Docker daemon not running")
            ok = False
    else:
        print("  [FAIL] Docker not found in PATH")
        ok = False

    # Docker Compose
    result = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  [OK]   {result.stdout.strip()}")
    else:
        print("  [FAIL] docker compose not available")
        ok = False

    # Docker memory
    if docker:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.MemTotal}}"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            try:
                mem_bytes = int(result.stdout.strip())
                mem_gb = mem_bytes / (1024 ** 3)
                needed = demo_cfg.get("docker_memory_gb", 10) if demo_cfg else 10
                if mem_gb >= needed:
                    print(f"  [OK]   Docker memory: {mem_gb:.1f} GB (need {needed} GB)")
                else:
                    print(f"  [WARN] Docker memory: {mem_gb:.1f} GB (need {needed} GB)")
                    print(f"         Increase Docker Desktop memory in Settings > Resources")
                    warns += 1
            except (ValueError, TypeError):
                pass

    # PyYAML
    if HAS_YAML:
        print(f"  [OK]   PyYAML available")
    else:
        print(f"  [WARN] PyYAML not installed (using JSON fallback)")
        warns += 1

    # Check for pre-built csi images (tag comes from CSI_IMAGE_TAG in .env)
    csi_tag = os.environ.get("CSI_IMAGE_TAG", "dream")
    # Derived from the platform compose file (see platform_build_services) so a
    # service retired from compose stops being reported as a missing image that
    # `demo.py build --platform` could never produce.
    required_images = platform_required_images(csi_tag)
    result = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                            capture_output=True, text=True)
    available = set(result.stdout.strip().split("\n")) if result.returncode == 0 else set()

    missing_images = [img for img in required_images if img not in available]
    if missing_images:
        print(f"  [WARN] Missing pre-built images ({len(missing_images)}):")
        for img in missing_images:
            print(f"         {img}")
        print("         Build with: demo.py build --platform")
        warns += 1
    else:
        print(f"  [OK]   All {len(required_images)} required platform images present")

    # Platform status / preview limits check
    dashboard_url = None
    if demo_cfg:
        port = demo_cfg.get("ports", {}).get("csi-dashboard")
        if port:
            dashboard_url = f"http://localhost:{port}/v1/status"
    status_payload = fetch_platform_status(dashboard_url)
    if status_payload is None:
        print("  [WARN] Preview limits: platform not running")
        warns += 1
    else:
        limits_line, limits_ok = format_limits_status(status_payload)
        print(f"  {limits_line}")
        if not limits_ok:
            ok = False
        elif limits_line.startswith("[WARN]"):
            warns += 1

    # Per-demo checks
    if demo_cfg:
        print(f"\n  Demo: {demo_cfg.get('name', demo_name)}")
        print("  " + "-" * 40)

        # Resolve first working LLM provider before checking required_env
        if "OPENAI_API_KEY" in demo_cfg.get("required_env", []):
            ensure_llm_provider()

        # Required env vars
        missing_env = [v for v in demo_cfg.get("required_env", []) if not os.environ.get(v)]
        if missing_env:
            print(f"  [WARN] Missing required env vars: {', '.join(missing_env)}")
            print(f"         Copy .env.example and export them, or set in your shell")
            warns += 1
        elif demo_cfg.get("required_env"):
            print(f"  [OK]   All required env vars set")

        # Port conflicts
        conflicts = check_port_conflicts(demo_cfg, catalog)
        if conflicts:
            print(f"  [WARN] Ports already in use:")
            for svc, port in conflicts:
                print(f"         :{port} ({svc})")
            warns += 1
        else:
            port_count = len(demo_cfg.get("ports", {}))
            if port_count:
                print(f"  [OK]   All {port_count} ports available")

        # Compose files exist
        missing_compose = []
        for f in demo_cfg.get("compose_files", []):
            if not (ROOT / f).exists():
                missing_compose.append(f)
        if missing_compose:
            print(f"  [FAIL] Missing compose files:")
            for f in missing_compose:
                print(f"         {f}")
            ok = False
        else:
            print(f"  [OK]   All compose files present")

        # Bucket validation (buckets.yaml vs config.json)
        buckets = load_buckets(demo_name)
        if buckets:
            bucket_names = [spec.get("name", "") for spec in buckets.values() if spec.get("name")]
            print(f"  [OK]   Buckets defined: {', '.join(bucket_names)}")
            issues = validate_config_buckets(demo_name, buckets)
            for level, msg in issues:
                print(f"  [{level}] {msg}")
                if level == "FAIL":
                    ok = False
                else:
                    warns += 1
            if not issues:
                print(f"  [OK]   config.json bucket references match buckets.yaml")

    # Summary
    print()
    if ok and warns == 0:
        print("  All prerequisites met.")
    elif ok:
        print(f"  Prerequisites met with {warns} warning(s) -- see above.")
    else:
        print("  Some prerequisites FAILED -- see above.")
    return 0 if ok else 1


# ── Runner Mode Commands ──────────────────────────────────────────────


def _pycaber_version_from_checkout():
    """The ``version = "..."`` line of PyCaber's packages/pycaber/pyproject.toml, or None."""
    pyproject = PYCABER_ROOT / "packages" / "pycaber" / "pyproject.toml"
    try:
        text = pyproject.read_text()
    except OSError:
        return None, pyproject
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return (match.group(1) if match else None), pyproject


def _run_locked_ci(args):
    """``ci --locked``: resolve the defaults, then hand off to ``run_locked``."""
    from runner.core.llm_provider import PROVIDER_NAMES
    from runner.mode_ci.ci_runner import run_locked
    version = args.pycaber
    if version is None:
        version, pyproject = _pycaber_version_from_checkout()
        if version is None:
            print(f"\n  [ci] ABORT: --locked needs a pycaber version: pass --pycaber, or "
                  f"provide a 'version = \"...\"' line in {pyproject} (CABER_PYCABER_ROOT)")
            return 2
    caber_core = args.caber_core or f"ghcr.io/caber-systems/caber-core:{version}"
    if args.providers.strip().lower() == "all":
        providers = list(PROVIDER_NAMES)
    else:
        providers = [n.strip().upper() for n in args.providers.split(",") if n.strip()]
        unknown = [n for n in providers if n not in PROVIDER_NAMES]
        if unknown or not providers:
            print(f"\n  [ci] ABORT: --providers: unknown provider(s) {', '.join(unknown) or '(none)'}; "
                  f"known: {', '.join(n.lower() for n in PROVIDER_NAMES)}")
            return 2
    # Locked runs never take the NER overlay; exported so an inherited value cannot leak in.
    os.environ[WITH_NER_ENV] = "0"
    output_dir = args.output_dir or os.environ.get("CI_ARTIFACTS_DIR") or "./ci-artifacts"
    # Same 2 s policy pickup as the developer ci path (see cmd_runner_ci below).
    from runner.core.ci_fast_pickup import (
        ci_env_delta, host_snippet_dir, sweep_ci_snippet, write_ci_snippet)
    ci_snippet_dir = host_snippet_dir()
    os.environ.update(ci_env_delta())
    try:
        write_ci_snippet(ci_snippet_dir)
        return run_locked(args.demo, pycaber_version=version, caber_core_ref=caber_core,
                          providers=providers, output_dir=output_dir)
    finally:
        sweep_ci_snippet(ci_snippet_dir)


def cmd_runner_ci(args, catalog):
    """CI Test Mode: full lifecycle with PASS/FAIL and failure reporting."""
    print_preview_edition_notice()
    demos = catalog.get("demos", {})
    cfg = demos.get(args.demo, {})

    # Up-front refusals: each conflict is knowable from the arguments alone.
    explicit_dataset = getattr(args, "dataset", None)
    with_ner = getattr(args, "with_ner", False)
    # Ruling P2-14: refused before any side effect (corpus generation, the
    # NER_VERDICT_REQUIRED / CSI_NER_VERDICT_ENABLE / GLINER_ENABLE exports).
    verdict_refusal = verdict_without_ner_refused(
        args.demo, getattr(args, "verdict", False), with_ner)
    if verdict_refusal:
        print(f"  [ci] ABORT: {verdict_refusal}", file=sys.stderr)
        sys.exit(2)
    for refusal in (
            ner_overlay_refused(args.demo, with_ner),
            _generate_verdict_conflict(getattr(args, "generate", False),
                                       getattr(args, "verdict", False)),
            _sensitive_corpus_via_dataset(explicit_dataset,
                                          getattr(args, "verdict", False))):
        if refusal:
            print(f"\n  [ci] ABORT: {refusal}")
            return 1
    # build_parser refuses every developer flag combined with --locked (Ruling P2-49).
    locked = getattr(args, "locked", False)

    # Select and validate the first working LLM provider (ANTHROPIC, NVIDIA,
    # GEMINI, OPENAI, GROQ) and export OPENAI_API_KEY / OPENAI_BASE_URL /
    # USE_LLM_MODEL into os.environ before building run_env or launching.
    # A locked run selects each provider itself, one run per provider.
    if not locked:
        llm_model = (getattr(args, "llm_model", None) or "").strip()
        if llm_model:
            # Read by resolve_llm_provider, and by demo_compose_env for every
            # compose call made outside demo.py (it copies os.environ).
            os.environ[CI_LLM_MODEL_ENV] = llm_model
        ensure_llm_provider()

    # A demo shipping a frozen corpus seeds it unless this run generates its own.
    # Locked runs (--locked) force the committed snapshot so developer state
    # never substitutes a different corpus.
    if not _use_frozen_corpus(
            args.demo, explicit_dataset=explicit_dataset,
            generate=getattr(args, "generate", False),
            verdict=getattr(args, "verdict", False),
            force=locked):
        return 1

    # Load bucket env and build run_env baseline
    bucket_env = prepare_bucket_env_for_demo(args.demo)
    run_env = dict(os.environ, **bucket_env) if bucket_env else None

    # Resolve dataset: CLI --dataset flag > catalog default_dataset
    dataset_env_vars = None

    dataset_name = explicit_dataset or cfg.get("default_dataset")
    if dataset_name:
        dataset_result = _dataset_env(args.demo, dataset_name)
        if dataset_result:
            dataset_env_vars, _data_mount = dataset_result  # mount path; compose file wires the actual bind mount
            run_env = dict(run_env or os.environ, **dataset_env_vars)
            print(f"\n  Dataset: {dataset_name} -> env vars injected (mount wired in compose file)")
            log_corpus_choice(
                dataset_env_vars["DEMO_DATA_DIR_HOST"],
                f"--dataset {dataset_name!r}" if explicit_dataset
                else f"catalog default_dataset {dataset_name!r} for "
                     f"{args.demo!r} (no --dataset given)")

    # Propagate dataset-specific env vars into os.environ so run_ci picks them up.
    # Only inject the small dataset delta -- not the full run_env (which is a copy of os.environ + overrides).
    if dataset_env_vars:
        os.environ.update(dataset_env_vars)

    # When a frozen or generated corpus is active for this demo, point
    # DEMO_DATA_DIR_HOST at its resolved directory so both ci_runner (coverage /
    # warm-up) and the one-shot seed container mount the active corpus rather
    # than the raw default_dataset root.
    if os.environ.get("DEMO_CORPUS_PINNED") != "1":
        from runner.core.generated_data import resolve_generated_data
        gen_overrides = resolve_generated_data(args.demo)
        if gen_overrides and gen_overrides.get("_generated_data_dir"):
            gen_host = str(Path(gen_overrides["_generated_data_dir"]).resolve())
            displaced = os.environ.get("DEMO_DATA_DIR_HOST")
            os.environ["DEMO_DATA_DIR_HOST"] = gen_host
            log_corpus_choice(
                gen_host,
                f"active corpus for {args.demo!r}",
                displaced=displaced if displaced != gen_host else None,
                displaced_reason=f"catalog default_dataset {dataset_name!r}",
            )

    if locked:
        return _run_locked_ci(args)

    # Factory mode: generate ground truth so the stack ingests it (via resolve_generated_data)
    # and arm the shape gate. Without --generate the gate skips (base-data run).
    if getattr(args, "generate", False):
        from pathlib import Path as _Path
        from runner.core.generated_data import run_generate
        print("\n  [ci] Factory mode: generating ground-truth dataset before launch ...")
        rc_gen = run_generate(
            args.demo, domain="contracts", topology="mixed",
            target_docs=5, chains_per_user=2, base_seed=42,
            docs_per_user=10,
        )
        if rc_gen != 0:
            print("  [ci] Factory generation FAILED -- aborting CI")
            return rc_gen
        if not any((_Path(__file__).resolve().parent / "data" / "agentgov-docs" / "generated").rglob("ground_truth.json")):
            print("  [ci] Generation produced no ground_truth.json -- aborting CI")
            return 1
        os.environ["IDF_SHAPE_REQUIRED"] = "1"

    # Verdict mode: regenerate the sensitive_documents corpus, SEED it, enable
    # the deployed NER verdict pass (GLiNER ingest enrichment + opt-in
    # ner_analysis), and arm the verdict gate. Without --verdict the gate skips
    # (it is fail-closed only when armed). The runtime enable flags are exported
    # into os.environ so docker compose interpolates them into the csiMVP
    # services at container-create time; CSI_NER_FINGERPRINT_SECRET /
    # OPENAI_API_KEY must already be present (loaded from .env by ci_runner --
    # never read or written here).
    #
    # This is a DATASET choice, not a policy mode. Policy activation is the same
    # on every run: ci_runner activates the demo's Governance_Policies exactly
    # as authored. What --verdict changes is which corpus is seeded, so every
    # consumer that reads the seeded corpus (the verdict gate, the warm-up
    # traffic) sees the sensitive one.
    if getattr(args, "verdict", False):
        from pathlib import Path as _Path
        from runner.core.generated_data import run_generate_sensitive, sensitive_corpus_dir

        # Refused BEFORE generating anything: the conflict is knowable from the
        # arguments alone.
        conflict = _sensitive_corpus_dataset_conflict(explicit_dataset)
        if conflict:
            print(f"\n  [ci] ABORT: {conflict}")
            return 1

        print("\n  [ci] Verdict mode: generating sensitive_documents corpus before launch ...")
        # require_questions=True: this run seeds the corpus, and ci_runner drives
        # the seeded corpus's own per-region questions as warm-up traffic (the
        # only object access before activation). A corpus without them cannot
        # be exercised, and the failure is knowable here -- before the stack is
        # launched and half an hour of indexing has been spent.
        rc_gen = run_generate_sensitive(require_questions=True)
        if rc_gen != 0:
            print("  [ci] Sensitive-corpus generation FAILED -- aborting CI")
            return rc_gen
        if not (sensitive_corpus_dir() / "manifest.json").exists():
            print("  [ci] Generation produced no manifest.json -- aborting CI")
            return 1

        # Seed the corpus just generated. Without this the run falls back to
        # seed.py's _DEFAULT_DATA_DIR (data/agentgov-docs, the amy/bob corpus):
        # the graph is built over documents that carry no SSN and the verdict
        # gate scores a manifest nothing was ingested for.
        corpus_env, corpus_err = _sensitive_corpus_env(sensitive_corpus_dir())
        if corpus_err:
            print(f"\n  [ci] ABORT: {corpus_err}")
            return 1
        displaced = os.environ.get("DEMO_DATA_DIR_HOST")
        os.environ.update(corpus_env)
        log_corpus_choice(
            corpus_env["DEMO_DATA_DIR_HOST"],
            "--verdict generated the sensitive_documents corpus; "
            "DEMO_DATA_DIR_HOST set from "
            "runner.core.generated_data.sensitive_corpus_dir()",
            displaced=(displaced
                       if displaced != corpus_env["DEMO_DATA_DIR_HOST"] else None),
            displaced_reason=f"catalog default_dataset {dataset_name!r}")
        print(f"  [corpus] DEMO_CORPUS_PINNED=1 -- the corpus activated at "
              f"data/agentgov-docs/generated will NOT be adopted for this run")

        os.environ["NER_VERDICT_REQUIRED"] = "1"
        # Runtime enable flags for the deployed pipeline (compose ${VAR} interpolation).
        # The same NER pass produces the region labels Edge attributes such as
        # edge.gov.auth.ner.labels are derived from.
        os.environ["CSI_NER_VERDICT_ENABLE"] = "1"
        os.environ["GLINER_ENABLE"] = "1"
        if not os.environ.get("CSI_NER_FINGERPRINT_SECRET"):
            print("  [ci] WARN: CSI_NER_FINGERPRINT_SECRET not set -- the deployed "
                  "verdict pass fails closed without it (set it in .env).")
        if not os.environ.get("OPENAI_API_KEY"):
            print("  [ci] WARN: OPENAI_API_KEY not set -- the deployed verdict pass "
                  "cannot call the OpenAI provider (set it in .env).")

    from runner.mode_ci.ci_runner import run_ci
    # gliner2 models-only GPU serving: start the host Apple-MPS model server and
    # export CSI_GLINER2_MODEL_URL so run_ci's compose routes gliner2 to the Mac GPU.
    # No-op unless DocParser.gliner is enabled with engine gliner2
    # (GLINER_ENABLE/GLINER_ENGINE override); aborts if that NER config is on but the
    # server is unreachable. ci is one-shot, so the server is stopped after the run
    # (run_ci does its own container teardown). agentic-tools ships without NER,
    # so it needs the server only with --with-ner.
    ner_server = host_ner_server_required(args.demo, with_ner)
    if ner_server:
        require_gliner2_gpu_server()
    # --with-ner: every compose call of this run (run_ci's launch and the
    # recreates its pytest stages make) appends the NER overlay, as `up` does.
    # Exported as "0" otherwise so an inherited value cannot leak in.
    os.environ[WITH_NER_ENV] = "1" if with_ner else "0"
    # docling-serve native GPU sidecar (Plan C section 2.5): opt-in via
    # DOCLING_GPU_ENABLE=1. No CPU fallback -- on failure CSI_DOCLING_ENDPOINT
    # is left unset and the container-side guards fail parses loudly.
    ensure_docling_gpu_server()

    # CI only: 2 s ENV_Manager tick (compose forwards it to csi-init) and a 2 s
    # SDK heartbeat snippet, so policy changes land in ~2 s instead of ~20 s.
    # The snippet is swept in the finally below, and by every non-CI `up`.
    from runner.core.ci_fast_pickup import (
        ci_env_delta, host_snippet_dir, sweep_ci_snippet, write_ci_snippet)
    ci_snippet_dir = host_snippet_dir()
    os.environ.update(ci_env_delta())
    try:
        write_ci_snippet(ci_snippet_dir)
        rc = run_ci(
            args.demo,
            output_dir=args.output_dir,
            headless=not getattr(args, "visible", False),
            slow_mo=getattr(args, "slow_mo", 0),
            no_teardown=getattr(args, "keep_open", False),
            dashboard=getattr(args, "dashboard", False),
        )
        # Close the other end of the bring-up gate. run_ci can only report on what
        # it observed; a model server that died partway through leaves it green
        # with zero entities indexed from that point on. Probed BEFORE the teardown
        # below, which would otherwise stop the server and destroy the evidence.
        if ner_server and not gliner2_gpu_server_survived():
            rc = 1
        return rc
    finally:
        # Always sweep, even on failure or interrupt: the snippet is CI-only.
        sweep_ci_snippet(ci_snippet_dir)
        # Respect --keep-open: leave the servers up when containers are left running.
        if not getattr(args, "keep_open", False):
            stop_gliner2_gpu_server()
            stop_docling_gpu_server()


def cmd_runner_video(args, catalog):
    """Production Demo Video: 4K narrated video."""
    ensure_llm_provider()
    from runner.mode_video.video_runner import produce_video
    path = produce_video(
        args.demo,
        output_path=args.output,
        tts_engine=args.tts,
        resolution=args.resolution,
        headless=not getattr(args, "visible", False),
        slow_mo=getattr(args, "slow_mo", 0),
    )
    print(f"Video saved to: {path}")
    return 0


def cmd_generate(args, catalog):
    """Generate factory seed data using CaberDataFactory."""
    # Adoption reuses documents that already exist, so it runs BEFORE the
    # CaberDataFactory import check and the OPENAI_API_KEY check: it neither
    # generates nor embeds anything, and failing it for a missing generator
    # would be a failure for the wrong reason.
    if getattr(args, "adopt", None):
        from runner.core.generated_data import adopt_corpus_for_demo
        demos = catalog.get("demos", {})
        if args.demo not in demos:
            print(f"Unknown demo: {args.demo}")
            return 1
        try:
            target = adopt_corpus_for_demo(args.adopt, args.demo)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[adopt] FAILED: {exc}")
            return 1
        print(f"[adopt] {args.demo} now SHARES {target}")
        print(f"[adopt] Config: demos/{args.demo}/config.generated.json "
              f"(derived from this demo's own overlay; the corpus keeps its own)")
        print(f"[adopt] Run 'demo.py up {args.demo}' to launch with it. The demo "
              f"that generated the corpus is unaffected and still resolves it.")
        return 0

    try:
        from caber_datafactory import DataFactory  # noqa: F401
    except ImportError:
        print("ERROR: caber_datafactory not installed.")
        print("  Install with: pip install -e ../CaberDataFactory")
        return 1

    # The sensitive_documents corpus is fully deterministic (no LLM/network), so
    # it needs no OPENAI_API_KEY and uses its own generator path + output dir.
    if getattr(args, "sensitive", False):
        from runner.core.generated_data import run_generate_sensitive
        return run_generate_sensitive(
            num_documents=getattr(args, "num_documents", 4),
            sections_per_doc=getattr(args, "sections_per_doc", 4),
            seed=args.seed if args.seed is not None else 42,
        )

    ensure_llm_provider()

    demos = catalog.get("demos", {})
    cfg = demos.get(args.demo)
    if not cfg:
        print(f"Unknown demo: {args.demo}")
        return 1

    from runner.core.generated_data import run_generate
    return run_generate(
        demo_name=args.demo,
        domain=args.domain,
        topology=args.topology,
        target_docs=args.target_docs,
        chains_per_user=args.chains_per_user,
        base_seed=args.seed,
        docs_per_user=args.docs_per_user,
        frontmatter_base_date=getattr(args, "frontmatter_base_date", None),
    )


# ── Main ──────────────────────────────────────────────────────────────

# `ci` flags a locked run refuses (Ruling P2-49): the published preview runs one
# shape only. (flag, namespace attribute, the default that means "not given")
_LOCKED_REFUSED_CI_FLAGS = (
    ("--with-ner", "with_ner", False),
    ("--generate", "generate", False),
    ("--verdict", "verdict", False),
    ("--dataset", "dataset", None),
    ("--keep-open", "keep_open", False),   # run_ci's no_teardown
    ("--dashboard", "dashboard", False),
    ("--visible", "visible", False),
    ("--slow-mo", "slow_mo", 0),
    ("--llm-model", "llm_model", None),
)


class _DemoArgumentParser(argparse.ArgumentParser):
    """Refuses the locked-only ``ci`` flags without ``--locked``, and the
    developer ``ci`` flags with it."""

    def parse_known_args(self, args=None, namespace=None):
        ns, extras = super().parse_known_args(args, namespace)
        # Subparsers share this class; only the top-level namespace carries `command`.
        if getattr(ns, "command", None) != "ci":
            return ns, extras
        if getattr(ns, "locked", False):
            refused = [flag for flag, attr, default in _LOCKED_REFUSED_CI_FLAGS
                       if getattr(ns, attr, default) != default]
            if refused:
                self.error(f"--locked cannot be combined with {', '.join(refused)}")
        else:
            given = [flag for flag, set_ in (
                ("--pycaber", getattr(ns, "pycaber", None) is not None),
                ("--caber-core", getattr(ns, "caber_core", None) is not None),
                ("--providers", getattr(ns, "providers", "all") != "all")) if set_]
            if given:
                self.error(f"{', '.join(given)} require --locked")
        return ns, extras


def build_parser() -> argparse.ArgumentParser:
    parser = _DemoArgumentParser(
        description="Caber Demo Catalog -- launch any demo from one place",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # -- Shared browser flags (for ci, video) -----------------------------
    browser_flags = argparse.ArgumentParser(add_help=False)
    browser_flags.add_argument("--visible", action="store_true",
                               help="Show browser window on screen")
    browser_flags.add_argument("--slow-mo", type=int, default=0,
                               help="Slow down browser actions by N ms")
    browser_flags.add_argument("--keep-open", action="store_true",
                               help="Keep browser open after completion")

    # -- Utilities --------------------------------------------------------
    sub.add_parser("list", help="List available demos")

    p_check = sub.add_parser("check", help="Verify prerequisites")
    p_check.add_argument("demo", nargs="?", default=None,
                         help="Optional demo name for demo-specific checks")

    p_info = sub.add_parser("info", help="Show demo details")
    p_info.add_argument("demo", help="Demo name (from catalog.yaml)")

    # -- Human workflow (visible, interactive by default) ------------------
    p_up = sub.add_parser("up", help="Launch + seed (visible, interactive)")
    p_up.add_argument("demo", help="Demo name (from catalog.yaml)")
    p_up.add_argument("--no-visible", action="store_true",
                      help="Run headless (default: visible)")
    p_up.add_argument("--slow-mo", type=int, default=300,
                      help="Slow down browser actions by N ms (default: 300)")
    p_up.add_argument("--no-keep-open", action="store_true",
                      help="Close browser after seed (default: keep open)")
    p_up.add_argument("--no-seed", action="store_true",
                      help="Skip data seeding (default: seed)")
    p_up.add_argument("--dashboard", action="store_true",
                      help="Launch CaberUserConsole dashboard alongside the demo")
    p_up.add_argument("--dataset", default=None,
                      help="Named dataset from data/ to mount at /demo-data (overrides catalog default_dataset)")
    p_up.add_argument("--with-ner", action="store_true",
                      help="agentic-tools only: add the NER overlay "
                           "(demos/agentic-tools/docker-compose.ner.yaml) and start the host "
                           "GLiNER2 model server. The base stack ships without NER.")

    p_down = sub.add_parser("down", help="Tear down a demo")
    p_down.add_argument("demo", help="Demo name (from catalog.yaml)")

    # -- AI / automation workflow (headless, fast by default) --------------
    p_ci = sub.add_parser("ci", parents=[browser_flags],
                          help="Full lifecycle: launch -> seed -> test -> teardown")
    p_ci.add_argument("demo", help="Demo name")
    p_ci.add_argument("--output-dir", default=None,
                      help="Override CI_ARTIFACTS_DIR for test artifacts")
    p_ci.add_argument("--dashboard", action="store_true",
                      help="Launch CaberUserConsole dashboard for lean audit E2E tests")
    p_ci.add_argument("--dataset", default=None,
                      help="Named dataset from data/ to mount at /demo-data (overrides catalog default_dataset)")
    p_ci.add_argument("--generate", action="store_true",
                      help="Generate factory ground-truth data before launch and enforce the "
                           "IDF version-DAG shape gate (exports IDF_SHAPE_REQUIRED=1)")
    p_ci.add_argument("--verdict", action="store_true",
                      help="Generate the sensitive_documents corpus and seed it (instead of "
                           "the catalog default dataset), enable the deployed NER "
                           "verdict pass, and enforce the verdict gate (exports "
                           "NER_VERDICT_REQUIRED=1 + CSI_NER_VERDICT_ENABLE=1, GLINER_ENABLE=1). "
                           "Cannot be combined with --dataset. On agentic-tools, --verdict "
                           "requires --with-ner.")
    p_ci.add_argument("--with-ner", action="store_true",
                      help="agentic-tools only: compose with the NER overlay "
                           "(demos/agentic-tools/docker-compose.ner.yaml) and require the host "
                           "GLiNER2 model server for the run. The base stack ships without NER.")
    p_ci.add_argument("--llm-model", metavar="MODEL_ID", default=None,
                      help="Chat model for the chat app and agent, on the provider the run "
                           "would select anyway (the provider never changes; a provider that "
                           "rejects the id fails the run). Default: the provider's configured "
                           "model (e.g. ANTHROPIC_MODEL). Also read from CABER_CI_LLM_MODEL. "
                           "Recorded under \"llm\" in run_summary.json. Not combinable with "
                           "--locked.")
    p_ci.add_argument("--locked", action="store_true",
                      help="Locked preview run: pycaber from PyPI, every provider, writes preview.lock.json")
    p_ci.add_argument("--pycaber", metavar="VERSION", default=None,
                      help="--locked only: pycaber version to install from PyPI (default: the "
                           "version in PyCaber packages/pycaber/pyproject.toml)")
    p_ci.add_argument("--caber-core", metavar="REF", default=None,
                      help="--locked only: caber-core image (default: "
                           "ghcr.io/caber-systems/caber-core:<pycaber version>)")
    p_ci.add_argument("--providers", default="all",
                      help="--locked only: 'all' (default, registry order) or a comma list of "
                           "provider names; one full ci run per provider")

    p_test = sub.add_parser("test", help="Run smoke tests on a running demo")
    p_test.add_argument("demo", help="Demo name")

    p_video = sub.add_parser("video", parents=[browser_flags],
                             help="Produce narrated demo video")
    p_video.add_argument("demo", help="Demo name")
    p_video.add_argument("--output", default=None,
                         help="Output file path for the video (default: auto-generated)")
    p_video.add_argument("--tts", choices=["openai", "say"], default="openai",
                         help="Text-to-speech engine: 'openai' (tts-1-hd) or 'say' (macOS) (default: openai)")
    p_video.add_argument("--resolution", choices=["4k", "1080p"], default="4k",
                         help="Video resolution (default: 4k)")

    # -- Build ------------------------------------------------------------
    p_build = sub.add_parser("build", help="Build demo and/or platform containers")
    p_build.add_argument("demo", nargs="?", default=None,
                         help="Demo name (from catalog.yaml) to build app containers")
    p_build.add_argument("--platform", action="store_true",
                         help="Build csi-* platform containers from mvp source")
    p_build.add_argument("--service", default=None,
                         help="Comma-separated platform services to build (default: all)")

    # -- Cloud ------------------------------------------------------------
    p_deploy = sub.add_parser("deploy", help="Deploy to cloud (or --destroy)")
    p_deploy.add_argument("demo", help="Demo name (from catalog.yaml)")
    p_deploy.add_argument("--target", default="aws",
                          help="Deployment target (default: aws)")
    p_deploy.add_argument("--ttl", default=None,
                          help="Auto-teardown after duration (e.g. 4h, 30m, 1d)")
    p_deploy.add_argument("--destroy", action="store_true",
                          help="Destroy deployment instead of creating it")

    p_undeploy = sub.add_parser("undeploy", help="Tear down cloud deployment")
    p_undeploy.add_argument("demo", help="Demo name (from catalog.yaml)")
    p_undeploy.add_argument("--target", default="aws",
                            help="Deployment target (default: aws)")

    p_gen = sub.add_parser("generate", help="Generate factory seed data for a demo")
    p_gen.add_argument("demo", help="Demo name (e.g., agentgov-mimic)")
    p_gen.add_argument("--domain", default="contracts",
                       choices=["financial", "contracts", "clinical"],
                       help="Corpus domain (default: contracts)")
    p_gen.add_argument("--topology", default="linear",
                       choices=["linear", "diamond", "pruned_tree", "mixed"],
                       help="Version chain topology (default: linear)")
    p_gen.add_argument("--target-docs", type=int, default=5,
                       help="Documents per chain (default: 5)")
    p_gen.add_argument("--chains-per-user", type=int, default=2,
                       help="Number of chains per user (default: 2)")
    p_gen.add_argument("--docs-per-user", type=int, default=None,
                       help="Exact generated document budget per user; distributes across chains")
    p_gen.add_argument("--seed", type=int, default=None,
                       help="Base seed for reproducibility")
    p_gen.add_argument("--frontmatter-base-date", metavar="YYYY-MM-DD", default=None,
                       help="Prepend title/version/last_revised frontmatter to every "
                            "document, last_revised = this date + authoring depth "
                            "days (how demos/agentic-tools/corpus was generated)")
    p_gen.add_argument("--adopt", metavar="CORPUS_DIR", default=None,
                       help="SHARE an EXISTING factory corpus with this demo instead of "
                            "generating one. Nothing is copied: this demo gets its own "
                            "activation link to the same directory plus a "
                            "config.generated.json derived from its own overlay, so both "
                            "demos can run the same documents at once. Needs no "
                            "OPENAI_API_KEY and no CaberDataFactory. Use when a corpus "
                            "generated for another demo already holds the documents you "
                            "want.")
    p_gen.add_argument("--sensitive", action="store_true",
                       help="Generate the sensitive_documents verdict-gate corpus "
                            "(+ per-region expected-verdict manifest) instead of version chains")
    p_gen.add_argument("--num-documents", type=int, default=4,
                       help="Sensitive-corpus only: number of documents (default: 4)")
    p_gen.add_argument("--sections-per-doc", type=int, default=4,
                       help="Sensitive-corpus only: sections (regions) per doc (default: 4)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    catalog = load_catalog()

    cmd_map = {
        "list": cmd_list,
        "info": cmd_info,
        "up": cmd_up,
        "down": cmd_down,
        "build": cmd_build,
        "test": cmd_test,
        "check": cmd_check,
        "deploy": cmd_deploy,
        "undeploy": cmd_undeploy,
        "ci": cmd_runner_ci,
        "video": cmd_runner_video,
        "generate": cmd_generate,
    }

    handler = cmd_map.get(args.command)
    if handler:
        return handler(args, catalog)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
