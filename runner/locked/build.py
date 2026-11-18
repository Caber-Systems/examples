# stdlib only; tested by tests/unit/test_locked_build_run_down.py
"""``./demo.py build`` in locked mode: pull every lock image by digest, build the app images.

The notice pointer is printed before the first pull, so the stranger sees
what governs the preview before anything is downloaded. Nothing is captured:
the stranger watches the pulls and the build as they happen.
"""

import subprocess
import sys
from pathlib import Path

from runner.locked.compose import compose_cmd, docker_run, load_tree, run_guarded
from runner.locked.demo_json import DemoJsonError
from runner.locked.lock import LockError

CORE_IMAGE = "caber-core"
# runner.mode_ci.locked_run.ZERO_DIGEST; restated because runner/locked is stdlib only.
ZERO_DIGEST = "sha256:" + "0" * 64


def _is_str(value) -> bool:
    return isinstance(value, str)


def _lock_problem(lock) -> "str | None":
    """What ``run_build`` needs from the lock and does not find, or None."""
    if not isinstance(lock, dict):
        return "the lock is not a JSON object"
    notice = lock.get("notice")
    if not isinstance(notice, dict) or not _is_str(notice.get("tag")) or not _is_str(notice.get("url")):
        return "notice.tag and notice.url must be strings"
    pycaber = lock.get("pycaber")
    if not isinstance(pycaber, dict) or not _is_str(pycaber.get("version")):
        return "pycaber.version must be a string"
    images = lock.get("images")
    if not isinstance(images, dict):
        return "images must be an object"
    for name, image in images.items():
        if not isinstance(image, dict) or not _is_str(image.get("ref")) or not _is_str(image.get("digest")):
            return f"images.{name} must carry string ref and digest"
    return None


def run_build(root: Path, *, lock, demo=None, run=subprocess.run, out=print,
              pycaber=None, tag=None, i_know_this_is_untested=False) -> int:
    """Pull the lock's images and build the demo's; 0 on success, 1 on a failed command, 2 refused."""
    if (pycaber or tag) and not i_know_this_is_untested:
        out("build: --pycaber and --tag change what the lock certified; "
            "add --i-know-this-is-untested to proceed.")
        return 2
    problem = _lock_problem(lock)
    if problem:
        out(f"preview.lock.json is malformed: {problem}")
        return 2
    notice = lock["notice"]
    out(f"This preview is governed by the Caber Community Preview Edition License "
        f"{notice['tag']}: {notice['url']}")
    for name, image in lock["images"].items():
        ref, digest = image["ref"], image["digest"]
        if tag and name == CORE_IMAGE:
            repo = ref.rsplit(":", 1)[0]
            target = f"{repo}:{tag}"
            out(f"untested: the compose file still pins the lock digest for caber-core; "
                f"edit its image line to {repo}:{tag} by hand.")
        elif digest == ZERO_DIGEST:
            out(f"{name}: placeholder digest in preview.lock.json; this image is not pulled")
            continue
        else:
            target = f"{ref}@{digest}"
        rc = run(["docker", "pull", target], check=False).returncode
        if rc != 0:
            out(f"docker pull {target} failed (rc={rc}); "
                "if the registry needs a login, run docker login and try again.")
            return 1
    spec = pycaber or lock["pycaber"]["version"]
    cmd = compose_cmd(root, demo, ["build", "--build-arg", f"PYCABER_SPEC==={spec}"],
                      env_file=(Path(root) / ".env.local").exists())
    rc = run(cmd, check=False).returncode
    if rc != 0:
        out(f"docker compose build failed (rc={rc}).")
        return 1
    return 0


def run_build_cli(root: Path, args) -> int:
    """``./demo.py build [--pycaber SPEC] [--tag TAG] [--i-know-this-is-untested]``."""
    try:
        lock, demo = load_tree(root)
    except (LockError, DemoJsonError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return run_guarded(lambda: run_build(root, lock=lock, demo=demo, pycaber=args.pycaber, tag=args.tag,
                                         i_know_this_is_untested=args.i_know_this_is_untested,
                                         run=docker_run()))
