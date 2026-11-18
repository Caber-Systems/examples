# stdlib only; tested by test_locked_package.py
"""``demo.py`` in an exported public-preview tree (locked mode).

``demo.py`` hands over here, before importing anything outside the standard
library, when ``preview.lock.json`` is present and no sibling ``PyCaber``
checkout exists. Each handler imports its subcommand module lazily, so a module
a later task adds can use what it needs without this file importing it
(``llm_probe`` in particular is never imported here).
"""

import argparse
import sys
from pathlib import Path

# The directory holding demo.py: runner/locked/cli.py -> parents[2].
ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent


def _not_implemented(name: str, exc: ImportError, module: "str | None" = None) -> int:
    """Report subcommand ``name`` whose module (``module``, default ``name``) does not exist yet.

    A real import failure (the module file is there) is re-raised.
    """
    if (_HERE / f"{module or name}.py").exists():
        raise exc
    print(f"{name}: not implemented", file=sys.stderr)
    return 2


def _check(root: Path, args: argparse.Namespace) -> int:
    try:
        from runner.locked import check as _m
    except ImportError as exc:
        return _not_implemented("check", exc)
    return _m.run_check_cli(root, args)


def _build(root: Path, args: argparse.Namespace) -> int:
    try:
        from runner.locked import build as _m
    except ImportError as exc:
        return _not_implemented("build", exc)
    return _m.run_build_cli(root, args)


def _run(root: Path, args: argparse.Namespace) -> int:
    try:
        from runner.locked import run as _m
    except ImportError as exc:
        return _not_implemented("run", exc)
    return _m.run_run_cli(root, args)


def _enforce(root: Path, args: argparse.Namespace) -> int:
    try:
        from runner.locked import mode as _m
    except ImportError as exc:
        return _not_implemented("enforce", exc, "mode")
    return _m.run_enforce_cli(root, args)


def _observe(root: Path, args: argparse.Namespace) -> int:
    try:
        from runner.locked import mode as _m
    except ImportError as exc:
        return _not_implemented("observe", exc, "mode")
    return _m.run_observe_cli(root, args)


def _down(root: Path, args: argparse.Namespace) -> int:
    try:
        from runner.locked import down as _m
    except ImportError as exc:
        return _not_implemented("down", exc)
    return _m.run_down_cli(root, args)


# (subcommand, one-line help, handler)
SUBCOMMANDS = (
    ("check", "Verify this machine can run the preview (docker, memory, ports, LLM key)", _check),
    ("build", "Pull the pinned images named in preview.lock.json and build the demo app images", _build),
    ("run", "Start the demo, seed it, and open the console and chat app", _run),
    ("enforce", "Switch the running demo to enforce mode (Caber redacts)", _enforce),
    ("observe", "Switch the running demo to observe mode (decisions recorded, nothing redacted)", _observe),
    ("down", "Stop the demo and remove its containers and volumes", _down),
)


# subcommand -> ((flag, add_argument keyword arguments), ...)
_FLAGS = {
    "build": (
        ("--pycaber", {"metavar": "SPEC", "help": "Build the app images against this PyCaber version instead of the lock's (untested)"}),
        ("--tag", {"metavar": "TAG", "help": "Pull caber-core by this tag instead of the lock digest (untested)"}),
        ("--i-know-this-is-untested", {"action": "store_true", "help": "Allow --pycaber and --tag, which the lock did not certify"}),
    ),
    "down": (
        ("--keep-images", {"action": "store_true", "help": "Keep the pulled and built images"}),
        ("--purge", {"action": "store_true", "help": "Also delete .env.local; the next run generates new secrets"}),
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="demo.py", description="Caber public preview demo")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for name, help_text, handler in SUBCOMMANDS:
        cmd = sub.add_parser(name, help=help_text, description=help_text)
        for flag, kwargs in _FLAGS.get(name, ()):
            cmd.add_argument(flag, **kwargs)
        cmd.set_defaults(handler=handler)
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return int(args.handler(ROOT, args))
