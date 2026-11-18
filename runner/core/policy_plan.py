# stdlib only; tested by tests/unit/test_policy_plan_and_mode.py
"""Which policy ids a demo flips before activation, shared by the CI runner and locked ``run``.

The source is ``demos/<demo>/demo.json`` ``policies``: an optional object with
optional ``enable`` and ``disable`` lists of policy ids. A missing key is an
empty list.

Developer fallback: a demo with no ``demo.json`` flips nothing. Its
``Governance_Policies`` are activated exactly as authored, which is what the CI
runner's ``_activate_demo_policies`` did before this module existed.

This module ships in the exported public-preview tree, so it imports nothing
outside the standard library and nothing from ``runner.locked``.
"""

import json
from dataclasses import dataclass
from pathlib import Path


class PolicyPlanError(Exception):
    """``demo.json`` exists but its ``policies`` cannot be read as two lists of ids."""


@dataclass(frozen=True)
class PolicyPlan:
    enable: tuple[str, ...]
    disable: tuple[str, ...]
    manifest_path: Path


def _ids(manifest_path: Path, policies: dict, key: str) -> tuple[str, ...]:
    value = policies.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise PolicyPlanError(f"{manifest_path}: policies.{key} must be a list of strings")
    return tuple(value)


def policy_plan(demo_name: str, demos_dir: Path) -> PolicyPlan:
    """The flips ``demos_dir/<demo_name>/demo.json`` asks for; none when that file is absent."""
    manifest_path = Path(demos_dir) / demo_name / "demo.json"
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return PolicyPlan((), (), manifest_path)
    except OSError as exc:
        raise PolicyPlanError(f"{manifest_path}: cannot be read: {exc}") from exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise PolicyPlanError(f"{manifest_path}: is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PolicyPlanError(f"{manifest_path}: must hold a JSON object")
    policies = data.get("policies", {})
    if not isinstance(policies, dict):
        raise PolicyPlanError(f"{manifest_path}: policies must be an object")
    return PolicyPlan(_ids(manifest_path, policies, "enable"),
                      _ids(manifest_path, policies, "disable"), manifest_path)
