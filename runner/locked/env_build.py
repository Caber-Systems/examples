# stdlib only; tested by tests/unit/test_locked_env_build.py
"""The one environment every locked compose call gets (P2-142..P2-144).

``merge_env_files`` layers, lowest first, ``<root>/.env`` (read only when it
exists; ``read_env`` returns nothing for a missing file), ``<root>/.env.local``
and the process environment: the shell wins over ``.env.local``, which wins over
``.env`` (P2-54). ``build_env`` adds the provider variables
``providers.resolve_providers`` sets, the stdlib mirror of the CI resolver, so a
stranger's ``run``, ``enforce`` and ``observe`` hand compose the same
``OPENAI_*``, ``SELECTED_LLM_*`` and ``SELECTED_EMBEDDING_*`` values a CI run
does. ``down`` needs no provider and takes the merge half alone. Pure: no
network, no writes.
"""

import os
from pathlib import Path
from typing import Mapping, Optional

from runner.locked.env_file import read_env
from runner.locked.providers import resolve_providers


def merge_env_files(root: Path, *, environ: Mapping[str, str] = os.environ, read=read_env) -> dict[str, str]:
    """``.env`` under ``.env.local`` under ``environ``; missing files contribute nothing."""
    root = Path(root)
    return {**read(root / ".env"), **read(root / ".env.local"), **environ}


def build_env(root: Path, *, environ: Mapping[str, str] = os.environ, read=read_env,
              embedding_dim: Optional[int] = None) -> dict[str, str]:
    """The merged env with the resolved provider variables; raises ``providers.ProviderError``."""
    return resolve_providers(merge_env_files(root, environ=environ, read=read), embedding_dim=embedding_dim)
