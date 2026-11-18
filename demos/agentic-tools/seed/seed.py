#!/usr/bin/env python3
"""Seed data for the agentic-tools demo: the object store, then Milvus.

Runs in two places with one implementation:

  - the one-shot `seed` compose service (`profiles: [seed]`), on the mcp-server
    image with this directory mounted read-only at `/seed` and the corpus at
    `/corpus` (`DEMO_DATA_DIR_HOST`);
  - the host, through `demos/agentic-tools/seed.py`, the shim `demo.py` runs
    (`runner/core/docker_lifecycle.py seed_demo` prefers this file when present).

So it imports only the stdlib, the image's own packages (boto3, pymilvus) and its
siblings in this directory:

  `_agentgov_seed.py`     the object-store upload, a copy of
                          demos/agentgov-mimic/seed.py with three edits (no
                          `__main__` block, the sibling `timestamp_realism`, and
                          a repo root that cannot raise inside the container);
  `timestamp_realism.py`  a copy of runner/core/timestamp_realism.py (the
                          LastModified schedule);
  `ingest_milvus.py`      the Milvus ingest (`demos/agentic-tools/ingest_milvus.py`
                          is its host CLI);
  `llm_usage.py`          a byte copy of shared/llm_usage.py.

agentgov-mimic stays frozen (ticket 05): its seed.py is copied, never edited.
`tests/unit/test_seed_service.py` asserts the copy keeps the source's function set.

Usage:
    python demos/agentic-tools/seed/seed.py [--data-dir DIR] [--skip-milvus]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _setup_sys_path() -> None:
    """Make `shared.milvus_query` and the siblings importable.

    `shared/milvus_query` is at the checkout root on the host (seed -> agentic-tools
    -> demos -> root) and at `/app/shared/milvus_query` in the mcp-server image.
    This directory goes in FRONT of both: run through the host shim, sys.path[0]
    is demos/agentic-tools, which holds the host CLI also named ingest_milvus.py.
    """
    parents = Path(__file__).resolve().parents
    front = []
    if len(parents) > 3 and (parents[3] / "shared" / "milvus_query").is_dir():
        front.append(parents[3])
        # Run from a checkout means run on the HOST, where `milvusdb` (the compose
        # network alias the shared query module defaults to) does not resolve and
        # Milvus is on the published port. The host CLI ingest_milvus.py does the
        # same; the compose service sets MILVUS_HOST explicitly. setdefault, so an
        # explicit MILVUS_HOST still wins.
        os.environ.setdefault("MILVUS_HOST", "localhost")
    if Path("/app").is_dir():
        front.append(Path("/app"))
    front.append(_HERE)
    for entry in front:
        text = str(entry)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


_setup_sys_path()

logger = logging.getLogger("agentic-tools.seed")


def _log(msg: str) -> None:
    print(f"[seed] {msg}", flush=True)


def _register_embedding_usage_recorder() -> None:
    """Record the ingest's embedding usage to the demo usage JSONL (app "seed").

    Mirrors CaberApp/mcp-server/app/main.py `_register_embedding_usage_recorder`.
    Nothing is registered unless `CABER_DEMO_USAGE_DIR` is set (the compose
    service sets it; a host run normally does not). The recorder never fails the
    ingest: shared/milvus_query swallows and logs anything it raises.
    """
    import llm_usage

    if not llm_usage.recording_enabled():
        return
    try:
        from shared.milvus_query import set_usage_recorder
    except Exception as exc:  # noqa: BLE001 -- accounting never stops the seed
        logger.warning("[USAGE] embedding usage recorder not registered: %s", exc)
        return

    def _record(embed_provider: str, model: str, tokens) -> None:
        chat_provider = llm_usage.selected_chat_provider() or embed_provider.lower()
        llm_usage.record_or_warn(
            logger, "seed", chat_provider, "embedding", model, tokens or 0, 0,
            embedding_provider=llm_usage.embedding_provider_for(chat_provider),
            usage_missing=tokens is None,
        )

    set_usage_recorder(_record)
    _log(f"[USAGE] embedding usage is recorded to {llm_usage.usage_dir_from_env()}")


def main() -> None:
    """Seed the object store, then ingest the same corpus into Milvus.

    Both stores must hold the SAME documents: the agent's choice between
    `search_files_bm25` and `search_semantic` is only a choice between views of
    one document store if the contents match. Seeding one without the other would
    make tool selection look wrong when it was the data that diverged.
    """
    parser = argparse.ArgumentParser(description="Seed the agentic-tools demo")
    parser.add_argument("--visible", action="store_true",
                        help="(ignored) accepted for interface consistency")
    parser.add_argument("--slow-mo", type=int, default=0,
                        help="(ignored) accepted for interface consistency")
    parser.add_argument("--keep-open", action="store_true",
                        help="(ignored) accepted for interface consistency")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory")
    parser.add_argument("--skip-milvus", action="store_true",
                        help="Seed the object store only. search_semantic will then "
                             "report a missing collection rather than silently "
                             "finding nothing.")
    args = parser.parse_args()

    import _agentgov_seed

    rc = _agentgov_seed.seed(headless=not args.visible, slow_mo=args.slow_mo,
                             keep_open=args.keep_open, data_dir=args.data_dir)
    if rc != 0:
        raise SystemExit(rc)

    if args.skip_milvus:
        _log("--skip-milvus: leaving the Milvus collection untouched")
        raise SystemExit(0)

    from ingest_milvus import _DEFAULT_DATA_DIR, ingest

    _register_embedding_usage_recorder()
    data_dir = Path(args.data_dir or os.getenv("DEMO_DATA_DIR_HOST")
                    or _DEFAULT_DATA_DIR)
    raise SystemExit(ingest(data_dir))


if __name__ == "__main__":
    main()
