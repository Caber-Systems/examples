---
title: Agentic Tools Seed Service
version: 1.0.0
last_updated: 2026-11-05
classification: internal
sync_status: in-sync
sync_last_verified: 2026-11-05
---

# Agentic Tools Seed Service

`seed.py` seeds the agentic-tools demo: it uploads the corpus to the object store on the timestamp schedule, then embeds the same corpus into the Milvus Lite collection that `mcp-server-back-door` hosts. Both stores must hold the same documents, so it is one step.

## Where it runs

| Route | How | Notes |
|---|---|---|
| Compose service `seed` | `docker compose --profile seed ... up seed` | One-shot (`restart: "no"`) on the `agentgov/mcp-server` image; this directory mounted read-only at `/seed`, the corpus at `/corpus` |
| Host | `python demos/agentic-tools/seed.py` (what `demo.py` runs) | The root `seed.py` is a shim; `runner/core/docker_lifecycle.py seed_demo` prefers `seed/seed.py` when present |

The `seed` service waits for `csi-dep-objstor`, `mock-idp` and `mcp-server-back-door` to be healthy. It is behind the `seed` profile, so a plain `up` never starts it.

## Files

| File | What it is |
|---|---|
| `seed.py` | Entry point: object store, then Milvus |
| `_agentgov_seed.py` | Copy of `demos/agentgov-mimic/seed.py` with three edits: no `__main__` block, `from timestamp_realism import`, and a repo root that cannot raise inside the container |
| `timestamp_realism.py` | Copy of `runner/core/timestamp_realism.py` |
| `ingest_milvus.py` | The Milvus ingest; `demos/agentic-tools/ingest_milvus.py` is its host CLI |
| `llm_usage.py` | Byte copy of `shared/llm_usage.py` |

These files import only the stdlib, `boto3`, `pymilvus`, `shared.milvus_query` and each other (`tests/unit/test_seed_service.py`). agentgov-mimic stays frozen: when its `seed.py` or `runner/core/timestamp_realism.py` changes, copy it again and reapply the edits.

## Environment

The compose service sets the object store (`MINIO_URL`, `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`, `OBJECT_STORE_REGION`, `DEMO_DEFAULT_BUCKET`, `DEMO_CORPUS_GROUP`), the corpus (`DEMO_DATA_DIR_HOST=/corpus`), Milvus (`MILVUS_HOST=milvusdb`, `MILVUS_PORT`, `MILVUS_COLLECTION`), usage recording (`CABER_DEMO_USAGE_DIR`, app name `seed`) and the same provider variables as `mcp-server-back-door`. The `DEMO_TS_*` timestamp settings have defaults. On the host, `MILVUS_HOST` defaults to `localhost`.
