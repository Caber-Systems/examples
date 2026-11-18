---
title: Agentic Tools Identity Service
version: 1.4.0
last_updated: 2026-11-16
classification: internal
sync_status: in-sync
sync_last_verified: 2026-11-16
---

# Agentic Tools Identity Service

The `mock-idp` compose service of the agentic-tools, graphrag and hipaa-ner demos is the Caber-Systems `mock-idp` image, and all three build it from this one directory (graphrag and hipaa-ner use build context `../../demos/agentic-tools/identity`), so they share one Dockerfile pin and one seed. Its service name, container name and hostname are `mock-idp`, it listens on port 8090 and serves SCIM at `/scim/v2`. caber-core's directory sync reads it through `Identity_Sources.mock_idp_directory` in each demo's `config.json` (`baseUrl` `http://mock-idp:8090/scim/v2`, issuer `http://mock-idp:8090`, `bearerTokenEnv` `MOCK_IDP_SCIM_TOKEN`). The set of demos is `MOCK_IDP_DEMOS` in `runner/core/mock_idp_local.py`. The remaining demos (agentgov-mimic, agentgov-auth) run a different program, the `CaberApp/zt-mimic` app, and keep the `zt-mimic` names. hipaa-ner also runs a zt-mimic, but only as its governance evidence source, with no directory role and no host port.

## What the image is

`identity/Dockerfile` starts from `ghcr.io/caber-systems/mock-idp` pinned by digest and copies `directory.json` to `/seed/directory.json`. `directory.json` holds the same users and groups as the legacy `CaberApp/zt-mimic/seed/directory.json`; only its `issuer` differs (`http://mock-idp:8090`). The image seeds its users and groups into a tenant at start (idempotent) and serves that tenant at the bare alias `/scim/v2/...` as well as `/t/<tenant>/scim/v2/...`.

## Environment

| Variable | Value | Purpose |
|---|---|---|
| `MOCKIDP_ADDR` | `:8090` | Listen address |
| `MOCKIDP_BASE_URL` | `http://mock-idp:8090` | Base URL in SCIM `meta.location` |
| `MOCKIDP_API_TOKEN` | `${MOCK_IDP_SCIM_TOKEN}` | Bearer token; when set, SCIM requests without it get 401 |
| `MOCKIDP_SEED_FILE` | `/seed/directory.json` | Directory seed |
| `MOCKIDP_SEED_TENANT` | `demo` | Tenant the seed is imported into |

`MOCKIDP_API_TOKEN` has no default. The runner's `.env` (or the locked runner) should supply `MOCK_IDP_SCIM_TOKEN`. With it unset, `docker compose config` prints a warning about the unset variable; that is expected. mock-idp then starts with no API token and accepts any non-empty bearer token, so caber-core's default still authenticates; it is not a secret boundary in that mode. The caber-core side keeps its own `:-` default form and is untouched.

## Digest placeholder

The digest in `identity/Dockerfile` is the all-zero placeholder `sha256:0000...0000`, because the image has not been published yet. Plan 2 Task 17 replaces it with the digest of the published `ghcr.io/caber-systems/mock-idp` image. Until then no registry can serve the pin.

### Local fallback while the pin is a placeholder

`demo.py build`, `demo.py up` and non-locked `demo.py ci` for any of the mock-idp demos detect the placeholder and build mock-idp from a local checkout instead. The helpers live in `runner/core/mock_idp_local.py`:

1. The checkout is `MOCK_IDP_SOURCE` if set, else `../mock-idp` next to this repository. A missing checkout is an error; there is no other fallback.
2. The image is tagged `caber-local/mock-idp:<12-char commit>`, with `-dirty` appended when the checkout has uncommitted changes. The build runs with `DOCKER_BUILDKIT=1`, because the mock-idp Dockerfile uses `$BUILDPLATFORM`.
3. `demo-output/mock-idp-local/context/` receives a copy of this directory whose Dockerfile replaces only the first `FROM` line with that tag, and `demo-output/mock-idp-local/docker-compose.mock-idp-local.yaml` points the `mock-idp` build at it. Both are gitignored and regenerated on every `build`, `up` and `ci` launch. Recreating services of a running demo (`recreate_services`) reuses the overlay the launch wrote and does not rebuild mock-idp.

`identity/Dockerfile` itself is never rewritten, and locked runs keep refusing the placeholder: the locked build uses `demo_compose_files`, which never carries the overlay. Once the real digest is pinned, the fallback turns itself off.

## Healthcheck

The image is distroless (no shell, no curl) and carries its own `HEALTHCHECK` (`/mock-idp -healthcheck`, which GETs `/healthz`). The compose service therefore declares no `healthcheck:` block, and `depends_on` entries with `condition: service_healthy` keep working through the inherited one.
