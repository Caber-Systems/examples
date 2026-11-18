---
title: Agentic Tools Frozen Corpus
version: 1.1.0
last_updated: 2026-11-05
classification: internal
sync_status: in-sync
sync_last_verified: 2026-11-05
---

# Agentic Tools Frozen Corpus

## What this is

This directory is the default document corpus for the `agentic-tools` demo. It
is a CaberDataFactory run captured once by `scripts/snapshot_corpus.py` and
committed to git, so a fresh checkout seeds the same bytes CI ran, with no LLM
key and no factory run.

Contents:

- 20 contract documents in 4 version chains, 5 documents per chain
  (`v01.md` to `v05.md`).
- Two groups, `cats` (user amy) and `dogs` (user bob), with two chains each.
  The chain topologies are linear, diamond and pruned_tree.
- Every document carries YAML frontmatter with `title` (the chain's source
  document title), `version` (the node number in the chain) and
  `last_revised` (the base date plus the document's authoring depth, in days).
  PyCaber maps `last_revised` to a property claim with date semantics
  `Revised`.
- Per chain: `ground_truth.json` (the version DAG) and `questions.json`.
- Corpus level: `buckets.yaml`, `factory_questions.json`,
  `factory_questions_eval.json` and `timestamp_plan.json`.
- `manifest.json` names the sha256 of every file above plus the generation
  parameters, including `llm_model` and `llm_temperature`. It does not list
  itself or this README.

The demo never seeds this directory directly. `runner/core/generated_data.py`
verifies it against the manifest, copies it into the gitignored
`data/agentgov-docs/generated-corpora/agentic-tools-frozen-<digest>/` and
adopts the copy, so the parser cache and the seed's rewritten
`timestamp_plan.json` never touch the committed files. Seeding sets
`DEMO_TS_COPY_RATE=0` and `DEMO_TS_SYNC_EVENTS=0` (where unset), so every
seeded storage timestamp agrees with the document's claimed `last_revised`.

## How it was generated

| Item | Value |
|------|-------|
| Generated | 2026-11-06T00:04Z (`generated_at`; a cache-only rerun of the 2026-11-05 generation) |
| Factory commit | `57ef09f75e36e0542fe6998435b2a236ab7ff28a` (CaberDataFactory) |
| LLM model | `gpt-6-luna` (`llm_model`) |
| Temperature | `default`, parameter omitted (`llm_temperature`) |
| Source demo | `agentgov-mimic` (recorded as `source_demo`) |
| Domain | `contracts` |
| Topology | `mixed` |
| `target_docs` | 5 |
| `chains_per_user` | 2 |
| `base_seed` | 42 |
| `docs_per_user` | 10 |
| `frontmatter_base_date` | 2026-01-05 |

The corpus is generated under `agentgov-mimic` because that demo carries the
user configuration the factory needs. The snapshot then records
`demo: agentic-tools` and `source_demo: agentgov-mimic`.

`gpt-6-luna` rejects any temperature other than its own default, so the run
sets `DATAFACTORY_OPENAI_TEMPERATURE=default`, which makes the factory omit the
parameter from every chat completion request.

`run_generate` records both settings in the factory run manifest and the
snapshot carries them: `llm_model` is the resolved `DATAFACTORY_OPENAI_MODEL`,
and `llm_temperature` is the resolved `DATAFACTORY_OPENAI_TEMPERATURE` setting,
which is `"site"` when unset (each factory call site keeps its own value),
`"default"`, or the float (0 to 2). The factory caches responses under
`static/domain_cache/<domain>/<model>/` with no temperature in the path or the
cached response, so a changed temperature takes effect only from a fresh
`domain_cache/<domain>/<model>` directory; otherwise the cached text, sampled
at the earlier temperature, is reused while the manifest records the new
setting.

## Refresh procedure

A refresh replaces the committed bytes, so do it deliberately and commit the
result as one change.

1. Generate a factory corpus with the same parameters. Use a CaberDataFactory
   checkout at the commit you intend to record, with an OpenAI key available
   by name (`OPENAI_API_KEY`). Make that checkout the one Python imports,
   either on the path or installed editable, because the venv otherwise
   imports whichever factory is already installed:

   ```bash
   export PYTHONPATH=<factory checkout>/src    # or: pip install -e <factory checkout>
   python -c "import caber_datafactory; print(caber_datafactory.__file__)"
   export DATAFACTORY_OPENAI_MODEL=gpt-6-luna
   export DATAFACTORY_OPENAI_TEMPERATURE=default
   ```

   The printed path must be inside the intended checkout.

   Then call the runner's generator from the CaberDemos repo root:

   ```python
   from runner.core.generated_data import run_generate
   run_generate("agentgov-mimic", domain="contracts", topology="mixed",
                target_docs=5, chains_per_user=2, base_seed=42,
                docs_per_user=10, frontmatter_base_date="2026-01-05")
   ```

   The output lands in
   `data/agentgov-docs/generated-corpora/agentgov-mimic-factory-<digest>/`.
   The run also repoints the tracked `data/agentgov-docs/generated` link and
   writes `demos/agentgov-mimic/docker-compose.generated.yaml`; restore the
   link with `git checkout -- data/agentgov-docs/generated` before
   committing.

2. Snapshot it over this directory:

   ```bash
   python scripts/snapshot_corpus.py \
     --from data/agentgov-docs/generated-corpora/agentgov-mimic-factory-<digest> \
     --to demos/agentic-tools/corpus \
     --demo agentic-tools --llm-model gpt-6-luna \
     --factory-commit <full factory sha>
   ```

   The script validates every document's frontmatter, writes a clean
   `timestamp_plan.json`, writes `manifest.json` last and keeps this README.
   `llm_model` and `llm_temperature` come from the run manifest; the
   `--llm-model` flag is checked against the recorded model and a mismatch
   stops the snapshot (for a run manifest too old to record one, the flag
   supplies it). Running it twice on the same source produces a
   byte-identical manifest.

3. Update the generation table above (factory commit, date, and any parameter
   that changed), then run `python -m pytest tests/unit/test_corpus_frontmatter.py`.

## Returning to the frozen snapshot

The snapshot is the default. These all take precedence over it for a run:
`demo.py generate agentic-tools --adopt <corpus dir>`, `up` or `ci` with
`--dataset`, `ci --verdict`, `DEMO_CORPUS_PINNED=1`, and a corpus a developer
activated earlier for the demo. Plain `demo.py generate agentic-tools` and
`ci agentic-tools --generate` are not override paths: both run the factory for
agentic-tools, which has no user configuration, so they fail. Generate under
`agentgov-mimic` and adopt the result instead.

To return to the snapshot after activating a developer corpus, remove the
activation link `data/agentgov-docs/generated-agentic-tools`. On the next
launch `choose_frozen_corpus` picks this directory again and logs
`frozen snapshot demos/agentic-tools/corpus (default for 'agentic-tools')`.
