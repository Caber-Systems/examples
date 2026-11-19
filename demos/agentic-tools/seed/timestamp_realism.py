"""Realistic storage timestamps for seeded demo corpora.

WHY THIS EXISTS
---------------
`LastModified` is assigned by MinIO at PUT time, so the ONLY faithful way to
give a demo object a wrong timestamp is to write it at the wrong moment.
Forging `last_modified_at` into the evidence bundle instead would bypass the
connector, the observation record and the governance write -- exactly the layers
a demo is supposed to exercise.

Today's seed writes every object back to back in enumeration order, so a whole
corpus lands inside one or two seconds. That is not a neutral default: it is the
degenerate case where every version of every document ties, the timestamp signal
is zero, and any ordering the resolver produces came from somewhere else. This
module replaces that burst with a schedule.

WHAT IT MODELS
--------------
Real storage timestamps do not fail per object, they fail per EVENT:

  clean       an in-place edit stamps a time genuinely after its parent's, so
              the clean fraction of a corpus is a correct linear extension of
              the provenance DAG. That is faithful, not a cheat.
  copy/sync   one bucket sync stamps a whole prefix at one instant. FORWARD
              ONLY -- a copy never predates its original -- and COHERENT: every
              object in the sync ties with every other. Coherence is what makes
              it hard, because two documents copied together are evidentially
              indistinguishable from a genuine consecutive pair.
  granularity S3 and CloudTrail both report to one second, so anything written
              inside a tick ties.

WHAT IT DELIBERATELY DOES NOT MODEL
-----------------------------------
Clock skew between environments. Skew is a constant offset on a DIFFERENT
clock, and every object here is stamped by one MinIO instance from one clock.
Producing it would require either a second MinIO with a shifted clock or
injecting metadata, and injecting metadata is the thing this module exists to
avoid. A knob that claimed to simulate skew by write order would be lying, so
there is no such knob. See `SKEW_NOT_MODELLED`.

THE SCHEDULE
------------
Objects are written in ROUNDS, one tick apart. Round r holds every object at
authoring depth r across every chain and user, so total wall clock is
`max_chain_depth * tick` rather than `n_objects * tick` -- fifteen seconds for a
corpus of hundreds, not minutes. Within a chain the versions are then strictly
ordered, which is the only ordering the resolver reads, because resolution runs
per cluster.

Objects assigned to a sync event are held out of their authoring round and
written at the end, grouped by event. Their timestamps therefore land after
every clean object and tie with their event peers, which is precisely the copy
signature.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

SKEW_NOT_MODELLED = (
    "cross-environment clock skew is not simulated: every object is stamped by "
    "one MinIO instance from one clock, so write order cannot produce a "
    "per-domain offset. Model it with a second endpoint or leave it out."
)

# MinIO reports LastModified to one second. A tick below this collapses rounds
# into ties, which is a legitimate thing to ask for (it simulates a batch
# writer) but must be deliberate rather than accidental.
MINIO_TIMESTAMP_RESOLUTION_SECS = 1.0

# THE CENTRAL DEFECT this module exists to surface, not just MinIO's clock.
#
# PyCaber's version resolver derives ordering constraints from storage
# timestamps and ABSTAINS on any pair whose timestamps differ by <= this many
# seconds -- crates/data_indexer/src/roaring_index/idf/metadata_penalty.rs,
# `storage_timestamp_constraint`, via the sweep param `CSI_IDF_META_MARGIN_SECS`
# (default 60, see `metadata_margin_secs()` in the same crate). The seeder's
# default tick is 1.0s, which does not clear a 60s margin: every version of
# every document lands within a handful of seconds, every pair falls inside
# the margin, and the resolver derives ZERO constraints from a schedule this
# module worked to make faithful. Measured empirically: a 1s-spaced schedule
# scores byte-identical whether CSI_IDF_META_BETA is 0 or 20, because the
# margin discards the evidence before beta ever sees it.
#
# This is a property of the RESOLVER, reproduced here as a literal so
# `build_schedule` can warn about it -- it is not configuration this module
# owns or can change by itself. See `build_schedule`'s note and
# demo_runner_ARCHITECTURE.md's "Seed-time timestamp realism" section for the
# margin the demo stack actually has to set.
RESOLVER_ABSTENTION_MARGIN_SECS = 60

# Default base seed, matching CaberDataFactory's own default so a demo run and
# the corpus it seeds can share one number.
DEFAULT_BASE_SEED = 42

# Band the derived copy rate is drawn from. Bounded above by the field estimate
# that a real deployment carries "20% or less" unreliable timestamps, and
# floored above zero so no run silently becomes the uncorrupted case -- which is
# the one cell whose result is the answer key rather than a measurement.
COPY_RATE_BAND = (0.05, 0.20)

# Sub-seed offsets, following CaberDataFactory's arithmetic derivation idiom
# (`base_seed * 100 + ...` in runner/core/generated_data.py). Separate streams so
# the rate and the victim choice are independently derived from one base seed:
# a caller who pins DEMO_TS_COPY_RATE still gets the same victims for a seed,
# and the rate draw cannot be perturbed by an unrelated change to sampling.
_SEED_OFFSET_RATE = 1
_SEED_OFFSET_VICTIMS = 2

# Env var names for `realism_enabled` / `realism_settings` /
# `seed_with_realistic_timestamps`, named once here so every seed.py that
# drives a schedule reads the SAME control surface rather than each caller
# re-typing (and risking a typo in) its own copy of the strings.
ENV_TIMESTAMP_REALISM = "DEMO_TIMESTAMP_REALISM"
ENV_TS_SEED = "DEMO_TS_SEED"
ENV_TS_COPY_RATE = "DEMO_TS_COPY_RATE"
ENV_TS_COPY_RATE_BAND = "DEMO_TS_COPY_RATE_BAND"
ENV_TS_SYNC_EVENTS = "DEMO_TS_SYNC_EVENTS"
ENV_TS_TICK_SECS = "DEMO_TS_TICK_SECS"
ENV_TS_PLAN = "DEMO_TS_PLAN"


def resolve_copy_rate(base_seed: int, explicit: float | None = None,
                      band: tuple[float, float] = COPY_RATE_BAND) -> tuple[float, str]:
    """The copy rate for a run, and where it came from.

    An explicitly configured rate is honoured exactly -- pinning it is how a
    sweep holds the rate fixed while varying something else. Otherwise the rate
    is DERIVED from the base seed, so that changing the seed changes both how
    much of the corpus is unreliable and which parts of it are, rather than only
    the latter. Two runs with one seed are identical; two runs with different
    seeds differ in both dimensions.

    Returns `(rate, source)` where source is "explicit" or "derived", because
    the plan artifact has to be able to say which, and a derived rate that looks
    surprising is otherwise indistinguishable from a misconfigured one.
    """
    if explicit is not None:
        if not 0.0 <= explicit <= 1.0:
            raise ValueError(f"copy_rate must be in [0, 1], got {explicit}")
        return explicit, "explicit"
    lo, hi = band
    if not 0.0 <= lo <= hi <= 1.0:
        raise ValueError(f"copy rate band must satisfy 0 <= lo <= hi <= 1, got {band}")
    rng = random.Random(base_seed * 100 + _SEED_OFFSET_RATE)
    # Quantised to whole percents: the rate is reported to humans and read back
    # off a demo panel, and 0.13 is legible where 0.1318744 is noise.
    return round(rng.uniform(lo, hi), 2), "derived"


@dataclass
class ScheduledPut:
    """One object, and when the schedule says to write it."""
    key: str
    bucket: str
    round_index: int
    # Source file on disk. Carried so the executor can PUT straight from the
    # schedule rather than re-deriving it and risking a mismatch.
    path: str = ""
    # None for clean objects; the sync event id for copied ones.
    event: int | None = None
    mechanism: str = "clean"


@dataclass
class Schedule:
    rounds: list[list[ScheduledPut]] = field(default_factory=list)
    events: list[list[ScheduledPut]] = field(default_factory=list)
    tick_secs: float = MINIO_TIMESTAMP_RESOLUTION_SECS
    notes: list[str] = field(default_factory=list)

    @property
    def total_objects(self) -> int:
        return sum(len(r) for r in self.rounds) + sum(len(e) for e in self.events)

    def estimated_seconds(self) -> float:
        groups = len(self.rounds) + len(self.events)
        return max(groups - 1, 0) * self.tick_secs


def authoring_depths(nodes: list[dict]) -> dict[str, int]:
    """Provenance depth per node: 0 for a root, else 1 + the deepest parent.

    Recomputed from `parents` rather than trusted from the file. A ground truth
    carrying a stale `depth` would otherwise produce a schedule in which a child
    is written before its parent, leaving the corpus disagreeing with its own
    answer key in a way nothing downstream checks.
    """
    parents = {n["node_id"]: list(n.get("parents") or []) for n in nodes}
    depth: dict[str, int] = {}

    def resolve(node_id: str, seen: frozenset = frozenset()) -> int:
        if node_id in depth:
            return depth[node_id]
        if node_id in seen:
            raise ValueError(f"cycle in provenance DAG at {node_id!r}")
        ps = [p for p in parents.get(node_id, []) if p in parents]
        d = 0 if not ps else 1 + max(resolve(p, seen | {node_id}) for p in ps)
        depth[node_id] = d
        return d

    for nid in parents:
        resolve(nid)
    return depth


def authoring_order(nodes: list[dict]) -> list[str]:
    """A causal write order for one chain: a topological order of the DAG,
    sorted by (depth, node_id) so a parent always precedes its child."""
    depth = authoring_depths(nodes)
    return sorted(depth, key=lambda n: (depth[n], n))


def build_schedule(
    objects: list[dict],
    *,
    copy_rate: float = 0.20,
    sync_events: int = 3,
    tick_secs: float = MINIO_TIMESTAMP_RESOLUTION_SECS,
    seed: int = 0,
) -> Schedule:
    """Plan the write order.

    `objects` is one dict per object with keys `key`, `bucket`, `chain` and
    `node_id`, plus `nodes` -- the chain's ground-truth node list. Objects
    without a chain (loose corpus files with no provenance) get round 0 and are
    never corrupted: there is no authoring order to be wrong about.

    Corruption is sampled GLOBALLY, not per chain. A prefix migration does not
    respect document boundaries, and sampling per chain would quietly guarantee
    every chain keeps a clean majority -- which is the assumption that makes
    corruption look easier to survive than it is.
    """
    if not 0.0 <= copy_rate <= 1.0:
        raise ValueError(f"copy_rate must be in [0, 1], got {copy_rate}")
    if copy_rate > 0 and sync_events < 1:
        raise ValueError("copy_rate > 0 requires at least one sync event")

    # Own sub-stream, so which files are corrupted is derived from the base seed
    # independently of how the rate was derived from it.
    rng = random.Random(seed * 100 + _SEED_OFFSET_VICTIMS)
    notes: list[str] = []

    # Round index is provenance DEPTH, not position in the authoring list.
    #
    # Siblings share a depth and therefore a round, so they tie. That is the
    # honest encoding: the ground truth asserts no order between siblings, and
    # writing one before the other would inject an ordering the answer key does
    # not contain. It also makes wall clock scale with the deepest chain rather
    # than with the object count.
    depth_of: dict[str, dict[str, int]] = {}
    for obj in objects:
        chain = obj.get("chain")
        if chain and chain not in depth_of and obj.get("nodes"):
            depth_of[chain] = authoring_depths(obj["nodes"])

    placed: list[ScheduledPut] = []
    for obj in objects:
        chain, node = obj.get("chain"), obj.get("node_id")
        depths = depth_of.get(chain or "", {})
        r = depths.get(node, 0)
        placed.append(ScheduledPut(key=obj["key"], bucket=obj["bucket"],
                                   path=str(obj.get("path", "")), round_index=r))

    eligible = [p for p, o in zip(placed, objects) if o.get("chain") and o.get("node_id")]
    n_copy = int(round(len(eligible) * copy_rate))
    copied = set()
    if n_copy and sync_events >= 1:
        victims = rng.sample(eligible, min(n_copy, len(eligible)))
        for n, put in enumerate(victims):
            put.event = n % sync_events
            put.mechanism = "copy_sync"
            copied.add(id(put))
        notes.append(
            f"{len(victims)} of {len(eligible)} versioned objects assigned to "
            f"{sync_events} sync event(s) (copy_rate={copy_rate})"
        )
    if len(placed) != len(eligible):
        notes.append(
            f"{len(placed) - len(eligible)} object(s) have no chain ground truth; "
            "written in round 0 and never corrupted"
        )

    max_round = max((p.round_index for p in placed if id(p) not in copied), default=0)
    rounds: list[list[ScheduledPut]] = [[] for _ in range(max_round + 1)]
    events: list[list[ScheduledPut]] = [[] for _ in range(sync_events if n_copy else 0)]
    for put in placed:
        if put.event is None:
            rounds[put.round_index].append(put)
        else:
            events[put.event].append(put)

    # LOUD and checked FIRST, because this is the defect that makes the whole
    # schedule pointless, not a lesser cousin of the MinIO-resolution note
    # below. At the module's own default (tick=1.0s) this fires on every run:
    # the resolver's default margin is 60s, so nothing this module writes at
    # a 1s tick is visible to it until CSI_IDF_META_MARGIN_SECS is lowered on
    # whichever service resolves versions.
    if tick_secs <= RESOLVER_ABSTENTION_MARGIN_SECS:
        notes.append(
            f"RESOLVER WILL SEE NO SIGNAL: tick={tick_secs}s does not clear "
            f"PyCaber's storage-timestamp abstention margin "
            f"(CSI_IDF_META_MARGIN_SECS, default "
            f"{RESOLVER_ABSTENTION_MARGIN_SECS}s -- see "
            f"RESOLVER_ABSTENTION_MARGIN_SECS in this module). "
            f"storage_timestamp_constraint() abstains on any pair whose "
            f"timestamps differ by <= the margin, so at this tick EVERY pair "
            f"in this schedule falls inside it and the resolver derives ZERO "
            f"ordering constraints -- these carefully-scheduled timestamps "
            f"carry no signal at all unless CSI_IDF_META_MARGIN_SECS is set "
            f"below {tick_secs}s on the service that resolves versions."
        )
    if tick_secs < MINIO_TIMESTAMP_RESOLUTION_SECS:
        notes.append(
            f"tick={tick_secs}s is below MinIO's {MINIO_TIMESTAMP_RESOLUTION_SECS}s "
            "LastModified resolution, so rounds will TIE. This simulates a batch "
            "writer and destroys ordering signal -- deliberate only. (A lesser, "
            "narrower concern than the resolver-abstention note above: that one "
            "fires even when rounds do NOT tie.)"
        )
    notes.append(SKEW_NOT_MODELLED)
    return Schedule(rounds=rounds, events=events, tick_secs=tick_secs, notes=notes)


def objects_from_uploads(uploads) -> list[dict]:
    """Attach chain provenance to an upload plan, by reading the
    `ground_truth.json` sitting beside each source file.

    Deliberately additive: `_plan_uploads` is not modified and knows nothing
    about this, so a corpus with no ground truth seeds exactly as before. An
    object whose directory has no ground truth carries no provenance, lands in
    round 0, and is never corrupted -- there is no authoring order for it to be
    wrong about.
    """
    cache: dict[Path, list[dict] | None] = {}
    objects = []
    for up in uploads:
        source = Path(up.path)
        chain_dir = source.parent
        if chain_dir not in cache:
            gt = chain_dir / "ground_truth.json"
            try:
                cache[chain_dir] = json.loads(gt.read_text()).get("nodes") or None
            except (OSError, ValueError):
                cache[chain_dir] = None
        nodes = cache[chain_dir]
        entry = {"key": up.key, "bucket": up.bucket, "path": up.path}
        if nodes:
            entry["chain"] = str(chain_dir)
            entry["node_id"] = source.stem
            entry["nodes"] = nodes
        objects.append(entry)
    return objects


def plan_artifact(schedule: Schedule, *, copy_rate: float, sync_events: int,
                  seed: int, copy_rate_source: str = "explicit",
                  copy_rate_band: tuple[float, float] = COPY_RATE_BAND) -> dict:
    """The corruption plan, as a labelled artifact.

    A demo that rewires a version DAG because a timestamp was wrong has to be
    able to say WHICH timestamp and WHY. This is that record: every object, the
    round or sync event it was written in, and the mechanism that put it there.

    `copy_rate_source` and `copy_rate_band` are recorded because a derived rate
    read back off a demo panel is otherwise indistinguishable from a
    misconfigured one, and because the band is what makes one seed's rate
    comparable to another's.
    """
    # `group_index` is the position of an object's write group in the schedule,
    # counting rounds first and then sync events. It makes the plan
    # SELF-DESCRIBING: a consumer recovers the timestamp an object would receive
    # as `base + group_index * tick_secs`, with no need to re-derive the
    # schedule. PyCaber's bench reads exactly this, and a cross-repo consumer
    # re-implementing the rule is a drift source, so the rule ships in the data.
    def _entry(p, group_index):
        d = asdict(p)
        d["group_index"] = group_index
        return d

    n_rounds = len(schedule.rounds)
    return {
        "schema": "demo_timestamp_plan.v1",
        "copy_rate": copy_rate,
        "copy_rate_source": copy_rate_source,
        "copy_rate_band": list(copy_rate_band),
        "sync_events": sync_events,
        "seed": seed,
        "tick_secs": schedule.tick_secs,
        "estimated_seconds": schedule.estimated_seconds(),
        "total_objects": schedule.total_objects,
        "notes": schedule.notes,
        "rounds": [[_entry(p, i) for p in r] for i, r in enumerate(schedule.rounds)],
        "events": [[_entry(p, n_rounds + i) for p in e]
                   for i, e in enumerate(schedule.events)],
    }


def realism_enabled() -> bool:
    """Whether a seed.py should write on a schedule instead of in one burst.

    Defaults ON. The legacy burst PUTs every object back to back, and for a
    CaberDataFactory corpus its order is `sorted()` by filename, which is the
    vNN creation order and therefore a topological order of the provenance
    DAG. That leaves it in one of two regimes depending only on corpus size
    and disk speed: every object ties and the resolver gets no timestamp
    evidence, or the PUTs span a tick and LastModified becomes a CORRECT
    linear extension of the ground truth. It can never be WRONG, only absent
    or true, so it can only flatter -- and the crossover is silent. Set
    DEMO_TIMESTAMP_REALISM=0 for the fast path, knowing that.

    One function for every seed.py that offers this switch, so "realism is on
    by default" is a single fact rather than one asserted per caller and
    liable to drift between them.
    """
    return os.getenv(ENV_TIMESTAMP_REALISM, "1") == "1"


def realism_settings() -> tuple[dict, str, tuple[float, float]]:
    """Resolve one run's settings from one base seed.

    Same seed -> same rate AND same victim files. Different seed -> both
    change, so two runs differ in how much of the corpus is unreliable as
    well as in which parts of it are. An explicit DEMO_TS_COPY_RATE pins the
    rate and is honoured exactly, which is how a sweep holds it fixed while
    varying the seed.

    Returns `(cfg, rate_source, band)` where `cfg` is ready to splat into
    `build_schedule(objects, **cfg)`.
    """
    base_seed = int(os.getenv(ENV_TS_SEED, str(DEFAULT_BASE_SEED)))
    raw_rate = os.getenv(ENV_TS_COPY_RATE)
    band = COPY_RATE_BAND
    raw_band = os.getenv(ENV_TS_COPY_RATE_BAND)
    if raw_band:
        lo, _, hi = raw_band.partition(",")
        band = (float(lo), float(hi))
    rate, source = resolve_copy_rate(
        base_seed, float(raw_rate) if raw_rate else None, band)
    return {
        "copy_rate": rate,
        "sync_events": int(os.getenv(ENV_TS_SYNC_EVENTS, "3")),
        "tick_secs": float(os.getenv(ENV_TS_TICK_SECS, "1.0")),
        "seed": base_seed,
    }, source, band


def seed_with_realistic_timestamps(uploads, data_path: Path | str, put_fn, log=print) -> int:
    """PUT `uploads` in depth-ordered rounds, then in sync-event batches.

    The single implementation behind every demo's realism switch: originally
    inlined in agentgov-mimic's seed.py, extracted here so a second seed.py
    wanting the same behaviour does not have to re-derive it (and cannot
    quietly drift from it). Callers supply `put_fn(bucket, key, path) -> int`
    (0 on success, matching each seed.py's existing `_put` helper) and a `log`
    callable, so this function stays free of any particular S3 client or log
    prefix.

    MinIO assigns LastModified at PUT time, so the write schedule IS the
    timestamp model -- nothing here forges metadata, and the connector, the
    observation record and the governance write all see exactly what they
    would see in production.

    `uploads` items need only `.bucket`, `.key` and `.path` attributes
    (`objects_from_uploads` reads no more than that), so any seed.py's own
    Upload-like record works without adapting it to this module's types.
    """
    data_path = Path(data_path)
    try:
        cfg, rate_source, band = realism_settings()
        objects = objects_from_uploads(uploads)
        schedule = build_schedule(objects, **cfg)
    except ValueError as err:
        log(f"ERROR: invalid timestamp realism settings: {err}")
        return 1

    rate_note = (f"{cfg['copy_rate']} ({rate_source} from seed "
                 f"{cfg['seed']}, band {band[0]}-{band[1]})"
                 if rate_source == "derived" else f"{cfg['copy_rate']} (pinned)")
    log(f"  [timestamps] realism ON: copy_rate={rate_note} "
        f"sync_events={cfg['sync_events']} tick={cfg['tick_secs']}s "
        f"seed={cfg['seed']}")
    log(f"  [timestamps] {len(schedule.rounds)} authoring round(s) + "
        f"{len(schedule.events)} sync event(s), "
        f"~{schedule.estimated_seconds():.0f}s of seed wall clock")
    for note in schedule.notes:
        log(f"  [timestamps] {note}")

    groups = [("round", i, g) for i, g in enumerate(schedule.rounds)]
    groups += [("sync-event", i, g) for i, g in enumerate(schedule.events)]
    for n, (kind, idx, group) in enumerate(groups):
        if not group:
            continue
        if n:
            time.sleep(cfg["tick_secs"])
        log(f"    [{kind} {idx}] {len(group)} object(s)")
        for put in group:
            rc = put_fn(put.bucket, put.key, put.path)
            if rc != 0:
                return rc

    plan_path = Path(os.getenv(ENV_TS_PLAN) or (data_path / "timestamp_plan.json"))
    try:
        # tick_secs is carried on the schedule, not passed again here.
        write_plan(plan_path, plan_artifact(
            schedule, copy_rate=cfg["copy_rate"],
            sync_events=cfg["sync_events"], seed=cfg["seed"],
            copy_rate_source=rate_source, copy_rate_band=band))
        log(f"  [timestamps] plan written to {plan_path}")
    except OSError as err:
        # The corpus is already in MinIO at this point. Losing the plan costs
        # traceability, not correctness, so say so loudly and keep the seed.
        log(f"  [timestamps] WARNING: could not write plan to {plan_path}: {err}. "
            f"The corpus is seeded but no record of which objects were corrupted "
            f"exists, so a wrong version DAG cannot be traced to its cause.")
    return 0


def corpus_objects(corpus_dir: Path | str) -> list[dict]:
    """Enumerate versioned objects straight from a factory corpus tree.

    The offline counterpart to `objects_from_uploads`: it needs no S3 client and
    no upload plan, so a plan can be generated for measurement without standing
    up MinIO. Keys are corpus-relative paths, which are unique where a bare
    chain slug is not -- 10 slugs repeat across the generated corpora on disk.
    """
    corpus_dir = Path(corpus_dir)
    objects = []
    for gt in sorted(corpus_dir.rglob("ground_truth.json")):
        try:
            nodes = json.loads(gt.read_text()).get("nodes") or None
        except (OSError, ValueError):
            continue
        if not nodes:
            continue
        chain_dir = gt.parent
        for md in sorted(chain_dir.glob("*.md")):
            objects.append({
                "key": str(md.relative_to(corpus_dir)),
                "bucket": chain_dir.name,
                "path": str(md),
                "chain": str(chain_dir),
                "node_id": md.stem,
                "nodes": nodes,
            })
    return objects


def main(argv=None) -> int:
    """Generate a corruption plan for a corpus without writing anything to S3."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--corpus", required=True, help="corpus root to enumerate")
    ap.add_argument("--out", required=True, help="where to write the plan JSON")
    ap.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    ap.add_argument("--copy-rate", type=float, default=None,
                    help="pin the rate; omit to derive it from --seed")
    ap.add_argument("--sync-events", type=int, default=3)
    ap.add_argument("--tick-secs", type=float, default=MINIO_TIMESTAMP_RESOLUTION_SECS)
    args = ap.parse_args(argv)

    objects = corpus_objects(args.corpus)
    if not objects:
        print(f"ERROR: no chains with ground_truth.json under {args.corpus}")
        return 1
    rate, source = resolve_copy_rate(args.seed, args.copy_rate)
    schedule = build_schedule(objects, copy_rate=rate, sync_events=args.sync_events,
                              tick_secs=args.tick_secs, seed=args.seed)
    plan = plan_artifact(schedule, copy_rate=rate, sync_events=args.sync_events,
                         seed=args.seed, copy_rate_source=source)
    write_plan(args.out, plan)
    corrupted = sum(len(e) for e in schedule.events)
    print(f"{len(objects)} objects, {len(schedule.rounds)} rounds, "
          f"{len(schedule.events)} sync event(s), "
          f"{corrupted} corrupted (rate {rate}, {source}) -> {args.out}")
    return 0


def write_plan(path: Path | str, plan: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
