#!/usr/bin/env python3
"""Seed data for AgentGov demos.

Uploads files from the demo dataset to MinIO with group-prefixed paths
so the F2U identity provider writes the correct group into the object's
permits (``cats/report.md`` -> ``permits: [cats]``). Groups are defined in
the mock-idp directory seed (demos/agentic-tools/identity/directory.json).

Three dataset layouts are supported:

  Bucketed (bucket name = subdir name):
    cats/contracts/*.{md,txt}  -> contracts bucket, key cats/{filename}
    cats/customers/*.{md,txt}  -> customers bucket, key cats/{filename}

  Per-group flat (all files go to DEMO_DEFAULT_BUCKET, default "contracts"):
    cats/*.{md,txt}            -> contracts bucket, key cats/{filename}

  Root flat (a generated corpus with no group or bucket dimension):
    *.{md,txt}                -> contracts bucket, key {FLAT_CORPUS_OWNER}/{filename}

The root-flat layout exists for the factory corpora, which are generated as a
bag of documents: data/sensitive-docs/generated/sensitive_000.md and friends,
with no directory carrying a group or a bucket. Both dimensions still have to
exist in MinIO, so they are supplied here -- see FLAT_CORPUS_OWNER and
DEFAULT_BUCKET for which values and why.

Usage:
    python demos/agentgov-mimic/seed.py [--visible] [--slow-mo MS] [--keep-open]
"""

import argparse
import os
import re
import sys
from collections import namedtuple
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from timestamp_realism import (  # noqa: E402
    realism_enabled, realism_settings, seed_with_realistic_timestamps,
)

MINIO_URL = os.getenv("MINIO_URL", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "madmin")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "caberdev")
OBJECT_STORE_REGION = os.getenv("OBJECT_STORE_REGION", "us-east-1")

# Objects must land in a bucket the demo's config.json describes. This demo
# configures exactly two Object_Sources, contracts and customers
# (Regex.customer_bucket pins the pair), and the compose file pre-creates both.
# A bucket outside that set is not a soft failure: the Kafka scanner logs
# "Configuration does not exist for object storage source" and the objects are
# never indexed, so nothing downstream ever sees them.
DEFAULT_BUCKET = os.getenv("DEMO_DEFAULT_BUCKET", "contracts")

# Group prefix for the root-flat layout, where no directory supplies one.
#
# A group is not optional. The filenameToUser (F2U) Identity Source writes the
# object's permits from the leading path segment of object.id, so an object
# stored at the bucket root carries no permits and no caller's directory
# groups can meet them.
#
# "cats" rather than "dogs" or an extra-owner group: cats is amy's group in
# the mock-idp directory seed, and the policy priority e2e and the CI warm-up
# ask every question as amy (runner/mode_ci/ci_runner.py
# warmup_corpus_owner). An object under dogs/ is denied to amy because her
# groups do not meet `permits: [dogs]`, so the retrieval would return nothing
# and the policy would have no Edge to evaluate -- a corpus present in MinIO
# and invisible to the demo. Override with DEMO_CORPUS_GROUP for a corpus
# meant for a different group; the asking user must be a member of it. The
# choice is logged at seed time either way.
FLAT_CORPUS_OWNER = os.getenv("DEMO_CORPUS_GROUP", "cats")

_p = Path(__file__).resolve().parents; _REPO_ROOT = _p[3] if len(_p) > 3 else Path("/")
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "agentgov-docs"
DATA_DIR = Path(os.getenv("DEMO_DATA_DIR_HOST", str(_DEFAULT_DATA_DIR)))

_TEXT_EXTS = {".md", ".txt"}
_MAX_BUCKET_NAME_LEN = 25

# One object to upload: which bucket, under which key, from which file, and
# which layout rule put it there. `layout` exists purely so the seed log can
# state the shape it found; a count with no layout does not distinguish "4 files
# from a flat generated corpus" from "4 files because three user directories
# were empty".
Upload = namedtuple("Upload", "bucket key path layout")

LAYOUT_ROOT_FLAT = "root-flat"
LAYOUT_BUCKETED = "bucketed"
LAYOUT_USER_FLAT = "user-flat"


def data_dir_source(data_dir_arg=None):
    """Where the corpus path came from, as a phrase for the seed log.

    seed.py is the process that actually reads the corpus, so this is the
    authoritative statement of which corpus a run ingested -- more so than
    anything demo.py prints, because ``--data-dir`` silently outranks
    ``DEMO_DATA_DIR_HOST`` and only this process sees both.
    """
    if data_dir_arg:
        return "--data-dir argument (overrides DEMO_DATA_DIR_HOST)"
    if os.getenv("DEMO_DATA_DIR_HOST"):
        return "DEMO_DATA_DIR_HOST"
    return (f"built-in default {_DEFAULT_DATA_DIR} "
            f"(neither --data-dir nor DEMO_DATA_DIR_HOST was set)")


def format_enumeration(data_path, uploads):
    """Lines stating what was found, under which layout, and where it lands.

    One summary line plus one line per (layout, bucket, owner) group. The owner
    is included because it is not cosmetic: filenameToUser derives the object's
    user from the key's leading segment, and a corpus uploaded under the wrong
    owner is invisible to the demo's questions while being present in MinIO.
    """
    lines = [f"enumerated {len(uploads)} object(s) under {data_path}"]
    groups = {}
    for upload in uploads:
        owner = upload.key.split("/")[0] if "/" in upload.key else "(none)"
        groups.setdefault((upload.layout, upload.bucket, owner), 0)
        groups[(upload.layout, upload.bucket, owner)] += 1
    for (layout, bucket, owner), count in sorted(groups.items()):
        lines.append(f"  layout={layout} bucket={bucket!r} owner={owner!r} "
                     f"files={count}")
    return lines


def log(msg):
    print(f"[seed] {msg}", flush=True)

# ---------------------------------------------------------------------------
# MinIO/S3 Seeding Logic
#   Do NOT use a User-Agent starting with "CSI." — Caber's self-activity filter
#   (hideCSIsvcs) drops all S3 traffic with the CSI prefix to prevent recursive
#   indexing of internal reads. Since PyCaber relies on VersityGW access logs
#   for object discovery, a CSI prefix causes all seeded objects to be discarded.
# ---------------------------------------------------------------------------
def _s3_client():
    adv_config = Config(s3={'addressing_style': 'path'},
                        connect_timeout=7,
                        read_timeout=15)

    return boto3.client(
        "s3",
        endpoint_url=MINIO_URL,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name=OBJECT_STORE_REGION,
        use_ssl=False,
        config=adv_config
    )


def _valid_bucket_name(name):
    if not isinstance(name, str) or not (3 <= len(name) <= _MAX_BUCKET_NAME_LEN):
        return False
    return bool(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])", name))


def _text_files(directory):
    """The .md/.txt files directly inside a directory, in stable order."""
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix in _TEXT_EXTS)


def _plan_uploads(data_path, flat_owner=None):
    """Enumerate every object a corpus should upload, without touching S3.

    Pure directory walk, separated from :func:`seed` so the three layouts can be
    exercised against real directory trees rather than a mocked S3 client, and
    so bucket names can be validated before any client is constructed.

    The layout is decided by directory shape, not configuration, because the
    corpora are produced by different generators that do not know about each
    other:

      ``<group>/<bucket>/*`` bucketed  -- the hand-authored cats/dogs corpus
      ``<group>/*``          per-group flat
      ``*`` at the root      root flat -- a factory corpus (no group, no bucket)

    Root-flat applies ONLY when the corpus has no group directories at all.
    Loose files beside user directories are documentation, not content:
    ``data/agentgov-mimic-minimal/README.md`` is the concrete precedent, and
    uploading it would put a file describing the corpus into the corpus. The
    rule is "directories win", and the skipped files are listed in the notes
    rather than dropped quietly, because a silent drop is how a genuinely
    misplaced document disappears.

    Args:
        data_path: corpus root.
        flat_owner: group prefix for root-flat objects. Defaults to
            :data:`FLAT_CORPUS_OWNER`.

    Returns:
        ``(uploads, notes)``. ``notes`` are operator-facing lines the caller
        logs: every skip, default and empty directory says why it happened, so
        an under-seeded corpus is visible in the seed output rather than only in
        a later coverage failure.

    Raises:
        ValueError: root-flat files exist but no group is available. Not
            defaulted to a group, because guessing one puts the corpus
            somewhere the demo's questions cannot see it and reports success.
    """
    data_path = Path(data_path)
    owner = flat_owner or FLAT_CORPUS_OWNER
    uploads = []
    notes = []

    entries = sorted(data_path.iterdir())
    # The factory's corpus stores (generated, generated-<demo>, generated-corpora)
    # sit beside the cats/dogs groups in data/agentgov-docs, and they are not
    # groups: each corpus in them is seeded by pointing DEMO_DATA_DIR_HOST at it.
    # Walked as a group, a corpus README one level down becomes a bucket named
    # after the corpus, which fails bucket validation and seeds nothing.
    stores = [e for e in entries if e.is_dir()
              and (e.name == "generated" or e.name.startswith("generated-"))]
    for store in stores:
        notes.append(f"skipping {store.name}/: a factory corpus store, not a "
                     f"group directory (seed one of its corpora by pointing "
                     f"DEMO_DATA_DIR_HOST at it)")
    entries = [e for e in entries if e not in stores]
    # Dot-prefixed entries are tooling state, never corpus objects. The factory
    # writes .parser-cache/ into every corpus it generates; treating it as a
    # user directory would invent a user named ".parser-cache".
    for entry in entries:
        if entry.name.startswith("."):
            notes.append(f"skipping {entry.name}: dot-prefixed entries are "
                         f"tooling state (e.g. the factory parser cache), not "
                         f"corpus objects")

    root_files = _text_files(data_path)
    user_dirs = [d for d in entries if d.is_dir() and not d.name.startswith(".")]
    if root_files and user_dirs:
        notes.append(
            f"ignoring {len(root_files)} file(s) at the corpus root "
            f"({', '.join(f.name for f in root_files)}): this corpus has user "
            f"directories ({', '.join(d.name for d in user_dirs)}), so loose "
            f"root files are documentation rather than content. A corpus with "
            f"NO user directories is treated as root-flat instead.")
        root_files = []

    if root_files:
        if not owner:
            raise ValueError(
                f"{data_path} holds {len(root_files)} file(s) at its root with "
                f"no group directory to own them, and no group is configured. "
                f"Set DEMO_CORPUS_GROUP to the directory group the corpus belongs to. "
                f"Refusing to pick one: an object under the wrong group is "
                f"denied to the asking user by the group permits, so the corpus "
                f"would be present in MinIO and invisible to the demo, and the "
                f"seed would report success.")
        notes.append(
            f"root-flat corpus: {len(root_files)} file(s) at the corpus root "
            f"with no group or bucket level. Uploading to bucket "
            f"{DEFAULT_BUCKET!r} under group {owner!r} "
            f"(override with DEMO_DEFAULT_BUCKET / DEMO_CORPUS_GROUP).")
        for filepath in root_files:
            uploads.append(Upload(DEFAULT_BUCKET, f"{owner}/{filepath.name}",
                                  filepath, LAYOUT_ROOT_FLAT))

    for user_dir in entries:
        if not user_dir.is_dir() or user_dir.name.startswith("."):
            continue
        user = user_dir.name
        subdirs = [d for d in sorted(user_dir.iterdir())
                   if d.is_dir() and not d.name.startswith(".")]
        flat_files = _text_files(user_dir)

        if subdirs:
            for bucket_dir in subdirs:
                files = _text_files(bucket_dir)
                if not files:
                    notes.append(f"WARNING: no text/markdown files in {bucket_dir}")
                    continue
                notes.append(f"{user}/{bucket_dir.name}/ -> bucket "
                             f"{bucket_dir.name!r} ({len(files)} files)")
                for filepath in files:
                    uploads.append(
                        Upload(bucket_dir.name, f"{user}/{filepath.name}",
                               filepath, LAYOUT_BUCKETED))
        elif flat_files:
            notes.append(f"{user}/ (flat) -> bucket {DEFAULT_BUCKET!r} "
                         f"({len(flat_files)} files)")
            for filepath in flat_files:
                uploads.append(
                    Upload(DEFAULT_BUCKET, f"{user}/{filepath.name}",
                           filepath, LAYOUT_USER_FLAT))
        else:
            notes.append(f"WARNING: no text/markdown files found under {user_dir}")

    return uploads, notes


def _bucket_validation_errors(uploads):
    """Buckets in a plan whose names MinIO/the demo config would not accept.

    Reported per bucket with an example key so the offending directory is
    identifiable, and checked before the S3 client is built so an invalid name
    fails as a name problem rather than as a create_bucket error.
    """
    errors = []
    seen = set()
    for upload in uploads:
        if upload.bucket in seen or _valid_bucket_name(upload.bucket):
            continue
        seen.add(upload.bucket)
        errors.append(f"{upload.bucket} (e.g. {upload.key})")
    return errors


_ensured_buckets = set()


def _ensure_bucket(s3, bucket):
    if bucket in _ensured_buckets:
        return
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as error:
        error_code = str(error.response.get("Error", {}).get("Code", ""))
        status_code = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if error_code not in {"404", "NoSuchBucket", "NotFound"} and status_code != 404:
            raise
        s3.create_bucket(Bucket=bucket)
        log(f"  Created bucket: {bucket}")
    _ensured_buckets.add(bucket)


def _put(s3, bucket, key, filepath):
    _ensure_bucket(s3, bucket)
    with open(filepath, "rb") as f:
        try:
            resp = s3.put_object(Bucket=bucket, Key=key, Body=f)
        except ClientError as e:
            log(f"    ERROR uploading {key}: {e}")
            return 1
    status = resp.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
    if 200 <= status < 300:
        log(f"    {bucket}/{key}")
        return 0
    log(f"    ERROR uploading {key}: HTTP {status}")
    return 1


# Thin, agentgov-mimic-named wrappers around the shared implementation in
# runner/core/timestamp_realism.py. Kept as separate names (rather than
# importing and calling the shared functions inline at each call site)
# because tests/unit/test_agentgov_seed_timestamp_realism.py binds to these
# exact names via `seed_mod._realism_enabled` etc., and because the archived
# seeders under _archived/demos/ carried the identical wrapper shape.
def _realism_enabled():
    """Whether to write on a schedule instead of in one burst. See
    `timestamp_realism.realism_enabled` for the reasoning."""
    return realism_enabled()


def _realism_settings():
    """Resolve the run's settings from one base seed. See
    `timestamp_realism.realism_settings` for the reasoning."""
    return realism_settings()


def _seed_with_realistic_timestamps(s3, uploads, data_path):
    """PUT in depth-ordered rounds, then in sync-event batches. See
    `timestamp_realism.seed_with_realistic_timestamps` for the reasoning;
    this just binds it to this demo's own `_put` and `log`."""
    return seed_with_realistic_timestamps(
        uploads, data_path,
        put_fn=lambda bucket, key, path: _put(s3, bucket, key, path),
        log=log,
    )


def seed(headless=True, slow_mo=0, keep_open=False, data_dir=None):
    """Upload demo data to MinIO. Returns 0 on success.

    The plan is built and validated in full before the first PUT, so a corpus
    that cannot be seeded correctly fails before half of it is in MinIO.
    """
    data_path = Path(data_dir) if data_dir else DATA_DIR
    source = data_dir_source(data_dir)
    if not data_path.is_dir():
        log(f"ERROR: data directory not found at {data_path} (source: {source})")
        log("Run via demo.py up/ci with --dataset agentgov-docs, or set DEMO_DATA_DIR_HOST")
        return 1

    log("Seeding AgentGov demo data to MinIO...")
    log(f"  MinIO endpoint: {MINIO_URL}")
    # The authoritative corpus line: stated by the process that reads it, with
    # the reason it was chosen. --data-dir outranks DEMO_DATA_DIR_HOST and only
    # this process sees both, so a disagreement with demo.py's [corpus] line is
    # itself diagnostic.
    log(f"  [corpus] source data: {data_path} (source: {source})")

    try:
        uploads, notes = _plan_uploads(data_path)
    except ValueError as err:
        log(f"ERROR: {err}")
        return 1
    for note in notes:
        log(f"  {note}")
    for line in format_enumeration(data_path, uploads):
        log(f"  [corpus] {line}")

    bucket_errors = _bucket_validation_errors(uploads)
    if bucket_errors:
        log("ERROR: invalid bucket names in seed data:")
        for item in bucket_errors:
            log(f"  {item}")
        log("Bucket names must be 3-25 chars, lowercase letters/digits/hyphens, "
            "and start/end with a letter or digit.")
        return 1

    if not uploads:
        # Never a quiet success. A zero count names the directory it looked in
        # and how that directory was chosen, because "seeded nothing" and
        # "looked in the wrong place" are the same line of output otherwise.
        log(f"ERROR: no files were uploaded -- {data_path} (source: {source}) "
            f"contains no .md or .txt files in any supported layout "
            f"(<user>/<bucket>/, <user>/, or at the corpus root)")
        return 1

    s3 = _s3_client()
    if _realism_enabled():
        rc = _seed_with_realistic_timestamps(s3, uploads, data_path)
        if rc != 0:
            return rc
    else:
        # Legacy burst: every PUT back to back, so the whole corpus lands inside
        # a second or two and every version of every document ties. Kept as the
        # default so enabling realism is a deliberate act, but it is NOT a
        # neutral baseline -- see _realism_enabled().
        for upload in uploads:
            rc = _put(s3, upload.bucket, upload.key, upload.path)
            if rc != 0:
                return rc

    log(f"  {len(uploads)} files uploaded to MinIO")
    log("AgentGov seed complete.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Seed AgentGov demo data")
    parser.add_argument("--visible", action="store_true",
                        help="(ignored) accepted for interface consistency")
    parser.add_argument("--slow-mo", type=int, default=0,
                        help="(ignored) accepted for interface consistency")
    parser.add_argument("--keep-open", action="store_true",
                        help="(ignored) accepted for interface consistency")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory")
    args = parser.parse_args()
    sys.exit(seed(headless=not args.visible, slow_mo=args.slow_mo,
                  keep_open=args.keep_open, data_dir=args.data_dir))
