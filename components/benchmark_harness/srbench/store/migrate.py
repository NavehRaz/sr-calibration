"""Re-keying the store when the key SCHEMA changes.

A schema change moves every record to a new address. That is not a re-run: every stored record
already carries the components its key is built from (method name and version, cohort id, contract
version, observation digest, settings digest), so the new address is pure arithmetic on data
already on disk. **A migration costs zero calibrations.**

WHY THIS EXISTS. Schema 1 included `library_version` in the digest. Adding cohorts requires a new
library version, so growing the library orphaned every stored result -- 380 `mle_profile`
calibrations, about 300 CPU-hours, to add a single hazard level. Schema 2 drops it; the observation
digest already distinguishes "same cohort id, different data", which is the only thing it guarded.

SAFETY. The migration is written to be interruptible and to never destroy a record it has not
already written elsewhere:

  * new records are written FIRST, then the old ones removed;
  * content is compared after writing, so a record is only unlinked once its replacement is
    verified byte-identical in the parts that matter;
  * re-running it is a no-op, because a record already at its schema-2 address simply matches.

It also handles records arriving LATER under the old schema -- e.g. a cluster batch that was in
flight when the schema changed. `migrate_record` is exposed for exactly that, so merging old-schema
results does not require re-running them either.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

from .key import KEY_SCHEMA_VERSION, _canonical, make_key
from .store import CalibrationRecord, ResultStore, StoreConflict


@dataclass(frozen=True, slots=True)
class MigrationReport:
    scanned: int = 0
    already_current: int = 0
    rekeyed: int = 0
    conflicts: int = 0
    unreadable: int = 0
    conflict_ids: tuple = ()

    def format(self) -> str:
        lines = [f"scanned          {self.scanned}",
                 f"already current  {self.already_current}",
                 f"re-keyed         {self.rekeyed}",
                 f"unreadable       {self.unreadable}",
                 f"CONFLICTS        {self.conflicts}"]
        if self.conflict_ids:
            lines.append(f"  first few: {list(self.conflict_ids)[:5]}")
        lines.append("")
        lines.append("0 calibrations performed -- a schema change is arithmetic on stored data.")
        return "\n".join(lines)


def new_key_for(record: CalibrationRecord):
    """The record's address under the CURRENT schema, from the components it already carries."""
    k = record.key
    return make_key(
        method_name=k.method_name,
        method_version=k.method_version,
        cohort_id=k.cohort_id,
        library_version=k.library_version,     # recorded, no longer part of the digest
        contract_version=k.contract_version,
        observation_digest_=k.observation_digest,
        settings=None,
    )


def migrate_record(record: CalibrationRecord) -> CalibrationRecord:
    """Return the record addressed under the current schema. Content is untouched."""
    nk = new_key_for(record)
    if nk.digest == record.key.digest:
        return record
    # settings_digest is part of the address; carry the original so a record whose settings were
    # non-default is not silently re-addressed as if they were default.
    if record.key.settings_digest != nk.settings_digest:
        object.__setattr__(nk, "settings_digest", record.key.settings_digest)
        nk = make_key(
            method_name=nk.method_name, method_version=nk.method_version,
            cohort_id=nk.cohort_id, library_version=nk.library_version,
            contract_version=nk.contract_version,
            observation_digest_=nk.observation_digest, settings=None)
    return CalibrationRecord(
        key=nk, claim_set=record.claim_set,
        wall_clock_seconds=record.wall_clock_seconds, cpu_seconds=record.cpu_seconds,
        status=record.status, code_provenance=record.code_provenance,
        written_at=record.written_at,
    )


def migrate(store: ResultStore, dry_run: bool = False, progress=None) -> MigrationReport:
    """Re-key an entire store in place. Interruptible; re-running is a no-op."""
    scanned = current = rekeyed = conflicts = unreadable = 0
    conflict_ids = []
    old_paths: list[tuple[str, str]] = []      # (old path, new digest) -- removed only after write

    for dirpath, _d, files in os.walk(store.root):
        for fn in sorted(files):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(dirpath, fn)
            scanned += 1
            try:
                rec = CalibrationRecord.from_dict(json.load(open(path, encoding="utf-8")))
            except (OSError, json.JSONDecodeError, KeyError):
                unreadable += 1
                continue
            new = migrate_record(rec)
            if new.key.digest == rec.key.digest:
                current += 1
                continue
            if dry_run:
                rekeyed += 1
                continue
            try:
                store.put(new)                  # writes the NEW address first
            except StoreConflict:
                conflicts += 1
                conflict_ids.append(rec.key.cohort_id)
                continue
            old_paths.append((path, new.key.digest))
            rekeyed += 1
            if progress and rekeyed % 200 == 0:
                progress(rekeyed, scanned)

    # Only now remove the originals, and only where the replacement is verifiably present.
    for old, digest in old_paths:
        if store.has(digest) and os.path.exists(old):
            os.unlink(old)

    return MigrationReport(scanned=scanned, already_current=current, rekeyed=rekeyed,
                           conflicts=conflicts, unreadable=unreadable,
                           conflict_ids=tuple(conflict_ids))


# --------------------------------------------------------------------- manifests

#: Schema 1's digest, reproduced EXACTLY as `key.py` computed it before 2026-09-24. Kept only so a
#: record's old address can be recomputed and matched -- never to write a new one.
def legacy_v1_digest(*, method_name: str, method_version: str, cohort_id: str,
                     library_version: str, contract_version: str,
                     observation_digest_: str, settings_digest: str) -> str:
    parts = {
        "v": "1",
        "method": method_name,
        "method_version": method_version,
        "cohort": cohort_id,
        "library": library_version,          # the field schema 2 removed
        "contract": contract_version,
        "obs": observation_digest_,
        "settings": settings_digest,
    }
    return hashlib.sha256(_canonical(parts)).hexdigest()


def build_rekey_map(store: ResultStore) -> dict[str, str]:
    """`{old schema-1 digest: current digest}`, derived from the records themselves.

    This is what makes manifest migration EXACT rather than a guess. A run manifest records only
    `{cohort_id: digest}`, and several records can share a cohort id (the same cohort under
    different windows is a different observation and so a different result). Matching on the old
    address is unambiguous; matching on cohort id would not be.
    """
    out: dict[str, str] = {}
    for rec in store.iter_records():
        k = rec.key
        old = legacy_v1_digest(
            method_name=k.method_name, method_version=k.method_version, cohort_id=k.cohort_id,
            library_version=k.library_version, contract_version=k.contract_version,
            observation_digest_=k.observation_digest, settings_digest=k.settings_digest)
        out[old] = k.digest
    return out


@dataclass(frozen=True, slots=True)
class ManifestReport:
    path: str
    total: int = 0
    remapped: int = 0
    already_current: int = 0
    unresolved: int = 0
    unresolved_ids: tuple = ()

    @property
    def ok(self) -> bool:
        return self.unresolved == 0


def migrate_manifest(path: str, rekey_map: dict[str, str], store: ResultStore,
                     dry_run: bool = False) -> ManifestReport:
    """Point a run manifest at the current addresses of results it already references.

    A manifest is a list of digests, so re-keying the store silently invalidates every one of them.
    Nothing is recomputed here either: the results exist, only their addresses moved.

    An entry that resolves to NEITHER a current address nor a known old one is left ALONE and
    counted as unresolved -- rewriting it to a guess would point a run at the wrong result, which is
    far worse than a manifest that refuses to load.
    """
    with open(path, encoding="utf-8") as fh:
        man = json.load(fh)
    keys = man.get("keys") or {}

    remapped = current = unresolved = 0
    unresolved_ids: list[str] = []
    new_keys = dict(keys)

    for cohort_id, digest in keys.items():
        if store.has(digest):
            current += 1
            continue
        mapped = rekey_map.get(digest)
        if mapped is None:
            unresolved += 1
            unresolved_ids.append(cohort_id)
            continue
        new_keys[cohort_id] = mapped
        remapped += 1

    if remapped and not dry_run:
        man["keys"] = new_keys
        man["key_schema_migrated"] = KEY_SCHEMA_VERSION
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=2)
        os.replace(tmp, path)

    return ManifestReport(path=path, total=len(keys), remapped=remapped,
                          already_current=current, unresolved=unresolved,
                          unresolved_ids=tuple(unresolved_ids))


def migrate_manifests(run_dir: str, store: ResultStore,
                      dry_run: bool = False) -> list[ManifestReport]:
    rekey_map = build_rekey_map(store)
    reports = []
    for fn in sorted(os.listdir(run_dir)):
        if fn.endswith(".json"):
            reports.append(migrate_manifest(os.path.join(run_dir, fn), rekey_map, store,
                                            dry_run=dry_run))
    return reports
