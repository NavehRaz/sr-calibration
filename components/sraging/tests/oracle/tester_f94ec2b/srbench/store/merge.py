"""Bringing results home: merging one store into another.

Calibration happens on the cluster, scoring happens on a laptop, and the two have separate stores.
This is the seam between them, and it is the step where a careless implementation destroys work.

THREE PROPERTIES, each guarding a specific way this goes wrong:

  * THE SOURCE IS READ-ONLY. Nothing here writes to, moves, or unlinks anything under `src`. The
    cluster store is the record of hundreds of CPU-hours; a merge that "tidies up" after itself
    turns one bad run into permanent loss. Pull as many times as you like -- re-merging is a no-op
    because the destination already holds identical records.

  * RECORDS ARE RE-KEYED ON THE WAY IN, not re-run. A batch computed before a key-schema change
    arrives with old addresses. Its CONTENT is not stale -- the same method at the same version saw
    the same observation and produced the same claims -- so the honest response is to compute its
    address under the current schema and store it there. `migrate_record` does exactly that, from
    components the record already carries. Re-running would be paying twice for one answer.
    (Concretely: cluster job 530882 was in flight when schema 2 landed.)

  * A CONFLICT STOPS NOTHING BUT IS NEVER SWALLOWED. If the destination already holds a DIFFERENT
    result at the same address, that is the store's central invariant reporting a real problem --
    almost always a method whose behaviour changed without its `version` being bumped. The merge
    records it, leaves both copies untouched, and carries on, so one bad cohort cannot block 359
    good ones. The report names them; a non-zero conflict count means investigate before scoring.

The merge is therefore safe to run early, often, and on a partially-finished array.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .migrate import migrate_record
from .store import CalibrationRecord, ResultStore, StoreConflict


@dataclass(frozen=True, slots=True)
class MergeReport:
    scanned: int = 0
    added: int = 0
    already_present: int = 0
    rekeyed: int = 0
    conflicts: int = 0
    unreadable: int = 0
    conflict_ids: tuple = ()
    by_method: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.conflicts == 0 and self.unreadable == 0

    def format(self) -> str:
        lines = [
            f"scanned          {self.scanned}",
            f"added            {self.added}",
            f"already present  {self.already_present}   (re-merging is a no-op)",
            f"re-keyed on entry {self.rekeyed}   (arrived under an older key schema; content"
            f" reused, not re-run)",
            f"unreadable       {self.unreadable}",
            f"CONFLICTS        {self.conflicts}",
        ]
        if self.by_method:
            lines.append("")
            lines.append("added by method:")
            for m, n in sorted(self.by_method.items()):
                lines.append(f"  {m:<16} {n}")
        if self.conflict_ids:
            lines.append("")
            lines.append("CONFLICTING cohorts -- the destination holds a different result for each.")
            lines.append("Nothing was overwritten. Usually a method whose behaviour changed without")
            lines.append("its version being bumped; investigate before scoring.")
            for c in list(self.conflict_ids)[:10]:
                lines.append(f"  {c}")
            if len(self.conflict_ids) > 10:
                lines.append(f"  ... and {len(self.conflict_ids) - 10} more")
        lines.append("")
        lines.append("0 calibrations performed -- a merge only moves results that already exist.")
        return "\n".join(lines)


def iter_store_records(root: str):
    """Yield `(path, record_or_None)` for every JSON file under `root`. Never writes."""
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, encoding="utf-8") as fh:
                    yield path, CalibrationRecord.from_dict(json.load(fh))
            except (OSError, json.JSONDecodeError, KeyError):
                yield path, None


def merge(src: ResultStore | str, dst: ResultStore | str, *,
          dry_run: bool = False, progress=None) -> MergeReport:
    """Copy every record in `src` into `dst`, re-keying any that predate the current schema.

    `src` is never modified. Returns a `MergeReport`; `report.ok` is False if anything conflicted
    or could not be read.
    """
    src_root = src.root if isinstance(src, ResultStore) else os.path.abspath(src)
    dst = dst if isinstance(dst, ResultStore) else ResultStore(dst)
    if os.path.abspath(src_root) == os.path.abspath(dst.root):
        raise ValueError("source and destination are the same store; nothing to merge")
    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"source store does not exist: {src_root}")

    scanned = added = present = rekeyed = conflicts = unreadable = 0
    conflict_ids: list[str] = []
    by_method: dict[str, int] = {}

    for _path, rec in iter_store_records(src_root):
        scanned += 1
        if rec is None:
            unreadable += 1
            continue

        current = migrate_record(rec)
        # Counted here, BEFORE any add/already-present branch. A record that arrived under the old
        # schema was re-keyed whether or not the destination turned out to hold it already --
        # reporting 0 in that case would read as "no old-schema results were seen", which is the
        # opposite of the truth and exactly what someone checks this number to find out.
        if current.key.digest != rec.key.digest:
            rekeyed += 1

        if dst.has(current.key):
            existing = dst.get(current.key)
            if existing is not None and existing._comparable() == current._comparable():
                present += 1
                continue

        if dry_run:
            added += 1
            by_method[current.key.method_name] = by_method.get(current.key.method_name, 0) + 1
            continue

        try:
            wrote = dst.put(current)
        except StoreConflict:
            conflicts += 1
            conflict_ids.append(f"{current.key.method_name}/{current.key.cohort_id}")
            continue

        if wrote:
            added += 1
            by_method[current.key.method_name] = by_method.get(current.key.method_name, 0) + 1
        else:
            present += 1

        if progress and added and added % 100 == 0:
            progress(added, scanned)

    return MergeReport(scanned=scanned, added=added, already_present=present, rekeyed=rekeyed,
                       conflicts=conflicts, unreadable=unreadable,
                       conflict_ids=tuple(conflict_ids), by_method=by_method)
