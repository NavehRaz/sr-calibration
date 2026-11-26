"""The result store: the only durable artifact in this system.

Everything else -- runs, scores, verdicts, reports, explorers -- is DERIVED and can be deleted and
rebuilt from what is here. Calibration is the only expensive operation (roughly an hour of CPU
each, thousands per full run); everything downstream is arithmetic that costs seconds. So the
single most valuable property this design can have is that no calibration is ever performed twice.

THREE INVARIANTS, each with a failure it prevents:

  * IMMUTABLE. A record is written once and never rewritten. If a second write under an existing
    key carries a different value, something that affects the result is missing from the key --
    that is a contract violation to be surfaced, not an update to be applied.

  * ATOMIC. Write to a temporary file in the same directory, then `os.replace`. Calibrations are
    dispatched in parallel across a cluster, and a half-written record that parses is worse than
    one that does not.

  * FAILURES ARE STORED. A `status != ok` record is a result. Not storing it means retrying it
    forever, and means it is missing from the denominator -- and a failure quietly absent from a
    denominator is how a method's availability gets overstated.

Layout: one JSON file per record, sharded two levels by the digest so a directory never holds
hundreds of thousands of entries.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field

from ..contract.types import ClaimSet
from .key import StoreKey


class StoreConflict(RuntimeError):
    """A second write under an existing key carried a different result.

    This is not a race to be retried. It means the key is missing something that changes the
    answer -- most often a method that changed behaviour without bumping its `version`.
    """


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    """One persisted calibration: the immutable unit of expensive work."""

    key: StoreKey
    claim_set: ClaimSet
    wall_clock_seconds: float | None = None
    cpu_seconds: float | None = None
    status: str = "ok"
    code_provenance: dict = field(default_factory=dict)
    written_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "key": self.key.to_dict(),
            "claim_set": self.claim_set.to_dict(),
            "wall_clock_seconds": self.wall_clock_seconds,
            "cpu_seconds": self.cpu_seconds,
            "status": self.status,
            "code_provenance": dict(self.code_provenance),
            "written_at": self.written_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationRecord":
        return cls(
            key=StoreKey(
                digest=d["key"]["digest"],
                method_name=d["key"]["method_name"],
                method_version=d["key"]["method_version"],
                cohort_id=d["key"]["cohort_id"],
                library_version=d["key"]["library_version"],
                contract_version=d["key"]["contract_version"],
                observation_digest=d["key"]["observation_digest"],
                settings_digest=d["key"]["settings_digest"],
            ),
            claim_set=ClaimSet.from_dict(d["claim_set"]),
            wall_clock_seconds=d.get("wall_clock_seconds"),
            cpu_seconds=d.get("cpu_seconds"),
            status=d.get("status", "ok"),
            code_provenance=d.get("code_provenance") or {},
            written_at=float(d.get("written_at", 0.0)),
        )

    def _comparable(self) -> dict:
        """The parts that must not differ between two writes of the same key.

        Timings and provenance are EXCLUDED: the same calibration run twice legitimately takes a
        different number of seconds, and on a different machine. Only the claims are the result.
        """
        return self.claim_set.to_dict() | {"status": self.status}


class ResultStore:
    """Append-only, content-addressed, concurrent-safe."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)

    # ------------------------------------------------------------------ paths
    def _path(self, digest: str) -> str:
        return os.path.join(self.root, digest[:2], digest[2:4], f"{digest}.json")

    def has(self, key: StoreKey | str) -> bool:
        return os.path.exists(self._path(str(key)))

    # ------------------------------------------------------------------ read
    def get(self, key: StoreKey | str) -> CalibrationRecord | None:
        p = self._path(str(key))
        if not os.path.exists(p):
            return None
        try:
            with open(p, encoding="utf-8") as fh:
                return CalibrationRecord.from_dict(json.load(fh))
        except (OSError, json.JSONDecodeError, KeyError) as ex:
            raise StoreConflict(
                f"stored record {str(key)[:12]} is unreadable ({type(ex).__name__}: {ex}). "
                "The store is append-only and written atomically, so this should not happen; "
                "it usually means the file was edited or truncated by something else."
            ) from ex

    # ------------------------------------------------------------------ write
    def put(self, record: CalibrationRecord, *, allow_identical: bool = True) -> bool:
        """Store a record. Returns True if it was written, False if an identical one existed.

        Raises `StoreConflict` when an existing record under the same key differs -- see the
        class docstring for why that is a design error rather than a race.
        """
        digest = str(record.key)
        p = self._path(digest)
        existing = self.get(record.key) if os.path.exists(p) else None
        if existing is not None:
            if allow_identical and existing._comparable() == record._comparable():
                return False
            raise StoreConflict(
                f"key {digest[:12]} already holds a DIFFERENT result.\n"
                f"  method   : {record.key.method_name} v{record.key.method_version}\n"
                f"  cohort   : {record.key.cohort_id}\n"
                "The store is immutable, so this means something that changes the answer is not\n"
                "part of the key. The usual cause is a method whose behaviour changed without\n"
                "its `version` being bumped."
            )

        os.makedirs(os.path.dirname(p), exist_ok=True)
        rec = record if record.written_at else _with_time(record)
        # Atomic: same directory (so `replace` stays on one filesystem), then rename.
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(p), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(rec.to_dict(), fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, p)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return True

    # ------------------------------------------------------------------ survey
    def iter_records(self):
        if not os.path.isdir(self.root):
            return
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for fn in sorted(filenames):
                if fn.endswith(".json"):
                    try:
                        with open(os.path.join(dirpath, fn), encoding="utf-8") as fh:
                            yield CalibrationRecord.from_dict(json.load(fh))
                    except (OSError, json.JSONDecodeError, KeyError):
                        continue

    def count(self) -> int:
        return sum(1 for _ in self.iter_records())


def _with_time(record: CalibrationRecord) -> CalibrationRecord:
    return CalibrationRecord(
        key=record.key, claim_set=record.claim_set,
        wall_clock_seconds=record.wall_clock_seconds, cpu_seconds=record.cpu_seconds,
        status=record.status, code_provenance=record.code_provenance,
        written_at=time.time(),
    )


def default_store_root() -> str:
    env = os.environ.get("SRBENCH_STORE")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, "data", "store")
