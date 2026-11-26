"""Reading the frozen library at run time. **Imports nothing from the system under test.**

This is the module that makes the standalone property real: everything downstream of the build --
scoring, gates, reports, the explorer, the entire default test suite -- goes through here and sees
only data.

Checksums are verified on load, not on request. A cohort whose bytes have drifted from the
manifest is not a cohort with a caveat; it is a different cohort wearing the same identity, and
every number computed from it would be attributed to the wrong specification.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .spec import CohortSpec


class LibraryError(RuntimeError):
    """The frozen library is missing, incomplete, or does not match its manifest."""


def default_root(version: str = "v2") -> str:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, "data", "library", version)


class Library:
    """The ordered, checksummed cohort library."""

    def __init__(self, root: str | None = None, version: str = "v2"):
        self.root = root or default_root(version)
        mpath = os.path.join(self.root, "manifest.json")
        if not os.path.exists(mpath):
            raise LibraryError(
                f"no manifest at {mpath}. The cohort arrays are not committed (they are large and "
                "deterministic); build them once with:\n"
                "    python -m srbench.library.build\n"
                "which takes about 20 minutes and reproduces the committed checksums."
            )
        with open(mpath, encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        self.specs = [CohortSpec.from_dict(d) for d in self.manifest["specs"]]
        self._by_id = {s.cohort_id: s for s in self.specs}
        self._cache: dict[str, tuple] = {}

    # ------------------------------------------------------------------ identity
    @property
    def version(self) -> str:
        return self.manifest["library_version"]

    @property
    def tiers(self) -> dict:
        return self.manifest["tiers"]

    def __len__(self) -> int:
        return len(self.specs)

    def spec(self, cohort_id: str) -> CohortSpec:
        if cohort_id not in self._by_id:
            raise LibraryError(f"no cohort {cohort_id!r} in library {self.version}")
        return self._by_id[cohort_id]

    # ------------------------------------------------------------------ data
    def data(self, cohort_id: str, verify: bool = True):
        """(ages, events) for one cohort, checksum-verified."""
        if cohort_id in self._cache:
            return self._cache[cohort_id]
        p = os.path.join(self.root, "cohorts", f"{cohort_id}.npz")
        if not os.path.exists(p):
            raise LibraryError(
                f"cohort data missing: {p}\nRun `python -m srbench.library.build` to regenerate; "
                "the manifest's checksums make the rebuild verifiable.")
        with np.load(p) as z:
            ages = np.asarray(z["ages"], dtype=float)
            events = np.asarray(z["events"], dtype=int)
        if verify:
            from .build import cohort_digest
            want = self.manifest["checksums"].get(cohort_id)
            got = cohort_digest(ages, events)
            if want and got != want:
                raise LibraryError(
                    f"{cohort_id}: checksum mismatch.\n  manifest: {want[:16]}\n  on disk : "
                    f"{got[:16]}\nThis is not a cohort with a caveat -- it is different data "
                    "wearing the same identity. Rebuild, or use the library version that matches."
                )
        self._cache[cohort_id] = (ages, events)
        return ages, events

    def observation(self, cohort_id: str, reveal=()):
        """What a method is shown: truth stripped, window applied.

        `reveal` is the method's `descriptor.knows` -- the observation-model levels it asked to be
        told. Default empty, so an unaware caller gets exactly the observation this has always
        produced, with the same digest.
        """
        from .observation import observation_for
        ages, events = self.data(cohort_id)
        return observation_for(self.spec(cohort_id), ages, events, reveal=reveal)

    def truth(self, cohort_id: str) -> dict[str, float]:
        return self.spec(cohort_id).physical_truth()

    def truths(self, cohort_ids=None) -> dict[str, dict[str, float]]:
        ids = list(cohort_ids) if cohort_ids is not None else [s.cohort_id for s in self.specs]
        return {cid: self.truth(cid) for cid in ids}

    def cluster_map(self, cohort_ids=None) -> dict[str, str]:
        ids = list(cohort_ids) if cohort_ids is not None else [s.cohort_id for s in self.specs]
        return {cid: self.spec(cid).cluster_key for cid in ids}

    # ------------------------------------------------------------------ selection
    def select(self, n: int) -> list[CohortSpec]:
        from .order import prefix
        return prefix(self.specs, n)

    def groups(self) -> dict[str, int]:
        """`{group: count}`. "main" is the tiered body; anything else is an optional block."""
        out: dict[str, int] = {}
        for sp in self.specs:
            out[sp.group] = out.get(sp.group, 0) + 1
        return out

    def group(self, name: str) -> list[CohortSpec]:
        """Every cohort in one group, in library order.

        A group outside "main" is reachable ONLY through here -- never as a prefix -- so adding one
        cannot change what any tier means. That is what makes an optional block safe to append.
        """
        out = [sp for sp in self.specs if sp.group == name]
        if not out:
            raise LibraryError(
                f"library {self.version} has no group {name!r}; it has {sorted(self.groups())}")
        return out

    def tier(self, name: str) -> list[CohortSpec]:
        """Tier boundaries come from the MANIFEST, so they describe this library rather than the
        planned one. `full` in particular shrinks when the build excludes a cohort."""
        tiers = self.manifest.get("tiers") or {}
        if name not in tiers:
            raise LibraryError(f"library {self.version} has no tier {name!r}; "
                               f"it has {sorted(tiers)}")
        return self.select(int(tiers[name]))

    # ------------------------------------------------------------------ integrity
    def verify_all(self, progress=None) -> dict:
        """Verify every cohort against its checksum. Reports counts; never repairs silently."""
        ok = bad = missing = 0
        bad_ids = []
        for i, s in enumerate(self.specs):
            try:
                self.data(s.cohort_id, verify=True)
                ok += 1
            except LibraryError as ex:
                if "missing" in str(ex):
                    missing += 1
                else:
                    bad += 1
                    bad_ids.append(s.cohort_id)
            if progress and (i + 1) % 200 == 0:
                progress(i + 1, len(self.specs))
        return {"ok": ok, "mismatched": bad, "missing": missing,
                "mismatched_ids": bad_ids[:20], "total": len(self.specs)}
