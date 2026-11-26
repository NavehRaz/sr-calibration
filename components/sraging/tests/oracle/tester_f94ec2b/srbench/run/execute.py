"""Running a method over a cohort selection — consulting the store before every calibration.

THE SHAPE THAT MATTERS: a run is a MANIFEST OF STORE KEYS, not a container of results. The
expensive work lives in the store and belongs to nobody; a run only records which keys it wanted.

That is what makes all of this free:

    re-run the same thing            -> 0 calibrations
    grow quick -> medium -> full     -> only the new cohorts
    add a view, a gate, a statistic  -> 0 calibrations
    re-point the reference           -> 0 calibrations

And it is why every run reports RE-USED versus COMPUTED before and after it runs: a number that is
never printed is a number nobody checks.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from ..contract.method import CalibrationMethod
from ..contract.types import ClaimSet, CONTRACT_VERSION
from ..contract.validate import validate_claimset, validate_observation
from ..store.key import make_key
from ..store.store import CalibrationRecord, ResultStore


@dataclass
class RunPlan:
    """What a run intends to do, known BEFORE anything expensive happens."""

    method_name: str
    method_version: str
    library_version: str
    cohort_ids: list[str]
    keys: dict[str, str] = field(default_factory=dict)      # cohort_id -> digest
    reuse: list[str] = field(default_factory=list)
    compute: list[str] = field(default_factory=list)

    @property
    def n_reuse(self) -> int:
        return len(self.reuse)

    @property
    def n_compute(self) -> int:
        return len(self.compute)

    def to_dict(self) -> dict:
        return {"method": self.method_name, "method_version": self.method_version,
                "library_version": self.library_version,
                "n_cohorts": len(self.cohort_ids), "n_reuse": self.n_reuse,
                "n_compute": self.n_compute, "keys": dict(self.keys)}

    def format(self) -> str:
        return (f"cohorts {len(self.cohort_ids)}   already stored {self.n_reuse}   "
                f"to compute {self.n_compute}")


def _reveal(method) -> tuple:
    """What this method asked to be told. Read in ONE place so the plan and the run cannot differ.

    If these two disagreed, `plan_run` would compute one address and `run` another, so every run
    would report full reuse and then recompute everything -- a failure that costs CPU-hours and
    looks like nothing at all.
    """
    return tuple(getattr(method.descriptor, "knows", ()) or ())


def plan_run(method: CalibrationMethod, library, cohort_ids, store: ResultStore,
             settings: dict | None = None) -> RunPlan:
    """Work out what would be reused and what computed, WITHOUT running anything.

    This is the answer to "what will this cost me?" and it is available before committing, which
    is the whole point of storing results in the first place.
    """
    d = method.descriptor
    plan = RunPlan(method_name=d.name, method_version=d.version,
                   library_version=library.version, cohort_ids=list(cohort_ids))
    for cid in cohort_ids:
        obs = library.observation(cid, reveal=_reveal(method))
        key = make_key(method_name=d.name, method_version=d.version, cohort_id=cid,
                       library_version=library.version, contract_version=CONTRACT_VERSION,
                       observation=obs, settings=settings)
        plan.keys[cid] = key.digest
        (plan.reuse if store.has(key) else plan.compute).append(cid)
    return plan


def run(method: CalibrationMethod, library, cohort_ids, store: ResultStore,
        settings: dict | None = None, progress=None, on_record=None) -> dict:
    """Calibrate every cohort not already stored, and return the run manifest.

    Each result is written to the store the MOMENT it completes, so an interruption loses at most
    the calibration in flight -- and a resume is a store lookup rather than a recomputation.
    """
    from ..provenance import harness_provenance

    d = method.descriptor
    plan = plan_run(method, library, cohort_ids, store, settings)
    t0 = time.time()
    computed = failed = 0

    for i, cid in enumerate(plan.compute):
        obs = library.observation(cid, reveal=_reveal(method))
        validate_observation(obs)
        key = make_key(method_name=d.name, method_version=d.version, cohort_id=cid,
                       library_version=library.version, contract_version=CONTRACT_VERSION,
                       observation=obs, settings=settings)
        t1 = time.perf_counter()
        c1 = time.process_time()
        try:
            cs = method.calibrate(obs)
            validate_claimset(cs, d, obs)
            status = cs.status
        except Exception as ex:  # noqa: BLE001
            # A failure is a RESULT. Storing it means it is not retried forever and is not quietly
            # missing from a denominator -- which would overstate the method's availability.
            status = f"error:{type(ex).__name__}"
            cs = ClaimSet(cohort_id=cid, method_name=d.name, status=status,
                          diagnostics={"exception": str(ex)[:500]})
            failed += 1
        wall = time.perf_counter() - t1
        cpu = time.process_time() - c1

        rec = CalibrationRecord(
            key=key, claim_set=cs, status=status,
            wall_clock_seconds=cs.wall_clock_seconds if cs.wall_clock_seconds else wall,
            cpu_seconds=cs.cpu_seconds if cs.cpu_seconds else cpu,
            code_provenance=harness_provenance().to_dict(),
        )
        store.put(rec)
        computed += 1
        if on_record:
            on_record(rec)
        if progress and (i + 1) % 25 == 0:
            progress(i + 1, len(plan.compute), time.time() - t0)

    return {
        "method": d.name, "method_version": d.version,
        "library_version": library.version, "contract_version": CONTRACT_VERSION,
        "cohort_ids": list(cohort_ids), "keys": dict(plan.keys),
        "n_cohorts": len(cohort_ids), "n_reused": plan.n_reuse, "n_computed": computed,
        "n_failed": failed, "wall_seconds": round(time.time() - t0, 2),
        "settings": dict(settings or {}),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
    }


def load_results(manifest: dict, store: ResultStore) -> list[ClaimSet]:
    """The claim sets a run refers to, fetched from the store.

    A missing key is reported rather than skipped: a run whose results silently shrank would
    produce figures with the wrong denominator, which is the error this project keeps making.
    """
    out, missing = [], []
    for cid, digest in manifest["keys"].items():
        rec = store.get(digest)
        if rec is None:
            missing.append(cid)
            continue
        out.append(rec.claim_set)
    if missing:
        raise KeyError(
            f"{len(missing)} of {len(manifest['keys'])} results referenced by this run are not in "
            f"the store (first few: {missing[:5]}). Re-run to fill them; do NOT score a partial "
            "set as if it were whole."
        )
    return out


def save_manifest(manifest: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return path
