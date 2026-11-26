"""Spec 003d, M4 (Naveh 2026-10-12): do library v5's homogeneous cohorts rebuild byte for byte without SRK?

Each v5 cohort without heterogeneity is regenerated with SRK's committed `materialize` logic (SRK cb191f5,
`SRK/benchmark/cohorts.py` 462-473 and 625-689: inverse CDF from `fpt_full_density` on the generation grid, the
competing external hazard, censoring at the sample quantile, floor then clamp to delta/2), with sraging's verbatim
`compat.v033.fp_likelihood.fpt_full` as the solver, then the tester's first-bin correction
(`srbench/library/build.py:correct_first_bin`), and checksummed with the tester's `cohort_digest`. The tester's
manifest and arrays are read-only; SRK is never imported (its cohorts module writes a cache into SRK's tree).

A match means the stored v5 arrays are reproducible from sraging alone; it is also the backward-Euler control arm of
spec 003d (the same uniforms, the old grid). Heterogeneous cohorts (332, 40 solves each) are not in M4.

    .venv/bin/python tools/v5_reproduce_003d.py            # needs a clean committed tree
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

TESTER = os.environ.get("SRAGING_TESTER_MAIN", os.path.join(os.environ.get("SRAGING_CODE5") or "/SRAGING_CODE5-is-unset",
                                                            "sr_calibration_tester"))   # release 0.1.1
V5 = os.path.join(TESTER, "data", "library", "v5")
GEN = {"N": 1500, "dt": 5e-4, "tau_max": 30.0, "Gtol": 1e-12}   # SRK cohorts.py GEN_N, GEN_DT, GEN_TAU_MAX, GEN_GTOL
SEED_DOC = "docs/planning/spec003d_seed.md"


def homogeneous(spec) -> bool:
    return not (spec.get("het_param") and (spec.get("het_var") or 0) > 0)


def curve(rho_eta, rho_beta, rho_eps, kappa):
    """SRK's fpt_full_density (tau, S) on the generation grid: S forced monotone and clipped to [0, 1]."""
    from sraging.compat.v033.fp_likelihood import fpt_full
    _m, _cv, _s, tau, S = fpt_full(rho_eta, rho_beta, rho_eps, kappa, return_survival=True, scheme="be", nstart=20,
                                   **GEN)
    return np.asarray(tau, float), np.clip(np.minimum.accumulate(np.asarray(S, float)), 0.0, 1.0)


def materialize_srk(spec, tau, S):
    """SRK's committed materialize for a clean homogeneous spec, then the tester's first-bin correction."""
    rng = np.random.default_rng(spec["seed"])
    n = int(spec["n"])
    t = np.interp(1.0 - rng.random(n), S[::-1], tau[::-1]) * spec["L_phys"]
    e = np.ones(t.size, int)
    if spec.get("external_hazard") is not None:
        h = float(np.exp(-spec["external_hazard"]))
        if h > 0:
            t_ext = rng.exponential(scale=1.0 / h, size=t.size)
            t = np.where(t_ext < t, t_ext, t)
    f = spec.get("censor_frac") or 0.0
    if f > 0:
        c = float(np.quantile(t, 1.0 - f))
        cens = t > c
        t, e = np.where(cens, c, t), np.where(cens, 0, e)
    d = spec.get("delta")
    if d:
        t = np.floor(t / d) * d
        t = np.maximum(t, d * 0.5)
    order = np.argsort(t)
    ages, events = t[order], e[order]
    return correct_first_bin(ages, d), np.asarray(events, dtype=int)


def correct_first_bin(ages, delta):
    """The tester's correction: a clamped first-bin death (at delta/2) goes back to 0."""
    if not delta:
        return ages
    a = np.asarray(ages, dtype=float)
    return np.where(np.abs(a - 0.5 * float(delta)) < 1e-12, 0.0, a)


def cohort_digest(ages, events) -> str:
    """The tester's checksum: sha256 over repr'd float ages, '#', then the int events."""
    h = hashlib.sha256()
    h.update(b"|".join(repr(float(a)).encode() for a in np.asarray(ages)))
    h.update(b"#")
    h.update(b"|".join(str(int(e)).encode() for e in np.asarray(events)))
    return h.hexdigest()


def compare(cid, ages, events):
    """Where a rebuilt cohort differs from the stored v5 arrays (read-only)."""
    z = np.load(os.path.join(V5, "cohorts", f"{cid}.npz"))
    a0, e0 = z["ages"], z["events"].astype(int)
    if a0.shape != ages.shape:
        return {"n_stored": int(a0.size), "n_rebuilt": int(ages.size)}
    return {"ages_differ": int((a0 != ages).sum()), "events_differ": int((e0 != events).sum()),
            "max_abs_age_diff": float(np.max(np.abs(a0 - ages))) if a0.size else 0.0}


def main(argv=None):
    import acceptance_003 as A
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev", action="store_true", help="a development run on a dirty tree (never counts)")
    ap.add_argument("--limit", type=int, help="the first N homogeneous cohorts only (a development run)")
    args = ap.parse_args(argv)
    man_path = os.path.join(V5, "manifest.json")
    raw = open(man_path, "rb").read()
    man = json.loads(raw)
    specs = [s for s in man["specs"] if homogeneous(s)]
    if args.limit:
        specs = specs[:args.limit]
    points = sorted({(s["rho_eta"], s["rho_beta"], s["rho_eps"], s["kappa"]) for s in specs})
    run = A.Run("v5-reproduce", manifests=[SEED_DOC], dev=args.dev or bool(args.limit), suffix="003d",
                cost=f"{len(specs)} homogeneous v5 cohorts at {len(points)} parameter points: one backward-Euler solve "
                     f"per point on SRK's generation grid {GEN} (~5 CPU-min, sr-manager estimate)")
    curves = {p: curve(*p) for p in points}
    rows, mism = [], []
    for s in specs:
        ages, events = materialize_srk(s, *curves[(s["rho_eta"], s["rho_beta"], s["rho_eps"], s["kappa"])])
        dig = cohort_digest(ages, events)
        ok = dig == man["checksums"][s["cohort_id"]]
        rows.append({"cohort_id": s["cohort_id"], "arm": s["arm"], "group": s.get("group"), "match": ok})
        if not ok:
            mism.append({"cohort_id": s["cohort_id"], "arm": s["arm"], **compare(s["cohort_id"], ages, events)})
    by = {}
    for r in rows:
        k = by.setdefault(r["arm"], {"n": 0, "match": 0})
        k["n"] += 1
        k["match"] += r["match"]
    tester_head = subprocess.run(["git", "--no-optional-locks", "-C", TESTER, "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
    run.write({"status": "written", "n": len(rows), "n_match": sum(r["match"] for r in rows), "by_arm": by,
               "n_points": len(points), "mismatches": mism, "grid": GEN,
               "v5": {"manifest": man_path, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                      "library_version": man.get("library_version"), "generator": man.get("generator"),
                      "tester_main_head": tester_head},
               "not_covered": "the 332 heterogeneous cohorts"}, name="v5_reproduce")


if __name__ == "__main__":
    main()
