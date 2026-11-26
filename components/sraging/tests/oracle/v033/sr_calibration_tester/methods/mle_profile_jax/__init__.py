"""`mle_profile_jax` -- the incumbent estimator on a JAX likelihood with exact-gradient optimisers.

A challenger built to answer one question: does replacing derivative-free Nelder-Mead with exact
JAX gradients make `mle_profile` cheaper WITHOUT changing what it concludes? The idea comes from
Avi's `saturated-removal-manifold` (`scripts/fpt_jax.py`, `scripts/fit_mm_jax.py`), where an exact
`jax.jacfwd` Jacobian replaced a finite-difference one at identical fit quality.

Everything statistical is the reference's: likelihood, grid, seed, starts, box, profile walk,
repair, `profile_ci` verdicts. See `estimator.py` for the two marked differences and `fp_jax.py`
for the likelihood port and how it was checked against the NumPy oracle.

This adapter, like `mle_profile`'s, imports the system under test (`SRK`) -- read-only, for the
seed, the grid rule, Kaplan-Meier and `profile_ci`. The claim mapping below is copied from
`methods/mle_profile` unchanged, so the two methods cannot differ in how a result becomes a claim.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import time

from srbench.contract.method import CohortCharacteristics, register
from srbench.contract.types import Claim, ClaimSet, CostDeclaration, MethodDescriptor, Observation

_HERE = os.path.dirname(os.path.abspath(__file__))

BOUNDS = {
    "eta": (1e-12, 1e3),
    "beta": (1e-12, 1e3),
    "epsilon": (1e-12, 1e3),
    "kappa": (1e-3, 300.0),
}
KAPPA_MAX = 300.0
LEVEL = 0.95

VERDICT_TO_SHAPE = {
    "identified": "closed",
    "one-sided:<": "upper_bound",
    "one-sided:>": "lower_bound",
    "flat": "none",
    "insufficient": "none",
}


def _sibling(name):
    """Load a sibling module by path: discovery imports this package without a parent package, so
    relative imports do not resolve. Cached in sys.modules so jitted functions are built once."""
    key = f"srbench_methods.mle_profile_jax.{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


class MleProfileJax:
    descriptor = MethodDescriptor(
        name="mle_profile_jax",
        # 0.3.0 is DEAD: its 30 stored quick-tier records are all `error:ProvenanceError` -- the runs
        # were launched without SRK on PYTHONPATH, so no calibration ran. Same code as 0.3.1.
        # 0.3.1: v0.3 without the kappa-corner polish starts (quick tier + cluster memory pilot).
        # 0.3.3: v0.3.2 + P1 -- a rejected walk step no longer uses up one of a side's node slots.
        # 0.3.4 is NOT ADOPTED (its 30 quick-tier records stay in the store): 0.3.3 + P2-lean, a
        # bracket polish re-solving both nodes around each edge. Same quick coverage, median CPU
        # 389.5 s = 1.85x v0.3.2 against a 1.6x limit. Its estimator is kept, not run, at
        # reports/jax34-quick/estimator_v0.3.4.py.
        version="0.3.3",
        bounds=BOUNDS,
        # "censoring" is deliberately NOT declared (audited 2026-09-27). A right-censored record does
        # contribute log S(recorded age), value and gradient, matching SRK's NumPy oracle. But under a
        # recording resolution delta, deaths recorded in the same bin as the censoring age are still
        # scored over the whole bin [t, t+delta), although they died before the censoring point.
        # On the library's censor arm (censor at a quantile, then round down) that biases eta upward:
        # closed eta coverage 0.64 vs 0.84 on the paired pure cohorts, and like-for-like only where
        # fewer than 0.5% of the cohort are such tied deaths. mle_profile shares the same term.
        # [T-TESTER-DEFECTS] item 7. Descriptor metadata is not in the store key: no version bump.
        capabilities=(),
        description=("mle_profile's estimator on a JAX Fokker-Planck likelihood (hand-written "
                     "reverse sweep), exact-gradient L-BFGS-B searches, and an edge polish."),
        contact="sr_calibration_tester / mle_profile_jax",
    )

    def cost(self, spec: CohortCharacteristics) -> CostDeclaration:
        """MEASURED, v0.3.3, FULL tier, 2026-09-26: CPU seconds per cohort over all 1,990 cohorts,
        1 core per cohort on WEXAC (30 quick-tier cohorts on the Mac):

            median 588   mean 1,553   p10 335   p90 2,786   max 69,410   total 858 CPU-h

        The tail is the long-march real-anchored cohorts (every E. coli variant 7-19 h: its recording
        resolution forces ~63,000 solver steps; celegans up to ~4 h) and the high-kappa cohorts only the
        full tier contains. On the medium tier the median is 438 s (mle_profile: 2,988 s). Peak memory
        <= 718 MB. Why the tail is slow is open ([T-JAX-SLOW]).
        """
        return CostDeclaration(median_seconds=590.0, min_seconds=140.0, max_seconds=70000.0)

    def calibrate(self, observation: Observation) -> ClaimSet:
        from srbench.provenance import format_code_line, resolve_sut
        prov = resolve_sut()            # refuses on the stale mount; applies the numpy shim
        os.environ.setdefault("SRK_CACHE_READONLY", "1")

        t0, c0 = time.perf_counter(), time.process_time()
        try:
            fj = _sibling("fp_jax")
            est = _sibling("estimator")
            res = est.fit(fj, list(observation.ages), list(observation.events),
                          delta=observation.delta, levels=(LEVEL,), kappa_max=KAPPA_MAX)
        except Exception as ex:  # noqa: BLE001
            return ClaimSet(cohort_id=observation.cohort_id, method_name=self.descriptor.name,
                            status=f"error:{type(ex).__name__}",
                            wall_clock_seconds=time.perf_counter() - t0,
                            cpu_seconds=time.process_time() - c0,
                            diagnostics={"exception": str(ex)[:500]},
                            code_provenance=prov.to_dict())

        theta = list(res["theta"])
        point = {"eta": theta[0], "beta": theta[1], "epsilon": theta[2], "kappa": theta[3]}
        verdicts = (res.get("diagnostics") or {}).get("verdicts") or {}

        # ---- claim mapping: verbatim from methods/mle_profile ----
        claims = []
        for p in ("eta", "beta", "epsilon", "kappa"):
            lo, hi = res["ci"].get(f"{p}@{LEVEL}", (float("nan"), float("nan")))
            shape = VERDICT_TO_SHAPE.get(verdicts.get(p), "none")
            lo_ok = lo is not None and math.isfinite(lo)
            hi_ok = hi is not None and math.isfinite(hi)
            if shape == "closed" and not (lo_ok and hi_ok):
                shape = "upper_bound" if hi_ok else ("lower_bound" if lo_ok else "none")
            if shape == "upper_bound" and not hi_ok:
                shape = "none"
            if shape == "lower_bound" and not lo_ok:
                shape = "none"
            kw = {"parameter": p, "shape": shape, "level": LEVEL}
            if shape in ("closed", "lower_bound"):
                kw["lower"] = float(max(lo, BOUNDS[p][0]))
            if shape in ("closed", "upper_bound"):
                kw["upper"] = float(min(hi, BOUNDS[p][1]))
            if shape != "none":
                v = point[p]
                if math.isfinite(v):
                    kw["point"] = float(min(max(v, BOUNDS[p][0]), BOUNDS[p][1]))
            claims.append(Claim(**kw))

        diag = dict(res.get("diagnostics") or {})
        diag.pop("profiles", None)
        diag["loglik"] = res.get("loglik")
        diag["dt_ceiling"] = res.get("dt_ceiling")
        diag["code"] = format_code_line(prov)

        return ClaimSet(
            cohort_id=observation.cohort_id,
            method_name=self.descriptor.name,
            claims=tuple(claims),
            status="ok",
            wall_clock_seconds=res.get("wall_s") or (time.perf_counter() - t0),
            cpu_seconds=time.process_time() - c0,
            diagnostics=diag,
            code_provenance=prov.to_dict(),
        )


register(MleProfileJax())
