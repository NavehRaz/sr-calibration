"""`mle_profile_jax2` 1.0.0rc1 -- CANDIDATE successor to `mle_profile_jax` v0.3.3. NOT REGISTERED.

Phase A of the jax2 plan: a release candidate run only by the basin/candidate cluster driver
(`reports/basin/candidate/cluster/driver.py`). It is never placed under `methods/` and never calls
`srbench.contract.method.register`, so `srbench methods` / `srbench run` cannot see it, and nothing it
computes can reach a store. Its siblings load under their OWN sys.modules keys
(`srbench_candidate.mle_profile_jax2_rc1.<name>`), so it can share a process with the frozen
`mle_profile_jax` (keys `srbench_methods.mle_profile_jax.<name>`) without either replacing the other.

What it is, against v0.3.3 (8b4c2f1), each change measured before it was put in:

  1. fix B, `fp_jax.py` (md5 5834e106): a death's interval mass from the absorbed mass F = 1 - S where
     that is the difference with the smaller operands -- no more S(t) - S(t+delta) round-off on large,
     sharp cohorts (reports/basin/likelihood_fix);
  2. v1 + v2, `estimator.py`: re-fit on a witnessed better profile node, and library starts for the
     polished edges after a re-fit (reports/basin/fix, fix_v2). Bit-identical where it never fires;
  3. the floored dt ceiling, `estimator.solver_grid` (reports/dt_ceiling);
  4. the censoring-tie relabel, `censor_tie_relabel` below: OFF by default in rc1, switched by
     `run_fit(observation, tie_relabel=True)` / `MleProfileJax2(tie_relabel=True)`. The setting is
     part of `settings()`, which a caller passes as the store key's method settings, so a run with and
     a run without the relabel can never share a key.

The claim mapping is HEAD's `methods/mle_profile_jax/__init__.py` (= `methods/mle_profile`'s) unchanged.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import time

from srbench.contract.types import Claim, ClaimSet, CostDeclaration, MethodDescriptor, Observation

_HERE = os.path.dirname(os.path.abspath(__file__))
_KEY = "srbench_candidate.mle_profile_jax2_rc1"

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

# Censoring-tie relabel: a death is in the delta-bin containing tc when b <= tc < b + delta. The strict
# upper edge is applied with this relative margin so that a recorded age one grid step below an ON-grid
# tc (tc - b = delta up to round-off) is never taken for the bin itself.
_TIE_REL_TOL = 1e-9


def _sibling(name):
    """Load a sibling module by path under this candidate's own sys.modules key (never the frozen
    method's), cached so jitted functions are built once per process."""
    key = f"{_KEY}.{name}"
    if key in sys.modules:
        return sys.modules[key]
    path = os.path.join(_HERE, f"{name}.py")
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    assert os.path.realpath(mod.__file__) == os.path.realpath(path), (mod.__file__, path)
    return mod


def censor_tie_relabel(ages, events, delta, enabled=True):
    """(ages, events, record): the generalised censoring-tie relabel (Naveh's choice (a), 2026-10-02).

    PATTERN. `delta` is set, there is at least one censored record, ALL censored records share ONE age tc,
    and tc equals the maximum recorded age (an end-of-follow-up censoring: the censor arm's study end, a
    realfull study end, or a window close). Then a death recorded at age b with b <= tc < b + delta -- the
    delta-bin that contains tc -- may have died AFTER the censoring point: under rounding-down it died
    somewhere in [b, b + delta), and the censored were alive at tc. Scoring it over the whole bin while the
    censored are credited S(tc) counts the sliver [tc, b + delta) twice (implied total probability > 1),
    which biases eta upward (reports/censor_tie). Every such death is RELABELLED CENSORED AT ITS OWN
    RECORDED AGE b ("alive at b", which is true). On the grid (tc = b) that is "alive at tc", exactly the
    censored records' statement; off the grid (b < tc, a window closing between grid points) it is the
    weaker true statement S(b). Ages are never moved. Needs no knowledge of the censoring point.

    `enabled=False` runs the detector only: the record says what WOULD be relabelled (`n_tied`, `dose`),
    and the data are returned unchanged. The record is `diagnostics.censor_tie`.
    """
    a = [float(x) for x in ages]
    ev = [int(x) for x in events]
    n = len(a)
    cens = [x for x, e in zip(a, ev) if e == 0]
    rec = dict(enabled=bool(enabled), detected=False, applied=False, reason="", n=n,
               n_censored=len(cens), delta=(float(delta) if delta else None), tc=None, on_grid=None,
               n_tied=0, n_relabelled=0, dose=0.0)
    if not delta or not (float(delta) > 0):
        rec["reason"] = "no delta"
        return a, ev, rec
    d = float(delta)
    if not cens:
        rec["reason"] = "no censored records"
        return a, ev, rec
    tc = max(cens)
    if min(cens) != tc:
        rec["reason"] = "censored records at more than one age"
        return a, ev, rec
    if max(a) != tc:
        rec["reason"] = "a record lies beyond the censoring age"
        return a, ev, rec
    rec["tc"] = tc
    q = tc / d
    rec["on_grid"] = bool(abs(q - round(q)) <= 1e-9 * max(1.0, abs(q)))
    tied = [i for i, (x, e) in enumerate(zip(a, ev)) if e == 1 and x <= tc and (tc - x) < d * (1.0 - _TIE_REL_TOL)]
    rec["n_tied"] = len(tied)
    rec["dose"] = len(tied) / n if n else 0.0
    if not tied:
        rec["reason"] = "no death in the censoring bin"
        return a, ev, rec
    rec["detected"] = True
    rec["tied_ages"] = sorted(set(a[i] for i in tied))
    if not enabled:
        rec["reason"] = "detected; relabel disabled"
        return a, ev, rec
    for i in tied:
        ev[i] = 0
    rec.update(applied=True, n_relabelled=len(tied), reason="relabelled")
    return a, ev, rec


def run_fit(observation: Observation, tie_relabel: bool = False):
    """(res, censor_tie): the estimator's fit on this observation, after the (optional) tie relabel.

    `res` is `estimator.fit`'s dict, with `diagnostics.censor_tie` added. Raises on failure (the caller --
    `calibrate`, or the cluster driver -- decides how a failure is recorded)."""
    fj = _sibling("fp_jax")
    est = _sibling("estimator")
    ages, events, tie = censor_tie_relabel(observation.ages, observation.events, observation.delta,
                                           enabled=tie_relabel)
    res = est.fit(fj, ages, events, delta=observation.delta, levels=(LEVEL,), kappa_max=KAPPA_MAX)
    res.setdefault("diagnostics", {})["censor_tie"] = tie
    return res, tie


def claim_kwargs(theta, ci, verdicts):
    """The claims as keyword dicts. ---- claim mapping: verbatim from methods/mle_profile ----"""
    point = {"eta": theta[0], "beta": theta[1], "epsilon": theta[2], "kappa": theta[3]}
    claims = []
    for p in ("eta", "beta", "epsilon", "kappa"):
        lo, hi = ci.get(f"{p}@{LEVEL}", (float("nan"), float("nan")))
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
        claims.append(kw)
    return claims


class MleProfileJax2:
    descriptor = MethodDescriptor(
        name="mle_profile_jax2",
        # 1.0.0rc1: Phase-A candidate (fix B + v1/v2 + floored dt ceiling + optional tie relabel), cluster
        # test only. NOT registered; 1.0.0 is decided after Phase A.
        version="1.0.0rc1",
        bounds=BOUNDS,
        # "censoring" is NOT declared in rc1: the tie relabel is off by default (HEAD's v0.3.3 comment
        # below still describes the unrelabelled term). Phase A's CENSOR_TIE set decides 1.0.0's default.
        # HEAD (069680f): A right-censored record does contribute log S(recorded age), value and gradient,
        # matching SRK's NumPy oracle. But under a recording resolution delta, deaths recorded in the same
        # bin as the censoring age are still scored over the whole bin [t, t+delta), although they died
        # before the censoring point. On the library's censor arm (censor at a quantile, then round down)
        # that biases eta upward: closed eta coverage 0.64 vs 0.84 on the paired pure cohorts, and
        # like-for-like only where fewer than 0.5% of the cohort are such tied deaths.
        capabilities=(),
        description=("mle_profile's estimator on a JAX Fokker-Planck likelihood (hand-written "
                     "reverse sweep, cancellation-free interval mass), exact-gradient L-BFGS-B searches, "
                     "an edge polish, a re-fit on a witnessed better point, and a floored dt ceiling."),
        contact="sr_calibration_tester / mle_profile_jax2 (candidate)",
    )

    def __init__(self, tie_relabel: bool = False):
        self.tie_relabel = bool(tie_relabel)

    def settings(self) -> dict:
        """The store key's method settings: the relabel switch, so on and off never collide."""
        return {"censor_tie_relabel": self.tie_relabel}

    def cost(self, spec) -> CostDeclaration:
        """PLACEHOLDER: HEAD's v0.3.3 declaration (full tier, 2026-09-26: median 588, max 69,410 CPU s,
        peak <= 718 MB). rc1 is expected cheaper on the 48 floor-bound cohorts (step ratio 0.21 E. coli,
        0.69 C. elegans) and ~1.1-1.9x where the re-fit fires; Phase A measures it."""
        return CostDeclaration(median_seconds=590.0, min_seconds=140.0, max_seconds=70000.0)

    def claimset(self, observation: Observation, res: dict, prov, t0: float, c0: float) -> ClaimSet:
        """`calibrate`'s second half: the fit result -> a ClaimSet (HEAD's mapping and diagnostics)."""
        from srbench.provenance import format_code_line
        theta = list(res["theta"])
        verdicts = (res.get("diagnostics") or {}).get("verdicts") or {}
        claims = [Claim(**kw) for kw in claim_kwargs(theta, res["ci"], verdicts)]
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

    def calibrate(self, observation: Observation) -> ClaimSet:
        from srbench.provenance import resolve_sut
        prov = resolve_sut()            # refuses on the stale mount; applies the numpy shim
        os.environ.setdefault("SRK_CACHE_READONLY", "1")

        t0, c0 = time.perf_counter(), time.process_time()
        try:
            res, _tie = run_fit(observation, tie_relabel=self.tie_relabel)
        except Exception as ex:  # noqa: BLE001
            return ClaimSet(cohort_id=observation.cohort_id, method_name=self.descriptor.name,
                            status=f"error:{type(ex).__name__}",
                            wall_clock_seconds=time.perf_counter() - t0,
                            cpu_seconds=time.process_time() - c0,
                            diagnostics={"exception": str(ex)[:500]},
                            code_provenance=prov.to_dict())
        return self.claimset(observation, res, prov, t0, c0)


# NOT registered: no register(MleProfileJax2()) here, deliberately (Phase A ground rules).
