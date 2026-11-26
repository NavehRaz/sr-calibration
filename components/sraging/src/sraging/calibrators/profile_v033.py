"""`profile_v033`: the v0.3.3 reference calibrator behind named objects (T032; FR-004 to FR-011).

Cohort -> sraging.compat.v033 (verbatim numerics) -> FitResult. The estimator is called exactly as the
reference adapter calls it (tests/oracle/v033/sr_calibration_tester/methods/mle_profile_jax/__init__.py
lines 107-110): ages and events as lists, delta, levels=(0.95,), kappa_max=300.0. Claims and diagnostics
come from the verbatim claim code (`compat.v033.claims.build`). Positional theta lives only inside
`compat`; it is converted here, by name, with its arity asserted (constitution VIII).
"""
from __future__ import annotations

import contextlib
import math
import time

from sraging import provenance, registry
from sraging.compat.v033 import claims as _claims
from sraging.compat.v033 import estimator, fp_jax, runtime
from sraging.labels import labels
from sraging.params import SRParams
from sraging.result import CHECK_CODES, CheckRecord, Claim, FitResult

NAME = "profile_v033"
VERSION = "0.3.3"
KAPPA_MAX = 300.0
LEVELS = (0.95,)

#: Every setting of the reference calibrator, explicit (FR-011; data-model CalibratorSpec).
SETTINGS = {
    "kappa_max": KAPPA_MAX,
    "levels": LEVELS,
    "n_profile": 11,            # recorded as v0.3.3 passes it; inert there (register D-012)
    "polish": True,
    "box": estimator.box_for(KAPPA_MAX),
    "dt_ceiling_rule": "delta/L/3 when delta/L < auto dt",
    "N": 900,
    "Gtol": 1e-11,
    "scheme": "bdf2",
    "pjrt_nproc": 1,
    "x64": True,
}

#: The reference descriptor's declared cost, quoted (not measured here): methods/mle_profile_jax/
#: __init__.py:86-98, "MEASURED, v0.3.3, FULL tier, 2026-09-26".
REFERENCE_COST_SECONDS = {"median": 590.0, "min": 140.0, "max": 70000.0}

PARAMS = ("eta", "beta", "epsilon", "kappa")


def _not_run():
    return tuple(CheckRecord(c, False, False) for c in CHECK_CODES)


def _checks(cohort, claims, diagnostics, theta_hat, floor_report=None):
    """The six truth-free checks (FR-016 to FR-018; spec 002 FR-026); they never change a number."""
    from sraging.warnings import evaluate
    return evaluate(cohort, claims, diagnostics, theta_hat=None if theta_hat is None else theta_hat.as_dict(),
                    box=SETTINGS["box"], floor_report=floor_report)


#: spec 006 (FR-025; research R6): an end within this of the walk's best node, as jax2's extents are of ll_best
KAPPA_END_NATS = 1.92
KAPPA_END_DEFINITION = "walk nodes within 1.92 nats of the walk's best"


@contextlib.contextmanager
def capture_kappa_walk():
    """Keep the kappa profile walk's nodes, which v0.3.3 drops after its repair (estimator.py: rows keep value and
    loglik only): `_repair`'s last call for kappa leaves them repaired in place, [kappa, loglik, log nuisance (eta,
    beta, epsilon)]. Changes no number: the wrapper calls the original and copies its list."""
    orig, seen, k = estimator._repair, {}, PARAMS.index("kappa")

    def _recorded(obj, theta, i, fidx, pts):
        out = orig(obj, theta, i, fidx, pts)
        if i == k:
            seen.update(fidx=list(fidx), pts=[[float(v), float(ll), None if z is None else [float(x) for x in z]]
                                              for v, ll, z in pts])
        return out

    estimator._repair = _recorded
    try:
        yield seen
    finally:
        estimator._repair = orig


def kappa_ends_v033(nodes: dict, box: dict) -> dict | None:
    """diagnostics["kappa_ends"] (data-model.md) from the captured walk: the lowest- and highest-kappa nodes within
    KAPPA_END_NATS of the best node, theta in PARAMS order, each end box-limited when it lies at the kappa box edge.
    None with no captured walk."""
    pts = [p for p in (nodes or {}).get("pts") or [] if p[2] is not None and math.isfinite(p[1])]
    if not pts:
        return None
    best = max(p[1] for p in pts)
    inside = sorted((p for p in pts if p[1] >= best - KAPPA_END_NATS), key=lambda p: p[0])
    fidx, k = nodes["fidx"], PARAMS.index("kappa")

    def theta(p):
        th = [0.0] * len(PARAMS)
        for j, zj in zip(fidx, p[2]):
            th[j] = math.exp(zj)
        th[k] = p[0]
        return th

    def at_edge(x, edge):
        return abs(x - edge) <= 1e-9 * abs(edge)
    lo, hi = inside[0], inside[-1]
    return {"definition": KAPPA_END_DEFINITION, "kappa_lo": lo[0], "kappa_hi": hi[0], "theta_lo": theta(lo),
            "theta_hi": theta(hi), "box_limited_lo": at_edge(lo[0], box["kappa"][0]),
            "box_limited_hi": at_edge(hi[0], box["kappa"][1])}


def fit(cohort) -> FitResult:
    """Fit one cohort with the v0.3.3 reference numerics."""
    if cohort.has_features:
        from sraging.errors import UnsupportedOptionError
        what = ("entry ages" if cohort.entry is not None else "a fit window" if cohort.window is not None
                else "a per-row delta")
        raise UnsupportedOptionError(what, f"{NAME} (v0.3.3) has no likelihood for {what}; the engine "
                                     f"(sraging.engine) evaluates it, and calibrators that fit it arrive in "
                                     f"specs 003 and 006")
    runtime.check()
    spec = registry.get(NAME)
    delta = cohort.delta                                          # None only for continuous
    L, kw, ceiling = estimator.solver_grid(cohort.ages, cohort.events, delta)
    grid = {"N": kw["N"], "dt": kw["dt"], "tau_max": kw["tau_max"], "Gtol": kw["Gtol"],
            "scheme": kw["scheme"], "sampling_dt": kw.get("sampling_dt"), "L_km": float(L),
            "dt_ceiling": ceiling}
    prov = provenance.collect(spec, grid, runtime.state())
    summary = {"name": cohort.name, "n": cohort.n, "n_deaths": cohort.n_deaths,
               "n_censored": cohort.n_censored, "delta": cohort.delta, "continuous": cohort.continuous}

    t0, c0 = time.perf_counter(), time.process_time()
    try:
        with capture_kappa_walk() as walk:                     # spec 006 FR-025: the kappa ends (no number moves)
            res = estimator.fit(fp_jax, list(cohort.ages), list(cohort.events), delta=delta, levels=LEVELS,
                                kappa_max=KAPPA_MAX)
    except Exception as ex:  # noqa: BLE001 -- recorded as a status, exactly as the reference does
        return FitResult(
            status=f"error:{type(ex).__name__}", cohort=summary, theta_hat=None, theta_labels={},
            claims=(), loglik=float("nan"), warnings=(), checks=_not_run(),
            diagnostics={"exception": str(ex)[:500]},
            timing={"wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0},
            provenance=prov)
    cpu = time.process_time() - c0

    claim_dicts, diag = _claims.build(res)
    diag["code"] = provenance.code_line(prov)
    ends = kappa_ends_v033(walk, estimator.box_for(KAPPA_MAX))
    if ends is not None:
        diag["kappa_ends"] = ends

    theta = list(res["theta"])
    assert len(theta) == len(PARAMS), f"estimator returned {len(theta)} parameters, expected 4"
    theta_labels = {}
    try:
        theta_hat = SRParams(**{p: float(v) for p, v in zip(PARAMS, theta)})
    except ValueError as ex:                                     # e.g. an underflowed seed on failure
        theta_hat = None
        theta_labels["_theta_hat"] = f"theta-hat not representable as SRParams: {ex}"

    bare = tuple(Claim(c["parameter"], c["shape"], c["lower"], c["upper"], c["point"], c["level"],
                       c["n_components"]) for c in claim_dicts)
    lab = labels(bare, theta_hat)
    claims = tuple(Claim(c.parameter, c.shape, c.lower, c.upper, c.point, c.level, c.n_components,
                         lab[c.parameter]) for c in bare)
    theta_labels.update(lab)
    floor_report = None
    if theta_hat is not None:                  # spec 002: per-individual terms at theta-hat, after the fit
        from sraging.floors import report as _floor_report
        floor_report = _floor_report(cohort, theta_hat)
        diag["floor_report"] = floor_report
    from sraging.leverage import check_v033                    # spec 006 FR-024: after the fit, never altering it
    from sraging.warnings import V033_LEVERAGE_CAVEAT, leverage_warnings
    diag["leverage"] = check_v033(cohort, theta_hat, diag, fit_cpu_s=cpu)
    warnings, checks = _checks(cohort, claims, diag, theta_hat, floor_report)
    warnings = tuple(warnings) + leverage_warnings(diag["leverage"], caveat=V033_LEVERAGE_CAVEAT)  # R-9 failed

    return FitResult(
        status="ok", cohort=summary, theta_hat=theta_hat, theta_labels=theta_labels, claims=claims,
        loglik=res.get("loglik"), warnings=tuple(warnings), checks=tuple(checks), diagnostics=diag,
        timing={"wall_s": res.get("wall_s"), "cpu_s": cpu, "compile_s": diag.get("compile_s"),
                "seed_s": diag.get("seed_s"), "mle_s": diag.get("mle_s")},
        provenance=prov)


registry.register(registry.CalibratorSpec(name=NAME, version=VERSION, numerics="v033",
                                          settings=SETTINGS, fit=fit))
