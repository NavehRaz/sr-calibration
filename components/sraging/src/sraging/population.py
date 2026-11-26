"""The engine switch (spec 004 FR-017, US6; research R21; contracts/python-api.md §6; data-model "Population").

    pop = population(params, ages, heterogeneity=None, engine="auto", n=None, key=None, delta=None,
                     censor_frac=0.0, window=None, dt=None, wall=None)

| engine | n = None | n = int |
|---|---|---|
| `fp` | the FP curves (S, f, hazard at `ages`); a heterogeneity tuple, and every option FP cannot use, refused by name | gen-1 inside its domain; outside it, refused naming why |
| `langevin` | refused: "pass n" | `sraging.langevin.simulate` (the wall recorded) |
| `auto` | FP where FP can; a tuple refused: "pass n" (OD-11 (a)) | gen-1 inside its domain; outside it the Langevin, with ENGINE_AUTO_LANGEVIN naming the reason (P3's boundary, C10, tasks T130) |

- **gen-1's domain** (`sraging.generator`): xc or no heterogeneity (the truncated normal), n <= HET_N_CAP when
  heterogeneous, xc = 1 (its specs carry no threshold; the tester's conversion at xc = 1: rho_eta = eta L^2, rho_beta
  = beta L, rho_eps = epsilon L, L the model's own), no step and no wall, and a valid CohortSpec (a recording delta).
- **The FP curves with heterogeneity at M = "auto"** take their node count from R26's check for an exact population:
  the ladder's first rung M with sup over `ages` of |S_M - S_next| <= curve_tol (1e-4, recorded); unconverged at the
  cap reads so in the provenance and the reason. A fixed M is used as given.
- **A sample's S** is the Kaplan-Meier estimate at `ages` from its observed (ages, events), with 95 % bands from
  Greenwood's variance; f and hazard are None for a sample.
- **Under `auto`** an option the chosen engine does not use (dt, wall for FP or gen-1) is recorded in the provenance's
  `unused` with the engine that ignored it; under `fp` it is refused by name.
- **`auto`'s n = int boundary (P3_RULE)** is pilot P3's committed result (tests/acceptance/results/p3_cost_004.json
  #boundary, f16715c): gen-1's domain, not the cost. Inside it gen-1, whose law is exact up to its stated grid error,
  even where the Langevin is cheaper (below n ~ 11,000 on P3's human preset); outside it the Langevin (Dg).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sraging.engine.inputs import Heterogeneity, refuse_zero_spread
from sraging.errors import UnsupportedOptionError

ENGINES = ("auto", "fp", "langevin")
BAND_Z = 1.959963984540054              # the bands' 95 %
CURVE_TOL = 1e-4                        # R26: an exact population's node count, sup |S_M - S_next| <= CURVE_TOL
#: auto's n = int boundary (C10, tasks T130), encoded from pilot P3's committed result: its rule, verbatim, and where
#: it is recorded; auto_engine applies it
P3_RULE = {"rule": "gen-1 inside its domain (xc or no heterogeneity, n <= HET_N_CAP, a valid CohortSpec); the Langevin "
                   "outside it (Dg), with ENGINE_AUTO_LANGEVIN naming the reason",
           "inside": "gen-1", "outside": "langevin",
           "source": "tests/acceptance/results/p3_cost_004.json#boundary (pilot P3, T113, f16715c)"}


@dataclass(frozen=True, eq=False)
class Population:
    kind: str                           # "exact" (FP) or "sample" (gen-1 or the Langevin)
    S: np.ndarray
    f: np.ndarray | None
    hazard: np.ndarray | None
    sample: object                      # LangevinSample, gen-1's GeneratedCohort, or None
    bands: dict | None
    engine_used: str
    reason: str
    warnings: tuple = ()
    provenance: dict = field(default_factory=dict)


def _members(het) -> tuple:
    if het is None:
        return ()
    if isinstance(het, (tuple, list)):
        return tuple(het)
    return (het,)


def _definition(het) -> str | None:
    m = _members(het)
    return m[0].definition if m else None


def gen1_spec(params, heterogeneity, n, key, delta, censor_frac=0.0, window=None) -> tuple:
    """(gen-1's CohortSpec, None) inside its domain, else (None, why)."""
    from sraging import generator as GEN
    from sraging.engine.curves import model_L
    m = _members(heterogeneity)
    if len(m) > 1:
        return None, "more than one heterogeneous parameter: gen-1 draws xc heterogeneity only"
    if m and (m[0].parameter != "xc" or m[0].definition != "truncated_normal"):
        return None, (f"gen-1 draws xc heterogeneity (the truncated normal) only, not {m[0].parameter} "
                      f"({m[0].definition})")
    if m and n > GEN.HET_N_CAP:
        return None, f"n = {n:,} > gen-1's HET_N_CAP = {GEN.HET_N_CAP:,} for a heterogeneous cohort"
    if float(params.xc) != 1.0:
        return None, f"gen-1's specs are at xc = 1 (xc = {params.xc:g})"
    if key is None:
        return None, "gen-1 needs key (its seed)"
    L = float(model_L(params))
    kw = dict(name="population", rho_eta=params.eta * L * L, rho_beta=params.beta * L, rho_eps=params.epsilon * L,
              kappa=params.kappa, n=int(n), L_phys=L, seed=int(key), delta=delta, censor_frac=float(censor_frac),
              time_range=None if window is None else tuple(window))
    if m:
        kw.update(het_param="xc", het_var=float(m[0].sd))
    if params.h_ext > 0:
        kw["external_hazard"] = -math.log(float(params.h_ext))
    try:
        return GEN.CohortSpec(**kw), None
    except UnsupportedOptionError as ex:
        return None, f"not a valid gen-1 CohortSpec: {ex}"


def auto_engine(params, heterogeneity, n, key, delta, censor_frac=0.0, window=None) -> tuple:
    """(engine, reason) `auto` takes for n = int by P3_RULE: ("gen-1", None) inside gen-1's domain (gen1_spec), else
    ("langevin", why it is outside). Draws nothing."""
    spec, why = gen1_spec(params, heterogeneity, n, key, delta, censor_frac, window)
    return (P3_RULE["inside"], None) if spec is not None else (P3_RULE["outside"], why)


def kaplan_meier(t_obs, events, at) -> dict:
    """S at `at` from (t_obs, events), with Greenwood's 95 % bands (clipped to [0, 1])."""
    t_obs, events, at = np.asarray(t_obs, float), np.asarray(events, int), np.asarray(at, float)
    times = np.unique(t_obs[events == 1])
    at_risk = np.array([np.sum(t_obs >= u) for u in times], float)
    d = np.array([np.sum((t_obs == u) & (events == 1)) for u in times], float)
    S_t = np.cumprod(1.0 - d / at_risk)
    with np.errstate(divide="ignore", invalid="ignore"):
        g_t = np.cumsum(np.where(at_risk > d, d / (at_risk * (at_risk - d)), np.inf))
    j = np.searchsorted(times, at, side="right") - 1
    S = np.where(j >= 0, S_t[np.maximum(j, 0)], 1.0)
    g = np.where(j >= 0, g_t[np.maximum(j, 0)], 0.0)
    var = np.where(S > 0, S * S * np.where(np.isfinite(g), g, 0.0), 0.0)        # S = 0 once all at risk died
    half = BAND_Z * np.sqrt(np.where(np.isfinite(var), var, 0.0))
    return {"S": S, "S_lo": np.clip(S - half, 0.0, 1.0), "S_hi": np.clip(S + half, 0.0, 1.0), "level": 0.95,
            "method": "Kaplan-Meier with Greenwood's variance", "n": int(t_obs.size)}


def _refuse_unused(engine_name, why, **opts):
    for k, (v, unused) in opts.items():
        if unused(v):
            raise UnsupportedOptionError(k, f"{k}={v!r}: {why} ({engine_name} does not use it)")


def population(params, ages, *, heterogeneity=None, engine="auto", n=None, key=None, delta=None, censor_frac=0.0,
               window=None, dt=None, wall=None) -> Population:
    """contracts/python-api.md §6."""
    from sraging.params import SRParams
    if engine not in ENGINES:
        raise UnsupportedOptionError("engine", f"engine is one of {ENGINES}, got {engine!r}")
    if not isinstance(params, SRParams):
        raise UnsupportedOptionError("params", "params is an SRParams")
    if n is not None and (isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1):
        raise UnsupportedOptionError("n", f"n is None (FP curves) or an integer >= 1, got {n!r}")
    for h in _members(heterogeneity):
        if not isinstance(h, Heterogeneity):
            raise UnsupportedOptionError("heterogeneity", "pass a sraging.engine.Heterogeneity or a tuple of them")
    tuple_ = isinstance(heterogeneity, (tuple, list))
    if not tuple_:
        refuse_zero_spread(heterogeneity, "population")
    if engine == "fp":
        if wall is not None:
            raise UnsupportedOptionError("wall", f"wall={wall!r}: FP has no wall (the wall is the Langevin's)")
        if tuple_:
            raise UnsupportedOptionError("heterogeneity", "FP takes one heterogeneous parameter: a tuple is the "
                                         "Langevin's (engine=\"langevin\", or \"auto\" with n; OD-11 (a))")
        if dt is not None:
            raise UnsupportedOptionError("dt", f"dt={dt!r}: FP and gen-1 have no Langevin step")
        return _fp(params, ages, heterogeneity, key, delta, censor_frac, window, "fp") if n is None else \
            _gen1(params, ages, heterogeneity, n, key, delta, censor_frac, window, "fp", unused={})
    if engine == "langevin":
        if n is None:
            raise UnsupportedOptionError("n", "the Langevin draws a sample: pass n")
        return _langevin(params, ages, heterogeneity, n, key, delta, censor_frac, window, dt, wall, "langevin",
                         reason="engine=\"langevin\" requested", warn=None)
    # auto
    if n is None:
        if tuple_:
            raise UnsupportedOptionError("n", "a heterogeneity tuple has no FP curves (OD-11 (a)): pass n to draw it "
                                         "with the Langevin")
        unused = {k: v for k, v in (("dt", dt), ("wall", wall)) if v is not None}
        if unused:
            raise UnsupportedOptionError(sorted(unused)[0], f"{sorted(unused)[0]}={unused[sorted(unused)[0]]!r}: auto "
                                         f"with n = None gives the FP curves, which have no step and no wall; pass n "
                                         f"with engine=\"langevin\"")
        return _fp(params, ages, heterogeneity, key, delta, censor_frac, window, "auto")
    spec, why = gen1_spec(params, heterogeneity, n, key, delta, censor_frac, window)
    if spec is not None:
        return _gen1(params, ages, heterogeneity, n, key, delta, censor_frac, window, "auto",
                     unused={k: v for k, v in (("dt", dt), ("wall", wall)) if v is not None}, spec=spec)
    from sraging.warnings import sampler_warning
    reason = f"n = {n:,} is outside gen-1's domain ({why}): the Langevin, by P3's boundary ({P3_RULE['source']})"
    return _langevin(params, ages, heterogeneity, n, key, delta, censor_frac, window, dt, wall, "auto", reason=reason,
                     warn=sampler_warning("ENGINE_AUTO_LANGEVIN", reason=why))


def fp_curves(params, ages, heterogeneity, curve_tol=CURVE_TOL) -> tuple:
    """(Curves, the node-count record): homogeneous, a fixed M as given, or M = "auto" by R26's check on S at `ages`."""
    from sraging.engine import mixture as MIX
    from sraging.engine.curves import curves
    het = heterogeneity
    if het is None:
        return curves(params, ages), {"verdict": "homogeneous", "M": None}
    if het.M != "auto":
        return curves(params, ages, heterogeneity=het), {"verdict": "fixed", "M": het.M, "check": het.check}
    prev, ladder = None, []
    for M in MIX.RUNGS:
        c = curves(params, ages, heterogeneity=Heterogeneity(parameter=het.parameter, sd=het.sd, M=M, check=False))
        if prev is not None:
            d = float(np.max(np.abs(np.asarray(c.S) - np.asarray(prev[1].S))))
            ladder.append({"M": prev[0], "M_next": M, "sup_dS": d})
            if d <= curve_tol:
                return prev[1], {"verdict": "converged", "M": prev[0], "tol": curve_tol, "sup_dS": d,
                                 "ladder": ladder}
        prev = (M, c)
    return prev[1], {"verdict": "unconverged_at_cap", "M": MIX.CAP, "tol": curve_tol, "sup_dS": ladder[-1]["sup_dS"],
                     "ladder": ladder}


def _fp(params, ages, heterogeneity, key, delta, censor_frac, window, requested) -> Population:
    _refuse_unused("FP", "the FP curves are exact (n = None)", key=(key, lambda v: v is not None),
                   delta=(delta, lambda v: v is not None), censor_frac=(censor_frac, lambda v: float(v) != 0.0),
                   window=(window, lambda v: v is not None))
    c, check = fp_curves(params, ages, heterogeneity)
    reason = "n = None: the FP curves" + ("" if check["verdict"] != "unconverged_at_cap" else
                                          f" (node count unconverged at the cap: sup|dS| = {check['sup_dS']:.3g} > "
                                          f"{check['tol']:g})")
    return Population(kind="exact", S=np.asarray(c.S), f=np.asarray(c.f), hazard=np.asarray(c.hazard), sample=None,
                      bands=None, engine_used="fp", reason=reason,
                      provenance={"engine": "fp", "requested_engine": requested,
                                  "definition": _definition(heterogeneity), "step": c.grid.summary()
                                  if hasattr(c.grid, "summary") else dict(c.grid.__dict__), "key": None,
                                  "node_count": check, "reason": reason})


def _gen1(params, ages, heterogeneity, n, key, delta, censor_frac, window, requested, *, unused, spec=None):
    from sraging import generator as GEN
    if spec is None:
        spec, why = gen1_spec(params, heterogeneity, n, key, delta, censor_frac, window)
        if spec is None:
            raise UnsupportedOptionError("engine", f"engine=\"fp\" with n = {n:,} draws with gen-1, and this is "
                                         f"outside its domain: {why}")
    co = GEN.materialize(spec)
    km = kaplan_meier(co.death_times, co.events, ages)
    reason = "n = int inside gen-1's domain: gen-1 draws from the FP law"
    prov = {"engine": "gen-1", "requested_engine": requested, "definition": _definition(heterogeneity),
            "step": {"grid": dict(co.provenance["grid"]), "delta": spec.delta, "L_phys": spec.L_phys}, "key": key,
            "reason": reason, "caveats": list(co.caveats), "generator_version": GEN.GENERATOR_VERSION,
            "unused": {k: {"value": v, "ignored_by": "gen-1"} for k, v in unused.items()}}
    return Population(kind="sample", S=km["S"], f=None, hazard=None, sample=co,
                      bands={k: km[k] for k in ("S_lo", "S_hi", "level", "method", "n")}, engine_used="gen-1",
                      reason=reason, provenance=prov)


def _langevin(params, ages, heterogeneity, n, key, delta, censor_frac, window, dt, wall, requested, *, reason, warn):
    from sraging.langevin import simulate
    if key is None:
        raise UnsupportedOptionError("key", "the Langevin needs key (per-individual streams from it)")
    if dt is None:
        raise UnsupportedOptionError("dt", "the Langevin needs dt, its one step (sraging.langevin.grid_dt offers one "
                                     "dividing record ages)")
    kw = {} if wall is None else {"wall": wall}
    s = simulate(params, n=int(n), key=key, dt=dt, heterogeneity=heterogeneity, delta=delta, censor_frac=censor_frac,
                 window=window, **kw)
    km = kaplan_meier(s.ages, s.events, ages)
    warns = tuple(s.warnings) + ((warn,) if warn is not None else ())
    prov = {"engine": "langevin", "requested_engine": requested, "definition": _definition(heterogeneity),
            "step": s.provenance["dt"], "key": s.provenance["key"], "wall": s.provenance["wall"], "reason": reason}
    return Population(kind="sample", S=km["S"], f=None, hazard=None, sample=s,
                      bands={k: km[k] for k in ("S_lo", "S_hi", "level", "method", "n")}, engine_used="langevin",
                      reason=reason, warnings=warns, provenance=prov)
