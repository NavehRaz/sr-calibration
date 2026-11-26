"""`profile_jax2` and `profile_jax2_h`, at 1.0.0 (spec 003) and 1.0.1 (spec 003b): the tester's successor calibrator
behind named objects (spec 003, T015; FR-001 to FR-009; research R1, R5, R6, R7; spec 003b research R4, R5, R10).

Two frozen ports live side by side, `sraging.compat.jax2` (1.0.0) and `sraging.compat.jax2_101` (1.0.1). A `JaxPort`
names one of them; every function below that reads the port takes `port=` and defaults to 1.0.0, so existing
callers keep 1.0.0's behaviour; the 1.0.1 calibrator passes its own. Both versions are registered; a bare name means
1.0.1 (Naveh, 2026-10-19), and 1.0.0 is selected by its exact version.

Cohort -> sraging.compat.jax2 (verbatim numerics) -> FitResult /4. The fit is the package's own `run_fit`, called
with each name's class constants exactly as `calibrate` passes them (compat.jax2.settings; the oracle's
__init__.py:678-681), on an observation with the tester's field types (tuples of floats and ints, delta, window).
Its claims are the package's `final_claim_kwargs` (v0.3.3's mapping, then the self-consistency widening).
Positional theta lives only inside `compat`; it is converted here, by name, with its arity asserted.

The fit's archive (every point it evaluated) is local to the estimator; `run_port(..., capture=True)` records each
`_Objective` the fit creates by temporarily subclassing it -- no file is edited -- and reads the archive with the
port's own `_sc_archive` (research R7; tests/unit/test_profile_jax2.py checks that the capture changes no byte).
"""
from __future__ import annotations

import contextlib
import dataclasses
import math
import os
import time
from dataclasses import dataclass

import numpy as np

from sraging import provenance, registry
from sraging.compat.jax2 import claims as _jc
from sraging.compat.jax2 import estimator as _je
from sraging.compat.jax2 import fp_jax as _jf
from sraging.compat.jax2 import observation_model as _jom
from sraging.compat.jax2 import settings as _js
from sraging.compat.jax2_101 import claims as _jc1
from sraging.compat.jax2_101 import estimator as _je1
from sraging.compat.jax2_101 import fp_jax as _jf1
from sraging.compat.jax2_101 import observation_model as _jom1
from sraging.compat.jax2_101 import settings as _js1
from sraging.compat.v033 import runtime
from sraging.errors import UnsupportedOptionError
from sraging.labels import labels
from sraging.params import SRParams
from sraging.result import CHECK_CODES, CheckRecord, Claim, FitResult



@dataclass(frozen=True)
class JaxPort:
    """One frozen jax2 port (spec 003b research R5): its version, its compat modules, the registered names' classes
    and (1.0.1) the kappa-fixed classes it ships but sraging does not register."""

    version: str
    claims: object
    estimator: object
    settings: object
    fp_jax: object
    observation_model: object
    classes: dict
    kfix_classes: dict


PORTS = {
    "1.0.0": JaxPort("1.0.0", _jc, _je, _js, _jf, _jom,
                     {"profile_jax2": _js.MleProfileJax2, "profile_jax2_h": _js.MleProfileJax2H}, {}),
    "1.0.1": JaxPort("1.0.1", _jc1, _je1, _js1, _jf1, _jom1,
                     {"profile_jax2": _js1.MleProfileJax2, "profile_jax2_h": _js1.MleProfileJax2H},
                     {c.NAME: c for c in (_js1.MleProfileJax2KfixK0p001, _js1.MleProfileJax2KfixK0p5,
                                          _js1.MleProfileJax2KfixK1, _js1.MleProfileJax2HKfixK0p001,
                                          _js1.MleProfileJax2HKfixK0p5, _js1.MleProfileJax2HKfixK1)}),
}
for _v, _p in PORTS.items():
    assert _p.claims.VERSIONS["mle_profile_jax2"] == _p.claims.VERSIONS["mle_profile_jax2_h"] == _v, _v


class MleProfileJax2W(_js1.MleProfileJax2):
    """W (spec 006 FR-002; research R1): jax2 1.0.1's J with the window on and the hazard off. sraging's class, not a
    jax2 one: compat/jax2_101/settings.py stays verbatim, and no jax2 release ran this combination."""
    NAME = "sraging_profile_jax2_w"
    WINDOW = True


#: W's port: 1.0.1's modules under W's own version, which never reads as a jax2 release; kept out of PORTS, whose
#: versions are jax2's (the assert above)
PORT_W = dataclasses.replace(PORTS["1.0.1"], version="1.0.1+w1", classes={"profile_jax2_w": MleProfileJax2W},
                             kfix_classes={})
W_COMPOSITION = "jax2 1.0.1 (6884fc8), hazard off, window on; a sraging composition, never run by jax2"
#: provenance.calibrator.composition, for a name that is sraging's composition of a jax2 release
COMPOSITIONS = {"profile_jax2_w": W_COMPOSITION}
#: spec 006 FR-009, FR-011: the names (at 1.0.1) that fit a window through `window_path` -- W, and H as jax2 shipped it
#: (its class has the window on); any other refuses one, naming WINDOW_NAMES
WINDOW_FITTERS = ("profile_jax2_w", "profile_jax2_h")
WINDOW_NAMES = ("profile_jax2_w", "profile_jax2_h")


def fits_window(name: str, port) -> bool:
    return name in WINDOW_FITTERS and port.version != "1.0.0"
V100 = PORTS["1.0.0"]
CURRENT_VERSION = "1.0.1"                  # what a bare name means (Naveh, 2026-10-19)
CLASSES = V100.classes                     # 1.0.0's, as before spec 003b (callers that name no port)
VERSION = V100.version
PARAMS = ("eta", "beta", "epsilon", "kappa")
NO_EXTERNAL_MORTALITY = ("profile_jax2's claims assume no external mortality: on the tester's hazard arm v0.3.3's "
                         "closed eta claims covered the truth 8 times in 261 (PREREG_C:104); use profile_jax2_h "
                         "for a cohort with external deaths")


#: Leaves the tester's identity comparison treats as timing, at any depth of a result, its claims or its widening
#: record (reports/rc2/C/freeze_1.0.0/scripts/f1_identity_compare.py:5-6, 15). Never compared.
TIMING_LEAVES = frozenset({"wall_s", "compile_s", "seed_s", "mle_s", "cpu_s", "refit_cpu_s", "fit_wall_s"})


def strip_timing(x, skip=()):
    """A copy of `x` (dicts, lists, scalars) without any TIMING_LEAVES key, and without the keys in `skip`."""
    if isinstance(x, dict):
        return {k: strip_timing(v, skip) for k, v in x.items() if k not in TIMING_LEAVES and k not in skip}
    if isinstance(x, (list, tuple)):
        return [strip_timing(v, skip) for v in x]
    return x


@dataclass(frozen=True)
class Observation:
    """The fields `run_fit` reads, with the tester's types (srbench/contract/types.py:168-186)."""

    cohort_id: str
    ages: tuple
    events: tuple
    delta: float | None = None
    window: tuple | None = None


def run_kwargs(cls) -> dict:
    """`run_fit`'s keywords for a class, exactly as its version's `calibrate` passes them: 1.0.0's ten
    (__init__.py:678-681); 1.0.1 adds inner_polish_j, bracket_bisect, extent_z and box_flag, and kappa_fixed / combos
    only when set (handoff section 3.1; jax2's SWITCHES.tsv "run_fit (captured)" row, checked by a unit test)."""
    kw = dict(relabel=cls.RELABEL, hazard=cls.HAZARD, window=cls.WINDOW, shadow=cls.SHADOW,
              exact_from_F=cls.EXACT_F, self_consistent=cls.SELF_CONSISTENT, floor_early=cls.FLOOR_EARLY,
              n_rule=cls.N_RULE, dt_rule=cls.DT_RULE, early_all=cls.EARLY_ALL)
    if hasattr(cls, "INNER_POLISH_J"):                       # a 1.0.1 class
        kw.update(inner_polish_j=cls.INNER_POLISH_J, bracket_bisect=cls.BRACKET_BISECT, extent_z=cls.EXTENT_Z,
                  box_flag=cls.BOX_FLAG)
        if cls.KAPPA0 is not None:
            kw["kappa_fixed"] = cls.KAPPA0
        if cls.COMBOS:
            kw["combos"] = cls.COMBOS
    return kw


def settings(name: str, port: JaxPort = V100) -> dict:
    cls = port.classes[name]
    jc, je = port.claims, port.estimator
    return {**run_kwargs(cls), "kappa_max": jc.KAPPA_MAX, "levels": (jc.LEVEL,), "box": je.box_for(jc.KAPPA_MAX),
            "sc_edge_dex": jc.SC_EDGE_DEX, "srb_strict": True, "pjrt_nproc": 1, "x64": True}


def estimate_for(port: JaxPort, cls, res):
    """`diagnostics.estimate` as the port's adapter builds it (1.0.1's `claimset`: estimate_block(res, HAZARD,
    KAPPA0); spec 003b research R10), or None for a port without it (1.0.0)."""
    eb = getattr(port.claims, "estimate_block", None)
    return None if eb is None else eb(res, cls.HAZARD, getattr(cls, "KAPPA0", None))


def switches_for(cls):
    """`diagnostics.switches` as 1.0.1's `claimset` records it: the class constants, 1.0.0's 10 keys plus
    inner_polish_j, bracket_bisect, extent_z and box_flag (handoff 7.2 L4; kappa_fixed / combos belong to the
    unregistered kfix names). None for a 1.0.0 class: 1.0.0's FitResult is unchanged (spec 003)."""
    if not hasattr(cls, "INNER_POLISH_J"):
        return None
    return dict(hazard=bool(cls.HAZARD), window=bool(cls.WINDOW), relabel=bool(cls.RELABEL), shadow=bool(cls.SHADOW),
                exact_from_F=bool(cls.EXACT_F), self_consistent=bool(cls.SELF_CONSISTENT),
                floor_early=bool(cls.FLOOR_EARLY), n_rule=bool(cls.N_RULE), dt_rule=bool(cls.DT_RULE),
                early_all=bool(cls.EARLY_ALL), inner_polish_j=bool(cls.INNER_POLISH_J),
                bracket_bisect=bool(cls.BRACKET_BISECT), extent_z=bool(cls.EXTENT_Z), box_flag=bool(cls.BOX_FLAG),
                **({"kappa_fixed": float(cls.KAPPA0)} if cls.KAPPA0 is not None else {}),
                **({"combos": list(cls.COMBOS)} if cls.COMBOS else {}))


def version_warnings(port: JaxPort, cls) -> tuple:
    """The warnings a name raises at its version on every fit: H_NOT_USABLE on profile_jax2_h 1.0.0 only (spec 003b
    FR-011)."""
    from sraging.warnings import h_not_usable
    w = h_not_usable(port.version) if cls.HAZARD else None
    return () if w is None else (w,)


def notes_for(port: JaxPort, cls) -> list:
    """The notes a name carries at its version (diagnostics["notes"]; sraging's own): HAZARD_SHARP_LIMIT on
    profile_jax2_h 1.0.1 (spec 003b FR-011, research R12)."""
    import copy
    from sraging.warnings import HAZARD_SHARP_LIMIT
    return [copy.deepcopy(HAZARD_SHARP_LIMIT)] if cls.HAZARD and port.version != "1.0.0" else []


SHARP_SR_CV = 0.05          # the register's HZ-1 rule: a hazard cohort whose converged SR CV is below this


def sr_cv(theta_hat) -> float:
    """The SR model's lifetime CV at theta-hat, without the hazard: the converged curve of sraging-gen-1's grid (the
    grid library v6 was built on and the register's HZ-1 rule reads), in the model's own time scale."""
    import dataclasses
    from sraging import generator as GEN
    from sraging.engine.curves import model_L
    p = dataclasses.replace(theta_hat, h_ext=0.0)
    d = p.dimensionless(model_L(p))
    point = {"rho_eta": d["rho_eta"], "rho_beta": d["rho_beta"], "rho_eps": d["rho_epsilon"], "kappa": d["kappa"]}
    return float(GEN.curve_moments(GEN._curve(point, GEN.GRID))["cv"])


def h_stratum_note(cv) -> dict:
    """Spec 003g FR-013: whether this `_h` 1.0.1 fit sits in the stratum HAZARD_SHARP_LIMIT is about (the register's
    HZ-1: a hazard cohort whose SR CV is below 0.05), read from the fit's own SR CV at theta-hat."""
    if cv is None:
        msg = "the sharp-hazard stratum: this fit's SR CV could not be computed, so whether the limit applies is unknown"
    elif cv < SHARP_SR_CV:
        msg = (f"this cohort is in the sharp-hazard stratum (SR CV {cv:.3g} < {SHARP_SR_CV:g} at theta-hat), where "
               f"profile_jax2_h 1.0.1's grid is measured off (HAZARD_SHARP_LIMIT)")
    else:
        msg = (f"this cohort is not in the sharp-hazard stratum (SR CV {cv:.3g} >= {SHARP_SR_CV:g} at theta-hat); "
               f"HAZARD_SHARP_LIMIT is about that stratum")
    return {"code": "HAZARD_SHARP_STRATUM", "message": msg, "data": {"sr_cv": cv, "cut": SHARP_SR_CV}}


#: spec 003 measured combination-claim coverage on 1.0.0 only; the censored arm's eta/eps covered 0.818
COMBINATIONS_1_0_0_CENSOR_ETA_EPS = 0.818


def combination_limit_note(version: str, combos: dict | None) -> dict | None:
    """Spec 003g FR-012 (Naveh Db): a 1.0.1 fit that gives combination claims says they were not truth-scored on
    1.0.1. None on 1.0.0 (scored, spec 003) and when no combination claim was given."""
    if version == "1.0.0" or not (combos or {}).get("on"):
        return None
    return {"code": "COMBINATIONS_NOT_TRUTH_SCORED",
            "message": f"combination claims are not truth-scored on {version}: they were scored on 1.0.0 only "
                       f"(spec 003), where the censored arm's eta/eps covered {COMBINATIONS_1_0_0_CENSOR_ETA_EPS}; "
                       f"read them as the 1.0.0 claims' method applied to this fit",
            "data": {"version": version, "scored_on": "1.0.0",
                     "censor_eta_over_eps_coverage_1_0_0": COMBINATIONS_1_0_0_CENSOR_ETA_EPS}}


def sharp_limit_note(version: str, cls, diag: dict) -> dict | None:
    """Spec 003g N2 (Naveh 2026-10-29): a 1.0.1 J fit where DT_RULE fires (CV_c < 0.05) carries the limits measured in
    that regime (warnings.SHARP_COHORT_LIMIT). None on 1.0.0, on the `_h` names and outside the regime."""
    import copy
    from sraging.warnings import SHARP_COHORT_LIMIT
    rule = ((diag or {}).get("dt_floor") or {}).get("grid_rule") or {}
    if version == "1.0.0" or cls.HAZARD or not rule.get("dt_in_scope"):
        return None
    return copy.deepcopy(SHARP_COHORT_LIMIT)


KAPPA_END_DEFINITION = "archive points within 1.92 nats of ll_best"


def kappa_ends_jax2(diag: dict, box: dict) -> dict | None:
    """Spec 006 (FR-025; research R6): diagnostics["kappa_ends"] from the fit's own self-consistency extents of kappa
    (every jax2 record keeps them), each end box-limited when it lies at the kappa box edge (BOX_FLAG is off in 1.0.1,
    so sraging labels it). None when the fit kept no extents."""
    e = (((diag or {}).get("self_consistency") or {}).get("extents") or {}).get("kappa")
    if not e or e.get("theta_qmin") is None or e.get("theta_qmax") is None:
        return None

    def at_edge(x, edge):
        return abs(float(x) - edge) <= 1e-9 * abs(edge)
    return {"definition": KAPPA_END_DEFINITION, "kappa_lo": e["qmin"], "kappa_hi": e["qmax"],
            "theta_lo": list(e["theta_qmin"]), "theta_hi": list(e["theta_qmax"]),
            "box_limited_lo": at_edge(e["qmin"], box["kappa"][0]), "box_limited_hi": at_edge(e["qmax"], box["kappa"][1])}


def method_diagnostics(port: JaxPort, cls, res: dict, claim_kw, sc_block) -> dict:
    """The method's diagnostics of one fit, statement for statement as its version's adapter `claimset` assembles
    them, without `code` (sraging writes its own provenance line): run_fit's diagnostics, the self-consistency block,
    loglik and dt_ceiling. 1.0.1 adds `box_limited` under BOX_FLAG (dropping box_points), keeps `profiles` only under
    KEEP_PROFILES, and records `switches` and `estimate` (spec 003b FR-005: the FitResult carries the ClaimSet's
    fields). 1.0.0's assembly is unchanged. The kappa-fixed classes are refused (not registered: FR-007)."""
    diag = dict(res.get("diagnostics") or {})
    if sc_block is not None:
        diag["self_consistency"] = sc_block
    if switches_for(cls) is None:                         # 1.0.0
        diag.pop("profiles", None)
        diag["loglik"] = res.get("loglik")
        diag["dt_ceiling"] = res.get("dt_ceiling")
        return diag
    if cls.KAPPA0 is not None:
        raise ValueError(f"{cls.NAME}: a kappa-fixed class is not registered in sraging (spec 003b FR-007)")
    if cls.BOX_FLAG:
        bl = port.claims.box_limited_block(res, claim_kw, sc_block)
        if bl is not None:
            diag["box_limited"] = bl
            diag.pop("box_points", None)
    if not cls.KEEP_PROFILES:
        diag.pop("profiles", None)
    diag["loglik"] = res.get("loglik")
    diag["dt_ceiling"] = res.get("dt_ceiling")
    diag["switches"] = switches_for(cls)
    diag["estimate"] = estimate_for(port, cls, res)
    return diag


@contextlib.contextmanager
def capture_objectives(port: JaxPort = V100):
    """Record every `_Objective` the estimator creates while the block runs (research R7). Changes no number:
    the subclass only appends the instance to a list after the original constructor has run."""
    _je = port.estimator
    orig = _je._Objective
    seen = []

    class _Recorded(orig):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            seen.append(self)

    _je._Objective = _Recorded
    try:
        yield seen
    finally:
        _je._Objective = orig


@contextlib.contextmanager
def capture_nodes(port: JaxPort = V100):
    """Record each profile's final nodes [(value, ll, full z)] by parameter index (`_repair` is wrapped, its output
    returned unchanged; the tester's r3 scripts read nodes the same way). A node's stored z holds the free coordinates
    only (`fidx`, estimator.py:1636); the full z is rebuilt as the port's `_full_points` does (z[i] = log value).
    jax2 re-runs `_repair` after any polish improvement (estimator.py:1651), so the last capture is final."""
    _je = port.estimator
    orig = _je._repair
    nodes = {}

    def _recorded(obj, theta, i, fidx, pts):
        out = orig(obj, theta, i, fidx, pts)
        rows = []
        for p in pts:
            z = None
            if p[2] is not None:
                z = [0.0] * obj.NP
                z[int(i)] = math.log(float(p[0]))
                for k, zz in zip(fidx, p[2]):
                    z[int(k)] = float(zz)
            rows.append((float(p[0]), float(p[1]), z))
        nodes[int(i)] = rows
        return out

    _je._repair = _recorded
    try:
        yield nodes
    finally:
        _je._repair = orig


def fit_archive(objs, port: JaxPort = V100):
    """(Z, LL): every point the given objectives evaluated (jax2 `_sc_archive` of each, concatenated)."""
    parts = [port.estimator._sc_archive(o) for o in objs]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def main_objective(out, name, port: JaxPort = V100):
    """The objective the fit's claims were profiled on (the hazard one for profile_jax2_h), or None if not captured."""
    np_ = 5 if port.classes[name].HAZARD else 4
    objs = [o for o in (out.get("objectives") or []) if o.NP == np_]
    return objs[0] if objs else None


def observation(cohort) -> Observation:
    """The tester-typed observation of `cohort`, its window included: a windowed cohort reaches here only through
    `window_path` (spec 006), which has applied the tester's rule to its rows."""
    return Observation(cohort_id=cohort.name or "cohort", ages=tuple(float(a) for a in cohort.ages),
                       events=tuple(int(e) for e in cohort.events), delta=cohort.delta, window=cohort.window)


#: spec 006 R4: hi = inf maps to the first grid point past every record, floor(max_age/delta + tol) + 1 steps (the
#: tolerance is Cohort's grid tolerance, so an on-grid last age never maps onto itself)
HI_INF_RULE = "(floor(max_age / delta + 1e-9) + 1) * delta"
_HI_GRID_TOL = 1e-9


def window_path(cohort, cls, port: JaxPort = V100):
    """Spec 006 (FR-006 to FR-008; research R4): the window W and H fit, as (windowed cohort, record, row map). With
    no window: (cohort, None, None). In order:
      1. hi = inf with no delta is refused by name (the frozen snap cannot place it);
      2. hi = inf with delta is mapped to HI_INF_RULE, on the grid and past every record, and recorded;
      3. the tester's `apply_window` (verbatim, compat/tester_f94ec2b): ages below lo dropped, ages above hi censored
         at hi;
      4. fewer deaths left to fit than the class's free parameters (4, or 5 with h) is refused, naming the count --
         the deaths the frozen `prepare` leaves, its edge tie censored;
      5. the windowed Cohort of the kept rows, window (lo, hi used), and the row map (the kept rows' 0-based indices
         in `cohort`), so the fit, its floors and its leverage check all see the same rows;
      6. the frozen `prepare(window_on=True)`'s snap record: window_eff and the edge tie."""
    if cohort.window is None:
        return cohort, None, None
    from sraging.cohort import Cohort
    from sraging.compat.tester_f94ec2b.observation import apply_window
    lo, hi = float(cohort.window[0]), float(cohort.window[1])
    delta = cohort.delta
    mapped = None
    if not math.isfinite(hi):
        if delta is None:
            raise UnsupportedOptionError("window", "a window with hi = inf needs a recording resolution delta: the "
                                         "frozen snap places hi_eff on the delta grid; give a finite hi")
        mapped = "inf"
        hi = (math.floor(float(np.max(cohort.ages)) / float(delta) + _HI_GRID_TOL) + 1) * float(delta)
    ages, events = apply_window(cohort.ages, cohort.events, (lo, hi))
    rows = np.flatnonzero(np.asarray(cohort.ages, float) >= lo)
    need = len(PARAMS) + (1 if cls.HAZARD else 0)
    prep = port.observation_model.prepare(list(ages), list(events), delta, (lo, hi), window_on=True,
                                          relabel_on=cls.RELABEL)
    snap = prep.record["window_snap"] or {}
    n_fit = int(prep.record["n_events_out"])
    if n_fit < need:
        raise UnsupportedOptionError("window", f"the window [{lo!r}, {hi!r}] leaves {n_fit} death(s) to fit, fewer "
                                     f"than the {need} free parameters of {cls.NAME}")
    wc = Cohort.from_arrays(ages, events, delta=delta, continuous=cohort.continuous, name=cohort.name,
                            window=(lo, hi), time_unit=cohort.time_unit)
    record = {"window_given": [lo, "inf" if mapped else float(cohort.window[1])], "hi_mapped_from": mapped,
              "hi_used": hi, "hi_rule": HI_INF_RULE if mapped else None,
              "n_rows_given": cohort.n, "n_dropped_below_lo": int(cohort.n - rows.size),
              "n_censored_above_hi": int(np.sum(np.asarray(cohort.ages, float) > hi)),
              "n_rows_kept": int(rows.size), "row_map": "the kept rows (age >= lo), in the cohort's order",
              "window_eff": prep.record["window_eff"], "lo_rule": snap.get("lo_rule"),
              "n_deaths_censored": int(snap.get("n_deaths_censored", 0)),
              "n_censored_at_hi_eff": int(snap.get("n_censored_at_hi_eff", 0)),
              "n_deaths_fitted": n_fit, "min_deaths": need, "rule": "the tester's apply_window, then the frozen snap"}
    return wc, record, rows


#: the diagnostics blocks whose rows are the fitted cohort's (1-based): the floor report's deaths and every row the
#: leverage check names (share, influence, dropped, drops)
ROW_BLOCKS = ("floor_report", "leverage")


def map_window_rows(diag: dict, rows):
    """Spec 006 (research R4): `diag` with every 1-based `row` under ROW_BLOCKS renamed to the original cohort's row
    through the row map (the kept rows' 0-based indices), the windowed row kept as `row_windowed`. `rows` None (no
    window): `diag` itself, unchanged."""
    if rows is None:
        return diag

    def walk(x):
        if isinstance(x, dict):
            out = {k: walk(v) for k, v in x.items()}
            if isinstance(x.get("row"), int) and not isinstance(x.get("row"), bool):
                out["row_windowed"] = x["row"]
                out["row"] = int(rows[x["row"] - 1]) + 1
            return out
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x
    return {k: (walk(v) if k in ROW_BLOCKS else v) for k, v in diag.items()}


#: the frozen code's clip of S(lo) in the conditioning term (compat/jax2_101/fp_jax.py:850; spec 006 FR-010)
FROZEN_S_LO_CLIP = 1e-12


def s_lo_warning(s_lo: float, lo: float, name: str):
    """WINDOW_S_LO_CLIPPED when S(lo) at theta-hat is below the frozen clip (research R7), else None."""
    if not s_lo < FROZEN_S_LO_CLIP:
        return None
    from sraging.warnings import window_warning
    return window_warning("WINDOW_S_LO_CLIPPED", S_lo_min=float(s_lo), lo=float(lo), calibrator=name)


#: spec 006 T067 (Naveh 2026-11-11, A2's rule): A2 fitted W on v6's window arm at n >= 25,000, and more than 10 % of
#: its closed claims were truncated above, so W warns there. Per gated parameter, A2's closed claims and why their
#: upper ends were truncated, as a2_006.json's "truncation" records; and (truncated, closed) on A1's cohorts planted
#: at real life tables, n < 25,000 (Naveh 2026-11-15, sr-naveh M1: the same below the split). tests/unit/
#: test_profile_jax2_w.py checks both against the committed records.
W_LARGE_N_MIN = 25_000
W_LARGE_N_TRUNCATED = {"eta": {"closed": 32, "box_limited": 14, "widened": 13, "contradicted": 1},
                       "epsilon": {"closed": 29, "box_limited": 13, "widened": 16, "contradicted": 0}}
W_REAL_TABLES_BELOW = {"eta": (56, 67), "epsilon": (37, 51)}


def _truncated_text(q: str, c: dict) -> str:
    why = [f"{c['box_limited']} at the box", f"{c['widened']} widened"] + \
        ([f"{c['contradicted']} below the lead pair's walk"] if c["contradicted"] else [])
    return f"{q} {c['box_limited'] + c['widened'] + c['contradicted']} of {c['closed']} ({', '.join(why)})"


def large_n_warning(n: int, name: str):
    """W_LARGE_N on W's windowed fit of n rows given (A2's n, before the window drops any) at n >= W_LARGE_N_MIN,
    else None (H was not gated there)."""
    if name != "profile_jax2_w" or n < W_LARGE_N_MIN:
        return None
    from sraging.warnings import window_warning
    detail = ", ".join(_truncated_text(q, c) for q, c in W_LARGE_N_TRUNCATED.items())
    below = "spec 006's A1: " + ", ".join(f"{q} {t} of {c}" for q, (t, c) in W_REAL_TABLES_BELOW.items())
    return window_warning("W_LARGE_N", n=int(n), n_min=W_LARGE_N_MIN, detail=detail, below=below)


def run_port(cohort, name: str, *, capture: bool = True, port: JaxPort = V100) -> dict:
    """The port's fit of `cohort` under `name`: {result, claims, sc_block, archive, cpu_s}. `archive` is (Z, LL),
    the union of every captured objective's archive (None when not captured). `cohort` may be an `Observation`,
    passed to `run_fit` as is (window included): the acceptance runs give the port the oracle's own observation;
    the public fit refuses windows (`_refuse`) and passes a Cohort."""
    cls = port.classes[name]
    jc, je = port.claims, port.estimator
    obs = cohort if isinstance(cohort, Observation) else observation(cohort)
    c0 = time.process_time()
    nodes = None
    if capture:
        with capture_objectives(port) as objs, capture_nodes(port) as nodes:
            res = jc.run_fit(obs, **run_kwargs(cls))
    else:
        objs, res = None, jc.run_fit(obs, **run_kwargs(cls))
    cpu = time.process_time() - c0
    claims, sc_block = jc.final_claim_kwargs(res, cls.SELF_CONSISTENT)
    archive = None
    if objs is not None:
        parts = [je._sc_archive(o) for o in objs]
        np_ = max((p[0].shape[1] for p in parts), default=len(PARAMS))
        parts = [p for p in parts if p[0].shape[1] == np_]
        archive = (np.concatenate([p[0] for p in parts]) if parts else np.empty((0, np_)),
                   np.concatenate([p[1] for p in parts]) if parts else np.empty(0))
    return {"result": res, "claims": claims, "sc_block": sc_block, "archive": archive, "cpu_s": cpu, "objectives": objs,
            "nodes": nodes}


COMBINATION_MODES = ("lead", "all", "first3")
FIRST3 = 3          # spec 003d D1 (Naveh, 2026-10-13): the re-test walks the frozen order's first three


def check_combinations(value):
    """What a jax2 fit's `combinations` may be (Naveh, 2026-10-06): "lead" (the default: the frozen lead group),
    "all" (all six) or False (none); and "first3" (spec 003d D1, Naveh 2026-10-13: the frozen order's first three,
    the lead group then sqrt(eta)/eps, for the re-test). Anything else is refused, True included: it named "all six"
    before the decision, so it is ambiguous now."""
    if value is False or (isinstance(value, str) and value in COMBINATION_MODES):
        return value
    raise UnsupportedOptionError(f"combinations={value!r}", "a jax2 fit takes combinations='lead' (the default: "
                                 "the frozen lead group), 'all' (all six), 'first3' (the first three) or False (none)")


def combination_claims(out, name: str, theta_hat, mode: str = "all", port: JaxPort = V100) -> dict:
    """Spec 003 US2 (FR-014 to FR-019): a claim per combination, from its own profile over the original box (the
    walker, on the fit's own captured objective, so its points join the archive), widened by jax2's own rule over
    the archive. `mode` "lead" walks the frozen lead group only (a fit's default); "all" walks all six, the lead
    group first, and widens the lead claims over the archive as it stood after the lead walks, so they equal the
    default's; the rest are widened over the whole archive. "first3" (spec 003d D1) is "all" stopped after the frozen
    order's third combination (sqrt(eta)/eps), widened over the archive after three walks. Returns {claims: [Claim], walks, extents, gap_nats,
    cost_s, evals, order, order_source, mode, lead_group, param_claims_contradicted,
    combination_claims_contradicted}.
    theta-hat and the per-parameter claims are not touched."""
    from sraging.claims import combinations as K
    from sraging.claims import walker as W
    from sraging.claims import widening as Wd
    res = out["result"]
    cls = port.classes[name]
    objs = [o for o in (out.get("objectives") or []) if o.NP == (5 if cls.HAZARD else 4)]
    obj = main_objective(out, name, port)
    theta = list(res["theta"])
    z_hat = np.log(np.asarray(theta, float))
    ll_hat = float(res["loglik"])
    box = dict(obj.box)
    bounds = [(math.log(box[p][0]), math.log(box[p][1])) for p in obj.params]
    box4 = {p: box[p] for p in PARAMS}
    seed = getattr(obj, "seed_theta", None)          # 5-long under the hazard already (estimator.py:1109-1116)
    seed_z = None
    if seed is not None:
        s = list(seed) + ([theta[4]] if len(seed) == obj.NP - 1 else [])
        seed_z = np.log(np.clip(s, 1e-12, None)) if len(s) == obj.NP else None
    hazard_h = []
    if cls.HAZARD:
        lo, hi = box["h"]
        hazard_h = [lo] + [min(max(x / obj.L, lo), hi) for x in (0.01, 0.1, 0.5)]
    order, source = K.order()
    lead, _lead_source = K.lead_group()
    keys = [k for k in order if k in K.KEYS]            # kappa is fitted by both jax2 names: all six
    lead_keys = [k for k in keys if k in lead]
    rest = [k for k in keys if k not in lead]
    keys = lead_keys + {"all": rest, "first3": rest[:FIRST3 - len(lead_keys)]}.get(check_combinations(mode), [])
    c0 = time.process_time()

    def archive():
        return fit_archive(objs, port)

    def reconcile(group):
        """SC-004 (Naveh, 2026-10-06): re-walk any side the archive contradicts, while the archive grows."""
        for _round in range(W.RECONCILE_ROUNDS):
            Z, LL = archive()
            moved = False
            for key in group:
                walks[key], rec = W.reconcile(obj, key, walks[key], Z, LL, z_hat=z_hat, ll_hat=ll_hat, bounds=bounds,
                                              box4=box4, seed_z=seed_z, archive=(Z, LL), hazard_h=hazard_h)
                moved = moved or rec is not None
            if not moved:
                break

    walks = {}
    for group in (lead_keys, [k for k in keys if k not in lead_keys]):
        for key in group:
            walks[key] = W.walk(obj, key, z_hat, ll_hat, bounds=bounds, box4=box4, seed_z=seed_z, archive=archive(),
                                hazard_h=hazard_h)
        reconcile(group)
        if group is lead_keys:
            Z_lead, LL_lead = archive()                 # the archive as the default ("lead") fit ends with it
    Z, LL = archive()
    claims, extents, lead_contradicted = [], {}, []
    for key in keys:
        w = walks[key]
        point = math.exp(K.log_g(key, z_hat))
        kw = Wd.claim_from_walk(key, w, point, jc=port.claims)
        # The lead claims are widened over the archive after the lead walks, so "all" never changes the default
        # claims; a later walk's point beyond a lead end is reported, as for the per-parameter claims (FR-019).
        za, lla = (Z_lead, LL_lead) if key in lead_keys else (Z, LL)
        new, rec, _sc = Wd.widen(key, kw, za, lla, ll_hat, box, w["box_image"], jc=port.claims, je=port.estimator)
        if key in lead_keys and LL.size > LL_lead.size:
            full, _r, _s = Wd.widen(key, kw, Z, LL, ll_hat, box, w["box_image"], jc=port.claims, je=port.estimator)
            for side in ("lower", "upper"):
                a, b = new.get(side), full.get(side)
                if a is not None and b is not None and (b < a if side == "lower" else b > a):
                    lead_contradicted.append({"parameter": key, "side": side, "claim_end": float(a),
                                              "all_walks_end": float(b), "dex": abs(math.log10(b / a))})
        extents[key] = _sc["extents"].get(key)
        widening = None
        if rec.get("changed"):
            widening = {"claims_profile": {"shape": kw.get("shape"), "lower": kw.get("lower"), "upper": kw.get("upper")},
                        "lower_moved_dex": rec.get("lower_moved_dex"), "upper_moved_dex": rec.get("upper_moved_dex")}
        decline = None
        if new["shape"] == "none":
            decline = f"profile verdict {w['ci']['verdict']!r}; walk stops {w['stops']}"
        claims.append(Claim(key, new["shape"], new.get("lower"), new.get("upper"), new.get("point"), new["level"],
                            int(w["ci"].get("n_components") or 1), "", widening=widening, decline_reason=decline))
    sc_all = port.estimator.sc_extents(Z, LL, list(obj.params), box, ll_hat, 3.841458820694)
    contradicted = []                     # FR-019 keeps the per-parameter claims; the walks' points are only counted
    for kw in out["claims"]:
        e = (sc_all["extents"] or {}).get(kw["parameter"]) or {}
        for side, end, q, beyond in (("lower", kw.get("lower"), e.get("qmin"), lambda q, x: q < x),
                                     ("upper", kw.get("upper"), e.get("qmax"), lambda q, x: q > x)):
            if end is not None and q is not None and beyond(q, end):
                contradicted.append({"parameter": kw["parameter"], "side": side, "claim_end": float(end),
                                     "archive_point": float(q), "dex": abs(math.log10(q / end))})
    return {"param_claims_contradicted": contradicted, "combination_claims_contradicted": lead_contradicted,
            "claims": claims, "walks": walks, "extents": extents, "gap_nats": float(sc_all["gap_nats"]),
            "archive_points": int(LL.size),
            "cost_s": time.process_time() - c0, "evals": int(sum(w["evals"] for w in walks.values())),
            "order": keys, "order_source": source, "mode": mode, "lead_group": lead_keys}


def fit_inputs(cohort, name: str, port: JaxPort = V100) -> dict:
    """What `estimator.fit` solves on, built by its own statements on the port: the prepared data (relabelled events),
    L, the grid keywords (N rule, mesh, exact-time switch), the box and the effective window (tools/jax2_cases.py is
    the side-by-side twin). `cohort` may be an `Observation`, as for `run_port`."""
    cls = port.classes[name]
    _je, _om = port.estimator, port.observation_model
    obs = cohort if isinstance(cohort, Observation) else observation(cohort)
    prep = _om.prepare(obs.ages, obs.events, obs.delta, obs.window, window_on=cls.WINDOW, relabel_on=cls.RELABEL)
    t = np.asarray(prep.ages, float)
    e = np.asarray(prep.events, int)
    delta = obs.delta if obs.delta else None
    L, kw, ceiling, dt_floor = _je._grid(t, e, delta, floor_early=cls.FLOOR_EARLY, n_rule=cls.N_RULE,
                                         dt_rule=cls.DT_RULE, **({"early_all": True} if cls.EARLY_ALL else {}))
    if cls.EXACT_F:
        kw = dict(kw, exact_from_F=True)
    box = _je.box_for(port.claims.KAPPA_MAX)
    if cls.HAZARD:
        box = dict(box, h=_je._h_box(L))
    return {"t": t, "e": e, "L": float(L), "kw": kw, "box": box, "window": prep.window_eff, "hazard": cls.HAZARD,
            "delta": delta, "dt_floor": dt_floor}


def apply_labels(claims, out, combos, theta_hat, cohort, name, lsr=None, port: JaxPort = V100):
    """FR-020 to FR-024: box-limited, negligible, saturated and widened labels on the parameter and combination
    claims. The label text is appended to the claim's existing label; label_info carries the first that applies
    (box-limited, then negligible or saturated, then widened). Never changes a shape or a bound. Each profile's
    endpoint node is read against the profile's own maximum (the reference `profile_ci` measures from); a widened
    upper end is read at the archive point that set it (FR-020)."""
    import dataclasses
    from sraging.claims import shapes as SH
    res = out["result"]
    ll_hat = float(res["loglik"])
    obj = main_objective(out, name, port)
    box = dict(obj.box) if obj is not None else settings(name, port)["box"]
    params = tuple(obj.params) if obj is not None else PARAMS
    nodes = out.get("nodes") or {}
    walks = (combos or {}).get("walks_full") or {}
    ext_c = (combos or {}).get("extents_full") or {}
    ext_p = (((res.get("diagnostics") or {}).get("self_consistency") or {}).get("extents") or {})
    floors = load_floors()
    by_param = {c.parameter: c for c in claims}
    out_claims = []
    for c in claims:
        infos = []
        if c.parameter in PARAMS:
            ns = [(v, ll, None if z is None else np.asarray(z)) for v, ll, z in nodes.get(PARAMS.index(c.parameter), [])
                  if np.isfinite(ll)]
            ll_ref = max([ll_hat] + [n[1] for n in ns])
            infos.append(SH.box_limited_label(c, ns, ll_ref, box, params,
                                              archive_theta=_widened_point(c, ext_p.get(c.parameter))))
            if c.parameter == "kappa" and c.shape == "upper_bound":
                infos.append(SH.saturated_label(c.upper, cohort.n_deaths))
        elif c.parameter in walks:
            w = walks[c.parameter]
            ns = [(math.exp(cc), ll, np.asarray(z)) for cc, ll, z in w["nodes"]]
            infos.append(SH.box_limited_label(c, ns, w.get("ll_ref", ll_hat), box, params,
                                              archive_theta=_widened_point(c, ext_c.get(c.parameter))))
            if (c.parameter == "rbar_over_eta" and c.shape == "upper_bound" and floors is not None and lsr
                    and not ((c.widening or {}).get("upper_moved_dex") or 0) > 0):   # never on a widened upper side
                regime, why = negligible_regime(by_param.get("epsilon"), theta_hat, lsr, floors)
                if regime is not None:
                    lab = SH.negligible_label(c.upper / lsr, cohort.n_deaths, regime,
                                              {k: v["c_rbar"] for k, v in floors.items()})
                    infos.append(None if lab is None else dict(lab, L_SR=lsr, regime_rule=why))
        wt = SH.widened_text(c)
        if wt:
            infos.append({"kind": "widened", "edge": None, "log10_cap_over_floor": None, "regime": None, "text": wt})
        infos = [x for x in infos if x]
        if not infos:
            out_claims.append(c)
            continue
        text = " · ".join([c.label] + [x["text"] for x in infos] if c.label else [x["text"] for x in infos])
        info = {k: v for k, v in infos[0].items() if k != "text"}
        out_claims.append(dataclasses.replace(c, label=text, label_info=info))
    return out_claims


def _widened_point(claim, ext):
    """The archive point that set a widened upper end (the extents' theta_qmax), or None if the upper end did not move."""
    if not ext or claim.upper is None or not ((claim.widening or {}).get("upper_moved_dex") or 0) > 0:
        return None
    return ext.get("theta_qmax")


FLOORS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "claims", "floors.json")
RHO_EPS_MIN = 0.10          # the sec. 8b sweep's smallest rho_eps (space_search.py:72-77): no floor is measured below


def load_floors():
    """The frozen negligibility floors (claims/floors.json, copied from results/neg_floor_003.json after the
    `floor` phase); None until frozen, and then no "negligible" label is given."""
    import json
    if not os.path.exists(FLOORS_FILE):
        return None
    return json.load(open(FLOORS_FILE))["regimes"]


def negligible_regime(eps_claim, theta_hat, lsr, floors):
    """(regime, why) for the negligibility floor (sr-naveh implementation review 9). By rho_eps = eps L_SR alone
    (kappa is not identified when removal is negligible, and r-bar's coefficient barely depends on it: mid 1.535 against
    kappa_large 1.556): over the rho_eps range of the epsilon claim, the regime with the SMALLEST coefficient among
    those nearest any rho in that range (the conservative floor). Refused (None) when rho_eps-hat is below 0.1."""
    rho_hat = theta_hat.epsilon * lsr
    if not rho_hat >= RHO_EPS_MIN:
        return None, f"rho_eps {rho_hat:.3g} is below {RHO_EPS_MIN}: no floor is measured there"
    lo = (eps_claim.lower * lsr) if (eps_claim is not None and eps_claim.lower is not None) else RHO_EPS_MIN
    hi = (eps_claim.upper * lsr) if (eps_claim is not None and eps_claim.upper is not None) else math.inf
    lo = max(lo, RHO_EPS_MIN)

    def nearest(rho):
        d = {r: abs(math.log(rho / floors[r]["rho_eps"])) for r in floors}
        m = min(d.values())
        return {r for r, v in d.items() if v <= m + 1e-12}
    cands = nearest(rho_hat) | nearest(lo) | (nearest(hi) if math.isfinite(hi) else set())
    cands |= {r for r in floors if lo <= floors[r]["rho_eps"] <= hi}
    regime = min(sorted(cands), key=lambda r: floors[r]["c_rbar"])
    return regime, f"rho_eps {rho_hat:.3g} (claim range {lo:.3g}-{hi:.3g}): smallest coefficient among {sorted(cands)}"


def box_in_lsr(box, lsr):
    """FR-021: the box, and each combination's box image, in L_SR units (each quantity times L_SR^power, dimensionless)."""
    from sraging.claims import combinations as K
    out = {p: [box[p][0] * lsr ** K.LSR_POWER[p], box[p][1] * lsr ** K.LSR_POWER[p]] for p in box if p in K.LSR_POWER}
    box4 = {p: box[p] for p in PARAMS}
    for key in K.KEYS:
        lo, hi = K.box_image(key, box4)
        out[key] = [lo * lsr ** K.LSR_POWER[key], hi * lsr ** K.LSR_POWER[key]]
    return out


def _refuse(cohort, name, port: JaxPort = V100):
    if cohort.entry is not None:
        raise UnsupportedOptionError("entry ages", f"{name} does not fit entry ages (no tester arm scores them; spec "
                                     "006 Not Covered)")
    if cohort.window is not None and not fits_window(name, port):
        raise UnsupportedOptionError("window", f"{name}@{port.version} does not fit a window; use profile_jax2_w "
                                     "(h = 0) or profile_jax2_h (h fitted), which apply the tester's window rule "
                                     "(spec 006)")
    if np.ndim(cohort.delta) == 1:
        raise UnsupportedOptionError("per-row delta", f"{name} takes one recording resolution, not a per-row delta (spec 006)")


def _mechanisms(res, sc_block) -> dict:
    dg = res.get("diagnostics") or {}
    bf = dg.get("basin_fix") or {}
    ct = dg.get("censor_tie") or {}
    dtf = dg.get("dt_floor") or {}
    widened = [r["parameter"] for r in ((sc_block or {}).get("widening") or []) if r.get("changed")]
    return {"basin_refit": {"fired": bool(bf.get("fired")), "rounds": int(bf.get("n_refit") or 0)},
            "relabelled_rows": int(ct.get("n_relabelled") or 0) if ct.get("applied") else 0,
            "N": dtf.get("N"), "dt_floor": {"binds": bool(dtf.get("binds")), "dt": dtf.get("dt")},
            "n_fallback": int(dg.get("n_fallback") or 0), "widened": widened}


def _make_fit(name, port: JaxPort = V100):
    def fit(cohort, *, combinations="lead") -> FitResult:
        """Fit one cohort with the port's jax2 numerics (1.0.0 or 1.0.1), plus the combination claims: the frozen
        lead group by default, all six with combinations="all", none with False (Naveh, 2026-10-06)."""
        check_combinations(combinations)
        _refuse(cohort, name, port)
        runtime.check()
        t0 = time.perf_counter()
        wc, window, rows = window_path(cohort, port.classes[name], port) if fits_window(name, port) else \
            (cohort, None, None)                                  # spec 006: the window W and H fit, rows mapped back
        out = run_port(wc, name, capture=True, port=port)         # errors the frozen code raises propagate
        return fit_result(wc, name, out, port=port, combinations=combinations, t0=t0, window=window, rows=rows,
                          given=cohort)

    return fit


def fit_result(cohort, name, out, *, port: JaxPort = V100, combinations="lead", t0=None, window=None, rows=None,
               given=None) -> FitResult:
    """The FitResult of one port run `out` (`run_port`'s dict) of `cohort` under `name`: the method's diagnostics
    (`method_diagnostics`), sraging's labels, combination claims, floors, leverage check and warnings. The public fit
    calls it on its own run; spec 003b's parity phase on the run it compares with the oracle. Spec 006: for a window,
    `cohort` is `window_path`'s windowed cohort (the rows fitted), `window` its record (diagnostics["window"], with S(lo)
    at theta-hat), `rows` its row map (every reported row mapped back) and `given` the cohort as given (the summary)."""
    cls = port.classes[name]
    spec = registry.get(name, port.version)
    src = cohort if given is None else given
    summary = {"name": src.name, "n": src.n, "n_deaths": src.n_deaths,
               "n_censored": src.n_censored, "delta": src.delta, "continuous": src.continuous}
    t0 = time.perf_counter() if t0 is None else t0
    res, claim_kw, sc_block = out["result"], out["claims"], out["sc_block"]
    dg = dict(res.get("diagnostics") or {})
    grid = {**(dg.get("solver") or {}), "dt_ceiling": res.get("dt_ceiling"),
            "early": (dg.get("dt_floor") or {}).get("early")}
    prov = provenance.collect(spec, grid, runtime.state())
    if name in COMPOSITIONS:
        prov["calibrator"]["composition"] = COMPOSITIONS[name]
    diag = dg = method_diagnostics(port, cls, res, claim_kw, sc_block)   # dg: the same dict, as before
    diag["code"] = provenance.code_line(prov)
    ends = kappa_ends_jax2(diag, settings(name, port)["box"])          # spec 006 FR-025: sraging's own key
    if ends is not None:
        diag["kappa_ends"] = ends
    if not cls.HAZARD:
        diag["assumptions"] = NO_EXTERNAL_MORTALITY
    notes = notes_for(port, cls)
    if notes:
        diag["notes"] = notes

    theta = list(res["theta"])
    assert len(theta) == len(PARAMS) + (1 if cls.HAZARD else 0), f"{name}: {len(theta)} parameters"
    theta_labels, nuisance = {}, None
    h = float(theta[4]) if cls.HAZARD else 0.0
    try:
        theta_hat = SRParams(**{p: float(v) for p, v in zip(PARAMS, theta[:4])}, h_ext=h,
                             time_unit=cohort.time_unit)
    except ValueError as ex:
        theta_hat = None
        theta_labels["_theta_hat"] = f"theta-hat not representable as SRParams: {ex}"
    if cls.HAZARD:
        hz = dg.get("hazard") or {}
        L = hz.get("L")
        nuisance = {"h_ext": h, "hL": (h * float(L)) if L else None,
                    "ExtH": (-math.log(h) if h > 0 else None), "scored": False,
                    "would_be_claim": hz.get("h_claim")}

    bare = tuple(Claim(c["parameter"], c["shape"], c.get("lower"), c.get("upper"), c.get("point"),
                       c["level"], 1) for c in claim_kw)
    lab = labels(bare, theta_hat)
    prof = {c["parameter"]: c for c in ((sc_block or {}).get("claims_profile") or [])}
    recs = {r["parameter"]: r for r in ((sc_block or {}).get("widening") or [])}
    claims = []
    for c in bare:
        r = recs.get(c.parameter)
        widening = None
        if r is not None and r.get("changed"):
            p = prof.get(c.parameter, {})
            widening = {"claims_profile": {"shape": p.get("shape"), "lower": p.get("lower"), "upper": p.get("upper")},
                        "lower_moved_dex": r.get("lower_moved_dex"), "upper_moved_dex": r.get("upper_moved_dex")}
        claims.append(Claim(c.parameter, c.shape, c.lower, c.upper, c.point, c.level, 1, lab[c.parameter],
                            widening=widening))
    theta_labels.update(lab)
    Z, LL = out["archive"]
    ll_hat = res.get("loglik")
    sc = dg.get("self_consistency") or {}
    gap = float(sc["gap_nats"]) if sc.get("gap_nats") is not None else 0.0   # jax2's own: best in-box point - ll-hat
    n_points = int(LL.size)
    combos = {"on": False, "cost_s": 0.0, "order": [], "order_source": "off (combinations=False)"}
    if combinations and theta_hat is not None:
        cc = combination_claims(out, name, theta_hat, mode=combinations, port=port)
        claims.extend(cc["claims"])
        gap, n_points = max(gap, cc["gap_nats"]), cc["archive_points"]       # FR-017: a walk's better point counts
        combos = {"on": True, "mode": cc["mode"], "lead_group": cc["lead_group"], "cost_s": cc["cost_s"],
                  "fit_cost_s": out["cpu_s"], "evals": cc["evals"],
                  "param_claims_contradicted": cc["param_claims_contradicted"],
                  "combination_claims_contradicted": cc["combination_claims_contradicted"],
                  "ratio_to_fit": cc["cost_s"] / out["cpu_s"] if out["cpu_s"] else None,
                  "order": cc["order"], "order_source": cc["order_source"],
                  "walks": {k: {"ci": w["ci"], "stops": w["stops"], "evals": w["evals"], "gain": w["gain"],
                                "n_nodes": len(w["nodes"])} for k, w in cc["walks"].items()},
                  "walks_full": cc["walks"], "extents_full": cc["extents"]}

    lsr = None
    if theta_hat is not None:              # FR-021: L_SR and the box in L_SR units
        from sraging.claims.lsr import l_sr
        lsr_rec = l_sr(theta_hat)
        lsr = lsr_rec["L_SR"]
        diag["L_SR"] = {**lsr_rec, "time_unit": cohort.time_unit}
        mo = main_objective(out, name, port)
        diag["box_lsr"] = box_in_lsr(dict(mo.box) if mo is not None else settings(name, port)["box"], lsr)
        claims = apply_labels(claims, out, combos, theta_hat, cohort, name, lsr=lsr, port=port)
    floor_report = None
    if theta_hat is not None:              # FR-011: terms from the fit's own likelihood and grid, after the fit
        from sraging.floors import report_jax2
        floor_report = report_jax2(cohort, theta_hat, name, port=port)
        diag["floor_report"] = floor_report
    from sraging.leverage import check as leverage_check     # spec 003f: after the fit, never altering it
    diag["leverage"] = leverage_check(cohort, name, theta_hat, diag, fit_cpu_s=out["cpu_s"], port=port)
    window_warnings = ()
    if window is not None:                 # spec 006 (R4, R7): the window used, S(lo) at theta-hat, rows mapped back
        s_lo = None if floor_report is None else floor_report.get("S_lo")
        diag["window"] = {**window, "S_lo_at_theta_hat": s_lo}
        diag.update(map_window_rows({k: diag[k] for k in ROW_BLOCKS if k in diag}, rows))
        floor_report = diag.get("floor_report")
        w = None if s_lo is None else s_lo_warning(s_lo, float(window["window_eff"][0]), name)
        window_warnings = tuple(x for x in (w, large_n_warning(window["n_rows_given"], name)) if x is not None)
    from sraging.warnings import evaluate, leverage_warnings
    warnings, checks = evaluate(cohort, tuple(claims), diag,
                                theta_hat=None if theta_hat is None else theta_hat.as_dict(),
                                box=settings(name, port)["box"], floor_report=floor_report, gap_nats=gap,
                                relabel=dg.get("censor_tie"))
    assert spec.version == port.version
    warnings = warnings + leverage_warnings(diag["leverage"]) + version_warnings(port, cls) + window_warnings
    more = []                              # spec 003g FR-012, FR-013: notes that depend on the fit
    if cls.HAZARD and port.version != "1.0.0":
        try:
            cv = None if theta_hat is None else sr_cv(theta_hat)
        except Exception:  # noqa: BLE001 -- the note says it could not be computed
            cv = None
        more.append(h_stratum_note(cv))
    cl = combination_limit_note(port.version, combos)
    if cl is not None:
        more.append(cl)
    sl = sharp_limit_note(port.version, cls, diag)
    if sl is not None:
        more.append(sl)
    if more:
        diag["notes"] = list(diag.get("notes") or []) + more
    return FitResult(
        status="ok", cohort=summary, theta_hat=theta_hat, theta_labels=theta_labels, claims=tuple(claims),
        loglik=ll_hat, warnings=tuple(warnings), checks=tuple(checks), diagnostics=diag,
        timing={"wall_s": time.perf_counter() - t0, "cpu_s": out["cpu_s"]}, provenance=prov,
        nuisance=nuisance, mechanisms=_mechanisms(res, sc_block), gap_nats=gap, archive_points=n_points,
        combinations={k: v for k, v in combos.items() if k not in ("walks_full", "extents_full")})


for _version, _port in PORTS.items():                    # 1.0.0 first; 1.0.1 is what a bare name means
    for _name in _port.classes:
        registry.register(registry.CalibratorSpec(name=_name, version=_version, numerics="jax2",
                                                  settings=settings(_name, _port), fit=_make_fit(_name, _port)),
                          current=(_version == CURRENT_VERSION))
registry.register(registry.CalibratorSpec(name="profile_jax2_w", version=PORT_W.version, numerics="jax2",
                                          settings=settings("profile_jax2_w", PORT_W),
                                          fit=_make_fit("profile_jax2_w", PORT_W)), current=True)   # spec 006 FR-002
_KFIX_REASON = "registered in a later spec, after the kappa0 table (spec 003b FR-007; the jax2 1.0.1 handoff 1.2 (a))"
for _kname in PORTS["1.0.1"].kfix_classes:              # the six kappa-fixed names, by jax2's and sraging's spelling
    registry.UNREGISTERED[_kname] = _KFIX_REASON
    registry.UNREGISTERED[_kname.replace("mle_", "", 1)] = _KFIX_REASON


def _not_run():
    return tuple(CheckRecord(c, False, False) for c in CHECK_CODES)
