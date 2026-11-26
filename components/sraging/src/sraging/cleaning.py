"""The MAD cage cleaning, the lab's rule, reproducing the mosquito and fly records (spec 005 US1; FR-013 to FR-017;
research R4, R5; data-model "CleaningReport"; register D-501, D-503, D-504).

    report = clean(cages, by="cohort", k=2.5, seed=1)        # flags; removes nothing
    kept = drop(cages, report)                               # removes exactly the flagged cages, naming the report

The rule is `cage_cleaning.clean_by_mad` (<A>/code_5/mosquitos/cage_cleaning.py:96-267, which the mosquito reanalysis
(D5) and the fly redo (D2, D19) use), reimplemented formula for formula:
- each cage is scored by its KM median and IQR steepness (`sraging.survival`; by default first-crossing median and
  `nearest` quartiles, SRK Dataset's definitions, which the records use: C2);
- per axis, the centre is the median of the finite scores and σ = 1.4826 · MAD; the radius is k σ, raised to the
  floor k φ |centre| when smaller or when σ is not finite, and infinite when it is still not positive (every cage
  tied, no floor);
- each cage is judged against the OTHER cages of its cohort when the cohort has more than two (leave-one-out,
  :232), else against the whole group; the distance is axis-aligned, √(u² + v²) with u = (x - centre) / radius, or
  |u| on the median alone (axes="median", the fly's 2-8 LONG choice); a cage is flagged when its distance exceeds 1;
- the floors φ are pooled over the cohorts with at least `min_cages` (10) cages: the median over them of σ / |centre|
  (`pooled_relative_dispersion`, :126-146), or given.

Where it differs (register): a non-finite score on an axis the rule uses is refused, naming the cage and the axis,
where the legacy code kept the cage at distance 0 (:249; D-503); and nothing is dropped by `clean` (D-504).

The report carries, per cohort, the flags that Gaussian scatter of the scores alone would give under two named nulls
(R5, the fly redo's construction, recal_2026-10/src/clean.py:139-214): scores drawn around 1 with relative sd s per
axis, judged by the same rule with the same floors, R times from `numpy.random.default_rng` seeded by `seed` and the
cohort size. The nulls know nothing of the animals per cage:
- "floor": s = φ;
- "own_spread": s = the pool's own spread (`pooled_spread`, the fly redo's group_spreads, which HR-4 accepted): each
  cohort's leave-one-out relative deviations x_i / median(others) - 1, divided by r(m), the sd of such deviations
  for m iid N(0, 1) cages (the leave-one-out correction), pooled over the cohorts sharing the `spread_by` design value
  (every cohort of the set by default), robust sd. A cohort's own 3-15 cages are too few: two tied cages give s = 0
  (sr-naveh_005 B2). An s that is 0 or not finite is not measurable: no expected flags, a reason.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from types import MappingProxyType

import numpy as np

from sraging.cohorts import CohortSet
from sraging.errors import CohortDataError, CohortSpecError, UnsupportedOptionError
from sraging.survival import km, summary

MIN_CAGES = 10                    # cage_cleaning.MIN_CAGES_FOR_DISPERSION
MAD_SCALE = 1.4826
AXES = ("both", "median")
MEDIAN_DEFS = ("first-crossing",)
LOO_CORRECTION_REPS = 20_000      # the fly redo's loo_correction(reps=20000)
_NULL_TAG = {"floor": 1, "own_spread": 2, "loo_correction": 3}


# ------------------------------------------------------------------------------------------------- the scores
def cage_scores(cage_set: CohortSet, *, median_def="first-crossing", quartiles="nearest") -> dict:
    """{key: (median, steepness)} from each member's KM (`sraging.survival`), the definitions named (FR-014)."""
    if median_def not in MEDIAN_DEFS:
        raise UnsupportedOptionError("median_def", f"median_def is one of {MEDIAN_DEFS}, got {median_def!r}")
    out = {}
    for key, c in cage_set.cohorts.items():
        s = summary(km(c), quartiles=quartiles)
        out[key] = (s.median.value, s.steepness.value)
    return out


def definitions(*, median_def, quartiles) -> dict:
    return {"median": f"{median_def}: the first age where the KM's Ŝ <= 0.5 (lifelines' median)",
            "quartiles": quartiles,
            "steepness": f"median / (t(0.25) - t(0.75)) with {quartiles} quartile ages (SRK Dataset's getSteepness "
                         f"when nearest)"}


# --------------------------------------------------------------------------------------------- the geometry
def robust_sigma(values) -> float:
    """1.4826 · MAD over the finite values; NaN when there are none (cage_cleaning.robust_sigma)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(MAD_SCALE * np.median(np.abs(v - np.median(v))))


def axis_geometry(values, *, k, floor_rel) -> dict:
    """{centre, sigma, radius, is_floor} for one axis (cage_cleaning._axis_geometry, with whether the floor set the
    radius)."""
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    centre = float(np.median(finite)) if finite.size else float("nan")
    sigma = robust_sigma(v)
    radius = k * sigma if np.isfinite(sigma) else float("nan")
    floor_radius = k * floor_rel * abs(centre) if floor_rel is not None and np.isfinite(centre) else float("nan")
    is_floor = bool(np.isfinite(floor_radius) and not (np.isfinite(radius) and radius >= floor_radius))
    if is_floor:
        radius = floor_radius
    if not np.isfinite(radius) or radius <= 0:
        radius = math.inf
    return {"centre": centre, "sigma": sigma, "radius": float(radius), "is_floor": is_floor}


def pooled_floors(scores_by_cohort: dict, *, min_cages=MIN_CAGES) -> tuple:
    """(φ_ml, φ_st, cohorts used): per axis, the median over the cohorts with >= min_cages cages of |σ / centre|
    (cage_cleaning.pooled_relative_dispersion). φ is NaN on an axis no such cohort measures; both NaN is refused by
    the caller."""
    rel = ([], [])
    used = []
    for co, (ml, st) in scores_by_cohort.items():
        if len(ml) < min_cages:
            continue
        used.append(co)
        for values, sink in ((np.asarray(ml, float), rel[0]), (np.asarray(st, float), rel[1])):
            centre = np.median(values[np.isfinite(values)]) if np.isfinite(values).any() else np.nan
            sigma = robust_sigma(values)
            if np.isfinite(centre) and centre != 0 and np.isfinite(sigma):
                sink.append(abs(sigma / centre))
    phi = tuple(float(np.median(r)) if r else float("nan") for r in rel)
    return phi[0], phi[1], tuple(used)


def judge(ml, st, *, k, floors, axes="both", leave_one_out=True) -> dict:
    """One cohort's rule on its scores: {distance, flagged, loo, geometry: [(ml geometry, st geometry) per cage]}.
    `floors` = (φ_ml, φ_st); a φ of None applies no floor on that axis (the legacy default)."""
    ml, st = np.asarray(ml, float), np.asarray(st, float)
    m = ml.size
    loo = bool(leave_one_out and m > 2)
    both = axes == "both"

    def geo(sub_ml, sub_st):
        return (axis_geometry(sub_ml, k=k, floor_rel=floors[0]),
                axis_geometry(sub_st, k=k, floor_rel=floors[1]) if both else
                {"centre": float("nan"), "sigma": float("nan"), "radius": float("nan"), "is_floor": False})

    geos = [geo(np.delete(ml, i), np.delete(st, i)) for i in range(m)] if loo else [geo(ml, st)] * m
    dist = np.empty(m)
    for i, (gx, gy) in enumerate(geos):
        u = (ml[i] - gx["centre"]) / gx["radius"]
        dist[i] = np.sqrt(u ** 2 + ((st[i] - gy["centre"]) / gy["radius"]) ** 2) if both else np.sqrt(u ** 2)
    return {"distance": dist, "flagged": dist > 1.0, "loo": loo, "geometry": geos}


# ------------------------------------------------------------------------------------ the noise baseline (R5)
def _rng(seed, m, null):
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(m), _NULL_TAG[null]]))


def _geom_batch(v, k, floor):
    """axis_geometry along axis 1 of finite draws: (centre, radius)."""
    centre = np.median(v, axis=1)
    radius = k * (MAD_SCALE * np.median(np.abs(v - centre[:, None]), axis=1))
    if floor is not None:
        floor_radius = k * floor * np.abs(centre)
        radius = np.where(radius >= floor_radius, radius, floor_radius)
    return centre, np.where(radius > 0, radius, np.inf)


def _flags_batch(ml, st, k, floors, axes, leave_one_out) -> np.ndarray:
    """The rule on R draws of m cages at once ((R, m) arrays of finite scores): the flagged mask, as `judge`."""
    R, m = ml.shape
    both = axes == "both"
    if leave_one_out and m > 2:
        out = np.empty((R, m), bool)
        for i in range(m):
            cx, rx = _geom_batch(np.delete(ml, i, axis=1), k, floors[0])
            u = (ml[:, i] - cx) / rx
            if both:
                cy, ry = _geom_batch(np.delete(st, i, axis=1), k, floors[1])
                out[:, i] = np.sqrt(u ** 2 + ((st[:, i] - cy) / ry) ** 2) > 1.0
            else:
                out[:, i] = np.sqrt(u ** 2) > 1.0
        return out
    cx, rx = _geom_batch(ml, k, floors[0])
    u = (ml - cx[:, None]) / rx[:, None]
    if not both:
        return np.sqrt(u ** 2) > 1.0
    cy, ry = _geom_batch(st, k, floors[1])
    return np.sqrt(u ** 2 + ((st - cy[:, None]) / ry[:, None]) ** 2) > 1.0


def loo_correction(m, *, seed, reps=LOO_CORRECTION_REPS) -> float:
    """r(m): the robust sd of the leave-one-out deviations x_i - median(others) of m iid N(0, 1) cages (the fly redo's
    loo_correction); NaN for m < 2."""
    if m < 2:
        return float("nan")
    x = _rng(seed, m, "loo_correction").standard_normal((reps, m))
    dev = np.concatenate([x[:, i] - np.median(np.delete(x, i, axis=1), axis=1) for i in range(m)])
    return robust_sigma(dev)


def loo_relative_deviations(values) -> np.ndarray:
    """x_i / median(others) - 1 for each cage (the fly redo's group_spreads, before the correction)."""
    x = np.asarray(values, float)
    return np.array([x[i] / np.median(np.delete(x, i)) - 1 for i in range(x.size)]) if x.size > 1 else np.array([])


def pooled_spread(scores, *, seed) -> tuple:
    """(s_ml, s_st): the robust relative spread per axis of a pool of cohorts, `scores` = {cohort: (ml, st)}: each
    cohort's leave-one-out relative deviations divided by its r(m), pooled, robust sd (the fly redo's group_spreads,
    recal_2026-10/src/clean.py:150-164; HR-4). A one-cage cohort adds nothing; NaN when no cohort has two cages."""
    dev, r = ([], []), {}
    for ml, st in scores.values():
        if len(ml) not in r:
            r[len(ml)] = loo_correction(len(ml), seed=seed)
        for values, sink in ((ml, dev[0]), (st, dev[1])):
            sink.append(loo_relative_deviations(values) / r[len(ml)])
    return tuple(robust_sigma(np.concatenate(d) if d else []) for d in dev)


def _unmeasurable(m, s, axes) -> str | None:
    """Why a null of relative sd s = (s_ml, s_st) cannot be drawn for m cages, or None: an axis the rule uses whose s
    is not finite or not > 0 (a null without scatter flags nothing by construction)."""
    if m < 1:
        return "not measurable: no cages"
    bad = [f"s_{a} = {v!r}" for a, v in zip(("ml", "st"), s[:2 if axes == "both" else 1])
           if not (math.isfinite(v) and v > 0)]
    return (f"not measurable: {', '.join(bad)}; a null needs a finite spread > 0 on each axis the rule uses (0 when "
            f"the cages tie on the scores' grid)") if bad else None


def noise_baseline(m, k, *, floors, spread=None, reps, seed, axes="both", leave_one_out=True) -> dict:
    """The flags expected from Gaussian scatter of the scores alone in a cohort of m cages, under the two named nulls
    (FR-016; R5): {"floor": ..., "own_spread": ...}, each with s_ml, s_st, the flag rate per cage, the expected flags
    (rate x m), m, k, reps and seed. `floors` are the rule's; `spread` is the own-spread null's (s_ml, s_st),
    required: the report never shows one null alone. A null whose s is 0 or not finite on an axis the rule uses is
    not measurable: rate and expected flags NaN, with the reason."""
    if spread is None:
        raise UnsupportedOptionError("spread", "spread (s_ml, s_st) is required: the noise baseline always shows "
                                               "both nulls, sigma = floor and the pool's own spread (FR-016; "
                                               "pooled_spread())")
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 1:
        raise UnsupportedOptionError("reps", f"reps must be an integer >= 1, got {reps!r}")
    out = {}
    for null, s in (("floor", floors), ("own_spread", spread)):
        row = {"null": null, "s_ml": float(s[0]), "s_st": float(s[1]), "m": int(m), "k": float(k), "reps": reps,
               "seed": int(seed)}
        why = _unmeasurable(m, (row["s_ml"], row["s_st"]), axes)
        if why:
            out[null] = {**row, "rate": float("nan"), "expected_flags": float("nan"), "reason": why}
            continue
        rng = _rng(seed, m, null)
        ml = 1 + s[0] * rng.standard_normal((reps, m))
        st = 1 + (s[1] if math.isfinite(s[1]) else 0.0) * rng.standard_normal((reps, m))
        flags = int(_flags_batch(ml, st, k, floors, axes, leave_one_out).sum())
        out[null] = {**row, "rate": flags / (reps * m), "expected_flags": flags / reps}
    return out


# ------------------------------------------------------------------------------------------------- the report
@dataclass(frozen=True)
class CageVerdict:
    """One cage's row (data-model "CleaningReport", per cage); `data_sha256` fingerprints its ages and events."""

    cohort: object
    cage: object
    key: str
    n: int
    median: float
    steepness: float
    centre_ml: float
    centre_st: float
    radius_ml: float
    radius_st: float
    ml_radius_is_floor: bool
    st_radius_is_floor: bool
    distance: float
    verdict: str
    loo: bool
    data_sha256: str


@dataclass(frozen=True)
class CohortVerdict:
    """One cohort's row: k, the axes, the counts, whether leave-one-out applied, the rule in words, and the noise
    baseline's two nulls (the own-spread null names its pool: `spread_by`, `pool` (the value, or "all"),
    `pool_cohorts` and `pool_cages`, the cohorts with two or more cages whose deviations enter, and their cages)."""

    cohort: object
    k: float
    axes: str
    n_cages: int
    n_flagged: int
    loo: bool
    rule: str
    noise: tuple


@dataclass(frozen=True)
class CleaningReport:
    """The cleaning's verdicts: `cages`, `cohorts`, the floors and their source, the definitions, k, the axes,
    leave-one-out, the grouping column, the own-spread pools' column (None: one pool, every cohort), the seed and reps
    of the noise baseline. `sha256` identifies it (drop records it). Nothing in it removes anything."""

    cages: tuple
    cohorts: tuple
    floors: tuple
    floors_source: str
    floor_cohorts: tuple
    min_cages: int
    k: float
    axes: str
    leave_one_out: bool
    by: str
    spread_by: str | None = None
    definitions: dict = field(default_factory=dict)
    seed: int = 0
    noise_reps: int = 0
    time_unit: str = "input units"

    def __post_init__(self):
        object.__setattr__(self, "definitions", MappingProxyType(dict(self.definitions)))

    @property
    def flagged(self) -> tuple:
        return tuple(r.key for r in self.cages if r.verdict == "flagged")

    def as_dict(self) -> dict:
        d = {f: getattr(self, f) for f in ("floors", "floors_source", "floor_cohorts", "min_cages", "k", "axes",
                                          "leave_one_out", "by", "spread_by", "seed", "noise_reps", "time_unit")}
        d["definitions"] = dict(self.definitions)
        d["cages"] = [_row(r) for r in self.cages]
        d["cohorts"] = [{**_row(c), "noise": [dict(n) for n in c.noise]} for c in self.cohorts]
        return d

    @property
    def sha256(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True, allow_nan=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()


def _row(obj) -> dict:
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


def _fingerprint(c) -> str:
    h = hashlib.sha256(np.ascontiguousarray(c.ages, dtype=float).tobytes())
    h.update(np.ascontiguousarray(c.events, dtype=np.int64).tobytes())
    return h.hexdigest()


def _rule_text(k, axes, loo, floors_source):
    geometry = ("leave-one-out (each cage against the other cages)" if loo else
                "whole-group geometry (two or fewer cages, no leave-one-out; cage_cleaning.py:232)")
    on = "(median, steepness), axis-aligned" if axes == "both" else "the median alone (1-D)"
    return (f"MAD rule: radius max(k 1.4826 MAD, k phi |centre|) per axis at k = {k:g}, {geometry}, distance on {on}, "
            f"flagged when the distance exceeds 1; floors {floors_source}")


def _check_number(name, v, *, positive=True):
    if isinstance(v, bool) or not isinstance(v, (int, float, np.number)) or not math.isfinite(v) \
            or not (v > 0 if positive else v >= 0):
        raise UnsupportedOptionError(name, f"{name} must be a finite number {'> 0' if positive else '>= 0'}, got "
                                           f"{v!r}")
    return float(v)


def clean(cage_set: CohortSet, *, by: str, k: float, axes="both", leave_one_out=True, floors=None,
          min_cages=MIN_CAGES, median_def="first-crossing", quartiles="nearest", noise_reps=4000, spread_by=None,
          seed: int) -> CleaningReport:
    """Judge every cage of `cage_set` against the others of its cohort (the `by` column) and return the report;
    nothing is removed (FR-013 to FR-017). `k` and `seed` are required. `floors` = (φ_ml, φ_st) given, or None to pool
    them over the cohorts with >= `min_cages` cages (refused when there is none). The scores use `median_def` and
    `quartiles` (SRK Dataset's by default, the records' definitions), named in the report. The own-spread null's s is
    pooled over every cohort (`spread_by=None`) or over the cohorts sharing the value of the design column
    `spread_by`; a column not in the design table, or one whose value varies within a cohort, is refused."""
    k = _check_number("k", k)
    if axes not in AXES:
        raise UnsupportedOptionError("axes", f"axes is one of {AXES}, got {axes!r}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise UnsupportedOptionError("seed", f"seed must be an integer >= 0, got {seed!r}")
    if isinstance(noise_reps, bool) or not isinstance(noise_reps, int) or noise_reps < 1:
        raise UnsupportedOptionError("noise_reps", f"noise_reps must be an integer >= 1, got {noise_reps!r}")
    if isinstance(min_cages, bool) or not isinstance(min_cages, int) or min_cages < 1:
        raise UnsupportedOptionError("min_cages", f"min_cages must be an integer >= 1, got {min_cages!r}")
    if not isinstance(leave_one_out, bool):
        raise UnsupportedOptionError("leave_one_out", f"leave_one_out must be True or False, got {leave_one_out!r}")
    if spread_by is not None and (not isinstance(spread_by, str) or spread_by not in cage_set.columns):
        raise UnsupportedOptionError("spread_by", f"spread_by {spread_by!r} is not a column of the design table; its "
                                                  f"columns are {list(cage_set.columns)}")
    groups = cage_set.by(by)
    pool_of = {}
    for co, sub in groups.items():
        values = list(dict.fromkeys(sub.design[key][spread_by] for key in sub)) if spread_by is not None else ["all"]
        if len(values) > 1:
            raise UnsupportedOptionError("spread_by", f"spread_by {spread_by!r} varies within cohort {co!r} "
                                                      f"({values!r}): each cohort pools its spread with one group")
        pool_of[co] = values[0]
    scores = cage_scores(cage_set, median_def=median_def, quartiles=quartiles)
    used_axes = ("median", "steepness") if axes == "both" else ("median",)
    for key, (ml, st) in scores.items():
        for axis, v in zip(("median", "steepness"), (ml, st)):
            if axis in used_axes and not math.isfinite(v):
                raise CohortDataError(f"cage {key!r} has a non-finite {axis} ({v!r}) and axes={axes!r} uses it: "
                                      f"refused (the legacy rule kept such a cage at distance 0; D-503)"
                                      + (". axes='median' judges on the median alone" if axis == "steepness" else ""))
    by_cohort = {co: ([scores[key][0] for key in sub], [scores[key][1] for key in sub]) for co, sub in groups.items()}
    if floors is None:
        phi_ml, phi_st, used = pooled_floors(by_cohort, min_cages=min_cages)
        if not used or not math.isfinite(phi_ml) or (axes == "both" and not math.isfinite(phi_st)):
            raise CohortSpecError(f"no cohort has >= {min_cages} cages with finite scores, so no floors can be pooled; "
                                  f"pass floors=(phi_ml, phi_st) explicitly, or lower min_cages")
        source = f"pooled over the cohorts with >= {min_cages} cages"
    else:
        if not isinstance(floors, (tuple, list)) or len(floors) != 2:
            raise UnsupportedOptionError("floors", f"floors must be (phi_ml, phi_st), got {floors!r}")
        phi_ml, phi_st = _check_number("floors", floors[0], positive=False), \
            _check_number("floors", floors[1], positive=False)
        used, source = (), "given"
    pools = {}
    for co, pool in pool_of.items():
        pools.setdefault(pool, {})[co] = by_cohort[co]
    spreads = {pool: (pooled_spread(members, seed=seed),
                      {"pool_cohorts": sum(len(ml) >= 2 for ml, _ in members.values()),
                       "pool_cages": sum(len(ml) for ml, _ in members.values() if len(ml) >= 2)})
               for pool, members in pools.items()}
    baselines = {}                    # (m, pool) -> the noise baseline, the same for every cohort of m in the pool
    cages, cohorts = [], []
    for co, sub in groups.items():
        ml, st = by_cohort[co]
        j = judge(ml, st, k=k, floors=(phi_ml, phi_st), axes=axes, leave_one_out=leave_one_out)
        for i, key in enumerate(sub):
            gx, gy = j["geometry"][i]
            c = cage_set[key]
            cages.append(CageVerdict(cohort=co, cage=cage_set.design[key].get("cage", key), key=key, n=c.n,
                                     median=ml[i], steepness=st[i], centre_ml=gx["centre"], centre_st=gy["centre"],
                                     radius_ml=gx["radius"], radius_st=gy["radius"],
                                     ml_radius_is_floor=gx["is_floor"], st_radius_is_floor=gy["is_floor"],
                                     distance=float(j["distance"][i]),
                                     verdict="flagged" if j["flagged"][i] else "kept", loo=j["loo"],
                                     data_sha256=_fingerprint(c)))
        pool = pool_of[co]
        spread, counts = spreads[pool]
        key = (len(sub), pool)
        if key not in baselines:
            baselines[key] = noise_baseline(len(sub), k, floors=(phi_ml, phi_st), spread=spread, reps=noise_reps,
                                            seed=seed, axes=axes, leave_one_out=leave_one_out)
        nb = baselines[key]
        noise = (MappingProxyType(nb["floor"]),
                 MappingProxyType({**nb["own_spread"], "spread_by": spread_by, "pool": pool, **counts}))
        cohorts.append(CohortVerdict(cohort=co, k=k, axes=axes, n_cages=len(sub), n_flagged=int(j["flagged"].sum()),
                                     loo=j["loo"], rule=_rule_text(k, axes, j["loo"], source), noise=noise))
    return CleaningReport(cages=tuple(cages), cohorts=tuple(cohorts), floors=(phi_ml, phi_st), floors_source=source,
                          floor_cohorts=tuple(used), min_cages=min_cages, k=k, axes=axes,
                          leave_one_out=leave_one_out, by=by, spread_by=spread_by,
                          definitions=definitions(median_def=median_def, quartiles=quartiles),
                          seed=int(seed), noise_reps=noise_reps, time_unit=cage_set.time_unit)


def drop(cage_set: CohortSet, report: CleaningReport) -> CohortSet:
    """The set without the cages `report` flagged (D-504: removal is this separate call). The report must be the one
    computed on this set (the same keys and data); the returned set's notes name it by its sha256."""
    have = {r.key: r for r in report.cages}
    missing = [k for k in have if k not in cage_set]
    extra = [k for k in cage_set if k not in have]
    changed = [k for k, r in have.items() if k in cage_set and _fingerprint(cage_set[k]) != r.data_sha256]
    if missing or extra or changed:
        raise CohortSpecError("the cleaning report was not computed on this set: "
                              + "; ".join(([f"cages in the report, not in the set: {missing}"] if missing else [])
                                          + ([f"cages in the set, not in the report: {extra}"] if extra else [])
                                          + ([f"cages whose data differ from the report's: {changed}"] if changed
                                             else [])))
    flagged = set(report.flagged)
    keep = [k for k in cage_set if k not in flagged]
    note = (f"cleaned: dropped {len(flagged)} flagged cage(s) {sorted(flagged)} by the cleaning report sha256 "
            f"{report.sha256} (k = {report.k:g}, axes = {report.axes})")
    return CohortSet({k: cage_set[k] for k in keep}, design={k: cage_set.design[k] for k in keep},
                     time_unit=cage_set.time_unit, notes=(*cage_set.notes, note), columns=cage_set.columns)
