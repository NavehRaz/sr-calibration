"""One Kaplan-Meier estimator and the summary statistics, each named (spec 005 US3; FR-010 to FR-012; research R2, R3;
data-model "KMResult", "SummaryStatistics"; register D-501).

    k = km(cohort)                          # or km((ages, events)); condition_on=t0 for the KM given survival to t0
    s = summary(k)                          # quartiles="first-crossing" by default; "nearest" is SRK Dataset's
    s.median.value, s.median.definition, s.median.ci

- **The KM** (R2) is built on the distinct death times u, with r(u) = #{t >= u} (a censoring at u is at risk at u, so
  deaths come first at a tie) and d(u); np.unique and searchsorted make it O(n log n) and independent of row order.
  Ŝ is computed as lifelines 0.30.3 computes it, exp(Σ [log(r - d) - log r]), so it equals lifelines' (and SRK
  Dataset's, the cleaning records') bit for bit. A crossing at an exact rational level (Ŝ = 1/2 in exact arithmetic)
  follows the computed Ŝ, which can sit one ulp either side, as lifelines' median does.
- **The band** is the 95 % log(-log) Greenwood band, exp(-exp(log(-log Ŝ) ∓ z √V / log Ŝ)), V = Σ d / (r (r - d)),
  with lifelines' conventions: a term with r = d adds 0 to V, so the band is [0, 0] where Ŝ = 0, and [1, 1] before the
  first death. `population.kaplan_meier` keeps its linear band (spec 004), and the compat KMs are unchanged (FR-012,
  D-007).
- **The statistics** (R3): the median is first-crossing (the first age where Ŝ <= 0.5), with its CI where the band's
  limits first reach 0.5; quartile ages are first-crossing by default or `nearest` (the KM-timeline age minimising
  |Ŝ - q|, SRK Dataset's getSteepness); steepness is median / (t(0.25) - t(0.75)), computed as SRK computes it; the
  mean and CV come from the KM step function up to τ (the last observed age by default) and are labelled restricted
  when Ŝ(τ) > 0. Conditional versions use the KM of the individuals observed beyond t0, which is Ŝ(t) / Ŝ(t0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from sraging.cohort import Cohort
from sraging.errors import CohortDataError, UnsupportedOptionError

BAND = "log(-log) Greenwood"
QUARTILES = ("first-crossing", "nearest")


def _ro(a):
    a = np.array(a)
    a.setflags(write=False)
    return a


@dataclass(frozen=True, eq=False)
class KMResult:
    """The KM's step vertices (`times`, `S`, with `at_risk` and `deaths` there), its named band, and its inputs.
    `timeline` is lifelines' ({start} and every observed age), where `nearest` quartiles are taken; `condition` is None
    or (t0, Ŝ(t0)) of the unconditioned KM. `ages`, `events` are the data before any conditioning."""

    times: np.ndarray
    S: np.ndarray
    at_risk: np.ndarray
    deaths: np.ndarray
    S_lo: np.ndarray
    S_hi: np.ndarray
    band: str
    level: float
    z: float
    n: int
    n_deaths: int
    n_censored: int
    time_unit: str
    condition: tuple | None
    timeline: np.ndarray
    start: float
    last_age: float
    ages: np.ndarray = field(repr=False)
    events: np.ndarray = field(repr=False)

    def _step(self, values, ages, before):
        ages = np.asarray(ages, float)
        if self.times.size == 0:
            return np.full(ages.shape, before)
        j = np.searchsorted(self.times, ages, side="right") - 1
        return np.where(j >= 0, values[np.maximum(j, 0)], before)

    def S_at(self, ages) -> np.ndarray:
        """Ŝ at `ages`, stepping to the right (1 before the first death)."""
        return self._step(self.S, ages, 1.0)

    def band_at(self, ages) -> tuple:
        """(lower, upper) of the band at `ages`, stepping to the right ([1, 1] before the first death)."""
        return self._step(self.S_lo, ages, 1.0), self._step(self.S_hi, ages, 1.0)


def _data(data):
    """(ages, events, time_unit) from a Cohort or an (ages, events) pair, refused by name where km cannot take it."""
    if isinstance(data, Cohort):
        if data.entry is not None:
            raise UnsupportedOptionError("entry ages", "km does not take left truncation (entry ages) in spec 005")
        if data.window is not None:
            raise UnsupportedOptionError("window", "km does not take a fit window in spec 005")
        return np.asarray(data.ages, float), np.asarray(data.events, int), data.time_unit
    if isinstance(data, (tuple, list)) and len(data) == 2:
        try:
            a, e = np.asarray(data[0], float), np.asarray(data[1], float)
        except (TypeError, ValueError) as ex:
            raise CohortDataError(f"km: ages and events must be numeric: {ex}") from None
        if a.ndim != 1 or a.size < 1 or e.shape != a.shape:
            raise CohortDataError(f"km: ages and events must be 1-d of one length >= 1, got {a.shape} and {e.shape}")
        if not np.all(np.isfinite(a)) or np.any(a < 0):
            raise CohortDataError("km: every age must be finite and not negative")
        if not np.all(np.isin(e, (0.0, 1.0))):
            raise CohortDataError("km: every event must be 0 or 1")
        return a, e.astype(int), "input units"
    raise TypeError(f"km takes a sraging.Cohort or an (ages, events) pair, got {type(data).__name__}")


def _core(a, e):
    """(times, r, d, Ŝ, V) on the distinct death times, deaths before tied censorings, in lifelines' arithmetic."""
    times, d = np.unique(a[e == 1], return_counts=True)
    r = (a.size - np.searchsorted(np.sort(a), times, side="left")).astype(float)
    d = d.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.exp(np.cumsum(np.log(r - d) - np.log(r)))
        V = np.cumsum(np.where(r > d, d / (r * (r - d)), 0.0))
    return times, r, d, S, V


def _band(S, V, z):
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.log(S)
        lo = np.exp(-np.exp(np.log(-v) - z * np.sqrt(V) / v))
        hi = np.exp(-np.exp(np.log(-v) + z * np.sqrt(V) / v))
    return np.where(np.isnan(lo), 1.0, lo), np.where(np.isnan(hi), 1.0, hi)


def _km(a, e, unit, condition_on, level) -> KMResult:
    if isinstance(level, bool) or not isinstance(level, (int, float)) or not 0 < level < 1:
        raise UnsupportedOptionError("level", f"level must be a number in (0, 1), got {level!r}")
    start, cond, sa, se = 0.0, None, a, e
    if condition_on is not None:
        t0 = condition_on
        if isinstance(t0, bool) or not isinstance(t0, (int, float, np.number)) or not math.isfinite(t0) or t0 < 0:
            raise UnsupportedOptionError("condition_on", f"condition_on must be a finite age >= 0, got {t0!r}")
        t0 = float(t0)
        keep = a > t0
        if not keep.any():
            raise UnsupportedOptionError("condition_on", f"condition_on={t0!r}: no individual is observed beyond it")
        times, _, _, S, _ = _core(a, e)
        j = np.searchsorted(times, t0, side="right") - 1
        cond = (t0, float(S[j]) if j >= 0 else 1.0)
        start, sa, se = t0, a[keep], e[keep]
    times, r, d, S, V = _core(sa, se)
    z = float(norm.ppf(1 - (1 - level) / 2))
    lo, hi = _band(S, V, z)
    return KMResult(times=_ro(times), S=_ro(S), at_risk=_ro(r.astype(int)), deaths=_ro(d.astype(int)),
                    S_lo=_ro(lo), S_hi=_ro(hi), band=BAND, level=float(level), z=z, n=int(sa.size),
                    n_deaths=int(np.sum(se == 1)), n_censored=int(np.sum(se == 0)), time_unit=unit, condition=cond,
                    timeline=_ro(np.unique(np.concatenate([[start], sa]))), start=start, last_age=float(sa.max()),
                    ages=_ro(a), events=_ro(e))


def km(data, *, condition_on=None, level=0.95) -> KMResult:
    """The KM of a Cohort or an (ages, events) pair, with its `level` log(-log) Greenwood band (R2). With
    `condition_on=t0`, the KM of the individuals observed beyond t0 (Ŝ(t) / Ŝ(t0)), and `condition` = (t0, Ŝ(t0))."""
    a, e, unit = _data(data)
    return _km(a, e, unit, condition_on, level)


@dataclass(frozen=True)
class Stat:
    """A statistic's value, its definition (never implied) and its CI where it has one."""

    value: float
    definition: str
    ci: tuple | None = None


@dataclass(frozen=True)
class SummaryStatistics:
    """data-model "SummaryStatistics": each entry a Stat; `quartiles` the quartile definition; `tau` the horizon of
    the mean and CV, `restricted` whether Ŝ(τ) > 0; `condition` as the KM's."""

    median: Stat
    q25_age: Stat
    q75_age: Stat
    steepness: Stat
    restricted_mean: Stat
    cv: Stat
    quartiles: str
    tau: float
    restricted: bool
    condition: tuple | None
    n: int
    time_unit: str


def _first(times, values, q) -> float:
    idx = np.flatnonzero(values <= q)
    return float(times[idx[0]]) if idx.size else math.inf


def quartile_age(k: KMResult, q: float, *, quartiles: str) -> float:
    """The age at which Ŝ reaches q: first-crossing (the first age with Ŝ <= q; inf if never) or nearest (the
    KM-timeline age minimising |Ŝ - q|, the first on a tie, as SRK Dataset's getSteepness)."""
    if quartiles == "first-crossing":
        return _first(k.times, k.S, q)
    if quartiles == "nearest":
        return float(k.timeline[np.argmin(np.abs(k.S_at(k.timeline) - q))])
    raise UnsupportedOptionError("quartiles", f"quartiles is one of {QUARTILES}, got {quartiles!r}")


def steepness(median: float, t75: float, t25: float) -> float:
    """SRK's IQR steepness, -median / (t(0.75) - t(0.25)), in its arithmetic: non-finite when the IQR is zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(-median) / np.float64(t75 - t25))


def _moments(k: KMResult, tau: float) -> tuple:
    """(E min(T, τ), E min(T, τ)^2) given T > start, from the KM step function."""
    inner = k.times[(k.times > k.start) & (k.times < tau)]
    knots = np.concatenate([[k.start], inner, [tau]])
    vals = k.S_at(knots[:-1])
    a, b = knots[:-1], knots[1:]
    return k.start + float(np.sum(vals * (b - a))), k.start ** 2 + float(np.sum(vals * (b * b - a * a)))


def summary(data, *, quartiles="first-crossing", horizon=None, condition_on=None) -> SummaryStatistics:
    """The summary statistics of a Cohort or a KMResult, each named (FR-011; R3). `quartiles` "first-crossing"
    (default) or "nearest"; `horizon` τ for the mean and CV (default: the last observed age); `condition_on` t0."""
    if quartiles not in QUARTILES:
        raise UnsupportedOptionError("quartiles", f"quartiles is one of {QUARTILES}, got {quartiles!r}")
    if isinstance(data, KMResult):
        k = data
        if condition_on is not None:
            if k.condition is not None:
                raise UnsupportedOptionError("condition_on", f"condition_on={condition_on!r}: the KM is already "
                                                             f"conditioned on t0 = {k.condition[0]!r}; condition the "
                                                             f"raw data once")
            k = _km(np.asarray(k.ages), np.asarray(k.events), k.time_unit, condition_on, k.level)
    else:
        k = km(data, condition_on=condition_on)
    cond = "" if k.condition is None else \
        f"; conditional on survival to {k.condition[0]:g} (Ŝ({k.condition[0]:g}) = {k.condition[1]:.6g})"
    med = _first(k.times, k.S, 0.5)
    ci = (_first(k.times, k.S_lo, 0.5), _first(k.times, k.S_hi, 0.5))
    t75, t25 = quartile_age(k, 0.75, quartiles=quartiles), quartile_age(k, 0.25, quartiles=quartiles)
    how = {"first-crossing": "the first age where Ŝ <= {q}",
           "nearest": "the KM-timeline age minimising |Ŝ - {q}| (SRK Dataset's getSteepness)"}[quartiles]
    tau = k.last_age if horizon is None else horizon
    if isinstance(tau, bool) or not isinstance(tau, (int, float, np.number)) or not math.isfinite(tau) \
            or not tau > k.start:
        raise UnsupportedOptionError("horizon", f"horizon must be a finite age above {k.start!r}, got {horizon!r}")
    tau = float(tau)
    s_tau = float(k.S_at([tau])[0])
    restricted = s_tau > 0
    m1, m2 = _moments(k, tau)
    sd = math.sqrt(max(m2 - m1 * m1, 0.0))
    if restricted:
        mean_def = (f"restricted mean to τ = {tau:g}: start + the area under the KM step function up to τ "
                    f"(Ŝ(τ) = {s_tau:.6g} > 0, so a lower bound on the mean)")
        cv_def = f"restricted CV to τ = {tau:g}: the SD of min(T, τ) over its mean, from the KM step function"
    else:
        mean_def = f"mean: start + the area under the KM step function (Ŝ reaches 0 at τ = {tau:g})"
        cv_def = "CV: the KM step function's SD over its mean"
    return SummaryStatistics(
        median=Stat(med, "first-crossing: the first age where Ŝ <= 0.5 (lifelines' median); CI where the "
                         f"{k.band} band's lower and upper limits first reach 0.5{cond}", ci),
        q25_age=Stat(t25, f"{quartiles} quartile age at Ŝ = 0.25: {how.format(q=0.25)}{cond}"),
        q75_age=Stat(t75, f"{quartiles} quartile age at Ŝ = 0.75: {how.format(q=0.75)}{cond}"),
        steepness=Stat(steepness(med, t75, t25), f"IQR steepness: median / (t(0.25) - t(0.75)), quartile ages "
                                                 f"{quartiles}, as SRK's -median / (t75 - t25){cond}"),
        restricted_mean=Stat(m1, mean_def + cond),
        cv=Stat(sd / m1 if m1 > 0 else math.nan, cv_def + cond),
        quartiles=quartiles, tau=tau, restricted=restricted, condition=k.condition, n=k.n, time_unit=k.time_unit)
