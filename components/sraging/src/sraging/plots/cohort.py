"""Cohort plots (spec 005 US5; FR-019, FR-020): a cohort's KM survival with its band, its binned hazard, its death-time
distribution, and a set's survival bundle. Each takes a result object (a Cohort, a KMResult, a CohortSet), never a
data class with plotting inside, and returns (fig, data)."""
from __future__ import annotations

import numpy as np

from sraging.cohort import Cohort
from sraging.errors import UnsupportedOptionError
from sraging.plots.style import axes, finish, lab_style, result, sha
from sraging.survival import KMResult, km

HAZARD_METHODS = ("binned",)


def km_steps(k: KMResult, values=None):
    """(x, y) of the KM's step vertices: (start, 1), each death time, carried to the last observed age."""
    v = k.S if values is None else values
    x = np.concatenate([[k.start], k.times])
    y = np.concatenate([[1.0], v])
    if k.last_age > x[-1]:
        x, y = np.append(x, k.last_age), np.append(y, y[-1])
    return x, y


def _km_of(obj) -> KMResult:
    if isinstance(obj, KMResult):
        return obj
    if isinstance(obj, Cohort):
        return km(obj)
    raise UnsupportedOptionError("km_or_cohort", f"survival takes a KMResult or a Cohort, got {type(obj).__name__}")


def _unit_label(unit):
    return f"age ({unit})" if unit and unit != "input units" else "age"


def survival(km_or_cohort, *, ax=None):
    """The KM step function with its band (KMResult.band, named in data["band"])."""
    k = _km_of(km_or_cohort)
    with lab_style():
        fig, (a,) = axes(ax)
        x, y = km_steps(k)
        _, lo = km_steps(k, k.S_lo)
        _, hi = km_steps(k, k.S_hi)
        a.fill_between(x, lo, hi, step="post", alpha=0.25, color="C0", linewidth=0, gid="band")
        a.step(x, lo, where="post", color="C0", linewidth=0.8, alpha=0.6, gid="S_lo")
        a.step(x, hi, where="post", color="C0", linewidth=0.8, alpha=0.6, gid="S_hi")
        a.step(x, y, where="post", color="C0", linewidth=2.0, gid="S")
        a.set_ylim(-0.02, 1.02)
        finish(a, xlabel=_unit_label(k.time_unit), ylabel="survival")
    return result(fig, {"S": (x, y), "S_lo": (x, lo), "S_hi": (x, hi)},
                  {"input": "KMResult" if isinstance(km_or_cohort, KMResult) else "Cohort", "n": k.n,
                   "data_sha256": sha(k.ages, k.events), "condition": k.condition},
                  band=k.band, level=k.level)


def binned_hazard(cohort: Cohort, edges) -> tuple:
    """(midpoints, hazard, deaths, exposure) per bin [a, b): deaths over the person-time spent in it; bins with no
    exposure are left out."""
    a, e = np.asarray(cohort.ages, float), np.asarray(cohort.events, int)
    edges = np.asarray(edges, float)
    d = np.histogram(a[e == 1], bins=edges)[0]
    expo = np.array([np.sum(np.clip(np.minimum(a, b) - lo, 0.0, b - lo)) for lo, b in zip(edges[:-1], edges[1:])])
    keep = expo > 0
    mid = (edges[:-1] + edges[1:]) / 2
    return mid[keep], (d / np.where(keep, expo, 1.0))[keep], d[keep], expo[keep]


def _edges(cohort, bins):
    if isinstance(bins, (int, np.integer)) and not isinstance(bins, bool):
        if bins < 1:
            raise UnsupportedOptionError("bins", f"bins must be >= 1, got {bins!r}")
        return np.linspace(0.0, float(np.max(cohort.ages)) * (1 + 1e-9), int(bins) + 1)
    edges = np.asarray(bins, float)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0) or not np.all(np.isfinite(edges)):
        raise UnsupportedOptionError("bins", "bins is a count or increasing finite edges")
    return edges


def hazard(cohort, *, method="binned", bins, ax=None):
    """The binned hazard (deaths / person-time per bin), on a log axis."""
    if method not in HAZARD_METHODS:
        raise UnsupportedOptionError("method", f"method is one of {HAZARD_METHODS}, got {method!r}")
    edges = _edges(cohort, bins)
    mid, h, d, expo = binned_hazard(cohort, edges)
    with lab_style():
        fig, (a,) = axes(ax)
        pos = h > 0
        a.plot(mid, h, "o-", color="C1", linewidth=1.5, markersize=5, gid="hazard")
        finish(a, xlabel=_unit_label(cohort.time_unit), ylabel="hazard (per unit time)", log_y=bool(pos.any()))
    return result(fig, {"hazard": (mid, h)}, {"input": "Cohort", "n": cohort.n,
                                               "data_sha256": sha(cohort.ages, cohort.events)},
                  method=method, edges=edges, deaths=d, exposure=expo)


def death_times(cohort, *, bins, ax=None):
    """The distribution of the death ages (censored rows left out), as counts per bin."""
    edges = _edges(cohort, bins)
    a, e = np.asarray(cohort.ages, float), np.asarray(cohort.events, int)
    counts = np.histogram(a[e == 1], bins=edges)[0].astype(float)
    with lab_style():
        fig, (ax_,) = axes(ax)
        ax_.stairs(counts, edges, fill=True, alpha=0.6, color="C2", gid="death_times")
        finish(ax_, xlabel=_unit_label(cohort.time_unit), ylabel="deaths")
    return result(fig, {"death_times": (edges, counts)}, {"input": "Cohort", "n": cohort.n,
                                                           "data_sha256": sha(a, e)},
                  n_censored=cohort.n_censored)


def set_survival(cohort_set, *, ax=None):
    """Each member's KM step function, one line per key (a legend up to 6 members)."""
    from sraging.cohorts import CohortSet
    if not isinstance(cohort_set, CohortSet):
        raise UnsupportedOptionError("cohort_set", f"set_survival takes a CohortSet, got {type(cohort_set).__name__}")
    arrays = {}
    with lab_style():
        fig, (a,) = axes(ax)
        for i, key in enumerate(cohort_set):
            x, y = km_steps(km(cohort_set[key]))
            a.step(x, y, where="post", linewidth=1.5, color=f"C{i % 10}", label=str(key), gid=str(key))
            arrays[str(key)] = (x, y)
        a.set_ylim(-0.02, 1.02)
        finish(a, xlabel=_unit_label(cohort_set.time_unit), ylabel="survival", legend=0 < len(cohort_set) <= 6)
    return result(fig, arrays, {"input": "CohortSet", "members": list(map(str, cohort_set)),
                                "time_unit": cohort_set.time_unit})
