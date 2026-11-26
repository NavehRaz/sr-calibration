"""Cleaning plots (spec 005 US5; FR-019): one cohort's cages in the (median, steepness) plane with the ellipse each
cage was judged by (leave-one-out: its peers' centre and radii), the flagged cages marked; and the flagged count over k.
They take a CleaningReport or a cage set, never recompute a verdict silently."""
from __future__ import annotations

import numpy as np

from sraging.errors import UnsupportedOptionError
from sraging.plots.style import axes, finish, lab_style, result

THETA = np.linspace(0.0, 2 * np.pi, 181)


def cleaning(report, *, cohort, ax=None):
    """The cohort's cages (median, steepness), flagged ones marked, each cage's judging ellipse (an interval on the
    median when axes="median"; none when the radius is infinite: every peer tied, no floor)."""
    rows = [r for r in report.cages if r.cohort == cohort]
    if not rows:
        raise UnsupportedOptionError("cohort", f"no cohort {cohort!r} in the report; cohorts: "
                                               f"{sorted({str(r.cohort) for r in report.cages})}")
    ml = np.array([r.median for r in rows])
    st = np.array([r.steepness for r in rows])
    flagged = [r for r in rows if r.verdict == "flagged"]
    arrays = {}
    with lab_style():
        fig, (a,) = axes(ax)
        for r in rows:
            if report.axes == "both" and np.isfinite(r.radius_ml) and np.isfinite(r.radius_st):
                ex, ey = r.centre_ml + r.radius_ml * np.cos(THETA), r.centre_st + r.radius_st * np.sin(THETA)
                a.plot(ex, ey, color="0.6", linewidth=0.8, gid=f"ellipse:{r.cage}")
                arrays[f"ellipse:{r.cage}"] = (ex, ey)
            elif report.axes == "median" and np.isfinite(r.radius_ml):
                ex = np.array([r.centre_ml - r.radius_ml, r.centre_ml + r.radius_ml])
                ey = np.full(2, r.steepness if np.isfinite(r.steepness) else 0.0)
                a.plot(ex, ey, color="0.6", linewidth=0.8, gid=f"interval:{r.cage}")
                arrays[f"interval:{r.cage}"] = (ex, ey)
        a.plot(ml, st, "o", color="C0", markersize=7, linestyle="none", gid="cages", label="kept")
        fx, fy = np.array([r.median for r in flagged]), np.array([r.steepness for r in flagged])
        a.plot(fx, fy, "x", color="C3", markersize=11, markeredgewidth=2.5, linestyle="none", gid="flagged",
               label="flagged")
        arrays.update(cages=(ml, st), flagged=(fx, fy))
        finish(a, xlabel="median lifetime", ylabel="steepness", legend=True)
    return result(fig, arrays, {"report_sha256": report.sha256, "cohort": cohort, "k": report.k, "axes": report.axes,
                                "definitions": dict(report.definitions)})


def k_sweep(cage_set, *, by, ks, floors, ax=None):
    """The number of cages the rule flags at each k (judge on the cage set's scores, `floors` given or None to pool),
    with leave-one-out and both axes, as clean's defaults."""
    from sraging import cleaning as CL
    ks = np.asarray(ks, float)
    if ks.ndim != 1 or ks.size < 1 or not np.all(np.isfinite(ks)) or not np.all(ks > 0):
        raise UnsupportedOptionError("ks", "ks are finite numbers > 0")
    scores = CL.cage_scores(cage_set)
    groups = cage_set.by(by)
    by_cohort = {co: ([scores[k][0] for k in sub], [scores[k][1] for k in sub]) for co, sub in groups.items()}
    if floors is None:
        phi = CL.pooled_floors(by_cohort)[:2]
    else:
        phi = (float(floors[0]), float(floors[1]))
    counts = np.array([sum(int(CL.judge(ml, st, k=float(k), floors=phi)["flagged"].sum())
                           for ml, st in by_cohort.values()) for k in ks], float)
    with lab_style():
        fig, (a,) = axes(ax)
        a.plot(ks, counts, "o-", color="C3", linewidth=2.0, gid="flagged")
        finish(a, xlabel="k (robust sigmas)", ylabel="cages flagged")
    return result(fig, {"flagged": (ks, counts)}, {"by": by, "floors": list(phi), "n_cages": len(cage_set),
                                                   "definitions": CL.definitions(median_def="first-crossing",
                                                                                 quartiles="nearest")})
