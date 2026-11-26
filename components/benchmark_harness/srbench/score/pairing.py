"""Paired, cluster-robust comparison of two methods.

THE STATISTIC IS THE MEAN OF DIFFERENCES, NOT A DIFFERENCE OF MEANS' INTERVALS.

An earlier round of this project compared the two methods' MARGINAL confidence intervals, saw them
overlap, and concluded "statistically indistinguishable". That is the wrong test: the
cohort-to-cohort variance dominates each marginal and CANCELS in the difference, so overlapping
marginals say nothing about whether the difference is resolved. Recomputed properly the same data
gave z = 2.29 and excluded zero -- a real result that had been given away.

CLUSTERING IS ON THE PARAMETER POINT. Several cohorts are drawn from one parameter point, and
their outcomes are correlated; treating them as independent understates the standard error. The
cluster-robust (sandwich) form sums within clusters before squaring, which is what accounts for it.

AND THE COMPARISON IS RESTRICTED TO CELLS WHERE BOTH METHODS CLAIMED. A method that declines more
often would otherwise be compared on an easier subset -- which is the gameable part of a
coverage-only test, and the reason availability is a separate axis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..contract.types import Claim


@dataclass(frozen=True, slots=True)
class PairedResult:
    """A paired comparison and everything needed to read it.

    `n_cells` and `n_clusters` are both mandatory: a gap of +0.08 on 1,309 cells over 70 clusters
    and the same gap on 8 cells over 2 clusters are not the same evidence, and the second supports
    no conclusion at all.
    """

    mean_difference: float | None
    se: float | None
    z: float | None
    ci_lo: float | None
    ci_hi: float | None
    n_cells: int
    n_clusters: int
    quantity: str
    shape_class: str
    cohort_set: str = ""
    aggregation: str = "cell average"

    @property
    def resolved(self) -> bool:
        """Does the interval exclude zero?"""
        return (self.ci_lo is not None and self.ci_hi is not None
                and (self.ci_lo > 0 or self.ci_hi < 0))

    def to_dict(self) -> dict:
        return {
            "quantity": self.quantity, "shape_class": self.shape_class,
            "mean_difference": self.mean_difference, "se": self.se, "z": self.z,
            "ci_lo": self.ci_lo, "ci_hi": self.ci_hi,
            "n_cells": self.n_cells, "n_clusters": self.n_clusters,
            "cohort_set": self.cohort_set, "aggregation": self.aggregation,
            "resolved": self.resolved,
        }

    def format(self) -> str:
        if self.mean_difference is None:
            return f"{self.quantity:<22} no paired cells"
        return (f"{self.quantity:<22} d={self.mean_difference:+.4f} "
                f"[{self.ci_lo:+.4f}, {self.ci_hi:+.4f}] z={self.z:+.2f}  "
                f"cells={self.n_cells} clusters={self.n_clusters}  ({self.aggregation})")


def cluster_robust_mean(values: list[float], clusters: list) -> tuple[float, float, int, int]:
    """(mean, cluster-robust SE, n, n_clusters) for a set of per-cell differences.

    The sandwich form: sum the centred values WITHIN each cluster, square the cluster sums, and
    divide by n^2. Treating cells as independent would divide by n instead, understating the SE
    whenever cohorts from one parameter point behave alike -- which they do.
    """
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), 0, 0
    mean = sum(values) / n
    by_cluster: dict = {}
    for v, c in zip(values, clusters):
        by_cluster.setdefault(c, []).append(v - mean)
    g = len(by_cluster)
    if g < 2:
        return mean, float("nan"), n, g
    ss = sum(sum(vs) ** 2 for vs in by_cluster.values())
    # Small-cluster correction, as used for a sandwich estimator with few clusters.
    correction = g / (g - 1)
    var = correction * ss / (n * n)
    return mean, math.sqrt(var) if var > 0 else 0.0, n, g


def paired_hits(a_sets, b_sets, truths, parameter: str, cluster_of: dict,
                shape_class: str = "closed"):
    """Per-cell (hit_a - hit_b) on cohorts where BOTH methods claimed, plus their clusters.

    The restriction is what makes the comparison a comparison: a method that declines the hard
    cohorts must not be scored on an easier population than its rival.
    """
    a_by = {cs.cohort_id: cs for cs in a_sets}
    b_by = {cs.cohort_id: cs for cs in b_sets}
    diffs, clusters, ids = [], [], []
    for cid in sorted(set(a_by) & set(b_by)):
        truth = truths.get(cid, {}).get(parameter)
        if truth is None:
            continue
        ca, cb = a_by[cid], b_by[cid]
        if not (ca.ok and cb.ok):
            continue
        la, lb = ca.claim_for(parameter), cb.claim_for(parameter)
        if la is None or lb is None:
            continue
        if not (_in_class(la, shape_class) and _in_class(lb, shape_class)):
            continue
        ha, hb = la.contains(truth), lb.contains(truth)
        if ha is None or hb is None:
            continue
        diffs.append(float(ha) - float(hb))
        clusters.append(cluster_of.get(cid, cid))
        ids.append(cid)
    return diffs, clusters, ids


def _in_class(claim: Claim, shape_class: str) -> bool:
    from .claims import _in_class as impl
    return impl(claim, shape_class)


def paired_coverage_gap(a_sets, b_sets, truths, parameter: str, cluster_of: dict,
                        shape_class: str = "closed", cohort_set: str = "",
                        level: float = 0.95) -> PairedResult:
    """The headline comparison: does method A cover more often than method B, on the same cells?"""
    diffs, clusters, _ids = paired_hits(a_sets, b_sets, truths, parameter, cluster_of, shape_class)
    if not diffs:
        return PairedResult(None, None, None, None, None, 0, 0, parameter, shape_class, cohort_set)
    mean, se, n, g = cluster_robust_mean(diffs, clusters)
    zcrit = 1.959963984540054 if abs(level - 0.95) < 1e-9 else _normal_quantile(level)
    if math.isnan(se):
        return PairedResult(mean, None, None, None, None, n, g, parameter, shape_class, cohort_set)
    # se == 0 with a nonzero mean: every cell differs by the same amount. The SIGN is the result --
    # an unsigned inf scored a method that lost on every paired cell as winning (sraging, 2026-10-29).
    z = mean / se if se > 0 else math.copysign(math.inf, mean) if mean else 0.0
    return PairedResult(mean, se, z, mean - zcrit * se, mean + zcrit * se,
                        n, g, parameter, shape_class, cohort_set)


def _normal_quantile(level: float) -> float:
    """Two-sided normal critical value. Bisection on the error function -- no scipy needed here,
    and one fewer dependency in the path that produces the headline number."""
    target = 0.5 * (1.0 + level)
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def paired_cohort_set(a_sets, b_sets) -> set[str]:
    """Cohorts both methods produced a result for. Reported as the denominator."""
    return {cs.cohort_id for cs in a_sets if cs.ok} & {cs.cohort_id for cs in b_sets if cs.ok}


def describe_pairing(a_sets, b_sets) -> dict:
    """The accounting a comparison must print: how many each side had, and how many are shared.

    Reported even when it is uninteresting, because the moment it is only printed when it looks
    bad is the moment nobody checks it.
    """
    a_ok = {cs.cohort_id for cs in a_sets if cs.ok}
    b_ok = {cs.cohort_id for cs in b_sets if cs.ok}
    both = a_ok & b_ok
    return {
        "a_scored": len(a_ok), "b_scored": len(b_ok), "paired": len(both),
        "a_only": len(a_ok - b_ok), "b_only": len(b_ok - a_ok),
        "paired_fraction_of_a": len(both) / len(a_ok) if a_ok else None,
    }
