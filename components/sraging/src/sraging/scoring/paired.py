"""The paired, cluster-robust coverage gap (spec 003, T043; SC-005; research R14): the tester's statistic
(srbench/score/pairing.py `cluster_robust_mean`, `paired_coverage_gap`), for comparing a combination's claims with
the same fits' eta or epsilon claims.

The statistic is the mean over cells of hit_a - hit_b, where a cell is a cohort on which both claims exist in the
shape class (both fits ok); the SE is the cluster-robust sandwich over parameter points (the library's cluster_key)
with the g/(g-1) small-cluster correction. "Loses" is z < -2 (srbench/score/verdict.py:33, 97-106).
tests/unit/test_paired.py requires the numbers to equal the tester's own function's.

The gate on one gap (`gate`; sr-manager gate 5, sr-naveh implementation review 4): below MIN_CELLS paired cells or
MIN_CLUSTERS clusters it is "underpowered" (never a pass); a loss is z < -2 OR a 95% CI lower end <= -NI_MARGIN (the
non-inferiority margin). Beside it, the tester's availability and information axes (verdict.py:137-178, its
tolerances), reported, and the gate's power on a panel's cluster structure (`gate_power`), computed before any fit.
"""
from __future__ import annotations

import math
import random

Z_THRESHOLD = 2.0
ZCRIT_95 = 1.959963984540054
MIN_CELLS, MIN_CLUSTERS = 30, 10        # per gated (arm, quantity): below either, "underpowered"
NI_MARGIN = 0.05                        # the sr-naveh review's margin, an option of `gate`; for SC-005 Naveh dropped
#                                         it (2026-10-05: tests/acceptance/decisions_003.toml [coverage_gate])
DECLINE_TOLERANCE = 0.02                # srbench/score/verdict.py:37 (availability)
WIDTH_TOLERANCE_DEX = 0.05              # srbench/score/verdict.py:38 (information)
SHAPES = ("closed", "upper_bound", "lower_bound", "none", "failed")


def cluster_robust_mean(values, clusters):
    """(mean, cluster-robust SE, n, n_clusters) -- the tester's formula."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), 0, 0
    mean = sum(values) / n
    by = {}
    for v, c in zip(values, clusters):
        by.setdefault(c, []).append(v - mean)
    g = len(by)
    if g < 2:
        return mean, float("nan"), n, g
    ss = sum(sum(vs) ** 2 for vs in by.values())
    var = (g / (g - 1)) * ss / (n * n)
    return mean, math.sqrt(var) if var > 0 else 0.0, n, g


def paired_gap(hits_a: dict, hits_b: dict, cluster_of: dict) -> dict:
    """`hits_a`, `hits_b`: {cohort_id: True/False/None} (None: no claim in the shape class, or the fit failed).
    Returns {mean_difference, se, z, ci_lo, ci_hi, n_cells, n_clusters, outcome}. Cells are the cohorts where BOTH
    sides hold a scorable claim; a cell without a cluster in `cluster_of` is refused (the tester's fallback, one
    cluster per cohort, would understate the SE silently)."""
    cells = sorted(c for c in set(hits_a) & set(hits_b) if hits_a[c] is not None and hits_b[c] is not None)
    unclustered = [c for c in cells if c not in cluster_of]
    if unclustered:
        raise ValueError(f"paired_gap: no cluster for {len(unclustered)} cell(s), e.g. {unclustered[:3]}")
    diffs = [float(hits_a[c]) - float(hits_b[c]) for c in cells]
    clusters = [cluster_of[c] for c in cells]
    if not diffs:
        return {"mean_difference": None, "se": None, "z": None, "ci_lo": None, "ci_hi": None, "n_cells": 0,
                "n_clusters": 0, "outcome": "not measured"}
    mean, se, n, g = cluster_robust_mean(diffs, clusters)
    if math.isnan(se):
        return {"mean_difference": mean, "se": None, "z": None, "ci_lo": None, "ci_hi": None, "n_cells": n,
                "n_clusters": g, "outcome": "not measured"}
    z = mean / se if se > 0 else (float("inf") if mean > 0 else float("-inf") if mean < 0 else 0.0)
    outcome = "wins" if z > Z_THRESHOLD else "loses" if z < -Z_THRESHOLD else "no difference"
    return {"mean_difference": mean, "se": se, "z": z, "ci_lo": mean - ZCRIT_95 * se, "ci_hi": mean + ZCRIT_95 * se,
            "n_cells": n, "n_clusters": g, "outcome": outcome}


def gate(gap: dict, *, min_cells=MIN_CELLS, min_clusters=MIN_CLUSTERS, margin=NI_MARGIN) -> dict:
    """SC-005's verdict on one paired gap: {verdict, reasons}. "underpowered" below `min_cells` cells or
    `min_clusters` clusters, or with no SE (never a pass); else "loses" (z < -2), "inferior" (95% CI lower end
    <= -margin; skipped when `margin` is None, Naveh's Clarification rule being z < -2 alone), or "no loss"."""
    why = []
    if gap["n_cells"] < min_cells:
        why.append(f"{gap['n_cells']} paired cells < {min_cells}")
    if gap["n_clusters"] < min_clusters:
        why.append(f"{gap['n_clusters']} clusters < {min_clusters}")
    if gap["se"] is None or gap["ci_lo"] is None:
        why.append("the SE is not measured")
    if why:
        return {"verdict": "underpowered", "reasons": why}
    if gap["outcome"] == "loses":
        return {"verdict": "loses", "reasons": [f"z = {gap['z']:.3f} < -{Z_THRESHOLD:g}"]}
    if margin is not None and gap["ci_lo"] <= -margin:
        return {"verdict": "inferior", "reasons": [f"95% CI lower end {gap['ci_lo']:+.4f} <= -{margin:g}"]}
    return {"verdict": "no loss", "reasons": []}


def wilson(k: int, n: int, z: float = ZCRIT_95):
    """The Wilson score interval for k successes in n (absolute coverage, reported beside the paired gate)."""
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


# ------------------------------------------------------------------------------------- the tester's other axes
def availability(mix: dict) -> dict:
    """Decline and failure rates from a shape mix {closed, upper_bound, lower_bound, none, failed} (verdict.py's
    `availability`). A fit that raised in the frozen code is a decline here (sr-manager gate 5), so decline_rate
    counts `none` and `failed`; failure_rate is reported apart."""
    total = sum(mix.get(s, 0) for s in SHAPES)
    declined = mix.get("none", 0) + mix.get("failed", 0)
    return {"total": total, "answered": total - declined, "declined": declined, "failed": mix.get("failed", 0),
            "decline_rate": declined / total if total else None,
            "failure_rate": mix.get("failed", 0) / total if total else None}


def availability_axis(mix_a: dict, mix_b: dict, tol=DECLINE_TOLERANCE) -> dict:
    """verdict.py `availability_axis`: d = decline_rate(a) - decline_rate(b); loses above `tol`."""
    a, b = availability(mix_a), availability(mix_b)
    if a["decline_rate"] is None or b["decline_rate"] is None:
        return {"outcome": "not measured", "statistic": None, "a": a, "b": b}
    d = a["decline_rate"] - b["decline_rate"]
    return {"outcome": "loses" if d > tol else "wins" if d < -tol else "no difference", "statistic": d, "a": a, "b": b}


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


def information_axis(widths_a, widths_b, mix_a: dict, mix_b: dict, tol=WIDTH_TOLERANCE_DEX,
                     share_tol=DECLINE_TOLERANCE) -> dict:
    """verdict.py `information_axis`: the median width (dex) of CLOSED claims, a minus b, loses above `tol`; also
    loses when a's closed share falls more than `share_tol` below b's (one-sided bounds answer half the question)."""
    ma, mb = _median(widths_a), _median(widths_b)
    share = {k: m.get("closed", 0) / max(sum(m.get(s, 0) for s in SHAPES), 1)
             for k, m in (("a", mix_a), ("b", mix_b))}
    if ma is None or mb is None:
        return {"outcome": "not measured", "statistic": None, "median_dex": [ma, mb], "closed_share": share}
    d = ma - mb
    out = "loses" if d > tol else "wins" if d < -tol else "no difference"
    if share["a"] < share["b"] - share_tol:
        out = "loses"
    return {"outcome": out, "statistic": d, "median_dex": [ma, mb], "closed_share": share,
            "n_widths": [len(widths_a), len(widths_b)]}


# ------------------------------------------------------------------------------------- the gate's power
def gate_power(cluster_sizes, *, delta: float, p_ref: float = 0.95, icc: float = 0.0, concordance: float = 0.0,
               reps: int = 400, seed: int = 20261005, **gate_kw) -> dict:
    """Monte Carlo distribution of `gate`'s verdict on a panel's cluster structure, every cohort a cell (an upper
    bound: real cells are only those where both claims are closed). Model: the reference claim covers with p_ref,
    the combination's with p_ref + delta; within a cell P(both cover) = (1 - c) p_a p_ref + c min(p_a, p_ref) for
    `concordance` c (0: independent, the most discordant; 1: one's misses inside the other's; claims from one fit
    tend to co-cover); within a cluster, with probability `icc` every cell copies one draw (the cells' outcomes then
    correlate by icc). Returns {verdict: fraction, ...}; "power" is the fraction that reads a loss (loses or
    inferior); at delta = 0 it is the chance of failing a combination that loses nothing."""
    p_a = p_ref + delta
    if not (0.0 <= p_a <= 1.0 and 0.0 < p_ref < 1.0 and 0.0 <= icc <= 1.0 and 0.0 <= concordance <= 1.0):
        raise ValueError(f"gate_power: p_ref {p_ref}, p_ref + delta {p_a}, icc {icc}, concordance {concordance} out "
                         f"of range")
    p11 = (1.0 - concordance) * p_a * p_ref + concordance * min(p_a, p_ref)
    a_if_b, a_if_not_b = p11 / p_ref, (p_a - p11) / (1.0 - p_ref)
    rng = random.Random(seed)

    def draw():
        b = rng.random() < p_ref
        return rng.random() < (a_if_b if b else a_if_not_b), b
    count = {"underpowered": 0, "loses": 0, "inferior": 0, "no loss": 0}
    for _ in range(reps):
        ha, hb, cl = {}, {}, {}
        for g, m in enumerate(cluster_sizes):
            shared = rng.random() < icc
            one = draw()
            for j in range(m):
                c = f"g{g}c{j}"
                a, b = one if shared else draw()
                ha[c], hb[c], cl[c] = a, b, g
        count[gate(paired_gap(ha, hb, cl), **gate_kw)["verdict"]] += 1
    out = {k: v / reps for k, v in count.items()}
    out["power"] = out["loses"] + out["inferior"]
    return out
