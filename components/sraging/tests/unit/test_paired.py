"""The paired coverage gate reproduces the tester's own statistic (spec 003, T043; SC-005; research R14).

The tester's `cluster_robust_mean` and `paired_coverage_gap` are loaded from the live tester's srbench/score/pairing.py
(read only; its package-relative imports are neutralised so the functions run standalone) and compared with
sraging.scoring.paired on varied inputs. Plants (constitution V; contracts/acceptance.md): a version that ignores the
clustering, and a gap computed on unpaired cells, must disagree. Also: the gate's verdicts (underpowered below 30 cells
or 10 clusters, the non-inferiority margin), the tester's availability and information axes, and the gate's power.
"""
import math
import os
import random
import types

import pytest

from conftest import not_applicable
from sraging.scoring import paired as P

PAIRING = os.path.join(os.environ["SRAGING_TESTER_ROOT"], "srbench", "score", "pairing.py")   # conftest sets it (0.1.1)


@pytest.fixture(scope="module")
def tester():
    if not os.path.exists(PAIRING):
        not_applicable(f"the tester's pairing.py is not at {PAIRING}")
    src = open(PAIRING).read().replace("from ..contract.types import Claim", "Claim = object")
    import sys
    mod = types.ModuleType("srbench_live_pairing")
    sys.modules[mod.__name__] = mod                      # dataclasses looks its module up while decorating
    exec(compile(src, PAIRING, "exec"), mod.__dict__)
    return mod


def cases(seed=0):
    rng = random.Random(seed)
    for n, g in ((12, 4), (60, 15), (200, 70), (7, 2)):
        vals = [rng.choice([-1.0, 0.0, 0.0, 1.0]) for _ in range(n)]
        clus = [f"k{rng.randrange(g)}" for _ in range(n)]
        yield vals, clus


def test_the_formula_equals_the_testers(tester):
    for vals, clus in cases():
        a, b = P.cluster_robust_mean(vals, clus), tester.cluster_robust_mean(vals, clus)
        assert a[2:] == b[2:] and a[0] == b[0] and (math.isnan(a[1]) and math.isnan(b[1]) or a[1] == b[1])


def test_the_gap_and_its_outcome_follow_the_testers_rule():
    hits_a = {f"c{i}": i % 3 != 0 for i in range(30)}
    hits_b = {f"c{i}": True for i in range(30)}
    hits_b["c5"] = None                                   # no claim in class: not a cell
    clusters = {f"c{i}": f"k{i // 3}" for i in range(30)}
    r = P.paired_gap(hits_a, hits_b, clusters)
    assert r["n_cells"] == 29 and r["n_clusters"] == 10 and r["mean_difference"] < 0
    assert r["outcome"] == ("loses" if r["z"] < -2 else "no difference")


def test_a_gap_that_ignores_the_clustering_disagrees(tester):
    """Constitution V: treating cells as independent (one cluster per cell) understates the SE."""
    vals, clus = next(c for c in cases() if len(c[0]) == 60)
    right = tester.cluster_robust_mean(vals, clus)
    planted = P.cluster_robust_mean(vals, list(range(len(vals))))
    assert planted[1] != right[1]


def test_wilson_interval():
    lo, hi = P.wilson(45, 50)
    assert 0.78 < lo < 0.90 < hi < 0.97
    assert P.wilson(0, 0) == (None, None)


# ------------------------------------------------------------------------- the gap end to end, and the unpaired plant
class _Claim:
    def __init__(self, shape, covers):
        self.shape, self._covers = shape, covers

    def contains(self, truth):
        return self._covers


class _Set:
    def __init__(self, cid, claim, ok=True):
        self.cohort_id, self.ok, self._claim = cid, ok, claim

    def claim_for(self, parameter):
        return self._claim


def _unpaired_gap(hits_a, hits_b, cluster_of):
    """THE PLANT (contracts/acceptance.md): coverage of a on its own cells minus coverage of b on its own cells."""
    a = [float(v) for v in hits_a.values() if v is not None]
    b = [float(v) for v in hits_b.values() if v is not None]
    return {"mean_difference": sum(a) / len(a) - sum(b) / len(b)}


def _declining_case():
    """a declines (None) on exactly the cohorts b misses, so a's own coverage looks perfect while, paired, a and b
    are equal: an unpaired gap reads +0.25, the paired one 0."""
    hits_a, hits_b, clusters = {}, {}, {}
    for i in range(40):
        c = f"c{i}"
        miss = i % 4 == 0
        hits_a[c] = None if miss else True
        hits_b[c] = not miss
        clusters[c] = f"k{i // 4}"
    return hits_a, hits_b, clusters


def _as_sets(hits):
    return [_Set(c, None if h is None else _Claim("closed", h)) for c, h in hits.items()]


def test_the_gap_equals_the_testers_paired_coverage_gap(tester):
    """End to end against the tester's `paired_coverage_gap` (its shape-class helper stood in, the file untouched)."""
    tester._in_class = lambda claim, cls: claim.shape == "closed" if cls == "closed" else \
        claim.shape in ("upper_bound", "lower_bound")
    rng = random.Random(3)
    for hits_a, hits_b, clusters in [_declining_case()] + [
            ({f"c{i}": rng.choice([True, True, True, False, None]) for i in range(n)},
             {f"c{i}": rng.choice([True, True, False, None]) for i in range(n)},
             {f"c{i}": f"k{i % g}" for i in range(n)}) for n, g in ((60, 15), (25, 6))]:
        truths = {c: {"q": 1.0} for c in hits_a}
        t = tester.paired_coverage_gap(_as_sets(hits_a), _as_sets(hits_b), truths, "q", clusters)
        s = P.paired_gap(hits_a, hits_b, clusters)
        assert (s["mean_difference"], s["se"], s["z"], s["ci_lo"], s["ci_hi"], s["n_cells"], s["n_clusters"]) == \
            (t.mean_difference, t.se, t.z, t.ci_lo, t.ci_hi, t.n_cells, t.n_clusters)


def test_a_gap_on_unpaired_cells_is_caught(tester):
    """Constitution V plant: a gap computed on unpaired cells differs from the tester's paired statistic, so the
    equality the test above requires would fail on it."""
    tester._in_class = lambda claim, cls: claim.shape == "closed"
    hits_a, hits_b, clusters = _declining_case()
    truths = {c: {"q": 1.0} for c in hits_a}
    t = tester.paired_coverage_gap(_as_sets(hits_a), _as_sets(hits_b), truths, "q", clusters)
    assert P.paired_gap(hits_a, hits_b, clusters)["mean_difference"] == t.mean_difference == 0.0
    assert _unpaired_gap(hits_a, hits_b, clusters)["mean_difference"] == pytest.approx(0.25)
    assert _unpaired_gap(hits_a, hits_b, clusters)["mean_difference"] != t.mean_difference


def test_a_cell_without_a_cluster_is_refused():
    hits = {"a": True, "b": False, "c": True}
    with pytest.raises(ValueError, match="no cluster"):
        P.paired_gap(hits, hits, {"a": "k0", "b": "k1"})


# ------------------------------------------------------------------------- the gate on one gap
def _gap(mean, se, n=40, g=12):
    z = mean / se
    return {"mean_difference": mean, "se": se, "z": z, "ci_lo": mean - P.ZCRIT_95 * se, "ci_hi": mean + P.ZCRIT_95 * se,
            "n_cells": n, "n_clusters": g, "outcome": "loses" if z < -2 else "wins" if z > 2 else "no difference"}


def test_the_gate_is_underpowered_below_30_cells_or_10_clusters():
    assert P.gate(_gap(0.0, 0.01, n=29))["verdict"] == "underpowered"
    assert P.gate(_gap(0.0, 0.01, g=9))["verdict"] == "underpowered"
    assert P.gate(_gap(-0.5, 0.01, n=3, g=2))["verdict"] == "underpowered"      # a z < -2 on 3 cells is not a measure
    assert P.gate(P.paired_gap({}, {}, {}))["verdict"] == "underpowered"          # "not measured" never passes


def test_the_gate_loses_on_z_and_on_the_non_inferiority_margin():
    assert P.gate(_gap(-0.05, 0.02))["verdict"] == "loses"                       # z = -2.5
    assert P.gate(_gap(-0.02, 0.02))["verdict"] == "inferior"                    # z = -1, CI lower end -0.059
    assert P.gate(_gap(0.0, 0.05 / P.ZCRIT_95))["verdict"] == "inferior"         # CI lower end exactly -0.05
    assert P.gate(_gap(0.0, 0.01))["verdict"] == "no loss"


# ------------------------------------------------------------------------- the tester's other axes
VERDICT = PAIRING.replace("pairing.py", "verdict.py").replace("sr_calibration_tester/", "sr_calibration_tester_jax2/")


def test_the_axes_tolerances_are_the_testers():
    if not os.path.exists(VERDICT):
        not_applicable(f"the tester's verdict.py is not at {VERDICT}")
    import ast
    consts = {t.id: n.value.value for n in ast.parse(open(VERDICT).read()).body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
    assert consts["DECLINE_TOLERANCE"] == P.DECLINE_TOLERANCE and consts["WIDTH_TOLERANCE_DEX"] == P.WIDTH_TOLERANCE_DEX
    assert consts["Z_THRESHOLD"] == P.Z_THRESHOLD


def test_availability_counts_a_failed_fit_as_a_decline():
    a = {"closed": 40, "upper_bound": 5, "lower_bound": 0, "none": 3, "failed": 2}
    b = {"closed": 48, "upper_bound": 2, "lower_bound": 0, "none": 0, "failed": 0}
    av = P.availability(a)
    assert av["decline_rate"] == 5 / 50 and av["failure_rate"] == 2 / 50
    assert P.availability_axis(a, b)["outcome"] == "loses"
    assert P.availability_axis(b, b)["outcome"] == "no difference"


def test_information_loses_on_width_and_on_closed_share():
    mix = {"closed": 40, "upper_bound": 10, "lower_bound": 0, "none": 0, "failed": 0}
    assert P.information_axis([1.0, 1.2], [1.0, 1.0], mix, mix)["outcome"] == "loses"          # +0.1 dex
    assert P.information_axis([1.0, 1.02], [1.0, 1.0], mix, mix)["outcome"] == "no difference"
    fewer = {"closed": 30, "upper_bound": 20, "lower_bound": 0, "none": 0, "failed": 0}
    assert P.information_axis([0.5], [1.0], fewer, mix)["outcome"] == "loses"                  # narrower, fewer closed
    assert P.information_axis([], [1.0], mix, mix)["outcome"] == "not measured"


# ------------------------------------------------------------------------- the gate's power
def test_gate_power_has_its_size_and_power_and_the_clustering_costs_power():
    """z < -2 detects less under clustering; the non-inferiority margin then fails a true no-loss more often (a wide
    CI reaches -0.05), so a clustered panel passes a good combination less often."""
    sizes = [4] * 50                                                             # 200 cells, 50 clusters
    null = P.gate_power(sizes, delta=0.0, p_ref=0.95, reps=300)
    null_clus = P.gate_power(sizes, delta=0.0, p_ref=0.95, icc=1.0, reps=300)
    big = P.gate_power(sizes, delta=-0.15, p_ref=0.95, reps=300)
    clus = P.gate_power(sizes, delta=-0.05, p_ref=0.95, icc=1.0, reps=300)
    flat = P.gate_power(sizes, delta=-0.05, p_ref=0.95, icc=0.0, reps=300)
    assert null["loses"] < 0.08 and big["power"] > 0.95
    assert clus["loses"] < flat["loses"] and null_clus["no loss"] < null["no loss"]
    assert P.gate_power([4] * 5, delta=-0.3, reps=50)["underpowered"] == 1.0     # 20 cells, 5 clusters: never measured
    assert abs(sum(v for k, v in null.items() if k != "power") - 1.0) < 1e-12


def test_gate_power_matches_the_normal_approximation_when_unclustered():
    """At icc 0, P(z < -2) ~ Phi(|delta| / SE - 2), SE^2 = (q - delta^2) / n, q the independent claims' discordance."""
    sizes, d, p = [4] * 50, -0.05, 0.95
    n, pa = sum(sizes), p + d
    q = pa * (1 - p) + (1 - pa) * p
    se = math.sqrt((q - d * d) / n)
    expect = 0.5 * (1 + math.erf((abs(d) / se - 2.0) / math.sqrt(2)))
    got = P.gate_power(sizes, delta=d, p_ref=p, reps=1500, seed=11)["loses"]
    assert abs(got - expect) < 0.06, (got, expect)
