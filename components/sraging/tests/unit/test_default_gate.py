"""The default's gate rules PR-1..PR-6 (spec 003g "Pre-registered rules", Naveh's OK 2026-10-22; research R2;
contracts/gate-and-panel.md; tasks T006). Every rule fails its planted case and passes its twin (constitution V)."""
import math

import pytest

from sraging import default_gate as G


def wrow(cid, W, *, valid=True, informative=True, fresh=True):
    return {"cohort_id": cid, "valid": valid, "informative": informative,
            "witness": None if W is None else {"W": W, "fresh": fresh}}


# --------------------------------------------------------------------------------------------------------- PR-1
def test_pr1_one_valid_cohort_past_the_threshold_fails_its_stratum_and_its_twin_passes():
    fail = G.witness_verdict("S", [wrow("a", 0.2), wrow("b", 1.93)])
    ok = G.witness_verdict("S", [wrow("a", 0.2), wrow("b", 1.91)])
    assert fail.verdict == "FAIL" and fail.statistic["k"] == 1 and fail.n_valid == 2
    assert ok.verdict == "PASS" and ok.statistic["k"] == 0
    assert 0 < ok.statistic["upper95"] < 1                     # the exact one-sided bound, reported beside
    gap = G.witness_verdict("S", [wrow("a", 0.2), wrow("b", 0.3), {"cohort_id": "c", "valid": True,
                                                                   "informative": True, "witness": None}])
    assert gap.statistic["n"] == 3 and gap.statistic["n_witnessed"] == 2                   # N10: the bound on 2, not 3
    assert gap.statistic["upper95"] == G.clopper_pearson_upper(0, 2)


def test_pr1_a_missing_or_stale_witness_is_incomplete_never_pass():
    assert G.witness_verdict("S", [wrow("a", 0.1), wrow("b", None)]).verdict == "incomplete"
    assert G.witness_verdict("S", [wrow("a", 0.1), wrow("b", 0.1, fresh=False)]).verdict == "incomplete"


def test_pr1_reads_valid_informative_rows_only():
    v = G.witness_verdict("S", [wrow("a", 0.1), wrow("b", 50.0, valid=False), wrow("c", 50.0, informative=False)])
    assert v.verdict == "PASS" and v.n_valid == 1 and v.n_all == 3


def test_pr1_an_empty_stratum_is_empty_and_one_with_no_scorable_row_is_reported():
    assert G.witness_verdict("S", []).verdict == "empty"
    assert G.witness_verdict("S", [wrow("a", 3.0, valid=False)]).verdict == "reported"


def test_the_exact_upper_bound():
    assert G.clopper_pearson_upper(0, 30) == pytest.approx(1 - 0.05 ** (1 / 30))
    assert G.clopper_pearson_upper(30, 30) == 1.0


# --------------------------------------------------------------------------------------------------------- PR-2
def test_pr2_an_axis_that_loses_fails_and_its_twin_passes():
    fail = G.coverage_inside("S", "pure", {"eta": {"outcome": "loses", "n": 40}, "epsilon": {"outcome": "no difference", "n": 40}})
    ok = G.coverage_inside("S", "pure", {"eta": {"outcome": "no difference", "n": 40}, "epsilon": {"outcome": "wins", "n": 40}})
    assert fail.verdict == "FAIL" and fail.reasons == ["eta: loses"]
    assert ok.verdict == "PASS"
    assert G.coverage_inside("S", "het", {}).verdict == "reported"


def test_pr2_with_no_measured_axis_is_reported_not_passed():
    """DV-2 (Naveh 2026-10-28): every axis 'not measured' (too few claims to compare) measured nothing, so it reads
    'reported' and is listed under Not Covered (V: WARN is not PASS). Twins: one measured axis keeps PASS or FAIL."""
    nm = {"outcome": "not measured", "n": 3}
    v = G.coverage_inside("P-DTCEIL/inside", "pure", {"eta/calibration": nm, "epsilon/calibration": dict(nm)})
    assert v.verdict == "reported" and v.reasons == ["no axis measured"]
    s = G.gate_summary([v])
    assert s["status"] == "passed" and [x["stratum"] for x in s["not_covered"]] == ["P-DTCEIL/inside"]
    one = G.coverage_inside("S", "pure", {"eta/calibration": nm, "epsilon/calibration": {"outcome": "no difference",
                                                                                          "n": 40}})
    assert one.verdict == "PASS"
    lose = G.coverage_inside("S", "pure", {"eta/calibration": nm, "epsilon/calibration": {"outcome": "loses", "n": 40}})
    assert lose.verdict == "FAIL"



def test_pr2_a_pass_where_loses_cannot_occur_is_reported():
    """DV-5 (Naveh 2026-10-29), planted on LB-1x/inside censor's shape: its one measured axis has 2 cells, where z < -2
    is unreachable, so its PASS reads 'reported'. Twins: 3 cells keep PASS; a loss at any n still FAILs."""
    two = {"outcome": "no difference", "n": 2}
    nm = {"outcome": "not measured", "n": 1}
    v = G.coverage_inside("LB-1x/inside", "censor", {"eta/calibration": two, "epsilon/calibration": nm})
    assert v.verdict == "reported" and "3 cells" in v.reasons[0]
    assert G.coverage_inside("S", "censor", {"eta/calibration": dict(two, n=3)}).verdict == "PASS"
    assert G.coverage_inside("S", "censor", {"eta/calibration": {"outcome": "loses", "n": 2}}).verdict == "FAIL"


def test_pr2_n_min_loses_is_what_the_testers_own_estimator_can_reach():
    """DV-5's constant, recomputed: srbench's cluster_robust_mean (the 003b oracle snapshot of the tester) over every
    cell difference in {-1, 0, 1} and every partition into clusters; 'loses' needs a finite z < -2.0."""
    import itertools
    import os
    import sys
    root = os.path.join(os.path.dirname(__file__), "..", "oracle", "tester_f94ec2b")
    saved = {k: v for k, v in sys.modules.items() if k == "srbench" or k.startswith("srbench.")}
    sys.path.insert(0, root)
    try:
        for k in saved:
            del sys.modules[k]
        from srbench.score import pairing as P
    finally:
        sys.path.remove(root)
        for k in [k for k in sys.modules if k == "srbench" or k.startswith("srbench.")]:
            del sys.modules[k]
        sys.modules.update(saved)              # no snapshot module leaks into later tests

    def partitions(n):
        if n == 0:
            yield []
            return
        for p in partitions(n - 1):
            for i in range(len(p)):
                yield p[:i] + [p[i] + [n - 1]] + p[i + 1:]
            yield p + [[n - 1]]

    def reachable(n):
        for part in partitions(n):
            cl = [next(g for g, b in enumerate(part) if i in b) for i in range(n)]
            for d in itertools.product((-1.0, 0.0, 1.0), repeat=n):
                mean, se, _, _ = P.cluster_robust_mean(list(d), cl)
                if se == se and se > 0 and mean / se < -2.0:
                    return True
        return False
    assert not reachable(G.PR2_N_MIN_LOSES - 1) and reachable(G.PR2_N_MIN_LOSES)


# --------------------------------------------------------------------------------------------------------- PR-3
def test_pr3_a_drop_past_the_margin_fails_and_its_twin_passes():
    fail = G.coverage_outside("S", "pure", "eta", k_out=140, n_out=200, k_in=170, n_in=200)    # 0.70 vs 0.85
    ok = G.coverage_outside("S", "pure", "eta", k_out=168, n_out=200, k_in=170, n_in=200)      # 0.84 vs 0.85
    assert fail.verdict == "FAIL" and fail.interval[0] < -0.10
    assert ok.verdict == "PASS" and ok.interval[0] > -0.10


def test_pr3_passes_equal_coverage_at_the_panels_sizes_and_fails_a_real_drop():
    """The regression for the 0.05 margin (2026-10-22): at SRK-1's valid pure 173 and censor 56 against the reference
    set's 688 and 240, equal coverage must pass; a 5-point drop on pure must fail."""
    for n_out, n_in in ((173, 688), (56, 240)):
        assert G.coverage_outside("S", "a", "eta", k_out=round(0.85 * n_out), n_out=n_out, k_in=round(0.85 * n_in),
                                  n_in=n_in).verdict == "PASS"
    assert G.coverage_outside("S", "pure", "eta", k_out=round(0.80 * 173), n_out=173, k_in=round(0.85 * 688),
                              n_in=688).verdict == "FAIL"


def test_pr3_no_valid_row_outside_is_reported_and_no_comparator_is_incomplete():
    assert G.coverage_outside("S", "het", "eta", k_out=0, n_out=0, k_in=10, n_in=20).verdict == "reported"
    v = G.coverage_outside("S", "pure", "kappa", k_out=0, n_out=0, k_in=10, n_in=20, n_valid_out=172)
    assert v.reasons == ["no closed claim among 172 valid outside"]                    # N10: not "no valid cohort"
    assert G.coverage_outside("S", "pure", "eta", k_out=5, n_out=10, k_in=0, n_in=0).verdict == "incomplete"


def test_the_newcombe_interval_matches_a_hand_computation():
    lo, hi = G.newcombe_diff(56, 70, 48, 80, z=1.959963984540054)     # Newcombe (1998) example 1: 0.0524, 0.3339
    assert lo == pytest.approx(0.0524, abs=5e-4) and hi == pytest.approx(0.3339, abs=5e-4)


# --------------------------------------------------------------------------------------------------------- PR-4
def test_pr4_wider_than_the_null_fails_and_its_twin_passes():
    fail = G.null_verdict("S", "pure", {"eta": {"j": 2.0, "null": 1.5}, "epsilon": {"j": 0.3, "null": 15.0}})
    ok = G.null_verdict("S", "pure", {"eta": {"j": 0.4, "null": 15.0}, "epsilon": {"j": 0.3, "null": 15.0}})
    assert fail.verdict == "FAIL" and fail.reasons == ["eta: J 2 >= null 1.5"]
    assert ok.verdict == "PASS"
    assert G.null_verdict("S", "pure", {"eta": {"j": 0.4, "null": 15.0}}).verdict == "incomplete"   # epsilon missing


def test_pr4_reads_the_testers_information_outcome_when_given():
    """The tester's information_axis (method = J, reference = the null): "wins" = J's closed intervals narrower and its
    closed share not collapsed; anything else fails, even with narrower medians."""
    ok = {"eta": {"j": 0.4, "null": 15.0, "outcome": "wins"}, "epsilon": {"j": 0.3, "null": 15.0, "outcome": "wins"}}
    assert G.null_verdict("S", "pure", ok).verdict == "PASS"
    bad = dict(ok, epsilon={"j": 0.3, "null": 15.0, "outcome": "loses"})     # e.g. its closed share collapsed
    v = G.null_verdict("S", "pure", bad)
    assert v.verdict == "FAIL" and v.reasons == ["epsilon: the tester's information axis reads loses"]
    assert G.null_verdict("S", "pure", dict(ok, eta={"j": None, "null": 15.0, "outcome": "not measured"})).verdict \
        == "incomplete"


# --------------------------------------------------------------------------------------------------------- PR-5
def test_pr5_no_cohort_showing_the_defect_is_not_shown_and_one_is_shown():
    none = G.predecessor_verdict("S", [{"cohort_id": "a", "shown": False}, {"cohort_id": "b", "shown": None}])
    one = G.predecessor_verdict("S", [{"cohort_id": "a", "shown": False}, {"cohort_id": "b", "shown": True}])
    assert none.verdict == "not shown" and none.statistic["n_null"] == 1         # a null point is not shown
    assert one.verdict == "shown" and one.statistic["k_shown"] == 1
    assert G.predecessor_verdict("S", []).verdict == "empty"


# --------------------------------------------------------------------------------------------------------- PR-6
def _v(stratum, rule, verdict):
    return G.Verdict(stratum=stratum, rule=rule, verdict=verdict)


def test_pr6_any_fail_incomplete_or_not_shown_blocks_and_the_twin_passes():
    ok = [_v("A", "PR-1", "PASS"), _v("A", "PR-2", "PASS"), _v("A", "PR-5", "shown")]
    assert G.gate_summary(ok)["status"] == "passed"
    for bad in (_v("A", "PR-1", "FAIL"), _v("A", "PR-3", "incomplete"), _v("A", "PR-5", "not shown")):
        s = G.gate_summary(ok + [bad])
        assert s["status"] == "blocked" and s["blocking"][0]["verdict"] == bad.verdict


def test_pr6_an_empty_stratum_never_passes_and_does_not_block():
    s = G.gate_summary([_v("A", "PR-1", "PASS"), _v("C", "PR-1", "empty"), _v("H", "PR-1", "reported")])
    assert s["status"] == "passed"
    assert {(x["stratum"], x["verdict"]) for x in s["not_covered"]} == {("C", "empty"), ("H", "reported")}


def test_pr6_a_required_verdict_that_is_missing_blocks():
    s = G.gate_summary([_v("A", "PR-1", "PASS")], required=[("A", "PR-1"), ("B", "PR-1")])
    assert s["status"] == "blocked" and s["blocking"][0] == {"stratum": "B", "rule": "PR-1", "verdict": "missing"}
