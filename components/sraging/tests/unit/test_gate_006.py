"""Spec 006's gate arithmetic (tools/gate_006.py; spec.md rules R-1 to R-4; research R11-R13; tasks T022, T025):
R-1's cut from its frozen formula at the real N, the coverage verdict on the tester's closed-claim counts, R-2's paired
widths, R-3's truth witness and R-4's attribution. Pure functions: the tester's scorer gives the counts, these judge
them. Each planted defect beside its twin."""
import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

import gate_006 as G  # noqa: E402
import acceptance_006 as ACC  # noqa: E402

RULE = ACC.read_rules()["rules"]["R-1"]


@pytest.mark.parametrize("n,D,cut", [(200, 1.05, 0.914), (200, 1.23, 0.913), (150, 1.05, 0.912)])
def test_r1s_cut_is_research_r12s_table(n, D, cut):
    assert round(G.r1_cut(n, D=D, p_good=RULE["p_good"], p_bad=RULE["p_bad"], z=RULE["z"]), 3) == cut


def test_the_cut_is_the_midpoint_of_the_two_wilson_bounds():
    n_eff = 200 / 1.05
    lo = G.wilson(0.95, n_eff, RULE["z"])[0]
    hi = G.wilson(0.88, n_eff, RULE["z"])[1]
    assert G.r1_cut(200, D=1.05, p_good=0.95, p_bad=0.88, z=RULE["z"]) == pytest.approx((lo + hi) / 2, abs=0)
    assert round(G.wilson(0.95, 200, RULE["z"])[0], 3) == 0.910 and lo < 0.95 and 0.88 < hi   # gate M9 at n = 200


def test_a_true_088_fails_and_a_true_095_passes_at_200():
    """Planted (spec.md's equal case and real defect, at the real N)."""
    assert not G.r1_verdict(176, 200, RULE)["passes"]           # 0.88
    assert G.r1_verdict(190, 200, RULE)["passes"]               # 0.95, the twin
    v = G.r1_verdict(130, 235, RULE)                            # J's stored window eta [M research R11]
    assert not v["passes"] and round(v["coverage"], 3) == 0.553
    assert G.r1_verdict(181, 181, RULE)["passes"]               # H's stored window eta


def test_r1_gates_eta_and_epsilon_and_reports_beta_and_kappa():
    counts = {"eta": (190, 200), "epsilon": (170, 200), "beta": (5, 17), "kappa": (0, 0)}
    out = G.r1_table(counts, RULE)
    assert set(out["gated"]) == {"eta", "epsilon"} and set(out["reported"]) == {"beta", "kappa"}
    assert out["gated"]["eta"]["passes"] and not out["gated"]["epsilon"]["passes"] and not out["passes"]
    assert "passes" not in out["reported"]["beta"] and out["reported"]["kappa"]["coverage"] is None
    assert G.r1_table({**counts, "epsilon": (188, 200)}, RULE)["passes"]                      # the twin


def test_no_closed_claim_never_passes():
    v = G.r1_verdict(0, 0, RULE)
    assert v["coverage"] is None and v["passes"] is False


def test_r2_pairs_the_widths_of_cohorts_closed_in_both():
    w = {"a": (1.0, 10.0), "b": (1.0, 100.0), "c": None, "d": (2.0, 4.0)}
    h = {"a": (1.0, 100.0), "b": None, "c": (1.0, 10.0), "d": (1.0, 8.0)}
    r = G.paired_log10_ratio(w, h)
    # log10(width_W / width_H), width = upper / lower: the difference of the dex widths (a: 1 - 2, d: log 2 - log 8)
    assert r["n_paired"] == 2 and r["median_dex"] == pytest.approx(statistics_median([-1.0, math.log10(2 / 8)]))
    null = {k: (1e-9, 1e9) for k in w}
    assert G.r2_gate(w, null)["passes"] and not G.r2_gate(null, null)["passes"]       # the null box itself fails


def statistics_median(xs):
    import statistics
    return statistics.median(xs)


def test_r3_names_every_cohort_whose_truth_beats_theta_hat_by_more_than_192():
    rows = [{"cohort_id": "a", "loglik_hat": -100.0, "loglik_truth": -101.0},
            {"cohort_id": "b", "loglik_hat": -100.0, "loglik_truth": -97.5},           # truth 2.5 above: a search failure
            {"cohort_id": "c", "loglik_hat": -100.0, "loglik_truth": -98.1}]           # 1.9 below: passes
    r = G.r3(rows, 1.92)
    assert r["failures"] == ["b"] and not r["passes"]
    ok = G.r3(rows[:1] + rows[2:], 1.92)
    assert ok["passes"] and ok["wilks"]["n"] == 2 and ok["wilks"]["max"] == pytest.approx(2.0)


def test_r3_counts_a_missing_truth_as_a_failure():
    """sr-naveh F9: `r3` crashed on a None loglik_truth, which element_row writes when no objective was captured."""
    r = G.r3([{"cohort_id": "a", "loglik_hat": -100.0, "loglik_truth": None}], 1.92)
    assert r["failures"] == ["a"] and r["missing_truth"] == ["a"] and not r["passes"]


def test_r3_fails_a_theta_hat_moved_off_the_maximum():
    """The unit plant: a reported log-likelihood 2.5 nats below the truth's, as a search that stopped short gives."""
    rows = [{"cohort_id": c, "loglik_hat": -100.0, "loglik_truth": -100.0 - g} for c, g in zip("abc", (0.3, 1.1, 2.0))]
    assert G.r3(rows, 1.92)["passes"]
    rows[1]["loglik_hat"] = rows[1]["loglik_truth"] - 2.5
    assert G.r3(rows, 1.92)["failures"] == ["b"]


def test_r4_exact_compares_h_at_theta_w_with_hs_node_on_every_cohort():
    """Naveh, 2026-11-11: no applicability filter; the pilot's gaps were 2e-10 to 6e-6 nats [M]."""
    rows = [{"cohort_id": "a", "h_node": {"value": 1e-8, "loglik": -3350.598946}, "loglik_H_at_theta_w": -3350.598952},
            {"cohort_id": "b", "h_node": {"value": 1e-8, "loglik": -1719.9}, "loglik_H_at_theta_w": -1719.95}]
    assert G.r4(rows, 0.1)["passes"]
    bad = rows + [{"cohort_id": "c", "h_node": {"value": 1e-8, "loglik": -50.0}, "loglik_H_at_theta_w": -50.2}]
    assert G.r4(bad, 0.1)["failures"] == ["c"]                                   # W 0.2 below what H found
    miss = rows + [{"cohort_id": "d", "h_node": None, "loglik_H_at_theta_w": None}]
    r = G.r4(miss, 0.1)
    assert r["failures"] == ["d"] and r["not_evaluated"] == ["d"] and not r["passes"]


# ------------------------------------------------------- the table of 2026-11-11 (sr-naveh_cuts_006.md; T057a)
RULE_C = {**RULE, "collapse_p_bad": 0.70}


@pytest.mark.parametrize("n,good,bad", [(42, 40, 30), (26, 25, 18)])
def test_a2s_collapse_rule_passes_a_true_095_and_fails_a_true_070(n, good, bad):
    """At A2's 42 and 26 closed claims [M sr-naveh F1]: the cut 0.827 and 0.816."""
    c = G.collapse(RULE_C)
    assert round(G.r1_cut(n, D=c["D"], p_good=c["p_good"], p_bad=c["p_bad"], z=c["z"]), 3) == {42: 0.827, 26: 0.816}[n]
    assert G.r1_verdict(good, n, c)["passes"] and not G.r1_verdict(bad, n, c)["passes"]


def test_a1s_cut_passes_a_true_095_and_fails_a_true_088_at_193():
    assert G.r1_verdict(183, 193, RULE)["passes"] and not G.r1_verdict(170, 193, RULE)["passes"]


def _claim(q, shape, lo=None, hi=None, kind=None, label="", widened=None):
    return {"parameter": q, "shape": shape, "lower": lo, "upper": hi, "label": label,
            "label_info": {"kind": kind, "side": "upper"} if kind else None,
            "widening": {"upper_moved_dex": widened} if widened else None}


def test_truncated_above_names_box_limited_short_and_contradicted_upper_ends():
    """sr-naveh F7: Denmark 1890 M's eta, box-limited and widened, 0.74 dex below H's and contradicted by the walk."""
    den = {"cohort_id": "den", "claims": [_claim("eta", "closed", 1.01e-4, 9.86e-4, "box_limited")],
           "combinations": {"param_claims_contradicted": [{"parameter": "eta", "side": "upper", "dex": 0.757}]}}
    assert G.truncated_above(den, "eta", {"den": (1e-4, 5.39e-3)}) == "box_limited"
    short = {"cohort_id": "s", "claims": [_claim("eta", "closed", 1e-4, 1e-3)]}
    assert "0.50 dex below H's" in G.truncated_above(short, "eta", {"s": (1e-4, 10 ** -2.5)})
    assert G.truncated_above(short, "eta", {"s": (1e-4, 10 ** -2.9)}) is None              # the twin: 0.1 dex
    walk = {"cohort_id": "w", "claims": [_claim("eta", "closed", 1e-4, 1e-3)],
            "combinations": {"param_claims_contradicted": [{"parameter": "eta", "side": "upper", "dex": 0.4}]}}
    assert "contradicted above" in G.truncated_above(walk, "eta", {})
    t = G.truncation([den, short], {"eta": {"den": (1e-4, 5.39e-3), "s": (1e-4, 10 ** -2.9)}}, params=("eta",))
    assert t["eta"]["closed"] == 2 and t["eta"]["truncated"] == 1 and t["eta"]["rate"] == 0.5


def test_r2_against_h_passes_w_equal_to_h_and_fails_h_widened_by_03_dex():
    """Naveh, 2026-11-11: the paired median at most +0.1 dex; the planted defect is H's own claims widened 0.3 dex."""
    h = {f"c{i}": (10 ** (-4 - 0.01 * i), 10 ** (-3 + 0.01 * i)) for i in range(150)}
    assert G.r2_vs_h(dict(h), h, 0.1)["passes"]
    wide = {k: (lo / 10 ** 0.15, hi * 10 ** 0.15) for k, (lo, hi) in h.items()}
    r = G.r2_vs_h(wide, h, 0.1)
    assert not r["passes"] and r["median_dex"] == pytest.approx(0.3)


def test_a_truncated_claim_is_open_in_r2s_widths():
    rows = [{"cohort_id": "a", "claims": [_claim("eta", "closed", 1e-4, 1e-3, "box_limited")]},
            {"cohort_id": "b", "claims": [_claim("eta", "closed", 1e-4, 1e-3)]}]
    assert G.w_closed_bounds(rows, "eta", {}) == {"a": None, "b": (1e-4, 1e-3)}


def test_data_determined_one_sided_claims():
    assert G.data_determined_one_sided(_claim("eta", "lower_bound", lo=1e-4))
    assert G.data_determined_one_sided(_claim("beta", "upper_bound", hi=5.0))
    assert not G.data_determined_one_sided(_claim("beta", "upper_bound", hi=60.0, kind="box_limited"))
    assert not G.data_determined_one_sided(_claim("eta", "upper_bound", hi=1e-3, label="at least 1e-3; edge not found"))
    assert not G.data_determined_one_sided(_claim("eta", "closed", 1e-4, 1e-3))


def _one_sided_rows(n, hits):
    rows, truths = [], {}
    for i in range(n):
        rows.append({"cohort_id": f"c{i}", "claims": [_claim("eta", "lower_bound", lo=1e-4),
                                                      _claim("epsilon", "lower_bound", lo=1e-4)]})
        truths[f"c{i}"] = {"eta": 2e-4 if i < hits else 5e-5, "epsilon": 2e-4, "beta": 1.0, "kappa": 1.0}
    return rows, truths


def test_the_one_sided_gate_applies_from_20_claims_under_the_collapse_rule():
    """Naveh, 2026-11-11. At 20 claims: a true 0.95 passes with P 0.984, a true 0.70 with P 0.107 [M]."""
    rows, truths = _one_sided_rows(30, 29)
    assert G.one_sided_gate(rows, truths, RULE_C)["passes"]
    rows, truths = _one_sided_rows(30, 21)                                       # 0.70: fails
    out = G.one_sided_gate(rows, truths, RULE_C)
    assert not out["passes"] and out["by_param"]["eta"]["passes"] is False
    rows, truths = _one_sided_rows(19, 10)                                       # below 20: reported, not gated
    out = G.one_sided_gate(rows, truths, RULE_C)
    assert out["passes"] and out["by_param"]["eta"]["passes"] is None


def test_fr022_allows_5_percent_of_the_assessed_fits():
    """Naveh, 2026-11-11: 9 of 196 pass, 10 fail; not_assessed and failed are not assessed fits."""
    def rows(k):
        return ([{"cohort_id": f"f{i}", "leverage_flag": "flag", "leverage": {"dropped": {"age": 400.0}},
                  "window": {"window_eff": [365.0, 1100.0]}} for i in range(k)]
                + [{"cohort_id": f"n{i}", "leverage_flag": "no_flag"} for i in range(196 - k)]
                + [{"cohort_id": "x", "leverage_flag": "not_assessed"}, {"cohort_id": "y", "leverage_flag": "failed"}])
    ok = G.fr022(rows(9), 0.05)
    assert ok["passes"] and ok["assessed"] == 196 and ok["allowed"] == 9
    assert ok["flags"]["f0"] == {"dropped": {"age": 400.0}, "window": [365.0, 1100.0]}
    assert not G.fr022(rows(10), 0.05)["passes"]
    assert G.fr022([{"cohort_id": c, "leverage_flag": lab} for c, lab in
                    zip("abcd", ["flat_valley_slide", "no_flag", "not_assessed", "failed"])], 0.05)["leverage_flagged"] == 0


def test_misses_are_named_by_side_with_the_truths_margin():
    rows = [{"cohort_id": c, "claims": [_claim("eta", "closed", 1e-4, 1e-3)]} for c in "abc"]
    truths = {"a": {"eta": 5e-5}, "b": {"eta": 1e-2}, "c": {"eta": 5e-4}}
    m = G.misses(rows, truths, {"a": "k1", "b": "k1", "c": "k2"}, params=("eta",))["eta"]
    assert m["below"] == {"a": pytest.approx(math.log10(2))} and m["above"] == {"b": pytest.approx(1.0)}
    assert m["clusters_with_2_or_more"] == ["k1"]


def test_the_lead_pairs_coverage_is_scored_against_the_truths_combinations():
    from sraging.claims import combinations as K
    truth = {"eta": 2e-4, "beta": 0.1, "epsilon": 1e-3, "kappa": 1.0}
    q = {k: K.value(k, truth) for k in G.LEAD_PAIR}

    def row(c, hit):
        f = 1.0 if hit else 100.0
        return {"cohort_id": c, "combinations": {"on": True}, "negligible_labels": [],
                "claims": [_claim(k, "closed", v * f / 2, v * f * 2) for k, v in q.items()]}
    rows = [row(f"c{i}", i < 29) for i in range(30)]
    out = G.lead_pair_report(rows, {r["cohort_id"]: truth for r in rows}, RULE_C)
    assert out["n"] == 30 and out["by_quantity"]["eta_over_eps"]["hits"] == 29 and out["passes"]
    rows = [row(f"c{i}", i < 20) for i in range(30)]                              # 0.67: a collapse
    assert not G.lead_pair_report(rows, {r["cohort_id"]: truth for r in rows}, RULE_C)["passes"]
