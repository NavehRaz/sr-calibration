"""Spec 003d's acceptance tool (tools/acceptance_003d.py): the library driver's spec conversion and checksum, and the
statistics every fit-free phase rests on (research R10; panel_003d.toml thresholds). Fast; generates nothing large.
"""
import json
import math

import numpy as np
import pytest

import acceptance_003d as ACC
import v5_reproduce_003d as R
from sraging.errors import UnsupportedOptionError

ROW = {"cohort_id": "het-k0.01-p01-c0-s0", "library_version": "v5", "group": "main", "position": 3, "arm": "het",
       "stratum": "ratio-lo/cv-mid/n-1000", "cluster_key": "x", "real_name": None, "notes": "", "window": [3.0, 9.0],
       "rho_eta": 2.0, "rho_beta": 0.1, "rho_eps": 0.0158, "kappa": 0.01, "n": 1000, "L_phys": 100.0, "seed": 400100,
       "delta": 1.0, "censor_frac": 0.0, "external_hazard": None, "het_param": "xc", "het_var": 0.2}


def test_library_spec_drops_the_testers_organisational_fields_and_the_window():
    s = ACC.spec_from_library(ROW)
    assert (s.name, s.n, s.seed, s.het_param, s.het_var, s.time_range, s.external_hazard) == \
        ("het-k0.01-p01-c0-s0", 1000, 400100, "xc", 0.2, None, None)
    s2 = ACC.spec_from_library({**ROW, "het_param": None, "het_var": 0.0, "external_hazard": 5.5}, window=True)
    assert s2.external_hazard == 5.5 and s2.time_range == (3.0, 9.0) and not s2.heterogeneous
    with pytest.raises(UnsupportedOptionError):
        ACC.spec_from_library({**ROW, "window": [3.5, 9.0]}, window=True)        # off-grid, refused by the generator


def test_the_checksum_is_the_testers():
    assert ACC.cohort_digest is R.cohort_digest


def test_the_sc001_bound_and_rule():
    assert abs(ACC.dlncv_bound(100000) - 3.5355339e-4) < 1e-10 and abs(ACC.dlncv_bound(10000) - 1.1180340e-3) < 1e-9
    assert ACC.passes_sc001(kl=0.05 / (2 * 1000) * 0.99, dlncv=0.0, n=1000)
    assert not ACC.passes_sc001(kl=0.05 / (2 * 1000) * 1.01, dlncv=0.0, n=1000)
    b = ACC.dlncv_bound(1000)
    assert ACC.passes_sc001(kl=0.0, dlncv=0.74 * b, n=1000) and not ACC.passes_sc001(kl=0.0, dlncv=0.76 * b, n=1000)


def test_the_g_test_merges_to_expected_five_and_matches_a_hand_computation():
    obs = np.array([10, 20, 30, 3, 1, 0])
    exp = np.array([12.0, 18.0, 30.0, 2.0, 1.5, 0.5])
    g, df, p = ACC.g_test(obs, exp)
    # merged left to right until each expected count is >= 5: [12, 18, 30 + 2 + 1.5 + 0.5] with [10, 20, 34]
    O, E = np.array([10, 20, 34]), np.array([12.0, 18.0, 34.0])
    want = 2 * np.sum(O * np.log(O / E))
    assert abs(g - want) < 1e-12 and df == 2 and 0 < p < 1


def test_the_group_rule():
    m = 100
    assert abs(ACC.fraction_bound(m) - (0.01 + 3 * math.sqrt(0.0099 / m))) < 1e-15
    rng = np.random.default_rng(0)
    assert ACC.group_passes(rng.random(500))["pass"]
    assert not ACC.group_passes(np.full(50, 1e-4))["pass"]


def test_heterogeneous_configurations_are_judged_at_the_cap():
    assert ACC.n_eval(100000, True) == 20000 and ACC.n_eval(10000, True) == 10000
    assert ACC.n_eval(100000, False) == 100000


def test_the_heterogeneous_rule_combines_the_node_and_grid_parts():
    # Naveh 2026-10-14: lambda <= 0.2 at n_eval (<= 20,000); the tester's CV rule 2 n dlnCV^2 <= 0.1 at the real n;
    # the two parts combine as kl_total = 2 (kl_node + kl_grid) and |dlnCV| adds
    c = ACC.combine_het({"kl": 1e-7, "dlncv": 1e-4}, {"kl": 2e-7, "dlncv": -3e-4})
    assert abs(c["kl"] - 6e-7) < 1e-18 and abs(c["dlncv"] - 4e-4) < 1e-18
    assert ACC.passes_het(kl=0.2 / (2 * 20000) * 0.99, dlncv=0.0, n_eval=20000, n_max=100000)
    assert not ACC.passes_het(kl=0.2 / (2 * 20000) * 1.01, dlncv=0.0, n_eval=20000, n_max=100000)
    d = math.sqrt(0.1 / (2 * 100000))
    assert ACC.passes_het(kl=0.0, dlncv=0.99 * d, n_eval=20000, n_max=100000)
    assert not ACC.passes_het(kl=0.0, dlncv=1.01 * d, n_eval=20000, n_max=100000)


def test_the_node_part_is_read_per_point_and_delta(tmp_path):
    f = tmp_path / "nodes.json"
    f.write_text(json.dumps({"points": [{"point": [1.0, 2.0, 3.0, 4.0], "configs": [
        {"delta_over_L": 0.01, "rows": [{"K": 81, "kl": 3e-6, "dlncv": 2e-3}, {"K": 161, "kl": 1e-7, "dlncv": 1e-4}]}]}]}))
    part = ACC.node_part(str(f))
    assert part[(1.0, 2.0, 3.0, 4.0, 0.01)][161] == {"kl": 1e-7, "dlncv": 1e-4}


def test_the_fidelity_rule():
    # SC-002: mean ln CV of recorded ages over replicates within 1.5e-3 of the binned reference's ln CV
    assert abs(ACC.ln_sample_cv(np.array([1.0, 2.0, 3.0])) - math.log(math.sqrt(2 / 3) / 2.0)) < 1e-15
    assert ACC.fidelity_passes([-3.0 + 1e-3, -3.0 - 1e-3, -3.0], -3.0)
    assert not ACC.fidelity_passes([-3.0 + 2e-3] * 3, -3.0)


def test_counts_map_recorded_ages_and_censoring_onto_the_laws_cells():
    law = {"ages": np.array([0.0, 1.0, 2.0, 3.0]), "events": np.array([1, 1, 1, 0]), "C": 3.0}
    ages = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0])
    ev = np.array([1, 1, 1, 1, 0, 0, 0])
    assert list(ACC.cell_counts(ages, ev, law, delta=1.0)) == [2, 1, 1, 3]
    law2 = {"ages": np.array([0.0, 1.0, 2.0]), "events": np.array([1, 1, 1]), "C": None}
    assert list(ACC.cell_counts(np.array([0.0, 2.0, 2.0]), np.array([1, 1, 1]), law2, delta=1.0)) == [1, 0, 2]


def test_the_recording_checks():
    ok = ACC.recording_checks(np.array([0.0, 1.0, 2.0, 3.0]), np.array([1, 1, 1, 0]), delta=1.0, C=3.0)
    assert ok == {"on_grid": True, "no_age_in_first_bin_but_zero": True, "no_death_at_C": True, "censored_at_C": True}
    bad = ACC.recording_checks(np.array([0.5, 3.0]), np.array([1, 1]), delta=1.0, C=3.0)
    assert not bad["on_grid"] and not bad["no_death_at_C"]


def test_the_shards_cover_every_point_once():
    rows = [{"rho_eta": float(i), "rho_beta": 1.0, "rho_eps": 1.0, "kappa": 1.0, "cohort_id": f"c{i}-{j}"}
            for i in range(7) for j in range(2)]
    got = [r["cohort_id"] for k in range(1, 4) for rs in ACC.shard_points(rows, k, 3) for r in rs]
    assert sorted(got) == sorted(r["cohort_id"] for r in rows)


def test_williams_correction_and_the_adopted_statistic():
    g, df, p = 300.0, 299, None
    pw = ACC.williams_p(g, df, n=2000)
    q = 1 + (df + 2) / (6 * 2000)
    from scipy.stats import chi2
    assert abs(pw - chi2.sf(g / q, df)) < 1e-15 and pw > chi2.sf(g, df)
    ok = {"pass": True}
    bad = {"pass": False}
    # Naveh 2026-10-14: raw if every group passes; Williams if each raw-failing group also fails on exact draws
    assert ACC.adopted_statistic({"a": ok}, {"a": ok}) == "raw"
    assert ACC.adopted_statistic({"a": bad, "b": ok}, {"a": bad, "b": ok}) == "williams"
    assert ACC.adopted_statistic({"a": bad}, {"a": ok}) == "real-fail"


# ------------------------------------------------------------------------------------- the prediction (T039, R12)
def _toy_curve():
    tau = np.linspace(0.0, 30.0, 30001)
    return tau, np.exp(-(tau / 1.1) ** 6)


TOY = {"cohort_id": "censor-k0.1-p00-c1-s0", "L_phys": 100.0, "delta": 1.0, "censor_frac": 0.2,
       "external_hazard": None, "seed": 7, "n": 400000}


def test_predict_the_backward_euler_law_is_its_samplers_large_n_limit():
    tau, S = _toy_curve()
    law = ACC.be_law(TOY, tau, S)
    kc = law["ages"].size - 2
    assert law["events"][-1] == 0 and np.all(law["events"][:-1] == 1)
    assert law["ages"][-1] == law["ages"][-2] == kc * 1.0 == math.floor(law["C"])      # ties: deaths and censored
    assert abs(law["P"].sum() - 1.0) < 1e-12 and abs(law["P"][-1] - 0.2) < 1e-9
    ages, ev = R.materialize_srk(TOY, tau, S)                                          # SRK's own sampler
    obs = ACC.cell_counts(ages, ev, law, 1.0)
    exp = TOY["n"] * law["P"]
    assert obs.sum() == TOY["n"] and obs[-1] == 0.2 * TOY["n"]           # the sample quantile censors exactly f n
    # the tie cell [floor(C), C] moves with the sample quantile (its SE here ~0.04 of a 0.08-wide cell): it is tested
    # together with the censored cell, which the quantile does not move; every other cell on its own
    big = np.append(exp[:-2] >= 25, [False, False])
    assert np.max(np.abs(obs[big] - exp[big]) / np.sqrt(exp[big])) < 5.0
    assert abs(obs[-2:].sum() - exp[-2:].sum()) / math.sqrt(exp[-2:].sum()) < 5.0


def test_predict_refuses_what_it_does_not_model():
    tau, S = _toy_curve()
    with pytest.raises(ValueError, match="external_hazard"):
        ACC.be_law({**TOY, "external_hazard": 5.0}, tau, S)
    with pytest.raises(ValueError, match="censor_frac"):
        ACC.be_law({**TOY, "censor_frac": 0.0}, tau, S)


def test_predict_the_expected_count_cohort_rounds_each_cell():
    law = {"ages": np.array([0.0, 1.0, 2.0, 2.0]), "events": np.array([1, 1, 1, 0]),
           "P": np.array([0.1, 0.3349, 0.3651, 0.2])}
    ages, ev, m = ACC.expected_cohort(law, n=100, scale=10)
    assert list(m) == [100, 335, 365, 200] and ages.size == ev.size == 1000
    assert np.all(np.diff(ages) >= 0) and int((ev == 0).sum()) == 200 and np.all(ages[ev == 0] == 2.0)
    assert int(((ages == 2.0) & (ev == 1)).sum()) == 365


def test_predict_the_half_width_faces_the_truth_and_the_rule_uses_the_median():
    claims = [{"shape": "closed", "lower": 0.5, "upper": 1.1, "point": 1.0},
              {"shape": "closed", "lower": 0.8, "upper": 1.3, "point": 1.0},
              {"shape": "upper_bound", "lower": None, "upper": 2.0, "point": 1.0}]       # not closed: not counted
    assert abs(ACC.half_width(claims[0], "upper") - math.log10(1.1)) < 1e-15
    assert abs(ACC.half_width(claims[1], "lower") - math.log10(1 / 0.8)) < 1e-15
    med_up = (math.log10(1.1) + math.log10(1.3)) / 2
    p = ACC.predict_rule(-(med_up + 1e-3), claims)              # pseudo-true below the truth: the upper edge
    assert p["side"] == "upper" and abs(p["median_half_width"] - med_up) < 1e-15 and p["predicted_miss"]
    assert not ACC.predict_rule(-(med_up - 1e-3), claims)["predicted_miss"]
    q = ACC.predict_rule(+0.2, claims)                          # above the truth: the lower edge
    assert q["side"] == "lower" and q["predicted_miss"] == (0.2 > (math.log10(2) + math.log10(1.25)) / 2)
    assert ACC.predict_rule(-0.5, claims[2:])["predicted_miss"] is None                   # no closed claim


def test_predict_configurations_group_the_panel_by_point_and_n():
    base = {"rho_eta": 1.0, "rho_beta": 0.1, "rho_eps": 0.01, "kappa": 0.1, "L_phys": 100.0, "delta": 1.0,
            "censor_frac": 0.2, "external_hazard": None, "het_var": 0.0}
    specs = {f"censor-k0.1-p00-c{c}-s{s}": {**base, "cohort_id": f"censor-k0.1-p00-c{c}-s{s}",
                                             "n": 1000 * 10 ** c, "seed": s} for c in (0, 1) for s in (0, 1)}
    rows = [{"cohort_id": i, "arm": "censor"} for i in sorted(specs)][:3]
    cf = ACC.predict_configurations(rows, specs)
    assert sorted(cf) == ["censor-k0.1-p00-c0", "censor-k0.1-p00-c1"]
    assert cf["censor-k0.1-p00-c0"]["ids"] == ["censor-k0.1-p00-c0-s0", "censor-k0.1-p00-c0-s1"]
    assert cf["censor-k0.1-p00-c1"]["ids"] == ["censor-k0.1-p00-c1-s0"] and cf["censor-k0.1-p00-c1"]["n"] == 10000
    bad = {**specs, "censor-k0.1-p00-c0-s1": {**specs["censor-k0.1-p00-c0-s1"], "rho_eps": 0.02}}
    with pytest.raises(ValueError, match="rho_eps"):
        ACC.predict_configurations(rows, bad)


def test_predict_the_fit_is_the_calibrators_own_call_stopped_after_the_mle(monkeypatch):
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.compat.jax2 import claims as jc
    from sraging.compat.jax2 import estimator as est
    calls = []

    def fake_fit(fj, ages, events, **kw):
        calls.append((list(ages), list(events), kw))
        return {"theta": [1e-3, 1e-3, 1e-4, 0.1], "loglik": -1.0, "diagnostics": {"solver": {"N": 900}}}

    monkeypatch.setattr(est, "fit", fake_fit)
    ages, ev = [1.0, 2.0, 2.0, 3.0, 3.0], [1, 1, 1, 1, 0]                  # a death tied with the censoring
    jc.run_fit(PJ.Observation("x", tuple(ages), tuple(ev), 1.0, None), **PJ.run_kwargs(PJ.CLASSES["profile_jax2"]))
    out = ACC.theta_hat_only(ages, ev, 1.0, "profile_jax2")
    (a0, e0, k0), (a1, e1, k1) = calls
    assert (a0, e0) == (a1, e1) and e1 == [1, 1, 1, 0, 0]                  # the observation model ran (relabel)
    assert k1 == {**k0, "stages": ("mle",)}
    assert out["theta"] == {"eta": 1e-3, "beta": 1e-3, "epsilon": 1e-4, "kappa": 0.1} and out["loglik"] == -1.0


# ------------------------------------------------------------------------------ the re-test's scoring (T048, SC-011)
def test_the_witness_rule():
    ok = ACC.witness_rule([0.5, 1.0, 2.0, 3.0, 4.0, 12.0] + [1.0] * 10)            # 1/16 above 9.49, median 1.0
    assert ok["pass"] and ok["n_above_cut"] == 1 and ok["median"] == 1.0 and ok["code_findings"] == []
    many = ACC.witness_rule([10.0, 11.0] + [1.0] * 10)                               # 2/12 = 0.167 > 10/122
    assert not many["pass"] and "fraction" in " ".join(many["fails"])
    high = ACC.witness_rule([5.0] * 9)                                                # median 5.0 > 4.4
    assert not high["pass"] and "median" in " ".join(high["fails"])
    neg = ACC.witness_rule([1.0, 2.0, -0.2], ids=["a", "b", "c"])                     # a D below -0.1: a code finding
    assert not neg["pass"] and neg["code_findings"] == [{"cohort_id": "c", "D": -0.2}]


def test_the_v2_pairs_are_per_cohort_and_marked_where_not_like_for_like():
    claim = lambda lo, hi: {"parameter": "eta_over_eps", "shape": "closed", "lower": lo, "upper": hi, "point": 1.0}
    truth = {"eta": 2.0, "beta": 1.0, "epsilon": 1.0, "kappa": 1.0}                   # eta/eps = 2
    new = [{"cohort_id": "a", "calibrator": "profile_jax2", "status": "ok", "truth": truth,
            "claims": [claim(1.5, 3.0)]},
           {"cohort_id": "b", "calibrator": "profile_jax2", "status": "ok", "truth": truth, "claims": [claim(0.5, 1.5)]}]
    old = [{**new[0], "claims": [claim(0.5, 1.5)]}, {**new[1], "claims": [claim(0.5, 1.5)]}]
    p = ACC.v2_pairs(new, old, "eta_over_eps")
    assert p["pairs"] == {"a": [False, True], "b": [False, False]} and p["v2_miss_new_hit"] == 1
    assert p["v2_hit_new_miss"] == 0 and p["like_for_like"]
    assert not ACC.v2_pairs(new, old, "sqrt_eta_over_eps")["like_for_like"]       # v2 walked six, the re-test three


def test_the_witness_forecast_is_a_noncentral_chi_squared_count():
    from scipy.stats import binom, chi2
    f = ACC.witness_forecast([0.0] * 67, df=4)                        # a perfect calibrator: the chi-squared tail only
    p0 = chi2.sf(ACC.TH["witness_D_cut"], 4)
    assert abs(f["expected_above_cut"] - 67 * p0) < 1e-9
    k_fail = math.floor(ACC.TH["witness_frac_max"] * 67) + 1          # 6 of 67 exceed 10/122
    assert f["k_fail"] == k_fail and abs(f["p_fraction_fails"] - binom.sf(k_fail - 1, 67, p0)) < 1e-12
    g = ACC.witness_forecast([22.0] * 4 + [0.0] * 63, df=4)            # four large offsets raise the risk
    assert g["expected_above_cut"] > f["expected_above_cut"] + 3 and g["p_fraction_fails"] > f["p_fraction_fails"]


def test_predict_the_dt_rule_variant_differs_from_the_calibrators_call_by_dt_rule_only(monkeypatch):
    from sraging.compat.jax2 import estimator as est
    calls = []

    def fake_fit(fj, ages, events, **kw):
        calls.append(kw)
        return {"theta": [1e-3, 1e-3, 1e-4, 0.1], "loglik": -1.0, "diagnostics": {}}

    monkeypatch.setattr(est, "fit", fake_fit)
    ages, ev = [1.0, 2.0, 3.0], [1, 1, 0]
    ACC.theta_hat_only(ages, ev, 1.0, "profile_jax2")
    ACC.theta_hat_only(ages, ev, 1.0, "profile_jax2", dt_rule=True)
    assert "dt_rule" not in calls[0] and calls[1] == {**calls[0], "dt_rule": True}


def test_the_witness_pending_rows_are_named_not_hidden():
    probs = ["a: no row", "b: no row", "c: not the manifest's array"]
    kept, pending = ACC.pending_filter(probs, ["a"])
    assert kept == ["b: no row", "c: not the manifest's array"] and pending == ["a"]
    with pytest.raises(ValueError, match="z"):
        ACC.pending_filter(probs, ["z"])            # a pending id that is not missing is refused, by name
