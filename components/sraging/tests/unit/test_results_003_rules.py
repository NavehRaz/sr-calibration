"""Spec 003's acceptance pass rules (tests/acceptance/test_results_003.py `check_*`) on synthetic results: each passes
a good result and FAILS on a planted bad one (constitution V; sr-manager gate 5, 7, 11; sr-naveh review 4). The
coverage result is built by the coverage tool's own `summarise`, so the rule is tested on what the tool writes.
"""
import copy
import importlib.util
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location("results_003_rules",
                                               os.path.join(REPO, "tests", "acceptance", "test_results_003.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

import acceptance_003 as A        # noqa: E402  (tools/ is on sys.path: tests/conftest.py)
import acceptance_003_coverage as CV  # noqa: E402

NA = pytest.skip.Exception
FAIL = (AssertionError, pytest.fail.Exception)


def rt(d):
    """Through JSON, as the result file is read."""
    return json.loads(json.dumps(d, allow_nan=True))


# ------------------------------------------------------------------------------------------------ order
SPREADS = {"eta_over_eps": 0.003, "sqrt_eta_over_eps": 0.0031, "rbar_over_eta": 0.026, "rbar": 0.032,
           "beta_over_eta_kp1": 0.292, "beta_over_kp1": 0.295}       # research R13's valid-panel values (one nudged)
RULE_ORDER = ["eta_over_eps", "rbar_over_eta", "sqrt_eta_over_eps", "rbar", "beta_over_eta_kp1", "beta_over_kp1"]
HYP = ["eta_over_eps", "rbar_over_eta", "sqrt_eta_over_eps", "beta_over_eta_kp1", "rbar", "beta_over_kp1"]


def order_result(order=RULE_ORDER, lead=RULE_ORDER[:2], spreads=SPREADS):
    return rt({"status": "written", "order": order, "lead_group": lead, "spread_valid_panel_dex": spreads,
               "all_sets": {"panel valid": {k: [v, 0.1, 19] for k, v in spreads.items()}},
               "departures_from_hypothesis": [[i, a, b] for i, (a, b) in enumerate(zip(order, HYP)) if a != b]})


def test_order_rule_passes_the_order_the_rule_gives():
    R.check_order(order_result())


def test_order_rule_fails_a_hard_coded_hypothesis_order():
    """sr-manager gate 11: the hypothesis' order on these spreads puts beta/(eta(kappa+1)) ahead of r-bar."""
    with pytest.raises(FAIL):
        R.check_order(order_result(order=HYP))


def test_order_rule_fails_a_lead_that_is_not_ridge_invariant_or_a_tie():
    with pytest.raises(FAIL):          # sqrt(eta)/eps is narrow but not invariant on the scaling ridge
        R.check_order(order_result(order=["eta_over_eps", "sqrt_eta_over_eps", "rbar_over_eta"] + RULE_ORDER[3:],
                                   lead=["eta_over_eps", "sqrt_eta_over_eps", "rbar_over_eta"]))
    tied = dict(SPREADS, rbar=SPREADS["sqrt_eta_over_eps"])
    with pytest.raises(FAIL):
        R.check_order(order_result(spreads=tied))
    bad = order_result()
    bad["all_sets"]["panel valid"]["rbar"][0] = 0.07                 # the spread is not the panel's
    with pytest.raises(FAIL):
        R.check_order(bad)


# ------------------------------------------------------------------------------------------------ floor
NEG = {"meta": {"dlogl_tol": 0.5}, "beta": [
    {"regime": "sharp", "rho_eps": 0.1, "kappa": 0.05, "ratio_max_clean": 0.68,
     "floors": [{"n": 130, "floor_dlogl": 0.0772}, {"n": 400, "floor_dlogl": 0.0470}]},
    {"regime": "mid", "rho_eps": 0.3, "kappa": 0.15, "ratio_max_clean": 1.0,
     "floors": [{"n": 130, "floor_dlogl": 0.173}, {"n": 400, "floor_dlogl": 0.107}]},
    {"regime": "kappa_large", "rho_eps": 0.3, "kappa": 1.0, "ratio_max_clean": 1.4678,
     "floors": [{"n": 130, "floor_dlogl": 0.400}, {"n": 400, "floor_dlogl": 0.245}]},
    {"regime": "diffuse", "rho_eps": 0.5, "kappa": 0.5, "ratio_max_clean": 1.4678,
     "floors": [{"n": 130, "floor_dlogl": 1.4678}, {"n": 400, "floor_dlogl": 1.4678}]}]}


def floor_result():
    return rt({"status": "written", "sources": {"neg_floors.json": {"md5": R.NEG_FLOORS_MD5}},
               "regimes": A.floor_coefficients(NEG)})


def test_floor_coefficients_are_the_mean_of_floor_sqrt_n_and_a_lower_bound_when_censored():
    reg = A.floor_coefficients(NEG)
    assert reg["sharp"]["c_beta"] == pytest.approx((0.0772 * 130 ** 0.5 + 0.0470 * 20) / 2)
    assert reg["diffuse"]["lower_bound"] and reg["diffuse"]["c_beta"] == pytest.approx(1.4678 * 130 ** 0.5)
    assert reg["diffuse"]["censored_n"] == [130, 400] and not reg["mid"]["censored_n"]


def test_floor_measured_from_the_stored_sweep_reproduces_the_comment_but_diffuse():
    """The pinned neg_floors.json gives space_search.py's 0.93, 2.21, 5.07; diffuse's lower bound is 16.74, where
    the comment says 16.8 (the copy that sr-manager gate 10 replaced)."""
    if not os.path.exists(A.NEG_FLOORS_JSON):
        pytest.skip(f"NOT APPLICABLE: {A.NEG_FLOORS_JSON} is not on this host")
    import hashlib
    raw = open(A.NEG_FLOORS_JSON, "rb").read()
    assert hashlib.md5(raw).hexdigest() == A.NEG_FLOORS_MD5
    reg = A.floor_coefficients(json.loads(raw))
    assert {k: round(v["c_beta"], 2) for k, v in reg.items()} == {"sharp": 0.93, "mid": 2.21, "kappa_large": 5.07,
                                                                   "diffuse": 16.74}
    assert {k: round(v["c_rbar"], 2) for k, v in reg.items()} == {"sharp": 0.79, "mid": 1.53, "kappa_large": 1.56,
                                                                   "diffuse": 7.54}
    R.check_floor(rt({"status": "written", "sources": {"neg_floors.json": {"md5": A.NEG_FLOORS_MD5}},
                      "regimes": reg}))


def test_floor_rule_fails_a_copied_coefficient_or_another_sweep():
    R.check_floor(floor_result())
    f = floor_result()
    f["regimes"]["diffuse"]["c_beta"] = 16.8                          # the comment's value, not the sweep's
    f["regimes"]["diffuse"]["c_rbar"] = 16.8 * f["regimes"]["diffuse"]["rbar_factor"]
    with pytest.raises(FAIL):
        R.check_floor(f)
    g = floor_result()
    g["sources"]["neg_floors.json"]["md5"] = "0" * 32
    with pytest.raises(FAIL):
        R.check_floor(g)


# ------------------------------------------------------------------------------------------------ walker
def walker_result(ratio=1.5, decision=None):
    return {"checks": {"passed": 40, "total": 40}, "failures": [], "errors": [], "cost_ratio_median": ratio,
            "cost_limit": 2.0, "cost_decision": decision}


def test_walker_rule_needs_navehs_decision_above_the_cost_limit():
    R.check_walker(walker_result())
    with pytest.raises(FAIL):
        R.check_walker(walker_result(ratio=2.4))
    with pytest.raises(FAIL):
        R.check_walker(walker_result(ratio=2.4, decision={"decision": "(b)", "date": "2026-10-06", "by": "the agent"}))
    R.check_walker(walker_result(ratio=2.4, decision={"decision": "(b) the lead pair on", "date": "2026-10-06",
                                                      "by": "Naveh"}))
    with pytest.raises(FAIL):
        R.check_walker(dict(walker_result(), checks={"passed": 0, "total": 0}))
    with pytest.raises(FAIL):
        R.check_walker(dict(walker_result(), archive_failures=[{"key": "rbar", "ok": False}]))
    assert CV.walker_problems(dict(walker_result(), archive_failures=[{"key": "rbar"}]))
    for w, ok in ((walker_result(ratio=2.4), False),
                  (walker_result(ratio=2.4, decision={"decision": "c", "date": "2026-10-06", "by": "Naveh"}), True)):
        assert (not CV.walker_problems(w)) == ok                      # the coverage refusal reads the same field


# ------------------------------------------------------------------------------------------------ kmax
def kmax_result():
    def fit(cap):
        return {"beta_upper": cap, "combos": {"rbar": {"ci_hi": 1.0}, "rbar_over_eta": {"ci_hi": 2.0}}}
    kept = [f"k{i}" for i in range(10)]
    neg = ["n0", "n1", "n2"]
    rows = [{"cohort_id": c, "negative_control": False, "beta_cap_moved_dex": 0.8 if i < 8 else float("inf"),
             "rbar_cap_moved_dex": 0.0, "rbar_over_eta_cap_moved_dex": 0.01, "fit300": fit(1.0), "fit3000": fit(6.0)}
            for i, c in enumerate(kept)]
    rows += [{"cohort_id": c, "negative_control": True, "beta_cap_moved_dex": 0.0, "rbar_cap_moved_dex": 0.0,
              "rbar_over_eta_cap_moved_dex": 0.0, "fit300": fit(1.0), "fit3000": fit(1.0)} for c in neg]
    return rt({"status": "written", "list_ended": False, "kept": kept, "negative_controls": neg, "rows": rows,
               "n_moved_ge_0p5_dex": 10, "verdict": "box-limited", "rule": {"move_dex": 0.5, "need": 8, "of": 10}})


def test_kmax_rule_needs_the_negative_controls_kappa_300_fit_and_a_consistent_verdict():
    R.check_kmax(kmax_result())
    d = kmax_result()
    d["rows"][-1]["fit300"] = None                                     # sr-manager gate 4: the old confirm step
    with pytest.raises(FAIL):
        R.check_kmax(d)
    d = kmax_result()
    d["verdict"] = "not shown box-limited"
    with pytest.raises(FAIL):
        R.check_kmax(d)
    d = kmax_result()
    d["list_ended"] = True
    with pytest.raises(FAIL):
        R.check_kmax(d)


# ------------------------------------------------------------------------------------------------ coverage
BOX = {"eta": (1e-12, 1e3), "beta": (1e-12, 1e3), "epsilon": (1e-12, 1e3), "kappa": (1e-3, 300.0)}
TRUTH = {"eta": 1e-4, "beta": 3e-3, "epsilon": 2e-3, "kappa": 0.5}


def _claims(miss=()):
    """Closed claims on every quantity around the truth (x/2 to x*2), except `miss`, which sit far below it."""
    from sraging.claims import combinations as K
    out = []
    for q in R.COMBINATIONS | {"eta", "beta", "epsilon", "kappa"}:
        v = TRUTH[q] if q in TRUTH else K.value(q, TRUTH)
        f = 1e-3 if q in miss else 1.0
        out.append({"parameter": q, "shape": "closed", "lower": v * f / 2, "upper": v * f * 2, "point": v * f,
                    "label": "", "label_info": None, "widening": None})
    return out


def coverage_rows(n=56, cells=(("profile_jax2", "pure"), ("profile_jax2", "censor"), ("profile_jax2_h", "hazard")),
                  miss_every=0, miss=("rbar",), decline=()):
    done, jobs = {}, []
    for cal, arm in cells:
        for i in range(n):
            cid = f"{arm}-{i}"
            jobs.append((cid, cal))
            m = miss if miss_every and i % miss_every == 0 else ()
            cl = [c for c in _claims(m) if c["parameter"] not in decline]
            done[(cid, cal)] = {"cohort_id": cid, "calibrator": cal, "arm": arm, "cluster": f"{arm}-k{i // 4}",
                                "informative": True, "truth": TRUTH, "status": "ok", "claims": cl,
                                "combinations": {}, "cpu_s": 1.0, "negligible_floor": {"why": "synthetic"},
                                "host": {"node": f"cn{i % 3}", "cpu": "AMD EPYC 9754 128-Core Processor"}}
    return jobs, done


VALID = {"valid": [f"{arm}-{i}" for arm in ("pure", "censor", "hazard") for i in range(0, 56, 2)],
         "source": "/tester/reports/validity/validity.csv", "sha256": "0" * 64, "n": 168}


def test_coverage_rule_passes_what_the_tool_writes_for_a_good_run():
    jobs, done = coverage_rows()
    d = rt(CV.summarise(jobs, done, BOX, [], {"cells": {c: {} for c in R.GATED_CELLS}}, valid=VALID))
    assert d["status"] == "pass", d["reasons"]
    R.check_coverage(d)


def test_coverage_rule_needs_every_scored_fit_on_an_amd_epyc_host():
    """Naveh 2026-10-09 ([coverage_host] cpu_model): one CPU family for every scored fit; an Intel row fails it."""
    jobs, done = coverage_rows()
    done[jobs[0]]["host"] = {"node": "cn430", "cpu": "Intel(R) Xeon(R) Gold 5320 CPU @ 2.20GHz"}
    d = rt(CV.summarise(jobs, done, BOX, [], {"cells": {c: {} for c in R.GATED_CELLS}}, valid=VALID))
    assert d["status"] == "pass" and len(d["fit_hosts"]["rows_by_cpu"]) == 2
    with pytest.raises(FAIL):
        R.check_coverage(d)


def test_coverage_reports_the_valid_cohorts_beside_the_gate_which_counts_all():
    """Naveh 2026-10-08 ([coverage_validity]): the gate on every cohort; absolute coverage again on the valid ones.
    Planted: every 4th cohort misses r-bar; the valid half (even ids) holds every 4th of 0, 2, 4, ... i.e. half of
    them miss."""
    jobs, done = coverage_rows(miss_every=4)
    d = rt(CV.summarise(jobs, done, BOX, [], {"cells": {c: {} for c in R.GATED_CELLS}}, valid=VALID))
    q = d["scores"]["profile_jax2|pure"]["quantities"]["rbar"]
    assert q["closed_n"] == 56 and abs(q["closed_coverage"] - 42 / 56) < 1e-12
    assert q["valid"]["closed_n"] == 28 and abs(q["valid"]["closed_coverage"] - 14 / 28) < 1e-12
    assert d["scores"]["profile_jax2|pure"]["n_valid"] == 28 and d["validity"]["n_valid"] == len(VALID["valid"])
    jobs, done = coverage_rows()
    with pytest.raises(FAIL):                                       # no validity recorded: not the phase's result
        R.check_coverage(rt(CV.summarise(jobs, done, BOX, [], {"cells": {c: {} for c in R.GATED_CELLS}})))


def test_coverage_is_not_a_pass_when_the_gate_has_too_few_cells_or_clusters():
    """sr-manager gate 5, sr-naveh review 4: 'not measured' and a gate on 3 cells never pass."""
    jobs, done = coverage_rows(n=20)                                   # 20 cells, 5 clusters
    d = rt(CV.summarise(jobs, done, BOX, [], {"cells": {c: {} for c in R.GATED_CELLS}}))
    assert d["status"] == "not_applicable" and d["underpowered"]
    with pytest.raises(FAIL):
        R.check_coverage(d)
    jobs, done = coverage_rows(decline=("rbar",))                      # a combination never claimed
    d = rt(CV.summarise(jobs, done, BOX, [], {"cells": {c: {} for c in R.GATED_CELLS}}))
    assert d["status"] == "not_applicable" and any(u[1] == "rbar" for u in d["underpowered"])


def test_coverage_fails_a_loss_and_needs_the_hazard_arm_under_h():
    jobs, done = coverage_rows(miss_every=3)                           # r-bar misses a third of the truths
    d = rt(CV.summarise(jobs, done, BOX, [], {"cells": {c: {} for c in R.GATED_CELLS}}))
    assert d["status"] == "fail" and {x[1] for x in d["losses"]} == {"rbar"}
    with pytest.raises(FAIL):
        R.check_coverage(d)
    jobs, done = coverage_rows(cells=(("profile_jax2", "pure"), ("profile_jax2", "censor"), ("profile_jax2", "hazard")))
    d = rt(CV.summarise(jobs, done, BOX, [], {"cells": {c: {} for c in R.GATED_CELLS}}))
    assert d["status"] == "not_applicable" and d["absent_gated_cells"] == ["profile_jax2_h|hazard"]
    assert not d["scores"]["profile_jax2|hazard"]["gated"]            # reported, never gated
    with pytest.raises(FAIL):
        R.check_coverage(d)


def test_a_recorded_fail_is_checked_to_be_that_fail_and_nothing_else():
    """Naveh 2026-10-10 ([coverage_result]): a recorded fail holds only for its head, losses and underpowered count;
    another loss, a different head, or a run that now passes no longer matches it."""
    jobs, done = coverage_rows(miss_every=3)                           # r-bar loses on all three cells
    d = rt(CV.summarise(jobs, done, BOX, [], {"cells": {c: {} for c in R.GATED_CELLS}}, valid=VALID))
    d["repo_head"] = "a" * 40
    losses = sorted(list(x[:3]) for x in d["losses"])
    assert d["status"] == "fail" and {x[1] for x in d["losses"]} == {"rbar"} and len(losses) == 6
    rec = {"status": "fail", "head": "a" * 40, "losses": losses, "n_underpowered": len(d["underpowered"])}
    R.check_coverage(d, recorded=rec)
    for bad in ({**rec, "losses": losses[1:]}, {**rec, "head": "b" * 40}, {**rec, "n_underpowered": 1}):
        with pytest.raises(FAIL):
            R.check_coverage(d, recorded=bad)
    jobs, done = coverage_rows()                                       # a later run that passes: the record is stale
    good = rt(CV.summarise(jobs, done, BOX, [], {"cells": {c: {} for c in R.GATED_CELLS}}, valid=VALID))
    good["repo_head"] = "a" * 40
    with pytest.raises(FAIL):
        R.check_coverage(good, recorded=rec)


def test_coverage_rule_rejects_the_old_vacuous_fields():
    jobs, done = coverage_rows()
    d = rt(CV.summarise(jobs, done, BOX, [], {"cells": {c: {} for c in R.GATED_CELLS}}, valid=VALID))
    bad = copy.deepcopy(d)
    bad["scores"]["profile_jax2|pure"]["quantities"]["rbar"]["wilson95"] = [None, None]
    with pytest.raises(FAIL):
        R.check_coverage(bad)
    bad = copy.deepcopy(d)
    bad["scores"]["profile_jax2|pure"]["quantities"]["rbar"]["gate"]["vs_eta"]["n_cells"] = 3
    with pytest.raises(FAIL):
        R.check_coverage(bad)


def test_a_fit_that_raised_in_the_frozen_code_is_a_decline_not_missing():
    jobs, done = coverage_rows()
    k = jobs[0]
    done[k] = {**{f: done[k][f] for f in ("cohort_id", "calibrator", "arm", "cluster", "informative", "truth")},
               "error": "FloatingPointError: x", "error_origin": "frozen"}
    d = rt(CV.summarise(jobs, done, BOX, [], {}))
    assert not d["missing"] and d["frozen_code_failures"]
    q = d["scores"]["profile_jax2|pure"]["quantities"]["eta_over_eps"]
    assert q["shape_mix"]["failed"] == 1 and q["decline_rate"] == pytest.approx(1 / 56)
    del done[k]
    assert rt(CV.summarise(jobs, done, BOX, [], {}))["status"] == "fail"            # truly missing: a fail


# ------------------------------------------------------------------------------------------------ speed
def speed_result(start=R.LOAD_MAX - 0.6, end=R.LOAD_MAX + 0.3):
    return {"status": "pass", "ci95": [0.99, 1.02], "cores": {"logical": 15},
            "load_max": R.LOAD_MAX, "own_load": R.OWN_LOAD,
            "load_samples": [{"at": "start", "load": [start, 0.5, 0.5]}, {"at": "end", "load": [end, 1.0, 0.8]}]}


def checks_result():
    death = {"row": 1, "age": 0.0, "mass_refined": 4.6e-37, "ll_share": 0.041, "category": "impossible"}
    return rt({"status": "pass",
               "itp_age0": {"ok": True, "death": death, "listed_at": "theta_hat_without",
                            "absorbed": {"age0_deaths_fires": True, "listed_at_own_theta_hat": False,
                                         "shift_dex": {"eta": 1.38, "beta": 1.64, "epsilon": 1.58}}},
               "censor_tie": {"ok": True, "message": "... jax2's relabel did not apply ...", "cohort": "x"}})


def test_checks_rule_is_the_amended_sc009():
    R.check_checks(checks_result())
    for plant in (lambda d: d["itp_age0"].update(listed_at="theta_hat_with"),
                  lambda d: d["itp_age0"]["absorbed"].update(age0_deaths_fires=False),
                  lambda d: d["itp_age0"]["death"].update(mass_refined=2.3e-6),
                  lambda d: d["censor_tie"].update(message="CENSOR_TIE: relabelled"),
                  lambda d: d.update(status="fail")):
        d = checks_result()
        plant(d)
        with pytest.raises(FAIL):
            R.check_checks(d)


def test_speed_on_a_busy_machine_is_not_applicable_never_a_fail():
    R.check_speed(speed_result())
    L = R.LOAD_MAX
    for d in (speed_result(start=L + 0.4), speed_result(end=L + 1.6), {"status": "pass", "ci95": [0.99, 1.02]}):
        with pytest.raises(NA):
            R.check_speed(d)
    with pytest.raises(FAIL):
        R.check_speed(dict(speed_result(), ci95=[1.01, 1.05]))
    with pytest.raises(FAIL):                        # a run made under another busy rule is not judged by this one
        R.check_speed(dict(speed_result(), load_max=1.0))


# ------------------------------------------------------------------------------------------------ unchanged
def claims_001_result():
    return {"status": "pass", "claims_status": "pass", "rerun_required": True,
            "untouched": {"violations": []},
            "closure": {"files": ["src/sraging/calibrators/profile_v033.py"],
                        "changed_vs_base": ["src/sraging/result.py"]},
            "checks": {k: {"passed": 40, "total": 40} for k in ("claims_identity", "theta_hat", "value_bytes",
                                                               "grad_bytes")},
            "srbench_record": {"status": "not_applicable", "reason": "not re-run"}}


def test_claims_001_rule_needs_the_rerun_when_the_closure_changed():
    R.check_claims_001(claims_001_result())
    d = claims_001_result()
    d["checks"]["claims_identity"]["passed"] = 39
    with pytest.raises(FAIL):
        R.check_claims_001(d)
    d = claims_001_result()
    d["untouched"]["violations"] = ["M\tsrc/sraging/engine/grid.py"]
    with pytest.raises(FAIL):
        R.check_claims_001(d)
    d = dict(claims_001_result(), claims_status="not re-run: closure unchanged", checks={})
    with pytest.raises(FAIL):
        R.check_claims_001(d)
    R.check_claims_001(dict(d, rerun_required=False, closure={**d["closure"], "changed_vs_base": []}))


# ------------------------------------------------------------------------------------------------ coverage on WEXAC
def test_the_shards_partition_the_jobs_in_order():
    """Naveh 2026-10-08 ([coverage_host]): one LSF array element per shard; together they fit every job once."""
    jobs = [(f"c{i}", "profile_jax2") for i in range(215)]
    for of in (1, 7, 215):
        parts = [CV.shard_jobs(jobs, i, of) for i in range(1, of + 1)]
        assert sorted(j for p in parts for j in p) == sorted(jobs) and sum(map(len, parts)) == len(jobs)
    with pytest.raises(ValueError):
        CV.shard_jobs(jobs, 0, 7)


def test_the_mac_subset_is_five_per_gated_cell_and_seeded():
    jobs = CV.panel_jobs()
    x = CV.xhost_jobs(jobs)
    assert len(x) == 15 and x == CV.xhost_jobs(jobs) and set(x) <= set(jobs)
    for cal, arm in CV.GATED:
        assert sum(1 for c, k in x if k == cal and c.startswith(arm + "-")) == 5, (cal, arm)


def test_the_cross_host_comparison_counts_shapes_flips_and_end_shifts():
    jobs, done = coverage_rows(n=4, cells=(("profile_jax2", "pure"),))
    mac = [done[j] for j in jobs]
    cl = copy.deepcopy(mac)
    c = next(x for x in cl[0]["claims"] if x["parameter"] == "eta")
    c["lower"], c["upper"] = c["lower"] * 10 ** 0.02, c["upper"] * 10 ** -0.01          # ends move, still a hit
    c = next(x for x in cl[1]["claims"] if x["parameter"] == "eta")
    c["shape"], c["lower"] = "upper_bound", None                                        # shape differs
    c = next(x for x in cl[2]["claims"] if x["parameter"] == "epsilon")
    c["lower"], c["upper"] = c["lower"] * 1e3, c["upper"] * 1e3                          # the hit flips
    cl[3] = dict(cl[3], status="failed")
    out = CV.xhost_compare(list(zip(mac, cl)))
    e, p = out["eta"], out["epsilon"]
    assert e["n"] == 3 and e["shape_differs"] == 1 and e["hit_flips"] == 0 and e["failed_on_one_host"] == 1
    assert abs(e["end_shift_dex_max"] - 0.02) < 1e-12 and e["n_ends"] == 5            # 2 + 1 (upper only) + 2
    assert p["hit_flips"] == 1 and abs(p["end_shift_dex_max"] - 3.0) < 1e-12


def test_import_rows_takes_this_heads_rows_once_and_lists_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(CV, "ROWS", str(tmp_path / "rows.jsonl"))
    monkeypatch.setattr(CV, "panel_jobs", lambda panel=None: [("a", "profile_jax2"), ("b", "profile_jax2"),
                                                               ("c", "profile_jax2_h")])
    h = {"node": "cn1", "cpu": "x"}
    ok = [{"cohort_id": c, "calibrator": k, "head": "H", "host": h, "status": "ok"} for c, k in
          (("a", "profile_jax2"), ("b", "profile_jax2"))]
    err = {"cohort_id": "c", "calibrator": "profile_jax2_h", "head": "H", "host": h, "error": "Killed",
           "error_origin": "harness"}
    p1 = tmp_path / "s1.jsonl"
    p1.write_text("".join(json.dumps(r) + "\n" for r in ok + [ok[0], err]))
    rep = CV.import_rows([str(p1)], head="H")
    assert rep["imported"] == 2 and rep["errors"][0][:3] == ["c", "profile_jax2_h", "harness"]
    assert rep["missing"] == [["c", "profile_jax2_h"]]
    assert CV.import_rows([str(p1)], head="H")["imported"] == 0                      # not doubled
    p2 = tmp_path / "s2.jsonl"
    p2.write_text(json.dumps(dict(ok[0], head="OTHER")) + "\n")
    with pytest.raises(SystemExit):
        CV.import_rows([str(p2)], head="H")
    p3 = tmp_path / "s3.jsonl"
    p3.write_text(json.dumps({k: v for k, v in ok[0].items() if k != "host"}) + "\n")
    with pytest.raises(SystemExit):                                                    # a row must say where it ran
        CV.import_rows([str(p3)], head="H")


def test_coverage_records_where_its_fits_ran():
    jobs, done = coverage_rows()
    d = rt(CV.summarise(jobs, done, BOX, [], {"cells": {c: {} for c in R.GATED_CELLS}}, valid=VALID))
    assert d["fit_hosts"] == {"n_nodes": 3, "rows_by_cpu": {"AMD EPYC 9754 128-Core Processor": 168},
                              "rows_without_host": 0}
    bad = copy.deepcopy(d)
    bad["fit_hosts"]["rows_without_host"] = 1
    with pytest.raises(FAIL):
        R.check_coverage(bad)
