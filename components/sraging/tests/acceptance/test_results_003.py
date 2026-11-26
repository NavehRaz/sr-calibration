"""Checks on spec 003's COMMITTED acceptance results (specs/003-calibrator/contracts/acceptance.md; task T045). Fast:
reads files only, runs nothing.

A missing result, a development result (dirty tree, uncommitted manifest, or a tree that changed during the run), or a
smoke run is NOT APPLICABLE, never a pass (constitution V). Each pass rule is a `check_*` function of the result alone,
so tests/unit/test_results_003_rules.py can show that it fails on planted results.
"""
import json
import math
import os
import subprocess

import pytest

from conftest import not_applicable

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO, "tests", "acceptance", "results")
COMBINATIONS = {"eta_over_eps", "rbar_over_eta", "sqrt_eta_over_eps", "beta_over_eta_kp1", "rbar", "beta_over_kp1"}
GATED_CELLS = {"profile_jax2|pure", "profile_jax2|censor", "profile_jax2_h|hazard"}
LEAD_DEX = 0.05                                   # FR-016's statistic (Naveh, 2026-10-05)
NEG_FLOORS_MD5 = "609c91996f3696546bc34051255a93bc"
LOAD_MAX, OWN_LOAD = 5.0, 1.0                     # sr-manager gate 7: a busy machine is NOT APPLICABLE; the bar is
                                                  # spec 002's (Naveh 2026-10-10, decisions_003.toml [speed_load])


def load(name):
    p = os.path.join(RES, f"{name}_003.json")
    if not os.path.exists(p):
        not_applicable(f"{os.path.relpath(p, REPO)} does not exist yet")
    d = json.load(open(p))
    if d.get("status") == "not_applicable":
        not_applicable(f"{name}: {d.get('reasons')}")
    if str(d.get("status", "")).startswith("smoke"):
        not_applicable(f"{name} is a smoke run, not the acceptance run")
    if d.get("status_for_acceptance", "acceptance") != "acceptance":
        not_applicable(f"{name} is a development run (dirty tree or uncommitted manifest)")
    return d


def ancestor(commit, head="HEAD"):
    return subprocess.run(["git", "-C", REPO, "merge-base", "--is-ancestor", commit, head]).returncode == 0


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True)


# ------------------------------------------------------------------------------------------------ pass rules
def ridge_invariant(key):
    """Invariant by construction on the Gompertz scaling ridge (eta, beta, eps) -> l (eta, beta, eps)."""
    from sraging.claims import combinations as K
    z = [math.log(1e-4), math.log(3e-3), math.log(2e-3), math.log(0.5)]
    zl = [z[0] + 1.7, z[1] + 1.7, z[2] + 1.7, z[3]]
    return abs(K.log_g(key, z) - K.log_g(key, zl)) < 1e-9


def check_order(o):
    """FR-016 / SC-008: the recorded order is what the pre-registered rule gives on the recorded spreads (never a
    hard-coded order): the lead group is the ridge-invariant combinations whose validity-valid kappa-slice spread is
    <= 0.05 dex, each group by increasing spread; equal spreads are unresolved ties and do not pass."""
    from sraging.claims import combinations as K
    assert o["status"] == "written", o["status"]
    s = o["spread_valid_panel_dex"]
    assert set(s) == COMBINATIONS and set(o["order"]) == COMBINATIONS and len(o["order"]) == len(COMBINATIONS)
    for k, v in s.items():
        assert o["all_sets"]["panel valid"][k][0] == v, k              # the spread is the valid panel's median
    lead = {k for k in COMBINATIONS if ridge_invariant(k) and s[k] <= LEAD_DEX}
    n = len(lead)
    assert set(o["lead_group"]) == lead and list(o["order"][:n]) == list(o["lead_group"])
    for group in (o["order"][:n], o["order"][n:]):
        assert all(s[a] < s[b] for a, b in zip(group, group[1:])), group
    hyp = list(K.HYPOTHESIS_ORDER)
    assert [list(x) for x in o["departures_from_hypothesis"]] == [
        [i, a, b] for i, (a, b) in enumerate(zip(o["order"], hyp)) if a != b]


def check_floor(f):
    """FR-022 / T037: the coefficients are measured from the md5-pinned neg_floors.json (sr-manager gate 10): c is the
    mean over uncensored n of floor * sqrt(n) (all censored: the smallest, a lower bound), r-bar's is c f(kappa)."""
    assert f["status"] == "written" and f["sources"]["neg_floors.json"]["md5"] == NEG_FLOORS_MD5
    assert set(f["regimes"]) == {"sharp", "mid", "kappa_large", "diffuse"}
    for name, r in f["regimes"].items():
        k = r["kappa"]
        assert r["rbar_factor"] == pytest.approx(1.0 - k * math.log1p(1.0 / k), rel=1e-12), name
        assert r["c_rbar"] == pytest.approx(r["c_beta"] * r["rbar_factor"], rel=1e-12), name
        by_n = {int(n): v for n, v in r["c_beta_by_n"].items()}
        used = [v for n, v in by_n.items() if n not in r["censored_n"]]
        if r["lower_bound"]:
            assert not used and r["c_beta"] == pytest.approx(min(by_n.values()), rel=1e-12), name
        else:
            assert used and r["c_beta"] == pytest.approx(sum(used) / len(used), rel=1e-12), name


def cost_decided(cd):
    """Naveh's combination-cost decision as the walker result carries it (tests/acceptance/decisions_003.toml
    [combination_cost]: decision, date, by)."""
    return (isinstance(cd, dict) and bool(cd.get("decision")) and bool(cd.get("date"))
            and "naveh" in str(cd.get("by", "")).lower())


def check_walker(d):
    """SC-004: no endpoint narrower than brute force by > 0.01 dex, none wider by > 0.05 dex. The cost: <= the limit
    (2x the fit's own), or above it only with Naveh's decision recorded in the result (`cost_decision`)."""
    assert d["checks"]["passed"] == d["checks"]["total"] > 0 and not d["failures"] and not d["errors"]
    assert not d.get("archive_failures")                          # the free archive check (sr-manager gate 3b)
    assert d["cost_ratio_median"] is not None and d["cost_limit"] is not None
    if d["cost_ratio_median"] > d["cost_limit"] and not cost_decided(d.get("cost_decision")):
        pytest.fail(f"combinations cost {d['cost_ratio_median']:.2f}x the fit (> {d['cost_limit']}x): Naveh's decision "
                    f"must be recorded (walker result `cost_decision`; spec Clarifications) before this passes")


def check_kmax(d):
    """SC-007, FR-025: 10 kept (+3 negative controls), each with its kappa_max 300 and 3000 fits; the rule's verdict
    recomputed from the rows; the r-bar and r-bar/eta caps recorded in both fits."""
    assert d["status"] == "written" and not d["list_ended"]
    assert len(d["kept"]) == 10 and len(d["negative_controls"]) == 3
    rows = {r["cohort_id"]: r for r in d["rows"]}
    assert set(rows) == set(d["kept"]) | set(d["negative_controls"]) and len(rows) == 13
    for c, r in rows.items():
        assert r["fit300"] is not None and r["fit3000"] is not None, c
        for fit in (r["fit300"], r["fit3000"]):
            assert {"rbar", "rbar_over_eta"} <= set(fit["combos"]), c
            for key in ("rbar", "rbar_over_eta"):
                assert "ci_hi" in fit["combos"][key], (c, key)
        assert "rbar_cap_moved_dex" in r and "rbar_over_eta_cap_moved_dex" in r, c
    for c in d["kept"]:
        assert rows[c]["beta_cap_moved_dex"] is not None, c
    n = sum(1 for c in d["kept"] if rows[c]["beta_cap_moved_dex"] >= d["rule"]["move_dex"])
    assert n == d["n_moved_ge_0p5_dex"]
    assert d["verdict"] == ("box-limited" if n >= d["rule"]["need"] else "not shown box-limited")


def check_coverage(d, recorded=None):
    """SC-005 (Naveh 2026-10-05; sr-manager gate 5; sr-naveh review 4): every combination, on each gated cell
    ((profile_jax2, pure), (profile_jax2, censor), (profile_jax2_h, hazard)), paired against the same fits' eta and
    epsilon claims on >= 30 cells over >= 10 clusters: no z < -2 (and, only if Naveh had set a margin, a 95% CI lower
    end above it; he set none, decisions_003.toml [coverage_gate]); the absolute coverage, its Wilson interval, the
    same on the validity-valid cohorts (Naveh 2026-10-08, [coverage_validity]) and the tester's other axes reported;
    the power computed first.

    `recorded` is a fail Naveh recorded (decisions_003.toml [coverage_result]): the result must then be exactly that
    one -- its head, its losses, its number of underpowered gaps -- with everything else reported as above. It is
    still not a pass; the test shows it as xfailed."""
    gaps = {(cell, q, side): s["verdict"] for cell, c in d["scores"].items() if c["gated"]
            for q in COMBINATIONS & set(c["quantities"]) for side, s in c["quantities"][q]["gate"].items()
            if side in ("vs_eta", "vs_epsilon")}
    if recorded is None:
        assert d["status"] == "pass" and not d["losses"] and not d["missing"]
        assert not d["underpowered"] and not d["absent_gated_cells"] and not d["prerequisites"]
    else:
        assert d["status"] == "fail" and d["repo_head"] == recorded["head"], (d["status"], d["repo_head"])
        assert sorted(list(x[:3]) for x in d["losses"]) == sorted(recorded["losses"]), d["losses"]
        assert sorted(list(k) for k, v in gaps.items() if v == "loses") == sorted(recorded["losses"])
        assert len(d["underpowered"]) == recorded["n_underpowered"] == sum(v == "underpowered" for v in gaps.values())
        assert not d["missing"] and not d["absent_gated_cells"] and not d["prerequisites"]
    sc = d["scores"]
    assert {cell for cell, c in sc.items() if c["gated"]} == GATED_CELLS
    assert set(d["power"]["cells"]) == GATED_CELLS
    for cell in GATED_CELLS:
        c = sc[cell]
        assert c["n_informative"] >= 50, cell
        assert COMBINATIONS <= set(c["quantities"]), cell
        for q in COMBINATIONS:
            qd = c["quantities"][q]
            g = qd["gate"]
            assert recorded is not None or g["verdict"] == "no loss", (cell, q, g["verdict"])
            for side in ("vs_eta", "vs_epsilon"):
                s = g[side]
                assert {"availability", "information"} <= set(qd["axes"][side]), (cell, q, side)
                if recorded is None:
                    assert s["verdict"] == "no loss", (cell, q, side)
                elif s["verdict"] == "underpowered":
                    assert s["n_cells"] < 30 or s["n_clusters"] < 10 or s["se"] is None, (cell, q, side)
                    continue
                assert s["verdict"] in ("no loss", "loses") and s["n_cells"] >= 30 and s["n_clusters"] >= 10, \
                    (cell, q, side)
                assert s["z"] is not None and s["ci_lo"] is not None, (cell, q, side)
                assert (s["z"] >= -2.0) == (s["verdict"] == "no loss"), (cell, q, side)
                assert s.get("margin") is None or s["ci_lo"] > -s["margin"], (cell, q, side)
            lo, hi = qd["wilson95"]
            if recorded is not None and qd["closed_n"] < 30:      # a recorded fail reports thin cells as they are
                assert qd["closed_n"] == 0 or (lo is not None and 0.0 <= lo <= hi <= 1.0), (cell, q)
            else:
                assert qd["closed_n"] >= 30 and lo is not None and hi is not None and 0.0 <= lo <= hi <= 1.0, (cell, q)
            assert qd["decline_rate"] is not None
            qv = qd.get("valid")
            assert qv and qv["closed_n"] <= qd["closed_n"] and len(qv["wilson95"]) == 2, (cell, q)
        assert 0 <= c.get("n_valid", -1) <= c["n_informative"], cell
    fh = d.get("fit_hosts") or {}
    assert fh.get("n_nodes", 0) > 0 and fh.get("rows_without_host") == 0, fh      # where the fits ran ([coverage_host])
    assert fh.get("rows_by_cpu") and all("AMD EPYC" in k for k in fh["rows_by_cpu"]), fh   # one family (2026-10-09)
    v = d.get("validity")
    assert v and len(v["sha256"]) == 64 and v["source"].endswith("validity.csv") and v["n_valid"] > 0
    assert "negligible" in sc["profile_jax2|pure"]["quantities"]["rbar_over_eta"]


def busy(samples):
    """The 1-minute load above LOAD_MAX at the start, or above LOAD_MAX + this run's own process later."""
    return [s for i, s in enumerate(samples) if s["load"][0] > LOAD_MAX + (0 if i == 0 else OWN_LOAD)]


def check_checks(d):
    """SC-009 as amended 2026-10-06 (Naveh): AGE0_DEATHS fires on ITP + age-0; FLOORED_DEATHS at the theta-hat fitted
    WITHOUT that death lists it, refined mass < 1e-12, with its share; the pull is recorded; CENSOR_TIE fires."""
    a, c = d["itp_age0"], d["censor_tie"]
    assert d["status"] == "pass" and a["ok"] and c["ok"], (a.get("ok"), c.get("ok"))
    assert a["listed_at"] == "theta_hat_without" and a["absorbed"]["age0_deaths_fires"]
    assert a["death"]["mass_refined"] < 1e-12 and a["death"]["ll_share"] is not None and a["death"]["age"] == 0.0
    assert set(a["absorbed"]["shift_dex"]) == {"eta", "beta", "epsilon"}
    assert "relabel did not apply" in (c["message"] or "")


def check_speed(d):
    """SC-010: the 95% CI of the port/oracle CPU ratio includes 1 and its upper limit is <= 1.03, on an idle machine
    (sr-manager gate 7: a busy machine is NOT APPLICABLE, never a fail)."""
    samples = d.get("load_samples")
    if not samples or "cores" not in d:
        not_applicable("speed records no load samples and core counts")
    assert (d["load_max"], d["own_load"]) == (LOAD_MAX, OWN_LOAD), "the run used another busy rule"
    if busy(samples):
        not_applicable(f"speed ran on a busy machine: {[(s['at'], s['load'][0]) for s in busy(samples)]}")
    lo, hi = d["ci95"]
    assert d["status"] == "pass" and lo <= 1.0 <= hi <= 1.03


def check_claims_001(d):
    """SC-011: compat/v033, the engine and the oracles untouched against main; spec 001's claims re-run identical on
    40/40, required whenever a file of profile_v033's import closure changed on this branch, and whenever an engine
    file is exempt by name (spec 004 T144: untouched["spec_exempt"] non-empty)."""
    assert d["status"] == "pass" and not d["untouched"]["violations"]
    assert d["closure"]["files"] and "src/sraging/calibrators/profile_v033.py" in d["closure"]["files"]
    if d["untouched"].get("spec_exempt"):                  # spec 004 T144: a named engine exemption needs the re-run
        assert d["rerun_required"], "engine files exempt by SPEC_ENGINE_CHANGES without the claims re-run"
    if d["closure"]["changed_vs_base"] or d["rerun_required"]:
        assert d["claims_status"] == "pass"
        for name in ("claims_identity", "theta_hat", "value_bytes", "grad_bytes"):
            c = d["checks"][name]
            assert c["passed"] == c["total"] == 40, name
        assert d["srbench_record"]["status"] == "not_applicable" and d["srbench_record"]["reason"]


# ------------------------------------------------------------------------------------------------ the tests
def test_pilot_froze_the_panel_before_the_runs_that_read_it():
    pilot = load("pilot")
    for name in ("bytes", "fits", "walker", "kmax", "coverage", "speed"):
        p = os.path.join(RES, f"{name}_003.json")
        if os.path.exists(p):
            d = json.load(open(p))
            if d.get("status_for_acceptance", "acceptance") != "acceptance" or str(d.get("status", "")).startswith("smoke"):
                continue                                   # a development or smoke run never counts (module doc)
            assert ancestor(pilot["repo_head"], d["repo_head"]), name


def test_bytes_every_value_and_gradient_byte_equal():
    """SC-001: 240/240 per name (and _h at two hazard rates), C. elegans points included, floored ceiling binding."""
    d = load("bytes")
    assert d["status"] == "pass" and not d["errors"]
    for name, n in (("jax2", 240 + 4), ("jax2_h", 480 + 8)):
        for k in ("value", "vg", "grid_equal"):
            c = d["checks"][name][k]
            assert c["passed"] == c["total"] == n, (name, k, c)
    assert d["celegans_floored"] and all(d["celegans_floored"])


def test_fits_identical_to_the_oracle_and_the_stored_mac_fits():
    """SC-002: theta-hat bytes, results, claims and widening identical on the quick tier and the mechanism list, both
    names; equal to the tester's stored Mac fits where they overlap; every listed mechanism fires. The fits run of
    2026-10-06 read R3d6/L5's NaN strings as strings (12 of its 18 stored comparisons) and capped each diff at 4, so
    when it fails, results/fits_stored_003.json (`fits --stored-only`: the port against the same stored fits, the
    strings decoded, every diff counted) stands for its two stored checks, and every other check must still pass."""
    d = load("fits")
    assert not d["errors"]
    stored_keys = ("stored_result_equal", "stored_claims_equal")
    st = None if d["status"] == "pass" else load("fits_stored")
    for k, c in d["checks"].items():
        if st is None or k not in stored_keys:
            assert c["passed"] == c["total"], (k, c)
    if st is not None:
        assert st["status"] == "pass" and not st["errors"] and ancestor(d["repo_head"], st["repo_head"])
        assert {(r["cohort_id"], r["name"]) for r in st["rows"]} == \
            {(r["cohort_id"], r["name"]) for r in d["rows"] if r.get("stored")}
        assert all(r["result_equal"] and r["claims_equal"] and r["n_diffs"] == 0 for r in st["rows"])
    import tomllib
    fits = tomllib.load(open(os.path.join(REPO, "tests", "acceptance", "panel_003.toml"), "rb"))["fits"]
    assert d["checks"]["theta_bytes_equal"]["total"] == 2 * len(set(fits["quick"]) | set(fits["mechanisms"]))
    assert d["checks"]["stored_result_equal"]["total"] >= 6
    assert all(d["mechanisms"].values()) and len(d["mechanisms"]) == 6


def test_adapter_scores_equal_the_testers():
    d = load("adapter")
    assert d["status"] == "pass"


def test_order_and_floor_were_frozen_before_coverage():
    """SC-008 and FR-022: the order follows the pre-registered rule on its recorded spreads, the floors are measured
    from the pinned sweep, and both are committed before the coverage run."""
    o, f = load("order"), load("neg_floor")
    check_order(o)
    check_floor(f)
    from sraging.claims import combinations as K
    assert list(K.order()[0]) == o["order"]                     # the frozen file is the measured order
    cov = load("coverage")
    assert ancestor(o["repo_head"], cov["repo_head"]) and ancestor(f["repo_head"], cov["repo_head"])


def test_walker_within_sc004_and_its_cost_decided():
    check_walker(load("walker"))


def test_kmax_study_ran_before_the_raw_beta_label_text():
    check_kmax(load("kmax"))


def test_coverage_no_combination_loses_on_a_gated_arm():
    """A fail Naveh recorded ([coverage_result]) is checked to be that result and shown as xfailed: never a pass."""
    import tomllib
    rec = tomllib.load(open(os.path.join(REPO, "tests", "acceptance", "decisions_003.toml"), "rb")).get("coverage_result")
    d = load("coverage")
    check_coverage(d, recorded=rec)
    if rec is not None:
        pytest.xfail(f"SC-005 fails as recorded: {rec['losses']} (Naveh {rec['date']}, [coverage_result])")


def test_checks_fire_as_specified():
    check_checks(load("checks"))


def test_speed_port_equals_the_frozen_code():
    check_speed(load("speed"))


def test_claims_001_rerun_is_identical_for_this_branch():
    """SC-011: written by the `unchanged` phase. The run stands only if nothing in its recorded import closure changed
    between its HEAD and this one (as tests/acceptance/test_results_002.py rules for spec 002's re-run)."""
    d = load("claims-001")
    check_claims_001(d)
    r = git("diff", "--name-only", d["repo_head"], "HEAD", "--", *d["closure"]["files"])
    assert r.returncode == 0, f"cannot compare with the re-run's commit {d['repo_head']}: {r.stderr.strip()}"
    if r.stdout.split():
        not_applicable(f"the claims' import closure changed since the re-run ({r.stdout.split()}): re-run `unchanged`")
