"""Checks on spec 002's COMMITTED acceptance results (specs/002-fp-engine/contracts/acceptance.md). Fast: reads
files only, runs nothing.

A missing result, a development result (dirty tree, uncommitted manifest, or a tree that changed during the
run) or a smoke run is NOT APPLICABLE, never a pass (constitution V).
"""
import json
import os
import subprocess

import pytest

from conftest import not_applicable

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO, "tests", "acceptance", "results")


def load(name):
    p = os.path.join(RES, f"{name}.json")
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


def ancestor(commit, head):
    return subprocess.run(["git", "-C", REPO, "merge-base", "--is-ancestor", commit, head]).returncode == 0


def test_t0_the_engine_is_v033_with_every_feature_off():
    d = load("t0_002")
    assert d["status"] == "pass"
    assert d["checks"]["program_equal"] == {"passed": 40, "total": 40}
    assert d["checks"]["bytes"] == {"passed": 240, "total": 240}
    assert all(all(r["program_equal"].values()) for r in d["results"])       # port AND frozen oracle


def test_guard_pilot_froze_the_registers_constants():
    d = load("guard_pilot_002")
    from sraging import divergences
    by_id = {e.id: e.condition for e in divergences.load()}
    for eid in ("D-004", "D-106"):
        assert by_id[eid]["c_stiff"] == pytest.approx(d["c_stiff"], rel=1e-4)
        assert by_id[eid]["floor_ulps"] == pytest.approx(d["floor_ulps"], rel=1e-4)
    assert d["n_stiff"] >= 10                     # not one point (review T064 N2)


def test_guard_passed_and_every_plant_failed():
    d = load("guard_002")
    assert d["status"] == "pass"
    assert d["checks"]["guard"]["passed"] == d["checks"]["guard"]["total"] >= 60 + 18
    assert d["planted"] and all(p["failed"] and p["n_violations"] > 0 for p in d["planted"].values())
    assert {"censored", "hazard", "use_F", "stop_gradient_F"} <= set(d["planted"])
    assert {r["feature"] for r in d["results"] if "feature" in r} == {"hazard", "window", "het", "exact", "entry",
                                                                      "segments"}
    pilot = load("guard_pilot_002")
    assert ancestor(pilot["repo_head"], d["repo_head"])          # the constants were frozen before the guard ran


# Naveh's decisions of 2026-10-04 (spec.md SC-002, SC-005, SC-007, SC-010 amendments): the failures below are
# recorded limitations, held to the measured magnitudes; anything else failing is a failure.
STIFF_T1 = "realfull-real-Sweden_F_1910_homo-c0-s1"
STIFF_MPREF = {("pure-real-Denmark_F_1900_hetro-c0-s0", "truth"), ("censor-real-Denmark_M_1890_hetro-c0-s1", "truth"),
               ("censor-real-Denmark_F_1890_hetro-c0-s1", "truth"), (STIFF_T1, "eta x1.5"),
               (STIFF_T1, "epsilon x1.5")}


def test_t1_meets_its_thresholds_but_for_the_recorded_limitations():
    d = load("t1_002")
    assert d["thresholds"] == {"total": 1e-5, "term": 1e-8, "dS": 1e-8, "grad_rel": 1e-6}
    s = d["summary"]
    assert s["evaluations"] == 480 and s["largest_n"] >= 90000
    cd_fails = 0
    for r in d["results"]:
        assert "error" not in r, r.get("error")
        assert "n_routed" in r and "cut" in r                    # gate condition 3
        assert r["guard"]["ok"] is True                           # gate condition 4
        if r["grad_rel_fixed_length"] is not None:
            assert r["grad_rel_fixed_length"] <= 1e-6
        for f in r["fails"]:
            if f.startswith("total"):                             # float64 conditioning (SC-002 amendment)
                assert r["case"] == STIFF_T1 and abs(r["d_total"]) <= 2e-4 and r["K"] >= 1e5, (r["case"], f)
            elif f.startswith("central differences"):            # piecewise smooth: the gradient is exact
                cd_fails += 1
                assert r["grad_rel_fixed_length"] is not None or r["kind"] == "segments", (r["case"], f)
            else:
                raise AssertionError((r["case"], r["scheme"], r["theta"], f))
    assert cd_fails <= 2
    assert s["d107_window_cohorts"]                               # gate condition 6
    assert ancestor(load("t1_pilot_002")["repo_head"], d["repo_head"])


def test_mpref_v1_is_exact_but_at_the_recorded_stiff_points():
    d = load("mpref_002")
    assert d["routed"] and all(r["ok"] for r in d["routed"])
    assert max(r["sum_abs_dl_v1"] for r in d["routed"]) <= 1e-5
    for r in d["roundoff"]:
        if not r["ok"]:                                           # SC-007 amendment
            assert (r["cohort"], r["theta"]) in STIFF_MPREF, (r["cohort"], r["theta"])
            assert abs(r["dl_v1"]) <= 5e-4 and r["spread_v1"] <= 5e-4
    sweden = next(r for r in d["roundoff"] if r["theta"] == "stored")
    assert abs(sweden["dl_v1"]) <= 1e-5 and sweden["dl_v033"] > 80     # the round-off D-004 removes


def test_early_residuals_are_measured_and_all_but_nine_references_converged():
    d = load("early_002")
    rows = d["results"]
    n = sum(r["n_early"] for r in rows)
    conv = sum(r["n_converged"] for r in rows)
    assert n == 1322 and n - conv <= 9                            # SC-005 amendment
    for r in rows:
        assert r["residual_v1"]["max_abs"] >= 0 and r.get("max_halving_move_unconverged", 0) <= 5.0


def test_floors_matched_but_for_the_accepted_itp_case():
    d = load("floors_002")
    res = d["results"]
    assert all(v["ok"] for k, v in res.items() if k != "itp_plus_age0")
    itp = res["itp_plus_age0"]
    assert itp["fired"] and itp["cli_exit"] == 0
    post = json.load(open(os.path.join(RES, "floors_posthoc_002.json")))
    assert post["match"] and post["predicted"] == itp["got"]       # SC-010 amendment: verified post hoc


def test_score_features_pass_or_are_not_informative():
    d, w = load("score_002"), load("score_window_002")
    f = d["features"]
    assert {k: f[k]["verdict"] for k in ("pure", "hazard", "entry")} == {"pure": "pass", "hazard": "pass",
                                                                          "entry": "pass"}
    assert all(f[k]["engine_grid_certified"] for k in ("pure", "hazard", "entry"))
    assert f["entry"]["plant"]["detected"]
    assert w["features"]["window"]["verdict"] == "pass" and w["features"]["window"]["plant"]["detected"]
    h = load("score_het_002")["features"]["heterogeneity"]
    assert h["verdict"] in ("pass", "not informative at this R") and h["engine_grid_certified"]


def test_claims_001_rerun_is_identical_and_valid_for_this_branch():
    """Spec 001's claims, re-run on this branch (gate condition 10 and c): identical claims, theta-hat and bytes
    on 40/40; srbench's record is not applicable (the tester moved). The run stands only if nothing on the fit
    path before the FloorReport changed between its HEAD and this one."""
    d = load("claims-001_002")
    assert d["status"] == "pass"
    c = d["checks"]
    for name in ("claims_identity", "theta_hat", "value_bytes", "grad_bytes"):
        assert c[name]["passed"] == c[name]["total"] == 40, name
    assert d["srbench_record"]["status"] == "not_applicable"
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    fit_path = ["src/sraging/calibrators/profile_v033.py", "src/sraging/compat/__init__.py", "src/sraging/compat/v033",
                "src/sraging/cohort.py",
                "src/sraging/params.py", "src/sraging/result.py", "src/sraging/registry.py",
                "src/sraging/provenance.py", "src/sraging/adapters", "src/sraging/labels.py"]
    r = subprocess.run(["git", "-C", REPO, "diff", "--name-only", d["repo_head"], head, "--", *fit_path],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"cannot compare with the re-run's commit {d['repo_head']}: {r.stderr.strip()}"
    if r.stdout.split():
        # Spec 003 changes the fit path (result.py schema /4, registry, labels, adapters), so this re-run no longer
        # stands for HEAD. It is NOT APPLICABLE -- never a pass -- until spec 003's re-run exists; that re-run
        # (results/claims-001_003.json) is checked by tests/acceptance/test_results_003.py (spec 003 SC-011).
        not_applicable(f"the fit path changed since spec 002's claims re-run ({r.stdout.split()}); spec 003's re-run "
                       "(results/claims-001_003.json, test_results_003.py) is the check now")


@pytest.mark.parametrize("name", ["anchor_002", "speed_002"])
def test_later_phases_pass(name):
    d = load(name)
    assert d["status"] == "pass", (name, d.get("status"))
