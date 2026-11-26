"""Checks on the COMMITTED acceptance results (T046). Fast: reads files only, runs nothing.

A missing results file is NOT APPLICABLE, never a pass (constitution V).
"""
import json
import os
import subprocess

import pytest

from conftest import not_applicable

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO, "tests", "acceptance", "results")


def load(phase):
    p = os.path.join(RES, f"{phase}.json")
    if not os.path.exists(p):
        not_applicable(f"{os.path.relpath(p, REPO)} does not exist yet (run tools/acceptance.py {phase})")
    d = json.load(open(p))
    if d.get("status") == "not_applicable":
        not_applicable(f"{phase}: {d.get('reasons')}")
    if str(d.get("status", "")).startswith("smoke"):
        not_applicable(f"{phase}.json is a smoke run (--only / panel not committed), not the acceptance run")
    return d


def test_claims_phase_passed_with_expected_N():
    d = load("claims")
    assert d["status"] == "pass"
    c = d["checks"]
    assert c["claims_identity"]["passed"] == c["claims_identity"]["total"] == 40
    assert c["value_bytes"]["passed"] == c["grad_bytes"]["passed"] == 40
    assert sum(r["n_points"] for r in d["results"]) == 240                    # 30 x 6 + 10 x 6
    assert c["cohort_arrays_direct"]["passed"] == c["cohort_arrays_csv"]["passed"] == 40
    assert c["srbench_record"]["passed"] == c["srbench_record"]["total"] == 30   # each computed in the run
    assert c["theta_hat"]["passed"] == c["theta_hat"]["total"] == 40             # every parameter, review N3
    assert c["role_reconfirmed"]["passed"] == c["role_reconfirmed"]["total"] == 10
    assert c["tester_tree_unchanged"]["passed"] == 1
    assert d["tester"]["head"].startswith("f8eaed2")
    assert sum(len(r["point_values"]) for r in d["results"]) == 240              # stored values, review N2


def test_claims_ran_after_the_panel_was_frozen():
    d = load("claims")
    commit = d["panel_commit"]
    r = subprocess.run(["git", "-C", REPO, "merge-base", "--is-ancestor", commit, d["repo_head"]])
    assert r.returncode == 0, "the claims run's HEAD does not contain the panel commit"


def test_basin_cohorts_reproduced():
    """D-002/D-003/D-005: the known wrong fits are reproduced byte for byte, warnings as the reference."""
    d = load("claims")
    rows = {r["cohort_id"]: r for r in d["results"]}
    fires = [r for r in rows.values() if r["role"] == "basin_fires"]
    blind = [r for r in rows.values() if r["role"] == "basin_blind"]
    assert len(fires) == 4 and len(blind) == 2
    for r in fires + blind:
        assert r["checks"]["claims_identity"] == [], r["cohort_id"]
    for r in fires:
        assert r["basin_fired"] == {"port": True, "reference": True}, r["cohort_id"]


def test_adapter_phase_passed():
    d = load("adapter")
    assert d["status"] == "pass"
    c = d["checks"]
    assert c["adapter_claims_vs_stored"]["passed"] == c["adapter_claims_vs_stored"]["total"] == 30
    assert c["adapter_axes_no_difference"]["passed"] == 1
    assert c["tester_tree_unchanged"]["passed"] == 1


def test_perturbations_all_detected():
    p = os.path.join(RES, "perturbation.json")
    if not os.path.exists(p):
        not_applicable("perturbation.json not written yet (run tests/parity/test_perturbation.py)")
    d = json.load(open(p))
    assert set(d) >= {"value_bytes", "grad_bytes", "cohort_arrays", "claims_identity", "censor_tie",
                      "claims_identity_end_to_end", "warnings_counts"}
    assert all(v["detected"] for v in d.values())
