"""Spec 003d's frozen panel (FR-019): counts, provenance of every id, and the thresholds equal the spec's. Fast."""
import json
import os
import tomllib

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANEL = tomllib.load(open(os.path.join(REPO, "tests", "acceptance", "panel_003d.toml"), "rb"))
SPEC = open(os.path.join(REPO, "specs", "003d-cohort-generator", "spec.md")).read()
TESTER = os.environ.get("SRAGING_TESTER_MAIN", os.path.join(os.environ.get("SRAGING_CODE5") or "/SRAGING_CODE5-is-unset",
                                                            "sr_calibration_tester"))   # release 0.1.1
V5_MANIFEST = os.path.join(TESTER, "data", "library", "v5", "manifest.json")


def retest_ids():
    cov = tomllib.load(open(os.path.join(REPO, "tests", "acceptance", "panel_003.toml"), "rb"))["coverage"]
    return [c for c in cov["profile_jax2"] if c.startswith("censor-")], list(cov["profile_jax2_h"])


def test_counts():
    f, w = PANEL["fidelity"], PANEL["witness"]
    assert len(f["points"]) == len(f["source"]) == 6 and len(f["sharp"]) == 5 and f["broad_control"] in f["points"]
    assert all(src.startswith(f"pure-{p}-c0-s0") for p, src in zip(f["points"], f["source"]))
    assert len(w["sharp"]) == 6 and len(w["nladder"]) == 6
    cen, haz = retest_ids()
    assert len(cen) == 55 and len(haz) == 55 and all(h.startswith("hazard-") for h in haz)


def test_every_id_exists_in_library_v5():
    if not os.path.exists(V5_MANIFEST):
        pytest.skip("NOT APPLICABLE: the tester's library v5 is not readable here")
    ids = {s["cohort_id"] for s in json.load(open(V5_MANIFEST))["specs"]}
    cen, haz = retest_ids()
    want = PANEL["fidelity"]["source"] + PANEL["witness"]["sharp"] + PANEL["witness"]["nladder"] + cen + haz
    assert not set(want) - ids


def test_thresholds_are_the_specs():
    t = PANEL["thresholds"]
    assert t["lambda_max"] == 0.05 and "≤ 0.05" in SPEC
    assert t["dlncv_chi"] == 0.1 and "½·√(0.1/(2·n_max))" in SPEC
    assert abs(t["dlncv_refinement_factor"] - 4 / 3) < 1e-15 and "4/3 of the refinement difference" in SPEC
    assert (t["fidelity_n"], t["fidelity_replicates"], t["fidelity_max_abs_diff"]) == (10000, 200, 1.5e-3)
    assert "200 replicates at n = 10,000" in SPEC and "within 1.5e-3" in SPEC
    assert (t["gtest_min_expected"], t["gtest_ks_p_min"], t["gtest_alpha"]) == (5, 0.01, 0.01)
    assert "0.01 + 3·√(0.0099/m)" in SPEC
    assert t["witness_D_cut"] == 9.49 and abs(t["witness_frac_max"] - 10 / 122) < 1e-15
    assert (t["witness_median_max"], t["witness_D_min"]) == (4.4, -0.1) and "D > 9.49 is ≤ 10/122" in SPEC
    assert t["het_n_cap"] == 20000 and "min(n_max, 20,000)" in SPEC
    assert (t["het_lambda_max"], t["het_grid_nodes"]) == (0.2, 21) and "2·min(n, 20,000)·KL ≤ 0.2" in SPEC
    g = PANEL["grid"]
    assert g["K_start"] in g["K_ladder"] and g["K_start"] == 161
    assert (g["N_ladder"], g["dt_ladder"], g["K_ladder"], g["tau_max"], g["Gtol"]) == \
        ([1500, 3000, 6000, 12000, 24000], [2e-4, 1e-4], [41, 81, 161, 321], 30.0, 1e-12)
