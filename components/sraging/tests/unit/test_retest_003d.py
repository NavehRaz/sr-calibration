"""Spec 003d T045 (tools/retest_003d.py): the re-test panel's ids, the sha check on the shipped arrays, the truth's
conversion, and the fit entry's mode. Fast: the fit is replaced by a fake."""
import json
import types

import numpy as np
import pytest

import retest_003d as RT
import v5_reproduce_003d as R


def test_the_panel_is_spec_003s_55_censored_cohorts_then_the_12_witnesses():
    ids = RT.panel_ids()
    retest = [p["cohort_id"] for p in ids if p["kind"] == "retest"]
    witness = [p["cohort_id"] for p in ids if p["kind"] == "witness"]
    assert len(retest) == 55 and all(c.startswith("censor-") for c in retest) and len(set(retest)) == 55
    assert {p["source"] for p in ids if p["kind"] == "retest"} == {"v2"}
    assert len(witness) == 12 and witness[0] == "pure-k2-p05-c0-s0" and witness[-1] == "nlad-k0.1-p00-n10000-s1"
    assert {p["source"] for p in ids if p["kind"] == "witness"} == {"v5"}


def test_the_truth_is_the_testers_conversion():
    # censor-k0.01-p00-c0-s0's truth in spec 003's v2 rows (the snapshot loader's physical_truth)
    row = {"rho_eta": 0.0002125427697723628 * 100.0 ** 2, "rho_beta": 0.1, "rho_eps": 0.0039810717055349735,
           "kappa": 0.01, "L_phys": 100.0}
    t = RT.physical_truth(row)
    assert t == pytest.approx({"eta": 0.0002125427697723628, "beta": 0.001, "epsilon": 3.9810717055349735e-05,
                               "kappa": 0.01}, rel=1e-15)


def _planted(tmp_path):
    ages, ev = np.array([0.0, 3.0, 5.0, 5.0]), np.array([1, 1, 1, 0])
    np.savez(tmp_path / "c.npz", ages=ages, events=ev)
    return {"cohort_id": "c", "digest": R.cohort_digest(ages, ev), "delta": 1.0, "calibrator": "profile_jax2",
            "kind": "retest", "arm": "censor", "cluster": "k", "truth": {"eta": 1e-3, "beta": 1e-3, "epsilon": 1e-4,
                                                                          "kappa": 0.1}}


def test_the_sha_check_refuses_a_planted_change(tmp_path):
    row = _planted(tmp_path)
    a, e = RT.load_checked(row, root=str(tmp_path))
    assert list(a) == [0.0, 3.0, 5.0, 5.0]
    np.savez(tmp_path / "c.npz", ages=np.array([0.0, 3.0, 4.0, 5.0]), events=np.array([1, 1, 1, 0]))
    with pytest.raises(ValueError, match="sha check failed"):
        RT.load_checked(row, root=str(tmp_path))
    out = RT.fit_one(row, "head", root=str(tmp_path))                  # a refused array is a harness error, no fit
    assert out["error_origin"] == "harness" and "sha check failed" in out["error"]


def test_the_fit_entry_passes_first3_and_records_the_witness(tmp_path, monkeypatch):
    import sraging
    row = _planted(tmp_path)
    seen = {}

    def fake_fit(cohort, *, calibrator, combinations):
        seen.update(calibrator=calibrator, combinations=combinations, n=cohort.n, delta=cohort.delta)
        return types.SimpleNamespace(status="ok", cohort={"n_deaths": 3}, claims=(), combinations={"mode": "first3"},
                                     gap_nats=0.0, warnings=(), loglik=-7.0)

    monkeypatch.setattr(sraging, "fit_cohort", fake_fit)
    out = RT.fit_one(row, "head", root=str(tmp_path))
    assert seen == {"calibrator": "profile_jax2", "combinations": "first3", "n": 4, "delta": 1.0}
    assert out["status"] == "ok" and out["loglik"] == -7.0 and "witness" in out
    json.dumps(out, default=str)
