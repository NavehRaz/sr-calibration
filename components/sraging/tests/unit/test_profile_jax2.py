"""profile_jax2 / profile_jax2_h 1.0.0 behind named objects (spec 003, T011, T012; FR-001, FR-004, FR-007,
FR-008; research R7).

Refusals run fast (they raise before any fit). Fits are slow (opt-in: `pytest -m slow`): the capture-identity
test fits the small cohort twice, with and without the archive capture, and requires byte-identical outputs.
"""
import os

import numpy as np
import pytest

from sraging import Cohort, fit_cohort, registry
from sraging.calibrators import profile_jax2 as PJ
from sraging.errors import UnsupportedOptionError

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.path.join(REPO, "tests", "unit", "data", "small_cohort.csv")


def small(**kw):
    return Cohort.from_csv(CSV, delta=1.0, **kw)


@pytest.mark.parametrize("name", ["profile_jax2", "profile_jax2_h"])
def test_both_names_are_registered_at_1_0_0_and_1_0_1_and_the_default_is_unchanged(name):
    """Spec 003b (Naveh, 2026-10-19): the bare name means 1.0.1; 1.0.0 stays registered, by its exact version."""
    assert registry.get(name, "1.0.0").version == "1.0.0"
    assert registry.get(name).version == "1.0.1"
    assert registry.DEFAULT == "profile_jax2"                                    # spec 003g T052


@pytest.mark.parametrize("name,what,make", [
    ("profile_jax2", "window", lambda c: Cohort.from_arrays(c.ages, c.events, delta=1.0, window=(5.0, 60.0))),
    *[(n, w, m) for n in ("profile_jax2", "profile_jax2_h") for w, m in (
        ("entry", lambda c: Cohort.from_arrays(c.ages, c.events, delta=1.0, entry=np.zeros(c.n))),
        ("per-row delta", lambda c: Cohort.from_arrays(c.ages, c.events, delta=np.ones(c.n))))],
])
def test_windows_entry_ages_and_per_row_delta_are_refused_by_name(name, what, make):
    """Spec 006: H 1.0.1 fits windows now (FR-011, below); J still refuses one, and both refuse entry ages and a per-row
    delta."""
    with pytest.raises(UnsupportedOptionError) as ex:
        fit_cohort(make(small()), calibrator=name)
    assert what.split()[0] in str(ex.value)


def test_h_1_0_1_takes_a_window_and_its_1_0_0_and_js_do_not():
    """FR-011 (spec 006 T029): the window path is H's at 1.0.1 only."""
    assert PJ.fits_window("profile_jax2_h", PJ.PORTS["1.0.1"]) and PJ.fits_window("profile_jax2_w", PJ.PORT_W)
    assert not PJ.fits_window("profile_jax2_h", PJ.V100) and not PJ.fits_window("profile_jax2", PJ.PORTS["1.0.1"])
    c = Cohort.from_arrays(small().ages, small().events, delta=1.0, window=(5.0, 60.0))
    PJ._refuse(c, "profile_jax2_h", PJ.PORTS["1.0.1"])                    # no refusal
    with pytest.raises(UnsupportedOptionError, match="window"):
        fit_cohort(c, calibrator="profile_jax2_h", version="1.0.0")


@pytest.mark.slow
def test_h_fits_a_window_with_h_a_nuisance_and_its_sharp_limit_carried():
    """FR-011: H on a windowed cohort reports the window used and the edge tie, never claims h, and carries
    HAZARD_SHARP_LIMIT as the unwindowed H does."""
    r = fit_cohort(Cohort.from_csv(CSV, delta=1.0, window=(40.0, 70.0), time_unit="days"), calibrator="profile_jax2_h",
                   combinations=False)
    w = r.diagnostics["window"]
    assert w["window_eff"] == [40.0, 70.0] and "n_deaths_censored" in w and w["S_lo_at_theta_hat"] > 0
    assert "h_ext" not in {c.parameter for c in r.claims} and r.nuisance["scored"] is False
    assert "HAZARD_SHARP_LIMIT" in [n.get("code") for n in r.diagnostics.get("notes", [])]


def test_a_dt_ceiling_override_is_refused(monkeypatch):
    monkeypatch.setenv("SRK_DT_CEILING", "0.001")
    with pytest.raises(Exception, match="SRK_DT_CEILING"):
        fit_cohort(small(), calibrator="profile_jax2")


@pytest.mark.slow
def test_the_archive_capture_changes_no_byte():
    """R7: fitting with and without capturing the objective gives byte-identical theta-hat, loglik, claims and
    diagnostics (timings excluded), and the captured archive is not empty."""
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.result import canon
    c = small()
    plain = PJ.run_port(c, "profile_jax2", capture=False)
    captured = PJ.run_port(c, "profile_jax2", capture=True)
    strip = lambda x: canon(PJ.strip_timing(x))           # the tester's timing leaves, at any depth
    assert strip(plain["result"]) == strip(captured["result"])
    assert strip(plain["claims"]) == strip(captured["claims"]) and strip(plain["sc_block"]) == strip(captured["sc_block"])
    assert plain["archive"] is None and captured["archive"][0].shape[0] > 100


@pytest.mark.slow
def test_a_jax2_fit_records_its_mechanisms_and_assumptions():
    r = fit_cohort(small(), calibrator="profile_jax2", version="1.0.0")
    assert r.calibrator == {"name": "profile_jax2", "version": "1.0.0"} and r.status == "ok"
    m = r.mechanisms
    assert {"basin_refit", "relabelled_rows", "N", "dt_floor", "n_fallback", "widened"} <= set(m)
    assert m["N"] in (900, 1800, 3600) and r.gap_nats >= 0 and r.archive_points > 100
    assert r.nuisance is None and "no external mortality" in r.diagnostics["assumptions"]
    assert [c.parameter for c in r.claims][:4] == ["eta", "beta", "epsilon", "kappa"]


@pytest.mark.slow
def test_the_hazard_twin_reports_h_as_a_nuisance_never_a_claim():
    r = fit_cohort(small(), calibrator="profile_jax2_h")
    assert r.nuisance is not None and r.nuisance["scored"] is False
    assert {"h_ext", "hL", "ExtH"} <= set(r.nuisance) and r.theta_hat.h_ext == r.nuisance["h_ext"]
    assert "h" not in r.claims.keys() and "h_ext" not in r.claims.keys()


# ---------------------------------------------------------------------------------------- spec 003b: the 1.0.1 port
# T010: run_fit's keywords per port against the frozen calibrate's captured call (jax2's SWITCHES.tsv, copied into
# tests/data with its sha256), the estimate block on 1.0.1 only, the kappa-fixed classes ported but unregistered.

import hashlib  # noqa: E402

SWITCHES = os.path.join(REPO, "tests", "data", "jax2_101_SWITCHES.tsv")
SWITCHES_SHA256 = "a592141d7482f025b59d4458b1a9374002b52ebfb37a7b5816625288ac115469"
TEN = {"relabel", "hazard", "window", "shadow", "exact_from_F", "self_consistent", "floor_early", "n_rule", "dt_rule",
       "early_all"}


def captured(j2_name):
    """{keyword: repr(value)} that 1.0.1's calibrate passes to run_fit for `j2_name` (empty cell = not passed)."""
    rows = [ln.rstrip("\n").split("\t") for ln in open(SWITCHES) if not ln.startswith("#")]
    head = rows[0]
    col = head.index(j2_name)
    return {r[1]: r[col] for r in rows[1:] if r[0] == "run_fit" and r[1] != "run_fit.how" and r[col] != ""}


def test_the_switches_table_is_jax2s():
    assert hashlib.sha256(open(SWITCHES, "rb").read()).hexdigest() == SWITCHES_SHA256


@pytest.mark.parametrize("name,j2", [("profile_jax2", "mle_profile_jax2"), ("profile_jax2_h", "mle_profile_jax2_h")])
def test_run_kwargs_of_1_0_1_equal_the_frozen_calibrates_captured_call(name, j2):
    got = PJ.run_kwargs(PJ.PORTS["1.0.1"].classes[name])
    assert {k: repr(v) for k, v in got.items()} == captured(j2)


def test_run_kwargs_of_1_0_0_stay_its_ten():
    assert set(PJ.run_kwargs(PJ.PORTS["1.0.0"].classes["profile_jax2"])) == TEN
    assert PJ.run_kwargs(PJ.CLASSES["profile_jax2"]) == PJ.run_kwargs(PJ.PORTS["1.0.0"].classes["profile_jax2"])


def test_a_dropped_or_added_keyword_is_caught():
    good = {k: repr(v) for k, v in PJ.run_kwargs(PJ.PORTS["1.0.1"].classes["profile_jax2"]).items()}
    dropped = {k: v for k, v in good.items() if k != "extent_z"}
    added = dict(good, kappa_fixed="0.001")
    assert good == captured("mle_profile_jax2") and dropped != good and added != good


def _fake_result(hazard=False):
    th = [1e-3, 0.5, 0.02, 0.3] + ([1e-4] if hazard else [])
    return {"theta": th, "loglik": -123.5,
            "diagnostics": {"self_consistency": {"ll_best": -123.5, "gap_nats": 0.0, "stale": "none"},
                            "solver": {"N": 900, "dt": 0.002}, "dt_floor": {"grid_rule": {"dt_in_scope": False},
                                                                           "dl": 0.01, "binds": False}}}


def test_the_estimate_block_is_on_1_0_1_and_not_on_1_0_0():
    from sraging.result import SRAGING_DIAGNOSTICS
    p1, p0 = PJ.PORTS["1.0.1"], PJ.PORTS["1.0.0"]
    res = _fake_result()
    est = PJ.estimate_for(p1, p1.classes["profile_jax2"], res)
    assert est == p1.claims.estimate_block(res, False, None) and est["schema"] == 1
    assert est["theta_hat"] == {"eta": 1e-3, "beta": 0.5, "epsilon": 0.02, "kappa": 0.3}
    assert PJ.estimate_for(p0, p0.classes["profile_jax2"], res) is None              # the twin
    assert "estimate" not in SRAGING_DIAGNOSTICS                                      # the method's key, compared


SWITCH_CONST = {"hazard": "HAZARD", "window": "WINDOW", "relabel": "RELABEL", "shadow": "SHADOW",
                "exact_from_F": "EXACT_F", "self_consistent": "SELF_CONSISTENT", "floor_early": "FLOOR_EARLY",
                "n_rule": "N_RULE", "dt_rule": "DT_RULE", "early_all": "EARLY_ALL", "inner_polish_j": "INNER_POLISH_J",
                "bracket_bisect": "BRACKET_BISECT", "extent_z": "EXTENT_Z", "box_flag": "BOX_FLAG"}


def class_row(j2_name):
    """{constant: repr} of jax2's SWITCHES.tsv class rows for `j2_name`."""
    rows = [ln.rstrip("\n").split("\t") for ln in open(SWITCHES) if not ln.startswith("#")]
    col = rows[0].index(j2_name)
    return {r[1]: r[col] for r in rows[1:] if r[0] == "class"}


@pytest.mark.parametrize("name,j2", [("profile_jax2", "mle_profile_jax2"), ("profile_jax2_h", "mle_profile_jax2_h")])
def test_the_switches_block_is_the_frozen_claimsets_14_keys(name, j2):
    """1.0.1's claimset() records the class constants: 1.0.0's 10 keys + 4 (handoff 7.2 L4), values from jax2's
    table; kappa_fixed / combos only on the kfix names (not registered)."""
    sw = PJ.switches_for(PJ.PORTS["1.0.1"].classes[name])
    assert list(sw) == list(SWITCH_CONST)
    row = class_row(j2)
    assert sw == {k: row[c] == "True" for k, c in SWITCH_CONST.items()}
    assert all(type(v) is bool for v in sw.values())
    assert PJ.switches_for(PJ.PORTS["1.0.0"].classes[name]) is None                  # 1.0.0's FitResult unchanged


def _res_with_profiles():
    res = _fake_result()
    res["diagnostics"].update(profiles={"eta": [1.0]}, box_points=[[0.1]])
    res["dt_ceiling"] = 0.004
    return res


def test_1_0_1_method_diagnostics_follow_the_frozen_claimset():
    p1 = PJ.PORTS["1.0.1"]
    cls = p1.classes["profile_jax2"]
    res, sc = _res_with_profiles(), {"gap_nats": 0.0}
    d = PJ.method_diagnostics(p1, cls, res, [], sc)
    assert "profiles" not in d and d["box_points"] == [[0.1]]                        # KEEP_PROFILES, BOX_FLAG off
    assert d["self_consistency"] is sc and d["loglik"] == -123.5 and d["dt_ceiling"] == 0.004
    assert d["switches"] == PJ.switches_for(cls) and d["estimate"] == PJ.estimate_for(p1, cls, res)
    assert "code" not in d and "box_limited" not in d
    assert "profiles" in res["diagnostics"]                                           # the result is not mutated


def test_box_flag_and_keep_profiles_are_mirrored_with_a_twin(monkeypatch):
    p1 = PJ.PORTS["1.0.1"]
    planted = type("Planted", (p1.classes["profile_jax2"],), {"BOX_FLAG": True, "KEEP_PROFILES": True})
    monkeypatch.setattr(p1.claims, "box_limited_block", lambda res, kw, sc: {"flagged": ["beta"]})
    d = PJ.method_diagnostics(p1, planted, _res_with_profiles(), [], None)
    assert d["box_limited"] == {"flagged": ["beta"]} and "box_points" not in d and d["profiles"] == {"eta": [1.0]}
    assert d["self_consistency"] == _fake_result()["diagnostics"]["self_consistency"]   # run_fit's own, no sc block
    assert d["switches"]["box_flag"] is True
    monkeypatch.setattr(p1.claims, "box_limited_block", lambda res, kw, sc: None)     # the twin: nothing flagged
    d = PJ.method_diagnostics(p1, planted, _res_with_profiles(), [], None)
    assert "box_limited" not in d and d["box_points"] == [[0.1]]


def test_1_0_0_method_diagnostics_are_unchanged():
    p0 = PJ.PORTS["1.0.0"]
    res, sc = _res_with_profiles(), {"gap_nats": 0.0}
    d = PJ.method_diagnostics(p0, p0.classes["profile_jax2"], res, [], sc)
    want = list(dict.fromkeys([k for k in res["diagnostics"] if k != "profiles"]
                              + ["self_consistency", "loglik", "dt_ceiling"]))
    assert list(d) == want and d["box_points"] == [[0.1]] and d["self_consistency"] is sc


def test_method_diagnostics_refuse_a_kappa_fixed_class():
    p1 = PJ.PORTS["1.0.1"]
    with pytest.raises(ValueError, match="kappa-fixed"):
        PJ.method_diagnostics(p1, p1.kfix_classes["mle_profile_jax2_kfix_k1"], _res_with_profiles(), [], None)


def test_the_kappa_fixed_classes_are_ported_and_not_registered():
    from sraging import registry
    kf = PJ.PORTS["1.0.1"].kfix_classes
    assert sorted(kf) == sorted(["mle_profile_jax2_kfix_k0p001", "mle_profile_jax2_kfix_k0p5",
                                 "mle_profile_jax2_kfix_k1", "mle_profile_jax2_h_kfix_k0p001",
                                 "mle_profile_jax2_h_kfix_k0p5", "mle_profile_jax2_h_kfix_k1"])
    assert all(getattr(c, "KAPPA0", None) is not None for c in kf.values())
    assert not any(n.split("@")[0] in kf for n in registry.names())
    assert PJ.PORTS["1.0.0"].kfix_classes == {}
