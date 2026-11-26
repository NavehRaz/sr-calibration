"""Spec 003f (T003): the leverage check's pieces, each with a planted case and its twin (constitution V). Fast: no
fit runs here (the check on real fits is the acceptance phases' job, tools/acceptance_003f.py)."""
import io
import math
import os

import numpy as np
import pytest

from sraging import leverage as LEV
from sraging.result import SRAGING_DIAGNOSTICS, Warning
from test_result_json import make

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------------------------- free statistics
def test_gap_sd_flags_an_isolated_late_death_and_its_twin_does_not():
    base = np.linspace(20.0, 30.0, 30)                       # 30 deaths, evenly spread
    ev = np.ones(31, int)
    planted = LEV.gap_sd(np.append(base, 60.0), ev)
    twin = LEV.gap_sd(np.append(base, 30.3), ev)
    assert planted["applicable"] and planted["flagged"] and planted["g"] > 2
    assert twin["applicable"] and not twin["flagged"]
    few = LEV.gap_sd(np.linspace(20, 30, 19), np.ones(19, int))
    assert not few["applicable"] and not few["flagged"] and few["n_deaths"] == 19


def test_gap_sd_counts_deaths_only():
    ages = np.append(np.linspace(20.0, 30.0, 30), 60.0)
    ev = np.append(np.ones(30, int), 0)                      # the late one is censored: no isolated death
    assert not LEV.gap_sd(ages, ev)["flagged"]


def test_share_is_the_absolute_form_and_names_the_worst_row():
    terms = np.array([-2.0, -13.0, -1.0, -4.0])
    s = LEV.share(terms, np.array([5.0, 0.0, 7.0, 9.0]), np.array([1, 1, 0, 1]))
    assert s["row"] == 2 and s["age"] == 0.0 and s["event"] == 1          # 1-based, as the floor report
    assert s["share"] == pytest.approx(13.0 / 20.0)


# --------------------------------------------------------------------------------------------------- influence
def _exponential(t):
    """Per-individual log-likelihood of an exponential rate lambda = e^z, all deaths: l_i = z - e^z t_i."""
    lam = len(t) / t.sum()
    z = math.log(lam)
    G = (1.0 - lam * t)[:, None]                             # dl_i/dz
    H = np.array([[-lam * t.sum()]])                         # d2 sum l_i / dz2 = -n at the MLE
    return z, G, H, lam


def test_the_one_step_shift_matches_an_exact_drop_one_refit():
    rng = np.random.default_rng(7)
    t = rng.exponential(2.0, 40)
    z, G, H, lam = _exponential(t)
    inf = LEV.influence(G, H, np.array([z]), np.array([-50.0]), np.array([50.0]))
    exact = np.array([math.log((len(t) - 1) / (t.sum() - ti)) - z for ti in t])
    one_step = inf["DZ"][:, 0]
    x = t / t.sum()                                            # the dropped point's weight; the error is second order
    assert np.all(np.abs(one_step - exact) <= (1.0 / len(t) + x) ** 2)
    assert np.any(np.abs(one_step - exact) > 0.25 * (1.0 / len(t) + x) ** 2)   # and the bound is not vacuous
    assert int(np.argmax(np.abs(one_step))) == int(np.argmax(np.abs(exact)))


def test_a_coordinate_at_its_bound_is_not_free():
    G = np.array([[1.0, 2.0], [-1.0, -2.0]])
    H = np.array([[-4.0, 0.5], [0.5, -3.0]])
    z = np.array([0.0, math.log(1e-3)])
    inf = LEV.influence(G, H, z, np.array([-10.0, math.log(1e-3)]), np.array([10.0, 10.0]))
    assert inf["free"] == [True, False] and inf["active"] == [1]
    assert np.all(inf["DZ"][:, 1] == 0.0)
    twin = LEV.influence(G, H, np.array([0.0, 0.0]), np.array([-10.0, -10.0]), np.array([10.0, 10.0]))
    assert twin["free"] == [True, True]


def test_non_positive_and_tiny_information_directions_are_cut_and_recorded():
    # information I = -H with eigenvalues 5, 1e-14 (tiny) and -2e-3 (wrong sign, as measured on ITP)
    V = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))[0]
    I = V @ np.diag([5.0, 1e-14, -2e-3]) @ V.T
    inf = LEV.influence(np.ones((2, 3)), -I, np.zeros(3), -10 * np.ones(3), 10 * np.ones(3))
    assert inf["n_cut"] == 2 and inf["n_kept"] == 1 and inf["min_eigenvalue"] == pytest.approx(-2e-3)
    assert inf["n_negative"] == 1 and inf["min_relative"] == pytest.approx(-2e-3 / 5.0)
    twin = LEV.influence(np.ones((2, 3)), -np.diag([5.0, 3.0, 1.0]), np.zeros(3), -10 * np.ones(3), 10 * np.ones(3))
    assert twin["n_cut"] == 0 and twin["n_negative"] == 0


def test_the_cut_sits_above_the_hessians_accuracy():
    # P1: the FD Hessian agrees with forward-over-forward to 9e-8 relative, so a direction at 1e-7 of the largest is
    # noise and is cut; one at 1e-5 is kept (the twin)
    assert LEV.REL_CUT >= 1e-6
    planted = LEV.influence(np.ones((2, 2)), -np.diag([1.0, 1e-7]), np.zeros(2), -10 * np.ones(2), 10 * np.ones(2))
    twin = LEV.influence(np.ones((2, 2)), -np.diag([1.0, 1e-5]), np.zeros(2), -10 * np.ones(2), 10 * np.ones(2))
    assert planted["n_cut"] == 1 and twin["n_cut"] == 0


def test_a_tie_for_the_top_keeps_the_lowest_row_and_is_counted():
    sh = {q: np.array([0.1, 0.5, 0.2, 0.5]) for q in LEV.QUANTITIES}
    r = LEV.ranking(sh, "all4")
    assert r["top"] == 1 and r["n_tied"] == 2
    twin = LEV.ranking({q: np.array([0.1, 0.5, 0.2, 0.4]) for q in LEV.QUANTITIES}, "all4")
    assert twin["top"] == 1 and twin["n_tied"] == 1


def test_the_flag_quantities_and_their_gradients():
    th = dict(eta=1e-3, beta=0.5, epsilon=0.02, kappa=0.3)
    z = np.log([th[p] for p in LEV.PARAMS])
    q = LEV.quantities(z)
    rbar = th["beta"] * (1 - th["kappa"] * math.log1p(1 / th["kappa"]))
    assert q["eta_over_eps"] == pytest.approx(math.log(th["eta"] / th["epsilon"]))
    assert q["rbar_over_eta"] == pytest.approx(math.log(rbar / th["eta"]))
    assert q["eta"] == pytest.approx(z[0]) and q["epsilon"] == pytest.approx(z[2])
    g = LEV.quantity_gradients(z)
    assert g["eta_over_eps"] == pytest.approx([1, 0, -1, 0], abs=1e-8)
    assert g["rbar_over_eta"][0] == pytest.approx(-1) and g["rbar_over_eta"][1] == pytest.approx(1)


# ------------------------------------------------------------------------------------------------- the labels
@pytest.mark.parametrize("R, delta, want", [(2.0, 5.4, "flag"), (2.0, 0.5, "flat_valley_slide"),
                                            (2.0, 1.92, "flat_valley_slide"), (1.2, 5.4, "no_flag"),
                                            (1.5, 5.4, "no_flag"), (1.2, -0.05, "no_flag"),
                                            (2.0, -0.5, "failed"), (1.2, -3.8, "failed"),
                                            (math.nan, 1.0, "failed"), (math.inf, 1.0, "failed"),
                                            (2.0, math.nan, "failed"), (None, 1.0, "failed")])
def test_label(R, delta, want):
    # the rule of 2026-10-19: R over the four quantities, the gain in total nats, a refit below theta-hat is failed
    assert LEV.label(R, delta, ratio=1.5, delta_total_nats=1.92, fail_nats=0.1) == want


# ------------------------------------------------------------------------------------- the drop: which refit stands
def _fake_drop(monkeypatch, ll_ref, ll_mle, ll_warm, mle_ok=True, warm_ok=True):
    th_mle, th_warm = [1e-3, 0.5, 0.02, 0.3], [2e-3, 0.5, 0.02, 0.3]
    monkeypatch.setattr(LEV, "refit_mle", lambda c, n, j, **k: (
        {"status": "ok", "reason": None, "theta": th_mle, "cpu_s": 1.0} if mle_ok
        else {"status": "failed", "reason": "planted", "theta": None, "cpu_s": 1.0}))
    monkeypatch.setattr(LEV, "refit_warm", lambda c, n, j, z0: (
        {"status": "ok", "reason": None, "theta": th_warm, "cpu_s": 1.0} if warm_ok
        else {"status": "failed", "reason": "planted", "theta": None, "cpu_s": 1.0}))
    by = {"mle": ll_mle, "warm": ll_warm}

    def lls(c, n, j, zs):
        out = [ll_ref]
        for z in zs[1:]:
            out.append(by["mle"] if np.allclose(np.exp(z), th_mle) else by["warm"])
        return out
    monkeypatch.setattr(LEV, "reduced_loglik", lls)
    rule = {"ratio": 1.5, "delta_total_nats": 1.92, "fail_nats": 0.1}
    z = np.log([1e-3, 0.5, 0.02, 0.3])
    return LEV._drop(None, "profile_jax2", 0, z, np.array([0.0, 5.0, 7.0]), np.array([1, 1, 1]), rule)


def test_the_mle_stage_stands_unless_the_warm_start_beats_it_by_more_than_noise(monkeypatch):
    d = _fake_drop(monkeypatch, -100.0, -99.0, -99.0 + 0.5 * LEV.WARM_GAIN_TOL)
    assert d["refit"]["start"] == "mle_stage" and d["delta_total"] == pytest.approx(1.0)
    d = _fake_drop(monkeypatch, -100.0, -99.0, -99.0 + 10 * LEV.WARM_GAIN_TOL)
    assert d["refit"]["start"] == "warm" and d["delta_total"] == pytest.approx(1.0 + 10 * LEV.WARM_GAIN_TOL)
    assert d["ratios"]["eta"]["ratio"] == pytest.approx(2.0)                # compared with theta-hat, not the stage


def test_a_refit_ending_below_theta_hat_is_failed_and_its_twin_is_measured(monkeypatch):
    d = _fake_drop(monkeypatch, -100.0, -103.8, -100.5, warm_ok=True)       # planted: both below theta-hat
    assert d["labels"]["all4"] == "failed" and "below theta-hat" in LEV._failed_reason(d)
    twin = _fake_drop(monkeypatch, -100.0, -103.8, -100.0)                  # the warm start repairs it
    assert twin["refit"]["start"] == "warm" and twin["labels"]["all4"] == "flat_valley_slide"


def test_one_failed_start_leaves_the_other_and_two_leave_no_measurement(monkeypatch):
    d = _fake_drop(monkeypatch, -100.0, -99.0, -98.0, mle_ok=False)
    assert d["refit"]["start"] == "warm" and "labels" in d
    d = _fake_drop(monkeypatch, -100.0, -99.0, -98.0, mle_ok=False, warm_ok=False)
    assert "labels" not in d and "planted" in LEV._failed_reason(d)


def test_the_warm_start_is_the_estimators_own_lbfgs_on_the_reduced_data(monkeypatch):
    from sraging import Cohort
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.compat.jax2 import estimator as EST

    class Stop(Exception):
        pass
    seen = {}

    def fake(obj, fun_grad, fun_val, z0, bounds, opts, nm_opts, **kw):
        seen.update(z0=np.asarray(z0), bounds=bounds, opts=opts, fun_grad=fun_grad, obj=obj)
        raise Stop
    monkeypatch.setattr(EST, "_lbfgs", fake)
    c = Cohort.from_csv(os.path.join(REPO, "tests", "data", "itp_plus_age0.csv"), delta=1.0, time_unit="days")
    z0 = np.log([3.5e-5, 0.03, 0.0024, 0.0011])
    r = LEV.refit_warm(c, "profile_jax2", 0, z0)
    assert r["status"] == "failed" and "Stop" in r["reason"]                # recorded, never raised
    assert np.array_equal(seen["z0"], z0) and seen["opts"] is EST._LBFGS_MLE
    assert seen["fun_grad"] == seen["obj"].nll_grad
    inp = PJ.fit_inputs(LEV._without(c, 0), "profile_jax2")
    assert seen["bounds"] == [(math.log(inp["box"][p][0]), math.log(inp["box"][p][1])) for p in LEV.PARAMS]
    assert len(inp["t"]) == c.n - 1                                          # the reduced data


def test_degenerate_influence_is_not_assessable(monkeypatch):
    from sraging import Cohort
    c = Cohort.from_csv(os.path.join(REPO, "tests", "data", "itp_plus_age0.csv"), delta=1.0, time_unit="days")
    n = c.n
    box = {p: (1e-12, 1e3) for p in LEV.PARAMS}

    def td(H):
        return lambda cohort, name, z: {"terms": -np.ones(n), "cond": 0.0, "G": np.ones((n, 4)), "H": H,
                                        "gradient": np.zeros(4), "hessian_source": "planted",
                                        "inputs": {"t": np.asarray(c.ages, float), "box": box}}
    monkeypatch.setattr(LEV, "_drop", lambda *a, **k: {"row": 1, "age": 0.0, "event": 1, "refits": {},
                                                       "refit": {"status": "failed", "reason": "planted"}})
    th = {"eta": 1e-3, "beta": 0.5, "epsilon": 0.02, "kappa": 0.3}
    monkeypatch.setattr(LEV, "terms_and_derivatives", td(np.zeros((4, 4))))
    planted = LEV.check(c, "profile_jax2", th, {})
    assert planted["label"] == "not_assessable" and "no information direction" in planted["reason"]
    monkeypatch.setattr(LEV, "terms_and_derivatives", td(-np.eye(4)))
    twin = LEV.check(c, "profile_jax2", th, {})
    assert twin["label"] == "failed" and "both refits failed" in twin["reason"]


# ---------------------------------------------------------------------------------------- the refit, unchanged path
def test_the_refit_calls_the_estimator_exactly_as_run_fit_does_plus_the_mle_stage(monkeypatch):
    from sraging import Cohort
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.compat.jax2 import claims as JC
    from sraging.compat.jax2 import estimator as EST

    class Stop(Exception):
        pass
    seen = []

    def fake_fit(fj, ages, events, **kw):
        seen.append(dict(kw))
        raise Stop
    monkeypatch.setattr(EST, "fit", fake_fit)
    c = Cohort.from_csv(os.path.join(REPO, "tests", "data", "itp_plus_age0.csv"), delta=1.0, time_unit="days")
    for name in ("profile_jax2", "profile_jax2_h"):
        seen.clear()
        with pytest.raises(Stop):
            JC.run_fit(PJ.observation(c), **PJ.run_kwargs(PJ.CLASSES[name]))
        with pytest.raises(Stop):
            LEV.refit_mle(c, name, drop_row=0, raise_errors=True)
        ref, mine = seen
        assert mine.pop("stages") == ("mle",)
        assert mine == ref, name                              # the same keywords and values, nothing else
    bad = dict(ref, n_rule=not ref.get("n_rule", False))        # the comparator catches a planted difference
    assert bad != ref


def test_prepare_keeps_row_order_and_length():
    from sraging import Cohort
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.compat.jax2 import observation_model as om
    for csv in ("itp_plus_age0.csv",):
        c = Cohort.from_csv(os.path.join(REPO, "tests", "data", csv), delta=1.0, time_unit="days")
        obs = PJ.observation(c)
        prep = om.prepare(obs.ages, obs.events, obs.delta, obs.window, window_on=False, relabel_on=True)
        assert np.array_equal(np.asarray(prep.ages, float), np.asarray(obs.ages, float))
    # a censor tie that relabels: deaths in the censoring age's bin become censored, rows stay in place
    ages = np.array([5.0, 9.0, 10.0, 10.0, 10.0, 3.0])
    events = np.array([1, 1, 1, 0, 0, 1])
    prep = om.prepare(ages, events, 1.0, None, window_on=False, relabel_on=True)
    assert np.array_equal(np.asarray(prep.ages, float), ages) and len(prep.events) == len(events)


# ------------------------------------------------------------------------------------- warnings, printing, clean
def _block(label, **over):
    b = {"version": LEV.VERSION, "label": label, "reason": None,
         "dropped": {"row": 1, "age": 0.0, "event": 1},
         "ratios": {"eta_over_eps": {"ratio": 1.59}, "rbar_over_eta": {"ratio": 1.81}, "eta": {"ratio": 24.0},
                    "epsilon": {"ratio": 38.2}},
         "R": {"all4": 38.2, "lead_pair": 1.81, "raw": 38.2}, "delta_total": 5.42, "delta_nats": 0.0178,
         "refit": {"start": "mle_stage"}, "rule": {"quantities": "all4"},
         "gap_sd": {"g": 0.65, "flagged": False}, "cost": {"cpu_s": 120.0, "share_of_fit": 0.08}}
    b.update(over)
    return b


@pytest.mark.parametrize("label, fires", [("flag", True), ("not_assessable", True), ("failed", True),
                                          ("not_run", True), ("no_flag", False), ("flat_valley_slide", False)])
def test_the_leverage_warning(label, fires):
    from sraging.warnings import leverage_warnings
    ws = leverage_warnings(_block(label, reason=None if label in ("flag", "no_flag", "flat_valley_slide") else "x"))
    assert (len(ws) == 1 and ws[0].code == "LEVERAGE") if fires else ws == ()
    if label == "flag":
        assert "age 0" in ws[0].message and "1.81" in ws[0].message and "38.2" in ws[0].message
        assert "5.42 nats" in ws[0].message


def test_the_h_warning_names_its_reason():
    from sraging.warnings import h_not_usable
    w = h_not_usable()
    assert w.code == "H_NOT_USABLE" and "not validated on library v5" in w.message and "spec 003b" in w.message


def _jax2(**over):
    prov = dict(make().provenance, numerics="jax2", calibrator={"name": "profile_jax2", "version": "0.1.0+jax2-1.0.0"})
    return make(provenance=prov, warnings=(), **over)


def test_a_jax2_result_without_the_block_is_not_clean_and_v033_is_unchanged():
    from sraging.result import CHECK_CODES, CheckRecord
    quiet = tuple(CheckRecord(c, True, False) for c in CHECK_CODES)
    v033 = make(warnings=(), checks=quiet)
    assert v033.clean                                         # the twin: profile_v033, no block, clean as before
    j = _jax2(checks=quiet)
    assert not j.clean
    with_block = _jax2(checks=quiet, diagnostics=dict(j.diagnostics, leverage=_block("no_flag")))
    assert with_block.clean


def test_the_printed_line():
    from sraging.cli import _leverage_line
    j = _jax2(diagnostics=dict(make().diagnostics, leverage=_block("flag")))
    line = _leverage_line(j)
    assert line.startswith("leverage: flag") and "row 1" in line and "1.81x" in line and "120" in line
    assert "gains 5.42 nats (by the MLE stage)" in line and "gap/sd" not in line and "tied" not in line
    gapped = _jax2(diagnostics=dict(make().diagnostics, leverage=_block("flag", gap_sd={"g": 3.1, "flagged": True},
                                                                         n_tied_top=3)))
    assert "gap/sd 3.1 > 2" in _leverage_line(gapped) and "tied with 2 more" in _leverage_line(gapped)
    assert _leverage_line(_jax2()) == "leverage: NOT RUN (result written before spec 003f)"
    # profile_v033: its own check since spec 006 (FR-024); a v033 result written before it says so
    assert _leverage_line(make()) == "leverage: NOT RUN (a profile_v033 result written before spec 006)"
    v = make()
    v033 = type(v)(**{**v.__dict__, "diagnostics": dict(v.diagnostics, leverage=_block("flag"))})
    assert _leverage_line(v033).startswith("leverage: flag") and "row 1" in _leverage_line(v033)


def test_the_block_never_enters_the_reference_comparison():
    assert "leverage" in SRAGING_DIAGNOSTICS
    r = _jax2(diagnostics=dict(make().diagnostics, leverage=_block("flag")))
    assert "leverage" not in r.identity_view()["diagnostics"]


def test_the_rule_is_naveh_s_of_2026_10_19():
    from sraging.leverage_rule import RULE
    assert RULE["quantities"] == "all4" and set(LEV.QUANTITY_SETS["all4"]) == set(LEV.QUANTITIES)
    assert RULE["ratio"] == 1.5 and RULE["delta_total_nats"] == 1.92 and RULE["fail_nats"] == 0.1
    assert RULE["refit"] == "mle_stage_and_warm_start" and RULE["provisional"] is False


# ---------------------------------------------------------------------------- spec 003b T022: each port's own refit
ITP0 = os.path.join(REPO, "tests", "data", "itp_plus_age0.csv")


@pytest.mark.parametrize("version", ["1.0.0", "1.0.1"])
def test_the_refit_is_each_ports_own_run_fit_call_plus_the_mle_stage(monkeypatch, version):
    from sraging import Cohort
    from sraging.calibrators import profile_jax2 as PJ
    P = PJ.PORTS[version]

    class Stop(Exception):
        pass
    seen = []

    def fake_fit(fj, ages, events, **kw):
        seen.append((fj, dict(kw)))
        raise Stop
    monkeypatch.setattr(P.estimator, "fit", fake_fit)
    c = Cohort.from_csv(ITP0, delta=1.0, time_unit="days")
    for name in ("profile_jax2", "profile_jax2_h"):
        seen.clear()
        with pytest.raises(Stop):
            P.claims.run_fit(PJ.observation(c), **PJ.run_kwargs(P.classes[name]))
        with pytest.raises(Stop):
            LEV.refit_mle(c, name, drop_row=0, raise_errors=True, port=P)
        (fj_ref, ref), (fj_mine, mine) = seen
        assert fj_mine is fj_ref is P.fp_jax
        assert mine.pop("stages") == ("mle",)
        assert mine == ref, (version, name)
    if version == "1.0.1":                                     # 1.0.1's own keywords are there, and a drop is caught
        assert {"inner_polish", "bracket_bisect", "extent_z", "dt_rule"} <= set(ref)
        assert {k: v for k, v in mine.items() if k != "extent_z"} != ref
    else:
        assert not {"inner_polish", "bracket_bisect", "extent_z", "box_flag"} & set(ref)


def test_the_warm_start_on_1_0_1_is_its_own_objective_and_lbfgs(monkeypatch):
    from sraging import Cohort
    from sraging.calibrators import profile_jax2 as PJ
    P1 = PJ.PORTS["1.0.1"]

    class Stop(Exception):
        pass
    seen = {}

    def fake(obj, fun_grad, fun_val, z0, bounds, opts, nm_opts, **kw):
        seen.update(obj=obj, opts=opts, bounds=bounds, z0=np.asarray(z0))
        raise Stop
    monkeypatch.setattr(P1.estimator, "_lbfgs", fake)
    c = Cohort.from_csv(ITP0, delta=1.0, time_unit="days")
    z0 = np.log([3.5e-5, 0.03, 0.0024, 0.0011])
    r = LEV.refit_warm(c, "profile_jax2", 0, z0, port=P1)
    assert r["status"] == "failed" and "Stop" in r["reason"]
    assert isinstance(seen["obj"], P1.estimator._Objective) and seen["opts"] is P1.estimator._LBFGS_MLE
    inp = PJ.fit_inputs(LEV._without(c, 0), "profile_jax2", P1)
    assert seen["bounds"] == [(math.log(inp["box"][p][0]), math.log(inp["box"][p][1])) for p in LEV.PARAMS]
    assert np.array_equal(seen["z0"], z0)


def test_g0_is_the_warm_starts_gain_at_theta_hat_on_the_full_data(monkeypatch):
    from sraging.calibrators import profile_jax2 as PJ
    P1 = PJ.PORTS["1.0.1"]
    calls = {}
    z = np.log([1e-3, 0.5, 0.02, 0.3])

    def warm(c, n, j, z0, **k):
        calls["warm"] = (j, k, np.asarray(z0))
        return {"status": "ok", "reason": None, "theta": list(np.exp(np.asarray(z0) + 0.01)), "cpu_s": 2.0}
    lls = {"v": [-100.0, -99.75]}

    def reduced(c, n, j, zs, **k):
        calls["ll"] = (j, len(zs), k)
        return list(lls["v"])
    monkeypatch.setattr(LEV, "refit_warm", warm)
    monkeypatch.setattr(LEV, "reduced_loglik", reduced)
    g = LEV.g0(None, "profile_jax2", z, port=P1)
    assert g["status"] == "ok" and g["g0"] == 0.25 and g["below_start"] is False
    assert calls["warm"][0] is None and calls["ll"][:2] == (None, 2)              # the full data, theta-hat first
    assert calls["warm"][1] == {"port": P1} and calls["ll"][2] == {"port": P1}
    assert np.array_equal(calls["warm"][2], z)
    lls["v"] = [-100.0, -100.5]                                                    # planted: below its start
    g = LEV.g0(None, "profile_jax2", z, port=P1)
    assert g["g0"] == -0.5 and g["below_start"] is True                          # recorded, never clipped
    monkeypatch.setattr(LEV, "refit_warm", lambda c, n, j, z0, **k: {"status": "failed", "reason": "planted",
                                                                       "theta": None, "cpu_s": 1.0})
    assert LEV.g0(None, "profile_jax2", z)["status"] == "failed"


def _checked(monkeypatch, port=None):
    from sraging import Cohort
    c = Cohort.from_csv(ITP0, delta=1.0, time_unit="days")
    n = c.n
    box = {p: (1e-12, 1e3) for p in LEV.PARAMS}
    seen = {}
    monkeypatch.setattr(LEV, "terms_and_derivatives", lambda cohort, name, z, **k: (seen.update(td=k), {
        "terms": -np.ones(n), "cond": 0.0, "G": np.ones((n, 4)), "H": -np.eye(4), "gradient": np.zeros(4),
        "hessian_source": "planted", "inputs": {"t": np.asarray(c.ages, float), "box": box}})[1])
    lab = {s: "no_flag" for s in LEV.QUANTITY_SETS}
    monkeypatch.setattr(LEV, "_drop", lambda *a, **k: (seen.update(drop=k), {
        "row": 1, "age": 0.0, "event": 1, "refits": {}, "refit": {"status": "ok", "start": "mle_stage"},
        "ratios": {}, "R": {s: 1.0 for s in LEV.QUANTITY_SETS}, "ll_ref": -1.0, "ll_new": -1.0,
        "ll_by_start": {}, "delta_total": 0.0, "delta_nats": 0.0, "labels": lab})[1])
    monkeypatch.setattr(LEV, "g0", lambda cohort, name, z, **k: (seen.update(g0=k), {"status": "ok", "g0": 0.0})[1])
    th = {"eta": 1e-3, "beta": 0.5, "epsilon": 0.02, "kappa": 0.3}
    kw = {} if port is None else {"port": port}
    return LEV.check(c, "profile_jax2", th, {}, **kw), seen


def test_the_check_records_g0_and_its_ports_version(monkeypatch):
    from sraging.calibrators import profile_jax2 as PJ
    P1 = PJ.PORTS["1.0.1"]
    out, seen = _checked(monkeypatch, P1)
    assert out["label"] == "no_flag" and out["g0"] == {"status": "ok", "g0": 0.0}
    assert out["provenance"]["version"] == "1.0.1" and out["version"] == LEV.VERSION
    assert seen["td"] == {"port": P1} and seen["drop"] == {"port": P1} and seen["g0"] == {"port": P1}
    out0, seen0 = _checked(monkeypatch)                             # the twin: the default port is 1.0.0's
    assert out0["provenance"]["version"] == "1.0.0" and seen0["td"] == {} and seen0["g0"] == {}


@pytest.mark.slow
def test_g0_on_a_real_cohort_is_not_below_its_start():
    from sraging import Cohort
    from sraging.calibrators import profile_jax2 as PJ
    c = Cohort.from_csv(ITP0, delta=1.0, time_unit="days")
    g = LEV.g0(c, "profile_jax2", np.log([3.6e-5, 0.032, 0.0024, 0.0011]), port=PJ.PORTS["1.0.1"])
    assert g["status"] == "ok" and g["g0"] >= -1e-9
