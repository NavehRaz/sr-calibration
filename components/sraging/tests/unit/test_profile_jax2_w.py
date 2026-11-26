"""`profile_jax2_w` (W): jax2 1.0.1's frozen machinery with the hazard off and the window on, a sraging composition
never run by jax2 (spec 006 FR-002 to FR-010; research R1, R2, R7; data-model.md "Calibrator version W"; tasks T012,
T014, T015).

Registration, class and kwargs run fast. The decomposition (gate M2) is fit-free: W's windowed value is J's terms on
the same prepared data plus the conditioning term. Fits are slow (opt-in: `pytest -m slow`): hi = inf against the
mapped and a larger on-grid hi, and no S(lo) warning on an ordinary window."""
import json
import math
import os

import numpy as np
import pytest

from sraging import Cohort, SRParams, fit_cohort, registry
from sraging.calibrators import profile_jax2 as PJ

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SMALL = os.path.join(REPO, "tests", "unit", "data", "small_cohort.csv")
ITP = os.path.join(REPO, "examples", "data", "itp_c2016_het3_control_female.csv")
ITP_FIT = os.path.join(REPO, "examples", "results", "itp_c2016_het3_control_female_jax2.json")
W = "profile_jax2_w"


# -------------------------------------------------------------------------------------- registration (T012)
def test_w_is_registered_at_its_own_version_only_and_is_not_the_default():
    assert registry.get(W).version == "1.0.1+w1"
    assert [n for n in registry.names() if n.startswith(W + "@")] == [f"{W}@1.0.1+w1"]
    assert registry.DEFAULT == "profile_jax2"
    assert W not in PJ.PORTS["1.0.1"].classes and PJ.PORT_W.version == "1.0.1+w1"
    assert PJ.PORT_W.kfix_classes == {} and list(PJ.PORT_W.classes) == [W]


def test_ws_class_is_js_constants_with_the_window_on():
    cls, J = PJ.MleProfileJax2W, PJ.PORTS["1.0.1"].classes["profile_jax2"]
    assert issubclass(cls, J) and PJ.PORT_W.classes[W] is cls
    assert cls.WINDOW is True and cls.HAZARD is False and cls.NAME == "sraging_profile_jax2_w"
    assert {k: v for k, v in vars(cls).items() if not k.startswith("__")} == {"NAME": "sraging_profile_jax2_w",
                                                                              "WINDOW": True}
    assert PJ.run_kwargs(cls) == {**PJ.run_kwargs(J), "window": True}
    assert PJ.switches_for(cls) == {**PJ.switches_for(J), "window": True}
    assert registry.get(W).settings["window"] is True and registry.get(W).numerics == "jax2"


def test_the_port_shares_101s_modules_and_settings_py_is_untouched():
    p, q = PJ.PORT_W, PJ.PORTS["1.0.1"]
    assert (p.claims, p.estimator, p.settings, p.fp_jax, p.observation_model) == \
        (q.claims, q.estimator, q.settings, q.fp_jax, q.observation_model)
    assert not hasattr(q.settings, "MleProfileJax2W")                      # W is sraging's class, not compat's


def test_the_composition_is_stated_and_js_sharp_note_is_ws():
    assert PJ.W_COMPOSITION == ("jax2 1.0.1 (6884fc8), hazard off, window on; a sraging composition, never run by "
                                "jax2")
    J = PJ.PORTS["1.0.1"].classes["profile_jax2"]
    diag = {"dt_floor": {"grid_rule": {"dt_in_scope": True}}}
    assert PJ.notes_for(PJ.PORT_W, PJ.MleProfileJax2W) == PJ.notes_for(PJ.PORTS["1.0.1"], J)
    assert PJ.sharp_limit_note(PJ.PORT_W.version, PJ.MleProfileJax2W, diag) == \
        PJ.sharp_limit_note("1.0.1", J, diag) is not None                    # research R2: W inherits J's limit


# ----------------------------------------------------------------------------------- the decomposition (T014)
@pytest.fixture(scope="module")
def decomposition():
    c = Cohort.from_csv(ITP, delta=1.0, time_unit="days", window=(365.0, 1100.0))
    wc, rec, _rows = PJ.window_path(c, PJ.MleProfileJax2W, port=PJ.PORT_W)
    inp = PJ.fit_inputs(wc, W, PJ.PORT_W)
    kw, fj = inp["kw"], PJ.PORT_W.fp_jax

    def mk(window):
        return fj.make_loglik(inp["t"], inp["e"], inp["L"], N=kw["N"], dt=kw["dt"], tau_max=kw["tau_max"],
                              Gtol=kw["Gtol"], sampling_dt=kw.get("sampling_dt"), hazard=False, window=window,
                              with_terms=True, exact_from_F=bool(kw.get("exact_from_F")), early=kw.get("early"))
    th = json.load(open(ITP_FIT))["theta_hat"]
    z0 = np.log([th["eta"], th["beta"], th["epsilon"], th["kappa"]])
    return inp, mk(inp["window"]), mk(None), z0


@pytest.mark.parametrize("shift", [(0.0, 0.0, 0.0, 0.0), (0.5, -0.3, 0.2, 2.0), (-1.0, 1.0, -0.5, 6.0),
                                   (3.0, -2.0, 1.0, 0.0)])
def test_ws_windowed_value_is_js_terms_plus_the_conditioning_term(decomposition, shift):
    """Gate M2: on the same prepared data, W's per-row terms are J's, bit for bit, and W's value is J's plus
    -n log S(lo) (exact at the fit's theta). The last shift drives S(lo) towards the clip; a planted wrong conditioning term fails."""
    inp, (fw, _vw, tw), (fj_, _vj, tj), z0 = decomposition
    z = z0 + np.asarray(shift)
    dw, dj = tw(z), tj(z)
    assert np.array_equal(np.asarray(dw["term"]), np.asarray(dj["term"]))
    cond = float(np.asarray(dw["cond"]))
    s_lo = float(np.asarray(dw["S_lo"]))
    assert cond == -inp["t"].size * math.log(min(max(s_lo, 1e-12), 1.0))
    # J's value (its own jitted sum) plus the term: exact at the fit's theta, within XLA's summation order (a few
    # ulp) far from it
    assert abs(float(fw(z)) - (float(fj_(z)) + cond)) <= 4 * math.ulp(abs(float(fw(z))))
    assert shift != (0.0, 0.0, 0.0, 0.0) or float(fw(z)) == float(fj_(z)) + cond
    assert float(fw(z)) != float(fj_(z)) + 0.99 * cond or cond == 0.0                          # planted


# ------------------------------------------------------------------------------ hi = inf and S(lo) (T015)
@pytest.mark.parametrize("s_lo,warns", [(1e-13, True), (9.99e-13, True), (1e-12, False), (0.3, False)])
def test_the_s_lo_warning_fires_below_the_frozen_clip_only(s_lo, warns):
    w = PJ.s_lo_warning(s_lo, 365.0, W)
    assert (w is not None) == warns
    if warns:
        assert w.code == "WINDOW_S_LO_CLIPPED" and w.data["S_lo_min"] == s_lo and w.data["lo"] == 365.0


def _small(window):
    return Cohort.from_csv(SMALL, delta=1.0, window=window, time_unit="days")


@pytest.mark.slow
def test_hi_inf_fits_as_the_mapped_end_and_as_a_larger_on_grid_end():
    """FR-007: the likelihood reads only lo_eff, so any on-grid hi past the last record gives the same fit."""
    inf = fit_cohort(_small((40.0, math.inf)), calibrator=W, combinations=False)
    hi = inf.diagnostics["window"]["hi_used"]
    assert inf.diagnostics["window"]["hi_mapped_from"] == "inf"
    for end in (hi, hi + 50.0):
        r = fit_cohort(_small((40.0, end)), calibrator=W, combinations=False)
        assert r.theta_hat.as_dict() == inf.theta_hat.as_dict() and r.loglik == inf.loglik
        assert [c.__dict__ for c in r.claims] == [c.__dict__ for c in inf.claims]


# ------------------------------------------------------------------------------ n >= 25,000 (T067)
A2 = os.path.join(REPO, "tests", "acceptance", "results", "a2_006.json")
RULES = os.path.join(REPO, "tests", "acceptance", "rules_006.toml")


@pytest.mark.parametrize("n,name,warns", [(25_000, W, True), (100_000, W, True), (24_999, W, False),
                                          (100_000, "profile_jax2_h", False)])
def test_the_large_n_warning_is_ws_from_25000_rows_given(n, name, warns):
    w = PJ.large_n_warning(n, name)
    assert (w is not None) == warns
    if warns:
        assert w.code == "W_LARGE_N" and w.data["n"] == n
        assert "eta 28 of 32 (14 at the box, 13 widened, 1 below the lead pair's walk)" in w.message
        assert "epsilon 29 of 29 (13 at the box, 16 widened)" in w.message and "eta 56 of 67" in w.message


def _why(text):
    return "box_limited" if text == "box_limited" else "widened" if text == "widened" else \
        "contradicted" if text.startswith("contradicted above") else text


def test_the_large_n_numbers_are_the_committed_records():
    """Naveh 2026-11-11 (A2's rule): A2 tripped the truncation rule; the stated numbers are a2_006.json's (which
    test_results_006.py rescores from its rows), split by why each end was truncated, and the threshold is the panel's.
    Naveh 2026-11-15 (sr-naveh M1): the rate below the split is A1's, on its cohorts planted at real life tables."""
    import collections
    import glob
    import tomllib
    d = json.load(open(A2))
    assert d["consequence"]["warning_n_ge_25000"] is True and d["gates"]["truncated_above"] is False
    for q, v in d["truncation"].items():
        why = collections.Counter(_why(t) for t in v["named"].values())
        assert {"closed": v["closed"], **{k: why.get(k, 0) for k in ("box_limited", "widened", "contradicted")}} == \
            PJ.W_LARGE_N_TRUNCATED[q] and sum(why.values()) == v["truncated"], q
    assert tomllib.load(open(RULES, "rb"))["panel"]["a2"]["n_at_least"] == PJ.W_LARGE_N_MIN
    a1 = json.load(open(A2.replace("a2_006", "a1_006")))
    rows = {os.path.basename(f)[:-5]: json.load(open(f))
            for f in glob.glob(os.path.join(os.path.dirname(A2), "gate_006", "a1", "*-real-*.json"))}
    assert rows and max(r["n"] for r in rows.values()) < PJ.W_LARGE_N_MIN
    for q, (t, c) in PJ.W_REAL_TABLES_BELOW.items():
        closed = [k for k, r in rows.items() if any(x["parameter"] == q and x["shape"] == "closed" for x in r["claims"])]
        assert (sum(k in a1["truncation"][q]["named"] for k in closed), len(closed)) == (t, c), q


@pytest.mark.slow
def test_a_windowed_fit_at_the_threshold_carries_the_large_n_warning(monkeypatch):
    """The wiring: with the threshold at the small cohort's rows given, its windowed fit warns; one row more does not.
    The fit itself is untouched (theta-hat and claims equal)."""
    c = _small((40.0, 70.0))
    monkeypatch.setattr(PJ, "W_LARGE_N_MIN", c.n)
    r = fit_cohort(c, calibrator=W, combinations=False)
    assert [w for w in r.warnings if w.code == "W_LARGE_N"][0].data["n"] == c.n
    monkeypatch.setattr(PJ, "W_LARGE_N_MIN", c.n + 1)
    q = fit_cohort(c, calibrator=W, combinations=False)
    assert "W_LARGE_N" not in [w.code for w in q.warnings]
    assert q.theta_hat.as_dict() == r.theta_hat.as_dict() and [x.__dict__ for x in q.claims] == \
        [x.__dict__ for x in r.claims]


@pytest.mark.slow
def test_an_ordinary_window_carries_no_s_lo_warning_and_reports_its_window():
    r = fit_cohort(_small((40.0, 70.0)), calibrator=W, combinations=False)
    assert "WINDOW_S_LO_CLIPPED" not in [w.code for w in r.warnings]
    win = r.diagnostics["window"]
    assert win["window_eff"] == [40.0, 70.0] and win["S_lo_at_theta_hat"] > 1e-12
    assert r.provenance["calibrator"] == {"name": W, "version": "1.0.1+w1", "composition": PJ.W_COMPOSITION}
