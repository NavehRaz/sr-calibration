"""Spec 006's row-index trap (research R4; tasks T009, T010): `fit_inputs`, the floor report and the leverage check
rebuild the fit's data from the cohort they are given. Given the windowed Cohort from `window_path`, all three see
the windowed rows the fit saw; given the original cohort they cannot silently score unwindowed rows (before T008 the
observation dropped the window, so floors scored the unwindowed likelihood and leverage's row check returned not_run;
now the frozen snap refuses the rows below lo). Every row the result reports maps back to the original cohort.

On the example ITP control (examples/data, n = 304), window [365, 1100] days, at the example fit's theta with a small h
for H. Fit-free; the floor report and the derivatives compile the windowed likelihood (seconds)."""
import json
import math
import os

import numpy as np
import pytest

from sraging import Cohort, SRParams
from sraging.calibrators import profile_jax2 as PJ
from sraging.compat.tester_f94ec2b.observation import apply_window

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.path.join(REPO, "examples", "data", "itp_c2016_het3_control_female.csv")
FIT = os.path.join(REPO, "examples", "results", "itp_c2016_het3_control_female_jax2.json")
P101 = PJ.PORTS["1.0.1"]
NAME = "profile_jax2_h"                       # H has the window on in 1.0.1; W's class joins at T016
WINDOW = (365.0, 1100.0)


@pytest.fixture(scope="module")
def case():
    c = Cohort.from_csv(CSV, delta=1.0, time_unit="days", window=WINDOW)
    th = json.load(open(FIT))["theta_hat"]
    theta = SRParams(eta=th["eta"], beta=th["beta"], epsilon=th["epsilon"], kappa=th["kappa"], h_ext=1e-5,
                     time_unit="days")
    wc, rec, rows = PJ.window_path(c, P101.classes[NAME], port=P101)
    return c, wc, rec, rows, theta


def _z(theta):
    return np.log([theta.eta, theta.beta, theta.epsilon, theta.kappa, theta.h_ext])


def test_the_case_drops_and_censors_rows(case):
    c, wc, rec, rows, _ = case
    assert rec["n_dropped_below_lo"] > 0 and rec["n_censored_above_hi"] > 0 and wc.n < c.n


def test_fit_inputs_of_the_windowed_cohort_are_the_windowed_rows(case):
    c, wc, rec, rows, _ = case
    inp = PJ.fit_inputs(wc, NAME, P101)
    a, e = apply_window(c.ages, c.events, WINDOW)
    prep = P101.observation_model.prepare(list(a), list(e), 1.0, WINDOW, window_on=True, relabel_on=True)
    assert np.array_equal(inp["t"], np.asarray(prep.ages, float)) and np.array_equal(inp["e"], prep.events)
    assert list(inp["window"]) == rec["window_eff"] and inp["t"].size == wc.n
    assert np.array_equal(inp["t"], wc.ages)          # the snap moves no age: the row check below can hold


def test_the_original_cohort_cannot_be_scored_unwindowed(case):
    """Planted (the trap): the original cohort, rows below lo included, is refused by the frozen snap rather than
    scored on its unwindowed rows, as fit_inputs, the floors and leverage did before T008."""
    c, *_ = case
    with pytest.raises(AssertionError, match="below lo_eff"):
        PJ.fit_inputs(c, NAME, P101)


def test_the_floor_report_scores_the_windowed_likelihood(case):
    from sraging.floors import report_jax2
    _c, wc, _rec, _rows, theta = case
    rep = report_jax2(wc, theta, NAME, port=P101)
    inp = PJ.fit_inputs(wc, NAME, P101)
    kw = inp["kw"]
    _f, _vg, tf = P101.fp_jax.make_loglik(inp["t"], inp["e"], inp["L"], N=kw["N"], dt=kw["dt"],
                                          tau_max=kw["tau_max"], Gtol=kw["Gtol"], sampling_dt=kw.get("sampling_dt"),
                                          hazard=True, window=inp["window"], with_terms=True,
                                          exact_from_F=bool(kw.get("exact_from_F")), early=kw.get("early"))
    d = tf(_z(theta))
    total = float(np.sum(np.asarray(d["term"], float)) + float(np.asarray(d["cond"])))
    assert rep["loglik_terms_total"] == total and float(np.asarray(d["cond"])) != 0.0      # the conditioning term
    assert all(1 <= x["row"] <= wc.n for x in rep["deaths"])


def test_the_leverage_terms_keep_the_windowed_rows_in_order(case):
    """The leverage check's own guard (leverage.py: the prepared data keep the cohort's rows in order) holds."""
    from sraging import leverage as LV
    _c, wc, _rec, _rows, theta = case
    td = LV.terms_and_derivatives(wc, NAME, _z(theta), port=P101)
    assert td["terms"].size == wc.n and np.array_equal(np.asarray(td["inputs"]["t"], float), wc.ages)
    assert np.all(np.isfinite(td["terms"])) and np.all(np.isfinite(td["G"]))


# ---------------------------------------------------------------------------------------- rows map back (T010)
def test_reported_rows_map_back_to_the_original_cohort(case):
    c, wc, _rec, rows, _ = case
    diag = {"floor_report": {"deaths": [{"row": 1, "age": float(wc.ages[0])}, {"row": wc.n, "age": float(wc.ages[-1])}]},
            "leverage": {"dropped": {"row": 3, "age": float(wc.ages[2])},
                         "influence": {"all4": {"top": {"row": 2, "age": float(wc.ages[1])},
                                                "median": {"row": 5, "age": float(wc.ages[4])}}}},
            "other": {"row": 1}}
    out = PJ.map_window_rows(diag, rows)
    for got, w in ((out["floor_report"]["deaths"][0], 1), (out["floor_report"]["deaths"][1], wc.n),
                   (out["leverage"]["dropped"], 3), (out["leverage"]["influence"]["all4"]["top"], 2),
                   (out["leverage"]["influence"]["all4"]["median"], 5)):
        assert got["row_windowed"] == w and got["row"] == int(rows[w - 1]) + 1
        assert c.ages[got["row"] - 1] == got["age"] or got["age"] == WINDOW[1]   # censored at hi keeps hi as its age
    assert out["other"] == {"row": 1}                    # only the floor report and the leverage check are mapped
    shifted = [w for w in range(1, wc.n + 1) if int(rows[w - 1]) != w - 1]    # planted: unmapped, these rows name
    assert shifted and c.ages[shifted[0] - 1] != wc.ages[shifted[0] - 1]       # another individual of the cohort
    assert PJ.map_window_rows(diag, None) is diag         # no window: unchanged
