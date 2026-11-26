"""FLOORED_DEATHS for jax2 fits (spec 003, T020; FR-011; research R12).

Terms come from the port's own likelihood (compat.jax2 fp_jax `make_loglik(with_terms=True)`) on the fit's own
grid, refined by dt/4^k with N fixed. Under fix B there is no "cancellation" category. The cohort and the head-death
theta are spec 002's (tests/unit/test_floors.py): at HEAD the two earliest deaths have essentially no mass. Fast:
no fit, a few compiled objectives.
"""
import numpy as np
import pytest

from sraging import floors
from test_floors import COHORT, HEAD, QUIET


def cats(rep):
    return {d["row"]: d["category"] for d in rep["deaths"]}


def test_the_head_deaths_are_impossible_with_their_mass_and_share():
    rep = floors.report_jax2(COHORT, HEAD, "profile_jax2")
    c = cats(rep)
    assert c.get(1) == "impossible" and c.get(2) == "impossible"
    assert "cancellation" not in c.values() and set(rep["counts"]) == set(floors.CATEGORIES_JAX2)
    for d in rep["deaths"]:
        if d["row"] in (1, 2):
            assert d["mass_refined"] < floors.IMPOSSIBLE_MASS and 0 < d["ll_share"] <= 1
    assert "share of the log-likelihood" in floors.message(rep)


def test_the_graded_mesh_resolves_the_early_deaths_even_at_a_quiet_theta():
    """Measured 2026-10-05: at QUIET spec 002's v1 report lists only the age-400 death, because v0.3.3's coarse grid
    overstates the early deaths' mass (the grid error spec 002 measured: register D-004 early_residuals). jax2's
    graded early mesh resolves them (masses 3.6e-19 and 3.3e-15; refining the step moves neither term by 1 nat), so
    they are listed as impossible under the fit's own likelihood; the age-400 death lies beyond the march."""
    rep = floors.report_jax2(COHORT, QUIET, "profile_jax2")
    c = cats(rep)
    assert c.get(1) == "impossible" and c.get(2) == "impossible" and c.get(14) == "horizon"
    for d in rep["deaths"]:
        if d["row"] in (1, 2):
            assert abs(d["term_refined"] - d["term_fit"]) < floors.LIST_NATS


def test_a_planted_threshold_changes_the_listing(monkeypatch):
    """Constitution V: with the impossible-mass threshold far below the head deaths' mass, they are not impossible."""
    monkeypatch.setattr(floors, "IMPOSSIBLE_MASS", 1e-300)
    monkeypatch.setattr(floors, "CANDIDATE_MASS", 1e-300)
    c = cats(floors.report_jax2(COHORT, HEAD, "profile_jax2"))
    assert c.get(1) != "impossible" or c.get(2) != "impossible"


def test_a_death_above_the_candidate_cut_on_the_fit_grid_is_still_refined(monkeypatch):
    """sr-naveh implementation review 11: FR-011 is about the refined mass. With the candidate cut planted below the
    early deaths' fit-grid mass, the screen (fit-grid mass < 1e-6) still refines them and lists them as impossible;
    with the screen planted off, they are not listed."""
    monkeypatch.setattr(floors, "CANDIDATE_MASS", 1e-300)
    rep = floors.report_jax2(COHORT, QUIET, "profile_jax2")
    c = cats(rep)
    assert c.get(1) == "impossible" and c.get(2) == "impossible", c
    assert all(d["screened"] for d in rep["deaths"] if d["row"] in (1, 2)) and rep["n_screened"] >= 2
    monkeypatch.setattr(floors, "SCREEN_MASS", 1e-300)
    c = cats(floors.report_jax2(COHORT, QUIET, "profile_jax2"))
    assert 1 not in c and 2 not in c, c




def test_the_itp_age0_death_alone_is_refined_and_impossible_at_the_fit_without_it():
    """Measured 2026-10-05: at the profile_jax2 theta-hat of the ITP example WITHOUT its added age-0 death, that death
    is the only candidate; its refinement once asked early_schedule for a march shorter than the graded start
    (ValueError). It must refine past the graded mesh and read impossible."""
    import os
    from sraging import Cohort
    from sraging.params import SRParams
    here = os.path.dirname(os.path.abspath(__file__))
    itp = Cohort.from_csv(os.path.join(here, "..", "data", "itp_plus_age0.csv"), delta=1.0, time_unit="days")
    th = SRParams(eta=3.57913138545272e-05, beta=0.03166671731968555, epsilon=0.002369154067613347,
                  kappa=0.0010000000000000002, h_ext=0.0, time_unit="days")
    rep = floors.report_jax2(itp, th, "profile_jax2")
    a0 = [d for d in rep["deaths"] if d["age"] == 0.0]
    assert a0 and a0[0]["category"] == "impossible" and a0[0]["mass_refined"] < floors.IMPOSSIBLE_MASS, rep["deaths"]
    assert rep["refinement"]["steps"], rep["refinement"]
