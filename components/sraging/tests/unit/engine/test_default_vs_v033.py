"""The constitution VII CI guard (T028; spec FR-024, SC-008; research R14; register D-004, D-106-D-108).

Numerics "v1" may differ from "v033" only where a register entry's condition says so: covered terms within
the entry's bound, every other term equal in bits, and the gradient over the uncovered terms equal to
round-off. Shown to fail on planted unregistered changes (constitution V).
"""
import os
import tomllib

import numpy as np
import pytest

from engine_cases import cohort, theta_points
from sraging import Cohort, SRParams, divergences
from sraging.engine import guard
from sraging.engine import observation as OBS
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy

pytestmark = [pytest.mark.parity, pytest.mark.engine]
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PANEL = tomllib.load(open(os.path.join(REPO, "tests", "acceptance", "panel_002_guard.toml"), "rb"))
CONDS = guard.conditions([{"id": e.id, "condition": e.condition} for e in divergences.load()])


def test_the_register_supplies_conditions_for_every_v1_change():
    """Spec 006 retired D-108: v1 clips S(t_lo) at 1e-12 as v033 does, so no condition covers the lo floor; a lo-floor
    difference would now be unregistered."""
    assert {eid for eid, _ in CONDS} == {"D-004", "D-106", "D-107"}


@pytest.mark.parametrize("cid", PANEL["cohorts"])
def test_guard_passes_and_names_terms(oracle_v033, cid):
    c = cohort(cid)
    e33, e1 = Evaluator(c, numerics="v033"), Evaluator(c, numerics="v1")
    for label, th in theta_points(cid):
        rep = guard.compare(e33, e1, SRParams(**th), CONDS, f"{cid}:{label}")
        assert rep.ok, (label, rep.violations[:5], rep.grad_uncovered_rel)


# -- synthetic cases where D-106 (exact times) and D-107/D-108 (windows) fire ---------------------------
T = np.array([0.4, 0.9, 2.0, 30.0, 41.0, 50.0, 52.0, 60.0, 61.0, 70.0, 75.0, 90.0])
E = np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0])
GRID = GridPolicy("fixed", N=200, dt=2e-3, tau_max=2.5, L=50.0)
SHARP = SRParams(eta=8e-3, beta=2.0, epsilon=2e-2, kappa=0.05)       # S ~ 1 long after the first deaths
MODERATE = SRParams(eta=4e-4, beta=0.02, epsilon=4e-3, kappa=0.05)    # resolved masses ~0.01, head deaths small


def _pair(c, p=SHARP):
    return Evaluator(c, numerics="v033", grid=GRID), Evaluator(c, numerics="v1", grid=GRID)


def test_exact_time_cohort_is_guarded():
    e33, e1 = _pair(Cohort.from_arrays(T, E, continuous=True))
    rep = guard.compare(e33, e1, SHARP, CONDS)
    assert rep.ok, rep.violations[:5]
    assert rep.covered["D-106"] > 0


def test_window_close_and_lo_floor_are_guarded():
    c = Cohort.from_arrays(T, E, delta=1.0, window=(30.0, 70.0))   # the death recorded at 70 reaches past it
    e33, e1 = _pair(c)
    rep = guard.compare(e33, e1, SHARP, CONDS)
    assert rep.ok, rep.violations[:5]
    assert rep.covered["D-107"] == 1


# -- planted unregistered changes must fail, naming the terms (constitution V) ----------------------------
def _planted(monkeypatch, which):
    real = OBS.Assembly.assemble

    def assemble(self, pts, vals=None):
        t, aux = real(self, pts, vals)
        if self.numerics == "v1":
            died = self.died
            if which == "censored":
                t = t + np.where(np.asarray(died), 0.0, 1e-12)
            elif which == "hazard":
                t = t * (1.0 + 1e-12)
            elif which == "use_F":                    # beyond C_S/mass for every covered death here
                t = t + np.where(np.asarray(died), 1e-3, 0.0)
        return t, aux
    monkeypatch.setattr(OBS.Assembly, "assemble", assemble)


def test_the_unplanted_synthetic_case_passes():
    e33, e1 = _pair(Cohort.from_arrays(T, E, delta=1.0))
    rep = guard.compare(e33, e1, MODERATE, CONDS)
    assert rep.ok and rep.covered["D-004"] > 0


@pytest.mark.parametrize("which", ["censored", "hazard", "use_F"])
def test_a_planted_unregistered_change_fails_the_guard(monkeypatch, which):
    _planted(monkeypatch, which)
    c = Cohort.from_arrays(T, E, delta=1.0)
    e33, e1 = _pair(c)
    rep = guard.compare(e33, e1, MODERATE, CONDS)
    assert not rep.ok
    assert rep.violations and all(isinstance(r, int) for r, _, _ in rep.violations)


# -- review T064 (sr-naveh N1-N3, N8) -----------------------------------------------------------------------
def test_a_hazard_cohort_is_guarded():
    """N1: with an external hazard, v1's composition e^{-H}[m + S_hi(1 - e^{-h dl})] applies only where use_F
    holds; every other death is v033's own expression, bit for bit (before the fix: 1-4 ulp, uncovered)."""
    from sraging.engine import stepper as ST  # noqa: F401
    c = Cohort.from_arrays(T, E, delta=1.0)
    p = SRParams(**{**MODERATE.as_dict(), "h_ext": 1e-2})
    kw = dict(grid=GRID, fixed={"h_ext": 1e-2})
    rep = guard.compare(Evaluator(c, numerics="v033", **kw), Evaluator(c, numerics="v1", **kw), p, CONDS)
    assert rep.ok, rep.violations[:5]
    assert rep.covered["D-004"] > 0 and rep.use_F_disagree == 0


def test_a_floored_v033_term_gets_no_free_pass(monkeypatch):
    """N2: the old bound let a term on v033's floor take any v1 value. Now m_v033 = 0 there and v1 may rescue it
    only to a mass the S route could not resolve: planting 1e-9 of extra mass on the rescued deaths fails."""
    c = Cohort.from_arrays(T, E, delta=1.0)
    e33, e1 = _pair(c)
    t33, _ = e33.terms(SHARP)
    assert np.any(t33[E == 1] <= -700 + 1e-9)                     # SHARP floors the head deaths under v033
    real = OBS.Assembly.assemble

    def assemble(self, pts, vals=None):
        t, aux = real(self, pts, vals)
        if self.numerics == "v1":
            import jax.numpy as jnp
            m = aux["mass"] + jnp.where(jnp.asarray(self.died), 1e-9, 0.0)
            t = jnp.where(jnp.asarray(self.died), jnp.log(m), t)
            aux = {**aux, "mass": m}
        return t, aux
    monkeypatch.setattr(OBS.Assembly, "assemble", assemble)
    rep = guard.compare(*_pair(c), SHARP, CONDS)
    assert not rep.ok and any(eid == "D-004" for _, eid, _ in rep.violations)


def test_rescued_terms_have_their_gradient_checked(monkeypatch):
    """N3: a planted stop_gradient on the F route leaves the values (and every bit-level check) unchanged but
    breaks the rescued terms' gradient; central differences must catch it."""
    c = Cohort.from_arrays(T, E, delta=1.0)
    rep0 = guard.compare(*_pair(c), SHARP, CONDS)
    assert rep0.ok and rep0.rescued > 0 and rep0.rescued_cd_ratio <= 1.0
    real = OBS.Assembly.points

    def points(self, march_out, diagnostics=False):
        import jax
        pts = real(self, march_out, diagnostics)
        if self.numerics == "v1" and "F_hi" in pts:
            pts = {**pts, "F_hi": jax.lax.stop_gradient(pts["F_hi"]), "F_st": jax.lax.stop_gradient(pts["F_st"])}
        return pts
    monkeypatch.setattr(OBS.Assembly, "points", points)
    rep = guard.compare(*_pair(c), SHARP, CONDS)
    assert not rep.ok and any(eid == "grad_rescued" for _, eid, _ in rep.violations)


def test_use_F_is_checked_against_v033s_own_survival(monkeypatch):
    """N8: the guarded code supplies use_F; the guard recomputes it from v033's S. A planted use_F that is
    true everywhere makes the late deaths disagree far from the switch."""
    real = OBS.Assembly.assemble

    def assemble(self, pts, vals=None):
        if self.numerics == "v1" and "F_hi" in pts:
            pts = {**pts, "F_hi": pts["F_hi"] * 0.0 - 1.0 + pts["S_st"]}     # F_hi < S_st everywhere
        return real(self, pts, vals)
    c = Cohort.from_arrays(T, E, delta=1.0)
    late = SRParams(**{**MODERATE.as_dict(), "eta": 1.2e-3})          # S < 0.5 at the late deaths
    _, a1 = Evaluator(c, numerics="v1", grid=GRID).terms(late)
    assert np.any(~a1["use_F"][E == 1])                                # the plant changes something
    monkeypatch.setattr(OBS.Assembly, "assemble", assemble)
    rep = guard.compare(*_pair(c), late, CONDS)
    assert not rep.ok and rep.use_F_disagree > 0


def test_entry_with_a_hazard_is_guarded():
    """T1 at d8dcf87 (review T064 follow-up): with entry and a hazard the masses of the deaths outside D-004 were
    bit-identical, but XLA rounded the log 1 ulp differently in the two programs; uncovered terms are held to
    guard.UNCOVERED_ULPS ulps, not bits. (A real change is far larger: the planted ones are >= 1e-12.)"""
    en = np.array([0, 0.2, 1.0, 10.0, 20.0, 25.0, 30.0, 40.0, 41.0, 50.0, 60.0, 70.0])
    c = Cohort.from_arrays(T, E, delta=1.0, entry=en)
    p = SRParams(**{**MODERATE.as_dict(), "eta": 1.2e-3, "h_ext": 1e-2})
    kw = dict(grid=GRID, fixed={"h_ext": 1e-2})
    rep = guard.compare(Evaluator(c, numerics="v033", **kw), Evaluator(c, numerics="v1", **kw), p, CONDS)
    assert rep.ok, rep.violations[:5]
