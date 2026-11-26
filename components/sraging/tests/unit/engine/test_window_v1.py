"""The window's legacy rules and their registered change under "v1" (T030; register D-107), and the lo floor both
numerics share since spec 006 (register D-108, retired: v1 clips S(t_lo) at 1e-12 as v033 does)."""
import math

import numpy as np
import pytest

from sraging import Cohort, SRParams
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy

pytestmark = pytest.mark.engine
T = np.array([20.0, 35.0, 41.0, 50.0, 52.0, 60.0, 65.0, 70.0, 75.0, 90.0])
E = np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 0])
GRID = GridPolicy("fixed", N=120, dt=4e-3, tau_max=3.0, L=50.0)
P = SRParams(eta=4e-4, beta=0.02, epsilon=4e-3, kappa=0.05)


def terms(numerics, cohort, p=P):
    return Evaluator(cohort, numerics=numerics, grid=GRID).terms(p)


def log_S(age, p=P, numerics="v1"):
    c = Cohort.from_arrays(np.array([age, 30.0]), np.array([0, 1]), delta=1.0)
    return terms(numerics, c, p)[0][0]


def test_window_close_is_censored_under_v1():
    """Register D-107. The death recorded at t_hi = 65 occurred in [65, 66): after the window closes. v033
    (SRK) scores it as a death over the whole bin; v1 as alive at t_hi."""
    c = Cohort.from_arrays(T, E, delta=1.0, window=(30.0, 65.0))
    t33, _ = terms("v033", c)
    t1, aux = terms("v1", c)
    i = int(np.flatnonzero(T == 65.0)[0])
    lo = aux["log_S_lo"]
    assert t1[i] == pytest.approx(log_S(65.0) - lo, rel=1e-13)
    assert t33[i] != t1[i]
    others = np.ones(T.size, bool)
    others[i] = False
    # nothing else moves beyond D-004's round-off on the death masses (S_lo >= 1e-12 here)
    assert np.allclose(t33[others], t1[others], rtol=1e-9, atol=0)


def test_the_lo_floor_is_srks_clip_in_both_numerics():
    """Register D-108 (retired by spec 006, Naveh's Db). With S_obs(t_lo) < 1e-12 (the window opens after nearly
    everyone died), both numerics clip the conditioning at log 1e-12. Planted: spec 002-005's v1 rule, the common log+
    floor, gives a different (lower) log S_lo there, which made the windowed value a spurious maximum. Above 1e-12 the
    two rules are equal."""
    from sraging.engine import observation as OBS
    c = Cohort.from_arrays(T, E, delta=1.0, window=(30.0, 95.0))
    early = SRParams(eta=4e-2, beta=0.02, epsilon=4e-3, kappa=0.05)     # S(30) far below 1e-12
    _, a33 = terms("v033", c, early)
    _, a1 = terms("v1", c, early)
    assert a33["log_S_lo"] == a1["log_S_lo"] == pytest.approx(math.log(1e-12), rel=1e-15)
    assert float(OBS._logp(a1["S_lo"])) < math.log(1e-12)                  # the old v1 rule, planted
    _, b33 = terms("v033", c)
    _, b1 = terms("v1", c)
    assert b33["log_S_lo"] == b1["log_S_lo"] == float(OBS._logp(b1["S_lo"]))
