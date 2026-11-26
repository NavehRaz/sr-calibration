"""The engine's S(t_lo) clip at numerics "v1" (spec 006 FR-015; Naveh's Db, 2026-11-10; register D-108 retired to
ported-as-is; research R8; tasks T032). Spec 002-005's v1 conditioned on log+ S(t_lo) (floor e^-700): where S(t_lo)
underflows, -log S(t_lo) is huge and the windowed value turns positive -- a spurious maximum. v1 now clips at 1e-12 as
v033 and jax2 do, and reports when the clip binds. The twin: a theta with no underflow keeps its value bytes."""
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
EARLY = SRParams(eta=4e-2, beta=0.02, epsilon=4e-3, kappa=0.05)         # S(30) far below 1e-12
C = Cohort.from_arrays(T, E, delta=1.0, window=(30.0, 95.0))


def ev(numerics="v1"):
    return Evaluator(C, numerics=numerics, grid=GRID)


def test_the_underflow_is_clipped_and_is_no_longer_a_spurious_maximum():
    e = ev()
    v_early, v_ok = e.value(EARLY), e.value(P)
    assert v_early <= 0.0 and v_early < v_ok               # a conditional log-likelihood; before the clip it was > 0
    _, aux = e.terms(EARLY)
    assert aux["log_S_lo"] == pytest.approx(math.log(1e-12), rel=1e-15) and bool(aux["lo_clipped"])
    assert float(aux["S_lo"]) < 1e-12


def test_v1_and_v033_condition_on_the_same_clipped_s_lo():
    _, a1 = ev("v1").terms(EARLY)
    _, a33 = ev("v033").terms(EARLY)
    assert a1["log_S_lo"] == a33["log_S_lo"]


def test_the_report_names_the_clip_and_the_warning_fires():
    rep = ev().window_report(EARLY)
    assert rep["lo_clipped"] is True and rep["S_lo_min"] < 1e-12
    assert rep["warning"].code == "ENGINE_S_LO_CLIPPED" and rep["warning"].data["numerics"] == "v1"
    ok = ev().window_report(P)                                                       # the twin
    assert ok["lo_clipped"] is False and ok["warning"] is None and ok["S_lo_min"] > 1e-12


def test_without_underflow_nothing_moves():
    """The twin: at P, S(30) is far above 1e-12, so the clip does not bind and v1's value equals its unclipped form
    (the log+ floor), which is what spec 002-005's v1 computed."""
    _, aux = ev().terms(P)
    assert not bool(aux["lo_clipped"]) and aux["log_S_lo"] == math.log(float(aux["S_lo"]))


def test_a_window_report_without_a_window_is_refused():
    from sraging.errors import UnsupportedOptionError
    e = Evaluator(Cohort.from_arrays(T, E, delta=1.0), numerics="v1", grid=GRID)
    with pytest.raises(UnsupportedOptionError, match="window"):
        e.window_report(P)
