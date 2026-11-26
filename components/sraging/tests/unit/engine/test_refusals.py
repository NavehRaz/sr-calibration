"""User Story 1's refusals (T020; spec FR-029, Edge Cases; constitution VIII): each names the input and
produces no result."""
import math

import numpy as np
import pytest

from sraging import Cohort, SRParams
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy
from sraging.errors import CohortDataError, UnsupportedOptionError

pytestmark = pytest.mark.engine
T = np.array([20.0, 35.0, 41.0, 50.0, 60.0])
E = np.array([1, 1, 0, 1, 1])
GRID = GridPolicy("fixed", N=60, dt=1e-2, tau_max=3.0, L=50.0)
P = SRParams(eta=4e-4, beta=0.02, epsilon=4e-3, kappa=0.05)


def c(**kw):
    return Cohort.from_arrays(T, E, delta=1.0, **kw)


def test_mismatched_time_units_are_refused():
    e = Evaluator(c(time_unit="days"), numerics="v033", grid=GRID)
    with pytest.raises(UnsupportedOptionError, match="parameters are in 'years' but the cohort is in 'days'"):
        e.value(SRParams(eta=4e-4, beta=0.02, epsilon=4e-3, kappa=0.05, time_unit="years"))


@pytest.mark.parametrize("h", [math.nan, -1e-3, math.inf])
def test_a_nan_negative_or_infinite_hazard_is_refused(h):
    """Register D-109: SRK treats NaN and -inf as 'off'; here they never reach the likelihood."""
    with pytest.raises(ValueError, match="h_ext"):
        SRParams(eta=4e-4, beta=0.02, epsilon=4e-3, kappa=0.05, h_ext=h)
    with pytest.raises(ValueError, match="fixed h_ext"):
        Evaluator(c(), numerics="v033", grid=GRID, fixed={"h_ext": h})


@pytest.mark.parametrize("kw,match", [
    (dict(numerics="v032"), "unknown numerics"),
    (dict(scheme="cn"), "unknown scheme"),
    (dict(free=("eta", "entry")), "snaps to the grid"),
    (dict(free=("eta", "beta", "epsilon", "kappa"), fixed={"kappa": 0.1}), "is free; it cannot also be fixed"),
    (dict(free=("eta", "beta", "epsilon")), "neither free nor fixed"),
    (dict(fixed={"gamma": 1.0}), "unknown fixed input"),
])
def test_unknown_or_contradictory_options_are_refused(kw, match):
    with pytest.raises(UnsupportedOptionError, match=match):
        Evaluator(c(), grid=GRID, **{"numerics": "v033", **kw})


def test_unknown_keywords_are_a_type_error():
    with pytest.raises(TypeError):
        Evaluator(c(), numerics="v033", grid=GRID, tail_extrap=True)


def test_a_fixed_value_the_parameters_contradict_is_refused():
    e = Evaluator(c(), numerics="v033", grid=GRID, fixed={"h_ext": 1e-3})
    with pytest.raises(UnsupportedOptionError, match="h_ext is fixed at 0.001"):
        e.value(P)


def test_nobody_at_risk_in_the_window_is_refused():
    with pytest.raises(CohortDataError, match="nobody is at risk"):
        Evaluator(c(window=(70.0, 80.0)), numerics="v033", grid=GRID)


def test_weights_of_the_wrong_length_are_refused():
    with pytest.raises(ValueError, match="one value per row"):
        Evaluator(c(), numerics="v033", grid=GRID).weighted_grad(P, np.ones(3))
