"""Per-individual log-likelihood terms (spec 005 FR-025; HR-9; data-model "Terms"; tasks T043): on spec 002's small
cohort (tests/unit/engine/test_observation_identities.py's T, E), the terms sum to the engine's value within 1e-12, at
numerics "v1" and "v033"; a planted wrong normalisation (the mean) fails."""
import numpy as np
import pytest

from sraging import Cohort, SRParams
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy
from sraging.errors import UnsupportedOptionError
from sraging.terms import Terms, per_individual

T = np.array([20.0, 35.0, 41.0, 50.0, 52.0, 60.0, 61.0, 70.0, 75.0, 90.0])
E = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 0])
GRID = GridPolicy("fixed", N=120, dt=4e-3, tau_max=3.0, L=50.0)
P = SRParams(eta=4e-4, beta=0.02, epsilon=4e-3, kappa=0.05)
C = Cohort.from_arrays(T, E, delta=1.0)


@pytest.mark.parametrize("numerics", ["v1", "v033"])
def test_the_terms_sum_to_the_engines_value(numerics):
    t = per_individual(C, P, numerics=numerics, grid=GRID)
    value = Evaluator(C, numerics=numerics, grid=GRID).value(P)
    assert isinstance(t, Terms) and t.terms.shape == (C.n,) and t.numerics == numerics
    assert abs(float(np.sum(t.terms)) - value) <= 1e-12 * max(1.0, abs(value))
    assert t.total == value and not t.terms.flags.writeable and t.grid["N"] == 120
    planted = float(np.mean(t.terms))                                  # a wrong normalisation
    assert abs(planted - value) > 1e-12 * max(1.0, abs(value))


def test_the_default_grid_and_a_refused_numerics():
    t = per_individual(C, P)
    assert t.numerics == "v1" and abs(float(np.sum(t.terms)) - t.total) <= 1e-12 * abs(t.total)
    with pytest.raises(UnsupportedOptionError, match="v2"):
        per_individual(C, P, numerics="v2")
