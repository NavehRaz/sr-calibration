"""The engine's window at hi = inf (spec 006 FR-014; register D-601; research R8; tasks T031). Today the value is
finite and the gradient NaN: `jnp.interp` at x = inf has a NaN derivative, and with the hazard on the unused log S_hi
back-propagates 0 x NaN through `jnp.where`. With hi = inf no S_hi term is assembled (the module's rule: an absent
feature adds no traced operation). The twin: a large finite on-grid hi, where nothing is censored at the edge either."""
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
PH = SRParams(eta=4e-4, beta=0.02, epsilon=4e-3, kappa=0.05, h_ext=1e-3)
FREE_H = ("eta", "beta", "epsilon", "kappa", "h_ext")


def vg(window, numerics, p=P, free=("eta", "beta", "epsilon", "kappa")):
    ev = Evaluator(Cohort.from_arrays(T, E, delta=1.0, window=window), numerics=numerics, grid=GRID, free=free)
    v, g = ev.value_and_grad(p)
    return float(v), {k: float(x) for k, x in g.items()} if isinstance(g, dict) else np.asarray(g, float)


def _finite(g):
    return all(math.isfinite(x) for x in (g.values() if isinstance(g, dict) else g))


@pytest.mark.parametrize("numerics", ["v1", "v033"])
@pytest.mark.parametrize("p,free", [(P, ("eta", "beta", "epsilon", "kappa")), (PH, FREE_H)], ids=["no-h", "h"])
def test_hi_inf_gives_a_finite_gradient_equal_to_a_large_finite_hi(numerics, p, free):
    v_inf, g_inf = vg((30.0, math.inf), numerics, p, free)
    v_big, g_big = vg((30.0, 1.0e6), numerics, p, free)                     # the twin
    assert math.isfinite(v_inf) and _finite(g_inf), g_inf
    assert v_inf == v_big
    assert g_inf == g_big if isinstance(g_inf, dict) else np.array_equal(g_inf, g_big)
