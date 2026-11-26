"""Old-code behaviour the engine keeps on purpose, pinned (register D-115; spec Not Covered)."""
import math

import numpy as np
import pytest

from sraging import Cohort, SRParams
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy

pytestmark = pytest.mark.engine


@pytest.mark.parametrize("numerics", ["v033", "v1"])
def test_d115_a_death_beyond_the_march_keeps_a_hazard_only_mass(numerics):
    """With an external hazard, a death recorded after the march stopped (S held at S_k < Gtol) scores
    S_k (e^{-h t} - e^{-h (t + delta)}) -- SRK's fp_loglik semantics, kept in both numerics. jax2 (B5) floors
    it instead, because S_k jumps when the stop index moves with theta (a sawtooth); that fix belongs with
    hazard fitting (spec 006)."""
    t = np.array([30.0, 40.0, 50.0, 400.0])
    e = np.array([1, 1, 1, 1])
    h = 2e-3
    ev = Evaluator(Cohort.from_arrays(t, e, delta=1.0), numerics=numerics,
                   grid=GridPolicy("fixed", N=120, dt=4e-3, tau_max=3.0, L=50.0), fixed={"h_ext": h})
    p = SRParams(eta=1e-3, beta=0.1, epsilon=4e-3, kappa=0.05, h_ext=h)
    terms, aux = ev.terms(p)
    assert bool(aux["beyond_march"][3]) and terms[3] > -700.0
    tau, S, _, k = ev.survival_on_grid(p)
    want = math.log(S[k] * (math.exp(-h * 400.0) - math.exp(-h * 401.0)))
    assert terms[3] == pytest.approx(want, rel=1e-9)
