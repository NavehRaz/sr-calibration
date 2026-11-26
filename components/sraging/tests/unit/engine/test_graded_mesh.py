"""Graded meshes without segments (FR-020; review T064 N4; task T071): the general march on any node list.

A node list equal to the uniform mesh gives the uniform march's values (different code, the same scheme); a
graded start is checked against tests/reference/numpy_scheme.py on the same nodes (S and the death and
censoring terms, like for like); and an early-death term converges as the graded start is refined.
"""
import numpy as np
import pytest

from reference import numpy_scheme as NS
from sraging import Cohort, SRParams
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy, graded_start

pytestmark = pytest.mark.engine
RNG = np.random.default_rng(11)
T = np.sort(np.floor(RNG.gamma(9.0, 14.0, 300)) + 1.0)
E = (RNG.random(300) < 0.9).astype(int)
L, N = 124.0, 160
P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)
RHO = (L ** 2 * P.eta, L * P.beta, L * P.epsilon, P.kappa)


def test_graded_start_nodes():
    t = np.asarray(graded_start(1e-6, 1.05, 4e-3, 2.0))
    d = np.diff(t)
    assert t[0] == 0.0 and d[0] == 1e-6 and np.all(d > 0) and t[-1] >= 2.0
    assert np.allclose(d[-10:], 4e-3) and np.all(np.diff(d[d < 4e-3 * (1 - 1e-12)]) > 0)


@pytest.mark.parametrize("scheme", ["bdf2", "be"])
def test_uniform_nodes_through_the_general_march_equal_the_uniform_march(scheme):
    c = Cohort.from_arrays(T, E, delta=1.0)
    dt, tau_max = 4e-3, 2.5
    nodes = tuple(np.arange(int(tau_max / dt) + 1) * dt)
    a = Evaluator(c, numerics="v033", grid=GridPolicy("fixed", N=N, dt=dt, tau_max=tau_max, L=L), scheme=scheme)
    b = Evaluator(c, numerics="v033", grid=GridPolicy("fixed", N=N, nodes=nodes, tau_max=tau_max, L=L), scheme=scheme)
    assert getattr(a, "_gspec", None) is None and b._gspec is not None
    assert b.value(P) == pytest.approx(a.value(P), rel=1e-12)
    va, ga = a.value_and_grad(P)
    vb, gb = b.value_and_grad(P)
    assert all(gb[k] == pytest.approx(ga[k], rel=1e-9) for k in ga)


@pytest.mark.parametrize("numerics", ["v033", "v1"])
@pytest.mark.parametrize("scheme", ["bdf2", "be"])
def test_a_graded_start_matches_the_numpy_reference_on_the_same_nodes(numerics, scheme):
    nodes = graded_start(1e-7, 1.05, 4e-3, 2.5)
    c = Cohort.from_arrays(T, E, delta=1.0)
    ev = Evaluator(c, numerics=numerics, grid=GridPolicy("fixed", N=N, nodes=nodes, tau_max=2.5, L=L), scheme=scheme)
    res = NS.march(N, np.asarray(nodes), theta=RHO, scheme=scheme, nstart=20, Gtol=1e-11)
    tau, S, _, k = ev.survival_on_grid(P)
    S_ref, _ = NS.monotone(res)
    n = min(S.size, S_ref.size)
    assert np.max(np.abs(S[:n] - S_ref[:n])) <= 1e-10
    terms, _ = ev.terms(P)
    route = "S" if numerics == "v033" else "auto"
    ref = NS.loglik(res, T / L, E == 1, 1.0 / L, route=route)
    assert abs(float(np.sum(terms)) - ref) <= 1e-7


def test_an_early_death_converges_as_the_graded_start_is_refined():
    """A death in [0, 1) day: its term on a uniform grid is the grid's (review T064 N4); a graded start that
    resolves the first day converges, and refining it further moves the term by less than 0.1 nat."""
    t = np.r_[0.0, T]
    e = np.r_[1, E]
    c = Cohort.from_arrays(t, e, delta=1.0)
    vals = []
    for dt0 in (1e-8, 1e-9, 1e-10):
        nodes = graded_start(dt0, 1.02, 4e-3, 2.5)
        ev = Evaluator(c, numerics="v1", grid=GridPolicy("fixed", N=N, nodes=nodes, tau_max=2.5, L=L))
        vals.append(float(ev.terms(P)[0][0]))
    uni = Evaluator(c, numerics="v1", grid=GridPolicy("fixed", N=N, dt=4e-3, tau_max=2.5, L=L)).terms(P)[0][0]
    assert abs(vals[-1] - vals[-2]) < 0.1 and abs(vals[-1] - uni) > 1.0
