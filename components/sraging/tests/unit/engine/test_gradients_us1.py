"""The hand-written reverse sweep is the exact gradient of the discrete likelihood (T022; spec SC-003).

Reference: JAX autodiff of a FIXED-LENGTH march (lax.scan, no early stop, autodiff through the
tridiagonal solve) feeding the same assembly. The grid is chosen so the early stop never triggers, so the
two compute the same function; they must agree to 1e-6 relative (Clarifications). Central differences
are a second, looser check: their error is about u*|loglik|/h (round-off) plus h^2 (truncation).
Covers the hazard (free), entry, the window and the checkpointed sweep.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sraging import Cohort, SRParams
from sraging.engine import stepper as ST
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy

pytestmark = pytest.mark.engine
T = np.array([20.0, 35.0, 41.0, 50.0, 52.0, 60.0, 61.0, 70.0, 75.0, 90.0, 33.0, 47.0])
E = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1])
GRID = GridPolicy("fixed", N=80, dt=5e-3, tau_max=2.5, L=50.0)
P = SRParams(eta=4e-4, beta=0.02, epsilon=4e-3, kappa=0.05, h_ext=2e-3)
FREE5 = ("eta", "beta", "epsilon", "kappa", "h_ext")


def fixed_length_ll(ev):
    asm, spec = ev._asm, ev._spec
    N, nsteps, dt, nstart = spec.N, spec.nsteps, spec.dt, spec.nstart
    h = 1.0 / N

    def ll(z):
        vals = ev._vals(z)
        theta = asm.theta(vals)
        p0 = jnp.zeros(N).at[0].set(1.0 / h)

        def body(carry, n):
            p, pp = carry
            q = ST._step_q(theta, n, p, pp, N, h, dt, nstart)
            pn = jnp.maximum(q, 0.0)
            return (pn, p), pn.sum() * h

        _, S = jax.lax.scan(body, (p0, jnp.zeros(N)), jnp.arange(nsteps))
        S_raw = jnp.concatenate([jnp.array([p0.sum() * h]), S])
        terms, _ = asm.terms(S_raw, vals)
        return jnp.sum(terms)
    return ll


CONFIGS = {
    "hazard": dict(cohort=dict(delta=1.0), free=FREE5),
    "entry+hazard": dict(cohort=dict(delta=1.0, entry=np.array([0, 5, 30, 40, 0, 55, 5, 60, 70, 80, 33.5, 0.0])),
                         free=FREE5),
    "window+hazard": dict(cohort=dict(delta=1.0, window=(30.0, 72.0)), free=FREE5),
    "exact times": dict(cohort=dict(continuous=True), free=FREE5),
    "checkpointed": dict(cohort=dict(delta=1.0), free=FREE5, q_budget_bytes=10_000),
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_custom_vjp_equals_fixed_length_autodiff(name):
    cfg = CONFIGS[name]
    ev = Evaluator(Cohort.from_arrays(T, E, **cfg["cohort"]), free=cfg["free"], numerics="v033", grid=GRID,
                   q_budget_bytes=cfg.get("q_budget_bytes"))
    assert (ST.grad_mode(ev._spec, cfg.get("q_budget_bytes")) == "vjp_ckpt") == (name == "checkpointed")
    z = ev.layout.pack(P.as_dict())
    v, g = ev.objective()[1](z)
    ref = fixed_length_ll(ev)
    v_ref, g_ref = jax.value_and_grad(ref)(z)
    assert float(v) == pytest.approx(float(v_ref), rel=1e-13)
    g, g_ref = np.asarray(g), np.asarray(g_ref)
    assert np.all(np.abs(g - g_ref) <= 1e-6 * np.maximum(np.abs(g_ref), 1e-3 * np.max(np.abs(g_ref))))


def test_gradient_against_central_differences():
    ev = Evaluator(Cohort.from_arrays(T, E, delta=1.0, window=(30.0, 72.0)), free=FREE5, numerics="v033",
                   grid=GRID)
    z = ev.layout.pack(P.as_dict())
    f = ev.objective()[0]
    v, g = ev.objective()[1](z)
    hstep = 1e-5
    bound = 50 * 2.2e-16 * abs(float(v)) / hstep + 1e-6 * np.max(np.abs(np.asarray(g)))
    for i in range(z.size):
        dz = np.zeros(z.size)
        dz[i] = hstep
        fd = (float(f(z + dz)) - float(f(z - dz))) / (2 * hstep)
        assert abs(fd - float(np.asarray(g)[i])) <= bound, ev.layout.names[i]


def test_weighted_gradient_of_all_ones_is_the_gradient():
    ev = Evaluator(Cohort.from_arrays(T, E, delta=1.0), free=FREE5, numerics="v033", grid=GRID)
    _, g = ev.value_and_grad(P)
    wg = ev.weighted_grad(P, np.ones(T.size))
    for k in g:
        assert wg[k] == pytest.approx(g[k], rel=1e-12)
