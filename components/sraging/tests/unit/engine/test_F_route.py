"""Numerics "v1"'s absorbed-mass route (T027; register D-004, D-106; research R3).

- F + S = 1 to accumulated round-off wherever S is resolved (F is the scheme's own mass balance);
- the second adjoint is the exact gradient: equal to autodiff of a fixed-length march carrying F;
- the cancellation is gone: under +-3-ulp moves of the parameters v0.3.3's value of a sharp cohort jumps by
  tens of nats while "v1"'s moves by round-off (the exact-time form is register D-106).
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
T = np.array([0.4, 0.9, 2.0, 3.0, 30.0, 41.0, 50.0, 52.0, 60.0, 61.0, 70.0, 75.0, 90.0])
E = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0])
GRID = GridPolicy("fixed", N=120, dt=4e-3, tau_max=2.5, L=50.0)
MODERATE = SRParams(eta=4e-4, beta=0.02, epsilon=4e-3, kappa=0.05)
SHARP = SRParams(eta=3e-3, beta=0.5, epsilon=4e-3, kappa=0.05)


def test_F_plus_S_is_one_to_round_off():
    g = GRID.grid(None)
    spec = ST.spec_for(g, carry_F=True)
    N = g.N
    yf = (jnp.arange(N) + 0.5) / N
    L = g.L
    theta = (L ** 2 * MODERATE.eta, L * MODERATE.epsilon, L * MODERATE.beta * yf / (yf + MODERATE.kappa))
    S_raw, F_raw = ST.make_march(spec, "value")(theta)
    S, F = np.asarray(S_raw), np.asarray(F_raw)
    ok = S > 1e-6
    # round-off accumulates over the march (measured 5.6e-13 after 625 steps here), far below C_S ~ 1e-9
    assert np.max(np.abs(F[ok] + S[ok] - 1.0)) < 1e-11


def _fixed_length_F(ev):
    asm, spec = ev._asm, ev._spec
    N, nsteps, dt, nstart = spec.N, spec.nsteps, spec.dt, spec.nstart
    h = 1.0 / N

    def ll(z):
        vals = ev._vals(z)
        theta = asm.theta(vals)
        p0 = jnp.zeros(N).at[0].set(1.0 / h)

        def body(carry, n):
            p, pp, F, Fm = carry
            q = ST._step_q(theta, n, p, pp, N, h, dt, nstart)
            Fn = ST._F_next(theta, n, q, F, Fm, N, h, dt, nstart)
            pn = jnp.maximum(q, 0.0)
            return (pn, p, Fn, F), (pn.sum() * h, Fn)

        z0 = jnp.zeros(())
        _, (S, F) = jax.lax.scan(body, (p0, jnp.zeros(N), z0, z0), jnp.arange(nsteps))
        S_raw = jnp.concatenate([jnp.array([p0.sum() * h]), S])
        F_raw = jnp.concatenate([jnp.zeros(1), F])
        terms, _ = asm.terms((S_raw, F_raw), vals)
        return jnp.sum(terms)
    return ll


@pytest.mark.parametrize("kind,budget", [("interval", None), ("exact", None), ("interval", 10_000)],
                         ids=["interval", "exact", "interval-checkpointed"])
def test_second_adjoint_equals_fixed_length_autodiff(kind, budget):
    c = Cohort.from_arrays(T, E, delta=1.0) if kind == "interval" else Cohort.from_arrays(T, E, continuous=True)
    ev = Evaluator(c, numerics="v1", grid=GRID, q_budget_bytes=budget)
    _, aux = ev.terms(SHARP)
    assert np.any(aux["use_F"] & (E == 1))                  # the F route is exercised
    z = ev.layout.pack(SHARP.as_dict())
    v, g = ev.objective()[1](z)
    v_ref, g_ref = jax.value_and_grad(_fixed_length_F(ev))(z)
    assert float(v) == pytest.approx(float(v_ref), rel=1e-13)
    g, g_ref = np.asarray(g), np.asarray(g_ref)
    assert np.all(np.abs(g - g_ref) <= 1e-6 * np.maximum(np.abs(g_ref), 1e-3 * np.max(np.abs(g_ref))))


def _ulp_spread(ev, p, k=3):
    z0 = ev.layout.pack(p.as_dict())
    f = ev.objective()[0]
    vals = []
    for i in range(z0.size):
        for s in range(-k, k + 1):
            z = z0.copy()
            for _ in range(abs(s)):
                z[i] = np.nextafter(z[i], np.inf if s > 0 else -np.inf)
            vals.append(float(f(z)))
    return max(vals) - min(vals)


@pytest.mark.parametrize("kind", ["interval", "exact"])
def test_the_cancellation_is_gone_under_ulp_moves(kind):
    c = Cohort.from_arrays(T, E, delta=1.0) if kind == "interval" else Cohort.from_arrays(T, E, continuous=True)
    spread33 = _ulp_spread(Evaluator(c, numerics="v033", grid=GRID), SHARP)
    spread1 = _ulp_spread(Evaluator(c, numerics="v1", grid=GRID), SHARP)
    assert spread1 <= 1e-9
    assert spread33 > 1.0 or kind == "exact"         # the round-off flip v0.3.3 has (interval deaths)


def test_exact_time_density_from_F():
    """Register D-106: where S's operands are kept, the density is v0.3.3's bit for bit; where F's are used,
    the term is finite and smooth (above), and the two agree where both are resolved."""
    c = Cohort.from_arrays(T, E, continuous=True)
    t33, _ = Evaluator(c, numerics="v033", grid=GRID).terms(MODERATE)
    t1, a1 = Evaluator(c, numerics="v1", grid=GRID).terms(MODERATE)
    late = (T > 40) & (E == 1)
    assert np.allclose(t1[late], t33[late], rtol=1e-9, atol=1e-9)
    assert np.all(np.isfinite(t1)) and np.all(t1[E == 1] > -700)
