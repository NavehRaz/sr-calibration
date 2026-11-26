"""The two-pass gradient through the per-node mixture and the gradient in sd (tasks T013; research R5).

Pass 1 computes every node's points and mixes them; the parent runs jax.vjp(assemble) at the mixture; pass 2 runs each
node's VJP with the cotangent w_k c. The sd term is sum_k [c.P_k dw_k/dsd + w_k (c.dP_k/ds)(z_k + sd dz_k/dsd)]. Every
gradient here is checked against central differences first, and only then against another gradient."""
import numpy as np
import pytest

from sraging import Cohort, SRParams
from sraging.engine import mixture as MIX
from sraging.engine import quadrature as Q
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy
from sraging.engine.inputs import Heterogeneity

pytestmark = pytest.mark.engine
RNG = np.random.default_rng(3)
T = np.sort(np.floor(RNG.gamma(9.0, 14.0, 300)) + 1.0)
E = (RNG.random(300) < 0.9).astype(int)
C = Cohort.from_arrays(T, E, delta=1.0)
CW = Cohort.from_arrays(T, E, delta=1.0, window=(62.0, 186.0))
GRID = GridPolicy("fixed", N=200, dt=2.0 ** -8, tau_max=2.5, L=124.0)
P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)
FREE = ("eta", "beta", "epsilon", "kappa")
H = 1e-5


def cd_log(e, params, name, h=H, **kw):
    """Central difference of the mixture's value in log(name)."""
    d = params.as_dict()
    up = SRParams(**{**d, name: d[name] * np.exp(h)})
    dn = SRParams(**{**d, name: d[name] * np.exp(-h)})
    return (e.mixture_value(up, **kw).value - e.mixture_value(dn, **kw).value) / (2 * h)


def tol(v, g):
    """Round-off of the value over 2h, plus a truncation term relative to the gradient."""
    return 50 * 2.2e-16 * abs(v) / H + 1e-6 * abs(g) + 1e-8


@pytest.mark.parametrize("cohort,par,free,fixed,hazard", [
    (C, "xc", FREE, None, False),
    (C, "beta", FREE, None, False),
    (CW, "eta", FREE, None, False),
    (C, "epsilon", FREE + ("h_ext",), None, True),
    (C, "kappa", FREE + ("xc",), None, False),
])
def test_the_two_pass_gradient_matches_central_differences(cohort, par, free, fixed, hazard):
    e = Evaluator(cohort, numerics="v1", grid=GRID, free=free, fixed=fixed,
                  heterogeneity=Heterogeneity(parameter=par, sd=0.2, M=11, check=False))
    p = SRParams(**{**P.as_dict(), "h_ext": 2e-3 if hazard else 0.0})
    r = e.mixture_value(p, grad=True)
    assert set(r.grad) == set(free)
    for name in free:
        fd = cd_log(e, p, name)
        assert abs(fd - r.grad[name]) <= tol(r.value, r.grad[name]), (name, fd, r.grad[name])


@pytest.mark.parametrize("par,sd,M", [("xc", 0.2, 11), ("xc", 0.35, 17), ("eta", 0.3, 11), ("epsilon", 0.5, 25)])
def test_het_sd_grad_matches_central_differences_in_sd(par, sd, M):
    e = Evaluator(C, numerics="v1", grid=GRID, heterogeneity=Heterogeneity(parameter=par, sd=sd, M=M, check=False))
    r = e.mixture_value(P, grad=True, het_sd_grad=True)
    h = 1e-5

    def at(x):
        return MIX.mixture_value(e, P, het=Heterogeneity(parameter=par, sd=x, M=M, check=False)).value
    fd = (at(sd + h) - at(sd - h)) / (2 * h)
    assert abs(fd - r.grad["het_sd"]) <= 50 * 2.2e-16 * abs(r.value) / h + 1e-5 * abs(fd) + 1e-7
    assert set(r.grad) == set(FREE) | {"het_sd"}


def test_het_sd_grad_alone_gives_only_het_sd_and_the_same_value():
    e = Evaluator(C, numerics="v1", grid=GRID, heterogeneity=Heterogeneity(parameter="xc", sd=0.2, M=11, check=False))
    a = e.mixture_value(P)
    b = e.mixture_value(P, het_sd_grad=True)
    c = e.mixture_value(P, grad=True, het_sd_grad=True)
    assert a.grad is None and set(b.grad) == {"het_sd"} and b.grad["het_sd"] == c.grad["het_sd"]
    assert a.value == b.value == c.value                                       # the value never depends on grad


def test_the_per_node_and_vmapped_gradients_agree_after_both_pass_central_differences():
    """research R4: the per-node and today's vmapped gradients differed by 2.1e-7 relative on Sweden; here both are
    first checked against central differences, then against each other."""
    sd, M = 0.2, 11
    rule = Q.truncated_normal_rule(sd, M)
    new = Evaluator(C, numerics="v1", grid=GRID, heterogeneity=Heterogeneity(parameter="xc", sd=sd, M=M, check=False))
    old = Evaluator(C, numerics="v1", grid=GRID,
                    heterogeneity=Heterogeneity(parameter="xc", definition="normal_frac_sd", v=sd, M=M))
    old._het_z, old._het_w = np.asarray(rule.z), np.asarray(rule.w)
    rn = new.mixture_value(P, grad=True)
    vo, go = old.value_and_grad(P)
    for name in FREE:
        fd_new = cd_log(new, P, name)
        d = P.as_dict()
        fd_old = (old.value(SRParams(**{**d, name: d[name] * np.exp(H)})) -
                  old.value(SRParams(**{**d, name: d[name] * np.exp(-H)}))) / (2 * H)
        assert abs(fd_new - rn.grad[name]) <= tol(rn.value, rn.grad[name]), name
        assert abs(fd_old - go[name]) <= tol(vo, go[name]), name
        assert abs(rn.grad[name] - go[name]) <= 1e-7 * max(1.0, abs(go[name])), name
