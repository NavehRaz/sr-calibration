"""Heterogeneity in one parameter (T051, T053; spec FR-003, FR-017, FR-018; research R6; register D-116, D-117)."""
import numpy as np
import pytest
from numpy.polynomial.hermite_e import hermegauss

from sraging import Cohort, SRParams
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy
from sraging.engine.inputs import Heterogeneity
from sraging.errors import UnsupportedOptionError

pytestmark = pytest.mark.engine
RNG = np.random.default_rng(3)
T = np.sort(np.floor(RNG.gamma(9.0, 14.0, 300)) + 1.0)
E = (RNG.random(300) < 0.9).astype(int)
C = Cohort.from_arrays(T, E, delta=1.0)
GRID = GridPolicy("fixed", N=200, dt=2.0 ** -8, tau_max=2.5, L=124.0)
P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)


def ev(het, **kw):
    return Evaluator(C, numerics=kw.pop("numerics", "v033"), grid=GRID, heterogeneity=het, **kw)


def test_nodes_and_weights_are_the_old_codes():
    """SRK's _gh_nodes: probabilists' Gauss-Hermite, weights omega/sqrt(2 pi) summing to 1."""
    e = ev(Heterogeneity(parameter="xc", definition="normal_frac_sd", v=0.2, M=7))
    z, w = hermegauss(7)
    assert np.array_equal(e._het_z, z) and np.allclose(e._het_w, w / np.sqrt(2 * np.pi), rtol=0, atol=0)
    assert abs(np.sum(e._het_w) - 1.0) < 1e-14


@pytest.mark.parametrize("par", ["eta", "beta", "epsilon", "kappa", "xc"])
def test_one_node_is_homogeneous(par):
    homo = Evaluator(C, numerics="v033", grid=GRID).value(P)
    het = Heterogeneity(parameter=par, definition="normal_frac_sd", v=0.3, M=1)
    assert ev(het).value(P) == pytest.approx(homo, rel=1e-13)


def test_dropped_nodes_are_labelled_and_the_v_gradient_refused_there():
    """Register D-116: M = 11 has z = -5.19; at v = 0.2 that node's value is negative and is dropped."""
    e = ev(Heterogeneity(parameter="eta", definition="normal_frac_sd", v=0.2, M=11),
           free=("eta", "beta", "epsilon", "kappa", "het_v"))
    assert e.nodes_dropped(0.2) == 1 and e.nodes_dropped(0.15) == 0
    assert np.isfinite(e.value(P, het_v=0.2))
    with pytest.raises(UnsupportedOptionError, match="register D-116"):
        e.value_and_grad(P, het_v=0.2)
    v, g = e.value_and_grad(P, het_v=0.15)
    assert np.isfinite(v) and np.isfinite(g["het_v"])


def test_d117_the_definition_puts_mass_on_non_positive_values():
    """Register D-117: the definition is P(1 + vZ) ('normal_frac_sd'), which for any v > 0 gives a
    non-positive parameter value probability Phi(-1/v) > 0; the old code's quadrature meets it only through
    the nodes it drops (D-116). Pinned so that a change of definition (spec 004) is a visible change."""
    from scipy.stats import norm
    h = Heterogeneity(parameter="beta", definition="normal_frac_sd", v=0.25, M=11)
    assert h.definition == "normal_frac_sd"
    e = ev(h)
    z = np.asarray(e._het_z)
    assert np.array_equal(1.0 + 0.25 * z <= 0, z < -4.0) and int(np.sum(z < -4.0)) == 1
    assert norm.cdf(-1.0 / 0.25) == pytest.approx(3.167e-5, rel=1e-3) and e.nodes_dropped(0.25) == 1


def test_two_parameters_or_an_unknown_definition_are_refused():
    with pytest.raises(UnsupportedOptionError, match="only the Langevin engine accepts"):
        Heterogeneity(parameter=("eta", "beta"), definition="normal_frac_sd", v=0.2, M=7)
    with pytest.raises(UnsupportedOptionError, match="unknown heterogeneity definition 'lognormal'"):
        Heterogeneity(parameter="eta", v=0.2, M=7, definition="lognormal")
    with pytest.raises(UnsupportedOptionError, match="SRK's choose_M"):
        Heterogeneity(parameter="eta", definition="normal_frac_sd", v=0.2, M="auto")


def test_conditioning_divides_the_mixture_not_each_node():
    """A two-node mixture by hand: each row's term is log(sum_k w_k m_k) - log(sum_k w_k S_k(t_lo)), with
    the node quantities read from homogeneous evaluators at the node values."""
    cw = Cohort.from_arrays(T, E, delta=1.0, window=(62.0, 186.0))
    het = Heterogeneity(parameter="beta", definition="normal_frac_sd", v=0.3, M=2)
    got, _ = Evaluator(cw, numerics="v033", grid=GRID, heterogeneity=het).terms(P)
    z, w = hermegauss(2)
    w = w / w.sum()
    S_lo, S_t, S_t1 = [], [], []
    for zk in z:
        pk = SRParams(**{**P.as_dict(), "beta": P.beta * (1 + 0.3 * zk)})
        e0 = Evaluator(C, numerics="v033", grid=GRID)
        a = e0.terms(pk)[1]["S_i"]
        lo = Evaluator(Cohort.from_arrays([62.0, 30.0], [0, 1], delta=1.0), numerics="v033", grid=GRID).terms(pk)[1]["S_i"][0]
        hi = Evaluator(Cohort.from_arrays(T + 1.0, E, delta=1.0), numerics="v033", grid=GRID).terms(pk)[1]["S_i"]
        S_lo.append(lo), S_t.append(a), S_t1.append(hi)
    S_lo, S_t, S_t1 = np.array(S_lo), np.array(S_t), np.array(S_t1)
    mix_lo = float(w @ S_lo)
    inside = (T >= 62) & (T <= 186) & (E == 1)
    want = np.log(np.clip(w @ S_t - w @ S_t1, 1e-300, None)) - np.log(mix_lo)
    assert np.allclose(got[inside], want[inside], rtol=1e-11, atol=1e-11)
    per_node = np.log(w @ (S_t - S_t1)) - w @ np.log(S_lo)                 # the wrong way: node by node
    assert not np.allclose(got[inside], per_node[inside], rtol=1e-6)


def test_het_v_gradient_matches_central_differences():
    e = ev(Heterogeneity(parameter="xc", definition="normal_frac_sd", v=0.15, M=7),
           free=("eta", "beta", "epsilon", "kappa", "het_v"))
    v, g = e.value_and_grad(P, het_v=0.15)
    f, _ = e.objective()
    z = e.layout.pack({**P.as_dict(), "het_v": 0.15})
    i = e.layout.index("het_v")
    dz = np.zeros(z.size)
    dz[i] = 1e-5
    fd = (float(f(z + dz)) - float(f(z - dz))) / 2e-5
    assert abs(fd - g["het_v"]) <= 50 * 2.2e-16 * abs(v) / 1e-5 + 1e-6 * abs(g["het_v"])


def test_the_m_report_lists_early_death_terms():
    c = Cohort.from_arrays(np.concatenate([[2.0, 5.0, 9.0], T]), np.concatenate([[1, 1, 1], E]), delta=1.0)
    het = Heterogeneity(parameter="xc", definition="normal_frac_sd", v=0.2, M=7)
    e = Evaluator(c, numerics="v033", grid=GRID, heterogeneity=het)
    rep = e.m_report(P, early_S=0.99)
    assert rep["M"] == 7 and rep["M_more"] == 11 and len(rep["rows"]) == len(rep["dterm"]) > 0
    assert rep["max_abs_dterm"] == max(abs(x) for x in rep["dterm"])
