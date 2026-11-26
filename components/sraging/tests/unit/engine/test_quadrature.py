"""The truncated-normal Gauss rule (tasks T010; research R1; data-model "Rule"; register D-401).

`sraging.engine.quadrature.truncated_normal_rule(sd, M)` is the het-tradeoff prototype's rule
(`docs/planning/het_tradeoff/evidence/hetlib.py:tn_gauss`: the monic discretized Stieltjes procedure on 4,000
Gauss-Legendre points over [max(-1/sd, -8), 8], then Golub-Welsch) written in jax.numpy, with its sd-derivatives from
one implementation (jax.jvp). The NumPy prototype is kept here as the oracle."""
import hashlib

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm

from sraging.engine import quadrature as Q
from sraging.engine.inputs import Heterogeneity

pytestmark = pytest.mark.engine
SDS = (0.1, 0.2, 0.3, 0.5)
MS = (11, 31, 64, 128, 192)


def tn_gauss(v, M, s_min=0.0, n_fine=4000):
    """The prototype, verbatim (hetlib.py:24-44)."""
    Z_HI = 8.0
    a = max(-(1.0 - s_min) / v, -Z_HI) if v > 0 else -Z_HI
    x, w = leggauss(n_fine)
    z = 0.5 * (Z_HI - a) * x + 0.5 * (Z_HI + a)
    w = 0.5 * (Z_HI - a) * w * np.exp(-0.5 * z * z)
    w = w / w.sum()
    al, be = np.zeros(M), np.zeros(M)
    p_prev, p = np.zeros_like(z), np.ones_like(z)
    nrm_prev = 1.0
    for k in range(M):
        nrm = np.sum(w * p * p)
        al[k] = np.sum(w * z * p * p) / nrm
        be[k] = nrm / nrm_prev if k else 1.0
        p_prev, p = p, (z - al[k]) * p - (be[k] if k else 0.0) * p_prev
        nrm_prev = nrm
    J = np.diag(al) + np.diag(np.sqrt(be[1:]), 1) + np.diag(np.sqrt(be[1:]), -1)
    lam, vec = np.linalg.eigh(J)
    return lam, vec[0] ** 2


@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("sd", SDS)
def test_the_rule_is_the_prototypes(sd, M):
    r = Q.truncated_normal_rule(sd, M)
    z0, w0 = tn_gauss(sd, M)
    assert np.max(np.abs(r.z - z0)) <= 1e-13 and np.max(np.abs(r.w - w0)) <= 1e-13


@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("sd", SDS)
def test_the_rule_fields(sd, M):
    r = Q.truncated_normal_rule(sd, M)
    assert (r.definition, r.spread, r.M) == ("truncated_normal", sd, M)
    assert r.z.shape == r.w.shape == r.s.shape == r.dz_dsd.shape == r.dw_dsd.shape == (M,)
    assert abs(np.sum(r.w) - 1.0) <= 1e-15
    assert np.all(np.diff(r.z) > 0)                                                          # ascending
    assert np.array_equal(r.s, 1.0 + sd * r.z) and r.s_min == r.s[0] > 0.0
    assert all(np.all(np.isfinite(a)) for a in (r.z, r.w, r.s, r.dz_dsd, r.dw_dsd))         # finite to the cap


@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("sd", SDS)
def test_the_moments_are_the_truncated_normals(sd, M):
    """The mean of s and E[z^2] of the normal truncated to s = 1 + sd z > 0 (z > a = -1/sd), in closed form."""
    r = Q.truncated_normal_rule(sd, M)
    a = -1.0 / sd
    Z = 1.0 - norm.cdf(a)
    mean_z = norm.pdf(a) / Z
    ez2 = 1.0 + a * norm.pdf(a) / Z
    assert abs(np.sum(r.w * r.s) - (1.0 + sd * mean_z)) <= 3e-13
    assert abs(np.sum(r.w * r.z ** 2) - ez2) <= 3e-13


@pytest.mark.parametrize("sd", (0.2, 0.3, 0.5))
@pytest.mark.parametrize("M", (11, 64, 192))
def test_the_sd_derivatives_are_the_central_differences(sd, M):
    r = Q.truncated_normal_rule(sd, M)
    h = 1e-6
    up, dn = Q._compute(sd + h, M), Q._compute(sd - h, M)
    cz, cw = (up.z - dn.z) / (2 * h), (up.w - dn.w) / (2 * h)
    assert np.max(np.abs(r.dz_dsd - cz)) <= 1e-6 * max(1.0, np.max(np.abs(cz)))
    assert np.max(np.abs(r.dw_dsd - cw)) <= 1e-6 * max(1.0, np.max(np.abs(cw)))
    assert abs(np.sum(r.dw_dsd)) <= 1e-12                                     # the weights keep summing to 1


def test_below_sd_one_eighth_the_range_is_fixed_and_the_nodes_do_not_move():
    """At sd < 1/8 the truncation point -1/sd lies below -8, so [a, 8] = [-8, 8]: z and w do not depend on sd."""
    r = Q.truncated_normal_rule(0.1, 31)
    assert np.all(r.dz_dsd == 0.0) and np.all(r.dw_dsd == 0.0)
    assert np.array_equal(r.z, Q.truncated_normal_rule(0.11, 31).z)


def test_a_cache_hit_is_bit_identical_to_a_fresh_computation_and_read_only():
    a = Q.truncated_normal_rule(0.2, 57)
    b = Q.truncated_normal_rule(0.2, 57)
    fresh = Q._compute(0.2, 57)
    assert a is b                                                                             # the memo
    for f in ("z", "w", "s", "dz_dsd", "dw_dsd"):
        assert getattr(a, f).tobytes() == getattr(fresh, f).tobytes(), f
        assert not getattr(a, f).flags.writeable, f
    assert a.sha256 == fresh.sha256 and a.s_min == fresh.s_min


def test_the_base_grid_is_computed_once_per_process():
    Q.truncated_normal_rule(0.3, 17)
    Q.truncated_normal_rule(0.21, 25)
    Q._compute(0.37, 38)
    info = Q._base_grid.cache_info()
    assert info.misses == 1 and info.currsize == 1
    x, w = Q._base_grid(Q.BASE_POINTS)
    x0, w0 = leggauss(4000)
    assert x.tobytes() == x0.tobytes() and w.tobytes() == w0.tobytes() and not x.flags.writeable


def test_the_sha256_is_of_z_then_w():
    r = Q.truncated_normal_rule(0.2, 11)
    assert r.sha256 == hashlib.sha256(r.z.tobytes() + r.w.tobytes()).hexdigest()
    assert r.sha256 != Q.truncated_normal_rule(0.2, 17).sha256


@pytest.mark.parametrize("sd,M,option", [(0.0, 11, "sd"), (-0.1, 11, "sd"), (1.5, 11, "sd"), (float("nan"), 11, "sd"),
                                         (0.2, 0, "M"), (0.2, 193, "M"), (0.2, 11.0, "M"), (0.2, "auto", "M")])
def test_a_rule_outside_its_domain_is_refused_naming_it(sd, M, option):
    from sraging.errors import UnsupportedOptionError
    with pytest.raises(UnsupportedOptionError) as ex:
        Q.truncated_normal_rule(sd, M)
    assert ex.value.option == option
    Q.truncated_normal_rule(0.2, 11)                                                          # twin


def test_the_lowest_node_approaches_zero_and_stays_positive():
    """research R1: at sd = 0.2 the lowest node's s is about 0.087 at M = 11, 1.1e-3 at M = 64, 1.1e-4 at M = 192."""
    s = {M: Q.truncated_normal_rule(0.2, M).s_min for M in (11, 64, 192)}
    assert s[11] == pytest.approx(0.087, abs=2e-3) and s[64] == pytest.approx(1.1e-3, rel=0.1)
    assert s[192] == pytest.approx(1.1e-4, rel=0.1) and s[192] > 0


def test_default_is_truncated_normal():
    """D-401's regression test: a heterogeneity given only its spread is the truncated normal with automatic M, checked
    (FR-002); spec 002's definition must be named (OD-8)."""
    h = Heterogeneity(parameter="xc", sd=0.2)
    assert (h.definition, h.sd, h.v, h.M, h.check) == ("truncated_normal", 0.2, None, "auto", True)
