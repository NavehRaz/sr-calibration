"""The drift-free closed forms (task T016; research R11 item 3): the eigen and image series are checked against
each other, and the density against the survival it must integrate to."""
import mpmath
import numpy as np
import pytest
from scipy.integrate import quad

from reference import diffusion as D

RHOS = [0.05, 0.3, 2.0]
TAUS = np.logspace(-6, 1, 36)
DPS = 50
MP_RESOLVED = mpmath.mpf("1e-30")      # min(S, 1 - S) above this is resolvable at 50 digits by both series


@pytest.mark.parametrize("rho", RHOS)
def test_the_two_series_agree_in_mpmath_wherever_resolvable(rho):
    """At 50 digits, |S_eigen - S_image| <= 1e-12 * min(S, 1 - S) wherever min(S, 1 - S) > 1e-30."""
    checked = 0
    for tau in TAUS:
        G_img = D.absorbed_image(tau, rho, dps=DPS)
        if G_img < MP_RESOLVED:
            continue                                  # 1 - S below 1e-30: the eigen series cannot resolve it
        S_eig = D.survival_eigen(tau, rho, dps=DPS)
        with mpmath.workdps(DPS):
            q = min(S_eig, G_img)
            if q < MP_RESOLVED:
                continue                              # S below 1e-30: the image series cannot resolve it
            assert abs(S_eig - (1 - G_img)) <= mpmath.mpf("1e-12") * q, (tau, S_eig, G_img)
            checked += 1
    assert checked >= 10


@pytest.mark.parametrize("rho", RHOS)
def test_the_two_series_agree_in_float64(rho):
    """In float64 both series agree to 3e-15 absolutely over tau in [1e-6, 10], and to 1e-12 relative to
    min(S, 1 - S) where that is >= 1e-3 (below it the subtraction 1 - S of the other series has no digits)."""
    rel_checked = 0
    for tau in TAUS:
        S_eig, G_img = D.survival_eigen(tau, rho), D.absorbed_image(tau, rho)
        diff = abs(S_eig - (1.0 - G_img))
        assert diff <= 1e-14
        q = min(S_eig, G_img)
        if q >= 1e-3:
            assert diff <= 1e-12 * q, (tau, S_eig, G_img)
            rel_checked += 1
    assert rel_checked >= 5


@pytest.mark.parametrize("rho", RHOS)
def test_the_density_series_agree(rho):
    for tau in TAUS:
        f_img = D.density_image(tau, rho, dps=DPS)
        if f_img < MP_RESOLVED:
            continue
        f_eig = D.density_eigen(tau, rho, dps=DPS)
        with mpmath.workdps(DPS):
            assert abs(f_eig - f_img) <= mpmath.mpf("1e-12") * f_img, tau
        if 0.1 <= rho * tau <= 1.0:                   # both converge fast and cancel little here
            a, b = D.density_eigen(tau, rho), D.density_image(tau, rho)
            assert abs(a - b) <= 1e-13 * abs(a)


@pytest.mark.parametrize("rho", RHOS)
def test_the_float64_choosers_are_relatively_accurate_everywhere(rho):
    """`survival`, `absorbed` and `density` against their 50-digit values: whichever of S and 1 - S is the
    smaller is relatively accurate (that is what the interval masses rely on); f likewise."""
    S, G, f = D.survival(TAUS, rho), D.absorbed(TAUS, rho), D.density(TAUS, rho)
    Sm, Gm, fm = D.survival(TAUS, rho, dps=DPS), D.absorbed(TAUS, rho, dps=DPS), D.density(TAUS, rho, dps=DPS)
    for i in range(len(TAUS)):
        smaller = (S[i], Sm[i]) if Sm[i] < Gm[i] else (G[i], Gm[i])
        for v, ref in (smaller, (f[i], fm[i])):
            if ref > 1e-300:
                assert abs(v - ref) <= 1e-12 * ref, (TAUS[i], v, ref)
            else:
                assert v < 1e-300                     # underflows in float64 rather than going wrong


INTERVALS = [(1e-4, 3e-4), (1e-3, 2e-3), (0.01, 0.02), (0.1, 0.12), (0.5, 0.6), (5.0, 6.0)]


@pytest.mark.parametrize("rho", RHOS)
def test_the_density_integrates_to_the_survival_drop(rho):
    for t0, t1 in INTERVALS:
        m = D.interval_mass(t0, t1, rho)
        if m < 1e-300:
            continue
        integral, _err = quad(lambda t: D.density(t, rho), t0, t1, epsabs=0.0, epsrel=1e-13, limit=200)
        assert integral == pytest.approx(m, rel=1e-10, abs=0.0), (t0, t1)


def test_an_early_interval_mass_has_no_cancellation():
    """At rho_eps = 0.3 the mass of [0.01, 0.02] is 1.4e-19: the naive S(t0) - S(t1) is 0 in float64, the
    image-series route agrees with 50 digits to 1e-12 relative; and masses far below 1e-300 exist in mpmath."""
    rho, t0, t1 = 0.3, 0.01, 0.02
    m, m_mp = D.interval_mass(t0, t1, rho), D.interval_mass(t0, t1, rho, dps=DPS)
    assert 1e-20 < m < 1e-18
    assert D.survival(t0, rho) - D.survival(t1, rho) == 0.0
    assert abs(m - m_mp) <= 1e-12 * m_mp
    tiny = D.interval_mass(1e-4, 2e-4, rho, dps=DPS)
    assert 0 < tiny < mpmath.mpf("1e-1000")


def test_limits_and_refusals():
    assert D.survival(0.0, 0.3) == 1.0 and D.absorbed(0.0, 0.3) == 0.0 and D.density(0.0, 0.3) == 0.0
    assert D.survival_eigen(1e-3, 0.3, dps=DPS) == pytest.approx(1.0, abs=1e-12)   # (4/pi)(1 - 1/3 + ...) = 1
    with pytest.raises(ValueError):
        D.survival(1.0, 0.0)
    with pytest.raises(ValueError):
        D.interval_mass(0.2, 0.1, 0.3)
