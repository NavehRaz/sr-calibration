"""Closed forms for the drift-free SR process: an independent reference (task T016; research R11 item 3).

The SR model with eta = 0 and beta = 0 is a pure diffusion. In the solver's units (algorithms section 2,
eq. s2-sde-dl) it is

    dy = sqrt(2 rho_eps) dW,    y(0) = 0 at the reflecting wall,    absorbed at y = 1,

so the density of surviving mass obeys  p_tau = rho_eps p_yy  on [0, 1] with  p_y(0) = 0,  p(1) = 0  and
p(y, 0) = delta(y) carrying unit mass inside [0, 1]. S(tau) = int_0^1 p dy, and the first-passage density is
f = -dS/dtau. Everything below depends on rho_eps and tau only through  x = rho_eps * tau.

Derivation
----------
*Unfolding the wall.* With no drift, reflection at 0 is the absolute value of a free process: if X is a
Brownian motion on the line with dX = sqrt(2 rho_eps) dW, X(0) = 0, then y = |X| is the reflected process,
and y reaches 1 exactly when X leaves (-1, 1). So S(tau) = P(X stays in (-1, 1) up to tau). Write q(x, tau)
for the density of X killed on leaving (-1, 1): q_tau = rho_eps q_xx, q(+-1) = 0, q(x, 0) = delta(x), and
S = int_{-1}^{1} q dx (p(y) = q(y) + q(-y) = 2 q(y) on [0, 1], q being even).

*Eigen series (good for large x).* The Dirichlet eigenfunctions on (-1, 1) that are even in x are
cos(mu_k x), mu_k = (2k + 1) pi / 2, with eigenvalues rho_eps mu_k^2 and int_{-1}^{1} cos^2(mu_k x) dx = 1.
Expanding delta(x) gives coefficients cos(mu_k * 0) / 1 = 1, so

    q(x, tau) = sum_{k>=0} cos(mu_k x) exp(-rho_eps mu_k^2 tau),
    S(tau)    = sum_k exp(-rho_eps mu_k^2 tau) int_{-1}^{1} cos(mu_k x) dx
              = sum_k 2 sin(mu_k)/mu_k exp(...)
              = sum_{k>=0} 4 (-1)^k / ((2k + 1) pi) * exp(-rho_eps (2k + 1)^2 pi^2 tau / 4),       (E)

using sin(mu_k) = (-1)^k. Check: at tau = 0 the sum is (4/pi)(1 - 1/3 + 1/5 - ...) = (4/pi)(pi/4) = 1.
Differentiating term by term,

    f(tau) = pi rho_eps sum_{k>=0} (-1)^k (2k + 1) exp(-rho_eps (2k + 1)^2 pi^2 tau / 4).          (E')

*Image series (good for small x).* The free kernel g(x, tau) = exp(-x^2 / (4 rho_eps tau)) / sqrt(4 pi
rho_eps tau) is made to vanish at +-1 by alternating images at the even integers:

    q(x, tau) = sum_{n in Z} (-1)^n g(x - 2n, tau).

At x = 1 the images n and 1 - n sit at mirror points 2n and 2 - 2n about 1 with opposite signs, so they
cancel in pairs; at x = -1 the pairs are n and -1 - n. (Probabilistically this is inclusion-exclusion over
the alternating sequence of first passages to +1 and -1, the reflection principle applied repeatedly.)
Integrating each image over (-1, 1), with s = 2 sqrt(rho_eps tau),

    int_{-1}^{1} g(x - 2n) dx = [erfc((2n - 1)/s) - erfc((2n + 1)/s)] / 2,

and collecting the n and -n terms with erfc(-z) = 2 - erfc(z) leaves

    1 - S(tau) = 2 sum_{j>=0} (-1)^j erfc((2j + 1) / (2 sqrt(rho_eps tau))).                       (I)

Check: the j = 0 term, 2 erfc(1 / (2 sqrt(x))), is twice the one-sided Levy hitting probability
P(max_{s<=tau} X_s >= 1) = erfc(1 / (2 sqrt(x))), one for each wall; the alternating corrections remove the
paths that touch both. With d/dtau erfc(c / (2 sqrt(rho_eps tau))) = c exp(-c^2/(4 rho_eps tau)) /
(2 sqrt(pi rho_eps) tau^{3/2}),

    f(tau) = sum_{j>=0} (-1)^j (2j + 1) / sqrt(pi rho_eps tau^3) * exp(-(2j + 1)^2 / (4 rho_eps tau)).   (I')

The j = 0 term is twice the Levy first-passage density to one wall. The two series are checked against
each other in `tests/unit/engine/test_reference_diffusion.py` (to 1e-12 relative to min(S, 1 - S) wherever
that number is resolvable at the working precision), and f is checked to integrate to the survival drop.

Numerics
--------
- (E) gives S itself to full relative precision when S is small (large x); (I) gives 1 - S to full relative
  precision when 1 - S is small (small x), because erfc is accurate relative to its value down to 1e-308.
  `survival`, `absorbed` and `density` switch at X_SWITCH = 0.38, where S ~ 1/2, so each returns the
  smaller of S and 1 - S without cancellation.
- `interval_mass(t0, t1)` = S(t0) - S(t1) is taken as the difference of whichever pair has the smaller
  operands: (1 - S)(t1) - (1 - S)(t0) when (1 - S)(t1) < S(t0) (early intervals), else S(t0) - S(t1). The
  mass of an early interval is therefore a difference of two small, relatively accurate numbers, not of two
  numbers equal to 1 in their leading digits. (A residual relative error ~ eps * (1 - S)(t1) / m remains for
  intervals so narrow that the two operands agree in many digits; pass `dps` to remove it.)
- Each series is summed until its terms have passed their peak and fall below tol times the largest term
  (tol = 1e-18 in float64, 10^-(dps + 2) in mpmath); float terms eventually underflow to 0. Where a series
  is ill-conditioned (E at small x, I at large x: an alternating sum much smaller than its terms) it still
  converges but returns only absolute accuracy; the choosers never use it there.
- `dps=None` computes in float64 with `math`; an integer `dps` computes in mpmath at that many digits and
  returns mpmath numbers (use it for masses below 1e-300 or for cancellation-free checks).

The engine's reflecting wall sits at y = -h/2 (algorithms section 2.2: node 0 has a full-width control
volume), so a discrete march converges to these closed forms at O(h); a test of the engine should measure
that rate rather than assume it (research R11).
"""
from __future__ import annotations

import math

import mpmath
import numpy as np

X_SWITCH = 0.38            # x = rho_eps * tau where S ~ 1/2: below it use (I), above it (E)
_KMAX = 2_000_000          # safety cap on the number of series terms
_TOL64 = 1e-18


def _sum_alternating(term, tol):
    """sum_{k>=0} term(k) for an alternating series whose |terms| rise to a peak, then fall to 0."""
    total = term(0)
    peak = abs(total)
    prev = peak
    for k in range(1, _KMAX):
        t = term(k)
        total += t
        a = abs(t)
        peak = max(peak, a)
        if a <= prev and a <= tol * peak:
            return total
        prev = a
    raise RuntimeError("diffusion: series did not converge")


class _Lib:
    """Elementary functions and tolerance for one working precision (float64 or mpmath)."""

    def __init__(self, dps):
        self.dps = dps
        if dps is None:
            self.num, self.exp, self.erfc, self.sqrt, self.pi = float, math.exp, math.erfc, math.sqrt, math.pi
            self.tol = _TOL64
        else:
            self.num, self.exp, self.erfc, self.sqrt = mpmath.mpf, mpmath.exp, mpmath.erfc, mpmath.sqrt
            self.pi = mpmath.pi
            self.tol = mpmath.mpf(10) ** (-(int(dps) + 2))


def _with_precision(fn):
    """Run fn(lib, *args) in float64 (dps=None) or inside mpmath.workdps(dps)."""
    def wrapped(*args, dps=None):
        if dps is None:
            return fn(_Lib(None), *args)
        with mpmath.workdps(int(dps)):
            return fn(_Lib(dps), *args)
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


def _x(lib, tau, rho_eps):
    tau, rho_eps = lib.num(tau), lib.num(rho_eps)
    if not (rho_eps > 0):
        raise ValueError(f"rho_eps must be > 0, got {rho_eps}")
    if not (tau >= 0):
        raise ValueError(f"tau must be >= 0, got {tau}")
    return tau, rho_eps, rho_eps * tau


# ---- the four series, each a function of (tau, rho_eps) --------------------------------------------------
def _survival_eigen(lib, tau, rho_eps):
    tau, rho_eps, x = _x(lib, tau, rho_eps)
    if x == 0:
        return lib.num(1)
    c = lib.pi ** 2 * x / 4
    return _sum_alternating(
        lambda k: (4 if k % 2 == 0 else -4) / ((2 * k + 1) * lib.pi) * lib.exp(-c * (2 * k + 1) ** 2), lib.tol)


def _absorbed_image(lib, tau, rho_eps):
    tau, rho_eps, x = _x(lib, tau, rho_eps)
    if x == 0:
        return lib.num(0)
    s = 2 * lib.sqrt(x)
    return 2 * _sum_alternating(lambda j: (1 if j % 2 == 0 else -1) * lib.erfc((2 * j + 1) / s), lib.tol)


def _density_eigen(lib, tau, rho_eps):
    tau, rho_eps, x = _x(lib, tau, rho_eps)
    c = lib.pi ** 2 * x / 4
    return lib.pi * rho_eps * _sum_alternating(
        lambda k: (1 if k % 2 == 0 else -1) * (2 * k + 1) * lib.exp(-c * (2 * k + 1) ** 2), lib.tol)


def _density_image(lib, tau, rho_eps):
    tau, rho_eps, x = _x(lib, tau, rho_eps)
    if x == 0:
        return lib.num(0)
    pref = 1 / lib.sqrt(lib.pi * rho_eps * tau ** 3)
    return pref * _sum_alternating(
        lambda j: (1 if j % 2 == 0 else -1) * (2 * j + 1) * lib.exp(-(2 * j + 1) ** 2 / (4 * x)), lib.tol)


@_with_precision
def survival_eigen(lib, tau, rho_eps):
    """S(tau) from the eigen series (E): relatively accurate where S is small."""
    return _survival_eigen(lib, tau, rho_eps)


@_with_precision
def absorbed_image(lib, tau, rho_eps):
    """1 - S(tau) from the image series (I): relatively accurate where 1 - S is small."""
    return _absorbed_image(lib, tau, rho_eps)


@_with_precision
def survival_image(lib, tau, rho_eps):
    """S(tau) = 1 - (I): accurate only in absolute terms (cancels when S is small)."""
    return 1 - _absorbed_image(lib, tau, rho_eps)


@_with_precision
def absorbed_eigen(lib, tau, rho_eps):
    """1 - S(tau) = 1 - (E): accurate only in absolute terms (cancels when 1 - S is small)."""
    return 1 - _survival_eigen(lib, tau, rho_eps)


@_with_precision
def density_eigen(lib, tau, rho_eps):
    """f(tau) = -dS/dtau from the eigen series (E')."""
    return _density_eigen(lib, tau, rho_eps)


@_with_precision
def density_image(lib, tau, rho_eps):
    """f(tau) = -dS/dtau from the image series (I')."""
    return _density_image(lib, tau, rho_eps)


# ---- the stable choosers ---------------------------------------------------------------------------------
def _map(fn, tau, *args, dps=None):
    if np.ndim(tau) == 0:
        return fn(float(tau) if dps is None else tau, *args, dps=dps)
    vals = [fn(float(t) if dps is None else t, *args, dps=dps) for t in np.asarray(tau).ravel()]
    out = np.array(vals, dtype=float if dps is None else object)
    return out.reshape(np.shape(tau))


def _early(tau, rho_eps):
    return float(rho_eps) * float(tau) < X_SWITCH


def _survival1(tau, rho_eps, dps=None):
    return survival_image(tau, rho_eps, dps=dps) if _early(tau, rho_eps) else survival_eigen(tau, rho_eps, dps=dps)


def _absorbed1(tau, rho_eps, dps=None):
    return absorbed_image(tau, rho_eps, dps=dps) if _early(tau, rho_eps) else absorbed_eigen(tau, rho_eps, dps=dps)


def _density1(tau, rho_eps, dps=None):
    return density_image(tau, rho_eps, dps=dps) if _early(tau, rho_eps) else density_eigen(tau, rho_eps, dps=dps)


def survival(tau, rho_eps, dps=None):
    """S(tau): (E) where S < ~1/2 (relatively accurate there), else 1 - (I)."""
    return _map(_survival1, tau, rho_eps, dps=dps)


def absorbed(tau, rho_eps, dps=None):
    """1 - S(tau): (I) where 1 - S < ~1/2 (relatively accurate there), else 1 - (E)."""
    return _map(_absorbed1, tau, rho_eps, dps=dps)


def density(tau, rho_eps, dps=None):
    """f(tau) = -dS/dtau: (I') for x < X_SWITCH, else (E'); both converge fast and cancel little there."""
    return _map(_density1, tau, rho_eps, dps=dps)


def interval_mass(t0, t1, rho_eps, dps=None):
    """S(t0) - S(t1) for 0 <= t0 <= t1, from the pair of operands that are smaller (no cancellation early).

    Early intervals: (1 - S)(t1) - (1 - S)(t0) from the image series; late ones: S(t0) - S(t1) from the
    eigen series. With `dps` the whole computation is done in mpmath at that precision.
    """
    if not (0 <= float(t0) <= float(t1)):
        raise ValueError(f"need 0 <= t0 <= t1, got t0={t0}, t1={t1}")
    G1 = _absorbed1(t1, rho_eps, dps=dps)
    S0 = _survival1(t0, rho_eps, dps=dps)
    if dps is None:
        if G1 < S0:
            return G1 - _absorbed1(t0, rho_eps)
        return S0 - _survival1(t1, rho_eps)
    with mpmath.workdps(int(dps)):
        if G1 < S0:
            return G1 - _absorbed1(t0, rho_eps, dps=dps)
        return S0 - _survival1(t1, rho_eps, dps=dps)
