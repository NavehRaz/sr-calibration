"""The Fokker-Planck log-likelihood in JAX, a twin of `SRK.fp_likelihood.fp_loglik(scheme="bdf2")`.

mle_profile_jax2 PACKAGE COPY (Phase B). This file is the jax2 rc1 candidate's `fp_jax.py` (md5 5834e106,
frozen at tests/fixtures/candidate_rc1/code/fp_jax.py) plus, all in `make_loglik` and the two small
`hazard_*_core` helpers (the march, its custom VJP, `_truncate*` and `_death_mass` are rc1's, untouched):
  * B2: the external hazard h = exp(z[4]) (survival S(t) e^{-ht}) and the fit window (left truncation at
    the snapped lo_eff), PHASE_B_DESIGN.txt section 2. With both OFF `make_loglik` runs rc1's assembly
    unchanged, so the compiled programs are rc1's (checked byte-for-byte at the fit level by
    tests/identity_rc1.py);
  * a trace-time shape check: `ll_of_z` raises ValueError unless z.shape == (5,) with the hazard, (4,)
    without. JAX CLAMPS out-of-bounds indexing, so without it a length-5 z would silently be read as its
    first four entries, and a length-4 z with the hazard on as h = exp(z[3]) = kappa;
  * `with_terms=True`: a third jitted function returning the per-observation terms (tests only);
  * B5 (1.0.0.dev3): with the hazard on, a death beyond the march's early stop scores the -700 floor, as
    rc1 scores it without the hazard (`hw_parts`; it removes a sawtooth in theta that B2 had introduced);
  * rc2 I3 (the floor early-time fix): `make_loglik(early=dict(dt0=, ratio=))` marches a GRADED early-time
    mesh with variable-step BDF2 (`early_schedule`, `_make_march_g`; see the block above `_death_mass`).
    `early=None` (the default) runs the uniform march unchanged.
  * rc2 I4, `make_loglik(exact_from_F=True)` (False = dev3): the EXACT-TIME density (sampling_dt=None) from the
    absorbed mass F where F's difference has the smaller operands (`_density_F`; fix B's idea for exact ages), on
    both paths. MEASURED (rc2 I4, hazard-k0.01-p00-c0-s0 with exact ages, L2's 60 points): the value under
    +-3-ulp changes of z spreads up to 2.1e-5 nats with the hazard and 1e4-1e5 nats without it (early deaths'
    S-difference densities are round-off: 0 -> the -700 floor, or noise; v0.3.3 has the same exact-time path) --
    the exact-time twin of the interval stall fix B removed; with F: <= 7.4e-10 nats on both paths; forward- vs
    reverse-mode AD 1.56e-7 -> 4.5e-11 (hazard; L2' exact 300/300 PASS) and 0.46 -> 7.4e-14 (hazard-free);
    terms with S < 1 - 1e-3 are unchanged to 1.2e-12 nats. Assumes the UNIFORM grid (`dt`); exact ages never meet
    a graded mesh (the floored ceiling, and so rc2 I3's early mesh, needs a sampling_dt).
The rest of this docstring is rc1's ("NOT ported: external hazard, time_range" is rc1's statement; B2 ports
both, validated against `fp_loglik_physical(external_hazard=, time_range=)` at rung L1).

WHY IT EXISTS. `mle_profile` spends essentially all of its CPU inside derivative-free Nelder-Mead
searches, one likelihood solve per step. This module makes the likelihood differentiable, so the same
searches run on EXACT gradients -- the idea from Avi's `saturated-removal-manifold` (`fpt_jax.py` +
`fit_mm_jax.py`), where replacing a finite-difference Jacobian with `jax.jacfwd` bought 1.46x.

WHAT IS PORTED, and the NumPy function each piece mirrors:

    NumPy (SRK/fp_likelihood.py)          here
    ------------------------------------  ------------------------------------------------------
    fpt_full(scheme="bdf2", nstart=20)    `_march_*`: BE startup, then BDF2, p<0 clipped
      ... early stop at S < Gtol          the march STOPS there too (lax.while_loop), as NumPy breaks
    fpt_full_density                      cummin, clip, p = -gradient(S)            (`_truncate`)
    _interp_S_p(tail_extrap=False)        jnp.interp with the same left/right values
    _interval_mass (sampling_dt)          S(t) - S(t+delta), clipped at 0
    fp_loglik log floors                  clip at _P_FLOOR, then max with _LOG_FLOOR

NOT ported, because the method under test does not use them: external hazard, `time_range`, the
heterogeneity mixture, `tail_extrap`, `absorb_flux`, the 'be'-only scheme.

VALIDATION of the v0.1 port against the NumPy oracle (144 evaluations over 24 cohorts x 6 points):
terms with a resolved probability mass agree to ~1e-4 nats (~1e-8 per individual). Larger
differences occur ONLY where a term's mass is below ~1e-11 -- S(t)-S(t+delta) of two survival values
equal to 14 digits, or an underflowed tail -- where BOTH backends return round-off (the two linear
solvers differ by 1e-13..7e-11 in S near 1). Such terms are noise in the NumPy reference too.

v0.3 SPEED CHANGES (2026-09-24), each measured and adversarially re-checked before adoption:

  1. HAND-WRITTEN REVERSE SWEEP OVER THE WHOLE MARCH (`_march_vjp`). JAX's autodiff of the scan stored
     ~23 (nsteps, N) residual arrays per gradient -- 169 MB at 1,000 steps, 0.9 GB at 5,500 (mice_F),
     3 GB (celegans), 10 GB (ecoli) -- and ran ~2.7-3x a value evaluation. The custom VJP stores only
     q_{n+1} = M_n^{-1} r_n per step and runs one transposed tridiagonal solve per step backwards.
     Above 64 MiB of q (nsteps > ~9,300 at N=900) it stores a checkpoint every ceil(sqrt(nsteps))
     steps and re-marches each block (`_march_vjp_ckpt`; +35-40% gradient CPU, memory ~MBs).
     Measured: value bit-identical to v0.2; gradient within production's own round-off; value+grad
     1.8-2.9x less CPU; peak RSS on a 1-core cluster element: mice_F 1,628 -> 708 MB, celegans
     3,884 -> 737 MB, ecoli killed at 1 GB -> 830 MB.
  2. EARLY STOP. The march stops where NumPy's does (S < Gtol after tau > 0.3) instead of marching
     to tau_max and masking: median stop at 76% of nsteps at the truth, 15-27% on the -700 plateau
     points Nelder-Mead wanders over (value 5-8x cheaper there).
  3. ONE TRANSCENDENTAL PER FACE for the Scharfetter-Gummel pair (`_bern_pair`): with a = |P|,
     B(a) = a/expm1(a) and B(-a) = a + B(a) -- exact algebra, no cancellation -- and an analytic
     derivative B'(z) = B(z)(1 - B(-z))/z written to subtract the SMALL member, with a 16-term even
     Bernoulli series for |z| < 1.5. Worst errors vs mpmath (50 digits, 76k points): value 6.7e-16,
     derivative 5.2e-16 (the autodiff derivative of the v0.2 form was off by 5.7e-6 near 0 and 100%
     for |z| > 354). Value 1.22x, value+grad 1.16x faster on top of (1).
  4. PJRT_NPROC=1. The v0.2 `XLA_FLAGS` line was inert: value+grad ran on ~1.4 cores and the tester
     bills process CPU. PJRT_NPROC pins XLA's CPU client to one thread (bit-identical results); it
     must be set before the first JAX computation. A no-op on WEXAC, where LSF binds one core.

Not adopted, measured: persistent compilation cache / AOT / shape-bucketed shared executables
("precompiling") -- compile is ~0.2-1.5% of cohort CPU and the per-eval penalty of bucketing cancels
the saving; lax.scan unroll, vmap batching, a hand-written Thomas solve, forward mode, Newton steps,
in-JIT optimisers and XLA flags were all within +-5% or slower.

BASIN-2 TASK B (scratch patch, NOT a released version): CANCELLATION-FREE INTERVAL MASS.
  With S near 1, S(t) - S(t+delta) of two survival values equal to 14-16 digits is round-off: exactly
  0 (-> the -700 floor) or 5e-14..3e-12 (-> ~ -30) under a 1-ulp change of z, which makes the
  objective discontinuous at the low-rho seed (reports/basin/rootcause). The march now also carries
  the ABSORBED MASS F = 1 - S, accumulated from the absorbing wall's outflow. The scheme's only
  non-zero column sum is 1^T A = -c B(-P_{N-1}) e_{N-1}^T, so with phi = h c B(-P_{N-1}) q_{N-1}:
      BE    F^{n+1} = F^n + dt phi^{n+1}                          + h sum min(q^{n+1}, 0)
      BDF2  F^{n+1} = (2 F^n - F^{n-1}/2 + dt phi^{n+1}) / 1.5    + h sum min(q^{n+1}, 0)
  which is 1 - S^{n+1} EXACTLY in exact arithmetic (the last term is the mass the p >= 0 clip adds),
  but is a sum of small positive terms, so it keeps full RELATIVE precision where S ~ 1. A death's
  interval mass is F(t+delta) - F(t) when F(t+delta) < S(t) (the difference with the smaller
  operands), else S(t) - S(t+delta) as before. The custom VJP carries the adjoint of the F recursion.
  Censored terms (log S) and the exact-time density (sampling_dt=None) are unchanged.
"""
from __future__ import annotations

import math
import os

# One thread per process (see 4. above). setdefault: an explicit user setting wins.
os.environ.setdefault("PJRT_NPROC", "1")

import numpy as np
import jax
import jax.numpy as jnp
from jax.lax.linalg import tridiagonal_solve

jax.config.update("jax_enable_x64", True)   # float64: the -700 floor and 1e-11 Gtol need it

_LOG_FLOOR = -700.0
_P_FLOOR = 9.859676543759771e-305      # == math.exp(_LOG_FLOOR), as in fp_likelihood
_CLIP = 500.0                          # fp_likelihood._bernoulli clips z to [-500, 500]
_SERIES_BELOW = 1.5                    # |z| below this: derivative from the even series
# c_k = B_2k / (2k)!, k = 1..16 (mpmath, 40 digits, rounded once): B(z) = 1 - z/2 + sum c_k z^2k
_BERN_CK = (
    0.08333333333333333, -0.001388888888888889, 3.306878306878307e-05, -8.267195767195768e-07,
    2.08767569878681e-08, -5.284190138687493e-10, 1.3382536530684679e-11, -3.3896802963225827e-13,
    8.586062056277845e-15, -2.174868698558062e-16, 5.5090028283602295e-18, -1.3954464685812522e-19,
    3.534707039629467e-21, -8.953517427037546e-23, 2.267952452337683e-24, -5.744790668872202e-26,
)
# Above this many bytes of stored q the gradient switches to the checkpointed sweep.
Q_BUDGET_BYTES = 64 * 2 ** 20


def _bernoulli(z):
    """B(z) = z/(e^z - 1), exactly `fp_likelihood._bernoulli` (double-where so autodiff is NaN-free)."""
    z = jnp.clip(z, -_CLIP, _CLIP)
    small = jnp.abs(z) < 1e-10
    zs = jnp.where(small, 1.0, z)
    return jnp.where(small, 1.0 - z / 2.0, zs / jnp.expm1(zs))


def _bern_pair_value(P):
    """(B(P), B(-P)) with ONE expm1: B(-a) = a + B(a) for a = |P| >= 0 (no cancellation)."""
    a = jnp.minimum(jnp.abs(P), _CLIP)
    Ba = _bernoulli(a)
    Bn = a + Ba
    pos = P >= 0
    return jnp.where(pos, Ba, Bn), jnp.where(pos, Bn, Ba)


@jax.custom_jvp
def _bern_pair(P):
    return _bern_pair_value(P)


@_bern_pair.defjvp
def _bern_pair_jvp(primals, tangents):
    """d/dz B(z) = B(z)(1 - B(-z))/z and d/dz B(-z) = B(-z)(1 - B(z))/z, each written so that it
    subtracts the SMALL member of the pair; the even Bernoulli series below |z| = 1.5; 0 outside the
    clip (the value is constant there)."""
    (z,), (dz,) = primals, tangents
    bp, bm = _bern_pair_value(z)
    zc = jnp.clip(z, -_CLIP, _CLIP)
    inside = jnp.abs(z) <= _CLIP
    small = jnp.abs(zc) < _SERIES_BELOW
    zs = jnp.where(small, 1.0, zc)
    pos = zc >= 0
    one_m_bm = jnp.where(pos, (1.0 - zc) - bp, 1.0 - bm)     # 1 - B(-z)
    one_m_bp = jnp.where(pos, 1.0 - bp, (1.0 + zc) - bm)     # 1 - B(z)
    dbp_l = bp * one_m_bm / zs
    dbm_l = bm * one_m_bp / zs
    zz = jnp.where(small, zc, 0.0)
    u = zz * zz
    K = len(_BERN_CK)
    acc = jnp.zeros_like(zz) + (2 * K) * _BERN_CK[K - 1]
    for k in range(K - 1, 0, -1):
        acc = acc * u + (2 * k) * _BERN_CK[k - 1]
    dev = acc * zz                                           # sum_k 2k c_k z^(2k-1)
    dbp = jnp.where(small, -0.5 + dev, dbp_l)
    dbm = jnp.where(small, 0.5 + dev, dbm_l)
    dbp = jnp.where(inside, dbp, 0.0)
    dbm = jnp.where(inside, dbm, 0.0)
    return (bp, bm), (dbp * dz, dbm * dz)


def _coeffs(theta, tn, alpha, N, h, dt):
    """(dl, dm, du) of the step matrix alpha*I - dt*A(tn) from theta = (rho_eta, rho_eps, rem[N]).

    `fpt_full`'s `build` + banded layout: lo[i] = c Bm[i-1] (i >= 1), di[i] = -c (Bm[i] + Bp[i-1]) with
    di[0] = -c Bm[0] (reflecting wall), up[i] = c Bp[i] (i <= N-2). In tridiagonal_solve's convention
    dl[i] multiplies x[i-1] in row i and du[i] multiplies x[i+1]; dl[0] and du[-1] are unused.
    """
    rho_eta, rho_eps, rem = theta
    P = (rho_eta * tn - rem) * h / rho_eps
    Bp, Bm = _bern_pair(P)
    c = rho_eps / h ** 2
    z1 = jnp.zeros(1)
    lo = c * jnp.concatenate([z1, Bm[:-1]])
    di = -c * (Bm + jnp.concatenate([z1, Bp[:-1]]))
    up = c * jnp.concatenate([Bp[:-1], z1])
    dl = jnp.concatenate([z1, -dt * lo[1:]])
    du = jnp.concatenate([-dt * up[:-1], z1])
    dm = alpha - dt * di
    return dl, dm, du


def _max0_grad_mask(q):
    """d max(q, 0)/dq as lax.max's JVP defines it (0.5 at a tie), so the custom VJP matches autodiff."""
    return jnp.where(q > 0, 1.0, jnp.where(q == 0, 0.5, 0.0))


def _bflux_coef(theta, tn, N, h):
    """Outflow through the absorbing wall per unit q_{N-1}: h c B(-P_{N-1}) = (rho_eps/h) B(-P_{N-1}).

    The step matrix's column sums are 0 except the last, 1^T A = -c Bm[N-1] e_{N-1}^T (`_coeffs`:
    lo[i] = c Bm[i-1] cancels di's Bm[i] term for i < N-1, up[i-1] = c Bp[i-1] its Bp[i-1] term), so
    d(h sum p)/dt = -h c Bm[N-1] p_{N-1}. Same P, same `_bern_pair` as `_coeffs`, last cell only.
    """
    rho_eta, rho_eps, rem = theta
    P = (rho_eta * tn - rem[-1:]) * h / rho_eps
    _Bp, Bm = _bern_pair(P)
    c = rho_eps / h ** 2
    return (c * h) * Bm[0]


def _F_next(theta, n, q, F, Fm, N, h, dt, nstart):
    """F^{n+1} = 1 - S^{n+1} from the scheme's mass balance (module docstring, BASIN-2 TASK B)."""
    bdf2 = n >= nstart
    alpha = jnp.where(bdf2, 1.5, 1.0)
    a = jnp.where(bdf2, 2.0, 1.0)
    b = jnp.where(bdf2, -0.5, 0.0)
    bc = _bflux_coef(theta, (n + 1) * dt, N, h)
    return (a * F + b * Fm + dt * bc * q[-1]) / alpha + h * jnp.sum(jnp.minimum(q, 0.0))


def _step_q(theta, n, p, pp, N, h, dt, nstart):
    """q_{n+1} = M_n^{-1} r_n: BE for n < nstart, then BDF2 (1.5 I - dt A) p^{n+1} = 2 p^n - 0.5 p^{n-1}."""
    bdf2 = n >= nstart
    alpha = jnp.where(bdf2, 1.5, 1.0)
    dl, dm, du = _coeffs(theta, (n + 1) * dt, alpha, N, h, dt)
    rhs = jnp.where(bdf2, 2.0 * p - 0.5 * pp, p)
    return tridiagonal_solve(dl, dm, du, rhs[:, None])[:, 0]


def _march_forward(theta, N, nsteps, dt, nstart, Gtol, store, K=None):
    """The time march with NumPy's early stop. Returns ((S_raw, F_raw)[0..nsteps], stored, k).

    Entries after the stop index k repeat S[k] (< Gtol), so `_truncate` finds the same first hit and
    every masked quantity equals the fixed-length march's. `store`: None (value only), "all" (every
    q_{n+1}, for `_march_vjp`) or "ckpt" (the carry (p_n, p_{n-1}) every K steps).
    BASIN-2 B: F_raw = the absorbed mass from `_F_next` (repeats F[k] after k like S_raw). The S path
    is the frozen code's, op for op.
    """
    h = 1.0 / N
    p0 = jnp.zeros(N).at[0].set(1.0 / h)
    buf = jnp.full(nsteps + 1, p0.sum() * h)
    fbuf = jnp.zeros(nsteps + 1)
    if store == "all":
        ST = jnp.zeros((nsteps + 1, N))
    elif store == "ckpt":
        ST = jnp.zeros(((nsteps + K - 1) // K, 2, N))
    else:
        ST = jnp.zeros((1, N))

    def body(st):
        n, p, pp, F, Fm, buf, fbuf, ST, done = st
        if store == "ckpt":
            b = n // K
            old = jax.lax.dynamic_slice(ST, (b, 0, 0), (1, 2, N))
            ST = jax.lax.dynamic_update_slice(ST, jnp.where(n % K == 0, jnp.stack([p, pp])[None], old),
                                              (b, 0, 0))
        q = _step_q(theta, n, p, pp, N, h, dt, nstart)
        pn = jnp.maximum(q, 0.0)
        S = pn.sum() * h
        Fn = _F_next(theta, n, q, F, Fm, N, h, dt, nstart)
        buf = jax.lax.dynamic_update_slice(buf, S[None], (n + 1,))
        fbuf = jax.lax.dynamic_update_slice(fbuf, Fn[None], (n + 1,))
        if store == "all":
            ST = jax.lax.dynamic_update_slice(ST, q[None, :], (n + 1, 0))
        done = (S < Gtol) & ((n + 1) * dt > 0.3)
        return n + 1, pn, p, Fn, F, buf, fbuf, ST, done

    z0 = jnp.zeros(())
    n, _, _, _, _, buf, fbuf, ST, _ = jax.lax.while_loop(
        lambda s: (s[0] < nsteps) & (~s[8]), body,
        (0, p0, jnp.zeros(N), z0, z0, buf, fbuf, ST, jnp.array(False)))
    idx = jnp.arange(nsteps + 1)
    return (jnp.where(idx <= n, buf, buf[n]), jnp.where(idx <= n, fbuf, fbuf[n])), ST, n


def _adjoint_step(theta, n, q, gS, gF, lam1, lam2, nu1, nu2, N, h, dt, nstart):
    """One step of the reverse sweep. lam = dL/dr_n where r_n is step n's right-hand side.

    p_{n+1} = max(q_{n+1}, 0) enters S_{n+1} (weight h), r_{n+1} (coefficient a(n+1): 2 under BDF2,
    else 1) and r_{n+2} (coefficient b(n+2): -1/2 under BDF2, else 0). Then lam_n = M_n^{-T} qbar, the
    band cotangent is -lam q^T, pulled back through the (recomputed) coefficient function.

    BASIN-2 B: nu_m = dL/dF^m (total) obeys nu_{n+1} = gF[n+1] + (a/alpha)(n+1) nu_{n+2} +
    (b/alpha)(n+2) nu_{n+3} (4/3 and -1/3 under BDF2, 1 and 0 under BE); nu1 = nu_{n+2}, nu2 = nu_{n+3}.
    F^{n+1} depends on q^{n+1} through dt bc q_{N-1} / alpha_n and h sum min(q, 0), and on theta through
    bc = `_bflux_coef` at t_{n+1}. Returns (lam_n, nu_{n+1}, dL/dtheta contribution of step n).
    """
    a_next = jnp.where(n + 1 >= nstart, 2.0, 1.0)
    b_next2 = jnp.where(n + 2 >= nstart, -0.5, 0.0)
    pbar = h * gS[n + 1] + a_next * lam1 + b_next2 * lam2
    cF1 = jnp.where(n + 1 >= nstart, 4.0 / 3.0, 1.0)
    cF2 = jnp.where(n + 2 >= nstart, -1.0 / 3.0, 0.0)
    nu = gF[n + 1] + cF1 * nu1 + cF2 * nu2
    mask = _max0_grad_mask(q)
    alpha = jnp.where(n >= nstart, 1.5, 1.0)
    (dl, dm, du, bc), pull = jax.vjp(
        lambda th: _coeffs(th, (n + 1) * dt, alpha, N, h, dt) + (_bflux_coef(th, (n + 1) * dt, N, h),),
        theta)
    qbar = pbar * mask + (nu * h) * (1.0 - mask)
    qbar = qbar.at[-1].add(nu * dt * bc / alpha)
    z1 = jnp.zeros(1)
    lam = tridiagonal_solve(jnp.concatenate([z1, du[:-1]]), dm, jnp.concatenate([dl[1:], z1]),
                            qbar[:, None])[:, 0]                           # M^T: sub=du, super=dl
    gdm = -lam * q
    gdl = jnp.concatenate([z1, -lam[1:] * q[:-1]])
    gdu = jnp.concatenate([-lam[:-1] * q[1:], z1])
    gb = nu * dt * q[-1] / alpha
    (g,) = pull((gdl, gdm, gdu, gb))
    return lam, nu, g


def _fold_tail(gS, k, nsteps):
    """Entries after k are copies of S[k]: their cotangent belongs to S[k]."""
    idx = jnp.arange(nsteps + 1)
    return jnp.where(idx < k, gS, 0.0).at[k].add(jnp.sum(jnp.where(idx >= k, gS, 0.0)))


def _make_march(N, nsteps, dt, nstart, Gtol, mode):
    """theta -> (S_raw, F_raw). mode: "value" (not reverse-differentiable), "vjp" or "vjp_ckpt"."""
    h = 1.0 / N
    if mode == "value":
        return lambda theta: _march_forward(theta, N, nsteps, dt, nstart, Gtol, None)[0]
    K = int(max(1, int(np.ceil(np.sqrt(nsteps))))) if mode == "vjp_ckpt" else None

    @jax.custom_vjp
    def march(theta):
        return _march_forward(theta, N, nsteps, dt, nstart, Gtol, None)[0]

    def fwd(theta):
        SF_raw, ST, k = _march_forward(theta, N, nsteps, dt, nstart, Gtol,
                                       "all" if K is None else "ckpt", K)
        return SF_raw, (theta, ST, k)

    def bwd_all(res, gSF):
        theta, Q, k = res
        gS, gF = gSF
        gS = _fold_tail(gS, k, nsteps)
        gF = _fold_tail(gF, k, nsteps)
        zth = jax.tree_util.tree_map(jnp.zeros_like, theta)

        def body(st):
            n, lam1, lam2, nu1, nu2, gth = st
            q = jax.lax.dynamic_slice(Q, (n + 1, 0), (1, N))[0]
            lam, nu, g = _adjoint_step(theta, n, q, gS, gF, lam1, lam2, nu1, nu2, N, h, dt, nstart)
            return n - 1, lam, lam1, nu, nu1, jax.tree_util.tree_map(jnp.add, gth, g)

        zN = jnp.zeros(N)
        z0 = jnp.zeros(())
        _, _, _, _, _, gth = jax.lax.while_loop(lambda s: s[0] >= 0, body,
                                                (k - 1, zN, zN, z0, z0, zth))
        return (gth,)

    def bwd_ckpt(res, gSF):
        theta, CP, k = res
        gS, gF = gSF
        gS = _fold_tail(gS, k, nsteps)
        gF = _fold_tail(gF, k, nsteps)
        zth = jax.tree_util.tree_map(jnp.zeros_like, theta)

        def blk(st):
            b, lam1, lam2, nu1, nu2, gth = st
            n0 = b * K
            n1 = jnp.minimum(n0 + K, k)
            cp = jax.lax.dynamic_slice(CP, (b, 0, 0), (1, 2, N))[0]

            def remarch(s):                        # recompute this block's q from its checkpoint
                n, p, pp, QB = s
                q = _step_q(theta, n, p, pp, N, h, dt, nstart)
                return n + 1, jnp.maximum(q, 0.0), p, jax.lax.dynamic_update_slice(QB, q[None, :],
                                                                                     (n - n0, 0))

            _, _, _, QB = jax.lax.while_loop(lambda s: s[0] < n1, remarch,
                                             (n0, cp[0], cp[1], jnp.zeros((K, N))))

            def adj(s):
                n, lam1, lam2, nu1, nu2, gth = s
                q = jax.lax.dynamic_slice(QB, (n - n0, 0), (1, N))[0]
                lam, nu, g = _adjoint_step(theta, n, q, gS, gF, lam1, lam2, nu1, nu2, N, h, dt, nstart)
                return n - 1, lam, lam1, nu, nu1, jax.tree_util.tree_map(jnp.add, gth, g)

            _, lam1, lam2, nu1, nu2, gth = jax.lax.while_loop(lambda s: s[0] >= n0, adj,
                                                              (n1 - 1, lam1, lam2, nu1, nu2, gth))
            return b - 1, lam1, lam2, nu1, nu2, gth

        zN = jnp.zeros(N)
        z0 = jnp.zeros(())
        _, _, _, _, _, gth = jax.lax.while_loop(lambda s: s[0] >= 0, blk,
                                                ((k - 1) // K, zN, zN, z0, z0, zth))
        return (gth,)

    march.defvjp(fwd, bwd_all if K is None else bwd_ckpt)
    return march


def _truncate(S_raw, tau, dt, Gtol, nsteps):
    """(S, p, k): `fpt_full_density` on the grid `fpt_full` would have returned, at fixed shape."""
    idx = jnp.arange(nsteps + 1)
    hit = (S_raw < Gtol) & (tau > 0.3) & (idx >= 1)
    k = jnp.where(hit.any(), jnp.argmax(hit), nsteps)
    # cummin is causal, so the cummin of the prefix IS the prefix of the cummin
    Sc = jnp.clip(jax.lax.cummin(S_raw), 0.0, 1.0)
    Sk = Sc[k]
    S = jnp.where(idx <= k, Sc, Sk)
    g = -jnp.gradient(Sc, dt)                          # central inside, one-sided at index 0
    g_end = (Sc[jnp.maximum(k - 1, 0)] - Sk) / dt      # np.gradient's one-sided end at k
    p = jnp.where(idx < k, g, jnp.where(idx == k, g_end, 0.0))
    return S, jnp.clip(p, 0.0, None), k


def _truncate_F(F_raw, k, nsteps):
    """F on the same truncated grid as `_truncate`'s S: cummax (= 1 - cummin(1 - F)), clip, hold F[k]."""
    idx = jnp.arange(nsteps + 1)
    Fc = jnp.clip(jax.lax.cummax(F_raw), 0.0, 1.0)
    return jnp.where(idx <= k, Fc, Fc[k])


# ------------------------------------------------------------------------------------------------------
# rc2 I3 -- THE FLOOR EARLY-TIME FIX: a GRADED EARLY-TIME MESH, marched with variable-step BDF2.
#
# Where the floored ceiling binds (dt = 3e-4 while delta/L ~ 1e-4), the uniform march cannot resolve the
# initial layer of the delta-start density: a death recorded in the first sampling bins carries an
# interval mass of ~1e-5..1e-7 that the first few steps decide (rc1: 20 backward-Euler steps cover
# tau < 6e-3 at the floor). Measured (reports/rc2/I3-floor): the ENTIRE floor-vs-reference dll error of
# both Phase A RO1 failures is ONE early death each (tau 5e-5 and 1.3e-3); every other term is converged.
# `early_schedule` replaces the start of the uniform grid by `nstart` BE steps of size dt0, then steps
# growing geometrically by `ratio` up to dt, then the uniform dt to tau_max as before. A step whose size
# ratio to its predecessor is w uses variable-step BDF2,
#     (1+2w)/(1+w) p^{n+1} - dt_n A(tau_{n+1}) p^{n+1} = (1+w) p^n - w^2/(1+w) p^{n-1},
# which is BDF2's 1.5 / 2 / -0.5 at w = 1 (exactly, in floating point) and zero-stable for w < 1 + sqrt 2.
# The absorbed-mass recursion (BASIN-2 B) and the custom VJP take the same per-step coefficients
# (alpha_n, a_n, b_n, dt_n, tau_{n+1}). OFF (`make_loglik(early=None)`) runs the uniform code above
# unchanged: the same compiled programs.
# ------------------------------------------------------------------------------------------------------

def early_schedule(dt, tau_max, nstart, dt0, ratio):
    """The graded time grid: (DT, TAU, ALPHA, A, B), numpy float64.

    DT[n] = tau_{n+1} - tau_n for the steps n = 0..nsteps-1; TAU = (0, tau_1, ..., tau_nsteps), tau_nsteps
    <= tau_max; step n solves ALPHA[n] p^{n+1} - DT[n] A(TAU[n+1]) p^{n+1} = A[n] p^n + B[n] p^{n-1}.
    Steps 0..nstart-1: BE at dt0. Then dt0*ratio^j while below dt; then dt."""
    dt, tau_max, dt0, ratio, nstart = float(dt), float(tau_max), float(dt0), float(ratio), int(nstart)
    if not (0.0 < dt0 <= dt and 1.0 < ratio < 1.0 + math.sqrt(2.0) and nstart >= 1):
        raise ValueError(f"mle_profile_jax2 fp_jax: bad early schedule dt0={dt0!r} ratio={ratio!r} "
                         f"nstart={nstart!r} dt={dt!r}")
    g = [dt0] * nstart
    d = dt0 * ratio
    while d < dt:
        g.append(d)
        d *= ratio
    g = np.asarray(g, float)
    TAUg = np.concatenate([[0.0], np.cumsum(g)])
    Tg = float(TAUg[-1])
    n_u = int((tau_max - Tg) / dt)
    if n_u < 2:
        raise ValueError(f"mle_profile_jax2 fp_jax: the graded start ({Tg!r}) leaves no uniform part "
                         f"before tau_max={tau_max!r}")
    TAU = np.concatenate([TAUg, Tg + dt * np.arange(1, n_u + 1)])
    DT = np.concatenate([g, np.full(n_u, dt)])
    ns = DT.size
    AL, A, B = np.ones(ns), np.ones(ns), np.zeros(ns)
    w = DT[nstart:] / DT[nstart - 1:-1]
    AL[nstart:] = (1.0 + 2.0 * w) / (1.0 + w)
    A[nstart:] = 1.0 + w
    B[nstart:] = -(w * w) / (1.0 + w)
    return DT, TAU, AL, A, B


def _sched_arrays(sched):
    """The schedule as jnp constants (DT, TAU[1:], ALPHA, A, B), each padded with 2 neutral entries so the
    reverse sweep may read index n+2 (it multiplies them by zero adjoints there)."""
    DT, TAU, AL, A, B = sched
    pad = lambda x, v: jnp.asarray(np.concatenate([x, [v, v]]))   # noqa: E731
    return (pad(DT, DT[-1]), pad(TAU[1:], TAU[-1]), pad(AL, 1.0), pad(A, 1.0), pad(B, 0.0))


def _step_q_g(theta, n, p, pp, N, h, sch):
    """`_step_q` on the graded grid: q_{n+1} = M_n^{-1} (A[n] p^n + B[n] p^{n-1})."""
    DT, T1, AL, A, B = sch
    dl, dm, du = _coeffs(theta, T1[n], AL[n], N, h, DT[n])
    rhs = A[n] * p + B[n] * pp
    return tridiagonal_solve(dl, dm, du, rhs[:, None])[:, 0]


def _F_next_g(theta, n, q, F, Fm, N, h, sch):
    """`_F_next` on the graded grid (the same mass balance with the step's own coefficients)."""
    DT, T1, AL, A, B = sch
    bc = _bflux_coef(theta, T1[n], N, h)
    return (A[n] * F + B[n] * Fm + DT[n] * bc * q[-1]) / AL[n] + h * jnp.sum(jnp.minimum(q, 0.0))


def _march_forward_g(theta, N, sch, nsteps, Gtol, store, K=None):
    """`_march_forward` on the graded grid (early stop at S < Gtol once tau_{n+1} > 0.3, as there)."""
    h = 1.0 / N
    T1 = sch[1]
    p0 = jnp.zeros(N).at[0].set(1.0 / h)
    buf = jnp.full(nsteps + 1, p0.sum() * h)
    fbuf = jnp.zeros(nsteps + 1)
    if store == "all":
        ST = jnp.zeros((nsteps + 1, N))
    elif store == "ckpt":
        ST = jnp.zeros(((nsteps + K - 1) // K, 2, N))
    else:
        ST = jnp.zeros((1, N))

    def body(st):
        n, p, pp, F, Fm, buf, fbuf, ST, done = st
        if store == "ckpt":
            b = n // K
            old = jax.lax.dynamic_slice(ST, (b, 0, 0), (1, 2, N))
            ST = jax.lax.dynamic_update_slice(ST, jnp.where(n % K == 0, jnp.stack([p, pp])[None], old),
                                              (b, 0, 0))
        q = _step_q_g(theta, n, p, pp, N, h, sch)
        pn = jnp.maximum(q, 0.0)
        S = pn.sum() * h
        Fn = _F_next_g(theta, n, q, F, Fm, N, h, sch)
        buf = jax.lax.dynamic_update_slice(buf, S[None], (n + 1,))
        fbuf = jax.lax.dynamic_update_slice(fbuf, Fn[None], (n + 1,))
        if store == "all":
            ST = jax.lax.dynamic_update_slice(ST, q[None, :], (n + 1, 0))
        done = (S < Gtol) & (T1[n] > 0.3)
        return n + 1, pn, p, Fn, F, buf, fbuf, ST, done

    z0 = jnp.zeros(())
    n, _, _, _, _, buf, fbuf, ST, _ = jax.lax.while_loop(
        lambda s: (s[0] < nsteps) & (~s[8]), body,
        (0, p0, jnp.zeros(N), z0, z0, buf, fbuf, ST, jnp.array(False)))
    idx = jnp.arange(nsteps + 1)
    return (jnp.where(idx <= n, buf, buf[n]), jnp.where(idx <= n, fbuf, fbuf[n])), ST, n


def _adjoint_step_g(theta, n, q, gS, gF, lam1, lam2, nu1, nu2, N, h, sch):
    """`_adjoint_step` on the graded grid: p_{n+1} enters r_{n+1} with A[n+1] and r_{n+2} with B[n+2];
    F^{n+1} enters F^{n+2} with A[n+1]/ALPHA[n+1] and F^{n+3} with B[n+2]/ALPHA[n+2]."""
    DT, T1, AL, A, B = sch
    pbar = h * gS[n + 1] + A[n + 1] * lam1 + B[n + 2] * lam2
    nu = gF[n + 1] + (A[n + 1] / AL[n + 1]) * nu1 + (B[n + 2] / AL[n + 2]) * nu2
    mask = _max0_grad_mask(q)
    alpha, dt, tn = AL[n], DT[n], T1[n]
    (dl, dm, du, bc), pull = jax.vjp(
        lambda th: _coeffs(th, tn, alpha, N, h, dt) + (_bflux_coef(th, tn, N, h),), theta)
    qbar = pbar * mask + (nu * h) * (1.0 - mask)
    qbar = qbar.at[-1].add(nu * dt * bc / alpha)
    z1 = jnp.zeros(1)
    lam = tridiagonal_solve(jnp.concatenate([z1, du[:-1]]), dm, jnp.concatenate([dl[1:], z1]),
                            qbar[:, None])[:, 0]                           # M^T: sub=du, super=dl
    gdm = -lam * q
    gdl = jnp.concatenate([z1, -lam[1:] * q[:-1]])
    gdu = jnp.concatenate([-lam[:-1] * q[1:], z1])
    gb = nu * dt * q[-1] / alpha
    (g,) = pull((gdl, gdm, gdu, gb))
    return lam, nu, g


def _make_march_g(N, sched, Gtol, mode):
    """`_make_march` on the graded grid `sched` (`early_schedule`'s tuple)."""
    h = 1.0 / N
    nsteps = int(sched[0].size)
    sch = _sched_arrays(sched)
    if mode == "value":
        return lambda theta: _march_forward_g(theta, N, sch, nsteps, Gtol, None)[0]
    K = int(max(1, int(np.ceil(np.sqrt(nsteps))))) if mode == "vjp_ckpt" else None

    @jax.custom_vjp
    def march(theta):
        return _march_forward_g(theta, N, sch, nsteps, Gtol, None)[0]

    def fwd(theta):
        SF_raw, ST, k = _march_forward_g(theta, N, sch, nsteps, Gtol, "all" if K is None else "ckpt", K)
        return SF_raw, (theta, ST, k)

    def bwd_all(res, gSF):
        theta, Q, k = res
        gS, gF = gSF
        gS = _fold_tail(gS, k, nsteps)
        gF = _fold_tail(gF, k, nsteps)
        zth = jax.tree_util.tree_map(jnp.zeros_like, theta)

        def body(st):
            n, lam1, lam2, nu1, nu2, gth = st
            q = jax.lax.dynamic_slice(Q, (n + 1, 0), (1, N))[0]
            lam, nu, g = _adjoint_step_g(theta, n, q, gS, gF, lam1, lam2, nu1, nu2, N, h, sch)
            return n - 1, lam, lam1, nu, nu1, jax.tree_util.tree_map(jnp.add, gth, g)

        zN = jnp.zeros(N)
        z0 = jnp.zeros(())
        _, _, _, _, _, gth = jax.lax.while_loop(lambda s: s[0] >= 0, body,
                                                (k - 1, zN, zN, z0, z0, zth))
        return (gth,)

    def bwd_ckpt(res, gSF):
        theta, CP, k = res
        gS, gF = gSF
        gS = _fold_tail(gS, k, nsteps)
        gF = _fold_tail(gF, k, nsteps)
        zth = jax.tree_util.tree_map(jnp.zeros_like, theta)

        def blk(st):
            b, lam1, lam2, nu1, nu2, gth = st
            n0 = b * K
            n1 = jnp.minimum(n0 + K, k)
            cp = jax.lax.dynamic_slice(CP, (b, 0, 0), (1, 2, N))[0]

            def remarch(s):
                n, p, pp, QB = s
                q = _step_q_g(theta, n, p, pp, N, h, sch)
                return n + 1, jnp.maximum(q, 0.0), p, jax.lax.dynamic_update_slice(QB, q[None, :],
                                                                                     (n - n0, 0))

            _, _, _, QB = jax.lax.while_loop(lambda s: s[0] < n1, remarch,
                                             (n0, cp[0], cp[1], jnp.zeros((K, N))))

            def adj(s):
                n, lam1, lam2, nu1, nu2, gth = s
                q = jax.lax.dynamic_slice(QB, (n - n0, 0), (1, N))[0]
                lam, nu, g = _adjoint_step_g(theta, n, q, gS, gF, lam1, lam2, nu1, nu2, N, h, sch)
                return n - 1, lam, lam1, nu, nu1, jax.tree_util.tree_map(jnp.add, gth, g)

            _, lam1, lam2, nu1, nu2, gth = jax.lax.while_loop(lambda s: s[0] >= n0, adj,
                                                              (n1 - 1, lam1, lam2, nu1, nu2, gth))
            return b - 1, lam1, lam2, nu1, nu2, gth

        zN = jnp.zeros(N)
        z0 = jnp.zeros(())
        _, _, _, _, _, gth = jax.lax.while_loop(lambda s: s[0] >= 0, blk,
                                                ((k - 1) // K, zN, zN, z0, z0, zth))
        return (gth,)

    march.defvjp(fwd, bwd_all if K is None else bwd_ckpt)
    return march


def _truncate_g(S_raw, tau, Gtol, nsteps):
    """`_truncate` on a non-uniform tau: np.gradient with coordinates, the one-sided end over tau[k]-tau[k-1]."""
    idx = jnp.arange(nsteps + 1)
    hit = (S_raw < Gtol) & (tau > 0.3) & (idx >= 1)
    k = jnp.where(hit.any(), jnp.argmax(hit), nsteps)
    Sc = jnp.clip(jax.lax.cummin(S_raw), 0.0, 1.0)
    Sk = Sc[k]
    S = jnp.where(idx <= k, Sc, Sk)
    g = -jnp.gradient(Sc, tau)
    km1 = jnp.maximum(k - 1, 0)
    g_end = (Sc[km1] - Sk) / jnp.where(k > 0, tau[k] - tau[km1], 1.0)
    p = jnp.where(idx < k, g, jnp.where(idx == k, g_end, 0.0))
    return S, jnp.clip(p, 0.0, None), k


def _death_mass(S, F, teval, thi, tau):
    """(mass, S_i, mass_S, mass_F, use_F) of each interval [t, t + delta): the cancellation-free one."""
    S_i = jnp.interp(teval, tau, S, left=1.0, right=S[-1])
    S_hi = jnp.interp(thi, tau, S, left=1.0, right=S[-1])
    F_i = jnp.interp(teval, tau, F, left=0.0, right=F[-1])
    F_hi = jnp.interp(thi, tau, F, left=0.0, right=F[-1])
    m_S = S_i - S_hi
    m_F = F_hi - F_i
    use_F = F_hi < S_i                 # the difference whose operands are smaller has the smaller ulp
    return jnp.clip(jnp.where(use_F, m_F, m_S), 0.0, None), S_i, m_S, m_F, use_F


def _density_F(F_raw, S, p, k, dt, nsteps):
    """rc2 I4: the exact-time density on the (uniform) grid with fix B's idea -- from the ABSORBED MASS F = 1 - S
    where F's difference has the smaller operands. Returns (p_new, use_F).

    `_truncate`'s p is -np.gradient(Sc, dt) (central inside, one-sided at index 0) with the one-sided end
    (Sc[k-1] - Sc[k]) / dt at the stop index k: a difference of two survival values. Where S ~ 1 (early deaths
    of a large, sharp cohort) those are equal to 14-16 digits and p is round-off. Here the SAME stencil is applied
    to Fc = clip(cummax(F_raw), 0, 1), which equals 1 - Sc in exact arithmetic (cummax(F) = 1 - cummin(1 - F)),
    so pF = p in exact arithmetic and only round-off differs; F is a sum of small positive terms (the wall flux),
    so its differences keep full RELATIVE precision where S ~ 1. Per grid node the difference whose LARGER
    operand is smaller is used (`_death_mass`'s rule): F's larger operand is F[min(i+1, k)], S's is
    S[max(i-1, 0)] (S = `_truncate`'s S, = clip(cummin(S_raw)) at every index <= k used here). Where S's is
    kept, p_new is `_truncate`'s p bit for bit. The gradient flows through F_raw's custom-VJP adjoint (the `gF`
    sweep fix B added): no new adjoint code."""
    idx = jnp.arange(nsteps + 1)
    Fc = jnp.clip(jax.lax.cummax(F_raw), 0.0, 1.0)
    km1 = jnp.maximum(k - 1, 0)
    gF = jnp.gradient(Fc, dt)                          # central inside, one-sided at index 0 (= -gradient(1 - F))
    gF_end = (Fc[k] - Fc[km1]) / dt                    # the one-sided end at k, as `_truncate`'s g_end
    pF = jnp.where(idx < k, gF, jnp.where(idx == k, gF_end, 0.0))
    F_big = Fc[jnp.minimum(idx + 1, k)]                # F's larger operand at node i
    S_big = S[jnp.maximum(idx - 1, 0)]                 # S's larger operand at node i
    use_F = (idx <= k) & (F_big < S_big)
    return jnp.where(use_F, jnp.clip(pF, 0.0, None), p), use_F


def z_size(hazard=False):
    """The length of z: the four SR parameters, plus log h when the external hazard is fitted."""
    return 5 if hazard else 4


def hazard_interval_core(m, S_hi, h, dl):
    """The bracket of an interval death's probability under survival S(t) e^{-ht}:

        P(t <= T < t + dl) = S(t) e^{-ht} - S(t+dl) e^{-h(t+dl)} = e^{-ht} [ m + S_hi (1 - e^{-h dl}) ]

    with m = S(t) - S(t+dl), taken cancellation-free from fix B's absorbed mass. Both summands are >= 0 and
    -expm1 is exact to an ulp, so the bracket has no cancellation anywhere (tested against mpmath). The
    e^{-ht} factor is applied in log space by the caller (no underflow at large h t)."""
    return m + S_hi * (-jnp.expm1(-h * dl))


def hazard_exact_core(p_phys, S_i, h):
    """The bracket of an exact-time death's density under S(t) e^{-ht}:  p(t) e^{-ht} -> e^{-ht} [p/L + h S]
    (`p_phys` is already p/L)."""
    return p_phys + h * S_i


def make_loglik(death_times, events, L, N=900, dt=1e-3, tau_max=20.0, Gtol=1e-11,
                sampling_dt=None, nstart=20, q_budget_bytes=None, hazard=False, window=None,
                with_terms=False, exact_from_F=False, early=None):
    """(loglik, value_and_grad), both jitted, over z = log(eta, beta, epsilon, kappa[, h]), xc = 1.

    Same semantics as `fp_loglik_physical(..., scheme="bdf2")`: physical parameters, `L` the
    Kaplan-Meier mean, `sampling_dt` the recording resolution (None = exact times). The value function
    uses the early-exit march; value_and_grad uses the custom-VJP march (checkpointed when the stored q
    would exceed `q_budget_bytes`, default Q_BUDGET_BYTES). The two are separate compiled programs, so
    their values can differ in the last ulp -- never mix them in one comparison.

    jax2 (PHASE_B_DESIGN.txt section 2):
      hazard=True   z has 5 entries; h = exp(z[4]) is a constant external hazard, survival S(t) e^{-ht}, as
                    `fp_loglik(external_hazard=-ln h)`. h enters ONLY the ordinary-autodiff assembly below;
                    the march and its custom VJP are untouched. Per observation (t physical, dl its delta):
                      censored        max(log(clip(S_i, P_FLOOR)) - h t, -700)
                      exact death     max(log(clip(p_i/L + h S_i, P_FLOOR)) - h t, -700)
                      interval death  max(log(clip(m + S_hi (-expm1(-h dl)), P_FLOOR)) - h t, -700)
                    m = fix B's cancellation-free mass; dl may differ per observation (`sampling_dt` an array).
                    B5: a death beyond the march's early stop (interval: t/L >= tau[k]; exact: t/L > tau[k])
                    scores -700, as without the hazard (`hw_parts`).
      window=(lo_eff, hi_eff)
                    left truncation at lo_eff (already snapped by the observation model):
                    ll -= n log(clip(S(lo_eff/L) e^{-h lo_eff}, 1e-12, 1)), as the oracle's `_windowed_ll`.
                    Every age must lie in [lo_eff, hi_eff] (raises ValueError otherwise): the observation model
                    has already censored everything at or beyond hi_eff.
      Both OFF runs rc1's assembly UNCHANGED (the same ops in the same order: the same compiled program).
      with_terms=True also returns a third jitted function z -> dict(term, core, S_i, S_hi, cond, S_lo) of
                    the per-observation terms and the conditioning term (tests only; value march).
    rc2 I4: exact_from_F=True scores exact-time deaths (sampling_dt=None) by `_density_F`; False = dev3.
    The z shape is checked at TRACE time: (5,) with the hazard, (4,) without -- JAX clamps out-of-bounds
    indexing, so a length-4 z would otherwise silently read h = exp(z[3]) = kappa.
    rc2 I3: `early=dict(dt0=, ratio=[, nstart=])` marches the graded early-time mesh (`early_schedule`:
    nstart BE steps of dt0, geometric growth by `ratio` up to dt, then dt; variable-step BDF2; the value,
    the custom VJP and the checkpointed VJP alike). None = the uniform grid, the code path above unchanged.
    """
    hazard = bool(hazard)
    nz = z_size(hazard)
    t = np.asarray(death_times, float)
    e = np.ones(t.size, int) if events is None else np.asarray(events, int)
    L = float(L)
    nsteps = int(tau_max / dt)                 # exactly as fpt_full
    tau = jnp.arange(nsteps + 1) * dt
    sched = None
    if early is not None:                      # rc2 I3: the graded early-time mesh
        unknown = set(early) - {"dt0", "ratio", "nstart"}
        if unknown:
            raise ValueError(f"mle_profile_jax2 fp_jax: unknown early-mesh keys {sorted(unknown)!r}")
        sched = early_schedule(dt, tau_max, early.get("nstart", nstart), early["dt0"], early["ratio"])
        nsteps = int(sched[0].size)
        tau = jnp.asarray(sched[1])
    teval = jnp.asarray(t / L)
    died = jnp.asarray(e == 1)
    if sampling_dt is not None and np.ndim(sampling_dt) == 0 and not (float(sampling_dt) > 0):
        sampling_dt = None                     # `_norm_sampling_dt`: non-positive scalar -> None
    if sampling_dt is not None:
        thi = jnp.asarray((t + np.broadcast_to(np.asarray(sampling_dt, float), t.shape)) / L)
    if exact_from_F and sampling_dt is None and sched is not None:   # rc2 I4 x I3: never met by the method
        raise ValueError("mle_profile_jax2 fp_jax: exact_from_F (`_density_F`) assumes the uniform grid; the graded "
                         "early mesh is only built where a sampling_dt is given")
    budget = Q_BUDGET_BYTES if q_budget_bytes is None else q_budget_bytes
    grad_mode = "vjp_ckpt" if (nsteps + 1) * N * 8 > budget else "vjp"
    h = 1.0 / N

    rc1_path = (not hazard) and window is None
    if not rc1_path:
        tphys = jnp.asarray(t)
        if sampling_dt is not None:
            dl_np = np.array(np.broadcast_to(np.asarray(sampling_dt, float), t.shape), float)
            if not (np.all(np.isfinite(dl_np)) and np.all(dl_np > 0)):
                raise ValueError("mle_profile_jax2 fp_jax: every sampling_dt must be finite and > 0")
            dl = jnp.asarray(dl_np)
        if window is not None:
            lo_eff, hi_eff = float(window[0]), float(window[1])
            if not (np.isfinite(lo_eff) and np.isfinite(hi_eff) and 0.0 <= lo_eff < hi_eff):
                raise ValueError(f"mle_profile_jax2 fp_jax: bad window {window!r}")
            tol = 1e-9 * max(1.0, abs(lo_eff), abs(hi_eff))
            if t.size and (float(t.min()) < lo_eff - tol or float(t.max()) > hi_eff + tol):
                raise ValueError(
                    f"mle_profile_jax2 fp_jax: ages [{t.min()!r}, {t.max()!r}] outside the snapped window "
                    f"[{lo_eff!r}, {hi_eff!r}] -- the observation model must censor at hi_eff first")
            n_risk = float(t.size)
            lo_tau = jnp.asarray(lo_eff / L)

    def hw_parts(z, S, F_raw, p, k):
        """jax2: the hazard / window assembly (never used when both are off).

        B5 (1.0.0.dev3), hazard on: a death BEYOND THE MARCH'S EARLY STOP scores the -700 floor, as it does
        without the hazard. There S is held at S[k] (< Gtol), so rc1's SR mass is exactly 0 (interval:
        t >= tau[k]; exact: t > tau[k], where rc1 masks p) and the term is floored; B2's hazard bracket
        un-floored it to log(S[k] (1 - e^{-h dl})) - h t, whose S[k] jumps by ~e^{-lambda dt} whenever the
        discrete stop index k moves with theta -- a sawtooth in theta that rc1 does not have (L2, B2 post-hoc:
        hazard-on FD error 5.8e-5 vs 5.7e-6 hazard-off at the same SR point). Censored terms keep log S[k] - h t,
        as rc1 keeps log S[k]."""
        hz = jnp.exp(z[4]) if hazard else None
        S_hi = None
        past = jnp.zeros(t.shape, bool)
        if sampling_dt is None:
            S_i = jnp.interp(teval, tau, S, left=1.0, right=S[-1])
            p_i = jnp.interp(teval, tau, (_density_F(F_raw, S, p, k, dt, nsteps)[0] if exact_from_F else p),
                                 left=0.0, right=0.0)      # rc2 I4: False = dev3
            p_phys = jnp.where(teval > tau[k], 0.0, p_i) / L
            if hazard:
                past = died & (teval > tau[k])
                core = jnp.where(past, 0.0, hazard_exact_core(p_phys, S_i, hz))
            else:
                core = p_phys
        else:
            F = _truncate_F(F_raw, k, nsteps)
            m, S_i, _mS, _mF, _uF = _death_mass(S, F, teval, thi, tau)
            if hazard:
                S_hi = jnp.interp(thi, tau, S, left=1.0, right=S[-1])
                past = died & (teval >= tau[k])
                core = jnp.where(past, 0.0, hazard_interval_core(m, S_hi, hz, dl))
            else:
                core = m
        if hazard:
            ht = hz * tphys
            logd = jnp.maximum(jnp.log(jnp.clip(core, _P_FLOOR, None)) - ht, _LOG_FLOOR)
            logc = jnp.maximum(jnp.log(jnp.clip(S_i, _P_FLOOR, None)) - ht, _LOG_FLOOR)
        else:
            logd = jnp.maximum(jnp.log(jnp.clip(core, _P_FLOOR, None)), _LOG_FLOOR)
            logc = jnp.maximum(jnp.log(jnp.clip(S_i, _P_FLOOR, None)), _LOG_FLOOR)
        term = jnp.where(died, logd, logc)
        cond = jnp.zeros(())
        S_lo = None
        if window is not None:
            S_lo = jnp.interp(lo_tau, tau, S, left=1.0, right=S[-1])
            if hazard:
                S_lo = S_lo * jnp.exp(-hz * lo_eff)
            cond = -n_risk * jnp.log(jnp.clip(S_lo, 1e-12, 1.0))
        return dict(term=term, core=jnp.where(died, core, S_i), S_i=S_i,
                    S_hi=(jnp.zeros_like(S_i) if S_hi is None else S_hi), cond=cond,
                    S_lo=(jnp.ones(()) if S_lo is None else S_lo), past=past)

    def build(march, terms=False):
        def ll_of_z(z):
            # jax2: runs at TRACE time (static shape), so it adds no op to the compiled program
            if tuple(jnp.shape(z)) != (nz,):
                raise ValueError(f"mle_profile_jax2 fp_jax: z must have shape ({nz},), got {tuple(jnp.shape(z))}")
            eta, beta, eps, kap = jnp.exp(z[0]), jnp.exp(z[1]), jnp.exp(z[2]), jnp.exp(z[3])
            rb, rh, re = L * beta, L ** 2 * eta, L * eps           # to_dimensionless, xc = 1
            yf = (jnp.arange(N) + 0.5) * h
            rem = rb * yf / (yf + kap)                              # time-invariant removal term
            S_raw, F_raw = march((rh, re, rem))
            if sched is None:
                S, p, k = _truncate(S_raw, tau, dt, Gtol, nsteps)
            else:                                                   # rc2 I3: graded mesh
                S, p, k = _truncate_g(S_raw, tau, Gtol, nsteps)
            if not rc1_path:                                        # jax2: hazard and/or window
                parts = hw_parts(z, S, F_raw, p, k)
                if terms:
                    return parts
                ll = jnp.sum(parts["term"]) + parts["cond"]
                return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)
            # ---- rc1's assembly, unchanged ----
            if sampling_dt is None:
                S_i = jnp.interp(teval, tau, S, left=1.0, right=S[-1])
                p_i = jnp.interp(teval, tau, (_density_F(F_raw, S, p, k, dt, nsteps)[0] if exact_from_F else p),
                                 left=0.0, right=0.0)      # rc2 I4: False = dev3
                p_phys = jnp.where(teval > tau[k], 0.0, p_i) / L
            else:
                F = _truncate_F(F_raw, k, nsteps)                  # BASIN-2 B
                p_phys, S_i, _mS, _mF, _uF = _death_mass(S, F, teval, thi, tau)
            logp = jnp.maximum(jnp.log(jnp.clip(p_phys, _P_FLOOR, None)), _LOG_FLOOR)
            logS = jnp.maximum(jnp.log(jnp.clip(S_i, _P_FLOOR, None)), _LOG_FLOOR)
            if terms:
                term = jnp.where(died, logp, logS)
                return dict(term=term, core=jnp.where(died, p_phys, S_i), S_i=S_i, S_hi=jnp.zeros_like(S_i),
                            cond=jnp.zeros(()), S_lo=jnp.ones(()))
            ll = jnp.sum(jnp.where(died, logp, logS))
            return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)
        return ll_of_z

    def mk(mode):
        if sched is None:
            return _make_march(N, nsteps, dt, nstart, Gtol, mode)
        return _make_march_g(N, sched, Gtol, mode)                 # rc2 I3: graded mesh

    value = build(mk("value"))
    grad = build(mk(grad_mode))
    out = (jax.jit(value), jax.jit(jax.value_and_grad(grad)))
    if with_terms:
        out = out + (jax.jit(build(mk("value"), terms=True)),)
    return out
