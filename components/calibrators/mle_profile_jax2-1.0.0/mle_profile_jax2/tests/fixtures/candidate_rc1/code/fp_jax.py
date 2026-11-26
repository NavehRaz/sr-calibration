"""The Fokker-Planck log-likelihood in JAX, a twin of `SRK.fp_likelihood.fp_loglik(scheme="bdf2")`.

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


def make_loglik(death_times, events, L, N=900, dt=1e-3, tau_max=20.0, Gtol=1e-11,
                sampling_dt=None, nstart=20, q_budget_bytes=None):
    """(loglik, value_and_grad), both jitted, over z = log(eta, beta, epsilon, kappa), xc = 1.

    Same semantics as `fp_loglik_physical(..., scheme="bdf2")`: physical parameters, `L` the
    Kaplan-Meier mean, `sampling_dt` the recording resolution (None = exact times). The value function
    uses the early-exit march; value_and_grad uses the custom-VJP march (checkpointed when the stored q
    would exceed `q_budget_bytes`, default Q_BUDGET_BYTES). The two are separate compiled programs, so
    their values can differ in the last ulp -- never mix them in one comparison.
    """
    t = np.asarray(death_times, float)
    e = np.ones(t.size, int) if events is None else np.asarray(events, int)
    L = float(L)
    nsteps = int(tau_max / dt)                 # exactly as fpt_full
    tau = jnp.arange(nsteps + 1) * dt
    teval = jnp.asarray(t / L)
    died = jnp.asarray(e == 1)
    if sampling_dt is not None and np.ndim(sampling_dt) == 0 and not (float(sampling_dt) > 0):
        sampling_dt = None                     # `_norm_sampling_dt`: non-positive scalar -> None
    if sampling_dt is not None:
        thi = jnp.asarray((t + np.broadcast_to(np.asarray(sampling_dt, float), t.shape)) / L)
    budget = Q_BUDGET_BYTES if q_budget_bytes is None else q_budget_bytes
    grad_mode = "vjp_ckpt" if (nsteps + 1) * N * 8 > budget else "vjp"
    h = 1.0 / N

    def build(march):
        def ll_of_z(z):
            eta, beta, eps, kap = jnp.exp(z[0]), jnp.exp(z[1]), jnp.exp(z[2]), jnp.exp(z[3])
            rb, rh, re = L * beta, L ** 2 * eta, L * eps           # to_dimensionless, xc = 1
            yf = (jnp.arange(N) + 0.5) * h
            rem = rb * yf / (yf + kap)                              # time-invariant removal term
            S_raw, F_raw = march((rh, re, rem))
            S, p, k = _truncate(S_raw, tau, dt, Gtol, nsteps)
            if sampling_dt is None:
                S_i = jnp.interp(teval, tau, S, left=1.0, right=S[-1])
                p_i = jnp.interp(teval, tau, p, left=0.0, right=0.0)
                p_phys = jnp.where(teval > tau[k], 0.0, p_i) / L
            else:
                F = _truncate_F(F_raw, k, nsteps)                  # BASIN-2 B
                p_phys, S_i, _mS, _mF, _uF = _death_mass(S, F, teval, thi, tau)
            logp = jnp.maximum(jnp.log(jnp.clip(p_phys, _P_FLOOR, None)), _LOG_FLOOR)
            logS = jnp.maximum(jnp.log(jnp.clip(S_i, _P_FLOOR, None)), _LOG_FLOOR)
            ll = jnp.sum(jnp.where(died, logp, logS))
            return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)
        return ll_of_z

    value = build(_make_march(N, nsteps, dt, nstart, Gtol, "value"))
    grad = build(_make_march(N, nsteps, dt, nstart, Gtol, grad_mode))
    return jax.jit(value), jax.jit(jax.value_and_grad(grad))
