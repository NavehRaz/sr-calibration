"""The engine's single Fokker-Planck stepper (research R1, R3-R5; spec FR-016, FR-021).

One march serves every path, so the duplicated stepper of SRK's fp_piecewise ([T-MERGEBACK]) does not
come back. It is v0.3.3's march (`sraging.compat.v033.fp_jax`, itself a twin of SRK's `fpt_full`
with scheme="bdf2"), written as a generalisation: every later feature is switched by the static
`MarchSpec`, and a feature that is off contributes NO traced operation. With every feature off the
march, its custom VJP and the checkpointed sweep are v0.3.3's statement for statement, so the engine's
programs lower to v0.3.3's (tests/parity/test_engine_t0.py compares the canonical program text and the
bytes).

The discretisation (SR_algorithms.pdf §2.2): N cells, h = 1/N, Scharfetter-Gummel face fluxes with the
drift at the new time level, a reflecting wall at node 0 and an absorbing node N; backward Euler for the
first `nstart` steps, then BDF2; p < 0 clipped after each solve; S = h * sum(p); the march stops at
S < Gtol once tau > 0.3. Scheme "be" is backward Euler throughout (the BDF2 switch is never reached).

Time is in units of L; `theta` = (rho_eta, rho_eps, rem) with rem_i = rho_beta y_i/(y_i + kappa) at the
faces, as v0.3.3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sraging.compat.v033 import runtime as _runtime
_runtime.configure()                    # x64 + one XLA thread, before the first JAX computation (spec 001 R3)

import jax                              # noqa: E402
import jax.numpy as jnp                 # noqa: E402
from jax.lax.linalg import tridiagonal_solve   # noqa: E402

jax.config.update("jax_enable_x64", True)

_LOG_FLOOR = -700.0
_P_FLOOR = 9.859676543759771e-305      # == math.exp(_LOG_FLOOR), as SRK's fp_likelihood
_CLIP = 500.0                          # SRK's _bernoulli clips z to [-500, 500]
_SERIES_BELOW = 1.5                    # |z| below this: the derivative from the even series
# c_k = B_2k / (2k)!, k = 1..16 (mpmath, 40 digits, rounded once): B(z) = 1 - z/2 + sum c_k z^2k
_BERN_CK = (
    0.08333333333333333, -0.001388888888888889, 3.306878306878307e-05, -8.267195767195768e-07,
    2.08767569878681e-08, -5.284190138687493e-10, 1.3382536530684679e-11, -3.3896802963225827e-13,
    8.586062056277845e-15, -2.174868698558062e-16, 5.5090028283602295e-18, -1.3954464685812522e-19,
    3.534707039629467e-21, -8.953517427037546e-23, 2.267952452337683e-24, -5.744790668872202e-26,
)
# Above this many bytes of stored q the gradient switches to the checkpointed sweep (as v0.3.3).
Q_BUDGET_BYTES = 64 * 2 ** 20


@dataclass(frozen=True)
class MarchSpec:
    """Everything static about one uniform march (v0.3.3's, plus the F carry of numerics "v1"; off by
    default, so the default spec is v0.3.3's march). Graded meshes and segments use the general march,
    `stepper_gen.GenSpec`."""

    N: int
    nsteps: int
    dt: float
    nstart: int
    Gtol: float
    carry_F: bool = False          # numerics "v1": also march the absorbed mass F (register D-004, D-106)


def spec_for(grid, carry_F: bool = False) -> MarchSpec:
    """The march of a uniform `Grid`. Scheme "be" never reaches the BDF2 switch."""
    if not grid.uniform:
        raise NotImplementedError("graded meshes arrive with segments (User Story 4)")
    nstart = grid.nstart if grid.scheme == "bdf2" else grid.nsteps + 2
    return MarchSpec(N=grid.N, nsteps=grid.nsteps, dt=grid.dt, nstart=nstart, Gtol=grid.Gtol,
                     carry_F=bool(carry_F))


def grad_mode(spec: MarchSpec, budget: int | None = None) -> str:
    budget = Q_BUDGET_BYTES if budget is None else budget
    return "vjp_ckpt" if (spec.nsteps + 1) * spec.N * 8 > budget else "vjp"


def _bernoulli(z):
    """B(z) = z/(e^z - 1), exactly SRK's `_bernoulli` (double-where so autodiff is NaN-free)."""
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

    `fpt_full`'s `build` in banded layout: lo[i] = c Bm[i-1] (i >= 1), di[i] = -c (Bm[i] + Bp[i-1]) with
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

    The step matrix's column sums are 0 except the last (SR_algorithms.pdf eq. s2-stencil), so
    d(h sum p)/dt = -h c Bm[N-1] p_{N-1}: the scheme's own discrete flux, same P and Bernoulli pair as
    `_coeffs`, last cell only ([T-JAX-BASIN] fix B; jax2 fp_jax.py:207-218, read-only reference)."""
    rho_eta, rho_eps, rem = theta
    P = (rho_eta * tn - rem[-1:]) * h / rho_eps
    _Bp, Bm = _bern_pair(P)
    c = rho_eps / h ** 2
    return (c * h) * Bm[0]


def _F_next(theta, n, q, F, Fm, N, h, dt, nstart):
    """F^{n+1} = 1 - S^{n+1} from the scheme's discrete mass balance plus the mass the p >= 0 clip adds:
    F^{n+1} = (a F^n + b F^{n-1} + dt bc q_{N-1}) / alpha + h sum min(q, 0) (research R3). A sum of small
    positive terms, so it keeps full relative precision where S ~ 1 (register D-004)."""
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


def _march_forward(theta, N, nsteps, dt, nstart, Gtol, store, K=None, carry_F=False):
    """The time march with SRK's early stop. Returns (S_raw[0..nsteps], stored, k), or with `carry_F`
    ((S_raw, F_raw), stored, k).

    Entries after the stop index k repeat S[k] (< Gtol), so `truncate` finds the same first hit and
    every masked quantity equals the fixed-length march's. `store`: None (value only), "all" (every
    q_{n+1}, for the custom VJP) or "ckpt" (the carry (p_n, p_{n-1}) every K steps). With `carry_F` the
    absorbed mass F rides along (`_F_next`); the S path is unchanged operation for operation, and without
    it nothing about F is traced.
    """
    h = 1.0 / N
    p0 = jnp.zeros(N).at[0].set(1.0 / h)
    buf = jnp.full(nsteps + 1, p0.sum() * h)
    if store == "all":
        ST = jnp.zeros((nsteps + 1, N))
    elif store == "ckpt":
        ST = jnp.zeros(((nsteps + K - 1) // K, 2, N))
    else:
        ST = jnp.zeros((1, N))

    def body(st):
        if carry_F:
            n, p, pp, F, Fm, buf, fbuf, ST, done = st
        else:
            n, p, pp, buf, ST, done = st
        if store == "ckpt":
            b = n // K
            old = jax.lax.dynamic_slice(ST, (b, 0, 0), (1, 2, N))
            ST = jax.lax.dynamic_update_slice(ST, jnp.where(n % K == 0, jnp.stack([p, pp])[None], old),
                                              (b, 0, 0))
        q = _step_q(theta, n, p, pp, N, h, dt, nstart)
        pn = jnp.maximum(q, 0.0)
        S = pn.sum() * h
        if carry_F:
            Fn = _F_next(theta, n, q, F, Fm, N, h, dt, nstart)
        buf = jax.lax.dynamic_update_slice(buf, S[None], (n + 1,))
        if carry_F:
            fbuf = jax.lax.dynamic_update_slice(fbuf, Fn[None], (n + 1,))
        if store == "all":
            ST = jax.lax.dynamic_update_slice(ST, q[None, :], (n + 1, 0))
        done = (S < Gtol) & ((n + 1) * dt > 0.3)
        if carry_F:
            return n + 1, pn, p, Fn, F, buf, fbuf, ST, done
        return n + 1, pn, p, buf, ST, done

    if carry_F:
        z0 = jnp.zeros(())
        n, _, _, _, _, buf, fbuf, ST, _ = jax.lax.while_loop(
            lambda s: (s[0] < nsteps) & (~s[8]), body,
            (0, p0, jnp.zeros(N), z0, z0, buf, jnp.zeros(nsteps + 1), ST, jnp.array(False)))
        idx = jnp.arange(nsteps + 1)
        return (jnp.where(idx <= n, buf, buf[n]), jnp.where(idx <= n, fbuf, fbuf[n])), ST, n
    n, _, _, buf, ST, _ = jax.lax.while_loop(lambda s: (s[0] < nsteps) & (~s[5]), body,
                                             (0, p0, jnp.zeros(N), buf, ST, jnp.array(False)))
    idx = jnp.arange(nsteps + 1)
    return jnp.where(idx <= n, buf, buf[n]), ST, n


def _adjoint_step(theta, n, q, gS, lam1, lam2, N, h, dt, nstart):
    """One step of the reverse sweep. lam = dL/dr_n where r_n is step n's right-hand side.

    p_{n+1} = max(q_{n+1}, 0) enters S_{n+1} (weight h), r_{n+1} (coefficient a(n+1): 2 under BDF2,
    else 1) and r_{n+2} (coefficient b(n+2): -1/2 under BDF2, else 0). Then lam_n = M_n^{-T} qbar, the
    band cotangent is -lam q^T, pulled back through the (recomputed) coefficient function.
    """
    a_next = jnp.where(n + 1 >= nstart, 2.0, 1.0)
    b_next2 = jnp.where(n + 2 >= nstart, -0.5, 0.0)
    pbar = h * gS[n + 1] + a_next * lam1 + b_next2 * lam2
    qbar = pbar * _max0_grad_mask(q)
    alpha = jnp.where(n >= nstart, 1.5, 1.0)
    (dl, dm, du), pull = jax.vjp(lambda th: _coeffs(th, (n + 1) * dt, alpha, N, h, dt), theta)
    z1 = jnp.zeros(1)
    lam = tridiagonal_solve(jnp.concatenate([z1, du[:-1]]), dm, jnp.concatenate([dl[1:], z1]),
                            qbar[:, None])[:, 0]                           # M^T: sub=du, super=dl
    gdm = -lam * q
    gdl = jnp.concatenate([z1, -lam[1:] * q[:-1]])
    gdu = jnp.concatenate([-lam[:-1] * q[1:], z1])
    (g,) = pull((gdl, gdm, gdu))
    return lam, g


def _adjoint_step_F(theta, n, q, gS, gF, lam1, lam2, nu1, nu2, N, h, dt, nstart):
    """`_adjoint_step` with the second adjoint nu_m = dL/dF^m (research R3; jax2 fp_jax.py:286-319,
    read-only reference): nu_{n+1} = gF[n+1] + (a/alpha)(n+1) nu_{n+2} + (b/alpha)(n+2) nu_{n+3}. F^{n+1}
    depends on q^{n+1} through dt bc q_{N-1}/alpha and h sum min(q, 0), and on theta through bc."""
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
                            qbar[:, None])[:, 0]
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


def make_march(spec: MarchSpec, mode: str):
    """theta -> S_raw (or (S_raw, F_raw) with spec.carry_F). mode: "value" (not reverse-differentiable),
    "vjp" or "vjp_ckpt"."""
    N, nsteps, dt, nstart, Gtol = spec.N, spec.nsteps, spec.dt, spec.nstart, spec.Gtol
    if spec.carry_F:
        return _make_march_F(N, nsteps, dt, nstart, Gtol, mode)
    h = 1.0 / N
    if mode == "value":
        return lambda theta: _march_forward(theta, N, nsteps, dt, nstart, Gtol, None)[0]
    K = int(max(1, int(np.ceil(np.sqrt(nsteps))))) if mode == "vjp_ckpt" else None

    @jax.custom_vjp
    def march(theta):
        return _march_forward(theta, N, nsteps, dt, nstart, Gtol, None)[0]

    def fwd(theta):
        S_raw, ST, k = _march_forward(theta, N, nsteps, dt, nstart, Gtol,
                                      "all" if K is None else "ckpt", K)
        return S_raw, (theta, ST, k)

    def bwd_all(res, gS):
        theta, Q, k = res
        gS = _fold_tail(gS, k, nsteps)
        zth = jax.tree_util.tree_map(jnp.zeros_like, theta)

        def body(st):
            n, lam1, lam2, gth = st
            q = jax.lax.dynamic_slice(Q, (n + 1, 0), (1, N))[0]
            lam, g = _adjoint_step(theta, n, q, gS, lam1, lam2, N, h, dt, nstart)
            return n - 1, lam, lam1, jax.tree_util.tree_map(jnp.add, gth, g)

        zN = jnp.zeros(N)
        _, _, _, gth = jax.lax.while_loop(lambda s: s[0] >= 0, body, (k - 1, zN, zN, zth))
        return (gth,)

    def bwd_ckpt(res, gS):
        theta, CP, k = res
        gS = _fold_tail(gS, k, nsteps)
        zth = jax.tree_util.tree_map(jnp.zeros_like, theta)

        def blk(st):
            b, lam1, lam2, gth = st
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
                n, lam1, lam2, gth = s
                q = jax.lax.dynamic_slice(QB, (n - n0, 0), (1, N))[0]
                lam, g = _adjoint_step(theta, n, q, gS, lam1, lam2, N, h, dt, nstart)
                return n - 1, lam, lam1, jax.tree_util.tree_map(jnp.add, gth, g)

            _, lam1, lam2, gth = jax.lax.while_loop(lambda s: s[0] >= n0, adj, (n1 - 1, lam1, lam2, gth))
            return b - 1, lam1, lam2, gth

        zN = jnp.zeros(N)
        _, _, _, gth = jax.lax.while_loop(lambda s: s[0] >= 0, blk, ((k - 1) // K, zN, zN, zth))
        return (gth,)

    march.defvjp(fwd, bwd_all if K is None else bwd_ckpt)
    return march


def _make_march_F(N, nsteps, dt, nstart, Gtol, mode):
    """theta -> (S_raw, F_raw) with the second adjoint (numerics "v1"). Same structure as `make_march`;
    the checkpoint stores only (p_n, p_{n-1}) -- F is not needed backwards, only its adjoint nu."""
    h = 1.0 / N
    if mode == "value":
        return lambda theta: _march_forward(theta, N, nsteps, dt, nstart, Gtol, None, carry_F=True)[0]
    K = int(max(1, int(np.ceil(np.sqrt(nsteps))))) if mode == "vjp_ckpt" else None

    @jax.custom_vjp
    def march(theta):
        return _march_forward(theta, N, nsteps, dt, nstart, Gtol, None, carry_F=True)[0]

    def fwd(theta):
        SF_raw, ST, k = _march_forward(theta, N, nsteps, dt, nstart, Gtol,
                                       "all" if K is None else "ckpt", K, carry_F=True)
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
            lam, nu, g = _adjoint_step_F(theta, n, q, gS, gF, lam1, lam2, nu1, nu2, N, h, dt, nstart)
            return n - 1, lam, lam1, nu, nu1, jax.tree_util.tree_map(jnp.add, gth, g)

        zN, z0 = jnp.zeros(N), jnp.zeros(())
        _, _, _, _, _, gth = jax.lax.while_loop(lambda s: s[0] >= 0, body, (k - 1, zN, zN, z0, z0, zth))
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
                q = _step_q(theta, n, p, pp, N, h, dt, nstart)
                return n + 1, jnp.maximum(q, 0.0), p, jax.lax.dynamic_update_slice(QB, q[None, :],
                                                                                     (n - n0, 0))

            _, _, _, QB = jax.lax.while_loop(lambda s: s[0] < n1, remarch,
                                             (n0, cp[0], cp[1], jnp.zeros((K, N))))

            def adj(s):
                n, lam1, lam2, nu1, nu2, gth = s
                q = jax.lax.dynamic_slice(QB, (n - n0, 0), (1, N))[0]
                lam, nu, g = _adjoint_step_F(theta, n, q, gS, gF, lam1, lam2, nu1, nu2, N, h, dt, nstart)
                return n - 1, lam, lam1, nu, nu1, jax.tree_util.tree_map(jnp.add, gth, g)

            _, lam1, lam2, nu1, nu2, gth = jax.lax.while_loop(lambda s: s[0] >= n0, adj,
                                                              (n1 - 1, lam1, lam2, nu1, nu2, gth))
            return b - 1, lam1, lam2, nu1, nu2, gth

        zN, z0 = jnp.zeros(N), jnp.zeros(())
        _, _, _, _, _, gth = jax.lax.while_loop(lambda s: s[0] >= 0, blk,
                                                ((k - 1) // K, zN, zN, z0, z0, zth))
        return (gth,)

    march.defvjp(fwd, bwd_all if K is None else bwd_ckpt)
    return march


def truncate(S_raw, tau, dt, Gtol, nsteps):
    """(S, p, k): SRK's `fpt_full_density` on the grid `fpt_full` would have returned, at fixed shape."""
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


def truncate_F(F_raw, k, nsteps):
    """F on `truncate`'s grid: cummax (= 1 - cummin(1 - F)), clip to [0, 1], held at F[k] after k."""
    idx = jnp.arange(nsteps + 1)
    Fc = jnp.clip(jax.lax.cummax(F_raw), 0.0, 1.0)
    return jnp.where(idx <= k, Fc, Fc[k])


def density_F(F_raw, S, p, k, dt, nsteps):
    """The exact-time density with fix B's idea (register D-106; jax2 rc2 I4, fp_jax.py:669-692, read-only
    reference): the same stencil as `truncate`'s p applied to F, used at a node where F's larger operand is
    smaller than S's. Where S's is kept, the result is `truncate`'s p bit for bit. Returns (p, use_F)."""
    idx = jnp.arange(nsteps + 1)
    Fc = jnp.clip(jax.lax.cummax(F_raw), 0.0, 1.0)
    km1 = jnp.maximum(k - 1, 0)
    gF = jnp.gradient(Fc, dt)
    gF_end = (Fc[k] - Fc[km1]) / dt
    pF = jnp.where(idx < k, gF, jnp.where(idx == k, gF_end, 0.0))
    F_big = Fc[jnp.minimum(idx + 1, k)]
    S_big = S[jnp.maximum(idx - 1, 0)]
    use_F = (idx <= k) & (F_big < S_big)
    return jnp.where(use_F, jnp.clip(pF, 0.0, None), p), use_F
