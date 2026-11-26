"""The general march: any time mesh, piecewise-constant parameters, the DPU accumulation rate and a moving absorbing
boundary (research R4, R5; spec FR-014 to FR-016; User Story 4).

Same discretisation as `stepper` (Scharfetter-Gummel, BE start then BDF2, clip, early stop; SR_algorithms.pdf
§2.2 and §6.2), generalised:
- the mesh is any increasing node list; BDF2 uses the variable-step coefficients (w = dt_n / dt_{n-1}):
  alpha = (1+2w)/(1+w), a = 1+w, b = -w^2/(1+w); BE for the first nstart steps and at every restart;
- the time axis is cut into SPANS at the segments' edges, which are mesh nodes; step n uses the span that
  contains (tau_n, tau_{n+1}) (SRK fp_piecewise's "ties go to the earlier segment");
- the production term rho_eta * Y uses the accumulated DPUs (damage production units) Y(tau) = Y_k + A_k (tau - tau_k),
  A the DPU accumulation rate (fp_piecewise and the simulator); removal, noise and the step are not rescaled; with no
  A segment Y = tau;
- a span with a lower threshold uses only M_k = round(N xc_k / x_ref) cells: at its first step the mass
  above the new threshold is absorbed at once (added to F) and the BDF2 history is discarded (one BE step);
  a raised threshold exposes empty cells. Below two cells is refused upstream (register D-114).

The value program early-stops in a while_loop. The gradient program marches the fixed length in a scan
with checkpointed blocks and is differentiated by JAX: past the stop index the survival is replaced by its
held value in `truncate_gen`, so the two compute the same function, and the gradient is exact by
construction (an amendment to the seed's "hand-written sweep everywhere"; plan research, amendments).
The uniform v0.3.3 path (`stepper`) is unchanged; tests show that one span on a uniform mesh here equals it
to round-off.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import jax
import jax.numpy as jnp
from jax.lax.linalg import tridiagonal_solve

from sraging.engine.stepper import _bern_pair


@dataclass(frozen=True)
class GenSpec:
    """Everything static about one general march (built on the host by `build_spec`)."""

    N: int
    tau: tuple                 # nodes, tau[0] = 0, len nsteps + 1
    span: tuple                # per step: span index
    M: tuple                   # per span: active cells
    lower: tuple               # per step: True where a lower threshold starts (absorb above it)
    AL: tuple
    A: tuple
    B: tuple                   # per step: variable-step BDF2 / BE coefficients
    span_start: tuple          # per span: start time (dimensionless)
    clock: bool                # any DPU accumulation rate (A) segment
    Gtol: float
    carry_F: bool

    @property
    def nsteps(self) -> int:
        return len(self.tau) - 1


def coefficients(tau, nstart, restart):
    """(AL, A, B) per step: BE for n < nstart and at restart steps, else variable-step BDF2."""
    dt = np.diff(np.asarray(tau, float))
    AL, A, B = np.ones(dt.size), np.ones(dt.size), np.zeros(dt.size)
    for n in range(dt.size):
        if n < nstart or restart[n] or n == 0:
            continue
        w = dt[n] / dt[n - 1]
        AL[n], A[n], B[n] = (1 + 2 * w) / (1 + w), 1 + w, -w * w / (1 + w)
    return AL, A, B


def build_spec(*, N, tau, span_bounds, M_by_span, clock, Gtol, nstart, carry_F, extra_restart=()):
    """`span_bounds`: dimensionless span starts (first 0) -- every finite one must be a node of `tau`."""
    tau = np.asarray(tau, float)
    starts = np.asarray(span_bounds, float)
    mid = 0.5 * (tau[:-1] + tau[1:])
    span = np.searchsorted(starts, mid, side="right") - 1
    M = np.asarray(M_by_span, int)
    lower = np.zeros(span.size, bool)
    restart = np.zeros(span.size, bool)
    for n in range(1, span.size):
        if span[n] != span[n - 1] and M[span[n]] != M[span[n - 1]]:
            restart[n] = True                         # history discarded at a threshold move (fp_piecewise)
            lower[n] = M[span[n]] < M[span[n - 1]]
    for n in extra_restart:
        restart[n] = True
    AL, A, B = coefficients(tau, nstart, restart)
    return GenSpec(N=int(N), tau=tuple(tau.tolist()), span=tuple(span.tolist()), M=tuple(M.tolist()),
                   lower=tuple(lower.tolist()), AL=tuple(AL.tolist()), A=tuple(A.tolist()), B=tuple(B.tolist()),
                   span_start=tuple(starts.tolist()), clock=bool(clock), Gtol=float(Gtol), carry_F=bool(carry_F))


class _Const:
    """The spec's arrays as JAX constants (made once per build)."""

    def __init__(self, s: GenSpec):
        self.tau = jnp.asarray(s.tau)
        self.dt = jnp.asarray(np.diff(np.asarray(s.tau)))
        self.span = jnp.asarray(s.span)
        self.M = jnp.asarray(s.M)
        self.lower = jnp.asarray(s.lower)
        self.AL, self.A, self.B = jnp.asarray(s.AL), jnp.asarray(s.A), jnp.asarray(s.B)
        self.span_start = jnp.asarray(s.span_start)
        self.idx = jnp.arange(s.N)


def _clock_starts(theta, c, s):
    """Y at each span start: Y_0 = tau_0, Y_{k+1} = Y_k + A_k (t_{k+1} - t_k)."""
    st = c.span_start
    inc = theta["alpha"][:-1] * (st[1:] - st[:-1])
    return jnp.concatenate([st[:1], st[:1] + jnp.cumsum(inc)])


def _step(theta, c, s, Ystart, n, p, pp, F, Fm):
    """One step: returns (p_next, F_next, F_hist) with the lowering applied to the incoming state. F_hist is
    this step's F after the lowering, the next step's BDF2 history: the absorbed mass is a jump of F at the
    step's start, so the history carries it too (review T064: without it BDF2 counted the jump 4/3 times)."""
    N, h = s.N, 1.0 / s.N
    k = c.span[n]
    Mk = c.M[k]
    act = c.idx < Mk
    if any(s.lower):
        low = c.lower[n]
        removed = h * jnp.sum(jnp.where(act, 0.0, p))
        p = jnp.where(low, jnp.where(act, p, 0.0), p)
        if s.carry_F:
            F = jnp.where(low, F + removed, F)
            Fm = jnp.where(low, Fm + removed, Fm)      # the jump is in the history too (the step is BE anyway)
    t1 = c.tau[n + 1]
    Y = Ystart[k] + theta["alpha"][k] * (t1 - c.span_start[k]) if s.clock else t1
    rho_eta, rho_eps, rem = theta["rho_eta"][k], theta["rho_eps"][k], theta["rem"][k]
    P = (rho_eta * Y - rem) * h / rho_eps
    Bp, Bm = _bern_pair(P)
    cc = rho_eps / h ** 2
    z1 = jnp.zeros(1)
    lo = cc * jnp.concatenate([z1, Bm[:-1]])
    di = -cc * (Bm + jnp.concatenate([z1, Bp[:-1]]))
    up = cc * jnp.concatenate([Bp[:-1], z1])
    dtn, al = c.dt[n], c.AL[n]
    dl = jnp.where(act, jnp.concatenate([z1, -dtn * lo[1:]]), 0.0)
    du = jnp.where(c.idx < Mk - 1, jnp.concatenate([-dtn * up[:-1], z1]), 0.0)
    dm = jnp.where(act, al - dtn * di, 1.0)
    rhs = jnp.where(act, c.A[n] * p + c.B[n] * pp, 0.0)
    q = tridiagonal_solve(dl, dm, du, rhs[:, None])[:, 0]
    pn = jnp.maximum(q, 0.0)
    if s.carry_F:
        bc = (cc * h) * Bm[Mk - 1]
        Fn = (c.A[n] * F + c.B[n] * Fm + dtn * bc * q[Mk - 1]) / al + h * jnp.sum(jnp.minimum(q, 0.0))
    else:
        Fn = F
    return pn, Fn, F


def make_march(s: GenSpec, mode: str, block: int | None = None):
    """theta (dict of per-span arrays: rho_eta[K], rho_eps[K], rem[K, N], alpha[K]) -> S_raw, or (S_raw,
    F_raw) with carry_F. mode "value": early-stopping while_loop; "grad": fixed-length checkpointed scan."""
    c = _Const(s)
    N, nsteps = s.N, s.nsteps
    h = 1.0 / N
    p0 = jnp.zeros(N).at[0].set(1.0 / h)
    S0 = p0.sum() * h

    if mode == "value":
        def march(theta):
            Ys = _clock_starts(theta, c, s) if s.clock else None
            buf = jnp.full(nsteps + 1, S0)
            fbuf = jnp.zeros(nsteps + 1)

            def body(st):
                n, p, pp, F, Fm, buf, fbuf, done = st
                pn, Fn, Fh = _step(theta, c, s, Ys, n, p, pp, F, Fm)
                S = pn.sum() * h
                buf = jax.lax.dynamic_update_slice(buf, S[None], (n + 1,))
                fbuf = jax.lax.dynamic_update_slice(fbuf, Fn[None], (n + 1,))
                done = (S < s.Gtol) & (c.tau[n + 1] > 0.3)
                return n + 1, pn, p, Fn, Fh, buf, fbuf, done

            z0 = jnp.zeros(())
            n, _, _, _, _, buf, fbuf, _ = jax.lax.while_loop(
                lambda st: (st[0] < nsteps) & (~st[7]), body,
                (0, p0, jnp.zeros(N), z0, z0, buf, fbuf, jnp.array(False)))
            idx = jnp.arange(nsteps + 1)
            S_raw = jnp.where(idx <= n, buf, buf[n])
            if s.carry_F:
                return S_raw, jnp.where(idx <= n, fbuf, fbuf[n])
            return S_raw
        return march

    K = int(block or max(1, int(np.ceil(np.sqrt(nsteps)))))
    nb = (nsteps + K - 1) // K

    def march(theta):
        Ys = _clock_starts(theta, c, s) if s.clock else None

        @jax.checkpoint
        def run_block(carry, b):
            def inner(carry, j):
                p, pp, F, Fm = carry
                n = b * K + j
                live = n < nsteps
                nn = jnp.minimum(n, nsteps - 1)
                pn, Fn, Fh = _step(theta, c, s, Ys, nn, p, pp, F, Fm)
                new = (jnp.where(live, pn, p), jnp.where(live, p, pp), jnp.where(live, Fn, F),
                       jnp.where(live, Fh, Fm))
                return new, (new[0].sum() * h, new[2])
            return jax.lax.scan(inner, carry, jnp.arange(K))

        z0 = jnp.zeros(())
        _, (S, F) = jax.lax.scan(run_block, (p0, jnp.zeros(N), z0, z0), jnp.arange(nb))
        S_raw = jnp.concatenate([S0[None], S.reshape(-1)[:nsteps]])
        if s.carry_F:
            return S_raw, jnp.concatenate([jnp.zeros(1), F.reshape(-1)[:nsteps]])
        return S_raw
    return march


def truncate_gen(S_raw, tau, Gtol, nsteps):
    """`stepper.truncate` on any mesh: the stop index, cummin, clip, held S, and the density by
    np.gradient with coordinates (SRK's own call) with the one-sided end at the stop."""
    idx = jnp.arange(nsteps + 1)
    hit = (S_raw < Gtol) & (tau > 0.3) & (idx >= 1)
    k = jnp.where(hit.any(), jnp.argmax(hit), nsteps)
    Sc = jnp.clip(jax.lax.cummin(S_raw), 0.0, 1.0)
    Sk = Sc[k]
    S = jnp.where(idx <= k, Sc, Sk)
    g = -jnp.gradient(Sc, tau)
    km1 = jnp.maximum(k - 1, 0)
    span = jnp.where(k > 0, tau[k] - tau[km1], 1.0)
    g_end = (Sc[km1] - Sk) / span
    p = jnp.where(idx < k, g, jnp.where(idx == k, g_end, 0.0))
    return S, jnp.clip(p, 0.0, None), k


def truncate_F_gen(F_raw, k, nsteps):
    idx = jnp.arange(nsteps + 1)
    Fc = jnp.clip(jax.lax.cummax(F_raw), 0.0, 1.0)
    return jnp.where(idx <= k, Fc, Fc[k])


def density_F_gen(F_raw, S, p, k, tau, nsteps):
    """`stepper.density_F` on any mesh (coordinates in np.gradient; register D-106)."""
    idx = jnp.arange(nsteps + 1)
    Fc = jnp.clip(jax.lax.cummax(F_raw), 0.0, 1.0)
    km1 = jnp.maximum(k - 1, 0)
    gF = jnp.gradient(Fc, tau)
    span = jnp.where(k > 0, tau[k] - tau[km1], 1.0)
    gF_end = (Fc[k] - Fc[km1]) / span
    pF = jnp.where(idx < k, gF, jnp.where(idx == k, gF_end, 0.0))
    F_big = Fc[jnp.minimum(idx + 1, k)]
    S_big = S[jnp.maximum(idx - 1, 0)]
    use_F = (idx <= k) & (F_big < S_big)
    return jnp.where(use_F, jnp.clip(pF, 0.0, None), p), use_F
