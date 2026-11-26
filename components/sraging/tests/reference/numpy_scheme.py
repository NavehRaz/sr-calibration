"""An independent NumPy/SciPy march of the discrete Fokker-Planck scheme (task T017; research R3, R4, R5, R11).

This is a reference, not an implementation: it exists to disagree with the JAX engine when the engine is
wrong. It is written from the algorithms document (sections 2 and 6) and SRK's NumPy `fpt_full`, not from
`sraging.compat` or `sraging.engine`, and it favours readability over speed (one banded solve per step, in
a Python loop).

The discrete problem
--------------------
Damage y in [0, 1]; N cells, h = 1/N. Unknowns p_i at nodes y_i = i h, i = 0..N-1; faces
y_{i+1/2} = (i + 1/2) h; the absorbing node y_N = 1 carries p_N = 0 and is not stored. Node 0 gets a
control volume of full width h, so the zero-flux wall sits at y = -h/2 (algorithms 2.2 (iii)), and
S = h sum_i p_i. The start is p_0 = 1/h (unit mass at node 0), S^0 = h * (1/h), F^0 = 0.

Drift at face i+1/2 at the NEW time level of a step,
    a_{i+1/2} = rho_eta * Y(tau_{n+1}) - rho_beta * y_{i+1/2} / (y_{i+1/2} + kappa),
where Y is the accumulated DPUs (Y = tau unless some segment has a DPU accumulation rate A != 1; only production
uses it).

Scharfetter-Gummel flux (eq. s2-sg), with P = a h / rho_eps and B(z) = z / (e^z - 1):
    J_{i+1/2} = (rho_eps / h) [B(-P_{i+1/2}) p_i - B(P_{i+1/2}) p_{i+1}].
B is evaluated as z / expm1(z) after clipping z to [-500, 500], and as 1 - z/2 for |z| < 1e-10 (SRK's
`_bernoulli`, part of the scheme). With J_{-1/2} = 0 and p_M = 0 (M active nodes, below), h dp_i/dtau =
J_{i-1/2} - J_{i+1/2}, i.e. dp/dtau = A p with c = rho_eps / h^2, B+-_i = B(+-P_{i+1/2}):
    A[i, i-1] = c B-_{i-1},   A[i, i] = -c (B-_i + B+_{i-1}),   A[i, i+1] = c B+_i,
with B+_{-1} = 0 in row 0 and no (i, i+1) entry in row M-1 (eq. s2-stencil).

Time stepping on any increasing node list tau_0 = 0 < tau_1 < ...: step n (tau_n -> tau_{n+1}, size
dtau_n) solves
    alpha_n p^{n+1} - dtau_n A(tau_{n+1}) p^{n+1} = a_n p^n + b_n p^{n-1}.
BE is (alpha, a, b) = (1, 1, 0). BDF2 on a non-uniform mesh, with w = dtau_n / dtau_{n-1}, is
    alpha = (1 + 2w)/(1 + w),  a = 1 + w,  b = -w^2/(1 + w),
which is (3/2, 2, -1/2) at w = 1 (research R4). Scheme "bdf2" takes BE for steps n < nstart (default 20)
and whenever there is no history (after a threshold move), BDF2 otherwise; scheme "be" is BE throughout.
After each solve p = max(q, 0), S^{n+1} = h sum p, and the march stops at the first S^{n+1} < Gtol with
tau_{n+1} > 0.3 (`early_stop=False` disables it).

The absorbed mass F (research R3). Every column of A sums to zero except the last active one, so
h 1^T A q = -bc q_{M-1}, with bc = h c B-_{M-1} = (rho_eps / h) B(-P_{M-1/2}). Summing the step equation
times h gives alpha S_q + dtau bc q_{M-1} = a S^n + b S^{n-1}, where S_q = h sum q. With S = 1 - F and
alpha = a + b (true for BE and for every w),
    F_q = (a F^n + b F^{n-1} + dtau bc q_{M-1}) / alpha,
and the clip p = max(q, 0) adds -h sum min(q, 0) to S, so
    F^{n+1} = F_q + h sum min(q^{n+1}, 0).
F is accumulated from small positive pieces, so it keeps full relative precision where S ~ 1, which is
the point of carrying it. F + S = 1 is an identity for exact solves; in float64 it drifts by the solves'
round-off, which grows with the step count and with dtau rho_eps / h^2 (measured: 5e-15 at N = 150 over
438 BDF2 steps, 8.6e-13 at N = 300 over 1564 steps at rho_eps = 0.2).

Segments (algorithms section 6; research R4, R5). A segment is (tau_start, tau_stop, rho_eta, rho_beta,
rho_eps, kappa, A, x_frac). Segments are contiguous; the last stop may be inf. Every edge inside the mesh
must be a mesh node (within 1e-12 relative); step n uses the segment that contains (tau_n, tau_{n+1}].
- Clock: Y_0 = tau_start of the first segment, Y_{k+1} = Y_k + A_k (tau_stop_k - tau_start_k), and inside
  segment k, Y(tau) = Y_k + A_k (tau - tau_start_k). With every A_k == 1 the drift uses tau itself.
- Moving threshold: segment k has M_k = round(N x_frac_k) active nodes 0..M_k-1 (x_frac is the threshold
  as a fraction of the grid's top), refused if M_k < 2 or x_frac > 1. When M changes at the start of a
  step: on lowering, the mass h sum_{i >= M_new} p_i is added to F and those nodes are zeroed (they have
  crossed and die at once); on raising, the exposed nodes are empty. Either way the BDF2 history is
  discarded, so that step is BE.
- An edge in any other parameter does NOT restart BDF2.

Outputs: `march` returns the node times, S_raw and F_raw up to and including the stop index k, and the
per-step scheme, active-node count and segment. `density_route` is SRK `fpt_full_density`'s gradient
route; `interval_masses` and `loglik` evaluate interval-censored deaths by both routes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

SCHEMES = ("be", "bdf2")
P_FLOOR = 9.859676543759771e-305       # exp(-700): the floor SRK applies before a log
LOG_FLOOR = -700.0
EDGE_RTOL = 1e-12                      # a segment edge matches a mesh node within this (relative) distance


@dataclass(frozen=True)
class Segment:
    tau_start: float
    tau_stop: float
    rho_eta: float
    rho_beta: float
    rho_eps: float
    kappa: float
    A: float = 1.0
    x_frac: float = 1.0


@dataclass(frozen=True)
class March:
    tau: np.ndarray          # nodes 0..k
    S_raw: np.ndarray        # h sum p at each node
    F_raw: np.ndarray        # absorbed mass by the scheme's mass balance
    k: int                   # index of the last node computed (the early-stop index, or the last node)
    stopped_early: bool
    bdf2: np.ndarray         # per step 0..k-1: True where the step was BDF2
    M: np.ndarray            # per step: number of active nodes
    segment: np.ndarray      # per step: segment index


def bernoulli(z):
    """B(z) = z / (e^z - 1), clipped to |z| <= 500, = 1 - z/2 for |z| < 1e-10."""
    z = np.clip(np.asarray(z, dtype=float), -500.0, 500.0)
    out = np.empty_like(z)
    tiny = np.abs(z) < 1e-10
    out[tiny] = 1.0 - z[tiny] / 2.0
    out[~tiny] = z[~tiny] / np.expm1(z[~tiny])
    return out


def uniform_mesh(dt, tau_max):
    """Nodes n * dt, n = 0..int(tau_max / dt), each the float64 product (as SRK's (n + 1) * dt)."""
    return np.arange(int(tau_max / dt) + 1) * float(dt)


def geometric_mesh(tau_end, nsteps, growth):
    """nsteps steps from 0 to tau_end whose sizes grow by one constant ratio, last/first = growth."""
    lam = math.log(growth) * nsteps / (nsteps - 1)
    s = np.arange(nsteps + 1) / nsteps
    tau = tau_end * np.expm1(lam * s) / math.expm1(lam)
    tau[-1] = tau_end
    return tau


def _check_mesh(tau):
    tau = np.asarray(tau, dtype=float)
    if tau.ndim != 1 or tau.size < 2:
        raise ValueError("tau must be a 1-D array of at least two nodes")
    if tau[0] != 0.0:
        raise ValueError(f"the mesh must start at tau = 0, got {tau[0]}")
    if not np.all(np.isfinite(tau)) or not np.all(np.diff(tau) > 0):
        raise ValueError("the mesh must be finite and strictly increasing")
    return tau


def _node_of(edge, tau):
    """Index of the mesh node equal to `edge` (within EDGE_RTOL); refuse an edge between nodes."""
    i = int(np.argmin(np.abs(tau - edge)))
    if abs(tau[i] - edge) > EDGE_RTOL * max(1.0, abs(edge)):
        raise ValueError(f"segment edge {edge!r} is not a mesh node (nearest {tau[i]!r}); insert it")
    return i


def _place(segments, tau, N):
    """Validate the segments; return (stop node index per segment, active-node count per segment)."""
    if not segments:
        raise ValueError("need at least one segment")
    if abs(segments[0].tau_start - tau[0]) > EDGE_RTOL:
        raise ValueError(f"the first segment must start at tau = 0, got {segments[0].tau_start}")
    for s0, s1 in zip(segments, segments[1:]):
        if not (abs(s0.tau_stop - s1.tau_start) <= EDGE_RTOL * max(1.0, abs(s1.tau_start))):
            raise ValueError(f"segments must be contiguous: {s0.tau_stop} != {s1.tau_start}")
    stop_idx, M_of = [], []
    last = len(tau) - 1
    for s in segments:
        if not (s.tau_stop > s.tau_start):
            raise ValueError(f"empty or reversed segment ({s.tau_start}, {s.tau_stop})")
        for name in ("rho_eta", "rho_beta", "rho_eps", "kappa", "A", "x_frac"):
            if not math.isfinite(getattr(s, name)):
                raise ValueError(f"segment {name} must be finite, got {getattr(s, name)}")
        if not (s.rho_eps > 0 and s.kappa > 0):
            raise ValueError("segment rho_eps and kappa must be > 0")
        if not (0 < s.x_frac <= 1):
            raise ValueError(f"x_frac must be in (0, 1], got {s.x_frac}")
        M = int(round(N * s.x_frac))
        if M < 2:
            raise ValueError(f"x_frac {s.x_frac} leaves {M} active node(s) at N={N}; need >= 2")
        M_of.append(M)
        stop_idx.append(last + 1 if s.tau_stop >= tau[-1] * (1 + EDGE_RTOL) else _node_of(s.tau_stop, tau))
    if stop_idx[-1] < last:
        raise ValueError(f"the segments end at {segments[-1].tau_stop}, before the mesh ({tau[-1]}); "
                         "extend the last one (its stop may be inf)")
    stop_idx[-1] = last + 1
    return stop_idx, M_of


def march(N, tau, segments=None, theta=None, *, scheme="bdf2", nstart=20, Gtol=1e-11, early_stop=True):
    """March the scheme on the node list `tau`.

    Give either `theta` = (rho_eta, rho_beta, rho_eps, kappa) for one parameter set over the whole mesh, or
    `segments`, a list of `Segment` (or 8-tuples in its field order).
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    N = int(N)
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}")
    tau = _check_mesh(tau)
    if (segments is None) == (theta is None):
        raise ValueError("give exactly one of `theta` and `segments`")
    if segments is None:
        segments = [Segment(0.0, math.inf, *map(float, theta))]
    segments = [s if isinstance(s, Segment) else Segment(*map(float, s)) for s in segments]
    stop_idx, M_of = _place(segments, tau, N)

    h = 1.0 / N
    yf = (np.arange(N) + 0.5) * h
    use_clock = any(s.A != 1.0 for s in segments)
    Y_start = [segments[0].tau_start]
    for s in segments[:-1]:
        Y_start.append(Y_start[-1] + s.A * (s.tau_stop - s.tau_start))

    nsteps = len(tau) - 1
    S = np.empty(nsteps + 1)
    F = np.empty(nsteps + 1)
    used_bdf2 = np.zeros(nsteps, dtype=bool)
    M_step = np.zeros(nsteps, dtype=int)
    seg_step = np.zeros(nsteps, dtype=int)

    p = np.zeros(N)
    p[0] = 1.0 / h
    S[0] = h * p.sum()
    F[0] = 0.0
    F_now = 0.0               # F^n, including any mass absorbed by a threshold move at tau_n
    history = None            # (p^{n-1}, F^{n-1}, dtau_{n-1}) while BDF2 may use it
    M = M_of[0]
    k_seg = 0
    k = nsteps
    stopped = False
    for n in range(nsteps):
        while n >= stop_idx[k_seg]:
            k_seg += 1
        seg = segments[k_seg]
        if M_of[k_seg] != M:                       # the threshold moves at tau_n
            M_new = M_of[k_seg]
            if M_new < M:
                F_now += h * p[M_new:].sum()       # above the new threshold: already crossed, dies now
                p[M_new:] = 0.0
            M = M_new
            history = None                         # the history lives on a different domain

        dtau = tau[n + 1] - tau[n]
        t_new = tau[n + 1]
        Y = Y_start[k_seg] + seg.A * (t_new - seg.tau_start) if use_clock else t_new
        a_face = seg.rho_eta * Y - seg.rho_beta * yf[:M] / (yf[:M] + seg.kappa)
        P = a_face * h / seg.rho_eps
        Bp, Bm = bernoulli(P), bernoulli(-P)
        c = seg.rho_eps / h ** 2

        if scheme == "bdf2" and n >= nstart and history is not None:
            p_old, F_old, dtau_old = history
            w = dtau / dtau_old
            alpha, a, b = (1.0 + 2.0 * w) / (1.0 + w), 1.0 + w, -(w * w) / (1.0 + w)
            rhs = a * p[:M] + b * p_old[:M]
            used_bdf2[n] = True
        else:
            F_old = 0.0
            alpha, a, b = 1.0, 1.0, 0.0
            rhs = p[:M].copy()

        # (alpha I - dtau A) q = rhs as a (1, 1)-banded system: row 0 upper, row 1 diagonal, row 2 lower.
        band = np.zeros((3, M))
        band[0, 1:] = -dtau * c * Bp[:M - 1]                                   # entry (i, i+1)
        band[1, :] = alpha + dtau * c * (Bm + np.concatenate(([0.0], Bp[:M - 1])))
        band[2, :-1] = -dtau * c * Bm[:M - 1]                                  # entry (i+1, i)
        q = solve_banded((1, 1), band, rhs)

        bc = (seg.rho_eps / h) * Bm[M - 1]
        F_next = (a * F_now + b * F_old + dtau * bc * q[M - 1]) / alpha + h * np.minimum(q, 0.0).sum()

        p_next = np.zeros(N)
        p_next[:M] = np.maximum(q, 0.0)
        history = (p, F_now, dtau)
        p, F_now = p_next, F_next
        S[n + 1] = h * p.sum()
        F[n + 1] = F_next
        M_step[n], seg_step[n] = M, k_seg
        if early_stop and S[n + 1] < Gtol and tau[n + 1] > 0.3:
            k, stopped = n + 1, True
            break

    return March(tau=tau[:k + 1].copy(), S_raw=S[:k + 1], F_raw=F[:k + 1], k=k, stopped_early=stopped,
                 bdf2=used_bdf2[:k], M=M_step[:k], segment=seg_step[:k])


# ---- what the likelihood reads off the march --------------------------------------------------------------
def density_route(tau, S_raw):
    """SRK `fpt_full_density`'s gradient route: (S monotone, f) with S = clip(running min, 0, 1) and
    f = clip(-np.gradient(S, tau), 0, None) (np.gradient with the node coordinates)."""
    tau = np.asarray(tau, dtype=float)
    S = np.clip(np.minimum.accumulate(np.asarray(S_raw, dtype=float)), 0.0, 1.0)
    return S, np.clip(-np.gradient(S, tau), 0.0, None)


def monotone(res):
    """(S, F) made monotone: S = clip(running min of S_raw, 0, 1), F = clip(running max of F_raw, 0, 1)."""
    S = np.clip(np.minimum.accumulate(res.S_raw), 0.0, 1.0)
    F = np.clip(np.maximum.accumulate(res.F_raw), 0.0, 1.0)
    return S, F


def interval_masses(res, t_lo, t_hi):
    """Masses of the intervals [t_lo, t_hi) (tau units) by both routes.

    S is interpolated linearly with S = 1 before the mesh and S_end after it; F with 0 before and F_end
    after. m_S = S(t_lo) - S(t_hi), m_F = F(t_hi) - F(t_lo); `use_F` marks where F's larger operand
    F(t_hi) is below S's larger operand S(t_lo) (the difference with the smaller ulp)."""
    S, F = monotone(res)
    t_lo, t_hi = np.asarray(t_lo, dtype=float), np.asarray(t_hi, dtype=float)
    S_lo = np.interp(t_lo, res.tau, S, left=1.0, right=S[-1])
    S_hi = np.interp(t_hi, res.tau, S, left=1.0, right=S[-1])
    F_lo = np.interp(t_lo, res.tau, F, left=0.0, right=F[-1])
    F_hi = np.interp(t_hi, res.tau, F, left=0.0, right=F[-1])
    return dict(m_S=S_lo - S_hi, m_F=F_hi - F_lo, use_F=F_hi < S_lo, S_lo=S_lo)


def _log_floor(x):
    return np.maximum(np.log(np.maximum(x, P_FLOOR)), LOG_FLOOR)


def loglik(res, ages, died, delta, route="auto"):
    """sum over deaths of log m([t, t + delta)) + sum over censored of log S(t), ages in tau units.

    route: "S" (m = S difference), "F" (m = F difference) or "auto" (F where `use_F`). Masses are clipped
    at 0 and every term is floored at log(P_FLOOR) = -700, as SRK does."""
    if route not in ("auto", "S", "F"):
        raise ValueError(f"route must be 'auto', 'S' or 'F', got {route!r}")
    ages = np.asarray(ages, dtype=float)
    died = np.asarray(died, dtype=bool)
    ms = interval_masses(res, ages, ages + np.broadcast_to(np.asarray(delta, dtype=float), ages.shape))
    m = {"S": ms["m_S"], "F": ms["m_F"], "auto": np.where(ms["use_F"], ms["m_F"], ms["m_S"])}[route]
    m = np.maximum(m, 0.0)
    return float(np.sum(_log_floor(m[died])) + np.sum(_log_floor(ms["S_lo"][~died])))
