"""The discrete Fokker-Planck scheme in mpmath (task T018; research R3, R11 item 1, R12).

Same discrete problem as `numpy_scheme.py` (its docstring states it), restricted to what the tester's
`reports/basin/likelihood_fix/v2/t7_mpref.py` covers: one parameter set, the uniform mesh tau_n = n dt, BE
for the first `nstart` steps and then BDF2 (scheme "be": BE throughout), the clip p = max(q, 0), S = h sum p,
SRK's early stop (S < Gtol once tau > 0.3), plus the absorbed mass F by the scheme's mass balance,
    F^{n+1} = (a F^n + b F^{n-1} + dt bc q_{N-1}) / alpha + h sum min(q^{n+1}, 0),  bc = (rho_eps/h) B(-P_{N-1/2}),
and the interval masses by both routes, m_S = S(t) - S(t + delta) and m_F = F(t + delta) - F(t), with linear
interpolation at the ages. It follows t7's approach (exact arithmetic for the float scheme, a tridiagonal
Thomas solve, np.interp semantics), not its code.

What is exact and what is float64
---------------------------------
- The inputs are float64 numbers, converted to mpmath exactly: theta, dt, and the mesh nodes tau_{n+1}, which
  are the float64 products (n + 1) * dt that the float marches use (as t7 does).
- Everything derived from them is computed at `dps` digits (default 40): h = 1/N, the faces, the removal term,
  the Bernoulli pair, the solve, S, F, the interpolation and the logs. (t7 took h and the removal term from
  float64; the difference is a relative 1e-16 perturbation of the problem, which moves no conclusion.)
- B(z) = z / expm1(z) after clipping z to [-500, 500], with B(-z) = z + B(z) (exact; the clip is odd).
  SRK's float switch to 1 - z/2 for |z| < 1e-10 is not reproduced: it is the same function to 1e-20.
In exact arithmetic F + S = 1 holds at every node (the mass balance is an identity of the scheme); here it
holds to about 10^-dps (tested). The point of the reference is that S(t) - S(t + delta) is exact here, so it
measures how much of a float64 interval mass is round-off: an early death's mass of 1e-14 is a difference of
two numbers within 1e-14 of 1, which float64 cannot resolve and F's route can.

Cost
----
One step costs N expm1 calls and about 20 N multiplications at `dps` digits, so a march costs
N * nsteps * (rate at dps). `estimate_seconds` calibrates the rate on a tiny march; `march` prints the
estimate before any run with N * nsteps > TINY_NODE_STEPS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import mpmath
import numpy as np

SCHEMES = ("be", "bdf2")
P_FLOOR = 9.859676543759771e-305       # exp(-700): the floor SRK applies before a log
LOG_FLOOR = -700.0
TINY_NODE_STEPS = 10_000               # a run with N * nsteps above this prints its cost estimate first
_CLIP = 500


@dataclass(frozen=True)
class MPMarch:
    tau: np.ndarray        # the float64 nodes n * dt, n = 0..k
    S_raw: list            # mpmath numbers, h sum p at each node
    F_raw: list            # mpmath numbers, the absorbed mass by the mass balance
    k: int                 # the last node computed (the early-stop index, or nsteps)
    stopped_early: bool
    dps: int
    seconds: float


def _bernoulli_pair(z, clip):
    """(B(z), B(-z)) with z clipped to [-clip, clip]; B(-z) = z + B(z)."""
    if z > clip:
        z = clip
    elif z < -clip:
        z = -clip
    if z == 0:
        return mpmath.mpf(1), mpmath.mpf(1)
    Bp = z / mpmath.expm1(z)
    return Bp, z + Bp


def march(theta, N, dt, tau_max, *, scheme="bdf2", nstart=20, Gtol=1e-11, early_stop=True, dps=40,
          announce=True):
    """March the scheme for theta = (rho_eta, rho_beta, rho_eps, kappa) on nodes n * dt, n <= int(tau_max/dt)."""
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    N, dt, dps = int(N), float(dt), int(dps)
    if N < 2 or not dt > 0:
        raise ValueError(f"need N >= 2 and dt > 0, got N={N}, dt={dt}")
    nsteps = int(tau_max / dt)
    if nsteps < 1:
        raise ValueError(f"tau_max={tau_max} is shorter than one step dt={dt}")
    if announce and N * nsteps > TINY_NODE_STEPS:
        est = estimate_seconds(N, nsteps, dps=dps)
        print(f"mp_scheme.march: N={N} x {nsteps} steps at {dps} digits, estimated {est:.0f} s "
              f"({est / 60:.1f} min)", flush=True)
    t_start = time.perf_counter()
    with mpmath.workdps(dps):
        mpf, fsum = mpmath.mpf, mpmath.fsum
        rho_eta, rho_beta, rho_eps, kappa = (mpf(float(v)) for v in theta)
        if not (rho_eps > 0 and kappa > 0):
            raise ValueError("rho_eps and kappa must be > 0")
        zero, clip = mpf(0), mpf(_CLIP)
        h = mpf(1) / N
        dtm = mpf(dt)
        dc = dtm * rho_eps / h ** 2                 # dt * c
        wall = rho_eps / h                          # bc = wall * B(-P_{N-1/2})
        removal = [rho_beta * y / (y + kappa) for y in ((i + mpf(1) / 2) * h for i in range(N))]

        p = [zero] * N
        p[0] = 1 / h
        S, F = [h * fsum(p)], [zero]
        p_old = None
        k, stopped = nsteps, False
        for n in range(nsteps):
            t_new = mpf((n + 1) * dt)               # the float64 node, as the float marches have it
            pairs = [_bernoulli_pair((rho_eta * t_new - removal[i]) * h / rho_eps, clip) for i in range(N)]
            Bp = [bp for bp, _ in pairs]
            Bm = [bm for _, bm in pairs]
            if scheme == "bdf2" and n >= nstart:
                alpha, a, b = mpf(3) / 2, mpf(2), mpf(-1) / 2
                rhs = [a * p[i] + b * p_old[i] for i in range(N)]
            else:
                alpha, a, b = mpf(1), mpf(1), zero
                rhs = list(p)
            # (alpha I - dt A) q = rhs: lower (i, i-1) = -dc Bm[i-1], diagonal alpha + dc (Bm[i] + Bp[i-1]),
            # upper (i, i+1) = -dc Bp[i]; Thomas elimination, then back substitution.
            diag = [alpha + dc * (Bm[i] + (Bp[i - 1] if i else zero)) for i in range(N)]
            for i in range(1, N):
                factor = (-dc * Bm[i - 1]) / diag[i - 1]
                diag[i] -= factor * (-dc * Bp[i - 1])
                rhs[i] -= factor * rhs[i - 1]
            q = [zero] * N
            q[N - 1] = rhs[N - 1] / diag[N - 1]
            for i in range(N - 2, -1, -1):
                q[i] = (rhs[i] + dc * Bp[i] * q[i + 1]) / diag[i]

            F_hist = b * F[n - 1] if b else zero
            F_next = (a * F[n] + F_hist + dtm * wall * Bm[N - 1] * q[N - 1]) / alpha \
                + h * fsum(v for v in q if v < 0)
            p_old, p = p, [v if v > 0 else zero for v in q]
            S.append(h * fsum(p))
            F.append(F_next)
            if early_stop and S[-1] < Gtol and (n + 1) * dt > 0.3:
                k, stopped = n + 1, True
                break
    return MPMarch(tau=np.arange(k + 1) * dt, S_raw=S, F_raw=F, k=k, stopped_early=stopped, dps=dps,
                   seconds=time.perf_counter() - t_start)


def estimate_seconds(N, nsteps, dps=40):
    """Seconds a march of N nodes x nsteps steps will take at `dps` digits, from a tiny calibration march
    (16 nodes x 24 steps, the faster of two runs). The cost is linear in N * nsteps."""
    n_cal, s_cal, dt = 16, 24, 1e-3
    best = float("inf")
    for _ in range(2):
        t0 = time.perf_counter()
        march((1.0, 1.0, 0.1, 0.1), n_cal, dt, (s_cal + 0.5) * dt, early_stop=False, dps=dps, announce=False)
        best = min(best, time.perf_counter() - t0)
    return best / (n_cal * s_cal) * int(N) * int(nsteps)


# ---- what the likelihood reads off the march ------------------------------------------------------------
def monotone(res):
    """(S, F) as lists: S = clip(running min of S_raw, 0, 1), F = clip(running max of F_raw, 0, 1)."""
    with mpmath.workdps(res.dps):
        zero, one = mpmath.mpf(0), mpmath.mpf(1)
        S, F = [], []
        s_run = f_run = None
        for s, f in zip(res.S_raw, res.F_raw):
            s_run = s if s_run is None or s < s_run else s_run
            f_run = f if f_run is None or f > f_run else f_run
            S.append(min(max(s_run, zero), one))
            F.append(min(max(f_run, zero), one))
    return S, F


def _interp(x, tau, vals, left, right):
    """np.interp(x, tau, vals, left, right), with the weight computed exactly from the float64 x and nodes."""
    if x < tau[0]:
        return left
    if x > tau[-1]:
        return right
    i = min(max(int(np.searchsorted(tau, x, side="right")), 1), len(tau) - 1)
    w = (mpmath.mpf(float(x)) - mpmath.mpf(float(tau[i - 1]))) / (mpmath.mpf(float(tau[i])) - mpmath.mpf(float(tau[i - 1])))
    return vals[i - 1] + w * (vals[i] - vals[i - 1])


def interval_masses(res, t_lo, t_hi):
    """For each interval [t_lo, t_hi) (tau units): dict of m_S, m_F, use_F (F(t_hi) < S(t_lo)) and S_lo."""
    S, F = monotone(res)
    out = dict(m_S=[], m_F=[], use_F=[], S_lo=[])
    with mpmath.workdps(res.dps):
        zero, one = mpmath.mpf(0), mpmath.mpf(1)
        for lo, hi in zip(np.atleast_1d(t_lo), np.atleast_1d(t_hi)):
            S_lo = _interp(lo, res.tau, S, one, S[-1])
            S_hi = _interp(hi, res.tau, S, one, S[-1])
            F_lo = _interp(lo, res.tau, F, zero, F[-1])
            F_hi = _interp(hi, res.tau, F, zero, F[-1])
            out["m_S"].append(S_lo - S_hi)
            out["m_F"].append(F_hi - F_lo)
            out["use_F"].append(bool(F_hi < S_lo))
            out["S_lo"].append(S_lo)
    return out


def loglik(res, ages, died, delta, route="auto"):
    """sum over deaths of log m([t, t + delta)) + sum over censored of log S(t), ages in tau units, in mpmath.

    route: "S", "F" or "auto" (F where F(t + delta) < S(t)). Masses are clipped at 0 and every term is
    floored at log(P_FLOOR) = -700, as SRK does."""
    if route not in ("auto", "S", "F"):
        raise ValueError(f"route must be 'auto', 'S' or 'F', got {route!r}")
    ages = np.asarray(ages, dtype=float)
    died = np.asarray(died, dtype=bool)
    hi = ages + np.broadcast_to(np.asarray(delta, dtype=float), ages.shape)
    ms = interval_masses(res, ages, hi)
    with mpmath.workdps(res.dps):
        zero, floor = mpmath.mpf(0), mpmath.mpf(P_FLOOR)

        def log_floor(x):
            return max(mpmath.log(max(x, floor)), mpmath.mpf(LOG_FLOOR))

        total = zero
        for j in range(len(ages)):
            if died[j]:
                m = {"S": ms["m_S"][j], "F": ms["m_F"][j]}.get(route)
                if m is None:
                    m = ms["m_F"][j] if ms["use_F"][j] else ms["m_S"][j]
                total += log_floor(max(m, zero))
            else:
                total += log_floor(ms["S_lo"][j])
    return total


# ---- per-row terms by definition, for the `mpref` acceptance phase (review T064, sr-naveh N5) -------------
def _density(res, S):
    """The density on the nodes as `stepper.truncate` builds it from the monotone S: central differences
    inside, one-sided at node 0 and at the last node k, in mpmath (uniform mesh)."""
    k = len(S) - 1
    dt = mpmath.mpf(float(res.tau[1] - res.tau[0])) if k >= 1 else mpmath.mpf(1)
    p = []
    for n in range(k + 1):
        if n == 0:
            g = -(S[1] - S[0]) / dt if k >= 1 else mpmath.mpf(0)
        elif n < k:
            g = -(S[n + 1] - S[n - 1]) / (2 * dt)
        else:
            g = (S[k - 1] - S[k]) / dt
        p.append(max(g, mpmath.mpf(0)))
    return p


def terms(marches, ages, died, delta, L, *, h=0.0, entry=None, window=None, rows=None):
    """Per-row log-likelihood terms (mpmath) by numerics "v1"'s observation model, by definition:
    - `marches`: one MPMarch, or [(weight, MPMarch), ...] for a mixture (S, F and the density are averaged
      over the nodes before anything nonlinear, as the engine does);
    - an interval death: log+ [S_obs(max(t, e)) - S_obs(t + delta)], S_obs(x) = S(x / L) e^{-h x};
      exact times (delta None): log+ [(p(t / L) / L + h S(t / L)) e^{-h t}], p the stencil density, 0 past
      the march's stop;
    - a censored row: log+ S_obs(t); each row minus log+ S_obs(e) with entry;
    - a window (lo, hi): rows with t < lo contribute 0; rows with t > hi, and deaths reaching past hi
      (register D-107), score log max(S_obs(hi), 1e-300) floored at -700; every at-risk row minus log+ S_obs(lo).
    log+ x = max(log(max(x, e^-700)), -700). Returns a list of mpf for `rows` (default: all)."""
    if isinstance(marches, MPMarch):
        marches = [(1.0, marches)]
    dps = marches[0][1].dps
    ages = np.asarray(ages, float)
    died = np.asarray(died, bool)
    rows = range(ages.size) if rows is None else rows
    with mpmath.workdps(dps):
        zero, one = mpmath.mpf(0), mpmath.mpf(1)
        floor, hh, LL = mpmath.mpf(P_FLOOR), mpmath.mpf(float(h)), mpmath.mpf(float(L))
        comps = []
        for w, res in marches:
            S, _ = monotone(res)
            comps.append((mpmath.mpf(float(w)), res, S, _density(res, S) if delta is None else None))

        def logp(x):
            return max(mpmath.log(max(x, floor)), mpmath.mpf(LOG_FLOOR))

        def S_obs(x):
            tx = float(x) / float(L)
            s = sum(w * _interp(tx, res.tau, S, one, S[-1]) for w, res, S, _ in comps)
            return s * mpmath.exp(-hh * mpmath.mpf(float(x)))

        def dens_obs(x):
            tx = float(x) / float(L)
            p = zero
            for w, res, S, pn in comps:
                if tx <= float(res.tau[-1]):
                    p += w * _interp(tx, res.tau, pn, zero, zero)
            return (p / LL + hh * S_obs(x) * mpmath.exp(hh * mpmath.mpf(float(x)))) * mpmath.exp(-hh * mpmath.mpf(float(x)))

        out = []
        lo_term = None if window is None else logp(S_obs(window[0]))
        for i in rows:
            t = float(ages[i])
            e = 0.0 if entry is None else float(entry[i])
            d = None if delta is None else float(np.broadcast_to(np.asarray(delta, float), ages.shape)[i])
            if window is not None and t < window[0]:
                out.append(zero)
                continue
            if window is not None and (t > window[1] or (died[i] and t + (d or 0.0) > window[1])):
                v = max(mpmath.log(max(S_obs(window[1]), mpmath.mpf(1e-300))), mpmath.mpf(LOG_FLOOR))
            elif died[i] and d is not None:
                v = logp(max(S_obs(max(t, e)) - S_obs(t + d), zero))
            elif died[i]:
                v = logp(dens_obs(t))
            else:
                v = logp(S_obs(t))
            if entry is not None:
                v -= logp(S_obs(e))
            if lo_term is not None:
                v -= lo_term
            out.append(v)
    return out
