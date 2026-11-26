"""Exact first-passage-time LIKELIHOOD for the saturating-removal (SR) FP model.

CANONICAL SINGLE SOURCE of the FP likelihood/density/mixture layer. This module is
SELF-CONTAINED (numpy + scipy only) so the cluster-shipped SRK stands alone; the research
copy `sr_fp_fit_share/scripts/fp_likelihood.py` is a thin shim that re-exports from here.

The Fokker-Planck solve yields the survival S(tau) of the first-passage time. Its density is,
by mass conservation (reflecting wall at y=0, absorbing wall at y=1),

    p(tau) = -dS/dtau   ==   the absorbing-boundary probability flux.

So a *proper, exact* likelihood is a thin layer on ONE PDE solve -- deterministic, no MC noise,
no KDE smoothing (unlike baysianDistance, a KDE of *simulated* death times):

    loglik = sum_{deaths} log p_t(t_i) + sum_{censored} log S_t(t_i),
    S_t(t)=S(t/L),  p_t(t)=p(t/L)/L   (tau = t/L; L = data mean lifetime).

Optional constant external ("Makeham") hazard h: S -> S e^{-h t}, p -> e^{-h t}[p/L + h S].

Public API
----------
    fpt_full_density(rho_eta, rho_beta, rho_eps, kappa, ...) -> (tau, S, p, mean, cv, s_end)
    fp_loglik(dt, ev, rho_eta, rho_beta, rho_eps, kappa, L, external_hazard=np.inf, ...)
    fp_loglik_physical(dt, ev, eta, beta, epsilon, xc, kappa_phys, L=None, ...)
    fp_mixture_curves_physical(...), fp_loglik_het_physical(...)   # heterogeneity mixture
    choose_M(spread), _spread_proxy(...)                          # heuristic auto-M

Gotcha: `to_dimensionless(...)` returns (rho_beta, rho_eta, rho_eps, kappa) but the solver
`fpt_full`/`fpt_full_density` take rho_eta FIRST -- reorder (as the bodies below do).
"""
import os as _os

import numpy as np
from scipy.linalg import solve_banded

__all__ = ["fpt_full_density", "fp_loglik", "fp_loglik_physical",
           "fp_mixture_curves_physical", "fp_loglik_het_physical"]

_LOG_FLOOR = -700.0          # ~log(1e-304); avoids -inf from a single underflowed point
# Clip densities/survivals at exp(_LOG_FLOOR) rather than at 0.0 BEFORE taking the log. Clipping at
# 0.0 lets exact zeros through to np.log(0), which emits a RuntimeWarning ("divide by zero
# encountered in log") on every ordinary call before np.maximum rescues the value. Same result,
# no warning -- and it matches what the JAX port already does.
_P_FLOOR = 9.859676543759771e-305    # == math.exp(_LOG_FLOOR)
_HET_PARAMS = ("eta", "beta", "epsilon", "xc", "kappa")


# ---------------------------------------------------------------------------
# vendored FP solver (Scharfetter-Gummel + backward Euler) -- exact copy of the
# validated fpt_full (sr_fp_fit_share/scripts/fpt_full_model.py), numpy+scipy only.
# ---------------------------------------------------------------------------
def _trapz(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2.0))


def _bernoulli(z):
    z = np.clip(z, -500.0, 500.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(z) < 1e-10, 1.0 - z / 2.0, z / np.expm1(z))


def fpt_full(rho_eta, rho_beta, rho_eps, kappa, N=3000, dt=1e-3, tau_max=40.0, Gtol=1e-9,
             return_survival=False, scheme="be", nstart=20, absorb_flux=False):
    """(mean, CV, S_end) of the first-passage time; (…, tau_grid, S_grid) if return_survival.

    scheme : 'be' (default, UNCHANGED -- bit-identical to the original) or 'bdf2'.
        'be'   backward Euler, first order in tau.
        'bdf2' second-order BDF2, (1.5 I - dt A^{n+1}) p^{n+1} = 2 p^n - 0.5 p^{n-1}, started with
               `nstart` backward-Euler steps to damp the delta-function transient.

        Measured (relative CV error vs an N=4000, dt=1e-4 reference; wall time in brackets):
            stiff  (rho_beta=30, kappa=0.02)   BE dt=1e-3 2.4e-2 [74ms]   BDF2 dt=4e-3 2.8e-3 [18ms]
            sharp  (rho_beta=10, rho_eps=.02)  BE dt=1e-3 4.7e-2 [69ms]   BDF2 dt=4e-3 5.1e-3 [17ms]
            benign (rho_beta=3,  kappa=0.1)    BE dt=1e-3 1.4e-3 [113ms]  BDF2 dt=4e-3 4.6e-4 [23ms]
        i.e. ~4x faster AND ~10x more accurate on the stiff/sharp cohorts (human, Jack Russell,
        mosquito). The error is set by dt, essentially NOT by N -- N=200 ties N=900 at fixed dt --
        so the historical (N=900, dt=1e-3) budget is mis-allocated toward space.

        On Crank-Nicolson: the original docstring rejected CN because on a delta initial condition
        it oscillates, the p<0 clip then injects mass, and survival comes out ~2x too large. That
        is a statement about CN *from step 1*. With the same `nstart` damped startup, CN matches
        BDF2 to 2.7e-3 on the stiff case. The defect was the startup, not second-order stepping.

    absorb_flux : also return the first-passage DENSITY read off the absorbing-boundary flux,
        F = (rho_eps/h) * B(-P_{M-1}) * p_{M-1}, accumulated per step. This is the exact discrete
        flux implied by the same Scharfetter-Gummel discretization, so it is non-negative by
        construction and integrates to 1 - S_end (verified to 1.6e-13) -- unlike the usual
        -np.gradient(S) route, which needs a monotonization repair first. NOTE it is NOT more
        accurate than the gradient (2.7e-3 vs 2.2e-3 max density error at the default grid);
        adopt it for consistency and positivity, not for accuracy.
        With absorb_flux=True the return gains a trailing F array.
    """
    h = 1.0 / N
    yf = (np.arange(N) + 0.5) * h
    M = N; re = rho_eps; c = re / h ** 2
    rem = rho_beta * yf / (yf + kappa)          # time-invariant: hoisted out of the step loop

    def build(a_face):
        P = a_face * h / re
        Bp = _bernoulli(P); Bm = _bernoulli(-P)
        lo = np.zeros(M); di = np.zeros(M); up = np.zeros(M)
        i = np.arange(1, M - 1)
        lo[i] = c * Bm[i - 1]; di[i] = -c * (Bm[i] + Bp[i - 1]); up[i] = c * Bp[i]
        di[0] = -c * Bm[0]; up[0] = c * Bp[0]
        lo[M - 1] = c * Bm[M - 2]; di[M - 1] = -c * (Bm[M - 1] + Bp[M - 2])
        return lo, di, up, Bm[M - 1]

    p = np.zeros(M); p[0] = 1.0 / h
    p_prev = None
    nsteps = int(tau_max / dt)
    S = np.empty(nsteps + 1); tl = np.empty(nsteps + 1); S[0] = p.sum() * h; tl[0] = 0.0; last = 0
    F = np.empty(nsteps + 1) if absorb_flux else None
    if absorb_flux:
        F[0] = 0.0
    use_bdf2 = (str(scheme).lower() == "bdf2")
    for n in range(nsteps):
        lo1, di1, up1, BmN = build(rho_eta * ((n + 1) * dt) - rem)
        if use_bdf2 and n >= nstart:
            # (1.5 I - dt A) p^{n+1} = 2 p^n - 0.5 p^{n-1}
            ab = np.zeros((3, M))
            ab[0, 1:] = -dt * up1[:-1]; ab[1, :] = 1.5 - dt * di1; ab[2, :-1] = -dt * lo1[1:]
            rhs = 2.0 * p - 0.5 * p_prev
        else:
            ab = np.zeros((3, M))
            ab[0, 1:] = -dt * up1[:-1]; ab[1, :] = 1.0 - dt * di1; ab[2, :-1] = -dt * lo1[1:]
            rhs = p
        p_new = solve_banded((1, 1), ab, rhs)
        p_new[p_new < 0] = 0.0
        p_prev = p; p = p_new
        Sv = p.sum() * h; S[n + 1] = Sv; tl[n + 1] = (n + 1) * dt; last = n + 1
        if absorb_flux:
            F[n + 1] = (re / h) * BmN * p[M - 1]
        if Sv < Gtol and (n + 1) * dt > 0.3:
            break
    S = S[:last + 1]; tl = tl[:last + 1]
    mean = _trapz(S, tl); m2 = 2 * _trapz(tl * S, tl)
    cv = np.sqrt(max(m2 - mean ** 2, 0.0)) / mean
    if absorb_flux:
        F = F[:last + 1]
        if return_survival:
            return mean, cv, S[-1], tl, S, F
        return mean, cv, S[-1], F
    if return_survival:
        return mean, cv, S[-1], tl, S
    return mean, cv, S[-1]


# ---------------------------------------------------------------------------
# dimensionless conversion + Kaplan-Meier (vendored from fp_curvefit_seed; same convention:
# to_dimensionless returns (rho_beta, rho_eta, rho_eps, kappa) -- beta first).
# ---------------------------------------------------------------------------
def to_dimensionless(eta, beta, epsilon, xc, L, k=None):
    rb = L * beta / xc
    rh = L ** 2 * eta / xc
    re = L * epsilon / xc ** 2
    if k is None:
        return rb, rh, re
    return rb, rh, re, k / xc


def kaplan_meier(death_times, events=None):
    """(t_steps, S_steps, mean) with mean = area under the KM curve (the lifetime scale L)."""
    t = np.asarray(death_times, dtype=float)
    if t.size == 0:
        raise ValueError("no death times supplied")
    e = np.ones(t.size, dtype=int) if events is None else np.asarray(events, dtype=int)
    order = np.argsort(t); t, e = t[order], e[order]
    n = t.size; S = 1.0; at_risk = n; tk = [0.0]; Sk = [1.0]
    for i in range(n):
        if e[i] == 1 and at_risk > 0:
            S *= (1.0 - 1.0 / at_risk)
        tk.append(t[i]); Sk.append(S); at_risk -= 1
    tk = np.asarray(tk); Sk = np.asarray(Sk)
    return tk, Sk, _trapz(Sk, tk)


def kaplan_meier_mean(death_times, events=None):
    return kaplan_meier(death_times, events)[2]


# ---------------------------------------------------------------------------
# density + likelihood
# Production grids (T2 saturation study): FAST N=900,dt=1e-3,tau_max=20 (~60ms, defaults);
# ACCURATE N=1200,dt=5e-4 (~150ms). dt dominates; N>=600 fine; tau_max free (early-stop at Gtol).
# ---------------------------------------------------------------------------
def fpt_full_density(rho_eta, rho_beta, rho_eps, kappa, N=900, dt=1e-3, tau_max=20.0, Gtol=1e-11,
                     scheme="be", nstart=20, absorb_flux=False):
    """(tau, S, p, mean, cv, s_end) with p(tau)=-dS/dtau>=0; S forced monotone (removes p<0 jitter).

    absorb_flux=True takes p from the exact discrete absorbing-boundary flux instead of
    differentiating S -- non-negative by construction and normalized to 1-S_end without the
    monotonize-then-gradient repair. Not more accurate; see fpt_full's docstring.
    scheme='bdf2' is second-order in tau (~4x faster and ~10x more accurate on stiff cohorts).
    """
    kw = dict(N=N, dt=dt, tau_max=tau_max, Gtol=Gtol, scheme=scheme, nstart=nstart)
    if absorb_flux:
        mean, cv, s_end, tau, S, F = fpt_full(rho_eta, rho_beta, rho_eps, kappa,
                                              return_survival=True, absorb_flux=True, **kw)
        tau = np.asarray(tau, float)
        S = np.clip(np.minimum.accumulate(np.asarray(S, float)), 0.0, 1.0)
        return tau, S, np.clip(np.asarray(F, float), 0.0, None), mean, cv, s_end
    mean, cv, s_end, tau, S = fpt_full(rho_eta, rho_beta, rho_eps, kappa,
                                       return_survival=True, **kw)
    S = np.clip(np.minimum.accumulate(np.asarray(S, float)), 0.0, 1.0)
    tau = np.asarray(tau, float)
    p = np.clip(-np.gradient(S, tau), 0.0, None)
    return tau, S, p, mean, cv, s_end


def auto_grid(death_times, L, tau_max_cap=40.0, tau_min=2.0, dt_floor=2e-3, target_steps=1200):
    """(tau_max, dt) sized to the DATA instead of a fixed horizon.

    tau_max = clip(1.15 * max(t)/L, tau_min, tau_max_cap);  dt = max(dt_floor, tau_max/target_steps)

    Measured against the fixed (tau_max=20, dt=1e-3) default on a 3000-individual cohort: 42 ms vs
    98 ms per likelihood evaluation (2.3x) for a log-likelihood difference of 0.009 nats out of 79.
    The saving multiplies by M for the heterogeneity mixture, which runs one solve per node.

    There is no accuracy cost because the solver never needs to integrate past the last observation:
    S beyond max(t) enters the likelihood only through censored individuals, all of whom are at or
    before it. The 15% margin covers interpolation at the final point.
    """
    t = np.asarray(death_times, float)
    tm = float(min(tau_max_cap, max(tau_min, 1.15 * float(np.max(t)) / float(L))))
    dt = max(dt_floor, tm / float(target_steps))
    # OPTIONAL ABSOLUTE CEILING on dt, off by default (SRK_DT_CEILING unset => behaviour unchanged).
    #
    # WHY IT EXISTS. `target_steps` is a fixed step COUNT, so dt scales with tau_max -- and tau_max grows
    # with the spread of the observed ages. A BROAD cohort therefore gets a COARSER step, which is exactly
    # backwards: accuracy is set by the dynamics, not by how long the tail is.
    #
    # MEASURED (2026-08-20), worst |dlogL| against a dt=5e-4 reference INSIDE the posterior support, on
    # the four noise-dominated probes (observed cv 0.72-0.82, tau_max 4.7-6.4):
    #
    #     dt      5e-3      4e-3      2e-3      1e-3
    #     err   1.44-1.66  0.97-1.24  0.24-0.36  0.048-0.072   nats
    #     %edge   75-87%     51-64%     12-19%      2.5-3.8%
    #
    # chi2_1(0.95)/2 = 1.921 nats sets an interval edge, so at the dt those cells actually receive
    # (4.1e-3 to 6.9e-3) the SOLVER error is a large fraction of the statistical threshold. 7 of 111
    # parameter points in the benchmark library are above 4e-3, all four noise probes among them.
    #
    # Left opt-in rather than made the default deliberately: flipping it silently would make already-
    # computed runs incomparable with new ones, and choosing the right rule (an absolute cap vs a
    # cv-adaptive one) is a solver-design decision that deserves its own measurement, not a side effect.
    _ceil = _os.environ.get("SRK_DT_CEILING")
    if _ceil:
        dt = min(dt, float(_ceil))
    return tm, dt


def _interp_S_p(tau_eval, tau, S, p, tail_extrap=False, n_fit=12):
    """Interpolate survival and density onto `tau_eval`.

    DEFAULT (`tail_extrap=False`) is unchanged and bit-identical: density is 0 beyond the last computed
    tau and survival is held flat at S[-1].

    WHY AN ALTERNATIVE EXISTS. `fpt_full` early-stops once survival drops below `Gtol` (1e-11), and the
    default `right=0.0` then makes the density of any death past that point EXACTLY zero -- so
    `log p` becomes `log 0`, which `fp_loglik` clips to the constant `_LOG_FLOOR = -700`. A constant has
    no derivative, so such an individual is INVISIBLE to any gradient- or simplex-based fit. Measured on
    the leverage self-test cohort: `dLL/dlog eps` differs by only **0.06%** with vs without a planted
    high-leverage death (14.773 vs 14.781) at the survival-RMS seed, then jumps to **+17.5 nats/dex**
    once epsilon crosses 3.2-3.3x and the term un-floors -- a ~2000x change in signal across a 10% move
    in one parameter. That is why the documented leverage protocol needs a high-epsilon start or MCMC.
    Interval censoring does NOT fix it, because `_interval_mass` reads the same truncated S.

    `tail_extrap=True` instead continues log(p) and log(S) LINEARLY in tau past the last grid point,
    with the slope estimated from the final `n_fit` points. In the far tail the FP density decays close
    to exponentially, so log-linear is the natural continuation and it restores a finite, differentiable
    value where the truncation would otherwise have produced a hard zero. Opt-in, because it changes the
    likelihood far from the mode and every stored result was produced without it.
    """
    if not tail_extrap:
        return (np.interp(tau_eval, tau, S, left=1.0, right=S[-1]),
                np.interp(tau_eval, tau, p, left=0.0, right=0.0))

    Si = np.interp(tau_eval, tau, S, left=1.0, right=S[-1])
    pi = np.interp(tau_eval, tau, p, left=0.0, right=0.0)
    beyond = np.asarray(tau_eval) > tau[-1]
    if not np.any(beyond) or tau.size < 3:
        return Si, pi
    k = int(min(max(n_fit, 3), tau.size))
    tt = tau[-k:]
    out = []
    for arr, last_val in ((p, p[-1]), (S, S[-1])):
        yy = arr[-k:]
        ok = yy > 0
        if ok.sum() >= 2:
            sl = np.polyfit(tt[ok], np.log(yy[ok]), 1)[0]
            sl = min(sl, 0.0)                       # the tail must decay, never grow
            base = float(np.log(max(last_val, _P_FLOOR)))
            ext = np.exp(np.clip(base + sl * (np.asarray(tau_eval) - tau[-1]),
                                 _LOG_FLOOR, 0.0))
        else:
            ext = np.full(np.shape(tau_eval), max(last_val, _P_FLOOR))
        out.append(ext)
    p_ext, S_ext = out
    pi = np.where(beyond, np.maximum(p_ext, _P_FLOOR), pi)
    Si = np.where(beyond, np.minimum(np.maximum(S_ext, _P_FLOOR), Si), Si)
    return Si, pi


def _norm_sampling_dt(sampling_dt, n):
    """None | scalar | per-observation array  ->  None or a float array of shape (n,).

    A PER-OBSERVATION width is needed when the recording interval is not constant. The usual cause
    is a scoring round that did not happen: the deaths from the missed day are counted on the next
    day scored, so that entry covers two days while its neighbours cover one. Feeding such data a
    single scalar delta mis-states every pooled entry, and the surplus deaths read as a burst --
    which a two-component mixture is flexible enough to buy as a second sub-population.

    A scalar broadcasts, so the scalar path stays bit-for-bit what it was. Non-positive scalars map
    to None, preserving the old `if sampling_dt:` falsy behaviour for 0.
    """
    if sampling_dt is None:
        return None
    d = np.asarray(sampling_dt, float)
    if d.ndim == 0:
        return None if not (d > 0) else np.full(n, float(d))
    if d.shape != (n,):
        raise ValueError(f"sampling_dt must be a scalar or an array of shape ({n},); got {d.shape}")
    if not np.all(np.isfinite(d)) or not np.all(d > 0):
        raise ValueError("every per-observation sampling_dt must be finite and > 0")
    return d


def _interval_mass(tau_hi_eval, tau, S, S_phys_lo, h, t, sampling_dt):
    """P(t <= T < t+delta) on the PHYSICAL, Makeham-adjusted survival.

    Deaths recorded at resolution `delta` did not occur AT t, they occurred somewhere in
    [t, t+delta). The exact contribution is S(t) - S(t+delta); the density form log(p(t)*delta) is
    its first-order approximation. See the `sampling_dt` note in fp_loglik.
    """
    S_hi = np.interp(tau_hi_eval, tau, S, left=1.0, right=S[-1])
    if h > 0.0:
        S_hi = S_hi * np.exp(-h * (t + sampling_dt))
    return np.clip(S_phys_lo - S_hi, 0.0, None)


def _windowed_ll(t, e, S_phys, p_phys, t_lo, t_hi, S_lo, S_hi):
    """Left-truncated (+ right-censored) FP log-likelihood on the fit window [t_lo, t_hi].

    `p_phys` supplies the DEATH term. It is the physical density when scoring exact times, or the
    interval mass P(t <= T < t+delta) when `sampling_dt` is in use -- the assembly is identical
    either way, only the meaning of the death weight changes.

    Absolute age -- NO origin shift (the SR drift is age-dependent, and this matches the
    Langevin windowing in distance_metrics). Every individual reaching t_lo is conditioned on
    S_lo=S(t_lo); individuals dying before t_lo are dropped; anyone surviving past t_hi is
    right-censored at t_hi (S_hi=S(t_hi)). S_phys/p_phys are the physical survival/density
    (already Makeham-adjusted) at the per-individual times t; S_lo/S_hi are scalars on the
    same (Makeham-adjusted) curve.
    """
    died = e == 1
    in_risk = t >= t_lo
    if not in_risk.any():
        return -np.inf
    in_win = in_risk & (t <= t_hi)
    dmask = in_win & died
    cmask = in_win & (~died)
    ll = np.maximum(np.log(np.clip(p_phys[dmask], _P_FLOOR, None)), _LOG_FLOOR).sum()
    if cmask.any():
        ll += np.maximum(np.log(np.clip(S_phys[cmask], _P_FLOOR, None)), _LOG_FLOOR).sum()
    n_beyond = int((in_risk & (t > t_hi)).sum())          # survived the window -> censored at t_hi
    if n_beyond:
        ll += n_beyond * max(float(np.log(max(S_hi, 1e-300))), _LOG_FLOOR)
    # condition on reaching t_lo: divide every at-risk individual by S_lo
    ll -= int(in_risk.sum()) * float(np.log(min(max(S_lo, 1e-12), 1.0)))
    return float(ll)


def fp_loglik(death_times, events, rho_eta, rho_beta, rho_eps, kappa, L,
              external_hazard=np.inf, time_range=None,
              N=900, dt=1e-3, tau_max=20.0, Gtol=1e-11, return_parts=False,
              sampling_dt=None, scheme="be", absorb_flux=False, tail_extrap=False):
    """Exact FP log-likelihood from dimensionless params. -inf for invalid params.

    time_range=(t_lo, t_hi): restrict to the aging window via absolute-age left-truncation
    (condition on S(t_lo), drop t<t_lo, right-censor survivors past t_hi). None => full data.

    sampling_dt : recorded-age RESOLUTION (same units as death_times). None (default) keeps the
        existing exact-time behaviour bit-for-bit. When set, each death scores by the probability of
        dying INSIDE its recorded interval,

            log P(t <= T < t+delta) = log( S(t) - S(t+delta) )

        rather than by the density log p(t). Our data is recorded to the day (mosquitoes, worms),
        the division (yeast RLS) or the year (much of the zoo/human data), so this is the correct
        likelihood, and it also handles TIES properly -- at day resolution many individuals share a
        recorded time, which the density form treats as exact coincidences.

        Scope, measured rather than assumed: the two forms differ by the constant n*log(delta),
        which CANCELS in the argmax, so the effect on fitted parameters is second order in delta.
        It matters where delta/L is not small -- yeast RLS ~4%, mosquitoes ~4%, short-lived species
        at year resolution 20-50% -- and is negligible for humans (~1.4%). It does NOT damp
        single-individual leverage: with a planted isolated death at 6 cohort SDs both forms bottom
        out at the same floor, so keep the leverage refit for that. It DOES make log-likelihoods
        comparable across cohorts of differing resolution, which the density form does not, and
        which the cross-cohort AIC and the SR-vs-Gompertz-Makeham comparison need.
    """
    t = np.asarray(death_times, float)
    e = np.ones(t.size, int) if events is None else np.asarray(events, int)
    if not (np.isfinite([rho_eta, rho_beta, rho_eps, kappa, L]).all() and L > 0
            and rho_eps > 0 and kappa > 0):
        return -np.inf
    tau, S, p, mean, cv, s_end = fpt_full_density(rho_eta, rho_beta, rho_eps, kappa,
                                                  N=N, dt=dt, tau_max=tau_max, Gtol=Gtol,
                                                  scheme=scheme, absorb_flux=absorb_flux)
    S_i, p_i = _interp_S_p(t / L, tau, S, p, tail_extrap=tail_extrap)
    S_phys = S_i
    p_phys = p_i / L
    h = 0.0 if not np.isfinite(external_hazard) else float(np.exp(-external_hazard))
    if h > 0.0:
        emht = np.exp(-h * t)
        p_phys = (p_phys + h * S_phys) * emht
        S_phys = S_phys * emht
    _sdt = _norm_sampling_dt(sampling_dt, t.size)
    if _sdt is not None:
        # interval-censored death weight; censored individuals keep log S(t) unchanged
        p_phys = _interval_mass((t + _sdt) / L, tau, S, S_phys, h, t, _sdt)
    if time_range is None:
        died = e == 1
        ll_death = np.maximum(np.log(np.clip(p_phys[died], _P_FLOOR, None)), _LOG_FLOOR).sum()
        ll_cens = np.maximum(np.log(np.clip(S_phys[~died], _P_FLOOR, None)), _LOG_FLOOR).sum() \
            if (~died).any() else 0.0
        ll = float(ll_death + ll_cens)
    else:
        t_lo, t_hi = float(time_range[0]), float(time_range[1])
        edge_S, _ = _interp_S_p(np.array([t_lo, t_hi]) / L, tau, S, p,
                                tail_extrap=tail_extrap)
        S_lo, S_hi = float(edge_S[0]), float(edge_S[1])
        if h > 0.0:
            S_lo *= float(np.exp(-h * t_lo))
            S_hi *= float(np.exp(-h * t_hi))
        ll = _windowed_ll(t, e, S_phys, p_phys, t_lo, t_hi, S_lo, S_hi)
        ll_death = ll_cens = float("nan")
    if not np.isfinite(ll):
        return -np.inf
    if return_parts:
        return dict(loglik=ll, ll_death=float(ll_death), ll_cens=float(ll_cens),
                    fp_mean=mean, fp_cv=cv, s_end=s_end, n=t.size, n_death=int((e == 1).sum()))
    return ll


def fp_loglik_physical(death_times, events, eta, beta, epsilon, xc, kappa_phys,
                       L=None, external_hazard=np.inf, **kw):
    """fp_loglik from PHYSICAL params. L defaults to the data mean lifetime.

    Accepts every fp_loglik kwarg, including sampling_dt (interval censoring) and scheme='bdf2'.
    """
    if L is None:
        _, _, L = kaplan_meier(death_times, events)
    rho_beta, rho_eta, rho_eps, kappa = to_dimensionless(eta, beta, epsilon, xc, L, k=kappa_phys)
    return fp_loglik(death_times, events, rho_eta, rho_beta, rho_eps, kappa, L,
                     external_hazard=external_hazard, **kw)


# ---------------------------------------------------------------------------
# population HETEROGENEITY: mixture over ONE parameter (pipeline uses xc, xc_var=0.2).
# S_bar = E_P[ S(t;P) ] via Gauss-Hermite quadrature; cost ~ M FP solves.
# ---------------------------------------------------------------------------
def _gh_nodes(center, var, M):
    z, w = np.polynomial.hermite_e.hermegauss(M)     # weight exp(-z^2/2)
    w = w / np.sqrt(2 * np.pi)
    vals = center * (1.0 + var * z)
    ok = vals > 0
    vals, w = vals[ok], w[ok]
    return vals, w / w.sum()


def fp_mixture_curves_physical(eta, beta, epsilon, xc, kappa_phys, L, tau_eval,
                               het_param="xc", het_var=None, M=7, xc_var=None, **solver_kw):
    """(S_bar, p_bar) mixed over het_param in {eta,beta,epsilon,xc,kappa} ~ P*(1+het_var*z).

    het_var<=0 or M<=1 => homogeneous. xc_var=<v> is an alias for het_param='xc', het_var=<v>.
    """
    if xc_var is not None:
        het_param, het_var = "xc", xc_var
    het_var = 0.0 if het_var is None else float(het_var)
    if het_param not in _HET_PARAMS:
        raise ValueError(f"het_param must be one of {_HET_PARAMS}, got {het_param!r}")
    tau_eval = np.asarray(tau_eval, float)
    base = dict(eta=eta, beta=beta, epsilon=epsilon, xc=xc, kappa=kappa_phys)
    if het_var <= 0 or M <= 1:
        vals, ws = np.array([base[het_param]]), np.array([1.0])
    else:
        vals, ws = _gh_nodes(base[het_param], het_var, M)
    S_bar = np.zeros_like(tau_eval); p_bar = np.zeros_like(tau_eval)
    for val, wi in zip(vals, ws):
        pp = dict(base); pp[het_param] = val
        rb, rh, re, kap = to_dimensionless(pp["eta"], pp["beta"], pp["epsilon"], pp["xc"], L, k=pp["kappa"])
        tau, S, p, *_ = fpt_full_density(rh, rb, re, kap, **solver_kw)
        S_bar += wi * np.interp(tau_eval, tau, S, left=1.0, right=S[-1])
        p_bar += wi * np.interp(tau_eval, tau, p, left=0.0, right=0.0)
    return S_bar, p_bar


def fp_loglik_het_physical(death_times, events, eta, beta, epsilon, xc, kappa_phys,
                           het_param="xc", het_var=None, M=7, L=None, external_hazard=np.inf,
                           time_range=None, xc_var=None, sampling_dt=None, **solver_kw):
    """Heterogeneous FP log-likelihood (mixture over het_param). het_var=0 => fp_loglik_physical.

    time_range=(t_lo, t_hi): same absolute-age left-truncation as fp_loglik.
    sampling_dt: interval censoring, as in fp_loglik. Because the mixture averages S LINEARLY,
        E_P[S(t) - S(t+delta)] = S_bar(t) - S_bar(t+delta), so the interval mass is exact here too
        -- it just needs S_bar evaluated at the shifted times as well.
    """
    if xc_var is not None:
        het_param, het_var = "xc", xc_var
    het_var = 0.2 if het_var is None else float(het_var)
    t = np.asarray(death_times, float)
    e = np.ones(t.size, int) if events is None else np.asarray(events, int)
    # validity guard, matching fp_loglik. The het path had none, so an invalid walker reached the
    # solver instead of being rejected.
    if not (np.isfinite([eta, beta, epsilon, xc, kappa_phys]).all()
            and epsilon > 0 and kappa_phys > 0 and xc > 0):
        return -np.inf
    if L is None:
        _, _, L = kaplan_meier(t, e)
    if not (np.isfinite(L) and L > 0):
        return -np.inf
    tau_eval = t / L
    n_extra = 0
    _sdt = _norm_sampling_dt(sampling_dt, t.size)
    if _sdt is not None:                             # shifted times for the interval mass
        tau_eval = np.concatenate([tau_eval, (t + _sdt) / L])
        n_extra += t.size
    if time_range is not None:                       # evaluate the mixture at the window edges too
        tau_eval = np.concatenate([tau_eval, np.array([float(time_range[0]),
                                                        float(time_range[1])]) / L])
        n_extra += 2
    S_bar, p_bar = fp_mixture_curves_physical(eta, beta, epsilon, xc, kappa_phys, L, tau_eval,
                                              het_param=het_param, het_var=het_var, M=M, **solver_kw)
    S_lo = S_hi = None
    if time_range is not None:
        S_lo, S_hi = float(S_bar[-2]), float(S_bar[-1])
        S_bar, p_bar = S_bar[:-2], p_bar[:-2]
    S_shift = None
    if _sdt is not None:
        S_shift = S_bar[t.size:t.size * 2]
        S_bar, p_bar = S_bar[:t.size], p_bar[:t.size]
    p_phys = p_bar / L
    h = 0.0 if not np.isfinite(external_hazard) else float(np.exp(-external_hazard))
    if h > 0.0:
        emht = np.exp(-h * t)
        p_phys = (p_phys + h * S_bar) * emht
        S_bar = S_bar * emht
        if S_shift is not None:
            S_shift = S_shift * np.exp(-h * (t + _sdt))
        if time_range is not None:
            S_lo *= float(np.exp(-h * float(time_range[0])))
            S_hi *= float(np.exp(-h * float(time_range[1])))
    if S_shift is not None:
        p_phys = np.clip(S_bar - S_shift, 0.0, None)
    if time_range is None:
        died = e == 1
        ll = np.maximum(np.log(np.clip(p_phys[died], _P_FLOOR, None)), _LOG_FLOOR).sum()
        if (~died).any():
            ll += np.maximum(np.log(np.clip(S_bar[~died], _P_FLOOR, None)), _LOG_FLOOR).sum()
    else:
        ll = _windowed_ll(t, e, S_bar, p_phys, float(time_range[0]), float(time_range[1]), S_lo, S_hi)
    return float(ll) if np.isfinite(ll) else -np.inf


# ---------------------------------------------------------------------------
# heuristic auto-M: spread proxy (var x stiffness) -> M, +30% margin, cap.
# Calibrated on the T6a xc sweep (94% coverage). M = ceil(1.3 * interp(spread)), floor 3, cap 31.
# ---------------------------------------------------------------------------
_S_KNOTS = (0.30, 0.50, 0.65, 0.80, 0.90, 0.97, 1.00)
_M_KNOTS = (3.0,  3.0,  5.0,  9.0,  12.0, 20.0, 25.0)


def _spread_proxy(eta, beta, epsilon, xc, kappa, L, het_param, het_var, N=900, dt=1e-3, tau_max=20.0):
    """KS between the survival at het_param*(1-var) and *(1+var) -- captures var x stiffness."""
    base = dict(eta=eta, beta=beta, epsilon=epsilon, xc=xc, kappa=kappa)
    taus = np.linspace(1e-3, 4.0, 400)

    def surv(scale):
        pp = dict(base); pp[het_param] = base[het_param] * scale
        rb, rh, re, kap = to_dimensionless(pp["eta"], pp["beta"], pp["epsilon"], pp["xc"], L, k=pp["kappa"])
        _t, S, *_ = fpt_full_density(rh, rb, re, kap, N=N, dt=dt, tau_max=tau_max)
        return np.interp(taus, _t, S, left=1.0, right=S[-1])
    return float(np.max(np.abs(surv(1.0 - het_var) - surv(1.0 + het_var))))


def choose_M(spread, margin=1.30, cap=31, floor=3):
    """Heuristic node count from the spread proxy, with a +30% safety margin (per user)."""
    raw = float(np.interp(spread, _S_KNOTS, _M_KNOTS))
    return int(min(cap, max(floor, np.ceil(margin * raw))))
