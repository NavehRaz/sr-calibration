"""Absolute goodness-of-fit, model comparison, and identifiability for the SR FP pipeline.

The FP likelihood tells you which of two SR fits is better. It cannot tell you whether SR
describes a cohort at all, nor whether it beats the standard demographic model. This module adds
the three things that answer those questions, plus the profile machinery that separates a genuinely
flat parameter direction from an under-resolved one.

    saturated_loglik / deviance   ABSOLUTE fit quality: how much likelihood the SR fit gives up
                                  relative to the best any model could do on this cohort (the KM
                                  curve itself). G2 = 0 means perfect.
    gof_bootstrap_ks              a p-value: simulate from OUR OWN fit, censoring at the same
                                  per-individual horizons as the data, and ask how often the
                                  simulated KM-vs-model gap exceeds the observed one.
    gm_mle                        Gompertz-Makeham (h = alpha*exp(beta*t) + lambda), 3 params,
                                  fitted with the SAME censored likelihood so AIC is comparable to
                                  the 4-param SR fit. Gives a per-cohort "is SR worth its extra
                                  parameter" number.
    ext_death_frac                what share of deaths the fitted background hazard accounts for.
    profile_likelihood            fix one parameter on a grid, re-maximize the rest, at TWO solver
                                  resolutions. Flat in both => real degeneracy. Flat in one =>
                                  under-resolution artifact.

A NOTE ON THE DEVIANCE, because it is easy to get wrong
------------------------------------------------------
For continuous data the saturated likelihood is unbounded (a density can be made arbitrarily
large), so `2*(ll_sat - fp_loglik)` would be meaningless. The deviance here is therefore computed
in the DISCRETE-HAZARD parameterization on both sides:

    ll_sat   = sum_j [ d_j log(d_j/Y_j) + (Y_j-d_j) log(1 - d_j/Y_j) ]      (KM is its MLE)
    ll_model = sum_j [ d_j log(h_j)     + (Y_j-d_j) log(1 - h_j)     ]      h_j from the SR fit

with d_j deaths and Y_j at risk at the j-th distinct observed death time. Both sides are then
probabilities of the same observed events, so G2 = 2*(ll_sat - ll_model) >= 0 is a real deviance.
`model_discrete_loglik` is consequently NOT equal to `fp_loglik` and must not be compared to it.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

try:
    from . import fp_likelihood as _fpl
except ImportError:  # pragma: no cover - standalone / shim load path
    import importlib.util as _ilu
    import os as _os
    _spec = _ilu.spec_from_file_location(
        "srk_fp_likelihood",
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "fp_likelihood.py"))
    _fpl = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_fpl)

fpt_full_density = _fpl.fpt_full_density
fp_loglik_physical = _fpl.fp_loglik_physical
to_dimensionless = _fpl.to_dimensionless
kaplan_meier = _fpl.kaplan_meier

__all__ = ["risk_table", "saturated_loglik", "model_discrete_loglik", "deviance",
           "gof_bootstrap_ks", "gm_mle", "ext_death_frac", "profile_likelihood"]

_EPS = 1e-300
_LOG_FLOOR = -700.0


# ---------------------------------------------------------------------------
# risk table -- the shared basis of the deviance
# ---------------------------------------------------------------------------
def risk_table(death_times, events=None):
    """(t_j, d_j, Y_j) at each distinct OBSERVED DEATH time: deaths and number at risk.

    Censored individuals contribute to Y_j while they are at risk but never create a t_j.
    """
    t = np.asarray(death_times, float)
    e = np.ones(t.size, int) if events is None else np.asarray(events, int)
    tj = np.unique(t[e == 1])
    if tj.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    d = np.array([np.sum((t == x) & (e == 1)) for x in tj], float)
    Y = np.array([np.sum(t >= x) for x in tj], float)
    return tj, d, Y


def saturated_loglik(death_times, events=None):
    """Discrete-hazard NPMLE log-likelihood -- the best ANY model could do on this cohort.

    The Kaplan-Meier estimator is its maximizer (h_j = d_j/Y_j). Left-truncation cancels in the
    deviance difference, so it is not carried here.
    """
    tj, d, Y = risk_table(death_times, events)
    if tj.size == 0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.clip(d / np.maximum(Y, 1.0), 0.0, 1.0)
        a = np.where(d > 0, d * np.log(np.clip(p, _EPS, None)), 0.0)
        b = np.where(Y - d > 0, (Y - d) * np.log(np.clip(1.0 - p, _EPS, None)), 0.0)
    return float(np.sum(a + b))


def model_discrete_loglik(death_times, events, eta, beta, epsilon, xc, kappa,
                          L=None, external_hazard=np.inf, **solver_kw):
    """The SR fit's log-likelihood in the SAME discrete-hazard form as `saturated_loglik`.

    h_j = 1 - S(t_j)/S(t_{j-1}) from the fitted survival, i.e. the model's conditional probability
    of dying in the j-th observed interval. NOT comparable to fp_loglik (see the module docstring).
    """
    t = np.asarray(death_times, float)
    e = np.ones(t.size, int) if events is None else np.asarray(events, int)
    if L is None:
        _, _, L = kaplan_meier(t, e)
    tj, d, Y = risk_table(t, e)
    if tj.size == 0:
        return 0.0
    rb, rh, re_, kap = to_dimensionless(eta, beta, epsilon, xc, L, k=kappa)
    tau, S, _p, _m, _cv, _se = fpt_full_density(rh, rb, re_, kap, **solver_kw)
    edges = np.concatenate([[0.0], tj])
    Se = np.interp(edges / L, tau, S, left=1.0, right=S[-1])
    h = 0.0 if not np.isfinite(external_hazard) else float(np.exp(-external_hazard))
    if h > 0:
        Se = Se * np.exp(-h * edges)
    Se = np.clip(Se, _EPS, 1.0)
    hj = np.clip(1.0 - Se[1:] / Se[:-1], 1e-12, 1.0 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(d > 0, d * np.log(hj), 0.0)
        b = np.where(Y - d > 0, (Y - d) * np.log(1.0 - hj), 0.0)
    val = float(np.sum(a + b))
    return val if np.isfinite(val) else -np.inf


def deviance(death_times, events, eta, beta, epsilon, xc, kappa, L=None,
             external_hazard=np.inf, n_params=4, bin_dt=None, **solver_kw):
    """G2 = 2*(ll_saturated - ll_model) >= 0, plus a per-individual version and AIC.

    *** bin_dt IS EFFECTIVELY REQUIRED. Read this before using g2. ***

    G2 is only an absolute measure of fit quality when the data has TIES. On tie-free continuous
    death times the saturated model has one free hazard per observation (d_j = 1, Y_j = n-j+1), so
    it fits every death exactly and G2 is dominated by its own overfitting rather than by the SR
    model's inadequacy. Measured on 1500 simulated deaths, evaluated AT THE TRUE PARAMETERS:

        bin_dt      mean deaths/bin    G2/n at truth    ratio G2(eps x2)/G2(truth)
        none                   1.00            1.171                         1.59
        0.010 * L             13.51            0.064                        11.49
        0.020 * L             25.00            0.034                        19.94
        0.050 * L             55.56            0.030                        19.97
        0.100 * L            107.14            0.069                         7.30

    So binning at ~2-5% of the mean lifetime drops G2/n at the truth by ~34x AND improves the
    discrimination between a right and a wrong model by ~12x. Too coarse (>= 10%) washes the signal
    out again. Use the data's own recording resolution (day, division, year) -- the same value that
    `fp_loglik(..., sampling_dt=...)` wants, which is not a coincidence: both need the data treated
    as the discrete observations it actually is.

    With bin_dt=None the result carries `tie_free=True` and `g2_interpretable=False`; do not quote
    g2 as an absolute fit statistic in that case. Model ORDERING is still usable (G2 was minimal at
    the truth in every case above), just not the magnitude.
    """
    t = np.asarray(death_times, float)
    e = np.ones(t.size, int) if events is None else np.asarray(events, int)
    if bin_dt:
        # bin to the recording resolution, then place each death at the bin CENTRE so the model is
        # evaluated mid-interval rather than at an edge
        b = float(bin_dt)
        t = np.floor(t / b) * b + 0.5 * b
    ll_sat = saturated_loglik(t, e)
    ll_mod = model_discrete_loglik(t, e, eta, beta, epsilon, xc, kappa,
                                   L=L, external_hazard=external_hazard, **solver_kw)
    n = int(t.size)
    tj, d, _Y = risk_table(t, e)
    mean_d = float(d.mean()) if d.size else float("nan")
    tie_free = bool(d.size and mean_d < 1.5)
    g2 = 2.0 * (ll_sat - ll_mod)
    return dict(g2=g2, g2_per_ind=(g2 / n if n else float("nan")),
                ll_saturated=ll_sat, ll_model_discrete=ll_mod, n=n,
                n_distinct=int(tj.size), mean_deaths_per_bin=mean_d,
                tie_free=tie_free, g2_interpretable=(not tie_free), bin_dt=bin_dt,
                aic_discrete=2.0 * n_params - 2.0 * ll_mod)


# ---------------------------------------------------------------------------
# parametric-bootstrap KS goodness of fit
# ---------------------------------------------------------------------------
def gof_bootstrap_ks(death_times, events, eta, beta, epsilon, xc, kappa, L=None,
                     external_hazard=np.inf, K=200, seed=0, **solver_kw):
    """p-value for "does the SR fit describe this cohort", by parametric bootstrap on the KS gap.

    Simulates K cohorts from the FITTED survival by inverse-CDF, right-censoring each replicate at
    the SAME per-individual horizons as the real data (so an informative censoring pattern is not
    silently discarded), and counts how often the replicate's KM-vs-model sup-gap is at least as
    large as the observed one.

    p < 0.05 => the model is rejected for this cohort. K=200 gives 0.005 p-resolution.

    MUST be calibration-checked before use: on data simulated FROM the model these p-values have to
    come out ~Uniform(0,1). A miscalibrated p-value cannot reject anything. See
    `SRK.benchmark.selftest` / verification step 9.
    """
    t = np.asarray(death_times, float)
    e = np.ones(t.size, int) if events is None else np.asarray(events, int)
    n = t.size
    if L is None:
        _, _, L = kaplan_meier(t, e)
    rb, rh, re_, kap = to_dimensionless(eta, beta, epsilon, xc, L, k=kappa)
    tau, S, _p, _m, _cv, _se = fpt_full_density(rh, rb, re_, kap, **solver_kw)
    h = 0.0 if not np.isfinite(external_hazard) else float(np.exp(-external_hazard))

    # model survival on a dense physical grid, Makeham-adjusted
    t_hi = max(float(t.max()) * 1.5, L * 3.0)
    gt = np.linspace(0.0, t_hi, 4000)
    gS = np.interp(gt / L, tau, S, left=1.0, right=S[-1])
    if h > 0:
        gS = gS * np.exp(-h * gt)
    gS = np.clip(gS, 0.0, 1.0)

    def ks(times, evs):
        tk, Sk, _ = kaplan_meier(times, evs)
        Sm = np.interp(tk, gt, gS, left=1.0, right=float(gS[-1]))
        return float(np.max(np.abs(Sk - Sm)))

    d_obs = ks(t, e)
    # censoring horizon per individual: observed time for censored, +inf for deaths
    cens_t = np.where(e == 0, t, np.inf)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(K):
        u = rng.random(n)
        td = np.interp(1.0 - u, gS[::-1], gt[::-1])
        ev_b = (td <= cens_t).astype(int)
        ab = np.minimum(td, cens_t)
        if not np.isfinite(ab).all():
            ab = np.where(np.isfinite(ab), ab, td)
        cnt += (ks(ab, ev_b) >= d_obs)
    return dict(gof_p=cnt / K, ks_d=d_obs, K=K)


# ---------------------------------------------------------------------------
# Gompertz-Makeham baseline
# ---------------------------------------------------------------------------
def gm_mle(death_times, events=None, t_trunc=0.0, restarts=3, seed=11):
    """Gompertz-Makeham MLE:  h(t) = alpha*exp(beta*t) + lambda.  3 parameters.

    Fitted with the same right-censored, left-truncated likelihood used for SR, so the two
    log-likelihoods are directly comparable and AIC (2k - 2ll) charges SR for its 4th parameter.
    This is the baseline any new aging model has to beat, and we currently have no such number.

    Returns alpha, beta, lam, loglik, aic, converged.
    """
    t = np.asarray(death_times, float)
    e = np.ones(t.size, int) if events is None else np.asarray(events, int)
    dead = e == 1
    L0 = float(np.median(t[dead])) if dead.any() else float(np.median(t))
    if not np.isfinite(L0) or L0 <= 0:
        L0 = float(np.mean(t))

    def nll(z):
        alpha, bta, lam = np.exp(np.clip(z, np.log(1e-14), np.log(1e6)))
        # cumulative hazard; expm1 keeps small beta*t accurate
        def cumH(x):
            return (alpha / bta) * np.expm1(bta * x) + lam * x
        with np.errstate(over="ignore", invalid="ignore"):
            hz = alpha * np.exp(bta * t) + lam
            ll = (np.log(np.clip(hz[dead], _EPS, None)).sum()
                  - cumH(t).sum() + t.size * cumH(np.float64(t_trunc)))
        return 1e18 if not np.isfinite(ll) else -ll

    x0 = np.log([1.0 / (L0 * math.e), 1.0 / L0, 1e-6 + 1.0 / (100 * L0)])
    lo = np.log([1e-12, 1e-6 / L0, 1e-12])
    hi = np.log([10.0, 50.0 / L0, 10.0])
    rng = np.random.default_rng(seed)
    starts = [x0] + [np.clip(x0 + rng.normal(0, 0.5, 3), lo, hi) for _ in range(restarts)]

    best, best_v, ok = x0, nll(x0), False
    for s in starts:
        try:
            r = minimize(nll, s, method="Nelder-Mead",
                         bounds=list(zip(lo, hi)),
                         options=dict(maxiter=600, maxfev=1200, xatol=1e-4, fatol=1e-4))
            if r.fun < best_v:
                best, best_v, ok = r.x, float(r.fun), bool(r.success)
        except Exception:
            continue
    alpha, bta, lam = np.exp(best)
    ll = -best_v
    return dict(alpha=float(alpha), beta=float(bta), lam=float(lam),
                loglik=float(ll), aic=float(2 * 3 - 2 * ll), n_params=3, converged=ok)


def ext_death_frac(death_times, events, eta, beta, epsilon, xc, kappa, L=None,
                   external_hazard=np.inf, cut=0.0, **solver_kw):
    """Share of deaths the fitted background (Makeham) hazard actually accounts for.

    h0 * integral(S dt) / (1 - S(end)) on the CONDITIONAL survival, so it does not over-read
    "external" on a genuinely aging curve (a large h0 with a steep S contributes few deaths).
    Returns nan when no external hazard is fitted.
    """
    if not np.isfinite(external_hazard):
        return float("nan")
    t = np.asarray(death_times, float)
    e = np.ones(t.size, int) if events is None else np.asarray(events, int)
    if L is None:
        _, _, L = kaplan_meier(t, e)
    h = float(np.exp(-external_hazard))
    if h <= 0:
        return 0.0
    rb, rh, re_, kap = to_dimensionless(eta, beta, epsilon, xc, L, k=kappa)
    tau, S, _p, _m, _cv, _se = fpt_full_density(rh, rb, re_, kap, **solver_kw)
    grid = np.linspace(cut, float(t.max()), 400)
    Sg = np.interp(grid / L, tau, S, left=1.0, right=S[-1]) * np.exp(-h * grid)
    S0 = max(Sg[0], _EPS)
    Sc = Sg / S0                                        # conditional on surviving to `cut`
    ext = h * _fpl._trapz(Sc, grid)      # SRK's own trapz: numpy>=2 removed np.trapz
    tot = 1.0 - float(Sc[-1])
    return float(ext / tot) if tot > 0 else float("nan")


# ---------------------------------------------------------------------------
# profile likelihood at TWO resolutions
# ---------------------------------------------------------------------------
_PARAM_IDX = {"eta": 0, "beta": 1, "epsilon": 2, "kappa": 3}

# (N, dt) pairs. Flat at BOTH => the direction is genuinely unidentified. Flat at the coarse grid
# only => the flatness was under-resolution. This is the test for the recorded gotcha that a kappa
# profile can read EXACTLY flat as an artifact rather than as non-identifiability.
GRIDS = {"coarse": dict(N=900, dt=1e-3, tau_max=20.0, Gtol=1e-11),
         "fine": dict(N=1800, dt=5e-4, tau_max=20.0, Gtol=1e-11)}


def profile_likelihood(death_times, events, theta0, which="kappa", grid_values=None,
                       xc=1.0, L=None, external_hazard=np.inf,
                       free=("eta", "beta", "epsilon"), resolutions=("coarse", "fine"),
                       maxiter=300, bounds=None, verbose=False):
    """Profile `which` over `grid_values`, re-maximizing `free` at each point, at each resolution.

    theta0 = [eta, beta, epsilon, kappa] (physical, xc fixed).

    Returns {resolution: [{value, loglik, theta, nfev}, ...]} plus a `flatness` summary giving the
    total log-likelihood span per resolution. Interpretation:

        span_coarse ~ 0 and span_fine ~ 0     -> genuinely flat: the parameter is unidentified
        span_coarse ~ 0 but span_fine > 0     -> the coarse grid was hiding real curvature
        both large                            -> the parameter is identified

    Reporting the two spans side by side is the whole point; a single flat profile is not evidence.
    """
    t = np.asarray(death_times, float)
    e = np.ones(t.size, int) if events is None else np.asarray(events, int)
    if L is None:
        _, _, L = kaplan_meier(t, e)
    if which not in _PARAM_IDX:
        raise ValueError(f"which must be one of {sorted(_PARAM_IDX)}")
    if grid_values is None:
        if which == "kappa":
            grid_values = np.geomspace(0.01, 1.0, 9)
        else:
            base = float(theta0[_PARAM_IDX[which]])
            grid_values = np.geomspace(base / 10.0, base * 10.0, 9)
    grid_values = np.asarray(grid_values, float)

    default_bounds = {"eta": (1e-12, 1e6), "beta": (1e-12, 1e6),
                      "epsilon": (1e-12, 1e6), "kappa": (1e-4, 10.0)}
    box = dict(default_bounds)
    box.update(bounds or {})
    free = tuple(f for f in free if f != which)
    fidx = [_PARAM_IDX[f] for f in free]

    out = {}
    for res in resolutions:
        kw = GRIDS[res]
        rows = []
        for v in grid_values:
            th = list(map(float, theta0))
            th[_PARAM_IDX[which]] = float(v)

            def nll(z):
                cur = list(th)
                for i, f in enumerate(free):
                    cur[_PARAM_IDX[f]] = math.exp(z[i])
                for name, idx in _PARAM_IDX.items():
                    lo, hi = box[name]
                    if not (lo <= cur[idx] <= hi):
                        return np.inf
                val = fp_loglik_physical(t, e, cur[0], cur[1], cur[2], xc, cur[3],
                                         L=L, external_hazard=external_hazard, **kw)
                return np.inf if not np.isfinite(val) else -val

            z0 = np.log([max(th[j], 1e-12) for j in fidx])
            try:
                r = minimize(nll, z0, method="Nelder-Mead",
                             options=dict(maxiter=maxiter, xatol=1e-4, fatol=1e-3))
                th_best = list(th)
                for f, zz in zip(free, r.x):
                    th_best[_PARAM_IDX[f]] = float(math.exp(zz))
                ll = -float(r.fun)
                nfev = int(r.nfev)
            except Exception:
                th_best, ll, nfev = th, float("-inf"), 0
            rows.append(dict(value=float(v), loglik=ll, theta=th_best, nfev=nfev))
            if verbose:
                print(f"  [{res}] {which}={v:.4g}  logL={ll:.3f}", flush=True)
        out[res] = rows

    flat = {}
    for res, rows in out.items():
        lls = np.array([r["loglik"] for r in rows], float)
        good = lls[np.isfinite(lls)]
        flat[res] = dict(span=float(good.max() - good.min()) if good.size else float("nan"),
                         argmax=(float(rows[int(np.nanargmax(lls))]["value"])
                                 if good.size else float("nan")))
        flat[res].update(profile_ci(rows, level=0.95))
    out["flatness"] = flat
    return out


# chi2_1 quantiles for the likelihood-ratio threshold on a 1-D profile: a value is inside the
# level-alpha CI when 2*(logL_max - logL(value)) <= chi2_1(alpha).
_CHI2_1 = {0.50: 0.4549, 0.68: 0.9891, 0.80: 1.6424, 0.90: 2.7055, 0.95: 3.8415, 0.99: 6.6349}


def profile_ci(rows, level=0.95):
    """Likelihood-ratio confidence interval from a 1-D profile, plus an identifiability verdict.

    THIS IS THE STEP THAT MAKES A PROFILE INTO A CI. The reference implementation this borrows from
    only ever read profile SHAPE qualitatively ("a minimum that rises on either side bounds the
    parameter; a flat profile signals an unbounded interval") and never applied a threshold, so it
    produced no intervals at all. The threshold is standard and cheap: a value is inside the
    level-alpha interval when

        2 * (logL_max - logL(value))  <=  chi2_1(alpha)          [3.8415 at 95%]

    Returns lo/hi (nan where the profile never crosses the threshold inside the grid) and a verdict:

        'identified'   crosses on BOTH sides -> a closed interval
        'one-sided:>'  crosses only above the max -> report a LOWER bound (e.g. "kappa > 0.05")
        'one-sided:<'  crosses only below -> report an UPPER bound
        'flat'         never crosses anywhere in the grid -> unidentified over this range

    A one-sided bound is usually the most that can honestly be said and is far more useful than
    "unidentified", which is why it is a first-class outcome rather than a failure.
    """
    thr = _CHI2_1.get(round(float(level), 2), _CHI2_1[0.95])
    v = np.array([r["value"] for r in rows], float)
    ll = np.array([r["loglik"] for r in rows], float)
    ok = np.isfinite(ll)
    if ok.sum() < 3:
        return dict(ci_lo=float("nan"), ci_hi=float("nan"), ci_level=level,
                    verdict="insufficient", ci_thr=thr)
    v, ll = v[ok], ll[ok]
    order = np.argsort(v); v, ll = v[order], ll[order]
    imax = int(np.argmax(ll))
    d = 2.0 * (ll[imax] - ll)          # deviance relative to the profile maximum, >= 0

    def cross(idx_range):
        """First crossing of `thr`, interpolated in sqrt(deviance) vs log(value).

        THE INTERPOLANT MATTERS ENORMOUSLY ON A COARSE GRID and getting it wrong manufactures
        false non-coverage. Near its maximum the profile is locally quadratic,
        d(v) ~ c*(log v - log v*)^2, so sqrt(d) is LINEAR in log v while d itself is not.

        Measured failure of the naive version (linear in d, linear in v): on corner_benign_n400 with
        5 nodes the deviance jumps 357.7 -> 0 between adjacent nodes, and linear-in-value
        interpolation placed the crossing at 3.186 against an MLE of 3.214 -- an interval of width
        0.03 that EXCLUDED the truth (4.375). The same cohort with 9 nodes gave [2.94, 11.52].
        With sqrt(d) vs log(v) the 5-node grid gives ~[2.7, 12.4], which contains the truth and is
        close to the 9-node answer, i.e. the estimate is no longer an artefact of grid spacing.
        """
        prev = None
        for i in idx_range:
            if prev is not None and (d[prev] - thr) * (d[i] - thr) <= 0 and d[i] != d[prev]:
                # interpolate on sqrt-deviance vs log-value; fall back to linear-in-value if a
                # value is non-positive (log undefined)
                if v[prev] > 0 and v[i] > 0:
                    sp_, si = math.sqrt(max(d[prev], 0.0)), math.sqrt(max(d[i], 0.0))
                    st = math.sqrt(thr)
                    if si != sp_:
                        f = (st - sp_) / (si - sp_)
                        return float(math.exp(math.log(v[prev])
                                              + f * (math.log(v[i]) - math.log(v[prev]))))
                f = (thr - d[prev]) / (d[i] - d[prev])
                return float(v[prev] + f * (v[i] - v[prev]))
            prev = i
        return float("nan")

    # THE INTERVAL IS THE WHOLE 1.92-NAT SET, NOT THE COMPONENT AROUND THE MAXIMUM.
    #
    # The likelihood-ratio region is { value : deviance(value) <= thr }. Nothing says that set is
    # connected, and for the SR model it frequently is NOT: measured on the six real cohorts in
    # `run7_inputs/vol/`, 20 of 72 one-dimensional profiles (28%) have a DISJOINT 95% set, and on 4
    # of 24 volumes the planted truth sits in a component other than the one containing the MLE.
    #
    # The earlier version walked outward from `imax` and returned the FIRST crossing on each side,
    # i.e. the connected component around the maximum. Where the profile is multimodal that reports a
    # strictly narrower interval than the definition, always in the same direction, so it depresses
    # coverage -- and it does so most on exactly the flat/multimodal parameters (beta, kappa) where
    # coverage matters most. The same bug was found and fixed independently in
    # `benchmark/kappa_value._interval`, where a yeast cohort's truth sat 0.0797 nats inside the
    # threshold but in the other component; this is that fix at the source.
    #
    # IS THE PROFILE REALLY MULTIMODAL, OR IS SOMETHING BROKEN? Tested, because the answer changes
    # what to do (`benchmark/dip_test.py`): re-solving a dipped node with a 16x larger optimiser
    # budget, warm-started from its better neighbour, recovers 0.0003 nats of 9.3; and refining the
    # solver grid 16x in dt and 3x in N moves the value by 0.016 nats of 41-47. Neither the optimiser
    # nor the integrator: the surface genuinely has secondary modes.
    #
    # STRICT GENERALISATION. When the set IS connected, the outermost accepted nodes are the
    # boundaries of the single run containing `imax`, so `lo`/`hi` are bit-identical to before. This
    # can only widen an interval, never narrow one.
    acc = np.flatnonzero(d <= thr)
    if acc.size:
        iL, iR = int(acc[0]), int(acc[-1])
        n_comp = 1 + int((np.diff(acc) > 1).sum())
    else:
        iL = iR = imax
        n_comp = 0
    lo = cross(range(iL, -1, -1))              # outward from the LEFTMOST accepted node
    hi = cross(range(iR, len(v)))              # outward from the RIGHTMOST accepted node
    # the component around the maximum, kept so the widening is auditable rather than invisible
    lo_conn = cross(range(imax, -1, -1))
    hi_conn = cross(range(imax, len(v)))
    has_lo, has_hi = np.isfinite(lo), np.isfinite(hi)
    if has_lo and has_hi:
        verdict = "identified"
    elif has_hi:
        verdict = "one-sided:<"                # bounded above only
    elif has_lo:
        verdict = "one-sided:>"                # bounded below only
    else:
        verdict = "flat"
    return dict(ci_lo=lo, ci_hi=hi, ci_level=float(level), ci_thr=thr, verdict=verdict,
                argmax_value=float(v[imax]), max_deviance=float(np.nanmax(d)),
                n_components=int(n_comp), ci_lo_connected=lo_conn, ci_hi_connected=hi_conn)
