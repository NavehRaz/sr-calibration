"""Is a cohort one population or two? Discrete SR mixtures, fitted and compared by BIC.

A bimodal death-time distribution cannot be produced by the single-population SR model, nor by the
usual "individuals vary" story: spreading any one parameter (SRK's `fp_loglik_het_physical`) averages
survival curves that all have one hump, so it only makes that hump *broader*. Two humps need a
**discrete** mixture -- two sub-populations with genuinely different parameters.

This module fits and compares the three nested models a bimodality claim has to clear:

    M1  one population                     free: the named params           1 hump
    M2  one population + continuous spread free: + het_var                  1 broader hump
    M3  two populations                    free: + w, + split params        2 humps

`compare_models` returns one row per cohort with all three log-likelihoods and the BIC differences,
so "these data need two populations" becomes a number rather than an impression.

Everything is organism-agnostic: pass death times, events, and a starting theta.

Typical use
-----------
    from SRK.intervention import describe_modes, compare_models, plot_two_population

    modes = describe_modes(dt)                       # descriptive: are there two humps at all?
    row   = compare_models(dt, ev, theta0)           # inferential: does a mixture pay for itself?
    fig   = plot_two_population(dt, ev, row["fit_2pop"])

Notes
-----
* The split parameter defaults to `eta` (damage rate): under the near-ballistic SR regime the median
  scales as ~1/sqrt(eta), so two lifespans in ratio r imply eta in ratio ~r^2 -- which also gives
  `fit_two_population` its starting guess.
* `beta` is pinned by default (`PIN_BETA`): in these cohorts repair is unidentifiable, and leaving it
  free only adds a flat direction to a fit that already has a label-switching symmetry.
* The likelihood is the exact FP mixture: LL = sum_i log[w f(t_i|th_f) + (1-w) f(t_i|th_s)] over
  deaths and log[w S(t_i|th_f) + (1-w) S(t_i|th_s)] over censored. Mixing S linearly is exact.
* Fits are multi-start (`n_starts`): a two-population NLL has a saddle at th_fast == th_slow that a
  single start falls into regularly, which reads as "no support" when the truth is a bad optimum.
"""
import numpy as np
from scipy.optimize import minimize

from ..fp_likelihood import (fpt_full_density, to_dimensionless, kaplan_meier, _interp_S_p,
                             _interval_mass, _norm_sampling_dt, _LOG_FLOOR, fp_loglik_physical,
                             fp_loglik_het_physical)
from .metrics import PARAM_NAMES, PARAM_IDX

__all__ = ["describe_modes", "fit_one_population", "fit_heterogeneous", "fit_two_population",
           "compare_models", "verdict", "plot_two_population",
           "PIN_BETA", "FP_SOLVER", "MIN_WEIGHT", "DBIC_BAR"]

PIN_BETA = 1e-8                 # repair off: unidentifiable here, see module docstring
MIN_WEIGHT = 0.10               # a "population" has to be more than a few animals -- see compare_models
DBIC_BAR = 10.0                 # conventional "decisive" bar
FP_SOLVER = dict(N=900, dt=1e-3, tau_max=20.0, Gtol=1e-11)
_HET_VARS = (0.1, 0.2, 0.3)     # profiled over, so M2 gets its best shot


# --------------------------------------------------------------------------- descriptive
def describe_modes(death_times, events=None, bw=None, n_grid=512, min_frac=0.05):
    """Count humps in the death-time density by counting local maxima of a KDE.

    Deliberately NOT a mixture BIC: fitting two Gaussians to a one-humped sample happily "finds" two
    components, so it answers a different question. Here a mode must be a real turning point of the
    smoothed density, and both modes must carry at least `min_frac` of the sample.

    Returns {n_modes, modes, valley, early_frac, early_mean, late_mean, ratio, bw}. `valley` is the
    density minimum between the two highest modes -- the natural place to split the sample.
    """
    from scipy.stats import gaussian_kde
    t = np.asarray(death_times, float)
    if events is not None:
        t = t[np.asarray(events, int) == 1]
    t = t[np.isfinite(t)]
    out = dict(n_modes=1, modes=[float(np.median(t))], valley=np.nan, early_frac=np.nan,
               early_mean=np.nan, late_mean=np.nan, ratio=np.nan, bw=np.nan)
    if t.size < 20:
        return out
    kde = gaussian_kde(t, bw_method=bw) if bw is not None else gaussian_kde(t)
    out["bw"] = float(kde.factor * t.std(ddof=1))
    g = np.linspace(t.min(), t.max(), n_grid)
    d = kde(g)
    up = np.flatnonzero((d[1:-1] > d[:-2]) & (d[1:-1] >= d[2:])) + 1
    if up.size < 2:
        out["modes"] = [float(g[int(np.argmax(d))])]
        return out
    # keep the two tallest modes, then require each side of their valley to be populated
    top = up[np.argsort(d[up])[::-1][:2]]
    lo, hi = sorted(g[top])
    seg = (g >= lo) & (g <= hi)
    valley = float(g[seg][int(np.argmin(d[seg]))])
    early, late = t[t <= valley], t[t > valley]
    frac = early.size / t.size
    if min(frac, 1 - frac) < min_frac:
        out["modes"] = [float(g[int(np.argmax(d))])]
        return out
    out.update(n_modes=2, modes=[lo, hi], valley=valley, early_frac=float(frac),
               early_mean=float(early.mean()), late_mean=float(late.mean()),
               ratio=float(late.mean() / early.mean()))
    return out


# --------------------------------------------------------------------------- likelihood pieces
def _dt_repr(sampling_dt, n):
    """One CSV-safe cell describing what interval width the row was fitted with.

    None, a float when every observation shares a width, else a summary of the per-observation
    widths -- a results table has to say which likelihood produced it.
    """
    d = _norm_sampling_dt(sampling_dt, n)
    if d is None:
        return None
    u = np.unique(d)
    if u.size == 1:
        return float(u[0])
    return f"per-row(mean={d.mean():.3f}, widened={int((d > u.min()).sum())}/{d.size})"


def _curves(theta, xc, t, L, sampling_dt=None, **solver_kw):
    """Physical (S, death-weight) at times `t` for one SR population.

    `sampling_dt=None` (default) returns the density p(t) as the death weight -- the original,
    bit-for-bit behaviour. Set it to the recorded-age resolution and the death weight becomes the
    interval mass P(t <= T < t+delta) = S(t) - S(t+delta) instead, which is the correct likelihood
    when many individuals share a recorded time. It matters where delta/L is not small: the worm
    paraquat cohorts are recorded to the day against a 4-11 day median (delta/L = 9-25 %), so the
    density form treats hundreds of same-day deaths as exact coincidences.

    `_mix_ll` needs no change to support this. Interval mass is LINEAR in the mixture, exactly as
    the density is: S_mix(t) - S_mix(t+delta) = sum_i w_i [S_i(t) - S_i(t+delta)], so summing the
    components' interval masses with the same weights is exact. (Same argument as the one in
    `fp_loglik_het_physical`, which already takes `sampling_dt` for the continuous-spread model.)
    """
    eta, beta, epsilon, kappa = theta
    rb, rh, re, kp = to_dimensionless(eta, beta, epsilon, xc, L, k=kappa)
    tau, S, p, *_ = fpt_full_density(rh, rb, re, kp, **{**FP_SOLVER, **solver_kw})
    Si, pi = _interp_S_p(t / L, tau, S, p)
    dt = _norm_sampling_dt(sampling_dt, t.size)
    if dt is not None:
        # h=0: no Makeham hazard in this module, so the physical and dimensionless S coincide here.
        return Si, _interval_mass((t + dt) / L, tau, S, Si, 0.0, t, dt)
    return Si, pi / L


def _mix_ll(t, e, parts, weights):
    """log-likelihood of a discrete mixture given each component's (S, p) at the observed times."""
    d = e == 1
    S = sum(w * s for w, (s, _) in zip(weights, parts))
    p = sum(w * q for w, (_, q) in zip(weights, parts))
    with np.errstate(divide="ignore"):               # log(0) -> -inf, then floored: expected
        return (np.maximum(np.log(np.clip(p[d], 0, None)), _LOG_FLOOR).sum()
                + np.maximum(np.log(np.clip(S[~d], 0, None)), _LOG_FLOOR).sum())


def _pack(theta, free):
    return np.log([theta[PARAM_IDX[p]] for p in free])


def _unpack(z, theta0, free):
    th = np.asarray(theta0, float).copy()
    for p, v in zip(free, np.exp(z)):
        th[PARAM_IDX[p]] = v
    return th


# --------------------------------------------------------------------------- M1
def fit_one_population(death_times, events, theta0, free=("eta", "epsilon"), xc=1.0,
                       pin_beta=PIN_BETA, maxiter=400, sampling_dt=None, **solver_kw):
    """M1: the ordinary single-population FP fit, restricted to `free` (log-space Nelder-Mead).

    `sampling_dt` -- recorded-age resolution; see `_curves`. Taken as an explicit argument rather
    than through `solver_kw`, which is splatted into `fpt_full_density` and would reject it.
    """
    t = np.asarray(death_times, float)
    e = np.ones(t.size, int) if events is None else np.asarray(events, int)
    _, _, L = kaplan_meier(t, e)
    th0 = np.asarray(theta0, float).copy()
    if pin_beta is not None:
        th0[PARAM_IDX["beta"]] = pin_beta

    def nll(z):
        th = _unpack(z, th0, free)
        if not (np.isfinite(th).all() and (th[[0, 2, 3]] > 0).all()):
            return 1e12
        S, p = _curves(th, xc, t, L, sampling_dt=sampling_dt, **solver_kw)
        return -_mix_ll(t, e, [(S, p)], [1.0])

    m = minimize(nll, _pack(th0, free), method="Nelder-Mead",
                 options=dict(maxiter=maxiter, xatol=1e-4, fatol=1e-3))
    th = _unpack(m.x, th0, free)
    return dict(model="1pop", theta=th, xc=xc, loglik=float(-m.fun), n=int(t.size),
                k=len(free), L=float(L), converged=bool(m.success))


# --------------------------------------------------------------------------- M2
def fit_heterogeneous(death_times, events, theta, xc=1.0, het_param="xc", het_vars=_HET_VARS,
                      M=11, sampling_dt=None, **solver_kw):
    """M2: one population with a continuous spread in `het_param`, profiled over `het_vars`.

    Uses SRK's existing mixture likelihood. Counts one extra parameter (the spread).
    `fp_loglik_het_physical` already takes `sampling_dt`; naming it here keeps the three models
    consistent and stops it being passed positionally by accident.
    """
    t = np.asarray(death_times, float)
    e = np.ones(t.size, int) if events is None else np.asarray(events, int)
    eta, beta, epsilon, kappa = np.asarray(theta, float)
    lls = [fp_loglik_het_physical(t, e, eta, beta, epsilon, xc, kappa,
                                  het_param=het_param, het_var=v, M=M,
                                  sampling_dt=sampling_dt, **solver_kw)
           for v in het_vars]
    j = int(np.argmax(lls))
    return dict(model="het", theta=np.asarray(theta, float), xc=xc, het_param=het_param,
                het_var=float(het_vars[j]), loglik=float(lls[j]), n=int(t.size), k=None)


# --------------------------------------------------------------------------- M3
def fit_two_population(death_times, events, theta0, split=("eta",), shared=("epsilon",), xc=1.0,
                       pin_beta=PIN_BETA, guess_ratio=None, n_starts=9, maxiter=2000,
                       n_jobs=1, seed=0, init=None, sampling_dt=None, **solver_kw):
    """M3: two sub-populations differing in `split`, sharing `shared`.

    Free parameters: the mixing weight w, `split` twice (fast and slow), `shared` once.

    Multi-start over the split ratio and w. The NLL has a saddle at fast == slow (where the mixture
    degenerates to one population); a single start lands there often enough that a lone fit collapsing
    to ratio 1.0 says nothing about the data. `guess_ratio` seeds the ladder -- for a near-ballistic
    SR cohort, (late_median / early_median)**2 from `describe_modes` is the natural guess.

    Returns the best start, with `starts` listing every one for inspection.
    """
    t = np.asarray(death_times, float)
    e = np.ones(t.size, int) if events is None else np.asarray(events, int)
    _, _, L = kaplan_meier(t, e)
    th0 = np.asarray(theta0, float).copy()
    if pin_beta is not None:
        th0[PARAM_IDX["beta"]] = pin_beta
    split, shared = tuple(split), tuple(shared)

    def nll(z):
        w = 1.0 / (1.0 + np.exp(-z[0]))
        nf = len(split)
        th_f = _unpack(z[1:1 + nf], th0, split)
        th_s = _unpack(z[1 + nf:1 + 2 * nf], th0, split)
        if shared:
            sh = z[1 + 2 * nf:]
            th_f = _unpack(sh, th_f, shared)
            th_s = _unpack(sh, th_s, shared)
        for th in (th_f, th_s):
            if not (np.isfinite(th).all() and (th[[0, 2, 3]] > 0).all()):
                return 1e12
        pf = _curves(th_f, xc, t, L, sampling_dt=sampling_dt, **solver_kw)
        ps = _curves(th_s, xc, t, L, sampling_dt=sampling_dt, **solver_kw)
        return -_mix_ll(t, e, [pf, ps], [w, 1.0 - w])

    if init is not None:
        w0, thf0, ths0 = init
        z = [np.log(float(w0) / (1.0 - float(w0)))]
        z += list(_pack(np.asarray(thf0, float), split))
        z += list(_pack(np.asarray(ths0, float), split))
        z += list(_pack(np.asarray(thf0, float), shared))
        starts = [np.array(z, float)]
        n_starts = 1
    base = guess_ratio if (guess_ratio and np.isfinite(guess_ratio)) else 3.0
    ratios = np.unique(np.round([base, base * 0.5, base * 2.0, 1.6, 5.0], 4))
    ws = (0.5, 0.3, 0.7)
    if init is None:
        starts = []
    for r in ratios if init is None else []:
        for w in ws:
            z = [np.log(w / (1 - w))]
            z += list(_pack(th0, split) + np.log(r) / 2)      # fast: scaled up
            z += list(_pack(th0, split) - np.log(r) / 2)      # slow: scaled down
            z += list(_pack(th0, shared))
            starts.append(np.array(z, float))
            if len(starts) >= n_starts:
                break
        if len(starts) >= n_starts:
            break

    def run(z0):
        m = minimize(nll, z0, method="Nelder-Mead",
                     options=dict(maxiter=maxiter, xatol=1e-4, fatol=1e-3))
        return float(-m.fun), m.x, bool(m.success)

    if n_jobs != 1 and len(starts) > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(run)(z) for z in starts)
    else:
        res = [run(z) for z in starts]

    ll, z, ok = max(res, key=lambda r: r[0])
    nf = len(split)
    w = 1.0 / (1.0 + np.exp(-z[0]))
    th_f = _unpack(z[1:1 + nf], th0, split)
    th_s = _unpack(z[1 + nf:1 + 2 * nf], th0, split)
    if shared:
        th_f = _unpack(z[1 + 2 * nf:], th_f, shared)
        th_s = _unpack(z[1 + 2 * nf:], th_s, shared)
    # order components so the fast-ageing one (larger split param) is first
    if th_f[PARAM_IDX[split[0]]] < th_s[PARAM_IDX[split[0]]]:
        w, th_f, th_s = 1.0 - w, th_s, th_f
    ratios_out = {p: float(th_f[PARAM_IDX[p]] / th_s[PARAM_IDX[p]]) for p in split}
    return dict(model="2pop", w_fast=float(w), theta_fast=th_f, theta_slow=th_s, xc=xc,
                loglik=float(ll), n=int(t.size), k=1 + 2 * nf + len(shared), L=float(L),
                split=split, shared=shared, ratio=ratios_out, converged=ok,
                starts=[(r[0], bool(r[2])) for r in res])


# --------------------------------------------------------------------------- comparison
def _bic(ll, k, n):
    return -2.0 * ll + k * np.log(n)


def verdict(row, min_weight=MIN_WEIGHT, dbic_bar=DBIC_BAR, require_two_modes=True):
    """Joint two-population verdict from a `compare_models` row (or any mapping with those keys).

    Separated out so an existing results table can be re-scored without refitting. See
    `compare_models` for why dBIC alone is not enough.
    """
    ok = (float(row["dBIC_2pop_vs_1"]) > dbic_bar
          and min(float(row["w_fast"]), 1.0 - float(row["w_fast"])) >= min_weight)
    if require_two_modes:
        ok = ok and int(row.get("n_modes", 0)) == 2
    return bool(ok)


def compare_models(death_times, events, theta0, free=("eta", "epsilon"), split=("eta",),
                   shared=("epsilon",), xc=1.0, het_param="xc", n_starts=9, n_jobs=1,
                   modes=None, min_weight=MIN_WEIGHT, dbic_bar=DBIC_BAR, require_two_modes=True,
                   sampling_dt=None, **solver_kw):
    """Fit M1/M2/M3 to one cohort and return a single comparison row.

    dBIC_* > 0 favours the richer model. `modes` may be a `describe_modes` result: it supplies M3's
    starting ratio, and (with `require_two_modes`) is part of the verdict.

    `sampling_dt` -- the recorded-age resolution, applied identically to all three models so the BIC
    differences stay comparable. Leave it None for exact death times; SET IT whenever delta/L is not
    small, or the density form will treat same-day deaths as exact coincidences and the mixture will
    buy that misfit with a spurious second component. See `_curves`.

    THE VERDICT IS DELIBERATELY STRICTER THAN dBIC ALONE. A two-component mixture is a very flexible
    tail model, so it wins on BIC whenever the single-population shape misses the tail -- including when
    the "second population" is a handful of animals, or one outlier. Measured example: a cohort with a
    single animal dying 17 days after the rest scored dBIC +121 with w_fast = 0.995 (minor component
    ~1 animal); deleting that one animal took it to -11. So `two_populations` requires all of

        dBIC_2pop_vs_1 > dbic_bar          the mixture pays for its parameters
        min(w, 1-w)   >= min_weight        BOTH components hold a real share of the cohort
        two visible humps                  (if require_two_modes and `modes` was supplied)

    `dBIC_2pop_vs_1` and `minor_frac` are reported separately, so a cohort failing only the weight test
    is visible as what it is: a tail/contamination effect, not two ageing rates.
    """
    m1 = fit_one_population(death_times, events, theta0, free=free, xc=xc,
                            sampling_dt=sampling_dt, **solver_kw)
    m2 = fit_heterogeneous(death_times, events, m1["theta"], xc=xc, het_param=het_param,
                           sampling_dt=sampling_dt, **solver_kw)
    guess = None
    if modes and modes.get("n_modes", 1) == 2 and np.isfinite(modes.get("ratio", np.nan)):
        guess = modes["ratio"] ** 2                      # ballistic: median ~ 1/sqrt(eta)
    m3 = fit_two_population(death_times, events, m1["theta"], split=split, shared=shared, xc=xc,
                            guess_ratio=guess, n_starts=n_starts, n_jobs=n_jobs,
                            sampling_dt=sampling_dt, **solver_kw)

    n = m1["n"]
    b1 = _bic(m1["loglik"], m1["k"], n)
    b2 = _bic(m2["loglik"], m1["k"] + 1, n)
    b3 = _bic(m3["loglik"], m3["k"], n)
    minor = min(m3["w_fast"], 1.0 - m3["w_fast"])
    n_modes = int(modes["n_modes"]) if modes else 0
    row = dict(n=n, ll_1pop=m1["loglik"], ll_het=m2["loglik"], ll_2pop=m3["loglik"],
               het_var=m2["het_var"], dBIC_het_vs_1=b1 - b2, dBIC_2pop_vs_1=b1 - b3,
               w_fast=m3["w_fast"], minor_frac=minor, minor_n=minor * n, n_modes=n_modes,
               gain_per_animal=(m3["loglik"] - m1["loglik"]) / n,
               converged_2pop=m3["converged"],
               sampling_dt=_dt_repr(sampling_dt, n))
    row["two_populations"] = bool(verdict(row, min_weight=min_weight, dbic_bar=dbic_bar,
                                         require_two_modes=require_two_modes and bool(modes)))
    for p in PARAM_NAMES:
        row[f"{p}_1pop"] = m1["theta"][PARAM_IDX[p]]
    for p in split:
        row[f"{p}_fast"] = m3["theta_fast"][PARAM_IDX[p]]
        row[f"{p}_slow"] = m3["theta_slow"][PARAM_IDX[p]]
        row[f"{p}_ratio"] = m3["ratio"][p]
    for p in shared:                       # both components share these; recorded so the row alone
        row[f"{p}_shared"] = m3["theta_fast"][PARAM_IDX[p]]      # reconstructs the fitted mixture
    row["fit_1pop"], row["fit_het"], row["fit_2pop"] = m1, m2, m3
    return row


# --------------------------------------------------------------------------- figure
def plot_two_population(death_times, events, fit_2pop, fit_1pop=None, ax=None, label="",
                        n_bins=25, **solver_kw):
    """Death-time histogram + both models' densities, and survival below. Returns the figure.

    The point of the figure is that the one-population density has one hump wherever the data have
    two; the mixture's two components are drawn separately (dashed) so the split is visible.
    """
    import matplotlib.pyplot as plt
    from .factor_plots import STYLE, _despine
    t = np.asarray(death_times, float)
    e = np.ones(t.size, int) if events is None else np.asarray(events, int)
    L = fit_2pop["L"]
    grid = np.linspace(max(t.min() * 0.6, 1e-3), t.max() * 1.05, 400)

    with plt.rc_context(STYLE):
        if ax is None:
            fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True,
                                     gridspec_kw=dict(height_ratios=[1.3, 1]))
        else:
            axes = np.atleast_1d(ax)
            fig = axes[0].figure
        a0 = axes[0]
        a0.hist(t[e == 1], bins=n_bins, density=True, color="#CCCCCC", edgecolor="white",
                label="data")
        # DENSITY on purpose -- these overlay a density histogram on a fine grid, so they must stay
        # p(t) even when the fit used sampling_dt. Do not thread sampling_dt in here.
        Sf, pf = _curves(fit_2pop["theta_fast"], fit_2pop["xc"], grid, L, **solver_kw)
        Ss, ps = _curves(fit_2pop["theta_slow"], fit_2pop["xc"], grid, L, **solver_kw)
        w = fit_2pop["w_fast"]
        a0.plot(grid, w * pf, "--", color="#D55E00", lw=1.6,
                label=f"fast-ageing ({100*w:.0f} %)")
        a0.plot(grid, (1 - w) * ps, "--", color="#0072B2", lw=1.6,
                label=f"slow-ageing ({100*(1-w):.0f} %)")
        a0.plot(grid, w * pf + (1 - w) * ps, "-", color="#000000", lw=2.2, label="two populations")
        if fit_1pop is not None:
            S1, p1 = _curves(fit_1pop["theta"], fit_1pop["xc"], grid, L, **solver_kw)
            a0.plot(grid, p1, "-", color="#888888", lw=2.0, label="one population")
        a0.set_ylabel("death-time density")
        ttl = f"{label}: two populations vs one" if label else "two populations vs one"
        a0.set_title(ttl)
        _despine(a0)
        a0.legend(fontsize=9, frameon=False)

        a1 = axes[-1]
        ts, S_km, _ = kaplan_meier(t, e)
        a1.step(ts, S_km, where="post", color="#333333", lw=2.0, label="data (KM)")
        a1.plot(grid, w * Sf + (1 - w) * Ss, "-", color="#000000", lw=2.0,
                label="two populations")
        if fit_1pop is not None:
            a1.plot(grid, S1, "-", color="#888888", lw=2.0, label="one population")
        a1.set_xlabel("age (days)")
        a1.set_ylabel("survival")
        a1.set_ylim(-0.02, 1.02)
        _despine(a1)
        a1.legend(fontsize=9, frameon=False)
        fig.tight_layout()
    return fig
