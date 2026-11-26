"""Steepness-longevity parameter plane: single-parameter effect lines (with baseline uncertainty)
and condition points, in fractional-change coordinates relative to a control.

Coordinates (matching the killifish param-plane convention):
    x = median(intervened) / median(control) - 1      (longevity fractional change)
    y = steepness(intervened) / steepness(control) - 1 (steepness fractional change)

- ``intervention_lines`` sweeps a multiplicative effect on each SR parameter (chronic) and returns
  the (x, y) trajectory relative to the *same* theta's un-intervened metrics -> origin at effect 0.
- Baseline uncertainty on those lines comes from the control's FP posterior, two ways:
  ``line_band_bootstrap`` (draw baseline theta from the posterior, redo the sweep, take percentiles)
  and ``line_band_linearized`` (Jacobian of (x,y) w.r.t. theta x posterior covariance) -- a cheap
  analytic cross-check on one control.
- ``condition_points`` places each condition as a data (model-free) fractional change vs the control
  data, with a per-individual bootstrap CI (as in killifish ``bootstrap_metric_ci``).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .metrics import fp_metrics, data_metrics, Intervention, PARAM_NAMES

try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except Exception:                                        # pragma: no cover
    _HAVE_JOBLIB = False

DEFAULT_EFFECTS = np.round(np.linspace(-0.4, 0.4, 17), 4)
_COLORS = {"eta": "#0072B2", "beta": "#D55E00", "epsilon": "#009E73", "kappa": "#CC79A7",
           "xc": "#E69F00"}


# --------------------------------------------------------------------------- effect lines
def _effects_for(effects, param):
    """`effects` may be one array for every parameter, or a dict {param: array} (with an optional
    'default' key). Per-parameter grids matter because equal multiplicative ranges do NOT produce
    equal excursions in the plane -- xc in particular has far more leverage per factor than eta/eps,
    so sharing one grid makes its arm dwarf everything else."""
    if isinstance(effects, dict):
        return effects.get(param, effects.get("default", DEFAULT_EFFECTS))
    return effects


def intervention_lines(theta, params=PARAM_NAMES, effects=DEFAULT_EFFECTS, xc=1.0, L=1.0, N=500):
    """(x, y) fractional-change trajectory for a chronic multiplicative sweep of each parameter.

    `effects` accepts a dict {param: array} for per-parameter grids (see `_effects_for`).
    """
    base = fp_metrics(theta, xc=xc, L=L, N=N)
    rows = []
    for p in params:
        for e in _effects_for(effects, p):
            if abs(e) < 1e-12:
                x = y = 0.0
            else:
                m = fp_metrics(theta, [Intervention(p, float(e))], xc=xc, L=L, N=N)
                x = m["median"] / base["median"] - 1.0
                y = m["steepness"] / base["steepness"] - 1.0
            rows.append({"param": p, "effect": float(e), "x": x, "y": y})
    return pd.DataFrame(rows)


def line_band_bootstrap(thetas, params=PARAM_NAMES, effects=DEFAULT_EFFECTS, xc=1.0, L=1.0, N=500,
                        n_draws=150, seed=0, n_jobs=-1):
    """Posterior-bootstrapped effect lines: draw baseline theta, redo the sweep, take percentiles.

    thetas : (M, 4) posterior samples [eta,beta,epsilon,kappa]. Returns per (param, effect) the
    median line and 68/95% envelopes (x_lo/x_hi = 16/84, x_lo95/x_hi95 = 2.5/97.5; same for y).
    """
    thetas = np.asarray(thetas, float)
    rng = np.random.default_rng(seed)
    m = len(thetas)
    idx = rng.choice(m, min(n_draws, m), replace=(n_draws > m))

    def one(i):
        return intervention_lines(thetas[i], params, effects, xc, L, N)

    if _HAVE_JOBLIB and len(idx) > 1:
        dfs = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(one)(i) for i in idx)
    else:
        dfs = [one(i) for i in idx]

    stacked = pd.concat(dfs, ignore_index=True)
    out = []
    for (p, e), g in stacked.groupby(["param", "effect"]):
        xv, yv = g["x"].to_numpy(), g["y"].to_numpy()
        # NaN-SAFE reduction. `np.percentile` propagates: ONE non-finite draw turns a whole node into
        # NaN, and matplotlib then draws nothing for that arm while still registering a legend entry
        # -- which is exactly how this figure once came out with a five-parameter legend and no lines.
        # nanpercentile degrades gracefully instead, and `n_finite` records how much was usable so a
        # thin node cannot pass unnoticed.
        fx, fy = np.isfinite(xv), np.isfinite(yv)
        ok = fx & fy
        if not ok.any():
            continue
        xv, yv = xv[ok], yv[ok]
        out.append({
            "param": p, "effect": e, "n_finite": int(ok.sum()), "n_draws": int(ok.size),
            "x": np.nanmedian(xv), "y": np.nanmedian(yv),
            "x_lo": np.nanpercentile(xv, 16), "x_hi": np.nanpercentile(xv, 84),
            "y_lo": np.nanpercentile(yv, 16), "y_hi": np.nanpercentile(yv, 84),
            "x_lo95": np.nanpercentile(xv, 2.5), "x_hi95": np.nanpercentile(xv, 97.5),
            "y_lo95": np.nanpercentile(yv, 2.5), "y_hi95": np.nanpercentile(yv, 97.5),
        })
    return pd.DataFrame(out).sort_values(["param", "effect"]).reset_index(drop=True)


def line_band_linearized(thetas, params=PARAM_NAMES, effects=DEFAULT_EFFECTS, xc=1.0, L=1.0, N=500,
                         rel_step=0.02):
    """Analytic 1-sigma band: Jacobian of (x,y) w.r.t. theta at the posterior mean x posterior cov.

    Cheap cross-check for one control. (x,y) depend on theta through both the intervened and the
    baseline metric, so the finite-difference perturbs theta and recomputes the full fractional
    change. Returns per (param, effect): x, y and x_err, y_err (1-sigma).
    """
    thetas = np.asarray(thetas, float)
    theta0 = thetas.mean(axis=0)
    cov = np.cov(thetas.T)                                # 4x4 posterior covariance

    def xy(theta):
        base = fp_metrics(theta, xc=xc, L=L, N=N)
        vals = {}
        for p in params:
            for e in _effects_for(effects, p):
                if abs(e) < 1e-12:
                    vals[(p, float(e))] = (0.0, 0.0)
                else:
                    m = fp_metrics(theta, [Intervention(p, float(e))], xc=xc, L=L, N=N)
                    vals[(p, float(e))] = (m["median"] / base["median"] - 1.0,
                                           m["steepness"] / base["steepness"] - 1.0)
        return vals

    base_vals = xy(theta0)
    # finite-difference Jacobian: d(x,y)/d(theta_k) for each node
    keys = list(base_vals.keys())
    Jx = {k: np.zeros(4) for k in keys}
    Jy = {k: np.zeros(4) for k in keys}
    for k_idx in range(4):
        dtheta = theta0.copy()
        step = rel_step * abs(theta0[k_idx]) or rel_step
        dtheta[k_idx] += step
        pert = xy(dtheta)
        for k in keys:
            Jx[k][k_idx] = (pert[k][0] - base_vals[k][0]) / step
            Jy[k][k_idx] = (pert[k][1] - base_vals[k][1]) / step

    out = []
    for (p, e) in keys:
        x0, y0 = base_vals[(p, e)]
        xerr = float(np.sqrt(max(Jx[(p, e)] @ cov @ Jx[(p, e)], 0.0)))
        yerr = float(np.sqrt(max(Jy[(p, e)] @ cov @ Jy[(p, e)], 0.0)))
        out.append({"param": p, "effect": e, "x": x0, "y": y0, "x_err": xerr, "y_err": yerr})
    return pd.DataFrame(out).sort_values(["param", "effect"]).reset_index(drop=True)


# --------------------------------------------------------------------------- condition points
def _km_median(dt, ev):
    from SRK.fp_likelihood import kaplan_meier
    tk, Sk, _ = kaplan_meier(dt, ev)
    return float(tk[np.argmin(np.abs(np.asarray(Sk) - 0.5))])


def _arr_metrics(dt, ev):
    """median (KM) + inverse-CV steepness directly from arrays (fast, matches data_metrics)."""
    died = dt[ev == 1]
    if died.size < 2:
        return np.nan, np.nan
    std = np.std(died)
    steep = np.mean(died) / std if std > 0 else np.nan
    return _km_median(dt, ev), float(steep)


def condition_points(cond_ds, control_ds, n_boot=500, seed=0):
    """Data fractional change of a condition vs control, with a per-individual bootstrap CI.

    Returns dict(x, y, x_lo/x_hi (95%), y_lo/y_hi (95%), x_lo68/x_hi68, y_lo68/y_hi68).
    """
    cm = data_metrics(control_ds)
    tm = data_metrics(cond_ds)
    x = tm["median"] / cm["median"] - 1.0
    y = tm["steepness"] / cm["steepness"] - 1.0

    c_dt = np.asarray(control_ds.death_times, float); c_ev = np.asarray(control_ds.events, int)
    t_dt = np.asarray(cond_ds.death_times, float); t_ev = np.asarray(cond_ds.events, int)
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n_boot):
        ci = rng.integers(0, c_dt.size, c_dt.size)
        ti = rng.integers(0, t_dt.size, t_dt.size)
        c_med, c_st = _arr_metrics(c_dt[ci], c_ev[ci])
        t_med, t_st = _arr_metrics(t_dt[ti], t_ev[ti])
        if c_med and c_st and np.isfinite(c_med) and np.isfinite(c_st) and c_med > 0 and c_st > 0:
            xs.append(t_med / c_med - 1.0); ys.append(t_st / c_st - 1.0)
    xs, ys = np.asarray(xs), np.asarray(ys)
    return {
        "x": x, "y": y,
        "x_lo": np.percentile(xs, 2.5), "x_hi": np.percentile(xs, 97.5),
        "y_lo": np.percentile(ys, 2.5), "y_hi": np.percentile(ys, 97.5),
        "x_lo68": np.percentile(xs, 16), "x_hi68": np.percentile(xs, 84),
        "y_lo68": np.percentile(ys, 16), "y_hi68": np.percentile(ys, 84),
    }


# --------------------------------------------------------------------------- plot
def plot_param_plane(ax, band_df, points_df, control_label="control", title=None,
                     linearized_df=None, ci="68", xlim=None, ylim=None, focus_pad=0.18,
                     band_style="fill", band_alpha=0.18):
    """Render the plane: effect lines with posterior band, condition points with error bars.

    band_df       : output of line_band_bootstrap (per param,effect x,y,+envelopes).
    points_df     : DataFrame with columns label,x,y and (x_lo,x_hi,y_lo,y_hi) 95% CI.
    linearized_df : optional line_band_linearized output -> overlaid dashed error bars (one control).
    ci            : '68' or '95' for the band width drawn on the lines.
    band_style    : 'fill' (default) shades the arm's uncertainty as a filled ribbon; 'errorbar'
                    restores the old per-node crosses. The arms are CURVES, so a ribbon reads as one
                    continuous region of plausible trajectories, whereas a cross at every node reads
                    as a scatter of unrelated points and clutters the panel badly once several arms
                    overlap.
    """
    lo, hi = ("x_lo", "x_hi") if ci == "68" else ("x_lo95", "x_hi95")
    ylo, yhi = ("y_lo", "y_hi") if ci == "68" else ("y_lo95", "y_hi95")

    for p, sub in band_df.groupby("param"):
        sub = sub.sort_values("effect")
        c = _COLORS.get(p, "#333333")
        ax.plot(sub["x"], sub["y"], "-o", ms=3, lw=1.6, color=c, label=f"{p}", zorder=3)
        if band_style == "fill":
            # Closed ribbon: up one envelope along the effect grid and back down the other. Both x
            # and y vary along an arm, so this is the region the arm sweeps under the reference's
            # own posterior uncertainty -- not a vertical band around a function.
            fx = np.concatenate([sub[hi].to_numpy(), sub[lo].to_numpy()[::-1]])
            fy = np.concatenate([sub[yhi].to_numpy(), sub[ylo].to_numpy()[::-1]])
            good = np.isfinite(fx) & np.isfinite(fy)
            if good.sum() >= 3:
                ax.fill(fx[good], fy[good], color=c, alpha=band_alpha, lw=0, zorder=1)
        else:
            ax.errorbar(sub["x"], sub["y"],
                        xerr=[sub["x"] - sub[lo], sub[hi] - sub["x"]],
                        yerr=[sub["y"] - sub[ylo], sub[yhi] - sub["y"]],
                        fmt="none", ecolor=c, alpha=0.35, capsize=0, zorder=2)

    if linearized_df is not None:
        for p, sub in linearized_df.groupby("param"):
            sub = sub.sort_values("effect")
            c = _COLORS.get(p, "#333333")
            ax.errorbar(sub["x"], sub["y"], xerr=sub["x_err"], yerr=sub["y_err"],
                        fmt="none", ecolor=c, alpha=0.5, capsize=2, ls=":", zorder=2)

    pts = points_df
    ax.errorbar(pts["x"], pts["y"],
                xerr=[pts["x"] - pts["x_lo"], pts["x_hi"] - pts["x"]],
                yerr=[pts["y"] - pts["y_lo"], pts["y_hi"] - pts["y"]],
                fmt="none", ecolor="#666666", alpha=0.6, capsize=2, zorder=4)
    ax.scatter(pts["x"], pts["y"], s=60, color="#F0E442", edgecolor="#222222", lw=0.8,
               zorder=5, label="conditions")
    for _, r in pts.iterrows():
        ax.annotate(str(r["label"]), (r["x"], r["y"]), fontsize=9,
                    xytext=(4, 4), textcoords="offset points")

    # Keep the view on the conditions (+ a pad); long effect arms simply run off-panel rather than
    # compressing everything into a corner. Explicit xlim/ylim always win.
    if xlim is None or ylim is None:
        xs = np.concatenate([points_df["x_lo"].to_numpy(float), points_df["x_hi"].to_numpy(float), [0.0]])
        ys = np.concatenate([points_df["y_lo"].to_numpy(float), points_df["y_hi"].to_numpy(float), [0.0]])
        padx = focus_pad * max(np.ptp(xs), 1e-3); pady = focus_pad * max(np.ptp(ys), 1e-3)
        if xlim is None:
            xlim = (xs.min() - padx, xs.max() + padx)
        if ylim is None:
            ylim = (ys.min() - pady, ys.max() + pady)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    ax.axhline(0, color="k", lw=0.6, alpha=0.4); ax.axvline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("Longevity fractional change  (median vs %s)" % control_label)
    ax.set_ylabel("Steepness fractional change  (1/CV vs %s)" % control_label)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=10, loc="best")
    return ax
