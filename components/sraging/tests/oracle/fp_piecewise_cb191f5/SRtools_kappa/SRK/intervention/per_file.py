"""Per-condition ('per file') summary vs a control: longevity/steepness change, posterior parameter
ratios, and a data-vs-FP survival overlay.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .metrics import data_metrics, PARAM_NAMES
from .best_pair import posterior_ratio


def fp_survival_curve(theta, t_grid, xc=1.0, N=700, dt=1e-3):
    """FP model survival on a physical time grid (L=1 -> tau is physical when xc=1)."""
    from SRK.fp_likelihood import to_dimensionless, fpt_full
    eta, beta, epsilon, kappa = theta
    rho_beta, rho_eta, rho_eps, kap = to_dimensionless(eta, beta, epsilon, xc, 1.0, kappa)
    tmax = float(np.max(t_grid)) * 1.05
    _, _, _, tl, S = fpt_full(rho_eta, rho_beta, rho_eps, kap, N=N, dt=dt,
                              tau_max=tmax, return_survival=True)
    return np.interp(t_grid, tl, np.clip(S, 0, 1), left=1.0, right=float(S[-1]))


def per_file_table(control_label, control_ds, control_post, conditions, params=PARAM_NAMES):
    """Summary row per condition vs control.

    conditions : list of dicts each {label, ds, post[, meta]}. Returns a DataFrame with
    Δlongevity%, Δsteepness%, per-parameter posterior ratios (median + 68% CI), and the parameter
    that moved most in the posterior.
    """
    cm = data_metrics(control_ds)
    rows = []
    for cond in conditions:
        label, ds, post = cond["label"], cond["ds"], cond["post"]
        tm = data_metrics(ds)
        row = {"condition": label,
               "median_ctrl": cm["median"], "median_cond": tm["median"],
               "dlongevity_pct": 100.0 * (tm["median"] / cm["median"] - 1.0),
               "steepness_ctrl": cm["steepness"], "steepness_cond": tm["steepness"],
               "dsteepness_pct": 100.0 * (tm["steepness"] / cm["steepness"] - 1.0)}
        row.update(cond.get("meta", {}))
        if control_post is not None and post is not None:
            pr = posterior_ratio(control_post, post, params=params)
            for _, r in pr.iterrows():
                row[f"{r['param']}_ratio"] = r["ratio_median"]
                row[f"{r['param']}_ratio_lo"] = r["ratio_lo"]
                row[f"{r['param']}_ratio_hi"] = r["ratio_hi"]
            row["top_moved_param"] = pr.iloc[0]["param"]
            row["top_moved_log2"] = pr.iloc[0]["log2_ratio"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_survival_overlay(ax, ds, theta_best, xc=1.0, label=None, N=700):
    """Data KM survival + FP best-fit survival on one axis."""
    try:
        ds.plotSurvival(ax=ax, CI=False, label=f"data (n={ds.npeople})")
    except Exception:
        t, s = ds.survival
        ax.plot(t, s, label="data")
    tmax = float(np.nanmax(ds.death_times)) * 1.1
    tg = np.linspace(0, tmax, 400)
    S = fp_survival_curve(theta_best, tg, xc=xc, N=N)
    ax.plot(tg, S, "--", label="FP best-fit")
    ax.set_xlabel("age"); ax.set_ylabel("survival")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if label:
        ax.set_title(label, fontsize=9)
    return ax
