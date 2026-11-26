"""Which SR parameter(s) changed vs control? Two independent methods (report both, needn't agree).

1. ``fp_scan`` -- FP-likelihood forward scan. Apply single and paired multiplicative changes to the
   control theta, score each against the condition's death-time data with the exact FP likelihood
   ``fp_loglik_physical`` (analytic, no simulation). Returns best single param and best pair.
   ``params`` may include ``'xc'`` (see metrics.SWEEP_NAMES): the threshold is sweepable here even
   though it is never fitted. It is exactly degenerate with a coordinated (eta, beta, epsilon, kappa)
   change, so read an xc win as "a threshold change fits too", not as an identified parameter.
2. ``posterior_ratio`` -- direct comparison of the two FP posteriors: the distribution of each
   parameter ratio condition/control, ranked by how far it sits from 1.

Each is runnable on its own.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

from .metrics import apply_chronic, split_xc, Intervention, PARAM_NAMES, SWEEP_NAMES

try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except Exception:                                        # pragma: no cover
    _HAVE_JOBLIB = False

DEFAULT_SCAN_EFFECTS = (-0.6, -0.4, -0.25, -0.15, -0.07, 0.07, 0.15, 0.25, 0.4, 0.6)
DEFAULT_PAIR_EFFECTS = (-0.6, -0.4, -0.2, 0.2, 0.4, 0.6)


def fp_scan(control_theta, cond_dt, cond_ev, params=PARAM_NAMES,
            effects=DEFAULT_SCAN_EFFECTS, pair_effects=DEFAULT_PAIR_EFFECTS,
            xc=1.0, external_hazard=np.inf, pairs=True, n_jobs=-1):
    """Score single/paired chronic interventions on control_theta against the condition data.

    Returns dict(baseline_ll, singles, pairs, best_single, best_pair) where singles/pairs are
    DataFrames of log-likelihoods and *_ll_gain = LL - baseline_ll.
    """
    from SRK.fp_likelihood import fp_loglik_physical
    control_theta = np.asarray(control_theta, float)
    cond_dt = np.asarray(cond_dt, float); cond_ev = np.asarray(cond_ev, int)

    def score(interventions):
        ivs, xc_scale = split_xc(interventions)
        eta, beta, eps, kap = apply_chronic(control_theta, ivs)
        return float(fp_loglik_physical(cond_dt, cond_ev, eta, beta, eps, xc * xc_scale, kap,
                                        external_hazard=external_hazard))

    ll0 = score([])

    single_cands = [(p, float(e)) for p in params for e in effects]
    pair_cands = []
    if pairs:
        for p1, p2 in itertools.combinations(params, 2):
            for e1 in pair_effects:
                for e2 in pair_effects:
                    pair_cands.append((p1, float(e1), p2, float(e2)))

    def s_single(pe):
        p, e = pe; return score([Intervention(p, e)])

    def s_pair(c):
        p1, e1, p2, e2 = c; return score([Intervention(p1, e1), Intervention(p2, e2)])

    runner = (Parallel(n_jobs=n_jobs, prefer="processes") if _HAVE_JOBLIB else None)
    if runner is not None:
        s_ll = runner(delayed(s_single)(c) for c in single_cands)
        p_ll = runner(delayed(s_pair)(c) for c in pair_cands) if pair_cands else []
    else:
        s_ll = [s_single(c) for c in single_cands]
        p_ll = [s_pair(c) for c in pair_cands]

    singles = pd.DataFrame(single_cands, columns=["param", "effect"])
    singles["ll"] = s_ll; singles["ll_gain"] = singles["ll"] - ll0
    best_single = singles.loc[singles["ll"].idxmax()].to_dict()

    pairs_df = pd.DataFrame(pair_cands, columns=["param1", "effect1", "param2", "effect2"])
    best_pair = None
    if len(pairs_df):
        pairs_df["ll"] = p_ll; pairs_df["ll_gain"] = pairs_df["ll"] - ll0
        best_pair = pairs_df.loc[pairs_df["ll"].idxmax()].to_dict()

    return {"baseline_ll": ll0, "singles": singles, "pairs": pairs_df,
            "best_single": best_single, "best_pair": best_pair}


def posterior_ratio(control_thetas, cond_thetas, params=PARAM_NAMES, seed=0):
    """Distribution of each parameter ratio condition/control from the two FP posteriors.

    Only the FITTED parameters -- ``xc`` has no posterior (it is pinned) and must not be passed here.

    Pairs random draws from the (independent) posteriors and reports the ratio median + 68/95% CI
    per parameter, ranked by |log2(median ratio)| (how far from unity / how much it moved).
    """
    control_thetas = np.asarray(control_thetas, float)
    cond_thetas = np.asarray(cond_thetas, float)
    rng = np.random.default_rng(seed)
    n = min(len(control_thetas), len(cond_thetas))
    c = control_thetas[rng.permutation(len(control_thetas))[:n]]
    t = cond_thetas[rng.permutation(len(cond_thetas))[:n]]
    ratio = t / c                                        # (n, 4)
    rows = []
    for i, p in enumerate(params):
        r = ratio[:, i]
        r = r[np.isfinite(r) & (r > 0)]
        med = float(np.median(r))
        rows.append({
            "param": p, "ratio_median": med,
            "ratio_lo": float(np.percentile(r, 16)), "ratio_hi": float(np.percentile(r, 84)),
            "ratio_lo95": float(np.percentile(r, 2.5)), "ratio_hi95": float(np.percentile(r, 97.5)),
            "log2_ratio": float(np.log2(med)),
            "moved_from_1": abs(float(np.log2(med))),
        })
    df = pd.DataFrame(rows).sort_values("moved_from_1", ascending=False).reset_index(drop=True)
    return df
