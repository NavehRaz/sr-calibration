"""Analytic FP metrics (longevity + steepness) for the SR kappa-model, with an intervention spec.

An *intervention* is a multiplicative change to one SR parameter, optionally restricted to an age
window. The current evaluator, ``fp_const``, handles the constant-parameter (chronic, whole-life)
case analytically via the Fokker-Planck first-passage solver ``SRK.fp_likelihood.fpt_full`` -- no
SDE simulation. Genuinely time-dependent (windowed) interventions are deferred to a future
``piecewise_fp``/``sim`` evaluator; ``fp_const`` raises a clear error on a windowed intervention so
adding time-dependence later is a new evaluator, not an API change.

Reuses (does not reimplement): ``to_dimensionless`` and ``fpt_full`` from ``SRK.fp_likelihood``.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

from SRK.fp_likelihood import to_dimensionless, fpt_full

PARAM_NAMES = ("eta", "beta", "epsilon", "kappa")     # the FITTED parameters (xc is pinned)
PARAM_IDX = {p: i for i, p in enumerate(PARAM_NAMES)}

# xc can be SWEPT as an intervention (a change in the death threshold) but is never FITTED: with a
# normalized survival curve it is exactly degenerate with a coordinated change of the other four --
# xc -> c is bit-identical to (eta/c, beta/c, epsilon/c^2, kappa/c) at xc=1. So it may appear in
# effect lines and in the forward likelihood scan, but it has no posterior and must never be handed
# to `posterior_ratio`.
SWEEP_NAMES = PARAM_NAMES + ("xc",)


@dataclass(frozen=True)
class Intervention:
    """A multiplicative change to one SR parameter, optionally restricted to a time window.

    Attributes
    ----------
    param : str
        One of ``eta``, ``beta``, ``epsilon``, ``kappa``.
    effect : float
        Fractional multiplicative effect: ``param *= (1 + effect)``.
    window : (float, float) or None
        Age window in which the change is active. ``None`` => whole life (chronic), the only case
        the ``fp_const`` evaluator supports.
    """
    param: str
    effect: float
    window: Optional[Tuple[float, float]] = None

    def is_chronic(self) -> bool:
        return self.window is None

    @property
    def scale(self) -> float:
        return 1.0 + self.effect


def split_xc(interventions):
    """Split interventions into (theta interventions, cumulative xc scale factor).

    ``xc`` is sweepable but not a fitted coordinate, so it is carried as a multiplier on the fixed
    threshold rather than an entry of theta.
    """
    theta_ivs, xc_scale = [], 1.0
    for iv in interventions:
        if iv.param == "xc":
            if not iv.is_chronic():
                raise NotImplementedError(
                    f"fp_const supports only chronic (whole-life) interventions; got windowed {iv!r}.")
            xc_scale *= iv.scale
        else:
            theta_ivs.append(iv)
    return theta_ivs, xc_scale


def apply_chronic(theta, interventions) -> np.ndarray:
    """Return a copy of ``theta`` = [eta, beta, epsilon, kappa] with chronic interventions applied.

    Raises ``NotImplementedError`` on a windowed intervention (needs a time-dependent evaluator).
    ``xc`` is NOT part of theta -- pass such interventions through :func:`split_xc` first.
    """
    th = np.array(theta, dtype=float).copy()
    for iv in interventions:
        if not iv.is_chronic():
            raise NotImplementedError(
                f"fp_const supports only chronic (whole-life) interventions; got windowed {iv!r}. "
                f"Time-dependent interventions need a 'piecewise_fp'/'sim' evaluator (future work).")
        if iv.param == "xc":
            raise KeyError("'xc' is not part of theta (it is pinned, not fitted); route it through "
                           "split_xc() and scale the xc argument instead.")
        if iv.param not in PARAM_IDX:
            raise KeyError(f"unknown param {iv.param!r}; expected one of {SWEEP_NAMES}")
        th[PARAM_IDX[iv.param]] *= iv.scale
    return th


def fp_metrics(theta, interventions=(), xc=1.0, L=1.0, steepness="inverseCV",
               evaluator="fp_const", N=500, dt=1e-3, tau_max=None, Gtol=1e-9):
    """Analytic FP longevity (median) + steepness for an SR theta, with optional interventions.

    Parameters
    ----------
    theta : sequence
        Physical [eta, beta, epsilon, kappa] (xc fixed = ``xc``).
    interventions : iterable of Intervention
        Chronic only for ``evaluator='fp_const'``.
    L : float
        Lifetime rescaling. Physical median = ``L`` * (dimensionless median). Pass a reference
        lifetime (e.g. the control's data mean) so dimensionless tau ~ O(1) and the solver is
        fast/robust; ``L=1`` gives physical-unit tau directly (with xc=1). Cancels in ratios;
        steepness is L-independent.
    steepness : str
        ``'inverseCV'`` -> 1/CV = mean/std (dimensionless).

    Returns
    -------
    dict with keys median, mean, cv, steepness, tail, absorbed.
    """
    if evaluator != "fp_const":
        raise NotImplementedError(f"evaluator={evaluator!r} not implemented (only 'fp_const').")
    interventions, _xc_scale = split_xc(interventions)
    xc = xc * _xc_scale
    eta, beta, epsilon, kappa = apply_chronic(theta, interventions)
    # to_dimensionless returns (rho_beta, rho_eta, rho_eps, kappa) -- beta first (reorder for fpt_full)
    rho_beta, rho_eta, rho_eps, kap = to_dimensionless(eta, beta, epsilon, xc, L, kappa)
    if tau_max is None:
        tau_max = 8.0 if not np.isclose(L, 1.0) else 40.0
    mean_dl, cv, s_end, tl, S = fpt_full(rho_eta, rho_beta, rho_eps, kap,
                                         N=N, dt=dt, tau_max=tau_max, Gtol=Gtol,
                                         return_survival=True)
    S = np.clip(np.asarray(S, float), 0.0, 1.0)
    tl = np.asarray(tl, float)
    absorbed = bool(S[-1] < 1e-2)                       # tail essentially gone -> median reliable
    median = float(L * tl[np.argmin(np.abs(S - 0.5))]) if S.min() <= 0.5 else float("nan")
    cv = float(cv)
    steep = (1.0 / cv) if (steepness == "inverseCV" and np.isfinite(cv) and cv > 0) else float("nan")
    return {"median": median, "mean": float(L * mean_dl), "cv": cv,
            "steepness": steep, "tail": float(s_end), "absorbed": absorbed}


def data_metrics(dataset, steepness="inverseCV", time_range=None):
    """(median, steepness) computed directly from a deathTimesDataSet (model-free)."""
    return {"median": float(dataset.getMedianLifetime()),
            "steepness": float(dataset.getSteepness(method=steepness, time_range=time_range))}
