"""Constrained SR refits (a chosen subset of parameters free) + how good the restriction is.

Motivating question: does an intervention need all four parameters, or do a couple of them explain it?
``constrained_fit`` starts from a reference (e.g. control) theta and optimizes ONLY the parameters you
name, pinning the rest -- so "only eta and epsilon changed" becomes a fittable model rather than an
assertion. ``approximation_quality`` then measures the log-likelihood gap between that restricted fit
and an unrestricted one, per individual (so cohorts of different size are comparable).

The FP log-likelihood is a plain sum of per-individual terms, so ``loglik_terms`` exposes them and the
gap's confidence interval comes from resampling those terms -- exact, and with no refitting.

Reuses ``SRK.fp_likelihood`` throughout (including its private ``_interp_S_p`` / ``_LOG_FLOOR``) so the
per-individual terms are bit-identical to ``fp_loglik``; ``loglik_terms`` is asserted against
``fp_loglik_physical`` in ``check_terms``.
"""
from __future__ import annotations

import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
from scipy.optimize import minimize

from SRK.fp_likelihood import (
    fp_loglik_physical, fpt_full_density, to_dimensionless, kaplan_meier,
    _interp_S_p, _LOG_FLOOR,
)
from .metrics import PARAM_NAMES, PARAM_IDX, SWEEP_NAMES

# fp_loglik's own solver defaults -- keep in sync so terms match the canonical likelihood exactly.
_FP_KW = dict(N=900, dt=1e-3, tau_max=20.0, Gtol=1e-11)


# --------------------------------------------------------------------------- per-individual terms
def loglik_terms(death_times, events, theta, xc=1.0, L=None, external_hazard=np.inf, **kw):
    """Per-individual log-likelihood contributions (mirrors ``fp_likelihood.fp_loglik``).

    Returns an array aligned with ``death_times``: ``log p(t_i)`` for deaths, ``log S(t_i)`` for
    censored, clipped at ``_LOG_FLOOR``. ``sum()`` equals ``fp_loglik_physical`` (see ``check_terms``).
    Only the untruncated case (``time_range=None``) is supported -- the windowed likelihood is not a
    plain per-individual sum.
    """
    t = np.asarray(death_times, float)
    e = np.ones(t.size, int) if events is None else np.asarray(events, int)
    eta, beta, epsilon, kappa = (float(v) for v in theta)
    if L is None:
        _, _, L = kaplan_meier(t, e)
    if not (np.isfinite([eta, beta, epsilon, kappa, L]).all() and L > 0 and epsilon > 0 and kappa > 0):
        return np.full(t.size, _LOG_FLOOR)

    solver = {**_FP_KW, **kw}
    rho_beta, rho_eta, rho_eps, kap = to_dimensionless(eta, beta, epsilon, xc, L, k=kappa)
    tau, S, p, _mean, _cv, _s_end = fpt_full_density(rho_eta, rho_beta, rho_eps, kap, **solver)
    S_i, p_i = _interp_S_p(t / L, tau, S, p)
    S_phys, p_phys = S_i, p_i / L
    h = 0.0 if not np.isfinite(external_hazard) else float(np.exp(-external_hazard))
    if h > 0.0:
        emht = np.exp(-h * t)
        p_phys = (p_phys + h * S_phys) * emht
        S_phys = S_phys * emht

    died = e == 1
    terms = np.empty(t.size, float)
    terms[died] = np.maximum(np.log(np.clip(p_phys[died], 0.0, None)), _LOG_FLOOR)
    terms[~died] = np.maximum(np.log(np.clip(S_phys[~died], 0.0, None)), _LOG_FLOOR)
    return terms


def check_terms(death_times, events, theta, xc=1.0, L=None, external_hazard=np.inf, tol=1e-6, **kw):
    """Assert loglik_terms sums to fp_loglik_physical. Returns (terms_sum, canonical, ok)."""
    s = float(loglik_terms(death_times, events, theta, xc=xc, L=L,
                           external_hazard=external_hazard, **kw).sum())
    eta, beta, epsilon, kappa = (float(v) for v in theta)
    ref = float(fp_loglik_physical(death_times, events, eta, beta, epsilon, xc, kappa,
                                   L=L, external_hazard=external_hazard, **{**_FP_KW, **kw}))
    return s, ref, bool(abs(s - ref) <= tol * max(1.0, abs(ref)))


# --------------------------------------------------------------------------- constrained fitting
def constrained_fit(theta0, death_times, events, free=("eta", "epsilon"), xc=1.0, L=None,
                    external_hazard=np.inf, ci=True, bounds=None,
                    maxiter=400, xatol=1e-4, fatol=1e-3, restarts=5, **kw):
    """Fit only the named parameters, starting from (and pinning the rest to) ``theta0``.

    Optimizes in log-space (all SR params are positive) with Nelder-Mead. ``free`` is generic:
    ``('eta','epsilon')``, ``('eta',)``, ``('beta','kappa')`` all work.

    ``free`` may include ``'xc'``. Globally xc is non-identifiable -- (xc, eta, beta, eps, kappa) ->
    (c*xc, c*eta, c*beta, c^2*eps, c*kappa) leaves the curve invariant -- but that direction needs ALL
    of eta, beta, eps, kappa to move, so inside a restricted set such as ('eta','xc') or
    ('epsilon','xc'), where the others are pinned, the pair IS identifiable. The fitted xc is returned
    as ``out['xc']`` (it is not part of ``theta``).

    bounds : dict {param: (lo, hi)}, optional
        Box constraints on free parameters, e.g. ``{'kappa': (0.01, 1.0)}`` to stay inside the MCMC
        prior support. Enforced by rejecting out-of-box proposals.
    restarts : int
        Extra Nelder-Mead runs, each re-seeded from the previous optimum with a fresh simplex, kept only
        if they improve. Nelder-Mead stalls on a collapsed simplex and reports ``success=True`` while
        doing so -- measured here as a restricted fit landing 774 nats below the same cohort's fit from
        a different (equivalent) starting control. Restarting from the stalled point recovers it. Set 0
        to reproduce the single-run behaviour.

    Returns dict: theta (full 4-vector), loglik, nfev, free, ci95 ({param: (lo, hi)} or None),
    converged. ``ci95`` is the asymptotic interval from the numerical Hessian of the NLL at the
    optimum (delta-method in log-space, exponentiated back).
    """
    theta0 = np.asarray(theta0, float).copy()
    free = tuple(free)
    unknown = [p for p in free if p not in SWEEP_NAMES]
    if unknown:
        raise KeyError(f"unknown free param(s) {unknown}; expected from {SWEEP_NAMES}")
    theta_free = [p for p in free if p != "xc"]
    fit_xc = "xc" in free                     # xc is not part of theta: it is the pinned threshold,
    idx = [PARAM_IDX[p] for p in theta_free]   # fittable only when the others are held (see note below)
    t = np.asarray(death_times, float)
    e = np.ones(t.size, int) if events is None else np.asarray(events, int)
    if L is None:
        _, _, L = kaplan_meier(t, e)
    solver = {**_FP_KW, **kw}
    box = {p: bounds[p] for p in free if bounds and p in bounds}

    def unpack(z):
        th = theta0.copy()
        if idx:
            th[idx] = np.exp(z[:len(idx)])
        xcv = float(np.exp(z[len(idx)])) if fit_xc else xc
        return th, xcv

    def theta_of(z):
        return unpack(z)[0]

    def nll(z):
        th, xcv = unpack(z)
        for p, (lo, hi) in box.items():
            v = xcv if p == "xc" else th[PARAM_IDX[p]]
            if v < lo or v > hi:
                return np.inf
        val = fp_loglik_physical(t, e, th[0], th[1], th[2], xcv, th[3],
                                 L=L, external_hazard=external_hazard, **solver)
        return np.inf if not np.isfinite(val) else -val

    z0 = np.log(np.concatenate([theta0[idx], [xc]]) if fit_xc else theta0[idx])
    opts = dict(maxiter=maxiter, xatol=xatol, fatol=fatol)
    res = minimize(nll, z0, method="Nelder-Mead", options=opts)
    nfev = int(res.nfev)
    for _ in range(int(restarts)):
        again = minimize(nll, res.x, method="Nelder-Mead", options=opts)
        nfev += int(again.nfev)
        if not (again.fun < res.fun - 1e-9):
            break
        res = again
    theta, xc_fit = unpack(res.x)
    out = {"theta": theta, "xc": xc_fit, "loglik": float(-res.fun), "nfev": nfev,
           "free": free, "converged": bool(res.success), "ci95": None}

    if ci:
        z_names = tuple(theta_free) + (("xc",) if fit_xc else ())   # must match the z ordering
        out["ci95"] = _hessian_ci95(nll, res.x, z_names)
    return out


def _hessian_ci95(nll, z, free, rel_step=1e-3):
    """Asymptotic 95% CI per free param from the numerical Hessian of the NLL in log-space."""
    k = len(z)
    h = np.maximum(np.abs(z) * rel_step, rel_step)
    H = np.zeros((k, k))
    f0 = nll(z)
    for i in range(k):
        for j in range(i, k):
            if i == j:
                zp, zm = z.copy(), z.copy()
                zp[i] += h[i]; zm[i] -= h[i]
                H[i, i] = (nll(zp) - 2 * f0 + nll(zm)) / (h[i] ** 2)
            else:
                zpp, zpm, zmp, zmm = (z.copy() for _ in range(4))
                zpp[i] += h[i]; zpp[j] += h[j]
                zpm[i] += h[i]; zpm[j] -= h[j]
                zmp[i] -= h[i]; zmp[j] += h[j]
                zmm[i] -= h[i]; zmm[j] -= h[j]
                H[i, j] = H[j, i] = (nll(zpp) - nll(zpm) - nll(zmp) + nll(zmm)) / (4 * h[i] * h[j])
    try:
        cov = np.linalg.inv(H)
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(sd)):
        return None
    return {p: (float(np.exp(z[i] - 1.96 * sd[i])), float(np.exp(z[i] + 1.96 * sd[i])))
            for i, p in enumerate(free)}


# --------------------------------------------------------------------------- restriction quality
def approximation_quality(death_times, events, theta_restricted, theta_full, xc=1.0, L=None,
                          external_hazard=np.inf, n_boot=2000, seed=0,
                          xc_restricted=None, xc_full=None, **kw):
    """How much log-likelihood the restricted model gives up vs the unrestricted one, per individual.

    ``dll_per_ind = (LL_full - LL_restricted) / n``. ~0 means the restriction costs nothing.
    Pass ``xc_restricted``/``xc_full`` when the two models use different thresholds (e.g. a model that
    fits xc against an unrestricted reference at xc=1). Because
    the likelihood is a sum of per-individual terms, the CI comes from resampling individuals in the
    paired term vectors -- exact and refit-free.

    Returns dict: ll_restricted, ll_full, dll, n, dll_per_ind, ci95 (tuple), frac_better.
    """
    t = np.asarray(death_times, float)
    e = np.ones(t.size, int) if events is None else np.asarray(events, int)
    if L is None:
        _, _, L = kaplan_meier(t, e)
    xcr = xc if xc_restricted is None else xc_restricted
    xcf = xc if xc_full is None else xc_full
    a = loglik_terms(t, e, theta_restricted, xc=xcr, L=L, external_hazard=external_hazard, **kw)
    b = loglik_terms(t, e, theta_full, xc=xcf, L=L, external_hazard=external_hazard, **kw)
    d = b - a                                            # per-individual gain of the full model
    n = t.size
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1) if n_boot else np.array([])
    ci = ((float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
          if boot.size else (float("nan"), float("nan")))
    return {"ll_restricted": float(a.sum()), "ll_full": float(b.sum()),
            "dll": float(d.sum()), "n": int(n), "dll_per_ind": float(d.mean()),
            "ci95": ci, "frac_better": float((d > 0).mean())}
