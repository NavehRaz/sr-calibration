"""Curves for a parameter set without a cohort: S, the density f and the hazard f/S at requested ages
(User Story 6; spec FR-009; research R9 "model" policy).

The "model" grid policy needs no data: the time unit L is the model's own mean lifetime, found as SRK's
fp_curves.auto_L finds it -- probe a coarse march (N = 300, dt = 4e-3, tau_max = 20, Gtol = 1e-6) at L, and
grow L x4 until the probe has absorbed (S_end < 1e-3), then L = L * (the probe's mean in units of L); then
the standard grid (N = 900, dt = 1e-3, tau_max = 20, Gtol = 1e-11), as SRK's fpt_full_density defaults.
The curves are the observed ones: with an external hazard, S_obs = S e^{-H} and f_obs = (f + h S) e^{-H};
with heterogeneity, the mixture; with segments, the general march.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import jax.numpy as jnp

from sraging.engine import stepper as ST
from sraging.engine.grid import Grid, GridPolicy
from sraging.errors import UnsupportedOptionError

PROBE = dict(N=300, dt=4e-3, tau_max=20.0, Gtol=1e-6)
STANDARD = dict(N=900, dt=1e-3, tau_max=20.0, Gtol=1e-11)


@dataclass(frozen=True)
class Curves:
    ages: np.ndarray
    S: np.ndarray
    f: np.ndarray
    hazard: np.ndarray
    grid: Grid


def model_L(params, *, scheme="be", max_tries=12, t_ref=None) -> float:
    """SRK auto_L's rule with the engine's march (no data). `t_ref`: an optional first guess of L."""
    nst = int(PROBE["tau_max"] / PROBE["dt"])
    spec = ST.MarchSpec(N=PROBE["N"], nsteps=nst, dt=PROBE["dt"],
                        nstart=20 if scheme == "bdf2" else nst + 2, Gtol=PROBE["Gtol"])
    march = ST.make_march(spec, "value")
    N, h = PROBE["N"], 1.0 / PROBE["N"]
    yf = (np.arange(N) + 0.5) * h
    L = float(t_ref) if t_ref else 1.0
    xc = params.xc
    for _ in range(max_tries):
        rh, rb, re, kap = L ** 2 * params.eta / xc, L * params.beta / xc, L * params.epsilon / xc ** 2, params.kappa / xc
        S_raw = np.asarray(march((rh, re, jnp.asarray(rb * yf / (yf + kap)))))
        hit = np.flatnonzero((S_raw < PROBE["Gtol"]) & (np.arange(nst + 1) * PROBE["dt"] > 0.3))
        k = int(hit[0]) if hit.size else nst
        tau = np.arange(k + 1) * PROBE["dt"]
        mean = float(np.trapezoid(S_raw[: k + 1], tau))
        if S_raw[k] < 1e-3 and np.isfinite(mean) and mean > 0:
            return float(L * mean)
        L *= 4.0
    raise UnsupportedOptionError("L", "the model never absorbs within the probe horizon, so no lifetime scale "
                                 "can be inferred; pass a grid with an explicit L")


def model_grid_policy(params, scheme="bdf2") -> GridPolicy:
    L = model_L(params, scheme=scheme)
    return GridPolicy("fixed", N=STANDARD["N"], dt=STANDARD["dt"], tau_max=STANDARD["tau_max"],
                      Gtol=STANDARD["Gtol"], L=L)


def curves(params, ages, *, segments=None, heterogeneity=None, grid=None, numerics="v1", scheme="bdf2",
           segment_values=None) -> Curves:
    from sraging.cohort import Cohort
    from sraging.engine.evaluator import Evaluator
    a = np.asarray(ages, float)
    if a.ndim != 1 or a.size < 1 or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("ages must be a 1-d array of finite values >= 0")
    policy = grid or GridPolicy("model")
    used = model_grid_policy(params, scheme) if policy.name == "model" else policy
    tiny = 1e-12 * max(1.0, float(np.max(a)))
    pos = np.where(a > 0, a, tiny)                              # age 0 is read at a tiny positive age
    coh = Cohort.from_arrays(pos, np.ones(a.size, int), continuous=True, time_unit=params.time_unit)
    ev = Evaluator(coh, numerics=numerics, grid=used, scheme=scheme, segments=segments,
                   heterogeneity=heterogeneity, fixed={"xc": params.xc, "h_ext": params.h_ext})
    _, aux = ev.terms(params, segment_values)
    S = np.asarray(aux["S_i"], float)
    f = np.asarray(aux["mass"], float)                          # the observed density (hazard included)
    if ev.hazard:
        vals = {"h_ext": params.h_ext}
        h = ev._h_rates({**vals, **{"seg:" + s.id: (segment_values or {}).get(s.id, s.value) for s in segments}}) \
            if (segments is not None and ev._asm.hz is not None) else params.h_ext
        S = S * np.asarray(ev._asm._E(jnp.asarray(pos), h))
    with np.errstate(divide="ignore", invalid="ignore"):
        hz = np.where(S > 0, f / S, np.nan)
    g = ev.grid if policy.name != "model" else Grid(**{**ev.grid.__dict__, "policy": "model"})
    ro = lambda x: (x.setflags(write=False), x)[1]              # noqa: E731
    return Curves(ages=ro(a.copy()), S=ro(S), f=ro(f), hazard=ro(hz), grid=g)
