"""Shared helpers for spec 003's side-by-side checks: the jax2 objective built exactly as `estimator.fit` builds
it, from either the port's modules or the oracle's (research R4; tasks T013, T018).

    mods = port_modules()            # or oracle_modules() (tools/oracle_jax2.py)
    obj, info = objective(mods, obs, "jax2")
    v = obj.f(z); vg = obj.vg(z)     # the compiled value and value+gradient programs (no memo)

The steps are fit's own (compat/jax2/estimator.py `fit`, lines after the SRK_DT_CEILING refusal): the observation
model with the name's switches, `_grid` with them, the exact-time switch, the box (with jax2_h's hazard box), and
`_Objective`. Nothing here is a reimplementation of numerics; both sides run the same statements on their own
modules.
"""
from __future__ import annotations

import math
import os
import sys
import tomllib

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_V1 = os.path.join(REPO, "tests", "acceptance", "panel_v1.toml")
PARAMS = ("eta", "beta", "epsilon", "kappa")
SHORT = {"profile_jax2": "jax2", "profile_jax2_h": "jax2_h", "jax2": "jax2", "jax2_h": "jax2_h"}


def port_modules() -> dict:
    from sraging.compat.jax2 import claims, estimator, fp_jax, observation_model, settings
    return {"om": observation_model, "est": estimator, "fj": fp_jax, "claims": claims,
            "classes": {"jax2": settings.MleProfileJax2, "jax2_h": settings.MleProfileJax2H}, "side": "port"}


def oracle_modules() -> dict:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import oracle_jax2 as OJ
    o = OJ.load()
    return {"om": o.adapter._sibling("observation_model"), "est": o.estimator, "fj": o.fp_jax, "claims": o.adapter,
            "classes": o.classes, "side": "oracle"}


def objective(mods, obs, name):
    """(_Objective, info) for `obs` under `name`, as `fit` builds it."""
    cls = mods["classes"][SHORT[name]]
    om, est, fj = mods["om"], mods["est"], mods["fj"]
    prep = om.prepare(obs.ages, obs.events, obs.delta, obs.window, window_on=cls.WINDOW, relabel_on=cls.RELABEL)
    t = np.asarray(prep.ages, float)
    e = np.asarray(prep.events, int)
    delta = obs.delta if obs.delta else None
    L, kw, ceiling, dt_floor = est._grid(t, e, delta, floor_early=cls.FLOOR_EARLY, n_rule=cls.N_RULE,
                                         dt_rule=cls.DT_RULE, **({"early_all": True} if cls.EARLY_ALL else {}))
    if cls.EXACT_F:
        kw = dict(kw, exact_from_F=True)
    box = est.box_for(300.0)
    if cls.HAZARD:
        box = dict(box, h=est._h_box(L))
    obj = est._Objective(fj, t, e, L, kw, box, hazard=cls.HAZARD, window=prep.window_eff)
    return obj, {"L": float(L), "N": kw["N"], "dt": kw["dt"], "binds": bool((dt_floor or {}).get("binds")),
                 "early": kw.get("early"), "box": box}


def theta_points(cohort_id) -> list[tuple[str, dict]]:
    """Spec 001's six frozen theta points of a panel cohort (tests/acceptance/panel_v1.toml)."""
    for t in tomllib.load(open(PANEL_V1, "rb"))["theta"]:
        if t["cohort_id"] == cohort_id:
            return [(pt["label"], {k: pt[k] for k in PARAMS}) for pt in t["points"]]
    raise KeyError(cohort_id)


def panel_ids() -> list[str]:
    p = tomllib.load(open(PANEL_V1, "rb"))
    return list(p["quick"]) + [n["cohort_id"] for n in p["named"]]


def h_points(info) -> list[tuple[str, float]]:
    """research R4: the hazard box's floor, and h such that h*L = 0.1."""
    lo, hi = info["box"]["h"]
    return [("box_floor", float(lo)), ("hL=0.1", float(min(max(0.1 / info["L"], lo), hi)))]


def z_of(theta: dict, h: float | None = None) -> np.ndarray:
    z = [math.log(theta[p]) for p in PARAMS]
    if h is not None:
        z.append(math.log(h))
    return np.asarray(z, float)
