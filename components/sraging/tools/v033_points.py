"""v0.3.3's point from its stored record, and its truth witness in the candidate's form (spec 003g research R6; tasks
T027). No fit: likelihood evaluations only. Read-only on every record.

A v0.3.3 record has no theta: its claims carry the points (theta-hat, clipped to the box), and a claim of shape "none"
has none. The recipe is the tester's basin scope's (reports/basin/scope/oracle_job.py, classify.py): all four claim
points when they exist; else eta, beta, epsilon from the claims and kappa by a 1-D maximisation of v0.3.3's own
likelihood over log kappa on the record's own grid (17-point scan of the box, then a bounded refine of the two best
cells), kept iff it reproduces the stored log-likelihood to within 0.5 nats (recon_dll > -0.5); else no point ("not
shown"). The likelihood is sraging's port of v0.3.3's fp_jax (bit-identical to v0.3.3 on this Mac; spec 001), with
L the Kaplan-Meier mean, as the scope used it.
"""
from __future__ import annotations

import math

import numpy as np

P = ("eta", "beta", "epsilon", "kappa")
KLO, KHI = math.log(1e-3), math.log(300.0)
RECON_DLL_MIN = -0.5


def maximise_logk(ll):
    """The scope's `maximise_logk`, copied: a 17-point scan of log kappa over the box, then a bounded refine of the
    two best cells. (best z, best value, evaluations)."""
    from scipy.optimize import minimize_scalar
    grid = [KLO + (KHI - KLO) * i / 16 for i in range(17)]
    cache = {}

    def g(z):
        z = min(max(float(z), KLO), KHI)
        if z not in cache:
            v = ll(z)
            cache[z] = v if np.isfinite(v) else -np.inf
        return cache[z]

    vals = [g(z) for z in grid]
    order = sorted(range(17), key=lambda i: -vals[i])
    best_z, best_v = grid[order[0]], vals[order[0]]
    for i in order[:2]:
        a, b = grid[max(i - 1, 0)], grid[min(i + 1, 16)]
        r = minimize_scalar(lambda z: -g(z), bounds=(a, b), method="bounded", options=dict(xatol=1e-7, maxiter=60))
        for z in (r.x, a, b):
            if g(z) > best_v:
                best_z, best_v = float(z), g(z)
    return best_z, best_v, len(cache)


def claim_points(claims) -> dict:
    return {c["parameter"]: c.get("point") for c in claims}


def v033_loglik(ages, events, solver: dict):
    """v0.3.3's (value, value_and_grad) programs over z = log(eta, beta, epsilon, kappa) on a record's grid."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from sraging.compat.v033 import fp_jax
    from sraging.compat.v033.fp_likelihood import kaplan_meier
    t, e = np.asarray(ages, float), np.asarray(events, int)
    L = float(kaplan_meier(t, e)[2])
    f, vg = fp_jax.make_loglik(t, e, L, N=solver["N"], dt=solver["dt"], tau_max=solver["tau_max"],
                               Gtol=solver["Gtol"], sampling_dt=solver.get("sampling_dt"))
    return f, vg, L


def recover(claims, solver: dict, loglik_stored: float, ages, events) -> dict:
    """{point, source (claims / recon / recon_failed / no_points), value (v0.3.3's value_and_grad at the point),
    kappa, recon_dll, n_eval, kept}."""
    pts = claim_points(claims)
    if not all(pts.get(p) is not None for p in P[:3]):
        return {"point": None, "source": "no_points", "kept": False}
    f, vg, L = v033_loglik(ages, events, solver)
    if all(pts.get(p) is not None for p in P):
        th = [float(pts[p]) for p in P]
        return {"point": th, "source": "claims", "value": float(vg(np.log(th))[0]), "kappa": th[3], "L": L,
                "kept": True, "recon_dll": None, "n_eval": 1}
    z3 = [math.log(float(pts[p])) for p in P[:3]]
    zk, v, n = maximise_logk(lambda z: float(f(np.array(z3 + [z]))))
    th = [float(pts[p]) for p in P[:3]] + [math.exp(zk)]
    value = float(vg(np.array(z3 + [zk]))[0])
    dll = value - float(loglik_stored)
    kept = dll > RECON_DLL_MIN
    return {"point": th if kept else None, "source": "recon" if kept else "recon_failed", "value": value,
            "kappa": math.exp(zk), "f_value": v, "recon_dll": dll, "n_eval": n, "L": L, "kept": kept}


def witness_j(cohort, truth: dict, point) -> dict:
    """W = ll(truth) - ll(point) in the candidate's likelihood (jax2 1.0.1, `profile_jax2`) on the candidate's own
    grid for the cohort (`profile_jax2.fit_inputs`: the prepared data, N, dt with DT_RULE; equal to jax2's census bit
    for bit, spec 003b), by the value-and-gradient program, as jax2's witness takes W4 (G/PREREG_V6.txt:517). z =
    log(clip(theta, 1e-300)). A human cohort's W_REF (Richardson D_REF) is not computed here (research R3)."""
    from sraging.calibrators import profile_jax2 as PJ
    P1 = PJ.PORTS["1.0.1"]
    inp = PJ.fit_inputs(cohort, "profile_jax2", P1)
    obj = P1.estimator._Objective(P1.fp_jax, inp["t"], inp["e"], inp["L"], inp["kw"], inp["box"],
                                  hazard=inp["hazard"], window=inp["window"])

    def ll(theta):
        z = np.log(np.clip(np.asarray(theta, float), 1e-300, None))
        return -float(obj.nll_grad(z)[0])
    th_true = [float(truth[p]) for p in P]
    l_truth, l_point = ll(th_true), ll(point)
    W = l_truth - l_point
    return {"W4": W, "l_truth": l_truth, "l_point": l_point, "shown": bool(W > 1.92), "grid": {
        k: inp["kw"].get(k) for k in ("N", "dt", "tau_max", "sampling_dt")},
        "dt_in_scope": bool(((inp.get("dt_floor") or {}).get("grid_rule") or {}).get("dt_in_scope"))}
