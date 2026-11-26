"""L_SR, the SR model's mean lifetime at a parameter point (spec 003, T036; research R11; constitution X: times are
reported in L_SR, the SR mean, not L_KM).

    L_SR = integral_0^inf S(t) dt

S comes from the spec 002 engine's curves (read-only use, numerics "v1", BDF2) on an explicit grid in the model's own
time scale; the integral is refined by doubling N and halving dt until it moves by less than `rtol` relative.
"""
from __future__ import annotations

import numpy as np

RTOL = 1e-4


def l_sr(params, *, rtol=RTOL, max_refinements=6, n_ages=4001) -> dict:
    """{L_SR, rel_change, refinements, tau_max} for SRParams `params` (h_ext is ignored: the SR mean, no hazard)."""
    import dataclasses
    from sraging.engine.curves import curves, model_L
    from sraging.engine.grid import GridPolicy
    p = dataclasses.replace(params, h_ext=0.0)
    L = model_L(p)
    tau_max, N, dt = 6.0, 450, 4e-3
    prev, out = None, None
    for k in range(max_refinements):
        while True:                                            # extend the horizon until S has died out
            ages = np.linspace(0.0, tau_max * L, n_ages)
            S = curves(p, ages, grid=GridPolicy("fixed", N=N, dt=dt, tau_max=tau_max, L=L)).S
            if S[-1] < 1e-9 or tau_max >= 200.0:
                break
            tau_max *= 2.0
        val = float(np.trapezoid(S, ages))
        out = {"L_SR": val, "rel_change": None if prev is None else abs(val - prev) / val, "refinements": k,
               "tau_max": tau_max, "N": N, "dt": dt, "converged": False}
        if prev is not None and abs(val - prev) <= rtol * val:
            out["converged"] = True
            return out
        prev, N, dt = val, N * 2, dt / 2.0
    return out
