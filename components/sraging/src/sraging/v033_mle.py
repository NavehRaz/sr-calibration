"""v0.3.3's MLE stage alone, replayed from outside `compat/v033` (spec 003g research R7, Naveh Dd; moved here from
tools/v033_mle.py by spec 006 T038 for profile_v033's own leverage check, research R17).

`compat/v033/estimator.py`'s `fit` has no `stages` argument and must not be edited (constitution VI). Its MLE stage is
built from module-level names (`solver_grid`, `box_for`, `_Objective`, `fp_curvefit_seed`, `_lbfgs`, `_LBFGS_MLE`),
so this replays it call for call, in the same order (the compile calls included), and stops before the profiles.
tests/unit/test_v033_mle.py checks the replay's theta-hat against a real v0.3.3 fit's, bit for bit.
"""
from __future__ import annotations

import math
import time

import numpy as np


def mle_stage(ages, events, delta=None, kappa_max=300.0) -> dict:
    """{theta, loglik, solver, L, dt_ceiling, compile_s, seed_s, mle_s, cpu_s, status}: v0.3.3's `fit` up to and
    including its MLE starts (estimator.py `fit`, lines from `solver_grid` to `theta = list(np.exp(best))`)."""
    from sraging.compat.v033 import estimator as E
    from sraging.compat.v033 import fp_jax as fj
    from sraging.compat.v033.fp_curvefit import fp_curvefit_seed
    c0 = time.process_time()
    t = np.asarray(ages, float)
    e = np.asarray(events, int)
    delta = delta if delta else None
    L, kw, ceiling = E.solver_grid(t, e, delta)
    box = E.box_for(kappa_max)
    tc = time.perf_counter()
    obj = E._Objective(fj, t, e, L, kw, box)
    obj.f(np.zeros(4)).block_until_ready()
    obj.vg(np.zeros(4))[0].block_until_ready()
    compile_s = time.perf_counter() - tc
    ts = time.perf_counter()
    s = fp_curvefit_seed(t, e, xc=1.0)
    seed_theta = [s["eta"], s["beta"], s["epsilon"], s["k"]]
    seed_s = time.perf_counter() - ts
    obj.seed_theta = list(seed_theta)
    starts = [np.log(np.clip(seed_theta, 1e-12, None)),
              np.log(np.clip([seed_theta[0], seed_theta[1], seed_theta[2] * 4.0, seed_theta[3]], 1e-12, None))]
    bounds4 = [(math.log(box[p][0]), math.log(box[p][1])) for p in E.PARAMS]
    best, best_v = None, np.inf
    tm = time.perf_counter()
    for z0 in starts:
        try:
            v, z, _ = E._lbfgs(obj, obj.nll_grad, obj.nll, z0, bounds4, E._LBFGS_MLE,
                               dict(maxiter=1200, xatol=1e-5, fatol=1e-4))
        except Exception:  # noqa: BLE001 -- as fit does
            continue
        if np.isfinite(v) and v < best_v:
            best, best_v = z, v
    mle_s = time.perf_counter() - tm
    base = {"solver": kw, "L": L, "dt_ceiling": ceiling, "compile_s": compile_s, "seed_s": seed_s, "mle_s": mle_s,
            "cpu_s": time.process_time() - c0}
    if best is None:
        return {**base, "status": "all starts failed", "theta": seed_theta, "loglik": float("nan")}
    return {**base, "status": "ok", "theta": [float(x) for x in np.exp(best)], "loglik": float(-best_v)}
