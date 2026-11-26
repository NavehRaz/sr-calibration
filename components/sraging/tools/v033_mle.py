"""v0.3.3's MLE stage alone, replayed from outside `compat/v033` (spec 003g research R7, Naveh Dd; tasks T028).

`compat/v033/estimator.py`'s `fit` has no `stages` argument and must not be edited (constitution VI). Its MLE stage is
built from module-level names (`solver_grid`, `box_for`, `_Objective`, `fp_curvefit_seed`, `_lbfgs`, `_LBFGS_MLE`),
so this replays it call for call, in the same order (the compile calls included), and stops before the profiles.
`v033-mle-verify` checks the replay's theta-hat against a real v0.3.3 fit's, bit for bit, side by side on this Mac.
Spec 006 (T038) moved `mle_stage` to `sraging.v033_mle`, re-exported here; the twin `full_fit_theta` stays.
"""
from __future__ import annotations

import time

from sraging.v033_mle import mle_stage  # noqa: F401  (spec 006 T038: moved to src/)


def full_fit_theta(ages, events, delta=None) -> dict:
    """A real v0.3.3 fit's theta-hat (the MLE stage's, which `fit` returns as `theta`) and its timers: the twin the
    replay must equal."""
    from sraging.compat.v033 import estimator as E
    from sraging.compat.v033 import fp_jax as fj
    c0 = time.process_time()
    res = E.fit(fj, ages, events, delta)
    return {"theta": [float(x) for x in res["theta"]], "loglik": float(res["loglik"]),
            "cpu_s": time.process_time() - c0}
