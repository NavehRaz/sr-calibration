"""Cohort from a tester `Observation`, without importing srbench (T040; FR-023, register D-104).

Duck-typed on `cohort_id`, `ages`, `events`, `delta`, `window`. Arrays are built as the reference does
(`np.asarray(list(obs.ages), float)`, `np.asarray(list(obs.events), int)`), and delta follows
`estimator.fit`'s `delta if delta else None`: a falsy delta means continuous times. A window is passed into the
Cohort (spec 006, register D-104 revised): profile_jax2_w and profile_jax2_h 1.0.1 fit it by the tester's rule (which
is idempotent on the tester's already-windowed observation); every other calibrator refuses it by name, where the
reference ignores it silently (constitution VIII). A window the Cohort cannot hold (off the delta grid) is refused here.
"""
from __future__ import annotations

from sraging.cohort import Cohort
from sraging.errors import CohortSpecError, UnsupportedOptionError


def cohort_from_observation(obs) -> Cohort:
    window = getattr(obs, "window", None)
    delta = obs.delta if obs.delta else None
    try:
        return Cohort.from_arrays(list(obs.ages), list(obs.events), delta=delta, continuous=delta is None,
                                  name=obs.cohort_id, window=None if window is None else tuple(float(x) for x in window))
    except CohortSpecError as ex:
        if window is None:
            raise
        raise UnsupportedOptionError("window", f"observation {obs.cohort_id!r}'s fit window {window!r} cannot be "
                                               f"fitted: {ex} (register D-104)") from None
