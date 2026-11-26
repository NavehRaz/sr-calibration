"""Turning a frozen cohort into what a method actually sees.

TWO JOBS, and the second is the reason this module exists at all.

1. STRIP THE TRUTH. A method receives ages, events, the recording resolution and the window --
   and nothing from which the planted parameters could be recovered. The narrowing is structural:
   `Observation` has no field for truth, so there is nowhere for it to leak.

2. APPLY THE WINDOW. This is implemented HERE, in the harness, because nothing upstream
   implements it. Measured 2026-09-22: `cohorts.materialize` never reads `time_range`, so a
   cohort asked for with `time_range=(20,150)` comes back BIT-IDENTICAL to a pure one, and the
   shipped `var_trunc` variant would emit pure-SR cohorts wearing a "truncated" label.

   That is the same failure class that previously cost this project a full re-run of 17
   configurations: `fp_model` silently discarded `time_range` for nine days while most configs
   scored the full curve. A silent no-op is the worst available outcome, so the guard against it
   here is a test asserting that a windowed observation is NEVER bit-identical to its source
   unless the window covers everything.

WHAT A WINDOW MEANS, precisely. The cohort is observed only inside `[lo, hi]` -- CLOSED at both
ends:

  * individuals who died before `lo` are ABSENT from the sample entirely (left truncation -- they
    were never observed, so they are not censored records, they are not there);
  * individuals still alive AFTER `hi` appear as RIGHT-CENSORED at `hi` (administrative
    censoring);
  * a death recorded exactly AT `hi` stays a DEATH. It was observed, so recording it as censored
    would discard an observed event and bias the likelihood toward longer survival;
  * everyone else is unchanged.

That last bullet is not a corner case, and this interval read `[lo, hi)` until 2026-10-12. The
half-open notation was always wrong about the code -- `apply_window` censors on `ages > hi`,
never `>=` -- and taking it literally would have converted 27,083 deaths into censored records,
across 213 of the 345 on-grid windowed cohorts in library v5 [measured 2026-10-12]. The count is
that large because ages are recorded on the delta grid and most window edges are grid points too,
so landing exactly on `hi` is common rather than a tie-breaking rarity. The sraging generator
session found the contradiction while implementing these semantics; the CODE was right and the
notation was not, so the notation moved.

The truth is untouched. Only what is observed changes -- which is exactly what makes the
`pure` vs `window` contrast a measurement of the observation model rather than of the parameters.
"""
from __future__ import annotations

import math

import numpy as np

from ..contract.types import Observation
from .spec import CohortSpec


def apply_window(ages, events, window: tuple[float, float] | None):
    """Return (ages, events) as observed through `window`. Pure function, no randomness.

    Deliberately total: with `window=None` it returns the inputs unchanged, so the caller has no
    branch to forget.
    """
    ages = np.asarray(ages, dtype=float)
    events = np.asarray(events, dtype=int)
    if window is None:
        return ages, events

    lo, hi = float(window[0]), float(window[1])

    # Left truncation: never observed, so removed -- NOT recorded as censored at lo. Recording
    # them as censored would assert they were known to be alive at lo, which is a different (and
    # false) statement about the sampling.
    keep = ages >= lo
    ages, events = ages[keep], events[keep]

    # Administrative right-censoring at the close of the window.
    past = ages > hi
    ages = np.where(past, hi, ages)
    events = np.where(past, 0, events)
    return ages, events


def known_for(spec: CohortSpec, reveal) -> dict | None:
    """The true values of the features a method ASKED to be told, and nothing else.

    `reveal` is the method's `descriptor.knows`. Empty for every method that does not opt in, so
    this returns None and the observation is byte-identical to what it has always been -- which is
    what keeps the whole result store valid.

    Only observation-model levels are ever revealed. The parameters a method is SCORED on stay
    invisible, so this is not a hole in the no-truth rule: being told that heterogeneity is 0.2 and
    then recovering eta is a harder, more honest test than being told eta.
    """
    if not reveal:
        return None
    out: dict = {}
    if "heterogeneity" in reveal and spec.het_param is not None:
        out["heterogeneity"] = {"param": str(spec.het_param), "var": float(spec.het_var)}
    if "censoring" in reveal and spec.censor_frac:
        out["censoring"] = {"fraction": float(spec.censor_frac)}
    if "external_hazard" in reveal and spec.external_hazard is not None:
        # Revealed as the RATE, the same convention `physical_truth` scores against -- telling a
        # method a log-level while scoring it on a rate is a unit trap waiting to happen.
        out["external_hazard"] = {"rate": math.exp(-float(spec.external_hazard))}
    if "fit_window" in reveal and spec.window is not None:
        out["fit_window"] = {"lo": float(spec.window[0]), "hi": float(spec.window[1])}
    return out or None


def observation_for(spec: CohortSpec, ages, events, reveal=()) -> Observation:
    """Build the `Observation` a method is shown for this cohort.

    Data-level features (hazard, heterogeneity, censoring) are already present in the frozen
    arrays and need no transform here. Only the window is presentation-level.
    """
    a, e = apply_window(ages, events, spec.window)
    return Observation(
        cohort_id=spec.cohort_id,
        ages=tuple(float(x) for x in a),
        events=tuple(int(x) for x in e),
        delta=spec.delta,
        window=spec.window,
        known=known_for(spec, reveal),
    )


def window_is_effective(spec: CohortSpec, ages, events) -> bool:
    """Did the window actually change anything?

    This is the guard against the upstream no-op, promoted to a callable so the build can use it
    as well as the tests. A cohort in the `window` or `realfull` arm whose window changes nothing
    is mislabelled, and the build fails rather than shipping it.
    """
    if spec.window is None:
        return False
    a0 = np.asarray(ages, dtype=float)
    e0 = np.asarray(events, dtype=int)
    a1, e1 = apply_window(a0, e0, spec.window)
    if len(a1) != len(a0):
        return True
    return not (np.array_equal(a0, a1) and np.array_equal(e0, e1))


def window_retention(spec: CohortSpec, ages) -> float:
    """Fraction of individuals surviving left truncation. Used at build time: a window that
    leaves almost nobody is not a usable cohort, and shipping it would put a tiny-n result into
    an arm average without anything saying so."""
    if spec.window is None:
        return 1.0
    a = np.asarray(ages, dtype=float)
    return float((a >= spec.window[0]).mean()) if len(a) else 0.0
