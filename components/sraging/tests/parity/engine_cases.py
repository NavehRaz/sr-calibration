"""Spec 002 parity helpers: spec 001's 40-cohort panel as sraging Cohorts (read-only tester library v1)."""
import functools
import os
import tomllib

import numpy as np

import oracle_v033 as O
from sraging import Cohort

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANEL = os.path.join(REPO, "tests", "acceptance", "panel_v1.toml")
PARAMS = ("eta", "beta", "epsilon", "kappa")


@functools.lru_cache(maxsize=1)
def panel():
    return tomllib.load(open(PANEL, "rb"))


def panel_ids() -> list[str]:
    p = panel()
    return list(p["quick"]) + [n["cohort_id"] for n in p["named"]]


def theta_points(cohort_id) -> list[tuple[str, dict]]:
    for t in panel()["theta"]:
        if t["cohort_id"] == cohort_id:
            return [(pt["label"], {k: pt[k] for k in PARAMS}) for pt in t["points"]]
    raise KeyError(cohort_id)


@functools.lru_cache(maxsize=None)
def arrays(cohort_id):
    """(ages float64, events int64, delta or None), exactly as spec 001's parity tests build them."""
    obs = O.observation(cohort_id)
    return (np.asarray(list(obs.ages), float), np.asarray(list(obs.events), int),
            obs.delta if obs.delta else None)


def cohort(cohort_id) -> Cohort:
    t, e, d = arrays(cohort_id)
    return Cohort.from_arrays(t, e, delta=d, name=cohort_id)
