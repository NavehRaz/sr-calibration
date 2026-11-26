"""Per-individual log-likelihood terms (spec 005 FR-025; HR-9; data-model "Terms").

    t = per_individual(cohort, params)       # t.terms[i] is row i's term; sum(t.terms) == t.total within 1e-12

The engine is called, never edited: `Evaluator(cohort, numerics=..., grid=...).terms(params)` and `.value(params)`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sraging.engine.evaluator import Evaluator


@dataclass(frozen=True, eq=False)
class Terms:
    """`terms` (one per row of the cohort, read-only), `total` (the engine's value), `numerics`, `grid` (its
    summary)."""

    terms: np.ndarray
    total: float
    numerics: str
    grid: dict


def per_individual(cohort, params, *, numerics="v1", grid=None) -> Terms:
    """Each row's log-likelihood term at `params` from the engine, with the engine's total (FR-025)."""
    ev = Evaluator(cohort, numerics=numerics, grid=grid, fixed={"h_ext": params.h_ext, "xc": params.xc})
    terms, _ = ev.terms(params)
    t = np.array(terms, float)
    t.setflags(write=False)
    return Terms(terms=t, total=ev.value(params), numerics=numerics, grid=ev.grid.summary())
