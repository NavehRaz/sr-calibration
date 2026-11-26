"""Fitting BENCHMARK for the SR Fokker-Planck pipeline.

Why this exists
---------------
Until now there was no way to answer "did that change make the fitter better?" other than
eyeballing individual cohorts. This package supplies the missing instrument:

    cohorts.py   a deterministic library of synthetic cohorts with KNOWN ground-truth parameters
                 (plus cohorts carrying PLANTED, LABELLED data defects for the QA flags)
    score.py     scorers: per-parameter recovery error, bound-pinning rate, optimizer/solver
                 sanity, and flag confusion matrices (TPR *and* FPR)
    run.py       runner + frozen baselines; every later change is reported as a diff
    compare.py   baseline-vs-candidate delta table

Design commitments, because they are what make the numbers trustworthy:

1. GROUND TRUTH IS THE PARAMETERS, not a stored array. A cohort is a compact `CohortSpec`
   (parameters + generation instructions + integer seed); `materialize()` regenerates the exact
   same death times anywhere. Nothing large is committed, and reproducibility is by construction.

2. EVERY COHORT IS IN THE tau-mean == 1 CONVENTION. `rho_eta` is solved (brentq) so the FP
   mean first-passage time is exactly 1, which is the convention the pipeline's `L` = KM-area
   mean puts the data in. So the recovered dimensionless parameters are directly comparable to
   the truth without a rescaling step -- the step that is easy to get wrong and which silently
   invalidates a recovery study.

3. THE BENCHMARK MUST REPRODUCE FAILURES WE ALREADY KNOW ARE REAL (see `selftest.py`):
   rho_beta collapsing to its lower bound, a kappa profile reading dead flat, and a single
   extreme individual inflating epsilon. A benchmark that reports everything is fine is not
   measuring the regime we work in.

4. NO ASSUMPTION ABOUT WHICH PARAMETERS ARE IDENTIFIABLE. Recovery error is reported for
   rho_eta, rho_beta, rho_eps and kappa INDIVIDUALLY, alongside the combinations
   (omega, s, bxe), sliced by regime -- so "kappa is unidentifiable" becomes a claim with a
   scope attached rather than a blanket one.
"""
from .cohorts import (
    CohortSpec,
    Cohort,
    solve_rho_eta_for_mean_one,
    solve_rho_eps_for_cv,
    materialize,
    build_library,
    ANCHORS,
    DEFECTS,
)

__all__ = [
    "CohortSpec",
    "Cohort",
    "solve_rho_eta_for_mean_one",
    "solve_rho_eps_for_cv",
    "materialize",
    "build_library",
    "ANCHORS",
    "DEFECTS",
]
