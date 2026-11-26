"""A fast, deterministic stand-in calibrator for the fit-set tests (spec 005 tasks T033, T034): seconds per fit, so
spawned workers are exercised without a real calibration. It runs JAX (the engine's value on a coarse fixed grid at
three kappa values) in the worker, picks the best, and returns a FitResult; a cohort whose name starts with "bad"
raises. Importable in a spawned worker because pytest puts this directory on sys.path, which spawn passes on."""
from __future__ import annotations

import time

import numpy as np

from sraging.params import SRParams
from sraging.registry import CalibratorSpec
from sraging.result import FitResult

KAPPAS = (0.03, 0.05, 0.08)


def fit(cohort) -> FitResult:
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.grid import GridPolicy
    if str(cohort.name).startswith("bad"):
        raise ValueError(f"planted failure for cohort {cohort.name!r}")
    t0, c0 = time.time(), time.process_time()
    L = float(np.median(cohort.ages))
    ev = Evaluator(cohort, numerics="v033", grid=GridPolicy("fixed", N=80, dt=5e-3, tau_max=3.0, L=L))
    cands = [SRParams(eta=4e-4 * (50.0 / L) ** 2, beta=0.02 * 50.0 / L, epsilon=4e-3 * 50.0 / L, kappa=k,
                      time_unit=cohort.time_unit) for k in KAPPAS]
    vals = [ev.value(p) for p in cands]
    best = cands[int(np.argmax(vals))]
    return FitResult(status="ok", cohort={"name": cohort.name, "n": cohort.n, "n_deaths": cohort.n_deaths,
                                          "n_censored": cohort.n_censored, "delta": cohort.delta,
                                          "continuous": cohort.continuous},
                     theta_hat=best, theta_labels={},
                     claims=(), loglik=float(max(vals)), warnings=(), checks=(), diagnostics={"candidates": vals},
                     timing={"wall_s": time.time() - t0, "cpu_s": time.process_time() - c0},
                     provenance={"calibrator": {"name": "fast_test", "version": "1"}, "numerics": "test"})


SPEC = CalibratorSpec(name="fast_test", version="1", numerics="test", settings={}, fit=fit)


def describe(carried):
    """In a spawned worker: the module and name of the fit `sraging.fitset._fit_of` resolves `carried` to (the
    registered-calibrator test; a closure cannot be sent back, its names can)."""
    from sraging.fitset import _fit_of
    f = _fit_of(carried)
    return f.__module__, f.__qualname__
