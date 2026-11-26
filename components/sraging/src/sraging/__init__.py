"""sraging: the SR (saturating-removal) ageing model.

Public API (contracts/python-api.md): SRParams, Cohort, FitResult, fit_cohort, registry, divergences, and
the errors. Anything under `sraging.compat` is legacy, verbatim numerics -- not an API.
"""
from __future__ import annotations

__version__ = "0.1.0"

from sraging import errors, registry
from sraging.cohort import Cohort
from sraging.errors import (CohortDataError, CohortSpecError, RuntimeConfigError, SRAgingError,
                            UnknownCalibratorError, UnsupportedOptionError)
from sraging.params import SRParams
from sraging.result import FitResult


NO_COMBINATIONS_NOTE = ("combination claims are given only by the jax2 calibrators, whose combination claims were "
                        "truth-scored at 1.0.0 (profile_jax2 1.0.0, profile_jax2_h 1.0.0; spec 003) and are not on "
                        "1.0.1 (a stated limit, spec 003g); {name} gives none (spec 003 FR-018)")


def fit_cohort(cohort: Cohort, *, calibrator: str = registry.DEFAULT, version: str | None = None,
               combinations: str | bool | None = None) -> FitResult:
    """Fit one cohort with the named calibrator (default `profile_jax2` 1.0.1 since spec 003g; `profile_v033`, the
    v0.3.3 reference, by name). `combinations`
    (jax2 calibrators): "lead" (their default, the frozen lead group), "all" or False. A calibrator that gives no
    combination claims ignores it, and the result's diagnostics say so (FR-018)."""
    if not isinstance(cohort, Cohort):
        raise TypeError(f"fit_cohort needs a sraging.Cohort, got {type(cohort).__name__}")
    if "@" in calibrator:                         # "name@version", as the CLI takes it (spec 003b)
        calibrator, at_version = calibrator.split("@", 1)
        if version is not None and version != at_version:
            raise ValueError(f"the version is given twice: {at_version!r} in the name and {version!r}")
        version = at_version
    spec = registry.get(calibrator, version)
    if combinations is None:
        return spec.fit(cohort)
    import inspect
    if "combinations" not in inspect.signature(spec.fit).parameters:
        r = spec.fit(cohort)
        r.diagnostics.setdefault("notes", []).append(
            f"combinations={combinations!r} ignored: " + NO_COMBINATIONS_NOTE.format(name=spec.name))
        return r
    return spec.fit(cohort, combinations=combinations)


def __getattr__(name):
    if name == "divergences":                       # loaded on demand (reads divergences/register.toml)
        import importlib
        return importlib.import_module("sraging.divergences")
    raise AttributeError(f"module 'sraging' has no attribute {name!r}")


__all__ = ["SRParams", "Cohort", "FitResult", "fit_cohort", "registry", "divergences", "errors",
           "SRAgingError", "CohortDataError", "CohortSpecError", "UnsupportedOptionError",
           "UnknownCalibratorError", "RuntimeConfigError", "__version__"]
