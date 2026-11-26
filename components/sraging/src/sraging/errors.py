"""Errors raised by sraging. Every message names the offending option, field or row (constitution VIII)."""
from __future__ import annotations


class SRAgingError(Exception):
    """Base class of every error sraging raises on purpose."""


class CohortDataError(SRAgingError):
    """A cohort's data were refused. `rows` lists every offending (row number, reason)."""

    def __init__(self, message: str, rows: list[tuple[int, str]] | None = None):
        self.rows = list(rows or [])
        if self.rows:
            detail = "\n".join(f"  row {r}: {why}" for r, why in self.rows)
            message = f"{message}\n{detail}"
        super().__init__(message)


class CohortSpecError(SRAgingError):
    """The cohort's specification is incomplete or contradictory (for example δ vs continuous)."""


class UnsupportedOptionError(SRAgingError):
    """An option this calibrator or spec does not support was requested. `option` names it."""

    def __init__(self, option: str, message: str | None = None):
        self.option = option
        super().__init__(message or f"unsupported option: {option!r}")


class UnknownCalibratorError(SRAgingError):
    """No calibrator of this name/version is registered. `registered` lists what is."""

    def __init__(self, requested: str, registered: list[str]):
        self.requested = requested
        self.registered = list(registered)
        super().__init__(f"no calibrator {requested!r}; registered: {', '.join(self.registered) or '(none)'}")


class UnregisteredCalibratorError(UnknownCalibratorError):
    """A name the frozen package ships but sraging does not register (spec 003b: the six kappa-fixed jax2 names)."""

    def __init__(self, requested: str, registered: list[str], reason: str):
        self.reason = reason
        UnknownCalibratorError.__init__(self, requested, registered)
        self.args = (f"{requested!r} is not registered: {reason}; registered: {', '.join(self.registered)}",)

    def __str__(self):
        return self.args[0]


class RuntimeConfigError(SRAgingError):
    """The numerical runtime is not configured as the calibrator requires (x64, XLA threads)."""
