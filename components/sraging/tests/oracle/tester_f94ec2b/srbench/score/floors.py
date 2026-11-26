"""Negligibility floors: below them a parameter is not measurable, so "wrong by 1000x" and
"correctly reported as zero" are the same statement and must score the same.

THE UNIT TRAP, which has produced the same bug TWICE in opposite directions.

What was calibrated is a bound on the **dimensionless** ratio:

        rho_beta / rho_eta  <  3.03 / sqrt(n_eff)

Physical `beta/eta` has units of TIME, so a dimensionless coefficient cannot bound it. The physical
floor therefore carries the mean-lifetime factor:

        beta_floor = 3.03 / sqrt(n_eff) * eta * L

Omitting that `L` has cost this project:
  * 926x TOO STRICT in one place, making "negligible" unreachable on any real cohort;
  * a factor of L TOO LENIENT in another, classifying 0 of 2,400 cohorts as negligible against 185.

So the primary entry point here, `negligible_beta`, is **unit-free**: it takes dimensionless
arguments and has no scale factor to get wrong. The physical form exists, is named so it cannot be
mistaken for the dimensionless one, and requires `L` explicitly.

AND THE COEFFICIENT IS NOT A THRESHOLD. 3.03 is one fit across four regimes whose own measured
coefficients are 0.93 / 2.21 / 5.07 / >16.8 -- an 18-fold spread. Against each regime the
production value is up to 3.3x too lenient in the sharpest corner and at least 5.5x too strict in
the most diffuse, and four of six real species sit outside the calibrated noise range entirely. A
verdict within ~3x of the floor is APPROXIMATE, not a test, and `regime_band` says so numerically.
"""
from __future__ import annotations

import math

#: Production coefficients, from the calibration.
BETA_COEF = 3.03
KAPPA_COEF = 0.364

#: Real data is not clean enough for the ideal 1/sqrt(n) scaling to continue indefinitely.
N_EFF_CAP = 2000

#: The four calibrated regimes' own measured coefficients. The spread is the point.
REGIME_COEFS = {
    "sharp": 0.93,        # rho_eps 0.10, kappa 0.05
    "mid": 2.21,          # rho_eps 0.30, kappa 0.15
    "kappa_large": 5.07,  # rho_eps 0.30, kappa 1.00
    "diffuse": 16.8,      # rho_eps 0.50, kappa 0.50, right-censored -- a LOWER bound
}

#: Parameters with no floor. There is no "negligible eta": it is the damage production rate that
#: makes the cohort die at all. An earlier version applied beta's coefficient to eta and reported
#: eta = 0.0033 as negligible against a floor of 0.152, on real data.
#:
#: `external_hazard` is here DELIBERATELY and not by omission: no negligibility bound has been
#: calibrated for a Makeham rate, and inventing one by analogy with beta's is exactly the
#: substitution that lost the scale factor twice. Listed explicitly so `floor_for` can refuse an
#: unknown name instead of silently returning None.
UNFLOORED = ("eta", "epsilon", "external_hazard")


def n_eff(n: int) -> int:
    return min(int(n), N_EFF_CAP)


# --------------------------------------------------------------------------- dimensionless

def beta_floor_dimensionless(n: int, coef: float = BETA_COEF) -> float:
    """The bound on `rho_beta / rho_eta`. This is the quantity that was actually calibrated."""
    return coef / math.sqrt(n_eff(n))


def negligible_beta(n: int, rho_beta: float, rho_eta: float, coef: float = BETA_COEF) -> bool:
    """THE UNIT-FREE ENTRY POINT. Prefer it: it has no scale factor to omit.

    Takes the dimensionless parameters and compares their ratio directly, so the dimensional
    mistake that has happened twice is not expressible here.
    """
    if rho_eta <= 0:
        return False
    return (rho_beta / rho_eta) < beta_floor_dimensionless(n, coef)


def kappa_floor(n: int, coef: float = KAPPA_COEF) -> float:
    """kappa is already dimensionless: no scale factor, no ratio."""
    return coef / math.sqrt(n_eff(n))


def negligible_kappa(n: int, kappa: float, coef: float = KAPPA_COEF) -> bool:
    return kappa < kappa_floor(n, coef)


# --------------------------------------------------------------------------- physical

def beta_floor_physical(n: int, eta: float, L: float, coef: float = BETA_COEF) -> float:
    """The floor on PHYSICAL beta. `L` is required and has no default.

    Naming and signature are both deliberate: `beta_floor_physical(n, eta, L)` cannot be called by
    someone who thinks the floor is dimensionless, because it will not run without the lifetime.
    """
    if L <= 0:
        raise ValueError("beta_floor_physical requires a positive mean lifetime L; the floor "
                         "bounds a dimensionless ratio, so the physical form needs the scale.")
    return beta_floor_dimensionless(n, coef) * eta * L


# --------------------------------------------------------------------------- honesty about it

def regime_band(n: int) -> tuple[float, float]:
    """The plausible range of the beta floor at this n, across the four calibrated regimes.

    Reported alongside any negligibility verdict. The production coefficient sits inside a band
    spanning a factor of 18, so a verdict within about 3x of the floor is not a test.
    """
    lo = min(REGIME_COEFS.values()) / math.sqrt(n_eff(n))
    hi = max(REGIME_COEFS.values()) / math.sqrt(n_eff(n))
    return lo, hi


def beta_verdict(n: int, rho_beta: float, rho_eta: float) -> str:
    """Three-way, never two-way: `negligible` / `detectable` / `unresolved`.

    The middle category exists because the coefficient's own 18-fold uncertainty makes a binary
    verdict near the floor a statement the calibration does not support.
    """
    if rho_eta <= 0:
        return "unresolved"
    ratio = rho_beta / rho_eta
    lo, hi = regime_band(n)
    if ratio < lo:
        return "negligible"
    if ratio > hi:
        return "detectable"
    return "unresolved"


def clamp(value: float, floor: float | None) -> float:
    """Clamp a value to its floor, or pass it through when the parameter has none.

    Applied to BOTH the estimate and the truth before any error is computed, so that correctly
    reporting a negligible parameter scores exactly zero rather than a huge error against an
    unmeasurable truth.
    """
    if floor is None:
        return value
    return max(value, floor)


def cohort_floor(parameter: str, spec) -> float | None:
    """The PHYSICAL floor for one cohort, from its own TRUTH scale (`spec.rho_eta`, `spec.L_phys`).

    Both methods in a comparison are scored against the same cohort's floor, so the floor is fixed
    across what is compared (the methods). It is NOT fixed across sample size -- comparing accuracy
    across n at each n's own floor is what G5 guards against, and that is a different comparison.
    """
    return floor_for(parameter, int(spec.n), rho_eta=float(spec.rho_eta), L=float(spec.L_phys))


def cohort_floors(specs_by_id, parameter: str) -> dict[str, float | None]:
    """{cohort_id: floor} for `claims.accuracy` / `verdict.accuracy_axis`.

    WHY THIS EXISTS (2026-09-25). Both callers of the accuracy axis passed `floors={}`, so the axis
    labelled "FIXED floor" clamped nothing: an unmeasurable beta scored wherever each optimiser
    happened to leave it (method ~1e-6 vs reference ~1e-12 against a truth of 1e-3 on the window arm).
    Measured on the mle_profile_jax medium comparison: the pooled beta accuracy "loss" was +1.21 dex
    without the floor and +0.062 dex [-0.083, +0.171] with it. 182 of 350 truths lie below the floor.
    """
    return {cid: cohort_floor(parameter, spec) for cid, spec in specs_by_id.items()}


def floor_for(parameter: str, n: int, *, rho_eta: float | None = None,
              L: float | None = None, physical: bool = True) -> float | None:
    """The floor for one parameter, or None if it has none.

    `physical=True` returns the floor in physical units and therefore requires `rho_eta` and `L`;
    `physical=False` returns the dimensionless one. There is no mode in which the caller can get a
    physical floor without supplying the scale.
    """
    if parameter in UNFLOORED:
        return None
    if parameter == "kappa":
        return kappa_floor(n)          # dimensionless either way
    if parameter == "beta":
        if not physical:
            return beta_floor_dimensionless(n)
        if rho_eta is None or L is None:
            raise ValueError(
                "a physical beta floor needs rho_eta and L: the calibrated bound is on the "
                "dimensionless ratio rho_beta/rho_eta, so the physical form carries a factor of L."
            )
        eta = rho_eta / (L * L)
        return beta_floor_physical(n, eta, L)
    raise ValueError(
        f"no floor rule for {parameter!r}. Returning None here would make an UNKNOWN parameter "
        f"indistinguishable from a deliberately unfloored one ({UNFLOORED}), which is how a "
        "missing floor goes unnoticed -- add it to UNFLOORED if it genuinely has none."
    )
