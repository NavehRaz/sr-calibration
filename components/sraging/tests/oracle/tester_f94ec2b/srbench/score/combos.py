"""Derived parameter combinations — THE SINGLE DEFINITION SITE.

The four raw parameters are not what a survival curve determines. What it determines is some
*function* of them, so all ten candidates are scored rather than one being assumed. With the death
threshold at 1, every `beta/...` entry is a rate and every `beta/(...*eta)` entry is a TIME: the
age at which linearly growing production overtakes that measure of repair.

WHY THIS FILE IS THE ONLY PLACE THEY ARE WRITTEN DOWN. A hand-maintained parallel list has already
drifted silently once in this project's history: five combinations were scored by the reporting
code from the start while the estimator never profiled any of them, so every combination column
rendered empty for an entire study and nobody noticed. Here the formulas, their inverses, their
floors and the report's column list are all generated from `COMBOS` below -- there is nothing to
keep in step by hand.

THE INVERSES ARE LOAD-BEARING AND EASY TO GET WRONG. Profiling a combination works by fixing its
value and solving ONE raw parameter for it, so the method rests entirely on the inverse being
exact. `beta = g*kappa` where `beta = g/kappa` was meant produces a smooth, plausible, completely
wrong interval, and nothing downstream notices: the walk converges, the deviance rises, the
threshold is crossed, a number is reported. `tests/unit/test_combos.py` round-trips every one of
them after moving the other three parameters along the surface -- checking at a single point would
pass for any self-consistent inverse.
"""
from __future__ import annotations

import math

#: Index of each raw parameter in a theta vector.
IDX = {"eta": 0, "beta": 1, "epsilon": 2, "kappa": 3}


def removal_mean_factor(kappa: float) -> float:
    """`1 - kappa*ln(1 + 1/kappa)` -- the MEAN of the removal shape over the damage range.

    It is the integral `(1/xc) * int_0^xc X/(X+kappa) dX` at xc = 1, and that identity is what
    justifies scoring the combination at all. A sign slip or a mis-transcribed simplification
    would still produce a smooth, plausible curve with plausible limits, so the test checks this
    against adaptive QUADRATURE of the integrand rather than against a rearrangement of the same
    closed form -- which would only re-test the transcription against itself.

    Limits: -> 1 as kappa -> 0 (removal saturated along the whole path);
            -> 1/(2*kappa) as kappa -> inf (linear regime; the mean of X over [0,1] is 1/2).
    """
    if kappa <= 0:
        return 1.0
    if kappa < 1e4:
        return 1.0 - kappa * math.log1p(1.0 / kappa)
    # Series for large kappa: log1p(1/k) loses precision as 1/k -> 0, and the difference of two
    # nearly-equal numbers below it is where the accuracy actually goes.
    u = 1.0 / kappa
    return u * (0.5 - u * (1.0 / 3.0 - u * 0.25))


def _inv_removal_mean(g_over_beta: float) -> float:  # pragma: no cover - not needed for scoring
    raise NotImplementedError("kappa is not solved for; combinations solve for beta or epsilon")


#: name -> (formula, which raw parameter the profile solves for, inverse, contains_beta)
#:
#: `inv(g, theta)` returns the value of the solved parameter that makes the combination equal `g`,
#: given the OTHER three. Every inverse here is exact and algebraic.
COMBOS: dict[str, dict] = {
    # --- no beta: unclamped, because the floor is a statement about beta ---
    "eta_over_eps": {
        "formula": lambda t: t[0] / t[2],
        "solve": "eta",
        "inv": lambda g, t: g * t[2],
        "has_beta": False,
        "reading": "production over noise",
    },
    "sqrt_eta_over_eps": {
        "formula": lambda t: math.sqrt(t[0]) / t[2],
        "solve": "eta",
        "inv": lambda g, t: (g * t[2]) ** 2,
        "has_beta": False,
        "reading": "the `s` combination",
    },
    # --- rates ---
    "beta_over_eps": {
        "formula": lambda t: t[1] / t[2],
        "solve": "beta",
        "inv": lambda g, t: g * t[2],
        "has_beta": True,
        "reading": "removal over noise",
    },
    "beta_over_kp1": {
        "formula": lambda t: t[1] / (t[3] + 1.0),
        "solve": "beta",
        "inv": lambda g, t: g * (t[3] + 1.0),
        "has_beta": True,
        "reading": "removal AT the threshold",
    },
    "beta_over_k": {
        "formula": lambda t: t[1] / t[3],
        "solve": "beta",
        "inv": lambda g, t: g * t[3],
        "has_beta": True,
        "reading": "removal rate constant in the linear regime",
    },
    "beta_rbar": {
        "formula": lambda t: t[1] * removal_mean_factor(t[3]),
        "solve": "beta",
        "inv": lambda g, t: g / removal_mean_factor(t[3]),
        "has_beta": True,
        "reading": "MEAN removal over the damage range",
    },
    # --- times (rate over production) ---
    "beta_over_eta": {
        "formula": lambda t: t[1] / t[0],
        "solve": "beta",
        "inv": lambda g, t: g * t[0],
        "has_beta": True,
        "reading": "crossover time against beta",
    },
    "beta_over_kp1_eta": {
        "formula": lambda t: t[1] / ((t[3] + 1.0) * t[0]),
        "solve": "beta",
        "inv": lambda g, t: g * (t[3] + 1.0) * t[0],
        "has_beta": True,
        "reading": "crossover against removal at the threshold",
    },
    "beta_over_k_eta": {
        "formula": lambda t: t[1] / (t[3] * t[0]),
        "solve": "beta",
        "inv": lambda g, t: g * t[3] * t[0],
        "has_beta": True,
        "reading": "crossover against the linear-regime constant",
    },
    "beta_rbar_over_eta": {
        "formula": lambda t: t[1] * removal_mean_factor(t[3]) / t[0],
        "solve": "beta",
        "inv": lambda g, t: g * t[0] / removal_mean_factor(t[3]),
        "has_beta": True,
        "reading": "crossover against MEAN removal",
    },
}

#: The scored combination names. Derived, never hand-listed -- see the module docstring.
COMBO_NAMES = tuple(COMBOS)


def combos(theta) -> dict[str, float]:
    """All ten combinations at a physical theta `[eta, beta, epsilon, kappa]`."""
    out = {}
    for name, spec in COMBOS.items():
        try:
            v = spec["formula"](theta)
        except (ZeroDivisionError, ValueError, OverflowError):
            v = float("nan")
        out[name] = v
    return out


def combo_floor(name: str, theta, beta_floor: float | None) -> float | None:
    """The floor for a combination: ONE substitution rule, not ten transcribed formulas.

        floor(g) = g evaluated at the truth, with beta replaced by beta's floor

    Every combination containing beta is monotone in it and vanishes as beta -> 0, so "beta is
    indistinguishable from zero" and "this combination is indistinguishable from zero" are the same
    statement in different units. Generating the floors from the definitions above means they
    cannot drift from them -- which is how the scale factor was lost twice.

    Combinations without beta are UNCLAMPED: the floor is a statement about beta, and there is no
    floor for eta or epsilon.

    NOT PROPAGATED: kappa's own floor. `beta_over_k` would diverge as kappa approaches its floor,
    and a negligible kappa does not make removal negligible -- it makes it SATURATED. Combinations
    are therefore flagged `derived, not independently calibrated`.
    """
    spec = COMBOS[name]
    if not spec["has_beta"] or beta_floor is None:
        return None
    t = list(theta)
    t[IDX["beta"]] = beta_floor
    try:
        return spec["formula"](t)
    except (ZeroDivisionError, ValueError, OverflowError):
        return None


def is_derived_floor(name: str) -> bool:
    """True for every combination floor: they are derived by substitution, not calibrated."""
    return COMBOS[name]["has_beta"]
