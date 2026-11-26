"""Labels for weakly determined values (FR-005): every value is shown, with what the data do and do not
support. A label never replaces a value. The κ-box note on raw-β upper ends is spec 003's κ_max = 3000 study
(FR-025, tests/acceptance/results/kmax_003.json): raising κ_max moved the raw-β caps with the box, and the closed
negative controls' upper ends too, so every raw-β upper end carries it (Naveh, 2026-10-06)."""
from __future__ import annotations

_SHAPE_TEXT = {
    "closed": "two-sided 95% profile-likelihood interval",
    "upper_bound": "only an upper bound is supported by the data; the lower side is not constrained "
                   "inside the prior box",
    "lower_bound": "only a lower bound is supported by the data; the upper side is not constrained "
                   "inside the prior box",
    "none": "not determined by the data inside the prior box (the profile does not close on either "
            "side); the value shown is the fit's representative point, not an estimate",
}
_RIDGE = ("theta-hat for kappa and raw beta is one point on a flat ridge; it depends on the start and the "
          "box and is not a regression target")
_BETA_BOX = ("its upper end is set by the kappa box (kappa_max = 300), not by the data (spec 003's kappa_max = "
             "3000 study): {}; the data determine combinations such as r-bar and beta/(1+kappa)")
_BETA_BOX_READ = {"closed": "read it as a lower bound", "upper_bound": "the data bound raw beta on neither side"}


def labels(claims, theta_hat) -> dict[str, str]:
    """One label per quantity, keyed by parameter name."""
    out = {}
    for c in claims:
        parts = [_SHAPE_TEXT[c.shape]]
        if c.parameter in ("kappa", "beta"):
            parts.append(_RIDGE)
        if c.parameter == "beta" and c.shape in _BETA_BOX_READ:
            parts.append(_BETA_BOX.format(_BETA_BOX_READ[c.shape]))
        out[c.parameter] = "; ".join(parts)
    return out
