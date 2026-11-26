"""The five axes and the composite acceptance rule.

WHY THE RULE IS COMPOSITE, AND NOT A COVERAGE TEST.

A paired coverage z is **gameable**, and the trivial null is the existence proof. An estimator can
raise its coverage by:

  * DECLINING more often, which shrinks the denominator to the easy cohorts; or
  * returning WIDER intervals, which contain the truth more often for free.

A coverage-only test rewards both. The data-ignoring null that returns its own parameter bounds
covers the flattest parameter at **1.0000 on all 462 cohorts at every level**, beating the selected
estimator's 0.822 -- so a coverage number alone is never a result.

Hence: a method supersedes the reference only if it wins CALIBRATION **without losing**
AVAILABILITY, INFORMATION or ACCURACY. COST is reported and never decides alone -- a method is not
refused for being expensive, but a tenfold cost for a hundredth of coverage is a different trade
from a tenfold saving, and the reader is entitled to see it.

This is not a hypothetical safeguard. Stating only the winning axis is how this project previously
talked itself into the wrong estimator for a period.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import claims as CL
from . import pairing as PA

AXES = ("calibration", "availability", "information", "accuracy", "cost")

#: The calibration axis must clear this to count as a win.
Z_THRESHOLD = 2.0

#: Tolerances for "no worse". A method is not refused for noise-level movement on a secondary
#: axis; it is refused for a real loss.
DECLINE_TOLERANCE = 0.02      # availability: fraction of cohorts
WIDTH_TOLERANCE_DEX = 0.05    # information: median interval width
ACCURACY_TOLERANCE_DEX = 0.05  # accuracy: median absolute log error


@dataclass(frozen=True, slots=True)
class AxisResult:
    """One axis, with everything required to read it.

    `denominator`, `cohort_set` and `paired` are mandatory fields, not optional extras. A figure
    without them is not interpretable, and the commonest way they go missing is a type that allows
    their absence.
    """

    axis: str
    outcome: str                    # "wins" | "no difference" | "loses" | "not measured"
    statistic: float | None
    uncertainty: float | None
    denominator: int
    cohort_set: str
    paired: bool
    aggregation: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "axis": self.axis, "outcome": self.outcome, "statistic": self.statistic,
            "uncertainty": self.uncertainty, "denominator": self.denominator,
            "cohort_set": self.cohort_set, "paired": self.paired,
            "aggregation": self.aggregation, "detail": dict(self.detail),
        }

    def format(self) -> str:
        stat = "n/a" if self.statistic is None else f"{self.statistic:+.4f}"
        unc = "" if self.uncertainty is None else f" +/-{1.96*self.uncertainty:.4f}"
        return (f"{self.axis:<14} {self.outcome:<14} {stat}{unc}   "
                f"n={self.denominator}  paired={self.paired}  [{self.aggregation}]")


@dataclass(frozen=True, slots=True)
class Verdict:
    overall: str
    axes: dict[str, AxisResult]
    arm: str
    untested_regimes: list[str]
    reference: dict
    null_comparison: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall": self.overall, "arm": self.arm,
            "axes": {k: v.to_dict() for k, v in self.axes.items()},
            "untested_regimes": list(self.untested_regimes),
            "reference": dict(self.reference),
            "null_comparison": dict(self.null_comparison),
            "reasons": list(self.reasons),
        }


def calibration_axis(a_sets, b_sets, truths, parameter, cluster_of,
                     shape_class="closed", cohort_set="") -> AxisResult:
    r = PA.paired_coverage_gap(a_sets, b_sets, truths, parameter, cluster_of,
                               shape_class, cohort_set)
    if r.mean_difference is None or r.se is None:
        outcome = "not measured"
    elif r.z is not None and r.z > Z_THRESHOLD:
        outcome = "wins"
    elif r.z is not None and r.z < -Z_THRESHOLD:
        outcome = "loses"
    else:
        outcome = "no difference"
    return AxisResult(axis="calibration", outcome=outcome, statistic=r.mean_difference,
                      uncertainty=r.se, denominator=r.n_cells, cohort_set=cohort_set,
                      paired=True, aggregation="cell average, cluster-robust on the parameter point",
                      detail=r.to_dict())


def calibration_one_sided_axis(a_sets, b_sets, truths, parameter, cluster_of,
                               cohort_set="") -> AxisResult:
    """Calibration of ONE-SIDED claims (lower/upper bounds), paired and cluster-robust, NEVER pooled
    with the two-sided figure.

    WHY THIS EXISTS (2026-09-25). Only closed claims were ever scored, so a parameter a method
    mostly answers with a bound -- kappa, and beta on the ratio-lo cells -- had calibration "not
    measured" and a verdict of "not measured", however wrong the bounds were. Counted directly on the
    medium tier, kappa bounds covered ~44% for EVERY method (mle_profile_jax 40/91, mle_profile 41/97)
    against a nominal 95%. A bound is the surviving edge of a 95% two-sided likelihood-ratio interval,
    so its nominal coverage is at least 0.95.
    """
    r = calibration_axis(a_sets, b_sets, truths, parameter, cluster_of,
                         shape_class="one_sided", cohort_set=cohort_set)
    return AxisResult(axis="calibration_one_sided", outcome=r.outcome, statistic=r.statistic,
                      uncertainty=r.uncertainty, denominator=r.denominator, cohort_set=cohort_set,
                      paired=True,
                      aggregation="ONE-SIDED claims only; cell average, cluster-robust on the "
                                  "parameter point",
                      detail=r.detail)


def availability_axis(a_sets, b_sets, parameter, cohort_set="", truths=None) -> AxisResult:
    """Does the method ANSWER as often? A method that answers less has narrowed the question.

    `truths` restricts the count to cohorts where the quantity exists. Without it an optional
    quantity reads as declined everywhere it never applied, and this axis reports "loses" about a
    quantity no method could have claimed.
    """
    a = CL.availability(a_sets, parameter, truths)
    b = CL.availability(b_sets, parameter, truths)
    if a["decline_rate"] is None or b["decline_rate"] is None:
        return AxisResult("availability", "not measured", None, None, 0, cohort_set, False, "rate")
    d = a["decline_rate"] - b["decline_rate"]
    outcome = ("loses" if d > DECLINE_TOLERANCE else
               "wins" if d < -DECLINE_TOLERANCE else "no difference")
    return AxisResult(axis="availability", outcome=outcome, statistic=d, uncertainty=None,
                      denominator=a["total"], cohort_set=cohort_set, paired=False,
                      aggregation="decline rate over all cohorts",
                      detail={"method": a, "reference": b})


def information_axis(a_sets, b_sets, parameter, cohort_set="") -> AxisResult:
    """Interval width AND claim-shape mix. Widening an interval raises coverage for free, and
    converting closed intervals into one-sided bounds answers half the question."""
    wa, wb = CL.widths_dex(a_sets, parameter), CL.widths_dex(b_sets, parameter)
    ma = _median(wa)
    mb = _median(wb)
    mix_a = CL.claim_shape_mix(a_sets, parameter)
    mix_b = CL.claim_shape_mix(b_sets, parameter)
    if ma is None or mb is None:
        return AxisResult("information", "not measured", None, None, len(wa), cohort_set,
                          False, "median width of CLOSED claims",
                          {"method_shape_mix": mix_a, "reference_shape_mix": mix_b})
    d = ma - mb
    outcome = ("loses" if d > WIDTH_TOLERANCE_DEX else
               "wins" if d < -WIDTH_TOLERANCE_DEX else "no difference")
    # A method whose closed-claim share collapses has lost information even if its widths look
    # good, because it is quoting fewer two-sided answers.
    share_a = mix_a["closed"] / max(sum(mix_a.values()), 1)
    share_b = mix_b["closed"] / max(sum(mix_b.values()), 1)
    if share_a < share_b - DECLINE_TOLERANCE:
        outcome = "loses"
    return AxisResult(axis="information", outcome=outcome, statistic=d, uncertainty=None,
                      denominator=len(wa), cohort_set=cohort_set, paired=False,
                      aggregation="median width of CLOSED claims, in dex",
                      detail={"method_median_dex": ma, "reference_median_dex": mb,
                              "method_shape_mix": mix_a, "reference_shape_mix": mix_b,
                              "method_closed_share": share_a, "reference_closed_share": share_b})


def accuracy_axis(a_sets, b_sets, truths, parameter, floors, cohort_set="") -> AxisResult:
    """Median floor-clamped error, at a FIXED floor.

    The floor must be fixed across whatever is being compared: it goes as 1/sqrt(n), and letting
    it move turned a flat ~0.60 dex into an apparent degradation to 0.892 dex.
    """
    aa = CL.accuracy(a_sets, truths, parameter, floors, floor_is_fixed=True, cohort_set=cohort_set)
    ab = CL.accuracy(b_sets, truths, parameter, floors, floor_is_fixed=True, cohort_set=cohort_set)
    if aa.median_dex is None or ab.median_dex is None:
        return AxisResult("accuracy", "not measured", None, None, aa.n, cohort_set, False,
                          "median |log10(est/truth)| dex, fixed floor")
    d = aa.median_dex - ab.median_dex
    outcome = ("loses" if d > ACCURACY_TOLERANCE_DEX else
               "wins" if d < -ACCURACY_TOLERANCE_DEX else "no difference")
    return AxisResult(axis="accuracy", outcome=outcome, statistic=d, uncertainty=None,
                      denominator=aa.n, cohort_set=cohort_set, paired=False,
                      aggregation="median |log10(est/truth)| in dex, FIXED floor",
                      detail={"method": aa.to_dict(), "reference": ab.to_dict()})


def cost_axis(a_sets, b_sets, cohort_set="") -> AxisResult:
    """Reported, never decisive. A method is not refused for being expensive."""
    ca = _median([cs.wall_clock_seconds for cs in a_sets if cs.wall_clock_seconds])
    cb = _median([cs.wall_clock_seconds for cs in b_sets if cs.wall_clock_seconds])
    if ca is None or cb is None:
        return AxisResult("cost", "not measured", None, None, 0, cohort_set, False,
                          "median wall-clock seconds per calibration")
    ratio = ca / cb if cb else float("inf")
    outcome = "cheaper" if ratio < 1 else ("dearer" if ratio > 1 else "no difference")
    return AxisResult(axis="cost", outcome=outcome, statistic=ratio, uncertainty=None,
                      denominator=len([c for c in a_sets if c.wall_clock_seconds]),
                      cohort_set=cohort_set, paired=False,
                      aggregation="median wall-clock seconds per calibration",
                      detail={"method_median_s": ca, "reference_median_s": cb,
                              "note": "reported; never decisive on its own"})


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


def decide(axes: dict[str, AxisResult]) -> tuple[str, list[str]]:
    """The composite rule. Returns (overall, reasons).

    `supersedes` requires calibration to WIN and none of availability, information or accuracy to
    LOSE. Cost is excluded from the decision by construction -- it is not passed a veto.

    ONE-SIDED CLAIMS (when `calibration_one_sided` is supplied; callers that omit it get the rule
    exactly as before). Two-sided and one-sided coverage are never pooled, so they enter separately:
      * losing one-sided calibration is a LOSS like the other three -- a method whose bounds are
        worse calibrated does not supersede, whatever its closed intervals do;
      * when there are no paired two-sided claims to score (two-sided calibration "not measured"),
        one-sided calibration stands in as THE calibration test, and the reasons say so.
    """
    reasons = []
    cal = axes.get("calibration")
    one = axes.get("calibration_one_sided")
    losses = [a for k, a in axes.items()
              if k in ("availability", "information", "accuracy") and a.outcome == "loses"]
    if one is not None and one.outcome == "loses" and cal is not None \
            and cal.outcome != "not measured":
        losses.append(one)

    if (cal is None or cal.outcome == "not measured") and one is not None \
            and one.outcome != "not measured":
        reasons.append("calibration taken from ONE-SIDED claims: there are no paired two-sided "
                       "claims to score")
        cal = one

    if cal is None or cal.outcome == "not measured":
        return "not measured", ["calibration could not be measured on this cohort set"]

    for a in losses:
        reasons.append(f"{a.axis}: loses ({a.statistic:+.4f}) -- {a.aggregation}")

    if cal.outcome == "wins" and not losses:
        reasons.insert(0, f"calibration: wins ({cal.statistic:+.4f}, z={cal.detail.get('z'):.2f})")
        return "supersedes", reasons
    if cal.outcome == "wins" and losses:
        reasons.insert(0, f"calibration wins ({cal.statistic:+.4f}) but the composite rule is not "
                          "satisfied: raising coverage by declining more or by widening intervals "
                          "is not an improvement")
        return "does not supersede", reasons
    if cal.outcome == "loses":
        reasons.insert(0, f"calibration: loses ({cal.statistic:+.4f})")
        return "does not supersede", reasons
    reasons.insert(0, f"calibration: no difference ({cal.statistic:+.4f})")
    return ("does not supersede" if losses else "no difference detected"), reasons
