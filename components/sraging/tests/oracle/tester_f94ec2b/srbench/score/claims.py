"""Scoring claims against truth: coverage, denominators, claim shapes, accuracy.

THREE RULES THAT ARE NOT NEGOTIABLE HERE, each written against a specific published error.

1. A NON-CLAIM IS NEITHER A HIT NOR A MISS. It is excluded from the coverage denominator and
   reported as a decline RATE beside the coverage. `Coverage` therefore cannot be constructed
   without its decline rate -- the type has no field for one without the other.

2. CLAIM SHAPES ARE NEVER POOLED. A one-sided claim contains the truth almost for free. Pooled,
   the flattest parameter reads 0.676; its two-sided intervals cover 0.250. `coverage()` scores
   one shape class at a time and raises if asked to mix them.

3. ACCURACY CLAMPS BOTH SIDES TO THE FLOOR. Below the detectability floor a parameter is not
   measurable, so "wrong by 1000x" and "correctly reported as zero" are the same statement.
   Clamping only the estimate would charge a method for being right.

And one that is easy to lose: a FAILED calibration is not a decline. Failures are counted in their
own category and never removed from a denominator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..contract.types import Claim, ClaimSet
from .floors import clamp

#: Claim-shape classes that may be scored together. `closed` alone is the honest default for
#: anything quoting a width; `any` is permitted only where the caller has said so explicitly.
SHAPE_CLASSES = ("closed", "one_sided", "any")


class PoolingError(ValueError):
    """An attempt to pool things this project has decided must not be pooled."""


@dataclass(frozen=True, slots=True)
class Coverage:
    """A coverage figure and everything required to read it.

    Every field here is mandatory on purpose. A coverage without its denominator, its decline rate
    and its shape class is not a result -- and the commonest way those go missing is a type that
    lets them be optional.
    """

    hits: int
    claimed: int          # the denominator: cohorts where a claim was made
    declined: int         # reported beside coverage, never folded into it
    failed: int           # calibrations that errored: their own category
    total: int            # every cohort considered
    shape_class: str
    parameter: str
    cohort_set: str = ""

    @property
    def coverage(self) -> float | None:
        """None when nothing was claimed. Not 0.0 -- that would read as 'covered nothing',
        which is a measurement, where the truth is 'measured nothing'."""
        return (self.hits / self.claimed) if self.claimed else None

    @property
    def decline_rate(self) -> float | None:
        return (self.declined / self.total) if self.total else None

    @property
    def failure_rate(self) -> float | None:
        return (self.failed / self.total) if self.total else None

    def to_dict(self) -> dict:
        return {
            "parameter": self.parameter, "shape_class": self.shape_class,
            "coverage": self.coverage, "hits": self.hits, "claimed": self.claimed,
            "declined": self.declined, "decline_rate": self.decline_rate,
            "failed": self.failed, "failure_rate": self.failure_rate,
            "total": self.total, "cohort_set": self.cohort_set,
        }

    def format(self) -> str:
        """Coverage never appears without its denominator and decline rate."""
        cov = "n/a" if self.coverage is None else f"{self.coverage:.3f}"
        return (f"{self.parameter:<20} {self.shape_class:<10} cov={cov:>6} "
                f"n={self.claimed:<6} declined={self.decline_rate:.3f} "
                f"failed={self.failure_rate:.3f}")


def _in_class(claim: Claim, shape_class: str) -> bool:
    if shape_class == "any":
        return claim.is_claim
    if shape_class == "closed":
        return claim.shape == "closed"
    if shape_class == "one_sided":
        return claim.shape in ("upper_bound", "lower_bound")
    raise PoolingError(f"unknown shape class {shape_class!r}; allowed: {SHAPE_CLASSES}")


def coverage(claim_sets, truths: dict[str, dict[str, float]], parameter: str,
             shape_class: str = "closed", cohort_set: str = "") -> Coverage:
    """Score one parameter over a set of calibrations.

    `truths` maps cohort_id -> {parameter: true value}.

    `shape_class` must be given. There is no default that silently pools, because pooling shapes
    is the single error that most inflates a coverage figure.
    """
    if shape_class not in SHAPE_CLASSES:
        raise PoolingError(f"unknown shape class {shape_class!r}")

    hits = claimed = declined = failed = total = 0
    for cs in claim_sets:
        truth = truths.get(cs.cohort_id, {}).get(parameter)
        if truth is None:
            continue
        total += 1
        if not cs.ok:
            failed += 1
            continue
        c = cs.claim_for(parameter)
        if c is None or not c.is_claim:
            declined += 1
            continue
        if not _in_class(c, shape_class):
            # A claim of a different shape is not scored here AND is not a decline. It belongs to
            # the other shape class's figure, which is reported separately.
            continue
        claimed += 1
        if c.contains(truth):
            hits += 1
    return Coverage(hits=hits, claimed=claimed, declined=declined, failed=failed, total=total,
                    shape_class=shape_class, parameter=parameter, cohort_set=cohort_set)


def coverage_all_shapes(claim_sets, truths, parameter: str, cohort_set: str = "") -> dict:
    """Both shape classes, reported side by side. The only sanctioned way to see everything."""
    return {
        "closed": coverage(claim_sets, truths, parameter, "closed", cohort_set),
        "one_sided": coverage(claim_sets, truths, parameter, "one_sided", cohort_set),
    }


# --------------------------------------------------------------------------- accuracy

def log_error(estimate: float | None, truth: float, floor: float | None) -> float | None:
    """`|log10(est/truth)|` with BOTH sides clamped to the floor.

    Clamping both is what makes "correctly reported as negligible" score exactly 0 instead of a
    huge error against a truth nobody could have measured. Clamping only the estimate would charge
    a method for being right about an unmeasurable parameter.
    """
    if estimate is None or estimate <= 0 or truth <= 0:
        return None
    e = clamp(estimate, floor)
    t = clamp(truth, floor)
    return abs(math.log10(e / t))


@dataclass(frozen=True, slots=True)
class Accuracy:
    median_dex: float | None
    n: int
    parameter: str
    floor_used: float | None
    floor_is_fixed: bool
    clamped_fraction: float
    cohort_set: str = ""

    def to_dict(self) -> dict:
        return {"parameter": self.parameter, "median_dex": self.median_dex, "n": self.n,
                "floor_used": self.floor_used, "floor_is_fixed": self.floor_is_fixed,
                "clamped_fraction": self.clamped_fraction, "cohort_set": self.cohort_set}


def accuracy(claim_sets, truths, parameter: str, floors: dict[str, float | None],
             floor_is_fixed: bool, cohort_set: str = "") -> Accuracy:
    """Median absolute log-10 error, floor-clamped.

    `floor_is_fixed` is REQUIRED and is carried into the result, because comparing accuracy across
    sample sizes at each n's OWN floor makes a flat quantity look like a trend: the floor shrinks
    as 1/sqrt(n), and that alone turned a flat ~0.60 dex into an apparent degradation to 0.892.
    """
    errs, clamped = [], 0
    for cs in claim_sets:
        if not cs.ok:
            continue
        truth = truths.get(cs.cohort_id, {}).get(parameter)
        if truth is None:
            continue
        c = cs.claim_for(parameter)
        if c is None or c.point is None:
            continue
        fl = floors.get(cs.cohort_id)
        e = log_error(c.point, truth, fl)
        if e is None:
            continue
        if fl is not None and (c.point < fl or truth < fl):
            clamped += 1
        errs.append(e)
    errs.sort()
    med = None
    if errs:
        m = len(errs) // 2
        med = errs[m] if len(errs) % 2 else 0.5 * (errs[m - 1] + errs[m])
    return Accuracy(median_dex=med, n=len(errs), parameter=parameter,
                    floor_used=None, floor_is_fixed=floor_is_fixed,
                    clamped_fraction=(clamped / len(errs)) if errs else 0.0,
                    cohort_set=cohort_set)


# --------------------------------------------------------------------------- widths / availability

def widths_dex(claim_sets, parameter: str) -> list[float]:
    """Widths of CLOSED claims only. A one-sided claim has no width, and returning a number for
    one is how widths get pooled across shapes."""
    out = []
    for cs in claim_sets:
        if not cs.ok:
            continue
        c = cs.claim_for(parameter)
        if c is None:
            continue
        w = c.width_dex()
        if w is not None:
            out.append(w)
    return sorted(out)


def scored_quantities(truths) -> tuple[str, ...]:
    """The four parameters, plus any OPTIONAL quantity this cohort set actually carries.

    An optional quantity earns a row by the data having a truth for it somewhere -- not by a
    method claiming it. If the run has hazard cohorts and no method estimates the hazard, the row
    still appears and reads "not measured", which is the honest statement; dropping the row would
    hide that nobody tried.

    Conversely a run with no hazard cohorts gets no hazard row at all, rather than a column of
    n/a that invites the reader to wonder what went wrong.
    """
    from ..contract.types import OPTIONAL_QUANTITIES, PARAMETERS
    extra = [q for q in OPTIONAL_QUANTITIES
             if any(t.get(q) is not None for t in truths.values())]
    return tuple(PARAMETERS) + tuple(extra)


def claim_shape_mix(claim_sets, parameter: str, truths=None) -> dict[str, int]:
    """How often each shape was returned. Part of the information axis: a method that converts
    closed intervals into one-sided bounds has answered less of the question, not more of it.

    `truths`, when given, restricts the count to cohorts where the quantity EXISTS. Without it an
    optional quantity reads as declined on every cohort that never had one -- so `external_hazard`
    would show a ~1.0 decline rate on a pure-arm run and `availability_axis` would report "loses"
    about a quantity nobody could have claimed. Every other scorer here already applies this
    filter (`coverage` at the truth lookup); this one did not.
    """
    mix = {"closed": 0, "upper_bound": 0, "lower_bound": 0, "none": 0, "failed": 0}
    for cs in claim_sets:
        if truths is not None and truths.get(cs.cohort_id, {}).get(parameter) is None:
            continue
        if not cs.ok:
            mix["failed"] += 1
            continue
        c = cs.claim_for(parameter)
        mix[c.shape if c is not None else "none"] += 1
    return mix


def availability(claim_sets, parameter: str, truths=None) -> dict:
    """Decline and failure rates -- the availability axis.

    A method that answers less often has narrowed the question rather than improved on it, which
    is exactly what a coverage-only comparison rewards.
    """
    mix = claim_shape_mix(claim_sets, parameter, truths)
    total = sum(mix.values())
    answered = total - mix["none"] - mix["failed"]
    return {
        "parameter": parameter, "total": total, "answered": answered,
        "declined": mix["none"], "failed": mix["failed"],
        "decline_rate": mix["none"] / total if total else None,
        "failure_rate": mix["failed"] / total if total else None,
        "shape_mix": mix,
    }
