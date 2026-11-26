"""The published calibration-method contract, as data.

Every type here is a NAMED RECORD, never a positional tuple. That is not stylistic: a 2-tuple
appended where the caller unpacked 3 silently dropped a whole cohort from a study in the
predecessor harness, and the failure was invisible because both shapes iterate.

Two further conventions that this module enforces rather than documents:

  * A dimensionless quantity and its physical counterpart NEVER share a name. Dimensionless
    parameters carry the `rho_` prefix; physical ones do not. Mislabelling one as the other has
    cost this project a factor of L^2 once and a factor of L twice.

  * `Claim.shape` is REQUIRED and is stated by the method. The harness never infers it from an
    infinite bound. One-sided and two-sided claims are scored separately -- pooling them turns a
    true coverage of 0.250 into an apparent 0.676 -- and inference is how pooling happens by
    accident.

Serialisation is plain JSON-compatible dicts, so the in-process and out-of-process paths carry
the same payload and are checked by the same validator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

CONTRACT_VERSION = "1.0"

#: The four parameters a method recovers. The death threshold is fixed at 1 and is NOT fitted --
#: it is not identifiable from a normalised survival curve -- so it never appears here.
PARAMETERS = ("eta", "beta", "epsilon", "kappa")

#: Observation-model features a method may declare it MODELS. Declaring one changes how a result
#: is LABELLED (like-for-like rather than misspecification), never whether it is scored.
CAPABILITIES = ("external_hazard", "fit_window", "heterogeneity", "censoring")

#: Claim shapes. `none` is a RESULT -- "the data cannot exclude any value in my bounds" -- not a
#: failure, and it is scored as a decline rather than as a miss.
SHAPES = ("closed", "upper_bound", "lower_bound", "none")

#: Integer codes used in the explorer payload. Kept here so the payload and the scorer cannot
#: drift apart on what a shape means.
SHAPE_CODE = {"none": 0, "upper_bound": 1, "lower_bound": 2, "closed": 3}


class ContractError(ValueError):
    """A method violated the published contract. Raised before any calibration runs."""


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


@dataclass(frozen=True, slots=True)
class CostDeclaration:
    """What a method SAYS one calibration costs. A hypothesis, not a measurement.

    Kept as its own type, distinct from `CostMeasurement`, so the divergence between the two
    stays reportable. A design that reconciled them would have nothing left to compare.
    """

    median_seconds: float
    min_seconds: float
    max_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CostDeclaration":
        return cls(
            median_seconds=float(d["median_seconds"]),
            min_seconds=float(d["min_seconds"]),
            max_seconds=float(d["max_seconds"]),
        )

    def contains(self, seconds: float) -> bool:
        return self.min_seconds <= seconds <= self.max_seconds


@dataclass(frozen=True, slots=True)
class CostMeasurement:
    """What the harness OBSERVED one calibration to cost.

    Carries both wall-clock and CPU time: wall-clock is what a user waits and what a budget is
    spent in, but on a shared preemptable cluster the two diverge, and reporting both is what
    lets a surprising forecast be diagnosed instead of guessed at.
    """

    median_seconds: float
    min_seconds: float
    max_seconds: float
    n_probes: int
    wall_clock_seconds: tuple[float, ...] = ()
    cpu_seconds: tuple[float, ...] = ()
    cohort_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("wall_clock_seconds", "cpu_seconds", "cohort_ids"):
            d[k] = list(d[k])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CostMeasurement":
        return cls(
            median_seconds=float(d["median_seconds"]),
            min_seconds=float(d["min_seconds"]),
            max_seconds=float(d["max_seconds"]),
            n_probes=int(d["n_probes"]),
            wall_clock_seconds=tuple(float(x) for x in d.get("wall_clock_seconds", ())),
            cpu_seconds=tuple(float(x) for x in d.get("cpu_seconds", ())),
            cohort_ids=tuple(str(x) for x in d.get("cohort_ids", ())),
        )


@dataclass(frozen=True, slots=True)
class MethodDescriptor:
    """The metadata half of the contract.

    `version` is the one field whose neglect produces a WRONG answer rather than a loud failure:
    it is part of the result-store key, so an unchanged version after a behaviour change makes
    the store serve the old result silently. Everything else in this contract fails loudly.
    """

    name: str
    version: str
    contract_version: str = CONTRACT_VERSION
    #: per parameter, the closed range the method searches
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: the observation-model features this method MODELS (subset of CAPABILITIES)
    capabilities: tuple[str, ...] = ()
    #: which of PARAMETERS it claims to estimate
    parameters_returned: tuple[str, ...] = PARAMETERS
    description: str = ""
    contact: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "contract_version": self.contract_version,
            "bounds": {k: [float(v[0]), float(v[1])] for k, v in self.bounds.items()},
            "capabilities": list(self.capabilities),
            "parameters_returned": list(self.parameters_returned),
            "description": self.description,
            "contact": self.contact,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MethodDescriptor":
        return cls(
            name=str(d["name"]),
            version=str(d["version"]),
            contract_version=str(d.get("contract_version", CONTRACT_VERSION)),
            bounds={k: (float(v[0]), float(v[1])) for k, v in (d.get("bounds") or {}).items()},
            capabilities=tuple(d.get("capabilities") or ()),
            parameters_returned=tuple(d.get("parameters_returned") or PARAMETERS),
            description=str(d.get("description", "")),
            contact=str(d.get("contact", "")),
        )

    def models(self, capability: str) -> bool:
        return capability in self.capabilities


@dataclass(frozen=True, slots=True)
class Observation:
    """What a method is SHOWN for one cohort.

    Deliberately distinct from the cohort's specification, because it contains NO TRUTH. A method
    that could see the truth would not be being tested, so the narrowing is structural rather
    than a matter of discipline.

    `window`, when present, means the cohort is observed only inside that interval: individuals
    who died before it opened are already absent from `ages`, and individuals alive at its close
    appear as censored there. The harness applies that transform -- nothing upstream implements
    it (see research.md M3).
    """

    cohort_id: str
    ages: tuple[float, ...]
    events: tuple[int, ...]
    #: recording resolution in the same time unit as `ages`; None means genuinely continuous
    delta: float | None = None
    window: tuple[float, float] | None = None

    @property
    def n_observed(self) -> int:
        return len(self.ages)

    def to_dict(self) -> dict:
        return {
            "cohort_id": self.cohort_id,
            "ages": list(self.ages),
            "events": [int(e) for e in self.events],
            "delta": self.delta,
            "window": list(self.window) if self.window is not None else None,
            "n_observed": self.n_observed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Observation":
        w = d.get("window")
        return cls(
            cohort_id=str(d["cohort_id"]),
            ages=tuple(float(x) for x in d["ages"]),
            events=tuple(int(x) for x in d["events"]),
            delta=None if d.get("delta") is None else float(d["delta"]),
            window=None if w is None else (float(w[0]), float(w[1])),
        )


@dataclass(frozen=True, slots=True)
class Claim:
    """What a method says about ONE parameter of ONE cohort.

    `shape` is required and is the method's own statement. See the module docstring for why it is
    never inferred.

    `n_components`: if the claimed region at the stated level is disconnected, how many components
    it has. For this model that is common -- 20 of 72 profiles from one set of real cohorts had a
    disjoint 95% set, and on 4 of 24 the truth sat in a component other than the optimum's.
    Returning only the optimum's component reports a strictly narrower interval, always in the
    same direction, so it depresses coverage most on exactly the flat parameters where it matters.
    """

    parameter: str
    shape: str
    lower: float | None = None
    upper: float | None = None
    point: float | None = None
    level: float = 0.95
    n_components: int = 1

    @property
    def is_claim(self) -> bool:
        """False for a decline. A decline is excluded from coverage denominators, never scored
        as a miss -- and never as a hit either."""
        return self.shape != "none"

    @property
    def is_two_sided(self) -> bool:
        return self.shape == "closed"

    @property
    def shape_code(self) -> int:
        return SHAPE_CODE[self.shape]

    def contains(self, truth: float) -> bool | None:
        """Does the claim contain `truth`? None for a decline -- which is neither hit nor miss.

        Returning None rather than False is load-bearing: a decline scored as False would let a
        method be punished for honesty, and a decline scored as True would let it be rewarded for
        silence. It belongs in neither numerator nor denominator.
        """
        if not self.is_claim:
            return None
        lo = -math.inf if self.lower is None else self.lower
        hi = math.inf if self.upper is None else self.upper
        return bool(lo <= truth <= hi)

    def width_dex(self) -> float | None:
        """log10(upper/lower), or None unless the claim is CLOSED.

        Width is undefined for a one-sided claim, and returning a number for one is how widths
        get pooled across shapes.
        """
        if self.shape != "closed" or not self.lower or not self.upper:
            return None
        if self.lower <= 0 or self.upper <= 0:
            return None
        return math.log10(self.upper / self.lower)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"parameter": self.parameter, "shape": self.shape,
                             "level": self.level, "n_components": self.n_components}
        for k in ("lower", "upper", "point"):
            v = getattr(self, k)
            if v is not None:
                d[k] = float(v)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Claim":
        return cls(
            parameter=str(d["parameter"]),
            shape=str(d["shape"]),
            lower=None if d.get("lower") is None else float(d["lower"]),
            upper=None if d.get("upper") is None else float(d["upper"]),
            point=None if d.get("point") is None else float(d["point"]),
            level=float(d.get("level", 0.95)),
            n_components=int(d.get("n_components", 1)),
        )


@dataclass(frozen=True, slots=True)
class ClaimSet:
    """Everything one method produced for one cohort, plus what it cost.

    A `status` other than "ok" is counted in its OWN category and is never silently removed from
    a denominator: a failed calibration and a declined parameter are different things and are
    reported as different things.

    `diagnostics` is carried through UNFILTERED. The predecessor harness computes three per-fit
    failure counters and then discards all three, because its diagnostic whitelist omits their
    key -- the exact production failure rate it wanted, computed and thrown away.
    """

    cohort_id: str
    method_name: str
    claims: tuple[Claim, ...] = ()
    status: str = "ok"
    wall_clock_seconds: float | None = None
    cpu_seconds: float | None = None
    diagnostics: dict = field(default_factory=dict)
    code_provenance: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def claim_for(self, parameter: str) -> Claim | None:
        for c in self.claims:
            if c.parameter == parameter:
                return c
        return None

    def to_dict(self) -> dict:
        return {
            "cohort_id": self.cohort_id,
            "method_name": self.method_name,
            "status": self.status,
            "wall_clock_seconds": self.wall_clock_seconds,
            "cpu_seconds": self.cpu_seconds,
            "claims": [c.to_dict() for c in self.claims],
            "diagnostics": dict(self.diagnostics),
            "code_provenance": dict(self.code_provenance),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ClaimSet":
        return cls(
            cohort_id=str(d["cohort_id"]),
            method_name=str(d["method_name"]),
            claims=tuple(Claim.from_dict(c) for c in d.get("claims", ())),
            status=str(d.get("status", "ok")),
            wall_clock_seconds=(None if d.get("wall_clock_seconds") is None
                                else float(d["wall_clock_seconds"])),
            cpu_seconds=None if d.get("cpu_seconds") is None else float(d["cpu_seconds"]),
            diagnostics=dict(d.get("diagnostics") or {}),
            code_provenance=dict(d.get("code_provenance") or {}),
        )
