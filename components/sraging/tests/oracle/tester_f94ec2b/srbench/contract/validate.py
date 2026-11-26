"""The SINGLE validator, shared by the in-process and out-of-process paths.

One validator, not two, is the point: if the subprocess path had its own checks it would be a
second, less-tested contract, and the path a non-Python method takes would be the one nobody
exercises.

Everything here is checked BEFORE any calibration runs, so a violation costs seconds rather than
a week of cluster time. That is why `validate_descriptor` is called at registration and
`validate_claimset` is called on a smoke calibration, not only on results as they stream back.
"""
from __future__ import annotations

import math

from .types import (OPTIONAL_QUANTITIES, 
    CAPABILITIES,
    CONTRACT_VERSION,
    PARAMETERS,
    SHAPES,
    Claim,
    ClaimSet,
    ContractError,
    CostDeclaration,
    MethodDescriptor,
    Observation,
)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def compatible_contract_version(declared: str) -> bool:
    """Same MAJOR is compatible. A minor bump adds fields; a major bump changes meanings."""
    try:
        return declared.split(".")[0] == CONTRACT_VERSION.split(".")[0]
    except (AttributeError, IndexError):
        return False


def validate_descriptor(d: MethodDescriptor) -> None:
    """Raise ContractError on anything that disqualifies a method before it runs."""
    if not d.name or not isinstance(d.name, str):
        raise ContractError("descriptor.name must be a non-empty string")

    # `version` is required and has no safe default. It is part of the result-store key, so a
    # missing or unchanged version after a behaviour change makes the store silently serve stale
    # results -- the only failure in this contract that is quiet rather than loud.
    if not d.version or not isinstance(d.version, str):
        raise ContractError(
            f"method '{d.name}': `version` is required. It is part of the result-store key, so "
            "without it a changed method would silently reuse the previous method's results."
        )

    if not compatible_contract_version(d.contract_version):
        raise ContractError(
            f"method '{d.name}': contract_version {d.contract_version!r} is incompatible with "
            f"this harness's {CONTRACT_VERSION!r} (major version must match)"
        )

    claimable = tuple(PARAMETERS) + tuple(OPTIONAL_QUANTITIES)
    for p in d.parameters_returned:
        if p not in claimable:
            raise ContractError(
                f"method '{d.name}': parameters_returned contains {p!r}; allowed: {claimable}. "
                "Note the death threshold is fixed at 1 and must not be returned -- it is not "
                "identifiable from a normalised survival curve."
            )

    # An OPTIONAL quantity is a feature of the observation model, so claiming to estimate one
    # without declaring that you model it is incoherent -- and would be scored against a truth the
    # method never acknowledged seeing.
    for p in d.parameters_returned:
        if p in OPTIONAL_QUANTITIES and p not in d.capabilities:
            raise ContractError(
                f"method '{d.name}': returns {p!r} but does not declare it in capabilities. "
                f"Estimating an observation-model quantity means modelling it; declare "
                f"capabilities=({p!r},)."
            )

    for c in d.knows:
        if c not in CAPABILITIES:
            raise ContractError(
                f"method '{d.name}': unknown entry {c!r} in knows; allowed: {CAPABILITIES}")
        if c not in d.capabilities:
            raise ContractError(
                f"method '{d.name}': asks to be told {c!r} but does not declare modelling it. "
                "Being handed a value you do not model is not a meaningful test."
            )
        if c in OPTIONAL_QUANTITIES:
            raise ContractError(
                f"method '{d.name}': {c!r} is scored as an estimate, so it cannot also be told. "
                "Drop it from `knows`, or stop returning it."
            )

    for c in d.capabilities:
        if c not in CAPABILITIES:
            raise ContractError(
                f"method '{d.name}': unknown capability {c!r}; allowed: {CAPABILITIES}")

    for p in d.parameters_returned:
        if p not in d.bounds:
            raise ContractError(
                f"method '{d.name}': no bounds declared for {p!r}. Bounds are needed to check "
                "that no planted truth sits ON a boundary -- a truth on the boundary cannot be "
                "covered by a bounded interval and would score a guaranteed miss that reads as "
                "a failure of your method."
            )

    for p, bound in d.bounds.items():
        if len(bound) != 2 or not all(_finite(b) for b in bound):
            raise ContractError(f"method '{d.name}': bounds[{p!r}] must be two finite numbers")
        lo, hi = float(bound[0]), float(bound[1])
        if not lo < hi:
            raise ContractError(f"method '{d.name}': bounds[{p!r}] = ({lo}, {hi}) is not ordered")


def validate_cost_declaration(c: CostDeclaration, method_name: str = "?") -> None:
    for k in ("median_seconds", "min_seconds", "max_seconds"):
        v = getattr(c, k)
        if not _finite(v) or v <= 0:
            raise ContractError(f"method '{method_name}': cost {k} must be positive and finite")
    if not (c.min_seconds <= c.median_seconds <= c.max_seconds):
        raise ContractError(
            f"method '{method_name}': cost must satisfy min <= median <= max, got "
            f"({c.min_seconds}, {c.median_seconds}, {c.max_seconds})"
        )


def validate_claim(c: Claim, descriptor: MethodDescriptor) -> None:
    """Every rejection here exists because the alternative is a silently mis-scored result."""
    allowed = tuple(PARAMETERS) + tuple(OPTIONAL_QUANTITIES)
    if c.parameter not in allowed:
        raise ContractError(f"claim for unknown parameter {c.parameter!r}; allowed: {allowed}")

    # The shape must be STATED. The harness will not infer it from an infinite bound, because
    # inference is how one-sided and two-sided claims get pooled by accident -- and pooled shapes
    # turn a true coverage of 0.250 into an apparent 0.676.
    if c.shape not in SHAPES:
        raise ContractError(
            f"claim for {c.parameter!r} has shape {c.shape!r}; it must be one of {SHAPES} and "
            "must be stated explicitly -- the harness never infers it from an absent bound."
        )

    if c.shape == "closed":
        if c.lower is None or c.upper is None:
            raise ContractError(f"closed claim for {c.parameter!r} needs both lower and upper")
        if not c.lower < c.upper:
            raise ContractError(
                f"closed claim for {c.parameter!r}: lower {c.lower} is not below upper {c.upper}")
    elif c.shape == "upper_bound":
        if c.upper is None:
            raise ContractError(f"upper_bound claim for {c.parameter!r} needs `upper`")
        if c.lower is not None:
            raise ContractError(
                f"upper_bound claim for {c.parameter!r} must not carry `lower`; use `closed`")
    elif c.shape == "lower_bound":
        if c.lower is None:
            raise ContractError(f"lower_bound claim for {c.parameter!r} needs `lower`")
        if c.upper is not None:
            raise ContractError(
                f"lower_bound claim for {c.parameter!r} must not carry `upper`; use `closed`")
    else:  # none
        if c.lower is not None or c.upper is not None:
            raise ContractError(f"declined claim for {c.parameter!r} must carry no bounds")
        # Declining and then supplying a point estimate is not a decline. A decline means "the
        # data cannot exclude any value in my bounds"; a point estimate contradicts that.
        if c.point is not None:
            raise ContractError(
                f"declined claim for {c.parameter!r} must not carry a point estimate. A decline "
                "is the statement that the data does not determine this parameter; quoting a "
                "point estimate alongside it contradicts the claim."
            )

    b = descriptor.bounds.get(c.parameter)
    if b is not None:
        lo, hi = b
        for name in ("lower", "upper", "point"):
            v = getattr(c, name)
            if v is not None and not (lo <= v <= hi):
                raise ContractError(
                    f"claim for {c.parameter!r}: {name}={v} lies outside the method's own "
                    f"declared bounds ({lo}, {hi})"
                )

    if not (0.0 < c.level < 1.0):
        raise ContractError(f"claim for {c.parameter!r}: level {c.level} must be in (0, 1)")
    if c.n_components < 1:
        raise ContractError(f"claim for {c.parameter!r}: n_components must be >= 1")


def validate_claimset(cs: ClaimSet, descriptor: MethodDescriptor,
                      observation: Observation | None = None) -> None:
    if observation is not None and cs.cohort_id != observation.cohort_id:
        raise ContractError(
            f"claimset cohort_id {cs.cohort_id!r} does not match the observation it answers "
            f"({observation.cohort_id!r})"
        )
    if cs.method_name != descriptor.name:
        raise ContractError(
            f"claimset method_name {cs.method_name!r} != descriptor name {descriptor.name!r}")

    # A failed calibration is a legitimate, reportable outcome -- it is counted in its own
    # category and never dropped from a denominator -- so it is not required to carry claims.
    if not cs.ok:
        return

    seen = set()
    for c in cs.claims:
        validate_claim(c, descriptor)
        if c.parameter in seen:
            raise ContractError(f"duplicate claim for {c.parameter!r}")
        seen.add(c.parameter)

    missing = set(descriptor.parameters_returned) - seen
    if missing:
        raise ContractError(
            f"method '{descriptor.name}' declared parameters_returned="
            f"{tuple(descriptor.parameters_returned)} but returned no claim for "
            f"{sorted(missing)}. To decline a parameter on this cohort, return it with "
            'shape="none" -- that is a result and is scored as a decline, not an omission.'
        )


def validate_observation(o: Observation) -> None:
    """Checked on the way OUT to a method, so a malformed observation never reaches one."""
    if len(o.ages) != len(o.events):
        raise ContractError(
            f"observation {o.cohort_id}: ages ({len(o.ages)}) and events ({len(o.events)}) "
            "differ in length"
        )
    if not o.ages:
        raise ContractError(f"observation {o.cohort_id}: no individuals")
    if any(not _finite(a) for a in o.ages):
        raise ContractError(f"observation {o.cohort_id}: non-finite age")
    if any(e not in (0, 1) for e in o.events):
        raise ContractError(f"observation {o.cohort_id}: events must be 0 or 1")
    if o.delta is not None and (not _finite(o.delta) or o.delta < 0):
        raise ContractError(f"observation {o.cohort_id}: delta must be non-negative and finite")
    if o.window is not None:
        lo, hi = o.window
        if not (_finite(lo) and _finite(hi)) or not lo < hi:
            raise ContractError(f"observation {o.cohort_id}: window {o.window} is not an ordered "
                                "pair of finite values")
    # The known-observation-model block, when a method asked to be told something.
    if getattr(o, "known", None):
        if not isinstance(o.known, dict):
            raise ContractError(f"observation {o.cohort_id}: known must be a dict keyed by "
                                "capability")
        for cap, v in o.known.items():
            if cap not in CAPABILITIES:
                raise ContractError(f"observation {o.cohort_id}: known carries unknown capability "
                                    f"{cap!r}; allowed: {CAPABILITIES}")
            vals = v.values() if isinstance(v, dict) else [v]
            for x in vals:
                if isinstance(x, str):
                    continue                       # a label, e.g. which parameter carries the het
                if not _finite(x):
                    raise ContractError(
                        f"observation {o.cohort_id}: known[{cap!r}] has a non-finite value. A "
                        "value told to a method must be usable, or it is worse than silence.")
