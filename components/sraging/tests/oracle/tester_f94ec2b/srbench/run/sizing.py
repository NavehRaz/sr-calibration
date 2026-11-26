"""The four sizing modes, and the budget guard.

    --tier quick|medium|full     a named nested tier
    --cohorts N                  an explicit count
    --budget B                   the longest prefix the MEASURED cost predicts will fit
    <size> --budget B            a size with a guard: refuse before running anything

All four resolve to a PREFIX of the fixed ordered library, which is what keeps them comparable:
two methods of different speeds produce prefixes where the shorter is a subset of the longer, so
the paired comparison survives on the shorter one automatically.

The budget guard refuses BEFORE running anything and names what would fit. "medium does not fit"
is much less useful than "medium does not fit; quick does, and so would 172 cohorts".
"""
from __future__ import annotations

from dataclasses import dataclass

from ..library import order as O


class SizingError(ValueError):
    """The requested size cannot be honoured. Raised before any calibration."""


class BudgetRefusal(SizingError):
    """The requested size does not fit the budget. Nothing was run."""


@dataclass(frozen=True, slots=True)
class Sizing:
    n: int
    mode: str
    tier: str | None
    budget_units: float | None
    forecast_units: float | None
    per_cohort_units: float | None
    note: str = ""

    def to_dict(self) -> dict:
        return {"n": self.n, "mode": self.mode, "tier": self.tier,
                "budget_units": self.budget_units, "forecast_units": self.forecast_units,
                "per_cohort_units": self.per_cohort_units, "note": self.note}


def resolve(library_size: int, *, tier: str | None = None, cohorts: int | None = None,
            budget_units: float | None = None, per_cohort_units: float | None = None,
            tiers: dict | None = None) -> Sizing:
    """Resolve a sizing request to a prefix length, or refuse.

    `tiers` is the LIBRARY'S OWN tier map when one is available. The module-level `order.TIERS`
    holds the PLANNED boundaries, which are not the built ones: `full` was planned at 1,992 and the
    build excluded 2 cohorts, so asking for the planned number either refused or -- once the
    library grew an optional block past 1,990 -- silently reached into it. A tier must mean the
    same cohorts before and after the library grows.
    """
    given = [x is not None for x in (tier, cohorts)]
    if sum(given) > 1:
        raise SizingError("give at most one of --tier and --cohorts")

    if tier is not None:
        if tiers and tier in tiers:
            n = int(tiers[tier])
        else:
            n = O.tier_length(tier)
        mode = "tier"
    elif cohorts is not None:
        if cohorts <= 0:
            raise SizingError("--cohorts must be positive")
        if cohorts > library_size:
            raise SizingError(
                f"requested {cohorts} cohorts but the library holds {library_size}. Refusing "
                "rather than truncating silently: a run the user believes is larger than it is "
                "produces figures whose denominators are wrong."
            )
        n = cohorts
        mode = "count"
    elif budget_units is not None:
        if per_cohort_units is None or per_cohort_units <= 0:
            raise SizingError(
                "sizing by budget needs a MEASURED per-calibration cost. Run `srbench cost` first "
                "-- a declared cost is a hypothesis, and this is not the place to believe one."
            )
        n = min(int(budget_units // per_cohort_units), library_size)
        if n <= 0:
            raise BudgetRefusal(
                f"a budget of {budget_units:g} does not cover a single calibration at the measured "
                f"{per_cohort_units:g} each. Nothing was run."
            )
        mode = "budget"
    else:
        raise SizingError("give one of --tier, --cohorts or --budget")

    forecast = n * per_cohort_units if per_cohort_units else None

    # The guard: a size WITH a budget must fit, or nothing runs.
    if budget_units is not None and mode != "budget":
        if forecast is None:
            raise SizingError(
                "a budget guard needs a measured per-calibration cost; run `srbench cost` first")
        if forecast > budget_units:
            best_tier, best_n = O.largest_tier_within(budget_units, per_cohort_units)
            raise BudgetRefusal(
                f"requested   {mode} {tier or cohorts} ({n} cohorts)\n"
                f"forecast    {forecast:.1f}   (measured cost {per_cohort_units:g} each)\n"
                f"budget      {budget_units:.1f}\n"
                f"REFUSED     it does not fit.\n"
                f"            largest tier that fits:  "
                f"{best_tier or '(none)'}"
                f"{f' ({O.TIERS[best_tier]} cohorts)' if best_tier else ''}\n"
                f"            largest count that fits: {best_n} cohorts\n"
                f"nothing was run."
            )

    note = ""
    below, above = O.tier_between(n)
    if mode != "tier":
        note = f"between tiers {below or '(none)'} and {above or '(beyond full)'}"
    return Sizing(n=n, mode=mode, tier=tier, budget_units=budget_units,
                  forecast_units=forecast, per_cohort_units=per_cohort_units, note=note)


def claim_note(specs) -> str:
    """The sentence a run prints about what its size does and does not support.

    Printed always, not only when the size is small: a caveat that appears only sometimes is a
    caveat nobody learns to look for.
    """
    cs = O.claim_strength(specs)
    lines = [f"NOTE  {cs['n']} cohorts."]
    if not cs["supports_per_arm"]:
        lines.append("      Supports a POOLED verdict only -- no per-arm and no per-stratum claim "
                     "is made at this size.")
    elif not cs["strata_supporting_a_figure"]:
        lines.append("      Supports a per-arm verdict; no per-stratum claim at this size.")
    else:
        lines.append(f"      Per-arm verdict; {len(cs['strata_supporting_a_figure'])} strata carry "
                     f"enough cells for a per-stratum figure "
                     f"({len(cs['strata_too_thin'])} do not and are omitted).")
    if cs["arms_absent"]:
        lines.append(f"      UNTESTED REGIMES in this run: {', '.join(cs['arms_absent'])}.")
    return "\n".join(lines)
