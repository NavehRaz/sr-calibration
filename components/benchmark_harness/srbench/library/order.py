"""Tiers, prefixes, and the ordering that makes every sizing mode comparable.

THE LOAD-BEARING IDEA: the library is a fixed ORDERED list and every sizing mode -- tier, explicit
count, runtime budget -- selects a PREFIX of it. Two methods of different speeds therefore produce
prefixes of different lengths where the shorter is a subset of the longer, so a paired comparison
survives automatically on the shorter one. Without the ordering, budget mode would hand two methods
incomparable cohort sets.

Two ordering properties, both tested:

  * ARMS AND KAPPA ARE INTERLEAVED within each tier band, so a prefix that stops part-way through
    a band is spread rather than concentrated in one corner. A prefix that happened to be all
    `pure` at one kappa would be a biased sample presented as a small one.

  * COMMON-CORE POINTS COME FIRST at every kappa. That is the same property that makes the
    cross-kappa comparison paired, applied to prefixes: even a short prefix is paired across kappa.
"""
from __future__ import annotations

from .spec import CohortSpec

#: Tier boundaries, as prefix lengths. See `specs/.../library-design.md` for the derivation.
#:   quick  -- `pure` only; a smoke test. Supports a POOLED verdict and nothing finer.
#:   medium -- all six arms; a pooled verdict per arm.
#:   full   -- per-stratum claims and the paired cross-kappa comparison.
#: `full` here is the PLANNED size of the tiered body. The built library's own manifest is what
#: sizing actually reads (it is smaller after build-time exclusions), so these two are allowed to
#: differ -- but `tests/unit/test_order.py` pins them together against the plan, which is what
#: keeps this constant from quietly going stale. 1,992 -> 2,112 at v3, -> 1,644 at v4 (cv-sharp and n=100k moved to opt-in groups).
TIERS = {"quick": 30, "medium": 360, "full": 1644}

#: Below this many cohorts in a stratum, no per-stratum figure is printed (gate G2). A stratum of
#: three cohorts produces a confident-looking coverage that is noise.
MIN_CELLS_PER_STRATUM = 10


def tier_length(tier: str) -> int:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; known: {sorted(TIERS)}")
    return TIERS[tier]


def prefix(specs: list[CohortSpec], n: int) -> list[CohortSpec]:
    """The first `n` cohorts in library order."""
    if n > len(specs):
        raise ValueError(
            f"requested {n} cohorts but the library holds {len(specs)}. Refusing rather than "
            "truncating silently -- a run the user believes is larger than it is produces "
            "figures whose denominators are wrong."
        )
    return specs[:n]


def tier_of(position: int) -> str | None:
    """The smallest tier containing this position, or None if beyond `full`."""
    for name in ("quick", "medium", "full"):
        if position <= TIERS[name]:
            return name
    return None


def tier_between(n: int) -> tuple[str | None, str | None]:
    """Which named tiers an arbitrary count falls between -- reported for `--cohorts N`."""
    below = None
    for name in ("quick", "medium", "full"):
        if TIERS[name] <= n:
            below = name
    above = next((name for name in ("quick", "medium", "full") if TIERS[name] > n), None)
    return below, above


def largest_tier_within(budget_units: float, per_cohort: float) -> tuple[str | None, int]:
    """(largest tier that fits, largest raw cohort count that fits) for a budget.

    Both are reported on a refusal, because "medium does not fit" is much less useful than
    "medium does not fit; quick does, and so would 172 cohorts".
    """
    if per_cohort <= 0:
        return "full", TIERS["full"]
    affordable = int(budget_units // per_cohort)
    best = None
    for name in ("quick", "medium", "full"):
        if TIERS[name] <= affordable:
            best = name
    return best, min(affordable, TIERS["full"])


def interleave(groups: list[list[CohortSpec]]) -> list[CohortSpec]:
    """Round-robin the groups so any prefix is spread across all of them.

    Groups are drained in order, longest-first within a round, so an exhausted group does not
    leave a tail concentrated in whatever remains.
    """
    out: list[CohortSpec] = []
    idx = [0] * len(groups)
    remaining = sum(len(g) for g in groups)
    while remaining:
        for gi, g in enumerate(groups):
            if idx[gi] < len(g):
                out.append(g[idx[gi]])
                idx[gi] += 1
                remaining -= 1
    return out


def renumber(specs: list[CohortSpec]) -> list[CohortSpec]:
    """Stamp contiguous 1-based positions after ordering. Positions define every prefix, so they
    are assigned once, at the end, and never inferred from list order downstream."""
    import dataclasses
    return [dataclasses.replace(s, position=i + 1) for i, s in enumerate(specs)]


def arm_counts(specs: list[CohortSpec]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in specs:
        out[s.arm] = out.get(s.arm, 0) + 1
    return out


def stratum_counts(specs: list[CohortSpec]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in specs:
        out[s.stratum] = out.get(s.stratum, 0) + 1
    return out


def qualifying_strata(specs: list[CohortSpec]) -> tuple[list[str], list[str]]:
    """(strata that support a figure, strata that do not).

    Gate G2 refuses a per-stratum FIGURE computed from too few cells -- it does not refuse every
    per-stratum figure because one stratum is thin. Reporting a single global boolean would let
    one sparse corner veto claims about well-populated ones, which throws away real evidence and
    is not what the gate says.
    """
    counts = stratum_counts(specs)
    ok = sorted(k for k, v in counts.items() if v >= MIN_CELLS_PER_STRATUM)
    thin = sorted(k for k, v in counts.items() if v < MIN_CELLS_PER_STRATUM)
    return ok, thin


def claim_strength(specs: list[CohortSpec]) -> dict:
    """What a given selection does and does NOT support -- printed with every run.

    A size below a threshold still runs; what changes is that the report says which claims it
    supports, rather than printing a confident-looking figure computed from three cohorts.
    """
    arms = arm_counts(specs)
    per_arm_ok = sorted(a for a, v in arms.items() if v >= MIN_CELLS_PER_STRATUM)
    ok, thin = qualifying_strata(specs)
    counts = stratum_counts(specs)
    return {
        "n": len(specs),
        "arms_present": sorted(arms),
        "arms_absent": sorted(set(_ALL_ARMS) - set(arms)),
        "arms_supporting_a_figure": per_arm_ok,
        "arms_too_thin": sorted(set(arms) - set(per_arm_ok)),
        "supports_pooled": bool(specs),
        "supports_per_arm": len(per_arm_ok) > 1,
        "strata_supporting_a_figure": ok,
        "strata_too_thin": {k: counts[k] for k in thin},
        "n_cohorts_in_thin_strata": sum(counts[k] for k in thin),
        "min_cells_per_stratum": MIN_CELLS_PER_STRATUM,
    }


_ALL_ARMS = ("pure", "hazard", "window", "het", "censor", "realfull")
