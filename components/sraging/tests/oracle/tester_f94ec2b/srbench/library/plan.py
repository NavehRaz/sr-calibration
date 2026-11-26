"""Turn the measured point selection into the ordered list of cohort specifications.

Pure: takes the stored feasibility measurement and the real-cohort anchors, produces `CohortSpec`s.
Imports NOTHING from the system under test -- generating the actual death times is a separate step
(`build.py`), and this one is the part that has to be reproducible and testable without a solver.

THE ALLOCATION (specs/.../library-design.md §6), heavy on `pure` and lean on the feature arms:

    pure      168 map points (24 per kappa) + 27 real  x2 conditions x2 seeds = 780
    hazard  \\
    window   |  42 map points ( 6 per kappa) + 27 real  x2 x2 = 276 each
    het      |
    censor  /
    realfull                                   27 real  x2 x2 = 108
                                                             ------
                                                              1,992

Feature arms take the FIRST 6 points per kappa, which are common-core points, so every feature arm
is paired with `pure` at the same manifold positions. "What does a background hazard cost?" is then
a paired difference rather than two unrelated numbers.
"""
from __future__ import annotations

import math

from .spec import CohortSpec

#: Points per kappa in the `pure` map, and in each feature arm.
N_MAP_PURE = 24
N_MAP_FEATURE = 6

#: The two observation conditions. Half the map points carry the sample-size contrast at fixed
#: resolution; the other half carry the recording-resolution contrast at fixed sample size. Both
#: are paired WITHIN a parameter point, which is what makes either readable -- and the second half
#: closes the confound the predecessor benchmark records as open (no parameter point exists there
#: both exact and rounded, so its delta effect cannot be separated from the parameter set).
N_CONTRAST = ({"n": 1000, "delta_over_L": 1e-2}, {"n": 10000, "delta_over_L": 1e-2})
DELTA_CONTRAST = ({"n": 2000, "delta_over_L": 1e-3}, {"n": 2000, "delta_over_L": 1e-1})

SEEDS = 2

#: Anchors EXCLUDED from the library, with the measurement that justifies each.
#:
#: `ecoli` costs more than a quarter of a full run on its own. Measured on the mle_profile_jax
#: full tier: 24 cohorts, 251 CPU-hours = 29.3% of the entire 858 CPU-hour run, median 35,057 s
#: against a library median of 593 s, worst single cohort 19.3 hours. The cause is its recording
#: resolution -- delta/L = 9.5e-5, three times finer than the next worst -- which forces ~63,000
#: solver steps. It is one organism out of 27 and nothing in the project's questions turns on it.
EXCLUDED_ANCHORS = ("ecoli",)

#: A third sample size for anchors far above it, so the large real cohorts are also present at a
#: size a real study actually has. The 11 human anchors are recorded at n = 100,000; `real_specs`
#: gives each anchor its own n and n//4, so without this they exist only at 100,000 and 25,000.
#: Adds 11 anchors x 2 seeds x 6 arms = 132 cohorts at ~622 s each.
REAL_EXTRA_N = 10_000

#: Recording-resolution overrides, as a MULTIPLIER on the anchor's recorded delta. Coarsening the
#: grid is the one lever that changes cost without changing the cohort's parameters.
#:
#: `celegans` is the most expensive anchor left after ecoli: 24 cohorts, 201 CPU-h under the
#: current methods, up to 23.6 CPU-h each (4.1 under v0.3.3). Its delta/L = 3e-4 is the finest in
#: the library bar ecoli's, and the solver step count follows it directly. x2 halves the fineness
#: to 6e-4, still finer than every other real anchor.
#:
#: ⚑ delta is a DATA field, so this regenerates those cohorts and their stored results no longer
#: apply -- correctly: a cohort recorded at a different resolution is different data.
ANCHOR_DELTA_SCALE = {"celegans": 2.0}

#: Cohorts moved OUT of the tiered body into opt-in groups. Same mechanism as the sample-size
#: ladder: appended, outside every tier, selected whole by `--group`.
#:
#: `sharp` -- the cv-sharp synthetic stratum. 334 cohorts, and under the current methods
#:   1,337 of 3,741 projected CPU-h: 36% of a full run. They are ALSO all flagged
#:   validity-INVALID (`cv_bias`): the backward-Euler generator inflates their lifespan CV by
#:   9-15%, so truth-based verdicts already exclude them ([T-TESTER-DEFECTS] 2). Paying a third of
#:   every run for cohorts the scorer then discards is the worst of both.
#:   The real fix is to regenerate them with a bdf2 generator at fine dt; until someone does that,
#:   they are opt-in rather than deleted, so the data survives for whoever does.
#:
#: `bign` -- the n=100,000 real cohorts. 132 cohorts; the cost is a genuine consequence of fitting
#:   them correctly (a finer grid at large n), not a defect, and v3 already gives the same anchors
#:   an n=10,000 and n=25,000 cohort. 20x slower to LOAD as well, which every payload build pays.
OPTIONAL_GROUPS = {
    "sharp": lambda s: "cv-sharp" in (s.stratum or "") and not s.real_name,
    "bign": lambda s: s.n >= 100_000,
}

#: THE SAMPLE-SIZE LADDER. An OPTIONAL block, appended after the tiered body, so no tier changes
#: meaning and no stored result is invalidated by its existence.
#:
#: WHY IT EXISTS. `N_CONTRAST` gives every map point exactly two sample sizes, so the only
#: n-comparison the library supports is 1,000 against 10,000 -- and a two-point comparison shows a
#: slope, never a shape. Measured on the full tier before this block: of the five sample sizes
#: carrying 100+ cohorts, only that one pair shared parameter points at all (84 of them); three
#: sizes shared 3 points, four shared none. So the n-dependence view could draw one segment or
#: refuse, and it refused.
#:
#: Every point appears at EVERY rung, which is what makes the comparison paired: a change across n
#: is then a change in the estimator and not in which cohorts happen to exist at each n. The
#: predecessor benchmark records the cost of getting this wrong -- 62% of one apparent coverage
#: decline across n turned out to be composition rather than behaviour.
LADDER_RUNGS = (150, 500, 1000, 3000, 10000)
N_LADDER_POINTS = 30
LADDER_DELTA_OVER_L = 1e-2          # fixed, so n is the only thing that varies along the ladder
LADDER_GROUP = "nladder"

#: Physical mean lifetime for map cohorts. Sets the time unit; 100 days matches the predecessor's
#: manifold arm, so delta/L = 1e-2 is one day at a hundred-day lifespan.
L_MAP = 100.0

#: Representative feature levels for the single-feature arms, applied to MAP cohorts. Real cohorts
#: carry their own recorded levels instead (see `_real_feature`).
FEATURE_LEVELS = {
    "hazard": {"external_hazard": 5.5},     # mid of the fitted ExtH range (4.0-7.1)
    "het": {"het_param": "xc", "het_var": 0.2},   # production's value
    "censor": {"censor_frac": 0.20},
    "window": {"window_quantiles": (0.05, 0.95)},  # resolved to times at build, per cohort
}


# --------------------------------------------------------------------------- the fit's time unit

#: A real fit records its window AND its external hazard in the FIT's time unit, which is not
#: always the unit its ages are in. Measured 2026-09-22 across the anchors:
#:
#:   * the 17 cohorts recorded at half-year resolution state both in YEARS while their ages are in
#:     DAYS (human windows [30,110], dog [3,24], cat [2,25]);
#:   * the two drosophila cohorts state theirs in days, matching their ages.
#:
#: THE HAZARD IS THE DANGEROUS ONE, because getting it wrong is silent and catastrophic rather
#: than merely odd. Taken as a per-DAY rate, `ExtH = 4.67` gives the Staffy a background mean
#: lifetime of 107 DAYS against an SR mean of 4,175 -- the background would kill the entire cohort
#: before ageing did anything, and the "hazard arm" would be measuring a pure exponential. Taken
#: per YEAR it is 107 years, i.e. the small Makeham term it is supposed to be.
FIT_UNIT_FACTORS = (1.0, 365.25)


def fit_unit_factor(anchor: dict) -> float:
    """How many age-units one fit-unit is, for this cohort.

    Determined from the recorded window against the fitted mean lifetime -- both available without
    generating anything, which is what keeps this out of the chicken-and-egg where a mis-scaled
    hazard corrupts the cohort used to detect the mis-scaling.
    """
    w = anchor.get("features", {}).get("window")
    L = float(anchor["L_phys"])
    if not w or L <= 0:
        return 1.0
    mid = 0.5 * (float(w[0]) + float(w[1]))
    if mid <= 0:
        return 1.0
    best = min(FIT_UNIT_FACTORS, key=lambda f: abs(math.log10(mid * f / L)))
    return float(best)


def hazard_in_age_units(ext_h: float, factor: float) -> float:
    """Convert a Makeham level from fit units to age units.

    The rate is `h = exp(-ExtH)` per fit-unit, so per age-unit it is `exp(-ExtH)/factor`, i.e.

        ExtH_ages = ExtH_fit + ln(factor)
    """
    return float(ext_h) + math.log(factor)


def _stratum(log_ratio: float, cv: float, n: int) -> str:
    """The cell used for per-stratum claims. Coarse on purpose: a stratum finer than the data can
    support produces confident-looking figures computed from three cohorts."""
    r = "lo" if log_ratio < -0.5 else ("mid" if log_ratio < 0.5 else "hi")
    c = "sharp" if cv < 0.05 else ("mid" if cv < 0.15 else "broad")
    return f"ratio-{r}/cv-{c}/n-{n}"


def _cluster_key(kappa: float, rho_beta: float, rho_eps: float) -> str:
    """The parameter point. Several cohorts share one; it is the unit of clustering for every
    uncertainty, so it must NOT include n, delta or seed."""
    return f"k{kappa:g}|b{rho_beta:.6g}|e{rho_eps:.6g}"


def map_specs(selection: dict, arm: str, n_points: int, seed0: int) -> list[CohortSpec]:
    """Cohort specs for one arm's share of the map."""
    out: list[CohortSpec] = []
    kappas = sorted(selection["by_kappa"])
    # Split points into the two condition halves by global index, so each half spans all kappa.
    flat = []
    for k in kappas:
        for rank, p in enumerate(selection["by_kappa"][k]["points"][:n_points]):
            flat.append((k, rank, p))
    half = len(flat) // 2

    for gi, (k, rank, p) in enumerate(flat):
        conditions = N_CONTRAST if gi < half else DELTA_CONTRAST
        contrast = "n" if gi < half else "delta"
        for ci, cond in enumerate(conditions):
            for s in range(SEEDS):
                delta = cond["delta_over_L"] * L_MAP
                feats = _map_feature(arm, p)
                out.append(CohortSpec(
                    cohort_id=f"{arm}-k{k:g}-p{rank:02d}-c{ci}-s{s}",
                    position=0,                     # stamped by order.renumber
                    arm=arm,
                    rho_eta=float(p["rho_eta"]), rho_beta=float(p["rho_beta"]),
                    rho_eps=float(p["rho_eps"]), kappa=float(k),
                    n=int(cond["n"]), L_phys=L_MAP,
                    seed=seed0 + gi * 100 + ci * 10 + s,
                    delta=delta,
                    stratum=_stratum(p["log_ratio"], p["cv"], cond["n"]),
                    cluster_key=_cluster_key(k, p["rho_beta"], p["rho_eps"]),
                    notes=f"map point rank {rank} at kappa={k:g}; {contrast}-contrast arm={arm}",
                    **feats,
                ))
    return out


def ladder_points(selection: dict, n_points: int = N_LADDER_POINTS) -> list[tuple]:
    """`n_points` map points SPREAD over the manifold, as `(kappa, rank, point)`.

    Taken rank-major and round-robin across kappa, so the block spans the kappa range and, within
    each kappa, the positions the selection itself spread over manifold cells. Taking the first
    `n_points` of one kappa instead would make every ladder result a statement about that corner of
    the manifold wearing the label "sample size".
    """
    kappas = sorted(selection["by_kappa"])
    out: list[tuple] = []
    rank = 0
    while len(out) < n_points:
        added = False
        for k in kappas:
            pts = selection["by_kappa"][k]["points"]
            if rank < len(pts):
                out.append((k, rank, pts[rank]))
                added = True
                if len(out) == n_points:
                    return out
        if not added:                       # every kappa exhausted before reaching n_points
            break
        rank += 1
    return out


def ladder_specs(selection: dict, seed0: int = 800_000,
                 rungs: tuple = LADDER_RUNGS,
                 n_points: int = N_LADDER_POINTS) -> list[CohortSpec]:
    """The optional sample-size ladder: every point at every rung, `SEEDS` seeds each.

    `pure` arm and a FIXED `delta/L`, so nothing varies along a row except n -- an observation
    feature or a changing recording resolution would be confounded with the thing being measured.

    `cluster_key` is deliberately the same across rungs for one point (it is built from kappa and
    the rho coordinates, which do not depend on n). That identity is what the n-dependence view
    pairs on, and what the cluster-robust SE clusters on.
    """
    out: list[CohortSpec] = []
    for gi, (k, rank, p) in enumerate(ladder_points(selection, n_points)):
        for ri, n in enumerate(rungs):
            for s in range(SEEDS):
                out.append(CohortSpec(
                    cohort_id=f"nlad-k{k:g}-p{rank:02d}-n{n}-s{s}",
                    position=0,                     # stamped by order.renumber
                    arm="pure",
                    group=LADDER_GROUP,
                    rho_eta=float(p["rho_eta"]), rho_beta=float(p["rho_beta"]),
                    rho_eps=float(p["rho_eps"]), kappa=float(k),
                    n=int(n), L_phys=L_MAP,
                    seed=seed0 + gi * 100 + ri * 10 + s,
                    delta=LADDER_DELTA_OVER_L * L_MAP,
                    stratum=_stratum(p["log_ratio"], p["cv"], n),
                    cluster_key=_cluster_key(k, p["rho_beta"], p["rho_eps"]),
                    notes=(f"sample-size ladder: map point rank {rank} at kappa={k:g}, n={n}; "
                           f"paired across rungs on cluster_key"),
                ))
    return out


def _map_feature(arm: str, point: dict) -> dict:
    """Feature fields for a map cohort in `arm`. `window` is resolved at build time, when the
    cohort's actual death times are known -- a window in absolute time means nothing until then."""
    if arm == "pure":
        return {}
    lv = FEATURE_LEVELS.get(arm, {})
    if arm == "hazard":
        return {"external_hazard": lv["external_hazard"]}
    if arm == "het":
        return {"het_param": lv["het_param"], "het_var": lv["het_var"]}
    if arm == "censor":
        return {"censor_frac": lv["censor_frac"]}
    if arm == "window":
        return {}     # window filled in by build.py from the generated cohort's quantiles
    return {}


def real_specs(anchors, arm: str, seed0: int) -> list[CohortSpec]:
    """Cohort specs for the 27 real fitted cohorts in one arm.

    In `pure` they are planted as pure-SR at their own fitted parameters. In `realfull` they carry
    the features each one actually had. In a single-feature arm they carry that one feature, at
    their own recorded level where they had it.

    The pure/realfull pair is the measurement: same truth, same parameters, with and without the
    real observation model, so the difference IS the cost of that observation model.
    """
    out: list[CohortSpec] = []
    for i, a in enumerate(anchors):
        if a["name"] in EXCLUDED_ANCHORS:
            continue
        # A third condition only where the anchor is far larger than REAL_EXTRA_N, so a cohort
        # already near that size does not gain a near-duplicate of itself.
        ns = [int(a["n"]), max(400, int(a["n"]) // 4)]
        if int(a["n"]) > 2 * REAL_EXTRA_N:
            ns.append(REAL_EXTRA_N)
        for ci, n in enumerate(ns):
            for s in range(SEEDS):
                feats = _real_feature(arm, a)
                out.append(CohortSpec(
                    cohort_id=f"{arm}-real-{a['name']}-c{ci}-s{s}",
                    position=0, arm=arm,
                    rho_eta=float(a["rho_eta"]), rho_beta=float(a["rho_beta"]),
                    rho_eps=float(a["rho_eps"]), kappa=float(a["kappa"]),
                    n=n, L_phys=float(a["L_phys"]),
                    seed=seed0 + i * 100 + ci * 10 + s,
                    delta=float(a["delta"]) * ANCHOR_DELTA_SCALE.get(a["name"], 1.0),
                    stratum=f"real/{a['name']}",
                    cluster_key=f"real|{a['name']}",
                    real_name=a["name"],
                    notes=f"real fitted cohort {a['name']}; flags: {a.get('flags','')}",
                    **feats,
                ))
    return out


def _real_feature(arm: str, anchor: dict) -> dict:
    """Which features a real cohort carries in a given arm.

    HAZARD SOURCE: `ExtH`, the FITTED value, not the config's `external_hazard`. The four
    Denmark_M configurations share one `external_hazard` to 15 significant figures while their
    `ExtH` values differ per cohort -- a quantity identical across a family is an input, a quantity
    varying within it is the result. These cohorts are planted at their fitted parameters, so the
    fitted hazard is the one that describes them. See library-design.md for the full table.
    """
    if arm == "pure":
        return {}
    f = anchor.get("features", {})
    factor = fit_unit_factor(anchor)
    if arm == "realfull":
        # The cohort's OWN recorded observation model -- including nothing at all, for the three
        # cohorts that really were pure-SR on a full window. Forcing a feature onto those would
        # make `realfull` mean "some features" instead of "the real model, whole", and would
        # destroy the pure-vs-realfull pairing for exactly the cohorts where it is cleanest.
        out = {}
        if f.get("external_hazard") is not None:
            out["external_hazard"] = hazard_in_age_units(f["external_hazard"], factor)
        if f.get("het"):
            out["het_param"], out["het_var"] = "xc", 0.2
        if f.get("censor_frac"):
            out["censor_frac"] = float(f["censor_frac"])
        if f.get("window"):
            # Recorded in the fit's own time unit, which is NOT always the ages' unit -- resolved
            # against the generated cohort at build time. See build._resolve_window.
            out["window"] = (float(f["window"][0]) * factor,
                             float(f["window"][1]) * factor)
        return out
    if arm == "hazard":
        v = f.get("external_hazard")
        if v is None:
            # This cohort's fit carried no hazard, so a representative level is used -- expressed
            # in the cohort's OWN age units, relative to its mean lifetime, rather than copied
            # from another cohort whose time unit differs.
            return {"external_hazard": math.log(float(anchor["L_phys"])) + 2.0}
        return {"external_hazard": hazard_in_age_units(v, factor)}
    if arm == "het":
        return {"het_param": "xc", "het_var": 0.2}
    if arm == "censor":
        v = f.get("censor_frac")
        return {"censor_frac": float(v) if v else FEATURE_LEVELS["censor"]["censor_frac"]}
    if arm == "window":
        w = f.get("window")
        if not w:
            return {}
        return {"window": (float(w[0]) * factor, float(w[1]) * factor)}
    return {}


def build_plan(selection: dict, anchors: list[dict]) -> list[CohortSpec]:
    """The whole ordered library, before generation.

    Order: interleave the arms so any prefix is spread; `quick` is the first 30 and is `pure` only,
    which is what makes it a smoke test that says so rather than a weak claim about everything.
    """
    from .order import interleave, renumber

    pure_map = map_specs(selection, "pure", N_MAP_PURE, seed0=100_000)
    pure_real = real_specs(anchors, "pure", seed0=200_000)

    feature_arms = ["hazard", "window", "het", "censor"]
    feat: list[list[CohortSpec]] = []
    for ai, arm in enumerate(feature_arms):
        feat.append(map_specs(selection, arm, N_MAP_FEATURE, seed0=300_000 + ai * 50_000)
                    + real_specs(anchors, arm, seed0=400_000 + ai * 50_000))
    realfull = real_specs(anchors, "realfull", seed0=600_000)

    # PARTITION FIRST, then order. The opt-in groups are removed from each source list BEFORE the
    # quick block is taken and before the interleave, because both of those are claims about what
    # the tiered body contains:
    #
    #   * `quick` is "the first 30, pure map only". Partitioning afterwards pulled cv-sharp
    #     cohorts out of that block and let interleaved `realfull` cohorts slide into it -- quick
    #     stopped being pure-only, silently. (Caught by test_quick_is_pure_only_and_says_so.)
    #   * the interleave exists so any PREFIX is spread across the arms; removing members after
    #     the round-robin leaves the survivors bunched.
    import dataclasses
    groups: dict[str, list[CohortSpec]] = {name: [] for name in OPTIONAL_GROUPS}

    def split(lst):
        keep = []
        for sp in lst:
            for name, matches in OPTIONAL_GROUPS.items():
                if matches(sp):
                    groups[name].append(dataclasses.replace(sp, group=name))
                    break
            else:
                keep.append(sp)
        return keep

    pure_map = split(pure_map)
    pure_real = split(pure_real)
    feat = [split(f) for f in feat]
    realfull = split(realfull)

    # quick = the first 30, `pure` map only.
    quick = pure_map[:30]
    rest_pure = pure_map[30:] + pure_real
    ordered = quick + interleave([rest_pure] + feat + [realfull])

    # THE LADDER IS APPENDED, NEVER INTERLEAVED. Every tier is a PREFIX, so interleaving an extra
    # block would renumber the body and silently change which cohorts `medium` means -- and with
    # the observation digest in the store key, a cohort that moved is still the same result, but a
    # tier that moved is a different measurement wearing an old name. Appending leaves positions
    # 1..len(ordered) byte-identical to the library without it.
    body = ordered

    # The ladder is appended last; every optional block sits between the body and it, each one
    # contiguous so `--group` is a slice rather than a scan.
    ladder = ladder_specs(selection)
    tail = [sp for name in sorted(groups) for sp in groups[name]]
    return renumber(body + tail + ladder)
