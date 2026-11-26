"""The MAD cage cleaning (spec 005 US1; FR-013 to FR-017; research R4, R5; data-model "CleaningReport"; tasks T016-T018;
register D-503, D-504). HR-2's synthetic fixture: a planted outlier cage, ties, a 3-cage cohort, a 2-cage cohort, a
tied cohort whose radius is the floor, and a zero-IQR cage. Side by side with a reference written here from
cage_cleaning.py's formulas (<A>/code_5/mosquitos/cage_cleaning.py:96-267; not imported: it is in no snapshot), with
the four planted defects. The scores against SRK Dataset from the frozen snapshot (oracle group). Synthetic data
only."""
import importlib
import math

import numpy as np
import pytest
from scipy.stats import norm

from sraging import cleaning as CL
from sraging.cleaning import clean, drop, noise_baseline
from sraging.cohort import Cohort
from sraging.cohorts import CohortSet
from sraging.errors import CohortDataError, CohortSpecError, UnsupportedOptionError

K, SEED = 2.5, 20261108


# ------------------------------------------------- the reference, from cage_cleaning.py's formulas (read only)
def ref_robust_sigma(values):
    """cage_cleaning.robust_sigma (:111-117)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def ref_axis_geometry(values, k, floor_rel):
    """cage_cleaning._axis_geometry (:170-185)."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    centre = float(np.median(finite)) if finite.size else float("nan")
    sigma = ref_robust_sigma(values)
    radius = k * sigma if np.isfinite(sigma) else float("nan")
    floor_radius = (k * floor_rel * abs(centre) if floor_rel is not None and np.isfinite(centre) else float("nan"))
    if np.isfinite(floor_radius) and not (np.isfinite(radius) and radius >= floor_radius):
        radius = floor_radius
    if not np.isfinite(radius) or radius <= 0:
        radius = np.inf
    return {"centre": centre, "sigma": sigma, "radius": radius}


def ref_clean_by_mad(ml, st, k, floor_ml, floor_st, leave_one_out=True):
    """cage_cleaning.clean_by_mad (:188-267), axis-aligned (orient=False): distances, keep mask and the geometry each
    cage was judged by."""
    ml, st = np.asarray(ml, float), np.asarray(st, float)
    m = len(ml)

    def radius_of(sub_ml, sub_st):
        return ref_axis_geometry(sub_ml, k, floor_ml), ref_axis_geometry(sub_st, k, floor_st)
    geos = []
    if leave_one_out and m > 2:
        distance = np.empty(m)
        for i in range(m):
            others = np.arange(m) != i
            gx, gy = radius_of(ml[others], st[others])
            distance[i] = np.sqrt(((ml[i] - gx["centre"]) / gx["radius"]) ** 2
                                  + ((st[i] - gy["centre"]) / gy["radius"]) ** 2)
            geos.append((gx, gy))
    else:
        gx, gy = radius_of(ml, st)
        distance = np.sqrt(((ml - gx["centre"]) / gx["radius"]) ** 2 + ((st - gy["centre"]) / gy["radius"]) ** 2)
        geos = [(gx, gy)] * m
    distance = np.asarray(np.where(np.isfinite(distance), distance, 0.0), dtype=float)
    return distance, distance <= 1.0, geos


def ref_pooled(groups, min_cages=10):
    """cage_cleaning.pooled_relative_dispersion (:126-146) on {cohort: (ml, st)}."""
    rel_ml, rel_st = [], []
    for ml, st in groups.values():
        if len(ml) < min_cages:
            continue
        for values, sink in ((np.asarray(ml, float), rel_ml), (np.asarray(st, float), rel_st)):
            centre = np.median(values[np.isfinite(values)]) if np.isfinite(values).any() else np.nan
            sigma = ref_robust_sigma(values)
            if np.isfinite(centre) and centre != 0 and np.isfinite(sigma):
                sink.append(abs(sigma / centre))
    return float(np.median(rel_ml)), float(np.median(rel_st))


# ------------------------------------------------------------------------------------- HR-2's fixture
def ages_of(center, width, n=40):
    """n deaths at Gaussian quantile ages around `center` (relative sd `width`), on a 1-day census (ties)."""
    return np.round(center * (1 + width * norm.ppf(np.linspace(0.02, 0.98, n))))


def fixture_cages(seed=0) -> dict:
    """{cohort: [ages per cage]}: `big` (12 cages, the floors), `out` (cage 5 a planted outlier), `three`, `two`,
    `tied` (cages 0-2 identical; cage 4 beyond the floor radius) and `four` (where leave-one-out matters)."""
    rng = np.random.default_rng(seed)
    g = {}
    g["big"] = [ages_of(50 * (1 + 0.06 * rng.standard_normal()), 0.2 * (1 + 0.15 * rng.standard_normal()))
                for _ in range(12)]
    g["out"] = [ages_of(50 * (1 + 0.04 * rng.standard_normal()), 0.2) for _ in range(5)] + [ages_of(75, 0.2)]
    g["three"] = [ages_of(50 * (1 + 0.05 * rng.standard_normal()), 0.2 * (1 + 0.1 * rng.standard_normal()))
                  for _ in range(3)]
    g["two"] = [ages_of(40, 0.2), ages_of(60, 0.3)]
    base = ages_of(50, 0.2)
    g["tied"] = [base, base.copy(), base.copy(), ages_of(53, 0.2), ages_of(59, 0.2)]
    g["four"] = [ages_of(50 * (1 + 0.03 * rng.standard_normal()), 0.2 * (1 + 0.15 * rng.standard_normal()))
                 for _ in range(4)]
    return g


def as_set(groups, time_unit="days") -> CohortSet:
    members = []
    for co, cages in groups.items():
        for i, a in enumerate(cages):
            members.append((f"{co}/{i}", Cohort.from_arrays(a, np.ones(a.size, int), delta=1.0, name=f"{co}/{i}",
                                                            time_unit=time_unit), {"cohort": co, "cage": i}))
    return CohortSet.from_members(members, time_unit=time_unit)


@pytest.fixture(scope="module")
def fixture():
    return as_set(fixture_cages())


@pytest.fixture(scope="module")
def report(fixture):
    return clean(fixture, by="cohort", k=K, seed=SEED, noise_reps=500)


def rows(rep, cohort):
    return [r for r in rep.cages if r.cohort == cohort]


def verdicts(rep) -> dict:
    return {r.key: r.verdict for r in rep.cages}


@pytest.fixture
def srk_dataset(request):
    """SRK's Dataset from the frozen snapshot tests/oracle/v033 (oracle group; skipped without it)."""
    pytest.importorskip("lifelines")
    request.getfixturevalue("oracle_v033")
    return importlib.import_module("SRK.deathTimesDataSet").Dataset


# ------------------------------------------------------------------------------------------ T016 (HR-2)
def test_a_planted_outlier_cage_is_flagged_and_nothing_is_removed(fixture, report):
    out = {r.cage: r.verdict for r in rows(report, "out")}
    assert out == {0: "kept", 1: "kept", 2: "kept", 3: "kept", 4: "kept", 5: "flagged"}
    assert len(fixture) == 32 and len(report.cages) == 32                                  # clean removes nothing
    c = next(c for c in report.cohorts if c.cohort == "out")
    assert (c.k, c.axes, c.n_cages, c.n_flagged, c.loo) == (K, "both", 6, 1, True)


def test_ties_and_a_three_cage_cohort_follow_leave_one_out(report):
    """Each cage of `three` and `tied` is judged against the others (cage_cleaning.py:232): its centre is the median
    of the other cages' scores."""
    for co in ("three", "tied"):
        rs = rows(report, co)
        assert all(r.loo for r in rs) and next(c for c in report.cohorts if c.cohort == co).loo
        ml = np.array([r.median for r in rs])
        for i, r in enumerate(rs):
            assert r.centre_ml == float(np.median(np.delete(ml, i)))


def test_a_two_cage_cohort_uses_whole_group_geometry_and_says_so(report):
    rs = rows(report, "two")
    c = next(c for c in report.cohorts if c.cohort == "two")
    assert not c.loo and "whole-group" in c.rule and not any(r.loo for r in rs)
    assert rs[0].centre_ml == rs[1].centre_ml == (rs[0].median + rs[1].median) / 2
    assert {r.verdict for r in rs} == {"kept"}


def test_more_than_half_tied_raise_the_radius_to_the_floor(report):
    """Cage 4 of `tied`: three of its four peers are identical, so their MAD is 0 and both radii are the floor,
    k φ |centre|; it is flagged there, and kept without floors (no floors: radius inf)."""
    r = rows(report, "tied")[4]
    phi_ml, phi_st = report.floors
    assert r.ml_radius_is_floor and r.st_radius_is_floor
    assert r.radius_ml == K * phi_ml * abs(r.centre_ml) and r.radius_st == K * phi_st * abs(r.centre_st)
    assert r.verdict == "flagged"


def test_without_floors_the_tied_cohorts_deviant_is_kept(fixture):
    rep = clean(fixture, by="cohort", k=K, seed=SEED, floors=(0.0, 0.0), noise_reps=50)
    r = rows(rep, "tied")[4]
    assert r.radius_ml == r.radius_st == math.inf and r.distance == 0.0 and r.verdict == "kept"


def test_a_zero_iqr_cage_is_refused_on_both_axes_and_reported_on_the_median_axis():
    """D-503 (C11): with first-crossing quartiles a cage whose KM falls through 0.75 and 0.25 within one census has a
    non-finite steepness. On axes="both" the call is refused, naming the cage and the axis; on axes="median" it is
    reported and the verdicts come from the median alone. The legacy rule kept it at distance 0."""
    g = fixture_cages()
    g["zero"] = [ages_of(50, 0.2), ages_of(52, 0.2), np.array([40.0] + [50.0] * 38 + [60.0]), ages_of(48, 0.2)]
    s = as_set(g)
    with pytest.raises(CohortDataError, match="zero/2") as ex:
        clean(s, by="cohort", k=K, seed=SEED, quartiles="first-crossing", noise_reps=50)
    assert "steepness" in str(ex.value) and "D-503" in str(ex.value)
    rep = clean(s, by="cohort", k=K, seed=SEED, quartiles="first-crossing", axes="median", noise_reps=50)
    z = rows(rep, "zero")[2]
    assert not math.isfinite(z.steepness) and z.verdict in ("kept", "flagged")
    assert all(math.isnan(r.radius_st) for r in rep.cages) and {c.axes for c in rep.cohorts} == {"median"}
    d_ref, _, _ = ref_clean_by_mad([r.median for r in rows(rep, "zero")], [0.0] * 4, K, rep.floors[0], None)
    assert [r.distance for r in rows(rep, "zero")] == d_ref.tolist()
    cens = dict(g, zero=[ages_of(50, 0.2), ages_of(52, 0.2), ages_of(48, 0.2)])
    s2 = as_set(cens)
    m = dict(s2.cohorts)
    m["zero/1"] = Cohort.from_arrays(np.arange(1.0, 11.0), [1, 1] + [0] * 8, delta=1.0, name="zero/1",
                                     time_unit="days")
    s3 = CohortSet(m, design=s2.design, time_unit="days")
    with pytest.raises(CohortDataError, match="zero/1.*median|median.*zero/1"):            # no median: refused
        clean(s3, by="cohort", k=K, seed=SEED, axes="median", noise_reps=50)


def test_no_condition_with_min_cages_and_no_floors_is_refused(fixture):
    small = {k: v for k, v in fixture_cages().items() if k != "big"}
    with pytest.raises(CohortSpecError, match="floors") as ex:
        clean(as_set(small), by="cohort", k=K, seed=SEED, noise_reps=50)
    assert "10" in str(ex.value)
    rep = clean(as_set(small), by="cohort", k=K, seed=SEED, floors=(0.05, 0.1), noise_reps=50)        # the twin
    assert rep.floors == (0.05, 0.1) and rep.floors_source == "given"


def test_k_and_seed_are_required(fixture):
    with pytest.raises(TypeError, match="'k'"):
        clean(fixture, by="cohort", seed=SEED)
    with pytest.raises(TypeError, match="'seed'"):
        clean(fixture, by="cohort", k=K)
    for bad in (0, -1.0, math.inf, True):
        with pytest.raises(UnsupportedOptionError, match="k"):
            clean(fixture, by="cohort", k=bad, seed=SEED)
    with pytest.raises(UnsupportedOptionError, match="axes"):
        clean(fixture, by="cohort", k=K, seed=SEED, axes="steepness")
    with pytest.raises(UnsupportedOptionError, match="median_def"):
        clean(fixture, by="cohort", k=K, seed=SEED, median_def="nearest")
    with pytest.raises(UnsupportedOptionError, match="'arm'"):
        clean(fixture, by="arm", k=K, seed=SEED)


def test_clean_flags_and_drop_removes_exactly_the_flagged_recording_the_report(fixture, report):
    """D-504 (Naveh's Da): clean never removes; drop(set, report) removes exactly the flagged cages and records the
    report's sha256; a report computed on another set is refused."""
    flagged = {r.key for r in report.cages if r.verdict == "flagged"}
    assert flagged and set(fixture) == {r.key for r in report.cages}
    cleaned = drop(fixture, report)
    assert set(cleaned) == set(fixture) - flagged and len(fixture) == 32
    assert any(report.sha256 in n and "dropped" in n for n in cleaned.notes)
    assert len(report.sha256) == 64 and report.sha256 == clean(fixture, by="cohort", k=K, seed=SEED,
                                                               noise_reps=500).sha256
    other = as_set({k: v for k, v in fixture_cages().items() if k != "two"})
    with pytest.raises(CohortSpecError, match="two/0"):
        drop(other, report)
    m = dict(fixture.cohorts)
    m["out/0"] = Cohort.from_arrays(ages_of(51, 0.2), np.ones(40, int), delta=1.0, time_unit="days")
    with pytest.raises(CohortSpecError, match="out/0"):                                      # same key, other data
        drop(CohortSet(m, design=fixture.design, time_unit="days"), report)


def test_the_report_names_its_definitions_floors_and_rule(report):
    assert report.definitions["median"].startswith("first-crossing")
    assert report.definitions["quartiles"] == "nearest" and "nearest" in report.definitions["steepness"]
    assert report.floors_source.startswith("pooled") and report.floor_cohorts == ("big",) and report.min_cages == 10
    assert (report.k, report.axes, report.leave_one_out, report.by, report.seed) == (K, "both", True, "cohort", SEED)


# ------------------------------------------------------------------- T017: side by side with the legacy formulas
def test_distances_radii_and_verdicts_equal_the_legacy_formulas(report):
    groups = {}
    for r in report.cages:
        groups.setdefault(r.cohort, []).append(r)
    phi = ref_pooled({co: ([r.median for r in rs], [r.steepness for r in rs]) for co, rs in groups.items()})
    assert report.floors == phi
    for co, rs in groups.items():
        d, keep, geos = ref_clean_by_mad([r.median for r in rs], [r.steepness for r in rs], K, *phi)
        assert [r.distance for r in rs] == d.tolist(), co
        assert [r.verdict == "kept" for r in rs] == keep.tolist(), co
        assert [r.radius_ml for r in rs] == [g[0]["radius"] for g in geos], co
        assert [r.radius_st for r in rs] == [g[1]["radius"] for g in geos], co
        assert [r.centre_st for r in rs] == [g[1]["centre"] for g in geos], co


@pytest.mark.parametrize("defect,kw", [("no leave-one-out", {"leave_one_out": False}),
                                       ("no floors", {"floors": (0.0, 0.0)}),
                                       ("k swapped", {"k": 3.0}),
                                       ("first-crossing quartiles", {"quartiles": "first-crossing"})])
def test_each_planted_defect_changes_a_verdict(fixture, report, defect, kw):
    planted = clean(fixture, by="cohort", seed=SEED, noise_reps=50, **{"k": K, **kw})
    flips = [k for k, v in verdicts(planted).items() if v != verdicts(report)[k]]
    assert flips, defect


def test_floors_of_zero_are_the_legacy_rule_without_floors(fixture):
    rep = clean(fixture, by="cohort", k=K, seed=SEED, floors=(0.0, 0.0), noise_reps=50)
    for co in ("big", "tied", "four", "two"):
        rs = rows(rep, co)
        d, _, _ = ref_clean_by_mad([r.median for r in rs], [r.steepness for r in rs], K, None, None)
        assert [r.distance for r in rs] == d.tolist(), co


def test_scores_are_srk_datasets(fixture, report, srk_dataset):
    """FR-014, C2: median and steepness equal SRK Dataset's getMedianLifetime and getSteepness (nearest) exactly."""
    for r in report.cages:
        c = fixture[r.key]
        ds = srk_dataset(np.asarray(c.ages), np.asarray(c.events))
        assert (r.median, r.steepness, r.n) == (ds.getMedianLifetime(), ds.getSteepness(), ds.n), r.key


# ------------------------------------------------------------------------------- T018: the noise baseline (R5)
def test_the_same_seed_gives_the_same_expected_counts_and_another_seed_changes_them():
    a = noise_baseline(5, K, floors=(0.17, 0.31), spread=(0.25, 0.4), reps=1000, seed=1)
    b = noise_baseline(5, K, floors=(0.17, 0.31), spread=(0.25, 0.4), reps=1000, seed=1)
    c = noise_baseline(5, K, floors=(0.17, 0.31), spread=(0.25, 0.4), reps=1000, seed=2)
    assert a == b
    assert (a["floor"]["expected_flags"], a["own_spread"]["expected_flags"]) != \
        (c["floor"]["expected_flags"], c["own_spread"]["expected_flags"])


def test_both_nulls_are_always_present_with_reps_seed_and_names(report):
    for c in report.cohorts:
        assert [n["null"] for n in c.noise] == ["floor", "own_spread"]
        assert all(n["reps"] == 500 and n["seed"] == SEED for n in c.noise)
        assert c.noise[0]["s_ml"] == report.floors[0] and c.noise[0]["s_st"] == report.floors[1]
    big = next(c for c in report.cohorts if c.cohort == "big")
    assert big.noise[1]["pool"] == "all" and math.isfinite(big.noise[1]["expected_flags"])
    with pytest.raises(UnsupportedOptionError, match="spread"):
        noise_baseline(5, K, floors=(0.17, 0.31), reps=100, seed=1)


def test_the_floor_null_rate_at_m3_k25_is_about_9_percent_not_the_analytic_4_4():
    """Gate M5: 9.0 % at m = 3, k = 2.5 (the fly's null_drops.csv); within 4 binomial SE on R = 4,000, where the
    analytic exp(-k^2/2) = 4.4 % lies outside. The rate rises slowly with the floors (measured: 8.8-9.0 % at
    phi = (0.05, 0.1), about 10 % at (0.17, 0.31), about 11.6 % at (0.3, 0.5)), since the floor radius scales with the
    others' centre; the fly's own cells, at its floors, are phase noise's (HR-4)."""
    phi = (0.05, 0.1)
    out = noise_baseline(3, K, floors=phi, spread=phi, reps=4000, seed=SEED)
    rate = out["floor"]["rate"]
    se = math.sqrt(0.09 * 0.91 / (4000 * 3))
    assert abs(rate - 0.090) <= 4 * se, rate
    assert abs(math.exp(-K * K / 2) - rate) > 4 * se
    assert out["floor"]["expected_flags"] == pytest.approx(rate * 3)


def test_the_null_applies_the_same_rule():
    """The Monte Carlo's batch rule flags exactly what the reference flags, draw by draw."""
    rng = np.random.default_rng(7)
    for m, loo in ((3, True), (6, True), (2, True), (5, False)):
        ml = 1 + 0.2 * rng.standard_normal((300, m))
        st = 1 + 0.3 * rng.standard_normal((300, m))
        got = CL._flags_batch(ml, st, K, (0.17, 0.31), "both", loo)
        want = np.array([~ref_clean_by_mad(ml[i], st[i], K, 0.17, 0.31, loo)[1] for i in range(300)])
        np.testing.assert_array_equal(got, want)
        got1 = CL._flags_batch(ml, st, K, (0.17, 0.31), "median", loo)                  # 1-D: no steepness axis
        want1 = np.array([~ref_clean_by_mad(ml[i], np.zeros(m), K, 0.17, None, loo)[1] for i in range(300)])
        np.testing.assert_array_equal(got1, want1)


# ------------------------------------------------------------- the own-spread null's pool (sr-naveh_005 B2, Naveh)
def scores_of(rep) -> dict:
    out = {}
    for r in rep.cages:
        ml, st = out.setdefault(r.cohort, ([], []))
        ml.append(r.median)
        st.append(r.steepness)
    return out


def tool_group_spread(cohorts, *, seed):
    """HR-4's accepted construction as tools/acceptance_005.group_spread computed it before B2 (the fly redo's
    group_spreads), returning s, not s / phi."""
    dm, ds = [], []
    for ml, st in cohorts.values():
        r = CL.loo_correction(len(ml), seed=seed)
        dm += list(CL.loo_relative_deviations(ml) / r)
        ds += list(CL.loo_relative_deviations(st) / r)
    return CL.robust_sigma(dm), CL.robust_sigma(ds)


def with_line(groups, line) -> CohortSet:
    """as_set with a design column `line` ({cohort: value})."""
    members = []
    for co, cages in groups.items():
        for i, a in enumerate(cages):
            members.append((f"{co}/{i}", Cohort.from_arrays(a, np.ones(a.size, int), delta=1.0, name=f"{co}/{i}",
                                                            time_unit="days"),
                            {"cohort": co, "cage": i, "line": line[co]}))
    return CohortSet.from_members(members, time_unit="days")


def test_the_pooled_spread_is_hr4s_accepted_construction(report):
    """pooled_spread equals the tool's construction exactly, on the fixture's scores, pooled and one cohort at a time."""
    sc = scores_of(report)
    assert CL.pooled_spread(sc, seed=SEED) == tool_group_spread(sc, seed=SEED)
    for co in sc:
        assert CL.pooled_spread({co: sc[co]}, seed=SEED) == tool_group_spread({co: sc[co]}, seed=SEED), co
    assert all(math.isnan(x) for x in CL.pooled_spread({"one": ([1.0], [2.0])}, seed=SEED))     # no cohort of 2


def test_clean_pools_over_every_cohort_and_names_the_pool(fixture, report):
    sc = scores_of(report)
    want = CL.pooled_spread(sc, seed=SEED)
    assert report.spread_by is None and report.as_dict()["spread_by"] is None
    for c in report.cohorts:
        o = c.noise[1]
        assert (o["spread_by"], o["pool"], o["pool_cohorts"], o["pool_cages"]) == (None, "all", 6, 32)
        assert (o["s_ml"], o["s_st"]) == want and math.isfinite(o["expected_flags"]) and "reason" not in o


def test_a_three_cage_cohort_with_two_tied_cages_gets_the_pools_nonzero_spread():
    """B2: on its own, a 3-cage cohort with two tied cages has s = 0 (the MAD of [d, d, e] is 0), so a per-cohort null
    would draw no scatter and read 0.00 expected flags; pooled with the others, it reads the pool's s > 0."""
    g = fixture_cages()
    base = ages_of(50, 0.2)
    g["tie3"] = [base, base.copy(), ages_of(56, 0.25)]
    rep = clean(as_set(g), by="cohort", k=K, seed=SEED, noise_reps=200)
    own = CL.pooled_spread({"tie3": scores_of(rep)["tie3"]}, seed=SEED)
    assert own == (0.0, 0.0)
    o = next(c for c in rep.cohorts if c.cohort == "tie3").noise[1]
    assert o["s_ml"] > 0 and o["s_st"] > 0 and o["expected_flags"] > 0 and o["pool_cohorts"] == 7


def test_spread_by_pools_within_a_design_value_and_an_all_tied_pool_is_not_measurable():
    """spread_by="line": each cohort's pool is its line's cohorts. A pool whose cages all tie has s = 0: not measurable,
    with the reason, never 0.00 expected flags."""
    g = fixture_cages()
    base = ages_of(50, 0.2)
    g["flat1"], g["flat2"] = [base, base.copy(), base.copy()], [base.copy(), base.copy(), base.copy(), base.copy()]
    line = {co: ("A" if co.startswith("flat") else "B") for co in g}
    rep = clean(with_line(g, line), by="cohort", k=K, seed=SEED, noise_reps=200, spread_by="line")
    assert rep.spread_by == "line"
    sc = scores_of(rep)
    for c in rep.cohorts:
        o = c.noise[1]
        members = {co: sc[co] for co in sc if line[co] == line[c.cohort]}
        assert (o["spread_by"], o["pool"], o["pool_cohorts"]) == ("line", line[c.cohort], len(members))
        assert o["pool_cages"] == sum(len(ml) for ml, _ in members.values() if len(ml) >= 2)
        assert (o["s_ml"], o["s_st"]) == CL.pooled_spread(members, seed=SEED)
        if line[c.cohort] == "A":
            assert (o["s_ml"], o["s_st"]) == (0.0, 0.0) and math.isnan(o["expected_flags"])
            assert o["reason"].startswith("not measurable: s_ml = 0.0, s_st = 0.0")
        else:
            assert math.isfinite(o["expected_flags"]) and o["s_ml"] > 0


def test_a_zero_or_nonfinite_spread_is_not_measurable_in_the_noise_baseline():
    out = noise_baseline(3, K, floors=(0.05, 0.1), spread=(0.0, 0.2), reps=100, seed=1)
    assert math.isnan(out["own_spread"]["expected_flags"]) and "s_ml = 0.0" in out["own_spread"]["reason"]
    assert math.isfinite(out["floor"]["expected_flags"]) and "reason" not in out["floor"]
    out = noise_baseline(3, K, floors=(0.0, 0.0), spread=(0.05, math.nan), reps=100, seed=1)
    assert out["floor"]["reason"].startswith("not measurable: s_ml = 0.0, s_st = 0.0")      # no scatter: no null
    assert "s_st = nan" in out["own_spread"]["reason"] and "s_ml" not in out["own_spread"]["reason"]
    med = noise_baseline(3, K, floors=(0.05, math.nan), spread=(0.05, 0.0), reps=100, seed=1, axes="median")
    assert all(math.isfinite(med[n]["expected_flags"]) for n in ("floor", "own_spread"))   # st unused on "median"


@pytest.mark.parametrize("spread_by,match", [("arm", "'arm' is not a column"), ("cage", "varies within cohort"),
                                             (3, "is not a column")])
def test_spread_by_a_column_not_in_the_design_or_varying_within_a_cohort_is_refused(fixture, spread_by, match):
    with pytest.raises(UnsupportedOptionError, match=match) as ex:
        clean(fixture, by="cohort", k=K, seed=SEED, noise_reps=50, spread_by=spread_by)
    assert ex.value.option == "spread_by" and repr(spread_by) in str(ex.value)


def test_spread_by_the_cohort_column_is_the_per_cohort_spread(fixture, report):
    rep = clean(fixture, by="cohort", k=K, seed=SEED, noise_reps=50, spread_by="cohort")
    sc = scores_of(report)
    for c in rep.cohorts:
        o = c.noise[1]
        assert o["pool"] == c.cohort and (o["s_ml"], o["s_st"]) == CL.pooled_spread({c.cohort: sc[c.cohort]},
                                                                                     seed=SEED)
    assert [c.n_flagged for c in rep.cohorts] == [c.n_flagged for c in report.cohorts]     # verdicts unchanged
