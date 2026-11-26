"""The Kaplan-Meier estimator and the summary statistics (spec 005 US3; FR-010 to FR-012; research R2, R3; data-model
"KMResult", "SummaryStatistics"; tasks T011, T012; register D-501). HR-3 side by side with lifelines 0.30.3, and the
`nearest` quartiles side by side with SRK Dataset's getSteepness from the frozen snapshot (oracle group). Synthetic data
only."""
import importlib
import itertools
import math
import time

import numpy as np
import pytest

from sraging.cohort import Cohort
from sraging.errors import CohortDataError, UnsupportedOptionError
from sraging.survival import KMResult, Stat, km, summary

Z95 = 1.959963984540054
TIES_T = (1.0, 2.0, 2.0, 2.0, 3.0)
TIES_E = (1, 1, 0, 0, 1)                      # one death and two censorings at t = 2 (sec03_likelihood.tex:316-319)


def legacy_row_walk(ages, events, order):
    """SRK's fp_likelihood.kaplan_meier's walk (sec03_likelihood.tex eq. s3-km), over the row order given: S <- S(1 -
    1/r) at a death, r <- r - 1 every row. Returns S after the last row at each distinct age."""
    a, e = np.asarray(ages, float)[order], np.asarray(events, int)[order]
    r, S, out = len(a), 1.0, {}
    for t, d in zip(a, e):
        if d:
            S *= 1.0 - 1.0 / r
        r -= 1
        out[float(t)] = S
    return out


def _random_cohorts(seed, reps=40):
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        n = int(rng.integers(3, 400))
        ages = rng.integers(1, 30, n).astype(float) * float(rng.choice([1.0, 0.5, 2.0]))
        events = (rng.random(n) < rng.uniform(0.4, 1.0)).astype(int)
        events[rng.integers(n)] = 1
        yield ages, events


@pytest.fixture
def lifelines():
    return pytest.importorskip("lifelines")


@pytest.fixture
def srk_dataset(request):
    """SRK's Dataset from the frozen snapshot tests/oracle/v033 (oracle group; skipped without it)."""
    pytest.importorskip("lifelines")
    request.getfixturevalue("oracle_v033")
    return importlib.import_module("SRK.deathTimesDataSet").Dataset


# --------------------------------------------------------------------------------------------- the KM (T011)
def test_deaths_come_before_tied_censorings_in_every_row_order():
    """FR-010: Ŝ(2) is 0.600 (deaths first) for every permutation of the rows. The planted twin, SRK's row walk with a
    censoring before the death at 2 (what its unstable argsort can return), gives 0.533."""
    for order in itertools.permutations(range(5)):
        k = km((np.asarray(TIES_T)[list(order)], np.asarray(TIES_E)[list(order)]))
        assert k.S_at([2.0])[0] == pytest.approx(0.6, abs=1e-15)
    assert legacy_row_walk(TIES_T, TIES_E, [0, 1, 2, 3, 4])[2.0] == pytest.approx(0.6, abs=1e-15)    # deaths first
    planted = legacy_row_walk(TIES_T, TIES_E, [0, 2, 1, 3, 4])[2.0]                                 # a censoring first
    assert planted == pytest.approx(8 / 15, abs=1e-15) and abs(planted - 0.6) > 0.06


def test_vertices_at_risk_and_deaths_by_hand():
    k = km((TIES_T, TIES_E))
    np.testing.assert_array_equal(k.times, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(k.at_risk, [5, 4, 1])
    np.testing.assert_array_equal(k.deaths, [1, 1, 1])
    np.testing.assert_allclose(k.S, [0.8, 0.6, 0.0], rtol=0, atol=1e-15)
    assert (k.n, k.n_deaths, k.n_censored) == (5, 3, 2) and k.condition is None
    np.testing.assert_allclose(k.S_at([0.0, 0.99, 1.0, 2.5, 3.0, 9.0]), [1, 1, 0.8, 0.6, 0, 0], atol=1e-15)
    k2 = km((np.arange(1.0, 11.0), np.ones(10, int)))                                    # n = 10, no censoring
    np.testing.assert_array_equal(k2.at_risk, np.arange(10, 0, -1))
    np.testing.assert_allclose(k2.S, np.arange(9, -1, -1) / 10, atol=1e-15)


def test_the_band_is_the_log_minus_log_greenwood_closed_form():
    """R2: V = Σ d/(r(r - d)); lower = Ŝ^exp(z √V / |log Ŝ|), upper = Ŝ^exp(-z √V / |log Ŝ|); at Ŝ = 0 the band is
    [0, 0], and before the first death [1, 1] (lifelines' conventions)."""
    k = km((TIES_T, TIES_E))
    assert k.band == "log(-log) Greenwood" and k.level == 0.95 and k.z == Z95
    S, r, d = np.array([0.8, 0.6]), np.array([5.0, 4.0]), np.array([1.0, 1.0])
    V = np.cumsum(d / (r * (r - d)))
    a = Z95 * np.sqrt(V) / np.abs(np.log(S))
    np.testing.assert_allclose(k.S_lo[:2], S ** np.exp(a), rtol=1e-14)
    np.testing.assert_allclose(k.S_hi[:2], S ** np.exp(-a), rtol=1e-14)
    assert k.S_lo[2] == 0.0 and k.S_hi[2] == 0.0
    lo, hi = k.band_at([0.5, 1.5])
    assert (lo[0], hi[0]) == (1.0, 1.0) and (lo[1], hi[1]) == (k.S_lo[0], k.S_hi[0])
    assert km((TIES_T, TIES_E), level=0.9).z == pytest.approx(1.6448536269514722)


def test_against_lifelines_s_band_and_median(lifelines):
    """HR-3: S at the vertices within 1e-12, the band within 1e-10 and the median identical, on random cohorts with
    ties (S is computed as lifelines computes it, exp of the cumulated log(r - d) - log r, so it agrees exactly)."""
    worst_s = worst_b = 0.0
    for ages, events in _random_cohorts(5):
        k = km((ages, events))
        ref = lifelines.KaplanMeierFitter().fit(ages, events)
        sf = ref.survival_function_.iloc[:, 0]
        ci = ref.confidence_interval_
        worst_s = max(worst_s, np.max(np.abs(k.S - sf.loc[k.times].to_numpy())))
        worst_b = max(worst_b, np.max(np.abs(k.S_lo - ci.iloc[:, 0].loc[k.times].to_numpy())),
                      np.max(np.abs(k.S_hi - ci.iloc[:, 1].loc[k.times].to_numpy())))
        assert summary(k).median.value == ref.median_survival_time_
    assert worst_s <= 1e-12 and worst_b <= 1e-10, (worst_s, worst_b)
    six = np.arange(5.0, 11.0)          # exact Ŝ(7) = 1/2; computed one ulp above it, by lifelines and here alike
    assert summary(km((six, np.ones(6, int)))).median.value == \
        lifelines.KaplanMeierFitter().fit(six, np.ones(6)).median_survival_time_ == 8.0


def test_a_cohort_and_its_arrays_give_the_same_km_and_unsupported_cohorts_are_refused():
    c = Cohort.from_arrays([3.0, 1.0, 2.0, 2.0], [1, 1, 0, 1], delta=1.0, time_unit="days")
    k = km(c)
    assert isinstance(k, KMResult) and k.time_unit == "days"
    np.testing.assert_array_equal(k.S, km((c.ages, c.events)).S)
    e = Cohort.from_arrays([3.0, 4.0], [1, 1], delta=1.0, entry=[1.0, 0.0])
    with pytest.raises(UnsupportedOptionError, match="entry"):
        km(e)
    with pytest.raises(CohortDataError, match="event"):
        km(([1.0, 2.0], [1, 2]))
    with pytest.raises(CohortDataError, match="negative|finite"):
        km(([1.0, -2.0], [1, 1]))


def test_n_1e5_runs_in_under_a_second():
    """A smoke bound for R2's O(n log n), not a claim."""
    rng = np.random.default_rng(3)
    ages = np.round(rng.weibull(3.0, 100_000) * 50, 1) + 0.1
    events = (rng.random(100_000) < 0.9).astype(int)
    t0 = time.perf_counter()
    k = km((ages, events))
    summary(k, quartiles="nearest")
    assert time.perf_counter() - t0 < 1.0 and k.n == 100_000


# --------------------------------------------------------------------------------------- statistics (T012)
def test_the_median_is_first_crossing_with_its_ci_from_the_band():
    ages = np.arange(1.0, 11.0)
    s = summary(km((ages, np.ones(10, int))))
    k = km((ages, np.ones(10, int)))
    assert s.median.value == 5.0 and "first-crossing" in s.median.definition
    lo = k.times[np.argmax(k.S_lo <= 0.5)]
    hi = k.times[np.argmax(k.S_hi <= 0.5)]
    assert s.median.ci == (lo, hi) and lo <= 5.0 <= hi
    never = summary(km(([1.0, 2.0, 3.0, 4.0], [1, 0, 0, 0])))
    assert never.median.value == math.inf and not math.isfinite(never.median.ci[1])


def test_every_stat_names_its_definition_and_quartiles_default_to_first_crossing():
    """FR-011, D-501: the summary's steepness uses first-crossing quartiles unless `nearest` is named, and every Stat
    carries its definition string; the definitions differ on a tied cohort."""
    ages = np.array([1, 2, 2, 3, 3, 3, 5, 6, 6, 8], float)          # Ŝ: .9 .7 .4 .3 .1 0, no crossing within ulps
    k = km((ages, np.ones(ages.size, int)))
    s = summary(k)
    assert s.quartiles == "first-crossing"
    for name in ("median", "q25_age", "q75_age", "steepness", "restricted_mean", "cv"):
        st = getattr(s, name)
        assert isinstance(st, Stat) and isinstance(st.definition, str) and st.definition
    assert "first-crossing" in s.q75_age.definition and "first-crossing" in s.steepness.definition
    n = summary(k, quartiles="nearest")
    assert n.quartiles == "nearest" and "nearest" in n.q25_age.definition and "nearest" in n.steepness.definition
    assert (s.q75_age.value, s.q25_age.value) == (2.0, 6.0) and (n.q75_age.value, n.q25_age.value) == (2.0, 5.0)
    assert s.steepness.value == pytest.approx(3.0 / 4.0) and n.steepness.value == pytest.approx(3.0 / 3.0)
    with pytest.raises(UnsupportedOptionError, match="quartiles"):
        summary(k, quartiles="linear")


def test_nearest_matches_srk_dataset_side_by_side_on_cohorts_with_ties(srk_dataset):
    """FR-014, C2: `nearest` quartiles and the median equal SRK Dataset's getSteepness and getMedianLifetime (frozen
    snapshot), exactly, on random cohorts with ties."""
    for ages, events in _random_cohorts(11, reps=30):
        ds = srk_dataset(ages, events)
        s = summary(km((ages, events)), quartiles="nearest")
        assert s.median.value == ds.getMedianLifetime()
        assert s.steepness.value == ds.getSteepness() or (math.isnan(s.steepness.value)
                                                          and math.isnan(ds.getSteepness()))


def test_a_zero_first_crossing_iqr_gives_a_non_finite_steepness():
    """C11: one census crossing both quartiles (the fly's 2-8 LONG); `nearest` stays finite."""
    ages = np.array([1.0] + [2.0] * 8 + [3.0])
    k = km((ages, np.ones(10, int)))
    s = summary(k)
    assert s.q75_age.value == s.q25_age.value == 2.0 and not math.isfinite(s.steepness.value)
    n = summary(k, quartiles="nearest")
    assert (n.q75_age.value, n.q25_age.value) == (1.0, 2.0) and n.steepness.value == pytest.approx(2.0)


def test_restricted_mean_and_cv_are_labelled_restricted_when_the_last_observation_is_censored():
    complete = summary(km(([1.0, 2.0, 3.0], [1, 1, 1])))
    assert complete.restricted_mean.value == pytest.approx(2.0) and not complete.restricted
    assert "restricted" not in complete.restricted_mean.definition and complete.tau == 3.0
    sd = math.sqrt((1 + 4 + 9) / 3 - 4.0)
    assert complete.cv.value == pytest.approx(sd / 2.0)
    cens = summary(km(([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 0])))
    assert cens.restricted and cens.tau == 4.0
    assert "restricted" in cens.restricted_mean.definition and "restricted" in cens.cv.definition
    assert cens.restricted_mean.value == pytest.approx(1 + 0.75 + 0.5 + 0.25)
    short = summary(km(([1.0, 2.0, 3.0], [1, 1, 1])), horizon=2.5)
    assert short.restricted and short.tau == 2.5 and short.restricted_mean.value == pytest.approx(1 + 2 / 3 + 0.5 / 3)


def test_conditional_statistics_condition_on_survival_to_t0():
    ages = np.arange(1.0, 11.0)
    k = km((ages, np.ones(10, int)), condition_on=3.0)
    assert k.condition == (3.0, pytest.approx(0.7)) and k.n == 7
    full = km((ages, np.ones(10, int)))
    np.testing.assert_allclose(k.S, full.S[3:] / full.S_at([3.0])[0], atol=1e-15)
    s = summary(Cohort.from_arrays(ages, np.ones(10, int), delta=1.0), condition_on=3.0)
    assert s.condition == k.condition and s.median.value == 7.0          # Ŝ(7 | 3) = 3/7
    assert "conditional on survival to 3" in s.median.definition and "conditional" in s.cv.definition
    assert s.restricted_mean.value == pytest.approx(3.0 + sum(np.arange(7, 0, -1) / 7))
    assert summary(full, condition_on=3.0).median.value == 7.0
    with pytest.raises(UnsupportedOptionError, match="condition_on"):
        summary(k, condition_on=5.0)
    with pytest.raises(UnsupportedOptionError, match="condition_on"):
        km((ages, np.ones(10, int)), condition_on=10.0)
