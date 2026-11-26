"""Cohort fields added in spec 002 (data-model.md; research R8): entry, window, time_unit, per-row delta."""
import numpy as np
import pytest

from sraging import Cohort
from sraging.errors import CohortDataError, CohortSpecError

T = np.array([3.0, 5.0, 8.0, 9.0, 12.0])
E = np.array([1, 1, 0, 1, 0])


def test_defaults_keep_spec_001_cohorts_unchanged():
    c = Cohort.from_arrays(T, E, delta=1.0)
    assert c.entry is None and c.window is None and c.time_unit == "input units"
    assert not c.has_features and c.n_at_risk == c.n == 5


def test_entry_ages_are_read_only_and_validated():
    c = Cohort.from_arrays(T, E, delta=1.0, entry=[0.0, 2.0, 8.0, 9.5, 1.0])
    assert c.has_features and not c.entry.flags.writeable
    # a death may enter inside its recording interval [t, t+delta): 9.5 < 9 + 1
    with pytest.raises(CohortDataError) as ex:
        Cohort.from_arrays(T, E, delta=1.0, entry=[0.0, 6.0, 9.0, 0.0, -1.0])
    msgs = dict(ex.value.rows)
    assert "at or after the end of the death's recording interval" in msgs[2]   # 6 >= 5 + 1
    assert "after the recorded age" in msgs[3]                                  # censored at 8, entry 9
    assert "finite and >= 0" in msgs[5]


def test_entry_with_exact_times_must_not_follow_the_age():
    with pytest.raises(CohortDataError, match="entry ages refused"):
        Cohort.from_arrays(T, E, continuous=True, entry=[0.0, 0.0, 0.0, 9.5, 0.0])


def test_window_on_the_grid_counts_the_excluded_rows():
    c = Cohort.from_arrays(T, E, delta=1.0, window=(4.0, 10.0))
    assert c.window == (4.0, 10.0) and c.n_excluded_by_window == 1 and c.n_at_risk == 4
    assert Cohort.from_arrays(T, E, delta=1.0, window=(0.0, np.inf)).window == (0.0, np.inf)


def test_window_off_the_grid_is_refused_with_the_nearest_grid_ages():
    with pytest.raises(CohortSpecError, match=r"t_lo = 4.5 is not on the recording grid.*4.0 and 5.0"):
        Cohort.from_arrays(T, E, delta=1.0, window=(4.5, 10.0))


@pytest.mark.parametrize("kw,match", [
    (dict(window=(5.0, 4.0)), "0 <= t_lo < t_hi"),
    (dict(window=(-1.0, 4.0)), "0 <= t_lo < t_hi"),
    (dict(window=(1.0, 4.0), entry=[0.0] * 5), "either per-row entry ages or a common window"),
    (dict(entry=[0.0] * 4), "one age per row"),
    (dict(time_unit=7), "time_unit must be a string"),
])
def test_contradictory_specifications_are_refused(kw, match):
    with pytest.raises(CohortSpecError, match=match):
        Cohort.from_arrays(T, E, delta=1.0, **kw)


def test_per_row_delta():
    c = Cohort.from_arrays(T, E, delta=[1.0, 1.0, 2.0, 1.0, 1.0])
    assert c.delta.shape == (5,) and not c.delta.flags.writeable and c.has_features
    with pytest.raises(CohortSpecError, match="one value per row"):
        Cohort.from_arrays(T, E, delta=[1.0, 1.0])
    with pytest.raises(CohortSpecError, match="finite and > 0 in every row"):
        Cohort.from_arrays(T, E, delta=[1.0, 0.0, 1.0, 1.0, 1.0])
    with pytest.raises(CohortSpecError, match="needs one recording resolution"):
        Cohort.from_arrays(T, E, delta=[1.0] * 5, window=(1.0, 4.0))


def test_a_window_with_continuous_times_has_no_grid_to_check():
    c = Cohort.from_arrays(T, E, continuous=True, window=(4.25, 10.5))
    assert c.window == (4.25, 10.5)


@pytest.mark.parametrize("kw", [dict(window=(1.0, 10.0)), dict(entry=[0.0] * 5),
                                dict(delta=[1.0, 1.0, 2.0, 1.0, 1.0])])
def test_spec_001_calibrator_refuses_the_new_fields(kw):
    from sraging import fit_cohort
    from sraging.errors import UnsupportedOptionError
    kw = {"delta": 1.0, **kw}
    with pytest.raises(UnsupportedOptionError, match="has no likelihood for"):
        fit_cohort(Cohort.from_arrays(T, E, **kw), calibrator="profile_v033")        # named since spec 003g T052
