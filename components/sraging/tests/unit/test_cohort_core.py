"""Cohort core: validation, the delta/continuous rule, immutability, reference construction (T018, T019)."""
import dataclasses

import numpy as np
import pytest

from sraging.cohort import Cohort
from sraging.errors import CohortDataError, CohortSpecError

AGES = [3.0, 5.0, 8.0, 13.0]
EVENTS = [1, 1, 0, 1]


def test_arrays_are_built_like_the_reference_and_read_only():
    c = Cohort.from_arrays(AGES, EVENTS, delta=1.0, name="x")
    ref_a, ref_e = np.asarray(AGES, float), np.asarray(EVENTS, int)
    assert c.ages.dtype == ref_a.dtype and c.ages.tobytes() == ref_a.tobytes()
    assert c.events.dtype == ref_e.dtype and c.events.tobytes() == ref_e.tobytes()
    assert not c.ages.flags.writeable and not c.events.flags.writeable
    assert (c.n, c.n_deaths, c.n_censored) == (4, 3, 1)


def test_caller_arrays_are_not_frozen_or_aliased():
    a, e = np.asarray(AGES, float), np.asarray(EVENTS, int)
    c = Cohort.from_arrays(a, e, delta=1.0)
    assert a.flags.writeable and e.flags.writeable
    a[0] = 99.0
    assert c.ages[0] == 3.0


def test_frozen():
    c = Cohort.from_arrays(AGES, EVENTS, delta=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.delta = 2.0
    with pytest.raises(ValueError):
        c.ages[0] = 1.0


@pytest.mark.parametrize("kw", [dict(), dict(delta=1.0, continuous=True)])
def test_delta_rule_explains_why(kw):
    with pytest.raises(CohortSpecError, match="changes how every death is scored"):
        Cohort.from_arrays(AGES, EVENTS, **kw)


@pytest.mark.parametrize("delta", [0.0, -1.0, float("nan"), float("inf"), True])
def test_bad_delta(delta):
    with pytest.raises(CohortSpecError, match="delta"):
        Cohort.from_arrays(AGES, EVENTS, delta=delta)


def test_continuous_is_explicit():
    c = Cohort.from_arrays(AGES, EVENTS, continuous=True)
    assert c.delta is None and c.continuous


@pytest.mark.parametrize("ages,events,row,what", [
    ([3.0, float("nan"), 8.0], [1, 1, 1], 2, "not finite"),
    ([3.0, 0.0, 8.0], [1, 0, 1], 2, "censored at age 0"),
    ([3.0, -2.0, 8.0], [1, 1, 1], 2, "negative"),
    ([3.0, 5.0, 8.0], [1, 2, 1], 2, "not 0 or 1"),
    ([3.0, 5.0, 8.0], [1, 0.5, 1], 2, "not 0 or 1"),
    ([3.0, 5.0, 8.0], [1, float("nan"), 1], 2, "not 0 or 1"),
])
def test_refusals_name_the_row(ages, events, row, what):
    with pytest.raises(CohortDataError) as ei:
        Cohort.from_arrays(ages, events, delta=1.0)
    assert ei.value.rows == [(row, ei.value.rows[0][1])]
    assert what in ei.value.rows[0][1]


def test_all_offending_rows_are_listed():
    with pytest.raises(CohortDataError) as ei:
        Cohort.from_arrays([-1.0, float("inf"), 5.0], [1, 1, 3], delta=1.0)
    assert [r for r, _ in ei.value.rows] == [1, 2, 3]


@pytest.mark.parametrize("ages,events", [([], []), ([1.0, 2.0], [1]), ([[1.0], [2.0]], [[1], [1]])])
def test_shape_refusals(ages, events):
    with pytest.raises(CohortDataError):
        Cohort.from_arrays(ages, events, delta=1.0)


def test_no_deaths_refused():
    with pytest.raises(CohortDataError, match="no deaths"):
        Cohort.from_arrays([1.0, 2.0], [0, 0], delta=1.0)


def test_a_death_at_age_zero_with_delta_is_the_first_interval():
    """Naveh 2026-10-03: a death found at the first check is interval-censored in [0, delta) (D-105)."""
    c = Cohort.from_arrays([0.0, 2.0, 4.0], [1, 1, 0], delta=2.0)
    assert list(c.ages) == [0.0, 2.0, 4.0] and c.n_deaths == 2


def test_age_zero_is_refused_for_continuous_times():
    with pytest.raises(CohortDataError) as ei:
        Cohort.from_arrays([0.0, 2.0, 4.0], [1, 1, 0], continuous=True)
    assert ei.value.rows[0][0] == 1 and "continuous" in ei.value.rows[0][1]
