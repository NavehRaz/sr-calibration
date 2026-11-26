"""Cohort.from_csv: the reference header rule, and refusal of every row the reference would skip (T021).

Header rule: verbatim from tests/oracle/v033/SRtools_kappa/SRK/fit_cohort.py:52-80 (first match wins;
a numeric first row is data). Refusals: FR-003 (clarified 2026-10-03), register D-101.
"""
import numpy as np
import pytest

from sraging.cohort import Cohort
from sraging.errors import CohortDataError


def write(tmp_path, text, name="c.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


@pytest.mark.parametrize("text,ages,events", [
    ("death times,events\n3,1\n5,0\n8,1\n", [3.0, 5.0, 8.0], [1, 0, 1]),
    ("time,death\n3,1\n5,0\n", [3.0, 5.0], [1, 0]),                  # icol=0 ('time'), ecol=1
    ("status,age\n1,3\n0,5\n", [3.0, 5.0], [1, 0]),                  # reversed order
    ("3,1\n5,0\n8,1\n", [3.0, 5.0, 8.0], [1, 0, 1]),                 # numeric first row is data
    ("age,censor\n2.5,1\n7.25,0\n", [2.5, 7.25], [1, 0]),
])
def test_header_rule(tmp_path, text, ages, events):
    c = Cohort.from_csv(write(tmp_path, text), delta=1.0)
    assert c.ages.tobytes() == np.asarray(ages, float).tobytes()
    assert c.events.tobytes() == np.asarray(events, int).tobytes()
    assert c.ages.dtype == np.float64 and c.events.dtype == np.asarray([1], int).dtype


def test_colliding_columns_refused_with_header(tmp_path):
    # no age-like column, an event-like column first: ecol=0, then icol defaults to 0 -> collision
    with pytest.raises(CohortDataError, match="could not tell the age column"):
        Cohort.from_csv(write(tmp_path, "status,value\n1,3\n"), delta=1.0)


def test_single_column_is_a_short_row(tmp_path):
    with pytest.raises(CohortDataError) as ei:
        Cohort.from_csv(write(tmp_path, "death_event\n3\n"), delta=1.0)
    assert ei.value.rows and "short row" in ei.value.rows[0][1]


def test_every_offending_row_is_listed_in_one_error(tmp_path):
    text = ("death times,events\n"
            "3,1\n"          # line 2 ok
            "5\n"            # line 3 short
            "x,1\n"          # line 4 non-numeric
            "nan,1\n"        # line 5 not finite
            "inf,0\n"        # line 6 not finite
            "0,0\n"          # line 7 censored at age 0 (dropped silently by the reference)
            "-2,1\n"         # line 8 negative
            "4,2\n"          # line 9 event 2 (recoded to 1 by the reference)
            "6,0.5\n"        # line 10 event 0.5
            "7,1\n")         # line 11 ok
    with pytest.raises(CohortDataError) as ei:
        Cohort.from_csv(write(tmp_path, text), delta=1.0)
    assert [r for r, _ in ei.value.rows] == [3, 4, 5, 6, 7, 8, 9, 10]
    reasons = dict(ei.value.rows)
    assert "short row" in reasons[3] and "non-numeric" in reasons[4]
    assert "not finite" in reasons[5] and "not finite" in reasons[6]
    assert "censored at age 0" in reasons[7] and "negative" in reasons[8]
    assert "not 0 or 1" in reasons[9] and "not 0 or 1" in reasons[10]
    assert "row 7" in str(ei.value)


def test_blank_lines_are_ignored(tmp_path):
    c = Cohort.from_csv(write(tmp_path, "death times,events\n3,1\n\n5,0\n\n"), delta=1.0)
    assert list(c.ages) == [3.0, 5.0]


def test_no_usable_rows(tmp_path):
    with pytest.raises(CohortDataError, match="no usable rows"):
        Cohort.from_csv(write(tmp_path, "death times,events\n"), delta=1.0)


def test_order_and_name(tmp_path):
    p = write(tmp_path, "death times,events\n9,1\n2,1\n5,0\n", name="mice.csv")
    c = Cohort.from_csv(p, delta=1.0)
    assert list(c.ages) == [9.0, 2.0, 5.0] and c.name == "mice.csv"
    assert Cohort.from_csv(p, delta=1.0, name="M").name == "M"


def test_delta_is_required_or_continuous(tmp_path):
    from sraging.errors import CohortSpecError
    p = write(tmp_path, "death times,events\n3,1\n")
    with pytest.raises(CohortSpecError, match="changes how every death is scored"):
        Cohort.from_csv(p)
    assert Cohort.from_csv(p, continuous=True).continuous


def test_a_death_at_age_zero_is_read_with_delta_and_refused_without(tmp_path):
    p = write(tmp_path, "death times,events\n0,1\n2,1\n4,0\n")
    assert list(Cohort.from_csv(p, delta=2.0).ages) == [0.0, 2.0, 4.0]
    with pytest.raises(CohortDataError) as ei:
        Cohort.from_csv(p, continuous=True)
    assert [r for r, _ in ei.value.rows] == [2] and "continuous" in ei.value.rows[0][1]
