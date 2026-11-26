"""Cohort sets and their reader (spec 005 US2; FR-004 to FR-006; data-model "CohortSet", "Reader"; tasks T006, T007;
register D-505). HR-10: exact typed selection, with SRK's DatasetCollection.getKeys from the frozen snapshot beside it.
Synthetic data only."""
import importlib

import numpy as np
import pytest

from sraging.cohort import Cohort
from sraging.cohorts import CohortSet, read_cohorts
from sraging.errors import CohortDataError, CohortSpecError, UnsupportedOptionError


def _c(n=5, name=None, time_unit="days", shift=0.0):
    ages = np.arange(1.0, n + 1.0) + shift
    return Cohort.from_arrays(ages, np.ones(n, int), delta=1.0, name=name, time_unit=time_unit)


def _hr10_set():
    """Columns `age` and `cage` with integer values: A has age 1, B has cage 1 (the substring trap), C has age 1.0."""
    members = {"A": _c(), "B": _c(shift=1), "C": _c(shift=2), "D": _c(shift=3)}
    design = {"A": {"age": 1, "cage": 2}, "B": {"age": 2, "cage": 1}, "C": {"age": 1.0, "cage": 3},
              "D": {"age": "1", "cage": 4}}
    return CohortSet(members, design=design, time_unit="days")


@pytest.fixture
def srk_datasets(request):
    """SRK's deathTimesDataSet from the frozen snapshot tests/oracle/v033 (oracle group; skipped without it)."""
    pytest.importorskip("lifelines")
    request.getfixturevalue("oracle_v033")
    return importlib.import_module("SRK.deathTimesDataSet")


# ------------------------------------------------------------------------------------------ HR-10 (T006)
def test_hr10_select_is_exact_and_typed():
    """HR-10 (D-505): `age` = 1 returns no `cage` row, and the integer 1 matches 1 and 1.0 but not the string '1'."""
    s = _hr10_set()
    got = s.select(age=1)
    assert list(got) == ["A", "C"] and len(got) == 2 and got.time_unit == "days"
    assert "B" not in got                                        # cage = 1 is not age = 1
    assert list(s.select(age="1")) == ["D"]                      # a string matches only the string
    assert list(s.select(age=1.0)) == ["A", "C"] and list(s.select(age=np.int64(1))) == ["A", "C"]
    assert list(s.select(age=1, cage=3)) == ["C"]


def test_hr10_srk_getkeys_shows_both_legacy_defects_beside_the_twin(srk_datasets):
    """The same selection through SRK's DatasetCollection.getKeys (read only): 'age' matches inside 'cage:_1', and the
    integer 1 matches nothing; sraging's select on the same table returns the exact rows."""
    keys = {"age:_1,cage:_2": None, "age:_2,cage:_1": None}
    coll = srk_datasets.DatasetCollection(datasets=dict(keys), warnings=False)
    assert coll.getKeys(properties=["age"], values=["1"]) == ["age:_1,cage:_2", "age:_2,cage:_1"]   # substring
    assert coll.getKeys(properties=["age"], values=[1]) == []                                      # strings only
    s = CohortSet({"A": _c(), "B": _c(shift=1)}, design={"A": {"age": 1, "cage": 2}, "B": {"age": 2, "cage": 1}},
                  time_unit="days")
    assert list(s.select(age=1)) == ["A"]                                                          # the twin


def test_an_unknown_selection_column_is_refused_naming_it():
    s = _hr10_set()
    with pytest.raises(UnsupportedOptionError, match="'ag'") as ex:
        s.select(ag=1)
    assert ex.value.option == "ag" and "age" in str(ex.value)
    with pytest.raises(UnsupportedOptionError, match="'sex'"):
        s.by("sex")


def test_a_selection_by_a_collection_is_refused():
    with pytest.raises(UnsupportedOptionError, match="'age'"):
        _hr10_set().select(age=[1, 2])


def test_an_empty_match_is_an_empty_set_with_its_count():
    s = _hr10_set().select(age=7)
    assert len(s) == 0 and s.count == 0 and list(s) == []
    assert "0 cohorts" in repr(s) and s.selection == {"age": 7}
    assert s.columns == ("age", "cage")
    with pytest.raises(UnsupportedOptionError, match="'sex'"):                # still refused on the empty set
        s.select(sex="F")


def test_duplicate_keys_are_refused_naming_them():
    members = [("A", _c(), {"cohort": "x", "cage": 1}), ("B", _c(), {"cohort": "x", "cage": 2}),
               ("A", _c(), {"cohort": "x", "cage": 1})]
    with pytest.raises(CohortSpecError, match="duplicate") as ex:
        CohortSet.from_members(members, time_unit="days")
    assert "'A'" in str(ex.value) and "'B'" not in str(ex.value)
    assert len(CohortSet.from_members(members[:2], time_unit="days")) == 2                       # the twin


def test_members_with_different_time_units_are_refused():
    with pytest.raises(CohortSpecError, match="time_unit") as ex:
        CohortSet({"A": _c(), "B": _c(time_unit="weeks")}, design={"A": {"g": 1}, "B": {"g": 1}}, time_unit="days")
    assert "'B'" in str(ex.value) and "weeks" in str(ex.value) and "convert" in str(ex.value)


def test_a_design_row_without_a_member_and_a_member_without_a_row_are_refused():
    with pytest.raises(CohortSpecError, match="'Z'"):
        CohortSet({"A": _c()}, design={"A": {"g": 1}, "Z": {"g": 2}}, time_unit="days")
    with pytest.raises(CohortSpecError, match="'B'"):
        CohortSet({"A": _c(), "B": _c()}, design={"A": {"g": 1}}, time_unit="days")
    with pytest.raises(CohortSpecError, match="columns"):
        CohortSet({"A": _c(), "B": _c()}, design={"A": {"g": 1}, "B": {"h": 1}}, time_unit="days")


def test_by_groups_cages_by_cohort_in_order():
    members = {f"{co}/{cg}": _c(shift=i) for i, (co, cg) in enumerate([("x", 1), ("y", 1), ("x", 2)])}
    design = {k: {"cohort": k.split("/")[0], "cage": int(k.split("/")[1])} for k in members}
    groups = CohortSet(members, design=design, time_unit="days").by("cohort")
    assert list(groups) == ["x", "y"]
    assert list(groups["x"]) == ["x/1", "x/2"] and list(groups["y"]) == ["y/1"]
    assert groups["x"].selection == {"cohort": "x"} and groups["x"].design["x/2"]["cage"] == 2


def test_the_set_is_immutable():
    s = _hr10_set()
    with pytest.raises(TypeError):
        s.cohorts["E"] = _c()
    with pytest.raises(TypeError):
        s.design["A"]["age"] = 5


# ------------------------------------------------------------------------------------------ the reader (T007)
def _write(path, text):
    path.write_text(text)
    return path


@pytest.mark.parametrize("time_h,event_h", [("death times", "events"), ("death_times", "event"),
                                            ("Death Time", "Events"), ("age", "event"), (" age ", " events ")])
def test_every_header_variant_is_recognised(tmp_path, time_h, event_h):
    f = _write(tmp_path / "c1.csv", f"{time_h},{event_h}\n1,1\n2,0\n3,1\n")
    s, refused = read_cohorts(f, delta=1.0, time_unit="days")
    assert list(s) == ["c1"] and refused == ()
    np.testing.assert_array_equal(s["c1"].ages, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(s["c1"].events, [1, 0, 1])


def test_a_header_with_no_time_or_no_event_column_is_refused_naming_it(tmp_path):
    with pytest.raises(CohortDataError, match="lifespan") as ex:
        read_cohorts(_write(tmp_path / "a.csv", "lifespan,events\n1,1\n"), delta=1.0, time_unit="days")
    assert "'death times'" in str(ex.value)
    with pytest.raises(CohortDataError, match="dead"):
        read_cohorts(_write(tmp_path / "b.csv", "age,dead\n1,1\n"), delta=1.0, time_unit="days")
    with pytest.raises(CohortDataError, match="censor"):                       # censor is never guessed
        read_cohorts(_write(tmp_path / "c.csv", "age,censored\n1,0\n"), delta=1.0, time_unit="days")
    with pytest.raises(CohortDataError, match="two"):
        read_cohorts(_write(tmp_path / "d.csv", "age,death times,event\n1,1,1\n"), delta=1.0, time_unit="days")


def test_a_long_csv_with_a_cohort_and_a_cage_column_is_one_cohort_per_cohort_and_cage(tmp_path):
    f = _write(tmp_path / "long.csv", "cohort,age,event,cage_id,sex\n"
                                      "ctrl,1,1,1,F\nctrl,2,1,2,F\nrapa,3,1,1,F\nctrl,4,0,1,F\nrapa,5,1,1,F\n")
    s, refused = read_cohorts(f, delta=1.0, time_unit="days", cohort_col="cohort", cage_col="cage_id")
    assert refused == () and list(s) == ["ctrl/1", "ctrl/2", "rapa/1"]
    assert s.design["ctrl/1"] == {"cohort": "ctrl", "cage": 1, "sex": "F"}
    np.testing.assert_array_equal(s["ctrl/1"].ages, [1.0, 4.0])
    np.testing.assert_array_equal(s["ctrl/1"].events, [1, 0])
    assert s["rapa/1"].name == "rapa/1" and s["rapa/1"].time_unit == "days" and s.time_unit == "days"
    assert list(s.by("cohort")) == ["ctrl", "rapa"] and list(s.select(cage=1)) == ["ctrl/1", "rapa/1"]


def test_a_column_that_varies_within_a_member_is_not_in_the_design_and_is_noted(tmp_path):
    f = _write(tmp_path / "x.csv", "age,event,cage_id,age_raw\n1,1,a,3\n2,1,a,4\n2,1,b,4\n")
    s, _ = read_cohorts(f, delta=1.0, time_unit="days", cage_col="cage_id")
    assert s.design["x/a"] == {"cohort": "x", "cage": "a"}
    assert any("age_raw" in n for n in s.notes)


def test_a_directory_without_a_cohort_column_names_each_cohort_by_its_file_stem(tmp_path):
    _write(tmp_path / "wt.csv", "age,event,cage_id,diet\n1,1,1,AL\n2,1,2,AL\n")
    _write(tmp_path / "mut.csv", "age,event,cage_id\n3,1,1\n")
    _write(tmp_path / "notes.txt", "not a cohort")
    s, refused = read_cohorts(tmp_path, delta=1.0, time_unit="days", cage_col="cage_id")
    assert list(s) == ["mut/1", "wt/1", "wt/2"] and refused == ()
    assert s.design["wt/2"] == {"cohort": "wt", "cage": 2}
    assert any("diet" in n and "every file" in n for n in s.notes)               # stated, not silently dropped


def test_entry_ages_cohort_refuses_are_listed_by_row(tmp_path):
    f = _write(tmp_path / "en.csv", "age,event,entry\n3,1,1\n5,0,6\n6,1,0\n")
    s, refused = read_cohorts(f, delta=1.0, time_unit="days", entry_col="entry")
    assert [(r.row, "after the recorded age" in r.reason) for r in refused] == [(3, True)]
    np.testing.assert_array_equal(s["en"].ages, [3.0, 6.0])
    np.testing.assert_array_equal(s["en"].entry, [1.0, 0.0])


def test_a_cohort_and_cage_in_two_files_is_refused_naming_it(tmp_path):
    _write(tmp_path / "a.csv", "cohort,age,event,cage\nx,1,1,1\n")
    _write(tmp_path / "b.csv", "cohort,age,event,cage\nx,2,1,1\ny,2,1,1\n")
    with pytest.raises(CohortDataError, match="duplicate") as ex:
        read_cohorts(tmp_path, delta=1.0, time_unit="days", cohort_col="cohort", cage_col="cage")
    assert "x/1" in str(ex.value) and "y/1" not in str(ex.value)


def test_optional_delta_and_entry_columns_are_used(tmp_path):
    f = _write(tmp_path / "e.csv", "age,event,entry,dt\n3,1,1,1\n5,0,2,1\n6,1,0,2\n")
    s, refused = read_cohorts(f, time_unit="days", entry_col="entry", delta_col="dt")
    c = s["e"]
    assert refused == ()
    np.testing.assert_array_equal(c.entry, [1.0, 2.0, 0.0])
    np.testing.assert_array_equal(c.delta, [1.0, 1.0, 2.0]) and c.has_features
    with pytest.raises(CohortSpecError, match="delta"):
        read_cohorts(f, delta=1.0, time_unit="days", delta_col="dt")
    with pytest.raises(CohortSpecError, match="delta"):
        read_cohorts(f, time_unit="days")                                         # neither delta nor continuous
    with pytest.raises(CohortDataError, match="'start'"):
        read_cohorts(f, delta=1.0, time_unit="days", entry_col="start")


def test_a_per_row_delta_is_refused_by_a_calibrator_that_does_not_accept_it(tmp_path):
    """FR-005: the reader honours a per-row delta column; spec 001's calibrator refuses it, naming the feature, before
    any fit starts."""
    from sraging import fit_cohort
    f = _write(tmp_path / "d.csv", "age,event,dt\n" + "".join(f"{a},1,1\n" for a in range(1, 30)))
    s, _ = read_cohorts(f, time_unit="days", delta_col="dt")
    with pytest.raises(UnsupportedOptionError, match="delta"):
        fit_cohort(s["d"], calibrator="profile_v033")


def test_every_refused_row_is_listed_with_file_row_and_reason(tmp_path):
    f = _write(tmp_path / "r.csv", "age,event,cage_id\n"
                                   "1,1,a\n"          # line 2: kept
                                   "x,1,a\n"          # line 3: non-numeric age
                                   "2,2,a\n"          # line 4: event 2
                                   "3,1\n"            # line 5: short
                                   "4,1,\n"           # line 6: no cage
                                   "0,0,a\n"          # line 7: censored at age 0
                                   "inf,1,a\n"        # line 8: not finite
                                   "\n"               # blank: ignored
                                   "5,1,a\n")         # line 10: kept
    s, refused = read_cohorts(f, delta=1.0, time_unit="days", cage_col="cage_id")
    np.testing.assert_array_equal(s["r/a"].ages, [1.0, 5.0])
    assert [(r.file, r.row) for r in refused] == [(str(f), n) for n in (3, 4, 5, 6, 7, 8)]
    why = {r.row: r.reason for r in refused}
    assert "non-numeric" in why[3] and "event" in why[4] and "cells" in why[5] and "cage" in why[6]
    assert "censored at age 0" in why[7] and "not finite" in why[8]


def test_a_member_cohort_from_arrays_refuses_is_listed_with_its_rows(tmp_path):
    """Each member goes through Cohort.from_arrays: a cage with no deaths is refused there, and its rows are listed."""
    f = _write(tmp_path / "m.csv", "age,event,cage\n1,1,a\n2,0,b\n3,0,b\n")
    s, refused = read_cohorts(f, delta=1.0, time_unit="days", cage_col="cage")
    assert list(s) == ["m/a"]
    assert [r.row for r in refused] == [3, 4] and all("no deaths" in r.reason for r in refused)


def test_continuous_ages_refuse_age_zero_as_cohort_does(tmp_path):
    f = _write(tmp_path / "z.csv", "age,event\n0,1\n1.5,1\n")
    s, refused = read_cohorts(f, continuous=True, time_unit="days")
    assert [r.row for r in refused] == [2] and "age 0 with continuous" in refused[0].reason
    np.testing.assert_array_equal(s["z"].ages, [1.5])


def test_a_missing_path_or_column_is_refused_naming_it(tmp_path):
    with pytest.raises(CohortDataError, match="nowhere"):
        read_cohorts(tmp_path / "nowhere", delta=1.0, time_unit="days")
    f = _write(tmp_path / "k.csv", "age,event\n1,1\n")
    with pytest.raises(CohortDataError, match="'cage_id'"):
        read_cohorts(f, delta=1.0, time_unit="days", cage_col="cage_id")
    with pytest.raises(TypeError, match="time_unit"):
        read_cohorts(f, delta=1.0)
