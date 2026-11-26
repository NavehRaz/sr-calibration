"""Spec 006's window path for W and H (FR-006 to FR-008; research R3, R4; data-model.md "The window path"; tasks T007).

The tester's window rule (`compat/tester_f94ec2b.apply_window`) against the snapshot's own function; then the path's
steps in order: hi = inf refused without delta and mapped with it, the rows dropped and censored counted, too few
deaths refused, the windowed Cohort and its row map, the frozen snap's record. The last tests reproduce why the steps
are needed: today's frozen snap overflows on hi = inf and asserts on rows below lo. Fit-free: seconds."""
import ast
import math
import os

import numpy as np
import pytest

from sraging import Cohort
from sraging.calibrators import profile_jax2 as PJ
from sraging.compat.tester_f94ec2b.observation import apply_window
from sraging.errors import CohortSpecError, UnsupportedOptionError

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SNAPSHOT = os.path.join(REPO, "tests", "oracle", "tester_f94ec2b", "srbench", "library", "observation.py")
P101 = PJ.PORTS["1.0.1"]
H = P101.classes["profile_jax2_h"]


class W(P101.classes["profile_jax2"]):
    """W's class as data-model.md defines it (J's constants, the window on), until T016 registers it."""
    WINDOW = True


def snapshot_apply_window():
    """The tester snapshot's own `apply_window`, compiled from its file (no srbench import)."""
    tree = ast.parse(open(SNAPSHOT).read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_window")
    ns = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), SNAPSHOT, "exec"), ns)
    return ns["apply_window"]


def cohort(ages, events, *, window, delta=1.0, continuous=False):
    return Cohort.from_arrays(ages, events, delta=None if continuous else delta, continuous=continuous,
                              window=window, name="w", time_unit="days")


AGES = np.array([3.0, 5.0, 5.0, 8.0, 10.0, 12.0, 12.0, 15.0, 20.0, 20.0, 21.0, 30.0])
EVENTS = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0])


# ------------------------------------------------------------------------------------------ the tester's rule
@pytest.mark.parametrize("window", [None, (5.0, 20.0), (0.0, 12.0), (12.0, 12.5), (4.0, 1e9)])
def test_apply_window_gives_the_snapshots_arrays(window):
    ref = snapshot_apply_window()
    a, e = apply_window(AGES, EVENTS, window)
    ra, re_ = ref(AGES, EVENTS, window)
    assert a.dtype == ra.dtype and e.dtype == re_.dtype
    assert np.array_equal(a, ra) and np.array_equal(e, re_)


def test_apply_window_drops_below_lo_censors_above_hi_and_keeps_an_age_at_hi():
    a, e = apply_window(AGES, EVENTS, (5.0, 20.0))
    assert a.min() == 5.0 and AGES[AGES < 5.0].size == 1 and a.size == AGES.size - 1     # 3.0 dropped, not censored
    assert list(a[-2:]) == [20.0, 20.0] and list(e[-2:]) == [0, 0]                        # 21 and 30 censored at 20
    at_hi = (a == 20.0) & (np.arange(a.size) < a.size - 2)
    assert list(e[at_hi]) == [1, 1]                         # ages == hi keep their event (the snap's edge tie later)


def test_apply_window_is_idempotent():
    once = apply_window(AGES, EVENTS, (5.0, 20.0))
    twice = apply_window(*once, (5.0, 20.0))
    assert all(np.array_equal(x, y) for x, y in zip(once, twice))


# ---------------------------------------------------------------------------------------------- the window path
def test_hi_inf_without_delta_is_refused_by_name():
    c = cohort(AGES + 0.25, EVENTS, window=(5.0, math.inf), continuous=True)
    with pytest.raises(UnsupportedOptionError) as ex:
        PJ.window_path(c, W, port=P101)
    assert ex.value.option == "window" and "delta" in str(ex.value) and "inf" in str(ex.value)


@pytest.mark.parametrize("ages_shift", [0.0, 0.5])
def test_hi_inf_is_mapped_to_an_on_grid_end_past_every_record(ages_shift):
    c = cohort(AGES + ages_shift, EVENTS, window=(5.0, math.inf))
    wc, rec, _rows = PJ.window_path(c, W, port=P101)
    hi = rec["hi_used"]
    assert rec["hi_mapped_from"] == "inf" and rec["window_given"] == [5.0, "inf"]
    assert hi > c.ages.max() and hi == 31.0 and abs(hi / 1.0 - round(hi)) == 0.0
    assert wc.window == (5.0, hi) and rec["n_censored_above_hi"] == 0


def test_the_rows_dropped_below_lo_and_censored_above_hi_are_counted():
    c = cohort(AGES, EVENTS, window=(5.0, 20.0))
    wc, rec, rows = PJ.window_path(c, W, port=P101)
    assert rec["n_rows_given"] == 12 and rec["n_dropped_below_lo"] == 1 and rec["n_censored_above_hi"] == 2
    assert rec["n_rows_kept"] == wc.n == 11 and rec["hi_mapped_from"] is None and rec["hi_used"] == 20.0
    a, e = apply_window(AGES, EVENTS, (5.0, 20.0))
    assert np.array_equal(wc.ages, a) and np.array_equal(wc.events, e)
    assert list(rows) == list(np.flatnonzero(AGES >= 5.0)) and len(rows) == wc.n
    assert wc.delta == c.delta and wc.time_unit == c.time_unit and wc.window == (5.0, 20.0)


def test_the_frozen_snaps_record_is_kept():
    """The edge tie: the two deaths at 20 (the bin of hi_eff) are censored by the frozen snap; with the two censored
    above hi, four records sit at hi_eff."""
    c = cohort(AGES, EVENTS, window=(5.0, 20.0))
    _wc, rec, _rows = PJ.window_path(c, W, port=P101)
    assert rec["window_eff"] == [5.0, 20.0] and rec["n_deaths_censored"] == 2 and rec["n_censored_at_hi_eff"] == 4
    assert rec["n_deaths_fitted"] == 5 and rec["lo_rule"] == "on the grid"       # 5, 8, 10, 12, 15; 20 censored


def test_lo_at_zero_is_no_left_truncation():
    c = cohort(AGES, EVENTS, window=(0.0, 20.0))
    wc, rec, _rows = PJ.window_path(c, W, port=P101)
    assert rec["n_dropped_below_lo"] == 0 and wc.n == 12 and rec["window_eff"][0] == 0.0
    assert rec["lo_rule"].startswith("lo <= delta/2")


@pytest.mark.parametrize("cls,needed", [(W, 4), (H, 5)])
def test_too_few_deaths_after_the_window_are_refused_naming_the_count(cls, needed):
    """Fewer deaths left to fit than the calibrator's free parameters (4 for W, 5 for H) is refused; the count is the
    deaths the frozen snap leaves (the edge tie censored)."""
    deaths = needed - 1
    ages = np.concatenate([10.0 + np.arange(deaths), [40.0, 41.0]])
    events = np.concatenate([np.ones(deaths, int), [0, 0]])
    c = cohort(ages, events, window=(10.0, 40.0))
    with pytest.raises(UnsupportedOptionError) as ex:
        PJ.window_path(c, cls, port=P101)
    assert ex.value.option == "window" and f"{deaths} death" in str(ex.value) and str(needed) in str(ex.value)
    ok = cohort(np.concatenate([ages, [20.5]]), np.concatenate([events, [1]]), window=(10.0, 40.0))   # the twin
    assert PJ.window_path(ok, cls, port=P101)[1]["n_deaths_fitted"] == needed


def test_no_window_passes_the_cohort_through_unchanged():
    c = cohort(AGES, EVENTS, window=None)
    wc, rec, rows = PJ.window_path(c, W, port=P101)
    assert wc is c and rec is None and rows is None


def test_an_off_grid_window_given_by_hand_stays_refused_by_cohort():
    with pytest.raises(CohortSpecError):
        cohort(AGES, EVENTS, window=(5.5, 20.0))


def test_observation_passes_the_windowed_cohorts_window():
    c = cohort(AGES, EVENTS, window=(5.0, 20.0))
    wc, _rec, _rows = PJ.window_path(c, W, port=P101)
    obs = PJ.observation(wc)
    assert obs.window == (5.0, 20.0) and len(obs.ages) == wc.n
    assert PJ.observation(cohort(AGES, EVENTS, window=None)).window is None


# ------------------------------------------------------------------------------- why the steps are needed (R4)
def test_todays_frozen_snap_overflows_on_hi_inf_and_asserts_on_rows_below_lo():
    om = P101.observation_model
    with pytest.raises(OverflowError):
        om.prepare(list(AGES), list(EVENTS), 1.0, (5.0, math.inf), window_on=True)
    with pytest.raises(AssertionError, match="below lo_eff"):
        om.prepare(list(AGES), list(EVENTS), 1.0, (5.0, 20.0), window_on=True)
    a, e = apply_window(AGES, EVENTS, (5.0, 20.0))                                         # the twin
    assert om.prepare(list(a), list(e), 1.0, (5.0, 20.0), window_on=True).record["window_snap"]["applied"]
