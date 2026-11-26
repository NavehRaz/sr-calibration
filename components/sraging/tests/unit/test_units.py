"""Explicit units (spec 005 US4; FR-007 to FR-009; research R6; data-model "Unit table"; tasks T024). Synthetic data
only, but for the factor table, checked against SRK's getTheta from the frozen snapshot (tools/oracle_presets_005.py):
SRK's rescaleTheta lives in SRK/utils.py, which is in no snapshot, and getTheta applies the same unit table."""
import itertools
import math
import os
import sys

import numpy as np
import pytest

from sraging.cohort import Cohort
from sraging.errors import UnsupportedOptionError
from sraging.params import SRParams
from sraging.units import UNITS, convert, factor, with_unit

P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1, xc=1.3, h_ext=2e-3, time_unit="days")
NAMES = ("hours", "days", "weeks", "years", "generations")


def _rel(a, b):
    return abs(a - b) / abs(b) if b else abs(a)


def test_the_unit_table_is_the_closed_list():
    assert dict(UNITS) == {"hours": 1 / 24, "days": 1.0, "weeks": 7.0, "years": 365.0, "generations": 1 / 8}
    assert factor("days", "years") == 365.0 and factor("weeks", "days") == 1 / 7
    with pytest.raises(TypeError):
        UNITS["months"] = 30.0


@pytest.mark.parametrize("a,b", list(itertools.permutations(NAMES, 2)))
def test_parameters_round_trip_within_1e_15(a, b):
    p = convert(P, to=a)
    back = convert(convert(p, to=b), to=a)
    for k in ("eta", "beta", "epsilon", "kappa", "xc", "h_ext"):
        assert _rel(getattr(back, k), getattr(p, k)) <= 1e-15, (k, a, b)
    assert back.time_unit == a


def test_eta_scales_as_s_squared_and_a_planted_s_scaling_fails():
    """η·s², β·s, ε·s, h·s; κ and x_c unchanged: the dimensionless groups at L are invariant when L is converted
    with the ages (L / s). The planted η·s moves ρ_η by the factor s."""
    s = factor("days", "weeks")
    w = convert(P, to="weeks")
    assert (w.eta, w.beta, w.epsilon, w.h_ext, w.kappa, w.xc) == (P.eta * 49.0, P.beta * 7.0, P.epsilon * 7.0,
                                                                  P.h_ext * 7.0, P.kappa, P.xc)
    L = 120.0
    want, got = P.dimensionless(L), w.dimensionless(L / s)
    assert all(_rel(got[k], want[k]) <= 1e-15 for k in want)
    planted = SRParams(eta=P.eta * s, beta=P.beta * s, epsilon=P.epsilon * s, kappa=P.kappa, xc=P.xc,
                       time_unit="weeks")
    assert _rel(planted.dimensionless(L / s)["rho_eta"], want["rho_eta"]) == pytest.approx(1 - 1 / s)


@pytest.mark.parametrize("bad", ["months", "Days", "year", "", "input units"])
def test_unlisted_units_are_refused_naming_them(bad):
    with pytest.raises(UnsupportedOptionError, match=repr(bad)) as ex:
        convert(P, to=bad)
    assert ex.value.option == "to"


def test_input_units_and_unlisted_sources_are_refused():
    with pytest.raises(UnsupportedOptionError, match="input units") as ex:
        convert(SRParams(eta=1e-3, beta=0.1, epsilon=1e-2, kappa=0.1), to="days")
    assert "with_unit" in str(ex.value)
    with pytest.raises(UnsupportedOptionError, match="'months'"):
        convert(SRParams(eta=1e-3, beta=0.1, epsilon=1e-2, kappa=0.1, time_unit="months"), to="days")
    with pytest.raises(TypeError, match="SRParams or Cohort"):
        convert([1, 2], to="days")


def test_convert_a_cohort_scales_ages_delta_entry_and_window_through_from_arrays():
    c = Cohort.from_arrays([7.0, 14.0, 21.0, 28.0], [1, 0, 1, 1], delta=7.0, entry=[0.0, 7.0, 0.0, 14.0],
                           name="c", time_unit="days")
    w = convert(c, to="weeks")
    assert isinstance(w, Cohort) and w.time_unit == "weeks" and w.name == "c" and w.delta == 1.0
    np.testing.assert_array_equal(w.ages, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(w.entry, [0.0, 1.0, 0.0, 2.0])
    np.testing.assert_array_equal(w.events, c.events)
    assert not w.ages.flags.writeable                                   # built by Cohort.from_arrays (read-only)
    win = Cohort.from_arrays([7.0, 14.0, 21.0], [1, 1, 1], delta=7.0, window=(7.0, math.inf), time_unit="days")
    assert convert(win, to="weeks").window == (1.0, math.inf)
    per_row = Cohort.from_arrays([1.0, 2.0], [1, 1], delta=[0.5, 1.0], time_unit="days")
    np.testing.assert_allclose(convert(per_row, to="hours").delta, [12.0, 24.0], rtol=1e-15)
    cont = Cohort.from_arrays([1.5, 2.5], [1, 1], continuous=True, time_unit="years")
    back = convert(convert(cont, to="days"), to="years")
    np.testing.assert_allclose(back.ages, cont.ages, rtol=1e-15)
    assert back.delta is None and back.continuous


def test_with_unit_labels_only_from_input_units():
    c = Cohort.from_arrays([1.0, 2.0], [1, 1], delta=1.0)
    d = with_unit(c, "days")
    assert d.time_unit == "days" and c.time_unit == "input units"
    np.testing.assert_array_equal(d.ages, c.ages)
    p = with_unit(SRParams(eta=1e-3, beta=0.1, epsilon=1e-2, kappa=0.1), "weeks")
    assert p.time_unit == "weeks" and p.eta == 1e-3
    with pytest.raises(UnsupportedOptionError, match="convert"):
        with_unit(d, "weeks")
    with pytest.raises(UnsupportedOptionError, match="'months'"):
        with_unit(c, "months")


def test_the_factors_equal_srk_getthetas_unit_table_side_by_side():
    """convert(get(name), to=u) against getTheta(name, time_unit=u) for every preset (mode_overall) and listed unit:
    within 1e-15 relative (exact in days); eta scaled by s, not s^2, fails (oracle group)."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(repo, "tools"))
    import acceptance_005 as ACC
    import oracle_presets_005 as OP
    if OP.preflight():
        pytest.skip(f"NOT APPLICABLE: {OP.preflight()}")
    r = ACC.presets_hr7(types=("mode_overall",), ext=(False,))
    assert r["units"] == list(NAMES) and r["cells"] == 35 * 5 and r["mismatches"] == []
    planted = ACC.presets_hr7(types=("mode_overall",), ext=(False,), units=("years",), planted="eta_s")
    assert len(planted["mismatches"]) == 35
