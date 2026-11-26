"""L0: the observation model (grid check, generic tie relabel, invariants) on synthetic edge cases, against
rc1's relabel, and over every library cohort."""
from __future__ import annotations

import os

import pytest

import _jax2_testlib as T


@pytest.fixture(scope="module")
def om():
    return T.pkg_module("observation_model")


@pytest.fixture(scope="module")
def rc1():
    return T.rc1_module("adapter")          # unregistered: safe in-process


# ---------------------------------------------------------------- the relabel, case by case

CASES = {
    # name: (ages, events, delta, expected n_relabelled, expected reason)
    "no_delta": ([1.0, 2.0, 3.0], [1, 1, 0], None, 0, "no delta"),
    "zero_delta": ([1.0, 2.0, 3.0], [1, 1, 0], 0.0, 0, "no delta"),
    "no_censored": ([1.0, 2.0, 3.0], [1, 1, 1], 1.0, 0, "no censored records"),
    "censored_two_ages": ([1.0, 2.0, 3.0, 3.0], [1, 0, 1, 0], 1.0, 0, "censored records at more than one age"),
    "record_beyond_tc": ([1.0, 2.0, 3.0], [1, 0, 1], 1.0, 0, "a record lies beyond the censoring age"),
    "on_grid_tie": ([1.0, 2.0, 3.0, 3.0, 3.0], [1, 1, 1, 1, 0], 1.0, 2, "relabelled"),
    "no_death_in_bin": ([1.0, 2.0, 3.0], [1, 1, 0], 1.0, 0, "no death in the censoring bin"),
    # tc - b = delta (one step below an on-grid tc, up to round-off): NOT the bin
    "one_step_below": ([0.1, 0.2, 0.3, 0.3], [1, 1, 0, 0], 0.1, 0, "no death in the censoring bin"),
    # off-grid tc (a window closing between grid points): deaths at b = floor_grid(tc) are in the bin
    "off_grid_tc": ([1.0, 2.0, 2.0, 2.5], [1, 1, 1, 0], 1.0, 2, "relabelled"),
}


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("relabel_on", [True, False])
def test_relabel_cases_and_rc1_equality(om, rc1, name, relabel_on):
    ages, events, delta, n_rel, reason = CASES[name]
    p = om.prepare(tuple(ages), tuple(events), delta, None, window_on=False, relabel_on=relabel_on)
    a, e, rec = rc1.censor_tie_relabel(ages, events, delta, enabled=relabel_on)
    assert (p.ages, p.events, p.censor_tie) == (a, e, rec)           # rc1 is the oracle
    assert p.ages == [float(x) for x in ages]                         # ages are never moved
    if relabel_on:
        assert p.censor_tie["n_relabelled"] == n_rel and p.censor_tie["reason"] == reason
        assert p.record["n_flipped"] == n_rel
    else:
        assert p.events == list(events) and p.record["n_flipped"] == 0
        assert p.censor_tie["n_tied"] == n_rel                        # the detector still says what WOULD be
    assert p.record["invariants"] == "ok" and p.record["relabel_on"] is relabel_on
    assert p.window_eff is None and p.record["window_eff"] is None


def test_on_grid_tie_relabels_only_the_tied_bin(om):
    p = om.prepare([1.0, 2.0, 3.0, 3.0, 3.0], [1, 1, 1, 1, 0], 1.0, relabel_on=True)
    assert p.events == [1, 1, 0, 0, 0]
    assert p.censor_tie["tc"] == 3.0 and p.censor_tie["on_grid"] is True
    assert p.censor_tie["dose"] == pytest.approx(2 / 5)


# ---------------------------------------------------------------- the window snap (B2)

SNAP = {
    # name: (ages, events, delta, window, expected (lo_eff, hi_eff), expected events out, expected ages out)
    # on-grid hi: the death recorded AT hi (T in [hi, hi+delta)) is censored there, with the censored
    "on_grid": ([2.0, 3.0, 4.0, 5.0, 5.0, 5.0], [1, 1, 1, 1, 0, 0], 1.0, (2.0, 5.0), (2.0, 5.0),
                [1, 1, 1, 0, 0, 0], [2.0, 3.0, 4.0, 5.0, 5.0, 5.0]),
    # off-grid hi = 5.5: the harness censored at 5.5 (T >= 6) and kept the death at 5 (T in [5, 6)):
    # hi_eff = 5, everything >= 5 censored at 5 (the 5.5 ages MOVE DOWN to 5)
    "off_grid_hi": ([2.0, 3.0, 5.0, 5.5, 5.5], [1, 1, 1, 0, 0], 1.0, (2.0, 5.5), (2.0, 5.0),
                    [1, 1, 0, 0, 0], [2.0, 3.0, 5.0, 5.0, 5.0]),
    # off-grid lo = 2.5: `ages >= 2.5` kept exactly T >= 3 -> lo_eff = ceil_grid(lo) = 3
    "off_grid_lo": ([3.0, 4.0, 5.0, 6.0, 6.0], [1, 1, 1, 0, 0], 1.0, (2.5, 6.0), (3.0, 6.0),
                    [1, 1, 1, 0, 0], [3.0, 4.0, 5.0, 6.0, 6.0]),
    # lo = delta/2 (the Guinea-pig window cohorts): the delta/2 clamp kept every record -> lo_eff = 0
    "lo_half_delta": ([0.5, 1.0, 2.0, 3.0, 3.0], [1, 1, 1, 0, 0], 1.0, (0.5, 3.0), (0.0, 3.0),
                      [1, 1, 1, 0, 0], [0.5, 1.0, 2.0, 3.0, 3.0]),
    # a decimal grid: 0.1-steps computed as k * 0.1 must be recognised on the grid; the censored record one
    # ulp ABOVE the on-grid hi = 0.7 moves DOWN to hi_eff = 0.7 (never up)
    "decimal_grid": ([0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001],
                     [1, 1, 1, 1, 0], 0.1, (0.3, 0.7), (0.3, 0.7), [1, 1, 1, 1, 0],
                     [0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7]),
}


@pytest.mark.parametrize("name", sorted(SNAP))
def test_window_snap_cases(om, name):
    ages, events, delta, window, weff, ev_out, a_out = SNAP[name]
    p = om.prepare(ages, events, delta, window, window_on=True, relabel_on=True)
    assert p.window_eff == pytest.approx(weff, rel=0, abs=1e-12)
    assert p.events == ev_out and p.ages == pytest.approx(a_out, rel=0, abs=0)
    rec = p.record["window_snap"]
    assert rec["applied"] is True and rec["reason"] == "snapped"
    assert p.record["window_eff"] == pytest.approx(list(weff), abs=1e-12) and p.record["invariants"] == "ok"
    lo_eff, hi_eff = p.window_eff
    assert all(lo_eff - 1e-12 <= x <= hi_eff + 1e-12 for x in p.ages)
    # after the snap no death lies in hi_eff's bin, so the generic relabel has nothing left to do there
    assert not any(e == 1 and x >= hi_eff - 1e-9 for x, e in zip(p.ages, p.events))


def test_window_snap_without_delta_or_off_grid(om):
    p = om.prepare([2.3, 3.7, 5.0], [1, 1, 0], None, (2.0, 5.0), window_on=True)
    assert p.window_eff == (2.0, 5.0) and p.record["window_snap"]["applied"] is False
    assert "no delta" in p.record["window_snap"]["reason"]
    p = om.prepare([2.0, 3.5, 5.0], [1, 1, 0], 1.0, (2.0, 5.0), window_on=True)   # death off the grid
    assert p.window_eff == (2.0, 5.0) and p.record["window_snap"]["applied"] is False
    assert "off-grid" in p.record["window_snap"]["reason"]


def test_window_on_without_a_window_is_the_relabel_only(om):
    p = om.prepare([1.0, 2.0, 3.0, 3.0], [1, 1, 1, 0], 1.0, None, window_on=True, relabel_on=True)
    q = om.prepare([1.0, 2.0, 3.0, 3.0], [1, 1, 1, 0], 1.0, None, window_on=False, relabel_on=True)
    assert p.window_eff is None and p.record["window_snap"] is None
    assert (p.ages, p.events, p.censor_tie) == (q.ages, q.events, q.censor_tie)


def test_window_ignored_when_off(om):
    p = om.prepare([1.0, 2.0, 2.0], [1, 1, 0], 1.0, (0.5, 2.0), window_on=False, relabel_on=True)
    assert p.record["window"] == [0.5, 2.0] and p.record["window_eff"] is None and p.window_eff is None


# ---------------------------------------------------------------- the grid check

def test_grid_check(om):
    g = om.grid_check([1.0, 2.0, 0.5, 2.5, 3.0], [1, 1, 1, 1, 0], 1.0)
    assert (g["n_deaths_on_grid"], g["n_half_clamp"], g["n_deaths_off_grid"]) == (2, 1, 1)
    assert g["grid_unverified"] is True and g["first_off_grid"] == 2.5
    g = om.grid_check([0.30000000000000004, 0.6, 1.2000000000000002], [1, 1, 0], 0.1)
    assert g["grid_unverified"] is False and g["n_deaths_on_grid"] == 2
    g = om.grid_check([1.0, 2.0, 2.5], [1, 1, 0], 1.0)                # an off-grid CENSORED age is legitimate
    assert g["grid_unverified"] is False and g["n_censored_off_grid"] == 1
    g = om.grid_check([1.3, 2.7], [1, 1], None)
    assert g["grid_unverified"] is None and g["reason"] == "no delta"


# ---------------------------------------------------------------- invariants, with positive controls

def test_invariants_positive_controls(om):
    ok = om.check_invariants([1.0, 2.0], [1, 1], [1.0, 2.0], [1, 0])
    assert ok == 1
    with pytest.raises(AssertionError, match="changed n"):
        om.check_invariants([1.0, 2.0], [1, 1], [1.0], [1])
    with pytest.raises(AssertionError, match="only 1 -> 0"):
        om.check_invariants([1.0, 2.0], [1, 0], [1.0, 2.0], [1, 1])
    with pytest.raises(AssertionError, match="only 1 -> 0"):
        om.check_invariants([1.0], [1], [1.0], [2])
    with pytest.raises(AssertionError, match="moved"):
        om.check_invariants([1.0, 2.0], [1, 1], [1.0, 2.5], [1, 1])


# ---------------------------------------------------------------- every library cohort

def test_prepare_on_every_library_cohort_matches_rc1(om, rc1):
    """All 2,290 cohorts of library v2: invariants hold, and prepare() == rc1's relabel exactly (ages,
    events and record), relabel on and off. Also a census of what the relabel and the grid check see."""
    from srbench.library.load import Library
    lib = Library(root=os.path.join(T.ROOT, "data", "library", "v2"), version="v2") \
        if os.path.isdir(os.path.join(T.ROOT, "data", "library", "v2", "cohorts")) else None
    if lib is None:
        pytest.skip("library v2 cohorts are not built in this checkout")
    ids = [s["cohort_id"] for s in lib.manifest["specs"]]
    assert len(ids) == 2290
    census = dict(applied=0, detected=0, flipped=0, unverified=0, windowed=0, windowed_applied=0)
    for cid in ids:
        obs = lib.observation(cid)
        lib._cache.clear()                    # the Library caches every array (~360 MB for all cohorts)
        for on in (True, False):
            p = om.prepare(obs.ages, obs.events, obs.delta, obs.window, window_on=False, relabel_on=on)
            a, e, rec = rc1.censor_tie_relabel(obs.ages, obs.events, obs.delta, enabled=on)
            assert p.ages == a and p.events == e and p.censor_tie == rec, cid
            assert p.record["invariants"] == "ok" and p.record["n"] == len(obs.ages)
            if on:
                census["applied"] += bool(rec["applied"])
                census["flipped"] += p.record["n_flipped"]
                census["unverified"] += bool(p.record["grid"]["grid_unverified"])
                census["windowed"] += obs.window is not None
                census["windowed_applied"] += bool(rec["applied"]) and obs.window is not None
            else:
                census["detected"] += bool(rec["detected"])
    assert census["applied"] == census["detected"] > 0
    print("\nCENSUS", census)


def _on(x, d):
    q = x / d
    return abs(q - round(q)) <= 1e-9 * max(1.0, abs(q))


def test_prepare_window_on_every_library_cohort(om):
    """B2: prepare(window_on=True), as mle_profile_jax2_h runs it, on all 2,290 v2 cohorts. Invariants hold;
    on every windowed cohort with a delta the snap applies and its output has the snap's properties, checked
    here independently of the implementation:
      lo_eff = 0 if lo <= delta/2, else on the grid with lo <= lo_eff < lo + delta;
      hi_eff on the grid with hi - delta < hi_eff <= hi;  every age in [lo_eff, hi_eff];
      a record is censored at (or ulp-below) hi_eff iff its input age was in hi_eff's bin or later, and every
      other record is unchanged; no death is left in hi_eff's bin.
    Unwindowed cohorts: identical to window_on=False."""
    from srbench.library.load import Library
    root = os.path.join(T.ROOT, "data", "library", "v2")
    if not os.path.isdir(os.path.join(root, "cohorts")):
        pytest.skip("library v2 cohorts are not built in this checkout")
    lib = Library(root=root, version="v2")
    ids = [s["cohort_id"] for s in lib.manifest["specs"]]
    assert len(ids) == 2290
    c = dict(windowed=0, snapped=0, lo_zero=0, lo_moved=0, hi_moved=0, deaths_censored=0, ages_moved=0,
             cohorts_with_edge_deaths=0, relabel_after_snap=0, unwindowed_identical=0)
    for cid in ids:
        obs = lib.observation(cid)
        lib._cache.clear()
        p = om.prepare(obs.ages, obs.events, obs.delta, obs.window, window_on=True, relabel_on=True)
        assert p.record["invariants"] == "ok" and p.record["n"] == len(obs.ages), cid
        if obs.window is None:
            q = om.prepare(obs.ages, obs.events, obs.delta, None, window_on=False, relabel_on=True)
            assert (p.ages, p.events, p.censor_tie, p.window_eff) == (q.ages, q.events, q.censor_tie, None), cid
            c["unwindowed_identical"] += 1
            continue
        c["windowed"] += 1
        assert obs.delta, cid                             # every library window cohort has a delta
        d = float(obs.delta)
        lo, hi = obs.window
        rec = p.record["window_snap"]
        assert rec["applied"] is True, (cid, rec["reason"])
        c["snapped"] += 1
        lo_eff, hi_eff = p.window_eff
        if lo <= d / 2 * (1 + 1e-9):
            assert lo_eff == 0.0, cid
            c["lo_zero"] += 1
        else:
            assert _on(lo_eff, d) and lo - 1e-9 * d <= lo_eff < lo + d, (cid, lo, lo_eff)
            c["lo_moved"] += lo_eff != lo
        assert _on(hi_eff, d) and hi - d < hi_eff <= hi + 1e-9 * d, (cid, hi, hi_eff)
        c["hi_moved"] += hi_eff != hi
        k_hi = round(hi_eff / d)
        n_dc = 0
        for a0, e0, a1, e1 in zip(obs.ages, obs.events, p.ages, p.events):
            assert lo_eff - 1e-9 * max(1.0, lo_eff) <= a1 <= hi_eff + 1e-9 * hi_eff, cid
            in_bin = a0 / d >= k_hi - 1e-9 * max(1.0, k_hi)
            if in_bin:
                assert e1 == 0 and a1 == min(a0, hi_eff), (cid, a0, e0, a1, e1)
                n_dc += e0
            else:
                assert a1 == a0, cid
        c["deaths_censored"] += n_dc
        c["cohorts_with_edge_deaths"] += n_dc > 0
        c["ages_moved"] += rec["n_moved"]
        assert rec["n_deaths_censored"] == n_dc, cid
        assert not any(e == 1 and a / d >= k_hi - 1e-9 * k_hi for a, e in zip(p.ages, p.events)), cid
        c["relabel_after_snap"] += bool(p.censor_tie["applied"])
    assert c["windowed"] == c["snapped"] == 350 and c["unwindowed_identical"] == 2290 - 350
    assert c["lo_zero"] == 4                              # the Guinea-pig window cohorts (lo = delta/2)
    print("\nSNAP CENSUS", c)
