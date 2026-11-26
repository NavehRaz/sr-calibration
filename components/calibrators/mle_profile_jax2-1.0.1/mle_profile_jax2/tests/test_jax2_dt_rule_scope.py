"""1.0.1.dev4 (estimator note 15) / 1.0.1.dev5 (note 16; PREREG_REL ADD3 rev 2): DT_RULE ON for every name, SCOPED --
since dev5 to the BINNING-CORRECTED CV_c = sqrt(KM CV^2 - (delta/L)^2/12) < 0.05 (`estimator._cv_c`).

  * the scope's constants: _DTR_CV_MAX = 0.05, _DTR_K = 80;
  * `_cv_c` itself: the formula; exact-time data (no delta) -> CV_c = CV; the EDGE CASE CV^2 <= (delta/L)^2/12 ->
    0.0 (continuous, non-decreasing in CV); a degenerate CV or delta/L passes CV through unchanged; CV_c <= CV;
  * the threshold is STRICT ON CV_c: with `_cv_c` pinned one ulp below / at / one ulp above 0.05 (and at 0.0, the edge
    case) while the KM CV reads 0.06 (out of dev4's scope), `_grid(dt_rule=True)` caps dt at KM CV/80 -- the
    UNcorrected CV -- below 0.05 only; at and above it the solver keywords are the dt_rule=False ones exactly; a
    degenerate KM CV is out of scope whatever CV_c reads;
  * the threshold on exact-time data (CV_c = KM CV): `_km_cv` pinned below / at / above 0.05, at the registered margin
    0.0509 and at degenerate values (dev4's strictness test, carried);
  * against 1.0.1.dev3, BIT FOR BIT (fixtures/dt_rule_dev3/dev3_grids.json: dev3's own `_grid` outputs, made by
    make_dev3_grids.py from `git show 787bbd1`, on 9 synthetic cohorts with KM CV 0.013-0.24, one floor-bound, two
    exact-time, one censored): dt_rule=False is dev3's grid exactly everywhere; dt_rule=True is dev3's dt_rule=False
    grid exactly where CV_c >= 0.05 (where dev3's UNSCOPED rule would have capped dt on 3 of them) and dev3's
    dt_rule=True grid exactly where CV_c < 0.05; the dt_floor record differs only by the rule's keys;
  * WINDOWED cohorts (fixture key "windowed"): the J view (window off) and the H view (window on: left truncation, the
    window snapped to the delta grid, deaths in hi_eff's bin censored) of 3 left-truncated cohorts with an off-grid
    window. The worktree's prepare reproduces dev3's prepared data exactly; the scope's CV is `_km_cv` of THAT view's
    prepared data; the grids are dev3's bit for bit as above -- including a cohort whose J view is out of scope and
    whose H view is in;
  * FINE delta (delta/L ~ 0.01) and exact-time cohorts = 1.0.1.dev4's DECISION AND GRID, bit for bit
    (fixtures/dt_rule_dev4/dev4_grids.json "fine": dev4's own `_grid` from its archived bytes, sha256 74c5201e...):
    every cohort and windowed view above, the record differing from dev4's only by grid_rule.cv_c;
  * BINNED cohorts (fixture "binned"; delta/L 0.10-0.25, deaths in 2-3 bins): two masses 10 apart (KM CV ~0.053, the
    shape of v6 pure-k2-p05-c1-s0) at n 2,000, at n 120 and censored at the last bin are OUT of scope under dev4
    (its stored grid = dev3's dt_rule=False grid) and IN under dev5 (CV_c ~0.043-0.047; the grid = dev3's
    dt_rule=True grid exactly); the edge case CV^2 <= (delta/L)^2/12 is IN under dev5 (CV_c = 0.0) whether dev4 had
    it out (KM CV 0.063, delta/L 0.25) or in (KM CV 0.024); broader binned cohorts (CV_c 0.057, 0.074) and a single
    recorded age (KM CV 0) stay OUT, on dev3's dt_rule=False grid exactly (= dev4's);
  * a BINNED AND WINDOWED cohort (fixture "binned_windowed"; added after the review of PREREG_REL ADD3 rev 2): two
    masses 10 apart, left-truncated, 40 records censored at an OFF-GRID window close (window (85.5, 104.5), delta 10)
    -- the shape of the library's windowed sharp-group c1 H views that dev5 adds. The worktree's prepare reproduces
    dev3's prepared data for both views; the H view's snap (hi_eff 100: deaths in its bin censored, censored ages
    moved down) changes L and the KM CV, so its scope reads THAT view's data: H is OUT under dev4 (KM CV ~0.053) and
    IN under dev5 (CV_c ~0.043) on dev3's dt_rule=True grid bit for bit; J (KM CV ~0.075) is OUT under both, on dev3's
    dt_rule=False grid, where dev3's unscoped rule would have capped it;
  * `run_fit`'s DEFAULT is still 1.0.0's call and fit (1.0.0 = git tag mle_profile_jax2-1.0.0): on a tiny SHARP
    cohort and a tiny BINNED cohort (two masses 10 apart, n 120: in scope under dev5 only), whose grids the class
    constant WOULD change, run_fit with its defaults passes the estimator 1.0.0's keyword set and returns 1.0.0's
    MLE-stage result bit for bit (timing keys aside); the slow-marked variant compares the FULL fit (every stage:
    MLE, profiles, CIs, claims) on the sharp cohort (~9 CPU-min, run with -m slow).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess

import pytest

import _jax2_testlib as T

FIXTURE = os.path.join(T.TESTS, "fixtures", "dt_rule_dev3", "dev3_grids.json")
FIXTURE4 = os.path.join(T.TESTS, "fixtures", "dt_rule_dev4", "dev4_grids.json")
DEV4_EST_SHA256 = "74c5201ec98a5bf94ce71240d4195f7454249ceddf9ef7f8a338b196d5188b6a"
GRID_KW = dict(floor_early=True, n_rule=True, early_all=True)       # the 1.0.1 constants of every name
RULE_KEYS = ("dt_rule", "dt_k", "dt_cap", "dt_cv_max", "cv_c", "dt_in_scope")


@pytest.fixture(scope="module")
def est():
    return T.pkg_module("estimator")


def _canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


def _js(o):
    """JSON round trip (tuples -> lists; floats by repr, so equality stays bit equality)."""
    return json.loads(json.dumps(o))


def _cvc_indep(cv, dl):
    """CV_c recomputed here, independently of `estimator._cv_c` (the same IEEE operations, so bit-equal)."""
    if dl is None:
        return cv
    v = cv * cv - dl * dl / 12.0
    return math.sqrt(v) if v > 0 else 0.0


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def test_scope_constants(est):
    assert est._DTR_CV_MAX == 0.05 and est._DTR_K == 80.0


def test_cv_c_formula_edge_case_and_degenerate_values(est):
    f = est._cv_c
    assert f(0.06, None) == 0.06 and f(0.0509, None) == 0.0509              # exact times: nothing to correct
    for cv, dl in ((0.0529, 0.1056), (0.06, 0.1), (0.0501, 0.00995), (0.2, 0.3)):
        want = math.sqrt(cv * cv - dl * dl / 12.0)
        assert f(cv, dl) == want and f(cv, dl) < cv
    assert f(0.0529, 0.1056) < 0.05 <= 0.0529                                # the binned gap: dev4 out, dev5 in
    assert f(0.0501, 0.00995) > 0.05                                         # fine delta: the correction is tiny
    # the EDGE CASE: cv^2 <= dl^2 / 12 -> 0.0 (in scope); continuous and non-decreasing in cv across it
    assert f(0.02, 0.1) == 0.0 and f(0.0625, 0.2458) == 0.0
    b = 0.2 / math.sqrt(12.0)                                                # the boundary for dl = 0.2
    assert f(b * (1 - 1e-12), 0.2) == 0.0 and 0.0 <= f(b * (1 + 1e-9), 0.2) < 1e-5
    prev = -1.0
    for k in range(1, 400):
        cv = k * 2.5e-4
        x = f(cv, 0.2)
        assert x >= prev and x <= cv
        prev = x
    # degenerate cv or dl: cv unchanged (the caller refuses a degenerate cv on its own)
    for cv in (0.0, -0.01):
        assert f(cv, 0.1) == cv
    assert math.isnan(f(float("nan"), 0.1)) and f(float("inf"), 0.1) == float("inf")
    for dl in (0.0, -0.1, float("nan"), float("inf")):
        assert f(0.06, dl) == 0.06


def _cohort(tag):
    doc = _load(FIXTURE)
    c = next(c for c in doc["cohorts"] if c["tag"] == tag)
    return c["ages"], c["events"], c["delta"]


@pytest.mark.sut
def test_threshold_is_strict_on_cv_c(est, monkeypatch):
    """CV_c pinned (via `_cv_c`) around 0.05 while the KM CV reads 0.06 -- out of 1.0.1.dev4's scope."""
    import numpy as np
    ages, events, delta = _cohort("sharp_cv0.012")
    t, e = np.asarray(ages, float), np.asarray(events, int)
    below, above = float(np.nextafter(0.05, 0.0)), float(np.nextafter(0.05, 1.0))
    km = 0.06
    monkeypatch.setattr(est, "_km_cv", lambda tk, Sk, L: km)
    seen = []
    monkeypatch.setattr(est, "_cv_c", lambda cv, dl: seen.append((cv, dl)) or pinned[0])
    for cvc, in_scope in ((0.0, True), (0.0125, True), (0.049, True), (below, True), (0.05, False), (above, False),
                          (0.0509, False), (0.06, False)):
        pinned = [cvc]
        L0, kw0, c0, f0 = est._grid(t, e, delta, dt_rule=False, **GRID_KW)
        L1, kw1, c1, f1 = est._grid(t, e, delta, dt_rule=True, **GRID_KW)
        rule = f1["grid_rule"]
        assert seen[-1] == (km, f1["dl"]) and f1["dl"] == float(delta) / L1     # _grid's own dl = delta / L
        assert L1 == L0 and c1 == c0, cvc
        assert rule["cv"] == km and rule["cv_c"] == cvc and rule["dt_in_scope"] is in_scope, cvc
        assert rule["dt_cv_max"] == 0.05 and rule["dt_k"] == 80.0
        assert {k: v for k, v in kw1.items() if k != "dt"} == {k: v for k, v in kw0.items() if k != "dt"}, cvc
        if in_scope:
            assert rule["dt_cap"] == km / 80.0 and kw0["dt"] == 0.002                 # the UNcorrected cv's cap
            assert kw1["dt"] == min(kw0["dt"], km / 80.0) and kw1["dt"] < kw0["dt"], cvc
            assert f1["dt"] == kw1["dt"] and rule["dt"] == kw1["dt"]
        else:
            assert rule["dt_cap"] is None, cvc
            assert repr(kw1) == repr(kw0), cvc                                         # bit for bit
            assert {k: v for k, v in f1.items() if k != "grid_rule"} == {k: v for k, v in f0.items()
                                                                          if k != "grid_rule"}, cvc
    # a degenerate KM CV is out of scope whatever CV_c reads (no positive cap exists)
    pinned = [0.0]
    for km in (0.0, -0.01, float("nan"), float("inf")):
        L1, kw1, c1, f1 = est._grid(t, e, delta, dt_rule=True, **GRID_KW)
        L0, kw0, c0, f0 = est._grid(t, e, delta, dt_rule=False, **GRID_KW)
        assert f1["grid_rule"]["dt_in_scope"] is False and f1["grid_rule"]["dt_cap"] is None, km
        assert repr(kw1) == repr(kw0), km


@pytest.mark.sut
def test_threshold_on_exact_time_data_is_dev4s(est, monkeypatch):
    """No delta: CV_c = KM CV, so 1.0.1.dev4's strictness test carries unchanged (KM CV pinned)."""
    import numpy as np
    ages, events, delta = _cohort("sharp_cv0.035_exact")
    assert delta is None
    t, e = np.asarray(ages, float), np.asarray(events, int)
    below, above = float(np.nextafter(0.05, 0.0)), float(np.nextafter(0.05, 1.0))
    cases = [(0.0125, True), (0.049, True), (below, True), (0.05, False), (above, False), (0.0509, False),
             (0.12, False), (0.0, False), (-0.01, False), (float("nan"), False), (float("inf"), False)]
    for cv, in_scope in cases:
        monkeypatch.setattr(est, "_km_cv", lambda tk, Sk, L, _cv=cv: _cv)
        L0, kw0, c0, f0 = est._grid(t, e, delta, dt_rule=False, **GRID_KW)
        L1, kw1, c1, f1 = est._grid(t, e, delta, dt_rule=True, **GRID_KW)
        rule = f1["grid_rule"]
        assert L1 == L0 and c1 == c0 and f1["dl"] is None, cv
        assert rule["dt_in_scope"] is in_scope and rule["dt_cv_max"] == 0.05 and rule["dt_k"] == 80.0, cv
        assert rule["cv_c"] == cv or (math.isnan(cv) and math.isnan(rule["cv_c"])), cv
        assert {k: v for k, v in kw1.items() if k != "dt"} == {k: v for k, v in kw0.items() if k != "dt"}, cv
        if in_scope:
            assert rule["dt_cap"] == cv / 80.0 and kw0["dt"] == 0.002
            assert kw1["dt"] == min(kw0["dt"], cv / 80.0) and kw1["dt"] < kw0["dt"], cv
        else:
            assert rule["dt_cap"] is None, cv
            assert repr(kw1) == repr(kw0), cv


def _check_against_dev3(est, t, e, delta, d3_off, d3_on, tag):
    """dev5's _grid on (t, e) vs dev3's stored outputs: off = dev3 off exactly; on = dev3 on exactly where CV_c < 0.05,
    dev3 off exactly elsewhere; dt_floor differs only by the rule's keys. Returns (cv, cv_c, in_scope, unscoped_caps)."""
    off = dict(zip(("L", "kw", "ceiling", "floor"), _js(est._grid(t, e, delta, dt_rule=False, **GRID_KW))))
    on = dict(zip(("L", "kw", "ceiling", "floor"), _js(est._grid(t, e, delta, dt_rule=True, **GRID_KW))))
    assert _canon(off) == _canon(d3_off), tag                                          # dt_rule off: dev3 exactly
    gr = on["floor"]["grid_rule"]
    cv = gr["cv"]
    cvc = _cvc_indep(cv, on["floor"]["dl"])
    assert gr["cv_c"] == cvc, tag
    insc = bool(0 < cv and cvc < 0.05)
    if insc:
        ref, extra = d3_on, dict(dt_cv_max=0.05, cv_c=cvc, dt_in_scope=True)            # dev3's rule, exactly
    else:
        ref, extra = d3_off, dict(dt_rule=True, dt_k=80.0, dt_cap=None, dt_cv_max=0.05, cv_c=cvc, dt_in_scope=False)
    assert _canon(on["kw"]) == _canon(ref["kw"]), tag                                  # the solver grid, bit for bit
    assert on["L"] == ref["L"] and on["ceiling"] == ref["ceiling"], tag
    want = dict(ref["floor"])
    want["grid_rule"] = dict(ref["floor"]["grid_rule"], **extra)
    assert _canon(on["floor"]) == _canon(want), tag                                    # only the rule's keys differ
    assert set(gr) - set(d3_off["floor"]["grid_rule"]) <= set(RULE_KEYS)
    return cv, cvc, insc, d3_on["kw"]["dt"] < d3_off["kw"]["dt"]


@pytest.mark.sut
def test_grids_against_dev3_bit_for_bit(est):
    import numpy as np
    doc = _load(FIXTURE)
    assert doc["dev3_estimator_sha256"] == "dad4d050f8ec5bc7ad0d4d47f978f5caad3c9f57272ec1ebb161ce3f10b676b8"
    assert doc["grid_kw"] == GRID_KW
    n_in = n_out = n_unscoped_would_cap = 0
    for c in doc["cohorts"]:
        t, e, delta = np.asarray(c["ages"], float), np.asarray(c["events"], int), c["delta"]
        cv, cvc, insc, caps = _check_against_dev3(est, t, e, delta, c["dt_rule_False"], c["dt_rule_True"], c["tag"])
        assert cv == c["km_cv"]
        n_in += insc
        n_out += not insc
        n_unscoped_would_cap += (not insc) and caps
    assert n_in >= 4 and n_out >= 4 and n_unscoped_would_cap >= 3
    floor_c = next(c for c in doc["cohorts"] if c["tag"] == "sharp_cv0.016_floor")
    assert floor_c["dt_rule_False"]["floor"]["binds"] and floor_c["dt_rule_True"]["kw"]["dt"] < 3e-4


@pytest.mark.sut
def test_windowed_views_scope_on_their_own_prepared_data(est):
    import numpy as np
    from SRK.fp_likelihood import kaplan_meier
    om = T.pkg_module("observation_model")
    doc = _load(FIXTURE)
    assert doc["dev3_observation_model_sha256"] == "deff521c80985a699f513758c494d9e0f3e8ab0fdf5cc97da292b342025d25d0"
    seen = {}
    for c in doc["windowed"]:
        for view, v in sorted(c["views"].items()):
            prep = om.prepare(c["ages"], c["events"], c["delta"], tuple(c["window"]), window_on=(view == "H"),
                              relabel_on=True)
            t, e = np.asarray(prep.ages, float), np.asarray(prep.events, int)
            assert [float(x) for x in t] == v["prep_ages"] and [int(x) for x in e] == v["prep_events"], c["tag"]
            assert (None if prep.window_eff is None else [float(x) for x in prep.window_eff]) == v["window_eff"]
            if view == "H":                                          # the snap changed the data the scope reads
                assert prep.window_eff is not None and list(prep.window_eff) != list(c["window"])
                assert prep.record["window_snap"]["applied"] and prep.record["window_snap"]["n_moved"] > 0
            cv, cvc, insc, caps = _check_against_dev3(est, t, e, c["delta"], v["dt_rule_False"], v["dt_rule_True"],
                                                      f"{c['tag']} {view}")
            assert cv == v["km_cv"] and cv == est._km_cv(*kaplan_meier(t, e))    # _km_cv of THIS view's data
            seen[(c["tag"], view)] = (insc, caps)
    assert seen[("win_straddle", "J")] == (False, True) and seen[("win_straddle", "H")] == (True, True)
    assert seen[("win_sharp_cv0.03", "H")][0] is True
    assert seen[("win_mid_cv0.07", "H")] == (False, True)                # dev3's unscoped rule would have capped it


@pytest.mark.sut
def test_fine_delta_cohorts_take_dev4s_decision_and_grid(est):
    """Fine delta (delta/L ~ 0.01) and exact times: dev5 = 1.0.1.dev4 bit for bit (record: + grid_rule.cv_c)."""
    import numpy as np
    d4 = _load(FIXTURE4)
    assert d4["dev4_estimator_sha256"] == DEV4_EST_SHA256 and d4["grid_kw"] == GRID_KW
    with open(FIXTURE, "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == d4["dev3_fixture_sha256"]
    d3 = _load(FIXTURE)
    data = {(c["tag"], None): (c["ages"], c["events"]) for c in d3["cohorts"]}
    data.update({(c["tag"], view): (v["prep_ages"], v["prep_events"]) for c in d3["windowed"]
                 for view, v in c["views"].items()})
    n = n_in = n_out = 0
    for r in d4["fine"]:
        ages, events = data[(r["tag"], r["view"])]
        t, e = np.asarray(ages, float), np.asarray(events, int)
        tag = f"{r['tag']} {r['view']}"
        new = dict(zip(("L", "kw", "ceiling", "floor"), _js(est._grid(t, e, r["delta"], dt_rule=True, **GRID_KW))))
        old = r["dev4_dt_rule_True"]
        dl = new["floor"]["dl"]
        assert dl is None or dl < 0.011, tag                                           # fine delta or exact times
        assert new["floor"]["grid_rule"]["dt_in_scope"] is old["floor"]["grid_rule"]["dt_in_scope"], tag
        assert _canon(new["kw"]) == _canon(old["kw"]) and new["L"] == old["L"] and new["ceiling"] == old["ceiling"]
        want = dict(old["floor"])
        want["grid_rule"] = dict(old["floor"]["grid_rule"], cv_c=new["floor"]["grid_rule"]["cv_c"])
        assert _canon(new["floor"]) == _canon(want), tag                               # + cv_c, nothing else
        n += 1
        n_in += old["floor"]["grid_rule"]["dt_in_scope"]
        n_out += not old["floor"]["grid_rule"]["dt_in_scope"]
    assert n == 15 and n_in >= 7 and n_out >= 6


@pytest.mark.sut
def test_binned_cohorts_in_scope_under_dev5_out_under_dev4(est):
    import numpy as np
    d4 = _load(FIXTURE4)
    by = {r["tag"]: r for r in d4["binned"]}
    want_in = {"bin2_p05_n2000", "bin2_n120", "bin2_censored", "edge_dev4_out", "edge_dev4_in"}
    want_out = {"out_bin2_mean55", "out_bin3", "degenerate_one_age"}
    assert set(by) == want_in | want_out
    for tag, r in sorted(by.items()):
        t, e = np.asarray(r["ages"], float), np.asarray(r["events"], int)
        d3_off, d3_on, g4 = r["dev3_dt_rule_False"], r["dev3_dt_rule_True"], r["dev4_dt_rule_True"]
        cv, cvc, insc, _caps = _check_against_dev3(est, t, e, r["delta"], d3_off, d3_on, tag)
        assert cv == r["km_cv"] and 0.09 < r["dl"] < 0.25, tag                        # coarsely recorded
        assert insc is (tag in want_in), (tag, cv, cvc)
        dev4_in = g4["floor"]["grid_rule"]["dt_in_scope"]
        assert dev4_in is bool(0 < cv < 0.05), tag                                     # dev4 read the KM CV
        if tag in want_in:
            assert d3_on["kw"]["dt"] < d3_off["kw"]["dt"], tag                         # the cap binds here
        if dev4_in:
            assert _canon(g4["kw"]) == _canon(d3_on["kw"]), tag
        else:
            assert _canon(g4["kw"]) == _canon(d3_off["kw"]), tag                       # dev4: dev3's off grid
    # the binned gap: two masses 10 apart read KM CV ~0.053 (dev4 out) and CV_c ~0.043-0.047 (dev5 in)
    for tag in ("bin2_p05_n2000", "bin2_n120", "bin2_censored"):
        r = by[tag]
        gr = r["dev4_dt_rule_True"]["floor"]["grid_rule"]
        assert gr["dt_in_scope"] is False and 0.05 <= r["km_cv"] < 0.057, tag
        assert 0.04 < _cvc_indep(r["km_cv"], r["dl"]) < 0.048, tag
    # the EDGE CASE (cv^2 <= dl^2/12): CV_c = 0.0, in scope, cap = the uncorrected KM CV / 80
    for tag, dev4_in in (("edge_dev4_out", False), ("edge_dev4_in", True)):
        r = by[tag]
        assert r["km_cv"] ** 2 <= r["dl"] ** 2 / 12.0 and r["km_cv"] > 0, tag
        assert r["dev4_dt_rule_True"]["floor"]["grid_rule"]["dt_in_scope"] is dev4_in, tag
        t, e = np.asarray(r["ages"], float), np.asarray(r["events"], int)
        L, kw, c, f = est._grid(t, e, r["delta"], dt_rule=True, **GRID_KW)
        gr = f["grid_rule"]
        assert gr["cv_c"] == 0.0 and gr["dt_in_scope"] is True and gr["dt_cap"] == r["km_cv"] / 80.0, tag
        assert kw["dt"] == r["km_cv"] / 80.0 > 0, tag
    assert by["edge_dev4_out"]["km_cv"] >= 0.05                                       # in scope ONLY under dev5
    # out of scope under both: a broader binned cohort and a single recorded age keep dev3's off grid (= dev4's)
    assert by["degenerate_one_age"]["km_cv"] == 0.0


@pytest.mark.sut
def test_binned_windowed_h_view_in_scope_under_dev5_out_under_dev4(est):
    """Coarse binning AND a snapped window: the H view's scope reads its own prepared data (dev5 IN, dev4 OUT)."""
    import numpy as np
    om = T.pkg_module("observation_model")
    d4 = _load(FIXTURE4)
    assert d4["dev3_observation_model_sha256"] == "deff521c80985a699f513758c494d9e0f3e8ab0fdf5cc97da292b342025d25d0"
    (c,) = d4["binned_windowed"]
    assert c["tag"] == "win_bin2_offgrid" and c["delta"] == 10.0 and c["window"] == [85.5, 104.5]
    seen = {}
    for view in ("J", "H"):
        v = c["views"][view]
        prep = om.prepare(c["ages"], c["events"], c["delta"], tuple(c["window"]), window_on=(view == "H"),
                          relabel_on=True)
        t, e = np.asarray(prep.ages, float), np.asarray(prep.events, int)
        assert [float(x) for x in t] == v["prep_ages"] and [int(x) for x in e] == v["prep_events"], view
        assert (None if prep.window_eff is None else [float(x) for x in prep.window_eff]) == v["window_eff"], view
        d3_off, d3_on, g4 = v["dev3_dt_rule_False"], v["dev3_dt_rule_True"], v["dev4_dt_rule_True"]
        cv, cvc, insc, caps = _check_against_dev3(est, t, e, c["delta"], d3_off, d3_on, f"win_bin2_offgrid {view}")
        assert cv == v["km_cv"] and 0.1 < v["dl"] < 0.11, view                          # coarsely recorded
        dev4_in = g4["floor"]["grid_rule"]["dt_in_scope"]
        assert dev4_in is bool(0 < cv < 0.05) and dev4_in is False, view               # dev4 read the KM CV: out
        assert _canon(g4["kw"]) == _canon(d3_off["kw"]), view                           # dev4: dev3's off grid
        assert caps, view                                                               # dev3's unscoped rule caps
        seen[view] = (cv, cvc, insc)
    # H: the snap moved the window close down to the grid (hi_eff 100) and censored the deaths in its bin
    snap = c["views"]["H"]["snap"]
    assert snap["applied"] and snap["hi_eff"] == 100.0 and snap["n_moved"] > 0 and snap["n_deaths_censored"] > 0
    cvH, cvcH, inH = seen["H"]
    assert inH is True and 0.05 <= cvH < 0.057 and 0.04 < cvcH < 0.048                 # dev4 out, dev5 IN
    cvJ, cvcJ, inJ = seen["J"]
    assert inJ is False and cvcJ >= 0.05 and cvJ != cvH                                 # the views differ


#: run in a fresh interpreter per package tree: run_fit with its DEFAULTS on tiny cohorts (COHORTS), the MLE stage only
#: (est.fit wrapped to add stages=("mle",); the wrapper records the keywords run_fit passed); the slow variant drops
#: the stages keyword (the FULL fit)
_RUN_DEFAULT = r"""
import sys, json, os, importlib.util
pkg = sys.argv[1]
s = importlib.util.spec_from_file_location("jax2_tests.dts.adapter", os.path.join(pkg, "__init__.py"))
ad = importlib.util.module_from_spec(s); sys.modules[s.name] = ad; s.loader.exec_module(ad)
assert ad.__name__ != ad._KEY
import numpy as np
from srbench import npcompat
from srbench.contract.types import Observation
est = ad._sibling("estimator")
calls = []
orig = est.fit
def fit(fj, ages, events, **kw):
    calls.append(sorted(kw))
    return orig(fj, ages, events, stages=("mle",), **kw)
est.fit = fit
COHORTS = ("sharp", "binned")
def cohort(which):
    if which == "sharp":
        rng = np.random.default_rng(20261015)
        t = np.ceil(rng.gamma(1 / 0.03 ** 2, 100 * 0.03 ** 2, 120))
        return Observation(cohort_id="dts-tiny-sharp", ages=tuple(float(x) for x in t), events=(1,) * t.size, delta=1.0)
    t = [90.0] * 60 + [100.0] * 60                                   # two masses 10 apart: dev4 out, dev5 in
    return Observation(cohort_id="dts-tiny-binned", ages=tuple(t), events=(1,) * len(t), delta=10.0)
TIMING = {"wall_s", "compile_s", "seed_s", "mle_s", "cpu_s", "refit_cpu_s", "fit_wall_s"}
def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in TIMING}
    if isinstance(o, (list, tuple)):
        return [strip(v) for v in o]
    if isinstance(o, np.ndarray):
        return strip(o.tolist())
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o
out = {}
for which in COHORTS:
    obs = cohort(which)
    calls.clear()
    res = ad.run_fit(obs, relabel=True)
    prep = ad._sibling("observation_model").prepare(obs.ages, obs.events, obs.delta, obs.window, window_on=False,
                                                     relabel_on=True)
    t, e = np.asarray(prep.ages, float), np.asarray(prep.events, int)
    g_off = est._grid(t, e, obs.delta, dt_rule=False)
    g_on = est._grid(t, e, obs.delta, dt_rule=True)
    rule = g_on[3].get("grid_rule") or {}
    out[which] = dict(calls=list(calls), result=json.dumps(strip(res), sort_keys=True, default=repr),
                      cv=rule.get("cv"), cv_c=rule.get("cv_c"), dt_off=g_off[1]["dt"], dt_on=g_on[1]["dt"])
print(json.dumps(dict(versions=sorted(set(ad.VERSIONS.values())), DT_RULE=ad.MleProfileJax2.DT_RULE, cohorts=out)))
"""


def _check_run_fit_default(tmp_path, full):
    code = _RUN_DEFAULT if not full else _RUN_DEFAULT.replace('stages=("mle",), ', '').replace(
        'COHORTS = ("sharp", "binned")', 'COHORTS = ("sharp",)')
    assert (code != _RUN_DEFAULT) is full
    v100 = tmp_path / "v100" / "mle_profile_jax2"
    v100.mkdir(parents=True)
    for f in ("__init__.py", "estimator.py", "fp_jax.py", "observation_model.py", "VERSION_LOCK.json"):
        r = subprocess.run(["git", "-C", T.ROOT, "show", f"mle_profile_jax2-1.0.0:methods/mle_profile_jax2/{f}"],
                           capture_output=True)
        if r.returncode != 0:
            pytest.skip(f"git tag mle_profile_jax2-1.0.0 is not available here: {r.stderr[-200:]!r}")
        (v100 / f).write_bytes(r.stdout)
    old = T.run_py(code.replace("sys.argv[1]", repr(str(v100))), timeout=3600)
    new = T.run_py(code.replace("sys.argv[1]", repr(T.PKG)), timeout=3600)
    assert old["_rc"] == 0, old["_stderr"]
    assert new["_rc"] == 0, new["_stderr"]
    assert old["versions"] == ["1.0.0"] and new["versions"] == ["1.0.1"]          # 1.0.1: the freeze (was 1.0.1.dev5)
    assert old["DT_RULE"] is False and new["DT_RULE"] is True
    assert sorted(new["cohorts"]) == (["sharp"] if full else ["binned", "sharp"])
    for which, n in new["cohorts"].items():
        o = old["cohorts"][which]
        # the class constant WOULD change this cohort's grid under 1.0.1.dev5 (so the identity is not vacuous)
        assert n["cv_c"] < 0.05 and n["dt_on"] < n["dt_off"], which
        if which == "binned":
            assert n["cv"] >= 0.05, n                                  # out of 1.0.1.dev4's scope: dev5's new line
        assert o["calls"] == n["calls"] and "dt_rule" not in n["calls"][0], which
        assert o["result"] == n["result"], which                                      # bit for bit
        res = json.loads(n["result"])
        assert math.isfinite(res["loglik"]), which
    return json.loads(new["cohorts"]["sharp"]["result"])


@pytest.mark.sut
def test_run_fit_default_is_the_1_0_0_fit_on_tiny_sharp_and_binned_cohorts(tmp_path):
    _check_run_fit_default(tmp_path, full=False)


@pytest.mark.slow
@pytest.mark.sut
def test_run_fit_default_full_fit_is_the_1_0_0_fit(tmp_path):
    res = _check_run_fit_default(tmp_path, full=True)
    assert res["ci"]                                         # the profile / CI stages ran (and were compared)
