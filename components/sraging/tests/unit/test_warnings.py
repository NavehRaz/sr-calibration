"""The seven truth-free checks (T047, T070; FR-016, FR-017, FR-017a, FR-017b, FR-018; spec 002 FR-026, FR-031)."""
import numpy as np
import pytest

from sraging.cohort import Cohort
from sraging.result import Claim
from sraging.warnings import evaluate

CODES = ("BASIN_GAIN", "BASIN_OUTSIDE", "CENSOR_TIE", "DELTA_GRID", "BOX_EDGE", "FLOORED_DEATHS", "AGE0_DEATHS")
BOX = {"eta": (1e-12, 1e3), "beta": (1e-12, 1e3), "epsilon": (1e-12, 1e3), "kappa": (1e-3, 300.0)}
THETA = {"eta": 0.01, "beta": 1.0, "epsilon": 0.1, "kappa": 1.0}


def cl(p, shape="closed", lo=1.0, hi=2.0, point=1.5):
    return Claim(p, shape, lo, hi, point, 0.95, 1, "")


CLAIMS = tuple(cl(p) for p in ("eta", "beta", "epsilon", "kappa"))
COHORT = Cohort.from_arrays([3.0, 5.0, 8.0, 13.0], [1, 1, 0, 1], delta=1.0)


def codes(ws):
    return [w.code for w in ws]


def test_all_seven_checks_run_when_given_their_inputs():
    ws, checks = evaluate(COHORT, CLAIMS, {}, theta_hat=THETA, box=BOX, floor_report={"deaths": []})
    assert [c.code for c in checks] == list(CODES) and all(c.ran for c in checks)
    assert ws == () and not any(c.fired for c in checks)


def test_box_edge_is_not_run_without_theta_and_box():
    _, checks = evaluate(COHORT, CLAIMS, {})
    ran = {c.code: c.ran for c in checks}
    assert ran.pop("BOX_EDGE") is False and ran.pop("FLOORED_DEATHS") is False and all(ran.values())


@pytest.mark.parametrize("p,value,fires", [
    ("beta", 999.9999999999998, True), ("beta", 1e3, True), ("beta", 999.5, True), ("beta", 998.0, False),
    ("eta", 1e3, True), ("epsilon", 1e3, True), ("kappa", 300.0, True), ("kappa", 299.0, False),
    ("kappa", 1e-3, False), ("beta", 1e-12, False),          # lower edges are values, not the box (T070)
])
def test_box_edge_fires_on_upper_edges_only(p, value, fires):
    ws, checks = evaluate(COHORT, CLAIMS, {}, theta_hat={**THETA, p: value}, box=BOX)
    assert ("BOX_EDGE" in codes(ws)) is fires
    assert next(c for c in checks if c.code == "BOX_EDGE").fired is fires


def test_box_edge_names_the_parameters_once():
    ws, _ = evaluate(COHORT, CLAIMS, {}, theta_hat={**THETA, "beta": 1e3, "kappa": 300.0}, box=BOX)
    w = [w for w in ws if w.code == "BOX_EDGE"]
    assert len(w) == 1 and w[0].data["parameters"] == ["beta", "kappa"]
    assert "D-018" in w[0].message and "not usable" in w[0].message


@pytest.mark.parametrize("gain,recentred,fires", [(1.93, True, True), (1.92, True, False), (50.0, False, False)])
def test_basin_gain(gain, recentred, fires):
    diag = {"grid": {"eta": {"recentred": recentred, "gain_nats": gain}, "beta": {"recentred": False}}}
    ws, checks = evaluate(COHORT, CLAIMS, diag)
    assert ("BASIN_GAIN" in codes(ws)) is fires
    assert dict((c.code, c.fired) for c in checks)["BASIN_GAIN"] is fires
    if fires:
        w = next(w for w in ws if w.code == "BASIN_GAIN")
        assert w.data["gain_nats"] == gain and w.data["parameters"] == ["eta"] and "T-JAX-BASIN" in w.message


@pytest.mark.parametrize("point,fires", [(2.0 * (1 + 2e-9), True), (2.0 * (1 + 5e-10), False), (0.999, True),
                                         (1.0 * (1 - 5e-10), False)])
def test_basin_outside_tolerance(point, fires):
    claims = (cl("eta", point=point),) + CLAIMS[1:]
    ws, _ = evaluate(COHORT, claims, {})
    assert ("BASIN_OUTSIDE" in codes(ws)) is fires


def test_basin_outside_one_warning_per_parameter_and_only_closed():
    claims = (cl("eta", point=5.0), cl("beta", point=5.0), cl("epsilon", "upper_bound", None, 2.0, 9.0),
              cl("kappa", "none", None, None, None))
    ws, _ = evaluate(COHORT, claims, {})
    outs = [w for w in ws if w.code == "BASIN_OUTSIDE"]
    assert [w.data["parameter"] for w in outs] == ["eta", "beta"]


def tie_cohort(n, tied, delta=1.0, end=True):
    ages = list(np.linspace(10.0, 40.0, n - tied - 1).round())
    ages += [50.0] * tied + [50.0 if end else 45.0]             # deaths at the censoring age, then the censoring
    events = [1] * (n - tied - 1) + [1] * tied + [0]
    if not end:
        ages[-1], events[-1] = 45.0, 0
        ages.append(60.0); events.append(1); ages.pop(0); events.pop(0)
    return Cohort.from_arrays(ages, events, delta=delta)


def test_censor_tie_at_exactly_half_a_percent_fires_and_just_below_does_not():
    ws, _ = evaluate(tie_cohort(1000, 5), CLAIMS, {})
    w = next(w for w in ws if w.code == "CENSOR_TIE")
    assert w.data["tied"] == 5 and w.data["fraction"] == 5 / 1000 and "eta upward" in w.message
    ws, _ = evaluate(tie_cohort(1000, 4), CLAIMS, {})
    assert "CENSOR_TIE" not in codes(ws)


def test_censor_tie_needs_end_of_study_censoring_and_delta():
    ws, _ = evaluate(tie_cohort(1000, 20, end=False), CLAIMS, {})
    assert "CENSOR_TIE" not in codes(ws)
    c = tie_cohort(1000, 20)
    cont = Cohort.from_arrays(c.ages, c.events, continuous=True)
    ws, checks = evaluate(cont, CLAIMS, {})
    assert "CENSOR_TIE" not in codes(ws) and dict((k.code, k.ran) for k in checks)["CENSOR_TIE"]
    alldead = Cohort.from_arrays([3.0, 5.0], [1, 1], delta=1.0)
    assert "CENSOR_TIE" not in codes(evaluate(alldead, CLAIMS, {})[0])


def test_delta_grid():
    on = Cohort.from_arrays([1.0, 2.0, 7.0], [1, 1, 1], delta=1.0)
    off = Cohort.from_arrays([1.0, 2.0 + 2e-6, 7.0], [1, 1, 1], delta=1.0)
    assert "DELTA_GRID" not in codes(evaluate(on, CLAIMS, {})[0])
    w = next(w for w in evaluate(off, CLAIMS, {})[0] if w.code == "DELTA_GRID")
    assert w.data["fraction"] == pytest.approx(1 / 3)
    cont = Cohort.from_arrays([1.5, 2.25], [1, 1], continuous=True)
    assert "DELTA_GRID" not in codes(evaluate(cont, CLAIMS, {})[0])


def test_warnings_change_no_number():
    diag = {"grid": {"eta": {"recentred": True, "gain_nats": 9.0}}, "loglik": -1.5}
    before = repr(diag), repr(CLAIMS)
    evaluate(COHORT, CLAIMS, diag)
    assert (repr(diag), repr(CLAIMS)) == before


# -- AGE0_DEATHS (spec 002 FR-031; Naveh, 2026-10-04) ---------------------------------------------------------
def _cohort(n_age0, n=300):
    rng = np.random.default_rng(5)
    ages = np.r_[np.zeros(n_age0), np.floor(rng.normal(900.0, 120.0, n))]
    return Cohort.from_arrays(ages, np.ones(ages.size, int), delta=1.0)


def test_age0_deaths_fires_on_a_few_deaths_in_the_first_interval_and_names_the_cutoff():
    ws, checks = evaluate(_cohort(1), CLAIMS, {})
    w = next(w for w in ws if w.code == "AGE0_DEATHS")
    assert "1 death(s) are recorded in the first interval [0, 1)" in w.message
    assert "refit without them (a cut-off at 1" in w.message and "with and without" in w.message
    assert "cannot fit a window until spec 006" in w.message
    assert w.data["deaths_first_interval"] == 1 and w.data["cutoff"] == 1.0
    assert next(c for c in checks if c.code == "AGE0_DEATHS").fired


@pytest.mark.parametrize("k,fires", [(0, False), (1, True), (5, True), (6, False), (40, False)])
def test_age0_deaths_threshold_is_max_5_or_1_percent(k, fires):
    """300 later deaths: max(5, 1% of 300 + k) = 5 for small k, so 6 and 40 are 'many' (the README covers
    infant mortality; the check is for a handful of isolated early deaths)."""
    ws, _ = evaluate(_cohort(k), CLAIMS, {})
    assert ("AGE0_DEATHS" in codes(ws)) is fires


def test_age0_deaths_scales_with_the_cohort():
    ws, _ = evaluate(_cohort(20, n=3000), CLAIMS, {})          # 20 <= 1% of 3020
    assert "AGE0_DEATHS" in codes(ws)


def test_age0_deaths_never_fires_without_delta():
    c = Cohort.from_arrays([0.5, 3.0, 5.0, 8.0], [1, 1, 1, 1], continuous=True)
    ws, checks = evaluate(c, CLAIMS, {})
    assert "AGE0_DEATHS" not in codes(ws) and next(x for x in checks if x.code == "AGE0_DEATHS").ran


# ---- spec 003 (T021): the checks for jax2 fits ------------------------------------------------------------

def test_basin_gain_takes_the_archive_gap_and_fires_only_above_1_92():
    import numpy as np
    from sraging import Cohort
    from sraging.warnings import evaluate
    c = Cohort.from_arrays(np.array([10.0, 20.0, 30.0]), np.array([1, 1, 1]), delta=1.0)
    ws, checks = evaluate(c, (), {}, gap_nats=0.3)
    assert not [w for w in ws if w.code == "BASIN_GAIN"]
    ws, _ = evaluate(c, (), {}, gap_nats=2.5)
    w = next(w for w in ws if w.code == "BASIN_GAIN")
    assert "2.5 nats" in w.message and "evaluated" in w.message and w.data["source"] == "archive"


def test_censor_tie_is_quiet_when_jax2_relabelled_and_says_so_when_it_did_not():
    import numpy as np
    from sraging import Cohort
    from sraging.warnings import evaluate
    ages = np.r_[np.arange(1.0, 196.0), np.full(5, 200.0), np.full(10, 200.0)]
    events = np.r_[np.ones(195, int), np.ones(5, int), np.zeros(10, int)]
    c = Cohort.from_arrays(ages, events, delta=1.0)
    ws, _ = evaluate(c, (), {}, relabel={"applied": True, "n_relabelled": 5})
    assert not [w for w in ws if w.code == "CENSOR_TIE"]
    ws, _ = evaluate(c, (), {}, relabel={"applied": False, "reason": "censored records at several ages"})
    w = next(w for w in ws if w.code == "CENSOR_TIE")
    assert "relabel did not apply" in w.message and "several ages" in w.message


# ---- spec 004 (T008): the sampler warnings ---------------------------------------------------------------
SAMPLER = {   # code -> (its data, what the rendered message must show of it)
    "HET_UNCONVERGED": ({"M": 128, "M_check": 192, "abs_dll": 0.0731, "threshold": 0.05}, ["M = 192", "0.0731"]),
    "HET_NOT_CHECKED": ({"M": 11}, ["M = 11", "check=False", "not clean"]),
    "HET_OUTSIDE_MEASURED_RANGE": ({"sd": 0.6, "lo": 0.1, "hi": 0.5}, ["sd = 0.6", "0.1-0.5"]),
    "LANGEVIN_STEP_BIAS_NOT_MEASURED": ({"dt": 0.01, "n": 25000}, ["dt = 0.01", "n = 25,000", "NOT MEASURED"]),
    "LANGEVIN_INCONSISTENT_AT_N": ({"n": 25000, "ks_D": 0.0123, "crit_95": 0.00859, "p_value": 1.2e-3},
                                   ["n = 25,000", "D = 0.0123", "crit_95 = 0.00859", "p = 0.0012"]),
    "LANGEVIN_STEP_BIAS_ABOVE_CRIT": ({"bias_at_dt": 0.1093, "crit_95": 0.03037, "n": 2000, "dt": 4.0},
                                      ["b̂(dt) = 0.1093", "crit_95 = 0.03037", "n = 2,000", "dt = 4", "not clean",
                                       "halve dt", 'wall="reflect"']),
    "LANGEVIN_SURVIVORS_AT_TMAX": ({"n_survived": 3, "n": 25000, "t_max": 120.0}, ["3 of 25,000", "t_max = 120"]),
    "LANGEVIN_NO_FP_REFERENCE": ({"reason": "a tensor product of 46,656 nodes exceeds the 20,000-node budget"},
                                 ["no FP reference: a tensor product of 46,656 nodes", "not clean"]),
    "LANGEVIN_PATHS_INCONSISTENT_AT_N": ({"age": 40.0, "n_alive": 18750, "wall": "clamp", "statistic": "censored",
                                          "ks_D": 0.021, "crit": 0.0146},
                                         ["age 40", "n_alive = 18,750", "clamp, censored", "D = 0.021", "0.0146"]),
    "LANGEVIN_PATHS_WALL_ATOM": ({"age": 10.0, "p0": 0.031, "limit": 0.00146, "x_c": 0.004},
                                 ["0.031", "age 10", "crit/10 = 0.00146", "x_c = 0.004", 'wall="reflect"']),
    "ENGINE_AUTO_LANGEVIN": ({"reason": "more than one heterogeneous parameter"},
                             ["Langevin", "more than one heterogeneous parameter"]),
}


def test_the_sampler_warnings_are_spec_004s_eleven_and_none_is_a_fit_check():
    from sraging.result import CHECK_CODES
    from sraging.warnings import SAMPLER_WARNINGS
    assert set(SAMPLER_WARNINGS) == set(SAMPLER) and len(SAMPLER_WARNINGS) == 11
    assert not set(SAMPLER_WARNINGS) & set(CHECK_CODES)


@pytest.mark.parametrize("code", sorted(SAMPLER))
def test_each_sampler_warning_renders_with_its_data(code):
    from sraging.warnings import sampler_warning
    data, shown = SAMPLER[code]
    w = sampler_warning(code, **data)
    assert w.code == code and w.data == data and "{" not in w.message
    for s in shown:
        assert s in w.message, (code, s, w.message)


def test_het_unconverged_reads_the_tasks_template():
    from sraging.warnings import sampler_warning
    w = sampler_warning("HET_UNCONVERGED", M=128, M_check=192, abs_dll=0.0731, threshold=0.05)
    assert w.message == "the mixture did not converge by M = 192: |ℓ(128) − ℓ(192)| = 0.0731 nats ≥ 0.05"


def test_a_sampler_warning_without_its_data_or_with_an_unknown_code_is_refused():
    from sraging.warnings import sampler_warning
    with pytest.raises(ValueError, match="abs_dll"):
        sampler_warning("HET_UNCONVERGED", M=128, M_check=192, threshold=0.05)
    with pytest.raises(ValueError, match="HET_NOPE"):
        sampler_warning("HET_NOPE", M=11)
    assert sampler_warning("HET_NOT_CHECKED", M=11).data == {"M": 11}                    # twin


def _clean_fit(**over):
    from sraging.params import SRParams
    from sraging.result import CHECK_CODES, CheckRecord, FitResult
    kw = dict(status="ok", cohort={"n": 4}, theta_hat=SRParams(**THETA), theta_labels={}, claims=CLAIMS, loglik=-1.0,
              warnings=(), checks=tuple(CheckRecord(c, True, False) for c in CHECK_CODES), diagnostics={},
              timing={}, provenance={"numerics": "v1"})
    kw.update(over)
    return FitResult(**kw)


@pytest.mark.parametrize("code", sorted(SAMPLER))
def test_any_one_sampler_warning_makes_a_result_not_clean(code):
    """A warning, not a note (spec 003b FR-011's SHARP_COHORT_LIMIT keeps a fit clean): in a result's warnings it makes
    the result not clean. FitResult is the result that exists before spec 004's own (MixtureResult, LangevinSample,
    Population), whose `clean` reads the same way (data-model.md: False iff any warning)."""
    from sraging.warnings import sampler_warning
    assert _clean_fit().clean                                                            # twin
    assert not _clean_fit(warnings=(sampler_warning(code, **SAMPLER[code][0]),)).clean
