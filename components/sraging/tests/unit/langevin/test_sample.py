"""`sraging.langevin.simulate` (tasks T059; contracts/python-api.md §5; data-model "LangevinConfig", "LangevinSample",
"ErrorStatement"; research R13-R16, R20, R26, R30): gen-1's own observation transform and heterogeneity law, the error
statement at the user's n (gate C1), the provenance, and every refusal by name beside a twin that runs."""
import math

import numpy as np
import pytest

pytest.importorskip("numba", reason="NOT APPLICABLE: numba (the langevin extra) is not installed")

from sraging import SRParams  # noqa: E402
from sraging import generator as GEN  # noqa: E402
from sraging.engine.inputs import Heterogeneity  # noqa: E402
from sraging.errors import UnsupportedOptionError  # noqa: E402
from sraging.langevin import sample as S  # noqa: E402
from sraging.langevin import simulate  # noqa: E402

P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)
BASE = dict(n=2000, key=7, dt=0.1)


@pytest.fixture(scope="module")
def smp():
    return simulate(P, **BASE)


def spy(monkeypatch, name):
    calls = []
    real = getattr(GEN, name)

    def wrapped(*a, **k):
        calls.append(name)
        return real(*a, **k)
    monkeypatch.setattr(GEN, name, wrapped)
    return calls


# ------------------------------------------------------------------------------------------------- the transform
def test_the_observation_transform_is_gen1s_own_functions(monkeypatch):
    calls = {n: spy(monkeypatch, n) for n in ("_hazard_times", "_censor_time", "_record", "_apply_time_range",
                                               "_het_z")}
    p = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1, h_ext=2e-3)
    s = simulate(p, n=600, key=3, dt=0.5, delta=1.0, censor_frac=0.2, window=(20.0, 300.0), error="ks",
                 heterogeneity=Heterogeneity(parameter="xc", sd=0.2))
    assert all(calls[n] for n in calls), {n: len(c) for n, c in calls.items()}
    assert s.ages.size == s.events.size <= 600 and np.all(s.ages >= 20.0) and np.all(s.ages <= 300.0)
    assert np.all(np.round(s.ages / 1.0) == s.ages / 1.0)                          # recorded on the delta grid
    assert s.times.size == 600 and np.all(s.times > 0)                                  # raw, before recording
    assert S.GEN is GEN                                                                 # gen-1 itself, read-only


def test_b11_no_draw_is_non_positive_at_sd_05():
    cfg = S.LangevinConfig(params=P, n=25_000, key=5, dt=0.5, heterogeneity=Heterogeneity(parameter="xc", sd=0.5))
    a = S.draw_parameters(cfg)
    s = a["xc"] / P.xc
    assert np.all(s > 0) and s.min() < 0.1 and abs(np.mean(s) - 1.0276) < 0.01      # truncated: mean of s 1.0276
    assert np.all(a["eta"] == P.eta)                                                     # only xc varies


def test_a_zero_hazard_draws_nothing_and_shifts_nothing():
    cfg0 = S.LangevinConfig(params=P, n=50, key=5, dt=0.5)
    a = S.draw_parameters(cfg0)
    assert np.all(np.isinf(a["t_ext"]))


# ------------------------------------------------- the register's fixes (T122): each one SRK's legacy would fail
def test_d403_the_realised_step_is_the_stated_dt():
    """D-403 (B2): SRK's kernels stepped t_end / ((N - 1) m) (dt = t[1] - t[0] on linspace(0, t_end, N)) while its grid
    helpers meant t_end / (N m), so the step it ran was not the one it stated. Here dt is the step: every death is at
    an integer number of the stated dt, bit for bit."""
    t_end, N = 214.0, 4999
    dt = t_end / N                                         # the stated step on SRK's grid of N outer steps
    assert t_end / (N - 1) != dt                           # SRK's kernel step for it: the planted legacy differs
    s = simulate(P, n=300, key=3, dt=dt, error="ks")
    assert s.provenance["dt"] == dt
    died = s.times[s.times < s.provenance["t_max"]]
    assert died.size and np.array_equal(np.rint(died / dt) * dt, died)


def test_d408_deaths_are_recorded_on_delta_not_on_the_step_grid():
    """D-408: SRK recorded a death at (j + 1) Delta t on its outer grid whatever m was, so with delta < Delta t every
    recorded age was a multiple of Delta t. Here the age is floor(t / delta) delta (gen-1's _record): at dt =
    Delta t / 16 and delta = Delta t / 4 (the plant's A8 configuration, scaled), ages fill the delta grid between."""
    Dt, delta = 1.0, 0.25
    s = simulate(P, n=1_000, key=4, dt=Dt / 16, delta=delta, error="ks")
    died = np.sort(s.times[s.times < s.provenance["t_max"]])
    ages = np.sort(s.ages[s.events == 1])
    assert np.array_equal(ages, np.floor(died / delta) * delta)                     # each death at floor(t/delta) delta
    assert np.any(np.mod(ages, Dt) != 0.0)                                           # legacy: all multiples of Delta t
    twin = simulate(P, n=1_000, key=4, dt=Dt / 16, delta=Dt, error="ks")            # delta = Delta t: one grid
    assert np.all(np.mod(twin.ages, Dt) == 0.0)


def test_d409_homogeneous_unless_named():
    """D-409: SRK's getSrHetro drew xc at sd 0.2 unless hetro=False (its defaults xc_var=0.2, hetro=True). Here a
    configuration that names no heterogeneity draws every individual at theta exactly; a named spread spreads."""
    cfg = S.LangevinConfig(params=P, n=500, key=5, dt=0.5)
    assert cfg.heterogeneity is None
    a = S.draw_parameters(cfg)
    assert all(np.all(a[p] == getattr(P, p)) for p in ("eta", "beta", "epsilon", "kappa", "xc"))
    named = S.draw_parameters(S.LangevinConfig(params=P, n=500, key=5, dt=0.5,
                                               heterogeneity=Heterogeneity(parameter="xc", sd=0.2)))
    assert np.unique(named["xc"]).size > 1 and np.all(named["eta"] == P.eta)           # the twin: named, xc spreads


# --------------------------------------------------------------------------------------------- the error statement
def test_the_error_statement_has_its_fields_and_the_default_p_hat(smp):
    e = smp.error
    assert e.n == 2000 and e.crit_95 == 1.358 / math.sqrt(2000) and 0 <= e.ks_D <= 1 and 0 <= e.p_value <= 1
    assert e.verdict in ("consistent at n", "inconsistent at n")
    assert e.fp_reference["kind"] == "single" and e.fp_reference["M"] is None and "grid" in e.fp_reference
    b = e.step_bias
    assert b["method"] == "coupled half-step rung" and b["p_hat"] == 0.5 and "not a bound" in b["p_hat_source"]
    assert "conservative" not in b["p_hat_source"]                     # T141 finding 6: b-hat is no bound
    assert b["bias_at_dt"] == b["D_dt_vs_dt2"] / (1 - 2 ** -0.5) and b["se"] > 0 and b["n"] == 2000
    assert set(b) >= {"method", "D_dt_vs_dt2", "se", "p_hat", "p_hat_source", "bias_at_dt"}
    assert smp.error.verdict == "consistent at n" and smp.clean, [w.code for w in smp.warnings]


def test_p_hat_is_one_half_at_a_p1_configuration_and_reads_no_test_result():
    # T142 C4 (Naveh 2026-11-05): the library's p-hat is 1/2 everywhere. The plant is a P1 configuration where the
    # old p_hat() returned P1's fitted 1.0 from tests/acceptance/results/p1_langevin_004.json (ecoli|het|clamp).
    import json
    import os
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[3]
    cells = json.load(open(root / "tests/acceptance/results/p1_langevin_004.json"))["p_hat_cells"]
    c = next(c for c in cells if c["cell"] == "ecoli|het|clamp")["config"]
    assert next(x for x in cells if x["cell"] == "ecoli|het|clamp")["p_hat"] == 1.0          # what the old code read
    cfg = S.LangevinConfig(params=SRParams(**c["params"]), n=500, key=3, dt=c["dt"], kernel=c["kernel"],
                           wall=c["wall"], heterogeneity=Heterogeneity(**c["heterogeneity"]))
    p, src = S.p_hat(cfg)
    assert p == 0.5 and "not a bound" in src and "P1" not in src
    t1, t2 = np.linspace(1.0, 2.0, 500), np.linspace(1.0, 2.0, 500) + 0.01
    b = S.step_bias(cfg, t1, t2)
    assert b["p_hat"] == 0.5 and b["bias_at_dt"] == b["D_dt_vs_dt2"] / (1 - 2 ** -0.5)
    twin = S.LangevinConfig(params=SRParams(**c["params"]), n=500, key=3, dt=c["dt"] / 3, kernel=c["kernel"],
                            wall=c["wall"], heterogeneity=Heterogeneity(**c["heterogeneity"]))
    assert S.p_hat(twin) == (p, src)                                    # the twin: no P1 cell, 1/2 as before
    text = open(os.path.join(os.path.dirname(S.__file__), "sample.py")).read()
    assert "p1_langevin_004" not in text and "tests/acceptance" not in text and not hasattr(S, "P1_RESULT")


def test_a_heterogeneous_reference_carries_its_curve_checked_m():
    s = simulate(P, n=1500, key=9, dt=0.5, heterogeneity=Heterogeneity(parameter="xc", sd=0.2), error="ks")
    ref = s.error.fp_reference
    assert ref["kind"] == "single" and ref["M"] in (5, 8, 11, 17, 25, 38, 57, 85, 128, 192)
    c = ref["curve_check"]
    assert c["verdict"] == "converged" and c["sup_dS"] <= c["bar"] == s.error.crit_95 / 10


def test_error_ks_reads_not_measured_and_is_not_clean():
    s = simulate(P, **{**BASE, "n": 800}, error="ks")
    assert s.error.step_bias["method"] == "NOT MEASURED" and s.error.step_bias["bias_at_dt"] is None
    assert "LANGEVIN_STEP_BIAS_NOT_MEASURED" in [w.code for w in s.warnings] and not s.clean
    t = simulate(P, **{**BASE, "n": 800})                                                       # the twin
    assert t.error.step_bias["method"] == "coupled half-step rung"
    assert t.times.tobytes() == s.times.tobytes()                       # the rung draws nothing the sample uses


# T141 finding 1 (sr-naveh_004.md): the README's ITP θ̂ (days) at clamp dt 4.0, n 2,000, key 7 states a step bias
# of 0.109 against a KS crit of 0.030 while the KS passes; before the fix it read clean.
ITP = SRParams(eta=3.58088451358447e-05, beta=0.03168620431375733, epsilon=0.002370584208801515,
               kappa=0.0010000000000000002, time_unit="days")


def test_a_stated_step_bias_above_crit_warns_and_is_not_clean():
    s = simulate(ITP, n=2000, key=7, dt=4.0)                                          # clamp, the default
    e = s.error
    assert e.verdict == "consistent at n" and e.step_bias["bias_at_dt"] > e.crit_95      # the KS passes; b̂ does not
    (w,) = [w for w in s.warnings if w.code == "LANGEVIN_STEP_BIAS_ABOVE_CRIT"]
    assert w.data == {"bias_at_dt": e.step_bias["bias_at_dt"], "crit_95": e.crit_95, "n": 2000, "dt": 4.0}
    assert "halve dt" in w.message and 'wall="reflect"' in w.message and not s.clean
    t = simulate(ITP, n=2000, key=7, dt=0.125, wall="reflect")                        # the twin: a converged step
    assert t.error.step_bias["bias_at_dt"] < t.error.crit_95 and t.clean, [w.code for w in t.warnings]


def test_error_ks_never_carries_the_bias_warning():
    s = simulate(ITP, n=2000, key=7, dt=4.0, error="ks")
    assert "LANGEVIN_STEP_BIAS_ABOVE_CRIT" not in [w.code for w in s.warnings]          # NOT MEASURED says it instead
    assert "LANGEVIN_STEP_BIAS_NOT_MEASURED" in [w.code for w in s.warnings]


def test_survivors_at_t_max_carry_their_warning():
    s = simulate(P, n=300, key=4, dt=0.5, t_max=20.0, error="ks")
    assert s.n_survived_t_max == 300 and np.all(s.times == 20.0)
    (w,) = [w for w in s.warnings if w.code == "LANGEVIN_SURVIVORS_AT_TMAX"]
    assert w.data == {"n_survived": 300, "n": 300, "t_max": 20.0} and not s.clean


def test_the_default_t_max_is_where_the_fp_survival_falls_below_001_over_n(smp):
    t = smp.provenance["t_max"]
    S_t = smp.error.fp_reference["S_obs_at_t_max"]
    assert S_t < 0.01 / 2000 and smp.n_survived_t_max == 0 and np.all(smp.times < t)


# -------------------------------------------------------------------------------------------------- provenance
def test_the_provenance(smp):
    pv = smp.provenance
    assert set(pv) >= {"engine", "kernel", "wall", "dt", "t_max", "key", "rng", "definition", "members", "n",
                       "record_ages", "chunk", "workers", "numba", "package", "host"}
    assert (pv["engine"], pv["kernel"], pv["wall"], pv["dt"], pv["key"], pv["n"]) == ("langevin", "bridge", "clamp",
                                                                                         0.1, 7, 2000)
    assert pv["rng"] == "xoshiro256** per individual, SplitMix64-seeded" and pv["members"] == []
    assert pv["definition"] is None and pv["record_ages"] is None


# ------------------------------------------------------------------------------------------------------ refusals
@pytest.mark.parametrize("name,value", [("traj_points", [10.0]), ("save_trajectory", True), ("segments", ()),
                                        ("interventions", {}), ("correlation", 0.5), ("nsteps", 6000),
                                        ("time_step_multiplier", 4), ("step_size", 0.1), ("parallel", True),
                                        ("n_jobs", 4), ("seed", 3)])
def test_srk_options_are_refused_by_name(name, value):
    with pytest.raises(UnsupportedOptionError) as ex:
        simulate(P, **{**BASE, "n": 10}, **{name: value})
    assert ex.value.option == name and name in str(ex.value)


def test_the_twin_runs(smp):
    assert smp.times.size == 2000 and smp.ages.size == 2000


@pytest.mark.parametrize("wall", ["mirror", "absorb", "", None])
def test_a_wall_other_than_clamp_or_reflect_is_refused_naming_both(wall):
    with pytest.raises(UnsupportedOptionError) as ex:
        simulate(P, **{**BASE, "n": 10}, wall=wall)
    assert ex.value.option == "wall" and "clamp" in str(ex.value) and "reflect" in str(ex.value)


def test_reflect_runs_and_is_recorded():
    s = simulate(P, **{**BASE, "n": 300}, wall="reflect", error="ks")
    assert s.provenance["wall"] == "reflect"


def test_normal_frac_sd_heterogeneity_is_refused():
    with pytest.raises(UnsupportedOptionError) as ex:
        simulate(P, **{**BASE, "n": 10}, heterogeneity=Heterogeneity(parameter="xc", definition="normal_frac_sd", v=0.2,
                                                                     M=7))
    assert ex.value.option == "heterogeneity" and "B11" in str(ex.value)


def test_numba_missing_is_refused_naming_the_extra(monkeypatch):
    monkeypatch.setattr(S, "_numba_available", lambda: False)
    with pytest.raises(UnsupportedOptionError, match=r"install sraging\[langevin\]"):
        simulate(P, **{**BASE, "n": 10})


@pytest.mark.parametrize("kw,option", [({"n": 0}, "n"), ({"n": 2.5}, "n"), ({"key": -1}, "key"),
                                       ({"key": 2 ** 63}, "key"), ({"dt": 0.0}, "dt"), ({"dt": math.inf}, "dt"),
                                       ({"kernel": "milstein"}, "kernel"), ({"error": "none"}, "error"),
                                       ({"chunk": 0}, "chunk"), ({"workers": 0}, "workers"),
                                       ({"t_max": -1.0}, "t_max"), ({"censor_frac": -0.1}, "censor_frac")])
def test_values_outside_their_domain_are_refused(kw, option):
    with pytest.raises(UnsupportedOptionError) as ex:
        simulate(P, **{**BASE, "n": 10, **kw})
    assert ex.value.option == option


def test_an_unknown_option_is_refused_by_name():
    with pytest.raises(UnsupportedOptionError) as ex:
        simulate(P, **{**BASE, "n": 10}, npeople=10)
    assert ex.value.option == "npeople"


def test_grid_dt_is_the_largest_dt_dividing_every_age():
    assert S.grid_dt([35.0, 70.0], 0.01) == pytest.approx(0.01)
    assert S.grid_dt([0.3, 0.45], 0.1) == pytest.approx(0.075)
    with pytest.raises(UnsupportedOptionError):
        S.grid_dt([1.0, math.pi], 0.1)
