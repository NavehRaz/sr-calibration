"""Adaptive M inside a series (tasks T028; research R7; data-model "Series and maximisation"; contracts/python-api.md
§4): explore at M_low (two rungs below the start rung, floor 5), the checked ladder afresh at every final point, and
`maximise_fixed_sd`'s phases A explore, B check, C polish, D check. A profile point is the same call with one more
parameter fixed."""
import numpy as np
import pytest

from sraging import Cohort, SRParams
from sraging.engine import mixture as MIX
from sraging.engine import series as SER
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy
from sraging.engine.inputs import Heterogeneity
from sraging.engine.series import AdaptivePolicy, maximise_fixed_sd
from sraging.errors import UnsupportedOptionError

pytestmark = pytest.mark.engine
RNG = np.random.default_rng(3)
T = np.sort(np.floor(RNG.gamma(9.0, 14.0, 300)) + 1.0)
E = (RNG.random(300) < 0.9).astype(int)
C = Cohort.from_arrays(T, E, delta=1.0)
GRID = GridPolicy("fixed", N=200, dt=2.0 ** -8, tau_max=2.5, L=124.0)
P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)
FREE2 = ("eta", "epsilon")                         # two free parameters keep the maximisations short


@pytest.fixture(scope="module")
def ev():
    return Evaluator(C, numerics="v1", grid=GRID, heterogeneity=Heterogeneity(parameter="xc", sd=0.2))


@pytest.fixture(scope="module")
def ev2():
    return Evaluator(C, numerics="v1", grid=GRID, free=FREE2, fixed={"beta": P.beta, "kappa": P.kappa},
                     heterogeneity=Heterogeneity(parameter="xc", sd=0.2))


@pytest.mark.parametrize("start,low", [(5, 5), (8, 5), (11, 5), (17, 8), (25, 11), (38, 17), (57, 25), (85, 38),
                                       (128, 57), (192, 85)])
def test_m_low_is_two_rungs_below_the_start_floor_5(start, low):
    assert AdaptivePolicy().low_rung(start) == low
    assert AdaptivePolicy(m_low=17).low_rung(start) == 17                       # an explicit rung wins


def test_a_non_rung_m_low_is_refused():
    with pytest.raises(UnsupportedOptionError) as ex:
        AdaptivePolicy(m_low=10)
    assert ex.value.option == "m_low"
    AdaptivePolicy(m_low=11)                                                    # the twin


def test_an_unknown_phase_is_refused(ev):
    with pytest.raises(UnsupportedOptionError) as ex:
        ev.series().evaluate(P, phase="polish")
    assert ex.value.option == "phase"


def test_an_explore_value_is_never_clean_and_a_final_one_carries_the_check(ev):
    s = ev.series(policy=AdaptivePolicy())
    x = s.evaluate(P, phase="explore")
    start = ev.mixture_value(P).node_record["check"]["start"]["rung"]
    assert x.node_record["check"]["verdict"] == "not_checked: explore" and not x.clean
    assert x.node_record["M_used"] == AdaptivePolicy().low_rung(start) == s.record().M_low
    f = s.evaluate(P, phase="final")
    assert f.node_record["check"]["verdict"] == "converged" and f.clean
    rec = s.record()
    assert [ph["name"] for ph in rec.phases] == ["explore", "final"]
    assert all(ph["n_evals"] == 1 and ph["cpu_s"] > 0 for ph in rec.phases)
    assert rec.M_final_by_point == (f.node_record["M_used"],)


def test_a_final_value_does_not_depend_on_the_call_history(ev):
    fresh = ev.series().evaluate(P, phase="final").value
    s = ev.series(policy=AdaptivePolicy(m_low=5))
    s.evaluate(SRParams(eta=7e-4, beta=0.09, epsilon=9e-3, kappa=0.12), phase="explore")
    s.evaluate(SRParams(eta=5e-4, beta=0.07, epsilon=7e-3, kappa=0.1), phase="final")
    assert s.evaluate(P, phase="final").value == fresh == ev.mixture_value(P).value


def test_maximise_records_phases_a_to_d_with_m_cpu_and_termination(ev2):
    r = maximise_fixed_sd(ev2, P)
    names = [ph["name"] for ph in r.phases]
    assert names[:4] == ["A explore", "B check", "C polish", "D check"]
    for ph in r.phases:
        assert ph["M"] in MIX.RUNGS and ph["cpu_s"] >= 0 and ph["n_evals"] >= 1 and "status" in ph["termination"]
    a, c = r.phases[0], [ph for ph in r.phases if ph["name"] == "C polish"][-1]
    assert a["M"] == r.M_low and c["M"] == r.M_final and a["termination"]["nit"] >= 1
    assert r.check["verdict"] == "converged" and r.ll_final == r.check["ll_M"] and r.clean
    assert r.n_evals == sum(ph["n_evals"] for ph in r.phases)
    assert r.cost_record["measured_cpu_s"] > 0 and r.cost_record["statements"]
    # a maximum: the free gradient at theta-hat is small
    g = MIX.mixture_value(ev2, r.theta_hat, grad=True, het=Heterogeneity(parameter="xc", sd=0.2, M=r.M_final,
                                                                         check=False)).grad
    assert max(abs(g[k]) for k in FREE2) < 1e-2


def test_the_full_m_arm_has_no_explore_phase(ev2):
    r = maximise_fixed_sd(ev2, P, full_M=True)
    assert [ph["name"] for ph in r.phases][:3] == ["B check", "C polish", "D check"] and r.M_low is None


def test_a_profile_point_is_the_same_call_with_one_more_parameter_fixed(ev2):
    r = maximise_fixed_sd(ev2, P, fixed={"epsilon": 9e-3})
    assert r.theta_hat.epsilon == 9e-3 and r.theta_hat.eta != P.eta
    assert [ph["name"] for ph in r.phases][:4] == ["A explore", "B check", "C polish", "D check"]
    assert r.free == ("eta",) and r.fixed == {"epsilon": 9e-3}


@pytest.mark.parametrize("free", [("eta", "het_sd"), ("eta", "sd"), ("eta", "het_v")])
def test_sd_in_free_is_refused(ev2, free):
    with pytest.raises(UnsupportedOptionError) as ex:
        maximise_fixed_sd(ev2, P, free=free)
    assert ex.value.option == "free" and "spec 006" in str(ex.value)


def test_a_free_or_fixed_name_the_evaluator_does_not_free_is_refused(ev2):
    with pytest.raises(UnsupportedOptionError):
        maximise_fixed_sd(ev2, P, free=("eta", "beta"))
    with pytest.raises(UnsupportedOptionError):
        maximise_fixed_sd(ev2, P, fixed={"kappa": 0.2})


def test_the_never_raise_plant_stops_at_m_low_and_is_never_clean(ev2):
    r = maximise_fixed_sd(ev2, P, policy=AdaptivePolicy(raise_M=False))
    assert [ph["name"] for ph in r.phases] == ["A explore"] and r.M_final == r.M_low and not r.clean
    assert r.check["verdict"] == "not_checked: explore"


def test_the_optimiser_is_the_registered_one():
    assert SER.OPTIMISER == {"method": "L-BFGS-B", "bounds_log": 5.0, "maxiter": 500, "ftol": 1e-12, "gtol": 1e-6}
