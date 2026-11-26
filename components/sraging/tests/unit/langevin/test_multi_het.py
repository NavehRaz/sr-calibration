"""Heterogeneity in more than one parameter, in the Langevin (tasks T103; research R27; data-model "Multi-parameter
heterogeneity"): a tuple of truncated normals, canonicalised by parameter name; per parameter one stream
hash64(key, i, "het:" + p); sd_p = 0 is s = 1.0 exactly with no draw, so a tuple whose other spreads are 0 equals the
single-parameter draw bit for bit under both walls; a shared stream (HR-11's plant) breaks that. Every refusal has its
twin."""
import math

import numpy as np
import pytest

pytest.importorskip("numba", reason="NOT APPLICABLE: numba (the langevin extra) is not installed")

from sraging import SRParams  # noqa: E402
from sraging.cohort import Cohort  # noqa: E402
from sraging.engine import curves  # noqa: E402
from sraging.engine.density import damage_density  # noqa: E402
from sraging.engine.evaluator import Evaluator  # noqa: E402
from sraging.engine.inputs import Heterogeneity  # noqa: E402
from sraging.errors import UnsupportedOptionError  # noqa: E402
from sraging.langevin import simulate  # noqa: E402
from sraging.langevin import sample as LS  # noqa: E402
from sraging.population import population  # noqa: E402

P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)
XC, ETA, EPS = (Heterogeneity(parameter="xc", sd=0.2), Heterogeneity(parameter="eta", sd=0.2),
                Heterogeneity(parameter="epsilon", sd=0.1))
ETA0 = Heterogeneity.zero("eta")
BASE = dict(n=600, key=13, dt=0.5, error="ks", t_max=300.0)


def cfg(het, **kw):
    return LS.LangevinConfig(params=P, heterogeneity=het, **{**BASE, **kw})


@pytest.mark.parametrize("het,why", [((XC, Heterogeneity(parameter="xc", sd=0.1)), "repeated"),
                                     ((XC, Heterogeneity(parameter="eta", definition="normal_frac_sd", v=0.2, M=7)),
                                      "normal_frac_sd"),
                                     ((), "empty"), ((XC, "eta"), "Heterogeneity")])
def test_a_bad_tuple_is_refused_by_name(het, why):
    with pytest.raises(UnsupportedOptionError) as ex:
        cfg(het)
    assert ex.value.option == "heterogeneity" and why in str(ex.value)
    cfg((XC, ETA))                                                                                     # the twin


def test_correlation_is_refused():
    with pytest.raises(UnsupportedOptionError) as ex:
        simulate(P, heterogeneity=(XC, ETA), correlation=0.5, **BASE)
    assert ex.value.option == "correlation"


def test_the_tuple_on_any_fp_entry_point_is_refused():
    coh = Cohort.from_arrays([10.0, 20.0], [1, 1], continuous=True)
    with pytest.raises(UnsupportedOptionError) as ex:
        Evaluator(coh, heterogeneity=(XC, ETA))
    assert ex.value.option == "heterogeneity"
    with pytest.raises(UnsupportedOptionError):
        curves(P, [10.0, 20.0], heterogeneity=(XC, ETA))
    with pytest.raises(UnsupportedOptionError):
        damage_density(P, [10.0], heterogeneity=(XC, ETA))
    with pytest.raises(UnsupportedOptionError):
        population(P, [10.0], heterogeneity=(XC, ETA), engine="fp")


def test_a_zero_spread_alone_is_refused_and_is_a_tuples_member_only():
    for call in (lambda: Evaluator(Cohort.from_arrays([10.0], [1], continuous=True), heterogeneity=ETA0),
                 lambda: curves(P, [10.0], heterogeneity=ETA0), lambda: damage_density(P, [10.0], heterogeneity=ETA0),
                 lambda: cfg(ETA0), lambda: population(P, [10.0], heterogeneity=ETA0)):
        with pytest.raises(UnsupportedOptionError) as ex:
            call()
        assert ex.value.option == "sd" and "omit" in str(ex.value)
    cfg((XC, ETA0))                                                                                    # the twin


def test_the_tuple_is_canonicalised_by_parameter_name():
    a, b = cfg((XC, ETA)), cfg((ETA, XC))
    order = [m.parameter for m in a.heterogeneity.members]
    assert order == ["eta", "xc"] == [m.parameter for m in b.heterogeneity.members]
    sa, sb = simulate(P, heterogeneity=(XC, ETA), **BASE), simulate(P, heterogeneity=(ETA, XC), **BASE)
    assert sa.times.tobytes() == sb.times.tobytes()
    assert sa.provenance["members"] == [{"parameter": "eta", "sd": 0.2}, {"parameter": "xc", "sd": 0.2}]


def test_a_zero_spread_is_s_one_exactly_with_no_draw():
    arr = LS.draw_parameters(cfg((XC, ETA0)))
    assert np.all(arr["eta"] == P.eta) and arr["eta"].tobytes() == np.full(BASE["n"], P.eta).tobytes()
    single = LS.draw_parameters(cfg(XC))
    assert all(arr[k].tobytes() == single[k].tobytes() for k in single)


def test_each_parameter_has_its_own_stream():
    both, eta_only = LS.draw_parameters(cfg((XC, ETA))), LS.draw_parameters(cfg(ETA))
    assert both["eta"].tobytes() == eta_only["eta"].tobytes()                # adding xc leaves eta's draws alone


@pytest.mark.parametrize("wall", ["clamp", "reflect"])
def test_the_reduction_is_exact_under_both_walls(wall):
    t = simulate(P, heterogeneity=(XC, ETA0), wall=wall, record_ages=[50.0], **BASE)
    s = simulate(P, heterogeneity=XC, wall=wall, record_ages=[50.0], **BASE)
    for a in ("times", "ages", "events"):
        assert getattr(t, a).tobytes() == getattr(s, a).tobytes()
    assert t.paths.X.tobytes() == s.paths.X.tobytes()
    assert t.error.fp_reference["kind"] == "single" and t.error.ks_D == s.error.ks_D


def test_a_shared_stream_breaks_the_reduction():
    plant_t, plant_s = LS.draw_parameters_shared_stream(cfg((XC, ETA0))), LS.draw_parameters_shared_stream(cfg(XC))
    assert plant_t["xc"].tobytes() != plant_s["xc"].tobytes()                    # HR-11's plant: eta0 shifts xc's draw
    real_t, real_s = LS.draw_parameters(cfg((XC, ETA0))), LS.draw_parameters(cfg(XC))
    assert real_t["xc"].tobytes() == real_s["xc"].tobytes()                      # the twin: per-parameter streams


def test_more_than_three_spreads_has_no_fp_reference_and_needs_t_max():
    four = (XC, ETA, EPS, Heterogeneity(parameter="beta", sd=0.1))
    s = simulate(P, heterogeneity=four, **BASE)
    assert s.error.fp_reference["kind"] == "none" and not s.clean and s.error.ks_D is None
    assert any(w.code == "LANGEVIN_NO_FP_REFERENCE" for w in s.warnings)
    with pytest.raises(UnsupportedOptionError) as ex:
        simulate(P, heterogeneity=four, **{**BASE, "t_max": None})
    assert ex.value.option == "t_max"


def test_two_spreads_are_judged_against_the_tensor_reference():
    s = simulate(P, heterogeneity=(XC, EPS), **BASE)
    ref = s.error.fp_reference
    assert ref["kind"] == "tensor" and set(ref["M"]) == {"epsilon", "xc"} and ref["node_count"] == math.prod(
        ref["M"].values())
    assert ref["curve_check"]["verdict"] == "converged" and ref["curve_check"]["bar"] == pytest.approx(
        1.358 / math.sqrt(BASE["n"]) / 10)
    assert s.error.ks_D is not None and s.error.verdict in ("consistent at n", "inconsistent at n")


def test_a_tensor_product_above_the_budget_reads_no_fp_reference(monkeypatch):
    from sraging.engine import tensor_reference as TR
    monkeypatch.setattr(TR, "BUDGET", 20)                         # 5 x 5 nodes already exceed it
    s = simulate(P, heterogeneity=(XC, EPS), **BASE)
    assert s.error.fp_reference["kind"] == "none" and "budget" in s.error.fp_reference["reason"]
    assert not s.clean and s.error.verdict == "no FP reference"


def test_multi_spread_paths_are_recorded_not_checked():
    """D-413 (Naveh 2026-10-30): a tuple with two or more spreads records its paths, but no tensor damage density is
    built in 004, so the paths statement reads "not checked" for every age and the sample is not clean."""
    s = simulate(P, heterogeneity=(XC, EPS), record_ages=[50.0, 100.0], **BASE)
    X = s.paths.X
    assert np.array_equal(np.isnan(X), s.times[:, None] <= np.array([50.0, 100.0])[None, :])     # recorded
    assert s.error.paths_reference["kind"] == "none" and "not checked" in s.error.paths_reference["reason"]
    assert all(r["verdict"] == "not assessable" and r["ks_D_full"] is None for r in s.error.paths)
    assert not s.clean and any(w.code == "LANGEVIN_NO_FP_REFERENCE" and "not checked" in w.message
                               for w in s.warnings)
    one = simulate(P, heterogeneity=(XC, ETA0), record_ages=[50.0, 100.0], **BASE)               # the twin
    assert one.error.paths_reference["kind"] == "damage_density"
