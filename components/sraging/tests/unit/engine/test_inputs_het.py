"""The extended Heterogeneity (tasks T011; data-model "Heterogeneity"; contracts/python-api.md §1; research R2): two
definitions by name, the truncated normal by default, and every refusal of the data model beside a twin that is
accepted. Each refusal is UnsupportedOptionError naming its option."""
import math

import numpy as np
import pytest

from sraging import Cohort
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy
from sraging.engine.inputs import AgeSegments, Heterogeneity, Segment
from sraging.errors import UnsupportedOptionError

pytestmark = pytest.mark.engine


def refused(option, match=None, **kw):
    with pytest.raises(UnsupportedOptionError, match=match) as ex:
        Heterogeneity(**kw)
    assert ex.value.option == option, (ex.value.option, str(ex.value))
    return str(ex.value)


def test_the_four_calls_of_the_contract():
    h = Heterogeneity(parameter="xc", sd=0.2)
    assert (h.parameter, h.definition, h.sd, h.v, h.M, h.check) == ("xc", "truncated_normal", 0.2, None, "auto", True)
    h = Heterogeneity(parameter="xc", sd=0.2, M=38)
    assert (h.M, h.check) == (38, True)
    h = Heterogeneity(parameter="xc", sd=0.2, M=11, check=False)
    assert (h.M, h.check) == (11, False)
    h = Heterogeneity(parameter="xc", definition="normal_frac_sd", v=0.2, M=11)
    assert (h.definition, h.v, h.sd, h.M) == ("normal_frac_sd", 0.2, None, 11)


def test_v_without_naming_the_definition_is_refused_naming_both():
    msg = refused("v", parameter="xc", v=0.2, M=7)
    assert "name the definition" in msg and "normal_frac_sd" in msg and "truncated_normal" in msg
    Heterogeneity(parameter="xc", definition="normal_frac_sd", v=0.2, M=7)                       # twin


def test_sd_with_normal_frac_sd_or_v_with_truncated_normal_is_refused():
    refused("sd", parameter="xc", definition="normal_frac_sd", sd=0.2, M=7)
    refused("v", parameter="xc", definition="truncated_normal", v=0.2)
    Heterogeneity(parameter="xc", definition="truncated_normal", sd=0.2)                         # twins
    Heterogeneity(parameter="xc", definition="normal_frac_sd", v=0.2, M=7)


def test_both_sd_and_v_or_neither_is_refused():
    refused("sd", parameter="xc", sd=0.2, v=0.2)
    refused("sd", parameter="xc", definition="normal_frac_sd", sd=0.2, v=0.2, M=7)
    refused("sd", parameter="xc")
    refused("v", parameter="xc", definition="normal_frac_sd", M=7)
    Heterogeneity(parameter="xc", sd=0.2)                                                        # twin


@pytest.mark.parametrize("M", [193, 1000, 0, -3, 11.0, 11.5, "11", None, True])
def test_an_m_outside_one_to_the_cap_or_not_an_integer_is_refused(M):
    refused("M", parameter="xc", sd=0.2, M=M, check=False)


@pytest.mark.parametrize("M", [1, 5, 128, 192])
def test_an_integer_m_in_one_to_the_cap_is_accepted(M):
    assert Heterogeneity(parameter="xc", sd=0.2, M=M, check=False).M == M                        # twins


def test_a_checked_fixed_m_above_128_is_refused():
    """A checked fixed M is compared with the smallest rung >= ceil(1.5 M); above 128 that rung would pass the cap."""
    msg = refused("M", parameter="xc", sd=0.2, M=129)
    assert "check=False" in msg and "192" in msg
    Heterogeneity(parameter="xc", sd=0.2, M=128)                                                 # twins
    Heterogeneity(parameter="xc", sd=0.2, M=129, check=False)


def test_check_false_with_automatic_m_is_refused():
    refused("check", parameter="xc", sd=0.2, check=False)
    refused("check", parameter="xc", sd=0.2, M="auto", check=False)
    Heterogeneity(parameter="xc", sd=0.2, M=11, check=False)                                     # twin


def test_xc_together_with_age_segments_is_refused():
    c = Cohort.from_arrays(np.arange(1.0, 41.0), np.ones(40, int), delta=1.0)
    seg = AgeSegments(Segment(id="s", parameter="beta", value=1.5, window=(10.0, 20.0)))
    grid = GridPolicy("fixed", N=60, dt=2.0 ** -6, tau_max=2.5, L=30.0)
    with pytest.raises(UnsupportedOptionError) as ex:
        Evaluator(c, grid=grid, segments=seg, heterogeneity=Heterogeneity(parameter="xc", sd=0.2))
    assert ex.value.option == "heterogeneity" and "segments" in str(ex.value)
    Evaluator(c, grid=grid, segments=seg, heterogeneity=Heterogeneity(parameter="beta", sd=0.2))   # twin


def test_a_tuple_parameter_or_a_tuple_of_heterogeneities_on_the_fp_path_is_refused():
    msg = refused("parameter", parameter=("xc", "eta"), sd=0.2)
    assert "Langevin" in msg
    c = Cohort.from_arrays(np.arange(1.0, 41.0), np.ones(40, int), delta=1.0)
    hs = (Heterogeneity(parameter="xc", sd=0.2), Heterogeneity(parameter="eta", sd=0.2))
    with pytest.raises(UnsupportedOptionError) as ex:
        Evaluator(c, heterogeneity=hs)
    assert ex.value.option == "heterogeneity" and "Langevin" in str(ex.value)
    Evaluator(c, heterogeneity=hs[0])                                                            # twin


@pytest.mark.parametrize("sd", [0.0, -0.2, 1.0000001, 2.0, math.inf, math.nan])
def test_an_sd_outside_zero_to_one_or_not_finite_is_refused(sd):
    refused("sd", parameter="xc", sd=sd)


@pytest.mark.parametrize("sd", [1e-6, 0.2, 0.5, 1.0])
def test_an_sd_in_zero_to_one_is_accepted(sd):
    assert Heterogeneity(parameter="xc", sd=sd).sd == sd                                         # twins


def test_automatic_m_with_normal_frac_sd_is_refused():
    msg = refused("M", parameter="xc", definition="normal_frac_sd", v=0.2, M="auto")
    assert "choose_M" in msg
    refused("M", parameter="xc", definition="normal_frac_sd", v=0.2)                             # its M has no default
    Heterogeneity(parameter="xc", definition="normal_frac_sd", v=0.2, M=7)                       # twin


def test_an_unknown_definition_or_parameter_is_refused():
    msg = refused("definition", parameter="xc", definition="lognormal", sd=0.2)
    assert "truncated_normal" in msg and "normal_frac_sd" in msg
    refused("parameter", parameter="h_ext", sd=0.2)
    for p in ("eta", "beta", "epsilon", "kappa", "xc"):                                          # twins
        Heterogeneity(parameter=p, sd=0.2)


def test_sd_above_the_measured_range_carries_a_warning_and_inside_it_none():
    h = Heterogeneity(parameter="xc", sd=0.6)
    assert [w.code for w in h.warnings] == ["HET_OUTSIDE_MEASURED_RANGE"]
    assert h.warnings[0].data == {"sd": 0.6, "lo": 0.1, "hi": 0.5} and "0.6" in h.warnings[0].message
    assert Heterogeneity(parameter="xc", sd=0.5).warnings == ()                                  # twins
    assert Heterogeneity(parameter="xc", definition="normal_frac_sd", v=0.6, M=7).warnings == ()


def test_check_must_be_a_bool():
    refused("check", parameter="xc", sd=0.2, M=11, check="no")
    refused("check", parameter="xc", sd=0.2, M=11, check=0)


def test_the_normal_frac_sd_v_keeps_spec_002s_domain():
    with pytest.raises(ValueError, match="v must be finite"):
        Heterogeneity(parameter="xc", definition="normal_frac_sd", v=-0.1, M=7)
    assert Heterogeneity(parameter="xc", definition="normal_frac_sd", v=0.0, M=7).v == 0.0      # twin
