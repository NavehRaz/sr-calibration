"""The FP damage density, the truth-free reference for paths (tasks T092; research R25; data-model "Damage density";
contracts/python-api.md §7): the march with every step stored (stepper._march_forward, store="all", read-only),
interpolated in tau, the conditional CDF among the living, the xc mixture on xc_k = xc s_k, the grid error against
(2N, dt/2), and its refusals."""
import os
import subprocess

import numpy as np
import pytest

from sraging import SRParams
from sraging.engine import density as DEN
from sraging.engine import quadrature as Q
from sraging.engine.grid import GridPolicy
from sraging.engine.inputs import AgeSegments, Heterogeneity, Segment
from sraging.errors import UnsupportedOptionError

pytestmark = pytest.mark.engine
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)
GRID = GridPolicy("fixed", N=200, dt=2.0 ** -9, tau_max=1.5, L=124.0)
AGES = np.array([40.0, 100.0, 140.0])


@pytest.fixture(scope="module")
def dd():
    return DEN.damage_density(P, AGES, grid=GRID)


def test_the_density_integrates_to_the_marchs_own_survival_and_the_cdf_reaches_one(dd):
    tau, S = dd.S_grid
    for j, a in enumerate(AGES):
        want = np.interp(a / GRID.L, tau, S)                            # the march's own S, interpolated in tau
        assert dd.S[j] == pytest.approx(want, rel=1e-12, abs=1e-15)
        assert dd.cdf_alive(P.xc, a) == pytest.approx(1.0, abs=1e-12)
        assert dd.cdf_alive(0.0, a) == 0.0
        x = np.linspace(0, P.xc, 50)
        F = dd.cdf_alive(x, a)
        assert np.all(np.diff(F) >= -1e-15)                             # a CDF


def test_the_hazard_leaves_the_conditional_law_unchanged(dd):
    dh = DEN.damage_density(SRParams(**{**P.as_dict(), "h_ext": 3e-3}), AGES, grid=GRID)
    x = np.linspace(0, 1, 101)
    for a in AGES:
        assert dh.cdf_alive(x, a).tobytes() == dd.cdf_alive(x, a).tobytes()


def test_an_xc_mixture_uses_xc_times_s_k():
    het = Heterogeneity(parameter="xc", sd=0.2, M=5, check=False)
    d = DEN.damage_density(P, AGES[:1], heterogeneity=het, grid=GRID)
    rule = Q.truncated_normal_rule(0.2, 5)
    assert d.nodes["xc"].tolist() == pytest.approx((P.xc * rule.s).tolist())
    assert d.cdf_alive(P.xc * rule.s[-1], AGES[0]) == pytest.approx(1.0, abs=1e-12)        # the largest threshold
    assert d.cdf_alive(P.xc * rule.s[0] * 0.999, AGES[0]) < 1.0


def test_the_grid_error_against_2n_and_dt_over_2_is_recorded(dd):
    assert len(dd.grid_error) == len(AGES) and all(0 <= g < 0.05 for g in dd.grid_error)
    assert dd.grid_error_grid == {"N": 2 * GRID.N, "dt": GRID.dt / 2}


def test_a_heterogeneous_node_count_comes_from_the_curve_check():
    d = DEN.damage_density(P, AGES[:2], heterogeneity=Heterogeneity(parameter="xc", sd=0.2), grid=GRID)
    c = d.curve_check
    assert c["verdict"] == "converged" and c["sup_dF"] <= c["tol"] == 1e-4 and d.M == c["M"]


def test_stepper_is_used_read_only_and_unchanged():
    import inspect
    src = inspect.getsource(DEN)
    assert "_march_forward(" in src and 'store="all"' in src.replace("'", '"')
    r = subprocess.run(["git", "-C", REPO, "diff", "--quiet", "HEAD", "--", "src/sraging/engine/stepper.py"])
    assert r.returncode == 0


@pytest.mark.parametrize("kw,option", [
    ({"heterogeneity": (Heterogeneity(parameter="xc", sd=0.2), Heterogeneity(parameter="eta", sd=0.2))},
     "heterogeneity"),
    ({"segments": AgeSegments(Segment(id="s", parameter="beta", value=1.5))}, "segments"),
    ({"heterogeneity": Heterogeneity(parameter="xc", definition="normal_frac_sd", v=0.2, M=7)}, "heterogeneity")])
def test_a_tuple_segments_or_normal_frac_sd_are_refused(kw, option):
    with pytest.raises(UnsupportedOptionError) as ex:
        DEN.damage_density(P, AGES, grid=GRID, **kw)
    assert ex.value.option == option


def test_a_negative_age_or_one_beyond_the_marchs_stop_is_refused():
    with pytest.raises(UnsupportedOptionError) as ex:
        DEN.damage_density(P, [-1.0, 50.0], grid=GRID)
    assert ex.value.option == "ages"
    with pytest.raises(UnsupportedOptionError, match="stop"):
        DEN.damage_density(P, [400.0], grid=GRID)                     # past the march's horizon and stop
    DEN.damage_density(P, [50.0], grid=GRID)                          # the twin
