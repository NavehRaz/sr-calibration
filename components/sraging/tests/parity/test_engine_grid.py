"""GridPolicy("v033") equals the frozen oracle's `solver_grid` on spec 001's 40 cohorts (T006; research R9)."""
import numpy as np
import pytest

from engine_cases import arrays, cohort, panel_ids
from sraging.engine.grid import GridPolicy
from sraging.errors import UnsupportedOptionError

pytestmark = [pytest.mark.parity, pytest.mark.engine]


@pytest.mark.parametrize("cohort_id", panel_ids())
def test_v033_policy_equals_the_oracle_solver_grid(oracle_v033, cohort_id):
    t, e, d = arrays(cohort_id)
    L, kw, ceiling = oracle_v033.estimator.solver_grid(t, e, d)
    g = GridPolicy("v033").grid(cohort(cohort_id))
    assert (g.L, g.N, g.dt, g.tau_max, g.Gtol, g.dt_ceiling) == (L, kw["N"], kw["dt"], kw["tau_max"],
                                                                   kw["Gtol"], ceiling)
    assert g.nsteps == int(kw["tau_max"] / kw["dt"]) and g.uniform and g.policy == "v033"


def test_fixed_policy_uniform_and_graded():
    c = None
    g = GridPolicy("fixed", N=100, dt=1e-2, tau_max=3.0, L=4.0).grid(c)
    assert (g.N, g.dt, g.nsteps, g.uniform) == (100, 1e-2, 300, True)
    nodes = np.concatenate([[0.0], np.cumsum(np.full(10, 1e-4)), 1e-3 + np.arange(1, 50) * 1e-2])
    g2 = GridPolicy("fixed", N=100, nodes=nodes, tau_max=float(nodes[-1]), L=4.0).grid(c)
    assert not g2.uniform and g2.nsteps == nodes.size - 1 and g2.summary()["mesh"]["kind"] == "graded"


@pytest.mark.parametrize("kw,match", [
    (dict(), "needs exactly one of dt or nodes"),
    (dict(dt=1e-2, nodes=[0.0, 1.0, 2.0]), "needs exactly one of dt or nodes"),
    (dict(dt=1e-2, N=100, tau_max=3.0), "needs L"),
])
def test_fixed_policy_refusals(kw, match):
    with pytest.raises(UnsupportedOptionError, match=match):
        GridPolicy("fixed", **kw)


def test_unknown_policy_and_scheme_are_refused():
    with pytest.raises(UnsupportedOptionError, match="unknown grid policy"):
        GridPolicy("v034")
    with pytest.raises(UnsupportedOptionError, match="unknown scheme"):
        GridPolicy("fixed", N=10, dt=0.1, tau_max=1.0, L=1.0).grid(None, scheme="cn")
    with pytest.raises(UnsupportedOptionError, match="takes no"):
        GridPolicy("v033", N=900)
