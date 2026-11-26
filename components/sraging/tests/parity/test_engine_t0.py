"""T0 (spec SC-001, FR-021; research R1): with every feature off, numerics "v033", grid "v033" and scheme
"bdf2", the engine IS v0.3.3 -- the same compiled program (canonical text) and the same value and gradient
bytes as spec 001's port (`sraging.compat.v033.fp_jax`, itself byte-identical to the frozen oracle).

Fast subset: 6 cohorts of spec 001's panel (2 quick-tier, the 2 dt-ceiling cohorts -- one of which is
checkpointed --, 2 basin cohorts) x their 6 frozen theta points. `tools/acceptance_002.py t0` runs all 40.
"""
import functools

import jax.numpy as jnp
import numpy as np
import pytest

from engine_cases import arrays, cohort, theta_points
from parity_checks import bytes_diff
from sraging.compat.v033 import estimator as port_est
from sraging.compat.v033 import fp_jax as V
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy
from sraging.engine.program import canonical_text

pytestmark = [pytest.mark.parity, pytest.mark.engine]
SUBSET = ["pure-k0.01-p00-c0-s0", "pure-k0.01-p05-c1-s1", "pure-k2-p04-c0-s0", "realfull-real-mice_M-c1-s1",
          "pure-real-Denmark_F_1900_hetro-c0-s0", "censor-k0.0316-p05-c0-s1"]
PARAMS = ("eta", "beta", "epsilon", "kappa")


@functools.lru_cache(maxsize=None)
def built(cohort_id, planted=False):
    t, e, d = arrays(cohort_id)
    L, kw, _ = port_est.solver_grid(t, e, d)
    vf, vg = V.make_loglik(t, e, L, N=kw["N"], dt=kw["dt"], tau_max=kw["tau_max"], Gtol=kw["Gtol"],
                           sampling_dt=kw.get("sampling_dt"))
    if not planted:
        grid = GridPolicy("v033")
    else:                                   # the planted defect: L off by one ulp, everything else equal
        grid = GridPolicy("fixed", N=kw["N"], dt=kw["dt"], tau_max=kw["tau_max"], Gtol=kw["Gtol"],
                          L=float(np.nextafter(L, np.inf)))
    ev = Evaluator(cohort(cohort_id), numerics="v033", grid=grid, scheme="bdf2")
    return (vf, vg), ev


def _cases():
    return [pytest.param(c, i, id=f"{c}:{i}") for c in SUBSET for i in range(6)]


@pytest.mark.parametrize("cohort_id", SUBSET)
def test_same_compiled_program(oracle_v033, cohort_id):
    (vf, vg), ev = built(cohort_id)
    z = jnp.zeros(4)
    assert canonical_text(vf.lower(z)) == ev.program_text("value")
    assert canonical_text(vg.lower(z)) == ev.program_text("value_and_grad")


@pytest.mark.parametrize("cohort_id,i", _cases())
def test_value_and_gradient_bytes(oracle_v033, cohort_id, i):
    label, theta = theta_points(cohort_id)[i]
    (vf, vg), ev = built(cohort_id)
    z = np.log(np.array([theta[p] for p in PARAMS], float))
    ef, evg = ev.objective()
    assert bytes_diff(ef(z), vf(z), "value") == []
    assert bytes_diff(evg(z), vg(z), "value_and_grad") == []


def test_a_one_ulp_change_is_caught(oracle_v033):
    """Constitution V: the comparison must fail on the defect it guards against. L one ulp larger changes
    a closed-over constant: the program differs and the bytes differ."""
    cid = SUBSET[0]
    (vf, vg), _ = built(cid)
    _, ev = built(cid, planted=True)
    z = np.log(np.array([theta_points(cid)[0][1][p] for p in PARAMS], float))
    assert canonical_text(vf.lower(jnp.zeros(4))) != ev.program_text("value")
    assert bytes_diff(ev.objective()[0](z), vf(z), "value") != []
