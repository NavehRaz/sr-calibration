"""Log-likelihood value and gradient bytes equal the oracle's (T037; SC-001, FR-021, research R6).

For each panel cohort and each of its 6 theta points, the port's and the oracle's `make_loglik` are built
from their own `solver_grid` (whose outputs must also be identical) and compared bit for bit:
value(z) vs value(z) and value_and_grad(z) vs value_and_grad(z) -- never across, because they are two
compiled programs that may differ in the last ulp (fp_jax.py:324-333).

Points 2-5 come from the pilot (tests/acceptance/panel_v1.toml). Before the pilot has run, only the
planted truth and the box corner are tested, and the missing points are reported NOT APPLICABLE.
"""
import functools
import os
import tomllib

import numpy as np
import pytest

import oracle_v033 as O
from conftest import not_applicable
from parity_checks import bytes_diff
from sraging.compat.v033 import estimator as port_est
from sraging.compat.v033 import fp_jax as port_fj

pytestmark = pytest.mark.parity
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANEL = os.path.join(REPO, "tests", "acceptance", "panel_v1.toml")
CORNER = {"eta": 1e-12, "beta": 1e3, "epsilon": 1e3, "kappa": 1e-3}
PARAMS = ("eta", "beta", "epsilon", "kappa")


def panel_points():
    """[(cohort_id, label, theta dict)] from the frozen panel, or the pre-pilot subset."""
    if os.path.exists(PANEL):
        panel = tomllib.load(open(PANEL, "rb"))
        return [(t["cohort_id"], p["label"], {k: p[k] for k in PARAMS})
                for t in panel["theta"] for p in t["points"]], True
    return None, False


def _cases():
    pts, frozen = panel_points()
    if frozen:
        return [pytest.param(c, l, th, id=f"{c}:{l}") for c, l, th in pts]
    # pre-pilot: quick-tier ids are known statically; the truth is read at run time
    from parity_cases import QUICK_TIER
    return ([pytest.param(c, "truth", None, id=f"{c}:truth") for c in QUICK_TIER]
            + [pytest.param(c, "corner", CORNER, id=f"{c}:corner") for c in QUICK_TIER])


@functools.lru_cache(maxsize=None)
def built(cohort_id):
    o = O.load()
    obs = O.observation(cohort_id)
    t, e = list(obs.ages), list(obs.events)
    delta = obs.delta if obs.delta else None
    pg = port_est.solver_grid(np.asarray(t, float), np.asarray(e, int), delta)
    og = o.estimator.solver_grid(np.asarray(t, float), np.asarray(e, int), delta)

    def make(fj, grid):
        L, kw, _ = grid
        return fj.make_loglik(np.asarray(t, float), np.asarray(e, int), L, N=kw["N"], dt=kw["dt"],
                              tau_max=kw["tau_max"], Gtol=kw["Gtol"], sampling_dt=kw.get("sampling_dt"))
    return pg, og, make(port_fj, pg), make(o.fp_jax, og)


@pytest.mark.parametrize("cohort_id,label,theta", _cases())
def test_value_and_grad_bytes(oracle_v033, cohort_id, label, theta):
    if theta is None:
        theta = O.library().truth(cohort_id)
    pg, og, (pf, pvg), (of, ovg) = built(cohort_id)
    assert repr(pg) == repr(og), "solver_grid differs"
    z = np.log(np.array([theta[p] for p in PARAMS], float))
    assert bytes_diff(pf(z), of(z), "value") == []
    pv, pgrad = pvg(z)
    ov, ograd = ovg(z)
    assert bytes_diff((pv, pgrad), (ov, ograd), "value_and_grad") == []


def test_pilot_points_present():
    if not os.path.exists(PANEL):
        not_applicable("theta points 2-5 (optimum, seed, floor, kappa_max) come from the pilot; "
                       "run tools/acceptance.py pilot and commit tests/acceptance/panel_v1.toml")
    labels = {p["label"] for t in tomllib.load(open(PANEL, "rb"))["theta"] for p in t["points"]}
    assert labels == {"truth", "optimum", "seed", "floor", "kmax", "corner"}
