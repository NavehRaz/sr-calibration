"""Value and value+gradient bytes of the jax2 port against the frozen jax2 1.0.0 oracle, side by side in one
process (spec 003, T013; SC-001; research R4).

Every case builds the objective exactly as `estimator.fit` does on both sides (tools/jax2_cases.py) and compares
value with value and value+gradient with value+gradient (never across). The full panel (40 cohorts x 6 theta per
name, _h at two hazard rates, the C. elegans points) is the `bytes` acceptance phase; this file runs a FAST subset
in the suite: 3 cohorts x 6 theta for profile_jax2, and 1 cohort x 6 theta x 2 h for profile_jax2_h.
NOT APPLICABLE (never a pass) away from this Mac or the pinned environment.
"""
import os
import sys

import numpy as np
import pytest

from conftest import not_applicable

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
import jax2_cases as C  # noqa: E402
from parity_checks import bytes_diff  # noqa: E402

FAST = ["pure-k0.01-p00-c0-s0", "pure-k0.01-p02-c1-s1", "pure-k0.01-p05-c0-s0"]


@pytest.fixture(scope="module")
def sides():
    import oracle_jax2 as OJ
    problems = OJ.preflight()
    if problems:
        not_applicable("; ".join(problems))
    return C.port_modules(), C.oracle_modules(), OJ


def compare(port_obj, orc_obj, z):
    out = []
    a, b = port_obj.f(z), orc_obj.f(z)
    out += bytes_diff(np.asarray(a), np.asarray(b), "value")
    (va, ga), (vb, gb) = port_obj.vg(z), orc_obj.vg(z)
    out += bytes_diff(np.asarray(va), np.asarray(vb), "vg.value") + bytes_diff(np.asarray(ga), np.asarray(gb), "vg.grad")
    return out


@pytest.mark.parametrize("cid", FAST)
def test_jax2_value_and_gradient_bytes(sides, cid):
    port, orc, OJ = sides
    obs = OJ.observation(cid)
    po, pi = C.objective(port, obs, "jax2")
    oo, oi = C.objective(orc, obs, "jax2")
    assert pi == oi                                     # the same grid rule on both sides (N, dt, mesh, box)
    for label, th in C.theta_points(cid):
        assert compare(po, oo, C.z_of(th)) == [], (cid, label)


def test_jax2_h_value_and_gradient_bytes_at_two_hazard_rates(sides):
    port, orc, OJ = sides
    cid = FAST[1]
    obs = OJ.observation(cid)
    po, pi = C.objective(port, obs, "jax2_h")
    oo, oi = C.objective(orc, obs, "jax2_h")
    assert pi == oi
    for label, th in C.theta_points(cid):
        for hl, h in C.h_points(pi):
            assert compare(po, oo, C.z_of(th, h)) == [], (cid, label, hl)


def test_a_one_ulp_change_is_caught(sides):
    """Constitution V: the byte comparison fails on a planted one-ulp perturbation of a value."""
    port, orc, OJ = sides
    obs = OJ.observation(FAST[0])
    po, _ = C.objective(port, obs, "jax2")
    z = C.z_of(C.theta_points(FAST[0])[0][1])
    v = np.asarray(po.f(z))
    assert bytes_diff(np.nextafter(v, np.inf), v, "value") != []
