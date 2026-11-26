"""The combinations' algebra (spec 003, T025; FR-014; research R8, R9): log g and its gradient, the box image, the
projection, and the definitions themselves (constitution X: r-bar already contains beta)."""
import itertools
import math

import numpy as np
import pytest

from sraging.claims import combinations as K

BOX = {"eta": (1e-12, 1e3), "beta": (1e-12, 1e3), "epsilon": (1e-12, 1e3), "kappa": (1e-3, 300.0)}
TH = {"eta": 3.6e-5, "beta": 0.12, "epsilon": 2.4e-3, "kappa": 0.7}
Z = np.log([TH["eta"], TH["beta"], TH["epsilon"], TH["kappa"]])


def test_the_definitions():
    f = 1 - TH["kappa"] * math.log(1 + 1 / TH["kappa"])
    rbar = TH["beta"] * f
    want = {"eta_over_eps": TH["eta"] / TH["epsilon"], "sqrt_eta_over_eps": math.sqrt(TH["eta"]) / TH["epsilon"],
            "rbar": rbar, "rbar_over_eta": rbar / TH["eta"], "beta_over_kp1": TH["beta"] / (1 + TH["kappa"]),
            "beta_over_eta_kp1": TH["beta"] / (TH["eta"] * (TH["kappa"] + 1))}
    for k in K.KEYS:
        assert K.value(k, TH) == pytest.approx(want[k], rel=1e-13), k


@pytest.mark.parametrize("key", K.KEYS)
def test_the_gradient_agrees_with_central_differences(key):
    g = K.grad_log_g(key, Z)
    for i in range(4):
        e = np.zeros(4)
        e[i] = 1e-6
        fd = (K.log_g(key, Z + e) - K.log_g(key, Z - e)) / 2e-6
        assert g[i] == pytest.approx(fd, rel=1e-6, abs=1e-9), (key, i)


@pytest.mark.parametrize("key", K.KEYS)
def test_the_box_image_is_the_range_over_a_dense_grid(key):
    lo, hi = K.box_image(key, BOX)
    vals = []
    for pt in itertools.product(*[np.geomspace(*BOX[p], 7) for p in ("eta", "beta", "epsilon", "kappa")]):
        vals.append(math.exp(K.log_g(key, np.log(pt))))
    assert min(vals) >= lo * (1 - 1e-12) and max(vals) <= hi * (1 + 1e-12)
    assert min(vals) == pytest.approx(lo, rel=1e-9) and max(vals) == pytest.approx(hi, rel=1e-9)


@pytest.mark.parametrize("key", K.KEYS)
def test_the_projection_lands_on_the_constraint(key):
    c = K.log_g(key, Z) + 0.3
    assert K.log_g(key, K.project(key, Z, c)) == pytest.approx(c, abs=1e-12)


def test_the_hypothesis_order_and_the_ridge_invariants():
    assert K.HYPOTHESIS_ORDER[:2] == ("eta_over_eps", "rbar_over_eta")
    for key in K.RIDGE_INVARIANT:                      # (eta, beta, eps) -> lambda (eta, beta, eps) leaves them unchanged
        zl = Z.copy()
        zl[[0, 1, 2]] += math.log(7.0)
        assert K.log_g(key, zl) == pytest.approx(K.log_g(key, Z), abs=1e-12)
    zl = Z.copy()
    zl[[0, 1, 2]] += math.log(7.0)
    assert abs(K.log_g("sqrt_eta_over_eps", zl) - K.log_g("sqrt_eta_over_eps", Z)) > 0.5
