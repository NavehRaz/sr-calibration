"""Combination claims and their widening by jax2's own rule (spec 003, T027; FR-015; research R9). Fast: synthetic
archives, no likelihood."""
import math

import numpy as np

from sraging.claims import combinations as K
from sraging.claims import widening as Wd

BOX = {"eta": (1e-12, 1e3), "beta": (1e-12, 1e3), "epsilon": (1e-12, 1e3), "kappa": (1e-3, 300.0)}
KEY = "eta_over_eps"
IMG = K.box_image(KEY, BOX)


def z(eta, beta=0.1, eps=1e-3, kappa=0.5):
    return np.log([eta, beta, eps, kappa])


def walk_ci(lo, hi, verdict="identified"):
    return {"ci": {"ci_lo": lo, "ci_hi": hi, "verdict": verdict}, "box_image": IMG}


def test_the_claim_uses_the_ports_mapping():
    kw = Wd.claim_from_walk(KEY, walk_ci(0.5, 2.0), 1.0)
    assert kw == {"parameter": KEY, "shape": "closed", "level": 0.95, "lower": 0.5, "upper": 2.0, "point": 1.0}
    kw = Wd.claim_from_walk(KEY, walk_ci(float("nan"), 2.0, "one-sided:<"), 1.0)
    assert kw["shape"] == "upper_bound" and "lower" not in kw


def test_an_archive_point_beyond_the_edge_widens_the_claim_by_hand():
    ll_hat = -100.0
    kw = Wd.claim_from_walk(KEY, walk_ci(0.5, 2.0), 1.0)
    Z = np.array([z(1.0e-3), z(3.0e-3), z(0.4e-3)])         # g = 1, 3, 0.4 (eps = 1e-3)
    LL = np.array([ll_hat, ll_hat - 1.0, ll_hat - 5.0])     # the g = 3 point is within thr/2; g = 0.4 is not
    new, rec, sc = Wd.widen(KEY, kw, Z, LL, ll_hat, BOX, IMG)
    assert rec["changed"] and math.isclose(new["upper"], 3.0, rel_tol=1e-12) and new["lower"] == 0.5
    assert math.isclose(rec["upper_moved_dex"], math.log10(3.0 / 2.0), rel_tol=1e-12)


def test_a_side_within_0_01_dex_of_the_box_image_opens():
    ll_hat = -100.0
    kw = Wd.claim_from_walk(KEY, walk_ci(0.5, 2.0), 1.0)
    g_hi = IMG[1] / 10 ** 0.005                            # inside 0.01 dex of the image's upper edge
    Z = np.array([np.log([1e3, 0.1, 1e3 / g_hi, 0.5])])
    new, rec, _ = Wd.widen(KEY, kw, Z, np.array([ll_hat - 0.5]), ll_hat, BOX, IMG)
    assert new["shape"] == "lower_bound" and "upper" in rec["opened"]
