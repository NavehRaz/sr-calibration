"""Shapes and labels (spec 003, T034; FR-020 to FR-024; SC-006): a planted positive and a planted negative per label
(constitution V). Fast: no likelihood."""
import math

import numpy as np
import pytest

from sraging.claims import shapes as S
from sraging.result import Claim

BOX = {"eta": (1e-12, 1e3), "beta": (1e-12, 1e3), "epsilon": (1e-12, 1e3), "kappa": (1e-3, 300.0)}


def claim(**kw):
    base = dict(parameter="beta", shape="upper_bound", lower=None, upper=2.0, point=0.5, level=0.95, n_components=1)
    base.update(kw)
    return Claim(**base)


def nodes(kappa_at_edge):
    k_edge = 300.0 if kappa_at_edge else 0.5
    z = lambda b, k: np.log([1e-4, b, 1e-3, k])          # noqa: E731
    return [(0.5, -100.0, z(0.5, 0.5)), (1.0, -100.8, z(1.0, 0.8)), (2.0, -101.9, z(2.0, k_edge)),
            (4.0, -110.0, z(4.0, k_edge))]


def test_box_limited_fires_when_the_cap_node_has_kappa_at_its_box():
    lab = S.box_limited_label(claim(), nodes(True), -100.0, BOX)
    assert lab["kind"] == "box_limited" and lab["edge"] == "kappa<=300" and "open above" in lab["text"]


def test_box_limited_is_silent_when_every_coordinate_is_clear():
    assert S.box_limited_label(claim(), nodes(False), -100.0, BOX) is None


def test_box_limited_ignores_lower_edges_the_physical_zeros():
    """sr-naveh implementation review 2: beta = 1e-12 (no removal), kappa = 1e-3 (saturated) at a setting node are the
    model's limits, not box artefacts; a lower end is never "open above"."""
    z = lambda b, k: np.log([1e-4, b, 1e-3, k])          # noqa: E731
    ns = [(1e-12, -100.0, z(1e-12, 1e-3)), (0.5, -100.5, z(0.5, 1e-3)), (2.0, -101.9, z(2.0, 1e-3)),
          (4.0, -110.0, z(4.0, 1e-3))]
    assert S.box_limited_label(claim(), ns, -100.0, BOX) is None
    closed = claim(shape="closed", lower=1e-11, upper=2.0)
    assert S.box_limited_label(closed, ns, -100.0, BOX) is None


def test_box_limited_reads_the_archive_point_that_widened_the_upper_end():
    """FR-020's "or archive point": the widened upper end's theta_qmax has kappa at 300."""
    lab = S.box_limited_label(claim(), nodes(False), -100.0, BOX, archive_theta=[1e-4, 3.0, 1e-3, 299.5])
    assert lab["edge"] == "kappa<=300" and "archive point" in lab["text"]
    assert S.box_limited_label(claim(), nodes(False), -100.0, BOX, archive_theta=[1e-4, 3.0, 1e-3, 5.0]) is None


def test_captured_nodes_carry_the_full_z():
    """sr-naveh implementation review 1: jax2 stores a node's free coordinates over fidx (estimator.py:1636); the
    capture rebuilds the full z as `_full_points` does, so z[3] is log kappa whichever parameter is profiled."""
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.compat.jax2 import estimator as JE

    class Obj:
        NP = 4

    pts = [[0.5, -101.0, [math.log(1e-4), math.log(1e-3), math.log(300.0)]],
           [1.0, -100.0, [math.log(2e-4), math.log(2e-3), math.log(0.5)]],
           [2.0, -101.5, [math.log(3e-4), math.log(3e-3), math.log(0.7)]]]
    with PJ.capture_nodes() as captured:
        JE._repair(Obj(), [2e-4, 1.0, 2e-3, 0.5], 1, [0, 2, 3], pts)
    rows = captured[1]
    assert [len(z) for _, _, z in rows] == [4, 4, 4]
    assert rows[0][2][1] == math.log(0.5) and rows[0][2][3] == math.log(300.0) and rows[0][2][0] == math.log(1e-4)
    ns = [(v, ll, np.asarray(z)) for v, ll, z in rows]
    assert S.box_limited_label(claim(shape="closed", lower=0.6, upper=2.0), ns, -100.0, BOX) is None
    # the kappa-300 node sets the LOWER end; move kappa 300 to the node that sets the upper end and it fires
    pts[2][2][2] = math.log(300.0)
    with PJ.capture_nodes() as captured:
        JE._repair(Obj(), [2e-4, 1.0, 2e-3, 0.5], 1, [0, 2, 3], pts)
    ns = [(v, ll, np.asarray(z)) for v, ll, z in captured[1]]
    assert S.box_limited_label(claim(shape="closed", lower=0.6, upper=2.0), ns, -100.0, BOX)["edge"] == "kappa<=300"


def test_negligible_fires_below_the_regime_floor_and_not_above():
    floors = {"sharp": 0.79, "mid": 1.53, "kappa-large": 1.56, "diffuse": 7.6}
    floor = 0.79 / math.sqrt(1000)
    lab = S.negligible_label(floor / 10, 1000, "sharp", floors)
    assert lab["kind"] == "negligible" and lab["log10_cap_over_floor"] == pytest.approx(-1.0)
    assert "0.79" in lab["text"] and "7.6" in lab["text"]
    assert S.negligible_label(floor * 2, 1000, "sharp", floors) is None


def test_saturated_fires_for_a_small_kappa_cap_only():
    assert S.saturated_label(1e-3, 1000)["kind"] == "saturated"
    assert "not no removal" in S.saturated_label(1e-3, 1000)["text"]
    assert S.saturated_label(1.0, 1000) is None


def test_a_widened_upper_side_reads_at_least():
    c = claim(widening={"claims_profile": {"shape": "upper_bound", "lower": None, "upper": 1.5},
                        "lower_moved_dex": 0.0, "upper_moved_dex": 0.125})
    assert S.widened_text(c).startswith("at least 2; edge not found")
    assert S.widened_text(claim()) is None


def test_n_eff_is_capped_at_2000():
    assert S.n_eff(10_000) == 2000 and S.n_eff(300) == 300
