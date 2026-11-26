"""v0.3.3's point from its record (spec 003g research R6; tasks T025): the claim points when all four exist; else kappa
rebuilt by the basin scope's 1-D maximisation, kept iff it reproduces the stored log-likelihood within 0.5 nats; a
rebuild that does not is no point ("not shown"), beside its kept twin."""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

import v033_points as VP  # noqa: E402

SOLVER = {"N": 300, "dt": 0.004, "tau_max": 2.0, "Gtol": 1e-11, "sampling_dt": 1.0}


def cohort():
    rng = np.random.default_rng(3)
    return np.round(rng.weibull(5.0, 120) * 50.0), np.ones(120, int)


def claims(kappa_point):
    pts = {"eta": 0.004, "beta": 0.4, "epsilon": 0.03, "kappa": kappa_point}
    return [{"parameter": p, "shape": "closed" if v is not None else "none", "point": v} for p, v in pts.items()]


def test_four_claim_points_are_the_point():
    a, e = cohort()
    r = VP.recover(claims(0.01), SOLVER, -1e9, a, e)
    assert r["source"] == "claims" and r["kept"] and r["point"] == [0.004, 0.4, 0.03, 0.01]


def test_a_rebuilt_kappa_is_kept_when_it_reproduces_the_stored_value_and_dropped_when_not():
    a, e = cohort()
    probe = VP.recover(claims(None), SOLVER, 0.0, a, e)            # learn the rebuilt value at these points
    kept = VP.recover(claims(None), SOLVER, probe["value"] + 0.4, a, e)       # stored 0.4 above: dll -0.4, kept
    dropped = VP.recover(claims(None), SOLVER, probe["value"] + 0.6, a, e)    # stored 0.6 above: dll -0.6, not kept
    assert kept["source"] == "recon" and kept["kept"] and kept["point"][3] == probe["kappa"]
    assert dropped["source"] == "recon_failed" and not dropped["kept"] and dropped["point"] is None


def test_no_eta_beta_eps_points_is_no_point():
    a, e = cohort()
    c = claims(0.01)
    c[0]["point"] = None
    assert VP.recover(c, SOLVER, 0.0, a, e) == {"point": None, "source": "no_points", "kept": False}
