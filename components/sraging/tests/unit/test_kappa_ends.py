"""θ at both ends of the κ profile in every FitResult (spec 006 FR-025, US6; research R6; data-model.md "κ ends"; tasks
T042). jax2 (J, H, W): the self-consistency extents the fit already keeps, "archive points within 1.92 nats of ll_best".
profile_v033: its κ walk's nodes, kept by a capture around `_repair` (v0.3.3 drops them after), "walk nodes within
1.92 nats of the walk's best". An end at the calibrator's κ box edge is labelled box-limited."""
import json
import math
import os

import numpy as np
import pytest

from sraging import Cohort, fit_cohort
from sraging.calibrators import profile_jax2 as PJ
from sraging.calibrators import profile_v033 as PV

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE = os.path.join(REPO, "examples", "results", "itp_c2016_het3_control_female_jax2.json")
SMALL = os.path.join(REPO, "tests", "unit", "data", "small_cohort.csv")
BOX = {"eta": (1e-12, 1e3), "beta": (1e-12, 1e3), "epsilon": (1e-12, 1e3), "kappa": (0.001, 300.0)}


def test_jax2s_ends_are_the_kept_extents_with_their_definition():
    diag = json.load(open(EXAMPLE))["diagnostics"]
    e = diag["self_consistency"]["extents"]["kappa"]
    ke = PJ.kappa_ends_jax2(diag, BOX)
    assert ke["definition"] == "archive points within 1.92 nats of ll_best"
    assert ke["kappa_lo"] == e["qmin"] and ke["kappa_hi"] == e["qmax"]
    assert ke["theta_lo"] == e["theta_qmin"] and ke["theta_hi"] == e["theta_qmax"]
    assert ke["box_limited_lo"] is True and ke["box_limited_hi"] is True          # 0.001 and 300: the box's edges
    inside = {**BOX, "kappa": (1e-4, 1e4)}                                          # the twin
    assert PJ.kappa_ends_jax2(diag, inside)["box_limited_lo"] is False
    assert PJ.kappa_ends_jax2({}, BOX) is None                                      # no extents: no ends


def test_v033s_ends_are_the_outermost_walk_nodes_within_192_nats():
    nodes = {"fidx": [0, 1, 2], "pts": [
        [0.001, -105.0, [-9.0, -3.0, -6.0]],          # 5 below the best: outside
        [0.01, -101.5, [-9.1, -3.1, -6.1]],           # 1.5 below: the lower end
        [0.1, -100.0, [-9.2, -3.2, -6.2]],            # the best
        [1.0, -101.9, [-9.3, -3.3, -6.3]],            # 1.9 below: the upper end
        [10.0, -102.0, [-9.4, -3.4, -6.4]]]}          # 2.0 below: outside (planted at the edge)
    ke = PV.kappa_ends_v033(nodes, BOX)
    assert ke["definition"] == "walk nodes within 1.92 nats of the walk's best"
    assert (ke["kappa_lo"], ke["kappa_hi"]) == (0.01, 1.0)
    assert ke["theta_lo"] == [math.exp(-9.1), math.exp(-3.1), math.exp(-6.1), 0.01]
    assert ke["box_limited_lo"] is False and ke["box_limited_hi"] is False
    nodes["pts"][0][1] = -100.5                                                        # the box floor within reach
    assert PV.kappa_ends_v033(nodes, BOX)["box_limited_lo"] is True
    assert PV.kappa_ends_v033({}, BOX) is None


@pytest.mark.slow
def test_a_v033_fit_carries_its_ends_and_the_capture_changes_no_byte():
    c = Cohort.from_csv(SMALL, delta=1.0, time_unit="days")
    r = fit_cohort(c, calibrator="profile_v033")
    ke = r.diagnostics["kappa_ends"]
    assert ke["definition"].startswith("walk nodes") and ke["kappa_lo"] <= r.theta_hat.kappa <= ke["kappa_hi"]
    from sraging.compat.v033 import estimator, fp_jax
    bare = estimator.fit(fp_jax, list(c.ages), list(c.events), delta=1.0, levels=PV.LEVELS, kappa_max=PV.KAPPA_MAX)
    assert [float(x) for x in bare["theta"]] == [r.theta_hat.eta, r.theta_hat.beta, r.theta_hat.epsilon,
                                                 r.theta_hat.kappa] and bare["loglik"] == r.loglik


@pytest.mark.slow
def test_a_jax2_fit_carries_its_ends():
    r = fit_cohort(Cohort.from_csv(SMALL, delta=1.0, time_unit="days"), calibrator="profile_jax2", combinations=False)
    ke = r.diagnostics["kappa_ends"]
    assert ke["definition"] == "archive points within 1.92 nats of ll_best" and len(ke["theta_lo"]) == 4
    assert ke["kappa_lo"] <= r.theta_hat.kappa <= ke["kappa_hi"]
