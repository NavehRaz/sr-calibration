"""Every byte check fails on a planted perturbation (T039; FR-020, SC-006; constitution V).

Each test runs the SAME comparison function the parity checks use (tools/parity_checks.py) on a
deliberately perturbed input and requires it to report a difference. Outcomes are recorded in
tests/acceptance/results/perturbation.json.
"""
import json
import math
import os

import numpy as np
import pytest

import oracle_v033 as O
from parity_checks import arrays_diff, bytes_diff, identity_diff, oracle_identity_view
from sraging.compat.v033 import estimator as port_est
from sraging.compat.v033 import fp_jax as port_fj

pytestmark = pytest.mark.parity
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "tests", "acceptance", "results", "perturbation.json")
COHORT = "pure-k0.01-p00-c0-s0"
_RESULTS = {}


def record(check, perturbation, detected, detail):
    _RESULTS[check] = {"perturbation": perturbation, "detected": bool(detected), "detail": detail[:3]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    prev.update(_RESULTS)
    with open(OUT, "w") as fh:
        json.dump(prev, fh, indent=1, sort_keys=True)


def loglik(fj, est, ages, events, delta):
    L, kw, _ = est.solver_grid(ages, events, delta)
    return fj.make_loglik(ages, events, L, N=kw["N"], dt=kw["dt"], tau_max=kw["tau_max"], Gtol=kw["Gtol"],
                          sampling_dt=kw.get("sampling_dt"))


def test_value_and_grad_checks_detect_a_perturbed_age(oracle_v033):
    obs = O.observation(COHORT)
    a, e = np.asarray(list(obs.ages), float), np.asarray(list(obs.events), int)
    a_pert = a.copy()
    k = int(np.argmax(e == 1))
    a_pert[k] = a_pert[k] * (1 + 1e-9)                      # one death's age, relative 1e-9
    z = np.log(np.array([O.library().truth(COHORT)[p] for p in ("eta", "beta", "epsilon", "kappa")]))
    pf, pvg = loglik(port_fj, port_est, a_pert, e, obs.delta)
    of, ovg = loglik(oracle_v033.fp_jax, oracle_v033.estimator, a, e, obs.delta)
    dv = bytes_diff(pf(z), of(z), "value")
    dg = bytes_diff(tuple(pvg(z)), tuple(ovg(z)), "value_and_grad")
    record("value_bytes", f"age[{k}] x (1+1e-9) in the port's input", dv, dv)
    record("grad_bytes", f"age[{k}] x (1+1e-9) in the port's input", dg, dg)
    assert dv and dg


def test_cohort_array_check_detects_swapped_events(oracle_v033):
    obs = O.observation(COHORT)
    a, e = np.asarray(list(obs.ages), float), np.asarray(list(obs.events), int)
    i, j = int(np.argmax(e == 1)), int(np.argmax(e == 0)) if np.any(e == 0) else None
    e2 = e.copy()
    if j is None:                                            # no censored record: flip one flag instead
        e2[i] = 0
        what = f"events[{i}] 1 -> 0"
    else:
        e2[i], e2[j] = e2[j], e2[i]
        what = f"events[{i}] <-> events[{j}]"
    d = arrays_diff(a, e2, a, e)
    record("cohort_arrays", what, d, d)
    assert d


def test_claims_identity_detects_a_one_ulp_bound(oracle_v033):
    rec = O.stored_reference_record(COHORT)
    ref = oracle_identity_view(rec["claim_set"])
    pert = json.loads(json.dumps(ref))
    c = next(c for c in pert["claims"] if c["upper"] is not None)
    c["upper"] = float(np.nextafter(c["upper"], math.inf))
    d = identity_diff(pert, ref)
    record("claims_identity", f"{c['parameter']}.upper + 1 ulp", d, d)
    assert d
    assert identity_diff(ref, json.loads(json.dumps(ref))) == []          # and no false alarm


def test_tree_diff_is_exact_on_signed_zero_and_nan():
    assert identity_diff({"x": -0.0}, {"x": 0.0}) != []
    assert identity_diff({"x": math.nan}, {"x": math.nan}) == []
    assert identity_diff({"x": 1}, {"x": 1.0}) != []


def test_censor_tie_check_reacts_to_one_fewer_tie():
    """At exactly 0.5 % of n the check fires; with one tied death fewer (0.5 % - 1/n) it does not."""
    from sraging.cohort import Cohort
    from sraging.warnings import evaluate
    n, tied = 1000, 5
    ages = [float(a) for a in np.linspace(10.0, 40.0, n - tied - 1).round()] + [50.0] * tied + [50.0]
    events = [1] * (n - 1) + [0]
    at = evaluate(Cohort.from_arrays(ages, events, delta=1.0), (), {})[0]
    ages2, events2 = list(ages), list(events)
    ages2[n - 2] = 41.0                                    # one tied death moved off the censoring age
    below = evaluate(Cohort.from_arrays(ages2, events2, delta=1.0), (), {})[0]
    fired_at = any(w.code == "CENSOR_TIE" for w in at)
    fired_below = any(w.code == "CENSOR_TIE" for w in below)
    record("censor_tie", "ties 5/1000 (fires) -> 4/1000 (must not fire)", fired_at and not fired_below,
           [f"at 0.5%: fired={fired_at}", f"at 0.4%: fired={fired_below}"])
    assert fired_at and not fired_below



def test_warnings_check_detects_a_planted_rule_change(oracle_v033, monkeypatch):
    """SC-006 / review N4: the warning-count comparison of test_warnings_parity (same `outcomes`) fails when the
    port's off-grid rule is planted wrong (tolerance 1e-6 -> any age is off-grid): on every quick-tier cohort
    with delta, the port's DELTA_GRID count then differs from the reference entry point's."""
    import sraging.warnings as W
    from parity_cases import QUICK_TIER
    from test_warnings_parity import outcomes
    real = W._delta_grid

    def planted(ages, delta):
        return real(ages + 0.5 * delta, delta) if delta else []       # every age moved off the grid
    monkeypatch.setattr(W, "_delta_grid", planted)
    differ = [cid for cid in QUICK_TIER if (o := outcomes(oracle_v033, cid))["port_grid"] != o["ref_grid"]]
    record("warnings_counts", "DELTA_GRID rule: every age moved half a delta off the grid",
           len(differ) > 0, differ)
    assert differ


@pytest.mark.slow
def test_claims_identity_detects_a_planted_change_in_the_port(oracle_v033, monkeypatch):
    """SC-006 / review N4, end to end: a real port fit with one planted change in the port's code path (the
    profile_ci table threshold 3.8415 -> 3.8416, fp_gof) no longer equals the stored v0.3.3 record that the
    unperturbed port matches byte for byte (claims.json). One fit of the fastest quick-tier cohort."""
    from sraging.adapters.observation import cohort_from_observation
    from sraging.calibrators import profile_v033
    from sraging.compat.v033 import fp_gof
    cid = "pure-k0.01-p02-c0-s1"
    monkeypatch.setitem(fp_gof._CHI2_1, 0.95, 3.8416)
    port = profile_v033.fit(cohort_from_observation(O.observation(cid)))
    d = identity_diff(port.identity_view(), oracle_identity_view(O.stored_reference_record(cid)["claim_set"]))
    record("claims_identity_end_to_end", "port fp_gof._CHI2_1[0.95] 3.8415 -> 3.8416, real fit of " + cid, d, d)
    assert d
