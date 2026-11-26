"""Warning outcomes equal the reference entry point's, on the stored records (T048; SC-007).

No calibration: for each quick-tier cohort (and each named cohort), the stored v0.3.3 record's claims and
diagnostics and the cohort's observation are fed both to sraging's checks and to the reference functions
(`convergence_warnings`, `censoring_tie_warning`, the off-grid delta expression), and the counts per kind
must be equal. A summary with N is written to tests/acceptance/results/warnings_stored.json.
"""
import json
import os

import numpy as np
import pytest

import oracle_v033 as O
from parity_cases import QUICK_TIER
from sraging.adapters.observation import cohort_from_observation
from sraging.result import Claim
from sraging.warnings import evaluate

pytestmark = pytest.mark.parity
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "tests", "acceptance", "results", "warnings_stored.json")
NAMED = ["pure-k2-p04-c0-s0", "realfull-real-mice_M-c1-s1", "pure-real-Sweden_F_1910_homo-c0-s0",
         "pure-real-Denmark_F_1900_hetro-c0-s0", "censor-k0.0316-p05-c0-s1", "het-k0.01-p01-c1-s0",
         "pure-k0.01-p15-c1-s0", "pure-k0.01-p15-c1-s1", "censor-k0.01-p05-c0-s0", "censor-k0.01-p05-c1-s1"]
_SUMMARY = {}


def outcomes(oracle, cid):
    rec = O.stored_reference_record(cid)["claim_set"]
    obs = O.observation(cid)
    from srbench.contract.types import Claim as SClaim, ClaimSet
    cs = ClaimSet(cohort_id=cid, method_name="mle_profile_jax", diagnostics=rec.get("diagnostics") or {},
                  claims=tuple(SClaim(**{k: c.get(k) for k in ("parameter", "shape", "lower", "upper", "point")
                                         if k in c or k in ("parameter", "shape")}) for c in rec["claims"]))
    ages, events = np.asarray(obs.ages, float), np.asarray(obs.events, int)
    ref_basin = len(oracle.convergence_warnings(cs))
    ref_tie = int(oracle.censoring_tie_warning(ages, events, obs.delta) is not None)
    ref_grid = int(bool(obs.delta) and oracle.delta_grid_fraction(ages, obs.delta) > 0)
    claims = tuple(Claim(c["parameter"], c["shape"], c.get("lower"), c.get("upper"), c.get("point"),
                         c.get("level", 0.95), c.get("n_components", 1)) for c in rec["claims"])
    ws, _ = evaluate(cohort_from_observation(obs), claims, rec.get("diagnostics") or {})
    port = {k: sum(w.code == k for w in ws) for k in ("BASIN_GAIN", "BASIN_OUTSIDE", "CENSOR_TIE", "DELTA_GRID")}
    return {"port_basin": port["BASIN_GAIN"] + port["BASIN_OUTSIDE"], "ref_basin": ref_basin,
            "port_tie": port["CENSOR_TIE"], "ref_tie": ref_tie,
            "port_grid": port["DELTA_GRID"], "ref_grid": ref_grid}


@pytest.mark.parametrize("cid", QUICK_TIER + NAMED)
def test_warning_counts_equal_the_reference(oracle_v033, cid):
    o = outcomes(oracle_v033, cid)
    _SUMMARY[cid] = o
    assert (o["port_basin"], o["port_tie"], o["port_grid"]) == (o["ref_basin"], o["ref_tie"], o["ref_grid"])


def test_write_summary(oracle_v033):
    """Counts per kind with N (runs last in this module)."""
    if len(_SUMMARY) < len(QUICK_TIER) + len(NAMED):
        pytest.skip("per-cohort tests did not all run")
    def tally(ids):
        return {"N": len(ids), **{k: sum(_SUMMARY[c][k] for c in ids) for k in next(iter(_SUMMARY.values()))}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"source": "stored v0.3.3 records (run mle_profile_jax-full), no calibration",
               "quick": tally(QUICK_TIER), "named": tally(NAMED), "per_cohort": _SUMMARY},
              open(OUT, "w"), indent=1, sort_keys=True)
