"""Which combinations a jax2 fit computes (FR-014 as decided by Naveh, 2026-10-06, from walker step 1's measured cost):
the frozen lead group by default, all six with `combinations="all"`, none with False; anything else is refused. The
lead claims are widened over the archive as it stands after the lead walks, so `"all"` adds four claims and never
changes the default two; a later walk's point beyond a lead end is reported, as for the per-parameter claims (FR-019).
Fast: the walker, the widening and the port's extents are replaced by fakes on a fake objective."""
import math

import numpy as np
import pytest

from sraging.calibrators import profile_jax2 as PJ
from sraging.claims import combinations as K
from sraging.claims import walker as W
from sraging.claims import widening as Wd

LEAD = ("eta_over_eps", "rbar_over_eta")


class FakeObjective:
    NP = 4
    params = ("eta", "beta", "epsilon", "kappa")
    box = {"eta": (1e-12, 1e3), "beta": (1e-12, 1e3), "epsilon": (1e-12, 1e3), "kappa": (1e-3, 300.0)}
    seed_theta = (1e-3, 0.2, 4e-3, 7.0)
    L = 50.0

    def __init__(self):
        self._vg_cache, self._f_cache = {}, {}

    def evaluate(self, n):
        for _ in range(n):
            z = np.random.default_rng(len(self._vg_cache)).normal(size=4)
            self._vg_cache[z.tobytes()] = (-100.0 - len(self._vg_cache), None)


@pytest.fixture
def fake(monkeypatch):
    obj = FakeObjective()
    obj.evaluate(5)                                   # the fit's own archive
    walked = []

    def walk(o, key, z_hat, ll_hat, **kw):
        walked.append(key)
        o.evaluate(3)                                 # each walk adds points to the archive
        return {"ci": {"verdict": "identified", "ci_lo": 1.0, "ci_hi": 2.0, "n_components": 1}, "stops": {},
                "gain": 0.0, "evals": 3, "nodes": [], "box_image": (1e-9, 1e9), "ll_ref": ll_hat}

    def widen(key, kw, Z, LL, ll_hat, box, image, **ports):    # a claim whose lower end depends on the archive's size
        lo = 1.0 / LL.size
        return dict(kw, lower=lo), {"changed": lo != kw["lower"], "lower_moved_dex": 0.0, "upper_moved_dex": 0.0}, \
            {"extents": {key: None}}

    def reconcile(o, key, w, Z, LL, **kw):            # the final pass: recorded, and moves nothing here
        walked.append(("reconcile", key, LL.size))
        return w, None

    monkeypatch.setattr(W, "walk", walk)
    monkeypatch.setattr(W, "reconcile", reconcile)
    monkeypatch.setattr(Wd, "claim_from_walk", lambda key, w, point, **ports: {"shape": "closed", "lower": 1.0,
                                                                                "upper": 2.0, "point": point,
                                                                                "level": 0.95})
    monkeypatch.setattr(Wd, "widen", widen)
    monkeypatch.setattr(PJ._je, "sc_extents", lambda Z, LL, params, box, ll_hat, thr: {"extents": {}, "gap_nats": 0.0})
    out = {"result": {"theta": [1e-3, 0.2, 4e-3, 7.0], "loglik": -100.0}, "objectives": [obj], "claims": []}
    return out, walked


def test_the_lead_group_is_the_frozen_one():
    lead, source = K.lead_group()
    assert lead == LEAD and source.startswith("measured")
    order, _ = K.order()
    assert tuple(order[:len(lead)]) == lead


def test_the_default_walks_only_the_lead_group(fake):
    out, walked = fake
    cc = PJ.combination_claims(out, "profile_jax2", None, mode="lead")
    assert [k for k in walked if isinstance(k, str)] == list(LEAD) and [c.parameter for c in cc["claims"]] == list(LEAD)
    assert [x[1] for x in walked if isinstance(x, tuple)] == list(LEAD)          # reconciled once each: nothing moved
    assert cc["mode"] == "lead" and cc["order"] == list(LEAD) and cc["combination_claims_contradicted"] == []


def test_all_adds_four_claims_and_leaves_the_lead_pair_as_the_default_gives_it(fake, monkeypatch):
    out, walked = fake
    lead = {c.parameter: c for c in PJ.combination_claims(out, "profile_jax2", None, mode="lead")["claims"]}
    walked.clear()
    out2 = {**out, "objectives": [FakeObjective()]}
    out2["objectives"][0].evaluate(5)
    cc = PJ.combination_claims(out2, "profile_jax2", None, mode="all")
    order, _ = K.order()
    assert [k for k in walked if isinstance(k, str)] == list(order) and \
        [c.parameter for c in cc["claims"]] == list(order)
    # each group is reconciled after its own walks: the lead pair before the other four walk, over its archive
    rec = [x for x in walked if isinstance(x, tuple)]
    assert [x[1] for x in rec] == list(order)
    assert walked.index(rec[len(LEAD)]) > walked.index(order[-1]) > walked.index(rec[len(LEAD) - 1])
    assert all(x[2] == 5 + 3 * len(LEAD) for x in rec[:len(LEAD)]) and all(x[2] == 5 + 3 * len(order) for x in rec[len(LEAD):])
    got = {c.parameter: c for c in cc["claims"]}
    for k in LEAD:                                    # widened over the archive after the lead walks: identical
        assert got[k].lower == lead[k].lower == 1.0 / (5 + 3 * len(LEAD)), k
    for k in order[len(LEAD):]:                       # the rest over the whole archive
        assert math.isclose(got[k].lower, 1.0 / (5 + 3 * len(order))), k
    # the whole archive reaches past each lead claim's lower end: reported, not applied (FR-019's policy)
    assert {(x["parameter"], x["side"]) for x in cc["combination_claims_contradicted"]} == {(k, "lower") for k in LEAD}


@pytest.mark.parametrize("bad", [True, "six", 2, None])
def test_an_unknown_combinations_value_is_refused(fake, bad):
    from sraging.errors import UnsupportedOptionError
    with pytest.raises(UnsupportedOptionError):
        PJ.check_combinations(bad)
    for good in ("lead", "all", False):
        assert PJ.check_combinations(good) == good


# ------------------------------------------------------------------------- spec 003d D1: the "first3" mode (T042)
def test_first3_walks_the_lead_pair_then_the_third_and_leaves_the_lead_claims_as_lead_gives_them(fake):
    """FR-022 (Naveh's D1, 2026-10-13): the re-test's three combinations. The lead pair is walked, reconciled and
    widened exactly as in "lead"; sqrt(eta)/eps is walked after it and widened over the archive after three walks."""
    out, walked = fake
    lead = {c.parameter: c for c in PJ.combination_claims(out, "profile_jax2", None, mode="lead")["claims"]}
    walked.clear()
    out2 = {**out, "objectives": [FakeObjective()]}
    out2["objectives"][0].evaluate(5)
    assert PJ.check_combinations("first3") == "first3"
    cc = PJ.combination_claims(out2, "profile_jax2", None, mode="first3")
    order, _ = K.order()
    three = list(order[:3])
    assert three == ["eta_over_eps", "rbar_over_eta", "sqrt_eta_over_eps"]
    assert [k for k in walked if isinstance(k, str)] == three and [c.parameter for c in cc["claims"]] == three
    rec = [x for x in walked if isinstance(x, tuple)]
    assert [x[1] for x in rec] == three
    assert walked.index(rec[2]) > walked.index(three[2]) > walked.index(rec[1])   # the lead pair reconciled first
    got = {c.parameter: c for c in cc["claims"]}
    for k in LEAD:
        assert got[k].lower == lead[k].lower == 1.0 / (5 + 3 * len(LEAD)), k
    assert math.isclose(got["sqrt_eta_over_eps"].lower, 1.0 / (5 + 3 * 3))
    assert cc["mode"] == "first3" and cc["order"] == three and cc["lead_group"] == list(LEAD)


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["lead", "all"])
@pytest.mark.parametrize("cohort", ["small", "small_censored"])
def test_lead_and_all_are_byte_identical_to_the_parent_commits(cohort, mode):
    """FR-022 / SC-012: the real fit's combination_claims output (timings stripped, floats as float.hex) equals the
    reference captured at the parent commit of the "first3" change (tools/combination_modes_003d.py capture; this
    host and the pinned environment only)."""
    import os
    import combination_modes_003d as CM
    ref = open(os.path.join(CM.DATA, f"{cohort}__{mode}.json")).read()
    assert CM.output(cohort, mode) == ref
