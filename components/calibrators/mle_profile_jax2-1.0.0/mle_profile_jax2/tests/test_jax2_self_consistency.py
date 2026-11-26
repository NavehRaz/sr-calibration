"""rc2 I1: claim self-consistency -- every claim contains the value of every point the fit evaluated within
chi2_1(0.95)/2 of the best log-likelihood, and a widened side that reaches the box edge opens.

  * `estimator.sc_extents` on synthetic node sets: the threshold is relative to ll_best (not ll_hat), points
    outside the box / non-finite are ignored, NaN coordinates are unknown, the stale-MLE classes;
  * `estimator._sc_archive` reads exactly the memo caches of a real objective;
  * `__init__.sc_widen_claim` / `sc_widen_claims` / `final_claim_kwargs`: every shape transition, the box-edge
    tolerance, never narrower, never a point on a decline, contract-valid output, the h claim;
  * a real (tiny-grid) fit: self_consistent=False is the dev3 fit; True adds only the archive block, and the
    widened claims contain every qualifying archived point (the strict rule, checked independently).
"""
from __future__ import annotations

import collections
import json
import math
import os
import random

import pytest

import _jax2_testlib as T

THR = 3.841458820694124            # chi2.ppf(0.95, 1)
HALF = THR / 2
P4 = ("eta", "beta", "epsilon", "kappa")


@pytest.fixture(scope="module")
def est():
    return T.pkg_module("estimator")


@pytest.fixture(scope="module")
def ad():
    """The adapter under a TEST key: the registration guard (`__name__ == srbench_methods.<dir>`) keeps it from
    registering anything, so it is safe in-process (no discover() is ever run here)."""
    m = T.load("jax2_tests.sc.adapter", os.path.join(T.PKG, "__init__.py"))
    assert m.__name__ != m._KEY
    return m


def _pts(rows):
    """[(ll, theta)] -> (Z, LL) arrays; theta entries may be None (unknown)."""
    import numpy as np
    Z = np.array([[np.nan if v is None else math.log(v) for v in th] for _ll, th in rows], float)
    LL = np.array([ll for ll, _th in rows], float)
    return Z, LL


def _box(ad):
    return {p: tuple(ad.BOUNDS[p]) for p in P4}


# ----------------------------------------------------------------------------------------------- sc_extents

def test_extents_threshold_is_relative_to_ll_best(est, ad):
    rows = [(0.0, [1e-3, 1.0, 1e-2, 1.0]),            # the MLE
            (0.5, [2e-3, 3.0, 1e-2, 2.0]),            # beats the MLE by 0.5: ll_best = 0.5 -> cut at 0.5 - 1.92
            (-1.0, [5e-3, 10.0, 3e-2, 5.0]),          # 1.5 below ll_best: qualifies
            (-1.5, [9e-3, 50.0, 9e-2, 9.0]),          # 2.0 below ll_best: does NOT (it would against ll_hat)
            (-9.0, [1e-1, 900.0, 1.0, 100.0])]        # far below
    Z, LL = _pts(rows)
    sc = est.sc_extents(Z, LL, P4, _box(ad), 0.0, THR)
    assert sc["ll_best"] == 0.5 and sc["gap_nats"] == 0.5 and sc["stale"] == "minor"
    assert sc["n_points"] == 5 and sc["n_in_box"] == 5 and sc["n_qualifying"] == 3
    e = sc["extents"]
    assert (e["eta"]["qmin"], e["eta"]["qmax"]) == pytest.approx((1e-3, 5e-3), rel=1e-12)
    assert (e["beta"]["qmin"], e["beta"]["qmax"]) == pytest.approx((1.0, 10.0), rel=1e-12)
    assert e["beta"]["theta_qmax"] == pytest.approx([5e-3, 10.0, 3e-2, 5.0], rel=1e-12)
    assert e["beta"]["ll_qmax"] == -1.0
    assert sc["best"]["ll"] == 0.5 and sc["best"]["theta"] == pytest.approx([2e-3, 3.0, 1e-2, 2.0], rel=1e-12)
    json.dumps(sc)                                     # JSON-ready (it goes into the record's diagnostics)


@pytest.mark.parametrize("gap,cls", [(0.0, "none"), (0.01, "none"), (0.0100001, "minor"), (HALF, "minor"),
                                     (HALF + 1e-9, "major"), (50.0, "major")])
def test_stale_classes(est, ad, gap, cls):
    Z, LL = _pts([(0.0, [1e-3, 1.0, 1e-2, 1.0]), (gap, [2e-3, 1.0, 1e-2, 1.0])])
    sc = est.sc_extents(Z, LL, P4, _box(ad), 0.0, THR)
    assert sc["stale"] == cls
    assert (sc["best"] is None) == (cls == "none")


def test_extents_ignore_out_of_box_and_nonfinite(est, ad):
    rows = [(0.0, [1e-3, 1.0, 1e-2, 1.0]),
            (5.0, [1e-3, 2e3, 1e-2, 1.0]),            # beta above its box: not a model point, not ll_best
            (5.0, [1e-3, 1.0, 1e-2, 5e-4]),           # kappa below its box
            (-0.5, [1e-3, 1e3 * (1 + 1e-12), 1e-2, 1.0]),   # one ulp-ish over the edge: inside (rtol)
            (float("nan"), [1e-3, 7.0, 1e-2, 1.0]),
            (float("inf"), [1e-3, 8.0, 1e-2, 1.0])]
    Z, LL = _pts(rows)
    sc = est.sc_extents(Z, LL, P4, _box(ad), 0.0, THR)
    assert sc["ll_best"] == 0.0 and sc["stale"] == "none"
    assert sc["n_out_of_box"] == 2 and sc["n_in_box"] == 2 and sc["n_qualifying"] == 2
    assert sc["extents"]["beta"]["qmax"] == pytest.approx(1e3, rel=1e-9)
    assert sc["extents"]["kappa"]["qmin"] == 1.0


def test_extents_partial_points(est, ad):
    """A NaN coordinate is unknown: it contributes to no extent and is not box-checked; ll still counts."""
    rows = [(0.0, [1e-3, 1.0, 1e-2, 1.0]),
            (0.2, [None, 40.0, None, None]),           # a beta-profile node without its nuisance
            (-0.3, [1e-2, None, None, None])]
    Z, LL = _pts(rows)
    sc = est.sc_extents(Z, LL, P4, _box(ad), 0.0, THR)
    assert sc["ll_best"] == 0.2 and sc["n_qualifying"] == 3
    assert sc["extents"]["beta"]["qmax"] == 40.0 and sc["extents"]["beta"]["n"] == 2
    assert sc["extents"]["eta"]["qmax"] == pytest.approx(1e-2) and sc["extents"]["eta"]["n"] == 2
    assert sc["extents"]["epsilon"]["n"] == 1 and sc["extents"]["kappa"]["n"] == 1
    assert sc["best"]["theta"] == [None, 40.0, None, None]


def test_extents_empty_and_shape_errors(est, ad):
    import numpy as np
    sc = est.sc_extents(np.empty((0, 4)), np.empty(0), P4, _box(ad), -3.0, THR)
    assert sc["ll_best"] == -3.0 and sc["n_points"] == 0
    assert all(v["n"] == 0 and v["qmin"] is None for v in sc["extents"].values())
    with pytest.raises(ValueError):
        est.sc_extents(np.zeros((3, 4)), np.zeros(2), P4, _box(ad), 0.0, THR)


# ----------------------------------------------------------------------------------------------- _sc_archive

class _FakeObj:
    def __init__(self, NP):
        self.NP = NP
        self._f_cache = {}
        self._vg_cache = {}


def test_archive_reads_both_caches(est):
    import numpy as np
    o = _FakeObj(4)
    za, zb, zc = np.array([0.1, 0.2, 0.3, 0.4]), np.array([1.0, 2.0, 3.0, 4.0]), np.array([-1.0, 0.0, 1.0, 2.0])
    o._vg_cache[za.tobytes()] = (-5.0, np.ones(4))
    o._vg_cache[zb.tobytes()] = (-7.5, np.ones(4))
    o._f_cache[zc.tobytes()] = -6.25
    Z, LL = est._sc_archive(o)
    got = sorted(zip(LL.tolist(), map(tuple, Z.tolist())))
    assert got == sorted([(-5.0, tuple(za)), (-7.5, tuple(zb)), (-6.25, tuple(zc))])
    o._f_cache[np.zeros(5).tobytes()] = 1.0
    with pytest.raises(ValueError, match="memo key"):
        est._sc_archive(o)
    Z0, LL0 = est._sc_archive(_FakeObj(5))
    assert Z0.shape == (0, 5) and LL0.shape == (0,)


@pytest.mark.sut
def test_archive_is_the_real_memo(est):
    """On a real objective every nll / nll_grad evaluation (and nothing else) is in the archive, with the value
    the optimiser saw."""
    import numpy as np
    T.jax_env()
    fj = T.pkg_module("fp_jax")
    t = np.array([0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7])
    e = np.array([1, 1, 1, 1, 1, 1, 1, 0])
    kw = dict(N=40, dt=0.02, tau_max=3.0, Gtol=1e-11, sampling_dt=0.1)
    obj = est._Objective(fj, t, e, 1.1, kw, est.box_for(300.0))
    zs = [np.log([0.5, 2.0, 0.3, 1.0]), np.log([0.6, 2.0, 0.3, 1.0]), np.log([0.5, 3.0, 0.2, 2.0])]
    want = {}
    for z in zs[:2]:
        v, _g = obj.nll_grad(z)
        want[z.tobytes()] = -v
    want[zs[2].tobytes()] = -obj.nll(zs[2])
    obj.nll_grad(zs[0])                                # a memo hit: no new point
    assert obj.nll(np.log([2e3, 1.0, 1.0, 1.0])) == np.inf       # outside the box: never evaluated
    Z, LL = est._sc_archive(obj)
    assert len(LL) == obj.n_f + obj.n_g == 3
    got = {np.asarray(z, float).tobytes(): ll for z, ll in zip(Z, LL)}
    assert got == want


# ----------------------------------------------------------------------------------------------- the widening

def _closed(p="beta", lo=0.5, hi=5.0, point=1.0):
    return {"parameter": p, "shape": "closed", "level": 0.95, "lower": lo, "upper": hi, "point": point}


def _ext(qmin, qmax, n=3):
    return {"n": n, "qmin": qmin, "qmax": qmax}


def test_inside_points_change_nothing(ad):
    kw = _closed()
    new, rec = ad.sc_widen_claim(kw, _ext(0.6, 4.0), (1e-12, 1e3))
    assert new == kw and new is not kw and rec["changed"] is False
    assert rec["shape_after"] == "closed" and rec["opened"] == []


def test_union_moves_both_edges_out_to_the_points(ad):
    new, rec = ad.sc_widen_claim(_closed(), _ext(0.05, 80.0), (1e-12, 1e3))
    assert new["shape"] == "closed" and new["lower"] == 0.05 and new["upper"] == 80.0 and new["point"] == 1.0
    assert rec["changed"] and rec["lower_moved_dex"] == pytest.approx(1.0) and \
        rec["upper_moved_dex"] == pytest.approx(math.log10(16.0))


@pytest.mark.parametrize("qmax,shape", [(1e3, "lower_bound"), (999.9999999999998, "lower_bound"),
                                        (1e3 * 0.98, "lower_bound"), (1e3 * 10 ** -0.0099, "lower_bound"),
                                        (1e3 * 10 ** -0.0101, "closed"), (900.0, "closed")])
def test_upper_side_opens_only_at_the_box_edge(ad, qmax, shape):
    new, rec = ad.sc_widen_claim(_closed(), _ext(0.6, qmax), (1e-12, 1e3))
    assert new["shape"] == shape
    if shape == "lower_bound":
        assert "upper" not in new and new["lower"] == 0.5 and new["point"] == 1.0 and rec["opened"] == ["upper"]
        assert rec["upper_moved_dex"] == pytest.approx(math.log10(1e3 / 5.0))
    else:
        assert new["upper"] == pytest.approx(qmax) and rec["opened"] == []


def test_lower_side_opens_at_the_kappa_floor(ad):
    kw = _closed("kappa", 0.05, 20.0, 1.0)
    new, rec = ad.sc_widen_claim(kw, _ext(1.01e-3, 3.0), (1e-3, 300.0))
    assert new["shape"] == "upper_bound" and "lower" not in new and new["upper"] == 20.0 and rec["opened"] == ["lower"]


def test_both_sides_open_to_a_decline_without_a_point(ad):
    new, rec = ad.sc_widen_claim(_closed(), _ext(1e-12, 1e3), (1e-12, 1e3))
    assert new == {"parameter": "beta", "shape": "none", "level": 0.95}
    assert sorted(rec["opened"]) == ["lower", "upper"] and rec["shape_after"] == "none"


def test_one_sided_claims(ad):
    ub = {"parameter": "beta", "shape": "upper_bound", "level": 0.95, "upper": 5.0, "point": 1.0}
    new, _ = ad.sc_widen_claim(ub, _ext(1e-9, 50.0), (1e-12, 1e3))     # the open lower side stays open
    assert new == dict(ub, upper=50.0)
    new, rec = ad.sc_widen_claim(ub, _ext(0.1, 1e3), (1e-12, 1e3))
    assert new == {"parameter": "beta", "shape": "none", "level": 0.95} and rec["opened"] == ["upper"]
    lb = {"parameter": "eta", "shape": "lower_bound", "level": 0.95, "lower": 1e-3, "point": 1e-2}
    new, _ = ad.sc_widen_claim(lb, _ext(1e-4, 5e2), (1e-12, 1e3))      # the open upper side stays open
    assert new == dict(lb, lower=1e-4)
    new, rec = ad.sc_widen_claim(lb, _ext(1e-12, 1.0), (1e-12, 1e3))
    assert new["shape"] == "none" and "point" not in new and rec["opened"] == ["lower"]


def test_decline_and_empty_extents_untouched(ad):
    none = {"parameter": "kappa", "shape": "none", "level": 0.95}
    assert ad.sc_widen_claim(none, _ext(1e-3, 300.0), (1e-3, 300.0))[0] == none
    kw = _closed()
    assert ad.sc_widen_claim(kw, None, (1e-12, 1e3))[0] == kw
    assert ad.sc_widen_claim(kw, {"n": 0, "qmin": None, "qmax": None}, (1e-12, 1e3))[0] == kw


def test_edge_tolerance_is_a_parameter(ad):
    kw = _closed()
    assert ad.sc_widen_claim(kw, _ext(0.6, 900.0), (1e-12, 1e3))[0]["shape"] == "closed"
    assert ad.sc_widen_claim(kw, _ext(0.6, 900.0), (1e-12, 1e3), edge_dex=0.05)[0]["shape"] == "lower_bound"
    assert ad.SC_EDGE_DEX == 0.01


def test_random_never_narrower_contract_valid_and_contains_points(ad):
    """Property check over random claims and extents: the result contains the old claim and every point value
    (or that side is open), its shape follows the open sides, and the tester's contract accepts it."""
    from srbench.contract.types import Claim
    from srbench.contract.validate import validate_claim
    desc = ad.MleProfileJax2.descriptor
    rnd = random.Random(20261003)
    box = ad.BOUNDS
    n = collections.Counter()
    for _ in range(4000):
        p = rnd.choice(P4)
        lo_b, hi_b = box[p]
        lg = lambda a, b: 10 ** rnd.uniform(math.log10(a), math.log10(b))     # noqa: E731
        a, b = sorted((lg(lo_b, hi_b), lg(lo_b, hi_b)))
        if a == b:
            continue
        pt = lg(a, b)
        shape = rnd.choice(["closed", "upper_bound", "lower_bound", "none"])
        kw = {"parameter": p, "shape": shape, "level": 0.95}
        if shape in ("closed", "lower_bound"):
            kw["lower"] = a
        if shape in ("closed", "upper_bound"):
            kw["upper"] = b
        if shape != "none":
            kw["point"] = pt
        if rnd.random() < 0.15:
            q0, q1 = lo_b, hi_b                                    # exactly on the box edges
        else:
            q0, q1 = sorted((lg(lo_b, hi_b), lg(lo_b, hi_b)))
        new, rec = ad.sc_widen_claim(kw, _ext(q0, q1), box[p])
        validate_claim(Claim(**new), desc)
        n["checked"] += 1
        n["changed"] += rec["changed"]
        n["opened"] += bool(rec["opened"])
        n[f"{shape}->{new['shape']}"] += 1
        lo1, hi1 = new.get("lower"), new.get("upper")
        if shape == "none":
            assert new == kw
            continue
        if "lower" in kw:                                          # never narrower
            assert lo1 is None or lo1 <= kw["lower"]
            assert lo1 is None or lo1 <= q0 * (1 + 1e-12)
            assert (lo1 is None) == ("lower" in rec["opened"])
        else:
            assert lo1 is None
        if "upper" in kw:
            assert hi1 is None or hi1 >= kw["upper"]
            assert hi1 is None or hi1 >= q1 * (1 - 1e-12)
            assert (hi1 is None) == ("upper" in rec["opened"])
        else:
            assert hi1 is None
        want = ("closed" if lo1 is not None and hi1 is not None else "lower_bound" if lo1 is not None
                else "upper_bound" if hi1 is not None else "none")
        assert new["shape"] == want
        assert ("point" in new) == (want != "none")
        if want != "none":
            assert new["point"] == pt
        for side, v in (("lower", lo1), ("upper", hi1)):
            if side in rec["opened"]:
                edge = q0 / lo_b if side == "lower" else hi_b / q1
                assert edge <= 10 ** ad.SC_EDGE_DEX * (1 + 1e-12)
    # the loop really exercised every transition (positive control on the generator)
    assert n["checked"] > 3900 and n["changed"] > 1000 and n["opened"] > 200
    for tr in ("closed->closed", "closed->lower_bound", "closed->upper_bound", "closed->none",
               "upper_bound->upper_bound", "upper_bound->none", "lower_bound->lower_bound", "lower_bound->none",
               "none->none"):
        assert n[tr] > 0, (tr, dict(n))


def test_h_claim_widening_keeps_its_extra_keys(ad):
    box = [1e-7, 1.0]
    h = ad.h_claim_kwargs(0.01, {"h@0.95": (2e-3, 5e-2)}, {"h": "identified"}, box)
    assert h["shape"] == "closed" and h["bounds"] == box and h["verdict"] == "identified"
    new, rec = ad.sc_widen_claim(h, _ext(1e-7 * 1.001, 0.04), box)        # a point at the h floor
    assert new["shape"] == "upper_bound" and "lower" not in new and new["upper"] == 5e-2
    assert new["bounds"] == box and new["verdict"] == "identified" and new["point"] == 0.01
    assert rec["opened"] == ["lower"]


def test_final_claim_kwargs(ad):
    theta = [1e-3, 1.0, 1e-2, 1.0]
    ci = {"eta@0.95": (5e-4, 2e-3), "beta@0.95": (0.5, 5.0), "epsilon@0.95": (5e-3, 2e-2),
          "kappa@0.95": (float("nan"), float("nan"))}
    verdicts = {"eta": "identified", "beta": "identified", "epsilon": "identified", "kappa": "flat"}
    res = {"theta": theta, "ci": ci, "diagnostics": {"verdicts": verdicts}}
    base = ad.claim_kwargs(theta, ci, verdicts)
    c0, b0 = ad.final_claim_kwargs(res, False)
    assert c0 == base and b0 is None
    with pytest.raises(ValueError, match="self_consistency"):
        ad.final_claim_kwargs(res, True)
    sc = {"ll_hat": 0.0, "ll_best": 0.0, "gap_nats": 0.0, "stale": "none",
          "extents": {"eta": _ext(6e-4, 1.5e-3), "beta": _ext(0.7, 1e3), "epsilon": _ext(1e-3, 1.5e-2),
                      "kappa": _ext(1e-3, 300.0)}}
    res["diagnostics"]["self_consistency"] = sc
    c1, b1 = ad.final_claim_kwargs(res, True)
    by = {c["parameter"]: c for c in c1}
    assert by["eta"] == base[0]                                     # inside: unchanged
    assert by["beta"]["shape"] == "lower_bound" and by["beta"]["lower"] == 0.5
    assert by["epsilon"]["lower"] == 1e-3 and by["epsilon"]["upper"] == 2e-2
    assert by["kappa"]["shape"] == "none"
    assert b1["claims_profile"] == base and b1["n_changed"] == 2 and b1["n_shape_changes"] == 1
    assert b1["enabled"] is True and b1["edge_dex"] == ad.SC_EDGE_DEX and len(b1["widening"]) == 4
    assert "self_consistency" in res["diagnostics"] and "widening" not in sc      # the input is not mutated
    none_res = {"theta": theta, "ci": {}, "diagnostics": {"verdicts": {}}}
    c2, b2 = ad.final_claim_kwargs(none_res, True)                 # e.g. "all starts failed": nothing stated
    assert all(c["shape"] == "none" for c in c2) and b2["enabled"] is False


def test_registered_classes_default_on(ad):
    assert ad.MleProfileJax2.SELF_CONSISTENT is True and ad.MleProfileJax2H.SELF_CONSISTENT is True


def test_claimset_uses_the_widened_claims(ad):
    from srbench.contract.types import Observation
    from srbench.provenance import CodeProvenance
    theta = [1e-3, 1.0, 1e-2, 1.0]
    ci = {"eta@0.95": (5e-4, 2e-3), "beta@0.95": (0.5, 5.0), "epsilon@0.95": (5e-3, 2e-2),
          "kappa@0.95": (float("nan"), float("nan"))}
    verdicts = {"eta": "identified", "beta": "identified", "epsilon": "identified", "kappa": "flat"}
    sc = {"ll_hat": 0.0, "ll_best": 0.0, "gap_nats": 0.0, "stale": "none",
          "extents": {"eta": _ext(6e-4, 1.5e-3), "beta": _ext(0.7, 1e3), "epsilon": _ext(5e-3, 2e-2),
                      "kappa": _ext(1e-3, 300.0)}}
    res = {"theta": theta, "ci": ci, "loglik": 0.0, "wall_s": 1.0,
           "diagnostics": {"verdicts": verdicts, "self_consistency": sc}}
    obs = Observation(cohort_id="x", ages=(1.0, 2.0), events=(1, 1), delta=None)
    prov = CodeProvenance(package="SRK", resolved_path="/nowhere")
    for cls, beta_shape, has_block in ((ad.MleProfileJax2, "lower_bound", True),
                                       (type("Off", (ad.MleProfileJax2,), {"SELF_CONSISTENT": False}), "closed", False)):
        cs = cls().claimset(obs, res, prov, 0.0, 0.0)
        assert cs.claim_for("beta").shape == beta_shape
        assert ("widening" in cs.diagnostics.get("self_consistency", {})) == has_block
        assert cs.diagnostics["switches"]["self_consistent"] is cls.SELF_CONSISTENT


# ----------------------------------------------------------------------------------------------- a real fit

@pytest.mark.sut
def test_real_fit_off_is_dev3_and_on_contains_every_qualifying_point(est, ad, monkeypatch):
    """Tiny real fit (80 records on a 0.5 grid, solver N=40): the flag changes nothing but the new block, and the
    claims built from it contain every archived point within thr/2 of ll_best -- checked from the archive
    itself, independently of `sc_extents`."""
    import numpy as np
    T.jax_env()
    fj = T.pkg_module("fp_jax")
    rng = np.random.default_rng(1)
    u = rng.random(80)
    t = np.floor(np.log(1 - np.log(u) / 0.05) / 0.3 / 0.5) * 0.5 + 0.5
    e = np.ones_like(t, dtype=int)
    e[t > 12] = 0
    t = np.minimum(t, 12.0)
    kw = dict(N=40, dt=0.02)
    r0 = est.fit(fj, t, e, delta=0.5, solver_kw=kw)
    seen = {}
    real = est._sc_archive

    def spy(obj):
        Z, LL = real(obj)
        seen["Z"], seen["LL"], seen["params"] = Z.copy(), LL.copy(), obj.params
        return Z, LL
    monkeypatch.setattr(est, "_sc_archive", spy)
    r1 = est.fit(fj, t, e, delta=0.5, solver_kw=kw, self_consistent=True)

    def strip(r):
        d = {k: v for k, v in r["diagnostics"].items() if not k.endswith("_s") and k != "self_consistency"}
        return json.dumps(dict(theta=r["theta"], ci=r["ci"], loglik=r["loglik"], d=d), sort_keys=True, default=str)
    assert "self_consistency" not in r0["diagnostics"]
    assert strip(r0) == strip(r1)
    sc = r1["diagnostics"]["self_consistency"]
    assert sc["ll_hat"] == r1["loglik"]
    assert sc["n_points"] == r1["diagnostics"]["n_eval"] + r1["diagnostics"]["n_grad"] == len(seen["LL"])
    claims, block = ad.final_claim_kwargs(r1, True)
    assert block["claims_profile"] == ad.claim_kwargs(r1["theta"], r1["ci"], r1["diagnostics"]["verdicts"])
    # the strict rule, from the raw archive
    TH, LL = np.exp(seen["Z"]), seen["LL"]
    box = est.box_for(300.0)
    inb = np.all([(TH[:, k] >= box[p][0] * (1 - 1e-9)) & (TH[:, k] <= box[p][1] * (1 + 1e-9))
                  for k, p in enumerate(seen["params"])], axis=0) & np.isfinite(LL)
    llb = max(r1["loglik"], LL[inb].max())
    q = inb & (LL >= llb - HALF)
    assert q.sum() == sc["n_qualifying"] > 0
    for c in claims:
        k = P4.index(c["parameter"])
        vals = TH[q, k]
        if c["shape"] == "none":
            continue
        if "lower" in c:
            assert vals.min() >= c["lower"] * (1 - 1e-12)
        if "upper" in c:
            assert vals.max() <= c["upper"] * (1 + 1e-12)


@pytest.mark.sut
def test_run_fit_hook_off_is_dev3_call_and_on_widens_h_claim(ad, monkeypatch):
    """run_fit with stub siblings (no fit): off -> the estimator is called with dev3's keyword set exactly and no h
    widening; on -> `self_consistent=True` is passed and the would-be h claim is widened (profile claim kept)."""
    from types import SimpleNamespace
    from srbench.contract.types import Observation
    calls = []
    box_h = [1e-7, 1.0]

    def fake_fit(fj, ages, events, **kw):
        calls.append(sorted(kw))
        dg = {"verdicts": {"eta": "flat", "beta": "flat", "epsilon": "flat", "kappa": "flat", "h": "identified"},
              "hazard": {"verdict": "identified", "box": box_h}, "exc_counts": {}}
        if kw.get("self_consistent"):
            dg["self_consistency"] = {"extents": {"h": {"n": 5, "qmin": 1.0001e-7, "qmax": 0.03}}}
        return {"theta": [1e-3, 1.0, 1e-2, 1.0, 0.01], "ci": {"h@0.95": (2e-3, 5e-2)}, "loglik": 0.0,
                "diagnostics": dg}
    prep = SimpleNamespace(ages=(1.0, 2.0), events=(1, 1), window_eff=None, censor_tie={}, record={})
    stubs = {"observation_model": SimpleNamespace(prepare=lambda *a, **k: prep),
             "fp_jax": SimpleNamespace(), "estimator": SimpleNamespace(fit=fake_fit)}
    monkeypatch.setattr(ad, "_sibling", lambda name: stubs[name])
    monkeypatch.setattr(ad, "version_lock_guard", lambda: None)
    obs = Observation(cohort_id="x", ages=(1.0, 2.0), events=(1, 1), delta=None)
    r0 = ad.run_fit(obs, relabel=True, hazard=True, window=False)
    assert calls[-1] == ["delta", "hazard", "kappa_max", "levels", "window"]      # dev3's call, exactly
    assert r0["diagnostics"]["hazard"]["h_claim"]["shape"] == "closed" and "h_claim_sc" not in r0["diagnostics"]["hazard"]
    r1 = ad.run_fit(obs, relabel=True, hazard=True, window=False, self_consistent=True)
    assert calls[-1] == ["delta", "hazard", "kappa_max", "levels", "self_consistent", "window"]
    hz = r1["diagnostics"]["hazard"]
    assert hz["h_claim_profile"]["shape"] == "closed" and hz["h_claim_profile"]["lower"] == 2e-3
    assert hz["h_claim"]["shape"] == "upper_bound" and hz["h_claim"]["upper"] == 5e-2 and "lower" not in hz["h_claim"]
    assert hz["h_claim_sc"]["opened"] == ["lower"]
