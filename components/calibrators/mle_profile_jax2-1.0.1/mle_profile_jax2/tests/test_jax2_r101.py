"""1.0.1.dev1: the four switches of estimator note 13, each OFF in `run_fit`'s defaults (all off = 1.0.0's call).

  * the adapter: INNER_POLISH_J, BRACKET_BISECT, EXTENT_Z on for both names, BOX_FLAG OFF since 1.0.1.dev2 (Naveh
    2026-10-08: dormant; the switch and its code stay); `run_fit` passes each one's estimator keyword only when it is
    on (off: 1.0.0's keyword set, exactly); `calibrate` passes the constants; diagnostics.switches records them; both
    names are 1.0.1.dev3 (the kfix names: test_jax2_kfix.py);
  * `_ci_brackets` is `fp_gof.profile_ci`'s own pair of nodes (random profiles, disjoint 95% sets included);
  * `_bisect_brackets` on an analytic objective with a non-quadratic profile: midpoint nodes at the geometric
    centre, every bracket <= 0.3 dex, the profile there solved, the edge closer to the truth;
  * INNER_POLISH_J: `_polish_edges` re-solves the inner node of a hazard-free objective only with the switch on;
  * EXTENT_Z: `sc_extents(store_z=True)` stores each extent point's row of Z bit for bit (JSON round trip too),
    and off is the 1.0.0 dict;
  * BOX_FLAG: `_box_status` (on the box within 2 ulps in z, beyond it, h's floor excepted) and the adapter's
    `box_limited_block` (which point set each final edge);
  * a real tiny-grid fit: EXTENT_Z + BOX_FLAG change nothing but their diagnostics; INNER_POLISH_J + BRACKET_BISECT
    leave no crossing bracket wider than 0.3 dex, and every bracket straddles its edge.
The full-size evidence (identity of the off path to 1.0.0 on 3 quick cohorts x both names, L0) is reports/r101/."""
from __future__ import annotations

import json
import math
import os

import pytest

import _jax2_testlib as T

THR = 3.841458820694124            # chi2.ppf(0.95, 1)
NEW = ("inner_polish_j", "bracket_bisect", "extent_z", "box_flag")
NEW_EST = ("inner_polish", "bracket_bisect", "extent_z", "box_flag")


@pytest.fixture(scope="module")
def est():
    T.jax_env()
    return T.pkg_module("estimator")


@pytest.fixture(scope="module")
def ad():
    """The adapter under a TEST key: it registers nothing there (see test_jax2_self_consistency.py)."""
    m = T.load("jax2_tests.r101.adapter", os.path.join(T.PKG, "__init__.py"))
    assert m.__name__ != m._KEY
    return m


# ----------------------------------------------------------------------------------------------- the adapter

def test_adapter_r101_defaults_and_switch_record():
    out = T.run_py("""
import json, os, importlib.util, inspect, sys
s = importlib.util.spec_from_file_location("jax2_tests.r101.adapter2", os.path.join("methods", "mle_profile_jax2", "__init__.py"))
ad = importlib.util.module_from_spec(s); sys.modules[s.name] = ad; s.loader.exec_module(ad)
sig = inspect.signature(ad.run_fit)
src_run = inspect.getsource(ad.run_fit); src_cal = inspect.getsource(ad._MleProfileJax2Base.calibrate)
src_cs = inspect.getsource(ad._MleProfileJax2Base.claimset)
C = ("INNER_POLISH_J", "BRACKET_BISECT", "EXTENT_Z", "BOX_FLAG")
A = ("inner_polish_j", "bracket_bisect", "extent_z", "box_flag")
E = ("inner_polish", "bracket_bisect", "extent_z", "box_flag")
print(json.dumps(dict(
    on=[[getattr(c, k) for k in C] for c in (ad.MleProfileJax2, ad.MleProfileJax2H)],
    others=[[c.EARLY_ALL, c.FLOOR_EARLY, c.N_RULE, c.DT_RULE, c.SELF_CONSISTENT, c.SHADOW, c.EXACT_F, c.RELABEL]
            for c in (ad.MleProfileJax2, ad.MleProfileJax2H)],
    defaults=[sig.parameters[a].default for a in A],
    run_passes_only_when_on=[('**({"%s": True} if %s else {})' % (e, a)) in src_run for a, e in zip(A, E)],
    calibrate_passes=[f"{a}=self.{c}" in src_cal for a, c in zip(A, C)],
    switches_record=[f"{a}=bool(self.{c})" in src_cs for a, c in zip(A, C)],
    box_block_in_claimset="box_limited_block(res, claims_kw, sc_block)" in src_cs,
    versions=ad.VERSIONS)))
""")
    assert out["_rc"] == 0, out["_stderr"]
    assert out["on"] == [[True, True, True, False]] * 2 and out["defaults"] == [False] * 4   # BOX_FLAG off (1.0.1.dev2)
    assert out["others"] == [[True, True, True, True, True, True, True, True]] * 2   # 1.0.0's + DT_RULE (1.0.1.dev4)
    assert all(out["run_passes_only_when_on"]) and all(out["calibrate_passes"]) and all(out["switches_record"])
    assert out["box_block_in_claimset"]
    from version_lock import versions
    assert out["versions"] == versions() and {out["versions"][n] for n in ("mle_profile_jax2", "mle_profile_jax2_h")} \
        == {"1.0.1"}                                                          # the freeze


@pytest.mark.sut
def test_run_fit_off_is_the_1_0_0_call_and_each_switch_adds_one_keyword(ad, monkeypatch):
    """run_fit with stub siblings (no fit), with the registered names' 1.0.0 constants: the four switches off ->
    the estimator gets 1.0.0's keyword set exactly; each switch on adds exactly its own keyword."""
    from types import SimpleNamespace
    from srbench.contract.types import Observation
    calls = []

    def fake_fit(fj, ages, events, **kw):
        calls.append(sorted(kw))
        return {"theta": [1e-3, 1.0, 1e-2, 1.0], "ci": {}, "loglik": 0.0, "diagnostics": {"exc_counts": {}}}
    prep = SimpleNamespace(ages=(1.0, 2.0), events=(1, 1), window_eff=None, censor_tie={}, record={})
    stubs = {"observation_model": SimpleNamespace(prepare=lambda *a, **k: prep),
             "fp_jax": SimpleNamespace(), "estimator": SimpleNamespace(fit=fake_fit)}
    monkeypatch.setattr(ad, "_sibling", lambda name: stubs[name])
    monkeypatch.setattr(ad, "version_lock_guard", lambda: None)
    obs = Observation(cohort_id="x", ages=(1.0, 2.0), events=(1, 1), delta=None)
    # 1.0.1.dev1's run_fit keyword SET (the plumbing under test), not 1.0.1's constants: dt_rule=False is kept on
    # purpose (DT_RULE = True for every name since 1.0.1.dev4; the class constants are checked above)
    base = dict(relabel=True, hazard=False, window=False, shadow=True, exact_from_F=True, self_consistent=True,
                floor_early=True, n_rule=True, dt_rule=False, early_all=True)
    v100 = sorted(["delta", "levels", "kappa_max", "hazard", "window", "shadow", "exact_from_F", "self_consistent",
                   "floor_early", "n_rule", "early_all"])
    ad.run_fit(obs, **base)
    assert calls[-1] == v100
    ad.run_fit(obs, **base, **{a: False for a in NEW})
    assert calls[-1] == v100
    for a, e in zip(NEW, NEW_EST):
        ad.run_fit(obs, **base, **{a: True})
        assert calls[-1] == sorted(v100 + [e]), a


# ----------------------------------------------------------------------------------------------- _ci_brackets

@pytest.mark.sut
def test_ci_brackets_are_profile_cis_own_pairs(est):
    """On random profiles (bumpy, so the 95% set is often disjoint; some NaN nodes): each edge profile_ci returns
    lies between the two nodes `_ci_brackets` names (inner accepted, outer not), and an edge is NaN exactly when
    there is no pair."""
    import numpy as np
    from SRK import fp_gof as GOF
    rng = np.random.default_rng(7)
    n_disjoint = 0
    for _ in range(400):
        n = int(rng.integers(3, 14))
        v = np.sort(10 ** rng.uniform(-3, 2, n))
        x = np.log10(v)
        ll = -0.5 * ((x - rng.uniform(-3, 2)) / rng.uniform(0.2, 2.0)) ** 2 * rng.uniform(1, 8) \
            + rng.normal(0, 1.5, n) * (rng.random() < 0.5)
        if rng.random() < 0.2:
            ll[int(rng.integers(n))] = float("nan")
        order = rng.permutation(n)                                   # pts need not be sorted
        pts = [[float(v[k]), float(ll[k]), None] for k in order]
        c = GOF.profile_ci([dict(value=p[0], loglik=p[1]) for p in pts], level=0.95)
        n_disjoint += int(c.get("n_components", 1) > 1)
        br = est._ci_brackets(pts, THR)
        for side, edge in (("lower", c["ci_lo"]), ("upper", c["ci_hi"])):
            if br[side] is None:
                assert not math.isfinite(edge), (side, edge, pts)
                continue
            j_in, j_out = br[side]
            fin = [p[1] for p in pts if math.isfinite(p[1])]
            dmax = max(fin)
            assert 2 * (dmax - pts[j_in][1]) <= THR < 2 * (dmax - pts[j_out][1])
            a, b = sorted((pts[j_in][0], pts[j_out][0]))
            assert math.isfinite(edge) and a * (1 - 1e-12) <= edge <= b * (1 + 1e-12), (side, edge, a, b)
    assert n_disjoint > 20                                            # the outermost-crossing rule was exercised


# ----------------------------------------------------------------------------------------------- an analytic objective

class _V(float):
    def block_until_ready(self):
        return self


MU = (math.log(1e-3), math.log(0.5), math.log(1e-2), math.log(1.0))
A4 = THR / (2.0 * math.log(10.0) ** 4)      # eta's profile deviance 2 A4 (z0 - mu0)^4 crosses thr at exactly 1 dex
B, G = 3.0, 0.4                             # nuisance curvature, and how the nuisance optimum moves with eta


def _ll(z):
    import numpy as np
    z = np.asarray(z, float)
    d0 = z[0] - MU[0]
    r = z[1:] - (np.asarray(MU[1:]) + G * d0)
    v = -A4 * d0 ** 4 - 0.5 * B * float(r @ r)
    g = np.empty(4)
    g[0] = -4 * A4 * d0 ** 3 + B * G * float(r.sum())
    g[1:] = -B * r
    return v, g


def _analytic_obj(est):
    import numpy as np

    class FJ:
        @staticmethod
        def make_loglik(t, e, L, **kw):
            return (lambda z: _V(_ll(z)[0]), lambda z: (_V(_ll(z)[0]), np.asarray(_ll(z)[1])))
    obj = est._Objective(FJ, None, None, 100.0, dict(N=1, dt=1.0, tau_max=1.0, Gtol=1e-11), est.box_for(300.0))
    return obj


def _node(dex):
    """[value, profile ll, nuisance z at its optimum] of eta's analytic profile at MLE + dex."""
    d0 = dex * math.log(10.0)
    return [math.exp(MU[0] + d0), -A4 * d0 ** 4, [m + G * d0 for m in MU[1:]]]


def _edge(pts, side):
    from SRK import fp_gof as GOF
    c = GOF.profile_ci([dict(value=p[0], loglik=p[1]) for p in pts], level=0.95)
    return c["ci_lo"] if side == "lower" else c["ci_hi"]


@pytest.mark.sut
def test_bisect_brackets_narrows_every_bracket_to_0_3_dex(est):
    import numpy as np
    obj = _analytic_obj(est)
    obj.n_bisect = obj.n_bisect_solves = 0
    pts = [_node(x) for x in (-1.6, -0.6, -0.3, 0.0, 0.3, 0.6, 1.6)]
    before = {s: math.log10(_edge(pts, s) / 1e-3) for s in ("lower", "upper")}
    z_mle = np.asarray(MU[1:])
    rec = est._bisect_brackets(obj, [1e-3, 0.5, 1e-2, 1.0], 0, [1, 2, 3], pts, 0.0, THR, z_mle, None)
    after = {s: math.log10(_edge(pts, s) / 1e-3) for s in ("lower", "upper")}
    for s, sgn in (("lower", -1.0), ("upper", 1.0)):
        r = rec[s]
        assert r["width_dex_before"] == pytest.approx(1.0) and r["width_dex"] <= est._BISECT_DEX and not r["capped"]
        assert [round(st["width_dex"], 9) for st in r["steps"]] == [1.0, 0.5]       # 1 dex -> 0.5 -> 0.25
        assert r["polish"] == []                                                      # exact solves: nothing to raise
        for st in r["steps"]:                                                         # each solved on the profile
            x = math.log10(st["value"] / 1e-3)
            assert st["ll"] == pytest.approx(-A4 * (x * math.log(10.0)) ** 4, abs=1e-6)
        assert abs(after[s] - sgn) < 0.01 < abs(before[s] - sgn)                     # truth: exactly 1 dex
    assert obj.n_bisect == 4 and obj.n_bisect_solves >= 4 * 4
    assert len(pts) == 11 and pts == sorted(pts, key=lambda q: q[0])
    br = est._ci_brackets(pts, THR)
    assert all(est._bracket_dex(pts, br[s]) <= 0.3 for s in ("lower", "upper"))
    # a bracket already <= 0.3 dex is left alone
    rec2 = est._bisect_brackets(obj, [1e-3, 0.5, 1e-2, 1.0], 0, [1, 2, 3], pts, 0.0, THR, z_mle, None)
    assert obj.n_bisect == 4 and rec2["lower"]["steps"] == [] and rec2["upper"]["steps"] == []


@pytest.mark.sut
def test_bisect_cap(est, monkeypatch):
    import numpy as np
    obj = _analytic_obj(est)
    obj.n_bisect = obj.n_bisect_solves = 0
    monkeypatch.setattr(est, "_BISECT_MAX", 1)
    pts = [_node(x) for x in (0.0, 0.3, 0.6, 1.6)]
    rec = est._bisect_brackets(obj, [1e-3, 0.5, 1e-2, 1.0], 0, [1, 2, 3], pts, 0.0, THR, np.asarray(MU[1:]), None)
    assert rec["upper"]["capped"] and len(rec["upper"]["steps"]) == 1 and rec["upper"]["width_dex"] == pytest.approx(0.5)
    assert rec["lower"]["steps"] == [] and rec["lower"]["width_dex"] is None


def _stuck_solver(est, reach_dex=0.3):
    """A stand-in for `_profile_ll_at` on the analytic objective: a node solve reaches the profile only from a start
    whose nuisance lies on the optimum path within `reach_dex` of the node (the start's implied eta from its beta
    coordinate); from farther away it stalls 3 nats low at its start (a ridge the search cannot follow)."""
    import numpy as np

    def solve(obj, theta, i, value, fidx, z0=None, maxiter=400, side=None):
        x = math.log10(value / 1e-3)
        x0 = (float(z0[0]) - MU[1]) / G / math.log(10.0)
        d0 = x * math.log(10.0)
        ll = -A4 * d0 ** 4
        if abs(x0 - x) <= reach_dex + 1e-12:
            return ll, np.asarray([m + G * d0 for m in MU[1:]])
        return ll - 3.0, np.asarray(z0, float).copy()
    return solve


@pytest.mark.sut
def test_bisect_cross_polish_rescues_a_stalled_midpoint_and_resumes(est, monkeypatch):
    """The German Shepherd failure, in miniature: the first midpoint (0.9 dex, truly INSIDE) stalls 3 nats low and is
    taken for the outer node; the second (0.7) is solved; the final bracket [0.7, 0.9] re-solves 0.9 from 0.7's
    nuisance (`_cross_polish`), 0.9 becomes accepted, the bracket moves to [0.9, 1.3] and bisection resumes (1.1).
    The edge lands within 0.01 dex of the truth (1 dex); without the cross-polish it would sit near 0.80."""
    import numpy as np
    monkeypatch.setattr(est, "_profile_ll_at", _stuck_solver(est))
    obj = _analytic_obj(est)
    obj.n_bisect = obj.n_bisect_solves = 0
    pts = [_node(x) for x in (0.0, 0.3, 0.5, 1.3)]
    pts[3][2] = None                                   # an outer node without its nuisance (no secant start)
    rec = est._bisect_brackets(obj, [1e-3, 0.5, 1e-2, 1.0], 0, [1, 2, 3], pts, 0.0, THR, np.asarray(MU[1:]), None)
    up = rec["upper"]
    xs = [round(math.log10(st["value"] / 1e-3), 9) for st in up["steps"]]
    assert xs == [0.9, 0.7, 1.1], xs
    assert len(up["polish"]) == 1 and up["polish"][0]["role"] == "outer"
    assert up["polish"][0]["ll"] - up["polish"][0]["ll_before"] == pytest.approx(3.0)
    assert round(math.log10(up["polish"][0]["value"] / 1e-3), 9) == 0.9
    assert up["width_dex"] == pytest.approx(0.2) and not up["capped"] and rec["lower"]["steps"] == []
    assert abs(math.log10(_edge(pts, "upper") / 1e-3) - 1.0) < 0.01
    assert obj.n_bisect == 3


@pytest.mark.sut
def test_inner_polish_runs_without_the_hazard_only_when_switched_on(est):
    """J (no hazard): the node just inside the upper edge carries a stalled solve (1 nat low, nuisance off its
    optimum). `_polish_edges` re-solves it only with obj.inner_polish (1.0.0: never, the gate was _h's SHADOW)."""
    import numpy as np
    out = {}
    for on in (False, True):
        obj = _analytic_obj(est)
        if on:
            obj.inner_polish = True
            obj.n_inner_polish = obj.n_inner_polish_improved = 0
        pts = [_node(x) for x in (-1.6, -0.6, -0.3, 0.0, 0.3, 0.6, 1.6)]
        pts[5] = [pts[5][0], pts[5][1] - 1.0, [z + 0.5 for z in pts[5][2]]]
        sides = {"down": {"reason": "dev_stop"}, "up": {"reason": "dev_stop"}}
        est._polish_edges(obj, [1e-3, 0.5, 1e-2, 1.0], 0, "eta", [1, 2, 3], pts, 0.0, THR, np.asarray(MU[1:]),
                          sides, 24)
        out[on] = (pts[5][1], getattr(obj, "n_inner_polish", None), getattr(obj, "n_inner_polish_improved", None))
    assert out[False][1] is None and out[False][0] == pytest.approx(_node(0.6)[1] - 1.0)       # untouched
    assert out[True][1] > 0 and out[True][2] == 1 and out[True][0] == pytest.approx(_node(0.6)[1], abs=1e-6)


# ----------------------------------------------------------------------------------------------- EXTENT_Z

def test_sc_extents_store_z_is_the_exact_row(est):
    import numpy as np
    rng = np.random.default_rng(3)
    box = est.box_for(300.0)
    Z = np.log(np.array([[1e-3, 0.5, 1e-2, 1.0]])) + rng.normal(0, 0.3, (200, 4))
    Z[:, 3] += rng.integers(-5, 6, 200) * np.spacing(np.abs(Z[:, 3]))       # ulp-scale neighbours
    LL = -rng.exponential(1.0, 200)
    args = (Z, LL, ("eta", "beta", "epsilon", "kappa"), box, -0.1, THR)
    off, on = est.sc_extents(*args), est.sc_extents(*args, store_z=True)
    assert set(on) == set(off)
    for p, e in on["extents"].items():
        e2 = dict(e)
        zmin, zmax = e2.pop("z_qmin"), e2.pop("z_qmax")
        assert e2 == off["extents"][p]
        k = ("eta", "beta", "epsilon", "kappa").index(p)
        rows = [j for j in range(200) if np.array_equal(Z[j], np.asarray(zmin))]
        TH = np.exp(Z)                                                       # theta = exp of the whole archive
        assert rows and TH[rows[0], k] == e["qmin"] and list(TH[rows[0]]) == e["theta_qmin"]
        for zq in (zmin, zmax):
            back = json.loads(json.dumps(zq))                                # the store's JSON keeps every bit
            assert np.asarray(back, float).tobytes() == np.asarray(zq, float).tobytes()
            assert any(np.asarray(zq, float).tobytes() == Z[j].tobytes() for j in range(200))
    assert all("z_qmin" not in e for e in off["extents"].values())


# ----------------------------------------------------------------------------------------------- BOX_FLAG

def test_box_status_is_on_the_edge_within_two_ulps_and_h_floor_is_exempt(est):
    import numpy as np
    from types import SimpleNamespace
    box = dict(est.box_for(300.0), h=(1e-8, 0.1))
    obj = SimpleNamespace(params=est.PARAMS_SR + ("h",), box=box)
    zb = math.log(1e3)
    z = np.log([1e-3, 1e3, 1e-2, 1.0, 1e-4])
    assert est._box_status(obj, z)["on"] == ["beta@hi"]
    for k in (-2, -1, 1, 3):                                   # 2 ulps inside, 1 inside, beyond (out of box)
        z2 = z.copy()
        z2[1] = zb + k * np.spacing(zb)
        assert est._box_status(obj, z2)["on"] == ["beta@hi"], k
    z3 = z.copy()
    z3[1] = zb - 3 * np.spacing(zb)
    s3 = est._box_status(obj, z3)
    assert s3["on"] == [] and s3["near"]["beta"][0] == "hi" and 0 < s3["near"]["beta"][1] < 1e-14
    z4 = np.log([1e-12, 2.0, 1e-2, 1e-3, 1e-8])                # eta, kappa on their floors; h on its floor
    s4 = est._box_status(obj, z4)
    assert s4["on"] == ["eta@lo", "kappa@lo"] and s4["h_at_floor"] is True
    assert est._box_status(obj, np.log([1e-3, 2.0, 1e-2, 1.0, 0.1]))["on"] == ["h@hi"]
    assert s4["near"]["beta"] == ["hi", pytest.approx(math.log10(1e3 / 2.0))]


def test_box_limited_block_picks_the_point_that_set_each_edge(ad):
    st_on = {"on": ["beta@hi"], "near": {}, "h_at_floor": False}
    st_off = {"on": [], "near": {}, "h_at_floor": False}

    def prof(a, b):
        return {"nodes": [{"role": "inner", "value": 1.0, "ll": 0.0, "status": a},
                          {"role": "outer", "value": 2.0, "ll": -3.0, "status": b}], "width_dex": 0.3,
                "straddles": True}
    bp = {"ulps": 2.0, "params": {
        "eta": {"lower": {"profile": prof(st_off, st_off), "sc": None},
                "upper": {"profile": prof(st_off, st_on), "sc": {"value": 3.0, "ll": -1.0, "status": st_off}}},
        "epsilon": {"lower": {"profile": prof(st_off, st_off), "sc": None},
                    "upper": {"profile": prof(st_off, st_off), "sc": {"value": 9.0, "ll": -1.5, "status": st_on}}},
        "beta": {"lower": {"profile": None, "sc": None}, "upper": {"profile": None, "sc": None}},
        "kappa": {"lower": {"profile": prof(st_on, None), "sc": None}, "upper": {"profile": None, "sc": None}}}}
    res = {"diagnostics": {"box_points": bp}}
    claims = [{"parameter": "eta", "shape": "closed", "lower": 0.5, "upper": 1.5},
              {"parameter": "epsilon", "shape": "closed", "lower": 0.5, "upper": 9.0},
              {"parameter": "beta", "shape": "none"},
              {"parameter": "kappa", "shape": "lower_bound", "lower": 0.01}]
    sc_block = {"widening": [{"parameter": "eta", "lower_moved_dex": 0.0, "upper_moved_dex": 0.0, "opened": []},
                             {"parameter": "epsilon", "lower_moved_dex": 0.0, "upper_moved_dex": 0.778, "opened": []},
                             {"parameter": "beta", "lower_moved_dex": 0.0, "upper_moved_dex": 0.0, "opened": []},
                             {"parameter": "kappa", "lower_moved_dex": 0.0, "upper_moved_dex": 0.7, "opened": ["upper"]}]}
    out = ad.box_limited_block(res, claims, sc_block)
    P = out["params"]
    assert P["eta"]["upper"]["setter"] == "profile" and P["eta"]["upper"]["flag"]          # the outer node: beta@hi
    assert P["eta"]["lower"]["flag"] is False
    assert P["epsilon"]["upper"]["setter"] == "self_consistency" and P["epsilon"]["upper"]["flag"]
    assert P["epsilon"]["upper"]["edge"] == 9.0 and P["epsilon"]["lower"]["setter"] == "profile"
    assert P["beta"] == {"lower": None, "upper": None}
    assert P["kappa"]["upper"] is None and P["kappa"]["lower"]["flag"] and P["kappa"]["lower"]["known"] is False
    assert out["flagged"] == ["eta:upper", "epsilon:upper", "kappa:lower"] and out["n_flagged"] == 3
    assert ad.box_limited_block({"diagnostics": {}}, claims, sc_block) is None
    # no widening block (SELF_CONSISTENT off): every edge is the profile's
    out2 = ad.box_limited_block(res, claims, None)
    assert out2["params"]["epsilon"]["upper"]["setter"] == "profile" and not out2["params"]["epsilon"]["upper"]["flag"]
    # the would-be h claim, with its own widening record
    bp["params"]["h"] = {"lower": {"profile": prof(st_off, st_off), "sc": None},
                         "upper": {"profile": prof(st_off, st_off), "sc": {"value": 0.5, "ll": -1, "status": st_on}}}
    res["diagnostics"]["hazard"] = {"h_claim": {"parameter": "h", "shape": "closed", "lower": 0.1, "upper": 0.5},
                                    "h_claim_sc": {"parameter": "h", "lower_moved_dex": 0.0, "upper_moved_dex": 0.2,
                                                   "opened": []}}
    out3 = ad.box_limited_block(res, claims, sc_block)
    assert out3["params"]["h"]["upper"]["setter"] == "self_consistency" and "h:upper" in out3["flagged"]


# ----------------------------------------------------------------------------------------------- a real fit

def _tiny():
    import numpy as np
    rng = np.random.default_rng(1)
    u = rng.random(80)
    t = np.floor(np.log(1 - np.log(u) / 0.05) / 0.3 / 0.5) * 0.5 + 0.5
    e = np.ones_like(t, dtype=int)
    e[t > 12] = 0
    return np.minimum(t, 12.0), e


def _strip(r, drop=()):
    d = {k: v for k, v in r["diagnostics"].items() if not k.endswith("_s") and k not in drop}
    return json.dumps(dict(theta=r["theta"], ci=r["ci"], loglik=r["loglik"], d=d), sort_keys=True, default=str)


KW_TINY = dict(N=40, dt=0.02)


@pytest.fixture(scope="module")
def tiny_fits(est):
    """Three tiny real fits (the self-consistency test's data, solver N=40), shared by the tests below:
    base = self_consistent only (1.0.0's switches); bf = + box_flag; all = + box_flag, extent_z, inner_polish,
    bracket_bisect (1.0.1.dev1's estimator switches)."""
    fj = T.pkg_module("fp_jax")
    t, e = _tiny()
    f = lambda **k: est.fit(fj, t, e, delta=0.5, solver_kw=KW_TINY, self_consistent=True, **k)  # noqa: E731
    return dict(fj=fj, t=t, e=e, base=f(), bf=f(box_flag=True),
                all=f(box_flag=True, extent_z=True, inner_polish=True, bracket_bisect=True))


@pytest.mark.sut
def test_real_fit_box_flag_changes_only_its_diagnostics(tiny_fits, ad):
    """BOX_FLAG makes no likelihood call and changes nothing but diagnostics.box_points -- the extents included
    (its exact z is read and dropped without EXTENT_Z); every bracket it names straddles its edge, and
    box_limited_block covers exactly the final claims' stated edges. No claim changes."""
    r0, r1 = tiny_fits["base"], tiny_fits["bf"]
    assert "box_points" not in r0["diagnostics"]
    assert _strip(r0) == _strip(r1, ("box_points",))
    assert all("z_qmin" not in x for x in r1["diagnostics"]["self_consistency"]["extents"].values())
    bp = r1["diagnostics"]["box_points"]
    for p, ent in bp["params"].items():
        for side in ("lower", "upper"):
            pr = ent[side]["profile"]
            lo, hi = r1["ci"][f"{p}@0.95"]
            edge = lo if side == "lower" else hi
            assert (pr is None) == (not math.isfinite(edge)), (p, side)
            if pr is not None:
                assert pr["straddles"], (p, side, pr)
    claims, block = ad.final_claim_kwargs(r1, True)
    assert claims == ad.final_claim_kwargs(r0, True)[0]
    bl = ad.box_limited_block(r1, claims, block)
    stated = {(c["parameter"], s) for c in claims for s in ("lower", "upper") if s in c}
    assert {(p, s) for p, ent in bl["params"].items() for s, v in ent.items() if v is not None} == stated


@pytest.mark.sut
def test_real_fit_extent_z_is_the_memo_key(tiny_fits, est):
    """EXTENT_Z: each extent's z_qmin / z_qmax gives theta = exp(z), and re-evaluating at it on a FRESH objective
    gives the recorded log-likelihood bit for bit (one of the two compiled programs, as the archive holds both)."""
    import numpy as np
    r = tiny_fits["all"]
    fj, t, e = tiny_fits["fj"], tiny_fits["t"], tiny_fits["e"]
    sc = r["diagnostics"]["self_consistency"]
    L, gkw, _c = est.solver_grid(t, e, 0.5)
    o = est._Objective(fj, t, e, L, dict(gkw, **KW_TINY), est.box_for(300.0))
    n = 0
    for p, ext in sc["extents"].items():
        if not ext.get("n"):
            continue
        for side in ("qmin", "qmax"):
            z = np.asarray(ext[f"z_{side}"], float)
            assert np.allclose(np.exp(z), ext[f"theta_{side}"], rtol=1e-15, atol=0)   # exp (SIMD may differ 1 ulp)
            assert ext[f"ll_{side}"] in (float(o.f(z)), float(o.vg(z)[0])), (p, side)
            n += 1
    assert n >= 6


@pytest.mark.sut
def test_real_fit_inner_polish_and_bisection_leave_no_wide_bracket(tiny_fits, ad):
    import numpy as np
    r0, r1 = tiny_fits["bf"], tiny_fits["all"]
    d0, d1 = r0["diagnostics"], r1["diagnostics"]
    assert "inner_polish" not in d0 and "bracket_bisect" not in d0
    assert d1["inner_polish"]["enabled"] and d1["inner_polish"]["n_solves"] > 0
    assert d1["inner_polish"]["already_on_via_shadow"] is False
    bb = d1["bracket_bisect"]
    assert bb["dex"] == 0.3 and set(bb["params"]) == {"eta", "beta", "epsilon", "kappa"}
    for p, ent in d1["box_points"]["params"].items():
        for side, s in ent.items():
            r = bb["params"][p][side]
            if s["profile"] is not None:
                failed = any(st.get("failed") for st in r["steps"])
                assert s["profile"]["width_dex"] <= 0.3 + 1e-12 or r["capped"] or failed, (p, side, s["profile"])
                assert s["profile"]["straddles"]
            if r["width_dex_before"] is not None and r["width_dex_before"] > 0.3:
                assert r["steps"], (p, side, r)                                      # a wide bracket was bisected
    print(f"\nBRACKET_BISECT tiny fit: {bb['n_nodes']} midpoint nodes, {bb['n_solves']} solves; inner polish "
          f"{d1['inner_polish']['n_solves']} solves, {d1['inner_polish']['n_improved']} improved; n_grad "
          f"{d0['n_grad']} -> {d1['n_grad']}")
    for c in ad.final_claim_kwargs(r1, True)[0]:
        assert c["shape"] in ("closed", "lower_bound", "upper_bound", "none")
    assert np.isfinite(r1["loglik"]) and r1["loglik"] >= r0["loglik"] - 1e-9
