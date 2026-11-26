"""1.0.1, THE FREEZE (the tester's spec 002; Naveh's decision of 2026-10-18, "Yes, at the freeze"): 1.0.1.dev5 plus ONE
ClaimSet.diagnostics key, `estimate` (ESTIMATE_KEY, `estimate_block`), ADAPTER-ONLY.

The oracle is 1.0.1.dev5's adapter, kept VERBATIM in fixtures/adapter_dev5/adapter_dev5.py (= git
258146a:methods/mle_profile_jax2/__init__.py; sha256 cd454d06..., the dev5 lock's __init__.py entry), with dev5's
VERSION_LOCK.json beside it (VERSION_LOCK_dev5.json). Loaded under a TEST key, it registers nothing.

  * provenance: the oracle's bytes are dev5's (its sha256 is dev5's lock entry; = git 258146a where git has it);
  * ONLY THE ADAPTER CHANGED: estimator.py / fp_jax.py / observation_model.py have dev5's sha256; every top-level
    statement of __init__.py has dev5's SOURCE except the allowed set (the module docstring only APPENDS; VERSIONS all
    1.0.1; the new `import copy`, ESTIMATE_KEY, ESTIMATE_SCHEMA, estimate_block; claimset = dev5's plus the one line that
    adds the key);
  * the claimset of every name == dev5's claimset + exactly diagnostics.estimate, on fake results covering every
    best_point branch and the "all starts failed" path; `res` is never modified; the block's content is read from `res`;
  * tiny real fits (estimator.fit, solver N = 40: J, H, J-kfix, H-kfix): the same, plus best_point and theta_hat
    checked against the captured evaluation archive;
  * calibrate end to end, in subprocesses, through a dev5 TREE (the oracle + this package's dev5-byte siblings + dev5's
    lock) and through this package: run_fit's result BIT FOR BIT and the ClaimSet up to the added key, for J, H and two
    kfix names (the MLE stage on a tiny grid; the full fit on the production grid is `slow`).
"""
from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import json
import math
import os
import shutil
import subprocess

import pytest

import _jax2_testlib as T

ORACLE_DIR = os.path.join(T.TESTS, "fixtures", "adapter_dev5")
ORACLE = os.path.join(ORACLE_DIR, "adapter_dev5.py")
ORACLE_LOCK = os.path.join(ORACLE_DIR, "VERSION_LOCK_dev5.json")
DEV5_COMMIT = "258146ac1a4aa1a41bc71ed03333806255b66b7b"
DEV5_SHA = {                                                    # 1.0.1.dev5's VERSION_LOCK.json (258146a)
    "__init__.py": "cd454d06e0def405ced19decb22f3b49868dfc3e9dea6eae688c93d886a1544e",
    "estimator.py": "e0335a5366237f5357fc31bbe5ea0e039b94ddca2d7dfd5fe077d6c0eb9a5ee2",
    "fp_jax.py": "e5b9db257789b252193ef5b402174b3e5ada516808af532164e4b4de3a55e253",
    "observation_model.py": "deff521c80985a699f513758c494d9e0f3e8ab0fdf5cc97da292b342025d25d0",
}
NAMES = ("mle_profile_jax2", "mle_profile_jax2_h", "mle_profile_jax2_kfix_k0p001", "mle_profile_jax2_kfix_k0p5",
         "mle_profile_jax2_kfix_k1", "mle_profile_jax2_h_kfix_k0p001", "mle_profile_jax2_h_kfix_k0p5",
         "mle_profile_jax2_h_kfix_k1")
KEY = "estimate"
#: what the freeze may change in __init__.py, by top-level name (everything else: dev5's source, character for character)
ALLOWED_NEW = {"ESTIMATE_KEY", "ESTIMATE_SCHEMA", "estimate_block", "import copy"}
TIMING = frozenset({"wall_s", "compile_s", "seed_s", "mle_s", "cpu_s", "refit_cpu_s", "fit_wall_s",
                    "wall_clock_seconds", "cpu_seconds"})


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def strip(o):
    """JSON-ready copy without the timing keys (the only fields two identical computations may differ in)."""
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in TIMING}
    if isinstance(o, (list, tuple)):
        return [strip(v) for v in o]
    if hasattr(o, "tolist") and not isinstance(o, (str, bytes)):
        return strip(o.tolist())
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return o.item()
    return o


def canon(o):
    return json.dumps(strip(o), sort_keys=True, separators=(",", ":"), default=repr)


@pytest.fixture(scope="module")
def ad():
    """This package's adapter under a TEST key (it registers nothing there)."""
    m = T.load("jax2_tests.freeze.adapter", os.path.join(T.PKG, "__init__.py"))
    assert m.__name__ != m._KEY
    return m


@pytest.fixture(scope="module")
def dev5():
    """1.0.1.dev5's adapter (the oracle) under a TEST key."""
    m = T.load("jax2_tests.freeze.adapter_dev5", ORACLE)
    assert m.__name__ != m._KEY and set(m.VERSIONS.values()) == {"1.0.1.dev5"}
    return m


# ----------------------------------------------------------------------------------------------- provenance / scope

def test_oracle_is_dev5s_bytes():
    assert _sha(ORACLE) == DEV5_SHA["__init__.py"]
    lock = json.load(open(ORACLE_LOCK))
    assert {n: e["version"] for n, e in lock["names"].items()} == {n: "1.0.1.dev5" for n in NAMES}
    assert all(e["sha256"] == DEV5_SHA for e in lock["names"].values())
    for rel, local in (("__init__.py", ORACLE), ("VERSION_LOCK.json", ORACLE_LOCK)):
        r = subprocess.run(["git", "-C", T.ROOT, "show", f"{DEV5_COMMIT}:methods/mle_profile_jax2/{rel}"],
                           capture_output=True)
        if r.returncode != 0:
            pytest.skip(f"commit {DEV5_COMMIT[:7]} is not in this checkout's git (the sha256 above still holds)")
        assert r.stdout == open(local, "rb").read(), rel


def test_versions_are_1_0_1_and_the_lock_holds():
    from version_lock import problems, versions
    assert versions() == {n: "1.0.1" for n in NAMES}
    assert problems() == []


def test_only_the_adapter_changed():
    """estimator / fp_jax / observation_model are dev5's bytes; __init__.py is dev5's source statement for statement,
    except: the docstring only APPENDS, VERSIONS is 1.0.1 throughout, ALLOWED_NEW is new, and claimset gains exactly
    the one line that adds diagnostics.estimate. So run_fit, calibrate and every claim builder are dev5's."""
    for f in ("estimator.py", "fp_jax.py", "observation_model.py"):
        assert _sha(os.path.join(T.PKG, f)) == DEV5_SHA[f], f
    new_src = open(os.path.join(T.PKG, "__init__.py")).read()
    old_src = open(ORACLE).read()

    def parts(src):
        tree = ast.parse(src)
        out = {}
        for node in tree.body:
            seg = ast.get_source_segment(src, node)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                key = node.name
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                key = node.targets[0].id
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                key = node.target.id
            elif isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) \
                    and node is tree.body[0]:
                key = "__doc__"
            else:
                key = seg                                            # imports, the registration guard, ...
            assert key not in out, key
            out[key] = seg
        return out, tree
    new, new_tree = parts(new_src)
    old, old_tree = parts(old_src)
    assert set(old) <= set(new), sorted(set(old) - set(new))                          # nothing removed
    assert set(new) - set(old) == ALLOWED_NEW, sorted(set(new) - set(old))
    changed = sorted(k for k in old if old[k] != new[k])
    assert changed == ["VERSIONS", "_MleProfileJax2Base", "__doc__"], changed
    assert ast.get_docstring(new_tree).startswith(ast.get_docstring(old_tree))          # history appended only
    vn = ast.literal_eval(new_tree.body[[k for k in new].index("VERSIONS")].value)
    vo = ast.literal_eval(old_tree.body[[k for k in old].index("VERSIONS")].value)
    assert set(vn) == set(vo) == set(NAMES) and set(vn.values()) == {"1.0.1"} and set(vo.values()) == {"1.0.1.dev5"}

    def methods(src, tree):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_MleProfileJax2Base")
        return {getattr(n, "name", None) or ast.get_source_segment(src, n): ast.get_source_segment(src, n)
                for n in cls.body}
    mn, mo = methods(new_src, new_tree), methods(old_src, old_tree)
    assert set(mn) == set(mo)
    assert sorted(k for k in mo if mo[k] != mn[k]) == ["claimset"]
    diff = [ln for ln in difflib.ndiff(mo["claimset"].splitlines(), mn["claimset"].splitlines())
            if ln[:2] in ("- ", "+ ")]
    assert len(diff) == 1 and diff[0].startswith("+ ") and diff[0][2:].strip().startswith(
        "diag[ESTIMATE_KEY] = estimate_block(res, self.HAZARD, self.KAPPA0)"), diff


# ----------------------------------------------------------------------------------------------- fake results

def _ext(lo, hi, th_lo, th_hi, ll_lo, ll_hi, z=False):
    e = {"n": 5, "qmin": lo, "qmax": hi, "theta_qmin": th_lo, "theta_qmax": th_hi, "ll_qmin": ll_lo, "ll_qmax": ll_hi}
    if z:
        e["z_qmin"] = [math.log(x) for x in th_lo]
        e["z_qmax"] = [math.log(x) for x in th_hi]
    return e


def _fake(case, hazard=False, kappa0=None):
    """A run_fit-shaped result. case: 'stale' (sc.best recorded), 'exact' (gap 0), 'extent' (0 < gap <= 0.01, an
    extent point at ll_best), 'unrecorded' (0 < gap <= 0.01, no point at ll_best), 'failed' (all starts failed)."""
    k = 0.3 if kappa0 is None else kappa0
    th = [1e-3, 0.5, 1e-2, k] + ([0.02] if hazard else [])
    P = ["eta", "beta", "epsilon", "kappa"] + (["h"] if hazard else [])
    if case == "failed":
        dg = {"error": "all starts failed", "dt_floor": {"binds": False, "dl": 0.05}, "exc_counts": {}}
        if kappa0 is not None:
            dg["kfix"] = {"kappa_fixed": kappa0, "n_components": None}
        return {"theta": th, "ci": {}, "loglik": float("nan"), "wall_s": 1.0, "diagnostics": dg}
    ll_hat = -100.0
    gap = {"stale": 0.5, "exact": 0.0, "extent": 0.004, "unrecorded": 0.004}[case]
    ll_best = ll_hat + gap
    near = [x * 1.01 for x in th]
    far = [x * 0.97 for x in th]
    ext = {p: _ext(far[i], near[i], far, near, ll_best - 0.3, ll_best if case == "extent" else ll_best - 0.2,
                   z=True) for i, p in enumerate(P)}
    sc = {"thr": 3.841458820694124, "half_thr": 1.920729410347062, "ll_hat": ll_hat, "ll_best": ll_best,
          "gap_nats": gap, "stale": "none" if gap <= 0.01 else "minor", "best": None, "extents": ext}
    if case == "stale":
        sc["best"] = {"ll": ll_best, "theta": [x * 1.02 for x in th]}
    ci = {f"{p}@0.95": (x * 0.5, x * 2.0) for p, x in zip(P, th) if not (kappa0 is not None and p == "kappa")}
    verdicts = {p: "identified" for p in P if not (kappa0 is not None and p == "kappa")}
    dg = {"verdicts": verdicts, "self_consistency": sc, "profiles": {"eta": []},
          "solver": {"N": 900, "dt": 0.004, "tau_max": 2.0, "Gtol": 1e-11, "scheme": "bdf2", "sampling_dt": 1.0,
                     "early": {"dt0": 1e-8, "ratio": 1.05}, "exact_from_F": True},
          "dt_floor": {"binds": False, "dl": 0.0125, "grid_rule": {"n": 80, "cv": 0.31, "cv_c": 0.3098,
                                                                   "dt_in_scope": False, "dt_cv_max": 0.05,
                                                                   "dt_k": 80.0, "dt_cap": None, "N": 900,
                                                                   "dt": 0.004}},
          "exc_counts": {}}
    if kappa0 is not None:
        dg["kfix"] = {"kappa_fixed": kappa0, "n_components": {"eta": 1, "beta": 1, "epsilon": 1}}
        dg["combo_grid"] = {"sqrt_eta_over_eps": {"kind": "walk", "centre": math.sqrt(th[0]) / th[2],
                                                  "ci": {"sqrt_eta_over_eps@0.95": [2.0, 5.0]},
                                                  "verdict": "identified", "n_components": 1}}
        sc["derived_extents"] = {"sqrt_eta_over_eps": _ext(1.8, 4.5, far, near, ll_best - 0.5, ll_best - 0.4)}
    return {"theta": th, "ci": ci, "loglik": ll_hat, "wall_s": 1.0, "diagnostics": dg}


def _both(ad, dev5, name, res):
    """(freeze ClaimSet dict, dev5 ClaimSet dict) for the SAME result, each through its own adapter's class."""
    from srbench.contract.types import Observation
    from srbench.provenance import CodeProvenance
    obs = Observation(cohort_id="freeze-fake", ages=(1.0, 2.0), events=(1, 1), delta=None)
    prov = CodeProvenance(package="SRK", resolved_path="/nowhere")
    new_cls = next(c for c in ad.CLASSES if c.NAME == name)
    old_cls = next(c for c in dev5.CLASSES if c.NAME == name)
    before = canon(res)
    a = new_cls().claimset(obs, copy.deepcopy(res), prov, 0.0, 0.0)
    r2 = copy.deepcopy(res)
    b = old_cls().claimset(obs, r2, prov, 0.0, 0.0)
    res_seen = copy.deepcopy(res)
    a2 = new_cls().claimset(obs, res_seen, prov, 0.0, 0.0)
    assert canon(res_seen) == before, "claimset modified run_fit's result"
    assert canon(a2.to_dict()) == canon(a.to_dict())
    return a.to_dict(), b.to_dict()


def _assert_dev5_plus_key(new, old):
    """THE FREEZE RULE: the ClaimSets are equal once diagnostics.estimate is removed from the freeze's, and dev5's has
    no such key. Claims compared first (the scored part), then everything."""
    assert canon(new["claims"]) == canon(old["claims"])
    assert KEY in new["diagnostics"] and KEY not in old["diagnostics"]
    assert set(new["diagnostics"]) == set(old["diagnostics"]) | {KEY}
    n2 = copy.deepcopy(new)
    n2["diagnostics"].pop(KEY)
    assert canon(n2) == canon(old)
    assert list(new["diagnostics"])[-1] == KEY                    # appended: dev5's keys keep their order
    return new["diagnostics"][KEY]


def _assert_estimate_from_res(est, res, hazard, kappa0):
    P = ["eta", "beta", "epsilon", "kappa"] + (["h"] if hazard else [])
    assert est["schema"] == 1 and est["params"] == P
    dg = res["diagnostics"]
    if math.isfinite(res["loglik"]):
        assert est["theta_hat"] == {p: float(v) for p, v in zip(P, res["theta"])}
        assert est["theta_hat_loglik"] == res["loglik"]
    else:
        assert est["theta_hat"] is None and math.isnan(est["theta_hat_loglik"])
    assert est["pinned"] == ({"kappa": kappa0} if kappa0 is not None else {})
    if kappa0 is not None and est["theta_hat"] is not None:
        assert est["theta_hat"]["kappa"] == kappa0                   # exactly
    sc = dg.get("self_consistency")
    if sc is None:
        assert est["best_point"] is None
    else:
        bp = est["best_point"]
        assert bp["loglik"] == sc["ll_best"] >= sc["ll_hat"] == res["loglik"]
        assert bp["gap_nats"] == sc["gap_nats"] and bp["stale"] == sc["stale"]
    g = est["grid"]
    assert g["solver"] == dg.get("solver") and g["grid_rule"] == (dg.get("dt_floor") or {}).get("grid_rule")
    assert g["dl"] == (dg.get("dt_floor") or {}).get("dl") and g["floor_binds"] == (dg.get("dt_floor") or {}).get("binds")
    json.dumps(est, allow_nan=True)                               # serialisable as the store writes records


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("case", ["stale", "exact", "extent", "unrecorded", "failed"])
def test_claimset_is_dev5s_plus_only_the_estimate_key(ad, dev5, name, case):
    by = {c.NAME: c for c in ad.CLASSES}
    cls = by[name]
    res = _fake(case, hazard=cls.HAZARD, kappa0=cls.KAPPA0)
    new, old = _both(ad, dev5, name, res)
    est = _assert_dev5_plus_key(new, old)
    _assert_estimate_from_res(est, res, cls.HAZARD, cls.KAPPA0)
    bp = est["best_point"]
    P = est["params"]
    if case == "failed":
        assert bp is None
    elif case == "stale":
        assert bp["source"] == "self_consistency.best" and bp["z"] is None
        assert bp["theta"] == dict(zip(P, res["diagnostics"]["self_consistency"]["best"]["theta"]))
    elif case == "exact":
        assert bp["source"].startswith("theta_hat") and bp["theta"] == est["theta_hat"]
        assert "assumes ll_hat is the likelihood at theta_hat" in bp["source"]     # the r.fun / r.x caveat
    elif case == "extent":
        ext = res["diagnostics"]["self_consistency"]["extents"]
        first = sorted(ext)[0]                                    # extents are scanned in sorted order, qmin first
        assert bp["source"] == f"self_consistency.extents.{first}.theta_qmax"
        assert bp["theta"] == dict(zip(P, ext[first]["theta_qmax"])) and bp["z"] == ext[first]["z_qmax"]
    else:
        assert bp["theta"] is None and bp["z"] is None and bp["source"].startswith("not recorded")
        assert "in log-likelihood only" in bp["source"]            # not a closeness in parameters


def test_estimate_block_is_a_pure_function_of_res(ad):
    res = _fake("stale", hazard=True)
    before = canon(res)
    a = ad.estimate_block(res, True)
    b = ad.estimate_block(res, True)
    assert canon(res) == before and canon(a) == canon(b)
    a["grid"]["solver"]["N"] = -1                                 # a copy, not a reference
    assert res["diagnostics"]["solver"]["N"] == 900


# ----------------------------------------------------------------------------------------------- tiny real fits

def _tiny():
    import numpy as np
    rng = np.random.default_rng(1)
    u = rng.random(80)
    t = np.floor(np.log(1 - np.log(u) / 0.05) / 0.3 / 0.5) * 0.5 + 0.5
    e = np.ones_like(t, dtype=int)
    e[t > 12] = 0
    return np.minimum(t, 12.0), e


KW_TINY = dict(N=40, dt=0.02)
SW = dict(self_consistent=True, inner_polish=True, bracket_bisect=True, extent_z=True, n_rule=True, dt_rule=True)
#   1.0.1's estimator switches; the grid is overridden (n_rule / dt_rule then only add the grid_rule record; early_all
#   is left out because the overridden grid would keep its graded mesh, ~3x the march for nothing tested here)


def _memo_seeds(FC, memo):
    """Memoise SRK.fp_curvefit's two seeds (fp_curvefit_seed, fp_curvefit_makeham): deterministic numpy curve fits of
    ~20 s each that every fit of the SAME data repeats. The estimator imports them inside `fit` / `_makeham_seed`, so
    the module attributes are what it calls. A deep copy per call: the caller may annotate what it gets.
    Returns the originals (to restore)."""
    import numpy as np
    real = (FC.fp_curvefit_seed, FC.fp_curvefit_makeham)

    def wrap(fn, tag):
        def w(t, e, **kw):
            k = (tag, np.asarray(t, float).tobytes(), np.asarray(e, int).tobytes(), tuple(sorted(kw.items())))
            if k not in memo:
                memo[k] = fn(t, e, **kw)
            return copy.deepcopy(memo[k])
        return w
    FC.fp_curvefit_seed = wrap(real[0], "seed")
    FC.fp_curvefit_makeham = wrap(real[1], "makeham")
    return real


@pytest.fixture(scope="module")
def tiny_fits():
    """Four tiny real fits (80 records on a 0.5 grid, solver N = 40) with the evaluation archive captured:
    J, H (SHADOW on), J-kfix at 1e-3 and H-kfix at 0.5 (both with the sqrt(eta)/eps walk). SRK's curve-fit seeds are
    computed once (`_memo_seeds`)."""
    T.jax_env()
    est = T.pkg_module("estimator")
    fj = T.pkg_module("fp_jax")
    import SRK.fp_curvefit as FC
    t, e = _tiny()
    real = est._sc_archive
    real_seeds = _memo_seeds(FC, {})
    out = {}

    def run(key, name, **kw):
        seen = {}

        def spy(obj):
            Z, LL = real(obj)
            seen.update(Z=Z.copy(), LL=LL.copy())
            return Z, LL
        est._sc_archive = spy
        try:
            r = est.fit(fj, t, e, delta=0.5, solver_kw=KW_TINY, **SW, **kw)
        finally:
            est._sc_archive = real
        out[key] = (name, r, seen)
    try:
        run("J", "mle_profile_jax2")
        run("H", "mle_profile_jax2_h", hazard=True, shadow=True)
        run("Jk", "mle_profile_jax2_kfix_k0p001", kappa_fixed=1e-3, combos=("sqrt_eta_over_eps",))
        run("Hk", "mle_profile_jax2_h_kfix_k0p5", kappa_fixed=0.5, combos=("sqrt_eta_over_eps",), hazard=True,
            shadow=True)
    finally:
        FC.fp_curvefit_seed, FC.fp_curvefit_makeham = real_seeds
    return out


@pytest.mark.sut
@pytest.mark.parametrize("key", ["J", "H", "Jk", "Hk"])
def test_real_tiny_fit_claimset_is_dev5s_plus_the_estimate(tiny_fits, ad, dev5, key):
    import numpy as np
    name, res, seen = tiny_fits[key]
    cls = next(c for c in ad.CLASSES if c.NAME == name)
    assert math.isfinite(res["loglik"]) and res["ci"]
    new, old = _both(ad, dev5, name, res)
    est = _assert_dev5_plus_key(new, old)
    _assert_estimate_from_res(est, res, cls.HAZARD, cls.KAPPA0)
    Z, LL = seen["Z"], seen["LL"]
    sc = res["diagnostics"]["self_consistency"]
    assert sc["ll_best"] == sc["ll_hat"] or sc["ll_best"] in set(LL.tolist())    # ll_best is a real evaluation
    bp = est["best_point"]
    P = est["params"]
    if bp["theta"] is not None:                                   # the identified best point IS an archived point ...
        th = np.array([bp["theta"][p] for p in P], float)
        rows = np.flatnonzero(LL == bp["loglik"])                 # ... AT ll_best exactly, whatever the source: for
        assert rows.size, bp["source"]                            # "theta_hat (gap 0)" this is the branch's assumption
        assert any(np.allclose(np.exp(Z[j]), th, rtol=1e-12, atol=0.0) for j in rows), bp["source"]
        if cls.KAPPA0 is not None:                                # kfix: pinned.kappa is kappa0 exactly; an archive
            k_arch = math.exp(math.log(cls.KAPPA0))               # point carries exp(log(kappa0)) (an ulp off at 1e-3)
            assert est["pinned"]["kappa"] == cls.KAPPA0
            assert bp["theta"]["kappa"] == (cls.KAPPA0 if bp["source"].startswith("theta_hat") else k_arch)
    if bp["z"] is not None:                                       # ... and its z, when given, is the memo key exactly
        assert any(list(Z[j]) == bp["z"] for j in np.flatnonzero(LL == bp["loglik"]))
    print(f"\n{key}: best_point source {bp['source']!r}, gap {bp['gap_nats']:.3g}, stale {bp['stale']}")


# ----------------------------------------------------------------------------------------------- calibrate, end to end

#: run in a fresh interpreter per package TREE: `calibrate` of CAL_NAMES on a tiny Observation, run_fit wrapped to capture
#: its result. MODE "tiny" (the default suite): estimator.fit wrapped to the MLE stage on a tiny grid (solver_kw N 40, dt
#: 0.02, no graded mesh; the full fits' ClaimSets are compared in-process above); MODE "production" (`slow`): the full fit
#: on the class constants' own grid. Prints each run_fit result (canonical JSON) and ClaimSet (insertion order), timing
#: keys stripped.
_CAL = r"""
import sys, json, os, copy, importlib.util
pkg = sys.argv[1]
s = importlib.util.spec_from_file_location("jax2_tests.freeze.cal_adapter", os.path.join(pkg, "__init__.py"))
ad = importlib.util.module_from_spec(s); sys.modules[s.name] = ad; s.loader.exec_module(ad)
assert ad.__name__ != ad._KEY
assert ad.version_lock_problems() == [], ad.version_lock_problems()
import numpy as np
from srbench import npcompat
from srbench.contract.types import Observation
est = ad._sibling("estimator")
import pickle
import SRK.fp_curvefit as FC
SEED_CACHE = sys.argv[2]           # SRK's curve-fit seeds, computed by the first tree, reused by the second
memo = pickle.load(open(SEED_CACHE, "rb")) if os.path.exists(SEED_CACHE) else {}
n_memo0 = len(memo)
def wrap(fn, tag):
    def w(t, e, **kw):
        k = (tag, np.asarray(t, float).tobytes(), np.asarray(e, int).tobytes(), tuple(sorted(kw.items())))
        if k not in memo:
            memo[k] = fn(t, e, **kw)
        return copy.deepcopy(memo[k])
    return w
FC.fp_curvefit_seed = wrap(FC.fp_curvefit_seed, "seed")
FC.fp_curvefit_makeham = wrap(FC.fp_curvefit_makeham, "makeham")
MODE = "tiny"
if MODE == "tiny":
    orig = est.fit
    def fit(fj, ages, events, **kw):
        assert "solver_kw" not in kw, kw                     # run_fit never passes a grid
        return orig(fj, ages, events, solver_kw=dict(N=40, dt=0.02, early=None), stages=("mle",), **kw)
    est.fit = fit
got = []
orig_run = ad.run_fit
def run_fit(obs, **kw):
    r = orig_run(obs, **kw)
    got.append(copy.deepcopy(r))
    return r
ad.run_fit = run_fit
TIMING = {"wall_s", "compile_s", "seed_s", "mle_s", "cpu_s", "refit_cpu_s", "fit_wall_s", "wall_clock_seconds",
          "cpu_seconds"}
def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in TIMING}
    if isinstance(o, (list, tuple)):
        return [strip(v) for v in o]
    if isinstance(o, np.ndarray):
        return strip(o.tolist())
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o
rng = np.random.default_rng(1)
u = rng.random(80)
t_raw = np.floor(np.log(1 - np.log(u) / 0.05) / 0.3 / 0.5) * 0.5 + 0.5
obs = Observation(cohort_id="freeze-tiny", ages=tuple(float(x) for x in np.minimum(t_raw, 12.0)),
                  events=tuple(int(x <= 12) for x in t_raw), delta=0.5)
out = {}
for cls in ad.CLASSES:
    if cls.NAME not in NAMES:
        continue
    got.clear()
    cs = cls().calibrate(obs)
    assert cs.status == "ok", (cls.NAME, cs.status, cs.diagnostics)
    out[cls.NAME] = dict(result=json.dumps(strip(got[-1]), sort_keys=True, default=repr),
                         claimset=json.dumps(strip(cs.to_dict()), default=repr))     # insertion order kept
if not n_memo0:
    pickle.dump(memo, open(SEED_CACHE, "wb"))
print(json.dumps(dict(versions=sorted(set(ad.VERSIONS.values())), out=out, seeds_reused=n_memo0)))
"""
CAL_NAMES = ("mle_profile_jax2", "mle_profile_jax2_h", "mle_profile_jax2_kfix_k0p5", "mle_profile_jax2_h_kfix_k1")


def _dev5_tree(tmp_path):
    """A complete, lock-consistent 1.0.1.dev5 package: the oracle adapter + this package's siblings (dev5's bytes,
    checked) + dev5's lock. Its directory is named like the package, as the real one (the sibling keys)."""
    d = tmp_path / "dev5" / "mle_profile_jax2"
    d.mkdir(parents=True)
    shutil.copyfile(ORACLE, d / "__init__.py")
    shutil.copyfile(ORACLE_LOCK, d / "VERSION_LOCK.json")
    for f in ("estimator.py", "fp_jax.py", "observation_model.py"):
        assert _sha(os.path.join(T.PKG, f)) == DEV5_SHA[f], f
        shutil.copyfile(os.path.join(T.PKG, f), d / f)
    return str(d)


def _check_calibrate(tmp_path, full):
    code = _CAL.replace("NAMES", repr(CAL_NAMES))
    if full:
        code = code.replace('MODE = "tiny"', 'MODE = "production"')
    cache = repr(str(tmp_path / "srk_seeds.pkl"))
    old = T.run_py(code.replace("sys.argv[1]", repr(_dev5_tree(tmp_path))).replace("sys.argv[2]", cache), timeout=3600)
    new = T.run_py(code.replace("sys.argv[1]", repr(T.PKG)).replace("sys.argv[2]", cache), timeout=3600)
    assert old["_rc"] == 0, old["_stderr"]
    assert new["_rc"] == 0, new["_stderr"]
    assert old["seeds_reused"] == 0 and new["seeds_reused"] > 0          # the same SRK seeds on both sides
    assert old["versions"] == ["1.0.1.dev5"] and new["versions"] == ["1.0.1"]
    assert sorted(old["out"]) == sorted(new["out"]) == sorted(CAL_NAMES)
    for name in CAL_NAMES:
        o, n = old["out"][name], new["out"][name]
        assert n["result"] == o["result"], name                                    # run_fit: bit for bit
        res = json.loads(n["result"])
        _assert_dev5_plus_key(json.loads(n["claimset"]), json.loads(o["claimset"]))
        est = json.loads(n["claimset"])["diagnostics"][KEY]
        cls_h = "_h" in name
        k0 = {"mle_profile_jax2_kfix_k0p5": 0.5, "mle_profile_jax2_h_kfix_k1": 1.0}.get(name)
        _assert_estimate_from_res(est, res, cls_h, k0)
        assert ("overridden" in res["diagnostics"]["dt_floor"]) is (not full)
        if full:
            assert est["best_point"]["loglik"] >= est["theta_hat_loglik"]
        else:
            assert est["best_point"] is None and res["diagnostics"]["stages"] == ["mle"]   # no archive block
    return new


@pytest.mark.sut
def test_calibrate_is_dev5s_bit_for_bit_plus_the_estimate_tiny_grid(tmp_path):
    _check_calibrate(tmp_path, full=False)


@pytest.mark.slow
@pytest.mark.sut
def test_calibrate_on_the_production_grid_is_dev5s_bit_for_bit_plus_the_estimate(tmp_path):
    _check_calibrate(tmp_path, full=True)
