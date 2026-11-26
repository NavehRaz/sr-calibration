"""rc2 D1 (1.0.0.dev5) / E1 (1.0.0.dev6): EARLY_ALL -- a graded early-time mesh on EVERY grid with a sampling_dt.
dev6: the mesh is estimator._EARLY_ALL_MESH (1e-8, x1.05) everywhere, floor-bound grids included (one ratio); dev5
used I3's _FLOOR_EARLY (x1.02). With EARLY_ALL off the floor-bound grids keep _FLOOR_EARLY (dev4).

  * `_grid(early_all=False)` (and `solver_grid` / `floor_info`) is dev4's grid exactly, on every library v2 cohort;
    `early_all=True` adds the mesh (and dt_floor["early_all"]) and changes nothing else; where the floor binds it
    replaces dev4's _FLOOR_EARLY by _EARLY_ALL_MESH and changes nothing else; the graded schedule builds on every
    cohort's grid (subprocess: SRK + library);
  * exact-time data (no sampling_dt) keep the uniform grid (EXACT_F's density assumes it) and say so;
  * `estimator.fit(early_all=True)` marches the mesh and records its step count; off records nothing new;
  * the adapter: EARLY_ALL on for both names, `run_fit`'s default off (dev4's call), `calibrate` passes the constant,
    diagnostics.switches records it;
  * the defect it fixes on a NON-floor grid: pure-k0.1-p22-c1-s1's early deaths at its Phase A MLE and worst edge node
    (I3 section 8: 3.1 nats off on the production grid).
The full-size evidence (the likelihood-level check on the HEALTHY / HUMAN_OK / hazard-window records, identity to
dev4 with the switch off, CPU) is in reports/rc2/dev5/ (x1.02) and reports/rc2/dev6/ (x1.05, + C. elegans floor).
"""
from __future__ import annotations

import os

import pytest

import _jax2_testlib as T

pytestmark = pytest.mark.sut

MESH = {"dt0": 1e-8, "ratio": 1.05}            # dev6 (estimator._EARLY_ALL_MESH)
MESH_DEV5 = {"dt0": 1e-8, "ratio": 1.02}       # dev5's EARLY_ALL mesh = dev4's floor mesh (estimator._FLOOR_EARLY)


def test_grid_early_all_on_every_library_cohort():
    if not os.path.isdir(os.path.join(T.ROOT, "data", "library", "v2", "cohorts")):
        pytest.skip("library v2 cohorts are not built in this checkout")
    out = T.run_py("""
import sys, json, os, importlib.util, collections
import numpy as np
from srbench import npcompat
def load(k, p):
    s = importlib.util.spec_from_file_location(k, p); m = importlib.util.module_from_spec(s)
    sys.modules[k] = m; s.loader.exec_module(m); return m
P = os.path.join("methods", "mle_profile_jax2")
est = load("ea.est", os.path.join(P, "estimator.py")); fj = load("ea.fj", os.path.join(P, "fp_jax.py"))
from srbench.library.load import Library
lib = Library(version="v2")
DEV4 = dict(floor_early=True, n_rule=True)
n = bad = n_floor = 0; problems = []; ratios = []; n_exact = 0
for sp in lib.specs:
    t, e = lib.data(sp.cohort_id); lib._cache.clear()
    t = np.asarray(t, float); e = np.asarray(e, int); d = sp.delta if sp.delta else None
    g4 = est._grid(t, e, d, **DEV4)
    off = est._grid(t, e, d, early_all=False, **DEV4)
    on = est._grid(t, e, d, early_all=True, **DEV4)
    n += 1
    ok = repr(off) == repr(g4)                                         # off: dev4's grid exactly
    ok &= repr(est.solver_grid(t, e, d, early_all=False, **DEV4)) == repr(g4[:3])
    ok &= repr(est.floor_info(t, e, d, early_all=False, **DEV4)) == repr(g4[3])
    ok &= repr(est.solver_grid(t, e, d, early_all=True, **DEV4)) == repr(on[:3])
    ok &= repr(est.floor_info(t, e, d, early_all=True, **DEV4)) == repr(on[3])
    ok &= on[0] == g4[0] and on[2] == g4[2]                            # L, ceiling
    ok &= {k: v for k, v in on[1].items() if k != "early"} == {k: v for k, v in g4[1].items() if k != "early"}
    ok &= {k: v for k, v in on[3].items() if k not in ("early", "early_all")} == \\
          {k: v for k, v in g4[3].items() if k not in ("early", "early_all")}
    if d:
        M = est._EARLY_ALL_MESH
        ok &= on[1].get("early") == on[3].get("early") == M and on[3]["early_all"]["applied"] is True
        DT = fj.early_schedule(on[1]["dt"], on[1]["tau_max"], 20, M["dt0"], M["ratio"])[0]
        ratios.append(DT.size / int(on[1]["tau_max"] / on[1]["dt"]))
        if g4[3]["binds"]:
            n_floor += 1                                                   # dev4 marched FLOOR_EARLY here;
            ok &= g4[1].get("early") == g4[3].get("early") == est._FLOOR_EARLY   # dev6 replaces it (one ratio)
            ok &= repr({k: v for k, v in on[1].items() if k != "early"}) == \\
                  repr({k: v for k, v in g4[1].items() if k != "early"})
        else:
            ok &= "early" not in g4[1]
    else:
        n_exact += 1
        ok &= "early" not in on[1] and on[3]["early_all"]["applied"] is False
    if not ok:
        bad += 1; problems.append(sp.cohort_id)
r = np.asarray(ratios)
print(json.dumps(dict(n=n, bad=bad, problems=problems[:10], n_floor=n_floor, n_exact=n_exact, n_mesh=int(r.size),
                      ratio=[float(r.min()), float(np.median(r)), float(r.max())],
                      mesh=est._EARLY_ALL_MESH, floor_mesh=est._FLOOR_EARLY)))
""", timeout=3600)
    assert out["_rc"] == 0, out["_stderr"]
    assert out["n"] == 2290 and out["bad"] == 0, out["problems"]
    assert out["mesh"] == MESH and out["floor_mesh"] == MESH_DEV5
    assert out["n_floor"] == 48 and out["n_mesh"] == 2290 - out["n_exact"]
    print(f"\nEARLY_ALL census (2,290 v2 cohorts, mesh {out['mesh']}): mesh on {out['n_mesh']} grids, floor-bound {out['n_floor']}, "
          f"exact-time {out['n_exact']}; graded/uniform steps min/median/max {out['ratio']}")


def test_exact_time_keeps_the_uniform_grid():
    """No sampling_dt: EARLY_ALL applies no mesh (EXACT_F's `_density_F` assumes the uniform grid, and
    `make_loglik` refuses the combination) and records why."""
    out = T.run_py("""
import sys, json, os, importlib.util
import numpy as np
from srbench import npcompat
s = importlib.util.spec_from_file_location("ea.est2", os.path.join("methods", "mle_profile_jax2", "estimator.py"))
est = importlib.util.module_from_spec(s); sys.modules[s.name] = est; s.loader.exec_module(est)
rng = np.random.default_rng(5)
t = rng.gamma(20.0, 5.0, 300); e = np.ones(t.size, int)
g4 = est._grid(t, e, None, floor_early=True, n_rule=True)
on = est._grid(t, e, None, floor_early=True, n_rule=True, early_all=True)
print(json.dumps(dict(same_kw=repr(on[1]) == repr(g4[1]), rec=on[3].get("early_all"),
                      rest=repr({k: v for k, v in on[3].items() if k != "early_all"}) == repr(g4[3]))))
""")
    assert out["_rc"] == 0, out["_stderr"]
    assert out["same_kw"] and out["rest"]
    assert out["rec"]["applied"] is False and "exact-time" in out["rec"]["scope"]


def test_fit_marches_the_mesh_and_records_it():
    """estimator.fit(early_all=True) (MLE stage, small synthetic interval cohort, not floor-bound): the solver kwargs
    carry the mesh and dt_floor.early_all.nsteps is the graded schedule's size; off records nothing new and has no
    mesh; exact-time data with exact_from_F run (no refusal) on the uniform grid."""
    out = T.run_py("""
import sys, json, os, importlib.util
import numpy as np
from srbench import npcompat
def load(k, p):
    s = importlib.util.spec_from_file_location(k, p); m = importlib.util.module_from_spec(s)
    sys.modules[k] = m; s.loader.exec_module(m); return m
P = os.path.join("methods", "mle_profile_jax2")
est = load("ea.est3", os.path.join(P, "estimator.py")); fj = load("ea.fj3", os.path.join(P, "fp_jax.py"))
rng = np.random.default_rng(11)
t = np.round(rng.gamma(12.0, 1.0 / 12.0, 300) * 100.0); t = np.maximum(t, 1.0); e = np.ones(t.size, int)
on = est.fit(fj, t, e, delta=1.0, stages=("mle",), floor_early=True, n_rule=True, early_all=True)
off = est.fit(fj, t, e, delta=1.0, stages=("mle",), floor_early=True, n_rule=True)
s_on, s_off = on["diagnostics"]["solver"], off["diagnostics"]["solver"]
ns = int(fj.early_schedule(s_on["dt"], s_on["tau_max"], 20, 1e-8, 1.05)[0].size)
ex = est.fit(fj, t + 0.37, e, delta=None, stages=("mle",), exact_from_F=True, early_all=True)
print(json.dumps(dict(binds=on["diagnostics"]["dt_floor"]["binds"], early_on=s_on.get("early"),
                      early_off=s_off.get("early"), rec_on=on["diagnostics"]["dt_floor"].get("early_all"),
                      rec_off="early_all" in off["diagnostics"]["dt_floor"], ns=ns,
                      uniform=int(s_on["tau_max"] / s_on["dt"]),
                      rest_same={k: v for k, v in s_on.items() if k != "early"} == s_off,
                      ll_on=on["loglik"], ll_off=off["loglik"],
                      ex_early=ex["diagnostics"]["solver"].get("early"), ex_rec=ex["diagnostics"]["dt_floor"].get("early_all"),
                      ex_ll=ex["loglik"])))
""", timeout=1800)
    assert out["_rc"] == 0, out["_stderr"]
    assert out["binds"] is False and out["early_off"] is None and out["rec_off"] is False
    assert out["early_on"] == MESH and out["rest_same"]
    assert out["rec_on"]["applied"] is True and out["rec_on"]["nsteps"] == out["ns"] > out["uniform"]
    assert abs(out["ll_on"] - out["ll_off"]) < 5.0                     # the same model, a finer start
    assert out["ex_early"] is None and out["ex_rec"]["applied"] is False and "nsteps" not in out["ex_rec"]
    assert out["ex_ll"] == out["ex_ll"]                                # finite (not NaN)


def test_adapter_early_all_defaults_and_switch_record():
    """Both names: EARLY_ALL on (dev5, dev6); run_fit's default off (dev4's call); calibrate passes the class constant;
    diagnostics.switches records it; the versions are VERSIONS (1.0.0 = 1.0.0.dev6 frozen for Phase C, rc2 F1; the
    successor 1.0.1.devN keeps EARLY_ALL as it is: test_jax2_r101.py, test_jax2_kfix.py)."""
    out = T.run_py("""
import json, os, importlib.util, inspect, sys
s = importlib.util.spec_from_file_location("jax2_tests.ea.adapter", os.path.join("methods", "mle_profile_jax2", "__init__.py"))
ad = importlib.util.module_from_spec(s); sys.modules[s.name] = ad; s.loader.exec_module(ad)
sig = inspect.signature(ad.run_fit)
src_run = inspect.getsource(ad.run_fit); src_cal = inspect.getsource(ad._MleProfileJax2Base.calibrate)
src_cs = inspect.getsource(ad._MleProfileJax2Base.claimset)
print(json.dumps(dict(
    on=[ad.MleProfileJax2.EARLY_ALL, ad.MleProfileJax2H.EARLY_ALL],
    others=[[c.FLOOR_EARLY, c.N_RULE, c.DT_RULE, c.SELF_CONSISTENT, c.SHADOW, c.EXACT_F, c.RELABEL]
            for c in (ad.MleProfileJax2, ad.MleProfileJax2H)],
    default=sig.parameters["early_all"].default,
    run_passes_only_when_on='**({"early_all": True} if early_all else {})' in src_run,
    calibrate_passes="early_all=self.EARLY_ALL" in src_cal,
    switches_record="early_all=bool(self.EARLY_ALL)" in src_cs,
    versions=ad.VERSIONS)))
""")
    assert out["_rc"] == 0, out["_stderr"]
    assert out["on"] == [True, True] and out["default"] is False
    assert out["others"] == [[True, True, True, True, True, True, True]] * 2       # dev4's, DT_RULE on (1.0.1.dev4)
    assert out["run_passes_only_when_on"] and out["calibrate_passes"] and out["switches_record"]
    from version_lock import versions
    assert out["versions"] == versions()                                             # 1.0.1.dev2: all eight names


@pytest.fixture(scope="module")
def fj():
    T.jax_env()
    return T.pkg_module("fp_jax")


def test_nonfloor_first_bin_deaths_are_resolved(fj):
    """The defect on a production (non-floor) grid: pure-k0.1-p22-c1-s1 (n 10,000, delta 1, L 99.61; dev4 grid N 900,
    dt 3.43e-3, uniform BE start) -- its 51 deaths at ages 2, 3, 4 (tau < 0.05) at the Phase A MLE and its worst edge
    node (eta:down). The march is causal, so a short tau_max gives these terms exactly. Reference: dt 1e-4 with a
    finer graded start (1e-11, x1.002); it moves by 4.4e-5 at dt 5e-5 (1e-12, x1.001). MEASURED (D1, E1): the
    uniform start puts these terms' dll 3.89 nats off; dev5's graded start (x1.02) 0.0083; dev6's (x1.05) 0.048."""
    import numpy as np
    L, delta, dt = 99.60990000000389, 1.0, 0.0034250276997231533
    t = np.array([2.0] * 10 + [3.0] * 14 + [4.0] * 27)
    e = np.ones(t.size, int)
    z_hat = np.array([-7.84084091736265, -2.352582823558673, -3.917869129813142, -6.8925755026774755])
    z_node = np.array([-7.99434659022892, -2.306162806267503, -4.086731101964123, -2.001361649953153])
    common = dict(N=900, tau_max=0.3, Gtol=1e-11, sampling_dt=delta, exact_from_F=True)

    def dll(**kw):
        f, _vg = fj.make_loglik(t, e, L, **dict(common, **kw))
        return float(f(z_node)) - float(f(z_hat))

    ref = dll(dt=1e-4, early=dict(dt0=1e-11, ratio=1.002))
    uniform = dll(dt=dt)
    graded = dll(dt=dt, early=MESH)
    graded5 = dll(dt=dt, early=MESH_DEV5)
    assert abs(uniform - ref) > 1.0, (uniform, ref)                  # the uniform BE start: 3.89 nats off
    assert abs(graded - ref) < 0.1, (graded, ref)                    # dev6's x1.05: resolved to 0.048
    assert abs(graded5 - ref) < 0.02, (graded5, ref)                 # dev5's x1.02: 0.0083
