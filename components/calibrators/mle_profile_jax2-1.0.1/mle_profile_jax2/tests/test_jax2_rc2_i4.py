"""rc2 I4 (1.0.0.dev4):
  (b) the VERSION LOCK on a FULLY CACHED run -- `descriptor` checks the lock, so `srbench.run.execute.plan_run` /
      `run` REFUSE before building a single store key, while discovery and listing keep working (the version reads
      `<version>+UNLOCKED.<digest>` outside a run). End to end through `execute.run` with a temporary store;
  (c) the EXACT-TIME density from the absorbed mass F (`fp_jax._density_F`, `make_loglik(exact_from_F=True)`):
      equal to rc1's S-difference density where that one is resolved, smooth where it is round-off; interval data
      keep rc1's program byte for byte (and exact_from_F=False keeps rc1's exact-time program:
      test_off_path_is_rc1_bit_for_bit);
  (a) the shadow chain's helpers (`estimator._shadow_for`, `_better`).
Anything that registers the adapter or needs SRK runs in a SUBPROCESS (see _jax2_testlib)."""
from __future__ import annotations

import math
import os
import shutil

import pytest

import _jax2_testlib as T
import version_lock as VL

pytestmark = pytest.mark.sut

KW = dict(N=200, dt=4e-3, tau_max=6.0, Gtol=1e-11)


# ---------------------------------------------------------------- (b) the lock on a fully cached run

_E2E_HEAD = """
import sys, json, os
from srbench.contract import method as M
from srbench.library.load import Library
from srbench.run import execute as X
from srbench.store.store import CalibrationRecord, ResultStore
from srbench.store.key import make_key
from srbench.contract.types import ClaimSet, CONTRACT_VERSION
tmp = {tmp!r}
store = ResultStore(os.path.join(tmp, "store"))
M.discover(os.path.join(tmp, "methods"))
lib = Library(version="v2")
CID = "pure-k0.01-p02-c1-s1"
NAMES = ("mle_profile_jax2", "mle_profile_jax2_h")
def nrec():
    return sum(sum(1 for f in fs if f.endswith(".json")) for _d, _s, fs in os.walk(store.root))
def attempt(f):
    try:
        return ["ran", f()]
    except SystemExit as ex:
        return [type(ex).__name__, str(ex)[:600]]
res = dict()
"""


def _tmp_tree(tmp_path):
    d = tmp_path / "methods" / "mle_profile_jax2"
    d.mkdir(parents=True)
    for f in VL.runtime_files() + ["VERSION_LOCK.json"]:
        shutil.copy2(os.path.join(VL.PKG, f), d / f)
    (tmp_path / "store").mkdir()
    return d


def test_a_fully_cached_run_refuses_after_a_code_change_under_the_same_version(tmp_path):
    """B5 re-review gap: `calibrate`'s guard never runs when every key is stored, so a code change under an
    unchanged version was served from the store silently. Now `descriptor` refuses inside a run. Steps: store a
    record under each name's key (synthetic: the gap is about keys, not results) -> a run reuses it; edit
    estimator.py in the copy (version unchanged) -> plan_run and run REFUSE for both names, nothing is stored,
    and outside a run the descriptor still answers, with an UNLOCKED version."""
    pkg = _tmp_tree(tmp_path)
    before = T.run_py(_E2E_HEAD.format(tmp=str(tmp_path)) + """
obs = lib.observation(CID)
for n in NAMES:
    d = M.get(n).descriptor
    key = make_key(method_name=d.name, method_version=d.version, cohort_id=CID, library_version=lib.version,
                   contract_version=CONTRACT_VERSION, observation=obs, settings=None)
    store.put(CalibrationRecord(key=key, claim_set=ClaimSet(cohort_id=CID, method_name=n, status="ok"), status="ok"))
    res[n] = dict(version=d.version, run=attempt(lambda: {k: v for k, v in X.run(M.get(n), lib, [CID], store).items()
                                                         if k in ("n_reused", "n_computed", "method_version")}))
res["records"] = nrec()
print(json.dumps(res))
""")
    assert before["_rc"] == 0, before["_stderr"]
    assert before["records"] == 2
    for n in ("mle_profile_jax2", "mle_profile_jax2_h"):
        assert before[n]["run"][0] == "ran" and before[n]["run"][1]["n_reused"] == 1, before[n]
        assert before[n]["run"][1]["n_computed"] == 0 and "+UNLOCKED" not in before[n]["version"]
    with open(pkg / "estimator.py", "a") as fh:
        fh.write("\n# a behaviour change nobody versioned\n")
    after = T.run_py(_E2E_HEAD.format(tmp=str(tmp_path)) + """
for n in NAMES:
    m = M.get(n)
    res[n] = dict(listed=m.descriptor.version,
                  plan=attempt(lambda: X.plan_run(m, lib, [CID], store).n_reuse),
                  run=attempt(lambda: X.run(m, lib, [CID], store)["n_reused"]))
res["records"] = nrec()
res["siblings"] = sorted(k for k in sys.modules if k.startswith("srbench_methods.mle_profile_jax2."))
print(json.dumps(res))
""")
    assert after["_rc"] == 0, after["_stderr"]
    for n in ("mle_profile_jax2", "mle_profile_jax2_h"):
        v = VL.versions()[n]
        assert after[n]["listed"].startswith(v + "+UNLOCKED.") and len(after[n]["listed"]) == len(v) + 22, after[n]
        for step in ("plan", "run"):
            assert after[n][step][0] == "RefusedEnvironment", (n, step, after[n][step])
            assert "CODE CHANGED BUT VERSION DID NOT" in after[n][step][1]
    assert after["records"] == 2 and after["siblings"] == []


def test_the_untouched_package_reports_its_registered_versions():
    out = T.run_py("""
import json
from srbench.contract import method as M
reg = M.registered()
print(json.dumps({n: reg[n].descriptor.version for n in __NAMES__}))
""".replace("__NAMES__", repr(sorted(VL.versions()))))
    assert out["_rc"] == 0, out["_stderr"]
    assert {n: out[n] for n in VL.versions()} == VL.versions()          # 1.0.1.dev2: every registered name


# ---------------------------------------------------------------- (c) the exact-time density from F

@pytest.fixture(scope="module")
def mods():
    T.jax_env()
    return T.pkg_module("fp_jax"), T.rc1_module("fp_jax")


def _sharp_cohort(n=400, seed=3):
    """Ages around 100 with CV ~0.1 and a few early deaths, recorded exactly (no rounding)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    t = rng.normal(100.0, 10.0, n)
    t[:5] = [35.0, 40.0, 45.0, 52.0, 58.0]
    t = np.maximum(t, 1.0)
    e = np.ones(t.size, int)
    e[t >= 125] = 0
    t[t >= 125] = 125.0
    return t, e, float(np.mean(t))


def test_exact_density_equals_rc1_where_s_is_resolved(mods):
    """Hazard-free exact times: where S's central difference is resolved (S_i < 1 - 1e-3) the F density gives
    rc1's term to 1e-10 nats; censored terms are rc1's exactly (the density only enters deaths). The reference is
    this package with exact_from_F=False, whose exact-time program IS rc1's (test_off_path_is_rc1_bit_for_bit,
    sd=None); the rc1 fixture itself has no `with_terms`."""
    import numpy as np
    fj, _rc = mods
    t, e, L = _sharp_cohort()
    a = fj.make_loglik(t, e, L, sampling_dt=None, with_terms=True, exact_from_F=True, **KW)[2]
    b = fj.make_loglik(t, e, L, sampling_dt=None, with_terms=True, **KW)[2]
    died = e == 1
    for zs in ([2.5e-4, 0.02, 1.5e-4, 0.3], [4e-4, 0.05, 3e-4, 1.0]):
        z4 = np.log(zs)
        ta, tb = a(z4), b(z4)
        res = died & (np.asarray(tb["S_i"]) < 1 - 1e-3)
        assert res.sum() > 100
        assert np.max(np.abs(np.asarray(ta["term"])[res] - np.asarray(tb["term"])[res])) <= 1e-10
        assert np.array_equal(np.asarray(ta["term"])[~died], np.asarray(tb["term"])[~died])


def test_density_F_keeps_rc1s_p_wherever_it_uses_S(mods):
    """`_density_F` returns `_truncate`'s p BIT FOR BIT at every grid node where S's difference is kept, and F's
    difference only where F's larger operand is the smaller one."""
    import numpy as np
    import jax.numpy as jnp
    fj, _ = mods
    N, dt, tau_max = 200, 4e-3, 6.0
    nsteps = int(tau_max / dt)
    march = fj._make_march(N, nsteps, dt, 20, 1e-11, "value")
    L = 100.0
    for zs in ([2.5e-4, 0.02, 1.5e-4, 0.3], [1e-3, 0.5, 2e-4, 0.01]):
        eta, beta, eps, kap = zs
        h = 1.0 / N
        yf = (np.arange(N) + 0.5) * h
        S_raw, F_raw = march((L ** 2 * eta, L * eps, jnp.asarray(L * beta * yf / (yf + kap))))
        tau = jnp.arange(nsteps + 1) * dt
        S, p, k = fj._truncate(S_raw, tau, dt, 1e-11, nsteps)
        pd, use_F = fj._density_F(F_raw, S, p, k, dt, nsteps)
        pd, use_F, p = np.asarray(pd), np.asarray(use_F), np.asarray(p)
        assert use_F.any() and (~use_F).any()
        assert np.array_equal(pd[~use_F], p[~use_F])
        idx = np.arange(nsteps + 1)
        assert not use_F[idx > int(k)].any()
        Sn = np.asarray(S)
        both = use_F & (Sn < 1 - 1e-3) & (p > 1e-6)          # F used AND S resolved: the same number
        if both.any():
            assert np.max(np.abs(pd[both] - p[both]) / p[both]) <= 1e-9


def test_interval_programs_are_still_rc1s_with_exact_from_F_on(mods):
    """exact_from_F only touches exact-time deaths: with it ON, interval data still compile to rc1's program text
    and give rc1's values and gradients bit for bit."""
    import numpy as np
    fj, rc = mods
    t, e, L = _sharp_cohort()
    t = np.floor(t)
    a = rc.make_loglik(t, e, L, sampling_dt=1.0, **KW)
    b = fj.make_loglik(t, e, L, sampling_dt=1.0, exact_from_F=True, **KW)
    z0 = np.log([2.5e-4, 0.02, 1.5e-4, 0.3])
    assert a[1].lower(z0).as_text(debug_info=False) == b[1].lower(z0).as_text(debug_info=False)
    va, ga = a[1](z0)
    vb, gb = b[1](z0)
    assert float(va) == float(vb) and np.array_equal(np.asarray(ga), np.asarray(gb))


def test_exact_time_value_is_smooth_on_a_sharp_library_cohort():
    """hazard-k0.01-p00-c0-s0 (n 1000) with EXACT ages on the method's grid, at the planted SR point: rc1's
    hazard-free exact-time value jumps by thousands of nats under 1-3 ulp changes of z (early deaths' S-difference
    densities are round-off: 0 -> the -700 floor), the F density's by < 1e-6 nats; forward- and reverse-mode AD
    agree to 1e-7 on both paths (rc2 I4 measured 1e4-1e5 nats vs <= 7.4e-10 over L2's 60 points)."""
    out = T.run_py("""
import json, math, os, sys
import numpy as np
from srbench import npcompat
import importlib.util
def load(k, p):
    s = importlib.util.spec_from_file_location(k, p); m = importlib.util.module_from_spec(s); sys.modules[k] = m
    s.loader.exec_module(m); return m
P = os.path.join("methods", "mle_profile_jax2")
fj = load("x.fj", os.path.join(P, "fp_jax.py"))
om = load("x.om", os.path.join(P, "observation_model.py"))
est = load("x.est", os.path.join(P, "estimator.py"))
import jax, jax.numpy as jnp
from srbench.library.load import Library
lib = Library(version="v2"); cid = "hazard-k0.01-p00-c0-s0"
obs = lib.observation(cid); sp = next(s for s in lib.specs if s.cohort_id == cid)
p = om.prepare(obs.ages, obs.events, obs.delta, obs.window, window_on=True, relabel_on=True)
t, e = np.asarray(p.ages, float), np.asarray(p.events, int)
L, kw, _c, _f = est._grid(t, e, None)
g = dict(N=kw["N"], dt=kw["dt"], tau_max=kw["tau_max"], Gtol=kw["Gtol"], sampling_dt=None)
Lp = float(sp.L_phys)
z4 = np.log([sp.rho_eta / Lp ** 2, sp.rho_beta / Lp, sp.rho_eps / Lp, float(sp.kappa)])
res = {}
for lab, m, hz in (("new4", fj, False), ("rc1", fj, False), ("new5", fj, True)):
    z = np.append(z4, math.log(0.4 / L)) if hz else z4
    kwz = dict(hazard=True) if hz else {}
    if lab != "rc1":                       # "rc1" = exact_from_F off: rc1's exact-time program
        kwz["exact_from_F"] = True
    f, vg, terms = m.make_loglik(t, e, L, with_terms=True, **kwz, **g)
    tot = jax.jit(lambda zz: jnp.sum(terms(zz)["term"]) + terms(zz)["cond"])
    vals = [float(f(z + np.eye(z.size)[k] * j * np.spacing(abs(z[k])))) for k in range(z.size) for j in range(-3, 4)]
    v, gr = vg(z)
    gf = np.array([float(jax.jvp(tot, (jnp.asarray(z),), (jnp.asarray(np.eye(z.size)[k]),))[1]) for k in range(z.size)])
    res[lab] = dict(spread=max(vals) - min(vals), v=float(v),
                    fwd=float(np.max(np.abs(np.asarray(gr) - gf) / np.maximum(np.abs(gf), 1.0))))
print(json.dumps(res))
""", timeout=1200)
    assert out["_rc"] == 0, out["_stderr"]
    assert out["rc1"]["spread"] > 1.0, out["rc1"]            # positive control: rc1's exact path is round-off
    assert out["new4"]["spread"] <= 1e-6 and out["new5"]["spread"] <= 1e-6, out
    assert out["new4"]["fwd"] <= 1e-7 and out["new5"]["fwd"] <= 1e-7, out
    assert out["new4"]["v"] > out["rc1"]["v"]                # the round-off deaths no longer sit on the floor


# ---------------------------------------------------------------- (a) the shadow chain's helpers

@pytest.fixture(scope="module")
def est():
    T.jax_env()
    return T.pkg_module("estimator")


def _obj(est, theta5, z_shadow0=None, ll_shadow0=None, hazard=True):
    from types import SimpleNamespace
    box = dict(est.box_for(300.0), h=(1e-8, 0.1))
    return SimpleNamespace(hazard=hazard, params=est.PARAMS_SR + (("h",) if hazard else ()), box=box,
                           z_shadow0=z_shadow0, ll_shadow0=ll_shadow0)


def test_shadow_for_is_off_without_the_hazard_and_for_kappa_and_h(est):
    import numpy as np
    th = [2e-3, 60.0, 6e-4, 300.0, 1e-8]
    o = _obj(est, th, z_shadow0=np.log([2.5e-3, 0.2, 1.8e-3, 0.017, 1e-8]), ll_shadow0=-10.0)
    assert est._shadow_for(o, 3, [0, 1, 2, 4], -9.0, th) is None          # kappa
    assert est._shadow_for(o, 4, [0, 1, 2, 3], -9.0, th) is None          # h
    o4 = _obj(est, th[:4], hazard=False)
    assert est._shadow_for(o4, 0, [1, 2, 3], -9.0, th[:4]) is None


def test_shadow_origin_is_the_jax2_like_mle_in_the_other_basin_else_the_reflected_mle(est):
    import numpy as np
    th = [2e-3, 60.0, 6e-4, 300.0, 1e-8]                                 # MLE in the HIGH kappa basin
    zj = np.log([2.5e-3, 0.2, 1.8e-3, 0.017, 1e-8])                     # jax2-like MLE in the LOW one
    o = _obj(est, th, z_shadow0=zj, ll_shadow0=-10.0)
    s = est._shadow_for(o, 0, [1, 2, 3, 4], -9.0, th)
    assert s["origin"] == "jax2like" and np.allclose(s["z"][:3], zj[[1, 2, 3]])
    assert s["z"][s["hpos"]] == math.log(1e-8) and s["dev_prev"] == pytest.approx(2.0)
    o2 = _obj(est, th, z_shadow0=np.log([2.5e-3, 70.0, 6e-4, 250.0, 1e-8]), ll_shadow0=-9.5)
    s2 = est._shadow_for(o2, 0, [1, 2, 3, 4], -9.0, th)                  # jax2-like in the SAME basin
    assert s2["origin"] == "reflected"
    beta, kap = math.exp(s2["z"][0]), math.exp(s2["z"][2])
    assert kap == pytest.approx(1e-3) and beta == pytest.approx(60.0 / 301.0 * (1.0 + 1e-3), rel=1e-12)
    th_low = [2e-3, 0.2, 6e-4, 0.003, 1e-8]                              # MLE low -> reflected to the ceiling
    s3 = est._shadow_for(_obj(est, th_low), 2, [0, 1, 3, 4], -9.0, th_low)
    assert s3["origin"] == "reflected" and math.exp(s3["z"][2]) == pytest.approx(300.0)
    assert math.exp(s3["z"][1]) == pytest.approx(0.2 / 1.003 * 301.0, rel=1e-12)


def test_better_prefers_the_walks_own_solve_on_ties_and_failures(est):
    import numpy as np
    a, b = (-5.0, np.zeros(3)), (-5.0, np.ones(3))
    assert est._better(a, b) is a
    assert est._better(a, (-4.0, np.ones(3))) is not a
    assert est._better(a, (float("nan"), np.ones(3))) is a and est._better(a, (-1.0, None)) is a
    assert est._better((float("-inf"), None), b) is b
